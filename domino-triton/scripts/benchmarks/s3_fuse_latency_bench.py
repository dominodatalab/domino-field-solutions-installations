#!/usr/bin/env python3
"""
Dedicated benchmark client for the S3-FUSE vs EFS serving latency test.

Runs inside the Triton pod itself (localhost gRPC, no Domino auth proxy needed —
this is an internal storage-backend comparison, not a test of the proxy layer).
Captures time-to-first-token (TTFT) by timestamping the first item yielded from
Triton's per-token gRPC stream (the vLLM backend always streams at the gRPC level
due to decoupled transaction policy, regardless of the request's "stream" flag).
"""

import argparse
import asyncio
import json
import statistics
import time

import numpy as np
import tritonclient.grpc.aio as grpcclient

PROMPTS = {
    "short": "What is the capital of France?",
    "medium": "Explain how photosynthesis works in about three sentences.",
    "long": (
        "Write a detailed explanation of how neural networks learn through "
        "backpropagation, covering forward pass, loss computation, gradient "
        "computation, and weight updates. Include why the chain rule is central."
    ),
}


def build_inputs(prompt: str, max_tokens: int = 128):
    text_input = grpcclient.InferInput("text_input", [1], "BYTES")
    text_input.set_data_from_numpy(np.array([prompt], dtype=np.object_))

    stream_input = grpcclient.InferInput("stream", [1], "BOOL")
    stream_input.set_data_from_numpy(np.array([True], dtype=bool))

    sampling_params = {"max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.9}
    sampling_input = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
    sampling_input.set_data_from_numpy(
        np.array([json.dumps(sampling_params)], dtype=np.object_)
    )

    exclude_input = grpcclient.InferInput("exclude_input_in_output", [1], "BOOL")
    exclude_input.set_data_from_numpy(np.array([True], dtype=bool))

    return [text_input, stream_input, sampling_input, exclude_input]


async def single_request(client, model: str, prompt: str, max_tokens: int) -> dict:
    inputs = build_inputs(prompt, max_tokens)
    outputs = [grpcclient.InferRequestedOutput("text_output")]

    async def request_iterator():
        yield {"model_name": model, "inputs": inputs, "outputs": outputs}

    start = time.time()
    ttft = None
    token_count = 0
    async for result, error in client.stream_infer(request_iterator()):
        if error:
            raise error
        if ttft is None:
            ttft = time.time() - start
        token = result.as_numpy("text_output")[0]
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        if token == "":
            break
        token_count += 1

    total = time.time() - start
    gen_time = max(total - (ttft or 0), 1e-6)
    # gen_time spans only the time *after* the first token was already received
    # (that production time is already counted inside ttft), so the throughput
    # numerator must be the tokens produced during gen_time -- i.e. excluding
    # the first token itself, not the full token_count.
    steady_state_tokens = max(token_count - 1, 0)
    return {
        "ttft_s": round(ttft or 0, 4),
        "total_s": round(total, 4),
        "token_count": token_count,
        "tokens_per_sec": round(steady_state_tokens / gen_time, 2) if steady_state_tokens else 0,
    }


async def run_battery(model: str, grpc_url: str, n_per_prompt: int, max_tokens: int):
    client = grpcclient.InferenceServerClient(url=grpc_url)
    # Discard one throwaway request first -- the very first inference after a
    # model load can be slower than steady-state (CUDA graph capture, kernel
    # autotuning caches, etc.), which would otherwise bias the "short" prompt's
    # stats specifically, since it's always first in PROMPTS iteration order.
    await single_request(client, model, PROMPTS["short"], max_tokens)
    results = []
    for label, prompt in PROMPTS.items():
        for _ in range(n_per_prompt):
            r = await single_request(client, model, prompt, max_tokens)
            r["prompt_label"] = label
            results.append(r)
    await client.close()
    return results


def summarize(results: list) -> dict:
    ttfts = [r["ttft_s"] for r in results]
    tps = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0]
    return {
        "n": len(results),
        "ttft_s": {
            "mean": round(statistics.mean(ttfts), 4),
            "median": round(statistics.median(ttfts), 4),
            "p95": round(sorted(ttfts)[int(len(ttfts) * 0.95) - 1], 4) if ttfts else None,
            "min": round(min(ttfts), 4),
            "max": round(max(ttfts), 4),
        },
        "tokens_per_sec": {
            "mean": round(statistics.mean(tps), 2) if tps else None,
            "median": round(statistics.median(tps), 2) if tps else None,
            "p95": round(sorted(tps)[int(len(tps) * 0.95) - 1], 2) if tps else None,
        },
        "raw": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TTFT and steady-state token throughput against a Triton vLLM model over local gRPC"
    )
    parser.add_argument(
        "--model",
        default="llama4scout-vllm-fp8-dynamic",
        help="Triton model name to benchmark (default: llama4scout-vllm-fp8-dynamic)",
    )
    parser.add_argument(
        "--grpc-url",
        default="localhost:8001",
        help="Triton gRPC endpoint, reached directly (no Domino auth proxy) since this runs inside the pod (default: localhost:8001)",
    )
    parser.add_argument(
        "--n-per-prompt",
        type=int,
        default=5,
        help="Number of requests to run per prompt label (short/medium/long) (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per request (default: 128)",
    )
    parser.add_argument(
        "--output",
        default="/tmp/bench_result.json",
        help="Path to write the JSON summary to (default: /tmp/bench_result.json)",
    )
    args = parser.parse_args()

    results = asyncio.run(
        run_battery(args.model, args.grpc_url, args.n_per_prompt, args.max_tokens)
    )
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
