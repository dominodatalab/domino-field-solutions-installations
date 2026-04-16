#!/usr/bin/env python3
"""
LLM Text Generation Client (gRPC)

Uses standard tritonclient.grpc library for Triton inference.
Supports SmolLM-135M and other instruction-tuned LLMs.
Supports async mode for concurrent prompt processing.

Usage:
    python llm_text_grpc_client.py --prompt "What is the capital of France?"
    python llm_text_grpc_client.py --prompts-file prompts.txt
    python llm_text_grpc_client.py --prompt "Explain AI" --max-tokens 100 --temperature 0.5
    python llm_text_grpc_client.py --prompts-file prompts.txt --async  # Async mode
    python llm_text_grpc_client.py --prompt "Hello" --model-version 2  # Use specific model version
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from auth_helper import get_auth_headers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directories
SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR.parent.parent / "results" / "llm"


def generate_text_sync(
    client: grpcclient.InferenceServerClient,
    headers: dict,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
    model_version: Optional[str] = None,
) -> dict:
    """Generate text for a single prompt (sync version)."""
    # Build input tensors
    prompt_input = grpcclient.InferInput("prompt", [1], "BYTES")
    prompt_input.set_data_from_numpy(np.array([prompt], dtype=np.object_))

    max_tokens_input = grpcclient.InferInput("max_tokens", [1], "INT32")
    max_tokens_input.set_data_from_numpy(np.array([max_tokens], dtype=np.int32))

    temperature_input = grpcclient.InferInput("temperature", [1], "FP32")
    temperature_input.set_data_from_numpy(np.array([temperature], dtype=np.float32))

    top_p_input = grpcclient.InferInput("top_p", [1], "FP32")
    top_p_input.set_data_from_numpy(np.array([top_p], dtype=np.float32))

    inputs = [prompt_input, max_tokens_input, temperature_input, top_p_input]

    if system_prompt:
        system_input = grpcclient.InferInput("system_prompt", [1], "BYTES")
        system_input.set_data_from_numpy(np.array([system_prompt], dtype=np.object_))
        inputs.append(system_input)

    # Build output requests
    outputs = [
        grpcclient.InferRequestedOutput("generated_text"),
        grpcclient.InferRequestedOutput("token_count"),
    ]

    start = time.time()
    response = client.infer(
        model_name=model,
        inputs=inputs,
        outputs=outputs,
        headers=headers,
        model_version=model_version or "",
    )
    inference_time = time.time() - start

    # Decode response
    generated_text = response.as_numpy("generated_text")[0]
    if isinstance(generated_text, bytes):
        generated_text = generated_text.decode("utf-8")

    token_count = int(response.as_numpy("token_count")[0])

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "token_count": token_count,
        "inference_ms": round(inference_time * 1000, 2),
        "tokens_per_sec": round(token_count / inference_time, 2) if inference_time > 0 else 0,
    }


# ==================== Async Implementation ====================


async def generate_text_async(
    client,
    headers: dict,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
    model_version: Optional[str] = None,
) -> dict:
    """Generate text for a single prompt (async version)."""
    import tritonclient.grpc.aio as grpcclient_aio

    # Build input tensors
    prompt_input = grpcclient_aio.InferInput("prompt", [1], "BYTES")
    prompt_input.set_data_from_numpy(np.array([prompt], dtype=np.object_))

    max_tokens_input = grpcclient_aio.InferInput("max_tokens", [1], "INT32")
    max_tokens_input.set_data_from_numpy(np.array([max_tokens], dtype=np.int32))

    temperature_input = grpcclient_aio.InferInput("temperature", [1], "FP32")
    temperature_input.set_data_from_numpy(np.array([temperature], dtype=np.float32))

    top_p_input = grpcclient_aio.InferInput("top_p", [1], "FP32")
    top_p_input.set_data_from_numpy(np.array([top_p], dtype=np.float32))

    inputs = [prompt_input, max_tokens_input, temperature_input, top_p_input]

    if system_prompt:
        system_input = grpcclient_aio.InferInput("system_prompt", [1], "BYTES")
        system_input.set_data_from_numpy(np.array([system_prompt], dtype=np.object_))
        inputs.append(system_input)

    # Build output requests
    outputs = [
        grpcclient_aio.InferRequestedOutput("generated_text"),
        grpcclient_aio.InferRequestedOutput("token_count"),
    ]

    start = time.time()
    response = await client.infer(
        model_name=model,
        inputs=inputs,
        outputs=outputs,
        headers=headers,
        model_version=model_version or "",
    )
    inference_time = time.time() - start

    # Decode response
    generated_text = response.as_numpy("generated_text")[0]
    if isinstance(generated_text, bytes):
        generated_text = generated_text.decode("utf-8")

    token_count = int(response.as_numpy("token_count")[0])

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "token_count": token_count,
        "inference_ms": round(inference_time * 1000, 2),
        "tokens_per_sec": round(token_count / inference_time, 2) if inference_time > 0 else 0,
    }


async def run_async(args, prompts):
    """Run inference in async mode with concurrent prompt processing."""
    import tritonclient.grpc.aio as grpcclient_aio

    logger.info("Running in ASYNC mode")

    # Create async Triton client
    client = grpcclient_aio.InferenceServerClient(url=args.grpc_url)

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    results = {
        "model": args.model,
        "model_version": args.model_version,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "async_mode": True,
        "generations": [],
        "stats": {},
    }

    logger.info(f"Starting async LLM inference via gRPC proxy: {args.grpc_url}")
    logger.info(f"Model: {args.model}" + (f" (version {args.model_version})" if args.model_version else ""))
    logger.info(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    logger.info("-" * 60)

    # Process all prompts concurrently
    logger.info(f"Processing {len(prompts)} prompts concurrently...")

    async def process_prompt(idx, prompt):
        logger.info(f"[{idx + 1}/{len(prompts)}] Prompt: {prompt[:50]}...")
        try:
            result = await generate_text_async(
                client=client,
                headers=headers,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                system_prompt=args.system_prompt,
                model_version=args.model_version,
            )
            logger.info(f"[{idx + 1}] Response ({result['token_count']} tokens, {result['inference_ms']:.0f}ms)")
            return result
        except Exception as e:
            logger.error(f"[{idx + 1}] Error: {e}")
            return {"prompt": prompt, "error": str(e)}

    tasks = [process_prompt(idx, prompt) for idx, prompt in enumerate(prompts)]
    generation_results = await asyncio.gather(*tasks)

    results["generations"] = generation_results

    await client.close()

    # Stats
    successful = [g for g in results["generations"] if "token_count" in g]
    if successful:
        total_tokens = sum(g["token_count"] for g in successful)
        total_time = sum(g["inference_ms"] for g in successful) / 1000

        results["stats"] = {
            "total_prompts": len(results["generations"]),
            "successful": len(successful),
            "total_tokens": total_tokens,
            "total_time_sec": round(total_time, 2),
            "avg_tokens_per_prompt": round(total_tokens / len(successful), 1),
            "avg_tokens_per_sec": round(total_tokens / total_time, 2) if total_time > 0 else 0,
            "async_mode": True
        }
        logger.info("-" * 60)
        logger.info(
            f"Generated {total_tokens} tokens in {total_time:.2f}s "
            f"({results['stats']['avg_tokens_per_sec']:.1f} tokens/sec) (async)"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


def run_sync(args, prompts):
    """Run inference in sync mode (original implementation)."""
    # Create Triton client
    client = grpcclient.InferenceServerClient(url=args.grpc_url)

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    results = {
        "model": args.model,
        "model_version": args.model_version,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "async_mode": False,
        "generations": [],
        "stats": {},
    }

    logger.info(f"Starting LLM inference via gRPC proxy: {args.grpc_url}")
    logger.info(f"Model: {args.model}" + (f" (version {args.model_version})" if args.model_version else ""))
    logger.info(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    logger.info("-" * 60)

    total_tokens = 0
    total_time = 0

    try:
        for idx, prompt in enumerate(prompts):
            logger.info(f"[{idx + 1}/{len(prompts)}] Prompt: {prompt[:50]}...")

            try:
                result = generate_text_sync(
                    client=client,
                    headers=headers,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    system_prompt=args.system_prompt,
                    model_version=args.model_version,
                )
                results["generations"].append(result)
                total_tokens += result["token_count"]
                total_time += result["inference_ms"] / 1000

                logger.info(f"Response ({result['token_count']} tokens, {result['inference_ms']:.0f}ms):")
                # Print response with word wrap
                response_text = str(result["generated_text"])
                for i in range(0, len(response_text), 80):
                    logger.info(f"  {response_text[i:i+80]}")

            except InferenceServerException as e:
                logger.error(f"Error: {e}")
                results["generations"].append({"prompt": prompt, "error": str(e)})

            logger.info("-" * 60)

    finally:
        client.close()

    # Stats
    if total_time > 0:
        results["stats"] = {
            "total_prompts": len(results["generations"]),
            "total_tokens": total_tokens,
            "total_time_sec": round(total_time, 2),
            "avg_tokens_per_prompt": round(total_tokens / len(results["generations"]), 1),
            "avg_tokens_per_sec": round(total_tokens / total_time, 2),
        }
        logger.info(
            f"Generated {total_tokens} tokens in {total_time:.2f}s "
            f"({results['stats']['avg_tokens_per_sec']:.1f} tokens/sec)"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="LLM Text Generation (gRPC)")
    parser.add_argument("--prompt", "-p", help="Single prompt to process")
    parser.add_argument("--prompts-file", help="File containing prompts (one per line)")
    parser.add_argument(
        "--grpc-url", "-u",
        default=os.environ.get("TRITON_GRPC_URL", "localhost:50051"),
        help="gRPC proxy URL (env: TRITON_GRPC_URL)"
    )
    parser.add_argument(
        "--model", "-m", default="smollm-135m-python", help="Model name"
    )
    parser.add_argument(
        "--model-version", "-v", default=None, help="Model version to use (e.g., '1', '2')"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--system-prompt", "-s", help="System prompt")
    parser.add_argument("--output", "-o", default=str(RESULTS_DIR / "llm_grpc.json"), help=f"Output JSON file (default: {RESULTS_DIR}/llm_grpc.json)")
    parser.add_argument("--async", dest="async_mode", action="store_true", help="Use async mode for concurrent prompt processing")
    args = parser.parse_args()

    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in one sentence.",
            "Write a haiku about programming.",
        ]
        logger.info("Using default sample prompts")

    if args.async_mode:
        asyncio.run(run_async(args, prompts))
    else:
        run_sync(args, prompts)


if __name__ == "__main__":
    main()
