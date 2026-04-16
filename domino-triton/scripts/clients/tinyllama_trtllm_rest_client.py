#!/usr/bin/env python3
"""
TinyLlama TensorRT-LLM Backend Client (REST)

Uses standard tritonclient.http library for Triton inference.
For the TensorRT-LLM optimized TinyLlama model.

TensorRT-LLM uses token IDs instead of text strings, so this client
handles tokenization/detokenization using the HuggingFace tokenizer.

Usage:
    python tinyllama_trtllm_rest_client.py --prompt "What is the capital of France?"
    python tinyllama_trtllm_rest_client.py --prompts-file prompts.txt
    python tinyllama_trtllm_rest_client.py --prompt "Explain AI" --max-tokens 100
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from auth_helper import get_auth_headers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directories
SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR.parent.parent / "results" / "tinyllama"

# Tokenizer is loaded on first use
_tokenizer = None


def get_tokenizer():
    """Lazy load the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        logger.info("Loading TinyLlama tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def generate_text(
    client: httpclient.InferenceServerClient,
    headers: dict,
    model: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    use_binary: bool = True,
) -> dict:
    """Generate text for a single prompt using TensorRT-LLM backend."""
    tokenizer = get_tokenizer()

    # Apply chat template for TinyLlama
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    input_ids = tokenizer.encode(formatted)
    input_length = len(input_ids)

    # Build input tensors for TensorRT-LLM
    # Shape: [batch_size, seq_len] for input_ids, [batch_size, 1] for scalars
    inputs = []

    input_ids_tensor = httpclient.InferInput("input_ids", [1, input_length], "INT32")
    input_ids_tensor.set_data_from_numpy(np.array([input_ids], dtype=np.int32), binary_data=use_binary)
    inputs.append(input_ids_tensor)

    input_lengths_tensor = httpclient.InferInput("input_lengths", [1, 1], "INT32")
    input_lengths_tensor.set_data_from_numpy(np.array([[input_length]], dtype=np.int32), binary_data=use_binary)
    inputs.append(input_lengths_tensor)

    request_output_len_tensor = httpclient.InferInput("request_output_len", [1, 1], "INT32")
    request_output_len_tensor.set_data_from_numpy(np.array([[max_tokens]], dtype=np.int32), binary_data=use_binary)
    inputs.append(request_output_len_tensor)

    end_id_tensor = httpclient.InferInput("end_id", [1, 1], "INT32")
    end_id_tensor.set_data_from_numpy(np.array([[tokenizer.eos_token_id]], dtype=np.int32), binary_data=use_binary)
    inputs.append(end_id_tensor)

    pad_id_tensor = httpclient.InferInput("pad_id", [1, 1], "INT32")
    pad_id_tensor.set_data_from_numpy(np.array([[tokenizer.pad_token_id or tokenizer.eos_token_id]], dtype=np.int32), binary_data=use_binary)
    inputs.append(pad_id_tensor)

    # Add optional sampling parameters (shape: [batch_size, 1])
    if temperature > 0:
        temp_tensor = httpclient.InferInput("temperature", [1, 1], "FP32")
        temp_tensor.set_data_from_numpy(np.array([[temperature]], dtype=np.float32), binary_data=use_binary)
        inputs.append(temp_tensor)

        top_k_tensor = httpclient.InferInput("top_k", [1, 1], "INT32")
        top_k_tensor.set_data_from_numpy(np.array([[top_k]], dtype=np.int32), binary_data=use_binary)
        inputs.append(top_k_tensor)

        top_p_tensor = httpclient.InferInput("top_p", [1, 1], "FP32")
        top_p_tensor.set_data_from_numpy(np.array([[top_p]], dtype=np.float32), binary_data=use_binary)
        inputs.append(top_p_tensor)

    # Build output requests
    outputs = [
        httpclient.InferRequestedOutput("output_ids", binary_data=use_binary),
        httpclient.InferRequestedOutput("sequence_length", binary_data=use_binary),
    ]

    start = time.time()
    response = client.infer(
        model_name=model,
        inputs=inputs,
        outputs=outputs,
        headers=headers,
    )
    inference_time = time.time() - start

    # Decode response
    output_ids = response.as_numpy("output_ids")[0]  # [beam, seq_len]
    sequence_length = int(response.as_numpy("sequence_length")[0].item())

    # If output_ids is 2D (beam search), take first beam
    if len(output_ids.shape) > 1:
        output_ids = output_ids[0]

    # Only decode the generated tokens (skip input tokens)
    generated_ids = output_ids[input_length:sequence_length]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    token_count = len(generated_ids)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "token_count": token_count,
        "input_tokens": input_length,
        "inference_ms": round(inference_time * 1000, 2),
        "tokens_per_sec": round(token_count / inference_time, 2) if inference_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="TinyLlama TensorRT-LLM Backend (REST)")
    parser.add_argument("--prompt", "-p", help="Single prompt to process")
    parser.add_argument("--prompts-file", help="File containing prompts (one per line)")
    parser.add_argument(
        "--rest-url", "-u",
        default=os.environ.get("TRITON_REST_URL", "http://localhost:8080"),
        help="REST proxy URL (env: TRITON_REST_URL)"
    )
    parser.add_argument(
        "--model", "-m", default="tinyllama-trtllm", help="Model name"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--output", "-o", default=str(RESULTS_DIR / "tinyllama_trtllm_rest.json"), help=f"Output JSON file (default: {RESULTS_DIR}/tinyllama_trtllm_rest.json)")
    parser.add_argument(
        "--use-json", action="store_true",
        help="Use JSON encoding instead of binary (larger but simpler)"
    )
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

    # Create Triton client (strip http:// prefix for tritonclient)
    url = args.rest_url.replace("http://", "").replace("https://", "").rstrip("/")
    client = httpclient.InferenceServerClient(url=url)

    # Build auth headers
    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    use_binary = not args.use_json
    encoding_mode = "JSON" if args.use_json else "Binary"

    results = {
        "model": args.model,
        "backend": "tensorrt-llm",
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "generations": [],
        "stats": {},
    }

    logger.info(f"Starting TinyLlama TensorRT-LLM inference via REST proxy: {args.rest_url}")
    logger.info(f"Model: {args.model}, Encoding: {encoding_mode}")
    logger.info(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    logger.info("-" * 60)

    total_tokens = 0
    total_time = 0

    for idx, prompt in enumerate(prompts):
        logger.info(f"[{idx + 1}/{len(prompts)}] Prompt: {prompt[:50]}...")

        try:
            result = generate_text(
                client=client,
                headers=headers,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_binary=use_binary,
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


if __name__ == "__main__":
    main()
