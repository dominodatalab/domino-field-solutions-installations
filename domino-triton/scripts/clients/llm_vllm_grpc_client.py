#!/usr/bin/env python3
"""
LLM Text Generation Client - vLLM backend (gRPC)

Targets the Triton vLLM backend (tinyllama-vllm, qwen25-vllm, or any model
using the native vllm Triton backend).

Tensor schema (differs from the Python backend):
  Input:  text_input (STRING), stream (BOOL), sampling_parameters (STRING JSON)
  Output: text_output (STRING)

Chat template formatting:
  The Triton vLLM backend accepts a raw string — it does NOT apply a chat
  template automatically. Use --apply-chat-template to format the prompt
  using the model's tokenizer template before sending. This is required for
  instruct models (Qwen2.5-Instruct, TinyLlama-Chat, etc.).

  Alternatively, pass a pre-formatted prompt directly.

Structured output (guided decoding):
  Passed at request time via --guided-json, --guided-regex, --guided-choice.
  The engine-level backend (xgrammar / outlines / lm-format-enforcer) is
  set in model.json.

Usage:
    # Basic
    python llm_vllm_grpc_client.py --prompt "What is the capital of France?"

    # With chat template (recommended for instruct models)
    python llm_vllm_grpc_client.py \\
        --model qwen25-vllm \\
        --prompt "What is the capital of France?" \\
        --apply-chat-template

    # Streaming
    python llm_vllm_grpc_client.py --prompt "Explain AI" --stream

    # Guided JSON output
    python llm_vllm_grpc_client.py \\
        --model qwen25-vllm \\
        --prompt "Extract: name and age from 'Alice is 30 years old'" \\
        --apply-chat-template \\
        --guided-json '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}'

    # Guided regex
    python llm_vllm_grpc_client.py \\
        --model qwen25-vllm \\
        --prompt "What is the US phone number format?" \\
        --guided-regex "\\\\(\\\\d{3}\\\\) \\\\d{3}-\\\\d{4}"

    # Guided choice
    python llm_vllm_grpc_client.py \\
        --model qwen25-vllm \\
        --prompt "Classify sentiment: 'I love this product'" \\
        --guided-choice '["positive","negative","neutral"]'
"""

import argparse
import json
import logging
import os
import queue
import time
from typing import Optional

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from auth_helper import get_auth_headers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

# Built-in templates for common models — used when the tokenizer is not
# locally available (e.g., no HuggingFace cache). When the tokenizer IS
# available, --apply-chat-template will use it directly.
_BUILTIN_TEMPLATES = {
    "qwen": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "tinyllama": (
        "<|system|>\n{system}</s>\n"
        "<|user|>\n{user}</s>\n"
        "<|assistant|>\n"
    ),
    "chatml": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}

# Map model name substrings to template keys
_MODEL_TO_TEMPLATE = {
    "qwen": "qwen",
    "tinyllama": "tinyllama",
}


def _detect_template_key(model_name: str) -> Optional[str]:
    model_lower = model_name.lower()
    for key in _MODEL_TO_TEMPLATE:
        if key in model_lower:
            return _MODEL_TO_TEMPLATE[key]
    return None


def apply_chat_template(
    model_name: str,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    tokenizer_name_or_path: Optional[str] = None,
) -> str:
    """
    Apply the model's chat template to a single-turn user message.

    Tries (in order):
    1. Load the tokenizer and use apply_chat_template() from HuggingFace
    2. Fall back to built-in template based on model name
    3. Return the raw prompt unchanged with a warning
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Try HuggingFace tokenizer first
    model_id = tokenizer_name_or_path or _hf_model_id_from_name(model_name)
    if model_id:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.debug(f"HuggingFace tokenizer unavailable ({e}), falling back to built-in template")

    # Built-in template fallback
    key = _detect_template_key(model_name)
    if key and key in _BUILTIN_TEMPLATES:
        logger.debug(f"Using built-in '{key}' chat template for model '{model_name}'")
        return _BUILTIN_TEMPLATES[key].format(system=system_prompt, user=user_prompt)

    logger.warning(
        f"No chat template found for '{model_name}'. "
        "Sending raw prompt. Use --tokenizer to specify a HuggingFace model ID."
    )
    return user_prompt


# Mapping from Triton model name → HuggingFace model ID for tokenizer loading
_TRITON_TO_HF = {
    "qwen25-vllm": "Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama-vllm": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def _hf_model_id_from_name(triton_model_name: str) -> Optional[str]:
    return _TRITON_TO_HF.get(triton_model_name)


# ---------------------------------------------------------------------------
# Triton inference
# ---------------------------------------------------------------------------

def _build_inputs(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    exclude_input_in_output: bool = True,
    guided_json: Optional[str] = None,
    guided_regex: Optional[str] = None,
    guided_choice: Optional[str] = None,
) -> list:
    """Build Triton input tensors for the vLLM backend."""
    text_input = grpcclient.InferInput("text_input", [1], "BYTES")
    text_input.set_data_from_numpy(np.array([prompt], dtype=np.object_))

    stream_input = grpcclient.InferInput("stream", [1], "BOOL")
    stream_input.set_data_from_numpy(np.array([stream], dtype=bool))

    # sampling_parameters is a JSON string passed directly to vLLM SamplingParams.
    # Guided decoding fields: guided_json, guided_regex, guided_choice.
    # Only one of them should be set per request.
    sampling_params: dict = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if guided_json is not None:
        sampling_params["guided_json"] = json.loads(guided_json)
    elif guided_regex is not None:
        sampling_params["guided_regex"] = guided_regex
    elif guided_choice is not None:
        sampling_params["guided_choice"] = json.loads(guided_choice)

    sampling_input = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
    sampling_input.set_data_from_numpy(
        np.array([json.dumps(sampling_params)], dtype=np.object_)
    )

    exclude_input = grpcclient.InferInput("exclude_input_in_output", [1], "BOOL")
    exclude_input.set_data_from_numpy(np.array([exclude_input_in_output], dtype=bool))

    return [text_input, stream_input, sampling_input, exclude_input]


def infer_non_streaming(
    client: grpcclient.InferenceServerClient,
    headers: dict,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    guided_json: Optional[str] = None,
    guided_regex: Optional[str] = None,
    guided_choice: Optional[str] = None,
) -> dict:
    inputs = _build_inputs(
        prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        stream=False,
        guided_json=guided_json, guided_regex=guided_regex, guided_choice=guided_choice,
    )
    outputs = [grpcclient.InferRequestedOutput("text_output")]

    start = time.time()
    response = client.infer(
        model_name=model, inputs=inputs, outputs=outputs, headers=headers,
    )
    elapsed = time.time() - start

    text = response.as_numpy("text_output")[0]
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    return {"prompt": prompt, "generated_text": text, "inference_ms": round(elapsed * 1000, 2)}


def infer_streaming(
    client: grpcclient.InferenceServerClient,
    headers: dict,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    guided_json: Optional[str] = None,
    guided_regex: Optional[str] = None,
    guided_choice: Optional[str] = None,
) -> dict:
    inputs = _build_inputs(
        prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        stream=True,
        guided_json=guided_json, guided_regex=guided_regex, guided_choice=guided_choice,
    )
    outputs = [grpcclient.InferRequestedOutput("text_output")]

    result_queue: queue.Queue = queue.Queue()

    def callback(result, error):
        result_queue.put(error if error else result)

    start = time.time()
    client.start_stream(callback=callback, headers=headers)

    try:
        client.async_stream_infer(model_name=model, inputs=inputs, outputs=outputs)

        full_text = []
        print("Streaming: ", end="", flush=True)

        while True:
            item = result_queue.get(timeout=60)
            if isinstance(item, Exception):
                raise item

            token = item.as_numpy("text_output")[0]
            if isinstance(token, bytes):
                token = token.decode("utf-8")
            print(token, end="", flush=True)
            full_text.append(token)

            if token == "":  # vLLM sends empty string to signal end of stream
                break

        print()
    finally:
        client.stop_stream()

    return {
        "prompt": prompt,
        "generated_text": "".join(full_text),
        "inference_ms": round((time.time() - start) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Text Generation via Triton vLLM backend (gRPC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", "-p", default="What is the capital of France?")
    parser.add_argument(
        "--grpc-url", "-u",
        default=os.environ.get("TRITON_GRPC_URL", "localhost:50051"),
        help="gRPC proxy URL (env: TRITON_GRPC_URL)",
    )
    parser.add_argument("--model", "-m", default="tinyllama-vllm")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--stream", action="store_true",
                        help="Stream tokens as they arrive")

    # Chat template
    chat_group = parser.add_argument_group("Chat template")
    chat_group.add_argument(
        "--apply-chat-template", action="store_true",
        help=(
            "Apply the model's chat template before sending. "
            "Required for instruct models (Qwen2.5-Instruct, TinyLlama-Chat). "
            "Tries HuggingFace tokenizer first, then built-in template."
        ),
    )
    chat_group.add_argument(
        "--tokenizer",
        help="HuggingFace model ID to load the tokenizer from (overrides auto-detection). "
             "Example: Qwen/Qwen2.5-7B-Instruct",
    )

    # Guided decoding (mutually exclusive)
    guided_group = parser.add_argument_group(
        "Guided decoding",
        "Structured output constraints sent per-request via sampling_parameters. "
        "The engine-level backend (xgrammar / outlines / lm-format-enforcer) is set in model.json.",
    )
    guided_exclusive = guided_group.add_mutually_exclusive_group()
    guided_exclusive.add_argument(
        "--guided-json",
        metavar="JSON_SCHEMA",
        help="Constrain output to a JSON Schema (JSON string). "
             "Example: '{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}'",
    )
    guided_exclusive.add_argument(
        "--guided-regex",
        metavar="PATTERN",
        help="Constrain output to a regex pattern.",
    )
    guided_exclusive.add_argument(
        "--guided-choice",
        metavar="JSON_ARRAY",
        help="Constrain output to one of the listed strings. "
             "Example: '[\"positive\",\"negative\",\"neutral\"]'",
    )

    args = parser.parse_args()

    # Optionally apply chat template
    prompt = args.prompt
    if args.apply_chat_template:
        prompt = apply_chat_template(
            model_name=args.model,
            user_prompt=args.prompt,
            system_prompt=args.system_prompt,
            tokenizer_name_or_path=args.tokenizer,
        )
        logger.debug(f"Formatted prompt:\n{prompt}")

    headers = get_auth_headers()
    client = grpcclient.InferenceServerClient(url=args.grpc_url)

    logger.info(f"Model: {args.model} | URL: {args.grpc_url}")
    if args.guided_json:
        logger.info("Guided decoding: JSON schema")
    elif args.guided_regex:
        logger.info(f"Guided decoding: regex  {args.guided_regex!r}")
    elif args.guided_choice:
        logger.info(f"Guided decoding: choice {args.guided_choice}")

    logger.info("-" * 60)

    infer_kwargs = dict(
        client=client, headers=headers, model=args.model, prompt=prompt,
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
        guided_json=args.guided_json,
        guided_regex=args.guided_regex,
        guided_choice=args.guided_choice,
    )

    try:
        if args.stream:
            result = infer_streaming(**infer_kwargs)
        else:
            result = infer_non_streaming(**infer_kwargs)
            logger.info(f"Response: {result['generated_text']}")

        logger.info(f"Inference time: {result['inference_ms']:.0f}ms")

    except InferenceServerException as e:
        logger.error(f"Triton error: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
