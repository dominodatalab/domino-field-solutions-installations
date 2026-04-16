#!/usr/bin/env python3
"""
benchmark_tinyllama.py

Benchmarks TinyLlama-1.1B on two backends:
  1. tinyllama-python   - HuggingFace Transformers (Python backend)
  2. tinyllama-trtllm   - TensorRT-LLM (optimized)

This is the definitive apples-to-apples comparison:
- Same model (TinyLlama-1.1B-Chat)
- Same prompts
- Same max_tokens
- Different backends

Usage:
    python scripts/benchmark_tinyllama.py
    python scripts/benchmark_tinyllama.py --num-prompts 10 --max-tokens 100
    python scripts/benchmark_tinyllama.py --report results/tinyllama_comparison.md

Requirements:
    - Both models loaded in Triton
    - For TensorRT-LLM: Triton must use the TRT-LLM image

Environment variables:
    TRITON_REST_URL  - REST proxy URL (default: http://localhost:8080)
    DOMINO_USER_API_KEY - API key for authentication
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# Add clients directory to path for auth_helper import
sys.path.insert(0, str(Path(__file__).parent.parent / "clients"))
from auth_helper import get_auth_headers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Script directory
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
RESULTS_BASE_DIR = PROJECT_ROOT / "results"
MODEL_RESULTS_DIR = RESULTS_BASE_DIR / "tinyllama"

# Default configuration
DEFAULT_REST_URL = os.environ.get("TRITON_REST_URL", "http://localhost:8080")
# Note: API key/token auth is now automatic via auth_helper

# Models to benchmark
MODELS = {
    "tinyllama-python": {
        "name": "TinyLlama (Python/HuggingFace)",
        "backend": "python",
        "input_type": "text",  # Accepts text prompts directly
        "description": "HuggingFace Transformers on Python backend"
    },
    "tinyllama-trtllm": {
        "name": "TinyLlama (TensorRT-LLM)",
        "backend": "tensorrtllm",
        "input_type": "tokens",  # Requires pre-tokenized input
        "description": "NVIDIA TensorRT-LLM optimized backend"
    }
}

# Test prompts
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain machine learning in one sentence.",
    "Write a haiku about coding.",
    "What is 2 + 2?",
    "Say hello in Spanish, French, and German.",
]


class TinyLlamaBenchmark:
    """Benchmark runner for TinyLlama models.

    Authentication is handled automatically via auth_helper which prefers:
    1. DOMINO_USER_TOKEN env var
    2. DOMINO_API_PROXY/access-token endpoint (inside Domino)
    3. DOMINO_USER_API_KEY env var
    """

    def __init__(self, rest_url: str):
        self.rest_url = rest_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        # Merge auth headers automatically
        auth_headers = get_auth_headers()
        if auth_headers:
            self.headers.update(auth_headers)

        # Load tokenizer for TensorRT-LLM (requires tokenization)
        self.tokenizer = None

    def _load_tokenizer(self):
        """Load tokenizer for TensorRT-LLM model."""
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                )
                logger.info("Loaded TinyLlama tokenizer for TensorRT-LLM")
            except ImportError:
                logger.error("transformers package required for TensorRT-LLM benchmark")
                logger.error("Install with: pip install transformers")
                raise

    def load_model(self, model_name: str) -> bool:
        """Load a model via repository API."""
        url = f"{self.rest_url}/v2/repository/models/{model_name}/load"
        try:
            resp = requests.post(url, headers=self.headers, timeout=300)
            if resp.status_code == 200:
                logger.info(f"Loaded model: {model_name}")
                return True
            else:
                logger.warning(f"Failed to load {model_name}: {resp.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Error loading {model_name}: {e}")
            return False

    def check_model_ready(self, model_name: str) -> bool:
        """Check if a model is ready for inference."""
        url = f"{self.rest_url}/v2/models/{model_name}/ready"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def warmup_model(self, model_name: str, model_config: dict) -> bool:
        """Warm up a model with a simple inference."""
        logger.info(f"Warming up {model_name}...")

        try:
            if model_config["input_type"] == "text":
                result = self._infer_python(model_name, "Hello", max_tokens=5)
            else:
                self._load_tokenizer()
                result = self._infer_trtllm(model_name, "Hello", max_tokens=5)

            if result["success"]:
                logger.info(f"  Warmup complete ({result['latency_ms']:.0f}ms)")
                return True
            else:
                logger.warning(f"  Warmup failed: {result.get('error', 'Unknown')}")
                return False
        except Exception as e:
            logger.warning(f"  Warmup error: {e}")
            return False

    def _infer_python(self, model_name: str, prompt: str, max_tokens: int = 50,
                      temperature: float = 0.1) -> dict:
        """Run inference on Python backend model (text input)."""
        url = f"{self.rest_url}/v2/models/{model_name}/infer"

        payload = {
            "inputs": [
                {
                    "name": "prompt",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [prompt]
                },
                {
                    "name": "max_tokens",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [max_tokens]
                },
                {
                    "name": "temperature",
                    "shape": [1],
                    "datatype": "FP32",
                    "data": [temperature]
                }
            ]
        }

        start_time = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
            end_time = time.perf_counter()

            if resp.status_code == 200:
                result = resp.json()
                outputs = {o["name"]: o["data"] for o in result.get("outputs", [])}

                generated_text = ""
                if "generated_text" in outputs:
                    data = outputs["generated_text"]
                    if isinstance(data, list) and len(data) > 0:
                        generated_text = data[0]

                token_count = 0
                if "token_count" in outputs:
                    data = outputs["token_count"]
                    if isinstance(data, list) and len(data) > 0:
                        token_count = int(data[0])

                return {
                    "success": True,
                    "generated_text": generated_text,
                    "token_count": token_count,
                    "latency_ms": (end_time - start_time) * 1000
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "latency_ms": (end_time - start_time) * 1000
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency_ms": 0
            }

    def _infer_trtllm(self, model_name: str, prompt: str, max_tokens: int = 50,
                      temperature: float = 0.1) -> dict:
        """Run inference on TensorRT-LLM backend model (token input)."""
        url = f"{self.rest_url}/v2/models/{model_name}/infer"

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt)
        input_length = len(input_ids)

        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [1, input_length],
                    "datatype": "INT32",
                    "data": input_ids
                },
                {
                    "name": "input_lengths",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[input_length]]
                },
                {
                    "name": "request_output_len",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[max_tokens]]
                }
            ]
        }

        start_time = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
            end_time = time.perf_counter()

            if resp.status_code == 200:
                result = resp.json()
                outputs = {o["name"]: o["data"] for o in result.get("outputs", [])}

                # Decode output tokens
                generated_text = ""
                token_count = 0
                if "output_ids" in outputs:
                    output_ids = outputs["output_ids"]
                    # Flatten if nested and skip input tokens
                    if isinstance(output_ids[0], list):
                        output_ids = output_ids[0]
                    # Get only new tokens (after input)
                    new_tokens = output_ids[input_length:]
                    token_count = len(new_tokens)
                    generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                return {
                    "success": True,
                    "generated_text": generated_text,
                    "token_count": token_count,
                    "latency_ms": (end_time - start_time) * 1000
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "latency_ms": (end_time - start_time) * 1000
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency_ms": 0
            }

    def benchmark_model(self, model_name: str, model_config: dict,
                        prompts: list, max_tokens: int) -> dict:
        """Run benchmark on a single model."""
        logger.info(f"\nBenchmarking: {model_config['name']}")
        logger.info(f"  Backend: {model_config['backend']}")
        logger.info(f"  Prompts: {len(prompts)}")
        logger.info(f"  Max tokens: {max_tokens}")

        # Check if model is ready
        if not self.check_model_ready(model_name):
            logger.warning(f"  Model not ready, attempting to load...")
            if not self.load_model(model_name):
                return {
                    "success": False,
                    "error": "Model not available",
                    "model_name": model_name
                }
            # Wait for model to be ready
            for _ in range(30):
                if self.check_model_ready(model_name):
                    break
                time.sleep(1)
            else:
                return {
                    "success": False,
                    "error": "Model failed to load",
                    "model_name": model_name
                }

        # Warmup
        if not self.warmup_model(model_name, model_config):
            logger.warning("  Warmup failed, continuing anyway...")

        # Prepare inference function
        if model_config["input_type"] == "text":
            infer_fn = self._infer_python
        else:
            self._load_tokenizer()
            infer_fn = self._infer_trtllm

        # Run benchmarks
        results = []
        total_tokens = 0
        total_time_ms = 0

        for i, prompt in enumerate(prompts):
            result = infer_fn(model_name, prompt, max_tokens=max_tokens)
            results.append({
                "prompt": prompt,
                "success": result["success"],
                "generated_text": result.get("generated_text", ""),
                "token_count": result.get("token_count", 0),
                "latency_ms": result.get("latency_ms", 0),
                "error": result.get("error")
            })

            if result["success"]:
                total_tokens += result.get("token_count", 0)
                total_time_ms += result.get("latency_ms", 0)
                logger.info(f"  [{i+1}/{len(prompts)}] {result['token_count']} tokens in {result['latency_ms']:.0f}ms")
            else:
                logger.error(f"  [{i+1}/{len(prompts)}] Failed: {result.get('error', 'Unknown')}")

        # Calculate stats
        successful = [r for r in results if r["success"]]
        failed = len(results) - len(successful)

        if successful:
            latencies = [r["latency_ms"] for r in successful]
            tokens = [r["token_count"] for r in successful]

            stats = {
                "total_prompts": len(prompts),
                "successful_prompts": len(successful),
                "failed_prompts": failed,
                "total_tokens": total_tokens,
                "total_time_ms": total_time_ms,
                "total_time_sec": total_time_ms / 1000,
                "avg_latency_ms": np.mean(latencies),
                "p50_latency_ms": np.percentile(latencies, 50),
                "p95_latency_ms": np.percentile(latencies, 95),
                "avg_tokens_per_prompt": np.mean(tokens),
                "tokens_per_sec": (total_tokens / total_time_ms * 1000) if total_time_ms > 0 else 0
            }
        else:
            stats = {
                "total_prompts": len(prompts),
                "successful_prompts": 0,
                "failed_prompts": failed,
                "error": "All inferences failed"
            }

        return {
            "success": len(successful) > 0,
            "model_name": model_name,
            "model_config": model_config,
            "stats": stats,
            "results": results
        }


def format_markdown_report(benchmark_results: dict) -> str:
    """Format benchmark results as a Markdown document."""
    lines = [
        "# TinyLlama Backend Comparison",
        "",
        "Apples-to-apples comparison of TinyLlama-1.1B on different Triton backends.",
        "",
        "## Test Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | TinyLlama-1.1B-Chat |",
        f"| Prompts | {benchmark_results['num_prompts']} |",
        f"| Max Tokens | {benchmark_results['max_tokens']} |",
        f"| Timestamp | {benchmark_results['timestamp']} |",
        "",
        "## Results Summary",
        "",
        "| Backend | Prompts | Tokens | Time (s) | Tokens/sec | Avg Latency | P95 Latency |",
        "|---------|---------|--------|----------|------------|-------------|-------------|",
    ]

    for model_name, data in benchmark_results["models"].items():
        if data["success"]:
            stats = data["stats"]
            lines.append(
                f"| {MODELS[model_name]['name']} | "
                f"{stats['successful_prompts']}/{stats['total_prompts']} | "
                f"{stats['total_tokens']} | "
                f"{stats['total_time_sec']:.2f} | "
                f"**{stats['tokens_per_sec']:.1f}** | "
                f"{stats['avg_latency_ms']:.0f}ms | "
                f"{stats['p95_latency_ms']:.0f}ms |"
            )
        else:
            lines.append(
                f"| {MODELS[model_name]['name']} | FAILED | - | - | - | - | - |"
            )

    # Performance comparison
    python_data = benchmark_results["models"].get("tinyllama-python", {})
    trtllm_data = benchmark_results["models"].get("tinyllama-trtllm", {})

    lines.extend([
        "",
        "## Performance Comparison",
        "",
    ])

    if python_data.get("success") and trtllm_data.get("success"):
        python_tps = python_data["stats"]["tokens_per_sec"]
        trtllm_tps = trtllm_data["stats"]["tokens_per_sec"]
        python_lat = python_data["stats"]["avg_latency_ms"]
        trtllm_lat = trtllm_data["stats"]["avg_latency_ms"]

        speedup = trtllm_tps / python_tps if python_tps > 0 else 0
        latency_improvement = python_lat / trtllm_lat if trtllm_lat > 0 else 0

        lines.extend([
            "| Metric | Python (HuggingFace) | TensorRT-LLM | Improvement |",
            "|--------|---------------------|--------------|-------------|",
            f"| Tokens/sec | {python_tps:.1f} | {trtllm_tps:.1f} | **{speedup:.1f}x faster** |",
            f"| Avg Latency | {python_lat:.0f}ms | {trtllm_lat:.0f}ms | **{latency_improvement:.1f}x lower** |",
            "",
            f"### Key Finding",
            "",
            f"**TensorRT-LLM is {speedup:.1f}x faster** than the Python/HuggingFace backend ",
            f"for TinyLlama-1.1B text generation.",
            "",
        ])
    elif python_data.get("success"):
        lines.extend([
            "**Note:** TensorRT-LLM model not available. ",
            "Build the TensorRT engine in a Domino workspace - see `docs/tensorrt_llm_poc.md`.",
            "",
        ])
    elif trtllm_data.get("success"):
        lines.extend([
            "**Note:** Python backend model not available. ",
            "Download weights with: `python scripts/download_tinyllama.py`",
            "",
        ])

    lines.extend([
        "## Why TensorRT-LLM is Faster",
        "",
        "| Optimization | Python Backend | TensorRT-LLM |",
        "|--------------|----------------|--------------|",
        "| Kernel fusion | None | Fused attention, MLP |",
        "| Memory layout | PyTorch default | Optimized for GPU cache |",
        "| KV cache | Naive | Paged, contiguous |",
        "| Batching | Static | In-flight (continuous) |",
        "| Quantization | FP16 (manual) | FP16 with tensor cores |",
        "",
        "## Sample Outputs",
        "",
    ])

    # Show a few sample outputs
    for model_name, data in benchmark_results["models"].items():
        if data["success"] and data.get("results"):
            lines.append(f"### {MODELS[model_name]['name']}")
            lines.append("")
            for i, result in enumerate(data["results"][:2]):  # First 2 samples
                if result["success"]:
                    lines.extend([
                        f"**Prompt:** {result['prompt']}",
                        "",
                        f"**Response:** {result['generated_text'][:200]}{'...' if len(result['generated_text']) > 200 else ''}",
                        "",
                        f"*({result['token_count']} tokens in {result['latency_ms']:.0f}ms)*",
                        "",
                    ])
            lines.append("")

    lines.extend([
        "---",
        "",
        "*Generated by benchmark_tinyllama.py*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TinyLlama-1.1B: Python vs TensorRT-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark
  python scripts/benchmark_tinyllama.py

  # More prompts, longer outputs
  python scripts/benchmark_tinyllama.py --num-prompts 10 --max-tokens 100

  # Custom report location
  python scripts/benchmark_tinyllama.py --report results/tinyllama_comparison.md

  # Only test Python backend (TensorRT-LLM not available)
  python scripts/benchmark_tinyllama.py --models tinyllama-python
        """
    )

    parser.add_argument(
        "--num-prompts", "-n",
        type=int,
        default=5,
        help="Number of prompts to test (default: 5)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate per prompt (default: 50)"
    )
    parser.add_argument(
        "--report", "-r",
        default=str(MODEL_RESULTS_DIR / "benchmark" / "tinyllama_benchmark.md"),
        help=f"Path to markdown report (default: {MODEL_RESULTS_DIR}/benchmark/tinyllama_benchmark.md)"
    )
    parser.add_argument(
        "--rest-url",
        default=DEFAULT_REST_URL,
        help=f"REST proxy URL (default: {DEFAULT_REST_URL})"
    )
    # Note: --api-key removed. Auth is now automatic via auth_helper which prefers:
    # 1. DOMINO_USER_TOKEN env var
    # 2. DOMINO_API_PROXY/access-token endpoint (inside Domino)
    # 3. DOMINO_USER_API_KEY env var
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to benchmark (default: all)"
    )

    args = parser.parse_args()

    # Setup - determine output directories
    report_path = Path(args.report)
    if not report_path.suffix:
        report_path = report_path.with_suffix(".md")

    # Determine benchmark directory from report path
    if report_path.parent.name == "benchmark":
        benchmark_dir = report_path.parent
    else:
        benchmark_dir = report_path.parent

    # Create benchmark/ and benchmark/json/ subdirectories
    json_dir = benchmark_dir / "json"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Markdown goes directly in benchmark/
    if report_path.parent != benchmark_dir:
        report_path = benchmark_dir / report_path.name

    # Generate prompts
    prompts = (DEFAULT_PROMPTS * ((args.num_prompts // len(DEFAULT_PROMPTS)) + 1))[:args.num_prompts]

    logger.info("=" * 60)
    logger.info("TinyLlama Backend Benchmark")
    logger.info("=" * 60)
    logger.info(f"REST URL: {args.rest_url}")
    logger.info(f"Prompts: {args.num_prompts}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Report: {report_path}")

    # Run benchmarks
    benchmark = TinyLlamaBenchmark(args.rest_url)

    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "rest_url": args.rest_url,
        "models": {}
    }

    for model_name in args.models:
        if model_name in MODELS:
            result = benchmark.benchmark_model(
                model_name=model_name,
                model_config=MODELS[model_name],
                prompts=prompts,
                max_tokens=args.max_tokens
            )
            benchmark_results["models"][model_name] = result

    # Save results to json/
    results_json_path = json_dir / "tinyllama_benchmark_results.json"
    with open(results_json_path, "w") as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_json_path}")

    # Generate report
    markdown = format_markdown_report(benchmark_results)
    with open(report_path, "w") as f:
        f.write(markdown)
    logger.info(f"Report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model_name, data in benchmark_results["models"].items():
        if data["success"]:
            stats = data["stats"]
            print(f"\n{MODELS[model_name]['name']}:")
            print(f"  Tokens/sec: {stats['tokens_per_sec']:.1f}")
            print(f"  Avg latency: {stats['avg_latency_ms']:.0f}ms")
            print(f"  Total tokens: {stats['total_tokens']}")
        else:
            print(f"\n{MODELS[model_name]['name']}: FAILED")
            print(f"  Error: {data.get('stats', {}).get('error', 'Unknown')}")

    # Print speedup if both succeeded
    python_data = benchmark_results["models"].get("tinyllama-python", {})
    trtllm_data = benchmark_results["models"].get("tinyllama-trtllm", {})

    if python_data.get("success") and trtllm_data.get("success"):
        speedup = trtllm_data["stats"]["tokens_per_sec"] / python_data["stats"]["tokens_per_sec"]
        print(f"\n{'=' * 60}")
        print(f"TensorRT-LLM is {speedup:.1f}x FASTER than Python/HuggingFace")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
