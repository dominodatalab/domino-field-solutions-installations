#!/usr/bin/env python3
"""
Benchmark Script for LLM Text Generation Clients

Runs REST and gRPC client implementations with identical parameters
and compares their performance.

Usage:
    python scripts/benchmark_llm_clients.py
    python scripts/benchmark_llm_clients.py --model tinyllama-python
    python scripts/benchmark_llm_clients.py --num-prompts 10
    python scripts/benchmark_llm_clients.py --prompts-file prompts.txt --report results/llm_benchmark.md

Output (in same directory as --report):
    llm_benchmark.md        - Markdown formatted report
    benchmark_results.json  - Detailed results from all clients
    benchmark_summary.txt   - Human-readable comparison table
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Script directory
SCRIPTS_DIR = Path(__file__).parent
CLIENTS_DIR = SCRIPTS_DIR.parent / "clients"
RESULTS_BASE_DIR = SCRIPTS_DIR.parent.parent / "results"
MODEL_RESULTS_DIR = RESULTS_BASE_DIR / "llm"

# Default REST proxy URL
DEFAULT_REST_URL = os.environ.get("TRITON_REST_URL", "http://localhost:8080")

# Default model for benchmarking
DEFAULT_MODEL_NAME = "smollm-135m-python"


def warmup_model(rest_url: str, model_name: str, api_key: str = None) -> bool:
    """
    Warm up a model by checking if it's ready.

    Args:
        rest_url: REST proxy URL (e.g., http://localhost:8080)
        model_name: Name of the model to warm up
        api_key: Optional Domino API key for authentication

    Returns:
        True if model is ready, False otherwise
    """
    url = f"{rest_url.rstrip('/')}/v2/models/{model_name}/ready"
    headers = {}
    if api_key:
        headers["X-Domino-Api-Key"] = api_key

    logger.info(f"Checking if model '{model_name}' is ready...")

    try:
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code == 200:
            logger.info(f"  Model '{model_name}' is ready")
            return True
        else:
            logger.warning(f"  Model not ready: HTTP {resp.status_code}")
            logger.warning(f"  First benchmark inference will include model loading time")
            return False
    except requests.exceptions.ConnectionError:
        logger.warning(f"  Could not connect to REST proxy at {rest_url}")
        logger.warning(f"  First benchmark inference will include model loading time")
        return False
    except Exception as e:
        logger.warning(f"  Error during warmup check: {e}")
        return False

# Client configurations
CLIENTS = [
    {
        "name": "REST",
        "script": "llm_text_rest_client.py",
        "extra_args": [],
        "output": "llm_rest.json",
        "description": "REST API with binary encoding"
    },
    {
        "name": "gRPC",
        "script": "llm_text_grpc_client.py",
        "extra_args": [],
        "output": "llm_grpc.json",
        "description": "gRPC with binary protobuf"
    },
]

# Default sample prompts for benchmarking
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
    "What is 2+2?",
    "Say hello in three languages.",
]


def run_client(
    script_name: str,
    output_path: str,
    prompts: list = None,
    prompts_file: str = None,
    max_tokens: int = 50,
    extra_args: list = None,
    model_name: str = None
) -> dict:
    """
    Run a client script and capture its results.

    Args:
        script_name: Name of the client script
        output_path: Path for output JSON
        prompts: List of prompts to process
        prompts_file: Path to file with prompts
        max_tokens: Max tokens to generate per prompt
        extra_args: Additional command line arguments
        model_name: Model name to use for inference

    Returns:
        Dict with success status, duration, and results
    """
    script_path = CLIENTS_DIR / script_name

    cmd = [
        sys.executable,
        str(script_path),
        "--output", output_path,
        "--max-tokens", str(max_tokens),
        "--temperature", "0.1",  # Low temperature for consistent benchmarking
    ]

    if model_name:
        cmd.extend(["--model", model_name])

    # Add any extra arguments (e.g., --use-json)
    if extra_args:
        cmd.extend(extra_args)

    if prompts_file:
        cmd.extend(["--prompts-file", prompts_file])
    elif prompts:
        # Write prompts to temp file since multiple --prompt args not supported
        temp_file = Path(output_path).parent / "temp_prompts.txt"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, "w") as f:
            f.write("\n".join(prompts))
        cmd.extend(["--prompts-file", str(temp_file)])

    logger.info(f"Running: {script_name}")

    try:
        start_time = datetime.now()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for LLM
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if result.returncode != 0:
            logger.error(f"  Failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
            return {
                "success": False,
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
                "duration": duration
            }

        # Load results from output file
        if Path(output_path).exists():
            with open(output_path) as f:
                results = json.load(f)
            return {
                "success": True,
                "duration": duration,
                "results": results
            }
        else:
            return {
                "success": False,
                "error": "Output file not created",
                "duration": duration
            }

    except subprocess.TimeoutExpired:
        logger.error(f"  Timeout after 10 minutes")
        return {
            "success": False,
            "error": "Timeout after 10 minutes",
            "duration": 600
        }
    except Exception as e:
        logger.error(f"  Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "duration": 0
        }


def format_summary(benchmark_results: dict) -> str:
    """Format benchmark results as a human-readable summary."""
    lines = [
        "=" * 80,
        "LLM BENCHMARK SUMMARY",
        "=" * 80,
        "",
        f"Prompts: {benchmark_results['num_prompts']}",
        f"Max tokens: {benchmark_results['max_tokens']}",
        f"Timestamp: {benchmark_results['timestamp']}",
        "",
        "-" * 80,
        f"{'Client':<20} {'Status':<10} {'Prompts':<8} {'Tokens':<10} {'Total (s)':<12} {'Tok/sec':<12}",
        "-" * 80,
    ]

    for client_name, data in benchmark_results["clients"].items():
        if data["success"]:
            stats = data["results"].get("stats", {})
            lines.append(
                f"{client_name:<20} "
                f"{'OK':<10} "
                f"{stats.get('total_prompts', 'N/A'):<8} "
                f"{stats.get('total_tokens', 'N/A'):<10} "
                f"{stats.get('total_time_sec', 'N/A'):<12} "
                f"{stats.get('avg_tokens_per_sec', 'N/A'):<12}"
            )
        else:
            lines.append(
                f"{client_name:<20} "
                f"{'FAILED':<10} "
                f"{'-':<8} "
                f"{'-':<10} "
                f"{'-':<12} "
                f"{'-':<12}"
            )

    lines.extend([
        "-" * 80,
        "",
        "PERFORMANCE COMPARISON:",
        ""
    ])

    # Calculate relative performance
    rest = benchmark_results["clients"].get("REST", {})
    grpc = benchmark_results["clients"].get("gRPC", {})

    if rest.get("success") and grpc.get("success"):
        rest_tps = rest["results"]["stats"].get("avg_tokens_per_sec", 0)
        grpc_tps = grpc["results"]["stats"].get("avg_tokens_per_sec", 0)
        rest_time = rest["results"]["stats"].get("total_time_sec", 0)
        grpc_time = grpc["results"]["stats"].get("total_time_sec", 0)

        if rest_tps > 0 and grpc_tps > 0:
            if grpc_tps > rest_tps:
                speedup = grpc_tps / rest_tps
                lines.append(f"  gRPC is {speedup:.2f}x faster than REST (tokens/sec)")
            else:
                speedup = rest_tps / grpc_tps
                lines.append(f"  REST is {speedup:.2f}x faster than gRPC (tokens/sec)")

        if rest_time > 0 and grpc_time > 0:
            time_diff = abs(rest_time - grpc_time)
            lines.append(f"  Time difference: {time_diff:.2f}s")

    lines.extend([
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


def format_markdown(benchmark_results: dict) -> str:
    """Format benchmark results as a Markdown document."""
    lines = [
        "# LLM Benchmark Results",
        "",
        "## Test Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Number of Prompts | {benchmark_results['num_prompts']} |",
        f"| Max Tokens | {benchmark_results['max_tokens']} |",
        f"| Temperature | 0.1 (for consistency) |",
        f"| Model | {benchmark_results.get('model', 'smollm-135m-python')} |",
        f"| Timestamp | {benchmark_results['timestamp']} |",
        "",
        "## Results Summary",
        "",
        "| Client | Status | Prompts | Total Tokens | Total Time (s) | Tokens/sec |",
        "|--------|--------|---------|--------------|----------------|------------|",
    ]

    for client_name, data in benchmark_results["clients"].items():
        if data["success"]:
            stats = data["results"].get("stats", {})
            lines.append(
                f"| {client_name} | OK | "
                f"{stats.get('total_prompts', 'N/A')} | "
                f"{stats.get('total_tokens', 'N/A')} | "
                f"{stats.get('total_time_sec', 'N/A')} | "
                f"{stats.get('avg_tokens_per_sec', 'N/A')} |"
            )
        else:
            lines.append(
                f"| {client_name} | FAILED | - | - | - | - |"
            )

    lines.extend([
        "",
        "## Performance Comparison",
        "",
    ])

    rest = benchmark_results["clients"].get("REST", {})
    grpc = benchmark_results["clients"].get("gRPC", {})

    if rest.get("success") and grpc.get("success"):
        rest_tps = rest["results"]["stats"].get("avg_tokens_per_sec", 0)
        grpc_tps = grpc["results"]["stats"].get("avg_tokens_per_sec", 0)
        rest_time = rest["results"]["stats"].get("total_time_sec", 0)
        grpc_time = grpc["results"]["stats"].get("total_time_sec", 0)

        lines.extend([
            "| Metric | REST | gRPC | Difference |",
            "|--------|---------------|---------------|------------|",
            f"| Total Time (s) | {rest_time} | {grpc_time} | {abs(rest_time - grpc_time):.2f}s |",
            f"| Tokens/sec | {rest_tps} | {grpc_tps} | {abs(rest_tps - grpc_tps):.2f} |",
        ])

        if grpc_tps > 0 and rest_tps > 0:
            if grpc_tps > rest_tps:
                speedup = grpc_tps / rest_tps
                lines.append(f"\n**gRPC is {speedup:.2f}x faster** in tokens/sec throughput.")
            else:
                speedup = rest_tps / grpc_tps
                lines.append(f"\n**REST is {speedup:.2f}x faster** in tokens/sec throughput.")

    lines.extend([
        "",
        "## Notes",
        "",
        "- LLM inference is **compute-bound** (model generation), not transport-bound",
        "- For text generation, REST vs gRPC difference is minimal compared to inference time",
        "- gRPC still has lower overhead for the actual request/response transport",
        "",
        "## Client Descriptions",
        "",
        "| Client | Protocol | Encoding |",
        "|--------|----------|----------|",
        "| REST | HTTP/REST | Binary |",
        "| gRPC | gRPC | Binary protobuf |",
        "",
        "---",
        "",
        "*Generated by benchmark_llm_clients.py*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM Text Generation Clients (REST vs gRPC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with 5 default prompts
  python scripts/benchmark_llm_clients.py

  # Benchmark a specific model
  python scripts/benchmark_llm_clients.py --model tinyllama-python

  # Benchmark with more prompts
  python scripts/benchmark_llm_clients.py --num-prompts 10

  # Benchmark with custom prompts file
  python scripts/benchmark_llm_clients.py --prompts-file prompts.txt

  # Custom max tokens
  python scripts/benchmark_llm_clients.py --max-tokens 100

  # Custom report path
  python scripts/benchmark_llm_clients.py --report results/llm_benchmark.md

Output files (in same directory as report):
  <report>.md             - Markdown formatted report
  benchmark_results.json  - Detailed JSON results
  benchmark_summary.txt   - Human-readable summary
        """
    )

    parser.add_argument(
        "--num-prompts", "-n",
        type=int,
        default=5,
        help="Number of sample prompts to process (default: 5)"
    )
    parser.add_argument(
        "--prompts-file",
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate per prompt (default: 50)"
    )
    parser.add_argument(
        "--report", "-r",
        default=str(MODEL_RESULTS_DIR / "benchmark" / "llm_benchmark.md"),
        help=f"Full path to markdown report (default: {MODEL_RESULTS_DIR}/benchmark/llm_benchmark.md)"
    )
    parser.add_argument(
        "--rest-url",
        default=DEFAULT_REST_URL,
        help=f"REST proxy URL for model preloading (default: {DEFAULT_REST_URL})"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DOMINO_USER_API_KEY"),
        help="Domino API key (default: from DOMINO_USER_API_KEY env var)"
    )
    parser.add_argument(
        "--skip-preload",
        action="store_true",
        help="Skip model warmup (not recommended for accurate benchmarks)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Model name to benchmark (default: {DEFAULT_MODEL_NAME})"
    )

    args = parser.parse_args()

    # Determine prompts to use
    prompts = None
    prompts_file = None
    num_prompts = args.num_prompts

    if args.prompts_file:
        if not Path(args.prompts_file).exists():
            logger.error(f"Prompts file not found: {args.prompts_file}")
            sys.exit(1)
        prompts_file = args.prompts_file
        with open(prompts_file) as f:
            num_prompts = sum(1 for line in f if line.strip())
    else:
        # Generate sample prompts by repeating defaults
        prompts = (DEFAULT_PROMPTS * ((args.num_prompts // len(DEFAULT_PROMPTS)) + 1))[:args.num_prompts]

    # Parse report path and determine output directories
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

    logger.info("=" * 60)
    logger.info("LLM Text Generation Client Benchmark")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Prompts: {num_prompts}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Report: {report_path}")
    logger.info(f"REST URL: {args.rest_url}")

    # Warmup model to ensure fair benchmark timing (auto-loads via inference)
    if not args.skip_preload:
        logger.info("")
        warmup_model(args.rest_url, args.model, args.api_key)
    else:
        logger.info("")
        logger.info("Skipping model warmup (--skip-preload specified)")
        logger.info("WARNING: First inference may include model loading time")

    logger.info("=" * 60)

    # Run all clients
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "num_prompts": num_prompts,
        "max_tokens": args.max_tokens,
        "clients": {}
    }

    for client in CLIENTS:
        logger.info("")
        logger.info(f"--- {client['name']} ---")
        logger.info(f"    {client['description']}")

        output_path = json_dir / client["output"]

        result = run_client(
            script_name=client["script"],
            output_path=str(output_path),
            prompts=prompts,
            prompts_file=prompts_file,
            max_tokens=args.max_tokens,
            extra_args=client.get("extra_args", []),
            model_name=args.model
        )

        benchmark_results["clients"][client["name"]] = result

        if result["success"]:
            stats = result["results"].get("stats", {})
            logger.info(f"    Prompts: {stats.get('total_prompts', 'N/A')}")
            logger.info(f"    Total tokens: {stats.get('total_tokens', 'N/A')}")
            logger.info(f"    Total time: {stats.get('total_time_sec', 'N/A')}s")
            logger.info(f"    Tokens/sec: {stats.get('avg_tokens_per_sec', 'N/A')}")
        else:
            logger.error(f"    Failed: {result.get('error', 'Unknown')}")

    # Save full results to json/
    results_path = json_dir / "llm_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info("")
    logger.info(f"Full results saved to: {results_path}")

    # Generate and save summary to json/
    summary = format_summary(benchmark_results)
    summary_path = json_dir / "llm_benchmark_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info(f"Summary saved to: {summary_path}")

    # Generate and save markdown report
    markdown = format_markdown(benchmark_results)
    with open(report_path, "w") as f:
        f.write(markdown)
    logger.info(f"Markdown report saved to: {report_path}")

    # Print summary
    logger.info("")
    print(summary)


if __name__ == "__main__":
    main()