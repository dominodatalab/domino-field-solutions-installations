#!/usr/bin/env python3
"""
Benchmark Script for Whisper Audio Transcription Clients

Runs REST and gRPC client implementations with identical parameters
and compares their performance for remote Triton inference.

Usage:
    python scripts/benchmark_whisper_clients.py
    python scripts/benchmark_whisper_clients.py --audio samples/audio_sample.wav
    python scripts/benchmark_whisper_clients.py --audio-dir audio_files/ --report results/whisper_benchmark.md

Output (in same directory as --report):
    whisper_benchmark.md       - Markdown formatted report
    benchmark_results.json     - Detailed results from all clients
    benchmark_summary.txt      - Human-readable comparison table
"""

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
RESULTS_BASE_DIR = PROJECT_ROOT / "results"
MODEL_RESULTS_DIR = RESULTS_BASE_DIR / "whisper"

# Default REST proxy URL
DEFAULT_REST_URL = os.environ.get("TRITON_REST_URL", "http://localhost:8080")

# Model to preload for benchmarking
MODEL_NAME = "whisper-tiny-python"


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
        "script": "whisper_audio_rest_client.py",
        "args": [],
        "output": "whisper_rest.json",
        "description": "REST API with binary encoding"
    },
    {
        "name": "gRPC",
        "script": "whisper_audio_grpc_client.py",
        "args": [],
        "output": "whisper_grpc.json",
        "description": "gRPC with binary protobuf encoding"
    },
]


def run_client(
    script_name: str,
    output_path: str,
    extra_args: list = None,
    audio_files: list = None,
    audio_dir: str = None
) -> dict:
    """
    Run a client script and capture its results.

    Args:
        script_name: Name of the client script
        output_path: Path for output JSON
        extra_args: Additional arguments for the client
        audio_files: List of audio file paths
        audio_dir: Directory with audio files

    Returns:
        Dict with success status, duration, and results
    """
    script_path = CLIENTS_DIR / script_name

    cmd = [
        sys.executable,
        str(script_path),
        "--output", output_path
    ]

    if extra_args:
        cmd.extend(extra_args)

    if audio_dir:
        cmd.extend(["--audio-dir", audio_dir])
    elif audio_files:
        cmd.append("--audio")
        cmd.extend(audio_files)

    logger.info(f"Running: {script_name} {' '.join(extra_args or [])}")

    try:
        start_time = datetime.now()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
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
        "=" * 90,
        "WHISPER BENCHMARK SUMMARY",
        "=" * 90,
        "",
        f"Audio Files: {benchmark_results['num_files']}",
        f"Timestamp: {benchmark_results['timestamp']}",
        "",
        "-" * 90,
        f"{'Client':<20} {'Status':<10} {'Files':<8} {'Inference (s)':<14} {'Payload (MB)':<14} {'RTF':<10}",
        "-" * 90,
    ]

    for client_name, data in benchmark_results["clients"].items():
        if data["success"]:
            stats = data["results"].get("stats", {})
            rtf = stats.get("avg_realtime_factor", "N/A")
            rtf_str = f"{rtf:.3f}x" if isinstance(rtf, (int, float)) else rtf
            payload = stats.get("total_payload_mb", "N/A")
            payload_str = f"{payload:.2f}" if isinstance(payload, (int, float)) else str(payload)
            lines.append(
                f"{client_name:<20} "
                f"{'OK':<10} "
                f"{stats.get('successful', 'N/A'):<8} "
                f"{stats.get('total_inference_sec', 'N/A'):<14} "
                f"{payload_str:<14} "
                f"{rtf_str:<10}"
            )
        else:
            lines.append(
                f"{client_name:<20} "
                f"{'FAILED':<10} "
                f"{'-':<8} "
                f"{'-':<14} "
                f"{'-':<14} "
                f"{'-':<10}"
            )

    lines.extend([
        "-" * 90,
        "",
        "PERFORMANCE COMPARISON:",
        "",
    ])

    # Compare clients (REST vs gRPC)
    rest = benchmark_results["clients"].get("REST", {})
    grpc = benchmark_results["clients"].get("gRPC", {})

    if rest.get("success") and grpc.get("success"):
        rest_time = rest["results"]["stats"].get("total_inference_sec", 0)
        grpc_time = grpc["results"]["stats"].get("total_inference_sec", 0)

        if rest_time and grpc_time:
            speedup = rest_time / grpc_time
            lines.append(f"  gRPC vs REST: {speedup:.1f}x faster")

    lines.extend([
        "",
        "NOTES:",
        "  - RTF (Real-Time Factor): Ratio of processing time to audio duration",
        "  - RTF < 1.0 means faster than real-time",
        "  - Both REST and gRPC use binary encoding",
        "",
        "=" * 90,
    ])

    return "\n".join(lines)


def format_markdown(benchmark_results: dict) -> str:
    """Format benchmark results as a Markdown document."""
    lines = [
        "# Whisper Benchmark Results",
        "",
        "## Test Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | whisper-tiny-python |",
        f"| Audio Files | {benchmark_results['num_files']} |",
        f"| Timestamp | {benchmark_results['timestamp']} |",
        "",
        "## Results Summary",
        "",
        "| Client | Status | Files | Inference (s) | Payload (MB) | Avg RTF |",
        "|--------|--------|-------|---------------|--------------|---------|",
    ]

    for client_name, data in benchmark_results["clients"].items():
        if data["success"]:
            stats = data["results"].get("stats", {})
            rtf = stats.get("avg_realtime_factor", "N/A")
            rtf_str = f"{rtf:.3f}x" if isinstance(rtf, (int, float)) else rtf
            payload = stats.get("total_payload_mb", "-")
            payload_str = f"{payload:.2f}" if isinstance(payload, (int, float)) else str(payload)
            lines.append(
                f"| {client_name} | OK | "
                f"{stats.get('successful', 'N/A')} | "
                f"{stats.get('total_inference_sec', 'N/A')} | "
                f"{payload_str} | "
                f"{rtf_str} |"
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

    # Compare clients (REST vs gRPC)
    rest = benchmark_results["clients"].get("REST", {})
    grpc = benchmark_results["clients"].get("gRPC", {})

    if rest.get("success") and grpc.get("success"):
        lines.extend([
            "| Metric | REST | gRPC |",
            "|--------|------|------|",
        ])

        rest_stats = rest["results"]["stats"]
        grpc_stats = grpc["results"]["stats"]

        lines.append(
            f"| Inference Time (s) | {rest_stats.get('total_inference_sec', '-')} | "
            f"{grpc_stats.get('total_inference_sec', '-')} |"
        )

    lines.extend([
        "",
        "## Transcription Samples",
        "",
    ])

    # Include transcription samples from one client
    for client_name, data in benchmark_results["clients"].items():
        if data["success"]:
            files = data["results"].get("files", [])
            if files:
                lines.append(f"### {client_name}")
                lines.append("")
                for f in files[:3]:
                    if "transcription" in f:
                        file_name = Path(f["file"]).name
                        lines.append(f"**{file_name}** ({f.get('duration_sec', '?')}s):")
                        lines.append(f"> {f['transcription'][:200]}{'...' if len(f['transcription']) > 200 else ''}")
                        lines.append("")
                break

    lines.extend([
        "## Protocol Comparison",
        "",
        "| Protocol | Encoding | Use Case |",
        "|----------|----------|----------|",
        "| gRPC | Binary protobuf | Production, high throughput |",
        "| REST | Binary | Web clients, HTTP environments |",
        "",
        "## Notes",
        "",
        "- **Real-Time Factor (RTF)**: Ratio of processing time to audio duration",
        "- RTF < 1.0 means faster than real-time transcription",
        "- Both REST and gRPC use binary encoding for efficiency",
        "",
        "---",
        "",
        "*Generated by benchmark_whisper_clients.py*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper Audio Transcription Clients (REST vs gRPC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with sample audio
  python scripts/benchmark_whisper_clients.py

  # Benchmark with specific audio file
  python scripts/benchmark_whisper_clients.py --audio samples/audio_sample.wav

  # Benchmark with multiple audio files
  python scripts/benchmark_whisper_clients.py --audio file1.wav file2.wav

  # Benchmark with audio directory
  python scripts/benchmark_whisper_clients.py --audio-dir audio_files/

  # Custom report path
  python scripts/benchmark_whisper_clients.py --report results/whisper_benchmark.md

Output files (in same directory as report):
  <report>.md             - Markdown formatted report
  benchmark_results.json  - Detailed JSON results
  benchmark_summary.txt   - Human-readable summary
        """
    )

    parser.add_argument(
        "--audio", "-a",
        nargs="+",
        help="Audio file(s) to transcribe"
    )
    parser.add_argument(
        "--audio-dir",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--report", "-r",
        default=str(MODEL_RESULTS_DIR / "benchmark" / "whisper_benchmark.md"),
        help=f"Full path to markdown report (default: {MODEL_RESULTS_DIR}/benchmark/whisper_benchmark.md)"
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

    args = parser.parse_args()

    # Determine audio files to use
    audio_files = None
    audio_dir = None
    num_files = 0

    if args.audio:
        audio_files = args.audio
        num_files = len(audio_files)
    elif args.audio_dir:
        audio_dir = args.audio_dir
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        num_files = sum(1 for f in Path(audio_dir).iterdir()
                       if f.suffix.lower() in audio_extensions)
    else:
        # Default to sample audio
        sample_path = PROJECT_ROOT / "samples" / "audio_sample.wav"
        if sample_path.exists():
            audio_files = [str(sample_path)]
            num_files = 1
        else:
            logger.error("No audio files specified and no sample found.")
            logger.error("Run download_whisper.py first to create a sample audio file.")
            sys.exit(1)

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
    logger.info("Whisper Audio Transcription Client Benchmark")
    logger.info("=" * 60)
    logger.info(f"Audio files: {num_files}")
    logger.info(f"Report: {report_path}")
    logger.info(f"REST URL: {args.rest_url}")

    # Warmup model to ensure fair benchmark timing (auto-loads via inference)
    if not args.skip_preload:
        logger.info("")
        warmup_model(args.rest_url, MODEL_NAME, args.api_key)
    else:
        logger.info("")
        logger.info("Skipping model warmup (--skip-preload specified)")
        logger.info("WARNING: First inference may include model loading time")

    logger.info("=" * 60)

    # Run all clients
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "num_files": num_files,
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
            extra_args=client.get("args"),
            audio_files=audio_files,
            audio_dir=audio_dir
        )

        benchmark_results["clients"][client["name"]] = result

        if result["success"]:
            stats = result["results"].get("stats", {})
            logger.info(f"    Files: {stats.get('successful', 'N/A')}")
            logger.info(f"    Inference time: {stats.get('total_inference_sec', 'N/A')}s")
            payload = stats.get('total_payload_mb')
            if payload:
                logger.info(f"    Payload: {payload:.2f}MB")
            logger.info(f"    Avg RTF: {stats.get('avg_realtime_factor', 'N/A')}")
        else:
            logger.error(f"    Failed: {result.get('error', 'Unknown')}")

    # Save full results to json/
    results_path = json_dir / "whisper_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info("")
    logger.info(f"Full results saved to: {results_path}")

    # Generate and save summary to json/
    summary = format_summary(benchmark_results)
    summary_path = json_dir / "whisper_benchmark_summary.txt"
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
