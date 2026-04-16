#!/usr/bin/env python3
"""
Benchmark Script for YOLOv8n Video Inference Clients

Runs all three client implementations (REST JSON, REST Base64, gRPC) with
identical parameters and compares their performance.

Usage:
    python scripts/benchmark_yolov8_clients.py --video samples/video.avi
    python scripts/benchmark_yolov8_clients.py --video samples/video.avi --max-frames 10
    python scripts/benchmark_yolov8_clients.py --video samples/video.avi --report results/my_benchmark.md

Output (in same directory as --report):
    benchmark_report.md     - Markdown formatted report
    benchmark_results.json  - Detailed results from all clients
    benchmark_summary.txt   - Human-readable comparison table
    benchmark_*.json        - Individual client results
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
CLIENTS_DIR = SCRIPTS_DIR.parent / "clients"
RESULTS_BASE_DIR = SCRIPTS_DIR.parent.parent / "results"
MODEL_RESULTS_DIR = RESULTS_BASE_DIR / "yolov8"

# Default REST proxy URL
DEFAULT_REST_URL = os.environ.get("TRITON_REST_URL", "http://localhost:8080")

# Model to preload for benchmarking
MODEL_NAME = "yolov8n"


def warmup_model(rest_url: str, model_name: str) -> bool:
    """
    Warm up a model by checking if it's ready.

    Args:
        rest_url: REST proxy URL (e.g., http://localhost:8080)
        model_name: Name of the model to warm up

    Returns:
        True if model is ready, False otherwise

    Note:
        Authentication is handled automatically via auth_helper which prefers:
        1. DOMINO_USER_TOKEN env var
        2. DOMINO_API_PROXY/access-token endpoint (inside Domino)
        3. DOMINO_USER_API_KEY env var
    """
    url = f"{rest_url.rstrip('/')}/v2/models/{model_name}/ready"
    headers = get_auth_headers() or {}

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
        "name": "REST (JSON)",
        "script": "yolov8n_video_rest_client.py",
        "output": "yolov8_rest_json.json",
        "description": "REST API with JSON encoding (~17MB/frame payload)",
        "extra_args": ["--json-encoding"]
    },
    {
        "name": "REST (Binary)",
        "script": "yolov8n_video_rest_client.py",
        "output": "yolov8_rest_binary.json",
        "description": "REST API with binary encoding (~5MB/frame payload)",
        "extra_args": []
    },
    {
        "name": "gRPC",
        "script": "yolov8n_video_grpc_client.py",
        "output": "yolov8_grpc.json",
        "description": "gRPC with binary tensor transfer (~5MB/frame payload)",
        "extra_args": []
    },
]


def run_client(
    script_name: str,
    video_path: str,
    output_path: str,
    fps: float = None,
    max_frames: int = 5,
    extra_args: list = None
) -> dict:
    """
    Run a client script and stream its output in real-time.

    Args:
        script_name: Name of the client script
        video_path: Path to input video
        output_path: Path for output JSON
        fps: Target FPS (optional)
        max_frames: Maximum frames to process
        extra_args: Additional command-line arguments

    Returns:
        Dict with success status, duration, and results
    """
    script_path = CLIENTS_DIR / script_name

    cmd = [
        sys.executable,
        str(script_path),
        "--video", video_path,
        "--max-frames", str(max_frames),
        "--output", output_path
    ]

    if fps:
        cmd.extend(["--fps", str(fps)])

    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running: {script_name}")

    try:
        start_time = datetime.now()
        # Use Popen to stream output in real-time while capturing stderr for errors
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line-buffered
        )

        # Stream stdout in real-time
        stderr_lines = []
        while True:
            # Read stdout line by line
            stdout_line = process.stdout.readline()
            if stdout_line:
                print(f"    {stdout_line.rstrip()}")

            # Check if process has finished
            if process.poll() is not None:
                # Read remaining stdout
                for line in process.stdout:
                    print(f"    {line.rstrip()}")
                # Read all stderr
                stderr_lines = process.stderr.readlines()
                break

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stderr_output = "".join(stderr_lines)

        if process.returncode != 0:
            logger.error(f"  Failed: {stderr_output[-500:] if stderr_output else 'Unknown error'}")
            return {
                "success": False,
                "error": stderr_output[-500:] if stderr_output else "Unknown error",
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
        process.kill()
        logger.error(f"  Timeout after 5 minutes")
        return {
            "success": False,
            "error": "Timeout after 5 minutes",
            "duration": 300
        }
    except Exception as e:
        logger.error(f"  Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "duration": 0
        }


def format_summary(benchmark_results: dict) -> str:
    """
    Format benchmark results as a human-readable summary.

    Args:
        benchmark_results: Full benchmark results dict

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 80,
        "BENCHMARK SUMMARY",
        "=" * 80,
        "",
        f"Video: {benchmark_results['video']}",
        f"Frames: {benchmark_results['max_frames']}",
        f"FPS: {benchmark_results.get('fps', 'native')}",
        f"Timestamp: {benchmark_results['timestamp']}",
        "",
        "-" * 80,
        f"{'Client':<20} {'Status':<10} {'Frames':<8} {'Total (s)':<12} {'Avg (ms)':<12} {'FPS':<8} {'Payload (MB)':<12}",
        "-" * 80,
    ]

    for client_name, data in benchmark_results["clients"].items():
        if data["success"]:
            stats = data["results"].get("stats", {})
            lines.append(
                f"{client_name:<20} "
                f"{'OK':<10} "
                f"{stats.get('total_frames', 'N/A'):<8} "
                f"{stats.get('total_time_sec', 'N/A'):<12} "
                f"{stats.get('avg_frame_ms', stats.get('avg_inference_ms', 'N/A')):<12} "
                f"{stats.get('fps', 'N/A'):<8} "
                f"{stats.get('total_payload_mb', 'N/A'):<12}"
            )
        elif data.get("expected_failure"):
            lines.append(
                f"{client_name:<20} "
                f"{'N/A*':<10} "
                f"{'-':<8} "
                f"{'-':<12} "
                f"{'-':<12} "
                f"{'-':<8} "
                f"{'-':<12}"
            )
        else:
            lines.append(
                f"{client_name:<20} "
                f"{'FAILED':<10} "
                f"{'-':<8} "
                f"{'-':<12} "
                f"{'-':<12} "
                f"{'-':<8} "
                f"{'-':<12}"
            )

    lines.extend([
        "-" * 80,
        "",
        "PERFORMANCE COMPARISON:",
        ""
    ])

    # Calculate relative performance
    rest_json = benchmark_results["clients"].get("REST (JSON)", {})
    rest_binary = benchmark_results["clients"].get("REST (Binary)", {})
    grpc = benchmark_results["clients"].get("gRPC", {})

    def get_avg_time(client_data):
        if not client_data.get("success"):
            return None
        stats = client_data["results"].get("stats", {})
        return stats.get("avg_frame_ms", stats.get("avg_inference_ms"))

    rest_json_time = get_avg_time(rest_json)
    rest_binary_time = get_avg_time(rest_binary)
    grpc_time = get_avg_time(grpc)

    if rest_json_time and rest_binary_time:
        speedup = rest_json_time / rest_binary_time
        lines.append(f"  REST (Binary) is {speedup:.1f}x faster than REST (JSON)")

    if rest_binary_time and grpc_time:
        speedup = rest_binary_time / grpc_time
        lines.append(f"  gRPC is {speedup:.1f}x faster than REST (Binary)")

    if rest_json_time and grpc_time:
        speedup = rest_json_time / grpc_time
        lines.append(f"  gRPC is {speedup:.1f}x faster than REST (JSON)")

    lines.extend([
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


def format_markdown(benchmark_results: dict) -> str:
    """
    Format benchmark results as a Markdown document.

    Args:
        benchmark_results: Full benchmark results dict

    Returns:
        Formatted Markdown string
    """
    lines = [
        "# Benchmark Results",
        "",
        "## Test Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Video | `{benchmark_results['video']}` |",
        f"| Frames | {benchmark_results['max_frames']} |",
        f"| FPS | {benchmark_results.get('fps') or 'native'} |",
        f"| Timestamp | {benchmark_results['timestamp']} |",
        "",
        "## Results Summary",
        "",
        "| Client | Status | Frames | Total (s) | Avg (ms) | FPS | Payload (MB) |",
        "|--------|--------|--------|-----------|----------|-----|--------------|",
    ]

    for client_name, data in benchmark_results["clients"].items():
        if data["success"]:
            stats = data["results"].get("stats", {})
            lines.append(
                f"| {client_name} | OK | "
                f"{stats.get('total_frames', 'N/A')} | "
                f"{stats.get('total_time_sec', 'N/A')} | "
                f"{stats.get('avg_frame_ms', stats.get('avg_inference_ms', 'N/A'))} | "
                f"{stats.get('fps', 'N/A')} | "
                f"{stats.get('total_payload_mb', 'N/A')} |"
            )
        elif data.get("expected_failure"):
            lines.append(
                f"| {client_name} | N/A* | - | - | - | - | - |"
            )
        else:
            lines.append(
                f"| {client_name} | FAILED | - | - | - | - | - |"
            )

    # Add note about expected failures
    expected_failures = [
        (name, data.get("failure_reason", "Known limitation"))
        for name, data in benchmark_results["clients"].items()
        if data.get("expected_failure")
    ]
    if expected_failures:
        lines.extend([
            "",
            "> **Note:** *N/A indicates the protocol is not suitable for this workload:*",
        ])
        for name, reason in expected_failures:
            lines.append(f"> - **{name}**: {reason}")
        lines.append("")

    lines.extend([
        "",
        "## Performance Comparison",
        "",
    ])

    # Calculate relative performance
    rest_json = benchmark_results["clients"].get("REST (JSON)", {})
    rest_binary = benchmark_results["clients"].get("REST (Binary)", {})
    grpc = benchmark_results["clients"].get("gRPC", {})

    def get_stats(client_data):
        if not client_data.get("success"):
            return None, None
        stats = client_data["results"].get("stats", {})
        time = stats.get("avg_frame_ms", stats.get("avg_inference_ms", "N/A"))
        payload = stats.get("total_payload_mb", "N/A")
        return time, payload

    rest_json_time, rest_json_payload = get_stats(rest_json)
    rest_binary_time, rest_binary_payload = get_stats(rest_binary)
    grpc_time, grpc_payload = get_stats(grpc)

    lines.append("| Metric | REST (JSON) | REST (Binary) | gRPC |")
    lines.append("|--------|-------------|---------------|------|")
    lines.append(f"| Avg frame (ms) | {rest_json_time} | {rest_binary_time} | {grpc_time} |")
    lines.append(f"| Total payload (MB) | {rest_json_payload} | {rest_binary_payload} | {grpc_payload} |")

    lines.append("")
    lines.append("### Speedup Summary")
    lines.append("")

    if rest_json_time and rest_binary_time and isinstance(rest_json_time, (int, float)) and isinstance(rest_binary_time, (int, float)):
        speedup = rest_json_time / rest_binary_time
        lines.append(f"- **REST (Binary) is {speedup:.1f}x faster** than REST (JSON)")

    if rest_binary_time and grpc_time and isinstance(rest_binary_time, (int, float)) and isinstance(grpc_time, (int, float)):
        speedup = rest_binary_time / grpc_time
        lines.append(f"- **gRPC is {speedup:.1f}x faster** than REST (Binary)")

    if rest_json_time and grpc_time and isinstance(rest_json_time, (int, float)) and isinstance(grpc_time, (int, float)):
        speedup = rest_json_time / grpc_time
        lines.append(f"- **gRPC is {speedup:.1f}x faster** than REST (JSON)")

    lines.extend([
        "",
        "## Client Descriptions",
        "",
        "| Client | Protocol | Encoding |",
        "|--------|----------|----------|",
        "| REST (JSON) | HTTP/REST | JSON arrays |",
        "| REST (Binary) | HTTP/REST | Binary tensor data |",
        "| gRPC | gRPC | Binary protobuf |",
        "",
        "---",
        "",
        f"*Generated by benchmark_yolov8_clients.py*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv8n Video Inference Clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with 5 frames (default)
  python scripts/benchmark_yolov8_clients.py --video samples/video.avi

  # Benchmark with 20 frames at 5 FPS
  python scripts/benchmark_yolov8_clients.py --video samples/video.avi --fps 5 --max-frames 20

  # Custom report path
  python scripts/benchmark_yolov8_clients.py --video samples/video.avi --report results/my_benchmark.md

Output files (in same directory as report):
  <report>.md             - Markdown formatted report
  benchmark_results.json  - Detailed JSON results
  benchmark_summary.txt   - Human-readable summary
  benchmark_*.json        - Individual client results
        """
    )

    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        help="Target FPS for frame extraction (default: video native)"
    )
    parser.add_argument(
        "--max-frames", "-n",
        type=int,
        default=5,
        help="Maximum number of frames to process (default: 5)"
    )
    parser.add_argument(
        "--report", "-r",
        default=str(MODEL_RESULTS_DIR / "benchmark" / "yolov8_benchmark.md"),
        help=f"Full path to markdown report (default: {MODEL_RESULTS_DIR}/benchmark/yolov8_benchmark.md)"
    )
    parser.add_argument(
        "--rest-url",
        default=DEFAULT_REST_URL,
        help=f"REST proxy URL for model preloading (default: {DEFAULT_REST_URL})"
    )
    # Note: --api-key removed. Auth is now automatic via auth_helper which prefers:
    # 1. DOMINO_USER_TOKEN env var
    # 2. DOMINO_API_PROXY/access-token endpoint (inside Domino)
    # 3. DOMINO_USER_API_KEY env var
    parser.add_argument(
        "--skip-preload",
        action="store_true",
        help="Skip model warmup (not recommended for accurate benchmarks)"
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async mode for gRPC client (concurrent batch processing)"
    )

    args = parser.parse_args()

    # Validate video path
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    # Parse report path and determine output directories
    report_path = Path(args.report)
    if not report_path.suffix:
        report_path = report_path.with_suffix(".md")

    # Determine benchmark directory from report path
    # If report is in .../benchmark/, use that; otherwise use report's parent
    if report_path.parent.name == "benchmark":
        benchmark_dir = report_path.parent
    else:
        benchmark_dir = report_path.parent

    # Create benchmark/ and benchmark/json/ subdirectories
    json_dir = benchmark_dir / "json"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Markdown goes directly in benchmark/, update path if needed
    if report_path.parent != benchmark_dir:
        report_path = benchmark_dir / report_path.name

    logger.info("=" * 60)
    logger.info("YOLOv8n Video Inference Client Benchmark")
    logger.info("=" * 60)
    logger.info(f"Video: {args.video}")
    logger.info(f"Max frames: {args.max_frames}")
    logger.info(f"REST URL: {args.rest_url}")

    # Warmup model to ensure fair benchmark timing (auto-loads via inference)
    if not args.skip_preload:
        logger.info("")
        warmup_model(args.rest_url, MODEL_NAME)
    else:
        logger.info("")
        logger.info("Skipping model warmup (--skip-preload specified)")
        logger.info("WARNING: First inference may include model loading time")
    logger.info(f"FPS: {args.fps or 'native'}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 60)

    # Run all clients
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "video": args.video,
        "max_frames": args.max_frames,
        "fps": args.fps,
        "clients": {}
    }

    for client in CLIENTS:
        logger.info("")
        logger.info(f"--- {client['name']} ---")
        logger.info(f"    {client['description']}")

        output_path = json_dir / client["output"]

        # Build extra args, adding --async for gRPC clients if requested
        extra_args = list(client.get("extra_args", []))
        if args.async_mode and "grpc" in client["script"]:
            extra_args.append("--async")
            logger.info(f"    (async mode enabled)")

        result = run_client(
            script_name=client["script"],
            video_path=args.video,
            output_path=str(output_path),
            fps=args.fps,
            max_frames=args.max_frames,
            extra_args=extra_args
        )

        # Handle expected failures (e.g., REST JSON with large image tensors)
        if not result["success"] and client.get("expected_failure"):
            result["expected_failure"] = True
            result["failure_reason"] = client.get("failure_reason", "Known limitation")
            logger.warning(f"    Expected failure: {client.get('failure_reason', 'Known limitation')}")

        benchmark_results["clients"][client["name"]] = result

        if result["success"]:
            stats = result["results"].get("stats", {})
            logger.info(f"    Frames: {stats.get('total_frames', 'N/A')}")
            logger.info(f"    Total time: {stats.get('total_time_sec', 'N/A')}s")
            logger.info(f"    Avg frame: {stats.get('avg_frame_ms', stats.get('avg_inference_ms', 'N/A'))}ms")
            logger.info(f"    Effective FPS: {stats.get('fps', 'N/A')}")
            logger.info(f"    Total payload: {stats.get('total_payload_mb', 'N/A')}MB")
        elif not result.get("expected_failure"):
            logger.error(f"    Failed: {result.get('error', 'Unknown')}")

    # Save full results to json/
    results_path = json_dir / "yolov8_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info("")
    logger.info(f"Full results saved to: {results_path}")

    # Generate and save summary to json/
    summary = format_summary(benchmark_results)
    summary_path = json_dir / "yolov8_benchmark_summary.txt"
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