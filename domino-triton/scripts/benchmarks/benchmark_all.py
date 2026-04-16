#!/usr/bin/env python3
"""
benchmark_all.py

Runs all benchmark scripts (BERT, YOLOv8, Whisper, LLM) against a Triton deployment.
Uses environment variables for proxy URLs with localhost fallback.

Usage:
    python scripts/benchmark_all.py

Environment variables:
    TRITON_GRPC_URL  - gRPC proxy URL (default: localhost:50051)
    TRITON_REST_URL  - REST proxy URL (default: http://localhost:8080)
    DOMINO_USER_API_KEY - API key for authentication

Requirements:
    pip install -r docker/requirements-client.txt
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Test data
TEST_VIDEO = SAMPLES_DIR / "video.avi"
TEST_AUDIO = SAMPLES_DIR / "audio_sample.wav"

# Get URLs from environment
REST_URL = os.environ.get("TRITON_REST_URL", "http://localhost:8080")


def run_benchmark(name: str, cmd: list, timeout: int = 600) -> tuple[bool, str, float]:
    """Run a benchmark script and stream output in real-time."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
            cwd=str(PROJECT_ROOT)
        )

        # Stream stdout in real-time
        output_lines = []
        stderr_lines = []
        while True:
            # Read stdout line by line
            stdout_line = process.stdout.readline()
            if stdout_line:
                print(f"  {stdout_line.rstrip()}")
                output_lines.append(stdout_line)

            # Check if process has finished
            if process.poll() is not None:
                # Read remaining stdout
                for line in process.stdout:
                    print(f"  {line.rstrip()}")
                    output_lines.append(line)
                # Read all stderr
                stderr_lines = process.stderr.readlines()
                break

        duration = time.time() - start
        output = "".join(output_lines) + "".join(stderr_lines)
        success = process.returncode == 0

        if success:
            print(f"PASSED ({duration:.2f}s)")
        else:
            print(f"FAILED ({duration:.2f}s)")
            if stderr_lines:
                print(f"Errors: {''.join(stderr_lines)[:500]}")

        return success, output, duration
    except subprocess.TimeoutExpired:
        process.kill()
        duration = time.time() - start
        print(f"TIMEOUT after {timeout}s")
        return False, "Timeout", duration
    except Exception as e:
        duration = time.time() - start
        print(f"ERROR: {e}")
        return False, str(e), duration


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmark scripts (BERT, YOLOv8, Whisper, LLM)"
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async mode for gRPC clients (concurrent processing)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Running All Benchmarks")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  REST URL: {REST_URL}")
    print(f"  API Key:  {'Set' if os.environ.get('DOMINO_USER_API_KEY') else 'Not set'}")
    print(f"  Samples:  {SAMPLES_DIR}")
    print(f"  Results:  {RESULTS_DIR}")
    print(f"  Async:    {args.async_mode}")

    results = {}
    python = sys.executable

    # Build common args for async mode
    async_args = ["--async"] if args.async_mode else []

    # =========================================================================
    # BERT Text Classification Benchmark
    # =========================================================================

    results["bert"] = run_benchmark(
        "BERT Text Classification",
        [python, "scripts/benchmarks/benchmark_bert_clients.py",
         "--rest-url", REST_URL,
         "--num-texts", "10",
         "--report", str(RESULTS_DIR / "bert" / "benchmark" / "bert_benchmark.md")] + async_args
    )

    # =========================================================================
    # YOLOv8n Video Object Detection Benchmark
    # =========================================================================

    if TEST_VIDEO.exists():
        results["yolov8n"] = run_benchmark(
            "YOLOv8n Video Object Detection",
            [python, "scripts/benchmarks/benchmark_yolov8_clients.py",
             "--rest-url", REST_URL,
             "--video", str(TEST_VIDEO),
             "--max-frames", "120",
             "--report", str(RESULTS_DIR / "yolov8" / "benchmark" / "yolov8_benchmark.md")] + async_args
        )
    else:
        print(f"\nSkipping YOLOv8n benchmark - video not found: {TEST_VIDEO}")
        results["yolov8n"] = (None, "Skipped - no video", 0)

    # =========================================================================
    # Whisper Audio Transcription Benchmark
    # =========================================================================

    if TEST_AUDIO.exists():
        results["whisper"] = run_benchmark(
            "Whisper Audio Transcription",
            [python, "scripts/benchmarks/benchmark_whisper_clients.py",
             "--rest-url", REST_URL,
             "--audio", str(TEST_AUDIO),
             "--report", str(RESULTS_DIR / "whisper" / "benchmark" / "whisper_benchmark.md")] + async_args
        )
    else:
        print(f"\nSkipping Whisper benchmark - audio not found: {TEST_AUDIO}")
        results["whisper"] = (None, "Skipped - no audio", 0)

    # =========================================================================
    # SmolLM Text Generation Benchmark
    # =========================================================================

    results["llm"] = run_benchmark(
        "SmolLM Text Generation",
        [python, "scripts/benchmarks/benchmark_llm_clients.py",
         "--rest-url", REST_URL,
         "--num-prompts", "5",
         "--max-tokens", "50",
         "--report", str(RESULTS_DIR / "llm" / "benchmark" / "llm_benchmark.md")] + async_args
    )

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for name, (success, output, duration) in results.items():
        if success is None:
            status = "SKIPPED"
            skipped += 1
        elif success:
            status = f"PASSED ({duration:.2f}s)"
            passed += 1
        else:
            status = f"FAILED ({duration:.2f}s)"
            failed += 1

        symbol = "+" if success else ("?" if success is None else "X")
        print(f"  [{symbol}] {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Reports saved to: {RESULTS_DIR}")
    print("\nGenerated reports:")
    print(f"  - {RESULTS_DIR}/bert/benchmark/bert_benchmark.md")
    print(f"  - {RESULTS_DIR}/yolov8/benchmark/yolov8_benchmark.md")
    print(f"  - {RESULTS_DIR}/whisper/benchmark/whisper_benchmark.md")
    print(f"  - {RESULTS_DIR}/llm/benchmark/llm_benchmark.md")

    # Exit with error if any benchmarks failed
    if failed > 0:
        sys.exit(1)

    print("\nAll benchmarks completed!")


if __name__ == "__main__":
    main()
