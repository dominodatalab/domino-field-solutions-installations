#!/usr/bin/env python3
"""
test_all_clients.py

Tests all inference clients (gRPC and REST) against all models.
Uses environment variables for proxy URLs with localhost fallback.

Usage:
    python scripts/test_all_clients.py

Environment variables:
    TRITON_GRPC_URL  - gRPC proxy URL (default: localhost:50051)
    TRITON_REST_URL  - REST proxy URL (default: http://localhost:8080)
    DOMINO_USER_API_KEY - API key for authentication

Requirements:
    pip install -r docker/requirements-client.txt
"""

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
TEST_TEXTS = ["This movie was amazing!", "The food was terrible."]
TEST_VIDEO = SAMPLES_DIR / "video.avi"
TEST_AUDIO = SAMPLES_DIR / "audio_sample.wav"
TEST_PROMPT = "What is the capital of France?"

# Get URLs from environment
GRPC_URL = os.environ.get("TRITON_GRPC_URL", "localhost:50051")
REST_URL = os.environ.get("TRITON_REST_URL", "http://localhost:8080")


def run_client(name: str, cmd: list, timeout: int = 120) -> tuple[bool, str, float]:
    """Run a client script and return (success, output, duration)."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )
        duration = time.time() - start
        output = result.stdout + result.stderr
        success = result.returncode == 0

        if success:
            print(f"✓ PASSED ({duration:.2f}s)")
        else:
            print(f"✗ FAILED ({duration:.2f}s)")
            print(f"Output: {output[:500]}")

        return success, output, duration
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        print(f"✗ TIMEOUT after {timeout}s")
        return False, "Timeout", duration
    except Exception as e:
        duration = time.time() - start
        print(f"✗ ERROR: {e}")
        return False, str(e), duration


def main():
    print("=" * 60)
    print("Testing All Inference Clients")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  GRPC URL: {GRPC_URL}")
    print(f"  REST URL: {REST_URL}")
    print(f"  API Key:  {'Set' if os.environ.get('DOMINO_USER_API_KEY') else 'Not set'}")
    print(f"  Samples:  {SAMPLES_DIR}")
    print(f"  Results:  {RESULTS_DIR}")

    results = {}
    python = sys.executable

    # =========================================================================
    # BERT Text Classification
    # =========================================================================

    # BERT gRPC
    results["bert_grpc"] = run_client(
        "BERT (gRPC)",
        [python, "scripts/clients/bert_text_grpc_client.py",
         "--grpc-url", GRPC_URL,
         "--texts"] + TEST_TEXTS + [
         "--output", str(RESULTS_DIR / "bert_grpc.json")]
    )

    # BERT REST
    results["bert_rest"] = run_client(
        "BERT (REST)",
        [python, "scripts/clients/bert_text_rest_client.py",
         "--rest-url", REST_URL,
         "--texts"] + TEST_TEXTS + [
         "--output", str(RESULTS_DIR / "bert_rest.json")]
    )

    # =========================================================================
    # YOLOv8n Video Object Detection
    # =========================================================================

    if TEST_VIDEO.exists():
        # YOLOv8n gRPC
        results["yolov8n_grpc"] = run_client(
            "YOLOv8n (gRPC)",
            [python, "scripts/clients/yolov8n_video_grpc_client.py",
             "--grpc-url", GRPC_URL,
             "--video", str(TEST_VIDEO),
             "--max-frames", "10",
             "--output", str(RESULTS_DIR / "yolov8n_grpc.json")]
        )

        # YOLOv8n REST (base64)
        results["yolov8n_rest"] = run_client(
            "YOLOv8n (REST)",
            [python, "scripts/clients/yolov8n_video_rest_client.py",
             "--rest-url", REST_URL,
             "--video", str(TEST_VIDEO),
             "--max-frames", "10",
             "--output", str(RESULTS_DIR / "yolov8n_rest.json")]
        )
    else:
        print(f"\n⚠ Skipping YOLOv8n tests - video not found: {TEST_VIDEO}")
        results["yolov8n_grpc"] = (None, "Skipped - no video", 0)
        results["yolov8n_rest"] = (None, "Skipped - no video", 0)

    # =========================================================================
    # Whisper Audio Transcription
    # =========================================================================

    if TEST_AUDIO.exists():
        # Whisper gRPC
        results["whisper_grpc"] = run_client(
            "Whisper (gRPC)",
            [python, "scripts/clients/whisper_audio_grpc_client.py",
             "--grpc-url", GRPC_URL,
             "--audio", str(TEST_AUDIO),
             "--output", str(RESULTS_DIR / "whisper_grpc.json")]
        )

        # Whisper REST
        results["whisper_rest"] = run_client(
            "Whisper (REST)",
            [python, "scripts/clients/whisper_audio_rest_client.py",
             "--rest-url", REST_URL,
             "--audio", str(TEST_AUDIO),
             "--output", str(RESULTS_DIR / "whisper_rest.json")]
        )
    else:
        print(f"\n⚠ Skipping Whisper tests - audio not found: {TEST_AUDIO}")
        results["whisper_grpc"] = (None, "Skipped - no audio", 0)
        results["whisper_rest"] = (None, "Skipped - no audio", 0)

    # =========================================================================
    # SmolLM Text Generation
    # =========================================================================

    # LLM gRPC
    results["llm_grpc"] = run_client(
        "SmolLM (gRPC)",
        [python, "scripts/clients/llm_text_grpc_client.py",
         "--grpc-url", GRPC_URL,
         "--prompt", TEST_PROMPT,
         "--max-tokens", "50",
         "--output", str(RESULTS_DIR / "llm_grpc.json")]
    )

    # LLM REST
    results["llm_rest"] = run_client(
        "SmolLM (REST)",
        [python, "scripts/clients/llm_text_rest_client.py",
         "--rest-url", REST_URL,
         "--prompt", TEST_PROMPT,
         "--max-tokens", "50",
         "--output", str(RESULTS_DIR / "llm_rest.json")]
    )

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
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

        symbol = "✓" if success else ("⚠" if success is None else "✗")
        print(f"  {symbol} {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Results saved to: {RESULTS_DIR}")

    # Exit with error if any tests failed
    if failed > 0:
        sys.exit(1)

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
