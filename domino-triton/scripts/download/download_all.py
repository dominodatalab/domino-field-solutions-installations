#!/usr/bin/env python3
"""
download_all.py

Downloads all supported models for Triton Inference Server.

Usage:
    python scripts/download_all.py

This script downloads:
    - YOLOv8n (object detection)
    - BERT base uncased (text classification)
    - Whisper tiny (audio transcription)
    - SmolLM-135M (LLM text generation)

Requirements:
    pip install ultralytics transformers torch onnx onnxruntime optimum[onnxruntime] librosa soundfile
"""

import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_script(script_name: str) -> bool:
    """Run a download script and return success status."""
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False

    print(f"\n{'#'*60}")
    print(f"# Running {script_name}")
    print(f"{'#'*60}\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR.parent.parent),
    )

    return result.returncode == 0


def main():
    """Download all models."""
    print("=" * 60)
    print("Downloading All Models for Triton Inference Server")
    print("=" * 60)

    scripts = [
        "download_yolov8n.py",
        "download_bert.py",
        "download_whisper.py",
        "download_llm.py",
    ]

    results = {}

    for script in scripts:
        success = run_script(script)
        results[script] = success

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    all_success = True
    for script, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        symbol = "[OK]" if success else "[X]"
        print(f"  {symbol} {script}: {status}")
        if not success:
            all_success = False

    if all_success:
        print("\nAll models downloaded successfully!")
        print("\nTo start Triton with all models:")
        print("  docker compose up --build")
    else:
        print("\nSome downloads failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()