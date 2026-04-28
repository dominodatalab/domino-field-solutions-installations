#!/usr/bin/env python3
"""
download_hf_model.py

Generic script to download any HuggingFace model to a specified local weights folder.
Useful for pre-downloading models before deploying to air-gapped / GovCloud environments
where the Triton container cannot reach huggingface.co at runtime.

Usage:
    python scripts/download/download_hf_model.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output triton-repo/weights/qwen25-vllm

    # Custom destination:
    python scripts/download/download_hf_model.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --output /mnt/s3/weights/llama-3b \
        --token hf_...

    # Skip verification (faster, useful for large models):
    python scripts/download/download_hf_model.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output triton-repo/weights/qwen25-vllm \
        --no-verify

After downloading, update the model's model.json to point to the local path:
    {
        "model": "/triton-repo/weights/qwen25-vllm",
        ...
    }
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model to a local weights folder"
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination directory for model weights"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HuggingFace API token (for gated models). Falls back to HF_TOKEN env var."
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip loading the model locally to verify the download"
    )
    parser.add_argument(
        "--ignore-ssl", action="store_true",
        help="Disable SSL certificate verification (use in corporate proxy environments)"
    )
    return parser.parse_args()


def check_dependencies():
    missing = []
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def download_model(model_id: str, output_dir: Path, token: str | None, ignore_ssl: bool):
    from huggingface_hub import snapshot_download

    if ignore_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        # Patch requests session to disable SSL verification
        import requests
        original_request = requests.Session.request
        def patched_request(self, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return original_request(self, method, url, **kwargs)
        requests.Session.request = patched_request

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model:       {model_id}")
    print(f"Destination: {output_dir}")
    print(f"{'='*60}\n")

    print("Downloading model files from HuggingFace Hub...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        token=token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    print(f"\nDownload complete: {output_dir}")
    files = list(output_dir.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"Files: {len([f for f in files if f.is_file()])}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")


def verify_model(output_dir: Path):
    from transformers import AutoConfig

    print(f"\n{'='*60}")
    print("Verifying download...")
    print(f"{'='*60}\n")

    try:
        config = AutoConfig.from_pretrained(str(output_dir))
        print(f"  Model type:   {config.model_type}")
        print(f"  Architecture: {config.architectures}")
        print("  Verification: PASSED")
    except Exception as e:
        print(f"  Verification FAILED: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    check_dependencies()

    output_dir = Path(args.output).resolve()

    download_model(args.model, output_dir, args.token, args.ignore_ssl)

    if not args.no_verify:
        verify_model(output_dir)

    print(f"\n{'='*60}")
    print("SUCCESS")
    print(f"{'='*60}")
    print(f"\nWeights saved to: {output_dir}")
    print(f"\nUpdate your model.json to use the local path:")
    print(f'    "model": "{output_dir}"')


if __name__ == "__main__":
    main()