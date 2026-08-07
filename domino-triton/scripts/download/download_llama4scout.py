#!/usr/bin/env python3
"""
download_llama4scout.py

Downloads Llama-4-Scout-17B-16E-Instruct weights and sets up the Triton vLLM
model directory for use with NVIDIA Triton Inference Server.

Llama 4 Scout is a Mixture-of-Experts model:
- 17B active parameters per forward pass, 109B total parameters
- 16 experts (MoE architecture)
- 128K token native context window
- Requires a HuggingFace token (gated by Meta)
- Recommended: 2x A100/H100 80GB or 4x A10G 24GB (tensor_parallel_size=2 or 4)

Usage:
    # With HF_TOKEN env var:
    HF_TOKEN=hf_... python scripts/download/download_llama4scout.py

    # With --token flag:
    python scripts/download/download_llama4scout.py --token hf_...

    # Custom base directory (e.g. an S3 FUSE mount or shared NFS path):
    python scripts/download/download_llama4scout.py --token hf_... --base-dir /mnt/s3/triton-repo

    # Custom tensor parallel size (for single-GPU or 4-GPU nodes):
    python scripts/download/download_llama4scout.py --token hf_... --tensor-parallel-size 1

    # Skip local verification (faster for pre-validated weights):
    python scripts/download/download_llama4scout.py --token hf_... --no-verify

Requirements:
    pip install huggingface_hub transformers

Output (relative to --base-dir, default: <project-root>/triton-repo):
    <base-dir>/weights/llama4scout-vllm/   <- downloaded HuggingFace weights
    <base-dir>/models/llama4scout-vllm/    <- Triton model directory
      config.pbtxt
      1/model.json                          <- vLLM engine args pointing to weights

Directory structure:
    <base-dir>/
    ├── models/llama4scout-vllm/
    │   ├── config.pbtxt
    │   └── 1/model.json
    └── weights/llama4scout-vllm/
        └── <HuggingFace model files>

"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
TRITON_MODEL_NAME = "llama4scout-vllm"
MODEL_VERSION = "1"
# Approximate size — inform the user before they start a long download
APPROX_SIZE_GB = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Llama-4-Scout and set up Triton vLLM model directory"
    )
    parser.add_argument(
        "--token",
        default=(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None,
        help="HuggingFace API token (required; also reads HF_TOKEN env var)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Base directory under which models/ and weights/ are created. "
            "Defaults to <project-root>/triton-repo. "
            "Use this to target an S3 FUSE mount, NFS share, or any custom path."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        choices=[1, 2, 4, 8],
        help="Number of GPUs for tensor parallelism (default: 2)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum sequence length for vLLM (default: 16384; max: 131072)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory to use for vLLM (default: 0.85)",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "int8"],
        default="none",
        help=(
            "Quantization to apply at model load time (default: none). "
            "'int8' uses bitsandbytes INT8, reducing runtime VRAM from ~218 GB to ~110 GB. "
            "Note: weights are always downloaded in BF16 (~200 GB); "
            "quantization is applied by vLLM on first load, not during download."
        ),
    )
    parser.add_argument(
        "--limit-mm-images",
        type=int,
        default=10,
        help=(
            "Max images per prompt (vllm limit_mm_per_prompt) -- needed for "
            "video/multi-frame testing (default: 10, matching the dashboard "
            "app's Max Frames UI default and the fp8-dynamic/int4 configs). "
            "vLLM rejects any multimodal request above this without erroring "
            "at config-validation time if it's missing entirely -- see "
            "docs/fp8_serving_configuration.md and "
            "docs/known_issues_and_todos.md."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip loading the model locally to verify the download",
    )
    parser.add_argument(
        "--ignore-ssl",
        action="store_true",
        help="Disable SSL certificate verification (corporate proxy environments)",
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


def _disable_ssl():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    import requests
    original_request = requests.Session.request
    def patched(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return original_request(self, method, url, **kwargs)
    requests.Session.request = patched


def download_model_weights(weights_dir: Path, token: str, ignore_ssl: bool) -> Path:
    from huggingface_hub import snapshot_download

    if ignore_ssl:
        _disable_ssl()

    model_weights_dir = weights_dir / TRITON_MODEL_NAME
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model:       {MODEL_ID}")
    print(f"Destination: {model_weights_dir}")
    print(f"Approx size: ~{APPROX_SIZE_GB} GB (bfloat16)")
    print(f"{'='*60}\n")

    if not token:
        print("ERROR: HuggingFace token required for Llama 4 Scout (gated model).")
        print("  Set HF_TOKEN env var or pass --token hf_...")
        print("  Request access at: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct")
        sys.exit(1)

    print("Downloading model files from HuggingFace Hub...")
    print("(Llama 4 Scout is a large model — this may take 30+ minutes on first run)\n")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(model_weights_dir),
        token=token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.ot"],
    )

    files = list(model_weights_dir.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"\nDownload complete.")
    print(f"  Files: {len([f for f in files if f.is_file()])}")
    print(f"  Total: {total_size / (1024**3):.1f} GB")
    print(f"  Path:  {model_weights_dir}")

    return model_weights_dir


def verify_model(model_weights_dir: Path):
    from transformers import AutoConfig

    print(f"\n{'='*60}")
    print("Verifying download...")
    print(f"{'='*60}\n")

    try:
        config = AutoConfig.from_pretrained(str(model_weights_dir))
        print(f"  Model type:       {config.model_type}")
        print(f"  Architecture:     {config.architectures}")
        # MoE-specific fields when present
        if hasattr(config, "num_experts"):
            print(f"  Num experts:      {config.num_experts}")
        if hasattr(config, "num_experts_per_tok"):
            print(f"  Active experts:   {config.num_experts_per_tok}")
        print("  Verification: PASSED")
    except Exception as e:
        print(f"  Verification FAILED: {e}")
        sys.exit(1)


def write_model_json(
    version_dir: Path,
    weights_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    quantization: str = "none",
    limit_mm_images: int = 10,
):
    model_json = {
        "model": weights_path,
        "tokenizer_mode": "auto",
        "dtype": "bfloat16",
        "limit_mm_per_prompt": {"image": limit_mm_images},
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "disable_log_requests": True,
        "enforce_eager": False,
    }

    if quantization == "int8":
        # bitsandbytes INT8: quantizes weights on first load; ~110 GB VRAM vs ~218 GB BF16.
        # float16 base dtype is required — bitsandbytes does not support bfloat16 as the
        # compute dtype when running in 8-bit mode.
        model_json["quantization"] = "bitsandbytes"
        model_json["load_format"] = "bitsandbytes"
        model_json["dtype"] = "float16"

    version_dir.mkdir(parents=True, exist_ok=True)
    model_json_path = version_dir / "model.json"
    with open(model_json_path, "w") as f:
        json.dump(model_json, f, indent=2)
        f.write("\n")

    print(f"\nvLLM engine config written: {model_json_path}")
    for k, v in model_json.items():
        print(f"  {k}: {v}")

    return model_json_path


def setup_triton_model(
    models_dir: Path,
    weights_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    quantization: str = "none",
    limit_mm_images: int = 10,
):
    source_dir = PROJECT_ROOT / "triton-repo-reference" / "models" / TRITON_MODEL_NAME
    source_config = source_dir / "config.pbtxt"

    target_dir = models_dir / TRITON_MODEL_NAME
    target_version_dir = target_dir / MODEL_VERSION
    target_config = target_dir / "config.pbtxt"

    print(f"\n{'='*60}")
    print(f"Setting up Triton model: {TRITON_MODEL_NAME}")
    print(f"{'='*60}\n")

    if not source_config.exists():
        print(f"ERROR: config.pbtxt not found at {source_config}")
        print("  Make sure triton-repo-reference/ is present in your workspace.")
        raise FileNotFoundError(f"config.pbtxt missing for {TRITON_MODEL_NAME}")

    target_version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_config, target_config)
    print(f"  config.pbtxt: COPIED")
    print(f"    Source: {source_config}")
    print(f"    Target: {target_config}")

    # Write model.json with resolved weights path and user-specified vLLM settings
    write_model_json(
        target_version_dir,
        weights_path,
        tensor_parallel_size,
        max_model_len,
        gpu_memory_utilization,
        quantization,
        limit_mm_images,
    )

    return target_dir


def main():
    args = parse_args()

    print("=" * 60)
    print("Llama-4-Scout-17B-16E-Instruct Setup for Triton vLLM")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)

    check_dependencies()

    triton_repo = Path(args.base_dir).resolve() if args.base_dir else PROJECT_ROOT / "triton-repo"
    models_dir = triton_repo / "models"
    weights_dir = triton_repo / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Download weights
    model_weights_dir = download_model_weights(weights_dir, args.token, args.ignore_ssl)

    # Verify the download
    if not args.no_verify:
        verify_model(model_weights_dir)

    # Use the resolved weights path directly in model.json so the container
    # sees the correct path regardless of where --base-dir points.
    weights_path = str(triton_repo / "weights" / TRITON_MODEL_NAME)

    # Set up Triton model directory
    setup_triton_model(
        models_dir,
        weights_path,
        args.tensor_parallel_size,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.quantization,
        args.limit_mm_images,
    )

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nBase directory: {triton_repo}")
    print(f"\nDirectory structure:")
    print(f"  {triton_repo}/")
    print(f"  ├── models/{TRITON_MODEL_NAME}/")
    print(f"  │   ├── config.pbtxt")
    print(f"  │   └── {MODEL_VERSION}/model.json   <- vLLM engine args")
    print(f"  └── weights/{TRITON_MODEL_NAME}/")
    print(f"      └── <HuggingFace model files>")
    print(f"\nWeights: {model_weights_dir}")
    print(f"GPU config: tensor_parallel_size={args.tensor_parallel_size}, "
          f"max_model_len={args.max_model_len}, "
          f"quantization={args.quantization}")
    print(f"\nTo start Triton with vLLM backend:")
    print(f"  docker compose --profile vllm up -d state backend-vllm http-proxy grpc-proxy")
    print(f"\nTo load the model:")
    print(f"  curl -X POST $TRITON_REST_URL/v2/repository/models/{TRITON_MODEL_NAME}/load")
    print(f"\nTo run inference:")
    print(f"  python scripts/clients/llm_vllm_grpc_client.py \\")
    print(f"    --model {TRITON_MODEL_NAME} \\")
    print(f"    --prompt 'What is the capital of France?' \\")
    print(f"    --apply-chat-template")


if __name__ == "__main__":
    main()
