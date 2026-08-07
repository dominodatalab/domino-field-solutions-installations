#!/usr/bin/env python3
"""
download_llama4scout_fp8_dynamic.py

Downloads RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic -- a
compressed-tensors-format FP8 quantization of Llama 4 Scout, community-
maintained with close ties to vLLM (as opposed to NVIDIA's own
modelopt-format FP8 checkpoint, which hit a structural checkpoint-naming
mismatch in vLLM's Llama4 MoE weight loader; see
docs/s3_fuse_latency_test_plan.md for that investigation). compressed-tensors
is vLLM's own native, first-class quantization format, so this checkpoint is
expected to load without the modelopt-specific naming issues.

Mirrors download_llama4scout_fp8.py's structure/conventions for consistency.

Usage:
    # With HF_TOKEN env var:
    HF_TOKEN=hf_... python scripts/download/download_llama4scout_fp8_dynamic.py

    # With --token flag:
    python scripts/download/download_llama4scout_fp8_dynamic.py --token hf_...

    # Custom base directory (e.g. an S3 FUSE mount or shared NFS path):
    python scripts/download/download_llama4scout_fp8_dynamic.py --token hf_... --base-dir /mnt/s3/triton-repo

    # Custom tensor parallel size (match it to your actual node's GPU count):
    python scripts/download/download_llama4scout_fp8_dynamic.py --token hf_... --tensor-parallel-size 2

    # Skip local verification (faster for pre-validated weights):
    python scripts/download/download_llama4scout_fp8_dynamic.py --token hf_... --no-verify

Requirements:
    pip install huggingface_hub transformers

Output (relative to --base-dir, default: <project-root>/triton-repo):
    <base-dir>/weights/llama4scout-vllm-fp8-dynamic/   <- downloaded HuggingFace weights
    <base-dir>/models/llama4scout-vllm-fp8-dynamic/    <- Triton model directory
      config.pbtxt
      1/model.json                                      <- vLLM engine args pointing to weights
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_ID = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"
TRITON_MODEL_NAME = "llama4scout-vllm-fp8-dynamic"
MODEL_VERSION = "1"
APPROX_SIZE_GB = 110  # confirmed via HfApi model_info: 106.8 GB across 36 files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download RedHatAI's compressed-tensors FP8 Llama-4-Scout checkpoint and set up Triton vLLM model directory"
    )
    parser.add_argument(
        "--token",
        default=(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None,
        help="HuggingFace API token (also reads HF_TOKEN env var, or cached ~/.cache/huggingface/token)",
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
        default=8,
        choices=[1, 2, 4, 8],
        help=(
            "Number of GPUs for tensor parallelism (default: 8, sized for an "
            "8x A10G/L4 node -- see download_llama4scout_fp8.py's GPU sizing "
            "note before overriding to a smaller value)"
        ),
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
        "--limit-mm-images",
        type=int,
        default=10,
        help=(
            "Max images per prompt (vllm limit_mm_per_prompt) -- needed for "
            "video/multi-frame testing (default: 10, matching the dashboard "
            "app's Max Frames UI default). vLLM rejects any multimodal "
            "request above this without erroring at config-validation time "
            "if it's missing entirely -- see docs/fp8_serving_configuration.md."
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
    print(f"Approx size: ~{APPROX_SIZE_GB} GB (FP8 compressed-tensors)")
    print(f"{'='*60}\n")

    # Unlike download_llama4scout_fp8.py's NVIDIA checkpoint, this one is not
    # gated behind Meta's license terms in the same way, so we don't hard-fail
    # without a token here -- snapshot_download() falls back to huggingface_hub's
    # own cached ~/.cache/huggingface/token when token=None.
    print("Downloading model files from HuggingFace Hub...")
    print("(This is a ~110GB download -- expect it to take a while on first run)\n")

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
        if hasattr(config, "num_experts"):
            print(f"  Num experts:      {config.num_experts}")
        if hasattr(config, "num_experts_per_tok"):
            print(f"  Active experts:   {config.num_experts_per_tok}")
        quant_config = getattr(config, "quantization_config", None)
        if quant_config:
            print(f"  Quantization config found in checkpoint: {quant_config}")
        else:
            print("  NOTE: no quantization_config found on the model config object -- "
                  "check config.json's quantization_config field in the downloaded directory directly.")
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
    limit_mm_images: int,
):
    model_json = {
        "model": weights_path,
        "tokenizer_mode": "auto",
        "dtype": "auto",
        "limit_mm_per_prompt": {"image": limit_mm_images},
        # NO "quantization" field -- compressed-tensors checkpoints carry their
        # own quantization_config in config.json that vLLM auto-detects,
        # unlike NVIDIA's modelopt format which needed an explicit hint (see
        # download_llama4scout_fp8.py's write_model_json for that contrast).
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "disable_log_requests": True,
        "enforce_eager": False,
    }

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
    limit_mm_images: int,
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

    write_model_json(
        target_version_dir,
        weights_path,
        tensor_parallel_size,
        max_model_len,
        gpu_memory_utilization,
        limit_mm_images,
    )

    return target_dir


def main():
    args = parse_args()

    print("=" * 60)
    print("Llama-4-Scout FP8-dynamic (compressed-tensors) Setup for Triton vLLM")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)

    check_dependencies()

    triton_repo = Path(args.base_dir).resolve() if args.base_dir else PROJECT_ROOT / "triton-repo"
    models_dir = triton_repo / "models"
    weights_dir = triton_repo / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_weights_dir = download_model_weights(weights_dir, args.token, args.ignore_ssl)

    if not args.no_verify:
        verify_model(model_weights_dir)

    weights_path = str(triton_repo / "weights" / TRITON_MODEL_NAME)

    setup_triton_model(
        models_dir,
        weights_path,
        args.tensor_parallel_size,
        args.max_model_len,
        args.gpu_memory_utilization,
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
          f"max_model_len={args.max_model_len}")
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
