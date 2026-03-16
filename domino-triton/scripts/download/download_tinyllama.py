#!/usr/bin/env python3
"""
download_tinyllama.py

Downloads TinyLlama-1.1B-Chat model weights for use with NVIDIA Triton Inference Server.

TinyLlama-1.1B-Chat:
- 1.1B parameters (~2GB weights)
- Llama architecture (same as Llama 2)
- Apache 2.0 license (open, no HuggingFace token needed)
- Ideal for PoC and benchmarking vs TensorRT-LLM

Usage:
    python scripts/download_tinyllama.py

Requirements:
    pip install transformers torch

Output:
    triton-repo/weights/tinyllama-python/1/  <- HuggingFace model files

Directory structure after download:
    triton-repo/
    ├── models/tinyllama-python/     <- Triton model config
    │   ├── config.pbtxt
    │   └── 1/model.py
    └── weights/tinyllama-python/    <- Downloaded weights
        └── 1/<HF model files>
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRITON_MODEL_NAME = "tinyllama-python"
MODEL_VERSION = "1"


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        missing.append("transformers")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def download_model_weights(weights_dir: Path):
    """Download TinyLlama model weights to the weights folder."""
    import json
    import shutil
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_weights_dir = weights_dir / TRITON_MODEL_NAME / MODEL_VERSION
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {MODEL_ID}")
    print(f"Destination: {model_weights_dir}")
    print(f"{'='*60}\n")

    # Download tokenizer files directly from HuggingFace Hub
    # This avoids issues with save_pretrained corrupting tokenizer_config.json
    print("Downloading tokenizer files...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for filename in tokenizer_files:
        try:
            cached_path = hf_hub_download(MODEL_ID, filename)
            dest_path = model_weights_dir / filename
            shutil.copy(cached_path, dest_path)
            print(f"  Downloaded {filename}")
        except Exception as e:
            print(f"  Skipping {filename}: {e}")

    # Verify tokenizer config has correct class
    tokenizer_config_path = model_weights_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r") as f:
            config = json.load(f)
        # Ensure tokenizer_class is LlamaTokenizer (not TokenizersBackend)
        if config.get("tokenizer_class") not in ["LlamaTokenizer", "LlamaTokenizerFast"]:
            config["tokenizer_class"] = "LlamaTokenizer"
            with open(tokenizer_config_path, "w") as f:
                json.dump(config, f, indent=2)
            print("  Fixed tokenizer_class to LlamaTokenizer")

    # Verify tokenizer loads correctly
    print("Verifying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_weights_dir)
    print(f"  Tokenizer loaded: {type(tokenizer).__name__}")

    # Download model
    print("\nDownloading model weights (~2GB)...")
    print("(This may take a few minutes on first run)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(model_weights_dir)

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"\nModel downloaded: {param_count:.2f}B parameters")
    print(f"Saved to: {model_weights_dir}")

    return tokenizer, model, model_weights_dir


def verify_model(weights_dir: Path):
    """Verify the model loads and runs correctly."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_weights_dir = weights_dir / TRITON_MODEL_NAME / MODEL_VERSION

    print(f"\n{'='*60}")
    print("Verifying model...")
    print(f"{'='*60}\n")

    # Load from local
    print(f"Loading from: {model_weights_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_weights_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_weights_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    # Test inference
    prompt = "What is the capital of France?"
    print(f"\nTest prompt: '{prompt}'")

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    if prompt in response:
        response = response.split(prompt)[-1].strip()

    print(f"Response: '{response[:150]}...'")
    print("\nVerification: PASSED")


def setup_triton_model(models_dir: Path):
    """
    Set up the Triton Python backend model by copying from source models folder.

    Source: models/<model_name>/  (checked into git)
    Target: triton-repo/models/<model_name>/  (deployment folder)
    """
    import shutil

    # Source directory (triton-repo-reference checked into git)
    source_dir = PROJECT_ROOT / "triton-repo-reference" / "models" / TRITON_MODEL_NAME
    source_config = source_dir / "config.pbtxt"
    source_model_py = source_dir / "1" / "model.py"

    # Target directory (triton-repo)
    target_dir = models_dir / TRITON_MODEL_NAME
    target_version_dir = target_dir / "1"
    target_config = target_dir / "config.pbtxt"
    target_model_py = target_version_dir / "model.py"

    print(f"\n{'='*60}")
    print(f"Setting up Triton model: {TRITON_MODEL_NAME}")
    print(f"{'='*60}\n")

    # Check source files exist
    if not source_config.exists() or not source_model_py.exists():
        print(f"ERROR: Source model files not found!")
        print(f"  Expected: {source_config}")
        print(f"  Expected: {source_model_py}")
        print(f"\n  Make sure triton-repo-reference/ folder is present in your workspace.")
        raise FileNotFoundError(f"Source model files missing for {TRITON_MODEL_NAME}")

    # Create target directories
    target_version_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"Copying model files...")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")

    shutil.copy2(source_config, target_config)
    print(f"  - config.pbtxt: COPIED")

    shutil.copy2(source_model_py, target_model_py)
    print(f"  - {MODEL_VERSION}/model.py: COPIED")

    # Display config summary
    print(f"\nModel configuration:")
    with open(target_config) as f:
        config_content = f.read()
        for line in config_content.split('\n'):
            if 'max_batch_size' in line or 'backend' in line or 'name:' in line:
                print(f"  {line.strip()}")

    return target_dir


def main():
    """Main entry point."""
    print("=" * 60)
    print("TinyLlama-1.1B Setup for Triton")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)

    check_dependencies()

    # Set up paths - triton-repo contains both models/ and weights/
    triton_repo = PROJECT_ROOT / "triton-repo"
    models_dir = triton_repo / "models"
    weights_dir = triton_repo / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Download model weights to weights folder
    download_model_weights(weights_dir)

    # Verify the model loads from weights folder
    verify_model(weights_dir)

    # Set up Triton model (copy config.pbtxt and model.py from models/ folder)
    setup_triton_model(models_dir)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nDirectory structure:")
    print(f"  triton-repo/                          <- Single mount point")
    print(f"  ├── models/{TRITON_MODEL_NAME}/       <- Triton loads from here")
    print(f"  │   ├── config.pbtxt")
    print(f"  │   └── {MODEL_VERSION}/model.py")
    print(f"  └── weights/{TRITON_MODEL_NAME}/      <- Model weights")
    print(f"      └── {MODEL_VERSION}/<HF files>")
    print(f"\nTo test locally:")
    print(f"  docker compose up --build")
    print(f"  python scripts/llm_text_grpc_client.py --model tinyllama-python --prompt 'Hello!'")
    print(f"\nFor TensorRT-LLM comparison, build the engine in a Domino workspace.")
    print(f"See: docs/tensorrt_llm_poc.md")


if __name__ == "__main__":
    main()
