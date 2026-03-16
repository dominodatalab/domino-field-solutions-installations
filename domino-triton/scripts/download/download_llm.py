#!/usr/bin/env python3
"""
download_llm.py

Downloads SmolLM-135M-Instruct model weights to the weights/ folder for use with
NVIDIA Triton Inference Server Python backend.

SmolLM-135M-Instruct is the smallest capable instruction-tuned LLM:
- 135M parameters (~270MB)
- Instruction-tuned for conversations
- Fast inference on CPU

Usage:
    python scripts/download_llm.py

Requirements:
    pip install transformers torch

Output:
    weights/smollm-135m-python/1/  <- HuggingFace model files downloaded here

The Python backend model (models/smollm-135m-python/1/model.py) loads weights
from the weights/ folder at runtime.

Directory structure:
    <base>/
    ├── models/                    # Triton model repository
    │   └── smollm-135m-python/
    │       ├── config.pbtxt
    │       └── 1/model.py
    └── weights/                   # Downloaded model binaries
        └── smollm-135m-python/
            └── 1/<HF model files>
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
TRITON_MODEL_NAME = "smollm-135m-python"
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
    """
    Download model weights to the weights folder.
    """
    import json
    import shutil
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Create weights directory
    model_weights_dir = weights_dir / TRITON_MODEL_NAME / MODEL_VERSION
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading LLM model: {MODEL_ID}")
    print(f"Destination: {model_weights_dir}")
    print(f"{'='*60}\n")

    # Download all tokenizer files directly from HuggingFace Hub
    # SmolLM uses GPT2-style tokenizer which needs vocab.json and merges.txt
    print("Downloading tokenizer files...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
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

    # Fix tokenizer config for compatibility - ensure GPT2Tokenizer is used
    tokenizer_config_path = model_weights_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r") as f:
            config = json.load(f)
        # Ensure tokenizer_class is set to GPT2Tokenizer for compatibility
        if config.get("tokenizer_class") != "GPT2Tokenizer":
            config["tokenizer_class"] = "GPT2Tokenizer"
            with open(tokenizer_config_path, "w") as f:
                json.dump(config, f, indent=2)
            print("  Set tokenizer_class to GPT2Tokenizer for compatibility")

    # Verify tokenizer loads correctly
    print("Verifying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_weights_dir)
    print(f"  Tokenizer loaded: {type(tokenizer).__name__}")

    # Download and save model
    print("Downloading model weights...")
    print("(This may take a while on first run)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(model_weights_dir)

    print(f"\nModel downloaded successfully to: {model_weights_dir}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    return tokenizer, model, model_weights_dir


def verify_model(weights_dir: Path):
    """Verify the model loads correctly from the weights folder."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_weights_dir = weights_dir / TRITON_MODEL_NAME / MODEL_VERSION

    print("\n" + "="*60)
    print("Verifying model from weights folder...")
    print("="*60 + "\n")

    # Load from local weights
    print(f"Loading from: {model_weights_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_weights_dir)
    model = AutoModelForCausalLM.from_pretrained(model_weights_dir)

    # Test prompt
    prompt = "What is the capital of France?"
    print(f"Test prompt: '{prompt}'")

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize and generate
    inputs = tokenizer(formatted, return_tensors="pt")
    print("Running test inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    print(f"Test response: '{response[:100]}...'")
    print("\nModel verification: PASSED")


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
    print("LLM Setup for Triton Inference Server")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Set up paths - triton-repo contains both models/ and weights/
    triton_repo = PROJECT_ROOT / "triton-repo"
    models_dir = triton_repo / "models"
    weights_dir = triton_repo / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Download model weights to weights folder
    tokenizer, model, model_weights_dir = download_model_weights(weights_dir)

    # Verify the model loads from weights folder
    verify_model(weights_dir)

    # Verify Triton model config exists
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
    print(f"\nWeights downloaded to: {model_weights_dir}")
    print(f"\nTo start Triton server:")
    print(f"  docker compose up --build")
    print(f"\nTo test text generation:")
    print(f"  python scripts/llm_text_grpc_client.py --prompt 'What is AI?'")


if __name__ == "__main__":
    main()