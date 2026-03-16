#!/usr/bin/env python3
"""
download_bert.py

Downloads BERT model from HuggingFace and exports it to ONNX format
for use with NVIDIA Triton Inference Server.

Usage:
    python scripts/download_bert.py

Requirements:
    pip install transformers torch onnx onnxruntime

Output:
    models/bert-base-uncased/
    ├── config.pbtxt
    └── 1/
        ├── model.onnx
        ├── tokenizer.json
        ├── vocab.txt
        └── ... (tokenizer files)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        missing.append("transformers")

    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def download_and_export_bert(
    models_dir: Path,
    model_name: str = "bert-base-uncased",
    triton_model_name: str = None,
    num_labels: int = 2,
    opset: int = 14,  # Use opset 14 for compatibility with older Triton versions (23.x)
    max_seq_length: int = 128,
):
    """
    Download BERT and export to ONNX format.

    Args:
        models_dir: Base directory for Triton models
        model_name: HuggingFace model name
        triton_model_name: Name for Triton model directory (defaults to model_name)
        num_labels: Number of classification labels
        opset: ONNX opset version
        max_seq_length: Maximum sequence length for the model
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if triton_model_name is None:
        triton_model_name = model_name

    print(f"\n{'='*60}")
    print(f"Downloading and exporting {model_name}")
    print(f"{'='*60}\n")

    # Create model directory structure
    model_dir = models_dir / triton_model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = version_dir / "model.onnx"

    # Download model and tokenizer
    print(f"Downloading {model_name} from HuggingFace...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()

    # Create dummy inputs for export
    print(f"Creating dummy inputs (max_seq_length={max_seq_length})...")
    dummy_text = "This is a sample text for model export."
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Export to ONNX
    print(f"Exporting to ONNX (opset={opset})...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    print(f"Exported ONNX model: {onnx_path}")

    # Save tokenizer for inference
    print(f"Saving tokenizer...")
    tokenizer.save_pretrained(str(version_dir))

    # Verify the export
    print("\nVerifying ONNX model...")
    verify_onnx_model(onnx_path, tokenizer)

    return onnx_path, tokenizer


def verify_onnx_model(onnx_path: Path, tokenizer):
    """Verify the ONNX model and print input/output shapes."""
    import onnx
    import onnxruntime as ort
    import numpy as np

    # Load and check model
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"ONNX model validation: PASSED")

    # Get input/output info using ONNX Runtime
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print("\nModel Inputs:")
    for inp in session.get_inputs():
        print(f"  - {inp.name}: shape={inp.shape}, dtype={inp.type}")

    print("\nModel Outputs:")
    for out in session.get_outputs():
        print(f"  - {out.name}: shape={out.shape}, dtype={out.type}")

    # Test inference
    print("\nTesting inference...")
    test_text = "This movie was great!"
    inputs = tokenizer(
        test_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )

    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )

    logits = outputs[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    print(f"  Input: '{test_text}'")
    print(f"  Logits: {logits}")
    print(f"  Probabilities: {probs}")
    print(f"  Prediction: {'Positive' if np.argmax(probs) == 1 else 'Negative'}")

    return session.get_inputs(), session.get_outputs()


def create_bert_config(
    models_dir: Path,
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
):
    """
    Create Triton config.pbtxt for BERT.

    BERT for sequence classification:
    - Input: input_ids [-1, -1] int64 (dynamic batch, dynamic sequence)
    - Input: attention_mask [-1, -1] int64
    - Output: logits [1, num_labels] float32 (static batch=1 from ONNX export)

    Note: The output shape is [1, num_labels] because torch.onnx.export creates
    a model with static batch size based on the dummy input. For dynamic batching,
    you would need to re-export with proper dynamic_axes configuration.
    """
    model_dir = models_dir / model_name
    config_path = model_dir / "config.pbtxt"

    config_content = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, {num_labels} ]
  }}
]

instance_group [
  {{
    kind: KIND_CPU
    count: 1
  }}
]

# For GPU deployment, replace instance_group with:
# instance_group [
#   {{
#     kind: KIND_GPU
#     count: 1
#   }}
# ]

version_policy {{
  all {{ }}
}}

# Dynamic batching (recommended for NLP models)
# dynamic_batching {{
#   preferred_batch_size: [ 1, 2, 4, 8, 16 ]
#   max_queue_delay_microseconds: 100
# }}

# Optimization settings for ONNX Runtime
# optimization {{
#   graph {{
#     level: 1
#   }}
# }}
'''

    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"\nCreated config.pbtxt: {config_path}")
    return config_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("BERT Model Downloader for Triton Inference Server")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Set up paths - use triton-repo structure
    models_dir = PROJECT_ROOT / "triton-repo" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = "bert-base-uncased"
    num_labels = 2  # Binary classification (positive/negative)

    # Download and export
    # Using opset=14 for compatibility with Triton 23.x (IR version 8)
    # For Triton 24.x+, you can use opset=17 or higher
    onnx_path, tokenizer = download_and_export_bert(
        models_dir=models_dir,
        model_name=model_name,
        num_labels=num_labels,
        opset=14,
        max_seq_length=128,
    )

    # Create Triton config
    create_bert_config(models_dir, model_name, num_labels)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nModel directory: {models_dir / model_name}")
    print(f"ONNX model: {onnx_path}")
    print(f"\nTo test with Triton:")
    print(f"  docker compose up --build")
    print(f"\nModel structure:")
    print(f"  {model_name}/")
    print(f"  ├── config.pbtxt")
    print(f"  └── 1/")
    print(f"      ├── model.onnx")
    print(f"      ├── tokenizer.json")
    print(f"      ├── vocab.txt")
    print(f"      └── ... (tokenizer files)")
    print(f"\nExample REST client usage:")
    print(f"  export MODEL_NAME=bert-base-uncased")
    print(f"  python src/rest_client.py")


if __name__ == "__main__":
    main()