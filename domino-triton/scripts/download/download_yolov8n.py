#!/usr/bin/env python3
"""
download_yolov8n.py

Downloads YOLOv8n model from Ultralytics and exports it to ONNX format
for use with NVIDIA Triton Inference Server.

Usage:
    python scripts/download_yolov8n.py

Requirements:
    pip install ultralytics onnx onnxruntime

Output:
    models/yolov8n/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
"""

import os
import sys
import shutil
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        from ultralytics import YOLO
    except ImportError:
        missing.append("ultralytics")

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


def download_and_export_yolov8n(
    models_dir: Path,
    model_name: str = "yolov8n",
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
):
    """
    Download YOLOv8n and export to ONNX format.

    Args:
        models_dir: Base directory for Triton models
        model_name: Name for the model directory (must match config.pbtxt)
        imgsz: Input image size (default 640x640)
        opset: ONNX opset version
        simplify: Whether to simplify the ONNX model
    """
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"Downloading and exporting {model_name}")
    print(f"{'='*60}\n")

    # Create model directory structure
    model_dir = models_dir / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = version_dir / "model.onnx"

    # Download and load YOLOv8n
    print(f"Loading YOLOv8n from Ultralytics Hub...")
    model = YOLO("yolov8n.pt")

    # Export to ONNX with dynamic batch size
    print(f"Exporting to ONNX (opset={opset}, imgsz={imgsz}, dynamic batch)...")
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=True,  # Dynamic batch size for batching support
    )

    # Move exported file to Triton model directory
    if Path(export_path).exists():
        shutil.move(export_path, onnx_path)
        print(f"Moved ONNX model to: {onnx_path}")

    # Clean up the downloaded .pt file
    pt_file = Path("yolov8n.pt")
    if pt_file.exists():
        pt_file.unlink()
        print("Cleaned up temporary .pt file")

    # Verify the export by loading with ONNX Runtime
    print("\nVerifying ONNX model...")
    verify_onnx_model(onnx_path)

    return onnx_path


def verify_onnx_model(onnx_path: Path):
    """Verify the ONNX model and print input/output shapes."""
    import onnx
    import onnxruntime as ort

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

    return session.get_inputs(), session.get_outputs()


def create_yolov8n_config(models_dir: Path, model_name: str = "yolov8n"):
    """
    Create Triton config.pbtxt for YOLOv8n.

    YOLOv8n with dynamic batch export:
    - Input: images [-1, 3, 640, 640] float32 (batch dim is dynamic)
    - Output: output0 [-1, 84, -1] float32 (batch and anchors are dynamic)
      (84 = 4 bbox coords + 80 class probabilities)
    """
    model_dir = models_dir / model_name
    config_path = model_dir / "config.pbtxt"

    config_content = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 16

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]

output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, -1 ]
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

# Dynamic batching - Triton will automatically batch requests
dynamic_batching {{
  preferred_batch_size: [ 1, 2, 4, 8 ]
  max_queue_delay_microseconds: 100
}}
'''

    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"\nCreated config.pbtxt: {config_path}")
    return config_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("YOLOv8n Model Downloader for Triton Inference Server")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Set up paths - use triton-repo structure
    models_dir = PROJECT_ROOT / "triton-repo" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = "yolov8n"

    # Download and export
    onnx_path = download_and_export_yolov8n(
        models_dir=models_dir,
        model_name=model_name,
        imgsz=640,
        opset=17,
    )

    # Create Triton config
    create_yolov8n_config(models_dir, model_name)

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
    print(f"      └── model.onnx")


if __name__ == "__main__":
    main()