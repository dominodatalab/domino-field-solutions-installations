#!/usr/bin/env python3
"""
download_yolov26n.py

Downloads YOLOv26n model from Ultralytics and exports it to ONNX format
for use with NVIDIA Triton Inference Server.

Usage:
    python scripts/download/download_yolov26n.py

Requirements:
    pip install ultralytics onnx onnxruntime

Output:
    triton-repo/models/yolov26n/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"


def check_dependencies():
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


def download_weights(dest: Path) -> Path:
    """Download yolo26n.pt from the Ultralytics assets release."""
    pt_path = dest / "yolo26n.pt"
    if pt_path.exists():
        print(f"Weights already present at {pt_path}, skipping download.")
        return pt_path

    print(f"Downloading weights from {WEIGHTS_URL} ...")

    def _progress(block_count, block_size, total_size):
        if total_size > 0:
            pct = min(block_count * block_size / total_size * 100, 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(WEIGHTS_URL, pt_path, reporthook=_progress)
    print()  # newline after progress
    print(f"Saved to {pt_path}")
    return pt_path


def download_and_export_yolov26n(
    models_dir: Path,
    model_name: str = "yolov26n",
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
):
    """
    Download YOLOv26n weights and export to ONNX format.

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

    model_dir = models_dir / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = version_dir / "model.onnx"

    pt_path = download_weights(PROJECT_ROOT)
    print(f"\nLoading {pt_path.name} ...")
    model = YOLO(str(pt_path))

    print(f"Exporting to ONNX (opset={opset}, imgsz={imgsz}, dynamic batch)...")
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=True,
    )

    if Path(export_path).exists():
        shutil.move(export_path, onnx_path)
        print(f"Moved ONNX model to: {onnx_path}")

    if pt_path.exists():
        pt_path.unlink()
        print("Cleaned up temporary .pt file")

    print("\nVerifying ONNX model...")
    inputs, outputs = verify_onnx_model(onnx_path)

    return onnx_path, inputs, outputs


def verify_onnx_model(onnx_path: Path):
    """Verify the ONNX model and print input/output shapes."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("ONNX model validation: PASSED")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print("\nModel Inputs:")
    for inp in session.get_inputs():
        print(f"  - {inp.name}: shape={inp.shape}, dtype={inp.type}")

    print("\nModel Outputs:")
    for out in session.get_outputs():
        print(f"  - {out.name}: shape={out.shape}, dtype={out.type}")

    return session.get_inputs(), session.get_outputs()


def create_yolov26n_config(models_dir: Path, model_name: str = "yolov26n"):
    """
    Create Triton config.pbtxt for YOLOv26n.

    YOLOv26n with dynamic batch export:
    - Input: images [-1, 3, 640, 640] float32 (batch dim is dynamic)
    - Output: output0 [-1, 84, -1] float32 (batch and anchors are dynamic)
      (84 = 4 bbox coords + 80 COCO class probabilities)

    If the exported model has a different number of output channels (e.g., more classes),
    update the dims field in the output section accordingly.
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

# Model type for dashboard UI (comma-separated: text, image, video, audio, text-llm)
parameters {{
  key: "model_type"
  value: {{ string_value: "video,image" }}
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
    print("=" * 60)
    print("YOLOv26n Model Downloader for Triton Inference Server")
    print("=" * 60)

    check_dependencies()

    models_dir = PROJECT_ROOT / "triton-repo" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = "yolov26n"

    onnx_path, inputs, outputs = download_and_export_yolov26n(
        models_dir=models_dir,
        model_name=model_name,
        imgsz=640,
        opset=17,
    )

    # Warn if output channels differ from the 84 expected for COCO-80
    for out in outputs:
        shape = out.shape
        if len(shape) >= 2 and isinstance(shape[1], int) and shape[1] != 84:
            print(f"\nWARNING: Output channel count is {shape[1]}, not 84.")
            print("Update the 'dims' field in config.pbtxt to match the actual shape.")

    create_yolov26n_config(models_dir, model_name)

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
