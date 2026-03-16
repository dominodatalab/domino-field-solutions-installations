#!/usr/bin/env python3
"""
Copy a model from one Triton repository to another.

Copies the model folder from {source}/models/{model_name} to {target}/models/{model_name}.
Optionally copies associated weights from {source}/weights/{model_name} if they exist.
Optionally modifies instance_group in config.pbtxt for target hardware.

Usage:
    python scripts/model_management/copy_model.py --target /path/to/target-triton-repo bert-base-uncased
    python scripts/model_management/copy_model.py --source ./triton-repo --target /mnt/edv/triton-repo yolov8n
    python scripts/model_management/copy_model.py --target /mnt/edv/triton-repo --include-weights smollm-135m-python
    python scripts/model_management/copy_model.py --source ./triton-repo --target /mnt/edv/triton-repo --all
    python scripts/model_management/copy_model.py --source ./triton-repo --target /mnt/edv/triton-repo --all --instance-kind KIND_GPU
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def _copytree_s3_safe(src: Path, dst: Path) -> None:
    """
    Copy directory tree with S3 fuse compatibility.

    S3 fuse doesn't support:
    - Overwriting existing files
    - Setting file metadata (timestamps, permissions)

    So we remove each destination file first and use copyfile (content only).
    """
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        src_path = item
        dst_path = dst / item.name

        if src_path.is_dir():
            _copytree_s3_safe(src_path, dst_path)
        else:
            # Remove destination file first (S3 fuse can't overwrite)
            if dst_path.exists():
                try:
                    dst_path.unlink()
                except OSError:
                    pass  # Ignore errors, copy will fail if file still exists
            # Use copyfile (content only) - S3 fuse doesn't support metadata
            shutil.copyfile(src_path, dst_path)


def update_instance_group(config_path: Path, kind: str, count: int, device: int = None) -> None:
    """
    Update or add instance_group in config.pbtxt.

    Args:
        config_path: Path to config.pbtxt
        kind: KIND_GPU or KIND_CPU
        count: Number of instances
        device: GPU device index (only used when kind=KIND_GPU)
    """
    with open(config_path, "r") as f:
        content = f.read()

    # Build the new instance_group block
    if kind == "KIND_GPU" and device is not None:
        new_instance_group = f"""instance_group [
  {{
    kind: {kind}
    count: {count}
    gpus: [{device}]
  }}
]"""
        device_info = f", gpus=[{device}]"
    else:
        new_instance_group = f"""instance_group [
  {{
    kind: {kind}
    count: {count}
  }}
]"""
        device_info = ""

    # Pattern to match existing instance_group block (handles nested braces)
    pattern = r'instance_group\s*\[[\s\S]*?\n\]'

    if re.search(pattern, content):
        # Replace existing instance_group
        content = re.sub(pattern, new_instance_group, content)
        print(f"  Updated instance_group: kind={kind}, count={count}{device_info}")
    else:
        # Append instance_group at the end
        content = content.rstrip() + "\n\n" + new_instance_group + "\n"
        print(f"  Added instance_group: kind={kind}, count={count}{device_info}")

    with open(config_path, "w") as f:
        f.write(content)


def copy_model(
    source_root: Path,
    target_root: Path,
    model_name: str,
    include_weights: bool = False,
    overwrite: bool = False,
    instance_kind: str = None,
    instance_count: int = None,
    device: int = None,
) -> bool:
    """
    Copy a model from source to target Triton repository.

    Args:
        source_root: Source triton-repo root directory
        target_root: Target triton-repo root directory
        model_name: Name of the model to copy
        include_weights: Also copy weights if they exist
        overwrite: Overwrite target if it exists
        instance_kind: KIND_GPU or KIND_CPU (optional)
        instance_count: Number of instances (optional)
        device: GPU device index (optional, default 0 for KIND_GPU)

    Returns:
        True if successful, False otherwise
    """
    source_model = source_root / "models" / model_name
    target_model = target_root / "models" / model_name

    # Validate source exists
    if not source_model.exists():
        print(f"Error: Source model not found: {source_model}")
        return False

    if not source_model.is_dir():
        print(f"Error: Source is not a directory: {source_model}")
        return False

    # Check if target exists
    if target_model.exists():
        if overwrite:
            print(f"Removing existing target: {target_model}")
            shutil.rmtree(target_model, ignore_errors=True)
            # Double-check removal succeeded (S3 fuse can be inconsistent)
            if target_model.exists():
                print(f"Warning: Could not fully remove {target_model}, proceeding anyway")
        else:
            print(f"Error: Target already exists: {target_model}")
            print("Use --overwrite to replace it")
            return False

    # Create target models directory if needed
    target_models_dir = target_root / "models"
    target_models_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    print(f"Copying model: {source_model} -> {target_model}")
    _copytree_s3_safe(source_model, target_model)
    print(f"  Copied model directory ({_get_dir_size(target_model)})")

    # Update instance_group if specified
    if instance_kind is not None and instance_count is not None:
        config_path = target_model / "config.pbtxt"
        if config_path.exists():
            # Default device to 0 for KIND_GPU if not specified
            gpu_device = device if device is not None else (0 if instance_kind == "KIND_GPU" else None)
            update_instance_group(config_path, instance_kind, instance_count, gpu_device)
        else:
            print(f"  Warning: config.pbtxt not found, skipping instance_group update")

    # Optionally copy weights
    if include_weights:
        source_weights = source_root / "weights" / model_name
        target_weights = target_root / "weights" / model_name

        if source_weights.exists():
            if target_weights.exists():
                if overwrite:
                    print(f"Removing existing weights: {target_weights}")
                    shutil.rmtree(target_weights, ignore_errors=True)
                else:
                    print(f"Warning: Target weights already exist: {target_weights}")
                    print("  Skipping weights copy (use --overwrite to replace)")
                    return True

            target_weights_dir = target_root / "weights"
            target_weights_dir.mkdir(parents=True, exist_ok=True)

            print(f"Copying weights: {source_weights} -> {target_weights}")
            _copytree_s3_safe(source_weights, target_weights)
            print(f"  Copied weights directory ({_get_dir_size(target_weights)})")
        else:
            print(f"  No weights directory found at: {source_weights}")

    return True


def _get_dir_size(path: Path) -> str:
    """Get human-readable directory size."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def list_models(triton_root: Path) -> list[str]:
    """List available models in a triton repository."""
    models_dir = triton_root / "models"
    if not models_dir.exists():
        return []
    return sorted([d.name for d in models_dir.iterdir() if d.is_dir()])


def main():
    parser = argparse.ArgumentParser(
        description="Copy a model from one Triton repository to another",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy all models (with weights) to target EDV
  python copy_model.py --source ./triton-repo --target /mnt/edv/triton-repo --all

  # Copy single model to target EDV
  python copy_model.py --target /mnt/edv/triton-repo bert-base-uncased

  # Copy with weights
  python copy_model.py --target /mnt/edv/triton-repo --include-weights smollm-135m-python

  # Overwrite existing
  python copy_model.py --target /mnt/edv/triton-repo --overwrite --all

  # Copy all models and configure for GPU (default: 1 instance)
  python copy_model.py --source ./triton-repo --target /mnt/edv/triton-repo --all --instance-kind KIND_GPU

  # Copy single model and configure for CPU with 2 instances
  python copy_model.py --target /mnt/edv/triton-repo --instance-kind KIND_CPU --instance-count 2 bert-base-uncased

  # List available models
  python copy_model.py --list
  python copy_model.py --source /other/triton-repo --list
""",
    )

    parser.add_argument(
        "model_name",
        nargs="?",
        help="Name of the model to copy",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=Path("./triton-repo"),
        help="Source triton-repo root (default: ./triton-repo)",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=Path,
        help="Target triton-repo root (required unless --list)",
    )
    parser.add_argument(
        "--include-weights",
        "-w",
        action="store_true",
        help="Also copy weights directory if it exists",
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Overwrite target if it already exists",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models in source repository",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Copy all models from source to target (includes weights)",
    )
    parser.add_argument(
        "--instance-kind",
        choices=["KIND_GPU", "KIND_CPU"],
        help="Set instance_group kind in config.pbtxt (KIND_GPU or KIND_CPU)",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        help="Set instance_group count in config.pbtxt (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=int,
        help="GPU device index for KIND_GPU (default: 0)",
    )

    args = parser.parse_args()

    # Default instance_count to 1 if instance_kind is provided
    if args.instance_kind is not None and args.instance_count is None:
        args.instance_count = 1

    # Resolve source path
    source_root = args.source.resolve()

    # Handle --list
    if args.list:
        if not source_root.exists():
            print(f"Error: Source directory not found: {source_root}")
            sys.exit(1)

        models = list_models(source_root)
        if models:
            print(f"Available models in {source_root}:")
            for model in models:
                model_path = source_root / "models" / model
                weights_path = source_root / "weights" / model
                has_weights = weights_path.exists()
                size = _get_dir_size(model_path)
                weights_info = " [+weights]" if has_weights else ""
                print(f"  {model} ({size}){weights_info}")
        else:
            print(f"No models found in {source_root}/models/")
        sys.exit(0)

    # Handle --all
    if args.all:
        if not args.target:
            parser.error("--target is required with --all")

        if not source_root.exists():
            print(f"Error: Source directory not found: {source_root}")
            sys.exit(1)

        target_root = args.target.resolve()
        models = list_models(source_root)

        if not models:
            print(f"No models found in {source_root}/models/")
            sys.exit(1)

        print(f"Copying all {len(models)} models from {source_root} to {target_root}")
        print(f"  (with weights included)\n")

        success_count = 0
        fail_count = 0

        for model in models:
            print(f"\n{'='*60}")
            success = copy_model(
                source_root=source_root,
                target_root=target_root,
                model_name=model,
                include_weights=True,  # Always include weights with --all
                overwrite=args.overwrite,
                instance_kind=args.instance_kind,
                instance_count=args.instance_count,
                device=args.device,
            )
            if success:
                success_count += 1
            else:
                fail_count += 1

        print(f"\n{'='*60}")
        print(f"SUMMARY: {success_count} succeeded, {fail_count} failed")
        print(f"  Source: {source_root}")
        print(f"  Target: {target_root}")

        if fail_count > 0:
            sys.exit(1)
        sys.exit(0)

    # Validate arguments for single model copy
    if not args.model_name:
        parser.error("model_name is required (or use --all to copy all models)")

    if not args.target:
        parser.error("--target is required")

    if not source_root.exists():
        print(f"Error: Source directory not found: {source_root}")
        sys.exit(1)

    target_root = args.target.resolve()

    # Perform copy
    success = copy_model(
        source_root=source_root,
        target_root=target_root,
        model_name=args.model_name,
        include_weights=args.include_weights,
        overwrite=args.overwrite,
        instance_kind=args.instance_kind,
        instance_count=args.instance_count,
        device=args.device,
    )

    if success:
        print(f"\nModel '{args.model_name}' copied successfully!")
        print(f"  Source: {source_root}")
        print(f"  Target: {target_root}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
