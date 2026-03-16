#!/usr/bin/env python3
"""
Test models sequentially - one at a time to avoid GPU memory issues.

This script:
1. Sets up environment variables for the Triton proxy
2. For each model:
   - Copies model to EDV with --overwrite --instance-kind KIND_GPU
   - Loads the model
   - Runs the client test
   - Runs the benchmark (optional)
   - Unloads the model
3. Moves to the next model

Usage:
    python scripts/testing/test_models_sequential.py
    python scripts/testing/test_models_sequential.py --models bert-base-uncased yolov8n
    python scripts/testing/test_models_sequential.py --skip-benchmark
    python scripts/testing/test_models_sequential.py --skip-copy

Environment Variables (set before running or edit CONFIGURATION section):
    TRITON_REST_URL - REST proxy URL
    TRITON_GRPC_URL - gRPC proxy URL
    DOMINO_USER_API_KEY - API key for authentication
    SOURCE_TRITON_ROOT - Local triton-repo path
    TARGET_TRITON_ROOT - EDV triton-repo path
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Determine project root and change to it (so relative paths work)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
os.chdir(PROJECT_ROOT)

# =============================================================================
# CONFIGURATION - Edit these for your environment
# =============================================================================

# Namespace where Triton is deployed
NAMESPACE = "domino-inference-dev"

# Default environment variables (can be overridden by actual env vars)
DEFAULT_CONFIG = {
    "TRITON_REST_URL": f"http://triton-inference-server-proxy.{NAMESPACE}.svc.cluster.local:8080",
    "TRITON_GRPC_URL": f"triton-inference-server-proxy.{NAMESPACE}.svc.cluster.local:50051",
    "SOURCE_TRITON_ROOT": "/mnt/triton-repo",
    "TARGET_TRITON_ROOT": "/domino/edv/domino-inference-dev-triton-repo-pvc",
}

# Models to test and their client/benchmark scripts
# Format: model_name -> {client: script, benchmark: script, args: extra args}
MODEL_TESTS = {
    "bert-base-uncased": {
        "client": "scripts/clients/bert_text_grpc_client.py",
        "benchmark": "scripts/benchmarks/benchmark_bert_clients.py",
        "client_args": ["--texts", "This is a test sentence for BERT classification."],
    },
    "yolov8n": {
        "client": "scripts/clients/yolov8n_video_grpc_client.py",
        "benchmark": "scripts/benchmarks/benchmark_yolov8_clients.py",
        "client_args": ["--video", "samples/video.avi", "--max-frames", "120"],
        "benchmark_args": ["--video", "samples/video.avi", "--max-frames", "120"],
    },
    "whisper-tiny-python": {
        "client": "scripts/clients/whisper_audio_grpc_client.py",
        "benchmark": "scripts/benchmarks/benchmark_whisper_clients.py",
        "client_args": ["--audio", "samples/audio_sample.wav"],
        "benchmark_args": ["--audio", "samples/audio_sample.wav"],
    },
    "smollm-135m-python": {
        "client": "scripts/clients/llm_text_grpc_client.py",
        "benchmark": "scripts/benchmarks/benchmark_llm_clients.py",
        "client_args": ["--prompt", "What is the capital of France?", "--max-tokens", "50"],
    },
    "tinyllama-python": {
        "client": "scripts/clients/llm_text_grpc_client.py",
        "benchmark": "scripts/benchmarks/benchmark_llm_clients.py",
        "client_args": ["--model", "tinyllama-python", "--prompt", "What is AI?", "--max-tokens", "50"],
        "benchmark_args": ["--model", "tinyllama-python"],
    },
    # tinyllama-trtllm excluded - requires TensorRT-LLM backend not available in standard deployment
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_environment():
    """Set up environment variables."""
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in os.environ:
            os.environ[key] = default_value
            print(f"  {key}={default_value} (default)")
        else:
            print(f"  {key}={os.environ[key]} (from env)")


def run_command(cmd: list, description: str, timeout: int = 300) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=False,  # Show output in real-time
        )
        if result.returncode == 0:
            print(f"\n[OK] {description}")
            return True
        else:
            print(f"\n[FAILED] {description} (exit code: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {description} (exceeded {timeout}s)")
        return False
    except Exception as e:
        print(f"\n[ERROR] {description}: {e}")
        return False


def copy_model(model_name: str, source_root: str, target_root: str, use_cpu: bool = False, device: int = 0) -> bool:
    """Copy model to EDV with instance configuration."""
    instance_kind = "KIND_CPU" if use_cpu else "KIND_GPU"
    cmd = [
        sys.executable,
        "scripts/model_management/copy_model.py",
        "--source", source_root,
        "--target", target_root,
        "--instance-kind", instance_kind,
        "--overwrite",
        "--include-weights",
    ]
    if not use_cpu:
        cmd.extend(["--device", str(device)])
    cmd.append(model_name)
    device_info = "" if use_cpu else f", device={device}"
    return run_command(cmd, f"Copying {model_name} to EDV ({instance_kind}{device_info})", timeout=600)


def load_model(model_name: str) -> bool:
    """Load a model on Triton."""
    cmd = [
        sys.executable,
        "scripts/model_management/load_model.py",
        model_name,
    ]
    return run_command(cmd, f"Loading {model_name}", timeout=300)


def unload_model(model_name: str) -> bool:
    """Unload a model from Triton."""
    cmd = [
        sys.executable,
        "scripts/model_management/unload_model.py",
        model_name,
    ]
    return run_command(cmd, f"Unloading {model_name}", timeout=60)


def run_client_test(model_name: str, config: dict) -> bool:
    """Run the client test for a model."""
    client_script = config.get("client")
    if not client_script or not Path(client_script).exists():
        print(f"[SKIP] No client script for {model_name}")
        return True

    cmd = [sys.executable, client_script] + config.get("client_args", [])
    return run_command(cmd, f"Testing {model_name} client", timeout=180)


def run_benchmark(model_name: str, config: dict) -> bool:
    """Run the benchmark for a model."""
    benchmark_script = config.get("benchmark")
    if not benchmark_script or not Path(benchmark_script).exists():
        print(f"[SKIP] No benchmark script for {model_name}")
        return True

    cmd = [sys.executable, benchmark_script] + config.get("benchmark_args", [])
    return run_command(cmd, f"Benchmarking {model_name}", timeout=600)


def get_model_status() -> None:
    """Print current model status."""
    cmd = [sys.executable, "scripts/model_management/model_status.py"]
    run_command(cmd, "Current model status", timeout=30)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test models sequentially to avoid GPU memory issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models
  python scripts/testing/test_models_sequential.py

  # Test specific models
  python scripts/testing/test_models_sequential.py --models bert-base-uncased yolov8n

  # Skip benchmarks (faster)
  python scripts/testing/test_models_sequential.py --skip-benchmark

  # Skip copy step (models already on EDV)
  python scripts/testing/test_models_sequential.py --skip-copy

  # List available models
  python scripts/testing/test_models_sequential.py --list
""",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (default: all)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use KIND_CPU instead of KIND_GPU (slower but avoids GPU memory issues)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index for KIND_GPU (default: 0)",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copying models to EDV (assume already there)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip running benchmarks",
    )
    parser.add_argument(
        "--skip-unload",
        action="store_true",
        help="Don't unload models after testing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # List models
    if args.list:
        print("Available models:")
        for model_name, config in MODEL_TESTS.items():
            client = config.get("client", "N/A")
            benchmark = config.get("benchmark", "N/A")
            print(f"  {model_name}")
            print(f"    Client:    {client}")
            print(f"    Benchmark: {benchmark}")
        return

    # Determine which models to test
    models_to_test = args.models if args.models else list(MODEL_TESTS.keys())

    # Validate models
    invalid_models = [m for m in models_to_test if m not in MODEL_TESTS]
    if invalid_models:
        print(f"Error: Unknown models: {invalid_models}")
        print(f"Available: {list(MODEL_TESTS.keys())}")
        sys.exit(1)

    # Setup
    print("\n" + "="*60)
    print("SEQUENTIAL MODEL TESTING")
    print("="*60)
    print(f"\nModels to test: {models_to_test}")
    if args.cpu:
        print(f"Instance kind: KIND_CPU")
    else:
        print(f"Instance kind: KIND_GPU (device={args.device})")
    print(f"Skip copy: {args.skip_copy}")
    print(f"Skip benchmark: {args.skip_benchmark}")

    print("\nSetting up environment variables:")
    setup_environment()

    source_root = os.environ["SOURCE_TRITON_ROOT"]
    target_root = os.environ["TARGET_TRITON_ROOT"]

    # Show initial status
    get_model_status()

    # Results tracking
    results = {}

    # Test each model
    for model_name in models_to_test:
        print("\n" + "#"*60)
        print(f"# TESTING: {model_name}")
        print("#"*60)

        config = MODEL_TESTS[model_name]
        model_results = {"copy": None, "load": None, "client": None, "benchmark": None, "unload": None}

        # Step 1: Copy model
        if not args.skip_copy:
            model_results["copy"] = copy_model(model_name, source_root, target_root, use_cpu=args.cpu, device=args.device)
            if not model_results["copy"]:
                print(f"[ABORT] Failed to copy {model_name}, skipping...")
                results[model_name] = model_results
                continue
        else:
            print(f"\n[SKIP] Copy step for {model_name}")
            model_results["copy"] = "skipped"

        # Step 2: Load model
        model_results["load"] = load_model(model_name)
        if not model_results["load"]:
            print(f"[ABORT] Failed to load {model_name}, skipping...")
            results[model_name] = model_results
            continue

        # Give the model a moment to fully initialize
        time.sleep(2)

        # Step 3: Run client test
        model_results["client"] = run_client_test(model_name, config)

        # Step 4: Run benchmark
        if not args.skip_benchmark:
            model_results["benchmark"] = run_benchmark(model_name, config)
        else:
            print(f"\n[SKIP] Benchmark for {model_name}")
            model_results["benchmark"] = "skipped"

        # Step 5: Unload model
        if not args.skip_unload:
            model_results["unload"] = unload_model(model_name)
            # Wait for unload to complete and GPU memory to free
            time.sleep(3)
        else:
            print(f"\n[SKIP] Unload for {model_name}")
            model_results["unload"] = "skipped"

        results[model_name] = model_results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Model':<25} {'Copy':<10} {'Load':<10} {'Client':<10} {'Bench':<10} {'Unload':<10}")
    print("-"*75)

    for model_name, model_results in results.items():
        def status(val):
            if val is True:
                return "OK"
            elif val is False:
                return "FAILED"
            elif val == "skipped":
                return "SKIP"
            else:
                return "-"

        print(f"{model_name:<25} "
              f"{status(model_results['copy']):<10} "
              f"{status(model_results['load']):<10} "
              f"{status(model_results['client']):<10} "
              f"{status(model_results['benchmark']):<10} "
              f"{status(model_results['unload']):<10}")

    # Final status
    get_model_status()

    # Exit code
    all_passed = all(
        r.get("client") is True or r.get("client") == "skipped"
        for r in results.values()
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
