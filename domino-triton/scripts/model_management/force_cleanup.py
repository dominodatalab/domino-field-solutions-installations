#!/usr/bin/env python3
"""
Force cleanup of GPU memory by unloading all models and clearing CUDA cache.

This script attempts to free GPU memory when normal unload doesn't work,
particularly for Python backend models that may have crashed during loading.

Usage:
    python scripts/model_management/force_cleanup.py
    python scripts/model_management/force_cleanup.py --kill-gpu-processes

Note: --kill-gpu-processes requires running inside the Triton pod or with
kubectl exec access.
"""

import argparse
import os
import subprocess
import sys
import requests


def get_rest_url() -> str:
    """Get REST URL from environment or default."""
    return os.environ.get("TRITON_REST_URL", "http://localhost:8080")


def get_api_key() -> str:
    """Get API key from environment."""
    return os.environ.get("DOMINO_USER_API_KEY", "")


def unload_all_models(rest_url: str, api_key: str) -> bool:
    """Unload all models via the API."""
    headers = {"X-Domino-Api-Key": api_key} if api_key else {}

    print("Step 1: Unloading all models via API...")
    try:
        resp = requests.post(
            f"{rest_url}/v1/models/unload-all",
            headers=headers,
            timeout=60
        )
        if resp.status_code == 200:
            print("  All models unloaded via API")
            return True
        else:
            print(f"  Warning: Unload returned {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def get_loaded_models(rest_url: str, api_key: str) -> list:
    """Get list of currently loaded models."""
    headers = {"X-Domino-Api-Key": api_key} if api_key else {}
    try:
        resp = requests.get(f"{rest_url}/v1/models/status", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", []) if m.get("loaded")]
        return []
    except Exception:
        return []


def force_unload_model(rest_url: str, api_key: str, model_name: str) -> bool:
    """Force unload a specific model using Triton's repository API."""
    headers = {
        "X-Domino-Api-Key": api_key,
        "Content-Type": "application/json"
    } if api_key else {"Content-Type": "application/json"}

    print(f"  Force unloading: {model_name}")
    try:
        # Try the v2 repository unload endpoint
        resp = requests.post(
            f"{rest_url}/v2/repository/models/{model_name}/unload",
            headers=headers,
            json={"parameters": {"unload_dependents": True}},
            timeout=30
        )
        if resp.status_code == 200:
            print(f"    {model_name}: unloaded")
            return True
        else:
            print(f"    {model_name}: failed ({resp.status_code})")
            return False
    except Exception as e:
        print(f"    {model_name}: error - {e}")
        return False


def clear_pytorch_cache():
    """Clear PyTorch CUDA cache (only works if running in same process)."""
    print("\nStep 2: Attempting to clear PyTorch CUDA cache...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  PyTorch CUDA cache cleared")
            return True
        else:
            print("  CUDA not available in this process")
            return False
    except ImportError:
        print("  PyTorch not installed in this environment")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def show_gpu_memory():
    """Show current GPU memory usage."""
    print("\nGPU Memory Status:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            used, total, free = result.stdout.strip().split(", ")
            print(f"  Used: {used} MiB / {total} MiB (Free: {free} MiB)")
        else:
            print("  Could not query GPU memory")
    except FileNotFoundError:
        print("  nvidia-smi not found (not running on GPU node?)")
    except Exception as e:
        print(f"  Error: {e}")


def show_gpu_processes():
    """Show processes using GPU."""
    print("\nGPU Processes:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,name",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                print(f"  PID       Memory    Process")
                print(f"  -------   --------  -------")
                for line in output.split("\n"):
                    parts = line.split(", ")
                    if len(parts) >= 3:
                        print(f"  {parts[0]:<8}  {parts[1]:<8}  {parts[2]}")
            else:
                print("  No processes using GPU")
        else:
            print("  Could not query GPU processes")
    except FileNotFoundError:
        print("  nvidia-smi not found")
    except Exception as e:
        print(f"  Error: {e}")


def kill_gpu_processes():
    """Kill all processes using the GPU (dangerous!)."""
    print("\nStep 3: Killing GPU processes...")
    try:
        # Get PIDs
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            pids = [p.strip() for p in pids if p.strip()]

            if not pids:
                print("  No GPU processes to kill")
                return True

            print(f"  Found {len(pids)} GPU processes: {pids}")

            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], timeout=5)
                    print(f"  Killed PID {pid}")
                except Exception as e:
                    print(f"  Failed to kill PID {pid}: {e}")

            return True
        else:
            print("  Could not query GPU processes")
            return False
    except FileNotFoundError:
        print("  nvidia-smi not found - run this inside the Triton pod")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Force cleanup of GPU memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal cleanup (unload all models)
  python scripts/model_management/force_cleanup.py

  # Aggressive cleanup (kill GPU processes) - run inside Triton pod
  python scripts/model_management/force_cleanup.py --kill-gpu-processes

  # Via kubectl
  kubectl exec -n <namespace> deployment/triton-inference-server -- \\
    python /path/to/force_cleanup.py --kill-gpu-processes
"""
    )
    parser.add_argument(
        "--kill-gpu-processes",
        action="store_true",
        help="Kill all processes using GPU (dangerous, use as last resort)"
    )
    parser.add_argument(
        "--rest-url",
        help="REST proxy URL"
    )

    args = parser.parse_args()

    rest_url = args.rest_url or get_rest_url()
    api_key = get_api_key()

    print("=" * 60)
    print("FORCE GPU CLEANUP")
    print("=" * 60)
    print(f"REST URL: {rest_url}")

    # Show initial state
    show_gpu_memory()
    show_gpu_processes()

    # Step 1: Unload via API
    loaded = get_loaded_models(rest_url, api_key)
    if loaded:
        print(f"\nCurrently loaded models: {loaded}")
        for model in loaded:
            force_unload_model(rest_url, api_key, model)

    unload_all_models(rest_url, api_key)

    # Step 2: Clear PyTorch cache (if available)
    clear_pytorch_cache()

    # Step 3: Kill GPU processes (if requested)
    if args.kill_gpu_processes:
        kill_gpu_processes()

    # Show final state
    print("\n" + "=" * 60)
    print("AFTER CLEANUP")
    print("=" * 60)
    show_gpu_memory()
    show_gpu_processes()

    # Check if models are still loaded
    remaining = get_loaded_models(rest_url, api_key)
    if remaining:
        print(f"\nWarning: Models still loaded: {remaining}")
        print("Try: kubectl rollout restart deployment/triton-inference-server -n <namespace>")
    else:
        print("\nAll models unloaded. GPU may still have memory used by Triton process.")
        if not args.kill_gpu_processes:
            print("If GPU memory is still not free, try --kill-gpu-processes or restart Triton.")


if __name__ == "__main__":
    main()
