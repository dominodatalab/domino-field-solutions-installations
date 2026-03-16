#!/usr/bin/env python3
"""Unload models from Triton Inference Server.

Uses Triton's v2 repository API with unload_dependents=True for cleaner unload.

Usage:
    # Unload a specific model
    python scripts/unload_model.py smollm-135m-python

    # Unload multiple models
    python scripts/unload_model.py smollm-135m-python whisper-tiny-python

    # Unload all loaded models
    python scripts/unload_model.py --all

    # With custom REST URL
    python scripts/unload_model.py smollm-135m-python --rest-url http://localhost:8080

Note: If GPU memory is still not freed after unload, restart Triton:
    kubectl rollout restart deployment/triton-inference-server -n <namespace>
"""

import argparse
import os
import sys
import requests


def get_rest_url() -> str:
    """Get REST URL from environment or default."""
    return os.environ.get("TRITON_REST_URL", "http://localhost:8080")


def get_api_key() -> str:
    """Get API key from environment."""
    return os.environ.get("DOMINO_USER_API_KEY", "")


def get_model_status(rest_url: str, api_key: str) -> dict:
    """Get status of all models."""
    headers = {"X-Domino-Api-Key": api_key} if api_key else {}
    try:
        resp = requests.get(f"{rest_url}/v1/models/status", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting status: {e}")
        return {}


def get_loaded_models(rest_url: str, api_key: str) -> list[str]:
    """Get list of currently loaded models."""
    status = get_model_status(rest_url, api_key)
    return [m["name"] for m in status.get("models", []) if m.get("loaded")]


def unload_model(rest_url: str, api_key: str, model_name: str) -> bool:
    """Unload a model using Triton v2 repository API. Returns True on success."""
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["X-Domino-Api-Key"] = api_key

    # Use v2 repository endpoint with unload_dependents for cleaner unload
    try:
        resp = requests.post(
            f"{rest_url}/v2/repository/models/{model_name}/unload",
            headers=headers,
            json={"parameters": {"unload_dependents": True}},
            timeout=60
        )
        if resp.status_code == 200:
            print(f"  {model_name}: UNLOADED")
            return True
        elif resp.status_code == 403:
            print(f"  {model_name}: FAILED - Admin privileges required")
            return False
        else:
            try:
                error = resp.json().get("detail", resp.text)
            except Exception:
                error = resp.text
            print(f"  {model_name}: FAILED - {error}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  {model_name}: ERROR - {e}")
        return False


def unload_all(rest_url: str, api_key: str) -> bool:
    """Unload all loaded models using v2 repository API."""
    loaded = get_loaded_models(rest_url, api_key)
    if not loaded:
        print("No models currently loaded")
        return True

    print(f"Unloading {len(loaded)} models...")
    success = 0
    failed = 0
    for model in loaded:
        if unload_model(rest_url, api_key, model):
            success += 1
        else:
            failed += 1

    print(f"\nUnloaded {success} models")
    if failed:
        print(f"Failed to unload: {failed}")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Unload models from Triton server")
    parser.add_argument("models", nargs="*", help="Model names to unload")
    parser.add_argument("--all", action="store_true", help="Unload all loaded models")
    parser.add_argument("--rest-url", help="REST proxy URL (default: $TRITON_REST_URL or localhost:8080)")
    args = parser.parse_args()

    rest_url = args.rest_url or get_rest_url()
    api_key = get_api_key()

    if not args.models and not args.all:
        # Show loaded models
        print(f"Checking models at {rest_url}...\n")
        loaded = get_loaded_models(rest_url, api_key)

        if loaded:
            print("Currently loaded models:")
            for model in loaded:
                print(f"  {model}")
            print("\nUsage: python scripts/unload_model.py <model_name> [model_name2 ...]")
            print("       python scripts/unload_model.py --all")
        else:
            print("No models currently loaded.")
        return

    # Unload all models using bulk endpoint
    if args.all and not args.models:
        print(f"Unloading all models from {rest_url}...")
        success = unload_all(rest_url, api_key)
        sys.exit(0 if success else 1)

    # Unload specific models
    models_to_unload = args.models
    if args.all:
        loaded = get_loaded_models(rest_url, api_key)
        models_to_unload = list(set(models_to_unload + loaded))

    print(f"Unloading models from {rest_url}:\n")
    success = 0
    failed = 0
    for model in models_to_unload:
        if unload_model(rest_url, api_key, model):
            success += 1
        else:
            failed += 1

    print(f"\nSummary: {success} unloaded, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
