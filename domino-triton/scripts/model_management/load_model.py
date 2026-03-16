#!/usr/bin/env python3
"""Load models on Triton Inference Server.

Usage:
    # Load a specific model
    python scripts/load_model.py smollm-135m-python

    # Load multiple models
    python scripts/load_model.py smollm-135m-python whisper-tiny-python bert-base-uncased

    # Load all available models
    python scripts/load_model.py --all

    # With custom REST URL
    python scripts/load_model.py smollm-135m-python --rest-url http://localhost:8080
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


def list_models(rest_url: str, api_key: str) -> list[str]:
    """List all available models in the repository."""
    headers = {"X-Domino-Api-Key": api_key} if api_key else {}
    try:
        resp = requests.get(f"{rest_url}/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.exceptions.RequestException as e:
        print(f"Error listing models: {e}")
        return []


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


def load_model(rest_url: str, api_key: str, model_name: str) -> bool:
    """Load a model. Returns True on success."""
    headers = {"X-Domino-Api-Key": api_key} if api_key else {}
    try:
        resp = requests.post(
            f"{rest_url}/v1/models/{model_name}/load",
            headers=headers,
            timeout=300  # 5 minutes - large models on S3 fuse can be slow
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success", True):
                print(f"  {model_name}: LOADED")
                return True
            else:
                print(f"  {model_name}: FAILED - {data.get('message', 'Unknown error')}")
                return False
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


def main():
    parser = argparse.ArgumentParser(description="Load models on Triton server")
    parser.add_argument("models", nargs="*", help="Model names to load")
    parser.add_argument("--all", action="store_true", help="Load all available models")
    parser.add_argument("--rest-url", help="REST proxy URL (default: $TRITON_REST_URL or localhost:8080)")
    args = parser.parse_args()

    rest_url = args.rest_url or get_rest_url()
    api_key = get_api_key()

    if not args.models and not args.all:
        # Show available models
        print(f"Checking models at {rest_url}...\n")
        status = get_model_status(rest_url, api_key)

        if status:
            print("Available models:")
            for model in status.get("models", []):
                state = "LOADED" if model.get("loaded") else "UNLOADED"
                print(f"  {model['name']}: {state}")
            print("\nUsage: python scripts/load_model.py <model_name> [model_name2 ...]")
            print("       python scripts/load_model.py --all")
        else:
            models = list_models(rest_url, api_key)
            if models:
                print("Available models:", ", ".join(models))
            else:
                print("No models found or cannot connect to server.")
        return

    # Determine which models to load
    if args.all:
        models_to_load = list_models(rest_url, api_key)
        if not models_to_load:
            print("No models found in repository.")
            sys.exit(1)
        print(f"Loading all {len(models_to_load)} models...")
    else:
        models_to_load = args.models

    # Load models
    print(f"Loading models from {rest_url}:\n")
    success = 0
    failed = 0
    for model in models_to_load:
        if load_model(rest_url, api_key, model):
            success += 1
        else:
            failed += 1

    print(f"\nSummary: {success} loaded, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
