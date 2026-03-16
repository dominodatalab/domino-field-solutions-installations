#!/usr/bin/env python3
"""Check status of models on Triton Inference Server.

Usage:
    # Show all model status
    python scripts/model_status.py

    # Show specific model status
    python scripts/model_status.py smollm-135m-python

    # With custom REST URL
    python scripts/model_status.py --rest-url http://localhost:8080

    # JSON output
    python scripts/model_status.py --json
"""

import argparse
import json
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
        print(f"Error getting status: {e}", file=sys.stderr)
        return {}


def format_idle_time(seconds: float) -> str:
    """Format idle time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_status_table(status: dict, filter_models: list[str] = None):
    """Print model status as a formatted table."""
    models = status.get("models", [])

    if filter_models:
        models = [m for m in models if m["name"] in filter_models]

    if not models:
        print("No models found.")
        return

    # Calculate column widths
    name_width = max(len(m["name"]) for m in models)
    name_width = max(name_width, 10)

    # Header
    print(f"{'Model':<{name_width}}  {'State':<10}  {'Pinned':<7}  {'Idle Time':<10}")
    print("-" * (name_width + 35))

    # Rows
    for model in sorted(models, key=lambda m: m["name"]):
        name = model["name"]
        state = "LOADED" if model.get("loaded") else "UNLOADED"
        pinned = "Yes" if model.get("pinned") else "No"
        idle_time = format_idle_time(model.get("idle_time_seconds", 0)) if model.get("loaded") else "-"

        # Color coding (ANSI)
        if state == "LOADED":
            state_str = f"\033[92m{state:<10}\033[0m"  # Green
        else:
            state_str = f"\033[90m{state:<10}\033[0m"  # Gray

        print(f"{name:<{name_width}}  {state_str}  {pinned:<7}  {idle_time:<10}")

    # Summary
    loaded = sum(1 for m in models if m.get("loaded"))
    print("-" * (name_width + 35))
    print(f"Total: {len(models)} models, {loaded} loaded")


def main():
    parser = argparse.ArgumentParser(description="Check model status on Triton server")
    parser.add_argument("models", nargs="*", help="Specific model names to check")
    parser.add_argument("--rest-url", help="REST proxy URL (default: $TRITON_REST_URL or localhost:8080)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    rest_url = args.rest_url or get_rest_url()
    api_key = get_api_key()

    status = get_model_status(rest_url, api_key)

    if not status:
        print("Could not retrieve model status. Is the server running?", file=sys.stderr)
        sys.exit(1)

    if args.json:
        if args.models:
            models = [m for m in status.get("models", []) if m["name"] in args.models]
            print(json.dumps({"models": models}, indent=2))
        else:
            print(json.dumps(status, indent=2))
    else:
        print(f"Model Status ({rest_url})\n")
        print_status_table(status, args.models if args.models else None)


if __name__ == "__main__":
    main()
