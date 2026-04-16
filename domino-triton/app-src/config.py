"""
Configuration for the Triton Admin Dashboard Application.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Settings:
    """Application settings."""

    # App server
    app_host: str
    app_port: int

    # Default proxy URL (can be overridden per-namespace)
    default_proxy_url: str

    # Paths
    samples_path: str
    scripts_path: str
    results_path: str
    namespaces_file: str

    # Timeouts
    proxy_timeout_secs: float

    # Local auth mode - allows UI to input Bearer token or API key
    # Should be disabled in Helm deployments (production)
    local_auth_mode: bool


def load_settings() -> Settings:
    """Load settings from environment variables."""
    app_dir = Path(__file__).parent

    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8888")),
        default_proxy_url=os.getenv("DEFAULT_PROXY_URL", "http://localhost:8080"),
        samples_path=os.getenv("SAMPLES_PATH", str(app_dir / "samples")),
        scripts_path=os.getenv("SCRIPTS_PATH", str(app_dir / "scripts")),
        results_path=os.getenv("RESULTS_PATH", str(app_dir / "results")),
        namespaces_file=os.getenv("NAMESPACES_FILE", str(app_dir / "namespaces.json")),
        proxy_timeout_secs=float(os.getenv("PROXY_TIMEOUT_SECS", "120.0")),
        # Local auth mode enabled by default for local dev, disable in Helm
        local_auth_mode=os.getenv("LOCAL_AUTH_MODE", "true").strip().lower() == "true",
    )


def load_namespaces() -> Dict[str, Any]:
    """Load namespace configuration from JSON file."""
    namespaces_file = Path(settings.namespaces_file)

    if not namespaces_file.exists():
        return {
            "deployments": [
                {
                    "name": "Local Development",
                    "namespace": "local",
                    "proxy_url": settings.default_proxy_url,
                    "admin_url": "http://localhost:8000",
                }
            ],
            "default": "local",
        }

    with open(namespaces_file, "r") as f:
        return json.load(f)


def get_deployment(namespace: str) -> Optional[Dict[str, Any]]:
    """Get deployment configuration by namespace."""
    namespaces = load_namespaces()
    for deployment in namespaces.get("deployments", []):
        if deployment.get("namespace") == namespace:
            return deployment
    return None


def get_proxy_url(namespace: str) -> str:
    """Get proxy URL for a namespace."""
    deployment = get_deployment(namespace)
    if deployment:
        return deployment.get("proxy_url", settings.default_proxy_url)
    return settings.default_proxy_url


def get_admin_url(namespace: str) -> str:
    """Get admin URL for a namespace."""
    deployment = get_deployment(namespace)
    if deployment:
        return deployment.get("admin_url", "")
    return ""


def get_grpc_url(namespace: str) -> str:
    """Get gRPC URL for a namespace."""
    deployment = get_deployment(namespace)
    if deployment:
        return deployment.get("grpc_url", "localhost:50051")
    return "localhost:50051"


def get_model_repo_path(namespace: str) -> str:
    """Get model repository path for a namespace.

    Can be configured explicitly in namespaces.json via 'model_repo_path',
    or derived from namespace name using pattern:
    /mnt/domino-inference-{namespace}-triton-repo-pvc/models
    """
    deployment = get_deployment(namespace)
    if deployment:
        # Check for explicit config
        if deployment.get("model_repo_path"):
            return deployment.get("model_repo_path")

    # Derive from namespace name
    # Pattern: dev -> /mnt/domino-inference-dev-triton-repo-pvc/models
    return f"/mnt/domino-inference-{namespace}-triton-repo-pvc/models"


def get_auth_headers(override_token: str = None, override_api_key: str = None) -> dict:
    """Get authentication headers for proxy and admin API requests.

    Auth resolution order:
    1. override_token - Bearer token passed from UI (local auth mode)
    2. override_api_key - API key passed from UI (local auth mode)
    3. DOMINO_USER_TOKEN env var - Bearer token for testing
    4. DOMINO_API_PROXY/access-token - Auto token fetch (inside Domino)
    5. DOMINO_USER_API_KEY env var - API key fallback

    Args:
        override_token: Bearer token to use (from UI input)
        override_api_key: API key to use (from UI input)

    Returns:
        Dict with appropriate auth header, or empty dict if no auth.
    """
    import requests

    # 1. Override token from UI
    if override_token:
        return {"Authorization": f"Bearer {override_token}"}

    # 2. Override API key from UI
    if override_api_key:
        return {"X-Domino-Api-Key": override_api_key}

    # 3. Check for manually set token (testing/development)
    token = os.environ.get("DOMINO_USER_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}

    # 4. Try to fetch from DOMINO_API_PROXY/access-token (inside Domino)
    api_proxy = os.environ.get("DOMINO_API_PROXY")
    if api_proxy:
        try:
            url = f"{api_proxy.rstrip('/')}/access-token"
            resp = requests.get(url, timeout=5.0)
            if resp.status_code == 200:
                fetched_token = resp.text.strip()
                if fetched_token:
                    return {"Authorization": f"Bearer {fetched_token}"}
        except Exception:
            pass  # Fall through to API key

    # 5. Fall back to API key from environment
    api_key = os.environ.get("DOMINO_USER_API_KEY", "")
    if api_key:
        return {"X-Domino-Api-Key": api_key}

    return {}


# Global settings instance
settings = load_settings()
