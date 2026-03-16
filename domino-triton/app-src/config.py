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


def get_auth_headers() -> dict:
    """Get authentication headers for proxy and admin API requests.

    Returns X-Domino-Api-Key header which is accepted by both
    the HTTP proxy and admin API for Domino authentication.
    """
    api_key = os.environ.get("DOMINO_USER_API_KEY", "")
    if api_key:
        return {"X-Domino-Api-Key": api_key}
    return {}


# Global settings instance
settings = load_settings()
