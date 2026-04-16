"""Placement management routes for the Triton Admin Dashboard.

These routes handle model placement management across Triton pods, including:
- Viewing placement information (which models are on which pods)
- Managing placements (add/remove)
- Detecting and cleaning up stale placements (from scaled-down pods)
- Providing load balancing recommendations
"""

import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from config import get_admin_url, get_proxy_url, get_auth_headers, load_namespaces, settings

logger = logging.getLogger(__name__)

# Cache for access token (to avoid fetching on every request)
_cached_token: Optional[str] = None


async def get_domino_access_token() -> Optional[str]:
    """Fetch access token from Domino API proxy.

    In Domino Apps, call $DOMINO_API_PROXY/access-token to get a token
    for authenticating with other Domino services.
    """
    global _cached_token

    import os
    api_proxy = os.environ.get("DOMINO_API_PROXY", "")
    if not api_proxy:
        logger.debug("DOMINO_API_PROXY not set, skipping token fetch")
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{api_proxy}/access-token")
            response.raise_for_status()
            # The endpoint returns raw JWT token, not JSON
            token = response.text.strip()
            if token:
                _cached_token = token
                logger.debug("Fetched Domino access token")
            return token
    except Exception as e:
        logger.warning(f"Failed to fetch Domino access token: {e}")
        # Return cached token if available
        return _cached_token


async def get_auth_headers_async() -> dict:
    """Get authentication headers for API requests.

    Priority:
    1. DOMINO_USER_API_KEY env var -> X-Domino-Api-Key header
    2. Access token from DOMINO_API_PROXY -> Authorization Bearer header
    """
    import os

    # First, try DOMINO_USER_API_KEY
    api_key = os.environ.get("DOMINO_USER_API_KEY", "").strip()
    if api_key:
        logger.debug("Using DOMINO_USER_API_KEY for auth")
        return {"X-Domino-Api-Key": api_key}

    # Fall back to access token from DOMINO_API_PROXY
    token = await get_domino_access_token()
    if token:
        logger.debug("Using Bearer token for auth")
        return {"Authorization": f"Bearer {token}"}

    logger.warning("No authentication available")
    return {}

router = APIRouter(prefix="/api/placement", tags=["Placement"])

# Templates will be set by server.py
templates: Optional[Jinja2Templates] = None


def set_templates(t: Jinja2Templates) -> None:
    """Set the Jinja2 templates instance."""
    global templates
    templates = t


# Response Models
class PodInfo(BaseModel):
    """Information about a Triton pod."""
    name: str
    status: str  # Running, Pending, Failed, etc.
    pod_ip: Optional[str] = None
    node: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    model_count: int = 0


class PlacementInfo(BaseModel):
    """Placement information for a model."""
    model_name: str
    pods: List[str]
    stale_pods: List[str] = Field(default_factory=list)  # Pods that no longer exist


class StalePlacement(BaseModel):
    """A stale placement entry."""
    model: str
    pod: str
    reason: str = "pod_not_found"


class ModelInfo(BaseModel):
    """Information about a model in the repository."""
    name: str
    state: str  # READY, UNAVAILABLE, etc.
    placements: List[str] = Field(default_factory=list)  # Pod names where placed


class PlacementOverview(BaseModel):
    """Overview of all placements in the system."""
    total_models_with_placement: int
    total_pods: int
    healthy_pods: int
    stale_placement_count: int
    placements: Dict[str, List[str]]
    stale_placements: List[StalePlacement]
    pods: List[PodInfo]
    all_models: List[ModelInfo] = Field(default_factory=list)  # All models in repository
    admin_available: bool = True  # False when admin API is unavailable (local dev)
    local_mode: bool = False  # True when running in local development mode


class Recommendation(BaseModel):
    """A placement recommendation."""
    model: str
    type: str  # single_point_of_failure, no_placement, unbalanced
    severity: str  # info, warning, critical
    message: str
    suggested_action: Optional[Dict[str, Any]] = None


class CleanupResult(BaseModel):
    """Result of stale placement cleanup."""
    removed: List[StalePlacement]
    failed: List[Dict[str, Any]]
    message: str


# Helper functions
async def get_proxy_client(namespace: str) -> httpx.AsyncClient:
    """Create an HTTP client for the proxy API."""
    proxy_url = get_proxy_url(namespace)
    headers = await get_auth_headers_async()
    return httpx.AsyncClient(
        base_url=proxy_url,
        timeout=settings.proxy_timeout_secs,
        headers=headers,
    )


async def get_admin_client(namespace: str) -> httpx.AsyncClient:
    """Create an HTTP client for the admin API."""
    admin_url = get_admin_url(namespace)
    if not admin_url:
        raise HTTPException(
            status_code=400,
            detail=f"No admin URL configured for namespace: {namespace}"
        )
    headers = await get_auth_headers_async()
    return httpx.AsyncClient(
        base_url=admin_url,
        timeout=30.0,
        headers=headers,
    )


async def get_running_pods(namespace: str, request: Request = None) -> tuple[List[PodInfo], bool]:
    """Get list of running Triton pods from admin API.

    Args:
        namespace: The namespace to query
        request: Optional FastAPI request (unused, kept for compatibility)

    Returns:
        Tuple of (pods list, admin_available flag)
    """
    try:
        admin_url = get_admin_url(namespace)
        if not admin_url:
            logger.info(f"No admin URL configured for namespace: {namespace}")
            return [], False

        # Get auth headers (from Domino API proxy or env)
        headers = await get_auth_headers_async()

        async with httpx.AsyncClient(
            base_url=admin_url,
            timeout=30.0,
            headers=headers,
        ) as client:
            response = await client.get("/v1/deployments/inference-server/pods")
            response.raise_for_status()
            data = response.json()
            pods = []
            for pod in data.get("pods", []):
                pods.append(PodInfo(
                    name=pod.get("name", ""),
                    status=pod.get("phase", "Unknown"),
                    pod_ip=pod.get("pod_ip"),
                    node=pod.get("node"),
                    models=[],
                    model_count=0,
                ))
            return pods, True
    except httpx.HTTPError as e:
        logger.warning(f"Failed to get pods from admin API: {e}")
        return [], False
    except Exception as e:
        logger.warning(f"Admin API error: {e}")
        return [], False


async def get_all_placements(namespace: str) -> Dict[str, List[str]]:
    """Get all placements from proxy API."""
    try:
        async with await get_proxy_client(namespace) as client:
            response = await client.get("/v1/placements")
            response.raise_for_status()
            data = response.json()
            return data.get("placements", {})
    except httpx.HTTPError as e:
        logger.warning(f"Failed to get placements from proxy API: {e}")
        return {}


async def get_loaded_models(namespace: str) -> List[Dict[str, Any]]:
    """Get list of loaded models from proxy API."""
    try:
        async with await get_proxy_client(namespace) as client:
            # Triton V2 repository index requires POST
            response = await client.post("/v2/repository/index")
            response.raise_for_status()
            models = response.json()
            # Filter for loaded models
            loaded = [m for m in models if m.get("state") == "READY"]
            return loaded
    except httpx.HTTPError as e:
        logger.warning(f"Failed to get models from proxy API: {e}")
        return []


async def get_all_repository_models(namespace: str) -> List[Dict[str, Any]]:
    """Get ALL models from repository (loaded and unloaded)."""
    try:
        async with await get_proxy_client(namespace) as client:
            # Triton V2 repository index requires POST
            response = await client.post("/v2/repository/index")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.warning(f"Failed to get repository models from proxy API: {e}")
        return []


async def sync_placements_with_triton(namespace: str, pods: List[PodInfo], placements: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Sync placement data with actual Triton state.

    Query loaded models and ensure placements are recorded.
    If a model is loaded but has no placement, auto-record it to the default pod.

    Returns updated placements dict.
    """
    # Determine default pod name
    if pods:
        # Use first running pod as default
        running_pods = sorted([p.name for p in pods if p.status == "Running"])
        default_pod = running_pods[0] if running_pods else "triton-inference-server-0"
    else:
        default_pod = "triton-inference-server-0"

    try:
        async with await get_proxy_client(namespace) as client:
            # Query loaded models via proxy (works for both local and K8s)
            response = await client.post("/v2/repository/index")
            response.raise_for_status()
            models = response.json()

            for model in models:
                model_name = model.get("name", "")
                state = model.get("state", "")

                # If model is loaded (READY) but has no placement, record it
                if model_name and state == "READY":
                    if model_name not in placements or not placements[model_name]:
                        # Add placement via proxy API
                        try:
                            await client.post(f"/v1/models/{model_name}/placement/{default_pod}")
                            placements[model_name] = [default_pod]
                            logger.info(f"Auto-synced placement for '{model_name}' to {default_pod}")
                        except Exception as e:
                            logger.warning(f"Failed to auto-sync placement for '{model_name}': {e}")

            return placements
    except Exception as e:
        logger.warning(f"Failed to sync placements: {e}")
        return placements


async def detect_stale_placements(namespace: str, pods: List[PodInfo] = None, request: Request = None) -> List[StalePlacement]:
    """
    Compare placements against running pods.
    Returns list of stale entries where pod no longer exists.

    Args:
        namespace: The namespace to check
        pods: Optional pre-fetched pods list. If None, will fetch from admin API.
        request: Optional FastAPI request to extract auth headers from
    """
    # Get running pods if not provided
    if pods is None:
        pods, _ = await get_running_pods(namespace, request)

    running_pod_names = {pod.name for pod in pods}

    # Get all placements
    placements = await get_all_placements(namespace)

    # In local mode (no pods), we can't detect stale placements
    if not running_pod_names:
        return []

    # Find stale entries
    stale = []
    for model_name, model_pods in placements.items():
        for pod in model_pods:
            if pod not in running_pod_names:
                stale.append(StalePlacement(
                    model=model_name,
                    pod=pod,
                    reason="pod_not_found"
                ))

    return stale


# API Endpoints
@router.get("/overview", response_model=PlacementOverview)
async def get_placement_overview(request: Request, namespace: str = Query(default="local")):
    """
    Get a complete overview of model placements.

    Returns placement data, pod health status, and stale placement warnings.
    Automatically syncs placement data with actual Triton state - if a model
    is loaded but has no placement recorded, it will be auto-assigned to the
    default pod (triton-inference-server-0).
    """
    try:
        # Fetch pods (may fail in local mode)
        pods, admin_available = await get_running_pods(namespace, request)
        placements = await get_all_placements(namespace)

        # Sync placements with actual Triton state (auto-record missing placements)
        placements = await sync_placements_with_triton(namespace, pods, placements)

        # Get all models from repository
        repo_models = await get_all_repository_models(namespace)

        # Detect stale placements (only if admin API available)
        stale = await detect_stale_placements(namespace, pods) if admin_available else []

        # Check if we're in local mode
        local_mode = namespace == "local" or not admin_available

        # Calculate pod model counts
        pod_model_map: Dict[str, List[str]] = {pod.name: [] for pod in pods}

        for model_name, model_pods in placements.items():
            for pod in model_pods:
                if pod in pod_model_map:
                    pod_model_map[pod].append(model_name)

        # Update pod info with model counts
        for pod in pods:
            pod.models = pod_model_map.get(pod.name, [])
            pod.model_count = len(pod.models)

        # Count healthy pods
        healthy_pods = sum(1 for p in pods if p.status == "Running")

        # Build all_models list with placement info
        all_models = []
        for model in repo_models:
            model_name = model.get("name", "")
            if model_name:
                all_models.append(ModelInfo(
                    name=model_name,
                    state=model.get("state", "UNAVAILABLE"),
                    placements=placements.get(model_name, [])
                ))

        # Sort by name for consistent display
        all_models.sort(key=lambda m: m.name)

        return PlacementOverview(
            total_models_with_placement=len(placements),
            total_pods=len(pods),
            healthy_pods=healthy_pods,
            stale_placement_count=len(stale),
            placements=placements,
            stale_placements=stale,
            pods=pods,
            all_models=all_models,
            admin_available=admin_available,
            local_mode=local_mode,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pods", response_model=List[PodInfo])
async def get_pods(request: Request, namespace: str = Query(default="local")):
    """
    Get list of Triton pods with their placement information.
    """
    try:
        pods, _ = await get_running_pods(namespace, request)
        placements = await get_all_placements(namespace)

        # Map models to pods
        pod_model_map: Dict[str, List[str]] = {pod.name: [] for pod in pods}
        for model_name, model_pods in placements.items():
            for pod in model_pods:
                if pod in pod_model_map:
                    pod_model_map[pod].append(model_name)

        # Update pod info
        for pod in pods:
            pod.models = pod_model_map.get(pod.name, [])
            pod.model_count = len(pod.models)

        return pods
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=PlacementInfo)
async def get_model_placement(
    request: Request,
    model_name: str,
    namespace: str = Query(default="local"),
):
    """
    Get placement information for a specific model.
    """
    try:
        pods, admin_available = await get_running_pods(namespace, request)
        running_pod_names = {pod.name for pod in pods}

        async with await get_proxy_client(namespace) as client:
            response = await client.get(f"/v1/models/{model_name}/placement")
            response.raise_for_status()
            data = response.json()

        all_pods = data.get("pods", [])

        # If admin API not available, we can't distinguish active vs stale
        if not admin_available:
            return PlacementInfo(
                model_name=model_name,
                pods=all_pods,
                stale_pods=[],
            )

        active_pods = [p for p in all_pods if p in running_pod_names]
        stale_pods = [p for p in all_pods if p not in running_pod_names]

        return PlacementInfo(
            model_name=model_name,
            pods=active_pods,
            stale_pods=stale_pods,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Proxy API unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/pods/{pod_name}")
async def add_placement(
    model_name: str,
    pod_name: str,
    namespace: str = Query(default="local"),
):
    """
    Add a placement for a model on a specific pod.
    """
    try:
        async with await get_proxy_client(namespace) as client:
            response = await client.post(f"/v1/models/{model_name}/placement/{pod_name}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Proxy API unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}/pods/{pod_name}")
async def remove_placement(
    model_name: str,
    pod_name: str,
    namespace: str = Query(default="local"),
):
    """
    Remove a placement for a model from a specific pod.
    """
    try:
        async with await get_proxy_client(namespace) as client:
            response = await client.delete(f"/v1/models/{model_name}/placement/{pod_name}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Proxy API unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup", response_model=CleanupResult)
async def cleanup_stale_placements(
    request: Request,
    namespace: str = Query(default="local"),
    reassign_to_default: bool = Query(default=True, description="Reassign models to default pod (pod-0) instead of just removing")
):
    """
    Clean up stale placement entries and optionally reassign to default pod.

    Stale placements are entries for pods that no longer exist
    (e.g., after scaling down the StatefulSet).

    When reassign_to_default=True (default), models from scaled-down pods
    are reassigned to the default pod (typically pod-0) for deterministic routing.
    """
    try:
        pods, admin_available = await get_running_pods(namespace, request)
        stale = await detect_stale_placements(namespace, pods)
        removed = []
        failed = []

        # Find the default pod (lowest numbered running pod, typically pod-0)
        running_pods = sorted([p.name for p in pods if p.status == "Running"])
        default_pod = running_pods[0] if running_pods else None

        async with await get_proxy_client(namespace) as client:
            for entry in stale:
                try:
                    # Remove stale placement
                    response = await client.delete(
                        f"/v1/models/{entry.model}/placement/{entry.pod}"
                    )
                    response.raise_for_status()

                    # Reassign to default pod if enabled and a running pod exists
                    if reassign_to_default and default_pod:
                        # Check if model already has placement on default pod
                        placements = await get_all_placements(namespace)
                        model_placements = placements.get(entry.model, [])
                        if default_pod not in model_placements:
                            # Add placement to default pod
                            add_response = await client.post(
                                f"/v1/models/{entry.model}/placement/{default_pod}"
                            )
                            add_response.raise_for_status()
                            logger.info(f"Reassigned {entry.model} from {entry.pod} to {default_pod}")

                    removed.append(entry)
                except Exception as e:
                    failed.append({
                        "model": entry.model,
                        "pod": entry.pod,
                        "error": str(e)
                    })

        msg = f"Cleaned up {len(removed)} stale placements"
        if reassign_to_default and default_pod:
            msg += f" (reassigned to {default_pod})"

        return CleanupResult(
            removed=removed,
            failed=failed,
            message=msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=List[Recommendation])
async def get_recommendations(request: Request, namespace: str = Query(default="local")):
    """
    Analyze current placements and suggest improvements.

    Checks for:
    - Single point of failure (model on only 1 pod when multiple available)
    - No placement recorded (model loaded but not tracked)
    - Load imbalance across pods
    """
    try:
        pods, admin_available = await get_running_pods(namespace, request)
        placements = await get_all_placements(namespace)
        loaded_models = await get_loaded_models(namespace)

        running_pod_names = [pod.name for pod in pods if pod.status == "Running"]
        recommendations = []

        # Rule 1: Single point of failure
        for model, model_pods in placements.items():
            active_pods = [p for p in model_pods if p in running_pod_names]
            if len(active_pods) == 1 and len(running_pod_names) > 1:
                # Suggest another pod
                other_pods = [p for p in running_pod_names if p not in active_pods]
                suggested_pod = other_pods[0] if other_pods else None
                recommendations.append(Recommendation(
                    model=model,
                    type="single_point_of_failure",
                    severity="warning",
                    message=f"Model '{model}' is only on 1 pod. Consider adding to another pod for redundancy.",
                    suggested_action={
                        "action": "add_placement",
                        "pod": suggested_pod
                    } if suggested_pod else None
                ))

        # Rule 2: No placement recorded for loaded models
        placement_models = set(placements.keys())
        for model in loaded_models:
            model_name = model.get("name", "")
            if model_name and model_name not in placement_models:
                recommendations.append(Recommendation(
                    model=model_name,
                    type="no_placement",
                    severity="info",
                    message=f"Model '{model_name}' is loaded but has no placement recorded. Using default routing."
                ))

        # Rule 3: Check for underutilized pods
        if len(running_pod_names) > 1 and placements:
            pod_model_counts = {p: 0 for p in running_pod_names}
            for model, model_pods in placements.items():
                for pod in model_pods:
                    if pod in pod_model_counts:
                        pod_model_counts[pod] += 1

            avg_models = sum(pod_model_counts.values()) / len(pod_model_counts)
            for pod, count in pod_model_counts.items():
                if count == 0 and avg_models > 0:
                    recommendations.append(Recommendation(
                        model="",
                        type="underutilized_pod",
                        severity="info",
                        message=f"Pod '{pod}' has no model placements. Consider assigning models for load balancing."
                    ))

        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stale", response_model=List[StalePlacement])
async def get_stale_placements(request: Request, namespace: str = Query(default="local")):
    """
    Get list of stale placements (placements for non-existent pods).
    """
    try:
        return await detect_stale_placements(namespace, request=request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
