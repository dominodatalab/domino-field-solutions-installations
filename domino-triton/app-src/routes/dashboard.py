"""Dashboard routes for the Triton Admin Dashboard.

These routes call the HTTP proxy API to manage models.
"""

import logging
from typing import Any, Dict, List, Optional

import asyncio

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import settings, get_proxy_url, get_admin_url, load_namespaces, get_auth_headers

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dashboard"])

# Templates will be set by server.py
templates: Optional[Jinja2Templates] = None


def set_templates(t: Jinja2Templates) -> None:
    """Set the Jinja2 templates instance."""
    global templates
    templates = t


# Response Models
class ModelInfo(BaseModel):
    """Model information from the proxy."""
    name: str
    state: str
    reason: str


class DashboardModel(BaseModel):
    """Extended model info for dashboard display."""
    name: str
    loaded: bool
    pinned: bool
    cordoned: bool
    last_access_time: Optional[str] = None
    last_accessed_by: Optional[str] = None
    access_count: int = 0
    idle_seconds: Optional[float] = None
    eligible_for_eviction: bool = False
    # Eviction timeout info
    timeout_secs: Optional[float] = None
    is_custom_timeout: bool = False
    time_until_eviction_secs: Optional[float] = None
    # Backend info (populated from Triton config for loaded models)
    backend: Optional[str] = None
    platform: Optional[str] = None


class DashboardOverview(BaseModel):
    """Dashboard overview statistics."""
    total_models: int
    loaded_models: int
    pinned_models: int
    cordoned_models: int
    triton_status: str
    namespace: str
    default_idle_timeout_secs: float = 1800.0  # Default idle timeout


class GPUMetrics(BaseModel):
    """GPU metrics from Triton."""
    gpu_id: str
    name: Optional[str] = None
    memory_total_bytes: Optional[float] = None
    memory_used_bytes: Optional[float] = None
    memory_free_bytes: Optional[float] = None
    utilization_percent: Optional[float] = None
    power_usage_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    temperature_celsius: Optional[float] = None


class CPUMemoryMetrics(BaseModel):
    """CPU and memory metrics from Triton."""
    cpu_utilization_percent: Optional[float] = None
    memory_total_bytes: Optional[float] = None
    memory_used_bytes: Optional[float] = None
    memory_free_bytes: Optional[float] = None


class InferenceMetrics(BaseModel):
    """Per-model inference metrics."""
    model: str
    version: Optional[str] = None
    inference_count: int = 0
    inference_success: int = 0
    inference_failure: int = 0
    avg_request_duration_ms: Optional[float] = None
    avg_queue_duration_ms: Optional[float] = None
    avg_compute_duration_ms: Optional[float] = None
    last_access_time: Optional[str] = None
    last_accessed_by: Optional[str] = None


class ResourceMetrics(BaseModel):
    """Combined resource metrics response."""
    gpu: List[GPUMetrics] = []
    cpu_memory: Optional[CPUMemoryMetrics] = None
    inference: List[InferenceMetrics] = []


# In-memory storage for local auth credentials (single-user dashboard)
# Defined here so helper functions can access it
_local_auth: Dict[str, str] = {
    "auth_type": "none",  # "none", "bearer", "apikey"
    "token": "",
    "api_key": "",
}


def get_local_auth_headers() -> dict:
    """Get auth headers with local auth override if set."""
    if _local_auth["auth_type"] == "bearer" and _local_auth["token"]:
        return get_auth_headers(override_token=_local_auth["token"])
    elif _local_auth["auth_type"] == "apikey" and _local_auth["api_key"]:
        return get_auth_headers(override_api_key=_local_auth["api_key"])
    else:
        return get_auth_headers()


# Helper functions
async def get_proxy_client(namespace: str) -> httpx.AsyncClient:
    """Create an HTTP client for the proxy."""
    proxy_url = get_proxy_url(namespace)
    return httpx.AsyncClient(
        base_url=proxy_url,
        timeout=settings.proxy_timeout_secs,
        headers=get_local_auth_headers(),
    )


async def get_admin_client(namespace: str) -> httpx.AsyncClient:
    """Create an HTTP client for the admin API."""
    admin_url = get_admin_url(namespace)
    return httpx.AsyncClient(
        base_url=admin_url,
        timeout=settings.proxy_timeout_secs,
        headers=get_local_auth_headers(),
    )


async def fetch_models_from_proxy(namespace: str) -> List[Dict[str, Any]]:
    """Fetch model list from the proxy."""
    async with await get_proxy_client(namespace) as client:
        try:
            # Get model repository index
            response = await client.post("/v2/repository/index", json={})
            response.raise_for_status()
            models_data = response.json()
            return models_data if isinstance(models_data, list) else []
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch models from proxy: {e}")
            return []


async def fetch_dashboard_models(namespace: str) -> List[Dict[str, Any]]:
    """Fetch models with complete state from the proxy dashboard endpoint."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.get("/v1/dashboard/models")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch dashboard models: {e}")
            return []


async def fetch_model_state(namespace: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Fetch detailed model state from the proxy."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.get(f"/v1/dashboard/models/{model_name}")
            if response.status_code == 200:
                return response.json()
            return None
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch model state: {e}")
            return None


async def fetch_triton_health(namespace: str) -> str:
    """Check Triton health via the proxy."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.get("/v2/health/ready")
            return "healthy" if response.status_code == 200 else "degraded"
        except httpx.HTTPError:
            return "unavailable"


async def fetch_gpu_metrics(namespace: str) -> List[Dict[str, Any]]:
    """Fetch GPU metrics from the proxy."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.get("/v1/metrics/gpu")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch GPU metrics: {e}")
            return []


async def fetch_cpu_metrics(namespace: str) -> Optional[Dict[str, Any]]:
    """Fetch CPU/memory metrics from the proxy."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.get("/v1/metrics/cpu")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch CPU metrics: {e}")
            return None


async def fetch_inference_metrics(namespace: str) -> List[Dict[str, Any]]:
    """Fetch per-model inference metrics from the proxy."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.get("/v1/metrics/inference")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch inference metrics: {e}")
            return []


async def fetch_dashboard_overview(namespace: str) -> Optional[Dict[str, Any]]:
    """Fetch dashboard overview from the proxy (includes default timeout)."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.get("/v1/dashboard/overview")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch dashboard overview: {e}")
            return None


async def fetch_triton_config(namespace: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Fetch model config from Triton's native /v2/models/{model}/config API.
    Only available for loaded models. Returns backend, platform, inputs, outputs, etc.
    """
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.get(f"/v2/models/{model_name}/config")
            if response.status_code == 200:
                return response.json()
            return None
        except httpx.HTTPError:
            return None


# API Endpoints
@router.get("/api/dashboard/overview")
async def get_overview(namespace: str = Query(default="local")):
    """Get dashboard overview statistics."""
    # Try to fetch from proxy which includes the default timeout
    proxy_overview = await fetch_dashboard_overview(namespace)

    if proxy_overview:
        return DashboardOverview(
            total_models=proxy_overview.get("total_models", 0),
            loaded_models=proxy_overview.get("loaded_models", 0),
            pinned_models=proxy_overview.get("pinned_models", 0),
            cordoned_models=proxy_overview.get("cordoned_models", 0),
            triton_status=proxy_overview.get("triton_status", "unavailable"),
            namespace=namespace,
            default_idle_timeout_secs=proxy_overview.get("default_idle_timeout_secs", 1800.0),
        )

    # Fallback: compute from models list
    models = await fetch_dashboard_models(namespace)
    triton_status = await fetch_triton_health(namespace)

    total = len(models)
    loaded = sum(1 for m in models if m.get("loaded", False))
    pinned = sum(1 for m in models if m.get("pinned", False))
    cordoned = sum(1 for m in models if m.get("cordoned", False))

    return DashboardOverview(
        total_models=total,
        loaded_models=loaded,
        pinned_models=pinned,
        cordoned_models=cordoned,
        triton_status=triton_status,
        namespace=namespace,
    )


@router.get("/api/dashboard/models")
async def get_models(namespace: str = Query(default="local")):
    """Get all models with their status."""
    models = await fetch_dashboard_models(namespace)

    # Fetch Triton config for all loaded models in parallel to get backend/platform
    loaded_names = [m["name"] for m in models if m.get("loaded")]
    triton_configs = await asyncio.gather(
        *[fetch_triton_config(namespace, name) for name in loaded_names],
        return_exceptions=True,
    )
    config_by_name = {
        name: cfg
        for name, cfg in zip(loaded_names, triton_configs)
        if isinstance(cfg, dict)
    }

    result = []
    for model in models:
        name = model.get("name", "")
        cfg = config_by_name.get(name, {})
        dashboard_model = DashboardModel(
            name=name,
            loaded=model.get("loaded", False),
            pinned=model.get("pinned", False),
            cordoned=model.get("cordoned", False),
            last_access_time=model.get("last_access_time"),
            last_accessed_by=model.get("last_accessed_by"),
            access_count=model.get("access_count", 0),
            idle_seconds=model.get("idle_seconds"),
            eligible_for_eviction=model.get("eligible_for_eviction", False),
            timeout_secs=model.get("timeout_secs"),
            is_custom_timeout=model.get("is_custom_timeout", False),
            time_until_eviction_secs=model.get("time_until_eviction_secs"),
            backend=cfg.get("backend") or cfg.get("platform") or None,
            platform=cfg.get("platform") or None,
        )
        result.append(dashboard_model)

    return {"models": result}


@router.get("/api/dashboard/models/{model_name}")
async def get_model(model_name: str, namespace: str = Query(default="local")):
    """Get details for a specific model."""
    state = await fetch_model_state(namespace, model_name)

    if not state:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    # Get model repository info
    models = await fetch_models_from_proxy(namespace)
    model_info = next((m for m in models if m.get("name") == model_name), None)

    return {
        "name": model_name,
        "loaded": model_info.get("state") == "READY" if model_info else False,
        "state": state,
    }


@router.post("/api/dashboard/models/{model_name}/load")
async def load_model(model_name: str, namespace: str = Query(default="local")):
    """Load a model."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.post(f"/v2/repository/models/{model_name}/load")
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} loaded"}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/dashboard/models/{model_name}/unload")
async def unload_model(model_name: str, namespace: str = Query(default="local")):
    """Unload a model."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.post(f"/v2/repository/models/{model_name}/unload")
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} unloaded"}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/dashboard/models/{model_name}/pin")
async def pin_model(model_name: str, namespace: str = Query(default="local")):
    """Pin a model to prevent eviction."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.post(
                f"/v1/models/{model_name}/pin",
                json={"pinned": True}
            )
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} pinned"}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/dashboard/models/{model_name}/unpin")
async def unpin_model(model_name: str, namespace: str = Query(default="local")):
    """Unpin a model."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.post(
                f"/v1/models/{model_name}/pin",
                json={"pinned": False}
            )
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} unpinned"}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/dashboard/models/{model_name}/cordon")
async def cordon_model(model_name: str, namespace: str = Query(default="local")):
    """Cordon a model to stop new requests."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.post(f"/v1/models/{model_name}/cordon")
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} cordoned"}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/dashboard/models/{model_name}/uncordon")
async def uncordon_model(model_name: str, namespace: str = Query(default="local")):
    """Uncordon a model."""
    async with await get_proxy_client(namespace) as client:
        try:
            response = await client.post(f"/v1/models/{model_name}/uncordon")
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} uncordoned"}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))




@router.get("/api/dashboard/models/{model_name}/config")
async def get_model_config(model_name: str, namespace: str = Query(default="local")):
    """Get model configuration via admin API."""
    try:
        async with await get_admin_client(namespace) as client:
            response = await client.get(f"/v1/models/{model_name}/config")
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Config not found for model '{model_name}'"
                )
            response.raise_for_status()
            return response.json()
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Failed to get config for {model_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Admin API unavailable: {e}")
    except Exception as e:
        logger.error(f"Failed to read config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")


@router.get("/api/dashboard/models/{model_name}/triton-config")
async def get_model_triton_config(model_name: str, namespace: str = Query(default="local")):
    """Get model configuration directly from Triton's native config API.
    Works without the admin API. Only available for loaded models.
    Returns the full parsed config: backend, platform, inputs, outputs, instance_group, parameters.
    """
    try:
        async with await get_proxy_client(namespace) as client:
            response = await client.get(f"/v2/models/{model_name}/config")
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_name}' not found or not loaded"
                )
            response.raise_for_status()
            return response.json()
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Failed to get Triton config for {model_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Proxy unavailable: {e}")


class ModelConfigUpdate(BaseModel):
    """Model config update request."""
    kind: Optional[str] = None  # KIND_CPU or KIND_GPU (None=don't change)
    count: Optional[int] = None
    gpus: Optional[List[int]] = None
    reload: bool = True
    idle_timeout_secs: Optional[int] = None  # Per-model idle timeout (None=don't change, 0=use default)
    model_types: Optional[List[str]] = None  # video, image, text, audio (None=don't change)


@router.put("/api/dashboard/models/{model_name}/config")
async def update_model_config(
    model_name: str,
    request: ModelConfigUpdate,
    namespace: str = Query(default="local"),
):
    """Update model configuration via admin API.

    1. Update config.pbtxt via admin endpoint
    2. Optionally unload the model via proxy (so Triton picks up the new config on next load)
    """
    try:
        # Build admin API payload (only include fields that are set)
        admin_payload = {}
        if request.kind is not None:
            admin_payload["kind"] = request.kind
        if request.count is not None:
            admin_payload["count"] = request.count
        if request.gpus is not None:
            admin_payload["gpus"] = request.gpus
        if request.idle_timeout_secs is not None:
            admin_payload["idle_timeout_secs"] = request.idle_timeout_secs
        if request.model_types is not None:
            admin_payload["model_types"] = request.model_types

        # 1. Update config via admin API
        async with await get_admin_client(namespace) as client:
            response = await client.put(
                f"/v1/models/{model_name}/config",
                json=admin_payload
            )
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Config not found for model '{model_name}'"
                )
            response.raise_for_status()
            admin_result = response.json()

        logger.info(f"Updated config for {model_name} via admin API: {admin_result.get('changes', [])}")

        # 2. Unload model via proxy (so changes take effect on next load)
        unload_message = ""
        if request.reload:
            async with await get_proxy_client(namespace) as client:
                try:
                    response = await client.post(f"/v2/repository/models/{model_name}/unload")
                    if response.status_code == 200:
                        unload_message = " Model unloaded - reload to apply changes."
                    else:
                        unload_message = f" Warning: Could not unload model (status {response.status_code})."
                except httpx.HTTPError as e:
                    unload_message = f" Warning: Could not unload model: {e}"

        return {
            "success": True,
            "message": f"{admin_result.get('message', 'Configuration updated')}.{unload_message}",
            "model": model_name,
            "changes": admin_result.get("changes", []),
        }

    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error(f"Failed to update config for {model_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Admin API unavailable: {e}")
    except Exception as e:
        logger.error(f"Failed to update config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {e}")


@router.get("/api/dashboard/metrics")
async def get_metrics(namespace: str = Query(default="local")):
    """Get Triton resource metrics (GPU, CPU, Memory, Inference)."""
    gpu_metrics = await fetch_gpu_metrics(namespace)
    cpu_metrics = await fetch_cpu_metrics(namespace)
    inference_metrics = await fetch_inference_metrics(namespace)

    # Get model state data to merge last_access_time and last_accessed_by
    models = await fetch_dashboard_models(namespace)
    model_state_map = {m.get("name"): m for m in models}

    # Merge inference metrics with model state
    merged_inference = []
    for inf in (inference_metrics or []):
        model_name = inf.get("model")
        model_state = model_state_map.get(model_name, {})
        merged_inference.append(InferenceMetrics(
            model=model_name,
            version=inf.get("version"),
            inference_count=inf.get("inference_count", 0),
            inference_success=inf.get("inference_success", 0),
            inference_failure=inf.get("inference_failure", 0),
            avg_request_duration_ms=inf.get("avg_request_duration_ms"),
            avg_queue_duration_ms=inf.get("avg_queue_duration_ms"),
            avg_compute_duration_ms=inf.get("avg_compute_duration_ms"),
            last_access_time=model_state.get("last_access_time"),
            last_accessed_by=model_state.get("last_accessed_by"),
        ))

    return ResourceMetrics(
        gpu=[GPUMetrics(**g) for g in gpu_metrics] if gpu_metrics else [],
        cpu_memory=CPUMemoryMetrics(**cpu_metrics) if cpu_metrics else None,
        inference=merged_inference,
    )


# HTML Routes
@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request, namespace: str = Query(default=None)):
    """Render the main dashboard page."""
    if not templates:
        raise HTTPException(status_code=500, detail="Templates not configured")

    namespaces_config = load_namespaces()
    # Use provided namespace or fall back to config default
    current_ns = namespace or namespaces_config.get("default", "local")

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "current_namespace": current_ns,
            "namespaces": namespaces_config.get("deployments", []),
        },
    )


@router.get("/models/{model_name}", response_class=HTMLResponse)
async def model_detail_page(
    request: Request,
    model_name: str,
    namespace: str = Query(default=None),
):
    """Render the model detail page."""
    if not templates:
        raise HTTPException(status_code=500, detail="Templates not configured")

    namespaces_config = load_namespaces()
    # Use provided namespace or fall back to config default
    current_ns = namespace or namespaces_config.get("default", "local")

    return templates.TemplateResponse(
        request,
        "model_detail.html",
        {
            "model_name": model_name,
            "current_namespace": current_ns,
            "namespaces": namespaces_config.get("deployments", []),
        },
    )


# =============================================================================
# Local Auth Mode API
#
# Allows users to provide Bearer token or API key via the UI for testing.
# This is intended for local development only and should be disabled in
# production (Helm deployments) by setting LOCAL_AUTH_MODE=false.
# =============================================================================

class LocalAuthSettings(BaseModel):
    """Local auth settings request/response."""
    auth_type: str  # "none", "bearer", "apikey"
    token: Optional[str] = ""
    api_key: Optional[str] = ""


class LocalAuthStatus(BaseModel):
    """Local auth mode status."""
    enabled: bool
    auth_type: str
    has_token: bool
    has_api_key: bool


@router.get("/api/auth/status")
async def get_local_auth_status() -> LocalAuthStatus:
    """Get local auth mode status."""
    return LocalAuthStatus(
        enabled=settings.local_auth_mode,
        auth_type=_local_auth["auth_type"],
        has_token=bool(_local_auth["token"]),
        has_api_key=bool(_local_auth["api_key"]),
    )


@router.post("/api/auth/settings")
async def set_local_auth_settings(auth_settings: LocalAuthSettings) -> dict:
    """Set local auth credentials (only works when LOCAL_AUTH_MODE=true)."""
    if not settings.local_auth_mode:
        raise HTTPException(
            status_code=403,
            detail="Local auth mode is disabled. Set LOCAL_AUTH_MODE=true to enable."
        )

    _local_auth["auth_type"] = auth_settings.auth_type
    _local_auth["token"] = auth_settings.token or ""
    _local_auth["api_key"] = auth_settings.api_key or ""

    return {
        "success": True,
        "auth_type": _local_auth["auth_type"],
        "message": f"Auth set to {_local_auth['auth_type']}"
    }


@router.delete("/api/auth/settings")
async def clear_local_auth_settings() -> dict:
    """Clear local auth credentials."""
    _local_auth["auth_type"] = "none"
    _local_auth["token"] = ""
    _local_auth["api_key"] = ""

    return {"success": True, "message": "Auth credentials cleared"}
