"""Admin routes for the Triton Admin Dashboard.

These routes handle Kubernetes operations like scaling Triton deployments
and managing resources.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from config import get_admin_url, get_auth_headers, load_namespaces, settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])

# Templates will be set by server.py
templates: Optional[Jinja2Templates] = None


def set_templates(t: Jinja2Templates) -> None:
    """Set the Jinja2 templates instance."""
    global templates
    templates = t


# Request/Response Models
class ScaleRequest(BaseModel):
    """Request to scale a deployment."""
    replicas: int = Field(..., ge=0, le=20, description="Desired replica count")


class ResourceSpec(BaseModel):
    """Resource specification for a container."""
    cpu: Optional[str] = Field(None, description="CPU (e.g., '100m', '1', '2000m')")
    memory: Optional[str] = Field(None, description="Memory (e.g., '128Mi', '1Gi')")
    gpu: Optional[int] = Field(None, ge=0, description="Number of GPUs")


class ResourceRequirements(BaseModel):
    """Resource requirements with requests and limits."""
    requests: Optional[ResourceSpec] = None
    limits: Optional[ResourceSpec] = None


class UpdateResourcesRequest(BaseModel):
    """Request to update container resources."""
    container_name: Optional[str] = Field(None, description="Container name")
    resources: ResourceRequirements


class ScaleResponse(BaseModel):
    """Response from scaling operation."""
    success: bool
    message: str
    current_replicas: Optional[int] = None
    desired_replicas: Optional[int] = None


class DeploymentStatus(BaseModel):
    """Deployment status information."""
    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    available_replicas: int
    overall_status: str


async def get_admin_client(namespace: str) -> httpx.AsyncClient:
    """Create an HTTP client for the admin API."""
    admin_url = get_admin_url(namespace)
    if not admin_url:
        raise HTTPException(
            status_code=400,
            detail=f"No admin URL configured for namespace: {namespace}"
        )
    return httpx.AsyncClient(
        base_url=admin_url,
        timeout=30.0,
        headers=get_auth_headers(),
    )


@router.get("/namespaces")
async def list_namespaces():
    """List available namespaces."""
    config = load_namespaces()
    return {
        "namespaces": [
            {
                "name": d.get("name"),
                "namespace": d.get("namespace"),
                "proxy_url": d.get("proxy_url"),
                "admin_url": d.get("admin_url"),
                "has_admin": bool(d.get("admin_url")),
            }
            for d in config.get("deployments", [])
        ],
        "default": config.get("default", "local"),
    }


@router.get("/deployment/status")
async def get_deployment_status(namespace: str = Query(default="local")):
    """Get the status of the Triton deployment."""
    try:
        async with await get_admin_client(namespace) as client:
            response = await client.get("/v1/deployments/inference-server/status")
            response.raise_for_status()
            data = response.json()

            return {
                "deployment": data.get("deployment"),
                "namespace": data.get("namespace"),
                "replicas": data.get("deployment_replicas", {}),
                "pods": data.get("pods", {}),
                "overall_status": data.get("overall_status"),
            }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Admin API unavailable: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployment/pods")
async def get_pods(namespace: str = Query(default="local")):
    """Get pods for the Triton deployment."""
    try:
        async with await get_admin_client(namespace) as client:
            response = await client.get("/v1/deployments/inference-server/pods")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Admin API unavailable: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployment/resources")
async def get_resources(namespace: str = Query(default="local")):
    """Get resource requests and limits for the deployment."""
    try:
        async with await get_admin_client(namespace) as client:
            response = await client.get("/v1/deployments/inference-server/resources")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Admin API unavailable: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/deployment/resources")
async def update_resources(
    request: UpdateResourcesRequest,
    namespace: str = Query(default="local"),
):
    """Update resource requests and limits for the deployment."""
    try:
        async with await get_admin_client(namespace) as client:
            response = await client.patch(
                "/v1/deployments/inference-server/resources",
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Admin API unavailable: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployment/scale")
async def scale_deployment(
    request: ScaleRequest,
    namespace: str = Query(default="local"),
):
    """Scale the Triton deployment."""
    if request.replicas < 0:
        raise HTTPException(status_code=400, detail="Replicas must be >= 0")

    if request.replicas > 20:
        raise HTTPException(
            status_code=400,
            detail="Cannot scale beyond 20 replicas without admin approval"
        )

    try:
        async with await get_admin_client(namespace) as client:
            response = await client.post(
                "/v1/deployments/inference-server/scale",
                json={"replicas": request.replicas},
            )
            response.raise_for_status()
            data = response.json()

            return ScaleResponse(
                success=True,
                message=f"Scaled to {request.replicas} replicas",
                desired_replicas=request.replicas,
            )
    except httpx.HTTPStatusError as e:
        return ScaleResponse(
            success=False,
            message=f"Failed to scale: {e.response.text}",
        )
    except httpx.HTTPError as e:
        return ScaleResponse(
            success=False,
            message=f"Admin API unavailable: {e}",
        )
    except HTTPException:
        raise
    except Exception as e:
        return ScaleResponse(
            success=False,
            message=str(e),
        )


@router.post("/deployment/restart")
async def restart_deployment(namespace: str = Query(default="local")):
    """Restart the Triton deployment (rolling restart)."""
    try:
        async with await get_admin_client(namespace) as client:
            response = await client.post("/v1/deployments/inference-server/restart")
            response.raise_for_status()
            return {
                "success": True,
                "message": "Rolling restart initiated",
                **response.json(),
            }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Admin API unavailable: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployment/logs")
async def get_logs(
    namespace: str = Query(default="local"),
    tail_lines: int = Query(default=200, ge=1, le=5000),
):
    """Get logs from the Triton deployment."""
    try:
        async with await get_admin_client(namespace) as client:
            response = await client.get(
                "/v1/deployments/inference-server/logs",
                params={"tail_lines": tail_lines},
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Admin API unavailable: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SafeResourceUpdateRequest(BaseModel):
    """Request for safe resource update with scale-to-zero workflow."""
    container_name: Optional[str] = Field(None, description="Container name")
    resources: ResourceRequirements


class SafeResourceUpdateResponse(BaseModel):
    """Response from safe resource update."""
    success: bool
    message: str
    step: str  # current step: scaling_down, updating, scaling_up, complete
    original_replicas: Optional[int] = None
    error: Optional[str] = None


@router.post("/deployment/resources-with-scale")
async def update_resources_with_scale(
    request: SafeResourceUpdateRequest,
    namespace: str = Query(default="local"),
):
    """
    Update resources with scale-to-zero workflow.
    1. Record current replica count
    2. Scale to 0, wait for pods to terminate
    3. Update resources (PATCH)
    4. Scale back to original count
    5. Wait for pods to be ready
    """
    try:
        async with await get_admin_client(namespace) as client:
            # Step 1: Get current replica count
            status_response = await client.get("/v1/deployments/inference-server/status")
            status_response.raise_for_status()
            status_data = status_response.json()
            original_replicas = status_data.get("deployment_replicas", {}).get("desired_replicas", 1)

            if original_replicas == 0:
                # Already scaled to 0, just update resources
                update_response = await client.patch(
                    "/v1/deployments/inference-server/resources",
                    json=request.model_dump(exclude_none=True),
                )
                update_response.raise_for_status()
                return SafeResourceUpdateResponse(
                    success=True,
                    message="Resources updated (deployment was already at 0 replicas)",
                    step="complete",
                    original_replicas=0,
                )

            # Step 2: Scale to 0
            scale_down_response = await client.post(
                "/v1/deployments/inference-server/scale",
                json={"replicas": 0},
            )
            scale_down_response.raise_for_status()

            # Wait for pods to terminate (poll up to 60 seconds)
            for _ in range(30):
                await asyncio.sleep(2)
                pods_response = await client.get("/v1/deployments/inference-server/pods")
                if pods_response.status_code == 200:
                    pods_data = pods_response.json()
                    if pods_data.get("total", 0) == 0:
                        break

            # Step 3: Update resources
            update_response = await client.patch(
                "/v1/deployments/inference-server/resources",
                json=request.model_dump(exclude_none=True),
            )
            update_response.raise_for_status()

            # Step 4: Scale back up
            scale_up_response = await client.post(
                "/v1/deployments/inference-server/scale",
                json={"replicas": original_replicas},
            )
            scale_up_response.raise_for_status()

            # Wait for pods to be ready (poll up to 120 seconds)
            for _ in range(60):
                await asyncio.sleep(2)
                status_check = await client.get("/v1/deployments/inference-server/status")
                if status_check.status_code == 200:
                    check_data = status_check.json()
                    ready = check_data.get("deployment_replicas", {}).get("ready_replicas", 0)
                    if ready >= original_replicas:
                        break

            return SafeResourceUpdateResponse(
                success=True,
                message=f"Resources updated successfully. Scaled back to {original_replicas} replicas.",
                step="complete",
                original_replicas=original_replicas,
            )

    except httpx.HTTPStatusError as e:
        return SafeResourceUpdateResponse(
            success=False,
            message="Failed to update resources",
            step="error",
            error=str(e.response.text),
        )
    except httpx.HTTPError as e:
        return SafeResourceUpdateResponse(
            success=False,
            message="Admin API unavailable",
            step="error",
            error=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        return SafeResourceUpdateResponse(
            success=False,
            message="Unexpected error",
            step="error",
            error=str(e),
        )


@router.get("/health")
async def check_admin_health(namespace: str = Query(default="local")):
    """Check health of the admin API for a namespace."""
    try:
        async with await get_admin_client(namespace) as client:
            response = await client.get("/healthz")
            response.raise_for_status()
            return {
                "healthy": True,
                "namespace": namespace,
                **response.json(),
            }
    except httpx.HTTPError as e:
        return {
            "healthy": False,
            "namespace": namespace,
            "error": str(e),
        }
    except HTTPException:
        return {
            "healthy": False,
            "namespace": namespace,
            "error": "No admin URL configured",
        }
    except Exception as e:
        return {
            "healthy": False,
            "namespace": namespace,
            "error": str(e),
        }
