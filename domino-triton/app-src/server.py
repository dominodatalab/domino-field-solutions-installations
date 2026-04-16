"""
Triton Admin Dashboard Application

A standalone dashboard application for managing Triton Inference Server deployments.
Communicates with Triton proxies across multiple namespaces.

Usage:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8888

For Domino deployment behind reverse proxy:
    Set ROOT_PATH environment variable to the app's base path
    e.g., ROOT_PATH=/app/triton-admin
"""

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routes import dashboard, testing, admin, placement
from config import settings, load_namespaces

# Get root path for reverse proxy support (Domino apps)
# Check multiple environment variables: APP_ROOT_PATH, ROOT_PATH, or DOMINO_RUN_HOST_PATH
ROOT_PATH = os.environ.get(
    "APP_ROOT_PATH",
    os.environ.get("ROOT_PATH", os.environ.get("DOMINO_RUN_HOST_PATH", ""))
).rstrip("/")


# Note: We don't use middleware for root_path because it interferes with StaticFiles.
# Instead, we add root_path as a Jinja2 global variable (see template setup below).

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Load namespace configuration
namespaces = load_namespaces()
logger.info(f"Loaded {len(namespaces['deployments'])} deployment configurations")

# API documentation tags
tags_metadata = [
    {
        "name": "Dashboard",
        "description": "Dashboard API for model management and monitoring.",
    },
    {
        "name": "Testing",
        "description": "Model testing and inference endpoints.",
    },
    {
        "name": "Admin",
        "description": "Kubernetes admin operations (scale, namespace management).",
    },
    {
        "name": "Placement",
        "description": "Model placement management across Triton pods.",
    },
    {
        "name": "Results",
        "description": "Benchmark results and reports.",
    },
]

app = FastAPI(
    title="Triton Admin Dashboard",
    version="1.0.0",
    description="""
## Triton Admin Dashboard

A comprehensive dashboard for managing NVIDIA Triton Inference Server deployments.

### Features

- **Multi-Namespace Support**: Manage Triton deployments across multiple Kubernetes namespaces
- **Model Management**: Load, unload, pin, and cordon models
- **Model Testing**: Test models with sample inputs directly from the UI
- **Scaling**: Scale Triton deployments up/down
- **Benchmarking**: View benchmark results and performance reports

### Quick Start

1. Select a deployment namespace from the dropdown
2. View model status and metrics
3. Load/unload models as needed
4. Test models with sample inputs
5. View benchmark results
""",
    openapi_tags=tags_metadata,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    # Note: Don't set root_path here - Domino's gateway strips the path prefix,
    # and setting root_path interferes with StaticFiles mounts.
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup templates
templates_dir = Path(__file__).parent / "templates"
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
    # Add root_path as a global variable for all templates
    templates.env.globals["root_path"] = ROOT_PATH
    dashboard.set_templates(templates)
    testing.set_templates(templates)
    admin.set_templates(templates)
    placement.set_templates(templates)

# Mount results directory for viewing benchmark reports
# Use configured results_path from settings (respects RESULTS_PATH env var)
results_dir = Path(settings.results_path)
if results_dir.exists():
    app.mount("/results", StaticFiles(directory=str(results_dir), html=True), name="results")
    logger.info(f"Mounted results directory: {results_dir}")

# Include routers
app.include_router(dashboard.router)
app.include_router(testing.router)
app.include_router(admin.router)
app.include_router(placement.router)


# HTML page routes
from fastapi import Query, Request
from fastapi.responses import HTMLResponse

@app.get("/results-page", response_class=HTMLResponse)
async def results_page(request: Request):
    """Render the benchmark results page."""
    namespaces_config = load_namespaces()
    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "current_namespace": namespaces_config.get("default", "local"),
            "namespaces": namespaces_config.get("deployments", []),
        },
    )


@app.get("/view-result/{path:path}", response_class=HTMLResponse)
async def view_result_page(request: Request, path: str):
    """Render the markdown viewer page for a result file."""
    from pathlib import Path
    filename = Path(path).name
    namespaces_config = load_namespaces()
    return templates.TemplateResponse(
        request,
        "markdown_viewer.html",
        {
            "filename": filename,
            "path": path,
            "current_namespace": namespaces_config.get("default", "local"),
            "namespaces": namespaces_config.get("deployments", []),
        },
    )


@app.get("/architecture", response_class=HTMLResponse)
async def architecture_page(request: Request):
    """Render the system architecture documentation page."""
    namespaces_config = load_namespaces()
    return templates.TemplateResponse(
        request,
        "architecture.html",
        {
            "current_namespace": namespaces_config.get("default", "local"),
            "namespaces": namespaces_config.get("deployments", []),
        },
    )


@app.get("/presentation", response_class=HTMLResponse)
async def presentation_page(request: Request):
    """Render the technical presentation page."""
    namespaces_config = load_namespaces()
    return templates.TemplateResponse(
        request,
        "presentation.html",
        {
            "current_namespace": namespaces_config.get("default", "local"),
            "namespaces": namespaces_config.get("deployments", []),
        },
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, namespace: str = Query(default=None)):
    """Render the admin page for deployment management."""
    namespaces_config = load_namespaces()
    current_ns = namespace or namespaces_config.get("default", "local")
    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "current_namespace": current_ns,
            "namespaces": namespaces_config.get("deployments", []),
        },
    )


@app.get("/placement", response_class=HTMLResponse)
async def placement_page(request: Request, namespace: str = Query(default=None)):
    """Render the placement management page."""
    namespaces_config = load_namespaces()
    current_ns = namespace or namespaces_config.get("default", "local")
    return templates.TemplateResponse(
        request,
        "placement.html",
        {
            "current_namespace": current_ns,
            "namespaces": namespaces_config.get("deployments", []),
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/namespaces")
async def get_namespaces():
    """Get available deployment namespaces."""
    return namespaces


def main():
    """Run the dashboard application."""
    import uvicorn

    logger.info(f"Starting Triton Admin Dashboard on {settings.app_host}:{settings.app_port}")

    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
