# System Design

## Architecture

```
Clients (tritonclient.grpc/http)
    │
    ├── gRPC Proxy (50051) ──► Triton Server (8001)
    │
    └── HTTP Proxy (8080)  ──► Triton Server (8000)
```

Both proxies provide:
- Domino authentication via API key headers
- Standard Triton V2 protocol compatibility
- Model lifecycle management (auto-load, LRU eviction, pin/cordon)
- Shared persistent state across restarts

## Components

| Component | Port | Protocol | Purpose |
|-----------|------|----------|---------|
| Triton Server | 8000/8001/8002 | HTTP/gRPC/Metrics | ML inference engine |
| gRPC Proxy | 50051 | Triton gRPC | Auth + model management |
| HTTP Proxy | 8080 | Triton V2 REST | Auth + model management |
| Admin API | 8000 | HTTP | Kubernetes management |

## Code Structure

```
src/
├── proxy/
│   ├── common/
│   │   ├── auth.py          # Domino auth interceptor
│   │   ├── config.py        # Environment configuration
│   │   └── state.py         # Model state + Triton control
│   ├── http/
│   │   ├── server.py        # FastAPI application
│   │   └── routes/          # V2 REST API + extensions
│   └── grpc/
│       ├── server.py        # gRPC server
│       └── servicer.py      # GRPCInferenceServiceServicer
└── admin/
    └── server.py            # Admin API for K8s
```

## Authentication

```
Request
  │
  ▼
Extract x-domino-api-key or Authorization header
  │
  ▼
Validate against Domino: GET /v4/auth/principal
  │
  ├─ Invalid ──► 401 Unauthorized
  │
  └─ Valid ──► Forward to Triton
```

## Model State Management

The proxies maintain shared state for model lifecycle:

```python
ModelState:
  - loaded: bool          # Currently in GPU memory
  - pinned: bool          # Protected from LRU eviction
  - cordoned: bool        # Reject new requests (maintenance)
  - last_used: timestamp  # For LRU tracking
```

**Auto-load**: Models load automatically on first inference request.

**LRU Eviction**: Idle models (default: 5 min) are unloaded to free GPU memory. Pinned models are protected.

**Cordon**: Cordoned models reject new requests, allowing graceful drain before maintenance.

State persists to `MODEL_STATE_DIR` (JSON files per model) and survives pod restarts.

## Triton V2 Protocol

Both proxies implement standard Triton protocol endpoints:

**HTTP Proxy:**
```
GET  /v2/health/live
GET  /v2/health/ready
POST /v2/models/{model}/infer
POST /v2/repository/models/{model}/load
POST /v2/repository/models/{model}/unload
```

**Extension Endpoints (HTTP only):**
```
GET  /v1/models/status             # Model status with LRU info
POST /v1/models/{model}/pin        # Pin/unpin model
POST /v1/models/{model}/cordon     # Cordon/uncordon model
```

## Client Usage

Standard `tritonclient` libraries work directly:

```python
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient("localhost:50051")
inputs = [grpcclient.InferInput("images", [1, 3, 640, 640], "FP32")]
inputs[0].set_data_from_numpy(tensor)

result = client.infer(
    "yolov8n",
    inputs,
    headers={"x-domino-api-key": api_key}
)
output = result.as_numpy("output0")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRITON_URL` | Triton gRPC endpoint | `127.0.0.1:8001` |
| `TRITON_HTTP_URL` | Triton HTTP endpoint | `http://127.0.0.1:8000` |
| `DOMINO_HOST` | Domino platform URL | Required |
| `SKIP_AUTH` | Disable auth (local dev) | `false` |
| `MODEL_STATE_DIR` | Persistent state directory | `/model-state` |
| `MODEL_IDLE_TIMEOUT_SECS` | LRU eviction timeout | `300` |

## Storage Layout

```
triton-repo/
├── models/              # Triton model repository
│   ├── yolov8n/
│   │   ├── config.pbtxt
│   │   └── 1/model.onnx
│   └── smollm-135m-python/
│       ├── config.pbtxt
│       └── 1/model.py
├── weights/             # HuggingFace model weights
│   └── smollm-135m-python/
└── model-state/         # Persistent model state (JSON)
```

## Deployment

Local (Docker Compose):
```bash
docker compose up --build -d
```

Kubernetes (Helm):
```bash
helm install domino-triton helm/domino-triton/ -n $NAMESPACE
```

See [helm-install.md](helm-install.md) for detailed Kubernetes deployment.
