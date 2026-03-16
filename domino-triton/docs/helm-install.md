# Helm Installation Guide

This guide provides comprehensive instructions for deploying the Triton Inference Server with gRPC and REST proxies on Kubernetes using Helm.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [S3 CSI Driver Setup](#s3-csi-driver-setup)
4. [Namespace Setup](#namespace-setup)
5. [Helm Chart Configuration](#helm-chart-configuration)
6. [Installation](#installation)
7. [Deployed Components](#deployed-components)
8. [Testing the Installation](#testing-the-installation)
9. [Admin API](#admin-api)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
                                    ┌─────────────────────────────────────┐
                                    │         Kubernetes Cluster          │
                                    │                                     │
   ┌──────────────┐                 │  ┌─────────────────────────────┐   │
   │   Domino     │                 │  │   domino-inference-{env}    │   │
   │  Workloads   │                 │  │        namespace            │   │
   │              │                 │  │                             │   │
   │  ┌────────┐  │    gRPC/REST    │  │  ┌───────────────────────┐ │   │
   │  │ Client │──┼─────────────────┼──┼─▶│   Proxy Pod           │ │   │
   │  └────────┘  │                 │  │  │  ┌─────────────────┐  │ │   │
   └──────────────┘                 │  │  │  │  gRPC Proxy     │  │ │   │
                                    │  │  │  │  :50051         │  │ │   │
                                    │  │  │  └────────┬────────┘  │ │   │
                                    │  │  │           │           │ │   │
                                    │  │  │  ┌────────▼────────┐  │ │   │
                                    │  │  │  │  REST Proxy     │  │ │   │
                                    │  │  │  │  :8080          │  │ │   │
                                    │  │  │  └────────┬────────┘  │ │   │
                                    │  │  └───────────┼───────────┘ │   │
                                    │  │              │             │   │
                                    │  │  ┌───────────▼───────────┐ │   │
                                    │  │  │   Triton Server       │ │   │
                                    │  │  │   :8000 (HTTP)        │ │   │
                                    │  │  │   :8001 (gRPC)        │ │   │
                                    │  │  │   :8002 (Metrics)     │ │   │
                                    │  │  └───────────┬───────────┘ │   │
                                    │  │              │             │   │
                                    │  │  ┌───────────▼───────────┐ │   │
                                    │  │  │   S3 Model Storage    │ │   │
                                    │  │  │   (via CSI Driver)    │ │   │
                                    │  │  └───────────────────────┘ │   │
                                    │  │                             │   │
                                    │  │  ┌───────────────────────┐ │   │
                                    │  │  │   Admin Service       │ │   │
                                    │  │  │   :8000               │ │   │
                                    │  │  └───────────────────────┘ │   │
                                    │  └─────────────────────────────┘   │
                                    └─────────────────────────────────────┘
```

### Components

| Component | Port | Purpose |
|-----------|------|---------|
| **Triton Server** | 8000, 8001, 8002 | ML inference engine |
| **gRPC Proxy** | 50051 | Binary protocol, high performance |
| **REST Proxy** | 8080 | JSON/HTTP, easy debugging |
| **Admin Service** | 8000 | Scaling, management operations |

---

## Prerequisites

### Required
- Kubernetes cluster (EKS recommended)
- Helm 3.x installed
- kubectl configured
- AWS CLI configured (for S3 setup)

### AWS Resources
- S3 bucket for model storage
- IAM role with S3 access
- OIDC provider configured for EKS

---

## S3 CSI Driver Setup

The S3 CSI driver mounts your S3 bucket as a volume in Kubernetes, allowing Triton to load models directly from S3.

### 1. Create IAM Policy

Create a policy named `domino-triton-s3-policy`:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::YOUR_BUCKET_NAME"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::YOUR_BUCKET_NAME/*"
            ]
        }
    ]
}
```

### 2. Create IAM Role

Create a role named `domino-triton-s3-role` with trust relationship:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/oidc.eks.REGION.amazonaws.com/id/OIDC_ID"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.REGION.amazonaws.com/id/OIDC_ID:aud": "sts.amazonaws.com",
                    "oidc.eks.REGION.amazonaws.com/id/OIDC_ID:sub": "system:serviceaccount:domino-platform:s3-csi-driver-sa"
                }
            }
        }
    ]
}
```

### 3. Install S3 CSI Driver

```bash
# Set variables
export AWS_ACCOUNT_ID=<AWS_ACCOUNT_ID>
export AWS_ROLE_ARN=arn:aws:iam::${AWS_ACCOUNT_ID}:role/domino-triton-s3-role
export PLATFORM_NAMESPACE=domino-platform
export S3_CSI_DRIVER_SA=s3-csi-driver-sa

# Add Helm repo
helm repo add aws-mountpoint-s3-csi-driver https://awslabs.github.io/mountpoint-s3-csi-driver
helm repo update

# Install driver (CPU nodes only)
helm upgrade --install aws-mountpoint-s3-csi-driver \
    --namespace ${PLATFORM_NAMESPACE} \
    --set node.serviceAccount.create=true \
    --set node.serviceAccount.name=${S3_CSI_DRIVER_SA} \
    --set node.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=${AWS_ROLE_ARN} \
    aws-mountpoint-s3-csi-driver/aws-mountpoint-s3-csi-driver
```

### 4. GPU Node Support (Required for GPU deployments)

**IMPORTANT:** The S3 CSI driver DaemonSet does NOT run on GPU-tainted nodes by default.
If your Triton pods will run on GPU nodes, you must add tolerations:

```bash
# Install/upgrade with GPU node tolerations
helm upgrade --install aws-mountpoint-s3-csi-driver \
    --namespace ${PLATFORM_NAMESPACE} \
    --set node.serviceAccount.create=true \
    --set node.serviceAccount.name=${S3_CSI_DRIVER_SA} \
    --set node.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=${AWS_ROLE_ARN} \
    --set 'node.tolerations[0].key=nvidia.com/gpu' \
    --set 'node.tolerations[0].operator=Exists' \
    --set 'node.tolerations[0].effect=NoSchedule' \
    aws-mountpoint-s3-csi-driver/aws-mountpoint-s3-csi-driver
```

Or use a values file (`s3-csi-values.yaml`):

```yaml
node:
  serviceAccount:
    create: true
    name: s3-csi-driver-sa
    annotations:
      eks.amazonaws.com/role-arn: "arn:aws:iam::ACCOUNT_ID:role/domino-triton-s3-role"
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

```bash
helm upgrade --install aws-mountpoint-s3-csi-driver \
    --namespace ${PLATFORM_NAMESPACE} \
    -f s3-csi-values.yaml \
    aws-mountpoint-s3-csi-driver/aws-mountpoint-s3-csi-driver
```

**Verify CSI driver pods run on GPU nodes:**
```bash
kubectl get pods -n ${PLATFORM_NAMESPACE} -l app.kubernetes.io/name=aws-mountpoint-s3-csi-driver -o wide
```

---

## Namespace Setup

Create a dedicated namespace for each environment (dev, test, prod):

```bash
# Set environment
export ENV=dev  # or test, prod
export NAMESPACE=domino-inference-${ENV}

# Create namespace
kubectl create namespace ${NAMESPACE}

# Required labels for Domino integration
kubectl label namespace ${NAMESPACE} domino-compute=true
kubectl label namespace ${NAMESPACE} domino-triton=true
```

---

## Helm Chart Configuration

### values.yaml Reference

```yaml
# =============================================================================
# Environment Configuration
# =============================================================================
env:
  # Domino compute namespace (for auth integration)
  compute_namespace: domino-compute

  # Target namespace for this installation
  namespace: domino-inference-dev

  # Node selector for GPU nodes
  node_selector: default-gpu

  # Resource allocation for Triton server
  memory: 24Gi
  cores: 1
  gpu: 1

  # Service name prefix (used for all resources)
  name: triton-inference-server

  # Label for resource identification
  type: domino-triton-inference

  # Enable claims-based authorization
  claim_based: true

# =============================================================================
# Proxy Configuration
# =============================================================================
proxy:
  # gRPC proxy image
  grpc_image: "quay.io/domino/domino-triton-grpc-proxy:v1"

  # REST proxy image
  rest_image: "quay.io/domino/domino-triton-rest-proxy:v1"

  # Number of proxy replicas
  replicas: 1

# =============================================================================
# Admin Service Configuration
# =============================================================================
admin:
  # Admin service image
  image: "quay.io/domino/domino-triton-admin:v1"

  # Number of admin replicas
  replicas: 1

  # Domino service auth key name
  domino_service_auth_key: "domino-service-auth"

# =============================================================================
# Triton Server Configuration
# =============================================================================
triton_inference_server:
  # Triton server image (use 24.10 or later for best compatibility)
  image: "nvcr.io/nvidia/tritonserver:24.10-py3"

  # Number of Triton replicas
  replicas: 1

# =============================================================================
# Storage Configuration
# =============================================================================
persistence:
  # S3 bucket name for model storage
  bucket: your-model-bucket

  # AWS region
  region: "us-west-2"

# =============================================================================
# Istio Configuration
# =============================================================================
istio:
  # Enable Istio service mesh integration
  enabled: false
```

### Configuration by Environment

| Parameter | Dev | Test | Prod |
|-----------|-----|------|------|
| `env.namespace` | domino-inference-dev | domino-inference-test | domino-inference-prod |
| `env.gpu` | 1 | 1 | 2+ |
| `env.memory` | 16Gi | 24Gi | 32Gi+ |
| `proxy.replicas` | 1 | 1 | 2+ |
| `triton_inference_server.replicas` | 1 | 1 | 2+ |

---

## Installation

### Install

```bash
export NAMESPACE=domino-inference-dev

helm install domino-triton helm/domino-triton/ \
    -n ${NAMESPACE} \
    -f helm/domino-triton/values-dev.yaml
```

### Upgrade

```bash
export NAMESPACE=domino-inference-dev

helm upgrade domino-triton helm/domino-triton/ \
    -n ${NAMESPACE} \
    -f helm/domino-triton/values-dev.yaml
```

### Uninstall

```bash
export NAMESPACE=domino-inference-dev

helm delete domino-triton -n ${NAMESPACE}
```

### Verify Installation

```bash
# Check all pods are running
kubectl get pods -n ${NAMESPACE}

# Expected output:
# NAME                                           READY   STATUS    RESTARTS   AGE
# triton-inference-server-xxx                    1/1     Running   0          5m
# triton-inference-server-proxy-xxx              2/2     Running   0          5m
# triton-inference-server-admin-xxx              1/1     Running   0          5m

# Check services
kubectl get svc -n ${NAMESPACE}

# Expected output:
# NAME                              TYPE        CLUSTER-IP       PORT(S)
# triton-inference-server           ClusterIP   10.100.x.x       8000,8001,8002
# triton-inference-server-proxy     ClusterIP   10.100.x.x       50051,8080
# triton-inference-server-admin     ClusterIP   10.100.x.x       8000
```

---

## Deployed Components

### Services Endpoints

| Service | DNS Name | Ports |
|---------|----------|-------|
| **Triton Server** | `triton-inference-server.{namespace}.svc.cluster.local` | 8000 (HTTP), 8001 (gRPC), 8002 (Metrics) |
| **Proxy (gRPC)** | `triton-inference-server-proxy.{namespace}.svc.cluster.local` | 50051 |
| **Proxy (REST)** | `triton-inference-server-proxy.{namespace}.svc.cluster.local` | 8080 |
| **Admin** | `triton-inference-server-admin.{namespace}.svc.cluster.local` | 8000 |

### S3 Model Path Structure

The S3 bucket is shared across multiple namespaces. Each namespace has its own isolated `triton-repo/` folder:

```
s3://YOUR_BUCKET/
├── domino-inference-dev/                <- Namespace 1
│   └── triton-repo/
│       ├── models/                      <- Triton model repository
│       │   ├── yolov8n/
│       │   │   ├── config.pbtxt
│       │   │   └── 1/model.onnx
│       │   ├── bert-base-uncased/
│       │   │   ├── config.pbtxt
│       │   │   └── 1/model.onnx
│       │   ├── whisper-tiny-python/
│       │   │   ├── config.pbtxt
│       │   │   └── 1/model.py
│       │   └── smollm-135m-python/
│       │       ├── config.pbtxt
│       │       └── 1/model.py
│       ├── weights/                     <- Model weights (HuggingFace)
│       │   └── smollm-135m-python/1/
│       │       ├── config.json
│       │       ├── model.safetensors
│       │       └── tokenizer.json
│       └── model-state/                 <- Persistent model state (JSON)
│           ├── yolov8n.json
│           └── bert-base-uncased.json
├── domino-inference-test/               <- Namespace 2
│   └── triton-repo/
│       ├── models/
│       ├── weights/
│       └── model-state/
└── domino-inference-prod/               <- Namespace 3
    └── triton-repo/
        ├── models/
        ├── weights/
        └── model-state/
```

**Key points:**
- Each namespace mounts only its own `triton-repo/` folder at `/triton-repo`
- All pods (Triton, gRPC proxy, REST proxy, Admin) share the same mount
- `model-state/` stores persistent state (pinned, cordoned status) across pod restarts
- Python backend models load weights from `weights/` using the `MODEL_REPO` env var

---

## Testing the Installation

### Prerequisites

Install required Python packages:

```bash
pip install -r docker/requirements-client.txt
```

### Environment Setup

```bash
# Set proxy addresses (from inside cluster)
export TRITON_GRPC_URL="triton-inference-server-proxy.domino-inference-dev.svc.cluster.local:50051"
export TRITON_REST_URL="http://triton-inference-server-proxy.domino-inference-dev.svc.cluster.local:8080"

# Set Domino API key for authentication
export DOMINO_USER_API_KEY=your-api-key
```

### Test 1: YOLOv8n Object Detection

**Using gRPC:**
```bash
python scripts/clients/yolov8n_video_grpc_client.py --video samples/video.avi --max-frames 5
```

**Using REST:**
```bash
python scripts/clients/yolov8n_video_rest_client.py --video samples/video.avi --max-frames 5
```

### Test 2: BERT Text Classification

**Using gRPC:**
```bash
python scripts/clients/bert_text_grpc_client.py --texts "This movie was fantastic!" "Terrible experience."
```

**Using REST:**
```bash
python scripts/clients/bert_text_rest_client.py --texts "This movie was fantastic!" "Terrible experience."
```

### Test 3: Whisper Audio Transcription

**Using gRPC:**
```bash
python scripts/clients/whisper_audio_grpc_client.py --audio samples/audio_sample.wav
```

**Using REST:**
```bash
python scripts/clients/whisper_audio_rest_client.py --audio samples/audio_sample.wav
```

### Test 4: LLM Text Generation

**Using gRPC:**
```bash
python scripts/clients/llm_text_grpc_client.py --prompt "What is the capital of France?"
```

**Using REST:**
```bash
python scripts/clients/llm_text_rest_client.py --prompt "What is the capital of France?"
```

### Test 5: Using Standard Triton Client Libraries

The proxies are compatible with standard `tritonclient` libraries:

```python
import tritonclient.grpc as grpcclient
import numpy as np

# Connect to gRPC proxy
client = grpcclient.InferenceServerClient(
    "triton-inference-server-proxy.domino-inference-dev.svc.cluster.local:50051"
)

# Build input tensor
inputs = [grpcclient.InferInput("images", [1, 3, 640, 640], "FP32")]
inputs[0].set_data_from_numpy(np.random.rand(1, 3, 640, 640).astype(np.float32))

# Run inference with auth header
result = client.infer(
    "yolov8n",
    inputs,
    headers={"x-domino-api-key": "your-api-key"}
)
print(result.as_numpy("output0").shape)
```

### Protocol Selection Guide

| Model Type | Recommended Protocol | Reason |
|------------|---------------------|--------|
| **YOLOv8n (Video)** | gRPC | 16x faster for streaming |
| **YOLOv8n (Single Image)** | Either | One-off request |
| **BERT (Text)** | gRPC | 1.2x faster |
| **Whisper (Audio)** | REST | Similar performance, simpler |

---

## Admin API

The Admin service provides management operations for the Triton deployment.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/deployments/inference-server/scale` | POST | Scale Triton replicas |
| `/v1/deployments/inference-server/scale` | GET | Get current scale |
| `/v1/deployments/inference-server/logs` | GET | Get Triton logs |
| `/healthz` | GET | Health check |

### Scale Triton Server

```bash
ADMIN_URL=http://triton-inference-server-admin.domino-inference-dev.svc.cluster.local:8000

# Scale up
curl -X POST "${ADMIN_URL}/v1/deployments/inference-server/scale" \
  -H "Content-Type: application/json" \
  -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" \
  -d '{"replicas": 2}'

# Get current scale
curl "${ADMIN_URL}/v1/deployments/inference-server/scale" \
  -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}"

# Get logs
curl "${ADMIN_URL}/v1/deployments/inference-server/logs?tail_lines=100" \
  -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}"
```

### Claims-Based Authorization

When `env.claim_based: true`, the admin service validates user claims:

- **Claim Service**: `triton-inference-server`
- **Claim Role**: `admin`

Users must have the appropriate Domino claims to perform admin operations. See [Claims-Based Authentication](claims-based-auth.md) for details.

---

## Troubleshooting

### Common Issues

#### 1. Pods not starting

```bash
# Check pod events
kubectl describe pod -n ${NAMESPACE} <pod-name>

# Check logs
kubectl logs -n ${NAMESPACE} <pod-name> -c triton
```

#### 2. Models not loading

```bash
# Check Triton logs for model errors
kubectl logs -n ${NAMESPACE} -l app=triton-inference-server -c triton | grep -i error

# Verify S3 mount
kubectl exec -n ${NAMESPACE} -it <triton-pod> -- ls -la /models
```

#### 3. Authentication failures

```bash
# Verify API key
curl -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" \
  "${TRITON_REST_URL}/healthz"

# Check proxy logs
kubectl logs -n ${NAMESPACE} -l app=triton-inference-server-proxy -c grpc-proxy
kubectl logs -n ${NAMESPACE} -l app=triton-inference-server-proxy -c rest-proxy
```

#### 4. GPU not detected

```bash
# Check GPU availability
kubectl describe node <gpu-node> | grep nvidia

# Verify Triton sees GPU
kubectl logs -n ${NAMESPACE} -l app=triton-inference-server | grep -i gpu
```

#### 5. S3 volume mount fails on GPU nodes

**Symptom:** Pod stuck in `ContainerCreating` with volume mount errors on GPU nodes.

**Cause:** The S3 CSI driver DaemonSet doesn't have tolerations for GPU-tainted nodes.

```bash
# Check if CSI driver pods run on GPU nodes
kubectl get pods -n domino-platform -l app.kubernetes.io/name=aws-mountpoint-s3-csi-driver -o wide

# Check for GPU node taints
kubectl get nodes -l nvidia.com/gpu.present=true -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.taints}{"\n"}{end}'
```

**Fix:** Reinstall S3 CSI driver with GPU tolerations (see [GPU Node Support](#4-gpu-node-support-required-for-gpu-deployments)).

### Health Checks

```bash
# Triton health
kubectl exec -n ${NAMESPACE} -it <triton-pod> -- \
  curl -s http://localhost:8000/v2/health/ready

# REST proxy health
curl ${TRITON_REST_URL}/v2/health/live

# List loaded models
curl ${TRITON_REST_URL}/v1/models/status \
  -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}"
```

### Logs Collection

```bash
# All component logs
kubectl logs -n ${NAMESPACE} -l type=domino-triton-inference --all-containers=true

# Triton server only
kubectl logs -n ${NAMESPACE} -l app=triton-inference-server -c triton

# gRPC proxy only
kubectl logs -n ${NAMESPACE} -l app=triton-inference-server-proxy -c grpc-proxy

# REST proxy only
kubectl logs -n ${NAMESPACE} -l app=triton-inference-server-proxy -c rest-proxy
```

---

## Appendix

### Port Reference

| Component | Port | Protocol | Purpose |
|-----------|------|----------|---------|
| Triton | 8000 | HTTP | REST API |
| Triton | 8001 | gRPC | gRPC API |
| Triton | 8002 | HTTP | Prometheus metrics |
| gRPC Proxy | 50051 | gRPC | Client-facing gRPC |
| REST Proxy | 8080 | HTTP | Client-facing REST |
| Admin | 8000 | HTTP | Management API |

### Environment Variables

| Variable | Component | Description |
|----------|-----------|-------------|
| `TRITON_URL` | gRPC Proxy | Triton gRPC address |
| `TRITON_HTTP_URL` | REST Proxy | Triton HTTP address |
| `DOMINO_HOST` | All Proxies | Domino auth endpoint |
| `INFERENCE_SERVER_DEPLOYMENT_NAME` | Admin | Triton deployment name |
| `CLAIM_BASED` | Admin | Enable claims auth |

### Image Versions

| Component | Image | Version |
|-----------|-------|---------|
| Triton Server | `nvcr.io/nvidia/tritonserver` | 24.10-py3 |
| gRPC Proxy | `quay.io/domino/domino-triton-grpc-proxy` | v1 |
| REST Proxy | `quay.io/domino/domino-triton-rest-proxy` | v1 |
| Admin | `quay.io/domino/domino-triton-admin` | v1 |
