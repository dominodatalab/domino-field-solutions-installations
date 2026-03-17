# Domino Triton Inference Server - System Architecture

## Overview

This document describes the architecture of the Domino-native Triton Inference Server deployment, focusing on the proxy-based authorization design and the benefits of decoupling inference engines from user-facing model endpoints.

---

## High-Level Architecture

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                    Domino Platform                          │
                                    │  ┌─────────────┐                                            │
                                    │  │   Domino    │                                            │
                                    │  │   Auth API  │◄──────────────────────┐                    │
                                    │  │  /v4/auth/  │                       │                    │
                                    │  └─────────────┘                       │                    │
                                    └────────────────────────────────────────┼────────────────────┘
                                                                             │
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                              Triton Inference Namespace                                       │
    │                                                                                              │
    │   ┌─────────────┐         ┌─────────────────────────────────────┐         ┌──────────────┐  │
    │   │             │         │         Authorization Proxy          │         │              │  │
    │   │   Domino    │  HTTP   │  ┌───────────┐    ┌───────────┐     │         │    Triton    │  │
    │   │  Workloads  │────────►│  │   HTTP    │    │   gRPC    │     │────────►│   Inference  │  │
    │   │  (Clients)  │  gRPC   │  │   Proxy   │    │   Proxy   │     │         │    Server    │  │
    │   │             │────────►│  │  :8080    │    │  :50051   │     │         │  :8000/8001  │  │
    │   └─────────────┘         │  └─────┬─────┘    └─────┬─────┘     │         └──────────────┘  │
    │                           │        │                │           │                           │
    │                           │        └───────┬────────┘           │                           │
    │                           │                │                    │                           │
    │                           │        ┌───────▼────────┐           │                           │
    │                           │        │  State Pod     │           │                           │
    │                           │        │  (Redis)       │           │                           │
    │                           │        │  LRU, Pin/Cord │           │                           │
    │                           │        └────────────────┘           │                           │
    │                           └─────────────────────────────────────┘                           │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Proxy-Based Authorization Design

### Authentication Flow

```
    Client Request                    Authorization Proxy                 Domino Auth              Triton
         │                                   │                               │                       │
         │  1. Request + API Key             │                               │                       │
         │──────────────────────────────────►│                               │                       │
         │                                   │                               │                       │
         │                                   │  2. Validate API Key          │                       │
         │                                   │──────────────────────────────►│                       │
         │                                   │                               │                       │
         │                                   │  3. User Principal            │                       │
         │                                   │◄──────────────────────────────│                       │
         │                                   │                               │                       │
         │                                   │  4. Forward Request                                   │
         │                                   │──────────────────────────────────────────────────────►│
         │                                   │                                                       │
         │                                   │  5. Inference Result                                  │
         │                                   │◄──────────────────────────────────────────────────────│
         │                                   │                               │                       │
         │  6. Response                      │                               │                       │
         │◄──────────────────────────────────│                               │                       │
         │                                   │                               │                       │
```

### Key Components

| Component | Port | Responsibility |
|-----------|------|----------------|
| **HTTP Proxy** | 8080 | REST API auth, model management, Triton V2 protocol |
| **gRPC Proxy** | 50051 | gRPC auth, streaming inference, Triton protocol |
| **State Pod** | 6379 | Shared model state (Redis) for horizontal scaling |
| **Triton Server** | 8000/8001 | ML inference engine (isolated from external access) |
| **Domino Auth** | Platform | API key validation, user identity |

### Authorization Headers

```
┌─────────────────────────────────────────────────────────────────┐
│                     Supported Auth Methods                       │
├─────────────────────────────────────────────────────────────────┤
│  Header: X-Domino-Api-Key: <api-key>                            │
│  Header: Authorization: Bearer <api-key>                         │
│  gRPC Metadata: x-domino-api-key: <api-key>                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Lifecycle Management

The proxy layer adds intelligent model management capabilities:

```
                          Model State Machine

    ┌─────────────┐     load      ┌─────────────┐     unload    ┌─────────────┐
    │             │──────────────►│             │───────────────►│             │
    │  UNLOADED   │               │   LOADED    │                │  UNLOADED   │
    │             │◄──────────────│             │◄───────────────│             │
    └─────────────┘   LRU evict   └──────┬──────┘   manual       └─────────────┘
                                         │
                      ┌──────────────────┼──────────────────┐
                      │                  │                  │
                      ▼                  ▼                  ▼
               ┌───────────┐      ┌───────────┐      ┌───────────┐
               │  PINNED   │      │  NORMAL   │      │ CORDONED  │
               │(no evict) │      │           │      │(no new    │
               └───────────┘      └───────────┘      │ requests) │
                                                     └───────────┘
```

### Auto-Load on First Request

```
    Client                     Proxy                      Triton
       │                         │                           │
       │  1. Infer Request       │                           │
       │────────────────────────►│                           │
       │                         │                           │
       │                         │  2. Check Model Status    │
       │                         │──────────────────────────►│
       │                         │                           │
       │                         │  3. Model Not Loaded      │
       │                         │◄──────────────────────────│
       │                         │                           │
       │                         │  4. Load Model            │
       │                         │──────────────────────────►│
       │                         │                           │
       │                         │  5. Model Ready           │
       │                         │◄──────────────────────────│
       │                         │                           │
       │                         │  6. Forward Inference     │
       │                         │──────────────────────────►│
       │                         │                           │
       │  7. Result              │                           │
       │◄────────────────────────│◄──────────────────────────│
       │                         │                           │
```

---

## Benefits of This Architecture

### 1. Domino-Native Integration

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Traditional Approach                                │
│                                                                            │
│   ┌─────────────┐                              ┌─────────────┐             │
│   │   Domino    │     No Auth Integration      │   Triton    │             │
│   │  Workload   │─────────────────────────────►│   Server    │             │
│   └─────────────┘     (Security Gap)           └─────────────┘             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         Domino-Native Approach                              │
│                                                                            │
│   ┌─────────────┐    ┌─────────────┐           ┌─────────────┐             │
│   │   Domino    │    │    Auth     │           │   Triton    │             │
│   │  Workload   │───►│   Proxy     │──────────►│   Server    │             │
│   └─────────────┘    └──────┬──────┘           └─────────────┘             │
│                             │                                              │
│                      ┌──────▼──────┐                                       │
│                      │   Domino    │                                       │
│                      │  Auth API   │                                       │
│                      └─────────────┘                                       │
│                                                                            │
│   Benefits:                                                                │
│   ✓ Same API keys used across all Domino services                         │
│   ✓ User identity tracked for audit/billing                               │
│   ✓ Consistent access control with Domino RBAC                            │
│   ✓ No separate credential management for inference                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2. Separation of Inference Engine from User Endpoint

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Decoupled Architecture Benefits                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      User-Facing Layer                               │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│   │  │ HTTP Proxy  │  │ gRPC Proxy  │  │ Admin API   │                  │  │
│   │  │             │  │             │  │             │                  │  │
│   │  │ • Auth      │  │ • Auth      │  │ • Model Ops │                  │  │
│   │  │ • Rate Limit│  │ • Streaming │  │ • Metrics   │                  │  │
│   │  │ • Routing   │  │ • Load Bal  │  │ • Dashboard │                  │  │
│   │  └─────────────┘  └─────────────┘  └─────────────┘                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                          ┌─────────▼─────────┐                             │
│                          │   Internal Only   │                             │
│                          └─────────┬─────────┘                             │
│                                    │                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Inference Engine Layer                          │  │
│   │  ┌─────────────────────────────────────────────────────────────┐    │  │
│   │  │                    NVIDIA Triton Server                      │    │  │
│   │  │                                                              │    │  │
│   │  │  • Pure inference focus (no auth overhead)                   │    │  │
│   │  │  • Optimal GPU utilization                                   │    │  │
│   │  │  • Native batching & scheduling                              │    │  │
│   │  │  • Multiple backend support (ONNX, TensorRT, Python, etc.)   │    │  │
│   │  │                                                              │    │  │
│   │  └─────────────────────────────────────────────────────────────┘    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

Key Benefits:

┌──────────────────────┬─────────────────────────────────────────────────────┐
│ Benefit              │ Description                                         │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ Security Isolation   │ Triton server never exposed directly; all access    │
│                      │ goes through authenticated proxy layer              │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ Independent Scaling  │ Scale proxies and inference servers independently   │
│                      │ based on auth load vs. GPU compute load             │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ Protocol Flexibility │ Add new protocols (WebSocket, custom) without       │
│                      │ modifying the inference engine                      │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ Upgradability        │ Upgrade Triton version without changing user API;   │
│                      │ proxy provides stable interface                     │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ Observability        │ Centralized logging, metrics, and tracing at        │
│                      │ proxy layer without Triton modifications            │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ Multi-Tenancy        │ Proxy can route requests to different Triton        │
│                      │ instances based on user/project/priority            │
└──────────────────────┴─────────────────────────────────────────────────────┘
```

### 3. Intelligent Resource Management

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      GPU Memory Management                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Without Proxy (Static Loading):                                          │
│   ┌─────────────────────────────────────────────────────────────────┐     │
│   │ GPU Memory                                                       │     │
│   │ ████████████████████████████████████████████░░░░░░░░░░░░░░░░░░░ │     │
│   │ Model A    Model B    Model C    Model D      (Wasted Space)    │     │
│   │ (unused)   (unused)   (active)   (unused)                       │     │
│   └─────────────────────────────────────────────────────────────────┘     │
│                                                                            │
│   With Proxy (Dynamic LRU via Redis State):                                │
│   ┌─────────────────────────────────────────────────────────────────┐     │
│   │ GPU Memory                                                       │     │
│   │ ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │     │
│   │ Model C    (Available)                                          │     │
│   │ (active)   Load on demand, evict when idle                      │     │
│   └─────────────────────────────────────────────────────────────────┘     │
│                                                                            │
│   Features:                                                                │
│   • Auto-load: Models loaded on first inference request                   │
│   • LRU Eviction: Idle models (default 5 min) automatically unloaded      │
│   • Pinning: Critical models protected from eviction                      │
│   • Cordoning: Graceful drain for maintenance                             │
│   • Horizontal Scaling: State shared via Redis across proxy replicas      │
│   • Fire-and-Forget: Access tracking adds no inference latency            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Deployment                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Namespace: domino-inference                                              │
│                                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Service: triton-proxy (LoadBalancer/ClusterIP)                      │  │
│   │ Ports: 8080 (HTTP), 50051 (gRPC)                                    │  │
│   └───────────────────────────────────┬─────────────────────────────────┘  │
│                                       │                                    │
│   ┌───────────────────────────────────▼─────────────────────────────────┐  │
│   │ Deployment: proxy (replicas: N)                                     │  │
│   │ ┌─────────────┐ ┌─────────────┐                                     │  │
│   │ │ http-proxy  │ │ grpc-proxy  │  ◄── Horizontally scalable         │  │
│   │ │  container  │ │  container  │                                     │  │
│   │ └──────┬──────┘ └──────┬──────┘                                     │  │
│   │        │               │                                            │  │
│   │        └───────┬───────┘                                            │  │
│   │                │                                                    │  │
│   │        ┌───────▼───────┐                                            │  │
│   │        │  State Pod    │                                            │  │
│   │        │  (Redis)      │  ◄── Shared state backend                  │  │
│   │        └───────┬───────┘                                            │  │
│   │                │                                                    │  │
│   │        ┌───────▼───────┐                                            │  │
│   │        │    Triton     │                                            │  │
│   │        │    Server     │  ◄── ML inference engine                   │  │
│   │        └───────────────┘                                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│   Volumes:                                                                 │
│   • model-repository (S3/PVC) - Triton model configs                      │
│   • model-weights (S3/PVC) - HuggingFace weights                          │
│                                                                            │
│   State Pod: Redis with no persistence (syncs from Triton on startup)     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Client Integration

### Standard Triton Clients Work Unchanged

```python
# Python gRPC Client
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(
    url="triton-proxy.domino-inference:50051"
)

# Just add Domino auth header
headers = {"x-domino-api-key": os.environ["DOMINO_USER_API_KEY"]}

result = client.infer(
    model_name="yolov8n",
    inputs=[input_tensor],
    headers=headers
)
```

```python
# Python HTTP Client
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(
    url="triton-proxy.domino-inference:8080"
)

headers = {"X-Domino-Api-Key": os.environ["DOMINO_USER_API_KEY"]}

result = client.infer(
    model_name="bert-base-uncased",
    inputs=[input_tensor],
    headers=headers
)
```

---

## Summary

| Aspect | Traditional | Domino-Native with Proxy |
|--------|-------------|--------------------------|
| **Authentication** | None or custom | Domino API keys (unified) |
| **User Tracking** | Not available | Full audit trail per user |
| **Model Loading** | Manual or all-at-once | Automatic on-demand |
| **GPU Efficiency** | Static allocation | Dynamic LRU management |
| **Horizontal Scaling** | Single instance | Multi-replica via Redis state |
| **Protocol Support** | Triton native only | Extensible via proxy |
| **Upgrades** | Breaking changes | Proxy abstracts changes |
| **Security** | Direct exposure | Isolated behind proxy |
| **Observability** | Limited | Centralized at proxy |

The proxy-based architecture transforms Triton from a standalone inference engine into a fully integrated, Domino-native service with enterprise-grade authentication, intelligent resource management, horizontal scaling, and operational flexibility.