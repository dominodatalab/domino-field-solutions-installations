# Domino Triton Inference Server
## Technical Architecture Deep-Dive

**Audience**: Product/Engineering
**Date**: March 2026

---

## Executive Summary

Domino Triton Inference Server provides a **secure, high-performance ML inference layer** for enterprise deployments. This presentation covers three critical architectural decisions:

1. **Unified Inference** — Why separating models from endpoints reduces cost and complexity
2. **Why gRPC over REST** — Up to 30x performance improvement for image/video workloads
3. **GPU Sharing Strategy** — Maximizing hardware utilization across multiple models

---

## Agenda

1. Architecture Overview
2. **Unified Inference: Why Separate Models from Endpoints**
3. Protocol Selection: gRPC vs REST
4. The 30x Performance Question
5. GPU Sharing Strategies
6. Recommendations

---

# Part 1: Architecture Overview

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Domino Platform                               │
│  ┌─────────────────┐         ┌─────────────────┐                │
│  │  Data Science   │         │  Production     │                │
│  │  Workspaces     │         │  Applications   │                │
│  └────────┬────────┘         └────────┬────────┘                │
└───────────┼───────────────────────────┼─────────────────────────┘
            │                           │
            │ Authenticated Requests    │
            ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Inference Namespace                              │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ gRPC Proxy   │    │ REST Proxy   │    │ Admin API        │  │
│  │ :50051       │    │ :8080        │    │ :8000            │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────────┘  │
│         │                   │                                   │
│         └─────────┬─────────┘                                   │
│                   ▼                                             │
│         ┌──────────────────┐                                    │
│         │ NVIDIA Triton    │◄──── S3 Model Repository           │
│         │ Inference Server │◄──── EBS HuggingFace Cache         │
│         │ (GPU/CPU)        │                                    │
│         └──────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported Model Types

| Model | Type | Backend | Use Case |
|-------|------|---------|----------|
| **YOLOv8n** | Object Detection | ONNX | Real-time video analysis |
| **BERT** | Text Classification | ONNX | Sentiment, NLP tasks |
| **Whisper** | Speech-to-Text | Python | Audio transcription |
| **SmolLM** | Text Generation | Python | LLM inference |

**Key Point**: Single infrastructure serves diverse ML workloads

---

# Part 2: Unified Inference Architecture

---

## The Traditional Approach: Model = Endpoint

Most organizations start with per-model deployments:

```
┌─────────────────────────────────────────────────────────────┐
│               Traditional Model Deployment                   │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ YOLOv8 Service  │  │ BERT Service    │  │ LLM Service │ │
│  │ ─────────────── │  │ ─────────────── │  │ ─────────── │ │
│  │ Pod + Container │  │ Pod + Container │  │ Pod + GPU   │ │
│  │ Ingress Route   │  │ Ingress Route   │  │ Ingress     │ │
│  │ HPA + Scaling   │  │ HPA + Scaling   │  │ HPA         │ │
│  │ Monitoring      │  │ Monitoring      │  │ Monitoring  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           ▼                    ▼                   ▼        │
│    /yolo/predict        /bert/predict       /llm/generate   │
└─────────────────────────────────────────────────────────────┘

Result: 3 models = 3 deployments = 3× operational overhead
```

---

## Problems with Model-per-Endpoint

| Problem | Impact |
|---------|--------|
| **GPU fragmentation** | Each service claims its own GPU(s), even when idle |
| **Operational overhead** | N models = N deployments to manage, monitor, update |
| **Scaling inefficiency** | HPA scales entire pods, not model instances |
| **Cold starts** | Each service has independent startup/warmup |
| **Resource waste** | Small models claim full GPU allocation |

**Cost Example**: 10 models × 1 GPU each = 10 GPUs
With 15% average utilization = **8.5 GPUs wasted**

---

## The Unified Approach: Decouple Models from Endpoints

Triton separates the **inference plane** from **model management**:

```
┌─────────────────────────────────────────────────────────────┐
│                 Unified Inference Architecture               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Single Inference Endpoint               │   │
│  │  ─────────────────────────────────────────────────  │   │
│  │  POST /v2/models/{model_name}/infer                 │   │
│  │                                                      │   │
│  │  • Auth proxy (API key / JWT)                       │   │
│  │  • Request routing                                  │   │
│  │  • Load balancing                                   │   │
│  │  • Protocol translation (gRPC ↔ HTTP)               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Triton Inference Server                 │   │
│  │  ─────────────────────────────────────────────────  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ YOLOv8n │ │ BERT    │ │ Whisper │ │ SmolLM  │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Result: N models = 1 deployment = 1× operational overhead
```

---

## Benefits of Unified Inference

| Benefit | How |
|---------|-----|
| **GPU consolidation** | Multiple models share GPU memory pool |
| **Dynamic loading** | Models load on-demand, unload when idle |
| **Single control plane** | One API for all models (load, unload, status) |
| **Consistent auth** | Same auth layer for all model access |
| **Simplified monitoring** | One set of metrics, logs, alerts |
| **Atomic updates** | Update model config without redeploying infrastructure |

**Cost Example**: 10 models on 2 GPUs with LRU = **8 GPUs saved**

---

## Model Endpoint Separation in Practice

```
┌────────────────────────────────────────────────────────────┐
│                     Client Perspective                      │
│                                                             │
│  Before (per-model):                                        │
│  ────────────────────                                       │
│  POST https://yolo.company.com/predict                     │
│  POST https://bert.company.com/classify                    │
│  POST https://llm.company.com/generate                     │
│                                                             │
│  After (unified):                                           │
│  ────────────────                                           │
│  POST https://inference.company.com/v2/models/yolov8n/infer │
│  POST https://inference.company.com/v2/models/bert/infer    │
│  POST https://inference.company.com/v2/models/smollm/infer  │
│                                                             │
│  Same endpoint, same auth, same client library              │
└────────────────────────────────────────────────────────────┘
```

---

## Operational Comparison

| Aspect | Per-Model Deployment | Unified Inference |
|--------|---------------------|-------------------|
| Add new model | Create deployment, service, ingress, HPA | Drop config in S3, call load API |
| Remove model | Delete all K8s resources | Call unload API |
| Update model | Rolling deployment | Hot-reload via API |
| Monitor health | N dashboards | 1 dashboard |
| Scale a model | Scale entire deployment | Increase instance count |

---

## Per-Model Scaling Without Infrastructure Changes

Scale individual models to meet demand — no new deployments needed:

```
┌─────────────────────────────────────────────────────────────┐
│              Instance Groups: Per-Model Scaling              │
│                                                              │
│  config.pbtxt (before):          config.pbtxt (after):      │
│  ──────────────────────          ─────────────────────      │
│  instance_group [                instance_group [           │
│    { count: 1, kind: KIND_GPU }    { count: 4, kind: KIND_GPU }
│  ]                               ]                          │
│                                                              │
│  ┌─────────┐                     ┌─────────┐ ┌─────────┐   │
│  │ YOLOv8  │        ───►         │ YOLOv8  │ │ YOLOv8  │   │
│  │ inst 1  │                     │ inst 1  │ │ inst 2  │   │
│  └─────────┘                     ├─────────┤ ├─────────┤   │
│                                  │ YOLOv8  │ │ YOLOv8  │   │
│  Throughput: 20 req/s            │ inst 3  │ │ inst 4  │   │
│                                  └─────────┘ └─────────┘   │
│                                                              │
│                                  Throughput: 80 req/s       │
└─────────────────────────────────────────────────────────────┘

No deployment changes. No new pods. Just update config and reload.
```

---

## No Dedicated Infrastructure Per Model

**Traditional approach requires committed resources per model:**

```
┌────────────────────────────────────────────────────────────┐
│  Per-Model Infrastructure (committed, always running)       │
│                                                             │
│  YOLOv8:   [Pod] [GPU] [PVC] [Service] [Ingress] [HPA]    │
│  BERT:     [Pod] [GPU] [PVC] [Service] [Ingress] [HPA]    │
│  Whisper:  [Pod] [GPU] [PVC] [Service] [Ingress] [HPA]    │
│  SmolLM:   [Pod] [GPU] [PVC] [Service] [Ingress] [HPA]    │
│                                                             │
│  Total: 4 GPUs committed even if models rarely used        │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Unified Infrastructure (shared, on-demand)                 │
│                                                             │
│  Triton:   [Pod] [GPU] [S3 Mount] [Service] [Ingress]     │
│                                                             │
│  Models loaded on first request, unloaded when idle:       │
│  • YOLOv8  → loaded (in use)                               │
│  • BERT    → loaded (in use)                               │
│  • Whisper → unloaded (idle 10 min)                        │
│  • SmolLM  → unloaded (idle 30 min)                        │
│                                                             │
│  Total: 1 GPU, 2 models active, 2 models ready to load    │
└────────────────────────────────────────────────────────────┘
```
| Scale inference | Scale each deployment | Scale Triton replicas |
| GPU allocation | Fixed per deployment | Dynamic per request |

---

# Part 3: Protocol Selection

---

## The Protocol Question

> "Why not just use REST? Everyone knows REST."

Two options for ML inference APIs:

| | REST + JSON | gRPC + Protobuf |
|---|-------------|-----------------|
| **Format** | Text (JSON) | Binary |
| **Transport** | HTTP/1.1 | HTTP/2 |
| **Streaming** | Polling/WebSocket | Native bidirectional |
| **Schema** | Optional (OpenAPI) | Required (.proto) |
| **Adoption** | Universal | Growing (Google, Netflix, Uber) |

---

## When REST Works Fine

REST is **acceptable** when inference time dominates:

| Model | Payload | Inference Time | Protocol Overhead |
|-------|---------|----------------|-------------------|
| BERT | ~50 KB | 20-40ms | <5% of total |
| Whisper | ~1.2 MB | 10-15s | <1% of total |
| SmolLM | ~1 KB | 1-5s | <0.1% of total |

**For text and audio**: Use whichever protocol fits your client ecosystem

---

## When REST Breaks Down

REST becomes a **bottleneck** for large tensor payloads:

| Model | Payload Size | Transfer Dominates |
|-------|--------------|-------------------|
| YOLOv8n | 5 MB/frame | Yes |
| Video pipeline | 5 MB × 30 FPS = 150 MB/s | Absolutely |
| Batch inference | N × payload | Scales linearly |

**For images and video**: Protocol choice determines throughput ceiling

---

## Three Protocols for Video Inference

The dashboard supports three protocol options for video/image workloads:

| Protocol | Encoding | Payload Size | Use Case |
|----------|----------|--------------|----------|
| **REST (JSON)** | Text arrays | ~17.5 MB/frame | Debugging, compatibility |
| **REST (Binary)** | Binary tensor | ~5 MB/frame | Short videos, HTTP infra |
| **gRPC** | Binary protobuf | ~5 MB/frame | Production, long videos |

---

## REST (JSON) — Maximum Compatibility

**How it works**: Tensor data encoded as JSON arrays of float values

```json
{
  "inputs": [{
    "name": "images",
    "shape": [1, 3, 640, 640],
    "datatype": "FP32",
    "data": [0.1234, 0.5678, 0.9012, ...]  // 1.2M floats as text
  }]
}
```

**Pros**:
- Human-readable (great for debugging)
- Works with any HTTP client (curl, Postman, browser)
- No special libraries required

**Cons**:
- 3.5× larger payload (text vs binary)
- CPU-intensive encoding/decoding (1.2M float↔string conversions)
- **1-3 FPS for video** — not viable for real-time

**Best for**: Debugging, one-off tests, environments where only HTTP/1.1 is available

---

## REST (Binary) — Balanced Approach

**How it works**: HTTP with binary tensor payload (Triton's binary extension)

```
POST /v2/models/yolov8n/infer
Content-Type: application/octet-stream

[Binary tensor data: 5 MB]
```

**Pros**:
- Same payload size as gRPC (~5 MB)
- Uses existing HTTP infrastructure (proxies, load balancers)
- No JSON parsing overhead
- **5-8 FPS for video** — acceptable for short clips

**Cons**:
- Still HTTP/1.1 (no multiplexing, new connection per request)
- Slightly higher latency than gRPC
- Performance degrades on longer videos

**Best for**: Short videos (<30 frames), single images, HTTP-only environments

---

## gRPC — Maximum Performance

**How it works**: HTTP/2 with Protocol Buffers and persistent connections

```
┌─────────────────────────────────────────────────────────┐
│  gRPC Channel (persistent HTTP/2 connection)            │
│                                                         │
│  Frame 1 ──► [5MB binary] ──► Response                 │
│  Frame 2 ──► [5MB binary] ──► Response   (multiplexed) │
│  Frame 3 ──► [5MB binary] ──► Response                 │
└─────────────────────────────────────────────────────────┘
```

**Pros**:
- Persistent connection (no TCP handshake per request)
- HTTP/2 multiplexing (parallel requests on single connection)
- Binary protobuf (zero-copy tensor transfer)
- **20+ FPS for video** — production-ready

**Cons**:
- Requires gRPC-capable client library
- HTTP/2 may need special proxy configuration
- Slightly more complex client setup

**Best for**: Production video pipelines, long videos, high-throughput scenarios

---

## Protocol Performance: Video Length Matters

Performance gap **widens** as video length increases:

| Video Length | REST (JSON) | REST (Binary) | gRPC | Winner |
|--------------|-------------|---------------|------|--------|
| 1 frame | 300ms | 80ms | 60ms | gRPC (marginal) |
| 10 frames | 3s | 1.2s | 0.5s | gRPC (2.4×) |
| 100 frames | 45s | 15s | 5s | gRPC (3×) |
| 1000 frames | 10min+ | 3min | 50s | **gRPC (3.6×)** |

**Key insight**: REST (Binary) is acceptable for short videos, but gRPC's persistent connection advantage compounds over longer sequences.

---

## Why gRPC Wins for Long Videos

```
REST (Binary) - 100 frames:
┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐
│TCP   │ │TCP   │ │TCP   │ ... │TCP   │  100 connections
│Setup │ │Setup │ │Setup │     │Setup │  100 × 20ms overhead
└──┬───┘ └──┬───┘ └──┬───┘     └──┬───┘
   │        │        │            │
   ▼        ▼        ▼            ▼
[Frame 1][Frame 2][Frame 3]...[Frame 100]

Total overhead: ~2 seconds just for connection setup

gRPC - 100 frames:
┌─────────────────────────────────────────────────┐
│ Single persistent HTTP/2 connection              │
│                                                  │
│ [F1][F2][F3][F4]...[F100]  ◄── Multiplexed      │
└─────────────────────────────────────────────────┘

Total overhead: ~20ms (one-time setup)
```

---

## Protocol Selection Summary for Video

```
                ┌─────────────────────────────┐
                │    Video Inference Task     │
                └──────────────┬──────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
   │ Debugging?    │   │ Short video   │   │ Production    │
   │ One-off test? │   │ (<30 frames)? │   │ Long video?   │
   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
           │                   │                   │
           ▼                   ▼                   ▼
   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
   │ REST (JSON)   │   │ REST (Binary) │   │ gRPC          │
   │ 1-3 FPS       │   │ 5-8 FPS       │   │ 20+ FPS       │
   │ Easy debug    │   │ Good balance  │   │ Best perf     │
   └───────────────┘   └───────────────┘   └───────────────┘
```

---

# Part 4: The 30x Performance Question

---

## Observed Benchmark Results

YOLOv8n object detection, 640×640 images:

| Protocol | Encoding | Payload | Throughput | Relative |
|----------|----------|---------|------------|----------|
| **gRPC** | Binary | 5 MB | **20 FPS** | 1.0× |
| REST | JSON | 17.5 MB | 1-3 FPS | 0.05-0.15× |

**Question**: If JSON payload is only 3.5x larger, why is performance 20-30x worse?

---

## Answer: It's Not Just Transfer Time

The payload size comparison is misleading. The real cost is **CPU processing**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Request Lifecycle                             │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Encode   │───►│ Transfer │───►│ Decode   │───►│ Inference│  │
│  │ (Client) │    │ (Network)│    │ (Server) │    │ (GPU)    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                  │
│  gRPC:  ~0ms        ~5ms           ~0ms           30ms          │
│  JSON:  ~50ms       ~18ms          ~50ms          30ms          │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Math Behind 30x

For a single YOLOv8 frame (1×3×640×640 = **1,228,800 floats**):

| Operation | gRPC Binary | REST JSON |
|-----------|-------------|-----------|
| **Encode** | memcpy (µs) | 1.2M float→string conversions |
| **Transfer** | 5 MB | 17.5 MB |
| **Decode** | pointer cast (µs) | 1.2M string→float parsing |
| **Memory** | 1 allocation | 1.2M+ allocations |

**Combined slowdown**: 3.5× network × 10× parsing = **~35× total**

---

## Why JSON Parsing Is Expensive

Each float value like `0.12345678` requires:

```
JSON String:  "0.12345678"
              │││││││││││
              ▼▼▼▼▼▼▼▼▼▼▼

1. Allocate string buffer
2. Read ASCII character '0'
3. Read ASCII character '.'
4. Read ASCII character '1'
5. ... (continue for each digit)
6. Track decimal position
7. Multiply/divide to build float
8. Handle edge cases (scientific notation, NaN, Inf)
9. Store result
10. Free string buffer
```

**Binary (gRPC)**: Read 4 bytes. Done. IEEE 754 format is CPU-native.

---

## Memory Allocation Comparison

| Protocol | Allocations per Frame | Memory Pattern |
|----------|----------------------|----------------|
| gRPC | 1 | Single contiguous buffer |
| JSON | 1,228,800+ | Fragmented heap allocations |

**Impact**:
- Heap fragmentation
- Cache misses
- GC pressure (Python/Java clients)
- Memory bandwidth saturation

---

## Visual: Where Time Goes

```
gRPC Request (50ms total)
├─ Encode:     ████ 1ms
├─ Network:    ████████████████ 15ms
├─ Decode:     ████ 1ms
└─ Inference:  ████████████████████████████████ 33ms
                                               ▲
                                          GPU-bound

JSON Request (1000ms+ total)
├─ Encode:     ████████████████████████████████████████ 200ms
├─ Network:    ████████████████████████████████████████████████████ 250ms
├─ Decode:     ████████████████████████████████████████ 200ms
└─ Inference:  ██████████ 33ms
               ▲
          CPU-bound (wasted GPU cycles)
```

---

## The Business Impact

| Scenario | gRPC | REST JSON | Cost Difference |
|----------|------|-----------|-----------------|
| 1 camera @ 20 FPS | 1 GPU | 20+ GPUs | 20× infrastructure |
| 10 cameras | 1 GPU | 200+ GPUs | Infeasible |
| Batch (1000 images) | 50s | 1000s+ | 20× latency |

**For image/video workloads, REST is not a viable option at scale.**

---

## Decision Framework

```
                    ┌─────────────────────┐
                    │ What's your payload?│
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │ Text/Audio      │               │ Images/Video    │
    │ (< 1MB)         │               │ (> 1MB)         │
    └────────┬────────┘               └────────┬────────┘
             │                                  │
             ▼                                  ▼
    ┌─────────────────┐               ┌─────────────────┐
    │ Either protocol │               │ gRPC required   │
    │ (inference-     │               │ (transport-     │
    │  bound)         │               │  bound)         │
    └─────────────────┘               └─────────────────┘
```

---

# Part 5: GPU Sharing Strategies

---

## The GPU Utilization Problem

**Enterprise ML platforms face a fundamental challenge**: GPU resources are expensive and often underutilized.

Typical single-model deployment:

```
┌────────────────────────────────────────────────────────────┐
│                      GPU (A100 80GB)                        │
│                                                             │
│  ┌─────────────┐                                           │
│  │ YOLOv8n     │  Used: 2GB                                │
│  │ Model       │  Utilization: 5-15% (bursty)              │
│  └─────────────┘                                           │
│                                                             │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                    Wasted: 78GB                             │
└────────────────────────────────────────────────────────────┘
```

**Problem**: $30K+ GPU sitting 85% idle

---

## Industry Approaches to GPU Sharing

Three primary strategies exist for sharing GPU resources across workloads:

| Approach | How It Works | Best For |
|----------|--------------|----------|
| **MIG** | Hardware partitions | Guaranteed isolation |
| **Time-Sharing** | Kubernetes operators | Fair resource division |
| **Multi-Model Serving** | Multiple models per GPU | Maximum flexibility |

Let's examine each approach's trade-offs.

---

## Approach 1: NVIDIA MIG (Multi-Instance GPU)

**How it works**: Hardware-level GPU partitioning (A100, H100 only)

```
┌────────────────────────────────────────────────────────────┐
│                  A100 80GB with MIG Enabled                 │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ MIG 1g.10gb │ │ MIG 1g.10gb │ │ MIG 1g.10gb │          │
│  │ Instance 1  │ │ Instance 2  │ │ Instance 3  │          │
│  │ (YOLOv8n)   │ │ (BERT)      │ │ (idle)      │ ◄── WASTE│
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ MIG 1g.10gb │ │ MIG 1g.10gb │ │ MIG 2g.20gb │          │
│  │ Instance 4  │ │ Instance 5  │ │ Instance 6  │          │
│  │ (Whisper)   │ │ (idle)      │ │ (SmolLM)    │ ◄── WASTE│
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└────────────────────────────────────────────────────────────┘
```

---

## MIG: Pros and Cons

| Pros | Cons |
|------|------|
| Hardware isolation (no noisy neighbor) | **Partitions locked even when idle** |
| Guaranteed memory/compute | Fixed partition sizes (can't resize dynamically) |
| QoS guarantees per workload | Only A100/H100 GPUs support MIG |
| Fault isolation between instances | Reconfiguration requires GPU reset |
| | Fewer SM cores per partition |

**Key Limitation**: If Instance 3 and 5 above are idle, that 20GB of GPU memory **cannot be reclaimed** by other workloads. MIG is excellent for multi-tenant isolation but poor for bursty ML inference workloads.

---

## Approach 2: Time-Sharing Operators

**How it works**: Kubernetes operators manage GPU access scheduling

Examples: NVIDIA GPU Operator time-slicing, RunAI, Volcano

```
┌─────────────────────────────────────────────────────────────┐
│                Time-Sliced GPU Access                        │
│                                                              │
│  Time ──────────────────────────────────────────►           │
│                                                              │
│  │ Pod A │ Pod B │ Pod A │ Pod C │ Pod B │ Pod A │          │
│  ├───────┼───────┼───────┼───────┼───────┼───────┤          │
│  │ YOLO  │ BERT  │ YOLO  │Whisper│ BERT  │ YOLO  │          │
│  └───────┴───────┴───────┴───────┴───────┴───────┘          │
│                                                              │
│  Scheduler: Round-robin or priority-based                    │
│  Context switch overhead: 10-50ms per switch                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Time-Sharing: Pros and Cons

| Pros | Cons |
|------|------|
| Works with any GPU | **Context switch overhead** (10-50ms) |
| Kubernetes-native scheduling | Memory not truly shared (each pod loads model) |
| Fair queuing across pods | Latency spikes during context switches |
| Familiar pod/resource model | GPU memory fragmentation |
| | Models must be small enough to fit individually |

**Key Limitation**: Each pod must load its own model into GPU memory. With 8 pods sharing a 40GB GPU, each pod effectively gets ~5GB — **not enough for larger models**. Also, context switching adds unpredictable latency.

---

## Approach 3: Multi-Model Serving (Triton)

**How it works**: Single inference server manages multiple models on one GPU

```
┌────────────────────────────────────────────────────────────┐
│                 Triton Inference Server                     │
│                      (Single Process)                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Shared GPU Memory Pool                  │   │
│  │                                                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐  │   │
│  │  │ YOLOv8n │ │ BERT    │ │ Whisper │ │ SmolLM    │  │   │
│  │  │ 2GB     │ │ 1GB     │ │ 3GB     │ │ 4GB       │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └───────────┘  │   │
│  │                                                       │   │
│  │  [Dynamic batching] [Request scheduling] [LRU evict] │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ✓ No context switches   ✓ Shared memory   ✓ Dynamic load  │
└────────────────────────────────────────────────────────────┘
```

---

## Multi-Model Serving: Pros and Cons

| Pros | Cons |
|------|------|
| **Zero context switch overhead** | Single point of failure (mitigated by replicas) |
| Dynamic model loading/unloading | Requires compatible model formats |
| Efficient memory pooling | More complex deployment model |
| Request-level scheduling | Less isolation between models |
| Native batching across requests | |
| LRU eviction for idle models | |

**Key Advantage**: Triton can load/unload models dynamically based on demand. Idle models don't waste memory — they get evicted and their memory becomes available for active models.

---

## GPU Sharing Comparison Summary

| Factor | MIG | Time-Sharing | Triton Multi-Model |
|--------|-----|--------------|-------------------|
| **Idle resource waste** | High (partitions locked) | Medium (per-pod memory) | **Low (LRU eviction)** |
| **Context switch overhead** | None | 10-50ms | **None** |
| **Isolation** | Hardware | Process | Request |
| **Dynamic scaling** | Requires reset | Pod scheduling | **Instant load/unload** |
| **Memory efficiency** | Fixed partitions | Per-pod allocation | **Shared pool** |
| **Best for** | Multi-tenant SLA | Kubernetes-native | **ML inference at scale** |

---

## Our Approach: Triton Multi-Model + LRU + Pinning

We implement the multi-model approach with these enhancements:

1. **Dynamic Loading**: Models load on first request (no pre-warming needed)
2. **LRU Eviction**: Idle models automatically unload after configurable timeout
3. **Model Pinning**: Critical production models protected from eviction
4. **Instance Groups**: Multiple model instances for high-throughput models
5. **Future: MPS**: NVIDIA Multi-Process Service for true concurrent execution

The following slides detail each strategy.

---

## Strategy 1: Multi-Model Serving (Detail)

Triton manages multiple models in a shared GPU memory pool:

```
┌────────────────────────────────────────────────────────────┐
│                      GPU (A100 80GB)                        │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ YOLOv8n     │ │ BERT        │ │ Whisper     │          │
│  │ 2GB         │ │ 1GB         │ │ 3GB         │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐                           │
│  │ SmolLM      │ │ Custom      │   Used: 15GB              │
│  │ 4GB         │ │ 5GB         │   Available: 65GB         │
│  └─────────────┘ └─────────────┘                           │
└────────────────────────────────────────────────────────────┘
```

**Key difference from MIG/time-sharing**: All 65GB of unused memory remains available for additional models or batched requests. No locked partitions, no per-pod memory isolation overhead.

---

## Strategy 2: Dynamic Loading (LRU)

Our implementation uses Least Recently Used eviction:

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Lifecycle                           │
│                                                              │
│  Request for Model A                                         │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐     ┌──────────────┐                      │
│  │ Model loaded?│──No─►│ Load Model A │                      │
│  └──────┬───────┘     └──────┬───────┘                      │
│         │Yes                 │                               │
│         ▼                    │                               │
│  ┌──────────────┐            │                               │
│  │ Update LRU   │◄───────────┘                               │
│  │ timestamp    │                                            │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Run inference│                                            │
│  └──────────────┘                                            │
│                                                              │
│  Background: Evict models idle > 5 minutes (configurable)    │
└─────────────────────────────────────────────────────────────┘
```

---

## LRU Implementation Details

| Feature | Description |
|---------|-------------|
| **Auto-load** | Models load on first inference request |
| **Idle timeout** | Configurable (default 5 min) |
| **Pinning** | Critical models protected from eviction |
| **Shared state** | gRPC + REST proxies share LRU state |
| **Source of truth** | Triton server (not cache) |

```yaml
# Configuration
MODEL_IDLE_TIMEOUT_SECS: 300    # 5 minutes
MODEL_LOAD_TIMEOUT_SECS: 120    # 2 minutes max load time
MODEL_UNLOAD_TIMEOUT_SECS: 30   # 30 seconds max unload
```

---

## Strategy 3: Model Pinning

Protect production-critical models:

```
┌─────────────────────────────────────────────────────────────┐
│                      GPU Memory                              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              PINNED (Protected)                      │    │
│  │  ┌─────────────┐ ┌─────────────┐                    │    │
│  │  │ YOLOv8n     │ │ Production  │  Never evicted     │    │
│  │  │ (critical)  │ │ LLM         │                    │    │
│  │  └─────────────┘ └─────────────┘                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              UNPINNED (LRU Eligible)                 │    │
│  │  ┌─────────────┐ ┌─────────────┐                    │    │
│  │  │ Dev Model A │ │ Test Model  │  Evicted when idle │    │
│  │  │ (idle 10m)  │ │ (idle 2m)   │                    │    │
│  │  └─────────────┘ └─────────────┘                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Strategy 4: NVIDIA MPS (Future)

Multi-Process Service for true GPU sharing:

```
┌─────────────────────────────────────────────────────────────┐
│                   Without MPS                                │
│                                                              │
│  Process A ████████░░░░░░░░░░░░░░░░░░░░░░░░ (waiting)       │
│  Process B ░░░░░░░░████████░░░░░░░░░░░░░░░░ (waiting)       │
│  Process C ░░░░░░░░░░░░░░░░████████░░░░░░░░ (waiting)       │
│                    Time-sliced (serial)                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    With MPS                                  │
│                                                              │
│  Process A ████████████████████████████████                 │
│  Process B ████████████████████████████████  Concurrent     │
│  Process C ████████████████████████████████                 │
│                    Spatial sharing                           │
└─────────────────────────────────────────────────────────────┘
```

**Status**: Planned for future release

---

## Strategy 5: Instance Groups

Triton-native parallelism:

```protobuf
# config.pbtxt
instance_group [
  { kind: KIND_GPU, count: 2, gpus: [0] },    # 2 instances on GPU 0
  { kind: KIND_GPU, count: 2, gpus: [1] }     # 2 instances on GPU 1
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100
}
```

**Benefits**:
- Overlap compute and memory transfers
- Better GPU utilization under load
- Automatic request batching

---

## GPU Sharing: Decision Matrix

| Strategy | Latency Impact | Utilization Gain | Complexity |
|----------|---------------|------------------|------------|
| Multi-model | None (warm) | High | Low |
| LRU eviction | First request | Medium | Medium |
| Model pinning | None | Medium | Low |
| MPS | Low | High | High |
| Instance groups | None | High | Low |

**Recommendation**: Start with multi-model + LRU + pinning

---

# Part 6: Recommendations

---

## Architecture Recommendations

### Protocol Selection

| Workload | Recommendation | Rationale |
|----------|---------------|-----------|
| Image/Video | **gRPC only** | 20-30x throughput |
| Text (NLP) | REST or gRPC | Inference-bound |
| Audio | REST or gRPC | Inference-bound |
| LLM | REST or gRPC | Token generation bound |

### GPU Strategy

| Scenario | Recommendation |
|----------|---------------|
| Production models | Pin critical models |
| Development | LRU with 5-min timeout |
| Multi-tenant | Namespace isolation + quotas |
| Cost optimization | Multi-model serving |

---

## Implementation Checklist

**Phase 1: Foundation**
- [ ] Deploy Triton with explicit model control
- [ ] Implement gRPC proxy with auth
- [ ] Set up S3 model repository
- [ ] Configure EBS cache for HuggingFace

**Phase 2: Optimization**
- [ ] Enable LRU eviction
- [ ] Pin production models
- [ ] Configure dynamic batching
- [ ] Set up monitoring/metrics

**Phase 3: Scale**
- [ ] Multi-GPU deployment
- [ ] Instance group tuning
- [ ] Evaluate MPS for dense workloads

---

## Key Takeaways

1. **gRPC is not optional for image/video** — 20-30x performance difference is architectural, not tunable

2. **JSON overhead is CPU, not network** — Parsing 1.2M floats from text is inherently expensive

3. **GPU sharing is table stakes** — Multi-model + LRU + pinning maximizes ROI

4. **Triton provides the primitives** — We build the enterprise layer (auth, LRU, admin)

---

## Questions?

### Resources

- **Design Document**: `docs/design.md`
- **Local Testing**: `docs/runbook_local_testing.md`
- **Protocol Analysis**: `docs/REST_vs_gRPC_tradeoffs.md`
- **API Documentation**: `http://<proxy>:8080/docs`

### Contact

- Architecture questions: [Chief Architect]
- Implementation details: [Engineering Lead]
- Product roadmap: [Product Manager]

---

## Appendix A: Benchmark Methodology

```bash
# YOLOv8 benchmark (includes both gRPC and REST)
python scripts/benchmarks/benchmark_yolov8_clients.py \
  --video samples/video.avi \
  --max-frames 100

# BERT benchmark
python scripts/benchmarks/benchmark_bert_clients.py \
  --num-texts 100

# LLM benchmark
python scripts/benchmarks/benchmark_llm_clients.py \
  --num-prompts 50
```

**Environment**:
- GPU: NVIDIA A100 40GB
- CPU: AMD EPYC 7R32 (8 cores)
- Network: Same-region (minimal latency)
- Image size: 640×640 RGB

---

## Appendix B: Storage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Triton Pod                                 │
│                                                              │
│  /triton-repo (S3)          /hf-cache (EBS)                 │
│  ├── models/                ├── hub/                        │
│  │   ├── yolov8n/           │   └── models--*/              │
│  │   ├── bert/              └── (HF downloads)              │
│  │   └── whisper/                                           │
│  └── weights/               /model-state (RAM)              │
│      └── smollm/            └── *.json (LRU state)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Volume Characteristics:
┌──────────────┬─────────┬───────────────┬──────────────────┐
│ Volume       │ Type    │ Access Mode   │ Purpose          │
├──────────────┼─────────┼───────────────┼──────────────────┤
│ triton-repo  │ S3      │ ReadWriteMany │ Model configs    │
│ hf-cache     │ EBS     │ ReadWriteOnce │ HF downloads     │
│ model-state  │ emptyDir│ Ephemeral     │ LRU tracking     │
└──────────────┴─────────┴───────────────┴──────────────────┘
```

---

## Appendix C: Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                   Request Flow                               │
│                                                              │
│  Client                                                      │
│    │                                                         │
│    │ x-domino-api-key: <key>                                │
│    │ OR                                                      │
│    │ Authorization: Bearer <jwt>                            │
│    ▼                                                         │
│  ┌─────────────┐                                            │
│  │ Proxy       │──── GET /v4/auth/principal ────► Domino    │
│  │             │◄─── { isAdmin: true/false } ────           │
│  └──────┬──────┘                                            │
│         │                                                    │
│         │ Authenticated + Authorized                         │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Triton      │  (Never exposed externally)                │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘

Role-Based Access:
┌──────────────┬─────────────────────────────────────────────┐
│ Role         │ Allowed Operations                          │
├──────────────┼─────────────────────────────────────────────┤
│ User         │ Inference, List models, View status         │
│ Admin        │ + Load, Unload, Pin, Verify models          │
└──────────────┴─────────────────────────────────────────────┘
```
