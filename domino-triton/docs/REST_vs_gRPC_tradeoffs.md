# REST vs gRPC Trade-offs

## Summary

| Factor | REST | gRPC |
|--------|------|------|
| **Payload Size** | ~15-20 MB/frame | ~5 MB/frame |
| **Serialization** | JSON (text) | Protocol Buffers (binary) |
| **Throughput** | ~1-5 FPS | ~15-30 FPS |
| **Ease of Use** | Simple (curl, browser) | Requires protobuf |
| **Streaming** | Polling/WebSockets | Native bidirectional |

**Recommendation**:
- **Image/Video**: Use gRPC (up to 20x faster)
- **Text/Audio/LLM**: Either protocol (inference-bound)
- **Debugging**: Use REST (human-readable)

---

## Why gRPC is Faster for Images

For a YOLOv8 frame (1×3×640×640 = 1,228,800 floats):

| Operation | gRPC | REST JSON |
|-----------|------|-----------|
| Encode | memcpy (~0ms) | 1.2M float→string conversions |
| Transfer | 5 MB | 17.5 MB |
| Decode | pointer cast | 1.2M string→float parsing |

**Combined**: 3.5× network × 10× parsing = **~35× slower**

JSON parsing is expensive because each float like `0.12345678` requires string allocation, digit-by-digit reading, decimal tracking, and float conversion. Binary protobuf reads 4 bytes directly.

---

## Benchmark Results (YOLOv8n, 640×640)

| Protocol | Payload | Throughput |
|----------|---------|------------|
| **gRPC** | 5 MB | ~20 FPS |
| REST | 17.5 MB | ~1-3 FPS |

---

## Protocol Selection by Workload

| Model | Payload | Inference Time | Recommendation |
|-------|---------|----------------|----------------|
| YOLOv8n (video) | 5 MB/frame | 50ms | **gRPC** (transport-bound) |
| BERT (text) | 50 KB | 250ms | Either (inference-bound) |
| Whisper (audio) | 1.2 MB | 10-15s | Either (inference-bound) |
| LLM | 1 KB | 1-5s | Either (inference-bound) |

**Rule**: If `transport_time >= inference_time`, use gRPC. Otherwise, either works.

---

## REST Optimizations

If REST is required for large tensors:

1. **gzip compression**: ~60-70% reduction (adds CPU overhead)
2. **Batching**: Reduce per-request overhead

Even with optimizations, gRPC remains significantly faster for image/video workloads.
