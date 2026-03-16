# Operational Notes

Lessons learned and troubleshooting tips for running Triton Inference Server.

---

## GPU Memory Management

### Problem: GPU Memory Not Freed After Model Unload

When unloading Python backend models (e.g., `tinyllama-python`, `smollm-135m-python`, `whisper-tiny-python`), GPU memory may not be fully released.

**Symptoms:**
- `unload_model --all` reports "Unloaded 0 models" or succeeds
- Model status shows all models as UNLOADED
- But next model load fails with `CUDA out of memory`
- `nvidia-smi` or error logs show processes still consuming GPU memory

**Example error:**
```
[tinyllama-python] Failed to load model: CUDA out of memory.
GPU 0 has a total capacity of 15.77 GiB of which 1.50 MiB is free.
Process 85917 has 13.96 GiB memory in use.
```

### Why This Happens

1. **PyTorch GPU Memory Caching**: PyTorch caches GPU memory allocations for performance. Even after tensors are deleted, the memory may remain reserved by PyTorch rather than being returned to the system.

2. **Crashed/Stuck Model Loads**: If a model load fails mid-way (e.g., OOM during loading), the Python subprocess may not clean up properly, leaving GPU memory allocated.

3. **Python Backend Process Lifecycle**: Triton's Python backend runs models in separate Python processes. If these processes don't terminate cleanly, their GPU memory remains allocated.

4. **CUDA Context Leaks**: CUDA contexts from crashed processes can leave "leaked" memory that persists until the parent process (Triton server) is restarted.

### Solution: Restart Triton Server

The only reliable way to fully free GPU memory is to restart the Triton pod:

```bash
# Restart the deployment
kubectl rollout restart deployment/triton-inference-server -n <namespace>

# Wait for it to be ready
kubectl rollout status deployment/triton-inference-server -n <namespace>
```

Or scale down and up:

```bash
kubectl scale deployment/triton-inference-server -n <namespace> --replicas=0
sleep 10
kubectl scale deployment/triton-inference-server -n <namespace> --replicas=1
```

### Best Practices

1. **Test models one at a time**: Use `scripts/testing/test_models_sequential.py` which unloads each model after testing.

2. **Don't load competing large models**: `tinyllama-python` (~2.2GB) and `tinyllama-trtllm` (~14GB) cannot coexist on a 16GB GPU.

3. **Monitor GPU memory**: Check memory before loading large models:
   ```bash
   kubectl exec -n <namespace> deployment/triton-inference-server -- nvidia-smi
   ```

4. **Restart between test sessions**: If you've been loading/unloading many models, restart Triton before production use.

---

## Model Loading Issues

### Problem: Model Not Finding Local Weights

Python backend models look for weights at `/triton-repo/weights/<model-name>/1/`.

**Symptoms:**
- Model tries to download from HuggingFace instead of using local weights
- Log shows: `Weights folder not found, using: <HuggingFace model ID>`
- Load times out or fails due to network issues

**Causes:**
1. Weights weren't copied to the EDV
2. `MODEL_REPO` environment variable not set correctly
3. EDV mounted at unexpected path in Triton pod

**Diagnosis:**
```bash
# Check weights exist in Triton pod
kubectl exec -n <namespace> deployment/triton-inference-server -- \
  ls -la /triton-repo/weights/<model-name>/1/

# Check MODEL_REPO env var
kubectl exec -n <namespace> deployment/triton-inference-server -- \
  env | grep MODEL_REPO
```

**Solution:**
1. Ensure weights are copied with `--include-weights` or `--all`:
   ```bash
   python scripts/model_management/copy_model.py --target $TARGET --all
   ```

2. Set `MODEL_REPO` environment variable if EDV isn't mounted at `/triton-repo`:
   ```yaml
   env:
     - name: MODEL_REPO
       value: /path/to/edv/models
   ```

---

## S3 Fuse (EDV) Quirks

### Problem: Copy Fails with "Operation not permitted"

When copying models to an S3-backed EDV, standard file operations may fail.

**Symptoms:**
```
shutil.Error: [Errno 1] Operation not permitted
PermissionError: [Errno 1] Operation not permitted
```

**Causes:**
- S3 fuse doesn't support overwriting existing files
- S3 fuse doesn't support setting file metadata (timestamps, permissions)
- `rmtree` may not fully delete directories

**Solution:**
The `copy_model.py` script handles these issues by:
1. Using `ignore_errors=True` for `rmtree`
2. Using `copyfile` instead of `copy2` (no metadata)
3. Removing destination files before copying

If issues persist, manually delete the target directory:
```bash
rm -rf /domino/edv/.../models/<model-name>
rm -rf /domino/edv/.../weights/<model-name>
```

---

## TensorRT-LLM vs Python Backend

### Memory Usage Comparison

| Model | Backend | GPU Memory | Notes |
|-------|---------|------------|-------|
| tinyllama-trtllm | TensorRT-LLM | ~14 GB | Optimized, faster inference |
| tinyllama-python | Python/HF | ~2.2 GB | Baseline, more flexible |
| smollm-135m-python | Python/HF | ~300 MB | Small model, good for testing |

### When to Use Each

**TensorRT-LLM (`tinyllama-trtllm`)**:
- Production deployments requiring maximum throughput
- When GPU memory is available (~14GB)
- Architecture-specific (must rebuild engine for different GPU)

**Python Backend (`tinyllama-python`)**:
- Development and testing
- When GPU memory is limited
- When you need flexibility (easy to modify model code)
- Portable across GPU architectures

### Cannot Run Both Simultaneously

On a 16GB GPU, you cannot load both `tinyllama-trtllm` (14GB) and `tinyllama-python` (2.2GB) at the same time. Choose one or the other.

---

## Useful Commands

### Check GPU Memory
```bash
kubectl exec -n <namespace> deployment/triton-inference-server -- nvidia-smi
```

### View Triton Logs
```bash
kubectl logs -n <namespace> deployment/triton-inference-server --tail=100
kubectl logs -n <namespace> deployment/triton-inference-server -f  # follow
```

### Check Model Status
```bash
python scripts/model_management/model_status.py
```

### Force Unload All Models
```bash
python scripts/model_management/unload_model.py --all
```

### Restart Triton (Clear GPU Memory)
```bash
kubectl rollout restart deployment/triton-inference-server -n <namespace>
```

### Test Models Sequentially (Avoids OOM)
```bash
python scripts/testing/test_models_sequential.py --models bert-base-uncased yolov8n
```

---

## REST vs gRPC Protocol Selection

### Understanding the Options

Triton supports three client protocols for inference:

| Protocol | Encoding | Payload Size | Client Required |
|----------|----------|--------------|-----------------|
| **REST (JSON)** | JSON arrays | ~12 bytes/float | Any HTTP client (curl, browser, requests) |
| **REST (Binary)** | Raw bytes | ~4 bytes/float | `tritonclient.http` only |
| **gRPC** | Binary protobuf | ~4 bytes/float | `tritonclient.grpc` only |

### REST Binary vs gRPC: Why Not Just Use REST Binary?

If REST with binary encoding has similar payload size to gRPC, why use gRPC at all?

**Key insight**: REST's main advantage is **universality** - it works with curl, browsers, and any HTTP client. Once you need the `tritonclient` library for binary encoding, you've lost that advantage.

| Factor | REST Binary | gRPC |
|--------|-------------|------|
| Client library required | `tritonclient.http` | `tritonclient.grpc` |
| Works with curl/browser | No | No |
| Protocol | HTTP/1.1 | HTTP/2 |
| Connection model | Request/response | Persistent, multiplexed |
| Streaming | Polling or WebSocket | Native bidirectional |
| Real-time video | New request per frame | Single persistent stream |

**gRPC advantages when using a specialized client:**
- **HTTP/2 multiplexing**: Multiple concurrent requests over a single connection
- **Bidirectional streaming**: Send frames continuously without reconnecting
- **Lower latency**: No connection setup overhead for each request
- **Better for sustained workloads**: Connection stays open

### When to Use Each Protocol

**Use REST (JSON)** when:
- Clients are browsers, curl, or standard HTTP tools
- Payload is small (text, audio metadata)
- Inference time dominates (LLMs, Whisper) - protocol overhead is negligible
- Debugging/inspection is important (HTTP traffic is easier to trace)

**Use REST (Binary)** when:
- gRPC is blocked by firewall or proxy
- Already have REST infrastructure
- Need HTTP-based load balancing

**Use gRPC** when:
- Using `tritonclient` library anyway
- Processing images or video (large tensor payloads)
- Need streaming (real-time video inference)
- High-throughput production workloads

### Performance Comparison (YOLOv8n, 640x640 images)

```
Tensor size: 1 x 3 x 640 x 640 = 1,228,800 floats

REST (JSON):
  - Payload: ~15 MB (each float as string "0.12345678")
  - CPU overhead: 1.2M string→float conversions
  - Result: ~1-3 FPS

REST (Binary):
  - Payload: ~5 MB (raw bytes in HTTP body)
  - CPU overhead: memcpy only
  - Result: ~15-20 FPS (similar to gRPC for single requests)

gRPC (Binary):
  - Payload: ~5 MB (raw IEEE 754 floats)
  - CPU overhead: memcpy only
  - Result: ~20 FPS
```

**REST JSON vs gRPC**: 10-30x slower due to JSON parsing overhead

**REST Binary vs gRPC**: Similar single-request performance, but gRPC wins for:
- Streaming workloads (persistent connection, no reconnect per frame)
- High concurrency (HTTP/2 multiplexing)
- Sustained throughput (connection reuse)

### Quantifying gRPC Advantages Over REST Binary

For single requests, REST Binary and gRPC perform similarly. The differences show up in sustained workloads:

**1. Streaming (Video Inference)**

```
30 FPS video for 10 seconds = 300 frames

REST Binary (HTTP/1.1):
  - Each frame = new HTTP request
  - TCP overhead: ~1-3ms per request (more with TLS)
  - Total overhead: 300 × 2ms = 600ms wasted on connections
  - Effective: ~27 FPS max (connection-limited)

gRPC (HTTP/2 streaming):
  - Single connection for all frames
  - Frames flow back-to-back
  - Total overhead: ~2ms (one connection)
  - Effective: ~30 FPS (inference-limited)
```

**2. Connection Reuse (Sustained Load)**

```
1000 inference requests over 1 minute:

REST Binary (HTTP/1.1 without keep-alive):
  - 1000 TCP handshakes × 2ms = 2 seconds overhead
  - With TLS: 1000 × 10ms = 10 seconds overhead

gRPC:
  - 1 connection, reused for all 1000 requests
  - Overhead: ~10ms total
```

**3. Concurrency (Multiple Simultaneous Requests)**

```
10 concurrent inference requests:

REST (HTTP/1.1):
  - Need 10 separate TCP connections
  - Or requests queue behind each other

gRPC (HTTP/2):
  - Single connection, all 10 multiplexed
  - No head-of-line blocking
```

**Summary:**

| Scenario | REST Binary | gRPC | Difference |
|----------|-------------|------|------------|
| Single request | 50ms | 48ms | ~4% (negligible) |
| 30 FPS video (10s) | ~27 FPS | ~30 FPS | ~10% |
| 1000 requests | +2s overhead | +10ms | Significant |
| With TLS | +10s overhead | +10ms | Major |

**Bottom line**: For occasional single requests, REST Binary is fine. For video/streaming or high-throughput production, gRPC's persistent connection makes a real difference. If you need a specialized client library anyway (for binary encoding), choose gRPC for the streaming and HTTP/2 benefits.

### Benchmark Command

```bash
# Runs REST (JSON) vs gRPC comparison
python scripts/benchmarks/benchmark_yolov8_clients.py --video samples/video.avi --max-frames 20
```

The benchmark now correctly compares REST with JSON encoding against gRPC to show the real-world performance difference.
