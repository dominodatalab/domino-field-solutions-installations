# Model Configuration Guide

This guide explains how to configure Triton model device placement, instance counts, and GPU allocation.

## Overview

Each model in Triton has a `config.pbtxt` file that controls how it runs. The key configuration is the `instance_group` section:

```protobuf
instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]
  }
]
```

## Configuration Options

### Device Kind

Specifies whether the model runs on CPU or GPU.

| Kind | Description | Use Case |
|------|-------------|----------|
| `KIND_CPU` | Run on CPU | Models without GPU support, or when GPU is unavailable |
| `KIND_GPU` | Run on GPU | GPU-accelerated inference for supported models |
| `KIND_AUTO` | Triton decides | Let Triton choose based on available hardware |

**Recommendations:**
- **ONNX models** (YOLOv8, BERT): Use `KIND_GPU` for best performance, `KIND_CPU` as fallback
- **Python backend models** (Whisper, LLMs): Check if the model code supports GPU; many use CPU by default
- **TensorRT models**: Must use `KIND_GPU`

### Instance Count

Controls how many parallel instances of the model are loaded.

```protobuf
instance_group [
  {
    kind: KIND_GPU
    count: 2      # Two parallel instances
    gpus: [ 0 ]
  }
]
```

**What it means:**
- `count: 1` - One model instance (default)
- `count: 2` - Two instances processing requests concurrently
- `count: N` - N parallel instances

**Trade-offs:**

| Instances | Throughput | Memory Usage | Latency |
|-----------|------------|--------------|---------|
| 1 | Baseline | Baseline | Baseline |
| 2+ | Higher (concurrent requests) | 2x+ (each instance loads weights) | Same per-request |

**Recommendations by model type:**

| Model Type | Recommended Count | Reason |
|------------|-------------------|--------|
| Large LLMs (>1B params) | 1 | Memory constraints |
| Small LLMs (<500M params) | 1-2 | Balance memory/throughput |
| BERT, text models | 2-4 | Small footprint, high throughput |
| YOLOv8, vision models | 2-4 | Benefits from parallelism |
| Whisper, audio models | 1-2 | Moderate memory usage |

### GPU Index

Specifies which GPU(s) to use when `kind: KIND_GPU`.

```protobuf
instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]    # Use GPU 0
  }
]
```

**Multi-GPU examples:**

```protobuf
# Single GPU (GPU 0)
gpus: [ 0 ]

# Single GPU (GPU 1)
gpus: [ 1 ]

# Multiple GPUs (instances distributed across GPUs)
gpus: [ 0, 1 ]
```

**Notes:**
- GPU indices are 0-based
- If `gpus` is omitted, Triton uses all available GPUs
- For `KIND_CPU`, the `gpus` field is ignored

## Using the Dashboard UI

The Triton Admin Dashboard provides a web interface to change model configuration.

### Accessing Device Configuration

1. Navigate to the Dashboard
2. Click on a model to open its detail page
3. Find the **Device Configuration** section

### Changing Settings

1. **Device Kind**: Select CPU or GPU radio button
2. **Instance Count**: Enter number of instances (1-8)
3. **GPU Device Index**: (GPU only) Select which GPU to use
4. **Reload model**: Keep checked to apply changes immediately

### Applying Changes

When you click **Update Configuration**:

1. The `config.pbtxt` file is updated on the shared storage (EDV)
2. The model is unloaded from Triton
3. On next inference request (or manual load), Triton loads the model with new settings

## Manual Configuration

You can also edit `config.pbtxt` files directly.

### File Location

Models are stored in the Triton model repository:

```
/mnt/domino-inference-{env}-triton-repo-pvc/models/{model_name}/config.pbtxt
```

Where `{env}` is `dev`, `staging`, or `prod`.

### Example Configurations

**CPU-only model:**
```protobuf
name: "bert-base-uncased"
platform: "onnxruntime_onnx"
max_batch_size: 8

instance_group [
  {
    kind: KIND_CPU
    count: 2
  }
]
```

**GPU model with specific GPU:**
```protobuf
name: "yolov8n"
platform: "onnxruntime_onnx"
max_batch_size: 8

instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]
  }
]
```

**High-throughput GPU model:**
```protobuf
name: "bert-base-uncased"
platform: "onnxruntime_onnx"
max_batch_size: 8

instance_group [
  {
    kind: KIND_GPU
    count: 4
    gpus: [ 0 ]
  }
]
```

### Applying Manual Changes

After editing `config.pbtxt`:

1. **Unload the model** (via Dashboard or API):
   ```bash
   curl -X POST "$TRITON_REST_URL/v2/repository/models/{model_name}/unload"
   ```

2. **Load the model** to apply new config:
   ```bash
   curl -X POST "$TRITON_REST_URL/v2/repository/models/{model_name}/load"
   ```

Or use the Dashboard UI to unload/load the model.

## Troubleshooting

### Model fails to load after config change

**Symptoms:** Model stuck in "UNAVAILABLE" state after changing device kind.

**Solutions:**
1. Check Triton logs: `kubectl logs -l app=triton-inference-server`
2. Verify GPU is available: `nvidia-smi`
3. Try `KIND_CPU` to confirm model works
4. Check for typos in `config.pbtxt`

### Out of memory with multiple instances

**Symptoms:** Model loads with `count: 1` but fails with `count: 2+`.

**Solutions:**
1. Reduce instance count
2. Use a smaller batch size (`max_batch_size`)
3. Use a different GPU with more memory
4. For LLMs, stick with `count: 1`

### Model not using GPU

**Symptoms:** Model configured for GPU but running on CPU.

**Possible causes:**
1. Python backend models may need explicit GPU code
2. ONNX model may not have GPU execution provider
3. GPU drivers not available in container

**Check:** Look at Triton startup logs for instance group loading messages.

## Idle Timeout and Model Eviction

Models that are not pinned can be automatically unloaded after a period of inactivity. This helps conserve GPU memory.

### Default Timeout

The default idle timeout is **30 minutes** (1800 seconds). This is configured via the `MODEL_IDLE_TIMEOUT_SECS` environment variable on the HTTP proxy.

### Per-Model Timeout

You can override the default timeout for individual models by adding a parameter to `config.pbtxt`:

```protobuf
# Keep this model loaded for 1 hour before eviction
parameters {
  key: "idle_timeout_secs"
  value { string_value: "3600" }
}
```

### Dashboard Display

The Dashboard shows eviction countdown for loaded models:
- **Green**: Model has plenty of time before eviction
- **Yellow (warning)**: Less than 5 minutes until eviction
- **Red (critical)**: Model is eligible for eviction now

**Pinned models** are never automatically evicted.

### Preventing Eviction

To prevent a model from being evicted:
1. **Pin the model** via the Dashboard UI
2. **Set a very long timeout** in config.pbtxt
3. **Access the model regularly** to reset the idle timer

## Best Practices

1. **Start simple**: Begin with `count: 1` and `KIND_CPU`, then optimize
2. **Monitor memory**: Check GPU memory before increasing instance count
3. **Test changes**: Make config changes in dev/staging before production
4. **Document settings**: Keep notes on why specific settings were chosen
5. **Use Dashboard**: Prefer the UI over manual edits to avoid syntax errors
6. **Pin production models**: Pin models that should always stay loaded
7. **Use custom timeouts**: Set appropriate timeouts for models with different usage patterns
