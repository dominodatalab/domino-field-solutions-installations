# Batching in Triton Python Backend Models

This document explains how to implement batching support in Triton Python backend models, including the key patterns for handling both batched and non-batched inputs.

## Overview

Triton Inference Server supports dynamic batching, which accumulates multiple client requests and processes them together for improved throughput. However, enabling batching changes the shape of input tensors, which requires careful handling in Python backend models.

## Configuration

### Enable Batching in config.pbtxt

```protobuf
name: "my-model"
backend: "python"
max_batch_size: 8  # Maximum requests to batch together

# Dynamic batching settings
dynamic_batching {
  max_queue_delay_microseconds: 100000  # 100ms - wait for batch to fill
}

# Input dimensions are PER-ELEMENT (Triton manages batch dimension)
input [
  {
    name: "prompt"
    data_type: TYPE_STRING
    dims: [ 1 ]  # Single element per batch item
  },
  {
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "result"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]
```

### Key Configuration Differences

| Setting | Non-Batched | Batched |
|---------|-------------|---------|
| `max_batch_size` | `0` | `> 0` (e.g., `8`) |
| `dims` | `[ -1 ]` (variable) | `[ 1 ]` (per-element) |
| `dynamic_batching` | Not needed | Recommended |

## The Shape Problem

When batching is enabled, Triton adds a batch dimension to all inputs:

```
Non-batched (max_batch_size: 0):
  Client sends:     np.array(["prompt"], dtype=object)
  Model receives:   shape [1]

Batched (max_batch_size: 8):
  Client sends:     np.array([["prompt"]], dtype=object)
  Model receives:   shape [1, 1]  (batch=1, element=1)
```

This breaks code that assumes a specific shape:

```python
# This works for non-batched but FAILS for batched:
value = tensor.as_numpy()[0]

# Non-batched: [0] → scalar ✓
# Batched:     [0] → array  ✗
```

## The Solution: Use `.flatten()`

The `.flatten()` method normalizes any array shape to 1D:

```python
# Works for ALL shapes:
value = tensor.as_numpy().flatten()[0]

# [value]            → flatten → [value]       → [0] → value
# [[value]]          → flatten → [value]       → [0] → value
# [[v1], [v2], [v3]] → flatten → [v1, v2, v3]  → [i] → vi
```

## Implementation Patterns

### Pattern 1: Extracting Optional Parameters

Use `.flatten()[0]` to safely extract scalar values:

```python
def _get_optional_param(self, request, name: str, default, dtype=None):
    """
    Extract an optional parameter from a Triton request.
    Works with both batched and non-batched inputs.
    """
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return default

    # .flatten()[0] handles both [N] and [batch, 1] shapes
    value = tensor.as_numpy().flatten()[0]

    # Handle string conversion (bytes from Triton → str)
    if dtype == str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    return value
```

**Usage:**
```python
temperature = self._get_optional_param(request, "temperature", 0.7)
max_tokens = self._get_optional_param(request, "max_tokens", 256)
system_prompt = self._get_optional_param(request, "system_prompt", None, dtype=str)
```

### Pattern 2: Processing Batched Inputs

Flatten the input array, then iterate:

```python
def execute(self, requests: list) -> list:
    responses = []

    for request in requests:
        # Get input tensor
        prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
        prompt_np = prompt_tensor.as_numpy()

        # Flatten to handle both [N] and [batch, 1] shapes
        prompts_flat = prompt_np.flatten()

        results = []
        for i in range(len(prompts_flat)):
            prompt = prompts_flat[i]

            # Handle bytes → string conversion
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

            # Process the prompt
            result = self.process(prompt)
            results.append(result.encode("utf-8"))

        # Build response with batched output
        response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("result", np.array(results, dtype=object)),
            ]
        )
        responses.append(response)

    return responses
```

### Pattern 3: Client Code for Batched Requests

Clients must provide the batch dimension when `max_batch_size > 0`:

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8080")

# Shape must be [batch, element] = [1, 1] for single request
prompt_data = np.array([["What is 2+2?"]], dtype=object)
temperature_data = np.array([[0.7]], dtype=np.float32)

inputs = [
    httpclient.InferInput("prompt", prompt_data.shape, "BYTES"),
    httpclient.InferInput("temperature", temperature_data.shape, "FP32"),
]
inputs[0].set_data_from_numpy(prompt_data)
inputs[1].set_data_from_numpy(temperature_data)

result = client.infer("my-model", inputs)

# Output also has batch dimension - use flatten to extract
output = result.as_numpy("result").flatten()[0]
if isinstance(output, bytes):
    output = output.decode("utf-8")
```

## Visual Guide

```
NON-BATCHED (max_batch_size: 0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Client sends:       ["prompt"]              Shape: [1]
                         │
tensor.as_numpy():    ["prompt"]            Shape: [1]
                         │
              [0] ───────┘
                         │
Result:              "prompt"               ✓ Scalar


BATCHED (max_batch_size: 8) - WITHOUT FIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Client sends:      [["prompt1"],            Shape: [3, 1]
                    ["prompt2"],
                    ["prompt3"]]
                         │
tensor.as_numpy():  [["prompt1"],           Shape: [3, 1]
                     ["prompt2"],
                     ["prompt3"]]
                         │
              [0] ───────┘
                         │
Result:              ["prompt1"]            ✗ Array (first row)!


BATCHED (max_batch_size: 8) - WITH FIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Client sends:      [["prompt1"],            Shape: [3, 1]
                    ["prompt2"],
                    ["prompt3"]]
                         │
tensor.as_numpy():  [["prompt1"],           Shape: [3, 1]
                     ["prompt2"],
                     ["prompt3"]]
                         │
         .flatten() ─────┘
                         │
                    ["prompt1",             Shape: [3]
                     "prompt2",
                     "prompt3"]
                         │
              [i] ───────┘
                         │
Result:              "prompt_i"             ✓ Scalar
```

## Common Errors and Solutions

### Error: Shape Mismatch

```
unexpected shape for input 'prompt' for model 'my-model'.
Expected [-1,1], got [1]
```

**Cause:** Client sending non-batched shape when model expects batched.

**Solution:** Change client from `np.array([value])` to `np.array([[value]])`.

### Error: String/Bytes Concatenation

```
can only concatenate str (not "bytes") to str
```

or

```
can only concatenate str (not "numpy.ndarray") to str
```

**Cause:** Model code using `tensor[0]` which returns an array with batching.

**Solution:** Use `.flatten()[0]` and handle bytes decoding:

```python
value = tensor.as_numpy().flatten()[0]
if isinstance(value, bytes):
    value = value.decode("utf-8")
```

### Error: IndexError with `[0][0]`

```
IndexError: invalid index to scalar variable
```

**Cause:** Using `tensor[0][0]` which fails for non-batched inputs.

**Solution:** Use `.flatten()[0]` which works for any shape.

## Performance Benefits

Dynamic batching improves throughput by processing multiple requests together:

```
Without Batching (sequential):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Request 1 ──► Process ──► Response 1
Request 2 ──► Process ──► Response 2
Request 3 ──► Process ──► Response 3
Request 4 ──► Process ──► Response 4

Total time: 4 × single_request_time


With Batching (concurrent):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Request 1 ─┐              ┌─► Response 1
Request 2 ─┼─► Process ───┼─► Response 2
Request 3 ─┤   (batch)    ├─► Response 3
Request 4 ─┘              └─► Response 4

Total time: ~1 × single_request_time (+ small overhead)
```

### Example Results

Testing with TinyLlama model (4 concurrent requests on CPU):

| Metric | Value |
|--------|-------|
| Total requests | 4 |
| Overall time | 20,238ms |
| Sequential estimate | 80,944ms |
| **Batch speedup** | **4.0x** |

## Best Practices

1. **Always use `.flatten()[0]`** for extracting scalar values from tensors
2. **Handle bytes conversion** - Triton often returns strings as bytes
3. **Test both modes** - Verify your model works with `max_batch_size: 0` and `> 0`
4. **Set appropriate queue delay** - 100ms is a good starting point for balancing latency and throughput
5. **Document input shapes** - Make it clear whether clients should send `[1]` or `[1, 1]`

## Quick Reference

```python
# Extracting a scalar (works for any shape)
value = tensor.as_numpy().flatten()[0]

# Iterating over batch (works for any shape)
values = tensor.as_numpy().flatten()
for v in values:
    process(v)

# String handling
if isinstance(value, bytes):
    value = value.decode("utf-8")

# Client input shape (batched)
data = np.array([[value]], dtype=object)  # Shape: [1, 1]

# Client output extraction (batched)
result = response.as_numpy("output").flatten()[0]
```

## See Also

- [Triton Dynamic Batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
- [constrained_decoding.md](./constrained_decoding.md) - Example model with batching support
