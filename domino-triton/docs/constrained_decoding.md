# Constrained Decoding for JSON Output

This document explains the design and implementation of JSON-constrained decoding in the `tinyllama-python` model.

## Overview

Constrained decoding forces a language model to generate output that conforms to a specific format or grammar. In this implementation, we constrain the TinyLlama model to output valid JSON with a predefined schema:

```json
{"answer": "<text>", "confidence": <number>}
```

This is useful for:
- Structured data extraction
- API responses that require predictable formats
- Integration with downstream systems expecting JSON
- Reducing post-processing and validation logic

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TinyLlama Python Model                       │
│                                                                  │
│  ┌──────────────┐    ┌─────────────────────┐    ┌────────────┐  │
│  │   Prompt     │───▶│  Chat Template      │───▶│  Tokenize  │  │
│  │   + System   │    │  Formatting         │    │            │  │
│  └──────────────┘    └─────────────────────┘    └─────┬──────┘  │
│                                                       │         │
│                                                       ▼         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Model.generate()                       │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │           JsonPrefixLogitsProcessor                 │  │  │
│  │  │  - Tracks generated tokens (not prompt)             │  │  │
│  │  │  - Forces { as first token                          │  │  │
│  │  │  - Constrains to "answer", "confidence" keys        │  │  │
│  │  │  - Masks invalid token probabilities to -inf        │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │              JsonStopCriteria                       │  │  │
│  │  │  - Stops when JSON is complete (balanced braces)    │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              JSON Parsing & Validation                    │  │
│  │  - Extract JSON from generated text                       │  │
│  │  - Parse and re-serialize for clean output                │  │
│  │  - Fallback error response if parsing fails               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. JsonPrefixLogitsProcessor

The core component that constrains token generation. It implements the `LogitsProcessor` interface from Hugging Face Transformers.

```python
class JsonPrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length  # Track where generation starts
```

**Key Design Decisions:**

1. **Track prompt length**: The processor only examines *generated* tokens, not the original prompt. This is critical because the system prompt contains JSON examples with `{` characters.

2. **State machine approach**: The processor implements a simple state machine based on the current generated text:

   ```
   [empty] → { → "answer" → : → "<value>" → , → "confidence" → : → <number> → }
   ```

3. **Token masking**: Invalid tokens are masked by setting their log probabilities to `-inf`, making them impossible to sample.

**Implementation:**

```python
def __call__(self, input_ids, scores):
    text = self._decode_generated(input_ids).rstrip()

    # Force { as the start of JSON output
    if "{" not in text:
        return self._mask(scores, ["{"])

    if text.endswith("{"):
        return self._mask(scores, ['"answer"', ' "answer"'])

    # ... more state transitions ...

    if re.search(r'"confidence"\s*:\s*-?\d+(\.\d+)?$', text):
        return self._mask(scores, ["}"])

    return scores
```

### 2. JsonStopCriteria

Determines when to stop generation. Stops when:
- At least one `{` has been generated
- Braces are balanced (`{` count equals `}` count)
- Text ends with `}`

```python
class JsonStopCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return (
            text.count("{") > 0
            and text.count("{") == text.count("}")
            and text.strip().endswith("}")
        )
```

### 3. System Prompt Injection

When `response_format="json"` is specified, a system prompt is injected to guide the model:

```python
if response_format == "json":
    messages.append({
        "role": "system",
        "content": 'Return ONLY JSON: {"answer":"string","confidence":0.0}'
    })
```

This helps the model understand the expected format even before the logits processor constrains the output.

### 4. Post-Processing

After generation, the output is parsed and re-serialized for clean JSON:

```python
if response_format == "json":
    try:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start:end+1]
        parsed = json.loads(candidate)
        text = json.dumps(parsed)
    except:
        text = json.dumps({
            "answer": "",
            "confidence": 0.0,
            "error": "invalid_json",
            "raw_output": text[:200]
        })
```

## Usage

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | STRING | Yes | The question or instruction |
| `response_format` | STRING | No | Set to `"json"` to enable constrained decoding |
| `temperature` | FP32 | No | Sampling temperature (default: 0.7) |
| `top_p` | FP32 | No | Nucleus sampling parameter (default: 0.9) |
| `max_tokens` | INT32 | No | Maximum tokens to generate (default: 256) |
| `system_prompt` | STRING | No | Custom system prompt |

### Output

| Output | Type | Description |
|--------|------|-------------|
| `generated_text` | STRING | The generated response (JSON string if constrained) |
| `token_count` | INT32 | Number of tokens generated |

### Example: REST API

```bash
curl -X POST "http://localhost:8080/v2/models/tinyllama-python/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "prompt", "shape": [1], "datatype": "BYTES",
       "data": ["What is the capital of France?"]},
      {"name": "response_format", "shape": [1], "datatype": "BYTES",
       "data": ["json"]},
      {"name": "temperature", "shape": [1], "datatype": "FP32",
       "data": [0.1]}
    ]
  }'
```

**Response:**
```json
{
  "outputs": [
    {
      "name": "generated_text",
      "data": ["{\"answer\": \"Paris\", \"confidence\": 1}"]
    },
    {
      "name": "token_count",
      "data": [12]
    }
  ]
}
```

### Example: Python Client

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8080")

prompt = np.array(["What is 2 + 2?"], dtype=object)
response_format = np.array(["json"], dtype=object)

inputs = [
    httpclient.InferInput("prompt", prompt.shape, "BYTES"),
    httpclient.InferInput("response_format", response_format.shape, "BYTES"),
]
inputs[0].set_data_from_numpy(prompt)
inputs[1].set_data_from_numpy(response_format)

result = client.infer("tinyllama-python", inputs)
print(result.as_numpy("generated_text")[0].decode())
# Output: {"answer": "4", "confidence": 1}
```

### Example: Dedicated Client Script

```bash
# Run test queries (uses REST by default)
python scripts/clients/tinyllama_json_client.py

# Single query
python scripts/clients/tinyllama_json_client.py --prompt "What is the speed of light?"

# Compare JSON vs free-form output
python scripts/clients/tinyllama_json_client.py --compare

# Disable JSON constraint
python scripts/clients/tinyllama_json_client.py --prompt "Explain AI" --no-json

# Test all constrained decoding versions (1-6)
# Loads each version, runs inference, shows inputs/outputs, then unloads
python scripts/clients/tinyllama_json_client.py --test-all-versions

# Use gRPC instead of REST (local development)
python scripts/clients/tinyllama_json_client.py --grpc-url localhost:50051

# Use REST with explicit URL (local development)
python scripts/clients/tinyllama_json_client.py --rest-url http://localhost:8080

# Test batch inference (concurrent requests to test Triton dynamic batching)
python scripts/clients/tinyllama_json_client.py --test-batch      # 8 concurrent requests (default)
python scripts/clients/tinyllama_json_client.py --test-batch 4    # 4 concurrent requests
```

### Batch Inference Testing

The `--test-batch` option sends multiple concurrent requests to test Triton's dynamic batching feature.

**Run batch test:**
```bash
python scripts/clients/tinyllama_json_client.py --test-batch      # 8 concurrent requests (default)
python scripts/clients/tinyllama_json_client.py --test-batch 4    # 4 concurrent requests
```

**Example output (4 concurrent requests on CPU):**
```
######################################################################
#  Batch Inference Test - 4 Concurrent Requests
######################################################################

REST URL: http://localhost:8080
Model: tinyllama-python v1
Requests: 4

======================================================================
Sending requests concurrently...
======================================================================
  [1] Submitted: What is the capital of France?...
  [2] Submitted: What is 2 + 2?...
  [3] Submitted: Who wrote Romeo and Juliet?...
  [4] Submitted: What is the largest planet?...

======================================================================
Results
======================================================================

[1] What is the capital of France?...
    Response: {"answer": "Paris", "confidence": 1}...
    Tokens: 15, Time: 20237ms

[2] What is 2 + 2?...
    Response: {"answer": "4", "confidence": 1}...
    Tokens: 14, Time: 20236ms

[3] Who wrote Romeo and Juliet?...
    Response: {"answer": "string", "confidence": 0}...
    Tokens: 11, Time: 20236ms

[4] What is the largest planet?...
    Response: {"answer": "Jupiter", "confidence": 0}...
    Tokens: 16, Time: 20235ms

======================================================================
Summary
======================================================================
Total requests:     4
Successful:         4
Failed:             0
Total tokens:       56
Overall time:       20238ms
Avg time/request:   20236ms
Throughput:         0.20 req/s

Sequential estimate: 80944ms (sum of individual times)
Batch speedup:       4.00x
```

The **Batch speedup** shows how much faster concurrent requests are compared to sending them sequentially. A 4x speedup means all 4 requests were processed in the same batch.

## Batch Processing in Python Backend

> **Note:** For a comprehensive guide on batching in Triton Python backends, see [batching.md](./batching.md).

### How Dynamic Batching Works

```
Client 1 ─┐                              ┌─ Response 1
Client 2 ─┼─► Triton Dynamic ─► model.py ─┼─ Response 2
Client 3 ─┤     Batcher        execute() ├─ Response 3
Client 4 ─┘   (100ms queue)      loop    └─ Response 4
```

1. **Clients send requests** concurrently
2. **Triton accumulates** requests up to `max_batch_size` or `max_queue_delay` (100ms)
3. **Single `execute()` call** with batched inputs
4. **Model processes** each item in a loop
5. **Responses returned** to respective clients

### Configuration

**config.pbtxt** - Enable batching:
```protobuf
max_batch_size: 8

dynamic_batching {
  max_queue_delay_microseconds: 100000  # 100ms - wait for batch to fill
}

# Input/output dims must be per-element (Triton manages batch dim)
input [
  {
    name: "prompt"
    data_type: TYPE_STRING
    dims: [ 1 ]  # Single element per batch item (not [-1])
  }
]
```

### Model Code Changes for Batching

With `max_batch_size > 0`, input tensors have shape `[batch, element]` instead of `[element]`. The model must handle both shapes for backwards compatibility.

**Key changes in `model.py`:**

1. **Extract scalars from batched tensors:**
   ```python
   # Before (non-batched): tensor.as_numpy()[0] → scalar
   # After (batched):      tensor.as_numpy()[0] → [scalar]  (array!)

   # Solution: Use .flatten()[0] to handle both shapes
   def _get_optional_param(self, request, name, default, dtype=None):
       tensor = pb_utils.get_input_tensor_by_name(request, name)
       if tensor is None:
           return default
       value = tensor.as_numpy().flatten()[0]  # Works for [N] and [batch, 1]
       if dtype == str:
           return value.decode("utf-8") if isinstance(value, bytes) else str(value)
       return value
   ```

2. **Process batch of prompts:**
   ```python
   def execute(self, requests):
       for request in requests:
           prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
           prompt_np = prompt_tensor.as_numpy()

           # Flatten to handle both [N] and [batch, 1] shapes
           prompts_flat = prompt_np.flatten()

           generated_texts = []
           token_counts = []

           for i in range(len(prompts_flat)):
               prompt = prompts_flat[i]
               if isinstance(prompt, bytes):
                   prompt = prompt.decode("utf-8")

               # ... process each prompt ...
               generated_texts.append(result)
               token_counts.append(tokens)

           # Return batched outputs
           response = pb_utils.InferenceResponse(output_tensors=[
               pb_utils.Tensor("generated_text", np.array(generated_texts, dtype=object)),
               pb_utils.Tensor("token_count", np.array(token_counts, dtype=np.int32)),
           ])
   ```

### Client Code for Batched Requests

With `max_batch_size > 0`, clients must provide inputs with shape `[batch, element]`:

```python
# Before (max_batch_size: 0)
prompt_data = np.array(["What is 2+2?"], dtype=object)  # Shape: [1]

# After (max_batch_size > 0)
prompt_data = np.array([["What is 2+2?"]], dtype=object)  # Shape: [1, 1]

# Output extraction also changes
# Before: result.as_numpy("generated_text")[0]
# After:  result.as_numpy("generated_text").flatten()[0]
```

### Benefits of Batching

| Scenario | Without Batching | With Batching |
|----------|------------------|---------------|
| 4 sequential requests | ~80 seconds | ~80 seconds |
| 4 concurrent requests | ~80 seconds | ~20 seconds |
| Speedup | 1x | 4x |

**Note:** The speedup comes from overlapping I/O and model loading overhead, not from parallel computation (CPU/GPU still processes sequentially in the execute loop).

## Limitations

1. **Fixed schema**: The current implementation only supports the `{"answer": ..., "confidence": ...}` schema. Extending to arbitrary schemas would require a more sophisticated grammar-based approach.

2. **Token alignment**: The masking works at the token level, but tokens don't always align with JSON syntax boundaries. This can occasionally cause issues with certain tokenizers.

3. **Confidence values**: The model doesn't have a reliable way to estimate confidence. The `confidence` field is generated by the model based on its "intuition" rather than actual probability calculations.

4. **Performance overhead**: The logits processor adds some overhead to each generation step as it decodes and analyzes the generated text.

## Extending the Design

### Custom Schemas

To support different JSON schemas, you would need to:

1. Define a grammar or schema specification
2. Implement a state machine that tracks valid transitions
3. At each step, determine which tokens are valid continuations
4. Mask all other tokens

Libraries like [Outlines](https://github.com/outlines-dev/outlines) or [Guidance](https://github.com/guidance-ai/guidance) provide more sophisticated grammar-based constrained decoding.

### Alternative Approaches

1. **Grammar-based decoding**: Use a formal grammar (e.g., JSON grammar) to constrain output
2. **Beam search with validation**: Generate multiple candidates and select valid ones
3. **Fine-tuning**: Train the model to always output valid JSON for certain prompt patterns
4. **Post-hoc correction**: Use a separate model or rules to fix malformed JSON

## Runbook

### Prerequisites

```bash
# Start the stack
docker compose up -d

# Verify services are running
docker compose ps

# Check Triton is healthy
curl -s http://localhost:8080/v2/health/ready
```

### Step 1: Load the Model

```bash
# Load tinyllama-python model
curl -X POST "http://localhost:8080/v2/repository/models/tinyllama-python/load"

# Verify model is ready (may take 30-60 seconds for model download on first load)
curl -s "http://localhost:8080/v2/models/tinyllama-python/ready"
```

### Step 2: Test JSON Constrained Output

**Basic query with JSON constraint:**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is the capital of France?"]},
    {"name": "response_format", "shape": [1], "datatype": "BYTES", "data": ["json"]},
    {"name": "temperature", "shape": [1], "datatype": "FP32", "data": [0.1]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected output: {"answer": "Paris", "confidence": 1}
```

**Math question:**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is 2 + 2?"]},
    {"name": "response_format", "shape": [1], "datatype": "BYTES", "data": ["json"]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected output: {"answer": "4", "confidence": 1}
```

**Science question:**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is the largest planet in our solar system?"]},
    {"name": "response_format", "shape": [1], "datatype": "BYTES", "data": ["json"]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected output: {"answer": "Jupiter", "confidence": 1}
```

**History question:**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What year did World War II end?"]},
    {"name": "response_format", "shape": [1], "datatype": "BYTES", "data": ["json"]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected output: {"answer": "1945", "confidence": 1}
```

### Step 3: Test Free-Form Output (No JSON Constraint)

```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is the capital of France?"]},
    {"name": "temperature", "shape": [1], "datatype": "FP32", "data": [0.1]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected output: The capital of France is Paris.
```

### Step 4: Using the Python Client

**Run all test queries (REST):**
```bash
python scripts/clients/tinyllama_json_client.py
```

**Run all test queries (gRPC - local development):**
```bash
python scripts/clients/tinyllama_json_client.py --grpc-url localhost:50051
```

**Test all constrained decoding versions (1-6):**
```bash
# Loads each version individually, runs inference showing inputs/outputs, then unloads
python scripts/clients/tinyllama_json_client.py --test-all-versions
```

**Expected output:**
```
Using REST client: http://localhost:8080

======================================================================
TinyLlama JSON Constrained Decoding Test
Model: tinyllama-python v1
JSON Mode: Enabled
======================================================================

Q: What is the capital of France?
A: {
  "answer": "Paris",
  "confidence": 1
}
   (tokens: 15)

Q: What is 2 + 2?
A: {
  "answer": "4",
  "confidence": 1
}
   (tokens: 14)
...
```

**Single query:**
```bash
python scripts/clients/tinyllama_json_client.py --prompt "What is the speed of light?"
```

**Compare JSON vs free-form output:**
```bash
python scripts/clients/tinyllama_json_client.py --compare
```

**Expected output:**
```
COMPARISON MODE: JSON vs Free-form

Q: What is the capital of Japan?
--------------------------------------------------
JSON mode:      {"answer": "Tokyo", "confidence": 1}
Free-form mode: The capital of Japan is Tokyo.

Q: What is 10 * 5?
--------------------------------------------------
JSON mode:      {"answer": "50", "confidence": 1}
Free-form mode: Yes, 10 multiplied by 5 is 50.
```

**Disable JSON constraint:**
```bash
python scripts/clients/tinyllama_json_client.py --prompt "Explain quantum computing" --no-json
```

**Use gRPC instead of REST:**
```bash
python scripts/clients/tinyllama_json_client.py --grpc-url localhost:50051
```

**With custom temperature:**
```bash
python scripts/clients/tinyllama_json_client.py --prompt "Tell me a fact" --temperature 0.5
```

**Test batch inference (concurrent requests):**
```bash
# Test with 4 concurrent requests
python scripts/clients/tinyllama_json_client.py --test-batch 4

# Test with 8 concurrent requests (default)
python scripts/clients/tinyllama_json_client.py --test-batch
```

### Step 5: Using with Authentication (Kubernetes)

```bash
# Set API key
export DOMINO_USER_API_KEY="your-api-key"

# REST with auth
curl -s -X POST "http://your-proxy-url/v2/models/tinyllama-python/infer" \
  -H "Content-Type: application/json" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is 5 + 5?"]},
    {"name": "response_format", "shape": [1], "datatype": "BYTES", "data": ["json"]}
  ]}'

# Python client with auth (automatically uses DOMINO_USER_API_KEY)
python scripts/clients/tinyllama_json_client.py --rest-url http://your-proxy-url
```

### Step 6: Troubleshooting

**Check model status:**
```bash
curl -s "http://localhost:8080/v2/models/tinyllama-python" | jq .
```

**Check Triton logs:**
```bash
docker compose logs backend | tail -50
```

**Reload model after config changes:**
```bash
curl -X POST "http://localhost:8080/v2/repository/models/tinyllama-python/unload"
sleep 2
curl -X POST "http://localhost:8080/v2/repository/models/tinyllama-python/load"
```

**Test model health:**
```bash
curl -s "http://localhost:8080/v2/models/tinyllama-python/ready"
# Expected: Returns empty 200 OK when ready
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| JSON error with `raw_output` | Model didn't follow constraint | Lower temperature, rephrase prompt |
| Model not ready | Model still loading | Wait 30-60 seconds, check logs |
| Connection refused | Services not running | Run `docker compose up -d` |
| 401 Unauthorized | Missing or invalid API key | Set `DOMINO_USER_API_KEY` env var |
| Shape mismatch error | Client using wrong input shape | With batching: use shape `[1,1]` not `[1]` |
| `str` + `bytes` error in model | model.py not handling batched inputs | Use `.flatten()[0]` to extract scalars |

### Step 7: Testing Library-Specific Versions

The `tinyllama-python` model includes multiple versions, each using a different constrained decoding library:

| Version | Library | Description | Schemas |
|---------|---------|-------------|---------|
| 1 | Custom LogitsProcessor | Manual JSON constraint (default) | N/A (uses `response_format`) |
| 2 | Outlines | JSON Schema, regex, grammar via token masking | qa, entity, sentiment, regex_phone |
| 3 | Guidance | Template DSL with gen/select blocks | qa, chain_of_thought, classification |
| 4 | LMQL | SQL-like query language for constraints | basic, constrained, scripted |
| 5 | Jsonformer | Structural JSON generation | qa, person, product, list_items |
| 6 | Instructor | Pydantic validation with retry | qa, extraction, analysis |

**Note:** Versions 2-6 require constrained decoding libraries installed system-wide in the Triton container. Use `Dockerfile.triton.local` for local development.

#### Testing Workflow

To test a specific version:

1. **Edit config.pbtxt** to load the desired version:
   ```bash
   # Edit triton-repo/models/tinyllama-python/config.pbtxt
   # Change the version_policy line to:
   version_policy { specific { versions: [2] } }  # For version 2
   ```

2. **Reload the model:**
   ```bash
   curl -X POST "http://localhost:8080/v2/repository/models/tinyllama-python/unload"
   curl -X POST "http://localhost:8080/v2/repository/models/tinyllama-python/load"
   ```

3. **Check model is ready:**
   ```bash
   curl -s "http://localhost:8080/v2/models/tinyllama-python" | jq .
   ```

4. **Run inference with Python client:**

#### Version 1 (Custom LogitsProcessor) - Default
```bash
# Edit config.pbtxt: version_policy { specific { versions: [1] } }
# Reload model, then:
python scripts/clients/tinyllama_json_client.py --version 1 --prompt "What is 2+2?"
python scripts/clients/tinyllama_json_client.py --version 1 --prompt "What is the capital of France?"
python scripts/clients/tinyllama_json_client.py --version 1 --no-json --prompt "Explain quantum computing"
```

#### Version 2 (Outlines) - JSON Schema/Regex
```bash
# Edit config.pbtxt: version_policy { specific { versions: [2] } }
# Reload model, then:
python scripts/clients/tinyllama_json_client.py --version 2 --schema qa --prompt "What is 2+2?"
python scripts/clients/tinyllama_json_client.py --version 2 --schema sentiment --prompt "I love this product!"
python scripts/clients/tinyllama_json_client.py --version 2 --schema entity --prompt "Apple was founded by Steve Jobs in Cupertino"
```

#### Version 3 (Guidance) - Template DSL
```bash
# Edit config.pbtxt: version_policy { specific { versions: [3] } }
# Reload model, then:
python scripts/clients/tinyllama_json_client.py --version 3 --schema qa --prompt "What is the speed of light?"
python scripts/clients/tinyllama_json_client.py --version 3 --schema chain_of_thought --prompt "Why is the sky blue?"
python scripts/clients/tinyllama_json_client.py --version 3 --schema classification --prompt "The mitochondria is the powerhouse of the cell"
```

#### Version 4 (LMQL) - SQL-like Queries
```bash
# Edit config.pbtxt: version_policy { specific { versions: [4] } }
# Reload model, then:
python scripts/clients/tinyllama_json_client.py --version 4 --schema basic --prompt "What is the largest ocean?"
python scripts/clients/tinyllama_json_client.py --version 4 --schema constrained --prompt "Explain why the sky is blue"
python scripts/clients/tinyllama_json_client.py --version 4 --schema scripted --prompt "What is 15 + 27?"
```

#### Version 5 (Jsonformer) - Structural JSON
```bash
# Edit config.pbtxt: version_policy { specific { versions: [5] } }
# Reload model, then:
python scripts/clients/tinyllama_json_client.py --version 5 --schema qa --prompt "What is the chemical symbol for gold?"
python scripts/clients/tinyllama_json_client.py --version 5 --schema person --prompt "Tell me about Albert Einstein"
python scripts/clients/tinyllama_json_client.py --version 5 --schema product --prompt "Describe the iPhone 15 Pro"
```

#### Version 6 (Instructor) - Pydantic Validation
```bash
# Edit config.pbtxt: version_policy { specific { versions: [6] } }
# Reload model, then:
python scripts/clients/tinyllama_json_client.py --version 6 --schema qa --prompt "What year did World War II end?"
python scripts/clients/tinyllama_json_client.py --version 6 --schema extraction --prompt "Apple Inc. was founded by Steve Jobs in Cupertino"
python scripts/clients/tinyllama_json_client.py --version 6 --schema analysis --prompt "The economy grew by 3% last quarter"
```

#### Using curl (Alternative)

**Test Version 2 (Outlines):**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/2/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is the capital of France?"]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["qa"]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected: {"answer": "Paris", "confidence": 0.95}
```

**Test Version 3 (Guidance) - Chain of Thought:**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/3/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Why is the sky blue?"]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["chain_of_thought"]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected: {"reasoning": {...}, "answer": "...", "confidence": "high"}
```

**Test Version 4 (LMQL) - Scripted Query:**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/4/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is 15 + 27?"]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["scripted"]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected: {"type": "math problem", "steps": [...], "result": "42"}
```

**Test Version 5 (Jsonformer) - Person Extraction:**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/5/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Tell me about Marie Curie"]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["person"]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected: {"name": "Marie Curie", "occupation": "physicist", "nationality": "Polish-French", ...}
```

**Test Version 6 (Instructor) - Entity Extraction:**
```bash
curl -s -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/6/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Extract entities: Apple Inc. was founded by Steve Jobs in Cupertino."]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["extraction"]}
  ]}' | jq -r '.outputs[0].data[0]'

# Expected: {"entities": [{"name": "Apple Inc.", "type": "organization"}, ...], "source_text": "..."}
```

**Using Python client with specific version:**
```python
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8080")

# Use version 2 (Outlines)
result = client.infer(
    "tinyllama-python",
    inputs,
    model_version="2"  # Specify version
)
```

**Available schemas per version:**

| Version | schema_name options |
|---------|---------------------|
| 2 (Outlines) | `qa`, `entity`, `sentiment`, `regex_phone` |
| 3 (Guidance) | `qa`, `chain_of_thought`, `classification` |
| 4 (LMQL) | `basic`, `constrained`, `scripted` |
| 5 (Jsonformer) | `qa`, `person`, `product`, `list_items` |
| 6 (Instructor) | `qa`, `extraction`, `analysis` |

## Constrained Decoding Libraries

The `tinyllama-python` model uses a custom `LogitsProcessor` for JSON constraint. For production use cases requiring more flexibility, several open-source libraries provide sophisticated constrained decoding. Example Triton models for each library are available in `scripts/models/`.

### Library Comparison

| Library | Approach | Guarantee | Best For |
|---------|----------|-----------|----------|
| **Outlines** | Token masking | 100% valid | Production JSON APIs |
| **Guidance** | Template DSL | 100% valid | Complex templates |
| **LMQL** | Query language | 100% valid | Declarative constraints |
| **Jsonformer** | Structural gen | 100% valid | Simple JSON, low overhead |
| **Instructor** | Validation+retry | High (retries) | API-based models |

### How Each Library Works

#### 1. Outlines (`scripts/models/outlines-example/`)

**Approach**: Compiles JSON Schema or regex into a finite state machine (FSM), then masks invalid tokens at each generation step.

```
┌─────────────────────────────────────────────────────────────┐
│                    Outlines Architecture                     │
│                                                              │
│  JSON Schema / Regex / Grammar                               │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                         │
│  │ Compile to FSM  │  (done once at initialization)          │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Generation Loop                         │    │
│  │  ┌─────────────┐    ┌──────────────┐    ┌────────┐  │    │
│  │  │ Current FSM │───▶│ Valid tokens │───▶│  Mask  │  │    │
│  │  │   state     │    │  from state  │    │ others │  │    │
│  │  └─────────────┘    └──────────────┘    └───┬────┘  │    │
│  │                                             │       │    │
│  │                     ┌───────────────────────┘       │    │
│  │                     ▼                               │    │
│  │              ┌─────────────┐                        │    │
│  │              │ Sample token│ (only valid options)   │    │
│  │              └──────┬──────┘                        │    │
│  │                     │                               │    │
│  │                     ▼                               │    │
│  │              ┌─────────────┐                        │    │
│  │              │ Update FSM  │                        │    │
│  │              │   state     │                        │    │
│  │              └─────────────┘                        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key Code**:
```python
from outlines import models, generate
from pydantic import BaseModel

class QAResponse(BaseModel):
    answer: str
    confidence: float

model = models.transformers("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
generator = generate.json(model, QAResponse)  # Compiles schema to FSM

result = generator("What is 2+2?")  # Returns QAResponse instance
# result.answer = "4", result.confidence = 0.95
```

**Supported Constraints**:
- JSON Schema (via Pydantic models)
- Regular expressions
- Context-free grammars
- Finite choice sets

---

#### 2. Guidance (`scripts/models/guidance-example/`)

**Approach**: Uses a domain-specific language (DSL) embedded in Python strings. Templates contain static text interleaved with `{{gen}}` and `{{select}}` blocks.

```
┌─────────────────────────────────────────────────────────────┐
│                    Guidance Architecture                     │
│                                                              │
│  Template: "Answer: {{gen 'answer'}} Confidence: {{select}}" │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Template Execution                      │    │
│  │                                                      │    │
│  │  "Answer: "  ──▶  Output literal text                │    │
│  │       │                                              │    │
│  │       ▼                                              │    │
│  │  {{gen 'answer'}}  ──▶  Generate until stop token    │    │
│  │       │                   Store in variable 'answer' │    │
│  │       ▼                                              │    │
│  │  " Confidence: "  ──▶  Output literal text           │    │
│  │       │                                              │    │
│  │       ▼                                              │    │
│  │  {{select ['high','low']}}  ──▶  Mask to choices     │    │
│  │                                  Sample one option   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Result: lm['answer'] = "4", lm['confidence'] = "high"       │
└─────────────────────────────────────────────────────────────┘
```

**Key Code**:
```python
from guidance import models, gen, select

llm = models.Transformers("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Template with generation and selection
lm = llm + f'''Question: {question}
Answer: {{gen 'answer' max_tokens=50 stop='.'}}
Confidence: {{select ['high', 'medium', 'low'] name='confidence'}}'''

print(lm['answer'])      # Generated answer
print(lm['confidence'])  # One of: high, medium, low
```

**Supported Constraints**:
- `gen`: Free-form generation with stop conditions
- `select`: Choose from predefined options
- Python control flow (`if`, `for`, `while`)
- Regex patterns within gen blocks

---

#### 3. LMQL (`scripts/models/lmql-example/`)

**Approach**: SQL-like query language for LLMs. Constraints are specified in `WHERE` clauses using type predicates.

```
┌─────────────────────────────────────────────────────────────┐
│                      LMQL Architecture                       │
│                                                              │
│  Query:                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ "Question: {q}"                                      │    │
│  │ "Answer: [ANSWER]" where STOPS_AT(ANSWER, ".")       │    │
│  │ "Score: [SCORE]" where INT(SCORE) and SCORE <= 10    │    │
│  └─────────────────────────────────────────────────────┘    │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Constraint Evaluation                   │    │
│  │                                                      │    │
│  │  For each token:                                     │    │
│  │  1. Check if adding token satisfies constraints      │    │
│  │  2. Mask tokens that would violate constraints       │    │
│  │  3. Sample from valid tokens                         │    │
│  │                                                      │    │
│  │  STOPS_AT(X, ".")  →  Stop when "." generated        │    │
│  │  INT(X)           →  Only allow digit tokens         │    │
│  │  X <= 10          →  Semantic constraint on value    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key Code**:
```python
import lmql

@lmql.query(model="local:TinyLlama/TinyLlama-1.1B-Chat-v1.0")
async def qa_query(question):
    '''lmql
    "Question: {question}\n"
    "Answer: [ANSWER]" where STOPS_AT(ANSWER, ".")
    "Confidence (1-10): [CONF]" where INT(CONF) and CONF >= 1 and CONF <= 10
    return {"answer": ANSWER, "confidence": CONF}
    '''

result = asyncio.run(qa_query("What is 2+2?"))
```

**Supported Constraints**:
- `STOPS_AT(var, str)`: Stop at string
- `STOPS_BEFORE(var, str)`: Stop before string
- `INT(var)`, `FLOAT(var)`: Type constraints
- `len(var) < N`: Length constraints
- `var in [...]`: Choice constraints
- Logical operators: `and`, `or`, `not`

---

#### 4. Jsonformer (`scripts/models/jsonformer-example/`)

**Approach**: Generates JSON structure deterministically (braces, keys, colons), only calling the LLM to fill in values. This is the most lightweight approach.

```
┌─────────────────────────────────────────────────────────────┐
│                   Jsonformer Architecture                    │
│                                                              │
│  Schema: {"answer": str, "confidence": float}               │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Structural Generation                   │    │
│  │                                                      │    │
│  │  Step 1: Output "{"           (deterministic)        │    │
│  │  Step 2: Output '"answer":'   (deterministic)        │    │
│  │  Step 3: Output '"'           (deterministic)        │    │
│  │  Step 4: Call LLM for string  ──▶ "Paris"           │    │
│  │  Step 5: Output '",'          (deterministic)        │    │
│  │  Step 6: Output '"confidence":' (deterministic)      │    │
│  │  Step 7: Call LLM for number  ──▶ 0.95              │    │
│  │  Step 8: Output "}"           (deterministic)        │    │
│  │                                                      │    │
│  │  Result: {"answer": "Paris", "confidence": 0.95}     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key Code**:
```python
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    }
}

jsonformer = Jsonformer(model, tokenizer, schema, "What is 2+2?")
result = jsonformer()  # {"answer": "4", "confidence": 0.9}
```

**Supported Constraints**:
- JSON Schema with: `string`, `number`, `boolean`, `array`, `object`
- Nested objects and arrays
- No regex or grammar support (JSON only)

---

#### 5. Instructor (`scripts/models/instructor-example/`)

**Approach**: Uses Pydantic models for schema definition, generates output, validates, and retries with error feedback if validation fails. Designed for OpenAI API but adaptable to local models.

```
┌─────────────────────────────────────────────────────────────┐
│                   Instructor Architecture                    │
│                                                              │
│  Pydantic Model: class QA(BaseModel): answer: str, conf: float
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Generation + Validation Loop            │    │
│  │                                                      │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │ Attempt 1                                     │   │    │
│  │  │  1. Include schema in prompt                  │   │    │
│  │  │  2. Generate response from LLM                │   │    │
│  │  │  3. Parse JSON from response                  │   │    │
│  │  │  4. Validate with Pydantic                    │   │    │
│  │  │     ├─ Success → Return result                │   │    │
│  │  │     └─ Failure → Retry with error message     │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  │                     │                                │    │
│  │                     ▼ (on failure)                   │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │ Attempt 2                                     │   │    │
│  │  │  Prompt: "Previous error: {error}. Fix it."   │   │    │
│  │  │  ... repeat validation ...                    │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  │                     │                                │    │
│  │                     ▼ (max retries)                  │    │
│  │              Raise ValidationError                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key Code**:
```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

class QAResponse(BaseModel):
    answer: str
    confidence: float

# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Automatic validation and retry
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=QAResponse,
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_retries=3
)
# response.answer = "4", response.confidence = 0.95
```

**Supported Constraints**:
- Full Pydantic validation (types, ranges, regex, custom validators)
- Nested models
- Automatic retry with error feedback
- Works best with instruction-following models

---

### Choosing a Library

| Use Case | Recommended Library |
|----------|---------------------|
| Production JSON APIs | **Outlines** - Most mature, efficient FSM compilation |
| Complex multi-step output | **Guidance** - Natural template syntax |
| Research/experimentation | **LMQL** - Clean declarative syntax |
| Simple JSON, minimal deps | **Jsonformer** - Lightweight, fast |
| OpenAI API integration | **Instructor** - Designed for APIs |
| Custom grammars (SQL, code) | **Outlines** - CFG support |

### Installation

```bash
# Install all libraries
pip install -r docker/requirements-constrained-decoding.txt

# Or install individually
pip install outlines      # Token masking
pip install guidance      # Template DSL
pip install lmql          # Query language
pip install jsonformer    # Structural JSON
pip install instructor    # Pydantic validation
```

### Deploying Example Models

```bash
# Copy an example to the Triton model repository
cp -r scripts/models/outlines-example triton-repo/models/

# Load the model
curl -X POST "http://localhost:8080/v2/repository/models/outlines-example/load"

# Test
curl -X POST "http://localhost:8080/v2/models/outlines-example/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is the capital of France?"]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["qa"]}
  ]}'
```

## References

- [Hugging Face LogitsProcessor](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.LogitsProcessor)
- [Outlines - Structured Generation](https://github.com/outlines-dev/outlines)
- [Guidance - Constrained Generation](https://github.com/guidance-ai/guidance)
- [LMQL - Language Model Query Language](https://github.com/eth-sri/lmql)
- [Jsonformer - Structured JSON Generation](https://github.com/1rgs/jsonformer)
- [Instructor - Structured Output for LLMs](https://github.com/jxnl/instructor)
- [NVIDIA Triton Python Backend](https://github.com/triton-inference-server/python_backend)
