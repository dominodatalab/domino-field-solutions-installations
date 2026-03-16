# LLM Support

## Backend Comparison

| Backend | Performance | Complexity | Use Case |
|---------|-------------|------------|----------|
| **TensorRT-LLM** | Best (10-20x faster) | High | Production |
| **vLLM** | Very Good | Medium | Flexible serving |
| **Python Backend** | Baseline | Low | Prototyping |

---

## TensorRT-LLM Quick Start

Build and deploy TinyLlama-1.1B from a Domino workspace:

```bash
# 1. Start workspace with GPU and TensorRT-LLM image
#    Image: nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3
#    IMPORTANT: Use same GPU type as inference cluster

# 2. Clone TensorRT-LLM for conversion scripts
cd /mnt && git clone --depth 1 --branch v0.14.0 https://github.com/NVIDIA/TensorRT-LLM.git

# 3. Download model
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0').save_pretrained('/mnt/models/tinyllama-hf')
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0').save_pretrained('/mnt/models/tinyllama-hf')
"

# 4. Convert to TRT-LLM checkpoint
python3 /mnt/TensorRT-LLM/examples/llama/convert_checkpoint.py \
    --model_dir /mnt/models/tinyllama-hf \
    --output_dir /mnt/models/tinyllama-ckpt \
    --dtype float16

# 5. Build TensorRT engine
trtllm-build \
    --checkpoint_dir /mnt/models/tinyllama-ckpt \
    --output_dir /mnt/models/tinyllama-engine \
    --gemm_plugin float16 \
    --max_batch_size 8 \
    --max_input_len 512 \
    --max_seq_len 1024

# 6. Copy to model repository
mkdir -p $TF/models/tinyllama-trtllm/{1,tensorrt_llm}
cp -r /mnt/models/tinyllama-engine/* $TF/models/tinyllama-trtllm/tensorrt_llm/

# 7. Test
python scripts/clients/tinyllama_trtllm_grpc_client.py --prompt "Hello!"
```

See [tensorrt_llm_deployment_guide.md](tensorrt_llm_deployment_guide.md) for GPU architecture compatibility and Kubernetes deployment details.

---

## TensorRT-LLM config.pbtxt

```protobuf
name: "tinyllama-trtllm"
backend: "tensorrtllm"
max_batch_size: 8

model_transaction_policy { decoupled: True }

input [
  { name: "input_ids", data_type: TYPE_INT32, dims: [ -1 ] },
  { name: "input_lengths", data_type: TYPE_INT32, dims: [ 1 ], reshape: { shape: [ ] } },
  { name: "request_output_len", data_type: TYPE_INT32, dims: [ 1 ], reshape: { shape: [ ] } },
  { name: "end_id", data_type: TYPE_INT32, dims: [ 1 ], reshape: { shape: [ ] }, optional: true },
  { name: "pad_id", data_type: TYPE_INT32, dims: [ 1 ], reshape: { shape: [ ] }, optional: true },
  { name: "temperature", data_type: TYPE_FP32, dims: [ 1 ], reshape: { shape: [ ] }, optional: true }
]

output [
  { name: "output_ids", data_type: TYPE_INT32, dims: [ -1, -1 ] },
  { name: "sequence_length", data_type: TYPE_INT32, dims: [ -1 ] }
]

instance_group [{ count: 1, kind: KIND_GPU, gpus: [ 0 ] }]

parameters { key: "gpt_model_type", value: { string_value: "inflight_fused_batching" } }
parameters { key: "gpt_model_path", value: { string_value: "/triton-repo/models/tinyllama-trtllm/tensorrt_llm" } }
parameters { key: "batching_type", value: { string_value: "inflight_fused_batching" } }
```

---

## vLLM Backend

For a middle ground between performance and simplicity:

```python
# models/llama-vllm/1/model.py
from vllm import LLM, SamplingParams
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.llm = LLM(
            model="meta-llama/Llama-2-7b-hf",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()[0].decode()
            params = SamplingParams(temperature=0.7, max_tokens=100)
            output = self.llm.generate([text], params)[0].outputs[0].text
            responses.append(pb_utils.InferenceResponse([
                pb_utils.Tensor("generated_text", np.array([output.encode()], dtype=np.object_))
            ]))
        return responses
```

---

## Python Backend (Prototyping)

Simplest approach using HuggingFace transformers directly:

```python
# models/llama-python/1/model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def execute(self, requests):
        # ... tokenize, generate, decode
```

---

## Performance Comparison (A100)

| Backend | Throughput | First Token Latency |
|---------|------------|---------------------|
| TensorRT-LLM | 2000-3000 tok/s | 50-100ms |
| vLLM | 1500-2000 tok/s | 100-150ms |
| Python (HF) | 50-100 tok/s | 500-1000ms |

## GPU Memory Requirements

| Model Size | FP16 | INT8 | INT4 |
|------------|------|------|------|
| 7B | 14GB | 7GB | 4GB |
| 13B | 26GB | 13GB | 7GB |
| 70B | 140GB | 70GB | 35GB |

---

## Troubleshooting

### "Failed to deserialize cuda engine"

Engine built on different GPU architecture. Rebuild on target GPU (V100=sm_70, A100=sm_80, L4=sm_89).

### "No module named tensorrt_llm.commands.convert_checkpoint"

Conversion scripts aren't pip-installed. Clone TensorRT-LLM repo and use example scripts directly.

### "at least one version must be available"

Missing `1/` directory in model folder. Create it (can be empty for TRT-LLM models).

### Version Compatibility

| Triton Image | TensorRT | TensorRT-LLM |
|--------------|----------|--------------|
| 24.10-trtllm-python-py3 | 10.4.0 | 0.14.0 |
| 24.08-trtllm-python-py3 | 10.3.0 | 0.12.0 |
