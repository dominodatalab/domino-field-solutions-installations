# TensorRT-LLM Proof of Concept

A minimal guide to deploy TinyLlama-1.1B with TensorRT-LLM from a Domino workspace.

---

## Prerequisites

### Critical: GPU Architecture Must Match

**TensorRT engines are NOT portable between GPU architectures.** The engine must be built on the same GPU type it will run on.

| GPU | Compute Capability | Instance Types |
|-----|-------------------|----------------|
| V100 | 7.0 | p3.2xlarge, p3.8xlarge |
| T4 | 7.5 | g4dn.xlarge |
| A10G | 8.6 | g5.xlarge |
| L4 | 8.9 | g6.xlarge, g6f.xlarge |
| A100 | 8.0 | p4d.24xlarge |

**Before building the engine:**
1. Check your workspace GPU: `nvidia-smi`
2. Check Triton deployment's `instance_type` in `values-dev.yaml`
3. **They must match!**

Example: If your Domino workspace has an L4 GPU, configure Triton to run on L4 nodes:
```yaml
# values-dev.yaml
env:
  instance_type: g6f.xlarge  # L4 GPU nodes
```

### Required Domino Environment

Your Domino workspace must use an environment based on NVIDIA's TensorRT-LLM image:

```
Base image: nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3
```

This image includes TensorRT-LLM, CUDA, OpenMPI, and all dependencies pre-installed.

### Required Hardware

- **GPU**: NVIDIA GPU with 8GB+ VRAM (must match Triton deployment target)
- **EDV**: Mount `domino-inference-dev-triton-repo-pvc` for model storage

### Verify Your Environment

```bash
# Check GPU is available
nvidia-smi

# Check TensorRT-LLM is installed (use system packages)
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
python -c "import tensorrt_llm; print(f'TensorRT-LLM: {tensorrt_llm.__version__}')"
```

---

## Model: TinyLlama-1.1B

| Property | Value |
|----------|-------|
| Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Parameters | 1.1B |
| Architecture | Llama |
| License | Apache 2.0 (open, no HuggingFace token needed) |
| GPU Memory | ~3GB (FP16) |

---

## Step 1: Set Up Environment

```bash
# IMPORTANT: Use system-installed packages, not user site-packages
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH

# Verify TensorRT-LLM
python -c "import tensorrt_llm; print(f'TensorRT-LLM: {tensorrt_llm.__version__}')"

# Install huggingface_hub for model download
pip install huggingface_hub
```

---

## Step 2: Download TinyLlama

```bash
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --local-dir ~/models/tinyllama-1.1b
```

---

## Step 3: Convert to TensorRT-LLM Checkpoint

The conversion scripts are not included in the pip package. Clone the TensorRT-LLM repo:

```bash
cd ~
git clone https://github.com/NVIDIA/TensorRT-LLM.git --depth 1 --branch v0.14.0

# Convert HuggingFace model to TRT-LLM checkpoint format
python ~/TensorRT-LLM/examples/llama/convert_checkpoint.py \
    --model_dir ~/models/tinyllama-1.1b \
    --output_dir ~/checkpoints/tinyllama-1.1b \
    --dtype float16

# Verify checkpoint was created
ls ~/checkpoints/tinyllama-1.1b/
# Should see: config.json, rank0.safetensors
```

---

## Step 4: Build TensorRT Engine

```bash
trtllm-build \
    --checkpoint_dir ~/checkpoints/tinyllama-1.1b \
    --output_dir ~/engines/tinyllama-1.1b \
    --gemm_plugin float16 \
    --max_batch_size 2 \
    --max_input_len 512 \
    --max_seq_len 640

# Verify engine was created
ls -lh ~/engines/tinyllama-1.1b/
# Should see: rank0.engine (~2-3GB)
```

**Build parameters:**
- `max_batch_size: 2` - Concurrent requests
- `max_input_len: 512` - Max input tokens
- `max_seq_len: 640` - Total sequence (input + output = 512 + 128)


Zip it all if you want download binaries

```bash
tar -czvf tinyllama-trtllm.tar.gz \                                                                                                                                                                                              
      engines/tinyllama-1.1b \                                                                                                                                                                                                     
      models/tinyllama-1.1b/tokenizer.json \                                                                                                                                                                                       
      models/tinyllama-1.1b/tokenizer_config.json \                                                                                                                                                                                
      models/tinyllama-1.1b/special_tokens_map.json
      
#or

tar -czvf tinyllama-trtllm-full.tar.gz engines models      
```
---

## Step 5: Copy to Model Repository (EDV)

```bash
# Your EDV mount path
EDV_PATH=/mnt/data/domino-inference-dev-triton-repo

# Create model directory
mkdir -p $EDV_PATH/models/tinyllama-trtllm/1
mkdir -p $EDV_PATH/models/tinyllama-trtllm/tensorrt_llm

# Copy engine files
cp /mnt/models/tinyllama-trtllm/engines/tinyllama-1.1b/* $EDV_PATH/models/tinyllama-trtllm/tensorrt_llm/

cp /mnt/models/tinyllama-trtllm/engines/tinyllama-1.1b/rank0.engine $EDV_PATH/models/tinyllama-trtllm/1/

# Copy tokenizer (for client-side encoding)
cp /mnt/models/tinyllama-trtllm/models/tinyllama-1.1b/tokenizer.json $EDV_PATH/models/tinyllama-trtllm/
cp /mnt/models/tinyllama-trtllm/models/tinyllama-1.1b/tokenizer_config.json $EDV_PATH/models/tinyllama-trtllm/
cp /mnt/models/tinyllama-trtllm/models/tinyllama-1.1b/special_tokens_map.json $EDV_PATH/models/tinyllama-trtllm/
cp /mnt/models/tinyllama-trtllm/config.pbtxt $EDV_PATH/models/tinyllama-trtllm/
```

---

## Step 6: Create Triton Config

Create `$EDV_PATH/models/tinyllama-trtllm/config.pbtxt`:

```protobuf
name: "tinyllama-trtllm"
backend: "tensorrtllm"
max_batch_size: 2

model_transaction_policy {
  decoupled: True
}

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
  },
  {
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
  },
  {
    name: "end_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "pad_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  }
]

output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1, -1 ]
  },
  {
    name: "sequence_length"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

parameters {
  key: "gpt_model_type"
  value: { string_value: "inflight_fused_batching" }
}

parameters {
  key: "gpt_model_path"
  value: { string_value: "/triton-repo/models/tinyllama-trtllm/tensorrt_llm" }
}

parameters {
  key: "batching_type"
  value: { string_value: "inflight_fused_batching" }
}
```

---

## Step 7: Deploy Triton with TensorRT-LLM Backend

Update `helm/domino-triton/values-dev.yaml`:

```yaml
triton_inference_server:
  image: "nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3"
  replicas: 1
```

Deploy:

```bash
helm upgrade domino-triton helm/domino-triton/ \
    -n domino-inference-dev \
    -f helm/domino-triton/values-dev.yaml
```

---

## Step 8: Test

### Load and Test the Model

```bash
# Load model
curl -X POST "http://$TRITON_REST_URL/admin/models/tinyllama-trtllm/load"

# Quick test with raw tokens ("Hello" = [1, 15043])
curl -X POST "http://$TRITON_REST_URL/v2/models/tinyllama-trtllm/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "input_ids", "shape": [1, 2], "datatype": "INT32", "data": [1, 15043]},
      {"name": "input_lengths", "shape": [1], "datatype": "INT32", "data": [2]},
      {"name": "request_output_len", "shape": [1], "datatype": "INT32", "data": [20]}
    ]
  }'
```

### Python Test Client

```python
#!/usr/bin/env python3
"""TensorRT-LLM test client."""
import requests
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
prompt = "What is the capital of France?"
input_ids = tokenizer.encode(prompt)

url = "http://localhost:8080/v2/models/tinyllama-trtllm/infer"
payload = {
    "inputs": [
        {"name": "input_ids", "shape": [1, len(input_ids)], "datatype": "INT32", "data": input_ids},
        {"name": "input_lengths", "shape": [1], "datatype": "INT32", "data": [len(input_ids)]},
        {"name": "request_output_len", "shape": [1], "datatype": "INT32", "data": [50]},
    ]
}

result = requests.post(url, json=payload).json()
output_ids = result["outputs"][0]["data"]
print(f"Response: {tokenizer.decode(output_ids, skip_special_tokens=True)}")
```

---

## Benchmarking: Python vs TensorRT-LLM

Compare the same model (TinyLlama-1.1B) on both backends:

```bash
# Run the benchmark script
python scripts/benchmarks/benchmark_tinyllama.py
```

### Expected Results

| Metric | Python (HuggingFace) | TensorRT-LLM | Speedup |
|--------|---------------------|--------------|---------|
| First token latency | ~500ms | ~50ms | **10x** |
| Throughput | 30-50 tok/s | 500-1000 tok/s | **15-20x** |
| GPU Memory | ~3GB | ~3GB | same |

### Why TensorRT-LLM is Faster

| Optimization | Python Backend | TensorRT-LLM |
|--------------|----------------|--------------|
| Kernel fusion | None | Fused attention, MLP |
| Memory layout | PyTorch default | Optimized for GPU cache |
| KV cache | Naive | Paged, contiguous |
| Batching | Static | In-flight (continuous) |

---

## Troubleshooting

### "No module named tensorrt_llm"

The system packages aren't in Python's path. Fix:

```bash
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
```

### CUDA initialization failure (error 35)

Your workspace doesn't have a GPU allocated. Start a new workspace with GPU hardware selected.

### Conflicting user-installed packages

If you pip-installed tensorrt or tensorrt-llm, they may conflict with system versions:

```bash
# Remove user-installed packages
pip uninstall tensorrt tensorrt-llm tensorrt-cu12 -y
rm -rf ~/.local/lib/python3.10/site-packages/tensorrt*

# Use system packages
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
```

### Engine build OOM

Reduce batch size and sequence length:

```bash
trtllm-build ... --max_batch_size 1 --max_seq_len 384
```

---

## Summary

| Step | Command | Time |
|------|---------|------|
| 1. Set up environment | `export PYTHONPATH=...` | 1 min |
| 2. Download model | `huggingface-cli download` | 2-3 min |
| 3. Convert checkpoint | `convert_checkpoint.py` | 1-2 min |
| 4. Build engine | `trtllm-build` | 5-10 min |
| 5. Copy to EDV | `cp` | 1 min |
| 6. Create config | manual | 1 min |
| 7. Deploy Triton | `helm upgrade` | 2-3 min |
| **Total** | | **~15-20 min** |

---

## Files Created

```
$EDV_PATH/models/tinyllama-trtllm/
├── config.pbtxt                 # Triton config
├── tokenizer.json               # For client-side encoding
├── tokenizer_config.json
├── special_tokens_map.json
├── 1/                           # Empty (Triton requirement)
└── tensorrt_llm/
    ├── config.json              # Generated by trtllm-build
    └── rank0.engine             # TensorRT engine (~2-3GB)
```
