# Triton Inference Server Runbook

## Table of Contents

- [Quick Start: Easy Testing](#quick-start-easy-testing)
- [Local Testing (Docker Compose)](#local-testing-docker-compose)
- [Remote Usage (Kubernetes/Domino)](#remote-usage-kubernetesdomino)
  - [1. Environment Variables](#1-environment-variables)
  - [2. Download Models](#2-download-models)
  - [3. Copy Models to EDV](#3-copy-models-to-edv)
  - [4. Invoke Models](#4-invoke-models)
  - [5. Run Benchmarks](#5-run-benchmarks)
- [Reference](#reference)

---

## Quick Start: Easy Testing

The fastest way to test models is using the sequential test script. This script handles copying, loading, testing, and unloading models one at a time to avoid GPU memory issues.

### Prerequisites

```bash
# Set environment variables
export DOMINO_USER_API_KEY="your-api-key"
export TRITON_REST_URL="http://triton-inference-server-proxy.domino-inference-dev.svc.cluster.local:8080"
export TRITON_GRPC_URL="triton-inference-server-proxy.domino-inference-dev.svc.cluster.local:50051"
export SOURCE_TRITON_ROOT="/mnt/triton-repo"
export TARGET_TRITON_ROOT="/domino/edv/domino-inference-dev-triton-repo-pvc"

# Install dependencies
pip install -r docker/requirements-client.txt
```

### Test All Models (One at a Time)

```bash
# Test all models on GPU (recommended - avoids memory issues)
python scripts/testing/test_models_sequential.py

# Test all models on CPU
python scripts/testing/test_models_sequential.py --cpu

# Quick test without benchmarks
python scripts/testing/test_models_sequential.py --skip-benchmark
```

### Test Specific Models

```bash
# Test specific models
python scripts/testing/test_models_sequential.py --models bert-base-uncased yolov8n

# Test with specific GPU device
python scripts/testing/test_models_sequential.py --models tinyllama-trtllm --device 1

# Skip copy step (models already on EDV)
python scripts/testing/test_models_sequential.py --models tinyllama-python --skip-copy

# Keep models loaded after testing
python scripts/testing/test_models_sequential.py --models smollm-135m-python --skip-unload
```

### Available Models

```bash
# List all available models
python scripts/testing/test_models_sequential.py --list
```

| Model | Type | Notes |
|-------|------|-------|
| `bert-base-uncased` | Text Classification | ONNX, fast |
| `yolov8n` | Object Detection | ONNX, requires video sample |
| `whisper-tiny-python` | Audio Transcription | Python backend |
| `smollm-135m-python` | LLM (135M params) | Python backend, lightweight |
| `tinyllama-python` | LLM (1.1B params) | Python backend |
| `tinyllama-trtllm` | LLM (TensorRT) | TensorRT-LLM, requires GPU |

### Test Script Options

| Flag | Description |
|------|-------------|
| `--models MODEL [MODEL ...]` | Specific models to test (default: all) |
| `--cpu` | Use KIND_CPU instead of KIND_GPU |
| `--device N` | GPU device index (default: 0) |
| `--skip-copy` | Skip copying models (already on EDV) |
| `--skip-benchmark` | Skip benchmarks (faster) |
| `--skip-unload` | Don't unload models after testing |
| `--list` | List available models and exit |

---

## Local Testing (Docker Compose)

For local development and testing before deploying to Kubernetes.

### Start Services

```bash
# Build and start all services
docker compose up --build -d

# Check status (wait for "healthy")
docker compose ps

# View logs
docker compose logs -f backend http-proxy grpc-proxy
```

### Local Endpoints

| Service | URL |
|---------|-----|
| gRPC Proxy | `localhost:50051` |
| REST Proxy | `http://localhost:8080` |
| Swagger Docs | `http://localhost:8080/docs` |
| Triton Metrics | `http://localhost:8002/metrics` |

### Local Environment (No Auth)

```bash
export TRITON_GRPC_URL="localhost:50051"
export TRITON_REST_URL="http://localhost:8080"
```


### Stop Services

```bash
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

### Build Docker Images

```bash
export VERSION=v7

# Triton Backend
docker buildx build --platform=linux/amd64 -f ./Dockerfile.triton \
  -t quay.io/domino/domino-triton-backend:${VERSION} --push .

# Triton vLLM Backend
docker buildx build --platform=linux/amd64 -f ./Dockerfile.triton.vllm \
  -t quay.io/domino/domino-triton-backend-vllm:${VERSION} --push .

# gRPC Proxy
docker buildx build --platform=linux/amd64 -f ./Dockerfile.proxy.grpc \
  -t quay.io/domino/domino-triton-grpc-proxy:${VERSION} --push .

# REST Proxy
docker buildx build --platform=linux/amd64 -f ./Dockerfile.proxy.http \
  -t quay.io/domino/domino-triton-http-proxy:${VERSION} --push .

# Admin API
docker buildx build --platform=linux/amd64 -f ./Dockerfile.triton.admin \
  -t quay.io/domino/domino-triton-admin:${VERSION} --push .
```

---

## Remote Usage (Kubernetes/Domino)

### 0. Setup Workspace

Copy these folders from the repository to your Domino workspace or local environment:

```bash
# Required folders
scripts/                  # Client scripts, benchmarks, download scripts
samples/                  # Sample input files (audio, video)
docker/                   # Requirements files
triton-repo-reference/    # Python backend model sources (config.pbtxt, model.py)
```

The download scripts copy Python backend models from `triton-repo-reference/models/` to `triton-repo/models/`. Without this folder, Python models (whisper, smollm, tinyllama) won't be set up correctly.

```bash
# Clone or copy the required folders
git clone https://github.com/domino-field/domino-triton-inference.git
cd domino-triton-inference

# Or if copying to an existing workspace
cp -r scripts/ samples/ docker/ triton-repo-reference/ /path/to/workspace/
```

**Directory structure after setup:**
```
/mnt/                             # Your workspace root
├── scripts/                      # Client and download scripts
├── samples/                      # Test audio/video files
├── docker/                       # Requirements files
├── triton-repo-reference/        # Model sources (REQUIRED for download scripts)
│   └── models/
│       ├── smollm-135m-python/   # config.pbtxt + 1/model.py
│       ├── tinyllama-python/     # config.pbtxt + 1/model.py
│       └── whisper-tiny-python/  # config.pbtxt + 1/model.py
└── triton-repo/                  # Created by download scripts
    ├── models/                   # Triton model configs (copied from reference)
    └── weights/                  # Downloaded model weights
```

---

### 1. Environment Variables

Set these in your Domino workspace or Kubernetes pod:

```bash
# Required: API key for authentication
export DOMINO_USER_API_KEY="your-api-key-here"

# Required: Proxy endpoints (adjust namespace as needed)
export TRITON_GRPC_URL="triton-inference-server-proxy.domino-inference-dev.svc.cluster.local:50051"
export TRITON_REST_URL="http://triton-inference-server-proxy.domino-inference-dev.svc.cluster.local:8080"
```

Verify connectivity:
```bash
curl -s "$TRITON_REST_URL/v2/health/ready" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"
```

---

### 2. Download Models

Download models to your local `triton-repo/` directory:

```bash
# Install dependencies
pip install -r docker/requirements-client.txt

# Download all models
python scripts/download/download_all.py

# Or download individually
python scripts/download/download_yolov8n.py      # Object detection (ONNX)
python scripts/download/download_bert.py          # Text classification (ONNX)
python scripts/download/download_whisper.py       # Audio transcription (Python)
python scripts/download/download_llm.py           # SmolLM-135M text generation (Python)
python scripts/download/download_tinyllama.py     # TinyLlama-1.1B (Python)
```

Verify the models directory:
```bash
ls -la triton-repo/models/
# Expected: yolov8n/, bert-base-uncased/, whisper-tiny-python/, smollm-135m-python/
```

---
### 3. Copy Models to EDV

Copy models from your local repo to an EDV or shared storage:

```bash
export SOURCE_TRITON_ROOT=/mnt/triton-repo
export TARGET_TRITON_ROOT=/domino/edv/domino-inference-dev-triton-repo-pvc

# List available models
python scripts/model_management/copy_model.py --source $SOURCE_TRITON_ROOT --list

# Copy ALL models (with weights) to target EDV
python scripts/model_management/copy_model.py --source $SOURCE_TRITON_ROOT --target $TARGET_TRITON_ROOT --all

# Copy a single model
python scripts/model_management/copy_model.py --source $SOURCE_TRITON_ROOT --target $TARGET_TRITON_ROOT bert-base-uncased

# Copy single model with weights
python scripts/model_management/copy_model.py --source $SOURCE_TRITON_ROOT --target $TARGET_TRITON_ROOT --include-weights smollm-135m-python

# Overwrite all existing models
python scripts/model_management/copy_model.py --source $SOURCE_TRITON_ROOT --target $TARGET_TRITON_ROOT --overwrite --all
```

**Options:**
| Flag | Description |
|------|-------------|
| `--source, -s` | Source triton-repo root (default: `./triton-repo`) |
| `--target, -t` | Target triton-repo root (required) |
| `--all, -a` | Copy all models (includes weights automatically) |
| `--include-weights, -w` | Also copy weights directory (for single model) |
| `--overwrite, -f` | Overwrite if target exists |
| `--list, -l` | List available models |
| `--instance-kind` | Set instance_group kind: `KIND_GPU` or `KIND_CPU` |
| `--instance-count` | Set instance_group count (default: 1) |
| `--device` | GPU device index for KIND_GPU (default: 0) |

**Configure instance_group for target hardware:**
```bash
# Copy all models and configure for GPU (default: device 0)
python scripts/model_management/copy_model.py --source $SOURCE_TRITON_ROOT --target $TARGET_TRITON_ROOT --all --instance-kind KIND_GPU

# Copy model configured for specific GPU device (e.g., device 1)
python scripts/model_management/copy_model.py --source $SOURCE_TRITON_ROOT --target $TARGET_TRITON_ROOT --instance-kind KIND_GPU --device 1 bert-base-uncased

# Copy single model and configure for CPU with 2 instances
python scripts/model_management/copy_model.py --source $SOURCE_TRITON_ROOT --target $TARGET_TRITON_ROOT --instance-kind KIND_CPU --instance-count 2 bert-base-uncased
```

This generates a `config.pbtxt` with an instance_group like:
```
instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [0]
  }
]
```

---

### 4. Invoke Models

#### Load/Unload Models

```bash
# Check model status
python scripts/model_management/model_status.py

# Load models
python scripts/model_management/load_model.py smollm-135m-python
python scripts/model_management/load_model.py bert-base-uncased yolov8n whisper-tiny-python
python scripts/model_management/load_model.py --all

# Unload models
python scripts/model_management/unload_model.py smollm-135m-python
python scripts/model_management/unload_model.py --all
```

#### LLM Text Generation

```bash
# gRPC (recommended)
python scripts/clients/llm_text_grpc_client.py --prompt "What is AI?"
python scripts/clients/llm_text_grpc_client.py --prompt "Explain AI" --max-tokens 100 --temperature 0.8

# REST
python scripts/clients/llm_text_rest_client.py --prompt "What is AI?"

# Use different model
python scripts/clients/llm_text_grpc_client.py --model tinyllama-python --prompt "Write a haiku"
```

**Options:** `--prompt`, `--prompts-file`, `--model`, `--max-tokens`, `--temperature`, `--top-p`, `--system-prompt`, `--output`

#### BERT Text Classification

```bash
# gRPC
python scripts/clients/bert_text_grpc_client.py --texts "This is great!"
python scripts/clients/bert_text_grpc_client.py --texts "Great!" "Terrible" "Average"

# REST
python scripts/clients/bert_text_rest_client.py --texts "This movie was amazing!"
```

**Options:** `--texts`, `--texts-file`, `--model`, `--batch-size`, `--output`

#### Whisper Audio Transcription

```bash
# gRPC
python scripts/clients/whisper_audio_grpc_client.py --audio samples/audio_sample.wav
python scripts/clients/whisper_audio_grpc_client.py --audio file1.wav file2.wav

# REST
python scripts/clients/whisper_audio_rest_client.py --audio samples/audio_sample.wav
```

**Options:** `--audio`, `--audio-dir`, `--model`, `--language`, `--task`, `--output`

#### YOLOv8n Object Detection

```bash
# gRPC (recommended for video)
python scripts/clients/yolov8n_video_grpc_client.py --video samples/video.avi --max-frames 10

# REST
python scripts/clients/yolov8n_video_rest_client.py --video samples/video.avi --max-frames 10
```

**Options:** `--video`, `--fps`, `--max-frames`, `--batch-size`, `--output`

#### TensorRT-LLM (TinyLlama)

TensorRT-LLM models require GPU and architecture-specific engines.

```bash
python scripts/clients/tinyllama_trtllm_grpc_client.py --prompt "What is AI?"
python scripts/clients/tinyllama_trtllm_rest_client.py --prompt "What is AI?"
```

**Options:** `--prompt`, `--prompts-file`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--output`

#### Test All Clients

```bash
# Run all client tests
python scripts/clients/test_all_clients.py

# Or use pytest
python -m pytest tests/ -v
```

---

### 5. Run Benchmarks

All benchmarks compare REST and gRPC client performance. They use the `TRITON_REST_URL` and `TRITON_GRPC_URL` environment variables.

```bash
# All benchmarks
python scripts/benchmarks/benchmark_all.py

# Individual benchmarks
python scripts/benchmarks/benchmark_bert_clients.py
python scripts/benchmarks/benchmark_llm_clients.py
python scripts/benchmarks/benchmark_llm_clients.py --model tinyllama-python  # Specify model
python scripts/benchmarks/benchmark_whisper_clients.py --audio samples/audio_sample.wav
python scripts/benchmarks/benchmark_yolov8_clients.py --video samples/video.avi

# TinyLlama comparison (Python vs TensorRT-LLM)
python scripts/benchmarks/benchmark_tinyllama.py
```

**Benchmark Options:**
| Flag | Description |
|------|-------------|
| `--iterations N` | Number of iterations per client (default: varies) |
| `--model MODEL` | Model name (for LLM benchmarks) |
| `--video PATH` | Video file (for YOLOv8) |
| `--audio PATH` | Audio file (for Whisper) |

Output files are saved to `results/` directory.

---

## Reference

### Available Models

| Model | Type | Client Scripts | Notes |
|-------|------|----------------|-------|
| `smollm-135m-python` | LLM (Python) | `llm_text_*_client.py` | Default LLM |
| `tinyllama-python` | LLM (Python) | `llm_text_*_client.py --model tinyllama-python` | Larger LLM |
| `tinyllama-trtllm` | LLM (TensorRT) | `tinyllama_trtllm_*_client.py` | Requires GPU |
| `bert-base-uncased` | Classification | `bert_text_*_client.py` | ONNX |
| `whisper-tiny-python` | Transcription | `whisper_audio_*_client.py` | Python |
| `yolov8n` | Detection | `yolov8n_video_*_client.py` | ONNX |

### Model Management via curl

The proxy supports both Triton v2 API and Domino v1 extensions.

**Triton v2 API (Standard):**
```bash
# Server health
curl -s "$TRITON_REST_URL/v2/health/live" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"
curl -s "$TRITON_REST_URL/v2/health/ready" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"

# Model ready check
curl -s "$TRITON_REST_URL/v2/models/bert-base-uncased/ready" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"

# Repository index (list models)
curl -s -X POST "$TRITON_REST_URL/v2/repository/index" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool

# Load a model
curl -X POST "$TRITON_REST_URL/v2/repository/models/smollm-135m-python/load" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"

# Unload a model (with dependents)
curl -X POST "$TRITON_REST_URL/v2/repository/models/smollm-135m-python/unload" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"unload_dependents": true}}'
```

**Domino v1 Extensions:**
```bash
# List all models (Domino format)
curl -s "$TRITON_REST_URL/v1/models" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY" | python3 -m json.tool

# Get model status (with state info)
curl -s "$TRITON_REST_URL/v1/models/status" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY" | python3 -m json.tool

# Load a model
curl -X POST "$TRITON_REST_URL/v1/models/smollm-135m-python/load" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"

# Unload a model
curl -X POST "$TRITON_REST_URL/v1/models/smollm-135m-python/unload" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"

# Unload all models
curl -X POST "$TRITON_REST_URL/v1/models/unload-all" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"
```

### Pin/Cordon Models

```bash
# Pin a model (protect from LRU eviction)
curl -X POST "$TRITON_REST_URL/v1/models/smollm-135m-python/pin" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"pinned": true}'

# Unpin
curl -X POST "$TRITON_REST_URL/v1/models/smollm-135m-python/pin" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"pinned": false}'

# Cordon (reject new requests for maintenance)
curl -X POST "$TRITON_REST_URL/v1/models/smollm-135m-python/cordon" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"

# Uncordon
curl -X POST "$TRITON_REST_URL/v1/models/smollm-135m-python/uncordon" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"
```

### Troubleshooting

#### Connection Refused
```bash
# Check services are running
docker compose ps                                    # Local
kubectl get pods -n <namespace>                      # Kubernetes

# Check health
curl -s "$TRITON_REST_URL/v2/health/ready"
```

#### API Key Issues
```bash
echo $DOMINO_USER_API_KEY
curl -s "$TRITON_REST_URL/v1/models/status" -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY"
```

#### Model Not Loading
```bash
docker compose logs backend | grep -A5 "failed to load"    # Local
kubectl logs <triton-pod> | grep -A5 "failed to load"      # Kubernetes
```

#### GPU Memory Issues (CUDA OOM)
If you see `CUDA out of memory` errors when loading multiple models:

```bash
# Test models one at a time (recommended)
python scripts/testing/test_models_sequential.py

# Unload all models first
python scripts/model_management/unload_model.py --all

# Use force cleanup if models are stuck
curl -X POST "$TRITON_REST_URL/v2/repository/models/{model}/unload" \
  -H "X-Domino-Api-Key: $DOMINO_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"unload_dependents": true}}'

# Or use CPU for testing (slower but no memory issues)
python scripts/testing/test_models_sequential.py --cpu
```

#### TensorRT Engine Mismatch
```
Failed to deserialize cuda engine
```
The TensorRT engine was built on a different GPU. Rebuild on target GPU (V100=sm_70, A100=sm_80, L4=sm_89).

#### Model Load Timeout
If models take too long to load, the proxy has a configurable timeout:
- Default: 300 seconds
- Configure via `model_load_timeout_secs` in Helm values
- For large models (LLMs), increase this value

---

### Performance Tips

| Workload | Recommended | Why |
|----------|-------------|-----|
| Image/Video (YOLOv8) | gRPC | Large payloads, binary encoding ~3x faster |
| Text (BERT, LLM) | Either | Small payloads, inference-bound |
| Audio (Whisper) | Either | Medium payloads, inference-bound |
| High throughput | gRPC | Lower latency, better batching |
| Debugging | REST | Easier to inspect with curl |

**Approximate Performance (single GPU):**

| Model | Protocol | Throughput |
|-------|----------|------------|
| YOLOv8n | gRPC | ~20 FPS |
| YOLOv8n | REST | ~3 FPS |
| BERT | Either | ~22 texts/sec |
| Whisper | Either | ~1 file/sec |
| SmolLM-135M | Either | ~50-100 tokens/sec |

---

### Further Reading

- [REST vs gRPC Trade-offs](REST_vs_gRPC_tradeoffs.md) - Protocol comparison
- [Design Document](design.md) - System architecture
- [LLM Support Plan](LLM_support_plan.md) - TensorRT-LLM deployment
- [Helm Installation](helm-install.md) - Kubernetes deployment
- [Python Model Environments](python_model_environments.md) - Model dependencies
