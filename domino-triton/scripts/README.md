# Scripts

Client scripts, benchmarks, and utilities for Triton Inference Server.

## Prerequisites

```bash
pip install -r docker/requirements-client.txt
```

## Directory Structure

```
scripts/
├── clients/           # Inference client scripts
├── benchmarks/        # Performance benchmarks
├── download/          # Model download scripts
├── model_management/  # Model lifecycle scripts
└── testing/           # End-to-end testing scripts
```

---

## Quick Reference

### Model Management

```bash
# Check status of all models
python scripts/model_management/model_status.py

# Load models
python scripts/model_management/load_model.py smollm-135m-python
python scripts/model_management/load_model.py --all

# Unload models
python scripts/model_management/unload_model.py smollm-135m-python
python scripts/model_management/unload_model.py --all

# Copy model to another location (e.g., EDV)
python scripts/model_management/copy_model.py --list
python scripts/model_management/copy_model.py --target /mnt/edv/triton-repo bert-base-uncased
python scripts/model_management/copy_model.py --target /mnt/edv/triton-repo --include-weights smollm-135m-python
python scripts/model_management/copy_model.py --target /mnt/edv/triton-repo --all
python scripts/model_management/copy_model.py --target /mnt/edv/triton-repo --all --instance-kind KIND_GPU
```

### Download Models

```bash
# Download all models
python scripts/download/download_all.py

# Download individual models
python scripts/download/download_yolov8n.py      # Object detection (ONNX)
python scripts/download/download_bert.py          # Text classification (ONNX)
python scripts/download/download_whisper.py       # Audio transcription (Python)
python scripts/download/download_llm.py           # SmolLM-135M text generation (Python)
python scripts/download/download_tinyllama.py     # TinyLlama-1.1B (Python)
```

### Inference Clients

```bash
# LLM Text Generation
python scripts/clients/llm_text_grpc_client.py --prompt "What is AI?"
python scripts/clients/llm_text_rest_client.py --prompt "What is AI?"

# BERT Text Classification
python scripts/clients/bert_text_grpc_client.py --texts "This is great!"
python scripts/clients/bert_text_rest_client.py --texts "This is great!"

# Whisper Audio Transcription
python scripts/clients/whisper_audio_grpc_client.py --audio samples/audio_sample.wav
python scripts/clients/whisper_audio_rest_client.py --audio samples/audio_sample.wav

# YOLOv8n Object Detection
python scripts/clients/yolov8n_video_grpc_client.py --video samples/video.avi --max-frames 5
python scripts/clients/yolov8n_video_rest_client.py --video samples/video.avi --max-frames 5

# TinyLlama TensorRT-LLM (if available)
python scripts/clients/tinyllama_trtllm_grpc_client.py --prompt "What is AI?"
python scripts/clients/tinyllama_trtllm_rest_client.py --prompt "What is AI?"

# Test all clients
python scripts/clients/test_all_clients.py
```

### Benchmarks

```bash
# Run all benchmarks
python scripts/benchmarks/benchmark_all.py

# Individual benchmarks
python scripts/benchmarks/benchmark_bert_clients.py
python scripts/benchmarks/benchmark_llm_clients.py
python scripts/benchmarks/benchmark_whisper_clients.py
python scripts/benchmarks/benchmark_yolov8_clients.py --video samples/video.avi
python scripts/benchmarks/benchmark_tinyllama.py
```

### End-to-End Testing

```bash
# Test all models sequentially (avoids GPU memory issues)
python scripts/testing/test_models_sequential.py

# Test specific models
python scripts/testing/test_models_sequential.py --models bert-base-uncased yolov8n

# Skip benchmarks (faster)
python scripts/testing/test_models_sequential.py --skip-benchmark

# Skip copy step (models already on EDV)
python scripts/testing/test_models_sequential.py --skip-copy

# List available models
python scripts/testing/test_models_sequential.py --list
```

---

## Environment Variables

Set these before running client scripts:

```bash
# For Kubernetes/Domino
export DOMINO_USER_API_KEY="your-api-key"
export TRITON_GRPC_URL="triton-proxy.namespace.svc.cluster.local:50051"
export TRITON_REST_URL="http://triton-proxy.namespace.svc.cluster.local:8080"

# For local Docker Compose (defaults)
export TRITON_GRPC_URL="localhost:50051"
export TRITON_REST_URL="http://localhost:8080"
```

---

---

## Workspace Setup

For remote environments (Kubernetes/Domino), copy these folders:

```bash
cp -r scripts/ samples/ docker/ triton-repo-reference/ /path/to/workspace/
```

The `triton-repo-reference/` folder contains Python model sources (config.pbtxt, model.py) that are copied to `triton-repo/models/` during download.

---

## Detailed Documentation

For detailed usage, options, and examples, see the [Runbook](../docs/runbook.md).

| Section | Description |
|---------|-------------|
| [Model Management](../docs/runbook.md#model-management) | Load, unload, copy, pin, cordon models |
| [Download Models](../docs/runbook.md#download-models) | Model download details and output structure |
| [BERT](../docs/runbook.md#bert-text-classification) | BERT client options and examples |
| [LLM](../docs/runbook.md#llm-text-generation) | LLM client options and parameters |
| [Whisper](../docs/runbook.md#whisper-audio-transcription) | Whisper client options |
| [YOLOv8](../docs/runbook.md#yolov8n-object-detection) | YOLOv8 video processing options |
| [Benchmarks](../docs/runbook.md#run-benchmarks) | Benchmark configuration |
