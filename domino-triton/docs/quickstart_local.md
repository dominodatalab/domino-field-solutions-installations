# Local Development Quickstart

This guide covers running the Triton inference stack locally using Docker Compose.

## Prerequisites

- Docker and Docker Compose v2+
- Python 3.9+ (for client scripts)
- 10+ GB disk space for model weights

## Quick Start

### 1. Download Models

```bash
# Download all supported models
python scripts/download/download_all.py

# Or download individually
python scripts/download/download_yolov8n.py
python scripts/download/download_bert.py
python scripts/download/download_whisper.py
python scripts/download/download_llm.py
```

### 2. Start the Stack

```bash
docker compose up -d
```

### 3. Verify Services

```bash
# Check all services are healthy
docker compose ps

# Test health endpoint
curl http://localhost:8080/v2/health/ready
```

### 4. Set Environment Variables

```bash
export TRITON_REST_URL=http://localhost:8080
export TRITON_GRPC_URL=localhost:50051
```

### 5. Run Tests

```bash
# Test all models (sync mode)
python scripts/testing/test_models_sequential.py --skip-copy

# Test all models (async mode - higher throughput)
python scripts/testing/test_models_sequential.py --skip-copy --async
```

## Service URLs

| Service | URL |
|---------|-----|
| HTTP Proxy | http://localhost:8080 |
| gRPC Proxy | localhost:50051 |
| Dashboard | http://localhost:8888 |
| Redis | localhost:6379 |
| Triton HTTP | http://localhost:8000 |
| Triton gRPC | localhost:8001 |
| Triton Metrics | http://localhost:8002 |

## Running Tests

### Install Client Dependencies

```bash
pip install -r docker/requirements-client.txt
```

### Environment Setup

Always set these before running client scripts:

```bash
export TRITON_REST_URL=http://localhost:8080
export TRITON_GRPC_URL=localhost:50051
```

### Sequential Model Testing

The `test_models_sequential.py` script tests models one at a time:

```bash
# Sync mode (sequential processing)
python scripts/testing/test_models_sequential.py --skip-copy

# Async mode (concurrent processing, higher throughput)
python scripts/testing/test_models_sequential.py --skip-copy --async

# Test specific models only
python scripts/testing/test_models_sequential.py --skip-copy --models yolov8n bert-base-uncased

# Async with specific models
python scripts/testing/test_models_sequential.py --skip-copy --async --models yolov8n

# Skip benchmarks (just test inference works)
python scripts/testing/test_models_sequential.py --skip-copy --skip-benchmark

# List available models
python scripts/testing/test_models_sequential.py --list
```

### Test Options

| Flag | Description |
|------|-------------|
| `--skip-copy` | Skip copying models (required for local Docker) |
| `--async` | Use async gRPC mode (concurrent processing) |
| `--models MODEL [...]` | Test specific models instead of all |
| `--skip-benchmark` | Skip benchmark step (just test inference) |
| `--skip-unload` | Don't unload models after testing |
| `--cpu` | Use CPU instead of GPU |
| `--device N` | GPU device index (default: 0) |
| `--list` | List available models and exit |

### Async vs Sync Mode

| Mode | Flag | Description | Best For |
|------|------|-------------|----------|
| Sync | (default) | Sequential, one request at a time | Debugging, latency measurement |
| Async | `--async` | Concurrent batch processing | Throughput, production-like load |

## Running Individual Benchmarks

```bash
# BERT text classification
python scripts/benchmarks/benchmark_bert_clients.py --rest-url http://localhost:8080
python scripts/benchmarks/benchmark_bert_clients.py --rest-url http://localhost:8080 --async

# YOLOv8 video object detection
python scripts/benchmarks/benchmark_yolov8_clients.py --rest-url http://localhost:8080 \
  --video samples/video.avi --max-frames 120
python scripts/benchmarks/benchmark_yolov8_clients.py --rest-url http://localhost:8080 \
  --video samples/video.avi --async

# Whisper audio transcription
python scripts/benchmarks/benchmark_whisper_clients.py --rest-url http://localhost:8080 \
  --audio samples/audio_sample.wav
python scripts/benchmarks/benchmark_whisper_clients.py --rest-url http://localhost:8080 \
  --audio samples/audio_sample.wav --async

# LLM text generation
python scripts/benchmarks/benchmark_llm_clients.py --rest-url http://localhost:8080
python scripts/benchmarks/benchmark_llm_clients.py --rest-url http://localhost:8080 --async
```

## Useful Endpoints

```bash
# Health check
curl http://localhost:8080/v2/health/ready

# List models
curl http://localhost:8080/v1/models

# Model status (with LRU metrics)
curl http://localhost:8080/v1/models/status

# Swagger UI
open http://localhost:8080/docs

# Dashboard
open http://localhost:8888
```

## Stopping Services

```bash
docker compose down
```

---

## Advanced: Scaling Configuration

The `docker-compose.scale.yml` is a standalone compose file for running with resource limits and multiple instances.

> **Note:** This file doesn't expose fixed ports (except dashboard on 8888). Use basic `docker-compose.yml` for local testing with client scripts.

### Resource Limits

| Service | CPU | Memory |
|---------|-----|--------|
| dashboard | 0.5 | 200 MB |
| http-proxy | 0.5 | 200 MB |
| grpc-proxy | 0.5 | 200 MB |
| state (redis) | 0.5 | 200 MB |
| backend (triton) | 1.0 | 1 GB |

### Scaling Commands

```bash
# Single instance with resource limits
docker compose -f docker-compose.scale.yml up -d

# Scale to 2 Triton backends and 2 HTTP proxies
docker compose -f docker-compose.scale.yml up -d \
  --scale backend=2 \
  --scale http-proxy=2

# Check running instances
docker compose -f docker-compose.scale.yml ps

# Stop scaled services
docker compose -f docker-compose.scale.yml down
```

---

## Troubleshooting

### Services Not Starting

```bash
docker compose logs backend
docker compose logs http-proxy
```

### Model Not Loading

```bash
ls -la triton-repo/models/
docker compose logs backend | grep -i error
```

### Connection Refused

```bash
# Ensure services are running
docker compose ps

# Test direct Triton access (bypassing proxy)
curl http://localhost:8000/v2/health/ready
```

### Out of Memory

```bash
# Unload unused models
curl -X POST http://localhost:8080/v1/models/large-model/unload

# Use CPU mode
python scripts/testing/test_models_sequential.py --skip-copy --cpu
```
