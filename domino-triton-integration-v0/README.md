# Triton Thin-Proxy Video Inference (gRPC) 

> Do not use this repo in production without consulting with Domino Data Lab PS

## Overview

This repo wires up a Triton Inference Server backend, a thin gRPC proxy that adds auth + zero-pre/post, and a client 
that does video → preprocess → stream frames → receive detections → postprocess/annotate.

It is intentionally minimal, predictable, and deployable anywhere you can run containers and open two ports.

### Architecture

```

flowchart LR
  V[Video file / camera] --> |BGR frames| C[Client]
  C --> |letterbox + NCHW FP32 bytes| P[Thin gRPC Proxy]
  P --> |InferInput(s)| T[Triton Server]
  T --> |raw tensors| P
  P --> |Ack JSON + base64 blobs| C
  C --> |parse + scale + draw| O1[annotated.mp4]
  C --> |counts per frame| O2[frame_counts.jsonl]

```

### Roles

- Triton backend (container `grpc-domino-triton-proxy:latest` from `Dockerfile.triton.proxy`)
    - Serves models from a mounted model repository at /models
    - gRPC enabled on 8001, metrics on 8002 (HTTP disabled by default)
- Thin proxy (`triton_proxy.py`)
  - Terminates gRPC from clients
  - Enforces simple auth (either X-Domino-Api-Key or Authorization: Bearer)    
  - For each request:
    - Reads control JSON from Meta.note (or Text.text) to know model name, input names, shapes, dtypes, outputs
    - Validates byte size vs shape × dtype
    - Calls Triton with those exact buffers
    - Returns Ack JSON whose outputs map contains base64-encoded raw tensors

- Client (client.py)
  - Opens a video, does YOLO-style letterbox to square size
  - Packs a float32 NCHW tensor to bytes per frame
  - Streams frames over bidi gRPC (Chat)
  - Decodes proxy outputs, converts to numpy arrays, filters by confidence
  - Scales boxes back to original frame size, draws overlays
  - Writes:
    - results/annotated.mp4
    - results/frame_counts.jsonl

### Triton Backend setup

```Dockerfile
# Dockerfile — Triton gRPC server (mount your model repo at /models)
FROM nvcr.io/nvidia/tritonserver:24.10-py3

# Where you will mount your Triton model repository at runtime
ENV MODEL_REPO=/models
ENV TRITON_GRPC_PORT=8001
ENV TRITON_HTTP_PORT=8000
ENV TRITON_METRICS_PORT=8002

# Make sure the mountpoint exists
RUN mkdir -p ${MODEL_REPO}

# Expose gRPC + metrics (HTTP is disabled below, but you can expose 8000 if you want it)
EXPOSE 8001 8002

# Start Triton with gRPC enabled and HTTP disabled
CMD tritonserver \
  --model-repository=${MODEL_REPO} \
  --allow-grpc=true --grpc-port=${TRITON_GRPC_PORT} \
  --allow-http=false --http-port=${TRITON_HTTP_PORT} \
  --allow-metrics=true --metrics-port=${TRITON_METRICS_PORT} \
  --log-verbose=0
```

**Notes**
- Mount your model repo at /models when you run the container
- gRPC on 8001, Prometheus metrics on 8002
- HTTP is disabled to simplify surface area

## Thin Proxy setup
File: `src/triton_proxy.py`

Key behavior

- Auth: checks `X-Domino-Api-Key` or Authorization: `Bearer <token>`
- Control JSON source (first found wins):
  1. packet.meta.note (preferred)
  2. If payload is Text, packet.text.text as JSON
  3. Otherwise, environment defaults
- Control JSON schema
```json
{
  "model": "yolov8n",
  "inputs": [
    {"name": "images", "shape": [1,3,640,640], "dtype": "FP32", "from": "frame"}
  ],
  "outputs": ["output0"],
  "timeout": 30.0
}
```
- `from` can be "image", "frame", or "video" which tells the proxy which field’s raw bytes to read
- **Size check**: proxy validates `len(bytes) == prod(shape) * sizeof(dtype)`
- Response: Ack.message is compact JSON whose outputs map contains per-output:
  - dtype (numpy-like string)
  - shape (list)
  - b64 (raw bytes base64)

#### Supported dtypes
`FP32`, `FP16`, `INT64`, `INT32`, `INT8`, `UINT8` mapped to numpy dtypes internally

#### gRPC server options
- Message limits set to 64 MB both directions
- Keepalive configured for long streams

#### Env vars
| Var                     | Purpose                                      | Default               |
|-------------------------|----------------------------------------------|-----------------------|
| TRITON_URL            | Triton backend address (host:port)           | localhost:8001        |
| TRITON_MODEL_NAME	| Default model if not in control JSON	| yolov8n
|TRITON_INPUT_NAME	| Default input name	| images
|TRITON_INPUT_SHAPE	| Default shape JSON	| [1,3,640,640]
|TRITON_INPUT_DTYPE	| Default dtype	| FP32
|TRITON_INPUT_FROM	| Default source field	| image
|TRITON_OUTPUT_NAMES	| Comma list of outputs	| output0
|TRITON_TIMEOUT_SECS	| Default timeout seconds	| 30
|PROXY_HOST	| Bind host	| 0.0.0.0
|PROXY_PORT	| Bind port	| 50051
|GRPC_EXPECT_API_KEY	| Expected API key header value	| unset
|GRPC_EXPECT_BEARER	| Expected bearer token	| unset  

### Client

File: `client.py`

**Pipeline**
1. Open video via OpenCV
2. Preprocess
- Letterbox to square IMG_SIZE
- BGR → RGB, normalize to [0,1]
- HWC → CHW, add batch, get (1,3,H,W) float32
3. Stream
- For each frame, build control JSON
- Pack tensor bytes and send in DataPacket.frame.data
- Use Meta.note for control JSON, Meta.correlation_id as f"{stream}:{seq}"
4. Postprocess
- Parse proxy Ack JSON
- Base64 decode outputs to numpy arrays
- YOLO-style parsing helper handles single-tensor (1,M,6) or four-head outputs
- Filter by confidence, scale boxes back to original frame using stored ratio and pads
- Draw overlays, write video and per-frame JSONL

#### Outputs

- results/annotated.mp4
- results/frame_counts.jsonl (one JSON object per frame with counts and detections)

### Env vars

| Var                     | Purpose                                      | Default               |
|-------------------------|----------------------------------------------|-----------------------|
|MM_ADDR	| Thin proxy address	| localhost:50051
|VIDEO_PATH	| Path to input video	| required
|RESULTS_DIR	| Output directory	| results
|STREAM_ID	| Logical stream name	| cam-001
|IMG_SIZE	| Letterbox square size	| 640
|CONF_THRES	| Confidence threshold	| 0.25
|FPS_CAP	| Throttle send FPS	| unset (source FPS)
|MAX_FRAMES	| Limit number of frames	| unset
|MODEL_NAME	| Model name for Triton	| yolov8n
|INPUT_NAME	| Triton input name	| images
|OUTPUT_NAMES	| Comma list of outputs	| output0
|DOMINO_API_KEY	| Adds X-Domino-Api-Key header	| unset
|GRPC_BEARER_TOKEN	| Adds Authorization: Bearer header	| unset

## Running Locally

### 0) Build and push images

```shell
docker buildx build --platform=linux/amd64 -f ./Dockerfile.triton -t quay.io/domino/grpc-domino-triton-backend:latest --push .
docker buildx build --platform=linux/amd64 -f ./Dockerfile.triton.proxy -t quay.io/domino/grpc-domino-triton-proxy:latest --push .

```

### 1) Start Triton

```shell
docker run --rm --gpus all \
  -p 8001:8001 -p 8002:8002 \
  -v $PWD/models:/models:ro \
  triton-backend:latest
```

### 2) Start Thin Proxy

```shell
export TRITON_URL=localhost:8001
python src/triton_proxy.py
```

### 3) Run Client

```shell
export MM_ADDR=localhost:50051
export VIDEO_PATH=sample_video.mp4
export GRPC_BEARER_TOKEN="GRPC_EXPECT_BEARER_VALUE"
python client.py
```

You should see `results/annotated.mp4` and `results/frame_counts.jsonl`.

-------

## Control JSON Examples

### Single-input, single-output

```json
{
  "model": "yolov8n",
  "inputs": [
    {"name": "images", "shape": [1,3,640,640], "dtype": "FP32", "from": "frame"}
  ],
  "outputs": ["output0"],
  "timeout": 30.0
}
```

### Multi-input, multi-output

```json
{"model":"yolov8n","inputs":[{"name":"images","shape":[1,3,640,640],"dtype":"FP32","from":"frame"}],
 "outputs":["num_dets","boxes","scores","classes"]}
```
If you omit control JSON, proxy falls back to environment defaults.

---

## Model Output Expectations


The client postprocessor supports two common patterns:

1. Single tensor: shape (1, M, 6) with [x1, y1, x2, y2, score, cls]

2. Four tensors:
   - num with shape (B,)
   - boxes with shape (B, M, 4)
   - scores with shape (B, M)
   - classes with shape (B, M)

Adjust `OUTPUT_NAMES` and your model’s config so the proxy fetches tensors in the order your client expects.
---

## Security

- This setup uses insecure gRPC inside a trusted network segment

- Proxy requires either header to pass:
  - X-Domino-Api-Key: <value>
  - Authorization: Bearer <token>

- To harden:
  - Put the proxy behind a TLS-terminating ingress or load balancer
  - Turn on mTLS for client→proxy and proxy→Triton if needed
  - Lock down ports with network policy or security groups (Done in the K8s deployment)


---

## Performance Tips

- Tune message size limits if you push higher resolutions or batch frames
- Consider batching frames on the client to send (N,3,H,W) for models that support it
- Keep letterbox size aligned with model’s training size to avoid accuracy loss
- If the model outputs are large, keep output set minimal
- Pin the proxy close to Triton to avoid cross-zone latency
- Watch 8002/metrics to confirm steady inference latency and no queue buildup


---

## Monitoring

- Triton exposes Prometheus metrics on :8002/metrics

- Useful gauges:
  - nv_inference_request_success
  - nv_inference_queue_duration_us
  - nv_inference_execution_duration_us
  - nv_gpu_utilization, nv_gpu_memory_used_bytes (with DCGM)

Export these into your stack and alert on tail latencies and GPU health.

---

## Troubleshooting

- Proxy says size mismatch
  - Check shape and dtype in control JSON vs the tensor you packed
  - Ensure np.ascontiguousarray before tobytes()

- No detections drawn
  - Lower `CONF_THRES`
  - Confirm your output names and shapes match the parser
  - Print decoded array shapes from the client

- Timeouts
  - Bump timeout in control JSON
  - Check Triton logs for model load or queue stalls

- gRPC auth errors
  - Ensure client and proxy bearer or api key are aligned
  - Remove one of the expected headers in proxy env if you only want the other

- High CPU on client
  - Resize frames once per frame and avoid extra copies
  - Cap `FPS_CAP` to reduce encode and draw overhead

--

## Extending

- Add light pre/post to the proxy if you want to centralize transforms
- Support different input sources by switching from to image or video
- Add model routing by letting control JSON pick models per frame or per stream
- Add batch collation in proxy if upstream sends per-frame but model wants batches


## Runbook using docker-compose


### 1) deps

```shell

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pi
python3 -m pip install "opencv-python-headless==4.10.0.84" "protobuf==5.27.2" "grpcio==1.67.1" "grpcio-tools==1.67.1" "numpy==2.2.6" "tritonclient==2.61.0"
# verify install
python3 -c "import grpc_tools, google.protobuf, grpc; print('OK')"
```

### 2) generate code
```shell
cd src
python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. multimodal.proto
cd ..
```

### 3) Docker-build and push
```shell
docker buildx build --platform=linux/amd64 -f ./Dockerfile.triton -t quay.io/domino/grpc-domino-triton-backend:v1 --push .
docker buildx build --platform=linux/amd64 -f ./Dockerfile.triton.proxy -t quay.io/domino/grpc-domino-triton-proxy:v1 --push .
```


### 4) Docker-compose up
```shell
docker compose up --build
```


### 5) Run client
```shell

export DOMINO_USER_API_KEY=<your_api_key_here>
#or
export Authorization=<your_oauth_token_here>

export MM_ADDR=localhost:50051
#export MM_ADDR=grpc-domino-triton-proxy.domino-inference-dev.svc.cluster.local:50051  # if using from inside the cluster
export VIDEO_PATH=./samples/video.avi
export MODEL_NAME=yolov8n
export MODEL_VERSION=1
export INPUT_NAME=images
export OUTPUT_NAMES=output0
export IMG_SIZE=640

export PARSE_NUMPY=1
python3 src/mm_client.py
```


