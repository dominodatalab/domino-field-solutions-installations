# Triton Bytes Client Library (`mm_client_api.py`)

## Purpose
This lightweight client library provides a clean, typed Python interface for sending raw tensor bytes to the **Triton Thin Proxy** over gRPC.  
It encapsulates all the boilerplate involved in:

- Creating a gRPC channel with Domino authentication headers.
- Constructing `Control`, `Tensor`, and `DataPacket` messages for each frame.
- Maintaining correlation IDs for reliable per-frame tracking.
- Streaming packets to the proxy using the `Chat` method.
- Optionally decoding Triton outputs into NumPy arrays.

The goal is to let you focus on **supplying model name + raw tensor bytes**, without worrying about the protobuf machinery or hand-building per-frame requests.

---

## Context
The Triton Thin Proxy expects **each request packet** to contain:

- A full `Control` block (model name, version, input spec, outputs).
- A `Tensor` payload containing raw bytes matching the declared shape + dtype.
- A correlation ID (CID) so the proxy can tag its responses.

The proxy treats every packet independently — there’s no batching, session state, or accumulated context.  
This client library mirrors that behavior: each frame is wrapped in a self-contained `DataPacket` and sent via a minimal per-frame Chat call.

The library is ideal when you have:

- A video stream broken into frames  
- A sequence of tensors you want to send one-by-one  
- Or any scenario where each inference is an atomic operation  

---

## Authentication
The client automatically attaches Domino-style headers to every gRPC call if the following environment variables exist:

- `DOMINO_USER_API_KEY` → sent as `x-domino-api-key`
- `Authorization` → sent as `authorization: Bearer <value>`

This matches the expectations of the server-side `DominoAuthInterceptor`.

---

## Key Class: `TritonBytesStreamer`

### Constructor Parameters

| Parameter | Meaning |
|----------|---------|
| `addr` | `<host>:<port>` of the thin proxy |
| `input_name` | Triton input tensor name (`"images"` by default) |
| `shape` | Fixed tensor shape for all frames |
| `dtype_enum` | Protobuf dtype enum (e.g. `pb.DT_FP32`) |
| `outputs` | Triton output tensor names |
| `stream_id` | Used to build correlation IDs (`<stream_id>:<seq>`) |
| `request_timeout_sec` | gRPC Chat deadline |
| `parse_outputs_numpy` | If true, converts outputs to np.ndarray |

---

## Example Usage

### 1. Create the streamer

```python
from mm_client_api import TritonBytesStreamer
import multimodal_pb2 as pb

streamer = TritonBytesStreamer(
    addr="localhost:50051",
    input_name="images",
    shape=(1, 3, 640, 640),
    dtype_enum=pb.DT_FP32,
    outputs=("output0",),
    parse_outputs_numpy=True,
)
```

### 2. Prepare your frame iterator
```python
def frame_bytes_iter():
    for frame in video_frames:        # frame is a NumPy array
        yield frame.tobytes(order="C")
```

### 3. Send frames for inference
```python
for seq, ack, outputs in streamer.stream_bytes(
    model="yolov8n",
    model_version="1",
    frame_bytes_iter=frame_bytes_iter(),
):
    print("Frame:", seq)
    print("Model:", ack.get("model"))
    if outputs:
        for name, arr in outputs.items():
            print(name, arr.shape, arr.dtype)
```

Each iteration returns:

```bash
(seq, ack_message_dict, decoded_outputs_or_None)
```

Where:

- seq: the frame index
- ack_message_dict: parsed JSON message from the proxy
- decoded_outputs_or_None: output tensors as numpy arrays (if enabled)

## Output Decoding

If you set parse_outputs_numpy=True, the client decodes the proxy’s:

```json
{
  "dtype": "FP32",
  "shape": [1, 84, 8400],
  "b64": "<base64-encoded bytes>"
}
```

into a real NumPy array using the `_decode_outputs_to_numpy()` helper.

## Design Philosophy

- Explicit over implicit: every frame carries a complete Control block.
- Stateless: client mirrors the server's stateless per-request behavior.
- Minimal API surface: one high-level method (stream_bytes) to send data.
- Transparent errors: proxy errors are returned via Ack and not swallowed.
- No batching magic: if you want batching, build it on top of this API.

## Summary

The `mm_client_api.py` library gives you:

- Clean per-frame Triton inference from Python
- Automatic Domino auth integration
- Fully typed protobuf request construction
- Optional NumPy output conversion
- Simple, predictable control-flow per frame

- It’s the lowest-friction way to push raw tensors through your Triton Thin Proxy in a production-grade environment.