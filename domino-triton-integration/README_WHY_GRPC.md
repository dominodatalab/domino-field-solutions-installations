# gRPC vs REST+Base64 for Sending Image Data

**Why gRPC is the correct transport for inference workloads**

When Domino workloads send images to a backend inference server, we have two broad choices:
1. REST API with JSON + base64-encoded image bytes
2. gRPC with protobuf messages carrying raw binary bytes

Both “work,” but they are not even close in performance, correctness, or efficiency.
This document explains why **gRPC is the required approach** for model inference and why REST+base64 should not be used for image payloads.

---

## 1. The core issue: how large binary data actually moves over the wire

**REST + JSON + base64**

REST cannot send raw binary inside JSON.
So image tensors must be converted to a base64 string, which introduces:

- 33% size inflation  
- CPU cost for encoding on the client
- CPU cost for decoding on the server
- Multiple memory copies during JSON parsing
- Massive latency spikes for large images

REST turns a binary payload into:

```shell
Binary → Base64 String → JSON → HTTP Body
```
At the server:

```pgsql
HTTP Body → JSON Parser → Base64 Decode → Binary
```

Each arrow means more memory copies, more CPU, and slower inference.

---

## 2. gRPC solves this by supporting binary payloads natively

Protobuf has a bytes field type designed specifically to carry raw binary data:
```protobuf
message ImageRequest {
    bytes image = 1;
}
```

Sending binary through gRPC looks like:
```shell
Binary → Protobuf Serialization → HTTP/2 Frame
```

On the server:

```shell
HTTP/2 Frame → Protobuf Deserialize → Binary
```
There is no encoding, no decoding, no JSON parsing, and no base64 overhead.

## 3. What “zero-copy” means in our context

We are not using Triton shared memory here. For this workflow, zero-copy simply means avoiding unnecessary transformations.

gRPC eliminates:

- Copying bytes into a base64 buffer

- Copying base64 text into JSON

- Copying the JSON-formatted string into a decoded byte array


So even without shared memory, gRPC effectively performs minimal copies and preserves the original binary payload through the stack.

This is exactly the “zero-copy principle”:

> Don’t move or transform large binary buffers unless you have to.

REST + base64 violates this principle heavily.

---

## 4. Performance comparison (qualitative)

| Aspect | REST + Base64 | gRPC + Bytes | 
|--------|----------------|---------------|
| Payload Size | +33% larger due to base64 | Original size |
| CPU Overhead | High (encode/decode) | Low |
| Latency | Worse p95/p99, unstable under load | Stable, lower latency |
| Memory Copies | Many (buffer → string → JSON → decode → buffer) | Few (buffer → protobuf → buffer) |
| Throughput | Lower, especially with large images | Much higher |
| Failure Modes | Decoding errors, JSON limits, string size issues | None specific to binary payloads |

---

## 5. Why REST+base64 becomes a scaling bottleneck

At small sizes (tiny images, low QPS), you won’t feel it.

But under realistic inference workloads:

- Images can be large (hundreds of KB to many MB)

- QPS can be high

- JSON parsing becomes a bottleneck

- CPU cost of base64 becomes noticeable

- Increased payload size wastes bandwidth

- Latency becomes unpredictable

Base64 forces the infrastructure to do work that provides zero value to the model.

---
## 6. Why gRPC is a perfect match for inferencing systems

### 6.1 Designed for binary, structured data

Protobuf is a compact binary format — ideal for large images.

### 6.2 Runs over HTTP/2

Multiplexing, connection reuse, lower overhead.

### 6.3 Strong typing

Less chance of malformed inputs.

### 6.4 Efficient serialization

Minimal overhead regardless of payload size.

---

## 7. Summary

### Why gRPC?

- Directly carries raw image bytes

- No base64 tax

- No JSON parsing

- Fast, stable, predictable latency

- Lower CPU and memory use

- Purpose-built for high-throughput inference workloads

### Why not REST+base64?

- 33% larger payloads

- Multiple expensive memory copies

- High CPU overhead for encode/decode

- JSON parsing slowdowns

- Harder to scale reliably