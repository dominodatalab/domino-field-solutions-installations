"""
mm_client_api.py

Thin, strongly-typed client helper for talking to the Triton thin proxy over gRPC.

This module hides the boilerplate required to:
  - Build a gRPC channel with Domino auth headers (DOMINO_USER_API_KEY / Authorization).
  - Construct `Control`, `Tensor`, and `DataPacket` messages for each frame.
  - Call the MultimodalService.Chat bidi-streaming RPC in a simple way.
  - Optionally decode the proxy's base64-encoded outputs into NumPy arrays.

Key ideas
---------
- The proxy is *stateless* with respect to the stream: each `DataPacket` contains a full
  `Control` block plus a single `Tensor` payload, and the proxy treats each packet
  independently.
- On the client side, you configure tensor semantics once (shape, dtype, input_name,
  output names) when constructing `TritonBytesStreamer`.
- For each frame (raw tensor bytes), the client builds a full `Control + Tensor` packet,
  sends it, and yields back the proxy's Ack + outputs.

Intended usage
--------------
1) Construct the streamer:

   streamer = TritonBytesStreamer(
       addr="proxy-host:50051",
       input_name="images",
       shape=(1, 3, 640, 640),
       dtype_enum=pb.DT_FP32,
       outputs=("output0",),
       stream_id="cam-001",
       request_timeout_sec=600.0,
       parse_outputs_numpy=True,  # get np.ndarray outputs
   )

2) Prepare an iterator of raw tensor bytes:

   def frame_bytes_iter():
       for frame in some_frame_source():
           yield frame.tobytes(order="C")

3) Call stream_bytes():

   for seq, ack_msg, outputs in streamer.stream_bytes(
       model="yolov8n",
       model_version="1",
       frame_bytes_iter=frame_bytes_iter(),
   ):
       # `seq` is the 0-based frame index
       # `ack_msg` is the JSON message dict returned by the proxy
       # `outputs` is either a dict[str, np.ndarray] or None (if parsing disabled)
       ...

Environment variables
---------------------
DOMINO_USER_API_KEY
    If set, a `x-domino-api-key` header is attached on every gRPC call.
Authorization
    If set, a `authorization: Bearer <value>` header is attached on every gRPC call.

These headers are expected by the server-side `DominoAuthInterceptor` to validate
the caller via Domino's `/v4/auth/principal` endpoint.
"""

import os
import json
import base64
import time
import grpc
from collections import namedtuple
from typing import Iterable, Iterator, Tuple, Dict, Any, Optional, List

import multimodal_pb2 as pb
import multimodal_pb2_grpc as pbg

# ---------------- gRPC auth + channel ----------------

# ClientCallDetails clone that we can modify inside an interceptor.
_ClientCallDetails = namedtuple(
    "_ClientCallDetails",
    ("method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"),
)


def _auth_headers() -> List[Tuple[str, str]]:
    """
    Build gRPC metadata headers for Domino-style authentication.

    Reads from environment:
      - DOMINO_USER_API_KEY  → "x-domino-api-key" header (API key auth)
      - Authorization        → "authorization: Bearer <value>" header (JWT/bearer auth)

    Returns
    -------
    list[tuple[str, str]]
        List of (key, value) metadata pairs to attach to every gRPC call.
        Returns an empty list if no auth-related env vars are set.
    """
    api_key = os.environ.get("DOMINO_USER_API_KEY")
    bearer = os.environ.get("Authorization")
    md: List[Tuple[str, str]] = []
    if api_key:
        md.append(("x-domino-api-key", api_key))
    if bearer:
        md.append(("authorization", f"Bearer {bearer}"))
    return md


class _MetaClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """
    Client-side interceptor that injects static metadata headers into every gRPC call.

    This mirrors the server-side DominoAuthInterceptor expectations:
    - Ensures that auth headers are present on unary and streaming calls.
    """

    def __init__(self, headers: List[Tuple[str, str]]):
        """
        Parameters
        ----------
        headers : list[tuple[str, str]]
            Metadata key/value pairs to attach to all outgoing RPCs.
        """
        self._headers = headers

    def _augment(self, continuation, d, req_or_it):
        """
        Create a new ClientCallDetails with merged metadata and forward the call.
        """
        base_md = list(d.metadata or [])
        base_md.extend(self._headers)
        new_d = _ClientCallDetails(
            method=d.method,
            timeout=d.timeout,
            metadata=base_md,
            credentials=d.credentials,
            wait_for_ready=getattr(d, "wait_for_ready", None),
            compression=getattr(d, "compression", None),
        )
        return continuation(new_d, req_or_it)

    def intercept_unary_unary(self, continuation, call_details, request):
        return self._augment(continuation, call_details, request)

    def intercept_unary_stream(self, continuation, call_details, request):
        return self._augment(continuation, call_details, request)

    def intercept_stream_unary(self, continuation, call_details, request_iterator):
        return self._augment(continuation, call_details, request_iterator)

    def intercept_stream_stream(self, continuation, call_details, request_iterator):
        return self._augment(continuation, call_details, request_iterator)


def build_insecure_channel(target: str) -> grpc.Channel:
    """
    Build an insecure gRPC channel with sensible defaults + optional auth metadata.

    This is the primary entry point for constructing a client channel to the thin proxy.

    Parameters
    ----------
    target : str
        `<host>:<port>` address of the Triton thin proxy.

    Behavior
    --------
    - Configures max send/receive message sizes (64 MiB).
    - Configures keepalive settings suitable for long-lived streams.
    - If DOMINO_USER_API_KEY / Authorization are present in the environment,
      wraps the base channel with a `_MetaClientInterceptor` that injects those
      headers into every call.

    Returns
    -------
    grpc.Channel
        Either the raw insecure channel, or an intercepted channel with auth metadata.
    """
    base = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
        ],
    )
    headers = _auth_headers()
    return grpc.intercept_channel(base, _MetaClientInterceptor(headers)) if headers else base


# ---------------- helpers ----------------

def _meta(cid: str, note: str = "") -> pb.Meta:
    """
    Construct a Meta message with a correlation id and optional note.

    Parameters
    ----------
    cid : str
        Correlation ID used to tie Acks back to requests.  Typically
        `<stream_id>:<seq_index>` for per-frame tracking.
    note : str, optional
        Arbitrary free-form text (not used by the thin proxy, but may be useful
        for logging or debugging).

    Returns
    -------
    pb.Meta
    """
    return pb.Meta(correlation_id=cid, note=note)


def _make_control(
    model: str,
    model_version: str,
    input_name: str,
    shape_dims: Iterable[int],
    dtype_enum: int,
    outputs: Iterable[str],
    timeout_secs: float = 30.0,
) -> pb.Control:
    """
    Construct a `Control` message describing a single tensor input and output set.

    This is the contract with the thin proxy:
      - A single `InputSpec` is defined (the one tensor payload you send).
      - `from` is set to `SRC_TENSOR` to indicate that the data comes from
        the DataPacket's `tensor` field.
      - The proxy forwards these specs directly to Triton.

    Parameters
    ----------
    model : str
        Triton model name.
    model_version : str
        Triton model version string (empty string is allowed; the proxy will pass it
        through as-is).
    input_name : str
        Triton input tensor name (e.g. "images").
    shape_dims : Iterable[int]
        Shape of the input tensor, e.g. (1, 3, 640, 640).
    dtype_enum : int
        Enum value from `pb.DataType`, e.g. `pb.DT_FP32`, `pb.DT_INT8`.
    outputs : Iterable[str]
        Names of the Triton output tensors to request, e.g. ["output0"].
    timeout_secs : float, default 30.0
        Per-request timeout (seconds) exposed to the proxy as `Control.timeout_secs`.

    Returns
    -------
    pb.Control
    """
    spec = pb.InputSpec(
        name=input_name,
        shape=pb.Shape(dims=list(shape_dims)),
        dtype=dtype_enum,
    )
    # "from" is a reserved word in Python, so we set it via setattr.
    setattr(spec, "from", pb.SRC_TENSOR)

    return pb.Control(
        model=model,
        model_version=(model_version or ""),
        timeout_secs=float(timeout_secs),
        inputs=[spec],
        outputs=[pb.OutputSpec(name=o) for o in outputs],
    )


def _decode_outputs_to_numpy(outs_desc: Dict[str, Any]):
    """
    Convert a `outputs` dict from the thin proxy back into NumPy arrays.

    The proxy responds with a structure like:

        {
          "output0": {
            "dtype": "FP32",
            "shape": [1, 84, 8400],
            "b64": "<base64-encoded raw bytes>"
          },
          "output1": {
            ...
          }
        }

    This helper:
      - Infers a NumPy dtype from the string dtype.
      - Reshapes the raw bytes into the given shape (if provided).
      - Returns a mapping of output name → np.ndarray.

    Parameters
    ----------
    outs_desc : dict[str, Any]
        The "outputs" sub-dict from the proxy's JSON message.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dict of output name to reconstructed NumPy array.
        Entries without a valid "b64" field are silently skipped.
    """
    import numpy as np

    out_arrays = {}
    for name, od in outs_desc.items():
        if not isinstance(od, dict) or "b64" not in od:
            continue

        # Map a variety of possible dtype strings to NumPy dtypes.
        np_dtype = {
            "FP32": "float32",
            "FLOAT32": "float32",
            "FLOAT": "float32",
            "FP16": "float16",
            "FLOAT16": "float16",
            "INT64": "int64",
            "INT32": "int32",
            "INT8": "int8",
            "UINT8": "uint8",
            "BYTE": "uint8",
        }.get(str(od.get("dtype", "FP32")).upper(), "float32")

        shape = tuple(int(x) for x in od.get("shape", []))
        raw = base64.b64decode(od["b64"])

        if shape:
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        else:
            # Fall back to flat interpretation if no shape is provided.
            arr = np.frombuffer(raw, dtype=np_dtype)

        out_arrays[name] = arr

    return out_arrays


# ---------------- public API ----------------

class TritonBytesStreamer:
    """
    High-level per-frame "byte streamer" for the Triton thin proxy.

    This class encapsulates:
      - gRPC channel + MultimodalServiceStub creation.
      - Building Control/Tensor/DataPacket messages consistent with your tensor shape/dtype.
      - A simple `stream_bytes()` API that takes an iterator of raw tensor bytes and yields
        back:
          (sequence_index, ack_message_dict, decoded_outputs_dict_or_None)

    Semantics
    ---------
    - The proxy treats each `DataPacket` independently (no batching, no cross-frame state).
    - Each frame's bytes are sent in a separate short-lived Chat invocation internally
      (this implementation opens a tiny stream per frame).
    - Control is attached to every packet; there is no concept of "set it once per stream".

    Parameters
    ----------
    addr : str
        `<host>:<port>` for the thin proxy.
    input_name : str, default "images"
        Triton input tensor name.
    shape : tuple[int, ...], default (1, 3, 640, 640)
        Fixed shape for all input tensors (must match the model's expectations).
    dtype_enum : int, default pb.DT_FP32
        Data type enum from `pb.DataType`.
    outputs : tuple[str, ...], default ("output0",)
        Names of the Triton outputs to request.
    stream_id : str, default "cam-001"
        Identifier used as the prefix for correlation IDs (CID = `<stream_id>:<seq>`).
    request_timeout_sec : float, default 600.0
        gRPC deadline for each Chat call.
    parse_outputs_numpy : bool, default False
        If True, `stream_bytes()` will return decoded NumPy arrays (via
        `_decode_outputs_to_numpy`). If False, it returns the raw "outputs" JSON map
        (dtype/shape/b64) from the proxy.

    Example
    -------
    >>> streamer = TritonBytesStreamer(
    ...     addr="localhost:50051",
    ...     input_name="images",
    ...     shape=(1, 3, 640, 640),
    ...     dtype_enum=pb.DT_FP32,
    ...     outputs=("output0",),
    ...     parse_outputs_numpy=True,
    ... )
    >>> def frame_bytes_iter():
    ...     for frame in frames:  # frame is a numpy array with the right shape/dtype
    ...         yield frame.tobytes(order="C")
    ...
    >>> for seq, ack, outputs in streamer.stream_bytes(
    ...     model="yolov8n",
    ...     model_version="1",
    ...     frame_bytes_iter=frame_bytes_iter(),
    ... ):
    ...     print(seq, ack.get("model"), {k: v.shape for k, v in (outputs or {}).items()})
    """

    def __init__(
        self,
        addr: str,
        input_name: str = "images",
        shape: Tuple[int, ...] = (1, 3, 640, 640),
        dtype_enum: int = pb.DT_FP32,
        outputs: Tuple[str, ...] = ("output0",),
        stream_id: str = "cam-001",
        request_timeout_sec: float = 600.0,
        parse_outputs_numpy: bool = False,
    ):
        self.addr = addr
        self.input_name = input_name
        self.shape = tuple(shape)
        self.dtype_enum = dtype_enum
        self.outputs = tuple(outputs)
        self.stream_id = stream_id
        self.request_timeout_sec = float(request_timeout_sec)
        self.parse_outputs_numpy = parse_outputs_numpy

        # Underlying gRPC channel and stub, built once per TritonBytesStreamer.
        self._channel = build_insecure_channel(addr)
        self._stub = pbg.MultimodalServiceStub(self._channel)

    def _pkt_iter(
        self,
        model: str,
        model_version: str,
        frame_bytes_iter: Iterator[bytes],
    ) -> Iterator[pb.DataPacket]:
        """
        Internal generator: turn an iterator of raw bytes into DataPacket messages.

        For each frame:
          - Construct a fresh `Control` describing the model + input spec + outputs.
          - Wrap the frame bytes in a `Tensor` with the configured dtype/shape.
          - Use `<stream_id>:<seq>` as the correlation_id.
        """
        seq = 0
        for b in frame_bytes_iter:
            ctrl = _make_control(
                model=model,
                model_version=(model_version or ""),
                input_name=self.input_name,
                shape_dims=self.shape,
                dtype_enum=self.dtype_enum,
                outputs=self.outputs,
                timeout_secs=30.0,
            )
            tensor = pb.Tensor(
                dtype=self.dtype_enum,
                shape=pb.Shape(dims=list(self.shape)),
                data=b,
            )
            cid = f"{self.stream_id}:{seq}"
            yield pb.DataPacket(
                meta=_meta(cid),
                control=ctrl,
                tensor=tensor,
                event=pb.SEV_DATA,
            )
            seq += 1

    def stream_bytes(
        self,
        model: str,
        model_version: str,
        frame_bytes_iter: Iterator[bytes],
    ) -> Iterator[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        Send an iterator of raw tensor bytes and yield proxy responses per frame.

        Parameters
        ----------
        model : str
            Triton model name to route to.
        model_version : str
            Triton model version; may be empty string if you want the server-side
            defaults (depending on how the proxy is configured).
        frame_bytes_iter : Iterator[bytes]
            Iterator yielding one `bytes` object per frame.
            Each `bytes` must represent a full tensor with the configured shape/dtype.

        Yields
        ------
        tuple[int, dict, dict or None]
            (seq, ack_json, decoded_outputs_or_None)

            seq : int
                0-based index of the frame/packet (in send order).
            ack_json : dict[str, Any]
                Parsed JSON from `Ack.message`, e.g.:

                {
                  "backend": "triton",
                  "cid": "cam-001:0",
                  "model": "yolov8n",
                  "model_version": "1",
                  "outputs": {
                    "output0": {
                      "dtype": "FP32",
                      "shape": [1, 84, 8400],
                      "b64": "..."
                    }
                  }
                }

            decoded_outputs_or_None : dict[str, np.ndarray] or dict[str, Any] or None
                - If `parse_outputs_numpy=True` and outputs are present, this is a mapping
                  of output name → np.ndarray (decoded via `_decode_outputs_to_numpy`).
                - If `parse_outputs_numpy=False`, this field is the raw "outputs" dict
                  from `ack_json` (dtype/shape/b64).
                - If the Ack contains no "outputs", this is None.

        Notes
        -----
        - Implementation detail: this sends one packet per short-lived Chat call
          (not a long-running multi-packet streaming session).
        - The thin proxy treats each packet independently, so this is functionally
          equivalent to a unary "Infer" RPC while still reusing the Chat interface.
        """
        seq_by_cid: Dict[str, int] = {}
        for i, pkt in enumerate(self._pkt_iter(model, model_version, frame_bytes_iter)):
            cid = pkt.meta.correlation_id if pkt.meta else f"{self.stream_id}:{i}"
            seq_by_cid[cid] = i
            # Each packet is sent in its own tiny stream.
            # `_drain_stream` will call Chat(...) and yield the Ack(s) for this packet.
            yield from self._drain_stream(seq_by_cid, pkt)

        # No SEV_END sent: the thin proxy doesn't maintain session state and
        # does not require explicit start/end semantics for this use case.

    def _drain_stream(
        self,
        seq_by_cid: Dict[str, int],
        first_pkt: pb.DataPacket,
    ) -> Iterator[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        Internal helper that:
          - Opens a Chat stream with a single DataPacket.
          - Consumes all Acks on that stream.
          - Maps correlation_id back to the original frame index.
          - Optionally decodes outputs to NumPy.
        """

        def gen():
            # Single-packet request stream
            yield first_pkt

        for ack in self._stub.Chat(gen(), timeout=self.request_timeout_sec):
            cid = (ack.correlation_id or "")
            seq = seq_by_cid.get(cid, -1)

            try:
                msg = json.loads(ack.message or "{}")
            except Exception:
                msg = {}

            outs_desc = msg.get("outputs")
            decoded = None

            if outs_desc:
                if self.parse_outputs_numpy:
                    decoded = _decode_outputs_to_numpy(outs_desc)
                else:
                    decoded = outs_desc

            yield (seq, msg, decoded)
