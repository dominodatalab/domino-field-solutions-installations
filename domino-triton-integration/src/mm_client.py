"""
mm_client.py

End-to-end video → Triton → annotated video pipeline.

Flow
----
Video file
  → (client) preprocess each frame into a FP32 NCHW tensor
  → (client) send tensor bytes to Triton Thin Proxy via TritonBytesStreamer
  → (proxy) Triton inference
  → (client) YOLO-style postprocessing on outputs
  → (client) draw boxes + write:
        - per-frame JSONL summary: results/frame_counts.jsonl
        - annotated video:        results/annotated.mp4

This script is meant as a reference / test client for your Triton thin proxy and
YOLO-like detection models.

Environment variables
---------------------
MM_ADDR
    `<host>:<port>` of the thin proxy. Default: `"localhost:50051"`.

VIDEO_PATH
    Path to a local video file to process. **Required**.

RESULTS_DIR
    Directory to write results (JSONL + MP4). Default: `"results"`.

STREAM_ID
    Logical stream identifier used on the client side. Default: `"cam-001"`.

IMG_SIZE
    Preprocessing target size (letterbox square). Default: `"640"`.

CONF_THRES
    Confidence threshold for detections. Default: `"0.25"`.

MAX_FRAMES
    Optional cap on the number of frames to process. If unset or non-numeric,
    all frames are processed.

MODEL_NAME
    Triton model name. Default: `"yolov8n"`.

MODEL_VERSION
    Triton model version. Default: `"1"`.

INPUT_NAME
    Triton input tensor name. Default: `"images"`.

OUTPUT_NAMES
    Comma-separated list of Triton output tensor names.
    Default: `"output0"`.

INPUT_SHAPE
    Comma-separated input shape. Default: `"1,3,640,640"`.

INPUT_DTYPE
    Input dtype enum name from `multimodal_pb2.DataType`, e.g `"DT_FP32"`.
    Default: `"DT_FP32"`.

PARSE_NUMPY
    When `"1"`, the client expects the proxy library to decode outputs into
    NumPy arrays. When `"0"`, the client decodes the base64 payload itself.
    Default: `"0"`.

REQ_TIMEOUT_SEC
    Request timeout in seconds for the streaming RPC. Default: `"600"`.
"""

import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

import multimodal_pb2 as pb
from mm_client_api import TritonBytesStreamer

# =============== Pre/Post helpers (same as original client.py) ===============

def letterbox_with_meta(im, new_shape=640):
    """
    Resize and pad an image to a square `new_shape x new_shape` while preserving aspect ratio.
    """
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))

    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left

    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return im_padded, r, (left, top)


def preprocess_bchw(frame_bgr, size=640):
    """
    Full frame → NCHW float tensor preprocessing.
    """
    img, ratio, pads = letterbox_with_meta(frame_bgr, size)
    img = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))                 # HWC->CHW
    nchw = img[np.newaxis, ...].astype(np.float32)     # (1,3,H,W)
    return nchw, ratio, pads


def pack_fp32_bytes(nchw: np.ndarray) -> bytes:
    """
    Ensure an NCHW tensor is float32 & contiguous, then return raw bytes.
    """
    if nchw.dtype != np.float32 or not nchw.flags["C_CONTIGUOUS"]:
        nchw = np.ascontiguousarray(nchw.astype(np.float32))
    return nchw.tobytes(order="C")


def parse_yolo_outputs(outs, conf_thres=0.25):
    """
    Parse YOLO-style outputs into (boxes, scores, classes).
    """
    # Single-tensor style (e.g., [B, N, 6])
    if len(outs) == 1:
        det = outs[0]
        if det.ndim == 3:
            det = det[0]
        if det.size == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)
        scores = det[:, 4]
        keep = scores >= conf_thres
        det = det[keep]
        if det.shape[0] == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)
        return det[:, 0:4].astype(np.float32), det[:, 4].astype(np.float32), det[:, 5].astype(np.int32)

    # Multi-tensor style: [num, boxes, scores, classes]
    elif len(outs) == 4:
        num = int(np.array(outs[0]).reshape(-1)[0])
        boxes = outs[1][0][:num].astype(np.float32)
        scores = outs[2][0][:num].astype(np.float32)
        classes = outs[3][0][:num].astype(np.int32)
        keep = scores >= conf_thres
        return boxes[keep], scores[keep], classes[keep]

    # Fallback: treat first as YOLO head
    else:
        det = outs[0]
        if det.ndim == 3:
            det = det[0]
        if det.size == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)
        scores = det[:, 4]
        keep = scores >= conf_thres
        det = det[keep]
        if det.shape[0] == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)
        return det[:, 0:4].astype(np.float32), det[:, 4].astype(np.float32), det[:, 5].astype(np.int32)


def scale_boxes_back(boxes_xyxy, ratio, pads):
    """
    Map predicted boxes from letterboxed space back to original frame coordinates.
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    padw, padh = pads
    b = boxes_xyxy.copy()
    b[:, 0] = (b[:, 0] - padw) / ratio
    b[:, 1] = (b[:, 1] - padh) / ratio
    b[:, 2] = (b[:, 2] - padw) / ratio
    b[:, 3] = (b[:, 3] - padh) / ratio
    return b


# COCO-80 class names used for labeling detections.
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


def draw_boxes(frame_bgr, boxes, scores, classes):
    """
    Draw bounding boxes and labels in-place on a frame.
    """
    if boxes.size == 0:
        return
    for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores, classes):
        name = COCO80[c] if 0 <= c < len(COCO80) else str(int(c))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"{name}:{s:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def count_by_class(class_ids):
    """
    Count occurrences of each class by name.
    """
    out: Dict[str, int] = {}
    for c in class_ids:
        name = COCO80[c] if 0 <= c < len(COCO80) else str(int(c))
        out[name] = out.get(name, 0) + 1
    return out


# =============== Video → bytes generator with metadata ===============

def video_frames_as_bytes_with_meta(
    path: str,
    meta_by_seq: Dict[int, Dict[str, Any]],
    max_frames: Optional[int] = None,
    size: int = 640,
):
    """
    Generator: read frames from a video, preprocess them, and yield raw tensor bytes.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        seq = 0
        while True:
            if max_frames is not None and seq >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            tensor_nchw, ratio, pads = preprocess_bchw(frame, size=size)
            raw = pack_fp32_bytes(tensor_nchw)

            meta_by_seq[seq] = {
                "frame": frame,
                "ratio": ratio,
                "pads": pads,
                "width": width,
                "height": height,
            }

            yield raw
            seq += 1
    finally:
        cap.release()


# =============== main runner (switchable decode mode) ===============

def main():
    """
    Main entry point: drive the end-to-end video → Triton → annotated video pipeline.
    """
    addr        = os.environ.get("MM_ADDR", "localhost:50051")  # thin proxy address
    video_path  = os.environ.get("VIDEO_PATH", "").strip()
    if not video_path:
        raise SystemExit("Set VIDEO_PATH to a local file to stream.")

    results_dir = os.environ.get("RESULTS_DIR", "results")
    stream_id   = os.environ.get("STREAM_ID", "cam-001")
    img_size    = int(os.environ.get("IMG_SIZE", "640"))
    conf_thres  = float(os.environ.get("CONF_THRES", "0.25"))
    max_frames_s = os.environ.get("MAX_FRAMES", "").strip()
    max_frames   = int(max_frames_s) if max_frames_s.isdigit() else None

    model_name  = os.environ.get("MODEL_NAME", "yolov8n")
    model_ver   = os.environ.get("MODEL_VERSION", "1")
    input_name  = os.environ.get("INPUT_NAME", "images")
    outputs     = tuple(
        s.strip()
        for s in os.environ.get("OUTPUT_NAMES", "output0").split(",")
        if s.strip()
    )
    shape       = tuple(
        int(x) for x in os.environ.get("INPUT_SHAPE", "1,3,640,640").split(",")
    )
    dtype_enum  = getattr(pb, os.environ.get("INPUT_DTYPE", "DT_FP32"), pb.DT_FP32)

    parse_numpy = bool(int(os.environ.get("PARSE_NUMPY", "0")))

    if not Path(video_path).exists():
        raise SystemExit(f"VIDEO_PATH not found: {video_path}")

    # Probe video once for writer: determine FPS + frame size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "frame_counts.jsonl"
    mp4_path   = results_dir / "annotated.mp4"

    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    streamer = TritonBytesStreamer(
        addr=addr,
        input_name=input_name,
        shape=shape,
        dtype_enum=dtype_enum,
        outputs=outputs,
        stream_id=stream_id,
        request_timeout_sec=float(os.environ.get("REQ_TIMEOUT_SEC", "600")),
        parse_outputs_numpy=parse_numpy,
    )

    meta_by_seq: Dict[int, Dict[str, Any]] = {}

    frame_bytes_iter = video_frames_as_bytes_with_meta(
        video_path, meta_by_seq=meta_by_seq, max_frames=max_frames, size=img_size
    )

    try:
        with open(jsonl_path, "w") as fout:
            for seq, ack_msg, decoded in streamer.stream_bytes(
                model_name, model_ver, frame_bytes_iter
            ):

                # ---------- STRICT ERROR HANDLING ON PROXY RESPONSE ----------
                if ack_msg is None:
                    raise RuntimeError(f"proxy returned no message for frame {seq}")

                if "error" in ack_msg and ack_msg["error"]:
                    raise RuntimeError(
                        f"proxy error for frame {seq}: {ack_msg['error']}"
                    )

                out_arrays = []

                if parse_numpy:
                    # decoded: dict[name -> np.ndarray] (or None)
                    if decoded is None:
                        raise RuntimeError(
                            f"proxy returned no decoded outputs for frame {seq} "
                            f"(model={model_name}, version={model_ver})"
                        )
                    for name in outputs:
                        arr = decoded.get(name)
                        if arr is None:
                            raise RuntimeError(
                                f"proxy returned no data for requested output "
                                f"'{name}' on frame {seq}"
                            )
                        out_arrays.append(arr)

                else:
                    # Manual decode from ack_msg["outputs"]
                    outs_desc = ack_msg.get("outputs")
                    if not outs_desc:
                        raise RuntimeError(
                            f"proxy response for frame {seq} contained no outputs "
                            f"(model={model_name}, version={model_ver})"
                        )

                    for name in outputs:
                        od = outs_desc.get(name)
                        if not od:
                            raise RuntimeError(
                                f"proxy returned no data for requested output "
                                f"'{name}' on frame {seq}"
                            )
                        np_dtype = {
                            "FP32": "float32", "FLOAT32": "float32", "FLOAT": "float32",
                            "FP16": "float16", "FLOAT16": "float16",
                            "INT64": "int64", "INT32": "int32", "INT8": "int8", "UINT8": "uint8"
                        }.get(str(od.get("dtype", "FP32")).upper(), "float32")
                        shape_arr = tuple(int(x) for x in od["shape"])
                        raw = base64.b64decode(od["b64"])
                        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape_arr)
                        out_arrays.append(arr)

                if not out_arrays:
                    raise RuntimeError(
                        f"no usable outputs decoded for frame {seq} "
                        f"(model={model_name}, version={model_ver})"
                    )

                meta = meta_by_seq.get(seq)
                if meta is None:
                    raise RuntimeError(
                        f"internal error: no metadata recorded for frame {seq}"
                    )

                boxes, scores, classes = parse_yolo_outputs(
                    out_arrays, conf_thres=conf_thres
                )
                boxes = scale_boxes_back(boxes, meta["ratio"], meta["pads"])
                if boxes.size:
                    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, meta["width"] - 1)
                    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, meta["height"] - 1)

                dets = []
                for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
                    name = COCO80[c] if 0 <= c < len(COCO80) else str(int(c))
                    dets.append({
                        "bbox":   [float(x1), float(y1), float(x2), float(y2)],
                        "score":  float(s),
                        "cls_id": int(c),
                        "cls_name": name,
                    })
                counts = count_by_class(classes)
                rec = {
                    "frame": int(seq),
                    "total": int(len(dets)),
                    "counts": counts,
                    "detections": dets,
                }
                print(rec)
                fout.write(json.dumps(rec, separators=(",", ":")) + "\n")

                frame = meta["frame"]
                draw_boxes(frame, boxes, scores, classes)
                writer.write(frame)
    finally:
        writer.release()

    print(f"[client] wrote:\n  - {jsonl_path}\n  - {mp4_path}")


if __name__ == "__main__":
    main()
