#!/usr/bin/env python3
"""
YOLOv8n Video Inference Client (REST)

Uses standard tritonclient.http library for Triton inference.
Supports batching and annotated video output with bounding boxes.

Usage:
    python yolov8n_video_rest_client.py --video input.mp4 --output results.json
    python yolov8n_video_rest_client.py --video input.mp4 --batch-size 4
    python yolov8n_video_rest_client.py --video input.mp4 --output-video annotated.mp4
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directories
SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR.parent.parent / "results" / "yolov8"

INPUT_SIZE = (640, 640)

# COCO 80 class names for YOLOv8
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def preprocess_frame(frame: np.ndarray):
    """
    Preprocess frame for YOLOv8n with letterbox resize.
    Returns: (tensor, ratio, padding)
    """
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Pad to target size
    img = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), 114, dtype=np.uint8)
    pad_h, pad_w = (INPUT_SIZE[0] - new_h) // 2, (INPUT_SIZE[1] - new_w) // 2
    img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    # BGR -> RGB, normalize, HWC -> CHW
    tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))

    return tensor, scale, (pad_w, pad_h)


def postprocess_yolo(output: np.ndarray, conf_thres: float = 0.25):
    """
    Parse YOLOv8 output tensor to detections.
    Output shape: [batch, 84, num_anchors] -> transpose to [batch, num_anchors, 84]
    Returns list of (boxes, scores, class_ids) per batch item.
    """
    results = []

    # Handle batch dimension
    if output.ndim == 2:
        output = output[np.newaxis, ...]

    for batch_idx in range(output.shape[0]):
        pred = output[batch_idx]

        # YOLOv8 output is [84, num_anchors], transpose to [num_anchors, 84]
        if pred.shape[0] == 84:
            pred = pred.T

        # Extract boxes (cx, cy, w, h) and class scores
        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]

        # Get best class per detection
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)

        # Filter by confidence
        mask = scores >= conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Convert cx,cy,w,h to x1,y1,x2,y2
        boxes_xyxy = np.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

        results.append((boxes_xyxy, scores, class_ids))

    return results


def scale_boxes(boxes: np.ndarray, scale: float, pad: tuple, orig_shape: tuple):
    """Scale boxes from letterbox space back to original image coordinates."""
    if len(boxes) == 0:
        return boxes

    pad_w, pad_h = pad
    boxes = boxes.copy()
    boxes[:, 0] = (boxes[:, 0] - pad_w) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_h) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_w) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_h) / scale

    # Clip to image bounds
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_shape[1])
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_shape[0])

    return boxes


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray):
    """Draw bounding boxes on frame."""
    for box, score, cls_id in zip(boxes.astype(int), scores, class_ids):
        x1, y1, x2, y2 = box
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        text = f"{label}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def extract_frames(video_path: str, fps: float = None, max_frames: int = None):
    """Extract frames from video."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps) if fps and fps < video_fps else 1

    logger.info(f"Video: {video_path}, FPS: {video_fps:.1f}, interval: {interval}")

    count = 0
    extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            yield count, frame
            extracted += 1
            if max_frames and extracted >= max_frames:
                break
        count += 1
    cap.release()


def process_batch(client, headers, batch_data, results, times, conf_thres, use_binary=True, video_writer=None):
    """Process a batch of frames."""
    if not batch_data:
        return 0  # Return payload size

    batch_tensors = [d["tensor"] for d in batch_data]
    batch_frame_nums = [d["frame_num"] for d in batch_data]

    # Stack tensors into batch: [N, 3, 640, 640]
    batch = np.stack(batch_tensors, axis=0).astype(np.float32)
    batch = np.ascontiguousarray(batch)

    # Calculate payload size based on encoding mode
    # Binary mode: raw tensor bytes (similar to gRPC)
    # JSON mode: each float becomes a string like "0.12345678" (~10-12 chars per float)
    if use_binary:
        payload_size_bytes = batch.nbytes
    else:
        # JSON encoding: estimate string representation size
        # Each float32 averages ~10 chars + comma/space overhead
        num_floats = batch.size
        payload_size_bytes = int(num_floats * 12)  # ~12 bytes per float in JSON

    # Build input tensor
    input_tensor = httpclient.InferInput("images", batch.shape, "FP32")
    input_tensor.set_data_from_numpy(batch, binary_data=use_binary)

    # Build output request
    output_tensor = httpclient.InferRequestedOutput("output0", binary_data=use_binary)

    start = time.time()
    try:
        response = client.infer(
            model_name="yolov8n",
            inputs=[input_tensor],
            outputs=[output_tensor],
            headers=headers,
        )
        inference_time = time.time() - start
        per_frame_time = inference_time / len(batch_data)

        # Get output as numpy
        output = response.as_numpy("output0")

        # Postprocess detections
        detections = postprocess_yolo(output, conf_thres)

        for i, (frame_num, det, data) in enumerate(zip(batch_frame_nums, detections, batch_data)):
            boxes, scores, class_ids = det

            # Scale boxes back to original frame coordinates
            boxes = scale_boxes(boxes, data["scale"], data["pad"], data["orig_shape"])

            frame_result = {
                "frame": frame_num,
                "batch_size": len(batch_data),
                "inference_ms": round(per_frame_time * 1000, 2),
                "detections": len(boxes)
            }

            # Add detection details
            if len(boxes) > 0:
                det_list = []
                for box, score, cls_id in zip(boxes, scores, class_ids):
                    det_list.append({
                        "class": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id),
                        "confidence": round(float(score), 3),
                        "bbox": [round(float(x), 1) for x in box]
                    })
                frame_result["detection_details"] = det_list

            results["frames"].append(frame_result)

            # Draw on frame and write to video
            if video_writer is not None:
                annotated = draw_boxes(data["orig_frame"].copy(), boxes, scores, class_ids)
                video_writer.write(annotated)

        times.append(inference_time)

        det_counts = [len(d[0]) for d in detections]
        payload_mb = payload_size_bytes / (1024 * 1024)
        logger.info(
            f"Batch [{','.join(str(f) for f in batch_frame_nums)}]: "
            f"inference={inference_time*1000:6.1f}ms, payload={payload_mb:.1f}MB, detections={det_counts}"
        )
        return payload_size_bytes

    except InferenceServerException as e:
        logger.error(f"Batch error: {e}")
        for frame_num in batch_frame_nums:
            results["frames"].append({"frame": frame_num, "error": str(e)})
        return 0


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n Video Inference (REST)")
    parser.add_argument("--video", "-v", required=True, help="Input video file")
    parser.add_argument("--rest-url", "-u",
                        default=os.environ.get("TRITON_REST_URL", "http://localhost:8080"),
                        help="REST URL (env: TRITON_REST_URL)")
    parser.add_argument("--fps", "-f", type=float, help="Target FPS")
    parser.add_argument("--max-frames", "-n", type=int, help="Max frames")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--conf-thres", "-c", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output", "-o", default=str(RESULTS_DIR / "yolov8n_rest.json"), help=f"Output JSON file (default: {RESULTS_DIR}/yolov8n_rest.json)")
    parser.add_argument("--output-video", default=str(RESULTS_DIR / "annotated_rest.mp4"), help=f"Output annotated video file (default: {RESULTS_DIR}/annotated_rest.mp4)")
    parser.add_argument("--json-encoding", action="store_true", help="Use JSON arrays instead of binary")
    args = parser.parse_args()

    # Setup video writer
    video_writer = None
    if args.output_video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
        logger.info(f"Writing annotated video to: {args.output_video}")

    # Create Triton client (strip http:// prefix for tritonclient)
    url = args.rest_url.replace("http://", "").replace("https://", "").rstrip("/")
    client = httpclient.InferenceServerClient(url=url)

    # Build auth headers
    api_key = os.environ.get("DOMINO_USER_API_KEY")
    headers = {"X-Domino-Api-Key": api_key} if api_key else None

    use_binary = not args.json_encoding

    results = {
        "video": args.video,
        "batch_size": args.batch_size,
        "conf_thres": args.conf_thres,
        "frames": [],
        "stats": {}
    }
    times = []
    total_payload_bytes = 0

    logger.info(f"Starting inference via REST: {args.rest_url}")
    logger.info(f"Batch size: {args.batch_size}, Confidence threshold: {args.conf_thres}")
    logger.info(f"Encoding: {'JSON arrays' if args.json_encoding else 'Binary'}")
    logger.info("-" * 60)

    batch_data = []

    try:
        for frame_num, frame in extract_frames(args.video, args.fps, args.max_frames):
            tensor, scale, pad = preprocess_frame(frame)

            batch_data.append({
                "frame_num": frame_num,
                "tensor": tensor,
                "scale": scale,
                "pad": pad,
                "orig_frame": frame,
                "orig_shape": frame.shape[:2]
            })

            # Process when batch is full
            if len(batch_data) >= args.batch_size:
                payload_bytes = process_batch(client, headers, batch_data, results, times, args.conf_thres, use_binary, video_writer)
                total_payload_bytes += payload_bytes
                batch_data = []

        # Process remaining frames
        if batch_data:
            payload_bytes = process_batch(client, headers, batch_data, results, times, args.conf_thres, use_binary, video_writer)
            total_payload_bytes += payload_bytes

    finally:
        if video_writer is not None:
            video_writer.release()
            # Transcode to H.264 for browser compatibility (mp4v codec not supported by browsers)
            if args.output_video and shutil.which("ffmpeg"):
                temp_file = args.output_video + ".temp.mp4"
                try:
                    os.rename(args.output_video, temp_file)
                    result = subprocess.run(
                        ["ffmpeg", "-i", temp_file, "-c:v", "libx264", "-preset", "fast",
                         "-crf", "23", "-c:a", "copy", "-y", args.output_video],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        os.remove(temp_file)
                        logger.info("Video transcoded to H.264 for browser compatibility")
                    else:
                        # Restore original if transcoding failed
                        os.rename(temp_file, args.output_video)
                        logger.warning(f"FFmpeg transcoding failed: {result.stderr}")
                except Exception as e:
                    logger.warning(f"Failed to transcode video: {e}")
                    if os.path.exists(temp_file):
                        os.rename(temp_file, args.output_video)

    # Stats
    total_frames = len(results["frames"])
    total_detections = sum(f.get("detections", 0) for f in results["frames"])
    total_payload_mb = total_payload_bytes / (1024 * 1024)
    if times:
        total_time = sum(times)
        results["stats"] = {
            "total_frames": total_frames,
            "total_detections": total_detections,
            "batch_size": args.batch_size,
            "total_batches": len(times),
            "total_time_sec": round(total_time, 2),
            "total_payload_mb": round(total_payload_mb, 2),
            "avg_batch_ms": round(np.mean(times) * 1000, 2),
            "avg_frame_ms": round(total_time / total_frames * 1000, 2),
            "fps": round(total_frames / total_time, 2)
        }
        logger.info("-" * 60)
        logger.info(
            f"Processed {total_frames} frames, {total_detections} detections, "
            f"avg {results['stats']['avg_frame_ms']:.1f}ms/frame, "
            f"total payload {total_payload_mb:.1f}MB, "
            f"{results['stats']['fps']:.2f} FPS"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

    if args.output_video:
        logger.info(f"Annotated video saved to: {args.output_video}")


if __name__ == "__main__":
    main()
