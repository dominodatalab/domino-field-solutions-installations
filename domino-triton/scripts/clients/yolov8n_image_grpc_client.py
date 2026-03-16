#!/usr/bin/env python3
"""
YOLOv8n Image Inference Client (gRPC)

Uses standard tritonclient.grpc library for Triton inference.
Supports single image or batch of images.

Usage:
    python yolov8n_image_grpc_client.py --image input.jpg
    python yolov8n_image_grpc_client.py --images img1.jpg img2.jpg img3.jpg
    python yolov8n_image_grpc_client.py --image input.jpg --output-image annotated.jpg
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
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


def preprocess_image(image: np.ndarray):
    """
    Preprocess image for YOLOv8n with letterbox resize.
    Returns: (tensor, ratio, padding)
    """
    h, w = image.shape[:2]
    scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

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


def draw_boxes(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray):
    """Draw bounding boxes on image."""
    for box, score, cls_id in zip(boxes.astype(int), scores, class_ids):
        x1, y1, x2, y2 = box
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        text = f"{label}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def process_image(client, headers, image_path: str, conf_thres: float, output_image: str = None):
    """Process a single image and return results."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    orig_shape = image.shape[:2]

    # Preprocess
    tensor, scale, pad = preprocess_image(image)
    tensor = tensor[np.newaxis, ...]  # Add batch dimension

    # Create input tensor
    input_tensor = grpcclient.InferInput("images", tensor.shape, "FP32")
    input_tensor.set_data_from_numpy(tensor)

    # Create output tensor
    output_tensor = grpcclient.InferRequestedOutput("output0")

    # Run inference
    start = time.time()
    response = client.infer(
        model_name="yolov8n",
        inputs=[input_tensor],
        outputs=[output_tensor],
        headers=headers,
    )
    inference_time = time.time() - start

    # Get output
    output = response.as_numpy("output0")

    # Postprocess
    detections = postprocess_yolo(output, conf_thres)
    boxes, scores, class_ids = detections[0]

    # Scale boxes to original image coordinates
    boxes = scale_boxes(boxes, scale, pad, orig_shape)

    # Build detection list
    detection_list = []
    for box, score, cls_id in zip(boxes.astype(int), scores, class_ids):
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
        detection_list.append({
            "class": label,
            "class_id": int(cls_id),
            "confidence": round(float(score), 4),
            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        })

    result = {
        "image": image_path,
        "detections": len(detection_list),
        "inference_ms": round(inference_time * 1000, 2),
        "objects": detection_list
    }

    # Draw and save annotated image if requested
    if output_image:
        annotated = draw_boxes(image.copy(), boxes, scores, class_ids)
        Path(output_image).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_image, annotated)
        result["output_image"] = output_image
        logger.info(f"Annotated image saved to: {output_image}")

    return result, inference_time


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n Image Inference (gRPC)")
    parser.add_argument("--image", "-i", help="Single input image")
    parser.add_argument("--images", nargs="+", help="Multiple input images")
    parser.add_argument("--grpc-url", "-u",
                        default=os.environ.get("TRITON_GRPC_URL", "localhost:50051"),
                        help="gRPC URL (env: TRITON_GRPC_URL)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output", "-o", default=str(RESULTS_DIR / "yolov8_image_grpc.json"),
                        help=f"Output JSON file (default: {RESULTS_DIR}/yolov8_image_grpc.json)")
    parser.add_argument("--output-image", help="Output annotated image (for single image mode)")
    args = parser.parse_args()

    # Get image list
    if args.image:
        images = [args.image]
    elif args.images:
        images = args.images
    else:
        parser.error("Either --image or --images is required")

    # Create client
    client = grpcclient.InferenceServerClient(url=args.grpc_url)

    # Build auth headers
    api_key = os.environ.get("DOMINO_USER_API_KEY")
    headers = {"x-domino-api-key": api_key} if api_key else None

    results = {
        "model": "yolov8n",
        "conf_thres": args.conf_thres,
        "images": [],
        "stats": {}
    }
    times = []

    logger.info(f"Starting inference via gRPC: {args.grpc_url}")
    logger.info(f"Confidence threshold: {args.conf_thres}")
    logger.info("-" * 60)

    try:
        for idx, image_path in enumerate(images):
            # Determine output image path
            if args.output_image and len(images) == 1:
                output_image = args.output_image
            elif args.output_image:
                # Multiple images - add index to filename
                base = Path(args.output_image)
                output_image = str(base.parent / f"{base.stem}_{idx}{base.suffix}")
            else:
                output_image = None

            result, inference_time = process_image(
                client, headers, image_path, args.conf_thres, output_image
            )
            results["images"].append(result)
            times.append(inference_time)

            logger.info(
                f"[{idx+1}/{len(images)}] {image_path}: "
                f"{result['detections']} detections, "
                f"{result['inference_ms']:.1f}ms"
            )

    except InferenceServerException as e:
        logger.error(f"Inference error: {e}")
        raise
    finally:
        client.close()

    # Stats
    if times:
        results["stats"] = {
            "total_images": len(results["images"]),
            "total_detections": sum(r["detections"] for r in results["images"]),
            "total_time_sec": round(sum(times), 2),
            "avg_inference_ms": round(np.mean(times) * 1000, 2),
            "images_per_sec": round(len(times) / sum(times), 2) if sum(times) > 0 else 0
        }
        logger.info("-" * 60)
        logger.info(
            f"Processed {len(images)} images, "
            f"{results['stats']['total_detections']} total detections, "
            f"avg {results['stats']['avg_inference_ms']:.1f}ms/image"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
