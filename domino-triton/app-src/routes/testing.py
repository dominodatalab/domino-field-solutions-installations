"""Testing routes for the Triton Admin Dashboard.

These routes handle model inference testing with sample files.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import settings, get_proxy_url, get_auth_headers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/testing", tags=["Testing"])

# Templates will be set by server.py
templates: Optional[Jinja2Templates] = None


def set_templates(t: Jinja2Templates) -> None:
    """Set the Jinja2 templates instance."""
    global templates
    templates = t


# Model input type definitions
MODEL_INPUT_TYPES: Dict[str, Dict[str, Any]] = {
    "yolov8n": {
        "input_types": ["video", "image"],
        "description": "Object detection model for images and video",
        "client_script": "yolov8n_video_grpc_client.py",
        "result_type": "video",
        "supports_model_arg": False,  # Script has hardcoded model name
    },
    "bert-base-uncased": {
        "input_types": ["text"],
        "description": "Text classification model",
        "client_script": "bert_text_grpc_client.py",
        "result_type": "classification",
        "supports_model_arg": True,
    },
    "whisper-tiny-python": {
        "input_types": ["audio"],
        "description": "Audio transcription model",
        "client_script": "whisper_audio_grpc_client.py",
        "result_type": "transcription",
        "supports_model_arg": True,
    },
    "smollm-135m-python": {
        "input_types": ["text"],
        "description": "Text generation model (SmolLM 135M)",
        "client_script": "llm_text_grpc_client.py",
        "result_type": "generation",
        "supports_model_arg": True,
    },
    "tinyllama-python": {
        "input_types": ["text"],
        "description": "Text generation model (TinyLlama 1.1B)",
        "client_script": "llm_text_grpc_client.py",
        "result_type": "generation",
        "supports_model_arg": True,
    },
    "tinyllama-trtllm": {
        "input_types": ["text"],
        "description": "Text generation model (TinyLlama TensorRT-LLM)",
        "client_script": "llm_text_grpc_client.py",
        "result_type": "generation",
        "supports_model_arg": True,
    },
}


# Request/Response Models
class SampleFile(BaseModel):
    """Sample file information."""
    filename: str
    path: str
    type: str
    size_bytes: int


class TestInferRequest(BaseModel):
    """Inference test request."""
    input_type: str
    protocol: str = "rest"  # "rest" (JSON), "rest-binary", or "grpc"
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    sample_file: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 0.7
    max_frames: int = 10


class TestInferResponse(BaseModel):
    """Inference test response."""
    model: str
    input_type: str
    protocol: str = "rest"
    success: bool
    result: Any
    result_type: Optional[str] = None
    inference_time_ms: Optional[float] = None
    total_inference_time_ms: Optional[float] = None
    payload_size_mb: Optional[float] = None
    fps: Optional[float] = None
    error: Optional[str] = None
    # For media results
    source_file: Optional[str] = None
    result_file: Optional[str] = None


def get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    ext = Path(filename).suffix.lower()
    type_map = {
        ".mp4": "video",
        ".avi": "video",
        ".mov": "video",
        ".mkv": "video",
        ".wav": "audio",
        ".mp3": "audio",
        ".flac": "audio",
        ".ogg": "audio",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".gif": "image",
        ".bmp": "image",
    }
    return type_map.get(ext, "unknown")


def get_mime_type(filename: str) -> str:
    """Get MIME type for a file."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "application/octet-stream")


@router.get("/samples")
async def list_samples():
    """List available sample files."""
    samples_dir = Path(settings.samples_path)

    if not samples_dir.exists():
        return {"samples": []}

    samples = []
    for file_path in samples_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith("."):
            file_type = get_file_type(file_path.name)
            if file_type != "unknown":
                samples.append(SampleFile(
                    filename=file_path.name,
                    path=str(file_path),
                    type=file_type,
                    size_bytes=file_path.stat().st_size,
                ))

    return {"samples": samples}


@router.get("/samples/{file_type}")
async def list_samples_by_type(file_type: str):
    """List sample files filtered by type."""
    all_samples = await list_samples()
    filtered = [s for s in all_samples["samples"] if s.type == file_type]
    return {"samples": filtered}


# Media serving endpoints
@router.get("/media/samples/{filename}")
async def serve_sample_file(filename: str):
    """Serve a sample file for playback."""
    samples_dir = Path(settings.samples_path)
    file_path = samples_dir / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample file not found")

    # Security check - make sure we're not serving files outside samples dir
    if not file_path.resolve().is_relative_to(samples_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=str(file_path),
        media_type=get_mime_type(filename),
        filename=filename,
    )


@router.get("/media/results/{path:path}")
async def serve_result_file(path: str):
    """Serve a result file (annotated video, etc.)."""
    results_dir = Path(settings.results_path)
    file_path = results_dir / path

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")

    # Security check
    if not file_path.resolve().is_relative_to(results_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=str(file_path),
        media_type=get_mime_type(file_path.name),
        filename=file_path.name,
    )


async def fetch_model_type_from_triton(model_name: str, namespace: str) -> Optional[str]:
    """Fetch model_types or model_type parameter from Triton model config."""
    proxy_url = get_proxy_url(namespace)

    async with httpx.AsyncClient(base_url=proxy_url, timeout=10.0, headers=get_auth_headers()) as client:
        try:
            # Try to get model config from Triton
            response = await client.get(f"/v2/models/{model_name}/config")
            if response.status_code == 200:
                config = response.json()
                # Look for model_types or model_type in parameters
                parameters = config.get("parameters", {})

                # Check model_types first (set via dashboard UI)
                if "model_types" in parameters:
                    # Parameters come as {"model_types": {"string_value": "video,image"}}
                    model_types_param = parameters["model_types"]
                    if isinstance(model_types_param, dict):
                        return model_types_param.get("string_value")
                    return model_types_param

                # Fall back to model_type (legacy/alternative parameter name)
                if "model_type" in parameters:
                    model_type_param = parameters["model_type"]
                    if isinstance(model_type_param, dict):
                        return model_type_param.get("string_value")
                    return model_type_param
        except Exception as e:
            logger.debug(f"Could not fetch model config from Triton: {e}")

    return None


def parse_model_type(model_type_str: str) -> dict:
    """Parse model_type string into input_types and result_type."""
    types = [t.strip() for t in model_type_str.split(",")]

    # Determine result type based on input types
    if "text-llm" in types:
        result_type = "generation"
        input_types = ["text"]
    elif "video" in types:
        result_type = "video"
        input_types = types
    elif "image" in types:
        result_type = "image"
        input_types = types
    elif "audio" in types:
        result_type = "transcription"
        input_types = ["audio"]
    else:
        result_type = "text"
        input_types = ["text"]

    return {
        "input_types": input_types,
        "result_type": result_type,
    }


@router.get("/model-types")
async def get_model_types():
    """Get input types for all known models."""
    return {"models": MODEL_INPUT_TYPES}


@router.get("/model-types/{model_name}")
async def get_model_type(model_name: str, namespace: str = Query(default="local")):
    """Get input types for a specific model.

    Priority:
    1. Fetch model_type from Triton model config (parameters.model_type)
    2. Fall back to hardcoded MODEL_INPUT_TYPES dictionary
    3. Fall back to all types (user picks)
    """
    # Try to fetch from Triton model config first
    triton_model_type = await fetch_model_type_from_triton(model_name, namespace)

    if triton_model_type:
        parsed = parse_model_type(triton_model_type)
        return {
            "model": model_name,
            "input_types": parsed["input_types"],
            "result_type": parsed["result_type"],
            "description": f"Model type from config: {triton_model_type}",
            "source": "triton_config",
        }

    # Fall back to hardcoded dictionary
    if model_name in MODEL_INPUT_TYPES:
        return {
            "model": model_name,
            **MODEL_INPUT_TYPES[model_name],
            "source": "hardcoded",
        }

    # Fall back to all types - let user pick
    return {
        "model": model_name,
        "input_types": [],  # Empty means show all types in UI
        "description": "Unknown model - select input type manually",
        "result_type": "text",
        "source": "unknown",
    }


def parse_json_result(results_dir: Path, model_type: str) -> Optional[Dict]:
    """Parse JSON result file from client script output."""
    # Look for the most recent JSON file in the results directory
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        return None

    # Get most recent
    latest = max(json_files, key=lambda f: f.stat().st_mtime)
    try:
        with open(latest) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse result JSON: {e}")
        return None


@router.post("/infer/{model_name}")
async def run_inference(
    model_name: str,
    request: TestInferRequest,
    namespace: str = Query(default="local"),
    protocol: str = Query(default="rest"),
):
    """Run inference on a model using client scripts."""
    from config import get_grpc_url

    # Use protocol from query param or request body
    use_protocol = protocol if protocol else request.protocol

    proxy_url = get_proxy_url(namespace)
    grpc_url = get_grpc_url(namespace)
    scripts_dir = Path(settings.scripts_path) / "clients"
    results_base = Path(settings.results_path)

    # Get model config
    model_config = MODEL_INPUT_TYPES.get(model_name, {})
    result_type = model_config.get("result_type", "text")

    if not model_config:
        # Try to infer based on input type
        if request.input_type == "text":
            base_script = "llm_text"
            result_type = "generation"
        elif request.input_type == "video":
            base_script = "yolov8n_video"
            result_type = "video"
        elif request.input_type == "image":
            base_script = "yolov8n_image"
            result_type = "image"
        elif request.input_type == "audio":
            base_script = "whisper_audio"
            result_type = "transcription"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    else:
        client_script = model_config.get("client_script", "")
        # Extract base script name (without _grpc_client.py or _rest_client.py)
        base_script = client_script.replace("_grpc_client.py", "").replace("_rest_client.py", "")
        # Override base_script for image input type (use image client instead of video client)
        if request.input_type == "image" and "video" in base_script:
            base_script = base_script.replace("video", "image")
            result_type = "image"

    # Select REST or gRPC client script
    # Protocols: "rest" (JSON encoding), "rest-binary" (binary encoding), "grpc"
    use_json_encoding = False
    if use_protocol == "grpc":
        client_script = f"{base_script}_grpc_client.py"
        url_flag = "--grpc-url"
        url_value = grpc_url
    elif use_protocol == "rest-binary":
        client_script = f"{base_script}_rest_client.py"
        url_flag = "--rest-url"
        url_value = proxy_url
        use_json_encoding = False  # Binary is default for REST clients
    else:  # "rest" defaults to JSON encoding
        client_script = f"{base_script}_rest_client.py"
        url_flag = "--rest-url"
        url_value = proxy_url
        use_json_encoding = True

    script_path = scripts_dir / client_script

    if not script_path.exists():
        # Fall back to direct API call (REST only)
        if use_protocol == "grpc":
            raise HTTPException(status_code=400, detail=f"gRPC client script not found: {client_script}")
        return await run_direct_inference(model_name, request, namespace)

    # Determine output paths based on model type
    # Map protocol to suffix for output files
    protocol_suffix = use_protocol.replace("-", "_")  # "rest-binary" -> "rest_binary"
    if "bert" in model_name.lower():
        output_dir = results_base / "bert"
        output_file = output_dir / f"bert_{protocol_suffix}.json"
    elif "whisper" in model_name.lower():
        output_dir = results_base / "whisper"
        output_file = output_dir / f"whisper_{protocol_suffix}.json"
    elif "yolov8" in model_name.lower():
        output_dir = results_base / "yolov8"
        output_file = output_dir / f"yolov8_{protocol_suffix}.json"
        output_video = output_dir / "annotated_output.mp4"
    elif "llm" in model_name.lower() or "smollm" in model_name.lower() or "llama" in model_name.lower():
        output_dir = results_base / "llm"
        output_file = output_dir / f"llm_{protocol_suffix}.json"
    else:
        output_dir = results_base / model_name
        output_file = output_dir / f"{model_name}_{protocol_suffix}.json"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command based on input type
    cmd = ["python", str(script_path), url_flag, url_value]

    # Only add --model if the script supports it
    supports_model_arg = model_config.get("supports_model_arg", True)
    if supports_model_arg:
        cmd.extend(["--model", model_name])

    cmd.extend(["--output", str(output_file)])

    # Add JSON encoding flag for REST (not REST binary)
    # Different clients use different flags:
    # - yolov8n_video_rest_client.py: --json-encoding
    # - yolov8n_image_rest_client.py: --json-encoding
    # - whisper_audio_rest_client.py: --no-binary
    # - Others: no flag (always use binary or don't support switching)
    if use_json_encoding and use_protocol == "rest":
        if "yolov8" in client_script:
            cmd.append("--json-encoding")
        elif "whisper" in client_script:
            cmd.append("--no-binary")

    source_file = None
    result_video_path = None
    result_image_path = None

    if request.input_type == "text":
        if request.text:
            if "llm" in client_script:
                cmd.extend(["--prompt", request.text])
            else:
                cmd.extend(["--texts", request.text])
        elif request.texts:
            cmd.extend(["--texts"] + request.texts)
        if "llm" in client_script:
            cmd.extend(["--max-tokens", str(request.max_tokens)])
    elif request.input_type == "video":
        if request.sample_file:
            sample_path = Path(settings.samples_path) / request.sample_file
            source_file = request.sample_file
            cmd.extend(["--video", str(sample_path)])
            cmd.extend(["--max-frames", str(request.max_frames)])
            # Add output video path for YOLOv8
            if "yolov8" in model_name.lower():
                result_video_path = output_dir / f"annotated_{request.sample_file}"
                # Convert to mp4 extension
                result_video_path = result_video_path.with_suffix(".mp4")
                cmd.extend(["--output-video", str(result_video_path)])
    elif request.input_type == "image":
        if request.sample_file:
            sample_path = Path(settings.samples_path) / request.sample_file
            source_file = request.sample_file
            cmd.extend(["--image", str(sample_path)])
            # Add output image path for YOLOv8
            if "yolov8" in model_name.lower():
                result_image_path = output_dir / f"annotated_{request.sample_file}"
                cmd.extend(["--output-image", str(result_image_path)])
    elif request.input_type == "audio":
        if request.sample_file:
            sample_path = Path(settings.samples_path) / request.sample_file
            source_file = request.sample_file
            cmd.extend(["--audio", str(sample_path)])

    try:
        logger.info(f"Running inference command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(scripts_dir.parent),
        )

        if result.returncode != 0:
            return TestInferResponse(
                model=model_name,
                input_type=request.input_type,
                protocol=use_protocol,
                success=False,
                result=None,
                result_type=result_type,
                error=result.stderr or result.stdout,
            )

        # Parse the JSON output file for structured results
        parsed_result = None
        inference_time_ms = None
        total_inference_time_ms = None
        payload_size_mb = None
        fps = None

        if output_file.exists():
            try:
                with open(output_file) as f:
                    parsed_result = json.load(f)
                    # Extract inference time if available
                    if "stats" in parsed_result:
                        stats = parsed_result["stats"]
                        inference_time_ms = stats.get("avg_text_ms") or stats.get("avg_batch_ms") or stats.get("avg_frame_ms")
                        # Calculate total inference time
                        if "total_time_sec" in stats:
                            total_inference_time_ms = stats["total_time_sec"] * 1000
                        elif "total_time_sec" in parsed_result:
                            total_inference_time_ms = parsed_result["total_time_sec"] * 1000
                        # Extract payload size and FPS
                        payload_size_mb = stats.get("total_payload_mb")
                        fps = stats.get("fps")
                    elif "total_time_sec" in parsed_result:
                        total_inference_time_ms = parsed_result["total_time_sec"] * 1000
                        inference_time_ms = total_inference_time_ms
            except Exception as e:
                logger.error(f"Failed to parse output JSON: {e}")
                parsed_result = result.stdout

        # Format result based on type
        formatted_result = format_result(parsed_result, result_type, result.stdout)

        # Determine result file path for video or image
        result_file_url = None
        if result_video_path and result_video_path.exists():
            result_file_url = f"yolov8/{result_video_path.name}"
        elif result_image_path and result_image_path.exists():
            result_file_url = f"yolov8/{result_image_path.name}"

        return TestInferResponse(
            model=model_name,
            input_type=request.input_type,
            protocol=use_protocol,
            success=True,
            result=formatted_result,
            result_type=result_type,
            inference_time_ms=inference_time_ms,
            total_inference_time_ms=total_inference_time_ms,
            payload_size_mb=payload_size_mb,
            fps=fps,
            source_file=source_file,
            result_file=result_file_url,
        )

    except subprocess.TimeoutExpired:
        return TestInferResponse(
            model=model_name,
            input_type=request.input_type,
            protocol=use_protocol,
            success=False,
            result=None,
            result_type=result_type,
            error="Inference timed out",
        )
    except Exception as e:
        return TestInferResponse(
            model=model_name,
            input_type=request.input_type,
            protocol=use_protocol,
            success=False,
            result=None,
            result_type=result_type,
            error=str(e),
        )


def format_result(parsed_result: Any, result_type: str, raw_output: str) -> Any:
    """Format the inference result based on result type."""
    if parsed_result is None:
        return raw_output

    if result_type == "classification":
        # BERT classification - extract text results
        if isinstance(parsed_result, dict) and "texts" in parsed_result:
            texts = parsed_result["texts"]
            stats = parsed_result.get("stats", {})
            return {
                "type": "classification",
                "results": texts,
                "stats": stats,
            }
    elif result_type == "transcription":
        # Whisper transcription - extract transcription text
        if isinstance(parsed_result, dict):
            if "transcriptions" in parsed_result:
                transcriptions = parsed_result["transcriptions"]
                return {
                    "type": "transcription",
                    "transcriptions": transcriptions,
                    "stats": parsed_result.get("stats", {}),
                }
            elif "files" in parsed_result:
                # Whisper REST client format: files array with transcription field
                files = parsed_result["files"]
                transcriptions = []
                for f in files:
                    if "transcription" in f:
                        transcriptions.append({
                            "file": f.get("file", ""),
                            "transcription": f["transcription"],
                            "inference_ms": f.get("inference_ms"),
                        })
                    elif "error" in f:
                        transcriptions.append({
                            "file": f.get("file", ""),
                            "error": f["error"],
                        })
                return {
                    "type": "transcription",
                    "transcriptions": transcriptions,
                    "stats": parsed_result.get("stats", {}),
                }
            elif "results" in parsed_result:
                return {
                    "type": "transcription",
                    "transcriptions": parsed_result["results"],
                    "stats": parsed_result.get("stats", {}),
                }
    elif result_type == "generation":
        # LLM text generation
        if isinstance(parsed_result, dict):
            if "results" in parsed_result:
                return {
                    "type": "generation",
                    "generations": parsed_result["results"],
                    "stats": parsed_result.get("stats", {}),
                }
            elif "generated_text" in parsed_result:
                return {
                    "type": "generation",
                    "generations": [{"text": parsed_result["generated_text"]}],
                }
    elif result_type == "video":
        # YOLOv8 video detection
        if isinstance(parsed_result, dict):
            return {
                "type": "video",
                "detections": parsed_result.get("detections", []),
                "stats": parsed_result.get("stats", {}),
                "frames_processed": parsed_result.get("frames_processed", 0),
            }
    elif result_type == "image":
        # YOLOv8 image detection
        if isinstance(parsed_result, dict):
            images = parsed_result.get("images", [])
            total_detections = sum(img.get("detections", 0) for img in images)
            return {
                "type": "image",
                "images": images,
                "total_detections": total_detections,
                "stats": parsed_result.get("stats", {}),
            }

    # Default: return as-is
    return parsed_result


async def run_direct_inference(
    model_name: str,
    request: TestInferRequest,
    namespace: str,
) -> TestInferResponse:
    """Run inference directly via the proxy API."""
    proxy_url = get_proxy_url(namespace)
    model_config = MODEL_INPUT_TYPES.get(model_name, {})
    result_type = model_config.get("result_type", "text")

    async with httpx.AsyncClient(base_url=proxy_url, timeout=120.0, headers=get_auth_headers()) as client:
        try:
            if request.input_type == "text":
                # For text models, use the generate endpoint if available
                if "llm" in model_name.lower() or "smollm" in model_name.lower():
                    # LLM inference
                    payload = {
                        "text_input": request.text or (request.texts[0] if request.texts else ""),
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                    }
                    response = await client.post(
                        f"/v2/models/{model_name}/generate",
                        json=payload,
                    )
                else:
                    # BERT-style inference
                    texts = request.texts or ([request.text] if request.text else [])
                    payload = {
                        "inputs": [
                            {
                                "name": "input_ids",
                                "datatype": "INT64",
                                "shape": [len(texts), 128],
                                "data": [[0] * 128] * len(texts),  # Placeholder
                            }
                        ]
                    }
                    response = await client.post(
                        f"/v2/models/{model_name}/infer",
                        json=payload,
                    )

                response.raise_for_status()
                return TestInferResponse(
                    model=model_name,
                    input_type=request.input_type,
                    success=True,
                    result=response.json(),
                    result_type=result_type,
                )
            else:
                return TestInferResponse(
                    model=model_name,
                    input_type=request.input_type,
                    success=False,
                    result=None,
                    result_type=result_type,
                    error=f"Direct API inference not supported for {request.input_type}. Use client scripts.",
                )

        except httpx.HTTPStatusError as e:
            return TestInferResponse(
                model=model_name,
                input_type=request.input_type,
                success=False,
                result=None,
                result_type=result_type,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return TestInferResponse(
                model=model_name,
                input_type=request.input_type,
                success=False,
                result=None,
                result_type=result_type,
                error=str(e),
            )


# Results endpoints
@router.get("/results")
async def list_results():
    """List available benchmark results."""
    results_dir = Path(settings.results_path)

    if not results_dir.exists():
        return {"results": []}

    results = []
    for file_path in results_dir.rglob("*.md"):
        results.append({
            "filename": file_path.name,
            "path": str(file_path.relative_to(results_dir)),
            "size_bytes": file_path.stat().st_size,
            "modified": file_path.stat().st_mtime,
        })

    # Also include HTML files
    for file_path in results_dir.rglob("*.html"):
        results.append({
            "filename": file_path.name,
            "path": str(file_path.relative_to(results_dir)),
            "size_bytes": file_path.stat().st_size,
            "modified": file_path.stat().st_mtime,
        })

    return {"results": sorted(results, key=lambda x: x["modified"], reverse=True)}


@router.get("/results/{path:path}/content")
async def get_result_content(path: str):
    """Get the content of a result file (for markdown rendering)."""
    results_dir = Path(settings.results_path)
    file_path = results_dir / path

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")

    # Security check
    if not file_path.resolve().is_relative_to(results_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    # Only allow markdown and text files
    if file_path.suffix.lower() not in [".md", ".txt", ".json"]:
        raise HTTPException(status_code=400, detail="Only markdown/text files supported")

    try:
        content = file_path.read_text(encoding="utf-8")
        return {
            "filename": file_path.name,
            "path": path,
            "content": content,
            "type": "markdown" if file_path.suffix.lower() == ".md" else "text",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
