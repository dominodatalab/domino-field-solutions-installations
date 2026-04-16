#!/usr/bin/env python3
"""
Whisper Audio Transcription Client (gRPC)

Uses standard tritonclient.grpc library for Triton inference.
Audio is preprocessed client-side to mel spectrogram, then sent to server.
Supports async mode for concurrent file processing.

Usage:
    python whisper_audio_grpc_client.py --audio sample.wav
    python whisper_audio_grpc_client.py --audio-dir audio_files/ --batch-size 4
    python whisper_audio_grpc_client.py --audio file1.wav file2.wav --async  # Async mode
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from auth_helper import get_auth_headers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directories
SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR.parent.parent / "results" / "whisper"

# Whisper expects 16kHz audio
SAMPLE_RATE = 16000
MODEL_NAME = "whisper-tiny-python"


def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    try:
        import librosa
    except ImportError:
        logger.error("librosa not installed. Run: pip install librosa")
        raise SystemExit(1)

    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio.astype(np.float32), sr
    except Exception as e:
        logger.error(f"Failed to load audio file {audio_path}: {e}")
        raise


def get_processor():
    """Load the Whisper processor for audio preprocessing."""
    try:
        from transformers import WhisperProcessor
        return WhisperProcessor.from_pretrained("openai/whisper-tiny")
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        raise SystemExit(1)


def preprocess_audio(processor, audio: np.ndarray) -> np.ndarray:
    """Convert audio waveform to mel spectrogram features."""
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np"
    )
    return inputs.input_features.astype(np.float32)


def process_file_sync(client, headers, processor, audio_path, model):
    """Process a single audio file (sync version). Returns result dict and time."""
    file_start = time.time()
    try:
        # Load and preprocess audio
        audio, sr = load_audio(audio_path)
        duration = len(audio) / sr

        # Convert to mel spectrogram
        preprocess_start = time.time()
        input_features = preprocess_audio(processor, audio)
        preprocess_time = time.time() - preprocess_start

        # Build input tensor
        input_tensor = grpcclient.InferInput("input_features", input_features.shape, "FP32")
        input_tensor.set_data_from_numpy(input_features)

        # Build output request
        output_tensor = grpcclient.InferRequestedOutput("transcription")

        # Send to Triton server
        inference_start = time.time()
        response = client.infer(
            model_name=model,
            inputs=[input_tensor],
            outputs=[output_tensor],
            headers=headers,
        )
        inference_time = time.time() - inference_start

        # Decode transcription
        transcription = response.as_numpy("transcription")[0]
        if isinstance(transcription, bytes):
            transcription = transcription.decode("utf-8")

        total_time = time.time() - file_start

        result = {
            "file": str(audio_path),
            "duration_sec": round(duration, 2),
            "transcription": transcription,
            "preprocess_ms": round(preprocess_time * 1000, 2),
            "inference_ms": round(inference_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
            "realtime_factor": round(total_time / duration, 3) if duration > 0 else None
        }

        logger.info(f"  {Path(audio_path).name}: "
                   f"preprocess={preprocess_time*1000:.1f}ms, "
                   f"inference={inference_time*1000:.1f}ms, "
                   f"RTF={total_time/duration:.2f}x")
        logger.info(f"    -> \"{transcription[:80]}{'...' if len(transcription) > 80 else ''}\"")

        return result, total_time

    except Exception as e:
        logger.error(f"  {audio_path}: Error - {e}")
        return {"file": str(audio_path), "error": str(e)}, 0


# ==================== Async Implementation ====================


async def process_file_async(client, headers, processor, audio_path, model):
    """Process a single audio file (async version). Returns result dict and time."""
    import tritonclient.grpc.aio as grpcclient_aio

    file_start = time.time()
    try:
        # Load and preprocess audio (CPU-bound, run in executor)
        loop = asyncio.get_event_loop()
        audio, sr = await loop.run_in_executor(None, load_audio, audio_path)
        duration = len(audio) / sr

        preprocess_start = time.time()
        input_features = await loop.run_in_executor(None, preprocess_audio, processor, audio)
        preprocess_time = time.time() - preprocess_start

        # Build input tensor
        input_tensor = grpcclient_aio.InferInput("input_features", input_features.shape, "FP32")
        input_tensor.set_data_from_numpy(input_features)

        # Build output request
        output_tensor = grpcclient_aio.InferRequestedOutput("transcription")

        # Send to Triton server
        inference_start = time.time()
        response = await client.infer(
            model_name=model,
            inputs=[input_tensor],
            outputs=[output_tensor],
            headers=headers,
        )
        inference_time = time.time() - inference_start

        # Decode transcription
        transcription = response.as_numpy("transcription")[0]
        if isinstance(transcription, bytes):
            transcription = transcription.decode("utf-8")

        total_time = time.time() - file_start

        result = {
            "file": str(audio_path),
            "duration_sec": round(duration, 2),
            "transcription": transcription,
            "preprocess_ms": round(preprocess_time * 1000, 2),
            "inference_ms": round(inference_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
            "realtime_factor": round(total_time / duration, 3) if duration > 0 else None
        }

        logger.info(f"  {Path(audio_path).name}: "
                   f"preprocess={preprocess_time*1000:.1f}ms, "
                   f"inference={inference_time*1000:.1f}ms, "
                   f"RTF={total_time/duration:.2f}x")
        logger.info(f"    -> \"{transcription[:80]}{'...' if len(transcription) > 80 else ''}\"")

        return result, total_time

    except Exception as e:
        logger.error(f"  {audio_path}: Error - {e}")
        return {"file": str(audio_path), "error": str(e)}, 0


async def run_async(args, audio_files, processor):
    """Run inference in async mode with concurrent file processing."""
    import tritonclient.grpc.aio as grpcclient_aio

    logger.info("Running in ASYNC mode")

    # Create async Triton client
    client = grpcclient_aio.InferenceServerClient(url=args.grpc_url)

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    results = {
        "model": args.model,
        "language": args.language,
        "task": args.task,
        "transport": "gRPC",
        "server": args.grpc_url,
        "async_mode": True,
        "files": [],
        "stats": {}
    }

    logger.info(f"\nStarting async transcription via gRPC: {args.grpc_url}")
    logger.info(f"Model: {args.model}, Language: {args.language}, Task: {args.task}")
    logger.info("-" * 60)

    # Process all files concurrently
    logger.info(f"Processing {len(audio_files)} files concurrently...")

    tasks = [
        process_file_async(client, headers, processor, audio_path, args.model)
        for audio_path in audio_files
    ]

    file_results = await asyncio.gather(*tasks)

    # Collect results
    times = []
    for result, total_time in file_results:
        results["files"].append(result)
        if total_time > 0:
            times.append(total_time)

    await client.close()

    # Calculate stats
    successful = [f for f in results["files"] if "transcription" in f]
    if successful:
        total_duration = sum(f.get("duration_sec", 0) for f in successful)
        total_time = sum(times)
        total_inference = sum(f.get("inference_ms", 0) for f in successful) / 1000

        results["stats"] = {
            "total_files": len(results["files"]),
            "successful": len(successful),
            "total_audio_duration_sec": round(total_duration, 2),
            "total_time_sec": round(total_time, 2),
            "total_inference_sec": round(total_inference, 2),
            "avg_inference_ms": round(total_inference / len(successful) * 1000, 2),
            "avg_realtime_factor": round(total_time / total_duration, 3) if total_duration > 0 else None,
            "throughput_files_per_sec": round(len(successful) / total_time, 2) if total_time > 0 else None,
            "async_mode": True
        }

        logger.info("-" * 60)
        logger.info(f"Transcribed {len(successful)} files, "
                   f"total audio: {total_duration:.1f}s, "
                   f"total time: {total_time:.1f}s, "
                   f"RTF: {total_time/total_duration:.2f}x (async)")

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

    return results


def run_sync(args, audio_files, processor):
    """Run inference in sync mode (original implementation)."""
    # Create Triton client
    client = grpcclient.InferenceServerClient(url=args.grpc_url)

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    results = {
        "model": args.model,
        "language": args.language,
        "task": args.task,
        "transport": "gRPC",
        "server": args.grpc_url,
        "async_mode": False,
        "files": [],
        "stats": {}
    }
    times = []

    logger.info(f"\nStarting transcription via gRPC: {args.grpc_url}")
    logger.info(f"Model: {args.model}, Language: {args.language}, Task: {args.task}")
    logger.info("-" * 60)

    try:
        for audio_path in audio_files:
            result, total_time = process_file_sync(client, headers, processor, audio_path, args.model)
            results["files"].append(result)
            if total_time > 0:
                times.append(total_time)

    finally:
        client.close()

    # Calculate stats
    successful = [f for f in results["files"] if "transcription" in f]
    if successful:
        total_duration = sum(f.get("duration_sec", 0) for f in successful)
        total_time = sum(times)
        total_inference = sum(f.get("inference_ms", 0) for f in successful) / 1000

        results["stats"] = {
            "total_files": len(results["files"]),
            "successful": len(successful),
            "total_audio_duration_sec": round(total_duration, 2),
            "total_time_sec": round(total_time, 2),
            "total_inference_sec": round(total_inference, 2),
            "avg_inference_ms": round(total_inference / len(successful) * 1000, 2),
            "avg_realtime_factor": round(total_time / total_duration, 3) if total_duration > 0 else None,
            "throughput_files_per_sec": round(len(successful) / total_time, 2) if total_time > 0 else None
        }

        logger.info("-" * 60)
        logger.info(f"Transcribed {len(successful)} files, "
                   f"total audio: {total_duration:.1f}s, "
                   f"total time: {total_time:.1f}s, "
                   f"RTF: {total_time/total_duration:.2f}x")

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

    return results


def find_audio_files(path: str) -> List[str]:
    """Find all audio files in a path (file or directory)."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma', '.aac'}
    path = Path(path)

    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        files = []
        for ext in audio_extensions:
            files.extend(str(f) for f in path.glob(f"*{ext}"))
            files.extend(str(f) for f in path.glob(f"*{ext.upper()}"))
        return sorted(files)
    else:
        return []


def main():
    parser = argparse.ArgumentParser(description="Whisper Audio Transcription (gRPC)")
    parser.add_argument("--audio", "-a", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("--audio-dir", help="Directory containing audio files")
    parser.add_argument("--grpc-url", "-u",
                        default=os.environ.get("TRITON_GRPC_URL", "localhost:50051"),
                        help="gRPC proxy URL (env: TRITON_GRPC_URL)")
    parser.add_argument("--model", "-m", default=MODEL_NAME, help="Model name")
    parser.add_argument("--language", "-l", default="en",
                       help="Target language (default: en) - Note: currently uses server default")
    parser.add_argument("--task", "-t", choices=["transcribe", "translate"],
                       default="transcribe",
                       help="Task: transcribe or translate - Note: currently uses server default")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--output", "-o", default=str(RESULTS_DIR / "whisper_grpc.json"), help=f"Output JSON file (default: {RESULTS_DIR}/whisper_grpc.json)")
    parser.add_argument("--async", dest="async_mode", action="store_true", help="Use async mode for concurrent file processing")
    args = parser.parse_args()

    # Collect audio files
    audio_files = []
    if args.audio:
        for path in args.audio:
            audio_files.extend(find_audio_files(path))
    if args.audio_dir:
        audio_files.extend(find_audio_files(args.audio_dir))

    if not audio_files:
        # Use default sample if it exists
        script_dir = Path(__file__).parent
        sample_path = script_dir.parent / "samples" / "audio_sample.wav"
        if sample_path.exists():
            audio_files = [str(sample_path)]
            logger.info(f"Using default sample audio: {sample_path}")
        else:
            logger.error("No audio files specified. Use --audio or --audio-dir")
            logger.error("Or run download_whisper.py to create a sample audio file.")
            raise SystemExit(1)

    logger.info(f"Found {len(audio_files)} audio file(s)")

    # Load processor for audio preprocessing
    logger.info("Loading Whisper processor...")
    processor = get_processor()

    if args.async_mode:
        asyncio.run(run_async(args, audio_files, processor))
    else:
        run_sync(args, audio_files, processor)


if __name__ == "__main__":
    main()
