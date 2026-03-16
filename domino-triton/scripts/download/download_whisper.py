#!/usr/bin/env python3
"""
download_whisper.py

Sets up OpenAI Whisper tiny model for use with NVIDIA Triton Inference Server
using the Python backend for remote inference.

Usage:
    python scripts/download_whisper.py

Requirements:
    pip install transformers torch "optimum[onnxruntime]" librosa soundfile

Output:
    triton-repo/
    ├── models/whisper-tiny-python/
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.py (Python backend)
    └── weights/whisper-tiny-python/
        └── 1/
            ├── config.json
            ├── encoder_model.onnx
            ├── decoder_model.onnx
            └── ... (ONNX model files)

    samples/
    └── audio_sample.wav (test audio)

The model.py loads weights from triton-repo/weights/ at runtime.
This avoids downloading from HuggingFace on every pod restart.
"""

import os
import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID = "openai/whisper-tiny"
TRITON_MODEL_NAME = "whisper-tiny-python"


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from transformers import WhisperProcessor
    except ImportError:
        missing.append("transformers")

    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    except ImportError:
        missing.append('"optimum[onnxruntime]"')

    try:
        import librosa
    except ImportError:
        missing.append("librosa")

    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def download_model_weights(weights_dir: Path):
    """
    Download model weights and save to local weights directory.
    This avoids downloading from HuggingFace on every pod restart.
    """
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import WhisperProcessor

    print(f"\n{'='*60}")
    print(f"Downloading Whisper model: {MODEL_ID}")
    print(f"{'='*60}\n")

    # Target path for weights
    target_path = weights_dir / TRITON_MODEL_NAME / "1"
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (target_path / "config.json").exists():
        print(f"Weights already exist at: {target_path}")
        print("Loading from local path...")
        processor = WhisperProcessor.from_pretrained(str(target_path))
        model = ORTModelForSpeechSeq2Seq.from_pretrained(str(target_path))
        return processor, model

    # Download processor
    print("Downloading processor...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    # Download and export to ONNX
    print("Downloading model and exporting to ONNX...")
    print("(This may take a while on first run)")
    model = ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, export=True)

    # Save to local weights directory
    print(f"\nSaving weights to: {target_path}")
    processor.save_pretrained(str(target_path))
    model.save_pretrained(str(target_path))

    print(f"Weights saved successfully")
    return processor, model


def verify_model(processor, model):
    """Verify the model works correctly."""
    import numpy as np

    print("\n" + "="*60)
    print("Verifying model...")
    print("="*60 + "\n")

    # Create dummy audio input (1 second of silence at 16kHz)
    print("Creating test audio input...")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    # Process audio
    inputs = processor(
        dummy_audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    print(f"Input features shape: {inputs.input_features.shape}")

    # Run inference
    print("Running test inference...")
    generated_ids = model.generate(inputs.input_features, max_length=50)

    # Decode output
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f"Test transcription (silence): '{transcription[0]}'")

    print("\nModel verification: PASSED")

    # Print model info
    print("\nModel Architecture:")
    print("  - Model: openai/whisper-tiny (39M parameters)")
    print("  - Encoder: Processes mel spectrogram features")
    print("  - Decoder: Generates text tokens autoregressively")
    print("  - Input: Audio waveform (16kHz) -> mel spectrogram [batch, 80, 3000]")
    print("  - Output: Token IDs -> decoded to text")


def setup_triton_model(models_dir: Path):
    """
    Set up the Triton Python backend model by copying from source models folder.

    Source: models/<model_name>/  (checked into git)
    Target: triton-repo/models/<model_name>/  (deployment folder)
    """
    # Source directory (triton-repo-reference checked into git)
    source_dir = PROJECT_ROOT / "triton-repo-reference" / "models" / TRITON_MODEL_NAME
    source_config = source_dir / "config.pbtxt"
    source_model_py = source_dir / "1" / "model.py"
    source_requirements = source_dir / "requirements.txt"

    # Target directory (triton-repo)
    target_dir = models_dir / TRITON_MODEL_NAME
    target_version_dir = target_dir / "1"
    target_config = target_dir / "config.pbtxt"
    target_model_py = target_version_dir / "model.py"
    target_requirements = target_dir / "requirements.txt"

    print(f"\n{'='*60}")
    print(f"Setting up Triton model: {TRITON_MODEL_NAME}")
    print(f"{'='*60}\n")

    # Check source files exist
    if not source_config.exists() or not source_model_py.exists():
        print(f"ERROR: Source model files not found!")
        print(f"  Expected: {source_config}")
        print(f"  Expected: {source_model_py}")
        print(f"\n  Make sure triton-repo-reference/ folder is present in your workspace.")
        raise FileNotFoundError(f"Source model files missing for {TRITON_MODEL_NAME}")

    # Create target directories
    target_version_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"Copying model files...")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")

    shutil.copy2(source_config, target_config)
    print(f"  - config.pbtxt: COPIED")

    shutil.copy2(source_model_py, target_model_py)
    print(f"  - 1/model.py: COPIED")

    # Copy requirements-model.txt if it exists
    if source_requirements.exists():
        shutil.copy2(source_requirements, target_requirements)
        print(f"  - requirements-model.txt: COPIED")
        print(f"\n  NOTE: Run './scripts/build_model_packages.sh whisper' to install packages")

    # Display config summary
    print(f"\nModel configuration:")
    with open(target_config) as f:
        config_content = f.read()
        for line in config_content.split('\n'):
            if 'max_batch_size' in line or 'backend' in line or 'name:' in line:
                print(f"  {line.strip()}")

    return target_dir


def create_sample_audio(samples_dir: Path):
    """Create a sample audio file for testing."""
    import numpy as np
    import soundfile as sf

    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_path = samples_dir / "audio_sample.wav"

    if sample_path.exists():
        print(f"\nSample audio already exists: {sample_path}")
        return sample_path

    print(f"\nCreating sample audio file: {sample_path}")

    # Create a 3-second test tone (440Hz sine wave with speech-like envelope)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Generate a simple tone (440Hz = A4 note) with amplitude envelope
    envelope = np.exp(-t * 0.5)  # Decay envelope
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) * envelope

    sf.write(str(sample_path), audio, sample_rate)
    print(f"Created test audio: {sample_path}")
    print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz")

    return sample_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Whisper Setup for Triton Inference Server")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Set up paths - use triton-repo structure
    models_dir = PROJECT_ROOT / "triton-repo" / "models"
    weights_dir = PROJECT_ROOT / "triton-repo" / "weights"
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = PROJECT_ROOT / "samples"

    # Download model weights to local weights directory
    processor, model = download_model_weights(weights_dir)

    # Verify the model works
    verify_model(processor, model)

    # Verify Triton model setup
    model_dir = setup_triton_model(models_dir)

    # Create sample audio
    create_sample_audio(samples_dir)

    weights_path = weights_dir / TRITON_MODEL_NAME / "1"

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nTriton model config: {model_dir}")
    print(f"Model weights: {weights_path}")
    print(f"\nModel structure:")
    print(f"  triton-repo/")
    print(f"  ├── models/{TRITON_MODEL_NAME}/")
    print(f"  │   ├── config.pbtxt      (Triton configuration)")
    print(f"  │   └── 1/model.py        (Python backend)")
    print(f"  └── weights/{TRITON_MODEL_NAME}/")
    print(f"      └── 1/                (ONNX model files)")
    print(f"\nTest audio: samples/audio_sample.wav")
    print(f"\nTo start Triton server:")
    print(f"  docker compose up --build")
    print(f"\nTo test transcription:")
    print(f"  # gRPC client")
    print(f"  python scripts/whisper_audio_grpc_client.py --audio samples/audio_sample.wav")
    print(f"\n  # REST client (with base64 encoding)")
    print(f"  python scripts/whisper_audio_rest_client.py --audio samples/audio_sample.wav")
    print(f"\n  # REST client (with JSON arrays - slower)")
    print(f"  python scripts/whisper_audio_rest_client.py --audio samples/audio_sample.wav --no-binary")
    print(f"\nTo benchmark clients:")
    print(f"  python scripts/benchmark_whisper_clients.py --audio samples/audio_sample.wav")


if __name__ == "__main__":
    main()
