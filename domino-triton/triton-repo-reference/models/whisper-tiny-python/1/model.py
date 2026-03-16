"""
Whisper Triton Python Backend Model

This model runs OpenAI Whisper tiny on the Triton server using the Python backend.
It accepts mel spectrogram features and returns transcribed text.

Input:
    - input_features: float32 [batch, 80, 3000] - mel spectrogram
    - decoder_input_ids: int64 [batch, seq] - decoder prompt tokens (optional)

Output:
    - transcription: string [batch] - transcribed text
    - token_ids: int64 [batch, max_length] - generated token IDs (optional)

Note: librosa/soundfile are CLIENT-side dependencies for audio preprocessing.
      The model receives pre-processed mel spectrograms, not raw audio.
"""

# Add model-specific packages to path (if any exist)
import sys
import os
_model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_packages_dir = os.path.join(_model_dir, "packages")
if os.path.isdir(_packages_dir) and _packages_dir not in sys.path:
    sys.path.insert(0, _packages_dir)

import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Whisper model for Triton Python backend."""

    def initialize(self, args):
        """Load the Whisper model and processor from weights folder."""
        self.model_config = json.loads(args["model_config"])
        self.model_instance_name = args["model_instance_name"]
        self.model_name = args["model_name"]
        self.model_version = args.get("model_version", "1")

        # Get model parameters from config
        parameters = self.model_config.get("parameters", {})
        use_onnx = parameters.get("use_onnx", {}).get("string_value", "true").lower() == "true"

        # Determine weights path
        # Model repo is at /triton-repo/models, weights are at /triton-repo/weights (sibling folder)
        model_repo = os.environ.get("MODEL_REPO", "/triton-repo/models")
        base_path = os.path.dirname(model_repo.rstrip("/"))
        weights_path = os.path.join(
            base_path, "weights", self.model_name, self.model_version
        )

        # Fallback to HuggingFace if weights folder doesn't exist
        if not os.path.exists(weights_path):
            weights_path = parameters.get("model_id", {}).get(
                "string_value", "openai/whisper-tiny"
            )
            print(f"[{self.model_name}] Weights folder not found, using: {weights_path}")
        else:
            print(f"[{self.model_name}] Loading weights from: {weights_path}")

        print(f"[{self.model_name}] Use ONNX: {use_onnx}")

        try:
            from transformers import WhisperProcessor

            if use_onnx:
                from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
                self.model = ORTModelForSpeechSeq2Seq.from_pretrained(weights_path)
            else:
                from transformers import WhisperForConditionalGeneration
                self.model = WhisperForConditionalGeneration.from_pretrained(weights_path)

            self.processor = WhisperProcessor.from_pretrained(weights_path)
            print(f"[{self.model_name}] Model loaded successfully")

        except Exception as e:
            print(f"[{self.model_name}] Failed to load model: {e}")
            raise

    def execute(self, requests):
        """Process inference requests."""
        responses = []

        for request in requests:
            try:
                # Get input features (mel spectrogram)
                input_features = pb_utils.get_input_tensor_by_name(request, "input_features")
                if input_features is None:
                    raise ValueError("Missing required input: input_features")

                input_features_np = input_features.as_numpy()

                # Convert to torch tensor
                import torch
                input_features_tensor = torch.from_numpy(input_features_np)

                # Get optional parameters
                language = "en"
                task = "transcribe"

                language_tensor = pb_utils.get_input_tensor_by_name(request, "language")
                if language_tensor is not None:
                    language = language_tensor.as_numpy()[0].decode("utf-8")

                task_tensor = pb_utils.get_input_tensor_by_name(request, "task")
                if task_tensor is not None:
                    task = task_tensor.as_numpy()[0].decode("utf-8")

                # Get decoder prompt IDs for language/task
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task=task
                )

                # Generate transcription
                generated_ids = self.model.generate(
                    input_features_tensor,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                )

                # Decode to text
                transcriptions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                # Prepare outputs
                batch_size = input_features_np.shape[0]

                # Transcription output (as bytes for string tensor)
                transcription_list = [t.strip().encode("utf-8") for t in transcriptions]
                transcription_array = np.array(transcription_list, dtype=np.object_)
                transcription_tensor = pb_utils.Tensor("transcription", transcription_array)

                # Token IDs output (padded to fixed length)
                max_length = 448
                token_ids_np = np.zeros((batch_size, max_length), dtype=np.int64)
                for i, ids in enumerate(generated_ids.numpy()):
                    length = min(len(ids), max_length)
                    token_ids_np[i, :length] = ids[:length]
                token_ids_tensor = pb_utils.Tensor("token_ids", token_ids_np)

                response = pb_utils.InferenceResponse(
                    output_tensors=[transcription_tensor, token_ids_tensor]
                )

            except Exception as e:
                error_msg = f"Inference error: {str(e)}"
                print(f"[{self.model_name}] {error_msg}")
                response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_msg)
                )

            responses.append(response)

        return responses

    def finalize(self):
        """Clean up resources."""
        print(f"[{self.model_name}] Finalizing model")
        self.model = None
        self.processor = None
