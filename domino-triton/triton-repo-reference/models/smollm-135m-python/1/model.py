"""
SmolLM-135M Triton Python Backend Model

This model runs SmolLM-135M-Instruct on the Triton server using the Python backend.
It accepts text prompts and returns generated text responses.

Model weights are loaded from the weights folder:
    <base>/weights/smollm-135m-python/1/

Directory structure:
    <base>/
    ├── models/smollm-135m-python/     <- Triton loads from here
    │   ├── config.pbtxt
    │   ├── packages/                  <- Model-specific packages (if any)
    │   └── 1/model.py (this file)
    └── weights/smollm-135m-python/    <- Model weights
        └── 1/<HF model files>

Input:
    - prompt: string [batch] - user prompt text
    - max_tokens: int32 [batch] - max tokens to generate (optional, default 256)
    - temperature: float32 [batch] - sampling temperature (optional, default 0.7)
    - top_p: float32 [batch] - nucleus sampling parameter (optional, default 0.9)
    - system_prompt: string [batch] - system prompt (optional)

Output:
    - generated_text: string [batch] - generated response text
    - token_count: int32 [batch] - number of tokens generated
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
    """SmolLM model for Triton Python backend."""

    def initialize(self, args):
        """Load the LLM model and tokenizer from weights folder."""
        self.model_config = json.loads(args["model_config"])
        self.model_instance_name = args["model_instance_name"]
        self.model_name = args["model_name"]
        self.model_version = args.get("model_version", "1")

        # Get model parameters from config
        parameters = self.model_config.get("parameters", {})
        self.max_length = int(
            parameters.get("max_length", {}).get("string_value", "2048")
        )
        self.default_max_tokens = int(
            parameters.get("default_max_tokens", {}).get("string_value", "256")
        )

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
                "string_value", "HuggingFaceTB/SmolLM-135M-Instruct"
            )
            print(f"[{self.model_name}] Weights folder not found, using: {weights_path}")
        else:
            print(f"[{self.model_name}] Loading weights from: {weights_path}")

        print(f"[{self.model_name}] Max length: {self.max_length}")
        print(f"[{self.model_name}] Default max tokens: {self.default_max_tokens}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[{self.model_name}] Using device: {self.device}")

            # Load tokenizer
            print(f"[{self.model_name}] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

            # Load model
            print(f"[{self.model_name}] Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                weights_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            self.model.eval()

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"[{self.model_name}] Model loaded successfully")
            print(
                f"[{self.model_name}] Parameters: "
                f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M"
            )

        except Exception as e:
            print(f"[{self.model_name}] Failed to load model: {e}")
            raise

    def _get_optional_param(self, request, name, default, dtype=None):
        """Get an optional parameter from the request."""
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        value = tensor.as_numpy()[0]
        if dtype == str:
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return value

    def execute(self, requests):
        """Process inference requests."""
        import torch

        responses = []

        for request in requests:
            try:
                # Get prompt (required)
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
                if prompt_tensor is None:
                    raise ValueError("Missing required input: prompt")

                prompt_np = prompt_tensor.as_numpy()
                batch_size = prompt_np.shape[0]

                # Get optional parameters
                max_tokens = self._get_optional_param(
                    request, "max_tokens", self.default_max_tokens
                )
                temperature = self._get_optional_param(request, "temperature", 0.7)
                top_p = self._get_optional_param(request, "top_p", 0.9)
                system_prompt = self._get_optional_param(
                    request, "system_prompt", None, dtype=str
                )

                generated_texts = []
                token_counts = []

                for i in range(batch_size):
                    # Decode prompt
                    prompt = prompt_np[i]
                    if isinstance(prompt, bytes):
                        prompt = prompt.decode("utf-8")
                    elif isinstance(prompt, np.ndarray):
                        prompt = prompt.item()
                        if isinstance(prompt, bytes):
                            prompt = prompt.decode("utf-8")

                    # Build chat messages
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    # Apply chat template
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Tokenize
                    inputs = self.tokenizer(
                        formatted,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length - int(max_tokens),
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    input_length = inputs["input_ids"].shape[1]

                    # Generate
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=int(max_tokens),
                            temperature=float(temperature) if temperature > 0 else None,
                            top_p=float(top_p) if temperature > 0 else None,
                            do_sample=temperature > 0,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                    # Decode only the generated part
                    generated_ids = outputs[0][input_length:]
                    generated_text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )

                    generated_texts.append(generated_text.encode("utf-8"))
                    token_counts.append(len(generated_ids))

                # Prepare outputs
                text_array = np.array(generated_texts, dtype=np.object_)
                text_tensor = pb_utils.Tensor("generated_text", text_array)

                count_array = np.array(token_counts, dtype=np.int32)
                count_tensor = pb_utils.Tensor("token_count", count_array)

                response = pb_utils.InferenceResponse(
                    output_tensors=[text_tensor, count_tensor]
                )

            except Exception as e:
                error_msg = f"Inference error: {str(e)}"
                print(f"[{self.model_name}] {error_msg}")
                import traceback

                traceback.print_exc()
                response = pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(error_msg)
                )

            responses.append(response)

        return responses

    def finalize(self):
        """Clean up resources."""
        print(f"[{self.model_name}] Finalizing model")
        self.model = None
        self.tokenizer = None