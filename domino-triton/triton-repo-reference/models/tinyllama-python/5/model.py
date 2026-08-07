"""
TinyLlama with Jsonformer Constrained Decoding (Version 5)

This version uses Jsonformer for structural JSON generation.
Jsonformer generates structure tokens deterministically, only calling the LLM for values.

Available schemas (set via schema_name input):
  - qa: Question-answering with confidence
  - person: Person information extraction
  - product: Product description
  - list_items: List generation

Example:
    curl -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/5/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Tell me about Marie Curie"]},
        {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["person"]}
      ]}'

References:
    - https://github.com/1rgs/jsonformer
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import triton_python_backend_utils as pb_utils

MODEL_DIR = Path(__file__).parent.parent
PACKAGES_DIR = MODEL_DIR / "packages"
if PACKAGES_DIR.exists() and str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Triton Python Backend using Jsonformer for structural JSON generation."""

    def initialize(self, args):
        """Initialize the model with Jsonformer."""
        self.model_config = json.loads(args["model_config"])

        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"[Jsonformer v5] Loading model: {self.model_id}")

        try:
            from jsonformer import Jsonformer
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            self.Jsonformer = Jsonformer

            # JSON Schema definitions
            self.schemas = {
                "qa": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["answer", "confidence"]
                },
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "occupation": {"type": "string"},
                        "nationality": {"type": "string"},
                        "birth_year": {"type": "number"},
                        "notable_for": {"type": "string"},
                    },
                    "required": ["name", "occupation", "nationality"]
                },
                "product": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string"},
                        "price": {"type": "number"},
                        "description": {"type": "string"},
                        "in_stock": {"type": "boolean"},
                    },
                    "required": ["name", "category", "price"]
                },
                "list_items": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "string"}},
                        "count": {"type": "number"},
                    },
                    "required": ["items", "count"]
                }
            }

            logger.info(f"[Jsonformer v5] Initialization complete. Schemas: {list(self.schemas.keys())}")

        except ImportError as e:
            logger.error(f"Failed to import Jsonformer: {e}")
            raise

    def execute(self, requests):
        """Execute inference with Jsonformer structural generation."""
        responses = []

        for request in requests:
            try:
                prompt = self._get_string_input(request, "prompt")
                schema_name = self._get_string_input(request, "schema_name", default="qa")
                max_tokens = self._get_int_input(request, "max_tokens", default=self.default_max_tokens)

                if schema_name not in self.schemas:
                    raise ValueError(f"Unknown schema: {schema_name}. Available: {list(self.schemas.keys())}")

                schema = self.schemas[schema_name]
                formatted = self._format_prompt(prompt, schema_name)

                jsonformer = self.Jsonformer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    json_schema=schema,
                    prompt=formatted,
                    max_string_token_length=max_tokens,
                )

                result = jsonformer()
                text = json.dumps(result)
                responses.append(self._build_response(text, len(text.split())))

            except Exception as e:
                logger.error(f"[Jsonformer v5] Error: {e}")
                responses.append(self._build_response(json.dumps({"error": str(e)}), 0))

        return responses

    def _format_prompt(self, prompt: str, schema_name: str) -> str:
        instructions = {
            "qa": f"Answer this question: {prompt}",
            "person": f"Provide information about: {prompt}",
            "product": f"Describe this product: {prompt}",
            "list_items": f"List items for: {prompt}",
        }
        return instructions.get(schema_name, prompt)

    def _get_string_input(self, request, name, default=None):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            if default is not None:
                return default
            raise ValueError(f"Missing required input: {name}")
        value = tensor.as_numpy()[0]
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def _get_int_input(self, request, name, default):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        return int(tensor.as_numpy()[0]) if tensor else default

    def _build_response(self, text, token_count):
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("generated_text", np.array([text], dtype=object)),
            pb_utils.Tensor("token_count", np.array([token_count], dtype=np.int32)),
        ])

    def finalize(self):
        logger.info("[Jsonformer v5] Model finalized")
