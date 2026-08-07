"""
TinyLlama with Outlines Constrained Decoding (Version 2)

This version uses Outlines for token-level constrained generation via JSON Schema.
Outlines compiles schemas into finite state machines for efficient token masking.

Available schemas (set via schema_name input):
  - qa: Question-answering with confidence score
  - entity: Named entity extraction
  - sentiment: Sentiment analysis
  - regex_phone: Phone number extraction (regex pattern)

Example:
    curl -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/2/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is 2+2?"]},
        {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["qa"]}
      ]}'

References:
    - https://github.com/outlines-dev/outlines
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import triton_python_backend_utils as pb_utils

# Add packages directory to path for model-specific dependencies
MODEL_DIR = Path(__file__).parent.parent
PACKAGES_DIR = MODEL_DIR / "packages"
if PACKAGES_DIR.exists() and str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Triton Python Backend using Outlines for constrained generation."""

    def initialize(self, args):
        """Initialize the model with Outlines generators."""
        self.model_config = json.loads(args["model_config"])

        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"[Outlines v2] Loading model: {self.model_id}")

        try:
            import outlines
            from outlines import models, generate
            from pydantic import BaseModel, Field

            # Load model with Outlines wrapper
            self.model = models.transformers(self.model_id)

            # Define Pydantic schemas
            class QAResponse(BaseModel):
                answer: str = Field(description="The answer to the question")
                confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

            class EntityExtraction(BaseModel):
                entities: list[str] = Field(description="List of extracted entities")
                entity_type: str = Field(description="Type of entities")

            class SentimentAnalysis(BaseModel):
                sentiment: str = Field(description="positive, negative, or neutral")
                score: float = Field(ge=-1.0, le=1.0, description="Sentiment score -1 to 1")
                reasoning: str = Field(description="Brief explanation")

            self.schemas = {
                "qa": QAResponse,
                "entity": EntityExtraction,
                "sentiment": SentimentAnalysis,
            }

            # Build generators
            self.generators = {}
            for name, schema in self.schemas.items():
                logger.info(f"[Outlines v2] Building generator for: {name}")
                self.generators[name] = generate.json(self.model, schema)

            # Regex generator for phone numbers
            self.generators["regex_phone"] = generate.regex(self.model, r"\(\d{3}\) \d{3}-\d{4}")

            logger.info("[Outlines v2] Initialization complete")

        except ImportError as e:
            logger.error(f"Failed to import Outlines: {e}")
            raise

    def execute(self, requests):
        """Execute inference with Outlines constrained generation."""
        responses = []

        for request in requests:
            try:
                prompt = self._get_string_input(request, "prompt")
                schema_name = self._get_string_input(request, "schema_name", default="qa")
                max_tokens = self._get_int_input(request, "max_tokens", default=self.default_max_tokens)

                if schema_name not in self.generators:
                    raise ValueError(f"Unknown schema: {schema_name}. Available: {list(self.generators.keys())}")

                generator = self.generators[schema_name]
                formatted = self._format_prompt(prompt, schema_name)
                result = generator(formatted, max_tokens=max_tokens)

                if hasattr(result, "model_dump"):
                    text = json.dumps(result.model_dump())
                else:
                    text = str(result)

                responses.append(self._build_response(text, len(text.split())))

            except Exception as e:
                logger.error(f"[Outlines v2] Error: {e}")
                responses.append(self._build_response(json.dumps({"error": str(e)}), 0))

        return responses

    def _format_prompt(self, prompt: str, schema_name: str) -> str:
        instructions = {
            "qa": "Answer concisely with confidence score.",
            "entity": "Extract named entities from the text.",
            "sentiment": "Analyze sentiment of the text.",
            "regex_phone": "Extract phone number in (XXX) XXX-XXXX format.",
        }
        return f"<|system|>\n{instructions.get(schema_name, '')}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"

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
        logger.info("[Outlines v2] Model finalized")
