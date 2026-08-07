"""
TinyLlama with Instructor Constrained Decoding (Version 6)

This version uses Instructor-style Pydantic validation with retry logic.
It generates output, validates against Pydantic models, and retries on failure.

Available schemas (set via schema_name input):
  - qa: Question-answering with confidence and reasoning
  - extraction: Named entity extraction
  - analysis: Sentiment analysis

Example:
    curl -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/6/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Extract entities: Apple was founded by Steve Jobs."]},
        {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["extraction"]}
      ]}'

References:
    - https://github.com/jxnl/instructor
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import triton_python_backend_utils as pb_utils

MODEL_DIR = Path(__file__).parent.parent
PACKAGES_DIR = MODEL_DIR / "packages"
if PACKAGES_DIR.exists() and str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Triton Python Backend using Instructor-style Pydantic validation."""

    def initialize(self, args):
        """Initialize the model with Pydantic schemas."""
        self.model_config = json.loads(args["model_config"])

        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"[Instructor v6] Loading model: {self.model_id}")

        try:
            from pydantic import BaseModel, Field
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
            )

            # Define Pydantic schemas
            class QAResponse(BaseModel):
                answer: str = Field(description="A concise, direct answer")
                confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
                reasoning: Optional[str] = Field(default=None, description="Brief explanation")

            class Entity(BaseModel):
                name: str = Field(description="Entity name as in text")
                type: str = Field(description="person, organization, location, date, other")

            class EntityExtraction(BaseModel):
                entities: List[Entity] = Field(description="Extracted entities")
                source_text: str = Field(description="Original text analyzed")

            class SentimentAnalysis(BaseModel):
                sentiment: str = Field(description="positive, negative, or neutral")
                score: float = Field(ge=-1.0, le=1.0, description="Score -1 to 1")
                aspects: List[str] = Field(default_factory=list, description="Aspects mentioned")
                summary: str = Field(description="Brief summary")

            self.schemas = {
                "qa": QAResponse,
                "extraction": EntityExtraction,
                "analysis": SentimentAnalysis,
            }

            self.max_retries = 3
            logger.info(f"[Instructor v6] Initialization complete. Schemas: {list(self.schemas.keys())}")

        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise

    def execute(self, requests):
        """Execute inference with Pydantic-validated generation."""
        responses = []

        for request in requests:
            try:
                prompt = self._get_string_input(request, "prompt")
                schema_name = self._get_string_input(request, "schema_name", default="qa")
                max_tokens = self._get_int_input(request, "max_tokens", default=self.default_max_tokens)
                temperature = self._get_float_input(request, "temperature", default=0.3)

                if schema_name not in self.schemas:
                    raise ValueError(f"Unknown schema: {schema_name}. Available: {list(self.schemas.keys())}")

                schema = self.schemas[schema_name]
                result = self._generate_with_validation(prompt, schema, max_tokens, temperature)

                text = result.model_dump_json()
                responses.append(self._build_response(text, len(text.split())))

            except Exception as e:
                logger.error(f"[Instructor v6] Error: {e}")
                responses.append(self._build_response(json.dumps({"error": str(e)}), 0))

        return responses

    def _generate_with_validation(self, prompt, schema, max_tokens, temperature):
        """Generate with Pydantic validation and retry."""
        schema_json = schema.model_json_schema()
        formatted = self._format_prompt_with_schema(prompt, schema_json)

        last_error = None

        for attempt in range(self.max_retries):
            try:
                if last_error:
                    formatted += f"\n\nPrevious error: {last_error}\nPlease fix.\n\n"

                outputs = self.generator(
                    formatted,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 0.01),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                generated = outputs[0]["generated_text"]
                if formatted in generated:
                    generated = generated[len(formatted):]

                json_str = self._extract_json(generated)
                data = json.loads(json_str)
                result = schema(**data)

                logger.info(f"[Instructor v6] Validation succeeded on attempt {attempt + 1}")
                return result

            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            except Exception as e:
                last_error = f"Validation error: {e}"
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

        raise ValueError(f"Failed after {self.max_retries} attempts. Last error: {last_error}")

    def _format_prompt_with_schema(self, prompt, schema_json):
        """Format prompt with schema guidance."""
        properties = schema_json.get("properties", {})
        fields = [f"  - {k} ({v.get('type', 'string')}): {v.get('description', '')}"
                  for k, v in properties.items()]

        return f"""<|system|>
Respond in valid JSON matching this schema:
{json.dumps(schema_json, indent=2)}

Fields:
{chr(10).join(fields)}

Respond ONLY with valid JSON.</s>
<|user|>
{prompt}</s>
<|assistant|>
"""

    def _extract_json(self, text):
        """Extract JSON object from text."""
        text = text.strip()
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found")

        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        return text[start:end]

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

    def _get_float_input(self, request, name, default):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        return float(tensor.as_numpy()[0]) if tensor else default

    def _build_response(self, text, token_count):
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("generated_text", np.array([text], dtype=object)),
            pb_utils.Tensor("token_count", np.array([token_count], dtype=np.int32)),
        ])

    def finalize(self):
        logger.info("[Instructor v6] Model finalized")
