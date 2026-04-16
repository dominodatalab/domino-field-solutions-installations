"""
Jsonformer Constrained Decoding Example for Triton Python Backend

This model demonstrates using the Jsonformer library for JSON-only constrained generation.
Jsonformer is a lightweight, focused library specifically designed to generate valid JSON
from language models. It's simpler than general-purpose libraries like Outlines or Guidance.

Key features:
1. JSON Schema input - Define structure with standard JSON Schema
2. Lightweight - No complex dependencies, just token manipulation
3. Guaranteed valid JSON - Structure is enforced, LLM only fills values
4. Fast - Minimal overhead compared to full grammar-based approaches

How it works:
- Jsonformer generates the JSON structure tokens itself (braces, colons, commas)
- The LLM is only called to generate string/number values
- This guarantees structural validity while allowing semantic flexibility

Requirements:
    pip install jsonformer torch transformers

Usage:
    Set schema_name input to one of: "qa", "person", "product"

Example curl:
    curl -X POST "http://localhost:8080/v2/models/jsonformer-example/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Tell me about Albert Einstein"]},
        {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["person"]}
      ]}'

References:
    - https://github.com/1rgs/jsonformer
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import triton_python_backend_utils as pb_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """
    Triton Python Backend model using Jsonformer for JSON-constrained generation.

    Jsonformer works by:
    1. Taking a JSON Schema that defines the expected structure
    2. Generating structural tokens (braces, keys, colons) deterministically
    3. Calling the LLM only for value generation (strings, numbers)
    4. Guaranteeing the output is valid JSON matching the schema

    Key advantage: Extremely simple and fast, with guaranteed structural validity.
    """

    def initialize(self, args):
        """
        Initialize the model and load Jsonformer components.

        Called once when the model is loaded. Sets up:
        - The base language model and tokenizer
        - Predefined JSON schemas for different use cases
        """
        self.model_config = json.loads(args["model_config"])

        # Get model parameters
        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.max_length = int(params.get("max_length", {}).get("string_value", "2048"))
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"Loading Jsonformer with model: {self.model_id}")

        try:
            from jsonformer import Jsonformer
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Load the model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            # Store Jsonformer class
            self.Jsonformer = Jsonformer

            # Define JSON schemas for different use cases
            # These follow the JSON Schema specification
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
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "count": {"type": "number"},
                    },
                    "required": ["items", "count"]
                }
            }

            logger.info(f"Jsonformer initialization complete. Available schemas: {list(self.schemas.keys())}")

        except ImportError as e:
            logger.error(f"Failed to import Jsonformer: {e}")
            logger.error("Install with: pip install jsonformer")
            raise

    def execute(self, requests):
        """
        Execute inference requests with Jsonformer constrained generation.

        For each request:
        1. Extract prompt and schema_name from inputs
        2. Create a Jsonformer instance with the schema
        3. Generate the constrained JSON output
        4. Return the result

        Args:
            requests: List of pb_utils.InferenceRequest objects

        Returns:
            List of pb_utils.InferenceResponse objects
        """
        responses = []

        for request in requests:
            try:
                # Extract inputs
                prompt = self._get_string_input(request, "prompt")
                schema_name = self._get_string_input(request, "schema_name", default="qa")
                max_tokens = self._get_int_input(request, "max_tokens", default=self.default_max_tokens)
                temperature = self._get_float_input(request, "temperature", default=0.7)

                logger.info(f"Jsonformer generation - schema: {schema_name}, prompt: {prompt[:50]}...")

                # Get the schema
                if schema_name not in self.schemas:
                    available = list(self.schemas.keys())
                    raise ValueError(f"Unknown schema: {schema_name}. Available: {available}")

                schema = self.schemas[schema_name]

                # Format the prompt
                formatted_prompt = self._format_prompt(prompt, schema_name)

                # Create Jsonformer instance and generate
                # Jsonformer takes the model, tokenizer, schema, and prompt
                jsonformer = self.Jsonformer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    json_schema=schema,
                    prompt=formatted_prompt,
                    max_string_token_length=max_tokens,
                )

                # Generate the JSON
                result = jsonformer()

                # Serialize result
                text = json.dumps(result)
                token_count = len(text.split())

                # Build response
                response = self._build_response(text, token_count)
                responses.append(response)

            except Exception as e:
                logger.error(f"Error in Jsonformer generation: {e}")
                error_response = json.dumps({
                    "error": str(e),
                    "answer": "",
                })
                responses.append(self._build_response(error_response, 0))

        return responses

    def _format_prompt(self, prompt: str, schema_name: str) -> str:
        """Format the prompt with appropriate context for the schema."""
        schema_instructions = {
            "qa": f"Answer this question: {prompt}",
            "person": f"Provide information about: {prompt}",
            "product": f"Describe this product: {prompt}",
            "list_items": f"List items for: {prompt}",
        }

        return schema_instructions.get(schema_name, prompt)

    def _get_string_input(self, request, name: str, default: Optional[str] = None) -> str:
        """Extract a string input from the request."""
        try:
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            if tensor is None:
                if default is not None:
                    return default
                raise ValueError(f"Missing required input: {name}")
            value = tensor.as_numpy()[0]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return value
        except Exception as e:
            if default is not None:
                return default
            raise

    def _get_int_input(self, request, name: str, default: int) -> int:
        """Extract an integer input from the request."""
        try:
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            if tensor is None:
                return default
            return int(tensor.as_numpy()[0])
        except Exception:
            return default

    def _get_float_input(self, request, name: str, default: float) -> float:
        """Extract a float input from the request."""
        try:
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            if tensor is None:
                return default
            return float(tensor.as_numpy()[0])
        except Exception:
            return default

    def _build_response(self, text: str, token_count: int):
        """Build a Triton inference response."""
        text_tensor = pb_utils.Tensor(
            "generated_text",
            np.array([text], dtype=object)
        )
        count_tensor = pb_utils.Tensor(
            "token_count",
            np.array([token_count], dtype=np.int32)
        )
        return pb_utils.InferenceResponse(output_tensors=[text_tensor, count_tensor])

    def finalize(self):
        """Clean up resources when model is unloaded."""
        logger.info("Jsonformer model finalized")
