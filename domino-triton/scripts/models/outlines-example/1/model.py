"""
Outlines Constrained Decoding Example for Triton Python Backend

This model demonstrates using the Outlines library for constrained text generation.
Outlines provides several constrained generation methods:
1. JSON Schema - Generate JSON conforming to a Pydantic model or JSON schema
2. Regex - Generate text matching a regular expression pattern
3. Grammar - Generate text following a context-free grammar
4. Choice - Select from a predefined set of options

Requirements:
    pip install outlines torch transformers

Usage:
    Set schema_name input to one of: "qa", "entity", "sentiment", "regex_phone"

Example curl:
    curl -X POST "http://localhost:8080/v2/models/outlines-example/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is the capital of France?"]},
        {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["qa"]}
      ]}'

References:
    - https://github.com/outlines-dev/outlines
    - https://outlines-dev.github.io/outlines/
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
    Triton Python Backend model using Outlines for constrained generation.

    Outlines works by:
    1. Defining a schema (JSON Schema, regex, grammar, or choices)
    2. Creating a generator that masks invalid tokens at each step
    3. Ensuring output always conforms to the specified schema

    Key advantage: Token-level masking means the model CANNOT produce invalid output.
    """

    def initialize(self, args):
        """
        Initialize the model and load Outlines components.

        Called once when the model is loaded. Sets up:
        - The base language model (TinyLlama)
        - Predefined JSON schemas using Pydantic
        - Outlines generator functions
        """
        self.model_config = json.loads(args["model_config"])

        # Get model parameters
        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.max_length = int(params.get("max_length", {}).get("string_value", "2048"))
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"Loading Outlines with model: {self.model_id}")

        try:
            import outlines
            from outlines import models, generate
            from pydantic import BaseModel, Field

            # Load the model with Outlines wrapper
            # Outlines wraps HuggingFace models to enable constrained generation
            self.model = models.transformers(self.model_id)

            # Define Pydantic schemas for different use cases
            # Outlines uses these to build token masks

            class QAResponse(BaseModel):
                """Schema for question-answering responses."""
                answer: str = Field(description="The answer to the question")
                confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

            class EntityExtraction(BaseModel):
                """Schema for named entity extraction."""
                entities: list[str] = Field(description="List of extracted entities")
                entity_type: str = Field(description="Type of entities (person, place, org)")

            class SentimentAnalysis(BaseModel):
                """Schema for sentiment analysis."""
                sentiment: str = Field(description="positive, negative, or neutral")
                score: float = Field(ge=-1.0, le=1.0, description="Sentiment score -1 to 1")
                reasoning: str = Field(description="Brief explanation")

            # Store schemas for lookup
            self.schemas = {
                "qa": QAResponse,
                "entity": EntityExtraction,
                "sentiment": SentimentAnalysis,
            }

            # Pre-build generators for each schema
            # This compiles the schema into an efficient token mask
            self.generators = {}
            for name, schema in self.schemas.items():
                logger.info(f"Building Outlines generator for schema: {name}")
                self.generators[name] = generate.json(self.model, schema)

            # Also create a regex generator for phone numbers as an example
            phone_pattern = r"\(\d{3}\) \d{3}-\d{4}"
            self.generators["regex_phone"] = generate.regex(self.model, phone_pattern)

            self.outlines = outlines
            self.generate = generate
            logger.info("Outlines initialization complete")

        except ImportError as e:
            logger.error(f"Failed to import Outlines: {e}")
            logger.error("Install with: pip install outlines")
            raise

    def execute(self, requests):
        """
        Execute inference requests with Outlines constrained generation.

        For each request:
        1. Extract prompt and schema_name from inputs
        2. Look up the corresponding Outlines generator
        3. Generate constrained output using the generator
        4. Return the JSON-serialized result

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

                logger.info(f"Outlines generation - schema: {schema_name}, prompt: {prompt[:50]}...")

                # Get the appropriate generator
                if schema_name not in self.generators:
                    available = list(self.generators.keys())
                    raise ValueError(f"Unknown schema: {schema_name}. Available: {available}")

                generator = self.generators[schema_name]

                # Format prompt with chat template
                formatted_prompt = self._format_prompt(prompt, schema_name)

                # Generate with Outlines
                # The generator automatically constrains output to the schema
                result = generator(formatted_prompt, max_tokens=max_tokens)

                # Serialize result to JSON string
                if hasattr(result, "model_dump"):
                    # Pydantic model
                    text = json.dumps(result.model_dump())
                else:
                    # Regex or other result
                    text = str(result)

                # Count tokens (approximate)
                token_count = len(text.split())

                # Build response
                response = self._build_response(text, token_count)
                responses.append(response)

            except Exception as e:
                logger.error(f"Error in Outlines generation: {e}")
                error_response = json.dumps({
                    "error": str(e),
                    "answer": "",
                    "confidence": 0.0
                })
                responses.append(self._build_response(error_response, 0))

        return responses

    def _format_prompt(self, prompt: str, schema_name: str) -> str:
        """Format the prompt with appropriate system message for the schema."""
        schema_instructions = {
            "qa": "Answer the question concisely. Provide a confidence score.",
            "entity": "Extract named entities from the text. Specify the entity type.",
            "sentiment": "Analyze the sentiment of the text. Provide reasoning.",
            "regex_phone": "Extract or generate a phone number in format (XXX) XXX-XXXX.",
        }

        instruction = schema_instructions.get(schema_name, "")

        return f"""<|system|>
{instruction}</s>
<|user|>
{prompt}</s>
<|assistant|>
"""

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
        logger.info("Outlines model finalized")
