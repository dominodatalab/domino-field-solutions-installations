"""
Instructor Constrained Decoding Example for Triton Python Backend

This model demonstrates using the Instructor library for Pydantic-based structured output.
Instructor is designed to work with OpenAI-compatible APIs but can be adapted for local models.
It uses Pydantic models to define schemas and provides automatic retry logic for validation.

Key features:
1. Pydantic models - Define schemas using familiar Python dataclasses
2. Validation - Automatic validation with retry on failure
3. OpenAI API compatible - Works with any OpenAI-compatible endpoint
4. Field descriptions - Rich descriptions guide the model
5. Nested schemas - Support for complex nested structures

How it works:
- Instructor patches the API client to extract structured data
- It uses function calling or JSON mode to get structured output
- Pydantic validates the output and triggers retries if needed

Requirements:
    pip install instructor pydantic openai transformers torch

Note: This example simulates Instructor's approach for local models.
For production use with local models, consider using Outlines or Guidance instead,
as Instructor is primarily designed for API-based models.

Usage:
    Set schema_name input to one of: "qa", "extraction", "analysis"

Example curl:
    curl -X POST "http://localhost:8080/v2/models/instructor-example/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Extract entities from: Apple Inc. was founded by Steve Jobs in Cupertino."]},
        {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["extraction"]}
      ]}'

References:
    - https://github.com/jxnl/instructor
    - https://python.useinstructor.com/
"""

import json
import logging
import os
import re
from typing import List, Optional

import numpy as np
import triton_python_backend_utils as pb_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """
    Triton Python Backend model demonstrating Instructor-style Pydantic validation.

    This implementation simulates Instructor's approach for local models:
    1. Define Pydantic schemas with rich field descriptions
    2. Generate JSON from the LLM with schema guidance
    3. Validate output against the Pydantic model
    4. Retry with error feedback if validation fails

    Key advantage: Familiar Pydantic syntax with automatic validation and retry.
    """

    def initialize(self, args):
        """
        Initialize the model and set up Pydantic schemas.

        Called once when the model is loaded. Sets up:
        - The base language model and tokenizer
        - Pydantic model definitions
        - Validation and retry logic
        """
        self.model_config = json.loads(args["model_config"])

        # Get model parameters
        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.max_length = int(params.get("max_length", {}).get("string_value", "2048"))
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"Loading Instructor-style model: {self.model_id}")

        try:
            from pydantic import BaseModel, Field, validator
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            # Load the model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            # Create a text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
            )

            # Define Pydantic schemas (similar to how Instructor uses them)
            # These schemas guide the model and validate output

            class QAResponse(BaseModel):
                """Response schema for question-answering tasks."""
                answer: str = Field(
                    description="A concise, direct answer to the question"
                )
                confidence: float = Field(
                    ge=0.0, le=1.0,
                    description="Confidence score between 0 and 1"
                )
                reasoning: Optional[str] = Field(
                    default=None,
                    description="Brief explanation of the answer"
                )

            class Entity(BaseModel):
                """A single extracted entity."""
                name: str = Field(description="The entity name as it appears in text")
                type: str = Field(description="Entity type: person, organization, location, date, or other")

            class EntityExtraction(BaseModel):
                """Response schema for entity extraction tasks."""
                entities: List[Entity] = Field(
                    description="List of extracted entities"
                )
                source_text: str = Field(
                    description="The original text that was analyzed"
                )

            class SentimentAnalysis(BaseModel):
                """Response schema for sentiment analysis tasks."""
                sentiment: str = Field(
                    description="Overall sentiment: positive, negative, or neutral"
                )
                score: float = Field(
                    ge=-1.0, le=1.0,
                    description="Sentiment score from -1 (very negative) to 1 (very positive)"
                )
                aspects: List[str] = Field(
                    default_factory=list,
                    description="Specific aspects mentioned"
                )
                summary: str = Field(
                    description="Brief summary of the sentiment"
                )

            # Store schemas
            self.schemas = {
                "qa": QAResponse,
                "extraction": EntityExtraction,
                "analysis": SentimentAnalysis,
            }

            self.BaseModel = BaseModel
            self.max_retries = 3

            logger.info(f"Instructor-style initialization complete. Schemas: {list(self.schemas.keys())}")

        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            logger.error("Install with: pip install pydantic transformers torch")
            raise

    def execute(self, requests):
        """
        Execute inference requests with Pydantic-validated generation.

        For each request:
        1. Extract prompt and schema_name from inputs
        2. Generate output with schema guidance in the prompt
        3. Parse and validate against Pydantic model
        4. Retry with error feedback if validation fails
        5. Return the validated result

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
                temperature = self._get_float_input(request, "temperature", default=0.3)

                logger.info(f"Instructor generation - schema: {schema_name}, prompt: {prompt[:50]}...")

                # Get the schema
                if schema_name not in self.schemas:
                    available = list(self.schemas.keys())
                    raise ValueError(f"Unknown schema: {schema_name}. Available: {available}")

                schema = self.schemas[schema_name]

                # Generate with validation and retry
                result = self._generate_with_validation(
                    prompt=prompt,
                    schema=schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Serialize result
                text = result.model_dump_json()
                token_count = len(text.split())

                # Build response
                response = self._build_response(text, token_count)
                responses.append(response)

            except Exception as e:
                logger.error(f"Error in Instructor generation: {e}")
                error_response = json.dumps({
                    "error": str(e),
                    "answer": "",
                })
                responses.append(self._build_response(error_response, 0))

        return responses

    def _generate_with_validation(self, prompt: str, schema, max_tokens: int, temperature: float):
        """
        Generate output with Pydantic validation and retry logic.

        This mimics Instructor's core behavior:
        1. Include schema information in the prompt
        2. Generate text from the LLM
        3. Extract and parse JSON
        4. Validate against Pydantic model
        5. If validation fails, retry with error message
        """
        # Get schema information for the prompt
        schema_json = schema.model_json_schema()

        # Build the prompt with schema guidance
        formatted_prompt = self._format_prompt_with_schema(prompt, schema, schema_json)

        last_error = None

        for attempt in range(self.max_retries):
            try:
                # If we had a previous error, add it to the prompt
                if last_error:
                    formatted_prompt += f"\n\nPrevious attempt had validation error: {last_error}\nPlease fix and try again.\n\n"

                # Generate text
                outputs = self.generator(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 0.01),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                generated_text = outputs[0]["generated_text"]

                # Extract the generated part (after the prompt)
                if formatted_prompt in generated_text:
                    generated_text = generated_text[len(formatted_prompt):]

                # Extract JSON from the generated text
                json_str = self._extract_json(generated_text)

                # Parse and validate with Pydantic
                data = json.loads(json_str)
                result = schema(**data)

                logger.info(f"Validation succeeded on attempt {attempt + 1}")
                return result

            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            except Exception as e:
                last_error = f"Validation error: {e}"
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

        # All retries failed, return a default response
        raise ValueError(f"Failed to generate valid output after {self.max_retries} attempts. Last error: {last_error}")

    def _format_prompt_with_schema(self, prompt: str, schema, schema_json: dict) -> str:
        """
        Format the prompt with schema information to guide the model.

        This is similar to how Instructor uses function calling or JSON mode
        to communicate the expected schema to the model.
        """
        # Get field descriptions for the prompt
        properties = schema_json.get("properties", {})
        field_descriptions = []
        for field_name, field_info in properties.items():
            desc = field_info.get("description", "")
            field_type = field_info.get("type", "string")
            field_descriptions.append(f"  - {field_name} ({field_type}): {desc}")

        fields_text = "\n".join(field_descriptions)

        return f"""<|system|>
You are a helpful assistant that responds in valid JSON format.
Your response must match this schema:

{json.dumps(schema_json, indent=2)}

Field descriptions:
{fields_text}

Respond ONLY with valid JSON, no additional text.</s>
<|user|>
{prompt}</s>
<|assistant|>
"""

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from generated text.

        The model might generate extra text before or after the JSON.
        This function extracts just the JSON object.
        """
        # Try to find JSON object
        text = text.strip()

        # Look for JSON between braces
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in generated text")

        # Find matching closing brace
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
        logger.info("Instructor model finalized")
