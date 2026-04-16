"""
Guidance Constrained Decoding Example for Triton Python Backend

This model demonstrates using the Guidance library for template-based constrained generation.
Guidance uses a domain-specific language (DSL) embedded in Python to define generation templates
with interleaved static text and dynamic generation.

Key features:
1. Template DSL - Mix static text with {{gen}} blocks for generation
2. Select - Choose from predefined options: {{select ['opt1', 'opt2']}}
3. Regex constraints - Generate text matching patterns
4. Control flow - Use Python logic within templates
5. Token healing - Automatically handles tokenization boundaries

Requirements:
    pip install guidance torch transformers

Usage:
    Set template_name input to one of: "qa", "chain_of_thought", "classification"

Example curl:
    curl -X POST "http://localhost:8080/v2/models/guidance-example/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What causes rain?"]},
        {"name": "template_name", "shape": [1], "datatype": "BYTES", "data": ["chain_of_thought"]}
      ]}'

References:
    - https://github.com/guidance-ai/guidance
    - https://guidance-ai.github.io/guidance/
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
    Triton Python Backend model using Guidance for template-based constrained generation.

    Guidance works by:
    1. Defining a template with static text and generation blocks
    2. Executing the template with the model, constraining generation at each step
    3. Capturing generated values into named variables

    Key advantage: Natural template syntax makes it easy to define complex output structures.
    """

    def initialize(self, args):
        """
        Initialize the model and load Guidance components.

        Called once when the model is loaded. Sets up:
        - The base language model (TinyLlama)
        - Predefined Guidance templates
        """
        self.model_config = json.loads(args["model_config"])

        # Get model parameters
        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.max_length = int(params.get("max_length", {}).get("string_value", "2048"))
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"Loading Guidance with model: {self.model_id}")

        try:
            import guidance
            from guidance import models, gen, select

            # Load the model with Guidance wrapper
            self.llm = models.Transformers(self.model_id)

            # Store guidance components for use in templates
            self.guidance = guidance
            self.gen = gen
            self.select = select

            logger.info("Guidance initialization complete")

        except ImportError as e:
            logger.error(f"Failed to import Guidance: {e}")
            logger.error("Install with: pip install guidance")
            raise

    def execute(self, requests):
        """
        Execute inference requests with Guidance template-based generation.

        For each request:
        1. Extract prompt and template_name from inputs
        2. Build and execute the appropriate Guidance template
        3. Extract generated values from the result
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
                template_name = self._get_string_input(request, "template_name", default="qa")
                max_tokens = self._get_int_input(request, "max_tokens", default=self.default_max_tokens)
                temperature = self._get_float_input(request, "temperature", default=0.7)

                logger.info(f"Guidance generation - template: {template_name}, prompt: {prompt[:50]}...")

                # Execute the appropriate template
                result = self._run_template(template_name, prompt, max_tokens)

                # Serialize result to JSON
                text = json.dumps(result)
                token_count = len(text.split())

                # Build response
                response = self._build_response(text, token_count)
                responses.append(response)

            except Exception as e:
                logger.error(f"Error in Guidance generation: {e}")
                error_response = json.dumps({
                    "error": str(e),
                    "answer": "",
                    "confidence": 0.0
                })
                responses.append(self._build_response(error_response, 0))

        return responses

    def _run_template(self, template_name: str, prompt: str, max_tokens: int) -> dict:
        """
        Run the specified Guidance template.

        Each template demonstrates different Guidance features:
        - qa: Basic question answering with structured output
        - chain_of_thought: Step-by-step reasoning before answering
        - classification: Multi-class classification with confidence
        """
        gen = self.gen
        select = self.select

        if template_name == "qa":
            # Simple Q&A template with JSON output
            # Uses gen() for free-form generation and select() for constrained choices
            lm = self.llm + f'''Question: {prompt}

Answer in JSON format:
{{"answer": "{gen('answer', max_tokens=100, stop='"')}", "confidence": {select(['0.9', '0.8', '0.7', '0.6', '0.5'], name='confidence')}}}'''

            return {
                "answer": lm["answer"],
                "confidence": float(lm["confidence"])
            }

        elif template_name == "chain_of_thought":
            # Chain-of-thought reasoning template
            # Forces the model to show reasoning steps before the final answer
            lm = self.llm + f'''Question: {prompt}

Let me think step by step:
Step 1: {gen('step1', max_tokens=50, stop='Step')}
Step 2: {gen('step2', max_tokens=50, stop='Step')}
Step 3: {gen('step3', max_tokens=50, stop='Therefore')}

Therefore, the answer is: {gen('answer', max_tokens=50, stop='.')}

Confidence: {select(['high', 'medium', 'low'], name='confidence')}'''

            return {
                "reasoning": {
                    "step1": lm["step1"].strip(),
                    "step2": lm["step2"].strip(),
                    "step3": lm["step3"].strip(),
                },
                "answer": lm["answer"].strip(),
                "confidence": lm["confidence"]
            }

        elif template_name == "classification":
            # Classification template with structured categories
            # Uses select() to constrain to valid categories
            lm = self.llm + f'''Text to classify: {prompt}

Classification result:
- Category: {select(['science', 'technology', 'history', 'geography', 'literature', 'other'], name='category')}
- Subcategory: {gen('subcategory', max_tokens=20, stop='\n')}
- Confidence: {select(['very_high', 'high', 'medium', 'low', 'very_low'], name='confidence')}
- Reasoning: {gen('reasoning', max_tokens=100, stop='\n')}'''

            return {
                "category": lm["category"],
                "subcategory": lm["subcategory"].strip(),
                "confidence": lm["confidence"],
                "reasoning": lm["reasoning"].strip()
            }

        else:
            raise ValueError(f"Unknown template: {template_name}. Available: qa, chain_of_thought, classification")

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
        logger.info("Guidance model finalized")
