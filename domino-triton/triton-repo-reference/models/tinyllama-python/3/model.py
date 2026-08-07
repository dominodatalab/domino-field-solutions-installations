"""
TinyLlama with Guidance Constrained Decoding (Version 3)

This version uses Guidance for template-based constrained generation.
Guidance provides a DSL with gen/select blocks for interleaved generation.

Available templates (set via schema_name input):
  - qa: Simple Q&A with JSON output
  - chain_of_thought: Step-by-step reasoning before answer
  - classification: Multi-class classification with confidence

Example:
    curl -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/3/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Why is the sky blue?"]},
        {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["chain_of_thought"]}
      ]}'

References:
    - https://github.com/guidance-ai/guidance
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
    """Triton Python Backend using Guidance for template-based generation."""

    def initialize(self, args):
        """Initialize the model with Guidance."""
        self.model_config = json.loads(args["model_config"])

        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"[Guidance v3] Loading model: {self.model_id}")

        try:
            import guidance
            from guidance import models, gen, select

            self.llm = models.Transformers(self.model_id)
            self.guidance = guidance
            self.gen = gen
            self.select = select

            logger.info("[Guidance v3] Initialization complete")

        except ImportError as e:
            logger.error(f"Failed to import Guidance: {e}")
            raise

    def execute(self, requests):
        """Execute inference with Guidance template-based generation."""
        responses = []

        for request in requests:
            try:
                prompt = self._get_string_input(request, "prompt")
                template_name = self._get_string_input(request, "schema_name", default="qa")
                max_tokens = self._get_int_input(request, "max_tokens", default=self.default_max_tokens)

                result = self._run_template(template_name, prompt, max_tokens)
                text = json.dumps(result)
                responses.append(self._build_response(text, len(text.split())))

            except Exception as e:
                logger.error(f"[Guidance v3] Error: {e}")
                responses.append(self._build_response(json.dumps({"error": str(e)}), 0))

        return responses

    def _run_template(self, template_name: str, prompt: str, max_tokens: int) -> dict:
        """Run the specified Guidance template."""
        gen = self.gen
        select = self.select

        if template_name == "qa":
            lm = self.llm + f'''Question: {prompt}

Answer in JSON format:
{{"answer": "{gen('answer', max_tokens=100, stop='"')}", "confidence": {select(['0.9', '0.8', '0.7', '0.6', '0.5'], name='confidence')}}}'''

            return {"answer": lm["answer"], "confidence": float(lm["confidence"])}

        elif template_name == "chain_of_thought":
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
            newline = '\n'
            lm = self.llm + f'''Text to classify: {prompt}

Classification result:
- Category: {select(['science', 'technology', 'history', 'geography', 'literature', 'other'], name='category')}
- Subcategory: {gen('subcategory', max_tokens=20, stop=newline)}
- Confidence: {select(['very_high', 'high', 'medium', 'low', 'very_low'], name='confidence')}
- Reasoning: {gen('reasoning', max_tokens=100, stop=newline)}'''

            return {
                "category": lm["category"],
                "subcategory": lm["subcategory"].strip(),
                "confidence": lm["confidence"],
                "reasoning": lm["reasoning"].strip()
            }

        else:
            raise ValueError(f"Unknown template: {template_name}. Available: qa, chain_of_thought, classification")

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
        logger.info("[Guidance v3] Model finalized")
