"""
TinyLlama with LMQL Constrained Decoding (Version 4)

This version uses LMQL for SQL-like query-based constrained generation.
LMQL provides declarative constraints via WHERE clauses.

Available query types (set via schema_name input):
  - basic: Simple generation with stop condition
  - constrained: Generation with INT, length constraints
  - scripted: Python control flow mixed with generation

Example:
    curl -X POST "http://localhost:8080/v2/models/tinyllama-python/versions/4/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is 15 + 27?"]},
        {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["scripted"]}
      ]}'

References:
    - https://github.com/eth-sri/lmql
"""

import asyncio
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
    """Triton Python Backend using LMQL for query-based generation."""

    def initialize(self, args):
        """Initialize the model with LMQL."""
        self.model_config = json.loads(args["model_config"])

        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"[LMQL v4] Loading model: {self.model_id}")

        try:
            import lmql
            self.lmql = lmql
            self.model_name = f"local:{self.model_id}"

            logger.info("[LMQL v4] Initialization complete")

        except ImportError as e:
            logger.error(f"Failed to import LMQL: {e}")
            raise

    def execute(self, requests):
        """Execute inference with LMQL query-based generation."""
        responses = []

        for request in requests:
            try:
                prompt = self._get_string_input(request, "prompt")
                query_type = self._get_string_input(request, "schema_name", default="basic")
                max_tokens = self._get_int_input(request, "max_tokens", default=self.default_max_tokens)

                result = self._run_query(query_type, prompt, max_tokens)
                text = json.dumps(result)
                responses.append(self._build_response(text, len(text.split())))

            except Exception as e:
                logger.error(f"[LMQL v4] Error: {e}")
                responses.append(self._build_response(json.dumps({"error": str(e)}), 0))

        return responses

    def _run_query(self, query_type: str, prompt: str, max_tokens: int) -> dict:
        """Run the specified LMQL query."""

        if query_type == "basic":
            @self.lmql.query(model=self.model_name)
            async def basic_query(question):
                '''lmql
                "Question: {question}\n"
                "Answer: [ANSWER]" where STOPS_AT(ANSWER, ".")
                return ANSWER
                '''

            result = asyncio.run(basic_query(prompt))
            return {"answer": result.strip(), "query_type": "basic"}

        elif query_type == "constrained":
            @self.lmql.query(model=self.model_name)
            async def constrained_query(question):
                '''lmql
                "Question: {question}\n"
                "Let me analyze this:\n"
                "- Reasoning: [REASONING]" where STOPS_AT(REASONING, "\n") and len(REASONING) < 200
                "- Confidence (1-10): [CONFIDENCE]" where INT(CONFIDENCE) and CONFIDENCE >= 1 and CONFIDENCE <= 10
                "- Answer: [ANSWER]" where STOPS_AT(ANSWER, "\n") and len(ANSWER) < 100
                return {"reasoning": REASONING, "confidence": CONFIDENCE, "answer": ANSWER}
                '''

            result = asyncio.run(constrained_query(prompt))
            return {
                "reasoning": result["reasoning"].strip(),
                "confidence": int(result["confidence"]),
                "answer": result["answer"].strip(),
                "query_type": "constrained"
            }

        elif query_type == "scripted":
            @self.lmql.query(model=self.model_name)
            async def scripted_query(question):
                '''lmql
                "Question: {question}\n\n"
                "This is a [QTYPE]" where QTYPE in ["factual question", "opinion question", "math problem", "other"]

                if QTYPE == "math problem":
                    "\nSolution:\n"
                    "Step 1: [STEP1]" where STOPS_AT(STEP1, "\n")
                    "Step 2: [STEP2]" where STOPS_AT(STEP2, "\n")
                    "Result: [RESULT]" where STOPS_AT(RESULT, "\n")
                    return {"type": QTYPE, "steps": [STEP1, STEP2], "result": RESULT}
                else:
                    "\nAnswer: [ANSWER]" where STOPS_AT(ANSWER, ".") and len(ANSWER) < 200
                    return {"type": QTYPE, "answer": ANSWER}
                '''

            result = asyncio.run(scripted_query(prompt))
            result["query_type"] = "scripted"
            return result

        else:
            raise ValueError(f"Unknown query_type: {query_type}. Available: basic, constrained, scripted")

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
        logger.info("[LMQL v4] Model finalized")
