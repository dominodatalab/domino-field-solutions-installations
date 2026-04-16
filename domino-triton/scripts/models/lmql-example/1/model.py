"""
LMQL Constrained Decoding Example for Triton Python Backend

This model demonstrates using the LMQL library for query-based constrained generation.
LMQL (Language Model Query Language) provides a SQL-like query syntax for specifying
constraints on language model outputs.

Key features:
1. SQL-like query syntax - Declarative constraints with WHERE clauses
2. Type constraints - STOPS_AT, STOPS_BEFORE, INT, FLOAT, etc.
3. Logical constraints - Combine constraints with AND, OR
4. Scripted prompts - Mix Python logic with LM generation
5. Distribution constraints - Control token distributions

Requirements:
    pip install lmql torch transformers

Usage:
    Set query_type input to one of: "basic", "constrained", "scripted"

Example curl:
    curl -X POST "http://localhost:8080/v2/models/lmql-example/infer" \\
      -H "Content-Type: application/json" \\
      -d '{"inputs": [
        {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is 2+2?"]},
        {"name": "query_type", "shape": [1], "datatype": "BYTES", "data": ["constrained"]}
      ]}'

References:
    - https://github.com/eth-sri/lmql
    - https://lmql.ai/docs/
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
    Triton Python Backend model using LMQL for query-based constrained generation.

    LMQL works by:
    1. Defining queries with a SQL-like syntax
    2. Specifying constraints in WHERE clauses
    3. Using special constraint types (STOPS_AT, INT, etc.)
    4. Executing queries against a language model

    Key advantage: Declarative constraint specification is intuitive and powerful.
    """

    def initialize(self, args):
        """
        Initialize the model and load LMQL components.

        Called once when the model is loaded. Sets up:
        - The base language model (TinyLlama)
        - LMQL query execution environment
        """
        self.model_config = json.loads(args["model_config"])

        # Get model parameters
        params = self.model_config.get("parameters", {})
        self.model_id = params.get("model_id", {}).get("string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.max_length = int(params.get("max_length", {}).get("string_value", "2048"))
        self.default_max_tokens = int(params.get("default_max_tokens", {}).get("string_value", "256"))

        logger.info(f"Loading LMQL with model: {self.model_id}")

        try:
            import lmql

            # Store LMQL module for query execution
            self.lmql = lmql

            # Set up the model for LMQL
            # LMQL can work with various backends
            self.model_name = f"local:{self.model_id}"

            logger.info("LMQL initialization complete")

        except ImportError as e:
            logger.error(f"Failed to import LMQL: {e}")
            logger.error("Install with: pip install lmql")
            raise

    def execute(self, requests):
        """
        Execute inference requests with LMQL query-based generation.

        For each request:
        1. Extract prompt and query_type from inputs
        2. Build and execute the appropriate LMQL query
        3. Extract results from the query output
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
                query_type = self._get_string_input(request, "query_type", default="basic")
                max_tokens = self._get_int_input(request, "max_tokens", default=self.default_max_tokens)
                temperature = self._get_float_input(request, "temperature", default=0.7)

                logger.info(f"LMQL generation - query_type: {query_type}, prompt: {prompt[:50]}...")

                # Execute the appropriate LMQL query
                result = self._run_query(query_type, prompt, max_tokens, temperature)

                # Serialize result to JSON
                text = json.dumps(result)
                token_count = len(text.split())

                # Build response
                response = self._build_response(text, token_count)
                responses.append(response)

            except Exception as e:
                logger.error(f"Error in LMQL generation: {e}")
                error_response = json.dumps({
                    "error": str(e),
                    "answer": "",
                })
                responses.append(self._build_response(error_response, 0))

        return responses

    def _run_query(self, query_type: str, prompt: str, max_tokens: int, temperature: float) -> dict:
        """
        Run the specified LMQL query.

        Each query type demonstrates different LMQL features:
        - basic: Simple generation with stop condition
        - constrained: Generation with type constraints
        - scripted: Python logic mixed with generation

        Note: LMQL queries are defined as decorated async functions.
        """
        import asyncio

        if query_type == "basic":
            # Basic LMQL query with stop condition
            # STOPS_AT ensures generation stops at the specified string

            @self.lmql.query(model=self.model_name)
            async def basic_query(question):
                '''lmql
                "Question: {question}\n"
                "Answer: [ANSWER]" where STOPS_AT(ANSWER, ".")
                return ANSWER
                '''

            # Run the async query
            result = asyncio.run(basic_query(prompt))
            return {
                "answer": result.strip(),
                "query_type": "basic"
            }

        elif query_type == "constrained":
            # Constrained query with multiple constraint types
            # Demonstrates INT, STOPS_AT, and length constraints

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
            # Scripted query with Python control flow
            # Shows how to mix Python logic with generation

            @self.lmql.query(model=self.model_name)
            async def scripted_query(question):
                '''lmql
                "Question: {question}\n\n"

                # First, classify the question type
                "This is a [QTYPE]" where QTYPE in ["factual question", "opinion question", "math problem", "other"]

                # Then generate appropriate response based on type
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
        logger.info("LMQL model finalized")
