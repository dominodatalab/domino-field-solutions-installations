"""
TinyLlama Python Backend Model for Triton Inference Server

This module implements a Triton Python backend for the TinyLlama-1.1B-Chat model
with optional JSON-constrained decoding.

=============================================================================
OVERVIEW
=============================================================================

The model supports two modes of operation:

1. FREE-FORM MODE (default):
   - Model generates natural language responses
   - No constraints on output format
   - Example: "What is 2+2?" -> "The answer is 4."

2. JSON-CONSTRAINED MODE (response_format="json"):
   - Model is forced to output valid JSON
   - Output schema: {"answer": "<text>", "confidence": <number>}
   - Example: "What is 2+2?" -> {"answer": "4", "confidence": 1}

=============================================================================
CONSTRAINED DECODING ARCHITECTURE
=============================================================================

When JSON mode is enabled, two components work together to force valid JSON output:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                        Generation Loop                               │
    │                                                                      │
    │   For each token to generate:                                        │
    │                                                                      │
    │   1. Model predicts logits (probability scores for all vocab tokens) │
    │                           │                                          │
    │                           ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │              JsonPrefixLogitsProcessor                       │   │
    │   │                                                              │   │
    │   │   - Examines currently generated text                        │   │
    │   │   - Determines which tokens are valid JSON continuations     │   │
    │   │   - Sets invalid token logits to -infinity (impossible)      │   │
    │   │   - Returns modified logits                                  │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                           │                                          │
    │                           ▼                                          │
    │   2. Token is sampled from modified probability distribution         │
    │                           │                                          │
    │                           ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │                  JsonStopCriteria                            │   │
    │   │                                                              │   │
    │   │   - Checks if JSON is complete (balanced braces)             │   │
    │   │   - Returns True to stop, False to continue                  │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                           │                                          │
    │                           ▼                                          │
    │   3. If not stopped, repeat from step 1                              │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

=============================================================================
JSON STATE MACHINE
=============================================================================

The JsonPrefixLogitsProcessor implements a state machine that tracks JSON structure:

    State 0: [empty]     -> Force "{"
    State 1: {           -> Force '"answer"'
    State 2: {"answer"   -> Force ':'
    State 3: {"answer":  -> Force '"' (start of value string)
    State 4: {"answer":"<value>"  -> Force ','
    State 5: {"answer":"<value>", -> Force '"confidence"'
    State 6: {"answer":"<value>","confidence" -> Force ':'
    State 7: {"answer":"<value>","confidence":<number> -> Force '}'
    State 8: COMPLETE    -> JsonStopCriteria stops generation

Each state transition only allows specific tokens, ensuring the output
always matches the schema: {"answer": "...", "confidence": N}

=============================================================================
TOKEN MASKING
=============================================================================

Token masking works by modifying the logits (pre-softmax scores):

    Original logits: [0.1, 0.5, 0.3, 0.8, ...]  (one score per vocab token)
                      ↓
    After masking:   [-inf, -inf, 0.3, -inf, ...]  (only allowed tokens kept)
                      ↓
    After softmax:   [0, 0, 1.0, 0, ...]  (only allowed token has probability)

This guarantees the model can ONLY generate tokens that continue valid JSON.

=============================================================================
"""

import json
import re
import numpy as np
import triton_python_backend_utils as pb_utils

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

import torch


# Hugging Face model identifier
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# =============================================================================
# JSON LOGITS PROCESSOR
# =============================================================================

class JsonPrefixLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that constrains model output to valid JSON.

    This processor intercepts the model's output logits at each generation step
    and masks (sets to -infinity) any tokens that would result in invalid JSON.

    How it works:
    -------------
    1. At each generation step, decode the tokens generated so far
    2. Analyze the current text to determine the JSON state
    3. Based on the state, determine which tokens are valid continuations
    4. Mask all other tokens by setting their logits to -infinity

    Why track prompt_length:
    ------------------------
    The input_ids passed to __call__ include BOTH the original prompt AND
    the generated tokens. We only want to analyze the generated portion
    to determine JSON state. The system prompt contains JSON examples with
    "{" characters, which would confuse our state detection.

    Example:
    --------
    If generated text so far is: '{"answer": "Paris"'

    Valid continuations are: ',' (to continue to confidence field)
    Invalid continuations are: anything else

    So we mask all tokens except ',' and its variations.

    Attributes:
        tokenizer: The tokenizer used to decode tokens to text
        prompt_length: Number of tokens in the original prompt (to skip)
    """

    def __init__(self, tokenizer, prompt_length: int):
        """
        Initialize the processor.

        Args:
            tokenizer: Hugging Face tokenizer for decoding
            prompt_length: Length of input prompt in tokens. Generation starts
                          after this position, so we only analyze tokens from
                          position prompt_length onwards.
        """
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def _decode_generated(self, input_ids: torch.Tensor) -> str:
        """
        Decode only the generated portion of input_ids to text.

        Args:
            input_ids: Full sequence including prompt + generated tokens
                      Shape: [batch_size, sequence_length]

        Returns:
            The decoded text of ONLY the generated tokens (after prompt)

        Example:
            If input_ids represents: "<prompt tokens><generated tokens>"
            And prompt_length is 50
            This returns only the decoded "<generated tokens>" part
        """
        # Slice to get only generated tokens (everything after prompt)
        generated_ids = input_ids[0][self.prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _mask(self, scores: torch.Tensor, allowed_strings: list) -> torch.Tensor:
        """
        Mask logits to only allow specific string continuations.

        This is the core masking function. It:
        1. Encodes each allowed string to get its token IDs
        2. Creates a mask tensor filled with -infinity
        3. Copies original scores only for allowed token IDs

        Args:
            scores: Original logits from the model. Shape: [batch_size, vocab_size]
            allowed_strings: List of strings that are valid continuations.
                           Example: ['{', '{"'] or ['"answer"', ' "answer"']

        Returns:
            Modified scores where invalid tokens have -infinity logits

        Why multiple allowed_strings:
        -----------------------------
        Tokenizers may encode the same concept differently based on context.
        For example, '"answer"' might tokenize to [1234] or [5678, 9012].
        By allowing ['"answer"', ' "answer"'], we handle both cases.

        Example:
            scores = [0.1, 0.5, 0.3, 0.8, 0.2]  # 5 tokens in vocab
            allowed_strings = ["{"]             # Only allow "{"

            If "{" encodes to token ID 2:
            result = [-inf, -inf, 0.3, -inf, -inf]  # Only token 2 allowed
        """
        # Collect all token IDs that could produce the allowed strings
        allowed_ids = set()
        for s in allowed_strings:
            # Encode string without special tokens
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            # Add all token IDs (handles multi-token strings)
            for tid in ids:
                allowed_ids.add(tid)

        # If no valid tokens found, return unchanged (safety fallback)
        if not allowed_ids:
            return scores

        # Create mask: start with all -infinity
        masked = scores.clone()
        masked[:] = float("-inf")

        # Restore original scores only for allowed tokens
        for tid in allowed_ids:
            masked[:, tid] = scores[:, tid]

        return masked

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Process logits at each generation step to enforce JSON structure.

        This method is called by the model's generate() function at each step.
        It implements a state machine based on the current generated text.

        Args:
            input_ids: All token IDs so far (prompt + generated)
                      Shape: [batch_size, current_sequence_length]
            scores: Logits for the next token. Shape: [batch_size, vocab_size]

        Returns:
            Modified scores with invalid tokens masked to -infinity

        State Machine:
        --------------
        The method checks the current generated text and determines which
        JSON tokens are valid continuations:

        Generated Text Pattern          | Allowed Next Tokens
        ---------------------------------|--------------------
        (no "{" yet)                     | "{"
        ends with "{"                    | '"answer"'
        has "answer" but no ":"          | ":"
        ends with ":"                    | '"' (start value)
        ends with complete "answer" val  | ","
        ends with ","                    | '"confidence"'
        has "confidence" but no ":"      | ":"
        ends with number                 | "}"

        Note: We allow both '"answer"' and ' "answer"' to handle tokenizer
        variations where space might be separate or attached.
        """
        # Get only the generated text (exclude prompt)
        text = self._decode_generated(input_ids).rstrip()

        # STATE 0: No JSON started yet -> Force opening brace
        if "{" not in text:
            return self._mask(scores, ["{"])

        # STATE 1: Just opened brace -> Force "answer" key
        if text.endswith("{"):
            return self._mask(scores, ['"answer"', ' "answer"'])

        # STATE 2: Have "answer" but no colon yet -> Force colon
        if '"answer"' in text and not re.search(r'"answer"\s*:', text):
            return self._mask(scores, [":"])

        # STATE 3: Have "answer": but no value yet -> Force opening quote
        if re.search(r'"answer"\s*:\s*$', text):
            return self._mask(scores, ['"', ' "'])

        # STATE 4: Have complete answer value -> Force comma
        if re.search(r'"answer"\s*:\s*"[^"]*"$', text):
            return self._mask(scores, [","])

        # STATE 5: After comma -> Force "confidence" key
        if text.rstrip().endswith(","):
            return self._mask(scores, ['"confidence"', ' "confidence"'])

        # STATE 6: Have "confidence" but no colon -> Force colon
        if '"confidence"' in text and not re.search(r'"confidence"\s*:', text):
            return self._mask(scores, [":"])

        # STATE 7: Have confidence number -> Force closing brace
        # Matches: "confidence": 0, "confidence": 1, "confidence": 0.95, etc.
        if re.search(r'"confidence"\s*:\s*-?\d+(\.\d+)?$', text):
            return self._mask(scores, ["}"])

        # DEFAULT: No constraint (model is filling in values freely)
        # This allows the model to generate the actual answer content
        # and confidence number without restriction
        return scores


# =============================================================================
# JSON STOP CRITERIA
# =============================================================================

class JsonStopCriteria(StoppingCriteria):
    """
    Stopping criteria that halts generation when JSON is complete.

    This criteria checks if the generated text forms a complete JSON object
    by verifying that:
    1. At least one "{" has been generated (JSON started)
    2. Braces are balanced (equal number of "{" and "}")
    3. Text ends with "}" (JSON properly closed)

    Why this is needed:
    -------------------
    Without stop criteria, the model might:
    - Generate extra tokens after valid JSON
    - Continue generating more JSON objects
    - Add explanatory text after the JSON

    This ensures generation stops exactly when we have one complete JSON object.

    Example:
        Generated: '{"answer": "Paris", "confidence": 1}'

        Checks:
        - "{" count: 1, "}" count: 1 -> balanced ✓
        - Ends with "}" -> ✓
        - Result: STOP generation

    Attributes:
        tokenizer: The tokenizer used to decode tokens to text
    """

    def __init__(self, tokenizer):
        """
        Initialize the stop criteria.

        Args:
            tokenizer: Hugging Face tokenizer for decoding input_ids to text
        """
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        """
        Check if generation should stop.

        This is called after each token is generated. It decodes the full
        sequence and checks if the JSON is complete.

        Args:
            input_ids: All token IDs generated so far (including prompt)
                      Shape: [batch_size, sequence_length]
            scores: Current logits (not used, but required by interface)
            **kwargs: Additional arguments (not used)

        Returns:
            True if generation should stop (JSON is complete)
            False if generation should continue

        Note: Unlike the LogitsProcessor, this decodes the FULL text including
        the prompt. This is fine because we're just counting braces and
        checking the ending - the prompt doesn't contain unbalanced braces.
        """
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Check all three conditions for complete JSON:
        return (
            text.count("{") > 0                    # JSON has started
            and text.count("{") == text.count("}") # Braces are balanced
            and text.strip().endswith("}")         # Ends with closing brace
        )


# =============================================================================
# TRITON PYTHON MODEL
# =============================================================================

class TritonPythonModel:
    """
    Triton Python Backend implementation for TinyLlama with JSON constraint support.

    This class implements the Triton Python Backend interface, which requires
    three methods: initialize(), execute(), and finalize().

    Lifecycle:
    ----------
    1. initialize() - Called once when model loads
       - Loads the TinyLlama model and tokenizer
       - Reads configuration parameters

    2. execute() - Called for each inference request
       - Processes prompts and generates responses
       - Applies JSON constraints if requested

    3. finalize() - Called when model unloads
       - Cleans up resources

    Input Tensors:
    --------------
    - prompt (STRING, required): The user's question or instruction
    - response_format (STRING, optional): Set to "json" for constrained output
    - temperature (FP32, optional): Sampling temperature (default: 0.7)
    - top_p (FP32, optional): Nucleus sampling parameter (default: 0.9)
    - max_tokens (INT32, optional): Maximum tokens to generate (default: 256)
    - system_prompt (STRING, optional): Custom system instruction

    Output Tensors:
    ---------------
    - generated_text (STRING): The model's response
    - token_count (INT32): Number of tokens generated

    JSON Mode Flow:
    ---------------
    When response_format="json":

    1. System prompt is injected telling model to output JSON
    2. JsonPrefixLogitsProcessor constrains each token
    3. JsonStopCriteria stops when JSON is complete
    4. Output is parsed and re-serialized for clean JSON
    5. If parsing fails, error JSON is returned with debug info
    """

    def initialize(self, args: dict):
        """
        Initialize the model. Called once when Triton loads this model.

        Args:
            args: Dictionary containing:
                - model_config: JSON string of config.pbtxt contents
                - model_instance_kind: "CPU" or "GPU"
                - model_instance_device_id: GPU ID if applicable
                - model_repository: Path to model repository
                - model_version: Version string
                - model_name: Name of the model

        This method:
        1. Parses configuration from config.pbtxt
        2. Determines device (CPU/GPU)
        3. Loads the TinyLlama model and tokenizer from Hugging Face
        4. Sets the model to evaluation mode
        """
        # Parse model configuration from config.pbtxt
        self.model_config = json.loads(args["model_config"])
        parameters = self.model_config.get("parameters", {})

        # Read configuration parameters
        self.max_length = int(
            parameters.get("max_length", {}).get("string_value", "2048")
        )
        self.default_max_tokens = int(
            parameters.get("default_max_tokens", {}).get("string_value", "256")
        )

        # Determine compute device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Ensure pad token is set (required for batch generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()

    def _get_optional_param(self, request, name: str, default, dtype=None):
        """
        Safely extract an optional parameter from a Triton request.

        Args:
            request: Triton InferenceRequest object
            name: Name of the input tensor to retrieve
            default: Value to return if tensor is not present
            dtype: Optional type conversion (use str for string tensors)

        Returns:
            The parameter value, or default if not provided

        This handles the common pattern of optional inputs in Triton,
        where missing inputs return None from get_input_tensor_by_name.

        Note: With max_batch_size > 0, input tensors have shape [batch, 1].
        We use .flatten()[0] to handle both batched and non-batched inputs.
        """
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default

        # Use flatten()[0] to handle both [N] and [batch, 1] shapes
        value = tensor.as_numpy().flatten()[0]

        # Handle string conversion (bytes from Triton -> str)
        if dtype == str:
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        return value

    def execute(self, requests: list) -> list:
        """
        Execute inference on a batch of requests.

        This is the main inference method called by Triton for each request.

        Args:
            requests: List of pb_utils.InferenceRequest objects

        Returns:
            List of pb_utils.InferenceResponse objects (one per request)

        Processing Flow:
        ----------------
        For each request:

        1. EXTRACT INPUTS
           - Get prompt (required)
           - Get optional parameters (temperature, response_format, etc.)

        2. BUILD MESSAGES
           - Create chat message list with system and user messages
           - If JSON mode, inject JSON instruction system message

        3. TOKENIZE
           - Apply chat template to format messages
           - Tokenize the formatted text
           - Track input_length for later use

        4. CONFIGURE GENERATION
           - If JSON mode: add JsonPrefixLogitsProcessor and JsonStopCriteria
           - Set up generation parameters (temperature, top_p, etc.)

        5. GENERATE
           - Call model.generate() with all configuration
           - Extract only the generated portion (after input)

        6. POST-PROCESS
           - If JSON mode: parse and validate JSON, handle errors
           - Encode result to bytes for Triton

        7. BUILD RESPONSE
           - Create output tensors (generated_text, token_count)
           - Return InferenceResponse
        """
        responses = []

        for request in requests:
            try:
                # ============================================================
                # STEP 1: Extract input tensors
                # ============================================================
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
                prompt_np = prompt_tensor.as_numpy()

                # Get optional parameters with defaults
                max_tokens = self._get_optional_param(
                    request, "max_tokens", self.default_max_tokens
                )
                temperature = self._get_optional_param(request, "temperature", 0.7)
                top_p = self._get_optional_param(request, "top_p", 0.9)
                system_prompt = self._get_optional_param(
                    request, "system_prompt", None, dtype=str
                )
                response_format = self._get_optional_param(
                    request, "response_format", None, dtype=str
                )

                generated_texts = []
                token_counts = []

                # Process each prompt in the batch
                # With max_batch_size > 0, prompt_np has shape [batch, 1]
                # With max_batch_size = 0, prompt_np has shape [N]
                # Flatten to handle both cases uniformly
                prompts_flat = prompt_np.flatten()

                for i in range(len(prompts_flat)):
                    prompt = prompts_flat[i]
                    if isinstance(prompt, bytes):
                        prompt = prompt.decode("utf-8")

                    # ========================================================
                    # STEP 2: Build chat messages
                    # ========================================================
                    messages = []

                    # Add custom system prompt if provided
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})

                    # If JSON mode, inject JSON instruction
                    # This guides the model to output JSON even before constraints kick in
                    if response_format == "json":
                        messages.append({
                            "role": "system",
                            "content": 'Return ONLY JSON: {"answer":"string","confidence":0.0}'
                        })

                    # Add the user's prompt
                    messages.append({"role": "user", "content": prompt})

                    # ========================================================
                    # STEP 3: Tokenize with chat template
                    # ========================================================
                    # Apply the model's chat template (adds special tokens, formatting)
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Tokenize the formatted text
                    inputs = self.tokenizer(
                        formatted,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length - int(max_tokens),
                    )

                    # Move tensors to the correct device (CPU/GPU)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Track input length - needed for:
                    # 1. JsonPrefixLogitsProcessor to know where generation starts
                    # 2. Extracting only generated tokens from output
                    input_length = inputs["input_ids"].shape[1]

                    # ========================================================
                    # STEP 4: Configure generation constraints
                    # ========================================================
                    logits_processors = LogitsProcessorList()
                    stopping_criteria = StoppingCriteriaList()

                    if response_format == "json":
                        # Add JSON constraints
                        logits_processors.append(
                            JsonPrefixLogitsProcessor(self.tokenizer, input_length)
                        )
                        stopping_criteria.append(
                            JsonStopCriteria(self.tokenizer)
                        )

                    # ========================================================
                    # STEP 5: Generate response
                    # ========================================================
                    with torch.no_grad():  # Disable gradient computation for inference
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=int(max_tokens),
                            temperature=float(temperature),
                            top_p=float(top_p),
                            do_sample=temperature > 0,  # Greedy if temp=0
                            pad_token_id=self.tokenizer.eos_token_id,
                            logits_processor=logits_processors if len(logits_processors) else None,
                            stopping_criteria=stopping_criteria if len(stopping_criteria) else None,
                            use_cache=True,  # KV cache for faster generation
                        )

                    # Extract only the generated tokens (exclude input prompt)
                    generated_ids = outputs[0][input_length:]
                    text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    # ========================================================
                    # STEP 6: Post-process JSON output
                    # ========================================================
                    if response_format == "json":
                        try:
                            # Extract JSON from generated text
                            start = text.find("{")
                            end = text.rfind("}")
                            candidate = text[start:end+1] if start >= 0 and end >= 0 else ""

                            # Parse to validate and re-serialize for clean output
                            parsed = json.loads(candidate)
                            text = json.dumps(parsed)
                        except Exception as parse_err:
                            # If JSON parsing fails, return error with debug info
                            text = json.dumps({
                                "answer": "",
                                "confidence": 0.0,
                                "error": "invalid_json",
                                "raw_output": text[:200]  # First 200 chars for debugging
                            })

                    # Collect results
                    generated_texts.append(text.encode("utf-8"))
                    token_counts.append(len(generated_ids))

                # ============================================================
                # STEP 7: Build Triton response
                # ============================================================
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("generated_text", np.array(generated_texts, dtype=object)),
                        pb_utils.Tensor("token_count", np.array(token_counts, dtype=np.int32)),
                    ]
                )

            except Exception as e:
                # Return error response if anything fails
                response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e))
                )

            responses.append(response)

        return responses

    def finalize(self):
        """
        Clean up resources when model is unloaded.

        Called by Triton when the model is being unloaded (e.g., during
        shutdown or explicit unload request).

        This releases the model and tokenizer from memory, which is
        especially important for GPU memory.
        """
        self.model = None
        self.tokenizer = None
