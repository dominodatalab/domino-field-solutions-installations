#!/usr/bin/env python3
"""
TinyLlama Constrained Decoding Client

This client demonstrates constrained decoding with the tinyllama-python model.
The model supports multiple versions, each using a different constrained decoding library:

  Version 1: Custom LogitsProcessor - Manual JSON constraint
  Version 2: Outlines - JSON Schema, regex, grammar via token masking
  Version 3: Guidance - Template DSL with gen/select blocks
  Version 4: LMQL - SQL-like query language for constraints
  Version 5: Jsonformer - Structural JSON generation
  Version 6: Instructor - Pydantic validation with retry

Usage:
    # Version 1 (default) - Custom JSON constraint
    python tinyllama_json_client.py --prompt "What is 2+2?"

    # Version 2 - Outlines with different schemas
    python tinyllama_json_client.py --version 2 --schema qa --prompt "What is the capital of France?"
    python tinyllama_json_client.py --version 2 --schema sentiment --prompt "I love this product!"

    # Version 3 - Guidance templates
    python tinyllama_json_client.py --version 3 --schema chain_of_thought --prompt "Why is the sky blue?"

    # Version 4 - LMQL queries
    python tinyllama_json_client.py --version 4 --schema scripted --prompt "What is 15 + 27?"

    # Version 5 - Jsonformer
    python tinyllama_json_client.py --version 5 --schema person --prompt "Tell me about Marie Curie"

    # Version 6 - Instructor
    python tinyllama_json_client.py --version 6 --schema extraction --prompt "Apple was founded by Steve Jobs"

    # Demo all versions
    python tinyllama_json_client.py --demo-all
"""

import argparse
import json
import os
import sys
import numpy as np

from auth_helper import get_auth_headers, merge_auth_headers

# Model configuration
MODEL_NAME = "tinyllama-python"

# Version information
VERSION_INFO = {
    "1": {
        "name": "Custom LogitsProcessor",
        "description": "Manual JSON constraint using HuggingFace LogitsProcessor",
        "schemas": None,  # Uses response_format instead
        "test_queries": [
            ("What is the capital of France?", None),
            ("What is 2 + 2?", None),
            ("Who wrote Romeo and Juliet?", None),
        ]
    },
    "2": {
        "name": "Outlines",
        "description": "JSON Schema, regex, grammar via token masking",
        "schemas": ["qa", "entity", "sentiment", "regex_phone"],
        "test_queries": [
            ("What is the capital of France?", "qa"),
            ("I absolutely love this new phone, it's amazing!", "sentiment"),
            ("Extract entities: Apple Inc. was founded in Cupertino by Steve Jobs.", "entity"),
            ("My phone number is hidden in this text somewhere.", "regex_phone"),
        ]
    },
    "3": {
        "name": "Guidance",
        "description": "Template DSL with gen/select blocks",
        "schemas": ["qa", "chain_of_thought", "classification"],
        "test_queries": [
            ("What is the speed of light?", "qa"),
            ("Why do leaves change color in autumn?", "chain_of_thought"),
            ("The mitochondria is the powerhouse of the cell.", "classification"),
        ]
    },
    "4": {
        "name": "LMQL",
        "description": "SQL-like query language for constraints",
        "schemas": ["basic", "constrained", "scripted"],
        "test_queries": [
            ("What is the largest ocean?", "basic"),
            ("Explain why the sky is blue.", "constrained"),
            ("What is 15 + 27?", "scripted"),
        ]
    },
    "5": {
        "name": "Jsonformer",
        "description": "Structural JSON generation (lightweight)",
        "schemas": ["qa", "person", "product", "list_items"],
        "test_queries": [
            ("What is the chemical symbol for gold?", "qa"),
            ("Tell me about Albert Einstein.", "person"),
            ("Describe the iPhone 15 Pro.", "product"),
            ("List the planets in our solar system.", "list_items"),
        ]
    },
    "6": {
        "name": "Instructor",
        "description": "Pydantic validation with retry logic",
        "schemas": ["qa", "extraction", "analysis"],
        "test_queries": [
            ("What year did World War II end?", "qa"),
            ("Extract entities: Microsoft was founded by Bill Gates in Seattle.", "extraction"),
            ("Analyze: The service was slow but the food was delicious.", "analysis"),
        ]
    },
}


def load_model_versions(rest_url: str, versions: list = None) -> bool:
    """Load model via REST API with optional version override."""
    import requests

    # Remove trailing slash
    base_url = rest_url.rstrip("/")

    # Build headers with auth (token > DOMINO_API_PROXY > api_key)
    headers = merge_auth_headers({"Content-Type": "application/json"})

    url = f"{base_url}/v2/repository/models/{MODEL_NAME}/load"
    version_str = str(versions) if versions else "default (v1)"
    print(f"Loading {MODEL_NAME} (versions: {version_str})...")

    # Build request payload
    # For version 1 or no version specified, use simple load (config.pbtxt default)
    # For other versions, use protobuf text format config override
    if versions and versions != ["1"] and versions != [1]:
        # Use protobuf text format (not JSON!) for config override
        version_list = ", ".join(str(v) for v in versions)
        config_override = f"version_policy {{ specific {{ versions: [{version_list}] }} }}"
        payload = {
            "parameters": {
                "config": config_override
            }
        }
    else:
        payload = {}

    try:
        if payload:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
        else:
            response = requests.post(url, headers=headers, timeout=120)

        if response.status_code == 200:
            print(f"Successfully loaded model")
            return True
        else:
            print(f"Failed to load model: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def unload_model(rest_url: str) -> bool:
    """Unload model via REST API."""
    import requests

    base_url = rest_url.rstrip("/")

    # Build headers with auth (token > DOMINO_API_PROXY > api_key)
    headers = merge_auth_headers({"Content-Type": "application/json"})

    url = f"{base_url}/v2/repository/models/{MODEL_NAME}/unload"
    print(f"Unloading {MODEL_NAME}...")

    try:
        response = requests.post(url, headers=headers, timeout=30)
        if response.status_code == 200:
            print(f"Successfully unloaded {MODEL_NAME}")
            return True
        else:
            print(f"Failed to unload model: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error unloading model: {e}")
        return False


def check_model_ready(rest_url: str, version: str, timeout: int = 60) -> bool:
    """Wait for a model version to be ready."""
    import requests
    import time

    base_url = rest_url.rstrip("/")
    if base_url.startswith("http://"):
        base_url = base_url[7:]
    elif base_url.startswith("https://"):
        base_url = base_url[8:]

    # Use tritonclient to check readiness
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=base_url)

    start = time.time()
    while time.time() - start < timeout:
        try:
            if client.is_model_ready(MODEL_NAME, version):
                return True
        except Exception:
            pass
        time.sleep(1)

    return False


def test_all_versions(rest_url: str, grpc_url: str = None, verbose: bool = True):
    """Load all versions one at a time, test each one, then unload."""
    import time

    all_versions = list(VERSION_INFO.keys())

    print()
    print("#" * 70)
    print("#  TinyLlama - Test All Versions (Load, Invoke, Unload)")
    print("#" * 70)
    print()

    # Determine which client to use for inference
    # Force REST for testing since gRPC proxy may have version routing issues
    protocol = "REST"
    inference_fn = lambda prompt, version, schema, use_json: run_rest_inference(
        rest_url, prompt, version, schema, use_json, 0.1
    )

    print(f"REST URL (load/unload): {rest_url}")
    print(f"Inference protocol: {protocol}")
    print()

    results = {}

    # Test each version individually: unload -> load specific version -> test -> repeat
    for version in all_versions:
        info = VERSION_INFO[version]
        print("=" * 70)
        print(f"Testing Version {version}: {info['name']}")
        print(f"Description: {info['description']}")
        print("=" * 70)

        # Step 1: Unload any existing model
        print("\n  [1/4] Unloading existing model...")
        unload_model(rest_url)
        time.sleep(1)

        # Step 2: Load this specific version
        print(f"  [2/4] Loading version {version}...")
        if not load_model_versions(rest_url, [version]):
            print(f"  SKIP: Failed to load version {version}")
            results[version] = {"success": False, "error": "Failed to load", "skipped": True}
            print()
            continue

        # Step 3: Wait for version to be ready
        print(f"  [3/4] Waiting for version {version} to be ready...")
        if not check_model_ready(rest_url, version, timeout=120):
            print(f"  SKIP: Version {version} not ready (timeout)")
            results[version] = {"success": False, "error": "Not ready (timeout)", "skipped": True}
            print()
            continue

        print(f"  Version {version}: Ready")

        # Step 4: Run inference test
        print(f"  [4/4] Running inference test...")
        test_prompt, test_schema = info['test_queries'][0]

        # Show inputs
        print(f"\n  --- INPUTS ---")
        print(f"  prompt: \"{test_prompt}\"")
        print(f"  temperature: 0.1")
        print(f"  model_version: \"{version}\"")
        if version == "1":
            print(f"  response_format: \"json\"")
        else:
            print(f"  schema_name: \"{test_schema}\"")

        try:
            start = time.time()
            if version == "1":
                response, tokens = inference_fn(test_prompt, version, None, True)
            else:
                response, tokens = inference_fn(test_prompt, version, test_schema, True)
            elapsed = time.time() - start

            # Show outputs
            print(f"\n  --- OUTPUTS ---")
            print(f"  generated_text: {response[:300]}..." if len(response) > 300 else f"  generated_text: {response}")
            print(f"  token_count: {tokens}")
            print(f"  inference_time: {elapsed*1000:.0f}ms")
            results[version] = {"success": True, "tokens": tokens, "time_ms": elapsed*1000}

        except Exception as e:
            print(f"  ERROR: {e}")
            results[version] = {"success": False, "error": str(e)}

        print()

    # Final cleanup: unload model
    print("=" * 70)
    print("CLEANUP: Unloading model")
    print("=" * 70)
    unload_model(rest_url)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results.values() if r.get("success"))
    skipped = sum(1 for r in results.values() if r.get("skipped"))
    failed = len(all_versions) - successful - skipped
    print(f"Versions tested: {len(all_versions)}")
    print(f"Successful: {successful}")
    print(f"Skipped (load failed): {skipped}")
    print(f"Failed (inference error): {failed}")
    print()
    for version, result in sorted(results.items()):
        info = VERSION_INFO[version]
        if result.get("success"):
            status = f"OK ({result.get('time_ms', 0):.0f}ms, {result.get('tokens', 0)} tokens)"
        elif result.get("skipped"):
            status = f"SKIPPED: {result.get('error', 'unknown')}"
        else:
            status = f"FAIL: {result.get('error', 'unknown')}"
        print(f"  v{version} ({info['name']}): {status}")
    print()


def run_grpc_inference(
    url: str,
    prompt: str,
    version: str = "1",
    schema_name: str = None,
    use_json: bool = True,
    temperature: float = 0.1
):
    """Run inference using gRPC client."""
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=url)

    # Check if model is ready
    if not client.is_model_ready(MODEL_NAME, version):
        print(f"Model {MODEL_NAME} v{version} is not ready")
        sys.exit(1)

    # Prepare inputs with batch dimension [batch=1, element=1]
    # Required when max_batch_size > 0 in config.pbtxt
    prompt_data = np.array([[prompt]], dtype=object)
    temperature_data = np.array([[temperature]], dtype=np.float32)

    inputs = [
        grpcclient.InferInput("prompt", prompt_data.shape, "BYTES"),
        grpcclient.InferInput("temperature", temperature_data.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(prompt_data)
    inputs[1].set_data_from_numpy(temperature_data)

    # Version 1 uses response_format, versions 2-6 use schema_name
    if version == "1":
        if use_json:
            response_format_data = np.array([["json"]], dtype=object)
            response_format_input = grpcclient.InferInput("response_format", response_format_data.shape, "BYTES")
            response_format_input.set_data_from_numpy(response_format_data)
            inputs.append(response_format_input)
    else:
        if schema_name:
            schema_data = np.array([[schema_name]], dtype=object)
            schema_input = grpcclient.InferInput("schema_name", schema_data.shape, "BYTES")
            schema_input.set_data_from_numpy(schema_data)
            inputs.append(schema_input)

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    # Run inference
    result = client.infer(MODEL_NAME, inputs, model_version=version, headers=headers)

    # Output has batch dimension [batch, 1], flatten to get scalar
    generated_text = result.as_numpy("generated_text").flatten()[0]
    if isinstance(generated_text, bytes):
        generated_text = generated_text.decode("utf-8")

    token_count = result.as_numpy("token_count").flatten()[0]

    return generated_text, token_count


def run_rest_inference(
    url: str,
    prompt: str,
    version: str = "1",
    schema_name: str = None,
    use_json: bool = True,
    temperature: float = 0.1
):
    """Run inference using REST client."""
    import tritonclient.http as httpclient

    # Remove http:// prefix if present for tritonclient
    if url.startswith("http://"):
        url = url[7:]
    elif url.startswith("https://"):
        url = url[8:]

    client = httpclient.InferenceServerClient(url=url)

    # Check if model is ready
    if not client.is_model_ready(MODEL_NAME, version):
        print(f"Model {MODEL_NAME} v{version} is not ready")
        sys.exit(1)

    # Prepare inputs with batch dimension [batch=1, element=1]
    # Required when max_batch_size > 0 in config.pbtxt
    prompt_data = np.array([[prompt]], dtype=object)
    temperature_data = np.array([[temperature]], dtype=np.float32)

    inputs = [
        httpclient.InferInput("prompt", prompt_data.shape, "BYTES"),
        httpclient.InferInput("temperature", temperature_data.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(prompt_data)
    inputs[1].set_data_from_numpy(temperature_data)

    # Version 1 uses response_format, versions 2-6 use schema_name
    if version == "1":
        if use_json:
            response_format_data = np.array([["json"]], dtype=object)
            response_format_input = httpclient.InferInput("response_format", response_format_data.shape, "BYTES")
            response_format_input.set_data_from_numpy(response_format_data)
            inputs.append(response_format_input)
    else:
        if schema_name:
            schema_data = np.array([[schema_name]], dtype=object)
            schema_input = httpclient.InferInput("schema_name", schema_data.shape, "BYTES")
            schema_input.set_data_from_numpy(schema_data)
            inputs.append(schema_input)

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    # Run inference
    result = client.infer(MODEL_NAME, inputs, model_version=version, headers=headers)

    # Output has batch dimension [batch, 1], flatten to get scalar
    generated_text = result.as_numpy("generated_text").flatten()[0]
    if isinstance(generated_text, bytes):
        generated_text = generated_text.decode("utf-8")

    token_count = result.as_numpy("token_count").flatten()[0]

    return generated_text, token_count


def print_response(response: str, tokens: int, indent: bool = True):
    """Pretty print a JSON response."""
    try:
        parsed = json.loads(response)
        if indent:
            print(json.dumps(parsed, indent=2))
        else:
            print(json.dumps(parsed))
    except json.JSONDecodeError:
        print(response)
    print(f"   (tokens: {tokens})")


def run_version_demo(inference_fn, version: str, verbose: bool = True):
    """Run demo queries for a specific version."""
    info = VERSION_INFO[version]

    print("=" * 70)
    print(f"Version {version}: {info['name']}")
    print(f"Description: {info['description']}")
    if info['schemas']:
        print(f"Available schemas: {', '.join(info['schemas'])}")
    print("=" * 70)
    print()

    for prompt, schema in info['test_queries']:
        schema_str = f" [schema: {schema}]" if schema else " [response_format: json]"
        print(f"Q: {prompt}{schema_str}")
        try:
            if version == "1":
                response, tokens = inference_fn(prompt, version, None, True)
            else:
                response, tokens = inference_fn(prompt, version, schema, True)

            print("A: ", end="")
            print_response(response, tokens, indent=verbose)
        except Exception as e:
            print(f"Error: {e}")
        print()


def run_all_demos(inference_fn, verbose: bool = True):
    """Run demos for all versions."""
    print()
    print("#" * 70)
    print("#  TinyLlama Constrained Decoding - All Versions Demo")
    print("#" * 70)
    print()

    for version in sorted(VERSION_INFO.keys()):
        try:
            run_version_demo(inference_fn, version, verbose)
        except Exception as e:
            print(f"Version {version} failed: {e}")
            print()


def test_batch_inference(rest_url: str, num_requests: int = 8, verbose: bool = True):
    """
    Test batch inference by sending multiple concurrent requests.

    Triton's dynamic batching will group these requests together.
    """
    import concurrent.futures
    import threading
    import time
    import tritonclient.http as httpclient

    # Remove http:// prefix if present
    url = rest_url
    if url.startswith("http://"):
        url = url[7:]
    elif url.startswith("https://"):
        url = url[8:]

    # Test prompts for batch
    test_prompts = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet?",
        "What is the speed of light?",
        "What is H2O?",
        "When did World War II end?",
        "What is the capital of Japan?",
        "What is 10 * 5?",
        "Who invented the telephone?",
    ]

    # Use only the requested number of prompts
    prompts = test_prompts[:num_requests]

    print()
    print("#" * 70)
    print(f"#  Batch Inference Test - {num_requests} Concurrent Requests")
    print("#" * 70)
    print()
    print(f"REST URL: {rest_url}")
    print(f"Model: {MODEL_NAME} v1")
    print(f"Requests: {num_requests}")
    print()

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    results = {}
    lock = threading.Lock()

    def send_request(idx: int, prompt: str):
        """Send a single inference request."""
        thread_id = threading.current_thread().name
        start_time = time.time()

        try:
            client = httpclient.InferenceServerClient(url=url)

            # Prepare inputs with batch dimension [batch=1, element=1]
            prompt_data = np.array([[prompt]], dtype=object)
            temperature_data = np.array([[0.1]], dtype=np.float32)
            response_format_data = np.array([["json"]], dtype=object)

            inputs = [
                httpclient.InferInput("prompt", prompt_data.shape, "BYTES"),
                httpclient.InferInput("temperature", temperature_data.shape, "FP32"),
                httpclient.InferInput("response_format", response_format_data.shape, "BYTES"),
            ]
            inputs[0].set_data_from_numpy(prompt_data)
            inputs[1].set_data_from_numpy(temperature_data)
            inputs[2].set_data_from_numpy(response_format_data)

            # Send request
            result = client.infer(MODEL_NAME, inputs, model_version="1", headers=headers)

            elapsed = time.time() - start_time

            # Output has batch dimension [batch, 1], flatten to get scalar
            generated_text = result.as_numpy("generated_text").flatten()[0]
            if isinstance(generated_text, bytes):
                generated_text = generated_text.decode("utf-8")
            token_count = result.as_numpy("token_count").flatten()[0]

            with lock:
                results[idx] = {
                    "prompt": prompt,
                    "response": generated_text,
                    "tokens": token_count,
                    "time_ms": elapsed * 1000,
                    "thread": thread_id,
                    "success": True,
                }

        except Exception as e:
            elapsed = time.time() - start_time
            with lock:
                results[idx] = {
                    "prompt": prompt,
                    "error": str(e),
                    "time_ms": elapsed * 1000,
                    "thread": thread_id,
                    "success": False,
                }

    # Send all requests concurrently
    print("=" * 70)
    print("Sending requests concurrently...")
    print("=" * 70)

    overall_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for idx, prompt in enumerate(prompts):
            future = executor.submit(send_request, idx, prompt)
            futures.append(future)
            print(f"  [{idx+1}] Submitted: {prompt[:50]}...")

        # Wait for all to complete
        concurrent.futures.wait(futures)

    overall_elapsed = time.time() - overall_start

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    successful = 0
    total_tokens = 0
    total_time = 0

    for idx in sorted(results.keys()):
        r = results[idx]
        if r["success"]:
            successful += 1
            total_tokens += r["tokens"]
            total_time += r["time_ms"]
            if verbose:
                print(f"\n[{idx+1}] {r['prompt'][:40]}...")
                print(f"    Response: {r['response'][:60]}...")
                print(f"    Tokens: {r['tokens']}, Time: {r['time_ms']:.0f}ms")
        else:
            print(f"\n[{idx+1}] {r['prompt'][:40]}...")
            print(f"    ERROR: {r['error']}")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total requests:     {num_requests}")
    print(f"Successful:         {successful}")
    print(f"Failed:             {num_requests - successful}")
    print(f"Total tokens:       {total_tokens}")
    print(f"Overall time:       {overall_elapsed*1000:.0f}ms")
    if successful > 0:
        print(f"Avg time/request:   {total_time/successful:.0f}ms")
        print(f"Throughput:         {successful/overall_elapsed:.2f} req/s")
    print()

    # Compare with sequential baseline
    if successful == num_requests and successful > 0:
        sequential_estimate = total_time
        print(f"Sequential estimate: {sequential_estimate:.0f}ms (sum of individual times)")
        print(f"Batch speedup:       {sequential_estimate/(overall_elapsed*1000):.2f}x")
        print()


def list_versions():
    """Print information about all available versions."""
    print()
    print("Available Versions:")
    print("-" * 70)
    for version, info in sorted(VERSION_INFO.items()):
        print(f"  Version {version}: {info['name']}")
        print(f"             {info['description']}")
        if info['schemas']:
            print(f"             Schemas: {', '.join(info['schemas'])}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="TinyLlama Constrained Decoding Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Version 1 - Custom LogitsProcessor (default)
  python tinyllama_json_client.py --prompt "What is 2+2?"
  python tinyllama_json_client.py --prompt "Explain AI" --no-json

  # Version 2 - Outlines (JSON Schema, regex)
  python tinyllama_json_client.py -v 2 -s qa --prompt "What is the capital of France?"
  python tinyllama_json_client.py -v 2 -s sentiment --prompt "I love this product!"
  python tinyllama_json_client.py -v 2 -s entity --prompt "Apple was founded by Steve Jobs"

  # Version 3 - Guidance (Template DSL)
  python tinyllama_json_client.py -v 3 -s qa --prompt "What is Python?"
  python tinyllama_json_client.py -v 3 -s chain_of_thought --prompt "Why is the sky blue?"
  python tinyllama_json_client.py -v 3 -s classification --prompt "E=mc^2 is Einstein's equation"

  # Version 4 - LMQL (Query language)
  python tinyllama_json_client.py -v 4 -s basic --prompt "What is the largest planet?"
  python tinyllama_json_client.py -v 4 -s constrained --prompt "Explain photosynthesis"
  python tinyllama_json_client.py -v 4 -s scripted --prompt "What is 15 + 27?"

  # Version 5 - Jsonformer (Structural JSON)
  python tinyllama_json_client.py -v 5 -s qa --prompt "What is H2O?"
  python tinyllama_json_client.py -v 5 -s person --prompt "Tell me about Marie Curie"
  python tinyllama_json_client.py -v 5 -s product --prompt "Describe a Tesla Model 3"

  # Version 6 - Instructor (Pydantic validation)
  python tinyllama_json_client.py -v 6 -s qa --prompt "When did WWII end?"
  python tinyllama_json_client.py -v 6 -s extraction --prompt "Microsoft was founded by Bill Gates"
  python tinyllama_json_client.py -v 6 -s analysis --prompt "The food was great but service was slow"

  # Demo modes
  python tinyllama_json_client.py --demo 2          # Demo version 2 only
  python tinyllama_json_client.py --demo-all        # Demo all versions
  python tinyllama_json_client.py --list-versions   # Show all versions and schemas

  # Test all versions (load, invoke, unload)
  python tinyllama_json_client.py --test-all-versions  # Load all versions, test each, unload

  # Test batch inference (concurrent requests)
  python tinyllama_json_client.py --test-batch         # 8 concurrent requests (default)
  python tinyllama_json_client.py --test-batch 4       # 4 concurrent requests

  # Compare modes
  python tinyllama_json_client.py --compare         # Compare JSON vs free-form (v1)
  python tinyllama_json_client.py --compare-versions 1 2 3  # Compare versions
        """
    )

    # Connection options
    parser.add_argument(
        "--rest-url",
        type=str,
        default=os.environ.get("TRITON_REST_URL", "http://localhost:8080"),
        help="REST API URL (default: $TRITON_REST_URL or http://localhost:8080)"
    )
    parser.add_argument(
        "--grpc-url",
        type=str,
        default=os.environ.get("TRITON_GRPC_URL", ""),
        help="gRPC URL. If set, uses gRPC instead of REST."
    )

    # Version and schema selection
    parser.add_argument(
        "-v", "--version",
        type=str,
        default="1",
        choices=["1", "2", "3", "4", "5", "6"],
        help="Model version (1-6, default: 1)"
    )
    parser.add_argument(
        "-s", "--schema",
        type=str,
        default=None,
        help="Schema/template name for versions 2-6"
    )

    # Query options
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON constraint (version 1 only)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling (default: 0.1)"
    )

    # Demo and comparison modes
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        metavar="VERSION",
        help="Run demo queries for a specific version"
    )
    parser.add_argument(
        "--demo-all",
        action="store_true",
        help="Run demo queries for all versions"
    )
    parser.add_argument(
        "--test-all-versions",
        action="store_true",
        help="Load all versions, test each one, then unload (full lifecycle test)"
    )
    parser.add_argument(
        "--test-batch",
        type=int,
        nargs="?",
        const=8,
        default=None,
        metavar="N",
        help="Test batch inference with N concurrent requests (default: 8, version 1 only)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare JSON vs free-form output (version 1)"
    )
    parser.add_argument(
        "--compare-versions",
        type=str,
        nargs="+",
        metavar="V",
        help="Compare same query across multiple versions"
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List all available versions and schemas"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON results (for dashboard integration)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )

    args = parser.parse_args()

    # Handle --list-versions
    if args.list_versions:
        list_versions()
        return

    # Handle --test-all-versions
    if args.test_all_versions:
        test_all_versions(args.rest_url, args.grpc_url, args.verbose)
        return

    # Handle --test-batch
    if args.test_batch is not None:
        test_batch_inference(args.rest_url, args.test_batch, args.verbose)
        return

    # Set default schema for versions 2-6 if not specified
    if args.version != "1" and args.schema is None:
        args.schema = VERSION_INFO[args.version]["schemas"][0] if VERSION_INFO[args.version]["schemas"] else None

    # Determine which client to use
    if args.grpc_url:
        print(f"Using gRPC client: {args.grpc_url}")
        inference_fn = lambda prompt, version, schema, use_json: run_grpc_inference(
            args.grpc_url, prompt, version, schema, use_json, args.temperature
        )
    else:
        print(f"Using REST client: {args.rest_url}")
        inference_fn = lambda prompt, version, schema, use_json: run_rest_inference(
            args.rest_url, prompt, version, schema, use_json, args.temperature
        )

    print()

    # Handle demo modes
    if args.demo_all:
        run_all_demos(inference_fn, args.verbose)
        return

    if args.demo:
        if args.demo not in VERSION_INFO:
            print(f"Invalid version: {args.demo}. Use 1-6.")
            return
        run_version_demo(inference_fn, args.demo, args.verbose)
        return

    # Handle comparison modes
    if args.compare:
        print("COMPARISON MODE: JSON vs Free-form (Version 1)")
        print()
        test_queries = [
            "What is the capital of Japan?",
            "What is 10 * 5?",
        ]
        for query in test_queries:
            print(f"Q: {query}")
            print("-" * 50)
            response_json, _ = inference_fn(query, "1", None, True)
            print(f"JSON mode:      {response_json}")
            response_free, _ = inference_fn(query, "1", None, False)
            print(f"Free-form mode: {response_free}")
            print()
        return

    if args.compare_versions:
        print(f"COMPARISON MODE: Versions {', '.join(args.compare_versions)}")
        print()
        test_prompt = args.prompt or "What is the capital of France?"
        print(f"Q: {test_prompt}")
        print("-" * 50)
        for v in args.compare_versions:
            if v not in VERSION_INFO:
                print(f"Version {v}: Invalid version")
                continue
            info = VERSION_INFO[v]
            schema = info["schemas"][0] if info["schemas"] else None
            try:
                if v == "1":
                    response, tokens = inference_fn(test_prompt, v, None, True)
                else:
                    response, tokens = inference_fn(test_prompt, v, schema, True)
                schema_str = f" ({schema})" if schema else ""
                print(f"Version {v} [{info['name']}{schema_str}]:")
                print(f"  {response}")
            except Exception as e:
                print(f"Version {v}: Error - {e}")
        print()
        return

    # Single prompt mode
    if args.prompt:
        use_json = not args.no_json
        version = args.version
        schema = args.schema

        info = VERSION_INFO[version]
        print(f"Model: {MODEL_NAME} v{version} ({info['name']})")
        if schema:
            print(f"Schema: {schema}")
        print(f"Prompt: {args.prompt}")
        print("-" * 50)

        import time
        start_time = time.time()

        try:
            if version == "1":
                response, tokens = inference_fn(args.prompt, version, None, use_json)
            else:
                response, tokens = inference_fn(args.prompt, version, schema, use_json)

            elapsed_time = time.time() - start_time

            print("Response:")
            print_response(response, tokens, args.verbose)

            # Save to output file if specified (for dashboard integration)
            if args.output:
                output_data = {
                    "results": [
                        {
                            "prompt": args.prompt,
                            "generated_text": response,
                            "token_count": tokens,
                            "version": version,
                            "schema": schema,
                            "inference_ms": elapsed_time * 1000,
                        }
                    ],
                    "stats": {
                        "total_time_sec": elapsed_time,
                        "version": version,
                        "library": info["name"],
                    }
                }
                from pathlib import Path
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error: {e}")
            # Still save error to output file if specified
            if args.output:
                output_data = {
                    "results": [],
                    "error": str(e),
                    "stats": {"version": version}
                }
                from pathlib import Path
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=2)
        return

    # Default: run demo for selected version
    run_version_demo(inference_fn, args.version, args.verbose)


if __name__ == "__main__":
    main()
