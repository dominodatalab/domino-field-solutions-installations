#!/usr/bin/env python3
"""
BERT Text Inference Client (gRPC)

Uses standard tritonclient.grpc library for Triton inference.
Supports batching multiple texts for improved throughput and async mode.

Usage:
    python bert_text_grpc_client.py --texts "I love this movie" "This is terrible"
    python bert_text_grpc_client.py --texts-file inputs.txt --batch-size 8
    python bert_text_grpc_client.py --texts "Great!" --async  # Async mode
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from auth_helper import get_auth_headers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directories
SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR.parent.parent / "results" / "bert"

MAX_SEQ_LENGTH = 128


def get_tokenizer():
    """Load the BERT tokenizer."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    except ImportError:
        logger.error("transformers library not installed. Run: pip install transformers")
        raise SystemExit(1)


def tokenize_texts(tokenizer, texts: List[str], max_length: int = MAX_SEQ_LENGTH) -> dict:
    """Tokenize multiple texts into batched tensors."""
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64)
    }


def process_batch(client, headers, model, batch_texts, batch_indices, tokenizer, results, times):
    """Process a batch of texts (sync version)."""
    if not batch_texts:
        return

    # Tokenize batch
    tokens = tokenize_texts(tokenizer, batch_texts)

    # Build input tensors
    input_ids = grpcclient.InferInput("input_ids", tokens["input_ids"].shape, "INT64")
    input_ids.set_data_from_numpy(tokens["input_ids"])

    attention_mask = grpcclient.InferInput("attention_mask", tokens["attention_mask"].shape, "INT64")
    attention_mask.set_data_from_numpy(tokens["attention_mask"])

    # Build output request
    output_tensor = grpcclient.InferRequestedOutput("logits")

    start = time.time()
    try:
        response = client.infer(
            model_name=model,
            inputs=[input_ids, attention_mask],
            outputs=[output_tensor],
            headers=headers,
        )
        inference_time = time.time() - start
        per_text_time = inference_time / len(batch_texts)

        # Get output as numpy
        logits = response.as_numpy("logits")

        # Class labels for sentiment classification
        class_labels = ["NEGATIVE", "POSITIVE"]

        for i, (idx, text) in enumerate(zip(batch_indices, batch_texts)):
            item_logits = logits[i] if logits is not None else None

            # Compute classification from logits
            classification = None
            confidence = None
            if item_logits is not None:
                # Softmax to get probabilities
                exp_logits = np.exp(item_logits - np.max(item_logits))
                probs = exp_logits / exp_logits.sum()
                predicted_class = int(np.argmax(probs))
                confidence = float(probs[predicted_class])
                classification = class_labels[predicted_class] if predicted_class < len(class_labels) else f"CLASS_{predicted_class}"

            results["texts"].append({
                "index": idx,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "classification": classification,
                "confidence": round(confidence, 4) if confidence else None,
                "batch_size": len(batch_texts),
                "inference_ms": round(per_text_time * 1000, 2),
                "output_shape": list(logits[i:i+1].shape) if logits is not None else None
            })
        times.append(inference_time)

        logger.info(
            f"Batch [{batch_indices[0]}..{batch_indices[-1]}]: "
            f"inference={inference_time*1000:6.1f}ms ({per_text_time*1000:.1f}ms/text)"
        )

    except InferenceServerException as e:
        logger.error(f"Batch error: {e}")
        for idx, text in zip(batch_indices, batch_texts):
            results["texts"].append({"index": idx, "error": str(e)})


# ==================== Async Implementation ====================


async def process_batch_async(client, headers, model, batch_texts, batch_indices, tokenizer):
    """Process a batch of texts (async version). Returns (results, inference_time)."""
    import tritonclient.grpc.aio as grpcclient_aio

    if not batch_texts:
        return [], 0

    # Tokenize batch
    tokens = tokenize_texts(tokenizer, batch_texts)

    # Build input tensors
    input_ids = grpcclient_aio.InferInput("input_ids", tokens["input_ids"].shape, "INT64")
    input_ids.set_data_from_numpy(tokens["input_ids"])

    attention_mask = grpcclient_aio.InferInput("attention_mask", tokens["attention_mask"].shape, "INT64")
    attention_mask.set_data_from_numpy(tokens["attention_mask"])

    # Build output request
    output_tensor = grpcclient_aio.InferRequestedOutput("logits")

    start = time.time()
    try:
        response = await client.infer(
            model_name=model,
            inputs=[input_ids, attention_mask],
            outputs=[output_tensor],
            headers=headers,
        )
        inference_time = time.time() - start
        per_text_time = inference_time / len(batch_texts)

        logits = response.as_numpy("logits")
        class_labels = ["NEGATIVE", "POSITIVE"]

        batch_results = []
        for i, (idx, text) in enumerate(zip(batch_indices, batch_texts)):
            item_logits = logits[i] if logits is not None else None

            classification = None
            confidence = None
            if item_logits is not None:
                exp_logits = np.exp(item_logits - np.max(item_logits))
                probs = exp_logits / exp_logits.sum()
                predicted_class = int(np.argmax(probs))
                confidence = float(probs[predicted_class])
                classification = class_labels[predicted_class] if predicted_class < len(class_labels) else f"CLASS_{predicted_class}"

            batch_results.append({
                "index": idx,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "classification": classification,
                "confidence": round(confidence, 4) if confidence else None,
                "batch_size": len(batch_texts),
                "inference_ms": round(per_text_time * 1000, 2),
                "output_shape": list(logits[i:i+1].shape) if logits is not None else None
            })

        logger.info(
            f"Batch [{batch_indices[0]}..{batch_indices[-1]}]: "
            f"inference={inference_time*1000:6.1f}ms ({per_text_time*1000:.1f}ms/text)"
        )

        return batch_results, inference_time

    except Exception as e:
        logger.error(f"Batch error: {e}")
        return [{"index": idx, "error": str(e)} for idx in batch_indices], 0


async def run_async(args, texts, tokenizer):
    """Run inference in async mode with concurrent batch processing."""
    import tritonclient.grpc.aio as grpcclient_aio

    logger.info("Running in ASYNC mode")

    # Create async Triton client
    client = grpcclient_aio.InferenceServerClient(url=args.grpc_url)

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    results = {"model": args.model, "batch_size": args.batch_size, "async_mode": True, "texts": [], "stats": {}}

    logger.info(f"Starting async inference via gRPC: {args.grpc_url}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("-" * 60)

    # Collect all batches
    batches = []
    batch_texts = []
    batch_indices = []

    for idx, text in enumerate(texts):
        batch_texts.append(text)
        batch_indices.append(idx)

        if len(batch_texts) >= args.batch_size:
            batches.append((batch_texts, batch_indices))
            batch_texts = []
            batch_indices = []

    if batch_texts:
        batches.append((batch_texts, batch_indices))

    # Process all batches concurrently
    logger.info(f"Processing {len(batches)} batches concurrently...")

    tasks = [
        process_batch_async(client, headers, args.model, bt, bi, tokenizer)
        for bt, bi in batches
    ]

    batch_results = await asyncio.gather(*tasks)

    # Collect results
    times = []
    for text_results, inference_time in batch_results:
        results["texts"].extend(text_results)
        if inference_time > 0:
            times.append(inference_time)

    # Sort by index
    results["texts"].sort(key=lambda x: x.get("index", 0))

    await client.close()

    # Stats
    total_texts = len(results["texts"])
    if times:
        total_time = sum(times)
        results["stats"] = {
            "total_texts": total_texts,
            "batch_size": args.batch_size,
            "total_batches": len(times),
            "total_time_sec": round(total_time, 2),
            "avg_batch_ms": round(np.mean(times) * 1000, 2),
            "avg_text_ms": round(total_time / total_texts * 1000, 2),
            "throughput": round(total_texts / total_time, 2),
            "async_mode": True
        }
        logger.info("-" * 60)
        logger.info(
            f"Processed {total_texts} texts in {len(times)} batches, "
            f"avg {results['stats']['avg_text_ms']:.1f}ms/text, "
            f"{results['stats']['throughput']:.2f} texts/sec (async)"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


def run_sync(args, texts, tokenizer):
    """Run inference in sync mode (original implementation)."""
    # Create Triton client
    client = grpcclient.InferenceServerClient(url=args.grpc_url)

    # Build auth headers (token > DOMINO_API_PROXY > api_key)
    headers = get_auth_headers()

    results = {"model": args.model, "batch_size": args.batch_size, "async_mode": False, "texts": [], "stats": {}}
    times = []

    logger.info(f"Starting inference via gRPC: {args.grpc_url}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("-" * 60)

    batch_texts = []
    batch_indices = []

    try:
        for idx, text in enumerate(texts):
            batch_texts.append(text)
            batch_indices.append(idx)

            # Process when batch is full
            if len(batch_texts) >= args.batch_size:
                process_batch(client, headers, args.model, batch_texts, batch_indices, tokenizer, results, times)
                batch_texts = []
                batch_indices = []

        # Process remaining texts
        if batch_texts:
            process_batch(client, headers, args.model, batch_texts, batch_indices, tokenizer, results, times)

    finally:
        client.close()

    # Stats
    total_texts = len(results["texts"])
    if times:
        total_time = sum(times)
        results["stats"] = {
            "total_texts": total_texts,
            "batch_size": args.batch_size,
            "total_batches": len(times),
            "total_time_sec": round(total_time, 2),
            "avg_batch_ms": round(np.mean(times) * 1000, 2),
            "avg_text_ms": round(total_time / total_texts * 1000, 2),
            "throughput": round(total_texts / total_time, 2)
        }
        logger.info("-" * 60)
        logger.info(
            f"Processed {total_texts} texts in {len(times)} batches, "
            f"avg {results['stats']['avg_text_ms']:.1f}ms/text, "
            f"{results['stats']['throughput']:.2f} texts/sec"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="BERT Text Inference (gRPC)")
    parser.add_argument("--texts", "-t", nargs="+", help="Input texts to process")
    parser.add_argument("--texts-file", help="File containing texts (one per line)")
    parser.add_argument("--grpc-url", "-u",
                        default=os.environ.get("TRITON_GRPC_URL", "localhost:50051"),
                        help="gRPC URL (env: TRITON_GRPC_URL)")
    parser.add_argument("--model", "-m", default="bert-base-uncased", help="Model name")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--output", "-o", default=str(RESULTS_DIR / "bert_grpc.json"), help=f"Output JSON file (default: {RESULTS_DIR}/bert_grpc.json)")
    parser.add_argument("--async", dest="async_mode", action="store_true", help="Use async mode for concurrent batch processing")
    args = parser.parse_args()

    # Get texts
    if args.texts:
        texts = args.texts
    elif args.texts_file:
        with open(args.texts_file) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [
            "I absolutely loved this movie, it was fantastic!",
            "This product is terrible, complete waste of money.",
            "The weather is nice today.",
            "I'm not sure how I feel about this.",
            "Best experience ever, highly recommend!"
        ]
        logger.info("Using default sample texts")

    tokenizer = get_tokenizer()

    if args.async_mode:
        asyncio.run(run_async(args, texts, tokenizer))
    else:
        run_sync(args, texts, tokenizer)


if __name__ == "__main__":
    main()
