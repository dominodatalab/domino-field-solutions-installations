#!/usr/bin/env python3
"""
BERT Text Inference Client (REST)

Uses standard tritonclient.http library for Triton inference.
Supports batching multiple texts for improved throughput.

Usage:
    python bert_text_rest_client.py --texts "I love this movie" "This is terrible"
    python bert_text_rest_client.py --texts-file inputs.txt --batch-size 8
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

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


def process_batch(client, headers, model, batch_texts, batch_indices, tokenizer, results, times, use_binary=True):
    """Process a batch of texts."""
    if not batch_texts:
        return

    # Tokenize batch
    tokens = tokenize_texts(tokenizer, batch_texts)

    # Build input tensors
    input_ids = httpclient.InferInput("input_ids", tokens["input_ids"].shape, "INT64")
    input_ids.set_data_from_numpy(tokens["input_ids"], binary_data=use_binary)

    attention_mask = httpclient.InferInput("attention_mask", tokens["attention_mask"].shape, "INT64")
    attention_mask.set_data_from_numpy(tokens["attention_mask"], binary_data=use_binary)

    # Build output request
    output_tensor = httpclient.InferRequestedOutput("logits", binary_data=use_binary)

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


def main():
    parser = argparse.ArgumentParser(description="BERT Text Inference (REST)")
    parser.add_argument("--texts", "-t", nargs="+", help="Input texts to process")
    parser.add_argument("--texts-file", help="File containing texts (one per line)")
    parser.add_argument("--rest-url", "-u",
                        default=os.environ.get("TRITON_REST_URL", "http://localhost:8080"),
                        help="REST URL (env: TRITON_REST_URL)")
    parser.add_argument("--model", "-m", default="bert-base-uncased", help="Model name")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--output", "-o", default=str(RESULTS_DIR / "bert_rest.json"), help=f"Output JSON file (default: {RESULTS_DIR}/bert_rest.json)")
    parser.add_argument("--json-encoding", action="store_true", help="Use JSON arrays instead of binary")
    args = parser.parse_args()

    use_binary = not args.json_encoding

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

    # Create Triton client (strip http:// prefix for tritonclient)
    url = args.rest_url.replace("http://", "").replace("https://", "").rstrip("/")
    client = httpclient.InferenceServerClient(url=url)

    # Build auth headers
    api_key = os.environ.get("DOMINO_USER_API_KEY")
    headers = {"X-Domino-Api-Key": api_key} if api_key else None

    results = {"model": args.model, "batch_size": args.batch_size, "texts": [], "stats": {}}
    times = []

    logger.info(f"Starting inference via REST: {args.rest_url}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Encoding: {'JSON arrays' if args.json_encoding else 'Binary'}")
    logger.info("-" * 60)

    batch_texts = []
    batch_indices = []

    for idx, text in enumerate(texts):
        batch_texts.append(text)
        batch_indices.append(idx)

        # Process when batch is full
        if len(batch_texts) >= args.batch_size:
            process_batch(client, headers, args.model, batch_texts, batch_indices, tokenizer, results, times, use_binary)
            batch_texts = []
            batch_indices = []

    # Process remaining texts
    if batch_texts:
        process_batch(client, headers, args.model, batch_texts, batch_indices, tokenizer, results, times, use_binary)

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


if __name__ == "__main__":
    main()
