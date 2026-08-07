#!/usr/bin/env python3
"""
quantize_llama4scout.py

Offline quantization of Llama-4-Scout-17B-16E-Instruct weights using llm-compressor.
Produces a compressed-tensors checkpoint that vLLM loads natively (no bitsandbytes
required at runtime).

Quantization schemes
--------------------
  int8 (default): W8A8 via SmoothQuant + QuantizationModifier  (~110 GB on disk)
  int4:           W4A16 via GPTQ                                (~55 GB on disk)

W8A8  — weights and activations in INT8.  SmoothQuant migrates activation outliers
         to the weights first, then both are quantized.  Minimal accuracy loss on
         most benchmarks.

W4A16 — weights in INT4, activations remain FP16.  GPTQ applies layer-wise second-
         order correction.  ~4x smaller than BF16; small but measurable quality gap.

IMPORTANT: this script loads the full BF16 model for calibration, so it requires
the same VRAM as the source weights (~218 GB across GPUs).  Run it on the same
multi-GPU node you intend to use for inference.

Usage
-----
    # INT8 (default), reads from <project-root>/triton-repo/weights/llama4scout-vllm
    python scripts/download/quantize_llama4scout.py

    # INT4
    python scripts/download/quantize_llama4scout.py --quantization int4

    # Custom source and base directory (e.g. S3 FUSE mount)
    python scripts/download/quantize_llama4scout.py \\
        --base-dir /mnt/s3/triton-repo \\
        --quantization int8

    # Override the number of calibration samples (default: 512)
    python scripts/download/quantize_llama4scout.py --num-calibration-samples 256

    # Update the existing Triton model.json to point at the quantized weights
    python scripts/download/quantize_llama4scout.py --update-model-json

Requirements
------------
    pip install llmcompressor transformers datasets

Output (relative to --base-dir, default: <project-root>/triton-repo):
    <base-dir>/weights/llama4scout-vllm-int8/   or   .../llama4scout-vllm-int4/
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SOURCE_MODEL_NAME = "llama4scout-vllm"
APPROX_SOURCE_GB = 200

# Calibration dataset: a small slice of C4 (Common Crawl) is standard for LLM quantization
CALIBRATION_DATASET = "allenai/c4"
CALIBRATION_DATASET_SPLIT = "train"
CALIBRATION_DATASET_FIELD = "text"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline INT8/INT4 quantization of Llama-4-Scout using llm-compressor"
    )
    parser.add_argument(
        "--quantization",
        choices=["int8", "int4"],
        default="int8",
        help=(
            "Quantization scheme (default: int8). "
            "int8 = W8A8 via SmoothQuant (~110 GB output). "
            "int4 = W4A16 via GPTQ (~55 GB output)."
        ),
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Base directory containing weights/ and models/ subdirectories. "
            "Defaults to <project-root>/triton-repo. "
            "Source weights are read from <base-dir>/weights/llama4scout-vllm/."
        ),
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples from C4 (default: 512)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max token length per calibration sample (default: 2048)",
    )
    parser.add_argument(
        "--smoothquant-strength",
        type=float,
        default=0.8,
        help=(
            "SmoothQuant migration strength for INT8 (default: 0.8). "
            "Higher values push more of the quantization difficulty onto weights. "
            "Ignored for INT4."
        ),
    )
    parser.add_argument(
        "--update-model-json",
        action="store_true",
        help=(
            "After quantization, rewrite <base-dir>/models/llama4scout-vllm/1/model.json "
            "to point at the new quantized weights directory."
        ),
    )
    parser.add_argument(
        "--ignore-ssl",
        action="store_true",
        help="Disable SSL certificate verification (corporate proxy environments)",
    )
    return parser.parse_args()


def check_dependencies():
    missing = []
    for pkg in ("llmcompressor", "transformers", "datasets", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def _disable_ssl():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    import requests
    original_request = requests.Session.request
    def patched(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return original_request(self, method, url, **kwargs)
    requests.Session.request = patched


def load_calibration_dataset(tokenizer, num_samples: int, max_seq_length: int):
    from datasets import load_dataset, Dataset

    print(f"\nLoading calibration dataset: {CALIBRATION_DATASET} ({num_samples} samples)")
    ds = load_dataset(
        CALIBRATION_DATASET,
        "en",
        split=CALIBRATION_DATASET_SPLIT,
        streaming=True,
    )

    input_ids_list = []
    attention_mask_list = []
    for row in ds:
        text = row[CALIBRATION_DATASET_FIELD].strip()
        if not text:
            continue
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
        )
        # Only keep samples that actually fill the context window to avoid
        # calibrating on trivially short sequences
        if len(tokenized["input_ids"]) >= max_seq_length // 2:
            input_ids_list.append(tokenized["input_ids"])
            attention_mask_list.append(tokenized["attention_mask"])
        if len(input_ids_list) >= num_samples:
            break

    print(f"  Collected {len(input_ids_list)} calibration samples (min length: {max_seq_length // 2} tokens)")
    # llmcompressor's dataset pipeline expects a real datasets.Dataset (needs
    # .column_names), not a plain list of tokenizer outputs.
    return Dataset.from_dict({"input_ids": input_ids_list, "attention_mask": attention_mask_list})


def build_recipe(quantization: str, smoothquant_strength: float):
    if quantization == "int8":
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

        return [
            SmoothQuantModifier(
                smoothing_strength=smoothquant_strength,
                # SmoothQuantModifier's own 'ignore' param does NOT filter which
                # layers get matched during mapping resolution (confirmed via
                # source: _resolve_mappings() explicitly does not consult
                # self.ignore when calling match_modules_set -- it only decides
                # whether to skip an ALREADY-resolved mapping). The default
                # Llama4ForConditionalGeneration mapping uses unscoped regexes
                # (re:.*q_proj, re:.*post_attention_layernorm, etc.) that match
                # both the text decoder AND the vision encoder's identically-
                # named submodules, causing 'must match a single smooth layer'
                # errors once the full multimodal model loads. Explicit
                # mappings scoped to language_model.* sidestep the ambiguity
                # entirely -- validated via a load-only dry run (96 mapping
                # groups resolved cleanly: 48 layers x 2 mapping types) before
                # committing to a full run. This also matches the recipe's own
                # intent: SmoothQuant should only ever touch what actually gets
                # quantized, and the vision tower never does (see
                # QuantizationModifier's ignore list below).
                mappings=[
                    (
                        ["re:.*language_model.*q_proj", "re:.*language_model.*k_proj", "re:.*language_model.*v_proj"],
                        "re:.*language_model.*input_layernorm",
                    ),
                    (
                        ["re:.*language_model.*gate_proj", "re:.*language_model.*up_proj"],
                        "re:.*language_model.*post_attention_layernorm",
                    ),
                ],
            ),
            QuantizationModifier(
                targets="Linear",
                scheme="W8A8",
                # lm_head operates on vocabulary logits; quantizing it often hurts
                # perplexity more than the size saving is worth. vision_model and
                # multi_modal_projector are kept at full precision to match NVIDIA's
                # official FP8 checkpoint, since image-task accuracy is a requirement.
                #
                # BUG FIXED (found post-hoc, after a full run + serving attempt):
                # bare strings like "lm_head" and "vision_model*" do NOT work as
                # ignore patterns here -- confirmed empirically: the resulting
                # checkpoint had weight_scale tensors on vision_model (207),
                # multi_modal_projector (1), and lm_head (1), meaning NONE of the
                # three intended exclusions actually took effect. Two distinct
                # causes: (1) "lm_head" assumed a top-level path, but on the full
                # multimodal model it lives at language_model.lm_head -- a bare
                # top-level string never matches the nested path; (2) a literal
                # trailing asterisk ("vision_model*") is not valid glob/regex
                # syntax for llmcompressor's ignore matching -- it needs the same
                # "re:" regex-prefix convention already required for
                # SmoothQuantModifier's mappings above. Both are now expressed as
                # proper regexes.
                #
                # self_attn, router, and Llama4TextAttention added after cross-
                # checking against the official reference recipe
                # (vllm-project/llm-compressor examples/multimodal_vision/
                # llama4_example.py) -- it excludes these in addition to the
                # three above. Confirmed via the checkpoint's actual tensor
                # shapes that these categories are small (self_attn ~3.0%,
                # router ~0.0% of total weight bytes) -- ignoring them costs
                # almost no compression while protecting the components most
                # likely to be quantization-sensitive (attention patterns,
                # MoE routing decisions). The reference example uses GPTQ/W4A16,
                # not our SmoothQuant/W8A8, so this is an adopted-by-analogy
                # choice, not a scheme-verified one -- but the asymmetric
                # cost/benefit (near-zero size cost, real quality-risk
                # reduction) supports adopting it regardless.
                ignore=[
                    "re:.*lm_head",
                    "re:.*self_attn",
                    "re:.*router",
                    "re:.*vision_model.*",
                    "re:.*multi_modal_projector.*",
                    "Llama4TextAttention",
                ],
            ),
        ]

    # int4: W4A16 GPTQ
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier

    return GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        # See the INT8 branch above for why these must be regexes with the
        # "re:" prefix, not bare strings -- the same bug applies here.
        # Also matches the official reference recipe's full ignore list
        # (self_attn, router, Llama4TextAttention) -- see INT8 branch comment.
        ignore=[
            "re:.*lm_head",
            "re:.*self_attn",
            "re:.*router",
            "re:.*vision_model.*",
            "re:.*multi_modal_projector.*",
            "Llama4TextAttention",
        ],
        # dampening_frac stabilises the Hessian inverse for large/sparse weight
        # matrices; 0.1 is a safe default for MoE expert layers
        dampening_frac=0.1,
    )


def _patch_llama4_linearize_config_bug():
    # MoE-specific bugfix (llmcompressor==0.12.0): load_context()'s linearize_moe()
    # resolves the config it hands to Llama4LinearExperts.from_experts_module() via
    # `getattr(module, "config", model.config)`. Llama4TextExperts submodules carry
    # their own `.config` attribute already set to the *inner* Llama4TextConfig,
    # which shadows the intended `model.config` (outer Llama4Config) fallback.
    # from_experts_module then does `config.text_config`, which crashes because the
    # config it received already *is* the text config -- confirmed via
    # AttributeError: 'Llama4TextConfig' object has no attribute 'text_config'.
    # This wraps from_experts_module to accept either config shape, so it works
    # regardless of which one linearize_moe happens to pass in. Only needed for
    # this MoE architecture / this llm-compressor version's linearization path --
    # not something a dense (non-MoE) quantization run would ever hit.
    import time
    import types
    from llmcompressor.modeling.moe.llama4 import Llama4LinearExperts

    original = Llama4LinearExperts.from_experts_module.__func__
    # Called once per decoder layer (48 total for Scout) during linearize_moe().
    # No progress is otherwise printed for this CPU-bound step, so a run can
    # look silently hung for a long time -- adding a per-layer counter/timer
    # here (the same call site already patched above) gives real visibility
    # without needing to instrument llm-compressor itself.
    state = {"count": 0, "start": time.time()}

    @classmethod
    def patched(cls, experts, config):
        if not hasattr(config, "text_config"):
            config = types.SimpleNamespace(text_config=config)
        result = original(cls, experts, config)
        state["count"] += 1
        elapsed = time.time() - state["start"]
        print(
            f"  linearize_moe progress: layer {state['count']} done "
            f"({elapsed:.1f}s elapsed, {elapsed / state['count']:.2f}s/layer avg)",
            flush=True,
        )
        return result

    Llama4LinearExperts.from_experts_module = patched


def quantize(
    source_dir: Path,
    output_dir: Path,
    quantization: str,
    num_calibration_samples: int,
    max_seq_length: int,
    smoothquant_strength: float,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from llmcompressor import oneshot
    from llmcompressor.utils import load_context

    _patch_llama4_linearize_config_bug()

    print(f"\n{'='*60}")
    print(f"Source:       {source_dir}")
    print(f"Output:       {output_dir}")
    print(f"Scheme:       {quantization.upper()}")
    print(f"{'='*60}\n")

    if not source_dir.exists():
        print(f"ERROR: source weights not found at {source_dir}")
        print(f"  Run download_llama4scout.py first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(source_dir))

    print("Loading model in BF16 onto CPU (sequential onloading streams layers to GPU one at a time)...")
    print(f"(Requires ~{APPROX_SOURCE_GB} GB system RAM to hold the full model before streaming)\n")
    # Llama4's MoE layers (Llama4TextExperts) pack all 16 experts per layer into
    # a few large batched tensors rather than separate per-expert nn.Linear
    # modules. device_map="auto" treats each such tensor as one atomic,
    # unsplittable unit, so whichever GPU happens to draw the largest MoE block
    # can end up carrying far more than an even per-GPU share -- confirmed via
    # two separate OOM crashes (p4d.24xlarge GPU 0, then g6e.48xlarge GPU 1)
    # where one GPU filled to its hardware limit while total cluster VRAM still
    # had hundreds of GB free.
    #
    # The documented fix (llm-compressor's own Llama4 reference example) is
    # sequential onloading: load the full model onto CPU (device_map=None) and
    # let oneshot()'s default "sequential" pipeline stream one layer at a time
    # onto GPU for calibration, then evict it, rather than statically placing
    # the whole model across GPUs up front. sequential_targets=["Llama4TextMLP"]
    # is Llama4-specific -- it forces the shared-expert MLP to be its own
    # onload/offload boundary; the routed experts are already broken into small
    # per-expert modules by load_context()'s linearization below, so the
    # default sequential tracing handles those without further hints. This
    # trades static multi-GPU VRAM for system RAM (the full model must fit in
    # CPU memory) -- see docs/llama4scout_int8/quantization_plan.md for the
    # sizing analysis behind that tradeoff.
    with load_context():
        model = AutoModelForImageTextToText.from_pretrained(
            str(source_dir),
            torch_dtype=torch.bfloat16,
            device_map=None,
        )

    calibration_data = load_calibration_dataset(tokenizer, num_calibration_samples, max_seq_length)
    recipe = build_recipe(quantization, smoothquant_strength)

    expected_gb = 110 if quantization == "int8" else 55
    print(f"\nRunning {quantization.upper()} calibration and quantization...")
    print(f"  This may take 1–3 hours for a 109B MoE model.")
    print(f"  Expected output size: ~{expected_gb} GB\n")

    oneshot(
        model=model,
        dataset=calibration_data,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
        output_dir=str(output_dir),
        # Llama4-specific sequential onload boundary -- see comment above the
        # model load. Only the shared-expert MLP needs to be named explicitly;
        # the routed experts are already small, individually-linearized modules
        # by this point.
        sequential_targets=["Llama4TextMLP"],
        # Llama4 is natively multimodal; without an explicit processor, newer
        # transformers versions try to auto-load a full multimodal AutoProcessor
        # (requires mistral_common) purely to prepare calibration inputs, even
        # though we already pass pre-tokenized text-only data above. Passing the
        # tokenizer directly skips that unnecessary auto-detection.
        processor=tokenizer,
    )

    # Save tokenizer alongside the quantized weights so the output directory
    # is a self-contained HuggingFace model directory
    tokenizer.save_pretrained(str(output_dir))

    files = list(output_dir.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"\nQuantization complete.")
    print(f"  Files: {len([f for f in files if f.is_file()])}")
    print(f"  Total: {total_size / (1024**3):.1f} GB")
    print(f"  Path:  {output_dir}")


def update_model_json(models_dir: Path, quantized_weights_path: str, quantization: str):
    model_json_path = models_dir / SOURCE_MODEL_NAME / "1" / "model.json"

    if not model_json_path.exists():
        print(f"\nWARNING: model.json not found at {model_json_path} — skipping update.")
        return

    with open(model_json_path) as f:
        config = json.load(f)

    config["model"] = quantized_weights_path
    # compressed-tensors quantized checkpoints are loaded without a separate
    # quantization flag — vLLM detects the format from the checkpoint metadata
    config.pop("quantization", None)
    config.pop("load_format", None)
    # INT8/INT4 compressed-tensors checkpoints work with bfloat16 compute dtype
    config["dtype"] = "bfloat16"

    with open(model_json_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"\nUpdated model.json: {model_json_path}")
    print(f"  model: {quantized_weights_path}")
    print(f"  dtype: bfloat16 (quantization inferred from checkpoint metadata)")


def main():
    args = parse_args()

    print("=" * 60)
    print("Llama-4-Scout-17B-16E Offline Quantization (llm-compressor)")
    print("=" * 60)

    check_dependencies()

    if args.ignore_ssl:
        _disable_ssl()

    triton_repo = Path(args.base_dir).resolve() if args.base_dir else PROJECT_ROOT / "triton-repo"
    weights_dir = triton_repo / "weights"
    models_dir = triton_repo / "models"

    source_dir = weights_dir / SOURCE_MODEL_NAME
    output_name = f"{SOURCE_MODEL_NAME}-{args.quantization}"
    output_dir = weights_dir / output_name

    quantize(
        source_dir=source_dir,
        output_dir=output_dir,
        quantization=args.quantization,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        smoothquant_strength=args.smoothquant_strength,
    )

    if args.update_model_json:
        update_model_json(models_dir, str(output_dir), args.quantization)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nQuantized weights: {output_dir}")
    print(f"\nTo use these weights with Triton, either:")
    print(f"  1. Re-run download_llama4scout.py pointing at the quantized directory:")
    print(f"       python scripts/download/download_llama4scout.py \\")
    print(f"           --base-dir {triton_repo} --no-verify")
    print(f"     then manually set \"model\": \"{output_dir}\" in model.json")
    print(f"  2. Pass --update-model-json to this script to do it automatically.")
    print(f"\nTo load the model in Triton:")
    print(f"  docker compose --profile vllm up -d state backend-vllm http-proxy grpc-proxy")
    print(f"  curl -X POST $TRITON_REST_URL/v2/repository/models/{SOURCE_MODEL_NAME}/load")


if __name__ == "__main__":
    main()
