# Llama 4 Scout — INT8 Precision/Accuracy Benchmarking Plan

Status as of 2026-07-17. Branch: `llama4scout-fp8-serving`. Split out from
`docs/llama4scout_int8/quantization_plan.md` (which covers getting the
quantization job itself to run successfully) since this is a distinct
follow-on workstream with its own tooling, timeline, and open questions.

## Prerequisite: the checkpoint this plan tests

The INT8 checkpoint referenced throughout this doc is complete and verified
— see `docs/llama4scout_int8/quantization_plan.md` for the full quantization
history. Location: `s3://marcdo126967-triton-models/default/llama4scout-int8/weights/`
(compressed-tensors format, 102.35 GiB exact — `109904872601` bytes via
`aws s3api list-objects-v2`, 5,475 tensors, produced via `llmcompressor==0.12.0`
SmoothQuant + `QuantizationModifier` W8A8).

**Sanity check on checkpoint size (2026-07-17)**: INT8 (102.35 GiB) is only
~4% smaller than the RedHatAI FP8-dynamic checkpoint (106.80 GiB, confirmed
via the same `list-objects-v2` method) — not "significantly" smaller as
might be intuitively expected. This is correct, not a bug: **INT8 and FP8
are both 8-bit (1-byte-per-weight) formats** — the difference between them
is the number representation (integer vs. floating-point), not the
bit-width, so similar output sizes are exactly what should happen. Both
land at roughly half of the BF16 original (202.39 GiB). A meaningfully
smaller checkpoint would require a 4-bit scheme (INT4/GPTQ), which is a
different quantization approach entirely, not a variant of what we ran.

## Hardware analysis for serving

**Compute capability requirement is much broader than FP8's.** Confirmed
directly from vLLM's own docs: *"INT8 computation is supported on NVIDIA
GPUs with compute capability > 7.5 (Turing, Ampere, Ada Lovelace, Hopper)"*
— unlike FP8's CC≥8.9 floor (Ada Lovelace/Hopper only), INT8 W8A8 opens up
Ampere (A10G, A100) and even Turing (T4). Blackwell (CC≥10.0) is explicitly
*not* supported. Volta (V100, CC7.0) remains excluded either way (one step
below Turing).

**Real sizing analysis** (checkpoint: 102.35 GiB; even-split memory =
102.35 ÷ GPU count, ignoring the per-GPU imbalance risk noted below):

| Instance | GPUs | VRAM/GPU | Even-split | Headroom | $/hr | AZ availability (our VPC) |
|---|---|---|---|---|---|---|
| `g5.24xlarge` | 4× A10G | 22.35GB | 25.6GB | negative — doesn't fit | $8.14 | 2a, 2b |
| `g6.24xlarge` | 4× L4 | 22.35GB | 25.6GB | negative — doesn't fit | $6.68 | 2a, 2b |
| `g6e.12xlarge` | 4× L40S | 44.7GB | 25.6GB | ~19GB/GPU | $10.49 | 2a, 2b |
| `p4d.24xlarge` | 8× A100-40GB | 40GB | 12.8GB (TP=8) | ~27GB/GPU | $21.96 | 2a, 2b |
| `g6e.48xlarge` | 8× L40S | 44.7GB | 12.8GB (TP=8) | ~32GB/GPU | $30.13 | 2a, 2b |
| `p4de.24xlarge` | 8× A100-80GB | 80GB | 12.8GB (TP=8) | ~67GB/GPU | $27.45 | 2a, 2b |

**Key findings:**

- `g5.24xlarge`/`g6.24xlarge` are ruled out by simple math, not just risk —
  even a perfectly even split (25.6GB/GPU) exceeds their 22.35GB card
  capacity, before accounting for KV cache or the per-GPU imbalance already
  observed once for a similarly-sized checkpoint (`g6.48xlarge` OOM'd
  serving the FP8 checkpoint at this exact VRAM class — see
  `docs/fp8_serving_configuration.md`).
- **`g6e.12xlarge` is the recommended primary target**: cheapest viable
  option with real headroom (~19GB/GPU spare), and it's in the
  "24xlarge-and-below" size tier where quantization saw dramatically better
  availability than the 48xlarge tier (round-1 success vs. 70+ rounds with
  no success for FP8's 8-GPU hunt).
- **`p4d.24xlarge` is a genuinely new candidate** specifically because INT8
  doesn't have FP8's CC≥8.9 floor — A100s were excluded for FP8 serving but
  are fully valid here. Historically scarce in this account/region per the
  quantization hunting history, so treated as a secondary target.
- **`g6e.48xlarge` is the known-working fallback** (already proven for the
  similarly-sized FP8 checkpoint), but it's the expensive, scarce 48xlarge
  tier the FP8-serving hunt never found across 70+ rounds.

**Recommended hunt priority**: `g6e.12xlarge` → `p4d.24xlarge` →
`g6e.48xlarge`.

## Goal

Measure whether INT8 quantization introduced meaningful accuracy/quality
loss relative to the full-precision (BF16) model — this has not yet been
tested; everything done so far only confirms the quantization pipeline
*runs* and produces a well-formed checkpoint, not that its outputs are
good.

## Baseline data — already have it, no need to run BF16 ourselves

NVIDIA's model card for `nvidia/Llama-4-Scout-17B-16E-Instruct-FP8`
publishes a direct BF16-vs-FP8 comparison:

| Benchmark      | BF16 | FP8 |
|----------------|------|-----|
| MMMU Pro       | 75   | 74  |
| GPQA Diamond   | 57   | 56  |
| HLE Challenge  | 4    | 4   |
| LiveCodeBench  | 36   | 32  |
| MATH-500       | 82   | 82  |
| AIME 2024      | 30   | 31  |

We can use these BF16 numbers as the reference point for our INT8 scores,
without needing to run a full-precision benchmark pass ourselves.

## Methodology

1. **Sanity check first** (cheap, fast, targeted at our specific risk):
   before any formal benchmark, run a handful of prompts spanning distinctly
   different domains (math, code, general knowledge, creative writing) —
   since Llama 4's MoE routes different content to different experts, this
   is a fast way to catch garbled/degenerate output that would signal an
   expert got silently corrupted by quantization.

2. **Formal benchmarks — reuse NVIDIA's comparison set** where practical:
   - **GPQA Diamond** (198 questions) and **MATH-500** (500 problems) as
     primary signal — both text-only, well-supported in EleutherAI's
     `lm-evaluation-harness`.
   - **AIME 2024** as a secondary check (only 30 problems — low statistical
     power alone).
   - Skip **MMMU Pro** initially (needs a multimodal eval harness — Llama 4
     Scout is vision-capable, but that's a bigger lift) and likely skip
     **LiveCodeBench**/**HLE** unless the above looks promising
     (code-execution sandboxing and a very large exam set, respectively,
     are heavier to stand up).

3. **Tooling**: EleutherAI's `lm-evaluation-harness`, vLLM backend, pointed
   at our INT8 checkpoint served via vLLM.

4. **Comparison basis**: our INT8 scores vs. NVIDIA's published BF16/FP8
   numbers directly.

5. **Later**, alongside serving work: latency/throughput comparison across
   BF16/FP8/INT8 — see `docs/s3_fuse_latency_test_results.md`'s "Reuse for
   INT8" section, which already scopes this as a follow-on to the
   S3-FUSE-vs-EFS methodology.

## Open questions to resolve as part of this work

- **Serving the INT8 checkpoint via vLLM is not yet attempted.** The
  FP8-serving investigation (`docs/fp8_serving_configuration.md`) found a
  working vLLM configuration for this exact Triton container
  (`vllm==0.9.2` + `transformers==4.51.3` + force-reinstalled
  `xformers==0.0.30`) for a `compressed-tensors`-format checkpoint. Since
  `llmcompressor`'s INT8 output is also `compressed-tensors` format, this is
  a strong starting point — but genuinely untested against the INT8
  checkpoint specifically. This needs to work before any of the benchmarks
  above can run (the harness needs a live serving endpoint to query).
- **The `self_attn`/`router` ignore-list question.** The official Llama4
  GPTQ example's recipe ignores `self_attn` and `router` in addition to
  `vision_model`/`multi_modal_projector`/`lm_head`; our SmoothQuant/W8A8
  recipe currently only ignores the latter three. Not confirmed whether
  this matters for W8A8 specifically (the official example is W4A16/GPTQ) —
  worth checking early in this work, since if it does matter, the
  quantization would need to be redone before benchmark numbers are
  meaningful.

## Open items / TODO

- [x] Hardware analysis for serving — done, see "Hardware analysis for
      serving" section above. Hunt candidates identified: `g6e.12xlarge`
      (primary) → `p4d.24xlarge` (secondary) → `g6e.48xlarge` (fallback).
- [ ] Run the hunt for serving hardware and get the INT8 checkpoint serving
      successfully via vLLM/Triton (see open question above) — blocking
      everything else in this doc.
- [ ] Run the sanity-check prompt battery across domains.
- [ ] Stand up `lm-evaluation-harness` against the served INT8 endpoint.
- [ ] Run GPQA Diamond + MATH-500 (primary), AIME 2024 (secondary).
- [ ] Resolve the `self_attn`/`router` ignore-list question — ideally
      before investing in the formal benchmark runs above, in case it
      changes what needs to be re-quantized.
- [ ] Compare results against NVIDIA's published BF16/FP8 numbers and write
      up findings.
- [ ] Cross-reference with `docs/s3_fuse_latency_test_results.md`'s planned
      INT8 latency/throughput follow-on.
