# S3-FUSE vs EFS Serving Latency — Results & Recommendation

Status as of 2026-07-16. Branch: `llama4scout-fp8-serving`. Companion to
`docs/s3_fuse_latency_test_plan.md` (the pre-registered plan/methodology this
document reports results against).

## Executive Summary

**Recommendation: use S3-FUSE for production serving.**

- **Cost**: S3 Standard storage is **~13x cheaper** than EFS Standard
  ($0.023/GB-month vs $0.30/GB-month, confirmed via AWS Pricing API,
  us-west-2). For a ~107GB checkpoint, that's ~$2.46/month on S3 vs
  ~$32/month on EFS.
- **Steady-state inference** (time-to-first-token, tokens/sec): **no
  meaningful difference** between backends. This matches the expected
  behavior — once weights are resident in GPU memory, the storage backend
  they were loaded from is irrelevant to per-token serving performance.
- **Cold-load time**: S3-FUSE was **~14% faster** in this test (807.88s vs
  937.42s). This is a real, precisely-measured result, but based on a single
  trial per backend rather than the 5 trials the original test plan called
  for — treat this specific number as directional, not conclusive (see
  Limitations below).

Since cost strongly favors S3-FUSE and performance is at parity-or-better,
there is no reason from this data to prefer EFS for serving this checkpoint.

## Experiment Design

**Question**: does serving Llama 4 Scout FP8 from an S3-FUSE mount
meaningfully slow down (a) the one-time model load, and/or (b) ongoing
inference — compared to an EFS (NFS-backed shared store) baseline?

**Hypothesis**: storage backend should only affect (a), not (b), since once
weights are in GPU memory the storage backend is irrelevant. This test
measured that directly rather than assuming it.

**Method**: for each backend, deploy a fresh Triton+vLLM pod pointed at that
backend's copy of the checkpoint, time the model load call
(`POST /v2/repository/models/<name>/load`, which blocks until the model is
fully loaded and ready), then run a battery of 15 inference requests (3
prompt lengths — short/medium/long — × 5 repeats each) via a dedicated
benchmark client, measuring time-to-first-token (TTFT) and post-first-token
throughput (tokens/sec) for each request.

## Hardware

- **Node**: `g6e.48xlarge` — 8× NVIDIA L40S GPUs (44.39 GiB VRAM each, ~355GB
  combined), us-west-2. Manually provisioned (bypassing Karpenter, which
  repeatedly failed to launch this instance class for reasons never fully
  root-caused).
- **On-demand cost**: $30.13/hr (confirmed via AWS Pricing API, us-west-2).

## Software

- **Base container**: `nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3`.
- **vLLM**: upgraded to `0.9.2` inside the pod at startup. The base image's
  bundled vLLM (`0.8.1+nv25.04`) has no native Llama4 model registration at
  all and falls back to a broken generic `transformers` code path. Newer
  vLLM releases (`0.10.0`, `0.12.0`) were also tried and rejected: `0.10.0`
  hit a checkpoint-schema naming mismatch specific to NVIDIA's ModelOpt FP8
  export format (see below), and `0.12.0` is incompatible with this Triton
  container's bundled backend glue code (a hard `pydantic` pin conflict plus
  a removed `vllm.engine.metrics` module) — confirmed checkpoint-agnostic,
  a pure import-time failure. `0.9.2` is the version confirmed compatible
  with this Triton container that also has full native Llama4 + MoE support
  for the checkpoint format actually used (see below).
- **Additional pins required for vLLM 0.9.2** (both are architecture-level
  fixes, not checkpoint-specific — they'd apply to any Llama4 Scout
  checkpoint on this vLLM version):
  - `transformers==4.51.3` — vLLM 0.9.2 only floors `transformers>=4.51.1`
    with no ceiling; an unpinned install resolves to `>=4.54.0`, which added
    native `aimv2` config registration that collides with vLLM's own bundled
    Ovis config shim.
  - `xformers==0.0.30`, force-reinstalled (`--force-reinstall --no-deps`) —
    the base image's local build (`0.0.30+nv25.04`) satisfies a plain
    version pin as a no-op but has an unfixed Triton-JIT API call in its
    bundled split-K kernel, which the vision tower's `MultiHeadAttention`
    hits regardless of checkpoint. The genuine public wheel already guards
    this call correctly.
- **Model**: `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic`
  (`compressed-tensors` FP8 quantization), `tensor_parallel_size=8`,
  `gpu_memory_utilization=0.85`, `max_model_len=16384`.

  **Important note on checkpoint choice**: the original target was NVIDIA's
  own `nvidia/Llama-4-Scout-17B-16E-Instruct-FP8` (ModelOpt format), but it
  hit an unresolved vLLM bug: the checkpoint's per-expert scale tensors use
  an underscore-joined naming convention (`down_proj_input_scale`) that
  vLLM's fused-MoE checkpoint-name-to-parameter substitution logic
  mishandles (it unconditionally appends `weight` to the replacement
  prefix, producing a name — `w2_weight_input_scale` — that never matches
  vLLM's actually-registered parameter, `w2_input_scale`). This is a
  structural bug in vLLM's substitution logic, not a version-lag issue —
  confirmed present in both `0.10.0` and `0.11.1`. NVIDIA's own model card
  also only documents TensorRT-LLM/SGLang as validated runtimes for this
  checkpoint, not vLLM. We switched to RedHatAI's `compressed-tensors`-format
  checkpoint instead, which loaded successfully — `compressed-tensors` is
  vLLM's own native quantization format and didn't hit this issue.

## Storage Backends Compared

| Backend | Setup |
|---|---|
| **S3-FUSE** | Mountpoint-for-S3 CSI driver PV/PVC, prefix `default/llama4scout-fp8-dynamic/weights/` in bucket `marcdo126967-triton-models` |
| **EFS** | NFS-backed shared store PVC (`domino-shared-store-domino-compute`), checkpoint at `/mnt/data/llama4scout-fp8/weights/llama4scout-vllm-fp8-dynamic` |

Both copies are the same ~107GB checkpoint (106.8GB / 35 real files per
`huggingface_hub`'s `model_info`, confirmed identical after copy via file
count and spot comparison).

## Metrics Captured

1. **Cold-load latency** — wall-clock time from issuing Triton's load call to
   it returning (blocks until fully loaded and ready).
2. **Time-to-first-token (TTFT)** — time from request sent to the first
   non-empty streamed token, over 15 requests (3 prompt lengths × 5 reps).
3. **Post-first-token throughput (tokens/sec)** — tokens produced after the
   first token, divided by the time elapsed after the first token (this
   specifically excludes the first token's production time from the
   denominator, fixing an off-by-one bug found in an earlier version of the
   benchmark script that inflated this number).

## Results

| Metric | S3-FUSE | EFS |
|---|---|---|
| Cold load | **807.88s** | **937.42s** |
| TTFT mean | 91.8ms | 92.9ms |
| TTFT median | 90.6ms | 93.0ms |
| TTFT p95 | 94.3ms | 95.8ms |
| TTFT min | 87.6ms | 87.9ms |
| TTFT max | 107.7ms | 104.8ms |
| Throughput mean | 53.9 tok/s | 54.2 tok/s |
| Throughput median | 54.6 tok/s | 54.7 tok/s |
| Throughput p95 | 54.7 tok/s | 55.3 tok/s |

## Cost

| Storage class | Price (us-west-2, confirmed via AWS Pricing API) | Cost for this ~107GB checkpoint |
|---|---|---|
| S3 Standard | $0.023/GB-month | ~$2.46/month |
| EFS Standard | $0.30/GB-month | ~$32.10/month |

(Storage cost only — not request/throughput pricing. For large, few-file
model checkpoints like this one, per-GB storage cost is the dominant factor
over request-based charges.)

## Analysis

1. **Steady-state inference is unaffected by storage backend.** TTFT and
   throughput are within noise of each other across backends (TTFT means
   91.8ms vs 92.9ms; throughput means 53.9 vs 54.2 tok/s). This confirms the
   stated hypothesis directly rather than assuming it: once weights are in
   GPU memory, the storage backend they came from doesn't matter for
   per-request serving performance.
2. **Cold-load shows a real, precisely-measured gap favoring S3-FUSE**
   (807.88s vs 937.42s, ~14% faster) — a genuine difference in this trial,
   not an artifact of measurement error (both were timed the same way: a
   single `curl` call from Triton-ready to model-ready). However, this is
   n=1 per backend against a plan that called for 5 trials each specifically
   to average out pod-to-pod and cache-state variance — so while the number
   itself is solid, whether it reflects a *systematic* S3-FUSE advantage or
   this specific trial's variance is not yet established.
3. **Cost decisively favors S3-FUSE**, independent of any performance-trial
   variance — a ~13x per-GB difference is a structural property of the two
   storage classes, not something additional trials would change.

## Limitations

- **Cold-load: 1 trial per backend**, not the 5 the original test plan
  specified (to average out fresh-pod/cache-state variance). The 14% gap
  reported above should be read as directional, not statistically confirmed.
- **Text-only inference.** Llama 4 Scout is natively multimodal, but none of
  the prompts used include image inputs — vision-path latency/throughput is
  untested here.
- **Tested against RedHatAI's `compressed-tensors` checkpoint, not NVIDIA's
  official ModelOpt checkpoint**, due to the unresolved vLLM bug described
  above. Results may not carry over exactly if/when the ModelOpt checkpoint
  becomes usable with vLLM, though there's no specific reason to expect
  storage-backend behavior to differ by quantization format.
- **15 sequential (not concurrent) requests per backend** — matches the
  original plan's sample size, which was explicitly scoped as a pragmatic
  sample for this investigation, not a rigorous statistical study. p95
  figures at this sample size should be read as directional.

## Recommendation

Use **S3-FUSE** for production serving of this checkpoint. It is
substantially cheaper (~13x on storage cost alone), shows no inference-time
penalty, and even showed a modest cold-load advantage in this test. If
cold-load time becomes business-critical, additional trials (per the
original plan's 5-trials-per-backend design) would sharpen confidence in
that specific number — but the cost differential alone is a strong,
structural reason to prefer S3-FUSE regardless of how that number resolves
with more trials.

## Reuse for INT8

Once INT8 quantization completes (tracked separately, see
`docs/llama4scout_int8/quantization_plan.md`), this same methodology and
tooling can be repeated against the INT8 checkpoint for a second, independent
data point on both storage-backend behavior and FP8-vs-INT8 serving
characteristics.
