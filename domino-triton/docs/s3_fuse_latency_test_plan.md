# S3-FUSE Serving Latency Test — Plan & Assumptions

Status as of 2026-07-15. Branch: `llama4scout-fp8-serving`.

## Question being answered

Does serving Llama 4 Scout FP8 directly from an S3-FUSE mount meaningfully slow down
(a) the one-time model load, and/or (b) ongoing inference — compared to a network
filesystem baseline (EFS) we already have working? Our architectural understanding says
S3-FUSE should only affect (a), not (b), since once weights are in GPU memory the storage
backend is irrelevant. This test measures that directly rather than assuming it.

## What we're measuring

1. **Cold-load latency** — wall-clock time from issuing Triton's `POST
   /v2/repository/models/<name>/load` to the call returning (it blocks until the model is
   fully loaded and ready to serve). Expected to differ meaningfully across backends.
2. **Steady-state inference latency**, once the model is loaded:
   - **Time-to-first-token (TTFT)** — time from request sent to the first non-empty token
     received.
   - **Inter-token throughput** — tokens/sec during generation.
   Expected to show **no** meaningful difference across backends (the control).

## Methodology

- **Cold-load**: 5 trials per backend, each on a **freshly created pod** (not repeated
  load/unload within one running container — that would let OS page-cache from a prior
  trial bias later results to look artificially fast). Report mean/median/min/max.
- **Inference**: ~15-20 requests per backend, varying prompt length (short/medium/long),
  against the already-loaded model. Report mean/median/p95 for both TTFT and tokens/sec.
- **Tooling**: Triton's vLLM backend always streams responses token-by-token at the gRPC
  level (`model_transaction_policy { decoupled: true }`), even for "non-streaming" calls —
  confirmed by reading `scripts/clients/llm_vllm_grpc_client.py`, which loops over
  `client.stream_infer(...)` receiving one token per iteration regardless of the
  `--stream` flag. That means TTFT is capturable by timestamping the first loop
  iteration — the existing client doesn't expose this, so a small dedicated benchmark
  script (`s3_fuse_latency_bench.py`, written for this test) does the timing and reports
  JSON results instead of modifying the general-purpose interactive client.

## Test matrix — storage backends

| Backend | Setup | Notes |
|---|---|---|
| **S3-FUSE** | `llama4scout-fp8-s3-pv`/`-pvc`, prefix `default/llama4scout-fp8/weights/` in bucket `marcdo126967-triton-models` | The thing we actually care about. Weights sit flat in this prefix (no `weights/llama4scout-vllm-fp8/` nesting) — a fresh `config.pbtxt`/`model.json` pair was built pointing directly at the mount. |
| **EFS (shared store)** | `domino-shared-store-domino-compute` PVC, subPath `filecache/2f26ea10-b72a-4b4a-9d21-f29b613a77df` | Already has a complete Triton model-repo structure from the original download script (`models/llama4scout-vllm-fp8/config.pbtxt` + `.../1/model.json`, plus `weights/llama4scout-vllm-fp8/`) — reused as-is, no new config needed. |
| **Local NVMe/EBS** *(not in this pass — optional follow-up)* | Would require copying the 104GB FP8 weights onto a dedicated fast local volume | Deferred; S3-FUSE vs. EFS already answers the primary question with zero extra setup cost. |

## Real bugs found and fixed while standing this up

1. **vLLM couldn't auto-detect the FP8 scheme from the checkpoint at all.** Our
   `model.json` needed an explicit `"quantization": "modelopt"` field — this was flagged
   as an open question in the original download script's own comments, now confirmed and
   fixed (in the script itself, and in both the S3 and EFS copies of the already-downloaded
   checkpoint's `model.json`).
2. **A genuine packaging inconsistency in NVIDIA's checkpoint.** With the fix above
   applied, loading still failed — traced (by reading vLLM's actual `get_quant_config()`
   source) to `config.json` containing its own `quantization_config` field in a
   *flattened* shape (`quant_algo`, `config_groups` at the top level), while the separate
   sibling `hf_quant_config.json` file has the *nested* shape
   (`{"producer":..., "quantization": {...}}`) that vLLM's ModelOpt loader actually expects.
   vLLM checks `config.json` first and stops there if anything is present, so it never
   reaches the correctly-shaped sibling file. **Fix**: removed the `quantization_config`
   key from `config.json` (both S3 and EFS copies) so vLLM falls through to the
   correctly-shaped `hf_quant_config.json`.
3. **Hardware-fundamental finding, not a bug**: vLLM's FP8 (ModelOpt) quantization path
   requires GPU compute capability **≥8.9**, confirmed directly from a real load error:
   *"The quantization method modelopt is not supported for the current GPU. Minimum
   capability: 89. Current capability: 86."* This rules out `g5.48xlarge` (A10G, Ampere,
   CC 8.6) — and, by the same logic, **also rules out `p4d.24xlarge`/`p4de.24xlarge`
   (A100, Ampere, CC 8.0)**, since FP8 tensor-core support was only introduced with Ada
   Lovelace/Hopper. Only Ada Lovelace/Hopper-class hardware (CC≥8.9) is viable for
   serving this FP8 checkpoint.
4. **`g6.48xlarge` (L4, 24 GiB/GPU) OOM'd during actual FP8 model loading** — confirmed
   real CUDA OOM on GPU 0 (22.03 GiB total, 1.32 GiB free) loading the ~104GB checkpoint
   with `tensor_parallel_size: 8`. Same per-GPU memory-imbalance pattern documented in
   `docs/llama4scout_int8/quantization_plan.md` for quantization: `device_map`/vLLM's
   internal weight-sharding doesn't split perfectly evenly, so a GPU with only ~22 GiB
   usable has too little headroom above the even-per-GPU share. Rather than debug that
   headroom on the spot, we dropped `g6.48xlarge` and switched the hunt to
   **`g6e.48xlarge`** (L40S, ~44 GiB/GPU — same Ada Lovelace CC 8.9, roughly double the
   per-GPU VRAM) plus `p5.48xlarge` (H100, 80 GiB/GPU) as the two live candidates, with
   `p5en.48xlarge` added after a 3-hour no-capacity window on either.

## Infrastructure

- **Compute**: needs a GPU generation with FP8 tensor-core support (CC≥8.9 — see finding
  above), with enough combined VRAM for the ~104GB checkpoint and enough per-GPU headroom
  above the ~22 GiB that already OOM'd on `g6.48xlarge` (see finding 4). Current hunt
  targets: `g6e.48xlarge` (8×L40S, CC 8.9, ~44 GiB/GPU) and `p5.48xlarge` (8×H100, CC 9.0,
  ~80 GiB/GPU), with `p5en.48xlarge` as a 3-hour-timeout fallback. This is a deliberately
  smaller/cheaper node than the quantization job where possible so the two don't compete
  for resources or interfere with each other.
- **Provisioning**: manual (bypassing Karpenter), same `nodeadm` bootstrap process used
  throughout this investigation — see `docs/llama4scout_int8/quantization_plan.md` for
  the general pattern (AMI, security group, IAM instance profile, labels/taints).
- **Karpenter cleanup**: the `gpu` NodePool's instance-type list and resource limits were
  reverted to their original values (`g5.2xlarge`/`g5.12xlarge`/`g5.24xlarge`/`g6.12xlarge`,
  `limits: {cpu: 80, memory: 320Gi, nvidia.com/gpu: 10}`) after concluding manual
  provisioning is more reliable for these one-off, large-hardware needs — Karpenter
  repeatedly failed to even attempt launches for 8-GPU/large instance types during this
  investigation for reasons never fully root-caused.

## Execution sequence

1. Manually provision a `g6e.48xlarge`/`p5.48xlarge` node (see hardware findings above —
   not `g5.48xlarge` or `g6.48xlarge`, neither of which is viable for this checkpoint).
2. For each backend (S3-FUSE, EFS): fresh Triton pod → time the load call → run the
   inference battery against the loaded model.
3. Aggregate into a comparison table (load-time and TTFT/throughput, mean/median/p95 per
   backend).
4. Report findings against the expectation stated above.

**Status as of 2026-07-16**: infrastructure and bug-fixing done, including two real bugs
found in `s3_fuse_latency_bench.py` itself: (a) `tokens_per_sec` had an off-by-one — the
denominator (`gen_time`) only spans time *after* the first token arrived, but the
numerator counted the first token too, inflating throughput; fixed by measuring
steady-state throughput as `(token_count - 1) / gen_time`. (b) no warm-up request before
the timed battery, so CUDA-graph/kernel-autotuning cost on the very first inference could
bias the "short" prompt's stats specifically (it's always first in iteration order); fixed
by issuing one discarded warm-up request first. A separate, dedicated hunt for
`g6e.48xlarge`/`p5.48xlarge`/`p5en.48xlarge` capacity is running in parallel; this
specific test run is executing on a `g6e.48xlarge` node originally provisioned for the
INT8 quantization job (reused once idle, since it independently meets the same CC≥8.9 +
per-GPU headroom bar) rather than waiting on that separate hunt.

## Assumptions worth flagging

- We assume the S3-FUSE CSI driver's read performance is representative regardless of
  which prefix/bucket is used — we're testing against the same bucket
  (`marcdo126967-triton-models`) already proven to work for other purposes in this
  cluster, not a synthetic benchmark bucket.
- We assume EFS is a fair "already available" baseline, not necessarily the fastest
  possible one — it's itself a network filesystem (not local NVMe), so a gap between
  S3-FUSE and EFS represents a *conservative* estimate of the S3-FUSE penalty; the true
  gap vs. genuinely fast local storage would likely be larger, hence the note about local
  NVMe as an optional deeper follow-up.
- 5 cold-load trials and ~15-20 inference requests per backend are a pragmatic sample
  size for this investigation, not a rigorous statistical study — reported spreads
  (min/max, p95) should be read as directional, not as tight confidence intervals.

## Reuse for INT8

Once the currently-running INT8 quantization job completes, this exact same methodology
and tooling (bench script, pod manifest templates, backend list) will be repeated against
the INT8 checkpoint (once copied to its own S3 prefix, mirroring the FP8 setup) to get a
second, independent data point — and to compare FP8 vs. INT8 serving characteristics as a
secondary output, alongside the S3-FUSE-vs-EFS comparison itself.
