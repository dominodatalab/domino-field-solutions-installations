# Llama 4 Scout — INT8 Quantization Experiment & Evaluation Plan

Status as of 2026-07-17 (originally 2026-07-15). Branch: `llama4scout-fp8-serving` (off
`mdoan_updates-infra-scripting-llama4scout`).

## Goal

Test whether Marc's `scripts/quantize/quantize_llama4scout.py` (generic SmoothQuant + W8A8
`QuantizationModifier`, no MoE-router-aware handling) can actually produce a usable INT8
checkpoint for Llama 4 Scout — this was previously only a documentation-based risk
(NVIDIA/Meta support matrices exclude INT8 for this architecture; llm-compressor's own
official example uses W4A16 instead). Rather than trust the docs alone, we ran the real
script against the real full-precision weights to see what actually happens.

## DEFINITIVE FINDING: V100 (Volta) hardware is NOT viable for this quantization job

**Do not attempt this quantization job on `p3dn.24xlarge` (or any other V100-based
instance) again without first sourcing a build of PyTorch with Volta (SM70) kernels
compiled in.** This was tested directly, not inferred from documentation:

- NVIDIA's Triton 25.04 release notes state a hard minimum of CUDA compute capability
  7.5+ — Volta (V100) is CC 7.0, one step below that floor.
- We confirmed this empirically: even with the image's **original**, untouched PyTorch
  build (`2.7.0a0+79aa17489c.nv25.04`), running an actual GPU computation (a basic
  embedding lookup, `torch.embedding`) fails with:
  ```
  RuntimeError: CUDA error: no kernel image is available for execution on the device
  ```
- Interestingly, **loading the model onto the GPUs (copying the 50 checkpoint shards into
  V100 memory via `device_map="auto"`) succeeds fine** — that's a simpler memory-copy
  operation that doesn't need architecture-specific compiled kernels. The failure only
  appears the moment real compute is attempted. This means "the model loads" is *not*
  evidence Volta will work — you have to get to an actual forward pass to find out.
- We reached this failure point after fixing three unrelated software issues first (see
  below), so this is a clean, isolated result: it is the *hardware/PyTorch-build*
  combination that fails, not a leftover software bug.

**If Volta support is worth pursuing further**, the untested next step is a *different*
PyTorch build — not NVIDIA's Triton-image-bundled one — specifically a standard PyPI
wheel from a version confirmed (not just assumed) to include Volta/SM70 in its compiled
kernel list. We did not verify this before running out of easy options; PyTorch's own
release notes don't document a compute-capability floor explicitly, so this would need
direct testing (e.g. a small script that runs `torch.embedding` on the target GPU) before
committing to a full environment build.

**Bottom line**: for now, this job needs an Ampere-or-newer GPU (A10G, L4, A100, H100).
`g5.48xlarge`/`g6.48xlarge` don't have enough VRAM (192GB nominal / ~179GiB real vs.
~202GiB required). The path forward is waiting for AWS capacity on `p4d.24xlarge`,
`p4de.24xlarge`, or `p5.48xlarge` — all confirmed correctly-sized and architecturally
compatible, just not currently available (see below).

## Infrastructure used

- **Compute**: full BF16 quantization needs ~202GiB VRAM just for weights (confirmed via
  actual byte count of the model in S3, not the script's rough "~218GB" comment).
  - Karpenter's `gpu` NodePool was widened to 8-GPU instances (`g5.48xlarge`, `g6.48xlarge`
    — 192GB nominal / ~179GiB real, insufficient for quantization but fine for FP8 serving)
    and then to `p4d.24xlarge` (8×A100-40GB, 320GB nominal). Karpenter never even attempted
    a launch for `p4d.24xlarge` (no NodeClaim created) — root cause not fully confirmed,
    leading theory is a gap in Karpenter's instance-type/pricing cache for this less-common
    family.
  - Bypassed Karpenter and called `aws ec2 run-instances` directly for `p4d.24xlarge`,
    `p4de.24xlarge` (A100-80GB), and `p5.48xlarge` (H100) — on-demand and spot, both AZs
    (`us-west-2a`/`us-west-2b`), tried twice on different days. **Every single attempt
    (10+ across both sessions) returned genuine `InsufficientInstanceCapacity` / "no Spot
    capacity available" errors.** This is a real, current AWS capacity shortage in this
    account/region for all three modern large-VRAM instance families, not a Karpenter bug
    or account/SCP restriction — confirmed via direct AWS API responses, not simulated.
  - Fell back to **`p3dn.24xlarge`** (8×V100-32GB, 256GB nominal / ~238GiB real — VRAM-wise
    sufficient, but see the Volta finding above). This launches successfully on the first
    attempt every time we tried (unlike the modern instance types).
  - **Manually joined** to the EKS cluster `marcdo126967` via the AL2023 `nodeadm` bootstrap
    mechanism (NOT the older `bootstrap.sh`), replicating Karpenter's exact labels/taint so
    existing pod tolerations/scheduling work unchanged:
    - Labels: `dominodatalab.com/domino-node=true`, `dominodatalab.com/node-pool=default-gpu`,
      `nvidia.com/gpu=true`
    - Taint: `nvidia.com/gpu=true:NoSchedule`
    - Same AMI (`ami-057618cde71cf156a`), security group (`sg-08f407a31d7e70bf6`), and IAM
      instance profile (`marcdo126967_6216622694565352415`) that Karpenter-managed nodes use.
  - **Cost**: `p3dn.24xlarge` on-demand = **$31.21/hr** (confirmed via AWS Pricing API).
    Manually-launched nodes are NOT managed by Karpenter — they do not auto-scale down.
    **Must be terminated manually** when done. (Both test instances used in this
    investigation, `i-041ad9f0da5893910` and `i-053f8c473a56a0122`, were terminated after
    their respective tests concluded.)
  - Tag manual nodes clearly, e.g. `purpose=manual-temporary-quantization-node-delete-when-done`,
    to avoid confusion with Karpenter-managed fleet nodes during cleanup.

- **Storage** (all via the AWS Mountpoint S3 CSI driver, bucket `marcdo126967-triton-models`
  — same bucket Marc's Terraform provisions, already proven working in this cluster):
  - Full BF16 source (read-only mount): prefix `default/triton-demo/triton-repo/weights/llama4scout-vllm/`
    (already existed in the bucket from an earlier demo — saved us a ~217GB re-download).
  - INT8 output (read-write mount): prefix `default/llama4scout-int8/weights/` — created,
    still empty (quantization never completed due to the Volta finding above).
  - FP8 checkpoint (already copied earlier, separate task): prefix `default/llama4scout-fp8/weights/`,
    verified byte-exact against source (111,619,471,066 bytes both sides).
  - When mounting a fresh S3-FUSE PV/PVC on a **newly, manually-joined** node, expect the
    CSI driver's per-node "Mountpoint Pod" helpers (namespace `mount-s3`) to take ~30-60s to
    start after the node joins — an immediate pod apply can hit a transient
    `FailedMount: mountpoint pod not ready` error that resolves itself shortly after.

- **Container image**: `nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3` (same image the
  repo's `Dockerfile.triton.vllm` uses for serving).
  - `pip install llmcompressor datasets` (latest, unpinned) pulls a cascade of upgrades —
    `torch` (`2.7.0a0+79aa17489c.nv25.4` → `2.12.0`), `transformers` (`4.51.3` → `5.10.1`),
    `compressed-tensors` (`0.9.2` → `0.17.1`), `numpy` (`1.26.4` → `2.4.6`) — that breaks the
    image's pinned vLLM and, separately, drops Volta support in the newly-installed torch.
    **Fix**: pin `llmcompressor==0.4.1` (Feb 2025) instead — its dependency floor
    (`torch>=1.7.0`, `transformers>4.0,<5.0`, `compressed-tensors==0.9.2`) is fully satisfied
    by what's already in the image, so pip installs it without touching anything else.
  - `llmcompressor==0.4.1` does **not** export `oneshot` at the top level like newer
    versions do. Marc's script's `from llmcompressor import oneshot` must be changed to
    `from llmcompressor.transformers import oneshot` for this version.
  - `llmcompressor==0.4.1`'s dataset pipeline requires a real `datasets.Dataset` object
    (needs `.column_names`), not a plain Python list. Marc's `load_calibration_dataset()`
    function must be rewritten to build a `Dataset.from_dict({"input_ids": [...],
    "attention_mask": [...]})` instead of appending tokenizer outputs to a list.
  - The container's `python3` binary is not aliased to `python` — invoke scripts with
    `python3` explicitly, and use `python3 -m pip install` rather than bare `pip install`
    (they can resolve to different environments in this image).
  - With all four of the above fixes applied, the pipeline runs correctly through model
    loading and calibration setup — it is *only* blocked by the Volta hardware limitation
    documented above, not by any remaining software issue.

## Precision/accuracy evaluation

Moved to its own document — see `docs/llama4scout_int8/benchmarking_plan.md`. That's a
distinct workstream (its own tooling, timeline, and open questions) from getting the
quantization job itself to run, which is what the rest of this document covers.

## Scope note for INT8 recipe — FIXED

Marc's INT8 recipe (`build_recipe()` in `quantize_llama4scout.py`) originally only excluded
`ignore=["lm_head"]` — unlike NVIDIA's official FP8 checkpoint, which explicitly excludes
`vision_model*` and `multi_modal_projector*` from quantization (kept at full precision).
Since image classification/benchmarking is a confirmed future requirement for this
project, this has been fixed: both the INT8 (`QuantizationModifier`) and INT4
(`GPTQModifier`) branches now use `ignore=["lm_head", "vision_model*",
"multi_modal_projector*"]`, matching NVIDIA's own approach.

## MoE-specific finding: got past hardware, hit an architecture-specific bug — and there's a real, already-built fix

Once we had real Ampere hardware (`p4d.24xlarge`, see below), the Volta blocker was gone —
model loaded in ~3 min, and the full 512-sample SmoothQuant calibration pass completed
successfully in ~23 min. It then failed immediately after, while applying the calculated
smoothing scales:

```
AttributeError: 'Llama4TextExperts' object has no attribute 'weight'
```

**Root cause**: Llama4's MoE layers store all 16 experts' weights as a few large *batched*
tensors (`Llama4TextExperts`) rather than as separate per-expert `nn.Linear` layers — this
is a serving-speed optimization common to most modern MoE architectures (Mixtral,
DeepSeek-MoE, etc.), not a Llama4-specific quirk. `llmcompressor`'s SmoothQuant step (the
version we were pinned to, `0.4.1`, from Feb 2025) uses a generic, architecture-agnostic
layer-mapping that assumes any matched layer exposes a normal `.weight` — confirmed in the
logs: *"Architecture Llama4ForCausalLM not found in mappings. Using default mappings"*.
This is exactly the risk flagged from documentation at the very start of this
investigation (NVIDIA/Meta support matrices excluding INT8 for this architecture), now
confirmed empirically rather than just inferred.

**The fix — `llmcompressor` added proper MoE support since we pinned the old version:**

- `llmcompressor` **v0.11.0** (June 2, 2026) introduced explicit MoE calibration support:
  *"Applying quantization to Mixture-of-Experts (MoE) models requires explicit
  linearization and class overriding in order to efficiently calibrate experts."*
- **v0.12.0** (June 15, 2026, current latest stable) refined this into a `load_context()`
  context manager for cleaner loading.
- Confirmed directly in the source (`llmcompressor/modeling/moe/`): there is a dedicated
  **`llama4.py`** file, and `conversion_mappings.py` explicitly registers
  `Llama4TextExperts` as a supported architecture.
- **What it actually does**: a `from_experts_module()` function iterates through each of
  the 16 experts, extracts that expert's slice out of the batched tensors, and builds it
  into a standalone module (`ExpertMLPWithGate`) with normal, individually-addressable
  `gate_proj`/`up_proj`/`down_proj` weights — i.e., temporarily "unfusing" the
  efficient-but-uncalibratable batched format into per-expert layers that generic
  calibration code can target one at a time. `load_context()` makes this swap happen
  transparently at model-load time.
- **Not confirmed**: whether there's a documented re-fusion step back into the efficient
  batched format for serving, and whether this path is validated for W8A8/SmoothQuant
  specifically (the upstream example only demonstrates W4A16/GPTQ). Both are real open
  questions to watch for during testing, not assumed to be fine.

**We no longer need the old `llmcompressor==0.4.1` pin** — that pin existed purely to avoid
pulling in a newer PyTorch build that dropped Volta support. Now that we're on Ampere
hardware, there's no reason to stay on the pre-MoE-aware library version.

**Fix applied to `quantize_llama4scout.py`** (2026-07-15):
- Switched back to `llmcompressor==0.12.0` (plus `torchvision`, matching torch's version,
  per the already-diagnosed torch/torchvision pairing issue from the Volta investigation).
- Wrapped the model-loading call in `with load_context():`, with an inline comment marking
  this as MoE-specific (not something a dense/non-MoE model would need):
  ```python
  # MoE-specific: Llama4's MoE layers (Llama4TextExperts) store all 16 experts'
  # weights as a few large batched tensors for fast serving, rather than as
  # separate per-expert Linear layers. ... load_context() intercepts model loading
  # and transparently swaps each Llama4TextExperts module for llm-compressor's
  # linearized equivalent ... This is only needed for MoE architectures — a dense
  # (non-MoE) model would not require this wrapper.
  with load_context():
      model = AutoModelForCausalLM.from_pretrained(...)
  ```
- Passed `processor=tokenizer` explicitly to `oneshot()`, to avoid `transformers`' newer
  versions trying to auto-load a full multimodal `AutoProcessor` (which requires
  `mistral_common` and previously crashed with a `ReasoningEffort` import error) purely to
  prepare calibration inputs we already pre-tokenized ourselves.
- Smoke-tested (2 calibration samples) before committing to a second full run — see
  status below.

## Memory-imbalance finding (2026-07-16): got past the MoE bug, hit real OOMs — and there's a better documented fix

With the `load_context()` fix above, the smoke test progressed past the `Llama4TextExperts`
bug and into actual weight loading — then hit `torch.OutOfMemoryError` twice, on two
different node types:

- `p4d.24xlarge` (A100-40GB): GPU 0 OOM'd with only 1.56 MiB free on a 39.49 GiB card.
- `g6e.48xlarge` (L40S-48GB): GPU 1 OOM'd with only 31.38 MiB free on a 44.39 GiB card,
  `torch.OutOfMemoryError: ... this process has 44.35 GiB memory in use`.

**Root cause**: `device_map="auto"` treats each layer's routed-expert block
(`Llama4TextExperts`, ~2.0B params ≈ 4 GB/layer in BF16 — computed from the real config:
`hidden_size=5120`, `intermediate_size=8192`, `num_local_experts=16`, all 48 layers MoE) as
one atomic, unsplittable unit. Its greedy bin-packer doesn't distribute these evenly, so
whichever GPU draws the most blocks ends up carrying far more than an even per-GPU share —
confirmed by the fact each OOM'd GPU filled to ~100% of its own capacity while total cluster
VRAM had hundreds of GB free.

**First attempt (superseded, do not reuse)**: capped `max_memory` per GPU below hardware
max to force more balanced placement. This was a guess, not a documented fix, and was
dropped once a better answer was found — noting it here only so it isn't reintroduced.

**Actual fix — sequential onloading (from `llm-compressor`'s own official Llama4 example,
`examples/multimodal_vision/llama4_example.py`)**: load the model with `device_map=None`
(full model onto CPU RAM) and pass `sequential_targets=["Llama4TextMLP"]` to `oneshot()`.
`oneshot()`'s default "sequential" pipeline then streams one layer at a time onto GPU for
calibration, evicting it afterward, instead of statically placing the whole model across
GPUs up front. `Llama4TextMLP` is specifically the shared-expert MLP submodule (confirmed
by reading `transformers`' `modeling_llama4.py`: each `Llama4TextMoe` layer contains
`self.experts` (`Llama4TextExperts`, the routed batch), `self.router`, and
`self.shared_expert` (`Llama4TextMLP`)) — the routed experts don't need to be named
explicitly because `load_context()`'s linearization (see MoE fix above) has already broken
them into small individual `ExpertMLPWithGate` modules (~251 MB each) that the default
sequential tracer handles on its own.

**This changes the hardware requirement.** Peak *GPU* VRAM drops to roughly one layer's
worth (~5 GB) plus always-resident embeddings/lm_head/vision tower (~5 GB) plus calibration
activations — comfortably inside any 22GB+-class GPU, and quantization has no CC≥8.9
requirement (that's serving-specific). The new binding constraint is **system RAM**: the
full BF16 model (≈114.5B total params × 2 bytes ≈ ~230 GB) must fit in CPU memory before
streaming. Checked real AWS specs (`aws ec2 describe-instance-types` /
`aws pricing get-products`, both in `us-west-2`, both AZs our VPC actually has subnets in):

| Instance | GPUs | System RAM | On-demand $/hr |
|---|---|---|---|
| `g6.24xlarge` | 4x L4 (22.9 GiB) | 384 GiB | $6.68 |
| `g5.24xlarge` | 4x A10G (22.9 GiB) | 384 GiB | $8.14 |
| `g6e.48xlarge` (previous target) | 8x L40S | 1536 GiB | $30.13 |
| `p4de.24xlarge` (previous target) | 8x A100-80GB | 1152 GiB | $27.45 |

**New target: `g6.24xlarge`** — clears the ~230 GB estimate with ~154 GB margin, ~78%
cheaper than the previous 8-GPU targets, and a much less contested instance size (should
find capacity faster too, though our VPC's subnets are still only in `us-west-2a`/`2b`
regardless of instance type).

**Fix applied to `quantize_llama4scout.py`** (2026-07-16): replaced the `max_memory` hack
with `device_map=None` + `sequential_targets=["Llama4TextMLP"]`. Pod manifests'
resource requests updated from `nvidia.com/gpu: 8` / `memory: 1000Gi` (sized for the old
8-GPU node) to `nvidia.com/gpu: 4` / `memory: 350Gi` (sized for `g6.24xlarge`, leaving ~34
GiB RAM headroom for OS/kubelet overhead).

**Confirmed (2026-07-16/17)**: smoke test succeeded on `g6.24xlarge` (2 samples, ~31 min
linearization + fast calibration), and auto-launched the full 512-sample run, which also
**completed successfully** — verified directly against the output checkpoint (3 safetensors
shards, ~102.3 GiB total, byte-exact match between EFS and S3 copies; the index file's
`total_size` matches the sum of shard sizes almost exactly; 5,475 tensors including both
`weight` and `weight_scale` per MoE expert layer, confirming real INT8 quantization;
`total_parameters: 107.8B`, consistent with Scout's known size).

**Real operational finding along the way — memory headroom on `g6.24xlarge` is tighter than
it looks**: the smoke test's `memory: 350Gi` limit (leaving ~34 GiB headroom below the
node's ~355.5 GiB allocatable) was **not enough for the full 512-sample run** — it hit a
clean container-level OOMKill (exit 137) partway through calibration. Bumping the limit to
`353Gi` (the highest safely-schedulable value on that node) let the full run complete —
memory peaked right at 353 GiB, held, then declined as the sequential pipeline progressed
through later layers. **However**, roughly 4 hours *after* the run had already finished
(confirmed via output-file timestamps), the node's kubelet and then its SSM agent both
stopped responding — consistent with node-level memory pressure, a more severe failure mode
than the earlier clean OOMKill, likely from leaving too little headroom below the node's
total capacity for sustained operation. The output checkpoint was safe regardless (it lives
on EFS, a separate system from the compute node), but **the node itself had to be
terminated as unrecoverable/unmanageable afterward**, not cleanly shut down. If this
pipeline is run again, worth either a small dedicated cleanup/exit step after the script
finishes (to reduce time spent idling near the memory ceiling) or a bit more headroom if a
larger instance is available. See `docs/llama4scout_int8/compute_wait_times_reference.md` for the full
timing breakdown of this run.

**Calibration time note**: the full run took roughly 14-15 minutes *per layer* for the
512-sample calibration pass (49 layers total) — projecting to **~11-12 hours** total,
well beyond this doc's original "1-3 hour" estimate, which predates the CPU-based
sequential-onload fix above. Worth factoring into future planning if faster turnaround
matters more than the memory savings this approach buys.

**Separate, smaller finding worth a look later**: whether the recipe's `ignore` list should
also exclude `self_attn`/`router` (the official Llama4 GPTQ example does; our SmoothQuant/
W8A8 recipe doesn't) — moved to `docs/llama4scout_int8/benchmarking_plan.md`'s open
questions, since it's really a precision-evaluation concern, not a "does the pipeline run"
concern.

## Open items / TODO

- [x] ~~Confirm quantization job actually completes~~ — **fully complete as of 2026-07-17**.
      Blocked in sequence by: Volta hardware incompatibility, the MoE `Llama4TextExperts`
      architecture bug, a GPU memory-imbalance OOM, and finally a too-tight system-RAM
      limit on the full run — each root-caused and fixed in turn (see sections above). The
      INT8 checkpoint is complete, verified byte-exact, and now lives in
      `s3://marcdo126967-triton-models/default/llama4scout-int8/weights/`.
- [x] Add `vision_model`/`multi_modal_projector` to the INT8 recipe's ignore list — done.
- [x] Switch to `llmcompressor==0.12.0` + `load_context()` + explicit `processor=` for
      MoE-aware calibration — done.
- [x] Switch to `device_map=None` + `sequential_targets=["Llama4TextMLP"]` (sequential
      onloading) — done, confirmed working end-to-end on `g6.24xlarge`.
- [ ] Serving the INT8 checkpoint, running the precision/accuracy benchmarks, and the
      `self_attn`/`router` ignore-list open question are all tracked in their own document
      now — see `docs/llama4scout_int8/benchmarking_plan.md`.
- [x] Terminate manual GPU nodes after each test that's actually finished — done for
      `i-041ad9f0da5893910`, `i-053f8c473a56a0122`, `i-07d999b017ee584f4` (`p4d.24xlarge`,
      from the earlier Volta/MoE investigation), and, from the 2026-07-16/17 session,
      `i-0f287430ca2404ad5` (`g6e.48xlarge`, used for both a failed quantization attempt and
      the FP8-serving latency test) and `i-02c3684b6c4de718a` (`g6.24xlarge`, used for the
      successful quantization run, terminated after it froze post-completion — see the
      memory-headroom finding above). No manually-launched GPU nodes are currently running.
- [x] ~~Next attempt should wait for AWS capacity, or pursue subnet expansion~~ — resolved
      differently than expected: switching to the sequential-onload fix meant quantization
      no longer needed an 8-GPU node at all, and the resulting smaller instance class
      (`g6.24xlarge`) found capacity on the very first hunt round. No subnet expansion was
      needed. (This finding does *not* extend to FP8 serving, which still genuinely needs
      an 8-GPU-class node for its CC≥8.9 + tensor-parallelism requirement — that hunt ran
      for 70+ rounds over 3+ hours without success before being abandoned in favor of
      reusing an already-provisioned node; see `docs/s3_fuse_latency_test_results.md`.)
- [x] ~~Push access working; this INT8 investigation is not yet committed~~ — done, this
      doc and all related script/infra changes are committed and pushed to
      `llama4scout-fp8-serving` (commit `4eae6f6`).
