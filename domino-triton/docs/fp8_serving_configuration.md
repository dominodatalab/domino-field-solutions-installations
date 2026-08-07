# FP8 Serving Configuration — What Worked, and Why It Was Nearly the Only Option

Status as of 2026-07-16. Branch: `llama4scout-fp8-serving`. Companion to
`docs/s3_fuse_latency_test_results.md` (the latency comparison this
configuration was used to produce) and `docs/s3_fuse_latency_test_plan.md`
(the original plan).

## Summary: the working configuration

- **Base container**: `nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3`
  (unmodified — no shared Dockerfile changes, see note at the end).
- **vLLM**: `0.9.2`, installed via `pip install` at pod startup, replacing
  the base image's bundled `0.8.1+nv25.04`.
- **Required pins alongside vLLM 0.9.2**:
  - `transformers==4.51.3`
  - `xformers==0.0.30`, installed with `--force-reinstall --no-deps`
    (a plain version pin is a no-op here — see below).
- **Model**: `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic`
  (`compressed-tensors` format), **not** NVIDIA's own
  `nvidia/Llama-4-Scout-17B-16E-Instruct-FP8` (`ModelOpt` format) — see
  "Why the checkpoint format matters" below.
- **`model.json`**: `tensor_parallel_size: 8`, `gpu_memory_utilization: 0.85`,
  `max_model_len: 16384`, `dtype: auto`. No `"quantization"` field —
  `compressed-tensors` checkpoints carry their own quantization metadata in
  `config.json` and vLLM auto-detects it (unlike `ModelOpt` checkpoints,
  which needed an explicit `"quantization": "modelopt"` hint).
- **Hardware**: `g6e.48xlarge` (8× NVIDIA L40S, Ada Lovelace, CC 8.9, 44.39
  GiB VRAM each). CC≥8.9 is a hard requirement for vLLM's FP8 tensor-core
  path regardless of quantization format.
- **`/dev/shm`**: pod needs an explicit `emptyDir { medium: Memory, sizeLimit: 32Gi }`
  mounted at `/dev/shm` — the Kubernetes default (64Mi) is far too small for
  8-way NCCL tensor-parallel communication and fails with a generic `NCCL
  error: unhandled system error` otherwise.

This combination is narrow, and getting here required ruling out several
plausible-looking alternatives one at a time. The rest of this document
explains why each one failed, so the reasoning doesn't have to be
rediscovered if this configuration needs to change later (e.g. if the base
Triton image is upgraded, or a newer vLLM/xformers/transformers release
changes the picture).

## Why the base image's default vLLM doesn't work at all

`nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3` ships `vllm==0.8.1+nv25.04`.
That build has **no native model class registered for
`Llama4ForConditionalGeneration`** — attempting to load any Llama4 Scout
checkpoint (regardless of quantization format) logs:

```
WARNING: Llama4ForConditionalGeneration has no vLLM implementation, falling
back to Transformers implementation. Some features may not be supported and
performance may not be optimal.
```

The fallback instantiates the raw HuggingFace `transformers` implementation,
which allocates each MoE layer's 16-expert weight tensor as one large,
un-sharded block per worker (`Llama4TextExperts.__init__` →
`torch.empty((16, 8192, 5120))`) rather than properly tensor-parallelizing
it. This OOMs on load, independent of storage backend or checkpoint choice —
confirmed by reproducing the identical failure on both the NVIDIA and
RedHatAI checkpoints, on two different storage backends. **Some vLLM upgrade
is mandatory**, not an optimization.

## Why we can't just pick any newer vLLM version

This is the part that made the search narrow. Four versions were tried, in
order, each ruled out for a different, specific reason:

| vLLM version | Native Llama4 support? | FusedMoE quant-method support? | Compatible with this Triton container? | Verdict |
|---|---|---|---|---|
| `0.8.1` (base image) | No | N/A | Yes (it's the base) | Falls back to broken generic path — see above |
| `0.9.2` | Yes | `compressed-tensors`: yes. `ModelOpt`: no (`get_quant_method` has no `FusedMoE` branch — confirmed via source) | **Yes** | **Works, for `compressed-tensors` checkpoints only** |
| `0.10.0` | Yes | Both: yes (`ModelOptFp8MoEMethod` added) | Yes | Native support exists, but a separate checkpoint-naming bug breaks the NVIDIA `ModelOpt` checkpoint specifically (see below) |
| `0.12.0` (vLLM's own documented version for this exact NVIDIA checkpoint) | Yes | Yes | **No** — Triton's bundled backend code (`backends/vllm/utils/metrics.py`) imports `vllm.engine.metrics`, a module removed by this version; also conflicts with `tritonfrontend`'s hard `pydantic==2.10.6` pin. Confirmed checkpoint-agnostic (same failure on both checkpoints) — a pure import-time incompatibility with this specific Triton container build. | Broken regardless of checkpoint |

So the viable window, given this specific Triton container, is roughly
**`vllm>=0.9.2` and `<0.12.0`** — below that, no native Llama4 support; at or
above `0.12.0`, Triton's own bundled integration code breaks. `0.9.2` is the
version actually used because it's the earliest in that window, and (for the
`compressed-tensors` checkpoint we ended up using) it's already sufficient —
there was no need to reach for `0.10.0` once the checkpoint format was
switched.

**If the Triton container image is ever upgraded** (e.g. to a version that
ships a vLLM backend already compatible with newer vLLM releases — Triton
25.12 ships vLLM 0.11.1 natively, for instance), this whole version-selection
problem should be revisited, since the constraint is specifically about
*this* Triton container's bundled glue code, not a fundamental vLLM
limitation.

## Why the checkpoint format matters: `compressed-tensors` vs `ModelOpt`

The original target was NVIDIA's own checkpoint,
`nvidia/Llama-4-Scout-17B-16E-Instruct-FP8` (`ModelOpt` format). It never
loaded successfully, on any vLLM version tried (`0.9.2` fails earlier, at
quant-method resolution; `0.10.0` and `0.11.1` get further but fail at
weight loading). The root cause, confirmed against the actual checkpoint
files and vLLM source:

- The checkpoint's per-expert scale tensors use underscore-joined names,
  e.g. `experts.down_proj_input_scale` (confirmed via
  `model.safetensors.index.json`).
- vLLM's fused-MoE weight-loading code
  (`vllm/model_executor/models/llama4.py`, `load_moe_expert_weights`) maps
  checkpoint names to registered parameters via a literal substring
  replacement, and **unconditionally appends `weight`** to the replacement
  prefix for every `down_proj`-related tensor — correct for the weight
  itself, but wrong for its scale tensors. The result,
  `w2_weight_input_scale`, never matches vLLM's actual registered parameter,
  `w2_input_scale`.
- This is a structural bug in the substitution logic, confirmed present at
  both `0.10.0` and `0.11.1` — not something a version bump alone fixes.

NVIDIA's own model card for this checkpoint only documents TensorRT-LLM and
SGLang as validated serving engines, not vLLM — vLLM support for this
checkpoint appears to be a community integration that hasn't been
fully validated against this exact export convention.

RedHatAI's `Llama-4-Scout-17B-16E-Instruct-FP8-dynamic`, by contrast, uses
`compressed-tensors` — vLLM's own native quantization format, with much
longer-established and more thoroughly exercised MoE support (present since
at least `0.9.2`, versus `ModelOpt`'s FP8+MoE support only landing at
`0.10.0`, with the naming bug above still unresolved as of the versions
checked). It loaded successfully on the first checkpoint-format switch, no
further debugging needed on the quantization-format side.

**Practical implication**: if the NVIDIA checkpoint is ever required
specifically (e.g. for a direct apples-to-apples comparison against
published NVIDIA benchmark numbers), this naming-mismatch bug would need
either an upstream vLLM fix, a newer vLLM version confirmed to have fixed
it (not yet verified — `0.12.0` was never checkpoint-tested against this
specific bug due to the separate Triton-compatibility failure), or a
one-time local patch renaming the checkpoint's scale tensors to match what
vLLM's substitution logic actually produces.

## vLLM has no native support for a text-only Llama4 checkpoint

vLLM 0.9.2's model registry (`vllm.model_executor.models.registry`) registers
exactly one Llama4-related class across both its `_MULTIMODAL_MODELS` and
`_VLLM_MODELS` dicts: `Llama4ForConditionalGeneration` (the full multimodal
wrapper, expecting a `Llama4Config` with nested `text_config`/`vision_config`).
There is no separate registration for a headless/text-only Llama4 decoder.

This matters because of a real bug encountered in the INT8 quantization
work (see `docs/llama4scout_int8/quantization_plan.md`): a checkpoint saved
via `AutoModelForCausalLM.from_pretrained(...)` on a Llama4 Scout source
materializes only the inner text submodule (`architectures:
['Llama4ForCausalLM']`, `model_type: llama4_text`) — no vision tower at
all. Such a checkpoint **cannot be served via vLLM regardless of config
changes**: vLLM's registry has no matching native class, and the generic
Transformers-backend fallback also fails, since `Llama4TextConfig` isn't
mapped to a plain `AutoModel` entry either (only `Llama4Config`, nested, is).
Any Llama4 Scout checkpoint intended for vLLM serving must be the full
multimodal export.

## Why the two additional pins are needed (architecture-level, not checkpoint-specific)

Both of these apply to **any** Llama4 Scout checkpoint on vLLM `0.9.2` in
this container — they're properties of the vision tower and the dependency
resolution at install time, unrelated to which quantization format is used.

### `transformers==4.51.3`

vLLM `0.9.2` only floors `transformers>=4.51.1`, with no ceiling. An
unpinned install resolves to the newest available `transformers` at install
time — currently `>=4.54.0`. `transformers` `4.54.0` (2025-07-08) added
native `aimv2` vision-model config registration, which collides with a
config name vLLM's own bundled Ovis-model config shim already registers
under the same name:

```
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
```

Pinning `4.51.3` (the version already present in the base image, confirmed
compatible with everything else) avoids the collision. This is a real,
currently-unresolved gap in vLLM `0.9.2`'s own dependency declaration (its
official `requirements/common.txt` also only floors this package, with no
ceiling), not something specific to this deployment.

### `xformers==0.0.30`, force-reinstalled

Llama4's vision tower (`Llama4VisionAttention` → `MultiHeadAttention`) has
hardcoded logic that converts any FlashAttention-eligible backend selection
into `XFORMERS` for its own attention implementation — this cannot be
overridden via the `VLLM_ATTENTION_BACKEND` environment variable, which only
affects the main decoder's attention backend selection (confirmed by
reading `vllm/attention/layer.py`; setting this env var to force an
alternative actually broke the main decoder instead, since not all backend
choices are valid there).

The base image already has an NVIDIA-patched `xformers` build tagged
`0.0.30+nv25.04`. A plain `pip install "xformers==0.0.30"` is a **no-op** —
pip treats the local-versioned build as already satisfying an exact-match
pin. That build's bundled Triton (the GPU-kernel compiler, unrelated to
Triton Inference Server) split-K kernel code still calls an old Triton JIT
API (`jitted_fn.src = new_src`), which the newer `triton==3.3.0` that
`vllm==0.9.2` also installs rejects:

```
AttributeError: Cannot set attribute 'src' directly. Use '_unsafe_update_src()'...
```

This is a known, currently-unfixed upstream bug in `xformers` itself
(`facebookresearch/xformers#1299`) — no version of the `0.0.30` line fixes
it in isolation. The genuine public `xformers==0.0.30` wheel (verified
directly against the `v0.0.30` tag on GitHub) already guards this exact call
with `hasattr(jitted_fn, "_unsafe_update_src")`, falling back to the new API
when available. `--force-reinstall --no-deps` is required specifically to
make pip actually replace the NVIDIA local build with the genuine public
wheel, since a plain pin is silently satisfied by the (broken) local build.

## Full command reference

```bash
python3 -m pip install --no-cache-dir "vllm==0.9.2" "transformers==4.51.3"
python3 -m pip install --no-cache-dir --force-reinstall --no-deps "xformers==0.0.30"
```

`model.json`:
```json
{
  "model": "/triton-repo/weights/llama4scout-vllm-fp8-dynamic",
  "tokenizer_mode": "auto",
  "dtype": "auto",
  "max_model_len": 16384,
  "gpu_memory_utilization": 0.85,
  "tensor_parallel_size": 8,
  "disable_log_requests": true,
  "enforce_eager": false
}
```

## A note on scope: no shared files were changed

This entire configuration lives in the per-pod startup command and
`model.json`/`config.pbtxt` — no changes were made to the shared
`Dockerfile.triton.vllm` (which pins the base image itself, last touched by
a different engineer in April 2025, unrelated to this branch) or any other
repo-wide infrastructure file. This was a deliberate choice: it keeps the
fix fully contained to this branch's own pod manifests rather than affecting
whatever else in the repo uses that shared base image.

## Fragility this configuration depends on

This is a narrow, version-pinned configuration, not a general-purpose
recipe. It would need to be revisited if any of the following changes:

- **The base Triton container image is upgraded** — the vLLM version
  constraint above is specifically tied to this container's bundled backend
  code (`tritonfrontend`'s pydantic pin, `vllm.engine.metrics` import). A
  different Triton release would need this whole analysis redone.
- **A newer `transformers` release** could reintroduce the `aimv2`-style
  collision for a different config name, or vLLM `0.9.2`'s floor could shift
  if PyPI metadata changes (unlikely, since published package metadata is
  immutable, but worth noting the pin is deliberately below the collision
  point, not an arbitrary older choice).
- **A newer `xformers` release** may or may not fix the underlying Triton
  JIT API bug — if it does, the `--force-reinstall --no-deps` step could
  potentially use a newer, non-NVIDIA-shadowed version instead of needing
  the exact `0.0.30` public wheel specifically.

## Redeploying this configuration

As of 2026-07-17, everything needed to redeploy this exact configuration is
committed to the repo (previously, the actual pod manifests had drifted out
of sync with this doc — the committed YAML still pointed at the abandoned
NVIDIA/ModelOpt checkpoint attempt and was missing the vLLM version-pin step
entirely; this has been corrected):

1. **Model config**: `triton-repo-reference/models/llama4scout-vllm-fp8-dynamic/`
   (`config.pbtxt` + `1/model.json`) — the Triton model definition.
2. **Pod manifest**: `scripts/benchmarks/triton_latency_s3fuse_pod.yaml` — the
   proven working pod (S3-FUSE, the recommended storage backend per
   `docs/s3_fuse_latency_test_results.md`). Includes the full vLLM
   0.9.2/transformers 4.51.3/xformers 0.0.30 fix inline.
3. An EFS variant also exists
   (`scripts/benchmarks/triton_latency_efs_pod.yaml`) for repeating the
   storage-backend comparison itself — not recommended for actual serving,
   and its weights `subPath` is a placeholder that needs to point at wherever
   the checkpoint is staged on EFS at deploy time.

To redeploy:

```bash
kubectl create configmap triton-s3fuse-fp8-dynamic-config -n domino-compute \
  --from-file=config.pbtxt=triton-repo-reference/models/llama4scout-vllm-fp8-dynamic/config.pbtxt \
  --from-file=model.json=triton-repo-reference/models/llama4scout-vllm-fp8-dynamic/1/model.json

kubectl apply -f scripts/benchmarks/triton_latency_s3fuse_pod.yaml
```

This assumes the `llama4scout-fp8-dynamic-s3-pvc` PVC (S3 CSI-backed,
pointed at the checkpoint's S3 prefix) already exists — it was created for
the original latency test and left in place since it costs nothing when
idle (S3 storage cost is billed regardless of the PVC object). If it's ever
deleted, it needs recreating against the S3 prefix documented in
`docs/llama4scout_int8/quantization_plan.md`'s FP8 checkpoint location (or
wherever the checkpoint currently lives — check S3 directly before assuming).
