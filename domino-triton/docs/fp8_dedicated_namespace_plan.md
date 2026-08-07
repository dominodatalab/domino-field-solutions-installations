# FP8 Dedicated Namespace/Release: Plan, Reasoning, and Status

This documents an effort (started and paused mid-execution on 2026-07-20) to serve Llama 4 Scout FP8 (`llama4scout-vllm-fp8-dynamic`) via its own dedicated Helm release/namespace, separate from the shared small-model Triton deployments (`triton-demo`, `domino-inference-dev`). Paused in favor of reusing the already-proven standalone-pod configuration (`scripts/benchmarks/triton_latency_s3fuse_pod.yaml`) to move faster on an actual test. Kept here in case the dedicated-namespace approach is worth finishing later.

## Why a separate namespace, not just bumping `triton-demo`

The existing small-model deployments run **one Triton process serving every loaded model** in that namespace, sized for `gpu: 1` / modest memory. FP8 needs 8 GPUs and ~700Gi memory — orders of magnitude more. Two concrete problems with just bumping the shared StatefulSet's resources to fit FP8:

1. **Every model would then need an 8-GPU node to schedule at all**, even trivial ones like BERT, since there's only one pod/process per namespace serving everything.
2. **Shared blast radius** — an FP8-related crash (NCCL error, vLLM engine issue) would take down small-model serving too, since they're the same process.

So: separate namespace → separate Helm release → separate StatefulSet, fully isolating the two. Still meant to be visible/interactive from the *same* dashboard app though — via a new entry in its `namespaces.json`-style config (a dropdown/selector) — not a separate app.

## What was actually done, in order

1. **Fixed `deployment_prefix` bug** in `values-dev.yaml`/`values-test.yaml` — hardcoded to `sameerw116894` (another engineer's Fleetcommand deployment), confirmed present on `main` too. This value is baked into the EDV's S3 mount path, so it wasn't cosmetic. Fixed to `marcdo126967`.
2. **Fixed the same class of bug** in `persistence.s3.bucket`/`iamRoleArn` — pointed at `domino-triton-s3-edv`/`domino-triton-s3-role` instead of the actually-working `marcdo126967-triton-models`/`marcdo126967-triton-s3-role`.
3. **Removed dead code**: the abandoned NVIDIA ModelOpt FP8 checkpoint path (`download_llama4scout_fp8.py`, `triton-repo-reference/models/llama4scout-vllm-fp8/`) — confirmed never worked (a vLLM weight-name substitution bug drops MoE scale tensors). Fixed a stale `--model` default in `s3_fuse_latency_bench.py` that still pointed at it.
4. **Corrected INT4 checkpoint docs** after actually running the download (28GB claimed → 61GB actual), added S3 staging manifests, fixed the seed job to reuse the existing `hf-token` secret instead of creating a redundant one.
5. **Created `values-llama4scout-fp8.yaml`** — a dedicated (not environment-tier) values file: `namespace: domino-inference-llama4scout`, `gpu: 8`, `memory: 700Gi`, `cores: 80`, a new `dshm_size: 32Gi` (added as a proper parameterized chart value — `/dev/shm` was hardcoded to `10Gi` in `triton-statefulset.yaml`; parameterized with a backward-compatible default so other deployments are unaffected), matching bucket/role/prefix, `istio.enabled`/`ambient.enabled: true` (matching the known-working `triton-demo`).
6. **`helm install`ed** this into the new `domino-inference-llama4scout` namespace.
7. **Hit a hard capacity wall**: the shared `gpu` Karpenter NodePool caps at 4 GPUs (`g5.2xlarge`/`g5.12xlarge`/`g5.24xlarge`/`g6.12xlarge`) with aggregate limits (`cpu: 80`, `memory: 320Gi`, `nvidia.com/gpu: 10`) far too small for even one 8-GPU node.
8. **Researched via Domino's internal Confluence** (Karpenter runbook + node-group runbook): confirmed the existing 4 NodePools (`compute`/`gpu`/`platform`/`trainium`) were all created by Fleetcommand's deployer at the exact same bootstrap timestamp as the `EC2NodeClass` — not something application repos manage. The "proper" way to add capacity is a PR against `cerebrotech/domino-cloud-configs` through the deployer pipeline; direct `kubectl` edits to a NodePool are live-only and get reverted on the deployment's next scheduled infra upgrade.
9. **Created a dedicated `llama4scout-fp8` NodePool** as a deliberate, documented stopgap (`scripts/infra/karpenter/llama4scout-fp8-nodepool.yaml`) — `g6e.48xlarge`/`p5.48xlarge` only, its own limits (`cpu: 200`, `memory: 1600Gi`, `nvidia.com/gpu: 16`), reusing the same `EC2NodeClass`, kept separate from the shared `gpu` pool so small-model serving stays untouched. Updated `values-llama4scout-fp8.yaml`'s `node_selector` to match.
10. **Ran a manual EC2 hunt script** (`hunt_fp8_node.sh`, adapted from the earlier INT8-era hunt script, corrected to only target CC≥8.9 instance types — `g6e.48xlarge`/`p5.48xlarge`, excluding `p4d`/`p4de` which are CC 8.0 and don't meet FP8's tensor-core requirement). Took ~2 hours (22 rounds of `InsufficientInstanceCapacity`) to land a `g6e.48xlarge` (`i-07e4544b7aadb8884`, node `ip-10-0-51-247.us-west-2.compute.internal`).
11. **Manually relabeled the joined node** from `default-gpu` (baked into the reused `nodeadm-userdata.yaml`) to `llama4scout-fp8` to match the dedicated NodePool's label.
12. **Staged the FP8-dynamic model's weights + config** into the new namespace's actual S3 prefix (`marcdo126967/domino-inference-llama4scout/triton-repo/{weights,models}/llama4scout-vllm-fp8-dynamic/`) via a disposable staging PVC + Job in `domino-compute` (PVCs can't be mounted cross-namespace, so this bridges the gap). Copied ~109GB S3-to-S3 in under 5 minutes — no HuggingFace re-download needed.
13. **Hit an unresolved networking issue**: the proxy pod's outbound auth-validation call to `nucleus-frontend.domino-platform` timed out. Diagnosed that the new namespace was missing the `istio.io/dataplane-mode: ambient` label (and other Domino-specific labels) that both `triton-demo` and `domino-platform` have — `--create-namespace` only does a bare namespace creation, not the full label set a properly-provisioned namespace gets. Added the missing labels and restarted the proxy/admin pods to pick up ambient enrollment — **this did not fix it** (got worse: a full connection timeout instead of the earlier auth-backend-specific one). Root cause not resolved.
14. **Also discovered**: the actual dashboard app's `NAMESPACES_FILE` (confirmed via the App's live pod spec) points at `/mnt/data/triton-server-testing/namespaces.json` — a Domino **Dataset** file entirely outside git version control, not any of the three `namespaces*.json` files committed in `app-src/`. Never got to actually add the new namespace entry there — was in the middle of coordinating access to Marc's project dataset when priorities shifted.

## Status update (2026-07-21): revived, fixed, and fully working

This was picked back up and finished. `llama4scout-vllm-fp8-dynamic` is now
loading and serving successfully through this dedicated namespace/Helm
release, visible and testable from the actual dashboard app (not just a
one-off script) — including video/multi-frame input.

The previously-unresolved proxy→`nucleus-frontend` networking issue did
**not** reproduce this time — a direct test (a plain curl pod in both
`triton-demo` and `domino-inference-llama4scout`) got an identical response
from both, and the real proxy/admin pods came up with no auth errors in
their logs. It's possible the original failure was a transient
ambient-mesh-enrollment timing issue on a freshly-labeled namespace, not a
structural problem — but this wasn't root-caused further since it simply
stopped happening.

Three separate, real bugs were found and fixed getting the model to
actually load (all in `helm/domino-triton/templates/triton-statefulset.yaml`,
scoped via a new `vllm_version_override` values field so only this
deployment pays the cost — see git history on this branch for the full
diffs and reasoning):

1. **CUDA fork crash** (`Cannot re-initialize CUDA in forked subprocess`) —
   vLLM's tensor-parallel workers default to Python's `fork` start method,
   which can't coexist with a parent process that's already touched CUDA.
   Fixed with `VLLM_WORKER_MULTIPROC_METHOD=spawn` (harmless for every
   deployment, not scoped).
2. **OOM from the base image's un-upgraded vLLM** — same root cause
   `docs/fp8_serving_configuration.md` already documented (`vllm==0.8.1`
   has no native Llama4 support, falls back to a generic loader that
   doesn't shard MoE experts across GPUs). Fixed by upgrading to
   `vllm==0.9.2`/`transformers==4.51.3`/`xformers==0.0.30` at pod startup,
   gated behind `vllm_version_override` so `triton-demo`/dev/test are
   unaffected.
3. **Container killed mid-install by the liveness probe** — the pip
   install above takes several minutes, but the existing `livenessProbe`
   (30s delay, 3×10s failures) doesn't tolerate that. Added a
   `startupProbe` with a ~15 min budget.

Two more gaps were closed to make this actually reproducible from scratch
(previously only true "by hand," from this session's live debugging):

- **`scripts/infra/namespace-llama4scout.yaml`** — the namespace's critical
  labels (`istio.io/dataplane-mode: ambient`, `marcdo126967-owned`,
  `domino-triton`) were applied by hand at some point and never captured
  anywhere. `helm install --create-namespace` does not set any of them.
  Missing the ambient-mode label specifically is the leading suspect for
  the original networking failure above — apply this manifest *before*
  `helm install`.
- **`limit_mm_per_prompt` added to the download scripts themselves** —
  the video/multimodal fix only existed in the live S3 data and the
  git-tracked *reference* config, never in
  `download_llama4scout_fp8_dynamic.py`/`download_llama4scout_int4.py`
  themselves. Both now write it via a new `--limit-mm-images` flag
  (default 10, matching the dashboard's Max Frames UI default).

**Still manual, not yet scripted**: applying the dedicated `llama4scout-fp8`
NodePool is still a separate `kubectl apply` step, not folded into the Helm
release itself. This is *not* at risk of being silently reverted by
Fleetcommand infra upgrades, though (see the corrected note in
`docs/known_issues_and_todos.md` — per Marc Doan, per-deployment compute
customization is done via EC2NodeClass additions, and separate/distinctly-
named NodePool objects aren't touched by the deployer's reconciliation).
GPU capacity availability is also inherently variable (took 35 minutes this
time, ~2 hours previously) — there's no way to script around that, only to
not be blocked waiting on it (Helm install can happen while the capacity hunt is
still in progress, since only the Triton pod itself needs the GPU node).

## Bring-up from scratch (validated 2026-07-21)

In order, assuming the S3-staged weights already exist (if not, run the
download script with `--base-dir` pointing at a mounted destination first):

```bash
# 1. Namespace, with the labels a fresh --create-namespace would not set
kubectl apply -f scripts/infra/namespace-llama4scout.yaml

# 2. Dedicated GPU NodePool (live-only, not in the deployer's own config repo)
kubectl apply -f scripts/infra/karpenter/llama4scout-fp8-nodepool.yaml

# 3. Helm release -- proxy/admin/state pods come up immediately; the Triton
#    pod itself stays Pending until Karpenter lands a node (independent of
#    step 4, can run in parallel)
helm install domino-triton helm/domino-triton/ \
  -n domino-inference-llama4scout --create-namespace \
  -f helm/domino-triton/values-llama4scout-fp8.yaml

# 4. Load the model once Triton reports the pod Running
curl -X POST http://<proxy-or-admin-url>/v2/repository/models/llama4scout-vllm-fp8-dynamic/load
```

Expect: ~a few minutes to tens of minutes for GPU capacity (variable, not
scriptable), ~5 min for the vLLM/xformers pip install on first pod start
(baked into the pod's startup command, not the image itself — see the
`vllm_version_override` values field), then ~10-13 min for the actual model
load (weight download from S3, then vLLM engine warmup). Total: budget at
least 20-30 minutes end to end even with GPU capacity immediately
available.

## What changed the plan

Partway through, the actual task requirement changed — the immediate need became a direct, one-off test (video summarization via gRPC), not a durable, dashboard-visible deployment. Given the unresolved networking issue and the need to move quickly, the pragmatic call was to abandon the dedicated-namespace path for now and reuse the already-proven standalone pod (`triton_latency_s3fuse_pod.yaml`) instead, repurposing the same already-hunted node. (This was later revisited and finished — see the status update above.)
