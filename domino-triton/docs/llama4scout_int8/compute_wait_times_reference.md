# Compute Wait-Times Reference — For Future Deployment Planning

Status as of 2026-07-17. Compiled from direct observation during the
Llama 4 Scout INT8 quantization (`docs/llama4scout_int8/quantization_plan.md`)
and FP8 serving latency (`docs/s3_fuse_latency_test_plan.md`,
`docs/s3_fuse_latency_test_results.md`) work. Intended as a planning
reference for estimating timelines on future GPU-heavy deployments in this
environment.

## Wait-times by category

| Category | Purpose | Observed wait | Notes |
|---|---|---|---|
| EC2 capacity — large/scarce tier (`p4de.24xlarge`, `g6e.48xlarge`, `p5.48xlarge`, `p5en.48xlarge`) | Needed when a job requires 8 GPUs in one node (aggregate VRAM, or FP8 serving's CC≥8.9 + 8-way tensor parallelism) | Quantization: succeeded at round 42 (~3.5+ hrs active polling, plus AWS session-expiry gaps). FP8-serving: never succeeded across 31+ rounds (~3+ hrs) before we abandoned it. | Highly scarce, unpredictable — hours, or may not appear in a practical window at all. |
| EC2 capacity — mid-tier (`g6.24xlarge`, `g5.24xlarge`) | Opened up once quantization was reshaped to need system RAM, not massive GPU VRAM | Round 1 — essentially instant (<1 min) | Night-and-day difference from the 48xlarge tier. |
| Node join / GPU registration | K8s node joining cluster with GPUs schedulable | Well under the 10-min timeout allowed; never blocked either successful hunt | Minor, predictable overhead once capacity is secured. |
| Model download (HuggingFace) | Getting a new checkpoint onto disk | ~107GB checkpoint: ~18.5 min | Scales with checkpoint size / HF bandwidth. |
| Storage copy (EFS ⟷ S3) | Moving/staging a checkpoint between backends | ~103-107GB: ~5-10 min via `cp`. 200GB BF16: ~80+ min (many-small-file S3-FUSE-source overhead added to raw size). | Budget real time for multi-hundred-GB checkpoint moves. |
| Container/dependency install at pod startup | Installing the working vLLM/transformers/xformers stack | Several minutes per fresh pod | Recurs every time; worth baking into a custom image for repeated workflows. |
| Model load time (serving) | Cold-load from storage into GPU memory | S3-FUSE: 807.88s (~13.5 min). EFS: 937.42s (~15.6 min). | Specific to ~107GB FP8 checkpoint on 8×L40S; scales with checkpoint size. |
| Quantization: linearization | One-time MoE architecture conversion before calibration | ~31 min (48 layers × ~40s/layer, after EFS fix) | Fixed cost, independent of sample count. Was ~5.3 hrs projected before the storage fix. |
| Quantization: calibration | The actual quantization compute | ~14.5 min/layer × 49 layers ≈ ~11-12 hrs (512 samples) | Largest single time cost in the whole engagement. |

## Takeaways for future deployments

1. **Instance size class matters more than raw need might suggest.** If a
   workload can be redesigned to fit in ≤4 GPUs (as sequential onloading let
   us do for quantization), expect near-instant capacity vs.
   multi-hour-to-never for 8-GPU nodes.
2. **Budget hours, not minutes, for large-instance-class hunts** if the
   workload truly needs 8 GPUs (e.g., FP8 serving's CC≥8.9 + tensor-parallel
   requirement) — there's no shortcut there, only patience or a fallback
   plan.
3. **The quantization calibration step is the dominant cost** (~12 hours) —
   if faster turnaround matters more than the memory savings, that's the
   tradeoff to revisit first (e.g., using GPU-resident processing on
   bigger/scarcer hardware instead of the CPU-based sequential path).
4. **Container/dependency setup recurs on every fresh pod** — worth baking a
   validated version stack into a custom image if this becomes a repeated
   workflow, rather than reinstalling each time.
