# Llama 4 Scout — INT8 Requantization Report

Status: complete. 2026-07-18. Branch: `llama4scout-fp8-serving`. This documents
the specific run that produced the corrected, full-multimodal INT8
checkpoint now at `s3://marcdo126967-triton-models/default/llama4scout-int8/weights/`.

For the earlier history (the original text-only checkpoint bug, the
missing-vision-tower root cause, and the SmoothQuant mapping-ambiguity bug),
see `docs/llama4scout_int8/quantization_plan.md`. This doc covers only the
final, corrected run: its exact configuration and the error it hit.

## Why this run was needed

The first INT8 checkpoint (produced by an earlier run of this same
pipeline) was discovered to be **text-only** — `scripts/quantize/quantize_llama4scout.py`
loaded the source model via `AutoModelForCausalLM.from_pretrained(...)`,
which silently materializes only the inner text-decoder submodule for a
Llama4 Scout checkpoint. The vision tower and `multi_modal_projector` were
never loaded, so they were never saved, regardless of the recipe's
`ignore=[...]` list (which only controls what's excluded *from quantization*,
not what's loaded in the first place). This is a real requirement for the
target deployment, which needs video/image processing capability — see
project context for why this was caught and required a full redo.

## Exact configuration used

**Base image**: `nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3` (unmodified).

**Dependencies** (installed together in one command — see
`docs/llama4scout_int8/quantization_plan.md` / the deployment history for why
`torch`/`torchvision` must move together with `transformers` in this
container, rather than being partially pinned):

```bash
pip install "llmcompressor==0.12.0" datasets torchvision
```

**Hardware**: `p4d.24xlarge` (8× A100-40GB, 96 vCPU, ~1152 GiB system RAM),
found via a capacity hunt across `g6e.48xlarge` → `p4de.24xlarge` →
`p4d.24xlarge` → `p5.48xlarge` (11 rounds, ~1hr). This tier was chosen after
two earlier attempts on a `g6e.12xlarge` (~369 GiB allocatable) failed on
insufficient system memory — see "Prior failures on this exact fix" below.

**Pod resources**:
```yaml
resources:
  limits:
    nvidia.com/gpu: "8"
    memory: "946Gi"    # 85% of ~1113 GiB allocatable on this node
    cpu: "91"
```

**Model loading fix** (`scripts/quantize/quantize_llama4scout.py`):
```python
from transformers import AutoTokenizer, AutoModelForImageTextToText
...
model = AutoModelForImageTextToText.from_pretrained(
    str(source_dir),
    torch_dtype=torch.bfloat16,
    device_map=None,
)
```
`device_map=None` + `load_context()` + `sequential_targets=["Llama4TextMLP"]`
for sequential CPU-onload calibration, matching the approach already
validated for the (previously) text-only run — this part of the recipe was
correct all along; the bug was purely which model class loaded the source.

**Quantization recipe** (INT8, W8A8):
```python
[
    SmoothQuantModifier(
        smoothing_strength=0.8,
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
        ignore=["lm_head", "vision_model*", "multi_modal_projector*"],
    ),
]
```
The explicit, `language_model`-scoped `mappings` on `SmoothQuantModifier` were
required once the vision tower actually loaded (see below) — the default
mapping for `Llama4ForConditionalGeneration` uses unscoped regexes that match
both the text decoder's and the vision encoder's identically-named layers
(`post_attention_layernorm`, `q_proj`, etc.), causing a
`SmoothQuant must match a single smooth layer for each mapping` crash.
Scoping to `language_model.*` resolves the ambiguity and matches the
recipe's own intent — SmoothQuant should only touch what's actually being
quantized.

**Calibration**: `allenai/c4` (English), 512 samples, `max_seq_length=2048`,
`num_calibration_samples=512`.

## Timeline

| Phase | Duration |
|---|---|
| Dependency install | ~2 min |
| Linearization (MoE expert restructuring, 48 layers) | ~28 min |
| Calibration (49 sequential steps: 48 decoder layers + 1) | ~8h |
| **Total wall-clock** | **~8h24m** (17:25 CDT Jul 17 → 01:50 CDT Jul 18) |

## The error

The run reached the **final calibration step (49/49)** before crashing:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 MiB.
GPU 0 has a total capacity of 39.49 GiB of which 3.56 MiB is free.
Including non-PyTorch memory, this process has 39.04 GiB memory in use.
...If reserved but unallocated memory is large try setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

**Root cause**: sequential CPU-onload calibration only actively uses **one**
GPU at a time (confirmed via `nvidia-smi` during the run — GPU 0 showed real
usage throughout, GPUs 1-7 sat at ~0 MiB the entire run). Over 49 sequential
layer-cycles, GPU 0's CUDA memory allocator accumulated enough fragmentation
that it could no longer satisfy a 40 MiB allocation despite having a large
total capacity — a classic PyTorch caching-allocator fragmentation failure,
not a true out-of-memory condition (total usage was 39.04 GiB out of 39.49
GiB, i.e. fragmented rather than genuinely exhausted). The traceback
originated in `compressed_tensors/offload/utils.py`'s `send_tensors`, a
device-transfer call inside `llmcompressor`'s own post-calibration
compression/save pipeline — i.e., this happened *after* all 49 layers'
calibration math had already completed successfully, during internal
cleanup/consolidation.

**Why this didn't cost the whole run**: `llmcompressor`'s `oneshot()`
(called with `output_dir=...`) writes the quantized model to disk as part of
its own internal pipeline, not as one final `save_pretrained()` call at the
very end. By the time the crash occurred, it had already written:

- All 3 safetensors shards (102.2 GiB total, byte-exact match against
  `model.safetensors.index.json`'s declared `total_size`)
- `config.json` (confirmed `model_type: llama4`,
  `architectures: ['Llama4ForConditionalGeneration']`, both `vision_config`
  and `text_config` present)
- `model.safetensors.index.json` (6,238 tensors: 760 `vision_model`, 2
  `multi_modal_projector`, 5,476 `language_model` — confirming the vision-tower
  fix worked)
- `generation_config.json`, `recipe.yaml`

The only thing missing was the tokenizer, which `quantize_llama4scout.py`
saves as its own explicit last step (`tokenizer.save_pretrained(...)`,
called after `oneshot()` returns) — a step the crash prevented from ever
running.

**Recovery**: rather than re-running the full ~8.5-hour calibration, the
tokenizer files (`tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`,
`special_tokens_map.json`, `chat_template.jinja`) were copied directly from
the BF16 source checkpoint — the tokenizer itself is unaffected by
quantization, so this is exactly equivalent to what
`tokenizer.save_pretrained()` would have produced, with one added benefit:
copying the source files directly (rather than letting a fresh call
re-serialize them) also avoided the earlier `tokenizer_class:
TokenizersBackend` bug from the first (broken) INT8 run, which was an
artifact of re-serialization, not present in the original source files.
Confirmed via a direct `AutoTokenizer.from_pretrained()` load-and-encode
test against the completed checkpoint.

## Prior failures on this exact fix (for context)

Before landing on the above, this same vision-tower fix hit two node-level
memory failures on smaller hardware:

1. `g6e.12xlarge` (~369 GiB allocatable), 340Gi pod memory limit —
   **OOMKilled** at layer 2/49 of calibration. The full multimodal model's
   larger working set needed more headroom than the original text-only run.
2. Same node, memory limit raised to 365Gi (~99% of allocatable) — the
   **node itself froze** (`kubelet stopped posting node status`) partway
   through calibration, the same failure signature seen once before during
   the original (text-only) run's aftermath, but this time during the run
   itself, not after.

Both required terminating the instance. The move to `p4d.24xlarge`'s ~1152
GiB system RAM (946Gi pod limit, ~15% headroom) resolved the system-RAM
class of failure entirely — the final crash was a different resource (GPU
VRAM fragmentation on a single GPU), not system RAM.

## Final artifact

`s3://marcdo126967-triton-models/default/llama4scout-int8/weights/`
— 12 files, ~102.2 GiB, verified complete and correct:
full multimodal `Llama4ForConditionalGeneration`, INT8 W8A8 quantized
text decoder (vision tower and `lm_head` kept at full precision per the
recipe), working tokenizer.
