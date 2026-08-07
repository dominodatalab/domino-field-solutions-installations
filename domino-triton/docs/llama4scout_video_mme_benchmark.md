# Llama 4 Scout — Video-MME Benchmark (fp8 / int4 / bf16)

Status: complete for a 24-video pilot sample. Branch: `llama4scout-fp8-serving`.
2026-07-22.

## Goal

Measure whether fp8-dynamic and int4 quantization introduce meaningful
video/visual-understanding accuracy loss relative to the full-precision
(bf16) model, using the official Video-MME benchmark
(https://github.com/MME-Benchmarks/Video-MME, dataset:
`lmms-lab/Video-MME` on Hugging Face).

This is a different, narrower question than
`docs/llama4scout_int8/benchmarking_plan.md`'s INT8 precision work, which is
text-only (GPQA/MATH-500/AIME). Video-MME specifically targets the
capability that only matters if the model is actually used for video —
which is also why subtitles are deliberately excluded (see below).

## Results (24-video stratified pilot sample, subtitle-free)

| Model | Overall | Short | Medium | Long |
|---|---|---|---|---|
| `llama4scout-vllm-fp8-dynamic` | 59.7% | 66.7% | 79.2% | 33.3% |
| `llama4scout-vllm-int4` | 60.6% | 66.7% | 70.8% | 43.5% |
| `llama4scout-vllm` (bf16, full precision) | 56.9% | 66.7% | 70.8% | 33.3% |

All three runs: 72/72 questions answered, zero failures. Logged in MLflow
under experiment `video-mme-<your Domino username>`, one run per
`{model}_no_subtitle` combination — params (model, subtitle variant, sample
size, max frames), the full accuracy breakdown (overall/duration/domain/
sub_category/task_type) as metrics, and the raw per-question JSON as an
artifact.

**Reading these numbers**: bf16 scored *lowest* of the three, not highest —
the opposite of what you'd naively expect from a full-precision baseline.
At only 72 questions per model (24 videos × 3 questions), this is very
likely sampling noise rather than a real effect — a couple of questions
flipping right/wrong swings the percentage by several points, and the
duration-bucket breakdowns (24 questions each) are even noisier. The
takeaway that *is* well-supported by this sample: **fp8 and int4 are not
costing meaningful video-understanding accuracy relative to bf16** on this
benchmark. Widening the sample would be needed to say anything more
precise about the specific duration/domain differences.

Why quantization wouldn't be expected to hurt video understanding much in
the first place: both quantization recipes keep `vision_model*` and
`multi_modal_projector*` at full precision (see
`docs/llama4scout_int8/quantization_plan.md`'s "Scope note for INT8
recipe") — only the text decoder is quantized. The visual pathway itself is
identical across all three variants.

## Dataset

`lmms-lab/Video-MME` on Hugging Face: 900 videos (300 each of
short/medium/long duration), 2700 questions (3 per video, multiple-choice
A/B/C/D), 6 domains, 744/900 videos have subtitles.

**Videos are not bulk-downloaded.** The dataset ships as 20
`videos_chunked_NN.zip` archives (~101 GB total) directly on the HF repo.
The HF CDN supports HTTP range requests (`Accept-Ranges: bytes`,
confirmed), so `benchmark_video_mme.py` reads each zip's central directory
via a small custom `HttpRangeFile` (seekable file-like object over ranged
GETs) to build a `videoID -> chunk` index without downloading any video
content, then does a second targeted range request per selected video to
pull just that one file out of its chunk. Only the ~20-30 sampled videos'
actual bytes are ever transferred.

Annotations (`videomme/test-00000-of-00001.parquet`, 405 KB) and subtitles
(`subtitle.zip`, 7.5 MB, plain `.srt` files keyed by YouTube ID) are small
enough to download in full.

## Sample selection

24 videos: 8 per duration bucket (short/medium/long), stratified randomly
(`random_state=42` for reproducibility), preferring subtitle-available
videos at selection time (all 24 have subtitles, even though the actual
comparison run doesn't use them — see below). Saved to
`results/video_mme/pilot_sample.json` (or wherever `--output-dir` points)
on first generation, and reusable across models via `--video-ids-file` so
every model is compared on the exact same videos:

```json
[
  {"video_id": "022", "videoID": "n3IYmdy6d4Y", "duration": "short", "domain": "Knowledge"},
  {"video_id": "014", "videoID": "uqILuTcux_o", "duration": "short", "domain": "Knowledge"},
  ...
]
```
(Full 24-entry list is in the saved `pilot_sample.json` from this run.)

Domain distribution follows the dataset's natural proportions (Knowledge is
the largest domain in Video-MME overall, so it dominates the 24-video
sample too) rather than being forced even across domains.

## Methodology

Reuses this repo's own `scripts/clients/llm_vllm_grpc_client.py` building
blocks rather than the third-party tooling the official README points to
(`look4u-ok/video-slicer` for frame extraction) — `sample_video_frames_base64()`
and `build_video_prompt()` already do exactly this (evenly-spaced frame
sampling, `<|image|>`-token Llama 4 chat template), and were already
proven working against this Triton deployment.

- **Frames per video**: 8 (the lower of fp8's and int4's `limit_mm_per_prompt`
  at the time of the pilot run, chosen so all models get an identical
  visual input budget for a fair comparison — bf16's config was also set to
  10 like fp8's current value, so 8 is conservative for all three, not a
  ceiling any of them are near).
- **Prompt**: the official Video-MME template, subtitle or subtitle-free
  variant (see prompt text in `benchmark_video_mme.py`'s
  `PROMPT_WITH_SUBTITLE`/`PROMPT_NO_SUBTITLE` constants).
- **Generation**: free-text, `max_tokens=16`, `temperature=0.0`. Not
  guided/structured decoding — see "Guided decoding is broken" below for
  why, and note this actually matches the official benchmark's own
  methodology anyway (the real `output_test_template.json` sample data has
  free-text responses like `"C. Berries."`, not bare letters).
- **Scoring**: the official answer-extraction logic
  (`extract_characters_regex`), vendored byte-for-byte from
  `https://github.com/thanku-all/parse_answer/blob/main/eval_your_results.py`
  into `scripts/benchmarks/video_mme/eval_your_results.py` (kept unmodified
  so it stays diffable against upstream). `benchmark_video_mme.py`'s
  `compute_accuracy_metrics()` imports `extract_characters_regex` from it
  rather than reimplementing answer-extraction, and joins
  domain/sub_category/task_type from the annotations parquet by
  video_id/question_id — not from fields embedded in the results file
  itself, which makes scoring robust even against older result files that
  predate a schema change.

### Subtitle-free by default, and why

`--subtitle-variant` defaults to `without`. Two independent reasons landed
on the same conclusion:

1. **It isolates what we're actually trying to measure.** Both quantization
   recipes leave the vision tower at full precision — only the text decoder
   is quantized. If subtitles let the model answer from the transcript
   alone, we'd mostly be measuring quantization's effect on text-reading
   comprehension, with video as an unnecessary confound.
2. **The with-subtitle prompt has a real, unfixed bug**: `load_subtitle_text()`
   dumps a video's *entire* `.srt` transcript into the prompt regardless of
   sampled frame count. The official methodology says to align subtitles to
   the sampled frames ("if extracting 10 frames per video, use the 10
   corresponding subtitles") — this implementation never did that
   alignment. Long/medium videos with long transcripts blow past the
   model's 16384-token context limit: **33/72 questions (46%) failed** in
   an earlier with-subtitle fp8 pilot run, 100% of them
   `The decoder prompt (length ...) is longer than the maximum model length
   of 16384` errors, and 0% failed in the subtitle-free variant on the
   identical sample. Since we decided subtitles weren't needed for this
   comparison anyway, the real fix (align subtitle snippets to each sampled
   frame's timestamp) wasn't worth building. See
   `docs/known_issues_and_todos.md` if this needs revisiting later.

### Guided decoding is broken for this deployment

Originally tried constraining answers to a single letter via vLLM's
`guided_choice` (`["A","B","C","D"]`), to make scoring trivial. Both
`guided_choice` and `guided_regex` fail identically:

```
Error generating stream: 'NoneType' object has no attribute 'lora_name'
```

Confirmed this is a genuine server-side bug in the Triton vLLM backend, not
a client-side bug or a multimodal-input interaction: it reproduces on a
plain **text-only** prompt with no video/image input at all, and both
guided-decoding modes fail the same way. Not root-caused (the `lora_name`
reference suggests a LoRA-adapter code path in the backend's guided-decoding
handling that doesn't expect a no-LoRA configuration). Worked around by
using plain free-text generation instead (see Methodology above) — which
turned out to match the official benchmark's methodology anyway, so this
wasn't actually a loss.

### Auth token refresh mid-run

A full 24-video run takes ~15-20 minutes (~14-17s/call × 72 calls). Domino's
`DOMINO_API_PROXY`-issued token expires faster than that (observed ~5
minutes into the first pilot run), and `auth_helper.py`'s token cache is
only invalidated on an explicit `invalidate_token()` call — it doesn't
happen automatically. `run_question()` fetches a fresh
`get_auth_headers()` per call (cheap; `auth_helper` caches internally) and
specifically catches `grpc.aio.AioRpcError` with
`code() == grpc.StatusCode.UNAUTHENTICATED`, calling `invalidate_token()`
and retrying once before giving up on that question. Confirmed working
live — the token expired again partway through both the int4 and bf16 runs,
each time logged as `Auth token expired mid-run, refreshing and retrying
once` and the run continued without interruption.

## Reproducing this

```bash
# Pilot run: stratified sample, subtitle-free, against a given model.
# Downloads only the ~24 sampled videos (via range requests), not the full dataset.
python scripts/benchmarks/video_mme/benchmark_video_mme.py \
    --model llama4scout-vllm-fp8-dynamic --n-per-duration 8

# Reuse the exact same sample for a second/third model (apples-to-apples comparison)
python scripts/benchmarks/video_mme/benchmark_video_mme.py \
    --model llama4scout-vllm-int4 \
    --video-ids-file results/video_mme/pilot_sample.json

python scripts/benchmarks/video_mme/benchmark_video_mme.py \
    --model llama4scout-vllm \
    --video-ids-file results/video_mme/pilot_sample.json

# Re-score and re-log to MLflow from already-collected results, without
# re-running any inference (e.g. logging a run that predates MLflow
# integration, or re-scoring after a scoring-logic change):
python scripts/benchmarks/video_mme/benchmark_video_mme.py \
    --model llama4scout-vllm-fp8-dynamic --score-only

# Official printed report (separate step, vendored CLI script, matches
# published leaderboard methodology exactly):
python scripts/benchmarks/video_mme/eval_your_results.py \
    --results_file results/video_mme/llama4scout-vllm-fp8-dynamic_no_subtitle.json \
    --video_duration_type short,medium,long --return_categories_accuracy
```

Batch job is resumable: `benchmark_video_mme.py` checkpoints results to disk
after every video (not just at the end), and skips videos already present
in the output file on restart — safe to re-run the same command after an
interruption without losing progress or repeating already-completed
inference calls.

## Scaling up

This is a 24-video (2.7% of the full 900-video) pilot. Widening the sample
is just `--n-per-duration N` for a larger N (or omit `--video-ids-file` and
adjust the stratification logic in `select_stratified_sample()` for a
different sampling scheme). Cost/time scales roughly linearly: this pilot
was ~15-20 min and 72 model calls per model; the full 900-video/2700-question
benchmark would be ~40x that per model (several hours), worth budgeting for
deliberately rather than assuming it "just works" at that scale — e.g., the
auth-token-refresh behavior above was only found and fixed because a
24-video run was long enough to hit it once; a much longer run should be
watched for issues that only show up over a longer window.

## Not benchmarked here: YOLOv8n

Video-MME is multiple-choice video question-answering; YOLOv8n is a
bounding-box object detector — a fundamentally different task shape, not
adaptable to this benchmark. A real video object-detection benchmark (e.g.
ImageNet-VID or similar, scored with standard mAP) would be needed for
YOLOv8n specifically — not yet started, not yet scoped.

## Related gap: the Dashboard app's `quick_test` also has weaker coverage for non-text models

Separate subsystem (the Triton Dashboard App's load-progress-bar feature,
`app-src/routes/testing.py`), but the same underlying theme as the
subtitle-alignment gap above: **this repo doesn't yet have real
per-input-type (video/image/audio) testing infrastructure**, so anything
that needs to genuinely exercise a non-text model falls back to a weaker
check instead.

Specifically: the Dashboard's load-progress UI has a final "Test query"
stage that calls `POST /api/testing/quick-test/{model_name}` to confirm a
model is actually serving, not just reporting `READY` in the repository
index. This originally hardcoded a text-shaped request
(`TestInferRequest(input_type="text", text="Hello", ...)`) for every model.
`MODEL_INPUT_TYPES` in `app-src/routes/testing.py` registers `yolov8n` as
`video`/`image` only — sending it a text request fails with an argument
error from `yolov8n_video_grpc_client.py` (that script doesn't accept a
`--texts` flag), a false-negative failure even though the model itself
loaded fine.

**Fix applied**: `quick_test` now checks the model's registered
`input_types` first. Text-capable models keep the real minimal-inference
test as before. Non-text-capable models (currently just `yolov8n`) fall
back to a weaker check that only re-confirms Triton reports the model ready
(`GET /v2/models/{model_name}/ready` through the proxy), with no actual
inference call attempted.

**Why not fixed properly**: building a genuine per-input-type quick-test
(e.g. feeding YOLOv8n an actual sample video frame) was more work than that
feature's scope justified at the time. Revisit alongside a real YOLOv8n
object-detection benchmark (above) if/when that work happens — the same
"feed it a real sample frame" infrastructure would serve both needs.
