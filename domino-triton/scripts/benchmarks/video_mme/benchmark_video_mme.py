#!/usr/bin/env python3
"""
Video-MME Benchmark - Multimodal video understanding eval for Llama 4 Scout variants

Runs the official Video-MME benchmark (https://github.com/MME-Benchmarks/Video-MME)
against a Triton-served Llama 4 Scout variant (fp8, int4, bf16), reusing this repo's
own llm_vllm_grpc_client.py for frame sampling and inference. Not applicable to
YOLOv8n (object detection, not video QA) -- see docs/known_issues_and_todos.md.

Dataset (lmms-lab/Video-MME on Hugging Face): videos are pulled directly out of
the 20 remote videos_chunked_NN.zip archives via HTTP range requests (the CDN
supports Accept-Ranges), so only the specific sampled videos are downloaded --
not the full ~101GB dataset.

Output matches the official eval_your_results.py's expected input schema
(vendored in this same directory) -- one JSON file per (model,
subtitle-variant) combination. After each run, accuracy is also computed
(reusing the vendored script's exact answer-extraction logic) and logged as
an MLflow run in Domino's Experiment Manager -- params (model, subtitle
variant, sample size), metrics (overall/duration/domain/sub_category/
task_type accuracy), and the raw results JSON as an artifact. Pass
--no-mlflow to skip this (the results files are written either way).

Default is subtitle-free (--subtitle-variant without) -- both quantization
recipes keep the vision tower at full precision, so subtitles would let the
model answer from text alone, confounding the video/visual-understanding
comparison this benchmark is actually for. Also sidesteps a real bug: the
"with subtitle" prompt dumps the entire transcript regardless of sampled
frame count, overflowing the model's context window for ~46% of medium/long
videos in the fp8 pilot (see docs/known_issues_and_todos.md) -- the correct
fix (align subtitles to sampled frame timestamps) wasn't worth building once
we decided subtitles weren't needed for this comparison anyway.

Usage:
    # Pilot run: stratified sample, subtitle-free, against fp8
    python scripts/benchmarks/video_mme/benchmark_video_mme.py \\
        --model llama4scout-vllm-fp8-dynamic --n-per-duration 8

    # Reuse an already-selected sample (e.g. for a second model, same videos)
    python scripts/benchmarks/video_mme/benchmark_video_mme.py \\
        --model llama4scout-vllm-int4 --video-ids-file results/video_mme/pilot_sample.json

    # Official printed report (separate step, for the vendored CLI script)
    python scripts/benchmarks/video_mme/eval_your_results.py \\
        --results_file results/video_mme/llama4scout-vllm-fp8-dynamic_no_subtitle.json \\
        --video_duration_type short,medium,long --return_categories_accuracy
"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import mlflow
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "clients"))
from auth_helper import get_auth_headers, invalidate_token  # noqa: E402
from llm_vllm_grpc_client import (  # noqa: E402
    build_video_prompt,
    infer_non_streaming,
    sample_video_frames_base64,
)
import grpc  # noqa: E402
import tritonclient.grpc.aio as grpcclient  # noqa: E402
from grpc.aio import AioRpcError  # noqa: E402
from tritonclient.utils import InferenceServerException  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_your_results import extract_characters_regex  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HF_REPO = "lmms-lab/Video-MME"
CHUNK_URL_TMPL = "https://huggingface.co/datasets/{repo}/resolve/main/videos_chunked_{i:02d}.zip"
NUM_CHUNKS = 20

REPO_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "video_mme"

PROMPT_WITH_SUBTITLE = (
    "This video's subtitles are listed below:\n{subtitles}\n"
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n"
    "The best answer is:"
)
PROMPT_NO_SUBTITLE = (
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n"
    "The best answer is:"
)


class HttpRangeFile(io.IOBase):
    """Seekable file-like object over HTTP range requests, for zipfile.ZipFile
    to read a remote archive's central directory and individual entries
    without downloading the whole file."""

    def __init__(self, url: str, session: requests.Session):
        self.url = url
        self.session = session
        self.pos = 0
        r = session.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        self.size = int(r.headers["content-length"])

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = (self.size - 1) if (n is None or n < 0) else (min(self.pos + n, self.size) - 1)
        if self.pos > end:
            return b""
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True


def build_chunk_index(cache_path: Path) -> dict:
    """Map videoID (YouTube ID, matches parquet's `videoID` column and the
    subtitle zip's filenames) -> which chunk .zip it lives in. Built from the
    20 zips' central directories only (a few HEAD+range requests each), not
    from downloading any video content."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    session = requests.Session()
    index = {}
    for i in range(1, NUM_CHUNKS + 1):
        url = CHUNK_URL_TMPL.format(repo=HF_REPO, i=i)
        z = zipfile.ZipFile(HttpRangeFile(url, session))
        for name in z.namelist():
            if name.endswith(".mp4"):
                vid = name.rsplit("/", 1)[-1].removesuffix(".mp4")
                index[vid] = {"chunk": i, "name": name}
        logger.info(f"Indexed chunk {i:02d}/{NUM_CHUNKS} ({len(index)} videos so far)")

    cache_path.write_text(json.dumps(index))
    return index


def fetch_video(video_id: str, chunk_index: dict, dest_dir: Path, session: requests.Session) -> Path:
    """Extract a single video's mp4 from its remote chunk zip via range
    requests, caching it locally under dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{video_id}.mp4"
    if dest.exists():
        return dest

    entry = chunk_index[video_id]
    url = CHUNK_URL_TMPL.format(repo=HF_REPO, i=entry["chunk"])
    z = zipfile.ZipFile(HttpRangeFile(url, session))
    dest.write_bytes(z.read(entry["name"]))
    return dest


def load_subtitle_text(video_id: str, subtitle_zip: zipfile.ZipFile) -> str:
    """Return cleaned (no HTML tags, no SRT timestamps/indices) subtitle text
    for a video, or "" if it has none (744/900 videos have subtitles)."""
    name = f"subtitle/{video_id}.srt"
    if name not in subtitle_zip.namelist():
        return ""
    raw = subtitle_zip.read(name).decode("utf-8", errors="replace")
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        line = line.replace('<font color="white" size=".72c">', "").replace("</font>", "")
        lines.append(line)
    return " ".join(lines)


def select_stratified_sample(annotations: pd.DataFrame, subtitle_zip: zipfile.ZipFile, n_per_duration: int, seed: int = 42) -> pd.DataFrame:
    sub_ids = {n.rsplit("/", 1)[-1].removesuffix(".srt") for n in subtitle_zip.namelist() if n.endswith(".srt")}
    videos = annotations.drop_duplicates("video_id")[["video_id", "videoID", "duration", "domain", "sub_category"]].copy()
    videos["has_subtitle"] = videos["videoID"].isin(sub_ids)

    parts = []
    for duration in ["short", "medium", "long"]:
        bucket = videos[(videos["duration"] == duration) & videos["has_subtitle"]]
        parts.append(bucket.sample(n=min(n_per_duration, len(bucket)), random_state=seed))
    return pd.concat(parts)


def _sanitize_metric_key(name: str) -> str:
    """MLflow metric/param names only allow alphanumerics, underscores,
    dashes, periods, spaces, and slashes -- category names like
    "Film & Television" need the "&" stripped."""
    return name.replace("&", "and").replace(" ", "_")


def compute_accuracy_metrics(results_path: Path, annotations: pd.DataFrame) -> dict:
    """Score a results file using the official Video-MME answer-extraction
    logic (extract_characters_regex, vendored from eval_your_results.py),
    returning a flat dict of accuracy percentages suitable for MLflow metrics.

    Looks up domain/sub_category/task_type from `annotations` by
    video_id/question_id rather than trusting those fields inside the
    results file itself -- results files produced before this function
    existed don't have sub_category embedded, and joining from the
    annotations (the actual source of truth) sidesteps that entirely.
    """
    results = json.loads(results_path.read_text())
    domain_by_video = annotations.drop_duplicates("video_id").set_index("video_id")[["domain", "sub_category"]].to_dict("index")
    task_type_by_question = annotations.set_index("question_id")["task_type"].to_dict()

    tallies = {"overall": {"correct": 0, "answered": 0}}

    def tally(bucket_key: str, is_correct: bool):
        b = tallies.setdefault(bucket_key, {"correct": 0, "answered": 0})
        b["answered"] += 1
        b["correct"] += int(is_correct)

    for video in results:
        video_id = video["video_id"]
        meta = domain_by_video.get(video_id, {})
        for q in video["questions"]:
            if "answer" not in q or "response" not in q:
                continue  # a recorded error entry, not a real response
            predicted = extract_characters_regex(q["response"])
            if predicted == "":
                continue
            correct = predicted == q["answer"]
            tally("overall", correct)
            tally(f"duration_{video['duration']}", correct)
            if "domain" in meta:
                tally(f"domain_{_sanitize_metric_key(meta['domain'])}", correct)
            if "sub_category" in meta:
                tally(f"subcategory_{_sanitize_metric_key(meta['sub_category'])}", correct)
            task_type = task_type_by_question.get(q["question_id"])
            if task_type:
                tally(f"tasktype_{_sanitize_metric_key(task_type)}", correct)

    return {
        f"accuracy_{key}": (100.0 * b["correct"] / b["answered"] if b["answered"] > 0 else 0.0)
        for key, b in tallies.items()
    }


async def run_question(
    client: grpcclient.InferenceServerClient,
    model: str,
    images_b64: list,
    question_row: pd.Series,
    subtitles: str,
) -> dict:
    options_block = "\n".join(question_row["options"])
    question_block = f"{question_row['question']}\n{options_block}"

    if subtitles:
        instruction = PROMPT_WITH_SUBTITLE.format(subtitles=subtitles, question=question_block)
    else:
        instruction = PROMPT_NO_SUBTITLE.format(question=question_block)

    prompt = build_video_prompt(len(images_b64), instruction)

    # guided_choice/guided_regex both hit a pre-existing bug in this deployment's
    # Triton vLLM backend ('NoneType' object has no attribute 'lora_name',
    # reproduces even on plain text with no video/multimodal input at all --
    # see docs/known_issues_and_todos.md). Falls back to plain free-text
    # generation, which matches the official benchmark's own methodology
    # anyway (output_test_template.json's real sample responses are free text
    # like "C. Berries.", not bare letters -- eval_your_results.py parses these).
    #
    # Auth token is fetched fresh per call rather than threaded through as a
    # fixed value -- auth_helper caches it internally already (so this is
    # cheap), and a run this long (~40 min for 144 calls) outlasts the
    # Domino token's lifetime, confirmed by an UNAUTHENTICATED failure ~5
    # minutes into the first pilot run. On that error, invalidate and retry
    # once with a fresh token before giving up on this question.
    start = time.time()
    try:
        result = await infer_non_streaming(
            client=client, headers=get_auth_headers(), model=model, prompt=prompt,
            max_tokens=16, temperature=0.0, top_p=1.0,
            images_b64=images_b64,
        )
    except AioRpcError as e:
        if e.code() != grpc.StatusCode.UNAUTHENTICATED:
            raise
        logger.warning("Auth token expired mid-run, refreshing and retrying once")
        invalidate_token()
        result = await infer_non_streaming(
            client=client, headers=get_auth_headers(), model=model, prompt=prompt,
            max_tokens=16, temperature=0.0, top_p=1.0,
            images_b64=images_b64,
        )
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "question_id": question_row["question_id"],
        "task_type": question_row["task_type"],
        "question": question_row["question"],
        "options": list(question_row["options"]),
        "answer": question_row["answer"],
        "response": result["generated_text"].strip(),
        "inference_ms": elapsed_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Video-MME benchmark against a Triton-served Llama 4 Scout variant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", required=True, help="Triton model name (e.g. llama4scout-vllm-fp8-dynamic)")
    parser.add_argument("--grpc-url", "-u", default=os.environ.get("TRITON_GRPC_URL", "localhost:50051"))
    parser.add_argument("--max-frames", type=int, default=8, help="Frames sampled per video (default: 8, the lower of fp8's and int4's current limit_mm_per_prompt, for a fair comparison)")
    parser.add_argument("--n-per-duration", type=int, default=8, help="Videos to sample per duration bucket (short/medium/long) if --video-ids-file is not given")
    parser.add_argument("--video-ids-file", help="Reuse a previously-selected sample (JSON list of {video_id, videoID, duration, domain}) instead of selecting a new one")
    parser.add_argument("--scratch-dir", default="/scratch/video-mme", help="Local scratch directory for downloaded videos/annotations (NOT the small persistent volume -- use a large ephemeral disk)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--subtitle-variant", choices=["both", "with", "without"], default="without", help="Default is subtitle-free: isolates video/visual understanding, since the quantization recipes keep the vision tower at full precision -- subtitles would let the model answer from text alone, confounding what we're actually trying to measure. Also sidesteps the with-subtitle context-overflow issue (see docs/known_issues_and_todos.md).")
    parser.add_argument("--mlflow-experiment", default=None, help="MLflow experiment name (default: video-mme-<your Domino username>, auto-detected). Experiment names must be unique per Domino deployment.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging (results files are always written regardless)")
    parser.add_argument("--score-only", action="store_true", help="Skip video download/inference entirely -- just score and MLflow-log the results files already in --output-dir for --model. For logging a run whose data collection already finished (e.g. under an older script version, before MLflow logging existed), without re-running any inference.")
    args = parser.parse_args()

    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    annotations_path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename="videomme/test-00000-of-00001.parquet", local_dir=str(scratch_dir))
    annotations = pd.read_parquet(annotations_path)

    variants = {"with": True, "without": False} if args.subtitle_variant == "both" else {args.subtitle_variant: args.subtitle_variant == "with"}
    out_paths = {
        name: output_dir / f"{args.model}_{'with' if name == 'with' else 'no'}_subtitle.json"
        for name in variants
    }
    # Resumable: a video already present in every variant's output file (e.g.
    # from a prior run cut short by a timeout) is skipped rather than re-run --
    # this is a long batch job (~15s/call, hundreds of calls) that shouldn't
    # have to restart from scratch after an interruption.
    results = {name: (json.loads(p.read_text()) if p.exists() else []) for name, p in out_paths.items()}
    done_video_ids = set.intersection(*[
        {v["video_id"] for v in videos} for videos in results.values()
    ]) if results else set()
    if done_video_ids:
        logger.info(f"Resuming: {len(done_video_ids)} videos already done, skipping")

    if args.score_only:
        n_videos = len(next(iter(results.values()))) if results else 0
        log_mlflow_runs(args, out_paths, results, annotations, n_videos)
        return

    subtitle_zip_path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename="subtitle.zip", local_dir=str(scratch_dir))
    subtitle_zip = zipfile.ZipFile(subtitle_zip_path)

    if args.video_ids_file:
        sample = pd.DataFrame(json.loads(Path(args.video_ids_file).read_text()))
    else:
        sample = select_stratified_sample(annotations, subtitle_zip, args.n_per_duration)
        sample_path = output_dir / "pilot_sample.json"
        sample[["video_id", "videoID", "duration", "domain"]].to_json(sample_path, orient="records", indent=2)
        logger.info(f"Selected {len(sample)} videos, saved sample to {sample_path}")

    chunk_index = build_chunk_index(scratch_dir / "chunk_index.json")
    http_session = requests.Session()

    def save_results():
        for variant_name, videos_out in results.items():
            out_paths[variant_name].write_text(json.dumps(videos_out, indent=2))

    n_calls = 0

    async def run_all():
        nonlocal n_calls
        client = grpcclient.InferenceServerClient(url=args.grpc_url)
        total_start = time.time()
        try:
            for _, video_row in sample.iterrows():
                video_id = video_row["video_id"]
                if video_id in done_video_ids:
                    continue
                youtube_id = video_row["videoID"]
                video_path = fetch_video(youtube_id, chunk_index, scratch_dir / "videos", http_session)
                images_b64 = sample_video_frames_base64(str(video_path), max_frames=args.max_frames)
                subtitle_text = load_subtitle_text(youtube_id, subtitle_zip)

                questions = annotations[annotations["video_id"] == video_id]
                for variant_name, use_subtitle in variants.items():
                    subs = subtitle_text if use_subtitle else ""
                    video_result = {
                        "video_id": video_id,
                        "duration": video_row["duration"],
                        "domain": video_row["domain"],
                        "questions": [],
                    }
                    for _, q in questions.iterrows():
                        try:
                            video_result["questions"].append(
                                await run_question(client, args.model, images_b64, q, subs)
                            )
                        except (InferenceServerException, AioRpcError) as e:
                            logger.error(f"video={video_id} question={q['question_id']} variant={variant_name}: {e}")
                            video_result["questions"].append({"question_id": q["question_id"], "error": str(e)})
                        n_calls += 1
                    results[variant_name].append(video_result)

                save_results()
                elapsed = time.time() - total_start
                logger.info(f"video {video_id} done -- {n_calls} calls so far, {elapsed:.0f}s elapsed ({elapsed/n_calls:.1f}s/call avg)")
        finally:
            await client.close()
        return time.time() - total_start

    total_elapsed = asyncio.run(run_all())

    for variant_name, videos_out in results.items():
        logger.info(f"Wrote {len(videos_out)} videos' results to {out_paths[variant_name]}")

    logger.info(f"TOTAL: {n_calls} new inference calls, {total_elapsed:.0f}s ({total_elapsed/max(n_calls,1):.1f}s/call avg)")

    if args.no_mlflow:
        return

    log_mlflow_runs(args, out_paths, results, annotations, len(sample))


def _domino_username() -> str:
    """DOMINO_STARTING_USERNAME (the env var Domino's own docs describe) is
    not actually set in this workspace -- confirmed by checking the live
    environment, not assumed. DOMINO_USER is just the generic container user
    ("ubuntu"), not the platform username. DOMINO_RUN_HOST_PATH's first path
    segment (e.g. "/mike_snyder/llama4scout-fp8/r/...") is the real Domino
    username in this environment, so that's tried first; DOMINO_USER_ID
    (numeric, but genuinely unique) is the fallback before finally giving up."""
    if os.environ.get("DOMINO_STARTING_USERNAME"):
        return os.environ["DOMINO_STARTING_USERNAME"]
    host_path = os.environ.get("DOMINO_RUN_HOST_PATH", "")
    parts = [p for p in host_path.split("/") if p]
    if parts:
        return parts[0]
    if os.environ.get("DOMINO_USER_ID"):
        return f"user{os.environ['DOMINO_USER_ID']}"
    return "local"


def log_mlflow_runs(args, out_paths: dict, results: dict, annotations: pd.DataFrame, n_videos: int):
    """Score each subtitle variant's results file and log it as an MLflow
    run. Only reads files already on disk -- never touches the model, so this
    is safe to call standalone (--score-only) against a run whose data
    collection already finished, without re-running any inference."""
    experiment_name = args.mlflow_experiment or f"video-mme-{_domino_username()}"
    try:
        mlflow.set_experiment(experiment_name=experiment_name)
    except Exception as e:
        logger.warning(f"MLflow experiment logging unavailable, skipping ({e}). Results files are unaffected.")
        return

    for variant_name in results:
        suffix = "with_subtitle" if variant_name == "with" else "no_subtitle"
        metrics = compute_accuracy_metrics(out_paths[variant_name], annotations)
        with mlflow.start_run(run_name=f"{args.model}_{suffix}"):
            mlflow.log_param("model", args.model)
            mlflow.log_param("subtitle_variant", suffix)
            mlflow.log_param("n_videos", n_videos)
            mlflow.log_param("max_frames", args.max_frames)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out_paths[variant_name]))
        logger.info(f"Logged MLflow run '{args.model}_{suffix}' to experiment '{experiment_name}' (overall accuracy: {metrics['accuracy_overall']:.1f}%)")


if __name__ == "__main__":
    main()
