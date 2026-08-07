# Known Issues & TODOs

Running list of things found during work on this branch that are worth fixing eventually but aren't blocking current work.

## `values-dev.yaml` may still have undiscovered drift from what's actually deployed to `triton-demo`

- **Background**: live Helm releases have repeatedly drifted from their git-tracked `values-*.yaml` files on this branch (targeted `--set` changes applied live without updating the source file). Every specific drift found so far (`admin.image`, `proxy.grpc_image`/`proxy.http_image`, `proxy.jwt.issuer`, `persistence.s3.iamRoleArn`) has been fixed, both live and in git.
- **Remaining**: no deliberate, full field-by-field diff between `values-dev.yaml` and `helm get values domino-triton -n triton-demo` has been done -- only the fields that happened to matter functionally (image tags, issuer, IAM role) were checked. Worth doing a complete diff at some point to rule out anything else silently diverged.
- **Related gotcha worth remembering**: a full `helm upgrade -f <values-file>` reapplies the entire file, silently undoing any manual `kubectl scale` (Helm doesn't track manual scaling). Prefer `helm upgrade --reuse-values --set <specific.field>=<value>` when only a narrow field needs to change and there's live-only state (like a manual scale-down) worth preserving.

## YOLOv8n video throughput is ~600ms/frame (~1.7 FPS) against `triton-demo`, cause unconfirmed

- Measured directly, even with `config.pbtxt`'s `instance_group: KIND_GPU` and `dynamic_batching` configured -- suspiciously slow for a "nano" model. Neither larger client-side batch sizes (tested up to 8) nor `--async` mode (tested; caused `DEADLINE_EXCEEDED` errors) improved it.
- Leading suspect: per-request overhead from shipping each frame as an uncompressed ~4.9MB FP32 tensor over gRPC through the HTTP/gRPC proxy layer, rather than actual model compute -- not confirmed, needs real profiling.
- This is why yolov8n's video test-inference uses a fixed low sampling rate (`sample_fps`, default 2) rather than processing every frame: at this throughput, a full-framerate pass over a real video would exceed the dashboard's 120s subprocess timeout (`app-src/routes/testing.py`). Fixing the underlying latency would allow raising that default meaningfully.

## YOLOv8n sometimes misclassifies real detections at low confidence

- Bounding boxes are correctly localized on real objects (e.g. cars in `samples/traffic.mp4`), but the class label is sometimes wrong at low confidence (e.g. a clearly-visible car labeled `cell phone: 0.49` or `bus: 0.32`).
- Leading hypothesis: `traffic.mp4` is shot from a top-down/bird's-eye angle, underrepresented in COCO's training data (COCO's "car" images are overwhelmingly street-level), so the pretrained YOLOv8n weights don't generalize well to this camera angle. Not investigated further -- open question whether this needs a different model/fine-tuning or is out of scope for this demo.

## Known, deliberately-untouched items (not bugs, just flagged so they aren't rediscovered)

- `src/proxy/http/routes/testing.py` contains a second, separate implementation of a `/v1/testing/infer/{model}` route with its own `max_frames`-driven YOLO inference logic, living inside the proxy service's own codebase. Confirmed dead code -- the Dashboard UI only ever calls `app-src/routes/testing.py`'s `/api/testing/infer/{model}`. Candidate for removal, but left alone in case anything else calls the proxy's route directly.
- `scripts/clients/yolov26n_video_grpc_client.py` and `scripts/benchmarks/benchmark_yolov26n_clients.py` are scaffolding for a **not-yet-implemented, deliberately deferred** model -- intentionally unregistered in the Dashboard, not an oversight.
