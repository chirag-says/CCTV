# Phase 1 report — stop wasting CPU

All six numbered items in `OPTIMIZATION_BRIEF.md` phase 1 are done, one commit
each, every one behind a config flag that reaches the old behaviour. Measured
on the RTX 5070 in this machine with `backend/load_test.py`, 20 fake 720p/30fps
cameras from looped copies of `uploads/videos/*.mp4`.

## Headline: before and after

Both runs are **the same commit**, minutes apart, on the same idle GPU. The
"before" column is produced by setting every Phase 1 flag back to its
pre-phase value, so this is a true A/B rather than a comparison across machine
conditions — and it doubles as proof the rollback path works.

`load_test_flagsoff_baseline_20cam.json` → `load_test_item6_20cam.json`

| 20 cameras, plan `pro` | before (flags off) | after (all six items) | |
|---|---|---|---|
| analysis fps per camera | 2.46 | **3.41** | +39% |
| JPEG encodes per camera/s | 8.74 | **0.00** | item 1 |
| VRAM peak | 9,964 MB | **8,433 MB** | −1,531 MB |
| CPU mean | 5.67 cores | 8.55 cores | see note |
| CPU p95 | 7.46 cores | 10.96 cores | |
| GPU utilisation | 29.9% | 33.6% | |
| outbound network calls | 0 | 0 | see note |

**CPU rose, and that is real.** `ANALYSIS_FPS=25` asks the pipeline to work as
hard as it can; `FRAME_SKIP=4` on a 29 fps source asked for about 7.3. More
analysis costs more CPU. The per-camera budget is settled by the acceptance
run below, not by this pair.

**The network row is not evidence.** ultralytics had already persisted
`sync=false` to its user-level settings file, so reverting our flags could not
bring the telemetry back. The real result is `12 → 0`, from
`load_test_baseline_20cam.json` → `load_test_item2_20cam.json`.

## Acceptance test

> "Basic at 10 fps must hold 20 cameras on the 5070 with under 4 CPU cores and
> no frame backlog. No network calls during the test."

`load_test_acceptance_basic_10fps.json`, `load_test_acceptance_basic_25fps.json`

| criterion | target | 20 Basic @ 10 fps | 20 Basic @ 25 fps | verdict |
|---|---|---|---|---|
| analysis fps per camera | 10 | 5.77 | 6.45 | **FAIL** |
| CPU | < 4 cores | 3.83 mean / 5.33 p95 | 5.07 / 6.41 | **mean passes, p95 does not** |
| frame backlog | none | 0 | 0 | **PASS** |
| network calls | 0 | 0 | 0 | **PASS** |
| VRAM peak | — | 1,963 MB | 1,917 MB | |
| GPU utilisation | — | 18.8% | 21.6% | |

**Two of four pass. The frame rate misses by 42%.** Phase 1 did not reach its
acceptance test and it is not close enough to round up.

### Why it misses, with the number that proves it

**The GPU is 81% idle while missing the target.** That is the whole finding.
At 18.8% utilisation the card is not the constraint — Python per-frame work
and the GIL are, across 20 camera threads. This is exactly what the brief
predicted ("on this GPU the model is no longer the bottleneck") and it is why
raising the target from 10 to 25 buys only 12% more throughput (5.77 → 6.45)
for 32% more CPU.

`bench_pipeline.py` on this card says where the headroom is:

| stage | mean ms |
|---|---|
| yolov8n 640 fp16, **batch 1** | 5.96 |
| yolov8n 640 fp16, **batch 8**, per frame | **1.96** |
| yolo11m 640 fp16, batch 1 | 8.40 |
| rfdetr_nano, batch 1 | 34.73 |
| scrfd_det10g 640 | 8.92 |
| arcface r50 per face | 3.27 |
| jpeg encode 720p (CPU) | 9.08 |

20 cameras × 10 fps = 200 person-detections per second.
At batch 1 that is **1,192 ms of GPU per second** — impossible on one stream.
At batch 8 it is **392 ms** — comfortable.

So the missing 42% is not a Phase 1 failure to be patched; it is precisely
what phase 2 items 4 and 5 (GPU letterbox, batching across cameras) exist to
deliver, and the bench now quantifies the 3.04× available from batching alone.

## Accuracy

**Unchanged, verified rather than assumed.** `test_face.py` was rewritten to
report ArcFace match distances (it previously printed only bounding boxes, so
it could not answer this question at all), then run against the pre-Phase-1
detector and the current one over the 37 enrolled faces in `snapshots/faces`:

```
compared 30 matched faces
largest distance change  0.00000000
PASS: every match distance and identity is unchanged
```

Structurally corroborated: Phase 1 touched neither `app/vision/recognizer.py`
nor `app/vision/anpr.py` at all (empty diff), and the only change to
`app/vision/detector.py` is the CPU-refusal branch — no detection maths.

`test_anpr_direct.py` **could not be run.** It reads
`C:\Users\lokes\...\14598470_2160_3840_30fps.mp4` (line 18), a path on a
different machine, and needs `models/license-plate-finetune-v1l.pt` (line 27);
there is no `models/` directory in this checkout. Both are missing inputs, not
failures. ANPR accuracy is therefore unverified by execution, though `anpr.py`
is byte-identical to its pre-Phase-1 state.

## What each item actually did

| # | item | measured effect |
|---|---|---|
| 1 | JPEG only when watched | 8.74 → 0.00 encodes/camera/s |
| 2 | Roboflow off, telemetry off | 12 → 0 outbound calls |
| 3 | refuse CPU inference | no runtime effect on a CUDA host, by design |
| 4 | drop DeepFace / TensorFlow | TF-family modules imported: none |
| 5 | YOLO11m weapons, RF-DETR nano | −1,466 MB VRAM, +24% analysis fps |
| 6 | wall-clock analysis rate | rate is now stated and held, not implied |

Item 5 is the only item with a clean, uncontaminated single-item measurement
(`load_test_item4` → `load_test_item5`, both on a quiet GPU). See
"measurement conditions" below for why the others are not.

## Five things found by measuring that reading would not have caught

1. **ultralytics posts to Google from inside `predict()`**
   (`utils/events.py:46`). 12 calls in the baseline, and the only outbound
   traffic in the whole system. Nothing in this repo asked for it.

2. **Ollama was holding 5.5 GB of the 12 GB card** for `qwen2.5vl:7b`, the
   Gallery Optimizer's VLM, plus the dev server. The baseline's "VRAM
   exhausted" reading was half somebody else's model. Four runs at identical
   load returned 0.46 / 0.14 / 2.17 / 2.59 analysis fps — a 15× spread that
   had nothing to do with the code.

3. **`time.monotonic()` resolves to 15.625 ms on Windows.** My first version of
   item 6 used it. A 25 fps target is a 40 ms interval, which that clock can
   only express in 15.6 ms steps, so the achieved rate would have sat up to
   13% under target for reasons no amount of spare GPU could fix, and no rate
   above ~64 fps was expressible. Now `perf_counter`, 100 ns.

4. **The item 6 rollback flag was a total outage.** In legacy `FRAME_SKIP`
   mode "due" means `_frame_count % skip == 0`, and that counter only advances
   inside `process_frame`. My grabber-side drop gate consulted it before
   consumption, so frame 1 failed the test, the grabber dropped it, and the
   counter never moved again. Measured: **0.00 analysis fps and 1.3% GPU on
   all 20 cameras.** Nothing except running the documented rollback would have
   found this. Pinned by `test_grabber_keeps_every_frame_in_legacy_mode`.

5. **SCRFD finds no face in a tight face crop.** `app.get()` returns 0 on the
   raw enrolled crops and 1 on the same image with a border — the face fills
   the frame and falls outside every anchor scale. Relevant to anything that
   feeds stored crops back through detection.

## Where the brief and the code disagreed

Reported rather than worked around, as instructed.

1. **Item 1's gate did not exist.** `ws_manager.stream_counts()` was never
   populated: `connect_stream`/`disconnect_stream`
   (`app/core/websocket.py:38,47`) had no callers, and the stream endpoint
   (`app/routes/cameras.py:238`) accepted its socket directly. Gating on it as
   written would have read a permanently empty dict and switched encoding off
   for good, on every camera. The endpoint now registers with the manager it
   was always meant to use.

2. **Item 5's setting was wired to nothing.** `RFDETR_MODEL_SIZE`
   (`app/config.py`) defaulted to `"large"` and was never read.
   `HazardDetector` and `TrafficMonitor` each defaulted the size in their own
   signature and `VisionPipeline` passed neither, so setting it to `nano`
   alone would have been a no-op that looked like a fix. Both call sites now
   receive it.

3. **The Basic tier could not be run.** Face detection ran unconditionally in
   face mode (`app/vision/pipeline.py`), so a camera sold as Basic still paid
   for SCRFD and ArcFace, and none of the six items introduced a plan switch —
   the acceptance test was unrunnable as written. Added `PIPELINE_PLAN`
   (`basic`/`plus`/`pro`, default `pro`) in its own commit, since it is not one
   of the brief's items.

4. **Neither accuracy test could run.** `test_face.py:9` read
   `training/videos/raw/*.mp4`, which does not exist here, and printed no match
   distance in any case; `test_anpr_direct.py:22` reads another machine's disk.

5. **`bench_NVIDIA_GeForce_RTX_5070.json` was not in the repo** although the
   brief cites it as a source. Generated by this run; only the `_trt_` variant
   existed.

## Measurement conditions — read before trusting the per-item numbers

The baseline and items 1–3 were measured while **`python run.py` and Ollama
were also using the GPU**, which was not noticed until item 4. Those runs are
committed as taken and their commit messages say so; the per-item deltas in
them are directional at best. The dev server came back up once mid-phase
(11:31:56) and that run was discarded rather than reported.

Everything from item 4 onward, and every number in the two tables at the top of
this report, was measured with the GPU idle at 742 MB of 12,227.

`bench_pipeline.py` also disagrees with the brief on one figure: the brief
quotes YOLOv8n at 1.26 ms, which is the **batch 16** number. The pipeline runs
batch 1, measured here at 5.96 ms. That gap is the phase 2 batching argument.

## Test suite

546 passing, up from 501 — 45 new tests across `test_jpeg_on_demand.py`,
`test_cpu_inference_guard.py`, `test_hazard_backend.py`, `test_analysis_rate.py`
and `test_plan_tiers.py`.

7 failures in `test_attendance_export.py`, `test_load_governor.py` and
`test_zone_tracker.py` **pre-date this work** — confirmed by re-running them
with these changes stashed. A further 3 modules
(`test_camera_groups.py`, `test_unknown_registry_shared.py`,
`test_zone_calibration_api.py`) fail to import against code that does not exist
yet; all are untracked files that were in the working tree before Phase 1.

## Recommended next

Phase 2, unchanged in order. The acceptance gap is a batching gap, and item 5
of phase 2 (one detector call per GPU tick across all cameras) is the single
change the numbers above point at. Two smaller things worth folding in:

- The YOLO person call passes neither `half=True` nor `imgsz`, so it runs
  fp32. Deliberately left alone — fp16 is phase 2 and needs the accuracy
  comparison this report's `test_face.py --compare` now makes possible.
- `ENABLE_PROFILING` is on by default and every camera thread takes a
  process-wide lock several times per frame. Worth measuring with it off
  before assuming the remaining CPU is inference.
