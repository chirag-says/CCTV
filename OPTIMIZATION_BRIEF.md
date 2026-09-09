# SentinelAI pipeline optimisation brief

Written 2026-09-09 from measurements on the RTX 5070 (see `backend/bench_NVIDIA_GeForce_RTX_5070.json`
and `backend/bench_trt_NVIDIA_GeForce_RTX_5070.json`). Work through the phases in order. Each phase has
an acceptance test. Do not start the next phase until the current one passes on the 5070.

## Goal

Cameras one RTX 5070 can analyse at 70 percent utilisation, 720p substreams:

| Plan | Today, measured | Target after phase 3 | Target after phase 4 |
|---|---|---|---|
| Basic (person detect + ByteTrack + zone rules), 25 to 30 fps | 11 | 22 | 30 |
| Plus (Basic + face recognition at 5 to 10 fps) | 8.5 at 10 fps | 15 | 29 |
| Pro (Plus + ANPR at 2 fps + weapons at 1 fps + ReID) | 6.4 at 10 fps | 11 | 18 |

Live view must be 25 to 30 fps with under 500 ms glass-to-glass latency regardless of analysis fps.

## What the measurements say

- YOLOv8n on the 5070: 1.26 ms per frame end to end at batch 16 in PyTorch, 1.08 ms in TensorRT. The engine
  itself is under 0.5 ms. The rest is Python: CPU letterbox, result objects, tensor handoffs.
- SCRFD face detection: 8.84 ms end to end in the pipeline, of which the model is 4.18 ms in ONNX Runtime
  CUDA fp32 and 1.35 ms in TensorRT fp16. The remaining 4.7 ms is insightface's Python pre/post and NMS.
- ArcFace: 3.44 ms end to end, model 3.11 ms fp32, 1.12 ms TensorRT fp16.
- JPEG encode of a 720p frame: 8.9 ms CPU. Today every camera encodes every frame whether or not anyone is
  watching. At 30 fps that is 270 ms of CPU per camera-second, which caps a node at about 16 cameras before
  the GPU matters at all.
- Roboflow cloud API is the primary vehicle and person detector (`ENABLE_ROBOFLOW_DETECTION`,
  `ENABLE_ROBOFLOW_PERSON` in `backend/app/config.py`), adding about 750 ms per processed frame, a metered
  bill, and sending frames off site.
- Three ML frameworks load in one process: ONNX Runtime (faces), PyTorch (YOLO, RF-DETR, ReID, EasyOCR),
  TensorFlow (DeepFace gender in `safety_analytics.py`). Two to three CUDA contexts on 12 GB.

Conclusion: on this GPU the model is no longer the bottleneck. CPU work per frame and Python glue are.

## Phase 1: stop wasting CPU (about two weeks)

1. JPEG encode only when a viewer is connected. `CameraWorker._jpeg_encoder_loop` in
   `backend/app/vision/camera_worker.py` runs unconditionally. Gate it on `ws_manager.stream_counts()` for
   that camera plus any active MJPEG request in `backend/app/routes/cameras.py`. When nobody watches, the
   thread sleeps. Keep the last encoded frame for snapshot requests.
2. Remove Roboflow. Set both flags default False, delete the `inference_sdk` path in
   `backend/app/vision/traffic.py`, keep the local YOLO / RF-DETR fallback chain. Fine-tune YOLOv8n or
   YOLO11n on the Indian vehicles dataset already under `backend/datasets` if the COCO model is not enough.
3. Fail loudly on CPU fallback. `_get_insightface_app` in `backend/app/vision/detector.py` already checks
   the active provider; make a missing `CUDAExecutionProvider` a startup error unless
   `ALLOW_CPU_INFERENCE=true`, and do the same for torch. Document that `onnxruntime` (CPU) must not be
   installed alongside `onnxruntime-gpu`.
4. Drop DeepFace and TensorFlow. `ENABLE_ATTRIBUTE_RECOGNITION` default False, remove `deepface` and
   `tf-keras` from `requirements.txt`. Gender is not a product feature.
5. Set `RFDETR_MODEL_SIZE` default to `nano` and prefer YOLO11m for weapons; RF-DETR large does not fit
   the budget on a 12 GB card shared by 20 plus cameras.
6. Analysis rate by wall clock, not `FRAME_SKIP`. Add `ANALYSIS_FPS` per camera (default 10, gate
   cameras 5) and drop frames from the grabber to hit it. Never decode-skip in a way that breaks tracking.

Acceptance: run 20 copies of `backend/uploads/videos/*.mp4` looped as fake cameras (add a
`backend/load_test.py` that starts N `VisionPipeline` instances from files and reports achieved analysis
fps per camera, GPU utilisation from `pynvml`, and CPU load). Basic at 10 fps must hold 20 cameras on
the 5070 with under 4 CPU cores and no frame backlog. No network calls during the test.

## Phase 2: TensorRT engines and one inference loop per GPU (two to three weeks)

Phase 1 result (2026-09-09, load_test.py, 20 Basic cameras at 10 fps): 5.77 fps per camera achieved
against 10 target, GPU 18.8 percent busy, CPU 3.83 cores mean. The GPU is idle and the target is
missed, so the limit is per-frame Python work serialised by the GIL across 20 camera threads, plus
batch-1 inference (5.96 ms per frame at batch 1 against 1.96 ms at batch 8 on this card). Phase 2 is
therefore an architecture change, not only an engine swap. Order matters: do items 0 to 2 first,
they are cheap and each is measurable alone.

0. Quick wins, measured one at a time with load_test.py:
   a. The person YOLO call passes neither `half=True` nor `imgsz`; it runs fp32. Set both. Compare
      person-box counts and ByteTrack id churn before and after on the load test videos.
   b. `ENABLE_PROFILING` default False in production. Measure CPU with it off before attributing
      the remaining CPU to inference.
   c. Keep `PIPELINE_PLAN` (basic/plus/pro) added in Phase 1; it is now part of the product.
1. YOLOv8n and YOLO11m: export with `ultralytics` to `.engine`, fp16, dynamic batch up to 32, imgsz 640.
   Build on first start if the engine is missing or the GPU name changed. Engines are per GPU and
   gitignored.
2. SCRFD det_10g and ArcFace w600k_r50: load through ONNX Runtime's `TensorrtExecutionProvider` with
   `trt_fp16_enable`, `trt_engine_cache_enable`, fixed profiles 1x3x640x640 and 1x3x112x112 (see
   `backend/bench_trt.py`). Verify with `test_face.py --compare`: report the largest match-distance
   change; anything above 0.02 needs a threshold recalibration before it ships.
3. One inference loop per GPU. Replace "every camera thread calls the detector" with a single
   `InferenceServer` per GPU that owns the engines, collects pending frames from all cameras every
   tick (up to batch 32, tick 10 to 33 ms), runs one detector call, and returns results per camera.
   Camera workers become decode + tracking + rules only. Measure the tick fill and batch size.
4. Camera workers out of the GIL. Run camera workers as processes (multiprocessing, N cameras per
   process, default 8), passing frames to the inference server through shared memory (torch shared
   tensors or `multiprocessing.shared_memory`), not pickled numpy. Events go back through a queue to
   the existing event pipeline. The API process stays as it is.
5. GPU letterbox and NMS: resize and normalise on the GPU with torch before the engine; for SCRFD
   replace insightface's numpy anchor decode and NMS with a torch implementation on the GPU. This is
   where 4.7 ms per face-frame goes today.
6. Face detection on tracked person crops, not full frames, at `FACE_FPS` (default 5) independent of
   the tracking rate, and ArcFace only for tracks without a locked identity. Plus and Pro pay for
   faces per person, not per frame.

Acceptance: `load_test.py` holds 20 Basic cameras at 25 fps or better with GPU utilisation above
50 percent and under 4 CPU cores, and 15 Plus cameras at 10 fps tracking with faces at 5 fps.
`bench_pipeline.py` shows YOLOv8n under 0.8 ms per frame at batch 16 and SCRFD end to end under 3 ms.
`test_face.py --compare` largest distance change reported and under 0.02.

Known missing inputs, not to be worked around silently: `test_anpr_direct.py` needs
`models/license-plate-finetune-v1l.pt` and a sample video from another machine; ask for the model
file before touching ANPR.

## Phase 3: live view without transcoding (one to two weeks, can run in parallel with phase 2)

1. Deploy go2rtc (or mediamtx) beside the backend. It pulls each camera's RTSP once and serves WebRTC to
   the browser. No JPEG, no re-encode, native 25 to 30 fps, 300 to 500 ms latency.
2. Backend publishes detections as a lightweight metadata stream per camera over the existing WebSocket
   (`backend/app/core/websocket.py`): boxes, ids, labels, timestamps. Frontend draws them over the
   WebRTC video in `frontend/src/app/monitor/page.js`.
3. MJPEG and JPEG-over-WebSocket remain only as a fallback for browsers without WebRTC.

Acceptance: 9 live streams in a 3x3 grid at 25 fps or better with backend CPU under 10 percent for
streaming. Measured latency under 500 ms with a clock in frame.

## Phase 4: native hot loop (two to three months)

Only after phases 1 to 3 are in production. Evaluate NVIDIA DeepStream first: NVDEC decode, GPU
pre-processing, TensorRT inference, tracker, all in one C++/GStreamer pipeline configured rather than
written. Python keeps the API, database, rules engine, events and dashboard. If DeepStream does not fit,
write the decode-detect-track loop as a C++ or Rust extension using the phase 2 engines. Expected gain
over phase 2: about 2x on Basic, 2.5x on Plus and Pro, from removing the remaining Python per-frame work.

Acceptance: 30 Basic cameras at 30 fps on one 5070 with under one CPU core per GPU.

## Rules

- Measure before and after every change with `bench_pipeline.py`, `bench_trt.py` and `load_test.py` on
  the 5070. Commit the JSON outputs. Do not claim a speedup you did not measure.
- Accuracy must not regress: run the existing face and ANPR tests (`test_face.py`, `test_anpr_direct.py`)
  and compare match distances before and after fp16.
- One PR per numbered item. Keep the current behaviour reachable behind a config flag until the new
  path has run for a week.
- Do not add cloud services. Do not add a fourth ML framework. Prefer deleting code to adding it.
