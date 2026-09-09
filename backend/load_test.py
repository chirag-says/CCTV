"""
SentinelAI multi-camera load test.

Starts N VisionPipeline instances (via CameraWorker, so the JPEG encoder
thread is included) fed by looped, real-time-paced copies of the .mp4 files
in uploads/videos, and reports:

  * achieved ANALYSIS fps per camera  (detection frames, not decoded frames)
  * GPU utilisation and VRAM from pynvml
  * CPU load, as whole cores and as a percentage of the machine
  * frame backlog, at every place this pipeline can queue or drop

Run:

    cd backend && python load_test.py --cameras 20 --duration 120 \
        --out load_test_baseline.json

Writes the JSON report and prints a per-camera table.

WHY A FAKE CAMERA AND NOT JUST cv2.VideoCapture(file)
A file capture returns frames as fast as the decoder can produce them, so a
pipeline reading one is never late -- it just reads slower. That measures
throughput and hides the only question this test exists to answer: can 20
cameras be kept AT a rate while frames keep arriving whether or not we are
ready. LoopedFileCapture therefore paces read() to the source frame rate and
loops at EOF, exactly like an RTSP stream that does not care about us.

WHAT "BACKLOG" MEANS HERE
The frame path is latest-wins at every hop (grabber slot, JPEG slot), so
nothing grows without bound there; falling behind shows up as DROPS, and is
reported as such. The unbounded queues in this process are the hazard and
traffic ThreadPoolExecutors, which accept a submit() per frame regardless of
whether the previous one finished. Those are sampled by queue depth.
"""

import argparse
import ctypes
import glob
import json
import os
import platform
import socket
import statistics
import sys
import threading
import time
from datetime import datetime, timezone

# Must precede any torch/onnxruntime import, same as the app does.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import numpy as np
import psutil


# ── Network guard ─────────────────────────────────────────────────────────
# The acceptance test says "no network calls during the test". Asserting that
# by reading the code is not evidence; this records every non-loopback
# connect() the process attempts, from any thread, including inside a vendored
# SDK. Loopback is excluded because Postgres lives there.

_net_lock = threading.Lock()
_net_calls = []
_net_armed = False


def _install_network_guard():
    global _net_armed
    if _net_armed:
        return
    _net_armed = True
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _record(addr):
        try:
            host = addr[0] if isinstance(addr, tuple) else str(addr)
        except Exception:
            host = str(addr)
        if host in ("127.0.0.1", "::1", "localhost", "0.0.0.0"):
            return
        with _net_lock:
            if len(_net_calls) < 200:
                _net_calls.append({"addr": str(addr), "t": time.time()})

    def connect(self, addr):
        _record(addr)
        return real_connect(self, addr)

    def connect_ex(self, addr):
        _record(addr)
        return real_connect_ex(self, addr)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


# ── Fake camera ───────────────────────────────────────────────────────────

class LoopedFileCapture:
    """A video file pretending to be a live camera.

    Implements the slice of the cv2.VideoCapture API that VisionPipeline's
    grabber uses: isOpened(), read(), set(), get(), release(). read() blocks
    until the next frame is due, so a slow consumer misses frames instead of
    slowing the camera down -- which is the entire point.

    Every instance decodes its own copy of the file. That is deliberate: H.264
    decode is a real per-camera CPU cost (bench_pipeline.py measures it as
    decode_720p_h264_cpu) and sharing one decoder across 20 cameras would
    delete it from the measurement.
    """

    def __init__(self, path: str, fps: float = None):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.fps = float(fps or src_fps)
        self._interval = 1.0 / self.fps
        self._next_due = time.perf_counter()
        self.served = 0
        self.loops = 0

    def isOpened(self):
        return self._cap is not None and self._cap.isOpened()

    def set(self, *_a, **_kw):
        return True

    def get(self, prop):
        return self._cap.get(prop) if self._cap else 0.0

    def read(self):
        if self._cap is None:
            return False, None
        # Pace to wall clock. If we are already late (the decoder itself fell
        # behind), do not try to catch up by bursting -- a real camera cannot.
        now = time.perf_counter()
        delay = self._next_due - now
        if delay > 0:
            time.sleep(delay)
            self._next_due += self._interval
        else:
            self._next_due = now + self._interval

        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.loops += 1
            ok, frame = self._cap.read()
            if not ok:
                return False, None
        self.served += 1
        return True, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


_FAKE_PREFIX = "loadtest:"
_fake_captures = {}          # camera_id -> LoopedFileCapture
_fake_lock = threading.Lock()


def _install_capture_patch(source_fps):
    """Route 'loadtest:<camera_id>:<path>' sources to LoopedFileCapture.

    Patching cv2.VideoCapture inside the pipeline module (rather than handing
    the pipeline a pre-built capture) keeps VisionPipeline.start() and the
    grabber's reconnect path on their real code, so the test exercises the
    shipping control flow.
    """
    from app.vision import pipeline as pipeline_mod
    real_vc = pipeline_mod.cv2.VideoCapture

    def factory(source, *args, **kwargs):
        if isinstance(source, str) and source.startswith(_FAKE_PREFIX):
            _, cam_id, path = source.split(":", 2)
            cap = LoopedFileCapture(path, fps=source_fps)
            with _fake_lock:
                _fake_captures[cam_id] = cap
            return cap
        return real_vc(source, *args, **kwargs)

    pipeline_mod.cv2.VideoCapture = factory


# ── Sampling ──────────────────────────────────────────────────────────────

def _gpu_handle():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None, None


def _executor_depth(obj):
    """Pending tasks in a module's ThreadPoolExecutor, or None."""
    ex = getattr(obj, "_executor", None)
    if ex is None:
        return None
    try:
        return ex._work_queue.qsize()
    except Exception:
        return None


class Sampler(threading.Thread):
    """Once a second: CPU, GPU, and every queue depth we can see."""

    def __init__(self, workers, proc, interval=1.0):
        super().__init__(daemon=True, name="loadtest-sampler")
        self.workers = workers
        self.proc = proc
        self.interval = interval
        self.stop = threading.Event()
        self.samples = []
        self.max_hazard_q = 0
        self.max_traffic_q = 0

    def run(self):
        pynvml, h = _gpu_handle()
        self.proc.cpu_percent(None)          # prime the counter
        psutil.cpu_percent(None)
        while not self.stop.wait(self.interval):
            s = {
                "t": round(time.time(), 2),
                "proc_cpu_pct": round(self.proc.cpu_percent(None), 1),
                "sys_cpu_pct": round(psutil.cpu_percent(None), 1),
                "rss_mb": round(self.proc.memory_info().rss / 2**20),
                "threads": self.proc.num_threads(),
            }
            if h is not None:
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    s["gpu_util_pct"] = u.gpu
                    s["gpu_mem_util_pct"] = u.memory
                    s["gpu_mem_used_mb"] = round(m.used / 2**20)
                except Exception:
                    pass
            for w in self.workers:
                hq = _executor_depth(w.pipeline.hazard_detector)
                tq = _executor_depth(w.pipeline.traffic_monitor)
                if hq:
                    self.max_hazard_q = max(self.max_hazard_q, hq)
                if tq:
                    self.max_traffic_q = max(self.max_traffic_q, tq)
            self.samples.append(s)


def _agg(samples, key):
    vals = [s[key] for s in samples if key in s]
    if not vals:
        return None
    return {
        "mean": round(statistics.mean(vals), 1),
        "p50": round(statistics.median(vals), 1),
        "p95": round(sorted(vals)[max(0, int(0.95 * len(vals)) - 1)], 1),
        "max": round(max(vals), 1),
    }


# ── Per-camera counters ───────────────────────────────────────────────────

def _snapshot(worker):
    p = worker.pipeline
    with _fake_lock:
        cap = _fake_captures.get(worker.camera_id)
    return {
        "served": cap.served if cap else 0,
        "loops": cap.loops if cap else 0,
        "consumed": p._frame_count,
        "analysis": getattr(p, "_analysis_frames", 0),
        "jpeg": getattr(worker, "_jpeg_encodes", 0),
        "jpeg_skipped": getattr(worker, "_jpeg_skipped", 0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", type=int, default=20)
    ap.add_argument("--duration", type=float, default=120.0,
                    help="measurement window in seconds (after warmup)")
    ap.add_argument("--warmup", type=float, default=60.0,
                    help="seconds to let every model load before measuring")
    ap.add_argument("--videos", default="uploads/videos/*.mp4")
    ap.add_argument("--source-fps", type=float, default=None,
                    help="override the fake camera's frame rate (default: the file's)")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--label", default=None, help="free-text label stored in the JSON")
    ap.add_argument("--viewers", type=int, default=0,
                    help="pretend this many cameras have a live viewer attached")
    ap.add_argument("--allow-db-writes", action="store_true",
                    help="allow embedding suggestions to be written (off by default so "
                         "a load test does not pollute the database)")
    ap.add_argument("--no-encodings", action="store_true",
                    help="skip loading the face gallery from the database")
    args = ap.parse_args()

    videos = sorted(glob.glob(args.videos))
    if not videos:
        sys.exit(f"no videos matched {args.videos}")

    if not args.allow_db_writes:
        # Only gate the one path that writes rows from a camera thread. This
        # skips a rare snapshot + INSERT, not any per-frame work, so it does
        # not move the numbers this test reports.
        os.environ["SUGGESTION_MIN_CONFIDENCE"] = "2.0"

    _install_network_guard()

    from app.config import settings
    _install_capture_patch(args.source_fps)
    from app.vision.camera_worker import CameraWorker

    proc = psutil.Process()
    n_cpu = psutil.cpu_count(logical=True)

    # ── config actually in force, recorded with the numbers ───────────
    cfg = {k: getattr(settings, k, None) for k in (
        "PIPELINE_MODE", "TRACKING_MODE", "FRAME_SKIP", "ANALYSIS_FPS",
        "INSIGHTFACE_DET_SIZE", "INSIGHTFACE_DET_THRESH", "DETECTION_SCALE",
        "ENABLE_ATTRIBUTE_RECOGNITION", "ENABLE_CROWD_DETECTION",
        "ENABLE_LOITERING_DETECTION", "ENABLE_HAZARD_DETECTION",
        "ENABLE_ANPR", "ENABLE_TRAFFIC_MONITOR", "ENABLE_REID",
        "ENABLE_ROBOFLOW_DETECTION", "ENABLE_ROBOFLOW_PERSON",
        "RFDETR_MODEL_SIZE", "HAZARD_MODEL_PATH", "ENABLE_PROFILING",
        "JPEG_ENCODE_ON_DEMAND", "ALLOW_CPU_INFERENCE", "WEAPON_MODEL_PATH",
    ) if hasattr(settings, k)}

    events = {"n": 0}
    ev_lock = threading.Lock()

    def on_event(*_a, **_kw):
        with ev_lock:
            events["n"] += 1

    print(f"starting {args.cameras} cameras from {len(videos)} video(s)")
    workers = []
    for i in range(args.cameras):
        cam_id = f"loadtest-{i:02d}"
        path = videos[i % len(videos)]
        w = CameraWorker(
            camera_id=cam_id,
            stream_source=f"{_FAKE_PREFIX}{cam_id}:{path}",
            on_event=on_event,
            on_unknown=on_event,
            on_frame=None,
        )
        workers.append(w)

    if not args.no_encodings:
        try:
            from app.services.person_service import PersonService
            encodings = PersonService.get_all_encodings()
            for w in workers:
                w.pipeline.recognizer.load_encodings(encodings)
            print(f"loaded {len(encodings)} face encodings into each pipeline")
        except Exception as e:
            print(f"could not load encodings ({e}); running with an empty gallery")

    # Viewers, for the on-demand JPEG gate. register_viewer() only exists once
    # item 1 has landed; before that every camera encodes unconditionally and
    # this is a no-op.
    viewer_tokens = []
    if args.viewers:
        try:
            from app.vision.viewers import viewer_registry
            for w in workers[:args.viewers]:
                viewer_tokens.append(viewer_registry.attach(w.camera_id))
            print(f"attached {len(viewer_tokens)} fake viewer(s)")
        except ImportError:
            print("no viewer registry in this build; --viewers ignored")

    t_start = time.time()
    for w in workers:
        w.start()

    sampler = Sampler(workers, proc)
    print(f"warming up for {args.warmup:.0f}s (model loads)...")
    time.sleep(args.warmup)

    sampler.start()
    before = {w.camera_id: _snapshot(w) for w in workers}
    t0 = time.perf_counter()
    print(f"measuring for {args.duration:.0f}s...")
    time.sleep(args.duration)
    elapsed = time.perf_counter() - t0
    after = {w.camera_id: _snapshot(w) for w in workers}
    sampler.stop.set()
    sampler.join(timeout=5)

    print("stopping cameras...")
    for w in workers:
        try:
            w.stop()
        except Exception as e:
            print(f"  stop {w.camera_id}: {e}")
    for tok in viewer_tokens:
        try:
            tok.release()
        except Exception:
            pass

    # ── per-camera results ────────────────────────────────────────────
    target = float(getattr(settings, "ANALYSIS_FPS", 0) or 0)
    cams = {}
    for w in workers:
        a, b = after[w.camera_id], before[w.camera_id]
        d = {k: a[k] - b[k] for k in a}
        served_fps = d["served"] / elapsed
        analysis_fps = d["analysis"] / elapsed
        cams[w.camera_id] = {
            "source_fps": round(served_fps, 2),
            "analysis_fps": round(analysis_fps, 2),
            "consumed_fps": round(d["consumed"] / elapsed, 2),
            "jpeg_fps": round(d["jpeg"] / elapsed, 2),
            "jpeg_encodes_skipped": d["jpeg_skipped"],
            "frames_served": d["served"],
            "frames_consumed": d["consumed"],
            # Frames the grabber overwrote before process_frame took them.
            # Expected and harmless when we are deliberately analysing at a
            # lower rate than the camera sends; it is only a fault if the
            # analysis rate itself is short of target.
            "grabber_dropped": d["served"] - d["consumed"],
            "loops": d["loops"],
        }

    fps_all = [c["analysis_fps"] for c in cams.values()]
    src_all = [c["source_fps"] for c in cams.values()]
    cpu = _agg(sampler.samples, "proc_cpu_pct")
    gpu = _agg(sampler.samples, "gpu_util_pct")
    vram = _agg(sampler.samples, "gpu_mem_used_mb")

    # A camera is "behind" when it did not reach the rate it was asked for.
    # With no ANALYSIS_FPS configured there is no target, so nothing can be
    # behind by definition -- report the spread instead of a pass/fail.
    behind = []
    if target > 0:
        behind = [cid for cid, c in cams.items()
                  if c["analysis_fps"] < target * 0.95]

    with _net_lock:
        net = list(_net_calls)

    res = {
        "label": args.label,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_logical": n_cpu,
            "cpu_physical": psutil.cpu_count(logical=False),
            "gpu": _gpu_name(),
        },
        "run": {
            "cameras": args.cameras,
            "videos": videos,
            "warmup_sec": args.warmup,
            "measured_sec": round(elapsed, 2),
            "viewers": len(viewer_tokens),
            "total_startup_sec": round(time.time() - t_start, 1),
        },
        "config": cfg,
        "totals": {
            "analysis_fps_target": target or None,
            "analysis_fps_mean": round(statistics.mean(fps_all), 2),
            "analysis_fps_min": round(min(fps_all), 2),
            "analysis_fps_max": round(max(fps_all), 2),
            "analysis_fps_total": round(sum(fps_all), 2),
            "source_fps_mean": round(statistics.mean(src_all), 2),
            "cameras_below_target": behind,
            "cpu_percent": cpu,
            "cpu_cores": {k: round(v / 100.0, 2) for k, v in cpu.items()} if cpu else None,
            "cpu_cores_available": n_cpu,
            "gpu_util_percent": gpu,
            "gpu_mem_used_mb": vram,
            "rss_mb": _agg(sampler.samples, "rss_mb"),
            "threads": _agg(sampler.samples, "threads"),
            "max_hazard_queue_depth": sampler.max_hazard_q,
            "max_traffic_queue_depth": sampler.max_traffic_q,
            "events_emitted": events["n"],
            "external_network_calls": len(net),
            "external_network_sample": net[:20],
        },
        "cameras": cams,
        "samples": sampler.samples,
    }

    out = args.out or f"load_test_{args.cameras}cam.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    t = res["totals"]
    print(f"\n{'camera':14s} {'src fps':>8s} {'analysis':>9s} {'consumed':>9s} "
          f"{'jpeg':>7s} {'dropped':>8s}")
    for cid, c in cams.items():
        print(f"{cid:14s} {c['source_fps']:8.2f} {c['analysis_fps']:9.2f} "
              f"{c['consumed_fps']:9.2f} {c['jpeg_fps']:7.2f} {c['grabber_dropped']:8d}")
    print(f"\ncameras            {args.cameras}")
    print(f"analysis fps       mean {t['analysis_fps_mean']}  "
          f"min {t['analysis_fps_min']}  max {t['analysis_fps_max']}  "
          f"target {t['analysis_fps_target']}")
    if cpu:
        print(f"CPU                mean {cpu['mean']}%  p95 {cpu['p95']}%  "
              f"= {t['cpu_cores']['mean']} / {t['cpu_cores']['p95']} cores of {n_cpu}")
    if gpu:
        print(f"GPU util           mean {gpu['mean']}%  p95 {gpu['p95']}%")
    if vram:
        print(f"VRAM               mean {vram['mean']} MB  max {vram['max']} MB")
    print(f"hazard queue max   {t['max_hazard_queue_depth']}")
    print(f"traffic queue max  {t['max_traffic_queue_depth']}")
    print(f"external net calls {t['external_network_calls']}")
    if behind:
        print(f"BEHIND TARGET      {len(behind)} camera(s): {', '.join(behind[:6])}")
    print(f"\nwrote {out}")


def _git_head():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _gpu_name():
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return n.decode() if isinstance(n, bytes) else n
    except Exception:
        return None


if __name__ == "__main__":
    main()
