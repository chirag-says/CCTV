"""
Benchmark utilities — shared infrastructure for all benchmark scripts.

READ-ONLY: Does not modify production configuration, database, or models.
All results are written to benchmarks/results/ as JSON.
"""

import json
import os
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
FRAMES_DIR = BACKEND_DIR / "snapshots" / "frames"
FACES_DIR = BACKEND_DIR / "snapshots" / "faces"
UNKNOWNS_DIR = BACKEND_DIR / "snapshots" / "unknowns"
VIDEOS_DIR = BACKEND_DIR / "uploads" / "videos"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PYTHON = r"C:\Users\chira\AppData\Local\Programs\Python\Python311\python.exe"


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class Timer:
    """Accumulates timing samples and computes statistics."""

    def __init__(self, name: str = ""):
        self.name = name
        self.samples: List[float] = []

    @contextmanager
    def measure(self):
        t0 = time.perf_counter()
        yield
        self.samples.append((time.perf_counter() - t0) * 1000.0)  # ms

    def time_once(self):
        """Return a context manager that records one sample."""
        return self.measure()

    @property
    def stats(self) -> Dict[str, float]:
        if not self.samples:
            return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "n": 0}
        s = sorted(self.samples)
        n = len(s)
        return {
            "avg": round(sum(s) / n, 3),
            "p50": round(s[n // 2], 3),
            "p95": round(s[int(n * 0.95)], 3),
            "p99": round(s[int(n * 0.99)], 3),
            "max": round(s[-1], 3),
            "n": n,
        }


# ---------------------------------------------------------------------------
# GPU metrics
# ---------------------------------------------------------------------------
def gpu_snapshot() -> Dict[str, Any]:
    """Capture current GPU utilization and VRAM usage via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        return {
            "gpu_name": name,
            "vram_used_mb": round(mem.used / 1024 / 1024, 1),
            "vram_total_mb": round(mem.total / 1024 / 1024, 1),
            "gpu_util_pct": util.gpu,
            "mem_util_pct": util.memory,
        }
    except Exception as e:
        return {"error": str(e)}


class GPUSampler:
    """Sample GPU utilization in the background at ~4 Hz."""

    def __init__(self):
        self._samples: List[Dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            self._samples.append(gpu_snapshot())
            time.sleep(0.25)

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self._samples:
            return {}
        utils = [s.get("gpu_util_pct", 0) for s in self._samples if "gpu_util_pct" in s]
        vrams = [s.get("vram_used_mb", 0) for s in self._samples if "vram_used_mb" in s]
        return {
            "gpu_util_avg": round(sum(utils) / len(utils), 1) if utils else 0,
            "gpu_util_max": max(utils) if utils else 0,
            "vram_avg_mb": round(sum(vrams) / len(vrams), 1) if vrams else 0,
            "vram_max_mb": max(vrams) if vrams else 0,
            "n_samples": len(self._samples),
        }


# ---------------------------------------------------------------------------
# CPU metrics
# ---------------------------------------------------------------------------
def cpu_snapshot() -> Dict[str, float]:
    """Process CPU and RAM."""
    import psutil
    proc = psutil.Process()
    return {
        "cpu_pct": proc.cpu_percent(interval=0.1),
        "ram_mb": round(proc.memory_info().rss / 1024 / 1024, 1),
        "system_cpu_pct": psutil.cpu_percent(interval=0.1),
    }


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------
def load_frames_from_dir(
    directory: Path,
    max_frames: int = 200,
    seed: int = 42,
) -> List[Tuple[str, np.ndarray]]:
    """Load a random sample of frames from a directory of images."""
    rng = np.random.RandomState(seed)
    files = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png')))
    if len(files) > max_frames:
        indices = rng.choice(len(files), max_frames, replace=False)
        files = [files[i] for i in sorted(indices)]
    result = []
    for f in files:
        img = cv2.imread(str(directory / f))
        if img is not None:
            result.append((f, img))
    return result


def load_frames_from_video(
    video_path: str,
    max_frames: int = 200,
    stride: int = 1,
) -> List[Tuple[int, np.ndarray]]:
    """Load frames from a video file at a given stride."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------
def save_results(name: str, data: Any):
    """Save benchmark results as JSON."""
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  → Saved {path}")
    return path


def bbox_height(box) -> int:
    """Height of a bounding box in pixels (supports various formats)."""
    if len(box) == 4:
        # xyxy
        return abs(int(box[3]) - int(box[1]))
    return 0


HEIGHT_BUCKETS = [
    ("<30px", 0, 30),
    ("30-49px", 30, 50),
    ("50-79px", 50, 80),
    ("80-119px", 80, 120),
    ("120-199px", 120, 200),
    ("200+px", 200, 99999),
]

FACE_SIZE_BUCKETS = [
    ("<32px", 0, 32),
    ("32-49px", 32, 50),
    ("50-64px", 50, 65),
    ("65-79px", 65, 80),
    ("80-119px", 80, 120),
    ("120+px", 120, 99999),
]


def bucket_value(value: float, buckets: list) -> str:
    """Assign a numeric value to a named bucket."""
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][0]
