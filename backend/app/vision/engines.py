"""
TensorRT engine cache for the YOLO models.

WHY
On this card yolov8n is 6.35 ms per frame at batch 1 in PyTorch and 2.59 ms as
a TensorRT engine (bench_trt_NVIDIA_GeForce_RTX_5070.json); at batch 16 it is
1.26 ms against 1.08 ms. The engine is not the whole Phase 2 story -- the
20-camera baseline has the GPU 67 percent idle, so inference time is not what
is capping throughput -- but it is a prerequisite for item 3, where one
inference loop runs a batch of up to 32 for every camera at once. A batched
PyTorch call would leave most of the win on the table.

ENGINES ARE BUILD ARTEFACTS, NOT SOURCE
A TensorRT engine is specific to the GPU, the driver, the TensorRT version and
the build parameters. Shipping one, or reusing a stale one, produces either a
hard failure or -- worse -- a silent fallback. So an engine here is addressed
by everything it depends on:

    yolov8n__NVIDIA_GeForce_RTX_5070__trt10.13.3.9__b32_640_fp16.engine

If any of those change the name changes, the file is missing, and it is built
again. `*.engine` and trt_cache/ are gitignored.

BUILDING IS SLOW AND HAPPENS ONCE
An export takes minutes. It happens on first use, is written to a temporary
name and renamed into place atomically, so a half-built engine is never
loaded, and concurrent camera workers racing to build the same engine end up
with one good file rather than a corrupt one.
"""

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_build_lock = threading.Lock()
_resolved: dict = {}          # cache key -> engine path (or None if unusable)


def cache_root() -> Path:
    return Path(os.environ.get("TRT_CACHE_DIR", "trt_cache"))


def cache_dir() -> Path:
    """Where ultralytics engines live.

    A subdirectory, because trt_cache/ is also where ONNX Runtime's
    TensorrtExecutionProvider writes ITS engine cache for SCRFD and ArcFace
    (item 2). Those files are named by ORT's own graph hash, are not ours to
    manage, and must not be swept up by describe() or invalidated by our
    naming scheme.
    """
    d = cache_root() / "yolo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown-gpu"


def _trt_version() -> str:
    try:
        import tensorrt
        return tensorrt.__version__
    except Exception:
        return "none"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def engine_path(weights: str, batch: int, imgsz: int, half: bool) -> Path:
    """Where the engine for this (weights, GPU, TRT, params) combination lives."""
    stem = Path(weights).stem
    name = (
        f"{_slug(stem)}__{_slug(_gpu_name())}__trt{_slug(_trt_version())}"
        f"__b{batch}_{imgsz}_{'fp16' if half else 'fp32'}.engine"
    )
    return cache_dir() / name


def _write_sidecar(path: Path, weights: str, batch: int, imgsz: int,
                   half: bool, build_sec: float):
    """Record what this engine is, next to it.

    A .engine is opaque. Without this, an engine in a cache directory cannot
    be told apart from one built for a different card or a different batch
    profile except by loading it and hoping.
    """
    meta = {
        "weights": str(weights),
        "gpu": _gpu_name(),
        "tensorrt": _trt_version(),
        "batch_max": batch,
        "imgsz": imgsz,
        "precision": "fp16" if half else "fp32",
        "dynamic": True,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_seconds": round(build_sec, 1),
    }
    try:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    except Exception as e:
        logger.debug(f"could not write engine sidecar: {e}")


def build_engine(weights: str, batch: int = 32, imgsz: int = 640,
                 half: bool = True) -> Optional[Path]:
    """Export `weights` to a TensorRT engine. Returns the path, or None.

    Returns None rather than raising: a failed engine build must degrade to
    the PyTorch path, not stop a camera from starting. The reason is logged
    at error level because a silent fall back to a slower path is exactly the
    failure mode Phase 1 item 3 exists to prevent.
    """
    target = engine_path(weights, batch, imgsz, half)
    if target.exists():
        return target

    with _build_lock:
        if target.exists():           # another thread won the race
            return target
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics is not installed; cannot build %s", target.name)
            return None

        logger.info(
            f"Building TensorRT engine {target.name} "
            f"(fp16={half}, dynamic batch<= {batch}, imgsz={imgsz}). "
            f"This takes minutes and happens once per GPU."
        )
        t0 = time.perf_counter()
        try:
            model = YOLO(weights)
            produced = model.export(
                format="engine",
                half=half,
                dynamic=True,      # so one engine serves batch 1..batch
                batch=batch,
                imgsz=imgsz,
                device=0,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"TensorRT export failed for {weights}: {e}")
            return None

        build_sec = time.perf_counter() - t0
        produced = Path(produced) if produced else None
        if not produced or not produced.exists():
            logger.error(f"TensorRT export reported success but produced no file "
                         f"for {weights}")
            return None

        # Move into the cache under the fully-qualified name, atomically. A
        # partially written engine must never be loadable.
        tmp = target.with_suffix(".engine.partial")
        try:
            shutil.move(str(produced), str(tmp))
            os.replace(tmp, target)
        except Exception as e:
            logger.error(f"could not install engine {produced} -> {target}: {e}")
            return None

        _write_sidecar(target, weights, batch, imgsz, half, build_sec)
        logger.info(f"Built {target.name} in {build_sec:.0f}s")
        return target


def get_engine(weights: str, batch: int = 32, imgsz: int = 640,
               half: bool = True, build: bool = True) -> Optional[str]:
    """Path to a usable engine for `weights`, building it if needed.

    Memoised per process, including the failure, so twenty camera workers do
    not each retry a build that cannot succeed.
    """
    key = (str(weights), batch, imgsz, half)
    if key in _resolved:
        return _resolved[key]

    target = engine_path(weights, batch, imgsz, half)
    result = None
    if target.exists():
        result = str(target)
    elif build:
        built = build_engine(weights, batch, imgsz, half)
        result = str(built) if built else None
    _resolved[key] = result
    return result


def resolve_weights(weights: str) -> str:
    """The path a YOLO() call should load: an engine if we can, else the .pt.

    This is the only function the rest of the codebase calls. It never raises
    and never returns something unloadable -- the worst case is the .pt it was
    given, which is the pre-item-1 behaviour. A missing engine is a slower
    pipeline; a raised exception here would be a camera that does not start.
    """
    try:
        from app.config import settings
        if not getattr(settings, "USE_TENSORRT_ENGINES", False):
            return weights
        # The per-camera path specifically. Engines are 2.4x faster in
        # isolation and 26% slower at 20 concurrent contexts; see
        # settings.TRT_PERCAM_ENGINES. Item 3's server calls get_engine()
        # directly and is unaffected by this.
        if not getattr(settings, "TRT_PERCAM_ENGINES", False):
            return weights
        if str(weights).endswith(".engine"):
            return weights          # already an engine
        if not Path(weights).exists():
            # A bare name like "yolov8n.pt" that ultralytics would download.
            # Let it: exporting from a file we do not have yet cannot work.
            if not Path(weights).name == str(weights):
                return weights
        got = get_engine(
            weights,
            # The per-camera profile. NOT TRT_MAX_BATCH: a batch-32 context
            # costs 1,131 MiB against 50 MiB for batch 1, and there is one
            # context per camera until item 3 lands.
            batch=int(getattr(settings, "TRT_PERCAM_BATCH", 1)),
            imgsz=int(getattr(settings, "PERSON_DET_IMGSZ", 640)),
            half=bool(getattr(settings, "PERSON_DET_HALF", True)),
            build=bool(getattr(settings, "TRT_BUILD_ON_START", True)),
        )
        return got or weights
    except Exception as e:
        logger.error(f"engine resolution failed for {weights}, using PyTorch: {e}")
        return weights


def describe() -> list:
    """Every engine in the cache, with its sidecar. For the load test report."""
    out = []
    for p in sorted(cache_dir().glob("*.engine")):
        meta = {}
        side = p.with_suffix(".json")
        if side.exists():
            try:
                meta = json.loads(side.read_text())
            except Exception:
                pass
        out.append({"engine": p.name, "size_mb": round(p.stat().st_size / 2**20, 1),
                    **meta})
    return out
