"""
Item 1 -- TensorRT engine cache for the YOLO models.

A TensorRT engine is a build artefact tied to the GPU, the driver, the
TensorRT version and the build parameters. The failure this guards against is
not "the build broke" -- that is loud. It is loading an engine built for
something else, which either dies at load or, worse, silently does not, and
the two ways this codebase has already been bitten by silent fallbacks
(onnxruntime on CPU, ultralytics defaults) are both in Phase 1's report.

So the engine's identity is its filename, and these tests pin that: change the
card, the TensorRT version, the batch profile, the input size or the
precision, and you get a different path, a cache miss, and a rebuild.
"""

import json
from pathlib import Path

import pytest

from app.config import Settings, settings
from app.vision import engines


# ── identity ──────────────────────────────────────────────────────────────

def test_engine_name_carries_everything_it_depends_on():
    p = engines.engine_path("yolov8n.pt", batch=32, imgsz=640, half=True).name
    assert "yolov8n" in p
    assert "trt" in p                      # TensorRT version
    assert "b32" in p and "640" in p       # build profile
    assert "fp16" in p
    assert p.endswith(".engine")


@pytest.mark.parametrize("a,b", [
    (dict(batch=32, imgsz=640, half=True), dict(batch=16, imgsz=640, half=True)),
    (dict(batch=32, imgsz=640, half=True), dict(batch=32, imgsz=1024, half=True)),
    (dict(batch=32, imgsz=640, half=True), dict(batch=32, imgsz=640, half=False)),
])
def test_different_build_parameters_are_different_engines(a, b):
    """A cache hit on an engine built with a different profile is the bug
    this naming exists to make impossible."""
    assert engines.engine_path("yolov8n.pt", **a) != engines.engine_path("yolov8n.pt", **b)


def test_different_weights_are_different_engines():
    assert (engines.engine_path("yolov8n.pt", 32, 640, True)
            != engines.engine_path("yolo11m.pt", 32, 640, True))


def test_gpu_name_is_in_the_path(monkeypatch):
    """Engines are per GPU. Moving this cache to another machine must miss."""
    monkeypatch.setattr(engines, "_gpu_name", lambda: "NVIDIA_Test_Card_9000")
    assert "NVIDIA_Test_Card_9000" in engines.engine_path("yolov8n.pt", 32, 640, True).name


def test_paths_are_filesystem_safe(monkeypatch):
    """torch reports GPU names with spaces, and TensorRT versions with dots."""
    monkeypatch.setattr(engines, "_gpu_name", lambda: "NVIDIA GeForce RTX 5070")
    name = engines.engine_path("yolov8n.pt", 32, 640, True).name
    assert " " not in name
    assert "/" not in name and "\\" not in name


# ── the yolo cache is not ORT's cache ─────────────────────────────────────

def test_ultralytics_engines_live_apart_from_ort_cache():
    """trt_cache/ is also where ONNX Runtime's TensorrtExecutionProvider
    writes its own engines for SCRFD and ArcFace (item 2). Those are named by
    ORT's graph hash, are not ours to manage, and must not be swept up by
    describe() or invalidated by our naming."""
    assert engines.cache_dir() != engines.cache_root()
    assert engines.cache_dir().parent == engines.cache_root()


# ── degradation ───────────────────────────────────────────────────────────

def test_resolve_returns_the_pt_when_engines_are_off(monkeypatch):
    monkeypatch.setattr(settings, "USE_TENSORRT_ENGINES", False)
    assert engines.resolve_weights("yolov8n.pt") == "yolov8n.pt"


def test_percam_engines_are_off_by_default():
    """Measured: 2.4x faster in isolation, 26 percent SLOWER at 20 concurrent
    contexts. The per-camera path stays on PyTorch until item 3 gives the
    engine a single consumer."""
    assert Settings().TRT_PERCAM_ENGINES is False


def test_percam_flag_gates_the_per_camera_path(monkeypatch):
    monkeypatch.setattr(settings, "USE_TENSORRT_ENGINES", True)
    monkeypatch.setattr(settings, "TRT_PERCAM_ENGINES", False)
    monkeypatch.setattr(engines, "get_engine",
                        lambda *a, **k: pytest.fail("must not resolve an engine"))
    assert engines.resolve_weights("yolov8n.pt") == "yolov8n.pt"


def test_resolve_passes_an_engine_through_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "USE_TENSORRT_ENGINES", True)
    monkeypatch.setattr(settings, "TRT_PERCAM_ENGINES", True)
    assert engines.resolve_weights("already.engine") == "already.engine"


def test_a_failed_build_degrades_to_pytorch(monkeypatch):
    """A camera that will not start is worse than a camera running the slower
    path. resolve_weights must never raise and never return something
    unloadable."""
    monkeypatch.setattr(settings, "USE_TENSORRT_ENGINES", True)
    monkeypatch.setattr(settings, "TRT_PERCAM_ENGINES", True)
    monkeypatch.setattr(engines, "get_engine", lambda *a, **k: None)
    assert engines.resolve_weights("yolov8n.pt") == "yolov8n.pt"


def test_an_exception_anywhere_degrades_to_pytorch(monkeypatch):
    monkeypatch.setattr(settings, "USE_TENSORRT_ENGINES", True)
    monkeypatch.setattr(settings, "TRT_PERCAM_ENGINES", True)

    def boom(*a, **k):
        raise RuntimeError("tensorrt exploded")

    monkeypatch.setattr(engines, "get_engine", boom)
    assert engines.resolve_weights("yolov8n.pt") == "yolov8n.pt"


def test_get_engine_memoises_failure(monkeypatch):
    """Twenty camera workers must not each retry a build that cannot work."""
    engines._resolved.clear()
    calls = []

    def fake_build(*a, **k):
        calls.append(1)
        return None

    monkeypatch.setattr(engines, "build_engine", fake_build)
    monkeypatch.setattr(engines, "engine_path",
                        lambda *a, **k: Path("does-not-exist.engine"))
    for _ in range(5):
        engines.get_engine("yolov8n.pt")
    assert len(calls) == 1
    engines._resolved.clear()


def test_build_is_skipped_when_asked_not_to(monkeypatch):
    engines._resolved.clear()
    monkeypatch.setattr(engines, "engine_path",
                        lambda *a, **k: Path("does-not-exist.engine"))
    monkeypatch.setattr(engines, "build_engine",
                        lambda *a, **k: pytest.fail("must not build"))
    assert engines.get_engine("yolov8n.pt", build=False) is None
    engines._resolved.clear()


# ── config ────────────────────────────────────────────────────────────────

def test_settings_exist_with_the_documented_defaults():
    s = Settings()
    assert s.USE_TENSORRT_ENGINES is True
    assert s.TRT_MAX_BATCH == 32       # item 3 batches up to this
    assert s.TRT_PERCAM_BATCH == 1     # 50 MiB a context, not 1,131
    assert s.TRT_BUILD_ON_START is True
