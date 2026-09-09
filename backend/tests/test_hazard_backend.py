"""
Item 5 -- YOLO11m for weapons, RF-DETR at nano, and a setting that is read.

THE MEASUREMENT BEHIND THIS
HazardDetector is constructed per camera, and it tried RF-DETR Large first,
unconditionally. Twenty cameras therefore built twenty RF-DETR Large models.
The 20-camera baseline on this 12 GB card sat at 12,024 MB of VRAM with the
GPU only 66% busy: out of memory long before it was out of compute. Three
runs at identical load then came back at 0.46, 0.14 and 2.17 analysis fps --
at the ceiling, throughput stops being reproducible at all.

AND A SETTING THAT DID NOTHING
settings.RFDETR_MODEL_SIZE existed, defaulted to "large", and was never read.
HazardDetector and TrafficMonitor each defaulted the size in their own
signature and the pipeline passed neither, so editing the config changed no
model. A test that only checked the default value would have passed against
that bug, so these check the call sites.
"""

import inspect

import pytest

from app.config import Settings, settings
from app.vision.hazard_detector import HazardDetector


def det(**kw):
    return HazardDetector(on_alert=None, **kw)


# ── defaults ──────────────────────────────────────────────────────────────

def test_yolo_is_tried_first_by_default():
    assert det().prefer == "yolo"


def test_default_weights_are_the_weapons_model():
    assert Settings().WEAPON_MODEL_PATH == "yolo11m.pt"
    assert det().model_path == "yolo11m.pt"


def test_rfdetr_defaults_to_nano():
    """RF-DETR Large does not fit a 12 GB card shared by 20+ cameras."""
    assert Settings().RFDETR_MODEL_SIZE == "nano"
    assert det().rfdetr_model_size == "nano"


# ── load order ────────────────────────────────────────────────────────────

def test_preference_picks_the_order(monkeypatch):
    """Both orders must be reachable, and the preferred backend must be the
    one actually tried first -- not merely recorded."""
    for prefer, expected_first in (("yolo", "yolo"), ("rfdetr", "rfdetr")):
        d = det(prefer=prefer)
        tried = []

        def ok(name):
            def loader():
                tried.append(name)
                d._model_available = True
                d._model_backend = name
                return True
            return loader

        monkeypatch.setattr(d, "_load_yolo", ok("yolo"))
        monkeypatch.setattr(d, "_load_rfdetr", ok("rfdetr"))
        assert d._load_model() is True
        assert tried == [expected_first]
        assert d._model_backend == expected_first


def test_falls_back_to_the_other_backend(monkeypatch):
    """A missing package must not disable hazard detection when the other
    backend is installed."""
    d = det(prefer="yolo")
    monkeypatch.setattr(d, "_load_yolo", lambda: False)   # ultralytics absent

    def rf():
        d._model_available = True
        d._model_backend = "rfdetr"
        return True

    monkeypatch.setattr(d, "_load_rfdetr", rf)
    assert d._load_model() is True
    assert d._model_backend == "rfdetr"


def test_an_exception_in_one_backend_still_tries_the_other(monkeypatch):
    """A corrupt weights file is not a reason to run blind."""
    d = det(prefer="yolo")

    def boom():
        raise RuntimeError("corrupt weights")

    def rf():
        d._model_available = True
        d._model_backend = "rfdetr"
        return True

    monkeypatch.setattr(d, "_load_yolo", boom)
    monkeypatch.setattr(d, "_load_rfdetr", rf)
    assert d._load_model() is True
    assert d._model_backend == "rfdetr"


def test_both_failing_disables_detection_without_raising(monkeypatch):
    """submit_frame runs on a thread pool; raising here would be swallowed
    and the camera would look fine."""
    d = det()
    monkeypatch.setattr(d, "_load_yolo", lambda: False)
    monkeypatch.setattr(d, "_load_rfdetr", lambda: False)
    assert d._load_model() is False
    assert d._model_available is False


def test_load_is_attempted_once(monkeypatch):
    """_load_model is called from _detect on every processed frame."""
    d = det()
    calls = []
    monkeypatch.setattr(d, "_load_yolo", lambda: calls.append(1) or False)
    monkeypatch.setattr(d, "_load_rfdetr", lambda: False)
    for _ in range(5):
        d._load_model()
    assert len(calls) == 1


def test_unknown_preference_falls_back_to_yolo():
    assert det(prefer="tensorrt").prefer == "yolo"


# ── the setting is now actually read ──────────────────────────────────────

def test_pipeline_passes_the_rfdetr_size_to_both_consumers():
    """The bug this item also fixes: RFDETR_MODEL_SIZE was config wired to
    nothing, because both consumers defaulted it themselves and the pipeline
    passed neither."""
    from app.vision import pipeline as pipeline_mod

    src = inspect.getsource(pipeline_mod.VisionPipeline.__init__)
    assert src.count("rfdetr_model_size=settings.RFDETR_MODEL_SIZE") == 2, (
        "HazardDetector and TrafficMonitor must both receive it"
    )
    assert "prefer=getattr(settings, \"HAZARD_BACKEND\"" in src


def test_hazard_backend_setting_exists_and_has_both_values():
    assert Settings().HAZARD_BACKEND == "yolo"
    assert det(prefer="rfdetr").prefer == "rfdetr"


def test_person_detector_can_share_the_hazard_model():
    """With the backend on "yolo", _load_yolo_person_detector reuses this
    model, so a camera loads one detector instead of two. That reuse is
    conditional on the backend name, which is why the default matters."""
    from app.vision import pipeline as pipeline_mod

    src = inspect.getsource(pipeline_mod.VisionPipeline._load_yolo_person_detector)
    assert '_model_backend == "yolo"' in src
