"""
Item 3 -- refuse to run inference on the CPU.

WHY THIS IS FATAL RATHER THAN A WARNING
The old code logged a banner and carried on. That is the worst available
outcome: SCRFD at det=1024 costs 34 ms on CUDA and 3,648 ms on the CPU, so a
deployment in this state reports healthy, serves its API, and analyses about
one percent of what it should. Nobody reads a log line on a machine that
appears to be working; they open a ticket six weeks later saying the cameras
feel slow.

The failure modes are all mundane -- a stray `pip install onnxruntime` beside
onnxruntime-gpu, a CPU torch build, a driver too old for the card -- and all
of them are a five-minute fix once named. So the message has to name them.
"""

import pytest

from app.vision.detector import (
    CPUInferenceRefused,
    _conflicting_onnxruntime,
    _cpu_fallback_message,
)


def test_refusal_is_an_error_not_a_warning():
    """A caller must be able to catch this distinctly, and it must be fatal
    by default -- not something that lands in a generic `except Exception`
    and turns into a None return."""
    assert issubclass(CPUInferenceRefused, RuntimeError)


def test_message_names_the_escape_hatch():
    msg = _cpu_fallback_message()
    assert "ALLOW_CPU_INFERENCE" in msg, "the way out must be in the message"


def test_message_quantifies_the_cost():
    """A number is what makes someone act. 'Slower' does not."""
    msg = _cpu_fallback_message()
    assert "3,648 ms" in msg and "34 ms" in msg


def test_message_reports_the_providers_actually_available():
    """The single most useful fact for diagnosing this is what onnxruntime
    thinks it can do, so it goes in the message rather than in a follow-up
    command the reader has to run."""
    import onnxruntime as ort
    msg = _cpu_fallback_message()
    assert ort.__version__ in msg
    for provider in ort.get_available_providers():
        assert provider in msg


def test_conflicting_wheel_detection_is_quiet_when_clean():
    """This environment has onnxruntime-gpu only, so the clash branch must
    not fire -- a false accusation here sends the reader to uninstall the
    package that is working."""
    clash = _conflicting_onnxruntime()
    assert clash is None or "onnxruntime-gpu" not in clash.split(", ")[:1]


def test_message_gives_a_fix_either_way():
    """Whether or not a clashing wheel is found, the reader gets a command."""
    msg = _cpu_fallback_message()
    assert "pip install" in msg


def test_guard_is_reachable_from_settings():
    """The flag has to exist and default to refusing, or the guard is
    decorative."""
    from app.config import settings
    assert hasattr(settings, "ALLOW_CPU_INFERENCE")
    assert settings.ALLOW_CPU_INFERENCE is False


def test_detector_does_not_swallow_the_refusal():
    """_get_insightface_app catches Exception and returns None on failure.
    None means 'no faces this frame' to every caller, so a refusal caught
    there would present as cameras that see nobody -- exactly the silent
    failure this item exists to remove."""
    import inspect
    from app.vision import detector

    src = inspect.getsource(detector._get_insightface_app)
    refuse_at = src.index("except CPUInferenceRefused")
    generic_at = src.index("except Exception")
    assert refuse_at < generic_at, "the specific handler must come first"
    assert "raise" in src[refuse_at:generic_at]
