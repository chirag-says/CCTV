"""
Item 6 -- analysis rate by wall clock rather than frame count.

WHAT WAS WRONG WITH FRAME_SKIP
It analysed every Nth DECODED frame, so the actual analysis rate was the
camera's frame rate divided by N -- a number nothing in the config stated and
nobody controlled. A 25 fps camera and a 30 fps camera on FRAME_SKIP=4 ran at
6.25 and 7.5 analyses per second. A camera that dropped to 12 fps silently
halved its own detection rate. And every capacity figure this project has is
quoted per second ("20 Basic cameras at 10 fps"), so the one unit the code
could not express was the unit the plan was written in.

These tests call the two predicates as unbound methods against a stub, which
keeps them honest without loading SCRFD, YOLO and ReID to check arithmetic.
"""

import time
import types

import pytest

from app.config import settings
from app.vision.pipeline import VisionPipeline


def stub(fps=10.0, frame_skip_override=None, camera_id="cam-1"):
    """The attributes _analysis_due actually reads, and nothing else."""
    s = types.SimpleNamespace()
    s.camera_id = camera_id
    s._analysis_fps = fps
    s._analysis_interval = 1.0 / fps if fps > 0 else 0.0
    s._last_analysis_at = 0.0
    s._frame_skip_override = frame_skip_override
    s._frame_count = 0
    return s


def due(s, consume=True):
    return VisionPipeline._analysis_due(s, consume=consume)


# ── the rate is the rate ──────────────────────────────────────────────────

def test_first_frame_is_always_analysed():
    """A camera that just started must not sit out an interval before it
    looks at anything."""
    assert due(stub()) is True


def test_a_burst_of_frames_yields_one_analysis():
    """The whole point: arrival count does not set the rate. Before this,
    200 frames at FRAME_SKIP=4 meant 50 analyses no matter how fast they
    came in."""
    s = stub(fps=10.0)
    analysed = sum(1 for _ in range(200) if due(s))
    assert analysed == 1, f"{analysed} analyses from one instant's worth of frames"


def test_rate_is_held_over_real_time():
    s = stub(fps=50.0)          # 20 ms apart, so the test stays quick
    analysed = 0
    start = time.monotonic()
    # Busy poll deliberately. time.sleep() on Windows rounds up to the ~15 ms
    # timer tick, which would cap the loop at fewer iterations than the rate
    # being tested and measure the sleep clock rather than the predicate.
    while time.monotonic() - start < 0.25:
        if due(s):
            analysed += 1
    # 0.25 s at 50 fps is 12-13 analyses. Allow for scheduler noise but not
    # for being off by a factor.
    assert 9 <= analysed <= 16, analysed


def test_rate_is_independent_of_source_frame_rate():
    """The same target must produce the same analysis count whether frames
    arrive at 15 fps or 60 fps. This is the property FRAME_SKIP lacked."""
    counts = []
    # Both intervals stay well clear of the ~15 ms Windows timer tick, so the
    # two arms really do differ in arrival rate rather than both collapsing to
    # the granularity floor.
    for source_interval in (1 / 12.0, 1 / 30.0):
        s = stub(fps=10.0)
        analysed = 0
        start = time.monotonic()
        while time.monotonic() - start < 0.4:
            if due(s):
                analysed += 1
            time.sleep(source_interval)
        counts.append(analysed)
    assert abs(counts[0] - counts[1]) <= 2, counts


def test_zero_fps_means_analyse_everything():
    s = stub(fps=0)
    assert all(due(s) for _ in range(10))


def test_peeking_does_not_consume_the_interval():
    """The grabber asks without consuming, so asking must not starve the
    analyser of the frame it was about to take."""
    s = stub(fps=10.0)
    assert due(s, consume=False) is True
    assert due(s, consume=False) is True, "a peek must not spend the interval"
    assert due(s, consume=True) is True
    assert due(s, consume=True) is False, "consuming must spend it"


# ── the load tier still has a say ─────────────────────────────────────────

def test_load_tier_override_lengthens_the_interval():
    """_apply_load_degradation raises the frame skip under burst load. That
    request has to survive the move to wall clock, or the degradation tier
    silently stopped working."""
    base = stub(fps=10.0)
    throttled = stub(fps=10.0, frame_skip_override=settings.FRAME_SKIP * 3)
    due(base)
    due(throttled)
    time.sleep(0.12)            # past 1/10 s, short of 3/10 s
    assert due(base) is True
    assert due(throttled) is False


def test_override_below_default_does_not_speed_things_up():
    """A skip override smaller than the configured default must not be read
    as permission to analyse faster than the plan allows."""
    s = stub(fps=10.0, frame_skip_override=1)
    due(s)
    assert due(s) is False


# ── the legacy path stays reachable ───────────────────────────────────────

def test_legacy_flag_restores_frame_counting(monkeypatch):
    monkeypatch.setattr(settings, "ANALYSIS_RATE_BY_CLOCK", False)
    monkeypatch.setattr(settings, "FRAME_SKIP", 4)
    s = stub(fps=10.0)
    got = []
    for n in range(1, 13):
        s._frame_count = n
        got.append(due(s))
    # Every 4th frame, by count, with no reference to the clock at all.
    assert got == [False, False, False, True] * 3


# ── per-camera resolution ─────────────────────────────────────────────────

def resolve(camera_id, config, monkeypatch):
    """Run _resolve_analysis_fps with a stubbed camera row."""
    import app.db.session as session_mod

    class _Row:
        def __init__(self, cfg):
            self.config = cfg

    class _Query:
        def __init__(self, cfg):
            self._cfg = cfg

        def filter(self, *_a):
            return self

        def first(self):
            return _Row(self._cfg) if self._cfg is not None else None

    class _DB:
        def __init__(self, cfg):
            self._cfg = cfg

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def query(self, *_a):
            return _Query(self._cfg)

    monkeypatch.setattr(session_mod, "SessionLocal", lambda: _DB(config))
    s = types.SimpleNamespace(camera_id=camera_id)
    return VisionPipeline._resolve_analysis_fps(s)


def test_plain_camera_gets_the_global_default(monkeypatch):
    assert resolve("cam-1", {}, monkeypatch) == pytest.approx(settings.ANALYSIS_FPS)


def test_gate_camera_gets_the_gate_default(monkeypatch):
    """is_gate is the convention already used by journey_service and
    presence_service; reusing it beats inventing a second marker."""
    got = resolve("cam-1", {"is_gate": True}, monkeypatch)
    assert got == pytest.approx(settings.ANALYSIS_FPS_GATE)
    assert got < settings.ANALYSIS_FPS


def test_explicit_config_beats_the_gate_default(monkeypatch):
    assert resolve("cam-1", {"is_gate": True, "analysis_fps": 12}, monkeypatch) == 12.0


def test_missing_camera_row_falls_back_rather_than_failing(monkeypatch):
    """The load test and browser sessions have no camera row. That is normal,
    not an error, and must not stop a pipeline from starting."""
    assert resolve("nonexistent", None, monkeypatch) == pytest.approx(settings.ANALYSIS_FPS)


def test_garbage_config_value_falls_back(monkeypatch):
    assert resolve("cam-1", {"analysis_fps": "fast"}, monkeypatch) == pytest.approx(
        settings.ANALYSIS_FPS
    )


def test_db_failure_falls_back(monkeypatch):
    import app.db.session as session_mod

    def boom():
        raise RuntimeError("database is down")

    monkeypatch.setattr(session_mod, "SessionLocal", boom)
    s = types.SimpleNamespace(camera_id="cam-1")
    assert VisionPipeline._resolve_analysis_fps(s) == pytest.approx(settings.ANALYSIS_FPS)


# ── the grabber's drop gate ───────────────────────────────────────────────

def wanted(s):
    # _frame_wanted consults _analysis_due on self, so the stub needs it bound.
    s._analysis_due = lambda consume=False: VisionPipeline._analysis_due(
        s, consume=consume
    )
    return VisionPipeline._frame_wanted(s)


def test_grabber_keeps_every_frame_in_legacy_mode(monkeypatch):
    """REGRESSION. The grabber drops frames nothing will consume, but in
    legacy FRAME_SKIP mode "due" means `_frame_count % skip == 0` and
    _frame_count only advances inside process_frame. Asking before anything
    is consumed freezes the counter: frame 1 fails, so the grabber drops it,
    so process_frame never runs, so the counter never moves, and every later
    frame fails identically.

    Measured when this was live: 0.00 analysis fps and 1.3% GPU across all
    20 cameras -- the flag documented as the rollback path was a total
    outage. Only a wall-clock rate can be evaluated by a thread that does
    not consume, because only the clock advances by itself.
    """
    monkeypatch.setattr(settings, "ANALYSIS_RATE_BY_CLOCK", False)
    monkeypatch.setattr(settings, "FRAME_SKIP", 4)
    s = stub(fps=10.0)
    s._frame_count = 1          # the frame after the first consumed one
    assert wanted(s) is True, "legacy mode must never drop in the grabber"
    for n in range(1, 20):
        s._frame_count = n
        assert wanted(s) is True


def test_grabber_drops_unwanted_frames_in_clock_mode(monkeypatch):
    """The saving this item exists for: a frame neither due nor watched costs
    a 1080p resize, a redraw and a JPEG handoff downstream."""
    monkeypatch.setattr(settings, "ANALYSIS_RATE_BY_CLOCK", True)
    import app.core.websocket as ws
    monkeypatch.setattr(ws.ws_manager, "is_watched", lambda _cam: False)

    s = stub(fps=10.0)
    assert wanted(s) is True, "first frame is due"
    due(s)                       # consume the interval
    assert wanted(s) is False, "not due and nobody watching"


def test_a_viewer_keeps_frames_flowing_between_analyses(monkeypatch):
    """Display runs at the camera's rate, not the analysis rate, so overlays
    stay smooth while detection runs slower."""
    monkeypatch.setattr(settings, "ANALYSIS_RATE_BY_CLOCK", True)
    import app.core.websocket as ws
    monkeypatch.setattr(ws.ws_manager, "is_watched", lambda _cam: True)

    s = stub(fps=10.0)
    due(s)                       # spend the interval
    assert wanted(s) is True, "a viewer wants frames even when not due"


def test_a_bookkeeping_failure_never_drops_frames(monkeypatch):
    """Losing video because a viewer lookup raised would be a far worse bug
    than the CPU this saves."""
    monkeypatch.setattr(settings, "ANALYSIS_RATE_BY_CLOCK", True)
    import app.core.websocket as ws

    def boom(_cam):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(ws.ws_manager, "is_watched", boom)
    s = stub(fps=10.0)
    due(s)
    assert wanted(s) is True
