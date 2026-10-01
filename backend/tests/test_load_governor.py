"""
Step A -- a process-wide view of load.

THE GAP
`VisionPipeline._apply_load_degradation` tiers on the face count of ITS OWN
camera. With 13 pipelines each seeing about one face, every one concludes
"normal operation" and keeps its full detection rate, while collectively they
oversubscribe the GPU and the GIL roughly threefold. Measured before this
existed: SCRFD p95 of 108.8 ms against an uncontended cost of 9.6-15 ms, with
the GPU at 22% utilisation -- idle, while every camera waited on the others.

No camera can see that from its own face count.

The governor closes the loop on measured detection latency, which needs no
capacity model and cannot go stale when the model, resolution or hardware
changes. These tests pin the control behaviour: it must react, it must settle,
and it must never throttle so hard that detection stops.
"""

import pytest

from app.vision.load_governor import LoadGovernor


def _gov(**kw):
    kw.setdefault("target_ms", 50.0)
    kw.setdefault("max_extra_skip", 6)
    kw.setdefault("adjust_interval", 0.0)   # adjust on every call
    kw.setdefault("window", 300)
    return LoadGovernor(**kw)


def _feed(gov, ms, n=40, camera="cam-a"):
    for _ in range(n):
        gov.record(ms, camera)


class TestThrottlesUnderLoad:
    def test_latency_over_target_raises_the_skip(self):
        gov = _gov()
        _feed(gov, 200.0)
        assert gov.extra_skip() == 1

    def test_it_keeps_climbing_while_load_persists(self):
        gov = _gov()
        for _ in range(4):
            _feed(gov, 200.0)
            gov.extra_skip()
        assert gov.extra_skip() >= 3

    def test_it_never_throttles_past_the_ceiling(self):
        """
        A governor that can shed unlimited work can shed all of it. The cap is
        what keeps a badly-tuned target from turning the cameras off.
        """
        gov = _gov(max_extra_skip=2)
        for _ in range(20):
            _feed(gov, 500.0)
            gov.extra_skip()
        assert gov.extra_skip() == 2


class TestRecoversWhenLoadClears:
    def test_low_latency_lowers_the_skip(self):
        gov = _gov()
        _feed(gov, 200.0)
        gov.extra_skip()
        assert gov._extra_skip == 1

        _feed(gov, 5.0)
        gov.extra_skip()
        assert gov._extra_skip == 0

    def test_it_decays_all_the_way_to_invisible(self):
        gov = _gov()
        for _ in range(5):
            _feed(gov, 300.0)
            gov.extra_skip()
        assert gov._extra_skip > 0

        for _ in range(20):
            _feed(gov, 3.0)
            gov.extra_skip()
        assert gov.extra_skip() == 0, "must not leave a permanent tax"


class TestStability:
    def test_latency_inside_the_dead_band_changes_nothing(self):
        """
        Detection latency is noisy. A governor that chases every fluctuation
        oscillates, which costs more than the load it is shedding.
        """
        gov = _gov(target_ms=50.0)
        _feed(gov, 200.0)
        gov.extra_skip()
        settled = gov._extra_skip

        for ms in (45.0, 50.0, 55.0, 48.0):   # 0.7x..1.2x of target
            _feed(gov, ms)
            gov.extra_skip()
        assert gov._extra_skip == settled

    def test_it_settles_rather_than_oscillating(self):
        """Feed exactly the target: the skip must stop moving."""
        gov = _gov(target_ms=50.0)
        history = []
        for _ in range(15):
            _feed(gov, 50.0)
            history.append(gov.extra_skip())
        assert len(set(history)) == 1

    def test_too_few_samples_is_not_evidence(self):
        gov = _gov()
        gov.record(500.0, "cam-a")
        assert gov.extra_skip() == 0, "one slow frame is not a trend"

    def test_the_adjustment_interval_is_respected(self):
        gov = _gov(adjust_interval=3600.0)
        _feed(gov, 500.0)
        assert gov.extra_skip() == 0, "not due to adjust yet"


class TestDisableAndIntrospect:
    def test_the_governor_can_be_turned_off(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "LOAD_GOVERNOR_ENABLED", False, raising=False)
        gov = _gov()
        _feed(gov, 500.0)
        assert gov.extra_skip() == 0

    def test_active_cameras_are_counted(self):
        gov = _gov()
        for cam in ("cam-a", "cam-b", "cam-c"):
            _feed(gov, 100.0, n=20, camera=cam)
        gov.extra_skip()
        assert gov.stats["active_cameras"] == 3

    def test_stats_report_the_measurement_behind_the_decision(self):
        gov = _gov(target_ms=50.0)
        _feed(gov, 200.0)
        gov.extra_skip()
        s = gov.stats
        assert s["detect_p95_ms"] > 50.0
        assert s["extra_frame_skip"] == 1
        assert s["raised"] == 1

    def test_reset_clears_everything(self):
        gov = _gov()
        _feed(gov, 500.0)
        gov.extra_skip()
        gov.reset()
        assert gov._extra_skip == 0
        assert gov.stats["active_cameras"] == 0
