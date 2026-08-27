"""
Step 0 — coasting: a track ByteTrack has moved to `lost` keeps its identity
and stays drawable at its LAST OBSERVED position for a short window.

These tests pin the two properties that make coasting safe:

  1. It is presentation only. `update()`'s return value is unchanged, so the
     pipeline's recognition / presence loop never sees a coasting track.
  2. It draws the last box that was actually SEEN, never the motion-predicted
     `bbox`, which predict() keeps drifting for the whole time a track is lost.
"""

import time

import numpy as np
import pytest

from app.vision.bytetrack import ByteTracker


def _box(x1, y1, x2, y2):
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


def _run(tracker, bbox, n=1):
    """Feed the same detection n times."""
    out = []
    for _ in range(n):
        out = tracker.update(bbox, np.array([0.9], dtype=np.float32))
    return out


def _establish(tracker, frames=6):
    """Create one moving, activated track and return it."""
    for i in range(frames):
        tracks = tracker.update(
            _box(100 + i * 10, 100, 160 + i * 10, 160),
            np.array([0.9], dtype=np.float32),
        )
    assert tracks, "expected an activated track"
    return tracks[0]


class TestCoasting:
    def test_track_coasts_after_detection_stops(self):
        t = ByteTracker()
        track = _establish(t)
        track.person_id, track.person_name = "p1", "Rahul"

        # Detection stops. Push past tracked_grace so it moves to `lost`.
        for _ in range(t.tracked_grace + 2):
            active = t.update(np.empty((0, 4)), np.empty(0))

        assert active == [], "track should have left the active pool"

        coasting = t.get_coasting_tracks()
        assert len(coasting) == 1, "lost track should still be coastable"
        got, age = coasting[0]
        assert got.person_name == "Rahul", "identity must survive the loss"
        assert age >= 0.0

    def test_coast_box_is_last_observed_not_predicted(self):
        """
        predict() drifts `bbox` forward every frame for lost tracks too.
        Drawing that would put the box somewhere the person was never seen.
        """
        t = ByteTracker()
        track = _establish(t)
        observed = track.last_seen_bbox.copy()

        for _ in range(t.tracked_grace + 10):
            t.update(np.empty((0, 4)), np.empty(0))

        coasting = t.get_coasting_tracks()
        assert coasting, "expected a coasting track"
        got, _age = coasting[0]

        np.testing.assert_allclose(got.last_seen_bbox, observed)
        # The predicted box really has drifted -- i.e. the test is meaningful.
        assert not np.allclose(got.bbox, observed), (
            "predict() should have moved bbox; last_seen_bbox must not follow it"
        )

    def test_coasting_expires(self):
        t = ByteTracker()
        _establish(t)
        for _ in range(t.tracked_grace + 2):
            t.update(np.empty((0, 4)), np.empty(0))

        assert t.get_coasting_tracks(), "should coast inside the window"
        assert t.get_coasting_tracks(max_age_sec=0.0) == [], "zero window disables"

        time.sleep(0.05)
        assert t.get_coasting_tracks(max_age_sec=0.01) == [], "should expire by age"

    def test_active_tracks_are_not_coasting(self):
        """The two pools are disjoint -- otherwise the pipeline double-draws."""
        t = ByteTracker()
        _establish(t)
        active = _run(t, _box(160, 100, 220, 160))
        assert len(active) == 1
        active_ids = {x.track_id for x in active}
        coast_ids = {x.track_id for x, _ in t.get_coasting_tracks()}
        assert active_ids.isdisjoint(coast_ids)

    def test_update_return_value_is_unchanged(self):
        """
        Coasting must NOT leak into update()'s return value: the pipeline runs
        recognition and PersonTracker.on_detection over everything it returns,
        so a coasting track there would mark someone present on a frame where
        nothing was detected.
        """
        t = ByteTracker()
        _establish(t)
        for _ in range(t.tracked_grace + 2):
            active = t.update(np.empty((0, 4)), np.empty(0))
        assert active == []
        assert t.get_coasting_tracks(), "but it is still coastable"

    def test_unactivated_tracks_never_coast(self):
        """A one-frame blip has no public id; drawing it would be noise."""
        t = ByteTracker()
        t.update(_box(10, 10, 40, 40), np.array([0.9], dtype=np.float32))
        for _ in range(t.tracked_grace + 3):
            t.update(np.empty((0, 4)), np.empty(0))
        for track, _ in t.get_coasting_tracks():
            assert track.is_activated

    def test_recovered_track_keeps_its_id(self):
        """Coasting is a view onto state that is still re-associable."""
        t = ByteTracker()
        track = _establish(t)
        original_id = track.track_id
        track.person_name = "Rahul"

        for _ in range(t.tracked_grace + 2):
            t.update(np.empty((0, 4)), np.empty(0))
        assert t.get_coasting_tracks()

        # Person turns back around at roughly the same place.
        active = _run(t, _box(150, 100, 210, 160), n=2)
        assert active, "should re-acquire"
        assert active[0].track_id == original_id, "should not mint a new id"
        assert active[0].person_name == "Rahul"


# ── Drawing path ────────────────────────────────────────────────
# _coasting_boxes / _scale_boxes / _draw_box are exercised against a stub so
# the real VisionPipeline (which loads SCRFD + ArcFace) stays out of the test.

class _StubPipeline:
    """Minimal stand-in carrying only what the drawing helpers touch."""
    camera_id = "cam-test"
    _stream_scale_x = 0.5
    _stream_scale_y = 0.5

    from app.vision.pipeline import VisionPipeline
    _scale_boxes = VisionPipeline._scale_boxes
    _coasting_boxes = VisionPipeline._coasting_boxes
    _draw_box = VisionPipeline._draw_box
    _dashed_rect = staticmethod(VisionPipeline._dashed_rect)


class TestCoastDrawing:
    def _coasting_tracker(self):
        t = ByteTracker()
        track = _establish(t)
        track.person_id, track.person_name = "p1", "Rahul"
        for _ in range(t.tracked_grace + 2):
            t.update(np.empty((0, 4)), np.empty(0))
        return t

    def test_builds_scales_and_draws(self):
        stub = _StubPipeline()
        stub.byte_tracker = self._coasting_tracker()

        boxes = stub._coasting_boxes()
        assert len(boxes) == 1
        location, label, color, conf, coasting = boxes[0]
        assert label == "Rahul"
        assert coasting is True
        assert conf == 0.0

        scaled = stub._scale_boxes(boxes)
        assert len(scaled[0]) == 5, "the coasting flag must survive scaling"
        top, right, bottom, left = scaled[0][0]
        assert (top, right, bottom, left) == (
            int(location[0] * 0.5), int(location[1] * 0.5),
            int(location[2] * 0.5), int(location[3] * 0.5),
        )

        # The real draw call, dashed branch included.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for draw_args in scaled:
            stub._draw_box(frame, *draw_args)
        assert frame.any(), "something should have been rendered"

    def test_live_boxes_still_draw_without_a_coasting_flag(self):
        """Existing 4-tuples must keep working -- hybrid mode still emits them."""
        stub = _StubPipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stub._draw_box(frame, (50, 200, 150, 100), "Priya", (0, 255, 0), 0.93)
        assert frame.any()

    def test_disabled_by_config(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "BYTETRACK_COAST_ENABLED", False, raising=False)
        stub = _StubPipeline()
        stub.byte_tracker = self._coasting_tracker()
        assert stub._coasting_boxes() == []

    def test_unidentified_track_labelled_by_track_id(self):
        t = ByteTracker()
        _establish(t)  # no identity assigned
        for _ in range(t.tracked_grace + 2):
            t.update(np.empty((0, 4)), np.empty(0))
        stub = _StubPipeline()
        stub.byte_tracker = t
        boxes = stub._coasting_boxes()
        assert boxes and boxes[0][1].startswith("Track #")
