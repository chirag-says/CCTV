"""
Step 1 -- tracking owns existence, recognition owns identity.

The behaviour under test is the architectural correction: a track is a PERSON,
and a face is an attribute that person sometimes has. Losing the face costs the
embedding for that frame, not the track and not the identity.

The helpers are exercised against a stub rather than a real VisionPipeline,
whose __init__ loads SCRFD + ArcFace.
"""

import numpy as np
import pytest

from app.vision.bytetrack import ByteTracker
from app.vision.pipeline import VisionPipeline


class _Stub:
    """Carries only what the Step 1 helpers touch."""

    camera_id = "cam-test"
    _person_fallback_logged = False
    _frame_person_dets = []

    _tracking_detections = VisionPipeline._tracking_detections
    _bind_face_to_track = VisionPipeline._bind_face_to_track

    def __init__(self, mode="person", persons=None):
        self._tracking_mode = mode
        self._persons = persons or []

    def _load_yolo_person_detector(self):
        return True

    def _get_yolo_person_detections(self, frame):
        return self._persons


def _person(x1, y1, x2, y2, conf=0.9):
    return {"confidence": conf, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def _face(top, right, bottom, left):
    return (top, right, bottom, left)


class _FakeTrack:
    def __init__(self, top, right, bottom, left):
        self._loc = (top, right, bottom, left)

    def to_location(self):
        return self._loc


# -- Detection source --------------------------------------------


class TestTrackingDetections:
    def test_person_mode_tracks_bodies_not_faces(self):
        stub = _Stub(persons=[_person(100, 50, 200, 400)])
        faces = [_face(60, 170, 110, 120)]
        boxes, scores, source = stub._tracking_detections(None, faces)

        assert source == "person"
        assert boxes.shape == (1, 4)
        np.testing.assert_allclose(boxes[0], [100, 50, 200, 400])

    def test_person_mode_with_no_people_returns_empty_not_faces(self):
        """
        Falling back to face boxes on a frame where YOLO found nobody would
        change what a track box MEANS mid-stream and churn every track id.
        """
        stub = _Stub(persons=[])
        faces = [_face(60, 170, 110, 120)]
        boxes, scores, source = stub._tracking_detections(None, faces)

        assert source == "person"
        assert len(boxes) == 0, "must not silently switch to face boxes"

    def test_low_confidence_people_are_dropped(self):
        stub = _Stub(persons=[_person(0, 0, 10, 10, conf=0.05)])
        boxes, _s, _src = stub._tracking_detections(None, [])
        assert len(boxes) == 0

    def test_face_mode_tracks_faces(self):
        stub = _Stub(mode="face")
        faces = [_face(60, 170, 110, 120)]
        boxes, scores, source = stub._tracking_detections(None, faces)

        assert source == "face"
        # (top, right, bottom, left) -> (x1, y1, x2, y2)
        np.testing.assert_allclose(boxes[0], [120, 60, 170, 110])

    def test_falls_back_to_face_mode_when_yolo_unavailable(self):
        stub = _Stub()
        stub._load_yolo_person_detector = lambda: False
        faces = [_face(60, 170, 110, 120)]
        _b, _s, source = stub._tracking_detections(None, faces)

        assert source == "face"
        assert stub._tracking_mode == "face", "the demotion must be permanent"


# -- Face binding ------------------------------------------------


class TestBindFaceToTrack:
    def _bind(self, stub, track, faces, embs=None, lms=None, source="person"):
        if embs is None:
            embs = [np.ones(4, dtype=np.float32)] * len(faces)
        if lms is None:
            lms = [None] * len(faces)
        return stub._bind_face_to_track(track, faces, embs, lms, source)

    def test_binds_face_inside_person_box(self):
        stub = _Stub()
        track = _FakeTrack(50, 200, 400, 100)  # body
        face = _face(60, 170, 110, 120)  # head, inside it
        loc, emb, _lm = self._bind(stub, track, [face])

        assert loc == face
        assert emb is not None

    def test_returns_none_when_facing_away(self):
        """The ordinary state Step 1 exists to survive -- not an error."""
        stub = _Stub()
        track = _FakeTrack(50, 200, 400, 100)
        loc, emb, lm = self._bind(stub, track, [])
        assert (loc, emb, lm) == (None, None, None)

    def test_does_not_bind_a_neighbouring_face(self):
        stub = _Stub()
        track = _FakeTrack(50, 200, 400, 100)
        far_face = _face(60, 900, 110, 850)
        loc, emb, _lm = self._bind(stub, track, [far_face])
        assert loc is None, "a face outside the body box is somebody else"

    def test_largest_face_wins(self):
        """Ties go to the nearer person, which is also the better embedding."""
        stub = _Stub()
        track = _FakeTrack(50, 300, 400, 100)
        small = _face(60, 150, 90, 120)
        large = _face(60, 260, 140, 180)
        loc, _e, _l = self._bind(stub, track, [small, large])
        assert loc == large

    def test_skips_faces_without_embeddings(self):
        stub = _Stub()
        track = _FakeTrack(50, 200, 400, 100)
        face = _face(60, 170, 110, 120)
        loc, emb, _lm = self._bind(stub, track, [face], embs=[None])
        assert loc is None

    def test_face_mode_uses_iou(self):
        stub = _Stub(mode="face")
        face = _face(60, 170, 110, 120)
        track = _FakeTrack(*face)  # in face mode the track IS the face box
        loc, emb, _lm = self._bind(stub, track, [face], source="face")
        assert loc == face

    def test_face_mode_ignores_non_overlapping(self):
        stub = _Stub(mode="face")
        track = _FakeTrack(60, 170, 110, 120)
        other = _face(600, 700, 650, 650)
        loc, _e, _l = self._bind(stub, track, [other], source="face")
        assert loc is None


# -- The regression this whole step is about ---------------------


class TestExistenceSurvivesFaceLoss:
    def test_person_track_survives_when_face_disappears(self):
        """
        Face mode: no faces -> tracker gets nothing -> the person ceases to
        exist. Person mode: the body is still detected, so the track lives.
        """
        tracker = ByteTracker()
        stub = _Stub(persons=[_person(100, 50, 200, 400)])

        # Walk in, face visible.
        faces = [_face(60, 170, 110, 120)]
        for _ in range(5):
            boxes, scores, _src = stub._tracking_detections(None, faces)
            active = tracker.update(boxes, scores)
        assert len(active) == 1
        active[0].person_id, active[0].person_name = "p1", "Rahul"
        track_id = active[0].track_id

        # Turns around: body still detected, ZERO faces.
        for _ in range(10):
            boxes, scores, _src = stub._tracking_detections(None, [])
            active = tracker.update(boxes, scores)

        assert len(active) == 1, "the person did not leave; the track must live"
        assert active[0].track_id == track_id, "and must keep its id"
        assert active[0].person_name == "Rahul", "and its identity"

        # No face is bound, so no recognition would run this frame.
        loc, emb, _lm = stub._bind_face_to_track(active[0], [], [], [], "person")
        assert loc is None and emb is None

    def test_face_mode_loses_the_track(self):
        """Pins the old behaviour, so the contrast stays honest."""
        tracker = ByteTracker()
        stub = _Stub(mode="face")

        faces = [_face(60, 170, 110, 120)]
        for _ in range(5):
            boxes, scores, _s = stub._tracking_detections(None, faces)
            active = tracker.update(boxes, scores)
        assert len(active) == 1

        for _ in range(tracker.tracked_grace + 3):
            boxes, scores, _s = stub._tracking_detections(None, [])
            active = tracker.update(boxes, scores)

        assert active == [], "face mode drops the person -- the bug Step 1 fixes"
