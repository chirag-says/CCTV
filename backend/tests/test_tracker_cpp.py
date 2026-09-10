"""
The compiled association core produces the same track ids as the Python one.

WHY THIS IS THE TEST THAT MATTERS
A tracker that is merely "close" is worse than no port at all. Track ids are
the spine of everything above them -- presence, entry and exit events,
attendance, the identity ledger, the unknown registry. One id that differs is
one person who left and arrived again, and it would show up as a wrong
attendance row days later, not as a failed assertion here.

So the assertion is EQUALITY of the id sequence, frame by frame, over a
recorded 300-frame run of real YOLO person boxes from
uploads/videos/synth_people_720p.mp4 (1,742 boxes, up to 13 people in frame).
Recorded rather than generated, because the interesting cases -- people
crossing, boxes merging, a detection dropping for a frame and coming back --
are what real footage produces and random boxes do not.

The second test pins the reason the first one passes. The C++ assignment
reproduces lap.lapjv's extend_cost/cost_limit construction exactly, and two
details of that construction are easy to get subtly wrong; both shift the
effective IoU threshold rather than failing outright, so both would show up as
occasional id churn in production and nowhere else. See the comment in
_core.cpp.
"""

import json
import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(__file__), "data", "detection_sequence_300.json")

cpp = pytest.importorskip(
    "app.vision.bytetrack_cpp",
    reason="compiled tracker not built: "
    "cd app/vision/bytetrack_cpp && python setup.py build_ext --inplace",
)


@pytest.fixture(scope="module")
def sequence():
    with open(DATA) as fh:
        payload = json.load(fh)
    frames = [
        (
            np.asarray(f["bboxes"], dtype=np.float64).reshape(-1, 4),
            np.asarray(f["scores"], dtype=np.float64).reshape(-1),
        )
        for f in payload["frames"]
    ]
    assert len(frames) == 300, "the stored sequence must stay 300 frames"
    return frames


def _run(frames, backend):
    """Track ids per frame, with the association forced to one backend.

    The tracker resolves its backend at import, so this swaps the module
    global directly -- the alternative is two interpreters, and the point is
    to compare the SAME lifecycle code over two association implementations.
    """
    from app.vision import bytetrack

    original = bytetrack._cpp
    bytetrack._cpp = cpp if backend == "cpp" else None
    try:
        tracker = bytetrack.ByteTracker()
        out = []
        for bboxes, scores in frames:
            tracks = tracker.update(bboxes, scores)
            out.append([t.track_id for t in tracks])
        return out
    finally:
        bytetrack._cpp = original


def test_track_ids_are_identical_over_300_frames(sequence):
    py_ids = _run(sequence, "py")
    cpp_ids = _run(sequence, "cpp")

    assert len(py_ids) == len(cpp_ids) == 300

    for frame_no, (a, b) in enumerate(zip(py_ids, cpp_ids)):
        assert a == b, (
            f"track ids diverge at frame {frame_no}: "
            f"python={a} cpp={b}"
        )

    # A sequence where nothing is ever tracked would pass the loop above
    # while proving nothing.
    assert sum(len(ids) for ids in py_ids) > 0, "the stored sequence tracks nobody"
    assert max(len(ids) for ids in py_ids) > 1, "never more than one track at once"


def test_assignment_matches_lap_exactly():
    """The property the id equality rests on, tested directly.

    Randomised shapes and thresholds, against lap itself. This is where a
    padding mistake shows up as a number instead of as drift.
    """
    lap = pytest.importorskip("lap")
    from app.vision.bytetrack import _linear_assignment

    from app.vision import bytetrack

    rng = np.random.default_rng(7)
    original = bytetrack._cpp
    mismatches = []
    try:
        for _ in range(2000):
            rows = int(rng.integers(0, 12))
            cols = int(rng.integers(0, 12))
            cost = rng.random((rows, cols))
            thresh = float(rng.choice([0.2, 0.5, 0.7, 0.8, 1.0]))

            bytetrack._cpp = None
            py = _linear_assignment(cost, thresh)
            bytetrack._cpp = cpp
            cc = _linear_assignment(cost, thresh)

            same = (
                sorted((int(a), int(b)) for a, b in py[0])
                == sorted((int(a), int(b)) for a, b in cc[0])
                and sorted(map(int, py[1])) == sorted(map(int, cc[1]))
                and sorted(map(int, py[2])) == sorted(map(int, cc[2]))
            )
            if not same:
                mismatches.append((rows, cols, thresh))
    finally:
        bytetrack._cpp = original

    assert not mismatches, f"{len(mismatches)} disagreements with lap, e.g. {mismatches[:3]}"


def test_iou_matches_numpy_bitwise():
    from app.vision import bytetrack

    rng = np.random.default_rng(11)
    original = bytetrack._cpp
    try:
        worst = 0.0
        for _ in range(200):
            n, m = int(rng.integers(1, 12)), int(rng.integers(1, 12))
            a = rng.random((n, 4)) * 100
            a[:, 2:] += a[:, :2] + 1
            b = rng.random((m, 4)) * 100
            b[:, 2:] += b[:, :2] + 1

            bytetrack._cpp = None
            ref = bytetrack._iou_batch(a, b)
            bytetrack._cpp = cpp
            got = bytetrack._iou_batch(a, b)
            worst = max(worst, float(np.abs(ref - got).max()))
        assert worst == 0.0, f"IoU differs by {worst}"
    finally:
        bytetrack._cpp = original


def test_degenerate_boxes_do_not_produce_nan():
    """A zero-area box gives IoU 0, not NaN.

    A NaN here would not raise; it would silently lose every comparison in
    the assignment and drop the track.
    """
    zero = np.array([[10.0, 10.0, 10.0, 10.0]])
    other = np.array([[0.0, 0.0, 20.0, 20.0]])
    got = cpp.iou_batch(zero, other)
    assert np.isfinite(got).all()
    assert got[0, 0] == 0.0
