"""
Body-appearance crops for cross-camera handoff.

The network itself is not under test here (it needs a GPU and its weights);
what is pinned is everything around it that decides whether a crop is worth
sending, how often, and that a crop becomes exactly one unit vector for the
right track. A bad crop is worse than none, so the gates matter as much as the
plumbing.
"""

import threading
import time

import numpy as np
import pytest

from app.vision.appearance import (
    CROP_H,
    CROP_W,
    AppearanceService,
    CropSchedule,
    encode_body_crop,
    usable_body_box,
)

W, H = 1920, 1080
PERSON = (800, 300, 900, 600)        # x1, y1, x2, y2: 100 wide, 300 tall


# ── Which crops are worth sending ────────────────────────────────────────────

def test_a_whole_upright_person_is_usable():
    assert usable_body_box(PERSON, W, H)


@pytest.mark.parametrize("box, why", [
    ((800, 300, 830, 380), "too small to read"),
    ((0, 300, 100, 600), "cut off by the left edge"),
    ((800, 780, 900, 1080), "cut off by the bottom edge"),
    ((800, 300, 1100, 600), "wider than tall: not a standing person"),
])
def test_unreadable_boxes_are_refused(box, why):
    assert not usable_body_box(box, W, H), why


def test_a_person_mostly_covered_by_another_is_refused():
    assert not usable_body_box(PERSON, W, H, others=[(820, 300, 920, 600)])
    assert usable_body_box(PERSON, W, H, others=[(890, 300, 990, 600)])   # just touching


def test_a_crop_travels_at_the_network_input_size():
    import cv2

    frame = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    jpeg = encode_body_crop(frame, PERSON)
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (CROP_H, CROP_W, 3)
    assert encode_body_crop(frame, (10, 10, 12, 14)) is None


# ── How often ────────────────────────────────────────────────────────────────

def test_a_new_track_is_looked_at_quickly_and_an_old_one_rarely():
    schedule = CropSchedule(early_interval=1.0, early_count=3, late_interval=5.0)
    sent_at = []
    for tick in range(200):
        now = tick / 10
        if schedule.due(7, now):
            schedule.sent(7, now)
            sent_at.append(now)
    assert sent_at[:3] == [0.0, 1.0, 2.0]
    assert np.diff(sent_at[3:]).min() >= 4.9


def test_the_schedule_forgets_tracks_that_are_gone():
    schedule = CropSchedule()
    for track_id in range(200):
        schedule.sent(track_id, 0.0)
    schedule.forget_except([1, 2])
    assert not schedule.due(1, 0.1)       # still remembered
    assert schedule.due(150, 0.1)         # forgotten, so due again


# ── Crops to vectors ─────────────────────────────────────────────────────────

def _service(received, embed=None):
    done = threading.Event()

    def on_embedding(camera_id, track_id, vector, timestamp):
        received.append((camera_id, track_id, vector, timestamp))
        done.set()

    service = AppearanceService(
        on_embedding=on_embedding,
        embed=embed or (lambda crops: np.array([[3.0, 4.0]] * len(crops))),
    )
    service.start()
    return service, done


def _jpeg():
    frame = np.full((H, W, 3), 128, dtype=np.uint8)
    return encode_body_crop(frame, PERSON)


def test_a_crop_becomes_one_unit_vector_for_its_track():
    received = []
    service, done = _service(received)
    assert service.submit("cam-a", 12, 100.0, _jpeg())
    assert done.wait(5.0)
    camera_id, track_id, vector, timestamp = received[0]
    assert (camera_id, track_id, timestamp) == ("cam-a", 12, 100.0)
    assert vector == pytest.approx([0.6, 0.8])
    service.stop()


def test_a_full_queue_drops_crops_rather_than_blocking_the_event_pump():
    gate = threading.Event()

    def slow(crops):
        gate.wait(5.0)
        return np.ones((len(crops), 2))

    service = AppearanceService(on_embedding=lambda *a: None, embed=slow, max_pending=4)
    service.start()
    started = time.perf_counter()
    results = [service.submit("cam", i, 0.0, _jpeg()) for i in range(40)]
    assert time.perf_counter() - started < 2.0
    assert not all(results) and service.stats["dropped"] > 0
    gate.set()
    service.stop()


def test_a_broken_model_switches_looks_off_and_nothing_else():
    def broken(crops):
        raise RuntimeError("weights missing")

    service = AppearanceService(on_embedding=lambda *a: None, embed=broken)
    service.start()
    service.submit("cam", 1, 0.0, _jpeg())
    deadline = time.time() + 5.0
    while service.available and time.time() < deadline:
        time.sleep(0.02)
    assert not service.available
    assert service.submit("cam", 2, 0.0, _jpeg()) is False     # refused quietly, never raises


def test_garbage_bytes_are_skipped():
    received = []
    service, done = _service(received)
    service.submit("cam", 1, 0.0, b"not a jpeg")
    service.submit("cam", 2, 0.0, _jpeg())
    assert done.wait(5.0)
    assert [r[1] for r in received] == [2]
    service.stop()
