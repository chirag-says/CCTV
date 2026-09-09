"""
The two properties the process split has to hold.

1. A camera process dying does not stop the server. It is the whole reason
   cameras were moved out: a crash in one camera's decode or tracking must cost
   that camera, not the GPU loop and not the other nineteen.

2. Results route to the correct camera. Twenty cameras share one request queue
   and eight share a result queue, so a routing bug would not crash anything --
   it would quietly show camera 3 the people standing in front of camera 7,
   which is the worst kind of wrong this system can be.

Both run against the real server loop with a stub model, so the routing under
test is the shipping routing rather than a description of it.
"""

import queue
import threading
import time

import numpy as np
import pytest

from app.vision.inference_server import (
    KIND_FACES,
    KIND_PERSONS,
    InferenceServer,
    Request,
)


class _Slots:
    """Stands in for shared memory. Each slot holds a distinct frame, so a
    result can be traced back to the slot it was computed from."""

    def __init__(self):
        self.frames = {}

    def write(self, slot, frame):
        self.frames[slot] = frame
        return frame.shape

    def read(self, slot, h, w, c):
        return self.frames[slot]


def _server(result_queues, slots, **kw):
    s = InferenceServer(
        request_q=queue.Queue(), result_queues=result_queues, slots=slots,
        tick_ms=10.0, **kw
    )
    # Stub the models: the routing is what is under test, not YOLO.
    # Each frame reports the value stashed in its top-left pixel, so a
    # misrouted result is visible rather than merely possible.
    s._run_persons = lambda reqs: [
        [{"confidence": 1.0,
          "bbox": {"x1": int(slots.read(r.slot, r.h, r.w, r.c)[0, 0, 0]),
                   "y1": 0, "x2": 1, "y2": 1}}]
        for r in reqs
    ]
    s._run_faces = lambda reqs: [([], [], []) for _ in reqs]
    return s


def _req(camera_id, frame_id, slot, worker, kind=KIND_PERSONS):
    return Request(camera_id=camera_id, frame_id=frame_id, kind=kind,
                   slot=slot, h=4, w=4, c=3, worker=worker)


# ── 1. a camera process dying does not stop the server ────────────────────

def test_a_dead_camera_process_does_not_stop_the_server():
    """The dead worker's queue raises on every put. The server must keep
    serving the live worker, and must not exit its loop."""

    class DeadQueue:
        def put_nowait(self, _item):
            raise BrokenPipeError("camera process is gone")

    alive = queue.Queue()
    slots = _Slots()
    srv = _server({0: DeadQueue(), 1: alive}, slots)
    srv.start()
    try:
        for i in range(5):
            slots.write(i, np.full((4, 4, 3), i, np.uint8))
            srv.request_q.put(_req("dead-cam", i, i, worker=0))

        slots.write(90, np.full((4, 4, 3), 42, np.uint8))
        srv.request_q.put(_req("live-cam", 1, 90, worker=1))

        cam, fid, kind, payload = alive.get(timeout=5)
        assert cam == "live-cam"
        assert payload[0]["bbox"]["x1"] == 42
        assert srv._thread.is_alive(), "the server thread must survive"

        # And it keeps going afterwards, rather than limping to a stop.
        slots.write(91, np.full((4, 4, 3), 43, np.uint8))
        srv.request_q.put(_req("live-cam", 2, 91, worker=1))
        cam, fid, kind, payload = alive.get(timeout=5)
        assert payload[0]["bbox"]["x1"] == 43
    finally:
        srv.stop()


def test_a_model_that_raises_does_not_stop_the_server():
    """An inference failure must fail that batch, not the loop."""
    out = queue.Queue()
    slots = _Slots()
    srv = _server({0: out}, slots)
    calls = {"n": 0}

    def sometimes_explode(reqs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA hiccup")
        return [[{"confidence": 1.0, "bbox": {"x1": 7, "y1": 0, "x2": 1, "y2": 1}}]
                for _ in reqs]

    srv._run_persons = sometimes_explode
    srv.start()
    try:
        slots.write(0, np.zeros((4, 4, 3), np.uint8))
        srv.request_q.put(_req("cam", 1, 0, worker=0))
        cam, fid, kind, payload = out.get(timeout=5)
        # None is the wire contract for "no result"; the client turns it into
        # the caller's empty value, so the camera sees a frame with nothing in
        # it rather than an exception or a hang.
        assert payload is None, "a failed batch still replies, rather than hanging"

        from app.vision.inference_server import InferenceClient
        client = InferenceClient("cam", 0, slots, queue.Queue(), queue.Queue(),
                                 first_slot=0, n_slots=2, timeout=1.0)
        client.inbox.put((cam, 1, KIND_PERSONS, payload))
        assert client._await(1, KIND_PERSONS, []) == []

        slots.write(1, np.zeros((4, 4, 3), np.uint8))
        srv.request_q.put(_req("cam", 2, 1, worker=0))
        cam, fid, kind, payload = out.get(timeout=5)
        assert payload[0]["bbox"]["x1"] == 7, "the next batch must still work"
    finally:
        srv.stop()


# ── 2. results route to the correct camera ────────────────────────────────

def test_results_route_to_the_right_camera_and_worker():
    """Three cameras across two workers, submitted interleaved in one tick.
    Every result must reach its own camera, on its own worker, carrying the
    payload computed from its own frame."""
    w0, w1 = queue.Queue(), queue.Queue()
    slots = _Slots()
    srv = _server({0: w0, 1: w1}, slots)
    srv.start()
    try:
        plan = [
            ("cam-a", 11, 0, 0, 100),
            ("cam-b", 22, 1, 0, 101),
            ("cam-c", 33, 2, 1, 102),
        ]
        for cam, fid, slot, worker, value in plan:
            slots.write(slot, np.full((4, 4, 3), value, np.uint8))
            srv.request_q.put(_req(cam, fid, slot, worker))

        got = {}
        for _ in range(2):
            cam, fid, kind, payload = w0.get(timeout=5)
            got[cam] = (fid, payload[0]["bbox"]["x1"])
        cam, fid, kind, payload = w1.get(timeout=5)
        got[cam] = (fid, payload[0]["bbox"]["x1"])

        assert got["cam-a"] == (11, 100)
        assert got["cam-b"] == (22, 101)
        assert got["cam-c"] == (33, 102)
        assert w0.empty() and w1.empty(), "no camera received a stray result"
    finally:
        srv.stop()


def test_a_camera_ignores_results_that_are_not_its_frame():
    """The client waits for ITS frame id. A stale result for an earlier frame,
    or one for another kind, must not be mistaken for the answer."""
    from app.vision.inference_server import InferenceClient

    inbox = queue.Queue()
    client = InferenceClient(
        camera_id="cam-a", worker=0, slots=_Slots(), request_q=queue.Queue(),
        inbox=inbox, first_slot=0, n_slots=2, timeout=2.0,
    )
    client._frame_id = 6                       # so the next submit is frame 7

    inbox.put(("cam-a", 5, KIND_PERSONS, [{"stale": True}]))     # older frame
    inbox.put(("cam-a", 7, KIND_FACES, ([], [], [])))            # wrong kind
    inbox.put(("cam-a", 7, KIND_PERSONS, [{"correct": True}]))   # ours

    got = client._await(7, KIND_PERSONS, [])
    assert got == [{"correct": True}]


def test_a_lost_result_times_out_as_an_empty_detection():
    """A worker whose result never arrives must carry on with nothing found,
    not block its camera thread for ever."""
    from app.vision.inference_server import InferenceClient

    client = InferenceClient(
        camera_id="cam-a", worker=0, slots=_Slots(), request_q=queue.Queue(),
        inbox=queue.Queue(), first_slot=0, n_slots=2, timeout=0.2,
    )
    t0 = time.perf_counter()
    assert client._await(1, KIND_PERSONS, []) == []
    assert time.perf_counter() - t0 < 2.0


def test_slots_are_reused_within_a_camera_and_never_shared():
    """A camera cycles its own band. Two cameras must never be handed the same
    slot, or one would overwrite a frame the server is still reading."""
    from app.vision.inference_server import InferenceClient

    slots = _Slots()
    a = InferenceClient("cam-a", 0, slots, queue.Queue(), queue.Queue(),
                        first_slot=0, n_slots=3)
    b = InferenceClient("cam-b", 0, slots, queue.Queue(), queue.Queue(),
                        first_slot=3, n_slots=3)

    frame = np.zeros((4, 4, 3), np.uint8)
    a_slots = {a._submit(frame, KIND_PERSONS) and
               a.request_q.get().slot for _ in range(6)}
    b_slots = {b._submit(frame, KIND_PERSONS) and
               b.request_q.get().slot for _ in range(6)}

    assert a_slots == {0, 1, 2}
    assert b_slots == {3, 4, 5}
    assert not (a_slots & b_slots)
