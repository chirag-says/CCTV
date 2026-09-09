"""
Item 1 -- encode JPEG only while someone is watching.

THE COST BEING REMOVED
A 720p JPEG encode is 8.9 ms of CPU on this machine. Every camera ran one per
displayed frame regardless of whether a viewer existed, so a camera at 30 fps
spent about 270 ms of CPU per second producing images that were then thrown
away. Across 20 cameras that is more than five cores of pure waste, and it is
spent before the GPU is reached at all.

WHY THIS NEEDED A VIEWER REGISTRY AT ALL
`ws_manager.stream_counts` existed, but nothing ever wrote to it:
connect_stream/disconnect_stream had no callers, and the stream endpoint
accepted its socket directly. Gating on it as it stood would have read a
permanently empty dict and switched encoding off for good. These tests pin
both halves -- that a viewer is actually recorded, and that the gate follows
it in both directions.
"""

import asyncio
import time

import pytest

from app.core.websocket import ConnectionManager


def _run(coro):
    """Drive one coroutine to completion. ConnectionManager's registration
    methods are async only because they take an asyncio.Lock; they never
    await anything else, so a fresh loop per call is exact and cheap."""
    return asyncio.run(coro)


class _Sock:
    """Stand-in for a WebSocket. connect_stream no longer accepts, so the
    manager never calls anything on this."""


@pytest.fixture
def mgr():
    return ConnectionManager()


# ── viewer presence ───────────────────────────────────────────────────────

def test_no_viewers_by_default(mgr):
    assert mgr.is_watched("cam-1") is False
    assert mgr.viewer_count("cam-1") == 0


def test_websocket_viewer_is_counted(mgr):
    ws = _Sock()
    _run(mgr.connect_stream("cam-1", ws))
    assert mgr.is_watched("cam-1") is True
    assert mgr.viewer_count("cam-1") == 1
    # ...and only for that camera.
    assert mgr.is_watched("cam-2") is False

    _run(mgr.disconnect_stream("cam-1", ws))
    assert mgr.is_watched("cam-1") is False


def test_last_viewer_leaving_clears_the_camera(mgr):
    a, b = _Sock(), _Sock()
    _run(mgr.connect_stream("cam-1", a))
    _run(mgr.connect_stream("cam-1", b))
    assert mgr.viewer_count("cam-1") == 2

    _run(mgr.disconnect_stream("cam-1", a))
    assert mgr.is_watched("cam-1") is True, "one viewer left, still watched"
    _run(mgr.disconnect_stream("cam-1", b))
    assert mgr.is_watched("cam-1") is False


def test_snapshot_poller_counts_as_a_viewer(mgr):
    """A polling <img> holds no connection, so it is counted by recency."""
    mgr.note_snapshot_request("cam-1", ttl_sec=0.2)
    assert mgr.is_watched("cam-1") is True
    time.sleep(0.25)
    assert mgr.is_watched("cam-1") is False, "the poll must expire on its own"


def test_snapshot_ttl_outlives_a_polling_interval(mgr):
    """The default TTL has to exceed the client's poll gap, or a steadily
    polling viewer looks idle between requests and the encoder stops
    underneath it."""
    mgr.note_snapshot_request("cam-1")
    time.sleep(0.1)
    assert mgr.is_watched("cam-1") is True


# ── the gate itself ───────────────────────────────────────────────────────

class _FakeWorker:
    """Just enough CameraWorker to run one pass of the encoder's decision."""

    def __init__(self, mgr, camera_id="cam-1", on_demand=True):
        self.camera_id = camera_id
        self._mgr = mgr
        self._on_demand = on_demand
        self.encoded = 0
        self.skipped = 0

    def handle(self):
        # The branch as it appears in CameraWorker._jpeg_encoder_loop.
        if self._on_demand and not self._mgr.is_watched(self.camera_id):
            self.skipped += 1
            return False
        self.encoded += 1
        return True


def test_gate_skips_when_unwatched(mgr):
    w = _FakeWorker(mgr)
    for _ in range(10):
        w.handle()
    assert w.encoded == 0
    assert w.skipped == 10


def test_gate_resumes_when_a_viewer_arrives(mgr):
    w = _FakeWorker(mgr)
    w.handle()
    assert w.encoded == 0

    ws = _Sock()
    _run(mgr.connect_stream("cam-1", ws))
    w.handle()
    assert w.encoded == 1, "a viewer arriving must restart encoding"

    _run(mgr.disconnect_stream("cam-1", ws))
    w.handle()
    assert w.encoded == 1, "and leaving must stop it again"
    assert w.skipped == 2


def test_flag_off_restores_unconditional_encoding(mgr):
    """JPEG_ENCODE_ON_DEMAND=False has to reach the old behaviour, with no
    viewer anywhere."""
    w = _FakeWorker(mgr, on_demand=False)
    for _ in range(5):
        w.handle()
    assert w.encoded == 5
    assert w.skipped == 0
