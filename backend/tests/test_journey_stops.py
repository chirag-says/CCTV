"""
Phase 4 — collapsing raw detections into a readable journey.

Tests JourneyService._collapse_to_stops() directly. The collapsing is the
whole feature: raw entry/exit rows are unreadable (a person standing at the
gate produces entry → exit → entry → exit indefinitely), and the failure
modes are subtle enough that they only show up on a real timeline.

Covers §9 tests 18-19 of COLLEGE_FEATURES_SPEC.md.

Run:  pytest tests/test_journey_stops.py -v
"""

import sys
import os
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings  # noqa: E402
from app.services.journey_service import JourneyService, _pick_display_camera  # noqa: E402


# ── Fakes ────────────────────────────────────────────────────────

class FakeCamera:
    def __init__(self, cid, name, stream_url=None, location="", config=None):
        self.id = cid
        self.name = name
        self.stream_url = stream_url
        self.location = location
        self.config = config or {}


class FakeEvent:
    def __init__(self, camera_id, created_at, event_type="entry"):
        self.camera_id = camera_id
        self.created_at = created_at
        self.event_type = event_type


BASE = datetime(2026, 8, 27, 3, 0, tzinfo=timezone.utc)  # 08:30 IST


def at(minutes, seconds=0):
    return BASE + timedelta(minutes=minutes, seconds=seconds)


# Two genuinely different lenses.
GATE = FakeCamera("cam-gate", "MainGate", "rtsp://nvr/live?channel=1&subtype=0",
                  "Main Entrance", {"zone": "Main Gate", "is_gate": True})
BLOCK = FakeCamera("cam-block", "CSEBlock", "rtsp://nvr/live?channel=2&subtype=0",
                   "CSE Block", {"zone": "CSE Block"})

# A SECOND REGISTRATION OF THE GATE — same channel, different stream quality.
# This is the real MainGate/MainStream pair in this deployment.
GATE_SUB = FakeCamera("cam-gate-sub", "MainStream", "rtsp://nvr/live?channel=1&subtype=1")

CAMERAS = {c.id: c for c in (GATE, BLOCK, GATE_SUB)}


def collapse(events):
    return JourneyService._collapse_to_stops(events, CAMERAS)


@pytest.fixture(autouse=True)
def default_gap(monkeypatch):
    monkeypatch.setattr(settings, "JOURNEY_STOP_GAP_SEC", 300)


# ── Basic collapsing ─────────────────────────────────────────────

def test_empty_day_has_no_stops():
    assert collapse([]) == []


def test_single_detection_is_one_stop_with_no_dwell():
    stops = collapse([FakeEvent("cam-gate", at(0))])
    assert len(stops) == 1
    assert stops[0]["zone"] == "Main Gate"
    assert stops[0]["dwell_sec"] == 0
    assert stops[0]["event_count"] == 1


def test_repeated_detections_at_one_place_collapse():
    """
    §9-10. The core case: entry → exit → entry → exit at one camera is ONE
    visit, not four. Rendered raw it would read as four separate arrivals.
    """
    stops = collapse([
        FakeEvent("cam-gate", at(0), "entry"),
        FakeEvent("cam-gate", at(0, 30), "exit"),
        FakeEvent("cam-gate", at(2), "entry"),
        FakeEvent("cam-gate", at(3), "exit"),
    ])
    assert len(stops) == 1
    assert stops[0]["event_count"] == 4
    assert stops[0]["dwell_sec"] == 180  # 08:30 → 08:33


def test_moving_between_cameras_creates_separate_stops():
    """§9-7. The journey the feature exists to show."""
    stops = collapse([
        FakeEvent("cam-gate", at(0)),
        FakeEvent("cam-block", at(5)),
        FakeEvent("cam-gate", at(60)),
    ])
    assert [s["zone"] for s in stops] == ["Main Gate", "CSE Block", "Main Gate"]


def test_revisit_is_a_separate_stop_not_a_merge():
    """Returning to the gate at the end of the day is a new stop."""
    stops = collapse([
        FakeEvent("cam-gate", at(0)),
        FakeEvent("cam-block", at(30)),
        FakeEvent("cam-gate", at(500)),
    ])
    assert len(stops) == 3
    assert stops[0]["zone"] == stops[2]["zone"] == "Main Gate"


# ── The gap rule ─────────────────────────────────────────────────

def test_gap_below_threshold_merges():
    stops = collapse([
        FakeEvent("cam-block", at(0)),
        FakeEvent("cam-block", at(4)),  # 240s < 300s
    ])
    assert len(stops) == 1


def test_gap_above_threshold_splits():
    """§9-19. Ten minutes away and back is two visits, not one long one."""
    stops = collapse([
        FakeEvent("cam-block", at(0)),
        FakeEvent("cam-block", at(10)),  # 600s > 300s
    ])
    assert len(stops) == 2
    assert stops[0]["dwell_sec"] == 0
    assert stops[1]["dwell_sec"] == 0


def test_gap_boundary_is_inclusive():
    """Exactly at the threshold still counts as the same visit."""
    assert len(collapse([FakeEvent("cam-block", at(0)), FakeEvent("cam-block", at(5))])) == 1
    assert len(collapse([FakeEvent("cam-block", at(0)), FakeEvent("cam-block", at(5, 1))])) == 2


def test_gap_is_measured_from_the_last_event_not_the_first():
    """
    A long continuous presence must not split. Events every 4 minutes for 20
    minutes is one stop, because each gap is under the limit even though the
    total span is far over it.
    """
    stops = collapse([FakeEvent("cam-block", at(m)) for m in (0, 4, 8, 12, 16, 20)])
    assert len(stops) == 1
    assert stops[0]["dwell_sec"] == 1200


def test_gap_is_configurable(monkeypatch):
    events = [FakeEvent("cam-block", at(0)), FakeEvent("cam-block", at(10))]
    assert len(collapse(events)) == 2
    monkeypatch.setattr(settings, "JOURNEY_STOP_GAP_SEC", 900)
    assert len(collapse(events)) == 1


# ── Duplicate camera registrations ───────────────────────────────

def test_one_lens_registered_twice_does_not_ping_pong():
    """
    The most important case in this file. MainGate and MainStream are the
    SAME physical camera (channel=1, differing only by stream quality). Keyed
    on camera_id, a person standing still would produce a fake
    Gate → Stream → Gate → Stream journey across four "places".
    """
    stops = collapse([
        FakeEvent("cam-gate", at(0)),
        FakeEvent("cam-gate-sub", at(1)),
        FakeEvent("cam-gate", at(2)),
        FakeEvent("cam-gate-sub", at(3)),
    ])
    assert len(stops) == 1, "duplicate registrations of one lens must be one stop"
    assert stops[0]["event_count"] == 4
    assert set(stops[0]["camera_ids"]) == {"cam-gate", "cam-gate-sub"}


def test_merged_stop_is_named_after_the_labelled_registration():
    """
    Given two names for one place, use the one an operator actually labelled
    with a zone -- not whichever happened to fire first.
    """
    stops = collapse([
        FakeEvent("cam-gate-sub", at(0)),  # unlabelled, fires first
        FakeEvent("cam-gate", at(1)),      # has zone "Main Gate"
    ])
    assert stops[0]["zone"] == "Main Gate"
    assert stops[0]["is_gate"] is True


def test_distinct_channels_stay_distinct():
    """Dedup must not over-merge two real lenses on one NVR."""
    stops = collapse([FakeEvent("cam-gate", at(0)), FakeEvent("cam-block", at(1))])
    assert len(stops) == 2


def test_display_camera_preference_order():
    zoned = FakeCamera("a", "A", config={"zone": "Z"})
    located = FakeCamera("b", "B", location="L")
    plain = FakeCamera("c", "C")
    assert _pick_display_camera([plain, located, zoned]) is zoned
    assert _pick_display_camera([plain, located]) is located
    assert _pick_display_camera([plain]) is plain
    assert _pick_display_camera([]) is None


# ── Transitions ──────────────────────────────────────────────────

def test_transitions_report_unobserved_time():
    """
    Time between cameras is reported as a gap, never folded into the previous
    stop's dwell — that would claim the person was somewhere they were not
    observed to be.
    """
    stops = collapse([FakeEvent("cam-gate", at(0)), FakeEvent("cam-block", at(7))])
    trans = JourneyService._transitions(stops)
    assert len(trans) == 1
    assert trans[0]["from_zone"] == "Main Gate"
    assert trans[0]["to_zone"] == "CSE Block"
    assert trans[0]["gap_sec"] == 420


def test_no_transitions_for_a_single_stop():
    assert JourneyService._transitions(collapse([FakeEvent("cam-gate", at(0))])) == []


def test_returning_to_the_same_place_is_flagged_not_a_move():
    """
    On a single-camera site EVERY transition is a return to the same place.
    Rendered as "Main Gate → Main Gate" a correct journey looks broken, so
    the UI needs to know this was a spell out of view, not travel.
    """
    stops = collapse([FakeEvent("cam-gate", at(0)), FakeEvent("cam-gate", at(20))])
    trans = JourneyService._transitions(stops)
    assert len(trans) == 1
    assert trans[0]["same_place"] is True
    assert trans[0]["gap_sec"] == 1200


def test_moving_between_places_is_not_flagged_same_place():
    stops = collapse([FakeEvent("cam-gate", at(0)), FakeEvent("cam-block", at(20))])
    assert JourneyService._transitions(stops)[0]["same_place"] is False


def test_same_place_uses_physical_camera_not_name():
    """Two registrations of one lens are the same place even under two names."""
    stops = collapse([
        FakeEvent("cam-gate", at(0)),
        FakeEvent("cam-gate-sub", at(20)),  # different camera_id, same lens
    ])
    assert len(stops) == 2  # split by the gap rule
    assert JourneyService._transitions(stops)[0]["same_place"] is True


def test_transition_count_is_one_less_than_stops():
    stops = collapse([
        FakeEvent("cam-gate", at(0)),
        FakeEvent("cam-block", at(20)),
        FakeEvent("cam-gate", at(40)),
    ])
    assert len(JourneyService._transitions(stops)) == len(stops) - 1


# ── Robustness ───────────────────────────────────────────────────

def test_events_with_no_timestamp_are_skipped():
    """A NULL created_at must not crash the page."""
    stops = collapse([
        FakeEvent("cam-gate", None),
        FakeEvent("cam-gate", at(0)),
    ])
    assert len(stops) == 1
    assert stops[0]["event_count"] == 1


def test_unregistered_camera_still_produces_a_stop():
    """A deleted camera should degrade to its id, not vanish or raise."""
    stops = collapse([FakeEvent("cam-ghost", at(0))])
    assert len(stops) == 1
    assert stops[0]["camera_name"] == "cam-ghost"


def test_stop_indexes_are_sequential():
    stops = collapse([
        FakeEvent("cam-gate", at(0)),
        FakeEvent("cam-block", at(20)),
        FakeEvent("cam-gate", at(40)),
    ])
    assert [s["index"] for s in stops] == [0, 1, 2]
