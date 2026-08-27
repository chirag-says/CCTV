"""
Phase 3 — the faculty presence state machine.

Tests PresenceService._state_for() directly. This is the logic that decides
whether the dashboard claims somebody is on campus, and it cannot be
exercised end-to-end without live cameras, so it is tested as the pure
function it is.

Covers §9 tests 13-17 of COLLEGE_FEATURES_SPEC.md.

Run:  pytest tests/test_presence_state.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings  # noqa: E402
from app.services.presence_service import (  # noqa: E402
    COUNTED_PRESENT,
    STATE_DEPARTED,
    STATE_LIVE,
    STATE_ON_CAMPUS,
    STATE_SEEN_EARLIER,
    PresenceService,
)


class FakeCamera:
    """Minimal stand-in — _state_for only reads .config."""

    def __init__(self, is_gate=False):
        self.config = {"is_gate": is_gate} if is_gate else {}


GATE = FakeCamera(is_gate=True)
INDOOR = FakeCamera(is_gate=False)


def state(is_live=False, minutes_since=None, camera=INDOOR, past_day_end=False):
    return PresenceService._state_for(
        is_live=is_live,
        minutes_since=minutes_since,
        last_camera=camera,
        past_day_end=past_day_end,
    )


@pytest.fixture(autouse=True)
def default_thresholds(monkeypatch):
    monkeypatch.setattr(settings, "PRESENCE_CONFIDENT_MIN", 20)
    monkeypatch.setattr(settings, "GATE_DEPART_MIN", 10)


# ── The four states ──────────────────────────────────────────────

def test_live_when_a_camera_can_see_them():
    """§9-13."""
    assert state(is_live=True, minutes_since=0) == STATE_LIVE


def test_on_campus_within_the_confidence_window():
    """§9-14. Four minutes ago is a claim worth making."""
    assert state(minutes_since=4) == STATE_ON_CAMPUS


def test_seen_earlier_beyond_the_confidence_window():
    """
    §9-15. Two hours ago is NOT a confident claim. It is still shown -- they
    probably are on campus -- but it must not inflate the headline count.
    """
    assert state(minutes_since=134) == STATE_SEEN_EARLIER
    assert STATE_SEEN_EARLIER not in COUNTED_PRESENT


def test_departed_after_a_gate_sighting():
    """
    §9-16. A gate camera is the only place the system can honestly infer a
    departure: the one location where being seen then unseen means leaving
    rather than walking into an unmonitored room.
    """
    assert state(minutes_since=15, camera=GATE) == STATE_DEPARTED


def test_departed_after_the_campus_day_ends():
    """§9-17."""
    assert state(minutes_since=45, past_day_end=True) == STATE_DEPARTED


# ── Precedence ───────────────────────────────────────────────────

def test_live_beats_the_gate_rule():
    """
    Standing at the gate talking for 15 minutes is not a departure while a
    camera is actively watching them do it.
    """
    assert state(is_live=True, minutes_since=15, camera=GATE) == STATE_LIVE


def test_live_beats_the_day_end_rule():
    """If a camera can see them at 21:30, they are on campus."""
    assert state(is_live=True, minutes_since=0, past_day_end=True) == STATE_LIVE


def test_gate_rule_beats_the_confidence_window():
    """
    Departure is a positive signal, not merely an absence of one, so it wins
    over a recent sighting -- being seen leaving IS the recent sighting.
    """
    assert state(minutes_since=12, camera=GATE) == STATE_DEPARTED


def test_recent_gate_sighting_is_not_yet_a_departure():
    """Below GATE_DEPART_MIN they may simply be walking in."""
    assert state(minutes_since=3, camera=GATE) == STATE_ON_CAMPUS


# ── Boundaries ───────────────────────────────────────────────────

def test_confidence_window_boundary_is_exclusive():
    assert state(minutes_since=19.9) == STATE_ON_CAMPUS
    assert state(minutes_since=20) == STATE_SEEN_EARLIER


def test_gate_boundary_is_inclusive():
    assert state(minutes_since=9.9, camera=GATE) == STATE_ON_CAMPUS
    assert state(minutes_since=10, camera=GATE) == STATE_DEPARTED


def test_no_detection_at_all_is_departed():
    """Defensive: a row with no last_seen should never claim presence."""
    assert state(minutes_since=None) == STATE_DEPARTED


def test_missing_camera_does_not_crash_the_gate_rule():
    """A deleted or unregistered camera must degrade, not raise."""
    assert state(minutes_since=5, camera=None) == STATE_ON_CAMPUS
    assert state(minutes_since=200, camera=None) == STATE_SEEN_EARLIER


# ── Headline arithmetic ──────────────────────────────────────────

def test_only_confident_states_feed_the_headline():
    """
    The whole point of the split headline: "18 present now" must not include
    somebody last detected eight hours ago.
    """
    assert set(COUNTED_PRESENT) == {STATE_LIVE, STATE_ON_CAMPUS}
    assert STATE_SEEN_EARLIER not in COUNTED_PRESENT
    assert STATE_DEPARTED not in COUNTED_PRESENT


def test_confidence_window_is_configurable(monkeypatch):
    assert state(minutes_since=45) == STATE_SEEN_EARLIER
    monkeypatch.setattr(settings, "PRESENCE_CONFIDENT_MIN", 90)
    assert state(minutes_since=45) == STATE_ON_CAMPUS
