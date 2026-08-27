"""
Phase 5 — the two reconstructed metrics in the daily summary.

Peak occupancy and the narrative are the only parts of this service that
compute rather than count, so they are the only parts that can be subtly
wrong. Both are tested as pure functions.

Run:  pytest tests/test_campus_summary.py -v
"""

import sys
import os
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.campus_summary_service import (  # noqa: E402
    OCCUPANCY_MERGE_GAP_SEC,
    STREAM_HEALTH_SUBTYPES,
    CampusSummaryService,
    _person_key,
)


BASE = datetime(2026, 8, 27, 3, 0, tzinfo=timezone.utc)


def at(minutes):
    return BASE + timedelta(minutes=minutes)


class Ev:
    """Stand-in for a DetectionEvent row."""

    def __init__(self, minutes, person_id=None, event_type="entry",
                 subtype=None, camera_id="cam1", metadata=None):
        self.created_at = at(minutes) if minutes is not None else None
        self.person_id = person_id
        self.event_type = event_type
        self.subtype = subtype
        self.camera_id = camera_id
        self.metadata_json = metadata or {}


peak = lambda events: CampusSummaryService._peak_occupancy(events)  # noqa: E731


# ── Identity keys ────────────────────────────────────────────────

def test_known_and_unknown_people_both_count():
    """
    Unknown people have no persons row — the FK guard moves their stable id
    into metadata. Counting only person_id would drop every stranger from the
    day's totals.
    """
    assert _person_key("p1", {}) == "known:p1"
    assert _person_key(None, {"unknown_person_id": "u1"}) == "unknown:u1"


def test_anonymous_track_handles_are_excluded():
    """
    `anon:` ids are ephemeral ByteTrack handles for faces too small to embed.
    They are not identities and would inflate a distinct-person count.
    """
    assert _person_key(None, {"anonymous": True}) is None
    assert _person_key(None, {}) is None
    assert _person_key(None, None) is None


# ── Peak occupancy ───────────────────────────────────────────────

def test_empty_day_has_zero_peak():
    assert peak([]) == {"peak": 0, "at": None}


def test_one_person_is_a_peak_of_one():
    result = peak([Ev(0, "p1"), Ev(5, "p1", event_type="exit")])
    assert result["peak"] == 1


def test_one_person_on_four_cameras_is_still_one_person():
    """
    The reason this is not a naive +1/-1 replay. entry/exit are PER CAMERA, so
    someone walking past four cameras emits four entries — a running total
    would report four people on campus.
    """
    result = peak([
        Ev(0, "p1", camera_id="cam1"),
        Ev(1, "p1", camera_id="cam2"),
        Ev(2, "p1", camera_id="cam3"),
        Ev(3, "p1", camera_id="cam4"),
    ])
    assert result["peak"] == 1


def test_concurrent_people_accumulate():
    result = peak([Ev(0, "p1"), Ev(1, "p2"), Ev(2, "p3")])
    assert result["peak"] == 3


def test_sequential_visits_do_not_accumulate():
    """
    Three people who each visit and leave before the next arrives never
    overlap, so the peak is one — not three.
    """
    gap = OCCUPANCY_MERGE_GAP_SEC // 60 + 5
    result = peak([
        Ev(0, "p1"), Ev(1, "p1", event_type="exit"),
        Ev(gap, "p2"), Ev(gap + 1, "p2", event_type="exit"),
        Ev(gap * 2, "p3"), Ev(gap * 2 + 1, "p3", event_type="exit"),
    ])
    assert result["peak"] == 1


def test_a_persons_two_visits_are_separate_intervals():
    """
    Someone who leaves for an hour and returns must not be counted as
    continuously present across the gap.
    """
    gap = OCCUPANCY_MERGE_GAP_SEC // 60 + 10
    result = peak([Ev(0, "p1"), Ev(gap, "p1"), Ev(gap + 2, "p2")])
    # p1's first visit ended long before p2 arrived; only the second overlaps.
    assert result["peak"] == 2


def test_brief_gaps_within_a_visit_merge():
    """A short spell out of view is one visit, not two."""
    within = max(1, (OCCUPANCY_MERGE_GAP_SEC // 60) - 2)
    result = peak([Ev(0, "p1"), Ev(within, "p1")])
    assert result["peak"] == 1


def test_peak_timestamp_is_reported():
    result = peak([Ev(0, "p1"), Ev(1, "p2")])
    assert result["at"] is not None
    assert datetime.fromisoformat(result["at"]) == at(1)


def test_unknown_people_count_toward_occupancy():
    result = peak([
        Ev(0, None, metadata={"unknown_person_id": "u1"}),
        Ev(1, None, metadata={"unknown_person_id": "u2"}),
    ])
    assert result["peak"] == 2


def test_anonymous_events_do_not_inflate_occupancy():
    result = peak([
        Ev(0, "p1"),
        Ev(1, None, metadata={"anonymous": True}),
        Ev(2, None, metadata={"anonymous": True}),
    ])
    assert result["peak"] == 1


def test_non_presence_events_are_ignored():
    """Vehicle and alert rows are not people."""
    result = peak([
        Ev(0, "p1"),
        Ev(1, None, event_type="vehicle_entry"),
        Ev(2, None, event_type="security_alert", subtype="gathering"),
    ])
    assert result["peak"] == 1


def test_events_without_timestamps_are_skipped():
    result = peak([Ev(None, "p1"), Ev(0, "p2")])
    assert result["peak"] == 1


def test_simultaneous_detections_both_count():
    """
    Two people detected at the same instant were both present at that
    instant, so the overlap is real. Ordering exits first would cancel it out
    and under-report a genuinely busy moment.
    """
    result = peak([
        Ev(0, "p1"),
        Ev(10, "p1", event_type="exit"),
        Ev(10, "p2"),
        Ev(20, "p2", event_type="exit"),
    ])
    assert result["peak"] == 2


def test_a_single_detection_still_counts_as_present():
    """
    A detection is an instant, not a span. Without a minimum presence window
    a lone sighting is a zero-length interval that contributes nothing, and
    three people each seen once would report a peak of zero.
    """
    assert peak([Ev(0, "p1")])["peak"] == 1
    assert peak([Ev(0, "p1"), Ev(1, "p2"), Ev(2, "p3")])["peak"] == 3


# ── Security metrics ─────────────────────────────────────────────

def test_stream_health_is_not_a_security_incident():
    """
    A flaky RTSP connection is camera plumbing. Counting it as an incident
    would make the security figure meaningless.
    """
    result = CampusSummaryService._security_metrics([
        Ev(0, event_type="security_alert", subtype="gathering"),
        Ev(1, event_type="security_alert", subtype="stream_offline"),
        Ev(2, event_type="security_alert", subtype="stream_recovered"),
    ])
    assert result["total"] == 1
    assert result["by_subtype"] == {"gathering": 1}


@pytest.mark.parametrize("subtype", STREAM_HEALTH_SUBTYPES)
def test_every_stream_subtype_is_excluded(subtype):
    result = CampusSummaryService._security_metrics([
        Ev(0, event_type="security_alert", subtype=subtype)
    ])
    assert result["total"] == 0


def test_real_alerts_are_grouped_by_subtype():
    result = CampusSummaryService._security_metrics([
        Ev(0, event_type="security_alert", subtype="hazard"),
        Ev(1, event_type="security_alert", subtype="hazard"),
        Ev(2, event_type="security_alert", subtype="loitering"),
    ])
    assert result["total"] == 3
    assert result["by_subtype"] == {"hazard": 2, "loitering": 1}


# ── Narrative ────────────────────────────────────────────────────

def base_metrics(**over):
    m = {
        "faculty_pct": 82.0, "faculty_total": 20, "faculty_present": 16,
        "security_incidents": 0, "incidents_by_subtype": {},
        "unknown_persons": 5, "unknown_faces_captured": 5,
        "camera_availability_pct": 100.0, "camera_downtime_sec": 0,
        "peak_occupancy": 40, "peak_occupancy_at": at(60).isoformat(),
        "peak_arrival_hour": 8, "peak_departure_hour": 17,
    }
    m.update(over)
    return m


def narrate(metrics, baseline=None, attendance=None):
    return CampusSummaryService._narrative(
        metrics, baseline or {}, attendance or {"departments": []}
    )


def test_narrative_is_capped_at_three_lines():
    lines = narrate(
        base_metrics(security_incidents=4, incidents_by_subtype={"hazard": 4},
                     camera_availability_pct=90.0, camera_downtime_sec=3600,
                     unknown_faces_captured=100, faculty_pct=40.0),
        {"avg_attendance_pct": 85.0, "avg_unknowns": 5, "days_counted": 7},
        {"departments": [{"name": "CSE", "total": 10, "pct": 30.0, "attending": 3}]},
    )
    assert len(lines) == 3


def test_narrative_leads_with_an_attendance_drop():
    lines = narrate(
        base_metrics(faculty_pct=60.0),
        {"avg_attendance_pct": 85.0, "avg_unknowns": None, "days_counted": 7},
    )
    assert "below" in lines[0]
    assert "85" in lines[0]


def test_no_baseline_means_no_comparison_claim():
    """Without history the narrative must not invent a trend."""
    lines = narrate(base_metrics(faculty_pct=20.0), {"days_counted": 0})
    assert not any("average" in line for line in lines)


def test_narrative_reports_incidents():
    lines = narrate(base_metrics(security_incidents=2,
                                 incidents_by_subtype={"loitering": 2}))
    assert any("2 security incidents" in line for line in lines)


def test_narrative_singularises_one_incident():
    lines = narrate(base_metrics(security_incidents=1,
                                 incidents_by_subtype={"hazard": 1}))
    assert any("1 security incident logged" in line for line in lines)


def test_narrative_flags_camera_downtime():
    lines = narrate(base_metrics(camera_availability_pct=92.5,
                                 camera_downtime_sec=1800))
    assert any("92.5%" in line for line in lines)


def test_narrative_is_silent_about_healthy_cameras():
    lines = narrate(base_metrics(camera_availability_pct=99.9))
    assert not any("availability" in line for line in lines)


def test_narrative_always_has_a_fallback_line():
    """A day with activity but nothing notable still gets a summary."""
    lines = narrate(base_metrics())
    assert len(lines) >= 1
    assert "Peak occupancy" in lines[-1]


def test_narrative_names_the_weakest_department():
    lines = narrate(
        base_metrics(),
        {},
        {"departments": [
            {"name": "CSE", "total": 10, "pct": 90.0, "attending": 9},
            {"name": "Civil", "total": 8, "pct": 37.5, "attending": 3},
        ]},
    )
    # 37.5% renders as "38%" — the assertion is on the department and its
    # counts, not on a rounding artefact.
    assert any("Civil" in line and "3 of 8" in line for line in lines)


def test_tiny_departments_are_not_singled_out():
    """A two-person department at 50% is one absence, not a trend."""
    lines = narrate(
        base_metrics(),
        {},
        {"departments": [{"name": "Civil", "total": 2, "pct": 50.0, "attending": 1}]},
    )
    assert not any("Civil" in line for line in lines)
