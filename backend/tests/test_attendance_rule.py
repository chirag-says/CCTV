"""
Phase 2 — the attendance decision rule.

Tests _decide() directly against synthetic evidence. No database, no camera:
the rule is a pure function of (sightings, span, distinct sources, first
detection time), and that is exactly the part that must not be wrong.

Covers §9 tests 1-9 of COLLEGE_FEATURES_SPEC.md.

Run:  pytest tests/test_attendance_rule.py -v
"""

import sys
import os
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings  # noqa: E402
from app.services.attendance_service import (  # noqa: E402
    STATUS_ABSENT,
    STATUS_LATE,
    STATUS_PRESENT,
    _confidence_filter,
    _decide,
    _normalise_department,
    normalise_stream_source,
    UNASSIGNED_DEPARTMENT,
)


# ── Fixtures ─────────────────────────────────────────────────────

def ist(hh, mm=0, day=27):
    """A campus-local (IST) wall-clock time, returned as the stored UTC value."""
    return datetime(2026, 8, day, hh, mm, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def evidence(first, last=None, sightings=1, sources=1):
    """Build the evidence dict _decide() consumes."""
    return {
        "first_seen": first,
        "last_seen": last if last is not None else first,
        "sightings": sightings,
        "camera_ids": {f"cam{i}" for i in range(sources)},
        "sources": {f"src{i}" for i in range(sources)},
        "last_camera_id": "cam0",
    }


@pytest.fixture(autouse=True)
def default_thresholds(monkeypatch):
    """Pin the rule's thresholds so a config change cannot silently rewrite
    what these tests assert."""
    monkeypatch.setattr(settings, "ATTENDANCE_MIN_SIGHTINGS", 2)
    monkeypatch.setattr(settings, "ATTENDANCE_MIN_SPAN_MIN", 5)
    monkeypatch.setattr(settings, "ATTENDANCE_MIN_CAMERAS", 2)
    monkeypatch.setattr(settings, "ATTENDANCE_LATE_AFTER", "09:15")


# ── The two-confirmation rule ────────────────────────────────────

def test_normal_working_day_is_present():
    """§9-1. Several sightings across a full day."""
    status, confirmed = _decide(
        evidence(ist(8, 42), ist(17, 21), sightings=14, sources=2)
    )
    assert (status, confirmed) == (STATUS_PRESENT, True)


def test_late_arrival():
    """§9-2. Confirmed present, but first detected after the cutoff."""
    status, confirmed = _decide(
        evidence(ist(9, 47), ist(17, 2), sightings=11, sources=2)
    )
    assert (status, confirmed) == (STATUS_LATE, True)
    # Late still means they attended.
    assert confirmed is True


def test_no_events_is_absent():
    """§9-3."""
    assert _decide(None) == (STATUS_ABSENT, False)
    assert _decide(evidence(None)) == (STATUS_ABSENT, False)


def test_double_false_positive_is_absent():
    """
    §9-4 and the whole reason the rule uses AND rather than OR.

    Two recognitions 60 seconds apart on ONE camera is the signature of a
    recogniser false accept, not of a person attending work. Sightings alone
    must never be sufficient.
    """
    status, confirmed = _decide(
        evidence(ist(8, 42), ist(8, 43), sightings=2, sources=1)
    )
    assert (status, confirmed) == (STATUS_ABSENT, False)


def test_two_cameras_short_span_is_present():
    """§9-5. Corroboration by a second physical source substitutes for time."""
    status, _ = _decide(
        evidence(ist(8, 42), ist(8, 43, ), sightings=2, sources=2)
    )
    assert status == STATUS_PRESENT


def test_two_sightings_long_span_one_camera_is_present():
    """
    §9-6. The single-gate case: seen on arrival and again on departure, one
    camera, eight hours apart. Must be present -- this is the common shape in
    a deployment with one gate camera.
    """
    status, _ = _decide(
        evidence(ist(8, 42), ist(17, 21), sightings=2, sources=1)
    )
    assert status == STATUS_PRESENT


def test_single_sighting_is_absent():
    """
    §9-7. One detection is never enough on its own -- but the caller surfaces
    `sightings` so the shortfall is visible in the UI instead of silent.
    """
    status, confirmed = _decide(
        evidence(ist(8, 39), sightings=1, sources=1)
    )
    assert (status, confirmed) == (STATUS_ABSENT, False)


def test_single_sighting_on_two_cameras_is_still_absent():
    """
    The sightings gate applies before the OR. One appearance that happened to
    land on two registered cameras must not pass.
    """
    status, _ = _decide(evidence(ist(8, 39), sightings=1, sources=2))
    assert status == STATUS_ABSENT


# ── Boundaries ───────────────────────────────────────────────────

def test_span_boundary_is_inclusive():
    """Exactly the minimum span qualifies; one minute under does not."""
    assert _decide(evidence(ist(8, 0), ist(8, 5), sightings=2, sources=1))[0] == STATUS_PRESENT
    assert _decide(evidence(ist(8, 0), ist(8, 4), sightings=2, sources=1))[0] == STATUS_ABSENT


def test_camera_boundary_is_inclusive():
    assert _decide(evidence(ist(8, 0), ist(8, 1), sightings=2, sources=2))[0] == STATUS_PRESENT
    assert _decide(evidence(ist(8, 0), ist(8, 1), sightings=2, sources=1))[0] == STATUS_ABSENT


def test_sighting_boundary_is_inclusive():
    assert _decide(evidence(ist(8, 0), ist(9, 0), sightings=2, sources=1))[0] == STATUS_PRESENT
    assert _decide(evidence(ist(8, 0), ist(9, 0), sightings=1, sources=1))[0] == STATUS_ABSENT


def test_late_boundary_evaluated_in_campus_local_time():
    """09:15 IST exactly is on time; 09:16 is late. The stored value is UTC."""
    assert _decide(evidence(ist(9, 15), ist(17, 0), sightings=5, sources=2))[0] == STATUS_PRESENT
    assert _decide(evidence(ist(9, 16), ist(17, 0), sightings=5, sources=2))[0] == STATUS_LATE


# ── Configurability ──────────────────────────────────────────────

def test_single_camera_deployment_can_relax_the_rule(monkeypatch):
    """
    On a one-camera deployment the MIN_CAMERAS branch can never fire, so a
    teacher seen only on arrival would be marked absent. The thresholds are
    config precisely so that can be tuned without a code change.
    """
    one_sighting = evidence(ist(8, 42), sightings=1, sources=1)
    assert _decide(one_sighting)[0] == STATUS_ABSENT

    monkeypatch.setattr(settings, "ATTENDANCE_MIN_SIGHTINGS", 1)
    monkeypatch.setattr(settings, "ATTENDANCE_MIN_SPAN_MIN", 0)
    assert _decide(one_sighting)[0] == STATUS_PRESENT


def test_late_cutoff_is_configurable(monkeypatch):
    ev = evidence(ist(9, 30), ist(17, 0), sightings=5, sources=2)
    assert _decide(ev)[0] == STATUS_LATE
    monkeypatch.setattr(settings, "ATTENDANCE_LATE_AFTER", "10:00")
    assert _decide(ev)[0] == STATUS_PRESENT


# ── Department normalisation ─────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("CSE", "CSE"),
    ("  CSE  ", "CSE"),
    ("", UNASSIGNED_DEPARTMENT),
    ("   ", UNASSIGNED_DEPARTMENT),
    (None, UNASSIGNED_DEPARTMENT),
])
def test_department_normalisation(raw, expected):
    """§9-22. A NULL or blank department must bucket, not vanish."""
    assert _normalise_department(raw) == expected


# ── Physical-camera de-duplication ───────────────────────────────
#
# The "seen on 2+ cameras" branch is only sound if those are two different
# lenses. This deployment registers one physical camera twice -- MainGate and
# MainStream are channel=1 of the same NVR, differing only by `subtype`
# (main stream vs sub stream). Left unmerged, one person walking past that
# single camera would satisfy the rule from a single appearance.

DAHUA = "rtsp://user:pw@172.20.0.245:554/video/live?channel={ch}&subtype={st}"


def test_quality_variants_of_one_camera_merge():
    """The actual MainGate / MainStream pair in this deployment."""
    a = normalise_stream_source(DAHUA.format(ch=1, st=0), "camA")
    b = normalise_stream_source(DAHUA.format(ch=1, st=1), "camB")
    assert a == b


def test_different_channels_do_not_merge():
    """Two real lenses on one NVR must stay distinct."""
    a = normalise_stream_source(DAHUA.format(ch=1, st=0), "camA")
    c = normalise_stream_source(DAHUA.format(ch=2, st=0), "camC")
    assert a != c


def test_credentials_are_ignored():
    """The same camera registered twice under different logins is one camera."""
    a = normalise_stream_source("rtsp://alice:1@10.0.0.5:554/live?channel=1", "camA")
    b = normalise_stream_source("rtsp://bob:2@10.0.0.5:554/live?channel=1", "camB")
    assert a == b


def test_different_hosts_do_not_merge():
    a = normalise_stream_source("rtsp://10.0.0.5:554/live?channel=1", "camA")
    b = normalise_stream_source("rtsp://10.0.0.6:554/live?channel=1", "camB")
    assert a != b


def test_hikvision_stream_index_is_stripped():
    """/Streaming/Channels/101 and /102 are camera 1; /201 is camera 2."""
    main = normalise_stream_source("rtsp://cam/Streaming/Channels/101", "camA")
    sub = normalise_stream_source("rtsp://cam/Streaming/Channels/102", "camB")
    other = normalise_stream_source("rtsp://cam/Streaming/Channels/201", "camC")
    assert main == sub
    assert main != other


@pytest.mark.parametrize("value", ["0", "1", "", None, "browser:bc_abc", "not a url"])
def test_unparseable_sources_fall_back_to_camera_id(value):
    """
    Falling back to the camera id treats cameras as DISTINCT, which is the
    safe direction: over-merging only makes attendance harder to earn, while
    under-merging lets one appearance count twice.
    """
    assert normalise_stream_source(value, "cam-fallback") == "cam-fallback"


def test_query_param_order_is_not_significant():
    a = normalise_stream_source("rtsp://h/live?channel=1&x=2", "camA")
    b = normalise_stream_source("rtsp://h/live?x=2&channel=1", "camB")
    assert a == b


# ── Confidence gate ──────────────────────────────────────────────
#
# `confidence <= 0` means NOT RECORDED, not "low confidence". The pipeline
# leaves it unset on every exit event (tracker.check_exits builds the dict
# without the key) and on entries confirmed from the ByteTrack cache or the
# too-small-face path. Measured on this deployment: 6238/6238 exits and
# 6361/6400 entries are 0.0. A filter that treats those as failures discards
# ~99% of genuine presence evidence and marks the entire faculty absent.

def _filter_sql():
    from sqlalchemy.dialects import postgresql
    return str(
        _confidence_filter().compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_unrecorded_confidence_is_never_filtered_out():
    """The clause must admit rows at or below zero, whatever the threshold."""
    sql = _filter_sql().lower()
    assert "<= 0" in sql or "<= 0.0" in sql
    assert " or " in sql, "must be a disjunction, not a bare >= threshold"


def test_null_confidence_is_never_filtered_out():
    assert "is null" in _filter_sql().lower()


def test_threshold_is_disabled_by_default():
    """
    Shipping default is 0.0 -- an uncalibrated threshold is worse than none,
    and the two-confirmation rule is what actually carries the decision.
    Raising this without measuring the confidence distribution for enrolled
    faculty will mark real people absent.
    """
    assert settings.ATTENDANCE_MIN_CONFIDENCE == 0.0


def test_raised_threshold_still_admits_unrecorded(monkeypatch):
    monkeypatch.setattr(settings, "ATTENDANCE_MIN_CONFIDENCE", 0.55)
    sql = _filter_sql().lower()
    assert "0.55" in sql          # the gate is applied…
    assert "<= 0" in sql          # …but unrecorded evidence still gets through
