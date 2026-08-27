"""
Phase 1 foundation tests — campus-local time and person classification.

These are the two pieces every college feature is built on:

  * campus_day_bounds() decides what "today" means. Everything is stored in
    UTC, and UTC midnight is 05:30 in IST, so a UTC-day boundary splits a
    college day in the wrong place. Attendance, journeys and the daily
    summary are all wrong if this is wrong.

  * person_type decides who counts as faculty. It used to be inferable only
    from the free-text `role` column, whose documented values did not match
    what the seed scripts actually wrote.

No database and no camera required — these are pure functions.

Run:  pytest tests/test_campus_time.py -v
"""

import sys
import os
from datetime import datetime, timezone, date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.time_utils import (  # noqa: E402
    campus_day_bounds,
    campus_tz,
    to_campus_local,
    parse_campus_date,
    is_after_campus_time,
    _parse_hhmm,
)
from app.models.person import infer_person_type, PERSON_TYPES  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────

def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# ── Timezone plumbing ────────────────────────────────────────────

def test_campus_tz_resolves():
    """
    Fails on Windows if the `tzdata` package is missing -- Windows ships no
    system IANA database. That is why tzdata is pinned in requirements.txt.
    """
    assert campus_tz() is not None
    assert str(campus_tz()) == "Asia/Kolkata"


def test_naive_datetimes_are_treated_as_utc():
    """
    The psycopg2 driver can hand back naive datetimes. Every writer in this
    codebase stores UTC, so a naive value must not be read as local time --
    that would shift every timestamp by the UTC offset.
    """
    naive = datetime(2026, 8, 27, 3, 12)
    aware = utc(2026, 8, 27, 3, 12)
    assert to_campus_local(naive) == to_campus_local(aware)


def test_to_campus_local_of_none_is_none():
    assert to_campus_local(None) is None


# ── Day boundaries ───────────────────────────────────────────────

def test_operating_day_bounds_are_utc_shifted():
    """06:00-21:00 IST is 00:30-15:30 UTC."""
    start, end = campus_day_bounds("2026-08-27")
    assert start == utc(2026, 8, 27, 0, 30)
    assert end == utc(2026, 8, 27, 15, 30)


def test_full_calendar_day_crosses_the_utc_boundary():
    """
    A full IST calendar day starts at 18:30 UTC on the PREVIOUS date. This
    is the case a naive UTC-midnight implementation gets wrong.
    """
    start, end = campus_day_bounds("2026-08-27", day_start="00:00", day_end="24:00")
    assert start == utc(2026, 8, 26, 18, 30)
    assert end == utc(2026, 8, 27, 18, 30)


def test_late_evening_event_lands_in_the_correct_campus_day():
    """
    §9 test 20. 23:50 IST on the 27th is 18:20 UTC on the 27th. Under a UTC
    day it belongs to the 27th by luck; the case that actually breaks is
    00:30 IST on the 28th (19:00 UTC on the 27th), which a UTC-day query
    would file under the 27th.
    """
    start, end = campus_day_bounds("2026-08-27", day_start="00:00", day_end="24:00")
    assert start <= utc(2026, 8, 27, 18, 20) < end          # 23:50 IST, 27th

    # 00:30 IST on the 28th must NOT fall in the 27th.
    assert not (start <= utc(2026, 8, 27, 19, 0) < end)

    nxt_start, nxt_end = campus_day_bounds("2026-08-28", day_start="00:00", day_end="24:00")
    assert nxt_start <= utc(2026, 8, 27, 19, 0) < nxt_end


def test_bounds_are_half_open():
    """end is exclusive, so adjacent days never double-count an event."""
    _, end = campus_day_bounds("2026-08-27", day_start="00:00", day_end="24:00")
    nxt_start, _ = campus_day_bounds("2026-08-28", day_start="00:00", day_end="24:00")
    assert end == nxt_start


def test_inverted_window_is_rejected():
    """A window ending before it starts would silently return zero rows."""
    with pytest.raises(ValueError, match="must be after"):
        campus_day_bounds("2026-08-27", day_start="21:00", day_end="06:00")


def test_equal_window_is_rejected():
    with pytest.raises(ValueError):
        campus_day_bounds("2026-08-27", day_start="09:00", day_end="09:00")


def test_malformed_date_is_rejected():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_campus_date("27-08-2026")


def test_default_date_is_campus_today_not_utc_today():
    """
    Between 18:30 and 24:00 UTC the campus date is already tomorrow. The
    default must follow the campus, not UTC.
    """
    assert parse_campus_date() == to_campus_local(datetime.now(timezone.utc)).date()
    assert isinstance(parse_campus_date("2026-08-27"), date)


# ── Late-arrival rule ────────────────────────────────────────────

def test_late_arrival_is_evaluated_in_campus_local_time():
    """§9 tests 1-2. 03:12 UTC = 08:42 IST (on time); 04:17 UTC = 09:47 IST (late)."""
    assert is_after_campus_time(utc(2026, 8, 27, 4, 17), "09:15") is True
    assert is_after_campus_time(utc(2026, 8, 27, 3, 12), "09:15") is False


def test_late_boundary_is_exclusive():
    """Arriving exactly at the cutoff is on time, not late."""
    assert is_after_campus_time(utc(2026, 8, 27, 3, 45), "09:15") is False  # 09:15 IST
    assert is_after_campus_time(utc(2026, 8, 27, 3, 46), "09:15") is True   # 09:16 IST


@pytest.mark.parametrize("value,expected", [
    ("09:15", (9, 15, 0)),
    ("9:5", (9, 5, 0)),
    ("06:00:30", (6, 0, 30)),
])
def test_hhmm_parsing(value, expected):
    t = _parse_hhmm(value)
    assert (t.hour, t.minute, t.second) == expected


@pytest.mark.parametrize("bad", ["", "nine", "09-15", None])
def test_bad_hhmm_is_rejected(bad):
    with pytest.raises(ValueError):
        _parse_hhmm(bad)


# ── Person classification ────────────────────────────────────────

@pytest.mark.parametrize("role,expected", [
    ("assistant_professor", "faculty"),
    ("associate_professor", "faculty"),
    ("professor", "faculty"),
    ("hod", "faculty"),
    ("lecturer", "faculty"),
    ("teacher", "faculty"),
    ("student", "student"),
    ("employee", "staff"),
    ("staff", "staff"),
    ("visitor", "visitor"),
    # 'vip' and 'banned' are access flags, not roles -- they must not
    # accidentally classify someone as faculty.
    ("vip", "visitor"),
    ("banned", "visitor"),
    ("", "visitor"),
    (None, "visitor"),
])
def test_person_type_inference(role, expected):
    assert infer_person_type(role) == expected


def test_person_type_inference_is_case_and_space_tolerant():
    assert infer_person_type("  Assistant_Professor  ") == "faculty"


def test_inference_only_returns_known_types():
    for role in ("hod", "student", "employee", "anything_else", None):
        assert infer_person_type(role) in PERSON_TYPES
