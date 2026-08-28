"""
Step 5 -- "who was in Classroom 204 at 14:00" becomes a query.

WHAT THIS REPLACES
The evidence pack answered with "unknown_faces rows whose FIRST_SEEN falls in
the window". first_seen is the first moment that identity was ever observed,
anywhere. So the old answer:

  * found strangers who merely happened to be newly discovered that minute
  * missed every repeat visitor -- which is most of them, and precisely the
    people most likely to be involved in an incident

Migration 011 promoted the stranger's id out of JSON into an indexed column,
so known and unknown subjects can now be selected together in one pass over
detection_events.

The subtle half is that presence is an INTERVAL. Someone who entered at 13:50
and left at 14:20 was present at 14:00 while having no event near it.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.incident_service import IncidentService

T0 = datetime(2026, 8, 28, 14, 0, tzinfo=timezone.utc)
LOOKBACK = timedelta(seconds=1800)


class _Row(tuple):
    """(person_id, unknown_person_id, event_type, subtype, created_at)."""


def _q(rows):
    """A stand-in for the SQLAlchemy chain _people_present builds."""
    class _Q:
        def __init__(self, r): self._r = r
        def filter(self, *a, **k): return self
        def order_by(self, *a, **k): return self
        def all(self): return self._r
    return _Q(rows)


class _FakePerson:
    def __init__(self, pid, name, role="student"):
        self.id, self.name, self.role = pid, name, role


class _FakeUnknown:
    def __init__(self, uid):
        self.id = uid
        self.snapshot_url = f"/snap/{uid}.jpg"
        self.full_frame = f"/frame/{uid}.jpg"
        self.occurrence = 3
        self.status = "pending"


class _FakeDB:
    """Returns event rows for the first query, then Person/UnknownFace rows."""

    def __init__(self, events, persons=(), unknowns=()):
        self._events, self._persons, self._unknowns = events, persons, unknowns
        self._n = 0

    def query(self, *cols):
        self._n += 1
        if self._n == 1:
            return _q(self._events)
        # Subsequent queries are the enrichment lookups, in order.
        model = getattr(cols[0], "__name__", "")
        if model == "Person" or (self._persons and self._n == 2):
            return _q(list(self._persons))
        return _q(list(self._unknowns))


def _present(events, persons=(), unknowns=(), window_sec=120):
    db = _FakeDB(events, persons, unknowns)
    lo, hi = T0 - timedelta(seconds=window_sec), T0 + timedelta(seconds=window_sec)
    return IncidentService._people_present(db, ["cam-1"], lo, hi, LOOKBACK)


def ev(person=None, unknown=None, kind="detection", sub=None, offset_sec=0):
    return (person, unknown, kind, sub, T0 + timedelta(seconds=offset_sec))


# -- The regression this step is about ---------------------------


class TestRepeatVisitorsAreFound:
    def test_a_stranger_seen_during_the_window_is_present(self):
        out = _present(
            [ev(unknown="u-1", offset_sec=10)],
            unknowns=[_FakeUnknown("u-1")],
        )
        assert [u["unknown_id"] for u in out["unknown"]] == ["u-1"]
        assert out["unknown"][0]["basis"] == "observed"

    def test_first_seen_long_ago_does_not_exclude_them(self):
        """
        The old query keyed on unknown_faces.first_seen, so a stranger whose
        first-ever sighting was weeks earlier was invisible to an incident
        happening right in front of the camera.
        """
        out = _present(
            [ev(unknown="u-9", kind="entry", offset_sec=-30),
             ev(unknown="u-9", kind="exit", offset_sec=30)],
            unknowns=[_FakeUnknown("u-9")],
        )
        assert len(out["unknown"]) == 1, "presence is about NOW, not discovery"

    def test_known_and_unknown_are_returned_together(self):
        out = _present(
            [ev(person="p-1", offset_sec=5), ev(unknown="u-1", offset_sec=8)],
            persons=[_FakePerson("p-1", "Rahul")],
            unknowns=[_FakeUnknown("u-1")],
        )
        assert [k["name"] for k in out["known"]] == ["Rahul"]
        assert [u["unknown_id"] for u in out["unknown"]] == ["u-1"]
        assert out["total_identified"] == 2


# -- Presence is an interval -------------------------------------


class TestIntervalReconstruction:
    def test_entered_before_and_left_after_counts_as_present(self):
        """
        Entered 13:50, left 14:20, incident at 14:00 -- no event anywhere near
        the window, yet unquestionably in the room.
        """
        out = _present(
            [ev(person="p-1", kind="entry", offset_sec=-600),
             ev(person="p-1", kind="exit", offset_sec=1200)],
            persons=[_FakePerson("p-1", "Priya")],
        )
        assert len(out["known"]) == 1
        assert out["known"][0]["basis"] == "spanned", "no event fell inside the window"

    def test_entered_before_and_never_left_counts_as_present(self):
        out = _present(
            [ev(person="p-1", kind="entry", offset_sec=-600)],
            persons=[_FakePerson("p-1", "Priya")],
        )
        assert len(out["known"]) == 1

    def test_left_before_the_window_is_absent(self):
        out = _present(
            [ev(person="p-1", kind="entry", offset_sec=-1200),
             ev(person="p-1", kind="exit", offset_sec=-600)],
            persons=[_FakePerson("p-1", "Priya")],
        )
        assert out["known"] == [], "they had already gone"

    def test_basis_distinguishes_observed_from_inferred(self):
        """An operator must be able to tell a sighting from an inference."""
        out = _present(
            [ev(person="p-1", kind="entry", offset_sec=-600),
             ev(person="p-1", kind="detection", offset_sec=0),
             ev(person="p-1", kind="exit", offset_sec=600)],
            persons=[_FakePerson("p-1", "Priya")],
        )
        assert out["known"][0]["basis"] == "observed"


# -- Unidentified occupants --------------------------------------


class TestUnidentifiedOccupants:
    def test_counted_but_never_named(self):
        """
        Step 1.5 bodies: tracked, never identified. They belong in the count
        of who was there, and can never be attributed to a person.
        """
        out = _present([
            ev(kind="detection", sub="unidentified_entry", offset_sec=5),
            ev(kind="detection", sub="unidentified_entry", offset_sec=40),
        ])
        assert out["unidentified_count"] == 2
        assert out["known"] == [] and out["unknown"] == []

    def test_only_counted_inside_the_window(self):
        out = _present([
            ev(kind="detection", sub="unidentified_entry", offset_sec=-900),
        ])
        assert out["unidentified_count"] == 0

    def test_they_do_not_inflate_the_identified_total(self):
        out = _present(
            [ev(person="p-1", offset_sec=1),
             ev(kind="detection", sub="unidentified_entry", offset_sec=2)],
            persons=[_FakePerson("p-1", "Rahul")],
        )
        assert out["total_identified"] == 1
        assert out["unidentified_count"] == 1


class TestShape:
    def test_empty_window_is_empty_not_an_error(self):
        out = _present([])
        assert out == {"known": [], "unknown": [], "unidentified_count": 0,
                       "total_identified": 0}

    def test_events_are_counted_per_subject(self):
        out = _present(
            [ev(person="p-1", offset_sec=i) for i in range(4)],
            persons=[_FakePerson("p-1", "Rahul")],
        )
        assert out["known"][0]["events"] == 4

    def test_rows_without_any_subject_are_ignored(self):
        """Security alerts and stream-health rows are not people."""
        out = _present([ev(kind="security_alert", offset_sec=0)])
        assert out["total_identified"] == 0
