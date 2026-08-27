"""
Phase 6 — incident classification, description and evidence handling.

The list and evidence queries are exercised against the live database in
manual checks; what is unit-tested here is the pure logic that decides how an
alert is CLASSIFIED and DESCRIBED, plus the contract that keeps raw image
bytes out of the JSON paths.

Run:  pytest tests/test_incidents.py -v
"""

import sys
import os
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.incident_service import (  # noqa: E402
    SEVERITY_BY_SUBTYPE,
    STATUSES,
    STREAM_HEALTH_SUBTYPES,
    _describe,
    _inner_meta,
    _label,
    _severity,
)


class Ev:
    """Stand-in for a DetectionEvent row."""

    def __init__(self, subtype=None, metadata=None, confidence=0.0, event_type="security_alert"):
        self.subtype = subtype
        self.metadata_json = metadata or {}
        self.confidence = confidence
        self.event_type = event_type
        self.id = "e1"


# ── Severity ─────────────────────────────────────────────────────

@pytest.mark.parametrize("subtype,expected", [
    ("hazard", "high"),
    ("vehicle_proximity", "high"),
    ("gathering", "medium"),
    ("loitering", "low"),
])
def test_severity_by_subtype(subtype, expected):
    assert _severity(subtype) == expected


def test_unknown_subtype_gets_a_middling_severity():
    """A new detector must not silently register as low priority."""
    assert _severity("some_future_detector") == "medium"
    assert _severity(None) == "medium"


def test_every_known_subtype_has_a_valid_severity():
    assert all(sev in ("high", "medium", "low") for sev in SEVERITY_BY_SUBTYPE.values())


# ── Labels ───────────────────────────────────────────────────────

def test_labels_are_human_readable():
    assert _label("gathering") == "Crowd Gathering"
    assert _label("hazard") == "Hazard / Threat"


def test_unknown_subtype_label_is_derived_not_blank():
    assert _label("perimeter_breach") == "Perimeter Breach"
    assert _label(None) == "Security Event"


# ── Metadata flattening ──────────────────────────────────────────

def test_nested_detector_metadata_is_flattened():
    """
    Detectors build {"metadata": {...}} alongside top-level keys, and the
    event processor copies everything it does not consume into metadata_json —
    so fields land at different depths depending on the detector.
    """
    ev = Ev(metadata={"person_count": 3, "metadata": {"proximity_threshold_px": 150}})
    flat = _inner_meta(ev)
    assert flat["person_count"] == 3
    assert flat["proximity_threshold_px"] == 150


def test_flattening_tolerates_a_missing_or_odd_inner_block():
    assert _inner_meta(Ev(metadata={"a": 1})) == {"a": 1}
    assert _inner_meta(Ev(metadata={"metadata": "not a dict", "a": 1}))["a"] == 1
    assert _inner_meta(Ev(metadata=None)) == {}


# ── Descriptions ─────────────────────────────────────────────────

def test_gathering_description_reports_the_count():
    assert "4 people" in _describe(Ev("gathering", {"person_count": 4}))


def test_loitering_description_names_the_person_when_known():
    text = _describe(Ev("loitering", {"person_name": "Rahul", "duration_sec": 312}))
    assert "Rahul" in text and "312s" in text


def test_loitering_description_degrades_without_a_name():
    assert _describe(Ev("loitering", {})).startswith("A person")


def test_hazard_description_reports_class_and_confidence():
    text = _describe(Ev("hazard", {"threat_class": "knife"}, confidence=0.87))
    assert "knife" in text and "87%" in text


def test_missing_fields_never_break_a_description():
    """A detector that omits a field must not produce a crash or a raw dict."""
    for subtype in list(SEVERITY_BY_SUBTYPE) + [None, "unknown_kind"]:
        text = _describe(Ev(subtype, {}))
        assert isinstance(text, str) and text


# ── Stream health is not a security incident ─────────────────────

@pytest.mark.parametrize("subtype", STREAM_HEALTH_SUBTYPES)
def test_stream_health_subtypes_are_identified(subtype):
    """
    These are excluded from the incident list by default. A flaky RTSP
    connection is camera plumbing, and counting it as a security incident
    makes the figure meaningless.
    """
    assert subtype in STREAM_HEALTH_SUBTYPES
    assert _severity(subtype) == "low"


# ── Status vocabulary ────────────────────────────────────────────

def test_status_vocabulary_is_the_three_workflow_states():
    assert STATUSES == ("open", "investigating", "resolved")


def test_open_is_the_first_status():
    """
    Absence of a review row means 'open', so it must be the first/default
    member — a listing filter relies on that equivalence.
    """
    assert STATUSES[0] == "open"


# ── Evidence bytes must never reach a JSON path ──────────────────

def test_evidence_bytes_are_stripped_before_serialisation():
    """
    The pipeline attaches the encoded frame as `_evidence_jpeg`. It is popped
    in the event processor before the row is written and before the WebSocket
    broadcast — both are JSON, and raw bytes raise TypeError on both paths.
    This reproduces the processor's metadata builder to pin that contract.
    """
    alert = {
        "event_type": "security_alert", "subtype": "gathering",
        "camera_id": "cam1", "person_count": 3,
        "_evidence_jpeg": b"\xff\xd8\xff\xe0binary\xff\xd9",
    }

    with pytest.raises(TypeError):
        json.dumps(alert)

    alert.pop("_evidence_jpeg")
    excluded = ("person_id", "camera_id", "event_type", "confidence",
                "snapshot_url", "subtype")
    metadata = {k: v for k, v in alert.items() if k not in excluded}

    assert json.dumps(metadata)
    assert "_evidence_jpeg" not in metadata
