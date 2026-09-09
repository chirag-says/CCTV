"""
Product plans: basic / plus / pro.

WHY THIS EXISTS
Every capacity and pricing figure in this project is stated per plan -- the
table in OPTIMIZATION_BRIEF.md, cost_model.py, phase_model.py, and the Phase 1
acceptance test ("20 Basic cameras at 10 fps") -- and until now no
configuration could run one. Face detection ran unconditionally in face mode,
so a camera sold as Basic still paid for SCRFD and ArcFace on every analysed
frame. The acceptance test was therefore unrunnable as written.

The plan CAPS the ENABLE_* flags rather than replacing them, so "pro"
reproduces exactly what the config did before plans existed. That is what
makes this safe to default to, and it is the first thing tested.
"""

import pytest

from app.config import Settings


def s(plan=None, **kw):
    if plan is not None:
        kw["PIPELINE_PLAN"] = plan
    return Settings(**kw)


# ── the default must not change anything ──────────────────────────────────

def test_default_plan_is_pro():
    """Anything else would silently switch features off in an existing
    deployment on upgrade."""
    assert Settings().plan == "pro"


def test_pro_allows_everything():
    cfg = s("pro")
    assert cfg.PLAN_ALLOWS_FACE
    assert cfg.PLAN_ALLOWS_ANPR
    assert cfg.PLAN_ALLOWS_WEAPONS
    assert cfg.PLAN_ALLOWS_REID


# ── the tiers, as the brief defines them ──────────────────────────────────

def test_basic_is_person_detection_and_rules_only():
    """Basic: person detect + ByteTrack + zone / crowd / loitering rules.
    No face pipeline, no ANPR, no weapons, no ReID."""
    cfg = s("basic")
    assert not cfg.PLAN_ALLOWS_FACE
    assert not cfg.PLAN_ALLOWS_ANPR
    assert not cfg.PLAN_ALLOWS_WEAPONS
    assert not cfg.PLAN_ALLOWS_REID


def test_plus_adds_face_recognition_and_nothing_else():
    cfg = s("plus")
    assert cfg.PLAN_ALLOWS_FACE
    assert not cfg.PLAN_ALLOWS_ANPR
    assert not cfg.PLAN_ALLOWS_WEAPONS
    assert not cfg.PLAN_ALLOWS_REID


def test_tiers_are_strictly_nested():
    """Each tier is a superset of the one below. If that ever stops being
    true, the pricing table stops meaning anything."""
    order = ["basic", "plus", "pro"]
    features = ["PLAN_ALLOWS_FACE", "PLAN_ALLOWS_ANPR",
                "PLAN_ALLOWS_WEAPONS", "PLAN_ALLOWS_REID"]
    enabled = [
        {f for f in features if getattr(s(p), f)} for p in order
    ]
    for lower, higher in zip(enabled, enabled[1:]):
        assert lower < higher or lower == higher, (lower, higher)
        assert lower.issubset(higher)


# ── the plan caps, it does not enable ─────────────────────────────────────

def test_plan_does_not_switch_a_disabled_feature_on():
    """Pro plus ENABLE_REID=False must still mean no ReID. The plan is an
    entitlement ceiling, not an override."""
    cfg = s("pro", ENABLE_REID=False)
    assert cfg.PLAN_ALLOWS_REID, "the plan permits it..."
    assert cfg.ENABLE_REID is False, "...but the operator said no"


# ── bad input ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["BASIC", " basic ", "Plus", "PRO"])
def test_case_and_whitespace_are_tolerated(value):
    assert s(value).plan == value.strip().lower()


@pytest.mark.parametrize("value", ["enterprise", "", "free", "1"])
def test_unknown_plan_falls_back_to_pro(value):
    """An unrecognised plan must not quietly downgrade a paying camera to
    Basic and stop detecting faces. Fall back to the most capable tier and
    let the bill be the thing that complains."""
    assert s(value).plan == "pro"
