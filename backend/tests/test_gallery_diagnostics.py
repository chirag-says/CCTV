"""
Gallery diagnostics suite.

Headless: no camera, GPU, or database. Builds galleries with KNOWN geometry so
each defect the tool is supposed to catch is planted deliberately, then asserts
it is caught.

Each test names the real-world failure it protects against. The value of this
tool is that it tells you to fix DATA rather than to loosen a threshold, so the
tests are mostly about not misdiagnosing: a clean gallery must not be flagged,
and a contaminated one must not be reported as merely "ambiguous".
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.vision.gallery_diagnostics import (
    RISK_GOOD,
    RISK_HIGH,
    RISK_MEDIUM,
    Thresholds,
    analyse_gallery,
    format_report,
    report_to_dict,
)

DIM = 512


def _unit(v):
    return v / np.linalg.norm(v)


def at_cosine(base, cos_target, rng):
    """Unit vector at exactly `cos_target` cosine similarity to base."""
    n = rng.randn(len(base))
    n -= base * float(np.dot(n, base))
    n = _unit(n)
    return base * cos_target + n * float(np.sqrt(1.0 - cos_target ** 2))


def _rng(seed=7):
    return np.random.RandomState(seed)


def _thresholds(**kw):
    """Explicit thresholds so tests do not drift when settings are retuned."""
    base = dict(
        match_distance=0.62,
        match_margin=0.10,
        verify_distance=0.55,
        verify_margin=0.12,
        duplicate_distance=0.15,
        min_embeddings=5,
        target_embeddings=10,
    )
    base.update(kw)
    return Thresholds(**base)


def _person(name, base, cosines, rng, quality=0.9, pid=None):
    """A person whose embeddings sit at the given cosine similarities to `base`."""
    pid = pid or f"id-{name}"
    return [
        {
            "person_id": pid,
            "person_name": name,
            "encoding_id": f"{pid}-{i}",
            "encoding": _unit(at_cosine(base, c, rng)),
            "quality": quality,
        }
        for i, c in enumerate(cosines)
    ]


def _tight(name, base, rng, n=8, cos=0.88, **kw):
    """A well-enrolled person: several embeddings, consistent with each other."""
    return _person(name, base, [cos] * n, rng, pid=kw.pop("pid", None), **kw)


# ── The core guarantee: a clean gallery is not flagged ────────────────────

def test_well_separated_gallery_is_clean():
    """
    PREVENTS: crying wolf. If the tool flags a healthy gallery, its output gets
    ignored and the real problems are never fixed.
    """
    rng = _rng()
    rows = []
    for i in range(4):
        # Orthogonal-ish bases => people are far apart in embedding space.
        rows += _tight(f"person{i}", _unit(rng.randn(DIM)), rng)

    report = analyse_gallery(rows, thresholds=_thresholds())

    assert report.person_count == 4
    assert report.total_misidentified == 0, "clean gallery reported as contaminated"
    assert not report.high_risk, [(p.person_name, p.reasons) for p in report.high_risk]
    assert all(p.risk == RISK_GOOD for p in report.persons), \
        [(p.person_name, p.risk, p.reasons) for p in report.persons]
    assert report.overall_recognise_rate() == 1.0


# ── The defect that produced "margin=0.0153" in the logs ─────────────────

def test_two_near_identical_people_are_flagged_as_confusable():
    """
    PREVENTS: reading `security 1 vs security 3, margin=0.0153` as a threshold
    problem and lowering FACE_MATCH_MIN_MARGIN — which would convert an honest
    rejection into a silent identity swap.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    rows = _tight("security 1", base, rng, pid="s1")
    # Second person sits almost on top of the first.
    rows += _tight("security 3", _unit(at_cosine(base, 0.93, rng)), rng, pid="s3")
    rows += _tight("reena", _unit(rng.randn(DIM)), rng, pid="reena")

    report = analyse_gallery(rows, thresholds=_thresholds())
    by_name = {p.person_name: p for p in report.persons}

    assert by_name["security 1"].nearest_person_name == "security 3"
    assert by_name["security 3"].nearest_person_name == "security 1"
    assert by_name["security 1"].risk == RISK_HIGH
    assert by_name["reena"].risk == RISK_GOOD, "unrelated person dragged into the alarm"

    worst = report.confusable_pairs[0]
    assert {worst.a_name, worst.b_name} == {"security 1", "security 3"}

    # The recommendation must name the confusable counterpart, not just say "bad".
    assert "security 3" in by_name["security 1"].recommendation


def test_confusable_people_fail_recovery_before_they_fail_recognition():
    """
    PREVENTS: misreading `verify_ambiguous_margin`. Recovery uses a STRICTER
    margin than recognition, so a pair can be recognisable yet unrecoverable —
    which is exactly what the recovery-rejection logs are reporting.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    rows = _tight("a", base, rng, pid="a")
    rows += _tight("b", _unit(at_cosine(base, 0.55, rng)), rng, pid="b")

    t = _thresholds(match_margin=0.02, verify_margin=0.50)
    report = analyse_gallery(rows, thresholds=t)

    assert report.overall_recognise_rate() == 1.0, "should still be recognisable"
    assert report.overall_verify_rate() < 1.0, \
        "stricter recovery gate should reject where recognition accepts"
    assert any("recovery" in r for p in report.persons for r in p.reasons)


# ── The most severe defect: wrong face enrolled under a person ───────────

def test_contaminated_embedding_is_reported_as_wrong_person():
    """
    PREVENTS: the worst gallery defect — someone else's face enrolled under this
    identity. It must be called out as contamination (remove the embedding), not
    as a margin problem (add more captures), because the two fixes are opposite.
    """
    rng = _rng()
    alice_base = _unit(rng.randn(DIM))
    bob_base = _unit(rng.randn(DIM))

    rows = _tight("alice", alice_base, rng, pid="alice")
    rows += _tight("bob", bob_base, rng, pid="bob")
    # Bob's face, enrolled under Alice.
    rows.append({
        "person_id": "alice",
        "person_name": "alice",
        "encoding_id": "alice-contaminated",
        "encoding": _unit(at_cosine(bob_base, 0.97, rng)),
        "quality": 0.9,
    })

    report = analyse_gallery(rows, thresholds=_thresholds())
    alice = next(p for p in report.persons if p.person_name == "alice")

    assert alice.misidentified_count == 1, \
        f"contamination missed (misidentified={alice.misidentified_count})"
    assert "alice-contaminated" in [c.encoding_id for c in alice.contaminated_encodings]
    assert alice.risk == RISK_HIGH
    assert "remove" in alice.recommendation.lower(), \
        f"told the user to add captures instead of removing: {alice.recommendation}"

    bad = next(c for c in alice.contaminated_encodings)
    assert bad.rejection_reason(report.thresholds) == "would_match_wrong_person"


def test_outlier_embedding_that_matches_nobody_is_flagged():
    """
    PREVENTS: a garbage embedding (extreme pose, motion blur, wrong crop) sitting
    in the gallery forever, dragging that person's worst-case numbers down while
    contributing nothing.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    rows = _tight("carol", base, rng, pid="carol")
    rows.append({
        "person_id": "carol",
        "person_name": "carol",
        "encoding_id": "carol-outlier",
        "encoding": _unit(rng.randn(DIM)),   # unrelated to anyone
        "quality": 0.9,
    })
    rows += _tight("dave", _unit(rng.randn(DIM)), rng, pid="dave")

    report = analyse_gallery(rows, thresholds=_thresholds())
    carol = next(p for p in report.persons if p.person_name == "carol")

    assert carol.risk in (RISK_HIGH, RISK_MEDIUM)
    assert any("do not match any other embedding" in r for r in carol.reasons), carol.reasons
    assert carol.intra_max > 0.62


# ── Split identities: one human, several person records ─────────────────

def test_same_person_enrolled_twice_is_reported_as_a_merge_not_a_re_enrollment():
    """
    PREVENTS: the misdiagnosis this whole tool exists to stop. When one human
    holds two person records, the recogniser is asked to tell them apart from
    themselves. "Add more captures" makes that WORSE — more embeddings of the
    same face on both sides drives the margin to zero. The only fix is a merge,
    so the report has to say merge.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    rows = _tight("security 1", base, rng, n=4, pid="rec-a")
    # Same human, enrolled again as a separate record.
    rows += _tight("security 1", _unit(at_cosine(base, 0.97, rng)), rng, n=3, pid="rec-b")
    rows += _tight("reena", _unit(rng.randn(DIM)), rng, pid="reena")

    report = analyse_gallery(rows, thresholds=_thresholds())

    assert len(report.name_collisions) == 1, report.name_collisions
    collision = report.name_collisions[0]
    assert set(collision.person_ids) == {"rec-a", "rec-b"}
    assert collision.embedding_counts in ([4, 3], [3, 4])
    assert collision.likely_same_person, collision.min_distance

    for pid in ("rec-a", "rec-b"):
        p = next(x for x in report.persons if x.person_id == pid)
        assert p.risk == RISK_HIGH
        assert "MERGE" in p.recommendation, p.recommendation
        # The merge instruction must come first — it invalidates the rest.
        assert p.recommendation.startswith("MERGE"), p.recommendation
        assert "enrolled 2 times" in p.reasons[0], p.reasons

    reena = next(x for x in report.persons if x.person_id == "reena")
    assert reena.risk == RISK_GOOD, "unrelated person caught in the collision report"


def test_distinct_people_sharing_a_name_are_not_told_to_merge():
    """
    PREVENTS: destroying data on a name clash. Two different people can legitimately
    share a name; the tool must flag the ambiguity without asserting they are the
    same person.
    """
    rng = _rng()
    rows = _tight("A Kumar", _unit(rng.randn(DIM)), rng, pid="k1")
    rows += _tight("a  kumar", _unit(rng.randn(DIM)), rng, pid="k2")  # normalised match

    report = analyse_gallery(rows, thresholds=_thresholds())

    assert len(report.name_collisions) == 1, "name normalisation missed a collision"
    collision = report.name_collisions[0]
    assert not collision.likely_same_person, collision.min_distance

    p = next(x for x in report.persons if x.person_id == "k1")
    assert "MERGE" not in p.recommendation, p.recommendation
    assert "distinct names" in p.recommendation, p.recommendation


def test_similar_but_differently_named_people_are_not_treated_as_split():
    """
    PREVENTS: fuzzy name matching quietly merging two real people. A typo'd name
    is a confusable pair, not an asserted duplicate — the human decides.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    rows = _tight("security 3", base, rng, pid="s3")
    rows += _tight("secutiry 3", _unit(at_cosine(base, 0.96, rng)), rng, pid="s3-typo")

    report = analyse_gallery(rows, thresholds=_thresholds())

    assert report.name_collisions == [], "fuzzy-matched two distinct names"
    # It must still be visible as a confusable pair, so a human can act on it.
    assert len(report.confusable_pairs) == 1
    pair = report.confusable_pairs[0]
    assert {pair.a_name, pair.b_name} == {"security 3", "secutiry 3"}
    assert pair.severity(_thresholds()) != RISK_GOOD, pair.min_distance


# ── Coverage and redundancy ──────────────────────────────────────────────

def test_single_embedding_person_is_flagged_but_not_called_contaminated():
    """
    PREVENTS: the tool inventing a defect. With one embedding there is nothing to
    leave out, so LOO is undefined — that must read as "cannot be measured", not
    as a failed match or a wrong-person hit.
    """
    rng = _rng()
    rows = _person("solo", _unit(rng.randn(DIM)), [1.0], rng, pid="solo")
    rows += _tight("other", _unit(rng.randn(DIM)), rng, pid="other")

    report = analyse_gallery(rows, thresholds=_thresholds())
    solo = next(p for p in report.persons if p.person_name == "solo")

    assert solo.risk == RISK_HIGH
    assert solo.misidentified_count == 0, "single embedding reported as contaminated"
    assert solo.worst_margin is None, "invented a margin with nothing to compare against"
    assert any("1 embedding" in r for r in solo.reasons), solo.reasons
    assert report.unmeasurable_count == 1
    # The measurable person must not be dragged down by the unmeasurable one.
    assert report.overall_recognise_rate() == 1.0


def test_thin_coverage_is_medium_not_high():
    """
    PREVENTS: risk inflation. Three good embeddings is thin, not broken — if it
    reads HIGH alongside genuine contamination, the worklist loses its ordering.
    """
    rng = _rng()
    rows = _tight("thin", _unit(rng.randn(DIM)), rng, n=3)
    rows += _tight("full", _unit(rng.randn(DIM)), rng, n=10, pid="full")

    report = analyse_gallery(rows, thresholds=_thresholds())
    thin = next(p for p in report.persons if p.person_name == "thin")

    assert thin.risk == RISK_MEDIUM, thin.reasons
    assert thin in report.reenrollment_candidates


def test_near_duplicate_embeddings_are_counted():
    """
    PREVENTS: "more embeddings = better recognition". Ten copies of one pose cost
    search time and add no coverage; the report has to say so.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    # 0.999 cosine => distance ~0.001, far below the 0.15 duplicate threshold.
    rows = _person("dup", base, [0.999] * 6, rng, pid="dup")
    rows += _tight("other", _unit(rng.randn(DIM)), rng, pid="other")

    report = analyse_gallery(rows, thresholds=_thresholds())
    dup = next(p for p in report.persons if p.person_name == "dup")

    assert dup.duplicate_pairs == 15, f"expected C(6,2)=15 pairs, got {dup.duplicate_pairs}"
    assert any("duplicate" in r for r in dup.reasons)
    # Redundancy alone is wasteful, not dangerous — it must not raise the band.
    assert dup.risk == RISK_GOOD, dup.reasons


# ── Ordering and output contracts ────────────────────────────────────────

def test_worklist_is_ordered_worst_first():
    """PREVENTS: an unactionable report — the first row must be the first fix."""
    rng = _rng()
    shared = _unit(rng.randn(DIM))
    rows = _tight("clean", _unit(rng.randn(DIM)), rng, n=10, pid="clean")
    rows += _tight("thin", _unit(rng.randn(DIM)), rng, n=3, pid="thin")
    rows += _tight("twin_a", shared, rng, pid="twin_a")
    rows += _tight("twin_b", _unit(at_cosine(shared, 0.95, rng)), rng, pid="twin_b")

    report = analyse_gallery(rows, thresholds=_thresholds())
    bands = [p.risk for p in report.persons]

    assert bands == sorted(bands, key=lambda b: {RISK_HIGH: 0, RISK_MEDIUM: 1, RISK_GOOD: 2}[b])
    assert report.persons[0].risk == RISK_HIGH
    assert report.persons[-1].person_name == "clean"


def test_unusable_rows_are_skipped_not_crashed_on():
    """PREVENTS: one corrupt DB row taking down the whole report."""
    rng = _rng()
    rows = _tight("ok", _unit(rng.randn(DIM)), rng, pid="ok")
    rows += _tight("ok2", _unit(rng.randn(DIM)), rng, pid="ok2")
    rows += [
        {"person_id": "bad", "person_name": "bad", "encoding_id": "z", "encoding": None},
        {"person_id": "bad", "person_name": "bad", "encoding_id": "w",
         "encoding": np.zeros(DIM, dtype=np.float32)},          # zero norm
        {"person_id": "bad", "person_name": "bad", "encoding_id": "y",
         "encoding": np.ones(128, dtype=np.float32)},           # wrong dimension
        {"person_id": "", "person_name": "no id", "encoding_id": "x",
         "encoding": _unit(rng.randn(DIM))},                    # missing person
    ]

    report = analyse_gallery(rows, thresholds=_thresholds())

    assert report.skipped_embeddings == 4
    assert report.person_count == 2
    assert "bad" not in [p.person_name for p in report.persons]


def test_empty_gallery_is_handled():
    """PREVENTS: a crash on a fresh install with nobody enrolled."""
    report = analyse_gallery([], thresholds=_thresholds())
    assert report.person_count == 0
    assert report.overall_recognise_rate() is None
    assert "empty" in format_report(report).lower()


def test_report_renders_and_serialises():
    """PREVENTS: a diagnostic that computes correctly but cannot be read or stored."""
    rng = _rng()
    base = _unit(rng.randn(DIM))
    rows = _tight("alpha", base, rng, pid="alpha")
    rows += _tight("beta", _unit(at_cosine(base, 0.9, rng)), rng, pid="beta")
    rows += _person("gamma", _unit(rng.randn(DIM)), [1.0], rng, pid="gamma")

    report = analyse_gallery(rows, thresholds=_thresholds())

    text = format_report(report, verbose=True)
    assert "FACE GALLERY DIAGNOSTICS" in text
    assert "alpha" in text and "gamma" in text
    assert text.isascii(), "non-ASCII output will break a Windows console"

    import json
    payload = json.loads(json.dumps(report_to_dict(report)))
    assert payload["summary"]["persons"] == 3
    assert len(payload["persons"]) == 3
    assert payload["summary"]["unmeasurable"] == 1
    # None must survive as null, not as a NaN that json cannot round-trip.
    assert payload["persons"][[p["person_name"] for p in payload["persons"]].index("gamma")][
        "worst_margin"] is None


def test_thresholds_are_replayable_without_restarting_anything():
    """
    PREVENTS: tuning by trial and error on a live camera. A stricter margin must
    be answerable offline, against the gallery you already have.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    rows = _tight("p1", base, rng, pid="p1")
    rows += _tight("p2", _unit(at_cosine(base, 0.6, rng)), rng, pid="p2")

    lenient = analyse_gallery(rows, thresholds=_thresholds(match_margin=0.01))
    strict = analyse_gallery(rows, thresholds=_thresholds(match_margin=0.95))

    assert lenient.overall_recognise_rate() > strict.overall_recognise_rate(), \
        "tightening the margin did not reduce the accept rate"
    assert not lenient.high_risk
    assert strict.high_risk


# ── Runner ───────────────────────────────────────────────────────────────

def _run():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    width = max(len(n) for n, _ in tests)
    print("=" * (width + 22))
    print("GALLERY DIAGNOSTICS")
    print("=" * (width + 22))
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  {name:<{width}}  PASS")
        except AssertionError as e:
            print(f"  {name:<{width}}  FAIL  {e}")
            failures.append(name)
        except Exception as e:
            print(f"  {name:<{width}}  ERROR {type(e).__name__}: {e}")
            failures.append(name)
    print("=" * (width + 22))
    print(f"  {len(tests) - len(failures)}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
