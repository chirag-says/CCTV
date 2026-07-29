"""
Unknown identity stability suite.

Headless: no camera, GPU, or database. Exercises the real UnknownRegistry.

Each test names the user-visible failure it prevents. The behaviour under test
is that an unenrolled person keeps ONE id across frames, restarts, and cameras,
so "who was that person" is answerable from the event log.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.vision.unknown_registry import UnknownRegistry, make_label

DIM = 512


def _unit(v):
    return v / np.linalg.norm(v)


def at_cosine(base, cos_target, rng):
    """Unit vector at exactly `cos_target` cosine similarity to base."""
    n = rng.randn(len(base))
    n -= base * float(np.dot(n, base))
    n = _unit(n)
    return base * cos_target + n * float(np.sqrt(1.0 - cos_target ** 2))


def _rng(seed=5):
    return np.random.RandomState(seed)


# ── The core guarantee ───────────────────────────────────────────────────

def test_same_person_keeps_one_id_across_sightings():
    """PREVENTS: the same stranger becoming Unknown-a1, Unknown-b2, Unknown-c3."""
    rng = _rng()
    base = _unit(rng.randn(DIM))
    reg = UnknownRegistry(match_threshold=0.70)

    first_id, first_label, is_new = reg.resolve(base)
    assert is_new and first_id

    for cos in (0.99, 0.95, 0.90, 0.85):
        uid, label, new = reg.resolve(at_cosine(base, cos, rng))
        assert not new, f"minted a second id at cosine {cos}"
        assert uid == first_id
        assert label == first_label


def test_different_people_get_different_ids():
    """PREVENTS: two strangers collapsing into one identity."""
    rng = _rng()
    reg = UnknownRegistry(match_threshold=0.70)
    a, _, _ = reg.resolve(_unit(rng.randn(DIM)))
    b, _, new = reg.resolve(_unit(rng.randn(DIM)))
    assert new and a != b


def test_id_survives_restart_via_db_refresh():
    """
    PREVENTS: labels resetting to #1 on restart.

    Simulates a process restart: a brand-new registry seeded from the DB must
    resolve the same person to the id the previous process assigned.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))

    before = UnknownRegistry()
    original_id, original_label, _ = before.resolve(base)

    # Restart: fresh registry, seeded from what the DB persisted.
    after = UnknownRegistry(refresh_interval=0.0)
    after.maybe_refresh(lambda: [{
        "id": original_id, "encoding": base,
        "occurrence": 3, "first_seen": 0.0, "last_seen": 0.0,
    }])

    uid, label, is_new = after.resolve(at_cosine(base, 0.93, rng))
    assert not is_new, "restart minted a new id for a known person"
    assert uid == original_id
    assert label == original_label


def test_label_is_stable_and_derived_from_id():
    """PREVENTS: camera A's 'Unknown #1' being a different person to camera B's."""
    rng = _rng()
    base = _unit(rng.randn(DIM))
    seed = [{"id": "abc123def456", "encoding": base,
             "occurrence": 1, "first_seen": 0.0, "last_seen": 0.0}]

    cam_a = UnknownRegistry(refresh_interval=0.0)
    cam_b = UnknownRegistry(refresh_interval=0.0)
    cam_a.maybe_refresh(lambda: seed)
    cam_b.maybe_refresh(lambda: seed)

    id_a, label_a, _ = cam_a.resolve(at_cosine(base, 0.95, rng))
    id_b, label_b, _ = cam_b.resolve(at_cosine(base, 0.92, rng))

    assert id_a == id_b == "abc123def456"
    assert label_a == label_b == make_label("abc123def456")


def test_convergence_when_another_camera_won_the_race():
    """
    PREVENTS: two cameras seeing a stranger simultaneously and keeping two ids.

    Both mint locally; the consumer reports the canonical row id back and the
    loser re-keys onto it.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    reg = UnknownRegistry()

    local_id, _, _ = reg.resolve(base)
    canonical = "canonical-row-id"
    reg.mark_persisted(local_id, canonical_id=canonical)

    uid, label, is_new = reg.resolve(at_cosine(base, 0.96, rng))
    assert not is_new
    assert uid == canonical, "did not converge on the canonical id"
    assert label == make_label(canonical)
    assert reg.get(local_id) is None, "stale local id still resident"


# ── Throttling and bounds ────────────────────────────────────────────────

def test_sighting_reports_are_throttled():
    """
    PREVENTS: `occurrence` counting frames instead of visits, and the
    EventQueue being flooded with one event per frame per unknown.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    reg = UnknownRegistry(sighting_cooldown=60.0)
    uid, _, _ = reg.resolve(base)

    assert reg.should_report_sighting(uid) is True
    for _ in range(20):
        assert reg.should_report_sighting(uid) is False, "throttle leaked"


def test_registry_is_bounded():
    """
    PREVENTS: the unbounded-growth defect the old _unknown_cache had, where
    both memory and per-face scan cost grew for the camera's whole lifetime.
    """
    rng = _rng()
    reg = UnknownRegistry(max_entries=25)
    for _ in range(200):
        uid, _, _ = reg.resolve(_unit(rng.randn(DIM)))
        reg.mark_persisted(uid)          # only persisted entries are evictable
    assert reg.size <= 25, f"registry grew to {reg.size}"


def test_unpersisted_entries_are_not_evicted():
    """PREVENTS: losing an identity before its DB row is written."""
    rng = _rng()
    reg = UnknownRegistry(max_entries=5)
    ids = [reg.resolve(_unit(rng.randn(DIM)))[0] for _ in range(20)]
    assert all(reg.get(i) is not None for i in ids), "dropped an unpersisted identity"


def test_zero_norm_embedding_is_rejected():
    reg = UnknownRegistry()
    uid, label, is_new = reg.resolve(np.zeros(DIM, dtype=np.float32))
    assert uid is None and label is None and is_new is False


# ── Runner ───────────────────────────────────────────────────────────────

def _run():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    width = max(len(n) for n, _ in tests)
    print("=" * (width + 22))
    print("UNKNOWN IDENTITY STABILITY")
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
