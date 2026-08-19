"""
Unknown identity stability suite.

Headless: no camera, GPU, or database. Exercises the real UnknownRegistry.

Each test names the user-visible failure it prevents. The behaviour under test
is that an unenrolled person keeps ONE id across frames, restarts, and cameras,
so "who was that person" is answerable from the event log.
"""

import os
import sys
import time

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


# ── Re-identification: one person, one identity, across poses ────────────

def test_pose_change_does_not_mint_a_second_identity():
    """
    PREVENTS: the duplicate-unknown defect measured on the live DB, where 4,129
    of 4,452 pending unknowns sat at occurrence=1. A single frozen embedding
    could not match the same person from a different angle, so the registry
    minted a new id and the DB dedup -- comparing against that same frozen
    vector -- agreed it was a new person.
    """
    rng = _rng()
    frontal = _unit(rng.randn(DIM))
    reg = UnknownRegistry(match_threshold=0.70, max_views=5, learn_min_distance=0.30)

    first_id, _, is_new = reg.resolve(frontal)
    assert is_new

    # Turning away: each view is reachable from the last, but the final one is
    # beyond the match threshold from the original frontal capture.
    profile = at_cosine(frontal, 0.55, rng)          # distance 0.45 from frontal
    far_profile = _unit(frontal * 0.15 + profile * 0.95)

    uid, _, is_new = reg.resolve(profile)
    assert not is_new and uid == first_id, "profile view minted a second identity"

    uid, _, is_new = reg.resolve(far_profile)
    assert not is_new and uid == first_id, (
        "third view minted a second identity -- the learned profile view "
        "should have bridged the gap from frontal"
    )

    assert reg.get(first_id).view_count >= 2, "no view was learned"


def test_redundant_views_are_not_stored():
    """
    PREVENTS: a person loitering in front of a camera filling every view slot
    with the same angle, which costs search time and covers nothing new.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    reg = UnknownRegistry(max_views=5, learn_min_distance=0.30)

    uid, _, _ = reg.resolve(base)
    for _ in range(50):
        reg.resolve(at_cosine(base, 0.99, rng))      # distance 0.01 — same pose

    assert reg.get(uid).view_count == 1, (
        f"stored {reg.get(uid).view_count} near-identical views"
    )


def test_views_per_identity_are_capped():
    """PREVENTS: unbounded memory per identity over a long run."""
    rng = _rng()
    base = _unit(rng.randn(DIM))
    reg = UnknownRegistry(match_threshold=0.70, max_views=3, learn_min_distance=0.10)

    uid, _, _ = reg.resolve(base)
    for cos in (0.85, 0.80, 0.75, 0.72, 0.70, 0.68):
        reg.resolve(at_cosine(base, cos, rng))

    assert reg.get(uid).view_count <= 3, reg.get(uid).view_count


def test_distinct_people_still_separate_after_learning_views():
    """
    PREVENTS: re-identification over-merging. Learning extra views widens what
    an identity matches, so the guarantee that two strangers stay distinct has
    to be re-established, not assumed.
    """
    rng = _rng()
    a_base = _unit(rng.randn(DIM))
    b_base = _unit(rng.randn(DIM))
    reg = UnknownRegistry(match_threshold=0.70, max_views=5, learn_min_distance=0.30)

    a_id, _, _ = reg.resolve(a_base)
    for cos in (0.60, 0.55, 0.50):
        reg.resolve(at_cosine(a_base, cos, rng))

    b_id, _, is_new = reg.resolve(b_base)
    assert is_new and b_id != a_id, "second stranger absorbed into the first identity"

    # And A must still resolve to A, not to the newcomer.
    again, _, _ = reg.resolve(at_cosine(a_base, 0.95, rng))
    assert again == a_id


# ── Persistence feedback ─────────────────────────────────────────────────

def test_canonical_id_from_another_thread_is_applied_on_drain():
    """
    PREVENTS: the convergence path staying dead. The consumer thread learns the
    canonical row id but must not touch the registry's structures, so it queues
    the result and the camera thread applies it.
    """
    rng = _rng()
    base = _unit(rng.randn(DIM))
    reg = UnknownRegistry()

    local_id, _, _ = reg.resolve(base)
    reg.submit_persistence_result(local_id, "canonical-row-id")

    # Nothing applied until the owning thread drains.
    assert reg.get(local_id) is not None
    assert reg.drain_persistence_results() == 1

    uid, label, is_new = reg.resolve(at_cosine(base, 0.96, rng))
    assert not is_new and uid == "canonical-row-id"
    assert label == make_label("canonical-row-id")


def test_converging_onto_a_resident_id_keeps_the_views():
    """
    PREVENTS: losing pose coverage on convergence. Both ids are the same person,
    so the loser's views are exactly the extra angles the survivor wants.
    """
    rng = _rng()
    a_base = _unit(rng.randn(DIM))
    reg = UnknownRegistry(match_threshold=0.40, max_views=5, learn_min_distance=0.10)

    winner, _, _ = reg.resolve(a_base)
    loser, _, is_new = reg.resolve(at_cosine(a_base, 0.30, rng))
    assert is_new and loser != winner

    reg.mark_persisted(loser, canonical_id=winner)

    assert reg.get(loser) is None
    assert reg.get(winner).view_count >= 2, "views discarded on convergence"


def test_persisted_flag_is_actually_set_in_the_normal_path():
    """
    PREVENTS: the eviction cap becoming unenforceable. _evict() only removes
    persisted entries, so if nothing is ever marked persisted the registry
    grows for the life of the camera regardless of max_entries.
    """
    rng = _rng()
    reg = UnknownRegistry(max_entries=10)
    uid, _, _ = reg.resolve(_unit(rng.randn(DIM)))

    assert reg.get(uid).persisted is False
    reg.submit_persistence_result(uid, uid)      # same id: a plain insert
    reg.drain_persistence_results()
    assert reg.get(uid).persisted is True


def test_stale_unpersisted_entries_eventually_become_evictable():
    """
    PREVENTS: a stalled consumer turning eviction protection into a memory leak.
    Protection that never expires is not protection.
    """
    rng = _rng()
    reg = UnknownRegistry(max_entries=5, sighting_cooldown=0.0)
    for _ in range(30):
        reg.resolve(_unit(rng.randn(DIM)))

    # Nothing was ever persisted; age every entry past the grace period.
    stale = time.time() - 10_000
    for ident in reg._identities.values():
        ident.first_seen = stale
        ident.last_seen = stale

    reg.resolve(_unit(rng.randn(DIM)))           # triggers _evict()
    assert reg.size <= 5 + 1, f"registry stuck at {reg.size} with a dead consumer"


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
