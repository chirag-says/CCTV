"""
Step 3.5 -- unknown identities survive the process.

THE MEASURED FAILURE
Cross-camera re-identification of unenrolled people was 0.0%: 415 strangers
across 7 live cameras, none matched on two of them. Cross-day was 0.3-0.5%.

The matching logic was fine. `UnknownRegistry` already matched best-of-K over
several views per identity. But `unknown_faces` stores ONE encoding, so every
refresh, restart and second camera rebuilt each identity from its single frozen
first view -- and best-of-K silently became best-of-1.

These tests pin the round trip that fixes it: a view learned in memory reaches
the database, and comes back as a view the next refresh can match on.
"""

import numpy as np
import pytest

from app.vision.unknown_registry import UnknownRegistry


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, seed: int, drift: float) -> np.ndarray:
    """
    A different pose of the same person: `base` pushed `drift` along an
    orthogonal direction, renormalised.

    Cosine distance works out to 1 - 1/sqrt(1 + drift**2), so:
        drift 0.1 -> ~0.005   same pose again
        drift 0.9 -> ~0.26    a distinct view, below learn_min_distance
        drift 1.2 -> ~0.36    LEARNED: past learn_min_distance (0.30) and
                              still inside match_threshold (0.70)
        drift 3.2 -> ~0.70    the edge of being the same person at all
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(base.shape).astype(np.float32)
    noise -= noise.dot(base) * base           # orthogonal component
    noise /= np.linalg.norm(noise)
    v = base + drift * noise
    return v / np.linalg.norm(v)


class TestViewsAreQueuedForPersistence:
    def test_a_learned_view_is_queued(self):
        reg = UnknownRegistry(max_entries=50)
        base = _vec(1)
        uid, _label, is_new = reg.resolve(base)
        assert is_new
        reg.mark_persisted(uid)          # the DB row now exists
        reg.drain_new_views()            # clear the flush from mark_persisted

        # Same person, a pose far enough to be worth keeping.
        reg.resolve(_near(base, 2, drift=1.2))

        queued = reg.drain_new_views()
        assert len(queued) == 1
        assert queued[0][0] == uid

    def test_draining_is_destructive(self):
        reg = UnknownRegistry(max_entries=50)
        base = _vec(3)
        uid, _l, _n = reg.resolve(base)
        reg.mark_persisted(uid)
        reg.drain_new_views()
        reg.resolve(_near(base, 4, drift=1.2))

        assert reg.drain_new_views()
        assert reg.drain_new_views() == [], "a view must not be written twice"

    def test_views_learned_before_the_row_exists_are_not_lost(self):
        """
        An identity minted this frame has no row to attach views to. Those
        views must be flushed once the consumer reports the canonical id,
        or everything learned during the first visit is dropped.
        """
        reg = UnknownRegistry(max_entries=50)
        base = _vec(5)
        uid, _l, _n = reg.resolve(base)
        reg.resolve(_near(base, 6, drift=1.2))   # learned while unpersisted

        assert reg.drain_new_views() == [], "nothing to attach them to yet"

        reg.mark_persisted(uid)
        queued = reg.drain_new_views()
        assert len(queued) >= 1
        assert all(q[0] == uid for q in queued)

    def test_the_canonical_first_view_is_not_re_queued(self):
        """encodings[0] IS unknown_faces.face_encoding, seeded as view #1."""
        reg = UnknownRegistry(max_entries=50)
        uid, _l, _n = reg.resolve(_vec(7))
        reg.mark_persisted(uid)
        assert reg.drain_new_views() == []

    def test_convergence_onto_another_cameras_id_carries_the_views(self):
        reg = UnknownRegistry(max_entries=50)
        base = _vec(8)
        local_uid, _l, _n = reg.resolve(base)
        reg.resolve(_near(base, 9, drift=1.2))

        reg.mark_persisted(local_uid, canonical_id="canonical-id")
        queued = reg.drain_new_views()
        assert queued, "views must follow the identity to its canonical id"
        assert all(q[0] == "canonical-id" for q in queued)


class TestRefreshRestoresEveryView:
    def test_all_views_are_loaded_not_just_the_first(self):
        base = _vec(10)
        profile = _near(base, 11, drift=0.9)

        reg = UnknownRegistry(max_entries=50)
        reg.maybe_refresh(lambda: [{
            "id": "u-1",
            "encodings": [base, profile],
            "occurrence": 4,
        }])

        ident = reg.get("u-1")
        assert ident is not None
        assert ident.view_count == 2, "best-of-K needs every view back"

    def test_a_second_camera_can_match_a_pose_it_never_saw(self):
        """
        The whole point. Camera A saw the person frontally and in profile.
        Camera B loads them from the database and meets them in profile.
        With only the frozen frontal view this returned a NEW identity --
        which is what 0.0% cross-camera looked like.
        """
        base = _vec(12)
        profile = _near(base, 13, drift=0.9)

        camera_b = UnknownRegistry(max_entries=50, match_threshold=0.70)
        camera_b.maybe_refresh(lambda: [{"id": "u-7", "encodings": [base, profile]}])

        uid, _label, is_new = camera_b.resolve(_near(profile, 14, drift=0.1))
        assert not is_new, "should recognise the person, not mint a stranger"
        assert uid == "u-7"

    def test_single_view_identity_still_loads(self):
        """Rows written before 012 seeded, or seeding skipped."""
        reg = UnknownRegistry(max_entries=50)
        reg.maybe_refresh(lambda: [{"id": "u-2", "encodings": [_vec(15)]}])
        assert reg.get("u-2").view_count == 1

    def test_legacy_single_encoding_key_is_accepted(self):
        reg = UnknownRegistry(max_entries=50)
        reg.maybe_refresh(lambda: [{"id": "u-3", "encoding": _vec(16)}])
        assert reg.get("u-3") is not None

    def test_views_are_capped_on_load(self):
        """An identity must not import more rows than max_views."""
        base = _vec(17)
        many = [base] + [_near(base, 20 + i, drift=0.5 + i * 0.05) for i in range(20)]
        reg = UnknownRegistry(max_entries=50, max_views=3)
        reg.maybe_refresh(lambda: [{"id": "u-4", "encodings": many}])
        assert reg.get("u-4").view_count <= 3

    def test_an_identity_with_no_usable_view_is_skipped(self):
        reg = UnknownRegistry(max_entries=50)
        reg.maybe_refresh(lambda: [{"id": "u-5", "encodings": [None]}])
        assert reg.get("u-5") is None


class TestCapacity:
    def test_defaults_come_from_settings_not_a_hardcoded_500(self):
        """
        500 against a pool of 9,624 meant each camera matched against 5% of
        the people it had already seen -- and the shortfall fed itself.
        """
        from app.config import settings

        reg = UnknownRegistry()
        assert reg.max_entries == settings.UNKNOWN_REGISTRY_MAX_ENTRIES
        assert reg.max_views == settings.UNKNOWN_REGISTRY_MAX_VIEWS
        assert reg.max_entries > 500

    def test_explicit_arguments_still_win(self):
        reg = UnknownRegistry(max_entries=7, max_views=2)
        assert reg.max_entries == 7
        assert reg.max_views == 2
