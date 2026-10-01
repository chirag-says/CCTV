"""
Step B -- write-time dedup: is this stranger somebody we already have a row for?

TWO THINGS WERE WRONG

1. The threshold. `.env` set UNKNOWN_SIMILARITY_THRESHOLD=0.50 -- STRICTER than
   FACE_MATCH_STRICT (0.52). The system would call two faces the same enrolled
   person at 0.62, then refuse to call two sightings the same stranger unless
   they were within 0.50, on faces that are by definition the ones recognition
   already struggled with. Measured on live data with the 300 most recent
   identities held out against a 6,000-identity corpus:

       threshold   merged into an existing row
         0.50            0.0%
         0.62           55.3%

   That 0.0% is why 81% of strangers had exactly one visit.

2. The comparison. It scored against `unknown_faces.face_encoding` alone -- one
   frozen first view -- and did so by re-querying every pending row and
   deserialising each encoding in a Python loop, measured at 280 ms per face on
   the EventQueue consumer thread.

These tests pin the index's behaviour. The threshold itself is a calibrated
config value, documented where it is defined.
"""

import numpy as np
import pytest

from app.services.unknown_dedup import UnknownDedupIndex


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, seed: int, drift: float) -> np.ndarray:
    """Cosine distance 1 - 1/sqrt(1+drift**2). drift 0.6 -> ~0.14, 1.2 -> ~0.36."""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(base.shape).astype(np.float32)
    n -= n.dot(base) * base
    n /= np.linalg.norm(n)
    v = base + drift * n
    return v / np.linalg.norm(v)


def _index(rows):
    idx = UnknownDedupIndex(refresh_sec=1e9)
    idx._rebuild(rows)
    idx._loaded_at = 1e18  # never auto-refresh during a test
    return idx


class TestMatching:
    def test_a_near_duplicate_matches(self):
        base = _vec(1)
        idx = _index([{"id": "u-1", "encodings": [base]}])
        uid, dist = idx.match(_near(base, 2, 0.3), threshold=0.62)
        assert uid == "u-1"
        assert dist < 0.62

    def test_a_stranger_does_not_match(self):
        idx = _index([{"id": "u-1", "encodings": [_vec(3)]}])
        uid, dist = idx.match(_vec(999), threshold=0.62)
        assert uid is None
        assert dist > 0.62

    def test_an_empty_index_mints(self, monkeypatch):
        # An index with no matrix always attempts a reload, so the loader has
        # to be stubbed or this reaches the real database.
        from app.services import unknown_service
        monkeypatch.setattr(
            unknown_service.UnknownService, "get_pending_encodings",
            staticmethod(lambda limit=None: []),
        )
        idx = _index([])
        assert idx.match(_vec(4), threshold=0.62) == (None, float("inf"))

    def test_zero_vector_is_rejected(self):
        idx = _index([{"id": "u-1", "encodings": [_vec(5)]}])
        assert idx.match(np.zeros(512, dtype=np.float32))[0] is None

    def test_threshold_decides(self):
        """The one knob. 0.50 was the bug; 0.62 matches the recogniser's bar."""
        base = _vec(6)
        probe = _near(base, 7, 1.0)  # ~0.29 distance
        idx = _index([{"id": "u-1", "encodings": [base]}])
        assert idx.match(probe, threshold=0.62)[0] == "u-1"
        assert idx.match(probe, threshold=0.20)[0] is None


class TestBestOfK:
    def test_matches_on_any_stored_view_not_just_the_first(self):
        """
        The structural fix. A person stored frontally, met in profile: the
        profile view is what makes them findable, and it only exists because
        migration 012 persists more than one.
        """
        frontal = _vec(10)
        # drift 3.2 puts the profile ~0.70 from the frontal view -- i.e. beyond
        # the 0.62 threshold, which is the whole point: the frontal shot alone
        # cannot find this person.
        profile = _near(frontal, 11, 3.2)

        one_view = _index([{"id": "u-1", "encodings": [frontal]}])
        all_views = _index([{"id": "u-1", "encodings": [frontal, profile]}])

        probe = _near(profile, 12, 0.2)
        assert one_view.match(probe, threshold=0.62)[0] is None, "the old behaviour"
        assert all_views.match(probe, threshold=0.62)[0] == "u-1"

    def test_nearest_view_wins_across_identities(self):
        a, b = _vec(13), _vec(14)
        idx = _index([
            {"id": "u-a", "encodings": [a]},
            {"id": "u-b", "encodings": [b]},
        ])
        assert idx.match(_near(b, 15, 0.2), threshold=0.62)[0] == "u-b"


class TestLiveUpdates:
    def test_a_row_written_now_is_matchable_immediately(self):
        """
        Otherwise somebody lingering in front of a camera mints a duplicate
        for every sighting until the next periodic reload.
        """
        idx = _index([{"id": "u-1", "encodings": [_vec(20)]}])
        fresh = _vec(21)
        assert idx.match(_near(fresh, 22, 0.2), threshold=0.62)[0] is None

        idx.add("u-new", [fresh])
        assert idx.match(_near(fresh, 23, 0.2), threshold=0.62)[0] == "u-new"

    def test_add_ignores_unusable_vectors(self):
        idx = _index([{"id": "u-1", "encodings": [_vec(24)]}])
        before = idx.stats["views"]
        idx.add("u-2", [np.zeros(512, dtype=np.float32)])
        assert idx.stats["views"] == before

    def test_dropping_an_identity_marks_the_cache_stale(self):
        """
        An enrolled or dismissed row must stop absorbing sightings of a person
        who now has a real identity.
        """
        idx = _index([{"id": "u-1", "encodings": [_vec(25)]}])
        assert idx._loaded_at > 0
        idx.drop("u-1")
        assert idx._loaded_at == 0.0, "next match() reloads"

    def test_dropping_an_unknown_id_is_a_no_op(self):
        idx = _index([{"id": "u-1", "encodings": [_vec(26)]}])
        stamp = idx._loaded_at
        idx.drop("not-in-index")
        assert idx._loaded_at == stamp

    def test_a_full_buffer_forces_a_reload_rather_than_dropping_views(self):
        idx = _index([{"id": "u-1", "encodings": [_vec(27)]}])
        idx._n_rows = idx._matrix.shape[0]      # simulate a full buffer
        idx.add("u-2", [_vec(28)])
        assert idx._loaded_at == 0.0


class TestStats:
    def test_counts_identities_and_views(self):
        base = _vec(30)
        idx = _index([
            {"id": "u-1", "encodings": [base, _near(base, 31, 1.2)]},
            {"id": "u-2", "encodings": [_vec(32)]},
        ])
        s = idx.stats
        assert s["identities"] == 2
        assert s["views"] == 3

    def test_match_and_mint_are_counted_separately(self):
        base = _vec(40)
        idx = _index([{"id": "u-1", "encodings": [base]}])
        idx.match(_near(base, 41, 0.2), threshold=0.62)
        idx.match(_vec(998), threshold=0.62)
        assert idx.stats["matched"] == 1
        assert idx.stats["minted"] == 1
