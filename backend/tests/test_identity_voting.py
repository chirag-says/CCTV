"""
Step 2 -- asymmetric identity voting.

Evidence is not symmetric. Claiming someone IS Rahul and claiming they are NO
LONGER Rahul carry different costs, so they sit behind different bars:

  * promote  -- a few confident matches for the SAME person, spread over time
  * demote   -- only CONFIDENT contradiction; a gray-zone miss never counts,
                and the absence of a face is not evidence at all
  * unknown  -- confirmed by N observations across a span, replacing a 500ms
                timer that one bad angle could satisfy

The last one is also what prevents most stranded unknowns from existing: no
confirmation, no unknown_faces row to strand.
"""

import numpy as np
import pytest

from app.vision.bytetrack import ByteTracker, IdentityState, STrack


def _track(**kw):
    t = STrack(track_id=1, bbox=np.array([0, 0, 10, 10], dtype=np.float32), score=0.9)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


class TestPromotion:
    def test_one_match_is_not_enough(self):
        t = _track()
        t.record_vote("p1", 0.9, now=100.0)
        assert not t.can_promote("p1", need=2, window=8.0, min_span=0.5, now=100.0)

    def test_two_spaced_matches_promote(self):
        t = _track()
        t.record_vote("p1", 0.9, now=100.0)
        t.record_vote("p1", 0.9, now=101.0)
        assert t.can_promote("p1", need=2, window=8.0, min_span=0.5, now=101.0)

    def test_a_burst_from_one_pose_is_one_observation(self):
        """Three consecutive agreeing frames are not three observations."""
        t = _track()
        for i in range(5):
            t.record_vote("p1", 0.9, now=100.0 + i * 0.05)
        assert not t.can_promote("p1", need=2, window=8.0, min_span=0.5, now=100.2)

    def test_votes_for_a_different_person_do_not_promote(self):
        t = _track()
        t.record_vote("p2", 0.9, now=100.0)
        t.record_vote("p2", 0.9, now=101.0)
        assert not t.can_promote("p1", need=2, window=8.0, min_span=0.5, now=101.0)

    def test_stale_votes_fall_out_of_the_window(self):
        t = _track()
        t.record_vote("p1", 0.9, now=100.0)
        t.record_vote("p1", 0.9, now=101.0)
        assert not t.can_promote("p1", need=2, window=8.0, min_span=0.5, now=200.0)


class TestDemotionIsHarder:
    def test_a_single_contradiction_does_not_demote(self):
        t = _track(person_id="p1", identity_state=IdentityState.KNOWN)
        t.record_vote(None, 0.1, now=100.0)
        assert not t.can_demote(need=3, window=10.0, now=100.0)

    def test_enough_confident_contradiction_demotes(self):
        t = _track(person_id="p1", identity_state=IdentityState.KNOWN)
        for i in range(3):
            t.record_vote(None, 0.1, now=100.0 + i)
        assert t.can_demote(need=3, window=10.0, now=102.0)

    def test_gray_zone_misses_never_demote(self):
        """
        The asymmetry, in one test. A borderline miss is the ambiguous middle:
        a usable face the recogniser could not decide on. Ten of them must not
        unseat a verdict that confident matches earned.
        """
        t = _track(person_id="p1", identity_state=IdentityState.KNOWN)
        for i in range(10):
            t.record_vote(None, 0.4, confident=False, now=100.0 + i)
        assert not t.can_demote(need=3, window=20.0, now=110.0)

    def test_a_confident_match_to_someone_else_counts_against(self):
        t = _track(person_id="p1", identity_state=IdentityState.KNOWN)
        for i in range(3):
            t.record_vote("p2", 0.9, now=100.0 + i)
        assert t.can_demote(need=3, window=10.0, now=102.0)

    def test_matches_for_the_held_identity_do_not_count_against(self):
        t = _track(person_id="p1", identity_state=IdentityState.KNOWN)
        for i in range(5):
            t.record_vote("p1", 0.9, now=100.0 + i)
        assert not t.can_demote(need=3, window=10.0, now=104.0)

    def test_absence_of_a_face_is_not_evidence(self):
        """
        Nothing records a vote when no face is bound, so a person who turns
        around can never accumulate evidence against their own identity --
        however long they stay turned away.
        """
        t = _track(person_id="p1", identity_state=IdentityState.KNOWN)
        t.record_vote("p1", 0.9, now=100.0)
        t.record_vote("p1", 0.9, now=101.0)
        # ... 60 seconds of back-turned frames record nothing at all ...
        assert not t.can_demote(need=3, window=10.0, now=161.0)
        assert t.identity_state == IdentityState.KNOWN


class TestUnknownConfirmation:
    def test_one_bad_angle_does_not_mint_an_identity(self):
        """
        The failure the 500ms timer allowed: a single unlucky pose, held
        briefly, minting an unknown_faces row for somebody already enrolled.
        """
        t = _track()
        for i in range(4):
            t.record_vote(None, 0.2, now=100.0 + i * 0.05)
        assert not t.can_confirm_unknown(need=3, window=6.0, min_span=1.0, now=100.2)

    def test_observations_across_a_span_confirm(self):
        t = _track()
        for i in range(3):
            t.record_vote(None, 0.2, now=100.0 + i)
        assert t.can_confirm_unknown(need=3, window=6.0, min_span=1.0, now=102.0)

    def test_gray_zone_misses_still_count_toward_stranger(self):
        """
        Non-confident evidence cannot unseat a known identity, but it is still
        evidence that this face matches nobody.
        """
        t = _track()
        for i in range(3):
            t.record_vote(None, 0.4, confident=False, now=100.0 + i)
        assert t.can_confirm_unknown(need=3, window=6.0, min_span=1.0, now=102.0)

    def test_a_match_interrupts_confirmation(self):
        t = _track()
        t.record_vote(None, 0.2, now=100.0)
        t.record_vote("p1", 0.9, now=101.0)
        t.record_vote(None, 0.2, now=102.0)
        assert not t.can_confirm_unknown(need=3, window=6.0, min_span=1.0, now=102.0)


class TestVoteStorage:
    def test_votes_are_bounded(self):
        t = _track()
        t._votes_max = 10
        for i in range(500):
            t.record_vote("p1", 0.9, now=100.0 + i)
        assert len(t._identity_votes) <= 10

    def test_clear_votes_resets(self):
        t = _track()
        t.record_vote("p1", 0.9, now=100.0)
        t.clear_votes()
        assert t._identity_votes == []

    def test_default_state_is_provisional(self):
        assert _track().identity_state == IdentityState.PROVISIONAL

    def test_a_new_track_from_the_tracker_starts_provisional(self):
        tr = ByteTracker()
        for i in range(4):
            active = tr.update(
                np.array([[100 + i, 100, 160 + i, 160]], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
            )
        assert active[0].identity_state == IdentityState.PROVISIONAL
        assert active[0].unknown_id is None
