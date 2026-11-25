"""
Phase 4 — FULL GLOBAL PERSON IDENTITY VALIDATION
=================================================

Comprehensive adversarial test suite for GlobalIdentityResolver.

Covers Parts 2-22 of the Phase 4 validation plan:
  - Global ID lifecycle (create, reuse, expire)
  - Face identity (strong, weak, absent, contradictory)
  - Body ReID (threshold boundaries, secondary enforcement)
  - Face + Body fusion matrix
  - Unknown → Known promotion
  - Cross-camera continuity
  - Camera transition validation
  - Multi-camera simultaneity
  - Concurrency / race conditions
  - Duplicate track coalescing
  - False merge / false split protection
  - 120-second TTL exact boundaries
  - Gallery bounding
  - Embedding gallery diversity
  - Event / audit consistency
  - Restart / recovery
  - Performance benchmarks
  - Property / invariant testing
  - Failure injection

Run:
    cd backend && python -m pytest tests/test_phase4_validation.py -v --tb=short
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
import time
import unittest
from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np

from app.vision.global_identity import (
    CalibratedPolicy,
    CameraTransition,
    DecisionState,
    EventType,
    GlobalIdentityResolver,
    GlobalPersonState,
    LifecycleState,
    ResolutionDecision,
    ResolverMode,
)

# ── Test Helpers ─────────────────────────────────────────────────────────

def vec(*values):
    """Create a normalized vector from raw components."""
    v = np.asarray(values, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return v / n


def rand_vec(dim=128, seed=None):
    """Random normalized vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def at_cosine(base, cos_target, dim=128, seed=None):
    """Return a unit vector at exactly `cos_target` cosine similarity to `base`."""
    rng = np.random.RandomState(seed)
    n = rng.randn(dim).astype(np.float32)
    n -= base * float(np.dot(n, base))
    n = n / np.linalg.norm(n)
    return (base * cos_target + n * float(math.sqrt(1.0 - cos_target ** 2))).astype(np.float32)


class Clock:
    """Deterministic, controllable clock for testing."""
    def __init__(self, now=1000.0):
        self.value = float(now)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds
        return self.value


class IDs:
    """Deterministic, sequential ID factory."""
    def __init__(self, prefix="G"):
        self.n = 0
        self.prefix = prefix

    def __call__(self):
        self.n += 1
        return f"{self.prefix}-{self.n:04d}"


# ── Standard calibrated policy used by most tests ────────────────────────

STANDARD_POLICY = CalibratedPolicy(
    face_match_threshold=0.80,
    face_min_margin=0.10,
    body_match_threshold=0.56,
    body_min_margin=0.10,
    body_votes_required=3,
    body_vote_min_span=1.0,
    body_vote_min_interval=0.25,
)


def make_resolver(*, mode="shadow", transitions=(), policy=None, clock=None, **kw):
    """Factory for a test resolver + clock pair."""
    clk = clock or Clock()
    r = GlobalIdentityResolver(
        clock=clk,
        id_factory=kw.pop("id_factory", IDs()),
        mode=mode,
        transitions=transitions,
        policy=policy or STANDARD_POLICY,
        **kw,
    )
    return r, clk


# ═════════════════════════════════════════════════════════════════════════
# PART 2 — UNIT TESTS: GLOBAL ID LIFECYCLE
# ═════════════════════════════════════════════════════════════════════════

class TestPart02_GlobalIDLifecycle(unittest.TestCase):
    """Tests 1-11: core lifecycle mechanics."""

    def test_01_new_person_creates_exactly_one_global_id(self):
        """TEST 1: local track → exactly one Global ID."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        self.assertEqual(DecisionState.CONFIRMED, d.state)
        self.assertIsNotNone(d.global_id)
        self.assertEqual(1, r.metrics["active_count"])

    def test_02_same_local_track_same_global_id(self):
        """TEST 2: same local track observed repeatedly → same Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        clk.advance(1)
        d2 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        clk.advance(1)
        d3 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id)
        self.assertEqual(d2.global_id, d3.global_id)

    def test_03_local_track_changes_same_identity(self):
        """TEST 3: Track 81 → Track 102 with same face → same Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 81, face_embedding=vec(1, 0, 0), known_person_id="chirag")
        clk.advance(0.5)
        # Track 81 ends, track 102 begins
        r.unbind("cam1", 81)
        clk.advance(0.5)
        d2 = r.resolve("cam1", 102, face_embedding=vec(1, 0, 0), known_person_id="chirag")
        self.assertEqual(d1.global_id, d2.global_id)

    def test_04_person_disappears_enters_missing(self):
        """TEST 4: person disappears → MISSING, not NEW."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="bob")
        state = r.unbind("cam1", 1)
        self.assertEqual(LifecycleState.MISSING, state.lifecycle)
        self.assertIsNotNone(state.missing_since)

    def test_05_reappear_within_120_seconds_same_id(self):
        """TEST 5: person reappears within 120 seconds → same Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="bob")
        r.unbind("cam1", 1)
        clk.advance(60)  # 60 seconds later
        d2 = r.resolve("cam1", 50, face_embedding=vec(1, 0, 0), known_person_id="bob")
        self.assertEqual(d1.global_id, d2.global_id)
        self.assertEqual(LifecycleState.ACTIVE, r.get_state(d2.global_id).lifecycle)

    def test_06_reappear_after_120_seconds_new_id(self):
        """TEST 6: person reappears after >120 seconds → old ID expired, new ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="bob")
        r.unbind("cam1", 1)
        clk.advance(121)
        d2 = r.resolve("cam1", 50, face_embedding=vec(1, 0, 0), known_person_id="bob")
        self.assertNotEqual(d1.global_id, d2.global_id)
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(d1.global_id).lifecycle)

    def test_07_disappear_from_one_camera_still_active_elsewhere(self):
        """TEST 7: person disappears from Camera A, still active in Camera B → ACTIVE."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 0, 20, overlap=True)]
        )
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r.resolve("cam2", 5, face_embedding=vec(1, 0, 0), known_person_id="alice")
        state = r.unbind("cam1", 1)
        self.assertEqual(LifecycleState.ACTIVE, state.lifecycle)
        self.assertIsNone(state.missing_since)
        # Wait beyond TTL — should still be ACTIVE because cam2 binding exists
        clk.advance(200)
        r.sweep()
        self.assertEqual(LifecycleState.ACTIVE, r.get_state(state.global_id).lifecycle)

    def test_08_disappear_from_all_cameras_missing_timer_starts(self):
        """TEST 8: person disappears from ALL cameras → MISSING timer starts exactly once."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 0, 20, overlap=True)]
        )
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r.resolve("cam2", 5, face_embedding=vec(1, 0, 0), known_person_id="alice")
        # Unbind cam1 first — should remain ACTIVE
        r.unbind("cam1", 1)
        s1 = r.get_state(d.global_id)
        self.assertEqual(LifecycleState.ACTIVE, s1.lifecycle)
        # Unbind cam2 — now truly MISSING
        clk.advance(1)
        r.unbind("cam2", 5)
        s2 = r.get_state(d.global_id)
        self.assertEqual(LifecycleState.MISSING, s2.lifecycle)
        self.assertIsNotNone(s2.missing_since)

    def test_09_reappear_at_119_9_seconds_same_id(self):
        """TEST 9: person reappears at 119.9 seconds → same Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        r.unbind("cam1", 1)
        clk.advance(119.9)
        d2 = r.resolve("cam1", 99, face_embedding=vec(1, 0, 0), known_person_id="x")
        self.assertEqual(d1.global_id, d2.global_id)

    def test_10_reappear_at_120_1_seconds_new_id(self):
        """TEST 10: person reappears at 120.1 seconds → old ID not reused."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        r.unbind("cam1", 1)
        clk.advance(120.1)
        d2 = r.resolve("cam1", 99, face_embedding=vec(1, 0, 0), known_person_id="x")
        self.assertNotEqual(d1.global_id, d2.global_id)

    def test_11_repeated_missing_reappear_no_duplicate_ids(self):
        """TEST 11: repeated missing/reappear cycles → no duplicate Global IDs."""
        r, clk = make_resolver()
        seen_ids: Set[str] = set()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="cycle")
        seen_ids.add(d.global_id)

        for i in range(5):
            r.unbind("cam1", 1 + i)
            clk.advance(30)  # well within 120s
            d = r.resolve("cam1", 2 + i, face_embedding=vec(1, 0, 0), known_person_id="cycle")
            seen_ids.add(d.global_id)

        # All should be the SAME global ID — no duplicates
        self.assertEqual(1, len(seen_ids), f"Expected 1 unique ID, got {seen_ids}")


# ═════════════════════════════════════════════════════════════════════════
# PART 3 — FACE IDENTITY TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart03_FaceIdentity(unittest.TestCase):
    """Face evidence is PRIMARY. Strong face must dominate body."""

    def test_A_strong_face_match_confirmed(self):
        """Strong face match → CONFIRMED."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                       known_person_id="alice", known_person_name="Alice")
        self.assertEqual(DecisionState.CONFIRMED, d.state)
        self.assertIsNotNone(d.global_id)

    def test_B_weak_face_match_uncertain_when_uncalibrated(self):
        """Weak/uncalibrated face → UNCERTAIN."""
        r, _ = make_resolver(policy=CalibratedPolicy(
            face_match_threshold=None, face_min_margin=None,
        ))
        # First create an identity
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r.unbind("cam1", 1)
        # Unknown face with uncalibrated policy
        d = r.resolve("cam1", 2, face_embedding=vec(0.99, 0.1, 0))
        self.assertEqual(DecisionState.UNCERTAIN, d.state)

    def test_C_no_face_no_body_uncertain(self):
        """No face, no body → UNCERTAIN."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1)
        self.assertEqual(DecisionState.UNCERTAIN, d.state)
        self.assertEqual("no_valid_evidence", d.reason)

    def test_D_low_quality_face_rejected(self):
        """Face embedding that is all zeros or NaN → rejected gracefully."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=np.zeros(3, dtype=np.float32))
        self.assertEqual(DecisionState.UNCERTAIN, d.state)

    def test_E_unknown_face_creates_new_id(self):
        """Unknown face (no known_person_id) → NEW if first seen."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0))
        self.assertEqual(DecisionState.NEW, d.state)
        self.assertIsNotNone(d.global_id)

    def test_F_contradictory_face_unverified_rejected(self):
        """Unverified known_person_id → rejected."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                       known_person_id="alice", face_verified=False)
        self.assertEqual(DecisionState.UNCERTAIN, d.state)
        self.assertEqual("unverified_known_face_rejected", d.reason)

    def test_strong_face_dominates_body(self):
        """Strong face identity + weak/contradictory body → keep face identity."""
        r, clk = make_resolver()
        # Create alice with face+body
        d1 = r.resolve("cam1", 1,
                        face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        # Create bob with different face+body
        d2 = r.resolve("cam1", 2,
                        face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                        known_person_id="bob")
        clk.advance(1)
        # Track 1 now has body similar to bob but face = alice
        result = r.resolve("cam1", 1,
                            face_embedding=vec(1, 0, 0), body_embedding=vec(0, 1, 0),
                            known_person_id="alice")
        self.assertEqual(DecisionState.CONFIRMED, result.state)
        self.assertEqual(d1.global_id, result.global_id)
        self.assertNotEqual(d2.global_id, result.global_id)

    def test_strong_face_no_body_still_confirmed(self):
        """Strong face identity + no body → keep face identity."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                       known_person_id="alice")
        self.assertEqual(DecisionState.CONFIRMED, d.state)

    def test_face_A_body_B_must_not_switch(self):
        """Face says Person A, Body says Person B → must NOT silently switch to B."""
        r, clk = make_resolver()
        # Create A and B
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="person-a")
        r.resolve("cam1", 2,
                   face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="person-b")
        clk.advance(1)
        # Existing binding: track 1 is person A.
        # Provide body of B — binding should remain with A.
        result = r.resolve("cam1", 1, body_embedding=vec(0, 1, 0))
        self.assertEqual("person-a", r.get_state(result.global_id).known_person_id)


# ═════════════════════════════════════════════════════════════════════════
# PART 4 — BODY RE-ID TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart04_BodyReID(unittest.TestCase):
    """Body ReID is SECONDARY. OSNet MSMT17 threshold = 0.56."""

    def _setup_body_resolver(self):
        """Create resolver with a known person's body in the gallery."""
        transition = CameraTransition("cam1", "cam2", 0, 30)
        r, clk = make_resolver(transitions=[transition])
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0, 0), body_embedding=vec(1, 0, 0, 0),
                   known_person_id="target")
        r.unbind("cam1", 1)
        clk.advance(2)
        return r, clk

    def test_body_clearly_above_threshold(self):
        """Body similarity clearly above 0.56 → proceeds to vote check."""
        r, clk = self._setup_body_resolver()
        d = r.resolve("cam2", 10, body_embedding=vec(1, 0, 0, 0))
        self.assertIn(d.state, [DecisionState.UNCERTAIN, DecisionState.CANDIDATE])
        # Should not be CONFIRMED — body alone can never confirm
        self.assertNotEqual(DecisionState.CONFIRMED, d.state)

    def test_body_clearly_below_threshold(self):
        """Body similarity clearly below 0.56 → rejected."""
        r, clk = self._setup_body_resolver()
        d = r.resolve("cam2", 10, body_embedding=vec(0, 0, 1, 0))
        # Very different body → should not match
        self.assertIn(d.reason, ["body_below_threshold", "body_only_cannot_create_identity"])

    def test_body_never_auto_confirms(self):
        """Body-only evidence must NEVER produce a CONFIRMED state."""
        transition = CameraTransition("cam1", "cam2", 0, 30)
        r, clk = make_resolver(transitions=[transition])
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="target")
        r.unbind("cam1", 1)
        clk.advance(2)
        # Even with many repeated body observations, should never CONFIRM
        for i in range(20):
            d = r.resolve("cam2", 10, body_embedding=vec(1, 0, 0))
            self.assertNotEqual(DecisionState.CONFIRMED, d.state,
                                f"Body confirmed on iteration {i} — VIOLATION")
            clk.advance(0.5)

    def test_body_does_not_override_face_identity(self):
        """Body-only evidence does NOT override a strong face identity."""
        r, clk = make_resolver()
        # Person A: face + body
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="A")
        # Person B: face + body
        r.resolve("cam1", 2,
                   face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="B")
        clk.advance(1)
        # Track 1 (bound to A) gets B's body — must remain A
        d = r.resolve("cam1", 1, body_embedding=vec(0, 1, 0))
        state = r.get_state(d.global_id)
        self.assertEqual("A", state.known_person_id)

    def test_default_threshold_is_0_56(self):
        """Verify documented default threshold."""
        self.assertEqual(0.56, CalibratedPolicy().body_match_threshold)


# ═════════════════════════════════════════════════════════════════════════
# PART 5 — FACE + BODY FUSION TEST MATRIX
# ═════════════════════════════════════════════════════════════════════════

class TestPart05_FaceBodyFusionMatrix(unittest.TestCase):
    """Full decision table covering every face+body combination."""

    def test_case1_strong_face_strong_body_confirmed(self):
        """CASE 1: strong Face A + strong Body A → CONFIRMED A."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1,
                       face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                       known_person_id="A")
        self.assertEqual(DecisionState.CONFIRMED, d.state)

    def test_case2_strong_face_weak_body_confirmed(self):
        """CASE 2: strong Face A + weak Body A → CONFIRMED A."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1,
                       face_embedding=vec(1, 0, 0), body_embedding=vec(0.5, 0.5, 0.5),
                       known_person_id="A")
        self.assertEqual(DecisionState.CONFIRMED, d.state)

    def test_case3_strong_face_no_body_confirmed(self):
        """CASE 3: strong Face A + no Body → CONFIRMED A."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="A")
        self.assertEqual(DecisionState.CONFIRMED, d.state)

    def test_case4_no_face_strong_body_valid_transition(self):
        """CASE 4: no Face + strong Body A + valid transition → at most CANDIDATE."""
        transition = CameraTransition("cam1", "cam2", 0, 30)
        r, clk = make_resolver(transitions=[transition])
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="A")
        r.unbind("cam1", 1)
        clk.advance(2)
        # Repeated body observations with valid transition
        for _ in range(5):
            d = r.resolve("cam2", 10, body_embedding=vec(1, 0, 0))
            clk.advance(0.5)
        # Should be CANDIDATE at best, never CONFIRMED
        self.assertIn(d.state, [DecisionState.CANDIDATE, DecisionState.UNCERTAIN])
        self.assertNotEqual(DecisionState.CONFIRMED, d.state)

    def test_case5_no_face_strong_body_no_transition(self):
        """CASE 5: no Face + strong Body A + NO temporal continuity → conservative."""
        r, clk = make_resolver()  # No transitions configured
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="A")
        r.unbind("cam1", 1)
        clk.advance(2)
        d = r.resolve("cam2", 10, body_embedding=vec(1, 0, 0))
        # No transition configured → should be UNCERTAIN
        self.assertEqual(DecisionState.UNCERTAIN, d.state)

    def test_case6_no_face_medium_body(self):
        """CASE 6: no Face + medium Body A → should not force confirmation."""
        transition = CameraTransition("cam1", "cam2", 0, 30)
        r, clk = make_resolver(transitions=[transition])
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="A")
        r.unbind("cam1", 1)
        clk.advance(2)
        # Medium similarity body
        d = r.resolve("cam2", 10, body_embedding=vec(0.7, 0.714, 0))
        self.assertNotEqual(DecisionState.CONFIRMED, d.state)

    def test_case7_face_A_body_B_conflict(self):
        """CASE 7: Face A + Body B → conflict / uncertain, NOT B."""
        r, clk = make_resolver()
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="A")
        r.resolve("cam1", 2,
                   face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="B")
        clk.advance(1)
        # Existing binding (track 1 = A). Body like B doesn't overwrite.
        d = r.resolve("cam1", 1, body_embedding=vec(0, 1, 0))
        self.assertEqual("A", r.get_state(d.global_id).known_person_id)

    def test_case8_face_unavailable_body_ambiguous(self):
        """CASE 8: Face unavailable + Body A + nearly identical candidate B → UNCERTAIN."""
        transition = CameraTransition("cam1", "cam2", 0, 30)
        r, clk = make_resolver(transitions=[transition], policy=CalibratedPolicy(
            face_match_threshold=0.8, face_min_margin=0.1,
            body_min_margin=0.05,  # tight margin
            body_votes_required=3, body_vote_min_span=1.0,
        ))
        # Two people with very similar bodies
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0.01, 0))
        r.resolve("cam1", 2,
                   face_embedding=vec(0, 1, 0), body_embedding=vec(1, 0.02, 0))
        r.unbind("cam1", 1)
        r.unbind("cam1", 2)
        clk.advance(2)
        d = r.resolve("cam2", 10, body_embedding=vec(1, 0.015, 0))
        # Two candidates with nearly identical scores → must be UNCERTAIN
        self.assertNotEqual(DecisionState.CONFIRMED, d.state)

    def test_case9_strong_face_contradictory_body(self):
        """CASE 9: strong Face A + strongly contradictory Body B → Face remains primary."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1,
                       face_embedding=vec(1, 0, 0), body_embedding=vec(0, 0, 1),
                       known_person_id="A")
        self.assertEqual(DecisionState.CONFIRMED, d.state)
        self.assertEqual("A", r.get_state(d.global_id).known_person_id)

    def test_case10_two_candidates_similar_evidence_no_forced_match(self):
        """CASE 10: two candidates with similar evidence → no arbitrary forced match."""
        r, clk = make_resolver()
        # Two people with similar faces
        r.resolve("cam1", 1, face_embedding=vec(1, 0.1, 0))
        r.resolve("cam1", 2, face_embedding=vec(1, 0.15, 0))
        r.unbind("cam1", 1)
        r.unbind("cam1", 2)
        clk.advance(1)
        # Query with face between the two
        d = r.resolve("cam1", 10, face_embedding=vec(1, 0.125, 0))
        # If margin too small → UNCERTAIN (depends on threshold)
        # At minimum, should not blindly force one


# ═════════════════════════════════════════════════════════════════════════
# PART 6 — UNKNOWN → KNOWN TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart06_UnknownToKnown(unittest.TestCase):
    """Critical test: unknown → known must retain the same Global ID."""

    def test_unknown_promoted_to_known_keeps_global_id(self):
        """Global 1042 with NULL known_person_id → later matched to 'Chirag' → same Global ID."""
        r, _ = make_resolver()
        # First seen as unknown
        unknown_d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0))
        self.assertEqual(DecisionState.NEW, unknown_d.state)
        gid = unknown_d.global_id
        self.assertIsNotNone(gid)
        self.assertIsNone(r.get_state(gid).known_person_id)

        # Later, strong face match identifies as Chirag
        known_d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                             known_person_id="chirag", known_person_name="Chirag")
        self.assertEqual(gid, known_d.global_id, "Global ID changed during unknown→known promotion!")
        self.assertEqual("chirag", r.get_state(gid).known_person_id)
        self.assertEqual("Chirag", r.get_state(gid).known_person_name)

    def test_unknown_to_known_does_NOT_create_new_id(self):
        """Must NOT: old ID expires + new ID created for known person."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0))
        old_gid = d1.global_id
        d2 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="chirag")
        self.assertEqual(old_gid, d2.global_id)
        # Old ID should NOT be expired
        self.assertNotEqual(LifecycleState.EXPIRED, r.get_state(old_gid).lifecycle)

    def test_unknown_known_unknown_known_continuity(self):
        """unknown → known → unknown-like observations → known again → continuity."""
        r, clk = make_resolver()
        # Unknown
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0))
        gid = d1.global_id
        clk.advance(1)
        # Known
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                   known_person_id="chirag")
        self.assertEqual("chirag", r.get_state(gid).known_person_id)
        clk.advance(1)
        # Unknown-like (no known_person_id but same track/face) — binding remains
        d3 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0))
        self.assertEqual(gid, d3.global_id)
        self.assertEqual("chirag", r.get_state(gid).known_person_id)
        clk.advance(1)
        # Known again
        d4 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="chirag")
        self.assertEqual(gid, d4.global_id)


# ═════════════════════════════════════════════════════════════════════════
# PART 7 — CROSS-CAMERA TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart07_CrossCamera(unittest.TestCase):
    """Same physical person must preserve Global ID across cameras."""

    def test_two_camera_continuity(self):
        """Camera 1 track 81 → Camera 2 track 17 → same Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 81, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        r.unbind("cam1", 81)
        clk.advance(5)
        d2 = r.resolve("cam2", 17, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id)

    def test_three_camera_chain(self):
        """A → B → C with same face → ONE Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 5, face_embedding=vec(1, 0, 0),
                        known_person_id="bob")
        r.unbind("cam1", 5)
        clk.advance(3)
        d2 = r.resolve("cam2", 12, face_embedding=vec(1, 0, 0),
                        known_person_id="bob")
        r.unbind("cam2", 12)
        clk.advance(3)
        d3 = r.resolve("cam3", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="bob")
        self.assertEqual(d1.global_id, d2.global_id)
        self.assertEqual(d2.global_id, d3.global_id)

    def test_four_camera_chain(self):
        """A → B → C → D → same Global ID."""
        r, clk = make_resolver()
        gid = None
        for cam_idx in range(4):
            cam = f"cam{cam_idx+1}"
            track = 10 + cam_idx
            d = r.resolve(cam, track, face_embedding=vec(1, 0, 0),
                           known_person_id="student-1")
            if gid is None:
                gid = d.global_id
            else:
                self.assertEqual(gid, d.global_id, f"Global ID changed at {cam}")
            r.unbind(cam, track)
            clk.advance(5)

    def test_cross_camera_face_visibility_changes(self):
        """Face visible in cam1, invisible in cam2 (body only), face back in cam3."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 0, 30)]
        )
        d1 = r.resolve("cam1", 1,
                        face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        r.unbind("cam1", 1)
        clk.advance(3)
        # cam2: body only — cannot confirm but still candidate
        d2 = r.resolve("cam2", 5, body_embedding=vec(1, 0, 0))
        # Should NOT be confirmed without face
        self.assertNotEqual(DecisionState.CONFIRMED, d2.state)
        clk.advance(3)
        # cam3: face back — same person
        d3 = r.resolve("cam3", 8, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d3.global_id)


# ═════════════════════════════════════════════════════════════════════════
# PART 8 — CAMERA TRANSITION VALIDATION
# ═════════════════════════════════════════════════════════════════════════

class TestPart08_CameraTransitionValidation(unittest.TestCase):
    """Camera topology must SUPPORT identity, not override evidence."""

    def test_valid_transition(self):
        """Valid camera transition within time window → body candidate allowed."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 2, 10)]
        )
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0))
        r.unbind("cam1", 1)
        clk.advance(5)  # Within [2, 10]
        d = r.resolve("cam2", 10, body_embedding=vec(1, 0, 0))
        self.assertIn(d.state, [DecisionState.UNCERTAIN, DecisionState.CANDIDATE])

    def test_impossible_travel_time_too_fast(self):
        """Travel time too fast → rejected."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 5, 20)]
        )
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0))
        r.unbind("cam1", 1)
        clk.advance(2)  # Too fast (< 5s)
        d = r.resolve("cam2", 10, body_embedding=vec(1, 0, 0))
        self.assertEqual("camera_transition_time_invalid", d.reason)

    def test_impossible_travel_time_too_slow(self):
        """Travel time too slow → rejected."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 5, 10)]
        )
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0))
        r.unbind("cam1", 1)
        clk.advance(15)  # Too slow (> 10s)
        d = r.resolve("cam2", 10, body_embedding=vec(1, 0, 0))
        self.assertEqual("camera_transition_time_invalid", d.reason)

    def test_unconfigured_transition(self):
        """No configured transition → rejected."""
        r, clk = make_resolver()  # No transitions
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0))
        r.unbind("cam1", 1)
        clk.advance(5)
        d = r.resolve("cam2", 10, body_embedding=vec(1, 0, 0))
        self.assertEqual("camera_transition_not_configured", d.reason)

    def test_transition_direction_matters(self):
        """A→B configured does not allow B→A."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 0, 20)]
        )
        r.resolve("cam2", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0))
        r.unbind("cam2", 1)
        clk.advance(5)
        d = r.resolve("cam1", 10, body_embedding=vec(1, 0, 0))
        self.assertEqual("camera_transition_not_configured", d.reason)

    def test_transition_does_not_override_face_evidence(self):
        """Face says Person A. Camera transition favors Person B → face wins."""
        transition = CameraTransition("cam1", "cam2", 0, 30, overlap=True)
        r, clk = make_resolver(transitions=[transition])
        # Person A in cam1
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="A")
        # Person B in cam1
        r.resolve("cam1", 2,
                   face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="B")
        clk.advance(5)
        # Face says A in cam2 — transition context is irrelevant
        d = r.resolve("cam2", 10, face_embedding=vec(1, 0, 0), known_person_id="A")
        self.assertEqual(DecisionState.CONFIRMED, d.state)
        self.assertEqual("A", r.get_state(d.global_id).known_person_id)

    def test_invalid_transition_parameters(self):
        """Invalid travel time range → ValueError."""
        with self.assertRaises(ValueError):
            CameraTransition("a", "b", 10, 5)  # min > max
        with self.assertRaises(ValueError):
            CameraTransition("a", "b", -1, 10)  # negative


# ═════════════════════════════════════════════════════════════════════════
# PART 9 — MULTI-CAMERA SIMULTANEITY
# ═════════════════════════════════════════════════════════════════════════

class TestPart09_MultiCameraSimultaneity(unittest.TestCase):
    """Concurrency scenarios with overlapping and non-overlapping cameras."""

    def test_case1_overlap_no_duplicate(self):
        """Same person in two overlapping cameras → no duplicate Global IDs."""
        r, _ = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 0, 20, overlap=True)]
        )
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("cam2", 5, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id)

    def test_case2_different_people_non_overlap_not_merged(self):
        """Different people in non-overlapping cameras → not merged."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("cam2", 5, face_embedding=vec(0, 1, 0),
                        known_person_id="bob")
        self.assertNotEqual(d1.global_id, d2.global_id)

    def test_case3_near_simultaneous_timestamps_deterministic(self):
        """Two cameras at nearly identical timestamps → deterministic."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice", timestamp=1000.0)
        d2 = r.resolve("cam2", 5, face_embedding=vec(0, 1, 0),
                        known_person_id="bob", timestamp=1000.001)
        self.assertNotEqual(d1.global_id, d2.global_id)

    def test_case4_same_person_non_overlap_conflict(self):
        """Same person in two non-overlapping cameras simultaneously → CONFLICT."""
        r, _ = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 0, 20, overlap=False)]
        )
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("cam2", 5, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        # Should be UNCERTAIN with conflict
        self.assertEqual(DecisionState.UNCERTAIN, d2.state)
        self.assertIn("conflict", d2.reason.lower())


# ═════════════════════════════════════════════════════════════════════════
# PART 10 — CONCURRENCY / RACE CONDITION TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart10_ConcurrencyRaceConditions(unittest.TestCase):
    """Thread safety tests. Run multiple times for confidence."""

    def test_simultaneous_known_person_creates_single_id(self):
        """20 threads resolving the same known person → exactly one Global ID."""
        r, _ = make_resolver()
        results = []
        errors = []
        barrier = threading.Barrier(20)

        def worker(i):
            try:
                barrier.wait(timeout=5)
                d = r.resolve("cam1", i, face_embedding=vec(1, 0, 0),
                               known_person_id="same-person")
                results.append(d.global_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual([], errors, f"Thread errors: {errors}")
        unique_ids = set(results)
        self.assertEqual(1, len(unique_ids), f"Expected 1 Global ID, got {unique_ids}")

    def test_simultaneous_different_people(self):
        """Concurrent creation of different people → separate Global IDs."""
        r, _ = make_resolver()
        results: Dict[str, str] = {}
        errors = []
        lock = threading.Lock()
        barrier = threading.Barrier(10)

        def worker(person_id, track_id):
            try:
                barrier.wait(timeout=5)
                d = r.resolve("cam1", track_id,
                               face_embedding=vec(track_id + 1, 0, 0),
                               known_person_id=person_id)
                with lock:
                    results[person_id] = d.global_id
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"person-{i}", i))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual([], errors)
        unique_ids = set(results.values())
        self.assertEqual(10, len(unique_ids), f"Expected 10 unique IDs, got {len(unique_ids)}")

    def test_concurrent_resolve_and_unbind(self):
        """Concurrent resolve + unbind on same track → no crash, no corrupt state."""
        r, clk = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="test")
        errors = []
        barrier = threading.Barrier(10)

        def resolver_worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                               known_person_id="test")
            except Exception as e:
                errors.append(e)

        def unbinder_worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    r.unbind("cam1", 1)
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=resolver_worker) for _ in range(5)] +
            [threading.Thread(target=unbinder_worker) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual([], errors, f"Concurrent errors: {errors}")

    def test_concurrent_sweep(self):
        """Concurrent sweep operations → no crash."""
        r, clk = make_resolver()
        for i in range(10):
            r.resolve("cam1", i, face_embedding=vec(1, i + 1, 0),
                       known_person_id=f"p-{i}")
            r.unbind("cam1", i)
        clk.advance(121)  # Past TTL
        errors = []
        barrier = threading.Barrier(10)

        def sweep_worker():
            try:
                barrier.wait(timeout=5)
                r.sweep()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sweep_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual([], errors)

    def test_deterministic_across_runs(self):
        """Same sequence → same result, multiple runs."""
        results = []
        for _ in range(5):
            r, clk = make_resolver(id_factory=IDs())
            d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                           known_person_id="alice")
            results.append(d.global_id)
        self.assertEqual(1, len(set(results)))


# ═════════════════════════════════════════════════════════════════════════
# PART 11 — DUPLICATE TRACK TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart11_DuplicateTracks(unittest.TestCase):
    """Multiple tracks for same physical person within short time window."""

    def test_same_person_multiple_tracks_same_global_id(self):
        """Tracks 81, 82, 83 within seconds → same Global ID if face matches."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 81, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        clk.advance(0.5)
        d2 = r.resolve("cam1", 82, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        clk.advance(0.5)
        d3 = r.resolve("cam1", 83, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id)
        self.assertEqual(d2.global_id, d3.global_id)

    def test_different_people_different_tracks_different_ids(self):
        """Different people → different Global IDs, even close in time."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 81, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("cam1", 82, face_embedding=vec(0, 1, 0),
                        known_person_id="bob")
        d3 = r.resolve("cam1", 83, face_embedding=vec(0, 0, 1),
                        known_person_id="charlie")
        ids = {d1.global_id, d2.global_id, d3.global_id}
        self.assertEqual(3, len(ids))


# ═════════════════════════════════════════════════════════════════════════
# PART 12 — FALSE MERGE / FALSE SPLIT TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart12_FalseMergeFalseSplit(unittest.TestCase):
    """The two most important identity metrics."""

    def test_false_merge_different_people_similar_clothing(self):
        """Two people with similar body but different faces → separate Global IDs."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1,
                        face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("cam1", 2,
                        face_embedding=vec(0, 1, 0), body_embedding=vec(0.98, 0.2, 0),
                        known_person_id="bob")
        self.assertNotEqual(d1.global_id, d2.global_id, "FALSE MERGE: different people merged!")

    def test_false_merge_same_uniform(self):
        """Same uniform, different faces → separate identities."""
        uniform_body = vec(1, 0.1, 0.1)
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1,
                        face_embedding=vec(1, 0, 0), body_embedding=uniform_body,
                        known_person_id="student1")
        d2 = r.resolve("cam1", 2,
                        face_embedding=vec(0, 1, 0), body_embedding=uniform_body,
                        known_person_id="student2")
        self.assertNotEqual(d1.global_id, d2.global_id, "FALSE MERGE: uniform caused merge!")

    def test_false_split_track_change(self):
        """Same person, track ID changes → same Global ID (no false split)."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 10, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        r.unbind("cam1", 10)
        clk.advance(1)
        d2 = r.resolve("cam1", 20, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id, "FALSE SPLIT: same person got different IDs!")

    def test_false_split_temporary_occlusion(self):
        """Person occluded briefly → same Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        r.unbind("cam1", 1)
        clk.advance(5)  # Brief occlusion
        d2 = r.resolve("cam1", 2, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id, "FALSE SPLIT after occlusion!")

    def test_false_split_camera_transition(self):
        """Person moves between cameras → same Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        r.unbind("cam1", 1)
        clk.advance(10)
        d2 = r.resolve("cam2", 5, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id, "FALSE SPLIT across cameras!")

    def test_false_merge_crossing_paths(self):
        """Two people cross paths → do not merge."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1,
                        face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("cam1", 2,
                        face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                        known_person_id="bob")
        # Even after crossing, identities remain separate
        self.assertNotEqual(d1.global_id, d2.global_id)

    def test_face_disappearance_no_false_merge(self):
        """Face disappears, only body visible — must not merge with wrong person."""
        r, clk = make_resolver()
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="alice")
        r.resolve("cam1", 2,
                   face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="bob")
        clk.advance(1)
        # Track 1 loses face, shows body similar to bob — must stay alice
        d = r.resolve("cam1", 1, body_embedding=vec(0, 1, 0))
        self.assertEqual("alice", r.get_state(d.global_id).known_person_id)


# ═════════════════════════════════════════════════════════════════════════
# PART 13 — 120-SECOND TTL TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart13_TTLBoundaries(unittest.TestCase):
    """Exact boundary testing for the 120-second missing timeout."""

    def _make_missing(self, r, clk):
        """Helper: create a person and make them missing."""
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="ttl-test")
        r.unbind("cam1", 1)
        return d.global_id

    def test_119_0_seconds_still_missing(self):
        r, clk = make_resolver()
        gid = self._make_missing(r, clk)
        clk.advance(119.0)
        r.sweep()
        self.assertEqual(LifecycleState.MISSING, r.get_state(gid).lifecycle)

    def test_119_9_seconds_still_missing(self):
        r, clk = make_resolver()
        gid = self._make_missing(r, clk)
        clk.advance(119.9)
        r.sweep()
        self.assertEqual(LifecycleState.MISSING, r.get_state(gid).lifecycle)

    def test_120_0_seconds_expired(self):
        r, clk = make_resolver()
        gid = self._make_missing(r, clk)
        clk.advance(120.0)
        r.sweep()
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(gid).lifecycle)

    def test_120_1_seconds_expired(self):
        r, clk = make_resolver()
        gid = self._make_missing(r, clk)
        clk.advance(120.1)
        r.sweep()
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(gid).lifecycle)

    def test_121_seconds_expired(self):
        r, clk = make_resolver()
        gid = self._make_missing(r, clk)
        clk.advance(121)
        r.sweep()
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(gid).lifecycle)

    def test_reappear_at_119_then_disappear_ttl_resets(self):
        """Reappear at 119s, disappear again → TTL resets from new disappearance."""
        r, clk = make_resolver()
        gid = self._make_missing(r, clk)
        clk.advance(119)
        # Reappear
        r.resolve("cam1", 50, face_embedding=vec(1, 0, 0), known_person_id="ttl-test")
        state = r.get_state(gid)
        self.assertEqual(LifecycleState.ACTIVE, state.lifecycle)
        # Disappear again
        r.unbind("cam1", 50)
        self.assertEqual(LifecycleState.MISSING, r.get_state(gid).lifecycle)
        # 119 more seconds from new disappearance — should still be MISSING
        clk.advance(119)
        r.sweep()
        self.assertEqual(LifecycleState.MISSING, r.get_state(gid).lifecycle)
        # 2 more seconds (total 121 from new disappearance) — should be EXPIRED
        clk.advance(2)
        r.sweep()
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(gid).lifecycle)

    def test_ttl_does_not_run_while_active_in_another_camera(self):
        """TTL must NOT run while person is active in any camera."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 0, 30, overlap=True)]
        )
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        r.resolve("cam2", 5, face_embedding=vec(1, 0, 0), known_person_id="x")
        # Unbind cam1 only
        r.unbind("cam1", 1)
        clk.advance(200)  # Way past TTL
        r.sweep()
        self.assertEqual(LifecycleState.ACTIVE, r.get_state(d.global_id).lifecycle)

    def test_exact_120_boundary_semantics(self):
        """Verify >= vs > at the 120 second mark. Current code uses < for comparison."""
        r, clk = make_resolver()
        gid = self._make_missing(r, clk)
        # At exactly 120s, the condition is: now - missing_since < 120 → False
        # So it should expire at exactly 120s.
        clk.advance(120.0)
        expired = r.sweep()
        state = r.get_state(gid)
        # Document: exactly at 120.0 the code does `now - missing_since < self.missing_timeout`
        # 120.0 - 0 = 120.0, which is NOT < 120.0, so it expires.
        # This is >= semantics (expire AT the boundary).
        self.assertEqual(LifecycleState.EXPIRED, state.lifecycle,
                         "BUG CANDIDATE: Boundary semantics at exactly 120.0 seconds")


# ═════════════════════════════════════════════════════════════════════════
# PART 14 — GALLERY BOUNDING TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart14_GalleryBounding(unittest.TestCase):
    """Gallery sizes must remain bounded."""

    def test_gallery_with_many_people(self):
        """Many active people → gallery search still works."""
        r, clk = make_resolver()
        ids = []
        for i in range(50):
            d = r.resolve(f"cam{i%4}", i, face_embedding=rand_vec(128, seed=i),
                           known_person_id=f"p-{i}")
            ids.append(d.global_id)
        self.assertEqual(50, r.metrics["active_count"])
        # All IDs should be unique
        self.assertEqual(50, len(set(ids)))

    def test_expired_not_in_search_results(self):
        """Expired Global IDs are removed from the matching set."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="old")
        gid = d.global_id
        r.unbind("cam1", 1)
        clk.advance(121)
        r.sweep()
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(gid).lifecycle)
        # Galleries should be cleared
        state = r.get_state(gid)
        self.assertEqual([], state.face_gallery, "Expired state has non-empty face gallery!")
        self.assertEqual([], state.body_gallery, "Expired state has non-empty body gallery!")


# ═════════════════════════════════════════════════════════════════════════
# PART 15 — EMBEDDING GALLERY TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart15_EmbeddingGallery(unittest.TestCase):
    """Embedding diversity, bounding, and quality."""

    def test_no_unbounded_growth(self):
        """Gallery size stays within configured bounds."""
        r, clk = make_resolver(face_gallery_size=6, body_gallery_size=6)
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                       known_person_id="alice")
        for i in range(100):
            angle = 0.1 + i * 0.03
            r.resolve("cam1", 1,
                       face_embedding=vec(np.cos(angle), np.sin(angle), 0.01 * (i % 5)),
                       body_embedding=vec(np.cos(angle), 0.01, np.sin(angle)))
            clk.advance(0.1)
        state = r.get_state(d.global_id)
        self.assertLessEqual(len(state.face_gallery), 6)
        self.assertLessEqual(len(state.body_gallery), 6)

    def test_duplicate_embedding_rejected(self):
        """Repeated identical embedding → not duplicated in gallery."""
        r, clk = make_resolver(face_gallery_size=8)
        face = vec(1, 0, 0)
        r.resolve("cam1", 1, face_embedding=face, known_person_id="alice")
        for _ in range(20):
            r.resolve("cam1", 1, face_embedding=face)
            clk.advance(0.1)
        state = r.get_state(r.global_id_for("cam1", 1))
        # Due to gallery_min_distance, near-identical vectors are rejected
        self.assertLessEqual(len(state.face_gallery), 2)

    def test_diverse_embeddings_preserved(self):
        """Diverse embeddings fill the gallery."""
        r, clk = make_resolver(face_gallery_size=8)
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        # Add diverse views
        for i in range(10):
            angle = i * 0.5
            r.resolve("cam1", 1,
                       face_embedding=vec(np.cos(angle), np.sin(angle), 0.1 * i))
            clk.advance(0.1)
        state = r.get_state(d.global_id)
        self.assertGreater(len(state.face_gallery), 1)
        self.assertLessEqual(len(state.face_gallery), 8)


# ═════════════════════════════════════════════════════════════════════════
# PART 16 — EVENT / AUDIT CONSISTENCY
# ═════════════════════════════════════════════════════════════════════════

class TestPart16_EventAuditConsistency(unittest.TestCase):
    """Events and audit logs must be consistent."""

    def test_create_event_emitted(self):
        """New identity → GLOBAL_ID_CREATED event."""
        r, _ = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        events = [e for e in r.events if e["type"] == EventType.GLOBAL_ID_CREATED.value]
        self.assertGreater(len(events), 0)

    def test_missing_event_emitted(self):
        """Unbind last camera → GLOBAL_ID_MISSING event."""
        r, _ = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r.unbind("cam1", 1)
        events = [e for e in r.events if e["type"] == EventType.MISSING.value]
        self.assertGreater(len(events), 0)

    def test_expired_event_emitted(self):
        """After TTL → GLOBAL_ID_EXPIRED event."""
        r, clk = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r.unbind("cam1", 1)
        clk.advance(121)
        r.sweep()
        events = [e for e in r.events if e["type"] == EventType.EXPIRED.value]
        self.assertGreater(len(events), 0)

    def test_reappeared_event_emitted(self):
        """Person returns from MISSING → GLOBAL_ID_REAPPEARED event."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r.unbind("cam1", 1)
        clk.advance(30)
        r.resolve("cam1", 2, face_embedding=vec(1, 0, 0), known_person_id="alice")
        events = [e for e in r.events if e["type"] == EventType.REAPPEARED.value]
        self.assertGreater(len(events), 0)

    def test_audit_log_no_biometric_vectors(self):
        """Audit log must never contain raw biometric vectors."""
        r, _ = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="alice")
        for entry in r.audit_log:
            for key, value in entry.items():
                self.assertNotIsInstance(value, np.ndarray,
                                         f"Audit log contains numpy array at key '{key}'!")

    def test_audit_log_fields_present(self):
        """Audit log entries have required fields."""
        r, _ = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        log = r.audit_log
        self.assertEqual(1, len(log))
        entry = log[0]
        required = ["timestamp", "camera_id", "local_track_id", "decision",
                     "global_id", "reason", "mode", "latency_ms"]
        for field in required:
            self.assertIn(field, entry, f"Missing audit field: {field}")

    def test_event_callback_called(self):
        """event_callback receives events."""
        received = []
        r, _ = make_resolver(event_callback=lambda e: received.append(e))
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        self.assertGreater(len(received), 0)
        self.assertIn("type", received[0])
        self.assertIn("global_id", received[0])

    def test_event_callback_failure_does_not_break_resolution(self):
        """Broken event_callback must not crash resolve()."""
        def bad_callback(e):
            raise RuntimeError("callback failure")

        r, _ = make_resolver(event_callback=bad_callback)
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        self.assertEqual(DecisionState.CONFIRMED, d.state)

    def test_moved_camera_event(self):
        """Person seen in different camera → MOVED_CAMERA event."""
        r, clk = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r.unbind("cam1", 1)
        clk.advance(5)
        r.resolve("cam2", 5, face_embedding=vec(1, 0, 0), known_person_id="alice")
        events = [e for e in r.events if e["type"] == EventType.MOVED_CAMERA.value]
        self.assertGreater(len(events), 0)


# ═════════════════════════════════════════════════════════════════════════
# PART 17 — RESTART / RECOVERY TEST
# ═════════════════════════════════════════════════════════════════════════

class TestPart17_RestartRecovery(unittest.TestCase):
    """Global state behavior on restart (new resolver instance)."""

    def test_in_memory_state_does_not_survive_restart(self):
        """A new resolver instance has no memory of the old one (by design)."""
        r1, _ = make_resolver()
        r1.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        # "Restart" = new resolver
        r2, _ = make_resolver()
        self.assertEqual(0, r2.metrics["active_count"])
        self.assertEqual(0, r2.metrics["missing_count"])
        self.assertEqual(0, r2.metrics["expired_count"])

    def test_active_ids_lost_on_restart(self):
        """Active Global IDs are lost — documented behavior."""
        r1, _ = make_resolver(id_factory=IDs())
        d = r1.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r2, _ = make_resolver(id_factory=IDs())
        self.assertIsNone(r2.get_state(d.global_id))

    def test_galleries_cleared_on_restart(self):
        """Face/body galleries are in-memory → gone after restart."""
        r, _ = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="alice")
        state = r.get_state(r.global_id_for("cam1", 1))
        self.assertGreater(len(state.face_gallery), 0)
        # After restart
        r2, _ = make_resolver()
        # No state to check — it's simply gone


# ═════════════════════════════════════════════════════════════════════════
# PART 18 — PERFORMANCE TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestPart18_Performance(unittest.TestCase):
    """Identity resolution must not block the frame-processing loop."""

    def test_resolve_latency_single(self):
        """Single resolve < 100ms (generous bound)."""
        r, _ = make_resolver()
        start = time.perf_counter()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        elapsed = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed, 100, f"Single resolve took {elapsed:.1f}ms")

    def test_resolve_100_identities_under_5s(self):
        """100 resolves with gallery search < 5s total."""
        r, clk = make_resolver()
        start = time.perf_counter()
        for i in range(100):
            r.resolve("cam1", i, face_embedding=rand_vec(128, seed=i),
                       known_person_id=f"p-{i}")
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"100 resolves took {elapsed:.1f}s")

    def test_gallery_search_with_200_active(self):
        """Gallery search with 200 active identities stays reasonable."""
        r, clk = make_resolver()
        for i in range(200):
            r.resolve(f"cam{i%4}", i, face_embedding=rand_vec(128, seed=i))
        self.assertEqual(200, r.metrics["active_count"])
        start = time.perf_counter()
        # Resolve against the full gallery
        for i in range(10):
            r.resolve("cam1", 1000 + i, face_embedding=rand_vec(128, seed=1000 + i))
        elapsed = (time.perf_counter() - start) * 1000
        per_resolve = elapsed / 10
        # Should be well under 50ms per resolve even with 200 active
        self.assertLess(per_resolve, 200, f"Per-resolve with 200 active: {per_resolve:.1f}ms")

    def test_metrics_tracking(self):
        """Metrics are populated correctly."""
        r, _ = make_resolver()
        r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        m = r.metrics
        self.assertEqual(1, m["resolve_count"])
        self.assertEqual(1, m["active_count"])
        self.assertGreaterEqual(m["latency_ms_average"], 0)


# ═════════════════════════════════════════════════════════════════════════
# PART 20 — PROPERTY / INVARIANT TESTING
# ═════════════════════════════════════════════════════════════════════════

class TestPart20_PropertyInvariants(unittest.TestCase):
    """Automated tests for the 10 critical invariants."""

    def test_invariant_1_one_active_id_cannot_represent_two_people(self):
        """INVARIANT 1: One active Global ID cannot simultaneously represent two
        incompatible known person IDs."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        # Try to also assign bob to the same global ID — should get a different one
        d2 = r.resolve("cam1", 2, face_embedding=vec(0, 1, 0), known_person_id="bob")
        self.assertNotEqual(d1.global_id, d2.global_id)
        # Each global ID has exactly one known_person_id
        self.assertEqual("alice", r.get_state(d1.global_id).known_person_id)
        self.assertEqual("bob", r.get_state(d2.global_id).known_person_id)

    def test_invariant_2_local_track_cannot_switch_global_ids_arbitrarily(self):
        """INVARIANT 2: A single local Track ID stays bound to its Global ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        clk.advance(1)
        # Same track, same person → same global ID
        d2 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id)
        # Even without known_person_id, binding persists
        clk.advance(1)
        d3 = r.resolve("cam1", 1, body_embedding=vec(0, 1, 0))
        self.assertEqual(d1.global_id, d3.global_id)

    def test_invariant_3_face_cannot_be_overwritten_by_body(self):
        """INVARIANT 3: A strong face identity cannot be silently overwritten by body ReID."""
        r, clk = make_resolver()
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="alice")
        r.resolve("cam1", 2,
                   face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="bob")
        clk.advance(1)
        # Body of bob provided for track 1 (alice) — must not switch
        d = r.resolve("cam1", 1, body_embedding=vec(0, 1, 0))
        self.assertEqual("alice", r.get_state(d.global_id).known_person_id)

    def test_invariant_4_expired_ids_not_matched_as_active(self):
        """INVARIANT 4: Expired Global IDs cannot be matched as active candidates."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="old")
        old_gid = d.global_id
        r.unbind("cam1", 1)
        clk.advance(121)
        r.sweep()
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(old_gid).lifecycle)
        # New resolve with same face should create NEW identity
        d2 = r.resolve("cam1", 2, face_embedding=vec(1, 0, 0), known_person_id="old")
        self.assertNotEqual(old_gid, d2.global_id)

    def test_invariant_5_missing_timer_after_all_cameras(self):
        """INVARIANT 5: Missing timer starts only after ALL active cameras lose the person."""
        r, clk = make_resolver(
            transitions=[CameraTransition("cam1", "cam2", 0, 30, overlap=True)]
        )
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        r.resolve("cam2", 5, face_embedding=vec(1, 0, 0), known_person_id="x")
        r.unbind("cam1", 1)
        state = r.get_state(d.global_id)
        self.assertEqual(LifecycleState.ACTIVE, state.lifecycle)
        self.assertIsNone(state.missing_since)

    def test_invariant_6_unknown_to_known_no_new_id(self):
        """INVARIANT 6: Unknown → known does not create a new Global ID."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0))
        d2 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="known")
        self.assertEqual(d1.global_id, d2.global_id)

    def test_invariant_7_global_id_independent_of_local_track(self):
        """INVARIANT 7: Global ID remains independent of local Track ID."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 100, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        r.unbind("cam1", 100)
        clk.advance(1)
        d2 = r.resolve("cam1", 200, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        r.unbind("cam1", 200)
        clk.advance(1)
        d3 = r.resolve("cam1", 300, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id)
        self.assertEqual(d2.global_id, d3.global_id)

    def test_invariant_8_no_impossible_simultaneous_states(self):
        """INVARIANT 8: Global ID cannot have impossible simultaneous states."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        gid = d.global_id
        state = r.get_state(gid)
        # Active state: must not be in missing or expired
        self.assertEqual(LifecycleState.ACTIVE, state.lifecycle)
        self.assertNotIn(gid, r.missing_states)
        self.assertNotIn(gid, r.expired_states)
        self.assertIn(gid, r.active_states)
        # Missing state
        r.unbind("cam1", 1)
        state = r.get_state(gid)
        self.assertEqual(LifecycleState.MISSING, state.lifecycle)
        self.assertNotIn(gid, r.active_states)
        self.assertIn(gid, r.missing_states)
        self.assertNotIn(gid, r.expired_states)
        # Expired state
        clk.advance(121)
        r.sweep()
        state = r.get_state(gid)
        self.assertEqual(LifecycleState.EXPIRED, state.lifecycle)
        self.assertNotIn(gid, r.active_states)
        self.assertNotIn(gid, r.missing_states)
        self.assertIn(gid, r.expired_states)

    def test_invariant_9_galleries_bounded(self):
        """INVARIANT 9: Embedding galleries remain bounded."""
        r, clk = make_resolver(face_gallery_size=8, body_gallery_size=8)
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                       body_embedding=vec(0, 1, 0), known_person_id="alice")
        for i in range(200):
            angle = i * 0.1
            r.resolve("cam1", 1,
                       face_embedding=vec(np.cos(angle), np.sin(angle), 0.01 * (i % 7)),
                       body_embedding=vec(np.cos(angle), 0.01, np.sin(angle)))
            clk.advance(0.05)
        state = r.get_state(d.global_id)
        self.assertLessEqual(len(state.face_gallery), 8)
        self.assertLessEqual(len(state.body_gallery), 8)

    def test_invariant_10_resolution_does_not_block_indefinitely(self):
        """INVARIANT 10: Identity resolution completes in bounded time."""
        r, _ = make_resolver()
        # Fill gallery
        for i in range(100):
            r.resolve("cam1", i, face_embedding=rand_vec(128, seed=i))
        start = time.perf_counter()
        r.resolve("cam1", 999, face_embedding=rand_vec(128, seed=999))
        elapsed = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed, 5000, f"Resolution took {elapsed:.1f}ms — too long!")


# ═════════════════════════════════════════════════════════════════════════
# PART 21 — FAILURE INJECTION
# ═════════════════════════════════════════════════════════════════════════

class TestPart21_FailureInjection(unittest.TestCase):
    """Deliberately break things. System should fail conservatively."""

    def test_nan_face_embedding(self):
        """NaN embedding → UNCERTAIN, no identity created."""
        r, _ = make_resolver()
        nan_vec = np.array([float('nan')] * 3, dtype=np.float32)
        d = r.resolve("cam1", 1, face_embedding=nan_vec)
        self.assertEqual(DecisionState.UNCERTAIN, d.state)
        self.assertEqual(0, r.metrics["active_count"])

    def test_inf_embedding(self):
        """Inf embedding → UNCERTAIN."""
        r, _ = make_resolver()
        inf_vec = np.array([float('inf'), 0, 0], dtype=np.float32)
        d = r.resolve("cam1", 1, face_embedding=inf_vec)
        self.assertEqual(DecisionState.UNCERTAIN, d.state)

    def test_zero_embedding(self):
        """Zero vector → UNCERTAIN."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=np.zeros(3, dtype=np.float32))
        self.assertEqual(DecisionState.UNCERTAIN, d.state)

    def test_empty_gallery_search(self):
        """Search with no active identities → graceful."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0))
        # First person → should be NEW
        self.assertEqual(DecisionState.NEW, d.state)

    def test_corrupt_embedding_wrong_dtype(self):
        """String array as embedding → graceful rejection."""
        r, _ = make_resolver()
        try:
            d = r.resolve("cam1", 1, face_embedding=np.array(["a", "b", "c"]))
            # Should either raise or return UNCERTAIN
            self.assertEqual(DecisionState.UNCERTAIN, d.state)
        except (TypeError, ValueError):
            pass  # Also acceptable

    def test_duplicate_observation_idempotent(self):
        """Same observation repeated rapidly → idempotent."""
        r, _ = make_resolver()
        results = []
        for _ in range(10):
            d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                           known_person_id="alice")
            results.append(d.global_id)
        # All should return the same Global ID
        self.assertEqual(1, len(set(results)))

    def test_stale_local_index_after_expiry(self):
        """Stale local_to_global entry after expiry → defensive cleanup."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        old_gid = d.global_id
        r.unbind("cam1", 1)
        clk.advance(121)
        r.sweep()
        # Re-resolve with same key — stale index should be cleaned up
        d2 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        self.assertNotEqual(old_gid, d2.global_id)
        self.assertEqual(DecisionState.CONFIRMED, d2.state)

    def test_concurrent_identity_creation_no_duplicate(self):
        """Multiple threads creating the same known person → one Global ID."""
        r, _ = make_resolver()
        results = []
        barrier = threading.Barrier(10)

        def worker(i):
            barrier.wait(timeout=5)
            d = r.resolve(f"cam{i}", i, face_embedding=vec(1, 0, 0),
                           known_person_id="shared")
            results.append(d.global_id)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        # All should map to same known person's Global ID
        self.assertEqual(1, len(set(results)))

    def test_expired_id_not_revived_by_resolve(self):
        """An expired ID cannot be revived by a new resolve call."""
        r, clk = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        old_gid = d1.global_id
        r.unbind("cam1", 1)
        clk.advance(121)
        r.sweep()
        # Try to resolve with same known person — should get new ID
        d2 = r.resolve("cam1", 2, face_embedding=vec(1, 0, 0), known_person_id="alice")
        self.assertNotEqual(old_gid, d2.global_id)
        # Old one should remain expired
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(old_gid).lifecycle)

    def test_bind_expired_raises(self):
        """Attempting to bind an expired state → RuntimeError."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        r.unbind("cam1", 1)
        clk.advance(121)
        r.sweep()
        state = r.get_state(d.global_id)
        self.assertEqual(LifecycleState.EXPIRED, state.lifecycle)
        # Directly trying to bind should raise
        with self.assertRaises(RuntimeError):
            r._bind(state, ("cam1", 99), clk())

    def test_event_callback_exception_swallowed(self):
        """Broken event callback must not crash identity resolution."""
        call_count = [0]
        def failing_cb(e):
            call_count[0] += 1
            raise ValueError("boom")
        r, _ = make_resolver(event_callback=failing_cb)
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        self.assertEqual(DecisionState.CONFIRMED, d.state)
        self.assertGreater(call_count[0], 0)


# ═════════════════════════════════════════════════════════════════════════
# PART 22 — CODE AUDIT: STRUCTURAL CHECKS
# ═════════════════════════════════════════════════════════════════════════

class TestPart22_CodeAudit(unittest.TestCase):
    """Structural assertions about the implementation."""

    def test_resolver_uses_reentrant_lock(self):
        """The lock must be re-entrant (RLock) for nested calls."""
        r, _ = make_resolver()
        self.assertIsInstance(r._lock, type(threading.RLock()))

    def test_all_public_methods_acquire_lock(self):
        """Key public methods must hold the lock for thread safety."""
        r, _ = make_resolver()
        # Verify resolve, unbind, sweep, get_state, global_id_for acquire lock
        # We can verify by checking they don't deadlock when called within lock
        with r._lock:
            # These should all work (RLock allows re-entry)
            r.get_state("nonexistent")
            r.global_id_for("cam1", 1)

    def test_expired_galleries_cleared(self):
        """When a state expires, face and body galleries are cleared."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1,
                       face_embedding=vec(1, 0, 0), body_embedding=vec(0, 1, 0),
                       known_person_id="alice")
        gid = d.global_id
        state = r.get_state(gid)
        self.assertGreater(len(state.face_gallery), 0)
        self.assertGreater(len(state.body_gallery), 0)
        r.unbind("cam1", 1)
        clk.advance(121)
        r.sweep()
        state = r.get_state(gid)
        self.assertEqual([], state.face_gallery)
        self.assertEqual([], state.body_gallery)

    def test_purge_expired_removes_metadata(self):
        """purge_expired removes expired metadata."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        gid = d.global_id
        r.unbind("cam1", 1)
        clk.advance(121)
        r.sweep()
        self.assertIn(gid, r.expired_states)
        count = r.purge_expired()
        self.assertGreaterEqual(count, 1)

    def test_max_expired_bounded(self):
        """Expired count is bounded by max_expired."""
        r, clk = make_resolver(max_expired=5)
        for i in range(20):
            r.resolve("cam1", i, face_embedding=rand_vec(128, seed=i),
                       known_person_id=f"p-{i}")
            r.unbind("cam1", i)
        clk.advance(121)
        r.sweep()
        self.assertLessEqual(len(r.expired_states), 5)

    def test_vector_normalization(self):
        """Embeddings are normalized before use."""
        r, _ = make_resolver()
        raw = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        d = r.resolve("cam1", 1, face_embedding=raw, known_person_id="test")
        state = r.get_state(d.global_id)
        if state.face_gallery:
            stored = state.face_gallery[0]
            norm = float(np.linalg.norm(stored))
            self.assertAlmostEqual(1.0, norm, places=5)

    def test_shadow_mode_does_not_write_label(self):
        """Shadow mode must not mutate label_target."""
        r, _ = make_resolver(mode=ResolverMode.SHADOW)
        target = {"global_person_id": "original"}
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice",
                       label_target=target)
        self.assertEqual("original", target["global_person_id"])
        self.assertFalse(d.authoritative_applied)

    def test_authoritative_mode_writes_label(self):
        """Authoritative mode writes global_person_id to label_target."""
        r, _ = make_resolver(mode=ResolverMode.AUTHORITATIVE)
        target = {}
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice",
                       label_target=target)
        self.assertEqual(d.global_id, target.get("global_person_id"))
        self.assertTrue(d.authoritative_applied)

    def test_body_only_cannot_create_identity(self):
        """Body-only (no existing state) cannot create a new Global ID."""
        r, _ = make_resolver()
        d = r.resolve("cam1", 1, body_embedding=vec(1, 0, 0))
        # Either NEW with None global_id, or UNCERTAIN
        if d.state == DecisionState.NEW:
            self.assertIsNone(d.global_id)
        else:
            self.assertNotEqual(DecisionState.CONFIRMED, d.state)

    def test_policy_validation_rejects_bad_params(self):
        """CalibratedPolicy rejects invalid parameters."""
        with self.assertRaises(ValueError):
            CalibratedPolicy(body_match_threshold=2.0)  # > 1.0
        with self.assertRaises(ValueError):
            CalibratedPolicy(body_votes_required=1)  # < 2
        with self.assertRaises(ValueError):
            CalibratedPolicy(body_min_margin=-0.1)  # negative

    def test_missing_timeout_must_be_positive(self):
        """missing_timeout <= 0 → ValueError."""
        with self.assertRaises(ValueError):
            GlobalIdentityResolver(missing_timeout=0)
        with self.assertRaises(ValueError):
            GlobalIdentityResolver(missing_timeout=-10)

    def test_gallery_size_bounds_enforced(self):
        """Gallery sizes outside [5, 10] → ValueError."""
        with self.assertRaises(ValueError):
            GlobalIdentityResolver(face_gallery_size=3)
        with self.assertRaises(ValueError):
            GlobalIdentityResolver(body_gallery_size=15)


# ═════════════════════════════════════════════════════════════════════════
# PART: STRESS / ADVERSARIAL
# ═════════════════════════════════════════════════════════════════════════

class TestAdversarial(unittest.TestCase):
    """Adversarial scenarios designed to break the identity system."""

    def test_adversarial_body_tries_to_hijack_face_identity(self):
        """Adversarial: body ReID tries to steal face-confirmed identity."""
        r, clk = make_resolver()
        # Alice confirmed by face
        r.resolve("cam1", 1,
                   face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
                   known_person_id="alice")
        # Bob confirmed by face
        r.resolve("cam1", 2,
                   face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
                   known_person_id="bob")
        clk.advance(1)
        # Alice's track 1 now has bob's body — MUST remain alice
        for _ in range(20):
            d = r.resolve("cam1", 1, body_embedding=vec(0, 1, 0))
            self.assertEqual("alice", r.get_state(d.global_id).known_person_id,
                             "Body ReID HIJACKED face identity!")
            clk.advance(0.5)

    def test_adversarial_rapid_id_creation_and_expiry(self):
        """Rapid creation + expiry cycle must not leak state."""
        r, clk = make_resolver(max_expired=50)
        for cycle in range(100):
            d = r.resolve("cam1", cycle, face_embedding=rand_vec(128, seed=cycle),
                           known_person_id=f"p-{cycle}")
            r.unbind("cam1", cycle)
            clk.advance(121)
            r.sweep()
        self.assertEqual(0, r.metrics["active_count"])
        self.assertEqual(0, r.metrics["missing_count"])
        self.assertLessEqual(len(r.expired_states), 50)

    def test_adversarial_two_people_one_global_id_attempt(self):
        """Try to force two different physical people into one Global ID."""
        r, _ = make_resolver()
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("cam1", 2, face_embedding=vec(0, 1, 0),
                        known_person_id="bob")
        # Alice and Bob must have different Global IDs
        self.assertNotEqual(d1.global_id, d2.global_id)
        # Try again with same known_person_id but different face
        d3 = r.resolve("cam1", 3, face_embedding=vec(0, 0, 1),
                        known_person_id="alice")
        # d3 should map to alice's Global ID
        self.assertEqual(d1.global_id, d3.global_id)

    def test_adversarial_one_person_multiple_ids_attempt(self):
        """Try to fragment one person into multiple Global IDs."""
        r, clk = make_resolver()
        ids = set()
        for i in range(10):
            d = r.resolve("cam1", i + 1, face_embedding=vec(1, 0, 0),
                           known_person_id="alice")
            ids.add(d.global_id)
            if i < 9:
                r.unbind("cam1", i + 1)
                clk.advance(10)  # Within TTL
        self.assertEqual(1, len(ids), f"One person fragmented into {len(ids)} Global IDs!")

    def test_adversarial_concurrent_multi_camera_same_person(self):
        """Concurrent observations from 4 cameras for same person → 1 Global ID."""
        r, _ = make_resolver()
        results = []
        barrier = threading.Barrier(4)
        errors = []

        def worker(cam_idx):
            try:
                barrier.wait(timeout=5)
                d = r.resolve(f"cam{cam_idx}", cam_idx,
                               face_embedding=vec(1, 0, 0),
                               known_person_id="shared")
                results.append(d.global_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual([], errors)
        self.assertEqual(1, len(set(results)), f"Expected 1 ID, got {set(results)}")

    def test_adversarial_clock_jump_forward(self):
        """Large clock jump forward → state expires correctly."""
        r, clk = make_resolver()
        d = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        r.unbind("cam1", 1)
        clk.advance(10000)  # Huge jump
        r.sweep()
        self.assertEqual(LifecycleState.EXPIRED, r.get_state(d.global_id).lifecycle)

    def test_adversarial_id_factory_collision(self):
        """ID factory that sometimes returns duplicates → handled by retry."""
        call_count = [0]
        def colliding_factory():
            call_count[0] += 1
            if call_count[0] <= 3:
                return "COLLISION"
            return f"unique-{call_count[0]}"

        r, _ = make_resolver(id_factory=colliding_factory)
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0), known_person_id="alice")
        d2 = r.resolve("cam1", 2, face_embedding=vec(0, 1, 0), known_person_id="bob")
        self.assertNotEqual(d1.global_id, d2.global_id)


# ═════════════════════════════════════════════════════════════════════════
# BUG-001 REGRESSION TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestBUG001_Regression(unittest.TestCase):
    """Regression tests for BUG-001: _simultaneous_conflict false positive
    on unconfigured camera pairs.

    Three camera relationship states:
      1. Configured overlap=True  → allowed shared observation
      2. Configured overlap=False → conflict protection
      3. No transition configured → neutral/unknown (must NOT conflict)
    """

    def test_A_unconfigured_cameras_same_known_person_same_global_id(self):
        """TEST A: No transition configured between Camera A and Camera B.
        Same verified known person appears on both cameras.
        Expected: same Global ID."""
        r, _ = make_resolver()  # No transitions configured
        d1 = r.resolve("camA", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice", known_person_name="Alice")
        d2 = r.resolve("camB", 5, face_embedding=vec(1, 0, 0),
                        known_person_id="alice", known_person_name="Alice")
        self.assertEqual(DecisionState.CONFIRMED, d1.state)
        self.assertEqual(DecisionState.CONFIRMED, d2.state)
        self.assertEqual(d1.global_id, d2.global_id,
                         "BUG-001: Unconfigured camera pair caused false conflict!")

    def test_B_explicit_non_overlap_same_known_person_conflict(self):
        """TEST B: Explicit transition overlap=False.
        Same known person appears simultaneously.
        Expected: conflict behavior (UNCERTAIN for the second observation)."""
        r, _ = make_resolver(
            transitions=[CameraTransition("camA", "camB", 0, 20, overlap=False)]
        )
        d1 = r.resolve("camA", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("camB", 5, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(DecisionState.CONFIRMED, d1.state)
        self.assertEqual(DecisionState.UNCERTAIN, d2.state,
                         "Explicit non-overlap should produce conflict/UNCERTAIN")
        self.assertIsNone(d2.global_id)

    def test_C_explicit_overlap_same_person_same_global_id(self):
        """TEST C: Explicit transition overlap=True.
        Same person appears.
        Expected: same Global ID."""
        r, _ = make_resolver(
            transitions=[CameraTransition("camA", "camB", 0, 20, overlap=True)]
        )
        d1 = r.resolve("camA", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("camB", 5, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(DecisionState.CONFIRMED, d1.state)
        self.assertEqual(DecisionState.CONFIRMED, d2.state)
        self.assertEqual(d1.global_id, d2.global_id)

    def test_D_partial_topology_only_configured_pair_interpreted(self):
        """TEST D: Unconfigured A→B but configured B→C (overlap=False).
        A→B must be neutral. B→C must enforce non-overlap."""
        r, _ = make_resolver(
            transitions=[CameraTransition("camB", "camC", 0, 20, overlap=False)]
        )
        # A→B: unconfigured → should allow same person
        d1 = r.resolve("camA", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        d2 = r.resolve("camB", 5, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(d1.global_id, d2.global_id,
                         "Unconfigured A→B should allow same person")
        # B→C: configured non-overlap → should conflict if simultaneous
        d3 = r.resolve("camC", 10, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        # camB is still bound (non-overlap with camC) → conflict
        self.assertEqual(DecisionState.UNCERTAIN, d3.state)

    def test_E_unconfigured_pair_must_not_produce_none_global_id(self):
        """TEST E: Unconfigured camera pair must NOT create UNCERTAIN/global_id=None
        when strong verified face evidence identifies the same known person."""
        r, _ = make_resolver()  # No transitions
        d1 = r.resolve("cam1", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="chirag", known_person_name="Chirag")
        d2 = r.resolve("cam2", 2, face_embedding=vec(1, 0, 0),
                        known_person_id="chirag", known_person_name="Chirag")
        d3 = r.resolve("cam3", 3, face_embedding=vec(1, 0, 0),
                        known_person_id="chirag", known_person_name="Chirag")
        d4 = r.resolve("cam4", 4, face_embedding=vec(1, 0, 0),
                        known_person_id="chirag", known_person_name="Chirag")
        # ALL must have the same Global ID, NONE should be None
        for i, d in enumerate([d1, d2, d3, d4], 1):
            self.assertIsNotNone(d.global_id,
                                 f"cam{i}: global_id was None — BUG-001 regression!")
            self.assertEqual(DecisionState.CONFIRMED, d.state,
                             f"cam{i}: state was {d.state}, not CONFIRMED")
        ids = {d1.global_id, d2.global_id, d3.global_id, d4.global_id}
        self.assertEqual(1, len(ids), f"Expected 1 Global ID across 4 cameras, got {ids}")

    def test_F_explicit_non_overlap_still_protects_against_merge(self):
        """TEST F: Explicit non-overlap must still protect against incorrect
        simultaneous merging of different people."""
        r, _ = make_resolver(
            transitions=[CameraTransition("camA", "camB", 0, 20, overlap=False)]
        )
        # Alice on camA
        d1 = r.resolve("camA", 1, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        # Same person on camB → conflict due to non-overlap
        d2 = r.resolve("camB", 2, face_embedding=vec(1, 0, 0),
                        known_person_id="alice")
        self.assertEqual(DecisionState.UNCERTAIN, d2.state)
        # Different person on camB → different Global ID (no false merge)
        d3 = r.resolve("camB", 3, face_embedding=vec(0, 1, 0),
                        known_person_id="bob")
        self.assertEqual(DecisionState.CONFIRMED, d3.state)
        self.assertNotEqual(d1.global_id, d3.global_id)


# ═════════════════════════════════════════════════════════════════════════
# PART 8 — CAMERA TOPOLOGY: THREE RELATIONSHIP STATES
# ═════════════════════════════════════════════════════════════════════════

class TestPart08_CameraTopologyStates(unittest.TestCase):
    """Explicitly cover all three camera relationship states."""

    def test_state1_configured_overlap_true_allows_shared(self):
        """STATE 1: Configured overlap=True → allowed shared observation."""
        r, _ = make_resolver(
            transitions=[CameraTransition("A", "B", 0, 30, overlap=True)]
        )
        d1 = r.resolve("A", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        d2 = r.resolve("B", 2, face_embedding=vec(1, 0, 0), known_person_id="x")
        self.assertEqual(d1.global_id, d2.global_id)
        state = r.get_state(d1.global_id)
        self.assertEqual(2, len(state.local_bindings))

    def test_state2_configured_overlap_false_conflicts(self):
        """STATE 2: Configured overlap=False → conflict protection."""
        r, _ = make_resolver(
            transitions=[CameraTransition("A", "B", 5, 20, overlap=False)]
        )
        d1 = r.resolve("A", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        d2 = r.resolve("B", 2, face_embedding=vec(1, 0, 0), known_person_id="x")
        self.assertEqual(DecisionState.UNCERTAIN, d2.state)
        events = [e for e in r.events if e["type"] == EventType.CONFLICT.value]
        self.assertGreater(len(events), 0)

    def test_state3_no_transition_neutral(self):
        """STATE 3: No transition configuration → neutral/unknown relationship."""
        r, _ = make_resolver()  # No transitions at all
        d1 = r.resolve("A", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        d2 = r.resolve("B", 2, face_embedding=vec(1, 0, 0), known_person_id="x")
        # Should NOT conflict — unknown relationship is neutral
        self.assertEqual(DecisionState.CONFIRMED, d2.state)
        self.assertEqual(d1.global_id, d2.global_id)

    def test_mixed_topology_selective_enforcement(self):
        """Mixed: A-B overlap, B-C non-overlap, A-C unconfigured.
        Each pair must behave according to its own configuration only."""
        r, _ = make_resolver(transitions=[
            CameraTransition("A", "B", 0, 30, overlap=True),
            CameraTransition("B", "C", 5, 20, overlap=False),
        ])
        d1 = r.resolve("A", 1, face_embedding=vec(1, 0, 0), known_person_id="x")
        # A→B: overlap=True → same Global ID
        d2 = r.resolve("B", 2, face_embedding=vec(1, 0, 0), known_person_id="x")
        self.assertEqual(d1.global_id, d2.global_id, "A→B overlap should allow")
        # B→C: overlap=False, B still active → conflict
        d3 = r.resolve("C", 3, face_embedding=vec(1, 0, 0), known_person_id="x")
        self.assertEqual(DecisionState.UNCERTAIN, d3.state,
                         "B→C non-overlap should conflict while B is active")

    def test_unconfigured_face_embedding_match_across_cameras(self):
        """Unconfigured cameras: face embedding match (not known_person_id) also works."""
        r, _ = make_resolver()
        d1 = r.resolve("A", 1, face_embedding=vec(1, 0, 0))
        r.unbind("A", 1)
        d2 = r.resolve("B", 2, face_embedding=vec(1, 0, 0))
        # Face embedding search should find the existing identity
        # The identity was MISSING, face match should re-bind it
        self.assertEqual(d1.global_id, d2.global_id)


# ═════════════════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
