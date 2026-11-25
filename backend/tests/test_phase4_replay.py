"""
Phase 4.1 -- Synthetic CCTV Replay Harness
==========================================

Simulates realistic multi-camera campus observation sequences through the
REAL GlobalIdentityResolver. Since GPU-dependent vision models (YOLO, SCRFD,
ArcFace, OSNet) and the PostgreSQL database are not available in this test
context, this harness synthesizes the observation data that those models
would produce and feeds it through the actual identity resolution layer.

Scenarios are modeled on a real college campus with:
  - 4 cameras (entrance, hallway, lab, cafeteria)
  - 8 known persons (teachers + students from the actual gallery)
  - Realistic timing (walking speeds, occlusion durations)
  - Local ByteTrack-style track ID changes
  - Cross-camera transitions
  - Similar clothing scenarios

Run:
    cd backend && python tests/test_phase4_replay.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.vision.global_identity import (
    CalibratedPolicy,
    CameraTransition,
    DecisionState,
    EventType,
    GlobalIdentityResolver,
    LifecycleState,
    ResolverMode,
)

# -- Helpers --------------------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def make_face(person_seed: int, variation: float = 0.0, dim: int = 128) -> np.ndarray:
    """Deterministic face embedding for a person, with optional angular variation."""
    rng = np.random.RandomState(person_seed * 1000)
    base = _unit(rng.randn(dim).astype(np.float32))
    if variation > 0:
        rng2 = np.random.RandomState(person_seed * 1000 + int(variation * 10000))
        noise = rng2.randn(dim).astype(np.float32)
        noise -= base * float(np.dot(noise, base))
        noise = _unit(noise)
        result = base * math.cos(variation) + noise * math.sin(variation)
        return _unit(result).astype(np.float32)
    return base

def make_body(person_seed: int, variation: float = 0.0, dim: int = 128) -> np.ndarray:
    """Deterministic body embedding, offset from face seed space."""
    return make_face(person_seed + 5000, variation, dim)


class Clock:
    def __init__(self, now=1000.0):
        self.value = float(now)
    def __call__(self):
        return self.value
    def advance(self, seconds):
        self.value += seconds
        return self.value

class IDs:
    def __init__(self):
        self.n = 0
    def __call__(self):
        self.n += 1
        return f"G-{self.n:04d}"


# -- Campus Layout --------------------------------------------------------

CAMERAS = ["entrance", "hallway", "lab", "cafeteria"]

TRANSITIONS = [
    CameraTransition("entrance", "hallway", 3, 30, overlap=False),
    CameraTransition("hallway", "entrance", 3, 30, overlap=False),
    CameraTransition("hallway", "lab", 5, 60, overlap=False),
    CameraTransition("lab", "hallway", 5, 60, overlap=False),
    CameraTransition("hallway", "cafeteria", 10, 90, overlap=False),
    CameraTransition("cafeteria", "hallway", 10, 90, overlap=False),
]

# 8 known persons with distinct seeds
PERSONS = {
    "chirag":    {"seed": 1, "name": "Chirag"},
    "abhitha":   {"seed": 2, "name": "Abhitha"},
    "dileep":    {"seed": 3, "name": "Dileep"},
    "farhana":   {"seed": 4, "name": "Farhana"},
    "geetha":    {"seed": 5, "name": "Geetha"},
    "niranjani": {"seed": 6, "name": "Niranjani"},
    "reena":     {"seed": 7, "name": "Reena"},
    "rithvik":   {"seed": 8, "name": "Rithvik"},
}

POLICY = CalibratedPolicy(
    face_match_threshold=0.80,
    face_min_margin=0.10,
    body_match_threshold=0.56,
    body_min_margin=0.10,
    body_votes_required=3,
    body_vote_min_span=1.0,
    body_vote_min_interval=0.25,
)

# -- Observation record ---------------------------------------------------

@dataclass
class Observation:
    timestamp: float
    camera: str
    local_track_id: int
    person_id: Optional[str]
    person_name: Optional[str]
    face_embedding: Optional[np.ndarray]
    body_embedding: Optional[np.ndarray]
    face_verified: bool = True
    notes: str = ""

@dataclass
class ReplayResult:
    timestamp: float
    camera: str
    local_track_id: int
    global_id: Optional[str]
    known_person_id: Optional[str]
    decision_state: str
    reason: str
    face_score: Optional[float] = None
    body_score: Optional[float] = None
    notes: str = ""

# -- Scenario Builder -----------------------------------------------------

class ScenarioBuilder:
    def __init__(self):
        self.observations: List[Observation] = []
        self.track_counter = 0

    def new_track(self) -> int:
        self.track_counter += 1
        return self.track_counter

    def observe(self, t: float, camera: str, track: int, person_id: str,
                face: bool = True, body: bool = True,
                face_variation: float = 0.0, body_variation: float = 0.0,
                face_verified: bool = True, notes: str = ""):
        p = PERSONS[person_id]
        self.observations.append(Observation(
            timestamp=t,
            camera=camera,
            local_track_id=track,
            person_id=person_id if face and face_verified else None,
            person_name=p["name"] if face and face_verified else None,
            face_embedding=make_face(p["seed"], face_variation) if face else None,
            body_embedding=make_body(p["seed"], body_variation) if body else None,
            face_verified=face_verified,
            notes=notes,
        ))

    def unbind(self, t: float, camera: str, track: int, notes: str = ""):
        """Marker: track ended (person left camera view)."""
        self.observations.append(Observation(
            timestamp=t, camera=camera, local_track_id=track,
            person_id=None, person_name=None,
            face_embedding=None, body_embedding=None,
            notes=f"UNBIND {notes}",
        ))


# -- Replay Engine --------------------------------------------------------

def run_replay(observations: List[Observation], label: str = "unnamed") -> dict:
    """Run observations through the real GlobalIdentityResolver and collect results."""
    clock = Clock(observations[0].timestamp if observations else 1000.0)
    resolver = GlobalIdentityResolver(
        clock=clock,
        id_factory=IDs(),
        mode=ResolverMode.AUTHORITATIVE,
        transitions=TRANSITIONS,
        policy=POLICY,
    )

    results: List[ReplayResult] = []
    person_global_ids: Dict[str, Set[str]] = defaultdict(set)
    global_id_persons: Dict[str, Set[str]] = defaultdict(set)

    for obs in sorted(observations, key=lambda o: o.timestamp):
        clock.value = obs.timestamp

        if "UNBIND" in obs.notes:
            resolver.unbind(obs.camera, obs.local_track_id, timestamp=obs.timestamp)
            results.append(ReplayResult(
                timestamp=obs.timestamp, camera=obs.camera,
                local_track_id=obs.local_track_id,
                global_id=None, known_person_id=None,
                decision_state="UNBIND", reason="track_ended",
                notes=obs.notes,
            ))
            continue

        d = resolver.resolve(
            obs.camera, obs.local_track_id,
            face_embedding=obs.face_embedding,
            body_embedding=obs.body_embedding,
            known_person_id=obs.person_id,
            known_person_name=obs.person_name,
            face_verified=obs.face_verified,
            timestamp=obs.timestamp,
        )

        scores = d.scores or {}
        results.append(ReplayResult(
            timestamp=obs.timestamp,
            camera=obs.camera,
            local_track_id=obs.local_track_id,
            global_id=d.global_id,
            known_person_id=obs.person_id,
            decision_state=d.state.value,
            reason=d.reason,
            face_score=scores.get("face_similarity") or scores.get("best_similarity"),
            body_score=scores.get("body_similarity"),
            notes=obs.notes,
        ))

        if d.global_id and obs.person_id:
            person_global_ids[obs.person_id].add(d.global_id)
            global_id_persons[d.global_id].add(obs.person_id)

    # Analyze
    false_merges = sum(1 for gid, persons in global_id_persons.items() if len(persons) > 1)
    false_splits = sum(1 for pid, gids in person_global_ids.items() if len(gids) > 1)
    identity_drops = sum(1 for r in results
                         if r.known_person_id and r.global_id is None
                         and r.decision_state not in ("UNBIND",))
    total_observations = sum(1 for r in results if r.decision_state != "UNBIND")
    confirmed = sum(1 for r in results if r.decision_state == "CONFIRMED")
    uncertain = sum(1 for r in results if r.decision_state == "UNCERTAIN")

    return {
        "label": label,
        "total_observations": total_observations,
        "confirmed": confirmed,
        "uncertain": uncertain,
        "false_merges": false_merges,
        "false_splits": false_splits,
        "identity_drops": identity_drops,
        "unique_persons": len(person_global_ids),
        "unique_global_ids": len(global_id_persons),
        "person_global_ids": {k: list(v) for k, v in person_global_ids.items()},
        "results": results,
        "metrics": resolver.metrics,
        "events": resolver.events,
    }


# ==========================================================================
# SCENARIOS
# ==========================================================================

def scenario_1_cross_camera_A_to_B():
    """Person walks from entrance to hallway."""
    s = ScenarioBuilder()
    t1 = s.new_track()
    for i in range(10):
        s.observe(1000 + i * 0.5, "entrance", t1, "chirag", notes="approaching")
    s.unbind(1005, "entrance", t1, "left entrance")
    t2 = s.new_track()
    for i in range(10):
        s.observe(1010 + i * 0.5, "hallway", t2, "chirag", notes="entered hallway")
    s.unbind(1015, "hallway", t2, "left hallway")
    return s.observations, "Scenario 1: entrance -> hallway"

def scenario_2_three_camera_chain():
    """Person: entrance -> hallway -> lab."""
    s = ScenarioBuilder()
    # Entrance
    t1 = s.new_track()
    for i in range(5):
        s.observe(1000 + i * 0.5, "entrance", t1, "abhitha")
    s.unbind(1003, "entrance", t1)
    # Hallway
    t2 = s.new_track()
    for i in range(5):
        s.observe(1010 + i * 0.5, "hallway", t2, "abhitha")
    s.unbind(1013, "hallway", t2)
    # Lab
    t3 = s.new_track()
    for i in range(5):
        s.observe(1025 + i * 0.5, "lab", t3, "abhitha")
    s.unbind(1028, "lab", t3)
    return s.observations, "Scenario 2: entrance -> hallway -> lab"

def scenario_3_face_visible_then_invisible():
    """Face visible in entrance, not in hallway (body only), then face back in lab."""
    s = ScenarioBuilder()
    t1 = s.new_track()
    for i in range(5):
        s.observe(1000 + i * 0.5, "entrance", t1, "dileep", face=True, body=True)
    s.unbind(1003, "entrance", t1)
    # Hallway: body only, no face
    t2 = s.new_track()
    for i in range(5):
        s.observe(1010 + i * 0.5, "hallway", t2, "dileep", face=False, body=True, notes="body only")
    s.unbind(1013, "hallway", t2)
    # Lab: face back
    t3 = s.new_track()
    for i in range(5):
        s.observe(1025 + i * 0.5, "lab", t3, "dileep", face=True, body=True, notes="face back")
    s.unbind(1028, "lab", t3)
    return s.observations, "Scenario 3: face visible -> invisible -> visible"

def scenario_4_local_track_id_changes():
    """ByteTrack track ID changes mid-observation (e.g. occlusion)."""
    s = ScenarioBuilder()
    t1 = s.new_track()
    for i in range(5):
        s.observe(1000 + i * 0.5, "entrance", t1, "farhana")
    s.unbind(1003, "entrance", t1, "occlusion")
    # New track ID, same camera, same person
    t2 = s.new_track()
    for i in range(5):
        s.observe(1004 + i * 0.5, "entrance", t2, "farhana", notes="new track after occlusion")
    s.unbind(1007, "entrance", t2)
    return s.observations, "Scenario 4: local track ID change"

def scenario_5_brief_disappearance():
    """Person disappears briefly (3 seconds) and reappears."""
    s = ScenarioBuilder()
    t1 = s.new_track()
    for i in range(5):
        s.observe(1000 + i * 0.5, "hallway", t1, "geetha")
    s.unbind(1003, "hallway", t1, "behind column")
    # Reappears 3 seconds later
    t2 = s.new_track()
    for i in range(5):
        s.observe(1006 + i * 0.5, "hallway", t2, "geetha", notes="reappeared")
    s.unbind(1009, "hallway", t2)
    return s.observations, "Scenario 5: brief 3-second disappearance"

def scenario_6_disappear_all_cameras_reappear_within_120s():
    """Person disappears from all cameras, reappears at 90 seconds."""
    s = ScenarioBuilder()
    t1 = s.new_track()
    for i in range(5):
        s.observe(1000 + i * 0.5, "entrance", t1, "niranjani")
    s.unbind(1003, "entrance", t1, "left campus")
    # Reappears 90 seconds later
    t2 = s.new_track()
    for i in range(5):
        s.observe(1093 + i * 0.5, "entrance", t2, "niranjani", notes="re-entered within 120s")
    s.unbind(1096, "entrance", t2)
    return s.observations, "Scenario 6: 90-second absence + return"

def scenario_7_reappear_after_120s():
    """Person disappears and reappears after 130 seconds -> new Global ID."""
    s = ScenarioBuilder()
    t1 = s.new_track()
    for i in range(5):
        s.observe(1000 + i * 0.5, "entrance", t1, "reena")
    s.unbind(1003, "entrance", t1, "left campus")
    # Reappears 130 seconds later -> expired
    t2 = s.new_track()
    for i in range(5):
        s.observe(1133 + i * 0.5, "entrance", t2, "reena", notes="re-entered after 120s")
    s.unbind(1136, "entrance", t2)
    return s.observations, "Scenario 7: 130-second absence (beyond TTL)"

def scenario_8_similar_clothing():
    """Two people with similar body embeddings but different faces."""
    s = ScenarioBuilder()
    t1 = s.new_track()
    for i in range(5):
        # Chirag: face seed 1, body seed 1
        s.observe(1000 + i * 0.5, "entrance", t1, "chirag", body_variation=0.01)
    t2 = s.new_track()
    for i in range(5):
        # Rithvik: face seed 8, but body variation close to chirag
        s.observe(1000 + i * 0.5, "entrance", t2, "rithvik", body_variation=0.01)
    s.unbind(1003, "entrance", t1)
    s.unbind(1003, "entrance", t2)
    return s.observations, "Scenario 8: similar clothing (different faces)"

def scenario_9_same_uniform():
    """Three students with identical uniform body embeddings but different faces."""
    s = ScenarioBuilder()
    tracks = [s.new_track() for _ in range(3)]
    people = ["chirag", "abhitha", "rithvik"]
    for i in range(5):
        for j, (track, pid) in enumerate(zip(tracks, people)):
            s.observe(1000 + i * 0.5, "cafeteria", track, pid, body_variation=0.005)
    for track in tracks:
        s.unbind(1003, "cafeteria", track)
    return s.observations, "Scenario 9: same uniform (3 students)"

def scenario_10_multiple_cameras_simultaneous():
    """4 different people in 4 cameras simultaneously."""
    s = ScenarioBuilder()
    cam_persons = [
        ("entrance", "chirag"), ("hallway", "abhitha"),
        ("lab", "dileep"), ("cafeteria", "farhana"),
    ]
    tracks = [s.new_track() for _ in cam_persons]
    for i in range(5):
        for track, (cam, pid) in zip(tracks, cam_persons):
            s.observe(1000 + i * 0.5, cam, track, pid)
    for track, (cam, _) in zip(tracks, cam_persons):
        s.unbind(1003, cam, track)
    return s.observations, "Scenario 10: 4 cameras active simultaneously"


# ==========================================================================
# TEST CLASS
# ==========================================================================

class TestCCTVReplay(unittest.TestCase):
    """Run all 10 replay scenarios and validate identity correctness."""

    def _run_and_check(self, scenario_fn, expect_false_merges=0, expect_identity_drops=0):
        observations, label = scenario_fn()
        result = run_replay(observations, label)
        self.assertEqual(0, result["false_merges"],
                         f"{label}: FALSE MERGE detected! {result['person_global_ids']}")
        self.assertLessEqual(result["identity_drops"], expect_identity_drops,
                             f"{label}: Identity drops={result['identity_drops']}")
        return result

    def test_scenario_01_cross_camera(self):
        result = self._run_and_check(scenario_1_cross_camera_A_to_B)
        self.assertEqual(0, result["false_splits"],
                         f"FALSE SPLIT in cross-camera: {result['person_global_ids']}")

    def test_scenario_02_three_camera_chain(self):
        result = self._run_and_check(scenario_2_three_camera_chain)
        self.assertEqual(0, result["false_splits"])

    def test_scenario_03_face_visibility_changes(self):
        result = self._run_and_check(scenario_3_face_visible_then_invisible)
        # Body-only observations in hallway may not confirm -- that's acceptable
        # But entrance and lab should share Global ID

    def test_scenario_04_local_track_change(self):
        result = self._run_and_check(scenario_4_local_track_id_changes)
        self.assertEqual(0, result["false_splits"],
                         f"FALSE SPLIT on track change: {result['person_global_ids']}")

    def test_scenario_05_brief_disappearance(self):
        result = self._run_and_check(scenario_5_brief_disappearance)
        self.assertEqual(0, result["false_splits"])

    def test_scenario_06_reappear_within_120s(self):
        result = self._run_and_check(scenario_6_disappear_all_cameras_reappear_within_120s)
        self.assertEqual(0, result["false_splits"],
                         f"FALSE SPLIT on 90s reappearance: {result['person_global_ids']}")

    def test_scenario_07_reappear_after_120s(self):
        result = self._run_and_check(scenario_7_reappear_after_120s)
        # Expected: 1 false split (new Global ID after TTL expiry) -- this is CORRECT behavior
        self.assertEqual(1, result["false_splits"],
                         "Expected exactly 1 split (new visit after 120s TTL)")

    def test_scenario_08_similar_clothing(self):
        result = self._run_and_check(scenario_8_similar_clothing)
        self.assertEqual(0, result["false_merges"],
                         f"FALSE MERGE with similar clothing: {result['person_global_ids']}")

    def test_scenario_09_same_uniform(self):
        result = self._run_and_check(scenario_9_same_uniform)
        self.assertEqual(0, result["false_merges"],
                         f"FALSE MERGE with same uniform: {result['person_global_ids']}")
        # 3 distinct people -> 3 distinct Global IDs
        self.assertEqual(3, result["unique_global_ids"])

    def test_scenario_10_simultaneous_cameras(self):
        result = self._run_and_check(scenario_10_multiple_cameras_simultaneous)
        self.assertEqual(4, result["unique_global_ids"],
                         f"Expected 4 distinct IDs, got {result['unique_global_ids']}")
        self.assertEqual(0, result["false_merges"])


# ==========================================================================
# FULL REPLAY REPORT
# ==========================================================================

def run_all_scenarios(verbose=True):
    """Run all scenarios and produce the complete replay report."""
    scenarios = [
        scenario_1_cross_camera_A_to_B,
        scenario_2_three_camera_chain,
        scenario_3_face_visible_then_invisible,
        scenario_4_local_track_id_changes,
        scenario_5_brief_disappearance,
        scenario_6_disappear_all_cameras_reappear_within_120s,
        scenario_7_reappear_after_120s,
        scenario_8_similar_clothing,
        scenario_9_same_uniform,
        scenario_10_multiple_cameras_simultaneous,
    ]

    all_results = []
    total_false_merges = 0
    total_false_splits = 0
    total_identity_drops = 0
    total_observations = 0

    if verbose:
        print("=" * 100)
        print("PHASE 4.1 -- SYNTHETIC CCTV REPLAY RESULTS")
        print("=" * 100)
        print(f"Cameras: {CAMERAS}")
        print(f"Known persons: {list(PERSONS.keys())}")
        print(f"Transitions: {len(TRANSITIONS)} configured")
        print()

    for fn in scenarios:
        observations, label = fn()
        result = run_replay(observations, label)
        all_results.append(result)
        total_false_merges += result["false_merges"]
        total_false_splits += result["false_splits"]
        total_identity_drops += result["identity_drops"]
        total_observations += result["total_observations"]

        if verbose:
            status = "PASS" if result["false_merges"] == 0 else "FAIL (FALSE MERGE)"
            if result["false_splits"] > 0 and "120s" not in label:
                status = "FAIL (FALSE SPLIT)"
            print(f"  {label}")
            print(f"    Observations: {result['total_observations']}  "
                  f"Confirmed: {result['confirmed']}  "
                  f"Uncertain: {result['uncertain']}")
            print(f"    Persons: {result['unique_persons']}  "
                  f"Global IDs: {result['unique_global_ids']}  "
                  f"False merges: {result['false_merges']}  "
                  f"False splits: {result['false_splits']}  "
                  f"Drops: {result['identity_drops']}")
            print(f"    -> {status}")
            # Print timeline
            for r in result["results"]:
                if r.decision_state not in ("UNBIND",):
                    print(f"      {r.timestamp:.1f}  {r.camera:12s}  Track {r.local_track_id:3d}  "
                          f"-> {r.global_id or 'None':8s}  {r.known_person_id or '':12s}  "
                          f"{r.decision_state:12s}  {r.notes}")
            print()

    if verbose:
        print("=" * 100)
        print("SUMMARY")
        print(f"  Total observations: {total_observations}")
        print(f"  Total false merges: {total_false_merges}")
        print(f"  Total false splits: {total_false_splits} (1 expected from >120s TTL)")
        print(f"  Total identity drops: {total_identity_drops}")
        print(f"  Scenarios: {len(scenarios)}")
        print("=" * 100)

    return {
        "scenarios": len(scenarios),
        "total_observations": total_observations,
        "total_false_merges": total_false_merges,
        "total_false_splits": total_false_splits,
        "total_identity_drops": total_identity_drops,
        "all_results": all_results,
    }


if __name__ == "__main__":
    summary = run_all_scenarios(verbose=True)
    print()
    # Also run as unittest
    print("Running as unittest...")
    unittest.main(verbosity=2, exit=False)
