"""Phase 4 global identity core tests (stdlib unittest + numpy only)."""

import threading
import time
import unittest

import numpy as np

from app.vision.global_identity import (
    CalibratedPolicy,
    CameraTransition,
    DecisionState,
    EventType,
    GlobalIdentityResolver,
    LifecycleState,
    ResolverMode,
)


def vec(*values):
    value = np.asarray(values, dtype=np.float32)
    return value / np.linalg.norm(value)


class Clock:
    def __init__(self, now=100.0):
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
        return "global-%03d" % self.n


# Acceptance matrix (the numbered test methods below are intentionally stable):
# 01 strong known face; 02 unknown->known; 03 face temporarily lost/body cannot
# overwrite; 04 weak one-frame body; 05 repeated body->face/candidate; 06 directed
# topology; 07 travel time; 08 similar clothing ambiguity; 09 uncalibrated fail
# closed; 10 non-overlap contradiction; 11 overlap; 12 multi-camera lifecycle;
# 13 deterministic 10/60/119/121 reappearance + local ID changes; 14 bounded
# galleries; 15 shadow/authoritative audit, event schema, metrics and latency.
ACCEPTANCE_MATRIX = {
    1: "strong_known_face", 2: "unknown_becomes_known", 3: "face_temporarily_lost",
    4: "weak_one_frame_body", 5: "body_to_face", 6: "directed_transition",
    7: "travel_window", 8: "similar_clothing", 9: "uncalibrated_policy",
    10: "contradiction_non_overlap", 11: "overlap", 12: "multi_camera_timer",
    13: "reappearance_boundaries_local_track_change", 14: "bounded_galleries",
    15: "mode_events_audit_metrics_latency",
}


class GlobalIdentityScenarios(unittest.TestCase):
    def make_resolver(self, *, mode="shadow", transitions=(), policy=None, **kwargs):
        clock = kwargs.pop("clock", Clock())
        resolver = GlobalIdentityResolver(
            clock=clock,
            id_factory=IDs(),
            mode=mode,
            transitions=transitions,
            policy=policy or CalibratedPolicy(
                face_match_threshold=0.80,
                face_min_margin=0.10,
                body_min_margin=0.10,
                body_votes_required=3,
                body_vote_min_span=1.0,
                body_vote_min_interval=0.25,
            ),
            **kwargs,
        )
        return resolver, clock

    # 1
    def test_matrix_01_strong_known_face_creates_stable_identity(self):
        resolver, _ = self.make_resolver()
        first = resolver.resolve(
            "entrance", 1, face_embedding=vec(1, 0, 0),
            known_person_id="student-7", known_person_name="Asha",
        )
        second = resolver.resolve(
            "entrance", 2, face_embedding=vec(0.99, 0.01, 0),
            known_person_id="student-7", known_person_name="Asha",
        )
        self.assertEqual(DecisionState.CONFIRMED, first.state)
        self.assertEqual(first.global_id, second.global_id)
        self.assertEqual("student-7", resolver.get_state(first.global_id).known_person_id)

    # 2
    def test_matrix_02_unknown_can_become_known_without_global_id_change(self):
        resolver, _ = self.make_resolver()
        unknown = resolver.resolve("hall", 11, face_embedding=vec(1, 0, 0))
        enrolled = resolver.resolve(
            "hall", 11, face_embedding=vec(1, 0, 0),
            known_person_id="teacher-3", known_person_name="Ravi",
        )
        self.assertEqual(DecisionState.NEW, unknown.state)
        self.assertEqual(unknown.global_id, enrolled.global_id)
        self.assertEqual("teacher-3", resolver.get_state(unknown.global_id).known_person_id)

    # 3
    def test_matrix_03_face_temporarily_lost_body_never_overwrites_known_binding(self):
        resolver, clock = self.make_resolver()
        one = resolver.resolve(
            "a", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
            known_person_id="known-a",
        )
        other = resolver.resolve(
            "a", 2, face_embedding=vec(0, 1, 0), body_embedding=vec(0, 1, 0),
            known_person_id="known-b",
        )
        clock.advance(1)
        result = resolver.resolve("a", 1, body_embedding=vec(0, 1, 0))
        self.assertEqual(DecisionState.CONFIRMED, result.state)
        self.assertEqual(one.global_id, result.global_id)
        self.assertNotEqual(other.global_id, result.global_id)

    # 4
    def test_matrix_04_weak_one_frame_body_is_not_assigned(self):
        transition = CameraTransition("a", "b", 1, 20)
        resolver, clock = self.make_resolver(transitions=[transition])
        identity = resolver.resolve("a", 1, face_embedding=vec(1, 0), body_embedding=vec(1, 0))
        resolver.unbind("a", 1)
        clock.advance(2)
        result = resolver.resolve("b", 9, body_embedding=vec(0.7, 0.714))
        self.assertEqual(DecisionState.UNCERTAIN, result.state)
        self.assertIsNone(result.global_id)
        self.assertEqual(identity.global_id, result.candidate_id)
        self.assertIsNone(resolver.global_id_for("b", 9))

    # 5
    def test_matrix_05_repeated_body_to_face_candidate_is_not_confirmation(self):
        transition = CameraTransition("a", "b", 1, 20)
        resolver, clock = self.make_resolver(transitions=[transition])
        known = resolver.resolve(
            "a", 1, face_embedding=vec(1, 0), body_embedding=vec(1, 0),
            known_person_id="known-a",
        )
        resolver.unbind("a", 1)
        clock.advance(2)
        results = []
        for _ in range(3):
            results.append(resolver.resolve("b", 9, body_embedding=vec(1, 0)))
            clock.advance(0.6)
        self.assertEqual(DecisionState.CANDIDATE, results[-1].state)
        self.assertEqual(known.global_id, results[-1].candidate_id)
        self.assertIsNone(results[-1].global_id)
        self.assertIsNone(resolver.global_id_for("b", 9))
        # A subsequent authoritative face, not the body candidate, performs link.
        face = resolver.resolve(
            "b", 9, face_embedding=vec(1, 0), body_embedding=vec(1, 0),
            known_person_id="known-a",
        )
        self.assertEqual(DecisionState.CONFIRMED, face.state)
        self.assertEqual(known.global_id, face.global_id)

    # 6
    def test_matrix_06_body_candidate_requires_configured_directed_transition(self):
        resolver, clock = self.make_resolver(
            transitions=[CameraTransition("b", "a", 0, 20)]
        )
        resolver.resolve("a", 1, face_embedding=vec(1, 0), body_embedding=vec(1, 0))
        resolver.unbind("a", 1)
        clock.advance(2)
        result = resolver.resolve("b", 1, body_embedding=vec(1, 0))
        self.assertEqual(DecisionState.UNCERTAIN, result.state)
        self.assertEqual("camera_transition_not_configured", result.reason)

    # 7
    def test_matrix_07_transition_travel_window_is_enforced(self):
        resolver, clock = self.make_resolver(
            transitions=[CameraTransition("a", "b", 5, 10)]
        )
        resolver.resolve("a", 1, face_embedding=vec(1, 0), body_embedding=vec(1, 0))
        resolver.unbind("a", 1)
        clock.advance(2)
        early = resolver.resolve("b", 1, body_embedding=vec(1, 0))
        clock.advance(10)
        late = resolver.resolve("b", 2, body_embedding=vec(1, 0))
        self.assertEqual("camera_transition_time_invalid", early.reason)
        self.assertEqual("camera_transition_time_invalid", late.reason)

    # 8
    def test_matrix_08_similar_clothing_ambiguous_body_candidates_are_uncertain(self):
        resolver, clock = self.make_resolver(
            transitions=[CameraTransition("a", "b", 0, 20)],
            policy=CalibratedPolicy(
                face_match_threshold=0.8,
                face_min_margin=0.1,
                body_min_margin=0.2,
            ),
        )
        resolver.resolve("a", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0))
        resolver.resolve("a", 2, face_embedding=vec(0, 1, 0), body_embedding=vec(0.98, 0.2, 0))
        resolver.unbind("a", 1)
        resolver.unbind("a", 2)
        clock.advance(2)
        result = resolver.resolve("b", 3, body_embedding=vec(1, 0.1, 0))
        self.assertEqual(DecisionState.UNCERTAIN, result.state)
        self.assertEqual("ambiguous_body_margin", result.reason)

    # 9
    def test_matrix_09_absent_calibrated_margin_fails_closed(self):
        resolver, clock = self.make_resolver(
            transitions=[CameraTransition("a", "b", 0, 20)],
            policy=CalibratedPolicy(face_match_threshold=0.8, face_min_margin=None, body_min_margin=None),
        )
        resolver.resolve("a", 1, face_embedding=vec(1, 0), body_embedding=vec(1, 0))
        resolver.unbind("a", 1)
        clock.advance(1)
        body = resolver.resolve("b", 1, body_embedding=vec(1, 0))
        face = resolver.resolve("b", 2, face_embedding=vec(1, 0))
        self.assertEqual("body_margin_uncalibrated", body.reason)
        self.assertEqual("face_policy_uncalibrated", face.reason)
        self.assertEqual(DecisionState.UNCERTAIN, face.state)

    # 10
    def test_matrix_10_contradiction_simultaneous_non_overlap_conflict_rejected(self):
        resolver, _ = self.make_resolver(
            transitions=[CameraTransition("a", "b", 0, 20, overlap=False)]
        )
        known = resolver.resolve(
            "a", 1, face_embedding=vec(1, 0), known_person_id="person-a"
        )
        conflict = resolver.resolve(
            "b", 2, face_embedding=vec(1, 0), known_person_id="person-a"
        )
        self.assertEqual(DecisionState.UNCERTAIN, conflict.state)
        self.assertIsNone(conflict.global_id)
        self.assertEqual(known.global_id, conflict.candidate_id)
        self.assertIn(EventType.CONFLICT.value, [e["type"] for e in resolver.events])

    # 11
    def test_matrix_11_overlap_topology_permits_simultaneous_face_confirmation(self):
        resolver, _ = self.make_resolver(
            transitions=[CameraTransition("a", "b", 0, 20, overlap=True)]
        )
        first = resolver.resolve("a", 1, face_embedding=vec(1, 0), known_person_id="p")
        second = resolver.resolve("b", 2, face_embedding=vec(1, 0), known_person_id="p")
        self.assertEqual(DecisionState.CONFIRMED, second.state)
        self.assertEqual(first.global_id, second.global_id)
        self.assertEqual(2, len(resolver.get_state(first.global_id).local_bindings))

    # 12
    def test_matrix_12_lifecycle_timer_starts_only_after_all_camera_bindings_leave(self):
        resolver, clock = self.make_resolver(
            transitions=[CameraTransition("a", "b", 0, 20, overlap=True)]
        )
        first = resolver.resolve("a", 1, face_embedding=vec(1, 0), known_person_id="p")
        resolver.resolve("b", 2, face_embedding=vec(1, 0), known_person_id="p")
        resolver.unbind("a", 1)
        clock.advance(200)
        resolver.sweep()
        state = resolver.get_state(first.global_id)
        self.assertEqual(LifecycleState.ACTIVE, state.lifecycle)
        self.assertIsNone(state.missing_since)
        resolver.unbind("b", 2)
        self.assertEqual(LifecycleState.MISSING, state.lifecycle)
        clock.advance(119.9)
        resolver.sweep()
        self.assertEqual(LifecycleState.MISSING, state.lifecycle)
        clock.advance(0.2)
        resolver.sweep()
        self.assertEqual(LifecycleState.EXPIRED, state.lifecycle)
        self.assertEqual([], state.face_gallery)

    # 13: explicit deterministic timeout matrix and changed local track IDs.
    def test_matrix_13_reappearance_10_60_119_same_and_121_new_global_id(self):
        for delay in (10, 60, 119):
            with self.subTest(delay=delay):
                resolver, clock = self.make_resolver()
                first = resolver.resolve(
                    "a", 1, face_embedding=vec(1, 0), known_person_id="p"
                )
                resolver.unbind("a", 1)
                clock.advance(delay)
                again = resolver.resolve(
                    "a", 100 + delay, face_embedding=vec(1, 0), known_person_id="p"
                )
                self.assertEqual(first.global_id, again.global_id)
                self.assertEqual(
                    LifecycleState.ACTIVE, resolver.get_state(first.global_id).lifecycle
                )
                self.assertIn(
                    EventType.REAPPEARED.value, [event["type"] for event in resolver.events]
                )

        resolver, clock = self.make_resolver()
        first = resolver.resolve("a", 1, face_embedding=vec(1, 0), known_person_id="p")
        resolver.unbind("a", 1)
        clock.advance(121)
        # resolve() sweeps before consulting known_person_id mappings.
        again = resolver.resolve("a", 999, face_embedding=vec(1, 0), known_person_id="p")
        self.assertNotEqual(first.global_id, again.global_id)
        self.assertEqual(LifecycleState.EXPIRED, resolver.get_state(first.global_id).lifecycle)
        self.assertEqual(LifecycleState.ACTIVE, resolver.get_state(again.global_id).lifecycle)
        self.assertEqual("p", resolver.get_state(first.global_id).known_person_id)
        self.assertEqual("p", resolver.get_state(again.global_id).known_person_id)
        self.assertIn(first.global_id, resolver.expired_states)
        self.assertNotIn(EventType.REAPPEARED, again.event_types)

    # 14
    def test_matrix_14_face_and_body_galleries_are_bounded_and_diverse(self):
        resolver, _ = self.make_resolver(face_gallery_size=5, body_gallery_size=6)
        created = resolver.resolve(
            "a", 1, face_embedding=vec(1, 0, 0), body_embedding=vec(1, 0, 0),
            known_person_id="p",
        )
        for i in range(30):
            angle = 0.05 + i * 0.025
            resolver.resolve(
                "a", 1,
                face_embedding=vec(np.cos(angle), np.sin(angle), 0.01 * (i % 3)),
                body_embedding=vec(np.cos(angle), 0.01 * (i % 4), np.sin(angle)),
            )
        state = resolver.get_state(created.global_id)
        self.assertLessEqual(len(state.face_gallery), 5)
        self.assertLessEqual(len(state.body_gallery), 6)
        self.assertGreater(len(state.face_gallery), 1)
        self.assertGreater(len(state.body_gallery), 1)

    # 15
    def test_matrix_15_shadow_authoritative_event_audit_metrics_and_latency(self):
        shadow, _ = self.make_resolver(mode=ResolverMode.SHADOW)
        production_label = {"global_person_id": "legacy"}
        decision = shadow.resolve(
            "a", 1, face_embedding=vec(1, 0), known_person_id="p",
            label_target=production_label,
        )
        self.assertEqual("legacy", production_label["global_person_id"])
        self.assertFalse(decision.authoritative_applied)
        self.assertEqual(1, len(shadow.audit_log))
        audit = shadow.audit_log[-1]
        self.assertEqual("CONFIRMED", audit["final_decision"])
        self.assertEqual(decision.global_id, audit["candidate_id"])
        self.assertIn("face_similarity", audit)
        self.assertIn("body_similarity", audit)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in audit.values()))
        expected_values = {
            "GLOBAL_ID_CREATED", "GLOBAL_ID_CONFIRMED", "GLOBAL_ID_REASSIGNED",
            "GLOBAL_ID_MOVED_CAMERA", "GLOBAL_ID_MISSING", "GLOBAL_ID_REAPPEARED",
            "GLOBAL_ID_EXPIRED", "GLOBAL_ID_CONFLICT",
        }
        self.assertEqual(expected_values, {event.value for event in EventType})
        confirmed_event = [e for e in shadow.events if e["type"] == "GLOBAL_ID_CONFIRMED"][-1]
        self.assertEqual("CONFIRMED", confirmed_event["details"]["final_decision"])
        self.assertIn("face_similarity", confirmed_event["details"])
        self.assertIn("body_similarity", confirmed_event["details"])
        self.assertEqual(1, shadow.metrics["resolve_count"])
        self.assertGreaterEqual(shadow.metrics["latency_ms_average"], 0.0)
        # Deliberately generous smoke bound: catches accidental blocking/I/O but
        # does not pretend CI scheduling is a precision benchmark.
        self.assertLess(shadow.metrics["latency_ms_max"], 2000.0)

        authoritative, _ = self.make_resolver(mode=ResolverMode.AUTHORITATIVE)
        label = {}
        applied = authoritative.resolve(
            "a", 1, face_embedding=vec(1, 0), known_person_id="p",
            label_target=label,
        )
        self.assertTrue(applied.authoritative_applied)
        self.assertEqual(applied.global_id, label["global_person_id"])


class GlobalIdentitySafetyTests(unittest.TestCase):
    def test_acceptance_matrix_has_exactly_15_named_scenarios(self):
        self.assertEqual(list(range(1, 16)), sorted(ACCEPTANCE_MATRIX))
        names = unittest.defaultTestLoader.getTestCaseNames(GlobalIdentityScenarios)
        for number in range(1, 16):
            self.assertTrue(
                any(name.startswith("test_matrix_%02d_" % number) for name in names),
                "acceptance matrix item %d has no demonstrable test" % number,
            )

    def test_invalid_vectors_are_uncertain_and_not_stored(self):
        resolver = GlobalIdentityResolver(id_factory=IDs())
        result = resolver.resolve("a", 1, face_embedding=np.zeros(3))
        self.assertEqual(DecisionState.UNCERTAIN, result.state)
        self.assertEqual(0, resolver.metrics["active_count"])

    def test_thread_safe_same_known_person(self):
        resolver = GlobalIdentityResolver(id_factory=IDs())
        results = []
        errors = []

        def worker(i):
            try:
                results.append(resolver.resolve(
                    "overlap", i, face_embedding=vec(1, 0), known_person_id="same"
                ).global_id)
            except Exception as exc:  # pragma: no cover - diagnostic
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([], errors)
        self.assertEqual(1, len(set(results)))

    def test_body_default_threshold_is_documented_value(self):
        self.assertEqual(0.56, CalibratedPolicy().body_match_threshold)

    def test_no_database_or_heavy_vision_dependency_in_hot_path(self):
        resolver = GlobalIdentityResolver(id_factory=IDs())
        started = time.perf_counter()
        for i in range(100):
            resolver.resolve("a", i, face_embedding=vec(1, i + 1))
        elapsed = time.perf_counter() - started
        self.assertEqual(100, resolver.metrics["resolve_count"])
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
