"""Thread-safe, volatile global person identity resolution.

This module deliberately sits *above* camera-local tracking and recognition.  It
neither imports nor changes ByteTrack, YOLO, SCRFD, ArcFace, OSNet, face
registries, or :class:`ReIDTracker`.  Callers provide already-computed evidence.
There is no database access in the hot path.

Privacy/retention contract
--------------------------
Face and body vectors live only in bounded in-memory galleries.  When an
identity expires its vectors are erased.  Expired metadata is bounded and may
be removed immediately with :meth:`GlobalIdentityResolver.purge_expired`.
Audit records contain scores/reasons, never biometric vectors.  A process
restart clears all state; persistence, consent, encryption and longer retention
are explicitly responsibilities of a separate policy-controlled layer.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Tuple
from uuid import uuid4

import numpy as np

LocalKey = Tuple[str, int]


class LifecycleState(str, Enum):
    ACTIVE = "ACTIVE"
    MISSING = "MISSING"
    EXPIRED = "EXPIRED"


class DecisionState(str, Enum):
    CONFIRMED = "CONFIRMED"
    CANDIDATE = "CANDIDATE"
    UNCERTAIN = "UNCERTAIN"
    NEW = "NEW"


class ResolverMode(str, Enum):
    SHADOW = "shadow"
    AUTHORITATIVE = "authoritative"


class EventType(str, Enum):
    """Public event names; serialized values are the stable integration API."""

    GLOBAL_ID_CREATED = "GLOBAL_ID_CREATED"
    CONFIRMED = "GLOBAL_ID_CONFIRMED"
    REASSIGNED = "GLOBAL_ID_REASSIGNED"
    MOVED_CAMERA = "GLOBAL_ID_MOVED_CAMERA"
    MISSING = "GLOBAL_ID_MISSING"
    REAPPEARED = "GLOBAL_ID_REAPPEARED"
    EXPIRED = "GLOBAL_ID_EXPIRED"
    CONFLICT = "GLOBAL_ID_CONFLICT"


@dataclass(frozen=True)
class CameraTransition:
    """A directed, calibrated path from one camera to another."""

    source_camera: str
    destination_camera: str
    min_travel_time: float
    max_travel_time: float
    overlap: bool = False

    def __post_init__(self) -> None:
        if self.min_travel_time < 0 or self.max_travel_time < self.min_travel_time:
            raise ValueError("invalid camera transition travel-time range")


@dataclass
class CalibratedPolicy:
    """Matching policy.

    Similarity margins have no universal safe value and therefore default to
    ``None``.  Searches fail closed (UNCERTAIN) until the deployment supplies
    calibrated margins.  The requested body similarity gate defaults to 0.56;
    it is only a gate into temporal candidate voting, never auto-confirmation.
    """

    body_match_threshold: float = 0.56
    body_min_margin: Optional[float] = None
    face_match_threshold: Optional[float] = None
    face_min_margin: Optional[float] = None
    body_votes_required: int = 3
    body_vote_window: float = 8.0
    body_vote_min_span: float = 1.0
    body_vote_min_interval: float = 0.25

    def __post_init__(self) -> None:
        if not -1.0 <= self.body_match_threshold <= 1.0:
            raise ValueError("body_match_threshold must be a cosine similarity")
        if self.body_votes_required < 2:
            raise ValueError("body evidence must be repeated at least twice")
        if self.body_vote_window <= 0 or self.body_vote_min_span < 0:
            raise ValueError("invalid body vote timing")
        for value in (self.body_min_margin, self.face_min_margin):
            if value is not None and value < 0:
                raise ValueError("similarity margins cannot be negative")


@dataclass
class GlobalIdentityEvent:
    event_type: EventType
    timestamp: float
    global_id: str
    camera_id: Optional[str] = None
    local_track_id: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "global_id": self.global_id,
            "camera_id": self.camera_id,
            "local_track_id": self.local_track_id,
            "details": dict(self.details),
        }


@dataclass
class GlobalPersonState:
    """Hot state for one global person.

    ``local_bindings`` may contain tracks from multiple overlapping cameras.
    The missing timer is set only after this mapping becomes empty.
    Galleries contain normalized vectors and are bounded by the resolver.
    """

    global_id: str
    created_at: float
    last_seen: float
    lifecycle: LifecycleState = LifecycleState.ACTIVE
    known_person_id: Optional[str] = None
    known_person_name: Optional[str] = None
    local_bindings: Dict[LocalKey, float] = field(default_factory=dict)
    face_gallery: List[np.ndarray] = field(default_factory=list, repr=False)
    body_gallery: List[np.ndarray] = field(default_factory=list, repr=False)
    last_camera: Optional[str] = None
    missing_since: Optional[float] = None
    expired_at: Optional[float] = None
    camera_history: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=32))
    departures: Dict[str, float] = field(default_factory=dict)

    @property
    def enrolled(self) -> bool:
        return self.known_person_id is not None


@dataclass
class ResolutionDecision:
    state: DecisionState
    global_id: Optional[str]
    reason: str
    candidate_id: Optional[str] = None
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    event_types: Tuple[EventType, ...] = ()
    authoritative_applied: bool = False

    @property
    def assigned(self) -> bool:
        return self.global_id is not None and self.state in (
            DecisionState.CONFIRMED,
            DecisionState.NEW,
        )


class GlobalIdentityResolver:
    """Thread-safe in-memory global identity resolver.

    Known, verified face labels are authoritative.  Face embeddings are the
    primary biometric evidence.  Body evidence can only produce a CANDIDATE
    after repeated, temporally separated observations and a valid directed
    camera transition; it never confirms or overwrites an identity by itself.

    In ``shadow`` mode the resolver maintains its private shadow state and emits
    decisions/audit data, but never writes ``global_person_id`` to a supplied
    ``label_target``.  In ``authoritative`` mode only assigned decisions mutate
    that field.  Thus rollout comparisons cannot silently change production
    labels.
    """

    def __init__(
        self,
        *,
        policy: Optional[CalibratedPolicy] = None,
        transitions: Iterable[CameraTransition] = (),
        mode: ResolverMode | str = ResolverMode.SHADOW,
        missing_timeout: float = 120.0,
        face_gallery_size: int = 8,
        body_gallery_size: int = 8,
        gallery_min_distance: float = 0.01,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] = lambda: str(uuid4()),
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_audit_records: int = 5000,
        max_events: int = 5000,
        max_expired: int = 1000,
    ) -> None:
        if not 5 <= face_gallery_size <= 10 or not 5 <= body_gallery_size <= 10:
            raise ValueError("default operational galleries must be bounded to 5..10 views")
        if missing_timeout <= 0:
            raise ValueError("missing_timeout must be positive")
        self.policy = policy or CalibratedPolicy()
        self.mode = ResolverMode(mode)
        self.missing_timeout = float(missing_timeout)
        self.face_gallery_size = int(face_gallery_size)
        self.body_gallery_size = int(body_gallery_size)
        self.gallery_min_distance = float(gallery_min_distance)
        self._clock = clock
        self._id_factory = id_factory
        self._event_callback = event_callback
        self._max_expired = max_expired
        self._lock = threading.RLock()

        self._active: Dict[str, GlobalPersonState] = {}
        self._missing: Dict[str, GlobalPersonState] = {}
        self._expired: "OrderedDict[str, GlobalPersonState]" = OrderedDict()
        self._local_to_global: Dict[LocalKey, str] = {}
        self._known_to_global: Dict[str, str] = {}
        self._transitions: Dict[Tuple[str, str], CameraTransition] = {
            (t.source_camera, t.destination_camera): t for t in transitions
        }
        self._body_votes: Dict[Tuple[LocalKey, str], Deque[float]] = {}
        self._events: Deque[GlobalIdentityEvent] = deque(maxlen=max_events)
        self._audit: Deque[Dict[str, Any]] = deque(maxlen=max_audit_records)
        self._decision_counts: Counter = Counter()
        self._event_counts: Counter = Counter()
        self._resolve_count = 0
        self._search_count = 0
        self._search_comparisons = 0
        self._total_latency_ms = 0.0
        self._max_latency_ms = 0.0

    # ---- public hot-path API -------------------------------------------------

    def resolve(
        self,
        camera_id: str,
        local_track_id: int,
        *,
        face_embedding: Optional[np.ndarray] = None,
        body_embedding: Optional[np.ndarray] = None,
        known_person_id: Optional[str] = None,
        known_person_name: Optional[str] = None,
        face_verified: bool = True,
        timestamp: Optional[float] = None,
        label_target: Optional[Any] = None,
    ) -> ResolutionDecision:
        """Resolve evidence for ``(camera_id, local_track_id)``.

        ``known_person_id`` must only be supplied for an upstream verified face
        result.  Set ``face_verified=False`` for provisional recognizer output;
        it is audited but cannot claim an enrolled identity.
        """
        started = time.perf_counter()
        now = self._now(timestamp)
        key = (str(camera_id), int(local_track_id))
        face = self._vector(face_embedding)
        body = self._vector(body_embedding)

        with self._lock:
            self._sweep_locked(now)
            if known_person_id is not None and face_verified:
                decision = self._resolve_known_face(
                    key, str(known_person_id), known_person_name, face, body, now
                )
            elif known_person_id is not None:
                decision = ResolutionDecision(
                    DecisionState.UNCERTAIN, None, "unverified_known_face_rejected"
                )
            elif key in self._local_to_global:
                state = self._state_locked(self._local_to_global[key])
                if state is None or state.lifecycle == LifecycleState.EXPIRED:
                    # Defensive cleanup: expired IDs are terminal and can never
                    # be revived through a stale local index.
                    self._local_to_global.pop(key, None)
                    decision = self._resolve_unbound(key, face, body, now)
                else:
                    # Existing bindings are sticky.  Body appearance can enrich
                    # them but can never overwrite a face-established identity.
                    scores = self._evidence_scores(state, face, body)
                    self._touch_and_learn(state, key, face, body, now)
                    decision = ResolutionDecision(
                        DecisionState.CONFIRMED,
                        state.global_id,
                        "existing_local_binding",
                        candidate_id=state.global_id,
                        scores=scores,
                    )
            else:
                decision = self._resolve_unbound(key, face, body, now)

            applied = self._apply_label(label_target, decision)
            decision.authoritative_applied = applied
            elapsed = (time.perf_counter() - started) * 1000.0
            self._record_decision(key, decision, now, elapsed)
            return decision

    # Friendly alias for integrations that call the operation "observe".
    observe = resolve

    def unbind(
        self, camera_id: str, local_track_id: int, *, timestamp: Optional[float] = None
    ) -> Optional[GlobalPersonState]:
        """Remove a camera-local binding; start MISSING only after the last one."""
        now = self._now(timestamp)
        key = (str(camera_id), int(local_track_id))
        with self._lock:
            gid = self._local_to_global.pop(key, None)
            if gid is None:
                return None
            state = self._state_locked(gid)
            if state is None:
                return None
            state.local_bindings.pop(key, None)
            state.departures[key[0]] = now
            if state.local_bindings:
                return state
            state.lifecycle = LifecycleState.MISSING
            state.missing_since = now
            self._active.pop(gid, None)
            self._missing[gid] = state
            self._emit(EventType.MISSING, state, now, key)
            return state

    release = unbind

    def sweep(self, *, timestamp: Optional[float] = None) -> List[str]:
        """Expire identities missing for at least ``missing_timeout`` seconds."""
        with self._lock:
            return self._sweep_locked(self._now(timestamp))

    expire = sweep

    def add_transition(self, transition: CameraTransition) -> None:
        with self._lock:
            self._transitions[(transition.source_camera, transition.destination_camera)] = transition

    def get_state(self, global_id: str) -> Optional[GlobalPersonState]:
        with self._lock:
            return self._state_locked(global_id)

    def global_id_for(self, camera_id: str, local_track_id: int) -> Optional[str]:
        with self._lock:
            return self._local_to_global.get((str(camera_id), int(local_track_id)))

    def purge_expired(self, *, before: Optional[float] = None) -> int:
        """Delete expired metadata (vectors were already erased at expiration)."""
        with self._lock:
            ids = [
                gid for gid, state in self._expired.items()
                if before is None or (state.expired_at is not None and state.expired_at <= before)
            ]
            for gid in ids:
                state = self._expired.pop(gid)
                if state.known_person_id and self._known_to_global.get(state.known_person_id) == gid:
                    self._known_to_global.pop(state.known_person_id, None)
            return len(ids)

    @property
    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [event.as_dict() for event in self._events]

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._audit]

    @property
    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            count = self._resolve_count
            return {
                "resolve_count": count,
                "active_count": len(self._active),
                "missing_count": len(self._missing),
                "expired_count": len(self._expired),
                "search_count": self._search_count,
                "search_comparisons": self._search_comparisons,
                "latency_ms_total": self._total_latency_ms,
                "latency_ms_average": self._total_latency_ms / count if count else 0.0,
                "latency_ms_max": self._max_latency_ms,
                "decision_counts": dict(self._decision_counts),
                "event_counts": dict(self._event_counts),
                "uncertain_count": self._decision_counts[DecisionState.UNCERTAIN.value],
                "conflict_count": self._event_counts[EventType.CONFLICT.value],
            }

    @property
    def active_states(self) -> Mapping[str, GlobalPersonState]:
        with self._lock:
            return dict(self._active)

    @property
    def missing_states(self) -> Mapping[str, GlobalPersonState]:
        with self._lock:
            return dict(self._missing)

    @property
    def expired_states(self) -> Mapping[str, GlobalPersonState]:
        with self._lock:
            return dict(self._expired)

    # ---- resolution internals -----------------------------------------------

    def _resolve_known_face(
        self,
        key: LocalKey,
        person_id: str,
        person_name: Optional[str],
        face: Optional[np.ndarray],
        body: Optional[np.ndarray],
        now: float,
    ) -> ResolutionDecision:
        old_gid = self._local_to_global.get(key)
        mapped_gid = self._known_to_global.get(person_id)
        old_state = self._state_locked(old_gid) if old_gid else None
        mapped = self._state_locked(mapped_gid) if mapped_gid else None
        previous_source = mapped_gid

        # Expiration is terminal.  A stale local or known-person index must not
        # resurrect an expired visit.  The same enrolled person starts a new
        # global visit/ID while the expired state remains available for audit.
        if old_state is not None and old_state.lifecycle == LifecycleState.EXPIRED:
            self._local_to_global.pop(key, None)
            old_state = None
        if mapped is not None and mapped.lifecycle == LifecycleState.EXPIRED:
            if self._known_to_global.get(person_id) == mapped.global_id:
                self._known_to_global.pop(person_id, None)
            mapped = None

        # An unknown active/missing identity may be enrolled later without
        # changing its global ID.
        if mapped is None and old_state is not None and not old_state.enrolled:
            state = old_state
            state.known_person_id = person_id
            state.known_person_name = person_name
            self._known_to_global[person_id] = state.global_id
        elif mapped is not None:
            state = mapped
            if person_name:
                state.known_person_name = person_name
        elif old_state is not None and old_state.known_person_id == person_id:
            state = old_state
            self._known_to_global[person_id] = state.global_id
        else:
            state = self._create_state(now, person_id, person_name)

        conflict = self._simultaneous_conflict(state, key, now)
        if conflict:
            self._emit(EventType.CONFLICT, state, now, key, conflict)
            return ResolutionDecision(
                DecisionState.UNCERTAIN,
                None,
                "simultaneous_non_overlap_conflict",
                candidate_id=state.global_id,
                event_types=(EventType.CONFLICT,),
            )

        events: List[EventType] = []
        if old_gid and old_gid != state.global_id:
            prior = self._state_locked(old_gid)
            if prior is not None:
                prior.local_bindings.pop(key, None)
                if not prior.local_bindings:
                    prior.lifecycle = LifecycleState.MISSING
                    prior.missing_since = now
                    self._active.pop(prior.global_id, None)
                    self._missing[prior.global_id] = prior
                    self._emit(EventType.MISSING, prior, now, key)
            self._emit(
                EventType.REASSIGNED,
                state,
                now,
                key,
                {"from_global_id": old_gid, "evidence": "verified_face"},
            )
            events.append(EventType.REASSIGNED)

        scores = {
            **self._evidence_scores(state, face, body),
            "previous_candidate_id": old_gid or previous_source,
            "source_camera": state.last_camera,
        }
        events.extend(self._bind(state, key, now))
        self._add_gallery(state.face_gallery, face, self.face_gallery_size)
        self._add_gallery(state.body_gallery, body, self.body_gallery_size)
        event_details = {
            "evidence": "verified_face",
            "candidate_id": state.global_id,
            "final_decision": DecisionState.CONFIRMED.value,
            "previous_candidate_id": old_gid or previous_source,
            **scores,
        }
        self._emit(EventType.CONFIRMED, state, now, key, event_details)
        events.append(EventType.CONFIRMED)
        return ResolutionDecision(
            DecisionState.CONFIRMED,
            state.global_id,
            "verified_known_face",
            candidate_id=state.global_id,
            scores=scores,
            event_types=tuple(events),
        )

    def _resolve_unbound(
        self,
        key: LocalKey,
        face: Optional[np.ndarray],
        body: Optional[np.ndarray],
        now: float,
    ) -> ResolutionDecision:
        if face is not None:
            return self._resolve_face_embedding(key, face, body, now)
        if body is not None:
            return self._resolve_body(key, body, now)
        return ResolutionDecision(DecisionState.UNCERTAIN, None, "no_valid_evidence")

    def _resolve_face_embedding(
        self, key: LocalKey, face: np.ndarray, body: Optional[np.ndarray], now: float
    ) -> ResolutionDecision:
        candidates = self._search(face, "face_gallery")
        if not candidates:
            state = self._create_state(now)
            events = self._bind(state, key, now)
            self._add_gallery(state.face_gallery, face, self.face_gallery_size)
            self._add_gallery(state.body_gallery, body, self.body_gallery_size)
            return ResolutionDecision(
                DecisionState.NEW,
                state.global_id,
                "first_unknown_face",
                event_types=tuple(events),
            )

        threshold = self.policy.face_match_threshold
        margin = self.policy.face_min_margin
        if threshold is None or margin is None:
            return ResolutionDecision(
                DecisionState.UNCERTAIN,
                None,
                "face_policy_uncalibrated",
                candidate_id=candidates[0][0].global_id,
                scores=self._score_dict(candidates),
            )
        top_state, top_score = candidates[0]
        second_score = candidates[1][1] if len(candidates) > 1 else None
        if top_score < threshold:
            state = self._create_state(now)
            events = self._bind(state, key, now)
            self._add_gallery(state.face_gallery, face, self.face_gallery_size)
            self._add_gallery(state.body_gallery, body, self.body_gallery_size)
            return ResolutionDecision(
                DecisionState.NEW,
                state.global_id,
                "face_no_match",
                scores=self._score_dict(candidates),
                event_types=tuple(events),
            )
        if second_score is not None and (top_score - second_score) < margin:
            return ResolutionDecision(
                DecisionState.UNCERTAIN,
                None,
                "ambiguous_face_margin",
                candidate_id=top_state.global_id,
                scores=self._score_dict(candidates),
            )
        conflict = self._simultaneous_conflict(top_state, key, now)
        if conflict:
            self._emit(EventType.CONFLICT, top_state, now, key, conflict)
            return ResolutionDecision(
                DecisionState.UNCERTAIN,
                None,
                "simultaneous_non_overlap_conflict",
                candidate_id=top_state.global_id,
                scores=self._score_dict(candidates),
                event_types=(EventType.CONFLICT,),
            )
        events = self._bind(top_state, key, now)
        self._add_gallery(top_state.face_gallery, face, self.face_gallery_size)
        self._add_gallery(top_state.body_gallery, body, self.body_gallery_size)
        self._emit(EventType.CONFIRMED, top_state, now, key, {
            "evidence": "face_embedding",
            "candidate_id": top_state.global_id,
            "final_decision": DecisionState.CONFIRMED.value,
            "face_similarity": top_score,
            "body_similarity": self._best_similarity(body, top_state.body_gallery),
            **self._score_dict(candidates),
        })
        events.append(EventType.CONFIRMED)
        return ResolutionDecision(
            DecisionState.CONFIRMED,
            top_state.global_id,
            "face_primary_match",
            scores=self._score_dict(candidates),
            event_types=tuple(events),
        )

    def _resolve_body(self, key: LocalKey, body: np.ndarray, now: float) -> ResolutionDecision:
        candidates = self._search(body, "body_gallery")
        if not candidates:
            return ResolutionDecision(DecisionState.NEW, None, "body_only_cannot_create_identity")

        top_state, top_score = candidates[0]
        second_score = candidates[1][1] if len(candidates) > 1 else None
        scores = {
            **self._score_dict(candidates),
            "body_similarity": top_score,
            "source_camera": top_state.last_camera,
        }
        if top_score < self.policy.body_match_threshold:
            return ResolutionDecision(DecisionState.NEW, None, "body_below_threshold", scores=scores)
        if self.policy.body_min_margin is None:
            return ResolutionDecision(
                DecisionState.UNCERTAIN,
                None,
                "body_margin_uncalibrated",
                candidate_id=top_state.global_id,
                scores=scores,
            )
        if second_score is not None and (top_score - second_score) < self.policy.body_min_margin:
            return ResolutionDecision(
                DecisionState.UNCERTAIN,
                None,
                "ambiguous_body_margin",
                candidate_id=top_state.global_id,
                scores=scores,
            )

        valid, reason = self._valid_transition(top_state, key[0], now)
        if not valid:
            if reason == "simultaneous_non_overlap_conflict":
                self._emit(EventType.CONFLICT, top_state, now, key, {"evidence": "body"})
            return ResolutionDecision(
                DecisionState.UNCERTAIN,
                None,
                reason,
                candidate_id=top_state.global_id,
                scores=scores,
                event_types=(EventType.CONFLICT,) if reason == "simultaneous_non_overlap_conflict" else (),
            )

        vote_key = (key, top_state.global_id)
        # A change of best candidate invalidates previous votes for this local track.
        for stale in [k for k in self._body_votes if k[0] == key and k != vote_key]:
            self._body_votes.pop(stale, None)
        votes = self._body_votes.setdefault(vote_key, deque())
        cutoff = now - self.policy.body_vote_window
        while votes and votes[0] < cutoff:
            votes.popleft()
        if not votes or now - votes[-1] >= self.policy.body_vote_min_interval:
            votes.append(now)
        enough = (
            len(votes) >= self.policy.body_votes_required
            and votes[-1] - votes[0] >= self.policy.body_vote_min_span
        )
        if not enough:
            return ResolutionDecision(
                DecisionState.UNCERTAIN,
                None,
                "insufficient_repeated_body_evidence",
                candidate_id=top_state.global_id,
                scores={**scores, "votes": float(len(votes))},
            )

        # Deliberately do not bind.  This remains a proposal even for excellent
        # body similarity, and especially cannot confirm an enrolled person.
        return ResolutionDecision(
            DecisionState.CANDIDATE,
            None,
            "repeated_body_transition_candidate",
            candidate_id=top_state.global_id,
            scores={**scores, "votes": float(len(votes))},
        )

    # ---- state/evidence helpers ---------------------------------------------

    def _create_state(
        self,
        now: float,
        person_id: Optional[str] = None,
        person_name: Optional[str] = None,
    ) -> GlobalPersonState:
        for _ in range(100):
            gid = str(self._id_factory())
            if self._state_locked(gid) is None:
                break
        else:
            raise RuntimeError("id_factory repeatedly returned duplicate IDs")
        state = GlobalPersonState(
            global_id=gid,
            created_at=now,
            last_seen=now,
            known_person_id=person_id,
            known_person_name=person_name,
        )
        self._active[gid] = state
        if person_id:
            self._known_to_global[person_id] = gid
        self._emit(EventType.GLOBAL_ID_CREATED, state, now, details={"known": bool(person_id)})
        return state

    def _bind(self, state: GlobalPersonState, key: LocalKey, now: float) -> List[EventType]:
        events: List[EventType] = []
        if state.lifecycle == LifecycleState.EXPIRED:
            raise RuntimeError("expired global identities are terminal and cannot be rebound")
        was_missing = state.lifecycle == LifecycleState.MISSING
        previous_camera = state.last_camera
        self._missing.pop(state.global_id, None)
        self._active[state.global_id] = state
        state.lifecycle = LifecycleState.ACTIVE
        state.missing_since = None
        state.expired_at = None
        state.local_bindings[key] = now
        self._local_to_global[key] = state.global_id
        state.last_seen = now
        state.last_camera = key[0]
        state.camera_history.append((key[0], now))
        if was_missing:
            self._emit(EventType.REAPPEARED, state, now, key)
            events.append(EventType.REAPPEARED)
        if previous_camera is not None and previous_camera != key[0]:
            self._emit(
                EventType.MOVED_CAMERA,
                state,
                now,
                key,
                {"from_camera": previous_camera, "to_camera": key[0]},
            )
            events.append(EventType.MOVED_CAMERA)
        return events

    def _touch_and_learn(
        self,
        state: GlobalPersonState,
        key: LocalKey,
        face: Optional[np.ndarray],
        body: Optional[np.ndarray],
        now: float,
    ) -> None:
        state.local_bindings[key] = now
        state.last_seen = now
        state.last_camera = key[0]
        state.camera_history.append((key[0], now))
        self._add_gallery(state.face_gallery, face, self.face_gallery_size)
        self._add_gallery(state.body_gallery, body, self.body_gallery_size)

    def _add_gallery(
        self, gallery: List[np.ndarray], vector: Optional[np.ndarray], limit: int
    ) -> None:
        if vector is None:
            return
        if any(1.0 - float(np.dot(vector, old)) < self.gallery_min_distance for old in gallery):
            return
        if len(gallery) < limit:
            gallery.append(vector.copy())
            return
        # Keep a diverse bounded set.  Replace the most redundant existing view
        # only when the incoming view increases that view's nearest-neighbour gap.
        if len(gallery) < 2:
            gallery[0] = vector.copy()
            return
        redundancies = []
        for i, item in enumerate(gallery):
            nearest_distance = min(
                1.0 - float(np.dot(item, other))
                for j, other in enumerate(gallery)
                if i != j
            )
            redundancies.append((nearest_distance, i))
        _, replace_i = min(redundancies)
        incoming_gap = min(1.0 - float(np.dot(vector, old)) for old in gallery)
        if incoming_gap > redundancies[replace_i][0]:
            gallery[replace_i] = vector.copy()

    @staticmethod
    def _best_similarity(
        vector: Optional[np.ndarray], gallery: List[np.ndarray]
    ) -> Optional[float]:
        if vector is None:
            return None
        compatible = [item for item in gallery if item.shape == vector.shape]
        return max((float(np.dot(vector, item)) for item in compatible), default=None)

    def _evidence_scores(
        self,
        state: GlobalPersonState,
        face: Optional[np.ndarray],
        body: Optional[np.ndarray],
    ) -> Dict[str, Optional[float]]:
        """Return scalar audit evidence only; biometric vectors never leave state."""
        return {
            "face_similarity": self._best_similarity(face, state.face_gallery),
            "body_similarity": self._best_similarity(body, state.body_gallery),
        }

    def _search(self, vector: np.ndarray, gallery_name: str) -> List[Tuple[GlobalPersonState, float]]:
        self._search_count += 1
        found: List[Tuple[GlobalPersonState, float]] = []
        # EXPIRED galleries are erased and never searched.
        for state in list(self._active.values()) + list(self._missing.values()):
            gallery = getattr(state, gallery_name)
            compatible = [v for v in gallery if v.shape == vector.shape]
            self._search_comparisons += len(compatible)
            if compatible:
                found.append((state, max(float(np.dot(vector, v)) for v in compatible)))
        found.sort(key=lambda row: row[1], reverse=True)
        return found

    @staticmethod
    def _score_dict(candidates: List[Tuple[GlobalPersonState, float]]) -> Dict[str, Optional[float]]:
        return {
            "best_similarity": candidates[0][1] if candidates else None,
            "second_similarity": candidates[1][1] if len(candidates) > 1 else None,
            "margin": (
                candidates[0][1] - candidates[1][1] if len(candidates) > 1 else None
            ),
        }

    def _valid_transition(
        self, state: GlobalPersonState, destination: str, now: float
    ) -> Tuple[bool, str]:
        source = state.last_camera
        if source is None or source == destination:
            return False, "camera_transition_required"
        transition = self._transitions.get((source, destination))
        if transition is None:
            return False, "camera_transition_not_configured"
        active_elsewhere = any(camera != destination for camera, _ in state.local_bindings)
        if active_elsewhere and not transition.overlap:
            return False, "simultaneous_non_overlap_conflict"
        reference = state.departures.get(source, state.last_seen)
        elapsed = now - reference
        if elapsed < transition.min_travel_time or elapsed > transition.max_travel_time:
            return False, "camera_transition_time_invalid"
        return True, "valid_camera_transition"

    def _simultaneous_conflict(
        self, state: GlobalPersonState, key: LocalKey, now: float
    ) -> Optional[Dict[str, Any]]:
        for (camera, _track), _seen in state.local_bindings.items():
            if camera == key[0]:
                continue
            transition = self._transitions.get((camera, key[0]))
            if transition is not None and not transition.overlap:
                return {"active_camera": camera, "requested_camera": key[0]}
        return None

    def _sweep_locked(self, now: float) -> List[str]:
        expired: List[str] = []
        for gid, state in list(self._missing.items()):
            if state.missing_since is None or now - state.missing_since < self.missing_timeout:
                continue
            state.lifecycle = LifecycleState.EXPIRED
            state.expired_at = now
            state.face_gallery.clear()
            state.body_gallery.clear()
            self._missing.pop(gid, None)
            self._expired[gid] = state
            if state.known_person_id and self._known_to_global.get(state.known_person_id) == gid:
                # Remove the hot lookup immediately.  The expired record retains
                # its known-person metadata for audit, but is never resolvable.
                self._known_to_global.pop(state.known_person_id, None)
            self._emit(EventType.EXPIRED, state, now, details={
                "final_decision": LifecycleState.EXPIRED.value,
                "retention": "biometric_galleries_cleared",
            })
            expired.append(gid)
        while len(self._expired) > self._max_expired:
            gid, state = self._expired.popitem(last=False)
            if state.known_person_id and self._known_to_global.get(state.known_person_id) == gid:
                self._known_to_global.pop(state.known_person_id, None)
        return expired

    def _state_locked(self, global_id: Optional[str]) -> Optional[GlobalPersonState]:
        if global_id is None:
            return None
        return self._active.get(global_id) or self._missing.get(global_id) or self._expired.get(global_id)

    @staticmethod
    def _vector(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return None
        return vector / norm

    def _now(self, timestamp: Optional[float]) -> float:
        return float(self._clock() if timestamp is None else timestamp)

    def _emit(
        self,
        event_type: EventType,
        state: GlobalPersonState,
        now: float,
        key: Optional[LocalKey] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = GlobalIdentityEvent(
            event_type,
            now,
            state.global_id,
            key[0] if key else None,
            key[1] if key else None,
            details or {},
        )
        self._events.append(event)
        self._event_counts[event_type.value] += 1
        if self._event_callback is not None:
            try:
                self._event_callback(event.as_dict())
            except Exception:
                # Observability must never break identity resolution.
                pass

    def _record_decision(
        self,
        key: LocalKey,
        decision: ResolutionDecision,
        now: float,
        elapsed_ms: float,
    ) -> None:
        self._resolve_count += 1
        self._decision_counts[decision.state.value] += 1
        self._total_latency_ms += elapsed_ms
        self._max_latency_ms = max(self._max_latency_ms, elapsed_ms)
        scores = dict(decision.scores)
        self._audit.append({
            "timestamp": now,
            "camera_id": key[0],
            "local_track_id": key[1],
            "decision": decision.state.value,
            "candidate_decision": (
                DecisionState.CANDIDATE.value if decision.candidate_id else None
            ),
            "final_decision": decision.state.value,
            "global_id": decision.global_id,
            "candidate_id": decision.candidate_id,
            "previous_candidate_id": scores.get("previous_candidate_id"),
            "source_camera": scores.get("source_camera"),
            "face_similarity": scores.get("face_similarity", scores.get("best_similarity") if "face" in decision.reason else None),
            "body_similarity": scores.get("body_similarity", scores.get("best_similarity") if "body" in decision.reason else None),
            "reason": decision.reason,
            "scores": scores,
            "mode": self.mode.value,
            "authoritative_applied": decision.authoritative_applied,
            "latency_ms": elapsed_ms,
        })

    def _apply_label(self, target: Optional[Any], decision: ResolutionDecision) -> bool:
        if target is None or self.mode != ResolverMode.AUTHORITATIVE or not decision.assigned:
            return False
        if isinstance(target, dict):
            target["global_person_id"] = decision.global_id
        else:
            setattr(target, "global_person_id", decision.global_id)
        return True


# -- Singleton factory --------------------------------------------------------

_global_resolver: Optional[GlobalIdentityResolver] = None
_global_resolver_lock = threading.Lock()


def get_global_resolver() -> Optional[GlobalIdentityResolver]:
    """Get or create the process-wide GlobalIdentityResolver singleton.

    Returns None if GLOBAL_ID_MODE is "off".  Thread-safe double-checked
    locking so every camera pipeline in the same process shares one resolver.
    """
    global _global_resolver
    if _global_resolver is not None:
        return _global_resolver

    with _global_resolver_lock:
        if _global_resolver is not None:
            return _global_resolver

        try:
            from app.config import settings
        except ImportError:
            return None

        mode_str = getattr(settings, "GLOBAL_ID_MODE", "off").lower()
        if mode_str == "off":
            return None

        mode = ResolverMode(mode_str)
        ttl = float(getattr(settings, "GLOBAL_ID_MISSING_TTL_SEC", 120.0))

        # Parse camera topology from JSON string
        transitions: list[CameraTransition] = []
        topo_raw = getattr(settings, "GLOBAL_ID_CAMERA_TOPOLOGY", "[]")
        if topo_raw:
            import json as _json
            try:
                topo_list = _json.loads(topo_raw) if isinstance(topo_raw, str) else topo_raw
                for t in topo_list:
                    transitions.append(CameraTransition(
                        source_camera=t["from"],
                        destination_camera=t["to"],
                        min_travel_time=float(t.get("min_sec", 0)),
                        max_travel_time=float(t.get("max_sec", 300)),
                        overlap=bool(t.get("overlap", False)),
                    ))
            except Exception as _e:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "Failed to parse GLOBAL_ID_CAMERA_TOPOLOGY: %s", _e
                )

        # Build calibrated policy from settings (None means fail-closed)
        policy = CalibratedPolicy(
            face_match_threshold=getattr(settings, "GLOBAL_ID_FACE_MATCH_THRESHOLD", None),
            face_min_margin=getattr(settings, "GLOBAL_ID_FACE_MIN_MARGIN", None),
            body_match_threshold=float(getattr(settings, "GLOBAL_ID_BODY_MATCH_THRESHOLD", 0.56)),
            body_min_margin=getattr(settings, "GLOBAL_ID_BODY_MIN_MARGIN", None),
            body_votes_required=int(getattr(settings, "GLOBAL_ID_BODY_VOTES_REQUIRED", 3)),
            body_vote_min_span=float(getattr(settings, "GLOBAL_ID_BODY_VOTE_MIN_SPAN_SEC", 1.0)),
            body_vote_min_interval=0.25,
        )

        _global_resolver = GlobalIdentityResolver(
            mode=mode,
            missing_timeout=ttl,
            transitions=transitions,
            policy=policy,
            face_gallery_size=int(getattr(settings, "GLOBAL_ID_FACE_GALLERY_SIZE", 8)),
            body_gallery_size=int(getattr(settings, "GLOBAL_ID_BODY_GALLERY_SIZE", 8)),
        )

        import logging as _log
        _log.getLogger(__name__).info(
            "GlobalIdentityResolver started (mode=%s, ttl=%.0fs, "
            "transitions=%d, policy=%s)",
            mode.value, ttl, len(transitions), policy,
        )

        return _global_resolver


__all__ = [
    "CalibratedPolicy",
    "CameraTransition",
    "DecisionState",
    "EventType",
    "GlobalIdentityEvent",
    "GlobalIdentityResolver",
    "GlobalPersonState",
    "LifecycleState",
    "ResolverMode",
    "ResolutionDecision",
    "get_global_resolver",
]

