"""
Zone tracker — one person, several cameras, at the same moment.

WHAT THIS SOLVES
----------------
Two cameras face each other down the COE corridor. When a person is identified
on one, the other should mark them immediately — but the far camera can see
neither their face (it is behind them) nor enough body pixels to match on
appearance. See ground_plane.py for why both appearance routes are structurally
dead in this geometry.

So association happens in FLOOR COORDINATES. Each camera projects a track's
foot point onto the zone's shared floor; two observations that land in the same
place are the same person. That works from behind, at any size, and through a
covered face.

This module owns the resulting cross-camera object: a ZonePerson, which is one
human with up to one track per camera, a fused position, and at most one
identity.

INHERITED IDENTITY IS NOT EVIDENCE
----------------------------------
The rule that keeps this honest: an identity may only ENTER a ZonePerson from a
member that recognised it LOCALLY, from its own face observation. An inherited
identity never propagates onward and never votes.

Without that rule the system launders a guess into a certainty — camera B
inherits "Chirag" from camera A, B's display is then read as agreement, and a
single bad match becomes unfalsifiable across the whole zone. It also honours
the codebase's existing split: tracking owns existence, recognition owns
identity. Geometry is tracking, so it may say WHERE and WHICH TRACK, never WHO.

Two consequences, both deliberate:
  * inheritance only ever fills a VOID. A track that locally recognised someone
    else keeps its own answer and the disagreement is logged, because a
    conflict is a real signal (bad calibration, or a genuine mis-match) and
    silently overwriting it would destroy the evidence.
  * an inherited identity is flagged, so the pipeline can draw it differently
    and so nothing downstream mistakes it for an independent sighting. Callers
    must not write entry/exit events from it: one person visible on two cameras
    would double-count footfall, the same bug the unidentified_entry/exit split
    already had to avoid.

WITHIN-CAMERA DISTINCTNESS IS AUTHORITATIVE
-------------------------------------------
Two tracks in the SAME camera are never merged into one ZonePerson, however
close their projections land. That camera's tracker had full pixel evidence and
concluded they are two people; a homography with half a metre of calibration
error does not get to overrule it.

THREADING
---------
Process-wide singleton shared by every camera thread, guarded by an RLock. The
work per call is a handful of small matrices, so contention is negligible.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackObservation:
    """One camera's view of one track, already projected onto the floor."""
    track_id: int
    x: float
    y: float
    sigma: float                      # position std-dev, metres

    # Identity as this camera RECOGNISED it. Never an inherited one -- passing
    # an inherited identity back in is precisely the laundering loop above.
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    identity_confirmed: bool = False  # track.identity_state is KNOWN

    unknown_id: Optional[str] = None
    unknown_label: Optional[str] = None


@dataclass
class Assignment:
    """What the zone knows about one of the calling camera's tracks."""
    zone_person_id: str
    x: float
    y: float
    sigma: float
    camera_count: int                 # how many cameras currently see this person

    # Filled only when this track has no local identity of its own and another
    # camera does. Always flagged as inherited.
    inherited_person_id: Optional[str] = None
    inherited_person_name: Optional[str] = None
    inherited_confidence: float = 0.0
    inherited_unknown_id: Optional[str] = None
    inherited_unknown_label: Optional[str] = None
    inherited_from: Optional[str] = None   # source camera_id

    @property
    def is_multi_camera(self) -> bool:
        return self.camera_count > 1

    @property
    def has_inherited_identity(self) -> bool:
        return bool(self.inherited_person_id or self.inherited_unknown_id)


@dataclass
class _Member:
    camera_id: str
    track_id: int
    x: float
    y: float
    sigma: float
    ts: float
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    identity_confirmed: bool = False
    unknown_id: Optional[str] = None
    unknown_label: Optional[str] = None


@dataclass
class _Identity:
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    unknown_id: Optional[str] = None
    unknown_label: Optional[str] = None
    source_camera: str = ""
    ts: float = 0.0


@dataclass
class _ZonePerson:
    zone_person_id: str
    members: Dict[str, _Member] = field(default_factory=dict)  # camera_id -> member
    identity: Optional[_Identity] = None
    created_at: float = 0.0

    def fused(self) -> Tuple[float, float, float]:
        """
        Inverse-variance weighted position.

        This is what makes an opposed-camera pair useful despite one of them
        being far away: ground-plane error grows sharply with distance, so the
        near camera's estimate carries most of the weight automatically, while
        the far one still contributes. A plain mean would let the worst
        observation drag the answer.
        """
        xs, ys, ws = [], [], []
        for m in self.members.values():
            if m.sigma <= 0 or not np.isfinite(m.sigma):
                continue
            w = 1.0 / (m.sigma * m.sigma)
            xs.append(m.x * w)
            ys.append(m.y * w)
            ws.append(w)
        if not ws:
            return 0.0, 0.0, float("inf")
        total = float(sum(ws))
        return float(sum(xs)) / total, float(sum(ys)) / total, float(1.0 / np.sqrt(total))

    @property
    def last_seen(self) -> float:
        return max((m.ts for m in self.members.values()), default=0.0)


class ZoneTracker:
    """Cross-camera association and identity propagation for ONE zone."""

    def __init__(
        self,
        zone: str,
        gate_sigma: float = 3.0,
        calib_floor_m: float = 0.5,
        max_dist_m: float = 3.0,
        member_ttl_sec: float = 2.0,
        identity_ttl_sec: float = 10.0,
    ):
        """
        Args:
            gate_sigma: how many combined standard deviations still count as
                the same person.
            calib_floor_m: assumed calibration error, added in quadrature. A
                hand-clicked homography is good to roughly this much, and
                without the floor two confident-but-slightly-offset cameras
                would never associate.
            max_dist_m: hard cap. A far-camera observation can have a huge
                sigma, which would otherwise make the statistical gate accept
                anything in the corridor.
            member_ttl_sec: how long an observation still anchors a person.
                Longer than a frame, so association survives the frame skipping
                the load governor may impose.
            identity_ttl_sec: how long an established identity survives after
                the camera that recognised it stops contributing.
        """
        self.zone = zone
        self.gate_sigma = gate_sigma
        self.calib_floor_m = calib_floor_m
        self.max_dist_m = max_dist_m
        self.member_ttl_sec = member_ttl_sec
        self.identity_ttl_sec = identity_ttl_sec

        self._people: Dict[str, _ZonePerson] = {}
        self._track_index: Dict[Tuple[str, int], str] = {}
        self._lock = threading.RLock()

        self.stat_created = 0
        self.stat_merged = 0
        self.stat_inherited = 0
        self.stat_conflicts = 0

    # ── Public API ────────────────────────────────────────────────────────

    def publish(
        self, camera_id: str, observations: List[TrackObservation], now: Optional[float] = None
    ) -> Dict[int, Assignment]:
        """
        Submit one camera's current floor observations and get back what the
        zone knows about each of them.

        Cameras call this independently at their own rates; the zone is a
        blackboard, not a synchronised step.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            self._associate(camera_id, observations, now)
            self._merge_across_cameras(camera_id)
            self._refresh_identities(now)
            return self._assignments(camera_id, observations, now)

    def snapshot(self) -> List[dict]:
        """Current people in the zone, for the API and for debugging."""
        with self._lock:
            out = []
            for p in self._people.values():
                x, y, sigma = p.fused()
                out.append({
                    "zone_person_id": p.zone_person_id,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "sigma": round(sigma, 3),
                    "cameras": sorted(p.members.keys()),
                    "camera_count": len(p.members),
                    "person_id": p.identity.person_id if p.identity else None,
                    "person_name": p.identity.person_name if p.identity else None,
                    "unknown_id": p.identity.unknown_id if p.identity else None,
                    "unknown_label": p.identity.unknown_label if p.identity else None,
                    "identified_by": p.identity.source_camera if p.identity else None,
                })
            return sorted(out, key=lambda d: -d["camera_count"])

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "people": len(self._people),
                "multi_camera": sum(1 for p in self._people.values() if len(p.members) > 1),
                "created": self.stat_created,
                "merged": self.stat_merged,
                "inherited": self.stat_inherited,
                "conflicts": self.stat_conflicts,
            }

    def reset(self):
        with self._lock:
            self._people.clear()
            self._track_index.clear()

    # ── Internals ─────────────────────────────────────────────────────────

    def _prune(self, now: float):
        for person in list(self._people.values()):
            for cam, member in list(person.members.items()):
                if now - member.ts > self.member_ttl_sec:
                    person.members.pop(cam, None)
                    self._track_index.pop((cam, member.track_id), None)
            if not person.members:
                # Keep nothing behind: an empty person cannot be re-associated
                # (it has no position) and would leak.
                self._people.pop(person.zone_person_id, None)

    def _gate_radius(self, sigma_a: float, sigma_b: float) -> float:
        combined = np.sqrt(
            sigma_a * sigma_a + sigma_b * sigma_b + self.calib_floor_m ** 2
        )
        return float(min(self.max_dist_m, self.gate_sigma * combined))

    def _associate(self, camera_id: str, observations: List[TrackObservation], now: float):
        people = list(self._people.values())
        if observations and people:
            cost = np.ones((len(observations), len(people)), dtype=np.float32) * 1e6
            for i, obs in enumerate(observations):
                for j, person in enumerate(people):
                    px, py, psigma = person.fused()
                    if not np.isfinite(psigma):
                        continue
                    # A person already holding a DIFFERENT track from this same
                    # camera is not a candidate: this camera's tracker says
                    # those are two people, and it had the pixels to know.
                    existing = person.members.get(camera_id)
                    if existing is not None and existing.track_id != obs.track_id:
                        continue
                    d = float(np.hypot(obs.x - px, obs.y - py))
                    radius = self._gate_radius(obs.sigma, psigma)
                    if radius <= 0:
                        continue
                    cost[i, j] = d / radius   # <= 1.0 means inside the gate
            from app.vision.bytetrack import _linear_assignment
            matches, unmatched_obs, _ = _linear_assignment(cost, thresh=1.0)
        else:
            matches, unmatched_obs = [], list(range(len(observations)))

        for i, j in matches:
            self._attach(people[j], camera_id, observations[i], now)
        for i in unmatched_obs:
            self._create(camera_id, observations[i], now)

    def _attach(self, person: _ZonePerson, camera_id: str, obs: TrackObservation, now: float):
        # A track may belong to exactly one person; moving it must not leave a
        # stale membership behind in its previous one.
        previous_id = self._track_index.get((camera_id, obs.track_id))
        if previous_id and previous_id != person.zone_person_id:
            previous = self._people.get(previous_id)
            if previous is not None:
                existing = previous.members.get(camera_id)
                if existing is not None and existing.track_id == obs.track_id:
                    previous.members.pop(camera_id, None)
                if not previous.members:
                    self._people.pop(previous_id, None)

        person.members[camera_id] = _Member(
            camera_id=camera_id, track_id=obs.track_id, x=obs.x, y=obs.y,
            sigma=obs.sigma, ts=now,
            person_id=obs.person_id, person_name=obs.person_name,
            confidence=obs.confidence, identity_confirmed=obs.identity_confirmed,
            unknown_id=obs.unknown_id, unknown_label=obs.unknown_label,
        )
        self._track_index[(camera_id, obs.track_id)] = person.zone_person_id

    def _create(self, camera_id: str, obs: TrackObservation, now: float) -> _ZonePerson:
        person = _ZonePerson(zone_person_id=str(uuid4()), created_at=now)
        self._people[person.zone_person_id] = person
        self._attach(person, camera_id, obs, now)
        self.stat_created += 1
        return person

    def _merge_across_cameras(self, camera_id: str):
        """
        Fuse people that different cameras created independently.

        Association alone cannot do this: each camera's track anchors its own
        ZonePerson, so both stay self-consistent and never meet. This pass is
        what actually creates the cross-camera link, and it is deliberately
        restricted to people with DISJOINT camera sets -- merging two people
        that share a camera would contradict that camera's own tracker.
        """
        candidates: List[Tuple[float, str, str]] = []
        people = list(self._people.values())
        for a_i in range(len(people)):
            for b_i in range(a_i + 1, len(people)):
                a, b = people[a_i], people[b_i]
                if set(a.members) & set(b.members):
                    continue
                if camera_id not in a.members and camera_id not in b.members:
                    continue   # only reconsider pairs this update could affect
                ax, ay, asig = a.fused()
                bx, by, bsig = b.fused()
                if not (np.isfinite(asig) and np.isfinite(bsig)):
                    continue
                d = float(np.hypot(ax - bx, ay - by))
                radius = self._gate_radius(asig, bsig)
                if radius > 0 and d <= radius:
                    candidates.append((d / radius, a.zone_person_id, b.zone_person_id))

        candidates.sort()
        merged: set = set()
        for _, a_id, b_id in candidates:
            if a_id in merged or b_id in merged:
                continue
            a, b = self._people.get(a_id), self._people.get(b_id)
            if a is None or b is None or set(a.members) & set(b.members):
                continue
            # The older person survives, so a zone_person_id is stable for as
            # long as the human is in the zone.
            keep, drop = (a, b) if a.created_at <= b.created_at else (b, a)
            for cam, member in drop.members.items():
                keep.members[cam] = member
                self._track_index[(cam, member.track_id)] = keep.zone_person_id
            if keep.identity is None or (
                drop.identity is not None
                and drop.identity.confidence > keep.identity.confidence
            ):
                keep.identity = keep.identity if drop.identity is None else drop.identity
            self._people.pop(drop.zone_person_id, None)
            merged.add(drop.zone_person_id)
            self.stat_merged += 1
            logger.debug(
                f"[{self.zone}] merged zone persons across cameras "
                f"{sorted(keep.members)} -> {keep.zone_person_id[:8]}…"
            )

    def _refresh_identities(self, now: float):
        """
        Recompute each person's identity from LOCALLY recognised members only.

        Sticky: an identity survives the frames where nobody currently has a
        face on the person, which is the normal state in an opposed-camera pair
        and the entire reason this feature exists. It expires after
        identity_ttl_sec so a stale name cannot ride along indefinitely.
        """
        for person in self._people.values():
            best: Optional[_Member] = None
            for member in person.members.values():
                if not (member.identity_confirmed and member.person_id):
                    continue
                if best is None or member.confidence > best.confidence:
                    best = member
            if best is not None:
                current = person.identity
                if (
                    current is None
                    or current.person_id != best.person_id
                    or best.confidence >= current.confidence
                ):
                    if (
                        current is not None
                        and current.person_id
                        and current.person_id != best.person_id
                    ):
                        self.stat_conflicts += 1
                        logger.info(
                            f"[{self.zone}] zone person {person.zone_person_id[:8]}… "
                            f"identity changed {current.person_name} -> {best.person_name} "
                            f"(camera {best.camera_id}, conf {best.confidence:.2f})"
                        )
                    person.identity = _Identity(
                        person_id=best.person_id, person_name=best.person_name,
                        confidence=best.confidence, source_camera=best.camera_id, ts=now,
                    )
                continue

            # No enrolled identity: a stable stranger id is still worth sharing,
            # so a confirmed Unknown-XXXXXX shows the same label on every camera.
            stranger = None
            for member in person.members.values():
                if member.unknown_id and (stranger is None or member.ts < stranger.ts):
                    stranger = member
            if stranger is not None and (
                person.identity is None or person.identity.person_id is None
            ):
                person.identity = _Identity(
                    unknown_id=stranger.unknown_id, unknown_label=stranger.unknown_label,
                    source_camera=stranger.camera_id, ts=now,
                )
            elif person.identity is not None and now - person.identity.ts > self.identity_ttl_sec:
                person.identity = None

    def _assignments(
        self, camera_id: str, observations: List[TrackObservation], now: float
    ) -> Dict[int, Assignment]:
        out: Dict[int, Assignment] = {}
        for obs in observations:
            person_id = self._track_index.get((camera_id, obs.track_id))
            person = self._people.get(person_id) if person_id else None
            if person is None:
                continue
            x, y, sigma = person.fused()
            assignment = Assignment(
                zone_person_id=person.zone_person_id,
                x=x, y=y, sigma=sigma, camera_count=len(person.members),
            )

            identity = person.identity
            if identity is not None and identity.source_camera != camera_id:
                if obs.identity_confirmed and obs.person_id:
                    # This camera recognised someone itself. Local recognition
                    # is the authority; geometry never overrides it. A
                    # disagreement is logged because it is real evidence --
                    # of a bad calibration or a genuine mis-match.
                    if identity.person_id and identity.person_id != obs.person_id:
                        self.stat_conflicts += 1
                        logger.warning(
                            f"[{self.zone}] identity conflict on zone person "
                            f"{person.zone_person_id[:8]}…: {camera_id} says "
                            f"{obs.person_name}, {identity.source_camera} says "
                            f"{identity.person_name}. Keeping each camera's own."
                        )
                elif identity.person_id and not obs.person_id:
                    assignment.inherited_person_id = identity.person_id
                    assignment.inherited_person_name = identity.person_name
                    assignment.inherited_confidence = identity.confidence
                    assignment.inherited_from = identity.source_camera
                    self.stat_inherited += 1
                elif identity.unknown_id and not obs.unknown_id and not obs.person_id:
                    assignment.inherited_unknown_id = identity.unknown_id
                    assignment.inherited_unknown_label = identity.unknown_label
                    assignment.inherited_from = identity.source_camera
                    self.stat_inherited += 1

            out[obs.track_id] = assignment
        return out


class ZoneRegistry:
    """All zones in this process. One ZoneTracker per zone, created on demand."""

    def __init__(self):
        self._zones: Dict[str, ZoneTracker] = {}
        self._lock = threading.Lock()

    def zone(self, name: str) -> ZoneTracker:
        tracker = self._zones.get(name)
        if tracker is None:
            with self._lock:
                tracker = self._zones.get(name)
                if tracker is None:
                    from app.config import settings
                    tracker = ZoneTracker(
                        zone=name,
                        gate_sigma=settings.ZONE_ASSOC_GATE_SIGMA,
                        calib_floor_m=settings.ZONE_ASSOC_CALIB_FLOOR_M,
                        max_dist_m=settings.ZONE_ASSOC_MAX_DIST_M,
                        member_ttl_sec=settings.ZONE_MEMBER_TTL_SEC,
                        identity_ttl_sec=settings.ZONE_IDENTITY_TTL_SEC,
                    )
                    self._zones[name] = tracker
        return tracker

    def all_zones(self) -> Dict[str, ZoneTracker]:
        return dict(self._zones)


_registry: Optional[ZoneRegistry] = None
_registry_lock = threading.Lock()


def get_zone_registry() -> ZoneRegistry:
    """Process-wide zone registry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ZoneRegistry()
    return _registry
