"""
Identity Ledger — the identity layer that sits above ByteTrack.

Architectural contract (see Phase 1 investigation):

    SCRFD + ArcFace
            │
            ▼
    ByteTrack  ...................  GEOMETRY ONLY. Knows nothing about people.
            │
            ▼
    IdentityLedger  ..............  owns the track_id -> identity binding,
                                    retirement, and recovery scoring.

ByteTracker.update() must NOT be modified to understand identity. Everything
here operates on the tracker's *output*, which is why it is immune to the four
blockers documented in the Phase 1 report:

  B1  stale `self._lost_stracks` read      -> N/A, we run after update() returns
  B2  carry-over IoU 0.30 > match IoU 0.20 -> N/A, we do not compete with Hungarian
  B3  `_removed_stracks` age gate is self-contradictory -> we keep our own buffer
  B4  pure-IoU cannot bridge displacement  -> we can also score appearance

── MILESTONE 1 SCOPE ────────────────────────────────────────────────────────
This module is currently OBSERVE-ONLY. `observe()` never mutates a track and
never assigns an identity. Its sole job is to answer, with numbers:

    "When ByteTrack recreates a track, how often did a recoverable
     predecessor actually exist, and how close was it?"

That is the baseline we need before changing behaviour. Milestone 2 turns the
same scoring into an actual assignment.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _centroid(bbox: np.ndarray) -> Tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


@dataclass
class RetiredIdentity:
    """
    Snapshot of a track taken at the moment it disappeared from the tracker.

    Deliberately captures the FULL recognition state, not just person_id.
    The Phase 1 report found that ByteTrack's own carry-over copied only
    person_id/person_name, so even a successful carry-over lost the identity
    lock and the temporal-rescue distance history -- exactly the state that
    protects an identity on the reacquisition frame.
    """
    track_id: int
    person_id: Optional[str]
    person_name: Optional[str]
    bbox: np.ndarray
    retired_frame: int
    retired_at: float
    # Full recognition-cache state, so Milestone 2 can restore it verbatim
    recognition_confidence: float = 0.0
    recognition_locked: bool = False
    identity_fail_streak: int = 0
    distance_history: List[float] = field(default_factory=list)
    tracklet_len: int = 0

    @property
    def centroid(self) -> Tuple[float, float]:
        return _centroid(self.bbox)


@dataclass
class RecoveryCandidate:
    """A scored pairing between a newly-created track and a retired identity."""
    retired: RetiredIdentity
    iou: float
    centroid_distance: float
    frames_since_retired: int
    seconds_since_retired: float

    @property
    def person_name(self) -> Optional[str]:
        return self.retired.person_name


@dataclass
class RecoveryOpportunity:
    """
    A track that ByteTrack created fresh, together with the best retired
    identity that could plausibly have been carried into it.
    """
    track_id: int
    bbox: np.ndarray
    best: Optional[RecoveryCandidate]
    n_eligible: int
    n_retired_total: int

    @property
    def recoverable(self) -> bool:
        """True if a plausible predecessor existed at all (any eligible candidate)."""
        return self.best is not None

    @property
    def unambiguous(self) -> bool:
        """
        True if exactly one retired identity was eligible.

        Milestone 1 measured 1 eligible candidate for genuine recoveries and 2
        for two people crossing. With >1 candidate the spatial evidence cannot
        distinguish them, and proposing either would risk an identity swap --
        which in an attendance record is strictly worse than a gap.
        """
        return self.n_eligible == 1


class IdentityLedger:
    """
    Tracks which identities have recently vanished and scores whether a newly
    created track is likely to be one of them.

    Not thread-safe by design: one instance per VisionPipeline, touched only
    from that pipeline's camera thread.
    """

    def __init__(
        self,
        retire_frames: int = 60,
        max_entries: int = 50,
        max_centroid_distance: float = 400.0,
    ):
        """
        Args:
            retire_frames: how many detection-frames a retired identity stays
                eligible for recovery. Counted in DETECTION frames, not source
                frames, so it is independent of FRAME_SKIP.
            max_entries: hard cap on the retirement buffer (bounded memory --
                the Phase 1 report flagged several unbounded caches, so this
                one is bounded from the start).
            max_centroid_distance: px. Beyond this a candidate is not
                considered plausible regardless of IoU.
        """
        self.retire_frames = retire_frames
        self.max_entries = max_entries
        self.max_centroid_distance = max_centroid_distance

        self._retired: List[RetiredIdentity] = []
        self._last_seen: Dict[int, dict] = {}   # track_id -> snapshot fields
        self._frame_index = 0
        # Proposals raised this frame, awaiting verification by the recogniser.
        # Consumed exactly once via take_proposal() so a rejected proposal is
        # never retried on the same track.
        self._proposals: Dict[int, RecoveryOpportunity] = {}

    # ── Observation ───────────────────────────────────────────────────────

    def observe(self, active_tracks: list) -> List[RecoveryOpportunity]:
        """
        Called once per DETECTION frame, immediately after ByteTracker.update().

        Does three things, none of which mutate the tracks:
          1. retire any track_id that was present last frame and is now gone
          2. refresh the snapshot of every currently-live track
          3. for every freshly-created track, score it against the retirement
             buffer and return the opportunity

        Returns:
            One RecoveryOpportunity per newly-created track (usually empty).
        """
        self._frame_index += 1
        now = time.time()

        live_ids = {t.track_id for t in active_tracks}

        # ── 1. Retire tracks that vanished ────────────────────────────
        for track_id, snap in list(self._last_seen.items()):
            if track_id not in live_ids:
                self._retire(snap, now)
                del self._last_seen[track_id]

        # ── 2. Find new tracks BEFORE refreshing snapshots ────────────
        # A track on its creation frame has tracklet_len == 0: STrack.activate()
        # zeroes it and STrack.update() only increments on a subsequent match.
        # This is stateless, so it cannot drift out of sync with the tracker.
        opportunities: List[RecoveryOpportunity] = []
        self._proposals.clear()
        for track in active_tracks:
            if track.track_id not in self._last_seen and track.tracklet_len == 0:
                opp = self._score(track)
                opportunities.append(opp)
                if opp.recoverable:
                    self._proposals[track.track_id] = opp

        # ── 3. Refresh snapshots of everything currently live ─────────
        for track in active_tracks:
            self._last_seen[track.track_id] = {
                "track_id": track.track_id,
                "person_id": track.person_id,
                "person_name": track.person_name,
                "bbox": np.asarray(track.bbox, dtype=np.float32).copy(),
                "recognition_confidence": track.recognition_confidence,
                "recognition_locked": track.recognition_locked,
                "identity_fail_streak": track.identity_fail_streak,
                "distance_history": list(getattr(track, "_distance_history", [])),
                "tracklet_len": track.tracklet_len,
            }

        self._expire(now)
        return opportunities

    def _retire(self, snap: dict, now: float):
        """Move a vanished track's snapshot into the retirement buffer."""
        # Only identities worth recovering are worth storing.
        if not snap.get("person_id"):
            return
        self._retired.append(RetiredIdentity(
            track_id=snap["track_id"],
            person_id=snap["person_id"],
            person_name=snap["person_name"],
            bbox=snap["bbox"],
            retired_frame=self._frame_index,
            retired_at=now,
            recognition_confidence=snap["recognition_confidence"],
            recognition_locked=snap["recognition_locked"],
            identity_fail_streak=snap["identity_fail_streak"],
            distance_history=snap["distance_history"],
            tracklet_len=snap["tracklet_len"],
        ))
        if len(self._retired) > self.max_entries:
            self._retired = self._retired[-self.max_entries:]

    def _score(self, track) -> RecoveryOpportunity:
        """Score a newly created track against every eligible retired identity."""
        bbox = np.asarray(track.bbox, dtype=np.float32)
        cx, cy = _centroid(bbox)

        best: Optional[RecoveryCandidate] = None
        n_eligible = 0

        for r in self._retired:
            age_frames = self._frame_index - r.retired_frame
            if age_frames > self.retire_frames:
                continue
            rcx, rcy = r.centroid
            dist = ((cx - rcx) ** 2 + (cy - rcy) ** 2) ** 0.5
            if dist > self.max_centroid_distance:
                continue

            n_eligible += 1
            cand = RecoveryCandidate(
                retired=r,
                iou=_iou(r.bbox, bbox),
                centroid_distance=dist,
                frames_since_retired=age_frames,
                seconds_since_retired=time.time() - r.retired_at,
            )
            # Rank by IoU first, then by proximity -- IoU is the stronger signal
            # when it exists, but is 0 for any real displacement.
            if best is None or (cand.iou, -cand.centroid_distance) > (best.iou, -best.centroid_distance):
                best = cand

        return RecoveryOpportunity(
            track_id=track.track_id,
            bbox=bbox,
            best=best,
            n_eligible=n_eligible,
            n_retired_total=len(self._retired),
        )

    def _expire(self, now: float):
        """Drop retired identities past the recovery window."""
        self._retired = [
            r for r in self._retired
            if (self._frame_index - r.retired_frame) <= self.retire_frames
        ]

    # ── Proposal handoff ──────────────────────────────────────────────────

    def take_proposal(self, track_id: int) -> Optional[RecoveryOpportunity]:
        """
        Consume this frame's recovery proposal for a track, if any.

        Consumed once: a proposal the recogniser rejects is not retried on the
        same track, so a persistent look-alike cannot be re-proposed every frame
        until it eventually passes by chance.
        """
        return self._proposals.pop(track_id, None)

    @staticmethod
    def restore(track, retired: RetiredIdentity, verified_confidence: float,
                lock_threshold: float):
        """
        Write a verified identity, plus its recognition state, back onto a track.

        Restores the temporal context that ByteTrack's own carry-over dropped:
        the rolling distance history (temporal rescue) and the fail streak
        (hysteresis). Without these the track is protected by neither mechanism
        on the frames right after reacquisition, which is exactly when a
        borderline embedding is most likely.

        Two fields are deliberately NOT inherited verbatim:

          confidence  taken from the FRESH verification, not the stale
                      snapshot, so it reflects a real current measurement.

          locked      re-earned, not inherited: requires that the retired track
                      was locked AND the fresh verification is itself strong
                      enough to lock. Inheriting a lock on a borderline
                      verification would suppress re-checks for seconds on an
                      identity we are not actually confident about.
        """
        track.person_id = retired.person_id
        track.person_name = retired.person_name
        track.recognition_confidence = verified_confidence
        track.recognition_time = time.time()
        track.recognition_locked = (
            retired.recognition_locked and verified_confidence >= lock_threshold
        )
        track.identity_fail_streak = retired.identity_fail_streak
        track._distance_history = list(retired.distance_history)

    def forget(self, person_id: str):
        """Drop every retired entry for a person (e.g. after a verified swap)."""
        self._retired = [r for r in self._retired if r.person_id != person_id]

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def retired_count(self) -> int:
        return len(self._retired)

    @property
    def live_count(self) -> int:
        return len(self._last_seen)

    def reset(self):
        self._retired.clear()
        self._last_seen.clear()
        self._frame_index = 0
