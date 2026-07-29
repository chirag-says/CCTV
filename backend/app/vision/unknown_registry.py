"""
Unknown Registry — stable identities for people who are not enrolled.

THE PROBLEM THIS SOLVES
-----------------------
Before this module, an unrecognised person got a fresh `uuid4()` minted in the
pipeline on every sighting session, and a label ("Unknown #1") derived from a
per-pipeline counter. Meanwhile the EventQueue consumer independently deduped
against the `unknown_faces` table and might merge that sighting into a
*different*, already-existing row — but the canonical id was never reported
back to the pipeline.

Consequences, all of which this module removes:
  * the same stranger accumulated a new `unknown_person_id` in detection_events
    on every restart, so "who was that person" could not be answered
  * "Unknown #1" on camera A and "Unknown #1" on camera B were different people
  * labels reset to #1 on every restart
  * three snapshots were written to disk for a sighting that then deduped into
    an existing row, orphaning the files

DESIGN
------
Mirrors the face gallery. The registry holds an in-memory, vector-indexed set
of pending unknown identities loaded from the database, and resolves a new
embedding to a CANONICAL id — the same id the `unknown_faces` row uses. The
label is derived from that id, so it is stable across restarts and identical on
every camera.

Threading: one instance per VisionPipeline, touched only from that pipeline's
camera thread. No locking, by the same convention as IdentityLedger.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


def make_label(unknown_id: str) -> str:
    """
    Human label derived from the canonical id.

    Deliberately NOT a counter. A counter is per-process and per-camera, so it
    renumbers on restart and collides across cameras. A short id slice is
    stable everywhere, at the cost of being less pretty than "Unknown-001".
    Sequential display numbering, if wanted, belongs in the API layer where a
    stable ordering (e.g. by first_seen) can be computed.
    """
    return f"Unknown-{unknown_id[:6]}"


@dataclass
class UnknownIdentity:
    """One unenrolled person, keyed by the id of their `unknown_faces` row."""
    unknown_id: str
    encoding: np.ndarray          # L2-normalised
    label: str
    first_seen: float
    last_seen: float
    occurrence: int = 1
    persisted: bool = False       # True once a DB row is known to exist
    last_reported: float = 0.0    # last time a sighting was sent to the queue

    @property
    def age(self) -> float:
        return time.time() - self.last_seen


class UnknownRegistry:
    """
    Resolves face embeddings to stable unknown identities.

    Bounded by construction: `max_entries` with LRU-by-last_seen eviction. The
    structure this replaces (`VisionPipeline._unknown_cache`) had no pruning at
    all and was scanned linearly on every unknown face, so both its memory and
    its per-face cost grew for the lifetime of the camera.
    """

    def __init__(
        self,
        match_threshold: float = 0.70,
        max_entries: int = 500,
        sighting_cooldown: float = 30.0,
        refresh_interval: float = 60.0,
    ):
        """
        Args:
            match_threshold: cosine DISTANCE below which two embeddings are the
                same person. Shared with the consumer-side DB dedup so both
                sides agree (settings.UNKNOWN_SIMILARITY_THRESHOLD).
            max_entries: LRU cap on resident identities.
            sighting_cooldown: minimum seconds between reporting repeat
                sightings of the same person, so `occurrence` counts visits
                rather than frames.
            refresh_interval: seconds between reloads of pending unknowns.
        """
        self.match_threshold = match_threshold
        self.max_entries = max_entries
        self.sighting_cooldown = sighting_cooldown
        self.refresh_interval = refresh_interval

        self._identities: Dict[str, UnknownIdentity] = {}
        self._matrix: Optional[np.ndarray] = None   # (N, D) stacked encodings
        self._ids: List[str] = []                   # row order of _matrix
        self._dirty = True
        self._last_refresh = 0.0

    # ── Index ─────────────────────────────────────────────────────────────

    def _rebuild(self):
        """Rebuild the search matrix. Cheap; called only when membership changes."""
        if not self._identities:
            self._matrix, self._ids = None, []
        else:
            self._ids = list(self._identities.keys())
            self._matrix = np.vstack([self._identities[i].encoding for i in self._ids])
        self._dirty = False

    @staticmethod
    def _normalise(encoding: np.ndarray) -> Optional[np.ndarray]:
        n = np.linalg.norm(encoding)
        if n <= 0:
            return None
        return (encoding / n).astype(np.float32)

    # ── Resolution ────────────────────────────────────────────────────────

    def resolve(self, encoding: np.ndarray) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Map an embedding to a stable unknown identity.

        Returns:
            (unknown_id, label, is_new). is_new True means no existing identity
            matched and a fresh canonical id was minted; the caller is expected
            to persist it. (None, None, False) on an unusable embedding.
        """
        q = self._normalise(encoding)
        if q is None:
            return None, None, False

        if self._dirty:
            self._rebuild()

        # Vectorised: one matvec instead of the per-entry Python loop this
        # replaces, so cost stays flat as the registry fills.
        if self._matrix is not None and len(self._ids):
            distances = 1.0 - (self._matrix @ q)
            best = int(np.argmin(distances))
            if float(distances[best]) < self.match_threshold:
                uid = self._ids[best]
                ident = self._identities[uid]
                ident.last_seen = time.time()
                ident.occurrence += 1
                return uid, ident.label, False

        uid = str(uuid4())
        label = make_label(uid)
        now = time.time()
        self._identities[uid] = UnknownIdentity(
            unknown_id=uid, encoding=q, label=label,
            first_seen=now, last_seen=now, occurrence=1, persisted=False,
        )
        self._dirty = True
        self._evict()
        return uid, label, True

    def should_report_sighting(self, unknown_id: str) -> bool:
        """
        Rate-limit repeat sighting reports so `occurrence` counts visits, not
        frames. Without this, reporting every frame would flood the EventQueue
        and make the occurrence count meaningless.
        """
        ident = self._identities.get(unknown_id)
        if ident is None:
            return False
        now = time.time()
        if now - ident.last_reported < self.sighting_cooldown:
            return False
        ident.last_reported = now
        return True

    def mark_persisted(self, unknown_id: str, canonical_id: Optional[str] = None):
        """
        Confirm a DB row exists. If the consumer deduped this sighting into a
        DIFFERENT pre-existing row (another camera saw them first), re-key the
        local entry onto that canonical id so both cameras converge.
        """
        ident = self._identities.get(unknown_id)
        if ident is None:
            return
        if canonical_id and canonical_id != unknown_id:
            logger.info(
                f"Unknown {unknown_id[:8]}… converged onto existing "
                f"{canonical_id[:8]}… (another camera saw them first)"
            )
            self._identities.pop(unknown_id, None)
            if canonical_id not in self._identities:
                ident.unknown_id = canonical_id
                ident.label = make_label(canonical_id)
                ident.persisted = True
                self._identities[canonical_id] = ident
            self._dirty = True
        else:
            ident.persisted = True

    # ── Database sync ─────────────────────────────────────────────────────

    def maybe_refresh(self, loader) -> bool:
        """
        Periodically reload pending unknowns from the database.

        Mirrors VisionPipeline._auto_refresh_encodings: it is how a pipeline
        learns about identities created by ANOTHER camera, or before this
        process started. Without it, stable ids would only be stable within one
        pipeline's lifetime.

        Args:
            loader: zero-arg callable returning
                    [{"id": str, "encoding": bytes|ndarray, "occurrence": int,
                      "first_seen": float, "last_seen": float}, ...]

        Returns:
            True if a refresh ran.
        """
        now = time.time()
        if now - self._last_refresh < self.refresh_interval:
            return False
        self._last_refresh = now

        try:
            rows = loader()
        except Exception as e:
            logger.warning(f"UnknownRegistry refresh failed: {e}")
            return False

        added = 0
        for row in rows:
            uid = row.get("id")
            if not uid or uid in self._identities:
                continue
            enc = row.get("encoding")
            if enc is None:
                continue
            arr = self._normalise(np.asarray(enc, dtype=np.float32))
            if arr is None:
                continue
            self._identities[uid] = UnknownIdentity(
                unknown_id=uid,
                encoding=arr,
                label=make_label(uid),
                first_seen=row.get("first_seen", now),
                last_seen=row.get("last_seen", now),
                occurrence=row.get("occurrence", 1),
                persisted=True,
            )
            added += 1

        if added:
            self._dirty = True
            self._evict()
            logger.info(
                f"UnknownRegistry: loaded {added} pending unknown identities "
                f"({len(self._identities)} resident)"
            )
        return True

    # ── Housekeeping ──────────────────────────────────────────────────────

    def _evict(self):
        """LRU eviction by last_seen. Unpersisted entries are never evicted."""
        if len(self._identities) <= self.max_entries:
            return
        evictable = [i for i in self._identities.values() if i.persisted]
        evictable.sort(key=lambda i: i.last_seen)
        overflow = len(self._identities) - self.max_entries
        for ident in evictable[:overflow]:
            self._identities.pop(ident.unknown_id, None)
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._identities)

    def get(self, unknown_id: str) -> Optional[UnknownIdentity]:
        return self._identities.get(unknown_id)

    def reset(self):
        self._identities.clear()
        self._matrix, self._ids = None, []
        self._dirty = True
