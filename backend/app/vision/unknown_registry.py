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

RE-IDENTIFICATION (Phase 3)
---------------------------
An identity holds SEVERAL embeddings, not one, and matches best-of-K -- the
same rule the enrolled face gallery uses.

The single-embedding version froze whatever view happened to arrive first. A
stranger first seen frontally then turning to profile exceeded the match
threshold against that one frozen frontal vector, so the registry minted a
second id and the consumer's DB dedup (comparing against the same one stored
vector) agreed it was a new person. One human became many rows. Measured on the
live database before this change: 4,129 of 4,452 pending unknowns had
occurrence=1, i.e. dedup was effectively not working.

Views are added only when a sighting matched but was not trivially close --
that is the definition of a view the identity does not already cover. Near
duplicates are rejected, so K stays small and each slot earns its place.

Threading: one instance per VisionPipeline, touched only from that pipeline's
camera thread -- with one exception. The EventQueue consumer learns the
CANONICAL row id after the fact and must report it back; it does that through
submit_persistence_result(), a queue, which the camera thread drains. The hot
path stays lock-free.
"""

import logging
import queue
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
    encodings: List[np.ndarray]   # L2-normalised views of this person, best-of-K
    label: str
    first_seen: float
    last_seen: float
    occurrence: int = 1
    persisted: bool = False       # True once a DB row is known to exist
    last_reported: float = 0.0    # last time a sighting was sent to the queue

    @property
    def age(self) -> float:
        return time.time() - self.last_seen

    @property
    def encoding(self) -> np.ndarray:
        """
        The first view ever seen. This is what gets written to the DB row, so
        it must stay stable: the consumer's dedup and a restart both key off it.
        """
        return self.encodings[0]

    @property
    def view_count(self) -> int:
        return len(self.encodings)


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
        max_entries: int = None,
        sighting_cooldown: float = 30.0,
        refresh_interval: float = 60.0,
        max_views: int = None,
        learn_min_distance: float = 0.30,
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
            max_views: embeddings retained per identity. Small on purpose --
                this is a short-lived re-identification cache, not a gallery,
                and every extra view costs a row in the search matrix for every
                stranger who ever walked past.
            learn_min_distance: a matched sighting is only kept as a new view if
                it is at least this far from every view already held. Closer
                than this it is the same pose again and adds nothing but cost.
        """
        from app.config import settings as _settings
        self.match_threshold = match_threshold
        self.max_entries = (
            max_entries if max_entries is not None
            else getattr(_settings, "UNKNOWN_REGISTRY_MAX_ENTRIES", 5000)
        )
        self.sighting_cooldown = sighting_cooldown
        self.refresh_interval = refresh_interval
        self.max_views = max(1, (
            max_views if max_views is not None
            else getattr(_settings, "UNKNOWN_REGISTRY_MAX_VIEWS", 5)
        ))
        self.learn_min_distance = learn_min_distance

        self._identities: Dict[str, UnknownIdentity] = {}
        self._matrix: Optional[np.ndarray] = None   # (V, D) stacked views
        self._ids: List[str] = []                   # owner id per matrix row
        self._owner_rows: Dict[str, np.ndarray] = {}  # id -> its row indices
        self._n_rows = 0                            # rows in use in _matrix
        self._dirty = True
        self._last_refresh = 0.0

        # Canonical ids reported back by the EventQueue consumer thread.
        self._persistence_inbox: "queue.SimpleQueue[Tuple[str, Optional[str]]]" = (
            queue.SimpleQueue()
        )

        # Views learned in memory that the database does not have yet.
        # Drained by the pipeline and shipped to the EventQueue consumer.
        # Before migration 012 these were never written anywhere, so every
        # refresh, restart and second camera rebuilt each identity from its one
        # frozen first view -- which is why cross-camera re-identification
        # measured 0.0%.
        self._unsaved_views: List[Tuple[str, np.ndarray]] = []

        # Counters for the overnight validation report.
        self.stat_resolved = 0
        self.stat_minted = 0
        self.stat_matched = 0
        self.stat_views_learned = 0
        self.stat_converged = 0
        self.stat_evicted = 0

    # ── Index ─────────────────────────────────────────────────────────────

    def _rebuild(self):
        """
        Rebuild the search matrix. One row per VIEW, not per identity, with
        _ids[row] naming the owner -- so a single matvec scores every view and
        argmin gives best-of-K across the whole registry at once.
        """
        if not self._identities:
            self._matrix, self._ids = None, []
            self._owner_rows = {}
            self._n_rows = 0
        else:
            rows: List[np.ndarray] = []
            owners: List[str] = []
            owner_rows: Dict[str, List[int]] = {}
            for uid, ident in self._identities.items():
                for view in ident.encodings:
                    owner_rows.setdefault(uid, []).append(len(rows))
                    rows.append(view)
                    owners.append(uid)
            # Allocated with headroom so a newly minted identity can be
            # APPENDED rather than triggering another full rebuild. Minting is
            # not rare, and vstacking 25k rows on each one measured at 40ms --
            # more than a whole frame budget at 30fps.
            need = len(rows)
            capacity = max(1024, need * 2)
            self._matrix = np.zeros((capacity, rows[0].shape[0]), dtype=np.float32)
            self._matrix[:need] = np.vstack(rows)
            self._n_rows = need
            self._ids = owners
            # Row indices per identity, so _maybe_learn_view can find an
            # identity's own rows by fancy-indexing instead of scanning every
            # row of the matrix in Python. At the pre-Step-3.5 cap (500
            # entries, mostly one view each) the scan was invisible; at 5000
            # entries x 5 views it is ~25k iterations on every matched face.
            self._owner_rows = {k: np.asarray(v, dtype=np.intp)
                                for k, v in owner_rows.items()}
        self._dirty = False

    def _append_row(self, uid: str, view: np.ndarray):
        """
        Add one view to the search matrix in place.

        The alternative -- flagging dirty and rebuilding -- is O(total views)
        and runs on the face path every time an identity is minted or learns a
        pose. Amortised O(1) here, with a doubling rebuild when the buffer
        fills.
        """
        if self._matrix is None or self._dirty:
            self._dirty = True   # a full rebuild is already pending
            return
        if self._n_rows >= self._matrix.shape[0]:
            self._dirty = True   # buffer full; _rebuild() will double it
            return
        self._matrix[self._n_rows] = view
        self._ids.append(uid)
        existing = self._owner_rows.get(uid)
        idx = np.asarray([self._n_rows], dtype=np.intp)
        self._owner_rows[uid] = (
            idx if existing is None else np.concatenate([existing, idx])
        )
        self._n_rows += 1

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

        self.stat_resolved += 1

        if self._dirty:
            self._rebuild()

        # Vectorised: one matvec scores every view of every identity, so cost
        # stays flat as the registry fills and best-of-K comes free.
        if self._matrix is not None and self._n_rows:
            distances = 1.0 - (self._matrix[:self._n_rows] @ q)
            best = int(np.argmin(distances))
            best_distance = float(distances[best])
            if best_distance < self.match_threshold:
                uid = self._ids[best]
                ident = self._identities[uid]
                ident.last_seen = time.time()
                ident.occurrence += 1
                self.stat_matched += 1
                self._maybe_learn_view(ident, q, distances)
                return uid, ident.label, False

        uid = str(uuid4())
        label = make_label(uid)
        now = time.time()
        self._identities[uid] = UnknownIdentity(
            unknown_id=uid, encodings=[q], label=label,
            first_seen=now, last_seen=now, occurrence=1, persisted=False,
        )
        self._append_row(uid, q)
        self.stat_minted += 1
        self._evict()
        return uid, label, True

    def _maybe_learn_view(
        self, ident: UnknownIdentity, q: np.ndarray, distances: np.ndarray
    ):
        """
        Keep this sighting as an additional view if it covers new ground.

        The gate is the distance to the identity's CLOSEST existing view. Below
        learn_min_distance it is the same angle again -- redundant, and storing
        it would push out a view that actually differs. Above it, this is a pose
        or lighting the identity cannot currently match, which is exactly the
        gap that used to cause a duplicate id to be minted.

        Never grows past max_views, so a person loitering all night costs a
        fixed amount of memory and search time.
        """
        if ident.view_count >= self.max_views:
            return

        rows = self._owner_rows.get(ident.unknown_id)
        if rows is None or not len(rows):
            return
        own = float(distances[rows].min())
        if own < self.learn_min_distance:
            return

        ident.encodings.append(q)
        self._append_row(ident.unknown_id, q)
        self.stat_views_learned += 1
        self._queue_view(ident, q)
        logger.debug(
            f"Unknown {ident.unknown_id[:8]}… learned view {ident.view_count} "
            f"(distance {own:.4f} from nearest existing view)"
        )

    def _queue_view(self, ident: UnknownIdentity, view: np.ndarray):
        """
        Mark a view for persistence.

        Only for identities that already have a DB row. An identity minted
        this frame has no row to attach views to yet; its views are flushed by
        mark_persisted() once the consumer reports the canonical id, so
        nothing learned before persistence is lost.
        """
        if not ident.persisted:
            return
        self._unsaved_views.append((ident.unknown_id, view))

    def drain_new_views(self) -> List[Tuple[str, np.ndarray]]:
        """
        Take the views awaiting a database write. Called from the camera
        thread; the caller is responsible for shipping them to the consumer.
        """
        if not self._unsaved_views:
            return []
        out = self._unsaved_views
        self._unsaved_views = []
        return out

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

    def submit_persistence_result(
        self, unknown_id: str, canonical_id: Optional[str] = None
    ):
        """
        Report a DB write result from ANOTHER thread (the EventQueue consumer).

        Thread-safe by being a queue append and nothing else. The camera thread
        applies it in drain_persistence_results(), so the search structures are
        still only ever mutated from one thread.
        """
        self._persistence_inbox.put((unknown_id, canonical_id))

    def drain_persistence_results(self) -> int:
        """Apply queued canonical ids. Camera thread only. Returns count applied."""
        applied = 0
        while True:
            try:
                unknown_id, canonical_id = self._persistence_inbox.get_nowait()
            except queue.Empty:
                break
            self.mark_persisted(unknown_id, canonical_id)
            applied += 1
        return applied

    def mark_persisted(self, unknown_id: str, canonical_id: Optional[str] = None):
        """
        Confirm a DB row exists. If the consumer deduped this sighting into a
        DIFFERENT pre-existing row (another camera saw them first), re-key the
        local entry onto that canonical id so both cameras converge.

        Until this was actually wired up it was dead code, with two effects:
        convergence never happened, and -- because _evict() only ever removes
        persisted entries -- nothing minted locally was evictable, so the
        registry grew without bound for the life of the camera despite its cap.
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
            existing = self._identities.get(canonical_id)
            if existing is None:
                ident.unknown_id = canonical_id
                ident.label = make_label(canonical_id)
                ident.persisted = True
                self._identities[canonical_id] = ident
            else:
                # Both ids are resident: fold this one's views into the winner
                # rather than discarding them. They are the same person, so the
                # views are exactly the extra coverage the survivor wants.
                for view in ident.encodings:
                    if existing.view_count >= self.max_views:
                        break
                    existing.encodings.append(view)
                existing.occurrence += ident.occurrence
                existing.first_seen = min(existing.first_seen, ident.first_seen)
            self.stat_converged += 1
            self._dirty = True
        else:
            ident.persisted = True

        # Views learned before the row existed were skipped by _queue_view.
        # Flush them now, minus encodings[0] -- that one IS the row's
        # `face_encoding` and is seeded as view #1 by migration 012.
        final = self._identities.get(canonical_id or unknown_id)
        if final is not None and final.persisted:
            for view in final.encodings[1:]:
                self._unsaved_views.append((final.unknown_id, view))

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
                    [{"id": str, "encodings": [ndarray, ...], "occurrence": int,
                      "first_seen": float, "last_seen": float}, ...]

                    `encodings` is a LIST as of migration 012. Loading a single
                    vector here was what silently reduced best-of-K matching to
                    best-of-1 on every camera but the one that first saw the
                    person. A legacy single "encoding" key is still accepted.

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
            raw = row.get("encodings")
            if raw is None:
                legacy = row.get("encoding")
                raw = [legacy] if legacy is not None else []
            views = []
            for enc in raw:
                if enc is None:
                    continue
                arr = self._normalise(np.asarray(enc, dtype=np.float32))
                if arr is not None:
                    views.append(arr)
                if len(views) >= self.max_views:
                    break
            if not views:
                continue
            self._identities[uid] = UnknownIdentity(
                unknown_id=uid,
                encodings=views,
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
            total_views = sum(i.view_count for i in self._identities.values())
            logger.info(
                f"UnknownRegistry: loaded {added} pending unknown identities "
                f"({len(self._identities)} resident, {total_views} views)"
            )
        return True

    # ── Housekeeping ──────────────────────────────────────────────────────

    def _evict(self):
        """
        LRU eviction by last_seen.

        Unpersisted entries are protected so an identity cannot be lost before
        its DB row is written -- but only for a grace period. Without the
        grace, a stalled or failed consumer would leave every entry
        unpersisted, `evictable` empty, and the cap unenforceable; the registry
        would then grow for the life of the camera. Protection that cannot
        expire is not protection, it is a leak.
        """
        if len(self._identities) <= self.max_entries:
            return

        now = time.time()
        grace = max(self.sighting_cooldown * 4, 60.0)
        evictable = [
            i for i in self._identities.values()
            if i.persisted or (now - i.first_seen) > grace
        ]
        evictable.sort(key=lambda i: i.last_seen)

        # Evict in BATCHES, down to a low-water mark, not just back to the
        # cap. Trimming exactly one identity per mint means a full matrix
        # rebuild per mint once the registry is full -- measured at ~40ms on a
        # 25k-row matrix, which is more than a whole frame at 30fps. Dropping
        # to 90% amortises that rebuild over the next ~10% of the cap.
        target = int(self.max_entries * 0.9)
        overflow = max(0, len(self._identities) - target)
        removed = 0
        for ident in evictable[:overflow]:
            self._identities.pop(ident.unknown_id, None)
            self.stat_evicted += 1
            removed += 1

        if len(self._identities) > self.max_entries:
            logger.warning(
                f"UnknownRegistry over cap: {len(self._identities)} resident vs "
                f"{self.max_entries} max, only {len(evictable)} evictable. "
                f"Is the event queue consumer keeping up?"
            )
        # Only when something actually went: an unconditional flag here made
        # every over-cap mint rebuild the matrix even when nothing was removed.
        if removed:
            self._dirty = True

    def stats(self) -> Dict[str, int]:
        """Aggregate counters for the validation report. Cheap; no allocation."""
        views = sum(i.view_count for i in self._identities.values())
        return {
            "resident": len(self._identities),
            "views": views,
            "persisted": sum(1 for i in self._identities.values() if i.persisted),
            "resolved": self.stat_resolved,
            "minted": self.stat_minted,
            "matched": self.stat_matched,
            "views_learned": self.stat_views_learned,
            "converged": self.stat_converged,
            "evicted": self.stat_evicted,
        }

    @property
    def size(self) -> int:
        return len(self._identities)

    def get(self, unknown_id: str) -> Optional[UnknownIdentity]:
        return self._identities.get(unknown_id)

    def reset(self):
        self._identities.clear()
        self._matrix, self._ids = None, []
        self._dirty = True
