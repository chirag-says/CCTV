"""
Write-time dedup for unknown faces — the authority that decides whether a
sighting is a NEW stranger or one we already have a row for.

WHY THIS EXISTS
Each VisionPipeline has its own UnknownRegistry, which resolves a face to a
stable id from that camera's resident set. Two cameras can therefore mint two
ids for one person at the same moment, and a camera that has evicted somebody
will mint a fresh id for a person the database already knows. The consumer is
the single point that sees every sighting, so it is where those get converged
onto one canonical row.

TWO THINGS WERE WRONG WITH THE PREVIOUS IMPLEMENTATION

1. It compared against `unknown_faces.face_encoding` only — ONE vector, the
   first view ever seen of that person. The same human at a different angle is
   easily further than the match threshold from their own frontal shot, so a
   new row was written, and the next visit did it again. Measured on live
   data: 8,563 of 11,327 pending identities had occurrence=1, and 81% of
   strangers had exactly one visit's worth of events.

   Migration 012 gave every identity several views. This matches best-of-K
   across all of them, which is the same rule UnknownRegistry and the enrolled
   face gallery already use. It does NOT loosen the threshold — it gives the
   comparison more evidence at the same bar. Loosening 0.70 was the obvious
   alternative and the wrong one: FACE_MATCH_LENIENT is 0.72, so anything
   looser starts merging different people.

2. It re-queried every pending row and deserialised every encoding in a Python
   loop, per face. Measured at 280 ms — on the EventQueue consumer thread,
   growing linearly with a table that only gets bigger. Now a cached matrix
   and one matvec.

THREADING
Owned by the EventQueue consumer thread. `add`/`drop`/`match` are called only
from there. The lock guards the periodic reload against API threads that
enroll or dismiss a row and want the index to notice.
"""

import logging
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class UnknownDedupIndex:
    """Vectorised, multi-view index of pending unknown identities."""

    def __init__(self, refresh_sec: float = None, load_limit: int = None):
        self._matrix: Optional[np.ndarray] = None   # (capacity, D) buffer
        self._n_rows = 0
        self._owners: List[str] = []                # unknown_face_id per row
        self._loaded_at = 0.0
        self._lock = threading.RLock()
        self._refresh_sec = (
            refresh_sec if refresh_sec is not None
            else getattr(settings, "UNKNOWN_DEDUP_REFRESH_SEC", 300.0)
        )
        self._load_limit = (
            load_limit if load_limit is not None
            else getattr(settings, "UNKNOWN_DEDUP_LOAD_LIMIT", 20000)
        )
        self.stat_matched = 0
        self.stat_minted = 0
        self.stat_reloads = 0

    # ── Index maintenance ────────────────────────────────────────────────

    @staticmethod
    def _unit(v) -> Optional[np.ndarray]:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        return (a / n).astype(np.float32) if n > 0 else None

    def _rebuild(self, rows):
        views, owners = [], []
        for row in rows:
            for enc in row.get("encodings") or []:
                u = self._unit(enc)
                if u is not None:
                    views.append(u)
                    owners.append(row["id"])
        if not views:
            self._matrix, self._n_rows, self._owners = None, 0, []
            return
        # Headroom so newly written identities append instead of forcing
        # another full rebuild -- the same reason UnknownRegistry does this.
        capacity = max(1024, len(views) * 2)
        self._matrix = np.zeros((capacity, views[0].shape[0]), dtype=np.float32)
        self._matrix[:len(views)] = np.vstack(views)
        self._n_rows = len(views)
        self._owners = owners

    def maybe_refresh(self, force: bool = False) -> bool:
        """Reload from the database if the cache is stale."""
        now = time.time()
        with self._lock:
            if not force and self._matrix is not None:
                if now - self._loaded_at < self._refresh_sec:
                    return False
            from app.services.unknown_service import UnknownService

            try:
                rows = UnknownService.get_pending_encodings(limit=self._load_limit)
            except Exception as e:
                logger.warning(f"Unknown dedup index reload failed: {e}")
                return False
            self._rebuild(rows)
            self._loaded_at = now
            self.stat_reloads += 1
            logger.info(
                f"Unknown dedup index: {len(rows):,} identities, "
                f"{self._n_rows:,} views"
            )
            return True

    def add(self, unknown_id: str, encodings) -> None:
        """Register a row that was just written, so the next face sees it."""
        with self._lock:
            for enc in encodings or []:
                u = self._unit(enc)
                if u is None:
                    continue
                if self._matrix is None or self._n_rows >= self._matrix.shape[0]:
                    # No buffer, or it is full: let the next refresh rebuild.
                    self._loaded_at = 0.0
                    return
                self._matrix[self._n_rows] = u
                self._owners.append(unknown_id)
                self._n_rows += 1

    def drop(self, unknown_id: str) -> None:
        """
        Forget an identity that is no longer pending (enrolled or dismissed).

        Marks the cache stale rather than compacting the matrix: this happens
        at human speed, and a rebuild is cheaper than maintaining holes.
        """
        with self._lock:
            if unknown_id in self._owners:
                self._loaded_at = 0.0

    # ── The question this class exists to answer ─────────────────────────

    def match(
        self, encoding, threshold: float = None
    ) -> Tuple[Optional[str], float]:
        """
        Is this face somebody we already have a row for?

        Returns (unknown_face_id, distance), or (None, inf) when it is a
        genuinely new person. Best-of-K: one matvec scores every view of every
        identity, and argmin gives the nearest view across the whole table.
        """
        if threshold is None:
            threshold = settings.UNKNOWN_SIMILARITY_THRESHOLD

        q = self._unit(encoding)
        if q is None:
            return None, float("inf")

        self.maybe_refresh()

        with self._lock:
            if self._matrix is None or not self._n_rows:
                self.stat_minted += 1
                return None, float("inf")
            distances = 1.0 - (self._matrix[:self._n_rows] @ q)
            best = int(np.argmin(distances))
            best_distance = float(distances[best])
            if best_distance < threshold:
                self.stat_matched += 1
                return self._owners[best], best_distance
            self.stat_minted += 1
            return None, best_distance

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "identities": len(set(self._owners)),
                "views": self._n_rows,
                "matched": self.stat_matched,
                "minted": self.stat_minted,
                "reloads": self.stat_reloads,
                "age_sec": round(time.time() - self._loaded_at, 1),
            }


_index: Optional[UnknownDedupIndex] = None
_index_lock = threading.Lock()


def get_dedup_index() -> UnknownDedupIndex:
    """Process-wide singleton. One table, one index."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = UnknownDedupIndex()
    return _index
