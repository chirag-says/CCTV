"""
Unknown Face Service — database access for unenrolled identities.

Exists so the vision layer can refresh its UnknownRegistry without embedding
SQLAlchemy queries in the pipeline, matching how PersonService.get_all_encodings()
feeds the face gallery.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import UnknownFace
from app.vision.recognizer import deserialize_encoding

logger = logging.getLogger(__name__)


class UnknownService:
    """Read/write helpers for the unknown_faces table."""

    @staticmethod
    def get_pending_encodings(limit: int = 500) -> List[Dict[str, Any]]:
        """
        Load pending unknown identities that carry a usable encoding.

        Ordered by last_seen DESC and capped, so a long-lived deployment with
        thousands of historical unknowns loads only the recently active ones
        into each pipeline's registry.

        Rows without an encoding (body-only detections) are skipped: they
        cannot be matched against a face embedding.
        """
        db: Session = SessionLocal()
        try:
            rows = (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.status == "pending",
                    UnknownFace.face_encoding.isnot(None),
                )
                .order_by(UnknownFace.last_seen.desc())
                .limit(limit)
                .all()
            )

            out: List[Dict[str, Any]] = []
            for r in rows:
                arr = deserialize_encoding(r.face_encoding)
                if arr is None:
                    continue
                out.append({
                    "id": r.id,
                    "encoding": arr,
                    "occurrence": r.occurrence or 1,
                    "first_seen": r.first_seen.timestamp() if r.first_seen else 0.0,
                    "last_seen": r.last_seen.timestamp() if r.last_seen else 0.0,
                })
            return out
        finally:
            db.close()

    @staticmethod
    def record_sighting(unknown_id: str, camera_id: str, timestamp) -> bool:
        """
        Bump occurrence/last_seen for an already-known unknown identity.

        Used for repeat sightings, which carry no snapshots — the row already
        has them. Returns False if the row is gone (enrolled or dismissed
        between the sighting and this write), which is not an error.

        An ARCHIVED row is reactivated rather than dropped. Archiving means
        "not seen for a long time", so a fresh sighting disproves it. Without
        this, a returning visitor whose row had aged out would keep resolving
        to an archived id in the pipeline's registry and silently lose every
        subsequent sighting.
        """
        db: Session = SessionLocal()
        try:
            row = db.query(UnknownFace).filter(UnknownFace.id == unknown_id).first()
            if row is None:
                return False
            if row.status == "archived":
                logger.info(
                    f"Unknown {unknown_id[:8]}… reappeared after archiving; "
                    f"restored to pending"
                )
                row.status = "pending"
            elif row.status != "pending":
                # Enrolled or dismissed while we were queued; drop silently.
                return False
            row.occurrence = (row.occurrence or 0) + 1
            row.last_seen = timestamp
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to record unknown sighting {unknown_id[:8]}…: {e}")
            return False
        finally:
            db.close()

    # ── Growth control ────────────────────────────────────────────────────
    #
    # The in-memory UnknownRegistry is bounded twice over: get_pending_encodings
    # caps its query and orders by last_seen DESC, and the registry itself holds
    # an LRU of UNKNOWN_REGISTRY_MAX_ENTRIES. So RAM and per-face search cost are
    # already flat no matter how big this table gets.
    #
    # The table is not. Rows only leave `pending` when a human enrolls or
    # dismisses them, so every stranger who ever walked past a camera stays in
    # the review queue forever. That is a review-workload and disk problem, not
    # a recognition one, and archiving is the right shape of fix: it is
    # reversible (see record_sighting), keeps the snapshot for audit, and drops
    # the row out of the pending query the registry loads from.

    @staticmethod
    def count_by_status() -> Dict[str, int]:
        """Row counts per status — how full the review queue actually is."""
        db: Session = SessionLocal()
        try:
            rows = (
                db.query(UnknownFace.status, func.count(UnknownFace.id))
                .group_by(UnknownFace.status)
                .all()
            )
            return {status or "unknown": int(count) for status, count in rows}
        finally:
            db.close()

    @staticmethod
    def count_stale_pending(unseen_days: int) -> int:
        """How many pending rows archive_stale() would touch right now."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=unseen_days)
        db: Session = SessionLocal()
        try:
            return (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.status == "pending",
                    UnknownFace.last_seen < cutoff,
                )
                .count()
            )
        finally:
            db.close()

    @staticmethod
    def archive_stale(unseen_days: int) -> int:
        """
        Move pending unknowns not seen for `unseen_days` to `archived`.

        Keyed on last_seen, not first_seen: a visitor who has been coming back
        for a year is active and must stay in the queue, while one sighting six
        months ago is not worth a reviewer's attention.

        Nothing is deleted. Archived rows remain queryable (`?status=archived`)
        and are restored automatically if the person is seen again.

        Returns the number of rows archived.
        """
        if unseen_days <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=unseen_days)
        db: Session = SessionLocal()
        try:
            archived = (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.status == "pending",
                    UnknownFace.last_seen < cutoff,
                )
                .update({"status": "archived"}, synchronize_session=False)
            )
            db.commit()
            if archived:
                logger.info(
                    f"Archived {archived} unknown identities not seen in "
                    f"{unseen_days} days"
                )
            return int(archived)
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to archive stale unknowns: {e}")
            return 0
        finally:
            db.close()
