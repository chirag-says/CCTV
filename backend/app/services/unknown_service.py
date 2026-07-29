"""
Unknown Face Service — database access for unenrolled identities.

Exists so the vision layer can refresh its UnknownRegistry without embedding
SQLAlchemy queries in the pipeline, matching how PersonService.get_all_encodings()
feeds the face gallery.
"""

import logging
from typing import Any, Dict, List, Optional

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
        """
        db: Session = SessionLocal()
        try:
            row = db.query(UnknownFace).filter(UnknownFace.id == unknown_id).first()
            if row is None:
                return False
            if row.status != "pending":
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
