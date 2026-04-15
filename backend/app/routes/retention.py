"""
Data Retention API — Configure and run cleanup policies for old data.
"""

import logging
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Query, HTTPException

from app.core.security import get_current_user, require_admin
from app.db.session import SessionLocal
from app.db.models import DetectionEvent, UnknownFace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/retention", tags=["Data Retention"])


@router.get("/status")
async def get_retention_status(
    current_user: dict = Depends(get_current_user),
):
    """Get current data retention statistics — counts and oldest records."""
    db = SessionLocal()
    try:
        event_count = db.query(DetectionEvent).count()
        unknown_count = db.query(UnknownFace).count()

        oldest_event = (
            db.query(DetectionEvent)
            .order_by(DetectionEvent.timestamp.asc())
            .first()
        )
        oldest_unknown = (
            db.query(UnknownFace)
            .order_by(UnknownFace.first_seen.asc())
            .first()
        )

        # Size estimates
        now = datetime.now(timezone.utc)
        events_30d = db.query(DetectionEvent).filter(
            DetectionEvent.timestamp >= now - timedelta(days=30)
        ).count()
        events_90d = db.query(DetectionEvent).filter(
            DetectionEvent.timestamp >= now - timedelta(days=90)
        ).count()

        return {
            "events": {
                "total": event_count,
                "last_30_days": events_30d,
                "last_90_days": events_90d,
                "oldest": oldest_event.timestamp.isoformat() if oldest_event and oldest_event.timestamp else None,
            },
            "unknown_faces": {
                "total": unknown_count,
                "oldest": oldest_unknown.first_seen.isoformat() if oldest_unknown and oldest_unknown.first_seen else None,
            },
        }
    finally:
        db.close()


@router.delete("/cleanup")
async def run_cleanup(
    events_older_than_days: int = Query(90, ge=7, le=365, description="Delete events older than N days"),
    dismissed_faces_older_than_days: int = Query(30, ge=7, le=365, description="Delete dismissed faces older than N days"),
    dry_run: bool = Query(True, description="If true, only count — don't delete"),
    current_user: dict = Depends(require_admin),
):
    """
    Run data retention cleanup. Deletes old events and dismissed unknown faces.
    Use dry_run=true first to see what would be deleted.
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)

        # Count events to delete
        event_cutoff = now - timedelta(days=events_older_than_days)
        events_to_delete = db.query(DetectionEvent).filter(
            DetectionEvent.timestamp < event_cutoff
        ).count()

        # Count dismissed faces to delete
        face_cutoff = now - timedelta(days=dismissed_faces_older_than_days)
        faces_to_delete = db.query(UnknownFace).filter(
            UnknownFace.status == "dismissed",
            UnknownFace.created_at < face_cutoff,
        ).count()

        if dry_run:
            return {
                "dry_run": True,
                "would_delete": {
                    "events": events_to_delete,
                    "events_cutoff": event_cutoff.isoformat(),
                    "dismissed_faces": faces_to_delete,
                    "faces_cutoff": face_cutoff.isoformat(),
                },
                "message": "No data deleted. Set dry_run=false to execute.",
            }

        # Actually delete
        deleted_events = db.query(DetectionEvent).filter(
            DetectionEvent.timestamp < event_cutoff
        ).delete()

        deleted_faces = db.query(UnknownFace).filter(
            UnknownFace.status == "dismissed",
            UnknownFace.created_at < face_cutoff,
        ).delete()

        db.commit()

        logger.info(
            f"Data retention cleanup: deleted {deleted_events} events, "
            f"{deleted_faces} dismissed faces"
        )

        return {
            "dry_run": False,
            "deleted": {
                "events": deleted_events,
                "dismissed_faces": deleted_faces,
            },
            "message": f"Cleaned up {deleted_events + deleted_faces} records.",
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Retention cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
