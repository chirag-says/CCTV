"""
Data Retention API — Configure and run cleanup policies for old data.
"""

import logging
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Query, HTTPException

from app.core.security import get_current_user, require_admin
from app.db.session import SessionLocal
from app.db.models import DetectionEvent, UnknownFace
from app.services.unknown_service import UnknownService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/retention", tags=["Data Retention"])

# How long an unknown identity may go unseen before it leaves the review queue.
# Archiving is reversible and deletes nothing, so this can be fairly aggressive.
DEFAULT_ARCHIVE_UNSEEN_DAYS = 60


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
            .order_by(DetectionEvent.created_at.asc())
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
            DetectionEvent.created_at >= now - timedelta(days=30)
        ).count()
        events_90d = db.query(DetectionEvent).filter(
            DetectionEvent.created_at >= now - timedelta(days=90)
        ).count()

        # Per-status breakdown: `total` alone hides the number that actually
        # matters, which is how many rows are sitting unreviewed in `pending`.
        # That is the only bucket that grows without a human touching it.
        by_status = UnknownService.count_by_status()
        stale_pending = UnknownService.count_stale_pending(
            DEFAULT_ARCHIVE_UNSEEN_DAYS
        )

        return {
            "events": {
                "total": event_count,
                "last_30_days": events_30d,
                "last_90_days": events_90d,
                "oldest": oldest_event.created_at.isoformat() if oldest_event and oldest_event.created_at else None,
            },
            "unknown_faces": {
                "total": unknown_count,
                "oldest": oldest_unknown.first_seen.isoformat() if oldest_unknown and oldest_unknown.first_seen else None,
                "by_status": by_status,
                "pending": by_status.get("pending", 0),
                "stale_pending": stale_pending,
                "stale_pending_unseen_days": DEFAULT_ARCHIVE_UNSEEN_DAYS,
            },
        }
    finally:
        db.close()


@router.delete("/cleanup")
async def run_cleanup(
    events_older_than_days: int = Query(90, ge=7, le=365, description="Delete events older than N days"),
    dismissed_faces_older_than_days: int = Query(30, ge=7, le=365, description="Delete dismissed faces older than N days"),
    archive_unknowns_unseen_days: int = Query(
        DEFAULT_ARCHIVE_UNSEEN_DAYS, ge=0, le=365,
        description="Archive pending unknowns not seen for N days (0 disables). Reversible — nothing is deleted.",
    ),
    dry_run: bool = Query(True, description="If true, only count — don't delete"),
    current_user: dict = Depends(require_admin),
):
    """
    Run data retention cleanup. Deletes old events and dismissed unknown faces,
    and archives pending unknown identities that have gone quiet.

    Archiving is deliberately separate from deletion. A pending unknown is an
    unreviewed record, so ageing one out must not destroy it: archiving takes it
    off the review queue and out of the pipeline's registry loader while keeping
    the row and its snapshot, and a new sighting restores it automatically.

    Use dry_run=true first to see what would happen.
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)

        # Count events to delete
        event_cutoff = now - timedelta(days=events_older_than_days)
        events_to_delete = db.query(DetectionEvent).filter(
            DetectionEvent.created_at < event_cutoff
        ).count()

        # Count dismissed faces to delete
        face_cutoff = now - timedelta(days=dismissed_faces_older_than_days)
        faces_to_delete = db.query(UnknownFace).filter(
            UnknownFace.status == "dismissed",
            UnknownFace.created_at < face_cutoff,
        ).count()

        # Count pending unknowns that have gone quiet
        unknowns_to_archive = (
            UnknownService.count_stale_pending(archive_unknowns_unseen_days)
            if archive_unknowns_unseen_days > 0 else 0
        )

        if dry_run:
            return {
                "dry_run": True,
                "would_delete": {
                    "events": events_to_delete,
                    "events_cutoff": event_cutoff.isoformat(),
                    "dismissed_faces": faces_to_delete,
                    "faces_cutoff": face_cutoff.isoformat(),
                },
                "would_archive": {
                    "pending_unknowns": unknowns_to_archive,
                    "unseen_days": archive_unknowns_unseen_days,
                },
                "message": "No data deleted. Set dry_run=false to execute.",
            }

        # Actually delete
        deleted_events = db.query(DetectionEvent).filter(
            DetectionEvent.created_at < event_cutoff
        ).delete()

        deleted_faces = db.query(UnknownFace).filter(
            UnknownFace.status == "dismissed",
            UnknownFace.created_at < face_cutoff,
        ).delete()

        db.commit()

        # Archive after the commit: it runs in its own session, and a failure
        # there must not roll back deletions that already succeeded.
        archived_unknowns = UnknownService.archive_stale(archive_unknowns_unseen_days)

        logger.info(
            f"Data retention cleanup: deleted {deleted_events} events, "
            f"{deleted_faces} dismissed faces, archived {archived_unknowns} unknowns"
        )

        return {
            "dry_run": False,
            "deleted": {
                "events": deleted_events,
                "dismissed_faces": deleted_faces,
            },
            "archived": {
                "pending_unknowns": archived_unknowns,
            },
            "message": (
                f"Cleaned up {deleted_events + deleted_faces} records, "
                f"archived {archived_unknowns} unknown identities."
            ),
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Retention cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
