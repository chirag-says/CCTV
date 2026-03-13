"""
Movement Service — CRUD and aggregation for detection_events table.
Now uses SQLAlchemy + PostgreSQL instead of Supabase.
"""

import logging
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import SessionLocal
from app.db.models import DetectionEvent, Person, Camera
from app.core.exceptions import NotFoundException
from app.utils.time_utils import utc_now, utc_now_iso, today_start_utc, days_ago_utc

logger = logging.getLogger(__name__)


def _event_to_dict(event: DetectionEvent) -> dict:
    """Convert a DetectionEvent ORM object to a dict."""
    return {
        "id": event.id,
        "person_id": event.person_id,
        "camera_id": event.camera_id,
        "event_type": event.event_type,
        "subtype": event.subtype,
        "confidence": event.confidence,
        "snapshot_url": event.snapshot_url,
        "metadata": event.metadata_json or {},
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }


class MovementService:
    """Service for movement (detection event) operations."""

    # ── CRUD ─────────────────────────────────────────────────

    @staticmethod
    def create_movement(data: dict) -> dict:
        """Record a new movement event."""
        db: Session = SessionLocal()
        try:
            event = DetectionEvent(
                id=str(uuid4()),
                person_id=data.get("person_id"),
                camera_id=data["camera_id"],
                event_type=data["event_type"],
                subtype=data.get("subtype"),
                confidence=data.get("confidence", 0.0),
                snapshot_url=data.get("snapshot_url"),
                metadata_json=data.get("metadata", {}),
                created_at=datetime.now(timezone.utc),
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            return _event_to_dict(event)
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to record movement: {e}")
            raise Exception("Failed to record movement")
        finally:
            db.close()

    @staticmethod
    def get_movement(movement_id: str) -> dict:
        """Get a specific movement event by ID."""
        db: Session = SessionLocal()
        try:
            event = db.query(DetectionEvent).filter(DetectionEvent.id == movement_id).first()
            if not event:
                raise NotFoundException("Movement", movement_id)
            return _event_to_dict(event)
        finally:
            db.close()

    @staticmethod
    def list_movements(
        person_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List movement events with comprehensive filtering."""
        db: Session = SessionLocal()
        try:
            query = db.query(DetectionEvent)

            if person_id:
                query = query.filter(DetectionEvent.person_id == person_id)
            if camera_id:
                query = query.filter(DetectionEvent.camera_id == camera_id)
            if event_type:
                query = query.filter(DetectionEvent.event_type == event_type)
            if start_date:
                query = query.filter(DetectionEvent.created_at >= start_date)
            if end_date:
                query = query.filter(DetectionEvent.created_at <= end_date)

            total = query.count()
            events = (
                query.order_by(DetectionEvent.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return {
                "data": [_event_to_dict(e) for e in events],
                "total": total,
            }
        finally:
            db.close()

    # ── Aggregations ─────────────────────────────────────────

    @staticmethod
    def get_movement_summary(
        person_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        days: int = 7,
    ) -> dict:
        """Get aggregated movement summary."""
        db: Session = SessionLocal()
        start = days_ago_utc(days)
        try:
            base_query = db.query(DetectionEvent).filter(DetectionEvent.created_at >= start)
            if person_id:
                base_query = base_query.filter(DetectionEvent.person_id == person_id)
            if camera_id:
                base_query = base_query.filter(DetectionEvent.camera_id == camera_id)

            total_events = base_query.count()
            total_entries = base_query.filter(DetectionEvent.event_type == "entry").count()
            total_exits = base_query.filter(DetectionEvent.event_type == "exit").count()
            total_unknown = base_query.filter(DetectionEvent.event_type == "unknown").count()

            # Unique persons
            unique_query = (
                db.query(func.count(func.distinct(DetectionEvent.person_id)))
                .filter(
                    DetectionEvent.created_at >= start,
                    DetectionEvent.person_id.isnot(None),
                )
            )
            if camera_id:
                unique_query = unique_query.filter(DetectionEvent.camera_id == camera_id)
            unique_persons = unique_query.scalar() or 0

            # Average confidence
            avg_query = (
                db.query(func.avg(DetectionEvent.confidence))
                .filter(
                    DetectionEvent.created_at >= start,
                    DetectionEvent.confidence > 0,
                )
            )
            if person_id:
                avg_query = avg_query.filter(DetectionEvent.person_id == person_id)
            if camera_id:
                avg_query = avg_query.filter(DetectionEvent.camera_id == camera_id)
            avg_conf = avg_query.scalar() or 0.0

            return {
                "total_events": total_events,
                "total_entries": total_entries,
                "total_exits": total_exits,
                "total_unknown": total_unknown,
                "unique_persons": unique_persons,
                "avg_confidence": round(float(avg_conf), 3),
                "period_start": start.isoformat(),
                "period_end": utc_now_iso(),
            }
        finally:
            db.close()

    @staticmethod
    def get_person_timeline(
        person_id: str,
        days: int = 7,
        limit: int = 100,
    ) -> List[dict]:
        """Get a chronological timeline of movements for a specific person."""
        db: Session = SessionLocal()
        start = days_ago_utc(days)
        try:
            events = (
                db.query(DetectionEvent)
                .filter(
                    DetectionEvent.person_id == person_id,
                    DetectionEvent.created_at >= start,
                )
                .order_by(DetectionEvent.created_at.desc())
                .limit(limit)
                .all()
            )
            return [_event_to_dict(e) for e in events]
        finally:
            db.close()

    @staticmethod
    def get_camera_activity(
        camera_id: str,
        hours: int = 24,
    ) -> dict:
        """Get movement activity for a specific camera over the last N hours."""
        db: Session = SessionLocal()
        start = utc_now() - timedelta(hours=hours)
        try:
            events = (
                db.query(DetectionEvent)
                .filter(
                    DetectionEvent.camera_id == camera_id,
                    DetectionEvent.created_at >= start,
                )
                .order_by(DetectionEvent.created_at.desc())
                .all()
            )

            # Build hourly distribution
            hourly: Dict[str, int] = {}
            for event in events:
                if event.created_at:
                    hour_key = event.created_at.strftime("%Y-%m-%d %H:00")
                    hourly[hour_key] = hourly.get(hour_key, 0) + 1

            event_dicts = [_event_to_dict(e) for e in events]

            return {
                "hourly_counts": hourly,
                "recent_events": event_dicts[:20],
                "total": len(event_dicts),
            }
        finally:
            db.close()

    @staticmethod
    def delete_old_movements(days: int = 90) -> int:
        """Delete detection events older than N days."""
        db: Session = SessionLocal()
        cutoff = days_ago_utc(days)
        try:
            deleted = (
                db.query(DetectionEvent)
                .filter(DetectionEvent.created_at < cutoff)
                .delete(synchronize_session=False)
            )
            db.commit()
            logger.info(f"Deleted {deleted} movement events older than {days} days")
            return deleted
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete old movements: {e}")
            return 0
        finally:
            db.close()
