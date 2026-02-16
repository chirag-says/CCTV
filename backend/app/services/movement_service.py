"""
Movement Service — CRUD and aggregation for the movements (detection_events) table.

This is the dedicated service layer for the `movements` API.
It wraps the detection_events table with movement-specific query patterns,
aggregations, and timeline generation.
"""

import logging
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app.database import get_admin_db
from app.core.exceptions import NotFoundException
from app.utils.time_utils import utc_now, utc_now_iso, today_start_utc, days_ago_utc

logger = logging.getLogger(__name__)


class MovementService:
    """Service for movement (detection event) operations."""

    # ── CRUD ─────────────────────────────────────────────────

    @staticmethod
    def create_movement(data: dict) -> dict:
        """
        Record a new movement event.

        Args:
            data: Dict with person_id, camera_id, event_type, confidence, etc.

        Returns:
            Created movement record
        """
        db = get_admin_db()

        movement = {
            "id": str(uuid4()),
            "person_id": data.get("person_id"),
            "camera_id": data["camera_id"],
            "event_type": data["event_type"],
            "confidence": data.get("confidence", 0.0),
            "snapshot_url": data.get("snapshot_url"),
            "metadata": data.get("metadata", {}),
            "created_at": utc_now_iso(),
        }

        if db is None:
            return movement

        result = db.table("detection_events").insert(movement).execute()
        if result.data:
            return result.data[0]

        raise Exception("Failed to record movement")

    @staticmethod
    def get_movement(movement_id: str) -> dict:
        """Get a specific movement event by ID."""
        db = get_admin_db()

        if db is None:
            raise NotFoundException("Movement", movement_id)

        result = (
            db.table("detection_events")
            .select("*")
            .eq("id", movement_id)
            .execute()
        )

        if not result.data:
            raise NotFoundException("Movement", movement_id)

        return result.data[0]

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
        """
        List movement events with comprehensive filtering.

        Supports filtering by person, camera, event type, and date range.
        Results are ordered by most recent first.
        """
        db = get_admin_db()

        if db is None:
            return {"data": [], "total": 0}

        query = db.table("detection_events").select("*", count="exact")

        if person_id:
            query = query.eq("person_id", person_id)
        if camera_id:
            query = query.eq("camera_id", camera_id)
        if event_type:
            query = query.eq("event_type", event_type)
        if start_date:
            query = query.gte("created_at", start_date)
        if end_date:
            query = query.lte("created_at", end_date)

        query = query.order("created_at", desc=True)
        query = query.range(offset, offset + limit - 1)

        result = query.execute()

        return {
            "data": result.data or [],
            "total": result.count or 0,
        }

    # ── Aggregations ─────────────────────────────────────────

    @staticmethod
    def get_movement_summary(
        person_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        days: int = 7,
    ) -> dict:
        """
        Get aggregated movement summary.

        Args:
            person_id: Filter to a specific person
            camera_id: Filter to a specific camera
            days: Number of days to look back

        Returns:
            Summary dict with counts and averages
        """
        db = get_admin_db()
        start = days_ago_utc(days).isoformat()

        if db is None:
            return {
                "total_events": 0,
                "total_entries": 0,
                "total_exits": 0,
                "total_unknown": 0,
                "unique_persons": 0,
                "avg_confidence": 0.0,
                "period_start": start,
                "period_end": utc_now_iso(),
            }

        base_filters = {"start_date": start}
        if person_id:
            base_filters["person_id"] = person_id
        if camera_id:
            base_filters["camera_id"] = camera_id

        def _build_query(event_type: Optional[str] = None):
            q = db.table("detection_events").select("id", count="exact")
            q = q.gte("created_at", start)
            if person_id:
                q = q.eq("person_id", person_id)
            if camera_id:
                q = q.eq("camera_id", camera_id)
            if event_type:
                q = q.eq("event_type", event_type)
            return q

        total = _build_query().execute()
        entries = _build_query("entry").execute()
        exits = _build_query("exit").execute()
        unknowns = _build_query("unknown").execute()

        # Get unique persons
        all_events = (
            db.table("detection_events")
            .select("person_id, confidence")
            .gte("created_at", start)
            .not_.is_("person_id", "null")
        )
        if camera_id:
            all_events = all_events.eq("camera_id", camera_id)
        all_result = all_events.execute()

        unique = set()
        total_conf = 0.0
        count_conf = 0
        for row in all_result.data or []:
            if row.get("person_id"):
                unique.add(row["person_id"])
            conf = row.get("confidence", 0)
            if conf and conf > 0:
                total_conf += conf
                count_conf += 1

        return {
            "total_events": total.count or 0,
            "total_entries": entries.count or 0,
            "total_exits": exits.count or 0,
            "total_unknown": unknowns.count or 0,
            "unique_persons": len(unique),
            "avg_confidence": round(total_conf / count_conf, 3) if count_conf else 0.0,
            "period_start": start,
            "period_end": utc_now_iso(),
        }

    @staticmethod
    def get_person_timeline(
        person_id: str,
        days: int = 7,
        limit: int = 100,
    ) -> List[dict]:
        """
        Get a chronological timeline of movements for a specific person.

        Returns a list of events sorted by timestamp (newest first).
        """
        db = get_admin_db()

        if db is None:
            return []

        start = days_ago_utc(days).isoformat()

        result = (
            db.table("detection_events")
            .select("*")
            .eq("person_id", person_id)
            .gte("created_at", start)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return result.data or []

    @staticmethod
    def get_camera_activity(
        camera_id: str,
        hours: int = 24,
    ) -> dict:
        """
        Get movement activity for a specific camera over the last N hours.

        Returns hourly event counts and recent events.
        """
        db = get_admin_db()
        start = (utc_now() - timedelta(hours=hours)).isoformat()

        if db is None:
            return {
                "hourly_counts": {},
                "recent_events": [],
                "total": 0,
            }

        result = (
            db.table("detection_events")
            .select("*")
            .eq("camera_id", camera_id)
            .gte("created_at", start)
            .order("created_at", desc=True)
            .execute()
        )

        # Build hourly distribution
        hourly: Dict[str, int] = {}
        for event in result.data or []:
            try:
                dt = datetime.fromisoformat(
                    event["created_at"].replace("Z", "+00:00")
                )
                hour_key = dt.strftime("%Y-%m-%d %H:00")
                hourly[hour_key] = hourly.get(hour_key, 0) + 1
            except Exception:
                pass

        return {
            "hourly_counts": hourly,
            "recent_events": (result.data or [])[:20],
            "total": len(result.data or []),
        }

    @staticmethod
    def delete_old_movements(days: int = 90) -> int:
        """
        Delete detection events older than N days.
        Useful for data retention policies.

        Returns number of deleted records.
        """
        db = get_admin_db()

        if db is None:
            return 0

        cutoff = days_ago_utc(days).isoformat()

        try:
            result = (
                db.table("detection_events")
                .delete()
                .lt("created_at", cutoff)
                .execute()
            )
            deleted = len(result.data or [])
            logger.info(f"Deleted {deleted} movement events older than {days} days")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete old movements: {e}")
            return 0
