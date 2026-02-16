"""
Analytics Service — Dashboard data, occupancy, peak times, movement logs.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

from app.database import get_admin_db
from app.vision.camera_worker import camera_manager

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for analytics and reporting."""

    @staticmethod
    def get_dashboard_summary() -> dict:
        """
        Get overall dashboard summary data.
        
        Returns:
            Dict with occupancy, today's stats, camera status, etc.
        """
        db = get_admin_db()
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Real-time occupancy from tracker
        occupancy = camera_manager.get_all_occupancy()
        camera_status = camera_manager.get_all_status()

        if db is None:
            return {
                "current_occupancy": occupancy,
                "cameras": camera_status,
                "today_entries": 0,
                "today_exits": 0,
                "today_unknown": 0,
                "total_persons": 0,
                "pending_unknowns": 0,
                "active_cameras": camera_status.get("total_active", 0),
            }

        # Today's event counts
        entries = (
            db.table("detection_events")
            .select("id", count="exact")
            .eq("event_type", "entry")
            .gte("created_at", today_start.isoformat())
            .execute()
        )

        exits = (
            db.table("detection_events")
            .select("id", count="exact")
            .eq("event_type", "exit")
            .gte("created_at", today_start.isoformat())
            .execute()
        )

        unknowns = (
            db.table("detection_events")
            .select("id", count="exact")
            .eq("event_type", "unknown")
            .gte("created_at", today_start.isoformat())
            .execute()
        )

        # Total persons
        total_persons = (
            db.table("persons")
            .select("id", count="exact")
            .eq("is_active", True)
            .execute()
        )

        # Pending unknowns
        pending_unknowns = (
            db.table("unknown_faces")
            .select("id", count="exact")
            .eq("status", "pending")
            .execute()
        )

        return {
            "current_occupancy": occupancy,
            "cameras": camera_status,
            "today_entries": entries.count or 0,
            "today_exits": exits.count or 0,
            "today_unknown": unknowns.count or 0,
            "total_persons": total_persons.count or 0,
            "pending_unknowns": pending_unknowns.count or 0,
            "active_cameras": camera_status.get("total_active", 0),
            "timestamp": now.isoformat(),
        }

    @staticmethod
    def get_peak_times(
        days: int = 7,
        granularity: str = "hour",
    ) -> dict:
        """
        Analyze peak entry/exit times.
        
        Args:
            days: Number of days to analyze
            granularity: "hour" or "day"
        """
        db = get_admin_db()

        if db is None:
            # Return mock data
            return {
                "peak_entry_hour": 9,
                "peak_exit_hour": 18,
                "hourly_distribution": {str(h): 0 for h in range(24)},
                "period_days": days,
            }

        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Get all entry events in period
        entries = (
            db.table("detection_events")
            .select("created_at")
            .eq("event_type", "entry")
            .gte("created_at", start_date)
            .execute()
        )

        # Build hourly distribution
        hourly = {str(h): 0 for h in range(24)}
        for event in entries.data or []:
            try:
                dt = datetime.fromisoformat(event["created_at"].replace("Z", "+00:00"))
                hourly[str(dt.hour)] += 1
            except Exception:
                pass

        # Find peak hours
        peak_entry_hour = max(hourly, key=hourly.get) if any(hourly.values()) else "9"

        return {
            "peak_entry_hour": int(peak_entry_hour),
            "hourly_distribution": hourly,
            "period_days": days,
            "total_entries": sum(hourly.values()),
        }

    @staticmethod
    def get_occupancy_data() -> dict:
        """Get real-time occupancy data across all cameras."""
        occupancy = camera_manager.get_all_occupancy()
        all_status = camera_manager.get_all_status()

        # Get active persons across all workers
        active_persons = []
        seen_ids = set()
        for camera_id in camera_manager.get_active_camera_ids():
            status = camera_manager.get_camera_status(camera_id)
            if status:
                for worker in [camera_manager._workers.get(camera_id)]:
                    if worker:
                        for person in worker.pipeline.tracker.get_active_persons():
                            if person["person_id"] not in seen_ids:
                                active_persons.append(person)
                                seen_ids.add(person["person_id"])

        return {
            "current_occupancy": occupancy,
            "active_persons": active_persons,
            "camera_breakdown": all_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def get_movement_logs(
        person_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get detailed movement / detection event logs."""
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

    @staticmethod
    def generate_report(
        report_type: str = "daily",
        date: Optional[str] = None,
    ) -> dict:
        """Generate a summary report."""
        db = get_admin_db()
        now = datetime.now(timezone.utc)

        if report_type == "daily":
            target_date = date or now.strftime("%Y-%m-%d")
            start = f"{target_date}T00:00:00+00:00"
            end = f"{target_date}T23:59:59+00:00"
        elif report_type == "weekly":
            start = (now - timedelta(days=7)).isoformat()
            end = now.isoformat()
        else:
            start = (now - timedelta(days=30)).isoformat()
            end = now.isoformat()

        if db is None:
            return {
                "report_type": report_type,
                "period_start": start,
                "period_end": end,
                "total_entries": 0,
                "total_exits": 0,
                "unique_persons": 0,
                "unknown_faces": 0,
                "avg_duration_sec": 0,
            }

        # Aggregate data
        entries = (
            db.table("detection_events")
            .select("id", count="exact")
            .eq("event_type", "entry")
            .gte("created_at", start)
            .lte("created_at", end)
            .execute()
        )

        exits = (
            db.table("detection_events")
            .select("id", count="exact")
            .eq("event_type", "exit")
            .gte("created_at", start)
            .lte("created_at", end)
            .execute()
        )

        sessions = (
            db.table("tracking_sessions")
            .select("person_id, duration_sec")
            .gte("entry_time", start)
            .lte("entry_time", end)
            .execute()
        )

        unique_persons = set()
        total_duration = 0
        for session in sessions.data or []:
            unique_persons.add(session["person_id"])
            total_duration += session.get("duration_sec", 0)

        avg_duration = total_duration / len(sessions.data) if sessions.data else 0

        return {
            "report_type": report_type,
            "period_start": start,
            "period_end": end,
            "total_entries": entries.count or 0,
            "total_exits": exits.count or 0,
            "unique_persons": len(unique_persons),
            "total_sessions": len(sessions.data or []),
            "avg_duration_sec": round(avg_duration),
        }
