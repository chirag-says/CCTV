"""
Analytics Service — Dashboard data, occupancy, peak times, movement logs.
Now uses SQLAlchemy + PostgreSQL instead of Supabase.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import SessionLocal
from app.db.models import Camera, DetectionEvent, Person, TrackingSession, UnknownFace
from app.vision.camera_worker import camera_manager

logger = logging.getLogger(__name__)


def _event_to_dict(event: DetectionEvent) -> dict:
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


class AnalyticsService:
    """Service for analytics and reporting."""

    @staticmethod
    def get_dashboard_summary() -> dict:
        """Get overall dashboard summary data."""
        db: Session = SessionLocal()
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Real-time occupancy from tracker
        occupancy = camera_manager.get_all_occupancy()
        camera_status = camera_manager.get_all_status()

        try:
            today_entries = (
                db.query(func.count(DetectionEvent.id))
                .filter(
                    DetectionEvent.event_type == "entry",
                    DetectionEvent.created_at >= today_start,
                )
                .scalar() or 0
            )

            today_exits = (
                db.query(func.count(DetectionEvent.id))
                .filter(
                    DetectionEvent.event_type == "exit",
                    DetectionEvent.created_at >= today_start,
                )
                .scalar() or 0
            )

            today_unknown = (
                db.query(func.count(DetectionEvent.id))
                .filter(
                    DetectionEvent.event_type == "unknown",
                    DetectionEvent.created_at >= today_start,
                )
                .scalar() or 0
            )

            total_persons = (
                db.query(func.count(Person.id))
                .filter(Person.is_active == True)
                .scalar() or 0
            )

            pending_unknowns = (
                db.query(func.count(UnknownFace.id))
                .filter(UnknownFace.status == "pending")
                .scalar() or 0
            )

            return {
                "current_occupancy": occupancy,
                "cameras": camera_status,
                "today_entries": today_entries,
                "today_exits": today_exits,
                "today_unknown": today_unknown,
                "total_persons": total_persons,
                "pending_unknowns": pending_unknowns,
                "active_cameras": camera_status.get("total_active", 0),
                "timestamp": now.isoformat(),
            }
        except Exception as e:
            logger.error(f"Dashboard summary error: {e}")
            return {
                "current_occupancy": occupancy,
                "cameras": camera_status,
                "today_entries": 0,
                "today_exits": 0,
                "today_unknown": 0,
                "total_persons": 0,
                "pending_unknowns": 0,
                "active_cameras": camera_status.get("total_active", 0),
                "timestamp": now.isoformat(),
            }
        finally:
            db.close()

    @staticmethod
    def get_hourly_distribution(
        event_type: str = "entry",
        days: int = 7,
        start_utc: Optional[datetime] = None,
        end_utc: Optional[datetime] = None,
    ) -> dict:
        """
        Bucket events by CAMPUS-LOCAL hour of day.

        Local hour is the only bucketing that means anything to a reader: an
        08:42 arrival belongs in the 08:00 bucket, not in whatever bucket the
        UTC offset lands it in.

        The explicit to_campus_local() call is deliberate hardening, not a bug
        fix. It currently changes nothing, because this deployment's Postgres
        runs with `TimeZone = Asia/Calcutta` and so psycopg2 already returns
        TIMESTAMPTZ values converted to +05:30 — reading `.hour` directly
        happens to give the local hour. That is a property of the SERVER
        CONFIGURATION, not of the code: point this at a database with the far
        more common `TimeZone = UTC` and the same `.hour` silently reports the
        morning rush as happening at 3am. Converting explicitly makes the
        result depend on CAMPUS_TIMEZONE rather than on a DBA's setting.

        Args:
            event_type: entry / exit / unknown / vehicle_entry / …
            days:       lookback window, used when explicit bounds are absent
            start_utc:  explicit window start (overrides `days`)
            end_utc:    explicit window end

        Returns:
            {"hourly": {"0".."23": count}, "peak_hour": int|None, "total": int}
        """
        from app.utils.time_utils import to_campus_local

        db: Session = SessionLocal()
        try:
            q = db.query(DetectionEvent.created_at).filter(
                DetectionEvent.event_type == event_type
            )
            if start_utc is not None:
                q = q.filter(DetectionEvent.created_at >= start_utc)
            if end_utc is not None:
                q = q.filter(DetectionEvent.created_at < end_utc)
            if start_utc is None and end_utc is None:
                q = q.filter(
                    DetectionEvent.created_at
                    >= datetime.now(timezone.utc) - timedelta(days=days)
                )

            hourly = {str(h): 0 for h in range(24)}
            for (created_at,) in q.all():
                local = to_campus_local(created_at)
                if local is not None:
                    hourly[str(local.hour)] += 1

            total = sum(hourly.values())
            peak = int(max(hourly, key=hourly.get)) if total else None
            return {"hourly": hourly, "peak_hour": peak, "total": total}
        finally:
            db.close()

    @staticmethod
    def get_peak_times(
        days: int = 7,
        granularity: str = "hour",
    ) -> dict:
        """
        Analyze peak entry times, bucketed by campus-local hour.

        Kept for the existing dashboard and analytics pages; the real work is
        in get_hourly_distribution().
        """
        dist = AnalyticsService.get_hourly_distribution(event_type="entry", days=days)
        return {
            # 9 is the historical fallback for "no data at all".
            "peak_entry_hour": dist["peak_hour"] if dist["peak_hour"] is not None else 9,
            "hourly_distribution": dist["hourly"],
            "period_days": days,
            "total_entries": dist["total"],
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

    @staticmethod
    def get_entry_exit_export(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        camera_id: Optional[str] = None,
        person_id: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Flat, human-readable entry/exit rows for CSV / Excel export.

        Deliberately NOT the same shape as get_movement_logs(). That returns raw
        person_id / camera_id UUIDs, which are useless in a spreadsheet handed to
        a security office. This resolves them to names via outer joins, so an
        unmatched camera or a soft-deleted person still yields a usable row
        rather than disappearing.

        Unknown persons are represented by the id the FK guard stashed in
        metadata_json.unknown_person_id, since detection_events.person_id is
        NULL for them by design.

        Returns:
            List of ordered dicts, one per event, newest first.
        """
        db: Session = SessionLocal()
        try:
            types = event_types or ["entry", "exit"]
            q = (
                db.query(DetectionEvent, Person, Camera)
                .outerjoin(Person, DetectionEvent.person_id == Person.id)
                .outerjoin(Camera, DetectionEvent.camera_id == Camera.id)
                .filter(DetectionEvent.event_type.in_(types))
            )
            if start_date:
                q = q.filter(DetectionEvent.created_at >= start_date)
            if end_date:
                q = q.filter(DetectionEvent.created_at <= end_date)
            if camera_id:
                q = q.filter(DetectionEvent.camera_id == camera_id)
            if person_id:
                q = q.filter(DetectionEvent.person_id == person_id)

            results = (
                q.order_by(DetectionEvent.created_at.desc())
                .limit(limit)
                .all()
            )

            rows: List[Dict[str, Any]] = []
            for ev, person, cam in results:
                meta = ev.metadata_json or {}
                ts = ev.created_at

                if person is not None:
                    name = person.name
                    role = person.role or ""
                    dept = person.department or ""
                elif meta.get("unknown_person_id"):
                    name = "Unknown"
                    role = "unknown"
                    dept = ""
                else:
                    name = meta.get("person_name") or "Unknown"
                    role = ""
                    dept = ""

                duration = meta.get("duration_sec")

                rows.append({
                    "Date": ts.strftime("%Y-%m-%d") if ts else "",
                    "Time": ts.strftime("%H:%M:%S") if ts else "",
                    "Event": (ev.event_type or "").upper(),
                    "Person": name,
                    "Role": role,
                    "Department": dept,
                    "Camera": (cam.name if cam else "") or ev.camera_id or "",
                    "Location": (cam.location if cam else "") or "",
                    "Confidence": round(ev.confidence or 0.0, 3),
                    "Duration (s)": duration if duration is not None else "",
                    "Timestamp (UTC)": ts.isoformat() if ts else "",
                    "Person ID": ev.person_id or meta.get("unknown_person_id") or "",
                    "Event ID": ev.id,
                })

            logger.info(
                f"Entry/exit export: {len(rows)} rows "
                f"(types={types}, range={start_date or 'all'}..{end_date or 'now'})"
            )
            return rows
        finally:
            db.close()

    @staticmethod
    def generate_report(
        report_type: str = "daily",
        date: Optional[str] = None,
    ) -> dict:
        """Generate a summary report."""
        db: Session = SessionLocal()
        now = datetime.now(timezone.utc)

        if report_type == "daily":
            target_date = date or now.strftime("%Y-%m-%d")
            start = datetime.fromisoformat(f"{target_date}T00:00:00+00:00")
            end = datetime.fromisoformat(f"{target_date}T23:59:59+00:00")
        elif report_type == "weekly":
            start = now - timedelta(days=7)
            end = now
        else:
            start = now - timedelta(days=30)
            end = now

        try:
            total_entries = (
                db.query(func.count(DetectionEvent.id))
                .filter(
                    DetectionEvent.event_type == "entry",
                    DetectionEvent.created_at >= start,
                    DetectionEvent.created_at <= end,
                )
                .scalar() or 0
            )

            total_exits = (
                db.query(func.count(DetectionEvent.id))
                .filter(
                    DetectionEvent.event_type == "exit",
                    DetectionEvent.created_at >= start,
                    DetectionEvent.created_at <= end,
                )
                .scalar() or 0
            )

            # Tracking sessions in period
            sessions = (
                db.query(TrackingSession.person_id, TrackingSession.duration_sec)
                .filter(
                    TrackingSession.entry_time >= start,
                    TrackingSession.entry_time <= end,
                )
                .all()
            )

            unique_persons = set()
            total_duration = 0
            for person_id, duration_sec in sessions:
                unique_persons.add(person_id)
                total_duration += duration_sec or 0

            avg_duration = total_duration / len(sessions) if sessions else 0

            return {
                "report_type": report_type,
                "period_start": start.isoformat(),
                "period_end": end.isoformat(),
                "total_entries": total_entries,
                "total_exits": total_exits,
                "unique_persons": len(unique_persons),
                "total_sessions": len(sessions),
                "avg_duration_sec": round(avg_duration),
            }
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {
                "report_type": report_type,
                "period_start": start.isoformat(),
                "period_end": end.isoformat(),
                "total_entries": 0,
                "total_exits": 0,
                "unique_persons": 0,
                "total_sessions": 0,
                "avg_duration_sec": 0,
            }
        finally:
            db.close()
