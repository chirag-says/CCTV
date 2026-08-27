"""
Daily Campus Summary — one day of campus activity, in the numbers an
administrator actually asks for.

Everything here is composed from data the system already records. No new
capture, no pipeline changes, no LLM.

TWO METRICS DESERVE EXPLANATION

1. Peak occupancy is ESTIMATED, and labelled as such.
   Occupancy is only ever a live in-memory count; it is never sampled or
   stored, so a past day's peak has to be reconstructed. Naively replaying
   entry/exit as +1/-1 does not work, because entry/exit are PER CAMERA:
   one person walking past four cameras would count as four people. Instead
   each person's events are merged into presence intervals and a sweep line
   finds the maximum number of DISTINCT people present at once.

2. Camera availability comes from stream health events.
   stream_offline / stream_recovered are already persisted as security_alert
   subtypes, and stream_recovered carries metadata.downtime_sec. Availability
   is therefore measured downtime over camera-time-elapsed. If no camera ran
   at all, this returns None rather than 100% -- claiming perfect uptime for
   a system that was switched off would be worse than saying nothing.

THE NARRATIVE IS TEMPLATED, NOT GENERATED
A handful of ranked rules produce the summary lines. This is deterministic,
instant, free, testable, and needs no model to be running. A 7B VLM for three
sentences would be an unforced dependency on the demo path -- vlm_service.py
already exists if that trade is ever worth making.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Camera, DetectionEvent, Person, UnknownFace
from app.db.session import SessionLocal
from app.services.analytics_service import AnalyticsService
from app.services.attendance_service import (
    ATTENDANCE_EVENT_TYPES,
    AttendanceService,
    _confidence_filter,
)
from app.utils.time_utils import (
    campus_day_bounds,
    campus_now,
    parse_campus_date,
    to_campus_local,
    utc_now,
)

logger = logging.getLogger(__name__)

# Stream-health alerts are camera plumbing, not security incidents. Counting a
# flaky RTSP connection as an "incident" would make the security figure
# meaningless.
STREAM_HEALTH_SUBTYPES = ("stream_offline", "stream_recovered")

# Consecutive detections of one person closer together than this are treated
# as one continuous presence when reconstructing occupancy.
OCCUPANCY_MERGE_GAP_SEC = 600

# How long a single detection is taken to represent.
#
# A detection is an instant, not a span, so a person seen exactly once yields a
# zero-length interval — which contributes nothing to a sweep line and would
# make three people each seen once report a peak of zero. Being detected means
# being present for at least a short while, and this is that "while". Kept
# small so it cannot manufacture large overlaps.
OCCUPANCY_MIN_PRESENCE_SEC = 120

# Days of history used as the baseline for "above/below average" narrative.
BASELINE_DAYS = 7


def _person_key(event_person_id: Optional[str], metadata: Optional[dict]) -> Optional[str]:
    """
    A stable identity for counting distinct people.

    Unknown people never have a persons row, so the FK guard in main.py moves
    their stable id into metadata_json.unknown_person_id. Counting only
    person_id would silently drop every stranger from the day's totals.
    `anon:` track handles are excluded: they are ephemeral ByteTrack ids for
    faces too small to embed, and would inflate the count with duplicates.
    """
    if event_person_id:
        return f"known:{event_person_id}"
    meta = metadata or {}
    if meta.get("anonymous"):
        return None
    unknown_id = meta.get("unknown_person_id")
    return f"unknown:{unknown_id}" if unknown_id else None


class CampusSummaryService:
    """Read-only daily summary. Composition over new capture."""

    @staticmethod
    def get_daily(date: Optional[str] = None) -> Dict[str, Any]:
        """
        One campus-local day, summarised.

        Raises:
            ValueError: malformed date, or a date in the future.
        """
        day = parse_campus_date(date)
        today = campus_now().date()
        if day > today:
            raise ValueError(f"Cannot summarise a future date ({day})")

        is_today = day == today
        # Full calendar day: activity outside the attendance window is still
        # campus activity and belongs in the day's totals.
        start_utc, end_utc = campus_day_bounds(
            day.isoformat(), day_start="00:00", day_end="24:00"
        )

        db: Session = SessionLocal()
        try:
            events = (
                db.query(
                    DetectionEvent.person_id,
                    DetectionEvent.event_type,
                    DetectionEvent.subtype,
                    DetectionEvent.camera_id,
                    DetectionEvent.created_at,
                    DetectionEvent.metadata_json,
                )
                .filter(
                    DetectionEvent.created_at >= start_utc,
                    DetectionEvent.created_at < end_utc,
                )
                .order_by(DetectionEvent.created_at.asc())
                .all()
            )

            people = CampusSummaryService._people_metrics(db, events, start_utc, end_utc)
            attendance = AttendanceService.get_summary(day.isoformat())
            occupancy = CampusSummaryService._peak_occupancy(events)
            security = CampusSummaryService._security_metrics(events)
            vehicles = sum(1 for e in events if e.event_type == "vehicle_entry")
            unknowns = CampusSummaryService._unknown_metrics(db, start_utc, end_utc)
            cameras = CampusSummaryService._camera_availability(
                db, events, start_utc, end_utc, is_today
            )
            arrival = AnalyticsService.get_hourly_distribution(
                "entry", start_utc=start_utc, end_utc=end_utc
            )
            departure = AnalyticsService.get_hourly_distribution(
                "exit", start_utc=start_utc, end_utc=end_utc
            )

            on_campus = None
            if is_today:
                # Only meaningful for today -- "currently on campus" has no
                # answer for a day that already ended.
                try:
                    from app.services.presence_service import PresenceService
                    on_campus = PresenceService.get_on_campus()["present_now"]
                except Exception as e:
                    logger.debug(f"Presence unavailable for summary: {e}")

            metrics = {
                "people_detected": people["total"],
                "known_detected": people["known"],
                "unknown_detected": people["unknown"],
                "students_detected": people["students"],
                "faculty_detected": people["faculty"],
                "faculty_present": attendance["totals"]["attending"],
                "faculty_total": attendance["totals"]["total"],
                "faculty_pct": attendance["totals"]["pct"],
                "on_campus_now": on_campus,
                "peak_occupancy": occupancy["peak"],
                "peak_occupancy_at": occupancy["at"],
                "security_incidents": security["total"],
                "incidents_by_subtype": security["by_subtype"],
                # Strangers who actually registered a presence, so this is a
                # SUBSET of people_detected and the two can be shown side by
                # side without contradicting each other.
                "unknown_persons": people["unknown"],
                # Faces captured but never confirmed as a presence: a face can
                # be snapshotted without the track surviving the 3s entry
                # threshold. Always >= unknown_persons, and reported
                # separately rather than as the headline figure — presenting
                # it as "unknown persons" made that number exceed the total
                # number of people detected.
                "unknown_faces_captured": unknowns["new_today"],
                "vehicles": vehicles,
                "camera_availability_pct": cameras["availability_pct"],
                "cameras_active": cameras["active"],
                "camera_downtime_sec": cameras["downtime_sec"],
                "peak_arrival_hour": arrival["peak_hour"],
                "peak_departure_hour": departure["peak_hour"],
                "total_events": len(events),
            }

            no_data = len(events) == 0
            baseline = (
                {} if no_data
                else CampusSummaryService._baseline(db, day, start_utc)
            )

            return {
                "date": day.isoformat(),
                "is_today": is_today,
                "no_data": no_data,
                "metrics": metrics,
                "hourly": {
                    "arrivals": arrival["hourly"],
                    "departures": departure["hourly"],
                },
                "attendance_by_department": attendance["departments"],
                "narrative": (
                    [] if no_data
                    else CampusSummaryService._narrative(metrics, baseline, attendance)
                ),
                "baseline": baseline,
                "generated_at": utc_now().isoformat(),
            }
        finally:
            db.close()

    # ── People ───────────────────────────────────────────────────

    @staticmethod
    def _people_metrics(
        db: Session, events: List[Any], start_utc: datetime, end_utc: datetime
    ) -> Dict[str, int]:
        """Distinct people seen, split by what we know about them."""
        keys = set()
        known_ids = set()
        for e in events:
            if e.event_type not in ATTENDANCE_EVENT_TYPES:
                continue
            key = _person_key(e.person_id, e.metadata_json)
            if key:
                keys.add(key)
                if e.person_id:
                    known_ids.add(e.person_id)

        by_type: Dict[str, int] = defaultdict(int)
        if known_ids:
            rows = (
                db.query(Person.person_type, func.count(Person.id))
                .filter(Person.id.in_(known_ids))
                .group_by(Person.person_type)
                .all()
            )
            by_type = {pt or "visitor": c for pt, c in rows}

        return {
            "total": len(keys),
            "known": len(known_ids),
            "unknown": len(keys) - len(known_ids),
            "faculty": by_type.get(settings.FACULTY_PERSON_TYPE, 0),
            "students": by_type.get("student", 0),
        }

    # ── Occupancy ────────────────────────────────────────────────

    @staticmethod
    def _peak_occupancy(events: List[Any]) -> Dict[str, Any]:
        """
        Reconstruct the day's maximum simultaneous occupancy.

        Per person, consecutive detections closer than OCCUPANCY_MERGE_GAP_SEC
        form one presence interval; a longer gap starts a new one. A sweep
        line over all intervals then finds the largest number of DISTINCT
        people present at the same moment.

        This is why it is not a naive +1/-1 replay of entry/exit: those are
        per-camera, so one person crossing four cameras would read as four
        people. It remains an estimate — someone out of camera view mid-visit
        splits into two intervals — hence the "estimated" label in the UI.
        """
        per_person: Dict[str, List[datetime]] = defaultdict(list)
        for e in events:
            if e.event_type not in ATTENDANCE_EVENT_TYPES or e.created_at is None:
                continue
            key = _person_key(e.person_id, e.metadata_json)
            if key:
                per_person[key].append(e.created_at)

        # Build merged intervals per person, each given a minimum duration so
        # a lone detection still represents a presence.
        min_span = timedelta(seconds=OCCUPANCY_MIN_PRESENCE_SEC)
        boundaries: List[Tuple[datetime, int]] = []

        def emit(start: datetime, end: datetime):
            boundaries.append((start, +1))
            boundaries.append((max(end, start + min_span), -1))

        for times in per_person.values():
            times.sort()
            start = prev = times[0]
            for t in times[1:]:
                if (t - prev).total_seconds() > OCCUPANCY_MERGE_GAP_SEC:
                    emit(start, prev)
                    start = t
                prev = t
            emit(start, prev)

        if not boundaries:
            return {"peak": 0, "at": None}

        # Entries before exits at the same instant: two people detected at the
        # same moment were both present at that moment, so the overlap is real
        # and must not be cancelled out by ordering.
        boundaries.sort(key=lambda b: (b[0], -b[1]))

        current = peak = 0
        peak_at: Optional[datetime] = None
        for ts, delta in boundaries:
            current += delta
            if current > peak:
                peak = current
                peak_at = ts

        return {"peak": peak, "at": peak_at.isoformat() if peak_at else None}

    # ── Security ─────────────────────────────────────────────────

    @staticmethod
    def _security_metrics(events: List[Any]) -> Dict[str, Any]:
        """
        Security alerts, excluding camera plumbing.

        stream_offline / stream_recovered are health events. Counting them as
        incidents would mean a flaky network connection reads as a security
        problem.
        """
        by_subtype: Dict[str, int] = defaultdict(int)
        for e in events:
            if e.event_type != "security_alert":
                continue
            if e.subtype in STREAM_HEALTH_SUBTYPES:
                continue
            by_subtype[e.subtype or "other"] += 1
        return {"total": sum(by_subtype.values()), "by_subtype": dict(by_subtype)}

    # ── Unknowns ─────────────────────────────────────────────────

    @staticmethod
    def _unknown_metrics(
        db: Session, start_utc: datetime, end_utc: datetime
    ) -> Dict[str, int]:
        """
        Strangers first seen on this day.

        Counted from unknown_faces.first_seen rather than from events, because
        the registry deduplicates a person across cameras and restarts — the
        row count is the number of distinct strangers, which is the question
        being asked.
        """
        new_today = (
            db.query(func.count(UnknownFace.id))
            .filter(
                UnknownFace.first_seen >= start_utc,
                UnknownFace.first_seen < end_utc,
            )
            .scalar() or 0
        )
        return {"new_today": int(new_today)}

    # ── Cameras ──────────────────────────────────────────────────

    @staticmethod
    def _camera_availability(
        db: Session,
        events: List[Any],
        start_utc: datetime,
        end_utc: datetime,
        is_today: bool,
    ) -> Dict[str, Any]:
        """
        Availability from persisted stream-health events.

        `stream_recovered` carries metadata.downtime_sec, so measured downtime
        is the sum of those. Cameras that never produced an event were not
        running, and are excluded — averaging in a switched-off camera would
        report an outage that nobody experienced.

        Returns availability_pct=None when nothing ran, so the UI can show a
        dash instead of claiming 100%.
        """
        active_cameras = {e.camera_id for e in events if e.camera_id}
        if not active_cameras:
            return {"availability_pct": None, "active": 0, "downtime_sec": 0}

        downtime = 0.0
        for e in events:
            if e.event_type == "security_alert" and e.subtype == "stream_recovered":
                meta = e.metadata_json or {}
                inner = meta.get("metadata") or meta
                downtime += float(inner.get("downtime_sec") or 0)

        window_end = min(utc_now(), end_utc) if is_today else end_utc
        elapsed = max(1.0, (window_end - start_utc).total_seconds())
        capacity = elapsed * len(active_cameras)
        pct = max(0.0, min(100.0, (1.0 - downtime / capacity) * 100.0))

        return {
            "availability_pct": round(pct, 1),
            "active": len(active_cameras),
            "downtime_sec": int(downtime),
        }

    # ── Baseline ─────────────────────────────────────────────────

    @staticmethod
    def _baseline(db: Session, day, start_utc: datetime) -> Dict[str, Any]:
        """
        Recent averages, so the narrative can say "below average" with a
        number behind it rather than as a figure of speech.

        Only days that actually recorded activity are averaged — including
        holidays and days the system was off would drag every average toward
        zero and make every working day look exceptional.
        """
        avg_attendance: List[float] = []
        avg_unknowns: List[int] = []

        for i in range(1, BASELINE_DAYS + 1):
            prev = day - timedelta(days=i)
            p_start, p_end = campus_day_bounds(
                prev.isoformat(), day_start="00:00", day_end="24:00"
            )
            had_activity = (
                db.query(DetectionEvent.id)
                .filter(
                    DetectionEvent.event_type.in_(ATTENDANCE_EVENT_TYPES),
                    DetectionEvent.created_at >= p_start,
                    DetectionEvent.created_at < p_end,
                )
                .first()
            )
            if not had_activity:
                continue
            try:
                avg_attendance.append(
                    AttendanceService.get_summary(prev.isoformat())["totals"]["pct"]
                )
            except Exception:
                pass
            avg_unknowns.append(
                db.query(func.count(UnknownFace.id))
                .filter(
                    UnknownFace.first_seen >= p_start,
                    UnknownFace.first_seen < p_end,
                )
                .scalar() or 0
            )

        return {
            "days_counted": len(avg_attendance),
            "avg_attendance_pct": (
                round(sum(avg_attendance) / len(avg_attendance), 1)
                if avg_attendance else None
            ),
            "avg_unknowns": (
                round(sum(avg_unknowns) / len(avg_unknowns), 1)
                if avg_unknowns else None
            ),
        }

    # ── Narrative ────────────────────────────────────────────────

    @staticmethod
    def _narrative(
        metrics: Dict[str, Any],
        baseline: Dict[str, Any],
        attendance: Dict[str, Any],
    ) -> List[str]:
        """
        Up to three plain-language lines about the day.

        Rules are evaluated in priority order and the first three that fire
        are kept, so the most notable thing leads. Every line is derived from
        a number already on the page — the prose never asserts anything the
        metrics do not support.
        """
        lines: List[str] = []
        m = metrics

        def hour_label(h: Optional[int]) -> str:
            return f"{h:02d}:00" if h is not None else "an unclear time"

        # 1. Attendance against the recent baseline.
        avg = baseline.get("avg_attendance_pct")
        if avg is not None and m["faculty_total"]:
            delta = m["faculty_pct"] - avg
            if delta <= -10:
                lines.append(
                    f"Faculty attendance is {abs(delta):.0f} points below the "
                    f"{baseline['days_counted']}-day average of {avg:.0f}%."
                )
            elif delta >= 10:
                lines.append(
                    f"Faculty attendance is {delta:.0f} points above the "
                    f"{baseline['days_counted']}-day average of {avg:.0f}%."
                )

        # 2. Departments that stand out.
        weak = [d for d in attendance.get("departments", []) if d["total"] >= 3 and d["pct"] < 60]
        if weak:
            worst = min(weak, key=lambda d: d["pct"])
            lines.append(
                f"{worst['name']} is the lowest department at {worst['pct']:.0f}% "
                f"({worst['attending']} of {worst['total']})."
            )

        # 3. Security.
        if m["security_incidents"] > 0:
            kinds = ", ".join(
                f"{n} {k.replace('_', ' ')}"
                for k, n in sorted(
                    m["incidents_by_subtype"].items(), key=lambda kv: -kv[1]
                )[:2]
            )
            lines.append(
                f"{m['security_incidents']} security "
                f"incident{'s' if m['security_incidents'] != 1 else ''} logged"
                f"{f' ({kinds})' if kinds else ''}."
            )

        # 4. Unknown-person spike. Compared against the same measure used for
        # the baseline (faces captured), not the headline subset.
        avg_unk = baseline.get("avg_unknowns")
        captured = m["unknown_faces_captured"]
        if avg_unk and avg_unk >= 1 and captured > avg_unk * 1.5:
            lines.append(
                f"Unfamiliar faces are {captured / avg_unk:.1f}x the recent average "
                f"({captured} against {avg_unk:.0f})."
            )

        # 5. Camera health.
        pct = m["camera_availability_pct"]
        if pct is not None and pct < 98:
            mins = m["camera_downtime_sec"] // 60
            lines.append(
                f"Camera availability was {pct:.1f}% — about {mins} minute"
                f"{'s' if mins != 1 else ''} of stream downtime."
            )

        # 6. Fallback so the card is never empty on a day that had activity.
        if m["peak_occupancy"]:
            at_local = (
                to_campus_local(datetime.fromisoformat(m["peak_occupancy_at"]))
                if m["peak_occupancy_at"] else None
            )
            when = at_local.strftime("%H:%M") if at_local else "an unclear time"
            lines.append(
                f"Peak occupancy of about {m['peak_occupancy']} was reached around "
                f"{when}, with arrivals busiest at {hour_label(m['peak_arrival_hour'])}."
            )

        return lines[:3]
