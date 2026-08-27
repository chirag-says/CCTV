"""
Faculty Attendance — derived read model over detection_events.

WHY THIS IS DERIVED AND NOT RECORDED
The pipeline already writes an `entry` row every time a known person becomes
visible on a camera, and an `exit` row when they leave that camera's view.
Those rows are committed, immutable, and survive a restart. Attendance is a
question you ask of them, not a separate thing to record -- which is why
nothing here touches the vision pipeline and why a backend restart cannot
corrupt an attendance figure.

WHY `tracking_sessions` IS NOT USED
A session is opened on entry and closed only when an exit event carrying its
session_id arrives. pipeline.stop() flushes exits only for tracks already
absent >= EXIT_THRESHOLD_SECONDS, so anyone visible at shutdown never gets
one, and a crash closes nothing at all. The table therefore accumulates rows
stuck in status='active' forever. detection_events are point-in-time facts
and have no such failure mode.

THE RULE
A single detection must not equal attendance: ENTRY_THRESHOLD_SECONDS is 3,
so three seconds of visibility creates an entry row, and a recogniser false
accept produces a real entry row for the wrong person. Two independent
confirmations are required:

    present = sightings >= ATTENDANCE_MIN_SIGHTINGS
              AND (span >= ATTENDANCE_MIN_SPAN_MIN
                   OR distinct_sources >= ATTENDANCE_MIN_CAMERAS)

The AND is load-bearing. Two sightings alone are not enough -- they must be
separated in time or corroborated by a second camera -- so two false accepts
a minute apart on one camera correctly yield 'absent'.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlparse

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.core.exceptions import NotFoundException
from app.db.models import Camera, DetectionEvent, FaceEncoding, Person
from app.db.session import SessionLocal
from app.utils.time_utils import (
    campus_day_bounds,
    campus_now,
    is_after_campus_time,
    parse_campus_date,
    to_campus_local,
)

logger = logging.getLogger(__name__)

# Attendance is decided from presence evidence only. `entry` marks an arrival
# at a camera; `exit` is included so that a visit whose entry was missed still
# contributes to the observed span.
ATTENDANCE_EVENT_TYPES = ("entry", "exit")

UNASSIGNED_DEPARTMENT = "Unassigned"

STATUS_PRESENT = "present"
STATUS_LATE = "late"
STATUS_ABSENT = "absent"


# ── Camera source de-duplication ─────────────────────────────────

# Query parameters that select a QUALITY VARIANT of a stream rather than a
# different camera. Two URLs differing only in these point at the same lens.
# (Dahua/generic use subtype=0|1; others use stream/profile/quality.)
_STREAM_QUALITY_PARAMS = {
    "subtype", "stream", "quality", "resolution", "profile", "streamtype",
}

# Hikvision encodes camera and stream in one path segment:
#   /Streaming/Channels/101  → camera 1, main stream
#   /Streaming/Channels/102  → camera 1, sub stream
# The trailing digit is the stream index, so it must be dropped when deciding
# whether two URLs are the same physical camera.
_HIK_CHANNEL_RE = re.compile(r"(/streaming/channels/)(\d+)", re.IGNORECASE)


def normalise_stream_source(stream_url: Optional[str], fallback: str) -> str:
    """
    Reduce a stream URL to a key identifying the PHYSICAL CAMERA behind it.

    Cameras are routinely registered more than once against the same lens --
    this deployment has MainGate and MainStream both on
    `172.20.0.245/video/live?channel=1`, differing only by `subtype`
    (main stream vs sub stream). If both workers run, one person walking past
    produces entry rows under two different camera_ids, which would satisfy
    the "seen on 2+ cameras" branch of the attendance rule from a single
    appearance and defeat the entire two-confirmation design.

    Normalisation:
      * credentials dropped  — the same camera may be registered with
        different logins
      * quality-selector query params dropped (see _STREAM_QUALITY_PARAMS)
      * Hikvision channel paths reduced to their camera number
      * remaining query params sorted, so ordering is not significant

    Anything unparseable (a bare webcam index like "0", an empty value) falls
    back to the camera's own id, so genuinely distinct cameras are never
    merged by accident. This errs toward treating cameras as DISTINCT, which
    is the safe direction: over-merging would make attendance harder to earn,
    but under-merging is what lets a single appearance count twice.
    """
    raw = (stream_url or "").strip()
    if not raw or raw.isdigit() or raw.startswith("browser:"):
        return fallback

    try:
        parsed = urlparse(raw)
        if not parsed.hostname:
            return fallback

        path = _HIK_CHANNEL_RE.sub(
            lambda m: f"{m.group(1)}{m.group(2)[:-2] or m.group(2)}", parsed.path
        )

        params = sorted(
            (k.lower(), v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _STREAM_QUALITY_PARAMS
        )

        host = parsed.hostname.lower()
        port = parsed.port or ""
        query = "&".join(f"{k}={v}" for k, v in params)
        return f"{parsed.scheme.lower()}://{host}:{port}{path.rstrip('/')}?{query}"
    except (ValueError, AttributeError):
        return fallback


def _confidence_filter():
    """
    Drop events whose RECORDED recognition confidence is below the threshold.

    Critically, `confidence <= 0` means "not recorded", not "low confidence",
    and unrecorded evidence must not be thrown away. In the current pipeline:

      * `exit` events carry no confidence at all -- tracker.check_exits()
        builds the event dict without the key, so every exit persists as 0.0.
      * `entry` events only carry one when the confirming frame came from the
        recognition path. The ByteTrack cache-hit path and the too-small-face
        path both pass 0.0, so most entries land at 0.0 too.

    Measured on this deployment: 6,238 of 6,238 exits and 6,361 of 6,400
    entries are 0.0. A naive `confidence >= 0.55` filter therefore discards
    ~99% of genuine presence evidence and marks the entire faculty absent.

    So the gate only rejects events that positively reported a weak match.
    That makes it a safety net against known-bad matches rather than the
    primary defence -- the two-confirmation rule in _decide() is what
    actually carries the decision.
    """
    return (
        (DetectionEvent.confidence >= settings.ATTENDANCE_MIN_CONFIDENCE)
        | (DetectionEvent.confidence <= 0)
        | (DetectionEvent.confidence.is_(None))
    )


def _camera_source_map(db: Session) -> Dict[str, str]:
    """Map camera_id → physical-source key. See normalise_stream_source."""
    return {
        cam_id: normalise_stream_source(stream_url, cam_id)
        for cam_id, stream_url in db.query(Camera.id, Camera.stream_url).all()
    }


# ── Core aggregation ─────────────────────────────────────────────

def _aggregate_events(
    db: Session,
    person_ids: List[str],
    start_utc: datetime,
    end_utc: datetime,
) -> Dict[str, Dict[str, Any]]:
    """
    Fold one campus-day of events into per-person presence evidence.

    One indexed query for the whole roster (ix_events_person_created), then
    Python folding. Camera ids come back as a list rather than a COUNT so
    they can be collapsed to distinct physical sources afterwards.
    """
    if not person_ids:
        return {}

    rows = (
        db.query(
            DetectionEvent.person_id,
            DetectionEvent.event_type,
            DetectionEvent.camera_id,
            DetectionEvent.created_at,
        )
        .filter(
            DetectionEvent.person_id.in_(person_ids),
            DetectionEvent.event_type.in_(ATTENDANCE_EVENT_TYPES),
            _confidence_filter(),
            DetectionEvent.created_at >= start_utc,
            DetectionEvent.created_at < end_utc,
        )
        .all()
    )

    sources = _camera_source_map(db)
    agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "first_seen": None,
            "last_seen": None,
            "sightings": 0,
            "camera_ids": set(),
            "sources": set(),
            "last_camera_id": None,
        }
    )

    for person_id, event_type, camera_id, created_at in rows:
        if created_at is None:
            continue
        a = agg[person_id]
        if a["first_seen"] is None or created_at < a["first_seen"]:
            a["first_seen"] = created_at
        if a["last_seen"] is None or created_at > a["last_seen"]:
            a["last_seen"] = created_at
            a["last_camera_id"] = camera_id
        if event_type == "entry":
            a["sightings"] += 1
        if camera_id:
            a["camera_ids"].add(camera_id)
            a["sources"].add(sources.get(camera_id, camera_id))

    return agg


def _decide(evidence: Optional[Dict[str, Any]]) -> Tuple[str, bool]:
    """
    Apply the attendance rule.

    Returns (status, confirmed). `confirmed` is the raw presence decision;
    `status` adds the late/absent distinction on top of it.
    """
    if not evidence or evidence["first_seen"] is None:
        return STATUS_ABSENT, False

    span_min = (
        evidence["last_seen"] - evidence["first_seen"]
    ).total_seconds() / 60.0

    confirmed = (
        evidence["sightings"] >= settings.ATTENDANCE_MIN_SIGHTINGS
        and (
            span_min >= settings.ATTENDANCE_MIN_SPAN_MIN
            or len(evidence["sources"]) >= settings.ATTENDANCE_MIN_CAMERAS
        )
    )
    if not confirmed:
        return STATUS_ABSENT, False

    if is_after_campus_time(evidence["first_seen"], settings.ATTENDANCE_LATE_AFTER):
        return STATUS_LATE, True
    return STATUS_PRESENT, True


def _normalise_department(dept: Optional[str]) -> str:
    d = (dept or "").strip()
    return d if d else UNASSIGNED_DEPARTMENT


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """Serialise UTC, as every other endpoint does. The UI localises."""
    return dt.isoformat() if dt else None


def _local_hhmm(dt: Optional[datetime]) -> Optional[str]:
    """Campus-local HH:MM, for exports where a human reads the raw cell."""
    local = to_campus_local(dt)
    return local.strftime("%H:%M") if local else None


# Designations whose title-case form is wrong ("hod" → "Hod").
_ROLE_ACRONYMS = {"hod": "HOD"}


def format_designation(role: Optional[str]) -> str:
    """Human-readable designation for reports: 'assistant_professor' → 'Assistant Professor'."""
    r = (role or "").strip().lower()
    if not r:
        return ""
    return _ROLE_ACRONYMS.get(r) or r.replace("_", " ").title()


class AttendanceService:
    """Read-only attendance derivation. No writes, no pipeline coupling."""

    # ── Roster ───────────────────────────────────────────────────

    @staticmethod
    def _faculty_roster(
        db: Session, department: Optional[str] = None
    ) -> List[Person]:
        q = db.query(Person).filter(
            Person.person_type == settings.FACULTY_PERSON_TYPE,
            Person.is_active == True,  # noqa: E712 — SQLAlchemy needs ==
        )
        if department and department != UNASSIGNED_DEPARTMENT:
            q = q.filter(
                func.upper(func.trim(Person.department)) == department.strip().upper()
            )
        elif department == UNASSIGNED_DEPARTMENT:
            q = q.filter(
                (Person.department.is_(None)) | (func.trim(Person.department) == "")
            )
        return q.order_by(Person.name.asc()).all()

    @staticmethod
    def _encoding_counts(db: Session, person_ids: List[str]) -> Dict[str, int]:
        """
        Faces enrolled per person.

        A faculty member with zero encodings can never be recognised, so they
        are absent every single day. Surfacing that as "No face enrolled"
        rather than a bare cross is the difference between an explainable
        gap and an accusation.
        """
        if not person_ids:
            return {}
        rows = (
            db.query(FaceEncoding.person_id, func.count(FaceEncoding.id))
            .filter(FaceEncoding.person_id.in_(person_ids))
            .group_by(FaceEncoding.person_id)
            .all()
        )
        return {pid: count for pid, count in rows}

    # ── Per-person records for a day ─────────────────────────────

    @staticmethod
    def get_day(
        date: Optional[str] = None,
        department: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Attendance records for every faculty member on one campus-local day.

        Args:
            date:       'YYYY-MM-DD' campus-local. None → campus today.
            department: case-insensitive filter, or 'Unassigned'.
            status:     present/late/absent filter.

        Raises:
            ValueError: malformed date, or a date in the future.
        """
        day = parse_campus_date(date)
        if day > campus_now().date():
            raise ValueError(f"Cannot report attendance for a future date ({day})")

        start_utc, end_utc = campus_day_bounds(day.isoformat())
        db: Session = SessionLocal()
        try:
            roster = AttendanceService._faculty_roster(db, department)
            person_ids = [p.id for p in roster]
            evidence = _aggregate_events(db, person_ids, start_utc, end_utc)
            enc_counts = AttendanceService._encoding_counts(db, person_ids)
            cameras = {c.id: c for c in db.query(Camera).all()}

            records: List[Dict[str, Any]] = []
            for p in roster:
                ev = evidence.get(p.id)
                rec_status, _ = _decide(ev)

                first_seen = ev["first_seen"] if ev else None
                last_seen = ev["last_seen"] if ev else None
                observed_min = (
                    int((last_seen - first_seen).total_seconds() // 60)
                    if first_seen and last_seen else 0
                )
                last_cam = cameras.get(ev["last_camera_id"]) if ev else None

                records.append({
                    "person_id": p.id,
                    "name": p.name,
                    "role": p.role,
                    "department": _normalise_department(p.department),
                    "avatar_url": p.avatar_url,
                    "status": rec_status,
                    # Deliberately "first/last detected", never "arrived/
                    # departed" -- the system observes cameras, not the
                    # campus boundary.
                    "first_seen": _iso(first_seen),
                    "last_seen": _iso(last_seen),
                    "minutes_observed": observed_min,
                    # Surfaced so a shortfall against the rule is visible
                    # rather than a silent, unexplained absence.
                    "sightings": ev["sightings"] if ev else 0,
                    "camera_count": len(ev["sources"]) if ev else 0,
                    "last_camera_id": ev["last_camera_id"] if ev else None,
                    "last_camera_name": (last_cam.name if last_cam else None),
                    "has_encodings": enc_counts.get(p.id, 0) > 0,
                    "encoding_count": enc_counts.get(p.id, 0),
                })

            if status:
                records = [r for r in records if r["status"] == status]

            return {
                "date": day.isoformat(),
                "records": records,
                "total": len(records),
                "no_data": AttendanceService._is_no_data_day(db, start_utc, end_utc),
                "rule": AttendanceService.describe_rule(),
            }
        finally:
            db.close()

    # ── Department rollup ────────────────────────────────────────

    @staticmethod
    def get_summary(date: Optional[str] = None) -> Dict[str, Any]:
        """
        Department-wise rollup plus campus totals for one campus-local day.

        `no_data` is set when the campus recorded no attendance-relevant
        events at all -- a holiday, or a day the cameras were down. Reporting
        that as "0% attendance" would be a confident lie, so the UI suppresses
        the headline instead.
        """
        payload = AttendanceService.get_day(date=date)
        records = payload["records"]

        buckets: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"present": 0, "late": 0, "absent": 0, "total": 0}
        )
        for r in records:
            b = buckets[r["department"]]
            b[r["status"]] += 1
            b["total"] += 1

        departments = []
        # Named departments alphabetically; 'Unassigned' always last.
        for name in sorted(buckets, key=lambda n: (n == UNASSIGNED_DEPARTMENT, n)):
            b = buckets[name]
            # Late still counts as attending -- they showed up.
            attending = b["present"] + b["late"]
            departments.append({
                "name": name,
                "present": b["present"],
                "late": b["late"],
                "absent": b["absent"],
                "attending": attending,
                "total": b["total"],
                "pct": round(attending / b["total"] * 100, 1) if b["total"] else 0.0,
            })

        total = len(records)
        attending = sum(1 for r in records if r["status"] in (STATUS_PRESENT, STATUS_LATE))
        return {
            "date": payload["date"],
            "departments": departments,
            "totals": {
                "present": sum(1 for r in records if r["status"] == STATUS_PRESENT),
                "late": sum(1 for r in records if r["status"] == STATUS_LATE),
                "absent": sum(1 for r in records if r["status"] == STATUS_ABSENT),
                "attending": attending,
                "total": total,
                "pct": round(attending / total * 100, 1) if total else 0.0,
                "no_encodings": sum(1 for r in records if not r["has_encodings"]),
            },
            "no_data": payload["no_data"],
            "rule": payload["rule"],
        }

    # ── One person over a range ──────────────────────────────────

    @staticmethod
    def get_person_range(
        person_id: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_days: int = 92,
    ) -> Dict[str, Any]:
        """
        One faculty member's attendance across a date range (inclusive).

        Defaults to the last 30 days ending today. Capped at `max_days` so a
        wide range cannot issue an unbounded number of day queries.
        """
        from datetime import timedelta

        today = campus_now().date()
        end = parse_campus_date(date_to) if date_to else today
        start = parse_campus_date(date_from) if date_from else end - timedelta(days=29)

        if start > end:
            raise ValueError("`from` must be on or before `to`")
        if end > today:
            end = today
        span_days = (end - start).days + 1
        if span_days > max_days:
            raise ValueError(f"Range too wide ({span_days} days) — maximum is {max_days}")

        db: Session = SessionLocal()
        try:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise NotFoundException("Person", person_id)

            enc_counts = AttendanceService._encoding_counts(db, [person_id])
            cameras = {c.id: c for c in db.query(Camera).all()}

            days: List[Dict[str, Any]] = []
            cursor = start
            while cursor <= end:
                s_utc, e_utc = campus_day_bounds(cursor.isoformat())
                ev = _aggregate_events(db, [person_id], s_utc, e_utc).get(person_id)
                day_status, _ = _decide(ev)
                last_cam = cameras.get(ev["last_camera_id"]) if ev else None
                days.append({
                    "day": cursor.isoformat(),
                    "status": day_status,
                    "first_seen": _iso(ev["first_seen"]) if ev else None,
                    "last_seen": _iso(ev["last_seen"]) if ev else None,
                    "sightings": ev["sightings"] if ev else 0,
                    "camera_count": len(ev["sources"]) if ev else 0,
                    "last_camera_name": (last_cam.name if last_cam else None),
                })
                cursor += timedelta(days=1)

            counted = len(days)
            present_days = sum(1 for d in days if d["status"] == STATUS_PRESENT)
            late_days = sum(1 for d in days if d["status"] == STATUS_LATE)
            absent_days = sum(1 for d in days if d["status"] == STATUS_ABSENT)

            return {
                "person": {
                    "id": person.id,
                    "name": person.name,
                    "role": person.role,
                    "person_type": person.person_type,
                    "department": _normalise_department(person.department),
                    "avatar_url": person.avatar_url,
                    "has_encodings": enc_counts.get(person_id, 0) > 0,
                },
                "from": start.isoformat(),
                "to": end.isoformat(),
                "days": days,
                "summary": {
                    "days_counted": counted,
                    "present_days": present_days,
                    "late_days": late_days,
                    "absent_days": absent_days,
                    "pct": round((present_days + late_days) / counted * 100, 1) if counted else 0.0,
                },
            }
        finally:
            db.close()

    # ── Export ───────────────────────────────────────────────────

    EXPORT_COLUMNS = [
        "Date", "Name", "Department", "Designation", "Status",
        "First Detected", "Last Detected", "Minutes Observed",
        "Sightings", "Cameras", "Face Enrolled", "Person ID",
    ]

    @staticmethod
    def export_rows(
        date: Optional[str] = None,
        department: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Flat, human-readable rows for CSV / Excel export.

        Times are rendered campus-local, because a spreadsheet cell has no
        timezone and a security office reading "03:12" for an 08:42 arrival
        would reasonably conclude the system is broken.
        """
        payload = AttendanceService.get_day(
            date=date, department=department, status=status
        )
        rows = []
        for r in payload["records"]:
            rows.append({
                "Date": payload["date"],
                "Name": r["name"],
                "Department": r["department"],
                "Designation": format_designation(r["role"]),
                "Status": r["status"].upper(),
                "First Detected": _local_hhmm(
                    datetime.fromisoformat(r["first_seen"]) if r["first_seen"] else None
                ) or "",
                "Last Detected": _local_hhmm(
                    datetime.fromisoformat(r["last_seen"]) if r["last_seen"] else None
                ) or "",
                "Minutes Observed": r["minutes_observed"] or "",
                "Sightings": r["sightings"],
                "Cameras": r["camera_count"],
                "Face Enrolled": "Yes" if r["has_encodings"] else "No",
                "Person ID": r["person_id"],
            })
        return rows

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _is_no_data_day(db: Session, start_utc: datetime, end_utc: datetime) -> bool:
        """
        Did the campus record any attendance-relevant event at all?

        Distinguishes "everyone was absent" from "nothing was running", which
        look identical in the aggregate and mean completely different things.
        """
        return (
            db.query(DetectionEvent.id)
            .filter(
                DetectionEvent.event_type.in_(ATTENDANCE_EVENT_TYPES),
                DetectionEvent.created_at >= start_utc,
                DetectionEvent.created_at < end_utc,
            )
            .first()
        ) is None

    @staticmethod
    def describe_rule() -> Dict[str, Any]:
        """
        The active rule, echoed to the UI so an admin can see exactly why
        somebody was marked absent without reading the source.
        """
        return {
            "min_sightings": settings.ATTENDANCE_MIN_SIGHTINGS,
            "min_span_minutes": settings.ATTENDANCE_MIN_SPAN_MIN,
            "min_cameras": settings.ATTENDANCE_MIN_CAMERAS,
            "min_confidence": settings.ATTENDANCE_MIN_CONFIDENCE,
            # Surfaced so nobody assumes the confidence gate is doing more
            # work than it is -- see _confidence_filter().
            "confidence_note": (
                "Applied only to events that recorded a confidence; the "
                "pipeline leaves it unset on exits and on cached recognitions."
            ),
            "late_after": settings.ATTENDANCE_LATE_AFTER,
            "day_start": settings.ATTENDANCE_DAY_START,
            "day_end": settings.ATTENDANCE_DAY_END,
            "timezone": settings.CAMPUS_TIMEZONE,
            "summary": (
                f"Present requires {settings.ATTENDANCE_MIN_SIGHTINGS}+ sightings "
                f"spanning {settings.ATTENDANCE_MIN_SPAN_MIN}+ minutes "
                f"or across {settings.ATTENDANCE_MIN_CAMERAS}+ cameras."
            ),
        }
