"""
Faculty presence — "Teachers Currently on Campus".

WHY THIS IS NOT JUST THE LIVE TRACKER
PersonTracker only knows "visible on a camera within the last
EXIT_THRESHOLD_SECONDS (30s)". A teacher sitting in a classroom is not
visible, and all tracker state is per-pipeline and in-memory, so it is wiped
on restart. A page built on it would read near-zero all day and reset to
empty every deploy.

So presence is derived from committed detection events, with the live tracker
layered on top as a "visible right now" badge. A restart costs the badge for
a few seconds; it costs nothing else.

THE FOUR STATES

    LIVE          in the live tracker now (~30s)     → "Visible now"
    ON_CAMPUS     last detected < PRESENCE_CONFIDENT_MIN → "On campus · 4 min ago"
    SEEN_EARLIER  detected today, older, no departure → "On campus · 2h 14m ago"
    DEPARTED      gate exit, or past the campus day   → "Departed 17:21"

Only LIVE + ON_CAMPUS feed the headline count. SEEN_EARLIER is reported
separately as "+ N seen earlier today" so neither number has to be defended:
a 20-minute-old sighting is a confident claim, an 8-hour-old one is not, and
collapsing them into one figure would make the whole page deniable. The
last-detected age is always rendered, whatever the state.

WHO APPEARS AT ALL
Anyone with at least one detection today, OR anyone the cameras can see right
now. Deliberately NOT restricted to attendance-confirmed faculty: attendance
needs two confirmations spanning five minutes, so at 08:45 nobody qualifies
yet and the page would be empty every morning -- while a camera was actively
looking at them. Attendance answers "did they work today"; presence answers
"are they here now". The attendance status travels alongside each row so the
UI can show both without conflating them.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Camera, Person
from app.db.session import SessionLocal
from app.services.attendance_service import (
    AttendanceService,
    _aggregate_events,
    _decide,
    _normalise_department,
    format_designation,
)
from app.utils.time_utils import (
    campus_day_bounds,
    campus_now,
    parse_campus_date,
    utc_now,
    _parse_hhmm,
)

logger = logging.getLogger(__name__)

STATE_LIVE = "LIVE"
STATE_ON_CAMPUS = "ON_CAMPUS"
STATE_SEEN_EARLIER = "SEEN_EARLIER"
STATE_DEPARTED = "DEPARTED"

# Ordering for the UI: most certain first, then by recency within each band.
_STATE_ORDER = {
    STATE_LIVE: 0,
    STATE_ON_CAMPUS: 1,
    STATE_SEEN_EARLIER: 2,
    STATE_DEPARTED: 3,
}

# States that count toward the headline "N present now".
COUNTED_PRESENT = (STATE_LIVE, STATE_ON_CAMPUS)


def _live_tracker_state() -> Dict[str, Dict[str, Any]]:
    """
    Snapshot every person the running pipelines can currently see.

    Best-effort by design: a camera that stopped between the sighting and this
    call simply contributes nothing, which is not an error. Failure here must
    degrade the badge, never the page -- presence still works from events.

    Note the tracker is per-camera, so the same person can appear on several
    workers at once; the most recent sighting wins.
    """
    live: Dict[str, Dict[str, Any]] = {}
    try:
        from app.vision.camera_worker import camera_manager

        for camera_id in camera_manager.get_active_camera_ids():
            worker = camera_manager._workers.get(camera_id)
            if not worker or not worker.pipeline:
                continue
            for person in worker.pipeline.tracker.get_active_persons():
                pid = person.get("person_id")
                # `anon:` handles are track ids, not identities, and unknown
                # ids have no persons row. Neither is a faculty member.
                if not pid or str(pid).startswith("anon:"):
                    continue
                seen = person.get("last_seen")
                existing = live.get(pid)
                if existing is None or (seen or "") > (existing.get("last_seen") or ""):
                    live[pid] = {
                        "camera_id": person.get("camera_id") or camera_id,
                        "last_seen": seen,
                        "duration_sec": person.get("duration_sec", 0),
                    }
    except Exception as e:
        logger.debug(f"Live tracker snapshot unavailable: {e}")
    return live


def _is_past_campus_day_end() -> bool:
    """Has the campus operating day ended in local time?"""
    try:
        return campus_now().time() > _parse_hhmm(settings.ATTENDANCE_DAY_END)
    except ValueError:
        return False


class PresenceService:
    """Read-only faculty presence. No writes, no pipeline coupling."""

    @staticmethod
    def get_on_campus(
        department: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Faculty presence right now.

        Args:
            department: case-insensitive filter, or 'Unassigned'.
            query:      case-insensitive name substring.
        """
        day = parse_campus_date(None)
        start_utc, end_utc = campus_day_bounds(day.isoformat())
        now = utc_now()
        past_day_end = _is_past_campus_day_end()

        db: Session = SessionLocal()
        try:
            roster = AttendanceService._faculty_roster(db, department)
            if query:
                needle = query.strip().lower()
                roster = [p for p in roster if needle in (p.name or "").lower()]

            person_ids = [p.id for p in roster]
            evidence = _aggregate_events(db, person_ids, start_utc, end_utc)
            cameras = {c.id: c for c in db.query(Camera).all()}
            live = _live_tracker_state()

            faculty: List[Dict[str, Any]] = []
            for p in roster:
                ev = evidence.get(p.id)
                lv = live.get(p.id)

                # No detection today and not visible now → not on campus, and
                # listing them would just duplicate the attendance page.
                if not ev and not lv:
                    continue

                first_seen = ev["first_seen"] if ev else None
                last_seen = ev["last_seen"] if ev else None
                last_camera_id = ev["last_camera_id"] if ev else None

                # A live sighting is more recent than anything committed: the
                # event for it may not have been written yet (entry only fires
                # after ENTRY_THRESHOLD_SECONDS, exits after 30s of absence).
                if lv:
                    last_camera_id = lv["camera_id"] or last_camera_id

                minutes_since = (
                    (now - last_seen).total_seconds() / 60.0
                    if last_seen else None
                )

                state = PresenceService._state_for(
                    is_live=lv is not None,
                    minutes_since=minutes_since,
                    last_camera=cameras.get(last_camera_id),
                    past_day_end=past_day_end,
                )

                cam = cameras.get(last_camera_id)
                att_status, _ = _decide(ev)

                faculty.append({
                    "person_id": p.id,
                    "name": p.name,
                    "role": p.role,
                    "designation": format_designation(p.role),
                    "department": _normalise_department(p.department),
                    "avatar_url": p.avatar_url,
                    "state": state,
                    "is_live": lv is not None,
                    # "current" when live, "last known" otherwise. The UI
                    # phrases it from `state`, never asserting a location the
                    # cameras are not currently confirming.
                    "camera_id": last_camera_id,
                    "camera_name": (cam.name if cam else None),
                    "camera_location": (cam.location if cam else "") or "",
                    "zone": ((cam.config or {}).get("zone") if cam else None),
                    "first_seen": first_seen.isoformat() if first_seen else None,
                    "last_seen": last_seen.isoformat() if last_seen else None,
                    "minutes_since_seen": (
                        int(minutes_since) if minutes_since is not None else None
                    ),
                    "minutes_observed": (
                        int((last_seen - first_seen).total_seconds() // 60)
                        if first_seen and last_seen else 0
                    ),
                    "sightings": ev["sightings"] if ev else 0,
                    "camera_count": len(ev["sources"]) if ev else 0,
                    # Carried so the UI can show presence and attendance
                    # together without conflating them.
                    "attendance_status": att_status,
                })

            faculty.sort(
                key=lambda f: (
                    _STATE_ORDER.get(f["state"], 9),
                    f["minutes_since_seen"] if f["minutes_since_seen"] is not None else 10**6,
                    f["name"],
                )
            )

            present_now = sum(1 for f in faculty if f["state"] in COUNTED_PRESENT)
            seen_earlier = sum(1 for f in faculty if f["state"] == STATE_SEEN_EARLIER)
            departed = sum(1 for f in faculty if f["state"] == STATE_DEPARTED)

            return {
                "date": day.isoformat(),
                "present_now": present_now,
                "seen_earlier": seen_earlier,
                "departed": departed,
                "faculty": faculty,
                "roster_total": len(roster),
                "live_tracking_active": bool(live),
                "thresholds": {
                    "confident_minutes": settings.PRESENCE_CONFIDENT_MIN,
                    "gate_depart_minutes": settings.GATE_DEPART_MIN,
                    "day_end": settings.ATTENDANCE_DAY_END,
                },
                "generated_at": now.isoformat(),
            }
        finally:
            db.close()

    @staticmethod
    def _state_for(
        is_live: bool,
        minutes_since: Optional[float],
        last_camera: Optional[Camera],
        past_day_end: bool,
    ) -> str:
        """
        Decide one person's presence state.

        Order matters. A live sighting beats every other signal including the
        day-end rule -- if a camera can see them at 21:30, they are on campus,
        whatever the timetable says.
        """
        if is_live:
            return STATE_LIVE

        if minutes_since is None:
            return STATE_DEPARTED

        # A gate camera is the only place this system can honestly infer a
        # departure: it is the one location where being seen and then not seen
        # means leaving rather than walking into an unmonitored room.
        if last_camera is not None and (last_camera.config or {}).get("is_gate"):
            if minutes_since >= settings.GATE_DEPART_MIN:
                return STATE_DEPARTED

        if past_day_end:
            return STATE_DEPARTED

        if minutes_since < settings.PRESENCE_CONFIDENT_MIN:
            return STATE_ON_CAMPUS

        return STATE_SEEN_EARLIER

    @staticmethod
    def get_person_presence(person_id: str) -> Optional[Dict[str, Any]]:
        """
        Presence for one person, for the "Where is someone?" search card.

        Returns None when the person is not faculty or has no presence today,
        so the caller can fall back to a plain person result.
        """
        result = PresenceService.get_on_campus()
        for f in result["faculty"]:
            if f["person_id"] == person_id:
                return f
        return None
