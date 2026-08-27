"""
Person Journey — where someone was seen today, in order.

The pipeline already records this. Every time a known person becomes visible
on a camera for ENTRY_THRESHOLD_SECONDS an `entry` row is written, and when
they leave that camera's view for EXIT_THRESHOLD_SECONDS an `exit` row is
written. A walk from the gate through the academic block to the CSE block is
therefore already in `detection_events` -- this service only has to read it
back in a shape a human recognises.

WHY RAW EVENTS ARE NOT THE ANSWER
Someone lingering at one camera produces entry → exit → cooldown → entry →
exit, over and over. Rendering rows directly would show "Main Gate" five
times in a row and look broken. Consecutive events at the same place collapse
into one STOP; a gap longer than JOURNEY_STOP_GAP_SEC splits them, because
that is a genuine revisit rather than a flicker.

WHY STOPS ARE KEYED ON PHYSICAL CAMERA, NOT camera_id
The same lens can be registered more than once (this deployment has MainGate
and MainStream, both channel=1 of one NVR at different stream qualities).
Keyed on camera_id, one person standing still would produce a fake
Gate → Stream → Gate → Stream ping-pong. normalise_stream_source() collapses
those onto one key, so a stop is a PLACE rather than a database row.

LANGUAGE
This is a "journey", never "tracking". Times are "first detected" and "last
detected", never "arrived" and "departed" -- the system observes cameras, not
the campus boundary (SPEC §1.3).
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.core.exceptions import NotFoundException
from app.db.models import Camera, DetectionEvent, Person
from app.db.session import SessionLocal
from app.services.attendance_service import (
    ATTENDANCE_EVENT_TYPES,
    _confidence_filter,
    _normalise_department,
    format_designation,
    normalise_stream_source,
)
from app.utils.time_utils import campus_day_bounds, campus_now, parse_campus_date

logger = logging.getLogger(__name__)


def _pick_display_camera(cameras: List[Camera]) -> Optional[Camera]:
    """
    Choose which registration to name a stop after.

    When one physical camera is registered several times the names differ
    ("MainGate" vs "MainStream") even though the place is identical. Prefer a
    registration that has been given a `zone`, then one with a location, then
    whichever came first -- so an operator who labelled one of them gets their
    label, and the choice is stable rather than arbitrary.
    """
    if not cameras:
        return None
    for cam in cameras:
        if (cam.config or {}).get("zone"):
            return cam
    for cam in cameras:
        if (cam.location or "").strip():
            return cam
    return cameras[0]


class JourneyService:
    """Read-only journey derivation. No writes, no pipeline coupling."""

    @staticmethod
    def get_journey(person_id: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        One person's journey for one campus-local day.

        Args:
            person_id: persons.id
            date:      'YYYY-MM-DD' campus-local. None → campus today.

        Raises:
            NotFoundException: unknown person
            ValueError:        malformed or future date
        """
        day = parse_campus_date(date)
        if day > campus_now().date():
            raise ValueError(f"Cannot show a journey for a future date ({day})")

        # A journey shows the whole calendar day, not the attendance operating
        # window: an out-of-hours detection is exactly the kind of thing
        # somebody opens this page to find.
        start_utc, end_utc = campus_day_bounds(
            day.isoformat(), day_start="00:00", day_end="24:00"
        )

        db: Session = SessionLocal()
        try:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise NotFoundException("Person", person_id)

            events = (
                db.query(DetectionEvent)
                .filter(
                    DetectionEvent.person_id == person_id,
                    DetectionEvent.event_type.in_(ATTENDANCE_EVENT_TYPES),
                    _confidence_filter(),
                    DetectionEvent.created_at >= start_utc,
                    DetectionEvent.created_at < end_utc,
                )
                .order_by(DetectionEvent.created_at.asc())
                .all()
            )

            cameras = {c.id: c for c in db.query(Camera).all()}
            stops = JourneyService._collapse_to_stops(events, cameras)
            transitions = JourneyService._transitions(stops)

            first_seen = stops[0]["arrived_at"] if stops else None
            last_seen = stops[-1]["left_at"] if stops else None
            observed_sec = (
                int((events[-1].created_at - events[0].created_at).total_seconds())
                if len(events) >= 2 else 0
            )

            return {
                "person": {
                    "id": person.id,
                    "name": person.name,
                    "role": person.role,
                    "designation": format_designation(person.role),
                    "person_type": person.person_type,
                    "department": _normalise_department(person.department),
                    "avatar_url": person.avatar_url,
                },
                "date": day.isoformat(),
                "first_seen": first_seen,
                "last_seen": last_seen,
                "camera_count": len({s["source_key"] for s in stops}),
                "stop_count": len(stops),
                "observed_sec": observed_sec,
                "event_count": len(events),
                "stops": stops,
                "transitions": transitions,
                "no_data": not stops,
            }
        finally:
            db.close()

    @staticmethod
    def _collapse_to_stops(
        events: List[DetectionEvent],
        cameras: Dict[str, Camera],
    ) -> List[Dict[str, Any]]:
        """
        Fold an ordered event list into places-visited.

        A new stop begins when the physical camera changes, or when the gap
        since the last event at this place exceeds JOURNEY_STOP_GAP_SEC.
        """
        gap_limit = settings.JOURNEY_STOP_GAP_SEC
        stops: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for ev in events:
            if ev.created_at is None:
                continue
            cam = cameras.get(ev.camera_id)
            source_key = normalise_stream_source(
                cam.stream_url if cam else None, ev.camera_id
            )

            same_place = current is not None and current["source_key"] == source_key
            within_gap = (
                same_place
                and (ev.created_at - current["_left"]).total_seconds() <= gap_limit
            )

            if within_gap:
                if ev.created_at > current["_left"]:
                    current["_left"] = ev.created_at
                current["event_count"] += 1
                if cam is not None and cam.id not in current["_camera_ids"]:
                    current["_camera_ids"].append(cam.id)
                    current["_cameras"].append(cam)
            else:
                if current is not None:
                    stops.append(current)
                current = {
                    "source_key": source_key,
                    "_arrived": ev.created_at,
                    "_left": ev.created_at,
                    "event_count": 1,
                    "_camera_ids": [cam.id] if cam else [ev.camera_id],
                    "_cameras": [cam] if cam else [],
                }

        if current is not None:
            stops.append(current)

        return [JourneyService._finalise_stop(s, i) for i, s in enumerate(stops)]

    @staticmethod
    def _finalise_stop(stop: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Turn the working stop into its API shape."""
        cam = _pick_display_camera(stop["_cameras"])
        config = (cam.config or {}) if cam else {}
        arrived, left = stop["_arrived"], stop["_left"]

        name = (cam.name if cam else None) or stop["_camera_ids"][0]
        location = (cam.location if cam else "") or ""

        return {
            "index": index,
            "source_key": stop["source_key"],
            "camera_id": (cam.id if cam else stop["_camera_ids"][0]),
            "camera_ids": stop["_camera_ids"],
            "camera_name": name,
            "camera_location": location,
            # `zone` is the human name for the place ("Main Gate"); the icon
            # hints at what kind of place it is. Both live in cameras.config.
            "zone": config.get("zone") or location or name,
            "zone_icon": config.get("zone_icon") or "camera",
            "is_gate": bool(config.get("is_gate")),
            "arrived_at": arrived.isoformat(),
            "left_at": left.isoformat(),
            "dwell_sec": int((left - arrived).total_seconds()),
            "event_count": stop["event_count"],
        }

    @staticmethod
    def _transitions(stops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Camera-to-camera moves between consecutive stops.

        `gap_sec` is time unaccounted for: the person was between cameras, or
        somewhere without coverage. It is reported, never silently absorbed
        into a dwell time -- pretending unobserved time was spent at the
        previous stop would be inventing data.

        `same_place` marks a return to the place just left. That is not a
        journey between locations, it is a spell out of view, and the two must
        read differently: on a single-camera site EVERY transition is
        same_place, and rendering those as "Main Building → Main Building"
        would make a correct journey look broken.
        """
        from datetime import datetime

        out = []
        for a, b in zip(stops, stops[1:]):
            left = datetime.fromisoformat(a["left_at"])
            arrived = datetime.fromisoformat(b["arrived_at"])
            out.append({
                "from_index": a["index"],
                "to_index": b["index"],
                "from_zone": a["zone"],
                "to_zone": b["zone"],
                "same_place": a["source_key"] == b["source_key"],
                "left_at": a["left_at"],
                "arrived_at": b["arrived_at"],
                "gap_sec": max(0, int((arrived - left).total_seconds())),
            })
        return out
