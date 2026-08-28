"""
Incident Evidence Pack — a security alert, with everything the system saw
around it.

An "incident" IS a detection_events row with event_type='security_alert'.
There is no separate incidents table: the alert already carries the camera,
the timestamp, the subtype and its detector metadata. Only the human review
state (status, and later notes) lives alongside, in incident_reviews.

WHAT COUNTS AS EVIDENCE
The system has no video. There is no recorder, no ring buffer, no clip
storage — pipeline._last_jpeg is a single in-memory frame, and the video
analysis feature is upload-and-analyse, not capture. So evidence is what
genuinely exists:

  * the alert snapshot   — the annotated frame at the moment of the alert
  * people present       — known AND unenrolled subjects who were in front of
                           the camera during the window, reconstructed from
                           entry/exit intervals in detection_events. This
                           previously asked which unknown_faces rows were
                           FIRST SEEN in the window, which found strangers who
                           happened to be new that minute and missed every
                           repeat visitor -- i.e. most of the people most
                           likely to be involved.
  * related face crops   — the images belonging to those people
  * full frames          — the wider context images saved with those faces
  * a timeline           — every event on that camera around the alert

Historical alerts predate snapshot capture and have none of the first kind;
the pack falls back to the correlated images and says so, rather than showing
a broken image.

LAZY REVIEW ROWS
An alert has no incident_reviews row until somebody acts on it. Absence means
status='open', so opening a list of a thousand alerts writes nothing.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.core.exceptions import NotFoundException
from app.db.models import (
    Camera, DetectionEvent, IncidentReview, Person, UnknownFace,
)
from app.db.session import SessionLocal
from app.services.attendance_service import normalise_stream_source
from app.utils.time_utils import campus_day_bounds, utc_now

logger = logging.getLogger(__name__)

# Stream health is camera plumbing, not security. Excluded by default so a
# flaky RTSP connection does not fill the incident queue.
STREAM_HEALTH_SUBTYPES = ("stream_offline", "stream_recovered")

STATUSES = ("open", "investigating", "resolved")

# Severity is derived from the subtype rather than stored, and returned by the
# API so the frontend stops keeping its own copy of this mapping.
SEVERITY_BY_SUBTYPE = {
    "hazard": "high",
    "vehicle_proximity": "high",
    "gathering": "medium",
    "loitering": "low",
    "stream_offline": "low",
    "stream_recovered": "low",
}

LABEL_BY_SUBTYPE = {
    "hazard": "Hazard / Threat",
    "vehicle_proximity": "Vehicle Proximity",
    "gathering": "Crowd Gathering",
    "loitering": "Loitering Detected",
    "stream_offline": "Camera Offline",
    "stream_recovered": "Camera Recovered",
}

# How far either side of an alert to look for corroborating events.
TIMELINE_WINDOW_SEC = 300
# Rows kept from each side, so the alert is always centred and never pushed
# off the end of a single capped query on a busy camera.
TIMELINE_SIDE_LIMIT = 25


def _severity(subtype: Optional[str]) -> str:
    return SEVERITY_BY_SUBTYPE.get(subtype or "", "medium")


def _label(subtype: Optional[str]) -> str:
    return LABEL_BY_SUBTYPE.get(
        subtype or "", (subtype or "Security Event").replace("_", " ").title()
    )


def _inner_meta(event: DetectionEvent) -> Dict[str, Any]:
    """
    Detector metadata, flattened.

    Detectors build their payload as {"metadata": {...}} alongside top-level
    keys, and _process_pipeline_event copies everything it does not consume
    into metadata_json — so the interesting fields end up one level deeper for
    some event types and top-level for others. Merge both.
    """
    meta = event.metadata_json or {}
    inner = meta.get("metadata")
    return {**meta, **inner} if isinstance(inner, dict) else dict(meta)


def _describe(event: DetectionEvent) -> str:
    """One plain sentence about what the detector saw."""
    meta = _inner_meta(event)
    subtype = event.subtype or ""
    if subtype == "gathering":
        return f"{meta.get('person_count', '?')} people detected in close proximity"
    if subtype == "loitering":
        dur = meta.get("duration_sec")
        who = meta.get("person_name") or "A person"
        return f"{who} stationary for {round(dur)}s" if dur else f"{who} loitering"
    if subtype == "hazard":
        conf = event.confidence or meta.get("confidence") or 0
        return f"{meta.get('threat_class', 'Threat')} detected ({conf * 100:.0f}% confidence)"
    if subtype == "vehicle_proximity":
        return "Vehicle detected in close proximity to a person"
    if subtype in STREAM_HEALTH_SUBTYPES:
        return f"Stream {subtype.replace('stream_', '')}"
    return "Security event detected"


class IncidentService:
    """Read-mostly. The only write is the review status."""

    # ── List ─────────────────────────────────────────────────────

    @staticmethod
    def list_incidents(
        status: Optional[str] = None,
        subtype: Optional[str] = None,
        severity: Optional[str] = None,
        camera_id: Optional[str] = None,
        date: Optional[str] = None,
        include_stream_health: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Security alerts with their review state joined in."""
        db: Session = SessionLocal()
        try:
            q = (
                db.query(DetectionEvent, Camera, IncidentReview)
                .outerjoin(Camera, DetectionEvent.camera_id == Camera.id)
                .outerjoin(IncidentReview, IncidentReview.event_id == DetectionEvent.id)
                .filter(DetectionEvent.event_type == "security_alert")
            )
            if not include_stream_health:
                q = q.filter(
                    (DetectionEvent.subtype.is_(None))
                    | (~DetectionEvent.subtype.in_(STREAM_HEALTH_SUBTYPES))
                )
            if subtype:
                q = q.filter(DetectionEvent.subtype == subtype)
            if severity:
                # Severity is derived from subtype, so it can be resolved back
                # to a subtype set and filtered in SQL. Filtering it in Python
                # after the query would leave `total` reporting the unfiltered
                # count, and the pagination would skip rows silently.
                matching = [
                    st for st, sev in SEVERITY_BY_SUBTYPE.items() if sev == severity
                ]
                if not matching:
                    return {
                        "data": [], "total": 0,
                        "counts_by_status": IncidentService._status_counts(
                            db, include_stream_health
                        ),
                    }
                q = q.filter(DetectionEvent.subtype.in_(matching))
            if camera_id:
                q = q.filter(DetectionEvent.camera_id == camera_id)
            if date:
                start, end = campus_day_bounds(date, day_start="00:00", day_end="24:00")
                q = q.filter(
                    DetectionEvent.created_at >= start,
                    DetectionEvent.created_at < end,
                )
            if status:
                if status == "open":
                    # No review row means nobody has touched it yet, which is
                    # exactly what "open" means.
                    q = q.filter(
                        (IncidentReview.status.is_(None))
                        | (IncidentReview.status == "open")
                    )
                else:
                    q = q.filter(IncidentReview.status == status)

            total = q.count()
            rows = (
                q.order_by(DetectionEvent.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            data = [
                IncidentService._summarise(ev, cam, review)
                for ev, cam, review in rows
            ]

            return {
                "data": data,
                "total": total,
                "counts_by_status": IncidentService._status_counts(
                    db, include_stream_health
                ),
            }
        finally:
            db.close()

    @staticmethod
    def _status_counts(db: Session, include_stream_health: bool) -> Dict[str, int]:
        """Counts per status, with untouched alerts counted as open."""
        q = db.query(
            IncidentReview.status, func.count(DetectionEvent.id)
        ).select_from(DetectionEvent).outerjoin(
            IncidentReview, IncidentReview.event_id == DetectionEvent.id
        ).filter(DetectionEvent.event_type == "security_alert")

        if not include_stream_health:
            q = q.filter(
                (DetectionEvent.subtype.is_(None))
                | (~DetectionEvent.subtype.in_(STREAM_HEALTH_SUBTYPES))
            )

        counts = {s: 0 for s in STATUSES}
        for status, n in q.group_by(IncidentReview.status).all():
            counts[status or "open"] = counts.get(status or "open", 0) + n
        return counts

    @staticmethod
    def _summarise(
        ev: DetectionEvent,
        cam: Optional[Camera],
        review: Optional[IncidentReview],
    ) -> Dict[str, Any]:
        return {
            "event_id": ev.id,
            "seq": review.seq if review else None,
            "subtype": ev.subtype,
            "label": _label(ev.subtype),
            "severity": (review.severity if review and review.severity else _severity(ev.subtype)),
            "status": review.status if review else "open",
            "camera_id": ev.camera_id,
            "camera_name": (cam.name if cam else None) or ev.camera_id,
            "camera_location": (cam.location if cam else "") or "",
            "zone": ((cam.config or {}).get("zone") if cam else None),
            "confidence": ev.confidence,
            "snapshot_url": ev.snapshot_url,
            "summary": _describe(ev),
            "created_at": ev.created_at.isoformat() if ev.created_at else None,
        }

    # ── Evidence pack ────────────────────────────────────────────

    @staticmethod
    def get_evidence_pack(event_id: str) -> Dict[str, Any]:
        """Everything the system recorded around one alert."""
        db: Session = SessionLocal()
        try:
            ev = (
                db.query(DetectionEvent)
                .filter(
                    DetectionEvent.id == event_id,
                    DetectionEvent.event_type == "security_alert",
                )
                .first()
            )
            if not ev:
                raise NotFoundException("Incident", event_id)

            cam = db.query(Camera).filter(Camera.id == ev.camera_id).first()
            review = (
                db.query(IncidentReview)
                .filter(IncidentReview.event_id == event_id)
                .first()
            )
            incident = IncidentService._summarise(ev, cam, review)
            incident["metadata"] = _inner_meta(ev)

            at = ev.created_at
            evidence: Dict[str, Any] = {
                "alert_snapshot": ev.snapshot_url,
                "faces": [],
                "full_frames": [],
                # Historical alerts predate snapshot capture; the UI says so
                # rather than rendering a broken image.
                "has_alert_snapshot": bool(ev.snapshot_url),
            }
            timeline: List[Dict[str, Any]] = []

            if at is not None:
                evidence.update(
                    IncidentService._correlated_images(db, ev.camera_id, at)
                )
                timeline = IncidentService._timeline(db, ev, at)

            return {
                "incident": incident,
                "evidence": evidence,
                "timeline": timeline,
                "review": {
                    "status": review.status if review else "open",
                    "updated_at": (
                        review.updated_at.isoformat()
                        if review and review.updated_at else None
                    ),
                },
                "evidence_window_sec": settings.INCIDENT_EVIDENCE_WINDOW_SEC,
                "presence_lookback_sec": settings.INCIDENT_PRESENCE_LOOKBACK_SEC,
            }
        finally:
            db.close()

    @staticmethod
    def _sibling_camera_ids(db: Session, camera_id: str) -> List[str]:
        """
        Every registration of the SAME PHYSICAL camera.

        One lens is routinely registered more than once -- this deployment has
        MainGate/MainStream on one address, and `security cabin` twice. Events
        for one person are then split across two camera_ids, so asking "who
        was on camera X" by id alone silently loses half of them.
        """
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if cam is None:
            return [camera_id]
        key = normalise_stream_source(cam.stream_url, cam.id)
        ids = [
            c.id for c in db.query(Camera).all()
            if normalise_stream_source(c.stream_url, c.id) == key
        ]
        return ids or [camera_id]

    @staticmethod
    def _people_present(
        db: Session, camera_ids: List[str], lo, hi, lookback
    ) -> Dict[str, Any]:
        """
        Who was actually in front of this camera between `lo` and `hi`.

        WHAT THIS REPLACES
        The previous answer was "unknown_faces rows whose FIRST_SEEN falls in
        the window". first_seen is the first moment that identity was ever
        observed, anywhere -- so it found strangers who happened to be new
        that day, and missed every repeat visitor, who is most of them. For an
        incident review that is close to the worst possible failure: the
        people most likely to be involved are the ones who come back.

        Migration 011 made the real question answerable by promoting the
        stranger's id out of JSON into an indexed column, so known and unknown
        subjects can be selected together in one pass over detection_events.

        PRESENCE IS AN INTERVAL, NOT A POINT
        Someone who entered at 13:50 and left at 14:20 was present at 14:00
        while having no event anywhere near it. Entry/exit pairs are therefore
        reconstructed into intervals and tested for overlap, which is why the
        query reaches `lookback` further back than the window it reports on.
        """
        rows = (
            db.query(
                DetectionEvent.person_id,
                DetectionEvent.unknown_person_id,
                DetectionEvent.event_type,
                DetectionEvent.subtype,
                DetectionEvent.created_at,
            )
            .filter(
                DetectionEvent.camera_id.in_(camera_ids),
                DetectionEvent.created_at >= lo - lookback,
                DetectionEvent.created_at <= hi,
            )
            .order_by(DetectionEvent.created_at.asc())
            .all()
        )

        subjects: Dict[tuple, List[tuple]] = {}
        unidentified = 0
        for person_id, unknown_id, event_type, subtype, ts in rows:
            if subtype == "unidentified_entry" and lo <= ts <= hi:
                # A tracked body that never became identifiable (Step 1.5).
                # Counted, never named -- it cannot be attributed to anyone.
                unidentified += 1
                continue
            if person_id:
                key = ("known", person_id)
            elif unknown_id:
                key = ("unknown", unknown_id)
            else:
                continue
            subjects.setdefault(key, []).append((ts, event_type))

        present: Dict[tuple, Dict[str, Any]] = {}
        for key, events in subjects.items():
            intervals: List[tuple] = []
            open_at = None
            touched = False
            for ts, event_type in events:
                if lo <= ts <= hi:
                    touched = True
                if event_type == "entry":
                    open_at = ts
                elif event_type == "exit" and open_at is not None:
                    intervals.append((open_at, ts))
                    open_at = None
            if open_at is not None:
                # Still inside as far as the record goes.
                intervals.append((open_at, hi))

            overlaps = any(a <= hi and b >= lo for a, b in intervals)
            if not (touched or overlaps):
                continue

            stamps = [ts for ts, _ in events]
            present[key] = {
                "first_seen": min(stamps).isoformat(),
                "last_seen": max(stamps).isoformat(),
                "events": len(events),
                # How we concluded they were here. An operator reading this
                # should be able to tell "seen during the window" from
                # "entered before it and had not left".
                "basis": "observed" if touched else "spanned",
            }

        known_ids = [k[1] for k in present if k[0] == "known"]
        unknown_ids = [k[1] for k in present if k[0] == "unknown"]

        known: List[Dict[str, Any]] = []
        if known_ids:
            for p in db.query(Person).filter(Person.id.in_(known_ids)).all():
                known.append({
                    "person_id": p.id,
                    "name": p.name,
                    "role": p.role,
                    **present[("known", p.id)],
                })

        unknown: List[Dict[str, Any]] = []
        if unknown_ids:
            for f in db.query(UnknownFace).filter(UnknownFace.id.in_(unknown_ids)).all():
                unknown.append({
                    "unknown_id": f.id,
                    "label": f"Unknown-{f.id[:6]}",
                    "snapshot_url": f.snapshot_url,
                    "full_frame": f.full_frame,
                    "occurrence": f.occurrence,
                    "status": f.status,
                    **present[("unknown", f.id)],
                })

        known.sort(key=lambda r: r["first_seen"])
        unknown.sort(key=lambda r: r["first_seen"])
        return {
            "known": known,
            "unknown": unknown,
            "unidentified_count": unidentified,
            "total_identified": len(known) + len(unknown),
        }

    @staticmethod
    def _correlated_images(db: Session, camera_id: str, at) -> Dict[str, Any]:
        """
        The people who were present around the alert, and their images.

        Now derived from _people_present() rather than from unknown_faces
        first_seen. `faces` and `full_frames` keep their shape because the
        evidence grid renders them; what changed is WHICH people are in them
        -- everyone actually present, instead of everyone who happened to be
        newly discovered that minute.
        """
        window = timedelta(seconds=settings.INCIDENT_EVIDENCE_WINDOW_SEC)
        lookback = timedelta(seconds=settings.INCIDENT_PRESENCE_LOOKBACK_SEC)
        camera_ids = IncidentService._sibling_camera_ids(db, camera_id)

        people = IncidentService._people_present(
            db, camera_ids, at - window, at + window, lookback
        )
        unknown = people["unknown"]

        return {
            "people": people,
            "cameras_correlated": len(camera_ids),
            "faces": [
                {
                    "unknown_id": u["unknown_id"],
                    "snapshot_url": u["snapshot_url"],
                    "full_frame": u["full_frame"],
                    "occurrence": u["occurrence"],
                    "first_seen": u["first_seen"],
                }
                for u in unknown if u["snapshot_url"]
            ][:12],
            "full_frames": [
                {"unknown_id": u["unknown_id"], "url": u["full_frame"]}
                for u in unknown if u["full_frame"]
            ][:12],
        }

    @staticmethod
    def _timeline(db: Session, ev: DetectionEvent, at) -> List[Dict[str, Any]]:
        """
        What else happened on this camera around the alert, as sentences.

        The alert itself is included and flagged, so the reader can see what
        led up to it and what followed.
        """
        window = timedelta(seconds=TIMELINE_WINDOW_SEC)

        def side(newer_than_alert: bool):
            q = (
                db.query(DetectionEvent, Person)
                .outerjoin(Person, DetectionEvent.person_id == Person.id)
                .filter(DetectionEvent.camera_id == ev.camera_id)
            )
            if newer_than_alert:
                return (
                    q.filter(
                        DetectionEvent.created_at >= at,
                        DetectionEvent.created_at <= at + window,
                    )
                    .order_by(DetectionEvent.created_at.asc())
                    .limit(TIMELINE_SIDE_LIMIT)
                    .all()
                )
            return (
                q.filter(
                    DetectionEvent.created_at >= at - window,
                    DetectionEvent.created_at < at,
                )
                .order_by(DetectionEvent.created_at.desc())
                .limit(TIMELINE_SIDE_LIMIT)
                .all()
            )

        # Queried as two halves rather than one capped window. A busy camera
        # can emit hundreds of events in five minutes, and a single ascending
        # LIMIT would fill up entirely with the lead-up and drop the alert
        # itself off the end -- the one row the page exists to show. Taking a
        # bounded slice from each side keeps the alert centred.
        rows = list(reversed(side(False))) + side(True)

        out = []
        for row, person in rows:
            if row.event_type == "entry":
                text = f"{person.name if person else 'A person'} detected"
            elif row.event_type == "exit":
                text = f"{person.name if person else 'A person'} no longer detected"
            elif row.event_type == "unknown":
                text = "Unrecognised face captured"
            elif row.event_type == "security_alert":
                text = _describe(row)
            elif row.event_type == "vehicle_entry":
                text = f"Vehicle {_inner_meta(row).get('plate', '')}".strip()
            else:
                text = row.event_type.replace("_", " ").title()

            out.append({
                "id": row.id,
                "at": row.created_at.isoformat() if row.created_at else None,
                "event_type": row.event_type,
                "subtype": row.subtype,
                "text": text,
                "is_alert": row.id == ev.id,
            })
        return out

    # ── Review ───────────────────────────────────────────────────

    @staticmethod
    def set_status(event_id: str, status: str, actor: Optional[str] = None) -> Dict[str, Any]:
        """
        Set an incident's review status, creating the review row on first use.

        Raises:
            NotFoundException: no such alert
            ValueError:        invalid status
        """
        if status not in STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Allowed: {', '.join(STATUSES)}"
            )

        db: Session = SessionLocal()
        try:
            ev = (
                db.query(DetectionEvent)
                .filter(
                    DetectionEvent.id == event_id,
                    DetectionEvent.event_type == "security_alert",
                )
                .first()
            )
            if not ev:
                raise NotFoundException("Incident", event_id)

            review = (
                db.query(IncidentReview)
                .filter(IncidentReview.event_id == event_id)
                .first()
            )
            if review is None:
                review = IncidentReview(event_id=event_id, status=status)
                db.add(review)
            else:
                review.status = status
                review.updated_at = utc_now()

            db.commit()
            db.refresh(review)
            return {
                "event_id": event_id,
                "seq": review.seq,
                "status": review.status,
                "updated_at": review.updated_at.isoformat() if review.updated_at else None,
            }
        except (NotFoundException, ValueError):
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to set incident status: {e}")
            raise
        finally:
            db.close()
