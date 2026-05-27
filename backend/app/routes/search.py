"""
Global Search API Route — searches across persons, events, and vehicles.
"""

import logging
from fastapi import APIRouter, Depends, Query
from typing import Optional
from sqlalchemy.orm import joinedload

from app.core.security import get_current_user
from app.db.session import SessionLocal
from app.db.models import Person, DetectionEvent, UnknownFace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["Search"])


@router.get("")
async def global_search(
    q: str = Query(..., min_length=1, max_length=100, description="Search query"),
    type: Optional[str] = Query(None, description="Filter: all, persons, events, plates"),
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_current_user),
):
    """
    Search across persons, events, unknown faces, and vehicle plates.
    Returns unified results grouped by category.
    """
    query = q.strip().lower()
    results = []
    db = SessionLocal()

    try:
        search_type = (type or "all").lower()

        # ── Search Persons ──────────────────────────────────
        if search_type in ("all", "persons"):
            try:
                persons = (
                    db.query(Person)
                    .filter(Person.name.ilike(f"%{query}%"))
                    .limit(limit)
                    .all()
                )
                for p in persons:
                    results.append({
                        "type": "person",
                        "id": p.id,
                        "title": p.name,
                        "subtitle": p.role or "No role",
                        "timestamp": p.created_at.isoformat() if p.created_at else None,
                        "thumbnail_url": f"/snapshots/{p.id}_face.jpg" if p.id else None,
                    })
            except Exception as e:
                logger.warning(f"Person search failed: {e}")

        # ── Search Events ───────────────────────────────────
        if search_type in ("all", "events"):
            try:
                events = (
                    db.query(DetectionEvent)
                    .outerjoin(Person, DetectionEvent.person_id == Person.id)
                    .filter(
                        (Person.name.ilike(f"%{query}%")) |
                        (DetectionEvent.event_type.ilike(f"%{query}%")) |
                        (DetectionEvent.camera_id.ilike(f"%{query}%"))
                    )
                    .order_by(DetectionEvent.created_at.desc())
                    .limit(limit)
                    .all()
                )
                for e in events:
                    # Get person name via relationship
                    person_name = e.person.name if e.person else "Unknown"
                    results.append({
                        "type": "event",
                        "id": e.id,
                        "title": f"{e.event_type}: {person_name}",
                        "subtitle": f"Camera: {e.camera_id}",
                        "timestamp": e.created_at.isoformat() if e.created_at else None,
                        "thumbnail_url": e.snapshot_url,
                    })
            except Exception as e:
                logger.warning(f"Event search failed: {e}")

        # ── Search Unknown Faces ────────────────────────────
        if search_type in ("all", "persons"):
            try:
                unknowns = (
                    db.query(UnknownFace)
                    .filter(UnknownFace.status.ilike(f"%{query}%"))
                    .limit(min(limit, 10))
                    .all()
                )
                for u in unknowns:
                    results.append({
                        "type": "unknown_face",
                        "id": u.id,
                        "title": f"Unknown Face #{u.id[:8]}",
                        "subtitle": f"Seen {u.occurrence}x • {u.status}",
                        "timestamp": u.first_seen.isoformat() if u.first_seen else None,
                        "thumbnail_url": u.snapshot_url,
                    })
            except Exception as e:
                logger.warning(f"Unknown face search failed: {e}")

        # ── Search Vehicle Plates ───────────────────────────
        if search_type in ("all", "plates"):
            try:
                # Search events with plate data in metadata_json
                plate_events = (
                    db.query(DetectionEvent)
                    .outerjoin(Person, DetectionEvent.person_id == Person.id)
                    .filter(
                        DetectionEvent.event_type == "vehicle_entry",
                        Person.name.ilike(f"%{query}%")
                    )
                    .order_by(DetectionEvent.created_at.desc())
                    .limit(limit)
                    .all()
                )
                for e in plate_events:
                    person_name = e.person.name if e.person else "Unknown Plate"
                    results.append({
                        "type": "vehicle",
                        "id": e.id,
                        "title": person_name,
                        "subtitle": f"Camera: {e.camera_id}",
                        "timestamp": e.created_at.isoformat() if e.created_at else None,
                        "thumbnail_url": e.snapshot_url,
                    })
            except Exception as e:
                logger.warning(f"Plate search failed: {e}")

    finally:
        db.close()

    return {
        "query": q,
        "total": len(results),
        "results": results[:limit],
    }
