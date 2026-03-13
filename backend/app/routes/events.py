"""
Events & Tracking API Routes.
Now uses SQLAlchemy + PostgreSQL.
"""

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query
from typing import Optional

from app.models.event import DetectionEventResponse, TrackingSessionResponse
from app.core.security import get_current_user
from app.core.websocket import ws_manager
from app.db.session import SessionLocal
from app.db.models import DetectionEvent, TrackingSession
from app.vision.camera_worker import camera_manager

router = APIRouter(prefix="/api", tags=["Events & Tracking"])


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


def _session_to_dict(s: TrackingSession) -> dict:
    return {
        "id": s.id,
        "person_id": s.person_id,
        "camera_id": s.camera_id,
        "entry_time": s.entry_time.isoformat() if s.entry_time else None,
        "exit_time": s.exit_time.isoformat() if s.exit_time else None,
        "duration_sec": s.duration_sec,
        "status": s.status,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }


@router.get("/events")
async def list_events(
    event_type: Optional[str] = Query(None, description="entry/exit/detection/unknown"),
    subtype: Optional[str] = Query(None, description="loitering/gathering/vehicle_proximity"),
    person_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List detection events with filtering."""
    db = SessionLocal()
    try:
        query = db.query(DetectionEvent)

        if event_type:
            query = query.filter(DetectionEvent.event_type == event_type)
        if subtype:
            query = query.filter(DetectionEvent.subtype == subtype)
        if person_id:
            query = query.filter(DetectionEvent.person_id == person_id)
        if camera_id:
            query = query.filter(DetectionEvent.camera_id == camera_id)
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


@router.websocket("/events/live")
async def live_events(websocket: WebSocket):
    """
    WebSocket endpoint for live detection events.
    Broadcasts entry/exit/detection/unknown events in real-time.
    """
    await ws_manager.connect_events(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect_events(websocket)


@router.get("/sessions")
async def list_sessions(
    person_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="active/completed"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List tracking sessions with filtering."""
    db = SessionLocal()
    try:
        query = db.query(TrackingSession)

        if person_id:
            query = query.filter(TrackingSession.person_id == person_id)
        if camera_id:
            query = query.filter(TrackingSession.camera_id == camera_id)
        if status:
            query = query.filter(TrackingSession.status == status)
        if start_date:
            query = query.filter(TrackingSession.entry_time >= start_date)
        if end_date:
            query = query.filter(TrackingSession.entry_time <= end_date)

        total = query.count()
        sessions = (
            query.order_by(TrackingSession.entry_time.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "data": [_session_to_dict(s) for s in sessions],
            "total": total,
        }
    finally:
        db.close()


@router.get("/sessions/active")
async def get_active_sessions(
    current_user: dict = Depends(get_current_user),
):
    """Get currently present (active) people from all cameras."""
    active_persons = []
    seen_ids = set()

    for camera_id in camera_manager.get_active_camera_ids():
        worker = camera_manager._workers.get(camera_id)
        if worker:
            for person in worker.pipeline.tracker.get_active_persons():
                if person["person_id"] not in seen_ids:
                    active_persons.append(person)
                    seen_ids.add(person["person_id"])

    return {
        "active_count": len(active_persons),
        "persons": active_persons,
    }
