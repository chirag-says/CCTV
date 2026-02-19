"""
Events & Tracking API Routes.
"""

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query
from typing import Optional

from app.models.event import DetectionEventResponse, TrackingSessionResponse
from app.core.security import get_current_user
from app.core.websocket import ws_manager
from app.database import get_admin_db
from app.vision.camera_worker import camera_manager

from app.core.mock_store import mock_store

router = APIRouter(prefix="/api", tags=["Events & Tracking"])


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
    db = get_admin_db()

    if db is None:
        # Mock mode — use in-memory store
        return mock_store.list_events(
            event_type=event_type,
            subtype=subtype,
            camera_id=camera_id,
            limit=limit,
            offset=offset,
        )

    query = db.table("detection_events").select("*", count="exact")

    if event_type:
        query = query.eq("event_type", event_type)
    if subtype:
        query = query.eq("subtype", subtype)
    if person_id:
        query = query.eq("person_id", person_id)
    if camera_id:
        query = query.eq("camera_id", camera_id)
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
    db = get_admin_db()

    if db is None:
        return {"data": [], "total": 0}

    query = db.table("tracking_sessions").select("*", count="exact")

    if person_id:
        query = query.eq("person_id", person_id)
    if camera_id:
        query = query.eq("camera_id", camera_id)
    if status:
        query = query.eq("status", status)
    if start_date:
        query = query.gte("entry_time", start_date)
    if end_date:
        query = query.lte("entry_time", end_date)

    query = query.order("entry_time", desc=True)
    query = query.range(offset, offset + limit - 1)

    result = query.execute()

    return {
        "data": result.data or [],
        "total": result.count or 0,
    }


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
