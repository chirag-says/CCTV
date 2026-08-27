"""
Security Incident API Routes.

An incident is a detection_events row with event_type='security_alert'. These
endpoints add the review workflow and assemble the evidence around it; they
never create alerts.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.core.exceptions import NotFoundException
from app.core.security import get_current_user, require_admin
from app.services.incident_service import STATUSES, IncidentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/incidents", tags=["Incidents"])


@router.get("")
async def list_incidents(
    status: Optional[str] = Query(None, description="open/investigating/resolved"),
    subtype: Optional[str] = Query(None, description="hazard/gathering/loitering/…"),
    severity: Optional[str] = Query(None, description="high/medium/low"),
    camera_id: Optional[str] = Query(None),
    date: Optional[str] = Query(None, description="Campus-local day (YYYY-MM-DD)"),
    include_stream_health: bool = Query(
        False, description="Include camera offline/recovered events"
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """
    Security incidents, newest first, with review status joined in.

    Camera stream-health alerts are excluded by default: a flaky RTSP
    connection is plumbing, not a security event, and counting it as one makes
    the incident figure meaningless.
    """
    if status and status not in STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{status}'. Allowed: {', '.join(STATUSES)}",
        )
    try:
        return IncidentService.list_incidents(
            status=status,
            subtype=subtype,
            severity=severity,
            camera_id=camera_id,
            date=date,
            include_stream_health=include_stream_health,
            limit=limit,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{event_id}")
async def get_incident(
    event_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Full evidence pack for one incident.

    Evidence is what the system genuinely holds: the annotated frame captured
    with the alert, face crops and full frames from the same camera within the
    correlation window, and a timeline of surrounding events. There is no
    video — this system has no recorder.
    """
    try:
        return IncidentService.get_evidence_pack(event_id)
    except NotFoundException:
        raise HTTPException(status_code=404, detail=f"Incident {event_id} not found")


@router.patch("/{event_id}")
async def update_incident(
    event_id: str,
    payload: dict = Body(..., example={"status": "investigating"}),
    current_user: dict = Depends(require_admin),
):
    """Update an incident's review status. Admin only."""
    status = (payload or {}).get("status")
    if not status:
        raise HTTPException(status_code=400, detail="`status` is required")
    try:
        return IncidentService.set_status(
            event_id, status, actor=current_user.get("id")
        )
    except NotFoundException:
        raise HTTPException(status_code=404, detail=f"Incident {event_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
