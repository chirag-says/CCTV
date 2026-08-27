"""
Faculty Attendance API Routes.

Read-only endpoints over the derived attendance model. Nothing here writes:
attendance is a question asked of detection_events, not a separate record.
See app/services/attendance_service.py for the rule and its rationale.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.exceptions import NotFoundException
from app.core.security import get_current_user
from app.services.attendance_service import AttendanceService
from app.services.presence_service import PresenceService
from app.utils.export_utils import export_response, validate_format

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])

_STATUSES = ("present", "late", "absent")


def _validate_status(status: Optional[str]) -> Optional[str]:
    if status is None or status == "":
        return None
    s = status.strip().lower()
    if s not in _STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{status}'. Allowed: {', '.join(_STATUSES)}",
        )
    return s


@router.get("/summary")
async def get_attendance_summary(
    date: Optional[str] = Query(
        None, description="Campus-local day (YYYY-MM-DD). Defaults to today."
    ),
    current_user: dict = Depends(get_current_user),
):
    """
    Department-wise attendance rollup with campus totals.

    `no_data` is true when the campus recorded no entry/exit events at all
    that day (a holiday, or cameras down). Clients must suppress the
    percentage headline in that case rather than reporting 0%.
    """
    try:
        return AttendanceService.get_summary(date=date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/day")
async def get_attendance_day(
    date: Optional[str] = Query(None, description="Campus-local day (YYYY-MM-DD)"),
    department: Optional[str] = Query(None, description="Department (case-insensitive)"),
    status: Optional[str] = Query(None, description="present/late/absent"),
    current_user: dict = Depends(get_current_user),
):
    """
    Per-person attendance records for one campus-local day.

    Each record carries `sightings` and `camera_count` alongside the status,
    so an absence that fell just short of the rule is visible and
    explainable rather than silent.
    """
    try:
        return AttendanceService.get_day(
            date=date,
            department=department,
            status=_validate_status(status),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/person/{person_id}")
async def get_person_attendance(
    person_id: str,
    date_from: Optional[str] = Query(None, alias="from", description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, alias="to", description="YYYY-MM-DD"),
    current_user: dict = Depends(get_current_user),
):
    """
    One person's attendance history. Defaults to the last 30 days.
    """
    try:
        return AttendanceService.get_person_range(
            person_id=person_id, date_from=date_from, date_to=date_to
        )
    except NotFoundException:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/on-campus")
async def get_on_campus(
    department: Optional[str] = Query(None, description="Department (case-insensitive)"),
    q: Optional[str] = Query(None, description="Filter by name"),
    current_user: dict = Depends(get_current_user),
):
    """
    Faculty presence right now.

    `present_now` counts only people seen within the confidence window (or
    visible on a camera this second). Anyone detected today but longer ago is
    reported separately as `seen_earlier` — a 20-minute-old sighting is a
    confident claim about someone being on campus, an 8-hour-old one is not,
    and merging them into one headline would make the whole figure deniable.

    Every row carries `minutes_since_seen`, so the UI can always state how
    fresh the claim is.
    """
    try:
        return PresenceService.get_on_campus(department=department, query=q)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/export")
async def export_attendance(
    date: Optional[str] = Query(None, description="Campus-local day (YYYY-MM-DD)"),
    format: str = Query("csv", description="csv, xlsx, or json"),
    department: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="present/late/absent"),
    current_user: dict = Depends(get_current_user),
):
    """
    Download a day's attendance as CSV, Excel (.xlsx), or JSON.

    Times are rendered campus-local: a spreadsheet cell carries no timezone,
    and a UTC arrival time would read as five and a half hours early.
    """
    fmt = validate_format(format)
    try:
        rows = AttendanceService.export_rows(
            date=date, department=department, status=_validate_status(status)
        )
        summary = AttendanceService.get_summary(date=date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    suffix = f"_{department.lower()}" if department else ""
    return export_response(
        rows=rows,
        columns=AttendanceService.EXPORT_COLUMNS,
        fmt=fmt,
        filename_prefix=f"sentinel_attendance_{summary['date']}{suffix}",
        sheet_title="Attendance",
        meta={
            "date": summary["date"],
            "department": department,
            "totals": summary["totals"],
            "rule": summary["rule"],
        },
    )
