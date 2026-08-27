"""
Analytics API Routes.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from app.services.analytics_service import AnalyticsService
from app.core.security import get_current_user

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/dashboard")
async def get_dashboard(
    current_user: dict = Depends(get_current_user),
):
    """Get dashboard summary data."""
    return AnalyticsService.get_dashboard_summary()


@router.get("/peak-times")
async def get_peak_times(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    current_user: dict = Depends(get_current_user),
):
    """Get peak entry/exit time analysis."""
    return AnalyticsService.get_peak_times(days=days)


@router.get("/daily-summary")
async def get_daily_summary(
    date: Optional[str] = Query(
        None, description="Campus-local day (YYYY-MM-DD). Defaults to today."
    ),
    current_user: dict = Depends(get_current_user),
):
    """
    Administration-facing summary of one campus day.

    Composed entirely from data already recorded — no new capture.

    Two metrics carry caveats the UI must preserve:
      * `peak_occupancy` is RECONSTRUCTED from presence intervals, since
        occupancy is never stored. Present it as an estimate.
      * `camera_availability_pct` is null when no camera ran that day. Show a
        dash, never 100%.

    `no_data` is true when the day recorded no events at all; clients must
    suppress headline figures rather than reporting zeros.
    """
    from app.services.campus_summary_service import CampusSummaryService

    try:
        return CampusSummaryService.get_daily(date=date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/occupancy")
async def get_occupancy(
    current_user: dict = Depends(get_current_user),
):
    """Get real-time occupancy data."""
    return AnalyticsService.get_occupancy_data()


@router.get("/movement")
async def get_movement_logs(
    person_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """Get detailed movement / event logs."""
    return AnalyticsService.get_movement_logs(
        person_id=person_id,
        camera_id=camera_id,
        event_type=event_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )


@router.get("/reports")
async def generate_report(
    report_type: str = Query("daily", description="daily/weekly/monthly"),
    date: Optional[str] = Query(None, description="Target date (YYYY-MM-DD)"),
    current_user: dict = Depends(get_current_user),
):
    """Generate analytics report."""
    return AnalyticsService.generate_report(
        report_type=report_type,
        date=date,
    )


@router.get("/export")
async def export_analytics(
    format: str = Query("csv", description="Export format: csv, xlsx, or json"),
    type: str = Query("entry_exit", description="Dataset: entry_exit, entries, exits, events"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    camera_id: Optional[str] = Query(None),
    person_id: Optional[str] = Query(None),
    limit: int = Query(10000, ge=1, le=50000),
    current_user: dict = Depends(get_current_user),
):
    """
    Export entry/exit records as CSV, Excel (.xlsx), or JSON.

    Rows are human-readable: person and camera are resolved to names, not the
    raw UUIDs stored on detection_events.

    `type` selects the dataset:
        entry_exit  both entries and exits (default)
        entries     entries only
        exits       exits only
        events      every event type, including alerts and unknowns
    """
    from app.utils.export_utils import export_response, validate_format

    TYPE_FILTERS = {
        "entry_exit": ["entry", "exit"],
        "entries": ["entry"],
        "exits": ["exit"],
        "events": None,   # no filter
    }
    if type not in TYPE_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid type '{type}'. Allowed: {', '.join(TYPE_FILTERS)}",
        )
    format = validate_format(format)

    event_types = TYPE_FILTERS[type]
    if event_types is None:
        event_types = ["entry", "exit", "unknown", "security_alert", "vehicle_entry"]

    rows = AnalyticsService.get_entry_exit_export(
        start_date=start_date,
        end_date=end_date,
        event_types=event_types,
        camera_id=camera_id,
        person_id=person_id,
        limit=limit,
    )

    # Stable, human-ordered columns. Declared explicitly rather than derived
    # from the rows, so the header is present even when there are no rows.
    COLUMNS = [
        "Date", "Time", "Event", "Person", "Role", "Department",
        "Camera", "Location", "Confidence", "Duration (s)",
        "Timestamp (UTC)", "Person ID", "Event ID",
    ]

    return export_response(
        rows=rows,
        columns=COLUMNS,
        fmt=format,
        filename_prefix=f"sentinel_{type}",
        sheet_title=type.replace("_", " ").title(),
        meta={"type": type},
    )

