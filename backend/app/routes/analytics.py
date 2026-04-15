"""
Analytics API Routes.
"""

from fastapi import APIRouter, Depends, Query
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
    format: str = Query("csv", description="Export format: csv or json"),
    type: str = Query("events", description="Data type: events, movements, summary"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, ge=1, le=10000),
    current_user: dict = Depends(get_current_user),
):
    """Export analytics data as CSV or JSON file download."""
    import csv
    import io
    from datetime import datetime
    from fastapi.responses import StreamingResponse

    # Get the data
    data = AnalyticsService.get_movement_logs(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=0,
    )

    events = data.get("events", data.get("movements", []))
    if not isinstance(events, list):
        events = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "csv":
        output = io.StringIO()
        if events:
            # Get all unique keys from events
            all_keys = set()
            for event in events:
                if isinstance(event, dict):
                    all_keys.update(event.keys())
            fieldnames = sorted(all_keys)

            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for event in events:
                if isinstance(event, dict):
                    writer.writerow(event)

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=sentinel_{type}_{timestamp}.csv"
            },
        )
    else:
        # JSON format
        import json
        json_data = json.dumps({"exported_at": timestamp, "count": len(events), "data": events}, indent=2, default=str)
        return StreamingResponse(
            iter([json_data]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=sentinel_{type}_{timestamp}.json"
            },
        )

