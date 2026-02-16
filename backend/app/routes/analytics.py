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
