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
    import csv
    import io
    import json
    from datetime import datetime
    from fastapi.responses import StreamingResponse, Response

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
    if format not in ("csv", "xlsx", "json"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{format}'. Allowed: csv, xlsx, json",
        )

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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"sentinel_{type}_{stamp}"

    # Stable, human-ordered columns. Taken from the first row so the header is
    # present even when there are no rows to write.
    COLUMNS = [
        "Date", "Time", "Event", "Person", "Role", "Department",
        "Camera", "Location", "Confidence", "Duration (s)",
        "Timestamp (UTC)", "Person ID", "Event ID",
    ]

    # ── CSV ───────────────────────────────────────────────────────
    if format == "csv":
        out = io.StringIO()
        # utf-8-sig BOM so Excel opens accented names correctly on Windows
        writer = csv.DictWriter(out, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        return Response(
            content=out.getvalue().encode("utf-8-sig"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{base}.csv"'},
        )

    # ── Excel ─────────────────────────────────────────────────────
    if format == "xlsx":
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.utils import get_column_letter
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Excel export requires openpyxl. Install it with: pip install openpyxl",
            )

        wb = Workbook()
        ws = wb.active
        ws.title = type.replace("_", " ").title()[:31]  # Excel caps sheet names at 31

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", fgColor="4F46E5")
        ws.append(COLUMNS)
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        for r in rows:
            ws.append([r.get(c, "") for c in COLUMNS])

        # Width from the widest cell, capped so IDs don't blow the layout out
        for i, col in enumerate(COLUMNS, start=1):
            longest = max([len(col)] + [len(str(r.get(col, ""))) for r in rows[:500]])
            ws.column_dimensions[get_column_letter(i)].width = min(longest + 2, 32)

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{max(len(rows) + 1, 1)}"

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{base}.xlsx"'},
        )

    # ── JSON ──────────────────────────────────────────────────────
    payload = json.dumps(
        {"exported_at": stamp, "type": type, "count": len(rows), "data": rows},
        indent=2, default=str,
    )
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{base}.json"'},
    )

