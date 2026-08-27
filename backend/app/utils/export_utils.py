"""
Tabular export writers — CSV / Excel / JSON.

Lifted verbatim (behaviour-for-behaviour) out of routes/analytics.py so the
attendance reports and the entry/exit export share one implementation rather
than growing a second, slightly-different Excel writer.

Every writer takes the same shape:
    rows:    list of dicts, one per record
    columns: explicit ordered column names -- NOT derived from the rows, so
             the header is still correct when there are zero rows
"""

import csv
import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from fastapi import HTTPException
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Formats every export endpoint accepts.
EXPORT_FORMATS = ("csv", "xlsx", "json")

# Indigo header fill, matching the app's --primary token.
_HEADER_FILL = "4F46E5"


def validate_format(fmt: str) -> str:
    """Normalise and validate an export format, or raise 400."""
    fmt = (fmt or "").strip().lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{fmt}'. Allowed: {', '.join(EXPORT_FORMATS)}",
        )
    return fmt


def export_filename(prefix: str, fmt: str, stamp: Optional[str] = None) -> str:
    """Build a timestamped filename: sentinel_attendance_20260827_104312.xlsx"""
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{fmt}"


def to_csv(rows: List[Dict[str, Any]], columns: Sequence[str], filename: str) -> Response:
    """
    CSV response.

    Encoded utf-8-sig: the BOM is what makes Excel on Windows read accented
    names correctly instead of mojibake.
    """
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return Response(
        content=out.getvalue().encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def to_xlsx(
    rows: List[Dict[str, Any]],
    columns: Sequence[str],
    filename: str,
    sheet_title: str = "Report",
) -> Response:
    """
    Styled Excel response: bold white-on-indigo header, frozen header row,
    auto-filter, and column widths sized to content.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Excel export requires openpyxl. Install it with: pip install openpyxl",
        )

    columns = list(columns)
    wb = Workbook()
    ws = wb.active
    ws.title = (sheet_title or "Report")[:31]  # Excel caps sheet names at 31

    ws.append(columns)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor=_HEADER_FILL)
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for r in rows:
        ws.append([r.get(c, "") for c in columns])

    # Width from the widest cell, capped so long IDs don't blow the layout
    # out. Sampled from the first 500 rows -- measuring 50k rows is slow and
    # makes no visible difference.
    for i, col in enumerate(columns, start=1):
        longest = max([len(str(col))] + [len(str(r.get(col, ""))) for r in rows[:500]])
        ws.column_dimensions[get_column_letter(i)].width = min(longest + 2, 32)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{max(len(rows) + 1, 1)}"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def to_json(
    rows: List[Dict[str, Any]],
    filename: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Response:
    """JSON response with a small envelope of export metadata."""
    payload = json.dumps(
        {
            "exported_at": datetime.now().isoformat(),
            **(meta or {}),
            "count": len(rows),
            "data": rows,
        },
        indent=2,
        default=str,
    )
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def export_response(
    rows: List[Dict[str, Any]],
    columns: Sequence[str],
    fmt: str,
    filename_prefix: str,
    sheet_title: str = "Report",
    meta: Optional[Dict[str, Any]] = None,
) -> Response:
    """
    Dispatch to the right writer. `fmt` must already be validated.

    Returns a FastAPI Response with Content-Disposition set, so the caller is
    a one-liner.
    """
    filename = export_filename(filename_prefix, fmt)
    if fmt == "csv":
        return to_csv(rows, columns, filename)
    if fmt == "xlsx":
        return to_xlsx(rows, columns, filename, sheet_title=sheet_title)
    return to_json(rows, filename, meta=meta)
