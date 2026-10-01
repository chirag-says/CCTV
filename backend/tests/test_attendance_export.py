"""
Attendance export — the camera-detail columns and the multi-sheet workbook.

The DB-backed halves (export_rows / export_camera_log_rows) are exercised
through their column contracts and the pure writer layer; the aggregation
fold that feeds them is tested here directly with synthetic rows, so no
database is required.

Run:  pytest tests/test_attendance_export.py -v
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.attendance_service import AttendanceService  # noqa: E402
from app.utils.export_utils import to_csv, to_xlsx  # noqa: E402

openpyxl = pytest.importorskip("openpyxl")


SUMMARY_ROWS = [
    {
        "Date": "2026-08-27", "Name": "Dr. Rao", "Department": "CSE",
        "Designation": "Professor", "Status": "PRESENT",
        "First Detected": "08:42", "First Camera": "MainGate",
        "Last Detected": "17:21", "Last Camera": "COE Entrance",
        "Minutes Observed": 519, "Sightings": 14, "Cameras": 2,
        "Cameras Seen": "COE Entrance; MainGate",
        "Face Enrolled": "Yes", "Person ID": "p1",
    },
]

LOG_ROWS = [
    {
        "Date": "2026-08-27", "Time": "08:42:03", "Name": "Dr. Rao",
        "Department": "CSE", "Camera": "MainGate", "Location": "Front gate",
        "Event": "Came into view", "Person ID": "p1",
    },
    {
        "Date": "2026-08-27", "Time": "08:44:19", "Name": "Dr. Rao",
        "Department": "CSE", "Camera": "MainGate", "Location": "Front gate",
        "Event": "Left view", "Person ID": "p1",
    },
]


def test_summary_columns_cover_camera_detail():
    """The whole point of the export: where, and on which camera."""
    for col in ("First Camera", "Last Camera", "Cameras Seen"):
        assert col in AttendanceService.EXPORT_COLUMNS
    assert set(SUMMARY_ROWS[0]) == set(AttendanceService.EXPORT_COLUMNS)


def test_log_columns_match_rows():
    assert set(LOG_ROWS[0]) == set(AttendanceService.CAMERA_LOG_COLUMNS)


def test_xlsx_carries_camera_log_as_second_sheet():
    resp = to_xlsx(
        SUMMARY_ROWS,
        AttendanceService.EXPORT_COLUMNS,
        "t.xlsx",
        sheet_title="Attendance",
        extra_sheets=[{
            "title": "Camera Log",
            "rows": LOG_ROWS,
            "columns": AttendanceService.CAMERA_LOG_COLUMNS,
        }],
    )
    wb = openpyxl.load_workbook(io.BytesIO(resp.body))
    assert wb.sheetnames == ["Attendance", "Camera Log"]

    log = wb["Camera Log"]
    header = [c.value for c in log[1]]
    assert header == AttendanceService.CAMERA_LOG_COLUMNS
    # Two data rows plus the header.
    assert log.max_row == 3
    assert log.cell(row=2, column=header.index("Camera") + 1).value == "MainGate"
    assert log.cell(row=3, column=header.index("Event") + 1).value == "Left view"


def test_xlsx_without_extra_sheets_is_single_sheet():
    """The analytics export path passes no extra_sheets — must be unchanged."""
    resp = to_xlsx(SUMMARY_ROWS, AttendanceService.EXPORT_COLUMNS, "t.xlsx")
    wb = openpyxl.load_workbook(io.BytesIO(resp.body))
    assert len(wb.sheetnames) == 1


def test_log_csv_has_header_even_with_no_rows():
    """Zero detections must still download a readable file, not an empty one."""
    resp = to_csv([], AttendanceService.CAMERA_LOG_COLUMNS, "t.csv")
    text = resp.body.decode("utf-8-sig")
    assert text.splitlines()[0].split(",")[:2] == ["Date", "Time"]


def test_aggregation_tracks_first_camera():
    """_aggregate_events must remember WHERE the first detection happened."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch

    t = lambda mm: datetime(2026, 8, 27, 3, mm, tzinfo=timezone.utc)  # noqa: E731
    events = [
        ("p1", "entry", "camB", t(20)),   # deliberately out of order
        ("p1", "entry", "camA", t(12)),   # earliest — first camera is camA
        ("p1", "exit", "camB", t(45)),    # latest — last camera is camB
    ]

    db = MagicMock()
    q = db.query.return_value
    q.filter.return_value.all.return_value = events

    from app.services import attendance_service as svc
    with patch.object(svc, "_camera_source_map", return_value={}):
        agg = svc._aggregate_events(db, ["p1"], t(0), t(59))

    a = agg["p1"]
    assert a["first_camera_id"] == "camA"
    assert a["last_camera_id"] == "camB"
    assert a["first_seen"] == t(12)
    assert a["last_seen"] == t(45)
