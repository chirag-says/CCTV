"""
Time and date utility functions used across the application.
"""

from datetime import datetime, timezone, timedelta, date, time
from typing import Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def utc_now() -> datetime:
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Get current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(iso_string: str) -> Optional[datetime]:
    """
    Parse an ISO 8601 datetime string, handling common variants.

    Args:
        iso_string: ISO format datetime string

    Returns:
        Parsed datetime or None if parsing fails
    """
    try:
        # Handle 'Z' suffix (common in JavaScript)
        cleaned = iso_string.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, AttributeError):
        return None


def format_duration(seconds: int) -> str:
    """
    Format seconds into human-readable duration string.

    Examples:
        45 → "45s"
        150 → "2m 30s"
        3661 → "1h 1m 1s"
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        parts = [f"{hours}h"]
        if minutes:
            parts.append(f"{minutes}m")
        if secs:
            parts.append(f"{secs}s")
        return " ".join(parts)


def today_start_utc() -> datetime:
    """
    Get the start of today (midnight) in UTC.

    NOTE: this is a UTC day, not a campus day. For anything that a human
    would call "today" (attendance, journeys, daily summaries) use
    campus_day_bounds() instead -- UTC midnight is 05:30 in IST, so a
    UTC-day boundary splits a college day in the wrong place.
    """
    now = utc_now()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def days_ago_utc(days: int) -> datetime:
    """Get a datetime N days ago from now in UTC."""
    return utc_now() - timedelta(days=days)


# ── Campus-local time ────────────────────────────────────────────
#
# Storage stays UTC everywhere. These helpers exist so that day
# boundaries and time-of-day rules ("late after 09:15") are evaluated
# in the campus's own timezone, which is the only way they mean
# anything. Convert at the edges: query bounds in, formatting out.

def campus_tz() -> ZoneInfo:
    """
    The campus timezone from settings.

    Raises at call time (not import time) if the zone is unknown, so a
    typo in CAMPUS_TIMEZONE surfaces as a clear error on the first
    request rather than as a failed application boot.

    Requires the `tzdata` package on Windows, which has no system
    timezone database. It is pinned in requirements.txt.
    """
    from app.config import settings

    name = settings.CAMPUS_TIMEZONE
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as e:
        raise RuntimeError(
            f"CAMPUS_TIMEZONE={name!r} is not a known IANA timezone. "
            f"On Windows this also happens when the `tzdata` package is "
            f"missing -- install it with: pip install tzdata"
        ) from e


def to_campus_local(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Convert a stored timestamp to campus-local time.

    Naive datetimes are assumed UTC, which is what every writer in this
    codebase produces (datetime.now(timezone.utc), sometimes stripped of
    tzinfo by the driver on read).
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(campus_tz())


def parse_campus_date(date_str: Optional[str] = None) -> date:
    """
    Parse a 'YYYY-MM-DD' string into a date, defaulting to the current
    campus-local date (NOT the UTC date -- between 18:30 and 24:00 UTC
    those differ in IST, and "today" must mean the campus's today).
    """
    if not date_str:
        return to_campus_local(utc_now()).date()
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid date {date_str!r} -- expected YYYY-MM-DD"
        ) from e


def campus_day_bounds(
    date_str: Optional[str] = None,
    day_start: Optional[str] = None,
    day_end: Optional[str] = None,
) -> Tuple[datetime, datetime]:
    """
    UTC half-open bounds [start, end) for one campus-local day.

    Args:
        date_str:  'YYYY-MM-DD' campus-local date. None → campus today.
        day_start: 'HH:MM' campus-local. None → settings.ATTENDANCE_DAY_START.
        day_end:   'HH:MM' campus-local. None → settings.ATTENDANCE_DAY_END.

    Returns:
        (start_utc, end_utc) -- pass straight into a SQLAlchemy filter:
            created_at >= start_utc, created_at < end_utc

    The window is the campus *operating* day, not midnight-to-midnight.
    Detections outside 06:00-21:00 local are not part of a working day and
    would otherwise let a 02:00 false positive count as an arrival.

    A day_end of "24:00" is accepted and means midnight at the end of the
    day, for callers that genuinely want the full calendar day.
    """
    from app.config import settings

    day = parse_campus_date(date_str)
    tz = campus_tz()

    start_t = _parse_hhmm(day_start or settings.ATTENDANCE_DAY_START)
    end_raw = day_end or settings.ATTENDANCE_DAY_END

    start_local = datetime.combine(day, start_t, tzinfo=tz)

    if end_raw.strip() in ("24:00", "23:59:59"):
        end_local = datetime.combine(day, time(0, 0), tzinfo=tz) + timedelta(days=1)
    else:
        end_local = datetime.combine(day, _parse_hhmm(end_raw), tzinfo=tz)

    # A day_end at or before day_start would silently return zero rows.
    if end_local <= start_local:
        raise ValueError(
            f"Campus day end ({end_raw}) must be after start "
            f"({day_start or settings.ATTENDANCE_DAY_START})"
        )

    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def is_after_campus_time(dt: datetime, hhmm: str) -> bool:
    """
    Is `dt` later than the given campus-local time-of-day on its own day?

    Used for the late-arrival rule: is_after_campus_time(first_seen,
    settings.ATTENDANCE_LATE_AFTER).
    """
    local = to_campus_local(dt)
    return local.timetz().replace(tzinfo=None) > _parse_hhmm(hhmm)


def campus_now() -> datetime:
    """Current time, campus-local. For time-of-day comparisons."""
    return to_campus_local(utc_now())


def _parse_hhmm(value: str) -> time:
    """Parse 'HH:MM' or 'HH:MM:SS' into a time."""
    try:
        parts = [int(p) for p in value.strip().split(":")]
        while len(parts) < 3:
            parts.append(0)
        return time(parts[0], parts[1], parts[2])
    except (ValueError, IndexError, AttributeError) as e:
        raise ValueError(
            f"Invalid time-of-day {value!r} -- expected HH:MM"
        ) from e
