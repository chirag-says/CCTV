"""
Time and date utility functions used across the application.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional


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
    """Get the start of today (midnight) in UTC."""
    now = utc_now()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def days_ago_utc(days: int) -> datetime:
    """Get a datetime N days ago from now in UTC."""
    return utc_now() - timedelta(days=days)
