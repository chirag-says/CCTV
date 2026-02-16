"""
Pydantic models for the Movements (detection_events) table.

Movements represent a time-series log of every face detection, entry,
exit, and unknown-face event. This is the core data table that powers
the analytics dashboard and movement-tracking features.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# ── Movement Event Models ─────────────────────────────────────

class MovementCreate(BaseModel):
    """Schema for recording a new movement event."""
    person_id: Optional[str] = None
    camera_id: str
    event_type: str = Field(
        ...,
        description="entry | exit | detection | unknown",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    snapshot_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MovementResponse(BaseModel):
    """Response schema for a movement event."""
    id: str
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    camera_id: str
    camera_name: Optional[str] = None
    event_type: str
    confidence: float
    snapshot_url: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class MovementSummary(BaseModel):
    """Aggregated movement summary for a person or camera."""
    total_events: int = 0
    total_entries: int = 0
    total_exits: int = 0
    total_unknown: int = 0
    unique_persons: int = 0
    avg_confidence: float = 0.0
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class MovementTimeline(BaseModel):
    """Timeline entry for movement history."""
    timestamp: str
    event_type: str
    person_name: Optional[str] = None
    camera_name: Optional[str] = None
    confidence: float = 0.0
    snapshot_url: Optional[str] = None
