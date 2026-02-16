"""
Pydantic models for Detection Events, Tracking Sessions, and Unknown Faces.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# ── Detection Event Models ────────────────────────────────────

class DetectionEventCreate(BaseModel):
    person_id: Optional[str] = None
    camera_id: str
    event_type: str = Field(..., description="entry/exit/detection/unknown")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    snapshot_url: Optional[str] = None
    metadata: Dict[str, Any] = {}


class DetectionEventResponse(BaseModel):
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


# ── Tracking Session Models ───────────────────────────────────

class TrackingSessionResponse(BaseModel):
    id: str
    person_id: str
    person_name: Optional[str] = None
    camera_id: str
    camera_name: Optional[str] = None
    entry_time: str
    exit_time: Optional[str] = None
    duration_sec: Optional[int] = None
    status: str = "active"
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


# ── Unknown Face Models ───────────────────────────────────────

class UnknownFaceResponse(BaseModel):
    id: str
    camera_id: str
    camera_name: Optional[str] = None
    snapshot_url: Optional[str] = None
    full_frame: Optional[str] = None
    occurrence: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    status: str = "pending"
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class EnrollUnknownFace(BaseModel):
    """Schema for enrolling an unknown face to a known person."""
    name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(default="visitor")
    department: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


# ── Live Event (WebSocket) ────────────────────────────────────

class LiveDetectionEvent(BaseModel):
    """Sent via WebSocket for real-time detection updates."""
    event_type: str
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    camera_id: str
    confidence: float = 0.0
    snapshot_url: Optional[str] = None
    timestamp: str
    bounding_box: Optional[Dict[str, int]] = None
