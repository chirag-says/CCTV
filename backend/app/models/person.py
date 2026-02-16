"""
Pydantic models for Person and Face Encoding entities.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4


# ── Person Models ─────────────────────────────────────────────

class PersonBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Full name")
    role: str = Field(default="visitor", description="employee/visitor/vip/banned")
    department: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=255)


class PersonCreate(PersonBase):
    """Schema for creating a new person."""
    pass


class PersonUpdate(BaseModel):
    """Schema for updating a person — all fields optional."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    role: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None


class PersonResponse(PersonBase):
    """Schema returned from API."""
    id: str
    avatar_url: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    encoding_count: Optional[int] = 0

    class Config:
        from_attributes = True


class PersonWithHistory(PersonResponse):
    """Person with tracking session history."""
    sessions: List[dict] = []
    total_visits: int = 0
    last_seen: Optional[str] = None


# ── Face Encoding Models ─────────────────────────────────────

class FaceEncodingCreate(BaseModel):
    """Schema for adding a face encoding."""
    person_id: str
    source_image: Optional[str] = None
    quality: float = Field(default=1.0, ge=0.0, le=1.0)


class FaceEncodingResponse(BaseModel):
    id: str
    person_id: str
    source_image: Optional[str] = None
    quality: float
    created_at: Optional[str] = None

    class Config:
        from_attributes = True
