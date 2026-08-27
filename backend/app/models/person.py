"""
Pydantic models for Person and Face Encoding entities.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4


# ── Person Models ─────────────────────────────────────────────

# Machine-facing person classes. Attendance and presence filter on these.
PERSON_TYPES = ("faculty", "student", "staff", "visitor")

# Designations that imply person_type='faculty' when one isn't supplied
# explicitly. Kept in sync with the backfill in migration 009.
FACULTY_ROLES = (
    "professor", "associate_professor", "assistant_professor",
    "hod", "lecturer", "teacher",
)


def infer_person_type(role: Optional[str]) -> str:
    """
    Derive person_type from a designation.

    Only used as a fallback when a caller supplies `role` but not
    `person_type` — an explicit person_type always wins. This keeps older
    API clients (and the seed scripts) working without them having to know
    the new field exists.
    """
    r = (role or "").strip().lower()
    if r in FACULTY_ROLES:
        return "faculty"
    if r == "student":
        return "student"
    if r in ("employee", "staff"):
        return "staff"
    return "visitor"


class PersonBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Full name")
    role: str = Field(
        default="visitor",
        description="Designation: assistant_professor/professor/hod/student/employee/visitor",
    )
    person_type: Optional[str] = Field(
        None,
        description="faculty/student/staff/visitor — inferred from role if omitted",
    )
    department: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=255)

    @field_validator("person_type")
    @classmethod
    def _valid_person_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = v.strip().lower()
        if v not in PERSON_TYPES:
            raise ValueError(
                f"person_type must be one of {', '.join(PERSON_TYPES)}"
            )
        return v


class PersonCreate(PersonBase):
    """Schema for creating a new person."""
    pass


class PersonUpdate(BaseModel):
    """Schema for updating a person — all fields optional."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    role: Optional[str] = None
    person_type: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None

    _valid_person_type = field_validator("person_type")(
        PersonBase._valid_person_type.__func__
    )


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
