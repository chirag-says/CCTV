"""
Pydantic models package — all request/response schemas.
"""

from app.models.person import (
    PersonBase,
    PersonCreate,
    PersonUpdate,
    PersonResponse,
    PersonWithHistory,
    FaceEncodingCreate,
    FaceEncodingResponse,
)
from app.models.camera import (
    CameraBase,
    CameraCreate,
    CameraUpdate,
    CameraResponse,
)
from app.models.event import (
    DetectionEventCreate,
    DetectionEventResponse,
    TrackingSessionResponse,
    UnknownFaceResponse,
    EnrollUnknownFace,
    LiveDetectionEvent,
)
from app.models.movement import (
    MovementCreate,
    MovementResponse,
    MovementSummary,
    MovementTimeline,
)
from app.models.auth import (
    LoginRequest,
    TokenResponse,
    AdminUserCreate,
    AdminUserResponse,
)

__all__ = [
    # Person
    "PersonBase", "PersonCreate", "PersonUpdate",
    "PersonResponse", "PersonWithHistory",
    "FaceEncodingCreate", "FaceEncodingResponse",
    # Camera
    "CameraBase", "CameraCreate", "CameraUpdate", "CameraResponse",
    # Event
    "DetectionEventCreate", "DetectionEventResponse",
    "TrackingSessionResponse", "UnknownFaceResponse",
    "EnrollUnknownFace", "LiveDetectionEvent",
    # Movement
    "MovementCreate", "MovementResponse",
    "MovementSummary", "MovementTimeline",
    # Auth
    "LoginRequest", "TokenResponse",
    "AdminUserCreate", "AdminUserResponse",
]
