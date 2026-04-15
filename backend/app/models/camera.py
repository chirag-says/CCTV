"""
Pydantic models for Camera entities.
"""

import re
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any


VALID_CAMERA_TYPES = {"webcam", "rtsp", "ip", "http", "usb"}

# Patterns that indicate a safe stream URL
SAFE_STREAM_PATTERNS = [
    r"^\d+$",                          # Webcam index (0, 1, 2, ...)
    r"^rtsp://",                       # RTSP stream
    r"^https?://",                     # HTTP/HTTPS stream
    r"^/dev/video\d+$",               # Linux device
]


class CameraBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Camera display name")
    location: str = Field(default="", max_length=255, description="Physical location")
    stream_url: str = Field(default="0", description="RTSP URL or webcam index")
    camera_type: str = Field(default="webcam", description="webcam/rtsp/ip/http/usb")

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        # Strip and collapse whitespace
        v = " ".join(v.strip().split())
        if not v:
            raise ValueError("Camera name cannot be empty")
        # Remove potentially dangerous characters
        v = re.sub(r'[<>&\'"\\]', '', v)
        return v

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return "0"  # Default to webcam

        # Check against safe patterns
        if any(re.match(pattern, v, re.IGNORECASE) for pattern in SAFE_STREAM_PATTERNS):
            return v

        # Block anything that looks like command injection
        dangerous_chars = [";", "|", "&", "`", "$", "(", ")", "{", "}", "\n", "\r"]
        if any(c in v for c in dangerous_chars):
            raise ValueError("Stream URL contains prohibited characters")

        return v

    @field_validator("camera_type")
    @classmethod
    def validate_camera_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_CAMERA_TYPES:
            raise ValueError(f"Invalid camera type. Must be one of: {', '.join(sorted(VALID_CAMERA_TYPES))}")
        return v

    @field_validator("location")
    @classmethod
    def sanitize_location(cls, v: str) -> str:
        v = v.strip()
        v = re.sub(r'[<>&\'"\\]', '', v)
        return v


class CameraCreate(CameraBase):
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    location: Optional[str] = None
    stream_url: Optional[str] = None
    camera_type: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, v):
        if v is None:
            return v
        v = v.strip()
        if any(re.match(pattern, v, re.IGNORECASE) for pattern in SAFE_STREAM_PATTERNS):
            return v
        dangerous_chars = [";", "|", "&", "`", "$", "(", ")", "{", "}", "\n", "\r"]
        if any(c in v for c in dangerous_chars):
            raise ValueError("Stream URL contains prohibited characters")
        return v

    @field_validator("camera_type")
    @classmethod
    def validate_camera_type(cls, v):
        if v is None:
            return v
        v = v.strip().lower()
        if v not in VALID_CAMERA_TYPES:
            raise ValueError(f"Invalid camera type. Must be one of: {', '.join(sorted(VALID_CAMERA_TYPES))}")
        return v

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v):
        if v is None:
            return v
        v = " ".join(v.strip().split())
        v = re.sub(r'[<>&\'"\\]', '', v)
        if not v:
            raise ValueError("Camera name cannot be empty")
        return v


class CameraResponse(CameraBase):
    id: str
    is_active: bool = True
    config: Dict[str, Any] = {}
    created_at: Optional[str] = None
    status: str = "offline"  # online/offline/error

    class Config:
        from_attributes = True



