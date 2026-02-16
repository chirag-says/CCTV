"""
Pydantic models for Camera entities.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class CameraBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Camera display name")
    location: str = Field(default="", max_length=255, description="Physical location")
    stream_url: str = Field(default="0", description="RTSP URL or webcam index")
    camera_type: str = Field(default="webcam", description="webcam/rtsp/ip")


class CameraCreate(CameraBase):
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    location: Optional[str] = None
    stream_url: Optional[str] = None
    camera_type: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


class CameraResponse(CameraBase):
    id: str
    is_active: bool = True
    config: Dict[str, Any] = {}
    created_at: Optional[str] = None
    status: str = "offline"  # online/offline/error

    class Config:
        from_attributes = True
