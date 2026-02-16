"""
Pydantic models for Authentication.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class LoginRequest(BaseModel):
    email: str = Field(..., description="Admin email")
    password: str = Field(..., min_length=6, description="Admin password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class AdminUserCreate(BaseModel):
    email: str = Field(...)
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(default="operator", description="superadmin/admin/operator")


class AdminUserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    is_active: bool = True
    created_at: Optional[str] = None

    class Config:
        from_attributes = True
