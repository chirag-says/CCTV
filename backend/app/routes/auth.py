"""
Authentication API Routes.
"""

from fastapi import APIRouter, Depends
from app.models.auth import LoginRequest, TokenResponse, AdminUserCreate, AdminUserResponse
from app.services.auth_service import AuthService
from app.core.security import get_current_user, require_admin

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate admin user and return JWT token."""
    result = AuthService.login(request.email, request.password)
    return result


@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    return current_user


@router.post("/register", response_model=AdminUserResponse)
async def register_admin(
    request: AdminUserCreate,
    current_user: dict = Depends(require_admin),
):
    """Register a new admin user (requires admin access)."""
    result = AuthService.create_admin(
        email=request.email,
        password=request.password,
        name=request.name,
        role=request.role,
    )
    return result


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout (client should discard token)."""
    return {"message": "Logged out successfully"}
