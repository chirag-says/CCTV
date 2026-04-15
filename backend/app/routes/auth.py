"""
Authentication API Routes.
"""

from fastapi import APIRouter, Depends, Request
from app.models.auth import LoginRequest, TokenResponse, AdminUserCreate, AdminUserResponse
from app.services.auth_service import AuthService
from app.core.security import get_current_user, require_admin

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, body: LoginRequest):
    """Authenticate admin user and return JWT token. Rate limited to 5 attempts/minute."""
    result = AuthService.login(body.email, body.password)
    return result


@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    return current_user


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """Refresh JWT token (returns a new token if current one is still valid)."""
    result = AuthService.refresh_token(current_user)
    return result


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
