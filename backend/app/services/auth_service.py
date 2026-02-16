"""
Authentication Service — handles admin login, token generation, and user management.
"""

import logging
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

from app.database import get_admin_db
from app.core.security import hash_password, verify_password, create_access_token
from app.core.exceptions import NotFoundException, DuplicateException
from app.config import settings

logger = logging.getLogger(__name__)


class AuthService:
    """Service for admin authentication operations."""

    @staticmethod
    def login(email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate an admin user and return a JWT token.
        
        Returns:
            Dict with access_token, token_type, expires_in, and user info
        """
        db = get_admin_db()

        if db is None:
            # Mock mode for development without Supabase
            if email == "admin@cctv.local" and password == "admin123":
                token = create_access_token({
                    "sub": "mock-admin-id",
                    "email": email,
                    "role": "superadmin",
                    "name": "Admin",
                })
                return {
                    "access_token": token,
                    "token_type": "bearer",
                    "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                    "user": {
                        "id": "mock-admin-id",
                        "email": email,
                        "name": "Admin",
                        "role": "superadmin",
                    },
                }
            raise NotFoundException("User")

        # Query admin user
        result = db.table("admin_users").select("*").eq("email", email).eq("is_active", True).execute()

        if not result.data:
            raise NotFoundException("User")

        user = result.data[0]

        # Verify password
        if not verify_password(password, user["password_hash"]):
            raise NotFoundException("User")  # Same error to prevent enumeration

        # Generate JWT
        token = create_access_token({
            "sub": user["id"],
            "email": user["email"],
            "role": user["role"],
            "name": user["name"],
        })

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "role": user["role"],
            },
        }

    @staticmethod
    def create_admin(email: str, password: str, name: str, role: str = "operator") -> dict:
        """Create a new admin user."""
        db = get_admin_db()

        if db is None:
            return {
                "id": str(uuid4()),
                "email": email,
                "name": name,
                "role": role,
                "is_active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        # Check if email exists
        existing = db.table("admin_users").select("id").eq("email", email).execute()
        if existing.data:
            raise DuplicateException("Admin user", "email")

        # Create user
        user_data = {
            "id": str(uuid4()),
            "email": email,
            "password_hash": hash_password(password),
            "name": name,
            "role": role,
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        result = db.table("admin_users").insert(user_data).execute()
        if result.data:
            user = result.data[0]
            del user["password_hash"]
            return user

        raise Exception("Failed to create admin user")
