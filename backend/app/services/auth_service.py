"""
Authentication Service — handles admin login, token generation, and user management.
Now uses SQLAlchemy + PostgreSQL instead of Supabase.
"""

import logging
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import AdminUser
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
        db: Session = SessionLocal()
        try:
            user = (
                db.query(AdminUser)
                .filter(AdminUser.email == email, AdminUser.is_active == True)
                .first()
            )

            if not user:
                raise NotFoundException("User")

            # Verify password
            if not verify_password(password, user.password_hash):
                raise NotFoundException("User")  # Same error to prevent enumeration

            # Generate JWT
            token = create_access_token({
                "sub": user.id,
                "email": user.email,
                "role": user.role,
                "name": user.name,
            })

            return {
                "access_token": token,
                "token_type": "bearer",
                "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "name": user.name,
                    "role": user.role,
                },
            }
        finally:
            db.close()

    @staticmethod
    def create_admin(email: str, password: str, name: str, role: str = "operator") -> dict:
        """Create a new admin user."""
        db: Session = SessionLocal()
        try:
            # Check if email exists
            existing = db.query(AdminUser).filter(AdminUser.email == email).first()
            if existing:
                raise DuplicateException("Admin user", "email")

            # Create user
            user = AdminUser(
                id=str(uuid4()),
                email=email,
                password_hash=hash_password(password),
                name=name,
                role=role,
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            return {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat() if user.created_at else None,
            }
        except (NotFoundException, DuplicateException):
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create admin user: {e}")
            raise Exception("Failed to create admin user")
        finally:
            db.close()

    @staticmethod
    def refresh_token(current_user: dict) -> Dict[str, Any]:
        """
        Issue a new JWT token for an already-authenticated user.
        This implements sliding token expiry.
        """
        token = create_access_token({
            "sub": current_user.get("id", current_user.get("sub", "")),
            "email": current_user.get("email", ""),
            "role": current_user.get("role", "operator"),
            "name": current_user.get("name", ""),
        })

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

