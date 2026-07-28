"""
JWT token generation, verification, and password hashing utilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# ── Password Hashing ──────────────────────────────────────────

import bcrypt


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    password_bytes = password.encode("utf-8")[:72]  # bcrypt max 72 bytes
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:72],
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


# ── JWT Token ─────────────────────────────────────────────────

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Auth Dependency ───────────────────────────────────────────

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """FastAPI dependency to get the current authenticated user."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    return {
        "id": user_id,
        "email": payload.get("email", ""),
        "role": payload.get("role", "operator"),
        "name": payload.get("name", ""),
    }


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Require admin or superadmin role."""
    if current_user.get("role") not in ("admin", "superadmin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify a JWT token string directly (used by WebSocket handlers
    where standard HTTP Bearer auth isn't available).

    Raises JWTError or HTTPException on invalid/expired tokens.
    Returns the decoded payload on success.
    """
    return decode_access_token(token)


async def require_ws_token(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    """
    Authenticate a WebSocket handshake from its ?token= query parameter.

    FastAPI's HTTP Bearer dependencies do not apply to WebSocket handshakes, so
    the JWT is passed as a query parameter instead. This helper makes the check
    MANDATORY -- a missing token is rejected exactly like an invalid one.

    Previously each WS route validated the token only `if token:`, which meant
    omitting it entirely bypassed authentication. Callers must treat a None
    return as "connection already closed, return immediately".

    Returns:
        The decoded JWT payload, or None if the socket was rejected.
    """
    token = websocket.query_params.get("token")
    if not token:
        # Closed before accept() -> the handshake is refused outright.
        await websocket.close(code=4001, reason="Authentication required")
        return None
    try:
        return decode_access_token(token)
    except Exception:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return None
