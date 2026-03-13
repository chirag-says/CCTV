"""
PostgreSQL database engine and session management.

Replaces the old Supabase client with SQLAlchemy + PostgreSQL.
Supports both sync sessions (for services) and provides
auto-table creation on startup.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)

# ── Build connection URL ─────────────────────────────────────────
DATABASE_URL = (
    f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

# ── Engine ───────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # verify connections before using
    echo=settings.DB_ECHO,  # log SQL when debugging
)

# ── Session Factory ──────────────────────────────────────────────
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Session:
    """
    Dependency injection for FastAPI routes.
    Usage:
        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all tables if they don't exist.
    Called on application startup.
    """
    logger.info(f"Connecting to PostgreSQL at {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created/verified successfully")


def create_default_admin():
    """
    Create a default admin user if no admin exists.
    Only runs on first startup.
    """
    from app.db.models import AdminUser
    from app.core.security import hash_password
    from uuid import uuid4
    from datetime import datetime, timezone

    db = SessionLocal()
    try:
        existing = db.query(AdminUser).first()
        if not existing:
            admin = AdminUser(
                id=str(uuid4()),
                email="admin@cctv.local",
                password_hash=hash_password("admin123"),
                name="Admin",
                role="superadmin",
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )
            db.add(admin)
            db.commit()
            logger.info("✅ Default admin user created: admin@cctv.local / admin123")
        else:
            logger.info("ℹ️  Admin user already exists, skipping default creation")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create default admin: {e}")
    finally:
        db.close()
