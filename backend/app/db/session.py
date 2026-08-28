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


def verify_schema() -> dict:
    """
    Report model columns that the live database does not have.

    WHY THIS EXISTS
    There is no Alembic here. `Base.metadata.create_all()` creates missing
    TABLES but never adds a COLUMN to a table that already exists, so every
    migration from 002 onward has to be applied by hand. When one is missed,
    the model still lists the column, SQLAlchemy still puts it in every
    INSERT, and Postgres rejects the statement.

    For `detection_events` that is not a degraded mode -- it is total,
    silent loss of the event log, discovered whenever somebody next looks at
    a dashboard. A loud line at boot naming the missing column and the file
    that adds it is worth far more than the failure it replaces.

    Deliberately does NOT raise. A schema drift in one table should not stop
    the cameras; it should be impossible to miss in the logs.

    Returns {table: [missing columns]}, empty when the schema is current.
    """
    from sqlalchemy import inspect as sa_inspect

    missing: dict = {}
    try:
        inspector = sa_inspect(engine)
        present_tables = set(inspector.get_table_names())
        for table_name, table in Base.metadata.tables.items():
            if table_name not in present_tables:
                continue  # create_all() will make it
            actual = {c["name"] for c in inspector.get_columns(table_name)}
            gap = [c.name for c in table.columns if c.name not in actual]
            if gap:
                missing[table_name] = gap
    except Exception as e:
        logger.warning(f"Schema verification skipped: {e}")
        return {}

    if missing:
        logger.critical("=" * 70)
        logger.critical("DATABASE SCHEMA IS BEHIND THE CODE — a migration was not applied.")
        for table_name, cols in sorted(missing.items()):
            logger.critical(f"  {table_name}: missing {', '.join(sorted(cols))}")
        logger.critical("")
        logger.critical("  Writes to these tables WILL FAIL until this is fixed.")
        logger.critical("  Apply the pending migrations from backend/, newest last:")
        logger.critical("    python apply_migration.py app/migrations/<NNN>_<name>.sql")
        logger.critical("  (list them with --list, re-check with --check)")
        logger.critical("=" * 70)

    return missing


def init_db():
    """
    Create all tables if they don't exist.
    Called on application startup.
    """
    logger.info(f"Connecting to PostgreSQL at {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created/verified successfully")
    verify_schema()


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
