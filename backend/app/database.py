"""
Database module — compatibility layer.

This file now re-exports from the new db package.
Old imports like `from app.database import get_db` still work.
"""

from app.db.session import get_db, init_db, create_default_admin, SessionLocal

__all__ = ["get_db", "init_db", "create_default_admin", "SessionLocal"]
