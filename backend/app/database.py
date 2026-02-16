"""
Supabase database client singleton.
Provides both anon and service-role clients.
"""

from supabase import create_client, Client
from app.config import settings
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@lru_cache()
def get_supabase_client() -> Client:
    """Get Supabase client with anon key (RLS enforced)."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        logger.warning("Supabase credentials not configured. Using mock mode.")
        return None
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


@lru_cache()
def get_supabase_admin() -> Client:
    """Get Supabase client with service role key (bypasses RLS)."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        logger.warning("Supabase admin credentials not configured. Using mock mode.")
        return None
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def get_db() -> Client:
    """Dependency injection for Supabase client."""
    return get_supabase_client()


def get_admin_db() -> Client:
    """Dependency injection for admin Supabase client."""
    return get_supabase_admin()
