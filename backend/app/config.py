"""
Application configuration using Pydantic Settings.
Loads from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # ── Supabase ──────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_KEY: str = ""

    # ── JWT Auth ──────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-this-secret-key-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── Vision Pipeline ───────────────────────────────────────
    DETECTION_MODEL: str = "hog"  # "hog" (CPU) or "cnn" (GPU)
    FACE_MATCH_TOLERANCE: float = 0.45
    FRAME_SKIP: int = 3
    DETECTION_SCALE: float = 0.5
    EXIT_THRESHOLD_SECONDS: int = 15  # 15 seconds — responsive for live monitoring
    ENTRY_THRESHOLD_SECONDS: int = 3
    UNKNOWN_SIMILARITY_THRESHOLD: float = 0.4

    # ── Camera ────────────────────────────────────────────────
    DEFAULT_CAMERA_SOURCE: str = "0"
    MAX_CAMERAS: int = 10

    # ── Server ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    # ── Storage ───────────────────────────────────────────────
    SNAPSHOT_DIR: str = "./snapshots"
    MAX_SNAPSHOT_SIZE_KB: int = 200

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

# Ensure snapshot directory exists
os.makedirs(settings.SNAPSHOT_DIR, exist_ok=True)
