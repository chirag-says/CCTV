"""
Camera Service — CRUD operations for cameras and pipeline control.
Uses Supabase when available, falls back to MockStore for persistence.
"""

import logging
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

from app.database import get_admin_db
from app.core.exceptions import NotFoundException, CameraException
from app.core.mock_store import mock_store
from app.vision.camera_worker import camera_manager

logger = logging.getLogger(__name__)


class CameraService:
    """Service for camera management operations."""

    @staticmethod
    def list_cameras(is_active: Optional[bool] = None) -> list:
        """List all cameras with their current status."""
        db = get_admin_db()

        if db is None:
            # Use MockStore for persistence in dev mode
            cameras = mock_store.list_cameras(is_active=is_active)
            for camera in cameras:
                status = camera_manager.get_camera_status(camera["id"])
                camera["status"] = "online" if status and status.get("running") else "offline"
                if status:
                    camera["fps"] = status.get("fps", 0)
                    camera["active_tracks"] = status.get("active_tracks", {})
            return cameras

        query = db.table("cameras").select("*")
        if is_active is not None:
            query = query.eq("is_active", is_active)

        result = query.order("created_at", desc=True).execute()

        cameras = result.data or []
        # Add live status
        for camera in cameras:
            status = camera_manager.get_camera_status(camera["id"])
            camera["status"] = "online" if status and status["running"] else "offline"
            if status:
                camera["fps"] = status.get("fps", 0)
                camera["active_tracks"] = status.get("active_tracks", {})

        return cameras

    @staticmethod
    def create_camera(data: dict) -> dict:
        """Register a new camera."""
        db = get_admin_db()

        camera_data = {
            "id": str(uuid4()),
            "name": data["name"],
            "location": data.get("location", ""),
            "stream_url": data.get("stream_url", "0"),
            "camera_type": data.get("camera_type", "webcam"),
            "is_active": True,
            "config": data.get("config", {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if db is None:
            # Persist in MockStore so it survives across requests
            return mock_store.add_camera(camera_data)

        result = db.table("cameras").insert(camera_data).execute()
        if result.data:
            return result.data[0]

        raise Exception("Failed to create camera")

    @staticmethod
    def get_camera(camera_id: str) -> dict:
        """Get camera details with status."""
        db = get_admin_db()

        if db is None:
            camera = mock_store.get_camera(camera_id)
            if not camera:
                raise NotFoundException("Camera", camera_id)
            status = camera_manager.get_camera_status(camera_id)
            camera["status"] = "online" if status and status.get("running") else "offline"
            return camera

        result = db.table("cameras").select("*").eq("id", camera_id).execute()
        if not result.data:
            raise NotFoundException("Camera", camera_id)

        camera = result.data[0]
        status = camera_manager.get_camera_status(camera_id)
        camera["status"] = "online" if status and status["running"] else "offline"

        return camera

    @staticmethod
    def update_camera(camera_id: str, data: dict) -> dict:
        """Update camera configuration."""
        db = get_admin_db()

        if db is None:
            update_data = {k: v for k, v in data.items() if v is not None}
            result = mock_store.update_camera(camera_id, update_data)
            if not result:
                raise NotFoundException("Camera", camera_id)
            return result

        update_data = {k: v for k, v in data.items() if v is not None}

        result = (
            db.table("cameras")
            .update(update_data)
            .eq("id", camera_id)
            .execute()
        )

        if not result.data:
            raise NotFoundException("Camera", camera_id)

        return result.data[0]

    @staticmethod
    def delete_camera(camera_id: str) -> bool:
        """Remove a camera (stops processing if active)."""
        # Stop pipeline if running
        camera_manager.stop_camera(camera_id)

        db = get_admin_db()
        if db is None:
            mock_store.delete_camera(camera_id)
            return True

        result = db.table("cameras").delete().eq("id", camera_id).execute()
        return True

    @staticmethod
    def start_camera(camera_id: str) -> dict:
        """Start AI processing for a camera."""
        db = get_admin_db()

        # Get camera info
        if db is not None:
            result = db.table("cameras").select("*").eq("id", camera_id).execute()
            if not result.data:
                raise NotFoundException("Camera", camera_id)
            camera = result.data[0]
            stream_url = camera["stream_url"]
        else:
            camera = mock_store.get_camera(camera_id)
            if not camera:
                raise NotFoundException("Camera", camera_id)
            stream_url = camera.get("stream_url", "0")

        success = camera_manager.start_camera(camera_id, stream_url)
        if not success:
            raise CameraException(f"Failed to start camera {camera_id}")

        return {"status": "started", "camera_id": camera_id}

    @staticmethod
    def stop_camera(camera_id: str) -> dict:
        """Stop AI processing for a camera."""
        success = camera_manager.stop_camera(camera_id)
        if not success:
            raise NotFoundException("Camera worker", camera_id)

        return {"status": "stopped", "camera_id": camera_id}
