"""
Camera Service — CRUD operations for cameras and pipeline control.
Now uses SQLAlchemy + PostgreSQL instead of Supabase / MockStore.
"""

import logging
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import Camera
from app.core.exceptions import NotFoundException, CameraException
from app.vision.camera_worker import camera_manager

logger = logging.getLogger(__name__)


def _camera_to_dict(camera: Camera) -> dict:
    """Convert a Camera ORM object to a dict matching the API response format."""
    return {
        "id": camera.id,
        "name": camera.name,
        "location": camera.location,
        "stream_url": camera.stream_url,
        "camera_type": camera.camera_type,
        "is_active": camera.is_active,
        "config": camera.config or {},
        "created_at": camera.created_at.isoformat() if camera.created_at else None,
    }


class CameraService:
    """Service for camera management operations."""

    @staticmethod
    def list_cameras(is_active: Optional[bool] = None) -> list:
        """List all cameras with their current status."""
        db: Session = SessionLocal()
        try:
            query = db.query(Camera)
            if is_active is not None:
                query = query.filter(Camera.is_active == is_active)

            query = query.order_by(Camera.created_at.desc())
            cameras = [_camera_to_dict(c) for c in query.all()]

            # Add live status from camera_manager
            for camera in cameras:
                status = camera_manager.get_camera_status(camera["id"])
                camera["status"] = "online" if status and status.get("running") else "offline"
                if status:
                    camera["fps"] = status.get("fps", 0)
                    camera["active_tracks"] = status.get("active_tracks", {})

            return cameras
        finally:
            db.close()

    @staticmethod
    def create_camera(data: dict) -> dict:
        """Register a new camera."""
        db: Session = SessionLocal()
        try:
            camera = Camera(
                id=str(uuid4()),
                name=data["name"],
                location=data.get("location", ""),
                stream_url=data.get("stream_url", "0"),
                camera_type=data.get("camera_type", "webcam"),
                is_active=True,
                config=data.get("config", {}),
                created_at=datetime.now(timezone.utc),
            )
            db.add(camera)
            db.commit()
            db.refresh(camera)
            return _camera_to_dict(camera)
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create camera: {e}")
            raise Exception("Failed to create camera")
        finally:
            db.close()

    @staticmethod
    def get_camera(camera_id: str) -> dict:
        """Get camera details with status."""
        db: Session = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                raise NotFoundException("Camera", camera_id)

            result = _camera_to_dict(camera)
            status = camera_manager.get_camera_status(camera_id)
            result["status"] = "online" if status and status.get("running") else "offline"
            if status:
                result["fps"] = status.get("fps", 0)
                result["active_tracks"] = status.get("active_tracks", {})
            return result
        finally:
            db.close()

    @staticmethod
    def update_camera(camera_id: str, data: dict) -> dict:
        """Update camera configuration."""
        db: Session = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                raise NotFoundException("Camera", camera_id)

            update_data = {k: v for k, v in data.items() if v is not None}
            for key, value in update_data.items():
                if hasattr(camera, key):
                    setattr(camera, key, value)

            camera.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(camera)
            return _camera_to_dict(camera)
        except NotFoundException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update camera: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def delete_camera(camera_id: str) -> bool:
        """Remove a camera (stops processing if active)."""
        # Stop pipeline if running
        camera_manager.stop_camera(camera_id)

        db: Session = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if camera:
                db.delete(camera)
                db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete camera: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def start_camera(camera_id: str) -> dict:
        """Start AI processing for a camera."""
        db: Session = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                raise NotFoundException("Camera", camera_id)

            stream_url = camera.stream_url
        finally:
            db.close()

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
