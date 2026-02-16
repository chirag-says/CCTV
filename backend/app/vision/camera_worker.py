"""
Camera Worker — Manages running vision pipelines for cameras.

Provides a central manager to start/stop pipeline workers,
handle events, and manage lifecycle.
"""

import asyncio
import threading
import time
import logging
import pickle
from typing import Dict, Optional, Callable
from uuid import uuid4

from app.vision.pipeline import VisionPipeline
from app.config import settings

logger = logging.getLogger(__name__)


class CameraWorker:
    """
    Wraps a VisionPipeline and runs it in a dedicated thread.
    """

    def __init__(
        self,
        camera_id: str,
        stream_source: str,
        on_event: Optional[Callable] = None,
        on_unknown: Optional[Callable] = None,
        on_frame: Optional[Callable] = None,
    ):
        self.camera_id = camera_id
        self.pipeline = VisionPipeline(
            camera_id=camera_id,
            stream_source=stream_source,
            on_event=on_event,
            on_frame=on_frame,
            on_unknown=on_unknown,
        )
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the camera worker in a background thread."""
        logger.info(f"Starting camera worker: {self.camera_id}")
        self.pipeline.start()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"camera-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the camera worker."""
        logger.info(f"Stopping camera worker: {self.camera_id}")
        self._stop_event.set()
        self.pipeline.stop()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run_loop(self):
        """Main processing loop running in background thread."""
        from app.utils.image_utils import resize_frame, frame_to_jpeg_bytes

        logger.info(f"Camera worker loop started: {self.camera_id}")
        target_fps = 30
        frame_interval = 1.0 / target_fps  # ~33ms per frame

        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                frame = self.pipeline.process_frame()
                if frame is not None:
                    # Pre-encode JPEG in THIS background thread (not the async loop!)
                    # This is the key optimization: JPEG encoding is CPU-heavy (~5-15ms)
                    # and must NOT run in the async event loop where it blocks everything.
                    small = resize_frame(frame, max_width=640, max_height=480)
                    self.pipeline._last_jpeg = frame_to_jpeg_bytes(small, quality=65)
            except Exception as e:
                logger.error(f"Camera {self.camera_id} error: {e}", exc_info=True)
                time.sleep(1)  # Brief pause on error
                continue

            # Throttle to target FPS for consistent frame delivery
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"Camera worker loop ended: {self.camera_id}")

    @property
    def is_running(self) -> bool:
        return self.pipeline.is_running

    @property
    def status(self) -> dict:
        return self.pipeline.status


class CameraManager:
    """
    Central manager for all camera workers.
    
    Handles:
    - Starting/stopping cameras
    - Event routing
    - Encoding distribution (loads encodings into all pipelines)
    - Status monitoring
    """

    def __init__(self):
        self._workers: Dict[str, CameraWorker] = {}
        self._event_callback: Optional[Callable] = None
        self._unknown_callback: Optional[Callable] = None
        self._frame_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        on_event: Optional[Callable] = None,
        on_unknown: Optional[Callable] = None,
        on_frame: Optional[Callable] = None,
    ):
        """Set event callbacks for all workers."""
        self._event_callback = on_event
        self._unknown_callback = on_unknown
        self._frame_callback = on_frame

    def start_camera(self, camera_id: str, stream_source: str) -> bool:
        """Start processing for a camera."""
        if camera_id in self._workers:
            if self._workers[camera_id].is_running:
                logger.warning(f"Camera {camera_id} already running")
                return False
            self._workers[camera_id].stop()

        if len(self._workers) >= settings.MAX_CAMERAS:
            logger.error(f"Max cameras ({settings.MAX_CAMERAS}) reached")
            return False

        try:
            worker = CameraWorker(
                camera_id=camera_id,
                stream_source=stream_source,
                on_event=self._event_callback,
                on_unknown=self._unknown_callback,
                on_frame=self._frame_callback,
            )
            worker.start()
            self._workers[camera_id] = worker
            return True
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
            return False

    def stop_camera(self, camera_id: str) -> bool:
        """Stop processing for a camera."""
        if camera_id not in self._workers:
            return False

        self._workers[camera_id].stop()
        del self._workers[camera_id]
        return True

    def stop_all(self):
        """Stop all camera workers."""
        for camera_id in list(self._workers.keys()):
            self.stop_camera(camera_id)

    def load_encodings_all(self, encodings_data: list):
        """Load face encodings into all active pipeline recognizers."""
        for worker in self._workers.values():
            worker.pipeline.recognizer.load_encodings(encodings_data)

        logger.info(
            f"Loaded {len(encodings_data)} encodings into "
            f"{len(self._workers)} active pipelines"
        )

    def get_camera_status(self, camera_id: str) -> Optional[dict]:
        """Get status for a specific camera."""
        if camera_id in self._workers:
            return self._workers[camera_id].status
        return None

    def get_all_status(self) -> dict:
        """Get status for all cameras."""
        return {
            "cameras": {
                cid: worker.status for cid, worker in self._workers.items()
            },
            "total_active": len(self._workers),
            "max_cameras": settings.MAX_CAMERAS,
        }

    def get_active_camera_ids(self) -> list:
        """Get IDs of all active cameras."""
        return list(self._workers.keys())

    def get_all_occupancy(self) -> int:
        """Get total occupancy across all cameras."""
        total = 0
        counted_persons = set()
        for worker in self._workers.values():
            for person in worker.pipeline.tracker.get_active_persons():
                if person["person_id"] not in counted_persons:
                    counted_persons.add(person["person_id"])
                    total += 1
        return total


# Singleton instance
camera_manager = CameraManager()
