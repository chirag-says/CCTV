"""
Camera Worker — Manages running vision pipelines for cameras.

Provides a central manager to start/stop pipeline workers,
handle events, and manage lifecycle.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, Optional, Callable
from uuid import uuid4

from app.vision.pipeline import VisionPipeline
from app.config import settings

logger = logging.getLogger(__name__)


class CameraWorker:
    """
    Wraps a VisionPipeline and runs it in a dedicated thread.

    Architecture (post-optimization):
      Main thread:  capture → detect → track → recognize → [hand off frame]
      JPEG thread:  [receive frame] → resize → encode → write _last_jpeg

    The main loop no longer blocks on JPEG encoding (~28ms saved per frame).
    A single-slot latest-frame-wins design ensures the encoder always works
    on the most recent frame and never builds up a backlog.
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

        # ── JPEG Encoder Thread State ─────────────────────────────
        # Single-slot handoff: main loop writes the latest frame here,
        # encoder thread reads and clears it. No FIFO queue — if the
        # encoder falls behind, it skips to the newest frame.
        self._jpeg_slot_frame = None       # Latest frame awaiting encode
        self._jpeg_slot_lock = threading.Lock()
        self._jpeg_slot_event = threading.Event()  # Signals encoder that a frame is ready
        self._jpeg_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the camera worker and its JPEG encoder thread."""
        logger.info(f"Starting camera worker: {self.camera_id}")
        self.pipeline.start()
        self._stop_event.clear()

        # Start the JPEG encoder thread BEFORE the main loop
        self._jpeg_slot_event.clear()
        self._jpeg_thread = threading.Thread(
            target=self._jpeg_encoder_loop,
            name=f"jpeg-{self.camera_id}",
            daemon=True,
        )
        self._jpeg_thread.start()

        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"camera-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the camera worker and JPEG encoder thread."""
        logger.info(f"Stopping camera worker: {self.camera_id}")
        self._stop_event.set()
        # Wake the JPEG thread so it can exit
        self._jpeg_slot_event.set()
        self.pipeline.stop()

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._jpeg_thread:
            self._jpeg_thread.join(timeout=3)
            self._jpeg_thread = None

    def _run_loop(self):
        """Main processing loop — detection, tracking, recognition.

        JPEG encoding has been moved to _jpeg_encoder_loop() so this
        loop is no longer blocked by imencode(). Frames are handed off
        via a single-slot exchange (~0.01ms instead of ~28ms).
        """
        from app.core.profiler import profiler

        logger.info(f"Camera worker loop started: {self.camera_id}")
        target_fps = 30
        frame_interval = 1.0 / target_fps  # ~33ms per frame

        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                with profiler.measure("frame_total"):
                    profiler.record_frame_received()
                    frame = self.pipeline.process_frame()
                    if frame is not None:
                        # Hand frame to JPEG encoder thread (non-blocking).
                        # Uses latest-frame-wins: overwrites any pending frame
                        # so the encoder always works on the newest data.
                        with self._jpeg_slot_lock:
                            self._jpeg_slot_frame = frame
                        self._jpeg_slot_event.set()  # Wake encoder
                    else:
                        profiler.record_frame_dropped()

                # Tick profiler for every frame (report every N total frames)
                profiler.tick_frame()

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

    def _jpeg_encoder_loop(self):
        """Background thread: encodes frames to JPEG for WebSocket streaming.

        Runs independently of the main processing loop. Picks up the latest
        frame from the single-slot exchange, resizes it, encodes to JPEG,
        and writes the result to pipeline._last_jpeg.

        This thread is the ONLY writer of _last_jpeg after startup.
        """
        from app.utils.image_utils import frame_to_jpeg_bytes
        from app.core.profiler import profiler

        logger.info(f"JPEG encoder thread started: {self.camera_id}")

        while not self._stop_event.is_set():
            # Wait for a frame to be available (with timeout to check stop)
            signaled = self._jpeg_slot_event.wait(timeout=0.1)
            if not signaled:
                continue  # Timeout — check stop flag and retry

            # Grab the latest frame (clear the slot)
            with self._jpeg_slot_lock:
                frame = self._jpeg_slot_frame
                self._jpeg_slot_frame = None
            self._jpeg_slot_event.clear()

            if frame is None:
                continue

            try:
                with profiler.measure("jpeg_encode"):
                    # Pipeline now returns a pre-resized 1080p display frame.
                    # No resize needed here — frame is already at streaming resolution.
                    # Record raw frame size for compression ratio
                    profiler.record_raw_frame_size(frame.nbytes)
                    # Use load-tier quality override if pipeline has one
                    jpeg_q = getattr(self.pipeline, '_jpeg_quality_override', None) or 80
                    jpeg_bytes = frame_to_jpeg_bytes(frame, quality=jpeg_q)
                    self.pipeline._last_jpeg = jpeg_bytes

                # Record WebSocket frame size for stats
                if jpeg_bytes:
                    profiler.record_ws_send(len(jpeg_bytes), 0.0)

            except Exception as e:
                logger.debug(f"JPEG encode error ({self.camera_id}): {e}")

        logger.info(f"JPEG encoder thread stopped: {self.camera_id}")

    @property
    def is_running(self) -> bool:
        return self.pipeline.is_running

    @property
    def status(self) -> dict:
        return self.pipeline.status

    @property
    def stream_health(self):
        """Proxy to pipeline's stream health monitor for WebSocket/status access."""
        return self.pipeline.stream_health


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
        self._cached_encodings: Optional[list] = None  # Cached for new cameras

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

            # Load cached face encodings into the new pipeline immediately
            # This is critical: without this, the recognizer starts with an
            # empty cache and every face is detected as "UNKNOWN".
            if self._cached_encodings:
                worker.pipeline.recognizer.load_encodings(self._cached_encodings)
                logger.info(
                    f"Auto-loaded {len(self._cached_encodings)} encodings "
                    f"into new camera {camera_id}"
                )
            else:
                # No cached encodings — try loading from DB now
                try:
                    from app.services.person_service import PersonService
                    encodings = PersonService.get_all_encodings()
                    if encodings:
                        self._cached_encodings = encodings
                        worker.pipeline.recognizer.load_encodings(encodings)
                        logger.info(
                            f"Loaded {len(encodings)} encodings from DB "
                            f"into new camera {camera_id}"
                        )
                except Exception as enc_err:
                    logger.warning(f"Could not load encodings for camera {camera_id}: {enc_err}")

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
        """Load face encodings into all active pipeline recognizers and cache for future cameras."""
        # Cache encodings so newly started cameras automatically get them
        self._cached_encodings = encodings_data

        for worker in self._workers.values():
            worker.pipeline.recognizer.load_encodings(encodings_data)

        logger.info(
            f"Loaded {len(encodings_data)} encodings into "
            f"{len(self._workers)} active pipelines (cached for future cameras)"
        )

    def refresh_encodings(self):
        """
        Reload face encodings from the database and distribute to all active pipelines.
        Call this after adding/removing face encodings to ensure all cameras are in sync.
        """
        try:
            from app.services.person_service import PersonService
            encodings = PersonService.get_all_encodings()
            self.load_encodings_all(encodings)
            logger.info(f"Refreshed {len(encodings)} encodings from database")
            return len(encodings)
        except Exception as e:
            logger.error(f"Failed to refresh encodings: {e}")
            return -1

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
