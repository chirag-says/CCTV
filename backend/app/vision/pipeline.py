"""
Main Vision Processing Pipeline.

Orchestrates frame capture → detection → recognition → tracking → event emission.
Runs as a background worker per camera.
"""

import cv2
import numpy as np
import pickle
import time
import asyncio
import logging
import os
from typing import Optional, Dict, List, Callable
from datetime import datetime, timezone
from uuid import uuid4

from app.config import settings
from app.vision.detector import FaceDetector
from app.vision.recognizer import FaceRecognizer, MatchResult
from app.vision.tracker import PersonTracker
from app.utils.image_utils import save_snapshot, frame_to_jpeg_bytes

logger = logging.getLogger(__name__)


class VisionPipeline:
    """
    Main processing pipeline for a single camera stream.
    
    Flow:
    1. Capture frame from camera
    2. Skip frames for performance
    3. Detect faces
    4. Generate encodings
    5. Recognize faces (match vs known)
    6. Track entry/exit
    7. Handle unknown faces
    8. Emit events via callbacks
    """

    def __init__(
        self,
        camera_id: str,
        stream_source: str = "0",
        on_event: Optional[Callable] = None,
        on_frame: Optional[Callable] = None,
        on_unknown: Optional[Callable] = None,
    ):
        self.camera_id = camera_id
        self.stream_source = stream_source
        self._running = False
        self._capture: Optional[cv2.VideoCapture] = None

        # Callbacks
        self.on_event = on_event  # Called on entry/exit/detection events
        self.on_frame = on_frame  # Called with processed frame (for streaming)
        self.on_unknown = on_unknown  # Called when unknown face detected

        # Sub-modules
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.tracker = PersonTracker(
            on_entry=self._handle_entry_event,
            on_exit=self._handle_exit_event,
        )

        # Pipeline state
        self._frame_count = 0
        self._last_frame = None  # Last processed frame (for MJPEG streaming)
        self._last_jpeg = None   # Pre-encoded JPEG bytes (ready-to-send, encoded in worker thread)
        self._last_exit_check = time.time()
        self._exit_check_interval = 2  # Check exits every 2 seconds for faster response
        self._unknown_cache: Dict[str, dict] = {}  # Temp unknown face dedup

        # Performance metrics
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

        logger.info(f"Pipeline created for camera {camera_id}: source={stream_source}")

    def start(self):
        """Start the video capture."""
        try:
            # Try to interpret as integer (webcam index)
            source = int(self.stream_source) if self.stream_source.isdigit() else self.stream_source
            self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                raise RuntimeError(f"Cannot open camera source: {self.stream_source}")

            # Optimize capture settings
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture.set(cv2.CAP_PROP_FPS, 30)

            self._running = True
            logger.info(f"Camera {self.camera_id} started: {self.stream_source}")

        except Exception as e:
            logger.error(f"Failed to start camera {self.camera_id}: {e}")
            raise

    def stop(self):
        """Stop the video capture and clean up."""
        self._running = False
        if self._capture:
            self._capture.release()
            self._capture = None
        # Flush remaining exits
        self.tracker.check_exits()
        logger.info(f"Camera {self.camera_id} stopped")

    def process_frame(self) -> Optional[np.ndarray]:
        """
        Process a single frame through the full pipeline.
        
        Returns:
            Annotated frame (with bounding boxes) or None if no frame available
        """
        if not self._running or not self._capture:
            return None

        ret, frame = self._capture.read()
        if not ret:
            logger.warning(f"Camera {self.camera_id}: Failed to read frame")
            return None

        self._frame_count += 1
        
        # Keep a clean copy for snapshots (before drawing boxes)
        clean_frame = frame.copy()

        # Frame skip for performance
        if self._frame_count % settings.FRAME_SKIP != 0:
            self._last_frame = frame
            return frame  # Return unprocessed frame for display

        # ── DETECTION ─────────────────────────────────────────
        face_locations, rgb_frame = self.detector.detect_faces(frame)

        if not face_locations:
            self._check_periodic_exits()
            self._update_fps()
            self._last_frame = frame
            return frame

        logger.info(f"Camera {self.camera_id}: Detected {len(face_locations)} face(s)")

        # ── ENCODING ──────────────────────────────────────────
        encodings = self.recognizer.batch_generate_encodings(rgb_frame, face_locations)

        # ── RECOGNITION & TRACKING ────────────────────────────
        for i, (location, encoding) in enumerate(zip(face_locations, encodings)):
            if encoding is None:
                continue

            top, right, bottom, left = location

            # Try to recognize
            result: MatchResult = self.recognizer.recognize_face(encoding)

            if result.matched:
                # ── KNOWN PERSON ──────────────────────────────
                self._draw_box(frame, location, result.person_name, (0, 255, 0), result.confidence)

                # Feed to tracker (handles entry/exit state transitions)
                self.tracker.on_detection(
                    person_id=result.person_id,
                    person_name=result.person_name,
                    camera_id=self.camera_id,
                    confidence=result.confidence,
                )

                # NOTE: We intentionally do NOT emit a "detection" event
                # every frame. The tracker already emits entry/exit events
                # on state changes. Broadcasting per-frame detections would
                # flood the WebSocket (10+ events/sec/person × 100s of people).
            else:
                # ── UNKNOWN PERSON ────────────────────────────
                self._draw_box(frame, location, "UNKNOWN", (0, 0, 255), 0.0)
                # Pass clean_frame to ensure snapshot doesn't have the red box
                self._handle_unknown_face(clean_frame, rgb_frame, location, encoding)

        # ── PERIODIC EXIT CHECK ───────────────────────────────
        self._check_periodic_exits()
        self._update_fps()

        self._last_frame = frame
        return frame

    def _handle_unknown_face(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray,
        location: tuple,
        encoding: np.ndarray,
    ):
        """Handle an unrecognized face — deduplicate and queue."""
        # Check if this unknown is similar to any recently seen unknowns
        similar_id = None
        for uid, udata in self._unknown_cache.items():
            distance = self.recognizer.compare_unknown_faces(
                encoding, udata["encoding"]
            )
            if distance < settings.UNKNOWN_SIMILARITY_THRESHOLD:
                similar_id = uid
                break

        if similar_id:
            # Update existing unknown
            self._unknown_cache[similar_id]["count"] += 1
            self._unknown_cache[similar_id]["last_seen"] = time.time()
        else:
            # New unknown face
            unknown_id = str(uuid4())

            # ── High-quality face crop with generous padding ──
            face_crop = self.detector.extract_face_crop(frame, location, padding=80)
            _, snapshot_url = save_snapshot(
                face_crop, f"unknown_{unknown_id}", subdirectory="unknowns", quality=95
            )

            # ── Wider context crop (head + shoulders) for better identification ──
            context_crop = self._extract_context_crop(frame, location)
            _, context_url = save_snapshot(
                context_crop, f"context_{unknown_id}", subdirectory="unknowns", quality=95
            )

            # ── Full frame at good quality ──
            _, full_frame_url = save_snapshot(
                frame, f"fullframe_{unknown_id}", subdirectory="frames", quality=90
            )

            self._unknown_cache[unknown_id] = {
                "encoding": encoding,
                "count": 1,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "snapshot_path": snapshot_url,
                "context_path": context_url,
                "full_frame_path": full_frame_url,
            }

            # Emit unknown face event
            if self.on_unknown:
                self.on_unknown({
                    "unknown_id": unknown_id,
                    "camera_id": self.camera_id,
                    "snapshot_path": snapshot_url,
                    "context_path": context_url,
                    "full_frame_path": full_frame_url,
                    "encoding": encoding,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "bounding_box": {
                        "top": location[0], "right": location[1],
                        "bottom": location[2], "left": location[3]
                    },
                })

    def _handle_entry_event(self, event: dict):
        """Callback when tracker confirms an entry."""
        if self.on_event:
            self.on_event(event)

    def _handle_exit_event(self, event: dict):
        """Callback when tracker detects an exit."""
        if self.on_event:
            self.on_event(event)

    def _check_periodic_exits(self):
        """Periodically check for exits."""
        now = time.time()
        if now - self._last_exit_check >= self._exit_check_interval:
            self.tracker.check_exits(now)
            self._last_exit_check = now

    def _draw_box(
        self,
        frame: np.ndarray,
        location: tuple,
        label: str,
        color: tuple,
        confidence: float = 0.0,
    ):
        """Draw a bounding box and label on the frame."""
        top, right, bottom, left = location

        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw label background
        label_text = f"{label}"
        if confidence > 0:
            label_text += f" ({confidence:.0%})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        cv2.rectangle(
            frame,
            (left, top - text_h - 10),
            (left + text_w + 10, top),
            color, cv2.FILLED,
        )
        cv2.putText(
            frame, label_text,
            (left + 5, top - 5),
            font, font_scale, (255, 255, 255), thickness,
        )

    def _extract_context_crop(
        self,
        frame: np.ndarray,
        location: tuple,
    ) -> np.ndarray:
        """
        Extract a wider context crop (head + shoulders) from the frame.
        This gives much better snapshots for unknown person identification.
        """
        top, right, bottom, left = location
        h, w = frame.shape[:2]

        face_w = right - left
        face_h = bottom - top

        # Expand to ~3x face width and ~4x face height (captures shoulders)
        cx = (left + right) // 2
        cy = (top + bottom) // 2

        crop_w = int(face_w * 3)
        crop_h = int(face_h * 4)

        crop_left = max(0, cx - crop_w // 2)
        crop_right = min(w, cx + crop_w // 2)
        crop_top = max(0, top - int(face_h * 0.8))  # Some space above head
        crop_bottom = min(h, crop_top + crop_h)

        return frame[crop_top:crop_bottom, crop_left:crop_right].copy()

    def _save_snapshot(self, image: np.ndarray, prefix: str) -> str:
        """Save a snapshot image to disk. Returns URL-accessible path."""
        _, url_path = save_snapshot(image, prefix)
        return url_path

    def _update_fps(self):
        """Calculate processing FPS."""
        self._fps_frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = now

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "running": self._running,
            "fps": self.fps,
            "frames_processed": self._frame_count,
            "faces_in_cache": self.recognizer.known_count,
            "active_tracks": self.tracker.stats,
            "unknown_cache_size": len(self._unknown_cache),
        }
