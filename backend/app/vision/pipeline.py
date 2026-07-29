"""
Main Vision Processing Pipeline.

Orchestrates frame capture Ã¢â€ â€™ detection Ã¢â€ â€™ recognition Ã¢â€ â€™ tracking Ã¢â€ â€™ event emission.

Supports two modes:
  FACE mode (default):
    - Face Detection Ã¢â€ â€™ Recognition Ã¢â€ â€™ Tracking Ã¢â€ â€™ Entry/Exit
    - Attribute Recognition (gender + clothing color)
    - Crowd / Gathering Detection
    - Loitering / Idle Detection
    - Hazard Detection (YOLOv8)

  TRAFFIC mode:
    - Vehicle Detection + License Plate Recognition (ANPR)
    - Human-Vehicle proximity safety alerts

Runs as a background worker per camera.
"""

# Ã¢â€â‚¬Ã¢â€â‚¬ Standard Library Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
import asyncio
import logging
import threading
import os
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

# Ã¢â€â‚¬Ã¢â€â‚¬ Force CUDA to use NVIDIA GPU (must be before any torch import) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Ã¢â€â‚¬Ã¢â€â‚¬ Third-Party Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
import cv2
import numpy as np

# Ã¢â€â‚¬Ã¢â€â‚¬ Internal: Config Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
from app.config import settings

# Ã¢â€â‚¬Ã¢â€â‚¬ Internal: Vision Core Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
from app.vision.detector import FaceDetector
from app.vision.recognizer import FaceRecognizer, MatchResult
from app.vision.tracker import PersonTracker
from app.vision.bytetrack import ByteTracker, STrack, VehicleByteTracker, VehicleSTrack
from app.vision.identity import IdentityLedger

# Ã¢â€â‚¬Ã¢â€â‚¬ Internal: Safety & Security Analytics Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
from app.vision.safety_analytics import (
    AttributeRecognizer,
    CrowdDetector,
    LoiteringDetector,
)
from app.vision.hazard_detector import HazardDetector

# Ã¢â€â‚¬Ã¢â€â‚¬ Internal: Traffic & ANPR Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
from app.vision.anpr import PlateRecognizer
from app.vision.traffic import TrafficMonitor
from app.vision.reid import get_reid_tracker, ReIDTracker

# Ã¢â€â‚¬Ã¢â€â‚¬ Internal: Utilities Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
from app.utils.image_utils import frame_to_jpeg_bytes, save_snapshot
from app.core.profiler import profiler

logger = logging.getLogger(__name__)

# Ã¢â€â‚¬Ã¢â€â‚¬ Log GPU status on module load Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
try:
    import torch
    _cuda_available = torch.cuda.is_available()
    _gpu_name = torch.cuda.get_device_name(0) if _cuda_available else "N/A"
    logger.info(f"Ã°Å¸â€“Â¥Ã¯Â¸Â  GPU Status: CUDA={'available' if _cuda_available else 'NOT available'}, Device={_gpu_name}")
except ImportError:
    logger.info("Ã°Å¸â€“Â¥Ã¯Â¸Â  GPU Status: PyTorch not installed, GPU acceleration disabled")


class VisionPipeline:
    """
    Main processing pipeline for a single camera stream.
    
    Supports FACE mode and TRAFFIC mode (see module docstring).
    Mode is controlled by settings.PIPELINE_MODE.
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

        # Sub-modules (SCRFD + ArcFace + ByteTrack)
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.tracker = PersonTracker(
            on_entry=self._handle_entry_event,
            on_exit=self._handle_exit_event,
        )

        # ByteTrack frame-to-frame face tracker
        # Maintains track IDs across frames, reducing the need to run
        # expensive ArcFace recognition on every single frame.
        self.byte_tracker = ByteTracker()

        # Identity layer, sitting ABOVE the tracker. ByteTrack stays
        # geometry-only; this owns the track_id -> identity binding.
        # Milestone 1: observe-only (measures recovery headroom, assigns nothing).
        self.identity_ledger = IdentityLedger(
            retire_frames=settings.IDENTITY_RETIRE_FRAMES,
            max_entries=settings.IDENTITY_LEDGER_MAX_ENTRIES,
            max_centroid_distance=settings.IDENTITY_MAX_RECOVERY_DISTANCE_PX,
        )
        # Tracks whose identity was recovered AND verified this frame. Used to
        # suppress the `tracklet_len < 3` forced re-recognition, which would
        # otherwise immediately redo the match we just made. Pruned every
        # detection frame against the live track set, so it stays bounded by
        # the number of concurrent faces.
        self._recovered_track_ids: set = set()

        # Recognition cache: time-based re-check intervals (FPS-independent)
        self._recognition_lock_confidence = settings.RECOGNITION_LOCK_CONFIDENCE
        self._recognition_recheck_sec = settings.RECOGNITION_RECHECK_SEC
        self._recognition_full_recheck_sec = settings.RECOGNITION_FULL_RECHECK_SEC

        # Ã¢â€â‚¬Ã¢â€â‚¬ YOLO Person Detector (for body-level person detection) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # Used in FACE mode to assist face detection when face_recognition
        # misses faces (e.g. partial occlusion, small faces, angled faces).
        # In HYBRID mode, we reuse the TrafficMonitor's YOLO detections.
        self._yolo_person_detector = None
        self._yolo_person_loaded = False
        self._yolo_person_lock = threading.Lock()
        self._yolo_body_unknown_cache: Dict[str, dict] = {}  # Dedup for body-only unknowns
        self._last_yolo_assist_check = 0.0  # Throttle: last time we ran YOLO-assist
        self._yolo_assist_min_interval = 1.0  # Minimum interval between YOLO-Assist runs

        # ── Load-tier degradation state ──────────────────────────
        self._active_face_count = 0         # Updated each detection frame
        self._jpeg_quality_override = None  # None = use default
        self._frame_skip_override = None    # None = use default
        self._skip_low_priority = False     # Skip P3 work (attributes, suggestions)

        # ── Safety & Security Modules ────────────────────────────────────────────────────────
        # Attribute Recognition (gender + clothing)
        self.attribute_recognizer: Optional[AttributeRecognizer] = None
        if settings.ENABLE_ATTRIBUTE_RECOGNITION:
            self.attribute_recognizer = AttributeRecognizer(max_workers=2)
            logger.info(f"[{camera_id}] Attribute Recognition enabled")

        # Crowd / Gathering Detection
        self.crowd_detector: Optional[CrowdDetector] = None
        if settings.ENABLE_CROWD_DETECTION:
            self.crowd_detector = CrowdDetector(
                proximity_px=settings.CROWD_PROXIMITY_PX,
                min_persons=settings.CROWD_MIN_PERSONS,
                sustain_seconds=settings.CROWD_SUSTAIN_SECONDS,
                on_alert=self._handle_security_alert,
            )
            logger.info(f"[{camera_id}] Crowd Detection enabled")

        # Loitering / Idle Detection
        self.loitering_detector: Optional[LoiteringDetector] = None
        if settings.ENABLE_LOITERING_DETECTION:
            self.loitering_detector = LoiteringDetector(
                movement_threshold_px=settings.LOITER_MOVEMENT_THRESHOLD_PX,
                time_window_sec=settings.LOITER_TIME_WINDOW_SEC,
                alert_cooldown_sec=settings.LOITER_ALERT_COOLDOWN_SEC,
                on_alert=self._handle_security_alert,
            )
            logger.info(f"[{camera_id}] Loitering Detection enabled")

        # Hazard Detection (YOLOv8)
        self.hazard_detector: Optional[HazardDetector] = None
        if settings.ENABLE_HAZARD_DETECTION:
            self.hazard_detector = HazardDetector(
                model_path=settings.HAZARD_MODEL_PATH,
                frame_interval=settings.HAZARD_FRAME_INTERVAL,
                alert_cooldown_sec=settings.HAZARD_ALERT_COOLDOWN_SEC,
                on_alert=self._handle_security_alert,
            )
            logger.info(f"[{camera_id}] Hazard Detection enabled")

        # ── Traffic Mode Modules ─────────────────────────────────────────────────────────────
        self.pipeline_mode = settings.PIPELINE_MODE.lower()

        # ANPR (License Plate Recognition)
        self._is_browser_camera = stream_source.startswith("browser:")
        self.plate_recognizer: Optional[PlateRecognizer] = None
        if self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_ANPR:
            languages = [l.strip() for l in settings.ANPR_OCR_LANGUAGES.split(",")]
            self.plate_recognizer = PlateRecognizer(
                yolo_model_path=settings.ANPR_MODEL_PATH,
                frame_interval=settings.ANPR_FRAME_INTERVAL,
                plate_cooldown_sec=settings.ANPR_PLATE_COOLDOWN_SEC,
                on_plate=self._handle_vehicle_event,
                languages=languages,
            )
            logger.info(f"[{camera_id}] ANPR enabled (languages={languages})")

        # Traffic Monitor (Vehicle-Person Safety)
        self.traffic_monitor: Optional[TrafficMonitor] = None
        if self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_TRAFFIC_MONITOR:
            self.traffic_monitor = TrafficMonitor(
                model_path=settings.TRAFFIC_MODEL_PATH,
                frame_interval=settings.TRAFFIC_FRAME_INTERVAL,
                proximity_px=settings.TRAFFIC_PROXIMITY_PX,
                alert_cooldown_sec=settings.TRAFFIC_ALERT_COOLDOWN_SEC,
                on_alert=self._handle_security_alert,
                on_vehicle_count=self._handle_vehicle_count,
            )
            logger.info(f"[{camera_id}] Traffic Monitor enabled (proximity={settings.TRAFFIC_PROXIMITY_PX}px)")

        # ── Person Re-Identification (cross-camera tracking) ──────────────────────────────────
        self.reid_tracker: Optional[ReIDTracker] = None
        if settings.ENABLE_REID:
            self.reid_tracker = get_reid_tracker()
            self._reid_frame_interval = settings.REID_FRAME_INTERVAL
            self._last_reid_frame = 0
            logger.info(f"[{camera_id}] ReID enabled (threshold={settings.REID_MATCH_THRESHOLD}, interval={self._reid_frame_interval})")

        # Pipeline state
        self._frame_count = 0
        self._last_frame = None  # Last processed frame (for MJPEG streaming)
        self._last_jpeg = None   # Pre-encoded JPEG bytes (ready-to-send, encoded in worker thread)
        self._last_exit_check = time.time()
        self._exit_check_interval = 5  # Check exits every 5 seconds (prevents premature exits)
        self._unknown_cache: Dict[str, dict] = {}  # Temp unknown face dedup
        self._last_safety_check = time.time()
        self._safety_check_interval = 1.0  # Run crowd/loiter checks every 1s
        self._last_face_draws: List[Tuple] = [] # Cache for drawing face boxes on skipped frames

        # ── Streaming Resolution ──────────────────────────────────────────────
        # AI detection runs on full 4MP for accuracy.
        # Drawing + JPEG encoding happen on a single 1080p resize.
        # This saves ~13ms/frame (no redundant resize in JPEG thread)
        # and reduces per-frame memory from 11.7MB to 6.2MB.
        self._stream_max_w = 1920
        self._stream_max_h = 1080
        self._stream_scale_x = 1.0  # Updated when first frame arrives
        self._stream_scale_y = 1.0
        self._stream_res_initialized = False

        # ── Threaded Frame Grabber ────────────────────────────────────────────
        # Solves the OpenCV buffer-lag problem: cv2.VideoCapture buffers frames
        # internally, and many backends (DSHOW on Windows) ignore BUFFERSIZE=1.
        # When process_frame() is slower than the camera FPS, read() returns
        # stale buffered frames, causing the stream to fall behind real-time
        # and then "jump" forward when the buffer is flushed.
        # The grabber thread continuously calls read() in a tight loop, always
        # overwriting _grabbed_frame with the latest frame. process_frame()
        # then picks up whichever frame is newest — never a stale buffer.
        self._grabbed_frame = None
        self._grab_lock = threading.Lock()
        self._grabber_thread: Optional[threading.Thread] = None
        self._grabber_stop = threading.Event()

        # Performance metrics
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

        # Auto-refresh encodings: safety net for out-of-process enrollment (e.g., seed_teachers.py)
        # Normal enrollment triggers instant reload via camera_manager.load_encodings_all()
        self._last_encoding_refresh = 0.0  # Force immediate check on first face detection
        self._encoding_refresh_interval = 300.0  # Safety-net re-check every 5 minutes
        self._encoding_db_count = 0  # Track DB count to detect new additions

        # Embedding suggestion cooldown: per-person throttle
        self._suggestion_cooldowns: Dict[str, float] = {}  # person_id → last_suggestion_time

        logger.info(f"Pipeline created for camera {camera_id}: source={stream_source}, mode={self.pipeline_mode}")

    def _grabber_loop(self):
        """
        Background thread: continuously reads frames from cv2.VideoCapture.
        Always overwrites _grabbed_frame with the latest frame so that
        process_frame() never reads a stale buffered frame.
        """
        while not self._grabber_stop.is_set():
            if not self._capture or not self._capture.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self._capture.read()
            if ret:
                with self._grab_lock:
                    self._grabbed_frame = frame
            else:
                # Brief pause on read failure to avoid busy-spin
                time.sleep(0.005)

    def start(self):
        """Start the video capture."""
        try:
            if hasattr(self, 'stream_source') and self.stream_source.startswith("browser:"):
                self._running = True
                self._capture = None
                logger.info(f"Camera {self.camera_id} started: {self.stream_source} (Browser)")
                return

            # Try to interpret as integer (webcam index)
            source = int(self.stream_source) if self.stream_source.isdigit() else self.stream_source
            self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                raise RuntimeError(f"Cannot open camera source: {self.stream_source}")

            # Optimize capture settings
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture.set(cv2.CAP_PROP_FPS, 30)

            # Start the background frame grabber thread.
            # This thread continuously calls read() in a tight loop,
            # draining the OpenCV buffer and always keeping the latest frame.
            self._grabber_stop.clear()
            self._grabber_thread = threading.Thread(
                target=self._grabber_loop,
                name=f"grabber-{self.camera_id}",
                daemon=True,
            )
            self._grabber_thread.start()

            self._running = True
            logger.info(f"Camera {self.camera_id} started: {self.stream_source}")

        except Exception as e:
            logger.error(f"Failed to start camera {self.camera_id}: {e}")
            raise

    def stop(self):
        """Stop the video capture and clean up."""
        self._running = False

        # Stop the grabber thread first (it holds a reference to _capture)
        self._grabber_stop.set()
        if self._grabber_thread and self._grabber_thread.is_alive():
            self._grabber_thread.join(timeout=3)
            self._grabber_thread = None

        if self._capture:
            self._capture.release()
            self._capture = None

        # Flush remaining exits
        self.tracker.check_exits()

        # Shutdown safety modules
        if self.attribute_recognizer:
            self.attribute_recognizer.shutdown()
        if self.hazard_detector:
            self.hazard_detector.shutdown()
        if self.plate_recognizer:
            self.plate_recognizer.shutdown()
        if self.traffic_monitor:
            self.traffic_monitor.shutdown()

        logger.info(f"Camera {self.camera_id} stopped")

    def process_frame(self) -> Optional[np.ndarray]:
        """
        Process a single frame through the full pipeline.
        Dispatches to face mode or traffic mode based on settings.
        
        Returns:
            Annotated frame (with bounding boxes) or None if no frame available
        """
        if not self._running:
            return None

        if hasattr(self, 'stream_source') and self.stream_source.startswith("browser:"):
            session_id = self.stream_source.split("browser:")[1]
            from app.routes.browser_camera import _browser_sessions
            session = _browser_sessions.get(session_id)
            if not session or session.last_frame is None:
                time.sleep(0.1)
                return None
            if not hasattr(self, '_last_browser_frame_at'):
                self._last_browser_frame_at = 0.0
            if self._last_browser_frame_at == session.last_frame_at:
                time.sleep(0.01)
                return None
            self._last_browser_frame_at = session.last_frame_at
            frame = session.last_frame.copy()
        else:
            # Grab the latest frame from the background grabber thread.
            # This is always the most recent frame — no stale buffer lag.
            with self._grab_lock:
                frame = self._grabbed_frame
                self._grabbed_frame = None  # Mark as consumed
            if frame is None:
                return None

        self._frame_count += 1

        # Auto-refresh face encodings from DB if cache is stale/empty
        self._auto_refresh_encodings()

        # ── TRAFFIC MODE ────────────────────────────────────────────────────────────
        if self.pipeline_mode == "traffic":
            return self._process_frame_traffic(frame)

        # ── HYBRID MODE ─────────────────────────────────────────────────────────────
        if self.pipeline_mode == "hybrid":
            return self._process_frame_hybrid(frame)

        # ── FACE MODE (default) ─────────────────────────────────────────────────────
        # Defer frame.copy() until a snapshot is actually needed.
        # Eagerly copying a 4 MP frame (~11 MB) every processed frame wastes
        # ~22 MB of memory bandwidth per frame (copy + cvtColor allocation).
        # Instead, we copy lazily inside _handle_unknown_face, save_snapshot, etc.
        clean_frame = frame  # Alias — only copied when needed for snapshots

        # ── Compute streaming scale factors (once per stream) ─────────
        if not self._stream_res_initialized:
            h, w = frame.shape[:2]
            if w > self._stream_max_w or h > self._stream_max_h:
                scale = min(self._stream_max_w / w, self._stream_max_h / h)
                self._stream_scale_x = scale
                self._stream_scale_y = scale
            else:
                self._stream_scale_x = 1.0
                self._stream_scale_y = 1.0
            self._stream_res_initialized = True
            logger.info(
                f"[{self.camera_id}] Stream scale: {w}x{h} -> "
                f"{int(w * self._stream_scale_x)}x{int(h * self._stream_scale_y)} "
                f"(scale={self._stream_scale_x:.3f})"
            )

        # ── Create 1080p display frame (one resize per frame) ─────────
        from app.utils.image_utils import resize_frame
        display_frame = resize_frame(frame, self._stream_max_w, self._stream_max_h)

        # Ã¢â€â‚¬Ã¢â€â‚¬ HAZARD DETECTION (runs on its own schedule) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # Submit frame to hazard detector Ã¢â‚¬â€ it internally throttles
        # to every Nth frame and runs inference on a background thread.
        # This is intentionally BEFORE the face-processing frame skip
        # so hazard detection has its own independent sampling rate.
        if self.hazard_detector:
            try:
                self.hazard_detector.submit_frame(clean_frame, self.camera_id)
            except Exception as e:
                logger.debug(f"Hazard detection submission failed: {e}")

        # Frame skip for performance (load-tier override or config default)
        effective_skip = self._frame_skip_override or settings.FRAME_SKIP
        if self._frame_count % effective_skip != 0:
            # Draw previously cached boxes on the display frame
            for draw_args in self._last_face_draws:
                self._draw_box(display_frame, *draw_args)
            self._last_frame = display_frame
            profiler.record_frame_skipped()
            return display_frame

        # Clear cache for this new detection frame
        self._last_face_draws.clear()
        profiler.record_detection_frame()

        # ── DETECTION ────────────────────────────────────────────────────────────────
        # Detect faces AND get embeddings in a SINGLE insightface pass
        # (sub-stage timing: scrfd_preprocess/scrfd_inference/scrfd_postprocess
        #  is recorded inside detector.detect_faces_with_embeddings)
        face_locations, embeddings, rgb_frame = self.detector.detect_faces_with_embeddings(frame)

        if not face_locations:
            # Update ByteTrack with empty detections
            self.byte_tracker.update(np.empty((0, 4)), np.empty(0))
            self._run_safety_analytics()
            self._check_periodic_exits()
            self._update_fps()
            self._last_frame = display_frame
            return display_frame

        logger.debug(f"Camera {self.camera_id}: Detected {len(face_locations)} face(s)")

        # Convert face locations to xyxy format for ByteTrack
        bboxes_xyxy = []
        det_scores = []
        for loc in face_locations:
            top, right, bottom, left = loc
            bboxes_xyxy.append([left, top, right, bottom])
            det_scores.append(1.0)  # SCRFD already filtered by confidence

        bboxes_array = np.array(bboxes_xyxy, dtype=np.float32)
        scores_array = np.array(det_scores, dtype=np.float32)

        # ByteTrack update: assigns stable track IDs across frames
        with profiler.measure("bytetrack"):
            active_tracks = self.byte_tracker.update(bboxes_array, scores_array)

        # ── Identity Ledger (Milestone 1: OBSERVE-ONLY) ──────────
        # Runs on the tracker's OUTPUT, so ByteTrack stays geometry-only.
        # Currently measures only: when a track is recreated, did a recoverable
        # predecessor exist? It does not assign identity yet.
        self._observe_identity_continuity(active_tracks)

        # Collect all face boxes to draw
        boxes_to_draw = []

        # Match ByteTrack tracks to detected faces by IoU
        # (tracks may have slightly different bboxes due to motion prediction)
        for track in active_tracks:
            location = track.to_location()  # (top, right, bottom, left)
            top, right, bottom, left = location
            centroid = ((left + right) // 2, (top + bottom) // 2)

            # Find the best matching detection (by IoU) to get its embedding
            best_embedding = None
            best_iou = 0.0
            tx1, ty1, tx2, ty2 = left, top, right, bottom

            for i, loc in enumerate(face_locations):
                ft, fr, fb, fl = loc
                # IoU
                ix1 = max(tx1, fl); iy1 = max(ty1, ft)
                ix2 = min(tx2, fr); iy2 = min(ty2, fb)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    a1 = (tx2 - tx1) * (ty2 - ty1)
                    a2 = (fr - fl) * (fb - ft)
                    iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0
                else:
                    iou = 0
                if iou > best_iou and embeddings[i] is not None:
                    best_iou = iou
                    best_embedding = embeddings[i]

            # ── Identity recovery (Milestone 2) ───────────────────
            # Runs BEFORE the recognition-cache decision so that a successfully
            # recovered track is treated as already-identified and does not
            # burn a forced ArcFace match on the frame it reappears.
            if best_embedding is not None:
                self._maybe_recover_identity(track, best_embedding)

            # ── Time-based recognition cache ──────────────────────
            # Only run ArcFace if the track needs re-recognition.
            # This is FPS-independent: uses wall-clock time, not frame counts.
            needs_recognition = self._should_recognize(track)
            profiler.record_track_seen()

            # Track re-created detection (lost and found)
            if track.tracklet_len < 3:
                profiler.record_track_recreated()

            # ── Face size gate for recognition ────────────────────
            # Only compute embeddings on faces large enough for reliable
            # matching. Small faces (32-80px) are still tracked but
            # reported as unidentified — never guess on insufficient pixels.
            face_w = abs(location[1] - location[3])  # right - left
            face_h = abs(location[2] - location[0])  # bottom - top
            face_too_small = (face_w < settings.MIN_FACE_RECOGNIZE_PX or
                             face_h < settings.MIN_FACE_RECOGNIZE_PX)

            if face_too_small:
                # Track but don't recognize — show as tracked face
                label = track.person_name if track.person_name else f"Track #{track.track_id}"
                color = (0, 200, 0) if track.person_name else (255, 165, 0)
                boxes_to_draw.append((location, label, color, 0.0))
                self.tracker.on_detection(
                    person_id=track.person_id or f"small_track_{track.track_id}",
                    person_name=label,
                    camera_id=self.camera_id,
                    confidence=0.0,
                    centroid=centroid,
                )
                continue

            if needs_recognition and best_embedding is not None:
                profiler.record_recognition_call()
                # Record WHY recognition was triggered for diagnostics
                if track.person_id is None:
                    profiler.record_cache_reason("no_identity")
                elif track.tracklet_len < 3:
                    profiler.record_cache_reason("new_track")
                elif track.recognition_locked:
                    profiler.record_cache_reason("recheck_locked")
                    profiler.record_forced_recheck()
                else:
                    profiler.record_cache_reason("recheck_unlocked")
                with profiler.measure("arcface"):
                    result: MatchResult = self.recognizer.recognize_face(
                        best_embedding,
                        face_quality=self._face_quality_for(frame, location),
                    )

                if result.matched:
                    # Identity swap detection: a single track flipping directly
                    # from one KNOWN person to a different KNOWN person, without
                    # passing through unknown. This is the metric that must not
                    # regress once identity recovery is enabled in Milestone 2.
                    if track.person_id and track.person_id != result.person_id:
                        profiler.record_identity_swap(
                            track.person_name or "?", result.person_name or "?",
                            track.track_id,
                        )
                        logger.warning(
                            f"[{self.camera_id}] IDENTITY SWAP on T{track.track_id}: "
                            f"{track.person_name} -> {result.person_name} "
                            f"(distance={result.distance:.4f}, margin={result.margin:.4f})"
                        )
                    track.person_id = result.person_id
                    track.person_name = result.person_name
                    # Update recognition cache on the track
                    track.recognition_confidence = result.confidence
                    track.recognition_time = time.time()
                    track.reset_fail_streak()
                    track.record_distance(result.distance)
                    if result.confidence >= self._recognition_lock_confidence:
                        track.recognition_locked = True

                    boxes_to_draw.append((location, result.person_name, (0, 255, 0), result.confidence))

                    self.tracker.on_detection(
                        person_id=result.person_id,
                        person_name=result.person_name,
                        camera_id=self.camera_id,
                        confidence=result.confidence,
                        centroid=centroid,
                    )

                    # ── Attribute extraction: only on track lock ─────
                    # Gate on recognition_locked to avoid unnecessary
                    # frame.copy() + DeepFace submission on every match.
                    # AttributeRecognizer.maybe_extract already deduplicates
                    # by track_id, but we save the copy + submission cost
                    # by not calling it until the track is confidently locked.
                    if (self.attribute_recognizer
                            and track.recognition_locked
                            and not self._skip_low_priority):
                        try:
                            self.attribute_recognizer.maybe_extract(
                                track_id=result.person_id,
                                frame=clean_frame,
                                face_location=location,
                                on_complete=self._handle_attributes_complete,
                            )
                        except Exception as e:
                            logger.debug(f"Attribute extraction error: {e}")

                    # ── Embedding Suggestion (throttled, diversity-gated) ──
                    # Skipped under load degradation (P3 low-priority work)
                    if not self._skip_low_priority:
                        self._maybe_suggest_encoding(
                            person_id=result.person_id,
                            encoding=best_embedding,
                            confidence=result.confidence,
                            frame=clean_frame,
                            location=location,
                        )
                else:
                    # ── HYSTERESIS: don't immediately drop a known identity ──
                    # If the track already has a person_id and the failure is
                    # borderline (gray zone), keep the previous identity for
                    # up to IDENTITY_GRACE_FRAMES consecutive failures.
                    # This eliminates Known→Unknown→Known flicker caused by
                    # ArcFace distance oscillating around the threshold.
                    track.record_distance(result.distance)

                    # Temporal voting rescue: if the rolling average distance
                    # is below threshold even though this single frame missed,
                    # treat it as a match (the person IS known on average).
                    temporal_rescue = False
                    if (track.person_id and track.person_name
                            and len(track._distance_history) >= 3):
                        # Use the balanced threshold for temporal comparison
                        temporal_threshold = settings.FACE_MATCH_BALANCED
                        if track.temporal_avg_distance <= temporal_threshold:
                            temporal_rescue = True
                            logger.info(
                                f"🔄 TEMPORAL RESCUE: {track.person_name} "
                                f"single_dist={result.distance:.4f} > threshold, "
                                f"but avg={track.temporal_avg_distance:.4f} <= "
                                f"{temporal_threshold} (window={len(track._distance_history)})"
                            )

                    if temporal_rescue:
                        # Temporal average says this person is known — keep identity
                        track.reset_fail_streak()
                        track.recognition_time = time.time()
                        boxes_to_draw.append((location, track.person_name, (0, 200, 0), track.recognition_confidence))
                        self.tracker.on_detection(
                            person_id=track.person_id,
                            person_name=track.person_name,
                            camera_id=self.camera_id,
                            confidence=track.recognition_confidence,
                            centroid=centroid,
                        )
                    elif track.person_id and track.person_name and result.in_gray_zone:
                        # Gray zone: borderline failure — apply hysteresis
                        track.identity_fail_streak += 1
                        if track.identity_fail_streak < settings.IDENTITY_GRACE_FRAMES:
                            # Keep previous identity (grace period)
                            logger.info(
                                f"⏳ HYSTERESIS: keeping {track.person_name} "
                                f"(T{track.track_id}) despite distance={result.distance:.4f}, "
                                f"fail {track.identity_fail_streak}/{settings.IDENTITY_GRACE_FRAMES}, "
                                f"avg={track.temporal_avg_distance:.4f}"
                            )
                            # Show with slightly dimmer green to indicate borderline
                            boxes_to_draw.append((location, track.person_name, (0, 200, 0), track.recognition_confidence))
                            self.tracker.on_detection(
                                person_id=track.person_id,
                                person_name=track.person_name,
                                camera_id=self.camera_id,
                                confidence=track.recognition_confidence,
                                centroid=centroid,
                            )
                        else:
                            # Exhausted grace period — drop identity
                            logger.info(
                                f"🚫 HYSTERESIS EXPIRED: dropping {track.person_name} "
                                f"(T{track.track_id}) after {track.identity_fail_streak} "
                                f"consecutive gray-zone failures"
                            )
                            track.person_id = None
                            track.person_name = None
                            track.recognition_locked = False
                            track.identity_fail_streak = 0
                            boxes_to_draw.append((location, "UNKNOWN", (0, 0, 255), 0.0))
                            self._handle_unknown_face(clean_frame, frame, location, best_embedding, centroid)
                    else:
                        # Definitive failure (distance far above threshold, or
                        # no previous identity) — immediate unknown
                        if track.person_id and track.person_name:
                            logger.info(
                                f"🚫 DEFINITE MISS: dropping {track.person_name} "
                                f"(T{track.track_id}), distance={result.distance:.4f} "
                                f"above gray ceiling"
                            )
                            track.person_id = None
                            track.person_name = None
                            track.recognition_locked = False
                            track.identity_fail_streak = 0
                        boxes_to_draw.append((location, "UNKNOWN", (0, 0, 255), 0.0))
                        self._handle_unknown_face(clean_frame, frame, location, best_embedding, centroid)
            elif not needs_recognition and track.person_id and track.person_name:
                # Reuse existing identity from ByteTrack cache (skip ArcFace)
                profiler.record_cache_hit()
                boxes_to_draw.append((location, track.person_name, (0, 255, 0), track.recognition_confidence))
                self.tracker.on_detection(
                    person_id=track.person_id,
                    person_name=track.person_name,
                    camera_id=self.camera_id,
                    confidence=track.recognition_confidence,
                    centroid=centroid,
                )
            else:
                # No embedding available or unidentified track
                label = track.person_name if track.person_name else f"Track #{track.track_id}"
                color = (0, 200, 0) if track.person_name else (255, 165, 0)
                boxes_to_draw.append((location, label, color, 0.0))

        # ── Scale bounding boxes to display resolution & draw ─────
        # boxes_to_draw contains coords in full 4MP space.
        # Scale them to 1080p display_frame coords for drawing.
        sx = self._stream_scale_x
        sy = self._stream_scale_y
        scaled_boxes = []
        for (location, label, color, conf) in boxes_to_draw:
            top, right, bottom, left = location
            scaled_loc = (
                int(top * sy), int(right * sx),
                int(bottom * sy), int(left * sx),
            )
            scaled_boxes.append((scaled_loc, label, color, conf))

        # Cache SCALED boxes for frame-skip redraw
        self._last_face_draws = scaled_boxes
        with profiler.measure("drawing"):
            for draw_args in scaled_boxes:
                self._draw_box(display_frame, *draw_args)

        # ── YOLO-ASSISTED DETECTION (adaptive trigger) ────────────
        # Runs on full-res clean_frame for maximum detection accuracy.
        if self._should_run_yolo_assist(active_tracks, face_locations):
            self._last_yolo_assist_check = time.time()
            try:
                yolo_persons = self._get_yolo_person_detections(clean_frame)
                if yolo_persons:
                    missed = self._find_missed_persons(yolo_persons, face_locations)
                    for person_det in missed:
                        self._attempt_yolo_assisted_detection(
                            frame, clean_frame, person_det
                        )
            except Exception as e:
                logger.debug(f"YOLO-assisted detection error: {e}")

        # ── Load-tier degradation ─────────────────────────────────
        # Adjust pipeline parameters based on current face count.
        self._active_face_count = len(face_locations)
        self._apply_load_degradation()

        self._run_safety_analytics()
        self._check_periodic_exits()
        self._update_fps()

        self._last_frame = display_frame
        return display_frame

    def _observe_identity_continuity(self, active_tracks: list):
        """
        Feed the tracker's output to the IdentityLedger and record metrics.

        MILESTONE 1: observe-only. Assigns nothing, mutates nothing. It answers
        the question the Phase 1 investigation could not answer statically:
        when ByteTrack recreates a track, how often did a recoverable
        predecessor actually exist, and how close was it?

        That number is the ceiling on what identity recovery can reclaim.
        """
        try:
            opportunities = self.identity_ledger.observe(active_tracks)
        except Exception as e:
            # Never let instrumentation break the pipeline.
            logger.warning(f"[{self.camera_id}] Identity ledger error: {e}")
            return

        # Prune the recovered-track set: drop tracks that are gone, and tracks
        # that have aged past the `tracklet_len < 3` window the flag exists to
        # cover. Keeps the set bounded by concurrent face count.
        live_ids = {t.track_id for t in active_tracks}
        matured = {t.track_id for t in active_tracks if t.tracklet_len >= 3}
        self._recovered_track_ids &= live_ids
        self._recovered_track_ids -= matured

        for opp in opportunities:
            if opp.recoverable:
                b = opp.best
                profiler.record_track_created(
                    recoverable=True,
                    best_iou=b.iou,
                    best_dist=b.centroid_distance,
                    gap_frames=b.frames_since_retired,
                )
                logger.info(
                    f"[{self.camera_id}] RECOVERABLE: new T{opp.track_id} could be "
                    f"{b.person_name} (was T{b.retired.track_id}) — "
                    f"iou={b.iou:.3f} dist={b.centroid_distance:.0f}px "
                    f"gap={b.frames_since_retired}f "
                    f"locked={b.retired.recognition_locked} "
                    f"hist={len(b.retired.distance_history)} | "
                    f"eligible={opp.n_eligible}/{opp.n_retired_total} "
                    f"[MILESTONE 1: observed, NOT applied]"
                )
            else:
                profiler.record_track_created(recoverable=False)
                logger.debug(
                    f"[{self.camera_id}] new T{opp.track_id} — no recoverable "
                    f"predecessor (retired pool={opp.n_retired_total})"
                )

    def _maybe_recover_identity(self, track: STrack, embedding: np.ndarray) -> bool:
        """
        Attempt to restore a previously-known identity onto a freshly created track.

        MILESTONE 2. Flow:

            ledger proposes  ->  recogniser verifies  ->  ledger restores state

        The ledger is NEVER the authority. It can only nominate a candidate;
        FaceRecognizer.verify_against_person() decides, using stricter
        thresholds than ordinary recognition. This bounds the identity-swap
        risk by the recogniser that is already in production, rather than
        introducing a second, divergent recognition path.

        Returns:
            True if an identity was verified and restored.
        """
        if not settings.IDENTITY_RECOVERY_ENABLED:
            return False

        proposal = self.identity_ledger.take_proposal(track.track_id)
        if proposal is None or proposal.best is None:
            return False

        # Coexistence guard: ByteTrack's legacy carry-over is still present and
        # untouched. It is measured at 0% success, but if it ever does fire we
        # must not double-assign on top of it.
        if track.person_id:
            profiler.record_legacy_carryover_fired()
            logger.warning(
                f"[{self.camera_id}] Legacy ByteTrack carry-over fired on "
                f"T{track.track_id} ({track.person_name}) — skipping ledger recovery"
            )
            return False

        if settings.IDENTITY_REQUIRE_UNAMBIGUOUS and not proposal.unambiguous:
            profiler.record_recovery_rejected("ambiguous")
            logger.info(
                f"[{self.camera_id}] RECOVERY DECLINED (ambiguous): T{track.track_id} "
                f"had {proposal.n_eligible} eligible candidates — "
                f"refusing to guess between them"
            )
            return False

        cand = proposal.best
        profiler.record_recovery_attempted()

        result = self.recognizer.verify_against_person(
            embedding, cand.retired.person_id,
        )

        if not result.matched:
            profiler.record_recovery_rejected(result.rejection_reason or "unknown")
            logger.info(
                f"[{self.camera_id}] RECOVERY REJECTED: T{track.track_id} proposed as "
                f"{cand.person_name} but verification failed "
                f"({result.rejection_reason}, distance={result.distance:.4f}) — "
                f"falling back to full recognition"
            )
            return False

        self.identity_ledger.restore(
            track, cand.retired,
            verified_confidence=result.confidence,
            lock_threshold=self._recognition_lock_confidence,
        )
        self._recovered_track_ids.add(track.track_id)
        profiler.record_recovery_verified()
        logger.info(
            f"[{self.camera_id}] ✅ RECOVERED: T{track.track_id} = {track.person_name} "
            f"(confidence={result.confidence:.3f}, margin={result.margin:.4f}, "
            f"gap={cand.frames_since_retired}f, dist={cand.centroid_distance:.0f}px, "
            f"locked={track.recognition_locked}, "
            f"history={len(track._distance_history)} restored)"
        )
        return True

    def _face_quality_for(self, frame: np.ndarray, location: tuple) -> float:
        """
        Quality score (0.0–1.0) used to pick the adaptive recognition threshold.

        Returns the neutral 0.5 unless ENABLE_ADAPTIVE_THRESHOLDS is set, which
        keeps recognition behaviour byte-identical to the previous (accidentally
        fixed-threshold) behaviour until a deployment has measured FAR/FRR.

        When enabled, delegates to FaceDetector.assess_face_quality, which blends
        face area, Laplacian sharpness, and brightness.
        """
        if not settings.ENABLE_ADAPTIVE_THRESHOLDS:
            return 0.5
        try:
            return self.detector.assess_face_quality(frame, location)
        except Exception as e:
            logger.warning(
                f"[{self.camera_id}] Face quality assessment failed, "
                f"falling back to neutral 0.5: {e}"
            )
            return 0.5

    def _should_recognize(self, track: STrack) -> bool:
        """
        Determine if a ByteTrack track needs ArcFace recognition.

        Uses wall-clock time (FPS-independent) instead of frame counts:
        - No identity yet → always recognize
        - Just created (tracklet_len < 3) → always recognize
        - Unlocked (low confidence) → re-check every RECOGNITION_RECHECK_SEC
        - Locked (high confidence) → re-check every RECOGNITION_FULL_RECHECK_SEC
        """
        # No identity yet — must recognize
        if track.person_id is None:
            return True

        # Track was just re-created (lost and found) — must re-verify.
        #
        # EXCEPTION: if the identity ledger already recovered this track and the
        # recogniser VERIFIED that identity this frame, re-running a full match
        # would be redundant work on an answer we just computed. Skipping it is
        # the cache-efficiency win of identity recovery.
        if track.tracklet_len < 3:
            if track.track_id in self._recovered_track_ids:
                return False
            return True

        now = time.time()
        elapsed = now - track.recognition_time

        if track.recognition_locked:
            # Confidently identified — long re-check interval
            return elapsed >= self._recognition_full_recheck_sec
        else:
            # Not yet confident — shorter re-check interval
            return elapsed >= self._recognition_recheck_sec

    def _should_run_yolo_assist(
        self,
        active_tracks: list,
        face_locations: list,
    ) -> bool:
        """
        Adaptive YOLO-Assist trigger — runs when detection needs help,
        not on a fixed timer.

        Conditions that trigger YOLO-Assist:
        1. Tracks with no face match (person body visible, face not detected)
        2. Tracker confidence drop (matched but degraded)
        3. Prolonged occlusion recovery (lost and re-created track)
        4. Person count mismatch (fewer faces than expected from trackers)

        Always respects minimum interval to prevent spam.
        """
        now = time.time()

        # Minimum interval gate — prevent spam
        if now - self._last_yolo_assist_check < self._yolo_assist_min_interval:
            return False

        # Under heavy load, widen the minimum interval
        if self._skip_low_priority:
            if now - self._last_yolo_assist_check < 3.0:
                return False

        # Trigger 1: Tracks with no face match
        # Person body visible but face not detected by SCRFD
        unmatched_tracks = [
            t for t in active_tracks
            if t.person_id is None and t.tracklet_len > 5
        ]
        if unmatched_tracks:
            return True

        # Trigger 2: Low confidence tracks that haven't locked
        low_confidence = [
            t for t in active_tracks
            if t.person_id and not t.recognition_locked
            and t.tracklet_len > 10
        ]
        if low_confidence:
            return True

        # Trigger 3: Occlusion recovery — track was lost and re-created
        recovered = [
            t for t in active_tracks
            if t.tracklet_len < 3 and t.person_id
        ]
        if recovered:
            return True

        # Trigger 4: Person count mismatch
        # Tracker knows about more people than SCRFD found faces
        tracker_count = len(self.tracker.get_person_centroids())
        face_count = len(face_locations)
        if tracker_count > face_count + 2:
            return True

        # No trigger condition met
        return False

    def _apply_load_degradation(self):
        """
        Adjust pipeline parameters based on active face count.
        Implements graceful degradation under burst load.

        Tiers:
          0-4 faces → Normal operation
          5-9 faces → Tier 1: defer low-priority work
          10+ faces → Tier 2: increase frame skip, reduce JPEG quality
        """
        n = self._active_face_count

        if n >= 10:
            # Tier 2: Heavy load
            self._frame_skip_override = max(settings.FRAME_SKIP, 4)
            self._yolo_assist_min_interval = 3.0
            self._jpeg_quality_override = 65
            self._skip_low_priority = True
        elif n >= 5:
            # Tier 1: Moderate load
            self._frame_skip_override = None  # Use default
            self._yolo_assist_min_interval = 2.0
            self._jpeg_quality_override = None
            self._skip_low_priority = True
        else:
            # Normal operation
            self._frame_skip_override = None
            self._yolo_assist_min_interval = 1.0
            self._jpeg_quality_override = None
            self._skip_low_priority = False

    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
    #  Safety & Security Integration
    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

    def _auto_refresh_encodings(self):
        """
        Auto-refresh face encodings from DB when the recognizer cache is empty
        or when new encodings have been added to the database.

        This handles the case where:
        - seed_teachers.py runs in a separate process after camera starts
        - New persons are enrolled via the Unknown Faces page from another client
        - The server restarts but the camera reconnects before encodings are cached
        """
        now = time.time()
        if now - self._last_encoding_refresh < self._encoding_refresh_interval:
            return

        self._last_encoding_refresh = now

        try:
            from app.services.person_service import PersonService
            encodings = PersonService.get_all_encodings()
            db_count = len(encodings)

            if db_count > 0 and db_count != self._encoding_db_count:
                # DB has encodings that aren't loaded yet (or count changed)
                self.recognizer.load_encodings(encodings)
                self._encoding_db_count = db_count
                logger.info(
                    f"[{self.camera_id}] Auto-refreshed {db_count} face encodings from DB "
                    f"(was {self.recognizer.known_count} before refresh)"
                )
        except Exception as e:
            logger.debug(f"[{self.camera_id}] Auto-refresh encodings failed: {e}")

    def _run_safety_analytics(self):
        """
        Run crowd and loitering detection at a throttled interval.
        Uses centroid data from the tracker — completely non-blocking
        since all heavy work is already done by the time centroids exist.
        """
        now = time.time()
        if now - self._last_safety_check < self._safety_check_interval:
            return
        self._last_safety_check = now

        try:
            centroids = self.tracker.get_person_centroids()
            names = self.tracker.get_person_names()

            # Crowd Detection
            if self.crowd_detector and len(centroids) >= 2:
                self.crowd_detector.update(centroids, self.camera_id)

            # Loitering Detection
            if self.loitering_detector and centroids:
                self.loitering_detector.update(
                    centroids, names, self.camera_id
                )
        except Exception as e:
            logger.debug(f"Safety analytics error: {e}")

    def _handle_security_alert(self, alert: dict):
        """
        Callback for all security alerts (crowd, loitering, hazard,
        vehicle_proximity). Routes through standard event system.
        """
        logger.warning(
            f"Security alert on camera {self.camera_id}: "
            f"{alert.get('subtype', 'unknown')} — {alert}"
        )
        if self.on_event:
            self.on_event(alert)

    def _handle_vehicle_event(self, event: dict):
        """
        Callback for ANPR vehicle entry events.
        Routes through the standard event system.
        """
        logger.info(
            f"Vehicle event on camera {self.camera_id}: "
            f"plate={event.get('metadata', {}).get('plate', '???')}"
        )
        if self.on_event:
            self.on_event(event)

    def _handle_vehicle_count(self, count_data: dict):
        """
        Callback for TrafficMonitor vehicle count updates.
        Broadcasts real-time vehicle/person counts to the frontend.
        """
        if self.on_event:
            self.on_event({
                "event_type": "vehicle_count",
                "camera_id": self.camera_id,
                "timestamp": count_data.get("timestamp", ""),
                "metadata": {
                    "total_vehicles": count_data.get("total_vehicles", 0),
                    "total_persons": count_data.get("total_persons", 0),
                    "vehicle_breakdown": count_data.get("vehicle_breakdown", {}),
                    "vehicle_details": count_data.get("vehicle_details", []),
                },
            })

    def _handle_attributes_complete(self, track_id: str, attributes: dict):
        """
        Called when attribute recognition finishes for a person.
        The attributes (gender, clothing color) can be attached to
        detection events via the metadata JSONB column.
        """
        logger.info(
            f"Attributes ready for {track_id[:8]}…: {attributes}"
        )
        # The attributes are cached in AttributeRecognizer._results
        # and will be included in future events via get_attributes()

    def _maybe_suggest_encoding(
        self,
        person_id: str,
        encoding: np.ndarray,
        confidence: float,
        frame: np.ndarray,
        location: tuple,
    ):
        """
        Evaluate whether a confident match should be queued as an embedding suggestion.

        Pipeline:
        1. Confidence gate (>= 0.85)
        2. Per-person cooldown (max 1 per 60s)
        3. Quality gate (must pass FaceQualityGate)
        4. Diversity check (pose bucket, quality, appearance)
        5. Save snapshot + create SuggestedEncoding row

        NEVER auto-adds — always queued for admin approval.
        """
        # Step 1: Confidence gate
        if confidence < settings.SUGGESTION_MIN_CONFIDENCE:
            return

        # Step 2: Per-person cooldown
        now = time.time()
        last_time = self._suggestion_cooldowns.get(person_id, 0)
        if now - last_time < settings.SUGGESTION_COOLDOWN_SEC:
            return

        try:
            from app.vision.quality import FaceQualityGate
            from app.services.embedding_suggestions import (
                EmbeddingSuggestionService,
                classify_pose_bucket,
                detect_appearance_tags,
            )

            # Step 3: Quality gate
            gate = FaceQualityGate()
            face_crop = self.detector.extract_face_crop(frame, location, padding=40)
            # NOTE: face_location must be the *unpadded* box — assess() uses it for
            # the size gate and size_score, which should measure the face itself,
            # not the padded crop.
            # TODO(phase2): pass real 5-point landmarks so estimate_pose() returns a
            # true yaw. With landmarks=None it returns (0,0,0), so every suggestion
            # buckets as "frontal" and the pose-coverage check (step 2 of the
            # diversity ladder) can never fire.
            quality_result = gate.assess(
                face_crop=face_crop,
                face_location=location,
                det_score=confidence,
                landmarks=None,
            )
            if not quality_result.passed:
                return

            metrics = quality_result.metrics

            # Step 4: Diversity check (only creates DB row if passes)
            result = EmbeddingSuggestionService.evaluate_suggestion(
                person_id=person_id,
                new_encoding=encoding,
                quality=quality_result.quality_score,
                yaw=metrics.get("yaw", 0.0),
                pitch=metrics.get("pitch", 0.0),
                blur_score=metrics.get("blur_score", 0.0),
                lighting=metrics.get("lighting"),
            )

            if not result.should_suggest:
                return

            # Step 5: Save snapshot and create suggestion
            from app.utils.image_utils import save_snapshot

            _, snapshot_url = save_snapshot(
                face_crop,
                f"suggestion_{person_id[:8]}_{int(now)}",
                subdirectory="suggestions",
                quality=95,
            )

            EmbeddingSuggestionService.create_suggestion(
                person_id=person_id,
                encoding=encoding,
                snapshot_url=snapshot_url,
                camera_id=self.camera_id,
                quality=quality_result.quality_score,
                yaw=metrics.get("yaw", 0.0),
                pitch=metrics.get("pitch", 0.0),
                roll=metrics.get("roll", 0.0),
                det_score=confidence,
                blur_score=metrics.get("blur_score", 0.0),
                lighting=metrics.get("lighting"),
                similarity=confidence,
                reason=result.reason,
                pose_bucket=result.pose_bucket,
                appearance_tags=result.appearance_tags,
            )

            # Update cooldown
            self._suggestion_cooldowns[person_id] = now

        except Exception as e:
            logger.debug(f"Embedding suggestion error: {e}")

    # ═════════════════════════════════════════════════════════════════════════════
    #  Original Pipeline Methods (unchanged logic)
    # ═════════════════════════════════════════════════════════════════════════════

    def _handle_unknown_face(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray,
        location: tuple,
        encoding: np.ndarray,
        centroid: Optional[Tuple[int, int]] = None,
    ):
        """Handle an unrecognized face Ã¢â‚¬â€ deduplicate, track, and queue."""
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

            # Track this unknown person so they appear in "People Present"
            unknown_label = self._unknown_cache[similar_id].get("label", "Unknown")
            self.tracker.on_detection(
                person_id=similar_id,
                person_name=unknown_label,
                camera_id=self.camera_id,
                confidence=0.0,
                centroid=centroid,
            )
        else:
            # New unknown face
            unknown_id = str(uuid4())

            # Assign a label like "Unknown #1", "Unknown #2", etc.
            unknown_number = len(self._unknown_cache) + 1
            unknown_label = f"Unknown #{unknown_number}"

            # Ã¢â€â‚¬Ã¢â€â‚¬ High-quality face crop with generous padding Ã¢â€â‚¬Ã¢â€â‚¬
            # Copy frame data NOW (frame reference changes next iteration)
            # These copies are fast (~0.5ms each) vs save_snapshot (~10-30ms each)
            face_crop = self.detector.extract_face_crop(frame, location, padding=80).copy()
            context_crop = self._extract_context_crop(frame, location).copy()
            full_frame_copy = frame.copy()

            # Ã¢â€â‚¬Ã¢â€â‚¬ Wider context crop (head + shoulders) for better identification Ã¢â€â‚¬Ã¢â€â‚¬

            # Ã¢â€â‚¬Ã¢â€â‚¬ Full frame at good quality Ã¢â€â‚¬Ã¢â€â‚¬
            self._unknown_cache[unknown_id] = {
                "encoding": encoding,
                "count": 1,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "label": unknown_label,
            }

            # Track this new unknown person so they appear in "People Present"
            self.tracker.on_detection(
                person_id=unknown_id,
                person_name=unknown_label,
                camera_id=self.camera_id,
                confidence=0.0,
                centroid=centroid,
            )

            # Enqueue for async processing (snapshot saving + DB + WS)
            # on_unknown() now just enqueues to EventQueue — non-blocking
            if self.on_unknown:
                self.on_unknown({
                    "unknown_id": unknown_id,
                    "camera_id": self.camera_id,
                    "face_crop": face_crop,
                    "context_crop": context_crop,
                    "full_frame": full_frame_copy,
                    "encoding": encoding,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "bounding_box": {
                        "top": location[0], "right": location[1],
                        "bottom": location[2], "left": location[3]
                    },
                })

    def _handle_entry_event(self, event: dict):
        """Callback when tracker confirms an entry."""
        # Enrich with attribute data if available
        if self.attribute_recognizer:
            attrs = self.attribute_recognizer.get_attributes(
                event.get("person_id", "")
            )
            if attrs:
                event.setdefault("metadata", {}).update(attrs)

        if self.on_event:
            self.on_event(event)

    def _handle_exit_event(self, event: dict):
        """Callback when tracker detects an exit."""
        # Enrich with attribute data if available
        if self.attribute_recognizer:
            attrs = self.attribute_recognizer.get_attributes(
                event.get("person_id", "")
            )
            if attrs:
                event.setdefault("metadata", {}).update(attrs)

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

    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
    #  YOLO-Assisted Person Detection
    #  When face_recognition misses a face but YOLO detects a person body,
    #  we attempt focused face detection on the head region and create an
    #  unknown snapshot if the face is still undetectable.
    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

    def _load_yolo_person_detector(self) -> bool:
        """Lazy-load a YOLO model for person detection (used in FACE mode).
        
        Reuses the hazard detector's YOLO model if available to avoid
        loading the same model twice (saves ~6MB GPU per camera).
        """
        with self._yolo_person_lock:
            if self._yolo_person_loaded:
                return self._yolo_person_detector is not None
            try:
                # Try to reuse the hazard detector's model first
                if (self.hazard_detector 
                    and self.hazard_detector._model_available 
                    and self.hazard_detector._model_backend == "yolo"
                    and self.hazard_detector._model is not None):
                    self._yolo_person_detector = self.hazard_detector._model
                    self._yolo_person_device = getattr(self.hazard_detector, '_device', 'cpu')
                    logger.info(f"[{self.camera_id}] YOLO person detector: reusing hazard detector model (shared)")
                else:
                    from ultralytics import YOLO
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model_path = getattr(settings, 'HAZARD_MODEL_PATH', 'yolov8n.pt')
                    self._yolo_person_detector = YOLO(model_path)
                    self._yolo_person_device = device
                    logger.info(f"[{self.camera_id}] YOLO person detector loaded on {device}")
            except Exception as e:
                logger.warning(f"[{self.camera_id}] YOLO person detector unavailable: {e}")
                self._yolo_person_detector = None
            finally:
                self._yolo_person_loaded = True
            return self._yolo_person_detector is not None

    def _get_yolo_person_detections(self, frame: np.ndarray) -> List[dict]:
        """
        Run YOLO person detection on a frame (FACE mode only).
        Returns list of person detection dicts with 'bbox' and 'confidence'.
        """
        if not self._load_yolo_person_detector():
            return []

        try:
            results = self._yolo_person_detector(
                frame,
                conf=0.25,
                classes=[0],  # person class only
                verbose=False,
                device=getattr(self, '_yolo_person_device', 'cpu'),
            )

            persons = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != 0:
                        continue
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                    persons.append({
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
            return persons
        except Exception as e:
            logger.debug(f"YOLO person detection failed: {e}")
            return []

    def _get_yolo_person_detections_from_traffic(self) -> List[dict]:
        """
        Get person detections from the TrafficMonitor (HYBRID mode).
        Avoids running YOLO twice since TrafficMonitor already does it.
        """
        if not self.traffic_monitor:
            return []

        try:
            with self.traffic_monitor._detections_lock:
                persons = list(self.traffic_monitor._last_detections.get("persons", []))
            return persons
        except Exception:
            return []

    def _find_missed_persons(
        self,
        yolo_persons: List[dict],
        face_locations: List[Tuple],
    ) -> List[dict]:
        """
        Find YOLO person detections that don't overlap with any detected face.
        These are persons whose faces were missed by face_recognition.
        """
        missed = []
        for person in yolo_persons:
            pb = person["bbox"]
            has_matching_face = False

            for face_loc in face_locations:
                if self._face_overlaps_person(face_loc, pb):
                    has_matching_face = True
                    break

            if not has_matching_face:
                missed.append(person)

        return missed

    @staticmethod
    def _face_overlaps_person(
        face_loc: Tuple[int, int, int, int],
        person_bbox: dict,
    ) -> bool:
        """
        Check whether a face detection (top, right, bottom, left) overlaps
        with a YOLO person bbox (x1, y1, x2, y2).
        """
        f_top, f_right, f_bottom, f_left = face_loc
        p_x1, p_y1, p_x2, p_y2 = (
            person_bbox["x1"], person_bbox["y1"],
            person_bbox["x2"], person_bbox["y2"],
        )

        # Check for overlap
        overlap_x = max(0, min(f_right, p_x2) - max(f_left, p_x1))
        overlap_y = max(0, min(f_bottom, p_y2) - max(f_top, p_y1))

        if overlap_x > 0 and overlap_y > 0:
            # Some overlap exists Ã¢â‚¬â€ check if it's significant
            face_area = (f_right - f_left) * (f_bottom - f_top)
            overlap_area = overlap_x * overlap_y
            # The face center should be inside the person box
            face_cx = (f_left + f_right) // 2
            face_cy = (f_top + f_bottom) // 2
            if p_x1 <= face_cx <= p_x2 and p_y1 <= face_cy <= p_y2:
                return True
            # Or significant overlap ratio
            if face_area > 0 and overlap_area / face_area > 0.3:
                return True

        return False

    def _attempt_yolo_assisted_detection(
        self,
        frame: np.ndarray,
        clean_frame: np.ndarray,
        person_det: dict,
    ):
        """
        Attempt to detect a face in the upper body of a YOLO person detection.
        If face detection succeeds, process normally. If not, create a body-only
        unknown person snapshot.
        """
        pb = person_det["bbox"]
        person_confidence = person_det.get("confidence", 0.0)

        # Skip very low confidence person detections
        if person_confidence < 0.30:
            return

        h, w = clean_frame.shape[:2]

        # Extract the upper body region (head + shoulders)
        head_region = self.detector.estimate_head_region(pb, clean_frame.shape)
        head_top, head_right, head_bottom, head_left = head_region

        # Ensure valid crop dimensions
        if head_bottom <= head_top or head_right <= head_left:
            return

        head_crop = clean_frame[head_top:head_bottom, head_left:head_right]
        if head_crop.size == 0:
            return

        # Attempt face detection on the focused crop at full resolution
        crop_face_locs, crop_rgb = self.detector.detect_faces_in_crop(head_crop)

        if crop_face_locs:
            # Face found in the crop! Process each face
            for crop_loc in crop_face_locs:
                c_top, c_right, c_bottom, c_left = crop_loc
                # Convert crop-local coordinates back to frame-global
                global_loc = (
                    c_top + head_top,
                    c_right + head_left,
                    c_bottom + head_top,
                    c_left + head_left,
                )

                # Generate encoding from the crop (better quality than downscaled)
                encoding = self.recognizer.generate_encoding(crop_rgb, crop_loc)
                if encoding is None:
                    continue

                centroid = ((global_loc[3] + global_loc[1]) // 2,
                            (global_loc[0] + global_loc[2]) // 2)

                # Try to recognize
                result = self.recognizer.recognize_face(
                    encoding,
                    face_quality=self._face_quality_for(clean_frame, global_loc),
                )
                if result.matched:
                    self._draw_box(frame, global_loc, result.person_name, (0, 255, 0), result.confidence)
                    self.tracker.on_detection(
                        person_id=result.person_id,
                        person_name=result.person_name,
                        camera_id=self.camera_id,
                        confidence=result.confidence,
                        centroid=centroid,
                    )
                else:
                    self._draw_box(frame, global_loc, "UNKNOWN", (0, 0, 255), 0.0)
                    # Pass BGR frame directly — InsightFace handles color internally
                    self._handle_unknown_face(clean_frame, clean_frame, global_loc, encoding, centroid)

            logger.info(
                f"[{self.camera_id}] YOLO-assisted: found {len(crop_face_locs)} face(s) "
                f"in missed person body (conf={person_confidence:.0%})"
            )
        else:
            # No face detected even in the focused crop Ã¢â‚¬â€ create body-only unknown
            # This person is visible but their face cannot be detected
            # (e.g., facing away, heavy occlusion, very small)
            self._handle_unknown_person_body(frame, clean_frame, person_det)

    def _handle_unknown_person_body(
        self,
        frame: np.ndarray,
        clean_frame: np.ndarray,
        person_det: dict,
    ):
        """
        Handle a person detected by YOLO whose face could NOT be detected.
        Creates a body-only unknown person snapshot and tracks them.

        Uses centroid-based dedup to avoid spamming unknowns for the
        same body detection across frames.
        """
        pb = person_det["bbox"]
        person_confidence = person_det.get("confidence", 0.0)

        # Compute person centroid
        pcx = (pb["x1"] + pb["x2"]) // 2
        pcy = (pb["y1"] + pb["y2"]) // 2
        centroid = (pcx, pcy)

        # Dedup: check if we already have a body-only unknown near this centroid
        DEDUP_DISTANCE = 200  # pixels — generous to prevent duplicates from natural movement
        similar_body_id = None
        now = time.time()
        for uid, udata in self._yolo_body_unknown_cache.items():
            dx = abs(udata["centroid"][0] - pcx)
            dy = abs(udata["centroid"][1] - pcy)
            if dx < DEDUP_DISTANCE and dy < DEDUP_DISTANCE:
                similar_body_id = uid
                break

        if similar_body_id:
            # Update existing body unknown
            self._yolo_body_unknown_cache[similar_body_id]["count"] += 1
            self._yolo_body_unknown_cache[similar_body_id]["last_seen"] = now
            self._yolo_body_unknown_cache[similar_body_id]["centroid"] = centroid

            # Keep tracking them
            unknown_label = self._yolo_body_unknown_cache[similar_body_id].get("label", "Unknown Person")
            self.tracker.on_detection(
                person_id=similar_body_id,
                person_name=unknown_label,
                camera_id=self.camera_id,
                confidence=0.0,
                centroid=centroid,
            )
        else:
            # New body-only unknown person
            unknown_id = str(uuid4())
            unknown_number = len(self._unknown_cache) + len(self._yolo_body_unknown_cache) + 1
            unknown_label = f"Unknown #{unknown_number}"

            # Crop the person body from the clean frame
            h, w = clean_frame.shape[:2]
            body_crop = clean_frame[
                max(0, pb["y1"]):min(h, pb["y2"]),
                max(0, pb["x1"]):min(w, pb["x2"]),
            ].copy()

            # Also extract head region for a better snapshot
            head_region = self.detector.estimate_head_region(pb, clean_frame.shape)
            head_top, head_right, head_bottom, head_left = head_region
            head_crop = clean_frame[head_top:head_bottom, head_left:head_right].copy()

            # Copy the full frame for snapshot saving (cheap ~1.5ms).
            # Snapshot saving is deferred to the EventQueue consumer thread
            # to avoid blocking the vision pipeline for ~64ms (3× save_snapshot).
            full_frame_copy = clean_frame.copy()

            self._yolo_body_unknown_cache[unknown_id] = {
                "centroid": centroid,
                "count": 1,
                "first_seen": now,
                "last_seen": now,
                "label": unknown_label,
                "confidence": person_confidence,
            }

            # Track this unknown person
            self.tracker.on_detection(
                person_id=unknown_id,
                person_name=unknown_label,
                camera_id=self.camera_id,
                confidence=0.0,
                centroid=centroid,
            )

            # Draw a box around the body
            body_loc = (pb["y1"], pb["x2"], pb["y2"], pb["x1"])
            self._draw_box(frame, body_loc, f"UNKNOWN (body)", (0, 128, 255), person_confidence)

            logger.info(
                f"[{self.camera_id}] Body-only unknown person detected: "
                f"{unknown_label} (YOLO conf={person_confidence:.0%})"
            )

            # Enqueue for async processing (snapshot saving + DB + WS).
            # Passes raw frame data to the EventQueue consumer thread which
            # handles save_snapshot() — same pattern as _handle_unknown_face.
            # This removes ~64ms of synchronous disk I/O from the critical path.
            if self.on_unknown:
                self.on_unknown({
                    "unknown_id": unknown_id,
                    "camera_id": self.camera_id,
                    "face_crop": body_crop,
                    "context_crop": head_crop,
                    "full_frame": full_frame_copy,
                    "encoding": None,  # No face encoding available
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "bounding_box": {
                        "top": pb["y1"], "right": pb["x2"],
                        "bottom": pb["y2"], "left": pb["x1"],
                    },
                    "detection_type": "body_only",
                })

        # Clean up old body unknowns (older than 60 seconds)
        stale_ids = [
            uid for uid, udata in self._yolo_body_unknown_cache.items()
            if now - udata["last_seen"] > 60
        ]
        for uid in stale_ids:
            del self._yolo_body_unknown_cache[uid]

    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
    #  Traffic Mode Processing
    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

    def _process_frame_traffic(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame in TRAFFIC mode.
        Skips face recognition entirely; runs ANPR + TrafficMonitor.
        """
        frame = self._process_frame_traffic_logic(frame)
        self._update_fps()
        self._last_frame = frame
        return frame


    def _process_frame_hybrid(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame in HYBRID mode.
        Runs BOTH Traffic (ANPR/Safety) and Face Recognition/Tracking.
        Face detection runs FIRST so we can filter false vehicle detections.
        """
        # Keep a truly clean copy for Face Detection (must be before ANY drawing)
        clean_frame_for_face = frame.copy()
        clean_frame = clean_frame_for_face

        # ── Compute streaming scale factors (once per stream) ─────────
        if not self._stream_res_initialized:
            h, w = frame.shape[:2]
            if w > self._stream_max_w or h > self._stream_max_h:
                scale = min(self._stream_max_w / w, self._stream_max_h / h)
                self._stream_scale_x = scale
                self._stream_scale_y = scale
            else:
                self._stream_scale_x = 1.0
                self._stream_scale_y = 1.0
            self._stream_res_initialized = True

        # Hazard detection (if enabled)
        if self.hazard_detector:
             try:
                 self.hazard_detector.submit_frame(clean_frame, self.camera_id)
             except Exception as e:
                 logger.debug(f"Hazard detection error: {e}")

        # ── 1. Run Face Detection FIRST (on clean frame) ──
        face_locations = []
        embeddings = []
        rgb_frame = None

        if self._frame_count % settings.FRAME_SKIP == 0:
            face_locations, embeddings, rgb_frame = self.detector.detect_faces_with_embeddings(clean_frame)

        # ── 2. Pass face locations to traffic module so it can filter overlaps ──
        if self.traffic_monitor and face_locations:
            # Convert face locations (top, right, bottom, left) to bbox format
            face_bboxes = []
            for top, right, bottom, left in face_locations:
                face_bboxes.append({"x1": left, "y1": top, "x2": right, "y2": bottom})
            self.traffic_monitor._known_face_zones = face_bboxes
        elif self.traffic_monitor:
            self.traffic_monitor._known_face_zones = []

        # ── 3. Run Traffic Modules (they draw on frame) ──
        frame = self._process_frame_traffic_logic(frame)

        # ── 4. Face Recognition & Tracking ──
        if self._frame_count % settings.FRAME_SKIP != 0:
            # Create display frame and draw cached boxes on it
            from app.utils.image_utils import resize_frame
            display_frame = resize_frame(frame, self._stream_max_w, self._stream_max_h)
            for draw_args in self._last_face_draws:
                self._draw_box(display_frame, *draw_args)
            self._update_fps()
            self._last_frame = display_frame
            return display_frame

        # Clear cache for this new detection frame
        self._last_face_draws.clear()

        if not face_locations:
            self._run_safety_analytics()
            self._check_periodic_exits()
            self._update_fps()
            from app.utils.image_utils import resize_frame
            display_frame = resize_frame(frame, self._stream_max_w, self._stream_max_h)
            self._last_frame = display_frame
            return display_frame

        # Collect all face boxes to draw
        boxes_to_draw = []

        # Recognition & Tracking (using embeddings from detection pass)
        # Track whether we should run ReID this frame
        run_reid = (
            self.reid_tracker
            and self._frame_count - self._last_reid_frame >= self._reid_frame_interval
        )

        for i, (location, encoding) in enumerate(zip(face_locations, embeddings)):
             if encoding is None: continue
             top, right, bottom, left = location
             centroid = ((left + right) // 2, (top + bottom) // 2)

             result = self.recognizer.recognize_face(
                 encoding,
                 face_quality=self._face_quality_for(clean_frame, location),
             )
             if result.matched:
                 # Known - Draw on the ACCUMULATED frame
                 boxes_to_draw.append((location, result.person_name, (0, 255, 0), result.confidence))
                 self.tracker.on_detection(
                     person_id=result.person_id,
                     person_name=result.person_name,
                     camera_id=self.camera_id,
                     confidence=result.confidence,
                     centroid=centroid,
                 )
                 # Attributes
                 if self.attribute_recognizer:
                     try:
                         self.attribute_recognizer.maybe_extract(result.person_id, clean_frame, location, self._handle_attributes_complete)
                     except Exception: pass

                 # ReID: extract body features for cross-camera tracking
                 if run_reid:
                     try:
                         # Expand face box to approximate body region
                         face_h = bottom - top
                         body_top = top
                         body_bottom = min(frame.shape[0], bottom + face_h * 3)  # Body ≈ 4x face height
                         body_left = max(0, left - face_h // 2)
                         body_right = min(frame.shape[1], right + face_h // 2)
                         body_bbox = (body_left, body_top, body_right, body_bottom)
                         self.reid_tracker.process_person(
                             clean_frame, body_bbox, self.camera_id,
                             face_person_id=result.person_id,
                             face_person_name=result.person_name,
                         )
                     except Exception as e:
                         logger.debug(f"ReID known person error: {e}")
             else:
                 # Unknown - Draw on the ACCUMULATED frame
                 boxes_to_draw.append((location, "UNKNOWN", (0, 0, 255), 0.0))
                 self._handle_unknown_face(clean_frame, clean_frame, location, encoding, centroid)

                 # ReID: also track unknown persons for cross-camera matching
                 if run_reid:
                     try:
                         face_h = bottom - top
                         body_top = top
                         body_bottom = min(frame.shape[0], bottom + face_h * 3)
                         body_left = max(0, left - face_h // 2)
                         body_right = min(frame.shape[1], right + face_h // 2)
                         body_bbox = (body_left, body_top, body_right, body_bottom)
                         self.reid_tracker.process_person(
                             clean_frame, body_bbox, self.camera_id,
                         )
                     except Exception as e:
                         logger.debug(f"ReID unknown person error: {e}")

        if run_reid:
            self._last_reid_frame = self._frame_count

        # Scale boxes to display resolution and draw on display frame
        sx = self._stream_scale_x
        sy = self._stream_scale_y
        scaled_boxes = []
        for (location, label, color, conf) in boxes_to_draw:
            top, right, bottom, left = location
            scaled_loc = (
                int(top * sy), int(right * sx),
                int(bottom * sy), int(left * sx),
            )
            scaled_boxes.append((scaled_loc, label, color, conf))

        self._last_face_draws = scaled_boxes

        from app.utils.image_utils import resize_frame
        display_frame = resize_frame(frame, self._stream_max_w, self._stream_max_h)
        for draw_args in scaled_boxes:
            self._draw_box(display_frame, *draw_args)

        # YOLO-assisted detection (for HYBRID mode)
        now_assist = time.time()
        if now_assist - self._last_yolo_assist_check >= self._yolo_assist_min_interval:
            self._last_yolo_assist_check = now_assist
            try:
                yolo_persons = self._get_yolo_person_detections_from_traffic()
                if not yolo_persons:
                    yolo_persons = self._get_yolo_person_detections(clean_frame)
                if yolo_persons:
                    missed = self._find_missed_persons(yolo_persons, face_locations)
                    for person_det in missed:
                        self._attempt_yolo_assisted_detection(
                            frame, clean_frame, person_det
                        )
            except Exception as e:
                logger.debug(f"YOLO-assisted detection (hybrid) error: {e}")

        # Analytics
        self._run_safety_analytics()
        self._check_periodic_exits()
        self._update_fps()
        self._last_frame = display_frame
        return display_frame

    def _process_frame_traffic_logic(self, frame: np.ndarray) -> np.ndarray:
        """Helper to run traffic logic and return modified frame."""
        clean_frame = frame.copy()
        predetected_vehicles = None

        # Traffic Monitor (Roboflow/RF-DETR/YOLO vehicle detection)
        if self.traffic_monitor:
            try:
                self.traffic_monitor.submit_frame(clean_frame, self.camera_id)
                frame = self.traffic_monitor.draw_detections(frame)

                # Grab the latest vehicle detections to share with ANPR
                # This avoids ANPR running a separate YOLO vehicle detection
                with self.traffic_monitor._detections_lock:
                    vehicles = self.traffic_monitor._last_detections.get("vehicles", [])
                    if vehicles:
                        predetected_vehicles = list(vehicles)
            except Exception as e:
                logger.debug(f"Traffic monitor error: {e}")

        # ANPR (pass pre-detected vehicles if available)
        if self.plate_recognizer:
            try:
                self.plate_recognizer.submit_frame(
                    clean_frame, self.camera_id,
                    predetected_vehicles=predetected_vehicles,
                )
            except Exception as e:
                logger.debug(f"ANPR error: {e}")
        return frame

    @property
    def status(self) -> dict:
        base = {
            "camera_id": self.camera_id,
            "running": self._running,
            "fps": self.fps,
            "frames_processed": self._frame_count,
            "pipeline_mode": self.pipeline_mode,
        }

        if self.pipeline_mode == "traffic":
            # Traffic mode stats
            if self.traffic_monitor:
                base["traffic_monitor"] = self.traffic_monitor.stats
            if self.plate_recognizer:
                base["plate_recognizer"] = self.plate_recognizer.stats
        elif self.pipeline_mode == "hybrid":
            # Hybrid stats
            if self.traffic_monitor:
                base["traffic_monitor"] = self.traffic_monitor.stats
            if self.plate_recognizer:
                base["plate_recognizer"] = self.plate_recognizer.stats
            base["faces_in_cache"] = self.recognizer.known_count
            base["active_tracks"] = self.tracker.stats
            base["unknown_cache_size"] = len(self._unknown_cache)
        else:
            # Face mode stats
            base["faces_in_cache"] = self.recognizer.known_count
            base["active_tracks"] = self.tracker.stats
            base["unknown_cache_size"] = len(self._unknown_cache)
            if self.hazard_detector:
                base["hazard_detector"] = self.hazard_detector.stats

        base["safety_modules"] = {
            "attribute_recognition": settings.ENABLE_ATTRIBUTE_RECOGNITION,
            "crowd_detection": settings.ENABLE_CROWD_DETECTION,
            "loitering_detection": settings.ENABLE_LOITERING_DETECTION,
            "hazard_detection": settings.ENABLE_HAZARD_DETECTION,
            "pipeline_mode": self.pipeline_mode,
            "anpr": self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_ANPR,
            "traffic_monitor": self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_TRAFFIC_MONITOR,
            "reid": settings.ENABLE_REID,
        }
        if self.reid_tracker:
            base["reid"] = self.reid_tracker.stats
        return base
