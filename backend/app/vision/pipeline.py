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
import pickle
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
from app.vision.bytetrack import ByteTracker, STrack

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

# Ã¢â€â‚¬Ã¢â€â‚¬ Internal: Utilities Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
from app.utils.image_utils import frame_to_jpeg_bytes, save_snapshot

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
        self._recognition_interval = 5  # Run ArcFace every N frames per track

        # Ã¢â€â‚¬Ã¢â€â‚¬ YOLO Person Detector (for body-level person detection) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # Used in FACE mode to assist face detection when face_recognition
        # misses faces (e.g. partial occlusion, small faces, angled faces).
        # In HYBRID mode, we reuse the TrafficMonitor's YOLO detections.
        self._yolo_person_detector = None
        self._yolo_person_loaded = False
        self._yolo_person_lock = threading.Lock()
        self._yolo_body_unknown_cache: Dict[str, dict] = {}  # Dedup for body-only unknowns
        self._last_yolo_assist_check = 0.0  # Throttle: last time we ran YOLO-assist
        self._yolo_assist_interval = 3.0    # Only run YOLO-assisted every 3 seconds

        # Ã¢â€â‚¬Ã¢â€â‚¬ Safety & Security Modules Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
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

        # Ã¢â€â‚¬Ã¢â€â‚¬ Traffic Mode Modules Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        self.pipeline_mode = settings.PIPELINE_MODE.lower()

        # ANPR (License Plate Recognition)
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
            )
            logger.info(f"[{camera_id}] Traffic Monitor enabled (proximity={settings.TRAFFIC_PROXIMITY_PX}px)")

        # Pipeline state
        self._frame_count = 0
        self._last_frame = None  # Last processed frame (for MJPEG streaming)
        self._last_jpeg = None   # Pre-encoded JPEG bytes (ready-to-send, encoded in worker thread)
        self._last_exit_check = time.time()
        self._exit_check_interval = 2  # Check exits every 2 seconds for faster response
        self._unknown_cache: Dict[str, dict] = {}  # Temp unknown face dedup
        self._last_safety_check = time.time()
        self._safety_check_interval = 1.0  # Run crowd/loiter checks every 1s
        self._last_face_draws: List[Tuple] = [] # Cache for drawing face boxes on skipped frames

        # Performance metrics
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

        logger.info(f"Pipeline created for camera {camera_id}: source={stream_source}, mode={self.pipeline_mode}")

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
        if not self._running or not self._capture:
            return None

        ret, frame = self._capture.read()
        if not ret:
            logger.warning(f"Camera {self.camera_id}: Failed to read frame")
            return None

        self._frame_count += 1

        # Ã¢â€â‚¬Ã¢â€â‚¬ TRAFFIC MODE Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if self.pipeline_mode == "traffic":
            return self._process_frame_traffic(frame)

        # Ã¢â€â‚¬Ã¢â€â‚¬ HYBRID MODE Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if self.pipeline_mode == "hybrid":
            return self._process_frame_hybrid(frame)

        # Ã¢â€â‚¬Ã¢â€â‚¬ FACE MODE (default) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # Keep a clean copy for snapshots (before drawing boxes)
        clean_frame = frame.copy()

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

        # Frame skip for performance
        if self._frame_count % settings.FRAME_SKIP != 0:
            # Draw previously cached boxes so they don't flicker
            for draw_args in self._last_face_draws:
                self._draw_box(frame, *draw_args)
            self._last_frame = frame
            return frame

        # Clear cache for this new detection frame
        self._last_face_draws.clear()

        # Ã¢â€â‚¬Ã¢â€â‚¬ DETECTION Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # Detect faces AND get embeddings in a SINGLE insightface pass
        face_locations, embeddings, rgb_frame = self.detector.detect_faces_with_embeddings(frame)

        if not face_locations:
            # Update ByteTrack with empty detections
            self.byte_tracker.update(np.empty((0, 4)), np.empty(0))
            self._run_safety_analytics()
            self._check_periodic_exits()
            self._update_fps()
            self._last_frame = frame
            return frame

        logger.info(f"Camera {self.camera_id}: Detected {len(face_locations)} face(s)")

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
        active_tracks = self.byte_tracker.update(bboxes_array, scores_array)

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

            # Only run recognition if track has no identity or periodic re-check
            needs_recognition = (
                track.person_id is None
                or track.tracklet_len % self._recognition_interval == 0
            )

            if needs_recognition and best_embedding is not None:
                result: MatchResult = self.recognizer.recognize_face(best_embedding)

                if result.matched:
                    track.person_id = result.person_id
                    track.person_name = result.person_name
                    boxes_to_draw.append((location, result.person_name, (0, 255, 0), result.confidence))

                    self.tracker.on_detection(
                        person_id=result.person_id,
                        person_name=result.person_name,
                        camera_id=self.camera_id,
                        confidence=result.confidence,
                        centroid=centroid,
                    )

                    if self.attribute_recognizer:
                        try:
                            self.attribute_recognizer.maybe_extract(
                                track_id=result.person_id,
                                frame=clean_frame,
                                face_location=location,
                                on_complete=self._handle_attributes_complete,
                            )
                        except Exception as e:
                            logger.debug(f"Attribute extraction error: {e}")
                else:
                    boxes_to_draw.append((location, "UNKNOWN", (0, 0, 255), 0.0))
                    self._handle_unknown_face(clean_frame, rgb_frame, location, best_embedding, centroid)
            elif not needs_recognition and track.person_id and track.person_name:
                # Reuse existing identity from ByteTrack (skip ArcFace)
                boxes_to_draw.append((location, track.person_name, (0, 255, 0), 0.0))
                self.tracker.on_detection(
                    person_id=track.person_id,
                    person_name=track.person_name,
                    camera_id=self.camera_id,
                    confidence=0.0,
                    centroid=centroid,
                )
            else:
                # No embedding available or unidentified track
                label = track.person_name if track.person_name else f"Track #{track.track_id}"
                color = (0, 200, 0) if track.person_name else (255, 165, 0)
                boxes_to_draw.append((location, label, color, 0.0))

        # Cache boxes for frame skip & draw them
        self._last_face_draws = boxes_to_draw
        for draw_args in boxes_to_draw:
            self._draw_box(frame, *draw_args)

        # YOLO-ASSISTED DETECTION (catch missed persons)
        now_assist = time.time()
        if now_assist - self._last_yolo_assist_check >= self._yolo_assist_interval:
            self._last_yolo_assist_check = now_assist
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

        self._run_safety_analytics()
        self._check_periodic_exits()
        self._update_fps()

        self._last_frame = frame
        return frame


    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
    #  Safety & Security Integration
    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

    def _run_safety_analytics(self):
        """
        Run crowd and loitering detection at a throttled interval.
        Uses centroid data from the tracker Ã¢â‚¬â€ completely non-blocking
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
            f"{alert.get('subtype', 'unknown')} Ã¢â‚¬â€ {alert}"
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

    def _handle_attributes_complete(self, track_id: str, attributes: dict):
        """
        Called when attribute recognition finishes for a person.
        The attributes (gender, clothing color) can be attached to
        detection events via the metadata JSONB column.
        """
        logger.info(
            f"Attributes ready for {track_id[:8]}Ã¢â‚¬Â¦: {attributes}"
        )
        # The attributes are cached in AttributeRecognizer._results
        # and will be included in future events via get_attributes()

    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
    #  Original Pipeline Methods (unchanged logic)
    # Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

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
            face_crop = self.detector.extract_face_crop(frame, location, padding=80)
            _, snapshot_url = save_snapshot(
                face_crop, f"unknown_{unknown_id}", subdirectory="unknowns", quality=95
            )

            # Ã¢â€â‚¬Ã¢â€â‚¬ Wider context crop (head + shoulders) for better identification Ã¢â€â‚¬Ã¢â€â‚¬
            context_crop = self._extract_context_crop(frame, location)
            _, context_url = save_snapshot(
                context_crop, f"context_{unknown_id}", subdirectory="unknowns", quality=95
            )

            # Ã¢â€â‚¬Ã¢â€â‚¬ Full frame at good quality Ã¢â€â‚¬Ã¢â€â‚¬
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
        """Lazy-load a YOLO model for person detection (used in FACE mode)."""
        with self._yolo_person_lock:
            if self._yolo_person_loaded:
                return self._yolo_person_detector is not None
            try:
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
                result = self.recognizer.recognize_face(encoding)
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
                    rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
                    self._handle_unknown_face(clean_frame, rgb_frame, global_loc, encoding, centroid)

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
        DEDUP_DISTANCE = 100  # pixels
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

            # Save snapshots
            _, body_url = save_snapshot(
                body_crop, f"unknown_body_{unknown_id}", subdirectory="unknowns", quality=95
            )
            _, head_url = save_snapshot(
                head_crop, f"unknown_head_{unknown_id}", subdirectory="unknowns", quality=95
            )
            _, full_frame_url = save_snapshot(
                clean_frame, f"fullframe_{unknown_id}", subdirectory="frames", quality=90
            )

            self._yolo_body_unknown_cache[unknown_id] = {
                "centroid": centroid,
                "count": 1,
                "first_seen": now,
                "last_seen": now,
                "snapshot_path": body_url,
                "head_path": head_url,
                "full_frame_path": full_frame_url,
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

            # Emit unknown face event (using body snapshot)
            if self.on_unknown:
                self.on_unknown({
                    "unknown_id": unknown_id,
                    "camera_id": self.camera_id,
                    "snapshot_path": body_url,
                    "context_path": head_url,
                    "full_frame_path": full_frame_url,
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
        """
        # Keep a truly clean copy for Face Detection (must be before ANY drawing)
        clean_frame_for_face = frame.copy()

        # 1. Run Traffic Modules (they draw on frame first)
        frame = self._process_frame_traffic_logic(frame)

        # 2. Run Face Mode Logic
        # Keep a copy for other uses (snapshots etc) - reusing the one above
        clean_frame = clean_frame_for_face

        # Hazard detection (if enabled)
        if self.hazard_detector:
             try:
                 # Hazard detector manages its own threading/copying
                 self.hazard_detector.submit_frame(clean_frame, self.camera_id)
             except Exception as e:
                 logger.debug(f"Hazard detection error: {e}")

        # Frame skip for Face Rec (heavy)
        if self._frame_count % settings.FRAME_SKIP != 0:
            # Draw previously cached boxes so they don't flicker
            for draw_args in self._last_face_draws:
                self._draw_box(frame, *draw_args)
            self._update_fps()
            self._last_frame = frame
            return frame

        # Clear cache for this new detection frame
        self._last_face_draws.clear()

        # Face Detection + Embedding (SINGLE pass on CLEAN frame)
        face_locations, embeddings, rgb_frame = self.detector.detect_faces_with_embeddings(clean_frame)

        if not face_locations:
            self._run_safety_analytics()
            self._check_periodic_exits()
            self._update_fps()
            self._last_frame = frame
            return frame

        # Collect all face boxes to draw
        boxes_to_draw = []

        # Recognition & Tracking (using embeddings from detection pass)
        for i, (location, encoding) in enumerate(zip(face_locations, embeddings)):
             if encoding is None: continue
             top, right, bottom, left = location
             centroid = ((left + right) // 2, (top + bottom) // 2)

             result = self.recognizer.recognize_face(encoding)
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
             else:
                 # Unknown - Draw on the ACCUMULATED frame
                 boxes_to_draw.append((location, "UNKNOWN", (0, 0, 255), 0.0))
                 self._handle_unknown_face(clean_frame, rgb_frame, location, encoding, centroid)

        # Cache boxes for frame skip & draw them
        self._last_face_draws = boxes_to_draw
        for draw_args in boxes_to_draw:
            self._draw_box(frame, *draw_args)

        # Ã¢â€â‚¬Ã¢â€â‚¬ YOLO-ASSISTED DETECTION (for HYBRID mode) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # Reuse TrafficMonitor's cached person detections (zero inference cost).
        # Throttled to every N seconds to avoid expensive crop-face-detection.
        now_assist = time.time()
        if now_assist - self._last_yolo_assist_check >= self._yolo_assist_interval:
            self._last_yolo_assist_check = now_assist
            try:
                yolo_persons = self._get_yolo_person_detections_from_traffic()
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
        self._last_frame = frame
        return frame

    def _process_frame_traffic_logic(self, frame: np.ndarray) -> np.ndarray:
        """Helper to run traffic logic and return modified frame."""
        clean_frame = frame.copy()
        # Traffic Monitor
        if self.traffic_monitor:
            try:
                self.traffic_monitor.submit_frame(clean_frame, self.camera_id)
                frame = self.traffic_monitor.draw_detections(frame)
            except Exception as e:
                logger.debug(f"Traffic monitor error: {e}")
        # ANPR
        if self.plate_recognizer:
            try:
                self.plate_recognizer.submit_frame(clean_frame, self.camera_id)
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
        }
        return base
