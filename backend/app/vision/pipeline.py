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
from app.vision.bytetrack import ByteTracker, STrack, IdentityState, VehicleByteTracker, VehicleSTrack
from app.vision.identity import IdentityLedger
from app.vision.unknown_registry import UnknownRegistry

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
from app.vision.unknown_registry import make_label as make_unknown_label
from app.vision.stream_health import StreamHealthMonitor, StreamState

# Ã¢â€â‚¬Ã¢â€â‚¬ Internal: Utilities Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
from app.utils.image_utils import frame_to_jpeg_bytes, save_snapshot
from app.core.profiler import profiler

logger = logging.getLogger(__name__)

# -- Torch CUDA is a start-up requirement, not a preference ----------------
# YOLO person detection, ReID and the hazard detector all run through torch.
# On the CPU they do not degrade gracefully, they stop being real time -- and
# they do it without raising, so the deployment looks healthy while analysing
# almost nothing. Same reasoning as the onnxruntime check in detector.py; see
# settings.ALLOW_CPU_INFERENCE for the measured cost.
try:
    import torch
    _cuda_available = torch.cuda.is_available()
    _gpu_name = torch.cuda.get_device_name(0) if _cuda_available else "N/A"
    logger.info(
        f"GPU status: CUDA={'available' if _cuda_available else 'NOT available'}, "
        f"device={_gpu_name}"
    )
except ImportError:
    _cuda_available = False
    _gpu_name = "N/A"
    logger.error("PyTorch is not installed; GPU acceleration is unavailable")

if not _cuda_available and not getattr(settings, "ALLOW_CPU_INFERENCE", False):
    raise RuntimeError(
        "torch reports no CUDA device, so YOLO person detection, ReID and "
        "hazard detection would run on the CPU. That is not a slower "
        "pipeline, it is one that silently misses most of what it sees. "
        "Install a CUDA build of torch, or set ALLOW_CPU_INFERENCE=true to "
        "run without a GPU on purpose."
    )


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
            on_unidentified_entry=self._handle_unidentified_event,
            on_unidentified_exit=self._handle_unidentified_event,
        )

        # ByteTrack frame-to-frame face tracker
        # Maintains track IDs across frames, reducing the need to run
        # expensive ArcFace recognition on every single frame.
        self.byte_tracker = ByteTracker()

        # What ByteTrack tracks: person boxes ("person") or face boxes
        # ("face"). See settings.TRACKING_MODE. Mutable, because a missing
        # YOLO model demotes this to "face" at runtime.
        self._tracking_mode = getattr(settings, "TRACKING_MODE", "person").lower()
        if self._tracking_mode not in ("person", "face"):
            logger.warning(
                f"[{self.camera_id}] unknown TRACKING_MODE "
                f"{self._tracking_mode!r}; using 'face'"
            )
            self._tracking_mode = "face"
        # Basic has no face detector, so face tracking would have nothing to
        # track. Force person tracking rather than starting a camera that
        # silently sees nobody.
        if not settings.PLAN_ALLOWS_FACE and self._tracking_mode != "person":
            logger.info(
                f"[{self.camera_id}] plan={settings.plan} has no face detection; "
                f"TRACKING_MODE forced to 'person'"
            )
            self._tracking_mode = "person"
        self._person_fallback_logged = False
        # Step 2 vote thresholds, read once (this is the per-frame hot path).
        self._promote_votes = getattr(settings, "IDENTITY_PROMOTE_VOTES", 2)
        self._promote_window = getattr(settings, "IDENTITY_PROMOTE_WINDOW_SEC", 8.0)
        self._promote_min_span = getattr(settings, "IDENTITY_PROMOTE_MIN_SPAN_SEC", 0.5)
        self._demote_votes = getattr(settings, "IDENTITY_DEMOTE_VOTES", 3)
        self._demote_window = getattr(settings, "IDENTITY_DEMOTE_WINDOW_SEC", 10.0)
        self._unknown_confirm_votes = getattr(settings, "UNKNOWN_CONFIRM_VOTES", 3)
        self._unknown_confirm_window = getattr(settings, "UNKNOWN_CONFIRM_WINDOW_SEC", 6.0)
        self._unknown_confirm_span = getattr(settings, "UNKNOWN_CONFIRM_MIN_SPAN_SEC", 1.0)
        self._frame_person_dets: List[dict] = []

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

        # Stable identities for unenrolled people, seeded from unknown_faces.
        # Replaces the old _unknown_cache, which minted a fresh uuid4() per
        # sighting session, was scanned linearly per face, and was never pruned.
        self.unknown_registry = UnknownRegistry(
            match_threshold=settings.UNKNOWN_SIMILARITY_THRESHOLD,
            max_entries=settings.UNKNOWN_REGISTRY_MAX_ENTRIES,
            sighting_cooldown=settings.UNKNOWN_SIGHTING_COOLDOWN_SEC,
            refresh_interval=settings.UNKNOWN_REGISTRY_REFRESH_SEC,
            max_views=settings.UNKNOWN_MAX_VIEWS,
            learn_min_distance=settings.UNKNOWN_LEARN_MIN_DISTANCE,
        )

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
        if settings.ENABLE_HAZARD_DETECTION and settings.PLAN_ALLOWS_WEAPONS:
            # RFDETR_MODEL_SIZE and the backend choice are passed EXPLICITLY.
            # They were not before: HazardDetector and TrafficMonitor each
            # defaulted the size to "large" in their own signature, so
            # settings.RFDETR_MODEL_SIZE was a knob wired to nothing.
            self.hazard_detector = HazardDetector(
                model_path=getattr(settings, "WEAPON_MODEL_PATH", None)
                           or settings.HAZARD_MODEL_PATH,
                frame_interval=settings.HAZARD_FRAME_INTERVAL,
                alert_cooldown_sec=settings.HAZARD_ALERT_COOLDOWN_SEC,
                on_alert=self._handle_security_alert,
                rfdetr_model_size=settings.RFDETR_MODEL_SIZE,
                prefer=getattr(settings, "HAZARD_BACKEND", "yolo"),
            )
            logger.info(f"[{camera_id}] Hazard Detection enabled")

        # ── Traffic Mode Modules ─────────────────────────────────────────────────────────────
        self.pipeline_mode = settings.PIPELINE_MODE.lower()

        # ANPR (License Plate Recognition)
        self._is_browser_camera = stream_source.startswith("browser:")
        self.plate_recognizer: Optional[PlateRecognizer] = None
        if (self.pipeline_mode in ("traffic", "hybrid")
                and settings.ENABLE_ANPR and settings.PLAN_ALLOWS_ANPR):
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
                rfdetr_model_size=settings.RFDETR_MODEL_SIZE,
            )
            logger.info(f"[{camera_id}] Traffic Monitor enabled (proximity={settings.TRAFFIC_PROXIMITY_PX}px)")

        # ── Person Re-Identification (cross-camera tracking) ──────────────────────────────────
        self.reid_tracker: Optional[ReIDTracker] = None
        if settings.ENABLE_REID and settings.PLAN_ALLOWS_REID:
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
        # NOTE: _unknown_cache is gone -- replaced by self.unknown_registry.
        # It minted a fresh uuid4() per sighting session, was scanned
        # linearly per unknown face, and was never pruned.
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
        # Frames decoded and then dropped because nothing wanted them.
        self._grabber_dropped = 0

        # ── Stream Health Monitor ─────────────────────────────────────────
        # Detects frozen/dead streams and manages auto-reconnection.
        # Runs inside the grabber thread — never blocks the main pipeline.
        self.stream_health = StreamHealthMonitor(
            camera_id=camera_id,
            on_event=on_event,
        )

        # -- Analysis rate (item 6) --------------------------------
        # The rate this camera is SUPPOSED to analyse at, in frames per
        # second, resolved once: explicit per-camera config, else the gate
        # default if this is a gate camera, else the global default.
        self._analysis_fps = self._resolve_analysis_fps()
        self._analysis_interval = (
            1.0 / self._analysis_fps if self._analysis_fps > 0 else 0.0
        )
        # perf_counter() of the last analysed frame. Starts at zero so the first
        # frame to arrive is analysed rather than waiting out an interval.
        self._last_analysis_at = 0.0

        # Performance metrics
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        # Monotonic count of ANALYSIS frames (frames that ran detection, not
        # frames decoded). `_fps` above is a one-second sliding rate and is
        # therefore stale on a camera that has stalled -- which is exactly the
        # camera a load test needs to see. This counter cannot lie about that.
        self._analysis_frames = 0

        # Auto-refresh encodings: safety net for out-of-process enrollment (e.g., seed_teachers.py)
        # Normal enrollment triggers instant reload via camera_manager.load_encodings_all()
        self._last_encoding_refresh = 0.0  # Force immediate check on first face detection
        self._encoding_refresh_interval = 300.0  # Safety-net re-check every 5 minutes
        # How many rows are loaded is asked of the recognizer, not tracked here:
        # CameraManager.start_camera() loads the cached gallery into this
        # pipeline before the first frame, and a counter owned by the pipeline
        # cannot see that. It started at 0, so the first check always disagreed
        # with the DB and reloaded a gallery that was already resident --
        # visible in the logs as two "Loaded N face encodings" lines
        # (version=1, version=2) seconds apart at every camera start.

        # Embedding suggestion cooldown: per-person throttle
        self._suggestion_cooldowns: Dict[str, float] = {}  # person_id → last_suggestion_time

        logger.info(f"Pipeline created for camera {camera_id}: source={stream_source}, mode={self.pipeline_mode}")

    def _frame_wanted(self) -> bool:
        """Is there any consumer for the frame just read?

        Two of them: the analyser, when its interval has elapsed, and a
        viewer, who wants display frames at the camera's own rate regardless
        of the analysis rate. With neither, the frame is decoded and dropped.
        """
        # LEGACY FRAME_SKIP MODE MUST KEEP EVERY FRAME.
        # There, "due" means `_frame_count % skip == 0`, and _frame_count only
        # advances inside process_frame. Asking here before anything has been
        # consumed freezes the counter: frame 1 fails the test, so the grabber
        # drops it, so process_frame never runs, so the counter never moves,
        # so every later frame fails the same test. The pipeline deadlocks
        # after exactly one frame -- measured as 0.00 analysis fps and 1.3%
        # GPU across all 20 cameras, which is how this was found.
        #
        # Only a wall-clock rate can be evaluated by a thread that does not
        # consume, because only the clock advances on its own.
        if not getattr(settings, "ANALYSIS_RATE_BY_CLOCK", True):
            return True
        if self._analysis_due():
            return True
        try:
            from app.core.websocket import ws_manager
            return ws_manager.is_watched(self.camera_id)
        except Exception:
            return True  # never drop a frame over a bookkeeping error

    def _grabber_loop(self):
        """
        Background thread: continuously reads frames from cv2.VideoCapture.
        Always overwrites _grabbed_frame with the latest frame so that
        process_frame() never reads a stale buffered frame.

        Also acts as a watchdog:
        - Detects read failures and frozen streams (identical frame hash)
        - Transitions stream_health through LIVE → STALE → RECONNECTING
        - Reconnects with exponential backoff
        - Clears _last_jpeg on STALE so WebSocket never serves stale images
        """
        while not self._grabber_stop.is_set():
            # ── Reconnection Logic ────────────────────────────────
            if self.stream_health.should_reconnect():
                self.stream_health.on_reconnect_attempt()
                try:
                    if self._capture:
                        self._capture.release()
                        self._capture = None
                    source = (
                        int(self.stream_source)
                        if self.stream_source.isdigit()
                        else self.stream_source
                    )
                    self._capture = cv2.VideoCapture(source)
                    if self._capture.isOpened():
                        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self._capture.set(cv2.CAP_PROP_FPS, 30)
                        self.stream_health.on_reconnect_success()
                    else:
                        logger.warning(
                            f"[{self.camera_id}] Reconnect failed: "
                            f"capture could not open"
                        )
                        if self._capture:
                            self._capture.release()
                            self._capture = None
                except Exception as e:
                    logger.error(
                        f"[{self.camera_id}] Reconnect error: {e}"
                    )
                continue

            if not self._capture or not self._capture.isOpened():
                time.sleep(0.01)
                self.stream_health.check_staleness()
                continue

            ret, frame = self._capture.read()
            if ret:
                is_new = self.stream_health.on_frame_read(frame)
                if is_new:
                    # Item 6: drop here rather than downstream. A frame that is
                    # neither due for analysis nor wanted by a viewer has no
                    # consumer at all, and carrying it costs a 1080p resize, a
                    # redraw and a JPEG slot handoff in process_frame.
                    #
                    # The READ above always happens. An RTSP connection has to
                    # be drained or it backs up, and a decoder fed
                    # discontiguous frames produces artefacts -- so this skips
                    # the WORK, never the decode, which is also why it cannot
                    # break tracking.
                    if self._frame_wanted():
                        with self._grab_lock:
                            self._grabbed_frame = frame
                    else:
                        self._grabber_dropped += 1
            else:
                self.stream_health.on_read_failure()
                # Brief pause on read failure to avoid busy-spin
                time.sleep(0.005)

            # ── Clear stale JPEG when stream dies ─────────────────
            # MANDATORY: no stale image should ever remain visible.
            if self.stream_health.state != StreamState.LIVE:
                self._last_jpeg = None

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
            self.stream_health.reset()
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

        # Keep the unknown registry in sync so identities created by ANOTHER
        # camera (or before this process started) are recognised as the same
        # person here. Self-throttling; a no-op on most frames.
        try:
            from app.services.unknown_service import UnknownService
            self.unknown_registry.maybe_refresh(UnknownService.get_pending_encodings)
            # Apply canonical row ids the EventQueue consumer reported back.
            # Cheap: an empty queue drain, on all but a handful of frames.
            self.unknown_registry.drain_persistence_results()
        except Exception as e:
            logger.debug(f"[{self.camera_id}] Unknown registry refresh skipped: {e}")

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

        # Analysis rate: wall clock (item 6), or legacy frame count.
        if not self._analysis_due(consume=True):
            # Not an analysis frame, but still a DISPLAY frame: redraw the
            # boxes the tracker last knew about, so a viewer sees continuous
            # overlays at the camera's frame rate rather than at the
            # analysis rate.
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
        if settings.PLAN_ALLOWS_FACE:
            face_locations, embeddings, face_landmarks, rgb_frame = (
                self.detector.detect_faces_with_embeddings(frame)
            )
        else:
            # Basic plan: no SCRFD, no ArcFace. Person mode treats "no faces
            # this frame" as the ordinary case -- a person facing away -- so
            # everything downstream already handles it: the tracker follows
            # bodies, _bind_face_to_track returns None, recognition is
            # skipped, and presence, zones, crowd and loitering are unaffected.
            face_locations, embeddings, face_landmarks, rgb_frame = [], [], [], None

        # In PERSON mode "no faces" is an ordinary frame, not an empty one:
        # people with their backs turned still have bodies to track. Only face
        # mode can conclude there is nothing here, because there the face IS
        # the track.
        if not face_locations and self._tracking_mode != "person":
            # Update ByteTrack with empty detections
            self.byte_tracker.update(np.empty((0, 4)), np.empty(0))

            # Coasting: zero faces this frame does NOT mean zero people. Every
            # track just moved toward `lost` still holds its identity and is
            # still a re-association candidate. This is the exact frame the box
            # used to vanish on -- someone turned their head and the whole
            # overlay went blank. Draw what the tracker still believes.
            coast = self._scale_boxes(self._coasting_boxes())
            self._last_face_draws = coast  # so frame-skip redraws keep them
            with profiler.measure("drawing"):
                for draw_args in coast:
                    self._draw_box(display_frame, *draw_args)

            self._run_safety_analytics()
            self._check_periodic_exits()
            self._update_fps()
            self._last_frame = display_frame
            return display_frame

        logger.debug(
            f"Camera {self.camera_id}: {len(face_locations)} face(s), "
            f"mode={self._tracking_mode}"
        )

        # What drives the tracker this frame: person boxes (Step 1) or face
        # boxes (legacy). `track_source` then decides how a face is bound back
        # to a track, since the two need different geometry.
        with profiler.measure("person_detect"):
            bboxes_array, scores_array, track_source = self._tracking_detections(
                clean_frame, face_locations
            )

        # ByteTrack update: assigns stable track IDs across frames
        with profiler.measure("bytetrack"):
            active_tracks = self.byte_tracker.update(bboxes_array, scores_array)

        # ── Identity Ledger (Milestone 1: OBSERVE-ONLY) ──────────
        # Runs on the tracker's OUTPUT, so ByteTrack stays geometry-only.
        # Currently measures only: when a track is recreated, did a recoverable
        # predecessor exist? It does not assign identity yet.
        self._observe_identity_continuity(active_tracks)

        # Collect all boxes to draw
        boxes_to_draw = []

        for track in active_tracks:
            # THE BOX WE DRAW vs THE BOX WE RECOGNISE.
            # In person mode `track.to_location()` is a BODY box: right for
            # drawing, for the centroid, and for presence -- and useless for
            # anything that needs pixels of a face. Those two were the same
            # rectangle before Step 1, and conflating them now would feed body
            # crops to ArcFace and pose estimation.
            draw_location = track.to_location()  # (top, right, bottom, left)
            d_top, d_right, d_bottom, d_left = draw_location
            centroid = ((d_left + d_right) // 2, (d_top + d_bottom) // 2)

            face_location, best_embedding, best_landmarks = self._bind_face_to_track(
                track, face_locations, embeddings, face_landmarks, track_source
            )

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
            # Measured on the FACE box, never the track box: a person box is
            # always wider than MIN_FACE_RECOGNIZE_PX, so measuring the track
            # would pass the gate on a body and hand ArcFace a torso.
            #
            # No bound face is not a failure -- it is someone facing away. The
            # track survives with whatever identity it already had, which is
            # the entire point of Step 1.
            if face_location is None:
                face_too_small = True
            else:
                face_w = abs(face_location[1] - face_location[3])  # right - left
                face_h = abs(face_location[2] - face_location[0])  # bottom - top
                face_too_small = (face_w < settings.MIN_FACE_RECOGNIZE_PX or
                                  face_h < settings.MIN_FACE_RECOGNIZE_PX)

            if face_too_small:
                # No usable face this frame: either none was bound (the person
                # is facing away) or it is too small to embed. Recognition is
                # skipped either way -- but the person is still demonstrably
                # HERE, which is the whole Step 1 distinction.
                if track.person_name:
                    label, color = track.person_name, (0, 200, 0)
                elif track.unknown_label:
                    label, color = track.unknown_label, (0, 140, 255)
                else:
                    label, color = f"Track #{track.track_id}", (255, 165, 0)
                boxes_to_draw.append((draw_location, label, color, 0.0))

                if track.person_id:
                    # Established identity, still visible as a body. Refresh
                    # presence: before Step 1 this person would have aged out
                    # toward EXIT_THRESHOLD while standing in frame with their
                    # back turned. Confidence carries over from the last real
                    # match -- we are asserting presence, not re-asserting a
                    # recognition we did not perform.
                    self.tracker.on_detection(
                        person_id=track.person_id,
                        person_name=track.person_name or label,
                        camera_id=self.camera_id,
                        confidence=track.recognition_confidence,
                        centroid=centroid,
                    )
                elif track.unknown_id:
                    # A stranger we can nonetheless tell apart: UnknownRegistry
                    # gave them a stable id that survives restarts and is the
                    # same on every camera. They are identified, just not known,
                    # so they keep refreshing presence under that id rather than
                    # decaying into an anonymous body.
                    self.tracker.on_detection(
                        person_id=track.unknown_id,
                        person_name=track.unknown_label or label,
                        camera_id=self.camera_id,
                        confidence=0.0,
                        centroid=centroid,
                    )
                else:
                    # No identity of any kind. Recorded as an UNIDENTIFIED
                    # OCCUPANT: counted, never named, and kept in a lane of its
                    # own. The `anon:<camera>:<track>` handle this replaces was
                    # a string pretending to be a person -- it sat in
                    # PersonTracker._active_tracks beside real identities, and
                    # because it embedded the ByteTrack id it minted a fresh
                    # "person" on every track recreation, one that
                    # PersonTracker._cooldowns then held forever.
                    self.tracker.on_unidentified(
                        track_id=track.track_id,
                        camera_id=self.camera_id,
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
                        face_quality=self._face_quality_for(frame, face_location),
                    )

                if result.matched:
                    prev_person_id = track.person_id
                    prev_person_name = track.person_name

                    # Evidence first, verdict second (Step 2).
                    track.record_vote(result.person_id, result.confidence)

                    # A CONFIRMED identity is not overwritten by a single
                    # match to somebody else -- that is the identity-swap
                    # failure mode. It takes the same confident contradiction
                    # that demotion takes, accumulated below.
                    contradicts = (
                        track.identity_state == IdentityState.KNOWN
                        and track.person_id
                        and track.person_id != result.person_id
                    )
                    if contradicts and not track.can_demote(
                        self._demote_votes, self._demote_window
                    ):
                        logger.info(
                            f"[{self.camera_id}] HOLDING {track.person_name} on "
                            f"T{track.track_id} against {result.person_name} "
                            f"(conf={result.confidence:.2f}): "
                            f"{len(track.votes_against(track.person_id, self._demote_window))}"
                            f"/{self._demote_votes} contradicting votes"
                        )
                        boxes_to_draw.append((
                            draw_location, track.person_name, (0, 200, 0),
                            track.recognition_confidence,
                        ))
                        self.tracker.on_detection(
                            person_id=track.person_id,
                            person_name=track.person_name,
                            camera_id=self.camera_id,
                            confidence=track.recognition_confidence,
                            centroid=centroid,
                        )
                        continue

                    # Identity swap: a track flipping straight from one known
                    # person to a different known person, without passing
                    # through unknown. Recorded HERE, after the hold check, so
                    # the metric counts swaps that happened -- not swaps the
                    # vote threshold just prevented. It is the number that says
                    # whether body-box association is mixing people up.
                    if prev_person_id and prev_person_id != result.person_id:
                        profiler.record_identity_swap(
                            prev_person_name or "?", result.person_name or "?",
                            track.track_id,
                        )
                        logger.warning(
                            f"[{self.camera_id}] IDENTITY SWAP on T{track.track_id}: "
                            f"{prev_person_name} -> {result.person_name} "
                            f"(distance={result.distance:.4f}, margin={result.margin:.4f})"
                        )

                    track.person_id = result.person_id
                    track.person_name = result.person_name

                    # A track that was a confirmed stranger and is now a named
                    # person leaves a stranded unknown identity behind. Record
                    # the link rather than orphaning it -- see
                    # _report_unknown_resolved.
                    if track.unknown_id:
                        self._report_unknown_resolved(track, result)
                    track.unknown_id = None
                    track.unknown_label = None

                    # Promotion needs several confident matches for the SAME
                    # person, spread over time. One good pose producing three
                    # consecutive agreeing frames is one observation.
                    if track.identity_state != IdentityState.KNOWN and track.can_promote(
                        result.person_id, self._promote_votes,
                        self._promote_window, self._promote_min_span,
                    ):
                        track.identity_state = IdentityState.KNOWN
                        logger.info(
                            f"[{self.camera_id}] IDENTITY CONFIRMED: "
                            f"{result.person_name} on T{track.track_id}"
                        )
                    # Update recognition cache on the track
                    track.recognition_confidence = result.confidence
                    track.recognition_time = time.time()
                    track.reset_fail_streak()
                    track.record_distance(result.distance)
                    if result.confidence >= self._recognition_lock_confidence:
                        track.recognition_locked = True

                    boxes_to_draw.append((draw_location, result.person_name, (0, 255, 0), result.confidence))

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
                                face_location=face_location,
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
                            location=face_location,
                            landmarks=best_landmarks,
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
                        boxes_to_draw.append((draw_location, track.person_name, (0, 200, 0), track.recognition_confidence))
                        self.tracker.on_detection(
                            person_id=track.person_id,
                            person_name=track.person_name,
                            camera_id=self.camera_id,
                            confidence=track.recognition_confidence,
                            centroid=centroid,
                        )
                    elif track.person_id and track.person_name and result.in_gray_zone:
                        # Gray zone: borderline failure — apply hysteresis
                        # Recorded as NON-confident: it counts toward confirming a
                        # stranger, never toward unseating a known identity.
                        track.record_vote(None, result.confidence, confident=False)
                        track.identity_fail_streak += 1
                        # ASYMMETRY. A gray-zone miss is the ambiguous middle:
                        # a usable face the recogniser could not decide on. It
                        # is enough to unseat a PROVISIONAL guess, and it is
                        # not enough to unseat a verdict that several confident
                        # matches already earned. Only confident contradiction
                        # does that, and it is counted separately.
                        if (track.identity_fail_streak < settings.IDENTITY_GRACE_FRAMES
                                or track.identity_state == IdentityState.KNOWN):
                            # Keep previous identity (grace period)
                            logger.info(
                                f"⏳ HYSTERESIS: keeping {track.person_name} "
                                f"(T{track.track_id}) despite distance={result.distance:.4f}, "
                                f"fail {track.identity_fail_streak}/{settings.IDENTITY_GRACE_FRAMES}, "
                                f"avg={track.temporal_avg_distance:.4f}"
                            )
                            # Show with slightly dimmer green to indicate borderline
                            boxes_to_draw.append((draw_location, track.person_name, (0, 200, 0), track.recognition_confidence))
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
                            # The verdict is withdrawn too. Leaving state at
                            # KNOWN with no person_id would make the next
                            # frame's promotion check skip entirely.
                            track.identity_state = IdentityState.PROVISIONAL
                            track.clear_votes()
                            boxes_to_draw.append((draw_location, "UNKNOWN", (0, 0, 255), 0.0))
                            track.unknown_id = self._handle_unknown_face(
                                clean_frame, frame, face_location, best_embedding,
                                centroid, track_id=track.track_id,
                                landmarks=best_landmarks,
                                confirmed=self._confirm_unknown(track),
                            )
                            track.unknown_label = (
                                make_unknown_label(track.unknown_id)
                                if track.unknown_id else None
                            )
                    else:
                        # Definitive failure: a usable face, far above the
                        # threshold. This IS confident contradiction, so it is
                        # recorded as a vote against whoever the track claims.
                        track.record_vote(None, result.confidence)

                        if (track.identity_state == IdentityState.KNOWN
                                and track.person_id
                                and not track.can_demote(
                                    self._demote_votes, self._demote_window)):
                            # Not yet enough contradiction to overturn a
                            # confirmed identity. Before Step 2 a single frame
                            # did this.
                            logger.info(
                                f"[{self.camera_id}] HOLDING {track.person_name} "
                                f"(T{track.track_id}) on definitive miss "
                                f"{len(track.votes_against(track.person_id, self._demote_window))}"
                                f"/{self._demote_votes}"
                            )
                            boxes_to_draw.append((
                                draw_location, track.person_name, (0, 200, 0),
                                track.recognition_confidence,
                            ))
                            self.tracker.on_detection(
                                person_id=track.person_id,
                                person_name=track.person_name,
                                camera_id=self.camera_id,
                                confidence=track.recognition_confidence,
                                centroid=centroid,
                            )
                            continue

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
                            # The verdict is withdrawn too. Leaving state at
                            # KNOWN with no person_id would make the next
                            # frame's promotion check skip entirely.
                            track.identity_state = IdentityState.PROVISIONAL
                            track.clear_votes()
                        boxes_to_draw.append((draw_location, "UNKNOWN", (0, 0, 255), 0.0))
                        track.unknown_id = self._handle_unknown_face(
                            clean_frame, frame, face_location, best_embedding,
                            centroid, track_id=track.track_id,
                            landmarks=best_landmarks,
                            confirmed=self._confirm_unknown(track),
                        )
                        track.unknown_label = (
                            make_unknown_label(track.unknown_id)
                            if track.unknown_id else None
                        )
            elif not needs_recognition and track.person_id and track.person_name:
                # Reuse existing identity from ByteTrack cache (skip ArcFace)
                profiler.record_cache_hit()
                boxes_to_draw.append((draw_location, track.person_name, (0, 255, 0), track.recognition_confidence))
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
                boxes_to_draw.append((draw_location, label, color, 0.0))

        # Coasting tracks: identities ByteTrack still holds but did not return
        # this frame (person present, face not currently detectable). Appended
        # AFTER the live boxes so a live box always wins the z-order.
        # `_tracked_stracks` and `_lost_stracks` are disjoint by construction,
        # so this cannot double-draw a track that is already above.
        boxes_to_draw.extend(self._coasting_boxes())

        # Scale bounding boxes from full-res detection space to display space.
        scaled_boxes = self._scale_boxes(boxes_to_draw)

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
                # In person mode the boxes for this frame were already
                # computed to drive the tracker; re-detecting would double the
                # per-frame YOLO cost for an identical answer.
                if self._tracking_mode == "person":
                    yolo_persons = self._frame_person_dets
                else:
                    yolo_persons = self._get_yolo_person_detections(clean_frame)
                if yolo_persons:
                    missed = self._find_missed_persons(yolo_persons, face_locations)
                    if self._tracking_mode == "person":
                        # "Person without a visible face" is the ordinary state
                        # here, not an anomaly, so this list is no longer short.
                        # Each entry costs a crop plus a full SCRFD pass; spend
                        # that on the nearest few, not on a whole corridor.
                        missed = sorted(
                            missed,
                            key=lambda d: d.get("confidence", 0.0),
                            reverse=True,
                        )[:3]
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
            loaded_count = self.recognizer.source_count

            if db_count > 0 and db_count != loaded_count:
                # DB has encodings that aren't loaded yet (or count changed)
                self.recognizer.load_encodings(encodings)
                logger.info(
                    f"[{self.camera_id}] Auto-refreshed {db_count} face encodings from DB "
                    f"(was {loaded_count} before refresh)"
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

            # Crowd Detection — only count actively-detected persons
            # Exclude stale tracks that are still in active_tracks but
            # haven't been seen recently (waiting for exit_threshold).
            # Guard on identified + unidentified together: five strangers
            # with their backs to the camera are a crowd, and none of them
            # would appear in `centroids`.
            anon_centroids = self.tracker.get_unidentified_centroids()
            if self.crowd_detector and (len(centroids) + len(anon_centroids)) >= 2:
                recent_centroids = {}
                active_tracks_snapshot = self.tracker.get_active_track_times()
                for pid, centroid in centroids.items():
                    last_det = active_tracks_snapshot.get(pid, 0)
                    if now - last_det < 2.0:  # Only count if seen in last 2 seconds
                        recent_centroids[pid] = centroid
                # Unidentified person tracks: bodies without names. Counted
                # for crowd density, deliberately NOT passed to loitering
                # below, which reasons about a specific person over time.
                recent_centroids.update(anon_centroids)
                if len(recent_centroids) >= 2:
                    self.crowd_detector.update(recent_centroids, self.camera_id)

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

        Attaches the latest encoded frame as evidence. An incident report
        without an image is not evidence, and alerts previously carried no
        snapshot at all.

        This is cheap on purpose: _last_jpeg is the ANNOTATED frame the JPEG
        encoder thread already produced for the live stream, so this is a
        reference copy of immutable bytes — no frame copy, no re-encode, and
        the saved image shows the detection boxes the operator would have
        seen. The disk write itself is deferred to the EventQueue consumer
        thread, keeping it off the vision loop entirely.

        The key is underscore-prefixed so _process_pipeline_event strips it
        before the event is persisted or broadcast — raw bytes would break
        JSON serialisation on both paths.
        """
        logger.warning(
            f"Security alert on camera {self.camera_id}: "
            f"{alert.get('subtype', 'unknown')} — {alert}"
        )

        if settings.ALERT_SNAPSHOTS_ENABLED and self._last_jpeg:
            alert["_evidence_jpeg"] = self._last_jpeg

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
        landmarks: Optional[np.ndarray] = None,
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
            # landmarks are SCRFD's 5 points for THIS detection, so estimate_pose()
            # returns a true yaw. Without them it returned (0,0,0) and every
            # suggestion bucketed as "frontal", which made the pose-coverage check
            # (step 2 of the diversity ladder) dead code.
            quality_result = gate.assess(
                face_crop=face_crop,
                face_location=location,
                det_score=confidence,
                landmarks=landmarks,
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

    def _confirm_unknown(self, track) -> bool:
        """
        Has this track earned the verdict "stranger"?

        Replaces the UNKNOWN_MATURATION_SEC timer, which a single bad angle
        held for half a second could satisfy. Confirmation now needs several
        non-matches SPREAD OVER TIME, so one unlucky pose cannot mint an
        identity for somebody the system already knows.

        This is what stops most stranded unknowns from existing at all: no
        confirmation, no unknown_faces row to strand.
        """
        if track.identity_state == IdentityState.UNKNOWN_CONFIRMED:
            return True
        if track.can_confirm_unknown(
            self._unknown_confirm_votes,
            self._unknown_confirm_window,
            self._unknown_confirm_span,
        ):
            track.identity_state = IdentityState.UNKNOWN_CONFIRMED
            return True
        track.identity_state = IdentityState.UNKNOWN_CANDIDATE
        return False

    def _report_unknown_resolved(self, track, result):
        """
        A track that was a confirmed stranger turned out to be someone enrolled.

        The `unknown_faces` row already written is now a stranger who is
        actually Rahul. It must not be silently orphaned in the review queue.

        We do NOT auto-merge it into the gallery. The repository's rule
        throughout is that the machine proposes and a human disposes, and an
        automatic merge on a single track's say-so is exactly how a gallery
        gets poisoned. Instead the observed link is recorded as an event, so
        it is durable, queryable, and available to the existing review flow --
        which already knows how to take an unknown_faces row and merge it into
        a person once somebody approves it.

        Same-track evidence is much stronger than the offline embedding
        similarity the review worker computes: these are literally the same
        person, seen continuously, within one tracklet.
        """
        payload = {
            "event_type": "detection",
            "subtype": "unknown_resolved",
            "camera_id": self.camera_id,
            "confidence": result.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "unknown_person_id": track.unknown_id,
            "resolved_person_id": result.person_id,
            "resolved_person_name": result.person_name,
            "track_id": track.track_id,
            "distance": float(result.distance),
        }
        logger.info(
            f"[{self.camera_id}] UNKNOWN RESOLVED: {track.unknown_label} "
            f"(T{track.track_id}) is {result.person_name} "
            f"(conf={result.confidence:.2f}) -- proposing merge for review"
        )
        if self.on_event:
            self.on_event(payload)

    def _handle_unknown_face(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray,
        location: tuple,
        encoding: np.ndarray,
        centroid: Optional[Tuple[int, int]] = None,
        track_id: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
        confirmed: bool = True,
    ) -> Optional[str]:
        """
        Handle an unrecognised face: resolve to a STABLE identity, track, queue.

        Returns the stable unknown id so the caller can bind it to the track.
        A stranger who then turns away keeps that id instead of decaying into
        an anonymous body -- which would count one human in both the
        identified and unidentified lanes during a single visit.

        The identity comes from UnknownRegistry, which is seeded from the
        `unknown_faces` table. That is what makes the id survive a restart and
        stay identical across cameras. Previously a fresh uuid4() was minted per
        sighting session and the label came from a per-pipeline counter, so the
        same stranger became a different "Unknown #1" every time.

        CONFIRMATION GATE (Step 2)
        `confirmed` is the caller's verdict from N-of-M voting: several
        non-matches spread over time, not one bad angle held for half a
        second. Only a confirmed stranger gets an unknown_faces row, which is
        what keeps someone who is merely hard to recognise from being minted
        as a new identity.

        Presence and the registry id are assigned either way -- an
        unconfirmed stranger still shows up in "People Present"; they just do
        not become a permanent row yet.

        Call sites with no track (YOLO-assist, hybrid) pass confirmed=True:
        they have no track state to vote with, which is the pre-Step-2
        behaviour for those paths.
        """
        unknown_id, unknown_label, is_new = self.unknown_registry.resolve(encoding)
        if unknown_id is None:
            return None  # unusable (zero-norm) embedding

        # Appears in "People Present" under the stable identity either way.
        self.tracker.on_detection(
            person_id=unknown_id,
            person_name=unknown_label,
            camera_id=self.camera_id,
            confidence=0.0,
            centroid=centroid,
        )

        if not self.on_unknown:
            return unknown_id

        # ── Confirmation gate ───────────────────────────────────────
        # Only applies to tracked faces creating a NEW identity. A sighting of
        # an identity that already exists is not in question.
        if track_id is not None and is_new and not confirmed:
            return unknown_id  # identified for this track, not yet persisted

        # Views the registry learned since the last call. They must reach the
        # database or they are lost on the next refresh -- which is exactly why
        # cross-camera re-identification measured 0.0% before migration 012.
        # Shipped independently of the sighting cooldown and of is_new: a
        # person can easily be gone before the next sighting report is due.
        new_views = self.unknown_registry.drain_new_views()
        if new_views:
            self.on_unknown({
                "views_only": True,
                "views": new_views,
                "camera_id": self.camera_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        if is_new:
            # First time anyone has seen this person. Copy the crops now: the
            # frame buffer is reused on the next iteration. These copies are
            # ~0.5ms each versus ~10-30ms for save_snapshot, which is why the
            # disk write itself stays deferred to the EventQueue consumer.
            face_crop = self.detector.extract_face_crop(frame, location, padding=80).copy()
            context_crop = self._extract_context_crop(frame, location).copy()
            full_frame_copy = frame.copy()

            # Pose + quality metadata for the row we are about to create.
            # Only computed on the first-ever sighting of this identity (the
            # only path that writes an unknown_faces row), so the cost stays
            # off the per-frame hot path. Stays None when SCRFD landmarks are
            # unavailable — a null pose is honest, a fabricated 0° yaw is not.
            pose_meta = self._pose_metadata(frame, location, landmarks)

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
                **pose_meta,
            })
        elif self.unknown_registry.should_report_sighting(unknown_id):
            # Known unknown, seen again. Report a SIGHTING so `occurrence`
            # counts real visits, throttled by UNKNOWN_SIGHTING_COOLDOWN_SEC.
            # No crops: the row already has snapshots, and writing three more
            # JPEGs per repeat sighting was orphaning files on disk.
            self.on_unknown({
                "unknown_id": unknown_id,
                "camera_id": self.camera_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sighting_only": True,
            })

        return unknown_id

    def _pose_metadata(
        self,
        frame: np.ndarray,
        location: tuple,
        landmarks: Optional[np.ndarray],
    ) -> dict:
        """
        Compute real pose + quality metadata for an unknown face crop.

        Reuses FaceQualityGate (the same gate the enrollment path uses) so
        unknown faces are described in exactly the same vocabulary as gallery
        encodings — that is what makes pose-coverage comparison meaningful.

        Returns {} when landmarks are missing: downstream code must treat a
        null pose as "unknown", never as "frontal".
        """
        if landmarks is None:
            return {}

        try:
            from app.vision.quality import FaceQualityGate
            from app.services.embedding_suggestions import classify_pose_bucket

            gate = FaceQualityGate()
            face_crop = self.detector.extract_face_crop(frame, location, padding=40)
            result = gate.assess(
                face_crop=face_crop,
                face_location=location,
                det_score=1.0,  # SCRFD already applied its own confidence filter
                landmarks=landmarks,
            )
            m = result.metrics
            yaw = m.get("yaw")
            if yaw is None:
                return {}

            return {
                "yaw": float(yaw),
                "pitch": float(m.get("pitch", 0.0)),
                "roll": float(m.get("roll", 0.0)),
                "pose_bucket": classify_pose_bucket(float(yaw)),
                "quality": float(result.quality_score),
                "blur_score": float(m.get("blur_score", 0.0)),
                "det_score": float(m.get("det_score", 0.0)),
                "lighting": m.get("lighting"),
            }
        except Exception as e:
            logger.debug(f"Pose metadata computation failed: {e}")
            return {}

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

    def _handle_unidentified_event(self, event: dict):
        """
        An unidentified occupant started or stopped being present.

        Deliberately does NOT run the attribute enrichment that
        _handle_entry_event does: that is keyed on person_id, and this event
        has none by definition.

        Emitted as event_type='detection' with an 'unidentified_*' subtype, so
        main.py routes it at NORMAL priority (droppable under load, which is
        right for occupancy telemetry) and no attendance/journey/footfall
        query can pick it up -- they all filter on event_type in
        ('entry', 'exit').
        """
        if not getattr(settings, "UNIDENTIFIED_EVENTS_ENABLED", True):
            return
        if self.on_event:
            self.on_event(event)

    def _check_periodic_exits(self):
        """Periodically check for exits."""
        now = time.time()
        if now - self._last_exit_check >= self._exit_check_interval:
            self.tracker.check_exits(now)
            self.tracker.expire_unidentified(now)
            self._last_exit_check = now

    # ==================================================================
    #  Step 1 -- tracking owns existence, recognition owns identity
    # ==================================================================

    def _tracking_detections(
        self,
        clean_frame: np.ndarray,
        face_locations: List[Tuple],
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        The boxes that drive ByteTrack this frame.

        In "person" mode these are YOLO person boxes, so a track is a PERSON
        and a face is something that person sometimes has. In "face" mode they
        are SCRFD face boxes -- the pre-Step-1 behaviour, where a person who
        turns away stops existing.

        Falls back to face boxes PERMANENTLY if the YOLO model cannot load,
        because silently tracking nothing is worse than tracking faces.

        A frame where YOLO finds no people is NOT a fallback: it returns empty
        boxes and the tracks coast. Switching detection source mid-stream would
        change what a track box MEANS, and every track id would churn.

        Returns:
            (bboxes_xyxy, scores, source) where source is "person" or "face".
        """
        if self._tracking_mode == "person":
            if self._load_yolo_person_detector():
                persons = self._get_yolo_person_detections(clean_frame)
                conf_floor = getattr(settings, "PERSON_DET_CONF", 0.35)
                boxes, scores = [], []
                for det in persons:
                    conf = det.get("confidence", 0.0)
                    if conf < conf_floor:
                        continue
                    b = det["bbox"]
                    boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                    scores.append(conf)
                # Kept for the YOLO-assist stage, which would otherwise run
                # a SECOND inference over the identical frame.
                self._frame_person_dets = [
                    d for d in persons
                    if d.get("confidence", 0.0) >= conf_floor
                ]
                if boxes:
                    return (
                        np.array(boxes, dtype=np.float32),
                        np.array(scores, dtype=np.float32),
                        "person",
                    )
                return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), "person"

            # Model unavailable -- degrade to face tracking rather than blind.
            self._tracking_mode = "face"
            if not self._person_fallback_logged:
                self._person_fallback_logged = True
                logger.warning(
                    f"[{self.camera_id}] TRACKING_MODE=person but the YOLO person "
                    f"detector could not be loaded; falling back to face tracking. "
                    f"Boxes will drop when a face turns away."
                )

        # Face mode (explicit, or fallen back to).
        boxes = []
        for top, right, bottom, left in face_locations:
            boxes.append([left, top, right, bottom])
        if not boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), "face"
        return (
            np.array(boxes, dtype=np.float32),
            np.full(len(boxes), 1.0, dtype=np.float32),  # SCRFD already filtered
            "face",
        )

    def _bind_face_to_track(
        self,
        track,
        face_locations: List[Tuple],
        embeddings: List,
        face_landmarks: List,
        source: str,
    ) -> Tuple[Optional[Tuple], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find the face belonging to a track, if one is visible this frame.

        THIS RETURNING None IS NORMAL, NOT AN ERROR. It is the ordinary state
        of a person facing away from the camera, and is exactly the case Step 1
        exists to survive: the track keeps its identity, no recognition runs,
        and nothing downstream is told the person left.

        Person mode binds on containment -- a face centre inside the (padded)
        person box -- because a face box and a body box barely overlap by area,
        so IoU is the wrong test. Ties go to the LARGEST face, which is both
        the nearer person and the better embedding.

        Face mode keeps the original IoU match: there, the track box IS a face
        box, and the only question is which detection it drifted from.

        Returns:
            (face_location, embedding, landmarks) -- all None when unbound.
        """
        if not face_locations:
            return None, None, None

        top, right, bottom, left = track.to_location()

        if source == "person":
            pad = getattr(settings, "PERSON_FACE_BIND_PAD", 12)
            x1, y1, x2, y2 = left - pad, top - pad, right + pad, bottom + pad
            best_area = 0.0
            best = (None, None, None)
            for i, loc in enumerate(face_locations):
                if embeddings[i] is None:
                    continue
                f_top, f_right, f_bottom, f_left = loc
                cx = (f_left + f_right) / 2.0
                cy = (f_top + f_bottom) / 2.0
                if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                    continue
                area = max(0, f_right - f_left) * max(0, f_bottom - f_top)
                if area > best_area:
                    best_area = area
                    best = (
                        loc,
                        embeddings[i],
                        face_landmarks[i] if i < len(face_landmarks) else None,
                    )
            return best

        # ---- face mode: original IoU association ----
        best_iou = 0.0
        best = (None, None, None)
        tx1, ty1, tx2, ty2 = left, top, right, bottom
        for i, loc in enumerate(face_locations):
            if embeddings[i] is None:
                continue
            ft, fr, fb, fl = loc
            ix1, iy1 = max(tx1, fl), max(ty1, ft)
            ix2, iy2 = min(tx2, fr), min(ty2, fb)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            a1 = (tx2 - tx1) * (ty2 - ty1)
            a2 = (fr - fl) * (fb - ft)
            denom = a1 + a2 - inter
            iou = inter / denom if denom > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best = (
                    loc,
                    embeddings[i],
                    face_landmarks[i] if i < len(face_landmarks) else None,
                )
        return best

    def _scale_boxes(self, boxes: List[tuple]) -> List[tuple]:
        """
        Scale draw args from full-res detection space to display space.

        Entries are (location, label, color, confidence[, coasting]); any
        trailing elements are carried through untouched.
        """
        sx = self._stream_scale_x
        sy = self._stream_scale_y
        out = []
        for entry in boxes:
            location, label, color, conf = entry[:4]
            top, right, bottom, left = location
            scaled_loc = (
                int(top * sy), int(right * sx),
                int(bottom * sy), int(left * sx),
            )
            out.append((scaled_loc, label, color, conf) + tuple(entry[4:]))
        return out

    def _coasting_boxes(self) -> List[tuple]:
        """
        Draw args for tracks ByteTrack has lost but has not forgotten.

        See ByteTracker.get_coasting_tracks(). These carry a retained
        identity at a last-observed position; nothing about them is
        re-recognised and no presence event is emitted for them. Purely
        what the tracker already believes, made visible.
        """
        if not getattr(settings, "BYTETRACK_COAST_ENABLED", True):
            return []
        try:
            coasting = self.byte_tracker.get_coasting_tracks()
        except Exception as e:
            logger.debug(f"[{self.camera_id}] coasting lookup failed: {e}")
            return []

        boxes = []
        for track, _age in coasting:
            x1, y1, x2, y2 = track.last_seen_bbox.astype(int)
            location = (y1, x2, y2, x1)  # top, right, bottom, left
            if track.person_name:
                label = track.person_name
                color = (0, 150, 0)      # dimmed green -- known, not currently seen
            else:
                label = f"Track #{track.track_id}"
                color = (140, 100, 0)    # dimmed amber -- tracked, unidentified
            boxes.append((location, label, color, 0.0, True))

        if boxes:
            profiler.record_coasting(len(boxes))
        return boxes

    @staticmethod
    def _dashed_rect(
        frame: np.ndarray,
        pt1: tuple,
        pt2: tuple,
        color: tuple,
        thickness: int = 1,
        dash: int = 10,
    ):
        """Rectangle drawn as dashes, marking a box we are inferring rather than seeing."""
        x1, y1 = pt1
        x2, y2 = pt2
        for x in range(x1, x2, dash * 2):
            xe = min(x + dash, x2)
            cv2.line(frame, (x, y1), (xe, y1), color, thickness)
            cv2.line(frame, (x, y2), (xe, y2), color, thickness)
        for y in range(y1, y2, dash * 2):
            ye = min(y + dash, y2)
            cv2.line(frame, (x1, y), (x1, ye), color, thickness)
            cv2.line(frame, (x2, y), (x2, ye), color, thickness)

    def _draw_box(
        self,
        frame: np.ndarray,
        location: tuple,
        label: str,
        color: tuple,
        confidence: float = 0.0,
        coasting: bool = False,
    ):
        """
        Draw a bounding box and label on the frame.

        `coasting` marks a box drawn from a LOST track's last observed
        position -- the person was not detected on this frame. It is rendered
        dashed and dimmed so an operator can always tell an inference from an
        observation. Never render a coast box identically to a live one: the
        whole value of the feature is that it degrades honestly.
        """
        top, right, bottom, left = location

        # Draw rectangle
        if coasting:
            self._dashed_rect(frame, (left, top), (right, bottom), color, 1)
        else:
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

    def _resolve_analysis_fps(self) -> float:
        """How many frames per second this camera should analyse.

        Read once at construction, from cameras.config when a row exists for
        this camera. A missing row is normal rather than an error -- the load
        test and browser sessions have none -- so it falls back to the global
        default instead of refusing to start.
        """
        default = float(getattr(settings, "ANALYSIS_FPS", 10.0) or 0)
        try:
            from app.db.session import SessionLocal
            from app.db.models import Camera
            with SessionLocal() as db:
                row = db.query(Camera).filter(Camera.id == self.camera_id).first()
                cfg = (row.config or {}) if row else {}
        except Exception as e:
            logger.debug(f"[{self.camera_id}] no camera config for analysis fps: {e}")
            cfg = {}

        explicit = cfg.get("analysis_fps")
        if explicit:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                logger.warning(
                    f"[{self.camera_id}] config analysis_fps={explicit!r} is not "
                    f"a number; using {default}"
                )
        if cfg.get("is_gate"):
            return float(getattr(settings, "ANALYSIS_FPS_GATE", default) or default)
        return default

    def _analysis_due(self, consume: bool = False) -> bool:
        """Should this frame be analysed?

        Wall clock, so the answer does not depend on the camera's frame rate,
        on how many frames the decoder managed, or on how far behind we fell
        earlier. Called with consume=False from the grabber, which uses it to
        decide whether a frame is worth keeping at all, and with consume=True
        from process_frame, the only caller allowed to mark the interval spent.

        The two callers race by design: the grabber may keep a frame that
        process_frame then declines. That costs one stored reference and can
        never skip an analysis, so it does not justify a lock on this path.
        """
        if not getattr(settings, "ANALYSIS_RATE_BY_CLOCK", True):
            # Legacy behaviour: every Nth decoded frame.
            effective_skip = self._frame_skip_override or settings.FRAME_SKIP
            return self._frame_count % effective_skip == 0

        if self._analysis_interval <= 0:
            return True  # ANALYSIS_FPS=0 means analyse everything

        # The load tier still has a say. Its frame-skip override is a request
        # to analyse less often under burst load, applied here as a longer
        # interval rather than as a frame count.
        interval = self._analysis_interval
        skip = self._frame_skip_override
        if skip and settings.FRAME_SKIP:
            interval *= max(1.0, skip / float(settings.FRAME_SKIP))

        # perf_counter, NOT monotonic. time.monotonic() on Windows resolves to
        # 15.625 ms (measured: time.get_clock_info). A 10 fps target is a 100 ms
        # interval, which that clock can only express in 15.6 ms steps -- so the
        # gate would fire at the first tick at or past 100 ms, anywhere up to
        # 115.6 ms, and the achieved rate would sit 0-13% under target for
        # reasons no amount of spare GPU could fix. It also capped the
        # expressible rate at about 64 fps. perf_counter is monotonic too, and
        # resolves to 100 ns.
        now = time.perf_counter()
        if now - self._last_analysis_at < interval:
            return False
        if consume:
            self._last_analysis_at = now
        return True

    def _update_fps(self):
        """Calculate processing FPS.

        Called once per ANALYSIS frame -- never on a frame-skipped one -- so
        both `_fps` and `_analysis_frames` measure the detection rate, which
        is the rate ANALYSIS_FPS and the capacity model are stated in.
        """
        self._analysis_frames += 1
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
            # half and imgsz are passed EXPLICITLY (item 0a). Without them
            # ultralytics runs fp32 and re-derives the input size from each
            # frame's shape, so the cost and the effective detection scale
            # both followed the camera resolution and neither was stated
            # anywhere in this repo.
            results = self._yolo_person_detector(
                frame,
                conf=0.25,
                classes=[0],  # person class only
                verbose=False,
                device=getattr(self, '_yolo_person_device', 'cpu'),
                half=getattr(settings, "PERSON_DET_HALF", True),
                imgsz=getattr(settings, "PERSON_DET_IMGSZ", 640),
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
        elif self._tracking_mode != "person":
            # No face even in the focused crop -- someone facing away, heavily
            # occluded, or very small.
            #
            # FACE MODE ONLY. In person mode this person already HAS a track:
            # the tracker saw their body and owns their existence. Running this
            # too would build a second, centroid-keyed pseudo-identity beside
            # the real track, double-count them for presence, and re-mint an id
            # whenever they moved more than the dedup radius.
            self._handle_unknown_person_body(frame, clean_frame, person_det)

    def _handle_unknown_person_body(
        self,
        frame: np.ndarray,
        clean_frame: np.ndarray,
        person_det: dict,
    ):
        """
        Handle a person detected by YOLO whose face could NOT be detected.

        These are tracked for presence and occupancy, and drawn on the live
        view, but they do NOT create an identity row by default.

        WHY NOT: a body-only detection has no face embedding. It cannot be
        deduplicated (the consumer's dedup is keyed on the embedding and skips
        rows without one), cannot be re-identified on a later visit, and cannot
        be enrolled from the review page -- there is no face to enroll. Every
        such row is therefore permanent, unmergeable clutter.

        Measured on the live database before this change: 1,714 of 4,532
        unknown_faces rows (38%) had no encoding, and the 60-second cache below
        meant one person standing in view generated a fresh row, three JPEGs,
        and a new "Unknown #N" label every minute.

        Set PERSIST_BODY_ONLY_UNKNOWNS=true to restore the old behaviour; the
        detection itself is unchanged either way.
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
            unknown_number = len(self._yolo_body_unknown_cache) + 1
            unknown_label = f"Unknown #{unknown_number}"

            # Crops are only worth their copy cost if a row will be written.
            body_crop = head_crop = full_frame_copy = None
            if settings.PERSIST_BODY_ONLY_UNKNOWNS:
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
            if self.on_unknown and settings.PERSIST_BODY_ONLY_UNKNOWNS:
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
        face_landmarks = []
        rgb_frame = None

        if self._frame_count % settings.FRAME_SKIP == 0:
            if settings.PLAN_ALLOWS_FACE:
                face_locations, embeddings, face_landmarks, rgb_frame = (
                    self.detector.detect_faces_with_embeddings(clean_frame)
                )
            else:
                face_locations, embeddings, face_landmarks, rgb_frame = [], [], [], None

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
                 self._handle_unknown_face(
                     clean_frame, clean_frame, location, encoding, centroid,
                     landmarks=(face_landmarks[i] if i < len(face_landmarks) else None),
                 )

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
            "plan": settings.plan,
            "analysis_fps_target": getattr(self, "_analysis_fps", None),
            "stream_health": self.stream_health.status,
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
            base["unknown_identities"] = self.unknown_registry.size
            base["unknown_registry"] = self.unknown_registry.stats()
        else:
            # Face mode stats
            base["faces_in_cache"] = self.recognizer.known_count
            base["active_tracks"] = self.tracker.stats
            base["unknown_identities"] = self.unknown_registry.size
            base["unknown_registry"] = self.unknown_registry.stats()
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
