"""
Hazard Detector — YOLOv8-based threat/object detection.

Detects hazardous objects (knives, guns, backpacks/abandoned objects) using
Ultralytics YOLOv8 (nano model for speed). Designed to run on a separate
thread or at reduced frame rates to avoid impacting the main video pipeline.

COCO class IDs of interest:
  - 24: backpack
  - 25: umbrella (optional)
  - 26: handbag
  - 43: knife
  - Not in COCO: gun, fire  — we add them for future custom models

This module gracefully handles the case where ultralytics isn't installed.
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import Optional, Dict, List, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── COCO class mapping for threat-relevant objects ────────────────────────────
# Full COCO has 80 classes. We only flag actual weapons/dangers:
THREAT_CLASSES: Dict[int, str] = {
    # Weapons / sharp objects
    43: "knife",
    76: "scissors",
}

# NOTE: backpack (24), handbag (26), suitcase (28) were removed because
# they trigger on any visible bag in normal environments. To detect truly
# ABANDONED objects, you'd need temporal logic (object present with no
# person nearby for X minutes), which is a separate feature.

# Future: custom-trained model classes (fire, gun, etc.)
CUSTOM_THREAT_CLASSES: Dict[str, str] = {
    "fire": "fire",
    "gun": "gun",
    "weapon": "weapon",
}

# Confidence thresholds per class
CLASS_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "knife": 0.50,
    "scissors": 0.55,
    "gun": 0.50,
    "fire": 0.40,
    "weapon": 0.50,
}

DEFAULT_CONFIDENCE = 0.50


class HazardDetector:
    """
    YOLOv8-based hazard/threat object detector.

    Features:
    - Lazy model loading (only loads when first detection is requested)
    - Runs inference on a ThreadPoolExecutor to avoid blocking video pipeline
    - Frame throttling (only processes every Nth frame)
    - Deduplication / cooldown to prevent alert spam
    - Graceful degradation if ultralytics isn't installed

    Usage:
        detector = HazardDetector(on_alert=my_callback)
        detector.submit_frame(frame, camera_id)  # Non-blocking
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        frame_interval: int = 30,
        alert_cooldown_sec: float = 30.0,
        on_alert: Optional[Callable] = None,
        max_workers: int = 1,
    ):
        """
        Args:
            model_path: Path to YOLO model weights (downloaded automatically)
            frame_interval: Only process every Nth frame
            alert_cooldown_sec: Minimum seconds between alerts for same object class
            on_alert: Callback(alert_dict) when a threat is detected
            max_workers: Thread pool size (1 is usually enough)
        """
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.alert_cooldown_sec = alert_cooldown_sec
        self.on_alert = on_alert

        self._model = None
        self._model_loaded = False
        self._model_available = False
        self._model_lock = threading.Lock()

        self._frame_counter = 0
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hazard-det",
        )

        # Cooldown tracking: (camera_id, class_name) → last_alert_timestamp
        self._alert_cooldowns: Dict[Tuple[str, str], float] = {}
        self._cooldown_lock = threading.Lock()

        # Statistics
        self._total_inferences = 0
        self._total_detections = 0
        self._last_inference_ms = 0.0

        logger.info(
            f"HazardDetector initialized: model={model_path}, "
            f"interval={frame_interval}, cooldown={alert_cooldown_sec}s"
        )

    def _load_model(self) -> bool:
        """Lazy-load YOLOv8 model. Thread-safe."""
        with self._model_lock:
            if self._model_loaded:
                return self._model_available

            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLOv8 model: {self.model_path}")
                self._model = YOLO(self.model_path)
                self._model_available = True
                logger.info("YOLOv8 model loaded successfully")
            except ImportError:
                self._model_available = False
                logger.warning(
                    "ultralytics not installed — hazard detection disabled. "
                    "Install with: pip install ultralytics"
                )
            except Exception as e:
                self._model_available = False
                logger.error(f"Failed to load YOLOv8 model: {e}")
            finally:
                self._model_loaded = True

            return self._model_available

    def submit_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
    ) -> bool:
        """
        Submit a frame for hazard detection (non-blocking).

        Only actually processes every `frame_interval`-th frame.

        Args:
            frame: BGR image
            camera_id: Camera identifier

        Returns:
            True if frame was submitted for processing, False if skipped
        """
        self._frame_counter += 1

        if self._frame_counter % self.frame_interval != 0:
            return False

        # Copy frame since it will be processed asynchronously
        frame_copy = frame.copy()
        self._executor.submit(self._detect, frame_copy, camera_id)
        return True

    def _detect(self, frame: np.ndarray, camera_id: str):
        """Run YOLO inference on a frame (runs in thread pool)."""
        if not self._load_model():
            return

        try:
            start_time = time.monotonic()

            # Run inference with low verbosity
            results = self._model(
                frame,
                conf=0.35,       # Base confidence threshold
                verbose=False,
                stream=False,
            )

            inference_ms = (time.monotonic() - start_time) * 1000
            self._last_inference_ms = inference_ms
            self._total_inferences += 1

            if not results:
                return

            now = time.time()
            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Check if this class is one we care about
                    class_name = None

                    if cls_id in THREAT_CLASSES:
                        class_name = THREAT_CLASSES[cls_id]
                    else:
                        # Check by name (for custom models)
                        try:
                            raw_name = result.names.get(cls_id, "").lower()
                            if raw_name in CUSTOM_THREAT_CLASSES:
                                class_name = CUSTOM_THREAT_CLASSES[raw_name]
                        except Exception:
                            pass

                    if class_name is None:
                        continue

                    # Check class-specific confidence threshold
                    min_conf = CLASS_CONFIDENCE_THRESHOLDS.get(
                        class_name, DEFAULT_CONFIDENCE
                    )
                    if confidence < min_conf:
                        continue

                    # Check cooldown
                    cooldown_key = (camera_id, class_name)
                    with self._cooldown_lock:
                        last_alert = self._alert_cooldowns.get(cooldown_key, 0)
                        if now - last_alert < self.alert_cooldown_sec:
                            continue  # Still in cooldown
                        self._alert_cooldowns[cooldown_key] = now

                    # Extract bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detection = {
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": {
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                        },
                    }
                    detections.append(detection)
                    self._total_detections += 1

                    # Emit alert
                    alert = {
                        "event_type": "security_alert",
                        "subtype": "hazard",
                        "camera_id": camera_id,
                        "threat_class": class_name,
                        "confidence": round(confidence, 3),
                        "bounding_box": detection["bbox"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "inference_time_ms": round(inference_ms, 1),
                        "metadata": {
                            "model": self.model_path,
                            "coco_class_id": cls_id,
                        },
                    }

                    logger.warning(
                        f"🚨 HAZARD ALERT: {class_name} detected "
                        f"(conf={confidence:.2f}) on camera {camera_id}"
                    )

                    if self.on_alert:
                        self.on_alert(alert)

        except Exception as e:
            logger.error(f"Hazard detection failed: {e}", exc_info=True)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[dict],
    ) -> np.ndarray:
        """
        Draw hazard detection boxes on a frame (for visualization).

        Args:
            frame: BGR image
            detections: List of detection dicts from _detect

        Returns:
            Annotated frame
        """
        for det in detections:
            bbox = det["bbox"]
            cls_name = det["class"]
            conf = det["confidence"]

            # Red boxes for all threats
            color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                color, 2,
            )

            label = f"⚠ {cls_name.upper()} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
            cv2.rectangle(
                frame,
                (bbox["x1"], bbox["y1"] - th - 10),
                (bbox["x1"] + tw + 10, bbox["y1"]),
                color, cv2.FILLED,
            )
            cv2.putText(
                frame, label,
                (bbox["x1"] + 5, bbox["y1"] - 5),
                font, 0.6, (255, 255, 255), 1,
            )

        return frame

    @property
    def stats(self) -> dict:
        return {
            "model_loaded": self._model_available,
            "total_inferences": self._total_inferences,
            "total_detections": self._total_detections,
            "last_inference_ms": round(self._last_inference_ms, 1),
        }

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
        logger.info("HazardDetector shut down")
