"""
Hazard Detector — RF-DETR-based threat/object detection.

Uses RF-DETR (Real-Time Detection Transformer) for superior object detection
accuracy compared to YOLOv8. RF-DETR achieves 62+ mAP on COCO vs ~50 mAP
for YOLOv8, resulting in significantly fewer false positives and missed detections.

Falls back to YOLOv8 if RF-DETR is not available.

COCO class IDs of interest:
  - 43: knife
  - 76: scissors
  - Not in COCO: gun, fire  — for future custom models

This module gracefully handles missing dependencies.
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
# RF-DETR uses standard COCO classes. We only flag actual weapons/dangers:
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
    RF-DETR-based hazard/threat object detector.

    Features:
    - Uses RF-DETR (DINOv2 backbone) for state-of-the-art detection accuracy
    - Falls back to YOLOv8 if rfdetr is not installed
    - Lazy model loading (only loads when first detection is requested)
    - Runs inference on a ThreadPoolExecutor to avoid blocking video pipeline
    - Frame throttling (only processes every Nth frame)
    - Deduplication / cooldown to prevent alert spam
    - Graceful degradation if dependencies aren't installed

    Usage:
        detector = HazardDetector(on_alert=my_callback)
        detector.submit_frame(frame, camera_id)  # Non-blocking
    """

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        frame_interval: int = 30,
        alert_cooldown_sec: float = 30.0,
        on_alert: Optional[Callable] = None,
        max_workers: int = 1,
        rfdetr_model_size: str = "nano",
        prefer: str = "yolo",
    ):
        """
        Args:
            model_path: YOLO weights for the weapons classes
            frame_interval: Only process every Nth frame
            alert_cooldown_sec: Minimum seconds between alerts for same object class
            on_alert: Callback(alert_dict) when a threat is detected
            max_workers: Thread pool size (1 is usually enough)
            rfdetr_model_size: RF-DETR size — "nano", "base", or "large"
            prefer: which backend to try first — "yolo" or "rfdetr"
        """
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.alert_cooldown_sec = alert_cooldown_sec
        self.on_alert = on_alert
        self.rfdetr_model_size = rfdetr_model_size
        self.prefer = prefer if prefer in ("yolo", "rfdetr") else "yolo"

        self._model = None
        self._model_loaded = False
        self._model_available = False
        self._model_backend = None  # "rfdetr" or "yolo"
        self._model_lock = threading.Lock()

        self._frame_counter = 0
        self._counter_lock = threading.Lock()
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
            f"HazardDetector initialized: prefer={self.prefer}, "
            f"yolo={model_path}, rfdetr_size={rfdetr_model_size}, "
            f"interval={frame_interval}, cooldown={alert_cooldown_sec}s"
        )

    def _load_model(self) -> bool:
        """Lazy-load the preferred backend, falling back to the other. Thread-safe.

        WHY YOLO IS THE DEFAULT NOW
        RF-DETR Large was tried first, unconditionally, and this object is
        per camera. Twenty cameras meant twenty RF-DETR Large models: the
        20-camera baseline on this 12 GB card sat at 12,024 MB of VRAM with
        the GPU only 66% busy -- out of memory long before it was out of
        compute.

        YOLO11m finds the same COCO threat classes for a fraction of that,
        and it composes: VisionPipeline._load_yolo_person_detector() reuses
        this model when the backend is "yolo", so a camera ends up loading
        one detector instead of two.

        settings.HAZARD_BACKEND="rfdetr" restores the old order.
        """
        with self._model_lock:
            if self._model_loaded:
                return self._model_available

            order = (["rfdetr", "yolo"] if self.prefer == "rfdetr"
                     else ["yolo", "rfdetr"])
            for backend in order:
                loader = self._load_rfdetr if backend == "rfdetr" else self._load_yolo
                try:
                    if loader():
                        self._model_loaded = True
                        return True
                except Exception as e:
                    logger.warning(f"Hazard {backend} load failed: {e}")

            self._model_available = False
            self._model_loaded = True
            logger.error(
                "Hazard detection disabled: neither ultralytics nor rfdetr "
                "could provide a model."
            )
            return False

    def _load_yolo(self) -> bool:
        """Load the weapons model. False (not an exception) if no ultralytics."""
        try:
            import torch
            from ultralytics import YOLO
        except ImportError:
            logger.info("ultralytics not installed; trying RF-DETR for hazards")
            return False

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Item 1: a TensorRT engine when one is available, else the .pt.
        from app.vision.engines import resolve_weights
        load_path = resolve_weights(self.model_path)
        self._model = YOLO(load_path)
        self._model_available = True
        self._model_backend = "yolo"
        logger.info(f"Hazard detector: YOLO {load_path} on {self._device}")
        return True

    def _load_rfdetr(self) -> bool:
        """Load RF-DETR at the configured size. False if rfdetr is absent."""
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError:
            logger.info("rfdetr not installed; trying YOLO for hazards")
            return False
        try:
            from rfdetr import RFDETRNano
        except ImportError:
            RFDETRNano = RFDETRBase  # older rfdetr has no nano

        model_map = {"nano": RFDETRNano, "base": RFDETRBase, "large": RFDETRLarge}
        model_cls = model_map.get(self.rfdetr_model_size, RFDETRNano)
        logger.info(f"Loading RF-DETR ({self.rfdetr_model_size}) for hazard detection...")
        self._model = model_cls()
        self._model_available = True
        self._model_backend = "rfdetr"
        logger.info(f"Hazard detector: RF-DETR {self.rfdetr_model_size}")
        return True

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
        with self._counter_lock:
            self._frame_counter += 1
            if self._frame_counter % self.frame_interval != 0:
                return False

        # Copy frame since it will be processed asynchronously
        frame_copy = frame.copy()
        self._executor.submit(self._detect, frame_copy, camera_id)
        return True

    def _detect(self, frame: np.ndarray, camera_id: str):
        """Run detection on a frame (runs in thread pool)."""
        if not self._load_model():
            return

        if self._model_backend == "rfdetr":
            self._detect_rfdetr(frame, camera_id)
        else:
            self._detect_yolo(frame, camera_id)

    def _detect_rfdetr(self, frame: np.ndarray, camera_id: str):
        """Run RF-DETR inference on a frame."""
        try:
            import supervision as sv
            from PIL import Image

            start_time = time.monotonic()

            # RF-DETR expects RGB PIL image or numpy array
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Run RF-DETR inference
            detections = self._model.predict(
                pil_image,
                threshold=0.35,
            )

            inference_ms = (time.monotonic() - start_time) * 1000
            self._last_inference_ms = inference_ms
            self._total_inferences += 1

            if detections is None or len(detections) == 0:
                return

            now = time.time()

            # RF-DETR returns sv.Detections with xyxy, confidence, class_id
            for i in range(len(detections)):
                cls_id = int(detections.class_id[i])
                confidence = float(detections.confidence[i])

                # Check if this class is one we care about
                class_name = None

                if cls_id in THREAT_CLASSES:
                    class_name = THREAT_CLASSES[cls_id]
                else:
                    # Check by name for custom models
                    try:
                        if hasattr(detections, 'data') and 'class_name' in detections.data:
                            raw_name = detections.data['class_name'][i].lower()
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
                        continue
                    self._alert_cooldowns[cooldown_key] = now

                # Extract bounding box
                x1, y1, x2, y2 = detections.xyxy[i].tolist()

                detection = {
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                    },
                }
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
                        "model": f"rfdetr-{self.rfdetr_model_size}",
                        "coco_class_id": cls_id,
                        "backend": "rfdetr",
                    },
                }

                logger.warning(
                    f"🚨 HAZARD ALERT [RF-DETR]: {class_name} detected "
                    f"(conf={confidence:.2f}) on camera {camera_id}"
                )

                if self.on_alert:
                    self.on_alert(alert)

        except Exception as e:
            logger.error(f"RF-DETR hazard detection failed: {e}", exc_info=True)

    def _detect_yolo(self, frame: np.ndarray, camera_id: str):
        """Run YOLO inference on a frame (fallback path)."""
        try:
            start_time = time.monotonic()

            # Run inference with low verbosity
            results = self._model(
                frame,
                conf=0.35,       # Base confidence threshold
                verbose=False,
                stream=False,
                device=getattr(self, '_device', 'cpu'),
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
                            "backend": "yolo",
                        },
                    }

                    logger.warning(
                        f"🚨 HAZARD ALERT [YOLO]: {class_name} detected "
                        f"(conf={confidence:.2f}) on camera {camera_id}"
                    )

                    if self.on_alert:
                        self.on_alert(alert)

        except Exception as e:
            logger.error(f"YOLO hazard detection failed: {e}", exc_info=True)

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
            "model_backend": self._model_backend or "none",
            "total_inferences": self._total_inferences,
            "total_detections": self._total_detections,
            "last_inference_ms": round(self._last_inference_ms, 1),
        }

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
        logger.info("HazardDetector shut down")
