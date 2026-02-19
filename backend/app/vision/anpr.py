"""
ANPR — License Plate Recognition Module.

Detects and reads vehicle license plates using:
  - YOLOv8 for vehicle detection (reuses existing ultralytics dependency)
  - EasyOCR for plate text extraction

Designed for non-blocking operation via ThreadPoolExecutor.

COCO vehicle class IDs:
  - 2: car
  - 3: motorcycle
  - 5: bus
  - 7: truck

This module gracefully handles the case where easyocr isn't installed.
"""

import cv2
import numpy as np
import re
import time
import logging
import threading
from typing import Optional, Dict, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# COCO vehicle classes
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Regex patterns for Indian license plates (can be extended)
PLATE_PATTERNS = [
    # Standard Indian: KA01AB1234 / KA 01 AB 1234
    re.compile(r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}'),
    # Older format: MH 12 1234
    re.compile(r'[A-Z]{2}\s*\d{2}\s*\d{4}'),
    # Generic: at least 2 letters + digits
    re.compile(r'[A-Z0-9]{4,12}'),
]


class PlateRecognizer:
    """
    License Plate Recognition (ANPR) using YOLOv8 + EasyOCR.

    Features:
    - YOLOv8 detects vehicles in frame
    - Plate region extraction via edge detection + contour analysis
    - EasyOCR reads detected plate text
    - Frame skipping for performance
    - Deduplication cooldown (same plate won't re-trigger within cooldown)
    - Runs inference in ThreadPoolExecutor (non-blocking)

    Usage:
        recognizer = PlateRecognizer(on_plate=my_callback)
        recognizer.submit_frame(frame, camera_id)  # Non-blocking
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        frame_interval: int = 5,
        plate_cooldown_sec: float = 60.0,
        on_plate: Optional[Callable] = None,
        languages: List[str] = None,
        max_workers: int = 1,
    ):
        """
        Args:
            yolo_model_path: Path to YOLO model weights
            frame_interval: Only process every Nth frame
            plate_cooldown_sec: Don't re-alert for same plate within this window
            on_plate: Callback(plate_dict) when a plate is recognized
            languages: EasyOCR language list (default: ['en'])
            max_workers: Thread pool size
        """
        self.yolo_model_path = yolo_model_path
        self.frame_interval = frame_interval
        self.plate_cooldown_sec = plate_cooldown_sec
        self.on_plate = on_plate
        self.languages = languages or ['en']

        self._yolo = None
        self._ocr = None
        self._model_loaded = False
        self._ocr_loaded = False
        self._model_available = False
        self._ocr_available = False
        self._model_lock = threading.Lock()
        self._ocr_lock = threading.Lock()

        self._frame_counter = 0
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="anpr",
        )

        # Cooldown tracking: plate_text → last_alert_timestamp
        self._plate_cooldowns: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        # Statistics
        self._total_inferences = 0
        self._total_plates = 0
        self._last_inference_ms = 0.0

        logger.info(
            f"PlateRecognizer initialized: model={yolo_model_path}, "
            f"interval={frame_interval}, cooldown={plate_cooldown_sec}s"
        )

    def _load_yolo(self) -> bool:
        """Lazy-load YOLOv8 model. Thread-safe."""
        with self._model_lock:
            if self._model_loaded:
                return self._model_available
            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLOv8 for ANPR: {self.yolo_model_path}")
                self._yolo = YOLO(self.yolo_model_path)
                self._model_available = True
                logger.info("YOLOv8 model loaded for ANPR")
            except ImportError:
                self._model_available = False
                logger.warning(
                    "ultralytics not installed — ANPR vehicle detection disabled. "
                    "Install with: pip install ultralytics"
                )
            except Exception as e:
                self._model_available = False
                logger.error(f"Failed to load YOLOv8 for ANPR: {e}")
            finally:
                self._model_loaded = True
            return self._model_available

    def _load_ocr(self) -> bool:
        """Lazy-load EasyOCR reader. Thread-safe."""
        with self._ocr_lock:
            if self._ocr_loaded:
                return self._ocr_available
            try:
                import easyocr
                logger.info(f"Loading EasyOCR: languages={self.languages}")
                self._ocr = easyocr.Reader(self.languages, gpu=False)
                self._ocr_available = True
                logger.info("EasyOCR loaded successfully")
            except ImportError:
                self._ocr_available = False
                logger.warning(
                    "easyocr not installed — plate text extraction disabled. "
                    "Install with: pip install easyocr"
                )
            except Exception as e:
                self._ocr_available = False
                logger.error(f"Failed to load EasyOCR: {e}")
            finally:
                self._ocr_loaded = True
            return self._ocr_available

    def submit_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
    ) -> bool:
        """
        Submit a frame for ANPR (non-blocking).
        Only processes every `frame_interval`-th frame.

        Returns:
            True if frame was submitted, False if skipped
        """
        self._frame_counter += 1
        if self._frame_counter % self.frame_interval != 0:
            return False

        frame_copy = frame.copy()
        self._executor.submit(self._process, frame_copy, camera_id)
        return True

    def _process(self, frame: np.ndarray, camera_id: str):
        """Detect vehicles → extract plate regions → OCR → emit events."""
        if not self._load_yolo():
            return

        try:
            start_time = time.monotonic()

            # Step 1: Detect vehicles
            results = self._yolo(
                frame,
                conf=0.40,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )

            inference_ms = (time.monotonic() - start_time) * 1000
            self._last_inference_ms = inference_ms
            self._total_inferences += 1

            if not results:
                return

            now = time.time()

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if cls_id not in VEHICLE_CLASSES:
                        continue

                    vehicle_type = VEHICLE_CLASSES[cls_id]
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]

                    # Step 2: Extract plate region from vehicle bounding box
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0:
                        continue

                    plate_text, plate_confidence = self._extract_plate(vehicle_crop)

                    if plate_text is None:
                        continue

                    # Step 3: Normalize plate text
                    normalized = self._normalize_plate(plate_text)
                    if not normalized or len(normalized) < 4:
                        continue

                    # Step 4: Cooldown check
                    with self._cooldown_lock:
                        last_seen = self._plate_cooldowns.get(normalized, 0)
                        if now - last_seen < self.plate_cooldown_sec:
                            continue
                        self._plate_cooldowns[normalized] = now

                    self._total_plates += 1

                    # Step 5: Emit event
                    event = {
                        "event_type": "vehicle_entry",
                        "camera_id": camera_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metadata": {
                            "plate": normalized,
                            "confidence": round(plate_confidence, 3),
                            "vehicle_type": vehicle_type,
                            "vehicle_confidence": round(confidence, 3),
                            "bounding_box": {
                                "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2,
                            },
                        },
                    }

                    logger.info(
                        f"🚗 PLATE DETECTED: {normalized} "
                        f"({vehicle_type}, conf={plate_confidence:.2f}) "
                        f"on camera {camera_id}"
                    )

                    if self.on_plate:
                        self.on_plate(event)

        except Exception as e:
            logger.error(f"ANPR processing failed: {e}", exc_info=True)

    def _extract_plate(
        self,
        vehicle_crop: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Extract license plate text from a vehicle crop image.

        Strategy:
        1. Focus on the lower half of the vehicle (where plates usually are)
        2. Preprocess: grayscale → bilateral filter → edge detection
        3. Find contours that look like plate regions (rectangular, right aspect ratio)
        4. Run EasyOCR on the best candidate region

        Returns:
            (plate_text, confidence) or (None, 0.0) if no plate found
        """
        if not self._load_ocr():
            return None, 0.0

        h, w = vehicle_crop.shape[:2]
        if h < 30 or w < 30:
            return None, 0.0

        # Focus on lower 60% of vehicle (plate area)
        plate_region = vehicle_crop[int(h * 0.4):, :]

        # Preprocessing for plate detection
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        plate_candidates = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # License plates are roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect_ratio = cw / ch if ch > 0 else 0

                # Plates typically have aspect ratio between 2:1 and 6:1
                if 1.5 < aspect_ratio < 7.0 and cw > 40 and ch > 10:
                    plate_candidates.append((x, y, cw, ch))

        # OCR on the best candidate(s) or fall back to full region
        best_text = None
        best_confidence = 0.0

        if plate_candidates:
            for x, y, cw, ch in plate_candidates[:3]:
                candidate = plate_region[y:y + ch, x:x + cw]
                text, conf = self._ocr_region(candidate)
                if text and conf > best_confidence:
                    best_text = text
                    best_confidence = conf
        else:
            # No good contours — try OCR on the whole plate region
            best_text, best_confidence = self._ocr_region(plate_region)

        return best_text, best_confidence

    def _ocr_region(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """Run EasyOCR on a region and return the best plate-like match."""
        try:
            if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
                return None, 0.0

            results = self._ocr.readtext(image, detail=1)

            if not results:
                return None, 0.0

            # Find the best match that looks like a plate
            for (bbox, text, confidence) in sorted(results, key=lambda r: r[2], reverse=True):
                cleaned = text.strip().upper().replace(' ', '')

                # Check if it matches known plate patterns
                for pattern in PLATE_PATTERNS:
                    if pattern.match(cleaned):
                        return cleaned, confidence

            # If no pattern match, return the highest-confidence text
            # (if it looks roughly alphanumeric)
            if results:
                best = max(results, key=lambda r: r[2])
                text = best[1].strip().upper().replace(' ', '')
                if len(text) >= 4 and any(c.isdigit() for c in text):
                    return text, best[2]

        except Exception as e:
            logger.debug(f"OCR failed on region: {e}")

        return None, 0.0

    def _normalize_plate(self, text: str) -> Optional[str]:
        """Clean and normalize plate text."""
        if not text:
            return None

        # Remove non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

        # Must have both letters and digits
        has_letters = any(c.isalpha() for c in cleaned)
        has_digits = any(c.isdigit() for c in cleaned)

        if has_letters and has_digits and len(cleaned) >= 4:
            return cleaned

        return None

    def draw_detections(
        self,
        frame: np.ndarray,
        plate_events: List[dict],
    ) -> np.ndarray:
        """Draw ANPR detections on a frame for visualization."""
        for event in plate_events:
            meta = event.get("metadata", {})
            bbox = meta.get("bounding_box", {})
            plate = meta.get("plate", "???")

            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)

            # Green box for vehicles with plates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"PLATE: {plate}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw + 10, y1),
                (0, 255, 0), cv2.FILLED,
            )
            cv2.putText(
                frame, label,
                (x1 + 5, y1 - 5),
                font, 0.7, (0, 0, 0), 2,
            )

        return frame

    @property
    def stats(self) -> dict:
        return {
            "model_loaded": self._model_available,
            "ocr_loaded": self._ocr_available,
            "total_inferences": self._total_inferences,
            "total_plates_detected": self._total_plates,
            "last_inference_ms": round(self._last_inference_ms, 1),
        }

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
        logger.info("PlateRecognizer shut down")
