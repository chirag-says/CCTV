"""
ANPR — License Plate Recognition Module.

Two-stage detection pipeline:
  Stage 1: YOLOv8 COCO model detects vehicles (car, motorcycle, bus, truck)
  Stage 2: Custom plate detector finds plate regions within vehicle crops
  Stage 3: EasyOCR reads detected plate text

Falls back to single-stage if only one model is available:
  - If only plate model: run plate detector on full frame
  - If only COCO model: use contour-based plate extraction + OCR

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

# Regex patterns for license plates — tuned for Indian plates.
# We keep these strict to avoid accepting random text.
PLATE_PATTERNS = [
    # Standard Indian: KA01AB1234 / KA-01-AB-1234 / KA 01 AB 1234
    re.compile(r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}'),
    # Condensed Indian (OCR output, no spaces): KA02HM1826
    re.compile(r'[A-Z]{1,3}\d{1,2}[A-Z]{1,3}\d{1,5}'),
    # Older Indian format (no series letters): MH 12 1234
    re.compile(r'[A-Z]{2}\s*\d{2}\s*\d{4}'),
]


class PlateRecognizer:
    """
    License Plate Recognition (ANPR) using two-stage YOLO + EasyOCR.

    Pipeline:
    1. YOLOv8 COCO model detects vehicles in full frame
    2. Custom plate model detects plate regions within each vehicle crop
    3. EasyOCR reads text from detected plate regions

    Features:
    - Two-stage detection for high accuracy
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
            yolo_model_path: Path to custom plate YOLO model weights
            frame_interval: Only process every Nth frame
            plate_cooldown_sec: Don't re-alert for same plate within this window
            on_plate: Callback(plate_dict) when a plate is recognized
            languages: EasyOCR language list (default: ['en'])
            max_workers: Thread pool size
        """
        self.plate_model_path = yolo_model_path
        self.frame_interval = frame_interval
        self.plate_cooldown_sec = plate_cooldown_sec
        self.on_plate = on_plate
        self.languages = languages or ['en']

        # Models
        self._vehicle_model = None    # COCO model for vehicle detection
        self._plate_model = None      # Custom model for plate detection
        self._ocr = None

        # Loading flags
        self._vehicle_model_loaded = False
        self._vehicle_model_available = False
        self._plate_model_loaded = False
        self._plate_model_available = False
        self._ocr_loaded = False
        self._ocr_available = False

        # Thread safety
        self._vehicle_model_lock = threading.Lock()
        self._plate_model_lock = threading.Lock()
        self._ocr_lock = threading.Lock()
        self._device = 'cpu'

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
            f"PlateRecognizer initialized: plate_model={yolo_model_path}, "
            f"interval={frame_interval}, cooldown={plate_cooldown_sec}s"
        )

    # ── Model Loading ─────────────────────────────────────────────────────

    def _load_vehicle_model(self) -> bool:
        """Lazy-load YOLOv8 COCO model for vehicle detection. Thread-safe."""
        with self._vehicle_model_lock:
            if self._vehicle_model_loaded:
                return self._vehicle_model_available
            try:
                import torch
                from ultralytics import YOLO
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Loading YOLOv8 COCO model for vehicle detection on {self._device}")
                self._vehicle_model = YOLO("yolov8n.pt")
                self._vehicle_model_available = True
                logger.info(f"Vehicle detection model loaded (COCO yolov8n) on {self._device}")
            except Exception as e:
                self._vehicle_model_available = False
                logger.error(f"Failed to load vehicle detection model: {e}")
            finally:
                self._vehicle_model_loaded = True
            return self._vehicle_model_available

    def _load_plate_model(self) -> bool:
        """Lazy-load custom plate detection model. Thread-safe."""
        with self._plate_model_lock:
            if self._plate_model_loaded:
                return self._plate_model_available
            try:
                from ultralytics import YOLO
                import os
                if not os.path.exists(self.plate_model_path):
                    logger.warning(f"Plate model not found: {self.plate_model_path}")
                    self._plate_model_available = False
                    return False

                logger.info(f"Loading custom plate detection model: {self.plate_model_path}")
                self._plate_model = YOLO(self.plate_model_path)
                model_names = getattr(self._plate_model, 'names', {})
                num_classes = len(model_names) if model_names else 0
                self._plate_model_available = True
                logger.info(
                    f"Custom plate model loaded: {num_classes} classes = "
                    f"{list(model_names.values())}"
                )
            except Exception as e:
                self._plate_model_available = False
                logger.error(f"Failed to load plate detection model: {e}")
            finally:
                self._plate_model_loaded = True
            return self._plate_model_available

    def _load_ocr(self) -> bool:
        """Lazy-load EasyOCR reader. Thread-safe."""
        with self._ocr_lock:
            if self._ocr_loaded:
                return self._ocr_available
            try:
                import torch
                import easyocr
                use_gpu = torch.cuda.is_available()
                logger.info(f"Loading EasyOCR: languages={self.languages}, gpu={use_gpu}")
                self._ocr = easyocr.Reader(self.languages, gpu=use_gpu)
                self._ocr_available = True
                logger.info(f"EasyOCR loaded successfully (GPU={'enabled' if use_gpu else 'disabled'})")
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

    # ── Frame Submission ──────────────────────────────────────────────────

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

    # ── Main Processing ───────────────────────────────────────────────────

    def _process(self, frame: np.ndarray, camera_id: str):
        """
        Two-stage ANPR pipeline:
        1. Detect vehicles using COCO model
        2. Find plates within each vehicle using plate model
        3. OCR the plate regions
        """
        try:
            start_time = time.monotonic()

            # Stage 1: Detect vehicles
            vehicles = self._detect_vehicles(frame)

            if not vehicles:
                return

            logger.debug(f"ANPR: Found {len(vehicles)} vehicles in frame")

            now = time.time()

            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle['box']
                vehicle_type = vehicle['type']
                vehicle_conf = vehicle['confidence']

                # Crop vehicle region
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 30 or vehicle_crop.shape[1] < 30:
                    continue

                # Stage 2: Find plate within vehicle crop
                plate_regions = self._detect_plates_in_crop(vehicle_crop)

                if not plate_regions:
                    # Fallback: try contour-based plate extraction
                    plate_text, plate_confidence = self._extract_plate_contours(vehicle_crop)
                    if plate_text:
                        plate_regions = [{'text': plate_text, 'confidence': plate_confidence}]
                    else:
                        continue

                # Stage 3: OCR each plate region
                for plate_info in plate_regions:
                    if 'crop' in plate_info:
                        # We have a crop from the plate detector
                        plate_text, ocr_confidence = self._ocr_region(plate_info['crop'])
                    elif 'text' in plate_info:
                        # Already got text from contour method
                        plate_text = plate_info['text']
                        ocr_confidence = plate_info['confidence']
                    else:
                        continue

                    if plate_text is None:
                        continue

                    # Normalize
                    normalized = self._normalize_plate(plate_text)
                    if not normalized or len(normalized) < 4:
                        continue

                    # Cooldown check
                    with self._cooldown_lock:
                        last_seen = self._plate_cooldowns.get(normalized, 0)
                        if now - last_seen < self.plate_cooldown_sec:
                            continue
                        self._plate_cooldowns[normalized] = now

                    self._total_plates += 1

                    # Emit event
                    event = {
                        "event_type": "vehicle_entry",
                        "camera_id": camera_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metadata": {
                            "plate": normalized,
                            "confidence": round(ocr_confidence, 3),
                            "vehicle_type": vehicle_type,
                            "vehicle_confidence": round(vehicle_conf, 3),
                            "bounding_box": {
                                "x1": vx1, "y1": vy1,
                                "x2": vx2, "y2": vy2,
                            },
                        },
                    }

                    logger.info(
                        f"🚗 PLATE DETECTED: {normalized} "
                        f"({vehicle_type}, conf={ocr_confidence:.2f}) "
                        f"on camera {camera_id}"
                    )

                    if self.on_plate:
                        self.on_plate(event)

            inference_ms = (time.monotonic() - start_time) * 1000
            self._last_inference_ms = inference_ms
            self._total_inferences += 1

        except Exception as e:
            logger.error(f"ANPR processing failed: {e}", exc_info=True)

    # ── Stage 1: Vehicle Detection ────────────────────────────────────────

    def _detect_vehicles(self, frame: np.ndarray) -> List[dict]:
        """Detect vehicles in the frame using COCO YOLOv8."""
        if not self._load_vehicle_model():
            return []

        results = self._vehicle_model(
            frame,
            conf=0.35,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
            device=self._device,
        )

        vehicles = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                vehicles.append({
                    'box': (x1, y1, x2, y2),
                    'type': VEHICLE_CLASSES[cls_id],
                    'confidence': confidence,
                })

        return vehicles

    # ── Stage 2: Plate Detection within Vehicle ──────────────────────────

    def _detect_plates_in_crop(self, vehicle_crop: np.ndarray) -> List[dict]:
        """
        Detect plate regions within a vehicle crop.
        
        Uses the custom plate detection model if available.
        Returns list of plate regions with their crops.
        """
        if not self._load_plate_model():
            return []

        h, w = vehicle_crop.shape[:2]

        results = self._plate_model(
            vehicle_crop,
            conf=0.25,
            verbose=False,
            device=self._device,
        )

        plates = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]

                # Clamp to vehicle crop bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                plate_crop = vehicle_crop[y1:y2, x1:x2]
                if plate_crop.size == 0 or plate_crop.shape[0] < 8 or plate_crop.shape[1] < 15:
                    continue

                plates.append({
                    'crop': plate_crop,
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                })

        return plates

    # ── Stage 3: Plate Text Extraction ────────────────────────────────────

    def _extract_plate_contours(
        self,
        vehicle_crop: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Fallback plate extraction using image processing.

        Strategy:
        1. Focus on the lower half of the vehicle (where plates usually are)
        2. Preprocess: grayscale → bilateral filter → edge detection
        3. Find contours that look like plate regions
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
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) >= 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect_ratio = cw / ch if ch > 0 else 0
                if 1.5 < aspect_ratio < 7.0 and cw > 40 and ch > 10:
                    plate_candidates.append((x, y, cw, ch))

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
            best_text, best_confidence = self._ocr_region(plate_region)

        return best_text, best_confidence

    def _ocr_region(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """Run EasyOCR on a region and return the best plate-like match."""
        if not self._load_ocr():
            return None, 0.0

        try:
            if image.size == 0 or image.shape[0] < 8 or image.shape[1] < 15:
                return None, 0.0

            # Preprocess: enhance contrast for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Resize small crops for better OCR
            h, w = enhanced.shape[:2]
            if w < 100:
                scale = 100 / w
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)

            results = self._ocr.readtext(enhanced, detail=1)

            if not results:
                # Try on original color image too
                results = self._ocr.readtext(image, detail=1)

            if not results:
                return None, 0.0

            # Find the best match that looks like a plate
            for (bbox, text, confidence) in sorted(results, key=lambda r: r[2], reverse=True):
                cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

                for pattern in PLATE_PATTERNS:
                    if pattern.match(cleaned):
                        return cleaned, confidence

            # No pattern matched — reject this reading entirely
            # (don't return random text as a "plate")

        except Exception as e:
            logger.debug(f"OCR failed on region: {e}")

        return None, 0.0

    def _normalize_plate(self, text: str) -> Optional[str]:
        """Clean and normalize plate text."""
        if not text:
            return None

        cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

        # Minimum 6 characters (e.g. KA02H1) and must have both letters and digits
        if len(cleaned) < 6:
            return None

        has_letters = any(c.isalpha() for c in cleaned)
        has_digits = any(c.isdigit() for c in cleaned)

        if not (has_letters and has_digits):
            return None

        # Must match an Indian plate format pattern
        for pattern in PLATE_PATTERNS:
            if pattern.match(cleaned):
                return cleaned

        return None

    # ── Visualization ─────────────────────────────────────────────────────

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

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "vehicle_model_loaded": self._vehicle_model_available,
            "plate_model_loaded": self._plate_model_available,
            "ocr_loaded": self._ocr_available,
            "total_inferences": self._total_inferences,
            "total_plates_detected": self._total_plates,
            "last_inference_ms": round(self._last_inference_ms, 1),
        }

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=True)
        logger.info(
            f"PlateRecognizer shut down "
            f"(total inferences: {self._total_inferences}, "
            f"plates detected: {self._total_plates})"
        )
