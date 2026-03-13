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

# ── Indian License Plate Validation ───────────────────────────────────────────

# All valid Indian state/UT codes (RTO registration prefixes)
INDIAN_STATE_CODES = {
    "AN",  # Andaman & Nicobar
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CG",  # Chhattisgarh
    "CH",  # Chandigarh
    "DD",  # Dadra & Nagar Haveli and Daman & Diu
    "DL",  # Delhi
    "GA",  # Goa
    "GJ",  # Gujarat
    "HP",  # Himachal Pradesh
    "HR",  # Haryana
    "JH",  # Jharkhand
    "JK",  # Jammu & Kashmir
    "KA",  # Karnataka
    "KL",  # Kerala
    "LA",  # Ladakh
    "LD",  # Lakshadweep
    "MH",  # Maharashtra
    "ML",  # Meghalaya
    "MN",  # Manipur
    "MP",  # Madhya Pradesh
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha
    "PB",  # Punjab
    "PY",  # Puducherry
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TN",  # Tamil Nadu
    "TR",  # Tripura
    "TS",  # Telangana
    "UK",  # Uttarakhand
    "UP",  # Uttar Pradesh
    "WB",  # West Bengal
}

# Common OCR misreads for characters on Indian plates
# Map of (wrong char) → (likely correct char)
OCR_CHAR_CORRECTIONS = {
    "0": "O",  # zero  → O (in letter positions)
    "O": "0",  # O     → zero (in digit positions)
    "1": "I",  # one   → I (in letter positions)
    "I": "1",  # I     → one (in digit positions)
    "8": "B",  # eight → B (in letter positions)
    "B": "8",  # B     → eight (in digit positions)
    "5": "S",  # five  → S (in letter positions)
    "S": "5",  # S     → five (in digit positions)
    "6": "G",  # six   → G (in letter positions)
    "G": "6",  # G     → six (in digit positions)
    "2": "Z",  # two   → Z (in letter positions)
    "Z": "2",  # Z     → two (in digit positions)
    "D": "0",  # D     → zero (in digit positions)
    "Q": "0",  # Q     → zero (in digit positions)
}

# Regex patterns for license plates — tuned for Indian plates.
PLATE_PATTERNS = [
    # Standard Indian: KA01AB1234 / KA-01-AB-1234 / KA 01 AB 1234
    re.compile(r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}'),
    # Condensed Indian (OCR output, no spaces): KA02HM1826
    re.compile(r'[A-Z]{1,3}\d{1,2}[A-Z]{1,3}\d{1,5}'),
    # Older Indian format (no series letters): MH 12 1234
    re.compile(r'[A-Z]{2}\s*\d{2}\s*\d{4}'),
]


def _fix_indian_plate(raw: str) -> Optional[str]:
    """
    Apply Indian-plate-aware OCR post-correction.

    Indian plates follow: SS DD LLL DDDD
      SS   = 2-letter state code (must be valid)
      DD   = 2-digit district code
      LLL  = 1-3 letter series
      DDDD = 1-4 digit vehicle number

    Common OCR errors:
      - 'X' read instead of 'K' (KA → XA or just X)
      - 'HH' instead of 'MH' (M misread as H)
      - '0' / 'O' confusion in wrong positions
    """
    if not raw or len(raw) < 6:
        return None

    text = re.sub(r'[^A-Z0-9]', '', raw.strip().upper())
    if len(text) < 6:
        return None

    # ── Step 1: Try to extract state code (first 2 letters) ──────────────
    state = text[:2]

    # If only 1 letter before digits (e.g., "X02HH7256" — K+A merged to X),
    # try to expand known single-char → state mappings
    if text[1].isdigit():
        # Only 1 letter before digits — try common expansions
        single_char_to_states = {
            "X": "KA",  # Very common: KA merged → X
            "K": "KA",
            "M": "MH",
            "D": "DL",
            "T": "TN",
            "A": "AP",
            "G": "GJ",
            "R": "RJ",
            "U": "UP",
            "W": "WB",
            "H": "HR",
            "P": "PB",
            "J": "JH",
            "C": "CG",
            "N": "NL",
            "B": "BR",
            "S": "SK",
        }
        first_char = text[0]
        if first_char in single_char_to_states:
            expanded = single_char_to_states[first_char]
            text = expanded + text[1:]
            state = expanded

    # ── Step 2: Fix common state code misreads ───────────────────────────
    state_corrections = {
        # Common OCR errors in state codes
        "XA": "KA",  "X4": "KA",  "KR": "KA",  "K4": "KA",
        "HH": "MH",  "NH": "MH",  "WH": "MH",
        "0L": "DL",  "OL": "DL",  "D1": "DL",
        "7N": "TN",  "IN": "TN",
        "6J": "GJ",  "GI": "GJ",
        "RJ": "RJ",  "R1": "RJ",
        "U9": "UP",  "U0": "UP",
        "W8": "WB",  "WR": "WB",
        "H8": "HR",  "HK": "HR",
        "P8": "PB",  "PR": "PB",
        "8R": "BR",
        "C6": "CG",
    }

    if state in state_corrections:
        corrected = state_corrections[state]
        text = corrected + text[2:]
        state = corrected

    # ── Step 3: Validate state code ──────────────────────────────────────
    if state not in INDIAN_STATE_CODES:
        return None

    # ── Step 4: Fix digit/letter confusion in known positions ────────────
    # Format: SS DD LLL DDDD
    # Positions 2-3 must be digits (district code)
    chars = list(text)
    for i in [2, 3]:
        if i < len(chars) and chars[i].isalpha():
            if chars[i] in OCR_CHAR_CORRECTIONS:
                replacement = OCR_CHAR_CORRECTIONS[chars[i]]
                if replacement.isdigit():
                    chars[i] = replacement

    # Find where the series letters start (after district digits)
    # and where the final number starts
    district_end = 2
    while district_end < len(chars) and chars[district_end].isdigit():
        district_end += 1
        if district_end > 4:  # Max 2 district digits
            break

    # Series letters: fix digits that should be letters
    series_end = district_end
    while series_end < len(chars) and chars[series_end].isalpha():
        series_end += 1
        if series_end - district_end > 3:  # Max 3 series letters
            break

    for i in range(district_end, min(series_end, len(chars))):
        if chars[i].isdigit() and chars[i] in OCR_CHAR_CORRECTIONS:
            replacement = OCR_CHAR_CORRECTIONS[chars[i]]
            if replacement.isalpha():
                chars[i] = replacement

    # Final number digits: fix letters that should be digits
    for i in range(series_end, len(chars)):
        if chars[i].isalpha() and chars[i] in OCR_CHAR_CORRECTIONS:
            replacement = OCR_CHAR_CORRECTIONS[chars[i]]
            if replacement.isdigit():
                chars[i] = replacement

    result = "".join(chars)

    # ── Step 5: Final format validation ──────────────────────────────────
    # Must be: 2 letters + 1-2 digits + 1-3 letters + 1-4 digits
    final_pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$')
    if final_pattern.match(result):
        return result

    # Also accept older format: 2 letters + 2 digits + 4 digits (no series)
    old_pattern = re.compile(r'^[A-Z]{2}\d{2}\d{4}$')
    if old_pattern.match(result):
        return result

    return None


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
                if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 20 or vehicle_crop.shape[1] < 20:
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
            conf=0.25,
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

        # Focus on lower 70% of vehicle (plate area) — wider region for angled views
        plate_region = vehicle_crop[int(h * 0.3):, :]

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
                # Relaxed: allow aspect ratio 1.2-8.0 for angled plates, lower min width
                if 1.2 < aspect_ratio < 8.0 and cw > 30 and ch > 8:
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
            if w < 120:
                scale = 120 / w
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)

            # Try deskewing for angled plates
            try:
                coords = np.column_stack(np.where(enhanced > 0))
                if len(coords) > 5:
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = -(90 + angle)
                    else:
                        angle = -angle
                    if abs(angle) > 2 and abs(angle) < 45:
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        enhanced = cv2.warpAffine(
                            enhanced, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
            except Exception:
                pass

            results = self._ocr.readtext(enhanced, detail=1)

            if not results:
                # Try on original color image too
                results = self._ocr.readtext(image, detail=1)

            if not results:
                return None, 0.0

            # Find the best match that looks like a plate
            for (bbox, text, confidence) in sorted(results, key=lambda r: r[2], reverse=True):
                cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

                # Try Indian plate correction first
                corrected = _fix_indian_plate(cleaned)
                if corrected:
                    return corrected, confidence

                # Fallback: raw pattern match (for non-Indian plates)
                for pattern in PLATE_PATTERNS:
                    if pattern.match(cleaned):
                        return cleaned, confidence

            # No pattern matched — reject this reading entirely

        except Exception as e:
            logger.debug(f"OCR failed on region: {e}")

        return None, 0.0

    def _normalize_plate(self, text: str) -> Optional[str]:
        """Clean, correct, and normalize plate text using Indian plate rules."""
        if not text:
            return None

        cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

        # Try Indian plate correction (handles OCR errors, validates state code)
        corrected = _fix_indian_plate(cleaned)
        if corrected:
            return corrected

        # Fallback: basic validation for any plate format
        if len(cleaned) < 6:
            return None

        has_letters = any(c.isalpha() for c in cleaned)
        has_digits = any(c.isdigit() for c in cleaned)

        if not (has_letters and has_digits):
            return None

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
