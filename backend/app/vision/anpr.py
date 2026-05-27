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

# Additional corrections for oblique CCTV angles.
# These map wrong_char → digit_replacement, letter_replacement
OCR_OBLIQUE_CORRECTIONS = {
    # Digit → letter corrections (when a digit is in a letter position)
    "3": "J",  # 3 misread as J at oblique angles
    "4": "A",  # 4 misread as A
    "7": "T",  # 7 misread as T
    "9": "G",  # 9 misread as G
    # Letter → digit corrections (when a letter is in a digit position)
    "J": "3",  # J is often a misread 3
    "A": "4",  # A is often a misread 4
    "U": "0",  # U is often a misread 0
    "L": "1",  # L is often a misread 1
    "T": "7",  # T is often a misread 7
    "N": "4",  # N can be misread 4
    "H": "4",  # H can be misread 4
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


def _merge_ocr_fragments(results) -> str:
    """
    Merge EasyOCR text fragments in left-to-right order.

    EasyOCR often reads Indian plates as separate fragments
    (e.g., "KA", "53", "MK", "0449"). This function concatenates
    them in spatial order to reconstruct the full plate text.
    """
    if not results:
        return ""

    # Sort by x-coordinate of leftmost point in bbox
    sorted_results = sorted(results, key=lambda r: min(p[0] for p in r[0]))

    merged = ""
    for bbox, text, conf in sorted_results:
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text.strip())
        merged += cleaned.upper()

    return merged


def _position_aware_fix(text: str) -> Optional[str]:
    """
    Apply position-aware character correction for Indian plates.

    Indian plate format: AA DD AA/A DDDD
    - Positions 0-1: must be letters (state code)
    - Positions 2-3: must be digits (district code)
    - Positions 4-5 (or 4-6): must be letters (series)
    - Last 4 positions: must be digits (vehicle number)

    Uses OCR_OBLIQUE_CORRECTIONS for CCTV-specific misreads.
    """
    if not text or len(text) < 8:
        return None

    chars = list(text)

    # Fix positions 0-1: must be letters
    for i in [0, 1]:
        if chars[i].isdigit():
            if chars[i] in OCR_CHAR_CORRECTIONS:
                chars[i] = OCR_CHAR_CORRECTIONS[chars[i]]

    # Fix positions 2-3: must be digits
    for i in [2, 3]:
        if i < len(chars) and chars[i].isalpha():
            if chars[i] in OCR_CHAR_CORRECTIONS:
                chars[i] = OCR_CHAR_CORRECTIONS[chars[i]]
            elif chars[i] in OCR_OBLIQUE_CORRECTIONS:
                chars[i] = OCR_OBLIQUE_CORRECTIONS[chars[i]]

    # Determine series length (1-3 letters after district)
    # and fix those positions to be letters
    series_start = 4
    series_end = series_start
    while series_end < len(chars) and series_end < series_start + 3:
        c = chars[series_end]
        if c.isalpha():
            series_end += 1
        elif c.isdigit() and c in OCR_CHAR_CORRECTIONS and OCR_CHAR_CORRECTIONS[c].isalpha():
            chars[series_end] = OCR_CHAR_CORRECTIONS[c]
            series_end += 1
        else:
            break

    # Fix remaining positions (vehicle number): must be digits
    for i in range(series_end, len(chars)):
        if chars[i].isalpha():
            if chars[i] in OCR_CHAR_CORRECTIONS:
                replacement = OCR_CHAR_CORRECTIONS[chars[i]]
                if replacement.isdigit():
                    chars[i] = replacement
            elif chars[i] in OCR_OBLIQUE_CORRECTIONS:
                replacement = OCR_OBLIQUE_CORRECTIONS[chars[i]]
                if replacement.isdigit():
                    chars[i] = replacement

    return "".join(chars)


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
        # CCTV oblique angle: TrOCR reads KA as KK, KS, KN, etc.
        "KK": "KA",  "KS": "KA",  "KN": "KA",  "KH": "KA",
        "HH": "MH",  "NH": "MH",  "WH": "MH",
        "0L": "DL",  "OL": "DL",  "D1": "DL",
        "7N": "TN",  "IN": "TN",
        "6J": "GJ",  "GI": "GJ",
        "RJ": "RJ",  "R1": "RJ",
        "U9": "UP",  "U0": "UP",  "UP": "UP",
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

    # ── Step 6: Last resort — aggressive position-aware fix ──────────────
    # Use oblique CCTV angle corrections for garbled reads
    position_fixed = _position_aware_fix(text)
    if position_fixed:
        if final_pattern.match(position_fixed):
            pf_state = position_fixed[:2]
            if pf_state in INDIAN_STATE_CODES:
                return position_fixed
        if old_pattern.match(position_fixed):
            pf_state = position_fixed[:2]
            if pf_state in INDIAN_STATE_CODES:
                return position_fixed

    return None


class PlateRecognizer:
    """
    License Plate Recognition (ANPR) using multi-stage detection pipeline.

    Pipeline:
    1. YOLOv8 COCO model detects vehicles in full frame
    2. Custom plate model detects plate regions within each vehicle crop
    3. Real-ESRGAN 4x super-resolution enhances plate crops
    4. EasyOCR reads text from enhanced plate regions
    5. Indian plate format correction and normalization

    Features:
    - Two-stage detection for high accuracy
    - Real-ESRGAN super-resolution for low-res CCTV plate enhancement
    - Direct plate-region OCR fallback when plate model fails
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
        self._esrgan = None           # Real-ESRGAN for plate super-resolution
        self._esrgan_available = False
        self._trocr_model = None       # TrOCR transformer for plate text reading
        self._trocr_processor = None
        self._trocr_available = False
        self._qwen_model = None        # Qwen2-VL for VLM-based plate reading
        self._qwen_processor = None
        self._qwen_available = False

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
        self._counter_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="anpr",
        )

        # Cooldown tracking: plate_text → last_alert_timestamp
        self._plate_cooldowns: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        # ── Consensus Voting State ────────────────────────────────
        # Track vehicles across frames by bounding box overlap.
        # Each tracked vehicle accumulates OCR readings until we have
        # enough votes to emit a high-confidence final plate.
        self._tracked_vehicles: Dict[int, dict] = {}  # track_id → state
        self._next_track_id = 0
        self._track_lock = threading.Lock()
        self._consensus_min_votes = 3      # Min readings before emitting
        self._consensus_max_votes = 10     # Stop after this many reads
        self._track_iou_threshold = 0.15   # IoU to match same vehicle (low for fast movers)
        self._track_centroid_dist = 200    # Centroid distance fallback (pixels)
        self._track_max_age_sec = 5.0      # Expire tracks after no match

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
                self._vehicle_model = YOLO("yolov8m.pt")
                self._vehicle_model_available = True
                logger.info(f"Vehicle detection model loaded (COCO yolov8m) on {self._device}")
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
                
                # Load Real-ESRGAN for 4x super-resolution
                try:
                    import sys, types
                    from torchvision.transforms import functional as F
                    # Patch for newer torchvision versions
                    sys.modules['torchvision.transforms.functional_tensor'] = types.ModuleType('torchvision.transforms.functional_tensor')
                    sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale = F.rgb_to_grayscale
                    
                    from realesrgan import RealESRGANer
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                    self._esrgan = RealESRGANer(
                        scale=4,
                        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                        model=esrgan_model,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=False,
                        device='cuda' if use_gpu else 'cpu',
                    )
                    self._esrgan_available = True
                    logger.info("Real-ESRGAN loaded successfully for plate upscaling")
                except Exception as e:
                    self._esrgan_available = False
                    logger.warning(f"Failed to load Real-ESRGAN, falling back to cv2.resize: {e}")
                
                # Load TrOCR for license plate text recognition
                try:
                    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                    self._trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                    try:
                        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                            'DunnBC22/trocr-base-printed_license_plates_ocr'
                        )
                        logger.info("License-plate-finetuned TrOCR loaded")
                    except Exception:
                        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                            'microsoft/trocr-base-printed'
                        )
                        logger.info("Base TrOCR loaded (plate-finetuned unavailable)")
                    self._trocr_model = self._trocr_model.to(
                        'cuda' if use_gpu else 'cpu'
                    ).eval()
                    self._trocr_available = True
                    logger.info("TrOCR loaded for plate text recognition")
                except Exception as e:
                    self._trocr_available = False
                    logger.warning(f"Failed to load TrOCR: {e}")

                # Load Qwen2-VL for VLM-based plate reading (best accuracy)
                try:
                    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor as QwenAutoProcessor
                    logger.info("Loading Qwen2-VL-2B-Instruct for plate OCR...")
                    self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        'Qwen/Qwen2-VL-2B-Instruct',
                        torch_dtype=torch.float16,
                        device_map='auto',
                    )
                    self._qwen_processor = QwenAutoProcessor.from_pretrained(
                        'Qwen/Qwen2-VL-2B-Instruct'
                    )
                    self._qwen_available = True
                    logger.info("Qwen2-VL loaded — VLM plate reading enabled")
                except Exception as e:
                    self._qwen_available = False
                    logger.warning(f"Failed to load Qwen2-VL: {e}")

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

    def _trocr_read(self, image_bgr: np.ndarray) -> str:
        """Run TrOCR on a BGR image crop. Returns recognized text."""
        if not self._trocr_available:
            return ""
        try:
            from PIL import Image as PILImage
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            device = next(self._trocr_model.parameters()).device
            pixel_values = self._trocr_processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values.to(device)
            import torch
            with torch.no_grad():
                generated_ids = self._trocr_model.generate(
                    pixel_values, max_new_tokens=20
                )
            text = self._trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return text
        except Exception as e:
            logger.debug(f"TrOCR read failed: {e}")
            return ""

    def _qwen_read_plate(self, image_bgr: np.ndarray) -> str:
        """Use Qwen2-VL to read plate text from a BGR image. Human-like accuracy."""
        if not self._qwen_available:
            return ""
        try:
            import tempfile, os
            from qwen_vl_utils import process_vision_info

            # Save image to a temporary file (Qwen2-VL needs file path)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, image_bgr)
                tmp_path = tmp.name

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{tmp_path}"},
                        {"type": "text", "text": (
                            "Read the vehicle license plate number visible in this image. "
                            "Indian plates follow format: 2 letters, 2 digits, 1-3 letters, 1-4 digits. "
                            "Return ONLY the plate text, nothing else."
                        )},
                    ],
                }
            ]

            text = self._qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._qwen_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(next(self._qwen_model.parameters()).device)

            import torch
            with torch.no_grad():
                generated_ids = self._qwen_model.generate(
                    **inputs, max_new_tokens=30
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self._qwen_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            logger.debug(f"Qwen2-VL plate read: '{output}'")
            return output
        except Exception as e:
            logger.debug(f"Qwen2-VL read failed: {e}")
            return ""

    # ── Frame Submission ──────────────────────────────────────────────────

    def submit_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        predetected_vehicles: Optional[List[dict]] = None,
    ) -> bool:
        """
        Submit a frame for ANPR (non-blocking).
        Only processes every `frame_interval`-th frame.

        Args:
            frame: The video frame
            camera_id: Camera identifier
            predetected_vehicles: Optional list of pre-detected vehicle dicts
                from TrafficMonitor. Each dict should have:
                  {'bbox': {'x1':..,'y1':..,'x2':..,'y2':..}, 'class_name': str, 'confidence': float}
                If provided, skips the Stage 1 YOLO vehicle detection entirely.

        Returns:
            True if frame was submitted, False if skipped
        """
        with self._counter_lock:
            self._frame_counter += 1
            if self._frame_counter % self.frame_interval != 0:
                return False

        frame_copy = frame.copy()
        self._executor.submit(self._process, frame_copy, camera_id, predetected_vehicles)
        return True

    # ── Main Processing ───────────────────────────────────────────────────

    def _process(self, frame: np.ndarray, camera_id: str, predetected_vehicles: Optional[List[dict]] = None):
        """
        ANPR pipeline with two detection strategies:

        Strategy A (PRIMARY): Full-frame plate detection
          - Run plate detector on the FULL frame at higher resolution (imgsz=1280)
          - This gives larger, cleaner plate crops for OCR
          - Associate detected plates with nearby vehicles by spatial overlap

        Strategy B (FALLBACK): Per-vehicle-crop plate detection
          - Crop each vehicle → run plate detector on the crop
          - Falls back to contour extraction or direct OCR if needed

        Strategy A is better for distant/small vehicles where the vehicle crop
        would be too small for reliable plate detection.
        """
        try:
            start_time = time.monotonic()

            # Stage 1: Detect vehicles (skip if pre-detected by TrafficMonitor)
            if predetected_vehicles:
                vehicles = []
                for v in predetected_vehicles:
                    bbox = v.get("bbox", {})
                    vehicles.append({
                        'box': (bbox.get("x1", 0), bbox.get("y1", 0),
                                bbox.get("x2", 0), bbox.get("y2", 0)),
                        'type': v.get("class_name", "vehicle"),
                        'confidence': v.get("confidence", 0.0),
                    })
                logger.debug(f"ANPR: Using {len(vehicles)} pre-detected vehicles from TrafficMonitor")
            else:
                vehicles = self._detect_vehicles(frame)

            if not vehicles:
                return

            logger.debug(f"ANPR: Processing {len(vehicles)} vehicles for plate detection")

            # ══════════════════════════════════════════════════════════════
            # Strategy A: Full-frame plate detection (PRIMARY)
            # ══════════════════════════════════════════════════════════════
            plates_emitted = set()  # Track which vehicles already got plates

            if self._load_plate_model():
                # Run plate detector on full frame at higher resolution
                full_results = self._plate_model(
                    frame, conf=0.20, imgsz=1280, verbose=False, device=self._device,
                )

                full_frame_plates = []
                for result in full_results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        px1, py1, px2, py2 = [int(c) for c in box.xyxy[0].tolist()]
                        pconf = float(box.conf[0])
                        pw, ph = px2 - px1, py2 - py1
                        if pw < 15 or ph < 8:
                            continue
                        full_frame_plates.append({
                            'box': (px1, py1, px2, py2),
                            'confidence': pconf,
                            'crop': frame[max(0,py1):min(frame.shape[0],py2),
                                          max(0,px1):min(frame.shape[1],px2)],
                        })

                if full_frame_plates:
                    logger.info(
                        f"ANPR Strategy A: {len(full_frame_plates)} plates detected "
                        f"on full frame (imgsz=1280)"
                    )

                for plate_det in full_frame_plates:
                    px1, py1, px2, py2 = plate_det['box']
                    plate_crop = plate_det['crop']
                    plate_det_conf = plate_det['confidence']

                    if plate_crop.size == 0:
                        continue

                    crop_h, crop_w = plate_crop.shape[:2]
                    logger.debug(
                        f"ANPR: Full-frame plate at ({px1},{py1},{px2},{py2}) "
                        f"size={crop_w}x{crop_h} conf={plate_det_conf:.3f}"
                    )

                    # OCR the plate
                    plate_text, ocr_confidence = self._ocr_region(plate_crop)

                    if plate_text is None:
                        logger.debug(f"ANPR: OCR returned None for plate at ({px1},{py1})")
                        continue

                    # Normalize
                    normalized = self._normalize_plate(plate_text)
                    if not normalized or len(normalized) < 6:
                        logger.debug(
                            f"ANPR: Normalization rejected '{plate_text}' → '{normalized}'"
                        )
                        continue

                    if ocr_confidence < 0.25:
                        logger.debug(
                            f"ANPR: Low confidence rejected: '{normalized}' conf={ocr_confidence:.3f}"
                        )
                        continue

                    # Find the closest vehicle to this plate
                    pcx = (px1 + px2) / 2
                    pcy = (py1 + py2) / 2
                    best_vehicle = None
                    best_dist = float('inf')

                    for i, vehicle in enumerate(vehicles):
                        vx1, vy1, vx2, vy2 = vehicle['box']
                        # Check if plate center is inside or near the vehicle box
                        # Allow some margin (plates can be slightly outside the vehicle bbox)
                        margin = max(vx2 - vx1, vy2 - vy1) * 0.3
                        if (vx1 - margin <= pcx <= vx2 + margin and
                            vy1 - margin <= pcy <= vy2 + margin):
                            vcx = (vx1 + vx2) / 2
                            vcy = (vy1 + vy2) / 2
                            dist = ((pcx - vcx)**2 + (pcy - vcy)**2)**0.5
                            if dist < best_dist:
                                best_dist = dist
                                best_vehicle = (i, vehicle)

                    if best_vehicle:
                        vi, vehicle = best_vehicle
                        vehicle_type = vehicle['type']
                        vehicle_conf = vehicle['confidence']
                        vx1, vy1, vx2, vy2 = vehicle['box']
                        plates_emitted.add(vi)
                    else:
                        # Plate detected but no matching vehicle — still emit
                        vehicle_type = "vehicle"
                        vehicle_conf = plate_det_conf
                        vx1, vy1, vx2, vy2 = px1, py1, px2, py2

                    # Cooldown check
                    with self._cooldown_lock:
                        now = time.time()
                        is_dup = False
                        expired = [p for p, t in self._plate_cooldowns.items()
                                   if now - t > self.plate_cooldown_sec]
                        for p in expired:
                            del self._plate_cooldowns[p]
                        for seen_plate, last_time in self._plate_cooldowns.items():
                            if self._plate_similarity(normalized, seen_plate) >= 0.6:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        self._plate_cooldowns[normalized] = now

                    self._total_plates += 1

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
                        f"🚗 PLATE: {normalized} "
                        f"({vehicle_type}, conf={ocr_confidence:.2f}) "
                        f"on camera {camera_id} [Strategy A: full-frame]"
                    )

                    if self.on_plate:
                        self.on_plate(event)

            # ══════════════════════════════════════════════════════════════
            # Strategy B: Per-vehicle-crop plate detection (FALLBACK)
            # Only for vehicles that didn't get plates from Strategy A
            # ══════════════════════════════════════════════════════════════
            for vi, vehicle in enumerate(vehicles):
                if vi in plates_emitted:
                    continue  # Already got a plate from Strategy A

                vx1, vy1, vx2, vy2 = vehicle['box']
                vehicle_type = vehicle['type']
                vehicle_conf = vehicle['confidence']

                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 20 or vehicle_crop.shape[1] < 20:
                    continue

                # Stage 2: Find plate within vehicle crop
                plate_regions = self._detect_plates_in_crop(vehicle_crop)

                if not plate_regions:
                    plate_text, plate_confidence = self._extract_plate_contours(vehicle_crop)
                    if plate_text:
                        plate_regions = [{'text': plate_text, 'confidence': plate_confidence}]

                if not plate_regions:
                    plate_text, plate_confidence = self._direct_plate_ocr(vehicle_crop)
                    if plate_text:
                        plate_regions = [{'text': plate_text, 'confidence': plate_confidence}]
                    else:
                        continue

                # Stage 3: OCR each plate region
                for plate_info in plate_regions:
                    if 'crop' in plate_info:
                        plate_text, ocr_confidence = self._ocr_region(plate_info['crop'])
                    elif 'text' in plate_info:
                        plate_text = plate_info['text']
                        ocr_confidence = plate_info['confidence']
                    else:
                        continue

                    if plate_text is None:
                        continue

                    normalized = self._normalize_plate(plate_text)
                    if not normalized or len(normalized) < 6:
                        logger.debug(
                            f"ANPR Strategy B: Normalization rejected '{plate_text}'"
                        )
                        continue

                    if ocr_confidence < 0.25:
                        continue

                    with self._cooldown_lock:
                        now = time.time()
                        is_dup = False
                        expired = [p for p, t in self._plate_cooldowns.items()
                                   if now - t > self.plate_cooldown_sec]
                        for p in expired:
                            del self._plate_cooldowns[p]
                        for seen_plate, last_time in self._plate_cooldowns.items():
                            if self._plate_similarity(normalized, seen_plate) >= 0.6:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        self._plate_cooldowns[normalized] = now

                    self._total_plates += 1

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
                        f"🚗 PLATE: {normalized} "
                        f"({vehicle_type}, conf={ocr_confidence:.2f}) "
                        f"on camera {camera_id} [Strategy B: vehicle-crop]"
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

    # ── Consensus Voting ──────────────────────────────────────────────────

    @staticmethod
    def _box_iou(box1: tuple, box2: tuple) -> float:
        """Compute IoU between two (x1, y1, x2, y2) boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _match_track(self, box: tuple) -> Optional[int]:
        """Find existing track matching this bounding box by IoU or centroid distance."""
        best_id, best_score = None, 0.0
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        for track_id, state in self._tracked_vehicles.items():
            # Try IoU first
            iou = self._box_iou(box, state['last_box'])
            if iou >= self._track_iou_threshold and iou > best_score:
                best_id, best_score = track_id, iou
                continue

            # Fallback: centroid distance (handles fast-moving vehicles)
            tb = state['last_box']
            tcx = (tb[0] + tb[2]) / 2
            tcy = (tb[1] + tb[3]) / 2
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if dist < self._track_centroid_dist and iou > 0:
                # Close enough AND at least some overlap
                score = 0.1  # lower priority than IoU match
                if score > best_score or best_id is None:
                    best_id, best_score = track_id, score

        return best_id

    def _add_vote(
        self,
        vehicle_box: tuple,
        plate_text: str,
        ocr_confidence: float,
        vehicle_type: str,
        vehicle_conf: float,
        camera_id: str,
    ):
        """Add an OCR reading to the consensus voting buffer for a vehicle."""
        now = time.time()

        with self._track_lock:
            # Expire old tracks
            expired = [
                tid for tid, s in self._tracked_vehicles.items()
                if now - s['last_seen'] > self._track_max_age_sec
            ]
            for tid in expired:
                # Before removing, emit whatever we have
                self._emit_consensus(tid)
                del self._tracked_vehicles[tid]

            # Find matching track
            track_id = self._match_track(vehicle_box)

            if track_id is None:
                # New vehicle — create track
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracked_vehicles[track_id] = {
                    'last_box': vehicle_box,
                    'last_seen': now,
                    'votes': [],
                    'vehicle_type': vehicle_type,
                    'vehicle_conf': vehicle_conf,
                    'camera_id': camera_id,
                    'emitted': False,
                }

            track = self._tracked_vehicles[track_id]
            track['last_box'] = vehicle_box
            track['last_seen'] = now
            track['votes'].append((plate_text, ocr_confidence))

            # Check if we have enough votes to emit
            n_votes = len(track['votes'])
            if not track['emitted'] and n_votes >= self._consensus_min_votes:
                self._emit_consensus(track_id)

            # Stop accumulating after max votes
            if n_votes >= self._consensus_max_votes:
                if not track['emitted']:
                    self._emit_consensus(track_id)

    @staticmethod
    def _plate_similarity(a: str, b: str) -> float:
        """Compute similarity between two plate strings using Levenshtein ratio.
        Returns 0.0-1.0 where 1.0 = identical.
        Handles insertions/deletions/substitutions correctly.
        """
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        # Levenshtein distance (dynamic programming)
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0.0
        # Create distance matrix
        d = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            d[i][0] = i
        for j in range(lb + 1):
            d[0][j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        # Convert distance to similarity ratio
        max_len = max(la, lb)
        return 1.0 - (d[la][lb] / max_len)

    def _emit_consensus(self, track_id: int):
        """Compute majority vote and emit the plate event."""
        track = self._tracked_vehicles.get(track_id)
        if not track or track['emitted'] or not track['votes']:
            return

        # Group similar votes (e.g. CH01COO109 and CH01CO0109 → same group)
        from collections import Counter
        vote_counts = Counter()
        conf_sums = {}
        for plate, conf in track['votes']:
            # Check if this plate is similar to an existing vote
            merged = False
            for existing in list(vote_counts.keys()):
                if self._plate_similarity(plate, existing) >= 0.7:
                    # Merge into the existing key (keeps the first-seen spelling)
                    vote_counts[existing] += 1
                    conf_sums[existing] = conf_sums.get(existing, 0) + conf
                    merged = True
                    break
            if not merged:
                vote_counts[plate] += 1
                conf_sums[plate] = conf_sums.get(plate, 0) + conf

        # Winner = most votes, break ties by total confidence
        winner = max(
            vote_counts.keys(),
            key=lambda p: (vote_counts[p], conf_sums[p])
        )
        total_votes = len(track['votes'])
        winner_votes = vote_counts[winner]
        avg_conf = conf_sums[winner] / winner_votes

        # Consensus confidence = base OCR conf * vote ratio
        consensus_conf = avg_conf * (winner_votes / total_votes)

        # Fuzzy cooldown check — if any recently seen plate is >70% similar, skip
        with self._cooldown_lock:
            now = time.time()
            for seen_plate, last_time in self._plate_cooldowns.items():
                if now - last_time < self.plate_cooldown_sec:
                    if self._plate_similarity(winner, seen_plate) >= 0.7:
                        track['emitted'] = True
                        return
            self._plate_cooldowns[winner] = now

        track['emitted'] = True
        self._total_plates += 1

        event = {
            "event_type": "vehicle_entry",
            "camera_id": track['camera_id'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "plate": winner,
                "confidence": round(consensus_conf, 3),
                "vehicle_type": track['vehicle_type'],
                "vehicle_confidence": round(track['vehicle_conf'], 3),
                "bounding_box": {
                    "x1": track['last_box'][0],
                    "y1": track['last_box'][1],
                    "x2": track['last_box'][2],
                    "y2": track['last_box'][3],
                },
                "consensus": {
                    "total_votes": total_votes,
                    "winner_votes": winner_votes,
                    "all_candidates": dict(vote_counts.most_common(5)),
                },
            },
        }

        logger.info(
            f"🚗 PLATE CONFIRMED: {winner} "
            f"({winner_votes}/{total_votes} votes, conf={consensus_conf:.2f}) "
            f"on camera {track['camera_id']}"
        )

        if self.on_plate:
            self.on_plate(event)

    # ── Stage 2: Plate Detection within Vehicle ──────────────────────────

    def _detect_plates_in_crop(self, vehicle_crop: np.ndarray) -> List[dict]:
        """
        Detect plate regions within a vehicle crop.
        
        Uses the custom plate detection model if available.
        Returns list of plate regions with their crops.
        
        IMPORTANT: The old model had 3 classes (Matricula, licenseplate, object)
        and class 2 ('object') had to be excluded. The new YOLOv11-L model from
        morsetechlab has a single class (0 = 'License_Plate'), so we accept all
        detections without class filtering.
        """
        if not self._load_plate_model():
            return []

        h, w = vehicle_crop.shape[:2]
        vehicle_area = h * w

        # Run plate detector — accept all classes (new model is single-class)
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
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]

                # Clamp to vehicle crop bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                plate_w = x2 - x1
                plate_h = y2 - y1
                plate_area = plate_w * plate_h

                # Skip crops that are too small to contain text
                if plate_w < 15 or plate_h < 8:
                    continue

                # Skip crops that are too large (>80% of vehicle = probably a false positive)
                # NOTE: Raised from 40% → 80% for YOLOv11-L model which only detects
                # actual plates. For distant/small vehicle crops, the plate legitimately
                # takes up 50-60% of the tight vehicle bounding box.
                if vehicle_area > 0 and plate_area / vehicle_area > 0.80:
                    logger.debug(
                        f"Skipping oversized plate detection: "
                        f"{plate_w}x{plate_h} = {plate_area/vehicle_area*100:.0f}% of vehicle"
                    )
                    continue

                plate_crop = vehicle_crop[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                plates.append({
                    'crop': plate_crop,
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': cls_id,
                })

        return plates

    # ── Stage 2b: Direct plate-region OCR (no plate model) ────────────────

    def _direct_plate_ocr(
        self,
        vehicle_crop: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Direct OCR on the plate area of a vehicle crop.

        When the plate detection model fails (common with low-res CCTV),
        extract the bottom half of the vehicle (where Indian plates are)
        and run OCR with aggressive preprocessing.

        Returns:
            (plate_text, confidence) or (None, 0.0) if no plate found
        """
        if not self._load_ocr():
            return None, 0.0

        try:
            h, w = vehicle_crop.shape[:2]
            if h < 40 or w < 40:
                return None, 0.0

            # Indian plates are on the bumper — focus on bottom 50%
            plate_area = vehicle_crop[int(h * 0.5):, :]
            ph, pw = plate_area.shape[:2]

            if ph < 15 or pw < 30:
                return None, 0.0

            # Super-resolution: skip ESRGAN for large plates (speed)
            if pw >= 100:
                # Plate already large enough — just bicubic upscale
                scale = max(2.0, 300 / pw)
                upscaled = cv2.resize(plate_area, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)
            elif self._esrgan_available:
                try:
                    upscaled, _ = self._esrgan.enhance(plate_area, outscale=4)
                except Exception as e:
                    logger.debug(f"ESRGAN enhance failed, falling back: {e}")
                    scale = max(3.0, 400 / pw)
                    upscaled = cv2.resize(plate_area, None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_CUBIC)
            else:
                scale = max(3.0, 400 / pw)
                upscaled = cv2.resize(plate_area, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)

            # Collect ALL candidates from multiple engines/preprocessing
            candidates = []  # list of (plate_text, confidence, source)

            # Note: Qwen2-VL removed from real-time pipeline for speed.
            # It's available via _qwen_read_plate() for dashboard verify.

            # EasyOCR only (fast path for live CCTV)
            # TrOCR removed from real-time — too slow (~500ms per call)
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

            for img_label, img in [("color", upscaled), ("bilateral", bilateral)]:
                results = self._ocr.readtext(img, detail=1)

                # EasyOCR: individual fragments
                for (bbox, text, confidence) in results:
                    cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                    corrected = _fix_indian_plate(cleaned)
                    if corrected:
                        candidates.append((corrected, confidence, f"easyocr_{img_label}"))

                # EasyOCR: merged fragments
                if results:
                    merged = _merge_ocr_fragments(results)
                    if merged and len(merged) >= 6:
                        corrected = _fix_indian_plate(merged)
                        if corrected:
                            avg_conf = sum(r[2] for r in results) / len(results)
                            candidates.append((corrected, avg_conf, f"easyocr_merged_{img_label}"))

            # Pick the best candidate: score by plate length + format validity
            if candidates:
                def _score(c):
                    plate, conf, src = c
                    score = len(plate) * 10  # longer = better
                    score += conf * 5        # higher confidence = better
                    # Bonus for full Indian format (10 chars like KA53MK0449)
                    if len(plate) >= 10:
                        score += 20
                    elif len(plate) >= 8:
                        score += 10
                    return score

                best = max(candidates, key=_score)
                logger.debug(
                    f"Direct plate OCR: {len(candidates)} candidates, "
                    f"best={best[0]} (conf={best[1]:.2f}, src={best[2]})"
                )
                return best[0], best[1]

            return None, 0.0

        except Exception as e:
            logger.debug(f"Direct plate OCR failed: {e}")
            return None, 0.0

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

            # Step 1: Super-resolution with Real-ESRGAN (AI 4x upscale)
            h_orig, w_orig = image.shape[:2]
            if self._esrgan_available and w_orig < 200:
                try:
                    sr_image, _ = self._esrgan.enhance(image, outscale=4)
                    image = sr_image  # Use the super-resolved image for all further processing
                except Exception as e:
                    logger.debug(f"ESRGAN enhance failed in OCR, using fallback: {e}")

            # Step 2: Enhance contrast for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Adaptive CLAHE: use smaller grid for tiny crops
            h, w = gray.shape[:2]
            grid_w = max(1, min(8, w // 8))
            grid_h = max(1, min(8, h // 8))
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(grid_w, grid_h))
            enhanced = clahe.apply(gray)

            # Resize if still small after ESRGAN
            h, w = enhanced.shape[:2]
            if w < 200:
                scale = 200 / w
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)

            # Apply sharpening to bring out text edges
            sharpen_kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

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
                        eh, ew = enhanced.shape[:2]
                        center = (ew // 2, eh // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        enhanced = cv2.warpAffine(
                            enhanced, M, (ew, eh),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
            except Exception:
                pass

            # Note: Qwen2-VL removed from real-time pipeline for speed.
            # Available as _qwen_read_plate() for on-demand dashboard verify.

            # Priority 2: Multi-pass EasyOCR
            results = self._ocr.readtext(enhanced, detail=1)

            if not results:
                results = self._ocr.readtext(sharpened, detail=1)

            if not results:
                # Try on the (possibly ESRGAN-enhanced) color image
                if image.shape[1] < 200:
                    scale = 200 / image.shape[1]
                    upscaled_color = cv2.resize(image, None, fx=scale, fy=scale,
                                                interpolation=cv2.INTER_CUBIC)
                else:
                    upscaled_color = image
                results = self._ocr.readtext(upscaled_color, detail=1)

            if not results:
                logger.debug("ANPR OCR: EasyOCR returned no results on any pass")
                return None, 0.0

            # Log what EasyOCR actually read
            raw_texts = [(text.strip(), round(conf, 3)) for (_, text, conf) in results]
            logger.debug(f"ANPR OCR raw reads: {raw_texts}")

            # Find the best match that looks like a plate
            best_text = None
            best_conf = 0.0

            for (bbox, text, confidence) in sorted(results, key=lambda r: r[2], reverse=True):
                cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

                # Try Indian plate correction first
                corrected = _fix_indian_plate(cleaned)
                if corrected and confidence > best_conf:
                    best_text = corrected
                    best_conf = confidence

            # Try EasyOCR fragment merging
            if not best_text:
                merged = _merge_ocr_fragments(results)
                if merged and len(merged) >= 6:
                    corrected = _fix_indian_plate(merged)
                    if corrected:
                        best_text = corrected
                        best_conf = sum(r[2] for r in results) / len(results)

            # Fallback: if no Indian-format match, return the raw merged text
            # Let _normalize_plate decide whether to accept it
            if not best_text:
                merged = _merge_ocr_fragments(results)
                if merged and len(merged) >= 5:
                    # Return raw text — _normalize_plate will do final filtering
                    best_text = re.sub(r'[^A-Z0-9]', '', merged.upper())
                    best_conf = sum(r[2] for r in results) / len(results)
                    logger.debug(
                        f"ANPR OCR: No Indian plate match, returning raw: '{best_text}' "
                        f"conf={best_conf:.3f}"
                    )

            # TrOCR removed from real-time for speed.
            # Available via manual verify endpoint.

            if best_text:
                return best_text, best_conf

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
            logger.debug(f"ANPR normalize: too short ({len(cleaned)}): '{cleaned}'")
            return None

        has_letters = any(c.isalpha() for c in cleaned)
        has_digits = any(c.isdigit() for c in cleaned)

        if not (has_letters and has_digits):
            logger.debug(f"ANPR normalize: no mixed chars: '{cleaned}'")
            return None

        # Check known plate patterns
        for pattern in PLATE_PATTERNS:
            if pattern.match(cleaned):
                return cleaned

        # Last resort: accept any mixed alphanumeric ≥6 chars
        # The plate might be from a non-Indian format or OCR might have
        # minor errors that prevent pattern matching. Better to show it
        # than to silently discard it.
        logger.debug(
            f"ANPR normalize: no pattern match but accepting: '{cleaned}'"
        )
        return cleaned

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
