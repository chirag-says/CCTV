"""
Traffic Monitor — Human-Vehicle Safety & Traffic Analytics.

Detects proximity between persons and vehicles using RF-DETR (primary)
or YOLOv8 (fallback). RF-DETR enables person+vehicle detection in a
single model for proximity safety alerts.

# RF-DETR uses COCO classes, mapped to our naming:
#   - 0: person
#   - 1: bicycle → bike
#   - 2: car
#   - 3: motorcycle → bike
#   - 5: bus
#   - 7: truck
#
# Custom gate_vehicles YOLO model (fallback):
#   - 0: bike, 1: car, 2: auto, 3: bus, 4: truck

This module gracefully handles missing dependencies.
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import Optional, Dict, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── RF-DETR COCO class mapping ────────────────────────────────────────────────
COCO_PERSON_CLASS = 0
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    1: "bike",        # bicycle
    2: "car",
    3: "bike",        # motorcycle → bike
    5: "bus",
    7: "truck",
}
COCO_VEHICLE_IDS = list(COCO_VEHICLE_CLASSES.keys())

# ── Custom YOLO model class mapping (fallback) ───────────────────────────────
PERSON_CLASS = -1  # Disabled in custom model
VEHICLE_CLASSES: Dict[int, str] = {
    0: "bike",
    1: "car",
    2: "auto",
    3: "bus",
    4: "truck",
}

ALL_TRAFFIC_CLASSES = list(VEHICLE_CLASSES.keys())

# ── Roboflow Indian Vehicles Dataset class mapping ────────────────────────────
# Maps Roboflow class names to our internal naming convention
ROBOFLOW_CLASS_MAP: Dict[str, str] = {
    "Car": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Two-wheelers": "bike",
    "Three-wheeler": "auto",
    "LCV": "truck",      # Light Commercial Vehicle → truck
    "Multi-axle": "truck",
}

# Per-class confidence thresholds — truck needs higher conf to reduce false positives
CLASS_CONFIDENCE: Dict[int, float] = {
    0: 0.35,   # bike
    1: 0.35,   # car
    2: 0.30,   # auto
    3: 0.45,   # bus
    4: 0.55,   # truck
}

# Minimum pixel movement between frames to count as "moving"
MOTION_THRESHOLD_PX = 15

# ── Vehicle Color Detection ──────────────────────────────────────────────────
# Named color ranges in HSV space for vehicle color classification
COLOR_RANGES = [
    # (name, lower_hsv, upper_hsv)
    ("Red",       np.array([0, 70, 50]),    np.array([10, 255, 255])),
    ("Red",       np.array([170, 70, 50]),   np.array([180, 255, 255])),
    ("Orange",    np.array([11, 70, 50]),    np.array([25, 255, 255])),
    ("Yellow",    np.array([26, 70, 50]),    np.array([35, 255, 255])),
    ("Green",     np.array([36, 40, 40]),    np.array([85, 255, 255])),
    ("Blue",      np.array([86, 40, 40]),    np.array([130, 255, 255])),
    ("Purple",    np.array([131, 40, 40]),   np.array([169, 255, 255])),
]

def detect_vehicle_color(crop: np.ndarray) -> str:
    """
    Detect the dominant color of a vehicle from its image crop.
    Uses HSV histogram analysis with predefined color ranges.
    Falls back to brightness-based classification for achromatic colors.

    Returns a human-readable color name string.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return "Unknown"

    try:
        # Focus on the center 60% of the crop (avoid background)
        h, w = crop.shape[:2]
        y1 = int(h * 0.2)
        y2 = int(h * 0.8)
        x1 = int(w * 0.2)
        x2 = int(w * 0.8)
        center = crop[y1:y2, x1:x2]

        if center.size == 0:
            center = crop

        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

        # Count pixels in each color range
        color_counts: Dict[str, int] = {}
        total_colored = 0
        for name, lower, upper in COLOR_RANGES:
            mask = cv2.inRange(hsv, lower, upper)
            count = int(cv2.countNonZero(mask))
            color_counts[name] = color_counts.get(name, 0) + count
            total_colored += count

        total_pixels = center.shape[0] * center.shape[1]

        # If less than 15% of pixels match any color → achromatic
        if total_colored < total_pixels * 0.15:
            # Classify by average value (brightness)
            avg_v = float(np.mean(hsv[:, :, 2]))
            avg_s = float(np.mean(hsv[:, :, 1]))
            if avg_v < 60:
                return "Black"
            elif avg_v > 180 and avg_s < 40:
                return "White"
            elif avg_s < 35:
                return "Silver" if avg_v > 120 else "Grey"
            else:
                return "Grey"

        # Return the dominant color
        best_color = max(color_counts, key=color_counts.get)
        return best_color

    except Exception:
        return "Unknown"


class TrafficMonitor:
    """
    Human-Vehicle safety monitor using RF-DETR (primary) or YOLOv8 (fallback).

    Features:
    - Detects persons and vehicles in the same frame
    - Calculates bounding box proximity/overlap
    - Triggers high-priority safety_alert when person is too close to vehicle
    - Vehicle counting and traffic flow statistics
    - Frame skipping for performance
    - Alert cooldown to prevent spam
    - Non-blocking via ThreadPoolExecutor

    Usage:
        monitor = TrafficMonitor(on_alert=my_callback)
        monitor.submit_frame(frame, camera_id)  # Non-blocking
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        frame_interval: int = 5,
        proximity_px: int = 50,
        alert_cooldown_sec: float = 15.0,
        on_alert: Optional[Callable] = None,
        on_vehicle_count: Optional[Callable] = None,
        max_workers: int = 1,
        rfdetr_model_size: str = "large",
    ):
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.proximity_px = proximity_px
        self.alert_cooldown_sec = alert_cooldown_sec
        self.on_alert = on_alert
        self.on_vehicle_count = on_vehicle_count
        self.rfdetr_model_size = rfdetr_model_size

        self._model = None
        self._model_loaded = False
        self._model_available = False
        self._model_backend = None  # "roboflow", "rfdetr", or "yolo"
        self._model_lock = threading.Lock()
        self._roboflow_client = None  # Roboflow InferenceHTTPClient

        self._frame_counter = 0
        self._counter_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="traffic-mon",
        )
        # Persistent pool for parallel Roboflow dual-model calls
        self._rf_dual_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="rf-dual",
        )

        self._alert_cooldowns: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        self._total_inferences = 0
        self._total_alerts = 0
        self._last_inference_ms = 0.0
        self._last_vehicle_count = 0
        self._last_person_count = 0
        self._prev_positions: Dict[str, dict] = {}

        self._last_detections: Dict[str, List[dict]] = {
            "persons": [],
            "vehicles": [],
            "alerts": [],
        }
        self._detections_lock = threading.Lock()

        logger.info(
            f"TrafficMonitor initialized: rfdetr_size={rfdetr_model_size}, "
            f"yolo_fallback={model_path}, interval={frame_interval}, "
            f"proximity={proximity_px}px, cooldown={alert_cooldown_sec}s"
        )

    def _load_model(self) -> bool:
        """Lazy-load Roboflow (primary) → RF-DETR → YOLOv8 (fallback). Thread-safe."""
        with self._model_lock:
            if self._model_loaded:
                return self._model_available

            # ── Try Roboflow Indian Vehicles API first ────────────────
            try:
                from app.config import settings as _s
                if getattr(_s, 'ENABLE_ROBOFLOW_DETECTION', False):
                    from inference_sdk import InferenceHTTPClient
                    self._roboflow_client = InferenceHTTPClient(
                        api_url=_s.ROBOFLOW_API_URL,
                        api_key=_s.ROBOFLOW_API_KEY,
                    )
                    # Quick health check — try a tiny test
                    self._model_available = True
                    self._model_backend = "roboflow"
                    self._model_loaded = True
                    logger.info(
                        f"✅ Roboflow Indian Vehicles API loaded — "
                        f"model={_s.ROBOFLOW_MODEL_ID}, "
                        f"confidence={_s.ROBOFLOW_CONFIDENCE_THRESHOLD}"
                    )
                    return True
            except ImportError:
                logger.info("inference-sdk not installed, trying RF-DETR")
            except Exception as e:
                logger.warning(f"Roboflow API init failed: {e}, trying RF-DETR")

            # ── Try RF-DETR ──────────────────────────────────────────
            try:
                from rfdetr import RFDETRBase, RFDETRLarge
                try:
                    from rfdetr import RFDETRNano
                except ImportError:
                    RFDETRNano = RFDETRBase

                model_map = {"nano": RFDETRNano, "base": RFDETRBase, "large": RFDETRLarge}
                model_cls = model_map.get(self.rfdetr_model_size, RFDETRLarge)
                logger.info(f"Loading RF-DETR ({self.rfdetr_model_size}) for TrafficMonitor...")
                self._model = model_cls()
                self._model_available = True
                self._model_backend = "rfdetr"
                self._model_loaded = True
                logger.info(
                    f"✅ RF-DETR ({self.rfdetr_model_size}) loaded for TrafficMonitor — "
                    f"person+vehicle detection enabled"
                )
                return True
            except ImportError:
                logger.info("rfdetr not installed, falling back to YOLOv8")
            except Exception as e:
                logger.warning(f"RF-DETR load failed: {e}, falling back to YOLOv8")

            # ── Fallback to YOLOv8 ────────────────────────────────────
            try:
                import torch
                from ultralytics import YOLO
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Loading YOLOv8 for TrafficMonitor: {self.model_path} on {self._device}")
                self._model = YOLO(self.model_path)
                self._model_available = True
                self._model_backend = "yolo"
                logger.info(f"YOLOv8 model loaded for TrafficMonitor on {self._device}")
            except ImportError:
                self._model_available = False
                logger.warning("Neither roboflow, rfdetr, nor ultralytics installed — traffic monitoring disabled.")
            except Exception as e:
                self._model_available = False
                logger.error(f"Failed to load any detection model: {e}")
            finally:
                self._model_loaded = True
            return self._model_available

    def submit_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
    ) -> bool:
        """
        Submit a frame for traffic analysis (non-blocking).
        Only processes every `frame_interval`-th frame.

        Returns:
            True if frame was submitted, False if skipped
        """
        with self._counter_lock:
            self._frame_counter += 1
            if self._frame_counter % self.frame_interval != 0:
                return False

        frame_copy = frame.copy()
        self._executor.submit(self._analyze, frame_copy, camera_id)
        return True

    def _analyze(self, frame: np.ndarray, camera_id: str):
        """Run detection and proximity analysis."""
        if not self._load_model():
            return

        if self._model_backend == "roboflow":
            self._analyze_roboflow(frame, camera_id)
        elif self._model_backend == "rfdetr":
            self._analyze_rfdetr(frame, camera_id)
        else:
            self._analyze_yolo(frame, camera_id)

    def _analyze_roboflow(self, frame: np.ndarray, camera_id: str):
        """
        Run Roboflow detection with TWO models:
        1. Indian Vehicles Dataset — detects cars, buses, trucks, bikes, autos
        2. Person Detection — detects persons (class "Persona")
        
        Person detections are used to filter out false vehicle detections
        that overlap with detected persons (the vehicles model often labels
        people as cars/autos since it has no person class).
        """
        try:
            from app.config import settings as _s
            import base64
            start_time = time.monotonic()

            # Downscale frame for faster API upload (640px width max)
            frame_h, frame_w = frame.shape[:2]
            api_frame = frame
            if frame_w > 640:
                scale = 640 / frame_w
                api_frame = cv2.resize(frame, (640, int(frame_h * scale)), interpolation=cv2.INTER_AREA)

            # Encode frame as JPEG for API upload (shared by both models)
            _, buffer = cv2.imencode('.jpg', api_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_bytes = buffer.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

            # Scale factor to map API coords back to original frame
            api_h, api_w = api_frame.shape[:2]
            sx = frame_w / api_w
            sy = frame_h / api_h

            # ── Run Person + Vehicle detection models IN PARALLEL ─────────
            # This cuts latency from ~1.5s (sequential) to ~750ms (parallel)
            from concurrent.futures import as_completed

            person_preds = []
            vehicle_preds = []

            def _call_person_model():
                if not getattr(_s, 'ENABLE_ROBOFLOW_PERSON', False):
                    return []
                try:
                    res = self._roboflow_client.infer(
                        img_b64,
                        model_id=_s.ROBOFLOW_PERSON_MODEL_ID,
                    )
                    return res.get("predictions", []) if isinstance(res, dict) else []
                except Exception as e:
                    logger.warning(f"Roboflow person detection failed: {e}")
                    return []

            def _call_vehicle_model():
                try:
                    res = self._roboflow_client.infer(
                        img_b64,
                        model_id=_s.ROBOFLOW_MODEL_ID,
                    )
                    return res.get("predictions", []) if isinstance(res, dict) else []
                except Exception as e:
                    logger.warning(f"Roboflow vehicle detection failed: {e}")
                    return []

            person_future = self._rf_dual_executor.submit(_call_person_model)
            vehicle_future = self._rf_dual_executor.submit(_call_vehicle_model)
            person_preds = person_future.result(timeout=10)
            vehicle_preds = vehicle_future.result(timeout=10)

            inference_ms = (time.monotonic() - start_time) * 1000
            self._last_inference_ms = inference_ms
            self._total_inferences += 1

            # ── Parse person detections ───────────────────────────────────
            persons = []
            person_conf_threshold = getattr(_s, 'ROBOFLOW_PERSON_CONFIDENCE', 0.50)
            for pred in person_preds:
                confidence = pred.get("confidence", 0)
                if confidence < person_conf_threshold:
                    continue

                cx = pred.get("x", 0) * sx
                cy = pred.get("y", 0) * sy
                w = pred.get("width", 0) * sx
                h = pred.get("height", 0) * sy

                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = min(frame_w, int(cx + w / 2))
                y2 = min(frame_h, int(cy + h / 2))

                persons.append({
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": confidence,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

            logger.debug(
                f"Roboflow: {len(persons)} persons, {len(vehicle_preds)} raw vehicles "
                f"in {inference_ms:.0f}ms (parallel) on camera {camera_id}"
            )

            # ── Parse vehicle detections ──────────────────────────────────
            predictions = vehicle_preds

            vehicles = []
            confidence_threshold = _s.ROBOFLOW_CONFIDENCE_THRESHOLD

            for pred in predictions:
                cls_name = pred.get("class", "")
                confidence = pred.get("confidence", 0)

                if confidence < confidence_threshold:
                    continue

                # Map Roboflow class to our internal naming
                internal_name = ROBOFLOW_CLASS_MAP.get(cls_name)
                if internal_name is None:
                    continue  # Skip unknown classes

                # Roboflow returns center (x, y) + width/height — scale to original frame
                cx = pred.get("x", 0) * sx
                cy = pred.get("y", 0) * sy
                w = pred.get("width", 0) * sx
                h = pred.get("height", 0) * sy

                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = min(frame_w, int(cx + w / 2))
                y2 = min(frame_h, int(cy + h / 2))

                det = {
                    "class_id": list(ROBOFLOW_CLASS_MAP.values()).index(internal_name),
                    "class_name": internal_name,
                    "confidence": confidence,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "roboflow_class": cls_name,  # Keep original class for debug
                }

                # Detect vehicle color from the crop
                crop = frame[y1:y2, x1:x2]
                det["color"] = detect_vehicle_color(crop)
                vehicles.append(det)

            # ── 3. Filter out false vehicle detections that overlap with persons ──
            # The vehicles model often labels people as cars/autos/bikes
            if persons and vehicles:
                filtered_vehicles = []
                for v in vehicles:
                    vb = v["bbox"]
                    v_cx = (vb["x1"] + vb["x2"]) / 2
                    v_cy = (vb["y1"] + vb["y2"]) / 2

                    overlaps_person = False
                    for p in persons:
                        pb = p["bbox"]
                        # Check if vehicle center is inside (expanded) person bbox
                        pw = pb["x2"] - pb["x1"]
                        ph = pb["y2"] - pb["y1"]
                        # Expand person box by 30% for tolerance
                        ex1 = pb["x1"] - pw * 0.3
                        ey1 = pb["y1"] - ph * 0.3
                        ex2 = pb["x2"] + pw * 0.3
                        ey2 = pb["y2"] + ph * 0.3

                        if ex1 <= v_cx <= ex2 and ey1 <= v_cy <= ey2:
                            overlaps_person = True
                            logger.debug(
                                f"Filtered false vehicle '{v['class_name']}' "
                                f"(conf={v['confidence']:.0%}) — overlaps with person"
                            )
                            break

                        # Also check IoU overlap
                        ix1 = max(vb["x1"], pb["x1"])
                        iy1 = max(vb["y1"], pb["y1"])
                        ix2 = min(vb["x2"], pb["x2"])
                        iy2 = min(vb["y2"], pb["y2"])
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            v_area = (vb["x2"] - vb["x1"]) * (vb["y2"] - vb["y1"])
                            p_area = (pb["x2"] - pb["x1"]) * (pb["y2"] - pb["y1"])
                            iou = inter / (v_area + p_area - inter) if (v_area + p_area - inter) > 0 else 0
                            if iou > 0.3:
                                overlaps_person = True
                                logger.debug(
                                    f"Filtered false vehicle '{v['class_name']}' "
                                    f"(IoU={iou:.2f} with person)"
                                )
                                break

                    if not overlaps_person:
                        filtered_vehicles.append(v)

                vehicles = filtered_vehicles

            # ── YOLO Fallback: if Roboflow found 0 vehicles, try local YOLO model ──
            # The Roboflow Indian Vehicles Dataset model sometimes misses vehicles
            # at certain camera angles. The custom gate_vehicles YOLO model is more
            # reliable as a secondary detector.
            if not vehicles:
                yolo_vehicles = self._yolo_fallback_detect(frame, camera_id)
                if yolo_vehicles:
                    # Apply lighter person-overlap filter to YOLO results
                    # (less aggressive than Roboflow since YOLO is more specific)
                    if persons:
                        yolo_filtered = []
                        for v in yolo_vehicles:
                            vb = v["bbox"]
                            v_area = (vb["x2"] - vb["x1"]) * (vb["y2"] - vb["y1"])
                            v_cx = (vb["x1"] + vb["x2"]) / 2
                            v_cy = (vb["y1"] + vb["y2"]) / 2
                            overlaps = False
                            for p in persons:
                                pb = p["bbox"]
                                # Only filter if vehicle is SMALL (person-sized)
                                # Large vehicles (autos, cars) naturally overlap with riders
                                p_area = (pb["x2"] - pb["x1"]) * (pb["y2"] - pb["y1"])
                                if v_area > p_area * 1.5:
                                    continue  # Vehicle is much larger than person — keep it
                                # Use IoU for more precise overlap check
                                ix1 = max(vb["x1"], pb["x1"])
                                iy1 = max(vb["y1"], pb["y1"])
                                ix2 = min(vb["x2"], pb["x2"])
                                iy2 = min(vb["y2"], pb["y2"])
                                if ix2 > ix1 and iy2 > iy1:
                                    inter = (ix2 - ix1) * (iy2 - iy1)
                                    iou = inter / (v_area + p_area - inter) if (v_area + p_area - inter) > 0 else 0
                                    if iou > 0.5:  # Very high overlap = likely same object
                                        overlaps = True
                                        break
                            if not overlaps:
                                yolo_filtered.append(v)
                        vehicles = yolo_filtered
                    else:
                        vehicles = yolo_vehicles
                    if vehicles:
                        logger.debug(
                            f"YOLO fallback: found {len(vehicles)} vehicles "
                            f"(Roboflow had 0) on camera {camera_id}"
                        )

            # Motion filtering
            moving_vehicles = self._filter_moving_vehicles(vehicles)
            self._last_person_count = len(persons)
            self._last_vehicle_count = len(moving_vehicles)

            # Check proximity alerts between persons and vehicles
            alerts = []
            if persons and moving_vehicles:
                alerts = self._check_proximity(persons, moving_vehicles, camera_id)

            with self._detections_lock:
                self._last_detections = {
                    "persons": persons,
                    "vehicles": moving_vehicles,
                    "alerts": alerts,
                }

            # Emit vehicle count update
            if self.on_vehicle_count and moving_vehicles:
                count_data = {
                    "camera_id": camera_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_vehicles": len(moving_vehicles),
                    "total_persons": len(persons),
                    "vehicle_breakdown": {},
                    "vehicle_details": [],
                }
                for v in moving_vehicles:
                    vtype = v["class_name"]
                    count_data["vehicle_breakdown"][vtype] = count_data["vehicle_breakdown"].get(vtype, 0) + 1
                    count_data["vehicle_details"].append({
                        "type": vtype,
                        "color": v.get("color", "Unknown"),
                        "confidence": round(v["confidence"], 3),
                        "bbox": v["bbox"],
                        "roboflow_class": v.get("roboflow_class", ""),
                    })
                self.on_vehicle_count(count_data)

            logger.debug(
                f"Roboflow: {len(moving_vehicles)} vehicles, {len(persons)} persons "
                f"(raw={len(predictions)}) in {inference_ms:.0f}ms on camera {camera_id}"
            )

        except Exception as e:
            logger.error(f"Roboflow traffic analysis failed: {e}", exc_info=True)

    def _analyze_rfdetr(self, frame: np.ndarray, camera_id: str):
        """Run RF-DETR detection — detects both persons and vehicles."""
        try:
            from PIL import Image
            start_time = time.monotonic()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            detections = self._model.predict(pil_image, threshold=0.45)

            inference_ms = (time.monotonic() - start_time) * 1000
            self._last_inference_ms = inference_ms
            self._total_inferences += 1

            if detections is None or len(detections) == 0:
                return

            persons = []
            vehicles = []

            for i in range(len(detections)):
                cls_id = int(detections.class_id[i])
                confidence = float(detections.confidence[i])
                x1, y1, x2, y2 = [int(c) for c in detections.xyxy[i].tolist()]

                if cls_id == COCO_PERSON_CLASS and confidence >= 0.45:
                    persons.append({
                        "class_id": cls_id,
                        "class_name": "person",
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
                elif cls_id in COCO_VEHICLE_CLASSES and confidence >= 0.55:
                    det = {
                        "class_id": cls_id,
                        "class_name": COCO_VEHICLE_CLASSES[cls_id],
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                    # Detect vehicle color
                    h_frame, w_frame = frame.shape[:2]
                    crop = frame[max(0,y1):min(h_frame,y2), max(0,x1):min(w_frame,x2)]
                    det["color"] = detect_vehicle_color(crop)
                    vehicles.append(det)

            # ── Filter out false vehicle detections near faces ──────────────
            # Problem: RF-DETR detects person's shirt/torso as "bike 98%".
            # Multiple checks:
            # 1. If any face/person center is within 80% expanded vehicle box → false
            # 2. Bikes/motorcycles with portrait aspect ratio → likely a person
            # 3. "Vehicle" covering >15% of frame → likely a false positive
            face_zones = getattr(self, '_known_face_zones', [])
            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_h * frame_w

            if vehicles:
                filtered_vehicles = []
                for v in vehicles:
                    vb = v["bbox"]
                    vw = vb["x2"] - vb["x1"]
                    vh = vb["y2"] - vb["y1"]
                    v_area = vw * vh

                    # Check 1: Aspect ratio for bikes — real bikes are wider than tall
                    if v["class_name"] == "bike" and vh > vw * 1.2:
                        continue  # Portrait-oriented "bike" = person's body

                    # Check 2: Frame area — vehicle covering >15% of frame is suspicious
                    if v_area / frame_area > 0.15:
                        continue  # Too large relative to frame

                    # Check 3: Face/person proximity check
                    # Expand vehicle box by 80% in all directions
                    ex1 = vb["x1"] - vw * 0.8
                    ey1 = vb["y1"] - vh * 0.8
                    ex2 = vb["x2"] + vw * 0.8
                    ey2 = vb["y2"] + vh * 0.8

                    has_face_nearby = False
                    # Check against RF-DETR person detections
                    for p in persons:
                        cx = (p["bbox"]["x1"] + p["bbox"]["x2"]) / 2
                        cy = (p["bbox"]["y1"] + p["bbox"]["y2"]) / 2
                        if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                            has_face_nearby = True
                            break
                    # Check against InsightFace detections
                    if not has_face_nearby:
                        for fz in face_zones:
                            cx = (fz["x1"] + fz["x2"]) / 2
                            cy = (fz["y1"] + fz["y2"]) / 2
                            if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                                has_face_nearby = True
                                break

                    if not has_face_nearby:
                        filtered_vehicles.append(v)
                vehicles = filtered_vehicles

            # Motion filtering + proximity (reuse existing logic)
            moving_vehicles = self._filter_moving_vehicles(vehicles)
            self._last_person_count = len(persons)
            self._last_vehicle_count = len(moving_vehicles)

            alerts = []
            if persons and moving_vehicles:
                alerts = self._check_proximity(persons, moving_vehicles, camera_id)

            with self._detections_lock:
                self._last_detections = {
                    "persons": persons,
                    "vehicles": moving_vehicles,
                    "alerts": alerts,
                }

            if self.on_vehicle_count and moving_vehicles:
                count_data = {
                    "camera_id": camera_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_vehicles": len(moving_vehicles),
                    "total_persons": len(persons),
                    "vehicle_breakdown": {},
                    "vehicle_details": [],
                }
                for v in moving_vehicles:
                    vtype = v["class_name"]
                    count_data["vehicle_breakdown"][vtype] = count_data["vehicle_breakdown"].get(vtype, 0) + 1
                    count_data["vehicle_details"].append({
                        "type": vtype,
                        "color": v.get("color", "Unknown"),
                        "confidence": round(v["confidence"], 3),
                        "bbox": v["bbox"],
                    })
                self.on_vehicle_count(count_data)

        except Exception as e:
            logger.error(f"RF-DETR traffic analysis failed: {e}", exc_info=True)

    def _analyze_yolo(self, frame: np.ndarray, camera_id: str):
        """Run YOLO detection and proximity analysis (fallback)."""
        if not self._load_model():
            return

        try:
            start_time = time.monotonic()

            # Detect persons + vehicles — use low base conf, per-class filter below
            results = self._model(
                frame,
                conf=0.25,
                classes=ALL_TRAFFIC_CLASSES,
                verbose=False,
                device=getattr(self, '_device', 'cpu'),
            )

            inference_ms = (time.monotonic() - start_time) * 1000
            self._last_inference_ms = inference_ms
            self._total_inferences += 1

            if not results:
                return

            # Separate persons and vehicles with per-class confidence filtering
            persons = []
            vehicles = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Per-class confidence threshold — trucks need higher conf
                    min_conf = CLASS_CONFIDENCE.get(cls_id, 0.35)
                    if confidence < min_conf:
                        continue

                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]

                    det = {
                        "class_id": cls_id,
                        "class_name": "person" if cls_id == PERSON_CLASS
                            else VEHICLE_CLASSES.get(cls_id, "vehicle"),
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }

                    # Detect vehicle color from the crop
                    if cls_id in VEHICLE_CLASSES:
                        h_frame, w_frame = frame.shape[:2]
                        crop = frame[
                            max(0, y1):min(h_frame, y2),
                            max(0, x1):min(w_frame, x2),
                        ]
                        det["color"] = detect_vehicle_color(crop)

                    if cls_id == PERSON_CLASS:
                        persons.append(det)
                    elif cls_id in VEHICLE_CLASSES:
                        vehicles.append(det)

            # ── Motion filtering: only keep vehicles that are moving ──────
            moving_vehicles = self._filter_moving_vehicles(vehicles)

            self._last_person_count = len(persons)
            self._last_vehicle_count = len(moving_vehicles)

            # Update cached detections
            alerts = []

            # Check proximity between each person and each vehicle
            if persons and moving_vehicles:
                alerts = self._check_proximity(persons, moving_vehicles, camera_id)

            with self._detections_lock:
                self._last_detections = {
                    "persons": persons,
                    "vehicles": moving_vehicles,
                    "alerts": alerts,
                }

            # Emit vehicle count update with per-vehicle details (moving only)
            if self.on_vehicle_count and moving_vehicles:
                count_data = {
                    "camera_id": camera_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_vehicles": len(moving_vehicles),
                    "total_persons": len(persons),
                    "vehicle_breakdown": {},
                    "vehicle_details": [],
                }
                for v in moving_vehicles:
                    vtype = v["class_name"]
                    count_data["vehicle_breakdown"][vtype] = (
                        count_data["vehicle_breakdown"].get(vtype, 0) + 1
                    )
                    count_data["vehicle_details"].append({
                        "type": vtype,
                        "color": v.get("color", "Unknown"),
                        "confidence": round(v["confidence"], 3),
                        "bbox": v["bbox"],
                    })
                self.on_vehicle_count(count_data)

        except Exception as e:
            logger.error(f"Traffic analysis failed: {e}", exc_info=True)

    def _yolo_fallback_detect(self, frame: np.ndarray, camera_id: str) -> List[dict]:
        """
        Fallback: run local YOLO model when Roboflow returns 0 vehicles.
        Uses the custom gate_vehicles model (trained on this camera's environment).
        Lazy-loads the YOLO model only when first needed.
        """
        try:
            # Lazy-load the YOLO fallback model
            if not hasattr(self, '_yolo_fallback_model'):
                self._yolo_fallback_model = None
                self._yolo_fallback_device = 'cpu'
                try:
                    import torch
                    from ultralytics import YOLO
                    self._yolo_fallback_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model_path = self.model_path  # gate_vehicles_v2.pt
                    self._yolo_fallback_model = YOLO(model_path)
                    logger.info(
                        f"Loaded YOLO fallback model ({model_path}) "
                        f"on {self._yolo_fallback_device} for Roboflow zero-vehicle recovery"
                    )
                except Exception as e:
                    logger.warning(f"Could not load YOLO fallback model: {e}")

            if self._yolo_fallback_model is None:
                return []

            results = self._yolo_fallback_model(
                frame,
                conf=0.30,
                classes=ALL_TRAFFIC_CLASSES,
                verbose=False,
                device=self._yolo_fallback_device,
            )

            vehicles = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    min_conf = CLASS_CONFIDENCE.get(cls_id, 0.35)
                    if confidence < min_conf:
                        continue

                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                    det = {
                        "class_id": cls_id,
                        "class_name": VEHICLE_CLASSES.get(cls_id, "vehicle"),
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                    # Detect vehicle color
                    h_frame, w_frame = frame.shape[:2]
                    crop = frame[
                        max(0, y1):min(h_frame, y2),
                        max(0, x1):min(w_frame, x2),
                    ]
                    det["color"] = detect_vehicle_color(crop)
                    vehicles.append(det)

            if vehicles:
                logger.debug(
                    f"YOLO fallback detected {len(vehicles)} vehicles on camera {camera_id}"
                )
            return vehicles

        except Exception as e:
            logger.debug(f"YOLO fallback detection error: {e}")
            return []


    def _filter_moving_vehicles(self, vehicles: List[dict]) -> List[dict]:
        """
        Filter out stationary vehicles by comparing positions across frames.
        
        A vehicle is considered "moving" if its centroid has shifted by 
        more than MOTION_THRESHOLD_PX pixels from its previous known position.
        New vehicles (first appearance) are let through with a grace period.
        """
        moving = []
        current_positions: Dict[str, dict] = {}

        for v in vehicles:
            bbox = v["bbox"]
            cx = (bbox["x1"] + bbox["x2"]) // 2
            cy = (bbox["y1"] + bbox["y2"]) // 2
            cls_name = v["class_name"]

            # Create a spatial key for matching to previous frame
            best_key = None
            best_dist = float("inf")
            for key, prev in self._prev_positions.items():
                if prev["class"] != cls_name:
                    continue
                dx = abs(cx - prev["cx"])
                dy = abs(cy - prev["cy"])
                dist = (dx**2 + dy**2) ** 0.5
                if dist < 300 and dist < best_dist:  # Match within 300px (wider for fast vehicles)
                    best_dist = dist
                    best_key = key

            if best_key:
                prev = self._prev_positions[best_key]
                prev["frames_seen"] = prev.get("frames_seen", 0) + 1

                # Check if vehicle has moved
                if best_dist > MOTION_THRESHOLD_PX:
                    prev["frames_moved"] = prev.get("frames_moved", 0) + 1
                    prev["cx"] = cx
                    prev["cy"] = cy
                    moving.append(v)
                elif prev.get("frames_moved", 0) > 0:
                    # Was moving before, now stopped — still include (might be temporary stop)
                    moving.append(v)
                # else: stationary object — skip it

                current_positions[best_key] = prev
            else:
                # New vehicle — first sighting, let it through (benefit of doubt)
                new_key = f"{cls_name}_{cx}_{cy}_{id(v)}"
                current_positions[new_key] = {
                    "cx": cx, "cy": cy, "class": cls_name,
                    "frames_seen": 1, "frames_moved": 1,
                }
                moving.append(v)

        # Update positions for next frame comparison
        self._prev_positions = current_positions
        return moving

    def _check_proximity(
        self,
        persons: List[dict],
        vehicles: List[dict],
        camera_id: str,
    ) -> List[dict]:
        """
        Check if any person is dangerously close to any vehicle.

        Uses bounding box IoU (overlap) and edge-to-edge distance.
        Triggers alert if overlap > 0 or distance < proximity_px.

        Returns list of alert dicts.
        """
        now = time.time()
        alerts = []

        for person in persons:
            pb = person["bbox"]
            for vehicle in vehicles:
                vb = vehicle["bbox"]

                # Calculate IoU (overlap)
                overlap = self._bbox_iou(pb, vb)

                # Calculate min edge distance
                distance = self._bbox_min_distance(pb, vb)

                # Trigger if overlapping OR within proximity threshold
                is_danger = overlap > 0 or distance < self.proximity_px

                if not is_danger:
                    continue

                # Cooldown check
                with self._cooldown_lock:
                    last_alert = self._alert_cooldowns.get(camera_id, 0)
                    if now - last_alert < self.alert_cooldown_sec:
                        continue
                    self._alert_cooldowns[camera_id] = now

                self._total_alerts += 1

                alert = {
                    "event_type": "security_alert",
                    "subtype": "vehicle_proximity",
                    "camera_id": camera_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "confidence": round(person["confidence"], 3),
                    "metadata": {
                        "subtype": "vehicle_proximity",
                        "vehicle_type": vehicle["class_name"],
                        "vehicle_confidence": round(vehicle["confidence"], 3),
                        "person_confidence": round(person["confidence"], 3),
                        "overlap_iou": round(overlap, 4),
                        "min_distance_px": round(distance, 1),
                        "person_bbox": pb,
                        "vehicle_bbox": vb,
                        "severity": "high" if overlap > 0 else "medium",
                    },
                }

                severity = "CRITICAL (OVERLAP)" if overlap > 0 else f"WARNING ({distance:.0f}px)"
                logger.warning(
                    f"🚨 VEHICLE PROXIMITY ALERT: Person too close to "
                    f"{vehicle['class_name']} — {severity} on camera {camera_id}"
                )

                if self.on_alert:
                    self.on_alert(alert)

                alerts.append(alert)

        return alerts

    @staticmethod
    def _bbox_iou(a: dict, b: dict) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(a["x1"], b["x1"])
        y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"])
        y2 = min(a["y2"], b["y2"])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _bbox_min_distance(a: dict, b: dict) -> float:
        """
        Calculate minimum edge-to-edge distance between two bounding boxes.
        Returns 0 if they overlap.
        """
        # Horizontal gap
        dx = max(0, max(a["x1"] - b["x2"], b["x1"] - a["x2"]))
        # Vertical gap
        dy = max(0, max(a["y1"] - b["y2"], b["y1"] - a["y2"]))

        return (dx ** 2 + dy ** 2) ** 0.5

    def draw_detections(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Draw traffic detections and alerts on a frame.

        Color coding:
        - Blue: vehicles
        - Green: persons (safe)
        - Red: persons in danger zone
        - Red line: connecting person-vehicle in alert
        """
        with self._detections_lock:
            persons = self._last_detections.get("persons", [])
            vehicles = self._last_detections.get("vehicles", [])
            alerts = self._last_detections.get("alerts", [])

        # Collect person bboxes that are in danger
        danger_bboxes = set()
        for alert in alerts:
            pb = alert.get("metadata", {}).get("person_bbox", {})
            danger_bboxes.add((pb.get("x1"), pb.get("y1"), pb.get("x2"), pb.get("y2")))

        # Draw vehicles (blue)
        for v in vehicles:
            b = v["bbox"]
            label = f"{v['class_name']} {v['confidence']:.0%}"
            cv2.rectangle(
                frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                (255, 180, 0), 2,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(
                frame,
                (b["x1"], b["y1"] - th - 8),
                (b["x1"] + tw + 6, b["y1"]),
                (255, 180, 0), cv2.FILLED,
            )
            cv2.putText(
                frame, label,
                (b["x1"] + 3, b["y1"] - 4),
                font, 0.5, (0, 0, 0), 1,
            )

        # Draw persons
        for p in persons:
            b = p["bbox"]
            bbox_tuple = (b["x1"], b["y1"], b["x2"], b["y2"])
            is_danger = bbox_tuple in danger_bboxes
            color = (0, 0, 255) if is_danger else (0, 255, 0)
            label = "⚠ DANGER" if is_danger else f"person {p['confidence']:.0%}"

            cv2.rectangle(
                frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                color, 2 if not is_danger else 3,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(
                frame,
                (b["x1"], b["y1"] - th - 8),
                (b["x1"] + tw + 6, b["y1"]),
                color, cv2.FILLED,
            )
            cv2.putText(
                frame, label,
                (b["x1"] + 3, b["y1"] - 4),
                font, 0.5, (255, 255, 255), 1,
            )

        # Draw alert connections
        for alert in alerts:
            meta = alert.get("metadata", {})
            pb = meta.get("person_bbox", {})
            vb = meta.get("vehicle_bbox", {})

            pcx = (pb.get("x1", 0) + pb.get("x2", 0)) // 2
            pcy = (pb.get("y1", 0) + pb.get("y2", 0)) // 2
            vcx = (vb.get("x1", 0) + vb.get("x2", 0)) // 2
            vcy = (vb.get("y1", 0) + vb.get("y2", 0)) // 2

            # Red dashed-style line between person and vehicle centroids
            cv2.line(frame, (pcx, pcy), (vcx, vcy), (0, 0, 255), 2)

            # Warning label at midpoint
            mx, my = (pcx + vcx) // 2, (pcy + vcy) // 2
            dist = meta.get("min_distance_px", 0)
            warn_text = f"DANGER {dist:.0f}px"
            cv2.putText(
                frame, warn_text,
                (mx - 40, my - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        # Traffic counter overlay
        counter_text = f"Vehicles: {len(vehicles)} | Persons: {len(persons)}"
        cv2.putText(
            frame, counter_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        return frame

    @property
    def stats(self) -> dict:
        return {
            "model_loaded": self._model_available,
            "model_backend": self._model_backend or "none",
            "total_inferences": self._total_inferences,
            "total_proximity_alerts": self._total_alerts,
            "last_vehicle_count": self._last_vehicle_count,
            "last_person_count": self._last_person_count,
            "last_inference_ms": round(self._last_inference_ms, 1),
        }

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
        logger.info("TrafficMonitor shut down")
