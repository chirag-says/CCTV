"""
Traffic Monitor — Human-Vehicle Safety & Traffic Analytics.

Detects proximity between persons and vehicles using YOLOv8 to
prevent accidents. Also provides vehicle counting and traffic flow data.

COCO class IDs:
  - 0: person
  - 2: car
  - 3: motorcycle
  - 5: bus
  - 7: truck

This module gracefully handles the case where ultralytics isn't installed.
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

# Detection classes
PERSON_CLASS = 0
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

ALL_TRAFFIC_CLASSES = [PERSON_CLASS] + list(VEHICLE_CLASSES.keys())


class TrafficMonitor:
    """
    Human-Vehicle safety monitor using YOLOv8.

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
    ):
        """
        Args:
            model_path: Path to YOLO model weights
            frame_interval: Only process every Nth frame
            proximity_px: Pixel threshold for "too close" alert
            alert_cooldown_sec: Min seconds between alerts
            on_alert: Callback(alert_dict) when proximity danger detected
            on_vehicle_count: Callback(count_dict) for traffic counting
            max_workers: Thread pool size
        """
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.proximity_px = proximity_px
        self.alert_cooldown_sec = alert_cooldown_sec
        self.on_alert = on_alert
        self.on_vehicle_count = on_vehicle_count

        self._model = None
        self._model_loaded = False
        self._model_available = False
        self._model_lock = threading.Lock()

        self._frame_counter = 0
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="traffic-mon",
        )

        # Alert cooldown: camera_id → last_alert_timestamp
        self._alert_cooldowns: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        # Statistics
        self._total_inferences = 0
        self._total_alerts = 0
        self._last_inference_ms = 0.0
        self._last_vehicle_count = 0
        self._last_person_count = 0

        # Latest detection results (for drawing)
        self._last_detections: Dict[str, List[dict]] = {
            "persons": [],
            "vehicles": [],
            "alerts": [],
        }
        self._detections_lock = threading.Lock()

        logger.info(
            f"TrafficMonitor initialized: model={model_path}, "
            f"interval={frame_interval}, proximity={proximity_px}px, "
            f"cooldown={alert_cooldown_sec}s"
        )

    def _load_model(self) -> bool:
        """Lazy-load YOLOv8 model. Thread-safe."""
        with self._model_lock:
            if self._model_loaded:
                return self._model_available
            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLOv8 for TrafficMonitor: {self.model_path}")
                self._model = YOLO(self.model_path)
                self._model_available = True
                logger.info("YOLOv8 model loaded for TrafficMonitor")
            except ImportError:
                self._model_available = False
                logger.warning(
                    "ultralytics not installed — traffic monitoring disabled. "
                    "Install with: pip install ultralytics"
                )
            except Exception as e:
                self._model_available = False
                logger.error(f"Failed to load YOLOv8 for TrafficMonitor: {e}")
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
        self._frame_counter += 1
        if self._frame_counter % self.frame_interval != 0:
            return False

        frame_copy = frame.copy()
        self._executor.submit(self._analyze, frame_copy, camera_id)
        return True

    def _analyze(self, frame: np.ndarray, camera_id: str):
        """Run YOLO detection and proximity analysis."""
        if not self._load_model():
            return

        try:
            start_time = time.monotonic()

            # Detect persons + vehicles
            results = self._model(
                frame,
                conf=0.35,
                classes=ALL_TRAFFIC_CLASSES,
                verbose=False,
            )

            inference_ms = (time.monotonic() - start_time) * 1000
            self._last_inference_ms = inference_ms
            self._total_inferences += 1

            if not results:
                return

            # Separate persons and vehicles
            persons = []
            vehicles = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]

                    det = {
                        "class_id": cls_id,
                        "class_name": "person" if cls_id == PERSON_CLASS
                            else VEHICLE_CLASSES.get(cls_id, "vehicle"),
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }

                    if cls_id == PERSON_CLASS:
                        persons.append(det)
                    elif cls_id in VEHICLE_CLASSES:
                        vehicles.append(det)

            self._last_person_count = len(persons)
            self._last_vehicle_count = len(vehicles)

            # Update cached detections
            alerts = []

            # Check proximity between each person and each vehicle
            if persons and vehicles:
                alerts = self._check_proximity(persons, vehicles, camera_id)

            with self._detections_lock:
                self._last_detections = {
                    "persons": persons,
                    "vehicles": vehicles,
                    "alerts": alerts,
                }

            # Emit vehicle count update
            if self.on_vehicle_count and vehicles:
                count_data = {
                    "camera_id": camera_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_vehicles": len(vehicles),
                    "total_persons": len(persons),
                    "vehicle_breakdown": {},
                }
                for v in vehicles:
                    vtype = v["class_name"]
                    count_data["vehicle_breakdown"][vtype] = (
                        count_data["vehicle_breakdown"].get(vtype, 0) + 1
                    )
                self.on_vehicle_count(count_data)

        except Exception as e:
            logger.error(f"Traffic analysis failed: {e}", exc_info=True)

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
