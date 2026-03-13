"""
Video Analysis Service — Process uploaded videos through the AI vision pipeline.

Reuses the same detection/recognition modules used for live camera feeds:
- Face Detection & Recognition
- Person Tracking (entry/exit)
- Hazard Detection (YOLOv8)
- ANPR (License Plate Recognition)
- Traffic Monitor (Vehicle-Pedestrian Safety)
- Crowd / Gathering Detection
- Loitering Detection
- Attribute Recognition (gender + clothing)

Each analysis job runs in a dedicated background thread.
"""

import cv2
import logging
import os
import threading
import time
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from uuid import uuid4

from app.config import settings
from app.utils.image_utils import save_snapshot

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Video Analysis Result Model
# ═══════════════════════════════════════════════════════════════════════════════

class VideoAnalysisResult:
    """Stores all results from a video analysis job."""

    def __init__(self, job_id: str, filename: str):
        self.job_id = job_id
        self.filename = filename
        self.status = "pending"  # pending → processing → completed / failed
        self.progress = 0.0     # 0.0 to 100.0
        self.error = None

        # Video metadata
        self.video_metadata = {
            "filename": filename,
            "fps": 0,
            "total_frames": 0,
            "duration_seconds": 0,
            "resolution": {"width": 0, "height": 0},
        }

        # Detection results
        self.detected_persons: List[dict] = []      # Known persons with appearances
        self.unknown_persons: List[dict] = []        # Unknown face snapshots
        self.detected_vehicles: List[dict] = []      # ANPR plate detections
        self.security_alerts: List[dict] = []        # Hazard, crowd, loitering
        self.traffic_alerts: List[dict] = []         # Vehicle-pedestrian proximity
        self.events_timeline: List[dict] = []        # Chronological event log
        self.key_frames: List[dict] = []             # Annotated frame snapshots

        # Summary stats (populated on completion)
        self.summary = {}

        # Timestamps
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "progress": round(self.progress, 1),
            "error": self.error,
            "video_metadata": self.video_metadata,
            "detected_persons": self.detected_persons,
            "unknown_persons": self.unknown_persons,
            "detected_vehicles": self.detected_vehicles,
            "security_alerts": self.security_alerts,
            "traffic_alerts": self.traffic_alerts,
            "events_timeline": self.events_timeline,
            "key_frames": self.key_frames,
            "summary": self.summary,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    def to_status_dict(self) -> dict:
        """Lightweight status response (no full results)."""
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "progress": round(self.progress, 1),
            "error": self.error,
            "video_metadata": self.video_metadata,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Video Analysis Job
# ═══════════════════════════════════════════════════════════════════════════════

class VideoAnalysisJob:
    """
    Processes a single video file through the full AI pipeline.

    Runs in a background thread, collecting all detection events
    into a VideoAnalysisResult.
    """

    def __init__(
        self,
        job_id: str,
        video_path: str,
        filename: str,
        on_progress: Optional[Callable] = None,
    ):
        self.job_id = job_id
        self.video_path = video_path
        self.result = VideoAnalysisResult(job_id, filename)
        self.on_progress = on_progress

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track unique persons and vehicles to avoid duplicates in results
        self._person_tracker: Dict[str, dict] = {}  # person_id → info
        self._unknown_tracker: Dict[str, dict] = {}  # encoding hash → info
        self._unknown_encodings: list = []            # List of (id, encoding) for dedup
        self._plate_tracker: Dict[str, dict] = {}    # plate → info (final, consolidated)
        self._raw_plate_readings: List[dict] = []    # All raw OCR readings for consolidation
        self._current_frame: int = 0                  # Updated in main loop for ANPR callback

    def start(self):
        """Start analysis in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_analysis,
            name=f"video-analysis-{self.job_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Video analysis started: {self.job_id}")

    def stop(self):
        """Stop the analysis."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def _run_analysis(self):
        """Main analysis loop — processes video frame by frame."""
        import numpy as np

        self.result.status = "processing"
        self._emit_progress(0)

        try:
            # ── Open Video ────────────────────────────────────────
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {self.video_path}")

            # Extract video metadata
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            self.result.video_metadata = {
                "filename": self.result.filename,
                "fps": round(fps, 1),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 1),
                "resolution": {"width": width, "height": height},
            }

            logger.info(
                f"Video analysis [{self.job_id[:8]}]: "
                f"{width}x{height} @ {fps:.1f}fps, "
                f"{total_frames} frames ({duration:.1f}s)"
            )

            # ── Initialize Vision Modules ─────────────────────────
            from app.vision.detector import FaceDetector
            from app.vision.recognizer import FaceRecognizer
            from app.vision.tracker import PersonTracker

            detector = FaceDetector()
            recognizer = FaceRecognizer()
            tracker = PersonTracker(
                on_entry=lambda e: self._handle_event("entry", e),
                on_exit=lambda e: self._handle_event("exit", e),
            )

            # Load face encodings
            try:
                from app.services.person_service import PersonService
                encodings = PersonService.get_all_encodings()
                if encodings:
                    recognizer.load_encodings(encodings)
                    logger.info(f"Loaded {len(encodings)} face encodings for video analysis")
            except Exception as e:
                logger.warning(f"Could not load face encodings: {e}")

            # Optional modules — initialize if available
            hazard_detector = None
            plate_recognizer = None
            traffic_monitor = None
            crowd_detector = None
            loitering_detector = None
            attribute_recognizer = None

            try:
                if settings.ENABLE_HAZARD_DETECTION:
                    from app.vision.hazard_detector import HazardDetector
                    hazard_detector = HazardDetector(
                        model_path=settings.HAZARD_MODEL_PATH,
                        frame_interval=settings.HAZARD_FRAME_INTERVAL,
                        alert_cooldown_sec=settings.HAZARD_ALERT_COOLDOWN_SEC,
                        on_alert=lambda a: self._handle_alert("hazard", a),
                    )
            except Exception as e:
                logger.warning(f"Hazard detector not available: {e}")

            try:
                if settings.ENABLE_ANPR:
                    from app.vision.anpr import PlateRecognizer
                    languages = [l.strip() for l in settings.ANPR_OCR_LANGUAGES.split(",")]
                    plate_recognizer = PlateRecognizer(
                        yolo_model_path=settings.ANPR_MODEL_PATH,
                        frame_interval=1,  # Don't skip — video analysis already does frame_skip
                        plate_cooldown_sec=0,  # No cooldown — consolidation handles deduplication
                        on_plate=lambda e: self._handle_vehicle_event(e),
                        languages=languages,
                    )
                    logger.info("ANPR initialized for video analysis (frame_interval=1, no cooldown)")
            except Exception as e:
                logger.warning(f"ANPR not available: {e}")

            try:
                if settings.ENABLE_TRAFFIC_MONITOR:
                    from app.vision.traffic import TrafficMonitor
                    traffic_monitor = TrafficMonitor(
                        model_path=settings.TRAFFIC_MODEL_PATH,
                        frame_interval=settings.TRAFFIC_FRAME_INTERVAL,
                        proximity_px=settings.TRAFFIC_PROXIMITY_PX,
                        alert_cooldown_sec=settings.TRAFFIC_ALERT_COOLDOWN_SEC,
                        on_alert=lambda a: self._handle_alert("traffic", a),
                    )
            except Exception as e:
                logger.warning(f"Traffic monitor not available: {e}")

            try:
                if settings.ENABLE_CROWD_DETECTION:
                    from app.vision.safety_analytics import CrowdDetector
                    crowd_detector = CrowdDetector(
                        proximity_px=settings.CROWD_PROXIMITY_PX,
                        min_persons=settings.CROWD_MIN_PERSONS,
                        sustain_seconds=settings.CROWD_SUSTAIN_SECONDS,
                        on_alert=lambda a: self._handle_alert("crowd", a),
                    )
            except Exception as e:
                logger.warning(f"Crowd detector not available: {e}")

            try:
                if settings.ENABLE_LOITERING_DETECTION:
                    from app.vision.safety_analytics import LoiteringDetector
                    loitering_detector = LoiteringDetector(
                        movement_threshold_px=settings.LOITER_MOVEMENT_THRESHOLD_PX,
                        time_window_sec=settings.LOITER_TIME_WINDOW_SEC,
                        alert_cooldown_sec=settings.LOITER_ALERT_COOLDOWN_SEC,
                        on_alert=lambda a: self._handle_alert("loitering", a),
                    )
            except Exception as e:
                logger.warning(f"Loitering detector not available: {e}")

            try:
                if settings.ENABLE_ATTRIBUTE_RECOGNITION:
                    from app.vision.safety_analytics import AttributeRecognizer
                    attribute_recognizer = AttributeRecognizer(max_workers=2)
            except Exception as e:
                logger.warning(f"Attribute recognizer not available: {e}")

            # ── Process Frames ────────────────────────────────────
            frame_skip = settings.VIDEO_ANALYSIS_FRAME_SKIP
            frame_count = 0
            processed_count = 0
            camera_id = f"video-{self.job_id[:8]}"

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                frame_count += 1
                self._current_frame = frame_count

                # Update progress
                progress = (frame_count / max(total_frames, 1)) * 100
                self.result.progress = min(progress, 99.9)
                if frame_count % 30 == 0:  # Emit every 30 frames
                    self._emit_progress(self.result.progress)

                # Frame skip for performance
                if frame_count % frame_skip != 0:
                    continue

                processed_count += 1
                clean_frame = frame.copy()
                video_timestamp = frame_count / fps if fps > 0 else 0

                # ── Hazard Detection ───────────────────────────
                if hazard_detector:
                    try:
                        hazard_detector.submit_frame(clean_frame, camera_id)
                    except Exception:
                        pass

                # ── ANPR ───────────────────────────────────────
                if plate_recognizer:
                    try:
                        submitted = plate_recognizer.submit_frame(clean_frame, camera_id)
                        if submitted and frame_count % 100 == 0:
                            logger.debug(f"ANPR frame submitted at frame {frame_count}")
                    except Exception as e:
                        logger.warning(f"ANPR submit error at frame {frame_count}: {e}")

                # ── Traffic Monitor ────────────────────────────
                if traffic_monitor:
                    try:
                        traffic_monitor.submit_frame(clean_frame, camera_id)
                    except Exception:
                        pass

                # ── Face Detection ─────────────────────────────
                try:
                    face_locations, rgb_frame = detector.detect_faces(frame)

                    if face_locations:
                        encodings = recognizer.batch_generate_encodings(
                            rgb_frame, face_locations
                        )

                        for location, encoding in zip(face_locations, encodings):
                            if encoding is None:
                                continue

                            top, right, bottom, left = location
                            centroid = ((left + right) // 2, (top + bottom) // 2)

                            result = recognizer.recognize_face(encoding)

                            if result.matched:
                                # ── Known Person ───────────────────
                                tracker.on_detection(
                                    person_id=result.person_id,
                                    person_name=result.person_name,
                                    camera_id=camera_id,
                                    confidence=result.confidence,
                                    centroid=centroid,
                                    timestamp=video_timestamp,
                                )

                                # Track unique persons
                                if result.person_id not in self._person_tracker:
                                    # Save face crop as key frame
                                    face_crop = detector.extract_face_crop(
                                        clean_frame, location, padding=40
                                    )
                                    _, snap_url = save_snapshot(
                                        face_crop,
                                        f"va_{self.job_id[:8]}_{result.person_id[:8]}",
                                        subdirectory="video_analysis",
                                        quality=90,
                                    )

                                    self._person_tracker[result.person_id] = {
                                        "person_id": result.person_id,
                                        "person_name": result.person_name,
                                        "first_seen_frame": frame_count,
                                        "first_seen_time": round(video_timestamp, 1),
                                        "last_seen_frame": frame_count,
                                        "last_seen_time": round(video_timestamp, 1),
                                        "appearances": 1,
                                        "max_confidence": result.confidence,
                                        "snapshot_url": snap_url,
                                    }
                                else:
                                    p = self._person_tracker[result.person_id]
                                    p["last_seen_frame"] = frame_count
                                    p["last_seen_time"] = round(video_timestamp, 1)
                                    p["appearances"] += 1
                                    p["max_confidence"] = max(
                                        p["max_confidence"], result.confidence
                                    )

                                # Attribute recognition
                                if attribute_recognizer:
                                    try:
                                        attribute_recognizer.maybe_extract(
                                            track_id=result.person_id,
                                            frame=clean_frame,
                                            face_location=location,
                                            on_complete=self._handle_attributes,
                                        )
                                    except Exception:
                                        pass

                            else:
                                # ── Unknown Person (with dedup) ────
                                self._track_unknown_face(
                                    encoding=encoding,
                                    recognizer=recognizer,
                                    detector=detector,
                                    frame=clean_frame,
                                    location=location,
                                    frame_count=frame_count,
                                    video_timestamp=video_timestamp,
                                    tracker=tracker,
                                    camera_id=camera_id,
                                    centroid=centroid,
                                )

                        # ── Crowd Detection ────────────────────
                        if crowd_detector:
                            try:
                                centroids = tracker.get_person_centroids()
                                if len(centroids) >= 2:
                                    crowd_detector.update(centroids, camera_id)
                            except Exception:
                                pass

                        # ── Loitering Detection ────────────────
                        if loitering_detector:
                            try:
                                centroids = tracker.get_person_centroids()
                                names = tracker.get_person_names()
                                if centroids:
                                    loitering_detector.update(
                                        centroids, names, camera_id
                                    )
                            except Exception:
                                pass

                except Exception as e:
                    logger.debug(f"Frame {frame_count} processing error: {e}")
                    continue

                # Check exits periodically (use video time)
                try:
                    tracker.check_exits(current_time=video_timestamp)
                except Exception:
                    pass

                # Save key frames at intervals (every ~5 seconds of video)
                if frame_count % int(fps * 5) == 0:
                    self._save_key_frame(frame, frame_count, video_timestamp)

            # ── Finalize ──────────────────────────────────────────
            cap.release()

            # Flush remaining exits
            try:
                tracker.check_exits()
            except Exception:
                pass

            # Wait for async modules (ANPR, hazard, traffic) to finish processing
            logger.info(f"Waiting for async modules to complete...")
            if plate_recognizer:
                try:
                    plate_recognizer.shutdown()
                    logger.info(f"ANPR finished: {plate_recognizer._total_plates} plates detected")
                except Exception as e:
                    logger.warning(f"ANPR shutdown error: {e}")
            if hazard_detector:
                try:
                    hazard_detector.shutdown()
                except Exception:
                    pass
            if traffic_monitor:
                try:
                    traffic_monitor.shutdown()
                except Exception:
                    pass
            if attribute_recognizer:
                try:
                    attribute_recognizer.shutdown()
                except Exception:
                    pass

            # Force-exit all remaining active person tracks
            try:
                far_future = duration + 9999  # Ensures all tracks exceed exit threshold
                tracker.check_exits(current_time=far_future)
            except Exception:
                pass

            # Consolidate plate readings into unique vehicles
            self._consolidate_plates()

            # Generate clean vehicle timeline events (entry/exit per vehicle)
            self._build_vehicle_timeline(fps)

            # Generate clean person timeline events
            self._build_person_timeline(fps)

            # Build final results
            self.result.detected_persons = list(self._person_tracker.values())
            self.result.unknown_persons = list(self._unknown_tracker.values())
            self.result.detected_vehicles = list(self._plate_tracker.values())

            # Sort timeline by frame number
            self.result.events_timeline.sort(key=lambda e: e.get("frame", 0))

            # Build summary
            self.result.summary = {
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "duration_seconds": round(duration, 1),
                "unique_persons_recognized": len(self._person_tracker),
                "unknown_faces_detected": len(self.result.unknown_persons),
                "vehicles_detected": len(self._plate_tracker),
                "raw_plate_readings": len(self._raw_plate_readings),
                "security_alerts": len(self.result.security_alerts),
                "traffic_alerts": len(self.result.traffic_alerts),
                "total_events": len(self.result.events_timeline),
                "key_frames_saved": len(self.result.key_frames),
            }

            self.result.status = "completed"
            self.result.progress = 100.0
            self.result.completed_at = datetime.now(timezone.utc).isoformat()
            self._emit_progress(100)

            logger.info(
                f"Video analysis completed [{self.job_id[:8]}]: "
                f"{len(self._person_tracker)} persons, "
                f"{len(self._plate_tracker)} vehicles, "
                f"{len(self.result.security_alerts)} alerts"
            )

        except Exception as e:
            logger.error(f"Video analysis failed [{self.job_id[:8]}]: {e}", exc_info=True)
            self.result.status = "failed"
            self.result.error = str(e)
            self._emit_progress(self.result.progress)

    # ── Event Handlers ────────────────────────────────────────────────────────

    def _handle_event(self, event_type: str, event: dict):
        """Handle entry/exit events from the tracker."""
        event["source"] = "video_analysis"
        event["job_id"] = self.job_id
        self.result.events_timeline.append(event)

    def _handle_vehicle_event(self, event: dict):
        """Collect raw ANPR plate readings for later consolidation.
        Does NOT add to timeline — clean events are generated after consolidation."""
        meta = event.get("metadata", {})
        plate = meta.get("plate", "")
        if plate:
            self._raw_plate_readings.append({
                "plate": plate,
                "vehicle_type": meta.get("vehicle_type", "unknown"),
                "confidence": meta.get("confidence", 0),
                "vehicle_confidence": meta.get("vehicle_confidence", 0),
                "timestamp": event.get("timestamp", ""),
                "bounding_box": meta.get("bounding_box", {}),
                "frame": self._current_frame,  # Track which frame this was
            })

    def _track_unknown_face(
        self, encoding, recognizer, detector, frame, location,
        frame_count, video_timestamp, tracker, camera_id, centroid
    ):
        """
        Track an unknown face with deduplication.

        Compares the face encoding against previously seen unknowns.
        If similar (distance < 0.55), updates the existing entry.
        Otherwise creates a new unknown person entry.
        Also feeds unknowns into the PersonTracker so entry/exit events work.
        """
        import numpy as np

        UNKNOWN_MATCH_TOLERANCE = 0.55
        matched_id = None

        # Compare against previously seen unknowns
        for uid, prev_encoding in self._unknown_encodings:
            try:
                distance = recognizer.compare_unknown_faces(encoding, prev_encoding)
                if distance <= UNKNOWN_MATCH_TOLERANCE:
                    matched_id = uid
                    break
            except Exception:
                continue

        if matched_id and matched_id in self._unknown_tracker:
            # Update existing unknown person
            entry = self._unknown_tracker[matched_id]
            entry["last_seen_frame"] = frame_count
            entry["last_seen_time"] = round(video_timestamp, 1)
            entry["appearances"] = entry.get("appearances", 1) + 1

            # Feed into tracker for entry/exit detection
            tracker.on_detection(
                person_id=matched_id,
                person_name=f"Unknown Person {entry.get('index', '?')}",
                camera_id=camera_id,
                confidence=0.0,
                centroid=centroid,
                timestamp=video_timestamp,
            )
        else:
            # New unknown person
            from app.utils.image_utils import save_snapshot

            unknown_index = len(self._unknown_tracker) + 1
            uid = f"unknown_{self.job_id[:8]}_{unknown_index}"

            face_crop = detector.extract_face_crop(frame, location, padding=40)
            _, snap_url = save_snapshot(
                face_crop,
                f"va_{uid}",
                subdirectory="video_analysis",
                quality=90,
            )

            top, right, bottom, left = location
            self._unknown_tracker[uid] = {
                "unknown_id": uid,
                "index": unknown_index,
                "first_seen_frame": frame_count,
                "first_seen_time": round(video_timestamp, 1),
                "last_seen_frame": frame_count,
                "last_seen_time": round(video_timestamp, 1),
                "appearances": 1,
                "snapshot_url": snap_url,
                "bounding_box": {
                    "top": top, "right": right,
                    "bottom": bottom, "left": left,
                },
            }

            # Store encoding for future comparisons
            self._unknown_encodings.append((uid, encoding))

            # Feed into tracker for entry/exit detection
            tracker.on_detection(
                person_id=uid,
                person_name=f"Unknown Person {unknown_index}",
                camera_id=camera_id,
                confidence=0.0,
                centroid=centroid,
                timestamp=video_timestamp,
            )

            logger.debug(
                f"New unknown person #{unknown_index} at frame {frame_count} "
                f"(video time: {video_timestamp:.1f}s)"
            )

    def _build_person_timeline(self, fps: float):
        """
        Generate clean timeline events from tracked persons (known + unknown).

        Produces 'Person Entered' and 'Person Left' events with video
        timestamps in mm:ss format.
        """
        fps = fps or 30.0

        def frame_to_video_time(frame_num: int) -> str:
            seconds = frame_num / fps
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}:{s:02d}"

        # Known persons from _person_tracker
        for pid, info in self._person_tracker.items():
            first_frame = info.get("first_seen_frame", 0)
            last_frame = info.get("last_seen_frame", first_frame)
            name = info.get("person_name", "Unknown")

            self.result.events_timeline.append({
                "event_type": "person_entered",
                "frame": first_frame,
                "video_time": frame_to_video_time(first_frame),
                "description": f"{name} entered the frame",
                "person_name": name,
                "metadata": {
                    "person_id": pid,
                    "confidence": info.get("max_confidence", 0),
                },
            })

            if last_frame > first_frame:
                self.result.events_timeline.append({
                    "event_type": "person_left",
                    "frame": last_frame,
                    "video_time": frame_to_video_time(last_frame),
                    "description": f"{name} left the frame",
                    "person_name": name,
                    "metadata": {"person_id": pid},
                })

        # Unknown persons from _unknown_tracker
        for uid, info in self._unknown_tracker.items():
            first_frame = info.get("first_seen_frame", 0)
            last_frame = info.get("last_seen_frame", first_frame)
            label = f"Unknown Person {info.get('index', '?')}"

            self.result.events_timeline.append({
                "event_type": "person_entered",
                "frame": first_frame,
                "video_time": frame_to_video_time(first_frame),
                "description": f"{label} entered the frame",
                "person_name": label,
                "metadata": {"unknown_id": uid, "appearances": info.get("appearances", 1)},
            })

            if last_frame > first_frame:
                self.result.events_timeline.append({
                    "event_type": "person_left",
                    "frame": last_frame,
                    "video_time": frame_to_video_time(last_frame),
                    "description": f"{label} left the frame",
                    "person_name": label,
                    "metadata": {"unknown_id": uid},
                })

    def _consolidate_plates(self):
        """
        Consolidate raw plate readings into unique vehicles.

        The same plate appears many times across frames with slightly different
        OCR readings (e.g., KA02HK4826, KA02HHL826, KA02KM1826 are all the
        same plate). This method:
        1. Pre-filters readings by plate format validity
        2. Clusters similar readings using edit distance
        3. Picks the best plate text per cluster (majority voting)
        4. Requires minimum evidence (readings across multiple frames)
        5. Populates self._plate_tracker with unique vehicles only
        """
        import re
        from collections import Counter

        if not self._raw_plate_readings:
            return

        logger.info(
            f"Consolidating {len(self._raw_plate_readings)} raw plate readings..."
        )

        # ── Step 1: Pre-filter and correct readings ─────────────────────────
        # Use Indian plate correction to fix OCR errors (e.g., X02HH7256 → KA02MH7256)
        from app.vision.anpr import _fix_indian_plate

        valid_readings = []
        for reading in self._raw_plate_readings:
            plate = reading["plate"]
            # Basic checks: must be 6-14 chars, have both letters and digits
            if len(plate) < 5 or len(plate) > 14:
                continue
            has_letters = any(c.isalpha() for c in plate)
            has_digits = any(c.isdigit() for c in plate)
            if not (has_letters and has_digits):
                continue

            # Try Indian plate correction (fixes OCR errors + validates state code)
            corrected = _fix_indian_plate(plate)
            if corrected:
                reading = reading.copy()  # Don't mutate original
                reading["plate"] = corrected
                valid_readings.append(reading)
            else:
                # Fallback: accept if matches basic plate patterns
                PLATE_PATTERNS = [
                    re.compile(r'^[A-Z]{1,3}\d{1,2}[A-Z]{1,3}\d{1,5}$'),
                    re.compile(r'^[A-Z0-9]{2}\d{2}[A-Z]{1,3}\d{2,5}$'),
                    re.compile(r'^[A-Z]{1,2}\d{2}[A-Z]{1,3}\d{1,4}$'),
                ]
                if any(p.match(plate) for p in PLATE_PATTERNS):
                    valid_readings.append(reading)

        logger.info(
            f"  Format filter: {len(self._raw_plate_readings)} → "
            f"{len(valid_readings)} valid-format readings"
        )

        if not valid_readings:
            logger.warning("  No readings matched plate format patterns")
            return

        # ── Step 2: Cluster similar readings ──────────────────────────────
        clusters: List[List[dict]] = []

        for reading in valid_readings:
            plate = reading["plate"]
            merged = False

            for cluster in clusters:
                # Compare with ALL readings in the cluster (not just first)
                # This handles drift where A≈B and B≈C but A≠C
                representative = cluster[0]["plate"]
                if self._plates_similar(plate, representative):
                    cluster.append(reading)
                    merged = True
                    break

            if not merged:
                clusters.append([reading])

        # ── Step 3: Filter by minimum evidence & build results ────────────
        MIN_READINGS = 2  # Must appear in at least 2 frames

        for cluster in clusters:
            if len(cluster) < MIN_READINGS:
                plates_in_cluster = list(set(r["plate"] for r in cluster))
                logger.debug(
                    f"  Discarded cluster with {len(cluster)} readings "
                    f"(< {MIN_READINGS} min): {plates_in_cluster}"
                )
                continue

            # Majority voting for best plate text
            plates = [r["plate"] for r in cluster]
            best_plate = self._majority_vote_plate(plates)

            # Best confidence reading
            best_reading = max(cluster, key=lambda r: r.get("confidence", 0))
            # Most common vehicle type
            vtypes = [r.get("vehicle_type", "unknown") for r in cluster]
            best_vtype = max(set(vtypes), key=vtypes.count)

            self._plate_tracker[best_plate] = {
                "plate": best_plate,
                "vehicle_type": best_vtype,
                "confidence": round(best_reading.get("confidence", 0), 3),
                "vehicle_confidence": round(best_reading.get("vehicle_confidence", 0), 3),
                "first_seen": cluster[0].get("timestamp", ""),
                "last_seen": cluster[-1].get("timestamp", ""),
                "total_readings": len(cluster),
                "all_readings": list(set(plates)),
            }

        logger.info(
            f"Plate consolidation: {len(self._raw_plate_readings)} raw → "
            f"{len(valid_readings)} valid → "
            f"{len(self._plate_tracker)} unique vehicles "
            f"(min {MIN_READINGS} readings required)"
        )
        for plate, info in self._plate_tracker.items():
            logger.info(
                f"  🚗 {plate} ({info['vehicle_type']}, "
                f"conf={info['confidence']:.2f}, "
                f"readings={info['total_readings']}, "
                f"raw: {info['all_readings'][:5]})"
            )

    def _build_vehicle_timeline(self, fps: float):
        """
        Generate clean timeline events from consolidated vehicles.

        Instead of 30+ raw 'vehicle_entry' events per plate, produce exactly
        2 events per unique vehicle: 'Vehicle Entered' and 'Vehicle Left',
        with video timestamps in mm:ss format.
        """
        if not self._plate_tracker:
            return

        fps = fps or 30.0  # Fallback

        def frame_to_video_time(frame_num: int) -> str:
            """Convert frame number to mm:ss video timestamp."""
            seconds = frame_num / fps
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}:{s:02d}"

        for plate, info in self._plate_tracker.items():
            # Find earliest and latest frame numbers for this vehicle
            readings_for_plate = [
                r for r in self._raw_plate_readings
                if self._plates_similar(r["plate"], plate)
            ]

            if not readings_for_plate:
                continue

            frames = [r.get("frame", 0) for r in readings_for_plate]
            first_frame = min(frames)
            last_frame = max(frames)

            # Vehicle Entered event
            self.result.events_timeline.append({
                "event_type": "vehicle_entered",
                "frame": first_frame,
                "video_time": frame_to_video_time(first_frame),
                "description": f"{info['vehicle_type'].title()} entered — Plate: {plate}",
                "metadata": {
                    "plate": plate,
                    "vehicle_type": info["vehicle_type"],
                    "confidence": info["confidence"],
                },
            })

            # Vehicle Left event (only if it was seen across multiple frames)
            if last_frame > first_frame:
                self.result.events_timeline.append({
                    "event_type": "vehicle_left",
                    "frame": last_frame,
                    "video_time": frame_to_video_time(last_frame),
                    "description": f"{info['vehicle_type'].title()} left — Plate: {plate}",
                    "metadata": {
                        "plate": plate,
                        "vehicle_type": info["vehicle_type"],
                    },
                })

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return VideoAnalysisJob._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    @staticmethod
    def _plates_similar(plate1: str, plate2: str, threshold: float = 0.6) -> bool:
        """
        Check if two plate readings are likely the same physical plate.

        Uses edit distance relative to the longer string length.
        A threshold of 0.6 means at least 60% of characters must match.
        """
        if not plate1 or not plate2:
            return False

        distance = VideoAnalysisJob._edit_distance(plate1, plate2)
        max_len = max(len(plate1), len(plate2))
        similarity = 1 - (distance / max_len) if max_len > 0 else 0

        return similarity >= threshold

    @staticmethod
    def _majority_vote_plate(plates: List[str]) -> str:
        """
        Given multiple OCR readings of the same plate, produce the best text
        using character-level majority voting.

        Example:
            ['KA02HK4826', 'KA02HHL826', 'KA02KM1826']
            → Position 0: K,K,K → K
            → Position 1: A,A,A → A
            → ...etc

        Falls back to the most frequent full reading if lengths vary too much.
        """
        from collections import Counter

        if not plates:
            return ""
        if len(plates) == 1:
            return plates[0]

        # First, try: most frequent exact reading
        freq = Counter(plates)
        most_common_plate, most_common_count = freq.most_common(1)[0]

        # If one reading appears significantly more than others, use it
        if most_common_count >= len(plates) * 0.4:
            return most_common_plate

        # Otherwise, do character-level majority voting
        # Use the most common length
        lengths = [len(p) for p in plates]
        target_len = Counter(lengths).most_common(1)[0][0]

        # Filter to readings with the target length (or close to it)
        filtered = [p for p in plates if abs(len(p) - target_len) <= 1]
        if not filtered:
            filtered = plates

        result_chars = []
        for pos in range(target_len):
            chars_at_pos = []
            for p in filtered:
                if pos < len(p):
                    chars_at_pos.append(p[pos])
            if chars_at_pos:
                result_chars.append(Counter(chars_at_pos).most_common(1)[0][0])

        voted = "".join(result_chars)
        return voted if voted else most_common_plate

    def _handle_alert(self, alert_type: str, alert: dict):
        """Handle security/safety alerts."""
        alert["alert_source"] = alert_type
        if alert_type == "traffic":
            self.result.traffic_alerts.append(alert)
        else:
            self.result.security_alerts.append(alert)
        self.result.events_timeline.append(alert)

    def _handle_attributes(self, track_id: str, attributes: dict):
        """Handle attribute recognition results."""
        if track_id in self._person_tracker:
            self._person_tracker[track_id]["attributes"] = attributes

    def _save_key_frame(self, frame, frame_count: int, video_time: float):
        """Save an annotated frame as a key frame snapshot."""
        try:
            _, url = save_snapshot(
                frame,
                f"va_keyframe_{self.job_id[:8]}_{frame_count}",
                subdirectory="video_analysis",
                quality=80,
            )
            self.result.key_frames.append({
                "frame": frame_count,
                "video_time": round(video_time, 1),
                "snapshot_url": url,
            })
        except Exception as e:
            logger.debug(f"Failed to save key frame: {e}")

    def _emit_progress(self, progress: float):
        """Emit progress update."""
        if self.on_progress:
            try:
                self.on_progress(self.job_id, progress, self.result.status)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Video Analysis Manager (Singleton)
# ═══════════════════════════════════════════════════════════════════════════════

class VideoAnalysisManager:
    """
    Central manager for all video analysis jobs.

    Handles:
    - Starting new analyses
    - Tracking active and completed jobs
    - Limiting concurrent analyses
    - Cleanup
    """

    def __init__(self):
        self._jobs: Dict[str, VideoAnalysisJob] = {}
        self._lock = threading.Lock()
        self._progress_callbacks: Dict[str, Callable] = {}

        # Ensure upload directory exists
        os.makedirs(settings.VIDEO_UPLOAD_DIR, exist_ok=True)

    def start_analysis(
        self,
        video_path: str,
        filename: str,
        on_progress: Optional[Callable] = None,
    ) -> str:
        """
        Start a new video analysis job.

        Args:
            video_path: Path to the uploaded video file
            filename: Original filename for display
            on_progress: Optional callback(job_id, progress, status)

        Returns:
            job_id: Unique identifier for this analysis

        Raises:
            RuntimeError: If max concurrent analyses reached
        """
        with self._lock:
            active_count = sum(
                1 for j in self._jobs.values()
                if j.result.status == "processing"
            )
            if active_count >= settings.VIDEO_ANALYSIS_MAX_CONCURRENT:
                raise RuntimeError(
                    f"Maximum concurrent analyses ({settings.VIDEO_ANALYSIS_MAX_CONCURRENT}) "
                    f"reached. Please wait for a running analysis to complete."
                )

            job_id = str(uuid4())
            job = VideoAnalysisJob(
                job_id=job_id,
                video_path=video_path,
                filename=filename,
                on_progress=on_progress,
            )
            self._jobs[job_id] = job
            job.start()

            return job_id

    def get_job(self, job_id: str) -> Optional[VideoAnalysisJob]:
        """Get a specific job by ID."""
        return self._jobs.get(job_id)

    def get_status(self, job_id: str) -> Optional[dict]:
        """Get status for a specific job."""
        job = self._jobs.get(job_id)
        if job:
            return job.result.to_status_dict()
        return None

    def get_results(self, job_id: str) -> Optional[dict]:
        """Get full results for a specific job."""
        job = self._jobs.get(job_id)
        if job:
            return job.result.to_dict()
        return None

    def get_all_jobs(self) -> List[dict]:
        """Get summary of all jobs."""
        return [
            job.result.to_status_dict()
            for job in sorted(
                self._jobs.values(),
                key=lambda j: j.result.created_at,
                reverse=True,
            )
        ]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated files."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        # Stop if running
        if job.result.status == "processing":
            job.stop()

        # Clean up video file
        try:
            if os.path.exists(job.video_path):
                os.remove(job.video_path)
        except Exception as e:
            logger.warning(f"Failed to delete video file: {e}")

        del self._jobs[job_id]
        return True

    def stop_all(self):
        """Stop all running analyses."""
        for job in self._jobs.values():
            if job.result.status == "processing":
                job.stop()


# Singleton instance
video_analysis_manager = VideoAnalysisManager()
