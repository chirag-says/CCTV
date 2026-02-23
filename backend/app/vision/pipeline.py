# """
# Main Vision Processing Pipeline.

# Orchestrates frame capture → detection → recognition → tracking → event emission.

# Supports two modes:
#   FACE mode (default):
#     - Face Detection → Recognition → Tracking → Entry/Exit
#     - Attribute Recognition (gender + clothing color)
#     - Crowd / Gathering Detection
#     - Loitering / Idle Detection
#     - Hazard Detection (YOLOv8)

#   TRAFFIC mode:
#     - Vehicle Detection + License Plate Recognition (ANPR)
#     - Human-Vehicle proximity safety alerts

# Runs as a background worker per camera.
# """

# import cv2
# import numpy as np
# import pickle
# import time
# import asyncio
# import logging
# import os
# from typing import Optional, Dict, List, Callable, Tuple
# from datetime import datetime, timezone
# from uuid import uuid4

# from app.config import settings
# from app.vision.detector import FaceDetector
# from app.vision.recognizer import FaceRecognizer, MatchResult
# from app.vision.tracker import PersonTracker
# from app.vision.safety_analytics import (
#     AttributeRecognizer,
#     CrowdDetector,
#     LoiteringDetector,
# )
# from app.vision.hazard_detector import HazardDetector
# from app.vision.anpr import PlateRecognizer
# from app.vision.traffic import TrafficMonitor
# from app.utils.image_utils import save_snapshot, frame_to_jpeg_bytes

# logger = logging.getLogger(__name__)


# class VisionPipeline:
#     """
#     Main processing pipeline for a single camera stream.
    
#     Supports FACE mode and TRAFFIC mode (see module docstring).
#     Mode is controlled by settings.PIPELINE_MODE.
#     """

#     def __init__(
#         self,
#         camera_id: str,
#         stream_source: str = "0",
#         on_event: Optional[Callable] = None,
#         on_frame: Optional[Callable] = None,
#         on_unknown: Optional[Callable] = None,
#     ):
#         self.camera_id = camera_id
#         self.stream_source = stream_source
#         self._running = False
#         self._capture: Optional[cv2.VideoCapture] = None

#         # Callbacks
#         self.on_event = on_event  # Called on entry/exit/detection events
#         self.on_frame = on_frame  # Called with processed frame (for streaming)
#         self.on_unknown = on_unknown  # Called when unknown face detected

#         # Sub-modules
#         self.detector = FaceDetector()
#         self.recognizer = FaceRecognizer()
#         self.tracker = PersonTracker(
#             on_entry=self._handle_entry_event,
#             on_exit=self._handle_exit_event,
#         )

#         # ── Safety & Security Modules ─────────────────────────────
#         # Attribute Recognition (gender + clothing)
#         self.attribute_recognizer: Optional[AttributeRecognizer] = None
#         if settings.ENABLE_ATTRIBUTE_RECOGNITION:
#             self.attribute_recognizer = AttributeRecognizer(max_workers=2)
#             logger.info(f"[{camera_id}] Attribute Recognition enabled")

#         # Crowd / Gathering Detection
#         self.crowd_detector: Optional[CrowdDetector] = None
#         if settings.ENABLE_CROWD_DETECTION:
#             self.crowd_detector = CrowdDetector(
#                 proximity_px=settings.CROWD_PROXIMITY_PX,
#                 min_persons=settings.CROWD_MIN_PERSONS,
#                 sustain_seconds=settings.CROWD_SUSTAIN_SECONDS,
#                 on_alert=self._handle_security_alert,
#             )
#             logger.info(f"[{camera_id}] Crowd Detection enabled")

#         # Loitering / Idle Detection
#         self.loitering_detector: Optional[LoiteringDetector] = None
#         if settings.ENABLE_LOITERING_DETECTION:
#             self.loitering_detector = LoiteringDetector(
#                 movement_threshold_px=settings.LOITER_MOVEMENT_THRESHOLD_PX,
#                 time_window_sec=settings.LOITER_TIME_WINDOW_SEC,
#                 alert_cooldown_sec=settings.LOITER_ALERT_COOLDOWN_SEC,
#                 on_alert=self._handle_security_alert,
#             )
#             logger.info(f"[{camera_id}] Loitering Detection enabled")

#         # Hazard Detection (YOLOv8)
#         self.hazard_detector: Optional[HazardDetector] = None
#         if settings.ENABLE_HAZARD_DETECTION:
#             self.hazard_detector = HazardDetector(
#                 model_path=settings.HAZARD_MODEL_PATH,
#                 frame_interval=settings.HAZARD_FRAME_INTERVAL,
#                 alert_cooldown_sec=settings.HAZARD_ALERT_COOLDOWN_SEC,
#                 on_alert=self._handle_security_alert,
#             )
#             logger.info(f"[{camera_id}] Hazard Detection enabled")

#         # ── Traffic Mode Modules ───────────────────────────────────
#         self.pipeline_mode = settings.PIPELINE_MODE.lower()

#         # ANPR (License Plate Recognition)
#         self.plate_recognizer: Optional[PlateRecognizer] = None
#         if self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_ANPR:
#             languages = [l.strip() for l in settings.ANPR_OCR_LANGUAGES.split(",")]
#             self.plate_recognizer = PlateRecognizer(
#                 yolo_model_path=settings.ANPR_MODEL_PATH,
#                 frame_interval=settings.ANPR_FRAME_INTERVAL,
#                 plate_cooldown_sec=settings.ANPR_PLATE_COOLDOWN_SEC,
#                 on_plate=self._handle_vehicle_event,
#                 languages=languages,
#             )
#             logger.info(f"[{camera_id}] ANPR enabled (languages={languages})")

#         # Traffic Monitor (Vehicle-Person Safety)
#         self.traffic_monitor: Optional[TrafficMonitor] = None
#         if self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_TRAFFIC_MONITOR:
#             self.traffic_monitor = TrafficMonitor(
#                 model_path=settings.TRAFFIC_MODEL_PATH,
#                 frame_interval=settings.TRAFFIC_FRAME_INTERVAL,
#                 proximity_px=settings.TRAFFIC_PROXIMITY_PX,
#                 alert_cooldown_sec=settings.TRAFFIC_ALERT_COOLDOWN_SEC,
#                 on_alert=self._handle_security_alert,
#             )
#             logger.info(f"[{camera_id}] Traffic Monitor enabled (proximity={settings.TRAFFIC_PROXIMITY_PX}px)")

#         # Pipeline state
#         self._frame_count = 0
#         self._last_frame = None  # Last processed frame (for MJPEG streaming)
#         self._last_jpeg = None   # Pre-encoded JPEG bytes (ready-to-send, encoded in worker thread)
#         self._last_exit_check = time.time()
#         self._exit_check_interval = 2  # Check exits every 2 seconds for faster response
#         self._unknown_cache: Dict[str, dict] = {}  # Temp unknown face dedup
#         self._last_safety_check = time.time()
#         self._safety_check_interval = 1.0  # Run crowd/loiter checks every 1s

#         # Performance metrics
#         self._fps = 0.0
#         self._last_fps_time = time.time()
#         self._fps_frame_count = 0

#         logger.info(f"Pipeline created for camera {camera_id}: source={stream_source}, mode={self.pipeline_mode}")

#     def start(self):
#         """Start the video capture."""
#         try:
#             # Try to interpret as integer (webcam index)
#             source = int(self.stream_source) if self.stream_source.isdigit() else self.stream_source
#             self._capture = cv2.VideoCapture(source)

#             if not self._capture.isOpened():
#                 raise RuntimeError(f"Cannot open camera source: {self.stream_source}")

#             # Optimize capture settings
#             self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             self._capture.set(cv2.CAP_PROP_FPS, 30)

#             self._running = True
#             logger.info(f"Camera {self.camera_id} started: {self.stream_source}")

#         except Exception as e:
#             logger.error(f"Failed to start camera {self.camera_id}: {e}")
#             raise

#     def stop(self):
#         """Stop the video capture and clean up."""
#         self._running = False
#         if self._capture:
#             self._capture.release()
#             self._capture = None
#         # Flush remaining exits
#         self.tracker.check_exits()

#         # Shutdown safety modules
#         if self.attribute_recognizer:
#             self.attribute_recognizer.shutdown()
#         if self.hazard_detector:
#             self.hazard_detector.shutdown()
#         if self.plate_recognizer:
#             self.plate_recognizer.shutdown()
#         if self.traffic_monitor:
#             self.traffic_monitor.shutdown()

#         logger.info(f"Camera {self.camera_id} stopped")

#     def process_frame(self) -> Optional[np.ndarray]:
#         """
#         Process a single frame through the full pipeline.
#         Dispatches to face mode or traffic mode based on settings.
        
#         Returns:
#             Annotated frame (with bounding boxes) or None if no frame available
#         """
#         if not self._running or not self._capture:
#             return None

#         ret, frame = self._capture.read()
#         if not ret:
#             logger.warning(f"Camera {self.camera_id}: Failed to read frame")
#             return None

#         self._frame_count += 1

#         # ── TRAFFIC MODE ──────────────────────────────────────
#         if self.pipeline_mode == "traffic":
#             return self._process_frame_traffic(frame)

#         # ── HYBRID MODE ───────────────────────────────────────
#         if self.pipeline_mode == "hybrid":
#             return self._process_frame_hybrid(frame)

#         # ── FACE MODE (default) ───────────────────────────────
#         # Keep a clean copy for snapshots (before drawing boxes)
#         clean_frame = frame.copy()

#         # ── HAZARD DETECTION (runs on its own schedule) ───────
#         # Submit frame to hazard detector — it internally throttles
#         # to every Nth frame and runs inference on a background thread.
#         # This is intentionally BEFORE the face-processing frame skip
#         # so hazard detection has its own independent sampling rate.
#         if self.hazard_detector:
#             try:
#                 self.hazard_detector.submit_frame(clean_frame, self.camera_id)
#             except Exception as e:
#                 logger.debug(f"Hazard detection submission failed: {e}")

#         # Frame skip for performance
#         if self._frame_count % settings.FRAME_SKIP != 0:
#             self._last_frame = frame
#             return frame  # Return unprocessed frame for display

#         # ── DETECTION ─────────────────────────────────────────
#         face_locations, rgb_frame = self.detector.detect_faces(frame)

#         if not face_locations:
#             self._run_safety_analytics()
#             self._check_periodic_exits()
#             self._update_fps()
#             self._last_frame = frame
#             return frame

#         logger.info(f"Camera {self.camera_id}: Detected {len(face_locations)} face(s)")

#         # ── ENCODING ──────────────────────────────────────────
#         encodings = self.recognizer.batch_generate_encodings(rgb_frame, face_locations)

#         # ── RECOGNITION & TRACKING ────────────────────────────
#         for i, (location, encoding) in enumerate(zip(face_locations, encodings)):
#             if encoding is None:
#                 continue

#             top, right, bottom, left = location

#             # Compute centroid for this detection
#             centroid = ((left + right) // 2, (top + bottom) // 2)

#             # Try to recognize
#             result: MatchResult = self.recognizer.recognize_face(encoding)

#             if result.matched:
#                 # ── KNOWN PERSON ──────────────────────────────
#                 self._draw_box(frame, location, result.person_name, (0, 255, 0), result.confidence)

#                 # Feed to tracker (handles entry/exit state transitions)
#                 self.tracker.on_detection(
#                     person_id=result.person_id,
#                     person_name=result.person_name,
#                     camera_id=self.camera_id,
#                     confidence=result.confidence,
#                     centroid=centroid,
#                 )

#                 # ── ATTRIBUTE RECOGNITION (once per track) ────
#                 if self.attribute_recognizer:
#                     try:
#                         self.attribute_recognizer.maybe_extract(
#                             track_id=result.person_id,
#                             frame=clean_frame,
#                             face_location=location,
#                             on_complete=self._handle_attributes_complete,
#                         )
#                     except Exception as e:
#                         logger.debug(f"Attribute extraction error: {e}")

#             else:
#                 # ── UNKNOWN PERSON ────────────────────────────
#                 self._draw_box(frame, location, "UNKNOWN", (0, 0, 255), 0.0)
#                 # Pass clean_frame to ensure snapshot doesn't have the red box
#                 self._handle_unknown_face(clean_frame, rgb_frame, location, encoding)

#         # ── SAFETY ANALYTICS (crowd + loitering) ──────────────
#         self._run_safety_analytics()

#         # ── PERIODIC EXIT CHECK ───────────────────────────────
#         self._check_periodic_exits()
#         self._update_fps()

#         self._last_frame = frame
#         return frame

#     # ═══════════════════════════════════════════════════════════════════════════
#     #  Safety & Security Integration
#     # ═══════════════════════════════════════════════════════════════════════════

#     def _run_safety_analytics(self):
#         """
#         Run crowd and loitering detection at a throttled interval.
#         Uses centroid data from the tracker — completely non-blocking
#         since all heavy work is already done by the time centroids exist.
#         """
#         now = time.time()
#         if now - self._last_safety_check < self._safety_check_interval:
#             return
#         self._last_safety_check = now

#         try:
#             centroids = self.tracker.get_person_centroids()
#             names = self.tracker.get_person_names()

#             # Crowd Detection
#             if self.crowd_detector and len(centroids) >= 2:
#                 self.crowd_detector.update(centroids, self.camera_id)

#             # Loitering Detection
#             if self.loitering_detector and centroids:
#                 self.loitering_detector.update(
#                     centroids, names, self.camera_id
#                 )
#         except Exception as e:
#             logger.debug(f"Safety analytics error: {e}")

#     def _handle_security_alert(self, alert: dict):
#         """
#         Callback for all security alerts (crowd, loitering, hazard,
#         vehicle_proximity). Routes through standard event system.
#         """
#         logger.warning(
#             f"Security alert on camera {self.camera_id}: "
#             f"{alert.get('subtype', 'unknown')} — {alert}"
#         )
#         if self.on_event:
#             self.on_event(alert)

#     def _handle_vehicle_event(self, event: dict):
#         """
#         Callback for ANPR vehicle entry events.
#         Routes through the standard event system.
#         """
#         logger.info(
#             f"Vehicle event on camera {self.camera_id}: "
#             f"plate={event.get('metadata', {}).get('plate', '???')}"
#         )
#         if self.on_event:
#             self.on_event(event)

#     def _handle_attributes_complete(self, track_id: str, attributes: dict):
#         """
#         Called when attribute recognition finishes for a person.
#         The attributes (gender, clothing color) can be attached to
#         detection events via the metadata JSONB column.
#         """
#         logger.info(
#             f"Attributes ready for {track_id[:8]}…: {attributes}"
#         )
#         # The attributes are cached in AttributeRecognizer._results
#         # and will be included in future events via get_attributes()

#     # ═══════════════════════════════════════════════════════════════════════════
#     #  Original Pipeline Methods (unchanged logic)
#     # ═══════════════════════════════════════════════════════════════════════════

#     def _handle_unknown_face(
#         self,
#         frame: np.ndarray,
#         rgb_frame: np.ndarray,
#         location: tuple,
#         encoding: np.ndarray,
#     ):
#         """Handle an unrecognized face — deduplicate and queue."""
#         # Check if this unknown is similar to any recently seen unknowns
#         similar_id = None
#         for uid, udata in self._unknown_cache.items():
#             distance = self.recognizer.compare_unknown_faces(
#                 encoding, udata["encoding"]
#             )
#             if distance < settings.UNKNOWN_SIMILARITY_THRESHOLD:
#                 similar_id = uid
#                 break

#         if similar_id:
#             # Update existing unknown
#             self._unknown_cache[similar_id]["count"] += 1
#             self._unknown_cache[similar_id]["last_seen"] = time.time()
#         else:
#             # New unknown face
#             unknown_id = str(uuid4())

#             # ── High-quality face crop with generous padding ──
#             face_crop = self.detector.extract_face_crop(frame, location, padding=80)
#             _, snapshot_url = save_snapshot(
#                 face_crop, f"unknown_{unknown_id}", subdirectory="unknowns", quality=95
#             )

#             # ── Wider context crop (head + shoulders) for better identification ──
#             context_crop = self._extract_context_crop(frame, location)
#             _, context_url = save_snapshot(
#                 context_crop, f"context_{unknown_id}", subdirectory="unknowns", quality=95
#             )

#             # ── Full frame at good quality ──
#             _, full_frame_url = save_snapshot(
#                 frame, f"fullframe_{unknown_id}", subdirectory="frames", quality=90
#             )

#             self._unknown_cache[unknown_id] = {
#                 "encoding": encoding,
#                 "count": 1,
#                 "first_seen": time.time(),
#                 "last_seen": time.time(),
#                 "snapshot_path": snapshot_url,
#                 "context_path": context_url,
#                 "full_frame_path": full_frame_url,
#             }

#             # Emit unknown face event
#             if self.on_unknown:
#                 self.on_unknown({
#                     "unknown_id": unknown_id,
#                     "camera_id": self.camera_id,
#                     "snapshot_path": snapshot_url,
#                     "context_path": context_url,
#                     "full_frame_path": full_frame_url,
#                     "encoding": encoding,
#                     "timestamp": datetime.now(timezone.utc).isoformat(),
#                     "bounding_box": {
#                         "top": location[0], "right": location[1],
#                         "bottom": location[2], "left": location[3]
#                     },
#                 })

#     def _handle_entry_event(self, event: dict):
#         """Callback when tracker confirms an entry."""
#         # Enrich with attribute data if available
#         if self.attribute_recognizer:
#             attrs = self.attribute_recognizer.get_attributes(
#                 event.get("person_id", "")
#             )
#             if attrs:
#                 event.setdefault("metadata", {}).update(attrs)

#         if self.on_event:
#             self.on_event(event)

#     def _handle_exit_event(self, event: dict):
#         """Callback when tracker detects an exit."""
#         # Enrich with attribute data if available
#         if self.attribute_recognizer:
#             attrs = self.attribute_recognizer.get_attributes(
#                 event.get("person_id", "")
#             )
#             if attrs:
#                 event.setdefault("metadata", {}).update(attrs)

#         if self.on_event:
#             self.on_event(event)

#     def _check_periodic_exits(self):
#         """Periodically check for exits."""
#         now = time.time()
#         if now - self._last_exit_check >= self._exit_check_interval:
#             self.tracker.check_exits(now)
#             self._last_exit_check = now

#     def _draw_box(
#         self,
#         frame: np.ndarray,
#         location: tuple,
#         label: str,
#         color: tuple,
#         confidence: float = 0.0,
#     ):
#         """Draw a bounding box and label on the frame."""
#         top, right, bottom, left = location

#         # Draw rectangle
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

#         # Draw label background
#         label_text = f"{label}"
#         if confidence > 0:
#             label_text += f" ({confidence:.0%})"

#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.6
#         thickness = 1
#         (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

#         cv2.rectangle(
#             frame,
#             (left, top - text_h - 10),
#             (left + text_w + 10, top),
#             color, cv2.FILLED,
#         )
#         cv2.putText(
#             frame, label_text,
#             (left + 5, top - 5),
#             font, font_scale, (255, 255, 255), thickness,
#         )

#     def _extract_context_crop(
#         self,
#         frame: np.ndarray,
#         location: tuple,
#     ) -> np.ndarray:
#         """
#         Extract a wider context crop (head + shoulders) from the frame.
#         This gives much better snapshots for unknown person identification.
#         """
#         top, right, bottom, left = location
#         h, w = frame.shape[:2]

#         face_w = right - left
#         face_h = bottom - top

#         # Expand to ~3x face width and ~4x face height (captures shoulders)
#         cx = (left + right) // 2
#         cy = (top + bottom) // 2

#         crop_w = int(face_w * 3)
#         crop_h = int(face_h * 4)

#         crop_left = max(0, cx - crop_w // 2)
#         crop_right = min(w, cx + crop_w // 2)
#         crop_top = max(0, top - int(face_h * 0.8))  # Some space above head
#         crop_bottom = min(h, crop_top + crop_h)

#         return frame[crop_top:crop_bottom, crop_left:crop_right].copy()

#     def _save_snapshot(self, image: np.ndarray, prefix: str) -> str:
#         """Save a snapshot image to disk. Returns URL-accessible path."""
#         _, url_path = save_snapshot(image, prefix)
#         return url_path

#     def _update_fps(self):
#         """Calculate processing FPS."""
#         self._fps_frame_count += 1
#         now = time.time()
#         elapsed = now - self._last_fps_time
#         if elapsed >= 1.0:
#             self._fps = self._fps_frame_count / elapsed
#             self._fps_frame_count = 0
#             self._last_fps_time = now

#     @property
#     def is_running(self) -> bool:
#         return self._running

#     @property
#     def fps(self) -> float:
#         return round(self._fps, 1)

#     # ═══════════════════════════════════════════════════════════════════════════
#     #  Traffic Mode Processing
#     # ═══════════════════════════════════════════════════════════════════════════

#     def _process_frame_traffic(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Process a frame in TRAFFIC mode.
#         Skips face recognition entirely; runs ANPR + TrafficMonitor.
#         """
#         frame = self._process_frame_traffic_logic(frame)
#         self._update_fps()
#         self._last_frame = frame
#         return frame


#     def _process_frame_hybrid(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Process a frame in HYBRID mode.
#         Runs BOTH Traffic (ANPR/Safety) and Face Recognition/Tracking.
#         """
#         # Keep a truly clean copy for Face Detection (must be before ANY drawing)
#         clean_frame_for_face = frame.copy()

#         # 1. Run Traffic Modules (they draw on frame first)
#         frame = self._process_frame_traffic_logic(frame)

#         # 2. Run Face Mode Logic
#         # Keep a copy for other uses (snapshots etc) - reusing the one above
#         clean_frame = clean_frame_for_face

#         # Hazard detection (if enabled)
#         if self.hazard_detector:
#              try:
#                  # Hazard detector manages its own threading/copying
#                  self.hazard_detector.submit_frame(clean_frame, self.camera_id)
#              except Exception as e:
#                  logger.debug(f"Hazard detection error: {e}")

#         # Frame skip for Face Rec (heavy)
#         if self._frame_count % settings.FRAME_SKIP != 0:
#             self._update_fps()
#             self._last_frame = frame
#             return frame

#         # Face Detection (Run on CLEAN frame, not the one with traffic boxes)
#         face_locations, rgb_frame = self.detector.detect_faces(clean_frame)

#         if not face_locations:
#             self._run_safety_analytics()
#             self._check_periodic_exits()
#             self._update_fps()
#             self._last_frame = frame
#             return frame

#         # Encoding
#         encodings = self.recognizer.batch_generate_encodings(rgb_frame, face_locations)

#         # Recognition & Tracking
#         for i, (location, encoding) in enumerate(zip(face_locations, encodings)):
#              if encoding is None: continue
#              top, right, bottom, left = location
#              centroid = ((left + right) // 2, (top + bottom) // 2)

#              result = self.recognizer.recognize_face(encoding)
#              if result.matched:
#                  # Known - Draw on the ACCUMULATED frame
#                  self._draw_box(frame, location, result.person_name, (0, 255, 0), result.confidence)
#                  self.tracker.on_detection(
#                      person_id=result.person_id,
#                      person_name=result.person_name,
#                      camera_id=self.camera_id,
#                      confidence=result.confidence,
#                      centroid=centroid,
#                  )
#                  # Attributes
#                  if self.attribute_recognizer:
#                      try:
#                          self.attribute_recognizer.maybe_extract(result.person_id, clean_frame, location, self._handle_attributes_complete)
#                      except Exception: pass
#              else:
#                  # Unknown - Draw on the ACCUMULATED frame
#                  self._draw_box(frame, location, "UNKNOWN", (0, 0, 255), 0.0)
#                  self._handle_unknown_face(clean_frame, rgb_frame, location, encoding)

#         # Analytics
#         self._run_safety_analytics()
#         self._check_periodic_exits()
#         self._update_fps()
#         self._last_frame = frame
#         return frame

#     def _process_frame_traffic_logic(self, frame: np.ndarray) -> np.ndarray:
#         """Helper to run traffic logic and return modified frame."""
#         clean_frame = frame.copy()
#         # Traffic Monitor
#         if self.traffic_monitor:
#             try:
#                 self.traffic_monitor.submit_frame(clean_frame, self.camera_id)
#                 frame = self.traffic_monitor.draw_detections(frame)
#             except Exception as e:
#                 logger.debug(f"Traffic monitor error: {e}")
#         # ANPR
#         if self.plate_recognizer:
#             try:
#                 self.plate_recognizer.submit_frame(clean_frame, self.camera_id)
#             except Exception as e:
#                 logger.debug(f"ANPR error: {e}")
#         return frame

#     @property
#     def status(self) -> dict:
#         base = {
#             "camera_id": self.camera_id,
#             "running": self._running,
#             "fps": self.fps,
#             "frames_processed": self._frame_count,
#             "pipeline_mode": self.pipeline_mode,
#         }

#         if self.pipeline_mode == "traffic":
#             # Traffic mode stats
#             if self.traffic_monitor:
#                 base["traffic_monitor"] = self.traffic_monitor.stats
#             if self.plate_recognizer:
#                 base["plate_recognizer"] = self.plate_recognizer.stats
#         elif self.pipeline_mode == "hybrid":
#             # Hybrid stats
#             if self.traffic_monitor:
#                 base["traffic_monitor"] = self.traffic_monitor.stats
#             if self.plate_recognizer:
#                 base["plate_recognizer"] = self.plate_recognizer.stats
#             base["faces_in_cache"] = self.recognizer.known_count
#             base["active_tracks"] = self.tracker.stats
#             base["unknown_cache_size"] = len(self._unknown_cache)
#         else:
#             # Face mode stats
#             base["faces_in_cache"] = self.recognizer.known_count
#             base["active_tracks"] = self.tracker.stats
#             base["unknown_cache_size"] = len(self._unknown_cache)
#             if self.hazard_detector:
#                 base["hazard_detector"] = self.hazard_detector.stats

#         base["safety_modules"] = {
#             "attribute_recognition": settings.ENABLE_ATTRIBUTE_RECOGNITION,
#             "crowd_detection": settings.ENABLE_CROWD_DETECTION,
#             "loitering_detection": settings.ENABLE_LOITERING_DETECTION,
#             "hazard_detection": settings.ENABLE_HAZARD_DETECTION,
#             "pipeline_mode": self.pipeline_mode,
#             "anpr": self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_ANPR,
#             "traffic_monitor": self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_TRAFFIC_MONITOR,
#         }
#         return base


"""
Main Vision Processing Pipeline.

Orchestrates frame capture → detection → recognition → tracking → event emission.

Supports two modes:
  FACE mode (default):
    - Face Detection → Recognition → Tracking → Entry/Exit
    - Attribute Recognition (gender + clothing color)
    - Crowd / Gathering Detection
    - Loitering / Idle Detection
    - Hazard Detection (YOLOv8)

  TRAFFIC mode:
    - Vehicle Detection + License Plate Recognition (ANPR)
    - Human-Vehicle proximity safety alerts

Runs as a background worker per camera.
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import asyncio
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

# ── Force CUDA to use NVIDIA GPU (must be before any torch import) ────────────
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── Third-Party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np

# ── Internal: Config ──────────────────────────────────────────────────────────
from app.config import settings

# ── Internal: Vision Core ─────────────────────────────────────────────────────
from app.vision.detector import FaceDetector
from app.vision.recognizer import FaceRecognizer, MatchResult
from app.vision.tracker import PersonTracker

# ── Internal: Safety & Security Analytics ─────────────────────────────────────
from app.vision.safety_analytics import (
    AttributeRecognizer,
    CrowdDetector,
    LoiteringDetector,
)
from app.vision.hazard_detector import HazardDetector

# ── Internal: Traffic & ANPR ──────────────────────────────────────────────────
from app.vision.anpr import PlateRecognizer
from app.vision.traffic import TrafficMonitor

# ── Internal: Utilities ───────────────────────────────────────────────────────
from app.utils.image_utils import frame_to_jpeg_bytes, save_snapshot

logger = logging.getLogger(__name__)

# ── Log GPU status on module load ─────────────────────────────────────────────
try:
    import torch
    _cuda_available = torch.cuda.is_available()
    _gpu_name = torch.cuda.get_device_name(0) if _cuda_available else "N/A"
    logger.info(f"🖥️  GPU Status: CUDA={'available' if _cuda_available else 'NOT available'}, Device={_gpu_name}")
except ImportError:
    logger.info("🖥️  GPU Status: PyTorch not installed, GPU acceleration disabled")


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

        # Sub-modules
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.tracker = PersonTracker(
            on_entry=self._handle_entry_event,
            on_exit=self._handle_exit_event,
        )

        # ── Safety & Security Modules ─────────────────────────────
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
        if settings.ENABLE_HAZARD_DETECTION:
            self.hazard_detector = HazardDetector(
                model_path=settings.HAZARD_MODEL_PATH,
                frame_interval=settings.HAZARD_FRAME_INTERVAL,
                alert_cooldown_sec=settings.HAZARD_ALERT_COOLDOWN_SEC,
                on_alert=self._handle_security_alert,
            )
            logger.info(f"[{camera_id}] Hazard Detection enabled")

        # ── Traffic Mode Modules ───────────────────────────────────
        self.pipeline_mode = settings.PIPELINE_MODE.lower()

        # ANPR (License Plate Recognition)
        self.plate_recognizer: Optional[PlateRecognizer] = None
        if self.pipeline_mode in ("traffic", "hybrid") and settings.ENABLE_ANPR:
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
            )
            logger.info(f"[{camera_id}] Traffic Monitor enabled (proximity={settings.TRAFFIC_PROXIMITY_PX}px)")

        # Pipeline state
        self._frame_count = 0
        self._last_frame = None  # Last processed frame (for MJPEG streaming)
        self._last_jpeg = None   # Pre-encoded JPEG bytes (ready-to-send, encoded in worker thread)
        self._last_exit_check = time.time()
        self._exit_check_interval = 2  # Check exits every 2 seconds for faster response
        self._unknown_cache: Dict[str, dict] = {}  # Temp unknown face dedup
        self._last_safety_check = time.time()
        self._safety_check_interval = 1.0  # Run crowd/loiter checks every 1s

        # Performance metrics
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

        logger.info(f"Pipeline created for camera {camera_id}: source={stream_source}, mode={self.pipeline_mode}")

    def start(self):
        """Start the video capture."""
        try:
            # Try to interpret as integer (webcam index)
            source = int(self.stream_source) if self.stream_source.isdigit() else self.stream_source
            self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                raise RuntimeError(f"Cannot open camera source: {self.stream_source}")

            # Optimize capture settings
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture.set(cv2.CAP_PROP_FPS, 30)

            self._running = True
            logger.info(f"Camera {self.camera_id} started: {self.stream_source}")

        except Exception as e:
            logger.error(f"Failed to start camera {self.camera_id}: {e}")
            raise

    def stop(self):
        """Stop the video capture and clean up."""
        self._running = False
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
        if not self._running or not self._capture:
            return None

        ret, frame = self._capture.read()
        if not ret:
            logger.warning(f"Camera {self.camera_id}: Failed to read frame")
            return None

        self._frame_count += 1

        # ── TRAFFIC MODE ──────────────────────────────────────
        if self.pipeline_mode == "traffic":
            return self._process_frame_traffic(frame)

        # ── HYBRID MODE ───────────────────────────────────────
        if self.pipeline_mode == "hybrid":
            return self._process_frame_hybrid(frame)

        # ── FACE MODE (default) ───────────────────────────────
        # Keep a clean copy for snapshots (before drawing boxes)
        clean_frame = frame.copy()

        # ── HAZARD DETECTION (runs on its own schedule) ───────
        # Submit frame to hazard detector — it internally throttles
        # to every Nth frame and runs inference on a background thread.
        # This is intentionally BEFORE the face-processing frame skip
        # so hazard detection has its own independent sampling rate.
        if self.hazard_detector:
            try:
                self.hazard_detector.submit_frame(clean_frame, self.camera_id)
            except Exception as e:
                logger.debug(f"Hazard detection submission failed: {e}")

        # Frame skip for performance
        if self._frame_count % settings.FRAME_SKIP != 0:
            self._last_frame = frame
            return frame  # Return unprocessed frame for display

        # ── DETECTION ─────────────────────────────────────────
        face_locations, rgb_frame = self.detector.detect_faces(frame)

        if not face_locations:
            self._run_safety_analytics()
            self._check_periodic_exits()
            self._update_fps()
            self._last_frame = frame
            return frame

        logger.info(f"Camera {self.camera_id}: Detected {len(face_locations)} face(s)")

        # ── ENCODING ──────────────────────────────────────────
        encodings = self.recognizer.batch_generate_encodings(rgb_frame, face_locations)

        # ── RECOGNITION & TRACKING ────────────────────────────
        for i, (location, encoding) in enumerate(zip(face_locations, encodings)):
            if encoding is None:
                continue

            top, right, bottom, left = location

            # Compute centroid for this detection
            centroid = ((left + right) // 2, (top + bottom) // 2)

            # Try to recognize
            result: MatchResult = self.recognizer.recognize_face(encoding)

            if result.matched:
                # ── KNOWN PERSON ──────────────────────────────
                self._draw_box(frame, location, result.person_name, (0, 255, 0), result.confidence)

                # Feed to tracker (handles entry/exit state transitions)
                self.tracker.on_detection(
                    person_id=result.person_id,
                    person_name=result.person_name,
                    camera_id=self.camera_id,
                    confidence=result.confidence,
                    centroid=centroid,
                )

                # ── ATTRIBUTE RECOGNITION (once per track) ────
                if self.attribute_recognizer:
                    try:
                        self.attribute_recognizer.maybe_extract(
                            track_id=result.person_id,
                            frame=clean_frame,
                            face_location=location,
                            on_complete=self._handle_attributes_complete,
                        )
                    except Exception as e:
                        logger.debug(f"Attribute extraction error: {e}")

            else:
                # ── UNKNOWN PERSON ────────────────────────────
                self._draw_box(frame, location, "UNKNOWN", (0, 0, 255), 0.0)
                # Pass clean_frame to ensure snapshot doesn't have the red box
                self._handle_unknown_face(clean_frame, rgb_frame, location, encoding, centroid)

        # ── SAFETY ANALYTICS (crowd + loitering) ──────────────
        self._run_safety_analytics()

        # ── PERIODIC EXIT CHECK ───────────────────────────────
        self._check_periodic_exits()
        self._update_fps()

        self._last_frame = frame
        return frame

    # ═══════════════════════════════════════════════════════════════════════════
    #  Safety & Security Integration
    # ═══════════════════════════════════════════════════════════════════════════

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

            # Crowd Detection
            if self.crowd_detector and len(centroids) >= 2:
                self.crowd_detector.update(centroids, self.camera_id)

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
        """
        logger.warning(
            f"Security alert on camera {self.camera_id}: "
            f"{alert.get('subtype', 'unknown')} — {alert}"
        )
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

    # ═══════════════════════════════════════════════════════════════════════════
    #  Original Pipeline Methods (unchanged logic)
    # ═══════════════════════════════════════════════════════════════════════════

    def _handle_unknown_face(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray,
        location: tuple,
        encoding: np.ndarray,
        centroid: Optional[Tuple[int, int]] = None,
    ):
        """Handle an unrecognized face — deduplicate, track, and queue."""
        # Check if this unknown is similar to any recently seen unknowns
        similar_id = None
        for uid, udata in self._unknown_cache.items():
            distance = self.recognizer.compare_unknown_faces(
                encoding, udata["encoding"]
            )
            if distance < settings.UNKNOWN_SIMILARITY_THRESHOLD:
                similar_id = uid
                break

        if similar_id:
            # Update existing unknown
            self._unknown_cache[similar_id]["count"] += 1
            self._unknown_cache[similar_id]["last_seen"] = time.time()

            # Track this unknown person so they appear in "People Present"
            unknown_label = self._unknown_cache[similar_id].get("label", "Unknown")
            self.tracker.on_detection(
                person_id=similar_id,
                person_name=unknown_label,
                camera_id=self.camera_id,
                confidence=0.0,
                centroid=centroid,
            )
        else:
            # New unknown face
            unknown_id = str(uuid4())

            # Assign a label like "Unknown #1", "Unknown #2", etc.
            unknown_number = len(self._unknown_cache) + 1
            unknown_label = f"Unknown #{unknown_number}"

            # ── High-quality face crop with generous padding ──
            face_crop = self.detector.extract_face_crop(frame, location, padding=80)
            _, snapshot_url = save_snapshot(
                face_crop, f"unknown_{unknown_id}", subdirectory="unknowns", quality=95
            )

            # ── Wider context crop (head + shoulders) for better identification ──
            context_crop = self._extract_context_crop(frame, location)
            _, context_url = save_snapshot(
                context_crop, f"context_{unknown_id}", subdirectory="unknowns", quality=95
            )

            # ── Full frame at good quality ──
            _, full_frame_url = save_snapshot(
                frame, f"fullframe_{unknown_id}", subdirectory="frames", quality=90
            )

            self._unknown_cache[unknown_id] = {
                "encoding": encoding,
                "count": 1,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "snapshot_path": snapshot_url,
                "context_path": context_url,
                "full_frame_path": full_frame_url,
                "label": unknown_label,
            }

            # Track this new unknown person so they appear in "People Present"
            self.tracker.on_detection(
                person_id=unknown_id,
                person_name=unknown_label,
                camera_id=self.camera_id,
                confidence=0.0,
                centroid=centroid,
            )

            # Emit unknown face event
            if self.on_unknown:
                self.on_unknown({
                    "unknown_id": unknown_id,
                    "camera_id": self.camera_id,
                    "snapshot_path": snapshot_url,
                    "context_path": context_url,
                    "full_frame_path": full_frame_url,
                    "encoding": encoding,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "bounding_box": {
                        "top": location[0], "right": location[1],
                        "bottom": location[2], "left": location[3]
                    },
                })

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

    def _check_periodic_exits(self):
        """Periodically check for exits."""
        now = time.time()
        if now - self._last_exit_check >= self._exit_check_interval:
            self.tracker.check_exits(now)
            self._last_exit_check = now

    def _draw_box(
        self,
        frame: np.ndarray,
        location: tuple,
        label: str,
        color: tuple,
        confidence: float = 0.0,
    ):
        """Draw a bounding box and label on the frame."""
        top, right, bottom, left = location

        # Draw rectangle
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

    def _update_fps(self):
        """Calculate processing FPS."""
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

    # ═══════════════════════════════════════════════════════════════════════════
    #  Traffic Mode Processing
    # ═══════════════════════════════════════════════════════════════════════════

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
        """
        # Keep a truly clean copy for Face Detection (must be before ANY drawing)
        clean_frame_for_face = frame.copy()

        # 1. Run Traffic Modules (they draw on frame first)
        frame = self._process_frame_traffic_logic(frame)

        # 2. Run Face Mode Logic
        # Keep a copy for other uses (snapshots etc) - reusing the one above
        clean_frame = clean_frame_for_face

        # Hazard detection (if enabled)
        if self.hazard_detector:
             try:
                 # Hazard detector manages its own threading/copying
                 self.hazard_detector.submit_frame(clean_frame, self.camera_id)
             except Exception as e:
                 logger.debug(f"Hazard detection error: {e}")

        # Frame skip for Face Rec (heavy)
        if self._frame_count % settings.FRAME_SKIP != 0:
            self._update_fps()
            self._last_frame = frame
            return frame

        # Face Detection (Run on CLEAN frame, not the one with traffic boxes)
        face_locations, rgb_frame = self.detector.detect_faces(clean_frame)

        if not face_locations:
            self._run_safety_analytics()
            self._check_periodic_exits()
            self._update_fps()
            self._last_frame = frame
            return frame

        # Encoding
        encodings = self.recognizer.batch_generate_encodings(rgb_frame, face_locations)

        # Recognition & Tracking
        for i, (location, encoding) in enumerate(zip(face_locations, encodings)):
             if encoding is None: continue
             top, right, bottom, left = location
             centroid = ((left + right) // 2, (top + bottom) // 2)

             result = self.recognizer.recognize_face(encoding)
             if result.matched:
                 # Known - Draw on the ACCUMULATED frame
                 self._draw_box(frame, location, result.person_name, (0, 255, 0), result.confidence)
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
             else:
                 # Unknown - Draw on the ACCUMULATED frame
                 self._draw_box(frame, location, "UNKNOWN", (0, 0, 255), 0.0)
                 self._handle_unknown_face(clean_frame, rgb_frame, location, encoding, centroid)

        # Analytics
        self._run_safety_analytics()
        self._check_periodic_exits()
        self._update_fps()
        self._last_frame = frame
        return frame

    def _process_frame_traffic_logic(self, frame: np.ndarray) -> np.ndarray:
        """Helper to run traffic logic and return modified frame."""
        clean_frame = frame.copy()
        # Traffic Monitor
        if self.traffic_monitor:
            try:
                self.traffic_monitor.submit_frame(clean_frame, self.camera_id)
                frame = self.traffic_monitor.draw_detections(frame)
            except Exception as e:
                logger.debug(f"Traffic monitor error: {e}")
        # ANPR
        if self.plate_recognizer:
            try:
                self.plate_recognizer.submit_frame(clean_frame, self.camera_id)
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
            base["unknown_cache_size"] = len(self._unknown_cache)
        else:
            # Face mode stats
            base["faces_in_cache"] = self.recognizer.known_count
            base["active_tracks"] = self.tracker.stats
            base["unknown_cache_size"] = len(self._unknown_cache)
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
        }
        return base