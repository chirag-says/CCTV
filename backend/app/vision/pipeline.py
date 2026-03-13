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
import threading
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

        # ── YOLO Person Detector (for body-level person detection) ────
        # Used in FACE mode to assist face detection when face_recognition
        # misses faces (e.g. partial occlusion, small faces, angled faces).
        # In HYBRID mode, we reuse the TrafficMonitor's YOLO detections.
        self._yolo_person_detector = None
        self._yolo_person_loaded = False
        self._yolo_person_lock = threading.Lock()
        self._yolo_body_unknown_cache: Dict[str, dict] = {}  # Dedup for body-only unknowns
        self._last_yolo_assist_check = 0.0  # Throttle: last time we ran YOLO-assist
        self._yolo_assist_interval = 3.0    # Only run YOLO-assisted every 3 seconds

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

        # ── YOLO-ASSISTED DETECTION (catch missed persons) ────
        # Throttled: only runs every N seconds to avoid killing FPS.
        # In FACE mode, runs a dedicated YOLO person detection.
        # In HYBRID mode, this path is NOT used (see _process_frame_hybrid).
        now_assist = time.time()
        if now_assist - self._last_yolo_assist_check >= self._yolo_assist_interval:
            self._last_yolo_assist_check = now_assist
            try:
                yolo_persons = self._get_yolo_person_detections(clean_frame)
                if yolo_persons:
                    missed = self._find_missed_persons(yolo_persons, face_locations)
                    for person_det in missed:
                        self._attempt_yolo_assisted_detection(
                            frame, clean_frame, person_det
                        )
            except Exception as e:
                logger.debug(f"YOLO-assisted detection error: {e}")

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
    #  YOLO-Assisted Person Detection
    #  When face_recognition misses a face but YOLO detects a person body,
    #  we attempt focused face detection on the head region and create an
    #  unknown snapshot if the face is still undetectable.
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_yolo_person_detector(self) -> bool:
        """Lazy-load a YOLO model for person detection (used in FACE mode)."""
        with self._yolo_person_lock:
            if self._yolo_person_loaded:
                return self._yolo_person_detector is not None
            try:
                from ultralytics import YOLO
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model_path = getattr(settings, 'HAZARD_MODEL_PATH', 'yolov8n.pt')
                self._yolo_person_detector = YOLO(model_path)
                self._yolo_person_device = device
                logger.info(f"[{self.camera_id}] YOLO person detector loaded on {device}")
            except Exception as e:
                logger.warning(f"[{self.camera_id}] YOLO person detector unavailable: {e}")
                self._yolo_person_detector = None
            finally:
                self._yolo_person_loaded = True
            return self._yolo_person_detector is not None

    def _get_yolo_person_detections(self, frame: np.ndarray) -> List[dict]:
        """
        Run YOLO person detection on a frame (FACE mode only).
        Returns list of person detection dicts with 'bbox' and 'confidence'.
        """
        if not self._load_yolo_person_detector():
            return []

        try:
            results = self._yolo_person_detector(
                frame,
                conf=0.25,
                classes=[0],  # person class only
                verbose=False,
                device=getattr(self, '_yolo_person_device', 'cpu'),
            )

            persons = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != 0:
                        continue
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                    persons.append({
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
            return persons
        except Exception as e:
            logger.debug(f"YOLO person detection failed: {e}")
            return []

    def _get_yolo_person_detections_from_traffic(self) -> List[dict]:
        """
        Get person detections from the TrafficMonitor (HYBRID mode).
        Avoids running YOLO twice since TrafficMonitor already does it.
        """
        if not self.traffic_monitor:
            return []

        try:
            with self.traffic_monitor._detections_lock:
                persons = list(self.traffic_monitor._last_detections.get("persons", []))
            return persons
        except Exception:
            return []

    def _find_missed_persons(
        self,
        yolo_persons: List[dict],
        face_locations: List[Tuple],
    ) -> List[dict]:
        """
        Find YOLO person detections that don't overlap with any detected face.
        These are persons whose faces were missed by face_recognition.
        """
        missed = []
        for person in yolo_persons:
            pb = person["bbox"]
            has_matching_face = False

            for face_loc in face_locations:
                if self._face_overlaps_person(face_loc, pb):
                    has_matching_face = True
                    break

            if not has_matching_face:
                missed.append(person)

        return missed

    @staticmethod
    def _face_overlaps_person(
        face_loc: Tuple[int, int, int, int],
        person_bbox: dict,
    ) -> bool:
        """
        Check whether a face detection (top, right, bottom, left) overlaps
        with a YOLO person bbox (x1, y1, x2, y2).
        """
        f_top, f_right, f_bottom, f_left = face_loc
        p_x1, p_y1, p_x2, p_y2 = (
            person_bbox["x1"], person_bbox["y1"],
            person_bbox["x2"], person_bbox["y2"],
        )

        # Check for overlap
        overlap_x = max(0, min(f_right, p_x2) - max(f_left, p_x1))
        overlap_y = max(0, min(f_bottom, p_y2) - max(f_top, p_y1))

        if overlap_x > 0 and overlap_y > 0:
            # Some overlap exists — check if it's significant
            face_area = (f_right - f_left) * (f_bottom - f_top)
            overlap_area = overlap_x * overlap_y
            # The face center should be inside the person box
            face_cx = (f_left + f_right) // 2
            face_cy = (f_top + f_bottom) // 2
            if p_x1 <= face_cx <= p_x2 and p_y1 <= face_cy <= p_y2:
                return True
            # Or significant overlap ratio
            if face_area > 0 and overlap_area / face_area > 0.3:
                return True

        return False

    def _attempt_yolo_assisted_detection(
        self,
        frame: np.ndarray,
        clean_frame: np.ndarray,
        person_det: dict,
    ):
        """
        Attempt to detect a face in the upper body of a YOLO person detection.
        If face detection succeeds, process normally. If not, create a body-only
        unknown person snapshot.
        """
        pb = person_det["bbox"]
        person_confidence = person_det.get("confidence", 0.0)

        # Skip very low confidence person detections
        if person_confidence < 0.30:
            return

        h, w = clean_frame.shape[:2]

        # Extract the upper body region (head + shoulders)
        head_region = self.detector.estimate_head_region(pb, clean_frame.shape)
        head_top, head_right, head_bottom, head_left = head_region

        # Ensure valid crop dimensions
        if head_bottom <= head_top or head_right <= head_left:
            return

        head_crop = clean_frame[head_top:head_bottom, head_left:head_right]
        if head_crop.size == 0:
            return

        # Attempt face detection on the focused crop at full resolution
        crop_face_locs, crop_rgb = self.detector.detect_faces_in_crop(head_crop)

        if crop_face_locs:
            # Face found in the crop! Process each face
            for crop_loc in crop_face_locs:
                c_top, c_right, c_bottom, c_left = crop_loc
                # Convert crop-local coordinates back to frame-global
                global_loc = (
                    c_top + head_top,
                    c_right + head_left,
                    c_bottom + head_top,
                    c_left + head_left,
                )

                # Generate encoding from the crop (better quality than downscaled)
                encoding = self.recognizer.generate_encoding(crop_rgb, crop_loc)
                if encoding is None:
                    continue

                centroid = ((global_loc[3] + global_loc[1]) // 2,
                            (global_loc[0] + global_loc[2]) // 2)

                # Try to recognize
                result = self.recognizer.recognize_face(encoding)
                if result.matched:
                    self._draw_box(frame, global_loc, result.person_name, (0, 255, 0), result.confidence)
                    self.tracker.on_detection(
                        person_id=result.person_id,
                        person_name=result.person_name,
                        camera_id=self.camera_id,
                        confidence=result.confidence,
                        centroid=centroid,
                    )
                else:
                    self._draw_box(frame, global_loc, "UNKNOWN", (0, 0, 255), 0.0)
                    rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
                    self._handle_unknown_face(clean_frame, rgb_frame, global_loc, encoding, centroid)

            logger.info(
                f"[{self.camera_id}] YOLO-assisted: found {len(crop_face_locs)} face(s) "
                f"in missed person body (conf={person_confidence:.0%})"
            )
        else:
            # No face detected even in the focused crop — create body-only unknown
            # This person is visible but their face cannot be detected
            # (e.g., facing away, heavy occlusion, very small)
            self._handle_unknown_person_body(frame, clean_frame, person_det)

    def _handle_unknown_person_body(
        self,
        frame: np.ndarray,
        clean_frame: np.ndarray,
        person_det: dict,
    ):
        """
        Handle a person detected by YOLO whose face could NOT be detected.
        Creates a body-only unknown person snapshot and tracks them.

        Uses centroid-based dedup to avoid spamming unknowns for the
        same body detection across frames.
        """
        pb = person_det["bbox"]
        person_confidence = person_det.get("confidence", 0.0)

        # Compute person centroid
        pcx = (pb["x1"] + pb["x2"]) // 2
        pcy = (pb["y1"] + pb["y2"]) // 2
        centroid = (pcx, pcy)

        # Dedup: check if we already have a body-only unknown near this centroid
        DEDUP_DISTANCE = 100  # pixels
        similar_body_id = None
        now = time.time()
        for uid, udata in self._yolo_body_unknown_cache.items():
            dx = abs(udata["centroid"][0] - pcx)
            dy = abs(udata["centroid"][1] - pcy)
            if dx < DEDUP_DISTANCE and dy < DEDUP_DISTANCE:
                similar_body_id = uid
                break

        if similar_body_id:
            # Update existing body unknown
            self._yolo_body_unknown_cache[similar_body_id]["count"] += 1
            self._yolo_body_unknown_cache[similar_body_id]["last_seen"] = now
            self._yolo_body_unknown_cache[similar_body_id]["centroid"] = centroid

            # Keep tracking them
            unknown_label = self._yolo_body_unknown_cache[similar_body_id].get("label", "Unknown Person")
            self.tracker.on_detection(
                person_id=similar_body_id,
                person_name=unknown_label,
                camera_id=self.camera_id,
                confidence=0.0,
                centroid=centroid,
            )
        else:
            # New body-only unknown person
            unknown_id = str(uuid4())
            unknown_number = len(self._unknown_cache) + len(self._yolo_body_unknown_cache) + 1
            unknown_label = f"Unknown #{unknown_number}"

            # Crop the person body from the clean frame
            h, w = clean_frame.shape[:2]
            body_crop = clean_frame[
                max(0, pb["y1"]):min(h, pb["y2"]),
                max(0, pb["x1"]):min(w, pb["x2"]),
            ].copy()

            # Also extract head region for a better snapshot
            head_region = self.detector.estimate_head_region(pb, clean_frame.shape)
            head_top, head_right, head_bottom, head_left = head_region
            head_crop = clean_frame[head_top:head_bottom, head_left:head_right].copy()

            # Save snapshots
            _, body_url = save_snapshot(
                body_crop, f"unknown_body_{unknown_id}", subdirectory="unknowns", quality=95
            )
            _, head_url = save_snapshot(
                head_crop, f"unknown_head_{unknown_id}", subdirectory="unknowns", quality=95
            )
            _, full_frame_url = save_snapshot(
                clean_frame, f"fullframe_{unknown_id}", subdirectory="frames", quality=90
            )

            self._yolo_body_unknown_cache[unknown_id] = {
                "centroid": centroid,
                "count": 1,
                "first_seen": now,
                "last_seen": now,
                "snapshot_path": body_url,
                "head_path": head_url,
                "full_frame_path": full_frame_url,
                "label": unknown_label,
                "confidence": person_confidence,
            }

            # Track this unknown person
            self.tracker.on_detection(
                person_id=unknown_id,
                person_name=unknown_label,
                camera_id=self.camera_id,
                confidence=0.0,
                centroid=centroid,
            )

            # Draw a box around the body
            body_loc = (pb["y1"], pb["x2"], pb["y2"], pb["x1"])
            self._draw_box(frame, body_loc, f"UNKNOWN (body)", (0, 128, 255), person_confidence)

            logger.info(
                f"[{self.camera_id}] Body-only unknown person detected: "
                f"{unknown_label} (YOLO conf={person_confidence:.0%})"
            )

            # Emit unknown face event (using body snapshot)
            if self.on_unknown:
                self.on_unknown({
                    "unknown_id": unknown_id,
                    "camera_id": self.camera_id,
                    "snapshot_path": body_url,
                    "context_path": head_url,
                    "full_frame_path": full_frame_url,
                    "encoding": None,  # No face encoding available
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "bounding_box": {
                        "top": pb["y1"], "right": pb["x2"],
                        "bottom": pb["y2"], "left": pb["x1"],
                    },
                    "detection_type": "body_only",
                })

        # Clean up old body unknowns (older than 60 seconds)
        stale_ids = [
            uid for uid, udata in self._yolo_body_unknown_cache.items()
            if now - udata["last_seen"] > 60
        ]
        for uid in stale_ids:
            del self._yolo_body_unknown_cache[uid]

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

        # ── YOLO-ASSISTED DETECTION (for HYBRID mode) ──────────
        # Reuse TrafficMonitor's cached person detections (zero inference cost).
        # Throttled to every N seconds to avoid expensive crop-face-detection.
        now_assist = time.time()
        if now_assist - self._last_yolo_assist_check >= self._yolo_assist_interval:
            self._last_yolo_assist_check = now_assist
            try:
                yolo_persons = self._get_yolo_person_detections_from_traffic()
                if yolo_persons:
                    missed = self._find_missed_persons(yolo_persons, face_locations)
                    for person_det in missed:
                        self._attempt_yolo_assisted_detection(
                            frame, clean_frame, person_det
                        )
            except Exception as e:
                logger.debug(f"YOLO-assisted detection (hybrid) error: {e}")

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