"""
Face Detection Module — SCRFD via InsightFace.

Detects faces using the SCRFD model from insightface, which provides:
- High-accuracy face bounding boxes
- 5-point facial landmarks (for alignment before ArcFace embedding)
- GPU acceleration via ONNX Runtime

Replaces the old dlib/face_recognition-based detector.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded insightface app (shared across instances for VRAM efficiency)
_insightface_app = None
_insightface_lock = None
_insightface_failed = False


def _get_insightface_app():
    """
    Lazy-load the insightface FaceAnalysis app.

    The app bundles both SCRFD (detection) and ArcFace (recognition),
    but we only use the detection model here. The recognition model
    is used separately in recognizer.py.

    We keep a module-level singleton to avoid loading the model
    multiple times (each load consumes GPU VRAM).
    """
    global _insightface_app, _insightface_lock, _insightface_failed
    import threading

    if _insightface_lock is None:
        _insightface_lock = threading.Lock()

    with _insightface_lock:
        if _insightface_app is not None:
            return _insightface_app

        if _insightface_failed:
            return None

        try:
            import insightface
            from insightface.app import FaceAnalysis

            model_name = settings.INSIGHTFACE_MODEL_NAME
            det_size = settings.INSIGHTFACE_DET_SIZE

            logger.info(f"Loading insightface model: {model_name} (det_size={det_size})")

            app = FaceAnalysis(
                name=model_name,
            )
            app.prepare(
                ctx_id=0,  # GPU 0
                det_size=(det_size, det_size),
                det_thresh=settings.INSIGHTFACE_DET_THRESH,
            )

            _insightface_app = app
            logger.info(
                f"InsightFace loaded successfully: {model_name}, "
                f"det_size={det_size}, thresh={settings.INSIGHTFACE_DET_THRESH}"
            )
            return _insightface_app

        except Exception as e:
            logger.error(f"Failed to load insightface: {e}", exc_info=True)
            _insightface_failed = True
            return None


class FaceDetector:
    """
    Detects faces in video frames using SCRFD (via insightface).

    Features:
    - GPU-accelerated face detection (ONNX Runtime + CUDA)
    - 5-point facial landmark output (for face alignment)
    - Bounding box extraction
    - Face quality assessment
    - Backward-compatible (top, right, bottom, left) output format
    """

    # Minimum face size to filter out tiny false-positive detections
    # (vehicles, reflections, distant objects). 60px ensures enough detail
    # for meaningful face recognition.
    MIN_FACE_SIZE_DEFAULT = 60

    def __init__(
        self,
        min_face_size: int = None,
        scale_factor: float = None,
        **kwargs,  # Accept extra kwargs for backward compat
    ):
        self.min_face_size = min_face_size or self.MIN_FACE_SIZE_DEFAULT
        self.scale_factor = scale_factor or settings.DETECTION_SCALE
        self._frame_count = 0
        self._app = None  # Lazy-loaded

        logger.info(
            f"FaceDetector initialized (SCRFD): "
            f"model={settings.INSIGHTFACE_MODEL_NAME}, "
            f"min_face={self.min_face_size}"
        )

    def _get_app(self):
        """Get the shared insightface app (lazy-loaded on first use)."""
        if self._app is None:
            self._app = _get_insightface_app()
        return self._app

    @staticmethod
    def _is_valid_face_geometry(face) -> bool:
        """
        Validate that the 5-point facial landmarks form a plausible face.

        InsightFace's SCRFD provides 5 landmarks:
          [0] left eye, [1] right eye, [2] nose tip,
          [3] left mouth corner, [4] right mouth corner

        Vehicle headlights, reflections, and license plates can trigger
        false positives but their "landmark" positions are geometrically
        implausible as faces. This method checks:
          1. Eyes should be roughly horizontal (not vertically stacked)
          2. Nose should be between the eyes (horizontally) and below them
          3. Mouth should be below the nose
          4. The face should have a reasonable height-to-width ratio
          5. Detection confidence should be above a secondary threshold

        Returns True if the detection looks like a real face.
        """
        # Check detection score (secondary threshold above SCRFD's base)
        det_score = getattr(face, 'det_score', None)
        if det_score is not None and det_score < 0.45:
            return False

        # Get landmarks (5-point)
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) < 5:
            # No landmarks available — can't validate, allow it through
            return True

        try:
            left_eye = kps[0]    # (x, y)
            right_eye = kps[1]
            nose = kps[2]
            left_mouth = kps[3]
            right_mouth = kps[4]

            # 1. Eye separation: eyes should be spread apart horizontally
            eye_dx = abs(right_eye[0] - left_eye[0])
            eye_dy = abs(right_eye[1] - left_eye[1])
            if eye_dx < 5:
                # Eyes are nearly vertically stacked — not a face
                return False
            # Eyes should be roughly horizontal (allow up to ~40° tilt)
            if eye_dy > eye_dx * 0.85:
                return False

            # 2. Nose should be roughly between the eyes horizontally
            eye_cx = (left_eye[0] + right_eye[0]) / 2
            nose_offset = abs(nose[0] - eye_cx)
            if nose_offset > eye_dx * 1.0:
                # Nose is way outside the eye region — not a face
                return False

            # 3. Nose should be below or at eye level
            eye_cy = (left_eye[1] + right_eye[1]) / 2
            if nose[1] < eye_cy - eye_dx * 0.3:
                # Nose is significantly above the eyes — not a face
                return False

            # 4. Mouth should be below the nose
            mouth_cy = (left_mouth[1] + right_mouth[1]) / 2
            if mouth_cy < nose[1] - eye_dx * 0.2:
                # Mouth is above the nose — not a face
                return False

            # 5. Overall face proportion check
            # Face height (eyes to mouth) should be reasonable vs width (eye span)
            face_height = mouth_cy - eye_cy
            if face_height < 0:
                return False  # Mouth above eyes
            if eye_dx > 0 and face_height / eye_dx > 4.0:
                # Face is extremely tall and narrow — likely not a face
                return False

            return True

        except (IndexError, TypeError):
            # If landmark parsing fails, allow the detection through
            return True

    def detect_faces(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect faces in a frame.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            Tuple of (face_locations, rgb_frame)
            face_locations: List of (top, right, bottom, left) tuples
                            (backward-compatible format)
            rgb_frame: The RGB-converted frame
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Scale down for performance if needed
        if self.scale_factor < 1.0:
            small_frame = cv2.resize(
                rgb_frame, (0, 0),
                fx=self.scale_factor,
                fy=self.scale_factor,
            )
        else:
            small_frame = rgb_frame

        # Run SCRFD detection
        app = self._get_app()
        if app is None:
            self._frame_count += 1
            return [], rgb_frame

        faces = app.get(small_frame)

        # Extract bounding boxes and scale back
        face_locations = []
        for face in faces:
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            # Scale back to original frame size
            if self.scale_factor < 1.0:
                inv_scale = 1.0 / self.scale_factor
                x1 = int(x1 * inv_scale)
                y1 = int(y1 * inv_scale)
                x2 = int(x2 * inv_scale)
                y2 = int(y2 * inv_scale)

            # Filter out too-small faces
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue

            # Validate face landmark geometry to reject non-face detections
            # (vehicles, reflections, etc. can trigger false positives)
            if not self._is_valid_face_geometry(face):
                continue

            # Convert to (top, right, bottom, left) format for backward compat
            face_locations.append((y1, x2, y2, x1))

        self._frame_count += 1
        return face_locations, rgb_frame

    def detect_faces_with_embeddings(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List, np.ndarray]:
        """
        Detect faces AND extract ArcFace embeddings in a SINGLE pass.

        This is the most efficient method — runs insightface once on the
        full-resolution frame, returning both bounding boxes and 512-d
        embeddings. No separate crop-and-re-detect needed.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            Tuple of (face_locations, embeddings, rgb_frame)
            face_locations: List of (top, right, bottom, left)
            embeddings: List of 512-d numpy arrays (or None per face)
            rgb_frame: The RGB-converted frame
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run on FULL resolution for better embedding quality
        app = self._get_app()
        if app is None:
            self._frame_count += 1
            return [], [], rgb_frame

        faces = app.get(rgb_frame)

        face_locations = []
        embeddings = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Filter too-small faces
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue

            # Validate face landmark geometry to reject non-face detections
            if not self._is_valid_face_geometry(face):
                continue

            # Convert to (top, right, bottom, left)
            face_locations.append((y1, x2, y2, x1))

            # Extract embedding (already computed by app.get)
            if hasattr(face, 'embedding') and face.embedding is not None:
                embeddings.append(face.embedding)
            else:
                embeddings.append(None)

        self._frame_count += 1
        return face_locations, embeddings, rgb_frame

    def detect_faces_raw(
        self, frame_rgb: np.ndarray
    ) -> list:
        """
        Detect faces and return raw insightface Face objects.
        These contain bbox, landmarks, and embedding if the recognition
        model is loaded.

        Args:
            frame_rgb: RGB image (numpy array)

        Returns:
            List of insightface Face objects
        """
        app = self._get_app()
        if app is None:
            return []
        faces = app.get(frame_rgb)
        return [f for f in faces if self._is_valid_face_geometry(f)]

    def extract_face_crop(
        self,
        frame: np.ndarray,
        location: Tuple[int, int, int, int],
        padding: int = 20,
    ) -> np.ndarray:
        """
        Extract a cropped face from a frame with padding.

        Args:
            frame: Original frame (BGR)
            location: (top, right, bottom, left) bounding box
            padding: Extra pixels around the face

        Returns:
            Cropped face image (BGR)
        """
        top, right, bottom, left = location
        h, w = frame.shape[:2]

        # Add padding, clamped to frame bounds
        top = max(0, top - padding)
        right = min(w, right + padding)
        bottom = min(h, bottom + padding)
        left = max(0, left - padding)

        return frame[top:bottom, left:right].copy()

    def assess_face_quality(
        self, frame: np.ndarray, location: Tuple[int, int, int, int]
    ) -> float:
        """
        Assess face image quality (0.0 to 1.0).

        Considers:
        - Face size (larger = better)
        - Blur level (sharper = better)
        - Brightness (balanced = better)
        """
        top, right, bottom, left = location
        face_crop = frame[top:bottom, left:right]

        if face_crop.size == 0:
            return 0.0

        # Size score (normalize face area)
        face_area = (bottom - top) * (right - left)
        size_score = min(1.0, face_area / (150 * 150))

        # Blur score (Laplacian variance)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500.0)

        # Brightness score (penalize too dark or too bright)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127) / 127.0

        # Weighted combination
        quality = (size_score * 0.3) + (blur_score * 0.4) + (brightness_score * 0.3)
        return round(max(0.0, min(1.0, quality)), 3)

    def detect_faces_in_crop(
        self, crop: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect faces in a pre-cropped region (e.g. from YOLO person bbox).
        Runs at full resolution for maximum sensitivity.

        Args:
            crop: BGR image crop (numpy array)

        Returns:
            Tuple of (face_locations_in_crop_coords, rgb_crop)
        """
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return [], cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.size > 0 else crop

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Run detection on the crop
        app = self._get_app()
        if app is None:
            return [], rgb_crop
            
        faces = app.get(rgb_crop)

        face_locations = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Filter impossibly small faces
            if (y2 - y1) < 10 or (x2 - x1) < 10:
                continue

            # Convert to (top, right, bottom, left)
            face_locations.append((y1, x2, y2, x1))

        return face_locations, rgb_crop

    @staticmethod
    def estimate_head_region(
        person_bbox: dict, frame_shape: tuple
    ) -> Tuple[int, int, int, int]:
        """
        Estimate the head/upper-body region from a YOLO person bounding box.
        Returns (top, right, bottom, left) in face_recognition format.

        The head is approximately the top 35% of the person bounding box,
        centered horizontally.
        """
        x1 = person_bbox["x1"]
        y1 = person_bbox["y1"]
        x2 = person_bbox["x2"]
        y2 = person_bbox["y2"]
        h, w = frame_shape[:2]

        person_h = y2 - y1
        person_w = x2 - x1

        # Head region: top 40% of the person box, with some padding
        head_bottom = y1 + int(person_h * 0.40)
        head_top = max(0, y1 - int(person_h * 0.05))  # small pad above head
        head_left = max(0, x1 - int(person_w * 0.1))
        head_right = min(w, x2 + int(person_w * 0.1))
        head_bottom = min(h, head_bottom)

        return head_top, head_right, head_bottom, head_left

    @staticmethod
    def location_to_xyxy(location: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Convert (top, right, bottom, left) to (x1, y1, x2, y2)."""
        top, right, bottom, left = location
        return left, top, right, bottom

    @staticmethod
    def xyxy_to_location(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        """Convert (x1, y1, x2, y2) to (top, right, bottom, left)."""
        return y1, x2, y2, x1

    @property
    def frames_processed(self) -> int:
        return self._frame_count
