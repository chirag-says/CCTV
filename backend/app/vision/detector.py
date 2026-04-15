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


def _get_insightface_app():
    """
    Lazy-load the insightface FaceAnalysis app.

    The app bundles both SCRFD (detection) and ArcFace (recognition),
    but we only use the detection model here. The recognition model
    is used separately in recognizer.py.

    We keep a module-level singleton to avoid loading the model
    multiple times (each load consumes GPU VRAM).
    """
    global _insightface_app, _insightface_lock
    import threading

    if _insightface_lock is None:
        _insightface_lock = threading.Lock()

    with _insightface_lock:
        if _insightface_app is not None:
            return _insightface_app

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
            raise RuntimeError(f"InsightFace initialization failed: {e}")


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

    def __init__(
        self,
        min_face_size: int = 20,
        scale_factor: float = None,
        **kwargs,  # Accept extra kwargs for backward compat
    ):
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor or settings.DETECTION_SCALE
        self._frame_count = 0
        self._app = None  # Lazy-loaded

        logger.info(
            f"FaceDetector initialized (SCRFD): "
            f"model={settings.INSIGHTFACE_MODEL_NAME}, "
            f"min_face={min_face_size}"
        )

    def _get_app(self):
        """Get the shared insightface app (lazy-loaded on first use)."""
        if self._app is None:
            self._app = _get_insightface_app()
        return self._app

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
        faces = app.get(frame_rgb)
        return faces

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
        faces = app.get(rgb_crop)

        face_locations = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Filter impossibly small faces
            if (y2 - y1) < 15 or (x2 - x1) < 15:
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
