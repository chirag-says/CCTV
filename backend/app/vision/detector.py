"""
Face Detection Module.

Handles face detection from video frames using face_recognition library.
Supports HOG (CPU-optimized) and CNN (GPU-optimized) models.
"""

import cv2
import numpy as np
import face_recognition
import logging
from typing import List, Tuple, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects faces in video frames.
    
    Features:
    - Configurable detection model (HOG/CNN)
    - Frame downscaling for performance
    - Bounding box extraction
    - Face quality assessment
    """

    def __init__(
        self,
        model: str = None,
        scale_factor: float = None,
        min_face_size: int = 30,
    ):
        self.model = model or settings.DETECTION_MODEL
        self.scale_factor = scale_factor or settings.DETECTION_SCALE
        self.min_face_size = min_face_size
        self._frame_count = 0
        
        logger.info(f"FaceDetector initialized: model={self.model}, scale={self.scale_factor}")

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
            rgb_frame: The RGB-converted, possibly scaled frame
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Scale down for faster detection
        if self.scale_factor < 1.0:
            small_frame = cv2.resize(
                rgb_frame, (0, 0),
                fx=self.scale_factor,
                fy=self.scale_factor
            )
        else:
            small_frame = rgb_frame

        # Detect face locations with upsampling for better small face detection
        upsample = 2 if self.scale_factor <= 1.0 else 1
        face_locations_raw = face_recognition.face_locations(
            small_frame, model=self.model, number_of_times_to_upsample=upsample
        )

        # Scale locations back to original size
        if self.scale_factor < 1.0 and face_locations_raw:
            inv_scale = 1.0 / self.scale_factor
            face_locations_raw = [
                (
                    int(top * inv_scale),
                    int(right * inv_scale),
                    int(bottom * inv_scale),
                    int(left * inv_scale),
                )
                for top, right, bottom, left in face_locations_raw
            ]

        # Filter out too-small faces
        # Format: (top, right, bottom, left)
        # Height = bottom - top, Width = right - left
        face_locations = [
            loc for loc in face_locations_raw
            if (loc[2] - loc[0]) >= self.min_face_size
            and (loc[1] - loc[3]) >= self.min_face_size
        ]

        self._frame_count += 1
        
        # Debug logging (uncomment for debugging)
        # if self._frame_count % 100 == 0:
        #     logger.debug(
        #         f"Detector stats: frame={self._frame_count}, "
        #         f"raw={len(face_locations_raw) if 'face_locations_raw' in dir() else 0}, "
        #         f"final={len(face_locations)}"
        #     )

        return face_locations, rgb_frame

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

    @property
    def frames_processed(self) -> int:
        return self._frame_count
