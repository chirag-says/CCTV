"""
Face Quality Gate — Evaluates face quality before enrollment/storage.

Rejects faces that are blurry, too dark/bright, too small, at extreme angles,
or have low detection confidence. Only high-quality embeddings should be stored.

Uses InsightFace's 5-point landmarks for pose estimation (yaw/pitch/roll).
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of a face quality assessment."""
    passed: bool
    quality_score: float  # 0.0–1.0 composite score
    rejection_reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class FaceQualityGate:
    """
    Evaluates face quality before embedding storage.

    Checks:
    - Blur (Laplacian variance)
    - Brightness (mean pixel value)
    - Face size (minimum px)
    - Detection confidence
    - Head yaw (horizontal rotation)
    - Head pitch (vertical tilt)
    """

    def __init__(
        self,
        min_blur: float = None,
        min_brightness: float = None,
        max_brightness: float = None,
        min_size: int = None,
        min_det_score: float = None,
        max_yaw: float = None,
        max_pitch: float = None,
    ):
        self.min_blur = min_blur or settings.FACE_QUALITY_MIN_BLUR
        self.min_brightness = min_brightness or settings.FACE_QUALITY_MIN_BRIGHTNESS
        self.max_brightness = max_brightness or settings.FACE_QUALITY_MAX_BRIGHTNESS
        self.min_size = min_size or settings.FACE_QUALITY_MIN_SIZE
        self.min_det_score = min_det_score or settings.FACE_QUALITY_MIN_DET_SCORE
        self.max_yaw = max_yaw or settings.FACE_QUALITY_MAX_YAW
        self.max_pitch = max_pitch or settings.FACE_QUALITY_MAX_PITCH

    def assess(
        self,
        face_crop: np.ndarray,
        face_location: Tuple[int, int, int, int],
        det_score: float = 1.0,
        landmarks: Optional[np.ndarray] = None,
    ) -> QualityResult:
        """
        Assess face quality for enrollment.

        Args:
            face_crop: BGR face image crop
            face_location: (top, right, bottom, left) bounding box
            det_score: SCRFD detection confidence
            landmarks: InsightFace 5-point landmarks (optional, for pose)

        Returns:
            QualityResult with pass/fail, score, and metrics
        """
        reasons = []
        metrics = {}

        if face_crop is None or face_crop.size == 0:
            return QualityResult(
                passed=False,
                quality_score=0.0,
                rejection_reasons=["Empty face crop"],
                metrics={},
            )

        top, right, bottom, left = face_location
        face_w = right - left
        face_h = bottom - top

        # ── 1. Face size ─────────────────────────────────────────
        metrics["face_width"] = float(face_w)
        metrics["face_height"] = float(face_h)
        if face_w < self.min_size or face_h < self.min_size:
            reasons.append(
                f"Face too small ({face_w}x{face_h}px, min={self.min_size}px)"
            )

        # ── 2. Detection confidence ──────────────────────────────
        metrics["det_score"] = float(det_score)
        if det_score < self.min_det_score:
            reasons.append(
                f"Low detection confidence ({det_score:.2f}, min={self.min_det_score})"
            )

        # ── 3. Blur score (Laplacian variance) ───────────────────
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        metrics["blur_score"] = blur_score
        if blur_score < self.min_blur:
            reasons.append(
                f"Face too blurry (blur={blur_score:.1f}, min={self.min_blur})"
            )

        # ── 4. Brightness ────────────────────────────────────────
        brightness = float(np.mean(gray))
        metrics["brightness"] = brightness
        if brightness < self.min_brightness:
            reasons.append(
                f"Face too dark (brightness={brightness:.0f}, min={self.min_brightness})"
            )
        elif brightness > self.max_brightness:
            reasons.append(
                f"Face too bright (brightness={brightness:.0f}, max={self.max_brightness})"
            )

        # ── 5. Head pose (yaw/pitch/roll) ────────────────────────
        yaw, pitch, roll = self.estimate_pose(landmarks, face_w)
        metrics["yaw"] = yaw
        metrics["pitch"] = pitch
        metrics["roll"] = roll

        if abs(yaw) > self.max_yaw:
            reasons.append(
                f"Head turned too far (yaw={yaw:.0f}°, max=±{self.max_yaw}°)"
            )
        if abs(pitch) > self.max_pitch:
            reasons.append(
                f"Head tilted too far (pitch={pitch:.0f}°, max=±{self.max_pitch}°)"
            )

        # ── 6. Lighting condition ────────────────────────────────
        lighting = self._classify_lighting(gray)
        metrics["lighting"] = lighting

        # ── Composite quality score ──────────────────────────────
        # Weighted combination of individual quality factors
        size_score = min(1.0, (face_w * face_h) / (200 * 200))
        det_score_norm = min(1.0, det_score)
        blur_score_norm = min(1.0, blur_score / 500.0)
        brightness_score = 1.0 - abs(brightness - 127) / 127.0
        pose_score = 1.0 - (abs(yaw) / 90.0 + abs(pitch) / 90.0) / 2.0

        quality_score = (
            size_score * 0.15
            + det_score_norm * 0.15
            + blur_score_norm * 0.30
            + brightness_score * 0.15
            + max(0.0, pose_score) * 0.25
        )
        quality_score = round(max(0.0, min(1.0, quality_score)), 3)

        return QualityResult(
            passed=len(reasons) == 0,
            quality_score=quality_score,
            rejection_reasons=reasons,
            metrics=metrics,
        )

    @staticmethod
    def estimate_pose(
        landmarks: Optional[np.ndarray],
        face_width: int = 100,
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (yaw, pitch, roll) from InsightFace 5-point landmarks.

        InsightFace provides 5 landmarks:
          [0] left eye, [1] right eye, [2] nose tip,
          [3] left mouth corner, [4] right mouth corner

        Uses geometric relationships between landmarks to estimate angles.
        Accuracy is approximately ±10° — sufficient for quality gating.

        Returns:
            (yaw, pitch, roll) in degrees
        """
        if landmarks is None or len(landmarks) < 5:
            return 0.0, 0.0, 0.0

        try:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]

            # ── Yaw (horizontal rotation) ────────────────────────
            # Asymmetry of nose position relative to eye midpoint
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            eye_span = abs(right_eye[0] - left_eye[0])
            if eye_span > 0:
                nose_offset = (nose[0] - eye_center_x) / eye_span
                # Scale: offset of 0.5 ≈ 45°
                yaw = nose_offset * 90.0
            else:
                yaw = 0.0

            # ── Pitch (vertical tilt) ────────────────────────────
            # Ratio of eye-nose distance to nose-mouth distance
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            eye_nose_dist = nose[1] - eye_center_y
            nose_mouth_dist = mouth_center_y - nose[1]

            if nose_mouth_dist > 0:
                ratio = eye_nose_dist / nose_mouth_dist
                # Neutral ratio is ~1.0. Higher = looking down, lower = looking up
                pitch = (ratio - 1.0) * 30.0
            else:
                pitch = 0.0

            # ── Roll (head tilt) ─────────────────────────────────
            # Angle of the line connecting the two eyes
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            roll = float(np.degrees(np.arctan2(dy, dx)))

            return (
                round(float(np.clip(yaw, -90, 90)), 1),
                round(float(np.clip(pitch, -90, 90)), 1),
                round(float(np.clip(roll, -90, 90)), 1),
            )

        except (IndexError, TypeError, ZeroDivisionError):
            return 0.0, 0.0, 0.0

    @staticmethod
    def _classify_lighting(gray: np.ndarray) -> str:
        """Classify lighting condition from grayscale face crop."""
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        if mean_val < 60:
            return "dim"
        elif mean_val > 200:
            return "bright"
        elif std_val > 70:
            # High contrast + medium brightness often indicates backlighting
            return "backlit"
        else:
            return "normal"

    @staticmethod
    def estimate_distance(face_width_px: int, frame_width: int = 1920) -> float:
        """
        Rough distance estimate from face width in pixels.
        Assumes average face width ~16cm and typical CCTV focal length.

        Returns estimated distance in meters (very approximate).
        """
        if face_width_px <= 0:
            return 0.0
        # Empirical: 200px face ≈ 1m at 1080p
        reference_px = 200.0 * (frame_width / 1920.0)
        return round(reference_px / face_width_px, 2)
