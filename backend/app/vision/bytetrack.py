"""
ByteTrack Multi-Object Tracker for CCTV Face Tracking.

Implements a simplified ByteTrack algorithm for frame-to-frame face tracking.
This tracks face detections across video frames using IoU-based association
and linear motion prediction (Kalman filter).

Key benefits:
- Maintains face track IDs across frames (even when detection briefly fails)
- Associates low-confidence detections with existing tracks
- Reduces the need to run expensive ArcFace recognition every frame
- Provides smooth bounding box interpolation

Reference: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
           (Zhang et al., ECCV 2022)
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import OrderedDict

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class STrack:
    """
    Single object track maintained by ByteTrack.

    Stores the track's bounding box, velocity, and identity information.
    """
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    person_id: Optional[str] = None  # Linked identity from ArcFace
    person_name: Optional[str] = None
    is_activated: bool = False
    frame_id: int = 0
    start_frame: int = 0
    tracklet_len: int = 0
    _velocity: Optional[np.ndarray] = field(default=None, repr=False)
    _prev_bbox: Optional[np.ndarray] = field(default=None, repr=False)

    def predict(self):
        """Predict the next position using simple linear motion."""
        if self._velocity is not None:
            self.bbox = self.bbox + self._velocity

    def update(self, new_bbox: np.ndarray, new_score: float, frame_id: int):
        """Update the track with a new detection."""
        old_bbox = self.bbox.copy()
        self.bbox = new_bbox
        self.score = new_score
        self.frame_id = frame_id
        self.tracklet_len += 1

        # Update velocity (simple exponential smoothing)
        new_velocity = new_bbox - old_bbox
        if self._velocity is not None:
            self._velocity = 0.7 * new_velocity + 0.3 * self._velocity
        else:
            self._velocity = new_velocity

    def activate(self, frame_id: int, track_id: int):
        """Activate a new track."""
        self.track_id = track_id
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @property
    def centroid(self) -> Tuple[int, int]:
        cx = int((self.bbox[0] + self.bbox[2]) / 2)
        cy = int((self.bbox[1] + self.bbox[3]) / 2)
        return cx, cy

    def to_location(self) -> Tuple[int, int, int, int]:
        """Convert bbox to (top, right, bottom, left) format."""
        x1, y1, x2, y2 = self.bbox.astype(int)
        return y1, x2, y2, x1


def _iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bboxes1: (N, 4) array of [x1, y1, x2, y2]
        bboxes2: (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))

    x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0:1].T)
    y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1:2].T)
    x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2:3].T)
    y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    iou = np.where(union > 0, inter / union, 0)

    return iou


def _linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """
    Solve the linear assignment problem using the lap library.

    Args:
        cost_matrix: (N, M) cost matrix (lower is better)
        thresh: Maximum cost threshold for valid assignments

    Returns:
        matches: list of (row, col) matched pairs
        unmatched_a: list of unmatched row indices
        unmatched_b: list of unmatched col indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

        matches = []
        unmatched_a = []
        unmatched_b = list(range(cost_matrix.shape[1]))

        for i, j in enumerate(x):
            if j >= 0:
                matches.append((i, j))
                if j in unmatched_b:
                    unmatched_b.remove(j)
            else:
                unmatched_a.append(i)

        return matches, unmatched_a, unmatched_b

    except ImportError:
        logger.warning("lap library not available, using scipy fallback")
        from scipy.optimize import linear_sum_assignment

        # Use scipy fallback
        if min(cost_matrix.shape) == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
                unmatched_a.remove(r)
                unmatched_b.remove(c)

        return matches, unmatched_a, unmatched_b


class ByteTracker:
    """
    ByteTrack multi-object tracker adapted for face tracking.

    Algorithm:
    1. Split detections into high and low confidence groups
    2. Match high-confidence detections to existing tracks using IoU
    3. Match remaining tracks to low-confidence detections
    4. Create new tracks from unmatched high-confidence detections
    5. Remove tracks that have been lost for too long

    This is adapted from the original ByteTrack paper but simplified
    for the CCTV face tracking use case.
    """

    def __init__(
        self,
        track_thresh: float = None,
        track_buffer: int = None,
        match_thresh: float = None,
    ):
        self.track_thresh = track_thresh or settings.BYTETRACK_TRACK_THRESH
        self.track_buffer = track_buffer or settings.BYTETRACK_TRACK_BUFFER
        self.match_thresh = match_thresh or settings.BYTETRACK_MATCH_THRESH

        # Track management
        self._tracked_stracks: List[STrack] = []  # Active tracks
        self._lost_stracks: List[STrack] = []  # Recently lost tracks
        self._removed_stracks: List[STrack] = []  # Removed tracks

        self._frame_id = 0
        self._next_id = 1

        logger.info(
            f"ByteTracker initialized: thresh={self.track_thresh}, "
            f"buffer={self.track_buffer}, match={self.match_thresh}"
        )

    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
    ) -> List[STrack]:
        """
        Update tracker with new detections.

        Args:
            bboxes: (N, 4) array of [x1, y1, x2, y2] face bounding boxes
            scores: (N,) array of detection confidence scores

        Returns:
            List of active STrack objects (with track IDs)
        """
        self._frame_id += 1

        if len(bboxes) == 0:
            # No detections — predict existing tracks forward
            for track in self._tracked_stracks:
                track.predict()

            # Move timed-out tracked stracks to lost
            new_tracked = []
            for track in self._tracked_stracks:
                if self._frame_id - track.frame_id <= 2:
                    new_tracked.append(track)
                else:
                    self._lost_stracks.append(track)

            self._tracked_stracks = new_tracked

            # Remove very old lost stracks
            self._lost_stracks = [
                t for t in self._lost_stracks
                if self._frame_id - t.frame_id <= self.track_buffer
            ]

            return [t for t in self._tracked_stracks if t.is_activated]

        # ── Step 1: Split detections by confidence ──────────────
        high_mask = scores >= self.track_thresh
        low_mask = ~high_mask & (scores >= 0.1)  # Very low scores are ignored

        high_bboxes = bboxes[high_mask]
        high_scores = scores[high_mask]
        low_bboxes = bboxes[low_mask]
        low_scores = scores[low_mask]

        # ── Step 2: Predict existing tracks forward ─────────────
        all_stracks = self._tracked_stracks + self._lost_stracks
        for track in all_stracks:
            track.predict()

        # ── Step 3: Match high-confidence detections ────────────
        if len(high_bboxes) > 0 and len(all_stracks) > 0:
            track_bboxes = np.array([t.bbox for t in all_stracks])
            iou_matrix = _iou_batch(track_bboxes, high_bboxes)
            cost_matrix = 1.0 - iou_matrix

            matches, unmatched_tracks, unmatched_dets = _linear_assignment(
                cost_matrix, self.match_thresh
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(all_stracks)))
            unmatched_dets = list(range(len(high_bboxes)))

        # Apply matches
        new_tracked = []
        for track_idx, det_idx in matches:
            track = all_stracks[track_idx]
            track.update(high_bboxes[det_idx], high_scores[det_idx], self._frame_id)
            if not track.is_activated:
                track.activate(self._frame_id, self._next_id)
                self._next_id += 1
            new_tracked.append(track)

        # ── Step 4: Match remaining tracks with low-confidence ──
        remaining_tracks = [all_stracks[i] for i in unmatched_tracks]

        if len(low_bboxes) > 0 and len(remaining_tracks) > 0:
            track_bboxes = np.array([t.bbox for t in remaining_tracks])
            iou_matrix = _iou_batch(track_bboxes, low_bboxes)
            cost_matrix = 1.0 - iou_matrix

            matches2, unmatched_tracks2, _ = _linear_assignment(
                cost_matrix, 0.5  # Lower threshold for low-conf matches
            )

            for track_idx, det_idx in matches2:
                track = remaining_tracks[track_idx]
                track.update(low_bboxes[det_idx], low_scores[det_idx], self._frame_id)
                new_tracked.append(track)

            # Truly unmatched tracks
            final_unmatched = [remaining_tracks[i] for i in unmatched_tracks2]
        else:
            final_unmatched = remaining_tracks

        # ── Step 5: Handle lost tracks ──────────────────────────
        new_lost = []
        for track in final_unmatched:
            if self._frame_id - track.frame_id <= self.track_buffer:
                new_lost.append(track)
            else:
                self._removed_stracks.append(track)

        # ── Step 6: Create new tracks from unmatched detections ─
        for det_idx in unmatched_dets:
            score = high_scores[det_idx]
            if score >= self.track_thresh:
                new_track = STrack(
                    track_id=self._next_id,
                    bbox=high_bboxes[det_idx].copy(),
                    score=score,
                )
                new_track.activate(self._frame_id, self._next_id)
                self._next_id += 1
                new_tracked.append(new_track)

        # Update state
        self._tracked_stracks = new_tracked
        self._lost_stracks = new_lost

        # Cap removed stracks memory
        if len(self._removed_stracks) > 1000:
            self._removed_stracks = self._removed_stracks[-500:]

        return [t for t in self._tracked_stracks if t.is_activated]

    def associate_identity(self, track_id: int, person_id: str, person_name: str):
        """
        Associate a recognized identity with a track.
        Once a track is identified, it retains the identity across frames.
        """
        for track in self._tracked_stracks:
            if track.track_id == track_id:
                track.person_id = person_id
                track.person_name = person_name
                return

    def get_track_by_id(self, track_id: int) -> Optional[STrack]:
        """Get a track by its ID."""
        for track in self._tracked_stracks:
            if track.track_id == track_id:
                return track
        return None

    def reset(self):
        """Reset all tracking state."""
        self._tracked_stracks = []
        self._lost_stracks = []
        self._removed_stracks = []
        self._frame_id = 0
        self._next_id = 1
        logger.info("ByteTracker state reset")

    @property
    def active_track_count(self) -> int:
        return len([t for t in self._tracked_stracks if t.is_activated])

    @property
    def stats(self) -> dict:
        return {
            "active_tracks": self.active_track_count,
            "lost_tracks": len(self._lost_stracks),
            "total_ids": self._next_id - 1,
            "frame_id": self._frame_id,
        }
