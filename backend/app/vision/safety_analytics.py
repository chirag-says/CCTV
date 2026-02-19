"""
Safety Analytics Module — Attribute Recognition, Crowd Detection, Loitering.

Provides three key safety/security features:
1. Attribute Recognition: Gender (via DeepFace) + Clothing Color (via HSV histograms)
2. Crowd / Gathering Detection: Euclidean proximity clustering of tracked persons
3. Loitering / Idle Detection: Centroid movement analysis over configurable time windows

All functions are designed to be called from the VisionPipeline without
blocking the main video processing thread. Heavy work (DeepFace) is
offloaded to a ThreadPoolExecutor.
"""

import cv2
import numpy as np
import time
import logging
import threading
from collections import defaultdict, deque
from typing import Optional, Dict, List, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Lazy-loaded DeepFace ──────────────────────────────────────────────────────
# We import DeepFace lazily to avoid startup cost and to gracefully handle
# environments where it isn't installed.

_deepface = None
_deepface_available = False


def _load_deepface():
    """Lazy-load DeepFace; returns True if available."""
    global _deepface, _deepface_available
    if _deepface is not None:
        return _deepface_available
    try:
        from deepface import DeepFace
        _deepface = DeepFace
        _deepface_available = True
        logger.info("DeepFace loaded successfully for attribute recognition")
    except ImportError:
        _deepface_available = False
        logger.warning(
            "DeepFace not installed — gender recognition disabled. "
            "Install with: pip install deepface"
        )
    return _deepface_available


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ATTRIBUTE RECOGNITION (Smart Search)
# ═══════════════════════════════════════════════════════════════════════════════

# Dominant HSV color ranges → human-readable names
_COLOR_RANGES: List[Tuple[str, np.ndarray, np.ndarray]] = [
    ("Red",      np.array([0,   70, 50]),  np.array([10,  255, 255])),
    ("Red",      np.array([170, 70, 50]),  np.array([180, 255, 255])),
    ("Orange",   np.array([11,  70, 50]),  np.array([25,  255, 255])),
    ("Yellow",   np.array([26,  70, 50]),  np.array([34,  255, 255])),
    ("Green",    np.array([35,  70, 50]),  np.array([85,  255, 255])),
    ("Blue",     np.array([86,  70, 50]),  np.array([130, 255, 255])),
    ("Purple",   np.array([131, 70, 50]),  np.array([160, 255, 255])),
    ("Pink",     np.array([161, 70, 50]),  np.array([169, 255, 255])),
    # Low saturation → achromatic
    ("White",    np.array([0, 0,   200]),  np.array([180, 30,  255])),
    ("Gray",     np.array([0, 0,   50]),   np.array([180, 40,  199])),
    ("Black",    np.array([0, 0,   0]),    np.array([180, 255, 49])),
]


def _dominant_clothing_color(frame: np.ndarray, location: tuple) -> str:
    """
    Estimate dominant clothing color using HSV histogram on the torso region
    (below the face bounding box).

    Args:
        frame: BGR image (full frame)
        location: (top, right, bottom, left) face bounding box

    Returns:
        Human-readable color string, e.g. "Red", "Blue", "Black"
    """
    try:
        top, right, bottom, left = location
        h, w = frame.shape[:2]
        face_h = bottom - top
        face_w = right - left

        # Torso region: start below the face, extend ~2× face height
        torso_top = min(bottom + 5, h - 1)
        torso_bottom = min(bottom + face_h * 2, h)
        torso_left = max(left - face_w // 4, 0)
        torso_right = min(right + face_w // 4, w)

        if torso_bottom <= torso_top or torso_right <= torso_left:
            return "Unknown"

        torso = frame[torso_top:torso_bottom, torso_left:torso_right]

        if torso.size == 0:
            return "Unknown"

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        best_color = "Unknown"
        best_score = 0

        for color_name, lower, upper in _COLOR_RANGES:
            mask = cv2.inRange(hsv, lower, upper)
            score = cv2.countNonZero(mask)
            if score > best_score:
                best_score = score
                best_color = color_name

        return best_color

    except Exception as e:
        logger.debug(f"Clothing color extraction failed: {e}")
        return "Unknown"


def _analyze_gender(face_crop_bgr: np.ndarray) -> str:
    """
    Use DeepFace to predict gender from a face crop.

    Returns:
        "Male", "Female", or "Unknown"
    """
    if not _load_deepface():
        return "Unknown"

    try:
        results = _deepface.analyze(
            face_crop_bgr,
            actions=["gender"],
            enforce_detection=False,
            detector_backend="skip",  # We already have the crop
            silent=True,
        )
        if results and isinstance(results, list):
            gender_data = results[0].get("gender", {})
            # Returns {"Man": 95.2, "Woman": 4.8}
            if gender_data:
                dominant = max(gender_data, key=gender_data.get)
                return "Male" if dominant == "Man" else "Female"
        return "Unknown"
    except Exception as e:
        logger.debug(f"DeepFace gender analysis failed: {e}")
        return "Unknown"


class AttributeRecognizer:
    """
    Extracts person attributes (gender, clothing color) exactly ONCE per
    tracked person to minimize performance impact.

    Uses a ThreadPoolExecutor so DeepFace analysis doesn't block the
    video pipeline.
    """

    def __init__(self, max_workers: int = 2):
        self._processed_tracks: set = set()  # track IDs already analyzed
        self._results: Dict[str, dict] = {}  # track_id → attributes
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="attr-recog",
        )
        self._lock = threading.Lock()

    def maybe_extract(
        self,
        track_id: str,
        frame: np.ndarray,
        face_location: tuple,
        on_complete: Optional[Callable] = None,
    ) -> Optional[dict]:
        """
        Submit attribute extraction for a NEW track. Skips if already done.

        Args:
            track_id: Unique track/person ID
            frame: Current BGR frame
            face_location: (top, right, bottom, left)
            on_complete: Optional callback(track_id, attributes_dict)

        Returns:
            Cached result if already available, else None (async pending).
        """
        with self._lock:
            if track_id in self._processed_tracks:
                return self._results.get(track_id)

            # Mark as in-progress immediately to prevent duplicate submissions
            self._processed_tracks.add(track_id)

        # Capture crops NOW (frame reference will change on next iteration)
        top, right, bottom, left = face_location
        h, w = frame.shape[:2]
        pad = 20
        face_crop = frame[
            max(0, top - pad): min(h, bottom + pad),
            max(0, left - pad): min(w, right + pad),
        ].copy()
        frame_copy = frame.copy()

        def _worker():
            try:
                gender = _analyze_gender(face_crop)
                color = _dominant_clothing_color(frame_copy, face_location)
                apparel = f"{color} Shirt" if color != "Unknown" else "Unknown"

                attrs = {
                    "gender": gender,
                    "apparel": apparel,
                    "clothing_color": color,
                }

                with self._lock:
                    self._results[track_id] = attrs

                logger.info(
                    f"Attributes for track {track_id[:8]}…: "
                    f"gender={gender}, apparel={apparel}"
                )

                if on_complete:
                    on_complete(track_id, attrs)

            except Exception as e:
                logger.error(f"Attribute extraction failed for {track_id}: {e}")

        self._executor.submit(_worker)
        return None

    def get_attributes(self, track_id: str) -> Optional[dict]:
        """Get cached attributes for a track (returns None if not yet ready)."""
        with self._lock:
            return self._results.get(track_id)

    def reset(self):
        """Clear all cached results."""
        with self._lock:
            self._processed_tracks.clear()
            self._results.clear()

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CROWD / GATHERING DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GatheringState:
    """Tracks a potential gathering of people."""
    person_ids: frozenset
    first_detected: float  # Unix timestamp when cluster first formed
    last_seen: float
    alerted: bool = False


class CrowdDetector:
    """
    Detects when 3+ tracked persons are within close proximity for a
    sustained duration, indicating a "gathering" or potential incident.

    Algorithm:
    - Compute pairwise Euclidean distance between all person centroids
    - Find clusters where every member is within `proximity_px` of at least
      one other member (single-linkage clustering)
    - If a cluster of `min_persons`+ persists for `sustain_seconds`, alert

    Thread-safe.
    """

    def __init__(
        self,
        proximity_px: int = 150,
        min_persons: int = 3,
        sustain_seconds: float = 5.0,
        on_alert: Optional[Callable] = None,
    ):
        self.proximity_px = proximity_px
        self.min_persons = min_persons
        self.sustain_seconds = sustain_seconds
        self.on_alert = on_alert

        self._gatherings: Dict[frozenset, GatheringState] = {}
        self._lock = threading.Lock()

    def update(
        self,
        person_centroids: Dict[str, Tuple[int, int]],
        camera_id: str,
    ):
        """
        Update crowd detection with current person positions.

        Args:
            person_centroids: {person_id: (cx, cy)} for all tracked persons
            camera_id: Camera identifier
        """
        now = time.time()

        if len(person_centroids) < self.min_persons:
            # Not enough people — expire old gatherings
            with self._lock:
                self._expire_gatherings(now)
            return

        # Build adjacency based on proximity
        ids = list(person_centroids.keys())
        n = len(ids)
        adjacency: Dict[str, set] = {pid: set() for pid in ids}

        for i in range(n):
            for j in range(i + 1, n):
                cx1, cy1 = person_centroids[ids[i]]
                cx2, cy2 = person_centroids[ids[j]]
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                if dist <= self.proximity_px:
                    adjacency[ids[i]].add(ids[j])
                    adjacency[ids[j]].add(ids[i])

        # Find connected components (single-linkage clusters)
        visited = set()
        clusters: List[set] = []

        for pid in ids:
            if pid in visited:
                continue
            cluster = set()
            stack = [pid]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.add(current)
                stack.extend(adjacency[current] - visited)
            if len(cluster) >= self.min_persons:
                clusters.append(cluster)

        with self._lock:
            # Update or create gathering states
            active_keys = set()
            for cluster in clusters:
                key = frozenset(cluster)
                active_keys.add(key)

                if key in self._gatherings:
                    self._gatherings[key].last_seen = now
                else:
                    # Check if this is a superset/subset of existing gathering
                    matched = False
                    for existing_key in list(self._gatherings.keys()):
                        overlap = len(key & existing_key) / max(len(key | existing_key), 1)
                        if overlap > 0.6:
                            # Merge: treat as same gathering
                            old = self._gatherings.pop(existing_key)
                            self._gatherings[key] = GatheringState(
                                person_ids=key,
                                first_detected=old.first_detected,
                                last_seen=now,
                                alerted=old.alerted,
                            )
                            active_keys.discard(existing_key)
                            matched = True
                            break

                    if not matched:
                        self._gatherings[key] = GatheringState(
                            person_ids=key,
                            first_detected=now,
                            last_seen=now,
                        )

                # Check if sustained long enough
                gathering = self._gatherings[key]
                duration = now - gathering.first_detected
                if duration >= self.sustain_seconds and not gathering.alerted:
                    gathering.alerted = True
                    alert = {
                        "event_type": "security_alert",
                        "subtype": "gathering",
                        "camera_id": camera_id,
                        "person_count": len(key),
                        "person_ids": list(key),
                        "duration_sec": round(duration, 1),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metadata": {
                            "proximity_threshold_px": self.proximity_px,
                            "sustain_threshold_sec": self.sustain_seconds,
                        },
                    }
                    logger.warning(
                        f"🚨 GATHERING ALERT: {len(key)} persons clustered "
                        f"for {duration:.1f}s on camera {camera_id}"
                    )
                    if self.on_alert:
                        self.on_alert(alert)

            self._expire_gatherings(now, active_keys)

    def _expire_gatherings(
        self, now: float, active_keys: set = None
    ):
        """Remove gatherings that haven't been seen recently."""
        expired = []
        for key, g in self._gatherings.items():
            if active_keys and key in active_keys:
                continue
            if now - g.last_seen > 5.0:  # 5s grace period
                expired.append(key)
        for key in expired:
            del self._gatherings[key]

    def reset(self):
        with self._lock:
            self._gatherings.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. LOITERING / IDLE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoiteringTrack:
    """Centroid history for a single person (for loitering analysis)."""
    person_id: str
    person_name: str
    # Deque of (timestamp, cx, cy) — keeps last N minutes
    history: deque = field(default_factory=lambda: deque(maxlen=600))
    alerted: bool = False
    last_alert_time: float = 0.0


class LoiteringDetector:
    """
    Identifies persons who remain in roughly the same location for
    an extended period, indicating potential loitering.

    Algorithm:
    - Maintain centroid history per track_id
    - Over a configurable time window, compute max displacement
    - If max displacement < threshold, trigger loitering alert
    - Cooldown prevents repeated alerts for the same person

    Thread-safe.
    """

    def __init__(
        self,
        movement_threshold_px: int = 50,
        time_window_sec: float = 300.0,  # 5 minutes
        alert_cooldown_sec: float = 600.0,  # 10 min between re-alerts
        on_alert: Optional[Callable] = None,
    ):
        self.movement_threshold_px = movement_threshold_px
        self.time_window_sec = time_window_sec
        self.alert_cooldown_sec = alert_cooldown_sec
        self.on_alert = on_alert

        self._tracks: Dict[str, LoiteringTrack] = {}
        self._lock = threading.Lock()

    def update(
        self,
        person_centroids: Dict[str, Tuple[int, int]],
        person_names: Dict[str, str],
        camera_id: str,
    ):
        """
        Feed current positions for all tracked persons.

        Args:
            person_centroids: {person_id: (cx, cy)}
            person_names: {person_id: display_name}
            camera_id: source camera
        """
        now = time.time()

        with self._lock:
            for pid, (cx, cy) in person_centroids.items():
                if pid not in self._tracks:
                    self._tracks[pid] = LoiteringTrack(
                        person_id=pid,
                        person_name=person_names.get(pid, "Unknown"),
                    )
                track = self._tracks[pid]
                track.history.append((now, cx, cy))

                # Check loitering
                self._check_loitering(track, camera_id, now)

            # Prune tracks not seen recently
            stale = [
                pid for pid, t in self._tracks.items()
                if t.history and (now - t.history[-1][0]) > 60
            ]
            for pid in stale:
                del self._tracks[pid]

    def _check_loitering(
        self, track: LoiteringTrack, camera_id: str, now: float
    ):
        """Analyze centroid history for a single person."""
        if not track.history:
            return

        # Filter history to the time window
        cutoff = now - self.time_window_sec
        recent = [(t, cx, cy) for t, cx, cy in track.history if t >= cutoff]

        if len(recent) < 10:
            # Not enough data points yet
            return

        # Check that the time span actually covers enough of the window
        time_span = recent[-1][0] - recent[0][0]
        if time_span < self.time_window_sec * 0.8:
            # Hasn't been tracked long enough yet
            return

        # Compute max displacement from the first recorded position
        base_cx, base_cy = recent[0][1], recent[0][2]
        max_disp = 0.0
        for _, cx, cy in recent:
            disp = np.sqrt((cx - base_cx) ** 2 + (cy - base_cy) ** 2)
            max_disp = max(max_disp, disp)

        if max_disp < self.movement_threshold_px:
            # Loitering detected — check cooldown
            if track.alerted and (now - track.last_alert_time) < self.alert_cooldown_sec:
                return  # Still in cooldown

            track.alerted = True
            track.last_alert_time = now

            alert = {
                "event_type": "security_alert",
                "subtype": "loitering",
                "camera_id": camera_id,
                "person_id": track.person_id,
                "person_name": track.person_name,
                "duration_sec": round(time_span, 1),
                "max_displacement_px": round(max_disp, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "movement_threshold_px": self.movement_threshold_px,
                    "time_window_sec": self.time_window_sec,
                },
            }
            logger.warning(
                f"🚨 LOITERING ALERT: {track.person_name} stationary for "
                f"{time_span:.0f}s (max displacement: {max_disp:.0f}px) "
                f"on camera {camera_id}"
            )
            if self.on_alert:
                self.on_alert(alert)

    def reset(self):
        with self._lock:
            self._tracks.clear()
