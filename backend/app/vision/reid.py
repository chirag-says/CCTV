"""
Person Re-Identification (ReID) Engine.

Uses a ResNet50 backbone (torchvision, pre-trained on ImageNet) to extract
512-dim appearance descriptors from person body crops. Maintains a global
gallery across all cameras for cross-camera person matching.

Architecture:
    1. Person detected (face or YOLO body) → crop full body from frame
    2. Extract appearance feature vector (512-dim, L2-normalized)
    3. Compare against global ReID gallery using cosine similarity
    4. If match found → assign same global ReID ID
    5. If no match → create new gallery entry
    6. Log sighting with camera_id + timestamp for cross-camera trail

Thread-safe: multiple camera pipelines can call concurrently.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Feature Extractor ────────────────────────────────────────────

class ReIDFeatureExtractor:
    """
    Extracts appearance features from person crops using ResNet50.

    The final classification layer is removed, producing a 2048-dim feature
    that is projected down to 512-dim via a learned linear layer, then
    L2-normalized for cosine similarity matching.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton — one model shared across all camera pipelines."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model = None
        self._device = None
        self._transform = None
        self._model_lock = threading.Lock()
        self._loaded = False
        logger.info("ReID Feature Extractor created (lazy-load on first use)")

    def _load_model(self):
        """Lazy-load the ResNet50 model on first inference."""
        if self._loaded:
            return

        with self._model_lock:
            if self._loaded:
                return

            try:
                import torch
                import torch.nn as nn
                from torchvision import models, transforms

                # Determine device
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                else:
                    self._device = torch.device("cpu")

                # Load pre-trained ResNet50, remove classification head.
                # Use the raw 2048-dim features from avgpool directly —
                # these are already discriminative for person appearance.
                # No untrained projection layers needed.
                backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

                self._model = nn.Sequential(
                    # All ResNet layers except fc
                    nn.Sequential(*list(backbone.children())[:-1]),  # → (B, 2048, 1, 1)
                    nn.Flatten(),                                     # → (B, 2048)
                ).to(self._device)
                self._model.eval()

                # Update feature dimension to match actual output
                self._feat_dim = 2048

                # Standard ImageNet normalization
                self._transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 128)),  # Standard ReID input size
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

                self._loaded = True
                logger.info(
                    f"ReID model loaded: ResNet50 → {self._feat_dim}-dim on {self._device}"
                )

            except Exception as e:
                logger.error(f"Failed to load ReID model: {e}", exc_info=True)
                raise

    def extract(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dim L2-normalized feature from a person crop.

        Args:
            person_crop: BGR image of a person (any size, will be resized)

        Returns:
            512-dim numpy array (L2-normalized), or None on failure
        """
        self._load_model()

        try:
            import torch

            # Convert BGR → RGB
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            # Transform and add batch dimension
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                features = self._model(tensor)  # (1, 512)

            # L2-normalize
            feat = features.cpu().numpy().flatten()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            return feat

        except Exception as e:
            logger.debug(f"ReID feature extraction failed: {e}")
            return None


# ── Sighting Record ──────────────────────────────────────────────

@dataclass
class ReIDSighting:
    """A single sighting of a person on a specific camera."""
    camera_id: str
    timestamp: float
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class ReIDIdentity:
    """A person identity in the global gallery."""
    reid_id: str
    feature: np.ndarray  # 512-dim mean feature (updated with EMA)
    first_seen: float
    last_seen: float
    sightings: List[ReIDSighting] = field(default_factory=list)
    face_person_id: Optional[str] = None  # Link to face-recognition person_id if known
    face_person_name: Optional[str] = None

    @property
    def cameras_seen(self) -> List[str]:
        """Unique cameras this person has been seen on."""
        return list(dict.fromkeys(s.camera_id for s in self.sightings))

    @property
    def trail(self) -> List[dict]:
        """Chronological trail of camera sightings."""
        return [
            {
                "camera_id": s.camera_id,
                "timestamp": datetime.fromtimestamp(s.timestamp, tz=timezone.utc).isoformat(),
                "confidence": round(s.confidence, 3),
            }
            for s in sorted(self.sightings, key=lambda s: s.timestamp)
        ]


# ── Global ReID Tracker ──────────────────────────────────────────

class ReIDTracker:
    """
    Cross-camera person tracker using appearance-based Re-Identification.

    Maintains a global gallery of person appearance features. When a person
    is detected on any camera, their body crop is compared against the
    gallery. If matched, the sighting is logged under the existing ID.
    If not matched, a new gallery entry is created.

    Thread-safe for concurrent camera pipelines.
    """

    def __init__(
        self,
        match_threshold: float = 0.65,
        gallery_max_size: int = 500,
        feature_ema_alpha: float = 0.3,
        sighting_cooldown: float = 5.0,
        on_cross_camera: Optional[callable] = None,
    ):
        """
        Args:
            match_threshold: Cosine similarity threshold for matching (0-1, higher = stricter)
            gallery_max_size: Maximum gallery entries before LRU eviction
            feature_ema_alpha: EMA alpha for updating gallery features (higher = faster adaptation)
            sighting_cooldown: Minimum seconds between sightings of same person on same camera
            on_cross_camera: Callback when person is seen on a NEW camera
        """
        self.match_threshold = match_threshold
        self.gallery_max_size = gallery_max_size
        self.feature_ema_alpha = feature_ema_alpha
        self.sighting_cooldown = sighting_cooldown
        self.on_cross_camera = on_cross_camera

        self._extractor = ReIDFeatureExtractor()
        self._gallery: Dict[str, ReIDIdentity] = {}
        self._lock = threading.RLock()

        # Performance: track last sighting per (reid_id, camera_id) to avoid spam
        self._last_sighting: Dict[Tuple[str, str], float] = {}

        logger.info(
            f"ReID Tracker initialized: threshold={match_threshold}, "
            f"gallery_max={gallery_max_size}, cooldown={sighting_cooldown}s"
        )

    def process_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        camera_id: str,
        face_person_id: Optional[str] = None,
        face_person_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Process a detected person for ReID.

        Args:
            frame: Full camera frame (BGR)
            bbox: Person bounding box (x1, y1, x2, y2)
            camera_id: Camera that detected the person
            face_person_id: If face recognition identified this person, link it
            face_person_name: Display name from face recognition

        Returns:
            reid_id (str) if matched/created, None on failure
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Minimum crop size
        if (x2 - x1) < 30 or (y2 - y1) < 50:
            return None

        # Crop person
        person_crop = frame[y1:y2, x1:x2]

        # Extract feature
        feature = self._extractor.extract(person_crop)
        if feature is None:
            return None

        now = time.time()

        with self._lock:
            # If face recognition already identified this person, try to find
            # their existing gallery entry by face_person_id first
            if face_person_id:
                for reid_id, identity in self._gallery.items():
                    if identity.face_person_id == face_person_id:
                        return self._update_identity(
                            identity, feature, camera_id, bbox, now, face_person_name
                        )

            # Search gallery by appearance similarity
            best_match_id = None
            best_similarity = 0.0

            for reid_id, identity in self._gallery.items():
                sim = float(np.dot(feature, identity.feature))
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_id = reid_id

            if best_match_id and best_similarity >= self.match_threshold:
                # Match found
                identity = self._gallery[best_match_id]
                # Link face ID if we have it and identity doesn't
                if face_person_id and not identity.face_person_id:
                    identity.face_person_id = face_person_id
                    identity.face_person_name = face_person_name
                return self._update_identity(
                    identity, feature, camera_id, bbox, now, face_person_name
                )
            else:
                # No match — create new gallery entry
                return self._create_identity(
                    feature, camera_id, bbox, now, face_person_id, face_person_name
                )

    def _update_identity(
        self,
        identity: ReIDIdentity,
        feature: np.ndarray,
        camera_id: str,
        bbox: Tuple[int, int, int, int],
        timestamp: float,
        face_name: Optional[str] = None,
    ) -> str:
        """Update an existing gallery identity with a new sighting."""
        # Check cooldown
        key = (identity.reid_id, camera_id)
        if key in self._last_sighting:
            if timestamp - self._last_sighting[key] < self.sighting_cooldown:
                return identity.reid_id

        # EMA update of gallery feature
        alpha = self.feature_ema_alpha
        identity.feature = alpha * feature + (1 - alpha) * identity.feature
        # Re-normalize
        norm = np.linalg.norm(identity.feature)
        if norm > 0:
            identity.feature = identity.feature / norm

        identity.last_seen = timestamp

        # Update name if we got a face match
        if face_name and not identity.face_person_name:
            identity.face_person_name = face_name

        # Check if this is a new camera (cross-camera event!)
        previous_cameras = set(s.camera_id for s in identity.sightings)
        is_cross_camera = camera_id not in previous_cameras and len(previous_cameras) > 0

        # Add sighting
        sighting = ReIDSighting(
            camera_id=camera_id,
            timestamp=timestamp,
            confidence=float(np.dot(feature, identity.feature)),
            bbox=bbox,
        )
        identity.sightings.append(sighting)

        # Keep only last 100 sightings per identity
        if len(identity.sightings) > 100:
            identity.sightings = identity.sightings[-100:]

        self._last_sighting[key] = timestamp

        # Fire cross-camera callback
        if is_cross_camera and self.on_cross_camera:
            try:
                self.on_cross_camera({
                    "reid_id": identity.reid_id,
                    "person_name": identity.face_person_name or f"Person-{identity.reid_id[:8]}",
                    "from_camera": list(previous_cameras)[-1],
                    "to_camera": camera_id,
                    "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                    "cameras_seen": identity.cameras_seen,
                })
            except Exception as e:
                logger.debug(f"Cross-camera callback error: {e}")

        return identity.reid_id

    def _create_identity(
        self,
        feature: np.ndarray,
        camera_id: str,
        bbox: Tuple[int, int, int, int],
        timestamp: float,
        face_person_id: Optional[str] = None,
        face_person_name: Optional[str] = None,
    ) -> str:
        """Create a new gallery identity."""
        # Evict oldest if gallery is full
        if len(self._gallery) >= self.gallery_max_size:
            oldest_id = min(self._gallery, key=lambda k: self._gallery[k].last_seen)
            del self._gallery[oldest_id]
            logger.debug(f"ReID gallery full — evicted {oldest_id[:8]}")

        reid_id = str(uuid4())
        identity = ReIDIdentity(
            reid_id=reid_id,
            feature=feature.copy(),
            first_seen=timestamp,
            last_seen=timestamp,
            face_person_id=face_person_id,
            face_person_name=face_person_name,
            sightings=[
                ReIDSighting(
                    camera_id=camera_id,
                    timestamp=timestamp,
                    confidence=1.0,
                    bbox=bbox,
                )
            ],
        )
        self._gallery[reid_id] = identity

        logger.debug(
            f"ReID new identity: {reid_id[:8]} on camera {camera_id} "
            f"(name={face_person_name or 'unknown'})"
        )
        return reid_id

    def get_identity(self, reid_id: str) -> Optional[dict]:
        """Get a specific identity's info and trail."""
        with self._lock:
            identity = self._gallery.get(reid_id)
            if not identity:
                return None
            return self._identity_to_dict(identity)

    def get_all_active(self, since_seconds: float = 300) -> List[dict]:
        """Get all identities seen in the last N seconds."""
        cutoff = time.time() - since_seconds
        with self._lock:
            return [
                self._identity_to_dict(identity)
                for identity in self._gallery.values()
                if identity.last_seen >= cutoff
            ]

    def get_cross_camera_persons(self) -> List[dict]:
        """Get only persons who have been seen on 2+ cameras."""
        with self._lock:
            return [
                self._identity_to_dict(identity)
                for identity in self._gallery.values()
                if len(identity.cameras_seen) >= 2
            ]

    def _identity_to_dict(self, identity: ReIDIdentity) -> dict:
        return {
            "reid_id": identity.reid_id,
            "person_name": identity.face_person_name or f"Person-{identity.reid_id[:8]}",
            "face_person_id": identity.face_person_id,
            "cameras_seen": identity.cameras_seen,
            "camera_count": len(identity.cameras_seen),
            "first_seen": datetime.fromtimestamp(identity.first_seen, tz=timezone.utc).isoformat(),
            "last_seen": datetime.fromtimestamp(identity.last_seen, tz=timezone.utc).isoformat(),
            "sighting_count": len(identity.sightings),
            "trail": identity.trail[-20:],  # Last 20 sightings
        }

    def link_face_identity(self, reid_id: str, person_id: str, person_name: str):
        """Link a ReID gallery entry to a known face-recognition person."""
        with self._lock:
            identity = self._gallery.get(reid_id)
            if identity:
                identity.face_person_id = person_id
                identity.face_person_name = person_name

    @property
    def stats(self) -> dict:
        with self._lock:
            cross_camera = sum(1 for i in self._gallery.values() if len(i.cameras_seen) >= 2)
            return {
                "gallery_size": len(self._gallery),
                "cross_camera_persons": cross_camera,
                "total_sightings": sum(len(i.sightings) for i in self._gallery.values()),
            }


# ── Global Singleton ─────────────────────────────────────────────

_reid_tracker: Optional[ReIDTracker] = None
_reid_lock = threading.Lock()


def get_reid_tracker(on_cross_camera: Optional[callable] = None) -> ReIDTracker:
    """Get or create the global ReID tracker singleton."""
    global _reid_tracker
    if _reid_tracker is None:
        with _reid_lock:
            if _reid_tracker is None:
                _reid_tracker = ReIDTracker(on_cross_camera=on_cross_camera)
    return _reid_tracker
