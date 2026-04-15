"""
Face Recognition Module — ArcFace via InsightFace.

Matches detected faces against a database of known face embeddings
using ArcFace 512-dimensional embeddings and cosine similarity.

Replaces the old dlib/face_recognition-based recognizer.
"""

import numpy as np
import pickle
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class KnownFace:
    """Represents a known face in the recognition cache."""
    person_id: str
    person_name: str
    encoding: np.ndarray  # 512-d ArcFace embedding
    encoding_id: str
    quality: float = 1.0


@dataclass
class MatchResult:
    """Result of a face matching operation."""
    matched: bool
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    distance: float = 1.0
    encoding: Optional[np.ndarray] = None


class FaceRecognizer:
    """
    Recognizes faces by comparing ArcFace embeddings against known faces.

    Uses cosine similarity for matching (ArcFace embeddings are normalized).

    Features:
    - In-memory encoding cache for fast matching
    - Thread-safe cache updates
    - Batch comparison using numpy vectorization
    - Configurable match tolerance
    - Unknown face similarity checking
    """

    def __init__(self, tolerance: float = None):
        self.tolerance = tolerance or settings.FACE_MATCH_TOLERANCE
        self._embedding_dim = settings.FACE_EMBEDDING_DIM  # 512
        self._known_faces: List[KnownFace] = []
        self._known_encodings: Optional[np.ndarray] = None  # Pre-computed matrix
        self._lock = threading.RLock()
        self._encoding_count = 0

        logger.info(
            f"FaceRecognizer initialized (ArcFace): "
            f"tolerance={self.tolerance}, dim={self._embedding_dim}"
        )

    def load_encodings(self, encodings_data: List[dict]):
        """
        Load known face encodings from database into memory.

        Args:
            encodings_data: List of dicts with keys:
                - person_id, person_name, encoding (bytes or ndarray),
                  encoding_id, quality
        """
        with self._lock:
            self._known_faces = []
            valid_encodings = []

            for data in encodings_data:
                try:
                    # Handle encoding in either format:
                    # - numpy array (already unpickled, e.g. from mock store)
                    # - bytes (pickled, from database)
                    raw_encoding = data["encoding"]
                    if isinstance(raw_encoding, np.ndarray):
                        encoding = raw_encoding
                    else:
                        encoding = pickle.loads(raw_encoding)

                    # Accept only matching dimension (e.g. 512-d for ArcFace)
                    if isinstance(encoding, np.ndarray) and encoding.shape == (self._embedding_dim,):
                        # Normalize for cosine similarity
                        norm = np.linalg.norm(encoding)
                        if norm > 0:
                            encoding = encoding / norm

                        face = KnownFace(
                            person_id=data["person_id"],
                            person_name=data.get("person_name", "Unknown"),
                            encoding=encoding,
                            encoding_id=data.get("encoding_id", ""),
                            quality=data.get("quality", 1.0),
                        )
                        self._known_faces.append(face)
                        valid_encodings.append(encoding)
                    else:
                        logger.warning(
                            f"Skipping encoding with unexpected shape: "
                            f"{encoding.shape if isinstance(encoding, np.ndarray) else type(encoding)}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load encoding: {e}")

            # Pre-compute encoding matrix for vectorized comparison
            if valid_encodings:
                self._known_encodings = np.array(valid_encodings)
            else:
                self._known_encodings = None

            self._encoding_count = len(self._known_faces)
            logger.info(f"Loaded {self._encoding_count} face encodings into cache")

    def add_encoding(self, face: KnownFace):
        """Add a single encoding to the cache (e.g., after enrollment)."""
        with self._lock:
            # Normalize the encoding
            norm = np.linalg.norm(face.encoding)
            if norm > 0:
                face.encoding = face.encoding / norm

            self._known_faces.append(face)
            if self._known_encodings is not None:
                self._known_encodings = np.vstack([
                    self._known_encodings, face.encoding.reshape(1, -1)
                ])
            else:
                self._known_encodings = face.encoding.reshape(1, -1)
            self._encoding_count += 1
            logger.info(f"Added encoding for {face.person_name}. Total: {self._encoding_count}")

    def remove_person_encodings(self, person_id: str):
        """Remove all encodings for a person from cache."""
        with self._lock:
            indices_to_keep = [
                i for i, f in enumerate(self._known_faces)
                if f.person_id != person_id
            ]
            self._known_faces = [self._known_faces[i] for i in indices_to_keep]
            if self._known_encodings is not None and indices_to_keep:
                self._known_encodings = self._known_encodings[indices_to_keep]
            else:
                self._known_encodings = None
            self._encoding_count = len(self._known_faces)

    def recognize_face(
        self, face_encoding: np.ndarray
    ) -> MatchResult:
        """
        Match a face encoding against all known faces using cosine distance.

        Args:
            face_encoding: ArcFace embedding vector (512-d or 128-d)

        Returns:
            MatchResult with best match info
        """
        if self._known_encodings is None or len(self._known_faces) == 0:
            return MatchResult(
                matched=False,
                encoding=face_encoding,
            )

        with self._lock:
            # Normalize query embedding
            norm = np.linalg.norm(face_encoding)
            if norm > 0:
                face_encoding_norm = face_encoding / norm
            else:
                return MatchResult(matched=False, encoding=face_encoding)

            # Cosine similarity: dot product of normalized vectors
            # similarity ∈ [-1, 1], higher is more similar
            similarities = np.dot(self._known_encodings, face_encoding_norm)

            # Convert to cosine distance: distance = 1 - similarity
            # distance ∈ [0, 2], lower is more similar
            distances = 1.0 - similarities

            # Find best match (lowest distance)
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]

            if best_distance <= self.tolerance:
                best_face = self._known_faces[best_idx]
                confidence = float(similarities[best_idx])  # Use similarity as confidence
                return MatchResult(
                    matched=True,
                    person_id=best_face.person_id,
                    person_name=best_face.person_name,
                    confidence=round(max(0.0, confidence), 3),
                    distance=round(float(best_distance), 4),
                    encoding=face_encoding,
                )
            else:
                best_similarity = float(similarities[best_idx])
                return MatchResult(
                    matched=False,
                    confidence=round(max(0.0, best_similarity), 3),
                    distance=round(float(best_distance), 4),
                    encoding=face_encoding,
                )

    def generate_encoding(
        self, rgb_frame: np.ndarray, face_location: tuple
    ) -> Optional[np.ndarray]:
        """
        Generate a face embedding from a frame and face location.

        Uses insightface to detect and compute the embedding for the face
        at the specified location.

        Args:
            rgb_frame: RGB frame (numpy array)
            face_location: (top, right, bottom, left) tuple

        Returns:
            512-d ArcFace embedding vector or None if encoding fails
        """
        try:
            from app.vision.detector import _get_insightface_app

            app = _get_insightface_app()

            # Crop the face region with generous padding for better detection
            top, right, bottom, left = face_location
            h, w = rgb_frame.shape[:2]

            # Add padding for better detection
            pad = max(int((bottom - top) * 0.3), int((right - left) * 0.3), 20)
            crop_top = max(0, top - pad)
            crop_bottom = min(h, bottom + pad)
            crop_left = max(0, left - pad)
            crop_right = min(w, right + pad)

            crop = rgb_frame[crop_top:crop_bottom, crop_left:crop_right]
            if crop.size == 0:
                return None

            # Run detection on the crop to get the embedding
            faces = app.get(crop)
            if not faces:
                return None

            # Return embedding from the first detected face
            # (the crop should contain exactly one face)
            if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                return faces[0].embedding
            return None

        except Exception as e:
            logger.error(f"Encoding generation failed: {e}")
            return None

    def batch_generate_encodings(
        self, rgb_frame: np.ndarray, face_locations: List[tuple]
    ) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple faces in a single frame.

        For efficiency, runs insightface detection on the full frame once
        and matches detected faces to the provided locations.
        """
        if not face_locations:
            return []

        try:
            from app.vision.detector import _get_insightface_app

            app = _get_insightface_app()

            # Run detection on the full frame
            faces = app.get(rgb_frame)

            if not faces:
                # Fall back to per-face generation
                return [
                    self.generate_encoding(rgb_frame, loc)
                    for loc in face_locations
                ]

            # Match detected faces to provided locations by IoU
            encodings = []
            for loc in face_locations:
                top, right, bottom, left = loc
                best_face = None
                best_iou = 0.0

                for face in faces:
                    if not hasattr(face, 'embedding') or face.embedding is None:
                        continue

                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)

                    # Calculate IoU
                    inter_x1 = max(left, fx1)
                    inter_y1 = max(top, fy1)
                    inter_x2 = min(right, fx2)
                    inter_y2 = min(bottom, fy2)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        area1 = (right - left) * (bottom - top)
                        area2 = (fx2 - fx1) * (fy2 - fy1)
                        union_area = area1 + area2 - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                    else:
                        iou = 0.0

                    if iou > best_iou:
                        best_iou = iou
                        best_face = face

                if best_face is not None and best_iou > 0.3:
                    encodings.append(best_face.embedding)
                else:
                    # Fallback: try individual crop-based encoding
                    encodings.append(self.generate_encoding(rgb_frame, loc))

            return encodings

        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return [None] * len(face_locations)

    def compare_unknown_faces(
        self, encoding1: np.ndarray, encoding2: np.ndarray
    ) -> float:
        """
        Compare two unknown face encodings to check similarity.
        Returns cosine distance (lower = more similar).
        """
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)

        if norm1 == 0 or norm2 == 0:
            return 1.0

        similarity = np.dot(encoding1 / norm1, encoding2 / norm2)
        return float(1.0 - similarity)

    @property
    def known_count(self) -> int:
        return self._encoding_count

    @property
    def known_persons(self) -> List[str]:
        """Get unique person IDs in cache."""
        return list(set(f.person_id for f in self._known_faces))
