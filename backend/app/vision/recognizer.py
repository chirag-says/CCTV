"""
Face Recognition Module.

Matches detected faces against a database of known face encodings.
Uses in-memory caching with periodic refresh for high-performance matching.
"""

import numpy as np
import face_recognition
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
    encoding: np.ndarray
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
    Recognizes faces by comparing encodings against known faces.
    
    Features:
    - In-memory encoding cache for fast matching
    - Thread-safe cache updates
    - Batch comparison using numpy vectorization
    - Configurable match tolerance
    - Unknown face similarity checking
    """

    def __init__(self, tolerance: float = None):
        self.tolerance = tolerance or settings.FACE_MATCH_TOLERANCE
        self._known_faces: List[KnownFace] = []
        self._known_encodings: Optional[np.ndarray] = None  # Pre-computed matrix
        self._lock = threading.RLock()
        self._encoding_count = 0
        
        logger.info(f"FaceRecognizer initialized: tolerance={self.tolerance}")

    def load_encodings(self, encodings_data: List[dict]):
        """
        Load known face encodings from database into memory.
        
        Args:
            encodings_data: List of dicts with keys:
                - person_id, person_name, encoding (bytes), encoding_id, quality
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
                    
                    if isinstance(encoding, np.ndarray) and encoding.shape == (128,):
                        face = KnownFace(
                            person_id=data["person_id"],
                            person_name=data.get("person_name", "Unknown"),
                            encoding=encoding,
                            encoding_id=data.get("encoding_id", ""),
                            quality=data.get("quality", 1.0),
                        )
                        self._known_faces.append(face)
                        valid_encodings.append(encoding)
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
        Match a face encoding against all known faces.
        
        Uses vectorized numpy operations for efficient batch comparison.
        
        Args:
            face_encoding: 128-d face encoding vector
            
        Returns:
            MatchResult with best match info
        """
        if self._known_encodings is None or len(self._known_faces) == 0:
            return MatchResult(
                matched=False,
                encoding=face_encoding,
            )

        with self._lock:
            # Vectorized distance computation
            distances = face_recognition.face_distance(
                self._known_encodings, face_encoding
            )

            # Find best match
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]

            if best_distance <= self.tolerance:
                best_face = self._known_faces[best_idx]
                confidence = 1.0 - best_distance  # Convert distance to confidence
                return MatchResult(
                    matched=True,
                    person_id=best_face.person_id,
                    person_name=best_face.person_name,
                    confidence=round(confidence, 3),
                    distance=round(best_distance, 4),
                    encoding=face_encoding,
                )
            else:
                return MatchResult(
                    matched=False,
                    confidence=round(1.0 - best_distance, 3),
                    distance=round(best_distance, 4),
                    encoding=face_encoding,
                )

    def generate_encoding(self, rgb_frame: np.ndarray, face_location: tuple) -> Optional[np.ndarray]:
        """
        Generate a 128-d face encoding from a frame and face location.
        
        Args:
            rgb_frame: RGB frame (numpy array)
            face_location: (top, right, bottom, left) tuple
            
        Returns:
            128-d encoding vector or None if encoding fails
        """
        try:
            encodings = face_recognition.face_encodings(
                rgb_frame, [face_location], num_jitters=1
            )
            if encodings:
                return encodings[0]
            return None
        except Exception as e:
            logger.error(f"Encoding generation failed: {e}")
            return None

    def batch_generate_encodings(
        self, rgb_frame: np.ndarray, face_locations: List[tuple]
    ) -> List[Optional[np.ndarray]]:
        """Generate encodings for multiple faces in a single frame."""
        try:
            encodings = face_recognition.face_encodings(
                rgb_frame, face_locations, num_jitters=1
            )
            return encodings
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return [None] * len(face_locations)

    def compare_unknown_faces(
        self, encoding1: np.ndarray, encoding2: np.ndarray
    ) -> float:
        """
        Compare two unknown face encodings to check similarity.
        Returns distance (lower = more similar).
        """
        distance = face_recognition.face_distance(
            [encoding1], encoding2
        )[0]
        return float(distance)

    @property
    def known_count(self) -> int:
        return self._encoding_count

    @property
    def known_persons(self) -> List[str]:
        """Get unique person IDs in cache."""
        return list(set(f.person_id for f in self._known_faces))
