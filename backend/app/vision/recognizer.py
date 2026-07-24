"""
Face Recognition Module — ArcFace via InsightFace.

Matches detected faces against a database of known face embeddings
using ArcFace 512-dimensional embeddings and cosine similarity.

v2 Changes:
- Multiple embeddings per person (5–20), best-score-per-person matching
- Top-K candidate retrieval with adaptive threshold + margin check
- SearchBackend abstraction (NumPy default, FAISS future)
- Batch recognition for multi-face frames
- Quality-aware threshold selection
- Replaces pickle with raw float32 byte deserialization
"""

import numpy as np
import pickle
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from app.config import settings
from app.vision.search_backend import SearchBackend, create_search_backend

logger = logging.getLogger(__name__)


# ── Encoding serialization helpers ───────────────────────────────

def serialize_encoding(encoding: np.ndarray) -> bytes:
    """Serialize a numpy encoding to raw float32 bytes."""
    return encoding.astype(np.float32).tobytes()


def deserialize_encoding(data: bytes) -> Optional[np.ndarray]:
    """
    Deserialize an encoding from bytes.
    Handles both new format (raw float32) and legacy format (pickle).
    """
    if data is None:
        return None

    # New format: raw float32 bytes (512 * 4 = 2048 bytes)
    expected_size = settings.FACE_EMBEDDING_DIM * 4  # float32 = 4 bytes
    if len(data) == expected_size:
        try:
            arr = np.frombuffer(data, dtype=np.float32).copy()
            if arr.shape == (settings.FACE_EMBEDDING_DIM,):
                return arr
        except Exception:
            pass

    # Legacy format: pickle
    try:
        arr = pickle.loads(data)
        if isinstance(arr, np.ndarray):
            return arr
    except Exception:
        pass

    return None


@dataclass
class KnownFace:
    """Represents a known face encoding in the recognition cache."""
    person_id: str
    person_name: str
    encoding: np.ndarray  # 512-d ArcFace embedding (normalized)
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
    quality_tier: str = "unknown"  # "strict" / "balanced" / "lenient"
    margin: float = 0.0  # Gap between best and second-best person
    top_k: List[dict] = field(default_factory=list)  # Top-K candidates
    rejection_reason: Optional[str] = None  # Why match was rejected


class FaceRecognizer:
    """
    Recognizes faces by comparing ArcFace embeddings against known faces.

    v2 Features:
    - Multiple embeddings per person (best-score matching)
    - Top-K candidate retrieval
    - Adaptive threshold based on face quality
    - Best-vs-second-best margin check to reject ambiguous matches
    - SearchBackend abstraction (NumPy default)
    - Batch recognition for multi-face frames
    - Thread-safe cache updates
    """

    def __init__(self, tolerance: float = None):
        self.tolerance = tolerance or settings.FACE_MATCH_TOLERANCE
        self._embedding_dim = settings.FACE_EMBEDDING_DIM  # 512
        self._known_faces: List[KnownFace] = []
        self._lock = threading.RLock()
        self._encoding_count = 0
        self._cache_version = 0

        # Per-person index: person_id → list of indices in _known_faces
        self._person_index: Dict[str, List[int]] = {}

        # Search backend (abstraction for future FAISS swap)
        self._backend: SearchBackend = create_search_backend()

        # Adaptive threshold settings
        self._strict_threshold = settings.FACE_MATCH_STRICT
        self._balanced_threshold = settings.FACE_MATCH_BALANCED
        self._lenient_threshold = settings.FACE_MATCH_LENIENT
        self._min_margin = settings.FACE_MATCH_MIN_MARGIN
        self._top_k = settings.FACE_MATCH_TOP_K

        logger.info(
            f"FaceRecognizer initialized (ArcFace v2): "
            f"dim={self._embedding_dim}, "
            f"thresholds=({self._strict_threshold}/{self._balanced_threshold}/{self._lenient_threshold}), "
            f"margin={self._min_margin}, top_k={self._top_k}"
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
            self._person_index = {}
            valid_encodings = []
            valid_person_ids = []
            valid_encoding_ids = []
            valid_qualities = []

            for data in encodings_data:
                try:
                    # Handle encoding in multiple formats:
                    # - numpy array (already unpickled/converted)
                    # - bytes (raw float32 or pickled, from database)
                    raw_encoding = data["encoding"]
                    if isinstance(raw_encoding, np.ndarray):
                        encoding = raw_encoding
                    elif isinstance(raw_encoding, bytes):
                        encoding = deserialize_encoding(raw_encoding)
                    else:
                        encoding = pickle.loads(raw_encoding)

                    # Accept only matching dimension (e.g. 512-d for ArcFace)
                    if isinstance(encoding, np.ndarray) and encoding.shape == (self._embedding_dim,):
                        # Normalize for cosine similarity
                        norm = np.linalg.norm(encoding)
                        if norm > 0:
                            encoding = encoding / norm

                        idx = len(self._known_faces)
                        person_id = data["person_id"]
                        face = KnownFace(
                            person_id=person_id,
                            person_name=data.get("person_name", "Unknown"),
                            encoding=encoding,
                            encoding_id=data.get("encoding_id", ""),
                            quality=data.get("quality", 1.0),
                        )
                        self._known_faces.append(face)
                        valid_encodings.append(encoding)
                        valid_person_ids.append(person_id)
                        valid_encoding_ids.append(face.encoding_id)
                        valid_qualities.append(face.quality)

                        # Build per-person index
                        if person_id not in self._person_index:
                            self._person_index[person_id] = []
                        self._person_index[person_id].append(idx)
                    else:
                        logger.warning(
                            f"Skipping encoding with unexpected shape: "
                            f"{encoding.shape if isinstance(encoding, np.ndarray) else type(encoding)}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load encoding: {e}")

            # Build search backend index
            if valid_encodings:
                enc_matrix = np.array(valid_encodings)
                self._backend.build_index(
                    enc_matrix, valid_person_ids, valid_encoding_ids, valid_qualities
                )
            else:
                self._backend.build_index(
                    np.empty((0, self._embedding_dim)), [], [], []
                )

            self._encoding_count = len(self._known_faces)
            self._cache_version += 1

            # Log per-person encoding counts
            person_counts = {
                pid: len(indices) for pid, indices in self._person_index.items()
            }
            logger.info(
                f"Loaded {self._encoding_count} face encodings "
                f"for {len(self._person_index)} persons into cache "
                f"(version={self._cache_version}). "
                f"Encodings per person: {person_counts}"
            )

    def add_encoding(self, face: KnownFace):
        """Add a single encoding to the cache (e.g., after enrollment)."""
        with self._lock:
            # Normalize the encoding
            norm = np.linalg.norm(face.encoding)
            if norm > 0:
                face.encoding = face.encoding / norm

            idx = len(self._known_faces)
            self._known_faces.append(face)

            # Update per-person index
            if face.person_id not in self._person_index:
                self._person_index[face.person_id] = []
            self._person_index[face.person_id].append(idx)

            # Update search backend
            self._backend.add(face.encoding, face.person_id, face.encoding_id, face.quality)

            self._encoding_count += 1
            self._cache_version += 1
            logger.info(
                f"Added encoding for {face.person_name} "
                f"(total for person: {len(self._person_index[face.person_id])}). "
                f"Total: {self._encoding_count}"
            )

    def remove_person_encodings(self, person_id: str):
        """Remove all encodings for a person from cache."""
        with self._lock:
            indices_to_keep = [
                i for i, f in enumerate(self._known_faces)
                if f.person_id != person_id
            ]
            self._known_faces = [self._known_faces[i] for i in indices_to_keep]

            # Rebuild per-person index
            self._person_index = {}
            for i, face in enumerate(self._known_faces):
                if face.person_id not in self._person_index:
                    self._person_index[face.person_id] = []
                self._person_index[face.person_id].append(i)

            # Rebuild search backend
            if self._known_faces:
                enc_matrix = np.array([f.encoding for f in self._known_faces])
                self._backend.build_index(
                    enc_matrix,
                    [f.person_id for f in self._known_faces],
                    [f.encoding_id for f in self._known_faces],
                    [f.quality for f in self._known_faces],
                )
            else:
                self._backend.build_index(
                    np.empty((0, self._embedding_dim)), [], [], []
                )

            self._encoding_count = len(self._known_faces)
            self._cache_version += 1

    def _get_adaptive_threshold(self, face_quality: float = 0.5) -> Tuple[float, str]:
        """
        Get quality-adaptive distance threshold.

        Higher quality faces get stricter thresholds (fewer false positives).
        Lower quality faces get more lenient thresholds (fewer false negatives).

        Returns:
            (threshold, quality_tier_name)
        """
        if face_quality >= 0.8:
            return self._strict_threshold, "strict"
        elif face_quality >= 0.5:
            return self._balanced_threshold, "balanced"
        else:
            return self._lenient_threshold, "lenient"

    def recognize_face(
        self, face_encoding: np.ndarray, face_quality: float = 0.5
    ) -> MatchResult:
        """
        Match a face encoding against all known faces.

        Uses Top-K candidate retrieval, per-person best-score grouping,
        adaptive threshold, and margin check.

        Args:
            face_encoding: ArcFace embedding vector (512-d)
            face_quality: Quality score (0.0–1.0) for adaptive threshold

        Returns:
            MatchResult with best match info and Top-K candidates
        """
        if self._backend.size == 0 or len(self._known_faces) == 0:
            logger.warning(
                "⚠️ Face recognition attempted but NO known encodings loaded! "
                "All faces will be marked as UNKNOWN."
            )
            return MatchResult(
                matched=False,
                encoding=face_encoding,
                rejection_reason="no_encodings_loaded",
            )

        with self._lock:
            # Normalize query embedding
            norm = np.linalg.norm(face_encoding)
            if norm > 0:
                query_norm = face_encoding / norm
            else:
                return MatchResult(
                    matched=False, encoding=face_encoding,
                    rejection_reason="zero_norm_embedding",
                )

            # ── Step 1: Top-K search via backend ─────────────────
            # Returns all similarities (we need per-person grouping)
            raw_results = self._backend.search(query_norm, top_k=self._backend.size)

            if not raw_results:
                return MatchResult(
                    matched=False, encoding=face_encoding,
                    rejection_reason="search_returned_empty",
                )

            # ── Step 2: Group by person, take best score per person ──
            person_best: Dict[str, dict] = {}
            for idx, similarity in raw_results:
                face = self._known_faces[idx]
                pid = face.person_id

                if pid not in person_best or similarity > person_best[pid]["similarity"]:
                    person_best[pid] = {
                        "person_id": pid,
                        "person_name": face.person_name,
                        "similarity": similarity,
                        "distance": 1.0 - similarity,
                        "encoding_quality": face.quality,
                        "encoding_id": face.encoding_id,
                    }

            # ── Step 3: Sort persons by distance (ascending) ─────
            sorted_persons = sorted(
                person_best.values(), key=lambda x: x["distance"]
            )

            # ── Step 4: Top-K candidates ─────────────────────────
            top_k_candidates = sorted_persons[:self._top_k]

            # ── Step 5: Adaptive threshold ───────────────────────
            threshold, quality_tier = self._get_adaptive_threshold(face_quality)

            best = sorted_persons[0]
            best_distance = best["distance"]

            # ── Step 6: Margin check ─────────────────────────────
            margin = float("inf")
            if len(sorted_persons) >= 2:
                second_best_distance = sorted_persons[1]["distance"]
                margin = second_best_distance - best_distance

            # ── Step 7: Decision ─────────────────────────────────
            if best_distance <= threshold:
                if margin < self._min_margin and len(sorted_persons) >= 2:
                    # Ambiguous — too close to second-best person
                    logger.info(
                        f"❌ AMBIGUOUS: {best['person_name']} "
                        f"(distance={best_distance:.4f}) vs "
                        f"{sorted_persons[1]['person_name']} "
                        f"(distance={sorted_persons[1]['distance']:.4f}), "
                        f"margin={margin:.4f} < {self._min_margin}"
                    )
                    return MatchResult(
                        matched=False,
                        confidence=round(max(0.0, best["similarity"]), 3),
                        distance=round(best_distance, 4),
                        encoding=face_encoding,
                        quality_tier=quality_tier,
                        margin=round(margin, 4),
                        top_k=top_k_candidates,
                        rejection_reason="ambiguous_margin",
                    )

                # ✅ Match!
                confidence = float(best["similarity"])
                logger.info(
                    f"✅ MATCHED: {best['person_name']} "
                    f"(distance={best_distance:.4f}, confidence={confidence:.3f}, "
                    f"tier={quality_tier}, margin={margin:.4f})"
                )
                return MatchResult(
                    matched=True,
                    person_id=best["person_id"],
                    person_name=best["person_name"],
                    confidence=round(max(0.0, confidence), 3),
                    distance=round(best_distance, 4),
                    encoding=face_encoding,
                    quality_tier=quality_tier,
                    margin=round(min(margin, 9.99), 4),
                    top_k=top_k_candidates,
                )
            else:
                # No match — best distance exceeds threshold
                logger.info(
                    f"❌ NO MATCH: closest={best['person_name']} "
                    f"(distance={best_distance:.4f}, threshold={threshold}, "
                    f"tier={quality_tier})"
                )
                return MatchResult(
                    matched=False,
                    confidence=round(max(0.0, best["similarity"]), 3),
                    distance=round(best_distance, 4),
                    encoding=face_encoding,
                    quality_tier=quality_tier,
                    margin=round(min(margin, 9.99), 4),
                    top_k=top_k_candidates,
                    rejection_reason="distance_exceeds_threshold",
                )

    def recognize_faces_batch(
        self,
        face_encodings: List[np.ndarray],
        face_qualities: Optional[List[float]] = None,
    ) -> List[MatchResult]:
        """
        Batch recognition for multiple faces in a single frame.

        Uses a single matrix multiply for all queries, then applies
        per-person grouping + adaptive threshold + margin check per query.

        Args:
            face_encodings: List of embedding vectors
            face_qualities: Optional quality scores per face

        Returns:
            List of MatchResult, one per input face
        """
        if not face_encodings:
            return []

        if face_qualities is None:
            face_qualities = [0.5] * len(face_encodings)

        if self._backend.size == 0:
            return [
                MatchResult(matched=False, encoding=enc, rejection_reason="no_encodings_loaded")
                for enc in face_encodings
            ]

        # For now, delegate to per-face recognition
        # (The backend already uses vectorized ops internally)
        # Full batch optimization would require batched per-person grouping
        results = []
        for enc, qual in zip(face_encodings, face_qualities):
            results.append(self.recognize_face(enc, face_quality=qual))
        return results

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
        return list(self._person_index.keys())

    @property
    def cache_version(self) -> int:
        return self._cache_version

    @property
    def person_encoding_counts(self) -> Dict[str, int]:
        """Get number of encodings per person."""
        return {pid: len(indices) for pid, indices in self._person_index.items()}
