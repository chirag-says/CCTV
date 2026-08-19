"""
VLM Service — Vision-Language Model interface for face comparison.

Part of the Gallery Optimizer pipeline. Provides a pluggable interface for
different VLM providers to compare face images and explain whether they're
the same person.

Architecture:
  VLMReviewer (base)
  ├── DisabledReviewer  — neutral scores for testing
  ├── OpenAIReviewer    — GPT-4o (cloud, requires API key)
  └── OllamaReviewer    — Qwen2.5-VL-7B (local, fully offline)

The VLM is advisory only — it never modifies the gallery. ArcFace remains
the biometric authority. The VLM only produces recommendations.

Never called from the realtime pipeline. Only used by the background
Gallery Optimizer worker.
"""

import base64
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ── VLM Result ───────────────────────────────────────────────────

@dataclass
class VLMResult:
    """Result from a VLM face comparison."""
    agreement: float = 0.5       # 0.0 = definitely different, 1.0 = definitely same
    confidence: str = "low"      # low/medium/high
    explanation: str = ""        # Natural language reasoning
    differences: List[str] = field(default_factory=list)  # Noted differences
    similarities: List[str] = field(default_factory=list)  # Noted similarities
    error: Optional[str] = None  # Error message if VLM call failed
    # Gallery Optimizer fields
    should_add_embedding: bool = False
    gallery_improvement_value: str = "none"    # none/low/medium/high
    quality: str = "unknown"                   # good/fair/poor/unknown
    detected_pose: str = ""                    # VLM's pose assessment


# ── VLM Cache ────────────────────────────────────────────────────

class VLMCache:
    """
    Cache VLM results by hash(cluster_id + person_id + gallery_version).

    Prevents re-querying Ollama for the same unknown-vs-person pair
    when nothing has changed. Invalidates when:
    - Person's gallery changes (gallery_version bumps)
    - Cache entry exceeds TTL
    - LRU eviction at max capacity
    """

    def __init__(self, max_entries: int = 2000, ttl_hours: int = 168):
        self._cache: OrderedDict[str, Tuple[VLMResult, float]] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_hours * 3600

    @staticmethod
    def _make_key(cluster_id: str, person_id: str, gallery_version: int) -> str:
        raw = f"{cluster_id}:{person_id}:{gallery_version}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, cluster_id: str, person_id: str, gallery_version: int) -> Optional[VLMResult]:
        """Get cached result, or None if miss/expired."""
        key = self._make_key(cluster_id, person_id, gallery_version)
        entry = self._cache.get(key)
        if entry is None:
            return None

        result, timestamp = entry
        if time.time() - timestamp > self._ttl_seconds:
            # Expired
            del self._cache[key]
            return None

        # Move to end (LRU)
        self._cache.move_to_end(key)
        return result

    def put(self, cluster_id: str, person_id: str, gallery_version: int, result: VLMResult):
        """Store a result in the cache."""
        key = self._make_key(cluster_id, person_id, gallery_version)
        self._cache[key] = (result, time.time())
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def clear(self):
        """Clear all cached results."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ── Base Class ───────────────────────────────────────────────────

class VLMReviewer:
    """Base class for VLM face comparison providers."""

    def compare_faces(
        self,
        image_a_path: str,
        image_b_path: str,
        context: Optional[str] = None,
    ) -> VLMResult:
        """
        Compare two face images and determine if they're the same person.

        Args:
            image_a_path: Path to the unknown face image (crop)
            image_b_path: Path to the gallery face image (crop)
            context: Optional context string

        Returns:
            VLMResult with agreement score, explanation, and analysis
        """
        raise NotImplementedError


# ── Disabled Reviewer ────────────────────────────────────────────

class DisabledReviewer(VLMReviewer):
    """Returns neutral scores when VLM is disabled."""

    def compare_faces(self, image_a_path, image_b_path, context=None):
        return VLMResult(
            agreement=0.5,
            confidence="low",
            explanation="VLM review disabled — using embedding similarity only",
        )


# ── OpenAI Reviewer ──────────────────────────────────────────────

class OpenAIReviewer(VLMReviewer):
    """Uses OpenAI GPT-4o for face comparison."""

    SYSTEM_PROMPT = """You are a biometric analysis assistant. You will be shown two face images.
Your job is to analyze whether these images show the SAME person.

Focus on:
- Facial bone structure (forehead, jawline, cheekbones)
- Nose shape and bridge
- Eye shape and spacing
- Ear shape (if visible)
- Distinctive features (moles, scars, dimples)

Do NOT focus on:
- Clothing or accessories (glasses, hats)
- Lighting conditions
- Expression
- Aging differences within ~5 years
- Hair style or color (can change)

Respond in JSON format:
{
    "agreement": <float 0.0-1.0>,
    "confidence": "<low|medium|high>",
    "explanation": "<2-3 sentence reasoning>",
    "similarities": ["<feature1>", "<feature2>"],
    "differences": ["<feature1>", "<feature2>"]
}"""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(
                    base_url="https://api.openai.com/v1",
                    headers={
                        "Authorization": f"Bearer {settings.IDENTITY_REVIEW_VLM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )
            except Exception as e:
                logger.error(f"Failed to create OpenAI client: {e}")
                raise
        return self._client

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Read and base64-encode an image file."""
        try:
            # Handle both absolute paths and relative snapshot paths
            path = Path(image_path)
            if not path.is_absolute():
                path = Path(settings.SNAPSHOT_DIR) / image_path.lstrip("/")

            if not path.exists():
                # Try stripping /snapshots/ prefix
                alt_path = Path(settings.SNAPSHOT_DIR) / Path(image_path).name
                if alt_path.exists():
                    path = alt_path
                else:
                    logger.warning(f"VLM image not found: {image_path}")
                    return None

            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None

    def compare_faces(self, image_a_path, image_b_path, context=None):
        try:
            img_a = self._encode_image(image_a_path)
            img_b = self._encode_image(image_b_path)

            if not img_a or not img_b:
                return VLMResult(
                    agreement=0.5,
                    confidence="low",
                    explanation="Could not load one or both images for comparison",
                    error="image_load_failed",
                )

            user_content = [
                {"type": "text", "text": "Are these two face images the same person? Analyze carefully."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_a}", "detail": "high"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b}", "detail": "high"},
                },
            ]

            if context:
                user_content.insert(1, {"type": "text", "text": f"Context: {context}"})

            client = self._get_client()
            response = client.post(
                "/chat/completions",
                json={
                    "model": settings.IDENTITY_REVIEW_VLM_MODEL,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
            )

            if response.status_code != 200:
                error_msg = f"OpenAI API error: {response.status_code}"
                logger.error(error_msg)
                return VLMResult(agreement=0.5, confidence="low", error=error_msg)

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            result = json.loads(content)

            return VLMResult(
                agreement=float(result.get("agreement", 0.5)),
                confidence=result.get("confidence", "low"),
                explanation=result.get("explanation", ""),
                similarities=result.get("similarities", []),
                differences=result.get("differences", []),
            )

        except Exception as e:
            logger.error(f"VLM comparison failed: {e}", exc_info=True)
            return VLMResult(
                agreement=0.5,
                confidence="low",
                explanation=f"VLM comparison error: {str(e)}",
                error=str(e),
            )


# ── Ollama Reviewer ──────────────────────────────────────────────

class OllamaReviewer(VLMReviewer):
    """
    Uses local Ollama (Qwen2.5-VL-7B) for face comparison.

    Single pairwise verification: ArcFace picks the best gallery image,
    VLM only verifies that one pair. Fully offline — no cloud API.

    HTTP only — no subprocess, no desktop application.
    """

    SYSTEM_PROMPT = (
        "You are reviewing a biometric face recognition system.\n\n"
        "Image 1 is an UNKNOWN face.\n"
        "Image 2 is a REGISTERED gallery image.\n\n"
        "Do NOT identify people.\n"
        "Do NOT perform biometric recognition.\n\n"
        "Determine whether the UNKNOWN face appears visually consistent "
        "with the REGISTERED person.\n\n"
        "Ignore: Hair, Clothes, Lighting, Expression, Accessories\n"
        "Focus on: Jawline, Forehead, Eye spacing, Ear structure, "
        "Nose bridge, Facial proportions\n\n"
        "Respond ONLY as JSON:\n"
        "{\n"
        '  "same_person": true,\n'
        '  "confidence": 91,\n'
        '  "quality": "good",\n'
        '  "pose": "extreme_left",\n'
        '  "reasoning": "...",\n'
        '  "similarities": ["..."],\n'
        '  "differences": ["..."]\n'
        "}\n\n"
        "Never return markdown. Never return prose. Never return code fences."
    )

    def __init__(self):
        self._client = None
        self._ollama_logged_offline = False

    def _get_client(self):
        """Lazy httpx client for Ollama HTTP API."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                base_url=settings.OLLAMA_HOST,
                timeout=settings.OLLAMA_TIMEOUT_SEC,
            )
        return self._client

    def _prepare_image(self, image_path: str) -> Optional[str]:
        """
        Load existing face crop, resize to 256×256, JPEG encode, base64.

        No face detection. No SCRFD. The pipeline already saved the crop.
        """
        try:
            path = Path(image_path)
            if not path.is_absolute():
                path = Path(settings.SNAPSHOT_DIR) / image_path.lstrip("/")

            if not path.exists():
                # Try stripping path prefix
                alt_path = Path(settings.SNAPSHOT_DIR) / Path(image_path).name
                if alt_path.exists():
                    path = alt_path
                else:
                    logger.warning(f"VLM image not found: {image_path}")
                    return None

            import cv2
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Failed to decode image: {image_path}")
                return None

            # Resize to 256×256 for consistent VLM input
            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

            # JPEG encode
            success, buffer = cv2.imencode(".jpg", img_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.warning(f"Failed to JPEG-encode image: {image_path}")
                return None

            return base64.b64encode(buffer.tobytes()).decode("utf-8")

        except Exception as e:
            logger.error(f"Image preparation failed for {image_path}: {e}")
            return None

    def _parse_response(self, raw_text: str) -> VLMResult:
        """
        Parse VLM JSON response into VLMResult.

        Handles common VLM quirks:
        - Code fence wrapping (```json ... ```)
        - Missing fields
        - Type mismatches
        """
        try:
            # Strip markdown code fences if present
            text = raw_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first line (```json) and last line (```)
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            data = json.loads(text)

            # Map same_person (bool) + confidence (int) to agreement (float)
            same_person = data.get("same_person", False)
            confidence_int = data.get("confidence", 50)

            # Clamp confidence to 0–100
            if isinstance(confidence_int, (int, float)):
                confidence_int = max(0, min(100, int(confidence_int)))
            else:
                confidence_int = 50

            # agreement: same_person=True → confidence/100, same_person=False → (100-confidence)/100
            if same_person:
                agreement = confidence_int / 100.0
            else:
                agreement = (100 - confidence_int) / 100.0

            # Map confidence integer to low/medium/high
            if confidence_int >= 80:
                confidence_str = "high"
            elif confidence_int >= 50:
                confidence_str = "medium"
            else:
                confidence_str = "low"

            return VLMResult(
                agreement=agreement,
                confidence=confidence_str,
                explanation=data.get("reasoning", ""),
                similarities=data.get("similarities", []),
                differences=data.get("differences", []),
                quality=data.get("quality", "unknown"),
                detected_pose=data.get("pose", ""),
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse VLM response: {e}")
            return VLMResult(
                agreement=0.5,
                confidence="low",
                explanation=f"VLM response parse error: {str(e)}",
                error="parse_error",
            )

    def compare_faces(self, image_a_path, image_b_path, context=None):
        """
        Single 1:1 face comparison via Ollama.

        image_a_path: unknown face crop
        image_b_path: ArcFace's best gallery match for this person
        """
        try:
            # Prepare images
            img_a = self._prepare_image(image_a_path)
            img_b = self._prepare_image(image_b_path)

            if not img_a or not img_b:
                return VLMResult(
                    agreement=0.5,
                    confidence="low",
                    explanation="Could not load one or both images for comparison",
                    error="image_load_failed",
                )

            # Build messages for Ollama chat API
            messages = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Compare these two face images. "
                        "Image 1 is the unknown face. Image 2 is the registered gallery image."
                    ),
                    "images": [img_a, img_b],
                },
            ]

            # POST to Ollama
            client = self._get_client()
            response = client.post(
                "/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                },
            )

            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code}"
                logger.error(error_msg)
                return VLMResult(agreement=0.5, confidence="low", error=error_msg)

            # Parse response
            data = response.json()
            content = data.get("message", {}).get("content", "")
            if not content:
                return VLMResult(
                    agreement=0.5,
                    confidence="low",
                    error="empty_response",
                    explanation="Ollama returned empty response",
                )

            self._ollama_logged_offline = False  # Reset offline flag on success
            return self._parse_response(content)

        except Exception as e:
            # Log once to avoid spam
            if not self._ollama_logged_offline:
                logger.error(f"Ollama VLM call failed: {e}")
                self._ollama_logged_offline = True

            return VLMResult(
                agreement=0.5,
                confidence="low",
                explanation=f"Ollama unavailable: {str(e)}",
                error=str(e),
            )


# ── Gallery Image Selection ──────────────────────────────────────

def select_best_gallery_image(
    person_id: str,
    unknown_encoding: np.ndarray,
) -> Optional[str]:
    """
    Select the best gallery image for VLM comparison.

    ArcFace picks the strongest biometric match: compute cosine similarity
    of the unknown encoding against all gallery encodings for this person,
    return the source_image path of the highest-scoring one.

    Args:
        person_id: The candidate person ID
        unknown_encoding: The unknown face embedding (512-dim ArcFace)

    Returns:
        Path to the best matching gallery image, or None
    """
    from app.db.session import SessionLocal
    from app.db.models import FaceEncoding
    from app.vision.recognizer import deserialize_encoding

    db = SessionLocal()
    try:
        encodings = (
            db.query(FaceEncoding)
            .filter(
                FaceEncoding.person_id == person_id,
                FaceEncoding.source_image.isnot(None),
            )
            .all()
        )

        if not encodings:
            return None

        # Normalize unknown encoding
        norm = np.linalg.norm(unknown_encoding)
        if norm == 0:
            return None
        unknown_norm = unknown_encoding / norm

        best_sim = -1.0
        best_path = None

        for enc in encodings:
            arr = deserialize_encoding(enc.encoding_data)
            if arr is None:
                continue
            enc_norm = np.linalg.norm(arr)
            if enc_norm == 0:
                continue

            similarity = float(np.dot(unknown_norm, arr / enc_norm))
            if similarity > best_sim:
                best_sim = similarity
                best_path = enc.source_image

        return best_path

    finally:
        db.close()


def select_gallery_images(person_id: str, max_images: int = 5) -> List[str]:
    """
    Select diverse gallery images for a person (one per pose bucket).

    Used for enriching review cards with gallery context.
    Not used for VLM comparison (that uses select_best_gallery_image).
    """
    from app.db.session import SessionLocal
    from app.db.models import FaceEncoding
    from app.services.embedding_suggestions import classify_pose_bucket

    db = SessionLocal()
    try:
        encodings = (
            db.query(FaceEncoding)
            .filter(
                FaceEncoding.person_id == person_id,
                FaceEncoding.source_image.isnot(None),
            )
            .order_by(FaceEncoding.quality.desc())
            .all()
        )

        # Pick best image per pose bucket
        seen_buckets: Dict[str, str] = {}
        for enc in encodings:
            bucket = enc.pose_bucket or classify_pose_bucket(enc.yaw or 0.0)
            if bucket not in seen_buckets:
                seen_buckets[bucket] = enc.source_image

            if len(seen_buckets) >= max_images:
                break

        return list(seen_buckets.values())

    finally:
        db.close()


# ── Factory ──────────────────────────────────────────────────────

# Module-level VLM cache (shared across worker cycles)
_vlm_cache = VLMCache(
    max_entries=2000,
    ttl_hours=settings.GALLERY_OPT_VLM_CACHE_TTL_HOURS,
)


def get_vlm_cache() -> VLMCache:
    """Get the module-level VLM result cache."""
    return _vlm_cache


def get_vlm_reviewer() -> VLMReviewer:
    """Factory: returns the configured VLM reviewer instance."""
    provider = settings.IDENTITY_REVIEW_VLM_PROVIDER.lower()

    if provider == "disabled":
        return DisabledReviewer()

    if provider == "ollama":
        logger.info(
            f"Gallery Optimizer using Ollama VLM "
            f"(host={settings.OLLAMA_HOST}, model={settings.OLLAMA_MODEL})"
        )
        return OllamaReviewer()

    if provider == "openai":
        if not settings.IDENTITY_REVIEW_VLM_API_KEY:
            logger.warning(
                "VLM provider 'openai' configured but no API key set. "
                "Using disabled reviewer."
            )
            return DisabledReviewer()
        return OpenAIReviewer()

    logger.warning(f"Unknown VLM provider '{provider}', using disabled reviewer")
    return DisabledReviewer()
