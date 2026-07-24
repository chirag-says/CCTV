"""
Embedding Suggestion Service — Smart Diversity Analysis.

"The system should learn coverage rather than accumulate images. The goal is not
to collect the largest number of embeddings, but to maintain a small, diverse,
and high-quality representation of each person that maximizes recognition accuracy."

Key design decisions:
1. Only store suggestions that PASS diversity — never queue duplicates
2. Pose buckets (frontal/left/right/extreme) instead of raw angle comparison
3. Appearance diversity (glasses, mask, cap, lighting)
4. Compare against representative embeddings per pose bucket, not ALL
5. Health scoring (usage_count + success_count + quality) for smart replacement
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.db.models import FaceEncoding, SuggestedEncoding, Person
from app.vision.recognizer import serialize_encoding, deserialize_encoding

logger = logging.getLogger(__name__)


# ── Pose Buckets ─────────────────────────────────────────────────

POSE_BUCKETS = ["frontal", "left", "right", "extreme_left", "extreme_right"]


def classify_pose_bucket(yaw: float) -> str:
    """
    Classify a yaw angle into a pose bucket.

    Buckets:
      frontal:       |yaw| <= 15°
      left:          15° < yaw <= 45°
      right:         -45° <= yaw < -15°
      extreme_left:  yaw > 45°
      extreme_right: yaw < -45°
    """
    if yaw is None:
        return "frontal"
    abs_yaw = abs(yaw)
    if abs_yaw <= settings.POSE_BUCKET_FRONTAL:
        return "frontal"
    elif abs_yaw <= settings.POSE_BUCKET_MODERATE:
        return "left" if yaw > 0 else "right"
    else:
        return "extreme_left" if yaw > 0 else "extreme_right"


# ── Appearance Tags ──────────────────────────────────────────────

def detect_appearance_tags(
    face_crop: Optional[np.ndarray] = None,
    lighting: Optional[str] = None,
) -> List[str]:
    """
    Detect appearance attributes from face crop.

    Currently uses simple heuristics. Can be upgraded to a lightweight
    attribute classifier (e.g., InsightFace attribute model) later.

    Returns list of tags like ["glasses", "bright_lighting"].
    """
    tags = []

    # Lighting-based tags
    if lighting:
        if lighting == "dim":
            tags.append("dim_lighting")
        elif lighting == "bright":
            tags.append("bright_lighting")
        elif lighting == "backlit":
            tags.append("backlit")

    # TODO: Add real attribute detection (glasses, mask, cap, beard)
    # when InsightFace attribute model is integrated.
    # For now, these are detected via the appearance_tags field
    # which can be manually set or detected by future models.

    return tags


# ── Health Score ─────────────────────────────────────────────────

def compute_health_score(encoding: FaceEncoding) -> float:
    """
    Compute a health score for an encoding based on quality and usefulness.

    Health = quality * 0.4 + usage_effectiveness * 0.4 + recency * 0.2

    Low health → candidate for replacement when cap is reached.
    """
    quality_score = (encoding.quality or 0.5) * 0.4

    # Usage effectiveness: success_count / max(usage_count, 1)
    usage = encoding.usage_count or 0
    success = encoding.success_count or 0
    if usage > 0:
        effectiveness = (success / usage) * 0.4
    elif success > 0:
        effectiveness = 0.4  # Never tested but has matches somehow
    else:
        effectiveness = 0.1  # Never used — low value

    # Recency: penalize encodings that haven't matched in a long time
    if encoding.last_matched:
        days_since = (datetime.now(timezone.utc) - encoding.last_matched).total_seconds() / 86400
        recency = max(0.0, 1.0 - days_since / 90.0) * 0.2  # Decays over 90 days
    else:
        recency = 0.05  # Never matched — very low recency

    return round(quality_score + effectiveness + recency, 3)


# ── Diversity Analysis ──────────────────────────────────────────

@dataclass
class SuggestionResult:
    """Result of a diversity evaluation."""
    should_suggest: bool
    reason: str = ""
    pose_bucket: str = "frontal"
    appearance_tags: List[str] = field(default_factory=list)


@dataclass
class ReplacementCandidate:
    """An encoding that could be replaced by a better suggestion."""
    encoding_id: str
    health_score: float
    quality: float
    pose_bucket: str
    usage_count: int
    success_count: int


class EmbeddingSuggestionService:
    """
    Evaluates whether a new face capture adds diversity to a person's
    embedding set, and manages the suggestion queue.
    """

    @staticmethod
    def evaluate_suggestion(
        person_id: str,
        new_encoding: np.ndarray,
        quality: float,
        yaw: float = 0.0,
        pitch: float = 0.0,
        blur_score: float = 0.0,
        lighting: Optional[str] = None,
        appearance_tags: Optional[List[str]] = None,
    ) -> SuggestionResult:
        """
        Evaluate whether a new encoding adds diversity to a person's profile.

        Pipeline: Duplicate Filter → Pose Bucket Check → Quality Check →
                  Sharpness Check → Appearance Check

        Only reaches the DB if it passes all diversity checks.
        """
        db: Session = SessionLocal()
        try:
            new_bucket = classify_pose_bucket(yaw)
            new_tags = appearance_tags or detect_appearance_tags(lighting=lighting)

            # Load existing encodings for this person
            existing = (
                db.query(FaceEncoding)
                .filter(FaceEncoding.person_id == person_id)
                .all()
            )

            if not existing:
                return SuggestionResult(
                    should_suggest=True,
                    reason="First additional embedding",
                    pose_bucket=new_bucket,
                    appearance_tags=new_tags,
                )

            # ── Step 1: Duplicate filter ─────────────────────────
            # Compare against representative embeddings per pose bucket
            # (not ALL embeddings — optimization per user request #4)
            bucket_representatives: Dict[str, Tuple[np.ndarray, FaceEncoding]] = {}
            all_qualities = []
            all_blur_scores = []
            covered_buckets = set()

            for enc in existing:
                arr = deserialize_encoding(enc.encoding_data)
                if arr is None:
                    continue

                enc_bucket = enc.pose_bucket or classify_pose_bucket(enc.yaw)
                covered_buckets.add(enc_bucket)
                all_qualities.append(enc.quality or 0.5)
                all_blur_scores.append(enc.blur_score or 0.0)

                # Keep the highest-quality representative per bucket
                if enc_bucket not in bucket_representatives:
                    bucket_representatives[enc_bucket] = (arr, enc)
                else:
                    _, existing_rep = bucket_representatives[enc_bucket]
                    if (enc.quality or 0.5) > (existing_rep.quality or 0.5):
                        bucket_representatives[enc_bucket] = (arr, enc)

            # Normalize new encoding
            norm = np.linalg.norm(new_encoding)
            if norm > 0:
                new_norm = new_encoding / norm
            else:
                return SuggestionResult(should_suggest=False, reason="Zero-norm encoding")

            # Check cosine distance against each bucket representative
            for bucket, (rep_enc, rep_model) in bucket_representatives.items():
                rep_norm = np.linalg.norm(rep_enc)
                if rep_norm > 0:
                    rep_normalized = rep_enc / rep_norm
                    distance = float(1.0 - np.dot(new_norm, rep_normalized))
                    if distance < settings.SUGGESTION_DUPLICATE_THRESHOLD:
                        return SuggestionResult(
                            should_suggest=False,
                            reason=f"Too similar to existing {bucket} encoding (dist={distance:.3f})",
                        )

            # Also check pending suggestions to avoid duplicate suggestions
            pending = (
                db.query(SuggestedEncoding)
                .filter(
                    SuggestedEncoding.person_id == person_id,
                    SuggestedEncoding.status == "pending",
                )
                .all()
            )
            for sug in pending:
                sug_arr = deserialize_encoding(sug.encoding_data)
                if sug_arr is not None:
                    sug_norm = np.linalg.norm(sug_arr)
                    if sug_norm > 0:
                        distance = float(1.0 - np.dot(new_norm, sug_arr / sug_norm))
                        if distance < settings.SUGGESTION_DUPLICATE_THRESHOLD:
                            return SuggestionResult(
                                should_suggest=False,
                                reason="Too similar to a pending suggestion",
                            )

            # ── Step 2: Pose bucket coverage ─────────────────────
            if new_bucket not in covered_buckets:
                bucket_label = new_bucket.replace("_", " ").title()
                return SuggestionResult(
                    should_suggest=True,
                    reason=f"New {bucket_label} profile",
                    pose_bucket=new_bucket,
                    appearance_tags=new_tags,
                )

            # ── Step 3: Quality improvement ──────────────────────
            max_quality = max(all_qualities) if all_qualities else 0.0
            if quality > max_quality + settings.SUGGESTION_QUALITY_MARGIN:
                return SuggestionResult(
                    should_suggest=True,
                    reason=f"Better quality ({quality:.2f} vs best {max_quality:.2f})",
                    pose_bucket=new_bucket,
                    appearance_tags=new_tags,
                )

            # ── Step 4: Sharpness improvement ────────────────────
            max_blur = max(all_blur_scores) if all_blur_scores else 0.0
            if max_blur > 0 and blur_score > max_blur * settings.SUGGESTION_SHARPNESS_MARGIN:
                return SuggestionResult(
                    should_suggest=True,
                    reason=f"Higher sharpness ({blur_score:.0f} vs best {max_blur:.0f})",
                    pose_bucket=new_bucket,
                    appearance_tags=new_tags,
                )

            # ── Step 5: Appearance diversity ─────────────────────
            existing_tags = set()
            for enc in existing:
                if enc.appearance_tags:
                    existing_tags.update(enc.appearance_tags.split(","))

            new_unique_tags = [t for t in new_tags if t not in existing_tags]
            if new_unique_tags:
                return SuggestionResult(
                    should_suggest=True,
                    reason=f"Different appearance ({', '.join(new_unique_tags)})",
                    pose_bucket=new_bucket,
                    appearance_tags=new_tags,
                )

            # ── Step 6: Frontal improvement ──────────────────────
            if new_bucket == "frontal" and abs(yaw) < 10 and abs(pitch) < 10:
                # Check if we can improve the existing frontal
                if "frontal" in bucket_representatives:
                    _, frontal_enc = bucket_representatives["frontal"]
                    if quality > (frontal_enc.quality or 0.5):
                        return SuggestionResult(
                            should_suggest=True,
                            reason="Improved frontal image",
                            pose_bucket=new_bucket,
                            appearance_tags=new_tags,
                        )

            # No diversity gain
            return SuggestionResult(
                should_suggest=False,
                reason="No additional diversity — existing embeddings already cover this",
            )

        finally:
            db.close()

    @staticmethod
    def create_suggestion(
        person_id: str,
        encoding: np.ndarray,
        snapshot_url: str,
        camera_id: Optional[str],
        quality: float,
        yaw: float,
        pitch: float,
        roll: float,
        det_score: float,
        blur_score: float,
        lighting: Optional[str],
        similarity: float,
        reason: str,
        pose_bucket: str,
        appearance_tags: List[str],
    ) -> dict:
        """Create a suggested encoding (already passed diversity check)."""
        db: Session = SessionLocal()
        try:
            sug = SuggestedEncoding(
                id=str(uuid4()),
                person_id=person_id,
                encoding_data=serialize_encoding(encoding),
                snapshot_url=snapshot_url,
                camera_id=camera_id,
                quality=quality,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                det_score=det_score,
                blur_score=blur_score,
                lighting=lighting,
                pose_bucket=pose_bucket,
                appearance_tags=",".join(appearance_tags) if appearance_tags else None,
                similarity=similarity,
                reason=reason,
                status="pending",
                created_at=datetime.now(timezone.utc),
            )
            db.add(sug)
            db.commit()

            logger.info(
                f"📝 Suggestion queued for {person_id[:8]}…: "
                f"reason=\"{reason}\", bucket={pose_bucket}, quality={quality:.2f}"
            )

            return {
                "id": sug.id,
                "person_id": person_id,
                "reason": reason,
                "quality": quality,
                "pose_bucket": pose_bucket,
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create suggestion: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def get_suggestions(person_id: str, status: str = "pending") -> List[dict]:
        """Get suggestions for a person."""
        db: Session = SessionLocal()
        try:
            query = (
                db.query(SuggestedEncoding)
                .filter(SuggestedEncoding.person_id == person_id)
            )
            if status:
                query = query.filter(SuggestedEncoding.status == status)

            results = query.order_by(SuggestedEncoding.created_at.desc()).all()

            return [
                {
                    "id": s.id,
                    "person_id": s.person_id,
                    "snapshot_url": s.snapshot_url,
                    "camera_id": s.camera_id,
                    "quality": s.quality,
                    "yaw": s.yaw,
                    "pitch": s.pitch,
                    "roll": s.roll,
                    "det_score": s.det_score,
                    "blur_score": s.blur_score,
                    "lighting": s.lighting,
                    "pose_bucket": s.pose_bucket,
                    "appearance_tags": s.appearance_tags.split(",") if s.appearance_tags else [],
                    "similarity": s.similarity,
                    "reason": s.reason,
                    "status": s.status,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in results
            ]
        finally:
            db.close()

    @staticmethod
    def get_suggestion_count(person_id: str) -> int:
        """Count pending suggestions for a person."""
        db: Session = SessionLocal()
        try:
            return (
                db.query(SuggestedEncoding)
                .filter(
                    SuggestedEncoding.person_id == person_id,
                    SuggestedEncoding.status == "pending",
                )
                .count()
            )
        finally:
            db.close()

    @staticmethod
    def approve_suggestion(
        suggestion_id: str,
        replace_encoding_id: Optional[str] = None,
    ) -> dict:
        """
        Approve a suggestion → creates a FaceEncoding.

        If replace_encoding_id is provided, deletes that encoding first
        (for cap enforcement). If the cap is reached and no replacement
        is specified, returns the weakest candidates for the admin to choose.
        """
        from app.services.person_service import PersonService

        db: Session = SessionLocal()
        try:
            sug = db.query(SuggestedEncoding).filter(SuggestedEncoding.id == suggestion_id).first()
            if not sug:
                raise ValueError(f"Suggestion {suggestion_id} not found")
            if sug.status != "pending":
                raise ValueError(f"Suggestion already {sug.status}")

            person_id = sug.person_id

            # Check encoding count
            current_count = (
                db.query(FaceEncoding)
                .filter(FaceEncoding.person_id == person_id)
                .count()
            )

            cap = settings.MAX_ENCODINGS_PER_PERSON

            if current_count >= cap and not replace_encoding_id:
                # Cap reached — return replacement candidates
                candidates = EmbeddingSuggestionService._get_replacement_candidates(
                    db, person_id
                )
                return {
                    "status": "cap_reached",
                    "current_count": current_count,
                    "cap": cap,
                    "replacement_candidates": candidates,
                    "message": f"Person has {current_count}/{cap} encodings. Choose one to replace.",
                }

            # If replacing, delete the old encoding
            if replace_encoding_id:
                old_enc = (
                    db.query(FaceEncoding)
                    .filter(
                        FaceEncoding.id == replace_encoding_id,
                        FaceEncoding.person_id == person_id,
                    )
                    .first()
                )
                if old_enc:
                    db.delete(old_enc)
                    logger.info(f"Replaced encoding {replace_encoding_id[:8]}… for person {person_id[:8]}…")

            # Create the new FaceEncoding
            encoding_arr = deserialize_encoding(sug.encoding_data)
            if encoding_arr is None:
                raise ValueError("Failed to deserialize suggestion encoding")

            # Get current version for the new encoding
            current_version = PersonService._get_or_create_version(db)
            new_version = current_version + 1

            new_enc = FaceEncoding(
                id=str(uuid4()),
                person_id=person_id,
                encoding_data=sug.encoding_data,  # Already serialized
                source_image=sug.snapshot_url,
                quality=sug.quality,
                yaw=sug.yaw,
                pitch=sug.pitch,
                roll=sug.roll,
                det_score=sug.det_score,
                blur_score=sug.blur_score,
                camera_id=sug.camera_id,
                lighting=sug.lighting,
                pose_bucket=sug.pose_bucket,
                appearance_tags=sug.appearance_tags,
                model_version="arcface_r50",
                version=new_version,
                created_at=datetime.now(timezone.utc),
            )
            db.add(new_enc)

            # Update suggestion status
            sug.status = "approved"

            # Increment encoding version
            PersonService._set_version(db, new_version)

            db.commit()

            logger.info(
                f"✅ Approved suggestion {suggestion_id[:8]}… for {person_id[:8]}…: "
                f"reason=\"{sug.reason}\""
            )

            return {
                "status": "approved",
                "encoding_id": new_enc.id,
                "person_id": person_id,
                "reason": sug.reason,
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to approve suggestion: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def reject_suggestion(suggestion_id: str) -> dict:
        """Reject a suggestion."""
        db: Session = SessionLocal()
        try:
            sug = db.query(SuggestedEncoding).filter(SuggestedEncoding.id == suggestion_id).first()
            if not sug:
                raise ValueError(f"Suggestion {suggestion_id} not found")

            sug.status = "rejected"
            db.commit()

            return {"status": "rejected", "id": suggestion_id}
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def get_person_encodings(person_id: str) -> List[dict]:
        """Get all encodings for a person with full metadata + health scores."""
        db: Session = SessionLocal()
        try:
            encodings = (
                db.query(FaceEncoding)
                .filter(FaceEncoding.person_id == person_id)
                .order_by(FaceEncoding.created_at.desc())
                .all()
            )

            return [
                {
                    "id": enc.id,
                    "person_id": enc.person_id,
                    "source_image": enc.source_image,
                    "quality": enc.quality,
                    "yaw": enc.yaw,
                    "pitch": enc.pitch,
                    "roll": enc.roll,
                    "det_score": enc.det_score,
                    "blur_score": enc.blur_score,
                    "camera_id": enc.camera_id,
                    "lighting": enc.lighting,
                    "pose_bucket": enc.pose_bucket or classify_pose_bucket(enc.yaw),
                    "appearance_tags": enc.appearance_tags.split(",") if enc.appearance_tags else [],
                    "usage_count": enc.usage_count or 0,
                    "success_count": enc.success_count or 0,
                    "last_matched": enc.last_matched.isoformat() if enc.last_matched else None,
                    "health_score": compute_health_score(enc),
                    "created_at": enc.created_at.isoformat() if enc.created_at else None,
                }
                for enc in encodings
            ]
        finally:
            db.close()

    @staticmethod
    def delete_encoding(person_id: str, encoding_id: str) -> dict:
        """Delete a specific encoding."""
        from app.services.person_service import PersonService

        db: Session = SessionLocal()
        try:
            enc = (
                db.query(FaceEncoding)
                .filter(
                    FaceEncoding.id == encoding_id,
                    FaceEncoding.person_id == person_id,
                )
                .first()
            )
            if not enc:
                raise ValueError(f"Encoding {encoding_id} not found for person {person_id}")

            db.delete(enc)

            # Increment version
            current_version = PersonService._get_or_create_version(db)
            PersonService._set_version(db, current_version + 1)

            db.commit()

            return {"status": "deleted", "encoding_id": encoding_id}
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _get_replacement_candidates(
        db: Session, person_id: str, count: int = 3
    ) -> List[dict]:
        """
        Get the weakest encodings for a person — candidates for replacement.

        Ranked by health score (lowest first). Health = quality * usage effectiveness * recency.
        """
        encodings = (
            db.query(FaceEncoding)
            .filter(FaceEncoding.person_id == person_id)
            .all()
        )

        scored = []
        for enc in encodings:
            scored.append({
                "encoding_id": enc.id,
                "health_score": compute_health_score(enc),
                "quality": enc.quality or 0.5,
                "pose_bucket": enc.pose_bucket or classify_pose_bucket(enc.yaw),
                "usage_count": enc.usage_count or 0,
                "success_count": enc.success_count or 0,
                "source_image": enc.source_image,
                "created_at": enc.created_at.isoformat() if enc.created_at else None,
            })

        # Sort by health score ascending (weakest first)
        scored.sort(key=lambda x: x["health_score"])
        return scored[:count]
