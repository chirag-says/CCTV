"""
Identity Review Service — Business logic for AI identity review operations.

Handles:
  - Candidate search (find nearest known persons for a cluster)
  - Confidence fusion (combine embedding, verifier, VLM scores)
  - Approval workflow (approve → gallery update → pipeline refresh)
  - Activity logging (audit trail)
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.db.models import (
    UnknownCluster, IdentityReviewCandidate, GalleryUpdate,
    IdentityReviewActivity, UnknownFace, Person, FaceEncoding,
)
from app.vision.recognizer import deserialize_encoding, serialize_encoding

logger = logging.getLogger(__name__)


class IdentityReviewService:
    """Business logic for the AI Identity Review system."""

    # ── Candidate Search ─────────────────────────────────────────

    @staticmethod
    def find_candidates_for_cluster(
        cluster_id: str,
        centroid: np.ndarray,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the Top-K nearest known persons for a cluster centroid.

        Uses numpy cosine similarity (no pgvector dependency).
        Returns sorted list of candidates with similarity scores.
        """
        top_k = top_k or settings.IDENTITY_REVIEW_TOP_K
        db = SessionLocal()
        try:
            # Load all known person encodings
            encodings = (
                db.query(FaceEncoding)
                .filter(FaceEncoding.encoding_data.isnot(None))
                .all()
            )

            if not encodings:
                return []

            # Normalize centroid
            norm = np.linalg.norm(centroid)
            if norm == 0:
                return []
            centroid_normalized = centroid / norm

            # Compute similarities
            candidates = {}
            for enc_row in encodings:
                enc = deserialize_encoding(enc_row.encoding_data)
                if enc is None:
                    continue
                enc_norm = np.linalg.norm(enc)
                if enc_norm == 0:
                    continue
                similarity = float(np.dot(centroid_normalized, enc / enc_norm))

                pid = enc_row.person_id
                # Keep the best similarity per person
                if pid not in candidates or similarity > candidates[pid]["similarity"]:
                    candidates[pid] = {
                        "person_id": pid,
                        "similarity": similarity,
                        "encoding_id": enc_row.id,
                        "source_image": enc_row.source_image,
                        "quality": enc_row.quality,
                    }

            # Sort by similarity descending, take top K
            sorted_candidates = sorted(
                candidates.values(),
                key=lambda c: c["similarity"],
                reverse=True,
            )[:top_k]

            # Enrich with person names
            for cand in sorted_candidates:
                person = db.query(Person).filter(Person.id == cand["person_id"]).first()
                if person:
                    cand["person_name"] = person.name
                    cand["avatar_url"] = person.avatar_url
                    cand["department"] = person.department

            return sorted_candidates

        finally:
            db.close()

    # ── Confidence Fusion ────────────────────────────────────────

    @staticmethod
    def compute_overall_score(
        embedding_similarity: float,
        verifier_similarity: Optional[float] = None,
        vlm_confidence: Optional[float] = None,
        image_quality: float = 0.5,
        pose_quality: float = 0.5,
        cluster_consistency: float = 0.5,
    ) -> float:
        """
        Fuse multiple signals into a single overall confidence score.

        Weights:
          - Embedding similarity: 35%
          - Verifier similarity: 20% (if available)
          - VLM agreement: 20% (if available)
          - Image quality: 10%
          - Pose quality: 5%
          - Cluster consistency: 10%

        If verifier or VLM is unavailable, their weight is redistributed
        to embedding similarity.
        """
        weights = {
            "embedding": 0.35,
            "verifier": 0.20,
            "vlm": 0.20,
            "quality": 0.10,
            "pose": 0.05,
            "cluster": 0.10,
        }

        score = 0.0
        total_weight = 0.0

        # Always-available signals
        score += weights["embedding"] * embedding_similarity
        total_weight += weights["embedding"]

        score += weights["quality"] * image_quality
        total_weight += weights["quality"]

        score += weights["pose"] * pose_quality
        total_weight += weights["pose"]

        score += weights["cluster"] * cluster_consistency
        total_weight += weights["cluster"]

        # Optional signals
        if verifier_similarity is not None:
            score += weights["verifier"] * verifier_similarity
            total_weight += weights["verifier"]

        if vlm_confidence is not None:
            score += weights["vlm"] * vlm_confidence
            total_weight += weights["vlm"]

        # Normalize to account for missing signals
        if total_weight > 0:
            return score / total_weight
        return 0.0

    @staticmethod
    def recommended_action(overall_score: float) -> str:
        """Determine recommended action based on overall score."""
        if overall_score >= 0.75:
            return "approve"
        elif overall_score <= 0.30:
            return "reject"
        return "review"

    # ── Candidate Creation ───────────────────────────────────────

    @staticmethod
    def create_candidate(
        cluster_id: str,
        person_id: str,
        embedding_similarity: float,
        verifier_similarity: Optional[float] = None,
        vlm_confidence: Optional[float] = None,
        vlm_explanation: str = "",
        image_quality_score: float = 0.5,
        pose_quality_score: float = 0.5,
        cluster_consistency: float = 0.5,
        # Gallery Optimizer enrichment
        gallery_improvement_value: str = "none",
        gallery_improvement_reason: str = "",
        coverage_before: float = 0.0,
        coverage_after: float = 0.0,
        vlm_detected_pose: str = "",
        vlm_quality_assessment: str = "",
        review_priority: str = "low",
    ) -> Optional[Dict[str, Any]]:
        """Create a review candidate record with Gallery Optimizer enrichment."""
        db = SessionLocal()
        try:
            # Check for existing candidate for this cluster+person
            existing = (
                db.query(IdentityReviewCandidate)
                .filter(
                    IdentityReviewCandidate.cluster_id == cluster_id,
                    IdentityReviewCandidate.person_id == person_id,
                    IdentityReviewCandidate.status == "pending",
                )
                .first()
            )
            if existing:
                logger.debug(
                    f"Candidate already exists for cluster={cluster_id[:8]}… "
                    f"+ person={person_id[:8]}…"
                )
                return None

            overall = IdentityReviewService.compute_overall_score(
                embedding_similarity=embedding_similarity,
                verifier_similarity=verifier_similarity,
                vlm_confidence=vlm_confidence,
                image_quality=image_quality_score,
                pose_quality=pose_quality_score,
                cluster_consistency=cluster_consistency,
            )

            # Gallery improvement boost/penalty
            if gallery_improvement_value == "high":
                overall = min(1.0, overall + 0.10)
            elif gallery_improvement_value == "none":
                overall = max(0.0, overall - 0.05)

            candidate = IdentityReviewCandidate(
                id=str(uuid4()),
                cluster_id=cluster_id,
                person_id=person_id,
                embedding_similarity=embedding_similarity,
                verifier_similarity=verifier_similarity,
                vlm_confidence=vlm_confidence,
                image_quality_score=image_quality_score,
                pose_quality_score=pose_quality_score,
                cluster_consistency=cluster_consistency,
                overall_score=overall,
                vlm_explanation=vlm_explanation,
                recommended_action=IdentityReviewService.recommended_action(overall),
                status="pending",
                # Gallery Optimizer enrichment
                gallery_improvement_value=gallery_improvement_value,
                gallery_improvement_reason=gallery_improvement_reason,
                coverage_before=coverage_before,
                coverage_after=coverage_after,
                vlm_detected_pose=vlm_detected_pose,
                vlm_quality_assessment=vlm_quality_assessment,
                review_priority=review_priority,
            )
            db.add(candidate)
            db.commit()

            logger.info(
                f"Created review candidate: cluster={cluster_id[:8]}… → "
                f"person={person_id[:8]}… (overall={overall:.3f}, "
                f"priority={review_priority}, "
                f"improvement={gallery_improvement_value}, "
                f"action={candidate.recommended_action})"
            )

            return {
                "id": candidate.id,
                "cluster_id": cluster_id,
                "person_id": person_id,
                "overall_score": overall,
                "recommended_action": candidate.recommended_action,
                "review_priority": review_priority,
                "gallery_improvement_value": gallery_improvement_value,
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create candidate: {e}", exc_info=True)
            return None
        finally:
            db.close()

    # ── Review Actions ───────────────────────────────────────────

    @staticmethod
    def approve_candidate(
        candidate_id: str,
        actor_id: str,
    ) -> Dict[str, Any]:
        """
        Approve a candidate — creates a GalleryUpdate in pending state.
        Does NOT immediately modify the face gallery.
        """
        db = SessionLocal()
        try:
            candidate = (
                db.query(IdentityReviewCandidate)
                .filter(IdentityReviewCandidate.id == candidate_id)
                .first()
            )
            if not candidate:
                raise ValueError(f"Candidate {candidate_id} not found")
            if candidate.status != "pending":
                raise ValueError(f"Candidate already {candidate.status}")

            # Get cluster centroid encoding for gallery update
            cluster = (
                db.query(UnknownCluster)
                .filter(UnknownCluster.id == candidate.cluster_id)
                .first()
            )

            # Mark candidate as approved
            candidate.status = "approved"
            candidate.reviewed_by = actor_id
            candidate.reviewed_at = datetime.now(timezone.utc)

            # Create gallery update
            gallery_update = GalleryUpdate(
                id=str(uuid4()),
                person_id=candidate.person_id,
                candidate_id=candidate.id,
                cluster_id=candidate.cluster_id,
                encoding_data=cluster.embedding_centroid if cluster else None,
                snapshot_url=cluster.representative_snapshot if cluster else None,
                quality=candidate.image_quality_score,
                action="append",
                status="pending",
            )
            db.add(gallery_update)

            # Log activity
            activity = IdentityReviewActivity(
                id=str(uuid4()),
                candidate_id=candidate_id,
                cluster_id=candidate.cluster_id,
                action="approved",
                actor=actor_id,
                details={
                    "person_id": candidate.person_id,
                    "overall_score": candidate.overall_score,
                },
            )
            db.add(activity)

            db.commit()

            return {
                "status": "approved",
                "gallery_update_id": gallery_update.id,
                "message": f"Candidate approved. Gallery update pending.",
            }

        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def reject_candidate(candidate_id: str, actor_id: str) -> Dict[str, Any]:
        """Reject a candidate."""
        db = SessionLocal()
        try:
            candidate = (
                db.query(IdentityReviewCandidate)
                .filter(IdentityReviewCandidate.id == candidate_id)
                .first()
            )
            if not candidate:
                raise ValueError(f"Candidate {candidate_id} not found")

            candidate.status = "rejected"
            candidate.reviewed_by = actor_id
            candidate.reviewed_at = datetime.now(timezone.utc)

            activity = IdentityReviewActivity(
                id=str(uuid4()),
                candidate_id=candidate_id,
                cluster_id=candidate.cluster_id,
                action="rejected",
                actor=actor_id,
            )
            db.add(activity)
            db.commit()

            return {"status": "rejected"}
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def skip_candidate(candidate_id: str, actor_id: str) -> Dict[str, Any]:
        """Skip a candidate for later review."""
        db = SessionLocal()
        try:
            candidate = (
                db.query(IdentityReviewCandidate)
                .filter(IdentityReviewCandidate.id == candidate_id)
                .first()
            )
            if not candidate:
                raise ValueError(f"Candidate {candidate_id} not found")

            candidate.status = "skipped"
            candidate.reviewed_by = actor_id
            candidate.reviewed_at = datetime.now(timezone.utc)

            activity = IdentityReviewActivity(
                id=str(uuid4()),
                candidate_id=candidate_id,
                cluster_id=candidate.cluster_id,
                action="skipped",
                actor=actor_id,
            )
            db.add(activity)
            db.commit()

            return {"status": "skipped"}
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    # ── Gallery Update Application ───────────────────────────────

    @staticmethod
    def apply_gallery_update(update_id: str) -> Dict[str, Any]:
        """
        Apply a pending gallery update — appends encoding to face_encodings.
        Refreshes pipeline encoding cache after successful application.
        """
        db = SessionLocal()
        try:
            update = (
                db.query(GalleryUpdate)
                .filter(GalleryUpdate.id == update_id)
                .first()
            )
            if not update:
                raise ValueError(f"Gallery update {update_id} not found")
            if update.status != "pending":
                raise ValueError(f"Gallery update already {update.status}")

            if not update.encoding_data:
                raise ValueError("No encoding data to apply")

            # Create new face encoding
            from app.services.person_service import PersonService
            encoding = deserialize_encoding(update.encoding_data)
            if encoding is not None:
                PersonService.add_face_encoding(
                    person_id=update.person_id,
                    encoding=encoding,
                    source_image=update.snapshot_url,
                    quality=update.quality or 0.7,
                )

            update.status = "applied"
            update.applied_at = datetime.now(timezone.utc)
            db.commit()

            # Reload pipeline encodings
            try:
                from app.vision.camera_worker import camera_manager
                all_encodings = PersonService.get_all_encodings()
                camera_manager.load_encodings_all(all_encodings)
                logger.info(
                    f"Gallery update {update_id[:8]}… applied and "
                    f"pipeline encodings refreshed"
                )
            except Exception as e:
                logger.warning(f"Pipeline refresh after gallery update failed: {e}")

            return {
                "status": "applied",
                "person_id": update.person_id,
            }

        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def rollback_gallery_update(update_id: str) -> Dict[str, Any]:
        """Rollback an applied gallery update."""
        db = SessionLocal()
        try:
            update = (
                db.query(GalleryUpdate)
                .filter(GalleryUpdate.id == update_id)
                .first()
            )
            if not update:
                raise ValueError(f"Gallery update {update_id} not found")
            if update.status != "applied":
                raise ValueError(f"Can only rollback applied updates, current: {update.status}")

            update.status = "rolled_back"
            db.commit()

            logger.info(f"Gallery update {update_id[:8]}… rolled back")
            return {"status": "rolled_back"}

        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    # ── Dashboard Stats ──────────────────────────────────────────

    @staticmethod
    def get_dashboard_stats() -> Dict[str, Any]:
        """Get stats for the identity review dashboard."""
        db = SessionLocal()
        try:
            unknown_count = db.query(UnknownFace).filter(UnknownFace.status == "pending").count()
            cluster_count = db.query(UnknownCluster).filter(UnknownCluster.status == "active").count()
            pending_reviews = db.query(IdentityReviewCandidate).filter(
                IdentityReviewCandidate.status == "pending"
            ).count()

            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            approved_today = db.query(IdentityReviewCandidate).filter(
                IdentityReviewCandidate.status == "approved",
                IdentityReviewCandidate.reviewed_at >= today,
            ).count()
            rejected_today = db.query(IdentityReviewCandidate).filter(
                IdentityReviewCandidate.status == "rejected",
                IdentityReviewCandidate.reviewed_at >= today,
            ).count()

            pending_updates = db.query(GalleryUpdate).filter(
                GalleryUpdate.status == "pending"
            ).count()
            applied_updates = db.query(GalleryUpdate).filter(
                GalleryUpdate.status == "applied"
            ).count()

            return {
                "unknown_faces": unknown_count,
                "active_clusters": cluster_count,
                "pending_reviews": pending_reviews,
                "approved_today": approved_today,
                "rejected_today": rejected_today,
                "pending_gallery_updates": pending_updates,
                "gallery_improvements": applied_updates,
            }
        finally:
            db.close()
