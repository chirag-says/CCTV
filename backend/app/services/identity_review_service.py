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
    def preview_candidate(
        candidate_id: str,
        selected_face_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Dry-run an approval: which cluster members would reach the gallery,
        which would not and why, and the resulting coverage.

        Changes nothing. Runs the identical code path as approve_candidate,
        then rolls back — so the numbers shown to the reviewer are measured,
        not estimated.
        """
        from app.services.gallery_selection import evaluate_candidate_members

        report = evaluate_candidate_members(
            candidate_id=candidate_id,
            selected_face_ids=selected_face_ids,
            apply=False,
        )
        return report.to_dict()

    @staticmethod
    def approve_candidate(
        candidate_id: str,
        actor_id: str,
        selected_face_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Approve a candidate — adds the SELECTED cluster members' individual
        embeddings to the person's gallery, each one gated for quality,
        duplication, pose coverage and capacity.

        The human approves the identity; the gates decide which of that
        identity's images the gallery actually needs. A member can therefore
        be approved as this person and still not be added — reported back as
        a rejection reason rather than silently dropped.

        `selected_face_ids` defaults to every member except detected outliers.
        """
        from app.services.gallery_selection import evaluate_candidate_members

        report = evaluate_candidate_members(
            candidate_id=candidate_id,
            selected_face_ids=selected_face_ids,
            apply=True,
            actor_id=actor_id,
        )
        result = report.to_dict()
        result["status"] = "approved"
        result["message"] = (
            f"Approved as {report.person_name}. "
            f"{len(report.accepted)} embedding(s) added, "
            f"{len(report.rejected)} not added. "
            f"Coverage {report.coverage_before:.1f}% → {report.coverage_after:.1f}%."
        )
        return result

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
        """
        Roll back an applied gallery update.

        Member-level updates record the face_encodings row they created, so
        rollback removes that encoding and returns the source unknown face to
        the pending pool. Legacy centroid-era rows carry no encoding_id; those
        are marked rolled_back without deleting anything, since there is no
        record of which encoding they produced.
        """
        from app.services.person_service import PersonService

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

            removed_encoding = False
            if update.encoding_id:
                enc = (
                    db.query(FaceEncoding)
                    .filter(FaceEncoding.id == update.encoding_id)
                    .first()
                )
                if enc:
                    db.delete(enc)
                    removed_encoding = True

            if update.source_face_id:
                (
                    db.query(UnknownFace)
                    .filter(UnknownFace.id == update.source_face_id)
                    .update({"status": "pending"}, synchronize_session=False)
                )

            update.status = "rolled_back"

            if removed_encoding:
                PersonService._set_version(
                    db, PersonService._get_or_create_version(db) + 1
                )

            db.add(IdentityReviewActivity(
                id=str(uuid4()),
                candidate_id=update.candidate_id,
                cluster_id=update.cluster_id,
                action="rolled_back",
                details={
                    "gallery_update_id": update.id,
                    "person_id": update.person_id,
                    "gallery_version": update.gallery_version,
                    "encoding_id": update.encoding_id,
                    "source_face_id": update.source_face_id,
                    "encoding_removed": removed_encoding,
                },
            ))
            db.commit()

            if removed_encoding:
                from app.services.gallery_selection import _refresh_pipeline_encodings
                _refresh_pipeline_encodings()

            logger.info(
                f"Gallery update {update_id[:8]}… rolled back "
                f"(encoding_removed={removed_encoding})"
            )
            return {"status": "rolled_back", "encoding_removed": removed_encoding}

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
