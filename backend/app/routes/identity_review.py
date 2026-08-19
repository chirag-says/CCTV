"""
API routes for AI Identity Review.

Provides endpoints for:
  - Dashboard stats
  - Review candidate listing and detail
  - Cluster listing and detail
  - Approval/rejection/skip actions
  - Gallery update management
  - Activity timeline
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.db.session import SessionLocal
from app.db.models import (
    UnknownCluster, IdentityReviewCandidate, GalleryUpdate,
    IdentityReviewActivity, UnknownFace, Person,
)
from app.services.identity_review_service import IdentityReviewService
from app.routes.auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/identity-review", tags=["Identity Review"])


# ── Request/Response Models ──────────────────────────────────────

class ReviewActionRequest(BaseModel):
    """Request body for approve/reject/skip actions."""
    pass  # actor_id comes from auth


class IgnoreClusterRequest(BaseModel):
    """Request body for ignoring a cluster."""
    reason: Optional[str] = None


class CreatePersonFromClusterRequest(BaseModel):
    """Request body for creating a new person from a cluster."""
    name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(default="visitor")
    department: Optional[str] = None


class MergeClustersRequest(BaseModel):
    """Request body for merging clusters."""
    target_cluster_id: str


# ── Dashboard ────────────────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard(user=Depends(require_admin)):
    """Get identity review dashboard statistics with priority breakdown."""
    try:
        stats = IdentityReviewService.get_dashboard_stats()

        # Add priority breakdown
        db = SessionLocal()
        try:
            from sqlalchemy import func
            priority_rows = (
                db.query(
                    IdentityReviewCandidate.review_priority,
                    func.count(IdentityReviewCandidate.id),
                )
                .filter(IdentityReviewCandidate.status == "pending")
                .group_by(IdentityReviewCandidate.review_priority)
                .all()
            )
            priority_breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for priority, count in priority_rows:
                if priority in priority_breakdown:
                    priority_breakdown[priority] = count
            stats["priority_breakdown"] = priority_breakdown
        finally:
            db.close()

        return stats
    except Exception as e:
        logger.error(f"Dashboard stats failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dashboard stats")


# ── Candidates ───────────────────────────────────────────────────

@router.get("/candidates")
async def list_candidates(
    status: Optional[str] = Query("pending", description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority (critical/high/medium/low)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user=Depends(require_admin),
):
    """List review candidates with filtering and pagination."""
    db = SessionLocal()
    try:
        query = db.query(IdentityReviewCandidate)

        if status:
            query = query.filter(IdentityReviewCandidate.status == status)
        if priority:
            query = query.filter(IdentityReviewCandidate.review_priority == priority)

        total = query.count()

        candidates = (
            query
            .order_by(
                # Critical first, then high, medium, low
                IdentityReviewCandidate.review_priority.asc(),
                IdentityReviewCandidate.overall_score.desc(),
            )
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = []
        for c in candidates:
            person = db.query(Person).filter(Person.id == c.person_id).first()
            cluster = db.query(UnknownCluster).filter(UnknownCluster.id == c.cluster_id).first()

            items.append({
                "id": c.id,
                "cluster_id": c.cluster_id,
                "person_id": c.person_id,
                "person_name": person.name if person else "Unknown",
                "person_avatar": person.avatar_url if person else None,
                "cluster_snapshot": cluster.representative_snapshot if cluster else None,
                "cluster_member_count": cluster.member_count if cluster else 0,
                "embedding_similarity": c.embedding_similarity,
                "verifier_similarity": c.verifier_similarity,
                "vlm_confidence": c.vlm_confidence,
                "overall_score": c.overall_score,
                "vlm_explanation": c.vlm_explanation,
                "recommended_action": c.recommended_action,
                "status": c.status,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                # Gallery Optimizer enrichment
                "gallery_improvement_value": c.gallery_improvement_value,
                "gallery_improvement_reason": c.gallery_improvement_reason,
                "coverage_before": c.coverage_before,
                "coverage_after": c.coverage_after,
                "vlm_detected_pose": c.vlm_detected_pose,
                "vlm_quality_assessment": c.vlm_quality_assessment,
                "review_priority": c.review_priority,
            })

        return {"items": items, "total": total, "limit": limit, "offset": offset}

    finally:
        db.close()


@router.get("/candidates/{candidate_id}")
async def get_candidate_detail(candidate_id: str, user=Depends(require_admin)):
    """Get full candidate detail with cluster images and gallery images."""
    db = SessionLocal()
    try:
        candidate = (
            db.query(IdentityReviewCandidate)
            .filter(IdentityReviewCandidate.id == candidate_id)
            .first()
        )
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Get cluster info and member images
        cluster = db.query(UnknownCluster).filter(UnknownCluster.id == candidate.cluster_id).first()
        cluster_members = []
        if cluster:
            faces = (
                db.query(UnknownFace)
                .filter(UnknownFace.cluster_id == cluster.id)
                .order_by(UnknownFace.last_seen.desc())
                .limit(20)
                .all()
            )
            cluster_members = [
                {
                    "id": f.id,
                    "snapshot_url": f.snapshot_url,
                    "occurrence": f.occurrence,
                    "first_seen": f.first_seen.isoformat() if f.first_seen else None,
                    "last_seen": f.last_seen.isoformat() if f.last_seen else None,
                    "camera_id": f.camera_id,
                }
                for f in faces
            ]

        # Get person info and gallery images
        person = db.query(Person).filter(Person.id == candidate.person_id).first()
        gallery_images = []
        if person:
            from app.db.models import FaceEncoding
            encodings = (
                db.query(FaceEncoding)
                .filter(FaceEncoding.person_id == person.id)
                .order_by(FaceEncoding.quality.desc())
                .all()
            )
            gallery_images = [
                {
                    "id": e.id,
                    "source_image": e.source_image,
                    "quality": e.quality,
                    "pose_bucket": e.pose_bucket,
                }
                for e in encodings
            ]

        # Compute PersonProfile (Gallery Coach data)
        person_profile = None
        if person:
            try:
                from app.services.gallery_health import compute_person_profile
                profile = compute_person_profile(person.id, db=db)
                person_profile = {
                    "health_score": profile.health_score,
                    "coverage_percentage": profile.coverage_percentage,
                    "covered_buckets": profile.covered_buckets,
                    "missing_buckets": profile.missing_buckets,
                    "total_encodings": profile.total_encodings,
                    "quality_score": profile.quality_score,
                    "recognition_score": profile.recognition_score,
                    "top_recommendation": profile.top_recommendation,
                    "encoding_health": [
                        {
                            "id": eh.encoding_id,
                            "pose_bucket": eh.pose_bucket,
                            "quality": eh.quality,
                            "usage_count": eh.usage_count,
                            "success_count": eh.success_count,
                            "health": eh.health,
                        }
                        for eh in profile.encoding_health
                    ],
                }
            except Exception as e:
                logger.debug(f"PersonProfile computation failed: {e}")

        return {
            "id": candidate.id,
            "status": candidate.status,
            "overall_score": candidate.overall_score,
            "embedding_similarity": candidate.embedding_similarity,
            "verifier_similarity": candidate.verifier_similarity,
            "vlm_confidence": candidate.vlm_confidence,
            "image_quality_score": candidate.image_quality_score,
            "pose_quality_score": candidate.pose_quality_score,
            "cluster_consistency": candidate.cluster_consistency,
            "vlm_explanation": candidate.vlm_explanation,
            "recommended_action": candidate.recommended_action,
            "reviewed_by": candidate.reviewed_by,
            "reviewed_at": candidate.reviewed_at.isoformat() if candidate.reviewed_at else None,
            "created_at": candidate.created_at.isoformat() if candidate.created_at else None,
            # Gallery Optimizer enrichment
            "gallery_improvement_value": candidate.gallery_improvement_value,
            "gallery_improvement_reason": candidate.gallery_improvement_reason,
            "coverage_before": candidate.coverage_before,
            "coverage_after": candidate.coverage_after,
            "vlm_detected_pose": candidate.vlm_detected_pose,
            "vlm_quality_assessment": candidate.vlm_quality_assessment,
            "review_priority": candidate.review_priority,
            "cluster": {
                "id": cluster.id if cluster else None,
                "member_count": cluster.member_count if cluster else 0,
                "representative_snapshot": cluster.representative_snapshot if cluster else None,
                "first_seen": cluster.first_seen.isoformat() if cluster and cluster.first_seen else None,
                "last_seen": cluster.last_seen.isoformat() if cluster and cluster.last_seen else None,
                "members": cluster_members,
            },
            "person": {
                "id": person.id if person else None,
                "name": person.name if person else "Unknown",
                "avatar_url": person.avatar_url if person else None,
                "role": person.role if person else None,
                "department": person.department if person else None,
                "gallery_images": gallery_images,
                "profile": person_profile,
            },
        }

    finally:
        db.close()


# ── Review Actions ───────────────────────────────────────────────

@router.post("/candidates/{candidate_id}/approve")
async def approve_candidate(candidate_id: str, user=Depends(require_admin)):
    """Approve a candidate — creates a pending gallery update."""
    try:
        result = IdentityReviewService.approve_candidate(
            candidate_id=candidate_id,
            actor_id=user["id"],
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Approve failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve candidate")


@router.post("/candidates/{candidate_id}/reject")
async def reject_candidate(candidate_id: str, user=Depends(require_admin)):
    """Reject a candidate."""
    try:
        result = IdentityReviewService.reject_candidate(
            candidate_id=candidate_id,
            actor_id=user["id"],
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/candidates/{candidate_id}/skip")
async def skip_candidate(candidate_id: str, user=Depends(require_admin)):
    """Skip a candidate for later review."""
    try:
        result = IdentityReviewService.skip_candidate(
            candidate_id=candidate_id,
            actor_id=user["id"],
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Clusters ─────────────────────────────────────────────────────

@router.get("/clusters")
async def list_clusters(
    status: Optional[str] = Query("active", description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user=Depends(require_admin),
):
    """List unknown clusters."""
    db = SessionLocal()
    try:
        query = db.query(UnknownCluster)
        if status:
            query = query.filter(UnknownCluster.status == status)

        total = query.count()
        clusters = (
            query
            .order_by(UnknownCluster.last_seen.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = [
            {
                "id": c.id,
                "representative_snapshot": c.representative_snapshot,
                "member_count": c.member_count,
                "first_seen": c.first_seen.isoformat() if c.first_seen else None,
                "last_seen": c.last_seen.isoformat() if c.last_seen else None,
                "status": c.status,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in clusters
        ]

        return {"items": items, "total": total}
    finally:
        db.close()


@router.get("/clusters/{cluster_id}")
async def get_cluster_detail(cluster_id: str, user=Depends(require_admin)):
    """Get cluster detail with all member images."""
    db = SessionLocal()
    try:
        cluster = db.query(UnknownCluster).filter(UnknownCluster.id == cluster_id).first()
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        from app.services.clustering_service import ClusteringService
        members = ClusteringService.get_cluster_members(cluster_id)

        return {
            "id": cluster.id,
            "representative_snapshot": cluster.representative_snapshot,
            "member_count": cluster.member_count,
            "first_seen": cluster.first_seen.isoformat() if cluster.first_seen else None,
            "last_seen": cluster.last_seen.isoformat() if cluster.last_seen else None,
            "status": cluster.status,
            "members": members,
        }
    finally:
        db.close()


@router.post("/clusters/{cluster_id}/ignore")
async def ignore_cluster(
    cluster_id: str,
    body: IgnoreClusterRequest,
    user=Depends(require_admin),
):
    """Ignore a cluster forever — won't generate candidates."""
    db = SessionLocal()
    try:
        cluster = db.query(UnknownCluster).filter(UnknownCluster.id == cluster_id).first()
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        cluster.status = "ignored"

        # Cancel any pending candidates for this cluster
        db.query(IdentityReviewCandidate).filter(
            IdentityReviewCandidate.cluster_id == cluster_id,
            IdentityReviewCandidate.status == "pending",
        ).update({"status": "skipped"})

        # Log activity
        from uuid import uuid4
        activity = IdentityReviewActivity(
            id=str(uuid4()),
            cluster_id=cluster_id,
            action="ignored_forever",
            actor=user["id"],
            details={"reason": body.reason},
        )
        db.add(activity)
        db.commit()

        return {"status": "ignored", "cluster_id": cluster_id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/clusters/{cluster_id}/create-person")
async def create_person_from_cluster(
    cluster_id: str,
    body: CreatePersonFromClusterRequest,
    user=Depends(require_admin),
):
    """Create a new person from a cluster's representative encoding."""
    db = SessionLocal()
    try:
        cluster = db.query(UnknownCluster).filter(UnknownCluster.id == cluster_id).first()
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        # Create the person via the existing service
        from app.services.person_service import PersonService
        from app.vision.recognizer import deserialize_encoding

        person = PersonService.create_person({
            "name": body.name,
            "role": body.role,
            "department": body.department,
        })

        # Add the cluster centroid as the first encoding
        if cluster.embedding_centroid:
            encoding = deserialize_encoding(cluster.embedding_centroid)
            if encoding is not None:
                PersonService.add_face_encoding(
                    person_id=person["id"],
                    encoding=encoding,
                    source_image=cluster.representative_snapshot,
                    quality=0.7,
                )

        # Mark cluster as merged
        cluster.status = "merged"

        # Mark cluster faces as enrolled
        db.query(UnknownFace).filter(
            UnknownFace.cluster_id == cluster_id,
        ).update({"status": "enrolled"})

        # Log activity
        from uuid import uuid4
        activity = IdentityReviewActivity(
            id=str(uuid4()),
            cluster_id=cluster_id,
            action="created_person",
            actor=user["id"],
            details={"person_id": person["id"], "person_name": body.name},
        )
        db.add(activity)
        db.commit()

        return {
            "status": "created",
            "person_id": person["id"],
            "person_name": body.name,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ── Gallery Updates ──────────────────────────────────────────────

@router.get("/gallery-updates")
async def list_gallery_updates(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user=Depends(require_admin),
):
    """List gallery updates."""
    db = SessionLocal()
    try:
        query = db.query(GalleryUpdate)
        if status:
            query = query.filter(GalleryUpdate.status == status)

        total = query.count()
        updates = (
            query
            .order_by(GalleryUpdate.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = []
        for u in updates:
            person = db.query(Person).filter(Person.id == u.person_id).first()
            items.append({
                "id": u.id,
                "person_id": u.person_id,
                "person_name": person.name if person else "Unknown",
                "snapshot_url": u.snapshot_url,
                "quality": u.quality,
                "action": u.action,
                "status": u.status,
                "applied_at": u.applied_at.isoformat() if u.applied_at else None,
                "created_at": u.created_at.isoformat() if u.created_at else None,
            })

        return {"items": items, "total": total}
    finally:
        db.close()


@router.post("/gallery-updates/{update_id}/apply")
async def apply_gallery_update(update_id: str, user=Depends(require_admin)):
    """Apply a pending gallery update (append encoding to person)."""
    try:
        result = IdentityReviewService.apply_gallery_update(update_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Apply gallery update failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to apply gallery update")


@router.post("/gallery-updates/{update_id}/rollback")
async def rollback_gallery_update(update_id: str, user=Depends(require_admin)):
    """Rollback an applied gallery update."""
    try:
        result = IdentityReviewService.rollback_gallery_update(update_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Activity ─────────────────────────────────────────────────────

@router.get("/activity")
async def list_activity(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(require_admin),
):
    """Get activity timeline."""
    db = SessionLocal()
    try:
        total = db.query(IdentityReviewActivity).count()
        activities = (
            db.query(IdentityReviewActivity)
            .order_by(IdentityReviewActivity.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = [
            {
                "id": a.id,
                "candidate_id": a.candidate_id,
                "cluster_id": a.cluster_id,
                "action": a.action,
                "actor": a.actor,
                "details": a.details,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in activities
        ]

        return {"items": items, "total": total}
    finally:
        db.close()


# ── PersonProfile (Gallery Coach) ────────────────────────────────

@router.get("/person-profile/{person_id}")
async def get_person_profile(person_id: str, user=Depends(require_admin)):
    """
    Get the full PersonProfile for a registered person.

    Returns gallery health, coverage analysis, encoding breakdown,
    and actionable recommendations (Gallery Coach view).
    """
    try:
        from app.services.gallery_health import compute_person_profile

        profile = compute_person_profile(person_id)

        if profile.total_encodings == 0 and profile.person_name == "Unknown":
            raise HTTPException(status_code=404, detail="Person not found")

        return {
            "person_id": profile.person_id,
            "person_name": profile.person_name,
            "total_encodings": profile.total_encodings,
            "health_score": profile.health_score,
            "coverage_percentage": profile.coverage_percentage,
            "covered_buckets": profile.covered_buckets,
            "missing_buckets": profile.missing_buckets,
            "coverage_score": profile.coverage_score,
            "quality_score": profile.quality_score,
            "diversity_score": profile.diversity_score,
            "recognition_score": profile.recognition_score,
            "redundancy_penalty": profile.redundancy_penalty,
            "improvement_potential": profile.improvement_potential,
            "top_recommendation": profile.top_recommendation,
            "encoding_health": [
                {
                    "id": eh.encoding_id,
                    "pose_bucket": eh.pose_bucket,
                    "quality": eh.quality,
                    "usage_count": eh.usage_count,
                    "success_count": eh.success_count,
                    "effectiveness": eh.effectiveness,
                    "health": eh.health,
                    "source_image": eh.source_image,
                    "last_matched": eh.last_matched,
                }
                for eh in profile.encoding_health
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PersonProfile failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute person profile")
