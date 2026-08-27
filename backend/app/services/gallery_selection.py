"""
Gallery Selection — decides WHICH cluster members reach a person's gallery.

This module owns no gating rules of its own. It sequences the gates that
already exist, in the order the Gallery Optimizer needs them:

    cluster member
      → reviewer selection      (outliers excluded by default)
      → metadata gate           (real pose/quality required — never fabricated)
      → capture quality gate    (GALLERY_OPT_MIN_* thresholds)
      → capacity gate           (MAX_ENCODINGS_PER_PERSON)
      → diversity gate          (EmbeddingSuggestionService.evaluate_suggestion:
                                 duplicate → pose coverage → quality →
                                 sharpness → appearance)
      → gallery

The point of this file is what it replaced. Approval used to write
`cluster.embedding_centroid` — ONE averaged vector — to the gallery. Averaging
an extreme-left crop, a low-light crop and a distant crop produces a vector
that represents none of them, which is the opposite of the goal: the gallery
should gain the real-world variation the cluster actually contains.

Preview and apply run the SAME code path. Preview flushes candidate encodings
into an open transaction, reads the resulting coverage, then rolls back. So
the "62% → 78%" the reviewer sees is not a model of what approval would do,
it is the result of doing it and undoing it.
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.db.models import (
    FaceEncoding, GalleryUpdate, IdentityReviewActivity,
    IdentityReviewCandidate, Person, UnknownCluster, UnknownFace,
)
from app.services.clustering_service import centroid_distances, outlier_flags
from app.services.embedding_suggestions import EmbeddingSuggestionService
from app.services.gallery_health import compute_person_profile
from app.vision.recognizer import deserialize_encoding, serialize_encoding

logger = logging.getLogger(__name__)


# Rarity ordering for processing members. Rarer poses are evaluated first so
# that when the per-person cap bites, the encodings that survive are the ones
# covering angles the gallery lacks — not another frontal shot.
_POSE_PROCESSING_RANK: Dict[str, int] = {
    "extreme_left": 0,
    "extreme_right": 0,
    "left": 1,
    "right": 1,
    "frontal": 2,
}


# Member outcomes. Everything except "accepted" explains why an image the
# human approved as this person still did not enter the gallery.
STATUS_ACCEPTED = "accepted"
STATUS_OUTLIER = "outlier_excluded"
STATUS_DESELECTED = "deselected_by_reviewer"
STATUS_NO_EMBEDDING = "no_embedding"
STATUS_NO_METADATA = "no_pose_metadata"
STATUS_LOW_QUALITY = "low_quality"
STATUS_REDUNDANT = "redundant"
STATUS_GALLERY_FULL = "gallery_full"


@dataclass
class MemberDecision:
    """What happened to one unknown face in the cluster, and why."""
    face_id: str
    snapshot_url: Optional[str] = None
    camera_id: Optional[str] = None
    occurrence: int = 1
    pose_bucket: Optional[str] = None
    yaw: Optional[float] = None
    quality: Optional[float] = None
    blur_score: Optional[float] = None
    lighting: Optional[str] = None
    distance_from_centroid: float = 0.0
    is_outlier: bool = False
    selected: bool = False
    status: str = STATUS_DESELECTED
    reason: str = ""
    encoding_id: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED


@dataclass
class SelectionReport:
    """Full account of an approval — preview or applied."""
    candidate_id: str
    cluster_id: str
    person_id: str
    person_name: str
    applied: bool = False

    total_members: int = 0
    outlier_count: int = 0
    outlier_threshold: float = 0.0

    encodings_before: int = 0
    encodings_after: int = 0
    capacity: int = 0

    coverage_before: float = 0.0
    coverage_after: float = 0.0
    coverage_delta: float = 0.0
    health_before: float = 0.0
    health_after: float = 0.0
    buckets_gained: List[str] = field(default_factory=list)

    gallery_version: Optional[int] = None
    members: List[MemberDecision] = field(default_factory=list)

    @property
    def accepted(self) -> List[MemberDecision]:
        return [m for m in self.members if m.accepted]

    @property
    def rejected(self) -> List[MemberDecision]:
        return [m for m in self.members if not m.accepted]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "cluster_id": self.cluster_id,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "applied": self.applied,
            "total_members": self.total_members,
            "outlier_count": self.outlier_count,
            "outlier_threshold": round(self.outlier_threshold, 4),
            "encodings_before": self.encodings_before,
            "encodings_after": self.encodings_after,
            "capacity": self.capacity,
            "coverage_before": self.coverage_before,
            "coverage_after": self.coverage_after,
            "coverage_delta": self.coverage_delta,
            "health_before": self.health_before,
            "health_after": self.health_after,
            "buckets_gained": self.buckets_gained,
            "gallery_version": self.gallery_version,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "members": [asdict(m) for m in self.members],
        }


# ── Member loading + outlier analysis ────────────────────────────

def _load_members(
    db: Session,
    cluster: UnknownCluster,
) -> Tuple[List[Tuple[UnknownFace, Optional[np.ndarray]]], List[float], List[bool], float]:
    """
    Load a cluster's member faces with their embeddings, centroid distances,
    and outlier flags.

    The centroid is recomputed from the members present rather than trusting
    the stored one, so a cluster that has changed since the worker last wrote
    it is still measured against itself.
    """
    faces = (
        db.query(UnknownFace)
        .filter(UnknownFace.cluster_id == cluster.id)
        .order_by(UnknownFace.last_seen.desc())
        .all()
    )

    members: List[Tuple[UnknownFace, Optional[np.ndarray]]] = []
    for f in faces:
        enc = deserialize_encoding(f.face_encoding) if f.face_encoding else None
        if enc is not None:
            norm = np.linalg.norm(enc)
            enc = (enc / norm) if norm > 0 else None
        members.append((f, enc))

    usable = [enc for _, enc in members if enc is not None]
    if usable:
        centroid = np.mean(usable, axis=0)
    else:
        centroid = deserialize_encoding(cluster.embedding_centroid) if cluster.embedding_centroid else None

    if centroid is None:
        return members, [1.0] * len(members), [False] * len(members), 0.0

    distances = centroid_distances(
        [enc if enc is not None else np.zeros_like(centroid) for _, enc in members],
        centroid,
    )
    flags, threshold = outlier_flags(distances, settings.GALLERY_OPT_OUTLIER_FACTOR)
    return members, distances, flags, threshold


def _passes_capture_quality(face: UnknownFace) -> Tuple[bool, str]:
    """
    Capture-quality gate using the GALLERY_OPT_* thresholds.

    Checks are skipped for metrics the row does not carry — a missing metric
    is not evidence of a bad capture, and the metadata gate has already
    rejected rows with no pose/quality at all.
    """
    if face.blur_score is not None and face.blur_score < settings.GALLERY_OPT_MIN_BLUR_SCORE:
        return False, (
            f"Too blurry (blur={face.blur_score:.0f}, "
            f"min={settings.GALLERY_OPT_MIN_BLUR_SCORE:.0f})"
        )
    if face.det_score is not None and face.det_score < settings.GALLERY_OPT_MIN_DET_SCORE:
        return False, (
            f"Low detection confidence ({face.det_score:.2f}, "
            f"min={settings.GALLERY_OPT_MIN_DET_SCORE:.2f})"
        )
    return True, ""


# ── The selection run ────────────────────────────────────────────

def evaluate_candidate_members(
    candidate_id: str,
    selected_face_ids: Optional[List[str]] = None,
    apply: bool = False,
    actor_id: Optional[str] = None,
) -> SelectionReport:
    """
    Run every cluster member through the gate ladder for one review candidate.

    Args:
        candidate_id: the IdentityReviewCandidate under review.
        selected_face_ids: which members the reviewer wants applied. None means
            "the default selection" — every member except detected outliers.
            An empty list means the reviewer deselected everything.
        apply: False runs a dry run and rolls back (for the preview panel).
            True commits: encodings are added, gallery_updates and an audit
            entry are written, and the candidate is marked approved.
        actor_id: admin id recorded on the audit trail. Required when apply.

    Raises ValueError if the candidate is missing, or if it has already been
    reviewed and this is an apply run.
    """
    if apply and not actor_id:
        raise ValueError("actor_id is required to apply a gallery update")

    db: Session = SessionLocal()
    try:
        candidate = (
            db.query(IdentityReviewCandidate)
            .filter(IdentityReviewCandidate.id == candidate_id)
            .first()
        )
        if not candidate:
            raise ValueError(f"Candidate {candidate_id} not found")
        if apply and candidate.status != "pending":
            raise ValueError(f"Candidate already {candidate.status}")

        cluster = (
            db.query(UnknownCluster)
            .filter(UnknownCluster.id == candidate.cluster_id)
            .first()
        )
        if not cluster:
            raise ValueError(f"Cluster {candidate.cluster_id} not found")

        person = db.query(Person).filter(Person.id == candidate.person_id).first()
        if not person:
            raise ValueError(f"Person {candidate.person_id} not found")

        report = SelectionReport(
            candidate_id=candidate.id,
            cluster_id=cluster.id,
            person_id=person.id,
            person_name=person.name,
        )

        # ── Baseline ────────────────────────────────────────────
        profile_before = compute_person_profile(person.id, db=db)
        report.coverage_before = profile_before.coverage_percentage
        report.health_before = profile_before.health_score
        buckets_before = set(profile_before.covered_buckets)

        report.encodings_before = (
            db.query(FaceEncoding)
            .filter(FaceEncoding.person_id == person.id)
            .count()
        )
        report.capacity = settings.MAX_ENCODINGS_PER_PERSON

        # ── Members + outliers ──────────────────────────────────
        members, distances, flags, threshold = _load_members(db, cluster)
        report.total_members = len(members)
        report.outlier_threshold = threshold
        report.outlier_count = sum(1 for f in flags if f)

        # Default selection: everything the outlier check did not flag.
        # A face that no longer resembles its own cluster is the single most
        # likely way a wrong identity enters a gallery, so it is opt-in.
        if selected_face_ids is None:
            selection = {
                face.id for (face, _), is_out in zip(members, flags) if not is_out
            }
        else:
            selection = set(selected_face_ids)

        decisions: Dict[str, MemberDecision] = {}
        for (face, enc), dist, is_out in zip(members, distances, flags):
            decisions[face.id] = MemberDecision(
                face_id=face.id,
                snapshot_url=face.snapshot_url,
                camera_id=face.camera_id,
                occurrence=face.occurrence or 1,
                pose_bucket=face.pose_bucket,
                yaw=face.yaw,
                quality=face.quality,
                blur_score=face.blur_score,
                lighting=face.lighting,
                distance_from_centroid=round(dist, 4),
                is_outlier=is_out,
                selected=face.id in selection,
            )

        # Rarest poses first so the cap protects the most valuable additions.
        ordered = sorted(
            members,
            key=lambda m: (
                _POSE_PROCESSING_RANK.get(m[0].pose_bucket or "", 3),
                -(m[0].quality or 0.0),
            ),
        )

        # ── Gate ladder ─────────────────────────────────────────
        encoding_count = report.encodings_before
        accepted_faces: List[Tuple[UnknownFace, FaceEncoding, str]] = []

        for face, enc in ordered:
            d = decisions[face.id]

            if not d.selected:
                d.status = STATUS_OUTLIER if d.is_outlier else STATUS_DESELECTED
                d.reason = (
                    f"Inconsistent with the rest of the cluster "
                    f"(distance={d.distance_from_centroid:.3f} > {threshold:.3f})"
                    if d.is_outlier else "Deselected by reviewer"
                )
                continue

            if enc is None:
                d.status = STATUS_NO_EMBEDDING
                d.reason = "No usable face embedding stored"
                continue

            # Metadata gate. A row written before migration 010, or one whose
            # detection had no SCRFD landmarks, has no real pose. Assuming
            # "frontal" would corrupt every coverage number downstream, so
            # such members are reported and skipped instead.
            if not face.pose_bucket or face.quality is None:
                d.status = STATUS_NO_METADATA
                d.reason = (
                    "No pose/quality metadata — cannot judge diversity without "
                    "fabricating a pose. Run backfill_unknown_pose.py."
                )
                continue

            ok, why = _passes_capture_quality(face)
            if not ok:
                d.status = STATUS_LOW_QUALITY
                d.reason = why
                continue

            if encoding_count >= report.capacity:
                d.status = STATUS_GALLERY_FULL
                d.reason = (
                    f"Gallery at capacity ({encoding_count}/{report.capacity}) — "
                    f"remove a weak encoding to make room"
                )
                continue

            # Diversity gate — the existing per-frame suggestion ladder.
            # Sharing `db` is what makes this sequential: each accepted
            # embedding is flushed below, so the next member is compared
            # against a gallery that already contains it.
            result = EmbeddingSuggestionService.evaluate_suggestion(
                person_id=person.id,
                new_encoding=enc,
                quality=face.quality,
                yaw=face.yaw or 0.0,
                pitch=face.pitch or 0.0,
                blur_score=face.blur_score or 0.0,
                lighting=face.lighting,
                db=db,
            )

            if not result.should_suggest:
                d.status = STATUS_REDUNDANT
                d.reason = result.reason
                continue

            new_enc = FaceEncoding(
                id=str(uuid4()),
                person_id=person.id,
                encoding_data=serialize_encoding(enc),
                source_image=face.snapshot_url,
                quality=face.quality,
                yaw=face.yaw,
                pitch=face.pitch,
                roll=face.roll,
                det_score=face.det_score,
                blur_score=face.blur_score,
                camera_id=face.camera_id,
                lighting=face.lighting,
                pose_bucket=face.pose_bucket,
                model_version="arcface_r50",
                created_at=datetime.now(timezone.utc),
            )
            db.add(new_enc)
            db.flush()

            d.status = STATUS_ACCEPTED
            d.reason = result.reason
            d.encoding_id = new_enc.id
            encoding_count += 1
            accepted_faces.append((face, new_enc, result.reason))

        report.members = [decisions[face.id] for face, _ in members]
        report.encodings_after = encoding_count

        # ── Resulting coverage (measured, not modelled) ─────────
        profile_after = compute_person_profile(person.id, db=db)
        report.coverage_after = profile_after.coverage_percentage
        report.coverage_delta = round(
            report.coverage_after - report.coverage_before, 1
        )
        report.health_after = profile_after.health_score
        report.buckets_gained = sorted(
            set(profile_after.covered_buckets) - buckets_before
        )

        if not apply:
            db.rollback()
            return report

        _commit_approval(db, candidate, cluster, person, report,
                         accepted_faces, selection, actor_id)
        report.applied = True
        return report

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _commit_approval(
    db: Session,
    candidate: IdentityReviewCandidate,
    cluster: UnknownCluster,
    person: Person,
    report: SelectionReport,
    accepted_faces: List[Tuple[UnknownFace, FaceEncoding, str]],
    selection: set,
    actor_id: str,
):
    """
    Persist an approval: gallery_updates, audit entry, candidate + cluster
    status, and the encoding-version bump that tells the pipeline to reload.

    One approval is one per-person gallery version, so rollback works at the
    granularity a human actually decided at.
    """
    from app.services.person_service import PersonService

    prev_version = (
        db.query(GalleryUpdate.gallery_version)
        .filter(
            GalleryUpdate.person_id == person.id,
            GalleryUpdate.gallery_version.isnot(None),
        )
        .order_by(GalleryUpdate.gallery_version.desc())
        .limit(1)
        .scalar()
    )
    new_version = (prev_version or 0) + 1
    report.gallery_version = new_version

    now = datetime.now(timezone.utc)

    # Global encoding version — what the pipeline's incremental cache syncs on.
    # One approval is one bump, applied to every encoding it created.
    enc_version = PersonService._get_or_create_version(db) + 1
    for _, enc, _ in accepted_faces:
        enc.version = enc_version

    for face, enc, reason in accepted_faces:
        db.add(GalleryUpdate(
            id=str(uuid4()),
            person_id=person.id,
            candidate_id=candidate.id,
            cluster_id=cluster.id,
            # The embedding of THIS face, not the cluster average.
            encoding_data=enc.encoding_data,
            snapshot_url=face.snapshot_url,
            pose_bucket=face.pose_bucket,
            quality=face.quality,
            action="append",
            gallery_version=new_version,
            source_face_id=face.id,
            selection_reason=reason,
            encoding_id=enc.id,
            coverage_before=report.coverage_before,
            coverage_after=report.coverage_after,
            status="applied",
            applied_at=now,
            created_at=now,
        ))

    # Every member the human affirmed as this person is resolved, including
    # the ones whose embedding was redundant. The human approved an IDENTITY;
    # the gates only decided which images the gallery needed. Leaving the
    # redundant ones "pending" would recluster and re-ask the same question.
    # Deselected and outlier members stay pending so they can cluster on their
    # own — approving Rahul must not silently resolve someone else's face.
    if selection:
        (
            db.query(UnknownFace)
            .filter(
                UnknownFace.cluster_id == cluster.id,
                UnknownFace.id.in_(selection),
            )
            .update({"status": "enrolled"}, synchronize_session=False)
        )

    cluster.status = "reviewed"
    cluster.updated_at = now

    candidate.status = "approved"
    candidate.reviewed_by = actor_id
    candidate.reviewed_at = now
    # Store what actually happened, replacing the worker's pre-review estimate.
    candidate.coverage_before = report.coverage_before
    candidate.coverage_after = report.coverage_after

    db.add(IdentityReviewActivity(
        id=str(uuid4()),
        candidate_id=candidate.id,
        cluster_id=cluster.id,
        action="approved",
        actor=actor_id,
        details={
            "person_id": person.id,
            "person_name": person.name,
            "overall_score": candidate.overall_score,
            "gallery_version": new_version,
            "coverage_before": report.coverage_before,
            "coverage_after": report.coverage_after,
            "buckets_gained": report.buckets_gained,
            "encodings_before": report.encodings_before,
            "encodings_after": report.encodings_after,
            "outlier_count": report.outlier_count,
            "accepted": [
                {
                    "face_id": m.face_id,
                    "encoding_id": m.encoding_id,
                    "pose_bucket": m.pose_bucket,
                    "quality": m.quality,
                    "snapshot_url": m.snapshot_url,
                    "reason": m.reason,
                }
                for m in report.accepted
            ],
            "rejected": [
                {
                    "face_id": m.face_id,
                    "pose_bucket": m.pose_bucket,
                    "status": m.status,
                    "reason": m.reason,
                }
                for m in report.rejected
            ],
        },
    ))

    if accepted_faces:
        PersonService._set_version(db, enc_version)

    db.commit()

    logger.info(
        f"Gallery update applied for {person.name} "
        f"(v{new_version}): +{len(accepted_faces)} encodings, "
        f"coverage {report.coverage_before:.1f}% → {report.coverage_after:.1f}%, "
        f"{len(report.rejected)} member(s) not added"
    )

    if accepted_faces:
        _refresh_pipeline_encodings()


def _refresh_pipeline_encodings():
    """Push the enlarged gallery to the live recognisers. Never fatal."""
    try:
        from app.services.person_service import PersonService
        from app.vision.camera_worker import camera_manager
        camera_manager.load_encodings_all(PersonService.get_all_encodings())
        logger.info("Pipeline encoding cache refreshed after gallery update")
    except Exception as e:
        logger.warning(f"Pipeline refresh after gallery update failed: {e}")
