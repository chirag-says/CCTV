"""
Gallery Health Service — PersonProfile, Coverage Analysis, and Gallery Health.

The core abstraction of the Gallery Optimizer. Every registered person has a
PersonProfile that tracks:
  - Gallery coverage (which pose/lighting/appearance buckets are filled)
  - Gallery health (a single 0–1 score combining coverage, quality, diversity,
    recognition success, and redundancy penalty)
  - Actionable recommendations (what's missing, what to approve next)
  - Coverage delta projections (what happens if a candidate is approved)

The VLM is advisory. ArcFace is the biometric engine. Gallery Health is the
metric that drives continuous improvement.

Never called from the realtime pipeline. Only used by the background
Gallery Optimizer worker and the review API.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.db.session import SessionLocal
from app.db.models import FaceEncoding, Person
from app.services.embedding_suggestions import classify_pose_bucket
from app.vision.recognizer import deserialize_encoding

logger = logging.getLogger(__name__)


# ── Coverage Buckets ─────────────────────────────────────────────
# What a complete gallery should contain, with importance weights.

COVERAGE_BUCKETS: Dict[str, float] = {
    # Pose (5 buckets, 60% total weight)
    "pose_frontal":       0.20,
    "pose_left":          0.12,
    "pose_right":         0.12,
    "pose_extreme_left":  0.08,
    "pose_extreme_right": 0.08,
    # Lighting (2 buckets, 15% total weight)
    "lighting_bright":    0.08,
    "lighting_dim":       0.07,
    # Appearance (3 buckets, 15% total weight)
    "appearance_plain":   0.05,
    "appearance_glasses":  0.05,
    "appearance_mask":    0.05,
    # Quality tiers (2 buckets, 10% total weight)
    "quality_high":       0.05,
    "quality_medium":     0.05,
}

# Mapping from FaceEncoding fields to coverage bucket names
_POSE_TO_BUCKET = {
    "frontal":       "pose_frontal",
    "left":          "pose_left",
    "right":         "pose_right",
    "extreme_left":  "pose_extreme_left",
    "extreme_right": "pose_extreme_right",
}

_LIGHTING_TO_BUCKET = {
    "bright":  "lighting_bright",
    "dim":     "lighting_dim",
    "backlit": "lighting_dim",  # Treat backlit as dim coverage
}


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class EncodingHealth:
    """Health assessment for a single encoding."""
    encoding_id: str
    pose_bucket: str
    quality: float
    usage_count: int
    success_count: int
    effectiveness: float    # success/usage ratio
    health: float           # Individual encoding health score
    source_image: Optional[str] = None
    last_matched: Optional[str] = None


@dataclass
class PersonProfile:
    """
    Complete gallery profile for a registered person.

    This is the core abstraction — a person is not just "name + embeddings"
    but a profile with coverage, health, and actionable recommendations.
    """
    person_id: str
    person_name: str
    total_encodings: int

    # Coverage
    coverage_percentage: float          # e.g. 63.0
    covered_buckets: List[str] = field(default_factory=list)
    missing_buckets: List[str] = field(default_factory=list)

    # Gallery Health (the headline number)
    health_score: float = 0.0           # 0.0–1.0

    # Health components
    coverage_score: float = 0.0         # Weighted bucket coverage
    quality_score: float = 0.0          # Average encoding quality
    diversity_score: float = 0.0        # Embedding spread
    recognition_score: float = 0.0      # Real-world matching effectiveness
    redundancy_penalty: float = 0.0     # Penalty for duplicate poses

    # Actionable
    improvement_potential: float = 0.0  # How much health could improve
    top_recommendation: Optional[str] = None  # "Needs extreme left profile"

    # Per-encoding breakdown
    encoding_health: List[EncodingHealth] = field(default_factory=list)


@dataclass
class CoverageDelta:
    """Projection of what happens if a candidate is approved."""
    current_coverage: float         # e.g. 63.0
    projected_coverage: float       # e.g. 74.0
    delta: float                    # e.g. +11.0
    current_health: float           # e.g. 0.82
    projected_health: float         # e.g. 0.91
    fills_bucket: Optional[str]     # "pose_extreme_left" or None
    improvement_value: str          # "high" / "medium" / "low" / "none"


# ── Health Computation ───────────────────────────────────────────

def _classify_encoding_buckets(enc: FaceEncoding) -> List[str]:
    """Determine which coverage buckets an encoding fills."""
    buckets = []

    # Pose bucket
    pose = enc.pose_bucket or classify_pose_bucket(enc.yaw or 0.0)
    bucket_name = _POSE_TO_BUCKET.get(pose)
    if bucket_name:
        buckets.append(bucket_name)

    # Lighting bucket
    if enc.lighting:
        lighting_bucket = _LIGHTING_TO_BUCKET.get(enc.lighting)
        if lighting_bucket:
            buckets.append(lighting_bucket)

    # Appearance buckets
    tags = set()
    if enc.appearance_tags:
        tags = set(enc.appearance_tags.split(","))
    if "glasses" in tags:
        buckets.append("appearance_glasses")
    if "mask" in tags:
        buckets.append("appearance_mask")
    if not tags.intersection({"glasses", "mask"}):
        buckets.append("appearance_plain")

    # Quality tier
    quality = enc.quality or 0.5
    if quality >= 0.8:
        buckets.append("quality_high")
    elif quality >= 0.5:
        buckets.append("quality_medium")

    return buckets


def _compute_coverage_score(covered: set) -> float:
    """Weighted coverage score based on which buckets are filled."""
    score = 0.0
    for bucket, weight in COVERAGE_BUCKETS.items():
        if bucket in covered:
            score += weight
    return score


def _compute_coverage_percentage(covered: set) -> float:
    """Coverage as a simple percentage of filled buckets."""
    if not COVERAGE_BUCKETS:
        return 0.0
    return round(100.0 * len(covered) / len(COVERAGE_BUCKETS), 1)


def _compute_quality_score(encodings: List[FaceEncoding]) -> float:
    """Average quality across all encodings."""
    if not encodings:
        return 0.0
    qualities = [(enc.quality or 0.5) for enc in encodings]
    return sum(qualities) / len(qualities)


def _compute_diversity_score(encodings: List[FaceEncoding]) -> float:
    """
    Embedding diversity — how spread the encodings are in embedding space.

    Uses average pairwise cosine distance. More spread = higher diversity.
    Capped at 1.0.
    """
    if len(encodings) < 2:
        return 0.0

    vectors = []
    for enc in encodings:
        arr = deserialize_encoding(enc.encoding_data)
        if arr is not None:
            norm = np.linalg.norm(arr)
            if norm > 0:
                vectors.append(arr / norm)

    if len(vectors) < 2:
        return 0.0

    # Average pairwise cosine distance
    total_dist = 0.0
    count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = 1.0 - float(np.dot(vectors[i], vectors[j]))
            total_dist += dist
            count += 1

    avg_dist = total_dist / count if count > 0 else 0.0
    # Normalize: typical cosine distances for same person range 0.1–0.5
    # Scale so 0.3 avg distance → diversity = 1.0
    return min(1.0, avg_dist / 0.3)


def _compute_recognition_score(encodings: List[FaceEncoding]) -> float:
    """
    Score based on real-world matching effectiveness.

    Encodings that have been used successfully in recognition are more
    valuable than encodings that have never matched.

    Uses existing FaceEncoding.usage_count and .success_count fields.
    """
    if not encodings:
        return 0.0

    scores = []
    for enc in encodings:
        usage = enc.usage_count or 0
        success = enc.success_count or 0
        if usage > 0:
            scores.append(success / usage)  # Effectiveness ratio
        elif success > 0:
            scores.append(0.8)              # Has matches but no usage tracking
        else:
            scores.append(0.1)              # Never used — low value
    return sum(scores) / len(scores)


def _compute_redundancy_penalty(encodings: List[FaceEncoding]) -> float:
    """
    Penalty for having too many encodings in the same pose bucket.

    3+ encodings in the same bucket = wasted gallery space.
    Returns 0.0–1.0 (higher = more redundancy).
    """
    if not encodings:
        return 0.0

    bucket_counts: Dict[str, int] = {}
    for enc in encodings:
        pose = enc.pose_bucket or classify_pose_bucket(enc.yaw or 0.0)
        bucket_counts[pose] = bucket_counts.get(pose, 0) + 1

    # Penalty = fraction of encodings that are redundant (3rd+ in same bucket)
    total_redundant = 0
    for count in bucket_counts.values():
        if count > 2:
            total_redundant += count - 2

    if len(encodings) == 0:
        return 0.0
    return min(1.0, total_redundant / len(encodings))


def _compute_encoding_health(enc: FaceEncoding) -> EncodingHealth:
    """Compute health for a single encoding."""
    usage = enc.usage_count or 0
    success = enc.success_count or 0
    effectiveness = (success / usage) if usage > 0 else 0.0

    quality = enc.quality or 0.5
    # Health = quality × 0.4 + effectiveness × 0.4 + recency × 0.2
    if enc.last_matched:
        days_since = (datetime.now(timezone.utc) - enc.last_matched).total_seconds() / 86400
        recency = max(0.0, 1.0 - days_since / 90.0)
    else:
        recency = 0.1

    health = quality * 0.4 + effectiveness * 0.4 + recency * 0.2

    return EncodingHealth(
        encoding_id=enc.id,
        pose_bucket=enc.pose_bucket or classify_pose_bucket(enc.yaw or 0.0),
        quality=quality,
        usage_count=usage,
        success_count=success,
        effectiveness=round(effectiveness, 3),
        health=round(health, 3),
        source_image=enc.source_image,
        last_matched=enc.last_matched.isoformat() if enc.last_matched else None,
    )


def _generate_recommendation(missing_buckets: List[str]) -> Optional[str]:
    """Generate the top recommendation based on missing coverage."""
    # Priority order: pose > lighting > appearance > quality
    priority = [
        ("pose_frontal", "Needs frontal face image"),
        ("pose_left", "Needs left profile image"),
        ("pose_right", "Needs right profile image"),
        ("pose_extreme_left", "Needs extreme left profile"),
        ("pose_extreme_right", "Needs extreme right profile"),
        ("lighting_dim", "Needs low-light image"),
        ("lighting_bright", "Needs well-lit image"),
        ("appearance_glasses", "Needs image with glasses"),
        ("appearance_mask", "Needs image with mask"),
    ]
    for bucket, message in priority:
        if bucket in missing_buckets:
            return message
    return None


# ── Public API ───────────────────────────────────────────────────

def compute_person_profile(person_id: str, db=None) -> PersonProfile:
    """
    Compute the full PersonProfile for a registered person.

    This is the main entry point. Queries all encodings, computes coverage,
    health, and recommendations.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            return PersonProfile(
                person_id=person_id,
                person_name="Unknown",
                total_encodings=0,
                coverage_percentage=0.0,
            )

        encodings = (
            db.query(FaceEncoding)
            .filter(FaceEncoding.person_id == person_id)
            .all()
        )

        # Compute covered buckets
        covered = set()
        for enc in encodings:
            for bucket in _classify_encoding_buckets(enc):
                covered.add(bucket)

        all_buckets = set(COVERAGE_BUCKETS.keys())
        missing = sorted(all_buckets - covered)

        # Compute scores
        coverage_score = _compute_coverage_score(covered)
        quality_score = _compute_quality_score(encodings)
        diversity_score = _compute_diversity_score(encodings)
        recognition_score = _compute_recognition_score(encodings)
        redundancy_penalty = _compute_redundancy_penalty(encodings)

        # Gallery Health formula:
        # Health = coverage × 0.40 + quality × 0.20 + diversity × 0.15
        #        + recognition × 0.15 − redundancy × 0.10
        health = (
            coverage_score * 0.40
            + quality_score * 0.20
            + diversity_score * 0.15
            + recognition_score * 0.15
            - redundancy_penalty * 0.10
        )
        health = max(0.0, min(1.0, health))

        # Per-encoding breakdown
        enc_health = [_compute_encoding_health(enc) for enc in encodings]

        # Recommendation
        recommendation = _generate_recommendation(missing)

        # Improvement potential = max possible health - current
        improvement_potential = max(0.0, 1.0 - health)

        return PersonProfile(
            person_id=person_id,
            person_name=person.name,
            total_encodings=len(encodings),
            coverage_percentage=_compute_coverage_percentage(covered),
            covered_buckets=sorted(covered),
            missing_buckets=missing,
            health_score=round(health, 3),
            coverage_score=round(coverage_score, 3),
            quality_score=round(quality_score, 3),
            diversity_score=round(diversity_score, 3),
            recognition_score=round(recognition_score, 3),
            redundancy_penalty=round(redundancy_penalty, 3),
            improvement_potential=round(improvement_potential, 3),
            top_recommendation=recommendation,
            encoding_health=enc_health,
        )

    finally:
        if close_db:
            db.close()


def compute_coverage_delta(
    person_id: str,
    new_pose_bucket: Optional[str] = None,
    new_quality: float = 0.7,
    new_lighting: Optional[str] = None,
    new_appearance_tags: Optional[List[str]] = None,
    db=None,
    new_pose_buckets: Optional[List[str]] = None,
) -> CoverageDelta:
    """
    Project what happens to coverage/health if new embeddings are approved.

    Used to populate the "63% → 74%" display on review cards.

    `new_pose_buckets` projects several additions at once — an unknown cluster
    typically contributes more than one pose, and scoring only its first member
    understates the gain. `new_pose_bucket` (singular) is the single-embedding
    form and still works. A bucket of None (pose not known) contributes no pose
    coverage rather than being counted as frontal.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        # Current profile
        current = compute_person_profile(person_id, db=db)

        # Simulate adding the new encoding
        new_covered = set(current.covered_buckets)

        # Add pose bucket(s)
        buckets_in = list(new_pose_buckets) if new_pose_buckets else []
        if new_pose_bucket:
            buckets_in.append(new_pose_bucket)

        fills_bucket = None
        for bucket in buckets_in:
            pose_bucket_name = _POSE_TO_BUCKET.get(bucket) if bucket else None
            if pose_bucket_name and pose_bucket_name not in new_covered:
                new_covered.add(pose_bucket_name)
                if fills_bucket is None:
                    fills_bucket = pose_bucket_name

        # Add lighting bucket
        if new_lighting:
            lighting_bucket = _LIGHTING_TO_BUCKET.get(new_lighting)
            if lighting_bucket and lighting_bucket not in new_covered:
                new_covered.add(lighting_bucket)
                if fills_bucket is None:
                    fills_bucket = lighting_bucket

        # Add appearance buckets
        tags = set(new_appearance_tags or [])
        if "glasses" in tags and "appearance_glasses" not in new_covered:
            new_covered.add("appearance_glasses")
        if "mask" in tags and "appearance_mask" not in new_covered:
            new_covered.add("appearance_mask")
        if not tags.intersection({"glasses", "mask"}):
            if "appearance_plain" not in new_covered:
                new_covered.add("appearance_plain")

        # Quality tier
        if new_quality >= 0.8:
            new_covered.add("quality_high")
        elif new_quality >= 0.5:
            new_covered.add("quality_medium")

        # Projected scores
        projected_coverage = _compute_coverage_percentage(new_covered)
        projected_coverage_score = _compute_coverage_score(new_covered)

        # Estimate projected health (reuse current non-coverage components)
        projected_health = (
            projected_coverage_score * 0.40
            + current.quality_score * 0.20
            + current.diversity_score * 0.15
            + current.recognition_score * 0.15
            - current.redundancy_penalty * 0.10
        )
        projected_health = max(0.0, min(1.0, projected_health))

        # Determine improvement value
        delta = projected_coverage - current.coverage_percentage
        if fills_bucket and delta > 5.0:
            improvement_value = "high"
        elif delta > 2.0:
            improvement_value = "medium"
        elif delta > 0.0:
            improvement_value = "low"
        else:
            improvement_value = "none"

        return CoverageDelta(
            current_coverage=current.coverage_percentage,
            projected_coverage=projected_coverage,
            delta=round(delta, 1),
            current_health=current.health_score,
            projected_health=round(projected_health, 3),
            fills_bucket=fills_bucket,
            improvement_value=improvement_value,
        )

    finally:
        if close_db:
            db.close()
