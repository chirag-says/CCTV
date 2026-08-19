"""
Tests for Gallery Health, PersonProfile, and Coverage Delta.

All tests use mocks — no real DB calls.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from app.services.gallery_health import (
    compute_person_profile,
    compute_coverage_delta,
    PersonProfile,
    CoverageDelta,
    COVERAGE_BUCKETS,
    _compute_coverage_score,
    _compute_coverage_percentage,
    _compute_quality_score,
    _compute_diversity_score,
    _compute_recognition_score,
    _compute_redundancy_penalty,
    _classify_encoding_buckets,
    _generate_recommendation,
)


def _make_mock_encoding(
    id="enc-1",
    person_id="person-1",
    pose_bucket="frontal",
    yaw=0.0,
    quality=0.8,
    lighting="bright",
    appearance_tags=None,
    usage_count=10,
    success_count=8,
    last_matched=None,
    encoding_data=None,
    source_image="crop.jpg",
):
    """Create a mock FaceEncoding object."""
    import numpy as np

    enc = MagicMock()
    enc.id = id
    enc.person_id = person_id
    enc.pose_bucket = pose_bucket
    enc.yaw = yaw
    enc.quality = quality
    enc.lighting = lighting
    enc.appearance_tags = appearance_tags
    enc.usage_count = usage_count
    enc.success_count = success_count
    enc.last_matched = last_matched or datetime.now(timezone.utc)
    enc.source_image = source_image

    if encoding_data is None:
        # Generate a random 512-dim embedding
        rng = np.random.RandomState(hash(id) % 2**31)
        vec = rng.randn(512).astype(np.float32)
        enc.encoding_data = vec.tobytes()
    else:
        enc.encoding_data = encoding_data

    return enc


def _make_mock_person(id="person-1", name="Chirag"):
    person = MagicMock()
    person.id = id
    person.name = name
    return person


# ── Coverage Bucket Classification ───────────────────────────────

class TestCoverageClassification:
    """Test coverage bucket assignment from encoding metadata."""

    def test_frontal_pose(self):
        enc = _make_mock_encoding(pose_bucket="frontal", lighting="bright")
        buckets = _classify_encoding_buckets(enc)
        assert "pose_frontal" in buckets
        assert "lighting_bright" in buckets

    def test_extreme_left_pose(self):
        enc = _make_mock_encoding(pose_bucket="extreme_left")
        buckets = _classify_encoding_buckets(enc)
        assert "pose_extreme_left" in buckets

    def test_glasses_appearance(self):
        enc = _make_mock_encoding(appearance_tags="glasses")
        buckets = _classify_encoding_buckets(enc)
        assert "appearance_glasses" in buckets

    def test_mask_appearance(self):
        enc = _make_mock_encoding(appearance_tags="mask")
        buckets = _classify_encoding_buckets(enc)
        assert "appearance_mask" in buckets

    def test_plain_appearance(self):
        enc = _make_mock_encoding(appearance_tags=None)
        buckets = _classify_encoding_buckets(enc)
        assert "appearance_plain" in buckets

    def test_high_quality_tier(self):
        enc = _make_mock_encoding(quality=0.9)
        buckets = _classify_encoding_buckets(enc)
        assert "quality_high" in buckets

    def test_medium_quality_tier(self):
        enc = _make_mock_encoding(quality=0.6)
        buckets = _classify_encoding_buckets(enc)
        assert "quality_medium" in buckets

    def test_dim_lighting(self):
        enc = _make_mock_encoding(lighting="dim")
        buckets = _classify_encoding_buckets(enc)
        assert "lighting_dim" in buckets

    def test_backlit_maps_to_dim(self):
        enc = _make_mock_encoding(lighting="backlit")
        buckets = _classify_encoding_buckets(enc)
        assert "lighting_dim" in buckets


# ── Coverage Score ───────────────────────────────────────────────

class TestCoverageScore:
    """Test weighted coverage score computation."""

    def test_empty_coverage(self):
        assert _compute_coverage_score(set()) == 0.0

    def test_frontal_only(self):
        score = _compute_coverage_score({"pose_frontal"})
        assert score == 0.20  # frontal weight

    def test_full_pose_coverage(self):
        pose_buckets = {
            "pose_frontal", "pose_left", "pose_right",
            "pose_extreme_left", "pose_extreme_right",
        }
        score = _compute_coverage_score(pose_buckets)
        assert score == 0.60  # 0.20 + 0.12 + 0.12 + 0.08 + 0.08

    def test_full_coverage(self):
        all_buckets = set(COVERAGE_BUCKETS.keys())
        score = _compute_coverage_score(all_buckets)
        assert abs(score - 1.0) < 0.001  # All weights sum to 1.0


class TestCoveragePercentage:
    """Test simple percentage coverage."""

    def test_empty(self):
        assert _compute_coverage_percentage(set()) == 0.0

    def test_half_buckets(self):
        half = set(list(COVERAGE_BUCKETS.keys())[:6])
        pct = _compute_coverage_percentage(half)
        expected = round(100.0 * 6 / len(COVERAGE_BUCKETS), 1)
        assert pct == expected

    def test_full(self):
        all_b = set(COVERAGE_BUCKETS.keys())
        assert _compute_coverage_percentage(all_b) == 100.0


# ── Quality Score ────────────────────────────────────────────────

class TestQualityScore:
    """Test average quality score computation."""

    def test_empty(self):
        assert _compute_quality_score([]) == 0.0

    def test_single_high(self):
        enc = _make_mock_encoding(quality=0.95)
        assert _compute_quality_score([enc]) == 0.95

    def test_mixed_qualities(self):
        encs = [
            _make_mock_encoding(id="e1", quality=0.9),
            _make_mock_encoding(id="e2", quality=0.5),
        ]
        score = _compute_quality_score(encs)
        assert abs(score - 0.7) < 0.001


# ── Recognition Score ────────────────────────────────────────────

class TestRecognitionScore:
    """Test real-world recognition success scoring."""

    def test_empty(self):
        assert _compute_recognition_score([]) == 0.0

    def test_high_effectiveness(self):
        enc = _make_mock_encoding(usage_count=100, success_count=90)
        score = _compute_recognition_score([enc])
        assert score == 0.9

    def test_never_used(self):
        enc = _make_mock_encoding(usage_count=0, success_count=0)
        score = _compute_recognition_score([enc])
        assert score == 0.1  # Low value — never used

    def test_mixed_effectiveness(self):
        encs = [
            _make_mock_encoding(id="e1", usage_count=50, success_count=40),   # 0.8
            _make_mock_encoding(id="e2", usage_count=0, success_count=0),     # 0.1
        ]
        score = _compute_recognition_score(encs)
        assert abs(score - 0.45) < 0.001


# ── Redundancy Penalty ───────────────────────────────────────────

class TestRedundancyPenalty:
    """Test penalty for duplicate pose buckets."""

    def test_no_redundancy(self):
        encs = [
            _make_mock_encoding(id="e1", pose_bucket="frontal"),
            _make_mock_encoding(id="e2", pose_bucket="left"),
        ]
        assert _compute_redundancy_penalty(encs) == 0.0

    def test_some_redundancy(self):
        encs = [
            _make_mock_encoding(id="e1", pose_bucket="frontal"),
            _make_mock_encoding(id="e2", pose_bucket="frontal"),
            _make_mock_encoding(id="e3", pose_bucket="frontal"),  # 3rd frontal = 1 redundant
        ]
        penalty = _compute_redundancy_penalty(encs)
        assert penalty > 0.0

    def test_heavy_redundancy(self):
        encs = [
            _make_mock_encoding(id=f"e{i}", pose_bucket="frontal")
            for i in range(5)
        ]
        penalty = _compute_redundancy_penalty(encs)
        assert penalty == 0.6  # 3 redundant out of 5


# ── Recommendation ───────────────────────────────────────────────

class TestRecommendation:
    """Test top recommendation generation."""

    def test_missing_frontal(self):
        rec = _generate_recommendation(["pose_frontal", "pose_left"])
        assert rec == "Needs frontal face image"

    def test_missing_extreme(self):
        rec = _generate_recommendation(["pose_extreme_left"])
        assert rec == "Needs extreme left profile"

    def test_missing_dim_lighting(self):
        rec = _generate_recommendation(["lighting_dim"])
        assert rec == "Needs low-light image"

    def test_nothing_missing(self):
        rec = _generate_recommendation([])
        assert rec is None


# ── PersonProfile Integration ────────────────────────────────────

class TestPersonProfile:
    """Test full PersonProfile computation."""

    @patch('app.services.gallery_health.SessionLocal')
    def test_empty_gallery_health(self, mock_session_cls):
        """No encodings → health 0%."""
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        mock_person = _make_mock_person()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_person
        mock_db.query.return_value.filter.return_value.all.return_value = []

        profile = compute_person_profile("person-1")

        assert profile.total_encodings == 0
        assert profile.health_score == 0.0
        assert profile.coverage_percentage == 0.0
        assert len(profile.missing_buckets) == len(COVERAGE_BUCKETS)

    @patch('app.services.gallery_health.SessionLocal')
    def test_person_not_found(self, mock_session_cls):
        """Non-existent person returns empty profile."""
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db
        mock_db.query.return_value.filter.return_value.first.return_value = None

        profile = compute_person_profile("nonexistent")

        assert profile.person_name == "Unknown"
        assert profile.total_encodings == 0


# ── Coverage Delta ───────────────────────────────────────────────

class TestCoverageDelta:
    """Test coverage delta projection."""

    @patch('app.services.gallery_health.compute_person_profile')
    @patch('app.services.gallery_health.SessionLocal')
    def test_fills_missing_bucket(self, mock_session_cls, mock_profile):
        """Adding extreme_left to a person missing it → high improvement."""
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        mock_profile.return_value = PersonProfile(
            person_id="p1",
            person_name="Chirag",
            total_encodings=5,
            coverage_percentage=50.0,
            covered_buckets=["pose_frontal", "pose_left", "pose_right",
                           "lighting_bright", "appearance_plain",
                           "quality_high"],
            missing_buckets=["pose_extreme_left", "pose_extreme_right",
                           "lighting_dim", "appearance_glasses",
                           "appearance_mask", "quality_medium"],
            health_score=0.50,
            coverage_score=0.50,
            quality_score=0.80,
            diversity_score=0.60,
            recognition_score=0.70,
            redundancy_penalty=0.0,
        )

        delta = compute_coverage_delta(
            person_id="p1",
            new_pose_bucket="extreme_left",
            new_quality=0.85,
            db=mock_db,
        )

        assert delta.projected_coverage > delta.current_coverage
        assert delta.fills_bucket == "pose_extreme_left"
        assert delta.improvement_value in ("high", "medium")
        assert delta.delta > 0

    @patch('app.services.gallery_health.compute_person_profile')
    @patch('app.services.gallery_health.SessionLocal')
    def test_already_covered(self, mock_session_cls, mock_profile):
        """Adding frontal when frontal exists → no improvement."""
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        mock_profile.return_value = PersonProfile(
            person_id="p1",
            person_name="Chirag",
            total_encodings=5,
            coverage_percentage=50.0,
            covered_buckets=["pose_frontal", "pose_left", "pose_right",
                           "lighting_bright", "appearance_plain",
                           "quality_high"],
            missing_buckets=["pose_extreme_left", "pose_extreme_right",
                           "lighting_dim", "appearance_glasses",
                           "appearance_mask", "quality_medium"],
            health_score=0.50,
            coverage_score=0.50,
            quality_score=0.80,
            diversity_score=0.60,
            recognition_score=0.70,
            redundancy_penalty=0.0,
        )

        delta = compute_coverage_delta(
            person_id="p1",
            new_pose_bucket="frontal",  # Already covered
            new_quality=0.85,
            db=mock_db,
        )

        # No new pose bucket filled (frontal already covered)
        assert delta.fills_bucket is None
        # But quality_high is already covered, so minimal delta
