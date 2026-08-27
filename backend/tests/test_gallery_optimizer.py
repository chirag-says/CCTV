"""
Tests for Gallery Optimizer worker pipeline: quality gate, confidence gate,
review priority, cluster outlier detection, and gallery image selection.

All tests use mocks — no real DB, VLM, or Ollama calls.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.identity_review_worker import (
    IdentityReviewWorker,
    _compute_review_priority,
)


# ── Review Priority Tests ────────────────────────────────────────

class TestReviewPriority:
    """Test priority assignment based on multiple signals."""

    def test_priority_critical(self):
        """High sim + high improvement + big cluster → critical."""
        priority = _compute_review_priority(
            similarity=0.85,
            gallery_improvement_value="high",
            cluster_size=10,
        )
        assert priority == "critical"

    def test_priority_high(self):
        """High sim + medium improvement → high."""
        priority = _compute_review_priority(
            similarity=0.70,
            gallery_improvement_value="medium",
            cluster_size=3,
        )
        assert priority == "high"

    def test_priority_medium(self):
        """Moderate sim + some improvement → medium."""
        priority = _compute_review_priority(
            similarity=0.60,
            gallery_improvement_value="low",
            cluster_size=2,
        )
        assert priority == "medium"

    def test_priority_low(self):
        """Low sim + no improvement → low."""
        priority = _compute_review_priority(
            similarity=0.40,
            gallery_improvement_value="none",
            cluster_size=1,
        )
        assert priority == "low"

    def test_priority_high_sim_no_improvement(self):
        """High sim but no improvement → low (no gallery value)."""
        priority = _compute_review_priority(
            similarity=0.90,
            gallery_improvement_value="none",
            cluster_size=10,
        )
        assert priority == "low"

    def test_priority_low_sim_high_improvement(self):
        """Low sim even with high improvement → low."""
        priority = _compute_review_priority(
            similarity=0.40,
            gallery_improvement_value="high",
            cluster_size=10,
        )
        assert priority == "low"

    def test_priority_boundary_critical(self):
        """Exactly at critical thresholds."""
        priority = _compute_review_priority(
            similarity=0.80,
            gallery_improvement_value="high",
            cluster_size=5,
        )
        assert priority == "critical"

    def test_priority_just_below_critical(self):
        """Just below critical sim → high."""
        priority = _compute_review_priority(
            similarity=0.79,
            gallery_improvement_value="high",
            cluster_size=5,
        )
        assert priority == "high"


# ── Confidence Gate Tests ────────────────────────────────────────

def _gate_settings(mock_settings):
    """
    Thresholds the gate actually reads.

    These tests previously stubbed GALLERY_OPT_VLM_SKIP_HIGH/LOW, names the
    gate stopped using when the safety-valve review window was added — every
    comparison was running against a MagicMock.
    """
    mock_settings.IDENTITY_REVIEW_REVIEW_MIN = 0.40
    mock_settings.IDENTITY_REVIEW_REVIEW_MAX = 0.98
    mock_settings.IDENTITY_REVIEW_TIER_HIGH = 0.85
    mock_settings.IDENTITY_REVIEW_TIER_LOW = 0.55


class TestConfidenceGate:
    """Test the three-tier confidence gate."""

    def setup_method(self):
        self.worker = IdentityReviewWorker()

    @patch('app.services.identity_review_worker.settings')
    def test_high_confidence_skips_vlm(self, mock_settings):
        """similarity > 0.85 → skip VLM, queue directly."""
        _gate_settings(mock_settings)
        assert self.worker._confidence_gate(0.90) == "skip_vlm_queue"

    @patch('app.services.identity_review_worker.settings')
    def test_medium_confidence_calls_vlm(self, mock_settings):
        """similarity 0.55–0.85 → call VLM."""
        _gate_settings(mock_settings)
        assert self.worker._confidence_gate(0.70) == "call_vlm"

    @patch('app.services.identity_review_worker.settings')
    def test_low_confidence_rejects(self, mock_settings):
        """Below the safety valve → reject without spending a VLM call."""
        _gate_settings(mock_settings)
        assert self.worker._confidence_gate(0.30) == "reject"

    @patch('app.services.identity_review_worker.settings')
    def test_inside_valve_below_tier_low_rejects(self, mock_settings):
        """Above REVIEW_MIN but below TIER_LOW → still rejected."""
        _gate_settings(mock_settings)
        assert self.worker._confidence_gate(0.45) == "reject"

    @patch('app.services.identity_review_worker.settings')
    def test_above_review_max_auto_queues(self, mock_settings):
        """Near-certain match → queue for a human without a VLM call."""
        _gate_settings(mock_settings)
        assert self.worker._confidence_gate(0.99) == "auto_queue"

    @patch('app.services.identity_review_worker.settings')
    def test_boundary_high(self, mock_settings):
        """Exactly at high threshold → skip VLM."""
        _gate_settings(mock_settings)
        assert self.worker._confidence_gate(0.85) == "skip_vlm_queue"

    @patch('app.services.identity_review_worker.settings')
    def test_boundary_low(self, mock_settings):
        """Exactly at low threshold → call VLM."""
        _gate_settings(mock_settings)
        assert self.worker._confidence_gate(0.55) == "call_vlm"


# ── Cluster Outlier Detection Tests ──────────────────────────────

class TestOutlierDetection:
    """Test cluster outlier detection."""

    def setup_method(self):
        self.worker = IdentityReviewWorker()

    @patch('app.services.identity_review_worker.settings')
    def test_outlier_detection(self, mock_settings):
        """Synthetic embeddings with one outlier → correctly flagged."""
        mock_settings.GALLERY_OPT_OUTLIER_FACTOR = 1.5

        # Create a centroid
        rng = np.random.RandomState(42)
        centroid = rng.randn(512).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        # Create 4 inliers (close to centroid)
        members = []
        for i in range(4):
            noise = rng.randn(512).astype(np.float32) * 0.05
            inlier = centroid + noise
            inlier /= np.linalg.norm(inlier)
            members.append((MagicMock(), inlier))

        # Create 1 outlier (very different)
        outlier_vec = rng.randn(512).astype(np.float32)
        outlier_vec /= np.linalg.norm(outlier_vec)
        members.append((MagicMock(), outlier_vec))

        inliers, outliers = self.worker._detect_outliers(members, centroid)

        assert len(outliers) >= 1  # At least the random vector is an outlier
        assert len(inliers) + len(outliers) == 5

    @patch('app.services.identity_review_worker.settings')
    def test_no_outliers(self, mock_settings):
        """All members close to centroid → no outliers."""
        mock_settings.GALLERY_OPT_OUTLIER_FACTOR = 1.5

        rng = np.random.RandomState(42)
        centroid = rng.randn(512).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        members = []
        for i in range(5):
            noise = rng.randn(512).astype(np.float32) * 0.01  # Very small noise
            inlier = centroid + noise
            inlier /= np.linalg.norm(inlier)
            members.append((MagicMock(), inlier))

        inliers, outliers = self.worker._detect_outliers(members, centroid)

        assert len(outliers) == 0
        assert len(inliers) == 5

    @patch('app.services.identity_review_worker.settings')
    def test_small_cluster_no_outlier_check(self, mock_settings):
        """Clusters with < 3 members skip outlier detection."""
        mock_settings.GALLERY_OPT_OUTLIER_FACTOR = 1.5

        centroid = np.random.randn(512).astype(np.float32)
        members = [
            (MagicMock(), np.random.randn(512).astype(np.float32)),
            (MagicMock(), np.random.randn(512).astype(np.float32)),
        ]

        inliers, outliers = self.worker._detect_outliers(members, centroid)

        assert len(inliers) == 2
        assert len(outliers) == 0


# ── Gallery Image Selection Tests ────────────────────────────────

class TestGalleryImageSelection:
    """Test ArcFace-driven best gallery image selection."""

    @patch('app.db.session.SessionLocal')
    def test_best_gallery_image(self, mock_session_cls):
        """ArcFace picks highest similarity gallery image."""
        from app.services.vlm_service import select_best_gallery_image

        # Create mock encodings with known embeddings
        rng = np.random.RandomState(42)
        unknown = rng.randn(512).astype(np.float32)
        unknown /= np.linalg.norm(unknown)

        # Best match: very similar to unknown
        best_enc_data = (unknown + rng.randn(512).astype(np.float32) * 0.01).astype(np.float32)
        best_enc_data /= np.linalg.norm(best_enc_data)

        # Worse match: quite different
        worse_enc_data = rng.randn(512).astype(np.float32)
        worse_enc_data /= np.linalg.norm(worse_enc_data)

        best_enc = MagicMock()
        best_enc.encoding_data = best_enc_data.tobytes()
        best_enc.source_image = "best_match.jpg"

        worse_enc = MagicMock()
        worse_enc.encoding_data = worse_enc_data.tobytes()
        worse_enc.source_image = "worse_match.jpg"

        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = [
            worse_enc, best_enc,
        ]

        result = select_best_gallery_image("person-1", unknown)
        assert result == "best_match.jpg"

    @patch('app.db.session.SessionLocal')
    def test_no_gallery_images(self, mock_session_cls):
        """No gallery images → returns None."""
        from app.services.vlm_service import select_best_gallery_image

        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = []

        result = select_best_gallery_image("person-1", np.zeros(512))
        assert result is None


# ── Worker Lifecycle Tests ───────────────────────────────────────

class TestWorkerLifecycle:
    """Test worker start/stop lifecycle."""

    def test_worker_starts_and_stops(self):
        worker = IdentityReviewWorker()
        assert not worker.is_running

        # We won't actually start the thread (it would try DB access),
        # just verify the interface
        assert worker.stats["running"] is False
        assert worker.stats["run_count"] == 0

    def test_worker_stats(self):
        worker = IdentityReviewWorker()
        stats = worker.stats
        assert "running" in stats
        assert "run_count" in stats
        assert "consecutive_failures" in stats
