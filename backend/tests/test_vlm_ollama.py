"""
Tests for OllamaReviewer, VLM cache, and provider factory.

All tests use mocks — no real Ollama, OpenAI, or DB calls.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from app.services.vlm_service import (
    OllamaReviewer,
    DisabledReviewer,
    OpenAIReviewer,
    VLMResult,
    VLMCache,
    get_vlm_reviewer,
)


# ── OllamaReviewer Tests ────────────────────────────────────────

class TestOllamaReviewer:
    """Tests for the Ollama VLM reviewer."""

    def setup_method(self):
        self.reviewer = OllamaReviewer()

    def test_ollama_single_comparison(self):
        """Single 1:1 comparison with valid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "same_person": True,
                    "confidence": 91,
                    "quality": "good",
                    "pose": "extreme_left",
                    "reasoning": "Jawline structure matches. Eye spacing consistent.",
                    "similarities": ["jawline", "eye spacing"],
                    "differences": ["lighting"],
                })
            }
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        self.reviewer._client = mock_client

        # Mock image preparation
        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")

        assert isinstance(result, VLMResult)
        assert result.agreement == 0.91
        assert result.confidence == "high"
        assert result.quality == "good"
        assert result.detected_pose == "extreme_left"
        assert "jawline" in result.similarities
        assert result.error is None

    def test_ollama_negative_result(self):
        """VLM says same_person=False with high confidence."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "same_person": False,
                    "confidence": 85,
                    "quality": "good",
                    "pose": "frontal",
                    "reasoning": "Different nose bridge shape.",
                    "similarities": [],
                    "differences": ["nose bridge", "chin shape"],
                })
            }
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        self.reviewer._client = mock_client

        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")

        # same_person=False, confidence=85 → agreement = (100-85)/100 = 0.15
        assert result.agreement == 0.15
        assert result.confidence == "high"

    def test_ollama_timeout_fallback(self):
        """Timeout produces neutral fallback, no crash."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")
        self.reviewer._client = mock_client

        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")

        assert isinstance(result, VLMResult)
        assert result.agreement == 0.5
        assert result.confidence == "low"
        assert result.error is not None
        assert "timed out" in result.error.lower()

    def test_ollama_connection_refused(self):
        """Connection refused produces neutral fallback, logs once."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        self.reviewer._client = mock_client

        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")

        assert result.agreement == 0.5
        assert result.error is not None

        # Second call should still work (not crash)
        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result2 = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")
        assert result2.agreement == 0.5

    def test_ollama_invalid_json(self):
        """Garbled response produces fallback VLMResult."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "This is not JSON at all!!!"}
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        self.reviewer._client = mock_client

        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")

        assert result.agreement == 0.5
        assert result.error == "parse_error"

    def test_ollama_markdown_fenced_json(self):
        """Response wrapped in code fences is still parsed correctly."""
        json_content = json.dumps({
            "same_person": True,
            "confidence": 78,
            "quality": "fair",
            "pose": "left",
            "reasoning": "Moderate match.",
            "similarities": ["forehead"],
            "differences": [],
        })

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": f"```json\n{json_content}\n```"}
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        self.reviewer._client = mock_client

        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")

        assert result.agreement == 0.78
        assert result.quality == "fair"
        assert result.detected_pose == "left"

    def test_ollama_missing_image(self):
        """Non-existent image path produces graceful fallback."""
        with patch.object(self.reviewer, '_prepare_image', return_value=None):
            result = self.reviewer.compare_faces(
                "nonexistent_a.jpg", "nonexistent_b.jpg"
            )

        assert result.agreement == 0.5
        assert result.error == "image_load_failed"

    def test_ollama_api_error(self):
        """Non-200 status code produces fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        self.reviewer._client = mock_client

        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")

        assert result.agreement == 0.5
        assert "500" in result.error

    def test_ollama_empty_response(self):
        """Empty response content produces fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": ""}}

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        self.reviewer._client = mock_client

        with patch.object(self.reviewer, '_prepare_image', return_value="base64data"):
            result = self.reviewer.compare_faces("unknown.jpg", "gallery.jpg")

        assert result.agreement == 0.5
        assert result.error == "empty_response"

    def test_image_preparation(self):
        """Crop load → 256×256 JPEG → valid base64."""
        import numpy as np
        import cv2
        import tempfile
        import os

        # Create a test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 2] = 200  # Red-ish

        tmp_path = os.path.join(
            tempfile.gettempdir(), "test_vlm_crop.jpg"
        )
        cv2.imwrite(tmp_path, img)

        try:
            result = self.reviewer._prepare_image(tmp_path)
            assert result is not None

            # Verify it's valid base64
            import base64
            decoded = base64.b64decode(result)
            assert len(decoded) > 0

            # Decode back to image to verify dimensions
            nparr = np.frombuffer(decoded, np.uint8)
            decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            assert decoded_img.shape == (256, 256, 3)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ── VLM Cache Tests ──────────────────────────────────────────────

class TestVLMCache:
    """Tests for VLM result caching."""

    def test_cache_hit(self):
        """Same hash returns cached result."""
        cache = VLMCache(max_entries=100, ttl_hours=24)
        result = VLMResult(agreement=0.91, confidence="high")

        cache.put("cluster1", "person1", 5, result)
        cached = cache.get("cluster1", "person1", 5)

        assert cached is not None
        assert cached.agreement == 0.91

    def test_cache_miss_different_version(self):
        """Gallery version change invalidates cache."""
        cache = VLMCache(max_entries=100, ttl_hours=24)
        result = VLMResult(agreement=0.91, confidence="high")

        cache.put("cluster1", "person1", 5, result)
        cached = cache.get("cluster1", "person1", 6)  # Different version

        assert cached is None

    def test_cache_miss_different_person(self):
        """Different person ID is a cache miss."""
        cache = VLMCache(max_entries=100, ttl_hours=24)
        result = VLMResult(agreement=0.91, confidence="high")

        cache.put("cluster1", "person1", 5, result)
        cached = cache.get("cluster1", "person2", 5)

        assert cached is None

    def test_cache_eviction(self):
        """Exceed max entries triggers LRU eviction."""
        cache = VLMCache(max_entries=3, ttl_hours=24)

        for i in range(5):
            cache.put(f"cluster{i}", "person1", 1, VLMResult(agreement=float(i) / 10))

        assert cache.size == 3
        # Oldest entries (0, 1) should be evicted
        assert cache.get("cluster0", "person1", 1) is None
        assert cache.get("cluster1", "person1", 1) is None
        # Newest entries should still be there
        assert cache.get("cluster4", "person1", 1) is not None

    def test_cache_ttl_expiry(self):
        """Expired entries are not returned."""
        import time

        cache = VLMCache(max_entries=100, ttl_hours=0)  # 0 hour TTL = immediate expiry
        cache._ttl_seconds = 0  # Immediate expiry

        result = VLMResult(agreement=0.91)
        cache.put("cluster1", "person1", 5, result)

        time.sleep(0.01)  # Ensure some time passes
        cached = cache.get("cluster1", "person1", 5)
        assert cached is None

    def test_cache_clear(self):
        """Clear removes all entries."""
        cache = VLMCache(max_entries=100, ttl_hours=24)
        cache.put("c1", "p1", 1, VLMResult())
        cache.put("c2", "p2", 1, VLMResult())

        assert cache.size == 2
        cache.clear()
        assert cache.size == 0


# ── Provider Factory Tests ───────────────────────────────────────

class TestProviderFactory:
    """Tests for the VLM provider factory."""

    @patch('app.services.vlm_service.settings')
    def test_provider_factory_ollama(self, mock_settings):
        """provider=ollama returns OllamaReviewer."""
        mock_settings.IDENTITY_REVIEW_VLM_PROVIDER = "ollama"
        mock_settings.OLLAMA_HOST = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "qwen2.5vl:7b"
        mock_settings.GALLERY_OPT_VLM_CACHE_TTL_HOURS = 168

        reviewer = get_vlm_reviewer()
        assert isinstance(reviewer, OllamaReviewer)

    @patch('app.services.vlm_service.settings')
    def test_provider_factory_disabled(self, mock_settings):
        """provider=disabled returns DisabledReviewer."""
        mock_settings.IDENTITY_REVIEW_VLM_PROVIDER = "disabled"
        mock_settings.GALLERY_OPT_VLM_CACHE_TTL_HOURS = 168

        reviewer = get_vlm_reviewer()
        assert isinstance(reviewer, DisabledReviewer)

    @patch('app.services.vlm_service.settings')
    def test_provider_factory_openai_with_key(self, mock_settings):
        """provider=openai + key returns OpenAIReviewer."""
        mock_settings.IDENTITY_REVIEW_VLM_PROVIDER = "openai"
        mock_settings.IDENTITY_REVIEW_VLM_API_KEY = "sk-test-key"
        mock_settings.GALLERY_OPT_VLM_CACHE_TTL_HOURS = 168

        reviewer = get_vlm_reviewer()
        assert isinstance(reviewer, OpenAIReviewer)

    @patch('app.services.vlm_service.settings')
    def test_provider_factory_openai_no_key(self, mock_settings):
        """provider=openai without key falls back to DisabledReviewer."""
        mock_settings.IDENTITY_REVIEW_VLM_PROVIDER = "openai"
        mock_settings.IDENTITY_REVIEW_VLM_API_KEY = ""
        mock_settings.GALLERY_OPT_VLM_CACHE_TTL_HOURS = 168

        reviewer = get_vlm_reviewer()
        assert isinstance(reviewer, DisabledReviewer)

    @patch('app.services.vlm_service.settings')
    def test_provider_factory_unknown(self, mock_settings):
        """Unknown provider falls back to DisabledReviewer."""
        mock_settings.IDENTITY_REVIEW_VLM_PROVIDER = "nonexistent"
        mock_settings.GALLERY_OPT_VLM_CACHE_TTL_HOURS = 168

        reviewer = get_vlm_reviewer()
        assert isinstance(reviewer, DisabledReviewer)


# ── Disabled Reviewer Tests ──────────────────────────────────────

class TestDisabledReviewer:
    """Verify DisabledReviewer returns neutral scores."""

    def test_disabled_returns_neutral(self):
        reviewer = DisabledReviewer()
        result = reviewer.compare_faces("a.jpg", "b.jpg")

        assert result.agreement == 0.5
        assert result.confidence == "low"
        assert result.error is None
