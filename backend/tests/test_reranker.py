"""Tests for the cross-encoder reranker module.

Tests the reranker's passthrough behavior, sorting logic, and edge cases.
Does NOT load the actual cross-encoder model (those would be integration tests).
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Passthrough Tests (no model loaded)
# ═══════════════════════════════════════════════════════════════════════════

class TestRerankerPassthrough:
    """When the reranker is disabled or unavailable, functions pass through."""

    def setup_method(self):
        """Reset reranker state before each test."""
        import reranker
        reranker._cross_encoder = None
        reranker._load_failed = False

    @patch("settings.settings")
    def test_disabled_returns_passthrough(self, mock_settings):
        """When RERANKER_ENABLED=False, rerank() returns original order."""
        mock_settings.RERANKER_ENABLED = False
        mock_settings.RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_settings.RERANKER_TOP_K = 4
        mock_settings.RETRIEVAL_K = 4
        from reranker import rerank
        docs = ["doc A", "doc B", "doc C"]
        result = rerank("test query", docs)
        # Should return docs in original order with score 0.0
        assert len(result) == 3
        assert result[0] == ("doc A", 0.0)
        assert result[1] == ("doc B", 0.0)
        assert result[2] == ("doc C", 0.0)

    @patch("settings.settings")
    def test_passthrough_respects_top_k(self, mock_settings):
        """Passthrough should still truncate to top_k."""
        mock_settings.RERANKER_ENABLED = False
        mock_settings.RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_settings.RERANKER_TOP_K = 2
        mock_settings.RETRIEVAL_K = 4
        from reranker import rerank
        docs = ["doc A", "doc B", "doc C", "doc D"]
        result = rerank("test query", docs, top_k=2)
        assert len(result) == 2

    def test_empty_documents_returns_empty(self):
        from reranker import rerank
        result = rerank("query", [])
        assert result == []

    @patch("settings.settings")
    def test_rerank_with_metadata_passthrough(self, mock_settings):
        """rerank_with_metadata should pass through when model unavailable."""
        mock_settings.RERANKER_ENABLED = False
        mock_settings.RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_settings.RERANKER_TOP_K = 4
        mock_settings.RETRIEVAL_K = 4
        from reranker import rerank_with_metadata
        docs = [("text A", {"source": "a"}), ("text B", {"source": "b"})]
        result = rerank_with_metadata("query", docs)
        assert len(result) == 2
        assert result[0] == ("text A", {"source": "a"}, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Mock Cross-Encoder Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRerankerWithMockModel:
    """Test reranking logic with a mocked cross-encoder."""

    def setup_method(self):
        import reranker
        reranker._cross_encoder = None
        reranker._load_failed = False

    @patch("reranker._get_cross_encoder")
    @patch("settings.settings")
    def test_rerank_sorts_by_score(self, mock_settings, mock_get_encoder):
        """Reranked results should be sorted by descending cross-encoder score."""
        mock_settings.RERANKER_TOP_K = 3
        mock_settings.RETRIEVAL_K = 4

        mock_encoder = MagicMock()
        # Return scores: doc C > doc A > doc B
        mock_encoder.predict.return_value = np.array([0.5, 0.2, 0.9])
        mock_get_encoder.return_value = mock_encoder

        from reranker import rerank
        docs = ["doc A", "doc B", "doc C"]
        result = rerank("test query", docs, top_k=3)

        assert result[0][0] == "doc C"  # Highest score
        assert result[1][0] == "doc A"
        assert result[2][0] == "doc B"  # Lowest score
        assert result[0][1] > result[1][1] > result[2][1]

    @patch("reranker._get_cross_encoder")
    @patch("settings.settings")
    def test_rerank_truncates_to_top_k(self, mock_settings, mock_get_encoder):
        """Should only return top_k results."""
        mock_settings.RERANKER_TOP_K = 2
        mock_settings.RETRIEVAL_K = 4

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array([0.3, 0.9, 0.1, 0.7])
        mock_get_encoder.return_value = mock_encoder

        from reranker import rerank
        docs = ["A", "B", "C", "D"]
        result = rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0][0] == "B"  # score 0.9
        assert result[1][0] == "D"  # score 0.7

    @patch("reranker._get_cross_encoder")
    @patch("settings.settings")
    def test_rerank_with_metadata_sorts(self, mock_settings, mock_get_encoder):
        """rerank_with_metadata should sort and preserve metadata."""
        mock_settings.RERANKER_TOP_K = 3
        mock_settings.RETRIEVAL_K = 4

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array([0.3, 0.8, 0.5])
        mock_get_encoder.return_value = mock_encoder

        from reranker import rerank_with_metadata
        docs = [
            ("text A", {"id": 1}),
            ("text B", {"id": 2}),
            ("text C", {"id": 3}),
        ]
        result = rerank_with_metadata("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0][0] == "text B"
        assert result[0][1] == {"id": 2}
        assert result[0][2] > result[1][2]

    @patch("reranker._get_cross_encoder")
    @patch("settings.settings")
    def test_encoder_called_with_correct_pairs(self, mock_settings, mock_get_encoder):
        """Cross-encoder should receive (query, doc) pairs."""
        mock_settings.RERANKER_TOP_K = 3
        mock_settings.RETRIEVAL_K = 4

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array([0.5, 0.3])
        mock_get_encoder.return_value = mock_encoder

        from reranker import rerank
        rerank("my query", ["doc1", "doc2"], top_k=2)

        call_args = mock_encoder.predict.call_args[0][0]
        assert call_args == [("my query", "doc1"), ("my query", "doc2")]


# ═══════════════════════════════════════════════════════════════════════════
#  Availability Check
# ═══════════════════════════════════════════════════════════════════════════

class TestRerankerAvailability:
    def setup_method(self):
        import reranker
        reranker._cross_encoder = None
        reranker._load_failed = False

    @patch("settings.settings")
    def test_is_available_when_disabled(self, mock_settings):
        mock_settings.RERANKER_ENABLED = False
        from reranker import is_available
        assert is_available() is False

    @patch("reranker._get_cross_encoder")
    def test_is_available_with_model(self, mock_get_encoder):
        mock_get_encoder.return_value = MagicMock()
        from reranker import is_available
        assert is_available() is True

    @patch("reranker._get_cross_encoder")
    def test_is_available_without_model(self, mock_get_encoder):
        mock_get_encoder.return_value = None
        from reranker import is_available
        assert is_available() is False
