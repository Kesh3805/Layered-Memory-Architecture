"""Tests for the hybrid search module (BM25 + vector RRF fusion).

Pure unit tests — no database required.  Tests the RRF algorithm,
configuration, and edge cases.
"""

import pytest
from hybrid_search import (
    reciprocal_rank_fusion,
    HybridConfig,
    DEFAULT_CONFIG,
)


# ═══════════════════════════════════════════════════════════════════════════
#  RRF Algorithm Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReciprocalRankFusion:
    """Test the core RRF fusion algorithm."""

    def test_single_list_returns_ranked_order(self):
        """A single ranked list should produce RRF scores in original order."""
        ranked = [(1, 0.9), (2, 0.8), (3, 0.7)]
        result = reciprocal_rank_fusion([ranked], [1.0], k=60)
        ids = [doc_id for doc_id, _score in result]
        assert ids == [1, 2, 3]

    def test_two_identical_lists_boost_scores(self):
        """Documents in both lists should have higher RRF scores."""
        list_a = [(1, 0.9), (2, 0.8), (3, 0.7)]
        list_b = [(1, 0.85), (2, 0.75), (3, 0.65)]
        result = reciprocal_rank_fusion([list_a, list_b], [1.0, 1.0], k=60)
        # Doc 1 appears at rank 1 in both → highest RRF score
        assert result[0][0] == 1
        # All docs should appear
        assert len(result) == 3

    def test_complementary_lists_merge_correctly(self):
        """Documents from different lists should merge by RRF score."""
        vector_results = [(10, 0.95), (20, 0.80)]
        bm25_results = [(30, 5.0), (10, 3.0)]
        result = reciprocal_rank_fusion(
            [vector_results, bm25_results], [1.0, 1.0], k=60,
        )
        ids = [doc_id for doc_id, _ in result]
        # Doc 10 is in both → should be top
        assert ids[0] == 10
        # Doc 30 and 20 are in one list each
        assert set(ids) == {10, 20, 30}

    def test_weights_bias_results(self):
        """Higher weight should favor documents from that list."""
        vector_results = [(1, 0.9)]  # Only in vector
        bm25_results = [(2, 5.0)]    # Only in BM25
        # Vector weight = 3.0, BM25 weight = 1.0
        result = reciprocal_rank_fusion(
            [vector_results, bm25_results], [3.0, 1.0], k=60,
        )
        # Doc 1 (vector) should be ranked higher due to 3× weight
        assert result[0][0] == 1
        assert result[0][1] > result[1][1]

    def test_empty_lists_return_empty(self):
        result = reciprocal_rank_fusion([], [], k=60)
        assert result == []

    def test_one_empty_list(self):
        """One empty list should still return results from the other."""
        results = reciprocal_rank_fusion(
            [[(1, 0.9), (2, 0.8)], []], [1.0, 1.0], k=60,
        )
        assert len(results) == 2

    def test_k_parameter_affects_scores(self):
        """Larger k should produce flatter (more uniform) RRF scores."""
        ranked = [(1, 0.9), (2, 0.5)]
        result_low_k = reciprocal_rank_fusion([ranked], [1.0], k=1)
        result_high_k = reciprocal_rank_fusion([ranked], [1.0], k=1000)

        # With low k, rank difference matters more
        spread_low = result_low_k[0][1] - result_low_k[1][1]
        spread_high = result_high_k[0][1] - result_high_k[1][1]
        assert spread_low > spread_high

    def test_rrf_scores_are_positive(self):
        """All RRF scores should be positive."""
        ranked = [(i, float(i)) for i in range(1, 20)]
        result = reciprocal_rank_fusion([ranked], [1.0], k=60)
        for _doc_id, score in result:
            assert score > 0

    def test_duplicate_doc_ids_across_lists(self):
        """Same doc ID in multiple lists should accumulate score."""
        list_a = [(1, 0.9)]
        list_b = [(1, 0.8)]
        list_c = [(1, 0.7)]
        result = reciprocal_rank_fusion([list_a, list_b, list_c], [1.0, 1.0, 1.0], k=60)
        assert len(result) == 1
        # Score should be 3 * (1 / (60 + 1))
        expected = 3 * (1.0 / 61)
        assert abs(result[0][1] - expected) < 0.001


# ═══════════════════════════════════════════════════════════════════════════
#  HybridConfig Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHybridConfig:
    def test_default_config_values(self):
        cfg = DEFAULT_CONFIG
        assert cfg.rrf_k == 60
        assert cfg.vector_weight == 1.0
        assert cfg.bm25_weight == 1.0
        assert cfg.candidate_multiplier == 3
        assert cfg.vector_min_similarity == 0.0
        assert cfg.fallback_to_vector is True

    def test_custom_config(self):
        cfg = HybridConfig(rrf_k=30, vector_weight=2.0, bm25_weight=0.5)
        assert cfg.rrf_k == 30
        assert cfg.vector_weight == 2.0
        assert cfg.bm25_weight == 0.5

    def test_frozen_dataclass(self):
        cfg = DEFAULT_CONFIG
        with pytest.raises(AttributeError):
            cfg.rrf_k = 100  # type: ignore

    def test_candidate_multiplier_scaling(self):
        """candidate_multiplier should determine how many candidates each arm fetches."""
        cfg = HybridConfig(candidate_multiplier=5)
        k = 4
        assert k * cfg.candidate_multiplier == 20


# ═══════════════════════════════════════════════════════════════════════════
#  Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_document_both_lists(self):
        """Single document appearing in both lists."""
        result = reciprocal_rank_fusion(
            [[(42, 0.9)], [(42, 3.0)]], [1.0, 1.0], k=60,
        )
        assert len(result) == 1
        assert result[0][0] == 42

    def test_large_number_of_documents(self):
        """RRF should handle hundreds of documents efficiently."""
        large_list = [(i, 1.0 / (i + 1)) for i in range(500)]
        result = reciprocal_rank_fusion([large_list], [1.0], k=60)
        assert len(result) == 500
        # First doc should have highest score
        assert result[0][0] == 0

    def test_zero_weight_ignores_list(self):
        """A list with weight 0 should not contribute to scores."""
        list_a = [(1, 0.9)]
        list_b = [(2, 5.0)]
        result = reciprocal_rank_fusion([list_a, list_b], [1.0, 0.0], k=60)
        # Doc 2 has 0 weight → 0 score, but doc 1 has positive score
        assert result[0][0] == 1
        assert result[1][1] == 0.0

    def test_many_lists_fusion(self):
        """RRF should work with more than 2 ranked lists."""
        lists = [[(i, float(10 - i))] for i in range(5)]
        weights = [1.0] * 5
        result = reciprocal_rank_fusion(lists, weights, k=60)
        # Each doc only appears in one list, all at rank 1
        # So all have the same RRF score
        assert len(result) == 5
        scores = [s for _, s in result]
        assert all(abs(s - scores[0]) < 0.001 for s in scores)
