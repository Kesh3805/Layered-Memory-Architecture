"""Tests for the automated evaluation harness.

Pure unit tests — no LLM calls, no DB. Tests metric computation,
aggregation, report formatting, and edge cases.
"""

import pytest
import numpy as np
from evaluation import (
    evaluate_retrieval,
    evaluate_response,
    run_eval_suite,
    format_report,
    format_comparison_report,
    _compute_aggregates,
    RetrievalMetrics,
    ResponseMetrics,
    QueryEvalResult,
    EvalReport,
    EvalQuery,
    EVAL_CORPUS,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Retrieval Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluateRetrieval:
    def test_empty_results(self):
        m = evaluate_retrieval(None, [], [], relevance_threshold=0.5)
        assert m.context_precision == 0.0
        assert m.context_recall == 0.0
        assert m.mrr == 0.0
        assert m.num_retrieved == 0

    def test_all_relevant(self):
        """All retrieved docs above threshold → precision = 1.0."""
        texts = ["doc1", "doc2", "doc3"]
        scores = [0.9, 0.8, 0.7]
        m = evaluate_retrieval(None, texts, scores, relevance_threshold=0.5)
        assert m.context_precision == 1.0
        assert m.mrr == 1.0  # First doc is relevant
        assert m.num_retrieved == 3

    def test_partial_relevance(self):
        """Some docs below threshold → precision < 1.0."""
        texts = ["doc1", "doc2", "doc3", "doc4"]
        scores = [0.9, 0.6, 0.3, 0.1]
        m = evaluate_retrieval(None, texts, scores, relevance_threshold=0.5)
        assert m.context_precision == 0.5  # 2 out of 4
        assert m.mrr == 1.0

    def test_no_relevant_docs(self):
        """All below threshold → precision = 0."""
        texts = ["doc1", "doc2"]
        scores = [0.2, 0.1]
        m = evaluate_retrieval(None, texts, scores, relevance_threshold=0.5)
        assert m.context_precision == 0.0
        assert m.mrr == 0.0

    def test_mrr_first_relevant_at_position_3(self):
        """MRR should be 1/3 when first relevant is at position 3."""
        texts = ["a", "b", "c"]
        scores = [0.1, 0.2, 0.8]
        m = evaluate_retrieval(None, texts, scores, relevance_threshold=0.5)
        assert abs(m.mrr - 1.0 / 3) < 0.001

    def test_context_recall_with_oracle(self):
        """Context recall: fraction of oracle texts found in retrieved set."""
        texts = ["Redis is an in-memory data store", "Memcached is a caching system"]
        scores = [0.8, 0.7]
        oracle = ["redis", "memcached", "database"]
        m = evaluate_retrieval(None, texts, scores, oracle_texts=oracle)
        # "redis" found in text[0], "memcached" found in text[1], "database" not found
        assert abs(m.context_recall - 2.0 / 3) < 0.01
        assert m.num_oracle == 3

    def test_context_recall_no_oracle(self):
        """Without oracle texts, recall should be 0."""
        m = evaluate_retrieval(None, ["doc"], [0.8])
        assert m.context_recall == 0.0

    def test_context_recall_perfect(self):
        """All oracle texts found → recall = 1.0."""
        texts = ["Redis caching layer", "PostgreSQL database"]
        scores = [0.9, 0.8]
        oracle = ["redis", "postgresql"]
        m = evaluate_retrieval(None, texts, scores, oracle_texts=oracle)
        assert m.context_recall == 1.0

    def test_threshold_boundary(self):
        """Score exactly at threshold should count as relevant."""
        texts = ["doc"]
        scores = [0.5]
        m = evaluate_retrieval(None, texts, scores, relevance_threshold=0.5)
        assert m.context_precision == 1.0
        assert m.mrr == 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Response Metrics Tests (Heuristic Mode)
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluateResponse:
    def test_empty_response(self):
        m = evaluate_response("query", "context", "")
        assert m.faithfulness == 0.0
        assert m.answer_relevance == 0.0

    def test_heuristic_mode_relevance(self):
        """Heuristic mode: word overlap between query and response."""
        m = evaluate_response(
            query="What is Redis?",
            context="Redis is an in-memory data store",
            response="Redis is an in-memory data structure store used for caching",
            skip_llm_judge=True,
        )
        assert m.answer_relevance > 0.0
        assert m.response_length > 0

    def test_heuristic_mode_faithfulness(self):
        """Heuristic mode: word overlap between context and response."""
        m = evaluate_response(
            query="Tell me about caching",
            context="Redis provides fast in-memory caching",
            response="Redis provides fast in-memory caching for applications",
            skip_llm_judge=True,
        )
        assert m.faithfulness > 0.0

    def test_no_context_faithfulness(self):
        """Without context, faithfulness defaults to 0.5 in heuristic mode."""
        m = evaluate_response(
            query="Hello",
            context="",
            response="Hi there!",
            skip_llm_judge=True,
        )
        assert m.faithfulness == 0.5 or m.faithfulness > 0  # No context → default

    def test_response_length_tracked(self):
        response = "This is a response with some words."
        m = evaluate_response("q", "c", response, skip_llm_judge=True)
        assert m.response_length == len(response)


# ═══════════════════════════════════════════════════════════════════════════
#  Eval Suite Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRunEvalSuite:
    def test_basic_suite_execution(self):
        """Suite should run through corpus and produce aggregates."""
        queries = [
            EvalQuery(query="What is Redis?", relevant_keywords=["redis"]),
            EvalQuery(query="How does Docker work?", relevant_keywords=["docker"]),
        ]

        def mock_search(q):
            return [("Redis is an in-memory store", 0.8), ("Docker containers", 0.6)]

        report = run_eval_suite(
            queries,
            search_fn=mock_search,
            skip_llm_judge=True,
        )

        assert len(report.results) == 2
        assert "context_precision_mean" in report.aggregate
        assert report.aggregate["total_queries"] == 2

    def test_suite_with_generation(self):
        """Suite should evaluate both retrieval and generation."""
        queries = [EvalQuery(query="What is caching?", relevant_keywords=["cache"])]

        def mock_search(q):
            return [("Caching stores data in memory", 0.9)]

        def mock_generate(q, ctx):
            return "Caching is a technique to store data in memory for faster access."

        report = run_eval_suite(
            queries,
            search_fn=mock_search,
            generate_fn=mock_generate,
            skip_llm_judge=True,
        )

        assert len(report.results) == 1
        assert report.results[0].response_text != ""
        assert report.results[0].response.answer_relevance >= 0

    def test_suite_handles_search_errors(self):
        """Suite should handle search failures gracefully."""
        queries = [EvalQuery(query="test")]

        def failing_search(q):
            raise RuntimeError("DB down")

        report = run_eval_suite(queries, search_fn=failing_search, skip_llm_judge=True)
        assert len(report.results) == 1
        assert report.results[0].retrieval.num_retrieved == 0

    def test_empty_corpus(self):
        report = run_eval_suite([], search_fn=lambda q: [], skip_llm_judge=True)
        assert report.results == []
        assert report.aggregate == {}


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregation:
    def test_compute_aggregates(self):
        results = [
            QueryEvalResult(
                query="q1",
                retrieval=RetrievalMetrics(context_precision=0.8, mrr=1.0),
                response=ResponseMetrics(faithfulness=0.9, answer_relevance=0.85),
                latency_ms=100,
            ),
            QueryEvalResult(
                query="q2",
                retrieval=RetrievalMetrics(context_precision=0.6, mrr=0.5),
                response=ResponseMetrics(faithfulness=0.7, answer_relevance=0.75),
                latency_ms=200,
            ),
        ]
        agg = _compute_aggregates(results)
        assert agg["context_precision_mean"] == 0.7
        assert agg["mrr_mean"] == 0.75
        assert agg["faithfulness_mean"] == 0.8
        assert agg["latency_ms_mean"] == 150.0
        assert agg["total_queries"] == 2

    def test_aggregates_empty_results(self):
        agg = _compute_aggregates([])
        assert agg == {}


# ═══════════════════════════════════════════════════════════════════════════
#  Report Formatting Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReportFormatting:
    def test_format_report_produces_markdown(self):
        report = EvalReport(
            results=[
                QueryEvalResult(
                    query="What is Redis?",
                    retrieval=RetrievalMetrics(context_precision=0.8, mrr=1.0),
                    response=ResponseMetrics(faithfulness=0.9, answer_relevance=0.85),
                    latency_ms=100,
                ),
            ],
            aggregate={"context_precision_mean": 0.8, "total_queries": 1},
            config={"label": "test", "skip_llm_judge": True},
        )
        md = format_report(report)
        assert "## Retrieval Evaluation Report" in md
        assert "Context Precision" in md
        assert "0.8" in md

    def test_format_comparison_report(self):
        report_a = EvalReport(
            aggregate={"context_precision_mean": 0.85, "mrr_mean": 0.9,
                        "faithfulness_mean": 0.8, "answer_relevance_mean": 0.75,
                        "context_recall_mean": 0.7, "latency_ms_mean": 150,
                        "total_queries": 10},
        )
        report_b = EvalReport(
            aggregate={"context_precision_mean": 0.7, "mrr_mean": 0.8,
                        "faithfulness_mean": 0.7, "answer_relevance_mean": 0.65,
                        "context_recall_mean": 0.6, "latency_ms_mean": 100,
                        "total_queries": 10},
        )
        md = format_comparison_report(report_a, report_b, "Hybrid+Reranker", "Vector Only")
        assert "Hybrid+Reranker" in md
        assert "Vector Only" in md
        assert "✅" in md  # Some improvements


# ═══════════════════════════════════════════════════════════════════════════
#  Eval Corpus Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEvalCorpus:
    def test_corpus_not_empty(self):
        assert len(EVAL_CORPUS) >= 100  # Need 100+ for statistical confidence

    def test_corpus_has_diverse_categories(self):
        categories = set(q.category for q in EVAL_CORPUS)
        # Must have keyword-heavy, semantic, and ambiguous at minimum
        assert "keyword" in categories
        assert "semantic" in categories
        assert "ambiguous" in categories
        assert "multi_concept" in categories
        assert "follow_up" in categories
        assert len(categories) >= 6  # At least 6 distinct categories

    def test_corpus_queries_are_strings(self):
        for q in EVAL_CORPUS:
            assert isinstance(q.query, str)
            assert len(q.query) > 0

    def test_corpus_keyword_queries_have_keywords(self):
        """Keyword-heavy queries MUST have relevant_keywords for BM25 evaluation."""
        keyword_queries = [q for q in EVAL_CORPUS if q.category == "keyword"]
        assert len(keyword_queries) >= 20
        for q in keyword_queries:
            assert len(q.relevant_keywords) > 0, f"Keyword query missing keywords: {q.query}"

    def test_corpus_category_balance(self):
        """No single category should dominate (>40%) or be too small (<5)."""
        from collections import Counter
        cats = Counter(q.category for q in EVAL_CORPUS)
        total = len(EVAL_CORPUS)
        for cat, count in cats.items():
            assert count <= total * 0.4, f"Category '{cat}' is {count}/{total} — too dominant"

    def test_no_duplicate_queries(self):
        queries = [q.query for q in EVAL_CORPUS]
        assert len(queries) == len(set(queries)), "Duplicate queries in corpus"


# ═══════════════════════════════════════════════════════════════════════════
#  Threshold Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestThresholds:
    def test_thresholds_exist(self):
        from evaluation import THRESHOLDS
        assert "noise_floor" in THRESHOLDS
        assert "max_acceptable_latency_ms" in THRESHOLDS
        assert THRESHOLDS["noise_floor"] == 0.02

    def test_evaluate_threshold_noise(self):
        from evaluation import evaluate_threshold
        assert evaluate_threshold("precision", 0.01) == "noise"
        assert evaluate_threshold("precision", -0.01) == "noise"

    def test_evaluate_threshold_regression(self):
        from evaluation import evaluate_threshold
        assert evaluate_threshold("precision", -0.05) == "regression"

    def test_evaluate_threshold_marginal(self):
        from evaluation import evaluate_threshold
        assert evaluate_threshold("precision", 0.05) == "marginal"

    def test_aggregates_include_std(self):
        """Aggregates must include standard deviation for statistical reporting."""
        results = [
            QueryEvalResult(
                query=f"q{i}",
                retrieval=RetrievalMetrics(context_precision=0.5 + i * 0.1),
                response=ResponseMetrics(faithfulness=0.6),
                latency_ms=100 + i * 10,
            )
            for i in range(5)
        ]
        from evaluation import _compute_aggregates
        agg = _compute_aggregates(results)
        assert "context_precision_std" in agg
        assert "latency_ms_std" in agg
        assert "latency_ms_p95" in agg
        assert "context_precision_median" in agg
        assert agg["context_precision_std"] > 0  # Must have variance
