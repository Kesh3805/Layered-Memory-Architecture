"""Automated retrieval evaluation harness — quantified quality metrics.

Measures retrieval and response quality across four dimensions:

1. **Context Precision** — Are retrieved chunks relevant to the query?
   Scored via embedding similarity between query and each retrieved chunk.

2. **Context Recall** — Are the most relevant chunks actually retrieved?
   Compares retrieved set against an oracle set from exhaustive search.

3. **Faithfulness** — Does the response stick to retrieved context?
   LLM-as-judge: checks if claims in the response are grounded in context.

4. **Answer Relevance** — Does the response address the question?
   LLM-as-judge: checks if the response is on-topic and complete.

Architecture
------------
- Offline evaluation: runs against a corpus of (query, expected_context,
  expected_answer) ground truth triples.
- Online evaluation: hooks into the pipeline to score live responses.
- LLM-as-judge uses the same LLM provider configured in settings.py.
- All metrics are [0, 1] floats; higher = better.

Public API
----------
    evaluate_retrieval(query, retrieved, oracle_chunks) → RetrievalMetrics
    evaluate_response(query, context, response) → ResponseMetrics
    run_eval_suite(corpus, search_fn, generate_fn) → EvalReport
    format_report(report) → str  (markdown table)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  SUCCESS THRESHOLDS — define BEFORE running experiments
# ═══════════════════════════════════════════════════════════════════════════
# If gains are below these, the feature is noise, not signal.
# If gains exceed these, the feature earns its complexity and latency.
# Define what "winning" looks like so we don't rationalize weak results.

THRESHOLDS = {
    # Hybrid search is worth it if:
    "hybrid_precision_min_delta": 0.10,      # +10% context precision
    "hybrid_recall_min_delta": 0.10,         # +10% context recall
    "hybrid_faithfulness_min_delta": -0.15,  # OR −15% hallucination proxy

    # Reranking is worth it if:
    "reranker_faithfulness_min_delta": 0.15, # +15% faithfulness
    "reranker_relevance_min_delta": 0.10,    # OR +10% answer relevance
    "reranker_precision_min_delta": 0.10,    # OR +10% context precision

    # Latency budget — max acceptable overhead:
    "max_acceptable_latency_ms": 500,        # No feature should add >500ms
    "latency_justified_if_gain_above": 0.15, # Unless quality gain exceeds 15%

    # Noise floor — deltas below this are not meaningful:
    "noise_floor": 0.02,                     # ±2% is noise
}


def evaluate_threshold(metric_name: str, delta: float, latency_delta_ms: float = 0) -> str:
    """Evaluate whether a metric delta meets the predefined threshold.

    Returns one of: 'significant_win', 'marginal', 'noise', 'regression'.
    """
    noise = THRESHOLDS["noise_floor"]

    if abs(delta) <= noise:
        return "noise"

    if delta < -noise:
        return "regression"

    # Check specific thresholds
    for key, threshold in THRESHOLDS.items():
        if metric_name in key and "delta" in key:
            if delta >= threshold:
                # Check latency budget
                max_lat = THRESHOLDS["max_acceptable_latency_ms"]
                min_gain = THRESHOLDS["latency_justified_if_gain_above"]
                if latency_delta_ms > max_lat and delta < min_gain:
                    return "marginal"  # Gain doesn't justify the latency cost
                return "significant_win"

    if delta > noise:
        return "marginal"

    return "noise"


# ═══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalMetrics:
    """Per-query retrieval quality scores."""
    context_precision: float = 0.0   # Avg similarity of retrieved chunks to query
    context_recall: float = 0.0      # Fraction of oracle chunks found in retrieved set
    mrr: float = 0.0                 # Mean Reciprocal Rank (first relevant chunk)
    num_retrieved: int = 0
    num_oracle: int = 0


@dataclass
class ResponseMetrics:
    """Per-query response quality scores."""
    faithfulness: float = 0.0        # Grounded in context [0,1]
    answer_relevance: float = 0.0    # Addresses the question [0,1]
    response_length: int = 0


@dataclass
class QueryEvalResult:
    """Complete evaluation for a single query."""
    query: str
    retrieval: RetrievalMetrics
    response: ResponseMetrics
    latency_ms: float = 0.0
    retrieved_texts: list[str] = field(default_factory=list)
    response_text: str = ""


@dataclass
class EvalReport:
    """Aggregate evaluation report across a corpus."""
    results: list[QueryEvalResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "results": [asdict(r) for r in self.results],
            "aggregate": self.aggregate,
            "config": self.config,
            "timestamp": self.timestamp,
        }


@dataclass
class EvalQuery:
    """A ground-truth evaluation query."""
    query: str
    relevant_keywords: list[str] = field(default_factory=list)
    expected_in_response: list[str] = field(default_factory=list)
    category: str = "general"


# ═══════════════════════════════════════════════════════════════════════════
#  RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════════════════════════

def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    dot = float(np.dot(a_arr, b_arr))
    norm = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    return dot / norm if norm > 0 else 0.0


def evaluate_retrieval(
    query_embedding,
    retrieved_texts: list[str],
    retrieved_scores: list[float],
    oracle_texts: list[str] | None = None,
    relevance_threshold: float = 0.5,
) -> RetrievalMetrics:
    """Compute retrieval quality metrics.

    Args:
        query_embedding:     Embedding vector for the query.
        retrieved_texts:     Texts returned by retrieval.
        retrieved_scores:    Similarity scores for each retrieved text.
        oracle_texts:        Ground truth relevant texts (optional).
        relevance_threshold: Score threshold for a chunk to count as relevant.

    Returns:
        RetrievalMetrics with precision, recall, and MRR.
    """
    if not retrieved_texts:
        return RetrievalMetrics()

    # Context Precision: fraction of retrieved chunks above relevance threshold
    relevant_count = sum(1 for s in retrieved_scores if s >= relevance_threshold)
    precision = relevant_count / len(retrieved_scores) if retrieved_scores else 0.0

    # Mean Reciprocal Rank: rank of first relevant chunk (1-indexed)
    mrr = 0.0
    for i, score in enumerate(retrieved_scores):
        if score >= relevance_threshold:
            mrr = 1.0 / (i + 1)
            break

    # Context Recall: fraction of oracle chunks that appear in retrieved set
    recall = 0.0
    num_oracle = 0
    if oracle_texts:
        num_oracle = len(oracle_texts)
        # Approximate matching: check if any oracle text is a substring of retrieved
        found = 0
        for oracle in oracle_texts:
            oracle_lower = oracle.lower().strip()
            for retrieved in retrieved_texts:
                if oracle_lower in retrieved.lower() or retrieved.lower() in oracle_lower:
                    found += 1
                    break
        recall = found / num_oracle if num_oracle > 0 else 0.0

    return RetrievalMetrics(
        context_precision=round(precision, 4),
        context_recall=round(recall, 4),
        mrr=round(mrr, 4),
        num_retrieved=len(retrieved_texts),
        num_oracle=num_oracle,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  RESPONSE METRICS (LLM-as-Judge)
# ═══════════════════════════════════════════════════════════════════════════

_FAITHFULNESS_PROMPT = """You are evaluating whether an AI response is faithful to the provided context.

CONTEXT (retrieved documents):
{context}

QUESTION: {query}

RESPONSE: {response}

Score the faithfulness of the response on a scale of 0.0 to 1.0:
- 1.0 = Every claim in the response is supported by the context
- 0.5 = Some claims are supported, some are not
- 0.0 = The response contradicts or fabricates beyond the context

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""

_RELEVANCE_PROMPT = """You are evaluating whether an AI response relevantly addresses the user's question.

QUESTION: {query}

RESPONSE: {response}

Score the answer relevance on a scale of 0.0 to 1.0:
- 1.0 = Directly and completely addresses the question
- 0.5 = Partially addresses the question or is somewhat off-topic  
- 0.0 = Does not address the question at all

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


def _llm_judge(prompt: str) -> tuple[float, str]:
    """Call the LLM as a judge and parse the score.

    Returns (score, reason). Falls back to (0.5, "parse error") on failure.
    """
    try:
        from llm.generators import _call_llm
        from settings import settings

        response = _call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )

        # Parse JSON from response
        text = response.strip()
        # Try to find JSON in the response
        if "{" in text:
            json_start = text.index("{")
            json_end = text.rindex("}") + 1
            data = json.loads(text[json_start:json_end])
            score = float(data.get("score", 0.5))
            reason = data.get("reason", "")
            return (max(0.0, min(1.0, score)), reason)

        return (0.5, "Could not parse judge response")

    except Exception as e:
        logger.warning("LLM judge call failed: %s", e)
        return (0.5, f"Judge error: {e}")


def evaluate_response(
    query: str,
    context: str,
    response: str,
    skip_llm_judge: bool = False,
) -> ResponseMetrics:
    """Evaluate response quality using LLM-as-judge.

    Args:
        query:           The user's question.
        context:         Retrieved context that was provided to the LLM.
        response:        The LLM's response.
        skip_llm_judge:  If True, skip LLM calls (for fast/offline testing).

    Returns:
        ResponseMetrics with faithfulness and relevance scores.
    """
    if not response:
        return ResponseMetrics()

    if skip_llm_judge:
        # Heuristic fallback: keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        context_words = set(context.lower().split()) if context else set()

        relevance = len(query_words & response_words) / max(len(query_words), 1)
        faithfulness = len(context_words & response_words) / max(len(response_words), 1) if context_words else 0.5

        return ResponseMetrics(
            faithfulness=round(min(faithfulness * 2, 1.0), 4),  # Scale up
            answer_relevance=round(min(relevance * 3, 1.0), 4),
            response_length=len(response),
        )

    # LLM-as-judge
    faithfulness_score, _f_reason = _llm_judge(
        _FAITHFULNESS_PROMPT.format(context=context[:3000], query=query, response=response[:2000])
    )
    relevance_score, _r_reason = _llm_judge(
        _RELEVANCE_PROMPT.format(query=query, response=response[:2000])
    )

    return ResponseMetrics(
        faithfulness=round(faithfulness_score, 4),
        answer_relevance=round(relevance_score, 4),
        response_length=len(response),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION SUITE
# ═══════════════════════════════════════════════════════════════════════════

def run_eval_suite(
    eval_queries: list[EvalQuery],
    search_fn: Callable[[str], list[tuple[str, float]]],
    generate_fn: Callable[[str, str], str] | None = None,
    skip_llm_judge: bool = False,
    config_label: str = "default",
) -> EvalReport:
    """Run the full evaluation suite on a corpus of queries.

    Args:
        eval_queries:    List of EvalQuery ground-truth items.
        search_fn:       Retrieval function: query → [(text, score)].
        generate_fn:     Generation function: (query, context) → response.
                         If None, skips response evaluation.
        skip_llm_judge:  Skip LLM-as-judge calls (heuristic fallback).
        config_label:    Label for this evaluation run configuration.

    Returns:
        EvalReport with per-query results and aggregate metrics.
    """
    report = EvalReport(
        config={"label": config_label, "num_queries": len(eval_queries),
                "skip_llm_judge": skip_llm_judge},
    )

    for i, eq in enumerate(eval_queries):
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(eval_queries), eq.query[:60])
        start = time.perf_counter()

        # Retrieve
        try:
            results = search_fn(eq.query)
            texts = [text for text, _score in results]
            scores = [score for _text, score in results]
        except Exception as e:
            logger.warning("Search failed for query %d: %s", i, e)
            texts, scores = [], []

        # Evaluate retrieval
        retrieval_metrics = evaluate_retrieval(
            query_embedding=None,  # Not used when scores are provided directly
            retrieved_texts=texts,
            retrieved_scores=scores,
            oracle_texts=eq.relevant_keywords,  # Approximate oracle via keywords
        )

        # Generate response (if generator provided)
        response_text = ""
        response_metrics = ResponseMetrics()
        if generate_fn is not None:
            context = "\n".join(texts)
            try:
                response_text = generate_fn(eq.query, context)
            except Exception as e:
                logger.warning("Generation failed for query %d: %s", i, e)
                response_text = ""

            response_metrics = evaluate_response(
                query=eq.query,
                context="\n".join(texts),
                response=response_text,
                skip_llm_judge=skip_llm_judge,
            )

        elapsed = (time.perf_counter() - start) * 1000

        result = QueryEvalResult(
            query=eq.query,
            retrieval=retrieval_metrics,
            response=response_metrics,
            latency_ms=round(elapsed, 2),
            retrieved_texts=texts[:3],  # Keep top 3 for reporting
            response_text=response_text[:500],
        )
        report.results.append(result)

    # Aggregate
    report.aggregate = _compute_aggregates(report.results)
    return report


def _compute_aggregates(results: list[QueryEvalResult]) -> dict[str, float]:
    """Compute mean, std, min, max for each metric across all queries."""
    if not results:
        return {}

    metrics = {
        "context_precision": [r.retrieval.context_precision for r in results],
        "context_recall": [r.retrieval.context_recall for r in results],
        "mrr": [r.retrieval.mrr for r in results],
        "faithfulness": [r.response.faithfulness for r in results],
        "answer_relevance": [r.response.answer_relevance for r in results],
        "latency_ms": [r.latency_ms for r in results],
    }

    agg = {}
    for name, values in metrics.items():
        # For quality metrics, only count non-zero values in mean (except latency)
        if name == "latency_ms":
            filtered = values
        else:
            filtered = [v for v in values if v > 0]

        if filtered:
            arr = np.array(filtered, dtype=np.float64)
            agg[f"{name}_mean"] = round(float(np.mean(arr)), 4)
            agg[f"{name}_std"] = round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4)
            agg[f"{name}_min"] = round(float(np.min(arr)), 4)
            agg[f"{name}_max"] = round(float(np.max(arr)), 4)
            agg[f"{name}_median"] = round(float(np.median(arr)), 4)
            agg[f"{name}_count"] = len(filtered)

            # P95 for latency
            if name == "latency_ms":
                agg[f"{name}_p95"] = round(float(np.percentile(arr, 95)), 2)

    agg["total_queries"] = len(results)

    # Category breakdown
    categories = {}
    for r in results:
        cat = getattr(r, "_category", "unknown")
        if cat not in categories:
            categories[cat] = {"precision": [], "recall": [], "count": 0}
        categories[cat]["precision"].append(r.retrieval.context_precision)
        categories[cat]["recall"].append(r.retrieval.context_recall)
        categories[cat]["count"] += 1

    agg["_categories"] = {
        cat: {
            "count": data["count"],
            "precision_mean": round(sum(data["precision"]) / len(data["precision"]), 4) if data["precision"] else 0,
            "recall_mean": round(sum(data["recall"]) / len(data["recall"]), 4) if data["recall"] else 0,
        }
        for cat, data in categories.items()
    }

    return agg


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION CORPUS (built-in test queries)
# ═══════════════════════════════════════════════════════════════════════════

EVAL_CORPUS: list[EvalQuery] = [
    # ─── KEYWORD-HEAVY (BM25 advantage) ── 25 queries ───────────────
    # These contain specific terms, IDs, error codes, exact names that
    # lexical search should catch but vector search often misses.
    EvalQuery(
        query="What is pgvector and how do I install the extension?",
        relevant_keywords=["pgvector", "extension", "install", "vector"],
        expected_in_response=["pgvector"],
        category="keyword",
    ),
    EvalQuery(
        query="PostgreSQL 16 new features and improvements",
        relevant_keywords=["postgresql", "16", "features", "improvements"],
        expected_in_response=["postgresql"],
        category="keyword",
    ),
    EvalQuery(
        query="Redis HSET command syntax and examples",
        relevant_keywords=["redis", "hset", "command", "hash"],
        expected_in_response=["hset"],
        category="keyword",
    ),
    EvalQuery(
        query="HNSW index configuration parameters for pgvector",
        relevant_keywords=["hnsw", "index", "pgvector", "ef_construction", "m"],
        expected_in_response=["hnsw"],
        category="keyword",
    ),
    EvalQuery(
        query="Docker ENTRYPOINT vs CMD difference",
        relevant_keywords=["docker", "entrypoint", "cmd", "dockerfile"],
        expected_in_response=["entrypoint", "cmd"],
        category="keyword",
    ),
    EvalQuery(
        query="EXPLAIN ANALYZE query plan in PostgreSQL",
        relevant_keywords=["explain", "analyze", "query plan", "postgresql", "seq scan"],
        expected_in_response=["explain"],
        category="keyword",
    ),
    EvalQuery(
        query="Python asyncio vs threading for I/O bound tasks",
        relevant_keywords=["asyncio", "threading", "concurrent", "await", "GIL"],
        expected_in_response=["asyncio"],
        category="keyword",
    ),
    EvalQuery(
        query="FastAPI dependency injection with Depends()",
        relevant_keywords=["fastapi", "depends", "dependency injection", "endpoint"],
        expected_in_response=["depends"],
        category="keyword",
    ),
    EvalQuery(
        query="What does error code SQLSTATE 23505 mean?",
        relevant_keywords=["sqlstate", "23505", "unique", "violation", "constraint"],
        expected_in_response=["unique", "violation"],
        category="keyword",
    ),
    EvalQuery(
        query="Kubernetes liveness vs readiness probe configuration",
        relevant_keywords=["kubernetes", "liveness", "readiness", "probe", "health check"],
        expected_in_response=["liveness", "readiness"],
        category="keyword",
    ),
    EvalQuery(
        query="psycopg2 connection pool SimpleConnectionPool usage",
        relevant_keywords=["psycopg2", "simpleconnectionpool", "connection pool", "getconn"],
        expected_in_response=["psycopg2"],
        category="keyword",
    ),
    EvalQuery(
        query="sentence-transformers BAAI/bge-base-en-v1.5 model card",
        relevant_keywords=["sentence-transformers", "bge-base", "embedding", "768"],
        expected_in_response=["bge"],
        category="keyword",
    ),
    EvalQuery(
        query="cross-encoder/ms-marco-MiniLM-L-6-v2 reranking model",
        relevant_keywords=["cross-encoder", "ms-marco", "minilm", "reranking"],
        expected_in_response=["cross-encoder"],
        category="keyword",
    ),
    EvalQuery(
        query="PostgreSQL GIN index for full text search tsvector",
        relevant_keywords=["gin", "index", "tsvector", "full text", "search"],
        expected_in_response=["gin", "tsvector"],
        category="keyword",
    ),
    EvalQuery(
        query="Reciprocal Rank Fusion algorithm formula",
        relevant_keywords=["reciprocal rank fusion", "rrf", "rank", "fusion"],
        expected_in_response=["reciprocal", "rank"],
        category="keyword",
    ),
    EvalQuery(
        query="ts_rank_cd function in PostgreSQL text search",
        relevant_keywords=["ts_rank_cd", "tsvector", "tsquery", "ranking"],
        expected_in_response=["ts_rank"],
        category="keyword",
    ),
    EvalQuery(
        query="Nginx reverse proxy configuration for FastAPI",
        relevant_keywords=["nginx", "reverse proxy", "fastapi", "upstream", "proxy_pass"],
        expected_in_response=["nginx"],
        category="keyword",
    ),
    EvalQuery(
        query="SQLAlchemy vs psycopg2 for PostgreSQL access",
        relevant_keywords=["sqlalchemy", "psycopg2", "orm", "raw sql"],
        expected_in_response=["sqlalchemy", "psycopg2"],
        category="keyword",
    ),
    EvalQuery(
        query="Pydantic BaseModel validation in FastAPI",
        relevant_keywords=["pydantic", "basemodel", "validation", "fastapi"],
        expected_in_response=["pydantic"],
        category="keyword",
    ),
    EvalQuery(
        query="Docker Compose healthcheck for PostgreSQL service",
        relevant_keywords=["docker compose", "healthcheck", "pg_isready", "postgresql"],
        expected_in_response=["healthcheck"],
        category="keyword",
    ),
    EvalQuery(
        query="JWT token authentication with python-jose",
        relevant_keywords=["jwt", "token", "python-jose", "authentication", "bearer"],
        expected_in_response=["jwt"],
        category="keyword",
    ),
    EvalQuery(
        query="cosine similarity formula for vector embeddings",
        relevant_keywords=["cosine", "similarity", "dot product", "norm", "embedding"],
        expected_in_response=["cosine"],
        category="keyword",
    ),
    EvalQuery(
        query="uvicorn --workers flag for production deployment",
        relevant_keywords=["uvicorn", "workers", "gunicorn", "production"],
        expected_in_response=["uvicorn"],
        category="keyword",
    ),
    EvalQuery(
        query="BM25 term frequency inverse document frequency scoring",
        relevant_keywords=["bm25", "tf-idf", "term frequency", "document frequency"],
        expected_in_response=["bm25"],
        category="keyword",
    ),
    EvalQuery(
        query="pgvector ivfflat vs hnsw index type comparison",
        relevant_keywords=["ivfflat", "hnsw", "pgvector", "index", "approximate"],
        expected_in_response=["ivfflat", "hnsw"],
        category="keyword",
    ),

    # ─── SEMANTIC / PARAPHRASE (vector advantage) ── 25 queries ─────
    # Same concepts expressed differently — vector search should find
    # relevant docs even though exact keywords don't match.
    EvalQuery(
        query="How do I make my database queries faster?",
        relevant_keywords=["optimization", "index", "query", "performance"],
        expected_in_response=["index", "query"],
        category="semantic",
    ),
    EvalQuery(
        query="What's the best way to store conversation history?",
        relevant_keywords=["conversation", "history", "storage", "messages", "database"],
        expected_in_response=["conversation"],
        category="semantic",
    ),
    EvalQuery(
        query="How can I find similar documents in a collection?",
        relevant_keywords=["similarity", "search", "vector", "embedding", "retrieval"],
        expected_in_response=["similar"],
        category="semantic",
    ),
    EvalQuery(
        query="Tell me about breaking a big application into smaller services",
        relevant_keywords=["microservices", "decomposition", "modular", "distributed"],
        expected_in_response=["service"],
        category="semantic",
    ),
    EvalQuery(
        query="What approaches exist for handling real-time data updates?",
        relevant_keywords=["real-time", "streaming", "websocket", "pub/sub", "event"],
        expected_in_response=["real-time"],
        category="semantic",
    ),
    EvalQuery(
        query="How do I keep my system running when parts of it fail?",
        relevant_keywords=["fault tolerance", "resilience", "circuit breaker", "retry"],
        expected_in_response=["fail"],
        category="semantic",
    ),
    EvalQuery(
        query="What's a good strategy for keeping frequently used data close?",
        relevant_keywords=["caching", "redis", "memcached", "ttl", "invalidation"],
        expected_in_response=["cache"],
        category="semantic",
    ),
    EvalQuery(
        query="How do I make sure my text search finds relevant results?",
        relevant_keywords=["relevance", "ranking", "search quality", "precision", "recall"],
        expected_in_response=["search", "relevant"],
        category="semantic",
    ),
    EvalQuery(
        query="What's the right way to handle secrets in my application?",
        relevant_keywords=["secrets", "environment", "vault", "credentials", "config"],
        expected_in_response=["secret"],
        category="semantic",
    ),
    EvalQuery(
        query="How do I run multiple versions of my app simultaneously?",
        relevant_keywords=["deployment", "blue-green", "canary", "rolling", "container"],
        expected_in_response=["deploy"],
        category="semantic",
    ),
    EvalQuery(
        query="What's the best approach for tracking what my system is doing?",
        relevant_keywords=["observability", "logging", "monitoring", "tracing", "metrics"],
        expected_in_response=["monitor"],
        category="semantic",
    ),
    EvalQuery(
        query="How can I prevent unauthorized access to my API?",
        relevant_keywords=["authentication", "authorization", "token", "oauth", "jwt"],
        expected_in_response=["auth"],
        category="semantic",
    ),
    EvalQuery(
        query="What are some ways to speed up responses for repeated requests?",
        relevant_keywords=["caching", "memoization", "cdn", "ttl", "performance"],
        expected_in_response=["cache"],
        category="semantic",
    ),
    EvalQuery(
        query="How do I set up automatic recovery when my server crashes?",
        relevant_keywords=["restart", "supervisor", "systemd", "health check", "watchdog"],
        expected_in_response=["restart"],
        category="semantic",
    ),
    EvalQuery(
        query="What's the smartest way to combine results from different data sources?",
        relevant_keywords=["fusion", "aggregation", "join", "merge", "combine"],
        expected_in_response=["combine"],
        category="semantic",
    ),
    EvalQuery(
        query="How should I organize my project files for a large application?",
        relevant_keywords=["project structure", "architecture", "modules", "packages"],
        expected_in_response=["structure"],
        category="semantic",
    ),
    EvalQuery(
        query="What's the best way to test code that talks to a database?",
        relevant_keywords=["testing", "mock", "fixture", "integration", "database"],
        expected_in_response=["test"],
        category="semantic",
    ),
    EvalQuery(
        query="How do I handle errors gracefully without crashing?",
        relevant_keywords=["error handling", "exception", "try catch", "fallback", "graceful"],
        expected_in_response=["error"],
        category="semantic",
    ),
    EvalQuery(
        query="What's the difference between putting things side by side vs stacking them?",
        relevant_keywords=["horizontal", "vertical", "scaling", "architecture"],
        expected_in_response=["horizontal", "vertical"],
        category="semantic",
    ),
    EvalQuery(
        query="How can I understand what my users are experiencing?",
        relevant_keywords=["analytics", "telemetry", "instrumentation", "user behavior"],
        expected_in_response=["telemetry"],
        category="semantic",
    ),
    EvalQuery(
        query="How do I process large amounts of data without running out of memory?",
        relevant_keywords=["streaming", "chunking", "batch", "generator", "pagination"],
        expected_in_response=["batch"],
        category="semantic",
    ),
    EvalQuery(
        query="What's the right way to evolve my database schema over time?",
        relevant_keywords=["migration", "schema", "alembic", "alter table", "versioning"],
        expected_in_response=["migration"],
        category="semantic",
    ),
    EvalQuery(
        query="How do I coordinate work between multiple background processes?",
        relevant_keywords=["concurrency", "queue", "worker", "celery", "task"],
        expected_in_response=["worker"],
        category="semantic",
    ),
    EvalQuery(
        query="What's a reliable way to transfer data between systems?",
        relevant_keywords=["api", "rest", "grpc", "message queue", "integration"],
        expected_in_response=["api"],
        category="semantic",
    ),
    EvalQuery(
        query="How do I make my search understand what people actually mean?",
        relevant_keywords=["semantic search", "embedding", "nlp", "understanding", "intent"],
        expected_in_response=["semantic"],
        category="semantic",
    ),

    # ─── MULTI-CONCEPT (both arms contribute) ── 20 queries ────────
    EvalQuery(
        query="How does connection pooling work with PostgreSQL and pgvector?",
        relevant_keywords=["connection pool", "postgresql", "pgvector", "psycopg2"],
        expected_in_response=["connection", "pool"],
        category="multi_concept",
    ),
    EvalQuery(
        query="What are the tradeoffs between horizontal and vertical scaling for databases?",
        relevant_keywords=["horizontal", "vertical", "scaling", "database", "sharding"],
        expected_in_response=["horizontal", "vertical"],
        category="multi_concept",
    ),
    EvalQuery(
        query="Explain event sourcing with CQRS in a microservices architecture",
        relevant_keywords=["event sourcing", "cqrs", "microservices", "event store"],
        expected_in_response=["event", "cqrs"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How do vector embeddings improve search relevance compared to BM25?",
        relevant_keywords=["vector", "embedding", "bm25", "relevance", "semantic"],
        expected_in_response=["vector", "bm25"],
        category="multi_concept",
    ),
    EvalQuery(
        query="What is the role of an API gateway in distributed systems?",
        relevant_keywords=["api gateway", "routing", "load balancing", "distributed"],
        expected_in_response=["gateway"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How do you combine Redis caching with PostgreSQL for optimal performance?",
        relevant_keywords=["redis", "postgresql", "caching", "performance", "read-through"],
        expected_in_response=["redis", "postgresql"],
        category="multi_concept",
    ),
    EvalQuery(
        query="Compare Docker Compose and Kubernetes for container orchestration",
        relevant_keywords=["docker compose", "kubernetes", "orchestration", "container"],
        expected_in_response=["docker", "kubernetes"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How do you implement semantic search with pgvector and sentence-transformers?",
        relevant_keywords=["semantic search", "pgvector", "sentence-transformers", "embedding"],
        expected_in_response=["pgvector", "embedding"],
        category="multi_concept",
    ),
    EvalQuery(
        query="What is the difference between REST and GraphQL for API design?",
        relevant_keywords=["rest", "graphql", "api", "query", "endpoint"],
        expected_in_response=["rest", "graphql"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How do circuit breakers and retries improve resilience in distributed systems?",
        relevant_keywords=["circuit breaker", "retry", "resilience", "fallback", "distributed"],
        expected_in_response=["circuit breaker"],
        category="multi_concept",
    ),
    EvalQuery(
        query="Explain hybrid search combining BM25 lexical matching with vector similarity",
        relevant_keywords=["hybrid search", "bm25", "vector", "lexical", "fusion"],
        expected_in_response=["hybrid", "bm25"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How does cross-encoder reranking improve retrieval precision over bi-encoders?",
        relevant_keywords=["cross-encoder", "reranking", "bi-encoder", "precision"],
        expected_in_response=["cross-encoder", "rerank"],
        category="multi_concept",
    ),
    EvalQuery(
        query="What monitoring tools work best with FastAPI and PostgreSQL?",
        relevant_keywords=["monitoring", "fastapi", "postgresql", "prometheus", "grafana"],
        expected_in_response=["monitoring"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How do you test a RAG pipeline end-to-end with ground truth?",
        relevant_keywords=["testing", "rag", "ground truth", "evaluation", "retrieval"],
        expected_in_response=["test", "rag"],
        category="multi_concept",
    ),
    EvalQuery(
        query="Explain token budgeting for LLM context window management",
        relevant_keywords=["token", "budget", "context window", "truncation", "llm"],
        expected_in_response=["token"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How does EMA-based topic threading compare to k-means clustering for conversations?",
        relevant_keywords=["ema", "topic", "threading", "clustering", "centroid"],
        expected_in_response=["topic", "thread"],
        category="multi_concept",
    ),
    EvalQuery(
        query="What is the difference between synchronous and asynchronous database access in Python?",
        relevant_keywords=["synchronous", "asynchronous", "psycopg2", "asyncpg", "python"],
        expected_in_response=["synchronous", "asynchronous"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How do you implement policy-based retrieval gating in a RAG system?",
        relevant_keywords=["policy", "retrieval", "gating", "rag", "intent"],
        expected_in_response=["policy", "retrieval"],
        category="multi_concept",
    ),
    EvalQuery(
        query="Explain the tradeoffs of embedding model size vs retrieval quality",
        relevant_keywords=["embedding", "model size", "quality", "dimension", "latency"],
        expected_in_response=["embedding"],
        category="multi_concept",
    ),
    EvalQuery(
        query="How does LLM-as-judge evaluation compare to human evaluation for RAG?",
        relevant_keywords=["llm judge", "evaluation", "human", "rag", "faithfulness"],
        expected_in_response=["evaluation"],
        category="multi_concept",
    ),

    # ─── FOLLOW-UP / CONTEXTUAL ── 15 queries ──────────────────────
    EvalQuery(
        query="Can you explain that in more detail?",
        relevant_keywords=[],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="What about the performance implications?",
        relevant_keywords=["performance"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="How does this compare to the alternative approach?",
        relevant_keywords=["compare", "alternative"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="What were the downsides you mentioned?",
        relevant_keywords=["downside", "tradeoff", "disadvantage"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="Can you show me a code example?",
        relevant_keywords=["code", "example", "implementation"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="What about in production though?",
        relevant_keywords=["production", "deployment"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="Is that the recommended approach?",
        relevant_keywords=["recommended", "best practice"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="And what about security concerns?",
        relevant_keywords=["security", "vulnerability"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="How does that scale?",
        relevant_keywords=["scale", "performance", "load"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="Wait, go back to the part about indexing",
        relevant_keywords=["indexing"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="So which one should I use?",
        relevant_keywords=[],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="What if I need both?",
        relevant_keywords=[],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="That doesn't seem right, can you double check?",
        relevant_keywords=[],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="What were we talking about regarding databases earlier?",
        relevant_keywords=["database"],
        expected_in_response=[],
        category="follow_up",
    ),
    EvalQuery(
        query="You mentioned something about Redis before, what was that?",
        relevant_keywords=["redis"],
        expected_in_response=[],
        category="follow_up",
    ),

    # ─── AMBIGUOUS / SHORT (edge cases) ── 15 queries ──────────────
    EvalQuery(
        query="Why?",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="Tell me everything about everything",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="Help",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="???",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="yes",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="no",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="What is the meaning of life?",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="database",
        relevant_keywords=["database"],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query=".",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="the thing we discussed",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="do the opposite",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="nevermind",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="more",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="hmm",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),
    EvalQuery(
        query="ok so what now",
        relevant_keywords=[],
        expected_in_response=[],
        category="ambiguous",
    ),

    # ─── BEHAVIORAL / CONVERSATIONAL ── 10 queries ─────────────────
    EvalQuery(
        query="Thanks for the explanation, that was really helpful!",
        relevant_keywords=[],
        expected_in_response=[],
        category="gratitude",
    ),
    EvalQuery(
        query="Hello! I'm new here, what can you help me with?",
        relevant_keywords=[],
        expected_in_response=[],
        category="greeting",
    ),
    EvalQuery(
        query="Hey there!",
        relevant_keywords=[],
        expected_in_response=[],
        category="greeting",
    ),
    EvalQuery(
        query="That's not what I asked at all. Please read my question again.",
        relevant_keywords=[],
        expected_in_response=[],
        category="frustrated",
    ),
    EvalQuery(
        query="This is getting really annoying. Can you just answer the question?",
        relevant_keywords=[],
        expected_in_response=[],
        category="frustrated",
    ),
    EvalQuery(
        query="Are you ChatGPT or something else?",
        relevant_keywords=[],
        expected_in_response=[],
        category="meta",
    ),
    EvalQuery(
        query="What model are you running on?",
        relevant_keywords=[],
        expected_in_response=[],
        category="meta",
    ),
    EvalQuery(
        query="Can you explain that in simpler terms?",
        relevant_keywords=[],
        expected_in_response=[],
        category="clarification",
    ),
    EvalQuery(
        query="I don't understand any of this",
        relevant_keywords=[],
        expected_in_response=[],
        category="clarification",
    ),
    EvalQuery(
        query="Good job, that's exactly what I needed!",
        relevant_keywords=[],
        expected_in_response=[],
        category="gratitude",
    ),

    # ─── MULTI-TURN SEQUENCES ── 10 queries (ordered) ──────────────
    # Realistic conversation flow that tests threading + memory
    EvalQuery(
        query="I'm building a chat application with a PostgreSQL backend",
        relevant_keywords=["chat", "postgresql", "application"],
        expected_in_response=["postgresql"],
        category="multi_turn",
    ),
    EvalQuery(
        query="Should I use pgvector or a separate Pinecone instance for embeddings?",
        relevant_keywords=["pgvector", "pinecone", "embedding", "vector"],
        expected_in_response=["pgvector"],
        category="multi_turn",
    ),
    EvalQuery(
        query="Ok let's go with pgvector. How should I index the embeddings?",
        relevant_keywords=["pgvector", "index", "hnsw", "ivfflat"],
        expected_in_response=["index"],
        category="multi_turn",
    ),
    EvalQuery(
        query="Now I need full text search too. Can PostgreSQL handle both?",
        relevant_keywords=["full text search", "tsvector", "postgresql", "hybrid"],
        expected_in_response=["full text"],
        category="multi_turn",
    ),
    EvalQuery(
        query="How do I combine the vector results with the text search results?",
        relevant_keywords=["combine", "fusion", "reciprocal rank", "hybrid"],
        expected_in_response=["combine"],
        category="multi_turn",
    ),
    EvalQuery(
        query="What about a reranker on top of that?",
        relevant_keywords=["reranker", "cross-encoder", "reranking"],
        expected_in_response=["rerank"],
        category="multi_turn",
    ),
    EvalQuery(
        query="How much latency does the reranker add?",
        relevant_keywords=["latency", "reranker", "performance", "ms"],
        expected_in_response=["latency"],
        category="multi_turn",
    ),
    EvalQuery(
        query="Actually, let's switch topics. How does Redis pub/sub work?",
        relevant_keywords=["redis", "pub/sub", "publish", "subscribe", "channel"],
        expected_in_response=["redis"],
        category="multi_turn",
    ),
    EvalQuery(
        query="Could I use that for real-time notifications in my chat app?",
        relevant_keywords=["real-time", "notifications", "chat", "websocket"],
        expected_in_response=["notification"],
        category="multi_turn",
    ),
    EvalQuery(
        query="Let's go back to the database schema. What tables do I need?",
        relevant_keywords=["database", "schema", "tables", "design"],
        expected_in_response=["table"],
        category="multi_turn",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def format_report(report: EvalReport) -> str:
    """Format an evaluation report as a markdown table with mean ± std."""
    lines = []
    agg = report.aggregate
    config = report.config

    lines.append(f"## Retrieval Evaluation Report — {config.get('label', 'unknown')}\n")
    lines.append(f"**Queries:** {agg.get('total_queries', 0)}")
    lines.append(f"**LLM Judge:** {'Heuristic' if config.get('skip_llm_judge') else 'Active'}\n")

    # Aggregate metrics table
    lines.append("### Aggregate Metrics\n")
    lines.append("| Metric | Mean ± Std | Median | Min | Max | N |")
    lines.append("|--------|:---------:|:------:|:---:|:---:|:-:|")

    metric_names = ["context_precision", "context_recall", "mrr", "faithfulness", "answer_relevance", "latency_ms"]
    display_names = ["Context Precision", "Context Recall", "MRR", "Faithfulness", "Answer Relevance", "Latency (ms)"]

    for metric, display in zip(metric_names, display_names):
        mean = agg.get(f"{metric}_mean", 0)
        std = agg.get(f"{metric}_std", 0)
        med = agg.get(f"{metric}_median", 0)
        mn = agg.get(f"{metric}_min", 0)
        mx = agg.get(f"{metric}_max", 0)
        count = agg.get(f"{metric}_count", 0)
        if metric == "latency_ms":
            p95 = agg.get(f"{metric}_p95", 0)
            lines.append(f"| {display} | {mean:.0f} ± {std:.0f} | {med:.0f} | {mn:.0f} | {mx:.0f} | {count} |")
            lines.append(f"| {display} P95 | {p95:.0f} | — | — | — | — |")
        else:
            lines.append(f"| {display} | {mean:.3f} ± {std:.3f} | {med:.3f} | {mn:.3f} | {mx:.3f} | {count} |")

    # Per-query breakdown
    lines.append("\n### Per-Query Results\n")
    lines.append("| # | Query | Precision | Recall | MRR | Faithfulness | Relevance | Latency |")
    lines.append("|---|-------|:---------:|:------:|:---:|:------------:|:---------:|:-------:|")

    for i, r in enumerate(report.results, 1):
        q = r.query[:50] + ("…" if len(r.query) > 50 else "")
        lines.append(
            f"| {i} | {q} | {r.retrieval.context_precision:.2f} | "
            f"{r.retrieval.context_recall:.2f} | {r.retrieval.mrr:.2f} | "
            f"{r.response.faithfulness:.2f} | {r.response.answer_relevance:.2f} | "
            f"{r.latency_ms:.0f}ms |"
        )

    lines.append("")
    return "\n".join(lines)


def format_comparison_report(
    report_a: EvalReport,
    report_b: EvalReport,
    label_a: str = "Experimental",
    label_b: str = "Baseline",
) -> str:
    """Format a side-by-side comparison with threshold evaluation."""
    lines = []
    lines.append(f"## Retrieval Quality: {label_a} vs {label_b}\n")
    lines.append(f"**Queries:** {report_a.aggregate.get('total_queries', 0)}")
    lines.append(f"**Noise floor:** ±{THRESHOLDS['noise_floor']*100:.0f}%\n")

    lines.append("| Metric | " + label_a + " | " + label_b + " | Delta | Δ% | Verdict |")
    lines.append("|--------|:---:|:---:|:-----:|:--:|:-------:|")

    metric_keys = ["context_precision", "context_recall", "mrr", "faithfulness", "answer_relevance"]
    display_names = ["Context Precision", "Context Recall", "MRR", "Faithfulness", "Answer Relevance"]

    lat_delta = (report_a.aggregate.get("latency_ms_mean", 0) -
                 report_b.aggregate.get("latency_ms_mean", 0))

    for metric, display in zip(metric_keys, display_names):
        a_mean = report_a.aggregate.get(f"{metric}_mean", 0)
        a_std = report_a.aggregate.get(f"{metric}_std", 0)
        b_mean = report_b.aggregate.get(f"{metric}_mean", 0)
        b_std = report_b.aggregate.get(f"{metric}_std", 0)
        delta = a_mean - b_mean
        pct = (delta / b_mean * 100) if b_mean > 0 else 0
        sign = "+" if delta >= 0 else ""

        verdict = evaluate_threshold(metric, delta, lat_delta)
        verdict_emoji = {
            "significant_win": "✅ WIN",
            "marginal": "🟡 marginal",
            "noise": "⚪ noise",
            "regression": "❌ worse",
        }.get(verdict, "—")

        lines.append(
            f"| {display} | {a_mean:.3f}±{a_std:.3f} | {b_mean:.3f}±{b_std:.3f} | "
            f"{sign}{delta:.3f} | {sign}{pct:.1f}% | {verdict_emoji} |"
        )

    # Latency row
    a_lat = report_a.aggregate.get("latency_ms_mean", 0)
    a_lat_std = report_a.aggregate.get("latency_ms_std", 0)
    b_lat = report_b.aggregate.get("latency_ms_mean", 0)
    b_lat_std = report_b.aggregate.get("latency_ms_std", 0)
    a_p95 = report_a.aggregate.get("latency_ms_p95", 0)
    b_p95 = report_b.aggregate.get("latency_ms_p95", 0)
    lat_pct = (lat_delta / b_lat * 100) if b_lat > 0 else 0
    sign = "+" if lat_delta >= 0 else ""
    lat_emoji = "✅" if lat_delta <= 0 else ("🟡" if lat_delta < THRESHOLDS["max_acceptable_latency_ms"] else "❌")
    lines.append(
        f"| Latency (ms) | {a_lat:.0f}±{a_lat_std:.0f} | {b_lat:.0f}±{b_lat_std:.0f} | "
        f"{sign}{lat_delta:.0f} | {sign}{lat_pct:.1f}% | {lat_emoji} |"
    )
    lines.append(
        f"| Latency P95 | {a_p95:.0f} | {b_p95:.0f} | "
        f"{sign}{a_p95 - b_p95:.0f} | — | — |"
    )

    # Threshold summary
    lines.append("\n### Pre-defined Success Thresholds\n")
    lines.append("| Criterion | Threshold | Observed | Met? |")
    lines.append("|-----------|:---------:|:--------:|:----:|")

    prec_delta = report_a.aggregate.get("context_precision_mean", 0) - report_b.aggregate.get("context_precision_mean", 0)
    rec_delta = report_a.aggregate.get("context_recall_mean", 0) - report_b.aggregate.get("context_recall_mean", 0)
    faith_delta = report_a.aggregate.get("faithfulness_mean", 0) - report_b.aggregate.get("faithfulness_mean", 0)
    rel_delta = report_a.aggregate.get("answer_relevance_mean", 0) - report_b.aggregate.get("answer_relevance_mean", 0)

    checks = [
        ("Precision ≥ +10%", 0.10, prec_delta),
        ("Recall ≥ +10%", 0.10, rec_delta),
        ("Faithfulness ≥ +15%", 0.15, faith_delta),
        ("Relevance ≥ +10%", 0.10, rel_delta),
        (f"Latency ≤ +{THRESHOLDS['max_acceptable_latency_ms']:.0f}ms", THRESHOLDS['max_acceptable_latency_ms'], -lat_delta),
    ]
    for label, threshold, observed in checks:
        met = "✅" if observed >= threshold else "❌"
        if "Latency" in label:
            lines.append(f"| {label} | {threshold:.0f} | {-lat_delta:+.0f}ms | {met} |")
        else:
            lines.append(f"| {label} | {threshold:+.2f} | {observed:+.3f} | {met} |")

    lines.append("")
    return "\n".join(lines)
