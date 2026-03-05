"""Retrieval quality A/B experiment — vector baseline vs hybrid vs hybrid+reranker.

Runs controlled experiments across 4 retrieval configurations and produces
a quantified comparison table suitable for README publication.

Usage:
    python experiments/eval_retrieval.py
    python experiments/eval_retrieval.py --url http://localhost:8000 --output experiments/results/retrieval_eval.md

Requires backend running on localhost:8000 with knowledge base indexed.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from experiments.runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    Experiment,
    EXPERIMENT_ARMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  RETRIEVAL EVALUATION QUERIES
# ═══════════════════════════════════════════════════════════════════════════

# Queries designed to test different retrieval strengths:
# - Keyword-heavy (BM25 advantage)
# - Semantic/paraphrase (vector advantage)
# - Hybrid (both arms contribute)
# - Edge cases (ambiguous, short)

RETRIEVAL_EVAL_QUERIES = [
    # ─── KEYWORD-HEAVY (BM25 advantage) ── 30 queries ───────────────
    "What is pgvector and how do I install it?",
    "PostgreSQL 16 new features and improvements",
    "Redis vs Memcached comparison for caching",
    "Docker container networking bridge mode",
    "HNSW index configuration parameters",
    "EXPLAIN ANALYZE query plan in PostgreSQL",
    "FastAPI dependency injection with Depends()",
    "What does error code SQLSTATE 23505 mean?",
    "psycopg2 connection pool SimpleConnectionPool",
    "Kubernetes liveness vs readiness probe",
    "sentence-transformers BAAI/bge-base-en-v1.5 model",
    "PostgreSQL GIN index for tsvector full text search",
    "ts_rank_cd function in PostgreSQL",
    "Reciprocal Rank Fusion algorithm",
    "cross-encoder/ms-marco-MiniLM-L-6-v2 reranking",
    "Nginx reverse proxy configuration for FastAPI",
    "SQLAlchemy vs psycopg2 for database access",
    "Pydantic BaseModel validation errors",
    "Docker Compose healthcheck pg_isready",
    "BM25 term frequency inverse document frequency",
    "pgvector ivfflat vs hnsw index comparison",
    "Python asyncio vs threading comparison",
    "uvicorn --workers flag production",
    "cosine similarity formula for embeddings",
    "JWT token authentication python-jose",
    "Redis HSET command syntax",
    "PostgreSQL ALTER TABLE ADD COLUMN",
    "Docker ENTRYPOINT vs CMD difference",
    "pip install sentence-transformers",
    "CREATE INDEX CONCURRENTLY PostgreSQL",

    # ─── SEMANTIC / PARAPHRASE (vector advantage) ── 30 queries ─────
    "How do I make my database queries faster?",
    "What's the best way to store conversation history?",
    "How can I find similar documents in a collection?",
    "Tell me about breaking a big application into smaller services",
    "What approaches exist for handling real-time data updates?",
    "How do I keep my system running when parts fail?",
    "What's a good strategy for keeping frequently used data close?",
    "How do I make sure my text search finds relevant results?",
    "What's the right way to handle secrets in my application?",
    "How do I run multiple versions of my app simultaneously?",
    "What's the best approach for tracking system health?",
    "How can I prevent unauthorized access to my API?",
    "How do I speed up responses for repeated requests?",
    "How do I set up automatic recovery when my server crashes?",
    "What's the smartest way to combine results from different sources?",
    "How should I organize my project files?",
    "What's a reliable way to test code that uses a database?",
    "How do I handle errors gracefully without crashing?",
    "How can I understand what my users are experiencing?",
    "How do I process large data without running out of memory?",
    "What's the right way to evolve my database schema over time?",
    "How do I coordinate work between background processes?",
    "What's a reliable way to transfer data between systems?",
    "How do I make my search understand what people mean?",
    "What are some ways to split a monolith into services?",
    "How do I detect when my application is performing poorly?",
    "What's the best way to share state between services?",
    "How do I prevent one slow query from blocking everything?",
    "What strategies exist for reducing API response times?",
    "How should I handle database connection failures?",

    # ─── MULTI-CONCEPT (both arms contribute) ── 20 queries ────────
    "How does connection pooling work with PostgreSQL and pgvector?",
    "What are the tradeoffs between horizontal and vertical scaling for databases?",
    "Explain event sourcing with CQRS in a microservices architecture",
    "How do vector embeddings improve search relevance compared to BM25?",
    "What is the role of an API gateway in distributed systems?",
    "How do you combine Redis caching with PostgreSQL?",
    "Compare Docker Compose and Kubernetes for orchestration",
    "How do you implement semantic search with pgvector and sentence-transformers?",
    "Difference between REST and GraphQL for API design",
    "How do circuit breakers improve resilience in distributed systems?",
    "Explain hybrid search combining BM25 with vector similarity",
    "How does cross-encoder reranking improve retrieval precision?",
    "What monitoring tools work best with FastAPI and PostgreSQL?",
    "How do you test a RAG pipeline end-to-end?",
    "Explain token budgeting for LLM context window management",
    "How does EMA-based topic threading compare to k-means clustering?",
    "What is the difference between sync and async database access in Python?",
    "How do you implement policy-based retrieval gating?",
    "Explain tradeoffs of embedding model size vs retrieval quality",
    "How does LLM-as-judge evaluation compare to human evaluation?",

    # ─── FOLLOW-UP / CONTEXTUAL ── 15 queries ──────────────────────
    "Can you explain that in more detail?",
    "What about the performance implications?",
    "How does this compare to the alternative approach?",
    "What were the downsides you mentioned?",
    "Can you show me a code example?",
    "What about in production though?",
    "Is that the recommended approach?",
    "And what about security concerns?",
    "How does that scale?",
    "Wait, go back to the part about indexing",
    "So which one should I use?",
    "What if I need both?",
    "That doesn't seem right, can you double check?",
    "What were we talking about regarding databases earlier?",
    "You mentioned something about Redis before",

    # ─── AMBIGUOUS / SHORT ── 15 queries ───────────────────────────
    "Why?",
    "Everything about databases",
    "Help",
    "???",
    "yes",
    "no",
    "database",
    ".",
    "the thing we discussed",
    "do the opposite",
    "nevermind",
    "more",
    "hmm",
    "ok so what now",
    "What is the meaning of life?",

    # ─── MULTI-TURN SEQUENCE ── 10 queries ─────────────────────────
    "I'm building a chat app with PostgreSQL",
    "Should I use pgvector or Pinecone for embeddings?",
    "Ok let's go with pgvector. How should I index the embeddings?",
    "Now I need full text search too. Can PostgreSQL handle both?",
    "How do I combine the vector results with text search results?",
    "What about a reranker on top of that?",
    "How much latency does the reranker add?",
    "Actually, let's switch topics. How does Redis pub/sub work?",
    "Could I use that for real-time notifications?",
    "Let's go back to the database schema. What tables do I need?",
]


@dataclass
class RetrievalArmResult:
    """Results from one retrieval configuration arm."""
    name: str
    description: str
    avg_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_similarity: float = 0.0
    std_similarity: float = 0.0
    avg_docs_retrieved: float = 0.0
    avg_response_tokens: float = 0.0
    error_count: int = 0
    queries: int = 0
    # Per-query retrieval info extracted from telemetry
    retrieval_details: list[dict] = None
    # Per-query latencies for statistical tests
    all_latencies: list[float] = None
    all_similarities: list[float] = None

    def __post_init__(self):
        if self.retrieval_details is None:
            self.retrieval_details = []
        if self.all_latencies is None:
            self.all_latencies = []
        if self.all_similarities is None:
            self.all_similarities = []


def _extract_retrieval_metrics(result: ExperimentResult) -> RetrievalArmResult:
    """Extract retrieval-specific metrics from an experiment result."""
    import numpy as np

    arm = RetrievalArmResult(
        name=result.arm,
        description=result.config.get("description", ""),
        queries=len(result.queries),
        error_count=len(result.errors),
    )

    latencies = [l for l in result.latencies_ms if l > 0]
    arm.all_latencies = latencies
    if latencies:
        arr = np.array(latencies)
        arm.avg_latency_ms = round(float(np.mean(arr)), 1)
        arm.std_latency_ms = round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 1)
        arm.p95_latency_ms = round(float(np.percentile(arr, 95)), 1)

    # Extract from responses
    similarities = []
    doc_counts = []
    token_counts = []
    for resp in result.responses:
        ri = resp.get("retrieval_info", {})
        if ri.get("rag_avg_similarity"):
            similarities.append(ri["rag_avg_similarity"])
        if ri.get("num_docs") is not None:
            doc_counts.append(ri["num_docs"])

        # Estimate tokens from response length (rough: 1 token ≈ 4 chars)
        resp_text = resp.get("response", "")
        if resp_text:
            token_counts.append(len(resp_text) // 4)

        arm.retrieval_details.append({
            "hybrid": ri.get("hybrid_search", False),
            "reranker": ri.get("reranker", False),
            "num_docs": ri.get("num_docs", 0),
            "best_sim": ri.get("rag_best_similarity", 0),
            "avg_sim": ri.get("rag_avg_similarity", 0),
        })

    arm.all_similarities = similarities
    if similarities:
        arr = np.array(similarities)
        arm.avg_similarity = round(float(np.mean(arr)), 4)
        arm.std_similarity = round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4)
    if doc_counts:
        arm.avg_docs_retrieved = round(sum(doc_counts) / len(doc_counts), 1)
    if token_counts:
        arm.avg_response_tokens = round(sum(token_counts) / len(token_counts), 0)

    return arm


def format_retrieval_comparison(arms: list[RetrievalArmResult]) -> str:
    """Format retrieval arm results as a README-ready markdown table.

    Produces the single comparison table engineers want:
    Arm | Precision | Faithfulness | Latency | Tokens
    Plus deltas with threshold evaluation.
    """
    lines = []
    lines.append("## Retrieval Quality: 4-Arm A/B Experiment\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Queries:** {arms[0].queries if arms else 0}")
    lines.append(f"**Query categories:** keyword-heavy (30), semantic (30), multi-concept (20), follow-up (15), ambiguous (15), multi-turn (10)\n")

    # ── Primary comparison table (the one for README) ──
    lines.append("### Results\n")
    lines.append("| Arm | Similarity (mean±std) | Docs/Query | Latency (mean±std) | P95 | Tokens | Errors |")
    lines.append("|:----|:---------------------:|:----------:|:------------------:|:---:|:------:|:------:|")

    for arm in arms:
        lines.append(
            f"| **{arm.name}** | "
            f"{arm.avg_similarity:.3f}±{arm.std_similarity:.3f} | "
            f"{arm.avg_docs_retrieved:.1f} | "
            f"{arm.avg_latency_ms:.0f}±{arm.std_latency_ms:.0f}ms | "
            f"{arm.p95_latency_ms:.0f}ms | "
            f"{arm.avg_response_tokens:.0f} | "
            f"{arm.error_count} |"
        )

    # ── Delta table vs baseline ──
    if len(arms) >= 2:
        baseline = arms[0]
        lines.append(f"\n### Deltas vs `{baseline.name}`\n")
        lines.append("| Arm | Similarity Δ | Δ% | Latency Δ | Verdict |")
        lines.append("|:----|:------------:|:--:|:---------:|:-------:|")

        for arm in arms[1:]:
            sim_delta = arm.avg_similarity - baseline.avg_similarity
            sim_pct = (sim_delta / baseline.avg_similarity * 100) if baseline.avg_similarity > 0 else 0
            lat_delta = arm.avg_latency_ms - baseline.avg_latency_ms
            sim_sign = "+" if sim_delta >= 0 else ""
            lat_sign = "+" if lat_delta >= 0 else ""

            # Verdict based on thresholds
            if abs(sim_delta) <= 0.02:
                verdict = "⚪ noise"
            elif sim_delta > 0.10:
                verdict = "✅ significant win" if lat_delta < 500 else "🟡 win but costly"
            elif sim_delta > 0.02:
                verdict = "🟡 marginal"
            else:
                verdict = "❌ regression"

            lines.append(
                f"| **{arm.name}** | "
                f"{sim_sign}{sim_delta:.4f} | "
                f"{sim_sign}{sim_pct:.1f}% | "
                f"{lat_sign}{lat_delta:.0f}ms | "
                f"{verdict} |"
            )

    # ── Pre-defined thresholds ──
    lines.append("\n### Pre-defined Success Thresholds\n")
    lines.append("*Defined before experiment to prevent post-hoc rationalization.*\n")
    lines.append("| Criterion | Threshold | Status |")
    lines.append("|-----------|:---------:|:------:|")
    lines.append("| Hybrid: precision ≥ +10% over vector | +10% | ⏳ pending |")
    lines.append("| Hybrid: recall ≥ +10% over vector | +10% | ⏳ pending |")
    lines.append("| Reranker: faithfulness ≥ +15% | +15% | ⏳ pending |")
    lines.append("| Reranker: relevance ≥ +10% | +10% | ⏳ pending |")
    lines.append("| Max latency overhead ≤ 500ms | 500ms | ⏳ pending |")
    lines.append("| Noise floor: Δ < ±2% is not signal | ±2% | — |")

    # ── Raw per-arm summary ──
    lines.append("\n### Per-Arm Details\n")
    for arm in arms:
        lines.append(f"**{arm.name}:** {arm.description}")
        hybrid_count = sum(1 for d in arm.retrieval_details if d.get("hybrid"))
        rerank_count = sum(1 for d in arm.retrieval_details if d.get("reranker"))
        lines.append(f"  - Hybrid active: {hybrid_count}/{arm.queries} queries")
        lines.append(f"  - Reranker active: {rerank_count}/{arm.queries} queries")
        if arm.all_similarities:
            lines.append(f"  - Similarity range: [{min(arm.all_similarities):.3f}, {max(arm.all_similarities):.3f}]")
        lines.append("")

    return "\n".join(lines)


def run_retrieval_experiment(
    runner: ExperimentRunner,
    queries: list[str] | None = None,
) -> tuple[list[ExperimentResult], list[RetrievalArmResult], str]:
    """Run the 4-arm retrieval quality experiment."""
    if queries is None:
        queries = RETRIEVAL_EVAL_QUERIES

    arms = EXPERIMENT_ARMS[Experiment.RETRIEVAL_QUALITY]

    logger.info(f"Running retrieval quality experiment: {len(arms)} arms × {len(queries)} queries")

    results = []
    arm_metrics = []

    for i, arm_config in enumerate(arms):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  ARM {i + 1}/{len(arms)}: {arm_config.name}")
        logger.info(f"  {arm_config.description}")
        logger.info(f"{'=' * 60}")

        result = runner.run_arm(arm_config, queries, "retrieval_quality")
        results.append(result)
        arm_metrics.append(_extract_retrieval_metrics(result))

    report = format_retrieval_comparison(arm_metrics)
    return results, arm_metrics, report


def main():
    parser = argparse.ArgumentParser(description="Retrieval quality A/B experiment")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--output", default=None, help="Output markdown path")
    args = parser.parse_args()

    runner = ExperimentRunner(base_url=args.url)

    results, arm_metrics, report = run_retrieval_experiment(runner)

    # Terminal output
    print("\n" + report)

    # Save markdown
    md_path = Path(args.output or f"experiments/results/retrieval_quality_{int(time.time())}.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report, encoding="utf-8")
    logger.info(f"Saved report to {md_path}")

    # Save raw results
    json_path = md_path.with_suffix(".json")
    ExperimentRunner.save_results(results, json_path)
    logger.info(f"Saved raw results to {json_path}")


if __name__ == "__main__":
    main()
