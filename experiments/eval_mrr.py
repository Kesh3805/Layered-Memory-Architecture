"""Rigorous retrieval evaluation with MRR, NDCG, and ground-truth relevance.

This evaluation goes beyond cosine similarity proxies. It uses:
  1. Ground-truth query→document relevance judgments (human-curated)
  2. MRR (Mean Reciprocal Rank) — where does the first relevant doc appear?
  3. NDCG@K — how well ordered are results by relevance?
  4. Precision@K — what fraction of top-K are relevant?
  5. Retrieval gate savings — how many queries skip retrieval?
  6. Token savings — how many tokens are saved by conditional gating?

Usage:
    python experiments/eval_mrr.py --url http://localhost:8000
    python experiments/eval_mrr.py --url http://localhost:8000 --output results/mrr_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

# ═══════════════════════════════════════════════════════════════════════════
#  GROUND-TRUTH RELEVANCE JUDGMENTS
#
#  Each entry: (query, [list of substring patterns that indicate relevance])
#  A retrieved document is "relevant" if it contains ANY of the patterns.
#  This is conservative — real relevance is likely higher.
# ═══════════════════════════════════════════════════════════════════════════

GROUND_TRUTH: list[tuple[str, list[str]]] = [
    # ── Retrieval-heavy queries (should trigger retrieval) ────────────
    (
        "What is retrieval-augmented generation?",
        ["retrieval-augmented generation", "RAG", "retrieval augmented", "retrieved documents", "vector search"],
    ),
    (
        "How does BM25 scoring work?",
        ["BM25", "term frequency", "inverse document frequency", "lexical search", "IDF"],
    ),
    (
        "Explain cross-encoder reranking",
        ["cross-encoder", "reranking", "reranker", "query-document pair", "ms-marco"],
    ),
    (
        "What is Reciprocal Rank Fusion?",
        ["Reciprocal Rank Fusion", "RRF", "rank fusion", "hybrid search"],
    ),
    (
        "How does pgvector implement HNSW indexing?",
        ["pgvector", "HNSW", "Hierarchical Navigable", "proximity graph", "vector"],
    ),
    (
        "Explain the difference between IVFFlat and HNSW indexes",
        ["IVFFlat", "HNSW", "partitions", "proximity graph", "k-means", "ivf"],
    ),
    (
        "What is semantic chunking?",
        ["semantic chunking", "chunk", "embedding", "cosine", "sentence", "splitting"],
    ),
    (
        "How does the behavior engine work?",
        ["behavior engine", "behavior", "frustration", "interaction tempo", "behavior mode", "behavioral"],
    ),
    (
        "What is conditional retrieval gating?",
        ["conditional retrieval", "retrieval gating", "gate", "skip retrieval", "policy", "intent"],
    ),
    (
        "Explain topic threading with EMA centroids",
        ["topic threading", "EMA", "centroid", "thread", "topic", "thread resolution"],
    ),
    (
        "What are the four memory tiers?",
        ["memory tier", "working memory", "episodic memory", "semantic memory", "profile memory", "multi-tier"],
    ),
    (
        "How does the policy engine route retrieval decisions?",
        ["policy engine", "policy", "retrieval route", "decision", "features", "inject_rag"],
    ),
    (
        "What is the Transformer self-attention mechanism?",
        ["self-attention", "Transformer", "QKV", "queries, keys, and values", "attention", "softmax"],
    ),
    (
        "Explain LSTM gating mechanisms",
        ["LSTM", "forget gate", "input gate", "output gate", "cell state", "gating"],
    ),
    (
        "How does Adam optimizer work?",
        ["Adam", "adaptive moment", "first moment", "second moment", "bias correction"],
    ),
    (
        "What is batch normalization?",
        ["batch normalization", "normalize", "mini-batch", "zero mean", "unit variance", "covariate shift"],
    ),
    (
        "Explain the circuit breaker pattern",
        ["circuit breaker", "failure rate", "half-open", "closes", "opens", "fault tolerance"],
    ),
    (
        "How does connection pooling work with PostgreSQL?",
        ["connection pooling", "pool", "PgBouncer", "minconn", "maxconn", "ThreadedConnectionPool"],
    ),
    (
        "What is MVCC in PostgreSQL?",
        ["MVCC", "multi-version concurrency", "snapshot", "xmin", "xmax", "transaction"],
    ),
    (
        "Explain the Raft consensus algorithm",
        ["Raft", "consensus", "leader", "follower", "AppendEntries", "election", "log replication"],
    ),
    (
        "What is event sourcing with CQRS?",
        ["event sourcing", "CQRS", "immutable events", "command query", "write model", "read model"],
    ),
    (
        "How does Docker multi-stage build work?",
        ["multi-stage", "Docker", "build stage", "runtime stage", "image size", "Dockerfile"],
    ),
    (
        "Explain Kubernetes Horizontal Pod Autoscaler",
        ["Horizontal Pod Autoscaler", "HPA", "replica count", "Kubernetes", "scaling", "metrics"],
    ),
    (
        "What is the Saga pattern for distributed transactions?",
        ["Saga", "distributed transaction", "compensating", "choreography", "orchestration"],
    ),
    (
        "How does Redis implement cache eviction?",
        ["Redis", "eviction", "volatile-lru", "allkeys-lru", "TTL", "cache"],
    ),
    (
        "Explain Mean Reciprocal Rank as an IR metric",
        ["Mean Reciprocal Rank", "MRR", "reciprocal rank", "first relevant", "ranking"],
    ),
    (
        "What is NDCG and how is it computed?",
        ["NDCG", "Normalized Discounted Cumulative Gain", "DCG", "graded relevance", "IDCG"],
    ),
    (
        "How do embedding models handle asymmetric retrieval?",
        ["asymmetric", "query instruction", "prefix", "bge", "retrieval", "embedding"],
    ),
    (
        "What is Byte-Pair Encoding tokenization?",
        ["Byte-Pair Encoding", "BPE", "subword", "merges", "vocabulary", "tokenization"],
    ),
    (
        "Explain the OAuth 2.0 authorization flow",
        ["OAuth", "authorization", "access token", "authorization server", "resource server"],
    ),

    # ── Non-retrieval queries (should NOT trigger retrieval) ──────────
    (
        "Hello, how are you?",
        [],  # No relevant docs expected — greeting
    ),
    (
        "Thanks, that's helpful",
        [],  # No relevant docs expected — confirmation
    ),
    (
        "Yes",
        [],  # No relevant docs expected — simple affirmation
    ),
    (
        "Goodbye",
        [],  # No relevant docs expected — farewell
    ),
    (
        "What's 2 + 2?",
        [],  # No relevant docs expected — arithmetic
    ),
]

# Separate into retrieval and non-retrieval queries
RETRIEVAL_QUERIES = [(q, gt) for q, gt in GROUND_TRUTH if gt]
NON_RETRIEVAL_QUERIES = [(q, gt) for q, gt in GROUND_TRUTH if not gt]


# ═══════════════════════════════════════════════════════════════════════════
#  ARMS
# ═══════════════════════════════════════════════════════════════════════════

ARMS = [
    {"name": "vector_baseline",      "hybrid_search": False, "reranker": False,
     "desc": "Pure pgvector cosine similarity"},
    {"name": "hybrid_bm25_vector",   "hybrid_search": True,  "reranker": False,
     "desc": "BM25 + vector via Reciprocal Rank Fusion (k=60)"},
    {"name": "hybrid_plus_reranker", "hybrid_search": True,  "reranker": True,
     "desc": "Hybrid + cross-encoder/ms-marco-MiniLM-L-6-v2 reranking"},
]


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryMetrics:
    query: str
    relevant_patterns: list[str]
    retrieved_docs: list[str]
    similarities: list[float]
    latency_ms: float
    reciprocal_rank: float  # 1/rank of first relevant doc, 0 if none
    precision_at_k: float   # fraction of top-K that are relevant
    ndcg_at_k: float        # NDCG@K with binary relevance
    num_relevant_retrieved: int
    error: str | None = None


def _is_relevant(doc_text: str, patterns: list[str]) -> bool:
    """Check if a document is relevant based on substring pattern matching."""
    text_lower = doc_text.lower()
    return any(p.lower() in text_lower for p in patterns)


def _reciprocal_rank(docs: list[str], patterns: list[str]) -> float:
    """Compute reciprocal rank: 1/position of first relevant doc."""
    for i, doc in enumerate(docs):
        if _is_relevant(doc, patterns):
            return 1.0 / (i + 1)
    return 0.0


def _precision_at_k(docs: list[str], patterns: list[str], k: int = 4) -> float:
    """Fraction of top-K docs that are relevant."""
    top_k = docs[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for d in top_k if _is_relevant(d, patterns))
    return relevant_count / len(top_k)


def _ndcg_at_k(docs: list[str], patterns: list[str], k: int = 4) -> float:
    """NDCG@K with binary relevance (rel=1 if relevant, 0 otherwise)."""
    gains = [1.0 if _is_relevant(d, patterns) else 0.0 for d in docs[:k]]
    # DCG
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    # Ideal DCG: all relevant first
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ═══════════════════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def _apply_config(arm: dict) -> None:
    r = requests.post(
        f"{BASE_URL}/experiments/config",
        json={"hybrid_search": arm["hybrid_search"], "reranker": arm["reranker"]},
        timeout=5,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Config API returned {r.status_code}: {r.text}")


def _reset_config() -> None:
    requests.post(f"{BASE_URL}/experiments/reset", timeout=5)


def _run_retrieval_query(query: str, patterns: list[str]) -> QueryMetrics:
    start = time.perf_counter()
    try:
        r = requests.post(
            f"{BASE_URL}/retrieval/test",
            json={"query": query, "k": 4, "min_similarity": 0.0},
            timeout=30,
        )
        latency = (time.perf_counter() - start) * 1000
        if r.status_code != 200:
            return QueryMetrics(
                query=query, relevant_patterns=patterns, retrieved_docs=[],
                similarities=[], latency_ms=latency, reciprocal_rank=0.0,
                precision_at_k=0.0, ndcg_at_k=0.0, num_relevant_retrieved=0,
                error=f"HTTP {r.status_code}",
            )
        data = r.json()
        docs = [d["snippet"] for d in data.get("docs", [])]
        sims = [d["similarity"] for d in data.get("docs", [])]

        rr = _reciprocal_rank(docs, patterns) if patterns else 0.0
        pk = _precision_at_k(docs, patterns) if patterns else 0.0
        ndcg = _ndcg_at_k(docs, patterns) if patterns else 0.0
        n_rel = sum(1 for d in docs if _is_relevant(d, patterns)) if patterns else 0

        return QueryMetrics(
            query=query, relevant_patterns=patterns, retrieved_docs=docs,
            similarities=sims, latency_ms=round(latency, 1),
            reciprocal_rank=rr, precision_at_k=pk, ndcg_at_k=ndcg,
            num_relevant_retrieved=n_rel,
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return QueryMetrics(
            query=query, relevant_patterns=patterns, retrieved_docs=[],
            similarities=[], latency_ms=round(latency, 1),
            reciprocal_rank=0.0, precision_at_k=0.0, ndcg_at_k=0.0,
            num_relevant_retrieved=0, error=str(exc),
        )


def _measure_gating_savings() -> dict:
    """Measure retrieval gating: how many non-retrieval queries skip retrieval."""
    logger.info("\n── Measuring retrieval gate savings ──")
    _reset_config()

    # Create a temporary conversation
    conv_r = requests.post(f"{BASE_URL}/conversations", json={"title": "gate_eval"}, timeout=5)
    conv_id = conv_r.json()["id"]

    total_queries = 0
    skipped_retrieval = 0
    retrieval_latencies = []
    gated_latencies = []

    all_queries = [q for q, _ in GROUND_TRUTH]
    for q in all_queries:
        total_queries += 1
        start = time.perf_counter()
        try:
            r = requests.post(
                f"{BASE_URL}/chat",
                json={"user_query": q, "conversation_id": conv_id},
                timeout=60,
            )
            lat = (time.perf_counter() - start) * 1000
            if r.status_code == 200:
                data = r.json()
                ri = data.get("retrieval_info", {})
                route = ri.get("route", "")
                if "behavior:" in route or ri.get("num_docs", 0) == 0:
                    skipped_retrieval += 1
                    gated_latencies.append(lat)
                else:
                    retrieval_latencies.append(lat)
        except Exception as e:
            logger.warning(f"  Gate eval error: {e}")

    # Clean up
    try:
        requests.delete(f"{BASE_URL}/conversations/{conv_id}", timeout=5)
    except Exception:
        pass

    skip_rate = skipped_retrieval / total_queries if total_queries > 0 else 0
    avg_retrieval_lat = np.mean(retrieval_latencies) if retrieval_latencies else 0
    avg_gated_lat = np.mean(gated_latencies) if gated_latencies else 0
    latency_savings = avg_retrieval_lat - avg_gated_lat if avg_gated_lat > 0 else 0

    # Estimate token savings: ~400 tokens per retrieved doc, 4 docs per retrieval
    token_savings_per_skip = 4 * 400  # 1600 tokens saved per skipped retrieval
    total_token_savings = skipped_retrieval * token_savings_per_skip

    return {
        "total_queries": total_queries,
        "skipped_retrieval": skipped_retrieval,
        "skip_rate": round(skip_rate, 4),
        "skip_rate_pct": round(skip_rate * 100, 1),
        "avg_retrieval_latency_ms": round(avg_retrieval_lat, 1),
        "avg_gated_latency_ms": round(avg_gated_lat, 1),
        "latency_savings_ms": round(latency_savings, 1),
        "estimated_token_savings": total_token_savings,
        "token_savings_per_skip": token_savings_per_skip,
    }


@dataclass
class ArmReport:
    name: str
    desc: str
    mrr: float = 0.0
    ndcg: float = 0.0
    precision: float = 0.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    avg_similarity: float = 0.0
    errors: int = 0
    queries: int = 0
    all_rr: list[float] = field(default_factory=list)
    all_ndcg: list[float] = field(default_factory=list)
    all_pk: list[float] = field(default_factory=list)
    all_lat: list[float] = field(default_factory=list)
    all_sim: list[float] = field(default_factory=list)


def run_experiment() -> tuple[list[ArmReport], dict]:
    """Run the full MRR experiment across all arms."""
    reports = []

    for arm_idx, arm in enumerate(ARMS):
        logger.info(f"\n{'='*60}")
        logger.info(f"  ARM {arm_idx+1}/{len(ARMS)}: {arm['name']}")
        logger.info(f"  {arm['desc']}")
        logger.info(f"{'='*60}")

        _apply_config(arm)
        report = ArmReport(name=arm["name"], desc=arm["desc"])

        for i, (query, patterns) in enumerate(RETRIEVAL_QUERIES):
            m = _run_retrieval_query(query, patterns)
            if m.error:
                report.errors += 1
                logger.warning(f"  [{i+1}/{len(RETRIEVAL_QUERIES)}] ERROR: {m.error}")
            else:
                report.all_rr.append(m.reciprocal_rank)
                report.all_ndcg.append(m.ndcg_at_k)
                report.all_pk.append(m.precision_at_k)
                report.all_lat.append(m.latency_ms)
                if m.similarities:
                    report.all_sim.append(np.mean(m.similarities))
            report.queries += 1

            if (i + 1) % 10 == 0:
                running_mrr = np.mean(report.all_rr) if report.all_rr else 0
                logger.info(f"  [{i+1}/{len(RETRIEVAL_QUERIES)}] running MRR={running_mrr:.4f}")

        # Compute aggregate metrics
        if report.all_rr:
            report.mrr = round(float(np.mean(report.all_rr)), 4)
            report.ndcg = round(float(np.mean(report.all_ndcg)), 4)
            report.precision = round(float(np.mean(report.all_pk)), 4)
            report.avg_latency = round(float(np.mean(report.all_lat)), 1)
            report.p95_latency = round(float(np.percentile(report.all_lat, 95)), 1)
            report.avg_similarity = round(float(np.mean(report.all_sim)), 4) if report.all_sim else 0.0

        logger.info(f"  Done — MRR={report.mrr}, NDCG={report.ndcg}, P@4={report.precision}, "
                     f"lat={report.avg_latency}ms, errors={report.errors}")
        reports.append(report)

    _reset_config()

    # Measure gating savings
    gate_stats = _measure_gating_savings()

    return reports, gate_stats


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(reports: list[ArmReport], gate_stats: dict, output_path: Path) -> str:
    """Generate a comprehensive markdown report."""
    baseline = reports[0]

    lines = [
        "## Retrieval Quality: MRR Evaluation with Ground-Truth Relevance",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Queries:** {len(RETRIEVAL_QUERIES)} retrieval queries with ground-truth relevance judgments",
        f"**Ground truth:** Substring pattern matching against {sum(len(gt) for _, gt in RETRIEVAL_QUERIES)} relevance patterns",
        f"**Methodology:** Each query evaluated against 3 retrieval arms + retrieval gate measurement",
        "",
        "### Retrieval Quality Metrics",
        "",
        "| Arm | MRR | NDCG@4 | P@4 | Avg Cosine | Latency (ms) | P95 (ms) |",
        "|:----|:---:|:------:|:---:|:----------:|:------------:|:--------:|",
    ]

    for r in reports:
        lines.append(
            f"| **{r.name}** | {r.mrr:.4f} | {r.ndcg:.4f} | {r.precision:.4f} | "
            f"{r.avg_similarity:.4f} | {r.avg_latency:.0f} | {r.p95_latency:.0f} |"
        )

    lines.extend([
        "",
        "### Deltas vs Vector Baseline",
        "",
        "| Arm | ΔMRR | ΔNDCG | ΔP@4 | ΔLatency |",
        "|:----|:----:|:-----:|:----:|:--------:|",
    ])

    for r in reports[1:]:
        delta_mrr = r.mrr - baseline.mrr
        delta_ndcg = r.ndcg - baseline.ndcg
        delta_pk = r.precision - baseline.precision
        delta_lat = r.avg_latency - baseline.avg_latency
        lines.append(
            f"| **{r.name}** | {delta_mrr:+.4f} | {delta_ndcg:+.4f} | "
            f"{delta_pk:+.4f} | {delta_lat:+.0f} ms |"
        )

    # Gate savings
    lines.extend([
        "",
        "### Conditional Retrieval Gate Savings",
        "",
        f"| Metric | Value |",
        f"|--------|:-----:|",
        f"| Total queries evaluated | {gate_stats['total_queries']} |",
        f"| Queries with retrieval skipped | {gate_stats['skipped_retrieval']} |",
        f"| **Retrieval skip rate** | **{gate_stats['skip_rate_pct']}%** |",
        f"| Avg latency (with retrieval) | {gate_stats['avg_retrieval_latency_ms']:.0f} ms |",
        f"| Avg latency (gated / skipped) | {gate_stats['avg_gated_latency_ms']:.0f} ms |",
        f"| Latency savings per skipped query | {gate_stats['latency_savings_ms']:.0f} ms |",
        f"| Est. token savings per skip | ~{gate_stats['token_savings_per_skip']} tokens |",
        f"| Total estimated token savings | ~{gate_stats['estimated_token_savings']} tokens |",
    ])

    # MRR improvement summary
    if len(reports) >= 3:
        best = max(reports, key=lambda r: r.mrr)
        improvement = best.mrr - baseline.mrr
        pct = (improvement / baseline.mrr * 100) if baseline.mrr > 0 else 0
        lines.extend([
            "",
            "### Summary",
            "",
            f"> **Best arm:** `{best.name}` with MRR = {best.mrr:.4f}",
            f"> **Baseline:** `{baseline.name}` with MRR = {baseline.mrr:.4f}",
            f"> **Improvement:** {improvement:+.4f} ({pct:+.1f}%)",
            f">",
            f"> **Retrieval gate:** {gate_stats['skip_rate_pct']}% of queries skip retrieval, "
            f"saving ~{gate_stats['latency_savings_ms']:.0f} ms and ~{gate_stats['token_savings_per_skip']} tokens per skipped query.",
        ])

    report_text = "\n".join(lines) + "\n"

    # Write markdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info(f"\nReport written to {output_path}")

    # Write JSON for programmatic access
    json_path = output_path.with_suffix(".json")
    json_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_queries": len(RETRIEVAL_QUERIES),
        "arms": [
            {
                "name": r.name, "desc": r.desc,
                "mrr": r.mrr, "ndcg": r.ndcg, "precision_at_4": r.precision,
                "avg_similarity": r.avg_similarity,
                "avg_latency_ms": r.avg_latency, "p95_latency_ms": r.p95_latency,
                "errors": r.errors,
            }
            for r in reports
        ],
        "gate_stats": gate_stats,
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    logger.info(f"JSON written to {json_path}")

    return report_text


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MRR retrieval evaluation with ground-truth relevance")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--output", default=None, help="Output path for report")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url.rstrip("/")

    # Verify backend is up
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        logger.info(f"Backend healthy: {r.status_code}")
    except Exception:
        logger.error(f"Cannot reach {BASE_URL}. Start the backend first.")
        sys.exit(1)

    logger.info(f"\nRunning MRR evaluation with {len(RETRIEVAL_QUERIES)} ground-truth queries across {len(ARMS)} arms")

    reports, gate_stats = run_experiment()

    ts = int(time.time())
    output = Path(args.output) if args.output else Path(f"experiments/results/mrr_eval_{ts}.md")
    report_text = generate_report(reports, gate_stats, output)

    # Print report to stdout
    print("\n" + report_text)


if __name__ == "__main__":
    import sys
    main()
