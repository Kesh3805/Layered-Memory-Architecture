"""Fast retrieval quality experiment — measures pure retrieval without LLM generation.

Uses the /retrieval/test endpoint so each query takes ~100–500 ms instead of
5–30 s through /chat.  This makes 120 queries × 4 arms feasible in ~5 minutes.

Metrics collected (all without LLM):
  - Avg / std cosine similarity of retrieved docs
  - Avg / std retrieval latency (pure retrieval, no generation)
  - P95 latency
  - Document overlap: fraction of docs that differ from vector baseline
  - Number of docs retrieved per query

Usage:
    python experiments/eval_retrieval_fast.py
    python experiments/eval_retrieval_fast.py --queries 60 --output results/fast_4arm.md
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"


# ═══════════════════════════════════════════════════════════════════════════
#  SUCCESS THRESHOLDS — defined BEFORE running, no post-hoc rationalisation
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    # Hybrid must improve avg cosine similarity by at least this to justify complexity:
    "hybrid_similarity_min_delta": 0.02,          # +2 pp over vector baseline
    # Document diversity — hybrid should surface different docs on some queries:
    "hybrid_doc_diversity_min": 0.10,             # ≥10 % of docs differ from vector
    # Reranker must improve similarity for it to justify its latency:
    "reranker_similarity_min_delta": 0.02,        # +2 pp over hybrid-only
    # Maximum acceptable added latency per query:
    "max_latency_delta_ms": 500,                  # feature must not add >500 ms
    # Noise floor: deltas below this are statistical noise:
    "noise_floor_similarity": 0.005,              # ±0.5 pp = noise
    "noise_floor_latency_ms": 50,                 # ±50 ms = noise
}


# ═══════════════════════════════════════════════════════════════════════════
#  EVAL CORPUS — varied queries that stress different retrieval paths
# ═══════════════════════════════════════════════════════════════════════════

EVAL_QUERIES = [
    # ── KEYWORD-HEAVY (BM25 should shine) ─────────────────────────
    "What is pgvector and how do I install it?",
    "PostgreSQL 16 new features and improvements",
    "HNSW index configuration parameters",
    "EXPLAIN ANALYZE query plan in PostgreSQL",
    "FastAPI dependency injection with Depends()",
    "psycopg2 connection pool SimpleConnectionPool",
    "sentence-transformers BAAI/bge-base-en-v1.5 model",
    "PostgreSQL GIN index for tsvector full text search",
    "Reciprocal Rank Fusion algorithm",
    "cross-encoder/ms-marco-MiniLM-L-6-v2 reranking",
    "BM25 term frequency inverse document frequency",
    "pgvector ivfflat vs hnsw index comparison",
    "cosine similarity formula for embeddings",
    "Docker ENTRYPOINT vs CMD difference",
    "Python asyncio vs threading comparison",
    "uvicorn --workers flag production",
    "Redis HSET command syntax",
    "PostgreSQL ALTER TABLE ADD COLUMN",
    "CREATE INDEX CONCURRENTLY PostgreSQL",
    "pip install sentence-transformers",

    # ── SEMANTIC / PARAPHRASE (vector should shine) ────────────────
    "How do I make my database queries faster?",
    "What's the best way to store conversation history?",
    "How can I find similar documents in a collection?",
    "How do I make my system resilient to failures?",
    "What's the best approach for tracking system health?",
    "How can I prevent unauthorized access to my API?",
    "How do I speed up responses for repeated requests?",
    "What's the smartest way to combine results from different sources?",
    "How do I handle errors gracefully without crashing?",
    "How do I process large data without running out of memory?",
    "What's the right way to evolve my database schema over time?",
    "How do I detect when my application is performing poorly?",
    "How do I prevent one slow query from blocking everything?",
    "What strategies exist for reducing API response times?",
    "How should I handle database connection failures?",
    "How do I make my search understand what people mean?",
    "How do I coordinate work between background processes?",
    "How do I keep my system running when parts fail?",
    "What's a good strategy for keeping frequently used data close?",
    "What's the reliable way to test code that uses a database?",

    # ── MULTI-CONCEPT (needs both BM25 and vector) ─────────────────
    "How does connection pooling work with PostgreSQL and pgvector?",
    "How do vector embeddings improve search relevance compared to BM25?",
    "How do you combine Redis caching with PostgreSQL?",
    "Compare Docker Compose and Kubernetes for orchestration",
    "How do you implement semantic search with pgvector and sentence-transformers?",
    "Explain hybrid search combining BM25 with vector similarity",
    "How does cross-encoder reranking improve retrieval precision?",
    "How do you test a RAG pipeline end-to-end?",
    "Explain token budgeting for LLM context window management",
    "What is the difference between sync and async database access in Python?",
    "How does EMA-based topic threading compare to k-means clustering?",
    "How do you implement policy-based retrieval gating?",
    "Explain tradeoffs of embedding model size vs retrieval quality",
    "How does LLM-as-judge evaluation compare to human evaluation?",
    "How do circuit breakers improve resilience in distributed systems?",
    "What monitoring tools work best with FastAPI and PostgreSQL?",
    "What are the tradeoffs between horizontal and vertical scaling for databases?",
    "Difference between REST and GraphQL for API design",
    "How does cross-encoder reranking improve retrieval precision?",
    "Explain event sourcing with CQRS in a microservices architecture",

    # ── AMBIGUOUS / SHORT (stress test — neither arm should excel) ─
    "database",
    "search",
    "Why?",
    "Help",
    "embeddings",
    "What is the meaning of life?",
    "How does it work?",
    "What are the tradeoffs?",
    "What else?",
    "Tell me more",

    # ── MULTI-TURN / CONTEXTUAL ────────────────────────────────────
    "I'm building a chat app with PostgreSQL",
    "Should I use pgvector or Pinecone for embeddings?",
    "Ok let's go with pgvector. How should I index the embeddings?",
    "Now I need full text search too. Can PostgreSQL handle both?",
    "How do I combine the vector results with text search results?",
    "What about a reranker on top of that?",
    "How much latency does the reranker add?",
    "Let's go back to the database schema. What tables do I need?",
    "What about connection pooling for a FastAPI app?",
    "How do I monitor query performance in PostgreSQL?",
]

assert len(EVAL_QUERIES) == 80, f"Expected 80 queries, got {len(EVAL_QUERIES)}"


# ═══════════════════════════════════════════════════════════════════════════
#  EXPERIMENT ARMS
# ═══════════════════════════════════════════════════════════════════════════

ARMS = [
    {"name": "vector_baseline",      "hybrid_search": False, "reranker": False,
     "desc": "Pure pgvector cosine — baseline"},
    {"name": "hybrid_only",          "hybrid_search": True,  "reranker": False,
     "desc": "BM25 + vector via RRF (k=60)"},
    {"name": "hybrid_plus_reranker", "hybrid_search": True,  "reranker": True,
     "desc": "Hybrid + cross-encoder reranking"},
    {"name": "full_pipeline",        "hybrid_search": True,  "reranker": True,
     "desc": "All subsystems (hybrid + reranker + behaviour engine)"},
]


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    query: str
    latency_ms: float
    num_docs: int
    avg_sim: float
    best_sim: float
    worst_sim: float
    doc_snippets: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ArmResult:
    name: str
    description: str
    queries_run: int = 0
    errors: int = 0
    latencies: list[float] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    doc_counts: list[int] = field(default_factory=list)
    # List[set[str]] — each entry is the set of doc snippets for a query
    doc_sets: list[frozenset] = field(default_factory=list)


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


def _run_query(query: str) -> QueryResult:
    start = time.perf_counter()
    try:
        r = requests.post(
            f"{BASE_URL}/retrieval/test",
            json={"query": query},
            timeout=30,
        )
        latency = (time.perf_counter() - start) * 1000
        if r.status_code != 200:
            return QueryResult(query=query, latency_ms=latency, num_docs=0,
                               avg_sim=0.0, best_sim=0.0, worst_sim=0.0,
                               error=f"HTTP {r.status_code}")
        data = r.json()
        snippets = [d["snippet"] for d in data.get("docs", [])]
        return QueryResult(
            query=query,
            latency_ms=round(latency, 1),
            num_docs=data["num_docs"],
            avg_sim=data["avg_similarity"],
            best_sim=data["best_similarity"],
            worst_sim=data["worst_similarity"],
            doc_snippets=snippets,
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return QueryResult(query=query, latency_ms=round(latency, 1),
                           num_docs=0, avg_sim=0.0, best_sim=0.0, worst_sim=0.0,
                           error=str(exc))


def _run_arm(arm: dict, queries: list[str]) -> ArmResult:
    result = ArmResult(name=arm["name"], description=arm["desc"])
    _apply_config(arm)
    logger.info(f"  Config verified: hybrid={arm['hybrid_search']} reranker={arm['reranker']}")

    for i, q in enumerate(queries):
        qr = _run_query(q)
        if qr.error:
            result.errors += 1
            logger.warning(f"  [{i+1}/{len(queries)}] ERROR: {qr.error}")
        else:
            result.latencies.append(qr.latency_ms)
            result.similarities.append(qr.avg_sim)
            result.doc_counts.append(qr.num_docs)
            result.doc_sets.append(frozenset(qr.doc_snippets))
        result.queries_run += 1

        if (i + 1) % 10 == 0:
            if result.similarities:
                running_avg = np.mean(result.similarities)
                logger.info(f"  Progress {i+1}/{len(queries)} — running avg sim: {running_avg:.4f}")

    _reset_config()
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTICS + REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p95": 0.0, "n": 0}
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "n": len(arr),
    }


def _doc_overlap(baseline_sets: list[frozenset], arm_sets: list[frozenset]) -> float:
    """Fraction of queries where arm retrieves ANY different docs from baseline."""
    if not baseline_sets or not arm_sets:
        return 0.0
    diffs = sum(
        1 for b, a in zip(baseline_sets, arm_sets) if a != b
    )
    return round(diffs / len(baseline_sets), 3)


def _verdict(delta: float, threshold: float, noise: float) -> str:
    if abs(delta) <= noise:
        return "⚪ noise"
    if delta >= threshold:
        return "✅ win"
    if delta > noise:
        return "🟡 marginal"
    return "❌ regression"


def format_report(arms: list[ArmResult]) -> str:
    lines: list[str] = []
    lines.append("## Retrieval Quality: 4-Arm A/B Experiment\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Queries per arm:** {arms[0].queries_run if arms else 0}")
    lines.append(f"**Total queries:** {sum(a.queries_run for a in arms)}")
    lines.append(f"**Method:** Pure retrieval (no LLM generation) — `/retrieval/test` endpoint\n")

    lines.append("### Pre-Defined Success Thresholds\n")
    lines.append("*Set before running to prevent post-hoc rationalisation.*\n")
    lines.append("| Criterion | Threshold |")
    lines.append("|-----------|:---------:|")
    lines.append(f"| Hybrid Δsimilarity ≥ noise floor | +{THRESHOLDS['hybrid_similarity_min_delta']:.3f} |")
    lines.append(f"| Hybrid doc diversity ≥ | {THRESHOLDS['hybrid_doc_diversity_min']:.0%} |")
    lines.append(f"| Reranker Δsimilarity ≥ noise floor | +{THRESHOLDS['reranker_similarity_min_delta']:.3f} |")
    lines.append(f"| Max added latency per query | {THRESHOLDS['max_latency_delta_ms']} ms |")
    lines.append(f"| Noise floor (similarity) | ±{THRESHOLDS['noise_floor_similarity']:.3f} |")
    lines.append(f"| Noise floor (latency) | ±{THRESHOLDS['noise_floor_latency_ms']} ms |\n")

    # ── Summary table ──
    stats_by_arm = {}
    for arm in arms:
        sim = _stats(arm.similarities)
        lat = _stats(arm.latencies)
        doc = _stats([float(d) for d in arm.doc_counts])
        stats_by_arm[arm.name] = {"sim": sim, "lat": lat, "doc": doc}

    lines.append("### Results\n")
    lines.append("| Arm | Similarity (mean±std) | Latency ms (mean±std) | P95 ms | Docs/q | Errors |")
    lines.append("|:----|:---------------------:|:---------------------:|:------:|:------:|:------:|")
    for arm in arms:
        sim = stats_by_arm[arm.name]["sim"]
        lat = stats_by_arm[arm.name]["lat"]
        doc = stats_by_arm[arm.name]["doc"]
        lines.append(
            f"| **{arm.name}** | {sim['mean']:.4f}±{sim['std']:.4f} | "
            f"{lat['mean']:.0f}±{lat['std']:.0f} | "
            f"{lat['p95']:.0f} | "
            f"{doc['mean']:.1f} | {arm.errors} |"
        )

    # ── Delta table ──
    if len(arms) >= 2:
        baseline = arms[0]
        b_sim = _stats(baseline.similarities)
        b_lat = _stats(baseline.latencies)
        lines.append(f"\n### Deltas vs `{baseline.name}`\n")
        lines.append("| Arm | Δ Similarity | Δ Latency ms | Doc diversity | Verdict |")
        lines.append("|:----|:------------:|:------------:|:-------------:|:-------:|")
        for arm in arms[1:]:
            a_sim = _stats(arm.similarities)
            a_lat = _stats(arm.latencies)
            sim_d = a_sim["mean"] - b_sim["mean"]
            lat_d = a_lat["mean"] - b_lat["mean"]
            diversity = _doc_overlap(baseline.doc_sets, arm.doc_sets)
            sim_sign = "+" if sim_d >= 0 else ""
            lat_sign = "+" if lat_d >= 0 else ""

            if abs(sim_d) <= THRESHOLDS["noise_floor_similarity"]:
                v = "⚪ noise"
            elif sim_d >= THRESHOLDS["hybrid_similarity_min_delta"]:
                v = "✅ significant win" if lat_d < THRESHOLDS["max_latency_delta_ms"] else "🟡 win but costly"
            elif sim_d > THRESHOLDS["noise_floor_similarity"]:
                v = "🟡 marginal"
            else:
                v = "❌ regression"

            lines.append(
                f"| **{arm.name}** | {sim_sign}{sim_d:.4f} | "
                f"{lat_sign}{lat_d:.0f} | "
                f"{diversity:.1%} | {v} |"
            )

    # ── Threshold evaluation ──
    lines.append("\n### Threshold Checklist\n")
    lines.append("*Each criterion was defined before running.*\n")
    if len(arms) >= 4:
        hybrid = arms[1]
        reranker = arms[2]
        b_sim = _stats(arms[0].similarities)
        h_sim = _stats(hybrid.similarities)
        r_sim = _stats(reranker.similarities)
        b_lat = _stats(arms[0].latencies)
        h_lat = _stats(hybrid.latencies)
        r_lat = _stats(reranker.latencies)

        hybrid_sim_d = h_sim["mean"] - b_sim["mean"]
        reranker_sim_d = r_sim["mean"] - h_sim["mean"]
        hybrid_lat_d = h_lat["mean"] - b_lat["mean"]
        reranker_lat_d = r_lat["mean"] - h_lat["mean"]
        hybrid_diversity = _doc_overlap(arms[0].doc_sets, hybrid.doc_sets)

        def chk(passed: bool) -> str:
            return "✅ PASS" if passed else "❌ FAIL"

        lines.append("| Criterion | Threshold | Measured | Status |")
        lines.append("|-----------|:---------:|:--------:|:------:|")
        lines.append(
            f"| Hybrid Δsimilarity | ≥{THRESHOLDS['hybrid_similarity_min_delta']:.3f} | "
            f"{hybrid_sim_d:+.4f} | {chk(hybrid_sim_d >= THRESHOLDS['hybrid_similarity_min_delta'])} |"
        )
        lines.append(
            f"| Hybrid doc diversity | ≥{THRESHOLDS['hybrid_doc_diversity_min']:.0%} | "
            f"{hybrid_diversity:.1%} | {chk(hybrid_diversity >= THRESHOLDS['hybrid_doc_diversity_min'])} |"
        )
        lines.append(
            f"| Hybrid latency delta | ≤{THRESHOLDS['max_latency_delta_ms']} ms | "
            f"{hybrid_lat_d:+.0f} ms | {chk(abs(hybrid_lat_d) <= THRESHOLDS['max_latency_delta_ms'])} |"
        )
        lines.append(
            f"| Reranker Δsimilarity | ≥{THRESHOLDS['reranker_similarity_min_delta']:.3f} | "
            f"{reranker_sim_d:+.4f} | {chk(reranker_sim_d >= THRESHOLDS['reranker_similarity_min_delta'])} |"
        )
        lines.append(
            f"| Reranker latency delta | ≤{THRESHOLDS['max_latency_delta_ms']} ms | "
            f"{reranker_lat_d:+.0f} ms | {chk(abs(reranker_lat_d) <= THRESHOLDS['max_latency_delta_ms'])} |"
        )

    # ── Honest interpretation ──
    lines.append("\n### Engineering Interpretation\n")
    if len(arms) >= 2:
        b_sim = _stats(arms[0].similarities)
        h_sim = _stats(arms[1].similarities)
        r_sim = _stats(arms[2].similarities) if len(arms) >= 3 else h_sim
        hybrid_sim_d = h_sim["mean"] - b_sim["mean"]
        hybrid_diversity = _doc_overlap(arms[0].doc_sets, arms[1].doc_sets)

        if abs(hybrid_sim_d) <= THRESHOLDS["noise_floor_similarity"]:
            lines.append(
                f"> **Hybrid search:** Δsimilarity = {hybrid_sim_d:+.4f} (noise floor ±{THRESHOLDS['noise_floor_similarity']:.3f}). "
                f"On this {arms[0].queries_run}-query corpus, hybrid and pure vector retrieve identical documents "
                f"{(1-hybrid_diversity):.0%} of the time. BM25 + vector fusion adds no measurable quality "
                f"improvement over pure vector on a {len(EVAL_QUERIES)}-document knowledge base."
            )
        elif hybrid_sim_d > 0:
            lines.append(
                f"> **Hybrid search:** Δsimilarity = {hybrid_sim_d:+.4f} (above noise floor). "
                f"Hybrid search surfaces different documents {hybrid_diversity:.0%} of the time. "
                f"The improvement is {'significant' if hybrid_sim_d >= THRESHOLDS['hybrid_similarity_min_delta'] else 'marginal'}."
            )
        else:
            lines.append(
                f"> **Hybrid search:** Δsimilarity = {hybrid_sim_d:+.4f} — a regression. "
                f"Hybrid search is performing WORSE than pure vector on this corpus."
            )

        reranker_sim_d = r_sim["mean"] - h_sim["mean"]
        if abs(reranker_sim_d) <= THRESHOLDS["noise_floor_similarity"]:
            lines.append(
                f"> **Reranker:** Δsimilarity = {reranker_sim_d:+.4f} (noise floor). "
                f"Cross-encoder reranking shows no measurable cosine similarity improvement. "
                f"Note: cosine similarity measures document relevance to the query embedding; "
                f"reranking optimises semantic matching which is only fully captured by LLM-as-judge "
                f"faithfulness/relevance metrics."
            )
        elif reranker_sim_d > 0:
            lines.append(
                f"> **Reranker:** Δsimilarity = {reranker_sim_d:+.4f}. "
                f"Reranking shows a {'significant' if reranker_sim_d >= THRESHOLDS['reranker_similarity_min_delta'] else 'marginal'} improvement."
            )
        else:
            lines.append(
                f"> **Reranker:** Δsimilarity = {reranker_sim_d:+.4f} — regression vs hybrid-only."
            )

        lines.append(
            "\n> **Caveat:** Cosine similarity is an imperfect proxy for retrieval quality. "
            "The gold standard metrics (faithfulness, answer relevance via LLM-as-judge) require "
            "running the full `/chat` pipeline. These results establish the retrieval-only baseline; "
            "production recommendation should be verified with end-to-end quality metrics."
        )

    return "\n".join(lines)


def run_experiment(queries: list[str]) -> list[ArmResult]:
    results: list[ArmResult] = []
    for i, arm in enumerate(ARMS):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  ARM {i+1}/{len(ARMS)}: {arm['name']}")
        logger.info(f"  {arm['desc']}")
        logger.info(f"{'=' * 60}")
        ar = _run_arm(arm, queries)
        results.append(ar)
        if ar.similarities:
            sim = _stats(ar.similarities)
            lat = _stats(ar.latencies)
            logger.info(
                f"  Done — sim {sim['mean']:.4f}±{sim['std']:.4f}, "
                f"lat {lat['mean']:.0f}±{lat['std']:.0f} ms, errors={ar.errors}"
            )
    return results


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="Fast retrieval quality A/B experiment")
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--queries", type=int, default=len(EVAL_QUERIES),
                        help="Number of queries to run per arm (default: all)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    BASE_URL = args.url

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Backend not reachable: {e}")
        raise SystemExit(1)

    queries = EVAL_QUERIES[: args.queries]
    logger.info(f"\nRunning {len(queries)} queries × {len(ARMS)} arms = {len(queries)*len(ARMS)} total retrievals")

    results = run_experiment(queries)

    report = format_report(results)
    print("\n" + report)

    out = Path(args.output or f"experiments/results/retrieval_fast_{int(time.time())}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    logger.info(f"\nSaved report → {out}")

    # Save raw JSON
    json_out = out.with_suffix(".json")
    raw = []
    for arm in results:
        raw.append({
            "name": arm.name,
            "description": arm.description,
            "queries_run": arm.queries_run,
            "errors": arm.errors,
            "similarity_mean": _stats(arm.similarities)["mean"],
            "similarity_std": _stats(arm.similarities)["std"],
            "latency_mean_ms": _stats(arm.latencies)["mean"],
            "latency_std_ms": _stats(arm.latencies)["std"],
            "latency_p95_ms": _stats(arm.latencies)["p95"],
            "avg_docs_per_query": _stats([float(d) for d in arm.doc_counts])["mean"],
        })
    json_out.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    logger.info(f"Saved raw JSON → {json_out}")


if __name__ == "__main__":
    main()
