"""Hybrid search — BM25 (full-text) + vector (pgvector) with Reciprocal Rank Fusion.

Why hybrid search
-----------------
Pure vector search excels at semantic similarity but misses exact keyword
matches (e.g. "PostgreSQL 16" vs "postgres sixteen").  Pure BM25 excels at
lexical matching but cannot capture paraphrases.  Combining both via RRF
consistently outperforms either alone on retrieval benchmarks.

Architecture
------------
1. PostgreSQL ``tsvector`` column on ``document_chunks`` for BM25 scoring.
2. ``pgvector`` HNSW index for cosine similarity (already exists).
3. Both result sets fused via Reciprocal Rank Fusion (RRF):

       RRF_score(d) = Σ  1 / (k + rank_i(d))

   where ``k`` is a smoothing constant (default 60) and ``rank_i(d)`` is
   the rank of document ``d`` in result set ``i``.

Public API
----------
    hybrid_search(query, embedding, k=4, ...)  → list[(text, score)]
    hybrid_search_texts(query, embedding, k=4, ...) → list[str]

Both functions require a DB connection.  Falls back to pure vector search
if the ``tsv`` column is missing (pre-migration databases).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HybridConfig:
    """Tuning knobs for hybrid search."""

    # RRF smoothing constant — higher values flatten the rank distribution.
    rrf_k: int = 60

    # Weight multiplier for vector results before RRF fusion.
    # 1.0 = equal weight; >1.0 biases toward semantic; <1.0 biases toward BM25.
    vector_weight: float = 1.0

    # Weight multiplier for BM25 results.
    bm25_weight: float = 1.0

    # How many candidates to fetch from each arm before fusion.
    # Must be >= final k to give RRF enough candidates.
    candidate_multiplier: int = 3

    # Minimum similarity for vector candidates (pre-fusion filter).
    vector_min_similarity: float = 0.0

    # When True, fall back to pure vector search if BM25 returns 0 results
    # (common for short/ambiguous queries).
    fallback_to_vector: bool = True


DEFAULT_CONFIG = HybridConfig()


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists via weighted RRF.

    Each ranked list is ``[(doc_id, score)]`` ordered by descending score.
    Returns ``[(doc_id, rrf_score)]`` sorted by descending RRF score.
    """
    scores: dict[int, float] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf = weight / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── BM25 Search (PostgreSQL tsvector) ────────────────────────────────────

def _bm25_search(
    cur,
    query: str,
    candidate_k: int,
) -> list[tuple[int, str, float]]:
    """Full-text search using PostgreSQL ts_rank_cd.

    Returns [(id, content, bm25_score)] ordered by descending score.
    Falls back to empty list if the tsv column doesn't exist.
    """
    try:
        # plainto_tsquery handles user input safely (no special syntax needed).
        # ts_rank_cd uses cover density ranking — better than ts_rank for passages.
        cur.execute("""
            SELECT id, content, ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS rank
            FROM document_chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, candidate_k))
        return cur.fetchall()
    except Exception as e:
        # Column doesn't exist (pre-migration) or other error.
        logger.debug("BM25 search unavailable: %s", e)
        try:
            cur.connection.rollback()
        except Exception:
            pass
        return []


# ── Vector Search ─────────────────────────────────────────────────────────

def _vector_search(
    cur,
    embedding,
    candidate_k: int,
    min_similarity: float = 0.0,
) -> list[tuple[int, str, float]]:
    """Cosine similarity search via pgvector.

    Returns [(id, content, similarity)] ordered by descending similarity.
    """
    emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    cur.execute("""
        SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (emb, emb, candidate_k))
    results = cur.fetchall()
    return [(r[0], r[1], r[2]) for r in results if r[2] is not None and r[2] >= min_similarity]


# ── Hybrid Search (public API) ───────────────────────────────────────────

def hybrid_search(
    conn,
    query: str,
    embedding,
    k: int = 4,
    config: HybridConfig | None = None,
) -> list[tuple[str, float]]:
    """Hybrid BM25 + vector search with RRF fusion.

    Args:
        conn:      Active psycopg2 connection.
        query:     Raw user query string (for BM25).
        embedding: Query embedding vector (for pgvector).
        k:         Number of final results to return.
        config:    Hybrid search configuration (uses defaults if None).

    Returns:
        List of (chunk_text, rrf_score) tuples, sorted by descending score.
    """
    if config is None:
        config = DEFAULT_CONFIG

    candidate_k = k * config.candidate_multiplier
    cur = conn.cursor()

    # Run both search arms
    vector_results = _vector_search(cur, embedding, candidate_k, config.vector_min_similarity)
    bm25_results = _bm25_search(cur, query, candidate_k)
    cur.close()

    # If BM25 returned nothing, optionally fall back to pure vector
    if not bm25_results and config.fallback_to_vector:
        return [(text, sim) for (_id, text, sim) in vector_results[:k]]

    # If both are empty, nothing to fuse
    if not vector_results and not bm25_results:
        return []

    # Build ID → content lookup
    content_map: dict[int, str] = {}
    for doc_id, text, _ in vector_results:
        content_map[doc_id] = text
    for doc_id, text, _ in bm25_results:
        content_map[doc_id] = text

    # Build ranked lists for RRF
    vector_ranked = [(doc_id, score) for doc_id, _text, score in vector_results]
    bm25_ranked = [(doc_id, score) for doc_id, _text, score in bm25_results]

    ranked_lists = [vector_ranked, bm25_ranked]
    weights = [config.vector_weight, config.bm25_weight]

    # Fuse
    fused = reciprocal_rank_fusion(ranked_lists, weights, k=config.rrf_k)

    # Map back to content
    results: list[tuple[str, float]] = []
    for doc_id, rrf_score in fused[:k]:
        text = content_map.get(doc_id)
        if text is not None:
            results.append((text, round(rrf_score, 6)))

    logger.info(
        "Hybrid search: %d vector + %d BM25 → %d fused (k=%d)",
        len(vector_results), len(bm25_results), len(results), k,
    )
    return results


def hybrid_search_texts(
    conn,
    query: str,
    embedding,
    k: int = 4,
    config: HybridConfig | None = None,
) -> list[str]:
    """Convenience wrapper returning only chunk texts."""
    return [text for text, _score in hybrid_search(conn, query, embedding, k, config)]
