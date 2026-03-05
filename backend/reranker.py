"""Cross-encoder reranker — re-score retrieved chunks for higher precision.

Why reranking
-------------
Bi-encoder retrieval (vector search) trades precision for speed: the query
and document are encoded independently, so subtle semantic mismatches slip
through.  A cross-encoder processes (query, document) as a single pair,
capturing fine-grained relevance at the cost of being ~100× slower per doc.

The standard pattern: retrieve broadly (k=12–20), rerank to top-k (4–6).

Architecture
------------
1. ``sentence-transformers`` ``CrossEncoder`` model loaded lazily.
2. Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — 22M params,
   ~5ms per pair on CPU, state-of-the-art for its size class.
3. Scores are logits (not probabilities) — higher = more relevant.
4. Falls back gracefully: if no cross-encoder is available, returns
   the input list unchanged (no-op passthrough).

Public API
----------
    rerank(query, documents, top_k=4) → list[(text, score)]
    rerank_with_metadata(query, docs_with_meta, top_k=4) → list[(text, meta, score)]

Settings
--------
    RERANKER_ENABLED:  bool  (default: True)
    RERANKER_MODEL:    str   (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_K:    int   (default: same as RETRIEVAL_K)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_cross_encoder = None
_load_failed = False


def _get_cross_encoder():
    """Lazy-load and cache the CrossEncoder model.

    If loading fails (missing dependency, OOM, etc.), sets _load_failed
    so subsequent calls skip the attempt and fall through to passthrough.
    """
    global _cross_encoder, _load_failed

    if _load_failed:
        return None

    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from settings import settings
        if not settings.RERANKER_ENABLED:
            logger.info("Reranker disabled via RERANKER_ENABLED=false")
            _load_failed = True
            return None

        from sentence_transformers import CrossEncoder

        model_name = settings.RERANKER_MODEL
        logger.info("Loading cross-encoder reranker: %s", model_name)
        _cross_encoder = CrossEncoder(model_name)
        logger.info("Reranker loaded successfully: %s", model_name)
        return _cross_encoder

    except ImportError:
        logger.warning(
            "CrossEncoder not available (sentence-transformers may be too old). "
            "Reranking disabled — falling through to retrieval order."
        )
        _load_failed = True
        return None

    except Exception as e:
        logger.warning("Failed to load cross-encoder reranker: %s", e)
        _load_failed = True
        return None


def rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Re-score documents against a query using a cross-encoder.

    Args:
        query:     The user's search query.
        documents: List of document chunk texts from initial retrieval.
        top_k:     How many to keep after reranking (None = use RETRIEVAL_K).

    Returns:
        List of (text, relevance_score) sorted by descending score,
        truncated to top_k.  If the reranker is unavailable, returns
        documents in their original order with score=0.0.
    """
    if not documents:
        return []

    if top_k is None:
        from settings import settings
        top_k = settings.RERANKER_TOP_K or settings.RETRIEVAL_K

    encoder = _get_cross_encoder()
    if encoder is None:
        # Passthrough — preserve original retrieval order
        return [(doc, 0.0) for doc in documents[:top_k]]

    # Build (query, document) pairs
    pairs = [(query, doc) for doc in documents]

    # Score all pairs in one batch
    scores = encoder.predict(pairs)

    # Zip, sort by score descending, truncate
    scored = sorted(
        zip(documents, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    results = [(text, round(float(score), 4)) for text, score in scored[:top_k]]

    logger.info(
        "Reranker: %d candidates → top %d (best=%.3f, worst=%.3f)",
        len(documents),
        len(results),
        results[0][1] if results else 0.0,
        results[-1][1] if results else 0.0,
    )
    return results


def rerank_with_metadata(
    query: str,
    docs_with_meta: list[tuple[str, Any]],
    top_k: int | None = None,
) -> list[tuple[str, Any, float]]:
    """Rerank documents that carry metadata (scores, IDs, etc.).

    Args:
        query:          The user's search query.
        docs_with_meta: List of (text, metadata) tuples from retrieval.
        top_k:          How many to keep.

    Returns:
        List of (text, metadata, rerank_score) sorted by descending score.
    """
    if not docs_with_meta:
        return []

    if top_k is None:
        from settings import settings
        top_k = settings.RERANKER_TOP_K or settings.RETRIEVAL_K

    texts = [text for text, _meta in docs_with_meta]
    metas = [meta for _text, meta in docs_with_meta]

    encoder = _get_cross_encoder()
    if encoder is None:
        return [(text, meta, 0.0) for text, meta in docs_with_meta[:top_k]]

    pairs = [(query, text) for text in texts]
    scores = encoder.predict(pairs)

    scored = sorted(
        zip(texts, metas, scores),
        key=lambda x: float(x[2]),
        reverse=True,
    )

    results = [(text, meta, round(float(score), 4)) for text, meta, score in scored[:top_k]]

    logger.info(
        "Reranker (with meta): %d → top %d (best=%.3f)",
        len(docs_with_meta), len(results),
        results[0][2] if results else 0.0,
    )
    return results


def is_available() -> bool:
    """Check whether the reranker is loaded and usable."""
    return _get_cross_encoder() is not None
