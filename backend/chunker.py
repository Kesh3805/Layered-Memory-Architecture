"""Text chunker — semantic (embedding-based) with rule-based fallback.

Single source of truth for chunking logic.  Used by:
  - ``main.py``  (startup auto-indexing)
  - ``cli.py``   (``python cli.py ingest``)
  - ``tests/test_chunker.py``

Primary strategy (SEMANTIC_CHUNKING_ENABLED=True)
-------------------------------------------------
1. Split text into sentences.
2. Embed every sentence with the same model used for retrieval.
3. Compute cosine distance between consecutive sentence embeddings.
4. Identify breakpoints where the distance exceeds the Nth percentile
   of all pairwise distances (default N=95 → top-5% semantic jumps).
5. Collapse the resulting sentence groups into chunks.  Groups that
   still exceed ``chunk_size`` are sub-split with the rule-based fallback.

Fallback strategy (fewer than 3 sentences, or SEMANTIC_CHUNKING_ENABLED=False)
-------------------------------------------------------------------------------
1. Split on blank lines → natural paragraphs (kept whole if ≤ chunk_size).
2. Paragraphs over ``chunk_size`` → sentence-split.
3. Sentences over ``chunk_size`` → sliding character windows as last resort.
4. Consecutive small atoms are merged up to ``chunk_size``.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# Minimum sentences required to run the semantic path.
# Texts shorter than this fall straight through to rule-based.
_SEMANTIC_MIN_SENTENCES = 3


# ── Pluggable embedding hook (monkeypatched in tests) ─────────────────────

def _embed(sentences: list[str]) -> np.ndarray:
    """Return an (N, dim) float32 embedding matrix for *sentences*.

    Imported lazily so the model is not loaded at module import time,
    and so tests can replace this function without side-effects.
    """
    from embeddings import get_embeddings  # noqa: PLC0415
    return get_embeddings(sentences)


# ── Sentence splitter ─────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into individual sentences, stripping blank lines first."""
    # Collapse multiple blank lines then flatten into one sentence stream.
    flat = re.sub(r"\n{2,}", " ", text.strip())
    return [s.strip() for s in _SENTENCE_RE.split(flat) if s.strip()]


# ── Rule-based fallback ───────────────────────────────────────────────────

def _rule_based_chunk(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph → sentence → character cascading chunker (no embeddings)."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    atoms: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            atoms.append(para)
        else:
            sentences = [s.strip() for s in _SENTENCE_RE.split(para) if s.strip()]
            for sent in sentences:
                if len(sent) <= size:
                    atoms.append(sent)
                else:
                    stride = max(1, size - overlap)
                    for i in range(0, len(sent), stride):
                        fragment = sent[i : i + size]
                        if fragment.strip():
                            atoms.append(fragment)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def _flush() -> None:
        if current_parts:
            chunks.append(" ".join(current_parts))

    for atom in atoms:
        atom_len = len(atom)
        if current_parts and current_len + 1 + atom_len > size:
            _flush()
            if overlap > 0 and current_parts:
                tail = " ".join(current_parts)[-overlap:]
                current_parts = [tail]
                current_len = len(tail)
            else:
                current_parts = []
                current_len = 0
        current_parts.append(atom)
        current_len += (1 + atom_len) if len(current_parts) > 1 else atom_len

    _flush()
    return [c for c in chunks if c.strip()]


# ── Semantic chunker ──────────────────────────────────────────────────────

def _semantic_chunk(
    sentences: list[str],
    chunk_size: int,
    chunk_overlap: int,
    breakpoint_percentile: int,
) -> list[str]:
    """Group *sentences* by embedding similarity, then enforce size caps."""
    embeddings = _embed(sentences)  # (N, dim)

    # Normalise rows to unit length for cosine distance.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-9)

    # Cosine distance between consecutive sentences.
    similarities = np.sum(normed[:-1] * normed[1:], axis=1)  # (N-1,)
    distances = 1.0 - similarities

    # Breakpoint = distance above the Nth percentile.
    threshold = float(np.percentile(distances, breakpoint_percentile))
    breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

    # Build sentence groups from breakpoints.
    groups: list[list[str]] = []
    prev = 0
    for bp in breakpoints:
        groups.append(sentences[prev:bp])
        prev = bp
    groups.append(sentences[prev:])

    # Convert groups → chunks, respecting chunk_size.
    chunks: list[str] = []
    for group in groups:
        combined = " ".join(group)
        if len(combined) <= chunk_size:
            chunks.append(combined)
        else:
            # Sub-split oversized groups with rule-based approach.
            chunks.extend(_rule_based_chunk(combined, chunk_size, chunk_overlap))

    return [c for c in chunks if c.strip()]


# ── Public API ────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split *text* into chunks using semantic (embedding-based) grouping.

    Falls back to rule-based splitting when:
    - ``SEMANTIC_CHUNKING_ENABLED`` is False in settings, or
    - the text contains fewer than 3 sentences, or
    - the embedding model is unavailable.

    Args:
        text:          Source text to chunk.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Character overlap for the rule-based fallback.

    Returns:
        List of non-empty chunk strings.
    """
    if not text or not text.strip():
        return []

    # Check settings — import lazily to avoid circular deps at module load.
    try:
        from settings import settings  # noqa: PLC0415
        semantic_enabled = settings.SEMANTIC_CHUNKING_ENABLED
        percentile = settings.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE
    except Exception:
        semantic_enabled = True
        percentile = 95

    if not semantic_enabled:
        return _rule_based_chunk(text, chunk_size, chunk_overlap)

    sentences = _split_sentences(text)

    if len(sentences) < _SEMANTIC_MIN_SENTENCES:
        # Too few sentences for meaningful embedding comparison.
        return _rule_based_chunk(text, chunk_size, chunk_overlap)

    try:
        return _semantic_chunk(sentences, chunk_size, chunk_overlap, percentile)
    except Exception as exc:
        logger.warning("Semantic chunking failed (%s) — falling back to rule-based.", exc)
        return _rule_based_chunk(text, chunk_size, chunk_overlap)
