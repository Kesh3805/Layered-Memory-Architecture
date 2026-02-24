"""Text chunker — semantic paragraph → sentence → character splitting.

Single source of truth for chunking logic.  Used by:
  - ``main.py``  (startup auto-indexing)
  - ``cli.py``   (``python cli.py ingest``)
  - ``tests/test_chunker.py``

Strategy
--------
1. Split on blank lines to get natural paragraphs.
2. Paragraphs that fit within ``chunk_size`` are kept whole.
3. Paragraphs larger than ``chunk_size`` are sentence-split first.
4. Sentences (or fragments) larger than ``chunk_size`` fall back to
   character windows — the same legacy behaviour, but only as a last
   resort rather than the primary path.
5. Consecutive small chunks are merged up to ``chunk_size`` so we
   avoid many tiny fragments.
"""

from __future__ import annotations

import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split *text* into semantically meaningful chunks.

    Args:
        text:          Source text to chunk.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    size = chunk_size
    overlap = chunk_overlap

    # ── 1. Paragraph split ────────────────────────────────────────
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # ── 2. Flatten each paragraph into sentence-level atoms ───────
    atoms: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            atoms.append(para)
        else:
            # Sentence split on .  !  ?  followed by whitespace
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            for sent in sentences:
                if len(sent) <= size:
                    atoms.append(sent)
                else:
                    # Character-window fallback for very long sentences
                    stride = max(1, size - overlap)
                    for i in range(0, len(sent), stride):
                        fragment = sent[i : i + size]
                        if fragment.strip():
                            atoms.append(fragment)

    # ── 3. Merge small atoms into chunks up to `size` ─────────────
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def _flush():
        if current_parts:
            chunks.append(" ".join(current_parts))

    for atom in atoms:
        atom_len = len(atom)
        # +1 for the space separator
        if current_parts and current_len + 1 + atom_len > size:
            _flush()
            # Start next chunk with overlap: carry the tail of the last chunk
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
