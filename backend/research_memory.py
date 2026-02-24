"""Research Memory — insight extraction + concept linking.

This module extracts structured knowledge units from conversations:

  - **Insights**: Decisions, conclusions, hypotheses, open questions,
    observations extracted from Q&A pairs via LLM analysis.
  - **Concepts**: Noun-phrase-level topics extracted cheaply (regex/heuristic),
    then embedded and linked across threads.

Memory tier D in the architecture:
  A) Episodic   — user_queries         ✅
  B) Semantic   — user_profile          ✅
  C) Conversational — conversation_state ✅
  D) Research   — research_insights + concept_links  ← THIS MODULE

Public API:
    extract_insights()   — LLM-powered insight extraction from a Q&A pair
    extract_concepts()   — cheap noun-phrase extraction (no LLM)
    link_concepts()      — embed concepts and store links
    get_research_context() — gather relevant insights + concepts for prompt injection
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import numpy as np

from settings import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  INSIGHT TYPES
# ═══════════════════════════════════════════════════════════════════════════

INSIGHT_TYPES = {
    "decision",       # "We chose X over Y because…"
    "conclusion",     # "Therefore, X is the case"
    "hypothesis",     # "If X, then Y might…"
    "open_question",  # "Still unclear whether…"
    "observation",    # "Note: X appears to do Y"
}


# ═══════════════════════════════════════════════════════════════════════════
#  INSIGHT EXTRACTION (LLM-powered)
# ═══════════════════════════════════════════════════════════════════════════

INSIGHT_EXTRACTION_PROMPT = """\
Analyze this Q&A exchange and extract research-relevant insights.
Return a JSON array of insights. Each insight:
  {"type": "<type>", "text": "<concise insight>", "confidence": <0.0-1.0>}

Types:
- decision: A choice or preference was stated ("chose X over Y")
- conclusion: A definitive finding or assertion was established
- hypothesis: A speculative or conditional claim ("if X then Y")
- open_question: Something explicitly left unresolved
- observation: A noteworthy factual observation

Rules:
1. Only extract insights that are clearly stated or strongly implied.
2. Keep each insight text concise (1-2 sentences max).
3. If NO research insights exist, return an empty array: []
4. Return raw JSON only — no markdown fences, no commentary.

Q: {query}
A: {response}
"""


def extract_insights(
    query: str,
    response: str,
    conversation_id: str,
    thread_id: str | None = None,
    source_message_id: str | None = None,
    *,
    db_enabled: bool = True,
) -> list[dict]:
    """Extract research insights from a Q&A pair using LLM analysis.

    Stores extracted insights in the DB and returns them.
    """
    if not settings.RESEARCH_INSIGHTS_ENABLED or not db_enabled:
        return []

    # Skip very short exchanges (unlikely to contain insights)
    if len(query.split()) < 5 and len(response.split()) < 20:
        return []

    try:
        from llm.client import completion

        prompt = INSIGHT_EXTRACTION_PROMPT.format(
            query=query[:500],
            response=response[:1000],
        )

        raw = completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
        )

        insights = _parse_insights_json(raw)
        if not insights:
            return []

        # Filter by confidence threshold
        min_conf = settings.RESEARCH_INSIGHT_MIN_CONFIDENCE
        insights = [i for i in insights if i.get("confidence", 0) >= min_conf]

        # Store in DB
        import query_db
        from embeddings import get_embedding

        stored = []
        for insight in insights:
            text = insight.get("text", "")
            itype = insight.get("type", "observation")
            conf = insight.get("confidence", 0.8)

            if itype not in INSIGHT_TYPES:
                itype = "observation"

            # Embed the insight for semantic search
            try:
                emb = get_embedding(text)
            except Exception:
                emb = None

            row_id = query_db.create_insight(
                conversation_id=conversation_id,
                insight_type=itype,
                insight_text=text,
                embedding=emb,
                thread_id=thread_id,
                confidence_score=conf,
                source_message_id=source_message_id,
            )

            if row_id:
                stored.append({
                    "id": row_id,
                    "type": itype,
                    "text": text,
                    "confidence": conf,
                })

        if stored:
            logger.info(f"Extracted {len(stored)} insights for conv={conversation_id[:8]}…")

        return stored

    except Exception as e:
        logger.error(f"Insight extraction failed: {e}")
        return []


def _parse_insights_json(raw: str) -> list[dict]:
    """Parse LLM output as JSON array of insights, with fallbacks."""
    if not raw:
        return []

    raw = raw.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        # Try to extract JSON array from mixed content
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  CONCEPT EXTRACTION (heuristic — no LLM needed)
# ═══════════════════════════════════════════════════════════════════════════

# Common stop words to exclude from concept extraction
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "should", "now", "also", "but", "and", "or", "if", "this", "that",
    "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "thing", "things", "way", "about",
    "like", "think", "know", "want", "see", "come", "make", "get", "go",
    "up", "one", "two", "said", "say", "use", "using", "use",
}

# Pattern for extracting potential noun phrases (simple heuristic)
_NOUN_PHRASE_RE = re.compile(
    r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'  # Capitalized sequences
    r'|'
    r'\b[a-z]+(?:[-_][a-z]+)+\b'                 # Hyphenated/underscored terms
    r'|'
    r'\b[A-Z]{2,}\b'                              # Acronyms
)

# Technical terms pattern (common in research conversations)
_TECH_TERM_RE = re.compile(
    r'\b(?:'
    r'[a-zA-Z]+(?:_[a-zA-Z]+)+'  # snake_case terms
    r'|[a-zA-Z]+(?:[A-Z][a-z]+)+'  # camelCase terms
    r'|[A-Z][a-z]+(?:[A-Z][a-z]+)+'  # PascalCase terms
    r')\b'
)


def extract_concepts(text: str, min_length: int = 3) -> list[str]:
    """Extract concept noun phrases from text using heuristics.

    This is intentionally cheap (no LLM, no NLP library) — it runs on
    every message. Precision over recall: better to miss a concept than
    to link garbage.

    Returns deduplicated list of concept strings.
    """
    if not text or len(text) < 10:
        return []

    concepts = set()

    # 1. Capitalized noun phrases (names, proper nouns, titles)
    for match in _NOUN_PHRASE_RE.finditer(text):
        term = match.group().strip()
        if len(term) >= min_length and term.lower() not in _STOP_WORDS:
            concepts.add(term)

    # 2. Technical terms (snake_case, camelCase, PascalCase)
    for match in _TECH_TERM_RE.finditer(text):
        term = match.group().strip()
        if len(term) >= min_length:
            concepts.add(term)

    # 3. Quoted terms (users often quote important concepts)
    for match in re.finditer(r'"([^"]{3,50})"', text):
        term = match.group(1).strip()
        if term.lower() not in _STOP_WORDS:
            concepts.add(term)

    # 4. Backtick-quoted code terms
    for match in re.finditer(r'`([^`]{2,40})`', text):
        term = match.group(1).strip()
        concepts.add(term)

    return sorted(concepts)


# ═══════════════════════════════════════════════════════════════════════════
#  CONCEPT LINKING (embed + store cross-thread links)
# ═══════════════════════════════════════════════════════════════════════════

def link_concepts(
    concepts: list[str],
    source_type: str,
    source_id: str,
    conversation_id: str,
    thread_id: str | None = None,
    *,
    db_enabled: bool = True,
) -> int:
    """Embed concepts and store them as concept links.

    Returns the number of concepts successfully linked.
    """
    if not settings.CONCEPT_LINKING_ENABLED or not concepts or not db_enabled:
        return 0

    try:
        import query_db
        from embeddings import get_embeddings

        # Batch embed all concepts
        embeddings = get_embeddings(concepts)

        count = 0
        for concept, emb in zip(concepts, embeddings):
            row_id = query_db.create_concept_link(
                concept=concept,
                embedding=emb,
                source_type=source_type,
                source_id=source_id,
                conversation_id=conversation_id,
                thread_id=thread_id,
            )
            if row_id:
                count += 1

        if count:
            logger.info(f"Linked {count} concepts for {source_type}={source_id[:8]}…")
        return count

    except Exception as e:
        logger.error(f"Concept linking failed: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════
#  RESEARCH CONTEXT ASSEMBLY (for prompt injection)
# ═══════════════════════════════════════════════════════════════════════════

def get_research_context(
    conversation_id: str,
    query_embedding: np.ndarray,
    thread_id: str | None = None,
    *,
    db_enabled: bool = True,
) -> dict:
    """Gather relevant research context for prompt injection.

    Returns a dict with:
      - related_insights: semantically similar insights
      - concept_links: related concepts across threads
    """
    if not db_enabled:
        return {"related_insights": [], "concept_links": []}

    try:
        import query_db

        # Find semantically similar insights
        related_insights = []
        if settings.RESEARCH_INSIGHTS_ENABLED:
            similar = query_db.search_similar_insights(
                query_embedding,
                k=5,
                conversation_id=conversation_id,
            )
            related_insights = [
                {
                    "type": i["insight_type"],
                    "text": i["insight_text"],
                    "similarity": round(i.get("similarity", 0), 3),
                }
                for i in similar
                if i.get("similarity", 0) >= 0.4
            ]

        # Find related concepts
        concept_links = []
        if settings.CONCEPT_LINKING_ENABLED:
            concepts = query_db.search_similar_concepts(
                query_embedding,
                k=settings.CONCEPT_LINK_K,
                conversation_id=conversation_id,
            )
            concept_links = [
                {
                    "concept": c["concept"],
                    "similarity": round(c.get("similarity", 0), 3),
                    "thread_id": c.get("thread_id"),
                }
                for c in concepts
                if c.get("similarity", 0) >= 0.4
            ]

        return {
            "related_insights": related_insights,
            "concept_links": concept_links,
        }

    except Exception as e:
        logger.error(f"Error gathering research context: {e}")
        return {"related_insights": [], "concept_links": []}
