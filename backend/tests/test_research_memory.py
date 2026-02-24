"""Tests for the research memory module."""

import pytest

from research_memory import (
    INSIGHT_TYPES,
    extract_concepts,
    _parse_insights_json,
)


# ═══════════════════════════════════════════════════════════════════════════
#  CONCEPT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractConcepts:
    """Noun-phrase concept extraction (heuristic, no LLM)."""

    def test_empty_input(self):
        assert extract_concepts("") == []
        assert extract_concepts("   ") == []
        assert extract_concepts("hi") == []  # too short

    def test_extracts_capitalized_terms(self):
        text = "We should use Redis for the Database Layer."
        concepts = extract_concepts(text)
        # "Redis" and "Database Layer" are capitalized noun phrases
        assert any("Redis" in c for c in concepts)

    def test_extracts_technical_terms(self):
        text = "The get_embedding function uses sentence_transformers internally."
        concepts = extract_concepts(text)
        assert any("get_embedding" in c for c in concepts)
        assert any("sentence_transformers" in c for c in concepts)

    def test_extracts_camel_case(self):
        text = "The BehaviorEngine evaluates conversational state."
        concepts = extract_concepts(text)
        assert any("BehaviorEngine" in c for c in concepts)

    def test_extracts_quoted_terms(self):
        text = 'The user said "cosine similarity" is important.'
        concepts = extract_concepts(text)
        assert any("cosine similarity" in c for c in concepts)

    def test_extracts_backtick_terms(self):
        text = "Use `vector_store.search()` for retrieval."
        concepts = extract_concepts(text)
        assert any("vector_store.search()" in c for c in concepts)

    def test_extracts_acronyms(self):
        text = "We use RAG with HNSW indexes on the DB."
        concepts = extract_concepts(text)
        assert any("RAG" in c for c in concepts)
        assert any("HNSW" in c for c in concepts)

    def test_ignores_stop_words(self):
        text = "The the the is is is are are are."
        concepts = extract_concepts(text)
        assert len(concepts) == 0

    def test_deduplication(self):
        text = "PostgreSQL PostgreSQL PostgreSQL is great."
        concepts = extract_concepts(text)
        pg_count = sum(1 for c in concepts if "PostgreSQL" in c)
        assert pg_count <= 1  # deduplicated


# ═══════════════════════════════════════════════════════════════════════════
#  INSIGHT JSON PARSING
# ═══════════════════════════════════════════════════════════════════════════

class TestParseInsightsJson:
    """JSON parsing with fallbacks for LLM output."""

    def test_valid_json_array(self):
        raw = '[{"type": "decision", "text": "Use PostgreSQL", "confidence": 0.9}]'
        result = _parse_insights_json(raw)
        assert len(result) == 1
        assert result[0]["type"] == "decision"

    def test_empty_array(self):
        assert _parse_insights_json("[]") == []

    def test_empty_string(self):
        assert _parse_insights_json("") == []

    def test_markdown_code_fence(self):
        raw = '```json\n[{"type": "conclusion", "text": "X is faster", "confidence": 0.8}]\n```'
        result = _parse_insights_json(raw)
        assert len(result) == 1
        assert result[0]["type"] == "conclusion"

    def test_non_array_returns_empty(self):
        assert _parse_insights_json('{"type": "decision"}') == []

    def test_invalid_json_returns_empty(self):
        assert _parse_insights_json("not json at all") == []

    def test_embedded_json_in_text(self):
        raw = 'Here are the insights:\n[{"type": "hypothesis", "text": "If X then Y", "confidence": 0.7}]\nDone.'
        result = _parse_insights_json(raw)
        assert len(result) == 1

    def test_multiple_insights(self):
        raw = '''[
            {"type": "decision", "text": "Use Redis for caching", "confidence": 0.9},
            {"type": "open_question", "text": "How to handle failover?", "confidence": 0.8}
        ]'''
        result = _parse_insights_json(raw)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  INSIGHT TYPES
# ═══════════════════════════════════════════════════════════════════════════

class TestInsightTypes:
    """Verify insight type constants."""

    def test_all_types_present(self):
        assert "decision" in INSIGHT_TYPES
        assert "conclusion" in INSIGHT_TYPES
        assert "hypothesis" in INSIGHT_TYPES
        assert "open_question" in INSIGHT_TYPES
        assert "observation" in INSIGHT_TYPES

    def test_types_is_set(self):
        assert isinstance(INSIGHT_TYPES, set)
