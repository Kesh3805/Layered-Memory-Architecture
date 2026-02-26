"""Tests for search_similar_insights (dynamic WHERE clause) and /insights/search endpoint."""

import types
import sys
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  search_similar_insights — WHERE clause construction
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchSimilarInsights:
    """Verify dynamic WHERE clause and parameter ordering in search_similar_insights."""

    def _make_mock_cursor(self, rows=None):
        cur = MagicMock()
        cur.fetchall.return_value = rows or []
        return cur

    def _make_mock_conn(self, cursor):
        conn = MagicMock()
        conn.cursor.return_value = cursor
        return conn

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_no_filters(self, mock_get_conn, mock_put_conn):
        """No conversation_id or insight_type — only base WHERE clause."""
        from query_db import search_similar_insights

        cur = self._make_mock_cursor()
        conn = self._make_mock_conn(cur)
        mock_get_conn.return_value = conn

        emb = [0.1] * 768
        search_similar_insights(emb, k=5)

        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]

        assert "embedding IS NOT NULL" in sql
        # WHERE clause should NOT contain filter predicates
        where_clause = sql.split("WHERE")[1].split("ORDER BY")[0]
        assert "conversation_id =" not in where_clause
        assert "insight_type =" not in where_clause
        # params: [emb, emb, k]  (first for similarity calc, then ORDER BY, LIMIT)
        assert params[-1] == 5  # k

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_conversation_id_filter(self, mock_get_conn, mock_put_conn):
        """With conversation_id — adds conversation_id = %s to WHERE."""
        from query_db import search_similar_insights

        cur = self._make_mock_cursor()
        conn = self._make_mock_conn(cur)
        mock_get_conn.return_value = conn

        emb = [0.1] * 768
        search_similar_insights(emb, k=5, conversation_id="conv-abc")

        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]

        assert "conversation_id = %s" in sql
        assert "conv-abc" in params

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_insight_type_filter(self, mock_get_conn, mock_put_conn):
        """With insight_type — adds insight_type = %s to WHERE."""
        from query_db import search_similar_insights

        cur = self._make_mock_cursor()
        conn = self._make_mock_conn(cur)
        mock_get_conn.return_value = conn

        emb = [0.1] * 768
        search_similar_insights(emb, k=5, insight_type="decision")

        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]

        assert "insight_type = %s" in sql
        assert "decision" in params

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_both_filters(self, mock_get_conn, mock_put_conn):
        """Both conversation_id and insight_type — both in WHERE clause."""
        from query_db import search_similar_insights

        cur = self._make_mock_cursor()
        conn = self._make_mock_conn(cur)
        mock_get_conn.return_value = conn

        emb = [0.1] * 768
        search_similar_insights(emb, k=3, conversation_id="conv-xyz", insight_type="hypothesis")

        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]

        assert "conversation_id = %s" in sql
        assert "insight_type = %s" in sql
        assert "conv-xyz" in params
        assert "hypothesis" in params
        assert params[-1] == 3  # k is last param

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_returns_formatted_dicts(self, mock_get_conn, mock_put_conn):
        """Verify result dict keys from returned rows."""
        from query_db import search_similar_insights

        rows = [
            ("id-1", "conv-1", "thr-1", "decision", "Use Redis", 0.9, 0.85),
        ]
        cur = self._make_mock_cursor(rows)
        conn = self._make_mock_conn(cur)
        mock_get_conn.return_value = conn

        result = search_similar_insights([0.1] * 768, k=5)
        assert len(result) == 1
        assert result[0]["id"] == "id-1"
        assert result[0]["conversation_id"] == "conv-1"
        assert result[0]["thread_id"] == "thr-1"
        assert result[0]["insight_type"] == "decision"
        assert result[0]["insight_text"] == "Use Redis"
        assert result[0]["confidence_score"] == 0.9
        assert result[0]["similarity"] == 0.85

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_handles_numpy_embedding(self, mock_get_conn, mock_put_conn):
        """numpy array embedding is converted to list."""
        from query_db import search_similar_insights

        cur = self._make_mock_cursor()
        conn = self._make_mock_conn(cur)
        mock_get_conn.return_value = conn

        emb = np.array([0.1] * 768)
        search_similar_insights(emb, k=5)

        params = cur.execute.call_args[0][1]
        # The embedding parameter should be a list, not numpy array
        assert isinstance(params[0], list)

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_db_error_returns_empty(self, mock_get_conn, mock_put_conn):
        """Database errors return empty list instead of raising."""
        from query_db import search_similar_insights

        mock_get_conn.side_effect = Exception("Connection refused")

        result = search_similar_insights([0.1] * 768, k=5)
        assert result == []

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_connection_returned_to_pool(self, mock_get_conn, mock_put_conn):
        """Connection is always returned via put_connection (finally block)."""
        from query_db import search_similar_insights

        cur = self._make_mock_cursor()
        conn = self._make_mock_conn(cur)
        mock_get_conn.return_value = conn

        search_similar_insights([0.1] * 768, k=5)
        mock_put_conn.assert_called_once_with(conn)

    @patch("query_db.put_connection")
    @patch("query_db.get_connection")
    def test_none_conversation_id_not_added_to_where(self, mock_get_conn, mock_put_conn):
        """Passing conversation_id=None explicitly should not add the filter."""
        from query_db import search_similar_insights

        cur = self._make_mock_cursor()
        conn = self._make_mock_conn(cur)
        mock_get_conn.return_value = conn

        search_similar_insights([0.1] * 768, k=5, conversation_id=None, insight_type=None)

        sql = cur.execute.call_args[0][0]
        where_clause = sql.split("WHERE")[1].split("ORDER BY")[0]
        assert "conversation_id =" not in where_clause
        assert "insight_type =" not in where_clause


# ═══════════════════════════════════════════════════════════════════════════
#  /insights/search endpoint (function-level test)
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchInsightsEndpoint:
    """Test the search_insights route function logic without full ASGI stack."""

    @patch("main.query_db")
    @patch("main.get_query_embedding")
    @patch("main.DB_ENABLED", True)
    def test_basic_search(self, mock_embed, mock_qdb):
        from main import search_insights

        mock_embed.return_value = [0.1] * 768
        mock_qdb.search_similar_insights.return_value = [
            {"id": "1", "insight_type": "decision", "insight_text": "Use Redis",
             "confidence_score": 0.9, "similarity": 0.85,
             "conversation_id": "c1", "thread_id": "t1"},
        ]

        result = search_insights(q="caching", k=10, type=None, conversation_id=None)
        assert result["count"] == 1
        assert result["results"][0]["insight_text"] == "Use Redis"
        mock_embed.assert_called_once_with("caching")

    @patch("main.query_db")
    @patch("main.get_query_embedding")
    @patch("main.DB_ENABLED", True)
    def test_passes_filters_through(self, mock_embed, mock_qdb):
        from main import search_insights

        mock_embed.return_value = [0.1] * 768
        mock_qdb.search_similar_insights.return_value = []

        search_insights(q="test", k=5, type="hypothesis", conversation_id="conv-abc")
        mock_qdb.search_similar_insights.assert_called_once_with(
            mock_embed.return_value, k=5, conversation_id="conv-abc", insight_type="hypothesis",
        )

    @patch("main.query_db")
    @patch("main.get_query_embedding")
    @patch("main.DB_ENABLED", True)
    def test_empty_results(self, mock_embed, mock_qdb):
        from main import search_insights

        mock_embed.return_value = [0.1] * 768
        mock_qdb.search_similar_insights.return_value = []

        result = search_insights(q="nothing", k=10, type=None, conversation_id=None)
        assert result["count"] == 0
        assert result["results"] == []

    @patch("main.DB_ENABLED", False)
    def test_db_disabled_raises_503(self):
        from main import search_insights
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            search_insights(q="test", k=10, type=None, conversation_id=None)
        assert exc_info.value.status_code == 503
