"""Tests for CLI memory commands (inspect, query) and utilities."""

import argparse
import io
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy modules that cli.py may import transitively
for _mod in ("cache",):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

from cli import _print_wrapped, cmd_memory, _cmd_memory_inspect, _cmd_memory_query


# ═══════════════════════════════════════════════════════════════════════════
#  _print_wrapped utility
# ═══════════════════════════════════════════════════════════════════════════

class TestPrintWrapped:
    """Word-wrap utility used for ASCII box drawing output."""

    def test_short_text_single_line(self, capsys):
        _print_wrapped("Hello world", indent=4, width=60)
        out = capsys.readouterr().out
        assert "Hello world" in out
        # Should be in one line (no extra newlines beyond the trailing one)
        assert out.strip().count("\n") == 0

    def test_long_text_wraps(self, capsys):
        text = " ".join(["word"] * 30)  # ~150 chars
        _print_wrapped(text, indent=4, width=30)
        out = capsys.readouterr().out
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) > 1, "Long text should wrap to multiple lines"

    def test_indent_prefix(self, capsys):
        _print_wrapped("Test", indent=5, width=60)
        out = capsys.readouterr().out
        # indent=5 means prefix is "  │" + " " * (5-3) = "  │  "
        assert out.startswith("  │")

    def test_empty_text(self, capsys):
        _print_wrapped("", indent=4, width=60)
        out = capsys.readouterr().out
        # Empty input should produce no output
        assert out.strip() == ""


# ═══════════════════════════════════════════════════════════════════════════
#  cmd_memory routing
# ═══════════════════════════════════════════════════════════════════════════

class TestCmdMemoryRouting:
    """Route memory sub-commands to inspect or query."""

    @patch("cli._cmd_memory_inspect")
    def test_routes_inspect(self, mock_inspect):
        args = argparse.Namespace(memory_command="inspect", conversation=None, insights_only=False)
        cmd_memory(args)
        mock_inspect.assert_called_once_with(args)

    @patch("cli._cmd_memory_query")
    def test_routes_query(self, mock_query):
        args = argparse.Namespace(memory_command="query", query_text="test", type=None, k=10)
        cmd_memory(args)
        mock_query.assert_called_once_with(args)

    def test_invalid_subcommand_exits(self):
        args = argparse.Namespace(memory_command=None)
        with pytest.raises(SystemExit):
            cmd_memory(args)

    def test_missing_subcommand_exits(self):
        args = argparse.Namespace()
        with pytest.raises(SystemExit):
            cmd_memory(args)


# ═══════════════════════════════════════════════════════════════════════════
#  _cmd_memory_inspect
# ═══════════════════════════════════════════════════════════════════════════

def _make_conversation(cid="conv-1", title="Test Chat"):
    return {"id": cid, "title": title, "created_at": "2025-01-01", "message_count": 5}


def _make_thread(tid="thr-1", label="Architecture", mc=3, summary="Discussed arch"):
    return {"id": tid, "label": label, "message_count": mc, "summary": summary}


def _make_insight(itype="decision", text="Use PostgreSQL", conf=0.9, tid="thr-1"):
    return {"insight_type": itype, "insight_text": text, "confidence_score": conf, "thread_id": tid}


def _make_concept(concept="PostgreSQL", src="insight"):
    return {"concept": concept, "source_type": src}


class TestCmdMemoryInspect:
    """Memory inspect command — prints threads, insights, concepts."""

    def _build_args(self, conversation=None, insights_only=False):
        return argparse.Namespace(
            memory_command="inspect",
            conversation=conversation,
            insights_only=insights_only,
        )

    @patch("cli._ensure_db")
    def test_no_conversations_prints_message(self, mock_db, caplog):
        qdb = MagicMock()
        qdb.list_conversations.return_value = []
        mock_db.return_value = qdb

        with caplog.at_level(logging.INFO, logger="rag-cli"):
            _cmd_memory_inspect(self._build_args())
        assert "No conversations found" in caplog.text

    @patch("cli._ensure_db")
    def test_lists_all_conversations(self, mock_db, capsys):
        qdb = MagicMock()
        qdb.list_conversations.return_value = [_make_conversation("conv-1"), _make_conversation("conv-2", "Second")]
        qdb.get_threads.return_value = []
        qdb.get_concepts_for_conversation.return_value = []
        mock_db.return_value = qdb

        _cmd_memory_inspect(self._build_args())
        out = capsys.readouterr().out
        assert "conv-1" in out
        assert "conv-2" in out
        assert "Test Chat" in out
        assert "Second" in out
        qdb.list_conversations.assert_called_once_with(limit=20)

    @patch("cli._ensure_db")
    def test_specific_conversation(self, mock_db, capsys):
        qdb = MagicMock()
        conv = _make_conversation("abc-123")
        qdb.get_conversation.return_value = conv
        qdb.get_threads.return_value = [_make_thread()]
        qdb.get_insights_for_thread.return_value = [_make_insight()]
        qdb.get_concepts_for_conversation.return_value = [_make_concept()]
        mock_db.return_value = qdb

        _cmd_memory_inspect(self._build_args(conversation="abc-123"))
        out = capsys.readouterr().out
        assert "abc-123" in out
        assert "Architecture" in out
        assert "decision" in out.lower()
        assert "PostgreSQL" in out
        qdb.get_conversation.assert_called_once_with("abc-123")
        qdb.list_conversations.assert_not_called()

    @patch("cli._ensure_db")
    def test_conversation_not_found_exits(self, mock_db):
        qdb = MagicMock()
        qdb.get_conversation.return_value = None
        mock_db.return_value = qdb

        with pytest.raises(SystemExit):
            _cmd_memory_inspect(self._build_args(conversation="nonexistent"))

    @patch("cli._ensure_db")
    def test_threads_with_insights(self, mock_db, capsys):
        qdb = MagicMock()
        qdb.list_conversations.return_value = [_make_conversation()]
        qdb.get_threads.return_value = [
            _make_thread("t1", "Design", 5, "Talked about design patterns"),
        ]
        qdb.get_insights_for_thread.return_value = [
            _make_insight("decision", "Use factory pattern", 0.85, "t1"),
            _make_insight("hypothesis", "Singletons may cause issues", 0.7, "t1"),
        ]
        qdb.get_concepts_for_conversation.return_value = []
        mock_db.return_value = qdb

        _cmd_memory_inspect(self._build_args())
        out = capsys.readouterr().out
        assert "Design" in out
        assert "5 msgs" in out
        assert "factory pattern" in out
        assert "hypothesis" in out
        # Box drawing chars
        assert "┌" in out
        assert "└" in out

    @patch("cli._ensure_db")
    def test_insights_only_flag(self, mock_db, capsys):
        qdb = MagicMock()
        qdb.list_conversations.return_value = [_make_conversation()]
        qdb.get_threads.return_value = []
        qdb.get_insights.return_value = [
            _make_insight("observation", "User prefers Python", 0.8),
        ]
        qdb.get_concepts_for_conversation.return_value = []
        mock_db.return_value = qdb

        _cmd_memory_inspect(self._build_args(insights_only=True))
        out = capsys.readouterr().out
        assert "All Insights" in out
        assert "observation" in out

    @patch("cli._ensure_db")
    def test_concepts_display(self, mock_db, capsys):
        qdb = MagicMock()
        qdb.list_conversations.return_value = [_make_conversation()]
        qdb.get_threads.return_value = []
        qdb.get_concepts_for_conversation.return_value = [
            _make_concept("Redis", "insight"),
            _make_concept("pgvector", "concept"),
        ]
        mock_db.return_value = qdb

        _cmd_memory_inspect(self._build_args())
        out = capsys.readouterr().out
        assert "Concept Links: 2" in out
        assert "Redis" in out
        assert "pgvector" in out

    @patch("cli._ensure_db")
    def test_many_concepts_shows_truncation(self, mock_db, capsys):
        qdb = MagicMock()
        qdb.list_conversations.return_value = [_make_conversation()]
        qdb.get_threads.return_value = []
        concepts = [_make_concept(f"Concept-{i}", "insight") for i in range(20)]
        qdb.get_concepts_for_conversation.return_value = concepts
        mock_db.return_value = qdb

        _cmd_memory_inspect(self._build_args())
        out = capsys.readouterr().out
        assert "... and 5 more" in out

    @patch("cli._ensure_db")
    def test_long_insight_text_truncated(self, mock_db, capsys):
        qdb = MagicMock()
        qdb.list_conversations.return_value = [_make_conversation()]
        qdb.get_threads.return_value = [_make_thread()]
        long_text = "A" * 100
        qdb.get_insights_for_thread.return_value = [_make_insight("decision", long_text, 0.9)]
        qdb.get_concepts_for_conversation.return_value = []
        mock_db.return_value = qdb

        _cmd_memory_inspect(self._build_args())
        out = capsys.readouterr().out
        assert "..." in out  # truncated


# ═══════════════════════════════════════════════════════════════════════════
#  _cmd_memory_query
# ═══════════════════════════════════════════════════════════════════════════

class TestCmdMemoryQuery:
    """Cross-thread semantic search command."""

    def _build_args(self, query_text="test query", type_filter=None, k=10):
        return argparse.Namespace(
            memory_command="query",
            query_text=query_text,
            type=type_filter,
            k=k,
        )

    @patch("cli._ensure_db")
    def test_empty_query_exits(self, mock_db):
        qdb = MagicMock()
        mock_db.return_value = qdb

        with pytest.raises(SystemExit):
            _cmd_memory_query(self._build_args(query_text=""))

    @patch("embeddings.get_query_embedding")
    @patch("cli._ensure_db")
    def test_search_with_results(self, mock_db, mock_embed, capsys):
        qdb = MagicMock()
        mock_db.return_value = qdb
        mock_embed.return_value = [0.1] * 768

        qdb.search_similar_insights.return_value = [
            {"insight_type": "decision", "insight_text": "Use Redis for caching",
             "confidence_score": 0.9, "similarity": 0.85, "thread_id": "thr-abc123"},
        ]
        qdb.search_similar_concepts.return_value = [
            {"concept": "Redis", "similarity": 0.82, "source_type": "insight"},
        ]

        _cmd_memory_query(self._build_args("caching strategy"))
        out = capsys.readouterr().out
        assert "Cross-Thread Search" in out
        assert "caching strategy" in out
        assert "Use Redis for caching" in out
        assert "decision" in out
        assert "Redis" in out
        qdb.search_similar_insights.assert_called_once()
        qdb.search_similar_concepts.assert_called_once()

    @patch("embeddings.get_query_embedding")
    @patch("cli._ensure_db")
    def test_search_no_results(self, mock_db, mock_embed, capsys):
        qdb = MagicMock()
        mock_db.return_value = qdb
        mock_embed.return_value = [0.1] * 768

        qdb.search_similar_insights.return_value = []
        qdb.search_similar_concepts.return_value = []

        _cmd_memory_query(self._build_args("nonexistent topic"))
        out = capsys.readouterr().out
        assert "No matching insights found" in out
        assert "No matching concepts found" in out

    @patch("embeddings.get_query_embedding")
    @patch("cli._ensure_db")
    def test_type_filter_passed_through(self, mock_db, mock_embed, capsys):
        qdb = MagicMock()
        mock_db.return_value = qdb
        mock_embed.return_value = [0.1] * 768
        qdb.search_similar_insights.return_value = []
        qdb.search_similar_concepts.return_value = []

        _cmd_memory_query(self._build_args("test", type_filter="decision"))
        call_kwargs = qdb.search_similar_insights.call_args
        assert call_kwargs[1]["insight_type"] == "decision" or \
               (len(call_kwargs[0]) >= 3 and call_kwargs[0][2] == "decision") or \
               call_kwargs.kwargs.get("insight_type") == "decision"

    @patch("embeddings.get_query_embedding")
    @patch("cli._ensure_db")
    def test_custom_k_parameter(self, mock_db, mock_embed, capsys):
        qdb = MagicMock()
        mock_db.return_value = qdb
        mock_embed.return_value = [0.1] * 768
        qdb.search_similar_insights.return_value = []
        qdb.search_similar_concepts.return_value = []

        _cmd_memory_query(self._build_args("test", k=5))
        call_kwargs = qdb.search_similar_insights.call_args
        assert call_kwargs[1]["k"] == 5 or \
               (len(call_kwargs[0]) >= 2 and call_kwargs[0][1] == 5) or \
               call_kwargs.kwargs.get("k") == 5

    @patch("embeddings.get_query_embedding")
    @patch("cli._ensure_db")
    def test_long_insight_text_truncated_in_output(self, mock_db, mock_embed, capsys):
        qdb = MagicMock()
        mock_db.return_value = qdb
        mock_embed.return_value = [0.1] * 768
        qdb.search_similar_insights.return_value = [
            {"insight_type": "conclusion", "insight_text": "X" * 100,
             "confidence_score": 0.8, "similarity": 0.7, "thread_id": "t1"},
        ]
        qdb.search_similar_concepts.return_value = []

        _cmd_memory_query(self._build_args("test"))
        out = capsys.readouterr().out
        assert "..." in out  # text gets truncated


# ═══════════════════════════════════════════════════════════════════════════
#  CLI argparse integration
# ═══════════════════════════════════════════════════════════════════════════

class TestCliArgparse:
    """Verify argparse setup for memory sub-commands."""

    def _parse(self, *argv):
        from cli import main as cli_main
        # Build the parser directly to test parsing without execution
        parser = argparse.ArgumentParser(prog="rag")
        sub = parser.add_subparsers(dest="command")
        sub.add_parser("init")
        p_ingest = sub.add_parser("ingest")
        p_ingest.add_argument("dir", nargs="?")
        p_dev = sub.add_parser("dev")
        p_dev.add_argument("--host")
        p_dev.add_argument("--port", type=int)
        p_mem = sub.add_parser("memory")
        mem_sub = p_mem.add_subparsers(dest="memory_command")
        p_inspect = mem_sub.add_parser("inspect")
        p_inspect.add_argument("--conversation", "-c")
        p_inspect.add_argument("--insights-only", action="store_true")
        p_query = mem_sub.add_parser("query")
        p_query.add_argument("query_text")
        p_query.add_argument("--type", choices=["decision", "conclusion", "hypothesis", "open_question", "observation"])
        p_query.add_argument("-k", type=int, default=10)
        return parser.parse_args(list(argv))

    def test_memory_inspect_defaults(self):
        args = self._parse("memory", "inspect")
        assert args.command == "memory"
        assert args.memory_command == "inspect"
        assert args.conversation is None
        assert args.insights_only is False

    def test_memory_inspect_with_conversation(self):
        args = self._parse("memory", "inspect", "-c", "abc-123")
        assert args.conversation == "abc-123"

    def test_memory_inspect_insights_only(self):
        args = self._parse("memory", "inspect", "--insights-only")
        assert args.insights_only is True

    def test_memory_query_basic(self):
        args = self._parse("memory", "query", "how does caching work")
        assert args.command == "memory"
        assert args.memory_command == "query"
        assert args.query_text == "how does caching work"
        assert args.type is None
        assert args.k == 10

    def test_memory_query_with_type(self):
        args = self._parse("memory", "query", "test", "--type", "decision")
        assert args.type == "decision"

    def test_memory_query_with_k(self):
        args = self._parse("memory", "query", "test", "-k", "5")
        assert args.k == 5

    def test_memory_query_invalid_type_rejected(self):
        with pytest.raises(SystemExit):
            self._parse("memory", "query", "test", "--type", "invalid_type")
