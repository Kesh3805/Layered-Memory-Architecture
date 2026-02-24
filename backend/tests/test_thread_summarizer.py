"""Tests for the thread summarizer module."""

import pytest
from unittest.mock import MagicMock, patch

from thread_summarizer import (
    summarize_thread,
    generate_thread_label,
    maybe_summarize,
    THREAD_SUMMARY_PROMPT,
    THREAD_LABEL_PROMPT,
)


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARIZE THREAD
# ═══════════════════════════════════════════════════════════════════════════

class TestSummarizeThread:
    """Thread summarization with injected completion_fn."""

    def test_empty_messages_returns_previous(self):
        assert summarize_thread("t1", [], previous_summary="old") == "old"

    def test_empty_messages_no_previous(self):
        assert summarize_thread("t1", []) == ""

    @patch("query_db.update_thread_summary")
    def test_generates_summary(self, mock_update):
        mock_fn = MagicMock(return_value="  A concise summary.  ")
        messages = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "Retrieval-Augmented Generation."},
        ]
        result = summarize_thread("t1", messages, completion_fn=mock_fn)
        assert result == "A concise summary."
        mock_fn.assert_called_once()
        mock_update.assert_called_once_with("t1", "A concise summary.")

    @patch("query_db.update_thread_summary")
    def test_includes_previous_summary(self, mock_update):
        mock_fn = MagicMock(return_value="Updated summary.")
        messages = [{"role": "user", "content": "Tell me more."}]
        summarize_thread("t1", messages, previous_summary="Prior info.", completion_fn=mock_fn)

        call_args = mock_fn.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Prior info." in prompt_content

    @patch("query_db.update_thread_summary")
    def test_truncates_long_messages(self, mock_update):
        mock_fn = MagicMock(return_value="Summary.")
        long_msg = {"role": "user", "content": "x" * 1000}
        summarize_thread("t1", [long_msg], completion_fn=mock_fn)
        call_args = mock_fn.call_args
        prompt_msgs = call_args[1]["messages"]
        content = prompt_msgs[0]["content"]
        # Message content should be truncated to 300 chars
        assert "x" * 301 not in content

    def test_llm_failure_returns_previous(self):
        mock_fn = MagicMock(side_effect=RuntimeError("LLM down"))
        result = summarize_thread("t1", [{"role": "user", "content": "hi"}],
                                  previous_summary="fallback", completion_fn=mock_fn)
        assert result == "fallback"

    def test_llm_failure_no_previous(self):
        mock_fn = MagicMock(side_effect=RuntimeError("LLM down"))
        result = summarize_thread("t1", [{"role": "user", "content": "hi"}],
                                  completion_fn=mock_fn)
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATE THREAD LABEL
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateThreadLabel:
    """Label generation with injected completion_fn."""

    def test_empty_messages(self):
        assert generate_thread_label("t1", []) == ""

    @patch("query_db.update_thread_label")
    def test_generates_label(self, mock_update):
        mock_fn = MagicMock(return_value='"RAG Architecture Design"')
        messages = [{"role": "user", "content": "Let's discuss RAG architecture."}]
        result = generate_thread_label("t1", messages, completion_fn=mock_fn)
        assert result == "RAG Architecture Design"
        mock_update.assert_called_once()

    @patch("query_db.update_thread_label")
    def test_strips_quotes(self, mock_update):
        mock_fn = MagicMock(return_value="'Database Indexing'")
        result = generate_thread_label("t1", [{"role": "user", "content": "test"}], completion_fn=mock_fn)
        assert result == "Database Indexing"

    def test_llm_failure_returns_empty(self):
        mock_fn = MagicMock(side_effect=RuntimeError("fail"))
        result = generate_thread_label("t1", [{"role": "user", "content": "test"}], completion_fn=mock_fn)
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
#  MAYBE SUMMARIZE
# ═══════════════════════════════════════════════════════════════════════════

class TestMaybeSummarize:
    """Interval checking + conditional summarization."""

    @patch("topic_threading.should_summarize_thread", return_value=False)
    def test_skips_when_not_due(self, mock_should):
        result = maybe_summarize("t1", "c1")
        assert result is None

    @patch("query_db.get_thread", return_value=None)
    @patch("topic_threading.should_summarize_thread", return_value=True)
    def test_skips_when_thread_not_found(self, mock_should, mock_get):
        result = maybe_summarize("t1", "c1")
        assert result is None

    @patch("query_db.get_thread", return_value={"message_ids": [], "summary": ""})
    @patch("topic_threading.should_summarize_thread", return_value=True)
    def test_skips_when_no_message_ids(self, mock_should, mock_get):
        result = maybe_summarize("t1", "c1")
        assert result is None

    @patch("query_db.update_thread_label")
    @patch("query_db.update_thread_summary")
    @patch("query_db.get_conversation_messages")
    @patch("query_db.get_thread")
    @patch("topic_threading.should_summarize_thread", return_value=True)
    def test_summarizes_and_labels_new_thread(self, mock_should, mock_get_thread,
                                               mock_get_msgs, mock_update_sum, mock_update_label):
        mock_get_thread.return_value = {
            "message_ids": ["m1", "m2"],
            "summary": "",
            "label": None,
        }
        mock_get_msgs.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        mock_completion = MagicMock(return_value="New summary.")

        result = maybe_summarize("t1", "c1", completion_fn=mock_completion)
        assert result == "New summary."
        # Label should also be generated since label is None
        assert mock_completion.call_count == 2  # summary + label

    @patch("query_db.update_thread_summary")
    @patch("query_db.get_conversation_messages")
    @patch("query_db.get_thread")
    @patch("topic_threading.should_summarize_thread", return_value=True)
    def test_summarizes_without_label_when_exists(self, mock_should, mock_get_thread,
                                                    mock_get_msgs, mock_update_sum):
        mock_get_thread.return_value = {
            "message_ids": ["m1"],
            "summary": "Old summary.",
            "label": "Existing Label",
        }
        mock_get_msgs.return_value = [
            {"role": "user", "content": "More details please"},
        ]
        mock_completion = MagicMock(return_value="Updated summary.")

        result = maybe_summarize("t1", "c1", completion_fn=mock_completion)
        assert result == "Updated summary."
        assert mock_completion.call_count == 1  # only summary, no label


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPTS SANITY
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptTemplates:
    """Ensure prompt templates have the expected placeholders."""

    def test_summary_prompt_has_placeholders(self):
        assert "{messages}" in THREAD_SUMMARY_PROMPT
        assert "{previous_summary_section}" in THREAD_SUMMARY_PROMPT

    def test_label_prompt_has_placeholder(self):
        assert "{messages}" in THREAD_LABEL_PROMPT
