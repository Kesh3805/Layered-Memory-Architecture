"""Thread Summarizer — per-thread progressive summarization.

Instead of summarizing the entire conversation, this module generates
and maintains per-thread summaries that progressively compress as
threads grow.

Key design:
  - Summaries are updated at THREAD_SUMMARY_INTERVAL milestones (8, 16, 24…)
  - New summaries incorporate the previous summary + recent messages
  - Summaries are stored in the conversation_threads table
  - Labels are auto-generated on first summary

Public API:
    summarize_thread()     — generate/update a thread summary
    generate_thread_label() — produce a short human-readable label
    maybe_summarize()      — check interval + summarize if needed
"""

from __future__ import annotations

import logging

from settings import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD SUMMARY PROMPT
# ═══════════════════════════════════════════════════════════════════════════

THREAD_SUMMARY_PROMPT = """\
Summarize this conversation thread for an ongoing research session.
Focus on:
- Key findings, decisions, and conclusions
- Open questions or unresolved points
- Technical details worth remembering
- The direction the discussion is heading

{previous_summary_section}

Recent messages in this thread:
{messages}

Write a concise summary (3-6 sentences). Focus on substance, not pleasantries.
Return only the summary text — no headers, no bullet points, no metadata."""

THREAD_LABEL_PROMPT = """\
Generate a short label (3-6 words) for this conversation thread.
The label should capture the core topic or question being discussed.

Messages:
{messages}

Return ONLY the label text. No quotes, no punctuation, no explanation."""


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARIZATION
# ═══════════════════════════════════════════════════════════════════════════

def summarize_thread(
    thread_id: str,
    messages: list[dict],
    previous_summary: str = "",
    completion_fn=None,
) -> str:
    """Generate or update a thread summary.

    Args:
        thread_id:        The thread to summarize.
        messages:         Recent messages in the thread [{role, content}].
        previous_summary: Existing summary to build upon (empty for first).
        completion_fn:    LLM completion function (injected for testability).

    Returns:
        The new summary text or empty string on failure.
    """
    if not messages:
        return previous_summary or ""

    if completion_fn is None:
        from llm.client import completion
        completion_fn = completion

    # Format messages for the prompt
    msg_text = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: "
        f"{m.get('content', '')[:300]}"
        for m in messages[-12:]  # last 12 messages max
    )

    prev_section = ""
    if previous_summary:
        prev_section = f"Previous summary of this thread:\n{previous_summary}\n"

    prompt = THREAD_SUMMARY_PROMPT.format(
        previous_summary_section=prev_section,
        messages=msg_text,
    )

    try:
        summary = completion_fn(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        summary = summary.strip()

        # Store the summary
        import query_db
        query_db.update_thread_summary(thread_id, summary)

        logger.info(f"Updated summary for thread {thread_id[:8]}… ({len(summary)} chars)")
        return summary

    except Exception as e:
        logger.error(f"Thread summarization failed: {e}")
        return previous_summary or ""


def generate_thread_label(
    thread_id: str,
    messages: list[dict],
    completion_fn=None,
) -> str:
    """Generate a short human-readable label for a thread.

    Called once when a thread first reaches the summary interval.
    """
    if not messages:
        return ""

    if completion_fn is None:
        from llm.client import completion
        completion_fn = completion

    msg_text = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: "
        f"{m.get('content', '')[:200]}"
        for m in messages[:6]  # first 6 messages (captures initial topic)
    )

    prompt = THREAD_LABEL_PROMPT.format(messages=msg_text)

    try:
        label = completion_fn(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )
        label = label.strip().strip('"').strip("'")

        import query_db
        query_db.update_thread_label(thread_id, label)

        logger.info(f"Thread {thread_id[:8]}… labeled: {label}")
        return label

    except Exception as e:
        logger.error(f"Thread label generation failed: {e}")
        return ""


def maybe_summarize(
    thread_id: str,
    conversation_id: str,
    completion_fn=None,
) -> str | None:
    """Check if a thread needs summarization and do it if so.

    Returns the new summary if generated, None otherwise.
    """
    import query_db
    from topic_threading import should_summarize_thread

    if not should_summarize_thread(thread_id):
        return None

    thread = query_db.get_thread(thread_id)
    if not thread:
        return None

    # Get recent messages for this thread
    message_ids = thread.get("message_ids", [])
    if not message_ids:
        return None

    # Load actual message content from the conversation
    all_msgs = query_db.get_conversation_messages(conversation_id, limit=200)
    # Filter to messages in this thread (by message_id if tracked)
    # Fallback: use last N messages from conversation
    recent = all_msgs[-12:] if all_msgs else []

    previous_summary = thread.get("summary", "")

    summary = summarize_thread(
        thread_id=thread_id,
        messages=recent,
        previous_summary=previous_summary,
        completion_fn=completion_fn,
    )

    # Generate label if thread doesn't have one yet
    if not thread.get("label") and recent:
        generate_thread_label(thread_id, recent, completion_fn)

    return summary
