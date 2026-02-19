"""Intent classification — pre-heuristics + LLM call."""

import json
import logging
from typing import Optional

from .client import completion, MAX_CLASSIFIER_TOKENS
from .prompts import INTENT_PROMPT

logger = logging.getLogger(__name__)

# ── Pre-heuristic signal lists ────────────────────────────────────────────

PRIVACY_SIGNALS = [
    "invasion of privacy", "privacy concern", "what do you know about me",
    "what data do you", "what information do you", "do you store",
    "do you track", "are you tracking", "delete my data", "erase my",
    "forget about me", "what have you stored", "my personal data",
    "data privacy", "how is my data", "who has access", "is my data safe",
    "stop storing", "do you collect", "what do you remember about me",
    "stored about me", "data about me", "what info do you have",
]

CONTINUATION_PRONOUNS = {
    "that", "it", "this", "those", "these", "they", "them",
    "its", "their", "the same",
}

VALID_INTENTS = {"general", "continuation", "knowledge_base", "profile", "privacy"}


def classify_intent(
    user_query: str,
    conversation_context: Optional[list] = None,
) -> dict:
    """Classify user intent using pre-heuristics then LLM fallback.

    Returns ``{"intent": str, "confidence": float}``.
    """
    query_lower = user_query.strip().lower()

    # ── Fast privacy check ────────────────────────────────────────────────
    if any(sig in query_lower for sig in PRIVACY_SIGNALS):
        logger.info("Pre-heuristic: privacy signal detected")
        return {"intent": "privacy", "confidence": 0.95}

    # ── Continuation check (needs context) ────────────────────────────────
    if conversation_context and len(conversation_context) >= 2:
        words = set(query_lower.split())
        if (
            len(words) <= 8
            and words & CONTINUATION_PRONOUNS
            and "?" in user_query
        ):
            logger.info("Pre-heuristic: continuation (pronoun + short question)")
            return {"intent": "continuation", "confidence": 0.85}

    # ── LLM classification ────────────────────────────────────────────────
    try:
        context_text = ""
        if conversation_context:
            recent = conversation_context[-6:]
            lines = [f"{m['role'].title()}: {m['content'][:120]}" for m in recent]
            context_text = "\nRecent conversation:\n" + "\n".join(lines)

        messages = [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": f"Message: {user_query}{context_text}"},
        ]

        response = completion(
            messages,
            temperature=0.0,
            max_tokens=MAX_CLASSIFIER_TOKENS,
        )
        raw = response.choices[0].message.content.strip()

        # Robust JSON extraction
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        result = json.loads(raw)
        intent = result.get("intent", "general")
        confidence = float(result.get("confidence", 0.5))

        if intent not in VALID_INTENTS:
            logger.warning(f"Unknown intent '{intent}' → general")
            intent = "general"

        return {"intent": intent, "confidence": confidence}

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"intent": "general", "confidence": 0.5}
