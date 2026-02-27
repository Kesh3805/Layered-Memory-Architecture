"""Behavioral Router — the intelligence layer between classification and retrieval.

This is the module that turns an intent classifier + RAG pipeline into
something that *feels* like a conversation partner.  It sits between
intent classification (Step 3) and policy resolve (Step 5) and answers
a fundamentally different question:

    Intent classifier:  "What TYPE of message is this?"
    Behavior engine:    "What EXPERIENCE should the user get?"

Key behaviors detected and routed:
  - Greeting loops        → skip retrieval, acknowledge naturally
  - Repetition patterns   → acknowledge the pattern, don't re-retrieve
  - Testing / probing     → meta-aware, honest, playful
  - Tone shifts           → adapt personality mode
  - Low-entropy inputs    → don't waste retrieval budget
  - Exploratory behavior  → broaden context window
  - Frustration           → empathetic mode, more careful responses

Public API:
    BehaviorDecision    — what the engine decided
    BehaviorEngine      — the router itself
    evaluate()          — (state, query, intent) → BehaviorDecision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from conversation_state import ConversationState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  BEHAVIOR DECISION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BehaviorDecision:
    """Output of the behavior engine — tells the pipeline HOW to behave.

    This is NOT the same as PolicyDecision (which controls WHAT to retrieve).
    BehaviorDecision modulates the *experience*: tone, retrieval necessity,
    prompt framing, and meta-awareness.
    """

    # ── Behavior mode ─────────────────────────────────────────────────────
    behavior_mode: str = "standard"
    """Primary behavior mode. One of:
    - standard:           Normal RAG pipeline
    - greeting:           Lightweight social response, minimal/no retrieval
    - repetition_aware:   Acknowledge pattern, vary response
    - testing_aware:      Meta-honest, engage with curiosity
    - meta_aware:         User commenting on the AI nature
    - frustration_recovery: Extra careful, empathetic framing
    - rapid_fire:         Concise, direct answers
    - exploratory:        Broader context, more diverse retrieval
    """

    # ── Retrieval modulation ──────────────────────────────────────────────
    skip_retrieval: bool = False
    """If True, skip RAG / QA retrieval entirely (e.g. greetings, testing)."""

    reduce_retrieval: bool = False
    """If True, reduce retrieval volume (fewer docs, higher similarity floor)."""

    boost_retrieval: bool = False
    """If True, increase retrieval volume (more docs, lower floor)."""

    rag_k_override: int | None = None
    """Override default rag_k if set."""

    rag_min_similarity_override: float | None = None
    """Override default rag_min_similarity if set."""

    # ── Response modulation ───────────────────────────────────────────────
    personality_mode: str = "default"
    """Personality mode for prompt framing: default | concise | detailed | playful | empathetic."""
    precision_mode: str = "analytical"
    """Research precision mode: concise | analytical | speculative | implementation | adversarial."""
    response_length_hint: str = "normal"
    """Suggested response length: brief | normal | detailed."""

    # ── Context hints for prompt assembly ─────────────────────────────────
    behavior_context: str = ""
    """Extra context string injected into the system prompt to guide behavior."""

    meta_instruction: str = ""
    """Specific instruction override for special behavioral modes."""

    # ── Diagnosis ─────────────────────────────────────────────────────────
    triggers: list[str] = field(default_factory=list)
    """List of trigger labels that caused this decision (for logging/debug)."""


# ═══════════════════════════════════════════════════════════════════════════
#  BEHAVIOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class BehaviorEngine:
    """Behavioral router — decides experience-level modulation.

    Pure function: no side effects, no DB access.

    Usage::

        decision = BehaviorEngine.evaluate(state, query, intent, confidence)
    """

    @staticmethod
    def evaluate(
        state: ConversationState,
        query: str,
        intent: str,
        confidence: float,
    ) -> BehaviorDecision:
        """Evaluate behavioral signals and produce a routing decision.

        The engine runs detectors in priority order. Multiple triggers
        can fire, but the highest-priority behavior mode wins.

        Priority (highest → lowest):
          1. Frustration recovery
          2. Testing / meta awareness
          3. Repetition awareness
          4. Greeting loop
          5. Rapid-fire mode
          6. Exploratory mode
          7. Standard
        """
        d = BehaviorDecision()
        q_lower = query.strip().lower()
        words = q_lower.split()
        word_count = len(words)

        # Set precision mode from conversation state (carried through all paths)
        d.precision_mode = getattr(state, "precision_mode", "analytical")

        # ── 1. Frustration recovery ───────────────────────────────────────
        if state.emotional_tone == "frustrated":
            d.triggers.append("frustrated_tone")
            d.behavior_mode = "frustration_recovery"
            d.personality_mode = "empathetic"
            d.response_length_hint = "detailed"
            d.behavior_context = (
                "The user seems frustrated. Be extra careful, acknowledge their "
                "concern, and provide a thorough, accurate response. If your "
                "previous answer was wrong, explicitly acknowledge that."
            )

            # If repetition + frustration → they're stuck, broaden retrieval
            if state.repetition_count >= 2:
                d.boost_retrieval = True
                d.rag_k_override = 6
                d.rag_min_similarity_override = 0.3
                d.triggers.append("frustration_repetition_boost")

            return d

        # ── 2. Testing / meta awareness ───────────────────────────────────
        if state.testing_flag or state.interaction_pattern == "testing":
            d.triggers.append("testing_behavior")
            d.behavior_mode = "testing_aware"
            d.personality_mode = "playful"
            d.skip_retrieval = True
            d.response_length_hint = "normal"
            d.behavior_context = (
                "The user is testing or probing the system. Be honest and "
                "self-aware. Acknowledge your nature as an AI assistant with "
                "good humor. Don't be defensive — be curious and engaging."
            )
            return d

        if state.meta_comment_count >= 2 and not state.testing_flag:
            d.triggers.append("meta_commentary")
            d.behavior_mode = "meta_aware"
            d.personality_mode = "playful"
            d.reduce_retrieval = True
            d.behavior_context = (
                "The user has made comments about your AI nature. Be "
                "self-aware and honest, but steer back toward being useful."
            )
            return d

        # ── 3. Repetition awareness ──────────────────────────────────────
        if state.repetition_count >= 3:
            d.triggers.append("high_repetition")
            d.behavior_mode = "repetition_aware"
            d.personality_mode = "empathetic"
            d.reduce_retrieval = True
            d.behavior_context = (
                "The user has asked similar questions multiple times. "
                "Acknowledge this pattern gently, try a different angle "
                "or framing, and ask if you can help clarify something specific."
            )
            d.meta_instruction = (
                "Don't just repeat your previous answer. Notice the pattern "
                "and try a fresh approach or ask what specifically isn't clear."
            )
            return d

        if state.repetition_count >= 2:
            d.triggers.append("mild_repetition")
            d.behavior_context = (
                "The user seems to be revisiting a similar topic. Vary your "
                "response angle slightly while staying helpful."
            )

        # ── 4. Greeting loop detection ────────────────────────────────────
        if intent == "general" and _is_low_entropy(q_lower, word_count):
            # Check if this is a greeting or trivial social input
            if state.message_count >= 2 and state.intent_streak >= 2:
                d.triggers.append("greeting_loop")
                d.behavior_mode = "greeting"
                d.skip_retrieval = True
                d.personality_mode = "playful"
                d.response_length_hint = "brief"
                d.behavior_context = (
                    "This is a repeated greeting or very short social message. "
                    "Be warm but brief. Don't repeat the same greeting pattern."
                )
                return d

            if _is_greeting_like(q_lower):
                d.triggers.append("simple_greeting")
                d.behavior_mode = "greeting"
                d.skip_retrieval = True
                d.response_length_hint = "brief"
                d.behavior_context = (
                    "This is a greeting. Be warm and natural — don't start "
                    "retrieval or provide unsolicited information."
                )
                return d

        # ── 5. Rapid-fire mode ────────────────────────────────────────────
        if state.interaction_pattern == "rapid_fire":
            d.triggers.append("rapid_fire_pattern")
            d.behavior_mode = "rapid_fire"
            d.personality_mode = "concise"
            d.response_length_hint = "brief"
            d.reduce_retrieval = True
            d.rag_k_override = 2
            d.behavior_context = (
                "The user is sending rapid, short messages. Match their "
                "pace — be concise and direct. Skip preambles."
            )
            return d

        # ── 6. Exploratory mode ──────────────────────────────────────────
        if state.interaction_pattern == "exploratory":
            d.triggers.append("exploratory_pattern")
            d.behavior_mode = "exploratory"
            d.personality_mode = "detailed"
            d.response_length_hint = "detailed"
            d.boost_retrieval = True
            d.rag_k_override = 6
            d.rag_min_similarity_override = 0.35
            d.behavior_context = (
                "The user is exploring broadly. Provide rich, varied context "
                "and connect ideas across topics when relevant."
            )
            return d

        # ── 7. Tone-based modulation (overlays, not modes) ───────────────
        if state.emotional_tone == "playful":
            d.triggers.append("playful_tone")
            d.personality_mode = "playful"

        if state.emotional_tone == "curious":
            d.triggers.append("curious_tone")
            d.personality_mode = "detailed"
            d.response_length_hint = "detailed"

        if state.emotional_tone == "positive":
            d.triggers.append("positive_tone")
            # Keep default but note the good vibe
            d.behavior_context = (
                "The conversation has a positive tone. "
                "Maintain the warm, productive atmosphere."
            )

        # ── Default: standard mode ────────────────────────────────────────
        if not d.triggers:
            d.triggers.append("standard")
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

_GREETING_WORDS = {
    "hello", "hi", "hey", "howdy", "sup", "yo", "hola", "greetings",
    "morning", "afternoon", "evening", "night",
}

_SOCIAL_FILLERS = {
    "how are you", "what's up", "whats up", "how's it going",
    "how are things", "what's new", "how do you do",
    "nice to meet you", "good to see you",
}


def _is_greeting_like(q_lower: str) -> bool:
    """Check if input is a greeting or trivial social exchange."""
    words = set(q_lower.strip("!?,. ").split())
    # Single greeting word
    if len(words) <= 2 and words & _GREETING_WORDS:
        return True
    # Social filler phrases
    if any(filler in q_lower for filler in _SOCIAL_FILLERS):
        return True
    return False


def _is_low_entropy(q_lower: str, word_count: int) -> bool:
    """Check if input is low-information (greeting, social, single word)."""
    if word_count <= 3:
        return True
    if _is_greeting_like(q_lower):
        return True
    # Very short, no question words, no technical terms
    _info_words = {
        "what", "how", "why", "where", "when", "which", "explain",
        "describe", "compare", "analyze", "implement", "create", "build",
        "code", "function", "class", "error", "bug", "fix", "help",
    }
    words = set(q_lower.split())
    if word_count <= 5 and not (words & _info_words):
        return True
    return False
