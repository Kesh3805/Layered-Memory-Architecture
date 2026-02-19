"""Behavior Policy Engine — deterministic context-injection rules.

Instead of intent alone determining what gets injected into the LLM
context, the policy engine computes *context features* and applies
deterministic rules to decide:

* What context to inject  (profile, RAG, Q&A, history)
* How to frame it  (privacy mode, greeting personalization)
* Which retrieval route to label  (for metadata / debugging)

This separates BEHAVIOR from MODEL-CALLING code.  When behavior is wrong
(e.g. greeting doesn't use the user's name), you fix a rule here — you
never edit prompt strings or generator functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL LISTS
# ═══════════════════════════════════════════════════════════════════════════

GREETING_PATTERNS = [
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "good night", "howdy", "sup", "what's up",
    "whats up", "yo", "hola", "greetings", "hi there",
    "hey there", "hello there",
]

PERSONAL_REF_SIGNALS = [
    "my job", "my role", "my name", "my work", "my career",
    "my background", "my preference", "my language", "my stack",
    "my project", "my company", "what do i do", "what am i",
    "who am i", "i work as", "my position", "my skills",
    "my experience", "for my job", "in my job", "on my job",
    "at my job", "my profession", "my title", "whats my",
    "what's my", "tell me my", "what is my", "my weight",
    "my height", "my age", "my location", "my timezone",
    "my degree", "my education",
]

PROFILE_STATEMENT_PREFIXES = (
    "i am ", "i'm ", "i have ", "i like ", "i prefer ", "i use ",
    "i work ", "i live ", "i weigh ", "i speak ", "i study ",
    "i graduated ", "i code ", "i built ", "i've ", "i am from ",
    "call me ", "my name is ", "remember that ", "i am a ", "i'm a ",
)


# ═══════════════════════════════════════════════════════════════════════════
#  CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContextFeatures:
    """Computed features for the current message — input to policy rules."""

    is_greeting: bool = False
    references_profile: bool = False
    privacy_signal: bool = False
    is_followup: bool = False
    is_profile_statement: bool = False
    is_profile_question: bool = False
    topic_similarity: Optional[float] = None
    has_profile_data: bool = False
    profile_name: Optional[str] = None
    conversation_length: int = 0


def extract_context_features(
    query: str,
    intent: str,
    profile_entries: list[dict] | None = None,
    conversation_length: int = 0,
    topic_similarity: float | None = None,
) -> ContextFeatures:
    """Compute deterministic context features from current state."""
    q = query.strip().lower()
    words = q.split()

    # ── Greeting detection ────────────────────────────────────────────────
    is_greeting = False
    if len(words) <= 8:
        for pat in GREETING_PATTERNS:
            if q == pat or q.startswith(pat + " ") or q.startswith(pat + ",") or q.startswith(pat + "!"):
                is_greeting = True
                break

    # ── Personal-reference detection ──────────────────────────────────────
    references_profile = any(sig in q for sig in PERSONAL_REF_SIGNALS)

    # ── Profile statement vs question ─────────────────────────────────────
    is_profile_statement = (
        q.startswith(PROFILE_STATEMENT_PREFIXES) and "?" not in query
    )
    is_profile_question = intent == "profile" and not is_profile_statement

    # ── Profile data check ────────────────────────────────────────────────
    has_profile_data = bool(profile_entries)
    profile_name: str | None = None
    if profile_entries:
        for entry in profile_entries:
            if entry.get("key") in ("name", "first_name", "full_name", "username"):
                profile_name = entry.get("value")
                break

    return ContextFeatures(
        is_greeting=is_greeting,
        references_profile=references_profile,
        privacy_signal=(intent == "privacy"),
        is_followup=(intent == "continuation"),
        is_profile_statement=is_profile_statement,
        is_profile_question=is_profile_question,
        topic_similarity=topic_similarity,
        has_profile_data=has_profile_data,
        profile_name=profile_name,
        conversation_length=conversation_length,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  POLICY DECISION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyDecision:
    """What the pipeline should do — determined by rules, not prompts."""

    inject_profile: bool = False
    inject_rag: bool = False
    inject_qa_history: bool = False
    use_curated_history: bool = True
    privacy_mode: bool = False
    greeting_name: str | None = None       # if set → personalise greeting
    retrieval_route: str = "llm_only"      # label for metadata
    rag_k: int = 4
    qa_k: int = 4
    qa_min_similarity: float = 0.65


# ═══════════════════════════════════════════════════════════════════════════
#  BEHAVIOR POLICY
# ═══════════════════════════════════════════════════════════════════════════

class BehaviorPolicy:
    """Deterministic policy engine — rules, not prompt rewrites.

    Usage::

        features = extract_context_features(query, intent, profile_entries, …)
        decision = BehaviorPolicy().resolve(features, intent)

    The returned :class:`PolicyDecision` tells the pipeline what to
    retrieve, what to inject, and how to frame the response.
    """

    def resolve(self, features: ContextFeatures, intent: str) -> PolicyDecision:
        d = PolicyDecision()

        # ── Privacy always takes highest priority ─────────────────────────
        if intent == "privacy":
            d.privacy_mode = True
            d.inject_profile = features.has_profile_data
            d.use_curated_history = False
            d.retrieval_route = "privacy"
            return d

        # ── Profile intent ────────────────────────────────────────────────
        if intent == "profile":
            if features.is_profile_statement:
                d.retrieval_route = "profile_update"
                d.use_curated_history = False
            else:
                d.inject_profile = features.has_profile_data
                d.retrieval_route = "profile"
            return d

        # ── Knowledge base ────────────────────────────────────────────────
        if intent == "knowledge_base":
            d.inject_rag = True
            d.inject_qa_history = True
            d.retrieval_route = "rag"
            return d

        # ── Continuation ──────────────────────────────────────────────────
        if intent == "continuation":
            d.inject_qa_history = True
            d.retrieval_route = "conversation"
            return d

        # ── General (default) ─────────────────────────────────────────────
        d.use_curated_history = False
        d.retrieval_route = "llm_only"

        # ── Cross-intent overlay policies ─────────────────────────────────
        # These fire regardless of the base intent computed above.

        # Greeting + known name → personalise
        if features.is_greeting and features.profile_name:
            d.greeting_name = features.profile_name

        # Personal reference in non-profile intents → inject profile
        if (
            features.references_profile
            and features.has_profile_data
            and intent not in ("profile", "privacy")
        ):
            d.inject_profile = True

        return d
