"""Tests for behavior_engine.py — behavioral routing layer."""

from conversation_state import ConversationState
from behavior_engine import BehaviorEngine, BehaviorDecision


class TestBehaviorDecision:
    """BehaviorDecision defaults."""

    def test_default_decision(self):
        d = BehaviorDecision()
        assert d.behavior_mode == "standard"
        assert d.skip_retrieval is False
        assert d.reduce_retrieval is False
        assert d.boost_retrieval is False
        assert d.personality_mode == "default"
        assert d.response_length_hint == "normal"
        assert d.triggers == []


class TestBehaviorEngine:
    """BehaviorEngine.evaluate() tests."""

    def test_standard_mode(self):
        """Fresh conversation, normal query → standard."""
        state = ConversationState(message_count=1)
        d = BehaviorEngine.evaluate(state, "what is python", "knowledge_base", 0.9)
        assert d.behavior_mode == "standard"
        assert d.skip_retrieval is False
        assert "standard" in d.triggers

    def test_frustration_recovery(self):
        """Frustrated tone → frustration_recovery mode."""
        state = ConversationState(
            emotional_tone="frustrated",
            message_count=3,
        )
        d = BehaviorEngine.evaluate(state, "that's still wrong", "general", 0.7)
        assert d.behavior_mode == "frustration_recovery"
        assert d.personality_mode == "empathetic"
        assert d.response_length_hint == "detailed"
        assert "frustrated_tone" in d.triggers

    def test_frustration_with_repetition_boosts_retrieval(self):
        """Frustrated + repeating → boost retrieval to help them."""
        state = ConversationState(
            emotional_tone="frustrated",
            repetition_count=3,
            message_count=5,
        )
        d = BehaviorEngine.evaluate(state, "try again", "general", 0.5)
        assert d.behavior_mode == "frustration_recovery"
        assert d.boost_retrieval is True
        assert d.rag_k_override == 6
        assert "frustration_repetition_boost" in d.triggers

    def test_testing_aware(self):
        """Testing flag → testing_aware mode, skip retrieval."""
        state = ConversationState(
            testing_flag=True,
            interaction_pattern="testing",
            message_count=2,
        )
        d = BehaviorEngine.evaluate(state, "are you really an AI", "general", 0.5)
        assert d.behavior_mode == "testing_aware"
        assert d.skip_retrieval is True
        assert d.personality_mode == "playful"
        assert "testing_behavior" in d.triggers

    def test_meta_aware(self):
        """Multiple meta comments → meta_aware mode."""
        state = ConversationState(
            meta_comment_count=3,
            testing_flag=False,
            message_count=5,
        )
        d = BehaviorEngine.evaluate(state, "you're just software", "general", 0.5)
        assert d.behavior_mode == "meta_aware"
        assert d.reduce_retrieval is True
        assert "meta_commentary" in d.triggers

    def test_high_repetition(self):
        """High repetition count → repetition_aware mode."""
        state = ConversationState(
            repetition_count=3,
            message_count=6,
        )
        d = BehaviorEngine.evaluate(state, "what is python again", "knowledge_base", 0.8)
        assert d.behavior_mode == "repetition_aware"
        assert d.reduce_retrieval is True
        assert d.personality_mode == "empathetic"
        assert "high_repetition" in d.triggers

    def test_mild_repetition_adds_context(self):
        """Mild repetition (2) → adds behavior_context but doesn't change mode."""
        state = ConversationState(
            repetition_count=2,
            message_count=4,
        )
        d = BehaviorEngine.evaluate(state, "what is python", "knowledge_base", 0.8)
        assert d.behavior_mode == "standard"  # not changed to repetition_aware
        assert "mild_repetition" in d.triggers
        assert "revisiting" in d.behavior_context.lower()

    def test_greeting_simple(self):
        """Simple greeting → greeting mode, skip retrieval."""
        state = ConversationState(message_count=0)
        d = BehaviorEngine.evaluate(state, "hello", "general", 0.97)
        assert d.behavior_mode == "greeting"
        assert d.skip_retrieval is True
        assert "simple_greeting" in d.triggers

    def test_greeting_loop(self):
        """Repeated greetings → greeting loop mode."""
        state = ConversationState(
            message_count=3,
            intent_streak=2,
        )
        d = BehaviorEngine.evaluate(state, "hi", "general", 0.97)
        assert d.behavior_mode == "greeting"
        assert d.skip_retrieval is True
        assert "greeting_loop" in d.triggers

    def test_rapid_fire(self):
        """Rapid-fire interaction pattern → concise mode."""
        state = ConversationState(
            interaction_pattern="rapid_fire",
            short_query_streak=5,
            message_count=6,
        )
        d = BehaviorEngine.evaluate(state, "ok", "general", 0.8)
        assert d.behavior_mode == "rapid_fire"
        assert d.personality_mode == "concise"
        assert d.response_length_hint == "brief"
        assert d.reduce_retrieval is True
        assert "rapid_fire_pattern" in d.triggers

    def test_exploratory(self):
        """Exploratory interaction pattern → detailed, boost retrieval."""
        state = ConversationState(
            interaction_pattern="exploratory",
            intent_history=["general", "knowledge_base", "profile", "continuation"],
            message_count=5,
        )
        d = BehaviorEngine.evaluate(state, "tell me about databases", "knowledge_base", 0.8)
        assert d.behavior_mode == "exploratory"
        assert d.personality_mode == "detailed"
        assert d.boost_retrieval is True
        assert d.rag_k_override == 6
        assert "exploratory_pattern" in d.triggers

    def test_playful_tone_overlay(self):
        """Playful tone sets personality without changing mode."""
        state = ConversationState(
            emotional_tone="playful",
            message_count=2,
        )
        d = BehaviorEngine.evaluate(state, "haha that's cool tell me more", "knowledge_base", 0.8)
        assert d.personality_mode == "playful"
        assert "playful_tone" in d.triggers

    def test_curious_tone_overlay(self):
        """Curious tone → detailed personality."""
        state = ConversationState(
            emotional_tone="curious",
            message_count=2,
        )
        d = BehaviorEngine.evaluate(state, "how does this work exactly", "knowledge_base", 0.9)
        assert d.personality_mode == "detailed"
        assert d.response_length_hint == "detailed"
        assert "curious_tone" in d.triggers

    def test_positive_tone_overlay(self):
        """Positive tone adds context about warm atmosphere."""
        state = ConversationState(
            emotional_tone="positive",
            message_count=2,
        )
        d = BehaviorEngine.evaluate(state, "thanks that's great", "general", 0.8)
        assert "positive_tone" in d.triggers
        assert "positive" in d.behavior_context.lower()

    def test_priority_frustration_over_testing(self):
        """Frustration has higher priority than testing."""
        state = ConversationState(
            emotional_tone="frustrated",
            testing_flag=True,
            message_count=5,
        )
        d = BehaviorEngine.evaluate(state, "this doesn't work", "general", 0.5)
        assert d.behavior_mode == "frustration_recovery"

    def test_priority_testing_over_repetition(self):
        """Testing has higher priority than repetition."""
        state = ConversationState(
            testing_flag=True,
            interaction_pattern="testing",
            repetition_count=5,
            message_count=6,
        )
        d = BehaviorEngine.evaluate(state, "are you really smart", "general", 0.5)
        assert d.behavior_mode == "testing_aware"

    def test_social_filler_detected_as_greeting(self):
        """How are you → greeting mode."""
        state = ConversationState(message_count=0)
        d = BehaviorEngine.evaluate(state, "how are you", "general", 0.9)
        assert d.behavior_mode == "greeting"
        assert d.skip_retrieval is True

    def test_non_greeting_not_skipped(self):
        """Technical question shouldn't trigger greeting mode."""
        state = ConversationState(message_count=1)
        d = BehaviorEngine.evaluate(state, "explain how neural networks learn through backpropagation", "knowledge_base", 0.9)
        assert d.behavior_mode != "greeting"
        assert d.skip_retrieval is False
