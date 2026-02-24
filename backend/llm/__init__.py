"""LLM package — re-exports for backward compatibility.

For new code, import from submodules directly::

    from llm.classifier import classify_intent
    from llm.generators import generate_response_stream
"""

from .classifier import classify_intent
from .generators import generate_response, generate_response_stream, generate_title
from .profile_detector import detect_profile_updates

__all__ = [
    "classify_intent",
    "generate_response",
    "generate_response_stream",
    "generate_title",
    "detect_profile_updates",
]
