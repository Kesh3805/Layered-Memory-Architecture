"""Pure Cerebras API client — zero behavior logic.

Every LLM call goes through ``completion()``.  This file knows nothing
about intents, prompts, profiles, or application behavior.
"""

import os
import logging

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Client ────────────────────────────────────────────────────────────────
_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")

# ── Token budgets (gpt-oss-120b context = 65 536) ────────────────────────
MAX_RESPONSE_TOKENS = 2048
MAX_CLASSIFIER_TOKENS = 50
MAX_PROFILE_DETECT_TOKENS = 300
MAX_TITLE_TOKENS = 20


def completion(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = MAX_RESPONSE_TOKENS,
    stream: bool = False,
):
    """Send a chat-completion request to Cerebras.

    Returns the full response object (``stream=False``) or an iterator
    (``stream=True``).
    """
    return _client.chat.completions.create(
        model=CEREBRAS_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )
