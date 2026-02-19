"""
Optional token estimation for context limits.

Uses a heuristic (no extra dependencies). For stricter token counts
you can plug in tiktoken or model-specific logic elsewhere.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a string (heuristic: ~4 chars per token for English/code).
    No external dependency; suitable for context capping when AGENT_CONTEXT_MODE=tokens.
    """
    if not text:
        return 0
    # Typical ratio for OpenAI/Anthropic tokenizers is ~3.5–4 chars per token
    return max(1, len(text) // 4)
