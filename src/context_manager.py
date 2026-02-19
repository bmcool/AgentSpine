from __future__ import annotations

import os
from typing import Any

from .context_estimate import estimate_tokens


def _message_text_size(message: dict[str, Any]) -> int:
    total = 0
    content = message.get("content")
    if isinstance(content, str):
        total += len(content)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments")
            name = fn.get("name")
            if isinstance(args, str):
                total += len(args)
            if isinstance(name, str):
                total += len(name)
    return total


def _message_tokens(message: dict[str, Any]) -> int:
    raw = ""
    content = message.get("content")
    if isinstance(content, str):
        raw += content
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments")
            name = fn.get("name")
            if isinstance(args, str):
                raw += args
            if isinstance(name, str):
                raw += name
    return estimate_tokens(raw)


class ContextManager:
    """
    Lightweight context control:
    - keep recent messages
    - hard cap by characters or estimated tokens (AGENT_CONTEXT_MODE)
    - optional simplified compaction summary of old messages
    """

    def __init__(
        self,
        *,
        max_chars: int = 24_000,
        keep_last_messages: int = 30,
        compact_trigger_chars: int = 36_000,
        compact_keep_tail: int = 16,
        mode: str = "chars",
        max_tokens: int = 24_000,
        compact_trigger_tokens: int = 36_000,
    ) -> None:
        self._mode = (mode or "chars").strip().lower()
        if self._mode not in ("chars", "tokens"):
            self._mode = "chars"
        self.max_chars = max_chars
        self.keep_last_messages = keep_last_messages
        self.compact_trigger_chars = compact_trigger_chars
        self.compact_keep_tail = compact_keep_tail
        self.max_tokens = max_tokens
        self.compact_trigger_tokens = compact_trigger_tokens

    @classmethod
    def from_env(cls) -> "ContextManager":
        mode = (os.getenv("AGENT_CONTEXT_MODE") or "chars").strip().lower()
        if mode not in ("chars", "tokens"):
            mode = "chars"
        max_chars = max(1000, int(os.getenv("AGENT_MAX_CHARS", "24000") or "24000"))
        max_tokens = max(500, int(os.getenv("AGENT_MAX_TOKENS", "24000") or "24000"))
        compact_trigger_chars = max(max_chars, int(os.getenv("AGENT_COMPACT_TRIGGER_CHARS", "36000") or "36000"))
        compact_trigger_tokens = max(max_tokens, int(os.getenv("AGENT_COMPACT_TRIGGER_TOKENS", "36000") or "36000"))
        keep_last = max(5, int(os.getenv("AGENT_KEEP_LAST_MESSAGES", "30") or "30"))
        compact_tail = max(4, int(os.getenv("AGENT_COMPACT_KEEP_TAIL", "16") or "16"))
        return cls(
            max_chars=max_chars,
            keep_last_messages=keep_last,
            compact_trigger_chars=compact_trigger_chars,
            compact_keep_tail=compact_tail,
            mode=mode,
            max_tokens=max_tokens,
            compact_trigger_tokens=compact_trigger_tokens,
        )

    def prepare_messages(
        self,
        *,
        system_prompt: str,
        history_messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Return messages to send and whether simplified compaction was applied.
        """
        working = history_messages[:]
        total = self._total_measure(working)
        trigger = self.compact_trigger_tokens if self._mode == "tokens" else self.compact_trigger_chars
        cap = self.max_tokens if self._mode == "tokens" else self.max_chars

        compacted = False
        if total > trigger and len(working) > self.compact_keep_tail:
            working = self._compact(working)
            compacted = True

        if len(working) > self.keep_last_messages:
            working = working[-self.keep_last_messages :]

        while self._total_measure(working) > cap and len(working) > 4:
            del working[0]

        with_system = [{"role": "system", "content": system_prompt}, *working]
        return with_system, compacted

    def _total_measure(self, messages: list[dict[str, Any]]) -> int:
        if self._mode == "tokens":
            return sum(_message_tokens(msg) for msg in messages)
        return sum(_message_text_size(msg) for msg in messages)

    def _compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        head = messages[: max(0, len(messages) - self.compact_keep_tail)]
        tail = messages[max(0, len(messages) - self.compact_keep_tail) :]
        summary = self._build_summary(head)
        return [summary, *tail]

    def _build_summary(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        points: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "unknown"))
            content = msg.get("content")
            if isinstance(content, str):
                text = content.strip().replace("\n", " ")
            else:
                text = ""
            if not text:
                continue
            short = text[:140] + ("..." if len(text) > 140 else "")
            points.append(f"- {role}: {short}")
            if len(points) >= 10:
                break
        if not points:
            points = ["- No significant earlier content."]
        summary_text = "[Compacted conversation summary]\n" + "\n".join(points)
        return {"role": "assistant", "content": summary_text}
