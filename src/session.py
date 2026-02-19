"""
Session data model (without persistence concerns).

Messages use a canonical chat structure:
  - {"role": "user", "content": "..."}
  - {"role": "assistant", "content": "...", "tool_calls": [...]}  (tool_calls optional)
  - {"role": "tool", "tool_call_id": "...", "content": "..."}
System prompt is built dynamically and injected at request time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionMeta:
    session_id: str
    provider: str
    model: str
    workspace_dir: str
    parent_session_id: str | None
    subagent_depth: int
    created_at: str
    updated_at: str


class Session:
    """In-memory conversation session (metadata + messages)."""

    def __init__(self, *, meta: SessionMeta, messages: list[dict[str, Any]] | None = None) -> None:
        self.meta = meta
        self.messages: list[dict[str, Any]] = messages[:] if messages else []

    # -- append helpers --------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.touch()

    def add_assistant_message(self, message: dict[str, Any]) -> None:
        """Append the raw assistant message dict (may contain tool_calls)."""
        self.messages.append(message)
        self.touch()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )
        self.touch()

    def add_system_event(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": f"[System Message] {content}"})
        self.touch()

    # -- utilities -------------------------------------------------------------

    def reset(self) -> None:
        """Clear all non-system conversation messages."""
        self.messages = []
        self.touch()

    def touch(self) -> None:
        self.meta.updated_at = utc_now_iso()

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"Session(id={self.meta.session_id}, messages={len(self.messages)})"
