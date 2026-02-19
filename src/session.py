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
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    usage_total_tokens: int = 0
    usage_cache_read_tokens: int = 0
    usage_cache_write_tokens: int = 0


class Session:
    """In-memory conversation session (metadata + messages)."""

    def __init__(
        self,
        *,
        meta: SessionMeta,
        messages: list[dict[str, Any]] | None = None,
        entries: list[dict[str, Any]] | None = None,
    ) -> None:
        self.meta = meta
        if entries is not None:
            self.entries: list[dict[str, Any]] = entries[:]
        else:
            self.entries = []
            for msg in messages[:] if messages else []:
                self.entries.append(
                    {
                        "type": "message",
                        "message": msg,
                        "timestamp": utc_now_iso(),
                    }
                )

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Backward-compatible chat history view."""
        return self.get_history_messages()

    @messages.setter
    def messages(self, history_messages: list[dict[str, Any]]) -> None:
        self.replace_history_messages(history_messages, preserve_non_history=False)

    # -- append helpers --------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self.entries.append(
            {
                "type": "message",
                "message": {"role": "user", "content": content},
                "timestamp": utc_now_iso(),
            }
        )
        self.touch()

    def add_assistant_message(self, message: dict[str, Any]) -> None:
        """Append the raw assistant message dict (may contain tool_calls)."""
        self.entries.append(
            {
                "type": "message",
                "message": message,
                "timestamp": utc_now_iso(),
            }
        )
        self.touch()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.entries.append(
            {
                "type": "message",
                "message": {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                },
                "timestamp": utc_now_iso(),
            }
        )
        self.touch()

    def add_system_event(self, content: str) -> None:
        self.entries.append(
            {
                "type": "message",
                "message": {"role": "assistant", "content": f"[System Message] {content}"},
                "timestamp": utc_now_iso(),
            }
        )
        self.touch()

    def add_custom_entry(self, custom_type: str, data: Any) -> None:
        self.entries.append(
            {
                "type": "custom",
                "custom_type": custom_type,
                "data": data,
                "timestamp": utc_now_iso(),
            }
        )
        self.touch()

    def add_custom_message(
        self,
        custom_type: str,
        content: str,
        *,
        details: Any | None = None,
        display: bool = True,
        role: str = "user",
    ) -> None:
        row: dict[str, Any] = {
            "type": "custom_message",
            "custom_type": custom_type,
            "content": content,
            "display": bool(display),
            "role": role if role in {"user", "assistant"} else "user",
            "timestamp": utc_now_iso(),
        }
        if details is not None:
            row["details"] = details
        self.entries.append(row)
        self.touch()

    def add_compaction_entry(self, summary: str, details: Any | None = None) -> None:
        row: dict[str, Any] = {
            "type": "compaction",
            "summary": summary,
            "timestamp": utc_now_iso(),
        }
        if details is not None:
            row["details"] = details
        self.entries.append(row)
        self.touch()

    def get_history_messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self.entries:
            kind = row.get("type")
            if kind == "message":
                message = row.get("message")
                if isinstance(message, dict):
                    out.append(message)
                continue
            if kind == "custom_message":
                content = row.get("content")
                if not isinstance(content, str):
                    continue
                role = str(row.get("role", "user")).strip().lower()
                if role not in {"user", "assistant"}:
                    role = "user"
                out.append({"role": role, "content": content})
        return out

    def replace_history_messages(self, history_messages: list[dict[str, Any]], *, preserve_non_history: bool) -> None:
        kept: list[dict[str, Any]] = []
        if preserve_non_history:
            for row in self.entries:
                kind = row.get("type")
                if kind not in {"message", "custom_message"}:
                    kept.append(row)
        replaced = kept + [
            {
                "type": "message",
                "message": msg,
                "timestamp": utc_now_iso(),
            }
            for msg in history_messages
            if isinstance(msg, dict)
        ]
        self.entries = replaced
        self.touch()

    def accumulate_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self.meta.usage_input_tokens += max(0, int(input_tokens))
        self.meta.usage_output_tokens += max(0, int(output_tokens))
        self.meta.usage_total_tokens += max(0, int(total_tokens))
        self.meta.usage_cache_read_tokens += max(0, int(cache_read_tokens))
        self.meta.usage_cache_write_tokens += max(0, int(cache_write_tokens))
        self.touch()

    # -- utilities -------------------------------------------------------------

    def reset(self) -> None:
        """Clear all non-system conversation messages."""
        self.entries = []
        self.touch()

    def touch(self) -> None:
        self.meta.updated_at = utc_now_iso()

    def __len__(self) -> int:
        return len(self.get_history_messages())

    def __repr__(self) -> str:
        return f"Session(id={self.meta.session_id}, messages={len(self.messages)})"
