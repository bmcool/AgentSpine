from __future__ import annotations

import json
from typing import Any, Callable

from anthropic import Anthropic

from .base import Provider, ProviderResponse, ToolCall


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _to_anthropic_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in openai_tools:
        fn = tool.get("function", {})
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        result.append(
            {
                "name": name,
                "description": str(fn.get("description", "")),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return result


def _append_block_message(
    out: list[dict[str, Any]],
    *,
    role: str,
    blocks: list[dict[str, Any]],
) -> None:
    if not blocks:
        return
    if out and out[-1]["role"] == role and isinstance(out[-1]["content"], list):
        out[-1]["content"].extend(blocks)
        return
    out.append({"role": role, "content": blocks})


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_text = _as_text(msg.get("content"))
            if system_text:
                system_parts.append(system_text)
            continue

        if role == "user":
            user_text = _as_text(msg.get("content"))
            _append_block_message(
                out,
                role="user",
                blocks=[{"type": "text", "text": user_text}],
            )
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            assistant_text = _as_text(msg.get("content"))
            if assistant_text:
                blocks.append({"type": "text", "text": assistant_text})
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                name = fn.get("name")
                arguments_raw = fn.get("arguments")
                if not isinstance(name, str) or not name:
                    continue
                if isinstance(arguments_raw, str):
                    try:
                        tool_input = json.loads(arguments_raw)
                    except json.JSONDecodeError:
                        tool_input = {"raw": arguments_raw}
                elif isinstance(arguments_raw, dict):
                    tool_input = arguments_raw
                else:
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tc.get("id", "")),
                        "name": name,
                        "input": tool_input,
                    }
                )
            _append_block_message(out, role="assistant", blocks=blocks)
            continue

        if role == "tool":
            tool_id = str(msg.get("tool_call_id", ""))
            tool_text = _as_text(msg.get("content"))
            _append_block_message(
                out,
                role="user",
                blocks=[
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": tool_text,
                    }
                ],
            )
            continue

    return "\n\n".join(system_parts), out


class AnthropicProvider(Provider):
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._client = Anthropic(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str | None = None,
        thinking_level: str = "off",
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        _ = session_id  # Reserved for future provider-side session-aware caching.
        system, anthropic_messages = _to_anthropic_messages(messages)
        anthropic_tools = _to_anthropic_tools(tools)
        thinking = _to_anthropic_thinking(thinking_level)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 2048,
            "system": system if system else None,
            "messages": anthropic_messages,
            "tools": anthropic_tools if anthropic_tools else None,
        }
        if thinking:
            create_kwargs["thinking"] = thinking

        if on_text_delta is None:
            resp = self._safe_messages_create(create_kwargs)
        else:
            with self._safe_messages_stream(create_kwargs) as stream:
                for text in stream.text_stream:
                    if text:
                        on_text_delta(text)
                resp = stream.get_final_message()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments_json=json.dumps(block.input or {}),
                    )
                )

        text = "\n".join([part for part in text_parts if part]).strip()
        assistant_message: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for tc in tool_calls
            ]
        return ProviderResponse(
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            text=text,
        )

    def _safe_messages_create(self, kwargs: dict[str, Any]):
        try:
            return self._client.messages.create(**kwargs)
        except Exception as exc:
            if kwargs.get("thinking") is not None and _is_unsupported_thinking_error(exc):
                fallback = dict(kwargs)
                fallback.pop("thinking", None)
                return self._client.messages.create(**fallback)
            raise

    def _safe_messages_stream(self, kwargs: dict[str, Any]):
        try:
            return self._client.messages.stream(**kwargs)
        except Exception as exc:
            if kwargs.get("thinking") is not None and _is_unsupported_thinking_error(exc):
                fallback = dict(kwargs)
                fallback.pop("thinking", None)
                return self._client.messages.stream(**fallback)
            raise


def _to_anthropic_thinking(thinking_level: str) -> dict[str, Any] | None:
    normalized = (thinking_level or "off").strip().lower()
    budgets: dict[str, int] = {
        "minimal": 1024,
        "low": 2048,
        "medium": 4096,
        "high": 8192,
        "xhigh": 12000,
    }
    if normalized not in budgets:
        return None
    return {"type": "enabled", "budget_tokens": budgets[normalized]}


def _is_unsupported_thinking_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "thinking" in text and ("unsupported" in text or "invalid" in text or "unknown" in text)
