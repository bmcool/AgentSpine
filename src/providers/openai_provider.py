from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import Provider, ProviderResponse, ToolCall


class OpenAIProvider(Provider):
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "openai"

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str | None = None,
        thinking_level: str = "off",
        on_text_delta: Any = None,
        api_key: str | None = None,
    ) -> ProviderResponse:
        if on_text_delta is not None:
            return self._complete_streaming(
                model=model,
                messages=messages,
                tools=tools,
                session_id=session_id,
                thinking_level=thinking_level,
                on_text_delta=on_text_delta,
                api_key=api_key,
            )

        request_kwargs = self._request_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            session_id=session_id,
            thinking_level=thinking_level,
        )
        response = self._safe_chat_create(api_key=api_key, **request_kwargs)
        msg = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments_json=tc.function.arguments,
            )
            for tc in (msg.tool_calls or [])
        ]
        assistant_message: dict[str, Any] = {"role": "assistant"}
        if msg.content is not None:
            assistant_message["content"] = msg.content
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
            text=msg.content or "",
            usage=_extract_openai_usage(getattr(response, "usage", None)),
        )

    def _complete_streaming(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str | None,
        thinking_level: str,
        on_text_delta: Any,
        api_key: str | None = None,
    ) -> ProviderResponse:
        request_kwargs = self._request_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            session_id=session_id,
            thinking_level=thinking_level,
            stream=True,
        )
        stream = self._safe_chat_create(api_key=api_key, **request_kwargs)

        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                text_parts.append(content)
                on_text_delta(content)

            delta_tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in delta_tool_calls:
                index = int(getattr(tc, "index", 0))
                entry = tool_parts.setdefault(index, {"id": "", "name": "", "arguments_json": ""})
                tc_id = getattr(tc, "id", None)
                if isinstance(tc_id, str) and tc_id:
                    entry["id"] = tc_id

                function = getattr(tc, "function", None)
                if function is not None:
                    fn_name = getattr(function, "name", None)
                    if isinstance(fn_name, str) and fn_name:
                        entry["name"] = fn_name
                    fn_args = getattr(function, "arguments", None)
                    if isinstance(fn_args, str) and fn_args:
                        entry["arguments_json"] += fn_args

        text = "".join(text_parts)
        tool_calls: list[ToolCall] = []
        for i in sorted(tool_parts):
            entry = tool_parts[i]
            if not entry["name"]:
                continue
            tool_calls.append(
                ToolCall(
                    id=entry["id"] or f"tool_call_{i}",
                    name=entry["name"],
                    arguments_json=entry["arguments_json"] or "{}",
                )
            )

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if text:
            assistant_message["content"] = text
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for tc in tool_calls
            ]
        return ProviderResponse(assistant_message=assistant_message, tool_calls=tool_calls, text=text, usage=None)

    def _request_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str | None,
        thinking_level: str,
        stream: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools if tools else None,
            "tool_choice": "auto" if tools else None,
            "stream": stream,
        }
        if not stream:
            kwargs.pop("stream")
        reasoning_effort = _to_openai_reasoning_effort(thinking_level)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        # Reserved for providers that add explicit session-aware caching knobs.
        if session_id:
            kwargs["user"] = session_id
        return kwargs

    def _safe_chat_create(self, *, api_key: str | None = None, **kwargs: Any):
        client = self._client if api_key is None else OpenAI(api_key=api_key, base_url=self._base_url)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Some model/backends reject reasoning params even when the SDK accepts them.
            if "reasoning_effort" in kwargs and _is_unsupported_reasoning_error(exc):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("reasoning_effort", None)
                return client.chat.completions.create(**fallback_kwargs)
            raise


def _extract_openai_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _to_openai_reasoning_effort(thinking_level: str) -> str | None:
    normalized = (thinking_level or "off").strip().lower()
    if normalized in {"off", ""}:
        return None
    if normalized in {"minimal", "low"}:
        return "low"
    if normalized == "medium":
        return "medium"
    if normalized in {"high", "xhigh"}:
        return "high"
    return None


def _is_unsupported_reasoning_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "reasoning_effort" in text and (
        "unknown" in text or "unsupported" in text or "not allowed" in text or "invalid" in text
    )
