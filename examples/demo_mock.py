from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import Agent
from src.providers.base import ProviderResponse, ToolCall


class FakeProvider:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = responses[:]
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "session_id": session_id,
                "thinking_level": thinking_level,
                "api_key": api_key,
            }
        )
        if not self._responses:
            raise AssertionError("FakeProvider exhausted")
        response = self._responses.pop(0)
        if on_text_delta is not None and response.text:
            on_text_delta(response.text)
        return response


def _assistant_with_tools(*calls: tuple[str, str, str]) -> ProviderResponse:
    tool_calls = [ToolCall(id=call_id, name=name, arguments_json=args) for call_id, name, args in calls]
    return ProviderResponse(
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for tc in tool_calls
            ],
        },
        tool_calls=tool_calls,
        text="",
    )


def _assistant_text(text: str) -> ProviderResponse:
    return ProviderResponse(
        assistant_message={"role": "assistant", "content": text},
        tool_calls=[],
        text=text,
    )


def _print_event(event: dict[str, Any]) -> None:
    et = event.get("type")
    if et == "message_update":
        return
    if et == "tool_execution_update":
        print(f"[event] {et}: {event.get('tool_name')} -> {event.get('partial')}")
        return
    if et == "tool_execution_start":
        print(f"[event] {et}: {event.get('tool_name')}({event.get('args')})")
        return
    if et == "tool_execution_end":
        print(f"[event] {et}: {event.get('tool_name')} result={event.get('result_preview')}")
        return
    print(f"[event] {et}: {json.dumps(event, ensure_ascii=False)}")


def main() -> None:
    print("=== AgentSpine Mock Demo (no API keys) ===")
    print("Shows: events, steer skip, follow_up, tool progress, continue_run")

    provider = FakeProvider(
        [
            _assistant_with_tools(
                ("tc1", "run_cmd", '{"command":"echo first"}'),
                ("tc2", "run_cmd", '{"command":"echo second"}'),
            ),
            _assistant_text("Steer handled. Remaining tool calls were skipped."),
            _assistant_with_tools(("tc3", "progress_tool", '{"value":"demo"}')),
            _assistant_text("Progress tool completed."),
            _assistant_text("First terminal turn completed."),
            _assistant_text("Follow-up turn completed."),
            _assistant_text("Continue run completed."),
        ]
    )

    def progress_tool(value: str, on_progress: Any = None) -> str:
        if on_progress is not None:
            on_progress(f"start:{value}")
            on_progress("middle")
            on_progress("finish")
        return f"progress_tool done for {value}"

    extra_tools = [
        {
            "name": "progress_tool",
            "definition": {
                "type": "function",
                "function": {
                    "name": "progress_tool",
                    "description": "Progress-aware demo tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            },
            "handler": progress_tool,
        }
    ]

    temp = tempfile.TemporaryDirectory()
    try:
        with patch.object(Agent, "_create_provider", return_value=provider):
            agent = Agent(
                provider="openai",
                model="gpt-4o",
                workspace_dir=temp.name,
                sessions_dir=temp.name,
                on_event=_print_event,
                extra_tools=extra_tools,
            )

        print("\n--- Demo 1: steer skips remaining tool calls ---")
        invocation_count = {"count": 0}

        from src import agent as agent_module

        original_execute_tool = agent_module.execute_tool

        def patched_execute_tool(*args: Any, **kwargs: Any) -> str:
            invocation_count["count"] += 1
            if invocation_count["count"] == 1:
                agent.steer("Pivot now. Ignore remaining calls in this turn.")
            return original_execute_tool(*args, **kwargs)

        with patch("src.agent.execute_tool", side_effect=patched_execute_tool):
            reply = agent.chat("Run two tools then handle steer.")
        print(f"[assistant] {reply}")

        skipped = [m for m in agent.session.messages if m.get("role") == "tool" and m.get("tool_call_id") == "tc2"]
        print(f"[check] skipped tool result exists: {bool(skipped)}")
        if skipped:
            print(f"[check] tc2 content: {skipped[0].get('content')}")

        print("\n--- Demo 2: tool progress updates ---")
        reply = agent.chat("Run progress_tool now.")
        print(f"[assistant] {reply}")

        print("\n--- Demo 3: follow_up queue ---")
        agent.follow_up("This is queued follow-up input.")
        reply = agent.chat("Run a terminal turn, then inject follow-up.")
        print(f"[assistant] {reply}")

        print("\n--- Demo 4: continue_run without chat() ---")
        agent.session.add_user_message("Please continue without calling chat().")
        reply = agent.continue_run()
        print(f"[assistant] {reply}")

        role_counts: dict[str, int] = {}
        for msg in agent.session.messages:
            role = str(msg.get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
        print("\n=== Session Transcript Summary ===")
        print(f"Provider calls: {len(provider.calls)}")
        print(f"Role counts: {role_counts}")
        print("Demo complete.")
    finally:
        temp.cleanup()


if __name__ == "__main__":
    main()
