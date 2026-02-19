from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
            raise AssertionError("fake provider exhausted")
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


def _assistant_text(text: str, usage: dict[str, int] | None = None) -> ProviderResponse:
    return ProviderResponse(
        assistant_message={"role": "assistant", "content": text},
        tool_calls=[],
        text=text,
        usage=usage,
    )


class AgentPiAlignmentTests(unittest.TestCase):
    def _new_agent(
        self,
        provider: FakeProvider,
        *,
        cancel_event: threading.Event | None = None,
        thinking_level: str | None = None,
        on_event: Any = None,
        transform_context: Any = None,
        convert_to_llm: Any = None,
        transform_messages_for_llm: Any = None,
        extra_tools: Any = None,
    ) -> Agent:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        with patch.object(Agent, "_create_provider", return_value=provider):
            return Agent(
                provider="openai",
                model="gpt-4o",
                workspace_dir=temp.name,
                sessions_dir=temp.name,
                cancel_event=cancel_event,
                thinking_level=thinking_level,
                on_event=on_event,
                transform_context=transform_context,
                convert_to_llm=convert_to_llm,
                transform_messages_for_llm=transform_messages_for_llm,
                extra_tools=extra_tools,
            )

    def test_steer_interrupts_remaining_tool_calls(self) -> None:
        provider = FakeProvider(
            [
                _assistant_with_tools(
                    ("tc1", "run_cmd", '{"command":"echo 1"}'), ("tc2", "run_cmd", '{"command":"echo 2"}')
                ),
                _assistant_text("done after steer"),
            ]
        )
        agent = self._new_agent(provider)
        tool_invocations = {"count": 0}

        def fake_execute_tool(*_: Any, **__: Any) -> str:
            tool_invocations["count"] += 1
            if tool_invocations["count"] == 1:
                agent.steer("please pivot")
            return "ok"

        with patch("src.agent.execute_tool", side_effect=fake_execute_tool):
            result = agent.chat("start")

        self.assertEqual(result, "done after steer")
        self.assertEqual(tool_invocations["count"], 1, "steer should skip remaining tool calls in turn")
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(any(msg.get("content") == "please pivot" for msg in agent.session.messages))
        tc2_results = [
            msg for msg in agent.session.messages if msg.get("tool_call_id") == "tc2" and msg.get("role") == "tool"
        ]
        self.assertEqual(len(tc2_results), 1)
        self.assertIn("Skipped due to user interrupt.", tc2_results[0]["content"])

    def test_follow_up_injects_after_terminal_turn(self) -> None:
        provider = FakeProvider([_assistant_text("first"), _assistant_text("second")])
        agent = self._new_agent(provider)
        agent.follow_up("run this afterwards")

        result = agent.chat("start")

        self.assertEqual(result, "second")
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(any(msg.get("content") == "run this afterwards" for msg in agent.session.messages))

    def test_abort_checked_before_each_tool_execution(self) -> None:
        cancel_event = threading.Event()
        provider = FakeProvider(
            [
                _assistant_with_tools(
                    ("tc1", "run_cmd", '{"command":"echo 1"}'), ("tc2", "run_cmd", '{"command":"echo 2"}')
                )
            ]
        )
        agent = self._new_agent(provider, cancel_event=cancel_event)
        tool_invocations = {"count": 0}

        def fake_execute_tool(*_: Any, **__: Any) -> str:
            tool_invocations["count"] += 1
            cancel_event.set()
            return "ok"

        with patch("src.agent.execute_tool", side_effect=fake_execute_tool):
            result = agent.chat("start")

        self.assertEqual(result, "(agent stopped: cancelled)")
        self.assertEqual(tool_invocations["count"], 1)

    def test_thinking_level_session_id_and_transform_hook(self) -> None:
        provider = FakeProvider([_assistant_text("ok")])
        captured = {"transformed": False}

        def transform(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            captured["transformed"] = True
            out = list(messages)
            out.append({"role": "assistant", "content": "transform-marker"})
            return out

        agent = self._new_agent(provider, thinking_level="high", transform_messages_for_llm=transform)

        result = agent.chat("hello")

        self.assertEqual(result, "ok")
        self.assertTrue(captured["transformed"])
        self.assertEqual(provider.calls[0]["thinking_level"], "high")
        self.assertEqual(provider.calls[0]["session_id"], agent.session.meta.session_id)
        seen_marker = any(msg.get("content") == "transform-marker" for msg in provider.calls[0]["messages"])
        self.assertTrue(seen_marker)

    def test_transform_context_then_convert_to_llm_pipeline(self) -> None:
        provider = FakeProvider([_assistant_text("ok")])
        calls: list[str] = []

        def transform_context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            calls.append("transform")
            out = list(messages)
            out.append({"role": "assistant", "content": "context-marker"})
            return out

        def convert_to_llm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            calls.append("convert")
            out = list(messages)
            out.append({"role": "assistant", "content": "convert-marker"})
            return out

        agent = self._new_agent(
            provider,
            transform_context=transform_context,
            convert_to_llm=convert_to_llm,
        )

        result = agent.chat("hello")

        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["transform", "convert"])
        sent_messages = provider.calls[0]["messages"]
        self.assertTrue(any(msg.get("content") == "context-marker" for msg in sent_messages))
        self.assertTrue(any(msg.get("content") == "convert-marker" for msg in sent_messages))

    def test_custom_entry_excluded_but_custom_message_included_in_llm_context(self) -> None:
        provider = FakeProvider([_assistant_text("ok")])
        agent = self._new_agent(provider)
        agent.session.add_custom_entry("memory", {"secret": "value"})
        agent.session.add_custom_message("inject", "custom-message-visible")
        agent.store.save(agent.session)

        result = agent.chat("hello")

        self.assertEqual(result, "ok")
        sent_messages = provider.calls[0]["messages"]
        contents = [m.get("content") for m in sent_messages if isinstance(m, dict)]
        self.assertIn("custom-message-visible", contents)
        flattened = json.dumps(sent_messages, ensure_ascii=False)
        self.assertNotIn('"secret": "value"', flattened)

    def test_on_event_emits_lifecycle_and_stream_updates(self) -> None:
        provider = FakeProvider([_assistant_text("streamed-text")])
        events: list[dict[str, Any]] = []
        agent = self._new_agent(provider, on_event=events.append)

        result = agent.chat_stream("hello", on_text_delta=lambda _: None)

        self.assertEqual(result, "streamed-text")
        event_types = [e.get("type") for e in events]
        self.assertIn("agent_start", event_types)
        self.assertIn("turn_start", event_types)
        self.assertIn("message_start", event_types)
        self.assertIn("message_update", event_types)
        self.assertIn("message_end", event_types)
        self.assertIn("turn_end", event_types)
        self.assertIn("agent_end", event_types)
        turn_end = next(e for e in events if e.get("type") == "turn_end")
        self.assertIn("tool_calls_count", turn_end)
        self.assertIn("assistant_message_preview", turn_end)
        self.assertIn("tool_results_preview", turn_end)

    def test_tool_progress_emits_tool_execution_update(self) -> None:
        provider = FakeProvider(
            [
                _assistant_with_tools(("tc1", "progress_tool", '{"value":"x"}')),
                _assistant_text("done"),
            ]
        )
        events: list[dict[str, Any]] = []

        def progress_tool(value: str, on_progress: Any = None) -> str:
            if on_progress is not None:
                on_progress(f"start:{value}")
                on_progress("finish")
            return "ok"

        extra_tools = [
            {
                "name": "progress_tool",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "progress_tool",
                        "description": "Progress-aware test tool",
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
        agent = self._new_agent(provider, on_event=events.append, extra_tools=extra_tools)

        result = agent.chat("run progress tool")

        self.assertEqual(result, "done")
        progress_events = [e for e in events if e.get("type") == "tool_execution_update"]
        self.assertEqual(len(progress_events), 2)
        self.assertEqual(progress_events[0].get("partial"), "start:x")
        self.assertEqual(progress_events[1].get("partial"), "finish")

    def test_extra_tool_exception_becomes_tool_error_result(self) -> None:
        provider = FakeProvider(
            [
                _assistant_with_tools(("tc1", "failing_tool", "{}")),
                _assistant_text("handled"),
            ]
        )

        def failing_tool() -> str:
            raise RuntimeError("boom")

        extra_tools = [
            {
                "name": "failing_tool",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "failing_tool",
                        "description": "Always fails",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                },
                "handler": failing_tool,
            }
        ]
        agent = self._new_agent(provider, extra_tools=extra_tools)

        result = agent.chat("run failing tool")

        self.assertEqual(result, "handled")
        tool_results = [m for m in agent.session.messages if m.get("role") == "tool" and m.get("tool_call_id") == "tc1"]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("[Tool Error] failing_tool: boom", tool_results[0]["content"])

    def test_continue_run_retries_without_new_user_message(self) -> None:
        provider = FakeProvider([_assistant_text("first"), _assistant_text("second")])
        agent = self._new_agent(provider)

        first = agent.chat("hello")
        agent.session.add_user_message("retry without appending via chat")
        second = agent.continue_run()

        self.assertEqual(first, "first")
        self.assertEqual(second, "second")
        self.assertEqual(len(provider.calls), 2)
        user_messages = [m for m in agent.session.messages if m.get("role") == "user"]
        self.assertEqual(len(user_messages), 2)

    def test_continue_run_requires_last_message_user_or_tool(self) -> None:
        provider = FakeProvider([_assistant_text("first")])
        agent = self._new_agent(provider)
        agent.chat("hello")

        with self.assertRaises(ValueError):
            agent.continue_run()

    def test_chat_with_one_tool_then_text_uses_real_execute_tool(self) -> None:
        """Full reactive loop: one tool call (read_file) then final text, no mock."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        workspace = Path(temp.name)
        test_file = workspace / "hello.txt"
        test_file.write_text("hello from file", encoding="utf-8")

        path_arg = json.dumps({"path": str(test_file)})
        provider = FakeProvider(
            [
                _assistant_with_tools(
                    ("tc1", "read_file", path_arg),
                ),
                _assistant_text("I read the file."),
            ]
        )
        with patch.object(Agent, "_create_provider", return_value=provider):
            agent = Agent(
                provider="openai",
                model="gpt-4o",
                workspace_dir=str(workspace),
                sessions_dir=str(workspace),
            )
        result = agent.chat("read hello.txt for me")

        self.assertEqual(result, "I read the file.")
        self.assertEqual(len(provider.calls), 2)
        # Tool result should be in session
        tool_results = [m for m in agent.session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("hello from file", tool_results[0]["content"])

    def test_get_api_key_is_forwarded_to_provider(self) -> None:
        provider = FakeProvider([_assistant_text("ok")])
        agent = self._new_agent(provider, extra_tools=None)
        agent.get_api_key = lambda provider_name: f"key-for-{provider_name}"

        result = agent.chat("hello")

        self.assertEqual(result, "ok")
        self.assertEqual(provider.calls[0]["api_key"], "key-for-openai")

    def test_before_turn_can_override_system_prompt(self) -> None:
        provider = FakeProvider([_assistant_text("ok")])

        def before_turn(_session_id: str, _round_no: int, _messages: list[dict[str, Any]], system_prompt: str):
            return (system_prompt + "\nTEST-HOOK-MARKER", None)

        agent = self._new_agent(provider, extra_tools=None)
        agent.before_turn = before_turn

        result = agent.chat("hello")

        self.assertEqual(result, "ok")
        first_messages = provider.calls[0]["messages"]
        system_message = first_messages[0]
        self.assertEqual(system_message.get("role"), "system")
        self.assertIn("TEST-HOOK-MARKER", system_message.get("content", ""))

    def test_usage_is_accumulated_into_session_meta(self) -> None:
        provider = FakeProvider(
            [
                _assistant_text(
                    "ok",
                    usage={
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "total_tokens": 14,
                        "cache_read_tokens": 2,
                        "cache_write_tokens": 1,
                    },
                )
            ]
        )
        agent = self._new_agent(provider)

        result = agent.chat("hello")

        self.assertEqual(result, "ok")
        self.assertEqual(agent.session.meta.usage_input_tokens, 10)
        self.assertEqual(agent.session.meta.usage_output_tokens, 4)
        self.assertEqual(agent.session.meta.usage_total_tokens, 14)
        self.assertEqual(agent.session.meta.usage_cache_read_tokens, 2)
        self.assertEqual(agent.session.meta.usage_cache_write_tokens, 1)

    def test_steer_emits_skipped_tool_events(self) -> None:
        provider = FakeProvider(
            [
                _assistant_with_tools(
                    ("tc1", "run_cmd", '{"command":"echo 1"}'),
                    ("tc2", "run_cmd", '{"command":"echo 2"}'),
                ),
                _assistant_text("done"),
            ]
        )
        events: list[dict[str, Any]] = []
        agent = self._new_agent(provider, on_event=events.append)
        invocations = {"count": 0}

        def fake_execute_tool(*_: Any, **__: Any) -> str:
            invocations["count"] += 1
            if invocations["count"] == 1:
                agent.steer("interrupt")
            return "ok"

        with patch("src.agent.execute_tool", side_effect=fake_execute_tool):
            result = agent.chat("start")

        self.assertEqual(result, "done")
        skipped_end = [e for e in events if e.get("type") == "tool_execution_end" and e.get("tool_call_id") == "tc2"]
        self.assertEqual(len(skipped_end), 1)
        self.assertTrue(bool(skipped_end[0].get("skipped")))

    def test_structured_tool_result_details_emitted(self) -> None:
        provider = FakeProvider([_assistant_with_tools(("tc1", "custom", "{}")), _assistant_text("done")])
        events: list[dict[str, Any]] = []
        agent = self._new_agent(provider, on_event=events.append)

        def fake_execute_tool(*_: Any, **__: Any) -> dict[str, Any]:
            return {"text": "plain-text", "details": {"artifact": "x"}}

        with patch("src.agent.execute_tool", side_effect=fake_execute_tool):
            result = agent.chat("start")

        self.assertEqual(result, "done")
        tool_result = next(
            m for m in agent.session.messages if m.get("role") == "tool" and m.get("tool_call_id") == "tc1"
        )
        self.assertEqual(tool_result.get("content"), "plain-text")
        end_events = [e for e in events if e.get("type") == "tool_execution_end" and e.get("tool_call_id") == "tc1"]
        self.assertEqual(len(end_events), 1)
        self.assertEqual(end_events[0].get("details"), {"artifact": "x"})


if __name__ == "__main__":
    unittest.main()
