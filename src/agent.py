"""
Reactive agent with prompt engineering, persistence, and context management.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .context_manager import ContextManager
from .lane_queue import LaneQueue
from .prompt_builder import PromptBuilder
from .providers import AnthropicProvider, OpenAIProvider, Provider
from .session_store import SessionStore
from .subagent_registry import SubagentRegistry
from .subagent_runtime import _GLOBAL_SUBAGENT_RUNTIME
from .tools import execute_tool, get_tool_definitions, get_tool_summaries

MAX_TOOL_ROUNDS = 20
MAX_TOOL_RESULT_CHARS = 8_000
MAX_TOOL_REPEAT_ROUNDS = 3


_GLOBAL_LANE_QUEUE = LaneQueue(max_concurrent=max(1, int(os.getenv("AGENT_MAX_CONCURRENT", "4"))))


def _default_model(provider: str) -> str:
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    return os.getenv("OPENAI_MODEL", "gpt-4o")


# Type for extra tool entry: name, OpenAI-style definition, handler returning str
ExtraTool = dict[str, Any]
EventHandler = Callable[[dict[str, Any]], None]
MessageTransformer = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
ContextTransformer = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
MessageConverter = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
BeforeTurnHook = Callable[[str, int, list[dict[str, Any]], str], tuple[str | None, list[dict[str, Any]] | None] | None]
ApiKeyResolver = Callable[[str], str | None]

SKIPPED_DUE_TO_STEER = "Skipped due to user interrupt."
TOOL_ERROR_PREFIX = "[Tool Error]"


def _default_thinking_level() -> str:
    return (os.getenv("AGENT_THINKING_LEVEL") or "off").strip().lower()


class Agent:
    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        workspace_dir: str | None = None,
        sessions_dir: str | None = None,
        enable_orchestration: bool = True,
        extra_tools: list[ExtraTool] | None = None,
        parent_session_id: str | None = None,
        subagent_depth: int = 0,
        cancel_event: threading.Event | None = None,
        thinking_level: str | None = None,
        on_event: EventHandler | None = None,
        transform_context: ContextTransformer | None = None,
        convert_to_llm: MessageConverter | None = None,
        transform_messages_for_llm: MessageTransformer | None = None,
        before_turn: BeforeTurnHook | None = None,
        get_api_key: ApiKeyResolver | None = None,
    ) -> None:
        self.provider_name = (provider or os.getenv("AGENT_PROVIDER") or "openai").strip().lower()
        if self.provider_name not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

        self.model = (model or _default_model(self.provider_name)).strip()
        self.workspace_dir = str(Path(workspace_dir or os.getcwd()).resolve())
        self.provider = self._create_provider(self.provider_name)
        self.prompt_builder = PromptBuilder(max_tool_output_chars=MAX_TOOL_RESULT_CHARS)
        self.context_manager = ContextManager.from_env()
        self.enable_orchestration = enable_orchestration
        self.extra_tools: list[ExtraTool] = list(extra_tools) if extra_tools else []
        self.parent_session_id = parent_session_id
        self.subagent_depth = max(0, int(subagent_depth))
        self.cancel_event = cancel_event
        self.thinking_level = (thinking_level or _default_thinking_level()).strip().lower() or "off"
        self.on_event = on_event
        self.transform_context = transform_context
        self.convert_to_llm = convert_to_llm
        self.transform_messages_for_llm = transform_messages_for_llm
        self.before_turn = before_turn
        self.get_api_key = get_api_key
        self._queue_lock = threading.Lock()
        self._steering_queue: list[str] = []
        self._follow_up_queue: list[str] = []

        default_sessions_dir = Path(__file__).resolve().parents[1] / "sessions"
        self.store = SessionStore(sessions_dir=sessions_dir or str(default_sessions_dir))
        self.subagent_registry = SubagentRegistry(file_path=str(Path(self.store.sessions_dir) / "subagents.json"))
        resolved_session_id = self.store.resolve_session_id(session_id)
        self.session = self.store.load_or_create(
            session_id=resolved_session_id,
            provider=self.provider_name,
            model=self.model,
            workspace_dir=self.workspace_dir,
            parent_session_id=self.parent_session_id,
            subagent_depth=self.subagent_depth,
        )

    def chat(self, user_input: str) -> str:
        lane_id = self.session.meta.session_id
        return _GLOBAL_LANE_QUEUE.run(
            lane_id,
            lambda: self._chat_impl(user_input),
            on_metrics=self._on_lane_metrics,
        )

    def chat_stream(self, user_input: str, on_text_delta: Callable[[str], None]) -> str:
        lane_id = self.session.meta.session_id
        return _GLOBAL_LANE_QUEUE.run(
            lane_id,
            lambda: self._chat_stream_impl(user_input, on_text_delta),
            on_metrics=self._on_lane_metrics,
        )

    def continue_run(self) -> str:
        lane_id = self.session.meta.session_id
        return _GLOBAL_LANE_QUEUE.run(
            lane_id,
            self._continue_impl,
            on_metrics=self._on_lane_metrics,
        )

    def continue_run_stream(self, on_text_delta: Callable[[str], None]) -> str:
        lane_id = self.session.meta.session_id
        return _GLOBAL_LANE_QUEUE.run(
            lane_id,
            lambda: self._continue_impl(on_text_delta=on_text_delta),
            on_metrics=self._on_lane_metrics,
        )

    def _chat_impl(self, user_input: str) -> str:
        self.session.add_user_message(user_input)
        self.store.save(self.session)
        return self._run_loop()

    def _chat_stream_impl(self, user_input: str, on_text_delta: Callable[[str], None]) -> str:
        self.session.add_user_message(user_input)
        self.store.save(self.session)
        return self._run_loop(on_text_delta=on_text_delta)

    def _continue_impl(self, on_text_delta: Callable[[str], None] | None = None) -> str:
        if not self.session.messages:
            raise ValueError("Cannot continue: no messages in context")
        last_role = str(self.session.messages[-1].get("role", ""))
        if last_role not in {"user", "tool"}:
            raise ValueError("Cannot continue: last message must be user or tool")
        return self._run_loop(on_text_delta=on_text_delta)

    def reset(self) -> None:
        self.session.reset()
        self.store.save(self.session)

    def steer(self, user_input: str) -> None:
        message = user_input.strip()
        if not message:
            return
        with self._queue_lock:
            self._steering_queue.append(message)

    def follow_up(self, user_input: str) -> None:
        message = user_input.strip()
        if not message:
            return
        with self._queue_lock:
            self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        with self._queue_lock:
            self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        with self._queue_lock:
            self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        with self._queue_lock:
            self._steering_queue.clear()
            self._follow_up_queue.clear()

    def _run_loop(self, on_text_delta: Callable[[str], None] | None = None) -> str:
        self._emit_event({"type": "agent_start"})
        last_tool_signature = ""
        repeat_rounds = 0
        for _round in range(MAX_TOOL_ROUNDS):
            round_no = _round + 1
            assistant_preview = ""
            tool_results_preview: list[str] = []
            self._emit_event({"type": "turn_start", "round": round_no})
            if self.cancel_event is not None and self.cancel_event.is_set():
                self._emit_event(
                    {
                        "type": "turn_end",
                        "round": round_no,
                        "status": "cancelled",
                        "tool_calls_count": 0,
                        "assistant_message_preview": assistant_preview,
                        "tool_results_preview": tool_results_preview,
                    }
                )
                return self._finish_run("(agent stopped: cancelled)")
            system_prompt = self.prompt_builder.build(
                provider=self.provider_name,
                model=self.model,
                workspace_dir=self.workspace_dir,
                tool_summaries=get_tool_summaries(
                    include_orchestration=self.enable_orchestration,
                    extra_tools=self.extra_tools,
                ),
            )
            history_messages = self.session.messages
            if self.transform_context is not None:
                transformed_history = self.transform_context(list(self.session.messages))
                if isinstance(transformed_history, list):
                    history_messages = transformed_history
            if self.before_turn is not None:
                hook_result = self.before_turn(
                    self.session.meta.session_id, round_no, list(history_messages), system_prompt
                )
                if isinstance(hook_result, tuple) and len(hook_result) == 2:
                    prompt_override, prepend_messages = hook_result
                    if isinstance(prompt_override, str) and prompt_override.strip():
                        system_prompt = prompt_override
                    if isinstance(prepend_messages, list):
                        history_messages = list(prepend_messages) + list(history_messages)
            llm_messages, compacted = self.context_manager.prepare_messages(
                system_prompt=system_prompt,
                history_messages=history_messages,
            )
            if compacted and history_messages is self.session.messages:
                # Keep session history aligned with compacted context (exclude system).
                self.session.messages = llm_messages[1:]
                self.store.save(self.session)
            if self.convert_to_llm is not None:
                converted = self.convert_to_llm(llm_messages)
                if isinstance(converted, list):
                    llm_messages = converted
            if self.transform_messages_for_llm is not None:
                transformed = self.transform_messages_for_llm(llm_messages)
                if isinstance(transformed, list):
                    llm_messages = transformed

            tool_definitions = get_tool_definitions(
                include_orchestration=self.enable_orchestration,
                extra_tools=self.extra_tools,
            )
            self._emit_event({"type": "message_start", "role": "assistant", "round": round_no})

            def _forward_delta(delta: str) -> None:
                if not delta:
                    return
                if on_text_delta is not None:
                    on_text_delta(delta)
                self._emit_event({"type": "message_update", "role": "assistant", "delta": delta})

            response = self._complete_with_retry(
                model=self.model,
                messages=llm_messages,
                tools=tool_definitions,
                session_id=self.session.meta.session_id,
                thinking_level=self.thinking_level,
                on_text_delta=_forward_delta if on_text_delta is not None else None,
            )
            self._emit_event(
                {
                    "type": "message_end",
                    "role": "assistant",
                    "round": round_no,
                    "text_preview": _truncate(response.text or "", 200),
                }
            )
            assistant_preview = _truncate(response.text or "", 200)
            self.session.add_assistant_message(response.assistant_message)
            if response.usage:
                self.session.accumulate_usage(
                    input_tokens=int(response.usage.get("input_tokens", 0) or 0),
                    output_tokens=int(response.usage.get("output_tokens", 0) or 0),
                    total_tokens=int(response.usage.get("total_tokens", 0) or 0),
                    cache_read_tokens=int(response.usage.get("cache_read_tokens", 0) or 0),
                    cache_write_tokens=int(response.usage.get("cache_write_tokens", 0) or 0),
                )
            self.store.save(self.session)

            if not response.tool_calls:
                queued_follow_up = self._pop_follow_up_message()
                if queued_follow_up is not None:
                    self._append_queued_user_message(queued_follow_up, source="follow_up", round_no=round_no)
                    self._emit_event(
                        {
                            "type": "turn_end",
                            "round": round_no,
                            "status": "follow_up_injected",
                            "tool_calls_count": 0,
                            "assistant_message_preview": assistant_preview,
                            "tool_results_preview": tool_results_preview,
                        }
                    )
                    continue
                self._emit_event(
                    {
                        "type": "turn_end",
                        "round": round_no,
                        "status": "completed",
                        "tool_calls_count": 0,
                        "assistant_message_preview": assistant_preview,
                        "tool_results_preview": tool_results_preview,
                    }
                )
                return self._finish_run(response.text)

            signature = "|".join(f"{call.name}:{call.arguments_json}" for call in response.tool_calls)
            if signature and signature == last_tool_signature:
                repeat_rounds += 1
            else:
                repeat_rounds = 1
                last_tool_signature = signature
            if repeat_rounds >= MAX_TOOL_REPEAT_ROUNDS:
                self._emit_event(
                    {
                        "type": "turn_end",
                        "round": round_no,
                        "status": "loop_detected",
                        "tool_calls_count": len(response.tool_calls),
                        "assistant_message_preview": assistant_preview,
                        "tool_results_preview": tool_results_preview,
                    }
                )
                return self._finish_run("(agent stopped: repeated tool-call loop detected)")

            steering_triggered = False
            tool_calls_count = len(response.tool_calls)
            for idx, call in enumerate(response.tool_calls):
                if self.cancel_event is not None and self.cancel_event.is_set():
                    self._emit_event(
                        {
                            "type": "turn_end",
                            "round": round_no,
                            "status": "cancelled",
                            "tool_calls_count": tool_calls_count,
                            "assistant_message_preview": assistant_preview,
                            "tool_results_preview": tool_results_preview,
                        }
                    )
                    return self._finish_run("(agent stopped: cancelled)")
                print(f"  [tool] {call.name}({_truncate(call.arguments_json, 96)})")
                self._emit_event(
                    {
                        "type": "tool_execution_start",
                        "round": round_no,
                        "tool_call_id": call.id,
                        "tool_name": call.name,
                        "args": call.arguments_json,
                    }
                )
                try:
                    tool_output = execute_tool(
                        call.name,
                        call.arguments_json,
                        runtime_hooks=self._runtime_tool_hooks(),
                        extra_handlers=self._extra_tool_handlers(),
                        on_progress=lambda text: self._emit_event(
                            {
                                "type": "tool_execution_update",
                                "round": round_no,
                                "tool_call_id": call.id,
                                "tool_name": call.name,
                                "partial": text,
                            }
                        ),
                    )
                except Exception as exc:
                    tool_output = f"{TOOL_ERROR_PREFIX} {call.name}: {exc}"
                result_text, result_details = self._normalize_tool_output(tool_output)
                truncated_result = self._truncate_tool_result(result_text)
                tool_results_preview.append(_truncate(truncated_result, 200))
                self.session.add_tool_result(
                    tool_call_id=call.id,
                    content=truncated_result,
                )
                end_event: dict[str, Any] = {
                    "type": "tool_execution_end",
                    "round": round_no,
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "result_preview": _truncate(truncated_result, 200),
                }
                if result_details is not None:
                    end_event["details"] = result_details
                self._emit_event(end_event)
                queued_steer = self._pop_steering_message()
                if queued_steer is not None:
                    for skipped_call in response.tool_calls[idx + 1 :]:
                        self._emit_event(
                            {
                                "type": "tool_execution_start",
                                "round": round_no,
                                "tool_call_id": skipped_call.id,
                                "tool_name": skipped_call.name,
                                "args": skipped_call.arguments_json,
                            }
                        )
                        skipped_preview = _truncate(SKIPPED_DUE_TO_STEER, 200)
                        self._emit_event(
                            {
                                "type": "tool_execution_end",
                                "round": round_no,
                                "tool_call_id": skipped_call.id,
                                "tool_name": skipped_call.name,
                                "result_preview": skipped_preview,
                                "skipped": True,
                            }
                        )
                        tool_results_preview.append(skipped_preview)
                        self.session.add_tool_result(
                            tool_call_id=skipped_call.id,
                            content=SKIPPED_DUE_TO_STEER,
                        )
                    self._append_queued_user_message(queued_steer, source="steer", round_no=round_no)
                    steering_triggered = True
                    break
            self.store.save(self.session)
            status = "steered" if steering_triggered else "tool_calls_processed"
            self._emit_event(
                {
                    "type": "turn_end",
                    "round": round_no,
                    "status": status,
                    "tool_calls_count": tool_calls_count,
                    "assistant_message_preview": assistant_preview,
                    "tool_results_preview": tool_results_preview,
                }
            )

        return self._finish_run("(agent stopped: too many tool rounds)")

    def _truncate_tool_result(self, text: str) -> str:
        if len(text) <= MAX_TOOL_RESULT_CHARS:
            return text
        omitted = len(text) - MAX_TOOL_RESULT_CHARS
        head = int(MAX_TOOL_RESULT_CHARS * 0.66)
        tail = MAX_TOOL_RESULT_CHARS - head
        return "\n".join(
            [
                text[:head],
                "",
                f"...[output truncated: omitted {omitted} chars for context safety]...",
                "",
                text[-tail:],
            ]
        )

    def _normalize_tool_output(self, output: Any) -> tuple[str, Any | None]:
        if isinstance(output, dict):
            text = output.get("text")
            details = output.get("details")
            if isinstance(text, str):
                return text, details
            return json.dumps(output, ensure_ascii=False), details
        if isinstance(output, str):
            return output, None
        return str(output), None

    def _pop_steering_message(self) -> str | None:
        with self._queue_lock:
            if not self._steering_queue:
                return None
            return self._steering_queue.pop(0)

    def _pop_follow_up_message(self) -> str | None:
        with self._queue_lock:
            if not self._follow_up_queue:
                return None
            return self._follow_up_queue.pop(0)

    def _append_queued_user_message(self, content: str, *, source: str, round_no: int) -> None:
        self._emit_event({"type": "message_start", "role": "user", "source": source, "round": round_no})
        self.session.add_user_message(content)
        self.store.save(self.session)
        self._emit_event(
            {
                "type": "message_end",
                "role": "user",
                "source": source,
                "round": round_no,
                "text_preview": _truncate(content, 200),
            }
        )

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            # Event handlers are best-effort and should not break agent execution.
            return

    def _finish_run(self, text: str) -> str:
        self._emit_event({"type": "agent_end", "final_text": text})
        return text

    def _create_provider(self, provider: str) -> Provider:
        if provider == "anthropic":
            return AnthropicProvider(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url=os.getenv("ANTHROPIC_BASE_URL"),
            )
        return OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def _runtime_tool_hooks(self) -> dict[str, Callable[..., Any]]:
        if not self.enable_orchestration:
            return {}
        return {
            "sessions_spawn": self._tool_sessions_spawn,
            "subagents": self._tool_subagents,
        }

    def _extra_tool_handlers(self) -> dict[str, Callable[..., Any]]:
        out: dict[str, Callable[..., Any]] = {}
        for t in self.extra_tools:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            handler = t.get("handler")
            if isinstance(name, str) and callable(handler):
                out[name] = handler
        return out

    def _on_lane_metrics(self, wait_ms: float, run_ms: float) -> None:
        warn_wait_ms = float(os.getenv("AGENT_LANE_WARN_WAIT_MS", "1200"))
        if wait_ms >= warn_wait_ms:
            self.session.add_system_event(
                f"Lane wait detected: waited={wait_ms:.0f}ms run={run_ms:.0f}ms session={self.session.meta.session_id}"
            )
            self.store.save(self.session)

    def _tool_sessions_spawn(
        self,
        task: str,
        provider: str | None = None,
        model: str | None = None,
        run_now: bool = True,
        background: bool = True,
    ) -> str:
        max_depth = max(0, int(os.getenv("AGENT_SUBAGENT_MAX_DEPTH", "2")))
        if self.session.meta.subagent_depth >= max_depth:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"subagent depth limit reached ({self.session.meta.subagent_depth}/{max_depth}); "
                        "increase AGENT_SUBAGENT_MAX_DEPTH to allow deeper nesting"
                    ),
                },
                ensure_ascii=False,
            )
        run = self.subagent_registry.spawn(
            parent_session_id=self.session.meta.session_id,
            task=task,
            provider=(provider or self.provider_name).strip().lower(),
            model=(model or self.model).strip(),
        )
        payload: dict[str, object] = {
            "status": "ok",
            "run_id": run.run_id,
            "child_session_id": run.child_session_id,
            "provider": run.provider,
            "model": run.model,
            "depth": self.session.meta.subagent_depth + 1,
        }
        self.session.add_system_event(
            f"Spawned subagent run={run.run_id} child_session={run.child_session_id} depth={self.session.meta.subagent_depth + 1}"
        )
        self.store.save(self.session)
        if run_now:
            if background:
                self._start_subagent_background(run.run_id, run.child_session_id, run.provider, run.model, task)
                payload["dispatched"] = "background"
            else:
                child = Agent(
                    provider=run.provider,
                    model=run.model,
                    session_id=run.child_session_id,
                    workspace_dir=self.workspace_dir,
                    sessions_dir=str(self.store.sessions_dir),
                    enable_orchestration=False,
                    parent_session_id=self.session.meta.session_id,
                    subagent_depth=self.session.meta.subagent_depth + 1,
                )
                self.subagent_registry.set_running(run.run_id)
                first_reply = child.chat(task)
                self.subagent_registry.set_completed(run.run_id, reply=first_reply)
                payload["first_reply"] = _truncate(first_reply, 1200)
                self.session.add_system_event(f"Subagent run={run.run_id} completed initial task.")
                self.store.save(self.session)
        return json.dumps(payload, ensure_ascii=False)

    def _tool_subagents(
        self,
        action: str,
        run_id: str | None = None,
        message: str | None = None,
        background: bool = False,
    ) -> str:
        normalized = (action or "").strip().lower()
        if normalized == "list":
            runs = self.subagent_registry.list(parent_session_id=self.session.meta.session_id)
            rows = [
                {
                    "run_id": r.run_id,
                    "child_session_id": r.child_session_id,
                    "status": r.status,
                    "task": _truncate(r.task, 120),
                    "created_at": r.created_at,
                    "updated_at": r.updated_at,
                    "provider": r.provider,
                    "model": r.model,
                    "last_reply": _truncate(r.last_reply, 180) if r.last_reply else None,
                    "last_error": _truncate(r.last_error, 180) if r.last_error else None,
                    "is_running_now": (
                        _GLOBAL_SUBAGENT_RUNTIME.is_running(r.run_id) if r.status != "killed" else False
                    ),
                }
                for r in runs
            ]
            return json.dumps({"status": "ok", "runs": rows}, ensure_ascii=False)

        if not run_id:
            return json.dumps(
                {"status": "error", "error": "run_id is required for this action"},
                ensure_ascii=False,
            )
        run = self.subagent_registry.get(run_id)
        if run is None:
            return json.dumps({"status": "error", "error": f"run not found: {run_id}"}, ensure_ascii=False)
        if run.parent_session_id != self.session.meta.session_id:
            return json.dumps({"status": "error", "error": "run does not belong to this session"}, ensure_ascii=False)

        if normalized == "get_result":
            running = _GLOBAL_SUBAGENT_RUNTIME.is_running(run.run_id)
            if run.status == "killed":
                running = False
            return json.dumps(
                {
                    "status": "ok",
                    "run_id": run.run_id,
                    "state": run.status,
                    "reply": run.last_reply,
                    "error": run.last_error,
                    "is_running_now": running,
                    "events": run.events,
                },
                ensure_ascii=False,
            )

        if normalized == "events":
            return json.dumps(
                {
                    "status": "ok",
                    "run_id": run.run_id,
                    "state": run.status,
                    "events": run.events,
                },
                ensure_ascii=False,
            )

        if normalized == "kill":
            _GLOBAL_SUBAGENT_RUNTIME.cancel(run_id)
            updated = self.subagent_registry.set_killed(run_id)
            self.session.add_system_event(f"Subagent run={run_id} marked as killed.")
            self.store.save(self.session)
            return json.dumps(
                {"status": "ok", "run_id": run_id, "new_status": updated.status if updated else "killed"},
                ensure_ascii=False,
            )

        if normalized == "steer":
            if run.status == "killed":
                return json.dumps({"status": "error", "error": f"run is not active: {run.status}"}, ensure_ascii=False)
            if not message or not message.strip():
                return json.dumps({"status": "error", "error": "message is required for steer"}, ensure_ascii=False)
            if background:
                # Guard: replace in-flight background task for the same run_id.
                _GLOBAL_SUBAGENT_RUNTIME.cancel(run.run_id)
                self._start_subagent_background(
                    run_id=run.run_id,
                    child_session_id=run.child_session_id,
                    provider=run.provider,
                    model=run.model,
                    task=message.strip(),
                )
                self.session.add_system_event(f"Subagent run={run_id} steered in background.")
                self.store.save(self.session)
                return json.dumps({"status": "ok", "run_id": run_id, "dispatched": "background"}, ensure_ascii=False)
            child = Agent(
                provider=run.provider,
                model=run.model,
                session_id=run.child_session_id,
                workspace_dir=self.workspace_dir,
                sessions_dir=str(self.store.sessions_dir),
                enable_orchestration=False,
                parent_session_id=run.parent_session_id,
                subagent_depth=self.session.meta.subagent_depth + 1,
            )
            self.subagent_registry.set_running(run_id)
            reply = child.chat(message.strip())
            self.subagent_registry.set_completed(run_id, reply=reply)
            self.session.add_system_event(f"Subagent run={run_id} steered with a new message.")
            self.store.save(self.session)
            return json.dumps({"status": "ok", "run_id": run_id, "reply": _truncate(reply, 2400)}, ensure_ascii=False)

        return json.dumps({"status": "error", "error": f"unknown action: {action}"}, ensure_ascii=False)

    def _complete_with_retry(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict],
        session_id: str,
        thinking_level: str,
        on_text_delta: Callable[[str], None] | None,
    ):
        max_retries = int(os.getenv("AGENT_MAX_RETRIES", "2"))
        base_delay_s = float(os.getenv("AGENT_RETRY_BASE_SECONDS", "1.0"))
        attempt = 0
        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise RuntimeError("cancelled")
            try:
                resolved_api_key = self.get_api_key(self.provider_name) if self.get_api_key is not None else None
                return self.provider.complete(
                    model=model,
                    messages=messages,
                    tools=tools,
                    session_id=session_id,
                    thinking_level=thinking_level,
                    on_text_delta=on_text_delta,
                    api_key=resolved_api_key,
                )
            except Exception as exc:
                transient = self._is_transient_error(exc)
                if (not transient) or attempt >= max_retries:
                    raise
                sleep_s = base_delay_s * (2**attempt)
                print(f"  [retry] transient model error, retrying in {sleep_s:.1f}s")
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise RuntimeError("cancelled")
                time.sleep(sleep_s)
                attempt += 1

    def _is_transient_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        transient_markers = [
            "timeout",
            "temporarily unavailable",
            "rate limit",
            "too many requests",
            "connection reset",
            "connection error",
            "502",
            "503",
            "504",
        ]
        return any(marker in text for marker in transient_markers)

    def _start_subagent_background(
        self,
        run_id: str,
        child_session_id: str,
        provider: str,
        model: str,
        task: str,
    ) -> None:
        # Guard: if the same run already has an in-flight background job, cancel it first.
        _GLOBAL_SUBAGENT_RUNTIME.cancel(run_id)

        timeout_sec = max(0, int(os.getenv("AGENT_SUBAGENT_RUN_TIMEOUT_SECONDS", "0") or 0))
        announce_completion = (os.getenv("AGENT_SUBAGENT_ANNOUNCE_COMPLETION") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        def _runner(cancel_event: threading.Event) -> None:
            timer: threading.Timer | None = None
            if timeout_sec > 0:

                def on_timeout() -> None:
                    cancel_event.set()
                    self.subagent_registry.set_failed(run_id, error="run timed out")
                    self._append_parent_system_event(f"Subagent run={run_id} timed out.")

                timer = threading.Timer(float(timeout_sec), on_timeout)
                timer.daemon = True
                timer.start()
            try:
                self.subagent_registry.set_running(run_id)
                child = Agent(
                    provider=provider,
                    model=model,
                    session_id=child_session_id,
                    workspace_dir=self.workspace_dir,
                    sessions_dir=str(self.store.sessions_dir),
                    enable_orchestration=False,
                    parent_session_id=self.session.meta.session_id,
                    subagent_depth=self.session.meta.subagent_depth + 1,
                    cancel_event=cancel_event,
                )
                reply = child.chat(task)
                if timer:
                    timer.cancel()
                if cancel_event.is_set():
                    # Timeout may have already set status to failed; do not overwrite.
                    run_after = self.subagent_registry.get(run_id)
                    if run_after is None or run_after.status != "failed":
                        self.subagent_registry.set_killed(run_id)
                        self._append_parent_system_event(f"Subagent run={run_id} cancelled before completion.")
                    return
                self.subagent_registry.set_completed(run_id, reply=reply)
                self._append_parent_system_event(f"Subagent run={run_id} completed in background.")
                if announce_completion and reply:
                    self._append_parent_completion_reply(run_id, reply)
            except Exception as exc:
                if timer:
                    timer.cancel()
                self.subagent_registry.set_failed(run_id, error=str(exc))
                self._append_parent_system_event(
                    f"Subagent run={run_id} failed in background: {_truncate(str(exc), 200)}"
                )
            finally:
                if timer:
                    timer.cancel()

        _GLOBAL_SUBAGENT_RUNTIME.submit(run_id, _runner)

    def _append_parent_system_event(self, message: str) -> None:
        parent_id = self.session.meta.session_id
        parent = self.store.load_or_create(
            session_id=parent_id,
            provider=self.provider_name,
            model=self.model,
            workspace_dir=self.workspace_dir,
            parent_session_id=self.parent_session_id,
            subagent_depth=self.subagent_depth,
        )
        parent.add_system_event(message)
        self.store.save(parent)

    def _append_parent_completion_reply(self, run_id: str, reply: str) -> None:
        """Append a human-readable assistant summary to the parent session when a subagent completes."""
        summary = _truncate(reply.strip(), 400)
        content = f"Subagent run={run_id} completed: {summary}"
        parent_id = self.session.meta.session_id
        parent = self.store.load_or_create(
            session_id=parent_id,
            provider=self.provider_name,
            model=self.model,
            workspace_dir=self.workspace_dir,
            parent_session_id=self.parent_session_id,
            subagent_depth=self.subagent_depth,
        )
        parent.add_assistant_message({"role": "assistant", "content": content})
        self.store.save(parent)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
