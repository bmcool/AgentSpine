# Observability Guide

This guide explains how to build trace-like telemetry from AgentSpine's `on_event` callback
without changing core runtime behavior.

For the canonical event schema, see [EVENTS.md](./EVENTS.md).

## Event-to-trace mapping

A single `chat()` / `chat_stream()` run can be represented as one trace:

- **Trace root**: `agent_start` -> `agent_end`
- **Turn spans**: each `turn_start` -> matching `turn_end` (`round` is the span key)
- **Message events**: `message_start` / `message_update` / `message_end` attached to the current turn
- **Tool spans**: each `tool_execution_start` -> `tool_execution_end` (`tool_call_id` is the span key)

Recommended identity strategy:

- `trace_id`: runtime-generated id per run (for example `uuid4().hex`)
- `run_key`: optional logical key such as `"{session_id}:{trace_id}"`
- `turn_span_id`: `f"turn:{round}"`
- `tool_span_id`: `f"tool:{tool_call_id}"`

## Suggested exported shape

Use any structure that is easy for your UI and storage. A practical baseline:

```python
{
    "trace_id": "d57a...",
    "session_id": "demo-1",
    "started_at": 1730000000.0,
    "ended_at": 1730000002.1,
    "turns": [
        {
            "round": 1,
            "status": "tool_calls_processed",
            "messages": [...],
            "tools": [...],
        }
    ],
    "raw_events": [...],  # optional
}
```

## Minimal in-memory tracer example

```python
from __future__ import annotations

import time
import uuid
from typing import Any


class BasicEventTracer:
    def __init__(self, session_id: str | None = None) -> None:
        self.trace_id = uuid.uuid4().hex
        self.session_id = session_id
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.turns: dict[int, dict[str, Any]] = {}
        self.tool_index: dict[str, tuple[int, dict[str, Any]]] = {}
        self.raw_events: list[dict[str, Any]] = []

    def on_event(self, event: dict[str, Any]) -> None:
        self.raw_events.append(event)
        typ = event.get("type")

        if typ == "turn_start":
            round_no = int(event["round"])
            self.turns.setdefault(
                round_no,
                {"round": round_no, "status": "started", "messages": [], "tools": []},
            )
            return

        if typ in {"message_start", "message_update", "message_end"}:
            round_no = int(event.get("round", -1))
            turn = self.turns.setdefault(round_no, {"round": round_no, "status": "started", "messages": [], "tools": []})
            turn["messages"].append(event)
            return

        if typ == "tool_execution_start":
            round_no = int(event["round"])
            turn = self.turns.setdefault(round_no, {"round": round_no, "status": "started", "messages": [], "tools": []})
            span = {
                "tool_call_id": event.get("tool_call_id"),
                "tool_name": event.get("tool_name"),
                "started": time.time(),
                "updates": [],
                "end": None,
            }
            turn["tools"].append(span)
            tool_call_id = str(event.get("tool_call_id"))
            self.tool_index[tool_call_id] = (round_no, span)
            return

        if typ == "tool_execution_update":
            tool_call_id = str(event.get("tool_call_id"))
            hit = self.tool_index.get(tool_call_id)
            if hit:
                _, span = hit
                span["updates"].append(event.get("partial", ""))
            return

        if typ == "tool_execution_end":
            tool_call_id = str(event.get("tool_call_id"))
            hit = self.tool_index.get(tool_call_id)
            if hit:
                _, span = hit
                span["end"] = event
            return

        if typ == "turn_end":
            round_no = int(event["round"])
            turn = self.turns.setdefault(round_no, {"round": round_no, "status": "started", "messages": [], "tools": []})
            turn["status"] = event.get("status", "completed")
            return

        if typ == "agent_end":
            self.ended_at = time.time()

    def export_trace(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "turns": [self.turns[k] for k in sorted(self.turns)],
            "raw_events": self.raw_events,
        }
```

## Usage with `Agent`

```python
from src.agent import Agent

tracer = BasicEventTracer(session_id="demo-1")
agent = Agent(session_id="demo-1", on_event=tracer.on_event)
final_text = agent.chat("Summarize my workspace files.")
trace_payload = tracer.export_trace()
```

## Notes

- `on_event` handler errors are ignored by core; keep handlers robust and low-latency.
- Treat event payloads as forward-compatible and ignore unknown keys.
- If you persist traces, avoid storing oversized `message_update` streams without truncation.
