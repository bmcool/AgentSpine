# AgentSpine Event Contract

Events are emitted by the core during `chat()` / `chat_stream()` when `on_event` is set. Payloads are plain dicts; all events include `"type"`. Ordering is guaranteed as below.

**Reference:** Borrowed from pi-agent-core's event flow; this document is the single source of truth for channel implementations.

## Event order (one run)

1. `agent_start`
2. For each round:
   - `turn_start`
   - `message_start` (user or assistant)
   - [If assistant and streaming] zero or more `message_update`
   - `message_end`
   - [If tool calls] for each tool: `tool_execution_start` → zero or more `tool_execution_update` → `tool_execution_end` (then possibly `message_start` / `message_end` for injected user message if steer). When steer interrupts a tool batch, remaining tools emit `tool_execution_start` / `tool_execution_end` with `skipped: true`.
   - `turn_end`
3. `agent_end`

## Event types and payloads

| type | When | Payload (all keys optional unless noted) |
|------|------|------------------------------------------|
| `agent_start` | Start of run | (no extra keys) |
| `turn_start` | Start of a turn (one LLM call + tool run) | `round: int` |
| `message_start` | A message begins | `role: "user" \| "assistant"`, `round: int`, `source?: str` (e.g. `"follow_up"`, `"steer"` for user) |
| `message_update` | Assistant stream chunk (streaming only) | `role: "assistant"`, `delta: str` |
| `message_end` | Message complete | `role: "user" \| "assistant"`, `round: int`, `text_preview?: str`, `source?: str` |
| `tool_execution_start` | Tool is about to run | `round: int`, `tool_call_id: str`, `tool_name: str`, `args: str` (JSON string) |
| `tool_execution_update` | Tool streams progress (optional) | `round: int`, `tool_call_id: str`, `tool_name: str`, `partial: str` |
| `tool_execution_end` | Tool finished | `round: int`, `tool_call_id: str`, `tool_name: str`, `result_preview: str`, `details?: Any`, `skipped?: bool` |
| `turn_end` | Turn finished | `round: int`, `status: str` (e.g. `"completed"`, `"tool_calls_processed"`, `"steered"`, `"follow_up_injected"`, `"cancelled"`, `"loop_detected"`), `tool_calls_count?: int`, `assistant_message_preview?: str`, `tool_results_preview?: list[str]` |
| `agent_end` | Run finished | `final_text: str` |

## Handler contract

- The core calls `on_event(event)` only when `on_event` is not `None`.
- Exceptions in `on_event` are caught and ignored; they must not affect the run.
- Payloads may contain additional keys in the future; consumers should ignore unknown keys.
