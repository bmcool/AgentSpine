# Tool UI Contract

This document defines a practical display contract for tool outputs in CLI or web UIs.

AgentSpine core supports the following tool execution result types:

- `str`
- `{"text": str, "details"?: Any}`

See `ToolExecutionResult` in [`src/tools.py`](../src/tools.py).

## Why this contract exists

- Keep core execution generic.
- Let UI consumers render richer outputs without coupling UI logic to core internals.
- Reuse `details` data through `tool_execution_end` events.

## `details` payload guidelines

`details` is optional and tool-specific. A suggested shape:

```python
{
    "kind": "file" | "command" | "fetch" | "subagent" | "custom",
    "summary": "Short human-readable line",
    # optional fields by kind:
    "path": "...",
    "cwd": "...",
    "url": "...",
    "status_code": 200,
    "exit_code": 0,
    "mime_type": "text/plain",
}
```

You can extend this as needed. Consumers should ignore unknown keys.

## Event mapping

When a tool returns structured output, the `details` object is forwarded to
`tool_execution_end` as `event["details"]`.

This enables UI renderers to use either:

- `tool_name` (for hardcoded tool-specific renderers), or
- `details.kind` (for schema-first generic renderers).

## Rendering recommendations

- **`read_file` / `write_file`**: show `details.path` and truncated content/summary.
- **`run_cmd`**: show command summary + `details.exit_code`.
- **`web_fetch`**: show URL + status code.
- **`sessions_spawn` / `subagents`**: show run id and state transitions.
- Fallback to `result_preview` for unknown tools.

## Minimal event consumer example

```python
def handle_event(event: dict) -> None:
    if event.get("type") != "tool_execution_end":
        return

    tool = event.get("tool_name", "unknown")
    details = event.get("details") or {}
    kind = details.get("kind")
    path = details.get("path")
    exit_code = details.get("exit_code")
    preview = event.get("result_preview", "")

    if path:
        print(f"[tool:{tool}] kind={kind} path={path}")
        return
    if exit_code is not None:
        print(f"[tool:{tool}] kind={kind} exit_code={exit_code}")
        return
    print(f"[tool:{tool}] {preview}")
```
