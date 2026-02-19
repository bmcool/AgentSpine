# AgentSpine Examples

This folder contains two demos that show what AgentSpine can do today.

## 1) Mock demo (no API keys required)

```bash
python examples/demo_mock.py
```

What it demonstrates:

- Reactive loop with deterministic tool calls
- Event stream logging via `on_event`
- Steering behavior (`steer`) and skipped tool results
- Follow-up queue behavior (`follow_up`)
- Tool progress updates (`tool_execution_update`)
- Continuing a run without `chat()` (`continue_run`)

Use this first if you want to understand behavior without model/API variability.

## 2) Real provider demo (requires API keys)

```bash
python examples/demo_real.py --provider openai --model gpt-4o
# or
python examples/demo_real.py --provider anthropic --model claude-3-5-sonnet-20241022
```

Optional flags:

- `--no-stream`: disable streaming output
- `--prompt "..."`
- `--continue-prompt "..."`

This demo shows the same core runtime shape against real model responses.

## Notes

- Run these commands from the repository root.
- For real-provider mode, ensure `.env` exists and contains matching provider keys.
