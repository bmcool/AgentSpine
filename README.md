# AgentSpine

[![CI](https://github.com/bmcool/AgentSpine/actions/workflows/ci.yml/badge.svg)](https://github.com/bmcool/AgentSpine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[Contributing](CONTRIBUTING.md) · [Development](docs/DEVELOPMENT.md) · [Code of Conduct](CODE_OF_CONDUCT.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

---

## Table of contents

- [Philosophy](#philosophy)
- [Core Concepts](#core-concepts)
- [Architecture Overview](#architecture-overview)
- [Design Goals](#design-goals)
- [Example Use Cases](#example-use-cases)
- [Why AgentSpine?](#why-agentspine)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Roadmap](#roadmap)
- [License](#license)
- [Vision](#vision)

---

**AgentSpine** is a composable backbone for building extensible, session-based AI agents.

It provides a minimal, reactive core that stays generic and flexible, while allowing capabilities, memory, tools, and workflows to grow outward without modifying the core engine.

---

## Philosophy

AgentSpine is built on one principle:

> Keep the core minimal. Let everything else grow around it.

The core is responsible only for:

* A reactive loop
* Session isolation
* Tool execution
* Provider abstraction

Everything else — roles, memory, RAG, workflows, channels — is layered on top.

This allows the same engine to power:

* Personal AI assistants
* Internal tooling agents
* Knowledge systems
* Customer support bots
* Vertical-specific AI applications

Without rewriting the foundation.

---

## Core Concepts

### Reactive Loop

Each interaction follows a simple cycle:

User input → LLM call (with tools available) → Tool calls (optional) → Tool execution → Results injected back into context → Repeat until response

The loop does not assume any business logic.

### Session-Based Isolation

Every conversation runs inside a session ID.

Sessions:

* Store conversation history
* Maintain state boundaries
* Enable per-user or per-task isolation

This allows:

* One AI per user
* One AI per workflow
* One AI per task

### Role = Configuration

A role consists of:

* System prompt
* Tool set
* Optional memory or state strategy

AgentSpine does not know what a "customer support agent" or "matchmaking agent" is.

It only runs a combination of:

prompt + tools + session

### Tools as Capabilities

All abilities are implemented as tools.

Examples include:

* File read and write
* Web fetch
* Knowledge search
* API calls
* Task execution
* Database access

The core:

* Registers tools
* Exposes them to the LLM
* Executes selected tools
* Returns results to the loop

No business logic is embedded in the engine.

### Channel Separation

AgentSpine is channel-agnostic.

Channels can include:

* REST APIs
* Slack bots
* Web applications
* Mobile backends
* CLI environments

Channels only send messages into a session and retrieve responses.

The engine remains independent.

---

## Architecture Overview

Channel → Session Manager → Reactive Loop → LLM Provider (pluggable) → Tool Execution Layer → Optional Modules (Memory / RAG / Workflow)

The spine stays stable. Capabilities extend outward.

---

## Design Goals

* Minimal core
* Business-agnostic architecture
* Pluggable tools
* Swappable LLM providers (OpenAI, Anthropic out of the box)
* Session-based isolation
* Composable structure
* Long-term extensibility

---

## Example Use Cases

AgentSpine can power:

### Matchmaking AI

* User preference tools
* Interaction history memory
* Recommendation logic

### Internal Developer Agent

* File system tools
* Deployment tools
* Workspace context

### Knowledge Assistant

* RAG search
* Summarization tools
* Document indexing

### Customer Support Agent

* Order lookup tools
* Escalation APIs
* Multi-stage workflows

All using the same core.

---

## Why AgentSpine?

Most agent frameworks tightly couple logic, tools, and workflows.

AgentSpine separates them.

It gives you:

* A stable backbone
* Infinite growth surface
* Clear architectural boundaries

---

## Installation

```bash
pip install -e ".[dev]"   # from repo root, for development
# or
pip install agentspine   # when published to PyPI
```

Then use the CLI as `agentspine` or `python main.py` (see Quick start below).

## Getting Started

### Quick start

```bash
git clone https://github.com/bmcool/AgentSpine.git
cd AgentSpine

# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API keys
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and/or ANTHROPIC_API_KEY

# 3. Run
python main.py --provider openai --model gpt-4o
python main.py --provider anthropic --model claude-3-5-sonnet-20241022
python main.py --provider openai --model gpt-4o --session demo-1
```

### Configuration (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | (required for OpenAI) |
| `OPENAI_MODEL` | Model to use | `gpt-4o` |
| `OPENAI_BASE_URL` | Custom API base URL (for compatible providers) | (OpenAI default) |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | (required for Anthropic) |
| `ANTHROPIC_MODEL` | Anthropic model | `claude-3-5-sonnet-20241022` |
| `AGENT_PROVIDER` | Default provider | `openai` |
| `AGENT_THINKING_LEVEL` | Reasoning level (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`) | `off` |
| `AGENT_MAX_RETRIES` | Transient error retry count | `2` |
| `AGENT_RETRY_BASE_SECONDS` | Exponential backoff base seconds | `1.0` |
| `AGENT_MAX_CONCURRENT` | Max concurrent lane executions | `4` |
| `AGENT_LANE_WARN_WAIT_MS` | Emit lane wait system event when exceeded | `1200` |
| `AGENT_SUBAGENT_MAX_DEPTH` | Maximum subagent nesting depth | `2` |
| `AGENT_SUBAGENT_MAX_WORKERS` | Background subagent worker pool size | `2` |
| `AGENT_SUBAGENT_RUN_TIMEOUT_SECONDS` | Background run timeout (0 = no limit) | `0` |
| `AGENT_SUBAGENT_ANNOUNCE_COMPLETION` | When set, append assistant summary to parent on background completion | `0` |
| `AGENT_CONTEXT_MODE` | Context limit by `chars` or `tokens` (heuristic, no extra deps) | `chars` |
| `AGENT_MAX_CHARS` / `AGENT_MAX_TOKENS` | Hard cap for history (when mode is chars / tokens) | `24000` |
| `AGENT_COMPACT_TRIGGER_CHARS` / `AGENT_COMPACT_TRIGGER_TOKENS` | Trigger compaction above this size | `36000` |
| `AGENT_KEEP_LAST_MESSAGES` | Keep at least this many recent messages | `30` |
| `AGENT_COMPACT_KEEP_TAIL` | After compaction, keep this many messages + summary | `16` |

### Extending the core

* **Workspace**: Pass any `workspace_dir` (defaults to cwd) for prompt context and tool cwd.
* **Custom tools**: Pass `extra_tools=[{"name": "...", "definition": {...}, "handler": fn}]` to `Agent(...)`. Each `definition` must be OpenAI-style. The handler receives keyword args and returns a string. Custom handlers may raise exceptions (recorded as tool errors) and can optionally accept `on_progress` to stream progress updates.
* **Minimal footprint**: Use `enable_orchestration=False` to disable `sessions_spawn` and `subagents`; only base tools (read_file, write_file, list_directory, run_cmd, web_fetch) are exposed.
* **Context**: Set `AGENT_CONTEXT_MODE=tokens` to cap history by estimated tokens (heuristic, no tiktoken).
* **Runtime steering**: Call `agent.steer("...")` or `agent.follow_up("...")` from another thread while the run is active.
* **Context transform**: Pass `transform_context=callable` to mutate session history before context compaction and request preparation.
* **LLM conversion**: Pass `convert_to_llm=callable` to mutate/filter final messages sent to the provider.
* **Pre-LLM transform (legacy)**: `transform_messages_for_llm=callable` remains supported for compatibility.

### Built-in tools

| Tool | Description |
|------|-------------|
| `read_file` | Read the full contents of a file |
| `write_file` | Create or overwrite a file |
| `list_directory` | List files and subdirectories |
| `run_cmd` | Execute a shell command |
| `web_fetch` | Fetch a URL (http/https) and return body as text |
| `sessions_spawn` | Spawn a subagent session |
| `subagents` | List / steer / kill subagent runs |

### Project structure

```
AgentSpine/
  main.py            # CLI entry point
  requirements.txt   # Python dependencies
  .env.example       # Environment variable template (copy to .env)
  .gitignore
  LICENSE            # MIT
  sessions/          # JSONL session files (auto-created)
  src/
    session.py       # Session model (metadata + messages)
    session_store.py # JSONL persistence
    lane_queue.py    # Lane-based serialization/concurrency
    subagent_registry.py
    context_estimate.py
    context_manager.py # History trimming + compaction
    prompt_builder.py
    tools.py         # Tool definitions + execution
    providers/       # OpenAI / Anthropic adapters
    agent.py         # Reactive loop
  tests/
```

### Streaming, steering, and events

* **Streaming**: Default mode streams assistant text in real time; use `--no-stream` for non-stream output.
* **Steer / follow_up**: `agent.steer(text)` injects an interrupt after the current tool; `agent.follow_up(text)` injects when the agent reaches a terminal turn. Use `clear_steering_queue()`, `clear_follow_up_queue()`, or `clear_all_queues()` as needed.
* **Continue**: `agent.continue_run()` retries from existing context without appending a new user message. `agent.continue_run_stream(on_text_delta=...)` provides streaming retries.
* **Events**: Pass `on_event` to `Agent(...)` for lifecycle events: `agent_start`, `agent_end`, `turn_start`, `turn_end`, `message_start`, `message_update`, `message_end`, `tool_execution_start`, `tool_execution_update`, `tool_execution_end`. `chat_stream(..., on_text_delta=...)` is also supported. Event types and payload shapes are documented in [docs/EVENTS.md](docs/EVENTS.md).

### Lanes and subagents

* Turns are serialized per `session_id`; different sessions can run concurrently (bounded by `AGENT_MAX_CONCURRENT`).
* `sessions_spawn` creates a child session; `subagents` supports `action=list`, `get_result`, `events`, `steer` (with `background=true`), and `kill`.
* Background runs respect `AGENT_SUBAGENT_MAX_DEPTH`, `AGENT_SUBAGENT_RUN_TIMEOUT_SECONDS`, and optional `AGENT_SUBAGENT_ANNOUNCE_COMPLETION`.

---

## Roadmap

* Structured memory modules
* Built-in RAG connectors
* Workflow phase engine
* Observability hooks
* Tool sandboxing
* Multi-agent orchestration layer

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Vision

AgentSpine is not another agent wrapper.

It is a backbone for building long-lived, extensible AI systems.

Build once.  
Extend forever.
