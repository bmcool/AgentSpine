# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* Added `continue_run()` and `continue_run_stream()` to resume a session when the last message is `user` or `tool` without appending a new user message.
* Added optional `transform_context` and `convert_to_llm` hooks to split context shaping from final LLM message conversion.
* Added optional `tool_execution_update` events for progress-aware custom tools via `on_progress`.
* Added `docs/EVENTS.md` as the canonical event contract reference.

### Changed

* Steering now appends explicit skipped tool results for remaining tool calls when an interrupt message is queued.
* Custom extra tool handlers can raise exceptions, which are recorded as explicit tool error results by the agent.

## [0.1.0] - 2025-02-19

### Added

* Initial release.
* Reactive loop with OpenAI and Anthropic providers.
* Session-based isolation with JSONL persistence.
* Built-in tools: read_file, write_file, list_directory, run_cmd, web_fetch, sessions_spawn, subagents.
* Lane queue for per-session serialization and concurrent sessions.
* Context trimming and compaction (chars or estimated tokens).
* Streaming, steer/follow_up message queueing, and lifecycle event callbacks.
* CLI entry point (`main.py` and `agentspine` console script).

[Unreleased]: https://github.com/bmcool/AgentSpine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bmcool/AgentSpine/releases/tag/v0.1.0
