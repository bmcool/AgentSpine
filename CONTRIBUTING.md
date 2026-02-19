# Contributing to AgentSpine

Thank you for your interest in contributing. This document outlines how to get set up and submit changes.

## Development setup

```bash
git clone https://github.com/bmcool/AgentSpine.git
cd AgentSpine
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys for local runs
```

Optional: install [pre-commit](https://pre-commit.com/) hooks so lint and format run before each commit:

```bash
pre-commit install
```

## Running tests

```bash
pytest
# or with coverage:
pytest --cov=src --cov=main --cov-report=term-missing
```

Tests use a fake provider and do not call real LLM APIs.

On Unix you can use the [Makefile](Makefile): `make test`, `make cov`, `make lint`, `make format`, `make check`. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for all commands.

## Code style

- Format and lint with [Ruff](https://docs.astral.sh/ruff/):
  ```bash
  ruff check src main.py tests
  ruff format src main.py tests
  ```
- Use type hints for public APIs.
- Prefer the existing patterns in the codebase (e.g. `src/agent.py`, `src/tools.py`).

## Submitting changes

1. **Open an issue** (optional but recommended for larger changes) to discuss the idea.
2. **Fork the repo** and create a branch from `main`.
3. **Make your changes** and add or update tests as needed.
4. **Run tests and lint**: `pytest` and `make check` (or `ruff check src main.py tests && ruff format --check src main.py tests`).
5. **Commit** with clear messages (e.g. `Add X`, `Fix Y`).
6. **Open a Pull Request** against `main`. Describe what changed and why; link any related issue.

## Pull request guidelines

- Keep PRs focused. Prefer several small PRs over one large one.
- Ensure CI passes (tests and lint).
- Update `CHANGELOG.md` under `[Unreleased]` for user-facing changes.

## Changelog rules

- Add new entries only under `## [Unreleased]`.
- Use these sections when applicable: `### Breaking Changes`, `### Added`, `### Changed`, `### Fixed`, `### Removed`.
- Do not modify already released sections (for example `## [0.1.0]`).
- When referencing tracked work, use linked attribution:
  - Internal changes: `Fixed ... ([#123](https://github.com/bmcool/AgentSpine/issues/123))`
  - External contribution: `Added ... ([#456](https://github.com/bmcool/AgentSpine/pull/456) by [@user](https://github.com/user))`

## Automation and AI-agent rules

- Follow `docs/AGENTS.md` for tool usage and Git safety requirements.
- Always read a file before editing it.
- Stage only task-specific files; do not use sweeping add/reset/clean/stash commands.

## Questions

Open a [Discussion](https://github.com/bmcool/AgentSpine/discussions) or an [Issue](https://github.com/bmcool/AgentSpine/issues).
