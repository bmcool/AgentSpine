# Development Guide

Short guide for contributors and maintainers.

## Setup

```bash
git clone https://github.com/bmcool/AgentSpine.git
cd AgentSpine
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional: for running main.py with real API keys
```

## Commands

| Task | Command (Unix) | Command (Windows / no make) |
|------|----------------|------------------------------|
| Install | `make install` | `pip install -e ".[dev]"` |
| Tests | `make test` | `pytest` |
| Tests + coverage | `make cov` | `pytest --cov=src --cov=main --cov-report=term-missing` |
| Lint | `make lint` | `ruff check src main.py tests` |
| Format | `make format` | `ruff format src main.py tests` |
| Lint + format check (CI) | `make check` | `ruff check src main.py tests && ruff format --check src main.py tests` |

## Pre-commit (optional)

Install hooks so lint/format run before each commit:

```bash
pre-commit install
```

Then `git commit` will run ruff and pre-commit-hooks. To run manually:

```bash
pre-commit run --all-files
```

## Project layout

- `main.py` — CLI entry; `agentspine` console script points here.
- `src/` — Core package: `agent.py` (reactive loop), `session*.py`, `context_*.py`, `tools.py`, `providers/`.
- `tests/` — Unit tests; use fake provider, no real API calls.
- `docs/` — Extra documentation (this file, etc.).

## Releasing

1. Bump version in `pyproject.toml` and `src/__init__.py`.
2. Move `[Unreleased]` changes in `CHANGELOG.md` into a new `[X.Y.Z]` section; add link.
3. Commit, tag: `git tag vX.Y.Z`, push: `git push origin vX.Y.Z`.
4. Create a GitHub Release from the tag; paste the changelog section.
5. Optionally publish to PyPI: `pip install build twine && python -m build && twine upload dist/*`.

## Resources

- [CONTRIBUTING.md](../CONTRIBUTING.md) — How to contribute and PR guidelines.
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) — Community standards.
- [SECURITY.md](../SECURITY.md) — Reporting vulnerabilities.
