# AgentSpine Collaboration Rules

These rules apply to both human contributors and AI coding agents.

## First message workflow

- If the first user message is not a concrete coding task, read `README.md` first.
- Then ask which module to work on (`agent.py`, `tools.py`, `providers/`, `session*`, `context_*`, etc.).
- Before changing code, read the target files fully and confirm scope.

## Tooling and file operations

- Do not use `sed`/`cat` style commands for source inspection in automated agent flows.
- Always read a file before editing it.
- For large files, prefer ranged reads with explicit offsets/limits.

## Git safety

- Only stage files you changed in the current task.
- Do not use sweeping staging commands such as `git add -A` or `git add .`.
- Do not use destructive commands that can discard uncommitted work:
  - `git reset --hard`
  - `git checkout .`
  - `git clean -fd`
  - `git stash`
- Never use `--no-verify` when committing.

## Commit and issue linkage

- Keep commit messages concise and in English.
- When a change resolves a tracked issue, include `fixes #<id>` or `closes #<id>` in the commit message body.

## Code quality baseline

- Follow existing patterns in `src/agent.py` and `src/tools.py`.
- Keep public interfaces type-annotated.
- Avoid broad refactors unrelated to the requested task.

## Adding a new LLM provider

Use this checklist to keep provider additions consistent:

1. Implement the provider class against `Provider` in `src/providers/base.py` (`name`, `complete`).
2. Register provider selection in `src/agent.py` (provider routing and default model behavior).
3. Add/update required environment variables in `.env.example`.
4. Update `README.md` configuration and quick-start examples.
5. Add or update provider tests in `tests/` (prefer deterministic/mock coverage).

## Parallel-agent git safety

- Stage only files you changed in this task.
- Use explicit staging (`git add <file1> <file2> ...`), not sweeping staging.
- Do not run commands that can drop unrelated uncommitted work:
  - `git reset --hard`
  - `git checkout .`
  - `git clean -fd`
  - `git stash`
