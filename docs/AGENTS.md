# AgentSpine Collaboration Rules

These rules apply to both human contributors and AI coding agents.

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
