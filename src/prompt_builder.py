from __future__ import annotations

import os
import platform
from pathlib import Path


class PromptBuilder:
    def __init__(self, *, max_tool_output_chars: int = 8000) -> None:
        self.max_tool_output_chars = max_tool_output_chars

    def build(
        self,
        *,
        provider: str,
        model: str,
        workspace_dir: str,
        tool_summaries: list[tuple[str, str]],
    ) -> str:
        sections: list[str] = []
        sections.extend(self._identity_section())
        sections.extend(self._tooling_section(tool_summaries))
        sections.extend(self._workspace_runtime_section(provider, model, workspace_dir))
        sections.extend(self._safety_section())
        return "\n".join(sections).strip()

    def _identity_section(self) -> list[str]:
        return [
            "## Identity",
            "You are a reactive coding agent.",
            "Work step-by-step with tools and return concise final answers.",
            "",
        ]

    def _tooling_section(self, tool_summaries: list[tuple[str, str]]) -> list[str]:
        lines = [
            "## Tooling",
            "Use tools when file or shell operations are needed.",
            "Prefer reading before writing and avoid guessing file paths.",
            "Available tools:",
        ]
        for name, description in tool_summaries:
            lines.append(f"- {name}: {description}")
        lines.append("")
        return lines

    def _workspace_runtime_section(self, provider: str, model: str, workspace_dir: str) -> list[str]:
        cwd = str(Path(workspace_dir).resolve())
        return [
            "## Workspace and Runtime",
            f"- Workspace root: {cwd}",
            f"- Provider/model: {provider}/{model}",
            f"- OS: {platform.system()} {platform.release()}",
            f"- Python: {platform.python_version()}",
            f"- Current working directory: {os.getcwd()}",
            "",
        ]

    def _safety_section(self) -> list[str]:
        return [
            "## Safety",
            "- For destructive actions, explain intent clearly before executing.",
            "- Keep command outputs concise and summarize key results.",
            f"- If a tool output is very long, keep the most relevant parts (target <= {self.max_tool_output_chars} chars).",
            "",
        ]
