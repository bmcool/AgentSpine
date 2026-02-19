"""Tests for PromptBuilder."""

from __future__ import annotations

import unittest

from src.prompt_builder import PromptBuilder


class PromptBuilderTests(unittest.TestCase):
    def test_build_contains_identity_section(self) -> None:
        pb = PromptBuilder()
        out = pb.build(
            provider="openai",
            model="gpt-4o",
            workspace_dir="/tmp",
            tool_summaries=[("read_file", "Read a file.")],
        )
        self.assertIn("## Identity", out)
        self.assertIn("reactive coding agent", out)

    def test_build_contains_tooling_section(self) -> None:
        pb = PromptBuilder()
        out = pb.build(
            provider="openai",
            model="gpt-4o",
            workspace_dir="/tmp",
            tool_summaries=[("read_file", "Read a file.")],
        )
        self.assertIn("## Tooling", out)
        self.assertIn("read_file", out)
        self.assertIn("Read a file.", out)

    def test_build_contains_workspace_and_runtime(self) -> None:
        pb = PromptBuilder()
        out = pb.build(
            provider="anthropic",
            model="claude-3-5-sonnet",
            workspace_dir="/tmp",
            tool_summaries=[],
        )
        self.assertIn("## Workspace and Runtime", out)
        self.assertIn("anthropic", out)
        self.assertIn("claude-3-5-sonnet", out)

    def test_build_contains_safety_section(self) -> None:
        pb = PromptBuilder()
        out = pb.build(
            provider="openai",
            model="gpt-4o",
            workspace_dir="/tmp",
            tool_summaries=[],
        )
        self.assertIn("## Safety", out)


if __name__ == "__main__":
    unittest.main()
