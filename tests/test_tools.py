"""Tests for tools (execute_tool, get_tool_definitions, get_tool_summaries)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.tools import execute_tool, get_tool_definitions, get_tool_summaries


class ExecuteToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)

    def test_read_file(self) -> None:
        path = self.workspace / "hello.txt"
        path.write_text("hello world", encoding="utf-8")
        result = execute_tool("read_file", json.dumps({"path": str(path)}))
        self.assertIn("hello world", result)

    def test_write_file(self) -> None:
        path = self.workspace / "out.txt"
        result = execute_tool("write_file", json.dumps({"path": str(path), "content": "written"}))
        self.assertIn("wrote", result.lower())
        self.assertEqual(path.read_text(encoding="utf-8"), "written")

    def test_list_directory(self) -> None:
        (self.workspace / "a").write_text("")
        (self.workspace / "b").mkdir()
        result = execute_tool("list_directory", json.dumps({"path": str(self.workspace)}))
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_run_cmd(self) -> None:
        result = execute_tool(
            "run_cmd",
            json.dumps({"command": "echo ok", "cwd": str(self.workspace)}),
        )
        self.assertIn("ok", result)

    def test_unknown_tool_returns_error(self) -> None:
        result = execute_tool("nonexistent_tool", "{}")
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("nonexistent_tool", result)

    def test_invalid_json_arguments_returns_error(self) -> None:
        result = execute_tool("read_file", "not json")
        self.assertTrue(result.startswith("Error:"))

    def test_extra_handler_accepts_on_progress(self) -> None:
        updates: list[str] = []

        def progress_handler(value: str, on_progress=None) -> str:
            if on_progress is not None:
                on_progress(f"update:{value}")
            return "ok"

        result = execute_tool(
            "custom_progress",
            json.dumps({"value": "v"}),
            extra_handlers={"custom_progress": progress_handler},
            on_progress=updates.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(updates, ["update:v"])

    def test_extra_handler_exception_propagates(self) -> None:
        def failing_handler() -> str:
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            execute_tool(
                "failing_custom",
                "{}",
                extra_handlers={"failing_custom": failing_handler},
            )


class GetToolDefinitionsTests(unittest.TestCase):
    def test_base_tools_without_orchestration(self) -> None:
        tools = get_tool_definitions(include_orchestration=False)
        names = [t["function"]["name"] for t in tools if t.get("type") == "function" and "function" in t]
        self.assertIn("read_file", names)
        self.assertIn("run_cmd", names)
        self.assertNotIn("sessions_spawn", names)

    def test_includes_orchestration_when_requested(self) -> None:
        tools = get_tool_definitions(include_orchestration=True)
        names = [t["function"]["name"] for t in tools if t.get("type") == "function" and "function" in t]
        self.assertIn("sessions_spawn", names)
        self.assertIn("subagents", names)


class GetToolSummariesTests(unittest.TestCase):
    def test_returns_list_of_tuples(self) -> None:
        summaries = get_tool_summaries(include_orchestration=False)
        self.assertIsInstance(summaries, list)
        for item in summaries:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], str)


if __name__ == "__main__":
    unittest.main()
