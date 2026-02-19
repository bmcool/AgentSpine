"""Tests for ContextManager."""

from __future__ import annotations

import unittest
from typing import Any

from src.context_manager import ContextManager


def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


class ContextManagerTests(unittest.TestCase):
    def test_prepare_messages_under_cap_returns_all(self) -> None:
        cm = ContextManager(
            max_chars=10_000,
            keep_last_messages=30,
            compact_trigger_chars=20_000,
            compact_keep_tail=16,
            mode="chars",
        )
        messages = [_msg("user", "a"), _msg("assistant", "b")]
        out, compacted = cm.prepare_messages(system_prompt="You are helpful.", history_messages=messages)
        self.assertFalse(compacted)
        # With system prepended
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["content"], "You are helpful.")
        self.assertEqual(len(out), 3)

    def test_prepare_messages_trims_to_keep_last(self) -> None:
        cm = ContextManager(
            max_chars=100_000,
            keep_last_messages=3,
            compact_trigger_chars=200_000,
            compact_keep_tail=16,
            mode="chars",
        )
        messages = [_msg("user", f"m{i}") for i in range(10)]
        out, _ = cm.prepare_messages(system_prompt="Sys", history_messages=messages)
        # system + last 3 history
        self.assertEqual(len(out), 4)

    def test_prepare_messages_compacts_when_over_trigger(self) -> None:
        cm = ContextManager(
            max_chars=50_000,
            keep_last_messages=30,
            compact_trigger_chars=100,  # low trigger
            compact_keep_tail=5,
            mode="chars",
        )
        messages = [_msg("user", "x" * 50) for _ in range(20)]
        out, compacted = cm.prepare_messages(system_prompt="Sys", history_messages=messages)
        self.assertTrue(compacted)
        # First history message should be summary (assistant role, compact text)
        first_history = out[1]
        self.assertEqual(first_history["role"], "assistant")
        self.assertIn("Compacted", first_history["content"])

    def test_prepare_messages_respects_char_cap(self) -> None:
        cm = ContextManager(
            max_chars=100,
            keep_last_messages=30,
            compact_trigger_chars=1000,
            compact_keep_tail=4,
            mode="chars",
        )
        messages = [_msg("user", "a" * 80) for _ in range(10)]
        out, _ = cm.prepare_messages(system_prompt="S", history_messages=messages)
        # Should trim so total is bounded (cap applies to history; system prepended)
        self.assertLessEqual(len(out), 5)  # system + at most a few history messages
        total_chars = sum(len(m.get("content", "")) for m in out)
        self.assertLessEqual(total_chars, 500)

    def test_from_env_returns_valid_manager(self) -> None:
        cm = ContextManager.from_env()
        self.assertIn(cm._mode, ("chars", "tokens"))
        self.assertGreater(cm.max_chars, 0)
        self.assertGreater(cm.keep_last_messages, 0)


if __name__ == "__main__":
    unittest.main()
