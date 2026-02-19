"""Tests for Session and SessionStore."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.session import Session, SessionMeta, utc_now_iso
from src.session_store import SessionStore


class SessionTests(unittest.TestCase):
    def test_add_user_message(self) -> None:
        meta = SessionMeta(
            session_id="s1",
            provider="openai",
            model="gpt-4o",
            workspace_dir="/tmp",
            parent_session_id=None,
            subagent_depth=0,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        session = Session(meta=meta)
        session.add_user_message("hello")
        self.assertEqual(len(session), 1)
        self.assertEqual(session.messages[0], {"role": "user", "content": "hello"})

    def test_add_tool_result(self) -> None:
        meta = SessionMeta(
            session_id="s1",
            provider="openai",
            model="gpt-4o",
            workspace_dir="/tmp",
            parent_session_id=None,
            subagent_depth=0,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        session = Session(meta=meta)
        session.add_tool_result("tc1", "result text")
        self.assertEqual(len(session), 1)
        self.assertEqual(session.messages[0]["role"], "tool")
        self.assertEqual(session.messages[0]["tool_call_id"], "tc1")
        self.assertEqual(session.messages[0]["content"], "result text")

    def test_reset_clears_messages(self) -> None:
        meta = SessionMeta(
            session_id="s1",
            provider="openai",
            model="gpt-4o",
            workspace_dir="/tmp",
            parent_session_id=None,
            subagent_depth=0,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        session = Session(meta=meta)
        session.add_user_message("a")
        session.add_assistant_message({"role": "assistant", "content": "b"})
        session.reset()
        self.assertEqual(len(session), 0)


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SessionStore(sessions_dir=self.temp.name)

    def test_resolve_session_id_uses_requested(self) -> None:
        sid = self.store.resolve_session_id("my-session")
        self.assertEqual(sid, "my-session")

    def test_resolve_session_id_generates_when_empty(self) -> None:
        sid = self.store.resolve_session_id("")
        self.assertTrue(len(sid) == 12 and sid.isalnum())
        sid2 = self.store.resolve_session_id(None)
        self.assertTrue(len(sid2) == 12 and sid2.isalnum())

    def test_load_or_create_creates_new_session(self) -> None:
        session = self.store.load_or_create(
            session_id="new1",
            provider="openai",
            model="gpt-4o",
            workspace_dir="/w",
        )
        self.assertEqual(session.meta.session_id, "new1")
        self.assertEqual(len(session), 0)
        path = Path(self.temp.name) / "new1.jsonl"
        self.assertTrue(path.is_file())

    def test_save_and_load_roundtrip(self) -> None:
        session = self.store.load_or_create(
            session_id="round",
            provider="anthropic",
            model="claude-3-5-sonnet",
            workspace_dir="/workspace",
        )
        session.add_user_message("hi")
        session.add_assistant_message({"role": "assistant", "content": "hello"})
        self.store.save(session)

        loaded = self.store.load_or_create(
            session_id="round",
            provider="anthropic",
            model="claude-3-5-sonnet",
            workspace_dir="/workspace",
        )
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded.messages[0]["content"], "hi")
        self.assertEqual(loaded.messages[1]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
