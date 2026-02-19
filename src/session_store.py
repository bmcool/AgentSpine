from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .session import Session, SessionMeta, utc_now_iso


class SessionStore:
    def __init__(self, *, sessions_dir: str) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def resolve_session_id(self, requested: str | None) -> str:
        return requested.strip() if requested and requested.strip() else uuid4().hex[:12]

    def load_or_create(
        self,
        *,
        session_id: str,
        provider: str,
        model: str,
        workspace_dir: str,
        parent_session_id: str | None = None,
        subagent_depth: int = 0,
    ) -> Session:
        path = self._session_path(session_id)
        if not path.is_file():
            now = utc_now_iso()
            meta = SessionMeta(
                session_id=session_id,
                provider=provider,
                model=model,
                workspace_dir=workspace_dir,
                parent_session_id=parent_session_id,
                subagent_depth=subagent_depth,
                created_at=now,
                updated_at=now,
            )
            session = Session(meta=meta, messages=[])
            self.save(session)
            return session

        meta: SessionMeta | None = None
        entries: list[dict] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            kind = row.get("type")
            if kind == "header":
                meta = SessionMeta(
                    session_id=str(row.get("session_id", session_id)),
                    provider=str(row.get("provider", provider)),
                    model=str(row.get("model", model)),
                    workspace_dir=str(row.get("workspace_dir", workspace_dir)),
                    parent_session_id=(
                        str(row.get("parent_session_id")) if row.get("parent_session_id") is not None else None
                    ),
                    subagent_depth=int(row.get("subagent_depth", subagent_depth)),
                    created_at=str(row.get("created_at", utc_now_iso())),
                    updated_at=str(row.get("updated_at", utc_now_iso())),
                    usage_input_tokens=int(row.get("usage_input_tokens", 0) or 0),
                    usage_output_tokens=int(row.get("usage_output_tokens", 0) or 0),
                    usage_total_tokens=int(row.get("usage_total_tokens", 0) or 0),
                    usage_cache_read_tokens=int(row.get("usage_cache_read_tokens", 0) or 0),
                    usage_cache_write_tokens=int(row.get("usage_cache_write_tokens", 0) or 0),
                )
            elif kind == "message":
                message = row.get("message")
                if isinstance(message, dict):
                    entries.append(
                        {
                            "type": "message",
                            "message": message,
                            "timestamp": str(row.get("timestamp", utc_now_iso())),
                        }
                    )
            elif kind in {"custom", "custom_message", "compaction"}:
                if isinstance(row, dict):
                    entries.append(row)

        if meta is None:
            now = utc_now_iso()
            meta = SessionMeta(
                session_id=session_id,
                provider=provider,
                model=model,
                workspace_dir=workspace_dir,
                parent_session_id=parent_session_id,
                subagent_depth=subagent_depth,
                created_at=now,
                updated_at=now,
            )
        else:
            meta.provider = provider
            meta.model = model
            meta.workspace_dir = workspace_dir
            if parent_session_id is not None:
                meta.parent_session_id = parent_session_id
            meta.subagent_depth = max(0, int(meta.subagent_depth))
            meta.updated_at = utc_now_iso()

        session = Session(meta=meta, entries=entries)
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        path = self._session_path(session.meta.session_id)
        lines = [
            json.dumps(
                {
                    "type": "header",
                    "session_id": session.meta.session_id,
                    "provider": session.meta.provider,
                    "model": session.meta.model,
                    "workspace_dir": session.meta.workspace_dir,
                    "parent_session_id": session.meta.parent_session_id,
                    "subagent_depth": session.meta.subagent_depth,
                    "created_at": session.meta.created_at,
                    "updated_at": session.meta.updated_at,
                    "usage_input_tokens": session.meta.usage_input_tokens,
                    "usage_output_tokens": session.meta.usage_output_tokens,
                    "usage_total_tokens": session.meta.usage_total_tokens,
                    "usage_cache_read_tokens": session.meta.usage_cache_read_tokens,
                    "usage_cache_write_tokens": session.meta.usage_cache_write_tokens,
                },
                ensure_ascii=False,
            )
        ]
        for entry in session.entries:
            if not isinstance(entry, dict):
                continue
            kind = entry.get("type")
            if kind not in {"message", "custom", "custom_message", "compaction"}:
                continue
            lines.append(json.dumps(entry, ensure_ascii=False))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _session_path(self, session_id: str) -> Path:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_")).strip()
        if not safe:
            safe = "default"
        return self.sessions_dir / f"{safe}.jsonl"
