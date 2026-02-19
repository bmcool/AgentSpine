from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

Status = Literal["queued", "running", "completed", "failed", "killed"]

# Event entry: {"type": Status, "at": iso_timestamp}
RunEvent = dict[str, str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(typ: Status) -> RunEvent:
    return {"type": typ, "at": _now()}


@dataclass
class SubagentRun:
    run_id: str
    parent_session_id: str
    child_session_id: str
    task: str
    status: Status
    created_at: str
    updated_at: str
    provider: str
    model: str
    last_reply: str | None = None
    last_error: str | None = None
    events: list[RunEvent] = field(default_factory=list)


class SubagentRegistry:
    def __init__(self, *, file_path: str) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.file_path.exists():
            self._write([])

    def spawn(
        self,
        *,
        parent_session_id: str,
        task: str,
        provider: str,
        model: str,
    ) -> SubagentRun:
        now = _now()
        run = SubagentRun(
            run_id=f"subrun-{uuid4().hex[:10]}",
            parent_session_id=parent_session_id,
            child_session_id=f"subsess-{uuid4().hex[:10]}",
            task=task,
            status="queued",
            created_at=now,
            updated_at=now,
            provider=provider,
            model=model,
            events=[_event("queued")],
        )
        with self._lock:
            rows = self._read()
            rows.append(asdict(run))
            self._write(rows)
        return run

    def list(self, *, parent_session_id: str) -> list[SubagentRun]:
        with self._lock:
            rows = self._read()
        result: list[SubagentRun] = []
        for row in rows:
            if str(row.get("parent_session_id")) != parent_session_id:
                continue
            try:
                result.append(
                    SubagentRun(
                        run_id=str(row.get("run_id", "")),
                        parent_session_id=str(row.get("parent_session_id", "")),
                        child_session_id=str(row.get("child_session_id", "")),
                        task=str(row.get("task", "")),
                        status=self._normalize_status(row.get("status")),
                        created_at=str(row.get("created_at", _now())),
                        updated_at=str(row.get("updated_at", _now())),
                        provider=str(row.get("provider", "")),
                        model=str(row.get("model", "")),
                        last_reply=str(row["last_reply"]) if row.get("last_reply") is not None else None,
                        last_error=str(row["last_error"]) if row.get("last_error") is not None else None,
                        events=self._parse_events(row.get("events")),
                    )
                )
            except TypeError:
                continue
        result.sort(key=lambda r: r.created_at)
        return result

    def get(self, run_id: str) -> SubagentRun | None:
        with self._lock:
            rows = self._read()
        for row in rows:
            if str(row.get("run_id")) == run_id:
                try:
                    return SubagentRun(
                        run_id=str(row.get("run_id", "")),
                        parent_session_id=str(row.get("parent_session_id", "")),
                        child_session_id=str(row.get("child_session_id", "")),
                        task=str(row.get("task", "")),
                        status=self._normalize_status(row.get("status")),
                        created_at=str(row.get("created_at", _now())),
                        updated_at=str(row.get("updated_at", _now())),
                        provider=str(row.get("provider", "")),
                        model=str(row.get("model", "")),
                        last_reply=str(row["last_reply"]) if row.get("last_reply") is not None else None,
                        last_error=str(row["last_error"]) if row.get("last_error") is not None else None,
                        events=self._parse_events(row.get("events")),
                    )
                except TypeError:
                    return None
        return None

    def set_status(self, run_id: str, status: Status) -> SubagentRun | None:
        with self._lock:
            rows = self._read()
            changed = False
            for row in rows:
                if str(row.get("run_id")) != run_id:
                    continue
                st = self._normalize_status(status)
                row["status"] = st
                row["updated_at"] = _now()
                events = row.get("events")
                if not isinstance(events, list):
                    events = []
                events.append(_event(st))
                row["events"] = events
                changed = True
                break
            if not changed:
                return None
            self._write(rows)
            return self.get(run_id)

    def set_running(self, run_id: str) -> SubagentRun | None:
        return self.set_status(run_id, "running")

    def set_completed(self, run_id: str, *, reply: str) -> SubagentRun | None:
        return self._set_result(run_id, status="completed", reply=reply, error=None)

    def set_failed(self, run_id: str, *, error: str) -> SubagentRun | None:
        return self._set_result(run_id, status="failed", reply=None, error=error)

    def set_killed(self, run_id: str) -> SubagentRun | None:
        return self._set_result(run_id, status="killed", reply=None, error="killed by request")

    def _set_result(
        self,
        run_id: str,
        *,
        status: Status,
        reply: str | None,
        error: str | None,
    ) -> SubagentRun | None:
        with self._lock:
            rows = self._read()
            for row in rows:
                if str(row.get("run_id")) != run_id:
                    continue
                st = self._normalize_status(status)
                row["status"] = st
                row["updated_at"] = _now()
                row["last_reply"] = reply
                row["last_error"] = error
                events = row.get("events")
                if not isinstance(events, list):
                    events = []
                events.append(_event(st))
                row["events"] = events
                self._write(rows)
                return self.get(run_id)
        return None

    def _read(self) -> list[dict]:
        try:
            raw = self.file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        if not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def _write(self, rows: list[dict]) -> None:
        tmp = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.file_path)

    def _normalize_status(self, raw: object) -> Status:
        candidate = str(raw or "").strip().lower()
        if candidate in {"queued", "running", "completed", "failed", "killed"}:
            return candidate  # type: ignore[return-value]
        # Backward compatibility with earlier state names.
        if candidate == "active":
            return "running"
        return "queued"

    def _parse_events(self, raw: object) -> list[RunEvent]:
        if not isinstance(raw, list):
            return []
        out: list[RunEvent] = []
        for item in raw:
            if isinstance(item, dict) and "type" in item and "at" in item:
                out.append({"type": str(item["type"]), "at": str(item["at"])})
        return out
