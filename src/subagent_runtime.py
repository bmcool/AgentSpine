from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


@dataclass
class SubagentJob:
    run_id: str
    cancel_event: threading.Event
    future: Future[None]


class SubagentRuntime:
    def __init__(self, *, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="subagent")
        self._jobs: dict[str, SubagentJob] = {}
        self._lock = threading.Lock()

    def submit(self, run_id: str, fn: Callable[[threading.Event], None]) -> None:
        cancel_event = threading.Event()

        def _wrapped() -> None:
            fn(cancel_event)

        future = self._executor.submit(_wrapped)
        job = SubagentJob(run_id=run_id, cancel_event=cancel_event, future=future)
        with self._lock:
            self._jobs[run_id] = job

        def _cleanup(_fut: Future[None]) -> None:
            with self._lock:
                self._jobs.pop(run_id, None)

        future.add_done_callback(_cleanup)

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            return False
        job.cancel_event.set()
        job.future.cancel()
        return True

    def is_running(self, run_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            return False
        return not job.future.done()


_GLOBAL_SUBAGENT_RUNTIME = SubagentRuntime(max_workers=max(1, int(os.getenv("AGENT_SUBAGENT_MAX_WORKERS", "2"))))
