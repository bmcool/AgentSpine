from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


class LaneQueue:
    """
    Minimal lane queue:
    - same lane -> serialized by per-lane lock
    - different lanes -> can run in parallel
    - optional global semaphore limits total concurrent tasks
    """

    def __init__(self, *, max_concurrent: int = 4) -> None:
        self._lane_locks: dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._global = threading.Semaphore(max(1, max_concurrent))

    @contextmanager
    def lane(self, lane_id: str) -> Iterator[None]:
        lock = self._lane_locks[lane_id]
        self._global.acquire()
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            self._global.release()

    def run(
        self,
        lane_id: str,
        fn: Callable[[], T],
        on_metrics: Callable[[float, float], None] | None = None,
    ) -> T:
        queued_at = time.monotonic()
        with self.lane(lane_id):
            started_at = time.monotonic()
            result = fn()
            ended_at = time.monotonic()
        if on_metrics is not None:
            wait_ms = max(0.0, (started_at - queued_at) * 1000.0)
            run_ms = max(0.0, (ended_at - started_at) * 1000.0)
            on_metrics(wait_ms, run_ms)
        return result
