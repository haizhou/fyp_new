"""Thread-pool concurrency with a request-per-minute rate limit for batch LLM passes.

The verifier deployment is rate-limited (grok: 50 RPM), so concurrency alone is not enough —
we must also space request starts. ``RateLimiter`` gates request starts to at most ``rpm`` per
minute regardless of worker count; ``run_concurrent`` runs ``fn`` over ``items`` on a thread pool
and calls ``on_result`` serially in the calling thread (so result handlers need no locking).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Iterable


class RateLimiter:
    """Spaces request starts to at most ``rpm`` per minute (thread-safe)."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
            wait = start - now
        if wait > 0:
            time.sleep(wait)


def run_concurrent(
    items: list[Any],
    fn: Callable[[Any], Any],
    *,
    workers: int,
    rpm: float,
    on_result: Callable[[Any, Any], None],
) -> None:
    """Run ``fn(item)`` across a thread pool under an RPM cap.

    ``on_result(item, result)`` is invoked once per item in the calling thread as results
    complete, so it can write output without locking. Exceptions from ``fn`` propagate as a
    ``("error", exc)`` tuple so a single failure does not kill the whole pass.
    """
    limiter = RateLimiter(rpm)

    def task(item: Any) -> tuple[Any, Any]:
        limiter.acquire()
        try:
            return item, fn(item)
        except Exception as exc:  # pragma: no cover - resilience for live API hiccups
            return item, ("error", exc)

    if workers <= 1:
        for item in items:
            limiter.acquire()
            try:
                on_result(item, fn(item))
            except Exception as exc:  # pragma: no cover
                on_result(item, ("error", exc))
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task, item) for item in items]
        for future in as_completed(futures):
            item, result = future.result()
            on_result(item, result)


__all__ = ["RateLimiter", "run_concurrent"]
