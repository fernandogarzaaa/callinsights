"""In-process sliding-window rate limiter (used to throttle login attempts).

State is per process. With several workers each keeps its own window, so the effective
limit is ``max_attempts * workers``; put a shared limiter at the proxy for stricter control.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(
        self,
        max_attempts: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_attempts = max_attempts
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits.setdefault(key, deque())
        cutoff = now - self.window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if not hits:
            # Keep memory bounded: forget keys with no recent attempts.
            del self._hits[key]
            return deque()
        return hits

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._prune(key, self._clock())) >= self.max_attempts

    def hit(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            self._prune(key, now)
            self._hits.setdefault(key, deque()).append(now)

    def retry_after(self, key: str) -> int:
        """Whole seconds until an attempt is allowed again (0 if not blocked)."""
        with self._lock:
            now = self._clock()
            hits = self._prune(key, now)
            if len(hits) < self.max_attempts:
                return 0
            return max(1, math.ceil(hits[0] + self.window - now))

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)
