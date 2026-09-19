"""Decode token observations. No token estimates from SSE chunk counts."""

import threading
import time
from collections import deque


class StreamMetrics:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.first_token = None
        self.last_token = None
        self.previous_token = None
        self.total = 0
        self.events = deque()
        self.lock = threading.Lock()
        self.smoothed_tps = None

    def observe(self, count=1):
        if count <= 0:
            return
        now = self.clock()
        with self.lock:
            self.total += count
            if self.first_token is None:
                self.first_token = now
            self.previous_token = self.last_token
            self.last_token = now
            self.events.append((now, count))
            self._prune(now)

    def _prune(self, now):
        while self.events and self.events[0][0] <= now - 15:
            self.events.popleft()

    def snapshot(self):
        now = self.clock()
        with self.lock:
            self._prune(now)
            age = now - self.first_token if self.first_token is not None else 0
            rates = {
                f"rolling_{seconds}s_tps": (
                    sum(n for t, n in self.events if t > now - seconds)
                    / min(age, seconds)
                    if age >= 1
                    else None
                )
                for seconds in (1, 5, 15)
            }
            # Suppress unstable sub-second warm-up samples for control decisions.
            rate = rates["rolling_5s_tps"]
            if rate is not None:
                self.smoothed_tps = rate
            interval = (
                self.last_token - self.previous_token
                if self.previous_token is not None
                else 0
            )
            return {
                "tokens": self.total,
                "instantaneous_tps": 1 / interval if interval > 0 else None,
                **rates,
                "lifetime_tps": (self.total - 1) / age if age > 0 else None,
                "ttft_ms": (
                    (self.first_token - self.started) * 1000
                    if self.first_token is not None
                    else None
                ),
                "source": "runtime_token_hook",
            }
