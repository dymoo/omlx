"""Bounded priority queue with atomic local admission and cancellation cleanup."""

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

from .policy import KeyPolicy
from .telemetry import StreamMetrics


class AdmissionDeniedError(Exception):
    def __init__(self, reason: str, action: str = "reject"):
        self.reason = reason
        self.action = action
        super().__init__(reason)


@dataclass
class Ticket:
    request_id: str
    key_id: str
    policy: KeyPolicy
    configuration: str
    bucket: int
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    arrived: float = field(default_factory=time.monotonic)
    allocated_seconds: float = 0


class Admission:
    def __init__(self, capacity, max_queue_depth=256):
        self.capacity = capacity
        self.max_queue_depth = max_queue_depth
        self.active: dict[str, Ticket] = {}
        self.waiting: dict[str, Ticket] = {}
        self.condition = asyncio.Condition()
        self.last_accounted = time.monotonic()
        self.mode = "auto"

    async def set_mode(self, mode):
        async with self.condition:
            self.mode = mode
            self.condition.notify_all()

    def _account_time(self):
        now = time.monotonic()
        if self.active:
            share = (now - self.last_accounted) / len(self.active)
            for ticket in self.active.values():
                ticket.allocated_seconds += share
        self.last_accounted = now

    def allocated_seconds(self, ticket):
        self._account_time()
        return ticket.allocated_seconds

    async def acquire(self, ticket):
        async with self.condition:
            if len(self.waiting) >= self.max_queue_depth:
                raise AdmissionDeniedError("queue_full", ticket.policy.queue_timeout)
            self.waiting[ticket.request_id] = ticket
            deadline = ticket.arrived + ticket.policy.max_local_queue_ms / 1000
            try:
                while True:
                    if self.mode != "auto":
                        raise AdmissionDeniedError(
                            self.mode,
                            "cloud" if self.mode == "cloud_only" else "reject",
                        )
                    head = min(
                        self.waiting.values(),
                        key=lambda t: (t.policy.priority, t.arrived, t.request_id),
                    )
                    allowed, reason = self.capacity.allows(
                        ticket, list(self.active.values())
                    )
                    if head is ticket and allowed:
                        self._account_time()
                        self.active[ticket.request_id] = ticket
                        return (time.monotonic() - ticket.arrived) * 1000
                    if ticket.policy.below_floor != "queue":
                        raise AdmissionDeniedError(reason, ticket.policy.below_floor)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AdmissionDeniedError(
                            "queue_deadline", ticket.policy.queue_timeout
                        )
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            self.condition.wait(), min(remaining, 0.25)
                        )
            finally:
                self.waiting.pop(ticket.request_id, None)
                self.condition.notify_all()

    async def release(self, ticket):
        async with self.condition:
            self._account_time()
            self.active.pop(ticket.request_id, None)
            self.condition.notify_all()
