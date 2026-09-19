"""Best-effort bounded JSONB log batches; financial writes never use this queue."""

import asyncio
import contextlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


def postgres_json(value):
    # PostgreSQL JSONB rejects NUL and unpaired UTF-16 surrogates. Normalize
    # those only in diagnostic logs so one hostile prompt cannot poison a batch.
    if isinstance(value, str):
        return (
            value.replace("\x00", "\ufffd").encode("utf-8", errors="replace").decode()
        )
    if isinstance(value, dict):
        return {postgres_json(k): postgres_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [postgres_json(v) for v in value]
    return value


class TraceWriter:
    def __init__(
        self,
        store,
        max_rows=100,
        flush_seconds=1.0,
        max_batch_bytes=4 * 1024 * 1024,
        max_queue_bytes=32 * 1024 * 1024,
    ):
        self.store = store
        self.max_rows = max_rows
        self.flush_seconds = flush_seconds
        self.max_batch_bytes = max_batch_bytes
        self.max_queue_bytes = max_queue_bytes
        self.queue = asyncio.Queue(maxsize=1000)
        self.queued_bytes = 0
        self.written = 0
        self.dropped = 0
        self.failed_batches = 0
        self.task = None
        self.closed = False

    def start(self):
        self.task = asyncio.create_task(self._run())

    def submit(self, request_id, document, ttl_days):
        payload = json.dumps(
            postgres_json(document), separators=(",", ":"), allow_nan=False
        )
        size = len(payload.encode())
        if (
            self.closed
            or self.queue.full()
            or self.queued_bytes + size > self.max_queue_bytes
        ):
            self.dropped += 1
            logger.warning("Gateway trace queue full: dropped=%s", self.dropped)
            return False
        now = datetime.now(UTC)
        row = (request_id, now, now + timedelta(days=ttl_days), payload)
        self.queue.put_nowait((row, size))
        self.queued_bytes += size
        return True

    async def _run(self):
        while not self.closed or not self.queue.empty():
            try:
                first = await asyncio.wait_for(self.queue.get(), self.flush_seconds)
            except TimeoutError:
                continue
            batch = [first]
            batch_bytes = first[1]
            deadline = time.monotonic() + self.flush_seconds
            while len(batch) < self.max_rows and batch_bytes < self.max_batch_bytes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    if self.closed:
                        entry = self.queue.get_nowait()
                    else:
                        entry = await asyncio.wait_for(self.queue.get(), remaining)
                except (TimeoutError, asyncio.QueueEmpty):
                    break
                batch.append(entry)
                batch_bytes += entry[1]
            try:
                for attempt in range(3):
                    try:
                        await self.store.write_traces([row for row, _ in batch])
                        self.written += len(batch)
                        break
                    except Exception:
                        self.failed_batches += 1
                        if attempt == 2:
                            self.dropped += len(batch)
                            logger.error(
                                "Gateway trace batch dropped: rows=%s total_dropped=%s",
                                len(batch),
                                self.dropped,
                            )
                        else:
                            await asyncio.sleep(0.1 * 2**attempt)
            finally:
                self.queued_bytes -= batch_bytes
                for _ in batch:
                    self.queue.task_done()

    async def close(self):
        self.closed = True
        if self.task:
            try:
                await asyncio.wait_for(self.task, 20)
            except TimeoutError:
                logger.error("Gateway trace flush timed out; pending logs may be lost")
                with contextlib.suppress(asyncio.CancelledError):
                    await self.task
