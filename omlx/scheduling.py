"""Generic per-request control hooks, independent of HTTP and gateway policy."""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestControl:
    priority: int = 100
    weight: int = 1
    max_context_tokens: int | None = None
    on_token: Callable[[int], None] | None = None


request_control: ContextVar[RequestControl | None] = ContextVar(
    "omlx_request_control", default=None
)

_local_inference_suspended = False


def set_local_inference_suspended(suspended: bool) -> None:
    global _local_inference_suspended
    _local_inference_suspended = suspended


def local_inference_suspended() -> bool:
    return _local_inference_suspended
