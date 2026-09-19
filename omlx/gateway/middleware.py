"""Single-process ASGI control plane around the existing oMLX application."""

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

import httpx

from omlx.scheduling import (
    RequestControl,
    request_control,
    set_local_inference_suspended,
)

from .admission import Admission, AdmissionDeniedError, Ticket
from .auth import bearer, digest_key
from .capacity import CapacityModel, context_bucket
from .policy import GatewayConfig
from .pricing import costs, token_cost
from .providers.openrouter import CircuitOpenError, OpenRouter
from .storage.object_store import ObjectStore
from .storage.postgres import LimitExceededError, Postgres
from .storage.trace_writer import TraceWriter
from .usage import UsageObserver, encode_json

logger = logging.getLogger(__name__)
AUTHORIZED = object()
PATHS = {"/v1/chat/completions", "/v1/responses", "/v1/completions"}
FIELDS = {
    "model",
    "messages",
    "input",
    "prompt",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "text",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "reasoning",
    "reasoning_effort",
    "stream",
    "stream_options",
    "n",
    "store",
    "metadata",
    "user",
    "x_gateway",
}


class BadRequestError(Exception):
    pass


async def error(send, status, code):
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": encode_json(
                {"error": {"message": code, "type": "gateway_error", "code": code}}
            ),
        }
    )


class Runtime:
    def __init__(self, config, store, pepper, cloud=None, traces=None):
        self.config = config
        self.store = store
        self.pepper = pepper
        self.cloud = cloud
        self.traces = traces
        self.trace_writer = None
        self.mode_task = None
        self.sample_group = ()
        self.sample_since = time.monotonic()
        self.last_capacity_sample = time.monotonic()
        self.admission = Admission(
            CapacityModel(config.capacity_points, config.capacity_margin),
            config.max_queue_depth,
        )

    @classmethod
    async def from_environment(cls, path):
        config = GatewayConfig.model_validate_json(Path(path).read_text())
        pepper = os.environ["OMLX_GATEWAY_PEPPER"].encode()
        digest_key("startup-validation", pepper)
        store = await Postgres.connect(os.environ["OMLX_GATEWAY_POSTGRES_DSN"])
        await store.claim_appliance()
        cloud = (
            OpenRouter(os.environ["OPENROUTER_API_KEY"])
            if config.cloud_enabled
            else None
        )
        traces = (
            ObjectStore(
                os.environ["OMLX_GATEWAY_TRACE_BUCKET"],
                os.getenv("OMLX_GATEWAY_S3_ENDPOINT"),
            )
            if os.getenv("OMLX_GATEWAY_TRACE_BUCKET")
            else None
        )
        runtime = cls(config, store, pepper, cloud, traces)
        # Explicit operator benchmarks override persisted observations.
        runtime.admission.capacity = CapacityModel(
            (*await store.capacity_points(), *config.capacity_points),
            config.capacity_margin,
        )
        await runtime.refresh_mode()
        runtime.mode_task = asyncio.create_task(runtime.watch_mode())
        if config.trace_backend == "postgres":
            runtime.trace_writer = TraceWriter(store)
            runtime.trace_writer.start()
        return runtime

    async def refresh_mode(self):
        mode = await self.store.routing_mode()
        if mode not in ("auto", "cloud_only"):
            raise RuntimeError("Missing or invalid gateway routing mode")
        await self.admission.set_mode(mode)
        set_local_inference_suspended(mode == "cloud_only")
        return mode

    async def watch_mode(self):
        while True:
            try:
                mode = await self.refresh_mode()
                state = {
                    "mode": mode,
                    "active_local": len(self.admission.active),
                    "hardware_released": False,
                }
                state["queue_depth"] = len(self.admission.waiting)
                state["streams"] = {
                    ticket.request_id: ticket.metrics.snapshot()
                    for ticket in self.admission.active.values()
                }
                state["bandwidth_bytes_per_second"] = None
                state["bandwidth_source"] = "unavailable"
                if self.trace_writer:
                    state["logging"] = {
                        "queued_bytes": self.trace_writer.queued_bytes,
                        "written": self.trace_writer.written,
                        "dropped": self.trace_writer.dropped,
                    }
                await self.sample_capacity()
                if mode == "cloud_only":
                    from omlx.server import get_engine_pool

                    pool = get_engine_pool()
                    if pool is not None:
                        state.update(await pool.unload_idle_models(include_pinned=True))
                        state["hardware_released"] = (
                            not state["loaded_models"]
                            and not state["loading_models"]
                            and not self.admission.active
                        )
                await self.store.report_runtime(state)
            except Exception:
                await self.admission.set_mode("unavailable")
                set_local_inference_suspended(True)
                logger.error("Gateway routing control unavailable")
            await asyncio.sleep(1)

    async def sample_capacity(self):
        active = list(self.admission.active.values())
        group = tuple(sorted(t.request_id for t in active))
        now = time.monotonic()
        if group != self.sample_group:
            self.sample_group, self.sample_since = group, now
        # Sample only a stable all-decode cohort. Do not teach pure-decode
        # estimates from cold prefill or a changing concurrency window.
        if (
            not active
            or now - self.sample_since < 5
            or now - self.last_capacity_sample < 5
        ):
            return
        if any(
            t.metrics.first_token is None or now - t.metrics.first_token < 5
            for t in active
        ):
            return
        if len({t.configuration for t in active}) != 1:
            return
        aggregate = sum(t.metrics.snapshot()["rolling_5s_tps"] or 0 for t in active)
        if aggregate <= 0:
            return
        configuration = active[0].configuration
        bucket = max(t.bucket for t in active)
        model = self.admission.capacity
        model.observe(configuration, bucket, len(active), aggregate)
        await self.store.record_capacity(
            configuration,
            bucket,
            len(active),
            model.points[(configuration, bucket, len(active))],
        )
        self.last_capacity_sample = now

    async def close(self):
        if self.mode_task:
            self.mode_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.mode_task
        if self.trace_writer:
            await self.trace_writer.close()
        if self.cloud:
            await self.cloud.close()
        await self.store.close()


def validate_body(body, policy, config):
    if not isinstance(body, dict) or set(body) - FIELDS:
        raise BadRequestError("unsupported_request_fields")
    if not isinstance(body.get("model"), str):
        raise BadRequestError("model_alias_required")
    name = body["model"]
    if name not in policy.allowed_model_aliases or name not in config.aliases:
        raise BadRequestError("model_not_allowed")
    if body.get("n", 1) != 1:
        raise BadRequestError("only_single_completion_supported")
    if not isinstance(body.get("stream", False), bool):
        raise BadRequestError("stream_must_be_boolean")
    stream_options = body.get("stream_options", {})
    if not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}:
        raise BadRequestError("invalid_stream_options")
    if not isinstance(stream_options.get("include_usage", False), bool):
        raise BadRequestError("invalid_stream_usage_option")
    if body.get("store"):
        raise BadRequestError("stateful_responses_not_enabled")
    # Until model-specific multimodal token accounting exists, accept text and
    # function tools only. This also excludes paid web-search/plugin execution.
    for tool in body.get("tools", []):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise BadRequestError("only_function_tools_supported")

    def check_text(value):
        if isinstance(value, dict):
            kind = value.get("type", "")
            if any(part in kind for part in ("image", "audio", "video", "file")):
                raise BadRequestError("multimodal_accounting_not_enabled")
            for child in value.values():
                check_text(child)
        elif isinstance(value, list):
            for child in value:
                check_text(child)

    check_text(body.get("messages", body.get("input", body.get("prompt", ""))))
    limits = [
        body[k]
        for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")
        if k in body
    ]
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in limits):
        raise BadRequestError("invalid_output_limit")
    if len(limits) > 1:
        raise BadRequestError("ambiguous_output_limits")
    limit = limits[0] if limits else policy.max_output_tokens
    if limit > policy.max_output_tokens:
        raise BadRequestError("output_limit_exceeded")
    # Conservative envelope, NOT a model tokenizer. Runtime enforces the exact
    # local prompt+output token limit before prefill through RequestControl.
    envelope = len(encode_json(body)) + 1024
    if envelope + limit > policy.max_context_tokens:
        raise BadRequestError("context_envelope_exceeded")
    extension = body.pop("x_gateway", {})
    if not isinstance(extension, dict) or set(extension) - {"include_usage"}:
        raise BadRequestError("unsupported_gateway_option")
    include_usage = extension.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise BadRequestError("invalid_gateway_usage_option")
    return name, config.aliases[name], limit, envelope, include_usage


class GatewayMiddleware:
    def __init__(self, app, config_path=None, runtime=None):
        self.app = app
        self.config_path = config_path
        self.runtime = runtime

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            try:
                if self.runtime is None:
                    self.runtime = await Runtime.from_environment(self.config_path)
                await self.app(scope, receive, send)
            finally:
                if self.runtime:
                    await self.runtime.close()
            return
        if scope["type"] != "http" or not scope["path"].startswith("/v1"):
            await self.app(scope, receive, send)
            return
        runtime = self.runtime
        if runtime is None:
            await error(send, 503, "gateway_not_ready")
            return
        key = bearer(scope.get("headers", []))
        try:
            identity = (
                await runtime.store.identity(digest_key(key, runtime.pepper))
                if key
                else None
            )
        except Exception:
            logger.error("Gateway authentication storage unavailable")
            await error(send, 503, "authentication_unavailable")
            return
        if identity is None:
            await error(send, 401, "invalid_api_key")
            return
        key_id, version, policy = identity
        path = scope["path"]
        if path == "/v1/models" and scope["method"] == "GET":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": encode_json(
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": name,
                                    "object": "model",
                                    "created": 0,
                                    "owned_by": "gateway",
                                }
                                for name in policy.allowed_model_aliases
                                if name in runtime.config.aliases
                            ],
                        }
                    ),
                }
            )
            return
        if path not in PATHS or scope["method"] != "POST":
            await error(send, 404, "gateway_endpoint_not_enabled")
            return
        body_bytes = bytearray()
        try:
            async with asyncio.timeout(15):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body_bytes.extend(message.get("body", b""))
                    if len(body_bytes) > runtime.config.max_request_bytes:
                        raise BadRequestError("request_too_large")
                    if not message.get("more_body", False):
                        break
            body = json.loads(body_bytes)
            name, alias, limit, envelope, detailed = validate_body(
                body, policy, runtime.config
            )
        except (
            BadRequestError,
            ValueError,
            TypeError,
            AttributeError,
            TimeoutError,
        ) as exc:
            await error(
                send,
                400,
                str(exc) if isinstance(exc, BadRequestError) else "invalid_request",
            )
            return
        for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            body.pop(field, None)
        if path == "/v1/responses":
            body["max_output_tokens"] = limit
            body["store"] = False
        else:
            body["max_tokens"] = limit
        ticket = Ticket(
            str(uuid.uuid4()),
            key_id,
            policy,
            alias.configuration,
            context_bucket(envelope + limit),
        )
        try:
            await runtime.store.begin(ticket.request_id, key_id, version, policy)
        except LimitExceededError as exc:
            await error(send, 429, str(exc))
            return
        except Exception:
            await error(send, 503, "accounting_unavailable")
            return

        # Monitor disconnect independently while queued AND while dispatching.
        async def disconnect():
            while True:
                if (await receive())["type"] == "http.disconnect":
                    return

        disconnected = asyncio.Event()

        async def watch_disconnect():
            await disconnect()
            disconnected.set()

        watcher = asyncio.create_task(watch_disconnect())
        work = asyncio.create_task(
            self.dispatch(
                scope, body, send, ticket, name, alias, detailed, disconnected
            )
        )
        try:
            done, _ = await asyncio.wait(
                {watcher, work}, return_when=asyncio.FIRST_COMPLETED
            )
            if work not in done:
                work.cancel()
            await work
        except asyncio.CancelledError:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    async def dispatch(
        self, scope, body, send, ticket, name, alias, detailed, disconnected
    ):
        runtime, policy = self.runtime, ticket.policy
        route = "local"
        queue_ms = 0
        cloud_reserved = False
        dispatched = False
        started = False
        streaming = False
        status = "failed"
        status_code = 500
        response_body = bytearray()
        trace_response = bytearray()
        trace_truncated = False
        local_started = None
        control_token = None

        def extension(usage):
            return {
                "request_id": ticket.request_id,
                "route": route,
                "provider": "openrouter" if route == "cloud" else "local",
                "physical_model": alias.cloud_model
                if route == "cloud"
                else alias.local_model,
                "queue_ms": queue_ms,
                "telemetry": ticket.metrics.snapshot() if route == "local" else None,
                "cost": costs(
                    runtime.config,
                    alias,
                    usage,
                    route,
                    runtime.admission.allocated_seconds(ticket),
                ),
            }

        observer = UsageObserver(
            name,
            extension if detailed else None,
            include_stream_usage=(
                scope["path"] == "/v1/responses"
                or detailed
                or bool(body.get("stream_options", {}).get("include_usage"))
            ),
        )

        async def response_send(message):
            nonlocal started, streaming, status_code, trace_truncated
            if message["type"] == "http.response.start":
                started = True
                status_code = message["status"]
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in (b"content-length", b"content-encoding")
                ]
                streaming = any(
                    k.lower() == b"content-type" and b"text/event-stream" in v
                    for k, v in headers
                )
                headers.append((b"x-request-id", ticket.request_id.encode()))
                await send({**message, "headers": headers})
                return
            if message["type"] != "http.response.body":
                await send(message)
                return
            data = message.get("body", b"")
            final = not message.get("more_body", False)
            if streaming:
                data = observer.feed(data, final)
            else:
                response_body.extend(data)
                if len(response_body) > 16 * 1024 * 1024:
                    raise ValueError("provider response too large")
                if not final:
                    return
                try:
                    data = encode_json(observer.document(json.loads(response_body)))
                    observer.completed = True
                except (ValueError, UnicodeDecodeError):
                    observer.failed = True
                    data = bytes(response_body)
            if policy.logging_policy == "full":
                available = runtime.config.max_trace_bytes - len(trace_response)
                trace_response.extend(data[:available])
                trace_truncated |= len(data) > available
            await send({**message, "body": data})

        try:
            async with asyncio.timeout(runtime.config.request_timeout_seconds):
                mode = await runtime.refresh_mode()
                if mode == "cloud_only" and policy.local_policy == "only":
                    raise LimitExceededError("cloud_only_mode_local_only_key")
                if (
                    mode == "cloud_only"
                    and policy.local_policy == "prefer"
                    and not policy.cloud_fallback
                ):
                    raise LimitExceededError("cloud_only_mode_fallback_disabled")
                if policy.local_policy == "never" or mode == "cloud_only":
                    route = "cloud"
                else:
                    try:
                        queue_ms = await runtime.admission.acquire(ticket)
                    except AdmissionDeniedError as exc:
                        if (
                            exc.action != "cloud"
                            or policy.local_policy == "only"
                            or not policy.cloud_fallback
                        ):
                            raise
                        route = "cloud"
                if route == "cloud":
                    if (
                        not runtime.cloud
                        or not alias.cloud_model
                        or not alias.price
                        or not alias.cloud_price_is_ceiling
                    ):
                        raise LimitExceededError(
                            "cloud_not_configured_with_price_ceiling"
                        )
                    if not alias.provider_only:
                        raise LimitExceededError("cloud_provider_allowlist_required")
                    reservation = token_cost(
                        alias.price, policy.max_context_tokens, policy.max_output_tokens
                    )
                    await runtime.store.reserve_cloud(
                        ticket.request_id,
                        ticket.key_id,
                        policy,
                        runtime.config,
                        reservation,
                    )
                    cloud_reserved = True
                    dispatched = True
                    await runtime.cloud.serve(scope["path"], body, alias, response_send)
                else:
                    local_started = time.monotonic()
                    body["model"] = alias.local_model
                    # Always request upstream usage internally for accounting.
                    if body.get("stream") and scope["path"] != "/v1/responses":
                        body["stream_options"] = {"include_usage": True}
                    payload = encode_json(body)
                    delivered = False

                    async def local_receive():
                        nonlocal delivered
                        if not delivered:
                            delivered = True
                            return {
                                "type": "http.request",
                                "body": payload,
                                "more_body": False,
                            }
                        await disconnected.wait()
                        return {"type": "http.disconnect"}

                    local_scope = dict(scope)
                    local_scope["omlx.gateway.authorized"] = AUTHORIZED
                    local_scope["headers"] = [
                        (k, v)
                        for k, v in scope.get("headers", [])
                        if k.lower()
                        not in (b"content-length", b"authorization", b"x-api-key")
                    ]
                    local_scope["headers"].append(
                        (b"content-length", str(len(payload)).encode())
                    )
                    control_token = request_control.set(
                        RequestControl(
                            priority=policy.priority,
                            weight=policy.scheduling_weight,
                            max_context_tokens=policy.max_context_tokens,
                            on_token=ticket.metrics.observe,
                        )
                    )
                    dispatched = True
                    await self.app(local_scope, local_receive, response_send)
                status = (
                    "complete"
                    if status_code < 400 and observer.completed and not observer.failed
                    else "failed"
                )
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except (AdmissionDeniedError, LimitExceededError, CircuitOpenError) as exc:
            if not started:
                await error(send, 429, str(exc))
        except (TimeoutError, httpx.HTTPError, ValueError):
            if not started:
                await error(send, 503, "inference_unavailable")
            elif streaming:
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'data: {"error":{"code":"gateway_stream_failed","message":"Generation interrupted"}}\n\n',
                        "more_body": False,
                    }
                )
        except Exception:
            logger.error("Gateway request failed: %s", ticket.request_id)
            if not started:
                await error(send, 503, "gateway_unavailable")
        finally:
            if control_token is not None:
                request_control.reset(control_token)

            # Cleanup/accounting is shielded from disconnect cancellation.
            async def finalize():
                allocated = runtime.admission.allocated_seconds(ticket)
                await runtime.admission.release(ticket)
                accounting = costs(
                    runtime.config, alias, observer.usage, route, allocated
                )
                actual_cloud = (
                    accounting["cash_cogs"]
                    if route == "cloud" and cloud_reserved
                    else None
                )
                if not dispatched:
                    actual_cloud = 0
                metadata = {
                    "route": route,
                    "alias": name,
                    "queue_ms": queue_ms,
                    "usage": observer.usage,
                    "cost": accounting,
                    "provider_id": observer.provider_id,
                    "configuration": alias.configuration,
                    "allocated_busy_seconds": allocated,
                    "telemetry": ticket.metrics.snapshot() if local_started else None,
                }
                trace_key = None
                if policy.logging_policy == "full":
                    # Bound BOTH input and output logging independently.
                    request_size = len(encode_json(body))
                    trace = {
                        "request": body
                        if request_size <= runtime.config.max_trace_bytes
                        else None,
                        "request_truncated": request_size
                        > runtime.config.max_trace_bytes,
                        "request_bytes": request_size,
                        "response": trace_response.decode("utf-8", errors="replace"),
                        "response_truncated": trace_truncated,
                    }
                if policy.logging_policy == "full" and runtime.trace_writer:
                    queued = runtime.trace_writer.submit(
                        ticket.request_id, trace, policy.trace_ttl_days
                    )
                    metadata["trace_status"] = "queued" if queued else "dropped"
                    trace_key = f"postgres:{ticket.request_id}" if queued else None
                elif policy.logging_policy == "full" and runtime.traces:
                    try:
                        trace_key = await runtime.traces.put(
                            ticket.request_id, trace, policy.trace_ttl_days
                        )
                    except Exception:
                        metadata["trace_status"] = "write_failed"
                        logger.error(
                            "Gateway trace write failed: %s", ticket.request_id
                        )
                elif policy.logging_policy == "full":
                    metadata["trace_status"] = "not_configured"
                if policy.logging_policy == "none":
                    metadata = {
                        "provider_id": observer.provider_id
                    }  # Minimal spend ledger remains mandatory.
                tokens = observer.total_tokens() if dispatched else 0
                await runtime.store.finish(
                    ticket.request_id, status, tokens, actual_cloud, metadata, trace_key
                )

            cleanup = asyncio.create_task(finalize())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
