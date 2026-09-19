import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from omlx.gateway.admission import Admission, AdmissionDeniedError, Ticket
from omlx.gateway.auth import create_key, digest_key
from omlx.gateway.capacity import CapacityModel
from omlx.gateway.middleware import AUTHORIZED, GatewayMiddleware, Runtime
from omlx.gateway.policy import Alias, CapacityPoint, GatewayConfig, KeyPolicy, Price
from omlx.gateway.pricing import costs, token_cost
from omlx.gateway.providers.openrouter import OpenRouter
from omlx.gateway.storage.postgres import LimitExceededError
from omlx.gateway.telemetry import StreamMetrics
from omlx.gateway.usage import UsageObserver
from omlx.request import Request, SamplingParams
from omlx.scheduling import RequestControl, request_control


def policy(**changes):
    return KeyPolicy(allowed_model_aliases=("coder",), **changes)


def test_key_entropy_and_hash():
    first, digest = create_key(b"a" * 32)
    second, _ = create_key(b"a" * 32)
    assert first != second and first not in digest
    assert digest == digest_key(first, b"a" * 32)
    assert digest != digest_key(first, b"b" * 32)


def test_runtime_context_and_tokens():
    counts = []
    token = request_control.set(RequestControl(priority=2, on_token=counts.append))
    try:
        request = Request("one", "hello", SamplingParams())
    finally:
        request_control.reset(token)
    request.append_output_token(5)
    assert request.priority == 2 and counts == [1]
    assert Request("two", "hello", SamplingParams()).control is None


def test_rolling_metrics_decay_when_stalled():
    now = [0.0]
    metrics = StreamMetrics(lambda: now[0])
    for i in range(100):
        now[0] = i / 10
        metrics.observe()
    assert metrics.snapshot()["rolling_5s_tps"] == pytest.approx(10)
    now[0] = 30
    assert metrics.snapshot()["rolling_5s_tps"] == 0
    assert metrics.snapshot()["tokens"] == 100


async def test_admission_protects_floor_and_cleans_cancelled_queue():
    model = CapacityModel(
        (
            CapacityPoint(
                configuration="a", context_bucket=1024, concurrency=1, aggregate_tps=50
            ),
            CapacityPoint(
                configuration="a", context_bucket=1024, concurrency=2, aggregate_tps=60
            ),
        ),
        margin=0,
    )
    admission = Admission(model)
    high = Ticket("high", "k", policy(priority=0, min_tps=40), "a", 1024)
    low = Ticket("low", "l", policy(priority=10, max_local_queue_ms=10000), "a", 1024)
    await admission.acquire(high)
    queued = asyncio.create_task(admission.acquire(low))
    await asyncio.sleep(0)
    assert "low" in admission.waiting and "low" not in admission.active
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert not admission.waiting
    await admission.release(high)
    await admission.acquire(low)
    assert "low" in admission.active


async def test_uncalibrated_floor_is_not_promised():
    admission = Admission(CapacityModel())
    with pytest.raises(AdmissionDeniedError):
        await admission.acquire(
            Ticket("one", "key", policy(min_tps=30, below_floor="reject"), "a", 1024)
        )


def test_sse_split_crlf_multiline_and_usage():
    observer = UsageObserver("coder", lambda usage: {"route": "cloud"})
    stream = b': ping\r\n\r\ndata: {"id":"gen-1",\r\ndata: "model":"physical","usage":{"prompt_tokens":3,"completion_tokens":2,"cost":0.01}}\r\n\r\ndata: [DONE]\r\n\r\n'
    output = b"".join(observer.feed(stream[i : i + 1]) for i in range(len(stream)))
    assert b'"model":"coder"' in output
    assert b'"x_gateway":{"route":"cloud"}' in output
    assert observer.total_tokens() == 5 and observer.completed
    assert observer.provider_id == "gen-1"


def test_usage_without_extension_and_truncation():
    observer = UsageObserver("coder")
    output = observer.feed(b'data: {"usage":{"prompt_tokens":1}}\n\n')
    assert b"x_gateway" not in output
    observer.feed(b'data: {"unfinished"', final=True)
    assert observer.failed and not observer.completed


def test_price_and_negative_savings_and_unknown_cloud_cost():
    price = Price(
        version="test",
        input_per_million=1,
        cached_input_per_million=0.1,
        output_per_million=2,
    )
    assert token_cost(price, 1000, 100, 500) == Decimal("0.00075")
    alias = Alias(local_model="m", configuration="a", price=price)
    config = GatewayConfig(
        aliases={"coder": alias},
        incremental_watts=100,
        electricity_usd_per_kwh=1,
        amortization_usd_per_busy_hour=1,
    )
    result = costs(
        config, alias, {"prompt_tokens": 1, "completion_tokens": 1}, "local", 3600
    )
    assert result["subsidy"] < 0  # Never clamp unfavorable economics away.
    assert costs(config, alias, {}, "cloud", 0)["cash_cogs"] is None


class Store:
    def __init__(self, pol, digest):
        self.policy = pol
        self.digest = digest
        self.records = {}
        self.cloud_calls = 0
        self.mode = "auto"

    async def routing_mode(self):
        return self.mode

    async def identity(self, digest):
        return (
            ("key", 1, self.policy)
            if digest == self.digest and self.policy.enabled
            else None
        )

    async def begin(self, request_id, key_id, version, policy):
        if (
            sum(r["status"] == "running" for r in self.records.values())
            >= policy.max_concurrency
        ):
            raise LimitExceededError("key_concurrency")
        self.records[request_id] = {"status": "running"}

    async def reserve_cloud(self, *args):
        self.cloud_calls += 1
        if args[-1] > self.policy.max_cloud_cost_usd:
            raise LimitExceededError("request_cloud_budget")

    async def finish(self, request_id, status, tokens, cost, metadata, trace_key=None):
        self.records[request_id] = dict(
            status=status, tokens=tokens, cost=cost, metadata=metadata
        )


async def local_app(scope, receive, send):
    assert scope["omlx.gateway.authorized"] is AUTHORIZED
    data = json.loads((await receive())["body"])
    assert data["model"] == "physical"
    req = Request("engine", "hello", SamplingParams())
    req.append_output_token(1)
    if data.get("stream"):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'data: {"model":"physical","choices":[{"delta":{"content":"ok"}}]}\n\n',
                "more_body": True,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'data: {"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\ndata: [DONE]\n\n',
                "more_body": False,
            }
        )
    else:
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
                "body": b'{"model":"physical","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
            }
        )


def setup_gateway(pol=None, config=None, cloud=None, app=local_app):
    plaintext, digest = create_key(b"p" * 32)
    store = Store(pol or policy(), digest)
    config = config or GatewayConfig(
        aliases={"coder": Alias(local_model="physical", configuration="test")}
    )
    runtime = Runtime(config, store, b"p" * 32, cloud=cloud)
    gateway = GatewayMiddleware(app, runtime=runtime)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway),
        base_url="http://test",
        headers={"Authorization": f"Bearer {plaintext}"},
    )
    return client, runtime, store


@pytest.mark.parametrize("stream", [False, True])
async def test_in_process_auth_alias_usage_and_cleanup(stream):
    client, runtime, store = setup_gateway()
    async with client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "coder",
                "messages": [],
                "stream": stream,
                "x_gateway": {"include_usage": True},
            },
        )
        assert response.status_code == 200
        assert '"route":"local"' in response.text and '"model":"coder"' in response.text
        assert not runtime.admission.active
        record = next(iter(store.records.values()))
        assert record["status"] == "complete" and record["tokens"] == 3
        assert record["metadata"]["telemetry"]["tokens"] == 1


async def test_dynamic_policy_revoke_and_client_cannot_override_routing():
    client, _, store = setup_gateway()
    async with client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "coder", "messages": [], "provider": {"only": ["evil"]}},
        )
        assert response.status_code == 400 and not store.records
        store.policy = policy(enabled=False)
        assert (await client.get("/v1/models")).status_code == 401


async def test_cloud_budget_failure_never_dispatches():
    called = []

    async def handler(request):
        called.append(request)
        return httpx.Response(200, json={})

    cloud = OpenRouter(
        "secret",
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://cloud/"
        ),
    )
    alias = Alias(
        local_model="physical",
        configuration="test",
        cloud_model="provider/model",
        provider_only=("provider",),
        cloud_price_is_ceiling=True,
        price=Price(
            version="v1",
            input_per_million=1,
            cached_input_per_million=1,
            output_per_million=1,
        ),
    )
    client, _, store = setup_gateway(
        policy(local_policy="never"),
        GatewayConfig(aliases={"coder": alias}, cloud_enabled=True),
        cloud,
    )
    async with client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "coder", "messages": []}
        )
    assert response.status_code == 429 and not called
    assert next(iter(store.records.values()))["status"] == "failed"
    await cloud.close()


async def test_openrouter_pinning_and_real_cost():
    calls = []

    async def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "gen-test",
                "model": "physical",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.003},
            },
        )

    cloud = OpenRouter(
        "secret",
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://cloud/"
        ),
    )
    alias = Alias(
        local_model="physical",
        configuration="test",
        cloud_model="provider/model",
        provider_only=("safe",),
        cloud_price_is_ceiling=True,
        price=Price(
            version="v1",
            input_per_million=1,
            cached_input_per_million=1,
            output_per_million=1,
        ),
    )
    client, _, store = setup_gateway(
        policy(local_policy="never", max_cloud_cost_usd=1),
        GatewayConfig(aliases={"coder": alias}, cloud_enabled=True),
        cloud,
    )
    async with client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "coder", "messages": []}
        )
    assert response.status_code == 200
    assert calls[0]["model"] == "provider/model" and calls[0]["provider"]["only"] == [
        "safe"
    ]
    assert next(iter(store.records.values()))["cost"] == Decimal("0.003")
    await cloud.close()
