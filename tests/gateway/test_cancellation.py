import asyncio
import json

from test_control_plane import policy, setup_gateway
from test_operations import cloud_alias

from omlx.gateway.middleware import GatewayMiddleware
from omlx.gateway.policy import GatewayConfig
from omlx.gateway.providers.openrouter import OpenRouter


async def test_disconnect_cancels_local_and_releases_admission():
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def local(scope, receive, send):
        await receive()
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    client, runtime, store = setup_gateway(app=local)
    key = client.headers["authorization"]
    await client.aclose()
    gateway = GatewayMiddleware(local, runtime=runtime)
    received_body = False

    async def receive():
        nonlocal received_body
        if not received_body:
            received_body = True
            return {
                "type": "http.request",
                "body": json.dumps({"model": "coder", "messages": []}).encode(),
            }
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    async with asyncio.timeout(1):
        await gateway(
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "method": "POST",
                "headers": [(b"authorization", key.encode())],
            },
            receive,
            send,
        )
    assert stopped.is_set() and not runtime.admission.active
    assert next(iter(store.records.values()))["status"] == "cancelled"


async def test_cloud_transport_failure_keeps_unknown_cost_and_never_retries():
    import httpx

    calls = []

    async def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("connection lost after dispatch")

    cloud = OpenRouter(
        "secret",
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://cloud/"
        ),
    )
    client, _, store = setup_gateway(
        policy(local_policy="never", max_cloud_cost_usd=1),
        GatewayConfig(aliases={"coder": cloud_alias()}, cloud_enabled=True),
        cloud,
    )
    async with client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "coder", "messages": []}
        )
    assert response.status_code == 503 and len(calls) == 1
    record = next(iter(store.records.values()))
    assert record["status"] == "failed" and record["cost"] is None
    await cloud.close()
