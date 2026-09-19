import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_control_plane import policy, setup_gateway

from omlx.gateway.admission import Admission, AdmissionDeniedError, Ticket
from omlx.gateway.capacity import CapacityModel
from omlx.gateway.policy import Alias, GatewayConfig, Price
from omlx.gateway.providers.openrouter import OpenRouter
from omlx.gateway.storage.trace_writer import TraceWriter
from omlx.scheduling import set_local_inference_suspended


@pytest.fixture(autouse=True)
def reset_suspension():
    set_local_inference_suspended(False)
    yield
    set_local_inference_suspended(False)


async def test_cloud_only_wakes_queue_without_cancelling_active_generation():
    admission = Admission(CapacityModel())
    active = Ticket("active", "k", policy(), "c", 1024)
    queued = Ticket("queued", "k", policy(max_local_queue_ms=10000), "c", 1024)
    await admission.acquire(active)
    waiter = asyncio.create_task(admission.acquire(queued))
    await asyncio.sleep(0)
    await admission.set_mode("cloud_only")
    with pytest.raises(AdmissionDeniedError) as error:
        await waiter
    assert error.value.action == "cloud"
    assert list(admission.active) == ["active"]
    assert not admission.waiting
    await admission.release(active)
    await admission.set_mode("auto")
    await admission.acquire(queued)


def cloud_alias():
    return Alias(
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


async def test_cloud_only_bypasses_local_and_still_rejects_local_only_keys():
    local_calls = []

    async def forbidden_local(*args):
        local_calls.append(args)
        raise AssertionError("local dispatch in cloud-only mode")

    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={"usage": {"prompt_tokens": 2, "completion_tokens": 1, "cost": 0.01}},
        )

    cloud = OpenRouter(
        "secret",
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://cloud/"
        ),
    )
    client, runtime, store = setup_gateway(
        policy(local_policy="prefer", cloud_fallback=True, max_cloud_cost_usd=1),
        GatewayConfig(aliases={"coder": cloud_alias()}, cloud_enabled=True),
        cloud,
        forbidden_local,
    )
    store.mode = "cloud_only"
    async with client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "coder", "messages": []}
        )
        assert response.status_code == 200 and len(calls) == 1
        store.policy = policy(local_policy="only")
        response = await client.post(
            "/v1/chat/completions", json={"model": "coder", "messages": []}
        )
        assert response.status_code == 429 and len(calls) == 1
    assert not local_calls and not runtime.admission.active
    await cloud.close()


def pool_method(name):
    # Exercise the actual small lifecycle method without importing Apple-only
    # dependencies on Linux. Hardware tests still gate deployment on macOS.
    source = Path("omlx/engine_pool.py").read_text()
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EnginePool"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(module), "engine_pool.py", "exec"), namespace
    )
    return namespace[name]


async def test_drain_unloads_pinned_idle_models_but_preserves_busy_models():
    unloaded = []
    entries = {
        "pinned": SimpleNamespace(
            engine=object(), is_loading=False, is_pinned=True, in_use=0
        ),
        "busy": SimpleNamespace(
            engine=object(), is_loading=False, is_pinned=False, in_use=1
        ),
        "loading": SimpleNamespace(
            engine=object(), is_loading=True, is_pinned=False, in_use=0
        ),
    }

    async def unload(name):
        unloaded.append(name)
        entries[name].engine = None

    pool = SimpleNamespace(
        _lock=asyncio.Lock(),
        _entries=entries,
        _entry_is_quiescent=lambda e: e.in_use == 0,
        _unload_engine=unload,
        get_loaded_model_ids=lambda: [
            k for k, e in entries.items() if e.engine is not None
        ],
    )
    state = await pool_method("unload_idle_models")(pool, include_pinned=True)
    assert unloaded == ["pinned"] and entries["pinned"].is_pinned
    assert state["loading_models"] == ["loading"]
    entries["busy"].in_use = 0
    await pool_method("unload_idle_models")(pool, include_pinned=True)
    assert unloaded == ["pinned", "busy"]


class TraceStore:
    def __init__(self, failures=0):
        self.failures = failures
        self.batches = []

    async def write_traces(self, rows):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("offline")
        self.batches.append(rows)


async def test_trace_copy_batches_and_flushes_on_shutdown():
    store = TraceStore(failures=1)
    writer = TraceWriter(store, max_rows=3, flush_seconds=0.01)
    writer.start()
    for i in range(7):
        assert writer.submit(str(i), {"request": "hello\x00"}, 7)
    await writer.close()
    assert [len(b) for b in store.batches] == [3, 3, 1]
    assert writer.written == 7 and writer.dropped == 0 and writer.queued_bytes == 0
    assert "\\u0000" not in store.batches[0][0][3]


async def test_trace_backpressure_drops_but_does_not_block():
    store = TraceStore()
    writer = TraceWriter(store, max_queue_bytes=10)
    assert not writer.submit("id", {"request": "large payload"}, 1)
    assert writer.dropped == 1 and not store.batches


async def test_responses_completed_event_retains_usage():
    from omlx.gateway.usage import UsageObserver

    observer = UsageObserver("coder")
    result = observer.feed(
        b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r","model":"physical","usage":{"input_tokens":7,"output_tokens":2}}}\n\n'
    )
    assert observer.completed and observer.total_tokens() == 9
    assert b'"model":"coder"' in result
