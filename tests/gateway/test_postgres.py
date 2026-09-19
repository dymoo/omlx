"""Real PostgreSQL gates. Set GATEWAY_TEST_DSN; CI supplies PostgreSQL 16."""

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from test_control_plane import policy

from omlx.gateway.policy import GatewayConfig
from omlx.gateway.storage.postgres import LimitExceededError, Postgres

pytestmark = pytest.mark.skipif(
    not os.getenv("GATEWAY_TEST_DSN"), reason="GATEWAY_TEST_DSN not set"
)


@pytest.fixture
async def store():
    import asyncpg

    dsn = os.environ["GATEWAY_TEST_DSN"]
    schema = "gateway_test_" + uuid.uuid4().hex
    admin = await asyncpg.connect(dsn)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(
        dsn, min_size=1, max_size=8, server_settings={"search_path": schema}
    )
    store = Postgres(pool)
    await store.migrate()
    yield store
    await store.close()
    await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
    await admin.close()


async def seed(store):
    pol = policy(
        local_policy="prefer",
        cloud_fallback=True,
        max_concurrency=20,
        daily_cloud_budget_usd=1,
        monthly_cloud_budget_usd=2,
        max_cloud_cost_usd=1,
    )
    await store.put_policy("p", pol)
    await store.put_key("k", "digest", "p")
    await store.cloud_switch(True)
    return pol


async def test_cloud_budget_race_is_atomic(store):
    pol = await seed(store)
    config = GatewayConfig(
        aliases={},
        cloud_enabled=True,
        global_daily_cloud_budget_usd=1,
        global_monthly_cloud_budget_usd=2,
    )

    async def reserve(i):
        await store.begin(str(i), "k", 1, pol)
        await store.reserve_cloud(str(i), "k", pol, config, Decimal("0.6"))

    results = await asyncio.gather(
        *(reserve(i) for i in range(10)), return_exceptions=True
    )
    assert sum(r is None for r in results) == 1
    assert all(r is None or isinstance(r, LimitExceededError) for r in results)
    assert await store.pool.fetchval(
        "SELECT sum(cloud_reserved) FROM gateway_requests"
    ) == Decimal("0.6")


async def test_unknown_cloud_cost_survives_midnight_and_restart(store):
    pol = await seed(store)
    config = GatewayConfig(
        aliases={},
        cloud_enabled=True,
        global_daily_cloud_budget_usd=1,
        global_monthly_cloud_budget_usd=2,
    )
    await store.begin("old", "k", 1, pol)
    await store.reserve_cloud("old", "k", pol, config, Decimal("0.6"))
    await store.finish("old", "cancelled", None, None, {})
    await store.pool.execute(
        "UPDATE gateway_requests SET created_at=now()-interval '40 days'"
    )
    await store.begin("new", "k", 1, pol)
    with pytest.raises(LimitExceededError):
        await store.reserve_cloud("new", "k", pol, config, Decimal("0.6"))
    await store.reconcile("old", Decimal("0.1"), 10)
    await store.reserve_cloud("new", "k", pol, config, Decimal("0.6"))


async def test_copy_idempotence_and_retention(store):
    now = datetime.now(UTC)
    rows = [
        (
            "one",
            now,
            now - timedelta(days=1),
            json.dumps({"request": {"content": "hello"}}),
        )
    ]
    await store.write_traces(rows)
    await store.write_traces(rows)
    assert await store.pool.fetchval("SELECT count(*) FROM gateway_traces") == 1
    await store.prune_traces()
    assert await store.pool.fetchval("SELECT count(*) FROM gateway_traces") == 0


async def test_persistent_routing_mode(store):
    assert await store.routing_mode() == "auto"
    await store.set_routing_mode("cloud_only")
    assert await store.routing_mode() == "cloud_only"
    await store.report_runtime({"mode": "cloud_only", "hardware_released": True})
    assert (await store.runtime_status())["fresh"]
