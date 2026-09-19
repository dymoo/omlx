"""Atomic quotas and spend reservations. Database errors fail admission closed."""

import json
from decimal import Decimal
from pathlib import Path

from ..policy import CapacityPoint, KeyPolicy

# One database transaction lock covers both global and per-key counters.
# Deliberately simple for a single appliance; no check-then-increment race.
LEDGER_LOCK = 72709131
APPLIANCE_LOCK = 72709132


class LimitExceededError(Exception):
    pass


class Postgres:
    def __init__(self, pool):
        self.pool = pool
        self.owner = None

    @classmethod
    async def connect(cls, dsn):
        import asyncpg

        return cls(
            await asyncpg.create_pool(dsn, min_size=1, max_size=8, command_timeout=15)
        )

    async def claim_appliance(self):
        self.owner = await self.pool.acquire()
        if not await self.owner.fetchval(
            "SELECT pg_try_advisory_lock($1)", APPLIANCE_LOCK
        ):
            await self.pool.release(self.owner)
            self.owner = None
            raise RuntimeError("Gateway supports one serving process per database")

    async def close(self):
        if self.owner:
            await self.pool.release(self.owner)
        await self.pool.close()

    async def migrate(self):
        await self.pool.execute(Path(__file__).with_name("schema.sql").read_text())

    async def identity(self, digest):
        row = await self.pool.fetchrow(
            "SELECT k.id, p.version, p.document FROM gateway_keys k "
            "JOIN gateway_policies p ON p.id=k.policy_id "
            "WHERE k.digest=$1 AND k.enabled",
            digest,
        )
        if row is None:
            return None
        policy = KeyPolicy.model_validate_json(row["document"])
        return (row["id"], row["version"], policy) if policy.enabled else None

    async def begin(self, request_id, key_id, policy_version, policy):
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock($1)", LEDGER_LOCK)
            valid = await connection.fetchval(
                "SELECT k.enabled AND (p.document->>'enabled')::boolean "
                "AND p.version=$2 FROM gateway_keys k JOIN gateway_policies p "
                "ON k.policy_id=p.id WHERE k.id=$1",
                key_id,
                policy_version,
            )
            if not valid:
                raise LimitExceededError("policy_changed_or_key_disabled")
            count = await connection.fetchval(
                "SELECT count(*) FROM gateway_requests WHERE key_id=$1 AND status='running'",
                key_id,
            )
            if count >= policy.max_concurrency:
                raise LimitExceededError("key_concurrency")
            used = await connection.fetchval(
                "SELECT COALESCE(sum(COALESCE(tokens_actual,tokens_reserved)),0) "
                "FROM gateway_requests WHERE key_id=$1 AND "
                "(created_at >= date_trunc('day',now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC' "
                "OR tokens_actual IS NULL)",
                key_id,
            )
            if (
                policy.daily_token_quota is not None
                and used + policy.max_context_tokens > policy.daily_token_quota
            ):
                raise LimitExceededError("daily_token_quota")
            await connection.execute(
                "INSERT INTO gateway_requests(id,key_id,policy_version,tokens_reserved) VALUES($1,$2,$3,$4)",
                request_id,
                key_id,
                policy_version,
                policy.max_context_tokens,
            )

    async def reserve_cloud(self, request_id, key_id, policy, config, amount):
        if amount <= 0 or amount > policy.max_cloud_cost_usd:
            raise LimitExceededError("request_cloud_budget")
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock($1)", LEDGER_LOCK)
            enabled = await connection.fetchval(
                "SELECT enabled FROM gateway_switches WHERE id='cloud'"
            )
            if not config.cloud_enabled or not enabled:
                raise LimitExceededError("cloud_disabled")
            previous = await connection.fetchval(
                "SELECT cloud_reserved FROM gateway_requests WHERE id=$1 AND key_id=$2 AND status='running' FOR UPDATE",
                request_id,
                key_id,
            )
            if previous is None or previous != 0:
                raise LimitExceededError("invalid_reservation_state")
            for window, key_limit, global_limit in (
                (
                    "day",
                    policy.daily_cloud_budget_usd,
                    config.global_daily_cloud_budget_usd,
                ),
                (
                    "month",
                    policy.monthly_cloud_budget_usd,
                    config.global_monthly_cloud_budget_usd,
                ),
            ):
                row = await connection.fetchrow(
                    "SELECT COALESCE(sum(COALESCE(cloud_actual,cloud_reserved)) FILTER(WHERE key_id=$1),0) AS key_used, "
                    "COALESCE(sum(COALESCE(cloud_actual,cloud_reserved)),0) AS total "
                    "FROM gateway_requests WHERE created_at >= date_trunc($2,now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC' "
                    "OR (cloud_actual IS NULL AND cloud_reserved>0)",
                    key_id,
                    window,
                )
                if (
                    row["key_used"] + amount > key_limit
                    or row["total"] + amount > global_limit
                ):
                    raise LimitExceededError(f"{window}_cloud_budget")
            await connection.execute(
                "UPDATE gateway_requests SET cloud_reserved=$2 WHERE id=$1",
                request_id,
                amount,
            )

    async def finish(
        self, request_id, status, tokens, cloud_cost, metadata, trace_key=None
    ):
        # Unknown cloud cost remains NULL, preserving the reservation across
        # UTC rollovers and restart. Only an explicit reconciliation releases it.
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock($1)", LEDGER_LOCK)
            await connection.execute(
                "UPDATE gateway_requests SET status=$2, finished_at=now(), tokens_actual=$3, "
                "cloud_actual=$4, metadata=$5::jsonb, trace_key=$6, provider_id=$7 "
                "WHERE id=$1 AND status='running'",
                request_id,
                status,
                tokens,
                cloud_cost,
                json.dumps(metadata, default=str),
                trace_key,
                metadata.get("provider_id"),
            )
            if cloud_cost is not None:
                reserved = await connection.fetchval(
                    "SELECT cloud_reserved FROM gateway_requests WHERE id=$1",
                    request_id,
                )
                if cloud_cost > reserved:
                    # A rate ceiling was wrong: persist the real cost and trip.
                    await connection.execute(
                        "UPDATE gateway_switches SET enabled=false WHERE id='cloud'"
                    )

    async def put_policy(self, policy_id, policy):
        await self.pool.execute(
            "INSERT INTO gateway_policies(id,document) VALUES($1,$2::jsonb) "
            "ON CONFLICT(id) DO UPDATE SET document=EXCLUDED.document, version=gateway_policies.version+1",
            policy_id,
            policy.model_dump_json(),
        )

    async def write_traces(self, rows):
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS gateway_trace_batch "
                "(LIKE gateway_traces INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
            )
            await connection.copy_records_to_table(
                "gateway_trace_batch",
                records=rows,
                columns=("request_id", "created_at", "expires_at", "payload"),
            )
            await connection.execute(
                "INSERT INTO gateway_traces SELECT * FROM gateway_trace_batch "
                "ON CONFLICT(request_id) DO NOTHING"
            )

    async def prune_traces(self, batch_size=1000):
        return await self.pool.execute(
            "DELETE FROM gateway_traces WHERE request_id IN ("
            "SELECT request_id FROM gateway_traces WHERE expires_at < now() "
            "ORDER BY expires_at LIMIT $1)",
            batch_size,
        )

    async def put_key(self, key_id, digest, policy_id):
        await self.pool.execute(
            "INSERT INTO gateway_keys(id,digest,policy_id) VALUES($1,$2,$3)",
            key_id,
            digest,
            policy_id,
        )

    async def revoke(self, key_id):
        await self.pool.execute(
            "UPDATE gateway_keys SET enabled=false WHERE id=$1", key_id
        )

    async def cloud_switch(self, enabled):
        await self.pool.execute(
            "UPDATE gateway_switches SET enabled=$1 WHERE id='cloud'", enabled
        )

    async def routing_mode(self):
        return await self.pool.fetchval(
            "SELECT document->>'mode' FROM gateway_runtime WHERE id='routing'"
        )

    async def capacity_points(self):
        rows = await self.pool.fetch(
            "SELECT configuration,context_bucket,concurrency,aggregate_tps,samples FROM gateway_capacity"
        )
        return tuple(CapacityPoint(**dict(row)) for row in rows)

    async def record_capacity(self, configuration, bucket, concurrency, estimate):
        await self.pool.execute(
            "INSERT INTO gateway_capacity(configuration,context_bucket,concurrency,aggregate_tps,samples) "
            "VALUES($1,$2,$3,$4,$5) ON CONFLICT(configuration,context_bucket,concurrency) DO UPDATE SET "
            "aggregate_tps=EXCLUDED.aggregate_tps,samples=EXCLUDED.samples,updated_at=now()",
            configuration,
            bucket,
            concurrency,
            estimate.aggregate_tps,
            estimate.samples,
        )

    async def set_routing_mode(self, mode):
        if mode not in ("auto", "cloud_only"):
            raise ValueError("invalid routing mode")
        await self.pool.execute(
            "UPDATE gateway_runtime SET document=$1::jsonb,updated_at=now() WHERE id='routing'",
            json.dumps({"mode": mode}),
        )

    async def report_runtime(self, document):
        await self.pool.execute(
            "INSERT INTO gateway_runtime(id,document) VALUES('status',$1::jsonb) "
            "ON CONFLICT(id) DO UPDATE SET document=EXCLUDED.document,updated_at=now()",
            json.dumps(document),
        )

    async def runtime_status(self):
        row = await self.pool.fetchrow(
            "SELECT document, updated_at > now()-interval '5 seconds' AS fresh FROM gateway_runtime WHERE id='status'"
        )
        return (
            {**json.loads(row["document"]), "fresh": row["fresh"]}
            if row
            else {"fresh": False}
        )

    async def reconcile(self, request_id, actual_cost, actual_tokens):
        if actual_cost < 0 or not actual_cost.is_finite() or actual_tokens < 0:
            raise ValueError("invalid reconciliation")
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock($1)", LEDGER_LOCK)
            await connection.execute(
                "UPDATE gateway_requests SET cloud_actual=$2, tokens_actual=$3, "
                "status=CASE WHEN status='running' THEN 'reconciled' ELSE status END, "
                "finished_at=COALESCE(finished_at,now()) WHERE id=$1",
                request_id,
                Decimal(actual_cost),
                actual_tokens,
            )
