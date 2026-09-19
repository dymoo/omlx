# Local-first inference gateway: implementation and operations

Status: initial implementation, not production validated. Based on upstream
`jundot/omlx` commit `14194fe74bab38b89c144bd89656fbedca641d14`
(`0.7.0.dev4`). No Apple hardware benchmarks or real cloud requests were run.

## What is implemented

The optional gateway wraps the existing ASGI application in the same Python
process. Local requests call the original application directly; there is no
loopback HTTP server or separate TypeScript service. Gateway-disabled behavior
is preserved. Enable it with `OMLX_GATEWAY_CONFIG`.

- HMAC-SHA256 hashes of cryptographically random 256-bit API keys. The separate
  pepper is required; plaintext keys are printed once by the creation command.
- Postgres policies reloaded on every authenticated request, versioned changes,
  revocation, exact alias allowlists, context/output ceilings and concurrency.
- Atomic daily token reservations and per-key/global daily/monthly cloud spend
  reservations. UTC calendar windows. Outstanding unknown costs survive rollover.
- Priority admission (lower number means higher priority), bounded queue and
  queue deadlines; conservative empirical aggregate-throughput estimates protect
  configured decode floors. Unknown capacity is not advertised as guaranteed TPS.
- Generic runtime token callbacks; instantaneous, 1/5/15-second and lifetime TPS.
  Stable decode cohorts update bucketed EWMA capacity estimates in Postgres.
- OpenRouter provider allowlists/order, model pinning, streaming, timeout,
  cancellation, circuit breaker, usage and provider-reported charge accounting.
- Persistent cloud-only mode, queue wakeup, graceful local draining and idle
  model unload, including pinned models without changing their pin preferences.
- Optional `usage.x_gateway` accounting. Standard token fields are retained.
- Batched Postgres JSONB request/response logs; optional S3 adapter remains.
- Local administration CLI, example configurations and Linux control-plane CI.

Product names and subscription tiers do not appear in policy logic.

## What is not implemented or validated

This is **not the complete performance appliance from the design brief**.
The following remain explicit implementation/benchmark gates:

- Weighted *decode* allocation. `scheduling_weight` is carried as generic
  metadata but does not change per-step GPU service. Current active batches
  share decode opportunities; priorities affect admission and waiting order.
- Cache-affinity routing, tenant cache namespaces, per-key retention/stickiness,
  cache-priority eviction, multiple local nodes and predicted cache-hit admission.
  Upstream prefix/RAM/SSD caching remains available, but it has no gateway-tier
  policy in this patch. Do not promise tenant-isolated prefix caching yet.
- Direct bandwidth/GPU counters, hard instantaneous TPS guarantees, predictive
  prefill interference modelling and automatic calibration of unseen concurrency.
- Exact GLM-5.3-Flash mixed 4/8 quant compatibility and quality, compressed KV,
  native MTP and DFlash/DFlash2 combinations on the target hardware.
- Speculation acceptance telemetry adapters and model-specific cloud tokenizers.
- Streaming Responses state retrieval, multimodal billing, embeddings, rerank,
  audio and Anthropic endpoints under gateway policy. When the gateway is on,
  unsupported `/v1` endpoints fail closed; when off, upstream endpoints remain.
- Durable trace spool, JSONB partition rotation, automatic provider-charge
  reconciliation and fully allocated idle-time hardware costs.

Gateway local routing currently accepts only upstream `BatchedEngine`. DFlash
and other engine paths are rejected until they implement the same exact
context/token control hooks. This prevents silently bypassing policy enforcement.

## Install and bootstrap on the Mac

Install the fork using upstream's supported macOS/Python prerequisites:

```sh
pip install -e '.[gateway]'
export OMLX_GATEWAY_POSTGRES_DSN='postgresql://USER:PASSWORD@HOST/omlx'
# Supply a persistent random secret of at least 32 bytes from your secret store.
export OMLX_GATEWAY_PEPPER='REPLACE_WITH_PERSISTENT_RANDOM_SECRET'
omlx-gateway migrate
omlx-gateway hardware
```

Use a dedicated database and ordinary logged tables. The serving process takes
a database advisory lock: run **one serving worker**, not multiple Uvicorn workers.
Metadata admission fails closed if the database is unavailable. Postgres provides
shared, durable state; SQLite would be sufficient for a strictly single-machine
deployment, but this implementation uses the requested Postgres backend.

Copy and edit `examples/gateway/config.json`, replacing the physical model and
configuration fingerprint with the installed model and actual hardware/runtime
settings. No model sizes, memory bandwidths, model slugs or token prices are
invented by the code. The fingerprint must change when model revision, weights,
KV precision, speculative settings, runtime revision or hardware changes.

```sh
omlx-gateway policy internal examples/gateway/policy.json
omlx-gateway key --policy internal
export OMLX_GATEWAY_CONFIG='/absolute/path/to/config.json'
omlx serve --model-dir ~/models
```

The initial example is local-only and best effort. With no measured capacity
points, it admits a single best-effort request; requests demanding an unmeasured
floor queue/reject. Calibrate before enabling paid floors or concurrency. A
singleton can produce observations but cannot safely infer higher concurrency.

Keep upstream's owner/admin authentication configured for a non-loopback bind.
Gateway keys authorize the listed inference routes, not administrative routes.
Use `OPENAI_BASE_URL=http://localhost:8000/v1` and the generated gateway key.

## Release the hardware while keeping the API online

Configure each cloud-capable alias with a verified OpenRouter model slug,
provider allowlist, optional provider order, fallback flag, and a versioned price
ceiling covering **every** eligible provider. Enable the applicable per-key and
global daily/monthly budgets, per-request limit and `cloud_enabled` configuration.
Set `OPENROUTER_API_KEY` in the service environment. The initial database cloud
switch is off even if configuration says cloud is enabled.

```sh
omlx-gateway cloud on
omlx-gateway mode cloud_only --wait
omlx-gateway status
# Return to local-first operation; models load on demand.
omlx-gateway mode auto
```

`cloud_only` is persisted in Postgres and honored on restart before pinned model
preloading. Each new dispatch reads it; queued requests are awakened by the
control loop, normally within one second. The mode does not require a restart.

1. New cloud-eligible requests go directly to OpenRouter.
2. Queued local requests take their permitted cloud route or reject.
3. Active local requests drain; generated streams are never migrated/replayed.
4. The pool stops accepting new local acquisitions, including admin-initiated
   inference. Idle engines are unloaded, respecting active leases and loading.
5. `--wait` returns only after a fresh runtime heartbeat reports no active local
   tickets, no loaded models and no loading models. It can time out while a long
   generation or load drains; it never force-aborts those requests.

Local-only keys reject in this mode. A prefer-local key needs `cloud_fallback=true`.
Provider restrictions, quotas and budgets remain in force. A cloud outage or
exhausted budget returns an error; it does not quietly start using local hardware.
The API remains dependent on this Mac, its network and Postgres being up. This
mode frees inference resources; it is not availability failover for the gateway
host itself.

`omlx-gateway cloud off` is a separate emergency spending switch. It prevents
new cloud reservations and does not recall already-dispatched provider work.

## JSONB logging: inexpensive bulk ingestion

Default: `trace_backend="postgres"`. Each full-log request produces one trace
row with `request_id`, timestamps, expiry and a JSONB payload. The large payload
table is separate from the small authoritative spend ledger. Metadata-only keys
never log prompt/response bodies; `none` keeps only mandatory accounting state.

The writer buffers up to 100 rows, approximately 4 MiB, or one second, whichever
threshold is reached. It uses asyncpg binary COPY into a transaction-local staging
table, followed by one idempotent insert into `gateway_traces`. Retrying a batch
cannot duplicate request IDs. No per-token SQL writes. SSE is stored as one
bounded response string inside the document, not one database row per event.

The queue has both row and byte ceilings (1000 / 32 MiB). Three failed batch
attempts or a full queue drop diagnostic logs with counters; inference does not
block on trace ingestion. A crash can lose buffered diagnostic logs. Financial
reservations and settlements are synchronous separate transactions and never
depend on the best-effort queue. `trace_status=queued` is not proof of durability;
query the trace table for persisted records. Runtime status exposes queue bytes,
written rows and dropped rows. Graceful shutdown drains the writer with a timeout.

Default full-log limits are 1 MiB for input and 1 MiB for output. Oversized input
is omitted with its size recorded; output is truncated and flagged. Full trace
content can contain user secrets from prompts; auth headers and the provider API
key are never included. NUL and unpaired surrogate characters are normalized in
diagnostic logs because Postgres JSONB rejects them.

Only request-ID, expiry and BRIN time indexes are created. Avoid a blanket GIN
index over every prompt. Add a narrow expression index only for an actual query.
Run bounded retention deletion periodically from the deployment's scheduler:

```sh
omlx-gateway prune-traces --batch-size 1000
```

At sustained large log volume, migrate to time partitions and drop expired
partitions; this patch does not create partitions automatically. Repeated huge
agent prefixes are the main storage risk. Retention/sampling/deduplication will
matter more than reducing the already-small per-batch SQL overhead. Postgres
compresses large values via its normal TOAST behavior; choose compression based
on the installed server build. Do not disable WAL or durability on budget tables.

S3 remains optional with `trace_backend="s3"`, `OMLX_GATEWAY_TRACE_BUCKET` and
optional `OMLX_GATEWAY_S3_ENDPOINT`. It requires retention lifecycle rules;
the object's Expires metadata is **not** an automatic deletion mechanism.

## Capacity and accounting semantics

Smaller priority numbers win. New lower-priority work is not admitted when the
empirical per-stream prediction falls below any active protected floor. Without
weighted decode scheduling, the predictor conservatively uses aggregate/N.
Targets are descriptive, floors constrain admission; neither is a hard realtime
guarantee. A long incumbent can delay a newly arrived high-priority request.
Use short output limits, queue deadlines and overflow policy to limit that case.

The first predictor uses configuration, power-of-two context envelope bucket and
concurrency. It does not pretend to know cached-prefix tokens before upstream
cache lookup. A conservative JSON byte envelope rejects oversized requests early;
the local scheduler checks actual tokenized prompt + output allowance before
prefill. Cloud reservation uses the full policy context ceiling plus output
ceiling, without assuming provider cache hits. Exact cloud template/tokenizer
integration is a remaining production validation gate.

Stable cohorts must all have emitted tokens and remain unchanged for five
seconds before updating the capacity EWMA. Mixed models/configurations on one
GPU are not treated as independent capacity pools. No DRAM bandwidth reading is
fabricated: runtime reports null with source `unavailable`.

Requests reserve their total context token quota conservatively at admission.
Final usage reconciles it. Missing usage retains the reservation. Cloud requests
reserve a ceiling before dispatch; uncertain charges retain the hold even across
UTC day/month boundaries. There are no automatic post-dispatch retries because
a network failure does not prove a request was unexecuted or uncharged.

The ledger distinguishes:

- `cash_cogs`: provider-reported charge for cloud; explicitly estimated incremental
  energy for local. Unknown means null, not zero.
- `local_estimated_cogs`: estimated energy plus configured amortization per busy
  hour. Concurrent active requests split time so machine cost is not multiplied
  by concurrency. Idle amortization and overhead are not allocated yet.
- `counterfactual_cloud_cogs`: price-versioned comparison using uncached input.
  Local prefix hits do not imply cloud cache discounts.
- `subsidy`: counterfactual cloud cost minus local economic estimate, signed.
  This is an avoided-cost estimate, **not** cash received, profit, a cloud credit
  or customer subsidy net of revenue. Negative savings remain negative.

Reconcile uncertain charges from actual provider evidence, never by timeout:

```sh
omlx-gateway reconcile REQUEST_ID --cost ACTUAL_USD --tokens ACTUAL_TOTAL
```

A provider charge above its reservation is recorded and trips the global cloud
switch. Spend protection cannot make an incorrect provider price ceiling correct;
set an OpenRouter-side spending cap as a second bound before public traffic.
Operator reconciliation is trusted and must use actual evidence. Crashed running
requests retain holds/concurrency until reconciled rather than silently freeing
potentially spent budgets.

For usage details, send `"x_gateway":{"include_usage":true}`. Streaming Chat
Completions standard usage remains opt-in through `stream_options.include_usage`;
detailed gateway usage also requests it. Responses usage stays in its native
terminal event. There are no fabricated draft acceptance or bandwidth metrics.

## Exact-model investigation and benchmark gate

Inspected upstream source already includes continuous batching, RAM/SSD prefix
cache, native MTP patches and TurboQuant. Its DFlash report describes a distinct
engine path and lists supported target/drafter pairs; GLM-5.3-Flash is not listed.
The GLM MTP module is explicitly documented as GLM-5.2. TurboQuant includes
capability exclusions including MLA and attention sinks. These facts do not
establish support for the desired GLM-5.3 quant or the complete feature stack.

Before declaring a production configuration supported:

1. Pin model repository/revision, tokenizer, quant manifest, runtime dependencies
   and actual Mac identity/memory. Verify that it fits with OS/KV/draft headroom.
2. Establish plain batched decode correctness, tool-call behavior and long-context
   coding quality. Test compressed KV against that baseline.
3. Verify native MTP head/checkpoint compatibility. Treat DFlash/DFlash2 as a
   separate target+drafter capability investigation, not a drop-in checkbox.
4. Benchmark real coding/tool-agent prompts, cold/warm prefix distributions and
   short/long contexts at measured C1/C2/C4/C8/etc. Record TTFT, per-stream percentiles,
   aggregate TPS, memory, errors, acceptance and actual runtime configuration.
5. Test a long cold prefill alongside protected decode, cancellation, SSD restore,
   memory pressure and cloud-only drain/unload on the Mac.
6. Benchmark any weighted decode prototype against equal batching before enabling
   it. Do not trade away substantial aggregate throughput without measured benefit.
7. Add tenant-safe cache affinity/policy hooks only after inspecting cache identity
   and refcount/lifecycle behavior. Never claim private namespaces from labels alone.

## Validation and upstream synchronization

Linux tests:

```sh
python -m pytest tests/gateway --confcutdir=tests/gateway -q
python -m ruff check omlx/gateway omlx/scheduling.py tests/gateway
```

At delivery: **21 passed, 4 skipped**. Skipped tests are real Postgres checks:
concurrent spend reservations, unknown charges across rollover, COPY/idempotence/
retention and routing-mode persistence. Run with `GATEWAY_TEST_DSN`; the added CI
workflow supplies Postgres 16. The workspace could not install/start a usable
Postgres server, so the real database integration is not claimed validated.
No full upstream MLX suite ran on this Linux host. Test numerical/inference
behavior and model release on Apple Silicon before deployment.

The local checkout has `origin=https://github.com/dymoo/omlx.git` and
`upstream=https://github.com/jundot/omlx.git`. The connected account returned 404
for the requested fork; no remote repository or PR was created or pushed.
The delivered patch series applies to the pinned upstream commit.

```sh
git fetch upstream
git switch gateway/control-plane
git rebase upstream/main
```

If starting from a shallow clone, fetch sufficient history before rebasing.
Likely conflict hotspots: `server.py` middleware/auth/engine selection/lifespan,
`request.py` token hooks, `scheduler.py` token limits and waiting queue,
`engine_pool.py` acquisition suspension and graceful idle unload, and packaging
extras. All policy/storage/provider code lives under `omlx/gateway/`. The generic
request-control and idle-unload hooks are candidates for separate upstream PRs.
Never force-push a shared branch just to hide a rebase conflict.

## Primary references

- [oMLX upstream at the inspected revision](https://github.com/jundot/omlx/tree/14194fe74bab38b89c144bd89656fbedca641d14)
- [Upstream DFlash integration report](https://github.com/jundot/omlx/blob/14194fe74bab38b89c144bd89656fbedca641d14/docs/experimental/dflash_mlx_integration.md)
- [OpenRouter streaming and cancellation](https://openrouter.ai/docs/api_reference/streaming)
- [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [Postgres COPY guidance](https://www.postgresql.org/docs/current/populate.html)
- [Postgres JSONB](https://www.postgresql.org/docs/current/datatype-json.html)
- [asyncpg API](https://magicstack.github.io/asyncpg/current/api/index.html)
