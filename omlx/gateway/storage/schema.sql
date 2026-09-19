CREATE TABLE IF NOT EXISTS gateway_policies (
    id text PRIMARY KEY,
    version bigint NOT NULL DEFAULT 1,
    document jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS gateway_keys (
    id text PRIMARY KEY,
    digest text UNIQUE NOT NULL,
    policy_id text NOT NULL REFERENCES gateway_policies(id),
    enabled boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS gateway_requests (
    id text PRIMARY KEY,
    key_id text NOT NULL REFERENCES gateway_keys(id),
    policy_version bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL DEFAULT 'running',
    tokens_reserved bigint NOT NULL CHECK(tokens_reserved >= 0),
    tokens_actual bigint CHECK(tokens_actual >= 0),
    cloud_reserved numeric(24,12) NOT NULL DEFAULT 0 CHECK(cloud_reserved >= 0),
    cloud_actual numeric(24,12) CHECK(cloud_actual >= 0),
    provider_id text,
    metadata jsonb NOT NULL DEFAULT '{}',
    trace_key text
);
CREATE INDEX IF NOT EXISTS gateway_requests_key_time
    ON gateway_requests(key_id, created_at);
CREATE INDEX IF NOT EXISTS gateway_requests_unsettled
    ON gateway_requests(status) WHERE status = 'running' OR cloud_actual IS NULL;
CREATE TABLE IF NOT EXISTS gateway_switches (
    id text PRIMARY KEY,
    enabled boolean NOT NULL
);
INSERT INTO gateway_switches(id, enabled) VALUES ('cloud', false)
    ON CONFLICT DO NOTHING;

-- Large payloads are isolated from the frequently queried spend ledger.
-- Deliberately no blanket GIN index over prompt/response JSON.
CREATE TABLE IF NOT EXISTS gateway_traces (
    request_id text PRIMARY KEY,
    created_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS gateway_traces_expiry ON gateway_traces(expires_at);
CREATE INDEX IF NOT EXISTS gateway_traces_time ON gateway_traces USING brin(created_at);

CREATE TABLE IF NOT EXISTS gateway_runtime (
    id text PRIMARY KEY,
    document jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO gateway_runtime(id,document) VALUES ('routing','{"mode":"auto"}')
    ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS gateway_capacity (
    configuration text NOT NULL,
    context_bucket bigint NOT NULL,
    concurrency integer NOT NULL,
    aggregate_tps double precision NOT NULL,
    samples bigint NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(configuration, context_bucket, concurrency)
);
