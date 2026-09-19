"""Versioned, product-independent configuration; lower priority wins."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class KeyPolicy(ConfigModel):
    enabled: bool = True
    priority: int = Field(default=100, ge=0)
    scheduling_weight: int = Field(default=1, ge=1, le=1000)
    allowed_model_aliases: tuple[str, ...]
    local_policy: Literal["only", "prefer", "never"] = "only"
    target_tps: float | None = Field(default=None, gt=0)
    min_tps: float | None = Field(default=None, gt=0)
    below_floor: Literal["queue", "cloud", "reject"] = "queue"
    queue_timeout: Literal["cloud", "reject"] = "reject"
    max_context_tokens: int = Field(default=32768, ge=1)
    max_output_tokens: int = Field(default=4096, ge=1)
    max_concurrency: int = Field(default=2, ge=1)
    max_local_queue_ms: int = Field(default=2000, ge=0)
    daily_token_quota: int | None = Field(default=None, ge=1)
    cloud_fallback: bool = False
    daily_cloud_budget_usd: Decimal = Field(default=Decimal(0), ge=0)
    monthly_cloud_budget_usd: Decimal = Field(default=Decimal(0), ge=0)
    max_cloud_cost_usd: Decimal = Field(default=Decimal(0), ge=0)
    logging_policy: Literal["none", "metadata", "full"] = "metadata"
    trace_ttl_days: int = Field(default=7, ge=1, le=365)

    @model_validator(mode="after")
    def consistent(self):
        if self.target_tps and self.min_tps and self.min_tps > self.target_tps:
            raise ValueError("min_tps must not exceed target_tps")
        if self.max_output_tokens > self.max_context_tokens:
            raise ValueError("output ceiling must fit context ceiling")
        if self.local_policy == "only" and self.cloud_fallback:
            raise ValueError("local-only policy cannot enable cloud fallback")
        return self


class Price(ConfigModel):
    version: str
    input_per_million: Decimal = Field(ge=0)
    cached_input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    per_request: Decimal = Field(default=Decimal(0), ge=0)


class Alias(ConfigModel):
    local_model: str
    # Must identify hardware, model revision, quant, KV and speculation settings.
    configuration: str
    cloud_model: str | None = None
    provider_order: tuple[str, ...] = ()
    provider_only: tuple[str, ...] = ()
    allow_provider_fallback: bool = False
    price: Price | None = None
    # Explicit operator assertion: rates bound all eligible provider charges.
    cloud_price_is_ceiling: bool = False


class CapacityPoint(ConfigModel):
    configuration: str
    context_bucket: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    aggregate_tps: float = Field(gt=0)
    samples: int = Field(default=1, ge=1)


class GatewayConfig(ConfigModel):
    aliases: dict[str, Alias]
    capacity_points: tuple[CapacityPoint, ...] = ()
    capacity_margin: float = Field(default=0.15, ge=0, lt=1)
    global_daily_cloud_budget_usd: Decimal = Field(default=Decimal(0), ge=0)
    global_monthly_cloud_budget_usd: Decimal = Field(default=Decimal(0), ge=0)
    cloud_enabled: bool = False
    max_queue_depth: int = Field(default=256, ge=1)
    max_request_bytes: int = Field(default=4 * 1024 * 1024, ge=1024)
    max_trace_bytes: int = Field(default=1024 * 1024, ge=0)
    trace_backend: Literal["postgres", "s3"] = "postgres"
    request_timeout_seconds: int = Field(default=600, ge=1, le=3600)
    electricity_usd_per_kwh: Decimal | None = Field(default=None, ge=0)
    incremental_watts: Decimal | None = Field(default=None, ge=0)
    amortization_usd_per_busy_hour: Decimal | None = Field(default=None, ge=0)
