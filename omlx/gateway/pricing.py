"""Decimal accounting. Unknown cost stays unknown; estimates are labelled."""

from decimal import Decimal, InvalidOperation

from .policy import Price

MILLION = Decimal(1_000_000)


def token_cost(price: Price, prompt: int, output: int, cached: int = 0):
    if min(prompt, output, cached) < 0 or cached > prompt:
        raise ValueError("invalid token accounting")
    return (
        price.per_request
        + (
            (prompt - cached) * price.input_per_million
            + cached * price.cached_input_per_million
            + output * price.output_per_million
        )
        / MILLION
    )


def costs(config, alias, usage, route, allocated_seconds):
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    output = usage.get("completion_tokens", usage.get("output_tokens"))
    valid_tokens = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (prompt, output)
    )
    # Local cache hits do not imply an equivalent cloud cache hit.
    counterfactual = (
        token_cost(alias.price, prompt, output)
        if alias.price and valid_tokens
        else None
    )
    cash = economic = None
    if route == "cloud":
        value = usage.get("cost")
        try:
            cash = Decimal(str(value)) if value is not None else None
        except InvalidOperation:
            cash = None
        if cash is not None and (not cash.is_finite() or cash < 0):
            cash = None
    elif (
        config.incremental_watts is not None
        and config.electricity_usd_per_kwh is not None
    ):
        hours = Decimal(str(allocated_seconds)) / 3600
        cash = hours * config.incremental_watts / 1000 * config.electricity_usd_per_kwh
        if config.amortization_usd_per_busy_hour is not None:
            economic = cash + hours * config.amortization_usd_per_busy_hour
    saving = (
        counterfactual - economic
        if route == "local" and counterfactual is not None and economic is not None
        else None
    )
    return {
        "currency": "USD",
        "cash_cogs": cash,
        "cash_cogs_source": "provider_reported"
        if route == "cloud"
        else "estimated_energy",
        "local_estimated_cogs": economic,
        "counterfactual_cloud_cogs": counterfactual,
        "counterfactual_cache_assumption": "uncached",
        "price_version": alias.price.version if alias.price else None,
        "subsidy": saving,
        "subsidy_percent": saving / counterfactual * 100
        if saving is not None and counterfactual
        else None,
    }
