"""Streaming OpenRouter transport with explicit provider policy and no replay."""

import time

import httpx


class CircuitOpenError(Exception):
    pass


class OpenRouter:
    def __init__(self, api_key, client=None):
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1/",
            timeout=httpx.Timeout(60, connect=10),
            follow_redirects=False,
        )
        self.failures = 0
        self.open_until = 0

    async def close(self):
        await self.client.aclose()

    def failed(self):
        self.failures += 1
        if self.failures >= 3:
            self.open_until = time.monotonic() + 30

    async def serve(self, path, body, alias, send):
        if time.monotonic() < self.open_until:
            raise CircuitOpenError("cloud_circuit_open")
        payload = dict(body)
        # Routing controls are never inherited from a caller's JSON.
        for field in ("models", "route", "provider", "plugins", "transforms", "usage"):
            payload.pop(field, None)
        payload["model"] = alias.cloud_model
        payload["provider"] = {
            "order": list(alias.provider_order),
            "only": list(alias.provider_only),
            "allow_fallbacks": alias.allow_provider_fallback,
            "require_parameters": True,
        }
        # Reject paid plugins and auxiliary services at gateway validation.
        # Pricing ceiling must cover every permitted provider and output token.
        response = None
        try:
            request = self.client.build_request(
                "POST",
                path.removeprefix("/v1/"),
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response = await self.client.send(request, stream=True)
            if response.status_code >= 500 or response.status_code == 429:
                self.failed()
            else:
                self.failures = 0
            headers = [
                (
                    b"content-type",
                    response.headers.get("content-type", "application/json").encode(),
                )
            ]
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status_code,
                    "headers": headers,
                }
            )
            async for data in response.aiter_bytes():
                await send(
                    {"type": "http.response.body", "body": data, "more_body": True}
                )
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except httpx.HTTPError:
            self.failed()
            raise
        finally:
            if response is not None:
                await response.aclose()
        # No retries after dispatch: timeout does not prove the provider did
        # not execute/charge. Each future explicit retry needs a new reservation.
