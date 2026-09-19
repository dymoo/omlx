"""High-entropy bearer keys; only a keyed digest is persisted."""

import hashlib
import hmac
import secrets


def digest_key(key: str, pepper: bytes) -> str:
    if len(pepper) < 32:
        raise ValueError("OMLX_GATEWAY_PEPPER must contain at least 32 bytes")
    return hmac.new(pepper, key.encode(), hashlib.sha256).hexdigest()


def create_key(pepper: bytes) -> tuple[str, str]:
    plaintext = "omlx_" + secrets.token_urlsafe(32)
    return plaintext, digest_key(plaintext, pepper)


def bearer(headers: list[tuple[bytes, bytes]]) -> str | None:
    values = [v for k, v in headers if k.lower() == b"authorization"]
    if len(values) != 1:
        return None
    scheme, _, value = values[0].partition(b" ")
    if scheme.lower() != b"bearer" or not value or len(value) > 256:
        return None
    try:
        return value.decode("ascii")
    except UnicodeDecodeError:
        return None
