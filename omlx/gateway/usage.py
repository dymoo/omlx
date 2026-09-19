"""Bounded SSE/JSON accounting with opt-in namespaced usage extensions."""

import json
import re
from decimal import Decimal


def encode_json(value):
    return json.dumps(
        value,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda x: float(x) if isinstance(x, Decimal) else str(x),
    ).encode()


class UsageObserver:
    def __init__(
        self,
        alias,
        extension=None,
        max_frame_bytes=4 * 1024 * 1024,
        include_stream_usage=True,
    ):
        self.alias = alias
        self.extension = extension
        self.max_frame_bytes = max_frame_bytes
        self.buffer = b""
        self.usage = {}
        self.provider_id = None
        self.failed = False
        self.completed = False
        self.include_stream_usage = include_stream_usage

    def document(self, document):
        if not isinstance(document, dict):
            return document
        self.failed |= bool(document.get("error")) or document.get("type") in (
            "error",
            "response.failed",
        )
        root = document.get("response")
        if not isinstance(root, dict):
            root = document
        if isinstance(root.get("id"), str):
            self.provider_id = root["id"]
        if "model" in root:
            root["model"] = self.alias
        usage = root.get("usage")
        if isinstance(usage, dict):
            # Copy before extension so private caller fields cannot poison ledger.
            self.usage.update({k: v for k, v in usage.items() if k != "x_gateway"})
            if self.extension:
                usage["x_gateway"] = self.extension(self.usage)
        if document.get("type") in ("response.completed", "response.incomplete"):
            self.completed = True
        return document

    def feed(self, data, final=False):
        self.buffer += data
        output = bytearray()
        while match := re.search(rb"\r?\n\r?\n", self.buffer):
            frame, self.buffer = (
                self.buffer[: match.start()],
                self.buffer[match.end() :],
            )
            if len(frame) > self.max_frame_bytes:
                raise ValueError("provider SSE frame too large")
            lines = frame.splitlines()
            payload = b"\n".join(
                line[5:].lstrip(b" ") for line in lines if line.startswith(b"data:")
            )
            if payload == b"[DONE]":
                self.completed = True
            elif payload:
                try:
                    value = self.document(json.loads(payload))
                    if (
                        not self.include_stream_usage
                        and isinstance(value, dict)
                        and "usage" in value
                    ):
                        value.pop("usage", None)
                        if value.get("choices") == []:
                            continue
                    lines = [line for line in lines if not line.startswith(b"data:")]
                    lines.append(b"data: " + encode_json(value))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.failed = True
            output.extend(b"\n".join(lines) + b"\n\n")
        if len(self.buffer) > self.max_frame_bytes:
            raise ValueError("provider SSE frame too large")
        if final and self.buffer.strip():
            self.failed = True  # Truncated event; never count as clean completion.
            self.buffer = b""
        return bytes(output)

    def total_tokens(self):
        prompt = self.usage.get("prompt_tokens", self.usage.get("input_tokens"))
        output = self.usage.get("completion_tokens", self.usage.get("output_tokens"))
        if all(
            isinstance(n, int) and not isinstance(n, bool) and n >= 0
            for n in (prompt, output)
        ):
            return prompt + output
        return None
