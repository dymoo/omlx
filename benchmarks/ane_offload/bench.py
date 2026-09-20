#!/usr/bin/env python3
"""ANE projection feasibility and OpenAI streaming benchmarks. No runtime patches."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def percentile(values, q):
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def distribution(values):
    return {
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "samples": values,
    }


def metadata():
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {},
    }
    for package in ("mlx", "mlx-lm", "dflash-mlx", "httpx"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            result["packages"][package] = importlib.metadata.version(package)
    for name, command in {
        "commit": ["git", "rev-parse", "HEAD"],
        "hardware": [
            "sysctl",
            "-n",
            "hw.model",
            "hw.memsize",
            "machdep.cpu.brand_string",
        ],
    }.items():
        try:
            result[name] = subprocess.check_output(
                command, stderr=subprocess.DEVNULL, timeout=5, text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            result[name] = None
    return result


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def summarize(rows, seconds):
    good = [r for r in rows if r["status"] == "ok"]
    counts = [r["completion_tokens"] for r in good]
    # Missing usage or failed requests must not produce an optimistic throughput.
    aggregate = (
        sum(counts) / seconds
        if good
        and len(good) == len(rows)
        and all(n is not None for n in counts)
        and seconds > 0
        else None
    )
    return {
        "requests": len(rows),
        "successes": len(good),
        "aggregate_output_tps": aggregate,
        "wall_seconds": seconds,
        "ttft_seconds": distribution(
            [r["ttft_seconds"] for r in good if r["ttft_seconds"] is not None]
        ),
        "latency_seconds": distribution([r["seconds"] for r in good]),
    }


async def stream_request(client, url, payload):
    started = time.perf_counter()
    row = {
        "status": "error",
        "ttft_seconds": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "cached_tokens": None,
        "usage": None,
    }
    output, finish, done = [], None, False
    try:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            data = []

            def consume():
                nonlocal finish, done
                if not data:
                    return
                raw = "\n".join(data)
                data.clear()
                if raw.strip() == "[DONE]":
                    done = True
                    return
                event = json.loads(raw)
                if "error" in event:
                    raise ValueError("Provider returned a streaming error")
                if event.get("usage"):
                    row["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta", {})
                    visible = any(
                        delta.get(k)
                        for k in (
                            "content",
                            "reasoning_content",
                            "reasoning",
                            "tool_calls",
                        )
                    )
                    if visible and row["ttft_seconds"] is None:
                        row["ttft_seconds"] = time.perf_counter() - started
                    if visible:
                        output.append(delta)
                    finish = choice.get("finish_reason") or finish

            async for line in response.aiter_lines():
                if line == "":
                    consume()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
            consume()
        if not done or finish is None:
            raise ValueError("Truncated stream: missing [DONE] or finish_reason")
        usage = row["usage"] or {}
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                row[key] = value
        row["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        row["finish_reason"] = finish
        row["status"] = "ok"
    except Exception as exc:
        # Never serialize headers or credentials into results.
        row["error"] = type(exc).__name__ + ": " + str(exc)
    row["seconds"] = time.perf_counter() - started
    row["output"] = output
    # Compare concatenated text, not provider-dependent SSE chunk boundaries.
    text = {
        k: "".join(str(d.get(k) or "") for d in output)
        for k in ("content", "reasoning_content", "reasoning")
    }
    row["text_sha256"] = digest(json.dumps(text, sort_keys=True).encode())
    return row


async def serve(args):
    import httpx

    config = json.loads(args.config.read_text())
    corpus = [
        json.loads(line)
        for line in args.corpus.read_text().splitlines()
        if line.strip()
    ]
    if not corpus or len({r["id"] for r in corpus}) != len(corpus):
        raise ValueError("Corpus must have unique request IDs and at least one request")
    for r in corpus:
        if not isinstance(r.get("messages"), list) or not r["messages"]:
            raise ValueError("Each corpus entry needs nonempty messages")
    result = {
        "kind": "serving",
        "metadata": metadata(),
        "configuration": config,
        "configuration_verified_by_harness": False,
        "corpus_sha256": digest(args.corpus.read_bytes()),
        "request_parameters": {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "groups": [],
    }
    headers = {}
    if os.environ.get(args.key_env):
        headers["Authorization"] = "Bearer " + os.environ[args.key_env]
    async with httpx.AsyncClient(
        headers=headers,
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=max(args.concurrency)),
    ) as client:
        for concurrency in args.concurrency:
            rows = []
            tasks = [
                (repeat, item) for repeat in range(args.repeats) for item in corpus
            ]
            random.Random(args.seed).shuffle(tasks)
            gate = asyncio.Semaphore(concurrency)

            async def run(repeat, item, gate=gate, rows=rows):
                async with gate:
                    payload = {
                        "model": args.model,
                        "messages": item["messages"],
                        "temperature": 0,
                        "max_tokens": args.max_tokens,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    row = await stream_request(
                        client, args.base_url.rstrip("/") + "/chat/completions", payload
                    )
                    row.update({"id": item["id"], "repeat": repeat})
                    rows.append(row)

            started = time.perf_counter()
            await asyncio.gather(*(run(repeat, item) for repeat, item in tasks))
            summary = summarize(rows, time.perf_counter() - started)
            result["groups"].append(
                {"concurrency": concurrency, "summary": summary, "rows": rows}
            )
            save(args.output, result)
    return (
        1
        if any(r["status"] != "ok" for g in result["groups"] for r in g["rows"])
        else 0
    )


def projection(args):
    result = {
        "kind": "projection_only",
        "metadata": metadata(),
        "rows": [],
        "synthetic": args.capture is None,
        "full_model_speedup_measured": False,
        "thresholds": {
            "max_relative_l2": args.max_relative_l2,
            "min_speedup": args.min_speedup,
        },
    }
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        result.update(
            status="unavailable",
            reason="Requires Apple Silicon, MLX and native ANE kernels",
        )
        save(args.output, result)
        return 2
    import mlx.core as mx

    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.qwen35_ane_available():
        result.update(
            status="unavailable", reason="Native private ANE runtime unavailable"
        )
        save(args.output, result)
        return 2
    mx.random.seed(args.seed)
    capture = mx.load(str(args.capture)) if args.capture else None
    if capture is not None:
        weight, activations = capture["weight"], capture["x"]
        result["capture_sha256"] = digest(args.capture.read_bytes())
        if weight.ndim != 2 or activations.ndim != 3 or activations.shape[0] != 1:
            raise ValueError("Capture requires weight[out,in], x[1,tokens,in]")
        if weight.shape[1] != activations.shape[-1]:
            raise ValueError("Weight and activation dimensions disagree")
    else:
        weight = mx.random.normal((args.output_dim, args.input_dim)) / math.sqrt(
            args.input_dim
        )
    weight = mx.contiguous(weight.astype(mx.float32))
    if not bool(mx.all(mx.isfinite(weight)).item()):
        raise ValueError("Nonfinite weights")
    quant, scales, biases = mx.quantize(
        weight, group_size=args.group_size, bits=args.bits
    )
    # ANE uses the same reconstructed weights as the GPU baseline, then INT8.
    reconstructed = mx.dequantize(
        quant, scales, biases, group_size=args.group_size, bits=args.bits
    )
    mx.eval(quant, scales, biases, reconstructed)
    result.update(
        bits=args.bits, group_size=args.group_size, weight_shape=list(weight.shape)
    )
    for tokens in args.tokens:
        for fraction in args.fractions:
            row = {"tokens": tokens, "requested_fraction": fraction, "status": "error"}
            try:
                mx.random.seed(args.seed + tokens)
                if capture is not None and activations.shape[1] < tokens:
                    raise ValueError(
                        "Capture is too short; no synthetic tiling of real activations"
                    )
                x = (
                    activations[:, :tokens, :]
                    if capture is not None
                    else mx.random.normal((1, tokens, weight.shape[1]))
                ).astype(mx.float16)
                if not bool(mx.all(mx.isfinite(x)).item()):
                    raise ValueError("Nonfinite activations")
                x = mx.contiguous(x)
                n = int(weight.shape[0] * fraction) // 64 * 64
                if n < 64 or n >= weight.shape[0]:
                    raise ValueError(
                        "Split needs >=64 ANE outputs and a nonempty GPU suffix"
                    )
                width = max(32, math.ceil(tokens / 32) * 32)
                w = mx.contiguous(reconstructed[:n].astype(mx.float32))
                suffix = [mx.contiguous(a[n:]) for a in (quant, scales, biases)]
                mx.eval(x, w, *suffix)
                mx.synchronize()
                started = time.perf_counter()
                program = fast.qwen35_ane_compile_linear(w, width, args.ane_instance)
                row["compile_seconds"] = time.perf_counter() - started

                def baseline(x=x):
                    return mx.quantized_matmul(
                        x,
                        quant,
                        scales,
                        biases,
                        transpose=True,
                        group_size=args.group_size,
                        bits=args.bits,
                    )

                def candidate(
                    x=x, width=width, tokens=tokens, suffix=suffix, program=program
                ):
                    padded = (
                        mx.pad(x, ((0, 0), (0, width - tokens), (0, 0)))
                        if width != tokens
                        else x
                    )
                    return fast.qwen35_ane_affine_qmm_t(
                        mx.contiguous(padded),
                        *suffix,
                        program,
                        args.bits,
                        8,
                        args.group_size,
                    )[:, :tokens, :]

                for call in (baseline, candidate):
                    mx.eval(call())
                mx.synchronize()
                samples = {"gpu": [], "hybrid": []}
                errors = []
                for repeat in range(args.repeats):
                    outputs = {}
                    order = [("gpu", baseline), ("hybrid", candidate)]
                    if repeat % 2:
                        order.reverse()
                    for name, call in order:
                        started = time.perf_counter()
                        value = call()
                        mx.eval(value)
                        mx.synchronize()
                        samples[name].append(time.perf_counter() - started)
                        outputs[name] = value.astype(mx.float32)
                    reference, value = outputs["gpu"], outputs["hybrid"]
                    relative_l2 = mx.sqrt(
                        mx.sum((value - reference) ** 2)
                        / mx.maximum(mx.sum(reference**2), 1e-20)
                    )
                    error = float(relative_l2.item())
                    if not math.isfinite(error):
                        raise ValueError("Nonfinite ANE output")
                    errors.append(error)
                ratio = statistics.median(samples["gpu"]) / statistics.median(
                    samples["hybrid"]
                )
                row.update(
                    status="ok",
                    compiled_tokens=width,
                    realized_fraction=n / weight.shape[0],
                    gpu_seconds=distribution(samples["gpu"]),
                    hybrid_seconds=distribution(samples["hybrid"]),
                    speedup=ratio,
                    relative_l2_samples=errors,
                    numerical_gate=max(errors) <= args.max_relative_l2,
                    candidate_for_full_model_test=(
                        ratio >= args.min_speedup
                        and max(errors) <= args.max_relative_l2
                    ),
                )
                del order, call, candidate, baseline, program
            except Exception as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)
            result["rows"].append(row)
            save(args.output, result)
    return 1 if any(r["status"] != "ok" for r in result["rows"]) else 0


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("projection", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--repeats", type=positive, default=5)
        p.add_argument("--seed", type=int, default=0)
        if name == "serve":
            p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
            p.add_argument("--key-env", default="OPENAI_API_KEY")
            p.add_argument("--model", required=True)
            p.add_argument(
                "--config",
                type=Path,
                required=True,
                help="Redacted model/runtime settings; labels are not hardware proof",
            )
            p.add_argument(
                "--corpus", type=Path, default=Path(__file__).with_name("coding.jsonl")
            )
            p.add_argument("--concurrency", nargs="+", type=positive, default=[1, 2, 4])
            p.add_argument("--max-tokens", type=positive, default=512)
            p.add_argument("--timeout", type=positive, default=600)
        else:
            p.add_argument(
                "--capture",
                type=Path,
                help="NPZ with real weight and x; otherwise synthetic",
            )
            p.add_argument("--input-dim", type=positive, default=2048)
            p.add_argument("--output-dim", type=positive, default=4096)
            p.add_argument(
                "--tokens", nargs="+", type=positive, default=[5, 16, 64, 512, 2048]
            )
            p.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5])
            p.add_argument("--bits", type=int, choices=[4, 8], default=4)
            p.add_argument("--group-size", type=int, choices=[64, 128], default=64)
            p.add_argument("--ane-instance", type=int, default=0)
            p.add_argument("--max-relative-l2", type=float, default=0.02)
            p.add_argument("--min-speedup", type=float, default=1.1)
    args = parser.parse_args()
    if args.command == "projection":
        if any(not 0 < f < 1 for f in args.fractions):
            parser.error("fractions must be in (0,1)")
        if (
            not 0 < args.max_relative_l2 < 1
            or not math.isfinite(args.min_speedup)
            or args.min_speedup < 1
        ):
            parser.error("Require 0 < max-relative-l2 < 1 and finite min-speedup >= 1")
    raise SystemExit(
        asyncio.run(serve(args)) if args.command == "serve" else projection(args)
    )


if __name__ == "__main__":
    main()
