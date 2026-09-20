#!/usr/bin/env python3
"""Compare serving runs; exact text agreement is a diagnostic, not a coding eval."""

import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    for field in ("corpus_sha256", "request_parameters"):
        if baseline.get(field) is None or baseline[field] != candidate.get(field):
            raise ValueError(f"Cannot compare different or missing {field}")
    left = {g["concurrency"]: g for g in baseline["groups"]}
    right = {g["concurrency"]: g for g in candidate["groups"]}
    if left.keys() != right.keys():
        raise ValueError("Concurrency grids differ")
    report = []
    for concurrency, a in left.items():
        b = right[concurrency]
        ar = {(r["id"], r["repeat"]): r for r in a["rows"]}
        br = {(r["id"], r["repeat"]): r for r in b["rows"]}
        if ar.keys() != br.keys():
            raise ValueError("Request sets differ")
        pairs = [
            (ar[key], br[key])
            for key in ar
            if ar[key]["status"] == br[key]["status"] == "ok"
        ]
        at, bt = (g["summary"]["aggregate_output_tps"] for g in (a, b))
        report.append(
            {
                "concurrency": concurrency,
                "aggregate_output_speedup": bt / at if at and bt else None,
                "baseline_ttft_p95": a["summary"]["ttft_seconds"]["p95"],
                "candidate_ttft_p95": b["summary"]["ttft_seconds"]["p95"],
                "successful_pairs": len(pairs),
                "requested_pairs": len(ar),
                "exact_text_matches": sum(
                    x["text_sha256"] == y["text_sha256"] for x, y in pairs
                ),
                "cache_counts_match": bool(pairs)
                and all(
                    x["cached_tokens"] is not None
                    and x["cached_tokens"] == y["cached_tokens"]
                    for x, y in pairs
                ),
                "coding_correctness_evaluated": False,
            }
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                json.loads(args.baseline.read_text()),
                json.loads(args.candidate.read_text()),
            ),
            indent=2,
        )
    )
