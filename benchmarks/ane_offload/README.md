# ANE draft and prefill feasibility benchmarks

These are runnable experiments, **not an ANE DFlash implementation**. No ANE
performance has been measured for this fork. Linux validation covers the
streaming collector and accounting only. Hardware results must come from a Mac.

## Findings and proposed path

The pinned oMLX DFlash adapter accepts Qwen, Gemma4, Laguna and Muse Glimmer;
GLM-5.3-Flash is rejected. A target-compatible drafter and target hidden-state /
KV rollback adapter are prerequisites even for GPU-only DFlash. The public
DFlash collection lists GLM-5.1, which does not establish GLM-5.3-Flash
compatibility. DFlash2's published MLX path includes Qwen3.8-27B. A small generic
coding LLM cannot substitute for a target-trained block-diffusion drafter.

Upstream ANE prefill is model-specific (Qwen3.5/3.6/3.8 and K2 Horizon). Its
Qwen path splits eligible dense projections between ANE and GPU, using private
Apple APIs and INT8-requantized weights. Decode and target verification fall
back to GPU. Our gateway currently disallows DFlash because its engine lacks
the gateway's exact token/context control hooks. Benchmark it directly through
upstream serving with the gateway disabled, on localhost.

**Experiment A: draft projections.** Try small token blocks (5, 16, 64) using
actual drafter projection weights and captured activations. ANE dispatch,
packing, synchronization and padding could cost more than the small GPU matmul.
A full drafter port would also need attention, target-hidden-state conditioning,
position/mask handling, dynamic context/cache behavior and acceptance tests.
Offloading a projection does not demonstrate those features. Draft and target
verification are sequentially dependent; do not assume parallel overlap.

**Experiment B: prompt projections.** Try 512/2048-token chunks first. Larger
matmuls can better amortize ANE overhead. For GLM, begin with eligible dense or
shared-expert projections; routed MoE experts require separate dynamic-routing
and resident-weight design. Do not compile the entire expert bank speculatively.
Any approximate target prefill changes the state used for generation and needs
long-context quality testing, not just matching one next token. For a workload
with nearly all prompt tokens cached, the benefit applies only to the uncached
suffix. Cold-prefill speedups must not be advertised as warm decode speedups.

## 1. Projection break-even on the actual Mac

Use the fork's supported Python environment and native-kernel build prerequisites
(full Xcode/Metal toolchain). From the repository root:

```sh
OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .
python benchmarks/ane_offload/bench.py projection \
  --tokens 5 16 64 512 2048 --fractions 0.25 0.5 --repeats 20 \
  --output /tmp/ane-synthetic.json
```

Default dimensions are **synthetic probes**, not GLM or DFlash model dimensions.
They screen for dispatch/shape overhead only. For useful results capture a real
projection's dequantized weight `[out,in]` and its actual input activations
`[1,tokens,in]` during a coding workload, then save with MLX:

```python
mx.savez("/tmp/projection.npz", weight=weight.astype(mx.float32), x=activations)
```

```sh
python benchmarks/ane_offload/bench.py projection \
  --capture /tmp/projection.npz --tokens 5 16 64 --repeats 20 \
  --output /tmp/ane-draft-projection.json
```

The harness quantizes the captured floating-point weight to a controlled q4/q8
GPU reference; this is a re-quantized projection comparison, not a bit-exact
replay of an arbitrary mixed-quant checkpoint. It reconstructs those same
weights for ANE INT8 compilation. The ANE prefix is output-channel aligned;
the remaining outputs run on GPU. Small token counts pad to 32, and that
padding cost is included. Use `--bits 8` to repeat with q8. Instance 0 is the
runtime default; no dual-die topology is assumed. `--ane-instance` is explicit.

Results include compilation time, paired/interleaved GPU/hybrid timings,
median/p95, every sample, realized split, and relative L2 error on every repeat.
Lazy MLX execution is evaluated and synchronized. Native packing, transfer and
merge work is inside the measured call. Warmup is outside timings. Compilation
errors are recorded per shape; Linux returns `unavailable` with exit code 2,
never simulated throughput. Compilation cache hits may affect compile time.

The default 2% relative-L2 / 1.1x speedup gate is an **exploration threshold**,
not proof of model quality. Repeat with real layers, activations and seeds.
Resident memory, power and sustained thermals need separate Mac measurements.
A single projection fitting does not show that all model programs can coexist.

## 2. End-to-end serving comparison

Run one configuration at a time on the same Mac. Keep the physical model,
weights, context, sampling, KV precision and output limit fixed. Disable cloud
fallback; confirm local routing in server logs. Start with a supported Qwen
model to validate the experiment. Do not relabel a GPU run as ANE: the harness
records configuration as **unverified operator metadata**. Capture native ANE
operation counters/server instrumentation separately to prove actual dispatch.

Save a redacted configuration as `/tmp/run-config.json`, for example:

```json
{
  "variant": "gpu-baseline",
  "target_revision": "RECORD_EXACT_REVISION",
  "quant": "RECORD_QUANT",
  "draft_revision": null,
  "kv_format": "RECORD_FORMAT",
  "prefix_cache": "disabled",
  "ane": false
}
```

Never put credentials in this file. Results contain prompt-derived generated
text; keep real private coding workload results out of the public repository.

```sh
pip install httpx
python benchmarks/ane_offload/bench.py serve \
  --model YOUR_LOCAL_MODEL --config /tmp/run-config.json \
  --concurrency 1 2 4 8 --repeats 10 --max-tokens 1024 \
  --output /tmp/gpu-baseline.json
# Reconfigure/restart the server, then repeat with an updated config/output.
python benchmarks/ane_offload/compare.py /tmp/gpu-baseline.json /tmp/candidate.json
```

The included three coding requests are smoke workloads. Supply `--corpus` with
representative JSONL `{ "id": "unique", "messages": [...] }` coding/agent tasks
for a decision. Include actual repo contexts at several token lengths and
multi-turn tool transcripts; don't inflate context by repeating synthetic text.
Results fingerprint the corpus and record server-reported prompt/output/cache
tokens, first content/reasoning latency, total latency, output text, stop reason
and complete usage. Missing usage remains unknown. SSE chunks are **not tokens**;
no per-stream decode TPS is fabricated from network chunks. Aggregate output
TPS uses all reported completion tokens divided by cohort wall time, including
prefill and queueing. It is serving throughput, not pure decode TPS. Server
phase/decode/speculation metrics, when present, remain in raw usage. Failed or
truncated streams invalidate cohort throughput and exit nonzero.

For a cold test, disable all prefix-cache tiers in the server. For a warm test,
prime the exact prompts first, then measure and verify `cached_tokens` against
`prompt_tokens`. This runner does not silently flush caches or prime prompts.
If cache counters are absent, the cache condition is unverified. Repeats and
higher concurrency otherwise warm caches; do not call that a cold benchmark.
Run A/B/B/A configurations with the same thermal/power conditions. Do not run
both full models simultaneously just to alternate requests on a memory-limited
Mac. Check token lengths/termination and independently score generated code;
text disagreement is diagnostic, and text agreement alone is not correctness.

## Decision gates

1. Establish an exact target/draft pairing and a functioning GPU DFlash baseline.
2. Profile draft, verify, replay, prefill and accepted/drafted token counts.
   Amdahl ceiling for speeding only draft is `1 / (1-f + f/s)`, where `f` is its
   measured baseline time fraction and `s` its measured speedup. Acceptance and
   all other times must remain unchanged for that estimate to hold.
3. Pass real-projection numerical and latency tests before writing an ANE model
   adapter. Compare aggregate throughput and high-priority stream floors too;
   DFlash's engine is not the continuous-batched gateway path.
4. Validate draft acceptance, deterministic outputs where applicable, coding
   task correctness and long-context state before enabling target prefill.
5. Require sustained end-to-end benefit, acceptable memory/energy, and no SLO
   regression. Neither a fast matmul nor ANE utilization alone meets that gate.

Existing full-Qwen prefill benchmark (synthetic input, separate from this harness):

```sh
python benchmarks/qwen35_ane_prefill_bench.py /path/to/supported-qwen \
  --force-lm --tokens 2048 --modes gpu single --repeats 10
```

## Sources and validation

- [oMLX DFlash adapter](https://github.com/jundot/omlx/blob/main/omlx/engine/dflash.py)
- [ANE prefill implementation and limitations](https://github.com/jundot/omlx/blob/main/docs/experimental/qwen35_ane_prefill.md)
- [DFlash / DFlash2 published model and backend support](https://github.com/z-lab/dflash)

Linux collector tests (no MLX dependencies):

```sh
python -m unittest discover -s benchmarks/ane_offload/tests -v
```

No full draft model, GLM prefill adapter, hardware inference benchmark, or
quality score is claimed implemented by these scripts.
