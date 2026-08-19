# K3 speculative-decode probe — VM agent instructions

You are operating on the TPU VM. This document is self-contained; do not assume
context from any other conversation.

**Goal:** one CDK job that answers every open question about enabling
speculative decoding for Kimi K3's KDA layers. Two serving runs, one job. Return
the collected marker lines as text.

Branch to use: **`shriharim/k3-spec-probe`**

---

## What this branch changes

1. Adds `src/vllm_torchtpu/_k3_probe.py`, a defensive diagnostic module. All of
   its output is single-line and greppable: `[K3PROBE_<TAG>] key=value ...`
2. Emits config markers from the KDA layer, the Kimi MoE layer, and the runner's
   mamba slot allocator.
3. Puts the KDA speculative-decode guard behind an environment variable.
   **Default behaviour is unchanged** — the guard still raises unless
   `K3_PROBE_ALLOW_SPEC=1` is set.

Nothing here is meant to merge. It is throwaway instrumentation.

---

## Run A — baseline, no speculative decoding

Serve K3 on the 32-core v7 slice with your normal working recipe for this model,
with these additions:

Environment:

```
K3_PROBE=1
```

Serve flags — add profiling (these are the flags the plugin actually reads;
`VLLM_TORCH_PROFILER_DIR` is ignored):

```
--profiler-config.profiler torch
--profiler-config.torch_profiler_dir $CDK_OUTPUT_DIR/xprof
```

Benchmark shape — this is the shape the hackathon rubric scores:

```
input length 8192, output length 1024, max concurrency 8
```

Keep the request count small (32 is plenty). We need attribution and a
throughput number, not a long sweep.

## Run B — speculative decoding, guard bypassed

Same recipe, same model, plus:

Environment:

```
K3_PROBE=1
K3_PROBE_ALLOW_SPEC=1
```

Serve flags — add ngram speculation. Ngram needs **no draft model**, which is
why we use it here:

```
--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'
```

Same benchmark shape as Run A.

**Run B is expected to be either wrong or to crash.** That is the point. The KDA
kernels do not yet implement the checkpoint-window protocol that the GDN v3
kernels use, so rolled-back state will be incorrect. We need to know which of
these three happens:

- it starts and serves → we get the throughput ceiling, and correctness is the
  remaining work
- it starts and crashes mid-run → capture the traceback
- it fails at startup → capture the traceback

All three outcomes are useful. Do not try to fix any failure. Capture it and
move on.

---

## What to return

Run both, then collect and paste back **all** of the following as text:

1. Every probe marker from both runs:

```bash
grep -h '\[K3PROBE_' <run-A-log> <run-B-log>
```

2. Any Python traceback from Run B — the full traceback, not a summary:

```bash
grep -A 40 'Traceback (most recent call last)' <run-B-log> | head -120
```

3. The benchmark summary block from each run — the lines reporting output token
   throughput, total token throughput, TTFT and TPOT.

4. If Run A produced a profile, the trace path or `cdk job xprof` output so the
   per-layer op timings can be pulled. The KDA ops are named per layer
   (`pallas::kimi_kda_*`, `pallas::kimi_sconv_*`, `pallas::kimi_chunk_kda_*`),
   so summing them gives the exact KDA share of step time.

---

## Constraints

- Do not inline a GitHub token into a clone URL. Use the existing secret or the
  mounted checkout.
- Do not modify kernel code. This job is measurement only.
- If the 32-core slice is unavailable, run both configurations at whatever
  smaller topology is free and say which topology you used — relative numbers
  are still useful.
- If Run A and Run B cannot share a job, run them as two jobs and label the logs
  clearly.
