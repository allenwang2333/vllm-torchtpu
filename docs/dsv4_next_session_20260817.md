# Start here — DeepSeek-V4, next session

> **Written 2026-08-17.** Supersedes `dsv4_next_session_20260816.md`. Read
> `dsv4_tracker_20260816.md` alongside it — that file is the evidence, this one
> is only the pickup.

## What changed on 2026-08-17

Eight code commits, every one gated with generations byte-identical to
`scratchpad/gate_09f6c694c.sub`:

| commit | what |
|---|---|
| `36abf661` | **KV over-allocation fixed.** SWA overlay was 21 of 43; now 42. Allocation 1.3223x of budget to 0.9839x. Utilization 0.55 to 0.75 |
| `421698f0` | Model package split like `kimi_k3`; `DeepseekV4Head` inlined |
| `8a82c3e9` | `_attn_op_cache` moved to its only consumer |
| `adda8b23` | KV probe instrumentation removed |
| `4c89ac6f` | Inert code removed: dead class names, dead env vars, inert backend flag |
| `d5632171` | **Weight cache consulted before dequantizing.** Startup mxfp4 phase 7 min to 20 s |
| `e7d501b8`, `9eef474f` | Benchmark config made runnable; decode-heavy case added |

Plus doc corrections, all in the tracker's Refuted table.

## Three things that are settled

**Why utilization caps at 0.75.** The compile OOM is the KV cache itself:
the failing module is 106.46 GiB and 63% of it is KV arrays keyed on
`num_blocks`. Rematerialization cannot help — KV is state, not a recomputable
intermediate. The only levers shrink the cache.

**The MMLU constant-`C` artifact.** `/v1/chat/completions` closes the partial
assistant turn the task ends with. Through `/v1/completions` the answers spread
across A/B/C/D and anatomy goes 0.05 to 0.90. Proven by a one-server A/B.

**Serving throughput is ~54 tok/s and no config changes it.** 53.8 at
concurrency 64, 54.7 at 256, 54.2 and 54.6 with 4.2x the KV cache. Per sequence
that is ~1 s/token, against tpu-inference's 1.39 s/token at batch 128, so it is
not something this port introduced. Throughput equals batch exactly, so the
decode step is weight-bound and batch-independent.

## Open questions for the next session

Ordered by value. The first two are new findings from today, not carry-over.

**1. Why does DSv4 take ~1 second per decode step?** Throughput is pinned at
~54 tok/s and *nothing* in the serving config moves it: concurrency 64 vs 256,
batched tokens 1024 vs 8192, KV cache 115K vs 485K tokens all give the same
answer. Throughput equals batch exactly, so per-sequence rate is 1.00 tok/s and
step time is batch-independent — a weight-bound step. A profile is the only way
in from here. `deepseek-v4-flash-tp8-ep-profile` is committed and fixed; it was
launched at 16:00 and stopped at wind-down, so it has never produced a trace.
Budget ~35 min. Note tpu-inference measures 1.39 s/token at batch 128, so this
may be the model rather than the port.

**2. The unexplained 3.35x swing in per-request KV charge.** Changing only
`max-num-batched-tokens` 8192 -> 1024 raises reported cache 4.2x but per-request
charge 3.35x, so concurrency only moves 64 -> 74. Same block size (1024), same
workload, blocks fully released between combos. Two candidate causes refuted
(unreleased blocks, differing block size); the third — multi-group block
accounting — is untested. The `[KV_PROBE]` pre-pass removed in `adda8b23`
logged exactly this and would settle it in one startup. A 1.32x over-allocation
was found in this same subsystem on 2026-08-16, so this deserves a look.

**3. Bound or relocate `/dev/shm/torch_tpu_cache`.** It grew 192 GB -> 348 GB in
one day, against a 473 GB tmpfs, and being RAM it evicts the page cache holding
the 149 GB of weights — safetensors loading degraded from 2.27 shards/second to
17.8 s/shard over the same period. The default path is hardcoded in
`torch_tpu/common/libpywrap_torch_tpu_common.so`; the same binary contains
`TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT`, which is *probably* the
override but has not been verified to be read. Pointing it at the NVMe is a
one-line change if it works.

**4. Cache reuse.** 1,517 distinct config-hash groups have accumulated. The key
(`compute_tpu_compilation_hash`) includes `gpu_memory_utilization`,
`num_gpu_blocks`, `max_model_len`, `max_num_batched_tokens`, `max_num_seqs`,
every non-ignored `vllm_torchtpu` env var, and the bytes of every kernel-reload
source file. That precision is deliberate — `ca5bf0585` added it so
configurations stop serving each other's artifacts — so reuse means varying
less, not hashing less. Highest-leverage idea, untested: pin
`--num-gpu-blocks-override` so utilization stops changing the key, since
`num_gpu_blocks` and `gpu_memory_utilization` are redundant in it.

## What to pick up

1. **Submit the stack.** ~40 commits absent from `origin/main`. This is what
   gxd3 asked for on PR #462, it has been sitting since 2026-08-14, and nothing
   gates it. The agent never opens PRs — draft and hand over.
2. **The throughput question is answered and closed.** The
   `max-num-batched-tokens` test ran: 54.6 tok/s at c=256 against the sweep's
   54.7. No configuration lever moves throughput. See open question 1 — the
   remaining work is the profile, not another config sweep.
3. **Size evals realistically.** Full MMLU projects **26.8 hours** at the
   measured rate; MMLU-Pro is worse per sample because outputs are longer. Both
   need subsets. MMLU-Pro at concurrency 32 measured 10.7 s/sample.

## Traps that cost time on 2026-08-17

- `pgrep -f "<pattern>"` matches **this shell** when the pattern appears in the
  command line. It killed my own process twice mid-edit. Match on a path, and
  exclude `$$`.
- `ls -t <dir>/*.json | tail -1` returns the **oldest** file. A watchdog used it
  to read the first benchmark result, got `config.json`, read no throughput, and
  never fired — so a sweep ran its full 2.5 h timeout instead of bailing.
- `run_benchmarks.sh` probes for the CLI with bare `python3`, which is
  `/usr/bin/python3` and has no vLLM. Put `${VENV}/bin` on `PATH`.
- A stall detector that watches log *activity* will not catch a hung eval: the
  server keeps printing throughput lines while requests hang. Watch the request
  counter instead. And a run that progresses but cannot finish in budget is
  invisible to a stall detector entirely — check projected ETA.
- Fixing a value in a config file does not fix it where it is hardcoded
  elsewhere. Grep for the old value before relaunching.
