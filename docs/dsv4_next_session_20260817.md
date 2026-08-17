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

**Serving throughput is KV-capped.** 53.8 tok/s at concurrency 64, 54.7 at 256
— saturated. KV usage 99.9%, batch capped at ~56 by 115,565 tokens. Per
sequence that is ~1 s/token, against tpu-inference's 1.39 s/token at batch 128,
so it is not something this port introduced.

## What to pick up

1. **Submit the stack.** ~40 commits absent from `origin/main`. This is what
   gxd3 asked for on PR #462, it has been sitting since 2026-08-14, and nothing
   gates it. The agent never opens PRs — draft and hand over.
2. **Finish the throughput question.** Under test at handoff:
   `deepseek-v4-flash-tp8-ep-b1024`, identical to the sweep except
   `max-num-batched-tokens` 8192 to 1024, which should give 4.2x the KV room. If
   throughput scales, it is a configuration win; if not, the bottleneck is
   per-step compute and the next step is the profile config
   (`deepseek-v4-flash-tp8-ep-profile`).
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
