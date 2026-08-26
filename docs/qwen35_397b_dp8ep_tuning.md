# Qwen3.5-397B-A17B-FP8 on tpu7x, DP8 + EP8 — serving throughput tuning

Result of an optimization campaign against the InferenceX serving benchmark at
ISL 8192 / OSL 1024 / concurrency 256. Baseline commit `78cd999d`.

## Headline

**1.819x total and output token throughput**, from a scheduler change plus three environment
variables. Arithmetic is unchanged: a greedy (`temperature=0`) probe is bit-identical to the
baseline in every job that ran this configuration, across two images and two nodes.

(The one job that broke bit-identity was the rejected mnbt=4096 variant — see below. Changing
`--max-num-batched-tokens` alters the compiled bucket ladder and *can* perturb arithmetic, so treat
bit-identity as a property to re-verify per configuration rather than an assumption.)

| Metric | Baseline `78cd999d` | With `TPU_DP_PREFILL_*` | Ratio |
| --- | ---: | ---: | ---: |
| Total token throughput (tok/s) | 17,512.68 | **31,859.46** | **1.819x** |
| Output token throughput (tok/s) | 1,946.15 | **3,540.49** | **1.819x** |
| Request throughput (req/s) | 2.11 | 3.84 | 1.82x |
| Benchmark duration (s) | 1212.77 | 666.64 | 0.55x |
| TPOT median (ms) | 132.83 | 64.05 | 2.07x |
| ITL mean (ms) | ~132.8 | 62.39 | 2.13x |
| ITL median (ms) | 34.02 | 31.04 | 1.10x |
| ITL P90 (ms) | 264.87 | 292.18 | 0.91x (worse) |
| TTFT P99 (ms) | 33,720.85 | 50,534.39 | 1.499x |

2560 prompts, `--random-range-ratio 0.8`, `--ignore-eos`, 512 warmups. Clean (non-profiled)
pass only — profiling costs ~22% throughput. gsm8k, 5-shot greedy, full 1319-item test split:
**96.97% strict-match / 96.29% flexible-extract**.

Reproduced across two images (`3afa93bf`, `96725fea`) and two nodes, 21 h apart, to **+0.10%**.
Working noise floor is ~0.5% (largest observed within-job spread).

## Winning configuration

Three environment variables on top of the stock server command:

```
TPU_DP_PREFILL_DEFER=1 TPU_DP_PREFILL_HARD=1 TPU_DP_PREFILL_DRAIN_STEPS=2
```

Full server command as benchmarked:

```
vllm serve <model> --max-model-len=9236 --max-num-batched-tokens=2048 --max-num-seqs=32
  --no-enable-prefix-caching --gpu-memory-utilization=0.92 --block-size=256 --async-scheduling
  --quantization=fp8 --kv-cache-dtype=fp8 --attention-backend CUSTOM
  --tensor-parallel-size=1 --data-parallel-size=8 --enable-expert-parallel
  --prefill-schedule-interval=8 --language-model-only
```

All three variables default off, so the build is behaviourally identical to `78cd999d` unless
they are set. They are in `_TPU_COMPILE_ENV_IGNORED`, so flipping them does not force a recompile.

## Why it works

With EP, `_dp_coordinated_step` all-reduces each rank's local token count with `ReduceOp.MAX`,
because EP combine requires matching buckets across the DP group. So **one rank prefilling drags
all eight onto the 2048-token program** — a rank whose local demand is 32 decode tokens executes
a 292 ms step instead of a 31 ms one.

`TpuDpScheduler` already existed to align prefill admission, but was close to a no-op under
saturation: `prefill_capacity_bound` was pinned `True` (so upstream's `defer_prefills` was always
`False`, and an in-progress prefill *chunk* could never be deferred), and the drain window closed
only when the local queue emptied, which never happens at c256. The three variables respectively
forward the DP-synchronized cadence to upstream, stop a backlog from re-opening admission
mid-window, and bound the window by a step counter.

That counter is **DP-identical by construction** — `throttle_prefills` is false iff
`step_counter % prefill_schedule_interval == 0`, and `step_counter` is DP-identical — so the
window synchronizes across ranks with **no extra collective**.

**The mechanism is frequency, not cost.** It does not make the inflated step cheaper (265 -> 292 ms,
marginally worse, consistent with prefill batching into fuller windows); it makes it rarer.
Solving the two-mode model `f = (mean ITL - median ITL) / (P90 ITL - median ITL)` gives a
slow-step fraction of **42.8% at baseline -> 12.0%**. P90 ITL cannot register this, because P90
only leaves the slow mode once `f` drops below 10%. **Use mean ITL or TPOT median as the mechanism
statistic at this operating point, not P90.**

Device traces confirm the slow mode to 0.26%: the measured bucket-2048 module wall time is
292.937 ms against a client-side P90 ITL of 292.18 ms.

## Sizing `W` (`TPU_DP_PREFILL_DRAIN_STEPS`)

`W` is the number of window-open steps per cadence period, so prefill supply is
`W * max_num_batched_tokens` tokens per `prefill_schedule_interval` steps. Steady-state demand at
c256/DP8: each rank holds `max_num_seqs = 32` slots for ~OSL = 1024 decode steps, so it retires one
request per 32 steps, each needing ISL = 8192 prefill tokens => **256 prefill tokens/step**.

| config | supply | vs demand |
| --- | --- | --- |
| `W=1`, mnbt 2048, interval 8 | 256 tok/step | 1.0x — knife edge, TTFT blows up |
| `W=2`, mnbt 2048, interval 8 | 512 tok/step | **2.0x — the shipped value** |

`W` does **not** control the slow-step fraction. `f` is pinned by a conservation law,
`f = (req/s * ISL / DP / steps_per_s) / mnbt`, which reproduces the measured 12.0% from
independently-measured quantities. `W` controls burstiness only.

## Things that did not work

| Idea | Verdict | Killed by |
| --- | --- | --- |
| `--max-num-batched-tokens` 2048 -> 1024 | **-21.15% throughput** | Mean TTFT +436%. That flag is simultaneously the decode bucket and the prefill chunk size, so an 8k prompt chunks 7-8 ways instead of 4. The decode half of the thesis was confirmed (P90 ITL -44.6%) and it still lost. |
| `TPU_MOE_COLLECTION_CHUNK_SIZE=8192` | Null (-0.36%, sub-noise) | Sized against inflated steps, which the scheduler fix already made rare. Bit-identical output, as predicted. |
| Counting sort for the MoE routing permute | Dead | 0.45-0.76x slower at every shape on real hardware, *and* wrong (parity MISMATCH at 4096 tokens). The `O(n log n) -> O(n)` argument does not survive contact with a TPU: the `[N, experts+1]` prefix sum costs more than the comparison sort. |
| Eliminate per-layer padding-mask all-gathers | Dead | XLA already CSE'd 60 into 1; total cost <=0.18% of a step. |
| `--max-num-batched-tokens` 2048 -> 4096 | **-18.5% throughput** | The mirror image of the 1024 experiment: TTFT improved (P99 50.5 -> 40.0 s) and decode got worse (mean ITL 62.4 -> 80.9 ms). `f` did **not** fall as predicted — it stayed pinned at 1/8 — so the bucket doubled in cost while staying only 46% full of real prefill tokens. Also **broke bit-identity**. |

## Budget model for future work

At **fixed** concurrency the benchmark is a closed queue, so

    E2EL ~= TTFT + OSL * ITL          throughput = concurrency / E2EL

This predicts measured req/s to within ~5% and is the arithmetic to run before proposing any knob
that trades prefill against decode. Decode latency is not privileged: at OSL 1024 a 1 ms/step ITL
win is worth ~1 s, while TTFT regressions run to tens of seconds.

Validated three times, most sharply on the mnbt=4096 run, where `TTFT + OSL * ITL` predicts mean E2EL
to **+0.10%** on both passes.

## mnbt=2048 is an optimum, bracketed by measured losses on both sides

| mnbt | total tok/s | vs 2048 | mean ITL | TTFT P99 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 25,095.57 | **-21.2%** | 43.54 ms | 69,727.80 ms |
| **2048** | **31,859.46** | — | 62.39 ms | 50,534.39 ms |
| 4096 | 25,977.97 | **-18.5%** | 80.85 ms | 39,963.89 ms |

The two failures are mirror images. Shrinking the bucket buys decode and pays for it in TTFT
(chunk count rises); growing it buys TTFT and pays for it in decode. Since `OSL * ITL` is ~90% of
E2EL at OSL 1024, the decode side dominates and both directions lose.

## `f` is pinned by the prefill cadence, not by `mnbt` or `W`

The slow-step fraction `f = (mean ITL - median ITL) / (P90 ITL - median ITL)` is the same in **every**
run with the scheduler fix enabled, across a 4x range of `mnbt` and `W` in {1, 2, 4}:

| run | mnbt | `W` | `f` |
| --- | ---: | ---: | ---: |
| baseline (fix off) | 2048 | — | 42.80% |
| J1 / K1 / J5 | 2048 | 2 | 11.99 / 12.26 / 12.01% |
| C1 | 1024 | 4 | 12.08% |
| J7 | 4096 | 1 | 12.41% |

All land within 4% of **1/8 = `1 / --prefill-schedule-interval`**. The mechanism is quantization: a
step is inflated if *any* of the 8 ranks prefills, the scheduler synchronizes the ranks onto a shared
cadence window, and each rank's per-period prefill demand fits inside **one** chunk — so exactly one
step per period of 8 is inflated, whatever the bucket size.

    f = ceil(prefill tokens per rank per period / mnbt) / prefill_schedule_interval

That numerator is fixed by throughput and ISL, not by `mnbt`. So **doubling `mnbt` does not halve `f`
— it just makes the same one inflated step per period twice as expensive while leaving it half
empty.** At mnbt 2048 the inflated step is ~86% real prefill tokens; at 4096, ~46%. Cost per *real*
prefill token accordingly went the wrong way, 0.165 -> 0.231 ms.

**Untested consequence, and the best remaining candidate:** the ceiling is what wastes the bucket, so
raise `--prefill-schedule-interval` **together with** `mnbt` to refill it. At mnbt 4096 and interval
16, per-period demand roughly doubles to ~91% of the bucket, `ceil()` still returns 1, and
`f` -> 1/16 = 6.25%. With the measured 431 ms cost of the 4096 step that gives mean ITL ~56 ms and
~+15% throughput. `W=1` still suffices for supply. Two caveats before spending a job: the projection
is self-referential (demand depends on the throughput it predicts), and it doubles the wait for an
admission window, which the model charges nothing for.

## A step-cost microbench is necessary but not sufficient

A chained `all_gather` -> `psum_scatter` microbench correctly predicted that step cost keeps scaling
sublinearly to bucket 4096 — it put the step at 549 ms against an actual ~431 ms, erring on the
pessimistic side, and reproduced the measured 1024 -> 2048 step cost to 1.4%. It still led to a
losing experiment, because the question it answered ("what does one inflated step cost?") was not the
question that decides the outcome ("how often is a step inflated, and how full is it?"). Cost per
prefill token is only a valid invariant when the bucket is full.

## Collectives are sublinear — do not size them off trace durations

An earlier round of this campaign read reduce-scatter as **superlinear** (368 us @1024 -> ~960 us
@2048, ~2.6x) from `hlo_stats`, and used that to argue for *shrinking* the bucket. Measured directly
— chained 20 deep so nothing can be overlapped or reordered away — it is not:

| per-rank tokens | AG->RS round trip | vs 2x prev | all-reduce | vs 2x prev |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 363.14 us | — | 70.07 us | — |
| 1024 | 699.68 us | 0.963x | 117.43 us | 0.838x |
| 2048 | 1358.35 us | 0.971x | 207.42 us | 0.883x |
| 4096 | 2672.48 us | **0.984x** | 389.19 us | **0.938x** |

The true 1024 -> 2048 ratio is **1.94x**, not 2.6x. Under DP lockstep every rank blocks until the
last one arrives, so a collective's *trace* duration measures arrival skew, not transfer. The real
per-step collective cost is ~94 ms of the 292 ms bucket-2048 step (32%), for 60 EP round trips plus
61 all-reduces. **Never size a collective optimization off trace durations on a lockstep workload** —
the microbench that settled this cost 6 minutes of pod time.

## Profiling caveat

Phase-triggered capture (`inference_phase_seen`) fires once per phase for the life of the process,
and during warmup a single long request already satisfies `decode_kv_len_threshold`. In practice
`DECODE_ONLY` is consumed at batch ~7 with one request in flight, so **every device-bearing capture
lands in the warmup ramp** and the production decode program (bucket 32 at 32 requests) is never
traced. Steady-state bucket histograms from these traces are not trustworthy. A fix would gate
`_start_profiling` on `num_reqs >= 0.5 * max_num_seqs` in addition to the KV-length threshold, and
ensure a device-bearing rank captures the `balanced` phase.
