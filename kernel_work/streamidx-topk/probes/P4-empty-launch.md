# P4 — what does an empty `pallas_call` cost at the nightly width?

**Verdict: ~0.55 µs, not the ~5.3 µs measured at AgentX — and that kills both design tiers.** A
zero-length-grid stage-A launch at the nightly score width costs **0.55 µs** (`seq_batch_size=1`) or
**0.04 µs** (`seq_batch_size=4`), roughly **10× cheaper** than the 5.40 / 5.22 µs seen at AgentX. The
`lax.cond` alternative costs **+5.2 to +5.9 µs whenever the branch is taken** and ~0 when skipped —
i.e. the fix is an order of magnitude more expensive than the thing it removes.

- **Tier 1 (merge the two decode calls into one):** recovers **0.55 µs/layer = 0.25 %** of the
  221.4 µs nightly decode call. **Not worth the change.**
- **Tier 2 (`lax.cond` the empty launch away):** saves **0.63 µs** on the pure-decode path and adds
  **5.2–5.9 µs** on any step where the branch is taken, plus 21 more `conditional`s per step to a
  bucket already at 4.6 % of the step. **Reject.**

The AgentX number was real but not transferable: it was measured with a **16.8 MB** output
allocation against nightly's **2.62 MB**. The brief's hypothesis was that the fixed cost is
invocation-driven rather than allocation-driven. **The nightly numbers falsify that** — a 6.4×
smaller output gives a ~10× cheaper empty launch, so the cost tracks the allocation, not the
invocation.

- job: **`j-e107f728-6fdc-47d8-98f9`**, shard 3. `JobStatus` at time of reading: `Running` — shard 3
  itself finished cleanly and printed its full JSON; the job stays `Running` only because shard 0's
  `pack` step is still tarring the P1 dumps. Do not read this as a green job.
- shape: nightly — 64 seqs × 1 token, `ctx=8704`, `max_model_len=9216`, page 1024, `kv_packing=4`,
  `lkv_dim=256`, H=32, D=128, k=2048, `bkv_p=2`, `bq_per_block=1`, `num_sublanes_total=80`,
  9 pages/seq, 576 total pages. Score buffer **2,621,440 B = 2.62 MB** f32; `cache_kv` 151.0 MB
- `streamindex_topk.py` md5 `1e221f10cbf3573aa0d2ae78c3463532` (`_stage_a.py` transcription of the
  unimportable `run_scores_kernel` closure; grid bounds stay traced device scalars, so **one
  executable serves grid extents 0, 1 and 16** — no recompilation confound)
- hardware: TPU7x, VMEM 64 MiB, HBM BW 3.7 TB/s

## Method

A single `pallas_call` at these widths is shorter than the host dispatch floor, so it cannot be
timed by launching it once. Each arm instead builds **a chain of M launches inside one jitted
program** and reports the **slope** over M = 1 → 9, which cancels the program's fixed cost:

```
per_launch = (t(M=9) − t(M=1)) / 8
```

Best-of-trials, each trial an iteration-batched chain blocked once (43–368 iterations per trial).
Two controls bound the floor the slope method is subtracting away:

| control | best |
| --- | ---: |
| `init_only` (broadcast a 2.62 MB `-inf` buffer, no pallas_call) | **20.06 µs** |
| `dispatch_floor` (1-arg jit, no work at all) | **16.05 µs** |

Both are ~20 µs, i.e. **36× the quantity being measured** — which is exactly why the intercept is
useless here and the slope is the only defensible read.

## Measured — per-launch slope (`j-e107f728`, shard 3)

| arm | seqs in flight | M=1 best | M=9 best | **per launch** |
| --- | ---: | ---: | ---: | ---: |
| `sbs1_grid0` — **the empty launch** | 0 | 19.05 µs | 23.46 µs | **0.55 µs** |
| `sbs1_grid1` | 1 | 20.73 µs | 60.02 µs | 4.91 µs |
| `sbs1_grid16` | 16 | 66.94 µs | 533.54 µs | 58.33 µs |
| `sbs4_grid0` — **the empty launch** | 0 | 19.47 µs | 19.78 µs | **0.04 µs** |
| `sbs4_grid1` | 4 | 20.06 µs | 96.72 µs | 9.58 µs |
| `sbs4_grid16` — **the full nightly decode** | 64 | 118.36 µs | 1001.39 µs | **110.38 µs** |

### The `lax.cond` arms

| arm | pred | per launch | vs unguarded |
| --- | :-: | ---: | ---: |
| `sbs1_cond_grid0` | **taken** | 5.77 µs | +5.22 µs |
| `sbs1_cond_grid16` | **taken** | 64.22 µs | +5.89 µs |
| `sbs1_cond_grid0` | skipped | **−0.08 µs** | −0.63 µs |
| `sbs1_cond_grid16` | skipped | **0.01 µs** | −58.32 µs |

The two skipped rows measuring −0.08 and +0.01 µs is the harness confirming itself: a branch XLA
elides should cost nothing, and it measures nothing, to within ±0.1 µs. That also sets the
**resolution of this probe at roughly ±0.1 µs**, which is well below the 0.55 µs it needs to
resolve.

## The decision arithmetic — modelled, from the measured numbers above

**Tier 1, merge the two decode calls.** Both empty calls disappear; one launch's overhead is
recovered per layer.
```
saving = 0.55 µs/layer                    (sbs1_grid0 slope, measured)
       = 0.55 / 221.4 = 0.25 % of the nightly decode call
       = 21 × 0.55 = 11.6 µs per decode step across 21 indexer layers
```
Against a change that must merge two distinct grid bands and their scratch. **Not worth it** — and
note the two AgentX launches at 5.40 + 5.22 µs would have projected ~10.6 µs/layer, 19× this.

**Tier 2, `lax.cond` the empty launch away.**
```
saving when the branch is skipped  = 0.55 − (−0.08) = 0.63 µs
cost   when the branch is taken    = +5.22 µs (grid 0) … +5.89 µs (grid 16)
break-even taken-fraction          = 0.63 / (0.63 + 5.22) = 10.8 %
```
So the `cond` only pays if the branch is skipped **more than ~89 %** of the time — and on any step
that is not pure decode it is taken, at ~9× the cost of what it saves, on top of 21 extra
`conditional`s per step into a bucket already at 4.6 %. **Reject.**

## Two findings this probe produced that were not asked for

**1. It cross-validates stage A's cost.** `sbs4_grid16 = 110.38 µs` is the **first direct isolated
measurement** of stage A at the nightly shape. `report.md` §6 derives 109.2 µs for stage A *by
subtraction* from the 221.4 µs call (`j-8b1e7acc`). The two agree to **1.1 %** — from a completely
different method, in a different job, on a contended pod. §6's decomposition of the nightly call
stands.

**2. `decode_req_batch_size=4` is 2.1× more efficient per sequence.**
```
sbs4_grid16: 110.38 µs / 64 seqs = 1.72 µs/seq
sbs1_grid16:  58.33 µs / 16 seqs = 3.65 µs/seq        →  2.12×
```
Two grid steps' worth of extrapolation also exposes an un-overlapped prologue: linear extrapolation
of `sbs4` from grid 1 (9.58 µs) and grid 16 (110.38 µs) puts the grid-0 intercept at
`9.58 − (110.38 − 9.58)/15 = 2.86 µs`, against a measured 0.04 µs. That **~2.9 µs gap is
first-grid-step prologue that does not overlap** with the steady-state pipeline — a real, and
much larger, target than either tier above.

## Caveats

- **Contended.** Four shards on one 4-chip pod. Absolute microseconds are not clean timings; the
  slope-vs-slope and per-seq ratios are the defensible reads. The 1.1 % agreement with §6 is
  reassurance that contention did not badly distort this shard, not a licence to quote 110.38 µs as
  a clean stage-A time.
- Only `seq_batch_size=1` was tested with `lax.cond`. The `sbs4` `cond` cost was not measured;
  there is no reason to expect it to be cheaper, and tier 2 is rejected on the `sbs1` numbers.
- The AgentX 5.40 / 5.22 µs figures are quoted from the step-1 profile, not re-measured here. The
  conclusion is that they do not transfer to nightly, not that they were wrong at AgentX.
- `M = 1, 9` gives a two-point slope, not a fitted line; a non-linearity between M=1 and M=9 would
  not show. The skipped-`cond` controls landing on zero argues against one.

## What this settles

- **Settled: tier 1 (merge the decode calls) is rejected** — 0.55 µs/layer, 0.25 % of the call.
- **Settled: tier 2 (`lax.cond`) is rejected**, and rejected *more strongly* — it costs ~9× what it
  saves whenever taken. This is the option that looked cheaper to write and is strictly worse.
- **Settled: the empty-launch cost is allocation-driven, not invocation-driven.** 6.4× less output
  → ~10× cheaper launch. The brief's stated hypothesis is falsified. Any future estimate of a
  Pallas launch's fixed cost must be taken at the output size that ships.
- **Settled as a side effect: `report.md` §6's 109.2 µs stage-A figure**, now confirmed directly at
  110.38 µs.
- **Newly opened, not settled: ~2.9 µs/layer of un-overlapped first-grid-step prologue** in the
  `sbs4` arm — 5× the size of tier 1 and unexamined. Worth a candidate.
