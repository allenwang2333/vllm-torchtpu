# Three gaps in the composition sweep, closed

`RESULTS.md` measured 5 arms x 17 configs in job `j-2c00d88c-e7de-415f-968b`. A critical
review found three things that sweep could not answer. This file answers them from **one**
job, `j-d17207d1-05fb-4e10-93b3`, a 4-shard fan-out on one `tpu7x` 2x2x1 pod
(`JobStatus=Succeeded`, `tpu-fanout: 4/4 shards succeeded`).

Method is `RESULTS.md`'s, unchanged and deliberately so: **`hlo_stats` device self time**,
TensorCore rows only, one `*.xplane.pb` per arm, read off-pod by
`read_gaps_hlo_stats.py`, which imports `read_merged_hlo_stats.one()` rather than
reimplementing the bucketing. Every one of the **86 traces passed** the
`|partition_residual| < 1e-6` and `rows_kept == rows_total` gate — `0 of 86` failed.
The SparseCore plane is reported but never added to a TensorCore total.

**No dispatch floor is subtracted from anything here.** Per-call wall clock is
`max(device, host)`, not `device + host`. The wall-clock numbers `bench_gaps_trace.py`
prints are contended across four shards and are not results; they exist so a hung arm is
visible in the log.

## The job

| | |
| --- | --- |
| job | `j-d17207d1-05fb-4e10-93b3` |
| status | `Succeeded`, `tpu-fanout: 4/4 shards succeeded` (rc=0 on all four) |
| image | `…/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` |
| shards | `e1` (chip 0), `e2a` (chip 1), `e2b` (chip 2), `e3` (chip 3) |
| traces | 28 + 14 + 14 + 30 = **86**, all with an xplane, all passing the gate |

**The image's kernels are stale**, so every shard reads back its module md5s before it is
allowed to measure and `bench_gaps_trace.py` returns 2 without measuring on a mismatch.
Read back on-pod, on all four shards:

| file | on-pod md5 | expected |
| --- | --- | --- |
| `src/…/deepseek_v4/streamindex_topk.py` | `1e221f10cbf3573aa0d2ae78c3463532` | matches |
| `src/…/deepseek_v4/sparsecore_topk.py` | `8bc9e25b8701a53a97704564b9fbc658` | matches |
| `variants/merged/streamindex_topk_merged.py` | `5bc0578427412d56209ef8b5d764fcae` | matches |
| `variants/merged/streamindex_topk_cond.py` | `951981dbae9e1e8d58709f574eb4abea` | matches |
| `instruments/bench_streamidx_topk.py` | `e8e492d13e59c4a1675cc7a98c5a0813` | matches |

`sparsecore_topk.py` is **absent from the image entirely**; it is only present because of
`--map-dir <checkout>/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src`.

Every arm's top-k indices were compared against `inc` on the same inputs before any
timing: **index-identical on all 86, `rows_mismatched = 0` everywhere.**

## The arms

`gaps_common.GAPS_ARM_SPECS`. The first five are `merged_common.ARMS`, unchanged.

| arm | module | `num_kv_pages_per_block` | early exit | score layout |
| --- | --- | --- | --- | --- |
| `inc` | `streamindex_topk.py` | `(2,2,2)` | on | head-minor |
| `bkvp` | `streamindex_topk.py` | recommended | on | head-minor |
| `d1` | `..._merged.py` | `(2,2,2)` | **deleted** | head-minor |
| `bkvp_d1` | `..._merged.py` | recommended | **deleted** | head-minor |
| `merged` | `..._merged.py` | recommended | **deleted** | head-major on MIXED |
| `c` | `..._cond.py` | `(2,2,2)` | on | head-major on MIXED |
| `cctl` | `..._cond.py` | `(2,2,2)` | on | head-minor |
| `inc_nee` | `streamindex_topk.py` | `(2,2,2)` | **off** (flag) | head-minor |

`streamindex_topk_cond.py` is `streamindex_topk_merged.py` with the `lax.cond` and
`_fast_path` put back and an `enable_early_exit` static arg. It exists because the merged
module *deleted* the cond, so a "C only" arm cannot be built from it — anything built from
`..._merged.py` has D1 baked in. `cctl` is the control that makes `c` readable: same
module, same cond, head-**minor**, so it must reproduce `inc`. It does, to
**+0.041% at production decode** and **-0.026% / -0.012% / -0.001% / +0.457%** at the four
prefill configs. Nothing in the copy other than the layout flag moves the number.

---

## E1 — a within-job error bar at the production config

`dec_b64_ctx8704_mml9216` (batch 64, ctx 8704, `max_model_len` 9216, page 1024, k=2048),
all 7 arms, **n=4 independent repeats inside the one job**. Reps re-enter the measurement
loop: each is a fresh `jax.profiler.trace` block over a fresh 20 iterations, not more
iterations inside one block. Reps are interleaved (rep 0 of every arm, then rep 1 of every
arm, …) so drift over the life of the job spreads across arms instead of loading onto
whichever ran last. Compilation is once per arm and outside every trace block.

| arm | n | mean full us | min | max | spread % of mean | sd % | Δ vs `inc` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inc` | 4 | 170.483 | 170.325 | 170.628 | 0.178% | 0.083% | — |
| `bkvp` | 4 | 143.281 | 143.195 | 143.346 | 0.105% | 0.046% | −15.956% |
| `d1` | 4 | 162.605 | 162.494 | 162.807 | 0.192% | 0.088% | −4.621% |
| `bkvp_d1` | 4 | 135.085 | 135.074 | 135.095 | 0.015% | 0.006% | −20.764% |
| `merged` | 4 | 135.064 | 135.038 | 135.108 | 0.052% | 0.023% | −20.776% |
| `c` | 4 | 170.541 | 170.335 | 170.640 | 0.179% | 0.083% | +0.034% |
| `cctl` | 4 | 170.553 | 170.397 | 170.619 | 0.130% | 0.062% | +0.041% |

**Largest within-job spread across arms: 0.192%** (full min-to-max range, `d1`). Largest
sd: 0.088%.

Recomputed from these means:

| quantity | µs | % of `inc` |
| --- | ---: | ---: |
| two-way residual `Δ(bkvp_d1) − Δ(bkvp) − Δ(d1)` | −0.319 | **−0.187%** |
| three-way residual `Δ(merged) − [Δ(bkvp) + Δ(d1) + Δ(c)]` | −0.396 | **−0.232%** |

### Is 0.72% inside or outside this error bar?

**Outside, by roughly 4x.** A residual is a signed sum of four means; if the four are
independent with sd ≈ 0.09% of `inc`, its sd is ≈ 0.18%, so 0.72% is about 4 sd — and it
is 3.75x the largest single-arm min-to-max range measured here. The 0.72% non-additivity
`RESULTS.md` reports at `dec_b8_ctx4096` is a real effect of the composition, not
measurement scatter. It is also small: at the *production* config the same residual is
−0.187% (two-way) and −0.232% (three-way), i.e. only just outside the bar.

Two cautions on the bar itself. It is **one config on one chip in one job**, so it bounds
run-to-run scatter and not the compile-to-compile or job-to-job kind — and those are
larger: this job's `merged` mean is 135.064 µs against `RESULTS.md`'s 135.927 µs, a
**0.63% between-job shift**, twelve times the 0.052% within-job spread of the same arm.
Any residual near 0.7% therefore survives within-job noise but would not survive being
assembled from arms measured in *different* jobs. And the bar is small partly because
these are large ops; do not carry 0.19% to a 10 µs config.

---

## E2 — Option C measured alone, and a genuine three-way residual

The sweep defined `Δ C ≡ Δ(merged) − Δ(bkvp_d1)`. Substituting that into
`Δ(merged) − [Δ(bkvp) + Δ(d1) + Δ C]` cancels `merged` identically and leaves
`Δ(bkvp_d1) − Δ(bkvp) − Δ(d1)` — a two-way test of `bkv_p` against D1 with Option C's
composition untested. The `c` arm fixes that: incumbent + Option C only, cond alive,
`(2,2,2)` kept, so all four terms below are measured independently against `inc`.

Full-op device self time, four prefill configs, single rep each:

| config | `inc` µs | Δ(bkvp) | Δ(d1) | **Δ(c)** | Δ(merged) | sum of parts | **three-way residual** | as % of `inc` | old two-way residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pre_q1024_ctx4096` | 381.320 | −0.107 | −9.236 | **−19.404** | −28.314 | −28.747 | **+0.433** | **+0.113%** | +0.189 |
| `pre_q1024_ctx16384` | 1195.071 | +0.158 | −9.198 | **−80.645** | −89.622 | −89.685 | **+0.063** | **+0.005%** | −0.059 |
| `pre_q4096_ctx4096` | 1109.617 | +2.698 | −12.816 | **−78.823** | −89.226 | −88.941 | **−0.285** | **−0.026%** | −2.690 |
| `pre_q4096_ctx16384` | 4437.800 | +2.123 | −12.605 | **−323.014** | −335.558 | −333.496 | **−2.062** | **−0.046%** | −1.737 |

Option C alone, as a percentage:

| config | Δ(c) full-op | Δ(c) stage A | Δ(merged) full-op | Δ(cctl) full-op (control) |
| --- | ---: | ---: | ---: | ---: |
| `pre_q1024_ctx4096` | **−5.089%** | −18.322% | −7.425% | −0.026% |
| `pre_q1024_ctx16384` | **−6.748%** | −20.457% | −7.499% | −0.012% |
| `pre_q4096_ctx4096` | **−7.104%** | −19.389% | −8.041% | +0.457% |
| `pre_q4096_ctx16384` | **−7.279%** | −20.802% | −7.561% | −0.001% |

The numbers, not a verdict word: the genuine three-way residual is **+0.113%, +0.005%,
−0.026%, −0.046%** of `inc` at the four prefill configs — all four inside the 0.192%
within-job bar of E1, three of them by an order of magnitude. Option C is the dominant
term at prefill (−5.1% to −7.3% of the whole op, −18% to −21% of stage A) and it carries
essentially none of `bkv_p`'s or D1's effect with it. Note that the `pre_q4096_ctx4096`
control `cctl` is +0.457%, larger than the residual at that config; treat that row's
residual as bounded by the control rather than resolved by it.

At **production decode**, Option C is gated off and measures it: Δ(c) = **+0.034%**
against a `cctl` control of +0.041%. The gate does what it claims.

---

## E3 — runtime `ctx` decoupled from `max_model_len`

`max_model_len` fixed at 9216 for all five points, runtime `ctx` swept, batch 64.
`recommended_num_kv_pages_per_block(9216, 1024)` returns **`bkv_p = 9`** (tuple
`(9,2,2)`; the decode slot only) at every point — that is the whole issue: it is
compile-time static, while `num_bkv = max(1, cdiv(kv_len_max, bkv_p * page_size))` is
runtime, so the block is the quantum of overfetch.

Scanned positions = `num_bkv × bkv_p × page_size`, computed per arm:

| ctx | `inc` bkv_p | `inc` num_bkv | `inc` scanned | rec. bkv_p | rec. num_bkv | rec. scanned | **scanned merged/inc** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2 | 1 | 2048 | 9 | 1 | 9216 | **4.500** |
| 2048 | 2 | 1 | 2048 | 9 | 1 | 9216 | **4.500** |
| 4096 | 2 | 2 | 4096 | 9 | 1 | 9216 | **2.250** |
| 6144 | 2 | 3 | 6144 | 9 | 1 | 9216 | **1.500** |
| 8704 | 2 | 5 | 10240 | 9 | 1 | 9216 | **0.900** |

Full-op device self time, per arm, against `inc`:

| ctx | arm | scanned | scanned/inc | full µs | Δ full % | stage A µs | stage B µs | residual µs | Δ full % ÷ scan ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | `inc` | 2048 | 1.000 | 7.712 | +0.000% | 0.000 | 0.000 | 7.712 | +0.000 |
| 1024 | `inc_nee` | 2048 | 1.000 | 57.417 | +644.472% | 27.688 | 10.558 | 19.172 | +644.472 |
| 1024 | `bkvp` | 9216 | 4.500 | 7.951 | +3.099% | 0.000 | 0.000 | 7.951 | +0.689 |
| 1024 | `d1` | 2048 | 1.000 | 57.395 | +644.187% | 27.734 | 10.472 | 19.189 | +644.187 |
| 1024 | `bkvp_d1` | 9216 | 4.500 | 113.814 | +1375.714% | 84.187 | 10.477 | 19.150 | +305.714 |
| 1024 | `merged` | 9216 | 4.500 | 113.878 | +1376.549% | 84.294 | 10.477 | 19.106 | +305.900 |
| 2048 | `inc` | 2048 | 1.000 | 7.751 | +0.000% | 0.000 | 0.000 | 7.751 | +0.000 |
| 2048 | `inc_nee` | 2048 | 1.000 | 58.941 | +660.417% | 27.873 | 11.969 | 19.100 | +660.417 |
| 2048 | `bkvp` | 9216 | 4.500 | 8.002 | +3.239% | 0.000 | 0.000 | 8.002 | +0.720 |
| 2048 | `d1` | 2048 | 1.000 | 59.071 | +662.082% | 27.901 | 12.070 | 19.100 | +662.082 |
| 2048 | `bkvp_d1` | 9216 | 4.500 | 115.187 | +1386.053% | 84.069 | 11.989 | 19.129 | +308.012 |
| 2048 | `merged` | 9216 | 4.500 | 115.146 | +1385.524% | 84.091 | 11.960 | 19.095 | +307.894 |
| 4096 | `inc` | 4096 | 1.000 | 99.039 | +0.000% | 49.969 | 22.054 | 27.017 | +0.000 |
| 4096 | `inc_nee` | 4096 | 1.000 | 91.141 | −7.975% | 49.938 | 22.052 | 19.151 | −7.975 |
| 4096 | `bkvp` | 9216 | 2.250 | 133.352 | +34.646% | 84.122 | 22.080 | 27.150 | +15.398 |
| 4096 | `d1` | 4096 | 1.000 | 91.335 | −7.779% | 50.179 | 22.066 | 19.091 | −7.779 |
| 4096 | `bkvp_d1` | 9216 | 2.250 | 125.129 | +26.343% | 84.033 | 22.036 | 19.060 | +11.708 |
| 4096 | `merged` | 9216 | 2.250 | 125.226 | +26.441% | 84.094 | 22.044 | 19.088 | +11.752 |
| 6144 | `inc` | 6144 | 1.000 | 122.731 | +0.000% | 69.845 | 25.832 | 27.054 | +0.000 |
| 6144 | `inc_nee` | 6144 | 1.000 | 114.545 | −6.669% | 69.619 | 25.815 | 19.111 | −6.669 |
| 6144 | `bkvp` | 9216 | 1.500 | 137.095 | +11.704% | 84.087 | 25.773 | 27.235 | +7.803 |
| 6144 | `d1` | 6144 | 1.000 | 114.670 | −6.567% | 69.764 | 25.768 | 19.139 | −6.567 |
| 6144 | `bkvp_d1` | 9216 | 1.500 | 128.876 | +5.007% | 84.041 | 25.713 | 19.123 | +3.338 |
| 6144 | `merged` | 9216 | 1.500 | 128.837 | +4.975% | 84.045 | 25.712 | 19.081 | +3.317 |
| 8704 | `inc` | 10240 | 1.000 | 170.307 | +0.000% | 111.511 | 31.761 | 27.035 | +0.000 |
| 8704 | `inc_nee` | 10240 | 1.000 | 162.363 | −4.665% | 111.437 | 31.842 | 19.084 | −4.665 |
| 8704 | `bkvp` | 9216 | 0.900 | 143.092 | **−15.980%** | 84.060 | 31.824 | 27.208 | −17.756 |
| 8704 | `d1` | 10240 | 1.000 | 162.445 | −4.616% | 111.435 | 31.902 | 19.108 | −4.616 |
| 8704 | `bkvp_d1` | 9216 | 0.900 | 134.879 | **−20.802%** | 84.060 | 31.733 | 19.087 | −23.114 |
| 8704 | `merged` | 9216 | 0.900 | 134.835 | **−20.829%** | 84.082 | 31.725 | 19.028 | −23.143 |

### Two effects, and they must be read apart

At `ctx ≤ 2048` with k=2048 and `compression_ratio=1`, `max(seq_lens) // 1 <= k` holds and
`inc` takes the **early-exit fast path** — 7.7 µs, no stage A, no stage B at all. So the
`Δ vs inc` column at those two rows is measuring D1's removal of the cond, not `bkv_p`.
The `inc_nee` arm (incumbent, `enable_early_exit=False`) and the second table below
separate them. `inc_nee` and `d1` should agree — same `bkv_p`, same layout, cond gone in
both, one by flag and one by deletion — and they do, to within 0.04%–0.22% at every
context, which is a useful independent check that the merged module's D1 deletion is
equivalent to turning the flag off and nothing more.

Against `d1` — the incumbent with the cond gone, which is on the scoring path at every
context — the `bkv_p` effect alone:

| ctx | scanned merged/inc | Δ(merged) full % vs `d1` | Δ(merged) stage A % vs `d1` |
| ---: | ---: | ---: | ---: |
| 1024 | 4.500 | **+98.411%** | +203.938% |
| 2048 | 4.500 | **+94.930%** | +201.391% |
| 4096 | 2.250 | **+37.107%** | +67.589% |
| 6144 | 1.500 | **+12.354%** | +20.470% |
| 8704 | 0.900 | **−16.997%** | −24.546% |

Stage A is the part `bkv_p` controls and it tracks scanned positions, sublinearly: at a
4.5x scan ratio stage A is 3.04x, at 2.25x it is 1.68x, at 1.5x it is 1.20x, at 0.90x it
is 0.755x. Larger blocks are **more efficient per position** — merged runs 9216 positions
at 0.00913 µs/pos against `d1`'s 0.01088–0.01353 — which is why the break-even sits below
a scan ratio of 1.0 rather than at it, and why the win at production (−20.8%) is larger
than the 10% of scan the ratio alone would buy. That is the mechanism behind the roughly
5.5 pp of the headline that the review attributed to overfetch, and it is real: it is
paid for at every other context in the table.

The **early-exit** effect is separate and, at short contexts, larger:

| ctx | `inc` (cond, taken) | `inc_nee` (no cond) | cost of deleting the cond |
| ---: | ---: | ---: | ---: |
| 1024 | 7.712 | 57.417 | **+644%** |
| 2048 | 7.751 | 58.941 | **+660%** |
| 4096 | 99.039 | 91.141 | −7.98% (the cond is *not* taken; the wrapper costs 7.9 µs) |
| 6144 | 122.731 | 114.545 | −6.67% |
| 8704 | 170.307 | 162.363 | −4.67% |

`merged` deletes the cond, so at `ctx ≤ 2048` it pays both: **+1376% and +1386% vs `inc`**,
114 µs against 7.7 µs, a 14.7x regression on a short request. `bkvp` — which keeps the
cond — is +3.1% / +3.2% there, because the fast path never touches `bkv_p`.

### Crossover

On full-op **against `inc`**, `merged` is slower at ctx 1024, 2048, 4096 and 6144 and
faster only at ctx 8704. The crossover is bracketed by the two measured points
**6144 (+4.975%) and 8704 (−20.829%)** and was not measured inside that bracket. Its
location is set by `inc`'s own quantization, not by a smooth curve: `inc` scans
`cdiv(cdiv(ctx,1024),2) × 2048`, so its scan jumps 6144 → 8192 → 10240 as `ctx` crosses
6144 and 8192, and the scan ratio goes 1.500 → 1.125 → 0.900. Both a scan-ratio argument
(merged is ~20% more efficient per position, so break-even is near ratio 1.2) and the
measured points put the crossover in `6144 < ctx ≤ 8192`. Anything in the bracket is
modelled, not measured.

On full-op **against `d1`** (the scoring-path-only comparison), the same bracket:
+12.354% at 6144, −16.997% at 8704.

### What this means for "compile `bkv_p` per context bucket"

It is not a safe recommendation on its own; it needs a request-length distribution
attached to it, and it needs the early exit handled separately.

- The measured win exists in a **narrow band of runtime lengths** — here only
  `6144 < ctx ≤ 9216`, the top ~30% of the `max_model_len` window. Below it the same
  compiled `bkv_p=9` costs +12% at 6144, +26% at 4096, +95% at 2048 on the scoring path.
- The overfetch is bounded by `bkv_p × page_size` = 9216 positions **regardless of how
  short the request is**, so the penalty is worst exactly where the request is cheapest,
  and it is a constant-cost floor rather than a proportional one.
- A per-bucket compilation fixes this by construction (each bucket gets a `bkv_p` matched
  to its own `kv_len_max`), but it multiplies compiled variants and it has to bucket on
  the *runtime* length, not on `max_model_len`. The sweep in `RESULTS.md` could not see
  any of this because every config there but production sets `mml == ctx`.
- **D1 is the separate decision.** Deleting the early exit buys 4.6% at production and
  costs 7.4x-7.6x on any request with `kv_len ≤ k`. Whether that trade is acceptable is a
  property of the serving mix, not of the kernel; a workload with any meaningful share of
  short requests should keep the cond, and D1's 4.6% is small enough that keeping it is
  cheap. (7.44x at ctx 1024, 7.60x at ctx 2048.) Note also the cond's cost when *not* taken is a flat ~7.9 µs of residual at every
  long context (27.0 vs 19.1 µs), i.e. D1's entire win is that fixed wrapper, which shrinks
  as a fraction as the op grows.

---

## Files

| file | what |
| --- | --- |
| `gaps_common.py` | arm table (adds `c`, `cctl`, `inc_nee`), the four plans, `scanned_positions()`, tag scheme. Imports `merged_common`; changes nothing in it. |
| `streamindex_topk_cond.py` | `streamindex_topk_merged.py` with `lax.cond`/`_fast_path` restored and an `enable_early_exit` static arg. Three functional hunks; nothing else differs. |
| `bench_gaps_trace.py` | capture driver. Fatal md5 gate, reps with the loop re-entered, inline index check vs `inc`, writes `trace_meta.<plan>.json`. |
| `gaps_shard.sh` | one fan-out shard: md5 read-back (fatal), j11 venv probe, trace. |
| `read_gaps_hlo_stats.py` | off-pod reader. Reuses `read_merged_hlo_stats.one()`; joins on `tag`; gates on partition residual and row completeness. |

Nothing under `src/vllm_torchtpu/` or `instruments/` was modified.

Reproduce:

```
cdk job create simple-run-command \
  DOCKER_IMAGE=<image above> \
  'RUN_CMD=… bash /root/fanout/tpu-fanout.sh "bash /root/kw/variants/merged/gaps_shard.sh e1" \
      "… e2a" "… e2b" "… e3"' \
  --map-dir <checkout>/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src \
  --map-dir <checkout>/kernel_work/streamidx-topk:tpu-job:/root/kw \
  --map-dir <kernel5>/.claude/skills/run-command-in-cdk/fanout:tpu-job:/root/fanout \
  -t agent-jobs,run-command-in-cdk
python read_gaps_hlo_stats.py <synced-dir> --out gaps.json --md gaps.md
```
