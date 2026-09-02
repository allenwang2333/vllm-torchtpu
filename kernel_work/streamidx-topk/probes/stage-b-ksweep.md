# Stage B — the `k` sweep, and stage B's first cost model that is not fitted to one point

**Verdict, in one line: stage B is issue-bound, not output-DMA-bound. 85.4 % of its prefill device
time is the length-bounded scan; the output DMA that `k` controls is ≤ 3.6 % and the input DMA is
5.3 %. The 481.0 µs prefill floor term is not device time — the device time at that exact shape is
335.5 µs, measured — and with that correction prefill is at 43–63 % of floor, not 80.4 %.**

## Provenance

| | |
| --- | --- |
| jobs | **`j-245d71f0-f6b2-41af-bd53`** (`JobStatus: Succeeded`) — `k` sweep, `b` sweep, dispatch floors<br>**`j-7a694535-6ea0-4918-bdef`** (`JobStatus: Succeeded`) — `eff_len` sweep and `n` sweep |
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` |
| `RUN_CMD` (read back from `cdk job recipe j-245d71f0-f6b2-41af-bd53`) | `cd /root/cloud-devkit/vllm-torchtpu && python -c "import vllm_torchtpu; print(vllm_torchtpu.__file__)" && python -c "from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as m; print(m.__file__)" && python /root/probes/probe_stage_b_ksweep.py` |
| `RUN_CMD` (job 2) | `cd /root/cloud-devkit/vllm-torchtpu && python -c "import vllm_torchtpu; print(vllm_torchtpu.__file__)" && python /root/probes/probe_stage_b_dma_scan.py` |
| `--map-dir` | `…/vllm-torchtpu/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src` and `…/kernel_work/streamidx-topk/probes:tpu-job:/root/probes` — both jobs, both flags printed a `Packaging dir` line |
| the mapping took | log line 1: `/root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/__init__.py`; line 2: `…/src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`. The kernel measured is the branch's, not the image's |
| `~/cloud-devkit` | `c4e1d4e`, one untracked file (`recipes/experimental/gdn-comparison.yaml`) |
| probes | `probes/probe_stage_b_ksweep.py`, `probes/probe_stage_b_dma_scan.py` (both new) |
| raw artifacts | `probes/results/stage-b-ksweep.j-245d71f0.json`, `probes/results/stage-b-dma-scan.j-7a694535.json` — committed, so every table below is recomputable from the emitted JSON rather than from this prose |
| hardware read back | `TPU7x`, 8 devices, `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16, vmem_capacity_bytes=524288, dma_granule_size_bytes=32)`, `mem_bw_bytes_per_second=3.7e12` |
| pinning | **none** — one unpinned process on the whole 4-chip pod, deliberately. A fanned-out capture is contended and yields ratios only; these numbers have to be comparable in absolute µs with `bench_prod.json`, so the pod was not shared with a sibling shard of this job. Other agents' jobs may have been running on other pods |

Nothing here is an end-to-end model benchmark. Everything is a stage-B microbenchmark.

---

## 0. Methodology — and why the ~42 µs dispatch floor is neither subtracted nor maxed

This is the part that changes every number downstream, so it comes first.

`instruments/bench_streamidx_topk.py:681` (`time_arm`) enqueues `iters` calls asynchronously and
blocks **once**, at the end. So the per-call figure it reports is not device time and not
`device + host`; on a long batch it is `max(device, host enqueue)`, and on a *short* batch it is
dominated by the un-amortised round-trip latency of the first call. `briefs/area5-compute-units.md`
§C.7 derives its 1.7 ns/elem constant by **subtracting** ~35 µs from a 67.4 µs batched
measurement. Subtraction is the wrong operation on a batched async measurement, and so is taking a
maximum.

So this probe measures device time a third way that needs no floor at all. **Chain R stage-B calls
inside one `jax.jit`** — dependency-linked through `rl + min(out[0,0], 0)`, which is identically
zero but which XLA's algebraic simplifier cannot fold the way it folds `x * 0`, so the R launches
can be neither CSEd nor reordered — and take the slope of `t_total(R)` over `R ∈ {1,2,3,4}`:

* the **slope** is marginal device time per call, with the host term differenced out;
* the **intercept** is what the host term actually was.

**Every chained fit came back at r² ≥ 0.9997, and 30 of 35 at r² ≥ 0.9999.** A chain that the
compiler had collapsed, or a pod that was interfering, would not be linear to four places.

Three numbers per point are reported throughout and they must not be confused:

| statistic | what it is |
| --- | --- |
| **chain slope** | dispatch-free device time per call. **This is the number to use.** |
| **batched best-of-9** | `bench_prod.json`'s methodology. `max(device, host enqueue)` plus an un-amortised first call. |
| **serial best-of-9** | one call with a `block_until_ready` after it: device + full host round trip. |

Measured dispatch floors, in-process, both jobs: `jit(x+1.0)` on a scalar = **17.71 µs** (job 1) /
**16.30 µs** (job 2); the same two-argument arity and buffers stage B gets, with a tiny output =
**21.48 / 16.46 µs**. `bench_prod.json` records `dispatch_floor_ms = 0.042005` for the same trivial
op. **The floor is not a constant of the hardware — it is a property of the session's host load,
and it moved by 2.4× between then and now.** It is reported here for comparability and is used for
nothing.

**Run-to-run spread on the chain slope is under 1 %, not ±8.5 %.** Two identical points measured in
two separate jobs, hours apart: `b=1024, n=10240, eff=8704, k=2048` → 332.92 µs and 331.89 µs
(**0.31 %**); `b=64` same → 29.00 µs and 29.12 µs (**0.41 %**). Within-job trial spread was
0.2–3.8 %. The ±8.5 % figure describes a different statistic; on this one, a 5 % difference is
resolvable.

Best-of-N is optimistic by construction and every timing here is a lower bound.

---

## 1. The `k` sweep — the answer is issue-bound, and it is not close

Held fixed: `b = 1024`, `n = 10240`, `row_lengths` = the production prefill pattern (8193…9216,
mean 8704.5, reproduced from `_row_lengths_ref` for `q_len=1024, seq_len=9216`). `_pick_partition`
returns **p = 1 at every `k` swept**, so the single-stage kernel is what ran throughout and `k` is
the only variable.

| `k` | chain slope (device) µs | batched µs | serial µs |
| ---: | ---: | ---: | ---: |
| 4096 | **316.46** | 401.39 | 566.78 |
| 2048 | **335.53** | 418.11 | 583.61 |
| 1024 | **331.17** | 412.49 | 580.67 |
| 512 | **324.47** | 405.02 | 577.90 |
| 256 | **323.48** | 403.89 | 563.99 |

**A 16× change in `k` moves device time by 6.0 %, non-monotonically.** The largest `k` is the
*fastest* point. A linear-in-`k` fit has r² = 0.14 and a negative slope. Output-DMA-bound predicts
`t(4096) = 16 × t(256) = 5176 µs`; observed 316 µs.

The decode arm agrees: `b=64, n=10240, eff=8704`, same five `k` — 28.52 / 29.17 / 29.09 / 27.99 /
28.89 µs, max/min = **1.042**.

**Issue-bound. Decisively.** The non-monotonicity is the one genuinely `k`-dependent term in the
source that is neither flat nor linear: how early the three `REFINEMENT_LEVELS` hit
`quota2 == cnt2` or `quota2 == 0` and set `done`, which is `k`- and data-dependent
(`sparsecore_topk.py:315-345`). It is worth ~6 % and it does not have a sign.

**Bound on the output DMA.** `4·k` bytes per row × 1024 rows: 8.39 MB at `k=2048`, 1.05 MB at
`k=256`. That 7.34 MB of difference bought at most `335.53 − 323.48 = 12.05 µs`, i.e. **≤ 3.6 % of
the call**. The brief's competing "50.3 MB at ~105 GB/s" DMA model required the write half to be
~17 % of the time. It is not.

---

## 2. Separating the un-bounded DMA from the length-bounded scan

The `k` sweep only exercises the *output* DMA. `sparsecore_topk.py:154-161` has the asymmetry that
lets the *input* side be separated too:

* every scan — `cvt`, `hist0_body`, `ref_body`, `emit_body` — is bounded by
  `walk_chunks = cdiv(min(slice_len, eff_len), 16)`, so it tracks **`eff_len`**;
* the resident-slice DMA is `dma(scores_hbm.at[pl.ds(s, slice_len)], …)` with
  `slice_len = align_to(n,16)` a **trace-time constant**, so it tracks **`n`** and is blind to
  `row_lengths`.

Two orthogonal levers, job `j-7a694535`. Per-wave figures = chain slope ÷ wave count.

**Arm A — `eff_len` swept, `n = 10240` fixed** (the DMA is byte-identical at every point):

| `eff_len` | b=64 (2 waves) µs | b=1024 (32 waves) µs |
| ---: | ---: | ---: |
| 1024 | 13.20 | 97.22 |
| 2048 | 14.05 | 110.46 |
| 4096 | 19.63 | 181.73 |
| 6144 | 24.87 | 262.63 |
| 8192 | 28.60 | 320.53 |
| 8704 | 29.12 | 331.89 |
| 10240 | 31.96 | 372.73 |

per-wave fits: `5.35 µs + 1.069 ns × eff_len` (b=64, r²=0.992); `1.80 µs + 0.984 ns × eff_len`
(b=1024, r²=0.994).

**Arm B — `n` swept, `eff_len = 2048` fixed** (`walk_chunks` = 128 at every point, so the scan work
is identical):

| `n` | b=64 (2 waves) µs | b=1024 (32 waves) µs |
| ---: | ---: | ---: |
| 2048 | 13.46 | 97.73 |
| 4096 | 13.59 | 102.05 |
| 8192 | 13.79 | 108.25 |
| 10240 | 14.00 | 110.62 |
| 16384 | 14.59 | 119.29 |
| 32768 | 16.30 | — |
| 65536 | 18.68 | — |

per-wave fits: `6.61 µs + 0.0424 ns × n` (b=64, r²=0.994); `2.99 µs + 0.0460 ns × n` (b=1024,
r²=0.995).

The two slopes reproduce across a 16× change in wave count, which is what makes them a model rather
than a calibration.

---

## 3. The cost model

Fitted over **35 single-stage points from two jobs**, spanning **waves 1 → 32**, **`n` 2048 →
65536**, **`eff_len` 1024 → 10240** — three orthogonal levers, none of them the point the model is
used at:

```
t_device (µs) = 8.15  +  waves × ( 0.702  +  0.0542 ns × n  +  1.0197 ns × eff_len )

  waves = ceil(b · p / 32)      n, eff_len in elements
```

r² = **0.9982**, rms residual **5.1 %**, max residual 15.9 % (at `b=1024, eff=1024` — the extreme
low-`eff_len` corner, where the model runs low; every point at production-like `eff_len` is inside
4 %).

This is **not** the 1.7 ns/elem constant. It is 1.02 ns/element of scan, plus a separately measured
0.054 ns/word of DMA, plus two fixed terms. §C.7's 1.7 ns was a single constant absorbing all four
of those at one operating point, which is why it mispredicted every other row of `bench_prod.json`
by 2–6×.

**Contrast with the wave-only model.** Fitting only `t = a + c·waves` at `n=10240, eff=8704` (the
`b` sweep alone, 8 points, waves 1→32) gives `t = 8.30 + 10.17 µs × waves`, r² = 0.99990. That is
the same model collapsed onto the production shape, and it is the one-line form worth carrying:
**one wave of 8704-element rows at `n=10240` costs 10.2 µs.**

**What the model does NOT cover, stated plainly:**

* **`p > 1`.** Every fitted point is single-stage. The two-stage path is §5 and is measured, not
  modelled.
* **`n > 65536`** and `eff_len > 10240` are extrapolation.
* **The score distribution.** All arms use the same fixed N(0,1)-with-`-inf`-tail matrix. The radix
  select is data dependent (§1's 6 % non-monotonicity is that dependence), so the model carries a
  few-percent data term it cannot see.
* It is a model of **the incumbent**, not a floor. See §6.

### What prefill's 335.5 µs is made of

At `waves=32, n=10240, eff_len=8704` the model gives 332.4 µs against the 335.5 µs measured
directly:

| term | µs | share |
| --- | ---: | ---: |
| scan, tracks `eff_len` | **284.0** | **85.4 %** |
| per-wave fixed × 32 | 22.5 | 6.8 % |
| input DMA, tracks `n` | 17.8 | 5.3 % |
| per-call fixed | 8.2 | 2.5 % |
| output DMA (`k`) | ≤ 12 (inside the above) | ≤ 3.6 % |

**Consequence for the brief's "single highest-value stage-B probe".** The brief nominates bounding
the input DMA to `row_lengths` as the top candidate. Priced with the measured DMA coefficient, at
the production shape it reads `n = 10240` where `eff_len = 8704`, so the waste is
`32 × 0.0542 ns × 1536 = 2.67 µs` — **0.80 % of the call.** Not worth a probe at this shape. It
*is* worth one at AgentX width: at `n/p = 65536` the DMA is 3.56 µs/wave against 8.35 µs/wave of
scan, i.e. **30 % of the wave**. The candidate is real; the shape it pays at is not the production
one, and the brief does not distinguish them.

The DMA coefficient implies 4 B / 0.0542 ns = **73.7 GB/s per row**, ×32 subcores = **2.36 TB/s
aggregate = 64 % of the reported 3.70 TB/s**. The read is not the problem.

---

## 4. Why `bench_prod.json`'s stage-B numbers are 35–150 µs above device time

| row | waves | model device µs | `bench_prod` wall µs | gap µs | `iters.stage_b` |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode b64 ctx8192 | 2 | 27.4 | 66.56 | 39.2 | 8 |
| decode b64 ctx8704 | 2 | 28.4 *(29.00 measured)* | 67.40 | 39.0 | 8 |
| decode b64 ctx9216 | 2 | 29.5 | 64.19 | 34.7 | 9 |
| decode b32 ctx8704 | 1 | 18.3 *(17.95 measured)* | 63.63 | 45.3 | 7 |
| decode b8 ctx8704 | *p=4* | *50.09 measured* | 90.30 | 40.2 | 7 |
| prefill ctx5120 | 32 | 198.8 | 338.71 | 139.9 | 4 |
| **prefill ctx9216** | 32 | 332.4 *(**335.5 measured**)* | **480.97** | **145.5** | **3** |

**The gap tracks `iters`, not the dispatch floor.** At `iters` 7–9 it is 35–45 µs; at `iters` 3–4 it
is 140–150 µs. The mechanism is the un-amortised round-trip latency of the first call in each
batch, which this probe measures directly: **serial** best-of-9 is 155–190 µs at decode shapes
against 13–29 µs of device time, and 564–609 µs at prefill shapes against 316–373 µs.

At the prefill ctx9216 shape the three statistics bracket it without any modelling:

```
chain slope (device)   335.53 µs
batched, iters = 84    418.11 µs
bench_prod, iters = 3  480.97 µs      <-- must land between the two below
serial (iters = 1)     583.61 µs
```

`480.97` is exactly where an `iters = 3` measurement has to sit. It is a wall clock with two thirds
of a round trip still in it.

**This also puts the rest of that row's decomposition in doubt.** `bench_prod`'s prefill ctx9216
row derives `stage_a_derived_ms = total − stage_b − init` from three quantities measured at
`iters` = 2, 3 and 10 respectively, each carrying a *different* amount of un-amortised latency.
The resulting 335.07 µs for stage A is a subtraction across inconsistent statistics. **This probe
did not measure stage A or the total and cannot correct them** — it is flagged as the largest
remaining uncertainty in the prefill record, not resolved.

---

## 5. The b=8 > b=64 inversion — `_pick_partition` picks the two-stage path

`_pick_partition(b, n=10240, k=2048)` (`sparsecore_topk.py:357-372`), evaluated from source before
anything ran: **p = 4 at b=8, p = 2 at b=16, p = 1 at every b ≥ 32** — because `max_p = 32//b` is
4 and 2 there, and `n//(p·2) ≥ k` permits the raise. p > 1 is the two-stage path: two
`_sc_topk_direct` launches plus two `jnp.take_along_axis` gathers on the critical path.

Measured, device time (chain slope), `n=10240, eff=8704, k=2048`:

| `b` | `p` chosen | public path µs | forced single-stage (`_sc_topk_direct`) µs | penalty |
| ---: | ---: | ---: | ---: | ---: |
| 8 | **4** | **50.09** | **17.22** | **2.91×** |
| 16 | **2** | **45.68** | **17.35** | **2.63×** |
| 32 | 1 | 17.95 | — | — |
| 64 | 1 | 29.00 | — | — |
| 128 | 1 | 50.74 | — | — |
| 256 | 1 | 90.11 | — | — |
| 512 | 1 | 172.19 | — | — |
| 1024 | 1 | 332.92 | — | — |

**The inversion is real in device time**: b=8 costs 50.09 µs against b=64's 29.00 µs. It is not a
host artifact and it is not the dispatch floor. It is `_pick_partition`.

And it composes exactly with §4's host gap to reproduce `bench_prod` without a free parameter:

```
b=8 :  50.09 device + 40.2 host  =  90.3     bench_prod: 90.30
b=64:  29.00 device + 38.4 host  =  67.4     bench_prod: 67.40
```

**Why p=4 loses.** At b=8, stage 1 does 32 rows × 2560 elements (1 wave, cheap), but then stage 2
does 8 rows × 8192 candidates (another wave, nearly the full cost of just doing it once), and
between them sit two `jnp.take_along_axis` gathers. The single-stage kernel does 8 rows × 8704 in
**one** wave for 17.22 µs — and the model predicts 18.29 µs for it, so nothing exotic is happening
on the p=1 side. The parallelism p>1 buys is real but the rows were already going to fit in one
wave; it buys nothing and pays for a second launch and two XLA ops.

**Candidate, one line, not benchmarked end to end:** make `_pick_partition` return 1 whenever
`cdiv(n, MAX_SLICE_WORDS) == 1` and `b·1 ≤ 32` — i.e. raise `p` only for the capacity reason, never
for the occupancy reason. Modelled saving at the measured device times: **32.9 µs/call at b=8
(65.7 % of stage B) and 28.3 µs at b=16**. It is small-batch decode only and it does not touch
b≥32 or prefill. It has **no correctness evidence** — the two paths must produce identical index
sets and that was not tested here.

---

## 6. What this does to the 481.0 µs floor term and the 80.4 % claim

`F_struct` = 689.3 µs prefill, of which `t_sc` = **481.0 µs = 69.8 %**. The remaining 208.3 µs is
everything that is not stage B.

The adversarial review's objection is upheld and then some. `t_sc = 481.0` was `32 × 8704 ×
1.7 ns = 473.5 µs`, with the 1.7 ns fitted to `stage_b_ms = 0.48097` — **and that 480.97 µs is not
even stage B's device time.** It is a wall clock at `iters = 3` carrying ~145 µs of round trip
(§4). The floor's dominant term was a measurement, and the wrong measurement.

Four treatments, denominator = `bench_prod`'s 857.3 µs prefill call:

| treatment | `t_sc` | `F_struct` | achieved |
| --- | ---: | ---: | ---: |
| (a) **as published** — `t_sc` = the measurement it was fitted to | 481.0 | 689.3 | **80.4 %** |
| (b) `t_sc` = **measured device time** at the same shape | **335.5** | 543.8 | **63.4 %** |
| (c) `t_sc` = **irreducible-scan floor**: §C.8's ~25 of ~65 ops/chunk, rescaled onto the measured 284.0 µs scan term, + the measured DMA and fixed terms | 157.6 | 365.9 | **42.7 %** |
| (d) `t_sc` = **pure HBM floor**: (41.9 + 8.4) MB @ 3.70 TB/s | 13.6 | 221.9 | 25.9 % |

**(a) is dead.** It is not merely circular — the number it is circular against is a host-contaminated
wall clock, so it overstates the floor by 43 % on its own dominant term.

**(b) is still circular in kind** — an achieved number used as a floor — but it is now the *right*
achieved number, and it is the honest ceiling on the achieved fraction: stage B cannot possibly have
a floor above what it already runs at. **63.4 % is an upper bound, not an estimate.**

**(c) is the best available point estimate**, ~43 %. Its op-count ratio (25/65) is `modelled` from
`briefs/area5-compute-units.md` §C.8's reading of the source, but it is now applied to a **measured**
284.0 µs scan term rather than to a fitted constant, so only the ratio is modelled.

**(d) is a real hardware floor and shows how far from it stage B is: 24×.** Stage B is not
bandwidth-limited by anything; it is limited by SparseCore vector issue. That is the finding that
makes (c) the relevant floor and not (d).

**So: the range narrows from 46.5 %–80.4 % to roughly 43 %–63 %, and the answer to "is prefill at
80.4 % of floor?" is no.** This matters beyond bookkeeping: `write-tpu-kernels` step 1 refuses to
write a kernel for a block already at ≥ 80 % of its modelled floor, and prefill was clearing that
bar **only** on the strength of the self-fitted term. It does not clear it. Stage B has headroom,
and §3's decomposition says where it is: **85.4 % of the time is the scan, so the lever is
operations per 16-lane chunk, not DMA, not `k`, not layout.**

Two caveats on the range that I did not resolve:

1. **The denominator is contaminated too.** 857.3 µs is `total_ms` at `iters = 2`. If it carries a
   comparable round-trip residue, every fraction above rises. I did not measure the whole op and
   have deliberately not folded a guess in.
2. **(c)'s 25/65 ratio is `modelled` from source op counts.** A candidate algorithm measured on
   hardware would replace it. Nothing in this probe measures a different algorithm.

---

## 7. What did not get measured

* **Nothing here bears on stage A**, on the whole-op total, or on the `vrot.slane` question — that
  is R3.
* **No correctness check was run.** The `_pick_partition` candidate in §5 is a perf observation with
  `⚠️ unverified` correctness; the two paths' index sets were not compared.
* **The two-stage path is measured at two points, not modelled.** The §3 model covers `p = 1` only.
* **Per-element linearity in `eff_len` is established between 1024 and 10240 at `n = 10240` only.**
  It is not established at other `n`, and arm B holds `eff_len` at 2048 rather than sweeping the
  pair jointly.
* **The score distribution is held fixed** at N(0,1) with a `-inf` tail. The ~6 % non-monotonicity in
  §1 is the data dependence the model cannot see; a real indexer score distribution could move the
  refinement-termination term further.
