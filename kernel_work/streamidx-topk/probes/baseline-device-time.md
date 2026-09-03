# The corrected device-time baseline, and candidate 0 measured against it

**In one line: the baseline of record was a wall clock, and the whole op's device time is
170.6 µs at decode (not 221.4) and 749.5 µs at prefill (not 857.3). Two independent methods agree
to 0.08 %–1.6 % everywhere, after one real disagreement was traced to its cause and closed. Every
achieved fraction rises, because every denominator falls: decode is at 33–53 % of its modelled
floor and its verdict is unchanged; prefill's band is 26–92 %, point estimate ~54 %, and its
*ceiling* has moved above the 80 % bar for the first time — so the prefill gate is now undecidable
rather than clearly-not-cleared. Candidate 0 survives as a clean absolute at −9.5 % on the decode
call at ctx 8704, it is confirmed to be the `9 = 3×3` alignment effect, and at ctx 8192 it inverts
to +2.7 %.**

> Every floor figure in this file is `modelled` and is labelled as such. An **achieved fraction is
> not a measurement** — it is a measured device time divided by a modelled floor, and it inherits
> every uncertainty in the floor. The measurements here are the device times and only the device
> times.

## Provenance

| | |
| --- | --- |
| jobs | **`j-79a410b8-46bc-4d06-862c`** (`Succeeded`) — Task 1, Task 2, `hlo_stats` traces, the unit test<br>**`j-60140613-7496-4b63-9a89`** (`Succeeded`) — §4, the stage-B input follow-up |
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` (reused; nothing was built) |
| `RUN_CMD` | `bash /root/probes/bdt_shard.sh` / `bash /root/probes/sbi_shard.sh` |
| `--map-dir` | `…/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src`, `…/tests:…/tests`, `…/kernel_work/streamidx-topk/instruments:tpu-job:/root/instruments`, `…/kernel_work/streamidx-topk/probes:tpu-job:/root/probes` — all four printed a `Packaging dir` line in both jobs |
| **md5 read-back, in-pod, both jobs** | `streamindex_topk.py` **`1e221f10cbf3573aa0d2ae78c3463532`**<br>`sparsecore_topk.py` **`8bc9e25b8701a53a97704564b9fbc658`**<br>`bench_streamidx_topk.py` **`e8e492d13e59c4a1675cc7a98c5a0813`**<br>`_stage_a.py` **`eea6a51bb40a2bbf7cf5b39571753b62`**<br>`tests/kernels/test_dsv4_streamindex_topk.py` **`becf9f23ace9424a7d67cd3b34de9afd`** |
| `~/cloud-devkit` | `c4e1d4e88e1d6b8c81d7e6051f43d9bfe19f1cb2`, clean |
| step rc table (job 1) | `env 0 · pytest_deps 0 · pytest 0 · chain 0 · j11_venv 0 · trace 0` |
| pinning | **none** — one unpinned process on the whole 4-chip pod in each job, deliberately. These numbers have to be absolutes comparable with `bench_prod.json`; `probes/P7-bkvp-and-spill.md` §A4.1 could only report ratios because its four shards contended. Other agents' jobs may have run on other pods |
| probes (new) | `probe_baseline_device_time.py`, `probe_bdt_trace.py`, `bdt_hlo_stats.py`, `bdt_shard.sh`, `probe_stage_b_input.py`, `sbi_shard.sh` |
| raw artifacts | `probes/results/baseline-device-time.j-79a410b8.json`, `…-hlostats.j-79a410b8.json`, `…/stage-b-input.j-60140613.json` — every table below recomputes from these, not from this prose |
| hardware read back | `TPU7x`, 8 devices, `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16, vmem_capacity_bytes=524288, dma_granule_size_bytes=32)` |

Nothing here is an end-to-end model benchmark. No entry was made in the shared experiment log.

The md5 guard is not ceremony. The image at `ec4bcecf` predates these kernel modules and they are
mapped in over its checkout; an earlier job in this exercise measured the image's *older* kernel
and returned rc=1 on all four shards behind a green `JobStatus`. A green `JobStatus` is not
evidence your code ran; the md5 line is.

---

## 0. Why the baseline of record needed replacing

`instruments/bench_streamidx_topk.py:681` (`time_arm`) enqueues `iters` calls asynchronously and
blocks **once**. The per-call figure it reports is therefore not device time and not
`device + host`: it is `max(device, host enqueue)` plus an un-amortised first round trip spread
over `iters`. `bench_prod.json`'s prefill row came from `iters: {'total': 2}` and its decode row
from `iters: 3`.

Two operations on that number are invalid and both have been performed in this exercise's
documents:

* **Subtracting a dispatch floor** (`briefs/area5-compute-units.md` §C.7 takes ~35 µs off a 67.4 µs
  batched measurement). A host floor on a batched async measurement is a **clamp, not an addend**.
* **Taking a maximum against one.** Same reason.

So this file does not use the dispatch floor for anything. It measures device time two ways that
need no floor at all, and reports the floor only so the reader can see it is not what the gap is
made of.

### Method 1 — chain slope

`R` calls chained inside one `jax.jit`, dependency-linked so XLA can neither CSE nor reorder them,
and the slope of `t_total(R)` taken over `R ∈ {1,2,3,4}`. The slope is marginal device time per
call; the intercept is the host term and is discarded.

The link is `x + jnp.minimum(out[0,0], jnp.int32(0))` — identically zero, but not foldable the way
`x * 0` is. Two guards on it, both reported in the JSON:

* `chain_perturbation_is_zero: true` — the perturbation was checked numerically to be zero at
  every ctx, so the chain does not change what is computed.
* `chain_intact` — the count of `StreamIdxTC` / `sc_topk` markers in
  `fn.lower(*args).compile().as_text()` grows with `R`. **`true` on all six Task-1 fits**
  (`R=1 → 12` and `R=4 → 48` for the full op; `3 → 15` stage A; `18 → 69` stage B). Taken only at
  the endpoint `R` values, because `lower().compile()` does not hit the tracing cache and doubles
  the compile cost.
* **Fit quality: every fit reported here has r² ≥ 0.999363**, and the six Task-1 fits are
  r² ≥ 0.99990. The pre-registered rejection threshold was 0.999; nothing was rejected.

### Method 2 — `hlo_stats` per-instruction device self time

Carries no dispatch floor by construction. Traces captured from a jax/jaxlib 0.11.0 side venv (the
image's 0.10.2 SIGSEGVs against libtpu 0.0.44.1), dumped with `--top-n 500 --trace-sample 0`, and
bucketed as `probes/R2.md` §1 does: TensorCore rows into stage A (`StreamIdxTC`), stage B
(`sc_topk`) and residual, with the SparseCore plane reported **separately and never added** to the
TensorCore total — the two planes run concurrently.

* **Partition check `sum(buckets) − TC total`: `+0.000000` on all ten traces**, with
  `rows_kept == rows_total` (42/42 decode, 27/27 prefill). R2's first dump silently lost rows to
  `--top-n 60`; both the wide `top-n` and the explicit row-count assertion are the fix.
* The reader was validated **before** the job was spent, by running it against R2's existing
  `xplane.pb`: it reproduced R2 exactly (TC 170.474 / A 111.688 / B 31.807 / residual 26.979,
  partition +0.000000).

### The host dispatch floor — measured, and used for nothing

| | µs |
| --- | ---: |
| `jit(x + 1.0)` on a scalar, in-process | **28.54** |
| same two-argument arity and buffers stage B gets (2.62 MB in, tiny out) | **30.20** |
| `bench_prod.json`'s recorded `dispatch_floor_ms` | 42.005 |
| `stage-b-ksweep.md`'s two earlier jobs | 16.30–21.48 |

**The floor is not a constant of the hardware.** It has moved by 2.6× across four jobs on the same
image. It is context only. It is not subtracted anywhere in this file, and the wall-clock gaps in
§1 are larger than it in every case where it would have mattered.

---

## 1. Task 1 — the corrected device-time baseline

### Decode, `prod_decode_b64_ctx8704` (batch 64, ctx 8704, `bkv_p = 2`)

| | chain slope (r²) | `hlo_stats` | Δ | `bench_prod` wall | wall / device |
| --- | ---: | ---: | ---: | ---: | ---: |
| **full op** | **173.37** (0.999827) | **170.579** | +1.6 % | 221.4 | **1.298** |
| full op, Task-2 control | 171.21 (0.999997) | 170.579 | +0.4 % | | |
| stage A | 109.65 (0.999998) | 111.708 | −1.8 % | 109.2 (derived) | 0.978 |
| stage A, single call only | 109.65 | 109.890 | **−0.22 %** | | |
| stage B | 28.77 (0.999900) | 31.731 | −9.3 % | 67.40 | **2.124** |
| residual (init, bitcast, `cond`, slices) | — | 27.140 | | 44.8 (`init_ms`) | |

### Prefill, `prod_prefill_q1024_ctx9216` (1024 q, ctx 9216)

| | chain slope (r²) | `hlo_stats` | Δ | `bench_prod` wall | wall / device |
| --- | ---: | ---: | ---: | ---: | ---: |
| **full op** | **750.14** (0.999996) | **749.538** | +0.08 % | 857.3 | **1.144** |
| stage A | 247.63 (0.999990) | 249.300 | −0.7 % | — | |
| stage A, single call only | 247.63 | 246.243 | **+0.56 %** | | |
| stage B | 336.48 (0.999999) | 431.153 | **−22.0 %** | 480.97 | 1.116 |
| residual | — | 69.085 | | 44.8 (`init_ms`) | |

Two accounting notes, both needed to read the "single call only" rows:

* **The chain's stage-A arm launches one `pallas_call`; the op launches three.** At decode the op
  runs the grouped `DECODE` call plus an empty remainder call plus an empty `MIXED` call; at
  prefill only the `MIXED` call does work. `hlo_stats` sees all three. Against the one that does
  the work the two methods agree to **0.22 %** (decode) and **0.56 %** (prefill). The two empty
  companion launches cost 1.82 µs at decode and 3.06 µs at prefill — consistent with `probes/P4`.
* **`hlo_stats`' `stage_b_on_tc` bucket is the TensorCore-side `sc_topk…call-done` row**, and it
  runs 431.020 µs at prefill against 430.830 µs on the SparseCore plane. The TensorCore is
  *blocked* for the SparseCore's duration; the two are not additive and are not overlapped with
  anything else here.

### Composition closes

```
prefill   A 249.300 + B 431.153 + residual 69.085 = 749.538   ==  TC total 749.538   (exact)
          chain full op 750.14                                        Δ = +0.08 %
decode    A 111.708 + B  31.731 + residual 27.140 = 170.579   ==  TC total 170.579   (exact)
          chain full op 171.21 (Task-2 control) / 173.37 (Task 1)     Δ = +0.4 % / +1.6 %
```

The exactness is the partition check, not a coincidence — it is what `+0.000000` means.

### How far the recorded wall clocks were from device time

| | wall | device | gap | note |
| --- | ---: | ---: | ---: | --- |
| decode call | 221.4 | 170.579 | **+50.8 µs (+29.8 %)** | `iters: 3` |
| prefill call | 857.3 | 749.538 | **+107.8 µs (+14.4 %)** | `iters: {'total': 2}` |
| decode stage B | 67.40 | 31.731 | +35.7 µs (+112 %) | `iters: 8` |
| prefill stage B | 480.97 | 431.153 | +49.8 µs (+11.6 %) | `iters: 3` |

**The gap is not the dispatch floor.** At decode it is 50.8 µs against an in-process floor of
28.5–30.2 µs; at prefill it is 107.8 µs against the same floor. It tracks `iters`, exactly as the
un-amortised-first-call account predicts, and that is why subtracting a constant ~35 µs from every
row was never going to work.

**This corrects `stage-b-ksweep.md` §4 in the direction that matters.** That section put
`bench_prod`'s prefill stage-B wall clock at 145 µs above device time and concluded the 481.0 µs
floor term "overstates the floor by 43 % on its own dominant term". Against the real in-situ
device time the wall clock is only **49.8 µs high**, and the `1.73 ns/elem` model output
(`32 × 8704 × 1.73 ns = 481.9 µs`) overstates by **11.8 %, not 43 %**. The term was still fitted to
a wall clock and is still circular — but it was accidentally close, and §4's "35–150 µs above
device time" was measured against the wrong stage-B baseline. §4 is superseded by this table.

---

## 2. Task 1 restated — the achieved fractions, and how the prefill gate reads now

Floors are `measurement_and_design.md` §1.6's, unchanged, all `modelled`:
`F_struct = init + stage A + t_sc` with HBM assumed to overlap compute and the compute units
assumed not to overlap each other; `F_common = max(HBM, TC compute, SC)` for the least-constrained
legal structure. Only the **denominators** change — from wall clock to measured device time.

The four `t_sc` treatments are `probes/stage-b-ksweep.md` §6's, with (b) and (c) re-derived onto
the in-situ measured stage-B device time rather than the isolated synthetic one (see §4 for why
that is the right input).

### Decode — denominator **170.58 µs** (was 221.4)

| treatment for `t_sc` | `t_sc` | `F_struct` | achieved (device) | *(as published, vs wall)* |
| --- | ---: | ---: | ---: | ---: |
| (a) published modelled 30.0 | 30.00 | 88.25 | **51.7 %** | *39.9 %* |
| (b) = measured device — circular, a ceiling | 31.73 | 89.98 | **52.8 %** | *40.6 %* |
| (c) irreducible-scan floor, rescaled | 18.76 | 77.01 | **45.1 %** | *34.8 %* |
| (d) pure HBM floor | 1.06 | 59.31 | 34.8 % | *26.8 %* |
| `F_common` (either `t_sc`; HBM dominates) | — | 56.66 | **33.2 %** | *25.6 %* |

**Decode: 33 %–53 %, point estimate ~45 %. The verdict does not change.** It was "26–57 %,
structural band, proceed"; it is now **33–53 %, structural band, proceed**, and the band is
narrower. Nothing in it approaches the 80 % bar, and the point estimate sits at the `<50 %`
"structural, a kernel usually pays" line.

### Prefill — denominator **749.54 µs** (was 857.3)

| treatment for `t_sc` | `t_sc` | `F_struct` | achieved (device) | *(as published, vs wall)* |
| --- | ---: | ---: | ---: | ---: |
| (a) published modelled 481.0 — fitted to a wall clock | 481.00 | 689.32 | **92.0 %** | *80.4 %* |
| (b) = measured device — circular, a ceiling | 431.15 | 639.47 | **85.3 %** | *74.6 %* |
| (c) irreducible-scan floor, rescaled | 195.67 | 403.99 | **53.9 %** | *47.1 %* |
| (d) pure HBM floor | 13.60 | 221.92 | 29.6 % | *25.9 %* |
| `F_common`, `t_sc` = measured | — | 431.15 | **57.5 %** | *56.1 %* |
| `F_common`, `t_sc` = (c) | — | 195.67 | 26.1 % | — |

#### How the prefill gate reads now — and it is worse news than last time

`stage-b-ksweep.md` closed with "the range narrows from 46.5 %–80.4 % to roughly 43 %–63 %, and the
answer to 'is prefill at 80.4 % of floor?' is **no**." **That conclusion no longer stands as
stated.** The 63.4 % ceiling in it was `F_struct` with `t_sc` = the *isolated synthetic* stage-B
device time (335.5 µs) over a *wall-clock* denominator (857.3 µs). Both inputs were wrong, and
they were wrong in the same direction:

* the denominator was 14.4 % too high → every fraction was too low;
* `t_sc` under treatment (b) was 22 % too low, because the synthetic score distribution makes the
  radix select cheaper than production data does (§4).

Correcting both moves the ceiling from **63.4 % to 85.3 %**, across the bar.

**So, plainly:**

* **Prefill does not clear ≥ 80 % on any non-circular treatment.** (c), the best available point
  estimate, is **53.9 %**; `F_common`, the document's own declared cross-option yardstick, is
  **57.5 %**. Both sit in the 50–80 % "judgement call" band, not in the "do not write a kernel"
  band.
* **But the ceiling now clears it.** Treatment (b) — grant that stage B is exactly at its floor,
  and charge stage A and init their modelled floors — reads 85.3 %. Under (b) only 110 µs of the
  749 µs call is headroom outside stage B.
* **Therefore the gate is undecidable, and it is undecidable for a reason that has not moved
  since `ROOFLINE-REVIEW.md` F1**: 431 µs of the 749 µs call is SparseCore radix select, and
  nobody has a first-principles floor for it. Every treatment above differs only in what it
  guesses that floor is, and they span 13.6 → 481 µs. `hardware-facts.md`: "No SparseCore fact is
  measured."
* **The one thing that is now settled is that the *denominator* is no longer part of the
  uncertainty.** `ROOFLINE-REVIEW.md` F4's cleanest objection — that the same configuration was
  measured twice at 857.3 and 929.9 µs and the two straddled the bar — is closed. The device time
  is 749.5 µs by two methods that agree to 0.08 %, and `stage-b-ksweep.md` §0's run-to-run
  reproducibility on the chain slope is 0.31 %.

#### Where the headroom actually is — the useful form of the same arithmetic

Measured device time minus the modelled floor, per bucket. This does not depend on how the floors
are summed, so it survives the `F_struct` / `F_common` argument entirely.

| | measured | modelled floor | headroom | share of call |
| --- | ---: | ---: | ---: | ---: |
| **decode** stage A | 111.71 | 57.37 | **54.3** | **31.8 %** |
| **decode** residual | 27.14 | 0.88 (init) | **26.3** | **15.4 %** |
| **decode** stage B | 31.73 | 18.76 (c) | 13.0 | 7.6 % |
| **prefill** stage B | 431.15 | 195.67 (c) | **235.5** | **31.4 %** |
| **prefill** stage A | 249.30 | 194.20 | 55.1 | 7.3 % |
| **prefill** residual | 69.09 | 14.12 (init) | 55.0 | 7.3 % |

Read across the two phases: **decode's headroom is stage A and the residual; prefill's is stage B,
with stage A and the residual tied a distant second.** Stage A is at **51.4 %** of its modelled
floor at decode and **77.9 %** at prefill — the same code, nearly at the bar in one phase and
half-way in the other, which is what §1.6 means when it says the two phases are bound by different
things (HBM-bound 2.6× at decode, VPU-bound 9.2× at prefill).

The residual is worth naming, because it is unmodelled in both phases and it is 15 % of the decode
call. Its top rows:

| | decode µs | prefill µs |
| --- | ---: | ---: |
| `bitcast_convert_type.3` | 3.47 | **28.29** |
| `broadcast.*.scores_in_hbm_ref` (the `-inf` init) | 1.66 | 13.75 |
| `cond.5.clone` | 6.63 | 6.87 |
| `reshape.1` | — | 6.35 |
| `reduce_max.7`, slices, iota/compare fusions | ~6.1 | ~4.5 |
| rest | ~9.3 | ~9.3 |

**The `-inf` init is not the residual.** §1.6 models it at 14.12 µs prefill and it measures
13.754 µs — a 2.7 % match, the best-predicted term in the whole model. This settles
`ROOFLINE-REVIEW.md`'s disposition item 4 and §1.1's disputed relabelling in the *other*
direction: `init_ms = 44.8` really was ~69 % host dispatch (13.75 device against a 44.8 wall), and
the genuine residual the review estimated at "≈60 µs" for decode is **26.3 µs at decode and
55.0 µs at prefill**, measured, and it is dominated by a `bitcast_convert_type` over the score
matrix (28.29 µs at prefill = 3.8 % of the call, an unmodelled extra pass) rather than by the
fill.

---

## 3. Task 2 — candidate 0, `bkv_p` at decode

`bkv_p` is `num_kv_pages_per_block`, a call-site parameter. Nothing in `src/` was edited; the
tuple is passed to `streamindex_topk`, and only its first entry (the decode entry) is varied, so
the tuple is `(p, 2, 2)`. All arms are in **the same process** as Task 1's baseline, so each is
read against its own controls.

### 3.1 The confound that had to be removed first

`num_sublanes_total = max(align_to(pages_per_seq, p) · page_size // 128 for p in the tuple)`, and
`pages_per_seq = cdiv(max_model_len 9216, 1024) = 9` at every ctx swept. So

* `(2,2,2)` and `(3,2,2)` both give `num_sublanes_total = 80` — the prefill entries pin it;
* `(4,2,2)` gives **96**, a 20 % wider score matrix for the *whole op*, prefill included.

`bkv_p = 4` therefore carries a structural penalty that `bkv_p ∈ {2,3}` do not, and a full-op
comparison at `bkv_p = 4` is not like-for-like. `probes/P7`'s four-shard result could not separate
these. Both views are reported: the full-op grid (which pays the penalty, and is what production
would pay) and an isolated stage-A grid run at a **common width** using `_stage_a.make_stage_a_fn`'s
explicit `num_sublanes_total` — at 96 for all three `bkv_p`, and again at 80 where it fits. The
`like_for_like_with_bkvp2` flag in the JSON marks which is which.

**Result: the width is not a confound at all.** Stage A at `nst = 96` and at `nst = 80` agree to
within 0.4 % at every point where both were run (90.24 / 90.28; 109.87 / 109.60; 87.47 / 87.77).
Stage A's cost does not depend on the score buffer's width, only on what it fetches.

### 3.2 The full-op grid — device time, chain slope

| ctx | `bkv_p` | tuple | `nst` | `num_bkv` | pages fetched | device µs | r² | vs `bkv_p=2` |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 2 | (2,2,2) | 80 | 4 | 8 | **149.89** | 0.999989 | — |
| 8192 | 3 | (3,2,2) | 80 | 3 | 9 | 153.97 | 0.999456 | **+2.72 %** |
| 8192 | 4 | (4,2,2) | 96 | 2 | 8 | 142.49 | 0.999363 | −4.94 % † |
| 8704 | 2 | (2,2,2) | 80 | 5 | 10 | **171.21** | 0.999997 | — |
| 8704 | 3 | (3,2,2) | 80 | 3 | 9 | **155.01** | 0.999422 | **−9.46 %** |
| 8704 | 4 | (4,2,2) | 96 | 3 | 12 | 177.24 | 0.999532 | +3.52 % † |
| 9216 | 2 | (2,2,2) | 80 | 5 | 10 | **171.91** | 0.999999 | — |
| 9216 | 3 | (3,2,2) | 80 | 3 | 9 | **156.32** | 0.999453 | **−9.06 %** |
| 9216 | 4 | (4,2,2) | 96 | 3 | 12 | 178.34 | 0.999555 | +3.74 % † |

† not like-for-like — `nst 96` vs `80`. Within-trial spread 0.9–2.3 %; the `bkv_p=2` control at
ctx 8704 reproduces Task 1's independent full-op fit to 1.3 % (171.21 vs 173.37).

`hlo_stats` confirms the same grid independently, at every one of the nine points
(`baseline-device-time-hlostats.j-79a410b8.json`): ctx 8704 TC total **170.579 → 148.166 µs**,
**−13.1 %**, with stage A **111.708 → 89.489** and stage B unchanged at 31.73 → 31.71.

### 3.3 The stage-A grid at a common width — the mechanism

| `nst` | ctx | `bkv_p` | blocks | pages | device µs | vs `p=2` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 8192 | 2 / 3 / 4 | 4 / 3 / 2 | 8 / 9 / 8 | 90.24 / 87.66 / **75.09** | — / −2.86 % / **−16.78 %** |
| 96 | 8704 | 2 / 3 / 4 | 5 / 3 / 3 | 10 / 9 / 12 | 109.87 / **87.47** / 108.49 | — / **−20.39 %** / −1.26 % |
| 96 | 9216 | 2 / 3 / 4 | 5 / 3 / 3 | 10 / 9 / 12 | 109.58 / **87.34** / 108.78 | — / **−20.30 %** / −0.73 % |
| 80 | 8192 | 2 / 3 / 4 | 4 / 3 / 2 | 8 / 9 / 8 | 90.28 / 87.47 / 75.38 | — / −3.11 % / −16.50 % |
| 80 | 8704 | 2 / 3 | 5 / 3 | 10 / 9 | 109.60 / 87.77 | — / −19.92 % |
| 80 | 9216 | 2 / 3 | 5 / 3 | 10 / 9 | 109.65 / 87.39 | — / −20.30 % |

All r² ≥ 0.999976.

The nine `nst = 96` points fit a two-term model with **ctx absent entirely**:

```
stage A decode (µs) = 6.24 + 7.41 × blocks + 6.65 × pages_fetched      r² = 0.9967
                      max residual 1.14 µs (1.3 %)
    blocks = ceil(P / bkv_p),  pages_fetched = blocks × bkv_p,  P = cdiv(ctx, 1024)
```
Neither term alone works (pages only r² = 0.674, blocks only r² = 0.428). **Stage A's decode cost
is set entirely by how many pages it fetches and how many grid steps it takes — not by the
sequence length.** That is the waste, stated as a formula: at ctx 8704 nine pages are needed and
`bkv_p = 2` fetches ten in five blocks.

### 3.4 Is it a `9 = 3×3` artifact? — yes, and here is the discriminator

`P = cdiv(ctx, 1024)` is the pages actually needed. `bkv_p` wins when it divides `P` exactly:

| ctx | `P` | `p=2` → blk × pg | `p=3` → blk × pg | `p=4` → blk × pg | winner |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8192 | 8 | 4 × 8 **exact** | 3 × 9 (+1) | 2 × 8 **exact** | `p=4`, then `p=3`, then `p=2` |
| 8704 | 9 | 5 × 10 (+1) | 3 × 9 **exact** | 3 × 12 (+3) | **`p=3` by 20 %** |
| 9216 | 9 | 5 × 10 (+1) | 3 × 9 **exact** | 3 × 12 (+3) | **`p=3` by 20 %** |

**The `9 = 3×3` hypothesis is confirmed.** `bkv_p = 3` wins precisely on the contexts that need
9 pages — `ctx ∈ (8192, 9216]` — where it is both exact *and* minimal in block count. At ctx 8192
its stage-A advantage collapses from 20.4 % to 2.9 %, and on the **full op** it inverts to
**+2.72 %**, a regression. The model above predicts a loss at any ctx where `p=2` is exact and
`p=3` is not: ctx 4096 (P=4) gives `p=2` → 2 × 4 against `p=3` → 2 × 6, a 50 % page overfetch.

Note `pages_per_seq` — which sets `num_sublanes_total` — is 9 at *every* ctx here, because it
comes from `max_model_len = 9216`, not from the sequence. The thing that varies is `num_bkv` and
what gets fetched. That distinction is why the earlier `9 = 3×3` framing needed testing rather
than asserting: the "9" that matters is `cdiv(ctx, 1024)`, not `pages_per_seq`.

### 3.5 Does the earlier −11.9 % survive?

**Directionally yes, quantitatively no.** `probes/P7-bkvp-and-spill.md` §A4.1 reported
`bkv_p = 2 → 3` at **−11.9 %** from a contended four-shard fan-out, ratios only. The clean
single-process absolute at the same shape is **−9.46 %** on the full-op device time (and −13.1 %
on the `hlo_stats` TensorCore total, which excludes the residual that the full-op chain includes).
The direction, the mechanism and the rough magnitude survive; the specific figure was 2.4 points
optimistic, which is what a contended measurement of a ratio would be expected to do.

### 3.6 What candidate 0 is worth, and what it is not

* At **ctx 8704**, the baseline of record's shape: **−9.46 % on the decode call**, 171.21 →
  155.01 µs device, one call-site parameter, no kernel written. **The incumbent is allowed to win
  and at this shape a differently-tuned incumbent does.**
* It is **not a general win**. It is a win on `ctx ∈ (8192, 9216]` and a ~2.7 % loss at ctx 8192.
  Whether it pays in production depends on the serving context distribution, which nothing here
  measures. A context-dependent `bkv_p` — choose the largest `p ≤ p_max` with
  `cdiv(P,p)·p == P` — is the obvious generalisation and is untested.
* **`bkv_p = 4` is the better answer at ctx 8192** (−16.8 % stage A, −4.9 % full op) and is neutral
  at 8704/9216 on stage A while *losing* 3.5 % on the full op to the wider score matrix. It is a
  real candidate at short contexts and is disqualified at long ones by `num_sublanes_total`.
* **The model extrapolates to something better than anything swept, and it was not measured.**
  `6.24 + 7.41·blk + 6.65·pg` at P=9 favours `p=9` → 1 block, 9 pages → **~67 µs modelled**, −39 %
  against `p=2`. `p ∈ {1,3,9}` are the exact divisors of 9. This is `modelled` extrapolation
  outside the swept range, VMEM may forbid it, and it is named here as the single cheapest
  follow-up rather than as a result.

---

## 4. The one methods disagreement, and its cause

Everything above agreed. Stage B did not: chain slope **336.48** against `hlo_stats` **431.15** at
prefill (+28.1 %), and 28.77 against 31.73 at decode (+10.3 %) — same sign, and far outside the
0.31 % run-to-run reproducibility of the chain slope. The brief's rule is that such a disagreement
*is* the result, so it was investigated rather than resolved by preference.

Three explanations were ruled out from the existing JSON before spending a job:

* **row lengths** — the isolated arm reports `eff_len_mean = 8704.5` at prefill, exactly the mean
  of `_effective_row_lengths` over `8193…9216`, and 8704.0 at decode. They already matched.
* **shape / partition** — `_pick_partition` returns `p = 1` in both settings, and the `hlo_stats`
  row name is character-identical: `sc_topk_direct_b1024_n10240_k2048`. Same kernel, same shape.
* **the method** — ruled out by construction below.

That leaves the **input data**. The isolated arm feeds
`bench_streamidx_topk.build_stage_b_scores`, N(0,1) with a `-inf` tail; its own docstring warns
that "the SparseCore radix select is data dependent" and that the distribution is held fixed for
comparability, "not because N(0,1) is what the indexer emits."

`probes/probe_stage_b_input.py`, job **`j-60140613-7496-4b63-9a89`**, holds the method fixed
(chain slope, both arms) and varies only the input: synthetic N(0,1) against stage A's **actual
output** for the same inputs, reshaped as `_common_path` reshapes it, with row lengths from the
op's own `_effective_row_lengths`.

| phase | arm | chain slope µs | r² | score distribution (finite entries) |
| --- | --- | ---: | ---: | --- |
| decode | synthetic | **28.81** | 0.999981 | mean 0.001, std 1.001, range [−4.73, 4.58] |
| decode | **real** | **35.39** | 0.999931 | mean 244.0, std 74.1, range [23.7, 698.8] |
| prefill | synthetic | **335.89** | 0.999997 | mean 0.00005, std 1.000, range [−5.12, 5.22] |
| prefill | **real** | **442.56** | 1.000000 | mean 244.1, std 72.8, range [9.28, 759.8] |

Row lengths, shapes, `p_partition` and the `-inf` count are identical between the two arms of each
pair (decode 557 056 finite / 98 304 `-inf`; prefill 8 913 408 / 1 572 352).

**Verdict: the disagreement is the input, not the method.**

* The synthetic arms reproduce the Task-1 controls to **0.18 %** (335.89 vs 336.48) and **0.15 %**
  (28.81 vs 28.77), so nothing about this probe's pipeline moved.
* The real arms land at **442.56 vs `hlo_stats`' 431.15 (+2.6 %)** and **35.39 vs 31.73 (+11.5 %)**.
  Prefill closes; decode overshoots, and the residual there is plausibly that the real arm runs
  one stage-A call where the op runs three, so the score matrix is not bit-identical to the one
  the op feeds itself.
* The mechanism is legible in the distributions. Real indexer scores are all positive, in
  [9, 760], with `std/mean ≈ 0.30` — a handful of distinct float exponents, and the top-2048
  threshold buried inside a narrow band. N(0,1) spans the sign bit with the threshold at ≈ +0.6.
  A radix select discriminates on high bits first; the real data needs more refinement passes for
  the same `k`.

**Consequences, and both are corrections to existing documents:**

1. **The in-situ `hlo_stats` figures are the ones that describe production stage B** — 31.73 µs
   decode, 431.15 µs prefill. They are used as such throughout §1 and §2.
2. **`stage-b-ksweep.md`'s 335.5 µs prefill stage B is stage B on synthetic data and understates
   the production cost by 22 %.** Its `k` sweep, `b` sweep, `n` sweep and `eff_len` sweep are all
   internally consistent and its *structural* conclusion is untouched — stage B is issue-bound in
   the length-bounded scan, not output-DMA-bound, and 85.4 % of the time is the scan. But its
   fitted cost model `t = 8.15 + waves × (0.702 + 0.0542 ns·n + 1.0197 ns·eff_len)` carries a
   **1.02 ns/element scan coefficient calibrated on N(0,1)**, and the production coefficient is
   about **1.37 ns/element**. The model's own "what it does NOT cover" list names the score
   distribution as an uncovered term and estimates it at "a few percent"; it is 28 %.
   Treatments (b) and (c) in §2 use the corrected scan term.

This also means the two methods, once fed the same workload, agree at **every** arm — see §5.

---

## 5. Did the two methods agree?

**Yes, everywhere, once §4's input mismatch was closed.**

| arm | chain slope | `hlo_stats` | Δ |
| --- | ---: | ---: | ---: |
| decode full op (Task-2 control) | 171.21 | 170.579 | **+0.4 %** |
| decode full op (Task-1 fit) | 173.37 | 170.579 | +1.6 % |
| decode stage A, single call | 109.65 | 109.890 | **−0.22 %** |
| decode stage B, real scores | 35.39 | 31.731 | +11.5 % |
| prefill full op | 750.14 | 749.538 | **+0.08 %** |
| prefill stage A, single call | 247.63 | 246.243 | **+0.56 %** |
| prefill stage B, real scores | 442.56 | 431.153 | **+2.6 %** |
| prefill stage B, synthetic scores | 336.48 | 431.153 | −22.0 % ✗ |

Six of seven like-for-like comparisons are inside 2.6 %; four are inside 0.6 %. The one remaining
outlier is decode stage B at +11.5 %, on a 31.7 µs arm where the isolated chain reconstructs the
score matrix from one stage-A call instead of three — a 3.7 µs absolute discrepancy, 2.1 % of the
decode call. It is named, not explained away.

The methods share no machinery: one is a host-clock regression across compiled programs of
different sizes, the other is per-instruction device self time read out of a profiler plane, and
they were run in different jaxlib versions in different steps of the same job. Agreement at 0.08 %
on the largest arm is the strongest evidence in this file.

---

## 6. Correctness

| | |
| --- | --- |
| job | **`j-79a410b8-46bc-4d06-862c`**, step `pytest`, **rc = 0** |
| command | `python -m pytest -q -p no:cacheprovider tests/kernels/test_dsv4_streamindex_topk.py` — **unmodified**, from the mapped checkout |
| test file md5, in-pod | `becf9f23ace9424a7d67cd3b34de9afd` |
| collection count | **20** (the stale-file tripwire was 13; 13 would have been a red flag, not a pass) |
| **verbatim result line** | **`20 passed, 14 warnings in 36.03s`** |

**What this does and does not cover, stated precisely.** The unmodified suite parametrizes
`bkv_p` over **{4, 8, 16, 32}**, always at `page_size × bkv_p = 128` (the scores kernel's DMA
contract) — `page_size` 4–32, sequences of 8–384. **It never runs `bkv_p ∈ {2, 3}`, and it never
runs `page_size = 1024`.** So it establishes the kernel's correctness across its `bkv_p` axis and
against a numpy reference, but it does not by itself certify the exact production configuration
either candidate 0 or the incumbent runs at.

The gap is closed at the production shape by a different check, run in the same process as the
timings (`bkvp_index_set_equality` in the chain JSON): the full op's output index set was compared
against `bkv_p = 2` at every one of the nine (ctx, `bkv_p`) points.

* `index_sets_match_bkvp2: true`, **64 / 64 rows**, at all nine points;
* `n_pad = 0`, `min_idx = 0`, `max_idx = ctx − 1` at each ctx — no padding leaked into the output;
* `chain_perturbation_is_zero: true`, so the timing chain did not alter what was computed.

Chained with the suite (`bkv_p = 2`'s kernel path is reference-checked at small shapes; `bkv_p = 3`
produces identical top-k index sets to `bkv_p = 2` at the production shape), this is good evidence
for candidate 0 — but it is **self-consistency against the incumbent, not a reference check at
production shape**, and it inherits whatever the incumbent gets wrong. Recommended status for
candidate 0: **provisional, correctness supported but not fully verified at the production
configuration.** The follow-up that would close it is a numpy top-k reference evaluated at
`prod_decode_b64_ctx8704`, which is cheap and was not run here.

---

## 7. What did not get measured

* **`bkv_p` at prefill.** Only the decode entry of the tuple was varied. The prefill entry is 2 at
  every point above.
* **Any ctx outside {8192, 8704, 9216}**, and any batch outside 64. The `blocks + pages` model
  predicts the shape of the `bkv_p` optimum elsewhere; nothing here tests it.
* **`bkv_p ∈ {1, 6, 9}`**, including the `p = 9` point the model likes best.
* **A first-principles SparseCore floor.** §2's whole spread comes from not having one. This is
  unchanged from `ROOFLINE-REVIEW.md`.
* **A reference correctness check at the production shape** — see §6.
* **Anything end-to-end.** No model benchmark was run and none of these numbers predicts one: the
  op's share of a real step is not measured here.
