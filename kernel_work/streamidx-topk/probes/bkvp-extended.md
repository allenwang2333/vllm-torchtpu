# `bkv_p` swept to 9 and past it, and candidate 0's correctness closed

Follow-up to `probes/baseline-device-time.md`. Measured on TPU7x, 4-chip pod
(`2x2x1`, 8 devices), image
`vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b`,
branch `annyan/streamidx-topk-opt`.

Jobs, all `JobStatus` green with every step `rc=0`:

| job | what | steps |
| --- | --- | --- |
| `j-6a35b078-a143-4f85-8228` | sections A, A2, B, C, D, E + 7 xprof traces | `env` `chain_le9` `chain_gt9` `j11_venv` `trace_le9` `trace_gt9`, all rc=0 |
| `j-ec3b2a64-dae6-481c-9166` | sections F (VMEM wall) and G (argmin, full op at short ctx) | `env` `fg`, both rc=0 |
| `j-26ac6fd7-f075-4a14-8cf1` | f64 oracle coverage extended to `p = 6, 7` | `env` `oracle`, both rc=0 |

**Kernel md5 read back on the pod in every job** — a green `JobStatus` is not
evidence the mapped code ran:

```
streamindex_topk.py     1e221f10cbf3573aa0d2ae78c3463532
sparsecore_topk.py      8bc9e25b8701a53a97704564b9fbc658
bench_streamidx_topk.py e8e492d13e59c4a1675cc7a98c5a0813
_stage_a.py             eea6a51bb40a2bbf7cf5b39571753b62
device TPU7x, 8 devices
```

Nothing in `src/` or `instruments/` was modified. No end-to-end model
benchmark was run and nothing here went to the experiment log.

Raw JSON: `probes/results/bkvp-extended-{le9,gt9,fg,oracle67}.shardx.json`,
`probes/results/bkvx_hlo_stats.json`. Probes:
`probes/probe_bkvp_extended.py` (sections A–G, chain slope + the f64 oracle),
`probes/probe_bkvx_trace.py` (xprof capture), `probes/bdt_hlo_stats.py`
(off-pod reader), driven by `probes/bkvx_shard{,2,3}.sh`.

---

## 0. The five answers, up front

1. **The exactness story does not survive.** `p = 5` (inexact, overfetches a
   page) ties `p = 3` (exact) at 88.72 vs 87.71 µs — 1.2 % apart. Divisibility
   buys nothing on its own. What governs is a two-regime cost in **blocks and
   pages**, §2.
2. **It is a function, not a constant.** The argmin is `p = P = cdiv(ctx,
   page_size)` at **all nine ctx buckets tested** (2048…9216), and a *fixed*
   `p = 9` costs **+38.0 % full-op device time at ctx 4096** and **+14.2 % at
   ctx 6144** while winning 14.6 % at ctx 8704. §4.
3. **It stops at `p = 32`, and the wall is the physical 63.94 MiB of
   TensorCore VMEM.** Raising `vmem_limit_bytes` does **not** move it: `p = 32`
   is rejected identically at 16, 32, 64, 100, 128 **and 256 MiB**. §3.
4. **Pipelining degrades but does not collapse.** The marginal cost of a page
   rises from 6.77 µs/page at 3 blocks to 7.40 at 2 and 8.79 at 1 — but the
   block saving still wins up to 9.75 pages, and at every production ctx the
   one-block arm is the fastest. §2.3.
5. **Candidate 0 is correct at every `bkv_p` from 1 to 9.** An f64 oracle says
   the emitted top-k index set is **byte-identical to the incumbent's** at all
   of them; the 295 oracle disagreements are shared with the incumbent and all
   sit under 0.51σ of the f32 noise floor. §5.
6. **Option A comes off the bench.** Candidate 0 lands at `p = P`, which is
   `9` at ctx 8704 — far above R1's "revisit A iff candidate 0 lands above
   `bkv_p ≈ 5`" threshold — and `p = 9` both fits (18 MiB against 63.94) and is
   correct. §6.

---

## 1. Method, and the independent confirmation

Unchanged from `baseline-device-time.md` §4: **chain slope**, R
dependency-linked calls in one `jit`, slope = per-call device time, intercept
(the host term) discarded. `r² ≥ 0.999` is a rejection threshold, not a
comment. Stage A is measured at **one common `num_sublanes_total` of 256** for
every `p`, so score-matrix width cannot be the confound. Section E re-checks
that at nst 80 and 128:

| `p` | nst 80 | nst 128 | nst 256 |
| --- | --- | --- | --- |
| 2 | 109.65 | 109.95 | 109.49 |
| 3 | 87.36 | 87.74 | 87.71 |
| 9 | 82.32 | 81.97 | 82.38 |

Spread ≤ 0.5 % across a 3.2× range of width. Width is not a confound.

### 1.1 `hlo_stats`, independently, with the partition check

Seven xprof traces of the **full op** at ctx 8704 (`j-6a35b078`), read off-pod
by `bdt_hlo_stats.py`. Per-instruction device self time; no host clock is
involved, so there is no dispatch floor at all.

```
arm       TC total     stage A   stage B   residual   partition   rows    SparseCore
dec_p1     194.389     135.360    31.784     27.245   +0.000000   42/42     30.972
dec_p2     170.724     111.775    31.738     27.211   +0.000000   42/42     30.944
dec_p3     148.252      89.440    31.750     27.062   -0.000000   42/42     30.926
dec_p5     149.569      90.820    31.713     27.036   -0.000000   42/42     30.907
dec_p9     143.387      84.279    31.730     27.378   -0.000000   42/42     30.920
dec_p10    151.970      93.019    31.719     27.232   +0.000000   42/42     30.944
dec_p16    206.234     144.762    32.270     29.201   +0.000000   42/42     31.388
```

* **Control.** `dec_p2` is R2's `nightly` arm and must reproduce its 170.474 µs
  TensorCore total. It comes back at **170.724 µs, +0.15 %**. The pipeline has
  not moved.
* **Partition check passes exactly** — `sum(buckets) − TC total = 0.000000` on
  all seven, with `rows_kept == rows_total == 42`, so nothing was lost to
  truncation.
* SparseCore (~30.9 µs) is reported separately and **not** added; the two
  planes run concurrently.

Agreement between the two methods on stage A (chain slope isolated at nst 256
vs the `hlo_stats` `StreamIdxTC` bucket inside the full op at nst 80):

| `p` | chain slope | `hlo_stats` | ratio |
| --- | --- | --- | --- |
| 1 | 133.52 | 135.360 | 1.014 |
| 2 | 109.49 | 111.775 | 1.021 |
| 3 | 87.71 | 89.440 | 1.020 |
| 5 | 88.72 | 90.820 | 1.024 |
| 9 | 82.38 | 84.279 | 1.023 |
| 10 | 91.04 | 93.019 | 1.022 |
| 16 | 143.36 | 144.762 | 1.010 |

A flat +2 % offset — the isolated harness omits the surrounding op — and an
**identical ranking**. `hlo_stats` puts `p = 9` at −24.6 % on stage A against
the incumbent; chain slope puts it at −24.8 %. The two methods agree.

On the full op the two differ by more, and the difference is itself
informative: chain slope 155.17 vs `hlo_stats` TC 148.25 at `p = 3`, but
194.43 vs 194.39 at `p = 1`. The gap is the **non-overlapped SparseCore
window** — as stage A gets faster, less of stage B's SparseCore work hides
behind it. Both methods still rank `p = 9` first (chain −14.6 %, `hlo_stats`
−16.0 %).

---

## 2. Section A — the sweep, and what actually governs it

Stage A, `prod_decode_b64_ctx8704`, common nst 256. Every `r² ≥ 0.99972`.
"modelled" is phase 1's `6.24 + 7.41·blocks + 6.65·pages`.

| `p` | blocks | pages | bkv buf | measured µs | r² | meas/modelled | vs `p=2` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9 | 9 | 2 MiB | 133.52 | 0.99999 | 1.006 | +21.9 % |
| **2** (incumbent) | 5 | 10 | 4 MiB | **109.49** | 1.00000 | 0.997 | — |
| 3 | 3 | 9 | 6 MiB | 87.71 | 0.99998 | 0.993 | −19.9 % |
| 4 | 3 | 12 | 8 MiB | 108.59 | 1.00000 | 1.003 | −0.8 % |
| 5 | 2 | 10 | 10 MiB | 88.72 | 1.00000 | 1.013 | −19.0 % |
| 6 | 2 | 12 | 12 MiB | 104.94 | 1.00000 | 1.040 | −4.2 % |
| 7 | 2 | 14 | 14 MiB | 121.16 | 1.00000 | 1.061 | +10.7 % |
| 8 | 2 | 16 | 16 MiB | 137.70 | 0.99999 | 1.080 | +25.8 % |
| **9** | 1 | 9 | 18 MiB | **82.38** | 1.00000 | 1.121 | **−24.8 %** |
| 10 | 1 | 10 | 20 MiB | 91.04 | 0.99996 | 1.136 | −16.9 % |
| 11 | 1 | 11 | 22 MiB | 99.49 | 1.00000 | 1.146 | −9.1 % |
| 12 | 1 | 12 | 24 MiB | 108.59 | 1.00000 | 1.162 | −0.8 % |
| 14 | 1 | 14 | 28 MiB | 125.71 | 1.00000 | 1.178 | +14.8 % |
| 16 | 1 | 16 | 32 MiB | 143.36 | 1.00000 | 1.194 | +30.9 % |
| 20 | 1 | 20 | 40 MiB | 177.19 | 1.00000 | 1.208 | +61.8 % |
| 24 | 1 | 24 | 48 MiB | 217.96 | 0.99972 | 1.258 | +99.1 % |
| 32 | 1 | 32 | 64 MiB | **rejected** | — | — | — |

`p = 9` is the measured minimum, at **82.38 µs, −24.8 % against the
incumbent**. Modelled was 73.5; the model is optimistic by 12 %, and §2.3
says why.

### 2.1 Prediction 2 was the falsifiable one, and exactness lost it

The exactness story predicts `p = 5` (2 blocks, **10** pages, overfetch)
should lose to `p = 3` (3 blocks, **9** pages, exact cover). Measured:

```
p = 3, exact   :  87.71 us
p = 5, inexact :  88.72 us     +1.2 %
```

They tie. One extra page (+6.65 modelled) almost exactly cancels one fewer
block (−7.41 modelled), and that trade is what is actually happening —
divisibility per se is not. The same pattern holds across §4's ctx sweep,
where inexact-but-fewer-blocks configurations repeatedly beat
exact-but-more-blocks ones (ctx 8192: `p = 3` exact-9-pages/3-blocks 87.62
loses to `p = 9` inexact-9-pages/1-block 82.11; and both lose to `p = 8`
exact-8-pages/1-block 73.98).

**The exactness story does not survive.** It made the right call at `p = 3`
for the wrong reason.

### 2.2 What replaces it: a two-regime cost in blocks and pages

Fitting all 31 stage-A arms from sections A, D and G1 (`blocks` from 1 to 9,
`pages` from 4 to 24, five ctx values):

```
2-term  t = -1.72 + 5.60*blocks + 8.51*pages                     r2 0.98505, max resid 10.9 %
3-term  t = -1.46 + 7.23*blocks + 7.69*pages + 0.94*pages*[blocks==1]
                                                                 r2 0.99731, max resid 10.4 %
```

Phase 1's `6.24 + 7.41·blk + 6.65·pg` was fitted only on `blocks ∈ {2,3,5}` and
is a good model *there*; it is 12 % optimistic in the one-block regime it was
never fitted on. Splitting by block count makes the mechanism visible:

```
blocks == 3 :  t =  27.08 + 6.77 * pages     (n = 4)
blocks == 2 :  t =  17.05 + 7.40 * pages     (n = 10)
blocks == 1 :  t =   3.49 + 8.79 * pages     (n = 14)
```

Fewer blocks: a much smaller fixed cost, a **larger** marginal cost per page.
That is the pipelining term, quantified.

### 2.3 Does pipelining collapse before VMEM does? No — it degrades

This was the more interesting failure mode to look for, and the answer is that
it is real, measurable, and not large enough to bite at any production shape.

**The measured curve does not turn up before the VMEM wall.** It falls to
`p = 9` and then rises monotonically, and the rise is *page count*, not a
pipelining cliff: every arm from `p = 9` to `p = 24` is one block, and the
marginal cost between consecutive one-block arms is a flat 8.4–9.1 µs/page
(8.67, 8.44, 9.10, 8.56, 8.83, 8.46, 10.19). Nothing discontinuous happens at
`num_bkv = 1`.

The page-count-matched pairs — same bytes moved, same width, blocks 2 vs 1 —
go the *right* way:

| ctx | pages | 2 blocks | 1 block | 1-block advantage |
| --- | --- | --- | --- | --- |
| 4096 | 4 | `p=2` 48.86 | `p=4` 40.59 | −16.9 % |
| 6144 | 6 | `p=3` 61.66 | `p=6` 57.03 | −7.5 % |
| 8192 | 8 | `p=4` 75.47 | `p=8` 73.98 | −2.0 % |

The advantage shrinks as pages grow, exactly as the two-regime fit says it
must. Setting the two lines equal:

```
3.49 + 8.79 n  ==  17.05 + 7.40 n     ->   n = 9.75 pages
```

**Below ~9.75 pages one block wins; above it, two.** The sweep confirms the
crossover independently: at 10 pages the two-block arm already wins
(`p = 5`, 2 blocks, 10 pages: 88.72 beats `p = 10`, 1 block, 10 pages: 91.04).

At `max_model_len = 9216` the largest page demand is 9, which is under the
crossover. So one block is the right regime for this deployment — but only
just, and this is the number to re-derive if `max_model_len` grows.

---

## 3. Section F — where it stops, and whether `vmem_limit_bytes` moves the wall

`vmem_limit_bytes` is a call-site static on `make_stage_a_fn`, so the scoped
limit is swept without touching `src/`. Compile-only (`lower().compile()` is
enough for Mosaic to accept or reject an allocation).

| `p` | bkv double buffer | min scoped limit that compiles | @100 MiB (production default) | @256 MiB |
| --- | --- | --- | --- | --- |
| 6 | 12 MiB | 16 MiB | ✅ | ✅ |
| 8 | 16 MiB | 32 MiB | ✅ | ✅ |
| **9** | **18 MiB** | **32 MiB** | ✅ | ✅ |
| 10 | 20 MiB | 32 MiB | ✅ | ✅ |
| 12 | 24 MiB | 32 MiB | ✅ | ✅ |
| 16 | 32 MiB | 64 MiB | ✅ | ✅ |
| 24 | 48 MiB | 64 MiB | ✅ | ✅ |
| 32 | 64 MiB | **none** | ❌ | ❌ |
| 48 | 96 MiB | **none** | ❌ | ❌ |

**It stops at `p = 32`, and raising `vmem_limit_bytes` does not move the
wall.** `p = 32` is rejected at all six limits with a byte-identical message:

```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in
memory space vmem. Used 85.00M of 63.94M vmem. Exceeded vmem capacity by 21.06M.
Program vmem requirement 85.00M:  scoped  85.00M
  1. Size: 64.00M  Operator: ... /StreamIdxTC-d-bq_1-bkvp_32/pallas_call
```

`63.94 M` is the **physical** TensorCore VMEM (`vmem_capacity_bytes =
67108864`), not a scoped budget. Past 32 the rejection changes mechanism —
`p ∈ {40, 48, 64}` fail in Mosaic's allocator on a *single* allocation
exceeding the physical size:

```
p=40  Allocation (size=83886080)  would exceed memory (size=67108864)
p=48  Allocation (size=100663296) would exceed memory (size=67108864)
p=64  Allocation (size=134217728) would exceed memory (size=67108864)
```

Two corrections to the premise this question was asked under:

* **Nothing here runs at a 32 MiB scoped default.** `streamindex_topk.py:28`
  already sets `DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024`, and both entry
  points take it as a call-site static.
* **`p = 9` needs only 32 MiB** and clears the production default with 3× of
  headroom. It is nowhere near any wall, scoped or physical.

**A raised limit does not change the speed of a `p` that already fits.**
Timed, same harness:

| `p` | @100 MiB | @256 MiB | Δ |
| --- | --- | --- | --- |
| 9 | 82.29 | 82.35 | +0.07 % |
| 12 | 108.69 | 108.09 | −0.55 % |
| 16 | 143.34 | 143.30 | −0.03 % |

All inside run-to-run noise (`p = 9` was independently measured at 82.38,
82.33, 82.34, 82.11, 82.30, 82.25, 82.00, 82.11, 81.90 across nine arms in
three jobs — a 0.6 % spread). Mosaic is not converting extra scoped headroom
into more pipelining, so "fits" and "fits well" are the same question here.

---

## 4. Sections C, D, G — constant, or function?

### 4.1 The argmin, bracketed rather than inferred (§G1)

Stage A, common nst 256. At each ctx both `P−1` and `P+1` were run, so the
minimum is bracketed. `*` marks `p = P = cdiv(ctx, 1024)`.

| ctx | `P` | measured µs | argmin | best vs incumbent `p=2` |
| --- | --- | --- | --- | --- |
| 2048 | 2 | p1 35.46, **p2 25.93\***, p3 33.26, p9 82.08 | `p=2` = P | — |
| 3072 | 3 | p2 48.59, **p3 33.31\***, p4 40.75, p9 82.35 | `p=3` = P | −31.5 % |
| 4096 | 4 | p2 48.68, p3 61.92, **p4 40.90\***, p5 48.89, p9 82.11 | `p=4` = P | −16.0 % |
| 5120 | 5 | p2 68.12, p4 74.97, **p5 48.99\***, p6 57.70, p9 82.30 | `p=5` = P | −28.1 % |
| 6144 | 6 | p2 67.90, p5 88.94, **p6 57.43\***, p7 65.65, p9 82.25 | `p=6` = P | −15.4 % |
| 7168 | 7 | p2 90.33, p6 105.09, **p7 65.41\***, p8 73.95, p9 82.00 | `p=7` = P | −27.6 % |
| 8192 | 8 | p2 90.05, p7 121.23, **p8 73.98\***, p9 82.11 | `p=8` = P | −17.8 % |
| 8704 | 9 | §A: **p9 82.38\*** is the global minimum over p ∈ 1…24 | `p=9` = P | −24.8 % |
| 9216 | 9 | §C full op: **p9** is the minimum | `p=9` = P | — |

**`argmin p = P` at nine buckets out of nine.** `p = 8` at ctx 8192 was the
one untested prediction; it came in at **73.98 µs** against the 3-term model's
74.80 (−1.1 %) and phase 1's 2-term 66.85 (which was 11 % optimistic).

The one-block arms depend only on page count, not on ctx, which is what makes
this a rule rather than nine coincidences:

```
1 block, pages -> us :  2 -> 25.93   3 -> 33.3   4 -> 40.8   5 -> 48.9
                        6 -> 57.4    7 -> 65.5   8 -> 74.0   9 -> 82.2
t ~ 9.6 + 8.07 * pages,  marginal 7.4-8.5 us/page, no discontinuity
```

`p = P` is the configuration that fetches exactly the pages the sequence needs
in exactly one block — minimum overfetch *and* minimum block count at once.
It is optimal as long as `P` stays under the 9.75-page crossover of §2.3,
which at `max_model_len = 9216` it always does.

### 4.2 The full op at the production contexts (§C)

Chain slope, `r² ≥ 0.99942`. `lfl` = like-for-like: `num_sublanes_total = 80`,
the same as the incumbent, so no width confound.

| ctx | p=1 | **p=2** | p=3 | p=5 | **p=9** | p=10 | p=4 (nst 96) | p=8 (nst 128) | p=16 (nst 128) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 181.65 | **149.81** | 153.52 | 153.70 | **145.14 (−3.1 %)** | 154.30 | — | — | — |
| 8704 | 194.43 | **170.89** | 155.17 (−9.2 %) | 154.41 (−9.6 %) | **145.95 (−14.6 %)** | 155.77 | 176.65 † | 204.80 † | 212.04 † |
| 9216 | 195.45 | **171.49** | 156.31 (−8.9 %) | 156.34 (−8.8 %) | **147.68 (−13.9 %)** | 156.48 | — | — | — |

† width-confounded, listed for completeness only.

`p = 9` wins at all three production contexts, including ctx 8192 where
`p = 3` — phase 1's candidate — *regresses* by 2.5 %.

### 4.3 The decisive measurement: a fixed `p = 9` on a short sequence (§G2)

`num_kv_pages_per_block` is a **compile-time static**; `num_bkv` is computed at
runtime from `jnp.max(seq_lens)` (`streamindex_topk.py:275-276`). So a constant
`p = 9` also runs on short sequences, and there it fetches 9 pages for a
sequence that needs 4. Full op, `p ∈ {2, 9}` only — both give
`num_sublanes_total = 80`, so this comparison has no width confound:

| ctx | `p = 2` (incumbent) | `p = 9` (fixed constant) | Δ |
| --- | --- | --- | --- |
| 2048 | 10.92 (r² 0.99996) | 21.28 (**r² 0.92448 — rejected**) | not reportable |
| 4096 | 98.96 | 136.54 | **+38.0 %** |
| 6144 | 122.66 | 140.07 | **+14.2 %** |
| 8192 | 149.29 | 145.00 | −2.9 % |
| 8704 | 170.89 (§C) | 145.95 (§C) | −14.6 % |

The ctx 2048 pair is **not reportable**: `p = 9`'s fit is `r² = 0.9245`, far
under the 0.999 threshold, so it is rejected rather than quoted. (The
incumbent's 10.92 µs there is real but reflects the early-exit path —
`k = 2048 ≥ ctx`, so there is nothing to select.) The ctx 8192 arm reproduces
§C's independent measurement (149.81 / 145.14) to within 0.4 %.

### 4.4 So: a function, not a constant

**A fixed `bkv_p` is the wrong shape of change.** The evidence is §4.3: a
constant `p = 9` buys 14.6 % at ctx 8704 and pays 38.0 % at ctx 4096, on the
same op with no width confound. Any workload with a mix of context lengths
loses on that trade unless it is almost entirely long.

The right change is **`p = cdiv(ctx_bucket, page_size)`** — §4.1's rule, which
held at nine buckets out of nine and is explained by a fitted two-regime cost
model rather than only by the data it was fitted on.

Two things that constrain how that function can be written, and they are the
part worth arguing about before anyone codes it:

* **`bkv_p` is a compile-time static** (`static_argnames` at
  `streamindex_topk.py:514`). "A function of `cdiv(ctx, page_size)`" therefore
  means **compiling one variant per context bucket and dispatching on the
  runtime max sequence length** — it cannot be read out of `seq_lens` inside
  the existing kernel. That is a real cost: compile time, cache footprint, and
  a dispatch decision that has to happen before the call.
* **The cheap 90 % of it is already a constant.** `p = cdiv(max_model_len,
  page_size)` is a single value known at engine start (9 here, from
  `max_model_len = 9216`) and captures the entire win at long context, where
  this op is expensive enough to matter. The full function only adds value on
  short sequences — where §4.3 shows the *absolute* stake is smaller
  (`p = 2` costs 98.96 µs at ctx 4096 against 149.29 at ctx 8192).

My reading: ship `p = cdiv(max_model_len, page_size)` first if the served
traffic is long-context, because it is a one-line change with a measured
−14.6 % full-op win at ctx 8704 and −3.1 % at ctx 8192; treat per-bucket
compilation as a second, separately-justified change whose value depends
entirely on the context-length distribution of the traffic, which this
exercise has not measured. But the *answer to the question asked* is
unambiguous: **the optimum is a function of the context bucket, and a fixed
`bkv_p` is measurably wrong at short context.**

---

## 5. Section B — candidate 0's correctness gap, closed

An **f64 numpy top-k oracle** at `prod_decode_b64_ctx8704`. K is dequantized
from the `(total_pages, 256, 4, 256)` uint8 cache by viewing
`cache[pg].reshape(1024,256)[:, :128]` as `ml_dtypes.float8_e4m3fn` (scale
bytes asserted `== 127`, i.e. `2^0 = 1.0`), scores recomputed in float64 as
`(max(q·Kᵀ, 0) * w).sum(heads)` with the kernel's own mask, and the top-`k`
taken by a stable `argsort`.

```
rows                  64, all with row length 8704
oracle score range    23.658763 .. 700.380066
k-th place margin     min 9.3699e-05, median 0.0169552
sigma  = max |stage-A f32 - oracle f64|  =  2.2408128   (mean 0.2663173,
                                                         0.32 % of score_max)
```

### 5.1 Every `bkv_p` produces the identical index set

Eleven values of `bkv_p`, spanning `num_bkv` from 9 down to 1 and both exact
and inexact page covers. Every row is identical:

| `bkv_p` | min recall | rows with a miss | missed indices | worst gap | unique to this `p` | job |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | `j-6a35b078` / `j-26ac6fd7` |
| 2 (incumbent) | 0.995605 | 64/64 | 295 | 1.1424 | — | both |
| 3 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | both |
| 4 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | both |
| 5 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | both |
| 6 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | `j-26ac6fd7` |
| 7 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | `j-26ac6fd7` |
| 8 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | both |
| **9** | 0.995605 | 64/64 | 295 | 1.1424 | **0** | both |
| 10 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | `j-6a35b078` |
| 16 | 0.995605 | 64/64 | 295 | 1.1424 | **0** | `j-6a35b078` |

`identical_to_incumbent: true` at every one. Not "similar recall" — the same
295 positions, in the same 64 rows. `emitted_indices_past_row_length = 0`
everywhere (checked because `num_positions` grows with `p` while the oracle is
only 8704 wide). `sigma` reproduced bit-for-bit across the two jobs
(2.2408127784729004), so the two runs are directly comparable.

**Candidate 0 introduces zero new disagreement.** Whatever the 295 are, they
are not `bkv_p`'s doing, and this spans `num_bkv` from 9 down to 1 and both
exact and inexact covers.

### 5.2 The 295 are f32 near-ties, and they are the incumbent's

This is the warning that was worth heeding — the oracle disagrees, and the
disagreement is pre-existing:

* Every disagreement is also the **incumbent's**:
  `n_also_missed_by_incumbent_bkvp2 = 295` of 295.
* **Worst gap 1.1424 against sigma 2.2408** — `gap/sigma = 0.51`. No
  disagreement anywhere in the 295 exceeds 0.51σ. Example record:
  ```
  row 0: missed 2744 (s=243.2194082736969), took 6145 (s=242.5889821052551),
         gap 0.6304, gap/sigma 0.2813
  ```
* The **k-th place margin** — the score gap between rank 2048 and rank 2049 in
  the f64 oracle — has median **0.0170**, two orders of magnitude *below*
  sigma. At k = 2048 out of 8704 the boundary is genuinely degenerate: an f32
  score matrix cannot resolve it, and no implementation that computes in f32
  will.

295 of 64 × 2048 = 131,072 emitted indices is 0.225 %, giving min recall
0.995605. This is the same phenomenon Option C's run hit as
`min_recall 0.996094` on the **incumbent**, and the same conclusion applies:
it is a property of doing this selection in f32 at a degenerate boundary, not
a defect introduced by any `bkv_p`.

### 5.3 The gate-2 recall criterion, and what it should be

`min_recall 0.9956 < 0.999` fails a literal reading of the 0.999 criterion —
**on the incumbent**, at every `bkv_p` including 2. A criterion the shipping
code fails is measuring the wrong thing. The discriminating test is the one
run here: **is the emitted index set identical to the incumbent's, and is
every oracle disagreement inside the f32 noise floor?** Both are yes, with
margin.

### 5.4 Coverage, stated honestly

The oracle covers **every `bkv_p` from 1 to 9**, plus 10 and 16. That is the
whole range §4.1's `p = P` rule can select at `max_model_len = 9216`, so no
value this report recommends is unchecked. `j-6a35b078` ran
`{1,2,3,4,5,8,9,10,16}` and `j-26ac6fd7` added `{6,7}`, which §4.1 promoted
from "swept for timing" to "would actually be deployed" (they are `P` at
ctx 6144 and 7168) after the first job had already run.

`p ∈ {11, 12, 14, 20, 24}` were timed but never oracle-checked. That gap is
left open deliberately: they lose on time (§2) and, being above
`pages_per_seq = 9`, they walk the block table past the end of the sequence's
row, so they are not deployable at this geometry regardless of what the oracle
would say.

**Verdict: `provisional`, correctness-checked.** Candidate 0 has a named
correctness job (`j-6a35b078`, the f64 oracle above, plus `j-79a410b8`'s
20/20 on `tests/kernels/test_dsv4_streamindex_topk.py` against these exact
md5s). It is a microbenchmark result, not an end-to-end one, so it does not
reach `accepted` on this evidence alone.

---

## 6. Option A — explicitly, so the inference is not left to the reader

**Option A comes off the bench.**

R1's rule was "revisit A iff candidate 0 lands above `bkv_p ≈ 5`". Candidate 0
lands at **`p = P`, which is 9 at ctx 8704 and 8 at ctx 8192** — above the
threshold at every production context, not marginally.

R1's DMA-charge-ratio saving as a function of concurrent page DMAs (in-flight
= `seq_batch_size(4) × p`), from jobs `j-b0e33be2-7d67-41da-b724` and
`j-3e588eaa-75e8-4ddb-9737`:

| `bkv_p` | in flight | saving from halving the K record |
| --- | --- | --- |
| 2 | 8 | 21.0 % |
| 3 | 12 | 25.4 % |
| 4 | 16 | 32.0 % |
| 6 | 24 | 42.7 % (incumbent at 3.335e12 B/s = 99.6 % of a contiguous read) |
| **9** | **36** | **above the top of R1's measured table** |

Candidate 0's operating point is 36 concurrent DMAs, past where R1 measured.
On R1's own monotonically rising series the saving there is **at least** the
42.7 % measured at 24 in flight — but that specific number is an
extrapolation of R1's data and is **not** something this exercise measured.
What this exercise establishes is the precondition: candidate 0 puts the
kernel at 36 in flight, and it does so at a configuration that is both
buildable and correct.

Buildable and correct, concretely:

* **`p = 9` fits with room.** 18 MiB bkv double buffer, compiles at a 32 MiB
  scoped limit, production default is 100 MiB, physical ceiling 63.94 MiB
  (§3). It is nowhere near the wall, so halving the K record is not competing
  with candidate 0 for VMEM — it would free 9 MiB at `p = 9`.
* **`p = 9` is correct.** Byte-identical emitted index set to the incumbent,
  every oracle disagreement under 0.51σ (§5).
* **Pipelining does not cap A's revival.** §2.3: at `num_bkv = 1` the marginal
  page cost is 8.79 µs rather than 6.77, but there is no cliff and the
  one-block regime is still the fastest at every production ctx. A's saving is
  on the DMA charge, which is the term that *grows* as blocks shrink — the two
  changes are complementary, and the regime candidate 0 selects is the one
  where A's lever is largest.

The one caveat worth carrying: A and candidate 0 have not been measured
**together**, and §2.3's crossover means their interaction is not simply
additive — cheaper pages move the 9.75-page crossover outward, which would
only reinforce `p = P`, but that is modelled, not measured.

---

## 7. What is `modelled` and what is measured

Everything in §2's tables, §3, §4 and §5 is **measured** on TPU7x in the named
jobs, by chain slope with `hlo_stats` confirming independently on the arms
that were traced.

**Modelled**, and labelled as such wherever quoted: phase 1's
`6.24 + 7.41·blk + 6.65·pg`; §2.2's 2-term and 3-term refits and the
per-block-count page rates; §2.3's 9.75-page crossover; §6's extrapolation of
R1's DMA saving to 36 in flight. None of these is a benchmark result and none
of them went to the experiment log.

**Corrected from phase 1.** `baseline-device-time.md` §3.6 first read "~67 µs
modelled, −39 %" for `p = 9`, which dropped the model's 6.24 µs intercept; the
corrected modelled figure is 73.5 µs / −32.9 %, and it now carries a dated
note. The **measured** figure is 82.38 µs / −24.8 % (§2) — the model was
optimistic by 12 %, for the reason §2.2 gives.
