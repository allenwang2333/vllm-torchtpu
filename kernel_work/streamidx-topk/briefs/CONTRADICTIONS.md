# Live disagreements between the area sheets

Written by the step-2 coordinator after the area briefs returned. The eight sheets in this
directory were produced independently, by agents who could not see each other's work, each reading
`report.md` plus one reference file plus the source. That independence is the point — where they
agree, the agreement is evidence; where they disagree, the disagreement is a probe.

**This file does not resolve anything by authority.** It states each conflict, says what is
actually in dispute, and names what would settle it. Where a conflict is only apparent — the two
sheets pricing different things, or the same thing at different phases — it says so and gives the
synthesis, because a designer who reads both sheets and sees a contradiction that is not there will
discard a good finding.

Vocabulary is the sheets': **stage A** is the three TensorCore Pallas scoring calls; **stage B** is
`sparsecore_topk`. **Nightly** (DP=16, 64 seqs, `max_model_len` giving `num_positions=10240`) is the
baseline of record; **AgentX** (`max_model_len` 262144, b=16) is not.

---

## 1. Is E1 — bounding stage B's HBM read by `row_lengths` — a large lever or a small one?

**Not a real contradiction. It is a decode/prefill split, and both sheets are right about their own
phase.** This is the most likely thing in the whole set to be misread, so it goes first.

Four sheets (areas 2, 5, 6, 7) independently established the *mechanism*, and this is the
best-evidenced fact in step 2: every scan loop in `_topk_kernel` is bounded by
`i_hi = min(slice_len, eff_len)`, but the DMA at `sparsecore_topk.py:160` fills `resident_vmem` with
a static `slice_len = align_to(n, 16)` unconditionally. Compute tracks the real sequence length; the
HBM read tracks the allocated width.

They then disagree about what it is worth:

| sheet | claim |
| --- | --- |
| area 2, `## Correction after §6.3` item 3 | **Demoted.** §6.3 measures a 16× width change at 1.16× on stage B at nightly decode — elasticity ≈0.05. "A small win, not the lever finding (1) made it look like. I would not spend a design on it." |
| area 6, E1 | **First-rank.** Exact, no accuracy gate, ~15% of stage B's transfer at nightly and ~97% at AgentX. |
| area 7, C5 | **"The largest one"** of its level-mismatch findings. |
| area 3, A13 | **Not a lever at all** — stage B demands ~0.05 TB/s, so its DMA is ≲2-5% of a compute-bound kernel. Explicitly tells the designer not to chase it. |

**RESOLVED — and the resolution kills E1 at the baseline of record.** Area 3, asked to reconcile,
redid the arithmetic and conceded that its A13 bound was too tight (≲7% at decode, ≲15% at prefill,
not ≲2–5%) — and then made the observation that settles the whole question, which none of the other
four sheets had made:

**The lever is not the whole DMA. It is only the difference between `slice_len` and `eff_len`** —
that is the only part a guard reclaims. At nightly, `slice_len` is 10240 and `eff_len` ≈ 8704, a
ratio of **1.18**. So the reclaimable share is 15% of a term that is itself ≤15%:

| phase | unconditional DMA | DMA share of stage B | **reclaimable by E1** | as % of the call |
| --- | --- | --- | --- | --- |
| nightly decode | 2.62 MB | 6.9% | 0.7 µs | **0.3%** |
| nightly prefill | 41.9 MB | 15.3% | 11 µs | **1.3%** |

**Both are under, or barely at, §6.3's own 0.8–2.5% noise floor.** That is the decisive argument and
it is stronger than the percentage: a step-3 variant carrying only this change **could not be shown
to have worked** with the instrument we have.

**Where I had it wrong.** An earlier draft of this file argued E1 was the leading explanation for
§6.3's prefill ratios. It is not, and the error is worth naming because it is easy to repeat:
**§6.3's 2.88× / 3.41× measure the penalty of running at a 16× wider buffer, not the penalty the
nightly configuration actually pays.** `slice_len` and `eff_len` diverge by `max_model_len /
context`, which at nightly is 1.18 and at AgentX is far larger. The ratio measured in a width sweep
is not the ratio the baseline is carrying.

**So the four-way convergence identified a real defect that does not pay here.** All five sheets are
correct about the mechanism; area 2 was right to demote it; area 6's ~97% is an AgentX figure. Two
qualifications so this kill is not over-read:

- **It is a kill at the nightly width only.** The defect is 100% of §6.3's prefill ratios and would
  be first-order at `max_model_len` 65536 or AgentX's 262144. It is a one-line change, so it is
  worth *recording* as a long-context lever rather than discarding.
- **The rate is `modelled`** (0.57 TB/s, fitted from two §6.3 prefill rows, not measured on
  SparseCore). But the conclusion is insensitive: a 3× better true rate moves decode to 0.1%, further
  in the same direction. There is no plausible rate at which this becomes worth taking at nightly.

**Prefill's 2.88× / 3.41× are explained, and by this same mechanism — which does not resurrect the
lever.** Area 5 built the model the source forces: at `p=1` exactly one term is width-dependent, so
`t_stageB = ceil(b/32) × (a·n + c·eff_len)`. A joint fit of two coefficients to four measurements
gives max residual **7.5%** across a 4× range in row count and a 16× range in width:

| arm | n | measured | model | residual |
| --- | ---: | ---: | ---: | ---: |
| `prefill_q1024_ctx4096`, mml 4096 | 4096 | 245 µs | 263 µs | +7.5% |
| `prefill_q1024_ctx4096`, mml 65536 | 65536 | 706 µs | 703 µs | −0.4% |
| `prefill_q4096_ctx4096`, mml 4096 | 4096 | 695 µs | 652 µs | −6.2% |
| `prefill_q4096_ctx4096`, mml 65536 | 65536 | 2373 µs | 2413 µs | +1.7% |

Two things lift this above a curve fit: `a` is stable to 9% when fitted independently on each pair,
and `c` = 2.04 ns/element lands within **19% of the 1.72–1.73 ns/element derived in area 5's §C.7
from a different job, a different context length and a different route.**

**And the two dissenting sheets turn out to agree numerically to two significant figures.** Applying
area 5's fitted `a` to the nightly widths — where `slice_len` = 10240 against `eff_len` ≈ 8704 —
gives the reclaimable term directly:

| phase | `waves × a × (slice_len − eff_len)` | area 5 (fitted model) | area 3 (0.57 TB/s route) |
| --- | --- | --- | --- |
| decode | 2 × 0.224 ns × 1536 | **0.69 µs = 0.31%** | 0.7 µs = 0.3% |
| prefill | 32 × 0.224 ns × 1536 | **11.0 µs = 1.28%** | 11 µs = 1.3% |

Area 3 reached this from a fitted bandwidth; area 5 from a two-parameter joint fit on four
measurements. **The kill is now derived twice, independently.** The mechanism fully explains the
width sweep *and* is worth under the noise floor at the width we actually run — those are consistent
statements, and the second is the one that governs the design.

Area 5 names what its model does not do: `c` should not depend on `q` and does (1.87 vs 2.23
ns/elem), so cross-prediction between the two prefill shapes is off by −14%/+18%. Its candidate
explanation — a wave's duration is set by its longest row, not its mean — is **not established**, and
is flagged rather than smoothed over. It does not affect the reclaimable-term arithmetic above, which
depends only on `a`.

---

## 2. Does stage A have headroom, or is it already near its achievable floor?

**A real contradiction, and it is the one that bears on the roofline gate itself.**

- `report.md` §4 models the decode floor at 47.5 µs and puts the incumbent at 21.5% of it — deep in
  the "structural, a kernel usually pays" band.
- **Area 3, A3 and D2**, argues the floor is wrong because it charges every byte at a flat rate.
  `_fetch_bkv` emits one DMA per (sequence, page) of exactly **262 144 B**, and pages are
  page-table-indirect, so the granule is structurally pinned at 256 KiB and cannot be enlarged. The
  block-size curve in `03-pipelining.md` puts a 256 KiB transfer at 25–43% of ceiling. On that
  reading stage A is at **70–100% of its achievable floor** and the DMA schedule is not the target
  at all.
- Area 3 also corrects the denominator: `report.md` §4 lists 2.97e12 B/s as measured in its own
  rates table and then models at the 3.70e12 spec rate. At the measured rate the decode floor is
  **59.2 µs and the achieved fraction 26.8%**, not 21.5%. That correction is unambiguous and should
  be applied regardless of how the rest resolves; it moves every downstream ratio by 1.25× and
  changes no verdict.

**What is actually in dispute** is not the byte count but whether a flat bandwidth is a legitimate
denominator for a kernel whose granule is fixed at 256 KiB.

**Why this is not yet decided — and area 3 has withdrawn half its own evidence.** Asked to make the
claim falsifiable, area 3 conceded the circularity outright: *"the circularity the coordinator
flagged is real, and I overstated the sheet's evidence."* The "two independent routes" are not
independent. Route 2 divides measured stage A by 80 steps to get "1.13–1.57 TB/s achieved" — but that
division only yields a *bandwidth* if the step time is DMA time, which is the conclusion. If each
step is instead compute-bound at 1.36–1.89 µs — which requires the VPU to be 3–4× slower than the
`assumed` 1.72e12 ops/s, a rate §7 explicitly failed to measure — route 2 produces the identical
number and means nothing. It is **a consistency check on route 1, not corroboration of it.** The
claim rests on one route plus a non-contradiction.

**What settles it, with the threshold named.** `/xprof-llo` on `StreamIdxTC-d` at the nightly decode
shape, captured through `/run-command-in-cdk` with the kernel-trace `LIBTPU_INIT_ARGS` an ordinary
trace does not set. The discriminating quantity is the fraction of kernel cycles in the DMA-semaphore
wait at `streamindex_topk.py:398` (`wait_fetch_bkv`) — semaphore-wait/DMA-stall bundles versus VLIW
bundles issuing MXU or VPU ops:

- **DMA-stall ≥ 50%** ⇒ confirms. Stage A is near its achievable floor; the schedule is not the
  target and only byte-removal pays.
- **DMA-stall ≤ 15%** ⇒ refutes. The transfers are hidden, stage A's 109–151 µs is compute, and the
  256 KiB story is wrong.
- Between 15% and 50%, report the split and budget both terms.

**This one capture is the highest-value probe in the set, because it discharges two questions at
once.** If it refutes A3/D2, then the time is compute — and modelled compute of 0.49 µs/step against
an observed ~1.4–1.9 µs/step means the `assumed` VPU rate is off by 3–4×, which is also the open
question §5 below is about and which §7's void probe failed to answer. One ~20-minute job on a 4-chip
pod, not an end-to-end benchmark. Per `CLAUDE.md`, nothing from the dump-side tools is a result and
none of it goes to the experiment log.

Area 3 also proposes a cheaper crude discriminator worth running first: sweep `bkv_p ∈ {1, 2, 4}` at
fixed decode shape. See its sheet for what each outcome implies.

**Note what does *not* change either way.** Byte-*removal* levers (the 129-of-256 padding, the 15%
bkv rounding, the within-group length spread) win under both readings, because they reduce the
number of 256 KiB transfers rather than trying to make each one faster. Area 3 says this explicitly
in G3 and it is the right default: **prefer byte-removal over schedule-tuning at equal modelled
payoff, until this is settled.**

---

## 3. Is stage B's DMA 2–5% of stage B, or is it the mechanism?

> **RESOLVED — see §1.** Area 3 widened its own bound to ≲7% (decode) / ≲15% (prefill) and showed
> the reclaimable part is smaller still. The `k`-sweep proposed below is **no longer needed to
> settle this**; it remains worth running only if a design targets stage B's algorithm, where
> issue-bound versus output-DMA-bound changes which rewrite pays. The original conflict and the
> reasoning are kept below because the second bullet — the fitted dispatch subtraction — is a
> methodological caution that still applies to every figure derived from area 5's ≈1.7 ns/element.

**A real contradiction at the same phase, and the two cannot both be right.**

- **Area 3, A13:** stage B moves 2.62 MB in + 0.52 MB out against 67 µs ⇒ ~0.05 TB/s demanded ⇒ the
  DMA is ≲2–5% of a compute-bound kernel. Recorded as a deliberate negative result "so the designer
  does not chase it."
- **Areas 5, 6, 7:** the unconditional full-width DMA is the mechanism by which `max_model_len`
  drives stage B at byte-identical real work. Area 5 adds that the transfer is *blocking*
  (`cp.start(); cp.wait()`, no double-buffering), so it is a stall rather than overlapped work.

**These are reconcilable in principle** — a transfer can be a small fraction of total bytes-time and
still be a serialization point that gates everything behind it — but not at the numbers given. If
the DMA is 2–5% of stage B, bounding it cannot produce §6.3's prefill ratios.

Two ways one of them is wrong, and the designer needs to know which:

1. Area 3's denominator is the *nightly decode* figure. At prefill the read is 41.9 MB, not 2.62 MB
   — 16× larger — and A13 may simply not have been computed at the phase where the effect lives.
2. Area 5's ~1.7 ns/element rate is derived by subtracting a ~35 µs host-dispatch term from a
   best-of-N host measurement. `report.md` §6.2 **retracts** a dispatch add-back of that shape as
   fitting. If the subtraction is wrong the device/DMA split moves.

**What settles it:** the same `k`-sweep area 5 already proposes — hold `b`, `n` and `row_lengths`
fixed and sweep `k` alone (2048 → 1024 → 512). Output-DMA-bound ⇒ time falls roughly linearly in
`k`; vector-issue-bound ⇒ it barely moves. One shard, seconds. Both sheets have been asked to
address the other.

---

## 4. Does the indexer K cache have one consumer or two?

**A real contradiction, with a runnable falsifier already written.**

`report.md` §7 and §8-finding-4 record the 49.6% K-cache padding as a *two-consumer* packing change,
on the grounds that the indexer's `cache_kv` is the same buffer MLA v2 attention reads. **Area 4, D1**
says that is wrong at this configuration: each layer name gets its own tensor, and GLM's MLA
attention layers take the `use_sparse` branch and receive a tuple of split (nope, rope) uint8
caches, never the 4-D array. What is shared is the shape *function*, not the buffer.

This matters more than any other item here: it is the objection that downgraded the padding from
"reclaimable" to "a two-consumer change", and the padding is 83.2 MB of 167.8 MB on the term that is
95% of decode HBM traffic.

**What settles it:** `briefs/area4-layout-precision.md` § "D1's falsifier" carries a runnable script
with explicit `D1_VERDICT PASS` / `FAIL` / `INCONCLUSIVE` output and worked examples of each. Run it
before any design leans on D1 either way. Area 4 itself flags the claim as needing verification and
should not be read as settled.

---

## 5. What binds prefill — VPU throughput, or VMEM capacity?

**Three sheets converge against `report.md`, and the report half-agrees with them already.**

`report.md` §4 models prefill at 623 µs and attributes it to the VPU term. §4 tags the VPU rate
`assumed` (1.72e12 ops/s, derived from a *ratio* in `05-compute-units.md`, never measured for this
op) and §7 records that the probe which tried to measure it **measured HBM traffic instead and is
void**. So the report's own confidence here is low.

Areas 2 (constraint 7), 4 (A7, D5) and 5 (A8) independently computed the same thing from source: the
`compute_scores` einsum output at `bq_sz=128` is `f32[1, 4096, 2048]` = **33.55 MB = 32.0 MiB**,
against a 32 MiB default scoped VMEM ceiling and 64 MiB physical. `DEFAULT_VMEM_LIMIT_BYTES` is set
to 100 MiB, above physical, so the compiler will not reject it — it will spill.

**If Mosaic materializes that intermediate rather than fusing the reduction into the einsum's output
loop, prefill is spill-bound and "prefill is VPU-bound" is the wrong diagnosis** — which changes the
prefill design target completely.

**What settles it:** one probe answers both, and it must be instrumented for **spills**, not only
wall-clock — a spill-bound and a VPU-bound prefill produce the same number and demand different
kernels. `/xprof-llo` reports spills; that is the read to take. Note that the cheap discriminator is
cheaper still: halving `bkv_p` (2→1) or `bq_sz` (128→64) halves the intermediate, so a sweep that
shows time falling faster than the work does is a spill signature.

---

## 6. How much headroom does stage B actually have?

**Not a contradiction between sheets, but a correction to `report.md` that shrinks the prize and
belongs on the record.**

`report.md` §4's floor for stage B is effectively a zero-compute floor — bytes moved over bandwidth
— which implies a large multiple of headroom. **Area 5, §C.8** builds a floor from the algorithm
instead, counting emitted vector ops per 16-lane chunk: `cvt` ~5, `hist0` ~9, `ref_body` ~13 per live
refinement level, `emit` ~18 ⇒ ~58–71 ops/chunk for the incumbent, against ~25 for an irreducible
exact select (one histogram pass, one emit pass).

That puts the incumbent at roughly **40% of the floor of a better exact algorithm** — ≈12 µs at
decode, ≈190 µs at prefill. Still inside the skill's "structural" band, and still worth a kernel, but
**not the 5× freebie the zero-compute floor implied.**

The op counts are read off the source; the floor is `modelled`. Anyone quoting it must say so, and
must not quote area 5's ≈1.7 ns/element as a bandwidth — it is a derived per-element rate with a
fitted dispatch subtraction in it (see §3 above).

---

## 7. Corrections to `references/hardware-facts.md`

Not disagreements between sheets — both sheets that touched this agree, and the reference file is
wrong. Recorded here because a designer who trusts the reference will discard a working option.

1. **SparseCore is reachable from Pallas on this stack.** `hardware-facts.md:317-318` says "there is
   no SparseCore path on this stack to check it against", and `05-compute-units.md` goes further:
   "You cannot reach it from Pallas on this stack… Do not write `plsc` code against this jax
   version." Both are wrong. The path is `from jax.experimental.pallas import tpu_sc as plsc` — the
   one module path the original `sparsecore_surface` probe never tried, as that probe's own Gaps
   entry admits. This repo ships a production kernel on it, and job **`j-0b008512-36f2-4eb2-bdc6`**
   ran it at four shapes on tpu7x, all exact against a numpy oracle. A design obeying the reference
   would have proposed rewriting a working SparseCore top-k onto the TensorCore for no reason.
2. **The subcore count is 2 × 16, not 32.** `sparsecore_topk.py`'s docstring claims "all 32 subcores
   on the chip"; the hardware reports `num_cores=2, num_subcores=16`. The product is 32. The
   kernel's arithmetic is right (it builds the mesh from the reported fields); the docstring and the
   hardcoded literals at `:85-86` are the hazard. **Any candidate must read
   `sc.num_cores * sc.num_subcores`, never a literal 32 and never the docstring.**
3. **Correct and load-bearing:** `sparse_core.vmem_capacity_bytes = 524288` is **per vector subcore
   and is not divided**. It is the constraint behind `MAX_SLICE_WORDS`, therefore behind `p`,
   therefore behind whether the two `take_along_axis` gathers exist at all.

Whether to amend `references/hardware-facts.md` is the user's call and has been raised with them; it
is outside this kernel's scope and no design should wait on it.

---

## 8. One methodological rule all the sheets are subject to

Several sheets quote per-arm times from `bench_streamidx_topk.py`. Two properties of that instrument
constrain how those numbers may be used, both recorded in `report.md` §6:

- **It reports best-of-N per call, not medians** (`:624`, `best = min(best, ...)`). Any subtraction
  of one arm from another is therefore optimistic twice over.
- **Run-to-run spread is ±8.5%** on the prefill arm (857.3 vs 929.9 µs for the same configuration).
  So the incumbent must be re-measured **in the same job** as every challenger; a challenger compared
  against a number from a previous job is comparing against noise of the same size as most of the
  levers in this document.

`report.md` §6.2 additionally **retracts** the ~43 µs dispatch add-back as fitting. Area 2's D1
reconciles this with `03-pipelining.md`'s 150 µs per-`pallas_call` dispatch floor correctly: that
floor is scoped to a single `block_until_ready`, and the bench enqueues `iters` calls and blocks
once, so it amortises. **Use 5.2–5.4 µs for an empty in-model launch; never 126.8 or 150 µs.**
