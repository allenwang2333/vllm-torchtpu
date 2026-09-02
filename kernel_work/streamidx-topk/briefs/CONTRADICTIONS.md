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

> **RESOLVED by P5 — one consumer set. `D1_VERDICT PASS`. See §14.1.**

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

> **RESOLVED by P7b — VPU, and specifically register pressure, not VMEM capacity.
> The question's own premise was false: there is no 32 MiB scoped ceiling. See §14.4 and §7.4.**

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

4. **There is no 32 MiB scoped VMEM ceiling on this stack.** `hardware-facts.md`'s claim that the
   scoped limit defaults to half the 64 MiB physical unless `vmem_limit_bytes` is raised is wrong
   here: `DEFAULT_VMEM_LIMIT_BYTES = 100 MiB` `read from source`, and neither production nor the
   microbenchmark overrides it. **Raising `vmem_limit_bytes` is a no-op**, and any design reasoning
   from a 32 MiB budget is reasoning about a constraint that does not exist. `measured (P7b,
   j-9e113359)` corroborates: the prefill kernel's real working set is 9.52 MiB and it compiles at
   a 12 MiB bisection bound.

   **I propagated this error into the P7 dispatch myself**, which is how it got tested. It is the
   third wrong claim in that file and the second that would have mis-shaped a design.

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

---

## 9. The A→B boundary is two edges with prizes three orders of magnitude apart

Area 1 supplies the distinction the other seven sheets were eliding, and it is the single most
useful structural point in step 2. Terms, because they are routinely conflated:

- a **kernel boundary** is a place where an intermediate is materialised to HBM and a launch is
  paid. Removing it means putting both sides in one `pallas_call`.
- a **unit boundary** is a place where work moves between physically distinct compute — the
  TensorCore's VPU/MXU and SparseCore's 2 cores × 16 subcores. Removing it means moving the
  algorithm to the other unit, or writing one kernel that spans both.

Stage A → stage B is **both at once**, and the two edges price out completely differently:

| edge | prize at nightly decode | source |
| --- | ---: | --- |
| kernel edge — the score matrix's HBM round trip, `2 × 2.62 MB / 3.0e12` | **1.75 µs = 0.79%** of the 221.4 µs call | `modelled`, area 1 A1; measured call `j-8b1e7acc` |
| unit edge — TensorCore stalled while SparseCore works | **≈71 µs/layer** attributable to `sc_topk_direct` | `measured (j-a71a648c, AgentX)`, area 1 C4(ii) / area 5 |

**A design that conflates them will spend its effort on the 1.75 µs.** Area 1's reframe is the
recommendation this document adopts: the unit edge is worth ~40× the kernel edge, **and attacking it
does not require TC↔SC shared memory** — overlapping the two units means pipelining *through* HBM,
stage A scoring group *g* while stage B selects on group *g−1*. Area 1 A4 establishes that the
incumbent's loop nest already emits complete score rows at a natural granularity (4 rows per decode
grid step), and A11 that the score write-back is already a deferred-wait async copy — i.e. this
block is already shaped for the overlap.

Two things must not be claimed together: that fusing A and B removes the HBM traffic **and**
delivers the overlap. Separate mechanisms, separate prizes; quoting one number for both
double-counts.

This also settles §2's framing indirectly. Whatever stage A's achieved fraction turns out to be, the
largest lever on the sheet is not inside stage A — it is the serialisation between the two stages.

---

## 10. The empty-launch price is now triangulated, and D1's expensive half is closed

`report.md` §8 listed `StreamIdxTC-d`'s bimodality as an open probe and area 1's D1 asked for a
hardware job to price an empty launch. **Area 7's CSE table closes the structural half from an
artifact already on GCS.** `measured (j-a71a648c rank 0, hlo_stats, untruncated 16554/16554 rows)`:

| scope | distinct HLO instrs | per layer |
| --- | ---: | ---: |
| `StreamIdxTC-d-bq_1-bkvp_2/…/pallas_call` | 42 | **2** |
| `StreamIdxTC-m-bq_128-bkvp_2/…/pallas_call` | 21 | **1** |

Two decode instructions per layer is exactly area 1's A7 structure — call 1 (`seq_batch_size=4`) and
call 2 (`seq_batch_size=1`) — and at AgentX b=16, call 1's grid is `16//4 = 4` while call 2's is
`16 % 4 = 0`. So the bimodal pair is **one running call at 114.63 µs and one empty call at 5.40 µs**,
not an unexplained bimodality; area 7 confirms the average over all 42 is 60.0134 µs = (114.63 +
5.40)/2 to four figures. Call 3 (`StreamIdxTC-m`) is empty at a pure-decode step and costs 5.22 µs.

**This is no longer a candidate explanation. Mark it confirmed** and supersede `report.md` §8's probe
accordingly.

What that leaves, and it is worth stating because it is the useful part: **two independent empty
launches landing within 3.5% of each other (5.40 and 5.22 µs) is strong evidence both are measuring
launch overhead rather than residual work.** It also refutes `07-precomputation.md`'s hypothesis that
the per-invocation cost is allocation-driven — these two launches carry a **16.8 MB** output
allocation and cost 5.4 µs, against that file's 26.2 µs at a smaller one. Invocation-driven, not
allocation-driven, which is the branch under which **merging the calls beats capping the score
width**.

Still open and still worth the cheap arm of D1: whether ~5.3 µs holds at the nightly score width.
The counts are structural and carry over unchanged; the µs do not.

**Consequence for the lever list.** Area 1's tier 1 — merge calls 1 and 2, both `MlaCase.DECODE`,
both `bq_sz=1`, differing only in `seq_batch_size`, handling the remainder by masking inside the
batched kernel — recovers ~5.4 µs/layer with **no scratch growth**, and area 2 c12 reaches the same
conclusion independently. That is now the best-evidenced cheap win in step 2. Note it is ~2.4% of a
221.4 µs decode call, which is inside §8's ±8.5% run-to-run spread: **it must be measured by the
microbenchmark in the same job, and cannot be shown end to end.**

---

## 11. RESOLVED — the copy is elided, my subtraction was wrong, and the bucket is now itemized

**This section previously argued that ~964 µs/step/core of the "indexer plumbing" bucket was
unattributed and that an un-elided KV-cache copy was the likely occupant. Probe P0 refutes both
halves.** The original argument is replaced rather than annotated, because two of its numbers were
wrong and leaving them readable invites requoting.

### 11.1 The copy is elided. VERDICT: **ELIDED**

`measured (j-a71a648c rank 0)`, two independent routes:

- **Static.** The module header declares `input_output_alias` with 177 entries. Of `ENTRY
  main.1741_spmd`'s 2070 parameters, exactly **21 have shape `u8[3027,64,4,256]` and all 21 are
  aliased** (`param 22 -> out{0}` … `param 1975 -> out{169}`, all `may-alias`). A whole-module
  opcode scan on that shape finds 63 `custom-call`, 42 `parameter`, 21 `scatter`, 21
  `get-tuple-element`, 21 `fusion` — and **zero `copy`, `copy-start`, `copy-done`, `bitcast`**. XLA
  inserted no defensive un-donation copy.
- **Dynamic.** The 21 aliased rows average **3.024 µs** self time, 63.50 µs/core-step. A real copy of
  `3027×64×4×256 = 198,377,472 B` read+write is 396.75 MB, which at this core's own
  `peak_hbm_bw = 3686.16 GB/s` cannot finish under **107.6 µs**; 3.024 µs would require 131 TB/s,
  36× over peak. Read the other way, `3.024 µs × 2.525 GiB/s = 8197 B` against the **8192 B** a
  16-row in-place scatter actually moves — agreement to 0.06%.

So `donate_argnums=(2,)` does its job and `args[2].copy_(new_cache)` at `mla:140` costs nothing.
Area 1's A10 and D3 are closed on the copy question. **Still not determined:** whether the elision is
*caused by* donation or whether XLA's copy insertion would reach the same in-place assignment
regardless. That needs an A/B compile and no design depends on it.

### 11.2 My arithmetic was wrong: the two figures were over disjoint sets

The subtraction `1230 − 265.7 ≈ 964` was invalid. Area 7's seven metadata scopes live under
`custom_kernel/custom_kernel/cond/branch_0_fun/*`; area 1's bucket is `conditional` + scope `cond:` +
scope `bitcast_convert_type:`. **These do not overlap**, so nothing was ever subtractable. P0
reproduces area 7's figure independently at 268.91 µs against its 265.7, so that number was fine —
the error was mine, in combining them.

Corrected statement, `measured (j-a71a648c, AgentX)`:

| | µs/core-step |
| --- | ---: |
| the `conditional` / `cond:` / `bitcast_convert_type:` bucket | ~1255 (area 1 reported 1230) |
| the `branch_0_fun` metadata scopes, **separately** | ~269 |
| **total non-kernel indexer overhead** | **~1524** |

### 11.3 What is actually in the bucket — 96% of it is three items

`measured (j-a71a648c, AgentX)`. This is the useful output of P0 and it is not what the probe went
looking for.

| µs/core-step | n | what |
| ---: | ---: | --- |
| **496.16** | 21 | `copy f32[2,8,2048,128] {3,2,1,0}S(1) -> {3,1,2,0}`, 23.63 µs each — a **transposing 16 MiB copy of the whole score buffer, once per indexer layer**. 352 MiB/core-step moved for zero arithmetic |
| **255.27** | 21 | `bitcast-convert s32[16,2048,128] <- f32[16,2048,128]`, 12.16 µs each. Same shape, same layout, same tiling — **a semantically free reinterpret that XLA has fully materialised** |
| 178.08 | 21 | `conditional` self time (the early-exit guard), plus 70.89 µs of boundary copies and 134.71 µs of `slice-start/done` it forces — ~384 µs total for a guard that never takes the cheap branch at these contexts |

Two of these close open questions with a **negative** answer that matters:

- **The `bitcast_convert_type` at `sc:320` is not free.** Area 1's C3 listed both bitcasts as "free in
  principle… whether XLA elides it is `[unverified]`". It does **not** elide it: 255 µs/core-step at
  AgentX for a reinterpret between two identically-shaped, identically-tiled buffers. This is
  plausibly the most directly deletable item in the block.
- **Stage A emits a layout its consumer does not want.** The transposing copy sits between stage A's
  output and stage B's input and is a *third and fourth* pass over the score matrix, on top of the
  write and the read area 1's A1 priced. A1's `2 × bytes` round-trip formula therefore **understates**
  the matrix's true HBM traffic — though see 11.4 before rescaling any conclusion off that.

### 11.4 The correction that guts all of it: this is an AgentX problem, not a nightly one

Items 1 and 2 are **linear in the score-buffer width**, which is 262144 on the profiled job and
**10240** at the baseline of record — 25.6× narrower. Rescaled `modelled`, the 496 µs becomes ~19 µs
and the 255 µs becomes ~10 µs, **per step across all 21 layers**, i.e. ~0.9 µs and ~0.5 µs per
221.4 µs decode call. Both fall under §8's ±8.5% run-to-run spread and neither is measurable end to
end.

**So the honest verdict on the whole plumbing bucket is: it nearly vanishes at nightly.** That is a
clean negative result and it is worth as much as a positive one — it removes an entire ~1.5 ms/step
line item from the nightly design's candidate list, and it does so before anyone spent a job on it.

What the designer should still carry from 11.3 is **structure, not µs**: stage A's output layout is
wrong for its consumer, and a candidate that changes that layout must check it has not made the
transpose worse rather than better. The `bitcast` is worth deleting on principle at any width.

### 11.5 One structural fact, unaffected by width

The 21 `scatter_fusion` instructions that write the KV cache sit in `ENTRY`, **outside** the
`conditional`. The KV write therefore runs unconditionally even when `STREAMIDX_ENABLE_EARLY_EXIT`
fires and the fast path is taken. Any option that touches the early-exit structure must preserve
this.

---

## 12. `6142d5e7` replaced an exact XLA op with the kernel, and the margin was never recorded

Area 1's D4, corroborated by area 6. `sparsecore_topk` displaced
`lax.approx_max_k(recall_target=1.0)` — which is **exact**, not approximate; this is the false
premise that withdrew `report.md` §8's exact-top-k probe — and it was displaced **for performance**.
**No margin was recorded in the commit, and no GLM-5.2 eval baseline exists in the repo**, so the
size of the win the kernel bought is unknown to this session.

Per §CLAUDE-file rule, a number nobody measured is not a number, and this one does not even exist.
It matters because it decides whether a design option is *allowed to propose deleting stage B*:

- if XLA's `approx_max_k` was close, a candidate may return stage B to XLA and delete 457 lines,
  which also dissolves §9's unit edge entirely — there would be no SparseCore boundary to overlap.
- if the kernel was much better, stage B's 30% has to be attacked from inside the kernel.

This is minutes on a pod — `lax.approx_max_k(recall_target=1.0)` and `lax.top_k` against
`sparsecore_topk` at `(64, 10240, 2048)`, one job — and it is the only probe that can *remove* a
design option rather than rank one. Ranked **P2**.

Note what it does **not** license: proposing the XLA path on the strength of the deletion alone. The
incumbent is a candidate in step 3 and this is measured against it in the same job, per §8.

---

## 13. The probe set, after all eight sheets

Reordered from the six articulated before areas 1 and 7 landed. Two probes are gone (the E1 lever,
killed by arithmetic in §1; `StreamIdxTC-d` bimodality, confirmed from source in §10) and two are
new.

| | probe | cost | what it gates |
| --- | --- | --- | --- |
| ~~**P0**~~ | ~~`donate_argnums` elision at `mla:140`~~ — **DONE: ELIDED** (§11) | free | Settled, negative. No copy exists, and the bucket it was probing nearly vanishes at nightly (§11.4) |
| **P1** | DMA-stall fraction in `StreamIdxTC-d` via `/xprof-llo` (semaphore-wait at `sx:398`) | one pod job | §2's gate question — stage A near its floor (**≥50% confirms, ≤15% refutes**) — *and*, on refutation, the `assumed` VPU rate |
| **P2** | `approx_max_k(recall_target=1.0)` and `lax.top_k` vs `sparsecore_topk` at `(64, 10240, 2048)` | minutes | §12. The only probe that can delete a design option |
| **P3** | Can a TensorCore grid and a `VectorSubcoreMesh` be in flight together? | one pod job | §9's unit edge — the largest lever on the sheet. Area 1 D2 and area 5 name the same probe |
| **P4** | Empty-`pallas_call` price at grid extent 0 / 1 / 16 at the **nightly** score width, plus a `lax.cond`-guarded arm | minutes | §10's residual. Tier 1 vs tier 2 of area 1 C2 |
| ~~**P5**~~ | ~~Area 4's D1 falsifier~~ — **DONE: `PASS`, one consumer set** (§14.1) | done | Settled. A repack touches one file. Route 1 refused as non-discriminating |
| ~~**P6**~~ | ~~`[:129]` strided read charge~~ — **DONE: charge ratio 1.02, but w=129 is inexpressible** (§14.2) | done | The one-line change does not exist; a 128+scale repack does, and realises ~98% |
| ~~**P7**~~ | ~~`bkv_p` sweep; prefill spill~~ — **DONE: `bkv_p=3` is −11.9%; prefill spills 23.5× over** (§14.3-4) | done | Settles §5. Stage A is DMA-**block**-bound, not bandwidth-bound — partially pre-empts P1 |

P0 ran first and alone, because it could have changed what the rest were for. It did not — see §11.
**P5–P7 have since returned (§14); P1–P4 are still in flight.** Four of the eight are closed.

**A note on P0's yield, because it generalises to the probes still running.** P0 answered its own
question in the negative and was still the highest-value probe run so far — it itemized 96% of a
1.5 ms/step overhead bucket nobody had opened, closed area 1's C3 `bitcast` question with a
measured "not free", found a per-layer transposing copy of the whole score buffer, and caught an
arithmetic error of mine that would otherwise have propagated into the design. **Judge the probes
still in flight on what they establish, not on whether they confirm the hypothesis that motivated
them.** A probe that refutes its own premise and itemizes the thing it was digging through has done
the more useful job.

---

## 14. Wave 2, first job: P5, P6 and P7

Three probes, `measured` on jobs `j-9e113359-47c6-41b2-ad85` and `j-170ed342-8cc7-4719-82b5`.
Full write-ups in `../probes/P5-cache-consumers.md`, `P6-strided-read-charge.md`,
`P7-bkvp-and-spill.md`. **Read the contention caveat in 14.5 before quoting any absolute µs from
this section.**

### 14.1 P5 — the indexer K cache has exactly one consumer set. `D1_VERDICT PASS`

`measured (j-a71a648c, hlo_stats, two sessions)` plus an independent source derivation. The cache
`u8[3027,64,4,256]` is a separate allocation from every MLA-side buffer, and its complete toucher
set is **three instructions, all in the indexer path**: readers `StreamIdxTC-d-bq_1-bkvp_2` and
`StreamIdxTC-m-bq_128-bkvp_2`, writer `scatter_fusion`. The MLA attention kernel reads different
buffers (`u8[16,8192,128]` + `u8[16,2048,128]`), and `sc_dsa_gather` reads different ones again.
Source agrees for a shape-independent reason: all 78 MLA layers take the split-tuple `use_sparse`
branch, while `DeepseekV32IndexerCache` has no `use_sparse` and takes the single-4-D branch.

**Settles:** a repack of the indexer K cache is confined to `streamindex_topk.py` and its one
writer. No MLA-side coordination, no second reader to keep in sync. **Does not settle:** whether the
repack pays — see 14.6.

**The falsifier's route 1 was correctly refused rather than run.** Its final `else` prints
`D1_VERDICT PASS distinct-buffers` whenever the indexer set is non-empty and the MLA single-tensor
set is empty — *including when the MLA side classified nothing at all* — so it would go green in a
state where it had proven nothing. Its `shared`-storage pointer check is also dead on this config.
This is exactly the hollow-check failure mode the review policy exists to catch, and it was caught
by the prober rather than downstream. Route 2 carried the verdict.

**One term correction.** The "83.2 MB of padding" quoted in §4 and in the dispatch is read *traffic*
per decode call (`report.md` §4), **not storage**. Storage padding is 98.4 MB/layer at AgentX
geometry and 74.9 MB at the nightly microbenchmark shape, `modelled`. Do not add them.

### 14.2 P6 — a strided read is charged for its bytes, but the 129-byte read cannot be written

`measured (j-170ed342, run alone on an uncontended 4-chip pod, RUN_CMD read back from
`cdk job recipe`)`:

| pattern | time | achieved B/s on useful bytes |
| --- | ---: | ---: |
| w=256 contiguous (control) | 160.39 µs | **3.347e12** |
| w=128 of every 256 | 81.84 µs | **3.280e12** |

Time ratio 0.510 against a byte ratio of 0.500 → **charge ratio 1.02**; the strided read achieves
**98.0%** of contiguous bandwidth. Calibration passed (112.7% of the 2.97e12 read-write reference,
90.5% of spec), so this is bandwidth-bound rather than latency-bound and the ratio is meaningful.

**But the probe's headline arm failed, and the failure is the finding.** `w = 32, 64, 96, 129, 160,
192, 224` **all refused to lower**: Mosaic requires *"Slice sizes along tiled dimensions … divisible
by the tile dimension 128."* So `P6_VERDICT: INCONCLUSIVE at w=129`, and **narrowing to 129 bytes is
not expressible in Pallas at any charge ratio.** The granule is bounded at ≤128 B and cannot be
located more precisely, because what binds first is Mosaic tile alignment, not the DMA engine —
`dma_granule_size_bytes = 32` is therefore neither confirmed nor refuted for TensorCore.

**Consequence, and it is a good one.** The "one-line movement-only change" this probe was meant to
price **does not exist**. What replaces it is better evidenced: since 129 = 128 + 1, and a
128-aligned strided read is charged 1.02× its bytes, the reachable design is a **repack into a
128-wide payload plus a separate scale array** — the nope/rope split — which would realise ~98% of
the byte saving. P5 says that repack touches one file. **P5 and P6 converge on the same design
option from opposite directions**, which is the strongest structural result in wave 2.

### 14.3 P7a — `bkv_p = 3` is worth −11.9%, four times the estimate

`measured (j-9e113359 shard 1)`, `prod_decode_b64_ctx8704`, **contended pod — ratios only**:

| `bkv_p` | total µs | stage A µs | vs `bkv_p=2` |
| ---: | ---: | ---: | ---: |
| 1 | 213.34 | 136.24 | +11.88% / +19.95% |
| **2** (incumbent) | 190.68 | 113.58 | — |
| **3** | **167.97** | **90.87** | **−11.91% / −19.99%** |
| 4 | 189.17 | 110.05 | ❌ confounded (`n_pos` 12288, not 10240) |

Area 2's top candidate is confirmed and is **~4× larger than its ≈2.5-3% estimate**. It is a
one-tuple change with no layout surface, which makes it the cheapest evidenced win in step 2.

**The second reading matters more than the first.** `P7a_DMA_VERDICT: MOVES` — time responds to
`bkv_p`, so the fetch *is* the constraint. But it moves at **near-constant bytes** (9 / 10 / 9 pages
fetched across the sweep), so **stage A is DMA-*block*-bound, not bandwidth-bound.** Those demand
opposite fixes: bandwidth-bound says move fewer bytes (the P5/P6 repack), block-bound says issue
fewer, larger transfers. Corroborated by the LLO decode program: **MXU 4.61%, 15.5% fully idle,
15.5% SALU bubble.**

This partially pre-empts P1, still in flight, and the two must be reconciled when it lands.

**Two caveats the designer must carry.** The absolute times are contended and disagree with the
221.4 µs baseline of record by more than `report.md`'s documented spread — use ratios only. And
`bkv_p=3` may be exploiting `9 = 3×3` at ctx 8704 specifically; **sweep other context lengths before
defaulting it.**

### 14.4 P7b — prefill spills registers 23.5× over, and the question's premise was false

All of this is **static compile-time analysis from the LLO dump, not measurement.** No timings.

**First, my dispatch was wrong and the probe caught it.** There is no 32 MiB scoped VMEM ceiling:
`DEFAULT_VMEM_LIMIT_BYTES = 100 MiB`, unoverridden. Recorded as §7.4.

**Not VMEM-capacity-bound.** Working set **9.52 MiB** — 4.20 MiB named scratch plus a **5.33 MiB
spill arena** — confirmed by an independent bisection (compiles at 12 MiB, rejected at 8 and 4).
Note `get_llo_fit_summary`'s headline of 4.2 MiB **omits the spill arena and is wrong by 2.3×**;
do not use it as a VMEM figure.

**It spills registers, hard.** 8,458 spills + 10,130 fills over 11,658 bundles = **1.59 per bundle**;
**1,506 live vregs against 64 physical, 23.5× oversubscribed**; `VSTORE:SPILL` is 97.1% of all
VSTOREs and `VLOAD:FILL` 87.8% of all VLOADs. VALU 94.19% (89.45% saturated) against MXU 35.13%.

**The 32 MiB intermediate is not a buffer but is produced in full.** The score block is
`f32[2,128,16,128]` — no H axis — so nothing of that size is allocated; but 8,744 `vpop` × 1024 f32
= **34.16 MiB** is produced, 6.6% above the modelled figure, and 86.2% of spill stores are
vpop-sourced, occupying 4.87 MiB of the arena.

**Settles §5, and against both of its options.** Prefill is **VPU-bound on the post-matmul
reduce/score/select chain**, not on the einsum and not on VMEM capacity: 12,320 `vadd` + 10,240
`vmul` + 8,960 `vrot.slane` + 8,192 `vmax` against **1,024 `vmatmul`**. Stop looking for a capacity
fix. **Opens:** register pressure as the lever — flagged as a *hypothesis*, because the remedy sits
near an `xprof-cli` recommendation that ships self-labelled `TRACK RECORD 0/5, directly refuted
twice`. Treat accordingly.

Also corrected: the `f32[1024,80,128]` buffers named in the OOM messages are **HBM arguments**, not
VMEM.

### 14.5 Contention, and what may be quoted from this section

`j-9e113359` fanned out on a shared pod; its absolute µs are contended and disagree with the
baseline of record. **P7a's ratios are usable; its absolute times are not.** P6 ran alone on the
whole pod and its absolutes are usable. P7b produced no timings at all by design.

### 14.6 The prober's cross-probe recommendation, and one reason to hold it lightly

Its conclusion: take `bkv_p = 2 → 3` first — cheap, evidenced, no layout surface — re-measure, and
only repack the cache if stage A then approaches its ~50 µs HBM floor, since it is currently
**1.81×** above even the un-repacked floor and the byte saving is therefore not yet binding.

That ordering is sound and this document adopts it. **The 1.81× is computed from a contended
number** (90.87 µs against a ~50 µs modelled floor), so it is an order-of-magnitude argument rather
than a measurement, and it should be recomputed once `bkv_p=3` is measured uncontended in the same
job as the incumbent, per §8.
