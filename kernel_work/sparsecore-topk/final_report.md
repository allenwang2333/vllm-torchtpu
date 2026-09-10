# `sparsecore_topk`: one composed candidate

**For whoever raises this change.** Subject:
`src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`, the SparseCore
radix top-k behind DeepSeek-V4 / GLM-5.2 sparse indexer attention. Sole
production caller: `streamindex_topk.py:831`.

| | |
| --- | --- |
| candidate | `kernel_work/sparsecore-topk/final/sparsecore_topk.py`, md5 `109a12a572116c00c795ee170256db33` |
| diff vs incumbent | `final/variant.diff`, md5 `fca4c3ac608d611ce84ca6e2484fbadb`, 368 `+` / 88 `−` |
| incumbent | md5 `44353b347c96e0eedbd742c710d4540a` |
| scope name | `sc_topk_direct_b{b}_n{n}_k{k}[_keys]_c{chunk}_final`; production `…_b1024_n10240_k2048_c2048_final` |
| correctness | `j-49ee915f-6906-43dc-ba3f` — **green, bit-identical** |
| composed speed-up | **−74.206 µs**, prefill whole-call, `measured`, `j-4dd339df-0495-4d6c-9afc` |

**Sign convention.** Every percentage is `(new − old)/old` of **time**;
negative is faster. Every number is tagged `measured`, `static-compile`,
`modelled` or `unmodelled`.

---

## The number

**`measured`, one job, `j-4dd339df-0495-4d6c-9afc`.** The composite against the
incumbent at prefill `b=1024`, `ρ=0.40005`, production arrangement. Nine gates
green in order; arm activation proved by the **observed scope name in the
lowered program** — `sc_topk_direct_b1024_n10240_k2048_c2048_final`, against
four incumbent slots all reading the bare name.

| cell / bracket | incumbent µs | composite µs | Δ µs | floor µs | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **`P.prefillX` whole** | 275.196 | **200.989** | **−74.206** | 0.200 | ✅ **RESOLVED FASTER** |
| `P.prefillX` inside | 240.593 | 191.325 | −49.268 | 0.105 | ✅ RESOLVED FASTER |
| `D.decodeX` whole | 49.487 | 44.751 | −4.737 | 0.465 | ✅ RESOLVED FASTER |
| `D.decodeX` inside | 46.747 | 42.760 | −3.987 | 0.155 | ⚠ NOT RESOLVED — 2/3 |
| `P.rho1` whole *(control)* | 425.511 | 341.134 | −84.377 | 1.419 | ✅ RESOLVED FASTER |
| `P.rho1` inside *(control)* | 391.927 | 333.834 | −58.093 | 3.362 | ✅ RESOLVED FASTER |

RESOLVED means all three estimators — `us_median`, the pooled slope and the
unpaired estimate — agree in sign and each clears its own corrected floor. The
`[MARGINAL]` flag fired **nowhere**. The prefill whole-call win is 371× its
floor.

**Quote the µs.** The prefill win is `−74.206 µs = −26.96%` whole-call and
`−49.268 µs = −20.48%` inside-kernel — one physical effect, two percentages,
because the denominator is bracket-specific and cell-specific. Only the
absolute freed device time converts to an e2e claim.

**The prefill whole-call cell was the pre-registered headline**, fixed before
any number existed: *the composition succeeded iff `P.prefillX | whole` is
RESOLVED FASTER.* It is.

**⛔ This number was not obtained by adding the three arms' µs, and it must not
be checked that way.** The arms are **not disjoint** — five mechanical
collisions, one of which `diff` reports as clean (A′ below) — so their deltas
are not additive quantities and no sum is a prediction this could fall short
of. It also happens not to have fallen short; see "Did it land short?" below,
where that non-shortfall is deliberately *not* offered as corroboration.

### `v12`'s glue deletion survived composition

Pre-registered reading (3), decided **within this one job** by the difference
of two brackets and no cross-job arithmetic: whole −74.206 against inside
−49.268 is a **24.9 µs** gap, so the entry-glue deletion is still there. Had
the merged entry reintroduced the fusion, whole and inside would have matched
and the composite would have been effectively `v2` + `c2048`.

That gap independently lands on `v12`'s static bytes model (≈24.9 µs) and its
standalone −25.15 µs. ⚠ **That agreement is a cross-check, not the source of
the claim** — and `v12`'s share stays `static-compile`, not `measured`, because
the AOT lowering feeds from a parameter where production feeds from a
custom-call output. Both are fusion barriers, so the argument transfers; it is
still an argument.

### Did it land short?

No — and the honest form of that statement is narrower than it looks. The
composite's −74.206 µs sits marginally *above* the arithmetic sum of the three
standalone arms (−73.79 µs). **That comparison spans two jobs, and this
exercise's own standing rule is that cross-pool subtraction never reconciles**
(`probes/18`). So it is reported as "not short", and nothing tighter is claimed
from it.

What it does settle is a real ambiguity: five mechanical collisions in the
merge did **not** translate into overlapping *effects*. A collision is a
statement about two diffs touching the same line — it says the merge needed
care, which it got, not that the arms compete for the same time.

### ⚠ One cell did not resolve, and the reason is an instrument fault

`D.decodeX | inside` is **NOT RESOLVED** — but not because the estimators
disagree about the effect. All three point estimates agree (−3.987 / −4.257 /
−4.044 µs) and two clear their floors by 2469% and 528%. The single failure is
the pooled slope, whose floor blows out to **5.062 µs** because one incumbent
slot, `#null_c`, read `us = 40.900` against its own `us_median = 46.663` and
`us_unpaired = 46.298`, at **hostbound 15.0% where every other slot in that
cell is ~5%**. A host-bound excursion corrupted that slot, and a floor taken as
the *range* of N slots is set by its worst member, so the excursion swallowed a
real effect.

**The verdict stands as pre-registered** — the rule was not reached back and
changed to rescue a cell. It is recorded as one contaminated slot rather than
three estimators disagreeing, with the slot values printed beside it. This is
the fifth paragraph of the method note below.

Decode is not the headline in any case: at ~1.2% of decode device time,
−4.737 µs is worth **~0.08% of the decode step at most**.

## What changed, and why

Three arms, composed into one file. Each was measured **alone**, at prefill
`b=1024`, `ρ=0.40005`, production arrangement, against an incumbent of
275.213 µs whole-call / 240.659 µs inside-kernel (floors 0.040 / 0.433 µs).
Those per-arm figures are `measured`; **their sum is not a prediction**.

| arm | mechanism | whole Δµs | inside Δµs |
| --- | --- | ---: | ---: |
| `v2` R=1 shared histogram | `plsc.addupdate_scatter` resolves intra-vector address conflicts in hardware, so sixteen lanes may scatter into **one** histogram. `addr = lane_iota * NUM_BUCKETS + bucket` becomes `addr = bucket`, the 16-deep fold collapses to a copy, and the scratch drops 16 KiB → 1 KiB. | −37.05 | −37.40 |
| `v4-c2048` ragged + pipelined score read | the row's **live** prefix moves instead of its padded capacity, in `tc = 2048`-word chunks that `pltpu.emit_pipeline` double-buffers, with the monotone-key pass fused into the pipeline body. | −11.59 | −11.19 |
| `v12` f32 input | the kernel takes `f32` and reinterprets in-register, deleting a caller-side `bitcast_convert_type` that XLA planned as a real `kLoop` fusion over the whole 41.9 MB score array. Entirely **outside** the kernel. | −25.15 | −0.08 |

Plus a hardware-constants cleanup (below), which is a **bit-identical
refactor**, not an optimization.

### The two places the arms collide

The merge was done by reading both diffs line by line, not by running a merge
tool. Full analysis in `final/README.md` §2; the two that matter:

- **A.** `v2` shrinks the `zero_priv` bound (one histogram, not sixteen); `v4`
  *moves* that same call out of its `@pl.when(i_hi > 0)` guard, because
  `emit_pipeline` allocates with `pl.run_scoped` and an allocation may not sit
  under a `pl.when`. Different axes, so both apply:
  `zero_priv(jnp.where(i_hi > 0, NUM_BUCKETS // lanes, 0))`.
- **A′.** `diff` reports the scatter-address line as *unchanged context* in the
  `v4` diff, because `v4` relocates the level-0 histogram rather than deleting
  it — while `v2` **edits** that exact line. A three-way `git merge-file`
  would have taken `v4`'s relocated copy verbatim, silently reverting the
  largest arm in the composition on the level-0 histogram while keeping it on
  the refinement levels — which is not merely slower but **out of bounds**:
  under `v2`, `priv_vmem` is 272 words and the reverted address reaches 4096.

So `v2` and `v4` are **not disjoint**, and this is the concrete reason no sum
of their µs appears anywhere in this report.

### Hardcoded hardware constants

Taken from `pltpu.get_tpu_info().sparse_core` where it genuinely supplies them:
`MAX_SLICE_WORDS` ← `vmem_capacity_bytes // 2 // 4`; the kernel body's literal
`num_subcores = 16` / `num_sc_cores = 2` and `_pick_partition`'s literal `32`
← `num_subcores` / `num_cores` / their product.

`LANES = 16` **stays a module constant** — it is a static vector *shape*
(`jnp.arange`, `pl.ds`, a shift amount) so it must be a Python `int` at trace
time, and the frozen test imports it at module scope before any TPU is
consulted. Instead, `_sc_topk_direct` now **refuses to lower** if
`sc.num_lanes != LANES`. The constant and the hardware can no longer silently
diverge.

Bit-identity of the cleanup is `static-compile`, with four host-AOT controls
(`final/aot_final.py`, no TPU, one arm per process): the normalized jaxpr is
**byte-identical** to a version with every `sc.*` read textually substituted
back to its tpu7x literal; patching `sc.num_subcores` → 8 moves exactly 26
jaxpr lines, which is what proves the values are *read* rather than computed
and discarded; patching `sc.num_lanes` → 8 raises as designed.

## Correctness — `j-49ee915f-6906-43dc-ba3f`, green

Two shards, one chip each: the candidate, and the **pinned incumbent through
the byte-identical harness** so a red could be told from a harness finding.
All three subject md5s gated in-job, twice, because a docker image tag is
mutable and this one was re-pushed mid-exercise.

| gate | result |
| --- | --- |
| arm activation, by **scope name in the lowered program** | `ARM_CHECK_FINAL_OK` — every name over all 7 frozen-suite shapes carries `_c<chunk>_final`, including both halves of the 5 two-stage shapes; the incumbent read the same way carries it on **none** |
| frozen 68, unmodified, no `-k` | **68 passed**, 0 fail / 0 err / 0 skip |
| cross-gate vs the incumbent **file** (30 cells) | 30 exact, **30 / 30 bit-identical** |
| cross-gate negative control | **went RED** — `bit_equal=False`, `max score difference 4.8411946, expected exactly 0.0` |

Bit-identity was the expectation and it held. It also closes every
value-mediated mechanism ahead of the timing job: whatever the composed number
turns out to be, it cannot be the kernel selecting different columns.

Static checks, all `static-compile`: **zero VREG spills** in either arm; VMEM
high-water **+1 024 B** (73 216 B, 14.0 % of the 512 KiB subcore tile memory) —
`v2`'s 15 360 B saving nearly pays for `v4`'s 16 384 B double buffer outright;
TEC bundles 1 168 → 941 (**code size only — a bundle count is blind to stall
cost and must not be read as a time**); scatter, popcount and scan primitive
counts all unchanged, which is what a pure addressing change should look like.
Private VMEM was priced **behaviourally by allocation failure** with paired
positive (4 KiB → compiles) and negative (8 MiB → `E3000 … spmem`) controls,
because XLA's memory-usage report is **blind** to Pallas/Mosaic SparseCore VMEM
and reports `vmem_bytes: "0"`.

`v12`'s call-site win is **confirmed in situ** (`static-compile`): the
production caller hands `sparsecore_topk` a plain `f32` array, and the
optimized-HLO ENTRY at the production shape shows the incumbent's
`kLoop fusion(%s.1)` over `f32[1024,10240]` becoming a free
`bitcast(%s.1)`. The deleted pass is 83.9 MB of HBM traffic ≈ **24.9 µs**
`modelled` at this exercise's tpu7x figure — against `v12`'s `measured`
−25.15 µs, 1 % agreement.

## Tried and rejected

| | why |
| --- | --- |
| `v3` fused emit | **fails correctness**, 46/68. Diagnosed: `0x41800000` — f32 `16.0` — appearing in the index array. |
| `v10-a2` | arithmetic **regression**, +121.60 µs whole. |
| `v11` retiled output emit | −9.45 µs whole / −0.08 inside — the **same** ~25 µs entry copy `v12` deletes, and less of it. Superseded, not additive. |
| `V2_BIGSCRATCH` | −36.85 / −36.91 against `v2`'s −37.05 / −37.40 — **inside the floor**, and the sign inverts between estimators. |
| `tc = 1024` | **refuted**, +14.1 µs against the unpipelined ragged read. |
| `MAX_SLICE_WORDS` 65536 → 40960 | verified P-neutral (0 of 216 shapes) but a *different* constant value; not shipped, and deliberately not folded into the cleanup. |

## On the original request

The ask was: *"the DMA is synchronously wait instead of pipelining; use
`emit_pipeline` instead for DMA to enable double buffering."* This is a
**partial vindication**, and the parts matter separately.

1. **`emit_pipeline` halves the marginal per-descriptor cost** — 248.4 ns
   against 517.5 ns raw. That is the *opposite* sign to the bookkeeping
   overhead both the design note and the builder assumed. The premise that
   pipelining costs extra per descriptor here is simply false.
2. **At `tc = 1024` pipelining is refuted**, +14.1 µs against ragged.
3. **At `tc = 2048` it is resolvable at prefill and it wins**:
   `c2048 − ragged` = **−4.500 µs** whole / **−5.024 µs** inside, stable across
   three estimators, against a 0.040 µs floor. At decode it is a **tie**
   (−0.663 µs against a 0.730 µs floor).
4. **Most of `v4`'s prize is the ragged blocking DMA, which involves no
   pipelining at all.** Moving the row's live prefix instead of its padded
   capacity is worth −7.09 µs whole / −6.17 µs inside on its own; the pipeline
   adds the remaining −4.50 / −5.02. So raggedness is **61 %** of the
   whole-call win and **55 %** of the inside-kernel win — reported as a family,
   because "two thirds" and "just over half" are both defensible readings of
   the same pair. Either way, attributing the full −11.59 µs to pipelining
   overstates it by ~2.6×.

A rule registered before the measurement said *a tie ships `ragged`*. That rule
existed to prevent a post-hoc coin-flip on an unresolvable comparison. Prefill
**resolved** — −4.500 µs against a 0.040 µs floor is 113× the floor, not a
tie — so `c2048` is adopted **on the measurement**, and the pre-registered rule
was satisfied rather than dropped. Had prefill also come back inside the floor,
`v4-ragged` (md5 `69c8834e130e23999e0284929230da3a`) would have shipped.

## Method note — the resolution floor, and why four verdicts moved

**This governs every µs in this report.** The per-arm table above, the `c2048`
adoption and the rejection rows are all read through the corrected rule below,
not through the harness's printed verdicts. Four of those printed verdicts were
noise and one rejection became an adoption. Lifted verbatim from
`probes/24-final/README.md`, where it was written before the composed number
existed:

> **A resolution floor measured from one slot is measured from a biased slot.** The harness gives the
> incumbent four slots — `base` plus three nulls that are the same source loaded as separate module
> objects — and takes the floor as the range of the four. But it took every candidate's delta against
> `base`, one particular slot. When the three nulls come out **one-signed**, `base` is sitting at an
> extreme of its own floor, and a candidate at the incumbent centroid shows a delta of up to a full
> floor width and reads as a win. In the GLM-5.2 sctopk decode inside-kernel cell the nulls were
> −0.205, −0.211 and −0.103 µs — all one-signed — and four printed verdicts were noise. The fix is
> one line of statistics: **measure the delta against the median of the incumbent slots and keep the
> range as the floor.** Pair it with a three-estimator agreement test (`us_median`, the pooled slope,
> and the unpaired estimate must agree in sign and each clear its own corrected floor); `us_median`
> alone was the pre-registered read and it was not sufficient.
>
> **Apply the correction to every row, not only the rows you already doubt.** That is what separates
> an instrument from an adjustment. Applied uniformly here it removed four wins and *added* one — a
> promotion for a candidate that had already been adopted on other evidence. **A correction that only
> ever removes wins is one that was applied only to the rows you already doubted.**
>
> **When you validate a corrected estimator against known arms, count the anchors honestly.** Five
> replays were run; four reproduce conclusions that already contained the hand cross-checks the rule
> was written from, so they prove the code implements the reasoning, not that the reasoning is right.
> The one external anchor was an arm that is a null **by construction** — a ragged-row optimization
> at ρ=1, where there are no ragged rows — because its ground truth was fixed by its own mechanism
> before any measurement. One external anchor, four implementation checks.
>
> **Report a marginal resolved call with its margin attached — and with its spread.** One promotion
> above cleared its floor by 0.61% on its weakest estimator. It is resolved by the pre-registered
> rule and the rule was not reached back and changed, but "resolved by 0.61% of the floor" and
> "resolved" read very differently to someone deciding what to build on. The *decomposition* then
> says more than the flag does: the other two estimators cleared by 98.2% and 80.7%, and one
> estimator on the line while the others clear widely is a different object from three clustered at
> the boundary. The first says the call is probably real and that one estimator is the conservative
> one; the second says the effect itself is borderline. Same flag, opposite readings — so print the
> spread beside the flag, never the flag alone.
>
> **A floor taken as the RANGE of N slots is set by its WORST slot, so one contaminated slot swallows
> a real effect.** This is the mirror of the bias above, same instrument, opposite direction. In the
> composite's decode inside-kernel cell all three estimators agreed (−3.987 / −4.257 / −4.044 µs) and
> two cleared by 2469% and 528%, yet the call read NOT RESOLVED — one incumbent slot ran at
> **host-bound 15.0% where every other slot in the cell was ~5%**, and its corrupted pooled slope blew
> the range floor out to 5.062 µs. Keep the verdict; a pre-registered rule reached back and changed to
> rescue a cell is not a rule. But print the per-slot values and their host-bound fractions beside it,
> because "three estimators disagree" and "one slot was contaminated" wear the same verdict and only
> the decomposition separates them.
>
> **⛔ A normalizing constant is a measurement too — and if it is less reproducible than the thing you
> are correcting, it is a bound, not a correction.** The per-call dispatch floor was quoted to justify
> reading these results in µs rather than percent. Five job-level estimates of that same constant,
> from the same frozen function, spread **22.86 / 31.19 / 31.53 / 31.96 / 35.18 µs — ±40%** — against
> a kernel that reproduces across jobs to **0.016%**. The measuring stick was ~2500× less reproducible
> than the thing it measured, because it alone satisfied **none** of the nine mandatory probe
> properties: one warm pass, one `lo`, one `hi`, a slope from a single round, no pooling, no
> interleaving, no replication, no null. Nobody checked it for the worst reason available — **the
> conclusion it supported happened to be right on other grounds.** A correction factor gets the same
> protocol as a result, or it gets quoted as a bound.

The one call in this report that carries that marginal flag is **`v4-c2048` at
decode, whole-call** — the promotion, weakest estimator clearing by 0.0045 µs.
It changes no decision here: `c2048` is adopted on its **prefill** evidence
(−4.500 µs against a 0.040 µs floor), where nothing is marginal.

## Carried items and follow-ups, with routes

| item | route |
| --- | --- |
| **`_pick_chunk` small-chunk cliff.** `_pick_chunk` returns the largest divisor of the slice length ≤ 2048; over 49 608 reachable `(b,n,k)`, **6 208 (12.5 %) resolve to < 128 words** (e.g. `slice_len = 2096 = 16 × 131`, 131 prime → chunk 16 → 131 grid steps of 16 words). No production or frozen-suite shape reaches it, so **it is not covered by the correctness job**. The file was deliberately **not** changed: a fallback selector would add a path no job has executed, and this md5 is already gated in a submitted job and pre-registered for probe 24. | `/write-tpu-kernels` follow-up, or extend `xgate.py:54`'s `SHAPES` |
| **The VMEM guard is permissive by −528 words** at extreme `n`, because it prices source allocas and not Mosaic's tiling padding. Unreachable through `_pick_partition`, so it is an early legible error rather than a safety property. | documentation, or tighten with a measured padding factor |
| **`v11` is superseded but not worthless** — it retiles the *output* emit, `v12` the *input*. Whether they compose was never measured. | a two-arm probe, if the composed number lands short |
| **`MAX_SLICE_WORDS` 40960 is P-neutral** and would free 96 KiB of tile memory. | a separate change with its own A/B |
| **`v12b` — declare `keys_hbm` `f32` as well.** `v12` converts the *scores* operand only; the keys operand still round-trips through the same kind of caller-side convert. Never built, never measured. | `/write-tpu-kernels` follow-up; one-line change, reuses probe 24's harness unchanged |
| **`v2`'s `PARKED` `cvt` re-measurement.** `v2` collapses the 16-deep fold to a copy, which changes the operand mix feeding the convert; the parked question is whether the `cvt` is now on the critical path. Not measured, and `v2` was adopted without it. | a probe against `probes/23-ranking`'s pool; does not gate the composed change |
| **`v10`'s two parked observations** — pivot-bucket compaction (untried; `v10`'s regression was the *sample-and-count* arm, not this), and that `REFINEMENT_LEVELS` is **3** where the data implies **1.740**. The second is a static mismatch between a compile-time constant and the distribution it was sized for. | `/write-tpu-kernels` follow-up; the `REFINEMENT_LEVELS` one is arithmetic before it is a probe |
| **⛔ The frozen test's `all_negative` comment at `:158-163` is false.** It claims no other `kind` covers negative winners, so a kernel dropping the involution entirely would still pass — but `strict_desc` is `-np.arange(n)`, making 2 047 of 2 048 winners strictly negative at a *running* `P=1` shape. The comment misstates the suite's own coverage, and it misled this exercise into nearly adding two redundant ids. Instruments were READ-ONLY here, so it was not edited. | a comment-only fix to `instruments/test_dsv4_sparsecore_topk.py`, outside a measurement window |
| **`references/` promotions** — `08-sparsecore.md`: offline bundle dumping with `bundle_stats.py` (zero cluster time), and that **a bundle count measures issue cost and is blind to stall cost** (`addupdate_scatter` is 5.2× a popcount in time but 0.82× in bundles). `03-pipelining.md`: **`emit_pipeline` lowers marginal per-descriptor cost**, 248.4 ns vs 517.5 ns — the reference currently implies the opposite sign. | reference edits; the pipelining one contradicts what is written today and should land first |
| **⛔ `references/08-sparsecore.md` contradicts itself** — `:541` and `:646` disagree. Whichever is wrong, one of them is being read by someone. | resolve against a probe, then edit |
| **D7 — SC↔TC overlap — remains `OPEN`.** Never closed at any step; it bounds how much of any SparseCore win reaches the step, and this exercise's ~1.2 %-of-decode ceiling is quoted without it. | `xprof-analyzer` on a decode capture, or a dedicated probe |

## Reproducing the static evidence

Everything in the `static-compile` column costs zero cluster time:
`final/aot_final.py` (host AOT against `topologies.get_topology_desc("tpu7x:2x2x1")`,
**one arm per process** — `jax.jit` caches on `(function object, avals)`),
`final/bundle_stats.py`, `final/vmemctl.py`. SparseCore dump flags go in
`LIBTPU_INIT_ARGS` **before** `import jax`; XLA rejects them in `XLA_FLAGS`.
`*sparse_core_specific_metadata.txt` is ~90 lines against 19 dump files and is
the cheaper first read.

**Nothing is committed.** The working tree is intentionally dirty; the new
paths are `kernel_work/sparsecore-topk/final/`,
`kernel_work/sparsecore-topk/probes/25-final-correctness/`, and this file.
