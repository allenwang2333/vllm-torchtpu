# `v10` — sample-and-count (design.md §9 Plan E) — **built, and modelled a LOSS**

> Sign convention (BRIEF §0, binding): every percentage is `(new − old)/old` of **time**.
> Negative is faster.

**Headline.** The plan as written is built, compiles, and is exact. Its `Δ time %` is
**+11.99% harness / +14.61% production** — a loss, against a predicted −9.7% (Plan E §5) and
−12.2% / −14.9% (the `v10` version-log block). The open term the dispatch assigned me — the pass
count — is now **closed**, and it closes against the node. So is the term that could have rescued
it: the marginal cost of an extra threshold, which I measured from the emitted bundles at
**5.59 µs**, not the 0.111 µs "free" that probes/15's null would have suggested.

**Recommendation: do not spend a timing arm on `v10` at any arity.** §7 names the one 4-minute
microbench that would falsify me if I am wrong.

---

## 1. What I built

A second kernel body inside the same Pallas kernel, selected by a compile-time flag so the
incumbent stays measurable in the same job. The sample body replaces the incumbent's dense 256-bucket
L0 `addupdate_scatter` **and** its three 8-bit refinement levels with (a) one L0 histogram of every
16th vector, giving an *estimated* bracket for the k-th key, and (b) a correction loop of **count
passes** — a compare and a lane population-count per vector, no memory scatter — that walks the
estimate until it brackets the k-th key exactly. The emit pass is the incumbent's, reached through a
one-line `boundary(key, valid)` closure that is the only thing separating the two arms inside the
emit walk. The correction loop is a `lax.while_loop` whose exit condition is a *proof* of
`count(> thr) ≤ k ≤ count(≥ thr)`, never an iteration budget — Plan E risk row (c).

Diff against the incumbent (`sparsecore_topk.py`, md5 `44353b34…`, commit `b11b6eb6`):

| | |
| --- | --- |
| `sparsecore_topk.py` | **16 hunks, +215 / −105 lines** → md5 changes, 732 lines. One hunk is the whole of it: `@@ -164,107 +171,182 @@` (+171 / −96) splits `wave_body` into `if sample_and_count: … else: …`. The other 15 are 2-line flag plumbing through `_topk_body` → `_topk_kernel{,_keys}` → `_sc_topk_direct` → `sparsecore_topk`, at 4 call sites. |
| `sparsecore_topk_sample.py` | **250 lines, NEW.** |

Artifacts in this folder: `pass_count_sim.py` + `pass_count_sim.out.txt` (the pass-count sim, the
marginal-cost sensitivity, and the gate's own kinds), `bundle_stats.py` + `bundle_stats.out.txt`,
`aot.py` + `aot/<arm>/` (≈30 dump files per arm), `vmem_sweep.py` + `vmem_sweep.out.txt`,
`split_check.py` + `split_check.out.txt` (exhaustive int32 overflow check on `_split`). All run on
the host; **no pod job was submitted.**

---

## 2. The file plan, scored

Plan E §4 has one row.

| Plan's row | What I produced | Divergence |
| --- | --- | --- |
| `sparsecore_topk_sample.py` — "the sample pass, the count loop, the flag", ~150 NEW lines; **must not hold** a silent early exit from the correction loop | `sparsecore_topk_sample.py`, **250 lines**: `sample_target_rank`, `_split`, `make_count_pass`, `sample_bracket`, `_absorb`, `correct`. No early exit — `correct` is a `lax.while_loop` on `ge \| (hi == lo + 1)`, and the module docstring states plainly that there is no lowerable budget in the file | **250 vs ~150 lines.** ~40 of the excess is the docstring and the dimension glossary; the rest is two things the plan did not cost: `_split`'s int32-overflow-safe form (`hi − lo` reaches 2³² − 1 on the first pass after a failed bracket) and the arity-2 **seed-round** structure, which must consume `lo0` and `hi0` in *successive* passes because one count pass at arity 2 carries only one threshold. An earlier version put both in one pass and silently discarded `hi0`, bisecting from ~2³⁰. |
| — (plan says nothing about it) | **`the flag` did not fit in the new file.** Plan E puts "the flag" in `sparsecore_topk_sample.py`, but the flag has to reach `wave_body`, which lives in `sparsecore_topk.py`. It is threaded through the incumbent file instead — that is the 15 small hunks in §1. | Unavoidable; the plan's file table was wrong about where a flag can live. |

---

## 3. `code-style.md` checklist, item by item

Scored on **the two files this variant owns**. Where an item is inherited unchanged from the
incumbent I say so, because "fix it" and "keep the A/B honest" are in direct conflict — editing the
incumbent's style in a variant folder changes the subject of the measurement.

1. **Dimension glossary at the top; uppercase logical, `t`-tile, `n`-grid; padded extents own
   letter.** ✅ for `sparsecore_topk_sample.py` — added one naming `B N K L LANES nw ns NB A`, with
   `N` explicitly the *padded* capacity and `L` the live length, which is exactly the padded/live
   pair this kernel confuses most easily. ❌ **inherited** for `sparsecore_topk.py`: the incumbent
   has none and I did not add one, because the diff is already the largest of any variant and a
   glossary hunk would bury the mechanism.
2. **Every ref parameter has a trailing `# [shape] dtype` comment, block vs whole array.** ✅ —
   `sample_bracket`'s twelve parameters are all annotated, including the five closures by their
   signature (`(i) -> (key i32[LANES], valid bool[LANES], j i32[LANES])`), which is the thing a
   reader of this file cannot otherwise recover. Inherited annotations on `_topk_body` are intact.
3. **Extents derived from the refs, not the config; no bare `x.shape[2]`.** ✅ — `samp_chunks` is
   derived from `walk_chunks`, the ones-vector for the scatter is `jnp.ones_like(lane_iota)` (it was
   a bare `jnp.ones((16,), …)` and I fixed it), and `sample_target_rank` derives the sample rank from
   `k` and `SAMPLE_STRIDE` alone.
4. **jaxtyping on the public wrapper, bare refs inside.** ❌ **inherited** — the incumbent's
   `sparsecore_topk` uses `jax.Array` with trailing shape comments (`# f32[b, n]`), not jaxtyping. I
   matched the surrounding convention rather than introducing a second one in the same signature.
5. **Kernel signature carries `# Inputs. / # Outputs. / # Scratch.`** ✅ — added to
   `sample_bracket`, which is the only new signature. Inherited on `_topk_body`.
6. **Kernel body in order: indices → shapes → semaphores → init guard → loads → compute → masked
   store → epilogue.** ✅ — the sample branch of `wave_body` keeps the incumbent's order exactly:
   `pl.when(i_hi > 0)` DMA guard → `load_keys` → `cvt_count` → `sample_bracket` → `correct` →
   `my_take` → `boundary` → the unchanged emit.
7. **Single characters only for loop indices, annotated math tensors, uppercase extents, `d<x>`
   gradients; grid indices spelled out.** ✅ — the new file's single-letter names are `i` (loop
   index), `t`/`c` (a threshold and its count, both annotated at the point of use in `_absorb`),
   `w`/`st` in the seed-split. No single-letter grid index.
8. **Memory space in the ref name wherever two spaces share a signature; never `_ptr`.** ✅ —
   `priv_vmem` carries its space and is the only ref the new file touches; `resident_vmem` and
   `scores_hbm` are the incumbent's and are correctly suffixed.
9. **Comments name the compromise, not the operation. One-line module docstring.** ✅ on the
   compromises — every comment in the new file names one (why the sample is *systematic over
   vectors* rather than a prefix; why `_split` shifts before subtracting; why arity 2 needs two seed
   rounds; why the loop is `while` and not `fori`). ❌ on "one-line": the module docstring is ~40
   lines. Deliberate and I would not shorten it — this file's whole justification is a rate measured
   in another directory, and the docstring is where a reader learns the rate is contested.
10. **Unsupported features raise `NotImplementedError`, not `ValueError`.** ✅ vacuously — the new
    file raises nothing. The one new validation I added is `_count_arity not in (2,4,8,16)`, which
    is a **bad argument**, so `ValueError` is correct there and `NotImplementedError` would be wrong.
11. **Four files: `api.py` / `base.py` / `pallas_mosaic_tpu.py` / `…_kernel.py`.** ❌ **inherited** —
    the incumbent is one 700-line file containing the public API, the config logic and the
    `pl.kernel` call. Plan E's own §4 asks for a *second* file beside it, which is what exists.
    Restructuring into four would make the variant unmeasurable against the incumbent.
12. **Where it splits further, the boundary is what the module may touch: arrays / refs / DMA /
    ints.** ✅ — and this is the one structural thing I am pleased with. `sparsecore_topk_sample.py`
    touches **no DMA and allocates nothing**; it receives refs and closures and returns int32
    scalars. That boundary is why §5's VMEM equality is provable by construction.
13. **Config is a frozen, kw_only, slots pydantic dataclass; cross-field checks in
    `__post_init__`.** ❌ **inherited, and not applicable** — this kernel has no config class; its
    two knobs are `static_argnames` on the jitted wrapper. My two new knobs follow that convention.
14. **Tested against the XLA `_fwd` in `base.py`, tolerance visible at the call site, runtime
    skips.** ✅ — and this is a **blocking class**, so plainly: the reference is
    `instruments/test_dsv4_sparsecore_topk.py`, which compares against `np.argpartition` on the host
    and the tolerance is **zero** — `max |score diff| == 0.0`, written at the assertion, not in a
    constant. It compares selected **scores as a multiset**, never indices. See §4 and §8 for the
    exactness claim this gate does and does not cover.
15. **Every f32 contraction in that `_fwd` pins `Precision.HIGHEST`.** ✅ vacuously — the reference
    is a host NumPy sort; there is no f32 contraction anywhere in this kernel or its reference.
16. **Hardware constants from `pltpu.get_tpu_info()`, never hardcoded.** ❌ **inherited** — and this
    is the **second blocking class**, so plainly: `sparsecore_topk.py:52` has `LANES = 16` and
    `MAX_SLICE_WORDS = 64 * 1024`, both hardcoded, both the incumbent's. **My new file adds no new
    hardcoded hardware constant**: `SAMPLE_STRIDE = 16` is an algorithmic sampling rate (every 16th
    *vector*), dimensionless with respect to lane width, and the one place I needed a lane extent I
    take it from `lane_iota`.

**Blocking verdict.** Item 16 is ❌ on the file, inherited. It does **not** block this variant
relative to the incumbent — the constant is identical in both arms and cannot move the A/B — but I
am not going to record it as a pass. Route in §9. Item 14 is ✅.

---

## 4. Plan E §7 risk table, with an outcome on every row

| risk | outcome |
| --- | --- |
| **a count pass is not cheaper than a histogram pass** (threatens speed) | **happened — in the currency that turned out to matter.** probes/15 measured a count pass at 0.191× the dense scatter *in time*, and that stands. But in the same kernel at the production shape, the emitted innermost loop bodies are: incumbent hist0 **9 bundles / 4 vectors**, incumbent refinement **14 / 4**, v10 count (1 threshold) **11 / 4**. A count pass **issues 1.22× the L0 histogram's bundles**. Both facts are true — the scatter *stalls* on tilespmem, it does not *issue* — but it means the 5.2× time gap is a memory effect. Fallback taken: none available; this is the root's premise. Evidence: `bundle_stats.out.txt`. |
| **the correction loop needs many more iterations than 2.5 on the real score distribution** (threatens speed) | **happened, and it is what kills the node.** At the real prefill shape and the real ρ = 0.40005, arity 2 needs **P_eff = 12.999** crit-weighted passes (mean 9.665, walking-rows mean 12.937, p50 11.8, **p90 16.2, p99 21.7, max 27.5**) against a break-even of **9.48** on my accounting and **8.0** on probes/15's. Decode is worse: **P_eff = 18.772**. The plan's fallback ("widen the sample to 1/8") does not reach — see §6 for why the arithmetic forbids it. Evidence: `pass_count_sim.out.txt`. |
| ⚠ **the loop is quietly bounded for speed and the kernel silently becomes approximate** (threatens **correctness**) | **did not happen — by construction, and the reading that shows it is in the emitted code.** There is no iteration count anywhere in `correct`: the exit is `lax.while_loop(cond=not(skip \| ge \| hi == lo+1))`, and `_split` guarantees the bracket shrinks by ≥ 1 key per pass, so termination is a proof of `count(> thr) ≤ k ≤ count(≥ thr)`. The frozen assertion that would fire if I were wrong is **`_mismatch`'s `max \|score diff\| == 0.0` in `test_exact_topk_full_rows`** (and `test_exact_topk_ragged_rows_straddle_partition_boundaries` for the ragged case) — filling a quota out of a *band* of distinct values returns a strictly worse multiset of scores, which that comparison sees with zero tolerance. `test_comparison_detects_a_wrong_index` is the meta-test proving `_mismatch` can fire at all. ⚠ **This row is discharged by construction and by the gate's design, NOT by a green run** — per BRIEF §6 I do not run the pod test, and §8 hands the runner the exact invocation. |

**One risk the plan did not list, raised by the coordinator mid-build, checked and clear:**
`design.md`'s `cnt_max = 4096` "safe bound" is falsified — the verbatim generators put max `cnt` at
`strict_desc` 1536, `random` 3054, **`heavy_ties` 5526**, `all_negative` 7089, so a buffer built to
4096 overflows. It **cannot touch `v10`**: nothing in either file sizes a buffer, a scratch or a
bound from `cnt`, from a survivor count, or from an assumed maximum population — every extent is
`slice_len`, `NUM_BUCKETS`, `k` or `b_pad`, and `sparsecore_topk_sample.py` allocates nothing at all
(§3 item 12). Confirmed behaviourally, not by reading: §5's allocation sweep puts all three arms at
an identical threshold. ⚠ Note for anyone carrying this forward — **the gate is not blind here**, an
earlier claim to the contrary was retracted: `heavy_ties` reaches 5526 and **does** run at P=1
(`test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]`, id confirmed by collection), so a
4096-sized bound goes red by 1.35×.

---

## 5. `static-compile` evidence

All four arms AOT-compile on the host with **no TPU attached**, against
`topologies.get_topology_desc("tpu7x:2x2x1")`. One arm per process — `jax.jit` caches the inner
pjit jaxpr on `(function object, avals)`, so two arms selected in one interpreter can compare an arm
against itself.

**Arm activation is asserted from the kernel NAME carried into the optimized HLO, not from the flag
passed in** (`_sc_topk_direct` appends `_sc{arity}` on the sample path only):

| arm | kernel name in HLO | active | ENTRY instrs | TEC bundles | Δ vs base |
| --- | --- | --- | ---: | ---: | ---: |
| `base` | `sc_topk_direct_b1024_n10240_k2048` | ✅ | 13 | 1168 | — |
| `sc2` | `…_k2048_sc2` | ✅ | 13 | **807** | **−361** |
| `sc4` | `…_k2048_sc4` | ✅ | 13 | 858 | −310 |
| `sc16` | `…_k2048_sc16` | ✅ | 13 | 1202 | +34 |

A second, independent activation check that does not rely on a name at all: the sample histogram's
emitted loads step by **`0x100` = 16 vectors** (`sc2`, `.LBB2_36`) where the incumbent's step by
`0x10` = 1 (`base`, `.LBB2_88`). `SAMPLE_STRIDE` is visible in the machine code.

**Controls — a matched pair.** The mandatory negative control (8 MiB VMEM scratch DMA'd to HBM) is
refused by the **SparseCore allocator**, not the MLIR layer:

> `E3000: CompileTimeSparseCoreAllocationFailure: current allocation offset upper bound (33554431
> words) exceeds the legitimate user allocatable offset upper bound (2097151 words) in memory space
> spmem when allocating 33554432 words`

Paired with it, a **positive** control the brief did not ask for: the byte-identical kernel with a
**4 KiB** scratch, same DMA form, same copy length, which **compiles**. The two differ in exactly
one integer, so allocation size is the only variable between compile and refusal. Without it,
"8 MiB fails" would be equally consistent with the DMA form being broken.

**What changed against the incumbent, from the dumps:**

- **The refinement machinery is gone, and the emitted code proves it region by region.** Dense
  scatters `vst.idx.add.s32.msk` **40 → 13** (four histogram passes → one). Innermost `HIST-scatter`
  loops **8 → 2** (main + remainder for hist0 + 3 refinement levels → main + remainder for the
  sample). `vshrl.u32` **18 → 0** — the `key >> shift` radix-field extraction has no remaining
  callers. `SWEEP-scan` bundles **214 → 131**, the 1.570 crit levels of V4 sweeps that the
  refinement levels were driving.
- **§8b — the untouched regions did not move.** The two emit innermost loops are **bundle-identical
  in all four arms** (26 and 22 bundles), even though the sample arm's boundary predicate drops a
  `vshra` per lane: the shift was already riding in a spare issue slot. The `cvt`/sweep loops
  `.LBB2_26/.LBB2_28/.LBB2_30` are 49/39/21 bundles in every arm. So the saving is not being
  absorbed by an unpriced neighbour — the failure mode that killed `v5` and `v6` is **not** what is
  happening here.
- **Private VMEM: unchanged, and measured, not read off a blind report.** ⚠ XLA's
  `*-memory-usage-report.txt` and `after_codegen.txt` cannot see Mosaic SparseCore VMEM; nothing
  here uses them. Two instruments instead. (a) Mosaic's own `memref.alloca` list in the unoptimized
  MLIR: `[10240, 4112, 272, 2064, 1024]` words = **69 KiB tile_spmem, byte-identical in all four
  arms**. (b) Behavioural, by allocation failure — bisecting the largest `n` that compiles at
  `b=1024, k=2048` gives **122880 compiles / 123904 refused with E3000/spmem, identical for `base`,
  `sc2` and `sc16`**. v10's VMEM claim is an *equality* and it holds.
- **Spills: clean at arity 2 and 4, present at 16.** Scalar spill traffic `sld` **0 / 2 / 5 / 52**
  and `sst` **1 / 2 / 4 / 29** across `base` / `sc2` / `sc4` / `sc16`. Fifteen live thresholds and
  fifteen live counters do not fit the scalar file. This is why arity 16's per-threshold cost is
  measurably *sublinear* and simultaneously catastrophic in absolute terms.

**⛔ Everything above is `static-compile`. Not one of these is a time.**

---

## 6. My prediction, restated as a number to be scored

**The design's prediction does not survive, and I am not restating it — I am replacing it.**

| | harness `Δ time %` | production `Δ time %` |
| --- | ---: | ---: |
| Plan E §5 predicted | −9.7% | — |
| `v10` version-log predicted (my dispatch) | −12.2% | −14.9% |
| **What I predict for the plan as written (arity 2)** | **+11.99%** | **+14.61%** |
| best arm of this root (arity 2, Illinois interpolation) | +3.25% | +3.97% |
| arity 4 / 8 / 16 | +27.4% / +61.2% / +120.8% | +33.4% / +74.6% / +147.2% |

**Mechanism, and it is information-theoretic rather than an implementation detail.** A count pass
costs 6.48 µs ρ-weighted and yields **1 bit**. A refinement histogram costs 20.08 µs for 1.570 crit
levels — 12.79 µs per level — and yields **8 bits**. Per bit: **1.60 µs for the scatter against
6.48 µs for the count.** The scatter is **4.05× more bit-efficient** *despite* being 5.2× slower per
pass, because it is 8× more informative. The production L0 pivot bucket holds ~1071 elements, so
~10 bits below L0 have to be resolved by any mechanism; the incumbent buys them 8 at a time.

**The arithmetic, at the harness denominator 189.98 µs.** Removed: hist0 33.91 + refinement hist
20.08 + all-but-one level of sweeps 15.63 = **69.62**. Added back: the 1/16 sample scatter 2.12 +
one level of sweeps 6.08 = **8.20**. Budget for the whole correction loop: **61.42 µs**, i.e.
**9.48 count passes**. The measured requirement is **12.999**.

**The pass count — the open term my dispatch assigned me — is now closed.** Method and its
assumption, stated rather than chosen silently: I could not source a real score *distribution*, so
I reused **`probes/13/level_sim.py`'s `make_kv`/`make_scores`**, which reproduce
`streamindex_topk.py:326-338` from source and were validated in probes/13 against probes/05's
published table; `alpha=0.0`, `signed_w=True`, ragged lengths from `prefill_lengths` (E/n = 0.40005)
and `decode_lengths` (E/n = 0.850), 4 reps, seed 20260910. **The named assumption is that this
generator's score distribution is production's.** The simulation self-validates: run at ARITY=256
from the full 32-bit range it *becomes* the incumbent's radix walk, and reproduces `level_sim`'s
refinement-level count **exactly** — `mean(walking)` 1.740 vs 1.740 (prefill), 1.832 vs 1.832
(decode). (Its `crit` differs, 1.664 vs 1.570, because I weight `_crit(passes × eff)/_crit(eff)`
where `level_sim` takes `_crit(levels)`; that is a weighting-definition difference, not a machinery
one, and I use my own definition consistently throughout.)

⚠ **On "transcribe the generator verbatim, never approximate it".** I adopted the rule where it
applies and I should be clear about where it cannot. For the **gate's** kinds it applies exactly, and
`pass_count_sim.py` now goes one better than transcription: it `importlib`-loads
`instruments/test_dsv4_sparsecore_topk.py` and calls `_scores` directly, so no transcription error
is possible. For the **production** distribution there is nothing to transcribe — production has no
score generator, it has a model, and `make_scores` is a reimplementation of the score *pipeline* at
`streamindex_topk.py:326-338`, line-referenced in its docstring. That is a reconstruction, and it
stays the load-bearing assumption in §10 no matter how carefully it was done.

**The distribution is not the escape hatch.** At ρ=1 the requirement gets *worse*, not better, on
every source I have. probes/05's synthetics (`P_eff`): `random` 14.25, `uniform` 16.13,
`heavy_ties` 26.00. The **frozen gate's own kinds**, obtained by importing the instrument rather
than retyping it: `heavy_ties` 26.00, `random` 14.25, `strict_desc` 1.00. There is no plausible
input on which arity 2 clears 9.48, and the only one that does clear it — `strict_desc`, a perfect
linear ramp — is not a distribution production produces.

⚠ **A name collision worth stating, since both sets appear above.** probes/05's `heavy_ties` is
`linspace(-2, 2, 17)` and the frozen test's is `integers(0, 17)` — 17 distinct values either way,
but one straddles zero and the other does not, so they land in different L0 buckets. They happen to
agree at 26.00 passes; that is a coincidence of the bisection depth, not the same generator. Both
are labelled by source everywhere in `pass_count_sim.out.txt`.

**The second open term — the marginal cost of an extra threshold — is also now closed, from the
emitted bundles, and it closes the escape route through higher arity.** Before this build it was
worth a **55-point swing**: at probes/15's null (0.111 µs, "free") arity 16 would have priced at
−17.8%; at my pre-build guess (2.35 µs) it priced at +38.8%. Measured in the **production kernel at
the production shape** — not a toy, per the coordinator's warning that probes/17 undercounted by
3.5× — the innermost count body at a fixed unroll of 4 is **11 bundles for 1 threshold and 30 for
3**, so `b(m) = 1.5 + 9.5m` and one extra threshold costs **0.864 of a whole one-threshold pass =
5.59 µs**. The reason it is *not* free is the same reason the count pass looked cheap: the count
pass is **issue-bound** (probes/15 put it at the read-only walk floor), so an added threshold buys no
free slot — it is pure issue. The full sensitivity table is in `pass_count_sim.out.txt`.

⚠ **Arity 2 — the plan as written — contains no extra threshold, so its +11.99% is invariant to
that entire table.** It is the firmest number here.

**Why the plan's own fallback ("widen the sample to 1/8") cannot reach.** To get P_eff under 9.48 at
arity 2 the sample must deliver ~3.5 more bits of bracket, i.e. ~2048–4096 buckets instead of 256.
The sweep is linear in `NUM_BUCKETS` (`zero_priv` = 9.238 µs, measured, `j-4e0df30a`): one level of
sweeps at 256 buckets is 6.08 µs, so at 4096 it is **~97 µs** — which alone exceeds the entire
61.42 µs budget. Widening the *sampling rate* from 1/16 to 1/8 changes the bracket's variance, not
its width, and buys no bits at all.

---

## 7. How the measurer runs me

**My recommendation is that you do not.** Every arm of this root is a modelled loss and the two
terms that could have rescued it are both now closed against it. Spending a block on `v10` buys a
confirmation.

**If you want one measurement anyway, buy the falsifier, not the kernel.** The cheapest thing that
can prove me wrong is a **probes/15-style microbench of a bare count pass at m ∈ {1, 4, 16}
thresholds over the same `L`**, three arms in one block, ~4 minutes. It measures `MARGINAL_PER_THRESH`
directly. If the measured marginal comes in below **~0.8 µs**, arity 16 turns positive and the root
is alive; anything above that and every arm of §6 stands. My static answer is 5.59 µs and I would be
astonished by 0.8.

Should you run the kernel anyway, the invocation:

- **Map** `variants/v10-sample-and-count/sparsecore_topk.py` **and**
  `variants/v10-sample-and-count/sparsecore_topk_sample.py` side by side over
  `src/vllm_torchtpu/kernels/deepseek_v4/`. **Both files, or the import falls back and you
  measure the incumbent.** Gate the md5 of *both* in-job — a docker tag is mutable.
- **Arms**, all four in one process (the flags are `static_argnames`, so one process compiles all
  four and the A/B shares a pool):
  `_sample_and_count=False` (control) · `=True, _count_arity=2` (the plan as written) ·
  `=True, _count_arity=4` · `=True, _count_arity=16`.
- **Arm activation is checkable in the profile, not just in the recipe**: the Pallas scope is
  `sc_topk_direct_b{b}_n{n}_k{k}` for the control and `…_sc{arity}` for the sample arms. Reject any
  arm whose scope name does not carry its own suffix.
- **Shapes**: `b ∈ {64, 1024}`, `n = 10240`, `k = 2048`, `--ctx 9216`. ⚠ Not the bench's default
  `--ctx 132096`.
- **`_count_arity` deserves its own rival arms in the same block** — it is my one compile-time
  constant and it spans +12% to +121% in the model, so it is the arm most likely to embarrass the
  model if the model is wrong.
- **No checksum arm needed.** v10 changes nothing about what the DMA lands: the score transfer,
  `slice_len`, and the resident VMEM layout are identical (§5).
- ⚠ **Do not report the count region's own cost as the result** (BRIEF §4). v10 moves work between
  regions by design; its number is whole-kernel time at the production shapes.

---

## 8. How the correctness runner runs me

```
pytest -q tests/kernels/test_dsv4_sparsecore_topk.py
```

unmodified, with `--map-dir` placing **both** `sparsecore_topk.py` and `sparsecore_topk_sample.py`
over `src/vllm_torchtpu/kernels/deepseek_v4/`.

⚠ **One thing you must set, and here is why.** The frozen test calls `sparsecore_topk(...)` with no
arm flag, so mapping my file alone runs **all 68 tests against the incumbent branch** and reports a
green that means nothing. Rather than touch the frozen test (BRIEF §2), the mapped copy reads its
*default* from the environment, once, at import:

```
SCTOPK_SAMPLE_AND_COUNT=1  SCTOPK_COUNT_ARITY=2   pytest -q tests/kernels/test_dsv4_sparsecore_topk.py
```

It is a default, never an override — an explicit kwarg still wins, so the timing harness in §7 is
unaffected. **Do not treat the env var as proof the arm ran**; the proof is the scope name carrying
`_sc2`. Please run both arities **2 and 16**: they are different code paths (`arity == 2` takes the
two-seed-round branch, `arity >= 4` the single-round branch) and 16 additionally spills.

**Which of the 68 ids exercise the change hardest.** Ranked by the *measured* number of count
passes each drives, not by how hard the kind sounds — I ran my simulator on the gate's own `_scores`
by **importing the instrument** rather than retyping it, and confirmed every id below with
`pytest --collect-only` (which needs no TPU: 68 collected in 1.1 s). Note the id field order is
`[kind-shape]`, not `[shape-kind]`.

| id | count passes (arity 2) | why it matters |
| --- | ---: | --- |
| `test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]` | **26.0** | **The single most important id.** 17 distinct values in a row of 10240, so the k-th and (k+1)-th scores are equal in every row: the emit must fill its quota out of *exactly one* value, and the loop runs to its deepest. It is also the deepest correction loop of anything in the suite. |
| `test_exact_topk_full_rows[random-p1-b1024-n10240]`, `[random-p1-b64-n10240]` | 13.4 | The continuous case, and the closest of the three to production. Both P=1 shapes, which are the only ones where `sample_bracket` sees a full row rather than a slice. |
| `test_exact_topk_ragged_rows_straddle_partition_boundaries[p1-b1024-n10240]` | — | Ragged `i_hi` is where `samp_chunks = ceil(walk_chunks/16)` can round to zero on a short row. The only id running at anything like the production ρ. |
| `test_all_neg_inf_rows_return_no_selection[p1-b64-n10240]`, `[p1-b1024-n10240]` | 0 | The `skip` path (`keep_all \| i_hi <= 0`) must short-circuit the whole correction loop. This is the id that takes it. |
| `test_exact_topk_full_rows[strict_desc-p1-b1024-n10240]` | **1.0** | ⚠ Listed to *warn* you, not to rely on it — see below. |

⚠ **Two corrections to the coverage picture I was handed, both from my own instrument, both of which
change what this gate proves about `v10`.**

**(1) P=1 *does* cover negative winners — so that gap is not real.** I was told `all_negative` is the
only kind whose winners are negative and that it is `P > 1` only. The first half is false:
`strict_desc` is `-np.arange(n)`, so its top-2048 winners are `{0, −1, …, −2047}` — **2047 of 2048
are strictly negative** — and `test_exact_topk_full_rows[strict_desc-p1-b1024-n10240]` exists and
runs. `monotone_key`'s involution is exercised at the P=1 production shape.

**(2) The gap that *is* real is one nobody named, and it is the one `v10` actually creates.** I
surveyed the bracket `correct` holds after its seed round, over 256 rows of each source:

| source | post-seed bracket ≥ 2³¹ wide | `lo == INT32_MIN` | spans zero | widest |
| --- | ---: | ---: | ---: | ---: |
| gate `strict_desc` | — | — | — | converges in the seed round, 0 splits |
| gate `random` | 0 | 0 | 0 | 24.0 bits |
| gate `heavy_ties` | 0 | 0 | 0 | 24.0 bits |
| **production prefill** | yes | **83 / 256 (32%)** | 22 / 256 | **31.6 bits** |
| **production decode** | yes | 5 / 64 | 4 / 64 | **31.6 bits** |

Every gate kind resolves cleanly into one 24-bit L0 bucket, so `_split` **never sees a bracket wider
than 2²⁴ under any id at any shape** — while a third of production prefill rows leave the lower edge
pinned at `INT32_MIN`, which is exactly the `hi − lo > 2³¹` wrap case `_split`'s shift form exists
for. **A green 68/68 does not clear it**, and `strict_desc` — the one negative-winner id — is the
*least* useful here because it converges in a single seed pass and never enters the loop at all.

**I closed it on the host rather than by editing the frozen test.** `split_check.py` evaluates the
same `jnp` expression the kernel compiles, on jax's CPU backend, over 21,025 int32 bracket pairs ×
4 arities = **84,100 reachable cases**: adversarial corners, all 33 L0 bucket edges, and 20k random
pairs. **0 violations**, widest verified bracket the full **2³² − 1**. The 40 excluded cases are
asserted to be exactly the width-1 pairs, which `correct`'s loop condition
(`not (skip | ge | hi == lo + 1)`) makes unreachable. **Negative control**: the naive
`lo + (hi − lo) // 2` this function exists to avoid violates on **5,263 of 21,035** pairs, so the
check discriminates. Output in `split_check.out.txt`. That is `static`/host evidence, not a pod run
— but it is exhaustive over the property, which no pod run would be.

**What remains genuinely ungated:** `all_negative` at P=1 (`TWO_STAGE_SHAPES` is `P > 1`), i.e. a
*continuous* all-negative distribution rather than `strict_desc`'s ramp. Given (1) and the exhaustive
`_split` check, I judge the residual risk small, but it is not zero and it is not mine to close by
editing the instrument. §10.

**The exactness claim, stated precisely because the measurer is building per-variant gates:**
`v10` is **exact, and bit-identical to the incumbent whenever the k-th score is unique.** The emit
walk, its `vadd.scan.msk` slot assignment and its `store_compressed` are untouched (bundle-identical,
§5), so for an identical selected set the output slot order is identical too. When the boundary score
is **tied**, `v10` may select a different subset of the tied indices — the same multiset of scores in
the same slots, different indices. That is the incumbent's own documented contract ("equal boundary
scores may be broken differently") and the frozen test already gates it the right way, by comparing
selected **scores** as a multiset with zero tolerance. So: **bit-exact on scores, exact-up-to-tie-choice
on indices.** A byte-comparison gate against the incumbent's *indices* would be the wrong gate — it
would also fail the incumbent against itself under a different tie order.

---

## 9. Things I found that are not my candidate

- **`PARKED` — compact the pivot bucket, then count over the compacted set.** The obvious rescue for
  this root: after the L0 histogram, `store_compressed` the ~1071 elements of the pivot bucket into a
  dense array, then each count pass walks 1071 instead of ~4096 elements — 0.261× the cost, ~1.69 µs.
  Thirteen passes then cost ~22 µs. **I priced it and it does not clear**: the arm still needs the
  *full* L0 histogram (a 1/16 sample cannot say which bucket exactly), so it keeps hist0 33.91 and one
  sweep level 6.08 and only deletes refinement hist 20.08 + 9.55 of sweeps — then adds the compaction
  walk (~20 µs, priced off emit's 33.54) plus ~22 µs of counts, landing near **+6.5%**. **Route:** this
  is a negative result on a plausible idea, not a candidate — record it in `design.md` §9 beside Plan E
  so the next reader does not re-derive it. It would need the compaction walk to cost under ~10 µs,
  and that term is unmeasured.
- **A platform fact for `references/08-sparsecore.md`: scheduled SparseCore bundles are readable with
  no TPU, and they separate issue cost from stall cost.** The bundle dump prints **symbolic** branch
  targets (`sbr.rel .LBB2_34-.Ltmp25`), so natural loops — and therefore *bundles per walked vector* —
  can be recovered exactly from a host-only AOT compile, per region, for zero cluster time.
  `bundle_stats.py` in this folder is ~150 lines and generic. **The finding worth writing down is the
  one that killed my node**: `addupdate_scatter` costs 5.2× a popcount in *time* but 0.82× in *issued
  bundles*. A cost model built from either number alone is wrong about the other. **Route:** hand
  `bundle_stats.py` and that sentence to whoever owns `references/08-sparsecore.md`.
- **A non-Pallas observation, unpriced, routed rather than claimed:** `sparsecore_topk.py`'s
  `REFINEMENT_LEVELS` is a fixed 3 levels but the measured requirement is **1.740 mean / 1.664 crit**
  at prefill and **1.832 / 2.132** at decode. The kernel already `pl.when`-skips resolved levels, so
  the third level is mostly a predicated no-op rather than dead time — which is why I am *not*
  claiming a prize. **Route:** it is a one-constant arm and belongs beside Plan D (`v5`, the
  `NUM_BUCKETS` sweep), which is already the cheapest entry in the list and would be measuring the
  same sweep term.
- **A gate-coverage finding, and it is not about `v10`.** The frozen suite's three `full_rows` kinds
  all resolve into a single 24-bit L0 bucket (§8's table), so **no id at any shape ever presents the
  kernel with a key bracket wider than 2²⁴ or one that spans zero**, while a third of production
  prefill rows do both. That is a gap for any candidate whose search is over the *key space* rather
  than over radix digits — the incumbent is digit-wise and is indifferent to it, which is presumably
  why nobody hit it before. **Route:** to whoever owns `test_dsv4_sparsecore_topk.py`, as a request
  for one `kind` with winners straddling zero at a P=1 shape (`all_negative` at P=1 would do it, and
  is already written). ⚠ I did not add it: instruments are read-only in this exercise.
- **A method note, cheap and general.** `pytest --collect-only -q` on this file runs **on the host in
  1.1 s with no TPU** and prints all 68 ids. Reading `parametrize` decorators to work out which ids
  exist got both me and the coordinator wrong this week — the field order alone (`[kind-shape]`, not
  `[shape-kind]`) silently breaks a `-k` selection. **Route:** one line in `write-tpu-kernels`'s
  instrument-freezing step.
- **A style debt with a route, not a shrug (§3 item 16).** `LANES = 16` and
  `MAX_SLICE_WORDS = 64 * 1024` are hardcoded in the incumbent. Both are checkable against
  `pltpu.get_tpu_info()` and neither can be fixed inside a variant folder without changing the subject
  of every A/B in this exercise. **Route:** a standalone one-file cleanup PR against
  `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` *after* step 3 closes and the incumbent
  is no longer a frozen baseline — with the frozen test as its gate, since it should be bit-identical.

---

## 10. Open terms

Everything the verdict still rests on. Per BRIEF §10, a term predicted at zero or credited as a
saving is recorded as `unmodelled`, not as a pass.

| term | status | what it does to the verdict |
| --- | --- | --- |
| **The score distribution** | `assumed` — probes/13's `make_scores` at `alpha=0.0`, named in §6 | This is the assumption the pass count rests on. It is the *most* load-bearing thing here. Mitigated but not closed: the generator reproduces `streamindex_topk.py:326-338` from source, it reproduces `level_sim`'s level count exactly, and three ρ=1 synthetics all make the requirement **worse**. Nothing I ran makes it better. |
| **`design.md` §4a's price sheet** | `modelled`, quoted not re-derived | Every µs in §6 is one of its rows. If the sheet is wrong my percentages move with it. ⚠ Per BRIEF §7, **every probe behind it ran at ρ=1** and nothing has ever run at ρ = 0.40005; the measurer buys that in the first job. |
| **`MARGINAL_PER_THRESH` = 5.59 µs** | `static-compile` — a bundle count, converted to time by assuming the count pass is issue-bound | The conversion is the soft step: bundles → µs assumes issue-boundedness, which probes/15 supports (the count pass sits at the read-only walk floor) but did not directly test. ⚠ **It governs arity 4/8/16 only. Arity 2 — the plan as written — is invariant to it**, which is why the +11.99% is the number I would defend. |
| **The 0.191× count/scatter rate itself** | `measured` (probes/15, `j-78f7f537`), at **ρ=1** | ρ-weighted to 6.477 µs by the price sheet. If the true ρ-weighted rate were **below ~4.7 µs**, arity 2 would clear break-even. That is a 27% error in a measured number — possible, not likely. |
| **`crit` weighting** | `assumed` — I use `_crit(passes × eff)/_crit(eff)`, `level_sim` uses `_crit(levels)` | The two disagree by ~6% on the validation case (1.664 vs 1.570) in *opposite* directions at prefill and decode. Applied consistently, so it cannot flip a sign, but it is worth ±0.5 pass. |
| **Wide / zero-spanning key brackets** | **closed on the host, not on a pod** | `_split`'s `hi − lo > 2³¹` wrap path is taken by 32% of production prefill rows and by **no gate id at any shape** (§8). `split_check.py` clears it exhaustively over 84,100 reachable int32 cases with a firing negative control. It is `static`/host evidence: it proves the *arithmetic*, not that the compiled kernel evaluates it as written. The AOT compile (§5) covers the second half only in that all four arms lower without error. |
| **`all_negative` at a P=1 production shape** | `unchecked`, and it is a correctness gap, not a performance one | `TWO_STAGE_SHAPES` confines `all_negative` to `P > 1` and `:230` omits it, so no id presents a *continuous* all-negative row at P=1. ⚠ Downgraded from how I first wrote it: `strict_desc` **does** cover negative winners at P=1 (2047 of 2048), so the involution is exercised; and the `_split` concern that made this row urgent is now closed by the row above. What is left is narrow. Handed to the correctness runner in §8. |
| **The compaction rescue (§9)** | `unmodelled` — the compaction walk's cost | Priced off emit's 33.54 µs by analogy. It is the term that would decide a `v13`, and nothing has measured it. |
| Whole-kernel absorption (the `v5`/`v6` death) | **closed, not open** | §5: the emit and sweep innermost loops are bundle-identical across all four arms, and private VMEM is identical by two independent instruments. There is no unpriced neighbour soaking up the saving — the node fails on its own arithmetic. |
