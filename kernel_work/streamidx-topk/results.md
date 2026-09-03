# Step 3 — variants built, measured and ranked

**Status: COMPLETE.** Every option is built, measured and ranked; the recommendation is §12. Options A, B, C,
D, E and candidate 0 are all complete and reported in full, as is the recalibrated baseline (§1).
Six options plus the incumbent were measured. Three are recommended (§12), three are rejected on
measurements, and one — the largest prefill number here — is handed to the runner because it is not
a kernel change. Do not quote this file as final until this banner is gone.

The block is `streamidx_topk_jax` (`src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py`),
the GLM-5.2 DSA lightning indexer, on TPU7x. Baseline branch `annyan/streamidx-topk-opt`, incumbent
commit `c3a18c95`.

---

## 0. Terms

Defined here because the numbers below are meaningless without them, and three of them collide with
a term of art elsewhere in the stack.

| term | meaning |
| --- | --- |
| **stage A** | the three **TensorCore** Pallas scoring kernels that write f32 indexer scores to HBM. Not "stage" in the pipeline-parallel sense. |
| **stage B** | `sparsecore_topk`, the exact MSB-first radix select that runs on **SparseCore** and consumes stage A's scores. |
| **incumbent** | the kernel shipping today at `c3a18c95`. Per the skill it is a candidate in this ranking and is allowed to win. |
| **device self time** | per-instruction time from `hlo_stats`, excluding children. Carries **no** host dispatch component and is immune to pod contention. |
| **chain slope** | marginal device time per call, obtained as the slope of R chained calls inside one `jit` joined by a real data dependency. See §1 — this is the only per-call timing method used here that is valid. |
| **`num_bkv`** | the number of KV blocks the grid walks per sequence group. 5 at the prefill shape measured; 1 when `ctx ≤ bkv_sz`. |
| **modelled** | computed from a floor or a cost model, **not measured**. Tagged every time it appears. |
| **Δ sign** | every percentage in this document is a change in **time**: **negative is faster**, positive is a regression. Option C's prefill −20.64% is a win; its decode +59.62% is not. Note that `variants/option-b-pipeline/RESULTS.md` uses the *opposite* sign for Option B's headline (its "+16.4%" is this document's −16.4%); §8 here is stated in this document's convention. |

---

## 1. The finding that governs every number in this exercise

**`time_arm`-style per-call wall clock is `max(device, host)`, not `device + host`.** The harness
enqueues asynchronously and blocks once. A host dispatch floor is therefore a **clamp, not an
addend**, and subtracting one — as several earlier briefs did, at ~35–42 µs — is invalid. It
inflates every small measurement and inflates it *more* the smaller the measurement is.

Device time must come from either the **chain slope** (R calls inside one `jit`, joined by a
dependency the compiler cannot fold — `x*0` gets constant-folded; use the kernel's own
`input_output_aliases`, or `+ jnp.minimum(out[0,0], 0)`) or from **`hlo_stats` device self time**.
Two arms measured on separate chips agreed to 0.02%, so the method is also contention-proof.

**Every number in this exercise that has been re-measured with a method immune to that floor has
come in smaller. Six for six, all explained by the same defect:**

| quantity | of record | re-measured | source of the correction |
| --- | ---: | ---: | --- |
| empty-launch overhead | 5.4 µs | **0.55** | P4 |
| Option D's claimed residual | 44.8 µs | **1.655** | R2 — the 44.8 was ~94% host dispatch |
| stage B, prefill | 481.0 µs | **431.15** | candidate-0 job, `hlo_stats` |
| full op, decode | 221.4 µs | **170.579** | candidate-0 job; chain slope 173.37 agrees to 1.6% |
| full op, prefill | 857.3 µs | **749.538** | candidate-0 job; chain slope 750.14 agrees to **0.08%** |
| **stage A, prefill — the Option C baseline** | **335.1 µs** | **246.58** | Option C, chain slope, r² = 0.9999995 |
| incumbent + `bkv_p=3` @ ctx 8704 | −11.9% (contended) | **−9.46%** | candidate 0 |

The stage-A row is the consequential one: **the ≤280 µs pass bar written for Option C was already
cleared by the incumbent**, so as written the gate discriminated nothing. C passes anyway, on a
like-for-like comparison against a correctly measured incumbent, but its *predicted* call-level
effect drops from −7.9% to **≈−5.9% (modelled)**.

**That correction is now cross-validated by two agents and two independent methods**: Option C's
chain slope gives 246.58, and the candidate-0 job's `hlo_stats` gives 246.243 with its own chain
slope at 247.63 — a 0.56% spread across different harnesses. The composition also closes exactly,
which is the strongest structural check available here: 249.300 + 431.153 + 69.085 = 749.538 at
prefill, and 111.708 + 31.731 + 27.140 = 170.579 at decode.

**One row of this table was itself wrong for one revision, and the correction runs the other way.**
An earlier draft recorded prefill stage B as 335.5 µs from the k-sweep. That figure was measured on
**synthetic N(0,1) scores**. Real indexer scores are all positive, in [9, 760], with std/mean ≈ 0.30
— few distinct exponents, so the MSB-first radix select needs more passes. Fed stage A's real
output, stage B measures **442.56 µs** (chain slope) against `hlo_stats` 431.15. The k-sweep
understates production stage B by **22%**, and its 1.02 ns/element scan coefficient should be
~1.37. Its *structural* conclusion — issue-bound, not output-DMA-bound — is untouched. The lesson
generalizes past this exercise: **a data distribution is part of a benchmark's configuration**, and
N(0,1) is not a neutral default for a radix algorithm.

### What this does to the roofline gate

`measurement_and_design.md`'s prefill gate was already flagged `PARTIALLY FALSIFIED` in
`briefs/ROOFLINE-REVIEW.md` for circularity. With correct denominators it is now **undecidable**,
and it moved in the direction that argues *against* prefill work rather than for it:

| | range across defensible floor treatments | verdict |
| --- | --- | --- |
| **decode**, denominator 170.58 µs | 33–53%, point estimate ~45% | **structural, proceed** — unchanged |
| **prefill**, denominator 749.54 µs | 53.9–92.0% | **undecidable** |

The prefill ceiling was 63.4% in the k-sweep and is now 85.3% — *above* the skill's do-not-write
bar — because that ceiling used a wall-clock denominator (14% too high) and a synthetic-data
stage B (22% too low), two errors compounding in the same direction. No non-circular treatment
*clears* 80% either (the two most defensible are 53.9% and 57.5%). **Do not quote 80.4%, do not
quote 39.9%, and do not quote 26–57%.** The blocker is unchanged and structural: 431 of 749 µs is
SparseCore, and there is no first-principles SparseCore floor — `references/hardware-facts.md` says
plainly that no SparseCore fact in it is measured.

**Where the headroom actually is**, measured minus modelled floor: at decode, stage A 54.3 µs (32%)
and residual 26.3 µs (15%); at prefill, stage B 235.5 µs (31%), with stage A and residual ~55 µs
each. Stage A sits at 51.4% of its floor at decode and 77.9% at prefill — which is the quantitative
reason decode is the better target, and it is not the reason the earlier documents gave.

---

## 1b. The denominator: this op is **4.299% of a GLM-5.2 decode step**

> **SUPERSEDED 2026-09-03 — the 3.80% below is wrong; use 4.299%.** Re-read from the same profile
> with **complete** `hlo_stats` (no `--top-n` truncation) and with the **TensorCore plane only** on
> both ends. The corrected table is §1c. Three things changed: the op scope now includes the
> `conditional` wrapper, the `branch_0_fun` plumbing and the KV-cache write, which the 2.5745 numerator
> did not account for; the denominator rose 67.756 -> 69.492 ms/step; and the prefill share, previously
> only ever estimated as "order 1%", is measured at **3.331%**. The direction is the opposite of what
> was assumed while withdrawing the merged op: this op is **larger** than believed, not smaller.
>
> The trap worth carrying: `hlo_stats` in this profile has a `TPU core type` column with both
> `TensorCore` and `SparseCore` rows, and summing without filtering **double-counts** -- the TC row for
> `sc_topk_direct` is the TensorCore's blocking wait while the SparseCore row is the SparseCore's
> concurrent execution. TC+SC = 81.488 ms/step *exceeds* the 72.039 ms measured module wall; TC-only
> lands at 96.5% of it. Filter to one plane, on numerator and denominator both.

Every percentage in this document is **op-level**. Until 2026-09-03 nothing here established what
share of a real GLM-5.2 step the op is, so none of them could be converted into a model-level
number. They now can, and the conversion is unflattering.

Measured from **`j-9b23bb32-0426-4080-be91`** (recipe `torchtpu-vllm-glm-5.2-v7x-16-profile`,
`Succeeded`, owner `wppark`). Read back from `cdk job recipe`: `zai-org/GLM-5.2-FP8`,
`--data-parallel-size=16 --max-model-len=9216 --max-num-seqs=64 --max-num-batched-tokens=1024
--kv-cache-dtype=fp8`, 2 ray nodes x 8 chips, v7x. **This is the nightly config the microbenchmark
targeted.** Rank 8, `/device:TPU:0`, mean of decode steps 1-4 (step 0 dropped as warm-up),
reproduced independently on rank 12 to within 0.2%.

| | ms/step | % TC busy | % step wall |
| --- | ---: | ---: | ---: |
| stage A, `StreamIdxTC-{d,m}-bq_*-bkvp_2` | 1.8965 | **2.80%** | 2.73% |
| stage B, `sc_topk_direct_b64_n10240_k2048` | 0.6780 | **1.00%** | 0.98% |
| **op total** | **2.5745** | **3.80%** | **3.71%** |
| denominator: TC busy (XLA Ops union) | 67.756 | | |
| denominator: module wall | 69.459 | | |

**`report.md`'s 32.8%-of-decode-step figure is a different model configuration and must not be used
here.** It is AgentX -- `j-a71a648c-3064-471a-a131`, `max_model_len=262144`, `n=65536`, TP=16. At
the nightly shape the score buffer is `n=10240` and the share is 3.80%. The gap is ~9x. This is the
exact failure `CLAUDE.md` warns about: a figure measured at a non-production shape reproduces
faithfully and mispredicts production. It also went unnoticed for the whole exercise because
`cdk job list` defaults to 32 jobs and the GLM-5.2 profile predates that window, so a search for a
denominator concluded there was none.

**What the recommendations in §12 are therefore worth** (arithmetic on the row above; each is the
op-level figure times the op's share, so each is an upper bound that assumes no interaction):

| change | op-level | ~ decode step | ~ prefill step |
| --- | ---: | ---: | ---: |
| candidate 0 (`bkv_p`) | -14.6% of op | **~0.55%** | -- |
| D1 | -10.0% of op | **~0.38%** | -- |
| Option C, prefill-gated | -20.6% of prefill stage A | 0 (gated off) | **~0.06%** |
| Option B + runner change | -24.4% of prefill op | ~0 (wash) | **~0.15%** |
| Option B x C + runner change | **-26.18%** of prefill op (G=4, `j-caf973f4-163d-42e0-89e6`) | ~0 (wash) | **~0.16%** |

> **Sign convention.** This table uses **`-` = faster**, matching Option C's documents. Option B's
> own `RESULTS.md` uses the opposite convention (`+` = faster) throughout. The `+24.4%` written
> there and the `-24.4%` here are the same measurement. Convert on import -- the collision has
> already produced one self-contradicting summary.

Prefill is worse: the op is **order 1%** of a prefill step. The `prefill_heavy` capture contains
exactly one indexer-active step (stage A 1.813 ms, stage B 2.007 ms of 620.5 ms TC busy = 0.62%),
but n=1, 37% of that step is collectives, and its per-call times are 3-4.5x below the microbenchmark
at the nominally identical shape -- so it is an early short-context chunk. Treat prefill as "order
1%", not as 0.62%. `prefill_only` rank 8 has **zero** StreamIdx events.

---

## 1c. The corrected denominator (2026-09-03) — supersedes §1b

Same profile, `j-9b23bb32-0426-4080-be91`, rank 8. Re-read with **complete** `hlo_stats` and
filtered to the **TensorCore plane on both ends**. See the note at the head of §1b for why the
plane filter is not optional. Every share below has its numerator and denominator stated.

**Denominators, TC plane, per model step** (each capture holds 5 decode steps x 2 cores, or 1
prefill step x 2 cores): decode **69.492 ms**, `prefill_heavy` **622.485 ms**, `prefill_only`
**635.463 ms**. Cross-check: TC-only is 96.5-98.3% of the measured `tt_jit_layernorm_...` module
wall time; TC+SC *exceeds* it, which is what exposed the double count.

| component | decode ms/step | % | `prefill_heavy` ms/step | % |
| --- | ---: | ---: | ---: | ---: |
| stage A `StreamIdxTC-{d,m}` | 1.8259 | 2.627% | 1.2491 | 0.201% |
| stage B `sc_topk_direct`, **TC rows** (blocking wait) | 0.6588 | 0.948% | 2.0123 | 0.323% |
| KV-cache write — **relayout round trip** (2 reshapes) | 0 | -- | **13.7780** | **2.213%** |
| KV-cache write — `scatter_fusion` | 0.1863 | 0.268% | 2.8100 | 0.451% |
| `conditional` wrapper | 0.1573 | 0.226% | 0.1438 | 0.023% |
| `branch_0_fun` plumbing | 0.1589 | 0.229% | 0.7427 | 0.119% |
| **op total** | **2.9872** | **4.299%** | **20.7359** | **3.331%** |
| *stage B, SparseCore rows — excluded, never added* | *0.3249* | *--* | *1.0013* | *--* |

Two readings that were not previously available:

- **The op is larger than §1b said, at both phases.** 4.299% not 3.80% at decode; 3.331% not
  "order 1%" at prefill. `report.md`'s 32.8% remains repudiated -- that is AgentX, a different shape.
- **At prefill the op is mostly not the kernel.** The relayout round trip is **68%** of the whole op
  (13.778 / 20.736) and stage A is 6%. Optimizing the scoring kernel at prefill is optimizing the
  small end of the op, which is itself the small end of the step.
- Stage B occupies **0.6588 ms** of TensorCore at decode against **0.3249 ms** of actual SparseCore
  work: the TC blocks about **2x** as long as the SC computes. That gap is what Option B attacks.

Prefill caveats that survive: **n=1 model step** per capture, and `prefill_only`'s captured step has
**zero** `StreamIdxTC` and zero `sc_topk_direct` rows (`min_kv_len = 0`), so only `prefill_heavy`
carries the indexer at all. Collectives are 40.4% of the prefill step.

### What each change is worth on the corrected denominators

Arithmetic on the rows above and on the measured op-level deltas — **derived, not observed.** No
end-to-end run supports any row.

| change | measured op-level effect | phase | ~ % of that step |
| --- | --- | --- | ---: |
| **KV-cache relayout removal** | 13.778 ms/step, measured directly in the profile | prefill | **~2.21%** |
| Option B, runner chunking, G=4 | -29.6% of stage A+B (3.2614 ms/step) | prefill | ~0.155% |
| **D1-lite** | -6.982 us/layer of residual x 21 layers, scaled 0.918 for the microbenchmark's larger residual bucket | decode | **~0.19-0.21%** |
| Option C, prefill-gated | -20.6% of stage A (1.2491 ms/step) | prefill | ~0.041% |
| *D1 (full) — withdrawn* | *-16.980 us/layer x 21* | *decode* | *~0.49-0.51%* |
| *`bkv_p` ladder — withdrawn as sized* | *-14.6% of op* | *decode* | *~0.63%* |

The two largest decode entries are the two that were withdrawn, and the largest entry overall is
not a kernel change. Ordering the work by these numbers rather than by op-level percentages is the
single most useful thing §1c provides.

### Three things from the profile that outlive this op

1. **`sc_dsa_gather` is 4.205 ms/step, 6.21% -- 1.6x this entire op.** It is the gather that
   consumes the top-k indices this op produces. The whole DSA-indexer complex is 6.78 ms = **10.0%**
   of the decode step, and the kernel optimized here is the small end of it. Rest of the step:
   `other` 25.7 ms, `sc_ragged_gather` (MoE dispatch/combine) 18.3, collectives 13.8, `moe_gmm` 5.9.

2. **TensorCore and SparseCore do not overlap, confirmed on a real model step.** The intersection of
   stage B's TC-side `call-done` window with the SparseCore-plane `sc_topk` interval is 0.6485 ms --
   *exactly* the whole SC-plane top-k time. The TC is blocked for the SC's entire duration, so stage
   B's device time is its wall-clock contribution with no discount. This confirms
   `probes/baseline-device-time.md:141-145` on hardware, and it means the 83.5-98.2% overlap Option
   B (§8) demonstrated is real headroom the incumbent does not take.

3. **The microbenchmark overstates stage A by 19%** -- 111.7 µs/layer against the profile's 90.3.
   Stage B agrees almost exactly (31.73 vs 32.3). Op-level stage-A wins measured here are therefore
   slightly optimistic when carried to the model.

At this shape stage A is the larger half (2.80% vs 1.00%), inverting the microbenchmark's
prefill-driven picture in which stage B was 57.5% of the call.

## 2. Scoreboard

| option | what it changes | verdict | best evidence |
| --- | --- | --- | --- |
| **C** — head-major score layout | transposes the q block `(bq,H,D)→(H,bq,D)` **inside** the kernel, once per bq block | **PASS on prefill, −17.8% to −20.7%** across all `num_bkv`; correctness verified, index-identical. **Must be gated to prefill — decode regresses +59.62%** | §3 |
| **D1** — delete the `cond` | drops the early-exit `lax.cond`; live in production via `STREAMIDX_ENABLE_EARLY_EXIT=True` | **PASS, −10.0%**; correctness verified | §4 |
| **E** — `_pick_partition` | picks a better stage-B partition | **correct but not worth adopting** — the win lands on an unreachable bucket | §5 |
| **0** — incumbent at `bkv_p=9` | one constant, on the shipping kernel | **−14.6% full op / −24.8% stage A at decode, ctx 8704. Correctness verified.** Wrong at short ctx (+38% at 4096), so it wants per-bucket dispatch | §6 |
| **F** — bf16 operands | *proposed on a false premise* — the operands are already fp8 | **REJECTED, +4.63% prefill / +9.80% decode.** Its by-product closes the block: stage A has no roofline headroom left | §11.2–11.3 |
| **A** — 128-byte K record | halves the K record so half the bytes move | **Gate passes at `bkv_p=9` (47.9%) and it does not matter** — 94.3% of the fetch is already hidden, so the bytes buy nothing. Net −15.30 µs vs an 18.64 µs floor. Not recommended | §7.1 |
| **B** — two-unit pipeline | overlaps TensorCore stage A with SparseCore stage B across chunks | **Gate PASSES by 5–12×; prefill −16.4%/−24.4%, i.e. faster. Do not build as a kernel change** — the prize needs the runner to repack the batch | §8 |

---

## 3. Option C — head-major score layout: **−20.64%**, the strongest result here

Full write-up: `variants/option-c-head-major/RESULTS.md`.
Correctness job **`j-cf9e7189-8331-4613-9c2a`** (`Succeeded`, 4/4 shards `rc=0`).

**Measured, chain slope, best-of-9, prefill stage A:**

| arm | µs/call | r² |
| --- | ---: | ---: |
| incumbent (unmodified `src/`) | **246.58** | 0.9999995 |
| `head_minor` — identity control through the variant's copy | 246.36 | — |
| **`head_major` — Option C** | **195.69** | 0.9999990 |

**−50.89 µs, −20.64%.** The `head_minor` control landing within +0.09% of the unmodified incumbent
is what licenses the comparison: the variant file is a hand transcription, and this is the check
that no divergence hid in it.

**The mechanism is confirmed instruction-for-instruction, not inferred from the time.**
`vrot.slane` goes 8,960 → **1,280**. The 7,680 removed are exactly the set R3 attributed —
6,912 reduction-tree rotates + 1,024 pack rotates, every one an *immediate-amount* rotate. The
1,280 residual are all *register-amount*: 1,024 pre-existing dynamic relayouts that R3 said C
cannot touch, plus 256 introduced by the transpose itself. Controls held on every arm
(`vmatmul` 1,024, `vmax` 8,192, unchanged).

**The spill risk R3 raised was real and did not bite.** Stores +4,523 and fills +4,649 — but spill
*slots* went **down**, 1,375 → 1,162, and total bundles fell 11,600 → **9,404, −18.93%**. The
measured −20.64% slightly beats the bundle ratio, which is the opposite of the failure mode.

**Why it is affordable:** the transpose is paid in `load_bq`, which runs 8× per call at this shape,
to save work in `compute_scores`, which runs 40×.

**Two corrections C forced, both against briefs that predated it:**

1. the 335.1 µs stage-A baseline was 36% high (§1);
2. **`head_major_hbm` — R3's optimistic arm, q pre-transposed to `(H,T,D)` in HBM — does not
   compile.** Mosaic rejects it on sublane tile alignment (`size at dimension 1 divisible by the
   tile dimension 8`). **R3's −9.24% projection is retired**; it was for an arm that cannot exist.
   The deployable option is the in-kernel transpose, and it needs no change to q's HBM layout,
   `_fetch_bq`, or `mla_attention_op.py`.

### 3.1 The `num_bkv` sweep — the amortization worry is retired, and decode is not

Job `j-fd7b8801-fce9-46ec-bb63`, chain slope best-of-7, `same_mask: True` and `max_abs ≤ 1.22e-04`
at every point.

| shape | phase | `num_bkv` | incumbent µs | `head_major` µs | Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| `q1024_ctx2048` | prefill | 1 | 52.60 | 43.23 | **−17.81%** |
| `q1024_ctx4096` | prefill | 2 | 100.85 | 81.35 | −19.34% |
| `q1024_ctx6144` | prefill | 3 | 149.31 | 119.23 | −20.15% |
| `q1024_ctx8192` | prefill | 4 | 197.69 | 157.67 | −20.25% |
| `q1024_ctx9216` | prefill | 5 | 246.35 | 196.25 | −20.34% |
| `q1024_ctx16384` | prefill | 8 | 392.28 | 311.13 | −20.69% |
| **`decode_b64_ctx8704`** | **decode** | 5 | **109.83** | **175.31** | **+59.62%** |

**Prefill: C never crosses zero.** Monotone, worst case −17.81% at `num_bkv=1`, asymptote −20.7%.
The whole amortization effect is 2.9 points, so the caveat that gated this recommendation is
retired. ctx 9216 reproduced the committed measurement across two pods to 0.09% / 0.29% — the best
repeatability evidence in the exercise.

**Decode regresses +59.62%, and this is the more important half of the result.** §3.2 predicted −0
to −2%, and so did I, from the correct observation that at `bq_sz = 1` the head-major permutation
is *literally the identity*. It is the identity and the compiler still does not make the degenerate
transpose free. **Gating C to the prefill/MIXED call is therefore a hard requirement, not a
preference.** The mechanism — plausibly a length-1 sublane axis padded to Mosaic's 8-row tile — is
**judgement, not finding**: no decode LLO dump was taken, and it is one decode shape, stage A only.

This is the exercise's best argument for measuring both phases even when a mechanism argument says
one of them cannot move.

---

## 4. Option D — the win is real, but it is not the hoist

Full write-up: `variants/option-d-hoist/RESULTS.md`. Correctness `20 passed in 39.93s`.

**Measured −10.0%** on device self time — and the option's *named* mechanism, hoisting the
`lax.cond` out of the per-layer loop, is worth **0.01%**.

The reason is that XLA compiles each `lax.cond` branch into its own computation **per call site**
(86 `region_*_spmd`, 84 `conditional`s), so CSE cannot span the branches. Remove the `cond`
boundary and XLA's ordinary CSE performs the entire 21→1 collapse by itself. **D1 — simply deleting
the `cond` — is the whole win.** It needs no new *kernel* code, but it is not free: production
sets `STREAMIDX_ENABLE_EARLY_EXIT = True` (`mla_attention_op.py:57`), so the `cond` is live in
production and D1 is a real change at the call site, not a default already in force. (An earlier
revision of this file said it "requires no new code" on the strength of the kernel signature's
`enable_early_exit: bool = False` default; the caller overrides it.) 17.053 µs/layer removed, 93% of the 18.39 µs R2 found reachable.

The builder also **discarded a 32.5 µs apparent win** on its own initiative: it came from
cross-layer overlap that its independent probe layers could exploit and the real decoder cannot,
and SparseCore plane time rose to match. That discard is the reason to trust the −10.0%.

**Caveat carried forward:** the four tests that establish `fast path ≡ common path` are
`test_streamindex_topk_early_exit_*` — exactly the path D1 deletes. They pass, but a reviewer should
note that D1's safety argument leans on tests of the branch it removes.

**New candidate, recorded but not built:** hoist the `cond` to wrap all 21 layers rather than each
one. Pays 6.591 µs once and *keeps* the fast path, where D1 discards it.

---

## 5. Option E — correct, and correctly rejected

Full write-up: `variants/option-e-partition/RESULTS.md`.

Validated hard: 426 cases × every feasible `p` = **1,100 arms against a numpy oracle, 0 index-set
mismatches**; `20 passed`.

It is still not worth adopting, for a reason that has nothing to do with its quality:
**`min_token_size = 16` is hardcoded** (`tpu_platform.py:1236-1276`), so no batch bucket below 16 is
ever compiled and **b=8 is unreachable**. Of the seven reachable buckets, E differs from the
incumbent at exactly one (b=16, n ≤ 32768) — and DP=16 makes the bucket selection an all-reduce MAX,
so steady state is bucket 64. The win has no workload to land on.

Recorded because it is durable: the next session's explore will find the oracle harness, and the
b=8 diagnosis below stands regardless.

**b=8 slower than b=64 — the inversion in the baseline data — is explained.** `_pick_partition(8,
10240, 2048) = 4` forces the two-stage path; 50.09 µs two-stage vs 17.22 µs forced single-stage,
**2.91×**. Adding back the host gap reproduces `bench_prod`'s recorded 90.30 µs and 67.40 µs to
within 0.01 µs. This retires finding F2's open question in `briefs/ROOFLINE-REVIEW.md`.

---

## 6. Candidate 0 — `bkv_p`: the largest decode win in the exercise, **−14.6% on the full op**

Write-ups: `probes/baseline-device-time.md` and `probes/bkvp-extended.md`. Correctness jobs
**`j-79a410b8-46bc-4d06-862c`** (`20 passed, 14 warnings in 36.03s`) and the f64 oracle below.

### 6.1 What it is worth

At `bkv_p = 9`, ctx 8704, decode: **−24.8% stage A and −14.6% on the full op** by chain slope, with
`hlo_stats` independently giving −24.6% / −16.0%. That is a bigger decode win than D1's −10.0%, on
the shipping kernel, from one parameter.

The first pass measured only `p ∈ {2,3,4}` and found −9.46% at `p=3`. The extended sweep supersedes
it.

### 6.2 The exactness story was wrong, and its own author falsified it

The first pass explained the win as the `9 = 3×3` factorization: `p=3` giving an exact 3-block ×
3-page cover where `p=2`'s 5 × 10 overfetches. That predicted divisibility was the mechanism.

**It is not.** `p = 5` — inexact, it overfetches a page — **ties** `p = 3`: 88.72 vs 87.71 µs, 1.2%
apart. Divisibility per se buys nothing. What governs is a two-regime trade between blocks and
pages, refit on 31 arms:

```
blocks == 3 :  t = 27.08 + 6.77 · pages
blocks == 2 :  t = 17.05 + 7.40 · pages
blocks == 1 :  t =  3.49 + 8.79 · pages
```

Fewer blocks costs more per page. That, not factorization, is why the argmin moves with ctx.

### 6.3 A function, not a constant — but the cheap 90% is still a constant

The argmin is **`p = cdiv(ctx, page_size)` at nine ctx buckets out of nine** (2048…9216), with
`P−1` and `P+1` measured at each so the minimum is *bracketed rather than inferred*. The one
untested prediction, `p=8` at ctx 8192, came in at 73.98 µs against the model's 74.80.

**A fixed `p=9` is measurably wrong at short context** — full op, `p ∈ {2,9}` only so both are
`nst 80`:

| ctx | `p=2` | `p=9` fixed | Δ |
| ---: | ---: | ---: | ---: |
| 4096 | 98.96 | 136.54 | **+38.0%** |
| 6144 | 122.66 | 140.07 | **+14.2%** |
| 8192 | 149.29 | 145.00 | −2.9% |
| 8704 | 170.89 | 145.95 | **−14.6%** |

**Implementation constraint that shapes the recommendation:** `bkv_p` is a **compile-time static**
(`static_argnames`, `streamindex_topk.py:514`) while `num_bkv` is runtime
(`jnp.max(seq_lens)`, line 275). So "a function" means compiling per bucket and dispatching — not
reading `seq_lens` at runtime. **The cheap 90% is `p = cdiv(max_model_len, page_size)` = 9**, a
one-line constant change that captures the whole long-context win and gives up the short-context
buckets.

### 6.4 Where it stops, and two corrected premises

**It stops at `p = 32`, and raising `vmem_limit_bytes` does not move the wall** — rejected
identically at 16, 32, 64, 100, 128 **and 256 MiB**, with `Used 85.00M of 63.94M vmem`: the physical
TensorCore VMEM, not a scoped limit.

Two premises I supplied were wrong and were corrected by measurement. `streamindex_topk.py:28`
already sets `DEFAULT_VMEM_LIMIT_BYTES = 100 MiB`, so nothing here was ever running at a 32 MiB
scoped default; and **`p=9` needs only 32 MiB**, 3× under the production default. Timings at 100 vs
256 MiB are noise (82.29 → 82.35, 108.69 → 108.09, 143.34 → 143.30).

**Pipelining degrades but does not collapse**, so the curve does not turn up before the VMEM wall.
Page-count-matched pairs go the right way (1-block beats 2-block by 16.9% at 4 pages, 7.5% at 6,
2.0% at 8), the advantage shrinking because the marginal page cost rises. The **modelled** crossover
is 9.75 pages, confirmed independently by the 2-block arm already winning at 10 pages. At
`max_model_len 9216` demand is ≤ 9 — under the crossover, but only just.

### 6.5 Correctness: verified, no longer `provisional`

f64 numpy oracle at ctx 8704. **Byte-identical emitted index set at every `bkv_p` from 1 to 9, plus
10 and 16** — the same 295 positions in the same 64 rows, `n_unique_to_this_p = 0` everywhere. All
295 are the **incumbent's** too. Worst gap 1.1424 against σ 2.2408 = **0.51σ**, and the k-th place
margin (median 0.017) is two orders below σ, so the boundary at k=2048 of 8704 is degenerate in f32
— the same conclusion §11 reaches by a different route. Gate 2's `min_recall ≥ 0.999` fails on the
*incumbent* at every `p`; it is measuring the wrong thing.

**Independent confirmation:** seven xprof traces, partition residual **0.000000 on all seven**,
`rows_kept == rows_total == 42`. Control `dec_p2` reproduces R2's 170.474 µs at 170.724 (+0.15%).

Two self-corrections worth recording: the agent revised its own earlier "~67 µs / −39% `modelled`"
to **73.5 µs / −32.9%** after finding it had dropped a 6.24 µs intercept (measured: 82.38 µs,
−24.8%), and it **rejected** a ctx-2048 arm that fit at r² = 0.9245 rather than quoting it.

---

## 7. Option A — the 128-byte K record: failed at shipping `bkv_p`, **re-opened at `bkv_p=9`**

Full write-up: `probes/R1.md`. Jobs `j-b0e33be2-7d67-41da-b724` and `j-3e588eaa-75e8-4ddb-9737`,
both `Succeeded`, both **alone on the whole 4-chip pod, serially, no fan-out** — so these are
absolutes, not merely ratios. Best-of-9, per-arm spread 0.1–1.4%.

The gate: does halving the K record buy enough of its bytes back to be worth the change?

| `bkv_p` | DMAs in flight | saving from halving the record |
| ---: | ---: | ---: |
| **2 — today** | 8 | **21.0%** |
| **3 — where candidate 0 wins** | 12 | **25.4%** |
| 4 | 16 | 32.0% |
| 6 | 24 | **42.7%** |

**A fails at every configuration the design proposes to ship**, so it ranks below C, D1 and
candidate 0. The sturdiest form of that: **the absolute stake never reaches the design's own
prediction at any measured `bkv_p`** — 15.2 / 17.8 / 20.3 / 26.4 µs against the 28.4–49.4 µs that
−13 to −22% of a 221.4 µs call requires. (And that 221.4 is itself a wall clock; against the correct
170.58 the requirement is smaller but the arms all shrink with it.)

**But "A is descriptor-bound" is the wrong lesson, and this is the part worth keeping.** The
mechanism is **per-descriptor issue cost**, and concurrency hides it: going 8→12 in flight removes
the same ~20.5 ns/DMA at both record sizes. By 24 in flight the incumbent reaches 3.335e12 B/s —
**99.6% of P6's contiguous read** — at which point byte reduction pays nearly in full. So **A and
candidate 0 are complementary, not competing**: A is worth revisiting iff `bkv_p` lands above ≈5.
Candidate 0's `p=9` sweep therefore decides two options, not one, and R1's note that **`bkv_p=6` may
not be buildable** (12.58 MB VMEM double buffer, only 2 blocks left to pipeline against) is the
thing most likely to kill both at once. Relayed to that agent.

**Two alternative explanations were tested and refuted**, which is what licenses the mechanism
claim above:

- **Alignment.** A's 258-row page is 4 KiB-misaligned in 3 DMAs out of 4. A 264-row *aligned*
  control is **slower**, by 0.04% of what its extra bytes predict. Padding would not rescue A.
- **Dispatch floor.** Decided at grid 128, with grid 16 agreeing across an 8× change in work.

Flip 1's other named mechanism, **address indirection, is free** — page table plus both clamps cost
0.2–1.0%. P6's charging rule is not defeated by indirection: `affine` re-measured at `_fetch_bkv`'s
geometry fails the gate too, at 21.4%.

**The largest untested item in A's case, named by the agent and not papered over:** Option A's
changed *consumer* is not modelled at all. Moving the scale byte out of each record forces
`load_bkv` to gather it differently on the VPU, and the design asserts "the unpack does not change"
— an assertion about a code path this probe never executes. Direction is against A; magnitude
unknown. Also, the probe has **no compute and no consumer of the fetched bytes**, so its Δ is an
*upper bound* on the call-level win — which biases toward A, and A still fails.

A `critical-review` pass on the agent's own first draft caught four wrong claims, including an
out-of-bounds `affine` offset at grid 128; the gate's own arms were always clamped and unaffected,
and the `affine` arms were re-measured.

---

### 7.1 Re-measured at `bkv_p=9`: the gate passes, and it does not matter

Job `j-18e51a27-db20-4088-9fb4`, `Succeeded`, serial, unpinned on the whole pod, best-of-9, grid 128.

| `bkv_p` | DMAs in flight | charge ratio | saved |
| ---: | ---: | ---: | ---: |
| 6 | 24 | 1.736 | 42.4% |
| **8** | **32** | 1.888 | **47.0%** |
| **9** | **36** | 1.921 | **47.9%** |

**The gate passes, and R1's rank-4 verdict is reversed on that criterion.** Floor-corrected the
ratio is 1.997 against a 1.9845 perfect-proportionality ideal: at 36 DMAs in flight the indirect,
clamped DMA is **byte-proportional**. The `bkv_p=6` control reproduces R1's 42.7% at 42.4%. Refusing
to inherit the ≥42.7% extrapolation was right and it was conservative — the measured value helps A.

**And the pass is irrelevant, because the bytes are not what stage A waits on.** At `bkv_p=9`,
ns/block, floor-corrected:

| mode | incumbent | Option A | delta |
| --- | ---: | ---: | ---: |
| compute (no DMA) | 4192.9 | 3219.0 | **−973.9** |
| fetch (no unpack) | 2656.5 | 1332.7 | −1323.8 |
| **full** | **4345.0** | **3389.0** | **−956.1 (−22.0%)** |

**94.3% of the incumbent's fetch is already hidden behind compute**, so the composite equals the
unpack term alone to 1.8% and the byte reduction contributes **+17.8 ns/block — nothing**.
Corroborated without the decomposition: real stage A at `p=9` runs its K fetch at 1.79e12 B/s
against the 3.37e12 the same descriptors sustain, i.e. 47% headroom.

**Option A is a relayout win wearing a byte-reduction's clothes**, and the gate above measured the
half of the change that does not pay. The consumer worry was also backwards: A's unpack is
**cheaper**, not more expensive.

**Net stake −15.30 µs on the decode call, against a correctly recomputed floor of 18.64 µs** — A
misses its own −13% target by ~18%, delivering 10.7% of the `p=9` call. (The design's −13/−22% was
written against the 221.4 µs *wall clock*; at 170.58 µs device the bar is 22.18–37.53 µs, and at the
143.39 µs `p=9` ship configuration it is 18.64–31.55 µs.) Correctness is **bit-identical** by
checksum at both `bkv_p` and both A forms.

### 7.2 The mechanism, settled — and it does not rescue A

A `/critical-review` pass found the first mechanism attribution wrong-footed (`a_concat` builds the
operand a completely different way and lands within 0.6%), and a first `/xprof-llo` census could not
resolve it either: the three arms are **collinear** — `inc` = {lane-128 scale, 256-lane record},
both A arms = {lane-major rows, 128-lane record} — so they vary the transpose (T) and the payload
width (W) together.

The missing 2×2 cell was compiled: **`w_reshape`, the incumbent's 256-lane record with Option A's
lane-major scale rows** (job `j-a0566857-c6f2-4667-9eed`). Position is
`(w − a_reshape) / (inc − a_reshape)`: **0.00 = follows scale placement, 1.00 = follows record
width.**

| metric | inc | a_reshape | w_reshape | position |
| --- | ---: | ---: | ---: | ---: |
| **XLU bundle occupancy** | 46.24% | 1.05% | 1.04% | **−0.000** |
| `tpu.transpose` | 288 | 0 | 0 | +0.000 |
| spills + fills | 10,715 | 1,168 | 1,218 | +0.005 |
| VLIW bundles | 8,417 | 6,590 | 6,614 | +0.013 |
| **VMEM read bytes/block** | 9,750,528 | 5,068,800 | 9,787,392 | **+1.008** |

**The instruction cost follows the scale placement, not the record width** — `w_reshape` carries the
incumbent's full byte traffic and still lands on `a_reshape` everywhere instructions are counted.
Clean separation, and `inc`/`a_reshape` reproduced the earlier job exactly, so the cross-job tie
holds. **All static compile-time analysis**; the arm was compiled, never timed.

**How the 973.9 ns divides between the two axes remains unmeasured**, and the agent struck its own
"98.7% is scale placement" from four places before reporting: 98.7% is a share of *VLIW bundles*, and
converting that to a share of a measured nanosecond delta needs bundles ∝ time and the byte channel
priced at zero — while a static dump cannot price bytes at all.

**The reader-side variant does not survive its own cost.** Folding the scale onto the K operand
(`scale[bkv_sz,1]` lane broadcast) is the only genuinely reader-side route, ceiling **−15.58 µs
(static ceiling, not a prediction)**. But an fp8 K operand cannot absorb an e8m0 scale without
saturating — e4m3 maxes at 448 — so the operand widens to **bf16**, which *is* Option F, measured at
**+9.80% decode / +4.63% prefill with HBM held byte-for-byte constant** (§11.2). Measured at another
geometry so not directly subtractable, but it is a counter-cost of the prize's own order. **There is
no dominating variant.**

**Unreproduced, and it is the whole rest of A:** everything measured here is the **reader**.
Producing layout A needs a changed caller-side scatter in XLA, unmeasured and unproven — and prefill
stage A is 75.4% issue/VPU with no headroom (§11.3), where 15.30 µs of decode saving is easy to
erase. Also absent: `start_send_scores`, stage B, and every ctx bucket other than 8704. Status
`provisional`, `⚠️ unverified`.

---

## 8. Option B — the two-unit pipeline: passes its gate by 5–12×, and should still not be built

Full write-up: `variants/option-b-pipeline/RESULTS.md`. Correctness job
**`j-562532b4-8705-4ad2-945f`**, 4/4 shards `rc=0`, `20 passed, 14 warnings` on all three suite
shards; gate job **`j-c397124b-3d9c-476b-a03e`**.

**`overlap` = `hidden / ideal`**, where `hidden = (t_a + t_b) − t_pipe` at that chunk count `G`,
and `ideal = min(t_a, t_b)/G × (G−1)` — the most a perfect pipeline could hide. All chain-slope
device times, every fit r² ≥ 0.99981.

The `Δ time` column is against `t_pipe` **at G=1** — the honest serial reference, since the G=1
program is itself slightly slower than `t_a + t_b` measured apart (its stage B reads a buffer stage A
has just written rather than a resident one). **Negative is faster**, per §0.

| phase | G | t_a | t_b | t_pipe | **overlap** | **Δ time vs G=1** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 1 | 114.11 | 34.90 | 149.80 | n/a | — |
| decode | 2 | 120.34 | 44.11 | 142.80 | **98.2%** | **−4.67%** |
| decode | 4 | 133.27 | 85.85 | 157.43 | 95.8% | **+5.09%** (slower) |
| prefill | 1 | 263.32 | 440.77 | 727.67 | n/a | — |
| prefill | 2 | 272.99 | 449.28 | 608.34 | **83.5%** | **−16.40%** |
| prefill | 4 | 288.09 | 466.34 | 549.84 | **94.7%** | **−24.44%** |

**PASS on both phases, against bars of 20% decode / 7% prefill — by ~5× and ~12×.** TensorCore and
SparseCore genuinely overlap; P3's 41.6% is exceeded at every `G`. The design's own worry, that XLA
would serialize chunk *g*'s stage B against chunk *g+1*'s stage A, does not materialize: chunks own
disjoint buffers and the scheduler exploits that.

The overlap was verified **structurally, not just from the times**. The scheduled HLO for all six
(phase, G) programs passes four checks, including that the operand path from stage B back to stage A
is `['bitcast', 'bitcast-convert']` **and nothing else** — stage B literally reads stage A's buffer,
so there is no arithmetic that could be constant-folded. Stage-A kernels in flight per stage-B async
window go `[0,0]` at G=1, `[1,0,1,0]` at G=2, `[3,2,1,0,…]` at G=4, matching `hidden ≈ 0` at G=1
and near-ideal above it.

### Why it should still not be built here

**The production prefill shape is one sequence of 1024 tokens, and it cannot be split.** Stage A's
grid is over *sequences* and its masking is driven by `cu_q_lens`, so a chunk boundary must fall on
a sequence boundary. The prefill numbers above were obtained by presenting that single chunk as `G`
**pseudo-sequences** — semantics-preserving, and measured honestly, but it is a change to **how the
runner packs the batch**, not a kernel-local change. B's entire prize sits behind modifying
`mla_attention_op.py`'s caller and the scheduler's packing.

That puts B outside this exercise: it is not a kernel change, so it belongs to `/preflight-check`
and an end-to-end benchmark rather than to a Pallas variant. **Recorded as a live runner-level
candidate with a measured prize of −16% to −24% on the op**, which is a good position for it to be in —
it is the largest prefill number in this document and the only one that is not a kernel edit.

Three further arguments, all the agent's own:

- **The chunk plan is data, and vLLM calls this op under `jax.jit`** where `cu_q_lens` and
  `distribution` are traced, so the runner must pass the plan down as static metadata.
- **Decode's −4.67% at G=2 is inside the ±8.5% end-to-end spread**, and `t_a` there counts one stage-A
  launch per chunk against the incumbent's three — ~23 µs unaccounted at decode G=4, which makes
  decode *worse*, not better. Decode is a wash.
- **Decode G=4 is blocked by the 32-row wave quantum, not by the partition cliff it was blamed on.**
  Forcing `p=1` does remove that cliff — stage B goes 203.44 → 85.85 µs, the pipeline
  +92.36% → +5.09% (both still slower than G=1) — so the gathers were binding and *are* removable.
  But per-chunk stage B is
  22.06 µs at b=32 and 21.46 µs at b=16: halving the rows buys 2.7%, so `t_b` simply doubles.
  **G ≤ num_tokens/32, i.e. G ≤ 2 at batch 64.**

### Two process notes worth keeping

An **in-job checker initially reported `every_B_depends_on_an_A=False`** — a false negative that
would have killed Option B for the wrong reason. It searched for stage B as a `custom-call`, but the
SparseCore top-k compiles to `call-start`/`call-done` into an `async_execution_thread="sparsecore"`
computation. Both the error and the fix are on the record.

The `_pick_partition` tripwire was **self-tested on hardware** rather than assumed live:
`tripwire self-test: PASS -- tripwire fired on a deliberate sparsecore_topk call`. It never fired on
any chunked path. An assertion nobody has proven can fire is not evidence, and this one was proven.

**Honest limit, printed by the harness itself:** the suite arms route 19 of 23 traces through
Option B's single-group path, so they prove *transparency at G=1*, not chunking — every case in that
file has 1–3 sequences. The chunking evidence is the six shapes in shard 0, where index sets match
the unmodified incumbent at every `G` and cross-`G` (`G1_vs_G2`, `G2_vs_G4`, all `True`).

---

## 9. Stage B is issue-bound, not output-DMA-bound

`probes/stage-b-ksweep.md`. A 16× change in `k` moves total time **6.0%**, and `k=4096` is the
fastest point — an output-DMA model predicts 5,176 µs against 316 observed. Length-bounding the
output DMA is therefore worth **0.80%** at the production shape: a clean negative result, and the
cheapest one in the exercise.

It also produced stage B's **first non-fitted cost model** (every earlier one was calibrated on the
single point it then "predicted"):

```
t_device(µs) = 8.152 + waves × (0.702 + 0.05424e-3·n + 1.01975e-3·eff_len)
```

r² = 0.9982 over 35 points from two jobs. `modelled`. **Its scan coefficient is calibrated on
synthetic N(0,1) data and understates real production scores by 22%** — see §1. Use ~1.37 ns/element,
not 1.02.

---

## 10. Open items

1. ~~Option C's `num_bkv` sweep~~ — **DONE, C is safe on prefill at every `num_bkv`** (§3.1).
2. ~~The pre-existing oracle disagreement~~ — **RESOLVED, the shipping kernel is correct** (§11).
3. ~~Option F — bf16 operands~~ — **REJECTED, premise was false** (§11.2). Its by-product is the
   more important result: **prefill stage A has no roofline headroom left** — 75.4% issue/VPU,
   which C already attacked, and an infinitely fast score matmul would save ≤5.4% (§11.3).
4. ~~Candidate 0's `p=9` sweep and reference check~~ — **DONE** (§6).
5. ~~Is Option A's 974 ns/block the transpose or the payload width?~~ — **SETTLED: scale placement**
   (§7.2). It does not rescue A; the reader-side variant needs a bf16 K operand, which is Option F,
   measured at +9.80% decode.
5. ~~Options A and B~~ — **DONE** (§7, §8).
6. **Option B belongs to the runner, and nobody owns it.** −16.4%/−24.4% on prefill (faster), measured, and
   it needs the scheduler to pack the batch into chunks and pass the plan down as static metadata.
   Route to `/preflight-check` and an end-to-end benchmark, not to a Pallas variant.
7. **Decode stage B, real scores: chain slope +11.5% over `hlo_stats`** (3.7 µs). Named by the
   measuring agent and not explained. Small, but it is the one like-for-like comparison of seven
   that did not close inside 2.6%.
8. **`references/hardware-facts.md` has four claims this exercise showed to be wrong**, and it has
   no measured SparseCore fact at all — which is the direct cause of the prefill gate being
   undecidable in §1. Whether to amend it is a question for the caller; it has been asked and not
   answered.

---

## 11. The incumbent is correct at the top-k boundary, and the gate that said otherwise is the wrong gate

Job `j-5d02a60e-2266-477c-82d6`, three seeds. The probe spies on the module-global
`sparsecore_topk` inside a trace of the **shipping** `streamindex_topk.__wrapped__`, so it compares
the real kernel, not a transcription.

**The 10-of-2,560 disagreement is near-ties. The shipping kernel is not wrong.**

Three-way localization: `_effective_row_lengths` agrees with the oracle everywhere; **stage B is
`EXACT`** — 0/8 rows, 0 indices differing, in all six runs, independently confirming P2 against a
different reference. So the disagreement is entirely stage A's f32 scoring.

**All 31 disagreements across three seeds are sign flips** — `gap_f64 > 0`, `gap_f32 < 0`, no
exceptions. In the kernel's own arithmetic the "missed" item genuinely scores below its displacer,
so the selection is correct *given the scores*. Max |gap|/σ per seed is 0.49 / 0.58 / 0.66, against
a benign threshold of ≤4 that was **written into the probe's docstring before any number existed**.
Every gap is under a single σ. The participating positions are not error outliers either: 2 of 62
exceed p99, at 1.12× and 1.33×, both under the observed max.

**An absolute recall-vs-f64 gate is the wrong gate for this op and will fail forever on correct
code.** At rank 512 of 2048 the true boundary gaps are ~2.5e-04 — smaller than the kernel's own
scoring noise — so which of two effectively-tied items returns is arbitrary. **Arm-vs-arm index
agreement against the incumbent is the right statistic**, and it is what every option in this
document was judged by.

### 11.1 Why stage A's scores carry that error — it is the fp8 MXU pass

The probe found stage A's score error to be 7.4e-04 relative median / 6.3e-03 max, roughly 10⁴×
what f32 accumulation over 32 heads should give, and recorded it as an open question.

**First answer, and it was wrong.** This document previously attributed it to
`references/hardware-facts.md:104-116` — f32 operands at `DEFAULT` precision measuring 2.2104e-03,
identical to bf16 operands, because `preferred_element_type` sets the accumulator and not the input
precision. That reference is sound and it **does not reach this kernel**: stage A has no f32 operand
to demote. Both matmul operands are already `float8_e4m3fn`, confirmed in the Mosaic MLIR —
`tpu.matmul … vector<32x128xf8E4M3FN>, vector<2048x128xf8E4M3FN>, vector<32x2048xf32>`.

**The precision ladder settles it** (`j-e9f4996a-a2bd-4e9d-a3fc`, 3 seeds, bit-identical
exactly-representable inputs, widening verified at 0.0):

| arm | stage-A `max_abs` | `median_rel` |
| --- | ---: | ---: |
| **`fp8_native` — what ships** | **1.0628** | **9.01e-04** |
| `bf16` | 3.05e-05 | 2.89e-08 |
| `f32_default` | 3.05e-05 | 2.89e-08 |
| `f32_highest` | 3.05e-05 | 2.89e-08 |

`fp8_native` reproduces the observed error to five figures; every other arm is **31,200× lower**.
**The error is the fp8 MXU pass itself** — a hardware property, ~3.5e-03 median relative on the
matmul, and the price of its 1.86× throughput.

The verdict is unchanged — not a defect, nothing to fix, and forcing higher precision would cost a
throughput cliff — but the mechanism is the one above, not the cited one. Note the elegant
cross-check: **`f32_DEFAULT` is bit-identical to `f32_HIGHEST` here**, which confirms the
reference's mechanism from the other side. fp8-valued operands are exactly representable in bf16, so
`DEFAULT`'s truncation has nothing to throw away.

### 11.2 Option F — bf16 operands: premise false, measured anyway, rejected

**Option F was proposed on the mistaken premise above** — that stage A moved q and k as f32 and got
bf16 arithmetic for it, so bf16 storage would halve the bytes for free. Since the operands are
already fp8, bf16 would **double** the bytes (23.07 → 46.14 MB at prefill) and **halve** MXU
throughput (1419.9 → 764.8 Tops/s). Stage A already sits in the *best* cell of
`05-compute-units.md`'s table, not the worst.

It was measured regardless (`j-8c300d39-2612-469d-8602`, min r² 0.999994), which is what makes the
rejection evidence rather than a retraction:

| stage-A chain slope, µs | incumbent | `f_bf16` | C | F+C |
| --- | ---: | ---: | ---: | ---: |
| prefill q1024/ctx9216 | 246.12 | 257.53 **+4.63%** | 196.21 −20.28% | 202.07 −17.93% |
| decode b64/ctx8704 | 110.00 | 120.78 **+9.80%** | 175.32 +59.38% | 192.01 **+74.55%** |

F loses in both phases **even with HBM traffic held byte-for-byte constant**, and F+C is worse than
C alone in both. The controls reproduce prior measurements closely (C at −20.28% vs §3's −20.64%;
decode +59.38% vs §3.1's +59.62%), which is what licenses the new rows.

### 11.3 The by-product is the most useful thing here: stage A has no roofline headroom left

Answering F's gate question — *is stage A HBM-bound at all?* — produced the closing statement for
prefill stage A:

| | prefill stage A | decode stage A |
| --- | ---: | ---: |
| arithmetic intensity | **1319 FLOP/B** (ridge 384) | 37 FLOP/B |
| share of measured time in the modelled HBM floor | **7.1%** | 35.8% |
| share in the modelled MXU floor | 24.6% | — |
| **remainder — issue / VPU** | **75.4%** | — |

Prefill stage A is not HBM-bound and is not close. The 75.4% issue/VPU share is exactly what
Option C attacked and proved on hardware: C deleted 15,987 VALU ops at **zero** byte change and
**zero** matmul change and bought −20.64%.

The static census makes the remaining ceiling exact. `f_bf16` changes **precisely one thing** —
`vmatmul` 1,024 → 2,048, VPU census unchanged to the instruction — so its +4.63% is attributable to
the matmul alone. Linear-extrapolated, **an infinitely fast score matmul would save ≤5.4% of prefill
stage A.** There is nothing material left on either roofline axis. C is close to the end of the road
for this block, and further stage-A work at prefill should be justified against that 5.4%.

Decode is the memory-side phase (37 FLOP/B, 35.8% in the HBM floor), which is where a byte-reduction
idea could still pay — but F makes the KV records, 98% of decode's bytes, *bigger*. That is Option
A's territory (§7), not F's.

### 11.4 Decode's +59.62% is now a finding, not judgement

The decode LLO dump was taken. At `bq_sz = 1`, head-major **adds** work rather than removing it:
`vrot.slane` 2,291 → 2,915, `vadd` 420 → 2,020, spills 2,405 → 3,972, bundles 2,347 → **4,469
(1.90×)** — at identical `vmatmul` (32) and an identical fp8 pass on both arms. Entirely issue-side.
The hard gating of C to prefill/MIXED (§3.1) is confirmed by mechanism, not just by the time.

**Two limitations recorded rather than smoothed over.** The capture script piped through `tail -80`
and both decode arms emitted exactly 80 MATMUL lines, so their geometry and `finite_frac` headers
were eaten; geometry was recovered from artifact filenames and the script now caps only the MATMUL
flood. And **F as originally specified — bf16 in the KV cache — was never built**: it needs an
`mla_attention_op.py` change, and it would only add doubled bytes to an already-negative result.

Also: `same_mask: True` everywhere on the F arms, but that is finite/`-inf` agreement of the *score*
tensor, not top-k index agreement. Stage A does not select, so this harness structurally cannot
produce the index statistic §11 argues for. `max_abs` vs incumbent is 2.92 / 2.24, about 2.7× the
incumbent's own deviation from f64.

---

## 12. Recommendation

Three changes are worth making, and they are worth making in this order. Everything else in this
document is a measured reason **not** to do something, which was most of the value.

### Adopt

**1. `bkv_p = 9` — the largest win, one line, decode (§6).**
**−14.6% on the full decode op** at ctx 8704 (chain slope; `hlo_stats` independently −16.0%),
correctness verified against an f64 oracle with byte-identical index sets. Ship it as
`p = cdiv(max_model_len, page_size)`, which is the cheap 90%. **Do not ship it as a bare constant
`9`** — it is +38.0% at ctx 4096 and +14.2% at 6144, so a deployment whose `max_model_len` is small,
or which serves short contexts through the same compiled bucket, gets slower. The full win needs
per-bucket compilation and dispatch, because `bkv_p` is a compile-time static while `num_bkv` is
runtime.

**2. D1 — delete the early-exit `cond` (§4).** **−10.0%** on device self time, correctness verified.
It needs no new kernel code, but it **is** a real change: production sets
`STREAMIDX_ENABLE_EARLY_EXIT = True` (`mla_attention_op.py:57`), so the `cond` is live and the
−10.0% is available rather than already banked. XLA's ordinary CSE
does the whole 21→1 collapse once the `cond` boundary is gone. Note the safety argument leans on
`test_streamindex_topk_early_exit_*` — tests of the branch being removed.

**3. Option C — head-major score layout, prefill only (§3).** **−17.8% to −20.7%** on prefill
stage A across every `num_bkv`, index-identical output, mechanism confirmed instruction-for-
instruction. **Gate it to prefill/MIXED as a hard requirement**: decode regresses **+59.62%**, now a
finding rather than a suspicion.

Call-level arithmetic for C, stated so it can be checked: prefill stage A is 249.300 µs of the
749.538 µs call. If all three stage-A launches benefit at the measured −20.3%, that is ~50.6 µs, or
**~6.7% of the prefill call**; if only the one launch that was measured benefits, ~2.2%. **The
proportional case is an assumption — only a single stage-A call was measured.** That gap is the
first thing to settle before quoting C at call level.

**Composition is unmeasured.** Candidate 0 and D1 both act at decode and look orthogonal — one is a
fetch parameter, the other a control-flow boundary — but "looks orthogonal" is what Option D's hoist
also looked like before XLA's CSE turned a 21→1 collapse into 0.01%. Measure the pair before
claiming −24%.

### Do not adopt

| | why |
| --- | --- |
| **A** — 128-byte K record | 17.9% short of its own floor. Passes its byte-reduction gate at `bkv_p=9` and the pass is irrelevant: 94.3% of the fetch is already hidden (§7.1) |
| **E** — `_pick_partition` | correct, and the win lands on a bucket `min_token_size=16` makes unreachable (§5) |
| **F** — bf16 operands | premise false; the operands are already fp8. +4.63%/+9.80% (§11.2) |

### Hand off, don't drop

**Option B is the largest prefill number in this document — −16.4%/−24.4% on the op, gate passed by
5–12× — and it is not a kernel change (§8).** It needs the runner to pack the batch into chunks and
pass the plan down as static metadata, because the production prefill shape is one sequence of 1024
tokens and stage A's grid is over sequences. Route it to `/preflight-check` and an end-to-end
benchmark. It is the single most valuable unowned item here.

### What this block has left

**Prefill stage A is done.** 1319 FLOP/B against a 384 ridge, 7.1% of measured time in the HBM
floor, **75.4% issue/VPU** — the axis C attacked. An infinitely fast score matmul would save ≤5.4%
(§11.3). After C there is no material headroom on either roofline axis, and further prefill stage-A
work should be justified against that 5.4% before it is started.

**Prefill as a whole is undecidable, and stage B is why.** 431 of 749 µs is SparseCore, and there is
no first-principles SparseCore floor to divide into —`references/hardware-facts.md` contains no
measured SparseCore fact. Defensible treatments span 53.9%–92.0% of floor. Building one is the
highest-leverage missing input in this whole exercise, and it is a reference-file gap, not a kernel
question.

**Decode remains structural at ~45% of floor** (33–53%), so candidate 0 is unlikely to be the end of
it.

### Two methodological results worth more than any single option

**A host dispatch floor is a clamp, not an addend.** Six quantities re-measured with a floor-immune
method all came in smaller, including the baseline that a pass/fail bar was written against — a bar
the incumbent had already cleared, so it discriminated nothing (§1). Any per-call wall clock in this
codebase should be treated as `max(device, host)` until shown otherwise.

**A data distribution is part of a benchmark's configuration.** The stage-B k-sweep was calibrated
on synthetic N(0,1) scores and understates production by 22%, because real indexer scores are
all-positive with few distinct exponents and the MSB-first radix select needs more passes. N(0,1) is
not a neutral default for a radix algorithm (§1).
