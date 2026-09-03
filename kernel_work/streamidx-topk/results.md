# Step 3 — variants built, measured and ranked

**Status: DRAFT, agents still in flight.** Options C, D, E and candidate 0 are complete and are
reported in full, as is the recalibrated baseline (§1). **Options A and B are placeholders**, and
three follow-ups are running: candidate 0's `p=9` sweep and reference check, and Option F (§9.2).
Do not quote this file as final until this banner is gone.

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

## 2. Scoreboard

| option | what it changes | verdict | best evidence |
| --- | --- | --- | --- |
| **C** — head-major score layout | transposes the q block `(bq,H,D)→(H,bq,D)` **inside** the kernel, once per bq block | **PASS on prefill, −17.8% to −20.7%** across all `num_bkv`; correctness verified, index-identical. **Must be gated to prefill — decode regresses +59.62%** | §3 |
| **D1** — delete the `cond` | drops the early-exit `lax.cond`; `enable_early_exit=False` already selects it | **PASS, −10.0%**; correctness verified | §4 |
| **E** — `_pick_partition` | picks a better stage-B partition | **correct but not worth adopting** — the win lands on an unreachable bucket | §5 |
| **0** — incumbent + `bkv_p=3` | one constant, on the shipping kernel | **−9.46% at decode, but only at ctx 8704/9216**; +2.72% at 8192. `provisional` | §6 |
| **A** — 128-byte K record | | `in flight` — R1 decides rank 1 vs rank 4 | |
| **B** — two-unit pipeline | | `in flight` — Flip 3 can retire it outright | |

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
the `cond` — is the whole win, and it requires no new code: `enable_early_exit=False` already
selects that path.** 17.053 µs/layer removed, 93% of the 18.39 µs R2 found reachable.

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

## 6. Candidate 0 — the incumbent with `bkv_p=3`: a real −9.46%, but it is a shape artifact

Full write-up: `probes/baseline-device-time.md`. Correctness job **`j-79a410b8-46bc-4d06-862c`**,
`20 passed, 14 warnings in 36.03s`.

Full op, decode, device time by chain slope, `hlo_stats` confirming all nine points independently:

| ctx | `p=2` (incumbent) | `p=3` | `p=4` |
| ---: | ---: | ---: | ---: |
| 8192 | 149.89 | 153.97 **+2.72%** | 142.49 −4.94% |
| **8704** | 171.21 | **155.01 −9.46%** | 177.24 +3.52% |
| **9216** | 171.91 | **156.32 −9.06%** | 178.34 +3.74% |

**P7's −11.9% does not survive as an absolute; −9.46% does.** P7 was 2.4 points optimistic, exactly
as a contended ratio would be.

**The win is the `9 = 3×3` factorization, not a better setting.** Decode stage A fits
`6.24 + 7.41·blocks + 6.65·pages` (r² 0.9967, max residual 1.3%) **with `ctx` absent from the model
entirely**. `bkv_p=3` wins precisely where `cdiv(ctx, 1024) = 9` factors exactly as 3 blocks × 3
pages, against `p=2`'s 5 × 10 — which overfetches. At ctx 8192 there are 8 pages, `p=2` is the exact
one, `p=3` overfetches, and the full op inverts to +2.7%.

**This is why candidate 0 is not simply "change a constant".** If the effect is exactness, the right
change is to *derive* `p` from `cdiv(ctx, page_size)` rather than pin it — and the model extrapolates
to `p=9` at ~67 µs, **−39% (`modelled`, unswept, VMEM untested)**, four times the measured win. Both
the sweep and the constant-vs-function question are dispatched.

**Correctness is `provisional`, and the gap was self-reported.** The suite parametrizes `bkv_p` over
{4, 8, 16, 32} at `page_size × bkv_p = 128`; it never runs `bkv_p ∈ {2, 3}` and never
`page_size = 1024`. The evidence at the production shape — `index_sets_match_bkvp2: true`, 64/64
rows at all nine (ctx, `bkv_p`) points — is **self-consistency against the incumbent, not a
reference check**, and would pass equally if both were wrong. A numpy top-k reference is dispatched.

---

## 7. Stage B is issue-bound, not output-DMA-bound

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

## 8. Open items

1. ~~Option C's `num_bkv` sweep~~ — **DONE, C is safe on prefill at every `num_bkv`** (§3.1).
2. ~~The pre-existing oracle disagreement~~ — **RESOLVED, the shipping kernel is correct** (§9).
3. **Option F — bf16 operands for stage A** (§9). Dispatched. Falls directly out of the accuracy
   work, and may be the decode-side option C cannot be.
4. **Candidate 0's `p=9` sweep and its numpy reference check** — both dispatched. The sweep decides
   whether this is a constant to change or a function to write, and it is worth up to −39%
   (`modelled`) rather than the −9.46% measured.
4. **Options A and B** — in flight.
5. **Decode stage B, real scores: chain slope +11.5% over `hlo_stats`** (3.7 µs). Named by the
   measuring agent and not explained. Small, but it is the one like-for-like comparison of seven
   that did not close inside 2.6%.
6. **`references/hardware-facts.md` has four claims this exercise showed to be wrong**, and it has
   no measured SparseCore fact at all — which is the direct cause of the prefill gate being
   undecidable in §1. Whether to amend it is a question for the caller; it has been asked and not
   answered.

---

## 9. The incumbent is correct at the top-k boundary, and the gate that said otherwise is the wrong gate

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

### 9.1 Why stage A's scores carry bf16-scale error — closed, with a citation

The probe found stage A's score error to be 7.4e-04 relative median / 6.3e-03 max: roughly 10⁴×
what f32 accumulation over 32 heads should give, on a path that reads f32 throughout
(`preferred_element_type=jnp.float32`, weights cast to f32, no quantization in
`prepare_index_weights`). It was recorded as an open question.

It is not a defect and it is documented. `references/hardware-facts.md:104-116`, measured relative
error:

| | |
| --- | ---: |
| f32 operands, `DEFAULT` | 2.2104e-03 |
| **bf16 operands, f32 accumulator** | **2.2104e-03** |
| f32 operands, `HIGH` | 1.2995e-05 |
| f32 operands, `HIGHEST` | 1.4283e-07 |

`f32_DEFAULT / bf16_operands = 1.000×` — identical, not close. **`preferred_element_type` sets the
accumulator, not the input precision**; without a named `precision` the MXU discards the mantissa
before it sees the operands. The measured 7.4e-04 / 6.3e-03 straddles that 2.2e-03 characteristic
error. `HIGHEST` would buy it back at a 4.5× throughput cliff, so this should **not** be "fixed".

### 9.2 It produces a new candidate — Option F, bf16 operands

Stage A moves q and k as f32 and receives bf16 arithmetic for it — what `hardware-facts.md` calls
"the worst cell in the table: bf16 accuracy at f32 bandwidth." Storing those operands as **bf16**
gives, per the measured table, an *identical* result distribution while halving the bytes they cross
HBM with.

The accuracy argument is already discharged by §9: the true boundary gaps (2.5e-04) sit an order of
magnitude *below* the error the kernel already incurs, so selection there is arbitrary today and
bf16 operands cannot make it meaningfully more so.

**What decides it is whether stage A is HBM-bound at all** — R3's entire account was `vrot.slane`,
i.e. issue-bound, in which case F buys ~nothing and is a cheap clean negative. Dispatched, with that
question first and a stop if the answer is no, plus a composition check against C (a −20% and a −X%
do not add if they attack the same bound) and both phases measured, since C's prefill-only gating
must not be inherited unexamined.
