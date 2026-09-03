# Step 3 — variants built, measured and ranked

**Status: DRAFT, four agents still in flight.** Options C, D and E are complete and are reported
here in full. Option A, Option B and the recalibrated incumbent baseline are marked `in flight`
and their rows are placeholders. Do not quote this file as final until this banner is gone.

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
| stage B, prefill | 481.0 µs | **335.5** | stage-B k-sweep |
| stage A per layer | 221.4 µs | **170.5** | R2 |
| **stage A, prefill — the Option C baseline** | **335.1 µs** | **246.58** | Option C, chain slope, r² = 0.9999995 |
| incumbent stage A + `bkv_p` | −11.9% (contended) | in flight | candidate 0 |

The last-but-one row is the consequential one: **the ≤280 µs pass bar written for Option C was
already cleared by the incumbent**, so as written the gate discriminated nothing. C passes anyway,
on a like-for-like comparison against a correctly measured incumbent, but its *predicted* call-level
effect drops from −7.9% to **≈−5.9% (modelled)**.

This also means `measurement_and_design.md`'s prefill roofline gate — already flagged
`PARTIALLY FALSIFIED` in `briefs/ROOFLINE-REVIEW.md` for being circular — rests on a denominator
that is itself ~36% too high. **Do not quote 80.4%, and do not quote 39.9%.** The honest statement
remains the review's: a 26–57% band, structural, proceed.

---

## 2. Scoreboard

| option | what it changes | verdict | best evidence |
| --- | --- | --- | --- |
| **C** — head-major score layout | transposes the q block `(bq,H,D)→(H,bq,D)` **inside** the kernel, once per bq block | **PASS, −20.64%** on prefill stage A; correctness verified, output index-identical | §3 |
| **D1** — delete the `cond` | drops the early-exit `lax.cond`; `enable_early_exit=False` already selects it | **PASS, −10.0%**; correctness verified | §4 |
| **E** — `_pick_partition` | picks a better stage-B partition | **correct but not worth adopting** — the win lands on an unreachable bucket | §5 |
| **A** — 128-byte K record | | `in flight` — R1 decides rank 1 vs rank 4 | |
| **B** — two-unit pipeline | | `in flight` — Flip 3 can retire it outright | |
| **0** — incumbent + `bkv_p` | the incumbent, as a candidate | `in flight` — clean absolute replacing a contended ratio | |

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

**Outstanding, dispatched:** a `num_bkv` sweep. The transpose amortizes over `num_bkv`, which is 5
at the measured shape. At `ctx ≤ bkv_sz` (`num_bkv` = 1) it is paid once per use and C may be flat
or negative. The builder named this itself as *"the likeliest way this disappoints"*. **C should
not be committed to before that sweep returns.**

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

## 6. Stage B is issue-bound, not output-DMA-bound

`probes/stage-b-ksweep.md`. A 16× change in `k` moves total time **6.0%**, and `k=4096` is the
fastest point — an output-DMA model predicts 5,176 µs against 316 observed. Length-bounding the
output DMA is therefore worth **0.80%** at the production shape: a clean negative result, and the
cheapest one in the exercise.

It also produced stage B's **first non-fitted cost model** (every earlier one was calibrated on the
single point it then "predicted"):

```
t_device(µs) = 8.152 + waves × (0.702 + 0.05424e-3·n + 1.01975e-3·eff_len)
```

r² = 0.9982 over 35 points from two jobs. `modelled`.

---

## 7. Open items

1. **Option C's `num_bkv` sweep** — dispatched. Gates the C recommendation.
2. **A pre-existing correctness question in the shipping kernel, owned by nobody.** Gate 2's 0.999
   recall criterion fails on the **incumbent**: `min_recall 0.996094`, identical across all three of
   C's arms, an f64 oracle disagreeing with the production kernel on **10 of 2,560 indices**. This
   is not caused by any option here. P2 established stage B's select is bit-identical to
   `approx_max_k(recall_target=1.0)` at max |Δscore| = 0.0, which points at stage A's f32 scoring
   rather than the select. **Dispatched: are the 10 near-ties, or is the shipping kernel wrong?**
   The score *gaps* at those positions discriminate; a recall number does not.
3. **Options A, B and candidate 0** — in flight.
4. **`references/hardware-facts.md` has four claims this exercise showed to be wrong.** Whether to
   amend it is a question for the caller; it has been asked and not answered.
