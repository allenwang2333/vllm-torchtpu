# Composed candidate: Option C (gated) + D1-lite, measured as one module

One op-level A/B of the `streamindex_topk` indexer call: the incumbent against a single module
carrying both surviving changes from the kernel-optimization exercise, plus each change on its own,
all measured independently against the incumbent **in the same job**.

Everything here comes from job **`j-bf290089-a5b2-417c-a900`** (`JobStatus` green, 4/4 shards
`rc=0`). Nothing in this file is modelled, extrapolated, or carried over from an earlier job.

---

## 0. How to read this file

### The four arms (plus one control)

Every arm is a way of calling the *same* composed module,
`variants/composed/streamindex_topk_composed.py`, except `inc`, which is the production entry point
untouched.

| arm | what it is | `hoist` | `head_major_mixed` |
| --- | --- | :---: | :---: |
| `inc` | **Incumbent.** `vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk`, unmodified, imported from the mapped-in `src/`. The baseline every Δ is taken against. | — | — |
| `ctl` | **Control.** The composed module with both changes switched *off*. It exists to prove the wrapper itself is free: if `ctl ≠ inc` by more than noise, every other number in this file is contaminated by the harness rather than by a change. | `False` | `False` |
| `c` | **Option C alone.** Head-major score layout, gated to the MIXED/prefill launch only. | `False` | `True` |
| `dl` | **D1-lite alone.** Branch-invariant metadata and the three scalar-prefetch operands hoisted into the entry computation. The `lax.cond` and `_fast_path` are untouched and verbatim. | `True` | `False` |
| `c_dl` | **The composed candidate.** Both changes, under their gating. This is the deliverable. | `True` | `True` |

`c` and `dl` are **not** derived by subtraction from `c_dl`. Each is its own compiled program, its
own trace, its own row in the table. This matters: an earlier version of this exercise defined one
term as a leftover of the others, which made the composed arm cancel algebraically and produced an
additivity check that could not fail. Here all four terms are measured, so the composition residual
below is a genuine three-way test and *can* come out non-zero.

### What Option C's gate actually is

The gate is the whole reason Option C survived. Ungated at decode it was measured at **+59.62%** —
a 1.6x regression. So:

| launch | `q_layout` under `c` / `c_dl` |
| --- | --- |
| DECODE grouped (`run_scores_kernel` call 1) | `head_minor` — unchanged from incumbent |
| DECODE mop-up (`run_scores_kernel` call 2) | `head_minor` — unchanged from incumbent |
| MIXED / prefill (`run_scores_kernel` call 3) | **`head_major`** |

The gate was verified two independent ways at runtime, not assumed — see §7.

### The metrics

"Op level" here means the **entire `streamindex_topk` call**, not a share of a model step. Nothing
in this file is normalised to a model step, and no host dispatch floor is subtracted from anything
(per-call wall clock is `max(device, host)`, not `device + host`; subtracting a floor produced
numbers wrong by up to 10x earlier in this exercise).

| column | definition |
| --- | --- |
| **full** | Total device self time of the whole call. Sum of the `Avg. self time (us)` column of `hlo_stats` over every distinct HLO op in the program, **TensorCore rows only**, divided by the number of `streamindex_topk` calls in the program (`layers`). Units: **µs per `streamindex_topk` call**, written µs/layer. |
| **stage A** | The Mosaic scoring kernel — `hlo_stats` rows matching `StreamIdxTC`. Includes the head-major variant, whose scope is `StreamIdxTC-hm-…`, so the bucket is arm-independent. |
| **stage B** | The SparseCore exact top-k — rows matching `sc_topk` / `sparsecore_topk` / `SparseCoreTopK`, **TensorCore plane only** (see the warning below). |
| **residual** | `full − stage A − stage B`: everything else inside the call. The `lax.cond`, the predicate reduction, metadata arithmetic, scalar-prefetch construction, stage-B input shaping, the `-inf` init, weight layout changes, boundary copies. This is where D1-lite operates. |
| **Δx** | `mean(x) − mean(inc)`, µs/layer. Negative is faster. |
| **R** | The composition residual, `Δ(c_dl) − [Δ(c) + Δ(dl)]`. `R < 0` means the two changes together beat the sum of their parts; `R > 0` means they overlap and the second one gets less than it did alone; `R ≈ 0` means they are additive. |
| **spread** | `(max − min) / mean` over replicate traces of the same arm in the same job, where the measurement loop was **re-entered** for each replicate rather than run longer. This is run-to-run variation, which is what a Δ has to clear. |

**Two rows are never added together.** `hlo_stats` carries a `TPU core type` column with both
`TensorCore` and `SparseCore` rows. For the offloaded stage-B op the TensorCore row is the
TensorCore's *blocking wait* and the SparseCore row is the SparseCore's *concurrent execution* —
the same wall-clock interval counted twice. Every number in this file is the TensorCore plane. The
per-trace table in `results/composed_hlo_stats.md` reports SparseCore separately, never added in.

### The gates every number passed

Two, both hard:

1. `|partition_residual| < 1e-6` — stage A + stage B + residual must reconstruct `full` exactly.
2. **`rows_kept == rows_total`** — the `hlo_stats` read must not be truncated. This one is not
   optional and is easy to miss: `bdt_hlo_stats.dump()` defaults to `--top-n 500`, a 21-layer
   program with 21 `lax.cond`s emits 822–961 rows, and a truncated read **still passes the
   partition check**, because it partitions only the rows it kept. The reader was run at
   `--top-n 4000`.

All **115** traces passed both. Zero dump errors, zero `FAILED` rows, every trace `rows=N/N`, every
`partition = ±0.000000`.

### What is not here

No wall-clock number. The job fanned four shards across four chips of one pod; a fanned-out job's
wall clock is contended and is not a result. The per-shard logs print one, labelled as such.

No end-to-end model benchmark. The claim in this file is an op-level claim only.

---

## 1. Headline — 21-layer chain at the production decode point

`L21_dec_b64_ctx8704`: batch 64, `q_len` 1, ctx 8704, `max_model_len` 9216, `page_size` 1024,
k = 2048, 21 chained `streamindex_topk` calls. GLM-5.2 has 21 indexer layers, and D1-lite's benefit
is cross-layer common-subexpression elimination, so 21 layers is the shape that matters — at a
single call the same change is worth about a third as much (§3).

n = 3 replicates per arm, loop re-entered each time.

| arm | full µs/layer | sd | spread | Δfull | Δfull % | stage A | stage B | residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inc` | 170.189 | 0.249 | 0.256% | — | — | 111.538 | 32.625 | 26.026 |
| `ctl` | 170.272 | 0.116 | 0.123% | +0.083 | +0.05% | 111.565 | 32.624 | 26.083 |
| `c` | 170.600 | 0.034 | 0.038% | +0.411 | +0.24% | 111.612 | 32.684 | 26.304 |
| `dl` | 163.383 | 0.009 | 0.011% | −6.806 | −4.00% | 111.579 | 32.634 | 19.169 |
| **`c_dl`** | **163.522** | 0.065 | 0.072% | **−6.667** | **−3.92%** | 111.773 | 32.623 | 19.126 |

### `inc` → `c_dl`, broken out

| metric | inc µs/layer | c_dl µs/layer | Δ µs/layer | Δ % |
| --- | ---: | ---: | ---: | ---: |
| **full op** | 170.189 | 163.522 | **−6.667** | **−3.92%** |
| stage A | 111.538 | 111.773 | +0.235 | +0.21% |
| stage B | 32.625 | 32.623 | −0.002 | −0.01% |
| residual | 26.026 | 19.126 | **−6.900** | **−26.51%** |

The whole win is in the residual, and it is D1-lite's: the hoist removes a quarter of everything in
the call that is not one of the two kernels. Stage A and stage B are untouched, which is exactly
what the gate predicts — at pure decode the MIXED launch never executes, so Option C has no kernel
to make faster and contributes +0.24% of noise-level cost.

`ctl` lands 0.05% from `inc`, inside `inc`'s own 0.256% replicate spread. The composed module
itself is free.

---

## 2. 21-layer chain at prefill

`L21_pre_q1024_ctx16384`: 1 sequence, `q_len` 1024, ctx 16384, 21 chained calls. n = 1 (no
replicates at this config), so there is no error bar on these Δ.

| metric | inc | Δctl | Δc | Δdl | **Δc_dl** | Δc+Δdl | R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **full op** | 1199.585 | −0.892 | −81.584 (−6.80%) | −3.903 (−0.33%) | **−84.236 (−7.02%)** | −85.487 | +1.252 |
| stage A | 394.085 | +0.010 | −80.232 (−20.36%) | −0.082 | **−80.432 (−20.41%)** | −80.314 | −0.117 |
| stage B | 709.943 | −0.012 | −0.274 | +0.043 | +0.012 | −0.232 | +0.244 |
| residual | 95.557 | −0.890 | −1.078 | −3.863 (−4.04%) | **−3.816 (−3.99%)** | −4.941 | +1.125 |

This is the mirror image of §1. At prefill the MIXED launch is the only one that runs, so Option C
gets its kernel and takes **20.4% off stage A** — the in-kernel transpose replaces the `vrot.slane`
rotate tree in `sum_h` with a stride-16 reduction over axis 1. D1-lite still contributes its
residual saving (−4.0%), a fifth of what it is worth at decode because the residual is a much
smaller fraction of a prefill call (8% vs 15%).

Together: **−7.02% of the full op**. The two changes hit disjoint parts of the call and the
residual `R = +1.252 µs/layer` is +0.104% of `inc` — 98.5% of the additive prediction is realised.

---

## 3. Single-call grid — 17 configurations

One `streamindex_topk` call per program, all four arms, n = 1 each. This is the **coverage** arm:
its job is not to produce the headline but to answer "does any configuration regress?"

All figures µs per call. `L1_dec_b64_ctx8704_mml9216` is the single-call twin of the headline shape.

| config | inc | Δctl | Δc | Δdl | **Δc_dl** | **Δc_dl %** | R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L1_dec_b8_ctx4096` | 76.337 | +0.050 | −0.110 | −0.701 | **−0.190** | **−0.25%** | +0.622 |
| `L1_dec_b8_ctx16384` | 105.540 | +0.164 | +0.200 | −1.756 | **−1.612** | **−1.53%** | −0.057 |
| `L1_dec_b8_ctx32768` | 133.044 | +0.032 | +0.108 | −1.743 | **−1.573** | **−1.18%** | +0.062 |
| `L1_dec_b8_ctx65536` | 185.952 | +0.461 | −0.082 | −1.429 | **−1.520** | **−0.82%** | −0.010 |
| `L1_dec_b32_ctx4096` | 66.219 | +0.059 | −0.061 | −1.253 | **−0.830** | **−1.25%** | +0.484 |
| `L1_dec_b32_ctx16384` | 144.416 | +0.184 | +0.225 | −1.948 | **−2.083** | **−1.44%** | −0.360 |
| `L1_dec_b32_ctx32768` | 251.586 | −0.146 | −0.171 | −2.109 | **−2.210** | **−0.88%** | +0.070 |
| `L1_dec_b32_ctx65536` | 464.353 | +0.421 | +0.673 | −2.141 | **−2.124** | **−0.46%** | −0.655 |
| `L1_dec_b64_ctx4096` | 99.029 | −0.286 | −0.157 | −1.530 | **−1.726** | **−1.74%** | −0.039 |
| `L1_dec_b64_ctx16384` | 255.729 | −0.335 | −0.525 | −2.907 | **−2.952** | **−1.15%** | +0.480 |
| `L1_dec_b64_ctx32768` | 464.945 | +0.531 | +0.776 | −2.308 | **−2.366** | **−0.51%** | −0.834 |
| `L1_dec_b64_ctx65536` | 897.911 | −1.636 | −1.825 | −4.695 | **−4.834** | **−0.54%** | +1.686 |
| `L1_dec_b64_ctx8704_mml9216` | 171.467 | −0.013 | +0.134 | −2.342 | **−2.257** | **−1.32%** | −0.049 |
| `L1_pre_q1024_ctx4096` | 381.472 | −0.038 | −19.272 | +3.229 | **−15.621** | **−4.10%** | +0.421 |
| `L1_pre_q1024_ctx16384` | 1194.253 | −0.487 | −80.988 | +2.520 | **−78.245** | **−6.55%** | +0.223 |
| `L1_pre_q4096_ctx4096` | 1111.159 | −0.798 | −76.607 | +0.986 | **−74.942** | **−6.74%** | +0.679 |
| `L1_pre_q4096_ctx16384` | 4439.276 | −1.876 | −323.864 | +0.568 | **−322.588** | **−7.27%** | +0.708 |

**No configuration in this grid regresses on `c_dl`.** Every one of the 17 is negative, from
−0.25% to −7.27%.

Three things the grid shows that the headline cannot:

- **Decode is D1-lite's, prefill is Option C's, and neither interferes with the other.** On the 13
  decode configs `Δc` is within ±1.9 µs of zero (mostly ±0.5) and `Δc_dl ≈ Δdl`. On the 4 prefill
  configs `Δc` is −19 to −324 µs and `Δc_dl ≈ Δc`. Option C's stage-A saving on prefill is a stable
  **19.3–20.8% of stage A** across all four prefill shapes.
- **D1-lite costs time at one layer on prefill.** `Δdl` is *positive* on all four prefill
  single-calls (+0.57 to +3.23 µs), all of it in the residual. The hoist computes metadata
  unconditionally in the entry computation; with only one call site there is nothing to share it
  with, so it is pure added work. At 21 layers the same hoist turns into −3.9 µs/layer (§2). This
  is the cross-layer-CSE story made measurable inside a single job, and it is the reason the
  headline had to be a chain and not a single call.
- **The single-call twin of the headline under-reports the composed win by 3x.** −2.257 µs/layer
  (−1.32%) at `L1_dec_b64_ctx8704_mml9216` against −6.667 µs/layer (−3.92%) at
  `L21_dec_b64_ctx8704`. Anyone measuring this change one call at a time would price it at a third
  of its value.

`R` is small everywhere: |R| ≤ 1.686 µs/layer and ≤ 0.73% of `inc` at every config, with no sign
pattern. At n = 1 these are indistinguishable from run-to-run noise (§4 puts the noise floor at
~0.25% of `inc`), so the grid supports additivity but does not resolve anything finer.

---

## 4. Within-job replicates and the composition residual

n = 3 replicates of each arm at `L21_dec_b64_ctx8704`, the measurement loop re-entered for each.

### The noise floor

| arm | values (µs/layer) | mean | sd | spread |
| --- | --- | ---: | ---: | ---: |
| `inc` | 170.329, 169.901, 170.336 | 170.189 | 0.249 | **0.256%** |
| `ctl` | 170.347, 170.138, 170.331 | 170.272 | 0.116 | 0.123% |
| `c` | 170.638, 170.588, 170.574 | 170.600 | 0.034 | 0.038% |
| `dl` | 163.383, 163.374, 163.391 | 163.383 | 0.009 | 0.011% |
| `c_dl` | 163.553, 163.565, 163.447 | 163.522 | 0.065 | 0.072% |

Widest spread of any arm: **0.256%**, i.e. **0.435 µs/layer** on the incumbent. Narrowest: 0.011%.
For context, the between-job shift on the same arm was previously measured at 0.63% — more than
twice the widest within-job spread here, which is why all four arms had to be in this one job.

The headline Δ of **−6.667 µs/layer (−3.92%)** is **15x the widest replicate spread**. It is not in
question.

### The three-way composition residual

At `L21_dec_b64_ctx8704`, all four arms measured independently:

| metric | inc | Δctl | Δc | Δdl | Δc_dl | Δc+Δdl | **R** | R % of inc | error bar | inside? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **full** | 170.189 | +0.083 | +0.411 | −6.806 | **−6.667** | −6.395 | **−0.272** | **−0.160%** | ±0.260 | marginal |
| stage A | 111.538 | +0.027 | +0.075 | +0.042 | +0.235 | +0.116 | +0.119 | +0.107% | ±0.066 | no |
| stage B | 32.625 | −0.001 | +0.059 | +0.010 | −0.002 | +0.068 | −0.070 | −0.215% | ±0.020 | no |
| residual | 26.026 | +0.057 | +0.278 | −6.857 | −6.900 | −6.579 | −0.321 | −1.234% | ±0.185 | no |

**Is it inside the error bar?** The two reasonable criteria disagree, and both are reported rather
than the flattering one:

- Against the **quadrature error bar** — the four arms' replicate standard deviations added in
  quadrature, ±0.260 µs/layer — `|R| = 0.272` is **just outside**, by 5%.
- Against the **widest single-arm replicate spread**, 0.435 µs/layer, `|R| = 0.272` is **inside**,
  at 63% of it.

So R sits exactly on the noise floor. The honest statement is: **the two changes are additive to
within this job's ability to measure additivity.** The composed candidate delivers 104.3% of the
sum of its parts on the full op; the 4.3% "bonus" is 0.16% of the op and is not resolvable. Nothing
in the data suggests the changes interfere, and nothing supports claiming a synergy either.

The sub-metric residuals are all outside their (much tighter) error bars, but each is under 1.3% of
its own baseline and they partly cancel. The one worth naming: stage A shows `R = +0.119`, i.e.
`c_dl` costs 0.12 µs/layer more in stage A than `c` and `dl` predict. At decode neither change
should touch stage A at all, so this is second-order compilation drift, not a mechanism.

---

## 5. The fast-path regime — the composed candidate regresses here

Contexts at or below k = 2048, batch 64, at 21 layers. The incumbent's `enable_early_exit`
predicate fires (`max(seq_lens) // compression_ratio <= k`), `_fast_path` returns `iota`, and no
kernel runs at all: stage A and stage B are **0.000** for every arm, so `full == residual`.

| config | inc | `ctl` | `c` | `dl` | **`c_dl`** | Δc_dl | **multiplier vs inc** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L21_dec_b64_ctx1024` | 7.115 | 7.238 (1.017x) | 10.619 (1.493x) | 8.001 (1.125x) | **11.479** | +4.365 (+61.35%) | **1.614x** |
| `L21_dec_b64_ctx2048` | 7.227 | 7.212 (0.998x) | 10.645 (1.473x) | 7.988 (1.105x) | **11.486** | +4.260 (+58.94%) | **1.589x** |

Both changes cost time here, and Option C costs four times what D1-lite does — which is surprising,
because **on the fast path Option C's kernel never executes**.

### Where the +3.5 µs/layer goes, and what causes it

Fine-bucket breakdown at `L21_dec_b64_ctx1024`, µs/layer:

| bucket | `inc` | `ctl` | `c` | `dl` | `c_dl` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cond` (the `conditional` HLO's own self time) | 0.747 | 0.759 | **4.179** | 0.730 | **4.220** |
| `metadata` | 5.634 | 5.744 | 5.712 | **6.532** | **6.527** |
| `predicate` | 0.063 | 0.065 | 0.060 | 0.062 | 0.064 |
| `weights_layout` | 0.612 | 0.612 | 0.614 | 0.617 | 0.614 |
| `boundary_copy` | 0.059 | 0.059 | 0.053 | 0.060 | 0.054 |

D1-lite's +0.89 µs/layer is entirely `metadata` and is the expected trade: the hoist computes
branch-invariant metadata in the entry computation, so it runs even when the branch that needed it
is skipped. It reproduces a previously measured 1.11x at this regime.

Option C's +3.4 µs/layer is entirely the `conditional` op's **own** self time, and the per-instance
distribution is the interesting part. Each program has 21 `conditional` rows; here is every one,
in program order, µs:

```
inc  [0.08, 8.62, 0.34, 0.31, 0.55, 0.36, 0.27, 0.55, 0.23, 0.29, 0.50, 0.28, 0.36, 0.55, 0.28, 0.32, 0.60, 0.32, 0.36, 0.47, 0.04]
c    [0.10, 8.75, 0.17, 8.56, 0.17, 8.68, 0.15, 8.65, 0.08, 8.48, 0.22, 8.66, 0.08, 8.86, 0.13, 8.43, 0.13, 8.68, 0.15, 8.59, 0.04]
```

| arm | conditionals > 3 µs (of 21) | mean of those | total µs/prog | µs/layer |
| --- | ---: | ---: | ---: | ---: |
| `inc` | 1 | 8.624 | 15.68 | 0.747 |
| `ctl` | 1 | 8.545 | 15.93 | 0.759 |
| `c` | **10** | 8.633 | 87.76 | 4.179 |
| `dl` | 1 | 8.449 | 15.33 | 0.730 |
| `c_dl` | **10** | 8.709 | 88.61 | 4.220 |

Identical structure at `ctx2048` (1 / 1 / 10 / 1 / 10, same ~8.6 µs unit), so it is reproducible
and not a one-trace artifact.

The measured facts are: a fixed ~8.6 µs cost, `Bound by = HBM`, attached to the `conditional`
instruction itself rather than to anything in either branch; the incumbent pays it **once** per
program and Option C pays it on **every other layer**. `c` has the same HLO instruction count as
`inc` (8476) and only a slightly larger HLO module (5.36 MB vs 5.32 MB) — the head-major kernel's
extra code is inside the Mosaic custom call, invisible to the HLO instruction count.

**Hypothesis, not established:** this is TPU instruction-memory pressure. The branch program has to
be resident to be entered; the incumbent's fits and is fetched once, and head-major's larger kernel
body pushes the program past what stays resident, so alternating layers pay an HBM program fetch.
The alternating (rather than every-layer) pattern is consistent with two overlays thrashing. This
was not confirmed against IMEM counters and should be labelled a hypothesis by anyone who repeats
it.

### How much this matters

In absolute terms it is small: +4.4 µs/layer × 21 layers ≈ **92 µs per forward pass**, against
7.1 µs/layer × 21 ≈ 149 µs for the incumbent's whole indexer chain in this regime and
170 µs/layer × 21 ≈ 3.6 ms at the production decode point. A 1.6x multiplier on the cheapest
regime the indexer has.

Two caveats in opposite directions. It only bites where context ≤ k = 2048, which at the production
`max_model_len` of 9216 is the first few hundred tokens of a sequence. And this was measured on a
program containing *only* 21 indexer calls; a real decoder layer also carries attention, MLP and
MoE code, so whatever resident-code budget is being exceeded here is under different pressure in
production — the effect could be larger, smaller or absent. This regression is an op-level finding
and does not by itself predict an end-to-end one.

---

## 6. Correctness

`c_dl` must be **index-identical** to `inc` — `np.array_equal`, not a tolerance — against a
separately constructed device call to the incumbent, over a grid chosen to cover every launch path.

`TOTAL rows_compared=34368 rows_differing=0 failures=0`

| config | why it is in the grid | fast path | MIXED launch runs | rows | `array_equal` for `ctl`/`c`/`dl`/`c_dl` |
| --- | --- | :---: | :---: | ---: | :---: |
| `pre_q1024_ctx16384` | MIXED launch, Option C's target | no | yes | 1024 | all True |
| `pre_q1024_ctx4096` | MIXED launch, short ctx | no | yes | 1024 | all True |
| `pre_q4096_ctx4096` | MIXED launch, large q | no | yes | 4096 | all True |
| `pre_q200_ctx4096` | partial `bq` block (200 < 128·2) | no | yes | 200 | all True |
| `pre_q1000_ctx8192` | non-multiple q length | no | yes | 1000 | all True |
| `pre_b2_q512_ctx4096` | 2 sequences, mixed | no | yes | 1024 | all True |
| `dec_b64_ctx8704` | **the production decode point** | no | no | 64 | all True |
| `dec_b32_ctx16384` | second decode point, longer ctx | no | no | 32 | all True |
| `dec_b64_ctx1024` | **fast path fires** (ctx < k) | **yes** | no | 64 | all True |
| `dec_b64_ctx2048` | **fast path fires** (ctx == k) | **yes** | no | 64 | all True |

Rows compared: **34368**. Rows differing: **0**. Elements differing: **0**. Every arm, every
config.

Two supporting checks:

- **The fast-path regime is genuinely exercised.** A numpy closed-form recomputation of
  `_fast_path` (`streamindex_topk.py:840-856`) was compared against the incumbent's own output at
  every config: it matched on exactly the two `ctx ≤ 2048` configs and differed on the other eight.
  So the two configs labelled "fast path" really do take that branch, and the eight labelled
  otherwise really do not — the regime split in §5 is evidenced, not assumed.
- **The incumbent's own kernel test passes on the mapped-in `src/`**:
  `tests/kernels/test_dsv4_streamindex_topk.py` → `16 passed, 14 warnings in 22.60s`. This is
  informational: it tests the incumbent, not the composition, and its value is as a check that the
  mapped-in checkout is healthy.

---

## 7. Provenance — that the right kernels were measured

The benchmark image's kernels are **stale**: `streamindex_topk.py` differs inside the image and
`sparsecore_topk.py` is absent from it entirely. Without a `--map-dir` the job imports stale code,
exits 0, and reports a number for the wrong kernel.

Three directories were mapped in (all three "Packaging dir …" lines confirmed at submit):

| host | pod |
| --- | --- |
| `<checkout>/src` | `/root/cloud-devkit/vllm-torchtpu/src` |
| `<checkout>/kernel_work/streamidx-topk` | `/root/kw` |
| `<kernel5_copy>/.claude/skills/run-command-in-cdk/fanout` | `/root/fanout` |

Read back **on the pod**, on each of the four shards, before any measurement (`readback.sh` exits
non-zero on mismatch and both `perf_shard.sh` and `correctness_shard.sh` abort with rc 90 if it
does):

```
PKG vllm_torchtpu /root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/__init__.py
MD5GATE streamindex_topk.py got=1e221f10cbf3573aa0d2ae78c3463532 want=1e221f10cbf3573aa0d2ae78c3463532 ok=True
MD5GATE sparsecore_topk.py got=8bc9e25b8701a53a97704564b9fbc658 want=8bc9e25b8701a53a97704564b9fbc658 ok=True
MD5GATE_FAILURES 0
```

### The Option C gate, verified two ways

Not assumed from the source. First, the layout map printed at runtime:

```
LAYOUTS head_major_mixed=False {'decode_grouped': 'head_minor', 'decode_mopup': 'head_minor', 'mixed': 'head_minor'}
LAYOUTS head_major_mixed=True  {'decode_grouped': 'head_minor', 'decode_mopup': 'head_minor', 'mixed': 'head_major'}
```

Second, a census of scope names in the **compiled** HLO of every program. At `L21_dec_b64_ctx8704`:

| arm | `StreamIdxTC-hm-` calls | `StreamIdxTC-` (non-hm) calls | HLO computations | HLO instructions |
| --- | ---: | ---: | ---: | ---: |
| `inc` | 0 | 252 | 632 | 8476 |
| `ctl` | 0 | 252 | 632 | 8476 |
| `c` | **84** | 168 | 632 | 8476 |
| `dl` | 0 | 252 | **532** | **7912** |
| `c_dl` | **84** | 168 | **532** | **7912** |

84 = 21 layers × 4 MIXED launch instances; 168 = 21 × 8 DECODE instances. Exactly the MIXED launch
is head-major.

Third, grepping the optimized HLO of a single-call prefill program by hand:

| arm | scopes found |
| --- | --- |
| `inc` | `4 × StreamIdxTC-m-bq_128-bkvp_2`, `8 × StreamIdxTC-d-bq_1-bkvp_2` |
| `c_dl` | `4 × StreamIdxTC-hm-m-bq_128-bkvp_2`, `8 × StreamIdxTC-d-bq_1-bkvp_2` |

The `-hm` suffix appears on the `m` (MIXED) scope and on nothing else. The two DECODE launches are
byte-identical to the incumbent's.

The `dl` arms' drop from 632 to 532 computations and 8476 to 7912 instructions is the cross-layer
CSE D1-lite exists to enable, visible in the compiled program before any timing.

### Job

- **Job id:** `j-bf290089-a5b2-417c-a900`, `simple-run-command`, `tpu7x` `2x2x1` (4 chips, 1 node),
  tag `agent-jobs,run-command-in-cdk`.
- **Image:** `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b`
- **`RUN_CMD`**, as read back from `cdk job recipe`:

```
uv pip install --system -q pytest pytest-mock pytest-timeout parameterized; cd /root/cloud-devkit/vllm-torchtpu && python -c "import vllm_torchtpu; print(vllm_torchtpu.__file__)" && md5sum /root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py /root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py && ls /root/kw/variants/composed && bash /root/fanout/tpu-fanout.sh "bash /root/kw/variants/composed/shards.sh 0" "bash /root/kw/variants/composed/shards.sh 1" "bash /root/kw/variants/composed/shards.sh 2" "bash /root/kw/variants/composed/shards.sh 3"
```

- **Fan-out result:**

```
shard0   chips=0        rc=0    bash /root/kw/variants/composed/shards.sh 0
shard1   chips=1        rc=0    bash /root/kw/variants/composed/shards.sh 1
shard2   chips=2        rc=0    bash /root/kw/variants/composed/shards.sh 2
shard3   chips=3        rc=0    bash /root/kw/variants/composed/shards.sh 3
tpu-fanout: 4/4 shards succeeded
```

  Per shard: `CX/0 failed_specs=0/35`, `CX/1 failed_specs=0/30`, `CX/2 failed_specs=0/30`,
  `CX/3 correctness_failed_steps=0` + `failed_specs=0/20`. 115 specs, 115 xplane files, 0 failures.

- **Shard allocation rule** (`shards.sh`): **every arm of a configuration runs on the same shard.**
  The arms of one config are compared against each other, so per-chip clock or thermal differences
  would land directly in the Δ if they were split across chips. Configs are what get spread; arms
  never are. Within the replicated headline the order is rep-major, so drift over the shard's
  lifetime moves all five arms together and surfaces as replicate spread rather than as a fake Δ.

- **Traced under jax 0.11.0.** jaxlib 0.10.2's profiler client SIGSEGVs against libtpu 0.0.44.1
  (`PLUGIN_Profiler_Api` struct size 80 vs 104), so every shard builds a side venv at
  `/tmp/j11_cx_<shard>` first.

---

## 8. What is deliberately not in the composed candidate

| withdrawn | why it is not here |
| --- | --- |
| the `bkv_p` ladder / `recommended_num_kv_pages_per_block` | Withdrawn from the exercise. `num_kv_pages_per_block` is `(2,2,2)` in every arm including `inc`. |
| **D1** (deleting the `lax.cond`) | Withdrawn as unsafe. D1-lite supersedes it and keeps the `lax.cond` and `_fast_path` **verbatim** — that is why §6 can compare index-identical outputs at fast-path-firing shapes. |
| the KV-cache relayout / v128 change | Out of scope for this job. |
| the earlier "merged op" | Withdrawn; two of its three parts were unsafe. This module is built from the incumbent, not from it. |
| the `dlc` hoist mode (hoist wrapped in an inner cond) | Measured worse than plain `dl` in the D1-lite exercise; dropped, not carried into the composed module. |

### Option B: dropped, did not fit

The optional fifth arm — Option B at G = 4 layered on `c_dl`, at the four prefill configs — was
**not run**. The reason is implementation risk, not shard time.

Option B's chunked launch loop lives in `streamindex_topk_b.py`, which is a fork of the
**withdrawn merged op**: it has no `enable_early_exit` and therefore no `lax.cond`, so porting it
means lifting `chunk_metadata`, `plan_chunks` and the chunked launch (~150 lines) onto a base that
*does* have the cond — and the chunking interacts directly with the D1-lite hoist, since the
hoisted quantities are exactly the per-chunk metadata. Getting that wrong produces a fast arm that
is silently incorrect, and the correctness grid in §6 would have had to be extended to cover it.
Items 1–6 were the priority and Option B was cut to protect them.

What is not affected: nothing in §1–§7 depends on Option B. The B×C sub-additivity question — B's
overlap ceiling being `min(t_a, t_b)/G × (G−1)` with `t_a < t_b`, against a previously realised
83.8% of the additive prediction — is **still open** and is the obvious next measurement, now that
`c_dl` is a measured, verified base to layer it on.

---

## 9. Two defects found along the way

Neither affects any number above; both are recorded so the next person does not rediscover them.

1. **`variants/option-c-head-major/streamindex_topk_c.py` never applies head-major through its own
   entry point.** Its `run_scores_kernel` declares `q_layout="head_minor"` as a default and no call
   site in that file passes the argument, so `streamindex_topk_c.streamindex_topk` silently runs
   head-minor on all three launches. This is pre-existing and predates this job. It is why the
   composed module imports only `_c.kernel` and passes `q_layout` explicitly at all three
   `_common_path` call sites, and why the gate is verified at runtime three ways in §7 rather than
   read off the source.

2. **A cosmetic bug in this job's own census counters.** `MlaCase.symbol` is lowercase (`m`, `d`),
   so the `StreamIdxTC-hm-M-` / `StreamIdxTC-hm-D-` breakdown regexes in `probe_composed.py`
   matched nothing and every log line reads `(M=0 D=0)`. The aggregate `stageA_hm` counter is
   correct and is what §7 quotes, and the gate was proven independently by grepping the optimized
   HLO. Make the regexes case-insensitive before reusing the probe.

3. Minor: the brief locates `runner_op.py` at `variants/option-b-pipeline/`; it is actually at
   `variants/option-b-runner/runner_op.py`.

---

## 10. Verdict

At the production decode point, 21 layers, the composed candidate takes **−6.667 µs per
`streamindex_topk` call (−3.92%)** off the incumbent, 15x the widest replicate spread in the job,
with **0 of 34368 output rows differing** from the incumbent across a 10-configuration grid
including both fast-path-firing shapes and the production shape.

At prefill, 21 layers, it takes **−7.02%**, of which Option C supplies **−20.4% of stage A**.

Across 17 single-call configurations, **none regresses**.

The two changes are **additive to within the job's resolution**: R = −0.272 µs/layer on the full
op, −0.160% of the incumbent, sitting on a noise floor of 0.26–0.44 µs/layer. Neither change is
paying for the other.

One regression is known and bounded: in the fast-path regime (ctx ≤ k = 2048) the composed
candidate is **1.59–1.61x** the incumbent, ~+4.3 µs/layer, all of it an HBM-bound cost on the
`conditional` instruction that Option C makes recur, plus D1-lite's expected unconditional
metadata. The absolute cost is ~92 µs per 21-layer forward pass.

This is an **op-level** result. It says nothing yet about a model step; the changes still have to
clear a preflight and an end-to-end benchmark before any of it counts as a production win, and the
fast-path regression in particular behaves differently in a program that contains more than 21
indexer calls.

---

## Artifacts

| path | what |
| --- | --- |
| `streamindex_topk_composed.py` | The composed module. `for_arm(arm)` returns the `(fn, kwargs)` for any of the five arms. |
| `probe_composed.py` | Compile, census and trace, many specs per process. Spec syntax `<config>:<arm>[:<rep>]`. |
| `correctness_composed.py` | The §6 grid; writes `correctness.json`, exits 1 on any mismatch. |
| `readback.sh` | The provenance gate of §7. Exits non-zero on md5 mismatch. |
| `perf_shard.sh` / `correctness_shard.sh` / `shards.sh` | The four shards and their allocation. |
| `read_composed_hlo_stats.py` | Off-pod reader. `--top-n` defaults to 4000. |
| `results/composed_hlo_stats.md` | Per-trace table (all 115), replicate aggregate, composition residual, fine buckets. Everything above is derived from this. |
| `results/composed_hlo_stats.json` | The same, machine-readable. |
