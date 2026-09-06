# Variant F — `mxu-head-sum`: move the 32-head reduction from the VPU to the MXU

**Verdict: rejected.** The block-diagonal ones-matmul is **+23.26% slower** at the inner step and
**+0.71% / +3.17% / +6.34% / +7.39% slower** end-to-end across the four regimes, every one of
those readable against its drift control. The premise behind F — the MXU is idle, so a reduction
placed on it is free — is measurably **half true and the wrong half**: the MXU's *MAC* pipeline
really is idle (the K-ladder stays flat even under F), but the head-sum-as-matmul does not consume
MAC capacity, it consumes MXU *issue* slots at a 64× worse weight-push ratio than the scoring
matmul, and it hands the VPU a full-tensor `f32 → bf16` cast in exchange for the rotate-add tree
it removes.

One result *does* survive and it is not F: **folding `w` into the matrix operand** is −14.38%
(bf16) / −27.17% (fp8) at the step level, and it wins **despite** the MXU reduction, not because
of it. It is carried below, not claimed here — the `W_BD` construction cost is `unmodelled` and
neither arm is numerically acceptable as measured.

All numbers `measured (j-ace83e22-c472-4463-b41d)`, `JobStatus=Succeeded`, 4/4 shards `rc=0`,
unless tagged otherwise.

---

## 1. Sign convention

Every percentage in this document is a **change in time**, `(new − old) / old`. **Negative is
faster.** It appears only in tables, in a column named `Δ time %`, and every delta sits in the
same row as the two absolute times it came from. A `speedup ×` column sits alongside where useful
and is a different number (>1 is faster). Prose states directions in words.

Every number is tagged `measured (job id)`, `modelled`, `assumed` or `unmodelled`.

---

## 2. The formulation, and why this one

F must keep the ReLU and reduce afterwards — that is what separates it from the `drop-relu`
variant, which is owned elsewhere. Four formulations were on the table.

| # | Formulation | Status |
| --- | --- | --- |
| 1 | **Block-diagonal ones-matmul** — `ONES_BD[q,n] = 1 iff n // 32 == q`, `(128, 4096) @ (4096, 2048) → (128, 2048)`. This is E6/H from `measurement_and_design.md` §2.5. | **Chosen as primary.** Exact by construction at the algebra level (it is a pure reassociation of the same 32 terms), needs no property of `w`, and needs no relayout of anything. Measured as `F1_iota` (matrix built in-kernel from `broadcasted_iota`) and `F1_ref` (matrix handed in). |
| 2 | **Fold `w` into the same matmul** — `W_BD[q,n] = w[q, n % 32] iff n // 32 == q`. | **Measured as a ceiling only** (`F2_wfold_bf16`, `F3_wfold_fp8`), with `W_BD` handed in pre-built. Building it in-kernel from a `(bq_sz, num_q_heads)` `w` needs a lane↔sublane relayout of a 2 MiB tensor, so its construction cost is `unmodelled` and belongs on the producer side (candidate C13's route). Unlike candidate C8's E1 arm, it needs **no sign guarantee on `w`** — which matters, because `w` is ~50% negative on the real GLM-5.2-FP8 checkpoint (0 of 32 rows all-non-negative), and that is exactly what killed E1. |
| 3 | **Reshape the head axis into the existing einsum as a contraction dimension.** | **Impossible, ruled out analytically, no probe needed.** The ReLU sits *between* the two contractions. `sum_h relu(s[h]) ≠ relu(sum_h s[h])`, so the head axis must be materialised before it can be reduced. This is precisely what makes F an exact reassociation and the `drop-relu` variant an approximation. |
| 4 | **bf16×2 split-float**, to buy back precision on the MXU. | **Ruled out analytically.** Two matmuls plus ~4 VPU ops per vreg to split the summands, against the ~2.25 vreg-ops per vreg the reduction it replaces costs (area 5's model, `modelled`). It pays more VPU than it saves and doubles the MXU. Self-defeating before it is written. |

The MXU is only reachable through bf16/fp8 operands, so precision arms were added deliberately
rather than assumed: `F4_ones_f32` (f32 with `Precision.DEFAULT`) and `F5_ones_f32_highest`
(f32 with `Precision.HIGHEST`). §3 shows why that mattered.

---

## 3. What the two matmuls actually reach the MXU with — confirmed, not inferred

Static compile-time analysis from `llo_f.py`, `mixed_prod`, program `StreamIdxTC-m-bq_128-bkvp_2.1`
(the MIXED band). **Nothing in this section is a timing.**

| impl | `vmatmul` total | bf16 | f8 | f32 | bf16 weight pushes |
| --- | --- | --- | --- | --- | --- |
| incumbent | 1024 | 0 | 1024 | 0 | 0 (`16 × vmatpush3.f8e4m3` per MXU) |
| variant (F1, bf16) | 2048 | **1024** | 1024 | 0 | **1024 `vmatpush1.bf16` per MXU, ×2 MXUs** |

Three things follow, and the third is the finding.

1. **The head-sum matmul is emitted as bf16.** F's exactness claim does not survive the hardware:
   the summands are rounded to bf16 before they are added. This is confirmed at the opcode level,
   not inferred from the JAX dtype — the whole point of running this shard.
2. **The scoring matmul stays native fp8** (1024 `vmatmul.mubr.f8e4m3fn.*` in both impls),
   reproducing P1 `measured (j-689b87cd-7460-4b90-bd87)`. F did **not** demote it, which is the
   one catastrophic outcome that was on the table (a bf16-operand scoring path is a 1.86×
   regression). Good news, and it means the +23% below is the head-sum matmul's own cost.
3. **The push-to-matmul ratio is 64× worse for the reduction than for scoring.** Scoring issues
   32 weight pushes total for 1024 `vmatmul`s (0.031 pushes per matmul): the `q` operand is
   small and stationary, `kv` streams. The reduction issues 2048 pushes for 1024 `vmatmul`s
   (2.0 per matmul): its stationary operand changes every issue. The reduction matmul is
   **push-bound, not MAC-bound**, and those pushes contend with the scoring matmul for the same
   two MXUs. This is the mechanism, and it is what the "3.1% MXU fill" objection in §2.5 was
   groping at without naming.

The DECODE programs show the same pattern at 1/32 the scale (`-d-bq_1-bkvp_2.2`: 32 f8 + 32 bf16
`vmatmul`, 64 bf16 pushes; incumbent: 32 f8, 0 bf16).

VMEM was **not** a constraint: the variant's allocation list is unchanged from the incumbent's
(same five scoped allocations, same sizes) — no Mosaic or VMEM rejection occurred anywhere in
this job.

---

## 4. Step level — all seven formulations, one process, same instrument

`step_probe.py`, grid=32, `bq_sz=128`, `heads=32`, `bkv_sz=2048`, `head_dim=128`. `Δ time %` is
against the `incumbent` arm at K=128 measured **first in this same process**. **Negative is
faster.**

| arm | K | us/step | spread % | baseline us/step | Δ time % | speedup × |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent | 128 | 7.276 | 3.23 | 7.276 | +0.00 | 1.000 |
| incumbent_K256 | 256 | 7.295 | 2.04 | 7.276 | +0.26 | 0.997 |
| **F1_iota** (ones-matmul, bf16, built in-kernel) | 128 | **8.968** | 2.09 | 7.276 | **+23.26** | 0.811 |
| F1_iota_K256 | 256 | 8.925 | 3.85 | 7.276 | +22.67 | 0.815 |
| F1_ref (ones-matmul, bf16, handed in) | 128 | 8.439 | 5.04 | 7.276 | +16.00 | 0.862 |
| F2_wfold_bf16 (`W_BD` handed in) | 128 | 6.229 | 0.57 | 7.276 | −14.38 | 1.168 |
| F3_wfold_fp8 (`W_BD` handed in) | 128 | 5.299 | 7.77 | 7.276 | −27.17 | 1.373 |
| F4_ones_f32 (`Precision.DEFAULT`) | 128 | 8.428 | 1.64 | 7.276 | +15.84 | 0.863 |
| F5_ones_f32_highest (`Precision.HIGHEST`) | 128 | 25.053 | 2.58 | 7.276 | +244.35 | 0.290 |
| `sum_vpu_CONTROL` | 128 | 5.008 | **10629.04** | 7.276 | −31.16 | 1.453 |
| `sum_ones_bf16_CONTROL` | 128 | 6.256 | **116.27** | 7.276 | −14.01 | 1.163 |
| incumbent_repeat (**drift control**) | 128 | 7.580 | 7.90 | 7.276 | +4.18 | 0.960 |

**Drift control: 7.276 → 7.580 us/step, `Δ time +4.18%`.** Any arm-to-arm delta smaller than
|4.18%| is not readable as an effect. F1_iota's +23.26% is 5.6× that and is readable. F1_ref
(+16.00%), F4 (+15.84%), F2 (−14.38%), F3 (−27.17%) and F5 (+244.35%) are all readable.
`incumbent_K256` (+0.26%) and the F1 K-ladder (−0.48%) are **not** readable as effects, which is
the point of §4.2.

### 4.1 The two `CONTROL` rows are unusable and are not quoted as a result

`sum_vpu_CONTROL` returned **10629% spread** and `sum_ones_bf16_CONTROL` **116%**. They were
designed to isolate the reduction by differing in exactly one thing (VPU rotate-tree vs MXU
matmul, everything else held), and the arithmetic on them — 5.008 → 6.256 us/step, `Δ time
+24.91%` — points the same way as F1 vs incumbent. **It is nonetheless not evidence**: at that
spread the medians are noise, most likely because the stripped arms let the compiler DCE part of
the body. Recorded here so nobody re-derives it and believes it. The conclusion in this document
rests entirely on the F1-vs-incumbent pair, whose spreads are 2.09% and 3.23%.

### 4.2 The MXU is still off the critical path — and that is the surprise

Doubling K doubles the scoring matmul's FLOPs and changes nothing else.

| arm | K=128 us/step | K=256 us/step | Δ time % |
| --- | --- | --- | --- |
| incumbent | 7.276 | 7.295 | +0.26 |
| F1_iota | 8.968 | 8.925 | −0.48 |

Both are inside the 4.18% drift band, so both read as **free** — reproducing shard 3's +2.18%
`measured (j-b54e9716-3db5-4f1f-b8d4)`. The scoring matmul's MAC time is still fully hidden
**even after F adds a second matmul of identical size** (512 MXU tile-ops, 2.147 GFLOP each).

So F's cost is **not** MXU MAC throughput. The MXU had the arithmetic capacity the design argued
it had. What it did not have was free *issue bandwidth*: §3's 2.0-pushes-per-matmul ratio, plus
the `f32 → bf16` cast of the whole `(128, 32, 2048)` score tensor, which lands on the VPU — the
resource F set out to relieve.

### 4.3 Precision: `f32 DEFAULT` is bf16, confirmed by numerics as well as by rate

Max deviation vs the incumbent's own output, same inputs, same kernel, one difference.
`norm_err` = `max|v − ref| / max|ref|`; a plain elementwise relative error is meaningless here
because `w` is ~50% negative, so `s_summed` passes through zero.

| arm | norm_err | rel err on top 1% | bit-identical | ±inf pattern |
| --- | --- | --- | --- | --- |
| incumbent | 0.000e+00 | 0.000e+00 | True | — |
| F1_iota | 2.457e-03 | 1.010e-01 | False | match |
| F1_ref | 2.457e-03 | 1.010e-01 | False | match |
| F2_wfold_bf16 | 3.750e-03 | 1.998e-01 | False | match |
| F3_wfold_fp8 | 4.803e-02 | 2.188e+00 | False | match |
| **F4_ones_f32** (`DEFAULT`) | **2.457e-03** | **1.010e-01** | False | match |
| **F5_ones_f32_highest** | **1.476e-07** | 3.980e-06 | False | match |

Reference: pure reassociation of the 32-term head sum is bounded by **2.46e-06** relative
`verified (j-ed517191)`, four decades below the 61-rank churn already present.

`F4_ones_f32` reproduces `F1_iota`'s error **to four significant figures** — 2.457e-03 in both.
That is independent confirmation that `f32 × Precision.DEFAULT` on this MXU is not f32 arithmetic
at all but bf16 internally, exactly as area 5's rate table says (`modelled` 661.4 Tops/s for f32
DEFAULT vs 764.8 for bf16). Only `F5` (`HIGHEST`) lands inside the reassociation bound at
1.476e-07 — and it costs +244.35%, i.e. the 4.5× f32-HIGHEST cliff arriving on schedule.

**The exactness/speed trade is therefore closed, not open:** the only formulation of F that
preserves exactness is 3.4× slower than the incumbent. F's selling point was exactness and the
hardware does not sell it at any price worth paying.

---

## 5. Four-regime A/B on the whole kernel

`bench_ab.py`, median ms per `streamindex_topk` call, single TensorCore, `early_exit=False`,
`reps=10`. Incumbent is the **shipped** `vllm_torchtpu.kernels.deepseek_v4.streamindex_topk`,
not the variant module with the knob off. Variant is `F_MXU_HEAD_SUM=1 F_DTYPE=bf16`.
**Negative is faster.**

| regime | arm | full_ms | spread % | baseline_ms | Δ time % | speedup × |
| --- | --- | --- | --- | --- | --- | --- |
| decode_prod | incumbent | 0.16693462 | 0.41 | 0.16693462 | +0.00 | 1.000 |
| decode_prod | **variant** | **0.16812285** | 0.45 | 0.16693462 | **+0.71** | 0.993 |
| decode_prod | incumbent-repeat | 0.16625648 | 0.71 | 0.16693462 | −0.41 | 1.004 |
| decode_64k | incumbent | 0.20087782 | 0.29 | 0.20087782 | +0.00 | 1.000 |
| decode_64k | **variant** | **0.20724762** | 1.01 | 0.20087782 | **+3.17** | 0.969 |
| decode_64k | incumbent-repeat | 0.20115735 | 0.42 | 0.20087782 | +0.14 | 0.999 |
| mixed_prod | incumbent | 0.67006801 | 2.81 | 0.67006801 | +0.00 | 1.000 |
| mixed_prod | **variant** | **0.71251987** | 1.58 | 0.67006801 | **+6.34** | 0.940 |
| mixed_prod | incumbent-repeat | 0.66785100 | 2.56 | 0.67006801 | −0.33 | 1.003 |
| mixed_64k | incumbent | 4.79921508 | 22.25 | 4.79921508 | +0.00 | 1.000 |
| mixed_64k | **variant** | **5.15369731** | 0.86 | 4.79921508 | **+7.39** | 0.931 |
| mixed_64k | incumbent-repeat | 4.78990641 | 1.24 | 4.79921508 | −0.19 | 1.002 |

**Drift control, per regime** — variant delta vs incumbent-to-incumbent-repeat gap:

| regime | incumbent ms | incumbent-repeat ms | drift Δ time % | variant Δ time % | readable? |
| --- | --- | --- | --- | --- | --- |
| decode_prod | 0.16693462 | 0.16625648 | −0.41 | +0.71 | READABLE |
| decode_64k | 0.20087782 | 0.20115735 | +0.14 | +3.17 | READABLE |
| mixed_prod | 0.67006801 | 0.66785100 | −0.33 | +6.34 | READABLE |
| mixed_64k | 4.79921508 | 4.78990641 | −0.19 | +7.39 | READABLE |

The drift control is tight in this shard (|0.14| to |0.41|%), so all four regressions are real.
`decode_prod`'s +0.71% is the marginal one — 1.7× drift — and is quoted as "slower, small" rather
than as a precise figure.

`mixed_64k`'s incumbent arm carries **22.25% spread**; its +7.39% should be read as "clearly
slower", not as 7.39 ± anything.

**Cross-check against §4**, `mixed_prod`: the variant adds 0.04245 ms to a call whose
scoring-by-difference is 0.26979 ms (`= full 0.67007 − topk-only CONTROL 0.40028`), i.e. +15.7%
of the scoring component, against the step probe's +23.26%. Same sign, same order; the two
harnesses do not share a grid so an exact match was not expected. Nothing here needs the
scoring-by-difference figures to be trusted — they are the `sparsecore_topk` control's
subtrahend and `report.md` already marks the long-KV ones UNUSABLE (the `mixed_64k` topk control
came back at 66.23% spread in this job too).

---

## 6. Step attribution against the target, and the gap

The brief's target: if F removes most of the 3.858 µs head-sum bundle from a 7.343 µs step and
adds MXU time off the critical path, the step should approach **~3.5–4 µs**.

| quantity | value | source |
| --- | --- | --- |
| reference step, shard 3 | 7.343 µs | `measured (j-b54e9716-3db5-4f1f-b8d4)` |
| incumbent step, this job | 7.276 µs | `measured (j-ace83e22)` |
| target for F | ~3.5–4 µs | `modelled` (the brief) |
| **F1_iota, measured** | **8.968 µs** | `measured (j-ace83e22)` |
| **gap vs target** | **+5.0 to +5.5 µs** | arithmetic on the two rows above |

The incumbent step reproduces to within 0.9% of shard 3's, so the harnesses agree and the gap is
F's, not the instrument's. Accounting for all of it:

| term | µs | how it is known |
| --- | --- | --- |
| VPU head-sum removed (the prize) | ≤ 3.5 | `modelled` — the 3.858 µs bundle minus ~0.35 µs of output store that F does not remove. **Not independently measured in this job**: the arm that would have isolated it is the unusable `sum_vpu_CONTROL` of §4.1. This is the one term in this table that is not measured, and it is the term F was buying. |
| bf16 matmul issue + weight-push serialisation | dominant, not separable | `measured` at the opcode level (§3: 2.0 pushes/matmul vs 0.031) but not separable in time from the next row by this instrument. |
| `f32 → bf16` cast of the `(128, 32, 2048)` score tensor, on the VPU | included above | `measured` in aggregate; `unmodelled` individually. |
| ones-matrix construction in-kernel | **0.529** | `F1_iota 8.968 − F1_ref 8.439`. Readable (7.3% > 4.18% drift). Avoidable by handing the matrix in, and F1_ref is still +16.00%, so removing it does not save F. |
| net | **+1.692** | `F1_iota 8.968 − incumbent 7.276` |

So: the reduction did move off the VPU, and the MXU did have spare MAC capacity for it (§4.2),
and the move still cost 1.692 µs. **What the roofline modelled as an idle unit was idle in the
wrong dimension.** MXU occupancy in this kernel is set by weight-push issue, not by MACs, and a
reduction-as-matmul is the worst possible shape for that: its stationary operand is the data, so
it re-pushes every issue. The 3.1% fill number in §2.5 was directionally right for the wrong
reason, and the brief was right that fill was the wrong criterion — but the right criterion turned
out to be push ratio, not overlap.

---

## 7. Correctness

`correctness.py` runs the READ-ONLY `instruments/test_correctness.py` unmodified on **both**
impls: float64 oracle, `RANK_SLACK=256`, `MIN_RECALL=0.99`, `k=2048`, allowed `worst_rank=2304`.
All ten runs PASS. Negative controls fired on **64 of 64** and **1024 of 1024** rows for both
impls, so the check is not vacuous.

| regime | incumbent worst_rank / min_recall | variant worst_rank / min_recall | identical? |
| --- | --- | --- | --- |
| decode_prod | 2109 / 0.99561 | 2109 / **0.99512** | no |
| mixed_prod | 2098 / 0.99512 | 2098 / 0.99512 | **yes** |
| decode_64k | 2105 / 0.99512 | 2105 / **0.99365** | no |
| mixed_32k | 2140 / 0.99316 | **2146** / **0.99268** | no |
| mixed_64k | 2169 / 0.99219 | **2170** / **0.99072** | no |

The incumbent margins reproduce the brief's reference exactly — `2109 / 0.99561` at `decode_prod`
and `2098 / 0.99512` at `mixed_prod` — so the gate is calibrated.

**Four of five regimes are NOT bit-identical, and per the brief that is a bug rather than a
tolerance. It was investigated, and the cause is §3:** the emitted matmul is bf16, so F as built
is not an exact reassociation at all — it rounds every summand to bf16 before adding. The observed
2.457e-03 numeric deviation (§4.3) is three decades above the 2.46e-06 reassociation bound and is
exactly the size bf16 rounding predicts. The margin movement is the expected consequence of a
precision reduction that was never intended, not an unexplained discrepancy.

The gate still passes with margin in every regime, so F is not *incorrect*. It is **not exact**,
which was its entire justification.

---

## 8. Carried result — folding `w` into the matrix operand

This is **not** F and is not claimed as F's result. It is a measurement F's arms happened to buy,
and dropping it would waste it.

| arm | us/step | baseline us/step | Δ time % | speedup × | numerics (norm_err) |
| --- | --- | --- | --- | --- | --- |
| F1_ref (ones, bf16, handed in) | 8.439 | 7.276 | +16.00 | 0.862 | 2.457e-03 |
| **F2_wfold_bf16** (`W_BD`, bf16, handed in) | **6.229** | 7.276 | **−14.38** | 1.168 | 3.750e-03 |
| **F3_wfold_fp8** (`W_BD`, fp8, handed in) | **5.299** | 7.276 | **−27.17** | 1.373 | 4.803e-02 |

`F2 − F1_ref = −2.210 µs`: folding `w` into the matrix removes the VPU `w`-multiply, and that
saving is **larger than everything the MXU reduction costs**, which is why F2 wins overall while
F1 loses. Both readable against the 4.18% drift.

Three conditions before this is worth anything, and none is met yet:

- **`W_BD` construction is `unmodelled`.** Both arms are handed the matrix pre-built. Building it
  in-kernel needs a lane↔sublane relayout of a 2 MiB tensor; the viable route is producer-side,
  i.e. candidate **C13**. The numbers above are a **ceiling**, not an offer.
- **Neither arm's numerics is acceptable as measured.** F2's 3.750e-03 is bf16 rounding and F3's
  4.803e-02 is fp8; neither went through the §7 gate. F3's `rel_err_on_top_1pct = 2.188` means
  errors comparable to the values themselves on the largest scores — the arm most likely to move
  ranks is the fastest one, which is the classic shape of a hollow win.
- It needs **no** sign guarantee on `w`, which is the one thing it has over C8's dead E1 arm.

Next action: hand the F2 shape to whoever owns **C13**, as "producer-side `W_BD` is worth up to
−14.38% at the step level `measured (j-ace83e22-c472-4463-b41d)`, gated on construction cost and
on passing §7's correctness gate."

---

## 9. Caveats that survive

- **This is a single-chip microbenchmark, not an end-to-end number.** §5 measures one
  `streamindex_topk` call on one TensorCore. Nothing here says what the indexer costs in a served
  GLM-5.2 step, and nothing here should be quoted as if it did.
- **Every absolute time in this job is contended.** Four shards shared one host's HBM and 16
  CPUs. Ratios taken *inside* one shard are the trustworthy quantity, and every conclusion above
  is built on one. The absolute µs/step and ms/call figures are not comparable to another job's.
- **The two `CONTROL` arms in §4 are unusable** (10629% and 116% spread) and their difference is
  not quoted as evidence. The consequence is §6's one unmeasured term: the VPU head-sum's own
  absolute cost in *this* harness was never isolated, so the "prize" side of the attribution
  rests on shard 3's 3.858 µs `measured (j-b54e9716-3db5-4f1f-b8d4)` rather than on this job.
- **`mixed_64k`'s incumbent arm has 22.25% spread**, and its `sparsecore_topk` control 66.23%.
  Its +7.39% is a direction, not a magnitude.
- **Everything in §3 is static compile-time analysis.** The `vmatmul` and push counts are opcode
  counts from `*-71-final_bundles.txt`; they are not timings and the LLO-instrumented wall-clock
  printed alongside them is contended and is not quoted anywhere.
- **The `xprof-cli` LLO readers were not used** — their pinned repo 404s under an authenticated
  `gh` and `setup.sh` exits 0 on that failure. The raw dumps were read instead, the substitution
  `probes/llo_capture_RESULT.md` already records.
- **Mosaic's `post-infer-vector-layout` dump produced no `StreamIdxTC` file** in either impl, so
  the `tpu.matmul` type lines are absent from this job. The operand-type conclusion in §3 rests
  on the LLO opcode names (`vmatmul.mubr.bf16.*` vs `vmatmul.mubr.f8e4m3fn.*`), which are at
  least as direct. No `tpu.matmul` type line is quoted here because none was captured.
- **No Mosaic or VMEM rejection occurred.** Every arm compiled and ran; VMEM allocations are
  byte-identical between impls.
- `early_exit=False` throughout, matching step 1's baseline table. Production sets it True
  (`mla_attention_op.py:57`); it is held constant across arms either way.

---

## 10. Provenance

| | |
| --- | --- |
| job | `j-ace83e22-c472-4463-b41d`, `job_status=Succeeded`, `state=Complete` |
| fan-out | `shard0 rc=0`, `shard1 rc=0`, `shard2 rc=0`, `shard3 rc=0` — 4/4 succeeded |
| image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694` |
| `RUN_CMD` (read back from `cdk job recipe`) | `export PYTHONPATH=/workspace:$PYTHONPATH && bash /root/fanout/tpu-fanout.sh "bash /workspace/kernel_work/streamindex_topk/variants/mxu-head-sum/shard.sh 0" "... 1" "... 2" "... 3"` |
| `--map-dir` | `/home/annyan_google_com/workspaces/vllm-torchtpu/kernel_work:tpu-job:/workspace/kernel_work` and `.../run-command-in-cdk/fanout:tpu-job:/root/fanout` |
| `vllm-torchtpu` HEAD at submit | `eb9946456ae0d26a302dc7b388389d0ca8f828d2`, branch `annyan/streamidx-scores` |
| working tree at submit | dirty — other agents' variant dirs (`narrow-record`, `right-size` modified; `half-width-key`, `mxu-head-sum`, `optimization_work/` untracked). No shared source was modified by this variant; the kernel change lives only in `mxu_head_sum.diff` and `streamindex_topk_mxu.py` under this directory. |
| `~/cloud-devkit` sha | `4c6383fbd1578d56afa876f56904535715155cd2`, dirty (`?? recipes/experimental/gdn-comparison.yaml`) |
| jax / devices | `jax 0.10.2`, `TpuDevice(id=0..1, coords=(0,0,0))` per shard — one chip, two cores, pin confirmed by 4/4 concurrent `rc=0` |

Files in this directory: `step_probe.py` (§4), `bench_ab.py` (§5), `correctness.py` (§7),
`llo_f.py` (§3), `shard.sh` (dispatcher), `streamindex_topk_mxu.py` + `mxu_head_sum.diff`
(the kernel change), `_load.py`.
