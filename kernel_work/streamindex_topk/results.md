# `streamindex_topk` — step 3 results

Step 3 of `/write-tpu-kernels`. One folder per candidate under `variants/`, each measured against
the incumbent, plus the ranking and the recommendation. The incumbent is a candidate and is allowed
to win.

**This document is in progress.** Variants A, B, C, F and A1 are in flight. What is recorded below
is settled and will not move: two of step 2's four `unmodelled` ranking defects are now closed, one
option is dead on measurement, and two options that are not in `measurement_and_design.md` have been
opened by a measurement that landed after it was written.

## 0. How to read this

### 0.1 Sign convention — binding on every number in this file

Every percentage is a change in **time**, `(new − old) / old`. **Negative is faster.**

Signed percentages appear **only** in a table column headed `Δ time %`, and every such row carries
the two absolute times the delta came from. Prose never carries a signed percentage; it says the
direction in words. A `speedup ×` column may sit alongside, where > 1 is faster; it is a different
number from the percentage (50% faster is 2.00×).

### 0.2 Tags — every number carries exactly one

`measured (job id)` came off hardware in a named job · `modelled` computed from a rate or a byte
count · `assumed` taken on belief, with the belief stated · `unmodelled` **the floor has no term for
it**, the word appears alone and there is never a number beside it.

### 0.3 Terms this document adds

- **drift control** — two arms in one job running *identical* code, measured as if they were a
  comparison. Its delta is the noise floor for an arm-to-arm comparison, and it is a **different and
  larger** quantity than the `spread %` within a single arm. §1.2 is why the distinction earned its
  own term.
- **the A/B shape** — the end-to-end benchmark this exercise feeds: GLM-5.2, ISL 65536, OSL 1024,
  `max_model_len` 66560, `--max-concurrency=64`, DP=16.
- **inner step** — one `(bq, bkv)` iteration of the kernel's `fori_loop` nest, ≈8.42 µs at the mixed
  band on the shipped kernel. Not a `pallas_call` grid step; there are 32 inner steps per grid step
  at `mixed_64k`.

---

## 1. Variant E — Retune (`bkv_p` (2,2,2) → (4,2,4))

`variants/retune/`, committed `e7e74ae4`. **Measured, correctness PASS, and its standing has
fallen sharply** — not because the kernel measurement moved, but because two things around it did.

### 1.1 The measurement

`measured (j-ce9a19a4-4639-4a8f-9059)`, `Succeeded`, 4/4 shards `rc=0`. Incumbent and variant in the
same process on the same chip across all four regimes — a pairing no prior job had. Median ms per
full call, single chip. **Negative is faster.**

**`pooled`** means the arithmetic mean of the two replicate arms of that kind in this job, per
`variants/retune/RESULTS.md` §3.1 — not a single arm. The distinction matters on the one row where
it does not hold: †`decode_64k`'s incumbent is the **first arm alone**, because its
`incumbent-repeat` was excluded (below).

| regime | incumbent (2,2,2) ms, **pooled** | variant (4,2,4) ms, **pooled** | Δ time % | speedup × |
| --- | ---: | ---: | ---: | ---: |
| `decode_prod` | 0.16582267 | 0.15299875 | −7.73 | 1.0838 |
| `decode_64k` † | 0.19944237 | 0.17761440 | −10.94 | 1.1229 |
| `mixed_prod` | 0.66758930 | 0.68516149 | +2.63 | 0.9744 |
| `mixed_64k` | 4.78415255 | 4.88059295 | +2.02 | 0.9802 |

† `decode_64k`'s `incumbent-repeat` arm returned `full_spread_pct = 111.716103128099` and is
unusable, so **this job establishes no drift bound at `decode_64k`**. That row rests on its two
variant arms agreeing (−10.83%, −11.06%) and on the prior job's −11.33%.

Correctness at `(4,2,4)`: PASS at `decode_prod` (`worst_rank=2109 of allowed 2304,
min_recall=0.99561`), `decode_64k` (`2105`, `0.99512`) and `mixed_prod` (`2098`, `0.99512`), all
**bit-identical to the incumbent's margins**, with the negative control firing 1024/1024 at
`worst_rank=8159, min_recall=0.42578` — missing by 3.5×, so the PASSes are not vacuous.

**Ships as `(4,2,4)`, not `(4,4,4)`.** The two measure identically (replicate gaps −0.62 / −0.25 /
+0.57 / −0.32%, no consistent sign), so the decision falls to the source: slot 1 is PREFILL, read by
no band that executes (`streamindex_topk.py:765/783/801` read slots `[0]`, `[0]`, `[2]`), yet it
still enters the `max` at `:603-605` and can only ever widen the shared scores buffer. Free today
because all three slots are 4; not free the first time anyone tunes the slots apart, which is what
both Option B and `probes/bkv_p_RESULT.md` propose.

### 1.2 Two corrections to the probe record, both against E

**The mixed regression is real, not noise.** `probes/bkv_p_RESULT.md`'s strongest reading was *"not
measurably slower at mixed"*, reached by comparing each delta against the **within-arm `spread %`**.
That is the wrong yardstick for an arm-to-arm delta. This job carried the drift control that one
lacked: it reads **−0.03% / −0.77% / +0.01%**, and the mixed deltas clear it in every arm. Eight
mixed arms across three jobs are now all on the slower side. **Price ~2% in; do not round it to
zero.**

**`decode_prod` is 7.73% faster, not the 9.25% on the record — and the cause is the baseline, not
the variant.** The variant reproduces across jobs to 0.55%, while this job's incumbent is 1.41%
faster than `j-98399433`'s, more than either arm's spread. `decode_64k` and both mixed incumbents
reproduce to within 0.15%. This is the failure mode of comparing across jobs, and it is why the
same-job pairing was worth buying.

### 1.3 What E is worth end to end: within ±1% of zero, sign unresolved

From the decode:mixed split in §2, E's decode prize and mixed cost very nearly cancel. Breakeven is
at a mixed regression of `m* = d · w_decode / w_mixed` = **1.60–1.95%** for `d = −11.69%`, and the
measured mixed regressions **straddle it**: `+0.97%` and `+1.61%` below, `+1.67%` on it, and this
job's drift-controlled `+2.02%` / `+2.63%` above. Net spans **−0.84% to +0.90%** `modelled`.

Neither end of the decode:mixed bracket flips the sign — the bracket is not what is uncertain. What
is uncertain is the mixed regression itself, measured against a breakeven it sits on top of.

**E is therefore not the safe one-line win the step-2 ranking had at joint rank 1.** Its ceiling is
still Option B, for the reason §4.2 of the design doc gives: the ~2% mixed cost is entirely the
`max`-coupled scores width, which is exactly the term B's masked partial-block store decouples.
**Measure E composed with B, not summed.**

---

## 2. The decode:mixed split — step 2's flipping assumption, settled

`probes/step3_decider_RESULT.md`, shard 0, committed `879b5d23`. **`modelled`** — derived from the
scheduler's chunking arithmetic at the A/B shape, not observed. An instrumented dispatch would make
it `measured` and is an open shard.

At the A/B shape: 4 requests per engine; a prefill step carries 3 decode tokens and **1021** prefill
tokens. Prefill balance `1021M = 65536R → M = 64.1881R`; decode balance
`4(N−M)+3M = 1023R → 4N = 1087.1881R → N = 271.7970R`; `M/N = 0.236168`. Cross-check on slot
lifetime: `4N/R = 1087.19 = 64.19 + 1023`. ✓

| band | share of **steps** | share of **time** |
| --- | ---: | ---: |
| DECODE | 76.4% | **12.1–14.3%** |
| MIXED | 23.6% | **85.7–87.9%** |

`share of steps` is the fraction of engine steps of each kind; `share of time` is that fraction
weighted by the per-call time of the corresponding regime. The two columns are now on the same
denominator, which they were not. `N` is the **total** step count and `M` the mixed count — the
cross-check above only closes on that reading — so the decode-only share is
`(N−M)/N = 1 − 0.236168 = 76.4%`, not the `M/(N+M) = 19.1%` complement an earlier revision printed
in this column.

**Only the step column changes.** The time bracket was already computed on `(N−M)/N`, so
12.1–14.3% / 85.7–87.9% stands unaltered, and everything weighted by it downstream — A's
0.93–1.10% in §5b-v2, E's breakeven in §1.3, §6's ranking — is unaffected. Had the 80.9/19.1
weights been used for time as well, the bracket would read 15.2–17.9% / 82.1–84.8%; they were not.

The time bracket comes from `decode_64k` at 0.20202844752930105 ms against `mixed_64k` at 4.76284 ms
(high) and 3.91366 ms (low — mean `num_bkv` over the 64 prefill chunks is exactly 16.5 of 32):
`0.7638 × 0.20203` against `0.2362 × 4.76284` gives 12.1%, and against `0.2362 × 3.91366` gives
14.3%.

**Consequence, in the design doc's own §9.2 words: mixed dominates, so "B is rank 1 outright, with A
second", and E is near zero.** §9.1's printed `E ≈ B` tie does not survive.

A source finding the design doc does not have, from the same shard: in a prefill step
`chunk_num_decode = 3`, so `decode_batch_end = 3//4*4 = 0` and **the batched decode call is empty** —
all three decoders take the `seq_batch_size=1` remainder path. Two of the three bands that exist run;
the one that runs most often is the remainder path, not the batched one.

---

## 3. Option D — No Round Trip: **dead on measurement**

`probes/step3_decider_RESULT.md`, shard 2. Design doc §9.4 item 1 called the TensorCore-side
top-2048-of-67584 cost the one `unmodelled` term that spanned the entire ranking: struck, D wins by
58.82%; priced above **~394 µs**, D loses. It is now priced.

1024 rows, `k = 2048`, against a same-job `sparsecore_topk` reference. All
`measured (j-b54e9716-3db5-4f1f-b8d4)`.

| width | `sc_topk` µs | `xla_topk` µs | `xla_approx` µs | `pass_count` µs | `pass_copy` (control) µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 541.71 (4.29%) | 4010.37 (0.52%) | 3530.22 (0.62%) | 153.62 (7.56%) | 151.28 (10.20%) |
| 67584 | 3071.85 (0.60%) | 46132.06 (0.90%) | 41751.20 (0.82%) | 232.31 (6.89%) | 188.74 (5.70%) |

The cheapest complete TensorCore formulation at `mixed_prod`'s width is **3530.22 µs — 8.96× the
~394 µs threshold**, and still 6.6× over after deflating by this job's 1.356 contention factor
(`sc_topk` reads 541.71 here against 399.60 µs at `j-5f8f4a08`).

**The strike does not fire. D's −58.82% is withdrawn. D stays last, and no variant folder is built
for it** — it was ~1790 lines, between one and two orders of magnitude more work than any other
option, justified only by a term that has now come back an order of magnitude the wrong way.

The reusable primitive, which is why this probe was worth running as a primitive rather than as a
variant: **compare-and-reduce is nearly free** (2.34 µs at 10240, 43.57 µs at 67584) — the traffic is
everything. One pass at 10240 costs 153.62 µs, so the 394 µs budget buys **two** passes, and no
top-2048-of-10240 exists in two passes. A 32-step bisection is `modelled` at 4915.9 µs. No multi-pass
TensorCore scheme reaches the threshold.

⚠ One arm of that shard, `pallas_topk`, errored on a malformed `BlockSpec` of `(1, W)`:
`"The Pallas TPU lowering currently requires that the last two dimensions of your block shape are
divisible by 8 and 128 respectively, or be equal to the respective dimensions of the overall
array"`. The op was never reached. That is recorded as **UNRESOLVED, not UNSUPPORTED** — a probe
defect, not a hardware verdict. It does not change D's conclusion, which rests on the XLA arms, but
it is an open shard and it must not be cited as evidence that a Pallas top-k is impossible.

---

## 4. The VPU hypothesis — **confirmed**, and aimed at a term no option targets

`probes/step3_decider_RESULT.md`, shard 3, run to design doc §7.4's pre-stated bars.

The published intercept rests on a row with **550.56% spread** (`mxu/full/K=128`) and is not usable.
The verdict therefore rests on the **non-linearity**, a within-shard ratio that does not depend on
that row:

| epilogue | K=128 µs | K=256 µs | Δ 128→256 | K=512 µs | Δ 256→512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 7.342914614127949 (550.56% ⚠) | 7.218867722258437 (3.36%) | **−1.69** | 10.59588339558104 (1.62%) | +46.78 |
| sum (control) | 4.815605279873125 (2.96%) | 4.920359060633928 (3.17%) | **+2.18** | 7.688320692977868 (0.83%) | +56.25 |

**Doubling the matmul's FLOPs costs +2.18% of step time on the clean control arm.** The MXU is not
on the critical path at the real `head_dim = 128`. Against §7.4's bars — CONFIRMED near 8.4 µs,
REFUTED at ≲2 µs — the non-MXU part is 6.26 µs by intercept and ≥7.22 µs by invariance:
**CONFIRMED**. Probe fidelity is good: 7.343 µs/step against the kernel's 8.42, and §7.4's literal
arm 1 cross-checks to 0.75%.

Per the standing instruction, **no spill-reduction candidate was opened**, and none is opened here.

### 4.1 The finding that matters more than the verdict

| term | µs/inner step | share |
| --- | ---: | ---: |
| **32-head-sum bundle** | **3.858** | **52.5%** |
| relu + `w` + scale + masks | 2.401 | 32.7% |
| MXU | 1.084 | 14.8% |

The output store inside the first bundle is only ~0.35 µs, so **the head reduction itself is the
single largest thing in the step — and not one of options A, B, C, D or E touches it.** Every option
in `measurement_and_design.md` moves bytes, sizes buffers, or relocates selection. The dominant term
is arithmetic on a saturated unit sitting next to an idle one.

> **The three shares in this table are `modelled`, not `measured`, and the split is not stable.**
> Two caveats found later, both of which the table's layout hides:
>
> 1. The 3.858 µs is a two-point fit whose upper point is the `full` row above — the row §4 itself
>    declares unusable at **550.56% spread**. The same table shows the fit is non-linear. An
>    equally consistent reading of the surviving points gives 65.6% rather than 52.5%, so treat
>    52.5% as one end of a 52–66% bracket.
> 2. §8.2's NUM_HEADS ladder measured the bundle's cost as **essentially constant in `NUM_HEADS`**
>    and its absolute size as **arm-dependent by up to 2.7×**. So the bundle is not per-head
>    reduction arithmetic — it is store plus a fixed per-step cost — and a single share figure for
>    it does not transfer between arms.
>
> Both apply wherever this split is reused: §4.1, §4.3, §5d, §6 and §8.2. Neither changes the
> qualitative finding, which is what §4.1 is for — the head-sum bundle is the largest term in the
> step under every reading — but no candidate should be sized against 52.5% as if it were measured.

### 4.2 A correction to the decider's own reading of this

`probes/step3_decider_RESULT.md` concludes that §7.5's promotion of area 6's **A1 — drop the
ReLU** is "aimed at the smaller third", scoping A1 to the 2.401 µs ReLU chain. **That is a
misreading, and it understates A1 by roughly a factor of three.**

The head-sum bundle exists *because* the ReLU forces the head axis to be materialised before it can
be summed. Remove the ReLU and `Σ_h relu(q_h·kv)·w_h` becomes `(Σ_h w_h q_h)·kv` — one 128-dim
contraction instead of 32 dot products, a ReLU, a weight multiply and a 32-term reduction. Both
sources say so explicitly: design doc §7.5 (*"the head axis collapses into the contraction, the VPU
tail vanishes entirely and matmul FLOPs fall 32×"*) and `step2/sheets/area6-algorithm.md` row A1
(*"the one that dissolves everything"*). **A1 removes the first two rows of the table above — about
85% of the step.**

### 4.3 Two options opened by this, neither in `measurement_and_design.md`

| option | what | exact? | status |
| --- | --- | --- | --- |
| **A1 — Drop the ReLU** | `score = (Σ_h w_h q_h) · kv`. Removes ~85% of the inner step. | **No** — computes a different function | gated first, per the area-6 sheet's *"must be evaluated before any perf number"*. In flight. |
| **F — Head sum on the MXU** | Keep the ReLU; move the 32-term reduction to the idle MXU as a matmul. | **Yes** — pure reassociation, bounded at 2.46e-06 relative [verified `j-ed517191`] | in flight |

F **overturns a step-2 rejection**, and the reason is worth recording. Design doc §2.5 dismissed
E6/H, the 32-head sum as a block-diagonal ones-matmul, as *"3.1% MXU fill"*. That rejection was made
on MXU **efficiency**, and efficiency is the wrong criterion once the MXU is idle: a 3.1%-filled
matmul that deletes 3.858 µs of VPU work from the critical path is a win even though it wastes 96.9%
of the unit it touches. Whether it lands off the critical path — rather than being serialised with
the scoring matmul by Mosaic — is the open question, and it is a measurement rather than an argument.

A1's gate is expected to be hard. The closest measured precedent is unfavourable: area-6 row A6,
head subsampling to 16 of 32 heads, failed at `worst_rank=8177 / min_recall=0.49512`
[`measured (j-6d675a58-7ecd-4b44-9171)`] — and the existing negative control **is** that
approximation, since `relu(0·kv) = 0`. If A1 fails, area-6 row **A3** (score all positions with A1's
cheap form, shortlist `M ≫ k`, exact-rescore the shortlist) recovers exactness while still paying the
cheap form over the full `S`, which is where the time is.

---

## 5. `bkv_p = 8`, and the arm that needs no run

`probes/step3_decider_RESULT.md`, shard 1. `(8,4,4)` PASSes at `decode_64k`
(`worst_rank=2105 of allowed 2304, min_recall=0.99512`) and `mixed_prod` (`2098`, `0.99512`), with
negative controls firing **6/6** (`worst_rank=62644`, `min_recall=0.26953`) and **1024/1024**
(`8159`, `0.42578`).

Sweep, same chip same job. Baselines `decode_64k` 0.20202844752930105 ms (0.92% spread) and
`mixed_prod` 0.6724997947458178 ms (2.83%). **Negative is faster.**

| regime | tuple | subl | ms | spread % | Δ time % |
| --- | --- | ---: | ---: | ---: | ---: |
| `decode_64k` | (4,4,4) | 544 | 0.17841560184024274 | 0.47 | −11.69 |
| `decode_64k` | (8,4,4) | 576 | 0.17382489750161767 | 0.51 | −13.96 |
| `decode_64k` | (8,8,8) | 576 | 0.17355122254230082 | 0.65 | −14.10 |
| `mixed_prod` | (4,4,4) | 96 | 0.6789957056753336 | 1.16 | +0.97 |
| `mixed_prod` | (8,4,4) | 128 | 0.7057664450258017 | **491.23** ⚠ | +4.95 ⚠ |
| `mixed_prod` | (8,8,8) | 128 | 0.723214476602152 | 2.26 | +7.54 |

⚠ `mixed_prod` at `(8,4,4)` carries a **491% spread against a 4.95% effect** — a hundredfold. It is
not a usable point estimate and is bracketed `[+0.97, +7.54]` by its neighbours. It needs one re-run.

**The `(8,2,2)` at `decode_64k` arm I asked for needs no run**, and the reason generalises: at a
pure-decode regime slots 1 and 2 are read by no executing band and `num_sublanes_total` is already
saturated by slot 0 — `align_to(65,8) · 1024/128 = 576` for both tuples. Confirmed from the other
side: `(8,4,4)` and `(8,8,8)` give identical `subl=576` and `num_bkv=8` at `decode_64k`, and times
0.16% apart, inside both spreads.

Still open: **8 in the MIXED slot**, reachable only via `(8,8,8)`, never value-checked. Nobody should
ship it — `(8,8,8)` is the worst mixed arm measured, reproducing at +7.54% here against +7.08%
cross-job.

---

## 5b. Variant A — Narrow Record: **both halves now measured** — writer 3.3×, reader decode-only

**Read §5b-v2 below for the settled result.** This first part records the v1 job, whose reader did
not compile; it is kept because its incumbent arms and its writer number are both reproduced by v2
and the Mosaic diagnosis is a `references/` promotion.

`variants/narrow-record/`, committed `11efd4ef`. `measured (j-ce74a9f2-a4cd-40cb-9e87)`,
`JobStatus: Failed`, 2/4 shards `rc=0` — shards 2 and 3 red from a single cause, and shard 3
completed the writer probe before dying.

### The reader side did not compile

Mosaic rejected the scale-plane DMA at `streamindex_topk_narrow.py:245`:

```
Offsets along tiled dimensions must be aligned to tiles. Failed to verify that the index at
dimension 0 is divisible by the tile dimension 32.
  %766 = "tpu.memref_slice"(%arg10, %755, %21) : (memref<576x1024xi8,
  #tpu.tiled<(32,128)(4,1),[8,1]>, ...) -> memref<1x1024xi8, ...>
```

Mosaic tiles the **last two** dims, so a `u8[num_pages, 1024]` scale plane is tiled on its *leading*
axis and a one-row per-page slice is illegal. `tpu.assume_multiple` does not apply — the page index
genuinely is not a multiple of 32. 576 and 390 pages give the same error. **A's modelled 13.42% at
`decode_prod` and 8.36% at `decode_64k` remain `modelled`: neither confirmed nor refuted.** No
correctness margins were obtained and the MXU operand types are **not confirmed** — P1's native-fp8
finding stands for the incumbent only.

The incumbent arms in that job are clean and worth keeping: `decode_prod` 0.166819 ms (0.644%),
`decode_64k` 0.201013 (0.602%), `mixed_prod` 0.666106 (2.064%), `mixed_64k` 4.797339 (1.111%), with
every drift-control delta inside its own spread (+0.17 / −0.11 / +0.77 / +0.06%) and all four
tracking step 1's figures.

A retry is authorised: re-declare the scale plane as `u8[num_pages, page_size//128, 128]` so the page
axis is untiled and the `(8,128)` tile is exactly one page of scales. e8m0 is exact in both bf16 and
f32, so widening for the reader's regroup keeps A bit-exact.

### The writer side is `measured`, and it closes the last ranking defect — in the opposite direction

Pure XLA at the production cache size (`u8[415,256,4,256]`, 424,960 positions), index arithmetic
copied verbatim from `mla_attention_op.py:103-104`. **Negative is faster.**

| T | `W0_today_wide` ms | spread % | `W12_split_both` ms | spread % | Δ time % | speedup × |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.842139 | 0.59 | 0.247234 | 1.91 | **−70.64** | **3.406** |
| 64 | 0.059036 | 22.92 | 0.080732 | 35.70 | +36.75 | 0.731 |

The wide scatter moves bytes at 258 GB/s against the split writer's 527, paying the same 1.98×
amplification on the write side that the reader pays.

**The T=64 row cannot support a conclusion** and must not be read as a decode-side regression: every
arm sits within 0.01–0.04 ms of the donated-noop floor of 0.041–0.048 ms, with spreads comparable to
the effect.

**This closes design doc §9.4 item 2, and inverts it.** That item named the writer-side scatter as
one of four `unmodelled` terms deciding the ranking, and §9.1 placed A at rank 3 *specifically*
because of it — *"the row that decides whether A is net positive … is `unmodelled` and lives in a
file this exercise does not own"* — with the note that if the writer's owner were engaged, A moves to
rank 1 outright. The strike test's best case was the term being **zero**. It is not zero: it is a
second win of 70.64% on the T=1024 row, which is the prefill/mixed path, which carries 85.7–87.9% of
indexer time (§2). **A is rank-1 material the moment its reader compiles.**

### Carried regardless of whether A ships

Today's writer leaves **~70% of its prefill time on the table**. That turns
`measurement_and_design.md` §11 item 1 from a profile reading (~291 µs per indexer layer, off an
earlier prefill profile) into a measured **3.406×** with a named job. It is not a
`streamindex_topk` edit — it is `mla_attention_op.py:103-104` — and it is filed as a carried item in
its own right, not as a footnote to A.

**One coupling to state plainly, because the two numbers invite the wrong reading:** the split writer
is not independently shippable. It writes a format only a split reader can consume, so the 3.406×
cannot be banked while A's reader is blocked. What the writer probe establishes is that A's format
change is *cheap on both sides*, not that there is a free win sitting in the writer today.

---

## 5b-v2. Variant A, second build: the reader compiles, and it is a **decode** optimization

`j-c2e7edde-deee-4592-b81d`, `JobStatus = Succeeded`, 4/4 shards `rc=0`. Commit `6b14ae8a`. The
authorised relayout — scale plane as `u8[num_pages, page_size//128, 128]`, so the page axis is
untiled and the `(8,128)` tile is exactly one page of scales — lowered without complaint.

Negative is faster. `drift %` is the drift-control delta (§0.3): two arms of identical code, its
magnitude being the noise floor for an arm-to-arm comparison.

| regime | incumbent ms (sp%) | variant ms (sp%) | Δ time % | speedup × | drift % | conclusive? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `decode_prod` | 0.166699 (0.59) | 0.149275 (0.79) | **−10.45** | 1.117 | +0.00 | yes, 13× spread |
| `decode_64k` | 0.202102 (2.44) | 0.186629 (0.47) | **−7.66** | 1.083 | −1.07 | yes, 3.1× spread |
| `mixed_prod` | 0.675440 (2.51) | 0.665794 (1.61) | −1.43 | 1.014 | −0.07 | **no** |
| `mixed_64k` | 4.788465 (1.11) | 4.771312 (1.52) | −0.36 | 1.004 | +0.50 | **no** |

**The two decode rows are real; the two mixed rows are not conclusions.** In both, the spread equals
or dwarfs the effect. That is the finding that matters, because mixed carries 85.7–87.9% of indexer
time at the A/B shape (§2). Taking the mixed rows as zero — which is what their spreads license —
**A's reader is worth ≈0.93–1.10% of indexer time**; taking them at face value gives 1.2–1.4%. The
defensible number is the lower one. A's reader was always a decode optimization: in mixed, top-k
dominates the call.

> **Both figures must be built from the `_64k` rows, not the `_prod` rows.** The weights
> 12.1–14.3% decode / 85.7–87.9% mixed are derived entirely from the A/B shape
> (`max_model_len = 66560`), so the per-phase delta they multiply has to come from the same shape.
> That is `decode_64k`'s −7.66%, giving 0.121 × 7.66 = 0.93% to 0.143 × 7.66 = 1.10%. An earlier
> revision of this section multiplied `decode_prod`'s −10.45% — a `max_model_len = 9216` shape —
> by those same A/B-derived weights, and overstated A's contribution by ≈1.4× (to "1.3–1.5%" and
> "2.5–2.7%"). Mixing a delta measured at one deployment shape with a share derived at another is
> the specific error; the numbers above are the corrected ones.

**This demotes A's reader from the rank-1 expectation step 2 set for it, and the demotion is a
denominator effect, not a failure of the change.** The mechanism works exactly as designed — the
*scoring* half improves 16.58% and 13.66% on the decode rows while the top-k control is unchanged,
and VMEM went **down** (3.699 / 2.184 / 0.599 MiB against the incumbent's 4.195 / 4.168 / 1.095), not
up as the design feared. The change is simply aimed at a phase that is 12.1–14.3% of the time.

Modelled −13.42 / −8.36 / −0.42 / −0.47 against measured −10.45 / −7.66 / −1.43 / −0.36.
`decode_64k` lands within 0.7 points. `decode_prod` delivers 78% of its modelled prize and **the
shortfall is unexplained**; the regroup that replaced the incumbent's transpose is the suspect and is
`unmodelled`.

### The writer reproduces, and the 3-D form is free

| T | `W0` ms (sp%) | `W12b` ms (sp%) | Δ time % | speedup × |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.844617 (0.83) | 0.252724 (3.81) | **−70.08** | **3.342** |

v1 gave −70.64 / 3.406× on a separate job; the two agree. The **3-D form costs nothing over the 2-D
one** (0.252724 vs 0.251830, inside both spreads), so the reader's tiling requirement imposes no
writer penalty — that caveat, opened by the Mosaic rejection, is closed.

`T = 1024` is the prefill path, so **unlike the reader win, the writer win lands on the dominant
phase.** The `T = 64` row is not resolvable on this instrument — every arm within 0.02 ms of a
0.054–0.059 ms noop floor, spreads 16–50%, and the two builds disagreeing by more than either effect
(+36.75 vs +16.58). No number is quoted for it, and v1's +36.75 should not be either.

### Correctness — bit-exact, which is the only acceptable outcome here

`0 of 131072`, `0 of 12288`, `0 of 2097152` index elements differ; 2,240,512 compared. Oracle margins
**bit-identical to the incumbent's**: `decode_prod` 2109 / 0.99561, `decode_64k` 2105 / 0.99512,
`mixed_prod` 2098 / 0.99512. Negative controls fired on every row (8177 / 0.49512 on 64 of 64;
8159 / 0.42578 on 1024 of 1024).

A is bit-exact **by construction**, and the argument is worth stating because it is what makes the
bit-exact result the expected one rather than a lucky one: e8m0 is a bare 8-bit exponent, and both
bf16 and f32 have 8 exponent bits, so both widens are exact and the regroup moves bytes without
touching a value.

### MXU operand types — confirmed native fp8, from the lowering

This was load-bearing and unconfirmed: widening scales to f32 could have dragged the matmul operands
to bf16 and cost the 1.69–1.86× fp8 rate. It did not.

```
tpu.matmul : vector<4096x128xf8E4M3FN>, vector<2048x128xf8E4M3FN>,
             vector<4096x2048xf32> -> vector<4096x2048xf32>
```

Character-for-character P1's incumbent signature. 1064 `vmatmul` bundles, **bf16 = 0**, f8 = 1064 —
identical count and opcode mix to the incumbent. HBM operands `memref<576x256x4x128xi8>` and
`memref<576x8x128xi8>` confirm the split reached the compiler.

### What A is worth, and why it does not simply rank behind B

A and B **touch different code and different phases**. B is the indexer kernel's scores buffer on the
mixed band; A's reader is the indexer's scoring half on the decode band; A's writer is
`mla_attention_op.py`, which B does not touch at all. Their measured wins are therefore not competing
for the same time, and step 4 should compose them rather than choose. A's writer saving — **3.342×
at `T = 1024`** — is the largest ratio in the exercise after B's confound row.

> Quote the writer as a **ratio**, never as its absolute 0.592 ms difference. `writer_probe.py`
> measures a standalone store in isolation, and its own header says the absolutes drift between
> jobs while the ratio does not: v1 and v2 gave 3.406× and 3.342× (agreeing) off absolute pairs
> that do not. The 0.592 ms is not a per-call saving that can be added to an end-to-end budget.

`/critical-review` has since run on §5b-v2 and §5c (see §9). It corrected the reader's weighted
contribution — the figure above — and raised no objection to A's bit-exactness. A stays
`provisional` until the composed step-4 candidate carries a named correctness job.

---

## 5c. Variant B — Right-Size: **the largest measured win in the exercise**

`j-d9502595-54a2-45bc-947b`, `JobStatus = Succeeded`, 4/4 shards `rc=0`. Both arms in one process
through the read-only `instruments/microbench.run_regime`. Commits `9bc5f5cd` (variant), `36aedcdc`
(the sixth regime, as an instrument addition), `a8197f38` (`RESULTS.md`).

**What B changes.** The incumbent sizes its scores buffer from `max_model_len`; B sizes it from a
static `kv_len_bucket` and deletes the `-1` fill that went with it. Nothing else.

Negative `Δ time %` is faster. `cols` is the scores-buffer width in columns, incumbent → B.

| regime | `cols` inc → B | incumbent ms (sp%) | B ms (sp%) | Δ time % | speedup × |
| --- | --- | ---: | ---: | ---: | ---: |
| `decode_prod` | 10240 → 8192 | 0.16805 (0.75) | 0.16747 (0.69) | −0.35 | 1.003 |
| `decode_64k` | 67584 → 65536 | 0.20305 (0.53) | 0.20014 (0.50) | −1.43 | 1.015 |
| `mixed_prod` | 10240 → 8192 | 0.67118 (2.53) | 0.65050 (1.79) | −3.08 | 1.032 |
| `mixed_64k` | 67584 → 65536 | 4.78598 (1.15) | 3.74624 (1.12) | −21.72 | 1.278 |
| **`mixed_confound`** | 67584 → 8192 | 2.27933 (1.85) | 0.61196 (0.59) | **−73.15** | **3.725** |

`decode_prod` and `mixed_prod` **cannot support a conclusion** — the effect is at or below the spread
of the arms it is a difference of. `decode_64k` is weak (2–3× spread). Only `mixed_64k` and the
confound are decisive, and both are on the band that carries 85.7–87.9% of indexer time (§2).

### Why the confound is a legitimate regime and not a shape picked to flatter B

All five step-1 regimes run `kv_len` within one page of `max_model_len` by construction, so the
incumbent's buffer is already within 3–8% of the live width in every one of them and **B's entire
prize has no room to be measured at any of them**. That is why §4.3(b) quoted `≤ −5.40%`: it had no
shape, and normalised against `mixed_64k`'s call as a deliberate upper-bound denominator. Design
§12 item 4 sanctions exactly one sixth regime for this. Verified before folding this in:
`instruments/` is untouched by both B commits and `shapes.py` has not changed since step 1 created
it (`b3d2a0ec`), so the five-regime yardstick did not move under the variant that is measured
against it.

The confound is `max_model_len = 66560` with an 8k live context — a 64k-capable deployment serving a
short batch. §8.1 records that as the *common* case, not the corner: mean live fraction ≈ 50%, and
the first sixteen chunks of a 64k prompt below 25%.

### The mechanism, which is not the one the design assumed

The design ranked B on **bytes**. 1554.5 of the confound's 1667.4 µs delta lands in
`sparsecore_topk`, against a `modelled` 0.7–0.9 ms for the whole term: narrowing `n` flips
`_pick_partition` from **P = 2 to P = 1** and collapses the two-stage top-k into one stage. The byte
saving is the small half — `mixed_64k` moves 3% fewer bytes and is 21.72% faster. B is a
**threshold** effect at `MAX_SLICE_WORDS = 65536`, not a proportional one.

This is what makes B robust rather than fragile, and it is worth being explicit because the
threshold reads at first like the caveat that kills it. At the A/B, `max_model_len = 66560` puts the
incumbent's width at 67584 — *just* above the threshold, so the incumbent pays two stages on **every
prefill chunk regardless of how short the live context is**. Every bucket B could plausibly ship
lands at or under 65536 and therefore keeps P = 1.

### What B is worth at the A/B shape — `modelled`, from two measured points

Both incumbent points sit in one partition regime (P = 2, 528 sublanes) and both B points in the
other (P = 1), so a linear fit within each is the right form. Fitting time against live width `w`
in sublanes:

| arm | fit (ms) | from |
| --- | --- | --- |
| incumbent | `1.92125 + 0.005595·w` | `mixed_confound` 2.27933 @ w=64, `mixed_64k` 4.78598 @ w=512 |
| B | `0.16422 + 0.006996·w` | `mixed_confound` 0.61196 @ w=64, `mixed_64k` 3.74624 @ w=512 |

The incumbent's 1.92 ms intercept **is** the `max_model_len` tax; B's is 0.16 ms. A 65536-token
prefill in 1024-token chunks has a mean live context of ≈ 33280 tokens, w ≈ 260:

`mean w` is the mean over the 64 chunks of the **bucketed** static width in sublanes (tokens / 128)
that B would allocate; the incumbent's width does not bucket, so its column is constant. Negative is
faster.

| bucketing of the shipped static width | mean w | B ms | incumbent ms | Δ time % |
| --- | ---: | ---: | ---: | ---: |
| exact `kv_len` (what was measured — a **ceiling**, not shippable) | 260 | 1.983 | 3.376 | **−41.3** |
| 8192-token granularity (8 buckets) | 288 | 2.179 | 3.376 | −35.5 |
| powers of two (7 buckets) | 341 | 2.550 | 3.376 | −24.5 |

Each `mean w` is derived rather than read off, so the derivation is here to be checked. Chunk *i*
of 64 has `kv_len = 1024·i`.
*Exact:* mean `kv_len` = 1024·(1+…+64)/64 = 33280 tokens → w = 260.
*8192-granularity:* eight chunks map to each of 8192…65536, mean = 8192·(1+…+8)/8 = 36864 → w = 288.
*Powers of two:* buckets 1024, 2048, 4096, 8192, 16384, 32768, 65536 take 1, 1, 2, 4, 8, 16 and 32
chunks respectively; mean = 2796544/64 = 43696 → w = 341.4.

> An earlier revision of this table used 292 and ~346, which no derivation reproduces. The
> corrected values move both rows **in B's favour** by under 1.2 points, so nothing downstream
> changes sign; the row that matters is superseded outright by §8.3 in any case.

> **Superseded — see §8.3.** A six-rung ladder measured both arms directly
> (`j-59bbfd2f-e408-4a75-a155`). The three figures above are **conservative**: the measured
> equivalents are **−46.2 / −40.9 / −30.2**. The direction of the error is the safe one, so nothing
> downstream is invalidated, but quote §8.3's numbers and not these. The suspicion recorded below
> about B's slope was correct and **mis-attributed** — B is the well-behaved arm.

**B survives bucketing** — coarsening the ladder costs only the byte term, because every bucket
≤ 65536 keeps the P = 1 collapse, which is the large half. That inverts B's most serious caveat: its
own `RESULTS.md` correctly flags the measured prize as a ceiling because the driver passes the exact
`kv_len`, and the arithmetic above says the floor under that ceiling is still a 23% prefill win.

Two honest limits on this table. The linearity is **untested** — two points cannot test the form
they were fitted with, and B's fitted slope coming out *steeper* than the incumbent's (0.006996 vs
0.005595) is unexplained and mildly suspicious. And the bucket count itself is `assumed`: the
recompile cost of the extra jit cache entries and the plumbing that computes the bucket at the call
site are both unmeasured (`RESULTS.md` caveat 9).

### Correctness — the strongest evidence in the exercise

B's index **sets** are identical to the incumbent's on every row at all six regimes (`set_diff = 0`),
reproducing `worst_rank`/`min_recall` to the digit. The risk B carries is a *reachable unwritten
lane*, since it deletes the `-1` fill — and the test was designed for that risk specifically. The
**poison arm** reinstates the aliased prefill as `+1e30` instead of `-inf`, so an unwritten lane
would **seize** the top-k rather than perturb it, and it comes back bit-identical to the clean arm at
all six regimes. All five negative controls FAILED as required (ranks 7968–62644 against 2304).

> ### ⚠️ CRITICAL — the poison arm could not have failed, and it does not cover what B ships
>
> `/critical-review` found this evidence hollow on the one risk it was built for.
> `variants/right-size/run_correctness.py:121-122` binds `kv_len_bucket = r.kv_len` in **every**
> arm. With bucket == `kv_len` the written width exactly covers the allocated buffer, so at all
> eight shapes ever run there are **zero unwritten lanes** and the `+1e30` poison is 100%
> overwritten before selection reads it. "Bit-identical to the clean arm" was guaranteed by the
> configuration, not earned by the kernel. The paragraph above claimed this was "decisive"; it is
> not, and that word is withdrawn.
>
> The configuration where unwritten lanes exist is `bucket > kv_len` — which is **what B ships
> under any bucket ladder**, and therefore under every row of the bucketing table above. It has
> only ever run perf-only, in `probes/followup1/bucket_ladder.py:249` with `measure_topk=False`,
> no oracle and no index-set comparison. The two rungs that do run it (kv 24576 → bucket 32768,
> 64 unwritten sublanes; kv 49152 → bucket 65536, 128 unwritten) are **exactly the two rungs the
> −30.2% pow2 figure in §8.3 and §6 rests on**.
>
> What the existing runs *do* establish stands: bit-identical index sets against the incumbent at
> six regimes, the masked-store tail at `kv_len = 8000`, and `run_sc_coverage.py`'s 12/12. Those
> cover B's arithmetic. They do not cover the deleted fill.
>
> **Disposition: the gate ran, and B passed it.** See §5c-gate immediately below. B is correct in
> the configuration it ships in; the withdrawal above applies to the *evidence*, which is now
> replaced, not to the kernel.

The masked partial-block store lowered first time, but note it needs a *static* destination offset,
and it is **not emitted at any of the six perf regimes** — all six have `kv_len` a multiple of
`bkv_sz`. Two correctness-only shapes at `kv_len = 8000` exist for it.

**`sparsecore_topk` had no test at all** (`measurement_and_design.md` §11 item 4, a precondition for
both B and C). It does now: `run_sc_coverage.py` drives it against a numpy oracle at all 12
`(b, n, live, ragged, P)` combinations the two arms produce, padded with `+1e30` **and not `-inf`**,
so it passes only if the `row_lengths` bound actually holds — which an `-inf`-padded test could not
tell you. All 12 PASS. That closes a carried item, independently of whether B ships.

### 5c-gate. B at `bucket > kv_len` — **PASS**, with lanes that were genuinely unwritten

`measured (j-6454262e-acd2-4371-b3ba)`, `JobStatus = Succeeded`, 4/4 shards `rc=0`, 2m25s, on the
existing image — no rebuild. `probes/bucket_correctness/`. HEAD `d81670d2`; `instruments/` verified
byte-identical on the pod (`shapes.py` md5 `8a9ac175495e73cf7b812cf6c6bda614`).

The unwritten count is **measured, not predicted**: a shim in the variant module's namespace counts
how many 128-column sublanes of the *actual device buffer* still hold the value it was pre-filled
with. The analytic prediction agreed at all 14 regime runs, so the parenthesised numbers below are
a cross-check rather than the source.

| shape | bucket | alloc | written | **unwritten** | poison vs clean | oracle `worst_rank` / `min_recall` | set diff vs incumbent |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| kv 24576 decode | 32768 | 256 | 192 | **64** | identical, PASS | 2082 / 0.99561 | 0 of 6 |
| kv 24576 mixed | 32768 | 256 | 192 | **64** | identical, PASS | 2124 / 0.99316 | 0 of 1024 |
| kv 8000 decode | 16384 | 128 | 64 | **64** | identical, PASS | 2073 / 0.99609 | 0 of 6 |
| kv 8000 mixed | 16384 | 128 | 64 | **64** | identical, PASS | 2101 / 0.99414 | 0 of 1024 |
| kv 49152 decode | 65536 | 512 | 384 | **128** | identical, PASS | 2111 / 0.99463 | 0 of 6 |
| kv 49152 mixed | 65536 | 512 | 384 | **128** | identical, PASS | 2158 / 0.99219 | 0 of 1024 |
| kv 8000 decode | 9216 † | 72 | 64 | **8** | identical, PASS | 2073 / 0.99609 | 0 of 6 |
| kv 8000 mixed | 9216 † | 72 | 64 | **8** | identical, PASS | 2101 / 0.99414 | 0 of 1024 |
| *control:* kv 24576 / 8000 / 49152, both bands | = `kv_len` | | | **0** | identical, PASS | as §5c | 0 |

Allowed rank 2304, required recall 0.99, at every row. † bucket 9216 is deliberately not a multiple
of `bkv_sz = 2048`, so B's masked partial-block store is *compiled in* (`n_tail_sublanes = 8`) while
unwritten lanes exist.

**Two negative controls, and the second is the one this job exists for.** `neg_control` (half the
query heads zeroed) went RED at all 14 runs — `bc_a_decode` gives `worst_rank = 23786` against 2304,
`min_recall = 0.41797`, 6 of 6 rows differing. `neg_control_unwritten` runs **the same poisoned
buffer through the same oracle with only the reachability gate removed** — the shim drops
`row_lengths` so selection may reach the unwritten lanes — and it went RED at every shape that has
them, with `index out of range` on row 0 and a `max_index_returned` that names the poison exactly:
26623 at bucket 32768 (= 24576 + 2048), 10239 at 16384, 51199 at 65536, 9215 at 9216. That is what
makes the PASS non-vacuous: the identical buffer, with one gate removed, is caught immediately.

The script also asserts `poison_cells_below_kv_len == 0` at every shape — measured 0 everywhere, so
there are no holes *inside* the region selection can reach.

**The review's claim is now confirmed empirically as well as by reading.** At `bucket == kv_len` the
measured unwritten count is **0** at all six control runs. The original test could not have failed.

Four refinements the gate produced, all of which correct something stated above or in the dispatch:

1. **The masked partial-block store and unwritten lanes are structurally mutually exclusive**, so
   "the tail store fires on both bands under a bucket ladder" — which §5c implies and the dispatch
   asserted — is false. `num_bkv = min(cdiv(kv_len, bkv_sz), max_bkv)`, so the tail *executes* only
   when it lands on the last slab, which is exactly the case where written == allocated. Whenever
   unwritten lanes exist the tail branch is dead code. Shape `bc_d` (bucket 9216) is as close as one
   shape gets: tail emitted, tail not executed, 8 unwritten lanes. The masked store is therefore
   exercised only by the `bucket == kv_len` arms — which did run it, at kv 8000, on both bands, PASS.
   This is good news for B, not bad: the two risks cannot compound.
2. **kv 8000 / bucket 16384 is 64 written and 64 unwritten**, not the 63/65 the dispatch predicted.
   At 16384 the buffer is a whole number of `bkv` slabs, so the fourth slab stores its full 16
   sublanes. The 63-sublane geometry exists only at `bucket == 8000`.
3. **A pow2 ladder would not generate that row at all** — `pow2_bucket(8000) = 8192` gives 0
   unwritten. Reaching it needed an explicit override to a coarser ladder (`BUCKET_OVERRIDE` in the
   script). It is the worst case in the set: bucket = 2.05× live, half the buffer unwritten.
4. **The zero-margin case passes.** kv 24576 / bucket 32768 decode has every row's `visible` exactly
   `kv_len`, last written position 24575, selection stopping one position before the first unwritten
   lane. That is the tightest configuration the design admits.

**Consequence: B carries a named correctness job covering the configuration it ships in.** What is
still not covered is unchanged from §5c — the SparseCore source half, end-to-end accuracy, and
bucket widths other than those measured. No timing was taken here.

### Two corrections B makes to the design document

1. **§4.3(b)'s `≤ −5.40%` bound is wrong by 13.5×**, in B's favour, and it was wrong by
   construction — it used `mixed_64k` as the denominator because no confound regime existed.
2. **The design's "TC subtotal 257.0 µs" double-counts the SparseCore read** — its 486.53 MB line is
   labelled "TC write *+ SC read*". The genuine TC term is the deleted fill alone: 93.2 µs
   `modelled` against a 112.9 µs measured residual. The TC roofline was fine; the subtotal was not.

Consequence for §4.6, which deferred B's SparseCore half: **the prize attributed to that half is
already collected, and exceeded, by the TC half.** Its expected value needs re-deriving before
anyone spends on it — this is a reason not to build it, not a reason to build it sooner.

**Not covered:** B's SparseCore source half (not attempted, `sparsecore_topk.py` unmodified);
end-to-end accuracy; bucket boundaries other than the six widths measured. The `mixed_prod` row of
the `sparsecore_topk` component table is unusable (885.74% incumbent spread against a 6.41% effect)
and is reported, not ranked.

---

## 5d. Option A1 — Drop the ReLU: **gate failed, no kernel written**

`j-5acf2993-b56b-4a38-9abe`, `JobStatus = Succeeded`, 4/4 shards `rc=0`, 94 s. Commit `35ea3bcc`.

The area-6 sheet required A1 to be evaluated for accuracy **before** any perf number, because it
computes a different function rather than a perturbation of the same one. It was, and it failed.

| regime | S | rows bad | `worst_rank` (allowed 2304) | over by | `min_recall` (req 0.99) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 8192 | 64 of 64 | 7873 | 3.42× | 0.69189 |
| `mixed_prod` | 8192 | 1024 of 1024 | 7777 | 3.37× | 0.67578 |
| `mixed_32k` | 32768 | 1024 of 1024 | 30832 | 13.4× | 0.48047 |
| `mixed_64k` | 65536 | 1024 of 1024 | 57391 | 24.9× | 0.39844 |
| `decode_64k` | 65536 | 6 of 6 | 40939 | 17.8× | 0.50293 |

Not one row of 3142 passed, and it degrades monotonically with S — mean recall 0.737 → 0.582 →
0.520 — so it is worst exactly where the A/B lives. Roughly half the returned set is wrong at long
context. An oracle self-reference arm was perfect at every regime (`worst_rank = 2048`,
`min_recall = 1.00000`), so the plumbing is sound and this is the function, not the harness.

**The gate ran against the *idealised* A1** — exact float64 arithmetic, exact top-k, visibility mask
lifted off the float64 ReLU oracle. That is what makes this cheap and final: the failure rules out
**every** A1 kernel, at any precision, so no kernel had to be written to reject it. It reuses
`instruments/test_correctness.py`'s `oracle_scores` and `check` unmodified.

**The dataflow claim A1 rested on is true; the ranking claim is not.** The head-axis collapse
`Σ_h w_h(q_h·kv) == (Σ_h w_h q_h)·kv` verified to 6.5e-16 relative. A1 does not die on the algebra —
it dies because the collapsed form is a poor *ranker*, Pearson 0.679 against the true score. That
distinction matters for **F**, which uses the same reassociation but keeps the ReLU and is therefore
exact; nothing here touches F.

**A3, the designated fallback: do not build it.** The sweep found the shortlist width `M` that
passes needs `M/S = 0.5` at both 32k and 64k. That is not a shortlist — it is a selection problem of
the same order as the one it was meant to cheapen, and A3 would run selection twice. The surviving
fraction is also quoted in MACs, which shard 3 established is the wrong currency: the MXU is 14.8%
of the step and doubling its FLOPs costs +2.18%, so the prize was the 85% VPU tail, which A3 merely
scales by `M/S` while adding an unpriced top-`M`-of-`S` pass at a width where a single
compare-and-count pass alone measures 232.31 µs `measured (j-b54e9716-3db5-4f1f-b8d4)`. This one is
judgement, not measurement, and it is falsifiable: a top-`M`-of-`S` priced far below 232 µs
overturns it.

### The consequence that outranks the verdict

**The 85% of the MIXED inner step now has no candidate.** A1 was the only proposal that removed both
the relu chain (32.7%) and the head-sum bundle (52.5%, `modelled` — see §4.1's caveat; the bracket is 52–66% and arm-dependent). The exact routes left are E1, E2 and F, and
only F touches the head sum. §7.5 item 1 resolves to **not inserted**.

Two things A1 hands forward, both cheap and both now scheduled:

1. **`a1_gate.py` is reusable and costs one shard (94 s).** It gates any rescoring against the
   float64 ReLU oracle on all five regimes, with a control and a self-check, without a kernel.
2. **Row A2 — clamp `w ← max(w, 0)` — has never been gated**, and it is a one-line change to the
   surrogate that makes **E1's precondition true by construction**. E1 is the measured-but-
   disqualified 13.70% in §11. If A2 passes, E1 stops being disqualified.

---

## 5e. Option C — Half-Width Key: **do not build.** The accuracy objection dissolves; the perf case does not

`j-ec4de538-453c-4956-b523`, `JobStatus = Succeeded`. Commit `19af7bc5`. Negative is faster.
`bf16out` is C's TensorCore half only; `cast` isolates the widen C would *not* pay, and is the
instrument that prices C's ceiling.

| regime | scores MB | incumbent ms (sp%) | `bf16out` ms (sp%) | Δ time % | speedup × |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 2.6 | 0.16760 (0.43) | 0.17134 (0.61) | +2.23 | 0.978 |
| `decode_64k` | 1.6 | 0.20059 (0.76) | 0.20242 (0.70) | +0.91 | 0.991 |
| `mixed_prod` | 41.9 | 0.67109 (1.93) | 0.70419 (2.44) | +4.93 | 0.953 |
| `mixed_64k` | 276.8 | 4.78892 (1.07) | 4.84624 (1.18) | +1.20 | 0.988 |

**Every arm is slower.** §7.5 predicted exactly this: under the confirmed VPU hypothesis a 16-bit key
adds a VPU-side encode on the unit that is already the bottleneck, so C spends time on the critical
path to save bytes off it.

> **This job carried no drift control**, so the four deltas are read against within-arm `spread %`
> — the yardstick §1.2 establishes is too *lenient* for an arm-to-arm delta. The two rows that
> would need a drift bound to be conclusive are `decode_64k` (+0.91% against 0.70–0.76% spread)
> and `mixed_64k` (+1.20% against ~1.1%). It does not change C's disposition: the verdict rests
> on the **ceiling** — C's best case is ≤ 2.53% and it is slower on every arm including the two
> that clear their spread by 2–4×. A drift control could move the two marginal rows toward zero;
> it cannot make a ≤ 2.53% prize worth building.

**C's ceiling, priced against a measured rate.** The `mixed_64k` cast arm is genuinely
bandwidth-bound — 415.2 MB / 0.16817 ms = **2.469 TB/s measured**. Dividing the bytes C removes by
that rate gives a best case of **0.63% / 0.33% / 2.53% / 2.34%** across the four regimes, and that
ceiling charges nothing for the pack/unpack C would have to add. An independent decomposition at
`mixed_64k` gives 2.3% ± 1.6%, agreeing with the ceiling rather than sharpening it. The decode cast
arms are *not* bandwidth-bound (0.07–0.10 TB/s, a fixed ≈36 µs dispatch floor — hence their 11–18%
spreads), so the decomposition is correctly reported only at `mixed_64k`.

So C's entire prize is ≤ 2.5% and its measured cost is +1.2 to +4.9%. There is no version of the
bucket arithmetic that rescues that.

### The accuracy objection was wrong, and the finding outlives the option

Step 2 held that a 16-bit key would cost rank quality. Measured (`j-7f77a816-5968-4ebd-80c3`), the
choice of 16-bit format decides it entirely, and **fp16 is not bf16**:

| format | `decode_64k` worst_rank / min_recall | mean index-set movement vs f32 (of k=2048) |
| --- | --- | --- |
| f32 (incumbent) | 2105 / 0.99512 | — |
| bf16 | **2170 / 0.99170** | 9.2 |
| **f16p** | **2105 / 0.99512** | **1.0 (0.049%), max 3** |
| m4 (negative control) | **FAIL** 2577 / 0.94482 | 53.7 |

bf16 passes but eats **two thirds of the recall margin** at `decode_64k` (0.00512 → 0.00170 above the
0.99 floor). fp16 with its 10 mantissa bits is indistinguishable from f32 everywhere. The control
fired on every arm. Range was checked rather than assumed — `max|s| = 209.91`, zero overflow, zero
subnormal — though that is a property of the instrument's synthetic inputs, not of production
weights, which is why the recommendation stays the precision-only key with the f32 exponent retained.

**This is a `references/` promotion, not a C result:** for a key that feeds a top-k, 16 bits is fine
and the mantissa is what matters. Any future option narrowing this key should use fp16 and should
not re-litigate the accuracy question.

### Three things C hands forward

1. **A second, independent `sparsecore_topk` test** (`j-9d6276c2-8eaa-4dc6-9bef`, 28 pass / 0 fail,
   `p ∈ [1,2,4,8,32]`, five corruptions all caught). B built one too (§5c). The carried precondition
   is now covered **twice over by two independent implementations that agree** — worth landing
   regardless of C, and worth consolidating rather than shipping both.
2. **Two invalidated jobs whose causes are platform facts**, both now guarded by loud controls and
   both `references/` material: the jaxpr memo in `jax._src.linear_util` **survives
   `f.clear_cache()`** and needs `jax.clear_caches()`; and **`--xla_allow_excess_precision` defaults
   true**, which silently deletes a bf16 round trip — narrowing has to be integer bit-masking. Any
   precision experiment on this stack that does neither is measuring nothing, quietly.
3. **An independent confirmation of the VPU hypothesis.** The scores buffer is at most ≈4.8% of the
   `mixed_64k` kernel by the most generous accounting, so 95% of the mixed band is not scores
   traffic. C's agent did not run the strike test and correctly declined to claim the inference.

### A cross-job check on the incumbent, which nobody set out to run

Three variant jobs measured the untouched incumbent at `mixed_64k` in three separate sessions:
**4.78415** (E, `j-ce9a19a4`), **4.78598** (B, `j-d9502595`), **4.78892** (C, `j-ec4de538`). Spread
across jobs is **0.10%**, well inside every arm's within-job spread. The baseline is stable across
jobs, which is what licenses comparing a delta measured in one job against a delta measured in
another — an assumption this document has been making since §1 without evidence. It now has some.

---

## 5f. Option F — head sum on the idle MXU: **rejected, and the premise failed in a dimension nobody modelled**

`j-ace83e22-c472-4463-b41d`, `JobStatus = Succeeded`, 4/4 shards `rc=0`. `variants/mxu-head-sum/`.

F was the option this document opened in §4.3 after the VPU hypothesis was confirmed: move the
3.858 µs head-sum bundle (52.5% of the inner step, `modelled` — §4.1's caveat) onto an MXU measured to be idle, as a
block-diagonal ones-matmul that keeps the ReLU and is therefore **exact by reassociation**, bounded
at 2.46e-06 relative. It also overturned §2.5's rejection of E6/H, on the grounds that "3.1% MXU
fill" is the wrong criterion when the unit is idle.

Negative is faster. `drift Δ %` is the drift control (§0.3).

| regime | incumbent ms | variant ms | Δ time % | drift Δ % | readable |
| --- | ---: | ---: | ---: | ---: | --- |
| `decode_prod` | 0.16693462 | 0.16812285 | +0.71 | −0.41 | yes |
| `decode_64k` | 0.20087782 | 0.20724762 | +3.17 | +0.14 | yes |
| `mixed_prod` | 0.67006801 | 0.71251987 | +6.34 | −0.33 | yes |
| `mixed_64k` | 4.79921508 | 5.15369731 | **+7.39** | −0.19 | yes |

Slower at every regime, worst on the dominant band, and every row is outside its drift control.

### It is also not exact, which was its entire justification

Four of five correctness margins are **not bit-identical** to the incumbent: 2109/0.99512,
2098/0.99512, 2105/0.99365, 2146/0.99268, 2170/0.99072. All ten runs PASS and every negative control
fired, so F is not *incorrect* — but a reassociation bounded at 2.46e-06 must be bit-identical, and
these are not. The cause was found rather than tolerated: **the head-sum matmul is emitted bf16**,
confirmed three ways — 1024 `bf16` vmatmul bundles in the lowering where the incumbent has zero, an
`f32 DEFAULT` arm whose numerics are *identical* to the bf16 arm (2.457e-03), and an `f32 HIGHEST`
arm that is the only one inside the 2.46e-06 bound and costs +244.35%.

So the exact version of F exists and is 3.4× slower than the incumbent. There is no configuration in
which F is both exact and fast.

### The mechanism, which is the durable result here

| impl | vmatmul | bf16 | f8 | bf16 weight pushes |
| --- | ---: | ---: | ---: | ---: |
| incumbent | 1024 | 0 | 1024 | 0 (16 `vmatpush3.f8e4m3` per MXU) |
| F | 2048 | 1024 | 1024 | 1024 `vmatpush1.bf16` per MXU, ×2 |

**2.0 weight pushes per matmul, against 0.031 for the scoring matmul — a 64× worse ratio.** The
reduction matmul is **push-bound, not MAC-bound**. Scoring pushes one weight tile and streams many
activation tiles through it; F's ones-matrix has to be pushed for essentially every matmul it feeds,
so it occupies the MXU's issue path while barely touching its MACs.

The K-ladder confirms both halves of this: it stayed **flat** under F (−0.48%, inside drift), so the
MAC pipeline really was idle and **stayed** idle. §4's finding that the MXU has spare MACs was
correct. The inference drawn from it — that spare MACs mean spare MXU — was not. That is a
`references/` promotion and it generalises well beyond this kernel: **MXU headroom measured in MACs
is not MXU capacity; for a reduction expressed as a matmul, check pushes-per-matmul.**

Step-level accounting: incumbent 7.276 → 8.968 µs, +23.26% against a target of ~3.5–4. Net +1.692 µs
accounted as ones-matrix construction 0.529 µs (avoidable — a reference arm with it hoisted is still
+16.00%), bf16 issue/push serialisation, and an `f32→bf16` cast of the whole score tensor landing
back **on the VPU**, against ≤3.5 µs of VPU rotate-tree removed. Honest gap: both `CONTROL` arms came
back at 10629% and 116% spread and are unusable, so the prize term is `modelled` from shard 3 rather
than measured in this job.

Two formulations were ruled out analytically without a probe, correctly: folding the head axis into
the existing einsum (the ReLU sits *between* the contractions, so the head axis must be materialised
— which is exactly what makes F exact and A1 an approximation), and a bf16×2 split-float (≈4 VPU
ops/vreg to split, against the 2.25 ops/vreg it would replace).

### One carried item, and it is not F

`W_BD` — fold `w` **into** the block-diagonal matrix — measures **14.38% faster (bf16) and 27.17%
faster (fp8) at the step level**, and it wins *despite* the MXU reduction rather than because of it:
it deletes the VPU `w`-multiply, worth −2.210 µs against the reference arm. It needs no sign
guarantee on `w`, so it survives where E1 needs A2's gate.

It is **not a result yet**, and the agent was right not to present it as one. Two gates: `W_BD`'s
construction cost is `unmodelled` and is producer-side, and **neither arm's numerics pass the §7
gate** — F3's top-1% relative error is 2.188, which is the classic hollow-win shape. Route: it is a
producer-side change, and it goes to that owner with both gates named, not into step 4's composition.

---

## 6. Ranking as it stands

**Every option is now resolved.** The four-shard follow-up probe (A2's gate, the `NUM_HEADS` ladder,
B's bucket ladder, the `(8,4,4)` re-run) has since returned — see §8 — and it sharpened the two
survivors rather than adding a candidate.

> **This ranking is unchanged by `/critical-review`, but two of its cells are.** B's original
> correctness claim was withdrawn — §5c's poison arm could not have failed — and **replaced** by
> `j-6454262e-acd2-4371-b3ba`, which passed at `bucket > kv_len` (§5c-gate). A's contribution is
> 0.93–1.10%, not the 1.3–1.5% an earlier revision gave it. Neither reorders anything: B's win is
> 20–40× A's on the band that carries the time, and no third candidate is close to either.

**A and B are not competing.** They land on different phases and, for the writer, different files.
The ranking below orders them by size of measured win, but step 4 composes rather than chooses.

| rank | option | standing | why it moved from step 2 |
| ---: | --- | --- | --- |
| **1** | **B — Right-Size** | **measured, correct, shippable** | 21.72% faster at `mixed_64k` and 3.725× at the confound (§5c), on the band carrying 85.7–87.9% of indexer time. Its `≤ −5.40%` design bound was 13.5× low **by construction**. Survives bucketing (≥ 23% `modelled`) because the prize is the P=2→P=1 threshold, not bytes. Correctness now covers the shipping configuration — §5c's evidence was hollow, and `j-6454262e-acd2-4371-b3ba` replaced it (§5c-gate). |
| **2** | **A — Narrow Record** | **both halves measured, correct, bit-exact** | writer 3.342× at `T=1024` on the dominant phase; reader 10.45% / 7.66% faster but **decode-only**, worth **0.93–1.10%** of indexer time once weighted (§5b-v2). Design doc §9.1's "A moves to rank 1 outright" was earned on the writer term and then **undone by the phase weighting** — the reader lands on 12.1–14.3% of the time. MXU operands confirmed native fp8. |
| **3** | **E — Retune** | **measured** | fell from joint rank 1. Net at the A/B is −0.84% to +0.90%, sign unresolved (§1.3). |
| **4** | **C — Half-Width Key** | **dead** | measured **slower at all four regimes** (+0.91 to +4.93%) against a ceiling of ≤ 2.53% (§5e). §7.5's prediction held exactly. Its accuracy objection dissolved — fp16 is free — but the perf case never existed. |
| **5** | **D — No Round Trip** | **dead** | priced at 8.96× its threshold (§3). Not built. |
| **6** | **A1 — Drop the ReLU** | **dead** | gate failed on all five regimes, worst at long context (§5d). Ruled out at the idealised limit, so no kernel exists that rescues it. A3 fallback also rejected. Not in step 2. |
| **7** | **F — MXU head sum** | **dead** | slower at all four regimes (+0.71 to +7.39%), outside drift, **and not exact** — the reduction lowers to bf16, and the exact `f32 HIGHEST` form costs +244% (§5f). The MXU's MACs were idle and stayed idle; its *issue path* was not. Not in step 2. |

**All four of step 2's `unmodelled` ranking defects are now closed** — item 1 (TC-side top-k, §3),
item 2 (writer-side scatter, §5b), item 3 (the mixed residual, §4) and item 4 (the decode:mixed
split, §2). Item 4 is `modelled` rather than `measured`, but it is settled well past the precision
the ranking needs; an instrumented dispatch to make it `measured` is an open shard.

Three of the four moved the ranking, and **two moved it against the direction step 2 guessed**: the
writer-side term was assumed to be a cost and is a win, and the TensorCore top-k was the one term
that could have promoted D and instead buried it. The fourth — the VPU residual — confirmed, and in
confirming it pointed at a term (§4.1) that none of the five options addresses.

That is worth stating as a result about the *process* rather than about the kernel: every option in
`measurement_and_design.md` was ranked on bytes, buffers or boundaries, and the measurement says the
dominant cost is arithmetic on a saturated unit.

### Two survivors of seven, and one target with nothing left aimed at it

**Five of the seven options are dead, and all five died on measurement rather than on argument** —
three of them (D, A1, and F's exact form) without a shippable kernel ever being written, which is
most of what step 3 was for. What survives is **B**, whose prize the design bounded 13.5× too low,
and **A**, whose prize the design put in the wrong half of the change.

The pattern across all seven is one thing: **every option that tried to move bytes lost, and the two
that won moved neither.** B deletes a partition stage; A's writer deletes an amplification factor.
C, D and F each traded work onto a unit that turned out not to have room — the VPU for C, the
SparseCore for D, the MXU's issue path for F. §4 established that the inner step is VPU-bound and
this document then opened two options on the strength of it; both are now dead, and F is dead
*because* the corollary drawn from §4 — spare MACs mean spare MXU — was false.

So the honest state of the 8.42 µs inner step is that **the head-sum bundle and the relu chain have no live
candidate between them** — together 85% of the step on §4.1's `modelled` split, more on the
alternative reading that section now brackets. A1 would have removed both and computes the wrong
function. F would have removed the first and is not exact. E1 and E2 remain, both unpriced, and the
`NUM_HEADS` ladder now in flight is the prerequisite for pricing either. That is a real limit on this
exercise and it should be stated in `final_report.md` rather than buried under two wins that come
from elsewhere in the kernel.

## 7. The question above all of these — **answered: not resolvable end to end**

`report.md` § "Why long context" established that at the checked-in production shape this block is
**2.69% of the decode step and 1.02% of the prefill step** against an end-to-end `spread_pct` of
**2.77%**, so a 20% kernel win is ~0.5% end to end. The A/B was to be run at long context on the
expectation that the indexer's share grows there — its cost is O(resident KV positions) while the
sparse-MLA it feeds saturates at `k = 2048`.

**A profiled baseline at the A/B shape has now measured it: the share barely moved, and the
expectation was wrong for a reason `report.md` had already written down.**

`measured (j-074e570b-13e3-4ab7-bb0c)`, `JobStatus = Succeeded`, GLM-5.2-FP8, ISL 65536 / OSL 1024,
`max_model_len` 66560, concurrency 64, `num_prompts` 192, DP=16, 2-node, `ENABLE_XPROF=1`. Shape
verified from the client rather than the recipe: `Total input tokens: 12582912 = 192 × 65536`
exactly, `Total generated tokens: 196608 = 192 × 1024` exactly. Numbers below are from the **clean**
pass (after `bench-start-event: name=benchmark`), not the profiled pass — which ran 27.9% slower.

### 7.1 The share, at the representative step

Only `decode_only` is representative. **Both prefill captures landed on chunks 1–6 of 64** — the
phased profiler fires at the start of the run, and resident KV there is ≤ 5115 of 65536, which is
precisely the quantity indexer cost is proportional to. Each retained one device step. So every
mixed/prefill figure below is a **floor**, not a measurement of long context.

| | decode step (50.110 ms/step) | mixed step (650.169 ms, early chunks) |
| --- | ---: | ---: |
| `StreamIdxTC*` — **the kernel this exercise changes** | **3.07%** | **0.256%** |
| `sparsecore_topk` | 2.478% | 1.239% |
| **whole indexer block (`conditional`)** | **10.13%** | **4.61%** |

**3.07% against the production shape's 2.69%.** The scoring kernel barely moved, and that
**confirms `report.md`'s own counterweight**: decode indexer cost is set by resident tokens per
rank, which is HBM-capped, not by `max_model_len`. A second dump pooling both TensorCores gave
2.79%; call it 2.8–3.1%.

**The decisive ratio is inside the block, not the block's size.** In a mixed step the Pallas kernel
is **5.5% of its own `conditional`**. The other 94.5% is fixed-size data movement — the `:104`
relayout (§7.3), the fully-exposed SparseCore wait, the gather — **none of which a scoring-kernel
edit touches.**

### 7.2 The arithmetic that ends the A/B

**No `spread_pct` exists at this shape**, and an n=1 run cannot establish one. The 2.77% is borrowed
from a sibling experiment at ISL 8192 (n=4, hours apart), and its own history is the warning: the
first estimate at that shape was 0.40% from n=3 taken 58 s apart — a **7× underestimate**. Using
2.77% is therefore already generous to the change.

At one run per arm the standard error of the difference is `σ√2 = 3.92%`, so the **smallest
resolvable end-to-end effect is ≈ 8%**. Against that:

| what improves | end-to-end effect | runs per arm to resolve |
| --- | ---: | ---: |
| the scoring kernels only (what the candidate does) | 0.21–0.70% | **250–2700** |
| the entire indexer block (which nothing here does) | 1.32–1.79% | 38–69 |
| *for scale:* deleting the indexer block outright | 6–8% | still at the threshold |

At 2.5–4 h a run, none of these is purchasable. **The one-run-per-arm A/B cannot distinguish the
composed candidate from the incumbent, and would not have been able to distinguish a candidate four
times better.** Judge the candidate on its microbenchmark and its correctness job — it has both.

This is the outcome the section was written to reach, and reaching it cost one baseline run rather
than two full arms.

### 7.3 What the profile found instead — and it is Option A's writer

The largest indexer-adjacent cost in prefill is **not the kernel**. It is the relayout triad at
`mla_attention_op.py:104` — reshape-in, `scatter_fusion`, reshape-out — which is
`measured (j-074e570b)`:

| phase | per indexer layer | × 21 layers | share of device time |
| --- | ---: | ---: | ---: |
| `prefill_only` | 773.48 µs | 34.033 ms | **2.72%** |
| `prefill_heavy` | 774.81 µs | 16.272 ms | **2.50%** |

That is **9.8× the entire `StreamIdxTC` kernel cost in the same step**, and the 290.18 µs
reshape-out reproduces an earlier ~291 µs to the microsecond. Two things sharpen it: it is a
**108.79 MB round trip to write 256 KB — 415× write amplification** — and it is **paid on the
early-exit fast path too**, appearing in full in a `prefill_only` step where the scoring kernel
never executed at all. It is absent in decode, where XLA keeps the `[415,256,4,256]` layout and the
scatter costs 2.694 µs/call.

**Option A's writer change targets exactly this line**, and is `measured` 3.342× faster at
`T = 1024` (§5b-v2). If it carries that ratio to the triad — `assumed`, since the probe measured a
standalone store and not the triad in situ — it removes 70.1% of 2.50–2.72%, i.e. **1.75–1.91% of
device time**, `modelled`. That is **2.5–9× the whole scoring-kernel prize** computed in §7.2, and
it lands on prefill, which is where TTFT comes from. **This inverts the exercise's ranking at the
end-to-end level**: B is the largest kernel win and A's writer is the largest *deployment* win.

The claim needs one measurement to stop being `modelled` — the triad, in situ, before and after A's
writer — and that is a `/preflight-check`-shaped question, not another kernel exercise.

### 7.4 Three carried findings from the same profile

1. **SparseCore is near saturated** — SC-0 busy is 94.7% of the decode step and 96.6% of the mixed
   step, and its `async-done` is TC-blocking with **zero overlap**. Any future candidate that wins
   by moving work *onto* SparseCore will not get it free. This retro-justifies §4.6's deferral.
2. **The decode:mixed split of §2 is not reproduced by the capture.** Weighting the measured
   per-step figures gives 41.7% / 58.3% as captured, or 30.2% / 69.8% `modelled` under a late-chunk
   extrapolation, against §2's `modelled` 12–14% / 86–88%. The **direction** holds — mixed is a
   minority of invocations and a majority of indexer time — but the magnitude does not, and the
   early-chunk capture biases the comparison against mixed. §2 stays `modelled`; a re-capture with
   `delay_iterations` past chunk 40 is the one run that would settle it, and is the highest-value
   follow-up this profile leaves.
3. **21 of 78 layers is by design, not truncation.** The model config gives
   `indexer_types = Counter({'shared': 57, 'full': 21})`, `index_topk_freq = 4`; 57 layers reuse the
   selection.

### 7.5 Baseline numbers, for the record

Clean pass, `j-074e570b-13e3-4ab7-bb0c`, 192/192 successful, 0 failed, 0 preemptions:

| metric | value |
| --- | ---: |
| output token throughput | 165.00 tok/s |
| total token throughput | 10725.09 tok/s |
| median / P99 ITL | 51.96 / 792.29 ms |
| median / P99 TTFT | 93600.87 / 243723.45 ms |
| median / P99 TPOT | 265.49 / 352.48 ms |
| median / P99 E2EL | 367635.69 / 505903.41 ms |

Where decode time actually goes, for whoever optimizes this model next: `gmm_v2` 17.99% + 9.74%,
`reduce-scatter` 10.57%, **indexer `conditional` 10.13%**, `fusion` 9.04%, `sc_ragged_gather_v2`
6.96%, `all_gather` 6.59%, **`MLA` only 5.36%**. MoE plus its communication is ≈56% of the decode
step. In `prefill_heavy`, fully-exposed collectives are 32.6% — the largest single bucket.

---

## 8. Follow-up probe — four questions, one job

`j-59bbfd2f-e408-4a75-a155`, `JobStatus = Succeeded`, 4/4 shards `rc=0`. Commits `3dc5932d`
(probes), `642dadc6` (`RESULTS.md`, raw shard logs, per-shard JSON), under `probes/followup1/`.
`instruments/` byte-identical throughout.

### 8.1 A2 — the `w` clamp: **FAIL at all five regimes. E1 stays disqualified.**

| regime | rows | bad | worst_rank / 2304 | min_recall / 0.99 | control | self-ref |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `decode_prod` | 64 | 64 | 8190 | 0.47656 | FIRED | OK |
| `mixed_prod` | 1024 | 1024 | 8175 | 0.44141 | FIRED | OK |
| `mixed_32k` | 1024 | 1024 | 32677 | 0.21973 | FIRED | OK |
| `mixed_64k` | 1024 | 1024 | 65286 | 0.20215 | FIRED | OK |
| `decode_64k` | 6 | 6 | 64614 | 0.43750 | FIRED | OK |

3142 of 3142 rows bad — the same total A1 failed, and recall misses by 2–5×. The self-reference arm
returned `worst_rank = 2048, min_recall = 1.00000` everywhere, so the gate was live.

**A2 was expected to be a much smaller change than A1 and is not.** Clamping `w ← max(w, 0)` zeroes
roughly 16 of 32 heads per token, which is close to the negative control's own mutation. **E1's
13.70% stays unusable.**

**The obvious objection, checked at the source rather than assumed.** The gate's inputs draw
`w ~ uniform(-1.5, 1.5)`, so ~50% are negative — which could have made this an artefact of a
synthetic generator rather than a fact about the model. It is not. In production
(`deepseek_v4_indexer.py:81`) `weights = indexer_weights * softmax_scale * (n_head**-0.5) *
q_scales`; the last three factors are non-negative, so the sign is entirely `indexer_weights`'s, and
that is `self.indexer.weights_proj(hidden_states)` (`models/vllm/deepseek_v4/attention.py:397`) — a
**raw linear projection with no softmax, sigmoid or ReLU on it**. Nothing in the architecture
constrains its sign, so a symmetric draw is a fair model and the burden is on E1 to prove
non-negativity, which nothing supplies. *Caveat:* this is read off the architecture, not off trained
weights; the empirical distribution under a real checkpoint is `unmodelled`. It does not change the
verdict, because an unconstrained projection is not a guarantee.

### 8.2 The `NUM_HEADS` ladder: **the head-sum term is constant in `NUM_HEADS`, not linear**

`t(H) = a·H + b` in µs per inner step, `H ∈ {4, 8, 16, 32, 64}`:

| series | a (µs/head) | b (µs) | R² | at H=32, linear share | dominates |
| --- | ---: | ---: | ---: | ---: | --- |
| `bcast_sum` (no matmul) | −0.000205 | 1.5370 | 0.011 | — | **constant** |
| `mxu_sum_mxu_removed` | +0.001056 | 3.0144 | 0.009 | 1.1% | **constant** |
| `mxu_sum_raw` | +0.038438 | 2.8803 | 0.905 | 29.9% | constant |
| `mxu_full` | +0.097370 | 3.5882 | 0.986 | 46.5% | constant |

The **ordering** is the finding. `a` is non-trivial only in the arms that retain a matmul whose FLOPs
are themselves ∝ H; remove the matmul and `a` collapses 36–92× and R² collapses to noise. The five
`bcast_sum` medians span 7.2% while H grows 16×.

**This invalidates the premise this probe was run on**, which was mine. §4.1 identified the 3.858 µs
head-sum bundle as the largest term and I treated it as per-head reduction *arithmetic*. It is not:
it lives in `b` — the store of the reduced result plus fixed per-step cost. **Any option priced on
"do less per-head reduction work" is over-priced, and that is E1, E2 and F.** All three move down,
and F is already dead for an unrelated reason.

Two caveats kept rather than buried: the `bcast` arm hit **300.57% spread** at H=8, so its
*coefficient* is unquotable (the qualitative conclusion survives only because a linear term would
have to be enormous to hide inside a 7.2% span), and the MXU-removed intercept rests on a two-point
K slope implying 1861–3891 Tops/s, at or above fp8 peak. Also: **3.858 µs is arm-dependent to 2.7×**
(1.53 µs `bcast/sum` against 4.13 µs `mxu/sum` at H=32), so any downstream estimate quoting it needs
that error bar.

### 8.3 Option B: linearity is **half** real, and §5c under-stated the prize

Six rungs, both arms, `max_model_len = 66560`:

| arm | refit | R² | residuals |
| --- | --- | ---: | --- |
| B | `t = 0.14745 + 0.007103·w` | **0.99990** | ≤ 2.1% |
| incumbent | `t = 2.32604 + 0.005338·w` | 0.88555 | to +12.1%, systematic arch |

**B is linear; the incumbent is not.** The incumbent's local slope is ≈0.0100 ms/sublane below
w=256 and 0.001526 then 0.003032 above — an unexplained **6.7× kink between w=256 and w=384**.

So §5c's flagged suspicion — that B's fitted slope came out steeper than the incumbent's — was
**correct and mis-attributed.** B has one slope everywhere; the incumbent has no single slope, and
§5c's two anchors happened to land one on each side of the kink. All four §5c anchors reproduced
within ~1% before any refit.

**Bucketing, measured rather than modelled.** Only two rungs distinguish the ladders. Cost of
coarsening (positive = slower): `kv 24576 →` pow2 32768 is **+2.25%** (1.49414 → 1.52770 ms);
`kv 49152 →` pow2 65536 is +1.16% (2.88717 → 2.92069 ms), which is at or below the 1.29–2.67% arm
spreads and **cannot alone support a conclusion**. Against B's 34–58% prize at those rungs the §5c
claim holds. Revised chunk-averaged prize, `modelled` by interpolation between measured rungs:

| ladder | §5c published | **measured-rung revision** |
| --- | ---: | ---: |
| exact `kv_len` | −41.3 | **−46.2** |
| 8192-granularity | −34.6 | **−40.9** |
| powers of two | −23.4 | **−30.2** |

**The P claim is confirmed by direct read-back**, not inferred from timing: shimming
`impl.sparsecore_topk` to record the real `(b, n, k)` and interrogate the real `_pick_partition`
gives **incumbent P = [2]** at n=67584 at every rung and **B P = [1]** at all six buckets
(8192…65536), with `MAX_SLICE_WORDS = 65536` and `b = 1024 ≥ 32 ⇒ max_p = 1`.

### 8.4 `(8,4,4)` at `mixed_prod`: usable now, and a **slowdown**

The 491.23% spread **did not reproduce** — the same `run_regime` path at reps=40 gives 8.00%, and
both published medians reproduce within 0.7%. The published median was fine; only its range was
poisoned, by a single outlier block (0.82235 against a 0.67238 median).

60 alternating paired blocks, one input build, both arms compiled once:

| old ms (2,2,2) | new ms (8,4,4) | Δ time % | speedup × |
| ---: | ---: | ---: | ---: |
| 0.67238 | 0.70413 | **+4.72** | 0.9549 |

Paired per-block median +4.52%, IQR [+3.91, +5.61], bootstrap 95% CI **[+4.23, +5.09]**, **59 of 60
blocks agree on the sign**. **USABLE: `(8,4,4)` is 4.2–4.7% slower at `mixed_prod`.** Not a proposal
to ship it — its buffer is 128 sublanes against 80.

### 8.5 Two process findings

1. **The 491% spread was not a property of the arm.** Other anomalous-spread rows in
   `j-b54e9716-3db5-4f1f-b8d4` — the `mxu/full/K=128` row at 550.56%, E's `decode_64k` drift control
   at 111.72%, F's two `CONTROL` arms at 10629% and 116% — were each discarded as unusable. At least
   one of them was a single outlier block in an otherwise clean arm. **Discarding on range is
   discarding information**; paired per-block statistics recovered this one at no extra hardware cost.
2. **The branch moved under the probe.** The dispatch named HEAD `eb994645`; the actual HEAD at
   commit time was `54e80686`, and the `--map-dir` tarball carried a working tree dirty in several
   `variants/*` files from concurrent agents. Immaterial here — shard 2's imports are clean files and
   its anchors reproduce §5c within 1% — but running many variant agents against one branch makes
   *what was measured* a question that has to be answered per job rather than assumed.

### 8.6 A second call site, so nobody later fixes the wrong one

`streamindex_topk` has two callers with **different tuples**: `mla_attention_op.py:118` uses
`(2, 2, 2)` and `deepseek_v4_indexer.py:93` uses `(3, 2, 2)`. `report.md` established
`mla_attention_op.py` as the production path and verified it against **profiled scope names**, not
just the source, so `(2, 2, 2)` is the right incumbent and every `bkv_p` measurement here is against
the right baseline. Recorded because the discrepancy is invisible until you look for it.

---

## 9. `/critical-review` of §5b-v2 and §5c

Run in a clean context on the two survivors, before the end-to-end arms rather than after them,
with the seven `cdk job recipe` read-backs pasted in verbatim (`/tmp/review_artifacts/`).

**Verdict: `PLAUSIBLE-UNCONFIRMED`.** Both wins reproduce arithmetically from the raw shard logs.
Neither is `accepted`, and B's is gated.

| # | severity | finding | where it is now recorded | state |
| ---: | --- | --- | --- | --- |
| 1 | **CRITICAL** | B's poison arm **could not have failed** — `run_correctness.py:121-122` binds `kv_len_bucket = r.kv_len` in every arm, so zero lanes are unwritten at all eight shapes ever run. The shipping configuration (`bucket > kv_len`) has only run perf-only, and the two rungs that ran it are the two the −30.2% pow2 figure rests on. | §5c ⚠️ block, §5c-gate | **resolved** — `j-6454262e-acd2-4371-b3ba` re-ran the oracle with 64/64/128/8 empirically-counted unwritten sublanes on both bands: **PASS**, bit-identical to the incumbent, both controls RED |
| 2 | major | A's weighted contribution overstated ≈1.4× — `decode_prod`'s −10.45% (a `max_model_len = 9216` shape) multiplied by A/B-derived phase weights. A/B-consistent is `decode_64k`'s −7.66%. | §5b-v2, §6 | **fixed** — 0.93–1.10%, face value 1.2–1.4% |
| 3 | major | §4.1's 52.5/32.7/14.8 split is `modelled` from a two-point fit whose upper point §4 itself declares unusable (550.56% spread), through a fit the same table shows non-linear. §8.2 additionally found the bundle arm-dependent to 2.7×, and that correction was never propagated. | §4.1 block, propagated to §5d, §5f, §6 | **fixed** — bracketed 52–66%, tagged `modelled` |
| 4 | minor | §5c's `mean w` inputs (292, ~346) match no derivation. | §5c | **fixed** — 288 and 341, derivations shown; both move in B's favour by < 1.2 pt |
| 5 | minor | §2's two columns are on different denominators — steps at `M/(N+M)`, time at `(N−M)/N`. | §2 | **fixed** — steps corrected to 76.4/23.6; the **time** bracket was already consistent and is unchanged |
| 6 | minor | A1/A2's row count is 3142 (64+1024+1024+1024+6), not 3138. | §5d, §8.1 | **fixed** |
| 7 | minor | §5b quotes the writer's absolute 0.592 ms against `writer_probe.py`'s own warning that absolutes drift and ratios do not. | §5b-v2 | **fixed** — quote 3.342× |
| 8 | minor | C's job carried no drift control, so two of its four rows are read against the lenient yardstick. | §5e | **noted**; C's verdict rests on the ceiling, not the deltas |

**What the review did not find.** It reproduced B's `mixed_confound` 3.725× and `mixed_64k` −21.72%
from the shard logs, confirmed the `RUN_CMD` read-backs match what the tables claim was run, and
raised no objection to A's bit-exactness argument or to the fp8 lowering evidence. Finding 1 is
about the *test*, not about the kernel: nothing here says B is wrong, only that the evidence
offered does not cover the risk B carries.

**Standing rule this exercise leaves with.** A correctness test whose configuration guarantees its
own pass is worse than no test, because it is spent. The tell is that the harness binds the very
parameter the risk is a function of. That is a `references/` promotion.
