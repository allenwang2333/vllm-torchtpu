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

| regime | incumbent (2,2,2) ms | variant (4,2,4) ms | Δ time % | speedup × |
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

| band | share of invocations | share of **time** |
| --- | ---: | ---: |
| DECODE | 80.9% | **12.1–14.3%** |
| MIXED | 19.1% | **85.7–87.9%** |

The time bracket comes from `decode_64k` at 0.20202844752930105 ms against `mixed_64k` at 4.76284 ms
(high) and 3.91366 ms (low — mean `num_bkv` over the 64 prefill chunks is exactly 16.5 of 32).

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

## 5b. Variant A — Narrow Record: reader blocked, **writer wins 3.4×**

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

## 6. Ranking as it stands

Not final — A, B, C, F and A1 are in flight, and F or A1 landing would rewrite the top of it.

| rank | option | standing | why it moved from step 2 |
| ---: | --- | --- | --- |
| **1** | **A — Narrow Record** | **writer measured, reader blocked; retry authorised** | its blocking `unmodelled` term came back a **win** (3.406×), not a cost (§5b). Design doc §9.1: struck, "A moves to rank 1 outright". Contingent entirely on the reader compiling. |
| **2** | **B — Right-Size** | in flight | mixed carries 85.7–87.9% of indexer time (§2), and B's prize is mixed- and confound-shape-only. Design doc §9.2's own stated consequence. |
| **3** | **E — Retune** | **measured** | fell from joint rank 1. Net at the A/B is −0.84% to +0.90%, sign unresolved (§1.3). |
| **4** | **C — Half-Width Key** | in flight | §7.5 predicts the confirmed VPU result makes C *actively worse*, since a 16-bit key adds a VPU-side encode on the bottleneck unit. |
| **5** | **D — No Round Trip** | **dead** | priced at 8.96× its threshold (§3). Not built. |
| **?** | **F — MXU head sum** | in flight | targets 52.5% of the inner step; exact. Not in step 2. |
| **?** | **A1 — Drop the ReLU** | in flight, gated | targets ~85% of the inner step; an approximation, and expected to fail its gate. Not in step 2. |

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
dominant cost is arithmetic on a saturated unit. The two candidates now most likely to matter, F and
A1, exist only because a strike test was run on a hypothesis the document itself rated poorly.

## 7. The question above all of these

`report.md` § "Why long context" establishes that at the checked-in production shape this block is
**2.69% of the decode step and 1.02% of the prefill step** against an end-to-end `spread_pct` of
**2.77%**, so a 20% kernel win is ~0.5% end to end and would need ~190 runs per arm to resolve —
*"the block is not measurable end to end at that shape, and saying so is part of the result."*

The A/B is being run at long context because the indexer's cost is O(resident KV positions) while
the sparse-MLA it feeds saturates at `k = 2048`, so its share should grow. **That expectation has
never been measured.** A profiled baseline run at the A/B shape is in flight to settle it. If the
share is still low single digits, no kernel change in this document is resolvable in the
one-run-per-arm A/B, and that is the finding — reached before two four-hour runs rather than after.
