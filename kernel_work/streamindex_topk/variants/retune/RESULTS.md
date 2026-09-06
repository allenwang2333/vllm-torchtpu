# Variant E — **Retune**

Step 3 of `/write-tpu-kernels` for `streamindex_topk`. One compile-time tuple, measured against the
incumbent **in a single job, by the read-only step-1 instruments**, which no prior job had done.

**Job `j-ce9a19a4-4639-4a8f-9059`**, `JobStatus: Succeeded`, `tpu-fanout: 4/4 shards succeeded`, all
four `rc=0`. Image
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.
One `tpu7x` pod, one chip per shard, one TensorCore.

---

## 0. Sign convention and tags — binding on every number below

Every percentage is a change in **time**, `(new − old)/old`. **Negative is faster.** Signed
percentages appear only in a table column headed `Δ time %`, and every such row carries the two
absolute times the delta came from. Prose says the direction in words. A `speedup ×` column sits
alongside where it is useful; it is `old/new`, **>1 is faster**, and it is a different number from
the delta.

Every number carries exactly one tag: `measured (job id)`, `modelled`, `assumed`, or `unmodelled`
(which appears alone, never with a number beside it).

---

## 1. What changed, and why

`num_kv_pages_per_block` at `mla_attention_op.py:119` goes from the shipped `(2, 2, 2)` to
**`(4, 2, 4)`**. `retune.patch` is the entire change: one tuple and the comment that explains why
its three slots are not equal. No arithmetic changes, no kernel body changes, no new file. It is a
static argument to `streamindex_topk`, so it is a recompile, not an edit.

**The mechanism.** `bkv_p` is how many 1024-position KV pages one inner `(bq, bkv)` iteration loads
and scores. `probes/step_cost_RESULT.md` measured a fixed cost of **≈500 ns per inner iteration**
[`measured (j-e8eaacd3-6be4-4cdf-95a2)`]. Doubling `bkv_p` halves the iteration count and so halves
what that fixed cost totals. It is an amortization, not a removal — the per-step cost is a property
of the hand-rolled inner loop and is still there.

**The counterweight, and why it is not free.** `num_sublanes_total` — the width of the f32 scores
buffer — is `max(align_to(pages_per_seq, bkv_p) for bkv_p in the tuple) × 8`
(`streamindex_topk.py:603-605`), because **one buffer serves all three `pallas_call`s**. So raising
any slot widens the buffer for every band. At `max_model_len = 9216`, `pages_per_seq = 9` and
`align_to(9, 4) = 12`, so the buffer goes 80 → 96 sublanes, a 20% widening, written by the
TensorCore and read again by the SparseCore. Decode carries 64 tokens and mixed carries 1024, so
mixed pays 16× the absolute width cost. That is why the two phases move in opposite directions in
§3, and it is the whole of §5's argument.

### 1.1 The slot order, and why slot 1 is pinned to 2

Verified in source this session, and the older probe record gets it wrong:

- The kernel makes **three** `pallas_call`s and they read slots **`[0]`, `[0]`, `[2]`** —
  `case=MlaCase.DECODE` twice (the batched path and the remainder path) and `case=MlaCase.MIXED`
  once, at `streamindex_topk.py:765`, `:783` and `:801`. Slot order is
  **`(DECODE, PREFILL, MIXED)`**. `probes/bkv_p_RESULT.md`'s terms section originally called it
  `(decode, mixed, prefill)`; `report.md` is right and that file now carries the correction.
- **Slot 1 is read by no band that executes.** `mla_attention_op.py:183` builds
  `distribution = torch.stack([rd[0], rd[0], rd[2]])`, so the PREFILL-only invocation spans zero
  sequences for GLM-5.2.
- **And it still participates in the `max`.** So slot 1 can only ever make the scores buffer wider,
  never narrower, and it does so for a band that does not run.

`(4,4,4)` and `(4,2,4)` are therefore **identical in everything that executes**: same `max` (4),
same decode band (slot 0 = 4), same mixed band (slot 2 = 4). §4 measures that identity rather than
asserting it, and decides which spelling to land.

---

## 2. How it was measured, and what is new about this job

`instruments/microbench.py` and `instruments/test_correctness.py` are **read-only after step 1** and
were not edited or forked — a yardstick that changes between candidates is not a yardstick.
`measure_retune.py` rebinds the module attribute `shapes.NUM_KV_PAGES_PER_BLOCK` around each call
and restores it in a `finally`; the instruments are imported and executed exactly as written. Same
mechanism `probes/bkv_p_prod.py` used, which is what makes these numbers comparable to that file's.

`early_exit=False` throughout, matching step 1's baseline table and every prior `bkv_p` arm. It is
held constant across both arms. Production sets it `True` (`mla_attention_op.py:57`); at these
shapes the gate is false, so it adds a `lax.cond` and nothing else — see §7 caveat 4.

**Four arms per regime, in this order, all in one process:**

| arm | tuple | what it is for |
| --- | --- | --- |
| `incumbent` | (2,2,2) | the denominator of every delta |
| `variant` | (4,2,4) | the variant, as `retune.patch` spells it |
| `variant-alt` | (4,4,4) | the variant as the probe record spells it — a **replicate**, per §1.1 |
| `incumbent-repeat` | (2,2,2) | the **drift control** |

The last two arms are the point of the exercise and neither existed before.

- The **replicate** `(4,2,4)` vs `(4,4,4)` is two measurements of a bit-identical hardware
  configuration. Whatever separates them is instrument, not design.
- The **drift control** re-measures the incumbent last. `full_spread_pct` is the dispersion
  *within* one arm's reps; it does not bound drift *across* arms in one process, where compile
  order, allocator state and host temperature all move. The prior mixed deltas (+1.61%, +1.67%)
  were about the size of a single arm's spread, so whether they mean anything turns on exactly this
  number — and it had never been taken.

Prior jobs measured decode in `j-98399433-7763-4385-81ca` and mixed in `j-3aa755dd-900b-4f31-9fa3`,
and never measured the `(4,2,4)` spelling at all. §6 shows the cross-job baseline shift that makes
same-job pairing worth the shard.

---

## 3. The four-regime table

**All rows `measured (j-ce9a19a4-4639-4a8f-9059)`.** Median of 10 reps, single chip, one
TensorCore, `early_exit=False`, whole `streamindex_topk` call (scoring *and* SparseCore selection).
`Δ time %` and `speedup ×` are against the `incumbent` arm of the **same regime in the same
process**, whose absolute time is in the table two rows above. **Negative `Δ time %` is faster;
`speedup ×` > 1 is faster.**

| regime | arm | tuple | subl | scores MB | num_bkv | full ms | spread % | Δ time % | speedup × |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | incumbent | (2,2,2) | 80 | 2.62 | 4 | 0.16585062257945538 | 0.86 | 0.00 | 1.0000 |
| `decode_prod` | variant | **(4,2,4)** | 96 | 3.15 | 2 | 0.15347349981311711 | 4.23 | **−7.46** | 1.0806 |
| `decode_prod` | variant-alt | (4,4,4) | 96 | 3.15 | 2 | 0.15252399898599830 | 0.57 | **−8.04** | 1.0874 |
| `decode_prod` | incumbent-repeat | (2,2,2) | 80 | 2.62 | 4 | 0.16579472285229716 | 0.76 | −0.03 | 1.0003 |
| `decode_64k` | incumbent | (2,2,2) | 528 | 1.62 | 32 | 0.19944237428717315 | 0.43 | 0.00 | 1.0000 |
| `decode_64k` | variant | **(4,2,4)** | 544 | 1.67 | 16 | 0.17783924995455890 | 4.31 | **−10.83** | 1.1215 |
| `decode_64k` | variant-alt | (4,4,4) | 544 | 1.67 | 16 | 0.17738954920787364 | 1.00 | **−11.06** | 1.1243 |
| `decode_64k` | incumbent-repeat | (2,2,2) | 528 | 1.62 | 32 | 0.20384559757076204 | **111.72** | +2.21 | 0.9784 |
| `mixed_prod` | incumbent | (2,2,2) | 80 | 41.94 | 4 | 0.67018477711826560 | 1.40 | 0.00 | 1.0000 |
| `mixed_prod` | variant | **(4,2,4)** | 96 | 50.33 | 2 | 0.68319969868753100 | 3.29 | **+1.94** | 0.9810 |
| `mixed_prod` | variant-alt | (4,4,4) | 96 | 50.33 | 2 | 0.68712327338289470 | 2.13 | **+2.53** | 0.9753 |
| `mixed_prod` | incumbent-repeat | (2,2,2) | 80 | 41.94 | 4 | 0.66499382664915170 | 1.41 | −0.77 | 1.0078 |
| `mixed_64k` | incumbent | (2,2,2) | 528 | 276.82 | 32 | 4.78381379507482000 | 0.95 | 0.00 | 1.0000 |
| `mixed_64k` | variant | **(4,2,4)** | 544 | 285.21 | 16 | 4.88845249637961400 | 1.97 | **+2.19** | 0.9786 |
| `mixed_64k` | variant-alt | (4,4,4) | 544 | 285.21 | 16 | 4.87273340113461000 | 1.80 | **+1.86** | 0.9818 |
| `mixed_64k` | incumbent-repeat | (2,2,2) | 528 | 276.82 | 32 | 4.78449129732325700 | 1.43 | +0.01 | 0.9999 |

`subl` is `num_sublanes_total`, the `max` over the whole tuple, so it moves for every band.
`num_bkv` is the inner iteration count for **this regime's own** band — note it does **not** move at
`mixed_*` between `(2,2,2)` and a tuple whose slot 2 stays at 2, which is why §5 does not propose
`(4,2,2)`.

### 3.1 Pooled reading — two incumbent arms against two variant arms

The replicate and the drift control are two measurements each of the same configuration, so the
honest headline pools them. Same job, same rows, arithmetic mean of the pair.

| regime | incumbent ms (pooled) | variant ms (pooled) | Δ time % | speedup × | reading |
| --- | ---: | ---: | ---: | ---: | --- |
| `decode_prod` | 0.16582267 | 0.15299875 | **−7.73** | 1.0838 | **7.73% faster** |
| `decode_64k` † | 0.19944237 | 0.17761440 | **−10.94** | 1.1229 | **10.94% faster** |
| `mixed_prod` | 0.66758930 | 0.68516149 | **+2.63** | 0.9744 | **2.63% slower** |
| `mixed_64k` | 4.78415255 | 4.88059295 | **+2.02** | 0.9802 | **2.02% slower** |

All `measured (j-ce9a19a4-4639-4a8f-9059)`.

† **`decode_64k`'s incumbent is the first arm alone, not a pool.** Its `incumbent-repeat` arm
returned a spread of **111.72%** — the arm is contaminated and its median is not usable, so it is
excluded from the pool and from the drift control below. That is a result, not a defect worked
around: it means this job establishes no arm-to-arm drift bound at `decode_64k`, and the
`decode_64k` row rests on the two variant arms agreeing (−10.83% and −11.06%) against a first
incumbent whose own spread is 0.43%, plus the prior job's independent −11.33%.

### 3.2 The controls — what this instrument can and cannot resolve

**Drift control**, same tuple, same regime, first arm against last:

| regime | first ms | last ms | drift % | usable? |
| --- | ---: | ---: | ---: | --- |
| `decode_prod` | 0.16585062 | 0.16579472 | **−0.03** | yes |
| `mixed_prod` | 0.67018478 | 0.66499383 | **−0.77** | yes |
| `mixed_64k` | 4.78381380 | 4.78449130 | **+0.01** | yes |
| `decode_64k` | 0.19944237 | 0.20384560 | +2.21 | **no** — repeat arm spread 111.72% |

**Replicate**, `(4,2,4)` against `(4,4,4)`, which execute identically:

| regime | (4,2,4) ms | (4,4,4) ms | gap % |
| --- | ---: | ---: | ---: |
| `decode_prod` | 0.15347350 | 0.15252400 | −0.62 |
| `decode_64k` | 0.17783925 | 0.17738955 | −0.25 |
| `mixed_prod` | 0.68319970 | 0.68712327 | +0.57 |
| `mixed_64k` | 4.88845250 | 4.87273340 | −0.32 |

All `measured (j-ce9a19a4-4639-4a8f-9059)`.

**What the two controls together say.** Across the three regimes where both are usable, arm-to-arm
reproducibility inside one process is **≤0.8%** — drift −0.03 / −0.77 / +0.01, replicate gaps
−0.62 / +0.57 / −0.32, no consistent sign. That is a much tighter bound than any single arm's
`spread %`, and it is the number the mixed deltas have to clear. See §5.

Note also that the two `(4,2,4)` decode arms carried spreads of 4.23% and 4.31% while their
bit-identical `(4,4,4)` twins carried 0.57% and 1.00%. Identical configuration, very different
dispersion: `full_spread_pct` is a property of the sample, not of the arm, and pooling the
replicate is the right response to it rather than picking the tidier row.

---

## 4. `(4,4,4)` or `(4,2,4)` — the decision

**Land `(4,2,4)`.** They are the same kernel; they differ in what the next reader believes.

**They measure identically, and here are both numbers.** Across all four regimes the replicate gap
is −0.62%, −0.25%, +0.57%, −0.32% — mean −0.155%, no consistent sign, every one of them inside the
≤0.8% arm-to-arm bound §3.2 establishes. There is no measured basis on which to prefer either
[`measured (j-ce9a19a4-4639-4a8f-9059)`]. Correctness is likewise identical: §5.1 verifies `(4,2,4)`
directly and its six margins are bit-for-bit those of `(4,4,4)` from `j-e13fc627-a8c5-4ef4-bd5e`.
So the decision is made on the source, not on the numbers.

**The argument for `(4,2,4)`.** Slot 1 is `(DECODE, PREFILL, MIXED)`'s PREFILL slot; no invocation
that executes for GLM-5.2 reads it, and it still enters the `max` that sets the scores width. A
value there can only cost sublanes, and only for a band that does not run. Today `(4,4,4)` costs
nothing for it because slots 0 and 2 are also 4, so the `max` is 4 either way — **the tax is zero
now and becomes real the moment the tuple stops being uniform.** That moment is not hypothetical:
`bkv_p_RESULT.md`'s own closing section, and Option B in `measurement_and_design.md`, both propose
choosing `bkv_p` per band, and §5 below is a direct argument for wanting mixed lower than decode.
A tuple that reads `(4,4,4)` teaches the next reader that the three slots move together; the first
person to act on that and write `(4,2,2)` or `(8,2,4)` will carry slot 1 along out of symmetry and
pay width for the empty band without any diagnostic saying so.

**The argument against, and why it does not hold.** `(4,4,4)` is what
`j-e13fc627-a8c5-4ef4-bd5e` correctness-verified, and a uniform tuple is harder to mistype. The
first half dissolves: shard 3 of this job verified `(4,2,4)` directly at three regimes with the
negative control firing, so the shipped spelling is the verified spelling and no inference is
needed. The second is real but small, and it is answered by the comment `retune.patch` adds, which
states the slot order, the three call sites, and why slot 1 is pinned — the asymmetry is the
documentation.

**What is *not* being claimed.** `(4,2,4)` is not faster than `(4,4,4)`, and nothing here says it
is. It is the same measurement written in a way that survives the next edit.

---

## 5. The mixed cost is real — a correction to `bkv_p_RESULT.md`

`probes/bkv_p_RESULT.md` states, of its `(4,4,4)` mixed rows: *"'Roughly free at mixed, and not
measurably slower' is the strongest reading these rows support — not '1.6% slower', and not
'free'."* Its reasoning was that `+1.61%` sat against a 1.77% spread on its own row and 1.68% on
the reference.

**This job supersedes that, and in the unhelpful direction.** The reasoning was right for the
evidence available: a delta the size of one arm's spread is not readable. But arm `spread %` is the
wrong yardstick for an arm-to-arm difference, and the drift control this job adds is the right one.
It reads **−0.77%** at `mixed_prod` and **+0.01%** at `mixed_64k`
[`measured (j-ce9a19a4-4639-4a8f-9059)`], with replicate gaps of +0.57% and −0.32%. The mixed
deltas — +1.94% and +2.53% at `mixed_prod`, +2.19% and +1.86% at `mixed_64k`, pooled **+2.63%** and
**+2.02%** — clear that bound in every arm.

The direction was never in doubt: **eight mixed arms across two jobs have now landed on the slower
side of shipped and none on the faster side** (four in `j-3aa755dd-900b-4f31-9fa3` at +1.61 to
+5.11, four here at +1.86 to +2.53). What changes is that the magnitude is now readable.

**Corrected reading: `(4,2,4)` costs about 2% of a mixed call — roughly 2.0% at `mixed_64k` and
2.6% at `mixed_prod` — and this must not be rounded to zero when the composed candidate is priced.**
The mechanism is the one §1 names and `measurement_and_design.md` §2.1 measured: the 20% wider
scores buffer, written by the TensorCore and read again by the SparseCore, against a mixed step
saving of only `16 × 500 ns ≈ 8 µs` on a 667 µs call [`modelled`].

This is the same fact `measurement_and_design.md` §4.2 already draws on: **Option B's masked
partial-block store decouples the width from the tuple, and it is the ceiling on E.** With the
width decoupled, mixed keeps its own step saving and stops paying for decode's. E and B are not
alternatives and must be measured composed, not summed.

### 5.1 Correctness

Float64 oracle, rank criterion (`RANK_SLACK=256` so `worst_rank ≤ 2304`, `MIN_RECALL=0.99`), through
the read-only `instruments/test_correctness.py`. **At the shipped spelling `(4,2,4)`**, which had
never been value-checked before this job. Verbatim from shard 3 of
`j-ce9a19a4-4639-4a8f-9059`:

```
[decode_prod tuple=(4, 2, 4)] PASS -- all 64 rows a valid top-2048 vs the float64 oracle. worst_rank=2109 of allowed 2304, min_recall=0.99561 of required 0.99
[decode_64k tuple=(4, 2, 4)] PASS -- all 6 rows a valid top-2048 vs the float64 oracle. worst_rank=2105 of allowed 2304, min_recall=0.99512 of required 0.99
[mixed_prod tuple=(4, 2, 4)] PASS -- all 1024 rows a valid top-2048 vs the float64 oracle. worst_rank=2098 of allowed 2304, min_recall=0.99512 of required 0.99
```

Negative control, same tuple, same job:

```
[mixed_prod tuple=(4, 2, 4)] NEGATIVE CONTROL OK -- fired on 1024 of 1024 rows. worst_rank=8159 of allowed 2304, min_recall=0.42578 of required 0.99
```

It fired on every row and missed `worst_rank` by 3.5×, so the three PASSes are not vacuous.

**Every one of those six figures is bit-for-bit identical to the incumbent's and to `(4,4,4)`'s:**

| regime | metric | shipped (2,2,2) | (4,4,4), `j-e13fc627-a8c5-4ef4-bd5e` | **(4,2,4), this job** | identical? |
| --- | --- | ---: | ---: | ---: | :---: |
| `decode_prod` | `worst_rank` (allowed 2304) | 2109 | 2109 | **2109** | yes |
| `decode_prod` | `min_recall` (required 0.99) | 0.99561 | 0.99561 | **0.99561** | yes |
| `decode_64k` | `worst_rank` | 2105 | not run | **2105** | yes (vs shipped) |
| `decode_64k` | `min_recall` | 0.99512 | not run | **0.99512** | yes (vs shipped) |
| `mixed_prod` | `worst_rank` | 2098 | 2098 | **2098** | yes |
| `mixed_prod` | `min_recall` | 0.99512 | 0.99512 | **0.99512** | yes |
| `mixed_prod` | control `worst_rank` | 8159 | 8159 | **8159** | yes |
| `mixed_prod` | control `min_recall` | 0.42578 | 0.42578 | **0.42578** | yes |

Shipped-tuple figures from `j-6d675a58-7ecd-4b44-9171` (decode) and `j-5f8f4a08-20b4-4fa0-bfc3`
(mixed), via `probes/bkv_p_RESULT.md`; `(4,4,4)` from `probes/bkv_p_444_correctness_RESULT.md`.

**Verdict: correctness verified, bit-identical, at the exact tuple the patch ships.** Not "within
tolerance" — the same numbers. `bkv_p` sets how the KV reduction is *tiled*; the accumulation is f32
either way, so the arithmetic is unchanged and the measurement confirms it rather than assuming it.

---

## 6. Against the two prior jobs — where it reproduces and where it does not

Same instrument, same image, same shapes, different jobs. This is the comparison the brief said no
prior job had made, and it is the reason it was worth a shard.

| regime | arm | prior ms | prior job | this job ms (pooled) | Δ time % across jobs |
| --- | --- | ---: | --- | ---: | ---: |
| `decode_prod` | incumbent (2,2,2) | 0.1682009256910533 | `j-98399433-7763-4385-81ca` | 0.16582267 | **−1.41** |
| `decode_prod` | raised, slot 0 = 4 | 0.1526356732938439 | `j-98399433` (4,2,2) | 0.15347350 (4,2,4) | +0.55 |
| `decode_64k` | incumbent | 0.19925692467950284 | `j-98399433` | 0.19944237 | +0.09 |
| `decode_64k` | raised, slot 0 = 4 | 0.17668985528871417 | `j-98399433` (4,2,2) | 0.17783925 (4,2,4) | +0.65 |
| `mixed_prod` | incumbent | 0.6666850706096739 | `j-3aa755dd-900b-4f31-9fa3` | 0.66758930 | +0.14 |
| `mixed_prod` | (4,4,4) | 0.6774073000997305 | `j-3aa755dd` | 0.68712327 | +1.43 |
| `mixed_64k` | incumbent | 4.789508902467787 | `j-3aa755dd` | 4.78415255 | −0.11 |
| `mixed_64k` | (4,4,4) | 4.869555 (`4.86955470405519`) | `j-3aa755dd` | 4.87273340 | +0.07 |

**The one thing that contradicts `bkv_p_RESULT.md`'s headline number.** That file reports
`decode_prod` at **9.25% faster**; measured here against an incumbent taken minutes away on the same
chip it is **7.73% faster** (pooled) — or 7.46% / 8.04% arm by arm. Both are the same direction and
both clear every control, so the finding stands; the *magnitude* shrinks by about a point and a
half.

**And the cause is not the variant — it is the baseline.** The variant reproduces across jobs to
within 0.55% at `decode_prod`, while the **incumbent came out 1.41% faster in this job than in
`j-98399433`**, which is larger than either arm's own spread (0.86% here, 0.90% there). A prize
computed against a baseline taken in a different job inherits that shift. `decode_64k` and both
mixed regimes reproduce to within 0.15% on the incumbent, so this is a `decode_prod` cross-job
artefact rather than a general one — but it is exactly the failure the same-job pairing exists to
remove, and it happens to be the regime with the headline number on it.

**Nothing else contradicts the probe record.** No arm errored. `decode_64k` at −10.94% pooled sits
against the prior −11.33%; `mixed_64k`'s `(4,4,4)` reproduces to 0.07%. The `subl` and `num_bkv`
columns confirm the band indexing behaves as `streamindex_topk.py:603-605` and `:765/783/801` say:
`(4,2,4)` and `(4,4,4)` give `subl` 96/544 and halve `num_bkv` at **both** decode and mixed, which
is the signature of slot 2 — not slot 1 — driving the mixed band.

---

## 7. Caveats that survive

Every one of these is load-bearing. None is a reason to discard the result; all of them bound what
it may be quoted as.

1. **This is a single-chip microbenchmark, not an end-to-end number, and it must not be scaled into
   one.** One TPU chip, one TensorCore, one layer, one call, through
   `instruments/microbench.py`. GLM-5.2 runs TP=1/DP=16/EP, so the indexer is *replicated* rather
   than sharded and single-chip is the right denominator for the kernel — but it cannot see HBM
   pressure from the other 77 layers, SparseCore contention from the rest of the step, or rank skew
   behind the step barrier.

2. **The block is 1–3% of device time, and only 21 of 78 layers run the indexer.**
   `index_topk_freq=4`, `index_skip_topk_offset=3` — a factor of 3.7 on every per-model-step figure
   (`measurement_and_design.md` §0.4; step 1 put the block at roughly 1–3% of device time at the
   production shape). A 7.7% kernel win on 1–3% of device time on 21 of 78 layers is a small
   end-to-end number before anything else is considered.

3. **The prize is decode-only and the cost is mixed-only.** Whether the net is positive at the A/B
   shape depends on the decode:mixed step mix, which **this job did not measure and does not
   claim**. `measurement_and_design.md` §9.2 makes that the single assumption that flips the
   ranking, and it is being measured separately. **No rank is claimed here.**

4. **`early_exit=False`; production sets it `True`.** Held constant across both arms, so the delta
   is not affected in first order. At these shapes the gate is false (`kv_len // 1 > k`), so it adds
   a `lax.cond` and nothing else, and step 1 measured that at 2.77% more time on a 3.70%-spread arm
   — *"suggestive, not established"*. Unmeasured at `(4,2,4)`: `unmodelled`.

5. **`decode_64k` has no drift bound in this job.** Its `incumbent-repeat` arm returned 111.72%
   spread; the arm is discarded and so is the +2.21% "drift" computed from it. The `decode_64k`
   conclusion rests on the two variant arms agreeing and on the prior job, not on a control taken
   here.

6. **`bkv_p = 8` is neither measured here nor value-checked anywhere.** At `decode_prod` 4 and 8 were
   tied in `j-98399433` (0.15264 vs 0.15270 against 1.3–1.4% spreads), but at `decode_64k` 8 was
   2.5 points better (−13.79% vs −11.33%) — so the long-context optimum is bracketed, not found, and
   `(8,2,8)` was never run. E takes 4. Anyone wanting 8 owes both a timing arm and a correctness
   arm; each is one shard.

7. **The achieved-vs-floor reading is not directly measurable from this run.** Step 1's 55%
   verdict is `41.7 µs / 76.42 µs`, where 76.42 µs is the **profile's Pallas scope** — the scoring
   half — and this job measured the **full call**, selection included, with `measure_topk=False`.
   Attributing the whole 12.82 µs saving at `decode_prod` to the scoring scope (`assumed`; selection
   is unchanged by `bkv_p` and the per-step cost lives in the scoring loop) puts the scope at
   63.60 µs and achieved at **66%** against F-id or **74%** against F-grid [both `modelled`, and
   they mix a microbenchmark delta into a profile absolute, which is why they are modelled and not
   measured]. Either way the exit-table reading is unchanged: **50–80%, so one more iteration with
   attribution** — E does not reach the ≥80% bar that would stop iterating.

8. **The `mixed_32k` regime was not run.** Three of step 1's five regimes are covered plus
   `decode_64k`; `mixed_32k` was dropped for shard budget and because its `topk ctrl` spread makes
   it the least informative of the mixed rows. It is expected to sit between the two mixed regimes
   measured; that expectation is `assumed`.

---

## 8. Reproducing it

```
export PYTHONPATH=/workspace:$PYTHONPATH && bash /root/fanout/tpu-fanout.sh "python3 /workspace/kernel_work/streamindex_topk/variants/retune/measure_retune.py --mode sweep --regimes decode_prod,decode_64k --json /tmp/retune_decode.json" "python3 /workspace/kernel_work/streamindex_topk/variants/retune/measure_retune.py --mode sweep --regimes mixed_prod --json /tmp/retune_mixed_prod.json" "python3 /workspace/kernel_work/streamindex_topk/variants/retune/measure_retune.py --mode sweep --regimes mixed_64k --json /tmp/retune_mixed_64k.json" "python3 /workspace/kernel_work/streamindex_topk/variants/retune/measure_retune.py --mode correct --regimes decode_prod,decode_64k,mixed_prod --tuples 4,2,4 && python3 /workspace/kernel_work/streamindex_topk/variants/retune/measure_retune.py --mode control --regimes mixed_prod --tuples 4,2,4"
```

`RUN_CMD` as read back from `cdk job recipe j-ce9a19a4-4639-4a8f-9059 --no-color`, reflowed from
the recipe printer's YAML folding to one line; no other change.

`--map-dir /home/annyan_google_com/workspaces/vllm-torchtpu/kernel_work:tpu-job:/workspace/kernel_work`
and `--map-dir <repo>/.claude/skills/run-command-in-cdk/fanout:tpu-job:/root/fanout`.

**Provenance.** `vllm-torchtpu` @ `11f26d7554bb8c86028e556bd87f4a3790b567e0`, branch
`annyan/streamidx-scores`. Working tree at submit: `?? kernel_work/streamindex_topk/variants/` (this
folder, then untracked) and `?? optimization_work/`; nothing else modified — in particular
`instruments/` and `src/` are clean, so the code that ran is the commit's. `~/cloud-devkit` @
`4c6383fbd1578d56afa876f56904535715155cd2`, dirty only with
`?? recipes/experimental/gdn-comparison.yaml`, which this recipe does not use. Submitted
2026-09-06 02:40 UTC.

Note that `retune.patch` was **not** applied for the measurement: the variant is a static argument,
and `measure_retune.py` reaches it by rebinding `shapes.NUM_KV_PAGES_PER_BLOCK`, which is what lets
both arms run in one process against one unmodified instrument. The patch is the same value at the
production call site and is what would be landed.

---

## 9. Where this stands

- **`(4,2,4)` is 7.73% faster at `decode_prod` and 10.94% faster at `decode_64k`, and 2.63% and
  2.02% slower at `mixed_prod` and `mixed_64k`** [all `measured (j-ce9a19a4-4639-4a8f-9059)`,
  pooled per §3.1], against an incumbent measured in the same job by the same instrument.
- **Correctness verified at the shipped tuple**, bit-identical to the incumbent on all six margins,
  with the negative control firing 1024/1024 [`measured (j-ce9a19a4-4639-4a8f-9059)`].
- **Land `(4,2,4)`, not `(4,4,4)`.** They measure identically; the asymmetric spelling is the one
  that stays correct when the tuple stops being uniform.
- **The mixed cost is a real ~2%, not noise.** This corrects `bkv_p_RESULT.md` and it must be
  carried into the composed candidate.
- **No rank is claimed.** E's prize is decode-only; the decode:mixed step mix at the A/B shape
  decides whether the net is positive, and it is not measured here.
- **E's ceiling is Option B.** B's masked partial-block store decouples the width from the tuple,
  which is exactly the term that makes E cost mixed anything. Measure E composed with B, not summed.

### Open, each one shard

| # | what | why it is not here |
| ---: | --- | --- |
| 1 | `(8,2,8)` timing **and** correctness | 8 wins by 2.5 points at `decode_64k` and is tied at `decode_prod`; never value-checked at any regime. E takes 4. |
| 2 | a `decode_64k` drift control | the repeat arm came back at 111.72% spread and was discarded (§7 caveat 5). |
| 3 | `mixed_32k` | dropped for shard budget (§7 caveat 8). |
| 4 | `early_exit=True` at `(4,2,4)` | production's setting; unmeasured at this tuple (§7 caveat 4). |
