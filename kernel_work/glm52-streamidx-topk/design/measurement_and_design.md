# Wave 4 — measurement and design

`streamidx_topk_jax` on GLM-5.2. Two regimes, designed separately. This document ranks; it does
not build. Step 3 builds what §11 ranks and measures it with §13's protocol.

---

## 0. Conventions and terms, binding on everything below

**Provenance.** Four tags, no others. `measured (job id)` — a number a named job printed.
`modelled` — arithmetic from measured inputs, stated. `assumed` — believed, with a consequence if
wrong. `unmodelled` — nothing was measured and no arithmetic exists; a term valued at zero because
nothing measured it is `unmodelled`, never `0`.

**Sign.** Every percentage is a change in **time**. Negative is faster. Signed percentages appear
only in a `Δ time %` column; prose says the direction in words.

**Numbers not measured this exercise.** A speedup carries its baseline, hardware, shapes, precision
and measurer or it is not a number. Where a figure comes from another exercise or another kernel,
its transfer to this block is tagged `assumed` at the point of use.

**Terms.**

| term | definition |
| --- | --- |
| **the block** | `streamidx_topk_jax`: the three TensorCore scoring `pallas_call`s in `streamindex_topk.py`, the XLA `-inf` fill and bitcast between them, and the `sparsecore_topk` selection. One HLO region; one number in `bench.py`. |
| **(A)** | the 8k1k deployment. `max_model_len` 9216, `page_size` 1024, `pages_per_seq` 9, **`n` = 10240**. |
| **(B)** | the AgentX deployment. 256k–1M context, `page_size` 512, `pages_per_seq` 2048, **`n` = 1048576**. |
| **`n`** | the **padded score extent**, `align_to(pages_per_seq, bkv_p) · page_size` (`streamindex_topk.py:602-605`). Not `pages_per_seq · page_size`. At (A) that is 10240, not 9216: the top 1024 score columns are padding no request can reach. `shapes.padded_n()` derives it. |
| **`b`** | rows handed to the selector = `T`, the padded token count. |
| **`p`** | SparseCore column partition; `_pick_partition(b,n,k)`, bounded by `MAX_SLICE_WORDS = 65536`. |
| **slice** | one `(row, column-partition)` unit of SparseCore work. There are `b·p` of them. |
| **wave** | 32 slices, run concurrently on 2 cores × 16 subcores. `num_waves = cdiv(b·p, 32)`. |
| **`X_crit`** | `Σ_waves max_row_walked_words` — the **sum over waves of the longest walked slice in that wave**. Not the sum of walked words. This distinction is the whole of §3. |
| **`f_w`** | SparseCore walk rate, ns per walked word. |
| **`C_sc`** | the length-independent part of `sparsecore_topk`'s measured time. |
| **`H`** | the microbenchmark harness's per-call floor (§4). A constant added to every absolute number in the corpus and to none of the paired deltas. |
| **block floor** | `roofline.py`'s `floor us` column. **The mandatory denominator of every `frac` column below, the same number for every option, without exception.** Definition: `max(t_mem_after_both_levers, t_mxu)`, i.e. HBM bytes with the KV row-padding removed and the score matrix kept out of HBM, over `mem_bw = 3.70e12 B/s`; MXU at fp8 `2.30e15 F/s` / bf16 `1.155e15 F/s`; hardware constants `measured (j-1054e9b4)` via `pltpu.get_tpu_info()`. |
| **composed floor** | the block floor **plus** the two serial terms the roofline has no vocabulary for: the SparseCore scan `f_w · X_crit_min`, and one irreducible Pallas launch. Defined and justified in §10. It appears **beside** the block-floor fraction, never instead of it. |

**Column meanings in the option table (§9).** `areas` — which of the seven the option decides
differently from the incumbent. `Δ time %` — predicted change against the *measured* baseline at
that shape, banded where inputs are ranges. `frac` — predicted `block floor / predicted time`.
`frac*` — the same against the composed floor. `prov` — the weakest tag among the option's leading
terms.

---

## 0.1 Step-3 corrections — read before using any number below

Six terms in this document have been **measured** since it was written, and five of them came back
different from what was predicted. The body below is left as written, deliberately: a design record
is worth more with its predictions intact than silently retconned. **Where this section and the
body disagree, this section wins.** Each item names the job that settled it.

**1. `H` is the harness — §11's first branch, not its second.** `measured (j-1fe72c72-f53a-4ffd-9d14)`.
A `jax.jit(lambda x: x+1)` on a *scalar* costs 152.0 µs through `bench.timeit`. An N-sweep over
N ∈ {1,2,5,10,25,50,100} fits `total(N) = 29.89·N + 143.5 µs` (R² = 0.99966) for the null arm and
`29.76·N + 185.0 µs` (R² = 0.99952) for a SparseCore arm — the two **slopes agree to 0.4%**, and the
30 µs slope is host dispatch, since the null arm with no device work produces it too. So §11's
harness branch is the live one: `report.md`'s (A) table overstates the block, O4 is a wash at (A),
and unpriced item 12 fires against every (A) absolute in this exercise.

Two qualifications, because the correction is easy to over-apply:

- **It is additive (~144 µs), not a ~40% multiplier.** §11 wrote "overstates by ~40%" because 40%
  is what it happens to be at one (A) arm. It is ~85% of a 170 µs arm and ~1% of a 16 ms arm.
- **The 143.5 µs intercept is *that job's* harness term, not a portable constant.** The same
  `sc_tiny` arm read 177.3 µs there against 140.1 µs in `j-69ee326d` — a 27% cross-job gap, larger
  than the 3–9% run-order drift the corpus documents. Subtracting 143.5 from a corpus number taken
  in a *different* job is inference, not measurement. The portable statement is the ratio: **~85% of
  a small arm's reported time is harness.** The residual real device cost of `sc_tiny` resolves only
  as a within-job paired difference against `null`: **+16.0%, i.e. ~25 µs ± ~10** (positive 5/5
  repeats, but spread 14.3pt against a 16.0% effect).

**2. The 26.2 µs per-launch constant is refuted.** `measured (j-dc5581dd, j-c462ca4b)`. §15(a)
listed this as a MANDATORY PROBE; O1's measurement ran it. Two empty launches are worth about
**18 µs at (B)** (resolved at B-D4-256k, n = 1M) and **nothing resolvable at (A)** (n = 10240) —
roughly 9 µs each where measurable, a third of the assumed figure, and it **scales with the padded
score extent rather than being a per-launch constant**. Everything costed with it is over-credited:
§7's `3 × 26.2 = 78.6 µs` row, §8 item 12's `1.65 ms per decode step`, §10's `F_comp` `+26.2` term
and the whole floor table, and O1's (A) entry in §9.

**§15(a)'s stated consequence is itself now wrong, and this is the correction that matters.** It
wrote that striking this term *"resolves the (A) tie to O4 under both `H` hypotheses."* That
inference was drawn before `H` was known. With `H` = harness, striking the launch constant removes
value from **both** sides — from O1's launch-skipping switch *and* from O4's entire rationale, which
was removing launches. What survives at (A) is O1's **growth guard**, which was never a launch term:
`measured (j-c462ca4b)` **−16.0% at A-D16-8k**, spread 4.5pt, with byte-identical selected sets. At
A-D64-8k O1 reads −0.2% against a 5.4pt spread — UNRESOLVED, landing exactly on §15(a)'s predicted
collapse. **The (A) resolution is O1, and O4 is not being built.**

**3. P-ITER's 0.500 µs/iteration is a decode-remainder-path constant, not a global one.**
`measured (j-1672346b, j-980e8710)`. It transfers exactly at B-D1-1M (`D=1`, so everything runs on
the `seq_batch_size=1` launch P-ITER actually measured: predicted −31.4%, measured **−33.0%**), is
~25% over on the `sbs=4` batched decode path (B-D4-256k: predicted −10.0%, measured −7.5%), and is
**~3× over** on the MIXED path (B-P1024-256k: predicted −2.0%, measured −0.7%; a separate two-arm
comparison bounds it at ≤0.06 µs/iteration there). The transfer across launch paths was
`unmodelled`, never confirmed. Do not cost an option with it off the decode-remainder path.

**4. The selector is fixed-cost-bound, not walked-word-bound, at these shapes.**
`measured (j-b58eeb97, j-dc832521)`. This contradicts M2's premise. O7 widened level 0 so that
carry-forward would discard ~2.6 full slice walks it previously could not — **it bought nothing**
(1.7pt *slower*). The only change that won touches no walked words at all and only cuts fixed
per-wave iterations (~1344 → ~320): **−3.4% at B-D4-256k, −4.0% at B-D1-1M**, reproduced across two
jobs. Independently, O3's two `p 16→4` transitions imply combined rates of 0.830 and 1.227
ns/walked word — a 1.5× spread across shapes making the *same* transition, with B-D4-256k landing
**below** M2's `f_w` lower bound even after attributing its entire saving to the walk.
**M2's `unmodelled` wave-count dependence of `C_sc` is material** — worth roughly 10 ms of the
30.9 ms O3 saved at B-P1024-256k. Refit M2 against these points before ranking anything else with
it.

**5. A `lax.switch` around a large array costs one full copy of it.** `measured (j-fad1a0f8,
j-3cab9e02)`, at ~1.30 TB/s effective and at two operand sizes 64× apart (51.0 µs on 64 MiB,
3310.2 µs on 4 GiB — the constant *rate* across a 64× ratio is what identifies it as a copy rather
than a fixed conditional overhead). Cause: an in-branch `bitcast → reshape(-1)` forces XLA to
materialise the operand inside the conditional region. Hoisting both above the switch removes it
entirely — after the hoist the switch is free on all five shapes. Any option wrapping the score
array in a conditional pays this, and **it is invisible unless a no-switch arm is timed beside it.**

**6. VMEM is 64.0 MiB, and §7's accounting is wrong in both directions.**
`measured (j-fb58ac5a)` via `pltpu.get_tpu_info().vmem_capacity_bytes`; XLA reports 63.94M usable.
The ~31 MiB scoped budget this exercise sized options against is **wrong by 2×** — and the fault is
this exercise's, not the reference's. `references/02-partitioning.md` states 31 MiB as the
*default* scoped limit and says in the same breath that `vmem_limit_bytes` moves it "up to roughly
64 MiB"; `references/hardware-facts.md` calls it a knob outright. The instruments pass 100 MB, so
the knob was already raised and the default never bound. A config declaring **32.53 MiB** of
scratch compiled and ran. In the other
direction, §7 counts only the four declared double buffers and **omits the two f32 einsum
intermediates**, which are 16.0 MiB at the (B) MIXED tile against the 2.53 MiB §7 prints. The temps
are not always materialised — Mosaic tiles them on the MIXED path (a tile priced at 289.25 MiB
compiles and runs) but not on the decode path (a config declaring 65.03 MiB was refused at 85.00M,
within 4% of declared + temps). Screen on declared scratch; report the temps; do not assume either.

---

## 1. Measurement record

### 1.1 Baselines and floors

Medians of the first of two 20-iteration blocks, `measured (j-59d0b337 / j-470d481e via report.md
§3)`. Floors from `roofline.py`, re-run this session.

| shape | meas µs | `sc` µs | sc share | block floor µs | floor/meas |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-D64-8k | 363.2 | 189.5 | 52% | 18.5 | 5.1% |
| A-D32-8k | 264.8 | 175.6 | 66% | 9.2 | 3.5% |
| A-D16-8k | 278.7 | 200.5 | 72% | 4.6 | 1.7% |
| A-P512-8k | 579.4 | 352.1 | 61% | 14.9 | 2.6% |
| A-P1024-8k | 841.6 | 556.8 | 66% | 29.9 | 3.6% |
| B-D4-256k | 1117.9 | 795.9 | 71% | 36.6 | 3.3% |
| B-D4-1M | 1826.7 | 823.5 | 45% | 146.3 | 8.0% |
| B-D1-1M | 1481.9 | 780.6 | 53% | 36.6 | 2.5% |
| B-D16-256k | 2048.8 | 1012.8 | 49% | 146.3 | 7.1% |
| B-RAG16 | 5813.5 | — | — | — | — |
| B-P1024-256k | 50916.8 | 43633.9¹ | — | 956.1 | 1.9% |
| B-P1024-1M | 66958.7 | 40007.8¹ | — | 3824.4 | 5.7% |

¹ from `j-dcf1e707`, a different job and a synthetic score distribution; not comparable to the
first-column figure by subtraction.

**A-D64-8k's block-to-block spread is 9.1%** against 0.2–3.2% elsewhere and is unexplained
(`report.md` unpriced item 13). Any single-digit claim at that shape is not believable until step 3
re-measures it.

### 1.2 Probes, in five fields

Every `measured (job id)` tag below is checkable against these three jobs. Image on all three:
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.

| probe | job | question | answer | what it decided here |
| --- | --- | --- | --- | --- |
| **P-P16** | `j-cefba169-8b6e-4fc6-80f6` (Succeeded) | does `_pick_partition`'s growth branch cost time at b=16? | full op −13.7% (spread 5.5 pt, 5/5), SC alone −16.8% (spread 2.0 pt), selected sets byte-identical | O1's largest single term at (A) decode, and the only *measured* full-op option delta at (A). Caveat: the SC-alone arms ran at n=9216 (`shapes.py:177`, since fixed); the full-op arms are unaffected. |
| **P-DECOMP** | `j-dcf1e707-14ff-40d0-b27b` (Succeeded) | where does the 16.2 ms B-P1024 1M-vs-256k gap live? | gap +31.8% real; the SC is **8.4% faster** at 1M; DiD puts **19835.6 µs outside `sparsecore_topk`** | kills the bandwidth story; the gap is TC scoring FLOPs. §2.3 closes it. |
| **P-WALK** | `j-69ee326d-e045-41fe-824a` shard0 (rc=0) | does SC time depend on walked words at fixed padded extent? | level A: −52.3% as walk falls 8× at fixed padding; **dead-half −3.4%, i.e. unresolvable**. Level B (p=16): shorter rows are **+20.6%** and **+27.9%** slower | M1 refuted; **M2** established; the driver is the per-wave maximum. `f_w = 0.895 ns/walked word`; `C_cfg = 325.2 µs at 8 waves`, **not portable**. |
| **P-SCBW** | `j-69ee326d` shard1 (rc=0) | what is the SC's achieved read bandwidth? | four DMA arms 140.1/139.6/141.0/146.1 µs over a 16× byte range — individually unresolvable. Slope → 1212.5 GB/s, ≥591 GB/s worst repeat. **FULLWALK control: +75.9 µs for 65536 words = 1.158 ns/walked word** | bytes are not a first-order SC cost; walked words are. Also supplies the 1-wave intercept **138.6 µs**, which §4 turns on. |
| **P-ITER** | `j-69ee326d` shard2 (rc=0) | is the in-body fixed cost per iteration, per copy, or per index computation? | **0.5002 µs per iteration**, linear to ±0.8% on two unfitted arms; per-copy refuted; correctness held | O2. Confirms area 3's 512 ns grid step to 2.3%. |
| **P-GATE** | — | approximation quality | **excluded by the caller** | O5 therefore carries no measured correctness evidence and needs sign-off (§12). |

**Two arms were not run and `probes.md` says so.** a1 arm 4 (coalesced strided copy) and a1 arm 5
(host byte-offset table). Consequence, carried forward as a defect rather than hidden: **area 7's
index-arithmetic hypothesis is neither confirmed nor refuted.** P-ITER predicts a flat table under
it and could never have discriminated it, and **an additional per-copy term on top of the 0.500
µs/iteration is not excluded.** Seam candidate S4 rests entirely on that term and is therefore
`unranked` in §9, not ranked low.

### 1.3 The four things wave 3 overturned

Design against these, not against the earlier sheets, which still carry the refuted wording.

1. **M1 (`T = C + f·X_pad`) is refuted as a length model.** Replacement is M2 (§3). Any option
   whose saving is "we walk fewer words" must say whether it shortens **the longest row in a wave**.
2. **E3 / block-max length pruning is negative where proposed** — `+20.6%` at b=16 and `+9.1%` at
   b=1024, two jobs. Pruning pays only where it repacks; it is inseparable from a packing decision.
3. **a0.5's closure is dead and its premise was false.** `sparsecore_topk.py:160` copies the full
   slice regardless of length; there were never 3 GiB to save, and the SC contributes to the
   B-P1024 gap with the wrong sign.
4. **The A-D16/A-D32 inversion is established and fully caused** by `_pick_partition`'s second loop,
   which only ever raises `p` and never checks that raising it reduces the wave count. Measured
   −13.7% full op to suppress it. The earlier −9.0% was modelled and lies outside the measured range.

---

## 2. What the incumbent decided, area by area

| area | the incumbent's decision | where |
| --- | --- | --- |
| **(1) fusion boundary** | Scoring and selection are **separate HLO ops** chained by a `f32[T,n]` HBM buffer, pre-filled with `-inf`, input-output-aliased, bitcast to `i32` between them. At (B) that buffer is 64 MB (decode) to 4 GiB (prefill). | `streamindex_topk.py:745+`, `sparsecore_topk.py:395` |
| **(2) partitioning** | TC: grid `{seq group g} × {query block bq} × {KV block bkv}`; `g` is the only cut axis; `h=32` and `d=128` live inside one einsum. SC: `b·p` slices over 32 subcores, `p` from **static `n`**, wave map `row = w·32 + flat` fixed row-major. `num_cores=1`, so a `parallel` TC grid axis has nothing to split. | `streamindex_topk.py:276`, `sparsecore_topk.py:360-377` |
| **(3) pipelining** | Hand-rolled double-buffered `make_async_copy` on a `4×2×sbs` semaphore array; `bkv_p = 2` everywhere; depth 2. The scoring loop measures at 1.00–1.09× of the `512 ns + bytes/3.48e12` stepped floor — it is *at* its floor, and the lever is the number of steps, not their efficiency. | area 3 sheet |
| **(4) layout / precision** | fp8_e4m3fn q and k with an **e8m0 scale byte packed inline**, so the cache row is 256 B of which **129 are used**; f32 accumulation, f32 score matrix, f32 head-weight vector on 32 of 128 lanes. | `streamindex_topk.py:265-268` |
| **(5) compute units** | Scoring on the TensorCore, selection on the SparseCore, strictly serial. `num_cores=1`. Lock-free by construction — no `subcore_barrier` anywhere. | both files |
| **(6) algorithm** | Exact 4-level 8-bit MSB radix select on the monotone i32 key, **re-scan** rather than compaction at every level, tie quota by `plsc.cumsum` + `store_compressed`. Two-stage when `p > 1`, with two `jnp.take_along_axis` gathers. | `sparsecore_topk.py:64-300` |
| **(7) precomputation** | Three launches per layer, all with **traced** grid extents; `-inf` fill re-materialized per call; per-grid-step page metadata computed in-kernel from `page_indices` in SMEM; histogram VMEM re-zeroed 4× per wave. | area 7 sheet |

---

## 3. The cost model the design ranks against

`report.md`'s unpriced item 11 asked wave 2 to build one, because the largest component of the op
had none. Here it is, refined from M2 and validated against the corpus.

### 3.1 The SparseCore: M2, with a band on both coefficients

```
T_sc  =  C_sc(config)  +  f_w · X_crit          X_crit = Σ_waves max_row_walked_words
```

- **`f_w` ∈ [0.895, 1.25] ns per walked word.** 0.895 `measured (j-69ee326d)` from P-WALK level A's
  endpoints; 1.158 `measured (j-69ee326d)` from P-SCBW's FULLWALK control, a different arm set and a
  different wave count; 1.249 `modelled` from a regression I ran here across the (A) family
  (A-P512-8k 352.1 and A-P1024-8k 556.8, `X_crit` 163840 and 327680). Three independent estimates
  bracketing M1's fitted 1.2237. **Only the slope is portable; the intercept is not** — `probes.md`
  says so and the fits below agree.
- **`C_sc` ∈ [138.6, 325.2] µs `measured`**, wave-count dependence `unmodelled`. The (A)-family
  regression gives an intercept of **147.5 µs**, flat from 1 to 32 waves.

**Validation against the corpus** (`f_w = 1.249`, `C_sc = 147.5`, both from the (A)-family fit; the
(B) rows are pure prediction):

| shape | waves | slice len | `X_crit` | predicted µs | measured `sc` µs | error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-D32-8k | 1 | 10240 | 10240 | 160.3 | 175.6 | −8.7% |
| A-D64-8k | 2 | 10240 | 20480 | 173.1 | 189.5 | −8.7% |
| A-P512-8k | 16 | 10240 | 163840 | 352.1 | 352.1 | fitted |
| A-P1024-8k | 32 | 10240 | 327680 | 556.8 | 556.8 | fitted |
| B-D4-256k | 8 | 65536 | 524288 | 802.4 | 795.9 | **+0.8%** |
| B-D1-1M | 8 | 65536 | 524288 | 802.4 | 780.6 | +2.8% |
| B-D4-1M | 8 | 65536 | 524288 | 802.4 | 823.5 | −2.6% |
| B-D4-512k | 8 | 65536 | 524288 | 802.4 | 832.1 | −3.6% |
| B-P1024-256k | 512 | 65536 | 33554432 | 42061 | 43633.9 | −3.6% |

Nine shapes, two free parameters fitted on two of them, worst error 8.7% and the two extremes
(1 wave and 512 waves, a 3200× range in `X_crit`) both inside 4%. **This is the cost model the block
did not have.** Unpriced item 11 is closed to that accuracy.

Three consequences, and they are the design:

- **`X_crit` is length-independent as the kernel is written.** Slices are always walked to
  `min(slice_len, eff_len)` and, at `p = 16`, a 256k-live row still contains four full-length live
  slices, so every wave's maximum is 65536 whatever the context length. B-D4-256k, B-D4-512k and
  B-D4-1M measure within 4.6% of each other and the model predicts one number for all three. **The
  only way to move `X_crit` is to change the number of *waves*, i.e. `p` or the wave assignment.**
  That is a partitioning decision, not a pruning one — which is exactly why E3 lost and O3 does not.
- **At (B) the walk is the op.** `f_w · X_crit` = 655 µs of a 1117.9 µs block at B-D4-256k, and
  30031–41943 µs of a 50917 µs block at B-P1024-256k.
- **At (A) the walk is nothing.** 12.8–25.6 µs of a 264.8–363.2 µs block. **At (A) the entire
  SparseCore cost is the 147.5 µs intercept**, which is 5–32× the whole block floor. §4 is about
  what that intercept is.

### 3.2 The TensorCore: the stepped-grid model, validated

```
T_tc  =  n_launch · 26.2 µs  +  n_step · (0.500 µs + bytes_per_step / 3.48e12)
```

`0.500 µs/iteration measured (j-69ee326d)`. `26.2 µs/launch measured (j-0a0a72c8)` **on a different
kernel — transfer to this block is `assumed`**, per area 7's own tag.

| shape | launches | steps | bytes/step | predicted TC µs | meas − `sc` µs¹ | error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-D64-8k | 3 | 80 | 2.10 MB | 78.6 + 88.2 = **166.8** | 173.7 | −4.0% |
| B-D4-256k | 3 | 256 | 1.05 MB | 78.6 + 205.3 = **283.9** (+17 fill) = 300.9 | 322.0 | −6.6% |

¹ **Not a paired delta and not a result.** `full − sc` is banned as a measurement (`probes.md`;
it moved 2.6× across three measurements of one shape). It is used here only as a cross-check on an
independently constructed model, and the model is what is quoted.

**Both halves of the block are now accounted for to under 9%.** The op is understood.

### 3.3 The B-P1024 gap, closed

`probes.md` left the 19835.6 µs difference-in-differences as "the leading explanation is scoring
FLOPs… the MXU peak itself is `unmodelled` — wave 4 should get the peak from area 5 before leaning
on it." The peak is in `report.md`'s hardware constants, `measured (j-1054e9b4)`: bf16
`1.155e15 F/s`, fp8 `2.30e15 F/s` per TensorCore.

Extra keys at 1M vs 256k: 786432 × 1024 tokens × 32 heads × 128 dims × 2 = **6.60e12 FLOP**
`modelled`. Over the 16209.5 µs full-op gap that is 407 TFLOP/s = **35.2% of the measured bf16 peak**
(17.7% of fp8). An ordinary MXU efficiency for a K=128 contraction with an f32 accumulator.

**The gap is scoring arithmetic at an unremarkable MXU efficiency.** It is not an anomaly, it is not
a bug, and there is nothing structural to recover from it — which is itself the useful result,
because it was the largest single number in the exercise and it now ranks no option. `probes.md`'s
first open item is closed.

---

## 4. `H` — the term that decides regime (A), and it may not be a kernel cost at all

**The observation.** P-SCBW's arm `n=4096, row_lengths=16` measures **140.1 µs
`measured (j-69ee326d)`** for a SparseCore kernel that DMAs 16 KiB and walks **one chunk of sixteen
words**. Model that arm: the DMA is 0.03 µs at the same probe's ≥591 GB/s bound; the walk is 0.014
µs at `f_w`; the length-independent in-kernel loops are 1344 iterations (`zero_priv` 256 × 4 levels,
`fold_priv` 16 × 4, `scan_glob` 16 × 4, `find_bucket` 16 × 4, `fill_body` 128 — counted from
`sparsecore_topk.py:108-300`) and cannot exceed the walk rate per iteration, giving **≤ 19 µs**.
**Total modelled ≤ 20 µs against 140.1 µs measured.** The residual ≥ 120 µs is `unmodelled`.

**The candidate explanation.** `instruments/bench.py:39-42` times
`t0 = perf_counter(); out = fn(); jax.block_until_ready(out)` — **one host dispatch and one
host↔device synchronisation per iteration**, inside the timed region. Every absolute figure in this
exercise carries exactly one of those. Corroborating: no arm anywhere in the corpus measures below
**139.6 µs**; the four P-SCBW DMA arms are flat across a 16× byte range at 139.6–146.1; and the (A)
SC-alone arms cluster at 168.6–203.3 across a 4× spread in work.

**The competing explanation** is that a SparseCore `pl.kernel` genuinely costs ~140 µs to dispatch
and load. Both explanations produce the same ~140 µs constant. They produce **opposite designs**:

| | `H` is harness dispatch | `H` is SparseCore launch |
| --- | --- | --- |
| A-D64-8k device time | ~233 µs | 363.2 µs |
| `sparsecore_topk` device time at (A) | ~60 µs | ~190 µs |
| per decode step, 21 layers | H does not exist in the served graph | **2.9 ms of pure SC launch** |
| the (A) lever | the three TC launches (78.6 µs, 34% of device time) → **O1** | getting off the SparseCore → **O4** |
| `report.md`'s (A) table | overstates the op by ~40% — unpriced item 12 fires | stands |

**What `H` does not touch.** It is a constant. It cancels in every paired difference, which is why
`f_w`, the 0.500 µs/iteration, the −13.7% growth-branch delta and the −31.4% `bkv_p` delta are all
safe, and why `probes.md` was right to say only `f_w` should be reused. `H` moves **denominators**,
never **deltas**. At (B) decode it is ≤12% of the baseline and at (B) prefill ≤0.3%: **it does not
touch the (B) ranking at all.** At (A) it is 38–61% of every baseline: **it is the (A) ranking.**

**It is the cheapest measurement in the exercise and step 3 gets it for free.** Add two arms to the
variant's own harness — a null arm (`jax.jit(lambda x: x + 1)` on a scalar) and an amortized arm
(N calls inside one `block_until_ready`) — to the `interleaved()` list that step 3 runs anyway. No
extra job, no extra slice. It is listed as step 3's first act in §13 and it is the assumption named
in §11.

I did not queue a job for it: the caller's brief prefers a recorded `assumed` over a queued job
while the cluster is throwing `START_SESSION` faults, and this one costs nothing when folded into a
run step 3 must make regardless.

---

## 5. The iteration space, per regime

Every axis, including the ones nobody tiles. The table in §9 is held against this list in §9.4.

### 5.1 Regime (A) — `n` = 10240, `page_size` 1024, 9 live pages

| # | axis | extent at (A) | cut today? | reason it is or is not |
| --- | --- | --- | --- | --- |
| 1 | seq group `g` | 16 (64 seqs / sbs 4) | **yes**, grid leading | the only TC parallel axis; `num_cores=1` so nothing else can be. |
| 2 | query block `bq` | **1** at decode, 8 at prefill | degenerate at decode | **degenerate axis**: `bq_sz=1`, so the whole `bq_sz` knob is inert in the deployment's dominant phase. Every tuning note about `num_queries_per_block` is prefill-only at (A). |
| 3 | KV block `bkv` | 5 (10 pages / `bkv_p` 2) | yes | only 5 steps deep — **the loop is too short for P-ITER's lever to matter here**, and widening it is *negative*: `align_to(9,16)=16` pages reads 78% more KV and widens `n` 60% into a SparseCore driven by that extent. Quantified in §9.2. |
| 4 | head `h` | 32 | **no** | inside the einsum's M dim. Splitting it would need a cross-block reduction of the largest array in the op. |
| 5 | head_dim `d` | 128 | **no** | the einsum's contraction dim and exactly one lane width = one MXU K tile. |
| 6 | score column `s` within a block | 2048 | **no** on TC | the einsum's N dim, materialised whole. |
| 7 | SC slice / **wave** | `cdiv(b·1, 32)` = 1–32 | yes, but **not chosen** | the wave map is a fixed row-major `row = w·32 + flat`. **This is the axis M2 is a function of and nobody selects deliberately.** |
| 8 | SC column partition `p` | **1** | degenerate — **and a bug when it is not** | at b=16 `_pick_partition` raises `p` to 2 without reducing the wave count. −13.7% to suppress, `measured (j-cefba169)`. |
| 9 | radix level `ℓ` | 4 levels × 8 bits | iterated in-kernel, **never as a config** | **the axis that lives in the code but not in any knob.** Digit width, level count and re-scan-vs-carry-forward are all hardcoded. tokamax picks 4 bits × 8 levels with carry-forward. O7. |
| 10 | histogram bucket `b` | 256, four times per wave | iterated, **never varied** | **the axis that lives in the data but not in anyone's loop.** 1024 of the 1344 length-independent iterations per wave are `zero_priv`. At (A) this axis is plausibly most of `C_sc`; nobody has swept it. |
| 11 | pass over the resident slice | up to 6 (`cvt`, `hist0`, 3×`ref_body`, `emit`) | **no** | **the axis that only appears once you write M2** — `f_w` aggregates all of them, so no sheet before M2 could see it. E5/E7 fuse pairs. |
| 12 | indexer layer `L` | 21 | outside the block | **the axis that only appears once the regimes are separated**: 3 launches × 26.2 µs × 21 = **1.65 ms per decode step**, of which 1.10 ms is launches with a zero-extent grid. |
| 13 | request class (`decode` / `decode-remainder` / `mixed`) | 3 static launches | **no** | **the degenerate axis with teeth**: 42 of 63 launches per step run an empty grid and all 63 pay. A runtime `lax.cond` is the whole of the remedy. |
| 14 | model step | — | n/a | the staleness axis for any precomputed table (area 7): a `-inf` buffer keyed on nothing is fatal across steps; bucketing `n` below the live max fails **silently** in the shrinking direction. |

### 5.2 Regime (B) — `n` = 1048576, `page_size` 512, 2048 pages

| # | axis | extent at (B) | cut today? | reason it is or is not — **only where it differs from (A)** |
| --- | --- | --- | --- | --- |
| 1 | seq group `g` | 1–4 | yes | far fewer sequences; the group max over `sbs=4` is where B-RAG16's +184% lives. |
| 2 | query block `bq` | 1 / 8 | degenerate at decode | as (A). |
| 3 | KV block `bkv` | **1024** at 1M, `bkv_p`=2 | yes | **the (B) lever, and the largest measured one**: 1024 × 0.500 µs = 512.2 µs = **35.8% of B-D1-1M**. `align_to` is a no-op at 2048 pages for every `bkv_p`, so the (A) penalty does not exist here. O2. |
| 7 | SC slice / **wave** | 8 at decode, **512** at prefill | yes, but not chosen | at (B) this axis *is* the op: 469–41943 µs. |
| 8 | SC column partition `p` | **16** | forced | `p_min = cdiv(1048576, 65536) = 16` is a **capacity constraint, not a tuning choice** — `MAX_SLICE_WORDS`. It is also the axis that makes 12 of 16 slices dead at 256k and pays stage-2 degeneracy for them. O3. |
| 9 | radix level `ℓ` | as (A) | | with `X_crit` 25× larger, the per-pass cost matters 25× more. tokamax's windowing removes the `MAX_SLICE_WORDS` cap entirely. |
| 15 | **live context length** | 256k … 1M | **not an axis anywhere** | **the axis that only exists once (B) is separated from (A).** At (A) every request is within 12% of `max_model_len`; at (B) the live length spans 4× and the batch is ragged. `_pick_partition` never sees it, `slice_len` never sees it, and the DMA at `:160` copies past it. Three of the five options are about this axis and it appears in no config field today. |
| 16 | selection rate `k/eff_len` | **0.2%** at 1M, 0.8% at 256k | — | vs 22–25% at (A). This single number is why candidate pruning is a (B) algorithm and does literally nothing at (A) — §9.2, O5. |

---

## 6. What is available to read: existing implementations

Surveyed this session. Rejecting one is legitimate; not looking is not.

| # | design | where | exact? | decoded into the seven areas | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | **tokamax distributed MSB radix select** | `openxla/tokamax`, `tokamax/_src/ops/experimental/tpu/topk/pallas_mosaic_tpu_kernel.py` (813 lines) | exact, unsorted, stable-optional | (2) *sequential windows with survivor carry-forward*, so VMEM ⊥ n and there is **no `MAX_SLICE_WORDS` cap**; (3) `num_seq_windows` is the pipeline knob; (5) **cooperating subcores with `pltpu.VMEM_SHARED` + `plsc.subcore_barrier()`**, no stage 2 and no gathers; (6) **4-bit digits × 8 levels** with compaction instead of re-scan; (7) `plsc.scan_count` pre-aggregation instead of 16 lane-private histograms | **O6 — `unranked`.** Its `subcore_barrier` over 16 cooperating subcores is blocked: **the barrier completes at 2 and 4 subcores and hangs the device at 8**, `measured (j-0222bc6d-c03a-4d93-97a7)`. Non-cooperating mode requires `n ≤ 4096`, hardcoded. Its two portable ideas are extracted as O7. **No performance number exists in the repo** — reported as an absence, not quoted. |
| 2 | **paired score+index emit** | `origin/hning/sctopk-pairs-e2e`, commit `bded1089` | exact | (1) elides **both** `jnp.take_along_axis` gathers at the stage-1→stage-2 seam via `emit_keys`/`payload_stage2` | **already written, and it composes with O3/O5**, which both keep `p > 1`. Folded into O3's file plan. |
| 3 | **`jax.lax.approx_max_k`** | `jax/_src/lax/ann.py`; recall math in XLA `approx_topk_shape.cc` | approximate, tunable | (6) partial max-reduce over lane-tiled windows; "recall" is a **window-collision probability, not an error bound** | **O5.** The reduction table is a shape fact, obtained by calling `_jax.approx_top_k_reduction_output_size` directly: at k=2048, **n=10240 → no reduction at any recall**; n=1M → 4× at recall 0.99, 32× at 0.9. |
| 4 | **the code this block replaced** | commit `6142d5e7`, removed lines | exact (`recall_target=1.0` reduces by a factor of 1) | its author's comment: *"jax.lax.approx_max_k wins due to the HBM read/write tax… direct VMEM streaming will allow SC to beat TensorCore performance"* | **A claim, not a number** — no baseline, hardware, shape, precision or measurer. Not quotable. But it names O4's thesis, from the previous author, before the swap. |
| 5 | **blockwise two-stage argmax, TC-only** | `git show 8107ab35:src/vllm_torchtpu/kernels/kimi_k3/router_topk.py` | exact | (5) never leaves the TensorCore; (6) O(k) argmax rounds + one-hot reduction, *"Mosaic cannot dynamically gather across the seven 128-wide source vregs"* | **rejected at k=2048**: O(k) passes over 10240 is 2048 × 10240 per row. Kept as the negative control and as the source of O4's constraint that a dynamic cross-vreg gather does not exist. |
| 6 | **threshold binary search on float bits** | `layers/common/binary_search.py` (© T5X) | **mask only; ties return more than k** | (6) 32 predicate evaluations, each a full reduction; zero per-row scratch | O4's threshold stage is this. Its two disqualifiers — returns a mask, breaks on ties — are exactly O4's `unmodelled` compaction term. |
| 7 | **vLLM CuteDSL stable top-k** | `vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py` | exact, stable | (6) **64-bit key packing score‖~token_id**, 11-bit radix, 6 passes — **no tie logic at all** | primitives do not transfer (shared-memory atomics). The **key-packing idea does**: it would delete the `plsc.cumsum(tie)` quota carry from `emit_body` at the cost of 6 passes over 64 bits instead of 4 over 32. Carried, not ranked. |
| 8 | **DeepSeek-V3.2-Exp reference** | `inference/model.py` | exact | `index_score.topk(min(index_topk, end_pos))` — a full sort | the reference has no selection design at all; its one optimisation (clamp k to live length) is what `row_lengths` already does. |

---

## 7. The joint budget the options are checked against

From wave 2, not re-derived.

- **TC VMEM: 31 MiB scoped** (64 MiB physical). Scratch: `bkv = 2·sbs·bkv_p·psz·256`,
  `bq = 2·sbs·bq_sz·32·128`, `bo = 2·sbs·bq_sz·bkv_p·psz·4`. Incumbent (B) MIXED is 2.53 MiB.
  Named conflicts: **decode `bkv_p=16` + a score-resident boundary = 32.28 MiB, over**; triple
  buffering + a (1024,4) tile = 37.9 MiB, over; (1024,8) = 41.75 MiB, over.
- **SC VMEM: 480 KiB per subcore accepted** (not 512). B-P1024 stage 1 is **345.2 KiB with 134.8
  free — the binding row.** A wider L0 (+119 KiB) *or* a pruning plane (+32 KiB), **not both**,
  unless area 7's `len_vmem` sizing fix (−62 KiB) lands first.
- **SMEM: 1021.5 KiB addressable.** `page_indices` i32[64·2048] is already **512.5 KiB**. Any
  byte-offset table needs `uint16` narrowing, gated on `total_num_pages ≤ 65535` with a
  `NotImplementedError` above it.

**O2 at (B) uses the (1024,4) MIXED tile at 25.25 MiB, which fits and strictly dominates area 2's
own (512,4) proposal** (128 iterations, 1 KV pass). O2 and any score-resident boundary are
mutually exclusive at (B) by the first named conflict; they are not proposed together.

---

## 8. The unpriced register: why nothing reaches 50% of the block floor

Read §10 first if you want the verdict; this is its evidence. Terms the block floor omits **by
construction**, priced at two shapes.

| term | A-D64-8k µs | B-D4-256k µs | tag |
| --- | ---: | ---: | --- |
| block floor (HBM bytes + MXU) | 18.5 | 36.6 | `modelled` |
| **SparseCore serial scan** `f_w · X_crit` | 18.3–25.6 | 469–655 | rate `measured (j-69ee326d)` |
| **Pallas launch fixed** 3 × 26.2 | 78.6 | 78.6 | `measured (j-0a0a72c8)`, transfer `assumed` |
| **grid-step fixed** 0.500 µs × steps | 40.0 | 128.0 | `measured (j-69ee326d)` |
| **SparseCore configuration** `C_sc` | 130–171 | 150–325 | intercept `measured`; composition `unmodelled` |
| **harness dispatch** `H` | 0–130 | 0–130 | `unmodelled` (§4) |
| modelled total | 285–464 | 862–1345 | |
| **measured** | **363.2** | **1117.9** | |

The floor is not wrong about bytes. **It is silent about four serial fixed-cost terms that together
are 5–24× the byte floor**, and it has no vocabulary for the largest of them: the SparseCore is a
16-lane serial scan at 0.895–1.25 ns/word, i.e. ~36 G words/s aggregate = **~143 GB/s against the
TensorCore's 3700 GB/s**. An op that streams `n` words through the SparseCore has a floor **26×
above** the HBM roofline for that stream, and no amount of kernel work removes it while the
SparseCore does the scan.

**The composed floor**, used in §9's `frac*` column and nowhere else:

```
F_comp  =  block_floor  +  f_w · X_crit_min  +  1 × 26.2 µs
```

with `f_w = 0.895` (the favourable end) and `X_crit_min` the wave structure a length-aware partition
could reach.

| shape | block floor | `f_w·X_crit_min` | launch | **F_comp** |
| --- | ---: | ---: | ---: | ---: |
| A-D64-8k | 18.5 | 18.3 | 26.2 | **63.0** |
| A-D16-8k | 4.6 | 9.2 | 26.2 | **40.0** |
| B-D4-256k | 36.6 | 117.3 | 26.2 | **180.1** |
| B-D1-1M | 36.6 | 469.2 | 26.2 | **532.0** |
| B-P1024-256k | 956.1 | 7508 | 26.2 | **8490** |

---

## 9. The options

Nine rows: the incumbent, six ranked, two `unranked`. `frac` is against the **block floor** and is
the mandatory column; `frac*` is against `F_comp` and sits beside it.

### 9.1 What each option decides differently

| | option | areas | one sentence |
| --- | --- | --- | --- |
| **O0** | incumbent | — | reference. |
| **O1** | **Stop paying for nothing** | (1)(2)(7) | Delete the `-inf` fill; make `_pick_partition` refuse to raise `p` unless it lowers the wave count; skip the empty launches with a runtime `lax.cond`. Three located, independently measured or modelled fixed costs, no algorithm change. |
| **O2** | **Widen the KV loop, at (B) only** | (2)(3)(4) | `bkv_p` 2 → 16 at (B), MIXED tile (1024,4) at 25.25 MiB, `bkv_p` held at 2 at (A) *because* `align_to(9,16)` widens `n` 60%. |
| **O3** | **Pack the waves** | (2)(6)(7) | Choose `p` from the batch-max **live** length instead of static `n`, via a `lax.switch` over precompiled `p ∈ {1,2,4,8,16}` bucketed **upward**; the stage-2 candidate set then contains no dead slices. |
| **O4** | **One op; scores never reach HBM; select on the TensorCore** — **the from-nothing target** | (1)(2)(4)(5)(6)(7) | Designed from the inputs, the output `int32[T,2048]` and `n=10240` alone: at (A) a 64-token score tile is 2.6 MB and fits VMEM, so there is no reason for an HBM buffer, a second HLO op, a second compute unit or a `-inf` fill to exist. Threshold by bit-bisection, then compact. **Shares no boundary with any other option and has no attribution story.** |
| **O5** | **Approximate prune, then exact select** | (1)(4)(6) | `approx_max_k(recall_target=0.99)` cuts a 1M row to 262144 before the exact selector runs; `p` then falls to 4 and `X_crit` with it. **(B) only — at (A) the reduction factor is exactly 1.** |
| **O6** | **tokamax, as shipped** | (2)(3)(5)(6)(7) | Windowed cooperating-subcore radix select. **`unranked`** — §9.3. |
| **O7** | **Rewrite the SparseCore inner loop** | (3)(6)(7) | tokamax's two portable ideas without its barriers: survivor carry-forward via `store_compressed` after level 0, and `scan_count` pre-aggregation replacing the 16 lane-private histograms; plus fold-and-zero, which deletes 768 of the 1344 length-independent iterations per wave. |
| **S4** | **host byte-offset table** | (4)(7) | wave-2 seam candidate. **`unranked`** — §9.3. |

### 9.2 Predicted effect, per shape

All `Δ time %` are against the measured baseline at that shape. Bands come from banded inputs and
**propagate; they do not collapse to a point.**

**A-D64-8k** — measured 363.2, block floor 18.5, `F_comp` 63.0.

| opt | Δ time % | predicted µs | frac | frac* | leading term, and its tag |
| --- | ---: | ---: | ---: | ---: | --- |
| O1 | **−14.4%** | 310.8 | 6.0% | 20% | 2 empty launches × 26.2 µs `measured (j-0a0a72c8)`, transfer `assumed`. |
| O2 | **+1.7%** | 369 | 5.0% | 17% | −35 µs steps, **+30 µs KV overshoot** (`align_to(9,16)`=16 pages vs 9 live) **+11 µs walk** (n 10240→16384). `modelled`. **Not applied at (A)**; the row exists to show the arithmetic P-ITER left `unmodelled`. |
| O3 | 0 | 363.2 | 5.1% | 17% | `p` is already 1. |
| **O4** | **−61% … −26%** (H=0) / **−26% … +10%** (H=130) | 140–270 / 270–400 | 13.2%–6.9% | 45%–23% | removes `sc` (−189.5 `measured`, or −60 `modelled` under H=130), 2 launches (−52.4), the score round trip (−1.3); **adds TC selection, `unmodelled`, banded [20, 150] µs**. |
| O5 | **0, exactly** | 363.2 | 5.1% | 17% | at n=10240, k=2048, `approx_top_k_reduction_output_size` returns `(10240, 0)` **at every recall target from 0.8 to 1.0** — the reduction factor is 1. Approximation buys nothing at (A) and costs a pass. |
| O7 | **−21.8% … −2.8%** (H=0) / **−6.6% … −1.4%** (H=130) | 284–353 / 339–358 | 6.5%–5.2% | 22%–18% | fold-and-zero on `C_sc`; carry-forward is worth ~nothing here (the walk is 18 µs). |

**A-D16-8k** — measured 278.7, block floor 4.6, `F_comp` 40.0.

| opt | Δ time % | predicted µs | frac | frac* | leading term |
| --- | ---: | ---: | ---: | ---: | --- |
| **O1** | **−31.6%** | 190.5 | 2.4% | 21% | growth branch **−35.8 µs `measured (j-cefba169)`**, 5/5 repeats, sets byte-identical + 2 empty launches −52.4. **The only option at (A) with a measured full-op component.** |
| O4 | **−87% … −77%** (H=0) / **−41% … −30%** (H=130) | 35–65 / 165–195 | 13.1%–7.1% | **114%–62%** | as above at b=16. `frac* > 100%` is a flag, not a win: it says `F_comp` is wrong at this shape, most likely because the launch term is not one launch. |
| O7 | −21% … −3% / −7% … −1% | 220–270 | 2.1%–1.7% | 18%–15% | |

**B-D4-256k** — measured 1117.9, block floor 36.6, `F_comp` 180.1.

| opt | Δ time % | predicted µs | frac | frac* | leading term |
| --- | ---: | ---: | ---: | ---: | --- |
| O1 | **−6.2%** | 1048.5 | 3.5% | 17% | 2 empty launches −52.4 + `-inf` fill 64 MB −17. |
| O2 | **−10.0%** | 1005.9 | 3.6% | 18% | steps 256 → 32 at 0.500 µs `measured (j-69ee326d)`. |
| **O3** | **−44% … −31%** | 629–766 | 5.8%–4.8% | 29%–24% | live 256k → `p`=4 → **2 waves instead of 8** → `X_crit` 524288 → 131072 → walk −352 to −491 µs. `modelled` on a `measured` rate. |
| O4 | n/a | — | — | — | a 1M score row is 4 MB per token; the fused form exists but the selection would then be a VPU scan of 1M words per token, which is the SparseCore's job done worse. Not offered, with that reason. |
| **O5** | **−29.4%** | 788.9 | 4.6% | 23% | recall 0.99 → 4× reduction (`(262144, 2)`, a shape fact read from `_jax.approx_top_k_reduction_output_size`) → same 2 waves as O3, **plus 23 µs for the reduce pass**. Unlike O3 it also fires at 1M and on ragged batches. |
| O7 | **−41% … −6%** | 660–1051 | 5.5%–3.5% | 27%–17% | carry-forward: `f_w` 0.895 → [0.3, 0.8] `modelled`. Band is wide because §9.3's reference rule points the other way. |

**B-D1-1M** — measured 1481.9, block floor 36.6, `F_comp` 532.0.

| opt | Δ time % | predicted µs | frac | frac* | leading term |
| --- | ---: | ---: | ---: | ---: | --- |
| **O2** | **−31.4%** | 1016.5 | 3.6% | 52% | **`measured (j-69ee326d)`, 5/5, spread 1.4 pt, correctness held.** The only measured full-op option delta in the exercise at (B). |
| O3 | **0** | 1481.9 | 2.5% | 36% | live = padded at 1M; `p` is already 16. The honest limit of O3. |
| O5 | **−23.8%** | 1129.9 | 3.2% | 47% | 4× reduction applies at 1M too. |
| O1 | −4.7% | 1412.5 | 2.6% | 38% | |
| O7 | −28% … −5% | 1067–1408 | 3.4%–2.6% | 50%–38% | |
| **O2 + O5** | **−53.6%** | 688 | 5.3% | **77%** | composed; the two touch disjoint halves (TC loop / SC extent). |

**B-P1024-256k** — measured 50916.8, block floor 956.1, `F_comp` 8490.

| opt | Δ time % | predicted µs | frac | frac* | leading term |
| --- | ---: | ---: | ---: | ---: | --- |
| **O3** | **−44.2%** | 28394 | 3.4% | 30% | 512 waves → 128 waves; walk 30031 → 7508 µs. The largest absolute saving anywhere in the exercise. |
| O5 | −44% | 28500 | 3.4% | 30% | same mechanism, different route. |
| O1 | −2.3% | 49750 | 1.9% | 17% | the `-inf` fill is 4096 MB here = 1107 µs, and it is **59.2% of modelled traffic and 2.3% of measured time** — the clearest demonstration in the block that the byte roofline is the wrong instrument. |
| O2 | −2.0% | 49900 | 1.9% | 17% | |

**B-RAG16** — measured 5813.5. **O3 gets nothing here** (`p` is set by the batch max, and one 1M row
per group of four sets it to 16); **O5 does**, because the reduction is per-row. This is the single
cleanest discriminator between O3 and O5 and step 3 should include the row.

### 9.3 The two `unranked` rows, and why they are not `rejected`

- **O6 — tokamax as shipped.** Its deciding term is not a cost: `plsc.subcore_barrier()`
  **completes at 2 and 4 subcores and hangs the device at 8**, `measured (j-0222bc6d-c03a-4d93-97a7)`,
  and tokamax barriers all 16 cooperating subcores on its final window. Its non-cooperating mode is
  hardcoded to `n ≤ 4096`. **`unranked`, not rejected**: this is a property of the toolchain on this
  slice, not of the design, and if the hang is a version artifact O6 becomes the strongest (B)
  option in the set — it removes stage 2, both gathers, and `MAX_SLICE_WORDS` at once. Its two
  portable ideas are already ranked as O7. **No performance number for it exists**; I report the
  absence rather than quote a bare figure.
- **S4 — host-precomputed byte-offset table (area 7 × area 4).** Its deciding term is the
  **per-copy** cost in the in-body loop, and **a1 arm 5 was not run**. P-ITER predicts a flat table
  under area 7's hypothesis and therefore could never discriminate it; the 0.500 µs/iteration result
  does not exclude an additional per-copy term. **`unranked` and surfaced as a defect**: this is the
  one place in the exercise where a missing measurement decides between options rather than merely
  sizing one. It also needs the `uint16` `page_indices` narrowing, which needs a
  `NotImplementedError` above 65535 pages.

Two more wave-2 seam candidates enter and are dispositioned rather than ranked:

- **S1 (layout × algebra): hoist the e8m0 scale out of the dot product.** Because e8m0 is a pure
  positive exponent, `relu(scale·(q·k8)) = scale·relu(q·k8)` and the head-weighted sum is linear, so
  the scale **factors out exactly** to a dense `[n]` multiply on the finished score. Correct by
  arithmetic, not by measurement. Its value is not speed — it is that it makes a 128-wide contiguous
  cache row *legal*, which is the only route to Lever 1 (`−46.9%` of the modelled memory term at
  (A)), because **the strided 128-of-256 read is measured not to compile** (`Not Implemented: The
  last dim size is not 128 in original base memref`, `measured (j-054e7ac4)`). The repack belongs to
  the cache **writer**, outside this block. **Carried, not ranked** — see §14.
- **S2 (compute units × algebra): overlap layer L's selection with layer L+1's scoring.** Area 5
  measures that overlap behind a **memory-bound** neighbour recovers only 13–16%, and it is a graph
  change at the call site rather than a kernel change. **`unranked`**; the deciding term — whether
  XLA will interleave two custom calls across a layer boundary — is `unmodelled`.

### 9.4 The mechanical checks

**Distance, not difference.** Symmetric difference of the area sets, every pair:

| | O1 {1,2,7} | O2 {2,3,4} | O3 {2,6,7} | O4 {1,2,4,5,6,7} | O5 {1,4,6} | O6 {2,3,5,6,7} | O7 {3,6,7} |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **O1** | — | 4 | 2 | 3 | 4 | 4 | 4 |
| **O2** | | — | 4 | 3 | 4 | 3 | 4 |
| **O3** | | | — | 3 | 4 | 2 | 2 |
| **O4** | | | | — | 3 | 3 | 3 |
| **O5** | | | | | — | 5 | 4 |
| **O6** | | | | | | — | 2 |

**No pair differs in exactly one area**, so no ordering of these options is a chain of single-area
steps. This is a space. All seven areas are covered: (1) O1,O4,O5 · (2) O1,O2,O3,O4,O6 · (3)
O2,O6,O7 · (4) O2,O4,O5 · (5) O4,O6 · (6) O3,O4,O5,O6,O7 · (7) O1,O3,O4,O6,O7.

**The table against the axis list (§5).** Axis 3 → O2. Axis 7 → O3, O5. Axis 8 → O1, O3. Axis 9 →
O6, O7. Axis 10 → O7 (fold-and-zero) and, negatively, the E6 finding in §14. Axis 11 → O7. Axis 13
→ O1. Axis 15 → O3, O5. Axis 16 → O5. Axes 1, 2, 4, 5, 6, 12, 14 have **no option**, each for a
reason recorded in §5: they are contraction dims, degenerate at the deployment's shapes, or outside
the block. Axis 12 (the 21-layer multiplier) has no *kernel* option and is carried in §14 as the
largest non-kernel item.

---

## 10. Reading the exit table at design time

`Where this stops`: fraction ≥80% done · 50–80% one iteration with attribution · **<50% something
structural is wrong in the design or the floor** · >100% not a win.

**Against the block floor — the mandatory denominator — the best composed option at every shape
lands between 3.4% and 13.2%.** The `<50%` verdict has landed on the entire option set, at every
shape, in both regimes. The prescribed response is another option or a corrected floor.

**It is the floor.** §8 shows the op decomposed to within 9% by a model made of terms the floor does
not contain, three of which are `measured` rates and none of which is byte traffic. The floor is
correct arithmetic about the wrong quantity: it prices an HBM stream at 3700 GB/s and the block's
dominant cost is a SparseCore serial scan at ~143 GB/s aggregate — a **26× gap that no kernel
written for this hardware can close while the SparseCore does the scan.**

Against `F_comp`, which adds the scan and one launch:

| shape | best composed option | predicted µs | `frac` (block floor) | `frac*` (`F_comp`) | verdict on `frac*` |
| --- | --- | ---: | ---: | ---: | --- |
| A-D16-8k | O1 | 190.5 | 2.4% | **21%** | still structural — and §4 says why |
| A-D16-8k | O4 (H=0) | 35–65 | 13.1%–7.1% | **114%–62%** | `>100%` at the low end: a flag on `F_comp`, not a win |
| A-D64-8k | O1 + O4 | 140–270 | 13.2%–6.9% | **45%–23%** | structural |
| B-D4-256k | O1+O2+O3+O7 | 485–800 | 7.5%–4.6% | **37%–23%** | structural |
| B-D1-1M | O2 + O5 | 688 | 5.3% | **77%** | **50–80%: one iteration with attribution.** |
| B-P1024-256k | O1+O2+O3 | 26234 | 3.6% | **32%** | structural |

**So the reading is regime-dependent and that is the design's main output.**

- **At (B) with a length-aware partition and a wide KV loop, the block reaches 77% of a floor that
  includes the scan.** That is the "one iteration with attribution" row, and the attribution is
  named: what remains is `C_sc` (150–325 µs, composition `unmodelled`) and the residual walk.
  **(B) is close to done once O2 and one of {O3, O5} land.**
- **At (A) nothing reaches 50% of either floor, and §4 is the reason.** A ~140 µs constant sits
  under every (A) measurement and it is either the harness or the SparseCore launch. **Until that is
  resolved, (A) does not have a floor**, and running more (A) variants without measuring `H` is
  spending hardware to move a number whose denominator is unknown.
- **The `>100%` cell is a result, not an error.** `F_comp` charges exactly one 26.2 µs launch; O4 at
  A-D16 predicts under it. Either the 26.2 µs (a `measured (j-0a0a72c8)` figure on a *different
  kernel*, transfer `assumed`) does not transfer, or O4's `unmodelled` selection band is optimistic
  at its low end. Both are step-3 questions and neither is a reason to believe the low end.

---

## 11. Ranking

**Two separate rankings, because the caller asked for two problems and the evidence gives two
different answers.**

### Regime (B) — AgentX, 256k–1M

| rank | option | why here |
| --- | --- | --- |
| **1** | **O2 — `bkv_p` 2 → 16 at (B)** | The only option in the exercise with a `measured` full-op delta at (B): **−31.4%, `measured (j-69ee326d)`**, 5/5 repeats, 1.4 pt spread, correctness held, and the VMEM budget checked (25.25 MiB of 31). It is one config field. Nothing else on the list is both this large and this certain. |
| **2** | **O5 — approximate prune at recall 0.99, then exact select** | −24% to −44% across every (B) shape *including 1M and including ragged*, on a reduction factor that is a **shape fact, not an estimate**. Ranked above O3 solely on coverage: O3's saving is zero at 1M and zero on B-RAG16, which are the shapes AgentX exists for. **It changes what "correct" means and needs sign-off (§12).** |
| **3** | **O3 — `p` from live length** | Same −31% to −44% at 256k, and −44.2% at B-P1024-256k, with **no change to what correct means** — the strongest reason to prefer it. Loses to O5 only on the 1M and ragged rows. If §12's sign-off is refused, O3 becomes rank 2 and there is no rank 2 gap. |
| **4** | **O7 — SparseCore inner loop** | Band −6% to −41% at (B), the widest on the table, and pointed at by the largest remaining term. Held back because area 5 records a reference rule — *"prefers an O(1) `[lo,hi]` re-scan over a data-dependent candidate buffer… re-scan beats compacting"* — that says O7's central change should **lose**, while tokamax's shipped design says it should win. Genuinely open, cheap to settle in step 3, and it composes with 1–3. |
| **5** | **O1 — stop paying for nothing** | −4.7% to −6.2%. Small at (B) but free and orthogonal; take it with everything. |
| — | O4, O6, S4, S2 | not applicable at (B) / `unranked`. |

**Compose, do not choose: O2 + O5 (or O3) + O1** is predicted at **−54% at B-D1-1M**, `frac*` 77%,
and that is the composition step 3 should build and measure as one candidate.

### Regime (A) — 8k1k

**The table prints a tie between O1 and O4, reached mechanically.**

The rank1–rank2 gap is smaller than the combined uncertainty of the non-measured terms behind it,
so the strike test on the margin returns a tie and I do not break it:

| | O1 | O4 |
| --- | ---: | ---: |
| A-D64-8k under `H = 0` | 310.8 µs | 140–270 µs → **O4 wins by 40–170 µs** |
| A-D64-8k under `H = 130` | 310.8 µs | 270–400 µs → **O1 wins by 0–90 µs** |

The bands **overlap and the ordering reverses**, on a term (`H`) that is `unmodelled` and on a second
term (O4's TC selection, banded [20, 150] µs) that is also `unmodelled`. Two overlapping bands are a
tie. Below the tie: **O7 (−3% to −22%), then O2 (positive, do not apply), then O3 and O5 (exactly
zero at (A), for two different and independently interesting reasons).**

O1 is the safe half of the tie and should be built regardless: at A-D16-8k **−31.6%**, of which
−13.7% is `measured (j-cefba169)` with byte-identical selected sets, and it is three small edits.
O4 is the option that would actually change the shape of the (A) answer, and it cannot be ranked
until `H` is known.

### The assumption that would flip it

**One assumption, and it is the same one for both regimes, though it only bites at (A):
`H` — whether the ~140 µs floor under every measurement in this corpus is the microbenchmark
harness's per-call host dispatch or the SparseCore's real launch cost.**

- If `H` is **harness dispatch**: the (A) op is really ~233 µs, not 363.2; `sparsecore_topk` is
  really ~60 µs at (A), not ~190; the lever is the three TensorCore launches; **O1 wins at (A) and
  O4 is roughly a wash**; and `report.md`'s entire (A) table overstates the block by ~40%, which
  fires unpriced item 12 against every (A) number this exercise has produced.
- If `H` is **SparseCore launch**: 21 layers pay 2.9 ms of pure dispatch per decode step; **O4 wins
  at (A) by 40–170 µs and by a much larger margin end to end**; and the (A) numbers stand.

It settles for nothing: two extra arms in the harness step 3 runs anyway (§13). Everything else in
this document — `f_w`, the 0.500 µs/iteration, −13.7%, −31.4%, the M2 fit, the whole (B) ranking —
is a slope or a paired delta and is **immune to `H` by construction**.

Second-order, and only inside (B) rank 4: whether **compaction beats re-scan** on SparseCore. The
reference rule and tokamax's shipped code disagree, and O7's band is that disagreement.

---

## 12. Correctness

**The gate exists and is validated; cite it, do not rebuild it.**
`instruments/correctness.py`, job **`j-ae538ec1-d606-48f4-8c11`** (Succeeded, exit 0):
`negative control: PASS`; the drop-block mutant fails 5/5; the tail-swap mutant passes 5/5; both
metrics print `load-bearing`; `TOL_DEFICIT = 0.015`, `TOL_CHURN = 0.03`. **Every step-3 variant is
gated by it.**

| option | changes what "correct" means? | what the gate would report |
| --- | --- | --- |
| O1 | **no** | pass. The growth-branch arm already measured **byte-identical selected sets** on all 16 rows, symmetric difference 0 of 32768 slots, `measured (j-cefba169)`. Deleting the `-inf` fill is safe only because the selector gates on `valid = (j < i_hi)` and never reads past the live extent — **if that gate is ever relaxed, O1 becomes a correctness bug**, so the variant must assert it. |
| O2 | **no** | pass. `measured (j-69ee326d)`: selected sets identical to the `bkv_p=2` arm for all three wider arms, `filled row0 = 2048/2048`. |
| O3 | **no** | pass. Changing `p` changes *how* the exact top-k is found, not *what* it is; the existing `p=1` and `p=16` paths already produce the same sets. Ties may break differently, which the gate's `churn` tolerance of 0.03 covers and the docstring already permits ("equal boundary scores may be broken differently"). |
| **O5** | **YES — this is the one** | `recall_target=0.99` means the window-collision probability is 1%, so ~20 of 2048 selected indices per row are expected to differ from the exact set. Predicted churn **~1% `modelled`** against `TOL_CHURN = 0.03`, and deficit ~0 because a missed element is replaced by the next-best, not by garbage. **So it is modelled to pass — and passing is not the point.** Recall is a *collision probability, not an error bound*: it says nothing about how wrong a missed element's score is, and it is an approximation the block does not currently make. **This needs the caller's sign-off before step 3 builds it, and if the gate trips `churn` that is the design working, not a bug to tune away.** P-GATE, the probe that would have measured approximation quality, was excluded by the caller. |
| O4 | **no in principle, and this is its largest risk in practice** | The output must be the exact top-2048 set. O4's threshold stage is exact; its **compaction stage is where exactness is easy to lose**, because the natural TensorCore shortcut — a fixed per-128-block quota — is an approximation that would produce a different set. The variant must not take it. `TOL_CHURN` would catch it; the variant should also assert set-equality against `base.py`'s `_fwd` directly. |
| O7 | **no** | carry-forward changes the traversal, not the predicate. Tie handling must be re-derived: the incumbent's `plsc.cumsum(tie)` quota carry assumes a full re-scan at the final level. |

**No option is `accepted` on a perf number.** A win whose correctness was not established by a named
job is `provisional`.

---

## 13. How step 3 must measure this

- **Run-order drift here is 3–9%, larger than the block-to-block spread.** Arms are timed
  **adjacently and differenced inside each repeat** (`bench.interleaved()`), ≥3 repeats, ≥5 where
  the effect is under 10%. An effect smaller than its own `Δ spread` is not a result.
- **Never form `implied = full − sc`.** It moved 2.6× across three measurements of one shape. Where
  both levels are needed, use a difference of two paired deltas and label it a DiD.
- **First act of step 3, before any variant: measure `H`.** Two extra arms in the variant harness —
  `null` (`jax.jit(lambda x: x + 1)` on a scalar) and `amortized` (N calls inside one
  `block_until_ready`, divided by N). Costs one arm-pair in a run step 3 makes anyway; decides the
  whole (A) ranking (§11) and the credibility of every absolute number in `report.md`'s (A) table.
- **Options whose predicted win is under ~3%, and how to resolve them.** O1's fill-deletion
  component alone is **2.28% at B-P1024-256k** and ~0.2% at (A) — below the drift. It must be
  measured as a **three-arm paired set** (`base` / `fill-only` / `full-O1`) in one repeat, not as a
  standalone A/B, and if `fill-only` is unresolvable that is the honest answer: the fill is 59.2% of
  modelled traffic and does not matter. O7's fold-and-zero at (A) under `H = 130` is 1.4–6.6% and
  needs the same treatment, plus the `H` arms to know its denominator.
- **Shapes step 3 must include, and why.** `A-D16-8k` (O1's measured component; the exit table's
  `≥80%` candidate), `A-D64-8k` (the (A) headline — **and re-measure it, its 9.1% block-to-block
  spread is unexplained and gates every (A) claim**), `B-D4-256k` (O3/O5/O7), `B-D1-1M` (O2's
  measured shape; the shape where O3 is worth zero), and **`B-RAG16`** — the single row that
  separates O3 from O5, which nothing else in the table does.
- **`X_crit`, not `n`, is the independent variable for every SparseCore arm.** An arm that varies
  length at fixed `p` measures the dead-half control and will read as no result; that is what
  P-WALK level B was.
- **Read the recipe back.** `BENCH_CMD` / `RUN_CMD` from `cdk job recipe`, not from the submission.

---

## 14. File plans

Per `design/code-style.md`: four files per operation — `api.py` / `base.py` /
`pallas_mosaic_tpu.py` / `pallas_mosaic_tpu_kernel.py` — in
`kernel_work/glm52-streamidx-topk/<variant>/`, and the same four in `final/`. **"No `pallas_call`
outside the last."** The incumbent's TensorCore/SparseCore file split is *not* a boundary the
reference sanctions; a variant keeping both launches puts both in `..._kernel.py`, or promotes the
selector to its own operation with its own four files.

### O1 — stop paying for nothing (~250 lines changed, no new files)

| file | what it holds | what it must not hold | ~lines |
| --- | --- | --- | ---: |
| `api.py` | unchanged from the incumbent's entry point | any knob | 0 |
| `base.py` | the XLA `_fwd` (dequant → score → mask → `lax.top_k`) — moved here from `instruments/correctness.py`, since §12 of the reference makes `_fwd` the test reference | Pallas, Mosaic, SparseCore | +80 |
| `pallas_mosaic_tpu.py` | **the fixed `_pick_partition`**: the growth loop gains `if cdiv(b*p*2,32) >= cdiv(b*p,32): break`. **The empty-launch decision**: `_get_heuristics_config` computes whether each of the three ranges is empty. | `pl.pallas_call` | +40 |
| `pallas_mosaic_tpu_kernel.py` | the three launches wrapped in `lax.cond` on the emptiness predicate; the `jnp.full(-inf)` deleted, with an **assertion that the selector's `valid = (j < i_hi)` gate is intact** | which config to use | +60 |
| `tiling.py` (ref-free, ints only) | `padded_n`, the wave-count arithmetic `cdiv(b*p,32)`, the emptiness predicates | any `jax` | +50 |

### O2 — widen the KV loop at (B)

| file | what it holds | what it must not hold | ~lines |
| --- | --- | --- | ---: |
| `pallas_mosaic_tpu.py` | `Config` gains `bkv_p` as a real field (frozen/kw_only/slots pydantic); `_get_heuristics_config` returns **16 when `page_size == 512`, 2 when `page_size == 1024`** — the per-regime split, with the `align_to` arithmetic of §9.2 in a comment; `_get_autotuning_configs` sweeps {2,4,8,16}; a `__post_init__` cross-field check that the resulting scratch is under 31 MiB | `pl.pallas_call` | +70 |
| `tiling.py` | the VMEM formulas `bkv = 2·sbs·bkv_p·psz·256`, `bq`, `bo`, and the 31 MiB assertion — **testable with no device** | `jax` | +60 |
| `pallas_mosaic_tpu_kernel.py` | the (1024,4) MIXED tile as an index-map change | policy | +20 |

### O3 — pack the waves

| file | what it holds | what it must not hold | ~lines |
| --- | --- | --- | ---: |
| `api.py` | unchanged signature | config | 0 |
| `pallas_mosaic_tpu.py` | `_pick_partition_from_live(batch_max_eff_len, n, k)`; the `p ∈ {1,2,4,8,16}` ladder; the **bucket-upward** rule with a comment naming area 7's silent-failure hazard in the shrinking direction | `pl.pallas_call` | +60 |
| `pallas_mosaic_tpu_kernel.py` | the `lax.switch` over five precompiled selector bodies; the stage-2 merge, now over live candidates only; **`emit_keys`/`payload_stage2` from `origin/hning/sctopk-pairs-e2e` (`bded1089`) to delete both `take_along_axis` gathers** | policy | +120 |
| `tiling.py` | the ladder and the bucketing, unit-tested on a laptop against a table of live lengths | `jax` | +40 |

### O4 — one op, scores never in HBM, selection on the TensorCore *(the from-nothing target)*

Designed from `q fp8[T,32,128]`, `weights f32[T,32]`, `cache_kv u8[nb,psz/4,4,256]`, the metadata,
the output `int32[T,2048]` and `n = 10240` — with no reference to where the incumbent's operation
boundaries fall.

| file | what it holds | what it must not hold | ~lines |
| --- | --- | --- | ---: |
| `api.py` | the free function the indexer call site invokes; jaxtyping `Integer[Array, "T k"]`; the fallback chain raising `NotImplementedError` when `n > n_vmem_max` — **which is how (B) is excluded, declaratively** | **no `config` at all** | 60 |
| `base.py` | `bind()` validation of the legality ladder (token bucket, `page_size*bkv_p % 128`, `n % 16`, `0 < k ≤ 4096`); the plain-XLA `_fwd` that **is** the test reference | any backend code | 140 |
| `pallas_mosaic_tpu.py` | the `Config` (token-block extent, `bkv_p`, threshold-bisection depth, compaction block width); `_get_heuristics_config` deriving them from `T`, `n`, `k`, `page_size` and `pltpu.get_tpu_info()`; `supported_on` requiring `n · 4 · t_block ≤ vmem_budget` | `pl.pallas_call` | 120 |
| `pallas_mosaic_tpu_kernel.py` | **the single `pallas_call`**, grid `{token block} × {KV block}`; the score tile as VMEM scratch `f32[t_block, 10240]` = 2.6 MB at `t_block=64`; the bisection; the compaction; the masked `int32[T,k]` store with `-1` padding | which config | 320 |
| `compute.py` (ref-free, arrays only) | the fp8×fp8 dot with f32 accumulation, ReLU, the head-weighted sum, the e8m0 dequant, the validity/causal masks, **and the selection algebra: monotone key, the 32-step threshold bisection, the per-block rank** — all array-in/array-out and testable against `_fwd` with no device | refs | 200 |
| `tiling.py` (ints only) | `padded_n`, grid extents, the group-max arithmetic, the compaction block width | `jax` | 60 |
| `load_store.py` (refs in, arrays out) | **the only file where `ref[...]` appears**: unpacking the 256 B cache row into fp8 values plus the e8m0 byte; the masked index store | DMA | 90 |
| `dma.py` | `make_async_copy`, semaphores, memory spaces — **the only file that can deadlock** | arrays | 80 |

**O4's `unmodelled` term, stated plainly.** Exact compaction of ~2048 indices out of 10240 on the
TensorCore has no cheap known primitive. The two routes: a per-128-lane permutation matmul on the
MXU (prefix-sum rank → `P[i,j] = (rank[j] == i) & mask[j]`, two 128×128 matmuls per block, cheap on
the MXU but ~44 µs `modelled` to build the comparison matrices at A-D64) followed by a
running-offset store; or a sort. **The running-offset store is the risk**: area 2 records that a
traced index into a VMEM ref is rejected on both trailing axes, `measured`. Banded [20, 150] µs, and
**the low end is not to be believed until a 20-line compile-and-time probe says the dynamic store
lowers.** That probe is O4's first step-3 act.

### O5 — approximate prune, then exact select

| file | what it holds | what it must not hold | ~lines |
| --- | --- | --- | ---: |
| `base.py` | the `_fwd` reference gains a `recall_target` argument and an **exactness assertion at 1.0** | backend code | +30 |
| `pallas_mosaic_tpu.py` | `recall_target` as a `Config` field with `__post_init__` rejecting anything below 0.99 without an explicit override; `_get_heuristics_config` returning **1.0 at `page_size == 1024`** — the (A) regime, where the reduction factor is 1 and the pass is pure cost | `pl.pallas_call` | +40 |
| `pallas_mosaic_tpu_kernel.py` | the `approx_max_k` call (an XLA op, so it sits in the wrapper) and the index-remap gather; the exact selector unchanged | policy | +50 |

### O7 — rewrite the SparseCore inner loop

| file | what it holds | what it must not hold | ~lines |
| --- | --- | --- | ---: |
| `pallas_mosaic_tpu.py` | `digit_width` and `carry_forward` as `Config` fields; `_get_autotuning_configs` sweeping {4,8} × {False,True} — **the first time axes 9 and 10 of §5 have ever been knobs** | `pl.pallas_call` | +40 |
| `pallas_mosaic_tpu_kernel.py` | fold-and-zero (`fold_priv` writes zeros into the slots it reads, deleting three of four `zero_priv(256)` calls); `plsc.scan_count` pre-aggregation replacing the 16 lane-private histograms; survivor carry-forward via `store_compressed` after level 0; **re-derived tie handling**, since the incumbent's quota carry assumes a full final re-scan | policy | +180 |
| `compute.py` | the radix algebra — bucket extraction, threshold from a cumulative histogram, quota — as pure array functions | refs | +80 |

One hazard, named not resolved, for every file above touching the selector: **`plsc.parallel_loop`
is free to reorder and overlap iterations**, and `emit_body` and `scan_glob` both depend on Ref
side effects across iterations.

---

## 15. Strike tests

**(a) Per-term.** Strike each `unmodelled` term from *every* option and see if the ranking moves.

| term struck | effect on the ranking | verdict |
| --- | --- | --- |
| **`H`** (§4) | **(A) rank 1 moves between O1 and O4 and the sign of O4's band changes.** (B) unaffected. | **MANDATORY PROBE.** §13's first act. |
| **O4's TC-selection cost** [20, 150] µs | struck to zero, O4 wins (A) under both `H` hypotheses; struck to 150, it loses under `H = 130`. | **MANDATORY PROBE** — the 20-line compaction compile-and-time, before O4 is built. |
| **`C_sc`'s composition** (what the 138.6–325.2 µs intercept is made of) | O7's band collapses from [−41%, −6%] to a point at (B) and from [−22%, −3%] to a point at (A). Rank 4 at (B); does not move rank 1–3. | probe if O7 is built; not before. |
| **the per-copy term** (a1 arm 5, not run) | S4 moves from `unranked` to somewhere on the table. No other option moves. | the exercise's named defect; not on the critical path. |
| **MXU peak** | struck — it is no longer `unmodelled`, §3.3. Nothing moves, because nothing was ranked on it. | closed. |
| **26.2 µs launch cost** (measured on a different kernel; transfer `assumed`) | O1's whole (A) value is 2 × 26.2. Struck to zero, **O1 collapses to −0.2% at A-D64 and the (A) tie resolves to O4 under both `H` hypotheses.** | **MANDATORY PROBE** — and it is the same measurement as `H` (a null arm and a 1-launch-vs-3-launch arm at fixed work). |

**(b) Per-seam-term.** Strike each quantity that lives at a seam and no single area priced.

| seam quantity | struck | verdict |
| --- | --- | --- |
| whether XLA materialises the `f32→i32` bitcast between the two ops (area 4 V7; at (B) prefill that would be 67 MB written and 67 MB read, i.e. the traffic table wrong by 2×) | **answered, and by an existing probe**: P-SCBW's four DMA arms are flat across a 16× range of exactly that buffer, so the bitcast does not cost bytes. No option moves. | closed by re-reading `j-69ee326d`. |
| whether the two operations already partly overlap (report item 7, `assumed`) | if they do, every additive prediction in §9 over-states. O1 and O2 touch the TC, O3/O5/O7 the SC, so a composition's saving would be `max`, not `+`. | **affects the composed numbers, not the ranking within a regime.** Step 3 measures the composition as one candidate, which is the protocol anyway. |
| the `-inf` fill's consumer (area 2: it "genuinely has no consumer") | struck the other way — if any consumer exists, O1 is a correctness bug, not a saving. | asserted in O1's file plan; cheap. |
| `plsc.parallel_loop` reordering across `emit_body`'s Ref writes | if reordering is real, O7's carry-forward is unsound as written. | O7 gate, not a ranking term. |

**(c) On the margin.** For each regime, is the rank1–rank2 gap larger than the combined uncertainty
of the non-measured terms behind it?

- **(B): yes.** Rank 1 is O2 at **−31.4% `measured`**, with a 1.4 pt spread. Rank 2 is O5 at −23.8%
  `modelled`. The gap is 7.6 points; O5's own uncertainty is a few points and O2's is 1.4. **The
  ranking holds.** More importantly, O2 and O5 are **not competitors** — they touch disjoint halves
  of the op and compose to −53.6%, so the margin question is nearly moot at (B).
- **(A): no. The table prints a tie.** O1 at −14.4% (whose leading term is an `assumed` transfer)
  against O4 at a band spanning −61% to +10% depending on `H`. The bands overlap and the ordering
  reverses. **Two overlapping bands are a tie, reached mechanically, and I do not break it.** §11
  names the one measurement that does.

---

## 16. Carried

Measured or established here, not kernel edits. Each has a route and a next action. Dropping these
wastes a measurement already paid for.

| # | item | route | next action |
| --- | --- | --- | --- |
| 1 | **`H` may invalidate `report.md`'s entire (A) table** (§4). Unpriced item 12 — "whether the microbenchmark overstates the op" — fires, and the answer decides whether the (A) rows overstate by ~40%. | this exercise | the two harness arms in §13. Then correct `report.md` §3 or confirm it. |
| 2 | **The B-P1024 gap is closed**: 407 TFLOP/s = 35.2% of the measured bf16 peak (§3.3). `probes.md`'s first open item. | `probes.md` | append the closure; it ranks nothing, which is the point. |
| 3 | **`references/08-sparsecore.md` is falsified**: it states `store_compressed` "does not lower" in either flag setting; `sparsecore_topk.py:279` calls it under `needs_layout_passes=False` and the kernel runs. The prescribed `cumsum`+`store_scatter` workaround is unnecessary. | skill reference | correct the reference. |
| 4 | **`roofline.py:86-95`** concludes the A-D16 > A-D32 inversion "is NOT established". It **is** — 5/5 adjacent repeats, and fully caused. | this checkout | fix the comment. |
| 5 | **The 129-of-256 cache row is a *capacity* win independent of bandwidth**: 127 B/token × 21 layers = **~2.7 kB/token of HBM freed**, and at (B) concurrency is capped by KV capacity. S1's scale-hoist (§9.3) makes the 128-wide repack legal. **The repack belongs to the cache writer, not this block.** | `mla_attention_op.py` / the cache spec | own exercise; the arithmetic is done. |
| 6 | **`args[2].copy_(new_cache)` at `mla_attention_op.py:140`, immediately after `donate_argnums=(2,)` at `:128`.** If not elided by storage aliasing this is a full indexer-KV-cache copy per layer per step — ~1 GB per layer at (B), which would dwarf the entire block. `unmodelled`. | call site | one xprof of a real AgentX step. **Larger than anything in §9 if it is real.** |
| 7 | **1.65 ms per decode step of Pallas launch fixed cost** across 21 layers, of which 1.10 ms is launches with an empty grid (axis 12, §5). O1 recovers it per-layer; **merging the three launches into one** recovers the rest and is an area-(2) change nobody has costed. | this block, later | after O1, measure 1-launch vs 3-launch at fixed work — the same arm that settles strike test (a)'s last row. |
| 8 | **`MAX_SLICE_WORDS = 65536` caps the selector at ~2 M columns at b=1**, and AgentX at 1M is at **50% of that ceiling**. tokamax's windowing removes the cap. A capability limit, not a perf one. | roadmap | note it before the next context-length increase. |
| 9 | **E6 (widen L0 to 11 bits) is worse than it looks, and this is new.** A 2048-bucket L0 makes `priv_vmem` 16 × 2048 = 128 KiB and turns `zero_priv` into 2048 iterations at level 0, against the 256 it saves per skipped refinement — and §7's budget says +119 KiB fits at B-P1024 **only** if it is the sole addition. Wave 2 listed it as a candidate; the zeroing cost was not counted. | `combined.md` | record it; it is a candidate correctly *not* on the table. |
| 10 | **`num_kv_pages_per_block[1]`, the dead PREFILL slot**, is load-bearing for `num_sublanes_total` and therefore for the whole score-buffer width. Lowering it shrinks the buffer with no compute effect. One line, unmeasured. | this block | fold into O1's variant as a fourth arm. |

---

## 17. What this design does not decide

- **Whether one kernel serves both regimes.** It does not, and that is a decision with a reason
  rather than an omission: the top option at (A) (O1/O4) is worth **zero** at (B), the top option at
  (B) (O2) is **positive** at (A) by §9.2's arithmetic, and O5's reduction factor is exactly 1 at
  (A). The regimes share `api.py` and `base.py` and diverge at `_get_heuristics_config`, keyed on
  `page_size` — one operation, two heuristic branches, which is what `code-style.md` sanctions.
- **Whether the SparseCore should be in this block at all at (A).** That is O4 and it is tied.
- **Whether compaction beats re-scan on SparseCore.** The reference says one thing and tokamax's
  shipped code does the other. O7's band is that disagreement, and step 3 settles it for the cost of
  one arm.
