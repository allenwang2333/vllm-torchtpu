## Constraint sheet — Area (6), algorithm change · `streamindex_topk` scoring half

### 1. What this area forbids or requires

The output is a **set** of indices, so a restructuring must preserve *membership* of the causal top-2048, not scores. But four things are exact obligations, not slack:

- **Cardinality, uniqueness, range.** `test_correctness.py:155-163` checks `len(got_valid) == min(k, n_visible)`, no duplicates, in-range. No restructuring may change how many indices come back or emit a masked position.
- **`-inf` is load-bearing, not a hazard.** `sparsecore_topk` bitcasts f32 to a monotone int32 (`sparsecore_topk.py:320, 51-52`) and derives `KEY_NEG_INF` from `-inf`'s bit pattern. This **inverts** tokamax's rule — *"Mask with a finite large negative, never `-inf`"* [source: `tokamax/_src/ops/experimental/tpu/splash_attention/base.py:28-30@9da2369`] — which exists because `-inf` NaNs under exponential rescaling. There is no exponential here. Swapping to a finite sentinel requires changing `sparsecore_topk`.
- **NaN is forbidden** (`sparsecore_topk.py:28`, "NaNs are unsupported").
- **f32 at the selection boundary** (`sparsecore_topk.py:395-396` raises otherwise), and **the accumulator cannot be narrowed**: a bf16 accumulator does not run at all, `VerificationError` [verified: j-050c7480-40a6-4a50-9e1b, `accumulator_precision`, tpu7x]. The einsum's `preferred_element_type=jnp.float32` (L330) is therefore already forced, not a choice.

### 2. Tagged facts from `06-algorithm.md` bearing on this block

| fact | tag |
| --- | --- |
| Reassociation noise floor, f32 sum of 1.05e6 terms: **2.46e-06 relative**; split count barely matters, so splitting a reduction is numerically free | [verified: j-ed517191-5f2d-4d00-8c0c, `split_contraction_accuracy`, tpu7x, f32] |
| f32 accumulation over **32 steps**: relative max error 4.892e-07 — the head sum is exactly this shape | [verified: j-050c7480-40a6-4a50-9e1b, `accumulator_precision`, tpu7x] |
| bf16 accumulator rejected by Mosaic; bf16→f32 764.8 vs bf16→bf16 772.3 Tops/s, under 1% apart | [verified: j-050c7480-40a6-4a50-9e1b] / [verified: j-ed517191-…, `mxu_peak_by_dtype`] |
| **VPU 1.72 Tflop/s vs MXU 661.4 Tops/s f32 — 385×** | [verified: j-ed517191-5f2d-4d00-8c0c, `vpu_fma_rate`, `mxu_peak_by_dtype`] |
| Sub-128-row MXU passes: **8 rows = 10.57×** per unit of work; 64 rows = 1.97× | [verified: j-ed517191-…, `mxu_block_rows_cost`] |
| Widening N from 128 to 512 costs 1.01× | [verified: j-ed517191-…, `mxu_contraction_and_column_granularity`] |
| Online softmax **lost by 44%** at S=8192, and moved the answer 1.89e-03 relative | [verified: j-050c7480-40a6-4a50-9e1b, `flash_vs_materialized_attention`, tpu7x] |
| A mostly-zero structured matmul buys nothing — causal conv1d stays on the VPU deliberately | [source: `tokamax/_src/ops/causal_conv1d_gated_delta_rule/compute_conv1d.py:34-44@9da2369`] |
| Concatenate two narrow matmuls sharing an operand along the **row** axis, split and scale after | [source: `tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py:719-726@9da2369`] |
| Collapse adjacent head/seq axes with a `reshape` for a larger matmul | [source: `tokamax/…/splash_attention_kernel.py:448-450@9da2369`] |

**One correction to carry upward:** step 1 records the VPU tail as `unmodelled` because `V7X.md` carries no VPU peak. `06-algorithm.md` carries a **measured** VPU rate with a job id (row 4 above). The term is therefore available from this reference rather than absent — but it has not been reconciled against step 1's measured `mixed_prod` scoring time, and until it is, it is a probe, not a floor term.

### 3. Exact restructurings

Already taken, and confirmed independently by DeepSeek (see §5) — recorded so they are not re-proposed: **the UE8M0 position scale is factored out of the head sum** (L337, applied on `[bq,bkv]` not `[bq,32,bkv]`); **`q_scale · softmax_scale · n_head^-0.5` is folded into `w`** (`deepseek_v4_indexer.py:81`); **head and query axes are collapsed into the matmul row axis by a `reshape`, not a transpose** (`load_bq`, L243-244) — the splash idiom.

| # | restructuring | buys | costs / precondition |
| --- | --- | --- | --- |
| **E1** | Fold `w[t,h]` into `q[t,h]` at the `rope_quant` quantizer, so the kernel's per-head multiply on `[T,32,S]` disappears. Identity: `relu(q·kv)·w = relu((w·q)·kv)` for `w ≥ 0` | removes one full VPU pass over `num_tokens × 32 × kv_len` f32 (1.7e7 elems decode, 2.7e8 mixed, both from step 1). q stays fp8 — the quantizer recomputes its own row scale, so no operand widens | **`w ≥ 0` is unproven and probably false.** `weights_proj` is a plain `Linear` with no activation (`attention.py:397`; DeepSeek `model.py:449,478`). If any `w<0`, a ±1 head mask on `[T,32,S]` returns and the win collapses. Crosses the kernel boundary into `rope_quant` |
| **E2** | Transpose the score tile to DeepSeek's `(position, head)` form: contract `k @ qᵀ → [bkv, 32]`, making the head sum a **lane**-axis reduction instead of the current sublane reduction over a 4-D reshape (L332-335) | at decode M becomes `bkv_sz ≥ 128`, **removing step 1's 4× MXU derate** entirely; removes the `[b,bq,32,bkv]` reshape | N drops to 32 of 128 lanes → 4× lane waste in the matmul; at mixed it *introduces* a derate where M=4096 currently has none. Relayout cost has no floor term |
| **E3** | Swap the loop order: `bkv` outer, `bq` inner, so KV is read once instead of `num_bq`=8 times | each score is an independent dot — bit-identical. Removes 7/8 of the KV re-read at mixed | q is re-read 8× instead (far smaller). Owned by area (2); listed here because it is exact |
| **E4** | Batch the sequence axis into the MIXED block (`seq_batch_size > 1` at L806). Einsum test passes: `b` is in the output, never contracted | **payoff 2 only** — fewer grid steps × `512 ns + bytes/3.48e12 B/s`. Payoff 1 is exactly zero (M=4096 already fills the MXU), and `06-algorithm.md` is explicit that this is the case the MXU check correctly rejects while it is still large | requires the "static shape, dynamic size" idiom for ragged `q_len`; L628-629 currently asserts `static_q_len == 1` whenever `seq_batch_size > 1`. VMEM grows linearly |
| **E5** | Concatenate the 4 decode sequences' query rows into one 128-row pass (S2 row-axis idiom) | kills the 4× derate | the 4 sequences have **disjoint** KV pages, so this computes a 4× cross product and discards 3/4 — FLOPs up 4× to remove a 4× derate. Expect break-even or loss |
| **E6** | Head sum as a matmul against a block-diagonal ones matrix `[T, T·32]` | moves the reduction VPU→MXU (385×) | 32/2048 = 1.5% fill. `06-algorithm.md`'s conv1d precedent and S2's inverse both say filling the MXU with zeros is not filling it. Enumerated; expect to lose |
| **E7** | Drop the `jnp.full(…, -inf)` scores prefill (L745). `eff_row_lengths` already tells `sparsecore_topk` the visible extent (L823-834) | removes one full write of the scores buffer, which step 1's `hbm_st` explicitly prices (41.9 MB at `mixed_prod`, 276.8 MB at `mixed_64k`) | must confirm `row_lengths` covers *every* untouched column, including interior gaps between `kv_len` and the bkv-rounded width, and rows the kernel skips |
| **E8** | Bound the bkv loop by each bq block's causal front rather than by `max(kv_lens)` (L275-276) | fewer inner iterations on early prefill chunks | small at the shapes measured; `max(kv_lens)` over the batch also makes it batch-dependent |
| **E9** | Force a native fp8 MXU pass instead of a bf16 promotion (step 1's probe #5) | operands are already fp8 and the accumulator stays f32, so this is exact, not an approximation | a lowering question, not a source one |

### 4. Approximations

Every row's metric is **rank + recall against the float64 oracle** in `instruments/test_correctness.py`, because the output is a set. That instrument is the right gate in form: the reference is strictly better than the thing tested (float64 vs f32), the metric is scale-free, both aggregates print on PASS, and it has been **shown to fire** (job `j-6d675a58-7ecd-4b44-9171`). Two reservations on adopting its thresholds unchanged: `RANK_SLACK` was widened 64→256 *after* watching the shipped kernel land at `worst_rank=2109` of an allowed 2112, and it covers `decode_prod`, `decode_64k` and `mixed_prod` only — no long-context mixed row exists.

| # | approximation | error introduced | metric | proposed gate |
| --- | --- | --- | --- | --- |
| **A1** | **Drop the ReLU** → `score[t,s] = (Σ_h w_h q_h[t]) · kv[s]`. This is the one that dissolves everything: without ReLU the head axis collapses into the contraction, matmul FLOPs fall 32× and the VPU tail vanishes | computes a different function; not a perturbation | rank / recall vs the ReLU oracle | existing check, `RANK_SLACK=256`, `MIN_RECALL=0.99`, run on all five regimes. Must be evaluated **before** any perf number |
| **A2** | Clamp `w ← max(w,0)`, making E1's precondition true by construction | drops the negative-weight head contributions | same, plus report the **fraction of \|w\| mass clamped** as the sensitivity dial | existing check; add the mass fraction as a reported statistic so a pass can be read for how much it was asked to absorb |
| **A3** | Two-stage: score all S with A1's cheap form, take a shortlist of M ≫ k, exact-rescore the shortlist | misses positions with high true score and low surrogate score | recall of the true top-2048 at each M — sweep M | existing check; the sweep sets M rather than a fixed threshold |
| **A4** | bf16 scores buffer (the dominant term at `mixed_64k`: 276.8 MB × 3) | bf16's 8-bit mantissa at score magnitudes 11-49 gives a quantum comparable to the 0.11-0.34 boundary churn already observed | rank / recall | existing check — **but read the margin, not the verdict**: decode currently uses 61 of 256 slack, so this candidate is the one most likely to consume it silently. **Blocked** unless `sparsecore_topk` takes bf16; it raises on non-f32 (`sparsecore_topk.py:395`), and an XLA upcast would re-add the pass this saves |
| **A5** | bf16 relu / weight-multiply / head sum, f32 matmul accumulator | `06-algorithm.md`'s bf16 ops cluster at ~1e-03 relative, f32 at ~1e-06 | rank / recall | existing check. A result landing between those decades is worth understanding before it is accepted |
| **A6** | **Head subsampling — 16 of 32 heads.** *Already measured and rejected* | `worst_rank=8177` (allowed 2304), `min_recall=0.49512` at `decode_prod`; `8159` / `0.42578` at `mixed_prod` | rank / recall | **fails, measured, job `j-6d675a58-7ecd-4b44-9171`**. The negative control zeroes the q rows of half the heads, which — since `relu(0·kv)=0` — *is* this approximation |

Pure reassociation of the 32-term head sum needs no gate: 2.46e-06 relative bounds it [verified: j-ed517191-…], four decades below the 61-rank churn already present.

### 5. Prior art — reached, and what it decided

**Inside the repo** (`Explore`): three implementations — the Pallas kernel (L326-335), the NumPy oracle (`test_dsv4_streamindex_topk.py:100-106`) and a torch reference (`tests/layers/vllm/test_tpu_indexer.py:229-233`) — all compute ReLU before the head sum, none folds `w` into `q`, all materialize the 32-head axis, all accumulate and store f32, none has a coarse stage or any approximation config. `sparsecore_topk`'s two stages are exact and belong to selection, not scoring. There are **two** call sites with different tuning: `mla_attention_op.py:119` uses `num_kv_pages_per_block=(2,2,2)` (GLM-5.2, step 1's block), `deepseek_v4_indexer.py:93` uses `(3,2,2)`.

**Outside the repo — reached by `curl`, not recalled.** `WebSearch` is refused on this model, so I fetched DeepSeek's own DSA indexer directly:

- `deepseek-ai/DeepSeek-V3.2-Exp@main:inference/model.py:435-486` (`class Indexer`) and `inference/kernel.py` (`fp8_index`, TileLang).
- **It computes the identical algebra in the identical order**, stated in its own docstring: `fp8 q @ fp8 k -> fp32 logits` → `relu(logits) * q_s (weights)` → `reduce_sum` over heads → `* k_s (e8m0)`. So both the ReLU-before-sum ordering and the *factoring of the e8m0 position scale out of the head sum* are settled prior art, arrived at twice independently.
- **It does not fold `w` into `q` either**, and `weights_proj` is a plain `Linear` with no activation (`model.py:449`), then scaled only by positive factors (`model.py:478-479`). That is corroborating evidence that **E1's `w ≥ 0` precondition is genuinely open, not merely unchecked here.**
- **Its tiling is the opposite of ours and is E2's existence proof.** `logits` is allocated `(blk_n2, h)` — positions on rows, heads on the contiguous axis — and `T.reduce_sum(…, dim=1)` reduces over heads on the *inner* axis. One query token per program (`i_m` is a grid dimension), `blk_n1=512` / `blk_n2=128`, `num_stages=2`. Our kernel puts `bq_sz·32` on rows and positions on lanes, making the head reduction a sublane reduction over a 4-D reshape.
- **No approximation, no coarse prefilter, no low-rank surrogate** anywhere in it; selection is a dense `index_score.topk(...)` in torch, which is less advanced than `sparsecore_topk`. `index_n_heads = 64` there against GLM-5.2's 32.

**Not reached, and named rather than fabricated:** the DSA paper's own text (`DeepSeek_V3_2.pdf` in that repo, not parsed) on whether any approximation of the indexer was evaluated; NSA / Native Sparse Attention's block-level coarse selection, which is the closest published relative of A3; and any other TPU/Pallas implementation of this operator. If search becomes available, the queries are *"DeepSeek sparse attention lightning indexer top-k approximation recall"*, *"NSA native sparse attention block selection"*, and *"DSA indexer ReLU head weight fold"*.

### 6. `## Gaps` from `06-algorithm.md` this block turns on

| gap | decision it would settle |
| --- | --- |
| *"The flash-vs-materialized result is one hand-written kernel at one shape, non-causal, single head, S ≤ 8192 … Causal masking, multi-head batching and longer sequences all move the comparison and none were measured."* | **Turned on, on all three axes at once**: this block is causal, 32-head, and runs to S=65536. The kernel already never materializes `[T,32,S]` — it reduces per bkv block, i.e. it is already the flash-shaped choice. The gap decides whether **E2**'s re-tiling and **A3**'s shortlist change that, and it is the one reference result that says a restructuring of this family can lose by 44% |
| *"No probe measures a genuine approximation being rejected by a gate. Every accuracy number in this file passed; the gate machinery is untested against a real failure."* | **This block closes it.** A6 — 16 of 32 heads — is a real approximation of this operator, and the rank/recall gate rejected it at 3.5× outside the band on two regimes, job `j-6d675a58-7ecd-4b44-9171`. That result belongs back in the reference file |
| *"The cumsum crossover — the scan length at which the O(n²) triangular matmul stops beating the unavailable scan — was not measured."* | Decides **E6** only (structured-matmul head sum). Marginal here: the fill is 1.5%, so the conv1d precedent likely settles it before the crossover matters |

The two remaining gaps — the jax 0.10.2 `DOES-NOT-LOWER` verdicts and the `Precision.HIGHEST` cumsum error — are not turned on: no candidate on this sheet needs a scan or a triangular matmul.

### 7. Tag `unranked`, not `rejected` — mandatory probes downstream

Each of these is justified by a quantity a FLOPs-and-bytes floor has no term for, so the position step 1's roofline assigns it is the **absence** of a rank. Probe the primitive, not the variant.

- **E1** — its entire claim is the removal of a VPU pass. The VPU is `unmodelled` in step 1's floor; `06-algorithm.md` supplies a measured rate that has not been reconciled against the measured time. **Probe:** the elementwise `max → multiply → 32-way sublane sum` on `[T,32,bkv]` f32, timed in isolation at the `decode_prod` and `mixed_prod` element counts, against the same body with the multiply removed.
- **E2** — two `unmodelled` terms compound: the relayout (no floor term, invisible in profiles) and the sublane-vs-lane reduction. The MXU derate it removes at decode is also `unmodelled` at M=32 specifically — the measured points are 8 rows (10.57×) and 64 rows (1.97×), and step 1's floor interpolates a flat 4×. **Probe:** `[bkv,32]` lane reduction vs `[32,bkv]` sublane reduction at equal element count; and an MXU pass at M=32 to fix the derate.
- **E4** — the claim is fewer grid steps. Step 1's floor **assumes** the 512 ns per-DMA fixed cost is fully hidden by double buffering, which prices this candidate at zero by construction. **Probe:** fit `T = F·steps + W·items` across `seq_batch_size` at the MIXED shape.
- **E5** — same M=32 derate term as E2, in the opposite direction, plus 4× the FLOPs. Its verdict is entirely the derate curve's shape.
- **A1** — the FLOP saving is modelled (32×), but the larger term, the vanished VPU tail, is not. Its *accuracy* is not `unranked`; that is measurable today with the existing gate and should be measured first, since a failing gate ends the candidate before any probe is bought.

Apply the strike test to each: remove the `unmodelled` term from both arms and re-read the ranking. Where it survives, the tag was bookkeeping; where it changes or ties, the probe is required.
