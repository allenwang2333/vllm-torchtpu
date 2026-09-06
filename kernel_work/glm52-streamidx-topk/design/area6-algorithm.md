# Area (6) constraint sheet — GLM-5.2 DSA Lightning Indexer top-k

**Ruling facts first.** Every shape is at 2-8% of its modelled floor [measured: `j-59d0b337`, `j-284a8842`], so **no candidate in this sheet may be rejected on FLOP or byte count** — the model that would do the rejecting is wrong by 5.7-32x on the incumbent. Per `AREAS.md` (6) and `06-algorithm.md` § Deciding step 4, anything whose only defect is arithmetic count is `unranked`.

**Three source facts the step-1 report does not have, found reading `sparsecore_topk.py`:**
- **The stage-1 DMA is *not* bounded by `eff_len`.** `dma(scores_hbm.at[pl.ds(s, slice_len)], resident_vmem, 1)` copies the full padded slice; only `walk_chunks` is bounded by `i_hi = min(slice_len, eff_len)` [source: `sparsecore_topk.py:155-160`]. The report's §2 claim to the contrary is wrong. This is a direct exact fix and part of Finding 2's mechanism.
- **`_pick_partition` is a function of static `n`, never of `row_lengths`** [source: `:360-377`]. At (B), `p = 16` always → stage 2 always merges `p*k_p = 32768` candidates, and both `jnp.take_along_axis` gathers run against the padded `[b, 1048576]` row [source: `:439,452`]. Those XLA gathers are inside the timed `sparsecore_topk`.
- **Waves cost the max over subcores.** `row = w*32 + flat` [source: `:148`]; at B-D4-256k slices 4..15 of each row are empty and idle while slices 0..3 walk a full 64K words. That is why 4x the live data costs +3.5%: live length shortens *some* subcores, never the wave. This is Finding 2's real mechanism and it is fixable exactly.
- **Both masks are exactly a per-row prefix length.** `visible = min(kv_len, token_pos//cr + 1)` [source: `streamindex_topk.py:504`] already encodes validity *and* causality as one scalar. Nothing about either mask requires materializing `-inf`.
- **Passes over the resident slice: 6** — `cvt`, `hist0`, three refinement walks, `emit` [source: `:158-291`], `done` collapsing some. At A-D64-8k that is ~2900 16-lane chunk-ops in 189.5 us, ≈65 ns/chunk: the cost is per-pass-per-chunk issue, not bytes.

### Exact restructurings (no accuracy gate; a perf number suffices)

| # | change | costs, in a term the roofline *has* | buys, in a term it *lacks* | rank |
|---|---|---|---|---|
| E1 | Bound the stage-1 DMA by `eff_len` (rounded to the 32 B granule), not `slice_len` | none — strictly fewer bytes | fewer bytes *and* fewer DMA-wait cycles | ranked; roofline sees part of it |
| E2 | **Length-aware partition**: choose `p` and slice boundaries from `eff_len`, so all 32 subcores get `sum(eff_len)/32`, not `n/p` with most idle | index metadata; dynamic (not static) slice bounds — an area-(7)/(2) seam | eliminates the wave-max idle; ~2x at B-D4-256k, ~2x at B-D1-1M by construction | **`unranked`** — idle-fraction has no roofline term |
| E3 | **Block-max prune (two-stage exact select)**: emit `f32[T, n/128]` block maxima during scoring (VPU reduce over registers already held); on SC, select the top-m blocks by max with `m*128 >= k`, take `τ_lb` = k-th largest within their union (`τ_lb <= τ*` since a subset's k-th is a lower bound), then walk only blocks with `max > τ_lb` | +0.8% HBM (`n/128` extra f32), one extra VPU reduce, two extra small passes | at (B) removes ~99% of the walked columns; **exact, no gate** | **`unranked`** — the term that improves is walked-element count |
| E4 | Drop the `-inf` fill entirely; carry the prefix length instead (already computed at `:504`) and rely on `j < i_hi` | none | removes the 4 GiB fill at B-P1024 (59.2% of that row's traffic) and the write/read of dead columns | ranked; ≤10% of measured time — take it, don't lead with it |
| E5 | Fuse `cvt` into `hist0`: compute `monotone_key` on the fly in each walk instead of a dedicated read-modify-write pass | +2 ALU ops per lane per walk (roofline: negligible FLOPs) | 6 passes → 5, and removes a full VMEM write pass | **`unranked`** — pass count is not a roofline term |
| E6 | Widen L0 to 11 bits (2048 buckets) so 11/11/10 covers 32 bits in 3 levels instead of 4 | histogram VMEM 16×2048×4 B = 128 KB/subcore, forcing `MAX_SLICE_WORDS` down and `p` up | one fewer full walk over the resident slice | **`unranked`**; interacts with E2 — probe together |
| E7 | Fuse `emit` into the final refinement level (the winner mask is derivable there) | register pressure | 5 passes → 4 | **`unranked`** |
| E8 | Warm-start row `t+1`'s radix prefix from row `t`'s at prefill (causal rows are nested prefixes, so the k-th score moves slowly); verify the count and back off if wrong | a verification pass in the rare backoff | skips 1-2 refinement levels on most prefill rows; targets the 50.9 ms row | **`unranked`** |
| E9 | Rate-dependent dispatch: **(A) and (B) do not want the same algorithm** — see below | one more code path | picks the right one of the above per shape | ranked as a wrapper, not a kernel |
| E10 | Sort/group sequences by length before the `seq_batch_size=4` group-max (`num_bkv = cdiv(max_i kv_lens[i], bkv_sz)`, `:276`) | a permutation and its inverse | Finding 3's measured **+184%** on mixed-length batches | ranked; the repo already has the TODO at `:751-754` |

**Already exploited, do not re-propose:** the output is unsorted and `plsc.store_compressed` already avoids any sort [source: `:279`]; `keep_all` is already a per-row early-out on `total <= k` [source: `:200`] (the *batch*-level `enable_early_exit` is the one that never fires).

### The (A)/(B) split — this is the real finding

E3 is worth almost everything at (B) and almost nothing at (A), and the reason is arithmetic, not taste. At (B) decode the selection rate is **0.2%**, so a 128-wide block is skippable with near-certainty. At (A) decode it is **22-25%**: a block survives unless all 128 of its entries fall below the ~75th percentile, i.e. with probability ≈ 0.75^128 ≈ 1e-16 under independence [unverified — the score distribution is right-skewed with an atom at the ReLU floor, so the real number is larger, but not by 16 orders]. **Candidate-set pruning is a (B) algorithm.** At (A) the only levers are the pass-count ones — E1, E2, E5, E6, E7 — because there is no sparsity to exploit and the cost is per-chunk issue overhead. Any single design that claims to fix both shapes with one mechanism should be disbelieved.

### Approximations

**Recommendation: none is warranted yet, and that is a decision on the record.** The exact list above is not exhausted, it contains at least two candidates (E2, E3) with structural factors rather than percentages, and `06-algorithm.md` is explicit that an exact restructuring can be adopted on a perf number alone while an approximation cannot. Buying an accuracy gate before the free wins are taken is paying for something not yet needed. The measured 0.3%-of-range boundary churn on the incumbent [measured: `j-284a8842`, Finding 5] is **a fact about the tolerance that exists, not a licence**; it is the *calibration* for the gate below, not a pre-approval.

Enumerated anyway, so they can be ranked and retracted rather than being absent:

| # | approximation | error introduced | metric | gate (threshold set in advance, evaluated *before* any perf number) |
|---|---|---|---|---|
| A1 | Truncate the radix key to 16 bits (stop after L0+L1); take the whole tie bucket and truncate arbitrarily to `k` | ties within one 2^16 key bucket resolved arbitrarily — bf16-grade, expect ~1e-3 relative, the decade `06-algorithm.md` puts bf16 ops in | `deficit` (worst selected score below true k-th, as fraction of top-to-cut range) and `churn` (fraction of true top-k missing), both from `instruments/correctness.py` | `deficit <= 0.005` and `churn <= 0.005` on all 5 correctness cases vs the f32 reference — i.e. no worse than the incumbent's own measured 0.3% [Finding 5] — **plus** the negative control below |
| A2 | Sampled threshold: estimate τ from a 1/16 subsample, keep everything above, fill any shortfall arbitrarily | drops true top-k members in the sampling tail; error grows as the selection rate falls, so it is *worse* at (B) where it is needed | same two metrics, plus end-metric recall against exact top-k | `churn <= 0.01`; reject outright if churn is shape-dependent, since it would then be untestable at (B)'s 1M shape |
| A3 | Per-KV-block fixed quota (top-`k/nblocks` within each block) | systematically drops concentrated mass; changes which columns MLA attends to | churn; then a real accuracy run on the model | not proposed — the error is structural, not tie-level, and no cheap proxy bounds it |

**The reference and the order.** The reference must be strictly better than the thing tested (`06-algorithm.md` § Writing the gate): an f32 recomputation mirroring the kernel's operation order, which `correctness.py` was rewritten to be [Finding 5]. The floor to argue thresholds against is **2.46e-06 relative** reassociation noise [verified: `j-ed517191-5f2d-4d00-8c0c`, `split_contraction_accuracy`]; anything landing between 1e-06 and 1e-03 needs understanding before acceptance. The downstream consumer is sparse MLA attention, so the *end* metric is model output quality and `churn` is only the cheap proxy — an approximation that passes the proxy still owes an accuracy run before `accepted`.

### What area (6) forbids and requires here

- **Forbids** adopting any of A1-A3 on a speedup: "a kernel that skips work is fast", and this kernel's entire cost is work it could skip. Forbids "the user will decide" as a gate.
- **Forbids** ranking E2, E3, E5-E8 by FLOPs or bytes and reading the result as a verdict. At 2-8% of floor the model did not score them low; it did not score them.
- **Requires** the exact list be enumerated even with no approximation on the table — done above, and it is where the recommendation sits.
- **Requires** the gate be written *first* for anything approximate, and evaluated before the perf number.

### `## Gaps` this block turns on

1. **"No probe measures a genuine approximation being *rejected* by a gate — the gate machinery is untested against a real failure."** Decision it settles: whether A1's threshold may be trusted at all. Under *untested*, A1 cannot be judged and must not be measured for speed; under *tested*, it can. The action is a **negative control** — deliberately drop one KV block and confirm `churn` red-lights. The report already predicts the two sides (~0.3% for boundary churn vs ~25% for a dropped block at 8k), so this is one cheap job and it makes the gate a discriminator instead of an assertion.
2. **"The flash-vs-materialized result is one hand-written kernel at one shape"** — and the finding it carries, that *the memory argument for a fused kernel is a memory argument, not a speed argument* (hand-written flash lost 44% at S=8192 exactly where it should win). Decision it settles: how much to expect from E4 / the report's Lever 2 (never materialize the score matrix). Under "the gap generalizes", E4 is a capacity and concurrency win worth ≤10% of time and must not be the headline; under "it does not", E4 could be more. Given the measured 2-8%-of-floor, I would act on the first reading and let E2/E3 carry the design — but the gap is nameable and the probe is a fused score+select at one (B) decode shape.

### Considered and not favoured

- **Move the whole select to the TensorCore, or split rows TC/SC** — real, but it is area (5) placement, not algebra; flagged as a seam, not claimed here.
- **Bitonic / sorting network select** — the output is unsorted, so any full ordering is work the consumer discards.
- **Single-pass threshold with no refinement** (histogram, take the bucket, stop) — this is A1 by another name; enumerate it once, under the gate.
- **Drop low-|weight| heads from the score** — the score is a sum of 32 rectified dot products with signed weights; dropping heads perturbs scores, not tie order, and no cheap metric bounds it.
- **Complement selection at (A)** (reject 7168 instead of selecting 2048) — the rejected set is larger than the selected one at a 22-25% rate; no saving.
- **Per-row `p` chosen from `k/eff_len`** — subsumed by E2, listed so it is not re-derived as separate.
- **Batching an independent axis into the block** (`06-algorithm.md`'s standing check) — the row axis `T` is already batched across subcores and waves; the axis is present, not absent. The failure is idle subcores (E2), not a missing axis.

**I have not chosen among these.** E2 and E3 are the two with structural factors rather than percentages behind them, and they are independent of each other; E1, E4, E5, E7 and E10 are cheap and exact and lose nothing by being taken alongside whichever wins.
