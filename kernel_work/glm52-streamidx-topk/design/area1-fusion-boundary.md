# Area (1) constraint sheet — kernel boundary, `streamindex_topk` (GLM-5.2 DSA indexer)

## 0. Traffic ledger (built from values; `W`=write, `R`=read)

**A-D64-8k** — T=64 rows all live, `n_alloc = align_to(9,bkv_p=2)*1024 = 10240` (11% wider than the 9216 the report quotes; index [1] of `num_kv_pages_per_block` sizes it, `:603-605`), live 8192/row.

| value | bytes | passes | why it crosses |
|---|---|---|---|
| `cache_kv` rows | 128 MiB R | 1 | input; **129 of 256 B used**, 63.5 MiB is padding |
| `q` / `indexer_weights` | 256 + 8 KiB R | 1 | inputs |
| `scores_init` (`jnp.full(-inf)`) | 2.5 MiB W | 1 | **no consumer** — see §1 |
| score matrix, Pallas | 2.0 MiB W | 1 | TC kernel and SC kernel are separate HLO ops |
| score matrix, SC stage 1 | 2.5 MiB R | 1 | same; DMA is `slice_len`=full width, **not** `eff_len` (`sparsecore_topk.py:160`) |
| `eff_row_lengths`, `topk_idxs` | 0.25 KiB + 512 KiB | 1 | outputs/metadata |
| **total** | **≈134 MiB → 36 us modelled** | | vs **363.2 us measured = 10.0%** |

**B-D4-256k** — T=16 rows (4 live), n=1048576, live 262144. SC takes the **two-stage** path (`_pick_partition(16,1M,2048)`: `p_min=16`, `max_p=32//16=2` → p=16), which adds five crossings the op list does not show.

| value | bytes | passes | why it crosses |
|---|---|---|---|
| `cache_kv` | 256 MiB R | 1 | input |
| `scores_init` | 64 MiB W | 1 | no consumer |
| score matrix, Pallas / SC s1 | 4 MiB W / 4 MiB R | 1 | HLO op split; 240 of 256 sub-rows skip their DMA (`i_hi==0`) yet still pay a wave |
| `local_indices`, `global_cand_indices`, `cand_scores`, s2 DMA, final gather | ≈10 MiB W+R | 1 each | **XLA glue between two SC kernel launches**; `take_along_axis` also re-touches the 64 MiB score matrix by random access |
| **total** | **≈340 MiB → 92 us modelled** | | vs **1117.9 us = 8.2%** |

**B-P1024-256k** (50.9 ms, largest in sweep) — T=1024, n=1048576, MIXED path, `bq_sz=128` → `num_bq=8`.

| value | bytes | passes | why it crosses |
|---|---|---|---|
| `cache_kv` | 64 MiB × **8** = 512 MiB R | **8** | the `bkv` loop is *nested inside* the `bq` loop (`:418` inside `:420`) — the same KV is re-read once per q block |
| `scores_init` | **4096 MiB W** | 1 | 56% of the whole ledger, no consumer |
| score matrix Pallas W / SC s1 R | 1024 + 1024 MiB | 1 | HLO op split |
| two-stage SC middle | ≈530 MiB | 1 each | as above, at 1024 rows |
| q / out | 4 + 8 MiB | 1 | |
| **total** | **≈7.2 GiB → 1.95 ms modelled** | | vs **50916.8 us = 3.8%** |

## 1. What area (1) forbids

- **Forbids expecting more than ~7% from the boundary.** Applying the reference's own formula (`saving ≈ 2·bytes/2.97e12`, 1× for the write-only fill): kill the `-inf` fill (−0.2% / −2.0% / −2.8% across the three shapes), the score round trip (−0.4 / −0.25 / −1.4), the two-stage middle (−0.4 at P1024), 7 of 8 KV passes (−0.3), and Lever 1's 129/256 (−6.3 / −4.0 / −0.2). **Zeroing the entire ledger is worth ≈−7%, −6.5%, −5%.** HBM traffic is 4–10% of measured time everywhere; a boundary justified as traffic removal cannot repay itself here.
- Forbids "fuses the elementwise ops" as an argument — XLA is at 0.97–1.01x on chains up to length 16 `[verified: j-050c7480-40a6-4a50-9e1b]`. The relu/weight/head-sum/scale/mask chain is already inside the kernel and is not a lever in either direction.
- Forbids hand-rolling a DMA to issue it earlier: `late/early = 1.00x`, and the naive hand-roll lost 27–31% to not fusing `[verified: j-a4210c9f-9e8f-44b5-bae6]`; the managed pipeline wins 1.62x when there is nothing to overlap `[verified: j-050c7480-40a6-4a50-9e1b, manual_dma_vs_managed]`. **The incumbent is already fully hand-rolled** — `pltpu.HBM` on all four operands, manual `make_async_copy` double-buffering — and its case-1 justification (page-table-indexed addresses) is weak, because `PrefetchScalarGridSpec` gives `index_map` those same scalars. Reference step 7 is the exact test and it has not been run on this kernel.

## 2. What area (1) requires

- **Requires a boundary that changes work, not bytes.** The two crossings the ledger indicts are indicted for *what they imply*, not their size: the `-inf` fill and the SC read are both sized by `n = pages_per_seq·page_size = max_model_len`, and the SC's cost is measurably blind to bytes (4× live data → +3.5%). The boundary that pays is drawn around **live rows × live columns**, not around an operator.
- **Requires the `-inf` fill be shown to have a consumer or be deleted.** `_effective_row_lengths` already returns `min(kv_len, token_pos//cr + 1)` — the exact causal bound — and `load_keys` masks on `j < i_hi`. Columns beyond `eff_len` are never examined by SparseCore; columns inside it are all written by the Pallas launches. The fill is redundant with a mask the SC applies anyway. (Register item 2 — whether XLA materializes it — decides whether deleting it is worth 2.8% or 0%.)
- Requires the unfused control before any fused claim (reference step 5). It already exists: scoring alone = full − SC = 173.7 us at A-D64-8k, 322.0 us at B-D4-256k.
- Requires a named XLA/size fallback threshold, as `ragged_gather_reduce` gates at 60% of VMEM `[source: ragged_gather_reduce/pallas_mosaic_tpu_kernel.py:473-486@9da2369]`. Here the natural gate is live-length vs `n`, or `T·n·4` vs VMEM.

## 3. Candidate boundaries the ledger generates

- **C1 Delete the `-inf` materialization.** −2.8% at B-P1024-256k; removes a 4 GiB allocation that also caps concurrency (register item 4).
- **C2 Score→select fused, score matrix never in HBM.** The score matrix is *the* intermediate at issue. One row is 40 KiB (A) / 4 MiB (B); 4–64 live rows fit 64 MiB VMEM. Removes 3 crossings.
- **C3 Boundary around live extent, not `max_model_len`.** A compacted `[Σ live rows, max live len]` score value. At B-D4-256k the buffer is [16, 1048576] and the content is [4, 262144] — **6.25% occupancy**. This boundary follows a *class of input*, not an operator, and appears in no op list.
- **C4 Streaming select: bkv-tile → running top-k threshold.** Boundary around a *phase*; the score value never exists at full width at any instant. Subsumes C1–C3.
- **C5 Invert the MIXED loop nest** (bkv outer, bq inner): KV read once instead of 8×. q for 1024 tokens is 4 MiB, resident.
- **C6 Fuse SC stage 1 and stage 2**, removing the XLA glue and the `take_along_axis` re-touch of the score matrix (have stage 1 emit (index, score) pairs).
- **C7 Lever 1**, narrow the cache row to 129 of 256 B: −6.3% at A-D64-8k if the strided read is free (register item 1, unprobed).

**The TC/SparseCore split is itself a boundary, and only a value-built ledger can question it.** An op list sees `pallas_call → pallas_call → sc_topk` and can only propose merging adjacent members; the ledger shows the score matrix crossing HBM once in each direction *for no reason other than that the two ends run on different units*, and that the two-stage select adds five more crossings for the same reason. This file has **no facts on a TC↔SC boundary** — that is `08-sparsecore.md`'s to price, and it is the one number this block most needs.

## 4. Considered, not favoured (designer decides; these are the losers)

- Merge the three TC launches — they write disjoint token ranges into one aliased buffer; two of the three are empty at every shape here. Removes ≈0 crossings.
- Fuse `_effective_row_lengths` in — 256 B to 4 KiB, below noise.
- Fuse the trailing `topk_idxs[:T,:k]` slice — a view, no crossing.
- Fuse the indexer QK into MLA attention proper — different k cache, and it contracts `d` while attention contracts elsewhere; reference step 6.
- Widen to include the indexer-KV-cache write — that is `args[2].copy_()` at the call site (register item 3), not a kernel boundary.
- Split scoring further (separate mask kernel) — anti-fusion in the wrong direction; adds a full `[T,n]` crossing.

## 5. Gaps this block turns on (each with the decision it settles)

- **"Whether Mosaic's hoisting of a hand-issued `.start()` holds at other shapes is unknown — observed at one grid and one tile size."** Settles: keep the incumbent's fully hand-rolled schedule, or return the operands to a managed `BlockSpec` grid with scalar-prefetch `index_map`. Directly implicated by the unattributed 685 ns/KV-block against 69 ns of DMA (report §6 item 3) and register item 5. **Rank this first.**
- **"The pipelining argument is sourced, not measured; the probe's fused arm does `write.start(); write.wait()` in the same step."** This kernel is the missing test: `start_send_scores` / `wait_send_scores` is exactly tokamax's deferred double-buffered writeback, with a matmul per step. Settles whether the score writeback is already hidden — which decides whether C2 is worth anything beyond its 0.4–1.4% of bandwidth.
- **"Two ops can only share a grid if they contract the same axis."** Scoring contracts `d`(=128); selection reduces over `s`(=n). They do not match, so a naive shared grid makes the select's accumulator whole-output-resident — but here that accumulator is only `k`=2048 i32 = 8 KiB/row, so it *fits*. Settles that the fused form must be C4 (streaming select carrying a running threshold), not a single grid over both, and that the usual "does not compile" failure mode does not apply.

Not turned on: the 1.11x matmul-epilogue figure and its one-workload caveat (no epilogue at issue); the `rhs_bias` gap; the residuals section entirely — this is inference-only, `None` residuals, so the whole residual constraint is `N/A because there is no backward pass`.
