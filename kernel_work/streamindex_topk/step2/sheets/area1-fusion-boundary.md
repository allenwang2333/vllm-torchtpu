## Constraint sheet — Area (1), kernel boundary / fusion

Block: `streamindex_topk` scoring half + `sparsecore_topk`, GLM-5.2 DSA lightning indexer.
Reference: `references/01-fusion-boundary.md`. Shapes/baseline/floor: `kernel_work/streamindex_topk/report.md` (step 1). I choose nothing below.

---

### 1. What this area forbids or requires at step 1's shapes

- **Forbidden: any candidate whose stated advantage is "fuses the elementwise ops."** The `-inf` fill, the `indexer_weights` f32 upcast, the `k_packed` pad, `_effective_row_lengths`, the `prepare_q_inputs` pad and the two `reshape`s are all XLA elementwise/bulk ops. XLA is within 3% of Pallas on elementwise chains at every length tested — `0.97–1.01x` [verified: j-050c7480-40a6-4a50-9e1b, `elementwise_chain_xla_vs_pallas`, tpu7x, f32 16384x1024]. Only *removing their HBM crossing* can pay; moving the arithmetic inside cannot.
- **Required before any epilogue claim: the unfused arm.** The measured epilogue tax is `fused/plain = 1.05–1.16` at *every* block, and `fused/xla = 1.11x` at 1024³ [verified: j-054e7ac4-4099-4176-ad2b shard3, tpu7x, bf16 4096³ + f32 bias + tanh-gelu]. The 2.56x this file once led with was a bad block, not fusion.
- **But that tax does not transfer here, and the direction is against the block.** This kernel's epilogue (ReLU, per-head scale, 32:1 head sum, two masks) runs on the *full* einsum output. At `mixed_prod` that is `1024×32×8192 = 2.68e8` f32 elements against `6.87e10` matmul FLOPs (modelled, from report.md's `mxu bf16 = 77.0 µs` @ 892.8 TFLOP/s); the reference's workload was `1.68e7` elements against `1.37e11` FLOPs. **This block is 32× more epilogue-per-FLOP** (modelled). The reference's "epilogue costs 5–16%" is out of range, and this corroborates report.md's unpriced-VPU suspicion for mixed sitting at 29% of floor.
- **Required: price every proposed fusion with `saving_seconds ≈ 2 · intermediate_bytes / 2.97e12`** [verified: j-050c7480-40a6-4a50-9e1b, `intermediate_hbm_roundtrip`; ceiling cross-checked against 2.973 TB/s from j-ed517191-5f2d-4d00-8c0c]. It is an upper bound. Applied below.
- **Residual constraint is N/A because —** this is inference-only, no backward, no `custom_vjp`. tokamax's MLA returns `None` residuals unconditionally [source: `tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py:82-110@9da2369`]. None of §"Where residuals force the boundary" binds.
- **VMEM does not currently forbid widening.** Scratch is `4.09 MiB` (decode, `seq_batch_size=4`) and `4.03 MiB` (mixed, `bq_sz=128`) — modelled from `streamindex_topk.py:654-692` — against the 31 MiB scratch ceiling `02-partitioning.md` owns. The operand-count collapse table [verified: j-ed517191-5f2d-4d00-8c0c, `operand_count_vmem_pressure`] does not apply directly: every `in_spec`/`out_spec` here is `memory_space=pltpu.HBM` with hand-rolled `make_async_copy`, so there is no `BlockSpec` double-buffer ladder to climb. Note `DEFAULT_VMEM_LIMIT_BYTES = 100 MiB` (`streamindex_topk.py:28`) advertises 3× more than the reference's measured ceiling — a source-vs-reference disagreement, not reconciled here.
- **Required: the incumbent's full hand-roll must be justified against the managed pipeline, not assumed.** The rule is narrow: hand-roll only for an operand `index_map` cannot address at all. Absent that, the managed pipeline wins **1.62x** on a pure-copy workload [verified: j-050c7480-40a6-4a50-9e1b, `manual_dma_vs_managed`, 1.213 vs 1.959 TB/s] and `emit_pipeline` ties `BlockSpec` at 0.99x [verified: j-a5c66200-aab5-4319-be73]. Two of the three qualifying cases plausibly hold here (data-dependent trip counts `num_bkv`/`num_bq` from `kv_lens`; `page_indices`-driven addresses — though those are *scalar prefetch* values, which the reference says `index_map` **does** cover). Case 3 (HBM-resident accumulator) does not hold.
- **Forbidden as a rationale: "we issue the DMA earlier."** `late/early = 1.00x` across a 16× compute sweep, with a `noread` control proving the read was on the critical path — Mosaic hoists the `.start()` [verified: j-a4210c9f-9e8f-44b5-bae6]. `start_fetch_bkv`/`start_fetch_bq` (`streamindex_topk.py:230-240`) are exactly that pattern.
- **Forbidden without sizing it first: fusing scoring and selection.** They contract the same axis (KV), so the reference's grid-axis conflict does not bar it [verified: j-a4210c9f-9e8f-44b5-bae6, `gmm_tgmm_grid_axis_conflict`] — but the top-k accumulator becomes whole-output resident: `ntok × k × 8 B` = **1.05 MiB** at `decode_prod`, **16.8 MiB** at `mixed_*` (modelled). Plus 4 MiB existing = 20.8 MiB of 31 MiB; at `ntok = 2048` it is 33.6 MiB and would not fit — the exact `O(K·N)` trap the reference documents at 4096³.
- **Permitted and already taken:** ReLU / scale / head-sum / masks fuse onto an einsum output that is in VMEM registers **for a reason the fusion did not invent**. That is the reference's one unambiguous case [source: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py:507-519@9da2369`]. No candidate should propose un-fusing these for HBM reasons.
- **A fallback gate is part of the design and already exists.** `_fast_path` under `lax.cond` (`streamindex_topk.py:858-865`) is the `ragged_gather_reduce` size-gate pattern [source: `.../ragged_gather_reduce/pallas_mosaic_tpu_kernel.py:473-486@9da2369`]. Its cost is measured at 2.77% more time at `decode_prod` — inside twice the arm's 3.70% spread, so **suggestive, not established** (report.md).
- **Conditionals are optimization barriers** [source: `.../ragged_dot/pallas_mosaic_tpu_kernel.py:805-815@9da2369`]. Any candidate adding a `pl.when` band inherits this; any candidate deleting the empty PREFILL band removes one.

---

### 2. Alternatives this area could take (one line each — enumerated, not ranked)

| # | Boundary | One-line |
| --- | --- | --- |
| A | Status quo | XLA scatter+reshape → XLA `-inf` fill → 3 aliased Pallas calls → XLA glue → SparseCore top-k. |
| B | Absorb the KV-cache insert into the scoring kernel as an `input_output_aliases` entry, the MLA precedent [source: `.../mla/pallas_mosaic_tpu_kernel.py:2493-2502@9da2369`] — removes ~4 whole-cache passes. |
| C | Leave the scatter in XLA but make it write the layout the kernel reads, deleting the `reshape` alone — ~2 whole-cache passes, no kernel change. |
| D | Absorb the `-inf` fill into the kernel (kernel writes `-inf` to uncovered sublanes) — 1 scores pass. |
| E | Size the scores buffer from the batch's actual max KV length rather than `max_model_len` (`streamindex_topk.py:603-605`) — shrinks all three scores passes; report.md probe #6. |
| F | Delete the `-inf` fill outright and rely on `row_lengths` (see ledger row S1) — 1 scores pass, no kernel change. |
| G | Narrow the KV record 256 B → 129/132 B — a boundary around the *cache format*, not an operator; requires changing the writer. |
| H | Emit scores in bf16 — halves all three scores passes; costs precision exactly at the rank-2048 cut. |
| I | Fuse scoring + selection into one kernel — removes all three scores passes; accumulator cost above; crosses to area (5)/(8). |
| J | Partial fusion: per-`bkv`-block local top-k emitting `nbkv × k_local` candidates instead of `sublanes × 128` — pays only if `k_local < 2048` (at `k_local = k` the widths are 8192 vs 10240 at `decode_prod` and 65536 vs 67584 at `mixed_64k`). |
| K | Split the boundary by **phase**, not operator: separate decode kernel (HBM-bound, M=32, 4× MXU derate) and mixed kernel (MXU-bound, M=4096) with separate output buffers. |
| L | Split by **class of input**: delete the structurally-empty PREFILL-band invocation (`mla_attention_op.py:183` makes `distribution[0] == distribution[1]`). |
| M | Merge the three `run_scores_kernel` invocations into one dispatch, since they already alias one buffer. |
| N | Give the schedule back: replace the hand-rolled DMA with `BlockSpec`/`emit_pipeline`, per the 1.62x prior. |
| O | Widen upward into the producer: fold q-quantize / weight-fold / k-pack (`VllmTPUIndexer.forward`, `mla_attention_op.py:239-282`) into the kernel's `bq` load — small (0.26 MB decode / 4.19 MB mixed). |
| P | Widen downward: fold the `topk_indices_buffer` scatter (`mla_attention_op.py:196-199`) into the top-k's write. |
| Q | Narrow rather than widen: hoist the epilogue (ReLU/scale/head-sum) out to XLA and keep only the einsum — the reference's `plain` arm, the cheapest discriminator of whether the VPU tail is the mixed gap. |
| R | N/A: leave the boundary where it is and spend the budget in areas (2)/(4)/(5). |

---

### 3. The traffic ledger (the `Emit …` for this area, filled as far as a briefer can)

Built from **values**, not operations. Sizes in bytes/MB; `passes` = HBM crossings per call. `2.97e12` used for time conversions (measured ceiling); note report.md's roofline uses `3364.9 GB/s` (V7X.md spec) — **the two documents disagree by 13% and every HBM figure inherits it. Not reconciled.**

| id | value | shape / dtype | decode_prod | mixed_prod | mixed_64k | passes | why it crosses | origin |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| I1 | `q_bytes` | u8[ntok,32,128] | 0.26 MB | 4.19 MB | 4.19 MB | 1 r | block input | modelled |
| I2 | `weights` (act dtype) | [ntok,32] | 4 KB | 66 KB | 66 KB | 1 r | block input | modelled |
| I3 | `k_packed` | u8[ntok,129] | 8.3 KB | 132 KB | 132 KB | 1 r | block input | modelled |
| X1 | `weights` upcast f32 | f32[ntok,32] | 8 KB | 131 KB | 131 KB | 1 w + 1 r | XLA op boundary only; `prepare_index_weights:463` | modelled |
| X2 | `k_packed` padded 129→256 | u8[ntok,256] | 16 KB | 262 KB | 262 KB | 1 w + 1 r | to match the cache's 256-B record (`mla_attention_op.py:88`) | modelled |
| **X3** | `cache_kv` after scatter | u8[424960,256] | **~218 MB** | ~218 MB | ~218 MB | 1 r + 1 w | XLA `scatter` on the whole pool to insert `ntok` rows | **measured** (xprof `optimization_work/glm-5-2-2026-09-05/xprof/`, `prefill_only` rank 0: `scatter_fusion` ~291 µs @ ~697 GiB/s) |
| **X4** | `cache_kv` after reshape | u8[424960,256] | **~218 MB** | ~218 MB | ~218 MB | 1 r + 1 w | layout mismatch between scatter output and kernel read | **measured** (same, `reshape` ~291 µs) |
| X5 | `cache_kv` torch copy-back | same | ≤218 MB | ≤218 MB | ≤218 MB | 0 or 2 | `args[2].copy_(new_cache)` at `mla_attention_op.py:140`; may be elided by `donate_argnums=(2,)` | **assumed** |
| K1 | `cache_kv` read by Pallas | u8, 256 B/pos | 134.2 MB ×1 | 2.1 MB ×8 | 16.8 MB ×8 | `nbq` | operand; re-read `nbq` times because `bq` is the outer loop (`:418/:420`) | modelled (report.md) |
| K1a | └ of which padding | bytes 129–255 | 66.6 MB | 8.3 MB | 66.6 MB | — | never read by `load_bkv:265-268`; crosses anyway | modelled (report.md) |
| **S1** | `scores_init` `-inf` fill | f32[ntok,subl,128] | 2.62 MB | 41.9 MB | 276.8 MB | 1 w | to make masked/unwritten columns `-inf` (`:745`) — **but `sparsecore_topk` is given `row_lengths`, and columns at or beyond `row_lengths` "are never selected" (`sparsecore_topk.py:28`, `:424-425`, `:443`), while in-range masked columns are already written `-inf` by the kernel (`:352`). No justification found for this crossing.** | **assumed** (source read, unmeasured) |
| S2 | `scores` kernel write | f32[ntok,subl,128] | 2.62 MB | 41.9 MB | 276.8 MB | 1 w | intermediate handed to a different core (SparseCore) | modelled |
| S3 | `scores` top-k read | same | 2.62 MB | 41.9 MB | 276.8 MB | 1 r | consumer is a separate kernel | modelled |
| S3a | └ padding beyond `kv_len` | `(subl·128 − kv_len)/subl·128` | 20% | 20% | 3% | — | width from `max_model_len`, not the batch | modelled |
| O1 | `topk_idxs` | i32[ntok,2048] | 0.52 MB | 8.39 MB | 8.39 MB | 1 w | block output | modelled |
| O2 | `topk_indices_buffer` copy | i32 | 0.52 MB | 8.39 MB | 8.39 MB | 1 r + 1 w | outside the block (`mla_attention_op.py:196-199`) | modelled |

**Upper-bound saving from removing each crossing** (`bytes / 2.97e12`, or `2·bytes/2.97e12` for a round trip; denominator is report.md's measured `full ms`; **modelled**):

| crossing removed | decode_prod `Δ time %` | mixed_prod `Δ time %` | mixed_64k `Δ time %` |
| --- | ---: | ---: | ---: |
| S1 (`-inf` fill) | −0.5% | −2.1% | −2.0% |
| S2+S3 (scores round trip) | −1.1% | −4.2% | −3.9% |
| S1+S2+S3 (all scores traffic) | −1.6% | −6.3% | −5.9% |
| K1a (KV padding) | −13.4% | −0.4% | −0.9% |
| X3+X4 (~405 MB, per layer per step) | −82% | −20% | −2.9% |

Two things this table says that no merge-of-two-ops question would have asked:

- **X3/X4 are not in any baseline number in report.md.** The Baseline section measures `streamindex_topk`; the scatter and reshape live in `_streamidx_topk_jax` *upstream* of it. ~405 MB of HBM traffic per layer per step — ~582 µs measured, **3.5× the entire measured `decode_prod` call** — to insert 8 KB of useful bytes at decode. This is the largest crossing in the ledger and it is invisible to the yardstick the exercise ranks variants by.
- **S1 appears to be a crossing nobody can justify** (see row). If so it is this area's answer before any merge question is asked, and it is a pure-XLA deletion with no kernel change.
- Two boundaries that are not operator boundaries fall out of the ledger: one around the **cache record format** (K1a + X2 + X4 + G), and one around **phase** (decode's cost is K1, mixed's is S1–S3 — the two regimes have disjoint dominant rows, alternative K).

**Disagreement, flagged not reconciled:** report.md's roofline `kv MB` column equals `nseq × kv_len × 256 B` for the two decode rows (134.2 MB = 64×8192×256; 100.7 MB = 6×65536×256) but exactly **one** sequence's KV for all three mixed rows (2.1 = 8192×256; 8.4 = 32768×256; 16.8 = 65536×256), against `nseq` of 64, 12 and 6 in the shapes table. If the mixed regimes' KV term is understated by that factor, mixed flips from MXU-bound to HBM-bound and step 2's inherited finding #3 ("mixed's headroom is the f32 scores round trip, not the KV") inverts.

---

### 4. Which of this file's `## Gaps` this block turns on

| gap | decision it would settle here |
| --- | --- |
| "The 1.11x is one workload… whether the ~10% epilogue tax holds for a **heavier epilogue** is unmeasured." | Whether the ReLU/scale/head-sum stays fused onto the einsum or moves out (alternative Q). This block is 32× more epilogue-per-FLOP than the measured workload (modelled), and the epilogue is the leading unpriced suspect for mixed at 29% of floor. The probe is the reference's own `plain` arm. |
| "The pipelining argument is sourced, not measured… the probe's fused arm does `write.start(); write.wait()` in the same step. A version that **defers the write** the way the cross-entropy kernel does would test the actual claim, **and has not been written**." | Whether the scores writeback is exposed on the critical path — i.e. whether S2 costs its bytes or is hidden. The incumbent *is* the deferred-write form (`wait_send_scores` waits the *other* buffer before overwriting, `:411-416`), so this block is the missing probe, and the answer decides whether alternatives D/F/I/H are pricing a cost that is already hidden. |
| "Whether Mosaic's hoisting of a hand-issued `.start()` holds at other shapes is unknown — observed at one grid and one tile size." | Whether the entire hand-rolled prefetch (`_fetch_bkv`/`_fetch_bq`, `sem_ids_ref`, 4×2×`sbs` semaphores) is buying anything, or whether alternative N applies. Two shapes here (decode `bq_sz=1`/`sbs=4`, mixed `bq_sz=128`/`sbs=1`) are far from the probed one. |
| "No probe here found a fused Pallas kernel that beats XLA at a realistic shape… the only measured fusion win is the HBM round-trip saving, Pallas-vs-Pallas." | The prior for alternatives B/C/D/O, all of which absorb an *XLA* op. It says the case for each must rest entirely on its ledger rows (X3, X4, S1) and not at all on doing the arithmetic better. |

Not turned on, and left open: the 2.60x→1.11x correction (history, no decision here), and `rhs_bias` in `gmm_v2` (no analogue).
