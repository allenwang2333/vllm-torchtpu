# Area (2) constraint sheet — `streamindex_topk` + `sparsecore_topk`

**Framing correction that governs everything below.** "Runs in parallel" has one worker on the TensorCore side: tpu7x reports `num_cores=1`, `is_split_chip=True`, `is_megacore=False`, so a `parallel` grid axis has nothing to split [verified: j-4fba8dc6, 02-partitioning §`dimension_semantics`]. The TC grid is therefore **not** a parallelism decision — it is a per-step-cost and pipelining decision. The only genuinely parallel hardware this block touches is the SparseCore's 2 cores × 16 subcores, and that partition is in `_pick_partition`, not in the Pallas grid. Both halves are emitted.

### The iteration space, in full

`{seq-group g} × {query block bq} × {KV block bkv} × {head h=32} × {head_dim d=128} × {score column within bkv}`. Cut: `g` only (the 1-D grid). Walked sequentially inside one kernel body: `bq` then `bkv` (`fori_loop`, `:418,:420`). Never cut: `h` and `d` — both live entirely inside one `jnp.einsum`, so **the contraction axis is inside the block and the "contraction innermost" rule is satisfied vacuously; there is no k axis to place**. All in/out specs are `pltpu.HBM` (idiom 3, `:645-652`), so **the (8,128) block-shape rule never fires** — that is why a `(4, 256)` trailing pair is legal in `bkv_x2_ref`, and it is a licence a `BlockSpec` rewrite would give up.

### The three shapes as configured today

| | A-D64-8k | B-D4-256k | B-P1024-256k |
|---|---|---|---|
| launch that runs | DECODE batched (`:757`) | DECODE batched | MIXED (`:793`) |
| **grid** | `(16,)` | **`(1,)`** | **`(1,)`** |
| `seq_batch_size`/`bq_sz`/`bkv_p` | 4 / 1 / 2 | 4 / 1 / 2 | 1 / 128 / 2 |
| `bkv_sz` | 2048 | 1024 | 1024 |
| `num_bq` × `num_bkv` per step | 1 × 4 | 1 × 256 | **8 × 256** |
| sequential inner iterations, total | 64 | 256 | **2048** |
| `make_async_copy` starts for KV | 128 | 512 | 1024 |
| **VMEM working set** (scratch only) | 4.10 MiB | **2.07 MiB** | 2.55 MiB |
| …against 31 MiB default budget | 13% | **6.7%** | 8.2% |
| **HBM traffic implied** | 135.5 MiB | ~346 MiB | **~7.3 GiB** |
| …modelled at 3.70e12 B/s | 38.4 us | 98 us | 2072 us |
| measured (report §6) | 363.2 us | 1117.9 us | 50916.8 us |

VMEM detail at B-D4-256k: `bkv` 2 MiB + `bq` 32 KiB + weights 1 KiB + `bo_scores` 32 KiB. **The block is ~15x smaller than VMEM allows and this is the clearest single finding on the TC side.** Traffic detail at B-P1024-256k: `-inf` fill 4096 MiB (56% of all bytes), score writes 1024 MiB, SC stage-1 reads 1024 MiB, KV 512 MiB, XLA candidate gathers ~272 MiB.

### Idle fraction, by mechanism

| mechanism | A-D64-8k | B-D4-256k | B-P1024-256k |
|---|---|---|---|
| KV row bytes unused (129 of 256) | 49.6% | 49.6% | 49.6% |
| score-matrix width padded to `max_model_len` | 10% (80 vs 72 sublanes) | **75%** | **75%** |
| KV re-read across `bq` blocks | 0 (num_bq=1) | 0 | **87.5% redundant** (8 passes) |
| SC stage-1 subcore slots dead | 0% (p=1, 64/64 live) | **75%** (64 live of 256) | **75%** (4096 of 16384) |
| SC stage-2 subcore slots dead | n/a | **50%** (16 of 32) | 0% |
| ragged group-max | ~0 in bench; **~37% modelled** at production ISL uniform[1638,9216] | 0 (uniform) | n/a (1 seq) |

### The ragged tail, priced exactly

Waste factor is `Σ_g sbs·cdiv(max_{i∈g} kv_len_i, bkv_sz) / Σ_i cdiv(kv_len_i, bkv_sz)`. At `B-RAG16` (4×1M + 12×4096, `sbs=4`): worst interleaving gives 16384 block-fetches against 4144 required = **3.95x ceiling**; measured inflation is 2.83x. **Sorting by length before grouping recovers the whole of it** — the sorted assignment (one all-1M group, three all-4096 groups) costs exactly 4144, i.e. 1.00x. The permutation is 64 i32 = 256 B in SMEM against ~1 MiB [verified: j-ed517191, `smem_scratch_bisect`], so cost is nil; the TODO at `:751-754` names it. Two things to check rather than assume: `cu_q_lens_ref` and `page_indices_ref` must be indexed through the same permutation, and output rows are addressed by `token_start = cu_q_lens_ref[...]` so a permuted grid *should* write correctly without re-permuting the output.

`seq_batch_size` is itself a partition parameter coupled to this: halving it to 2 halves the raggedness ceiling and doubles grid steps; sorting makes larger groups cheap and is the prerequisite for raising it.

### Traversal order — the one that costs 8x

`bq` outer / `bkv` inner at MIXED re-reads the entire KV span `num_bq` times: 512 MiB instead of 64 MiB at `B-P1024-256k`. The fix is not to swap the loops (that re-reads q 256×) but to **hold the whole q chunk resident**: `bq_sz=1024, bkv_p=2` gives `bq` 8 MiB + `bo` 8 MiB + `bkv` 512 KiB + weights 256 KiB = **16.8 MiB, fits under 31 MiB**, cuts KV traffic 8x *and* inner iterations 8x (2048 → 256). `bq_sz=512, bkv_p=4` is 21 MiB, 128 iterations, but 128 MiB of KV — a real trade curve, not one answer. Note `bo_scores` scales as `bq_sz × bkv_sz`, so it is what binds first, and `bq_sz=1024, bkv_p=8` (32 MiB `bo`) does **not** fit.

Decode side: at B-D4-256k, `bkv_p=16` gives `bkv` 16 MiB + `bo` 256 KiB ≈ 16.3 MiB, fits, and cuts 256 iterations to 32.

### What area (2) forbids and requires here, with provenance

- **`vmem_limit_bytes=100*2**20` (`:28`) is above the 64 MiB physical.** Raising the scoped limit does not create memory [verified: j-106c8a46, `vmem_limit_bytes_raises_ceiling`]. **A widened block that compiles is not thereby shown to fit** — check every candidate against 31 MiB (default) or ≤64 MiB, never against 100.
- **Grid extent is traced** (`distribution` is `i32[3]`), so two of the three TC launches run at grid 0 at every shape above and **cannot be compacted out at trace time**; only 02-partitioning's "dynamic grid length" route applies. Launches per layer per step: 5 at A (p=1), **8 at B** (3 TC + 2 SC `pl.kernel` + 2 XLA `take_along_axis` + fill) — ×21 layers = 105 / 168.
- **`arbitrary` on the grid axis is required, not conservative**: `sem_ids_ref` and `bo_sz_ref` carry state across steps, and the prologue/epilogue are `pl.when` on the first/last step. Mislabelling it `parallel` is a correctness bug [02-partitioning §`dimension_semantics`].
- **A traced index into a VMEM ref is rejected on both trailing axes** [verified: j-106c8a46, `dynamic_index_lane_versus_sublane`]. Today's `scores_block_x2_ref[bo_sem_idx, …]` is a *leading*-axis index and legal (leading axes are bookkeeping [hardware-facts §Relayout]); any variant wanting a data-dependent column offset into VMEM must express it as a DMA.
- **Every semaphore must be waited before exit; only a genuinely zero-size DMA is exempt** [verified: j-106c8a46, `zero_size_dma_needs_no_wait` / `undrained_semaphore_halts_core`]. The `sz=0` path in `_fetch_bq` relies on the exemption. An un-waited real DMA halts the core and poisons every later launch in the process — the failure mode most likely to be misread as infrastructure.
- **Triple buffering does not exist at a top-level grid `BlockSpec`** [verified: j-106c8a46, `buffer_count_cap_by_position`] — but this kernel is idiom 3 with a hand-rolled 2-deep toggle, so depth >2 *is* available to it. That is an asset for hiding a 256-iteration chain, and it is lost by a rewrite to plain `BlockSpec`s.
- **SparseCore working set, shown to fit:** stage-1 per-subcore = `resident` 256 KiB + `priv` 16.5 KiB + `glob` 1.1 KiB + `rowbuf` 8.1 KiB ≈ **281.7 KiB against the 480 KiB the compiler actually accepts** (not the 512 KiB reported, and not poolable across subcores) [hardware-facts §Not established, `08-sparsecore.md` is the authority]. `MAX_SLICE_WORDS = 64K` is what forces `p ≥ 16` at n=1M. Stage 2 at 32768 words = 128 KiB fits easily.
- **`_pick_partition` sizes `p` from `n`, never from the live length** (`:365`). At B-D4-256k, `p_min = cdiv(1048576, 65536) = 16` gives 256 slice-rows / 8 waves where the live 262144 would give `p=4`, 64 slice-rows, **2 waves** — and no dead slices, because dead slices are spread one-per-row and so cannot be skipped a wave at a time. This is Finding 2 restated as a partition decision, and it is where the 75% idle comes from.

### Working sets I cannot show to fit

None of the above. **Flagged instead:** hardware-facts measured "an int8 block needs ≥32 rows" for `BlockSpec` blocks and **float8 was never swept**; `bkv_x2_ref` is a `u8` scratch DMA'd into, not a blocked operand, so the rule should not apply — but a variant that converts KV to a pipelined `BlockSpec` inherits an unswept constraint.

### Gaps this block turns on

1. **"A grid-step ladder that re-reads the same resident block does not price a step" / the `512 ns + bytes/3.48e12` model is fitted on `pallas_call` grid steps.** This kernel's inner iteration is a hand-rolled `fori_loop` body with its own `make_async_copy` start+wait, and **the file does not say whether the 512 ns fixed term applies to it.** *Decision it settles:* whether the lever is fewer iterations (widen `bkv_p`) or fewer copies per iteration (`sbs × bkv_p` = 8 starts per fetch today). Under "fixed cost is per iteration", `bkv_p=16` at B-D4 cuts 256 → 32 iterations and, if the ~975 ns non-DMA residual of the measured 1258 ns/iteration is fixed, saves ~224 of 322 us of scoring (**modelled**, −20% of the full 1117.9 us). Under "fixed cost is per `make_async_copy`", widening `bkv_p` moves nothing — the copy count is unchanged — and the lever is instead coalescing contiguous pages into one copy. The measured 685 ns/block at B-D1-1M over 2 copies is ~340 ns/copy, uncomfortably close to the 512 ns grid-step term, which is why **this is the probe I would rank first.** Probe: hold bytes fixed, sweep `bkv_p ∈ {2,4,8,16}` and independently sweep copies-per-fetch, at B-D4-256k.
2. **"The block sweep does not extend past 1024 on any axis, so the shape at which VMEM turns the trend around was never found."** *Decision:* how far to widen. Under "monotone to VMEM", take the largest fit (16.3 MiB at B-D4, 16.8 MiB at B-P1024). Under "an interior optimum exists", stop at the measured knee. Same sweep answers both.
3. **Not turned on, per the two-question rule:** `dimension_semantics` (forced to `arbitrary` by carried state, and no Megacore to reward it), and the tokamax `k`-last deflation disagreement (our two block axes `bq_sz` and `bkv_sz` inflate `bo_scores` multiplicatively, so neither is the cheap one).

### Partitions enumerated and not favoured

- **Cut the head axis `h=32` across grid steps** — needs a cross-step `f32[T, bkv_sz]` accumulator, and `h×d` is 4 KiB/token which already fits; pure loss.
- **Cut the score-column axis `n` with a `BlockSpec` index map** — `page_indices` indirection is data-dependent, so no `index_map` describes the iteration; this is exactly idiom 3's admitting clause.
- **Cut across chips / a mesh** — N/A: indexer is replicated, TP=1, no mesh on the op (report §2). One die is the unit.
- **Label the seq-group axis `parallel` for Megacore** — N/A on tpu7x (`num_cores=1`), and forbidden anyway by carried scratch state.
- **One page per grid step (grid = total live pages)** — maximal grid extent, but 2048+ steps × 512 ns ≈ 1 ms at B-D1-1M and no way to carry the q block across steps.
- **Merge the three TC launches into one grid** — they differ in `static_q_len`, `bq_sz`, `seq_batch_size`, all trace-time constants; a merge forces one shape and reintroduces `bq_sz=128` for decode rows.
- **Rewrite to idiom 2 (`pl.kernel` + `emit_pipeline` + a `BufferedRef` subclass)** — the reference's named route for data-dependent offsets over a regular loop, gets the scheduler and possibly depth >2 for free. Not favoured only because it is a full rewrite of a working pipeline; it is a legitimate candidate and I am not ruling it out.
- **`bq_sz=512, bkv_p=4` at MIXED** — 21 MiB, 128 iterations, but 128 MiB of KV against 64; strictly dominated by `bq_sz=1024, bkv_p=2` unless the per-iteration term turns out to dominate the bandwidth term.
