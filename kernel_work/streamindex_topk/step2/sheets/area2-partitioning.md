## Constraint sheet — area (2), partitioning · block: `streamindex_topk` scoring half

Sources: `references/02-partitioning.md`; `kernel_work/streamindex_topk/report.md`; `streamindex_topk.py`, `mla_attention_op.py`, `instruments/shapes.py`. I choose nothing below.

---

### 0. The iteration space, enumerated (reference § Deciding, item 1)

| # | axis | extent, `decode_prod` / `mixed_prod` | where it lives now | origin |
|---|---|---|---|---|
| a | sequence-batch | 16 / 1 | **the Pallas grid** (`:643`) | derived (`streamindex_topk.py:643`) |
| b | seq-within-batch (`seq_batch_size`) | 4 / 1 | unrolled Python loop → einsum batch dim `b` | derived (`:118`, `:313-317`) |
| c | query block `bq` | 1 / 8 | `lax.fori_loop`, **outer** (`:420`) | derived |
| d | KV block `bkv` | 4 / 4 | `lax.fori_loop`, **inner** (`:418`) | derived |
| e | head `h` = 32 | 32 / 32 | **not an axis anywhere** — folded into einsum M via `reshape(bq_sz*num_q_heads, head_dim)` (`:244`), then reduced (`:335`) | derived |
| f | head_dim `d` = 128 | 1 block | the einsum's contraction, entirely inside one `jnp.dot` — the strongest form of "contraction innermost" | derived |
| g | MlaCase band | 3, of which 1–2 live | three separate `pallas_call`s (`:757/775/793`) | derived |
| h | layer | 78 | outside the kernel; 3 launches × 78 = **234 Pallas launches per model step** | derived (`shapes.py:45`) |

Axis (e) is the reference's named miss case — "an axis that exists in how the data is grouped but nowhere in the current loop". It is a reduction axis, so cutting it needs an accumulator; it is listed because it is the only untouched lever for freeing VMEM.

---

### 1. What this area forbids or requires at step 1's shapes

- **The grid is 1-D over sequence-batches and at every mixed regime its extent is 1.** `grid = ((end_seq_idx − start_seq_idx) // seq_batch_size,)` (`:643`); the mixed roofline rows price one sequence (`kv MB = 2.1` = 8192 × 256 B). So at mixed **Pallas schedules nothing** — all iteration is two `lax.fori_loop`s inside a single grid step, invisible to the pipeline. Every reference result about grid step count, `dimension_semantics` and traversal order applies to the *fori_loops*, which no Pallas machinery touches. Any partitioning change at mixed is an edit to loop bounds and `scratch_shapes`, not to a `BlockSpec` or an `index_map`.
- **No operand is pipelined.** All four `in_specs` and the `out_spec` are `memory_space=pltpu.HBM` (`:645-652`) with hand-rolled `make_async_copy`. This is reference idiom 1 + idiom 3 hybrid (`PrefetchScalarGridSpec` scalars, everything else raw DMA). Consequence: the budget rule `sum(block_bytes)*2 + scratch <= 31 MiB` collapses to **scratch only**.
- **`kv_packing = 4` must remain a whole axis of any bkv block.** `bkv_x2_ref` is `(2, seq_batch, bkv_buf_sz_per_kv_packing, 4, 256)`; its second-to-last extent 4 is not divisible by 8 and is legal only under the `or be equal to the respective dimensions of the overall array` clause [verified: j-ed517191-5f2d-4d00-8c0c, `native_tile_granularity`, tpu7x]. A variant that tiles that axis is **rejected at trace time**, not merely slow.
- **`seq_batch_size > 1` requires `static_q_len == 1`** (`:628-629`) — the decode batching axis is structurally unavailable to the mixed band. And `(end − start) % seq_batch_size == 0` is required (`:641-642`), which is exactly what forces `decode_batch_end = distribution[0]//4*4` (`:755`) and creates the remainder band.
- **`arbitrary` is forced on the grid axis** (`:721`) and correctly so: `sem_ids_ref`, `bo_sz_ref` and the double-buffer parity are read before written across grid steps — the reference's "scratch buffer read before it is written" case, where mislabelling is a correctness bug, not a perf one. There is nothing to win by relabelling: `is_megacore = False`, `num_cores = 1` [verified: j-4fba8dc6-d72c-41f9-948b, `tpu_info_reported_constants`].
- **`vmem_limit_bytes` is already set past the wall.** `DEFAULT_VMEM_LIMIT_BYTES = 100 MiB` (`:28`, `:535`) against `vmem_capacity_bytes = 67108864` [verified: j-4fba8dc6-d72c-41f9-948b]. Raising it further buys nothing; the real ceiling is ~64 MiB and the default scoped one is 31 MiB [verified: j-ed517191-5f2d-4d00-8c0c, `vmem_scratch_bisect`].
- **Measurement constraint.** `decode_prod` full-call is 167 µs (measured, j-6d675a58-7ecd-4b44-9171) against a host dispatch floor of median 149.0 µs [verified: j-ed517191-5f2d-4d00-8c0c, `dispatch_floor`]. Any decode variant must keep the harness's `inner` batching; a single `block_until_ready` at decode reports the host.

---

### 2. Tagged facts from `02-partitioning.md` that bear on this block

- Block shape is worth up to **7.86x**, best `1024³` vs worst `256³` [verified: j-ed517191-5f2d-4d00-8c0c, `matmul_block_size_sweep`]; **use the ranking, not the Tops/s column** — it is dispatch-floored ~1.6x low. Bigger is monotone better inside {256,512,1024}³; shrinking two axes costs multiplicatively, one axis additively. The decode einsum's M = 32 is **below every arm the sweep tested**, so the sweep cannot rank it.
- **VMEM scratch tops out just under 32 MiB**, `largest_ok = 31 MiB`, `first_fail = 32 MiB`, `CompileTimeScopedVmemOom` [verified: j-ed517191-5f2d-4d00-8c0c, `vmem_scratch_bisect`]; 48 MiB accepted under `vmem_limit_bytes=128 MiB` [verified: j-106c8a46-174c-4c03-a76a, `vmem_limit_bytes_raises_ceiling`]. Bytes/element track itemsize 1:2:4 with no floor, **int8 is one byte** [verified: j-106c8a46-174c-4c03-a76a, `vmem_bytes_per_element_by_dtype`] — the `u8` bkv buffer is priced correctly at 1 B/elem.
- **SMEM holds 261504 int32, 1021.5 KiB** [verified: j-ed517191-5f2d-4d00-8c0c, `smem_scratch_bisect`]. Relevant because the scalar prefetch carries `page_indices` at `i32[max_num_seqs * pages_per_seq]` — 64 × 66 = 4224 i32 at `decode_64k`, 1.6% of SMEM. Not binding.
- **CMEM is zero** [verified: j-4fba8dc6-d72c-41f9-948b].
- **A grid step costs `512 ns + bytes / 3.48e12 B/s`** [verified: j-ed517191-5f2d-4d00-8c0c], and **it is not a constant** — quote the model, never a single per-step figure. Operand count adds nothing: per-operand coefficient fits between −10.9 and +2.6 ns and changes sign [verified: j-054e7ac4-4099-4176-ad2b shard2].
- **Traversal order is worth 7%** for putting the reduction axis outermost [verified: j-ed517191-5f2d-4d00-8c0c, `grid_traversal_order`]; swapping two parallel axes moves 0.1%. Here `bq` and `bkv` are **both parallel** axes (the contraction is `d`, inside the dot), so the 7% result does *not* predict the `bq`/`bkv` swap — the 0.1% result is the closer analogue, and what actually moves is the re-read byte count (§4).
- **A ragged tail step costs 0.65 of a full step** [verified: j-050c7480-40a6-4a50-9e1b, `ragged_tail_cost`]. **This does not transfer to this block**: the reference's plausible mechanism is that the tail's DMA shrinks while the MXU still does the full block, and here `_fetch_bkv` always copies `bkv_p` whole pages with the page index clamped by `jnp.minimum` (`:179-185`) — nothing shrinks, so a partial bkv block should cost **1.0** of a step, not 0.65. Untested here.
- Empty work is skipped **three non-interchangeable ways** — compacted out of the grid at trace time, dynamic grid length, or `pl.when`/`lax.cond` per step. The three-band dispatch (`:757/775/793`) currently uses none of them at the *launch* level; the emptiness is expressed as a zero-extent grid.
- Idiom 2 (`pl.kernel` + `emit_pipeline`) is the reference's default and **the named route for "a ragged offset that will not go through a top-level `BlockSpec`"** — override `copy_in`/`copy_out` in a `BufferedRef` subclass, keep the index map trivial. The two schedulers tie at 0.99x on the same partition. Spellings that will bite: `pltpu.CompilerParams` not `TPUCompilerParams`, `pl.ANY` not `pltpu.ANY`, `pltpu.create_tensorcore_mesh("core")` not `pltpu.TensorCoreMesh`, and no `pl.load`/`pl.store`/`pl.swap` [verified: j-4fba8dc6-d72c-41f9-948b, `pltpu_symbol_surface`].
- **Start big and deflate**, `m` then `n` then `k`, floor 128, at `capacity/2 × 0.75` ≈ 24 MiB [source: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:272-380@9da2369`]. An estimator plus a descending search beats a hard-coded tile — directly applicable, since `(2,2,2)`/`(1,128,128)` are hard-coded constants at `mla_attention_op.py:119-120`.

---

### 3. `## Gaps` this block turns on, and the decision each would settle

1. **"The block sweep does not extend past 1024 on any axis, so the shape at which VMEM turns the monotone trend around was never found."** → Settles whether `bq_sz = 1024` at mixed (M = 1024×32 = 32768) is still on the monotone-better side, or past the turnaround. This is the largest single block-size candidate on the table (§5, B2) and the reference's evidence stops four doublings short of it.
2. **"`matmul_block_size_sweep`'s rates are dispatch-floored; only the ranking is quotable, and the true spread is known only to be ≥7.86x."** → Settles the *decode* question the sweep is silent on: the sweep's smallest arm is 256 and decode's M is **32**. Whether M=32 is on the same surface at all, or in a different regime below the MXU row cliff, decides whether the modelled 4× derate is an over- or under-estimate — and therefore whether any decode work should be spent chasing M.
3. **"Why the ragged tail is 0.65 of a step rather than 1.0 has not been isolated."** → Settles whether narrowing the last bkv block's DMA to its live pages (§5, C2) recovers anything. If the mechanism is "the DMA shrinks and the MXU does not", then this block — which never shrinks its DMA — is currently paying the full 1.0 and narrowing recovers the 0.35; if the mechanism is elsewhere, it recovers nothing.
4. **"A grid-step ladder that re-reads the same resident block does not price a step."** → A probe-design constraint on any block-size sweep written for this block: it must move different KV per iteration. `_fetch_bkv` naturally does, *except* in the one configuration where `bkv_p ≥ pages_per_seq` and `num_bq > 1`, where the same pages are re-fetched every bq trip — a sweep arm that lands there would measure the reference's named invalid pattern.

Not turned on: the operand-count gap (closed, and no pipelined operands here); the bytes-per-element gap (closed); the `k`-deflation disagreement (the contraction axis is `head_dim = 128`, fixed by the checkpoint and already at the 128 floor); the `dimension_semantics` gap (`arbitrary` is forced by carried state regardless of what the measurement would say).

---

### 4. `Emit` — grid · working set · invocations · HBM traffic · idle fraction

**Grid** (incumbent, per band, per layer, per call):

| band | source | grid extent, `decode_prod` | `mixed_prod` | `decode_64k` | `seq_batch_size` | `bq_sz` | `bkv_p` |
|---|---|---:|---:|---:|---:|---:|---:|
| DECODE batched | `:757` | **16** | 0 | **1** | 4 | 1 | 2 |
| DECODE remainder | `:775` | 0 | 0 | **2** | 1 | 1 | 2 |
| MIXED | `:793` | 0 | **1** | 0 | 1 | 128 | 2 |

Inner (`fori_loop`) trip counts per grid step: `num_bq × num_bkv` = 1×4 at `decode_prod`, 8×4 at `mixed_prod`, 8×32 at `mixed_64k` [modelled, `report.md` regime table]. Grid axis labelled `arbitrary`.

**Working set against capacity** (derived from `streamindex_topk.py:654-692`; `u8` cache, `f8e4m3` q, `f32` weights and scores):

| scratch | decode band (sb=4, bq=1, bkv_p=2) | mixed band (sb=1, bq=128, bkv_p=2) |
|---|---:|---:|
| `bkv_x2` | 4.000 MiB | 1.000 MiB |
| `bq_x2` | 32 KiB | 1.000 MiB |
| `bq_weights_x2` | 1.0 KiB | 32 KiB |
| `scores_block_x2` | 64 KiB | 2.000 MiB |
| **total** | **4.095 MiB** | **4.031 MiB** |
| against 31 MiB default scoped budget | **13.2%** | **13.0%** |
| against the 64 MiB wall | 6.4% | 6.3% |

Pipelined-operand contribution: **zero** (all `pltpu.HBM`). Growth law, derived: decode total ≈ `bkv_p × 2.031 MiB + 33 KiB`; mixed total ≈ `bkv_p × 0.5 MiB + bq_sz × (8448 B + 8192 B × bkv_p)`. Filled cells for the candidates are in §5. **The block is nowhere near capacity-bound today** — that is the single load-bearing number in this section, and it means the reference's "squeezed from both sides" crux is currently squeezed from one side only.

**Kernel invocations per step:** **3 Pallas launches per call per layer**, unconditionally (`:757/775/793`), of which **1 is live at `decode_prod` and at `mixed_prod`** and 2 at `decode_64k`. × 78 layers = **234 launches per model step, of which 78 do work at the production decode shape**. Plus, per call and outside the Pallas launches: one `jnp.full(-inf)` full-width fill (`:745`), one reshape/pad (`:811-817`), `_effective_row_lengths` (`:823`), and one `sparsecore_topk` (`:831`). Cost of a zero-extent-grid launch: **unmodelled** — `report.md`'s roofline has no launch term, and the reference's 149.0 µs floor is a *host* round trip, not a device-side custom call inside a jitted program.

**Total HBM traffic implied** (component ledger derived from `report.md`'s roofline inputs; the totals in bold are `report.md`'s, modelled):

| crossing | `decode_prod` | `mixed_prod` | `mixed_64k` |
|---|---:|---:|---:|
| `cache_kv` read × `num_bq` | 134.2 MB × 1 | 2.1 MB × 8 = 16.8 MB | 16.8 MB × 8 = 134.4 MB |
| — of which used bytes (129/256) | 67.6 MB | 8.5 MB | 67.7 MB |
| `q` read | 0.26 MB | 4.19 MB | 4.19 MB |
| `indexer_weights` read | 0.008 MB | 0.13 MB | 0.13 MB |
| scores `-inf` init write | 2.6 MB | 41.9 MB | 276.8 MB |
| scores kernel write | 2.1 MB | 33.5 MB | 268.4 MB |
| scores read by `sparsecore_topk` | 2.6 MB | 41.9 MB | 276.8 MB |
| **`hbm_st` (this loop structure)** | **42.5 µs** | **46.2 µs** | **290.5 µs** |
| **`hbm_id` (each byte once)** | **41.7 µs** | **29.3 µs** | **173.3 µs** |

The `hbm_id`/`hbm_st` gap is **0.8 µs at decode and 16.9 µs at mixed** [modelled, `report.md`] — the KV re-read is a mixed-phase cost, essentially nil at decode where `num_bq = 1`.

**A closed-form the loop-order decision turns on** (derived): with query block `B` and KV block `K` positions, the KV re-read costs `kv_len × 256 × (ntok/B)` bytes and the mirror-image `bkv`-outer order costs `ntok × 4096 × (kv_len/K)`. Their ratio is **`K / (16B)`**. At the shipped `K = 2048, B = 128` this is exactly **1.000** — swapping the loop order at mixed moves **zero** bytes. What moves bytes is raising `B` (fewer KV trips) or raising `K` *and* hoisting the fetch out of the `bq` loop, which the current structure does not do: `start_fetch_bkv` re-issues the same pages on every `bq` trip even when `num_bkv == 1`.

**Idle fraction:**

| source of idleness | `decode_prod` | `mixed_prod` | `mixed_64k` | `decode_64k` | origin |
|---|---:|---:|---:|---:|---|
| MXU rows unused (M=32 into a 128-row tile) | **75%** | 0% | 0% | **75%** | modelled (`report.md`, `derate` = 4) |
| launches spanning zero sequences | 2 of 3 | 2 of 3 | 2 of 3 | **1 of 3** | derived (`:755-809`, `mla_attention_op.py:183`) |
| scores sublanes never written by the kernel (`max_model_len` padding) | 16 of 80 = 20% | 20% | 16 of 528 = 3.0% | 3.0% | derived (`shapes.py:110-121`) |
| decode sequences demoted to the `sb=1` remainder band | 0 of 64 | n/a | n/a | **2 of 6** | derived (`:755`) |
| last `bkv` block partial | 0 | 0 | 0 | 0 | derived (all sweep `kv_len` are exact multiples of 2048) |
| scores computed then masked to `-inf` (causal + validity) | 0 | ~6% | ~0.8% | 0 | derived (`:340-352`) |
| `kv_len` skew inside a decode group of 4 (`num_bkv` from `jnp.max`) | **0 in the instruments; unmeasured in production** | n/a | n/a | 0 | derived (`:275-276`); the TODO at `:751-754` names it |

Cells I cannot fill without choosing: the per-launch cost of a zero-extent-grid Pallas call (`unmodelled`); the production `kv_len` skew (the instruments give uniform lengths, so it is `0` by construction and the real value is unmeasured); and whether the tail `bkv` block should be scored at 0.65 or 1.0 of a step (see gap 3).

A bound worth carrying, derived: `decode_prod` issues ≈704 hand-rolled DMAs per call (512 KV + 64 q + 64 weights + 64 scores-out). At the reference's 512 ns fixed term that is 360 µs, against a measured scope of 76.42 µs — **4.7× more than the whole measurement**, so either the per-DMA fixed cost overlaps almost completely or it is far below 512 ns. `report.md` asserts the former as an assumption; the reference's 512 ns was fitted for **grid steps**, and applying it per `make_async_copy` is an extrapolation the reference does not license either way.

---

### 5. Alternatives this area could take

**Grid shape**
- **A1** Keep the 1-D sequence-batch grid with both loops inside the body (incumbent; Pallas schedules nothing at mixed).
- **A2** Lift `bq` onto the grid: `grid = (num_seq_batches, num_bq)`, giving the pipeline something to schedule at mixed.
- **A3** Lift both: `grid = (seq, bq, bkv)`, `arbitrary` on `bkv`, with the scores accumulator in scratch.
- **A4** Swap to `bkv` outer / `bq` inner — byte-neutral at the shipped sizes (ratio `K/16B` = 1), non-neutral at any other.
- **A5** Two-level tile: an outer `bkv` group held resident across a full inner `bq` sweep — one KV read total, the only variant that removes the re-read without growing `bq_sz`.
- **A6** Cut the head axis (32) onto a loop with a partial-sum accumulator, trading VPU/accumulator work for a larger `bkv` or `bq`.
- **A7** Collapse the three bands into one launch driven by a scalar-prefetched band table — removes the two empty launches per layer per call.
- **A8** Keep three launches but compact the empty ones away — trace-time when `distribution` is static, `lax.cond` otherwise; the reference names these as three *non-interchangeable* mechanisms.
- **A9** Fuse across the layer axis — rejected here with a reason: `q`, `indexer_weights` and `cache_kv` all differ per layer.
- **A10** Move from the idiom-1+3 hybrid to idiom 2 (`pl.kernel` + `emit_pipeline` + a `BufferedRef` subclass), keeping the page gather hand-written while the pipeline owns the schedule.

**Block sizes** (working sets derived from `:654-692`)
- **B1** Decode `bkv_p` 2 → 4 / 8 / 9: `num_bkv` 4 → 2 / 1 / 1; working set 4.1 → 8.2 / 16.3 / 18.3 MiB.
- **B2** Mixed `bq_sz` 128 → 256 / 512 / 1024: `num_bq` 8 → 4 / 2 / 1; working set 4.03 → 5.6 / 13.1 / **25.25** MiB — the last fits under 31 MiB with no headroom, which the reference explicitly warns about.
- **B3** Decode `bkv_p` 2 → 1: halves the bkv buffer, doubles `num_bkv`, and **narrows `num_sublanes_total` from 80 to 72** at `max_model_len = 9216`, shrinking every scores crossing and the `sparsecore_topk` input width.
- **B4** `seq_batch_size` 4 → 8 / 16: grid 16 → 8 / 4, bkv buffer 4 → 8 / 16 MiB, more `max(kv_lens)` skew exposure.
- **B5** `seq_batch_size` 4 → 2: less skew idle, more grid steps, smaller remainder band.
- **B6** Decouple the three bands' `bkv_p` — with the caveat that `num_sublanes_total` takes the **max** over all three (`:603-605`), so one band's larger `bkv_p` widens the scores buffer for every band.
- **B7** Replace the hard-coded `(2,2,2)`/`(1,128,128)` with an estimator + descending search, per the reference's deflate-until-it-fits.

**Raggedness and idle**
- **C1** Sort decode sequences by `kv_len` before grouping into fours (the TODO at `:751-754`).
- **C2** Narrow the last `bkv` block's DMA to its live pages (turns on gap 3).
- **C3** Bound the `bkv` range per `bq` block by the causal frontier at mixed, rather than masking to `-inf` after computing.
- **C4** Size `num_sublanes_total` from the batch's actual max `kv_len` instead of `max_model_len`.
- **C5** Drop the `-inf` init fill and have the kernel (or `row_lengths`) cover the unwritten sublanes.

**API / semantics**
- **D1** Keep `arbitrary` (forced by carried SMEM state; nothing to win with one core).
- **D2** Set `vmem_limit_bytes` to a number that exists — 31 MiB to enforce the default budget, ~64 MiB to genuinely reach the wall — rather than the current nominal 100 MiB.

**Caller-side, outside this area's authority**
- **E1** Speculative decode with `static_q_len = γ+1` raises decode M from 32 to 32(γ+1) and removes the 4× derate at γ+1 ≥ 4; the kernel already has the TODO (`:626-627`), and the reference's row-cliff evidence sits at 128.

---

### Disagreements, carried rather than reconciled

1. **`report.md`'s regime table lists `nseq = 64` for `mixed_prod` while its roofline prices `kv MB = 2.1`** — one sequence's 8192 positions × 256 B. The mixed band's grid extent is `distribution[2] − distribution[1]`, the *real* sequence count, so the two readings give **grid = 64** or **grid = 1** and imply materially different partitions (and different KV totals). `shapes.py:155` documents `max_num_seqs` as "padded lane count, not real occupancy", which favours grid = 1, but the report's column is headed `nseq`. A briefer cannot settle this; the mixed grid extent must be read back before any mixed variant is sized.
2. **`report.md`: "at any one regime two of them span zero sequences".** True at `decode_prod` and `mixed_prod`; **false at `decode_64k`**, where `decode_batch_end = 6//4*4 = 4` puts 4 sequences in the batched band and 2 in the remainder band, leaving only one launch empty.
3. **`report.md` charges "the 512 ns per-DMA fixed cost … fully hidden by double buffering".** The reference's 512 ns is a per **grid step** term fitted on pipelined blocks [verified: j-ed517191-5f2d-4d00-8c0c]; this kernel has 16 or 1 grid steps and issues ~704 or ~112 raw DMAs. Importing the grid-step term onto `make_async_copy` is an extrapolation the reference does not license, in either direction.
4. **The reference's 0.65-of-a-step ragged tail does not apply here** (§2), because `_fetch_bkv` never narrows the tail DMA. Treat a partial `bkv` block as a full step until measured.
