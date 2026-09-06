# Area (7) constraint sheet — `streamindex_topk` / `sparsecore_topk` on GLM-5.2

**Scope, per `AREAS.md`:** scalar and index metadata plus grid-invariant tables only. Tensor recompute-vs-save is area (1); recomputing an activation is area (6). Tensor-shaped things I found and am **handing off, not deciding**: the `f32[T,n]` score matrix's HBM round trip → (1); `prepare_q_inputs`' 129→256 head-dim zero-pad → (4); the `n = max_model_len` structural fact behind Finding 2 → (2)/(6). I own only the *scalars that size them*.

**Level 2 (model load) is N/A here** — nothing in this block is weight-derived. On the record as a decision.

### Emission — one row per non-operand quantity

Values are `A-D64-8k / B-D4-256k / B-P1024-256k`. "×21" = rebuilt once per indexer layer per step. Denominators for the % column are `21 × measured` (7.63 ms / 23.5 ms / 1069 ms).

| quantity | invariant along | level today | level permitted | space, bytes | one rebuild × rebuilds/step → % of step |
|---|---|---|---|---|---|
| `page_indices` → HBM byte offset (`min/×/min`, in `_fetch_bkv`) | **layer (×21)** + whole decode step for a given seq | **5** (per grid step, per `bkv_p`, per batch member) | **3** — a flat `int32` byte-offset table *replacing* `page_indices`, same bytes | SMEM 2.25 KiB / **512 KiB** / 512 KiB | ~5 scalar ops ≈ 35 ns `[assumed]` × 67.2k / 215k / 430k → **6.1% / 6.3% / 0.28%** |
| `num_bkv = cdiv(max_i kv_len_i, bkv_sz)` | layer (×21); *not* grid — genuinely per 4-seq group | 5 (in-kernel, per grid step) | 3 — `int32[max_num_seqs/4]` table | SMEM 64 B | ~10 scalar ops ≈ 70 ns × 336 / 21 / 21 → 0.3% / <0.01% / <0.01% |
| `k_span`, validity + causal masks | `k_span` is iota-derived (compile-time const); masks depend on `bkv_idx` | 5 | 5 is correct; consume a **scalar prefix bound**, not a bool vector | registers | 38 / 19 / 152 ns × 1680 / 5376 / 43008 → **0.8% / 0.4% / 0.6%** |
| `jnp.full((T,·),-inf)` initializer | **layer (×21)** — all 21 layers dirty the identical (row, sublane) set | 3, per layer | **once per step** (fill once, alias 21×) | HBM 2.62 MB / 67.1 MB / **4.295 GB** | 0.71 / 18.1 / **1161 µs** × 21 → saving 20/21 = **0.2% / 1.5% / 2.2%** `[assumed: item 2]` |
| `_effective_row_lengths` (O(T×S)) | layer (×21) | 3, XLA | 3 — but move into `AttentionMetadata`, once per step | intermediate 16 KiB / 4 KiB / 256 KiB | 19 / 5 / 305 ns × 21 → **<0.001% everywhere** |
| `prepare_index_weights` pad + f32 cast | **nothing** — `weights` changes per layer | 3 | 3 — correct, **not a hoist candidate**; `num_q_heads=32` is already packing-aligned so the pad is a no-op | — | per-call, correctly |
| `distribution = stack([rd0,rd0,rd2])` | layer (×21) | 3, torch, per layer | once per step, in `AttentionMetadata` | 12 B | cost is not FLOPs — it is a **fresh tracer per layer, i.e. the CSE barrier** that makes every row above run 21× |
| `num_sublanes_total` (incl. dead index [1]) | run (static shapes) | **1 — already outermost** | 1 | sizes the HBM score buffer | at `(2,2,2)` all three entries are equal, so **the dead slot costs exactly zero today**; it is a latent coupling, not a live cost |
| ↳ `align(pages_per_seq, bkv_p)` padding | run | 1 | 1 | — | **(A) only: `align(9,2)=10` pads `n` 9216→10240, +11.1%**; `bkv_p ∈ {1,3,9}` divides 9 and removes it. Exact, free, level-1. Effect on SC time `unmodelled` (register item 11) |
| `_pick_partition(b,n,k)` → `p`, `slice_len`, `num_waves` | run (static) | **1 — correct, and cannot move** (`p` sets `scores_p`'s shape) | 1 | — | `p` = 1 / 16 / 16; `num_waves` = 2 / 8 / **512**. **The dominant cost is a correctly-placed level-1 quantity fed the wrong input (`n = max_model_len`). No level change fixes it — hand to (2)/(6).** |
| `lengths_p`, padded `lengths` | layer (×21) | 3 | 3, once per step | 256 B / 1 KiB / 64 KiB | 19 ns × 21 → negligible |
| `len_vmem` (whole `b_pad` on every subcore) | invocation | **4 — correct level, wrong size** | 4, sized by `num_waves` not `b_pad` | SC VMEM 256 B / 1 KiB / **64 KiB, of which 2 KiB is read** | 6.2 µs × 21 = 130 µs → 0.012%. **The cost is capacity, not time**: 64 KiB is 13% of the 480 KiB usable SC VMEM. Freeing it does *not* let `MAX_SLICE_WORDS` halve `p` (needs 512 KiB) |
| `sem_ids`, `bo_sz`, `[start,end]` | per launch | 3 | 3 | 28 B | mutable state, not invariants — correct |
| `bq_pos_compressed` | bkv loop; layer (×21) | 5, but already hoisted out of the inner loop | fine as is | 16 B / 16 B / 512 B | negligible |

### The two numbers that dominate my sheet

**1. 42 of 63 pallas launches per model step run a zero-extent grid.** At all three shapes exactly two of three launches are empty: (A) `D=64` → launches 2,3 empty; (B-D4) `distribution=[4,4,4]` → launches 2,3 empty; (B-P1024) `distribution=[0,0,1]` → launches 1,2 empty. Against the reference's empty-kernel floor of 19.7 µs `[verified: j-a35c902f]` / 31.9 µs `[verified: j-85d7b244]` — which **did not reproduce between jobs**, so treat it as a range — that is **0.83–1.34 ms per model step**, and at A-D64-8k it is **11–18% of the 363.2 µs measurement**. Whether a `grid=(0,)` Mosaic call short-circuits is `[assumed]` and binary. **This is the cheapest probe on the sheet and it should be run first.** The quantity ("is this range empty") is layer-invariant and available at level 3; the *spend* is a `lax.cond` skip, and the restructuring into fewer launches is (1)/(2)'s.

**2. Scalar-core page-index arithmetic is ~6% at both decode shapes** and is my area's share of the unattributed 685 ns/KV-block in report §6 item 3. At B-D1-1M the ~10 scalar ops per DMA setup ≈ 70 ns sit inside that 685 ns alongside 2 DMA issues and 5 `lax.select`s — plausibly a quarter to a half of it. `[assumed]`: the 6.9 ns/scalar figure is a *write* slope applied to arithmetic. The masks, by contrast, are **<1% at every shape** — they are not the missing term, and this is the row where quantifying beat assuming.

### SMEM budget at (B) — the binding constraint

`page_indices` int32[131072] = **524,288 B** + `seq_lens` 256 + `cu_q_lens` 260 + 28 B of control = **524,832 B = 512.5 KiB**, against the **1021.5 KiB** actually addressable `[verified: j-ed517191, smem_scratch_bisect]`. **50.2% consumed, 497 KiB headroom.** Consequence: a level-3 byte-offset table of equal size must **replace** `page_indices`, not join it — 512.5 + 512 = 1024.5 KiB **exceeds the measured ceiling**. At (A) the same total is 2.8 KiB, 0.3% — SMEM is not a constraint at (A) at all. At B-D4 only 4 of 64 sequences are live, so **32 KiB is read and 480 KiB is transferred and never touched**; ×63 launches = 31.5 MiB/step ≈ 8.5 µs `[modelled]`, two-thirds of it into the empty launches above.

### What area (7) forbids and requires here

- **Forbids** any hoist of `page_indices`-derived, `seq_lens`-derived or `cu_q_lens`-derived metadata past level 3. vLLM's batch shape and page table change every step; level 1 or 2 is a silent wrong answer.
- **Forbids** adding an SMEM table at (B) without displacing an existing one (arithmetic above).
- **Requires** that `p`, `slice_len` and `num_sublanes_total` stay at level 1 — they set static shapes and cannot legally move.
- **Requires** the static-max-plus-runtime-count pattern for anything new: allocate at the padded maximum, carry the true count separately.
- **Requires**, before any of the level-3 hoists are worth writing, an answer on whether the 21 layers CSE at all (below).

### `## Gaps` this block turns on — with the decision each settles

1. **"The CSE result did not isolate what it was meant to."** *Decides:* whether `_effective_row_lengths`, `lengths_p`, the `distribution` stack and `prepare_index_weights` run 1× or 21× per step. The reference's probe was 4 `pallas_call`s in **one** jit; here each layer is a **separately-named torch custom op** (`pallas::streamidx_topk_{prefix}`) built by its own `VllmTPUSparseAttnIndexer`, so cross-layer CSE is likely structurally unavailable. *Under "CSE holds":* do nothing, every level-3 hoist on this sheet is already free. *Under "it does not":* build all shape metadata once in `AttentionMetadata` and pass the *identical tensor* to all 21 layers. **Probe:** the one the gap already names — distinct weight operands, identical metadata — at 21 calls, not 4.
2. **"Where the dynamic grid's 26.2 µs fixed cost comes from is unknown — allocation or invocation."** *Decides:* the opposite change at (B). *Under "scales with allocation":* capping the 4 GiB score buffer is the lever and the 42 empty launches cost little. *Under "per invocation":* the empty-launch skip is the lever and the allocation matters only for HBM capacity (register item 4). This block has both a 4 GiB allocation and 63 launches at grid extents of 0–16, so it is exactly the discriminating case.
3. **"Narrow SMEM dtypes are accepted but not shown to save capacity."** *Decides:* uint16 `page_indices` at (B) — 512 KiB → 256 KiB, headroom 497 → 753 KiB, which is the difference between a replacement table and an additional one. *Probe:* re-run `smem_scratch_bisect` at int16. (Legality also needs `total_num_pages ≤ 65535`, currently unknown.)
4. **"`SmemWrapper` is quoted but never exercised."** *Decides:* whether the hand-strided 2-D SMEM index `page_indices_ref[(seq+b)*pages_per_seq + …]` — 40 scalar ops per inner iteration at (A) decode, inside the 685 ns/block budget — is as cheap as a flat 1-D read. *Under "as cheap":* leave it. *Under "not":* the flat byte-offset table is free in bytes and removes the stride multiply.
5. **"Nothing here measures metadata at a realistic vLLM decode shape."** This block *is* that shape: 512 KiB of scalar prefetch, 63 launches/step, a 21× multiplier. Every figure above resting on the 6.9 ns/scalar slope or the 19.7–31.9 µs floor is an extrapolation into a regime the file has never measured.

**Does not count:** the "recompute cost is a lower bound, possibly 5×" gap. `k_span` here is genuinely iota-derived, so the folding-favourable case applies, and I have costed the mask work at <1% at every shape — no decision turns on it.

### Staleness hazard per hoist (the counterpart risk, per quantity)

- **`-inf` buffer reused across layers:** safe *within* a step because all 21 layers dirty the identical write set; **fatal across steps.** A step with the same `T` but different `seq_lens` leaves last step's real scores where `-inf` is now required, and the top-k returns 2048 valid-looking indices. Must be keyed on the full `(seq_lens, cu_q_lens, distribution, T)` tuple, i.e. re-filled every step, never cached on `T`.
- **Page byte-offset table:** the page table changes every step as blocks are allocated and freed. Level 3 only. Cached on `pages_per_seq` it reads freed pages — garbage scores, no crash.
- **`num_bkv` per-group table:** depends on `seq_lens` *content*, not shape. A stale value under-runs the loop and silently drops whole KV blocks. Caught by the rewritten gate's **churn** metric — Finding 5 puts a dropped block at ~25% of the true top-k against the threshold, so this hazard is covered by an existing instrument. Say so when proposing it.
- **`eff_row_lengths` moved to per-step:** same hazard class; safe only if it lands where `AttentionMetadata` is already built.
- **Empty-launch skip:** `lax.cond` on a traced `distribution` cannot go stale. **Lowest-hazard, highest-value item on the sheet.**
- **`num_sublanes_total` dropping the dead index [1]:** no staleness, but re-enabling the prefill range later silently restores the coupling. Take the max over only the indices `_common_path` reads, and assert the dead slot is unused.

### Placements considered and not favoured — one line each, no choice made

- **Level 1 for `num_bkv` / page offsets** — depends on batch content, not static shape; this is the correctness bug the area warns about.
- **Level 2 (model load) for anything** — nothing here is weight-derived; genuinely N/A.
- **Level 4 (in-kernel SMEM scratch) for the page byte-offset table** — 131,072 entries at 6.9 ns/scalar ≈ 904 µs of serial scalar-core fill per invocation, far past the "few thousand entries" boundary; the reference's own step 3 sends this to level 3 because XLA parallelizes and the scalar core does not.
- **Level 4 for `num_bkv`** — only ~16 entries, so both levels measured equal (0.96×/1.07×); no reason to prefer it over level 3, which composes with the sorting Finding 3 wants.
- **`num_scalar_prefetch` driving a `BlockSpec` `index_map`** — the highest-value spend in the reference, but this kernel uses manual `make_async_copy` with `pl.ds(traced_offset, …)`, so reaching it is a pipeline restructuring owned by (2)/(3), not a placement I can take.
- **A `pltpu.BufferedRef` subclass reading offsets from the metadata ref** — same reason; it would replace the hand-rolled double-buffer, which is (3)'s call.
- **VMEM instead of SMEM for the offset table** — competes with the block sizes (2) already spent, and the reference measured the VMEM middle rung recovering only 13% of the gap.
- **Hoisting `_effective_row_lengths` for speed** — measured-out at <0.001% of the step at all three shapes; only worth moving as part of the `distribution` CSE-barrier fix, not on its own.
- **Sizing `len_vmem` by `num_waves`** — correct and cheap, but 0.012% of time; take it for the 62 KiB of SC VMEM, not for latency, and note it does not unlock a `p` halving.
- **Caching metadata at trace time with `functools.lru_cache`** (splash attention's level-1 idiom) — available only because splash's mask depends on static shape; here every candidate depends on batch content.
