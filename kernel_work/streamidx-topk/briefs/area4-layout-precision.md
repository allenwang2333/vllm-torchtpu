# Area 4 — Layout & precision: constraint sheet for `streamindex_topk`

Step-2 briefer output. **This sheet decides nothing.** It is a constraint sheet for the single
downstream designer, who reads it from disk. Block under study: the two-stage GLM-5.2 DSA
"lightning indexer" sparse-attention top-k, reached via `_streamidx_topk_jax` in
`mla_attention_op.py`. Stage A = three TensorCore Pallas scoring kernels writing f32 scores to HBM;
stage B = `sparsecore_topk`, an exact MSB-first radix select on SparseCore.

Baseline of record is the **nightly** config (`max_model_len` 9216, page 1024, TP1×DP16, score
width 10240). AgentX is a second operating point and is explicitly **not** the target.

## Tag vocabulary

| Tag | Meaning |
| --- | --- |
| `measured (job id)` | a number produced on hardware by a named job |
| `modelled` | arithmetic over shapes and rates; never a measurement |
| `assumed` | believed, with no artifact behind it |
| `[ref]` | from `.claude/skills/write-tpu-kernels/references/04-layout-precision.md` |
| `src(path:line)` | read out of the pinned source tree during this brief |

`src(...)` is a **fifth tag, added deliberately.** Facts read directly out of the code that will
compile and run are not `assumed` — nothing is being believed on faith, the shape is right there in
the file — but they are also not `measured`, because no job executed them. Tagging them `assumed`
would understate them and let a designer discount a shape that is simply true; tagging them
`measured` would launder source-reading into evidence. They get their own tag, and a `src(...)`
fact is falsifiable the cheap way: open the file at that line.

---

## A. Constraints

**A1. The 49.6% K-cache padding is forced by `unsigned_align_to(kv_dim, 128)`, and
`head_size_is_packed_width=True` does not escape it.** `align(129,128) = 256` by the direct route
too. The DS-MLA packed-width detour (`65 + 128 + 1 = 194 → 256`) coincidentally lands on the same
number. Narrowing the record therefore requires getting `kv_dim` to **128 or below**, not "getting
the formula right". `src(attention.py:1219-1264, mla/v2/kernel.py:83-106)`

**A2. The indexer K cache is NOT shared with the MLA v2 attention kernel at this configuration.**
This contradicts report §7/§8-finding-4 and is my largest correction — see §D1, and see
§"D1's falsifier" below for how to disprove it. Each layer name gets its own
`torch.zeros(kv_cache_shape)` `src(runner/tpu_runner.py:6316-6352)`; GLM's MLA attention layers take
the `use_sparse` branch and receive a **tuple of split (nope, rope) uint8 caches** from
`get_sparse_kv_cache_specs`, never the 4-D `cache_kv`
`src(tpu_runner.py:6319-6332, attention.py:1514-1560)`. Corroborated by the profile:
`sc_dsa_gather` ×78 is the split-cache gather `measured (j-a71a648c)` — the arithmetic behind why
**78** is the number that corroborates rather than the indexer's 21 is spelled out in
§"Why `sc_dsa_gather` ×78 is the corroborating count" below. What is shared is the *shape function*,
not the buffer. The indexer's `cache_kv` has exactly **one reader** (stage A) and **one writer**
(the XLA scatter at `mla_attention_op.py:103-104`). Repacking is a single-consumer change plus a
shape-function override — expensive, but the "two-consumer" objection is void.

**A3. `kv_packing = 4` is not free to change: it *is* the uint8 sublane tile.** The XLA layout probe
reports tiling `((4,128),(4,1))` on the 4-D uint8 array `measured (j-8b1e7acc)`, and 4 uint8 pack
into one 32-bit sublane word. Any repack must keep a 4-deep `kv_packing` axis, or the
`(512,4,256) → (2048,256)` collapse in `load_bkv` stops being free. `src(streamindex_topk.py:264)`

**A4. A repack to 128-wide values + separate scales gives a *perfectly* aligned value array.**
`u8[pages, page/4, 4, 128]` is exactly one lane tile, zero padding, and the existing 256 KiB
contiguous slab DMA becomes a 128 KiB contiguous slab DMA — still far above the bandwidth knee.
`modelled`. The scale array is the hard part: 1 byte per position, and the kernel needs it as a
**lane** vector `(1, bkv_sz)` (see A9).

**A5. Do not expect the full 2× from halving `lkv_dim`.** `[ref]`'s own dtype table: bf16→bf16 moves
0.50 of the bytes for **0.74×** the time; int8→bf16 moves 0.38 for 0.70×. Report §4's
`lkv_dim=129` column takes the full saving (47.5 → 25.0 µs) and is the **optimistic end**. The
measured sublinearity is per-block fixed cost, which is weaker here (256 KiB → 128 KiB, both above
the knee) but not zero. Budget 47.5 → **30–35 µs**, and note that at 39.0 µs `modelled` the VPU term
becomes binding first, so the realizable decode win from repacking alone is ≈ **47.5 → 39** µs of
floor, not 47.5 → 25.

**A6. DMA sizes: only the KV fetch is above the knee.** Decode, per call `src` + `modelled`:
KV 640 × 256 KiB ✓; score write-back 80 × **32 KiB** (exactly at the knee `[ref]`: "per-step fixed
cost dominates everything below ~32 KiB"); q 64 × **4 KiB**; indexer_weights 64 × **128 B**.
`[ref]` Deciding #1 says "make blocks ≥ 2 MiB". ~848 DMAs in a 221 µs call
`measured (j-8b1e7acc)` is a **260 ns/DMA** budget. The weights fetch in particular should be folded
into the q fetch or hoisted per-grid-step. Splitting the K record into two arrays **adds 640 more
small DMAs** unless the scales are batched per grid step — cost this before adopting A4.

**A7. Prefill's f32 accumulator is 32 MiB — exactly the default VMEM scope.**
`src(streamindex_topk.py:326-331)` with `bq_sz=128, H=32, bkv_sz=2048`: the einsum output is
`f32[1, 4096, 2048]` = **33.55 MB = 32.0 MiB**, on top of 4.03 MiB of declared scratch, under a
`vmem_limit_bytes` of **100 MiB** (`:28`, above the 64 MiB physical, so the compiler will not reject
it). **This is not in report.md anywhere** and it is a layout fact, not a VPU fact: §4's 586 µs
"VPU-bound" prefill term may be spill traffic. Note §7's VPU probe reached the same conclusion by a
different route ("bound by moving the tile, not by VPU throughput"). Decode is fine —
`f32[4,32,2048]` = 1 MiB. `modelled`, from source arithmetic; **flag as the single most probe-worthy
item in my area for prefill.** **Area 2 derived the same 33.55 MB independently — see D5.**

**A8. The score buffer's precision is not a bandwidth lever at either phase, and this is
counter-intuitive enough to state.** At decode the score round trip is 4.5% of HBM `modelled`;
halving f32→bf16 buys ~2% of decode traffic. At prefill it is 83% of HBM but prefill is not
HBM-bound (`t_hbm` 40.8 vs floor 623 `modelled`). Narrowing scores may still be a **stage-B** lever
(the radix select is 4 passes of 8-bit digits over a monotone-mapped f32 key
`src(sparsecore_topk.py:52-55)`; a 16-bit key is 2 passes) — that belongs to whoever owns stage B,
and it is an approximation with an accuracy gate, since the oracle compares index sets at zero
tolerance `measured (j-8b1e7acc, 13 passed)`.

**A9. `scale_val` costs an in-kernel transpose 320× per decode call.**
`pltpu.bitcast(flat_bkv[:, 128:129].T, e8m0)` `src(:267-268)` is a `(2048,1) → (1,2048)`
sublane→lane relayout, executed once per (grid step × bkv × seq) = 16×5×4 = 320 at decode, 40 at
prefill. `[ref]` measured only the **RHS** transpose (0.96×, no difference) and explicitly says
"treat an LHS transpose as expensive until measured". Unmeasured cost, and it is the thing a
pre-transposed scale array would remove.

**A10. Mask, do not pad — the incumbent is already right and must not be "fixed".** `[ref]`: mask
1.01×, pad 1.03×. The kernel masks at `:344-352` and never narrows a block. Two sub-rules the
designer must preserve: (a) the mask is built with rank-3 integer broadcasts, never a
`(BT,) → (BT,1)` bool reshape, which `[ref]` records as not lowering in Mosaic; (b) **the mask is
applied *last*, after the `× scale_val` multiply** `src(:337-352)` — folding it earlier risks
`0 × inf = NaN`, and `sparsecore_topk` states "NaNs are unsupported" `src(sparsecore_topk.py:29)`.

**A11. `-inf` is load-bearing here; `[ref]`'s `-0.7 × f32max` rule does not apply.** See §D2, which
states this as a **prohibition** rather than a preference: substituting a finite mask value is a
correctness bug, not a style regression.

**A12. The MXU precision rules the brief names do not bind this block's matmul, but they set the
trap.** The einsum is **fp8 × fp8 → f32** (`preferred_element_type=jnp.float32`) `src(:326-331)`, so
the `DEFAULT → BF16_BF16_F32` decomposition and the ~4.5× `HIGHEST` cliff (661.4 → 147.5 Tops/s
`[ref]`) are N/A **as written**. They bind the moment a candidate upcasts K to bf16 to dodge the fp8
unpack: that would cost 2× the HBM bytes *and* 1.86× the MXU rate. **int8 is doubly excluded**:
`[ref]` measures it at 746.8 Tops/s ≈ bf16 (no gain), it is absent from the reported native matmul
pair list, and int8 *elementwise* arithmetic does not lower in Mosaic at all ("Only vector<i16> and
vector<i32> are supported"). Storage-only int8 is fine — that is what `cache_kv` already is.

**A13. The MXU is 25% fed at decode by a layout choice.** LHS is `(bq_sz × H, D) = (32, 128)` at
decode `src(:243-244)` — 32 rows into a 128×128 MXU. `bq_sz=1` × `H=32` is the cause. Area 5's
number, but the lever is layout: stacking 4 sequences into the *contraction-free* batch axis
(`seq_batch_size=4`) does not fill MXU rows; folding them into the `n` axis would.

**A14. `prepare_index_weights` writes an f32 `[T,32]` array that the producer had in bf16.**
`src(mla_attention_op.py:277-280 → streamindex_topk.py:463)`. 8 KiB at decode — immaterial to
bandwidth, but it is a real HBM round trip and the kernel casts to f32 in-register at `:334` anyway.
Free to delete. Likewise verify XLA elides `prepare_q_inputs`' zero-width `jnp.pad` at GLM's
`(32, 128)` — it is a no-op in shape but a 4 MiB materialization at prefill if it is not folded.
**Unverified.**

---

## B. `Emit …` — precision and layout of every operand, and wasted fraction

Nightly decode, T=64, 64 seqs × 1 tok, S=8704, page 1024, `bkv_p=2`, `bkv_sz=2048`,
`num_positions=10240`, `seq_batch_size=4`, `bq_sz=1`. All byte figures `modelled` from `src`.

| # | operand | dir | HBM dtype / shape | movement dtype | accum dtype | tiling / minor dim | DMA unit | bytes/call | **wasted** | waste cause |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `cache_kv` | **R** | `u8[P, 256, 4, 256]` | u8 slab → VMEM, bitcast fp8_e4m3fn | — | minor 256 = 2 lane tiles; `((4,128),(4,1))` | 256 KiB ×640 | **167.8 MB** | **49.6%** (83.2 MB) | 127 pad bytes/record (A1) |
| 1b | ↳ scale byte | R | byte 128 of each record | u8 → e8m0 → **bf16** | — | needs `(1, bkv_sz)` lane vector | (in 1) | — | — | transpose A9 |
| 2 | `q` | R | `f8_e4m3fn[64, 32, 128]` | fp8 | — | minor 128 = 1 lane tile, exact | **4 KiB** ×64 | 256 KiB | **0%** (shape); **sub-knee DMA** | A6 |
| 3 | `indexer_weights` | R | `f32[64, 32]` | f32 → f32 | — | minor 32 = **¼ lane tile** | **128 B** ×64 | 8 KiB | 0% bytes; **75% of a lane tile idle**; f32 vs bf16 producer (A14) | A6, A14 |
| 4 | `scores_init` | **W** | `f32[64, 80, 128]` | f32 | — | minor 128 exact | one XLA broadcast | 2.62 MB | **≈100% redundant at ctx 8704** — `num_bkv×bkv_sz = 10240 = num_positions`, so stage A overwrites every sublane; only unvisited *token rows* need it | see note |
| 5 | `scores` out | **W** | `f32[64, 80, 128]` (aliased to 4) | f32 | f32 (MXU) | minor 128; write is `[4 tok, 16 subl, 128]` | **32 KiB** ×80 | 2.62 MB | **≥15%** (1536 of 10240 cols past `kv_len`) **+ intra-group** (every seq writes to the group-of-4 max) | `num_positions` padding + `kv_len_max` |
| 6 | `scores` → stage B | **R** | `f32[64, 10240]` (bitcast i32) | f32 bits → i32 monotone key | i32 | reshape `[64,80,128]→[64,10240]` is free (row-major) | full buffer | 2.62 MB | **≥15%**, same cause; `row_lengths` is passed but the buffer is `max_model_len`-wide | A8 |
| 7 | `k_packed` | **W** | `u8[64, 129]` **padded to `[64,256]`** at `:88-89`, then scattered | u8 | — | scatter, 64 rows | XLA scatter | 16 KiB | **49.6%** (8 KiB) | pads to match A1 |
| 8 | `topk_idxs` | W | `i32[64, 2048]` | i32 | — | minor 2048 | — | 512 KiB | 0% at ctx 8704 (all rows ≥ k); ~100% `-1` at short ctx | — |
| — | **VMEM `bkv_x2`** | scratch | `u8[2,4,512,4,256]` | — | — | — | — | **4.00 MiB** | **1.98 MiB is padding** | A1 — the padding costs VMEM twice over |
| — | VMEM `bq_x2` / `bq_w_x2` / `scores_block_x2` | scratch | fp8 / f32 / f32 | — | f32 | — | — | 32 KiB / 1 KiB / 64 KiB | — | — |
| — | **einsum accumulator** | reg/VMEM | `f32[4,32,2048]` decode; **`f32[1,4096,2048]` = 32.0 MiB prefill** | — | **f32** | — | — | 1 MiB / **33.55 MB** | — | **A7 — at the VMEM scope wall at prefill** |

Prefill deltas (T=1024, `bq_sz=128`, `num_bq=8`, `seq_batch_size=1`): row 1 → 21.0 MB, re-read 8×,
same 49.6%; rows 4/5/6 → 41.9 MB each, 125.8 MB total, waste ≥5.6% (9216 of 10240) + causal (a `bq`
block at position *p* has ~`p/S` of its row live, so the *causal* waste across a full prefill is
≈50% of the score buffer, which no mask recovers because the buffer is written full-width); row 5's
DMA unit becomes 1 MiB.

Note on row 4: `scores_init` is fully redundant *per sublane* whenever
`cdiv(kv_len_max, bkv_sz) × bkv_sz ≥ num_positions`. It is **not** removable wholesale — padded
token rows past `cu_q_lens[num_seqs]` and rows owned by the third (mixed) kernel call are never
touched by stage A. Profile puts the in-model cost at **6.02 µs, 2.79 TB/s, ~2% of the call**
`measured (j-a71a648c)`, so this is a correctness-relevant note, not a lever. Report §8 already
closed it.

---

## C. `## Gaps` in `04-layout-precision.md` that this block turns on

**C1. "No measurement of any layout change *across* a kernel boundary."** — Turns on it directly;
this is the 49.6% question in one line. **Decision:** *(a)* repack the cache to a 128-wide value
array + a separate scale array (A4) — requires overriding the shared shape function, splitting the
XLA scatter, and splitting `torch.cat([k_quant, k_scale])` at `mla_attention_op.py:165`; versus
*(b)* **movement-only**: leave the 256-wide record alone and have `_fetch_bkv` DMA `[..., :129]` (or
`[..., :128]` plus a batched scale fetch), a strided read of 129-of-256 bytes with zero change to
the allocation, the writer, or any other code. **(b) is by far the cheapest if HBM charges you the
129 and not the 256** — and that is exactly what this gap does not know. If the answer is
"charged 256", (b) is worthless and (a) is the only route; if "charged ~160" (32 B granule), (b)
recovers 37% of the dominant term for one line of code. **This is the highest-value probe in my area
and it is cheap: two `pallas_call`s over the production cache shape, one full-minor DMA and one
`[:129]` DMA, timed.**

**C2. "The LHS transpose was never measured — only the RHS."** — Turns on it via A9. **Decision:**
keep the scale byte inside the record and pay a `(2048,1)→(1,2048)` relayout 320× per call, versus
store the scales in a separate pre-transposed array (`u8[pages, 1, page_size]`, minor dim = page
size, loads as a lane vector with no relayout). Under "the transpose is free" the scale stays
in-record and C1(a) simplifies to a value-only repack; under "expensive" the pre-transposed scale
array is *required*, and it happens to be the same change that removes the padding — the two levers
collapse into one.

**C3. "The int8/int4 rows of the reported sublane-tiling table cannot be tested."** — Weakly turned
on. The incumbent already proves uint8 **storage and DMA** work at a 4-deep sublane axis, so the gap
is closed for everything this design does; it stays open only for uint8 *arithmetic*, which the
kernel never performs (it bitcasts to fp8 at `:266` before touching the values). **No decision hangs
on it.** Listed so the designer does not re-probe it.

Not turned on: the `>100%`-spread arms, int4 arithmetic, `pltpu.bitcast` on sub-byte types.

---

## D. Where `04-layout-precision.md` and `report.md` contradict each other

**D1. Report §7/§8-finding-4: "the cache is not the indexer's to reshape — it is the same `cache_kv`
the MLA v2 attention kernel reads."** **I believe this is wrong at this configuration**, and it is
not a reference-vs-report conflict but a report-vs-source one. Evidence in A2: separate tensors per
layer name; GLM's MLA attention takes the `use_sparse` split (nope, rope) branch and never sees the
4-D `cache_kv`; the profile's 78 `sc_dsa_gather` confirms the split path is live. The statement
would be correct on the *dense* MLA path, which is presumably where it came from. **This matters
because it is the objection that downgraded the padding from "reclaimable" to "a two-consumer
packing change".** It is still not free — the shape function is shared code and the XLA scatter
writer must change — but the designer should cost it as a **one-consumer** change. **Do not lean on
it until it has been run against the falsifier in the next section.**

**D2. `[ref]` "the mask *value* is a finite large negative, never `-inf`"
(`DEFAULT_MASK_VALUE = -0.7 × f32max`) vs the incumbent's `-jnp.inf` at `:352` and `:747`.**
**I believe the incumbent.** The reference's rule is scoped to *pre-softmax* masking — its stated
reason is "`-inf` produces NaN under softmax rescaling", and there is no softmax in this block.

> ### ⛔ PROHIBITION — for the step-3 author, not a style note
>
> **Do not replace `-jnp.inf` with `DEFAULT_MASK_VALUE`, `-0.7 × f32max`, `-1e30`, `finfo.min`, or
> any other finite sentinel, anywhere in this block** — not in the `jnp.where` mask at
> `streamindex_topk.py:352`, not in `scores_init` at `:745-749`, not in the `k`-padding fill at
> `:813-817`.
>
> `sparsecore_topk`'s contract is that **`-inf` scores are never selected**
> `src(sparsecore_topk.py:27-29)`. A finite sentinel is a perfectly ordinary float to an MSB-first
> radix select: it is **selectable**, and a masked or out-of-range position will be returned as a
> top-k index. This is a **correctness bug**, not a perf regression.
>
> It is also a bug that hides: it only surfaces on rows where fewer than `k = 2048` real positions
> exist, so every long-context shape in the microbenchmark passes. The index-set oracle
> (`tests/kernels/test_dsv4_streamindex_topk.py:58-118`, zero tolerance) does catch it — **run the
> unit tests, not only the microbenchmark, on any variant that touches a mask value.**
>
> The one legitimate reason to revisit this is if a candidate narrows the score dtype (A8). `-inf`
> in bf16 is still `-inf` and the contract still holds; the prohibition is on *finite* sentinels,
> not on the dtype.

`[ref]` should be amended to scope its rule to pre-softmax masking.

**D3. `[ref]`'s dtype table (0.74× time for 0.50× bytes) vs report §4's `lkv_dim=129` column (full 2×
saving).** Not a factual conflict — the report models the byte count, the reference measured the
realized time — but the report's 25.0 µs is the optimistic end and the designer should carry A5's
30–35 µs instead, with the further note that at 39.0 µs `modelled` the VPU term overtakes it anyway.

**D4. `[ref]` Deciding #1 "make blocks ≥ 2 MiB" vs the incumbent's actual DMA units** (256 KiB /
32 KiB / 4 KiB / 128 B, A6). The report never costs DMA size at all — it counts bytes, not
transfers. This is a gap in the report rather than a contradiction, but at 848 DMAs in a 221 µs call
it is a first-order omission and the designer should treat "how many transfers" as a live variable
alongside "how many bytes".

**D5. Report §4/§8 treat prefill's dominant term as VPU throughput; A7 says the accumulator is
32.0 MiB against a 32 MiB default scope.** The report already flags the VPU rate as `assumed` and
§7's probe as having failed, and §7's own reading ("bound by moving the tile, not by VPU
throughput") points the same way. I believe the binding term at prefill is **VMEM capacity /
spill**, not VPU ops/s. This is a hypothesis from source arithmetic, not a measurement.

**Convergence with area 2.** Area 2 independently computed the prefill einsum intermediate at
**33.55 MB** and flagged it as an alternative hypothesis to report §4's `assumed` VPU rate. Two
briefers reaching the same figure by different routes — area 2 from the iteration space and the
working-set budget, area 4 from operand precision and accumulator layout — raises this from "one
briefer's arithmetic" to the leading explanation for prefill's 586 µs term. It does not make it
measured.

**What the convergence changes about the probe.** The step-2 prefill probe must be instrumented for
**spills**, not only wall-clock. A spill-bound prefill and a VPU-bound prefill **produce the same
number and demand opposite designs**: a VPU-bound prefill wants fewer elementwise ops per element
(fold `relu(x)·w → relu(x·w)`, gated on `w ≥ 0`); a spill-bound prefill wants a smaller live
accumulator (reduce over `h` incrementally so the peak is `f32[1,128,2048]` = 1 MiB rather than
`f32[1,4096,2048]` = 32 MiB, or shrink `bq_sz`/`bkv_sz` for the prefill call, or accumulate in
bf16). Timing alone cannot separate them. `/xprof-llo` reports VMEM high-water and spill counts and
is the tool that can — the probe should capture with the kernel-trace `LIBTPU_INIT_ARGS` and report
**peak VMEM, spill/restore counts, and time**, all three, or it will come back with a number that
confirms whichever hypothesis the reader arrived with.

---

## D1's falsifier — written so someone else can run it

D1 overturns `report.md` §7 and §8-finding-4, which record the 49.6% K-cache padding as a
**two-consumer** packing change on the grounds that the indexer's `cache_kv` is the same buffer
MLA v2 attention reads. If D1 is right the padding is a **one-consumer** change and it is the
largest single lever surfaced so far: **83.2 MB of 167.8 MB**, on the term that is ~95% of nightly
decode HBM traffic `modelled`. A claim that load-bearing must be falsifiable rather than
well-argued.

### What is actually being claimed, reduced to one testable proposition

D1 rests on two links. The first needs no job:

- **(i) The runner allocates one tensor per layer name, with no aliasing between layer names.**
  `src(runner/tpu_runner.py:6316-6352)` — every branch ends in `torch.zeros(...)` or a tuple of
  them, keyed by `layer_name`. The only aliasing in the function is the explicit
  `bundle_view_by_name` path, which is the non-MLA `AttentionSpec` branch and is not on this path.
  **Falsify by reading those lines.** No hardware.

- **(ii) GLM-5.2's MLA attention layers have `use_sparse=True`, so they take the split
  (nope, rope) branch and never receive a 4-D `cache_kv`.** This is the empirical half. **Everything
  in D1 reduces to (ii).**

So the falsifier is: **does the MLA attention layer for GLM-5.2 hold a tuple of two split caches, or
a single 4-D `u8[..., 4, 256]` tensor?**

### Note on identity tests: `id()` is not a shared-buffer test

Two distinct Python `torch.Tensor` objects can be views over one allocation, so `id(a) == id(b)`
being False proves nothing and `is` is not the question. The genuine test for shared storage is
`a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()`.

**But under torch_xla that test is itself unreliable** — XLA tensors are lazy handles and
`data_ptr()` commonly returns `0` for all of them, which would make every pair compare "equal" and
silently manufacture a FAIL. Treat an all-zero `data_ptr()` column as **INCONCLUSIVE, never as a
pass or a fail.**

What genuinely discriminates here is **structural**, and it does not need pointer identity at all:
a `tuple` of two tensors cannot be the same object as a single 4-D tensor, and the two shapes are
not compatible. Container type and shape are the primary discriminator; `data_ptr()` is only
consulted in the one branch where both sides turn out to be single tensors.

### Route 0 — free: answer it from an artifact already on GCS

Prefer this. It costs nothing and needs no pod.

The existing profile `j-a71a648c-3064-471a-a131` rank 0 already yielded operand shapes: §6.1 read
`u8[3027,64,4,256]` off it as the indexer's `cache_kv`. The same `hlo_stats` / xplane carries
`sc_dsa_gather`, and **its operand shapes settle (ii)**:

```
gs://cloud-devkit/jobs/j-a71a648c-3064-471a-a131/outputs/j-a71a648c-3064-471a-a131-ray-node-0/\
  <pod>/vllm-node/xprof/plugins/profile/<ts>/rank0_*.xplane.pb
```

Dump it with `/xprof-dump` and read the operands of any `sc_dsa_gather` instruction.

- **PASS (D1 holds, one consumer):** `sc_dsa_gather` takes **two** uint8 cache operands of
  *different* minor widths — a wide nope cache and a narrow rope cache — and **neither** has the
  indexer's `[..., 4, 256]` geometry. Expected widths from
  `get_sparse_kv_cache_specs` `src(attention.py:1266-1300)`: nope from `head_size - rope_dim`
  (`576 - 64 = 512`) and rope from `rope_dim = 64`, each through its own
  `SparseMLAKVCacheSpec.create`. The exact numbers depend on the layout enum
  (`TPU_SPARSE_MLA_NOPE_LAYOUT` / `..._ROPE_LAYOUT`), so **match on "two operands, neither of them
  `[...,4,256]`", not on 512 and 64 exactly.**
- **FAIL (report §7 holds, two consumers):** some `sc_dsa_gather` or `MLA-*` instruction takes an
  operand with the indexer's exact geometry `u8[3027,64,4,256]`. One such operand is enough to
  refute D1.
- **INCONCLUSIVE:** the dump does not carry operand shapes for these instructions, or
  `sc_dsa_gather` is absent. Fall through to Route 1.

**Scope warning that applies to this route specifically:** `j-a71a648c` is the **AgentX** deployment
(`max_model_len` 262144, page 256, TP16×DP1). The buffer-identity question is a *model-definition*
property, not a serving-shape property — `use_sparse` is set on the layer, not by
`max_model_len` — so an AgentX answer should transfer to nightly. **"Should" is doing work in that
sentence.** If Route 0 returns PASS, it is strong evidence but it is evidence from the wrong
deployment, and the sheet's claim is scoped "at this configuration". Record it as
`measured (j-a71a648c), AgentX shape, transfer to nightly assumed`.

### Route 1 — in-pod, definitive, needs a model load

Run at the **nightly** configuration and nowhere else:
`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh`, recipe
`vllm-torchtpu-glm-5.2-v7x-16/vllm-torchtpu-glm-5.2-benchmark.yml`, topology `2x2x2` / `tpu7x`.
Read the topology and `--block-size` back from `cdk job recipe` rather than assuming them.

This check needs the real model loaded and the KV caches bound, so **do not pay for a dedicated
job** — fold it into a nightly benchmark run that is happening anyway. It runs once at startup and
prints a handful of lines.

Insert at the end of `TPUModelRunner.initialize_kv_cache`, after `kv_caches` is fully populated:

```python
# --- area4 D1 falsifier: is the indexer K cache a distinct buffer from the
# --- MLA attention KV cache? Prints exactly one verdict line.
def _area4_d1_check(kv_caches, static_forward_context):
    import torch

    def _ptr(t):
        try:
            return t.untyped_storage().data_ptr()
        except Exception:
            return None

    rows, indexer, mla = [], [], []
    for name, cache in kv_caches.items():
        mod = static_forward_context.get(name)
        cls = type(mod).__name__ if mod is not None else "None"
        if isinstance(cache, (tuple, list)):
            kind, shapes = f"tuple[{len(cache)}]", [tuple(t.shape) for t in cache]
            dtypes, ptrs = [str(t.dtype) for t in cache], [_ptr(t) for t in cache]
        else:
            kind, shapes = "tensor", [tuple(cache.shape)]
            dtypes, ptrs = [str(cache.dtype)], [_ptr(cache)]
        rows.append((name, cls, kind, shapes, dtypes, ptrs))
        # Classify by MODULE CLASS, never by shape -- classifying by shape would
        # assume the answer.
        if "Indexer" in cls:
            indexer.append((name, kind, shapes, ptrs))
        elif "MLAAttention" in cls or getattr(mod, "use_sparse", None) is not None:
            mla.append((name, kind, shapes, ptrs, bool(getattr(mod, "use_sparse", False))))

    for r in rows:
        print("D1_ROW", r)
    print(f"D1_COUNTS indexer_cache_layers={len(indexer)} mla_layers={len(mla)} "
          f"mla_use_sparse={sum(1 for m in mla if m[4])}")

    idx_ptrs = {p for _, _, _, ps in indexer for p in ps if p}
    mla_single = [m for m in mla if m[1] == "tensor"]
    shared = [m for m in mla_single if any(p in idx_ptrs for p in m[3] if p)]
    all_ptrs = [p for r in rows for p in r[5]]

    if not indexer:
        print("D1_VERDICT INCONCLUSIVE no-indexer-cache-layer-found")
    elif shared:
        print(f"D1_VERDICT FAIL shared-storage {[m[0] for m in shared]}")
    elif mla_single and all(p in (0, None) for p in all_ptrs):
        print("D1_VERDICT INCONCLUSIVE data_ptr-unavailable-under-xla "
              f"mla_single={[m[0] for m in mla_single]}")
    elif mla_single:
        print(f"D1_VERDICT FAIL mla-holds-single-4d-cache {[m[0] for m in mla_single]}")
    else:
        print("D1_VERDICT PASS distinct-buffers "
              f"indexer={len(indexer)} mla_split={len(mla)}")

_area4_d1_check(kv_caches,
                self.vllm_config.compilation_config.static_forward_context)
```

**What a PASS prints** — D1 holds, the padding is a one-consumer change:

```
D1_COUNTS indexer_cache_layers=21 mla_layers=78 mla_use_sparse=78
D1_VERDICT PASS distinct-buffers indexer=21 mla_split=78
```

Every `D1_ROW` for an `MLAAttention` module reads `tuple[2]` with two shapes of *different* minor
widths; every indexer row reads `tensor` with one shape `(P, page//4, 4, 256)` and dtype
`torch.uint8`.

**What a FAIL prints** — report §7 holds, the padding is a two-consumer change and D1 must be struck
from this sheet:

```
D1_VERDICT FAIL mla-holds-single-4d-cache ['model.layers.3.self_attn.attn', ...]
```

or, if both sides are single tensors and the pointers agree:

```
D1_VERDICT FAIL shared-storage ['model.layers.3.self_attn.attn', ...]
```

**What INCONCLUSIVE prints** — and it is not a pass:

```
D1_VERDICT INCONCLUSIVE data_ptr-unavailable-under-xla mla_single=[...]
```

This is the torch_xla case described above. It fires only when the MLA side holds a *single* tensor,
which is already the interesting case; read the `D1_ROW` shapes by hand before concluding anything.

### Scope: what the result does and does not generalise to

The check must pin which path it exercises, and the sheet must say the answer does not travel.

- A PASS is scoped to the **GLM-5.2 sparse-MLA path** (`use_sparse=True`, DSA indexer present). It
  says nothing about the dense MLA path.
- On the **dense MLA path** (`use_sparse=False`), the MLA attention layer *does* take
  `PallasMLAttentionBackend.get_kv_cache_shape` and receive a single 4-D `cache_kv`
  `src(tpu_runner.py:6334-6351)`. **Report §7's statement is plausibly correct there**, which is the
  likeliest origin of the error — it is a true sentence about the wrong branch. Any repack must not
  break the dense path, even though the dense path is not what GLM runs.
- The two indexer paths that are *not* under study (`VllmDeepseekV4Indexer` at
  `deepseek_v4_indexer.py:111` with an explicit `head_size=align_to(129,128)`, and the `rebind`
  `VllmTPUIndexer` at `mla_attention_op.py:235-236` with an unpadded 129) have their own cache
  specs (report §5) and are outside this result entirely.
- `mla_kv_spec` is written onto the module by the runner at `src(tpu_runner.py:6329)` and consumed
  at `src(attention.py:1539)`. Its presence on a layer is itself a PASS indicator and is worth
  printing if the snippet above needs debugging.

### Why `sc_dsa_gather` ×78 is the corroborating count

A2 cites `sc_dsa_gather` ×78 `measured (j-a71a648c)` as corroboration, next to the indexer's 21.
Two different counts sitting adjacent is not an argument, so here is the arithmetic.

GLM-5.2 has **78 decoder layers**. The checkpoint's `indexer_types` list marks **21** of them
`full` — those construct an indexer and its K cache — and **57** `shared`, which **reuse the
preceding `full` layer's top-k indices and do not run the indexer op** (report §3, confirmed against
the profile's `StreamIdxTC-m` ×21 and `sc_topk_direct_*` ×21).

But **all 78 layers run sparse MLA attention.** A `shared` layer still performs attention — it just
does so with borrowed indices. So the *gather of KV rows at the selected indices* fires once per
**MLA layer**, not once per **indexer layer**:

| quantity | expected count | observed | keyed to |
| --- | ---: | ---: | --- |
| `StreamIdxTC-m` (stage A) | 21 | 21 | indexer layers (`full` only) |
| `sc_topk_direct_*` (stage B) | 21 | 21 | indexer layers (`full` only) |
| `MLA-*` | 78 | 78 | all decoder layers |
| **`sc_dsa_gather`** | **78** | **78** | **all decoder layers** |

`sc_dsa_gather` matching **78 and not 21** is the point. It shows the gather is indexed by MLA layer
rather than by indexer layer, so it cannot be an artifact of the indexer's own machinery. And
`dsa_gather` is named in the runner's own comment as the consumer of the split cache — *"Split
(nope, rope) cache in uint8 dtype (expected by `dsa_gather`)"* `src(tpu_runner.py:6320-6321)` — so
its presence at all is evidence the `use_sparse` branch was taken and the split caches were
allocated.

Putting the two together: 78 MLA layers each gather from a split (nope, rope) pair, while only 21
indexer caches exist in the whole model. **78 consumers cannot be reading 21 buffers one-to-one**,
and the split-cache branch that allocates what they *do* read is a different branch of
`initialize_kv_cache` from the one that allocates the indexer's 4-D `cache_kv`.

**This is inference, not proof.** It is consistent with D1 and inconsistent with report §7, but it
is an argument from instruction counts and a code comment, and it does not directly observe the
operand shapes. That is precisely what Route 0 adds for free, and Route 1 settles. **Until one of
those runs, A2 and D1 are `src(...)` + inference, not `measured`.**
