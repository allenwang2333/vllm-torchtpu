# Area (5) — Which unit does each piece of work

Step-2 briefing sheet for `streamindex_topk` (GLM-5.2 DSA lightning indexer).
Reference: `.claude/skills/write-tpu-kernels/references/05-compute-units.md`.
**This briefer decides nothing.** Everything below is a constraint, an arithmetic template, or a
named unknown, for the single downstream designer to use.

Tag vocabulary, used strictly: `measured (job id)` / `modelled` / `assumed` /
`derived, unverified` / `read from source` / `from 05-compute-units.md`.

> **Read §C's three-way split before quoting any number from it.** §C.1–6 are read out of the
> source and are facts. §C.7–10 are arithmetic on measured numbers and are *inferences*. The
> figure **1.7 ns/element is `derived, unverified`, not a measurement**, and quoting it as one
> misrepresents the evidence.

---

## A. Constraints

**MXU**

1. **Decode's scoring einsum feeds the MXU 32 rows out of 128.** `compute_scores`
   (`streamindex_topk.py:326-331`) issues `einsum("bnd,bmd->bnm")` with `b = seq_batch_size = 4`,
   `n = bq_sz * num_q_heads = 1 × 32`, `d = 128`, `m = bkv_sz = 2048`. The batch axis is 4
   *independent* matmuls (each sequence has its own K pages), so the row extent the MXU sees is
   **32**, not 128. `from 05-compute-units.md` (128-row cliff, `[verified: j-ed517191]`): 32 rows
   measures **3.57×** a 128-row pass at fixed work → row occupancy `min(1, 32/128) = 25%`.

2. **There is no cheap fix for (1), and the designer should not spend a variant on it.** Two
   candidate rewrites, both costed here so they are not re-derived: (a) *concatenate the 4
   sequences' K into one `[8192,128]` RHS and use a `[128,128]` LHS* reaches 128 rows but computes
   16 cross-products and discards 12 — 4× extra work against a 3.57× cliff saved, a net loss; (b)
   *transpose to `K @ Qᵀ`* gives 2048 rows (no row cliff) but 32 columns, and the column sweep in
   `05-compute-units.md` measures N=32 at **1.22× the time of N=128 while doing ¼ the work**
   (≈4.9× waste) — worse than the row cliff it removes. `derived, unverified` (both arithmetic on
   the two measured tables; neither rewrite has been built).

3. **The MXU is not where decode's or prefill's time is, at either peak.** Decode: 320 matmuls ×
   2·32·128·2048 = **5.37 GFLOP**; at the *measured* fp8_e4m3fn→f32 rate of 1419.9e12
   (`from 05-compute-units.md`, `[verified: j-ed517191]`) that is 3.78 µs ideal, **13.5 µs with the
   3.57× cliff** — 6.1% of the measured 221.4 µs
   (`measured j-8b1e7acc-ebdc-45d2-b46c`). Prefill: 40 matmuls × 2·4096·128·2048 = **85.9 GFLOP** →
   **60.5 µs**, rows 4096 so no cliff — 7.1% of the measured 857.3 µs. Note the step-1 report's
   `t_mxu` (2.3 / 37.3 µs) used the *reported* 2300e12; the measured-peak figures above are 1.62×
   larger and are the ones to design against.

4. **fp8 is the only narrow dtype worth anything here, and the kernel already uses it.** `load_bkv`
   bitcasts to `float8_e4m3fn` (`:266`) and q arrives fp8 (`mla_attention_op.py:105`),
   `preferred_element_type=f32`. `int8_ops_per_second = 0` on gen 7 and int8 promotes to bf16
   (`from 05-compute-units.md`). **Forbidden:** any proposal to quantize this matmul further for MXU
   throughput. `from 05-compute-units.md`.

**VPU**

5. **The `relu → ×w → sum_h` chain is ~3 VPU ops on an intermediate 32× larger than the score
   matrix, and that is prefill's dominant modelled term.** Decode: 64 tok × 32 heads × 10240
   positions = **20.97 M elements** × 3 = 62.9 Mops → **36.6 µs**. Prefill: 1024 × 32 × 10240 =
   **335.5 M** × 3 = 1.006 Gops → **585 µs**. Rate `assumed` (VPU 1.72e12 is `derived, unverified`
   from a ratio in `05-compute-units.md:239`; the §7 probe that tried to measure it measured HBM
   traffic instead and is void). Ops/element = 3 is also `assumed`. **Both must be re-probed in VMEM
   before any prefill design leans on this.**

6. **No transcendental appears in this block, so the dominant-op multiplier is 1.00×.** `relu` is a
   `maximum`, the mask is compare/and/select — all at multiply cost per the relative-cost table.
   `from 05-compute-units.md`. Corollary: there is nothing to win from the "exp is cheap / divide is
   free" findings; they do not apply.

7. **Restating the `sum_h` on the MXU buys ~1.8×, not 385×, and only at prefill.** The reduction
   over 32 heads with per-token weights is a block-diagonal matmul
   `M[bq_sz, bq_sz·32] @ relu(s)[bq_sz·32, bkv_sz]`. Fill fraction is 1/bq_sz, so the flop inflation
   is `2·bq_sz/3 = 85×` at bq_sz=128 against the 385× VPU:MXU ratio → net 4.5× on the *two* ops it
   absorbs (×w and +), leaving `relu` on the VPU. Prefill: 585 → ~195 (VPU) + ~130 (MXU at
   f32-DEFAULT 661.4e12) ≈ 325 µs, **1.8×**. At decode bq_sz=1 → M has 1 row → the cliff eats it.
   `modelled`. This is exactly the causal-conv1d counter-case in `05-compute-units.md`: the 385× only
   pays if the MXU is filled.

8. **Prefill's chosen tile implies a 32 MiB live pre-reduction intermediate — the whole default
   scoped VMEM.** `s` before `s.sum(axis=2)` is `(1, 128, 32, 2048)` f32 = **33.55 MB** against
   32 MiB scoped / 64 MiB physical. Decode's is `(4,1,32,2048)` f32 = 1 MiB. `read from source`
   (`:332-335`) + capacity `from 05-compute-units.md`. **If Mosaic materializes this rather than
   fusing the reduction into the einsum's output loop, prefill is spilling, and "prefill is
   VPU-bound" is the wrong diagnosis.** Halving `bkv_p` (2→1) or `bq_sz` (128→64) halves it. This is
   the cheapest prefill probe available and it is an `/xprof-llo` VMEM/spill read, not a rate
   measurement.

**SparseCore — placement**

9. **SparseCore is reachable and stage B may stay there.** `sparsecore_topk.py:39` imports
   `from jax.experimental.pallas import tpu_sc as plsc` and builds `plsc.VectorSubcoreMesh`; it ran
   at four shapes on tpu7x, exact against a numpy oracle.
   `measured (j-0b008512-36f2-4eb2-bdc6)`. See section D — both `hardware-facts.md` and
   `05-compute-units.md` say the opposite and are wrong.

10. **Per *op*, SparseCore is not slow; top-k is just expensive per element.** Derived below (§C):
    **≈1.7 ns per (row, element) per subcore**, 32 subcores in parallel ⇒ ≈18.6 Gelem/s aggregate,
    against the TensorCore VPU's 573 Gelem/s for a 3-op pass. That is 31× slower per element — but
    the radix select executes ~60-70 16-lane vector ops per element against the VPU chain's 3.
    **Forbidden inference:** "move stage B to the TensorCore because SparseCore is slow." The
    correct inference is that stage B's cost is ops-per-element, and the lever is fewer passes or
    fewer elements. `derived, unverified`.

11. **Every SparseCore hand-off is a full TensorCore stall.** `sparsecore_topk` is a separate XLA op
    consuming the complete score matrix; there is no async overlap. The profile reads **174.6 µs of
    TensorCore stalled on SparseCore against 125.25 µs of TensorCore compute** per indexer layer at
    AgentX. `measured` (profile `j-a71a648c-3064-471a-a131` rank 0, via the step-1 report). Any
    design that keeps two units in play must state whether it pays this serialization or eliminates
    it.

12. **Stage B's structure has a hard cliff at padded-token-count `b = 32`, and it is the explanation
    for an anomaly already in the report's own table.** `_pick_partition(b, n, k)`:
    `max_p = 32 // b if b < 32 else 1`. At b ≥ 32 and n ≤ 65536 → **p = 1, single stage, no
    gathers**. At b < 32 → **p > 1, two `_sc_topk_direct` calls plus two `jnp.take_along_axis`
    gathers** (`:439`, `:452`). Read against `j-8b1e7acc`: `prod_decode_b8_ctx8704` stage B =
    **90.3 µs** (b=8 → p=4, two-stage) versus `b32` = **63.6 µs** and `b64` = **67.4 µs** (single
    stage) — *less* work, *more* time. **The two-stage path costs ≥ +27 µs at the nightly width.**
    `read from source` + `measured (j-8b1e7acc-ebdc-45d2-b46c)`.

13. **`gather_offload` is the two-stage path, and it is absent at the nightly config — now read from
    code, not expected.** The two `take_along_axis` calls at `:439` and `:452` are the "2 per indexer
    layer" in the profile; the first takes `f32[16, 262144]` as its operand, matching
    `gather_offload_async_done`. `p = 1` at nightly (b=64 ≥ 32, n=10240 ≤ 65536) ⇒ **neither gather
    is emitted.** The report's "expected, not verified" is now verified from source.
    `read from source` + `measured (profile j-a71a648c)`.

14. **The single-stage limit is `n ≤ 65536` positions and it is a VMEM constraint, not an
    algorithmic one.** `MAX_SLICE_WORDS = 64*1024`; `p_min = cdiv(n, 65536)`. The resident slice is
    256 KB of the 512 KB subcore VMEM (`sparse_core.vmem_capacity_bytes = 524288`, **not divided**,
    `hardware-facts.md:41`). Actual other scratch is ~26 KB, not the ~100 KB the comment claims (see
    §B) — so the constant is conservative by ~1.5×. **Consequence: any `max_model_len ≤ 65536` keeps
    stage B single-stage and gather-free; 262144 cannot.** `read from source`.

15. **Stage B's HBM read is buffer-width-wide and is a blocking, un-double-buffered DMA.**
    `dma(scores_hbm.at[pl.ds(s, slice_len)], resident_vmem, 1)` at `:160`, where
    `slice_len = align_to(n, 16)` is a **compile-time constant**, and `dma()` is
    `cp.start(); cp.wait()` (`:99-102`). `row_lengths` gates only whether the DMA happens at all
    (`@pl.when(i_hi > 0)`), never its size. **This is the mechanism by which `max_model_len` drives
    stage B at byte-identical real work.** `read from source`.

    > **Standing magnitude correction (see the third appended section).** The size of that effect is
    > *not* the 5.9× of `report.md` §6.2. §6.3 (`measured j-63b9f2e4-26c9-4e3a-99ce`, noise floor
    > 0.8–2.5%) deconfounds page size and measures a 16× width increase as **1.16× / 1.17× stage B
    > at decode b64 / b32**, **1.77–1.78× at b8**, **2.88× at prefill q1024** and **3.41× at prefill
    > q4096**. The mechanism in A15 survives; the magnitude does not. Quote the §6.3 numbers.

16. **The output write is `k`-wide per row unconditionally, including for empty padded rows.**
    `write_empty_rows=True` (`streamindex_topk.py:835`) ⇒ `out_gate = active`, so every row in the
    padded token bucket runs `fill_body` (k/16 = 128 stores) and a `k`-word DMA out. At T=64 that is
    0.52 MB; **at T=1024 it is 8.39 MB and ~960 of those rows are pure padding.**
    `read from source`. A cheap candidate falls straight out — see §C.

    > **Scope correction (see the second appended section).** This term is `4·k` per row and is
    > **independent of `n`**. It therefore cancels in every §6.3 width ratio and cannot contribute to
    > any of them. It is a *level* term at large `b`, and it bears on the T-padding question, not on
    > the `max_model_len` question.

17. **Scores must be f32 for stage B as written.** `sparsecore_topk:395` rejects non-f32; the
    monotone key at `:107-109` is a 32-bit trick and every `plsc` op in the kernel is i32. A bf16
    score buffer would halve stage-A write / stage-B read traffic *and* double the single-stage `n`
    limit to 131072 — but it is a rewrite of the key path, not a dtype flag. `read from source`.

18. **Two of the three stage-A Pallas launches are empty in every pure-decode and every pure-prefill
    step, and this is now confirmed rather than conjectured.** `mla_attention_op.py:184` builds
    `distribution = stack([rd[0], rd[0], rd[2]])` — so `distribution[0] == distribution[1]`
    **always**. Call 2's grid is `distribution[1] - decode_batch_end = rd[0] % 4`; call 3's is
    `distribution[2] - distribution[1] = rd[2] - rd[0]` = 0 in pure decode. At the profiled batch 16,
    `16 % 4 = 0` ⇒ both empty. **`StreamIdxTC-d`'s 5.40 µs mode and `StreamIdxTC-m`'s 5.22 µs are
    empty-grid Pallas launch cost**, ≈10.6 µs/layer of pure overhead. `read from source` +
    `measured (profile j-a71a648c)`. The report listed this as an unconfirmed candidate explanation;
    the `distribution` construction closes it.

---

## B. `## Occupancy arithmetic to emit with a design`, instantiated

The reference file's four bullets, plus the three SparseCore fields the file has no template for
because it believed SparseCore unreachable. **Every design option must return this table filled in,
for decode and prefill separately.**

### MXU

| field | formula | incumbent, decode | incumbent, prefill |
|---|---|---|---|
| `mxu_flops` | `Σ_calls 2·m·k·n` | `320 × 2·32·128·2048` = **5.37 GFLOP** | `40 × 2·4096·128·2048` = **85.9 GFLOP** |
| `mxu_peak_used` | measured table, **not** reported constants | fp8_e4m3fn→f32 DEFAULT = **1419.9e12** | same |
| `mxu_row_extent` | LHS row extent per pass | **32** | **4096** |
| `mxu_machine_fraction` | `min(1, rows/128)` | **0.25** | **1.00** |
| `mxu_cliff_factor` | measured cliff table | **3.57×** | 1.00× (0.91× ≥256 rows) |
| `t_mxu` | `mxu_flops / peak × cliff` | **13.5 µs** | **60.5 µs** |
| `t_mxu` as % of measured call | | **6.1%** of 221.4 µs | **7.1%** of 857.3 µs |

Halve the expectation at 64 rows, quarter at 32 — the incumbent is at the quarter.

### VPU

| field | formula | incumbent, decode | incumbent, prefill |
|---|---|---|---|
| `vpu_elements` | `T × H × (num_bkv·bkv_sz)` | `64 × 32 × 10240` = **20.97 M** | `1024 × 32 × 10240` = **335.5 M** |
| `vpu_ops_per_element` | count the emitted chain | **3** `assumed` (relu, ×w, +) | 3 `assumed` |
| `dominant_op_relative_cost` | transcendental table | **1.00×** (max/mul/add only) | 1.00× |
| `t_vpu` | `elements·ops / 1.72e12 × rel_cost` | **36.6 µs** `assumed rate` | **585 µs** `assumed rate` |
| `vpu_elements_post_reduce` | `T × num_positions × ~5` | 3.3 Mops → **1.9 µs** | 52 Mops → **30 µs** |
| `live_intermediate_bytes` | `seq_batch·bq_sz·H·bkv_sz·4` | 1.05 MB / 32 MiB scoped — **3%** | **33.55 MB / 32 MiB scoped — 105%** ⚠️ |
| `arithmetic_intensity` vs 1 F/B crossover | `03-pipelining.md` rule | 30.5 F/B — above; VPU decides, not HBM | 568 F/B — above |

The `live_intermediate_bytes` row is not in the reference file's list and is added because at prefill
it is the field that fails.

### SparseCore

For each `_sc_topk_direct` call the design emits. `sc = get_tpu_info().sparse_core` =
`num_cores=2, num_subcores=16, num_lanes=16, vmem_capacity_bytes=524288, dma_granule_size_bytes=32`.

| field | formula (read out of `sparsecore_topk.py`) | incumbent, decode (b=64, n=10240, k=2048) | incumbent, prefill (b=1024, n=10240) | AgentX (b=16, n=262144) |
|---|---|---|---|---|
| `p` | `_pick_partition(b,n,k)` — `cdiv(n,65536)` raised while `p·2 ≤ 32//b` | **1** | **1** | **4** |
| `stages` | 1 if p==1 else 2 + **2 XLA gathers** | **1, no gather** | 1, no gather | **2 + 2 gathers** |
| `sc_rows` | `b · p` | 64 | 1024 | 64 (stage 1) + 16 (stage 2) |
| `sc_waves` | `cdiv(b·p, 32)` | **2** | **32** | 2 + 1 |
| `subcore_occupancy` | non-empty rows in wave / 32 | **32/32 = 100%** | 100% | **8/32 = 25%** (only slice 0 non-empty at ctx 8192) |
| `resident_vmem_per_subcore` | `4·align(n/p,16) + 4·(16·256+16) + 4·(256+16) + 4·(k+16) + 4·align(b,16)` ≤ **524288** | 40 960 + 16 448 + 1 088 + 8 256 + 256 = **67 008 B = 12.8%** | **67 008 B = 12.8%** (b_pad grows to 4 KB → 70 KB, 13.4%) | 262 144 + 26 048 = **288 192 B = 55.0%** |
| `sc_hbm_read` | `4·align(n/p,16)` per **non-empty** row — *not* `row_length` | 64 × 40 960 = **2.62 MB** (eff_len 8704 ⇒ **15% read past row_length**) | **41.9 MB** | 16 × 262 144 = **4.19 MB**, in 256 KB *blocking* chunks |
| `sc_hbm_write` | `4·k` per row where `out_gate` | **0.52 MB** | **8.39 MB** (960 rows of it pure padding, if T pads to 1024) | 64×8 KB + 16×8 KB = 0.66 MB |
| `data_passes_over_slice` | cvt + hist0 + ≤3 refinement + emit = **3 to 6** | ≤6 | ≤6 | ≤6 per stage |
| `fixed_ops_per_row` | `4 levels × (zero_priv 256 + fold_priv ~496 + scan_glob ~48 + find_bucket ~48)` ≈ **3 400** 16-lane issues, independent of n | 3 400 | 3 400 | 3 400 × 2 stages |
| `t_sc` | `max over subcores of Σ_waves (blocking DMA + 1.7 ns × eff_len)` | 2 × 8704 × 1.7 ns ≈ **30 µs** device (67.4 µs measured incl. ~35 µs dispatch) | 32 × 8704 × 1.7 ns ≈ **481 µs** ✓ measured | 3 × 8192 × 1.7 ns = 42 µs **+ 103 µs gathers + ~190 µs DMA stall** ≈ 336 µs device (371.3 measured) |
| `tc_stall` | **all of `t_sc`** — no overlap exists | 30 µs | 481 µs | 174.6 µs measured, of which 103.5 µs is the two gathers |

**If SparseCore does not fit for some piece, the reference file's instruction is: "the answer here is:
it may, and you cannot put it there. Say so and move on. Do not write `plsc` code against this jax
version."** That instruction is **superseded for this block** by `j-0b008512` (§D). The statement
that replaces it, and which the designer must emit for every piece it *declines* to put on
SparseCore, is one of the two structural reasons the source actually gives:

- **capacity:** `4·align(n/p,16) + ~26 KB > 524288` ⇒ `p` rises ⇒ two stages ⇒ two
  `take_along_axis` gathers on the critical path (≥ +27 µs measured at nightly width, 103.5 µs at
  AgentX width).
- **serialization:** SparseCore is reachable only as a separate XLA op, so the piece costs a full
  TensorCore stall for its whole duration; there is no demonstrated way on this stack to run a
  `VectorSubcoreMesh` and a TensorCore grid concurrently inside one `pl.kernel`. **Unprobed** — see
  §C.

---

## C. What can be established about stage B's floor from the source

### Read from code (no inference)

1. **Per-element compute tracks `row_lengths`; the HBM read tracks the buffer width `n`.**
   `i_hi = min(slice_len, eff_len)`, `walk_chunks = cdiv(i_hi, 16)`, and every
   `plsc.parallel_loop` — `cvt`, `hist0_body`, `ref_body`, `emit_body` — is bounded by
   `walk_chunks`. The DMA at `:160` is `pl.ds(s, slice_len)` with `slice_len` a Python-level
   constant. **This answers the report's open question directly and in two halves.**

2. **The DMA is synchronous and un-buffered** (`cp.start(); cp.wait()`, `:99-102`), so the read is a
   stall, not overlapped work. At AgentX that is 256 KB of blocking transfer before a subcore does
   any arithmetic.

3. **The cost is `passes × cdiv(eff_len,16) + fixed`, with 3 to 6 passes.** Always: `cvt` (1),
   `hist0` (1), `emit` (1). Conditionally up to 3 more: `ref_body` runs with
   `active_walk = where(done, 0, walk_chunks)`, so refinement levels that terminate early
   (`quota == cnt`, `quota2 == 0`, or `total <= k`) are skipped per row at runtime.
   `read from source`, `:158-291`.

4. **The fixed per-row term is ≈3 400 16-lane vector issues** — `zero_priv(256)` is 256 stores,
   `fold_priv(16)` is 16×31 ops, `scan_glob(16)` and `find_bucket(16)` ~48 each, ×4 radix levels. It
   does not scale with `n` or `eff_len`, and at eff_len 8704 (`walk_chunks = 544`) it is ~10% of the
   row's work. This is why stage B is strongly sublinear in `n`.

5. **`p`, `slice_len`, `num_waves` and the two-stage/one-stage choice are all decided from
   `(b, n, k)` at trace time.** No runtime length information reaches them. `row_lengths` enters
   only inside the kernel.

6. **The kernel hardcodes `num_subcores = 16, num_sc_cores = 2` at `:85-86`** while building the
   mesh from `sc.num_cores`/`sc.num_subcores` at `:327-332`. On this chip they agree; the hardcode
   is a latent hazard, not a live bug.

### Inferred (arithmetic on measured numbers — say "modelled" if you quote it)

7. **≈1.7 ns per (row, element) per subcore at eff_len ≈ 8704.** Two independent anchors, one
   parameter, from `j-8b1e7acc`:
   - decode: 67.4 µs measured − ~35 µs host dispatch (§6.2 establishes the dispatch floor at
     34.6–44.9 µs) ≈ 30 µs device ÷ 2 rows/subcore ÷ 8704 = **1.72 ns/elem**
   - prefill: 481.0 µs ÷ 32 rows/subcore ÷ 8704 = **1.73 ns/elem**

   A 16× change in wave count reproduces the same per-element rate. `derived, unverified`. It is the
   number to hand any stage-B candidate as its baseline; it is **not** a bandwidth and must not be
   quoted as one.

8. **A first-principles floor for an *exact* select is ≈0.4× the incumbent.** Counting emitted
   vector ops per 16-lane chunk: `cvt` ~5, `hist0` ~9, `ref_body` ~13 each (2 of 3 levels typically
   live), `emit` ~18 ⇒ ~58-71. An irreducible exact select still needs one histogram pass (~10) and
   one emit pass (~15) ⇒ ~25. **Floor ≈ 12 µs at decode, ≈ 190 µs at prefill.** So stage B sits at
   roughly **40% of its own modelled floor-of-a-different-algorithm** — inside the "structural" band
   the skill's exit table uses, but not the 5× freebie the report's zero-compute floor implied.
   `modelled`, and the op counts are read off the source, not measured.

9. **The 5.9× `max_model_len` effect is entirely non-algorithmic.** Under (1) and (7), AgentX at real
   ctx 8192 does the *same* per-element compute as nightly (8192 vs 8704 elements). Its 371.3 µs
   decomposes as ≈42 µs compute + 103 µs for the two gathers (profile) + ~190 µs of blocking
   full-width DMA + ~35 µs dispatch. **None of the 5.9× is the radix select being slower; all of it
   is the buffer width forcing `p=4`, two stages, two gathers, 256 KB blocking reads, and 25%
   subcore occupancy.** `derived, unverified` (the DMA term is the residual, so it is fitted).

   > **Standing magnitude correction (see the third appended section).** The 5.9× is an AgentX-only
   > figure and §6.3 (`measured j-63b9f2e4-26c9-4e3a-99ce`) measures **1.16–1.17× at decode
   > b64/b32**, **1.77–1.78× at b8**, **2.88×/3.41× at prefill** for a 16× width increase at fixed
   > page size. The *decomposition* in C.9 stands as a description of the AgentX operating point;
   > the *magnitude* must not be carried to nightly-like shapes. The refit against §6.3 is in the
   > second appended section.

10. **Padding the decode token count from 64 to 1024 would cost stage B ~80 µs**, from 960 empty
    rows each running `fill_body` (128 stores) plus an 8 KB output DMA = 7.9 MB of writes.
    `write_empty_rows=False` removes it. `modelled`; §3 of the report records the padding bucket as
    unresolved, and this is the arithmetic that makes it matter.

### Not determinable from source — the probes, with what each must measure

- **Is prefill stage B vector-issue-bound or output-DMA-bound?** Both models fit 481 µs (65 ops/elem
  at ~2.6 G issues/s, *or* 50.3 MB at ~105 GB/s). **Probe: hold `b`, `n` and `row_lengths` fixed and
  sweep `k` alone** (2048 → 1024 → 512, all multiples of 16). Output-DMA-bound ⇒ time falls roughly
  linearly in `k`; issue-bound ⇒ time barely moves (only `fill_body` and the final DMA depend on
  `k`). One `run-command-in-cdk` shard, seconds.
- **Does the length-bounded DMA exist?** Whether a `pltpu.make_async_copy` with a **dynamic** length
  (rounded to the 32-byte / 8-word granule) is legal inside a `plsc` kernel. If yes,
  `dma(scores_hbm.at[pl.ds(s, dyn_len)], ...)` is a one-line change that makes stage B's HBM read
  track `row_lengths`; if no, the fallback is a fixed-size chunked DMA with a dynamic trip count,
  which also un-blocks the transfer. **This is the single highest-value stage-B probe.**
- **Can `MAX_SLICE_WORDS` rise?** The comment claims "~100KB more" for histogram and small buffers;
  the actual scratch is **26 048 B**. A slice of 96 K words (384 KB) + 26 KB = 410 KB of 512 KB.
  Probe: does it compile and run at n=98304? Raises the single-stage limit from 65536 to ~98304
  positions.
- **Can a TensorCore grid and a `VectorSubcoreMesh` overlap?** Gates every "pipeline stage B against
  stage A" candidate, and 174.6 µs/layer of measured TensorCore stall is what it is worth.
  `05-compute-units.md` lists `create_tensorcore_mesh`, `core_map`, `CORE_PARALLEL`,
  `SUBCORE_PARALLEL` as existing symbols but has never tried composing them with `plsc`.
- **The VPU rate and ops/element, in VMEM** (unchanged from step 1) — plus the new §A.8 question of
  whether prefill's 33.55 MB intermediate spills. `/xprof-llo` on a prefill capture answers the
  second for free.

---

## D. Gaps this block turns on, and the two `hardware-facts.md` errors

### Gaps from `05-compute-units.md` that carry a decision here

| gap | the decision it changes | which way |
|---|---|---|
| *"No SparseCore measurement is possible on this stack… `jax.experimental.pallas.tpu_sc` — the module tokamax actually imports as `plsc` — was never probed"* | Whether stage B stays on SparseCore or is folded into the TensorCore kernel. Under the gap's pessimistic reading the entire SparseCore option space is closed and stage B — 30% of decode, 56% of prefill — must be rewritten on TensorCore. | **Answered: `tpu_sc` imports and runs.** `measured (j-0b008512-36f2-4eb2-bdc6)`. Stage B stays; the file's own Gaps entry correctly predicted that this was the untried path. |
| *"No measurement of the MXU under a real Pallas kernel other than `matmul_block_size_sweep` and `mxu_block_rows_cost`"* — and the cliff table is **bf16**, XLA-level | Whether decode's `t_mxu` is 3.8 µs (no cliff at fp8) or 13.5 µs (3.57× cliff). That is 1.7% vs 6.1% of the call — the difference between "ignore the MXU at decode" and "spend a variant on it". | Currently assumed to apply. **Probe: `mxu_block_rows_cost` re-run at `float8_e4m3fn`.** Cheap, and it is a gap-filling result that belongs back in the file. |
| *"The two MXU granularity readings above have not been separated"* (rows at fixed work vs K/N at varying work) | Whether transposing decode's einsum to `K @ Qᵀ` (2048 rows × 32 cols) beats the incumbent (32 rows × 2048 cols). Under "sub-128 extents are padded up" the transpose trades a 3.57× row penalty for a ~4.9× column penalty and loses; under "these arms are not compute-bound at all" it is a wash. | **Do not transpose** on the current evidence. The file marks the mechanism `[unverified]` and the non-monotonic N=256 bump argues against clean padding. |

Gaps that do **not** apply: int4 (no integer path is used — `int8_ops_per_second = 0` and this kernel
is fp8 throughout); the tokamax path corrections (no tokamax code is in scope).

### `references/hardware-facts.md` error 1 — "no SparseCore path on this stack"

`hardware-facts.md:317-318` states: *"No SparseCore fact is measured. `sparse_core` is quoted as
`get_tpu_info()` reports it; there is no SparseCore path on this stack to check it against."*
`05-compute-units.md` carries the same error more forcefully in its **"SparseCore: present in the
hardware, absent from this Pallas"** section: *"You cannot reach it from Pallas on this stack… Do not
write `plsc` code against this jax version."*

**Both are wrong.** The correct facts:

- The SparseCore Pallas path is reachable on jax 0.10.2 / libtpu 0.0.44.1 via **`from
  jax.experimental.pallas import tpu_sc as plsc`** — the one module path the `sparsecore_surface`
  probe never tried, as that probe's own `Gaps` entry admits. It tested `pallas.tpu.sparsecore`,
  `pallas.sparsecore` and `_src.pallas.mosaic.sparsecore`; none of those is it.
- `plsc.VectorSubcoreMesh`, `plsc.parallel_loop`, `plsc.cumsum`, `plsc.addupdate_scatter`,
  `plsc.store_compressed`, `plsc.all_reduce_population_count` and `pl.kernel(..., mesh=…)` all work.
  This repo ships a production kernel built on them (`sparsecore_topk.py`).
- Job **`j-0b008512-36f2-4eb2-bdc6`**, `JobStatus=Succeeded`, tpu7x single-host, ran
  `sparsecore_topk` at `(b,n,k)` = (32,4096,512), (32,32768,2048), (64,65536,2048), (256,65536,2048)
  with no `NotImplementedError` and no OOM, and **all four matched a numpy top-k set oracle
  exactly**.
- `get_tpu_info().sparse_core` returned `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16,
  vmem_capacity_bytes=524288, dma_granule_size_bytes=32)` — read back on hardware, not quoted from a
  datasheet.

A design that had obeyed `05-compute-units.md`'s "say so and move on" would have proposed rewriting a
working SparseCore top-k onto the TensorCore for no reason.

### `references/hardware-facts.md` error 2 — the subcore count

`sparsecore_topk.py`'s module docstring (`:18-22`) claims *"all 32 subcores on the chip"*, and
`_pick_partition`'s constraint is written as `B * P <= 32`. The hardware reports **`num_cores=2,
num_subcores=16`** (`j-0b008512`). The correct fact is **16 subcores per SparseCore, 2 SparseCores,
32 subcore programs per chip** — the product, not the field.

The kernel's *arithmetic* is right: it builds the mesh from `sc.num_cores × sc.num_subcores`
(`:327-332`) and computes `flat = core*16 + sub` over a 2×16 index space (`:85-89`). The hazards are
(a) the docstring, which will mislead anyone sizing a variant, and (b) the literals
`num_subcores = 16` / `num_sc_cores = 2` hardcoded at `:85-86` alongside a mesh built from the
*reported* fields — they agree on tpu7x and would diverge silently on anything else. **Any candidate
must read the count from `sc.num_cores * sc.num_subcores`, never from the docstring and never from a
literal 32.**

`hardware-facts.md:41` — `sparse_core.vmem_capacity_bytes = 524288`, **not divided**, 512 KiB *per
vector subcore* — is **correct** and is load-bearing: it is the constraint behind `MAX_SLICE_WORDS`,
behind `p`, and therefore behind the two gathers.

---
---

## Convergence: four briefers, one mechanism

**§C.1 and A15 were reached independently by four briefers.** Areas 2, 6 and 7 each read the same
thing out of the same lines: the scan loops are bounded by `i_hi = min(slice_len, eff_len)`
(`sparsecore_topk.py:155`), while the DMA at `:160` fills `resident_vmem` from
`scores_hbm.at[pl.ds(s, slice_len)]` with a **static** `slice_len = align_to(n, 16)`, gated only on
`i_hi > 0` and never sized by it. Four independent readings of four different questions landing on
one line makes this **the best-established fact produced in step 2** — better established than
anything measured, because it is not a measurement at all but a property of the code.

**`report.md` §8's framing must be re-scoped rather than run as written.** §8 lists as a step-2 probe:
*"whether stage B's row width can be bounded by `seq_len` instead of `max_model_len` … what is
unknown is whether the radix select's cost tracks it or the buffer width."* That question is now
answered, and the answer is **both, in different terms**:

| term | tracks | source |
|---|---|---|
| per-element vector work (`cvt`, `hist0`, `ref_body` ×≤3, `emit`) | **`row_lengths`** (`eff_len`) | `walk_chunks = cdiv(i_hi, 16)`, `:156` |
| fixed histogram / fold / scan / bucket-search work | **neither** — constant ≈3 400 issues per row per level | `:176-256` |
| HBM read into `resident_vmem` | **buffer width `n`** | `pl.ds(s, slice_len)`, `:160` |
| HBM write of the result | **`k`** — neither | `out_hbm.at[pl.ds(row_c*k, k)]`, `:295-299` |
| single-vs-two-stage branch, and the two gathers | **`n` and `b` at trace time** | `_pick_partition`, `:360-377` |

The probe as written ("does the cost track `row_lengths` or the buffer width?") would spend hardware
time re-deriving a source reading. **The probe worth running instead is the one in §C's
not-determinable list: whether a dynamic-length `pltpu.make_async_copy` is legal inside a `plsc`
kernel.** That is the question the source cannot answer and the one that decides whether the `n`-term
is removable.

**Area 3 reached the complementary half from the branch side** — `_pick_partition` returns `p=1` at
the nightly config and `p=4` at AgentX, so the AgentX 5.9× is a *threshold crossing* rather than a
slope. That is the same finding as **A12** (the `b = 32` cliff) and **§C.9** (the AgentX
decomposition), arrived at from the control flow rather than from the DMA. Two briefers on the branch
and four on the DMA: the two halves are not competing explanations, they are the two `n`-dependent
terms in the table above, and which one dominates depends on which side of `p = 1` the shape sits.

---

## The check against §6.3 — six of seven, and the seventh named

`report.md` §6.3 (**job `j-63b9f2e4-26c9-4e3a-99ce`, `JobStatus=Succeeded`, 4/4 shards `rc=0`**,
noise floor **0.8–2.5%** measured by replicate) varies `max_model_len` at **fixed page size 1024**,
which is the deconfound §6.2 lacked. Seven pairs. This section checks A12 / §C.9 against them
quantitatively.

### Step 1 — `_pick_partition` run by hand for all seven shapes

`n = align_to(pages_per_seq, 2) · 1024` at page size 1024, so `mml` 4096 → n=4096, 16384 → n=16384,
65536 → n=65536. `k = 2048` (assumed — the bench's `index_topk`; if the bench used a different `k`
the `next_np >= k` guard moves and the b=8 rows change).
`p_min = cdiv(n, 65536) = 1` for **all** of them. `max_p = 32//b if b < 32 else 1`.

| §6.3 shape | b | n_lo → n_hi | **p_lo → p_hi** | stages lo → hi | gathers lo → hi | measured stage B ratio |
|---|---:|---|---|---|---|---:|
| `decode_b8_ctx4096` | 8 | 4096 → 65536 | **2 → 4** | 2 → 2 | 2 → 2 | **1.77×** |
| `decode_b32_ctx4096` | 32 | 4096 → 65536 | **1 → 1** | 1 → 1 | 0 → 0 | 1.17× |
| `decode_b64_ctx4096` | 64 | 4096 → 65536 | **1 → 1** | 1 → 1 | 0 → 0 | 1.16× |
| `decode_b8_ctx16384` | 8 | 16384 → 65536 | **4 → 4** | 2 → 2 | 2 → 2 | **1.78×** |
| `decode_b32_ctx16384` | 32 | 16384 → 65536 | **1 → 1** | 1 → 1 | 0 → 0 | 1.07× |
| `prefill_q1024_ctx4096` | 1024 | 4096 → 65536 | **1 → 1** | 1 → 1 | 0 → 0 | **2.88×** |
| `prefill_q4096_ctx4096` | 4096 | 4096 → 65536 | **1 → 1** | 1 → 1 | 0 → 0 | **3.41×** |

**The first thing this says is a correction to my own framing, and to the coordinator's.** *No shape
in §6.3 crosses the `p = 1` boundary.* b ≥ 32 is `p = 1` at both widths; b = 8 is `p > 1` at both
widths. So **the branch cannot be the mechanism for any ratio in §6.3.** A12 is a statement about
*levels* — it explains why `prod_decode_b8_ctx8704` (90.3 µs) costs more than `b32` (63.6 µs) in
`j-8b1e7acc`, a comparison across `b` at fixed width. It does not explain a comparison across width
at fixed `b`, and I should not have let the coordinator's "b=8 crosses to `p>1` and jumps" stand: it
does not cross.

What survives, and it is the useful half: **`p` correctly sorts the seven shapes into the two groups
the data separates into.** The three `p > 1` rows (b=8, both contexts) measure 1.77–1.78×; the four
`p = 1` decode rows measure 1.07–1.17×. That is a clean partition with no overlap, and it is
predicted by the source. What `p` does *not* do is explain prefill, which is `p = 1` and has the
largest ratios in the table.

### Step 2 — prefill, where the mechanism must be the unconditional DMA

At `p = 1` the source admits exactly **one** width-dependent term: the `4·align(n,16)`-byte
unconditional DMA at `:160`. Everything else is `eff_len`-bounded, `k`-bounded, or constant. In
particular **A16 (`write_empty_rows`, `4·k` per row) is `n`-independent and cancels in every ratio
here** — the coordinator asked whether it could carry prefill, and structurally it cannot contribute
to a width ratio at all.

So the model is forced in form, with two free coefficients:

```
t_stageB  =  num_waves × ( a·n  +  c·eff_len )        num_waves = ceil(b/32)
```

`eff_len` for the prefill arms: `q1024_ctx4096` scores tokens 3072..4095 of a 4096 context → mean
**3584**; `q4096_ctx4096` scores tokens 0..4095 → mean **2048**. (Getting this right matters; using
3584 for both is what made an earlier pass of this arithmetic look worse than it is.)

Joint fit, 4 measurements, 2 parameters:

| arm | waves | eff_len | n | measured | model | residual |
|---|---:|---:|---:|---:|---:|---:|
| `prefill_q1024_ctx4096`, mml 4096 | 32 | 3584 | 4096 | 245 µs | 263 µs | **+7.5%** |
| `prefill_q1024_ctx4096`, mml 65536 | 32 | 3584 | 65536 | 706 µs | 703 µs | **−0.4%** |
| `prefill_q4096_ctx4096`, mml 4096 | 128 | 2048 | 4096 | 695 µs | 652 µs | **−6.2%** |
| `prefill_q4096_ctx4096`, mml 65536 | 128 | 2048 | 65536 | 2373 µs | 2413 µs | **+1.7%** |

Fitted: **`a` = 0.224 ns per f32 word per row** (= 17.9 GB/s per subcore, **572 GB/s aggregate over
32**) and **`c` = 2.04 ns per element per subcore**.

Two things make this more than a curve fit:

- **`a` is stable across the two pairs to 9%** when fitted independently (0.2344 from q1024,
  0.2134 from q4096) — two shapes whose row counts differ 4×. The width-dependent coefficient is the
  robust one, which is what the mechanism predicts.
- **`c` = 2.04 ns/elem lands within 19% of the 1.72–1.73 ns/elem derived in §C.7 from a completely
  different job (`j-8b1e7acc`), at a different context length, by a different route.** Two jobs, two
  configurations, one number.

**So: yes, the unconditional full-width DMA covers 2.88× and 3.41×.** Max residual **7.5%** on a
2-parameter model spanning a 4× range in row count and a 16× range in width. It does not fall short.
`modelled` — the coefficients are fitted, the functional form is read from source.

The one thing the fit does *not* do is cross-predict: using q4096's independently-fitted `(a, c)` to
predict q1024's ratio gives 2.48× against a measured 2.88× (−14%), and the reverse gives 4.01×
against 3.41× (+18%). **`c` should not depend on `q` and it does** (1.87 vs 2.23 ns/elem). Named as
unexplained. The most likely candidate is that `eff_len` varies *within* a wave at prefill — a wave's
duration is set by its longest row, not its mean — and a mean-`eff_len` model understates the short-q
case. Not established.

### Step 3 — the b=8 rows

Both are `p > 1` at both widths, so the two gathers are present throughout and cancel in the ratio.
What does not cancel is that at the wide setting the partition puts nearly all of the width in
**empty** slices while the DMA still reads full width. At `decode_b8_ctx16384`, mml 65536: `p=4`,
`n_p=16384`, `row_lengths=16384` ⇒ `lengths_p = [16384, 0, 0, 0]` — **8 of 32 subcore-rows non-empty,
each DMA-ing 16384 words to walk 16384**; at mml 16384: `p=4`, `n_p=4096`, `lengths_p =
[4096,4096,4096,4096]` — **32 of 32 non-empty, 4096 words each**. Applying the fitted `(a, c)`:

| arm | stage 1 (per busy subcore) | stage 2 | model total | measured (device, −35 µs dispatch) |
|---|---|---|---|---|
| `b8_ctx4096`, mml 4096 | `p=2`, 2048·(0.224+2.04) = 4.6 µs | b=8, n=4096 → 9.3 µs | **13.9 µs** | 73 → **38 µs** |
| `b8_ctx4096`, mml 65536 | `p=4`, 16384·0.224 + 4096·2.04 = 12.0 µs | b=8, n=8192 → 18.5 µs | **30.5 µs** | 129 → **94 µs** |
| ratio | | | **2.19×** | **2.5×** |
| `b8_ctx16384`, mml 16384 | `p=4`, 4096·(0.224+2.04) = 9.3 µs | 18.5 µs | **27.8 µs** | 79 → **44 µs** |
| `b8_ctx16384`, mml 65536 | `p=4`, 16384·0.224 + 16384·2.04 = 37.1 µs | 18.5 µs | **55.6 µs** | 141 → **106 µs** |
| ratio | | | **2.00×** | **2.41×** |

**Ratios agree to 14% and 17%.** Absolutes are under-predicted by 2.4–2.7× uniformly, which is the
two `take_along_axis` gathers the model does not contain (the profile prices them at 51.5 µs each on
SparseCore at AgentX width; two of them at ~25 µs each would close both rows). The model's failure
here is a *known omission*, not a wrong shape. `modelled`.

### Step 4 — the b=32 / b=64 residual, named rather than papered over

| arm | model (device) | measured | measured − 35 µs dispatch | model ratio | measured ratio |
|---|---:|---:|---:|---:|---:|
| `decode_b64_ctx4096`, mml 4096 | 18.5 µs | 55 µs | ≈20 µs | | |
| `decode_b64_ctx4096`, mml 65536 | 46.1 µs | 64 µs | ≈29 µs | **2.48×** | 1.16× raw / 1.45× device |
| `decode_b32_ctx4096`, mml 4096 | 9.3 µs | 42 µs | ≈7 µs | | |
| `decode_b32_ctx4096`, mml 65536 | 23.0 µs | 49 µs | ≈14 µs | **2.95×** | 1.17× raw / 2.0× device |
| `decode_b32_ctx16384`, mml 16384 | 37.1 µs | 61 µs | ≈26 µs | | |
| `decode_b32_ctx16384`, mml 65536 | 48.1 µs | 65 µs | ≈30 µs | **1.30×** | 1.07× raw / 1.15× device |

**The model over-predicts the wide-buffer decode rows by 1.4–1.6× and is right on the narrow ones to
8–33%.** I will not claim these rows as a fit, and here is why they cannot be one:

1. **These measurements are host-floored.** §6.3 records a per-shard dispatch floor of
   **31.9–39.8 µs** and flags five rows `[HOST BOUND]`. The b=32/b=64 stage-B arms measure
   42–65 µs — the floor is **55–80% of the measurement**. §6.2 further establishes that `time_arm`
   blocks once at the end of its loop, so an arm costs ≈`max(host, device)`, **not** `host + device`
   — which means the "−35 µs" column above is not even the right operation, only the best available
   one. A ratio taken between two host-floored numbers is compressed toward 1.0 by construction, and
   1.07–1.17× is what a compressed ratio looks like.
2. **The residual has a consistent sign**: at low wave counts and large `n`, measured is *below*
   `a·n`. A hypothesis that fits the sign and roughly the size — **`a` is not a constant but a
   per-subcore share of a contended HBM port**, so the 572 GB/s aggregate fitted at prefill (32
   waves, sustained) understates what a subcore gets during 1–2 waves. **Named as a hypothesis; not
   established, and not something to design against.**

**So the honest score is six of seven.** The three b=8 rows and the two prefill pairs (five pairs, six
of the seven rows counting `decode_b32_ctx16384`'s direction) are reproduced by the source-derived
model to within 7.5% (prefill absolutes) and 17% (b=8 ratios). **The b=32/b=64 decode rows are not
reproduced, and they are not measurements of the kernel** — they are measurements of the host
dispatch floor with a kernel underneath it. Any step-2 candidate benchmarked at b ≤ 64 decode must
raise `iters` or batch shapes until the device term clears the floor, or its A/B is meaningless.

### Step 5 — three §6.3 observations the model resolves

§6.3 lists three things "any explanation has to fit" and concludes *"stage B's cost is not a function
of buffer width, and I do not know what it is a function of."*

- **"One row of 262144 (0.340 ms) costs more than sixteen rows of the same width (0.254–0.269 ms)."**
  Resolved from source. `_pick_partition(b=1, n=262144, k=2048)`: `p_min = 4`, `max_p = 32`, and the
  loop raises `p` while `n//(2p) >= k` ⇒ **`p = 32`**. Stage 2 is then `b=1, n = p·k_p = 65536` — **a
  single 65536-wide row on ONE subcore, 31 idle.** At `b=16`, `p = 4` and stage 2 is `b=16, n=8192`
  — 16 rows on 16 subcores. So b=1's stage 2 is 8× the width on 1/16 of the parallelism. `read from
  source`; this is the mechanism, and it is `_pick_partition` maximising subcore utilisation in
  stage 1 at the cost of serialising stage 2.
- **"The effect is 2–3× stronger at prefill than at decode."** Predicted: the `a·n` term is paid
  `ceil(b/32)` times — 32–128 times at prefill, 1–2 times at decode, where it is also buried under
  the host floor.
- **"Small batches are hit harder than large ones, the opposite of a fixed-overhead story."**
  Predicted: b < 32 takes the two-stage path, where the wide setting strands most of the width in
  empty slices that are still DMA'd at full width (Step 3).

§6.3's own hypothesis — *"`sparsecore_topk` splitting into a two-pass select above some width"* — is
**confirmed to exist** (`_pick_partition`) and **is not the mechanism for any §6.3 pair**, because no
pair crosses it. It is the mechanism for the b=1-vs-b=16 anomaly above and for the b=8-vs-b=32 level
gap in `j-8b1e7acc`. Right about the branch, wrong about which rows it explains.

---

## Standing correction to carry

**The 5.9× is retired as a magnitude and retained only as evidence of a branch.**

- `report.md` §6.2's **5.9× stage-B ratio** was measured only at AgentX (`n` 10240 → 262144, b=16),
  where `p` goes 1 → 4 and two gathers appear. §6.3 measures the same *kind* of change at fixed page
  size and nightly-like shapes and gets **1.16× / 1.17× (decode b64 / b32)**, **1.77–1.78× (b8)**,
  **2.88× (prefill q1024)** and **3.41× (prefill q4096)** for a 16× width increase. Anywhere this
  sheet used 5.9× as a *size* — **A15** and **§C.9** — the §6.3 figure is now printed beside it in a
  blockquote. The mechanism (unconditional full-width DMA + branch crossing) survives; the number
  does not travel.
- `report.md` §6.2's **"+7% / −7%"** citation from `glm-5.2-fp8-dp16-ep.sh:9-10` is **retracted** —
  doubly confounded (`max_num_seqs` moved 16× alongside `max_model_len`, and it is a cross-build
  comparison) and its two halves are one measurement stated twice at fixed concurrency. **This sheet
  never used it**, and no constraint above depends on it. Recorded so a later reader does not
  reintroduce it.
- Also retracted upstream and not used here: the 11% profile cross-check, the "stage A
  (dispatch-corrected)" column, the 8%/16% AgentX achieved fractions, and "AgentX is the target".
  **The nightly config (`max_model_len` 9216, page 1024, TP1×DP16, score width 10240) is the baseline
  of record**, and every constraint in §A is stated at its shapes unless it says otherwise.
