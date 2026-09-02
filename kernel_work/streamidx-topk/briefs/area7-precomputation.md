# Area 7 — What is computed once, and where it is kept

Constraint sheet for `streamindex_topk` (GLM-5.2 DSA lightning indexer). Step 2 briefer output.
Reference: `.claude/skills/write-tpu-kernels/references/07-precomputation.md`.
Baseline of record: **nightly config** (`max_model_len` 9216, page 1024, TP1×DP16, score width 10240).
AgentX (`max_model_len` 262144, page 256, TP16×DP1) is a secondary operating point, **not the target**.

## A. Constraints

1. **SMEM is already half-spent by the framework.** `PallasMLAttentionBackend.get_max_num_seqs` is literally `1 MiB // 2 // pages_per_seq // 4` — a budget for `page_indices` (`attention.py:462-484`). Nightly that is 2304 B actually used; AgentX 256 KiB. A new per-(seq × page) table competes with a buffer the runner has already sized to half of SMEM. `from source` + `from 07-precomputation.md § The SMEM budget`.
2. **Every new metadata array must be shaped `max_num_seqs`(=64) / `max_num_tokens`, with the true count carried in `distribution`.** jit needs static shapes; the incumbent already follows the static-max-plus-count pattern. `from 07-precomputation.md`, `source: streamindex_topk.py:493-506`.
3. **A traced grid length is available and already used** — `grid = ((end_seq_idx - start_seq_idx)//seq_batch_size,)` with `end_seq_idx` from `distribution` (`:643`). 07 ranks this the highest-value way to spend metadata; a design must not regress to a static grid over `max_num_seqs`. `source` + `from 07-precomputation.md § Spending it`.
4. **A short grid does not buy back the per-invocation cost.** 26.2 µs fixed + 0.427 µs/step in 07's one case (`modelled` here, `verified: j-0a0a72c8` there). The incumbent pays it **63×/step** (3 `pallas_call`s × 21 layers) and **two of the three have grid 0 at every pure-decode step** (`distribution[1]==distribution[0]`, `decode_batch_end==distribution[0]` at b64). Measured envelope: 5.40 + 5.22 = **10.62 µs of the 125.25 µs per-layer TensorCore total = 8.5%** — `measured (j-a71a648c rank 0, §6.1 finding 6)`, at AgentX. Forbid a fourth `pallas_call` for a decode-empty case.
5. **No metadata in VMEM.** All four VMEM scratches (`:686-692`) are operand double-buffers; area (2) keeps the whole scoped budget. `source`.
6. **Narrow SMEM dtypes buy nothing here.** Total prefetch metadata is **2848 B** at nightly. At AgentX 259 KiB, 99% of it `page_indices`, whose values exceed int16. `source` + `verified: j-85d7b244 (dtype support)`.
7. **`num_scalar_prefetch` is being used with no `index_map` at all** — every `in_spec` is `memory_space=pltpu.HBM` and every transfer is a hand-rolled `make_async_copy`. 07's decision rule ("does an `index_map` need it?") returns *no*, which argues level 4 / `with_memory_space_constraint`. `source: :645-652, :713-719` + `from 07-precomputation.md § Deciding step 3`.

## B. Inventory of non-operand quantities (nightly config, T = padded token count, 21 indexer layers/step)

| quantity | invariant along | level incumbent computes it at | space / bytes | rebuilds per step |
|---|---|---|---|---|
| `num_sublanes_total`=80 → score width 10240 | whole run | 1 (trace, `:603-605`) | none (sets a T×40 KiB HBM buffer) | 0 |
| `pages_per_seq`=9, `page_size`, `kv_packing`, `lkv_dim` | whole run | 1 (`:592-594`, `:104`) | none | 0 |
| `bkv_sz`=2048, `num_sublanes_bkv`=16, `bq_sz`, `bkv_sz_per_kv_packing` | whole run | 1 (`:636-639`, `:106-108`) | none | 0 |
| stage B `p`, `n_p`, `k_p`, `slice_len`, `num_waves`, `b_pad`, `offsets` | whole run | 1 (`sparsecore_topk.py:322-326, 409, 419-424`) | `offsets` i32[p] const | 0 |
| `seq_lens` i32[64] | one step (all 21 layers) | 3-host (`tpu_runner:2992`) + 1 H2D | SMEM 256 B | 1 (**tail stale — §C6**) |
| `page_indices` i32[64×9] | one step | 3-host, `flat.zero_()`+fill (`attention_metadata.py:461-481`) | SMEM 2304 B (AgentX 256 KiB) | 1 |
| `cu_q_lens` i32[65] | one step | 3-host, tail explicitly refilled (`:2984-2990`) | SMEM 260 B | 1 |
| `request_distribution` i32[3] | one step (cached across steps) | 3-host (`:3171-3177`) | HBM 12 B | 1 |
| `distribution = stack([rd0,rd0,rd2])` | one step | **3-XLA per layer** (`mla_attention_op.py:183`) | HBM 12 B | 21 (unless CSE) |
| `decode_batch_end`, `start_end_seq_idx` i32[2] | one call | 3-XLA (`:755-756`, `:698`) | SMEM 8 B | 63 |
| `sem_ids` i32[3], `bo_sz` i32[2] | one **invocation** | **3-XLA constants** (`:699-700`) — belongs at 4 | SMEM 20 B | 63 |
| grid length `(end-start)//seq_batch_size` | one call | 3-XLA traced → dynamic grid (`:643`) | scalar | 63 |
| `scores_init` = -inf | one call (value is a constant) | 3-XLA broadcast, donated (`:745`, `:729`) | HBM T×40 KiB | 21 · **6.02 µs each** `measured (j-a71a648c)` |
| token→seq map (`seq_id`) | one step | **3-XLA, three times per call**: `searchsorted` (`mla:93`), [T,64] mask+reduce (`:494-497`), `searchsorted` (`:843`) | 65 KB bool intermediate in the middle one | 63 |
| `eff_row_lengths` i32[T] | one step | 3-XLA (`:823-830`) | HBM 4T B | 21 |
| early-exit predicate `max(seq_lens)<=k` + `lax.cond` | one step | 3-XLA (`:861`) | scalar | 21 (envelope: `conditional`+cond copies **1.230 ms/core-step, 4.6%** `measured`) |
| `lengths_p = clip(row_lengths-offsets,0,n_p)` | one call | 3-XLA (`sparsecore:424`) | i32[b·p] | 21 (AgentX only; p=1 nightly) |
| `q_lens`/`seq_lens`/`kv_lens` (4 slots) | one grid step | 5, 12 SMEM reads (`:118-126`) | SREG | grid·21 |
| `kv_len_max`, `num_bkv` | one grid step (per group) | 5, stack+max+cdiv (`:275-276`) | SREG | grid·21 |
| `k_span` iota i32[2048], `valid_mask`, `causal_mask` | per (grid, bq, bkv) step | 5 (`:340-351`) | VREG 8 KiB + masks | grid·num_bq·num_bkv·21 |
| `page_idx`/`safe_page_offset` clamp arithmetic | per bkv fetch | 5 (`:179-185`) | SREG | ×`bkv_p` per fetch |
| `i_hi`, `walk_chunks` in `_topk_kernel` | per row per wave | 5, from `row_lengths` (`sparsecore:151-157`) | SREG | correct level |

## C. Where the level is wrong

- **C1 (too far in) — the -inf init. Already closed by `report.md` §6.2; not a lever, listed only for completeness.** Per *call* it is **6.02 µs at 2.79 TB/s, ~2% of the call** `measured (j-a71a648c rank 0, report.md §6.2)` — i.e. it is running at bandwidth and there is nothing to win in it. The 21×/step multiplicity below is a *different* quantity (a per-step figure summed over all 21 indexer layers, 21 × 6.02 ≈ 126 µs/step) and is **not** comparable to the per-call percentage; do not read it as reopening the closed item. It cannot be hoisted (it is donated via `input_output_aliases`); it could only be *deleted*, since it exists to cover the columns the dynamic grid never visits, which `row_lengths` already declares out of range. If C5's probe succeeds the deletion comes with it for free. Do not spend a probe on C1 by itself.
- **C2 (too far in) — three token→seq maps per call, 63 per step.** The mask-reduce form in `_effective_row_lengths` materializes T×`max_num_seqs` bools (65536 at T=1024) to compute what `searchsorted` computes twice elsewhere in the same op. One shared level-3 `token_to_seq` i32[T] would serve all three and the kernel. Upper bound on the whole class: the profile's 1.230 ms/core-step "indexer plumbing" bucket.
- **C3 (too far in) — `sem_ids` and `bo_sz` are level-4 state shipped at level 3.** Pure per-invocation scratch (`jnp.zeros((3,))`, `jnp.full((2,),-1)`) emitted as XLA constants and passed through `num_scalar_prefetch`, 63×/step. `scratch_shapes=[pltpu.SMEM(...)]` is the right spelling; 07's `scratch_persists_across_grid_steps` `[verified: j-85d7b244]` is exactly the precondition.
- **C4 (too far in) — `kv_len_max`/`num_bkv` rebuilt per grid step** from 4 SMEM reads re-issued per grid step. The value is per-*group*; a level-3 `group_kv_len_max` i32[16] (64 B SMEM, one `reduce_max` over a reshape, CSE-able across 21 layers) turns a stack+max+cdiv into one SMEM load. 07 rule 4 verbatim: *hoist the value you consume, not the intermediate* — the incumbent hoists nothing and consumes a reduction. **Note for the designer:** if the length-sorting TODO (`:751-754`) is taken, the permutation is exactly this level-3 i32[64] vector (256 B SMEM) — but it breaks `start_send_scores` (`:137-151`), which writes `seq_batch_size * sz` **contiguous** score rows in one DMA on the assumption that a group's four sequences are adjacent in token order. Sorting costs 4 DMAs where there was 1.
- **C5 (too far out) — `slice_len` in stage B, and this is the largest one.** `_sc_topk_direct` bounds *every* scan loop by `walk_chunks = cdiv(i_hi)`, `i_hi = min(slice_len, row_lengths[row])` — but the DMA that fills `resident_vmem` is `pl.ds(s, slice_len)`, the full `n` words, unconditionally (`sparsecore_topk.py:154-161`). **Stage B's compute already tracks the real sequence length; its HBM read does not.** `slice_len = align_to(n, LANES)` is level 1, derived from `num_sublanes_total`, derived from `max_model_len` — sitting on the exact term §6.2 measured at 5.9× with byte-identical KV. It also sizes the SparseCore VMEM scratch, so `MAX_SLICE_WORDS = 65536` forces `p = cdiv(262144, 65536) = 4` at AgentX, which is what produces the profile's `sc_topk_direct_b64_n65536_k2048` and the two `gather_offload`s. This is finding (1)'s mechanism read out of source rather than inferred from a ratio.
- **C6 (too far out) — the `seq_lens` staleness. Confirmed; a *performance* bug only, and worth almost nothing at the baseline of record. Not a lever — keep it off the lever list.** Area 6 reached the same verdict independently via its E4, for the same reason (ISL 8192, k 2048, so the predicate is false with or without the stale tail); two agreeing readings, recorded here so nobody re-derives it a third time. `seq_lens_cpu` is zeroed once at allocation (`tpu_runner:710-714`) and thereafter only `[:num_reqs]` is written (`:2992`). It is the **only** one of the three metadata buffers not reinitialized: `query_start_loc_np[num_reqs+1:]` *is* explicitly refilled (`:2989-2990`) and the block-table scratch *is* `flat.zero_()`d (`attention_metadata.py:467`). The device slice is a fixed `seq_lens_cpu[:num_reqs_max_model_len]` = 64 entries at nightly (`:3059`), so slots `[num_reqs, 64)` hold whatever request last occupied them, and `streamindex_topk.py:861` reduces `jnp.max` over all 64 with `STREAMIDX_ENABLE_EARLY_EXIT = True` (`mla_attention_op.py:57`). **Not a correctness bug:** every other consumer is guarded — `_effective_row_lengths` by `distribution[2]` (`:496`, `:505`), `_fast_path` by `cu_q_lens[-1]` (`:854`), and the kernel's `seq_lens_ref` reads by a grid bounded by `distribution`. Two qualifications on impact: the predicate is a `max`, so it already requires *every* real request ≤ k=2048; and at the nightly workload (8192-token prompts) it is false with or without the tail — the loss lands on short-context and freshly-started batches, not on the baseline of record. Fix is one host line, `self.seq_lens_np[num_reqs:] = 0`. Worth recording because the failure is invisible: a fast path silently not firing is indistinguishable from a fast path not applicable.

## D. Gaps this block turns on, and one contradiction

- **"The CSE result did not isolate what it was meant to."** Turned on directly, and this block *is* the un-isolated case: all 21 layers get the same `seq_lens`/`cu_q_lens`/`distribution` tensors but a **different, donated `cache_kv`**, so the `pallas_call`s cannot collapse and only the metadata subgraph could. Decision that flips: if it CSEs, C2's three maps and the early-exit reduce are 3×/step and not worth moving; if not, they are 63×/step and belong on the host. **Checkable in an artifact already on GCS with no new job** — count instructions for the `_effective_row_lengths` / `searchsorted` subgraph in `j-a71a648c` rank 0's `hlo_stats`, exactly as §6.1 counted `StreamIdxTC-m` at 21. Note the report already has one data point saying CSE does *not* save you here: the -inf `%broadcast.33911` appears **21 times**. *(Closed — see "The CSE check" below.)*
- **"Where the dynamic grid's 26.2 µs fixed cost comes from is unknown."** Turned on: the incumbent's two empty `pallas_call`s cost 5.40 and 5.22 µs against 07's 26.2 µs, at an *identical* full-size output allocation — a second data point arguing the fixed cost is not allocation-driven, contra 07's own hypothesis. Decision that flips: allocation-driven → capping `num_sublanes_total` beats merging the three calls; invocation-driven → merging beats capping.
- **"Nothing here measures metadata at a realistic vLLM decode shape."** Closable for free from this block: 6 prefetch arrays, 2848 B at nightly and 259 KiB at AgentX, against 07's 261504-int32 ceiling `[verified: j-ed517191]`.
- **Not turned on:** `SmemWrapper` (all SMEM arrays here are 1-D and there are no `index_map`s), narrow-dtype capacity (2848 B), level 2 / model load (no weights in this op's metadata), and the iota-recompute gap — the incumbent's per-step recompute *is* an iota (`k_span`, `:340`), so 07's 98 ns/step figure is the right one here rather than the 540 ns upper bound.
- **Contradiction to flag.** report.md §8 lists as open: *"whether stage B's row width can be bounded by `seq_len` instead of `max_model_len`… what is unknown is whether the radix select's cost tracks it or the buffer width."* Source answers half of it (C5): **the scan tracks `row_lengths`, the HBM fill does not.** The probe should be re-scoped from "does the cost track `row_lengths`" to "can `sparsecore_topk.py:161`'s DMA take a dynamic — or power-of-two bucketed — length on SparseCore, and does `resident_vmem` still have to be allocated at `slice_len`". Secondarily, 07's SMEM table assigns `PrefetchScalarGridSpec` to the case where "the `index_map`s need it"; the incumbent uses it with no `index_map`s, so a designer copying the incumbent's spelling is copying a choice 07's own procedure would not have made.

## Convergence on C5

**C5 is now the best-established fact in step 2 — four independent routes, one conclusion.** Areas
2, 5, 6 and 7 each arrived at it from their own material, without coordination: the scan loops in
`_topk_kernel` are bounded by `i_hi = jnp.minimum(slice_len, eff_len)` and `walk_chunks =
_cdiv_dyn(i_hi)` (`sparsecore_topk.py:150-157`), while the DMA that fills `resident_vmem` at
`sparsecore_topk.py:160` reads `scores_hbm.at[pl.ds(s, slice_len)]` with a **static**
`slice_len = _align_to(n, LANES)` (`:322`), unconditionally, inside the same `@pl.when(i_hi > 0)`.
Compute tracks the real row length; the HBM read does not. `from source`, corroborated by areas 2,
5 and 6.

**Keep the re-scoped probe** (§D's contradiction). `report.md` §8 asks *"does the radix select's
cost track `row_lengths` or the buffer width"*; source has already answered that half — the scan
tracks `row_lengths`. The question worth a job is the other half, and it is two questions that must
be answered together:

> **Can that DMA take a dynamic — or power-of-two-bucketed — length on SparseCore, and must
> `resident_vmem` still be allocated at `slice_len` regardless?**

A yes/no on the first alone decides nothing: a dynamic-length DMA that still requires a
`slice_len`-sized destination saves HBM traffic but not capacity, which is the term that sets `p`.

**Area 5's capacity arithmetic, folded in for the second half.** The resident slice is
`4 · align(n/p, 16)` bytes against **524288 B per subcore**, with roughly **26 KB** of other
scratch (`LANES·NUM_BUCKETS+LANES` histogram, `NUM_BUCKETS+LANES` refinement, `k+LANES` output,
`b_pad` lengths, DMA semaphores — `sparsecore_topk.py:330-341`). That leaves ~498 KB for the slice,
i.e. ~124k words, against `MAX_SLICE_WORDS = 64 * 1024` (`:44`). **`MAX_SLICE_WORDS` is conservative
by ~1.5×** `modelled (area 5)`. Raising it is the cheapest thing in this whole area: at AgentX it
takes `p = _cdiv(262144, 65536) = 4` down to `p = 2`, which halves the two-stage merge and the
`gather_offload`s; at nightly (`n` 10240) `p` is already 1 and it changes nothing. **Test it before
believing it** — a 1.5× headroom estimated from a source read is exactly the kind of number that
loses to an allocator's alignment or a reserved region.

## The CSE check — closed, no new job

**Verdict: the metadata subgraph does NOT CSE across the 21 indexer layers. C2's maps are
per-layer, and they belong on the host.** This is the `63×/step` branch of §D's decision, not the
`3×/step` branch.

`measured (j-a71a648c rank 0)` — `hlo_stats`, `rows_kept` 16554 of `rows_total` 16554 (untruncated),
single program id `17643488860325271500`, 36 module executions per core × 2 cores = **72
occurrences per instruction per capture**.

**How it was counted, and what did *not* work.** `Source Info` is useless for this: it stops at the
torch custom-op boundary, and `streamindex_topk.py` and `sparsecore_topk.py` appear in **zero** of
the 8731 rows that carry any source info at all (the deepest vLLM frames available are
`mla_attention_op.py` at 256 rows and `attention.py` at 390). Grepping the extracted `hlo_proto.pb`
for the Python names returns 0 as well. The column that *does* discriminate is **`Framework op
name`**, which carries the JAX trace scope; the whole op sits under
`custom_kernel/custom_kernel/cond/branch_0_fun/…`, i.e. inside `_common_path`. Counting distinct
HLO instructions under each scope is exactly the operation that gives `StreamIdxTC-m` = 21, so the
metadata counts below are on the same footing as §6.1's.

| `Framework op name` (prefix `custom_kernel/custom_kernel/`) | distinct HLO instrs | per layer | total self µs | µs / step / core |
|---|---|---|---|---|
| `cond/branch_0_fun/StreamIdxTC-m-bq_128-bkvp_2/…/pallas_call` | **21** | 1 | 7789.2 | 108.2 |
| `cond/branch_0_fun/StreamIdxTC-d-bq_1-bkvp_2/…/pallas_call` | **42** | 2 | 179320.1 | 2490.6 |
| `cond/branch_0_fun/lt` | **21** | 1 | 1732.2 | 24.1 |
| `cond/branch_0_fun/reduce_sum` | **21** | 1 | 175.7 | 2.4 |
| `cond/branch_0_fun/select_n` | **21** | 1 | 8.6 | 0.1 |
| `cond/branch_0_fun/dynamic_slice` | **21** | 1 | 956.7 | 13.3 |
| `cond/branch_0_fun/gather` | **105** | 5 | 6387.0 | 88.7 |
| `cond/branch_0_fun/slice` | **42** | 2 | 7010.1 | 97.4 |
| `cond/branch_0_fun/concatenate` | **42** | 2 | 2860.8 | 39.7 |
| `jit(_streamidx_topk_jax)/scatter` | **21** | 1 | 4516.4 | 62.7 |
| `sc_topk_direct_*/mpmd_map` | 168 | 8 | 181366.6 | 2519.0 |
| `sc_dsa_gather/mpmd_map` | 312 | — | 166497.8 | 2312.5 |
| `gather_page_ids/pallas_call` | **21** | 1 | 3346.0 | 46.5 |
| `cond/branch_1_fun/*` (the `_fast_path`) | **0** | — | 0 | 0 |

**Every metadata row is an exact multiple of 21.** Not one instruction is shared between layers.
The shapes confirm these are the intended ops rather than a name collision:

- `%iota_compare_fusion.5 = pred[64] fusion(s32[])` — `jnp.arange(64) < num_seqs`, the `max_num_seqs`
  guard at `streamindex_topk.py:496`.
- `%convert_reduce_fusion = s32[16] fusion(s32[64], pred[64])` — the `[num_tokens, max_num_seqs]`
  mask reduced to one seq id per token (`:494-497`). This *is* `_effective_row_lengths`' searchsorted
  substitute: 64 → 16, `max_num_seqs` → `num_tokens`.
- `%fusion.61 = (s32[16], s32[16], s32[16]) fusion(s32[16])` — a **three-way** fan-out of the token→seq
  map, which is C2's three consumers, emitted once per layer.
- `%dynamic_slice.27 = s32[1] dynamic-slice(s32[65], …)` — `cu_q_lens[num_seqs]`, the `:505` validity
  bound (65 = `max_num_seqs + 1`).

**Corrections to the `63×/step` figure in §B and §C2, now that it is measured rather than argued.**
The honest count is **42 token→seq map builds per step, not 63**: `_fast_path`'s `searchsorted`
(`:843`) lives in `branch_1_fun`, which has **zero rows in the whole table** — it never executed in
this capture. That is an independent confirmation of C6/area 6's E4 (the early-exit predicate is
false at this workload), obtained from a different column of the same artifact. The two that do run
— `_effective_row_lengths`' mask-reduce and the KV scatter's map, the latter fused into
`scatter_fusion` — are 21 each.

**Cost of the class.** Summing the seven metadata scopes (`lt`, `reduce_sum`, `select_n`,
`dynamic_slice`, `gather`, `slice`, `concatenate`) gives **19131.1 µs / 265.7 µs per step per core**,
which is **1.1%** of the 23.675 ms module step and **5.2%** of the three indexer kernels
(`StreamIdxTC-d` + `StreamIdxTC-m` + `sc_topk_direct` = 5117.8 µs/step). Two things the designer
should take from the breakdown rather than the total:

1. **The `_effective_row_lengths` reduce itself is nearly free** — `reduce_sum` + `select_n` are
   2.5 µs/step combined. C2 is worth doing for the *dependency* it removes (one shared level-3
   `token_to_seq` unblocks hoisting) and for the 65 KB bool intermediate, **not** for its own µs.
2. **The expensive metadata is the scalar plumbing around the `pallas_call`s, not the maps.**
   `slice` (97.4) + `concatenate` (39.7) + `gather` (88.7) = 225.8 of the 265.7 µs — that is
   unpacking `distribution` into three scalars, rebuilding `jnp.array([start_seq_idx, end_seq_idx])`
   and `jnp.zeros((3,))`/`jnp.full((2,), -1)` per call, and per-token gathers. C3 (`sem_ids`/`bo_sz`
   as level-4 scratch) and the `distribution` rebuild at `mla_attention_op.py:183` sit squarely in
   that 225.8 µs. **C3 was ranked third in §C; on this evidence it should be ranked above C2.**

**Two caveats on these µs, and they matter.**

- **This capture is at the 262144-wide operating point, not the nightly baseline of record.** Both
  `StreamIdxTC` kernels output `f32[16,2048,128]` — 16 tokens × **262144** scores — and
  `sc_topk_direct_b64_n65536_k2048` is present, i.e. `p = _cdiv(262144, 65536) = 4`. Nightly's score
  width is 10240 and `p` is 1. **The instruction *counts* are structural and carry over unchanged**
  (21 indexer layers, one `lax.cond`, no CSE — none of that depends on `max_model_len`). **The µs do
  not**, and neither does the 1.1% / 5.2% share; do not quote them as nightly figures.
- **The artifact is the one `report.md` §6.1-6.2 read**, which is a useful cross-check rather than a
  new fact: `StreamIdxTC-m` here is 5.2137 µs avg self against the report's 5.22, and
  `StreamIdxTC-d` averaged over all 42 instructions is 60.0134 µs, which is (114.63 + 5.40)/2 to
  four figures — the report's bimodal pair, at equal weight. The reads agree.

**Why there is no CSE, mechanically** — so the designer does not expect a compiler flag to fix it.
`_common_path` is a `lax.cond` branch, which XLA compiles to its own computation per call site
(`region_44.52_spmd`, `region_95.108_spmd`, `region_146.164_spmd`, … — 86 distinct `region_*_spmd`
names in the module, 84 `conditional`s). CSE operates within a computation. Every indexer layer gets
its own region, so identical metadata arithmetic in 21 regions stays 21 copies. Hoisting has to
happen **above** the `cond` — at level 3 in the caller, or on the host — and cannot be delegated to
the compiler. `from source` + `measured (j-a71a648c rank 0)`.
