Read: the full reference (1810 lines), `AREAS.md`, step 1's `report.md`, `sparsecore_topk.py` (457 lines), and the caller. Checkout pins `jax==0.10.2` (`pyproject.toml:50`) — the same version every probe in the reference file ran on.

---

# Constraint sheet — area (8), SparseCore — `sparsecore_topk` in the GLM-5.2 DSA indexer

## 1. What the SparseCore forbids or requires, at these shapes

- **No `dot_general`, at all** [reference: `sc_pl_ops_inside_kernel`, verified j-996520b8-2265-4c2c-b002]. The scoring einsum can never move onto this unit. The scoring/selection split is a *hardware* boundary, not a design choice.
- **One Pallas kernel cannot hold a TC mesh and an SC mesh** [verified j-996520b8, `sc_tc_one_kernel_two_meshes`; second refusal at `pallas_call_registration.py:606-613@jax 0.10.2`]. Consequence for this block: **the f32 scores buffer is forced to cross HBM.** Area (1) cannot fuse it away. At `mixed_64k` that is a mandatory 276.8 MB round trip (report.md's figure), read back at SparseCore bandwidth.
- **No transcendentals but `exp`; no `argmax`/`argmin`, `transpose`, `pad`, `broadcast_to`, RNG.** ReLU (`max`), multiply and compare all lower, so the scoring *tail* is not forbidden by the primitive set — only by the matmul above it.
- **480 KiB usable VMEM per vector subcore, not 512** [verified j-996520b8, `sc_vmem_per_subcore`: 448 and 480 KiB accepted, 512 refused with `'memref.alloca' op E3000`]. Not poolable across subcores. `MAX_SLICE_WORDS = 65536` (`sparsecore_topk.py:45`) = **256 KiB, i.e. 53% of the real budget** — see alternative A5.
- **SMEM 4 KiB, int32/f32 only** [same job, `sc_smem_capacity`]. The kernel uses no SMEM; it carries state in `plsc.parallel_loop(..., carry=)`, which is legitimate [`sc_pl_loop_init_carry`, verified j-f424981c-75b6-4619-b35d].
- **`needs_layout_passes=False` is set** (`sparsecore_topk.py:345`), so the strict `(16,)` lane-group rule is in force on every `Get`/`Swap` [`sc_lowering.py:142-143,243-244@jax 0.10.2`]. Every access in the kernel is `pl.ds(i*LANES, LANES)` — that is why, not style.
- **`disable_bounds_checks=True`** (`:344`). `rowbuf_vmem` is `k + LANES` words and `emit_body`'s `woff` is a carried dynamic offset; an overrun is silent wrong output, not an error. This is the reference's documented candidate-buffer hazard ("Re-scan beats compacting…": sized optimistically it "overflowed on 3 of 5 simulated inputs and returned a *wrong answer* rather than an error"). **There is no unit test for `sparsecore_topk`** — see §6.

## 2. Tagged facts, with units and scope

| fact | value | unit / scope | tag |
| --- | --- | --- | --- |
| SC streaming bandwidth | **337 GB/s** (79 6.9 µs median, 1% spread) | GB/s, **whole chip: 2 cores × 16 subcores per TensorCore**, `2*x` over 256 MiB f32 | `measured` [verified j-74a24183-08a9-49f3-a822, `sc_vs_tc_streaming_bandwidth`, uncontended pod] |
| same, contended | 338 GB/s | identical scope | `measured` [j-996520b8] — the two agree to 0.3%, so the rate is a figure |
| TC streaming, same probe | 2916–2941 GB/s (78.8–79.5% of a 3.70 TB/s peak) | GB/s, per TensorCore | `measured`, same jobs. **SC is 8.7× slower at bulk data** |
| per-subcore rate | 11.8 GB/s (1 of 32); 279 GB/s at 32 | GB/s | `measured` [`sc_subcore_scaling`, j-996520b8] — the ceiling is bandwidth, not sync |
| uncoordinated 32-subcore scaling | **23.70× / 24.14×** | ratio vs 1-core/1-subcore | `measured` [j-996520b8, j-f424981c] |
| SC kernel fixed cost | **≈ 35 µs** | µs, launch+setup floor of *a gather kernel*, not of this one | `measured`-by-analogy [`sc_gather_index_count_scaling`, j-75744a32-7f25-4a78-a7a5: 36.5 µs at 2^12 indices → 71.9 µs at 2^20] |
| overlap behind a **compute-bound** TC neighbour | 432.3 µs of SC work adds **61.2 µs** (score 0.86) | µs added to the step | `measured` [j-d2494d38-0074-49ec-8979, solo pod, `sc_tc_overlap_vs_neighbour_kind`] |
| overlap behind a **memory-bound** TC neighbour | the same 432.3 µs adds **346.2 / 374.8 µs** (score 0.45 / 0.55) | µs added | `measured`, same job. Halving the neighbour's HBM traffic did not recover it |
| serial SC→TC handoff | 0.963 | ratio, chained ÷ summed | `measured` [`sc_tc_dependent_handoff`, j-0ed646a2]. Crossing units is free; **the work still costs its full duration** |
| SC radix-select top-k vs `lax.top_k`, standalone | 1.54× / 3.79× / 3.83× / **9.08×** slower at n = 4K/16K/16K/64K, k=64 | ratio SC÷TC, **1 of 32 subcores** | `measured` [`sc_topk_vs_lax_topk`, j-0ed646a2] — **does not transfer**: that probe is single-subcore, `sparsecore_topk` is 32-subcore |
| the same top-k *beside a matmul* | +20.9 µs vs `lax.top_k`'s +17.5 µs → 1.19, SC loses | ratio of added times | `measured` [`sc_topk_frees_the_mxu`, j-d08138bf] |
| XLA gather cliff | flat ~30 µs to 98 304 indices, **1 608.9 µs at 131 072** — 50× in one step | µs, 1-wide f32 rows, 65 536-row table | `measured` [j-90026b0b, `sc_xla_gather_cliff_bracket`]; above it ~12 ns/index |
| SC scatter-add vs XLA | 6.6× slower on 1 subcore; **1.9× slower on all 32** (3.38× fanout only) | ratio | `measured` [j-d08138bf, j-48095a41]. **Does not transfer to this kernel** — see §5 |
| VPU / VMEM bandwidth on either unit | — | — | **`unmodelled`** — no rate anywhere in the reference or `V7X.md` |

## 3. What `sparsecore_topk.py` actually does

Structure: `sparsecore_topk` (`:381`) → `_pick_partition` (`:360`) → one or three `_sc_topk_direct` calls (`:306`), each a `VectorSubcoreMesh(num_cores=2, num_subcores=16)` kernel (`:327-332`) — **all 32 subcores, no `subcore_barrier`, no `fetch_and_add`, zero inter-subcore coordination.** Rows are the parallel axis: `row = w*32 + core*16 + subcore` (`:148`), `num_waves = ceil(b/32)` (`:325`), `lax.fori_loop` over waves (`:303`).

Per row, inside `wave_body` (`:147`):

| step | line | bound | cost axis |
| --- | --- | --- | --- |
| lengths DMA | `:94-97` | `b_pad` words | once per kernel |
| **scores DMA into `resident_vmem`** | **`:160`** | **`pl.ds(s, slice_len)` — the full padded width** | **O(padded width)** |
| monotone key convert `cvt` | `:162-166` | `walk_chunks` | O(`row_lengths`), 1 read + 1 write |
| L0 histogram `hist0_body` | `:178-187` | `walk_chunks` | O(`row_lengths`), 1 read + 1 `addupdate_scatter` |
| `fold_priv` + `scan_glob` + `find_bucket` | `:119-145`, `:188-202` | `NUM_BUCKETS=256` | **O(1) per row ≈ 4 096 words of VMEM traffic per level** |
| 3 refinement levels `ref_body` | `:214-256` | `walk_chunks`, gated to 0 by `done` | O(`row_lengths`) × 1–3 |
| emit `emit_body` | `:269-291` | `walk_chunks` | O(`row_lengths`), 1 read + `store_compressed` |
| output DMA | `:295-299` | `k` words | once per row |

**Answer to step 1's question: the cost is O(`row_lengths`) *except for one term, and that term is the HBM DMA at `:160`, which is O(padded width).*** `walk_chunks = ceil(i_hi/16)` with `i_hi = min(slice_len, eff_len)` (`:155-156`) — every compute loop follows `row_lengths`. `slice_len = align_to(n, 16)` (`:324`) with `n` the padded width, and the DMA moves all of it. On a unit whose measured ceiling is bandwidth and whose arithmetic is nearly free once resident (eight chained `exp`s cost 23% more than one multiply — "minimise trips, not work"), that is the term on the expensive axis.

**Padded width also has a second, structural effect that is not linear:** `_pick_partition` reads `n`, not `row_lengths` (`:365`, `p_min = ceil(n / 65536)`), so raising `max_model_len` past 65 536 columns flips the whole call onto the two-stage path — three SC kernel launches instead of one, plus **two `jnp.take_along_axis` gathers on the TensorCore** (`:439`, `:452`) that are not SparseCore work at all.

`_pick_partition` at each regime (`b` = score rows, `n` = padded cols):

| regime | b, n | p | path | rows in stage 1 | **subcore occupancy** |
| --- | --- | ---: | --- | ---: | ---: |
| `decode_prod` | 64, 10240 | 1 | direct, `num_waves=2` | 64 | 32/32, full |
| `mixed_prod` | 1024, 10240 | 1 | direct, `num_waves=32` | 1024 | 32/32, full |
| `mixed_32k` | 1024, 34816 | 1 | direct, `num_waves=32` | 1024 | 32/32, full |
| `decode_64k` | 6, 67584 | **4** | two-stage | 24 | **24/32 = 75%; stage 2 is 6/32 = 19%** |
| `mixed_64k` | 1024, 67584 | **2** | two-stage | 2048 | 32/32, but 2× the launches |

`decode_64k`'s `p=4` is set by `max_p = 32 // b = 5` rounded down to a power of two (`:370-376`) — 8 subcores are structurally idle in stage 1 and 26 in stage 2.

**Two contradictions with the reference file, both on hardware evidence:**

- **`plsc.store_compressed` lowers and is correct.** The reference lists it under `## Gaps` as not lowering in either flag setting (`Not implemented: Mask is not supported`). This kernel calls it at `:279-281` with a mask and a **dynamic** `TransformedRef` offset, under `needs_layout_passes=False`, on the **same jax 0.10.2**, and step 1's correctness job `j-6d675a58-7ecd-4b44-9171` passed at 0.995 recall with the negative control firing. The gap is therefore about the *spelling*, not the primitive — which is exactly what the reference warns of ("Do not read one primitive's mask refusal as evidence about another's"). The reference's `cumsum` + `store_scatter` workaround, which costs an extra pass, is **not** required here.
- **The `scan_count` histogram idiom is not the only correct one.** This kernel gives each of the 16 lanes a private 256-bucket histogram (`addr = lane_iota * NUM_BUCKETS + bucket`, `:182`, `priv_vmem` sized `16*256+16`, `:348`) and folds at `:119-128`, avoiding `scan_count` entirely. That matters for costing: the reference attributes scatter-add's 3.38× fanout saturation to "the per-vector `scan_count` prefix scan that serialises duplicate indices *within* a vector" — a cost this kernel does not pay. **Do not price this kernel with the `sc_scatter_add_*` rates.**

## 4. Which of the reference's `## Gaps` this block turns on

| gap | decision it would settle for this block |
| --- | --- |
| **"Why the memory-bound overlap collapses"** | Decides alternative A4 outright. The only neighbour available to hide `sparsecore_topk` behind is the *next* call's scoring kernel, which step 1's roofline says is **HBM-bound at decode** (`floor = hbm_id = 41.7 µs` against `mxu` 19.2/9.6 µs). Measured, a memory-bound neighbour turns 432 µs of SC work from +61 µs into +346 µs — 80% of it lands on the step. If that holds here, overlap buys almost nothing at decode. It is unresolved *why*, and the reference's own next step is a long memory-bound neighbour, which this block supplies. |
| **"`store_compressed` / `load_expanded` do not lower"** | Already contradicted above. Settles whether a step-3 variant may use masked compaction directly (one pass) or must pay `cumsum` + `store_scatter` (two). |
| **"XLA's gather has two implementations, threshold 2^17, mechanism unknown"** | Decides the two-stage path's real cost. `decode_64k`'s stage-2 gather is 6 × 8192 = 49 152 indices (below the 2^17 cliff); `mixed_64k`'s is 1024 × 4096 = 4.19 M (far above). At the cliff's 12 ns/index that second one models to **50 ms** against a whole measured `topk ctrl` of 3.01 ms, so `take_along_axis` over `axis=1` is plainly *not* the probe's HLO — the rate does not transfer, and which HLO it is has never been looked at. This is the same "needs the emitted HLO" the reference names. |
| **"Everything here is single-subcore except gather/scaling/bandwidth"** | This block **closes it**: `sparsecore_topk` is a shipping, correct, 32-subcore algorithm that reaches full fan-out by partitioning *rows*, not by coordinating within one. The reference's standing caveat that both worked kernels leave ~20× unclaimed does not apply here — **that headroom is already spent.** |
| "`subcore_barrier` ceiling is 4" | Turns on only for `decode_64k`, whose stage 2 runs 6 rows on 32 subcores (19%). Splitting one row across subcores needs coordination, capped at 4 of 32 — so a coordinated fix buys at most 4×, against the hierarchical merge already shipping. |
| scalar+vector MPMD; MoE-routing-is-synthetic | **Not turned on.** Vector-mesh only, no routing here. |

## 5. Occupancy and cost arithmetic, as far as a briefer can take it

**The one term with a rate.** Padded scores bytes the SparseCore must pull over HBM = `b × slice_len × 4`, divided by the 337 GB/s whole-chip figure [tag: `measured`, j-74a24183]:

| regime | padded words moved | `modelled` HBM µs @337 GB/s | measured `topk ctrl` µs (report.md) | share |
| --- | ---: | ---: | ---: | ---: |
| `decode_prod` | 655 360 | 7.8 | 55.62 | 14% |
| `decode_64k` (2-stage) | 454 656 | 5.4 | 86.15 | 6% |
| `mixed_prod` | 10 485 760 | 124.5 | 399.60 | 31% |
| `mixed_32k` | 35 651 584 | 423.2 | 1 210.32 ⚠ | 35% |
| `mixed_64k` (2-stage) | 73 400 320 | 871.2 | 3 009.69 ⚠ | 29% |

⚠ = the two rows report.md flags as resting on a 514.8% / 133.1% control spread.

Two things fall out, and they point in opposite directions:

- **At today's configs the padded DMA is at most ~35% of the selection time, and the padding in it is small.** live/padded is 8192/10240 = 80% at `_prod` and 65536/67584 = 97% at 64k. Sizing the DMA to `row_lengths` therefore saves ≤20% of ≤35% — **≤7%** [`modelled`].
- **At step 1's confound scenario it is the whole problem.** Raise `max_model_len` to 66 560 while requests stay at kv_len 8192: `mixed_prod` becomes b=1024, padded 67 584, live 8192 — live/padded = **12%**. The padded DMA alone models to **821 µs** against the 399.60 µs the whole selection costs today. That is a **modelled ≥2× regression on short requests, caused only by the config**, and it also flips `p` from 1 to 2, adding a launch and a TC gather. Step 1's worry is quantified and it is large.

**A derivation the existing numbers already support, with its confound named.** `decode_64k` moves **fewer** padded words than `decode_prod` (454 656 vs 655 360, 0.69×) and takes **1.55×** longer. Per-subcore live elements are near-identical (`decode_prod` 2 waves × 8192 = 16 384; `decode_64k` stage 1 ≈ 16 896 on 24 of 32 subcores). The 30.5 µs difference is close to the reference's ~35 µs SC launch floor, and `decode_64k` makes **two extra SC launches plus two TC gathers**. So at these two points the cost tracks per-subcore live elements and launch count, not padded width. **Confounded** — one-stage vs two-stage changed at the same time — so it narrows the question rather than closing it. The probe stands: hold `max_model_len` at 64k, sweep the real KV length, and **force `p` constant across arms** or the confound reappears.

**What has no rate — write the word, not a number.** Per live element the kernel does 1 read + 1 write (`cvt`), 1 read + 1 scatter (`hist0`), 1 read + 1 masked scatter × 1–3 (`ref_body`, gated by `done`), 1 read + 1 `store_compressed` (`emit`) = **3–6 VMEM reads, 1 write, 1–4 scatters, 1 compressed store**. Plus a per-row **fixed** ≈ 4 096 words of VMEM traffic per refinement level (`fold_priv` = 16 lane-groups × 16 accumulate reads, `:119-128`) — at `decode_prod` that is 64 rows × 4 levels × 4 096 ≈ 1.05 M words against ~3.1 M words of live-pass traffic, so **~25% of VMEM traffic is per-row fixed cost, and it grows as rows get shorter.** Time for all of this: **`unmodelled`** — neither the reference nor `V7X.md` carries a VMEM or scatter rate for this unit, and the `sc_scatter_add_*` rates do not transfer (§3).

**VMEM budget** [`modelled` off a `measured` capacity]: non-resident scratches are `priv` 16.06 KiB + `glob` 1.06 KiB + `rowbuf` 8.06 KiB + `len_vmem` (b_pad×4, ≤4 KiB) ≈ **29.2 KiB**. Against the measured 480 KiB that leaves ≈ **115 000 words** for `resident_vmem`, versus the `MAX_SLICE_WORDS = 65536` the file budgets. Caveat, and it is why this is `assumed` not `modelled`: 480 KiB was measured for a *single* f32 alloca, and this kernel has five — whether the allocator sums them the same way is unestablished, and the check is one compile.

**Variance.** Nothing in the kernel's structure predicts *rep-to-rep* variance at fixed input: every data-dependent branch (`done`, `keep_all`, `walk_chunks`, `b_star`) is deterministic given the scores tensor, so refinement depth is fixed across reps unless the harness regenerates the tensor. What the structure does predict is **size-dependence** — `mixed_32k`/`mixed_64k` put 32 subcores each pulling 132–136 KiB concurrently, ~4.3 MiB of in-flight DMA per wave, against `mixed_prod`'s 1.25 MiB. Three precedents in the reference say large SC spreads of exactly this magnitude were **outliers, not a wandering baseline**: 43% on the streaming probe collapsed to 1% on a solo pod while the median moved 0.3%; a 254% spread with median and min 1% apart; a 122% spread on the wide-`Get` arm. **The discriminator is min vs median**, which report.md does not carry — that is the cheapest probe on this sheet.

## 6. There is no test for `sparsecore_topk`

Confirmed. `grep -rn "sparsecore_topk" tests/` returns nothing; `tests/kernels/deepseek_v4/` holds `compress_store_test.py`, `csa_gather_test.py`, `mhc_test.py`, `mla_swa_test.py`, `mla_test.py`, `o_projection_test.py`, `rope_test.py`, `sparse_mla_test.py` and none of them touches it. The only coverage is indirect, through `tests/kernels/test_dsv4_streamindex_topk.py` exercising the whole `streamindex_topk` call, plus step 1's own `instruments/test_correctness.py` (decode + `mixed_prod`, negative control fired). Step 1's belief is right. Combined with `disable_bounds_checks=True` and a dynamic `woff`, **step 3 changing anything inside this kernel has no unit-level net under it** — and the reference names the test that would catch the two failure classes it has seen: an input where every key is identical, where the right answer is `n` and nothing else.

## 7. Alternatives — enumerated, not ranked

- **A1 — size the resident DMA to `row_lengths` instead of `slice_len` (`:160`).** Removes the only O(padded) term. A static `pl.ds` size forbids a traced length, so it must be a `pl.loop` over fixed windows with a **traced bound** — the exact shipping idiom the reference records as `ragged_gather_reduce_v2`'s ("bound the resident scratch with a window, not with the input size… loop bounds may be traced values"). Modelled gain **≤7% today, ≥2× in the raised-`max_model_len` scenario**.
- **A2 — narrow the scores buffer before selection** (bf16, or a two-pass coarse threshold on the TC that hands the SC a compacted candidate list). Halves the DMA term. Costs an extra TC pass over the same buffer and an accuracy gate, since a coarse pre-filter is an approximation (area 6). bf16 on the SC is carried as packed uint32 and widened by shift+mask+bitcast, measured exact, with `plsc.pack(first, second)` the correct order [`sc_bf16_widen_and_pack`, verified j-97fa6617].
- **A3 — size `num_sublanes_total` to the batch's real KV length rather than `max_model_len`** (`streamindex_topk.py:603-605`). Attacks the same term one level up, and shrinks the `-inf` fill and the scoring kernel's write too, so it is cross-area. Costs a recompile per distinct width — area (7): the quantity is invariant per *batch shape*, not per run, and JAX will retrace. **Expected to lose on shape-polymorphism grounds; enumerated because it is the only option that also fixes the `p`-flip in `_pick_partition`.**
- **A4 — overlap the SparseCore with the *next* call's TensorCore scoring work.** The reference's central offload thesis, and the measured evidence points against it here: the neighbour is HBM-bound at decode (step 1's roofline: `floor = hbm_id`), and behind a memory-bound neighbour the same SC work went from +61 µs to +346 µs (score 0.86 → 0.45/0.55) [j-d2494d38]. Also requires breaking a real data dependency, which the reference says must be sourced "from a different layer or the previous step". **Expected to lose at decode; genuinely open at `mixed_prod`, where the fp8/bf16 question decides whether the neighbour is MXU-bound.**
- **A5 — raise `MAX_SLICE_WORDS` from 65 536 toward ~115 000.** Makes both 64k regimes single-stage: removes two SC launches (≈35 µs each, `measured`-by-analogy) and **both `jnp.take_along_axis` gathers on the TensorCore**, whose cost is currently `unmodelled` and which sit on the open XLA-gather-cliff gap. Cheapest change on this sheet; gated only on the `assumed` five-scratch VMEM accounting.
- **A6 — keep the two-stage select but fix `decode_64k`'s occupancy.** `_pick_partition` forces `p` to a power of two, so 6 rows get `p=4` = 24 of 32 subcores in stage 1 and 6 of 32 in stage 2. Allowing non-power-of-two `p`, or fanning stage 2's 6 rows over sub-slices, recovers up to 1.33× on stage 1. **Small and regime-specific; enumerated because it is the only option that touches occupancy rather than bytes.**
- **A7 — move part of the selection back to the TensorCore** (e.g. a TC coarse bucket-count feeding an SC refine). The log already closed the *whole-algorithm* swap (`lax.top_k` 13.77%/27.22% slower, `argpartition` worse, `j-e83d16c4`), and standalone SC top-k loses 1.5–9× to `lax.top_k` [j-0ed646a2] — but both of those compared *whole* algorithms at *single-subcore*, and neither is a split. **Expected to lose; enumerated because the split has never been measured and the two closed results do not cover it.**
- **A8 — do nothing to the kernel; treat the selection half as a fixed cost and spend step 3 on the scoring half.** The null option. Its cost is Amdahl: report.md's 33% / 43% / 60% selection shares mean a scoring-only win at `mixed_prod` is diluted 2.5×.

---

**Two things I decided nothing about and am flagging as such:** every option above is enumerated, none is ranked, and no time in §5 that lacks a rate has been given a number. The `store_compressed` finding is a correction to the reference file's `## Gaps`, sourced to shipping code plus a named correctness job — per `AREAS.md`, that answer belongs written back into `references/08-sparsecore.md`, which is not mine to edit.
