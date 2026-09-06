# Area (5) constraint sheet — which unit does each piece of work

Reference: `.claude/skills/write-tpu-kernels/references/05-compute-units.md`. Block: `streamindex_topk.py`, `compute_scores` L304-354. Shapes and baselines from `kernel_work/streamindex_topk/report.md`; I re-derived only counts, not times. I choose nothing below.

---

## 1. What this area forbids or requires at step 1's shapes

- **Requires:** a bf16 accumulator is *rejected outright inside a Pallas kernel* [verified: j-4fba8dc6-d72c-41f9-948b, `accumulator_precision`, `VerificationError`]. The einsum's `preferred_element_type=jnp.float32` (L330) is not a choice, it is the only legal one. f32 accumulation is also free — 764.8 vs 772.3 Tops/s, under 1% [verified: j-ed517191-5f2d-4d00-8c0c, `mxu_peak_by_dtype`].
- **Forbids:** quantizing the einsum operands to int8 for MXU throughput. Generation 7's MXU admits twelve dtype pairs and **no integer pair**; int8 measures 746.8 Tops/s against bf16's 764.8 because it is promoted [source: `tpu_info.py:is_matmul_supported@jax 0.10.2`; verified: j-ed517191, `mxu_peak_by_dtype`]. fp8 is the only narrow dtype that buys compute — 1419.9 Tops/s, 1.86× bf16.
- **Forbids:** true f32 matmul anywhere on the hot path. `float32→f32 HIGHEST` is 147.5 Tops/s, a 4.5× cliff below `DEFAULT`'s 661.4, which is not f32 arithmetic at all [verified: j-ed517191].
- **Forbids** (SparseCore, condition test): a third column exists only if the work is (a) read-only narrow-row indexing into a table fitting 480 KiB, or (b) **independent of the matmul it would run beside**. `sparsecore_topk` (L831) consumes the scoring kernel's output — serial within the call. My file: *"Anything serial with the surrounding TensorCore work fails both and is a regression."* Today's placement is not defended by (a) or (b); it is defended only by measurement (`j-e83d16c4`, per report.md).
- **Requires, if overlap is ever attempted:** the neighbour must be compute-bound. Behind a *memory-bound* neighbour the same SC work added 346–375 µs instead of 61.2 µs — 80–87% of the step rather than 14% [verified: j-6449d3ea-b798-4f62-9be0 / j-97fa6617-e5e6-42ca-8dbf]. **Decode's TensorCore side is HBM-bound** (modelled `hbm_id` 41.7 µs against MXU 26.9 µs, below), so decode is exactly the unfavourable case; mixed is the favourable one if the einsum is bf16.
- **Structural, and not from my file:** the production call site (`mla_attention_op.py:107-119`) passes neither `vmem_limit_bytes` nor `decode_req_batch_size`, so both take the kernel defaults — `100 MiB` (`streamindex_topk.py:28`) and `4` (`:536`). **100 MiB exceeds `vmem_capacity_bytes = 67108864` (64 MiB)** [verified: j-4fba8dc6, `tpu_info_reported_constants`, and that field is per-core and must *not* be doubled]. What Mosaic does with an over-capacity limit is not in my file.

## 2. Tagged facts that bear on this block

| fact | value | tag |
| --- | --- | --- |
| **VPU f32 elementwise FMA** | **1.72 Tflop/s** — 512×1024, **2000 *dependent* FMAs**, median 1217.2 µs, spread 2% | [verified: j-ed517191-5f2d-4d00-8c0c, `vpu_fma_rate`, tpu7x] |
| VPU vs MXU on same dtype | VPU is **385×** slower than 661.4 Tops/s f32 | [verified: j-ed517191] |
| MXU, bf16 under real Pallas | **569.9 Tops/s**, 128-row arm, batched, spread 5% — my file: *"of those two rates only 569.9 is quotable"* | [verified: j-ed517191, `mxu_block_rows_cost`] |
| MXU, bf16→f32 XLA 8192³ | 764.8 Tops/s, unbatched, ~11% dispatch floor; a batched 4096³ arm gave 695.9 | [verified: j-ed517191 / j-054e7ac4-4099-4176-ad2b shard3] |
| MXU, fp8 e4m3→f32 XLA | 1419.9 Tops/s, spread 2%. **No Pallas fp8 rate exists.** | [verified: j-ed517191] |
| 128-row cliff, work held fixed | 32 rows = **3.57×**; 64 = 1.97×; 128 = 1.00×; 256 = 0.92×; 512 = 0.91× | [verified: j-ed517191, `mxu_block_rows_cost`] |
| K / N granularity at 512 rows | wall time near-independent of K and N across 16× of work; non-monotonic (N=256 = 1.23×) | [verified: j-ed517191, `mxu_contraction_and_column_granularity`] |
| relative op cost, per element | multiply 1.00, **divide 1.00**, exp 1.24, sigmoid 1.85 | [verified: j-ed517191, `transcendental_relative_cost`] |
| geometry | 128 lanes × 8 sublanes; 2 MXUs; `mxu_column_size` 256; VMEM 64 MiB; SMEM 1 MiB | [verified: j-4fba8dc6] |
| SC streaming | 337 GB/s, 9.1% of HBM peak, vs TensorCore 2916 GB/s / 78.8% — ~8.7× slower | [verified: j-6449d3ea] |
| SC overlap score | 0.93 when SC work is ⅓ the neighbour's length; **0.30** for a near-equal-length pair | [verified: j-97fa6617] |
| SC top-k, whole-algorithm | a radix-select top-k came out **1.5–9× slower** than `jax.lax.top_k` | [verified: j-48095a41-8de9-48ad-8837] |
| SC coordinated fan-out | `subcore_barrier` completes at 2 and 4 subcores and **deadlocks the device at 8** | [verified: j-97fa6617] |

**Is there a VPU throughput figure? There is a VPU *rate*, and it is not a throughput figure.** This directly contradicts report.md's *"`V7X.md` carries no VPU peak"* — `V7X.md` may not, but `05-compute-units.md` carries a measured 1.72 Tflop/s. **I state the disagreement rather than reconcile it, because reconciling it the obvious way is wrong: the probe is 2000 *dependent* FMAs, so it measures a serial-chain latency, not issue throughput, and §4 shows the block's own measurement falsifies it as a throughput.** The file also carries **no** cost for `max`, `add`, comparison, `select`, cross-sublane rotate, or any bf16 VPU op — the transcendental table lists none of them.

Two further disagreements, stated not reconciled: report.md's rates (892.8 TF/s bf16, 1785.6 fp8, from `V7X.md`) match neither `tpu_info`'s reported peaks (1155 / 2300 per TensorCore) nor the measured 764.8 / 1419.9. And my file's SparseCore bullets give 337 GB/s streaming while the fan-out bullet says 23.7× "saturating near 300 GB/s", and the occupancy section then writes `bytes_touched / 338e9` **"per subcore"** — the three cannot all be per-subcore. I use 338e9 as an aggregate and flag it.

## 3. Which of my file's `## Gaps` this block turns on

1. **"Of those two rates only 569.9 is quotable" / "the 764.8 denominator is not on the same footing."** *Settles:* whether decode's MXU term is 26.9 µs (Pallas rate) or 20.0 µs (XLA rate) — i.e. whether the MXU could ever contend with decode's 41.7 µs HBM floor, or is 1.5× below it under every rate. A single probe running an XLA arm and a Pallas arm at *this* einsum's shape (M=32/4096, K=128, N=2048, b=4/1) settles it and nothing else does.
2. **"The two MXU granularity readings above have not been separated"** — padding-up versus a fixed per-call cost. *Settles:* decode issues **64 einsum invocations** of M=32,K=128,N=2048. If the sub-128 penalty is padding, the 3.57× derate is the model and filling the tile is the fix; if it is a fixed per-call cost, the fix is fewer, larger invocations (raise `bkv_p`, unroll the bkv loop) and the derate is the wrong story. These point at different designs. The probe my file names — hold total work fixed while sweeping K and N — does not exist.
3. **Not in the `Gaps` list, and the largest hole this block turns on: there is no independent-op VPU throughput.** *Settles:* whether the post-einsum tail is 39 µs or 930 µs at mixed — a 24× range, on the term report.md names as the leading suspect for 29%. Probe: f32 elementwise `max`/`mul`/`add` at ILP ≥ 8 (not a dependent chain), plus a `relu → FMA` reduction over a **sublane** axis of extent 32, plus the same in bf16. Three arms, minutes.
4. **`subcore_barrier` deadlocks at 8 subcores; `fetch_and_add` has no hardware evidence.** *Settles:* only whether a fanned-out SC design is admissible — turned on if and only if alternative N or P below is pursued.

## 4. Occupancy arithmetic

Element and FLOP counts are derived from `compute_scores` L326-354 and report.md's shape table; they are facts. Times carry the rate they were divided by.

**MXU.** `2·ntok·32·128·kv_len`; derate from the measured cliff (3.57× at M=32, 1.00× at M=4096 — the table stops at 512 rows / 0.91×, so 4096 is extrapolated as 1.00 and is if anything pessimistic by ~8%).

| regime | einsum calls | M | FLOPs | derate | bf16 Pallas 569.9 | bf16 XLA 764.8 | fp8 XLA 1419.9 | fp8 Pallas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `decode_prod` | 64 | 32 | 4.295e9 | 3.57 | **26.9 µs** | 20.0 µs | 10.8 µs | `unmodelled` |
| `mixed_prod` | 32 | 4096 | 6.872e10 | 1.00 | **120.6 µs** | 89.8 µs | 48.4 µs | `unmodelled` |

Fraction of measured, at the quotable Pallas rate: decode **35%** of 76.42 µs; mixed **45%** of 269.5 µs. Cost of the decode derate: 26.9 − 7.54 = 19.4 µs discarded. **Note what that does to the floor: nothing.** `floor = max(hbm_id, mxu)` and decode's `hbm_id` is 41.7 µs, above the *underated* 26.9 µs, so eliminating the 3.57× derate entirely moves decode's modelled floor by zero. The derate is real waste and it is not on decode's critical path. At mixed the MXU term is the largest single modelled term if bf16 (120.6 vs `hbm_st` 46.2) and roughly ties HBM if fp8 (48.4) — which is report.md's probe #5, restated in this area's units.

**VPU.** Op slots per element of `s` (shape `b × bq_sz × 32 × bkv_sz`), read off L333-335:

| step | slots / element of `s` | note |
| --- | ---: | --- |
| `jnp.maximum(s, 0.0)` | 1 | no `max` cost in the file; `assumed` = 1 multiply |
| `s * w[:,:,:,None]` | 1 | plus a lane-broadcast of `w`, cost `unmodelled` |
| `.sum(axis=2)` over 32 heads | 0.97 … 2.25 | 31 adds / 32 elements if the head axis is on **lanes**; 9 vreg-ops per 4 input vregs (3 full adds + 3 rotate+add) if it is on **sublanes**, which is the layout the einsum's `(…, 32, bkv_sz)` output implies. `assumed`, area-(4) seam |
| on `s_summed` (1/32 the elements): dequant mul, 2 compares, `and`, `select` | 0.16 | |
| **total N** | **2.0 (floor, if mul+add fuse to one FMA) … 4.4** | |

| regime | elements of `s` | f32 bytes, one pass | N=2.0 | N=2.97 | N=4.4 | measured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 1.678e7 | 67.1 MB | 39.0 µs | 57.9 µs | 85.8 µs | 76.42 µs (whole scope) |
| `mixed_prod` | 2.684e8 | **1074 MB** | 624 µs | 927 µs | 1373 µs | 269.5 µs (scoring by diff) |

All six times are **modelled from `vpu_fma_rate`'s implied 8.6e11 issue slots/s (1.72 Tflop/s ÷ 2 flops per FMA), and at mixed the model is falsified by the block's own measurement**: even the most generous N=2.0 gives 624 µs against a measured 269.5 µs for *everything* in the scoring half. Turning that around, giving the VPU the entire 269.5 µs, mixed puts a hard lower bound of **≥1.99e12 slots/s** on independent-op VPU throughput (**≥2.96e12** at N=2.97) — 2.3× to 3.4× the dependent-chain probe. Decode does not falsify it (the same inversion gives 6.5e11, below 8.6e11).

**So the honest emission for the VPU tail is the element count and the op count per element above, and the time is `unmodelled`** — not because no rate exists, but because the one that exists is a latency figure that the block disproves as a throughput. Probe #3 in §3 is what would replace this row with a number.

**VMEM occupancy of the intermediates the VPU works on** (declared scratch, `run_scores_kernel` L654-692, against `vmem_capacity_bytes` 64 MiB):

| regime | `bkv_x2` | `bq_x2` | `bq_w_x2` | `scores_block_x2` | declared | **undeclared live `s`** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 4.00 MiB | 32 KiB | 1 KiB | 64 KiB | 4.09 MiB (6.4%) | 1.00 MiB |
| `mixed_prod` | 1.00 MiB | 1.00 MiB | 32 KiB | 2.00 MiB | 4.03 MiB (6.3%) | **32.0 MiB (50% of VMEM)** |

The einsum output at mixed is 32 MiB of f32 live at once, produced by a line the source comments *"Compute in Registers"* (L325). `num_accumulators` is 128. Whether Mosaic tiles this internally or spills is not in my file and not in report.md — it is a candidate mechanism for mixed's 29% that is neither the VPU rate nor the MXU, and it is testable by shrinking `bq_sz` at mixed alone.

**Scalar unit.** SMEM prefetches total <20 KB against 1 MiB (2% at 64k context, where `page_indices` is 64×65 i32 = 16.6 KB) — not a constraint, and area (7) has room. DMA issue count, counting `_fetch_bkv` (`seq_batch_size × bkv_p` per bkv iteration), `_fetch_bq` (2 per batch element) and `start_send_scores` (1 per bkv iteration):

| regime | bkv DMAs | bq DMAs | scores-out DMAs | total | measured µs | µs available per DMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` (grid 16) | 512 | 128 | 64 | **704** | 76.42 | 0.109 |
| `mixed_prod` (grid 1) | 64 | 16 | 32 | **112** | 269.5 | 2.41 |

Decode is 22× denser in DMA issues per unit time. My file carries no DMA-issue occupancy — `unmodelled`, route to area (3). One tension worth handing over: report.md assumes 512 ns per DMA, fully hidden; 704 × 512 ns = 360 µs against a measured 76.42 µs, so the 512 ns cannot be issue occupancy at decode.

**SparseCore third column, both conditions and the streaming floor.** (a) fails — the scores tensor is 2.6–277 MB streamed, not narrow rows into a 480 KiB resident table. (b) fails within a call — L831 consumes L811's output. Streaming floor at 338e9 B/s (aggregate; see the ambiguity flagged in §2), against report.md's `topk ctrl`:

| regime | scores MB | SC streaming floor | `topk ctrl` measured | ratio |
| --- | ---: | ---: | ---: | ---: |
| `decode_prod` | 2.6 | 7.7 µs `modelled` | 55.6 µs | 7.2× |
| `mixed_prod` | 41.9 | 124.0 µs `modelled` | 399.6 µs | 3.2× |
| `mixed_64k` | 276.8 | 818.9 µs `modelled` | 3009.7 µs | 3.7× |

report.md item 4 says no floor has been established for `sparsecore_topk`. This is one, and it is weak: 338 GB/s was measured on bulk streaming, and a top-k does more than stream. It says the selection half is 3–7× above a pure-streaming floor, not that 3–7× is recoverable.

**What it would take for the TensorCore to have independent work while SC runs.** Three requirements, all from my file: the SC work must be independent of the neighbouring matmul; the neighbour must be **compute-bound**; and the saving is set by the *ratio* of the two durations — 0.93 at 1:3, **0.30 near-equal**. The selection half is 33–60% of the call, i.e. near-equal, i.e. the 0.30 end. And my file's MoE-prologue result cuts the other way and must be carried: a kernel at *parity* standalone (1.02×) won by 29% measured as added step time beside a matmul, so **measure any offload against its neighbour, never on its own** — the three-arm pattern (`tc_only` / `tc_plus_baseline` / `tc_plus_sc`) is in `08-sparsecore.md`, which is the parallel briefer's, not mine.

## 5. Alternatives this area could take

- **A.** Incumbent: einsum on MXU with M = `bq_sz × 32`; relu, per-head scale and head-sum on VPU in f32.
- **B.** Fold the per-head weight into `q` before the einsum. `relu(a·x)·w = relu(a·x·w)` for `w ≥ 0`, so one full-size VPU multiply (1 of 2.0–4.4 slots on the 1074 MB array at mixed) moves onto an array 512× smaller. Preconditions: `indexer_weights ≥ 0`, and `q` is fp8 so the scale has to land somewhere that survives quantization.
- **C.** As B, but fold `w` into the fp8 block scale rather than into `q`'s values, keeping `q` in `float8_e4m3fn`.
- **D.** Split the einsum into a loop over the 32 heads: M becomes `bq_sz` (128 at mixed — exactly the cliff's 1.00× point; 1 at decode, off the bottom of a table that stops at 8 rows / 10.57×), the head reduction becomes a loop-carried FMA accumulator with no cross-sublane tree, and the 32 MiB live `s` collapses to 1 MiB.
- **E.** As D but in head *groups* of 4 or 8, trading MXU rows against accumulator count.
- **F.** Keep the head axis on **lanes** rather than sublanes so the reduction is a lane reduction — an area-(4) relayout of the einsum output, priced nowhere.
- **G.** Do relu / scale / head-sum in bf16 rather than f32, halving vreg count. Cost `unmodelled` (no bf16 VPU rate); numerics land on a top-k that already shows 61-rank boundary churn against float64.
- **H.** Move the head reduction to the MXU as a matmul against a ones-vector. Blocked as written: the relu sits between the two contractions, so the second matmul is M=1 per token — 128× MXU waste, and my file's causal-conv1d precedent is exactly this case arguing against.
- **I.** Emit `scores` in bf16 to halve the round trip that dominates mixed — but L820-821 casts back to f32 for `sparsecore_topk`, so this moves the convert rather than removing it. Area-(1)/(4) seam.
- **J.** Raise `bq_sz` at decode above 1 via speculative decoding (`static_q_len = γ+1`, the TODO at L626-627): γ+1 ≥ 4 puts M ≥ 128 and removes the 3.57× derate — which §4 shows moves decode's floor by zero.
- **K.** Concatenate the four decode batch elements' KV along N and their heads along M (M = 128), discarding the 3/4 off-diagonal — same waste, moved from the MXU tile to the N axis, where my file says width is nearly free.
- **L.** Raise `decode_req_batch_size` above 4: does not change M, amortizes grid and DMA-issue overhead, and VMEM has room (4.09 MiB of 64 used).
- **M.** Precompute `k_span` and `valid_mask` as area-(7) metadata in SMEM rather than rebuilding them per bkv iteration on the VPU.
- **N.** Restructure to a per-KV-block partial top-k so SC work for block *i* overlaps TC scoring of block *i+1* — the only route that makes condition (b) true within a call. Needs an incremental/mergeable top-k (area 6) and turns on the `subcore_barrier` gap.
- **O.** Do nothing about SC; accept the serialization and keep the selection half out of scope, per report.md's routing.
- **P.** Move the `jnp.full(..., -inf)` scores initialisation (L745) off the TensorCore. It is 2.6 MB at decode / 41.9 MB at mixed, it is **genuinely independent of the Pallas matmul** — the one part of this block that satisfies condition (b) today — and at mixed the TC has 269.5 µs for a 124 µs SC fill to hide behind. Alternatively eliminate it by making the kernel's writes cover the buffer.
- **Q.** Derive `num_sublanes_total` (L603-605) from the batch's KV length rather than `max_model_len`, shrinking both the fill in P and the SC top-k input. report.md probe #6 owns whether the SC cost follows padded width.
- **R.** Assign units differently per regime rather than parametrizing one kernel: decode is HBM-bound with a 3.57× MXU derate and 704 DMAs; mixed is MXU-bound-if-bf16 with a 32 MiB live intermediate and 112 DMAs. `_common_path` already dispatches separate `pallas_call`s per case (L757 / L775 / L793), so the split is nearly free structurally.
