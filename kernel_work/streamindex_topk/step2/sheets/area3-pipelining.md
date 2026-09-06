## Constraint sheet — area (3), pipelining memory against compute
### block: `streamindex_topk.py` scoring kernel, GLM-5.2 DSA lightning indexer
Reference: `references/03-pipelining.md`. Step 1: `kernel_work/streamindex_topk/report.md`. I choose nothing below.

---

## 0. What the incumbent already is (read, not proposed)

All five `in_specs`/`out_specs` are `pltpu.HBM` (`streamindex_topk.py:645-652`). By `03-pipelining.md § "When to drop to raw DMA"` this is **route 3 of 3** — the same spelling as tokamax's MLA `[source: .../mla/pallas_mosaic_tpu_kernel.py:2369-2379@9da2369]`. The `pallas_call` grid is `(nseq/seq_batch_size,)` and does *no* transfer work; every byte moves through `pltpu.make_async_copy`. The two inner loops (`num_bq`, `num_bkv`) are `lax.fori_loop`s with **traced** trip counts (`:276-281`), which a `pallas_call` grid cannot express and `emit_pipeline` can [capability 3, source: jax 0.10.2, `.../pipeline.py:1835-1848`].

Depth is 2 on every operand (`bkv_x2_ref`, `bq_x2_ref`, `bq_weights_x2_ref`, `scores_block_x2_ref`), `sems` = `(4, 2, seq_batch_size)` — channels {0 bkv, 1 bq, 2 scores-out, 3 bq-weights} × 2 slots × batch member. Waits use the reference's documented dummy self-copy (`:196-198`, `:225-228`) with the size matched to the *sum* of the `bkv_p` starts on one semaphore — the idiom is correct as written [source: `.../mla/pallas_mosaic_tpu_kernel.py:756-772@9da2369`, "the dma size must be correct"]. The prologue (`:424-427`) fires **once for the whole grid**, not per grid step; `sem_ids_ref` in SMEM carries the pipeline state across `pallas_call` grid steps.

---

## 1. What this area forbids or requires at step 1's shapes

**Per-inner-step transfer inventory, derived from the code (not from report.md):**

| | decode_prod (`seq_batch_size=4, bq_sz=1, bkv_p=2`) | mixed_prod (`seq_batch_size=1, bq_sz=128, bkv_p=2`) |
|---|---|---|
| bkv in | 2 MiB, **8 DMAs** × 256 KiB, contiguous | 512 KiB, **2 DMAs** × 256 KiB, contiguous |
| bq + weights in (amortized over `num_bkv`=4) | 4.1 KiB/step (issued as 8 DMAs of 4 KiB / 128 B, once per 4 steps) | 132 KiB/step (2 DMAs of 512 KiB / 16 KiB, once per 4 steps) |
| scores out | 32 KiB, 1 DMA, **strided**: 4 runs of 8 KiB | 1 MiB, 1 DMA, **strided**: 128 runs of 8 KiB |
| bytes/step | **2.036 MiB** | **1.63 MiB** |
| inner steps/call | 64 | 32 (see §5, disagreement D) |

- **Requires: the 256 KiB per-DMA size be treated as the transfer granularity, not the 2 MiB step.** `_fetch_bkv:177-194` issues `bkv_p` separate page DMAs because the page table is a gather; raising `bkv_p` raises bytes/step but leaves the descriptor at one page. On the block-size curve `[verified: j-ed517191-5f2d-4d00-8c0c, block_size_bandwidth_curve]` 256 KiB sits between the 128 KiB (15% of ceiling) and 512 KiB (43%) rows. Measured whole-kernel achieved is 134.2 MB / 76.42 µs = **1.756 TB/s = 59% of the 2.97 TB/s ceiling** — above the per-DMA-as-step reading (~30%) and below the 2 MiB-step reading (80%). **The 8 page DMAs coalesce partially, not fully.** `measured (profile scope StreamIdxTC-d-bq_1-bkvp_2, 76.42 µs, report.md)`, ratio `modelled`.
- **Forbids: pricing the 512 ns fixed term per DMA.** 512 bkv DMAs × 512 ns = 262 µs alone, against a measured 76.42 µs. `refuted by measurement`. Per *inner step* (64) it is 32.8 µs; per *grid step* (16) it is 8.2 µs. Which of these is the unit is not established.
- **Forbids: `2 MiB per operand block` being read as satisfied.** It is satisfied per step at decode and violated at mixed (512 KiB bkv), and violated at the descriptor level everywhere.
- **Requires the strided output be costed.** `start_send_scores:142-151` writes `[pl.ds(token_start, n), pl.ds(sublane_start, 16)]` into an `[ntok, 80, 128]` f32 array — a 16-of-80 sublane window, so the contiguous run is 8 KiB with a 40960 B row stride, 20% duty. Per the reference: *"A transfer's size is not the block's size when the block is strided… a block that is logically large but laid out with a stride between its rows can be issued as many small descriptors, which puts it on the top row of that table rather than the bottom one."* At mixed this is the **largest single per-step transfer (1 MiB of 1.63 MiB)** and it is the one that is strided. `unmodelled` — report.md's floor has no term for stride.
- **Requires an answer on decode's `num_bq = 1`.** With one query block the *inner* bq loop has a single iteration, but the bkv loop runs 4× and the prologue is amortized over all 16 grid steps, so steady state **is** reached; the non-steady part is one bkv fetch at call start. Decode's short-grid exposure is elsewhere: the `pallas_call` grid is 16 steps, exactly the `## Deciding` item-4 threshold.
- **Requires the decode remainder launch be treated as a separate regime.** At `decode_64k` (6 seqs) `decode_batch_end = 4`, so 32 inner steps run batched (2 MiB/step) and 64 run at `seq_batch_size=1` (512 KiB/step). The remainder half is where DMA most clearly dominates: 663 ns DMA-side against ~270 ns compute-side, `modelled`.
- **VMEM is not a constraint here and this is the loudest single fact in the sheet.** Total scratch: decode 4.10 MiB (bkv 4 MiB + bq 32 KiB + weights 1 KiB + scores 64 KiB); mixed 4.03 MiB (bkv 1 MiB + bq 1 MiB + weights 32 KiB + scores 2 MiB). Against the ~31 MiB budget `02-partitioning.md` quotes and the `vmem_limit_bytes = 100 MiB` actually set at `:535`. **~27 MiB unused.** Each extra bkv slot costs 2 MiB (decode) / 512 KiB (mixed).

---

## 2. Tagged facts from the reference that bear on this block

- `per_step = 512 ns + bytes / 3.48e12 B/s` — `[verified: j-5855c2a5-aef4-4a61-b161, grid_step_cost_model, tpu7x, f32, 268.4 MB r+w held constant, 2 operands]`. The task's brief quotes the same law; it **agrees with the reference verbatim**, same job id.
- **512 ns fixed / 3.48 TB/s marginal, carried separately**; 3.48e12 is 94% of the reported `mem_bw_bytes_per_second = 3.70e12` `[verified: j-4fba8dc6-d72c-41f9-948b, tpu_info_reported_constants]`.
- Whole-kernel achieved ceiling **2.97 TB/s**, and *"there is no read-only bonus"* — `[verified: j-ed517191-5f2d-4d00-8c0c, hbm_bandwidth_arms, tpu7x]`. Do not model the KV read as cheaper than the scores write.
- Block-size knee **512 KiB → 2 MiB**; 128 KiB = 15%, 512 KiB = 43%, 2 MiB = 80% of ceiling — `[verified: j-ed517191-5f2d-4d00-8c0c, block_size_bandwidth_curve]`.
- **Operand count is not a term**: per-operand coefficient between −10.9 and +2.6 ns, sign-changing — `[verified: j-054e7ac4-4099-4176-ad2b shard2]`. The block moves 4 operands per step; this is priced at zero, on evidence.
- **Below ~512 KiB/step a step is not bandwidth-bound at all**; a constant-`index_map` control costs a flat **340–570 ns at every byte level** — `[verified: j-054e7ac4-4099-4176-ad2b shard2]`. Bears directly on the block's 4 KiB / 128 B bq DMAs and its 32 KiB decode scores-out.
- Overlap is worth **1.62×** (serial 1.213 TB/s vs managed 1.959 TB/s) — `[verified: j-050c7480-40a6-4a50-9e1b, manual_dma_vs_managed, tpu7x, batch=32]`. This is the size of what the incumbent's hand-rolled double buffering is defending; a redesign that loses the overlap loses 1.62×.
- `emit_pipeline` vs managed grid = **0.99×** — `[verified: j-a5c66200-aab5-4319-be73, emit_pipeline_vs_blockspec_grid, tpu7x, batch=32]`. The reference flags quoting this as a general verdict as "the most common misreading of this sheet"; this block's access pattern is exactly outside the regime it was measured in.
- Deep input buffering is legal under `emit_pipeline` and illegal on a `pallas_call` grid; **outputs are capped at 2** — `[source: jax 0.10.2, .../pipeline.py:493-502]`, cap `[verified: j-106c8a46-174c-4c03-a76a, buffer_count_cap_by_position]`. The incumbent hand-rolls, so it is under neither cap.
- Lookahead is for *"an `index_map` [that] returns the same block for long runs"* — `[source: jax 0.10.2, jax/_src/pallas/core.py:217-221]`, implemented in `fetch_with_lookahead` `[source: .../pipeline.py:928-946]`. `bq_x2_ref` is exactly that: fetched once, reused across all `num_bkv` steps.
- `ep_wait_in` / `ep_run_kernel` per-phase scopes — `[source: jax 0.10.2, .../pipeline.py:1146-1152, 1212-1333]`, and **`[unverified]` as a measurement**.
- VPU asymptote **1.69 Tflop/s**, ridge at ~1 flop/byte — `[verified: j-ed517191-5f2d-4d00-8c0c, arithmetic_intensity_crossover]`, measured with **dependent** FMAs. See disagreement C.
- `dispatch_floor` 149.0 µs median, 23% spread, and a 126.8 µs affine intercept — `[verified: j-ed517191-5f2d-4d00-8c0c, pipeline_depth_and_warmup]`. See disagreement B.
- The reference's precision table covers f32/bf16 only. **There is no fp8 row and `_F32_DOT_PRECISION_MAP` `[source: tokamax/_src/precision.py:43-64@9da2369]` has no fp8 entry.** The reference cannot answer report.md's probe #5.

---

## 3. Gaps this block turns on, each with the decision it settles

| Gap (from `03-pipelining.md § Gaps`) | Decision it settles here |
|---|---|
| *"no probe has run a depth sweep through `emit_pipeline`… tokamax's `buffer_count=3` is a design decision this suite has not priced"* | Whether depth 3–4 on `bkv_x2_ref` pays. ~27 MiB of VMEM is free; each slot is 2 MiB (decode) / 512 KiB (mixed). Decides directly, because decode's DMA side (1126 ns) and compute side (383–1081 ns) are close enough that a deeper queue is the only thing that can absorb the residual. |
| *"The block-size curve was measured on a pure copy. A kernel with compute in it may reach the knee at a different size, since the transfer has more to hide behind."* | Whether the 256 KiB per-page descriptor is actually costing 43%-of-ceiling here, given ~1 µs/step of compute to hide behind. Decides whether `bkv_p` (2→4→8) is a lever at all, and whether coalescing contiguous page runs is worth the branch. |
| *"the arm to build is a `BufferedRef` subclass against a hand-rolled pipeline over the same ragged access pattern"* | **This block is that arm.** A page-table gather with a regular tile loop is the textbook capability-5 case (*"whenever the addresses are data-dependent but the iteration is regular"*). Decides whether the hand-rolled schedule is replaced, and it is the one probe that would close this gap for the knowledge base. |
| *"no probe has dumped an xprof of an `emit_pipeline` kernel"* / `ep_*` scopes `[unverified]` | Whether mixed's 29%-of-floor is DMA-starvation or compute. Right now separable only by arithmetic (§4 says compute, 7:1). `ep_wait_in` vs `ep_run_kernel` would settle it by measurement. |
| *"[the `emit_pipeline` comparison] says nothing about the traced-extent case that is `emit_pipeline`'s actual reason to exist"* | Whether `num_bq`/`num_bkv` — traced at `:276-281` — can become an `emit_pipeline` grid. Source says yes (capability 3); payoff is the gap. |

**Not turned on:** single-device-only bandwidth (the indexer is replicated at TP=1, so single-chip is the right denominator, per report.md); `HIGH` / `BF16_BF16_F32_X3` (operands are `float8_e4m3fn`, not f32).

---

## 4. `Emit …` schema, filled

### decode_prod — per inner (bq, bkv) step, 64 of them

- **bytes moved:** 2.036 MiB (bkv 2 MiB in / bq+weights 4.1 KiB in / scores 32 KiB out), across **9 DMAs** (8 contiguous 256 KiB + 1 strided 32 KiB), plus 2 bq DMAs every 4th step. `derived from source`
- **compute performed:** einsum `(4, 32, 128)×(4, 2048, 128)` = **67.1 MFLOP**, M=32 into a 128-row MXU tile → 4× derate. Then a VPU tail of ~3 passes over 262,144 f32 (`maximum`, `× weights`, 32-way head sum) ≈ **786 k element-ops**.
- **buffering depth / location:** 2 on all four operands, all VMEM scratch; 4.10 MiB of ~31 MiB.
- **prologue (non-overlapping):** one bkv fetch (2 MiB) + one bq fetch at grid step 0 only ≈ **1.13 µs**, 1.5% of the call. `modelled`
- **epilogue (non-overlapping):** drain of ≤2 outstanding 32 KiB out-DMAs ≈ **0.6 µs**. `modelled`
- **work too short for steady state:** none within a launch (64 inner steps). But `_common_path` always dispatches **three** `pallas_call`s (`:757`, `:775`, `:793`), two of which span zero sequences at this regime — each pays its own prologue/epilogue and grid entry with no steady state at all. `unmodelled`
- **predicted per-step cost:**

| side | fp8 MXU | bf16 MXU |
|---|---|---|
| DMA: 512 ns fixed + 614 ns bytes | **1126 ns** | 1126 ns |
| compute, VPU at the reference's 0.845e12 ops/s | 150 + 931 = **1081 ns** | 301 + 931 = 1232 ns |
| compute, VPU at ≥3.5e12 ops/s (implied by mixed_prod's own measurement, §5-C) | 150 + 225 = **375 ns** | 526 ns |
| **measured** (76.42 µs / 64) | **1194 ns** | 1194 ns |

- **which side dominates:** **not determined — it hinges entirely on the VPU rate, which is unmeasured for this block.** At the reference's dependent-FMA rate the two sides are within 10% and *neither* dominates (the worst case for a designer: cutting either alone yields at most half of its own size). At the rate mixed_prod's measurement implies, **DMA dominates 3:1** and the kernel is at 94% of the reference's stepped floor.

### mixed_prod — per inner (bq, bkv) step, 32 of them

- **bytes moved:** 1.63 MiB (bkv 512 KiB in via 2 DMAs / bq+weights 132 KiB amortized / **scores 1 MiB out, strided into 128 runs of 8 KiB**). The output is the largest transfer and the only strided one.
- **compute performed:** einsum `(1, 4096, 128)×(1, 2048, 128)` = **2.147 GFLOP**, M=4096, no derate. VPU tail ≈ 3 passes over 8,388,608 f32 = **25.2 M element-ops**.
- **buffering depth / location:** 2 on all four; 4.03 MiB VMEM, of which the *output* buffer is the largest at 2 MiB.
- **prologue:** one 512 KiB bkv fetch, ≈0.66 µs, once per call. **epilogue:** drain of 2 × 1 MiB out-DMAs ≈ 1.1 µs. Both < 1% of the 269.5 µs. `modelled`
- **predicted per-step cost:** DMA 512 + 491 = **1003 ns**. MXU **1202 ns** fp8 / **2405 ns** bf16. Measured **8422 ns** (269.5 µs / 32).
- **which side dominates: compute, by roughly 7:1.** DMA is ~12% of the step under any reading. **Deeper buffering, larger blocks and DMA coalescing cannot move mixed_prod**, and the ~6.2 µs/step that is neither DMA nor MXU is `unmodelled` — report.md names it the VPU tail and prices it at nothing.

### decode_64k remainder (`seq_batch_size=1`, 64 of the 96 inner steps)

DMA 512 + 151 = **663 ns**; compute ≈ 37 ns MXU (fp8) + ~232 ns VPU. **DMA-dominated ~3:1.** This is the one place in the block where the DMA side is unambiguously the bottleneck, and it exists only because `decode_batch_end = 6//4*4 = 4` strands 2 of 6 sequences into unbatched launches.

---

## 5. Disagreements between the reference file and `report.md` — not reconciled

**(A) The floor formula, and it is the largest.** report.md computes `hbm_id = bytes / 3364.9 GB/s` with **no step count**, giving 41.7 µs at decode and a verdict of 55% achieved / "the gate passes". `03-pipelining.md § Deciding` item 1 states the opposite in as many words: *"Compute `Σ_steps [512 ns + bytes_step / 3.48e12]`. That is the floor. `bytes_moved / 2.97e12` **is not** — a floor quoted without a step count understates a kernel whose steps are small."* Same bytes, four step counts:

| unit of the 512 ns term | steps | stepped floor | measured / floor |
|---|---:|---:|---:|
| report.md (no step term, 3364.9 GB/s) | — | 41.7 µs | 55% |
| `pallas_call` grid step | 16 | 46.8 µs | 61% |
| inner (bq, bkv) iteration | 64 | **71.3 µs** | **93%** |
| individual page DMA | 512 | 300.6 µs | refuted by the 76.42 µs measurement |

The exercise's continuation rests on "55%, inside the 50–80% band". Under the reference's own model at the inner-step reading it is **93%**, above the ≥80% bar at which the skill stops. Additionally, the rate itself differs: report.md uses 3364.9 GB/s from `V7X.md`, which matches neither the reference's measured 2.97 TB/s ceiling nor its 3.48 TB/s marginal.

**(B) Pipeline-start cost.** The reference measures `dispatch_floor` at 149.0 µs (23% spread) and recovers a 126.8 µs intercept independently. The block's *entire* `decode_prod` call — three `pallas_call` launches plus `sparsecore_topk` — measures 167 µs (`j-6d675a58-7ecd-4b44-9171`). One dispatch floor does not fit inside four launches. The likely reconciliation is that report.md's harness dispatches `inner` calls back-to-back before blocking, amortizing the host round trip the reference's probe deliberately left unbatched — but that is `assumed`, and if it is right, the reference's item-4 rule ("under ~16 grid steps the ~127 µs pipeline-start cost dominates everything") does not apply to this block as written.

**(C) The VPU rate cannot be both.** Applying the reference's 1.69 Tflop/s asymptote to mixed_prod's tail gives **29.8 µs per step** against a measured whole step of **8.42 µs** — and even a *single* pass over the 8.39 M f32 elements would cost 9.9 µs, more than the entire measured step. Either the reference's rate (measured on *dependent* FMAs, so a lower bound on independent-op throughput) understates by ≥4×, or the op count is wrong. This is the single largest lever on the decode answer: at 0.845e12 ops/s decode is balanced; at ≥3.5e12 it is DMA-bound at 94% of floor.

**(D) The mixed step count is ambiguous in report.md.** Its regime table lists `nseq = 64, nbq = 8` for `mixed_prod`, but its roofline row prices 2.1 MB of KV re-read 8× = 16.8 MB, which is **one** sequence of `q_len = 1024`. The code's step count is `nseq × num_bq × num_bkv`; the 64-sequence reading gives 256 inner steps and 134.2 MB. Arithmetic settles it in report.md's favour: 256 steps × 2.147 GFLOP = 550 GFLOP = 308 µs at *fp8 peak*, above the measured 269.5 µs — infeasible. The 32-step reading gives 68.7 GFLOP → 38.5 µs fp8 / 77.0 µs bf16, exactly report.md's `mxu` columns. `derived from arithmetic`. Flagged because every per-step figure in §4 depends on it and the table and the roofline row do not say the same thing.

---

## 6. Alternatives this area could take (enumerated, not ranked)

1. **Keep the incumbent** — hand-rolled depth-2 on all four operands, unchanged. The control arm.
2. **Deepen `bkv_x2_ref` to 3 or 4** (hand-rolled). +2 MiB/slot decode, +512 KiB/slot mixed, against ~27 MiB free.
3. **Deepen only `scores_block_x2_ref`.** Available *only* on the hand-rolled route — `emit_pipeline` refuses output depth > 2 `[source: jax 0.10.2, .../pipeline.py:493-502]`. Relevant because the out-DMA is the largest per-step transfer at mixed.
4. **Deepen only `bq_x2_ref`** — the tokamax pattern of triple-buffering exactly the operand reused across steps `[source: gmm_v2.py:391-402@9da2369]`; `bq` is reused across all `num_bkv` steps here.
5. **Replace the schedule with `pl.kernel` + `emit_pipeline` over a traced `grid=(num_bq, num_bkv)` plus a `BufferedRef` subclass** for the page gather (capabilities 3 + 5). Keeps slot rotation, semaphore lifecycle, prologue and drain; gives up the hand-rolled cross-grid-step carry at `:424-427`.
6. **`emit_pipeline` + `use_lookahead=True` on the bq spec** — the reference's stated case, an `index_map` returning the same block for long runs. Unmeasured anywhere.
7. **Coalesce contiguous page runs** into one descriptor at run time, lifting the bkv DMA from 256 KiB toward `bkv_p × 256` KiB when the shuffled page table happens to be sequential.
8. **Raise `bkv_p` 2 → 4 → 8.** Raises bytes/step (2 → 8 MiB decode) and cuts the step count 4×; leaves the descriptor at one page.
9. **Lower `bkv_p` to 1.** Enumerated so the sweep is two-sided and the knee is bracketed rather than assumed.
10. **Raise `decode_req_batch_size` above 4** — more sequences per grid step, more parallel DMAs per step, larger step, and it shrinks the unbatched-remainder regime that §4 shows is DMA-bound at `decode_64k`.
11. **Swap the loop order** — bkv outer, bq inner. Changes which operand is re-fetched `num_bq` times (report.md: the KV is re-read `num_bq` times today).
12. **Write scores out in bf16** instead of f32 — halves the dominant per-step transfer at mixed. Cross-area: the reference prices a narrow-HBM/wide-VMEM conversion at 0.74× time for 0.50× bytes.
13. **Accumulate scores in VMEM across all `num_bkv` steps and write once per bq block** — cuts the out-DMA count 4×, makes each write 4 MiB and contiguous in the sublane axis, and costs VMEM that is currently free.
14. **Un-stride the output** by giving the kernel a `[ntok, num_sublanes_bkv, 128]`-per-block layout and reassembling outside, trading a relayout for a contiguous descriptor.
15. **Split the VPU tail into its own pipeline stage** so it overlaps the *next* step's einsum rather than sitting between the DMA wait and the out-DMA.
16. **Move ReLU + scale + head-sum onto the MXU** (an area-6 move) — changes what area (3) has to hide behind, and at mixed the VPU tail is the 7:1 dominant term.
17. **Run one arm with `no_pipelining=True`** as a control, to measure what the overlap is currently worth against the reference's 1.62× `[verified: j-050c7480-40a6-4a50-9e1b]`.
18. **Collapse the `sems` array** from 4 channels × 2 slots × `seq_batch_size` (32 semaphores at decode) to fewer, cutting the wait count per step.
19. **Move the `jnp.full(..., -inf)` fill at `:745` into the kernel** — it is 41.9 MB at `mixed_prod` written by XLA and serialized in front of the pipeline, with no overlap at all.
20. **Fuse the three `pallas_call` launches into one** so the second and third do not each pay a prologue and epilogue over zero sequences.
21. **Narrow the KV read to the used 129 of 256 bytes with a strided DMA.** Enumerated with its cost attached: the reference warns a strided block *"can be issued as many small descriptors"*, which would move a 129-byte-per-row transfer onto the top row of the block-size table — a 12× risk against a 1.98× byte saving, and it also crosses into the cache writer.
