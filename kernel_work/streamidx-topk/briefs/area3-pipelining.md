# Area (3) — Pipelining memory movement against compute — `streamindex_topk`

Step-2 briefer sheet. Reference file: `.claude/skills/write-tpu-kernels/references/03-pipelining.md`.
Baseline of record: the **nightly** config (`max_model_len` 9216, page 1024, TP1×DP16, score width
10240). AgentX is a second operating point and is **not** the target.

Tags: `measured (job id)` / `modelled` / `assumed` / `from 03-pipelining.md`.

## A. Constraints

**A1. Use 2.97e12 B/s, not 3.70e12.** `from 03-pipelining.md` ("the reported constant is a spec sheet, the measured number is the budget"). `report.md` §4 states both and then models the floor at the spec rate. The decode floor is therefore **59.2 µs, not 47.5 µs** (175.9 MB / 2.97e12), and the raw achieved fraction is 26.8%, not 21.5%. Verdict unchanged (still structural); every downstream ratio moves by 1.25×.

**A2. The DMA granule is 256 KiB and cannot be enlarged.** `measured (source, streamindex_topk.py:187-194)`: `_fetch_bkv` emits one `_async_copy` per (sequence, page) of `page_size_per_kv_packing × kv_packing × lkv_dim` = 256 × 4 × 256 = **262 144 B**. Pages are page-table-indirect and non-contiguous in HBM, so no restructuring merges them. Raising `bkv_p` adds *more* 256 KiB DMAs, it does not make them bigger.

**A3. At 256 KiB, budget 0.8–1.5 TB/s, not 2.97.** `from 03-pipelining.md` block-size curve: 512 KiB → 1.263 TB/s (43%), 128 KiB → 0.455 TB/s (15%). This is the single largest correction to `report.md`'s floor and it is the constraint the whole design turns on. **Consistency check (independent of the curve):** stage A's steady state is 80 inner iterations moving 2 129 920 B each; the derived stage A of 109–151 µs implies **1.13–1.57 TB/s achieved, i.e. 38–53% of ceiling** — exactly where the curve puts a 256 KiB transfer. Two unrelated routes agree. Treat this as the working hypothesis, not a fact: the curve was measured on a serial pure copy, while this kernel keeps 8 DMAs outstanding. **See the falsifiability subsection appended below — the "two independent routes" claim is weaker than it reads here, and is corrected there.**

**A4. Per inner step, the bytes side dominates — but only by ~1.5×.** `modelled` at nightly decode (`bkv_p=2`, `seq_batch_size=4`, `bq_sz=1`):

| | value |
|---|---|
| HBM in | 4 seq × 2 pages × 256 KiB = 2 097 152 B (129/256 useful) |
| HBM out | 4 × 2048 × 4 B = 32 768 B |
| `t_hbm` | 0.72 µs @ 2.97 TB/s; **1.4–2.1 µs @ A3's rate** |
| MXU | 2·4·32·2048·128 = 6.71e7 FLOP → 0.03 µs (reported peak) / 0.12 µs (569.9 Tops/s) |
| VPU | 2.62e5 elts × 3 ops → 0.46 µs @ 1.72e12 (`assumed` rate) |
| **verdict** | **HBM-side; compute/DMA ≈ 0.68 at ceiling, ≈ 0.25 at A3's rate** |

There is *not* enough compute to hide the transfer. Any candidate that adds compute per byte is free; any that adds bytes is not.

**A5. Steady state IS reached — the brief's worry does not survive the source.** `measured (source, :424, :292-302, :391)`. The prologue at `:424` is guarded `batch_start_seq_idx == start_seq_idx` — it fires **once for the whole `pallas_call`, not per grid step**. `get_next_bkv_ids` rolls `next_seq_idx` forward past the grid-step boundary and `prefetch_next_bkv` at `:391` issues across it. So the pipeline is one continuous chain of **16 grid steps × 1 bq × 5 bkv = 80 inner iterations**, primed once. The "4-5 iterations" in the brief is the inner trip count per grid step, not the pipeline length. Non-overlapping prologue ≈ **1 step ≈ 0.7–2.1 µs ≈ 1-2% of stage A**. Epilogue (`:431`, two 32 KiB output waits) ≈ 0.02 µs, negligible.

**A6. The real non-steady-state cost is two empty `pallas_call`s, ~5× the prologue.** `measured (source + j-a71a648c)`. With 64 decode seqs and `decode_req_batch_size=4`, `decode_batch_end = 64`, so the second (`:775`) and third (`:793`) `run_scores_kernel` get grid `(0,)`. The profile's small `StreamIdxTC-d` mode (5.40 µs) and `StreamIdxTC-m` (5.22 µs) are those empty kernels — confirming `report.md` §8's conjecture. **≈10.6 µs, ~5% of the 221.4 µs call, 7-10% of stage A.** Do **not** use `03-pipelining.md`'s 126.8 µs / 149.0 µs dispatch floor here; that is an unbatched host figure and `report.md` §6.2 retracts exactly that add-back.

**A7. The ragged tail is 15% and is larger than the prologue and the empty kernels combined.** `modelled` from `:276`: `num_bkv = cdiv(kv_len_max, 2048)` = 5 at `kv_len_max` 8704 → 10 240 positions fetched for 8 704 wanted. **1 536/10 240 = 15% of the KV stream is DMA'd, matmul'd and masked to `-inf`.** Compounded by the group-of-4 waste (`report.md` finding 2, `:275-276`, `:168-194`): every slot pays the group max, unguarded.

**A8. VMEM is not a constraint at decode — 8× headroom.** `measured (source)`. Scratch: `bkv_x2` 4.00 MiB + `bq_x2` 32 KiB + `bq_weights_x2` 32 KiB (lane-padded from 1 KiB) + `scores_block_x2` 64 KiB = **4.13 MiB**, plus ~1-3 MiB of `compute_scores` temporaries. `mla_attention_op.py:119-120` does not pass `vmem_limit_bytes`, so `DEFAULT_VMEM_LIMIT_BYTES = 100 MiB` (`streamindex_topk.py:28`) applies — **the incumbent already opts out of the 32 MiB scope**; the real ceiling is 64 MiB physical. Buffer depth 3 or 4 on `bkv` costs +2 or +4 MiB and is affordable outright.

**A9. Prefill is the opposite regime and is VMEM-tight, not DMA-tight.** `modelled`. Per inner step (`bq_sz=128`, `seq_batch_size=1`): 1.5 MiB moved (0.53 µs) against 2.15e9 MXU FLOP (0.93-3.77 µs) and 2.52e7 VPU ops (14.6 µs at the `assumed` rate) — **compute/bytes ≈ 30:1**. 40 inner steps × ~15.5 µs ≈ 620 µs, reproducing §4's 623 µs. Depth 2 is already more than sufficient; **pipelining is a non-issue at prefill and the design must not spend budget there.** But the einsum output `s` is `[1, 4096, 2048]` f32 = **32.0 MiB live in VMEM**, plus 4.13 MiB scratch = 36 MiB of 64 MiB. Whether Mosaic tiles the `einsum → relu → ×w → sum_h` chain so `s` never fully materializes is unknown and decides both the prefill VPU floor and whether the tile even fits. This is the tightest VMEM point in the block.

**A10. The output DMA is 8 KiB per contiguous run and is the one clearly fixable schedule defect.** `measured (source, :137-151)`. `start_send_scores` writes `scores_hbm_ref[token_start:+4, bkv_idx*16 : +16]` — 4 discontiguous runs of 16×128×4 = **8 192 B**, 20 per grid step, 320 per decode call, 2.62 MB total. 8 KiB is below the bottom of the block-size curve (32 KiB → 4%). Fix is cheap and local: buffer a whole grid step's scores in VMEM (4 tokens × 80 sublanes × 128 × 4 = **160 KiB**, 320 KiB double-buffered — trivial against A8) and issue **4 contiguous 40 KiB DMAs per grid step instead of 20 strided 8 KiB ones**.

**A11. Deeper `bkv` buffering is reachable here and the usual cap does not apply.** `from 03-pipelining.md` + `measured (source)`. The file's `buffer_count ≤ 2` cap is on the *managed* `BlockSpec` grid; this kernel is hand-rolled raw DMA (`in_specs` all `pltpu.HBM`, `:645-652`), which the file's "When to drop to raw DMA" rule endorses — the page-table gather has no `index_map`, so `emit_pipeline` cannot express it either. Depth is a free parameter. Depth >2 does **not** help bandwidth-boundedness per se (A4), but it raises outstanding-DMA concurrency from 8 to 16-24, which is the only mechanism by which 256 KiB transfers recover ceiling bandwidth. See Gap G1.

**A12. Stage A and stage B are strictly serialized, and this is the largest lever in my area.** `measured (j-a71a648c)`: per indexer layer, 125.25 µs TC compute, **174.6 µs of TC stalled on SparseCore**, 203.6 µs SC. At nightly stage B is 67.4 µs = 30% of the call. Rows of the score matrix are independent, so splitting the 64 decode sequences into C chunks yields C independent (stage A → stage B) chains that can in principle occupy TensorCore and SparseCore concurrently. Modelled with A=148 µs, B=67 µs: C=2 → 181 µs (−16%); C=4 → 165 µs (−23%); asymptote max(A,B) = 148 µs (−31%). **Larger than anything obtainable inside stage A's DMA schedule.** Two costs to charge: C× stage B's fixed per-call cost, and C× the empty-kernel dispatches of A6 unless restructured. Whether XLA actually overlaps a TC `tpu_custom_call` with an SC `pl.kernel` is **unverified** and is the probe.

**A13. Stage B's fully-serial DMA is correctly ignorable — do not spend a lever on it.** `measured (source, sparsecore_topk.py:99-102, :160, :295)`. `_topk_kernel`'s `dma()` is `cp.start(); cp.wait()` — literally `03-pipelining.md`'s "hand-rolled serial DMA, no overlap" control arm, which lost 1.62×. But stage B moves only 2.62 MB in + 0.52 MB out against 67 µs, i.e. **~0.05 TB/s demanded — DMA is ≲2-5% of stage B**, which is compute-bound on ~6 SparseCore passes over the resident slice. Negative result, stated so the designer does not chase it. **A13's bound is revised upward — but not overturned — in the convergence section appended below. Read that before acting on this constraint.**

**A14. `report.md` finding (1) — "`max_model_len` drives stage B, 5.9×" — is an AgentX-branch effect and does not transfer to the nightly baseline.** `measured (source, sparsecore_topk.py:360-377, :409-457)`. `_pick_partition(b=64, n=10240, k=2048)` returns **p=1** at nightly (`p_min=1`; `b≥32` forces `max_p=1`), taking the single-stage `_sc_topk_direct` fast path — no second top-k, no `take_along_axis`, no `gather_offload`. At AgentX (`b=16, n=262144`) `p_min = cdiv(262144, 65536) = 4` → **p=4**, two-stage plus the XLA gather. The 5.9× is a **threshold crossing at n > 65536 or b < 32**, not smooth scaling. **The nightly config is already on the cheap branch, and this lever is not available at the baseline of record.** Cross-checked against `j-63b9f2e4` in the appended section; the reading holds and is sharpened there.

**A15. Writing scores in bf16 buys ~2% and is not worth the tie risk.** `modelled`. Score round trip is 4.5% of decode HBM traffic (`report.md` §4) and prefill is compute-bound, so halving it buys ~2% at decode and ~0 at prefill. `sparsecore_topk` requires f32 (`:395-396`) and immediately applies a monotone bit transform, so the change is mechanically possible — but the payoff does not justify an area-6 accuracy gate.

---

## B. `Emit …` schema — the fields the designer must fill for this block

*Per candidate, filled per phase (decode / prefill) and per `bkv_p` considered:*

1. `bytes_in_per_step`, `bytes_out_per_step`, `useful_fraction_of_bytes_in` — B, B, %.
2. `dma_count_per_step`, `bytes_per_individual_dma`, `max_outstanding_dmas` — the DMA *granule*, not the aggregate. A design that reports only "2 MiB per step" has not decided this area; A2/A3 say the granule is what prices it.
3. `t_hbm_per_step` quoted **twice**: at 2.97e12 (ceiling) and at the block-size-corrected rate implied by field 2. Both numbers, always. Name the source of the corrected rate.
4. `t_mxu_per_step`, `t_vpu_per_step`, and `t_compute = t_mxu + t_vpu` (the report's non-overlapping convention).
5. **`dominant_side`** = `bytes` or `compute`, with the ratio `t_compute / t_hbm_corrected`. This is the `Emit …` line's literal requirement.
6. `buffer_depth` per operand (`bkv`, `bq`, `bq_weights`, `scores_out`), `bytes_per_buffer`, `total_vmem_resident` including live einsum temporaries, against **64 MiB physical** (A8 — not 32 MiB; the incumbent already raises the limit).
7. `n_inner_steps_total`, and whether the pipeline is primed **once per `pallas_call`** or **once per grid step**. If per grid step, the design has regressed against the incumbent (A5) and must say why.

**Non-steady-state, itemised separately — four line items, not one:**

8. `t_prologue` = un-hidden first-block DMA, and its fraction of the steady state. Predicted ~1-2% here; a candidate claiming a large prologue cost at decode is mis-modelling A5.
9. `t_empty_pallas_calls` = (number of zero-grid `run_scores_kernel` invocations) × 5.2-5.4 µs. Predicted 10.6 µs / 7-10% of stage A. **A candidate that collapses the three scoring calls into one must claim this back explicitly; a candidate that adds calls must charge it.**
10. `tail_waste_fraction` = `(num_bkv·bkv_sz − kv_len_max) / (num_bkv·bkv_sz)` — 15% at the baseline (A7) — plus the group-of-4 over-fetch, quantified separately, with the batch length distribution it assumes stated as `assumed` if no artifact supplies it.
11. `t_epilogue` = final output-DMA drain. Predicted negligible; state it and move on.

**Cross-stage, which the `Emit` line's "any work short enough that steady state is never reached" clause reaches (A12):**

12. `tc_sc_overlap`: chunk count C, per-chunk stage A and stage B times, the predicted schedule `A/C + (C−1)·max(A/C, B/C) + B/C`, the **fixed per-call cost of `sparsecore_topk` charged C times**, and whether XLA overlap is `measured` or `assumed`. `C=1` is a legitimate answer and must be written as `N/A because —`, not omitted.

*Baseline row to beat, filled in for the incumbent at nightly decode (`modelled` except where noted):* 80 inner steps × 2 129 920 B; `t_hbm` 0.72 µs @ ceiling / 1.4-2.1 µs corrected; `t_compute` 0.49-0.58 µs; dominant side **bytes**; depth 2 everywhere; VMEM 4.13 MiB + temporaries of 64 MiB; primed once; prologue ~1-2%; empty calls 10.6 µs; tail 15%; epilogue ~0; no TC/SC overlap. Steady state alone predicts 57 µs @ ceiling and 109-170 µs at A3's rate, against a derived stage A of 109-151 µs `measured (j-8b1e7acc, best-of-N, soft)`.

---

## C. Gaps from `03-pipelining.md` this block turns on

**G1 — "no probe has run a depth sweep through `emit_pipeline`, so tokamax's `buffer_count=3` is a design decision this suite has not priced."**
Decision it flips: whether `bkv_x2_ref` becomes `bkv_x3/x4_ref`. **If deeper buffering raises achieved bandwidth on 256 KiB DMAs** (by taking outstanding DMAs from 8 to 16-24), it is a +2 to +4 MiB VMEM purchase — free per A8 — for up to 1.5× on the term that is 95% of decode traffic, and the design stays hand-rolled with a deeper ring. **If it does not** (i.e. 8 outstanding already saturates), depth stays 2 and the design must instead attack *bytes* (the 127/256 padding) or *stage B overlap* (A12). Note the file's gap is framed around `emit_pipeline`; here the cap is irrelevant (A11) and only the **payoff** half of the gap is live.

**G2 — "The block-size curve was measured on a pure copy. A kernel with compute in it may reach the knee at a different size, since the transfer has more to hide behind."**
This is the #1 probe for the whole block, and the two answers give opposite kernels. **If 8 concurrent 256 KiB DMAs with 0.5 µs of compute per step reach near 2.97 TB/s**, then stage A's floor is ~57 µs against a measured 109-151 µs, there is 2-2.6× of *schedule* headroom, and the design targets the DMA schedule (depth, concurrency, tail, the 8 KiB output DMAs of A10). **If they run at 1.1-1.6 TB/s as both A3 routes independently suggest**, stage A is already at **70-100% of its achievable floor**, the schedule is not the target at all, and the only remaining levers are bytes (packing 129/256, a two-consumer change) and TC/SC overlap (A12). The cheapest discriminator is `/xprof-llo` on a `run-command-in-cdk` capture, reading DMA-stall cycles inside `StreamIdxTC-d`.

**G3 — "`hbm_bandwidth_arms` measures one device. Nothing here touches collectives or multi-device bandwidth."**
Nightly is DP=16 with two engines per chip sharing one HBM controller, and `report.md` §3 records this as the one thing the single-chip microbenchmark cannot measure. Decision it flips: whether stage A is budgeted against 2.97 TB/s per TensorCore or roughly half that under co-resident contention. A design justified by squeezing the last 20% of *bandwidth* is worthless if the served denominator is 1.5 TB/s; a design that removes *bytes* wins under either. Absent a probe, **prefer byte-removal over schedule-tuning at equal modelled payoff.**

*Not turned on:* `use_lookahead` (the `index_map` is a page-table gather, not a run of repeated blocks — inapplicable), the `ep_*` scope-readability gap (`emit_pipeline` cannot express this access pattern at all; get the same number from `/xprof-llo`), and the `HIGH`/`BF16_BF16_F32_X3` gap (operands are fp8, not f32).

---

## D. Where the reference file and `report.md` contradict each other

**D1 — the HBM denominator. I believe `03-pipelining.md`.** The file is unambiguous ("Use 2.97 TB/s as the denominator for **every** roofline estimate"), and it is backed by three measured arms on tpu7x (`j-ed517191`). `report.md` §4 lists 2.97e12 as measured in its own rates table and then says "Figures below are modelled at the spec HBM rate," using 3.70e12. The report's number is internally inconsistent with its own table; the file's is measured. Decode floor 47.5 → **59.2 µs**; achieved 21.5% → 26.8%. Verdict unaffected, all ratios shift 1.25×.

**D2 — whether the floor may use a flat bandwidth at all. I believe `03-pipelining.md`, with its own caveat attached.** `report.md`'s floor charges every byte at one rate. The file's block-size curve says a 256 KiB transfer gets 25-43% of ceiling, and this kernel's granule is structurally pinned at 256 KiB (A2). This is the most consequential disagreement in the sheet: it can move stage A's floor by 2-4× and turn "21.5% achieved, huge headroom" into "stage A is near its achievable floor." I believe the file because it is the only measured statement about transfer size on this chip **and** because an entirely independent route — the measured stage A divided by 80 steps — lands in the same 38-53% band. I do not believe it *unconditionally*, because the file's own Gaps disclaim the compute case and the curve has no concurrency axis. That is precisely G2. **The "entirely independent route" phrasing is corrected in the appended falsifiability subsection; it is not independent.**

**D3 — "if your problem only has four blocks in it, it is too small for a kernel" vs `report.md`'s "~4-5 grid iterations per sequence." Both are right about different objects; the brief's framing follows the report and is wrong.** The pipelined unit is the inner `bkv` loop, which runs **80 continuous iterations** primed once (A5, `:424` + `:292-302`). 4-5 is the per-grid-step trip count. Steady state is comfortably reached, and the brief's "live question" resolves in the incumbent's favour.

**D4 — the magnitude of the fixed per-invocation cost. I believe `report.md`.** The file gives 126.8 µs (affine intercept) and 149.0 µs (`dispatch_floor`, 23% spread) from a deliberately unbatched host-timed probe. `report.md` §6.2 retracts precisely this style of add-back, showing it implies negative device time in 15 of 20 rows and varies 30% across concurrent shards, and supplies the in-model figure instead: an empty `StreamIdxTC` is **5.2-5.4 µs** and the `-inf` init is **6.02 µs at 2.79 TB/s**. Both are read off compiled HLO self-time. The file's number is a host round trip that the in-model case does not pay per call. **Use 5.2-5.4 µs; never 126.8 µs.** (The file's *structural* claim — that short grids are pure overhead — still stands and is what A6 charges.)

**D5 — hand-rolled serial DMA. I believe both, and the file's warning is correctly ignorable here.** The file prices no-overlap serial DMA at 1.62× worse than the managed pipeline. `sparsecore_topk._topk_kernel` is exactly that pattern (`cp.start(); cp.wait()`, `:99-102`, `:160`, `:295`) and `report.md` never notices. My arithmetic says it does not matter: stage B demands ~0.05 TB/s, so its DMA is ≲2-5% of a compute-bound kernel. Recording it so the next reader does not "discover" it as a lever, and so the file's rule is not over-applied.

---
---

# Appended after §6.3 landed

The three sections below were written after the sheet above, in response to `j-63b9f2e4-26c9-4e3a-99ce`
(`report.md` §6.3) and to convergent findings from the area 2, 6 and 7 briefers. Where they revise a
constraint above, the constraint carries a pointer. **A3, A13 and A14 are all touched. Read these
before designing against any of the three.**

## Cross-check: A14 against the deconfounded run (§6.3)

A14 predicts a **threshold crossing** in `_pick_partition`, not smooth scaling. §6.3 measured the
`max_model_len` effect in isolation at fixed page size — the experiment A14 was written without —
and the prediction can now be checked shape by shape. Working `_pick_partition(b, n, k=2048)`
through for every §6.3 pair, with `b = q.shape[0]` (the score matrix's **row count**, i.e. padded
token count, not `num_seqs`) and `n = num_sublanes_total × 128`:

| shape | b | n narrow → wide | p narrow → wide | stage B ratio `measured (j-63b9f2e4)` |
|---|---:|---|---|---:|
| `decode_b8_ctx4096` | 8 | 4096 → 65536 | **2 → 4** | 1.77× |
| `decode_b8_ctx16384` | 8 | 16384 → 65536 | **4 → 4** | 1.78× |
| `decode_b32_ctx4096` | 32 | 4096 → 65536 | **1 → 1** | 1.17× |
| `decode_b32_ctx16384` | 32 | 16384 → 65536 | **1 → 1** | 1.07× |
| `decode_b64_ctx4096` | 64 | 4096 → 65536 | **1 → 1** | 1.16× |
| `prefill_q1024_ctx4096` | 1024 | 4096 → 65536 | **1 → 1** | 2.88× |
| `prefill_q4096_ctx4096` | 4096 | 4096 → 65536 | **1 → 1** | 3.41× |

**The b≥32 decode rows confirm A14 cleanly and from the opposite direction.** `p_min = cdiv(65536, 65536) = 1`
— 65536 is exactly `MAX_SLICE_WORDS` — and `b ≥ 32` forces `max_p = 1`, so a 16× width change cannot
move `p` off 1, and the measured ratio collapses to **1.07–1.17×**. The nightly config (b=64, n=10240)
sits in this regime. A14 stands: **the branch-crossing lever is absent at the baseline of record.**

Three residuals the convergence must not paper over.

**1. `decode_b8` (1.77×, 1.78×) is not a threshold crossing either — I need to be precise about this.**
At `ctx16384` the pair is `p=4 → p=4`: *both* members are two-stage. At `ctx4096` it is `p=2 → p=4`.
So b=8's ratio is movement **within** the `p>1` regime, not across the boundary. What moves is the
XLA gather: `jnp.take_along_axis(scores, safe_cand_indices, axis=1)` at `sparsecore_topk.py:439`
gathers `p·k_p` candidates out of an `n`-wide row, and `n` goes 4096 → 65536. The AgentX profile
`measured (j-a71a648c)` prices exactly this operation — `gather_offload`, 2 per indexer layer — at
**51.54 µs each** at n=262144. Measured Δ for `decode_b8_ctx4096` is 129 − 73 = **56 µs**. Same
order, same sign, and the DMA model below under-predicts it by 15× (see item 3), so the gather is the
dominant moving term. **Correction to A14's wording:** the mechanism is not "p crosses 1" but "the
two-stage path exists at all, and its XLA gather scales with `n`." The gate is `b < 32 or n > 65536`;
once through it, cost keeps growing with `n`. At nightly the gate is not passed at all, so this is
moot for the baseline — but a candidate that *reduces* the batch row count below 32 walks into it.

**2. The prefill rows (2.88×, 3.41×) do NOT cross into `p > 1` — and that is the interesting result.**
Both stay at `p=1`: `b` is 1024 and 4096, far above 32, so `max_p = 1`; and `p_min = cdiv(65536, 65536) = 1`.
Neither term of `_pick_partition` moves. So a 2.88–3.41× ratio is produced entirely **inside the
single-stage path**, where the only quantity that scales with `n` is the unconditional `slice_len`
DMA at `sparsecore_topk.py:154-161`. The compute does not scale: `i_hi = min(slice_len, eff_len)` and
`eff_len ≈ 3584` for a 1024-token prefill at ctx 4096, unchanged between the pair members. Testing
that mechanism against both prefill rows, where total unconditional DMA is `b · n · 4` bytes:

| shape | narrow bytes | wide bytes | Δ bytes | Δ stage B | **implied rate** |
|---|---:|---:|---:|---:|---:|
| `prefill_q1024_ctx4096` | 16.78 MB | 268.4 MB | 251.6 MB | +0.461 ms | **0.546 TB/s** |
| `prefill_q4096_ctx4096` | 67.1 MB | 1073.7 MB | 1006.6 MB | +1.678 ms | **0.600 TB/s** |

**Two shapes differing 4× in volume imply the same rate to within 10%.** That is a fit across a real
lever arm, not a constant fitted to one point. The unconditional `slice_len` DMA, running at
~0.57 TB/s, accounts for the prefill ratios in full.

**3. The b≥32 decode residual of 7-17% is the same mechanism, but the model over-predicts it 2-3×.**
Applying 0.57 TB/s to the decode pairs:

| shape | Δ bytes | predicted Δ | measured Δ | ratio |
|---|---:|---:|---:|---:|
| `decode_b32_ctx4096` | 7.86 MB | 13.8 µs | 7 µs | 2.0× over |
| `decode_b64_ctx4096` | 15.73 MB | 27.6 µs | 9 µs | 3.1× over |
| `decode_b32_ctx16384` | 6.29 MB | 11.0 µs | 4 µs | 2.8× over |

So the answer to "what is the residual 7-17%" is: **the unconditional `slice_len` DMA is the right
mechanism — it is the only term in the single-stage path that moves with `n`, and it fits prefill to
10% — but at decode it accounts for only about a third of what it predicts, consistently across three
pairs.** Something recovers ~2/3 of the decode DMA that does not recover it at prefill. The obvious
structural difference is wave count: `num_waves = cdiv(b, 32)` is 1–2 at decode and 32–128 at prefill,
so at prefill each subcore serializes 32+ `start(); wait()` DMAs against its own compute (A13's
no-overlap pattern, paid 32 times), while at decode it pays it once or twice and the 32-way subcore
concurrency has proportionally more of the transfer in flight. That is a **hypothesis, `assumed`**, not
a measurement — I can name the mechanism and cannot price it. What is safe to carry: the decode
residual is *bounded above* by the DMA-volume model at 13.8/27.6/11.0 µs and is measured at
7/9/4 µs, and either figure is a single-digit-percent effect on the call.

**Net effect on A14: it survives and is sharpened.** The nightly config is on the `p=1` branch on both
sides of any plausible width change; the 5.9× of `report.md` finding (1) is unavailable at the
baseline; what remains at b≥32 is a 7-17% DMA-volume residual whose mechanism is item 3. §6.3's own
"stage B's cost is not a function of buffer width and I do not know what it is a function of" is
partly answered here: it is a function of **which branch of `_pick_partition` you are on** (the b8
rows), plus **`b · n · 4` bytes of unconditional DMA** (the prefill rows, to 10%; the decode rows to
within 3×). Both are `modelled` from source and fitted to `measured (j-63b9f2e4)`; neither is a
measurement of the mechanism itself. §6.3's `profile_b1_ctx110592` anomaly — one row of 262144
costing *more* than sixteen — is consistent with the branch story (b=1 → `max_p = 32`, deep
partitioning, a large gather) but I have not worked it through and do not claim it.

## Convergence with areas 2, 6 and 7 on the stage B DMA

Areas 2, 6 and 7 independently reached the same source fact: `_topk_kernel`'s scan loops are bounded
by `i_hi = min(slice_len, eff_len)`, while the DMA at `sparsecore_topk.py:154-161` fills
`resident_vmem` with a static `slice_len = align_to(n, 16)` words **unconditionally**. Stage B's
compute tracks the real sequence length; its HBM read does not. Four sheets pointing at one lever is
worth resolving now rather than in step 3.

**The source reading is correct, and my A13 did not account for it.** A13 computed stage B's DMA as
2.62 MB against 67 µs → ~0.05 TB/s demanded → "≲2-5%". That arithmetic used the *nightly* width and
implicitly assumed the DMA runs at ceiling. Redone with the ~0.57 TB/s rate that §6.3's prefill pair
independently implies (previous section, item 2):

| nightly phase | b | n | unconditional DMA `b·n·4` | at 0.57 TB/s | stage B `measured (j-8b1e7acc)` | **DMA share** |
|---|---:|---:|---:|---:|---:|---:|
| decode | 64 | 10240 | 2.62 MB | 4.6 µs | 67.4 µs | **6.9%** |
| prefill | 1024 | 10240 | 41.9 MB | 73.6 µs | 481.0 µs | **15.3%** |

**A13 needs its bound widened and its conclusion survives.** The correct statement is ≲7% of stage B
at nightly decode and ≲15% at nightly prefill, not ≲2-5%. But the lever the other three sheets are
pointing at is not the whole DMA — it is only the **difference** between `slice_len` and `eff_len`,
which is the part a guard would reclaim. At the nightly config those two are close:

- decode, ctx 8704: `slice_len` 10240 vs `eff_len` ≈ 8704 → 15% of 4.6 µs = **0.7 µs**, which is
  **0.3% of the 221.4 µs call**.
- prefill, ctx 9216: `slice_len` 10240 vs `eff_len` ≈ 8704 average → 15% of 73.6 µs = **11 µs**, which
  is **1.3% of the 857.3 µs call**.

**So: A13 holds. Do not spend a lever there at the baseline of record.** Guarding the `slice_len` DMA
is worth ~0.3% at decode and ~1.3% at prefill — under, or barely at, §6.3's own 0.8-2.5% noise floor,
which means a step-3 variant carrying only this change **could not be shown to have worked.** That is
the decisive argument, and it is stronger than the percentage: it is not merely small, it is
unmeasurable with the instrument we have.

Two qualifications, so this kill is not over-read:

- **It is only a kill at the nightly width.** The same defect is 100% of the §6.3 prefill ratios and
  would be worth 2.88–3.41× at `max_model_len` 65536, and correspondingly more at AgentX's 262144.
  `slice_len` and `eff_len` diverge by the ratio `max_model_len / context`, and at nightly that ratio
  is 1.18. The defect is real, the fix is correct, and the baseline of record is simply the one
  configuration where it does not pay. If the target ever moves to a long-`max_model_len` deployment,
  this becomes a first-order lever — and it is a one-line change, so it is worth recording as such
  rather than discarding.
- **The rate it is priced at is `modelled`.** 0.57 TB/s is fitted from two §6.3 prefill rows, not
  measured on SparseCore, and §6.3's decode rows say the same model over-predicts by 2-3×. If the true
  decode rate is 3× better, the decode figure drops to 0.1% rather than 0.3% — the conclusion moves
  further in the same direction. There is no plausible value of the rate at which this lever becomes
  worth taking at nightly, because the *reclaimable fraction* is 15% of a term that is itself ≤15%.
  That insensitivity is why I am willing to close it.

## Falsifying A3/D2 — the claim that stage A may already be near its floor

A3 and D2 assert: the DMA granule is pinned at 256 KiB (A2), the block-size curve puts that at 25-43%
of ceiling, therefore stage A's realistic floor is 109-170 µs rather than 57 µs, therefore stage A at
a derived 109-151 µs may be at **70-100% of its achievable floor** and the schedule is not the target.
This bears on the skill's roofline gate — an 80% reading would stop a kernel being written at all —
so it must be falsifiable rather than persuasive.

**First, the circularity the coordinator flagged is real, and I overstated the sheet's evidence.**
A3 and D2 claim "two independent routes." They are not independent. Route 1 is a prediction from the
block-size curve. Route 2 divides the measured stage A by 80 steps and by 2 129 920 B to obtain
"1.13-1.57 TB/s achieved" — but that division only yields a *bandwidth* if the step time is DMA time,
which is the conclusion. If instead each step is compute-bound at 1.36-1.89 µs — which requires the
VPU to be 3-4× slower than the `assumed` 1.72e12 ops/s, a rate `report.md` §7 explicitly failed to
measure — route 2 produces exactly the same number and means nothing. Route 2 is a **consistency
check on route 1, not corroboration of it**: it establishes that *something* costs 1.36-1.89 µs per
step and that DMA at the curve's rate would be about that size. A3's parenthetical
"independent of the curve" and D2's "an entirely independent route" should both be read as
"a residual the curve happens to fit." The claim rests on one route plus a non-contradiction.

Breaking the circularity requires a measurement that separates stall from issue **inside** the kernel.

**The single measurement that would confirm it.** `/xprof-llo` on `StreamIdxTC-d` at the nightly
decode shape, captured through `/run-command-in-cdk` with the kernel-trace `LIBTPU_INIT_ARGS` that an
ordinary trace does not set. The discriminating quantity is the fraction of the kernel's cycles spent
in the DMA-semaphore wait at `streamindex_topk.py:398` (`wait_fetch_bkv`) — in the LLO output, cycles
attributed to semaphore-wait / DMA-stall bundles versus VLIW bundles issuing MXU or VPU ops.
**Confirms A3/D2 if DMA-stall ≥ 50% of kernel cycles** — i.e. ≥ ~55 µs of a 109-151 µs stage A. A4
prices the transfer at 0.72 µs/step at ceiling against a 0.49 µs/step compute budget, so stalling for
half the kernel means the transfers are arriving at well under ceiling, and A2 says the granule is the
only thing they could be limited by.

**The single measurement that would refute it.** The same capture, **DMA-stall ≤ 15%**. Then the DMAs
are keeping up and are effectively hidden, stage A's 109-151 µs is compute, and the "256 KiB granule
costs 2-4×" story is wrong. Note what refutation implies: if the time is compute and the modelled
compute is 0.49 µs/step against an observed ~1.4-1.9 µs/step, the `assumed` VPU rate is off by 3-4×
in the direction that also rescues `report.md` §4's prefill floor — so this one capture discharges
both A3/D2 and the open VPU question that §7's probe failed to answer. Between 15% and 50% the answer
is mixed; report the split and budget both terms.

**Cost:** one `/run-command-in-cdk` job on a 4-chip pod plus the dump, ~20 minutes. Not an end-to-end
benchmark, and per CLAUDE.md nothing from the dump-side tools is a result — this is static/trace
analysis feeding a design decision, and it does not go to `/log-experiment`.

**The cheaper crude discriminator, worth running first.** Sweep `bkv_p ∈ {1, 2, 4}` at fixed decode
context in the existing `bench_streamidx_topk.py`. `bkv_p` changes outstanding DMAs per wait (4, 8, 16)
and inner-step count (10, 5, 3) while holding the 256 KiB granule and the total bytes **fixed** — it is
the one knob that varies concurrency alone. If A3 is right and concurrency is what recovers bandwidth,
stage A falls monotonically in `bkv_p`. If the DMAs are already at ceiling, stage A is flat or rises
(the ragged-tail overshoot of A7 grows: `num_bkv·bkv_sz` goes 9216 → 10240 → 12288 at `kv_len_max` 8704).
This does not identify the mechanism but it directly answers G1's design question, costs one job of
~10 minutes, and needs no new code. **Run the incumbent in the same job as every arm** — `report.md`
§6.2 measured +8.5% cross-job spread on an identical shape, so a cross-job comparison at this
magnitude decides nothing.

**The durable version, which belongs back in `03-pipelining.md`.** The reference file's block-size
curve was measured on a serial pure copy at concurrency 1. The gap that actually bites here is not
"compute to hide behind" (G2 as the file words it) but **outstanding-DMA concurrency**, which the file
does not sweep at all. An `hbm_bandwidth_arms`-style probe at a fixed 256 KiB block sweeping
N ∈ {1, 4, 8, 16, 24} concurrent copies would price A3 for every future kernel on this stack, not just
this one, and would convert A3 from `from 03-pipelining.md` + inference into `verified: j-…`. That is
the answer that should go back in the file per AREAS.md's rule on gaps.
