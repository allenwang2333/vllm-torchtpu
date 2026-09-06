# Area (3) — Pipelining memory movement against compute: constraint sheet

## Headline: the scoring loop is already at the stepped floor; the "10x gap" is the missing 512 ns term

Differencing the report's own §6 rows (`full − sparsecore_topk alone`) and fitting `T_score = F + c·num_bkv` across the three B-D4 rows, which hold the per-iteration shape fixed and vary only the block count (256/512/1024):

| row | scoring us | num_bkv | fit `F=101.6us + 0.883us·N` | residual |
|---|---|---|---|---|
| B-D4-256k | 322.0 | 256 | 327.7 | +1.8% |
| B-D4-512k | 562.3 | 512 | 553.8 | −1.5% |
| B-D4-1M | 1003.2 | 1024 | 1006.2 | +0.3% |

Now score each shape against `512 ns + bytes_step/3.48e12` [verified: j-5855c2a5-aef4-4a61-b161, `grid_step_cost_model`, tpu7x], after removing F:

| shape | steps | per-step model | per-step measured | ratio | fixed term's share of the model |
|---|---|---|---|---|---|
| A-D64-8k | 16 grid x 4 in-body = 64 | 512+603 = **1115 ns** | **1127 ns** | 1.01x | 46% |
| B-D4-256k | 1 grid x 256 | 512+301 = **813 ns** | **883 ns** | 1.09x | 63% |
| B-D1-1M | 1 grid x 1024 | 512+75 = **587 ns** | **586 ns** | 1.00x | **87%** |

**The 685 ns-against-69 ns gap the brief names as the sharpest fact in the sweep is not a mechanism. It is `bytes/BW` quoted without a step count** — the failure mode `03-pipelining.md § Deciding`, item 1 names explicitly ("a floor quoted without a step count is not a floor"). At B-D1-1M the bytes are 13% of the step. Cause (a), per-iteration fixed cost, at 1.00-1.09x of a model fitted on unrelated data. **Caveat: `c` is robust to the SC-alone subtraction (a constant bias cancels in the slope); `F` is not — it absorbs it. F needs P1 below before it is quoted.**

## Per-unit-of-work ledger

| | A-D64-8k | B-D4-256k | B-D1-1M |
|---|---|---|---|
| launch / grid | batched decode, 16 steps, sbs=4, bq_sz=1 | batched decode, **1 step**, sbs=4 | **remainder** decode, **1 step**, sbs=1 |
| in-body KV iterations | 4 per grid step (64 total) | 256 | 1024 |
| bytes in / step | 2.00 MiB as **8 descriptors x 256 KiB** | 1.00 MiB as **8 x 128 KiB** | 256 KiB as **2 x 128 KiB** |
| bytes out / step | 32 KiB as 4 x 8 KiB, stride 40 KiB | 16 KiB as **4 x 4 KiB, stride 4 MiB** | 4 KiB, 1 descriptor |
| MXU / step | 67.1 MFLOP fp8 → 29 ns peak, ~117 ns at M=32 | 33.6 MFLOP → 58 ns | 8.39 MFLOP → **15 ns** |
| VPU / step (relu, weight, 32-way sublane sum, mask, where) | 262 k elem x ~3 → **457 ns** | 131 k → 229 ns | 32.8 k → **57 ns** |
| unpriced / step | 4x transpose `(2048,1)→(1,2048)` | 4x `(1024,1)→(1,1024)` | 1x, no roofline term [hardware-facts, "Relayout: the cost with no name"] |
| buffers (depth 2, VMEM) | KV 4 MiB + Q 32 KiB + out 64 KiB | KV 2 MiB + out 32 KiB | **KV 512 KiB + out 8 KiB** |
| VMEM headroom | 4.1 of ~64 MiB used; `vmem_limit_bytes` already raised to 100 MiB | 2 of 64 MiB | **0.5 of 64 MiB — 128x unused** |
| which side dominates | fixed 512 ns > VPU 457 > bytes 603… **fixed+bytes**, transfer-side | **fixed (63%)**, neither bound | **fixed (87%)**, nothing else close |
| prologue / epilogue | one first-fetch at grid step 0; 2 `wait_send_scores` at the last; **F ≈ 58% of scoring** | 1 of 256 iterations; **F ≈ 32%** | 1 of 1024; **F ≈ 14%** |
| steady state reached? | 4 iterations per grid step — **barely**; 16 grid steps sits exactly on the `Deciding` item-4 boundary | yes, dominant | yes, dominant |

**The contrast is itself the constraint:** at (B) the loop is long and the per-call fixed cost is amortized, so the lever is step count (bkv_p). At (A) the loop is 4 iterations deep and F is the majority of the pass, so the lever is launches, not blocks. A candidate tuned at one will not move the other, and a single-seqlen measurement cannot tell F from c.

## What area (3) requires here

- Score every candidate against `Σ_steps [512 ns + bytes_step/3.48e12]` over the **in-body iteration count**, never the grid. Report §3's `bytes/3.7e12` floor is why the gap read as a missing mechanism; retract it as this block's floor.
- Because 46-87% of a step is fixed, **re-grouping the same bytes is a first-class lever and any floor that cannot rank it is disqualified** [03-pipelining, "a model with a constant bandwidth term cannot rank that lever at all"]. bkv_p is that parameter: (B) at 256 KiB/step/sequence is far below the 512 KiB-2 MiB knee [verified: j-ed517191, `block_size_bandwidth_curve`].
- Report descriptor counts, not block shapes. KV reads are whole pages and contiguous — the model applies. **The output writes are not:** at (B), 4 x 4 KiB strided 4 MiB apart, which is the "logically large, physically many small descriptors" case [03-pipelining, "a transfer's size is not the block's size when the block is strided"]. Hand the layout half to (4).
- Say out loud that at (B) the `pallas_call` grid is **1 step** and pipelines nothing; 100% of the movement is hand-rolled DMA inside one kernel body.
- Any A-D64-8k claim is gated on re-measuring the 9.1% spread (report register 13).

## What area (3) forbids here

- **Deepening the KV buffer past 2 as a first move.** The step is fixed-cost bound; extra slots hide a latency that is already hidden, at 1.5-2x the VMEM. (Also `buffer_count=3` is a hard `LoweringException` at a top-level grid [verified: j-106c8a46, `buffer_count_cap_by_position`] — moot while the DMAs are hand-rolled, but it forbids the managed spelling.)
- **Adopting raw DMA as a fresh decision.** The two-clause test [03-pipelining, "When to drop to raw DMA"] is not met: the iteration is regular (a clean loop, `bkv_p x seq_batch_size` copies per step, statically known) and only the *addresses* are data (a page-table lookup from SMEM). That is clause 2 → capability 5. The incumbent's hand-rolled schedule must justify itself **against** a `BufferedRef` subclass, not be assumed as the baseline.
- **Attributing the 5.7-32x gap to the pipeline.** At fixed block size there is at most ~1.1x here. The pipelining money is in step count, F, and TC/SC overlap.
- **Quoting a per-variant win from one seqlen.** F and c separate only across a sweep.

## The probe that attributes the per-iteration cost (five arms, one fan-out job, minutes)

All arms reuse `kernel_work/glm52-streamidx-topk/instruments/bench.py`; the shape table and `check_reachable` already exist.

1. **P1 — separate F from c without the subtraction.** Time the scoring launches *only* (drop the `sparsecore_topk` call from `_common_path`) at B-D1 and B-D4 across seqlen ∈ {64k,128k,256k,512k,1M} → num_bkv ∈ {64,…,1024}. Fit `T = F + c·N`. Affine with small residuals ⇒ **steady state is reached, cause (b) refuted**; curvature ⇒ (b) lives. Also removes the SC-subtraction bias from F.
2. **P2 — the discriminating arm. Sweep `num_kv_pages_per_block` ∈ {2,4,8,16} at B-D1-1M**, constant total bytes, step count 1024→128, per-step 256 KiB→2 MiB. Model predicts scoring 601→339→208→143 us (**4.2x**). Following the curve ⇒ **(a), and block size is the lever**. Flat ⇒ **(d), the transfer is not overlapping**. Worsening ⇒ **(c), compute superlinear in bkv_sz**. One parameter change.
3. **P3 — no-compute and no-DMA controls, isolating (c).** Arm A: keep the loop, the DMAs and the stores; replace `s_summed` with a constant. Arm B: keep the compute; pin the page index so every fetch re-reads one block. Arm C: replace `flat_bkv[:,128:129].T` with a constant bf16 vector, pricing the unpriced narrow transpose. 586 ns → ~512 means compute is 13%; → ~150 means the loop overhead is small and (c) is the story.
4. **P4 — is the double buffer overlapping?** Move `start_fetch_bkv(next)` from the top of the body to after the compute (the `no_pipelining` analogue). No change ⇒ **(d) confirmed**; the control for scale is the measured 1.62x serial-vs-managed penalty on a bandwidth-bound copy [verified: j-050c7480, `manual_dma_vs_managed`].
5. **P5 — price F directly.** Three-launch chain (two of which have a **zero-extent grid** at both B-D1 and A-D64, since `decode_batch_end` or the mixed range is empty) against a single-launch build at the same work. F is 58% of A-D64-8k's scoring pass.

## Gaps in `03-pipelining.md` this block turns on

- **"The block-size curve was measured on a pure copy; a kernel with compute may reach the knee at a different size."** Decides bkv_p at (B). Same knee ⇒ bkv_p=16 at B-D1 / 4 at B-D4. Knee moves earlier because compute hides the transfer ⇒ a smaller bkv_p suffices and the VMEM goes to the `s` intermediate instead. **P2 answers it and the answer belongs back in the file.**
- **"No probe has run a `BufferedRef` subclass on tpu7x; the file names the arm to build as a subclass against a hand-rolled pipeline over the *same* ragged access pattern."** This block **is** that arm, already measured on the hand-rolled side: page-table addresses, regular iteration. Decides whether the KV walk keeps its hand-written semaphore protocol or inherits `emit_pipeline`'s schedule (plus >2 input depth and the `ep_*` scopes). Closing it costs one variant.
- **"The `ep_*` named scopes have never been captured in an xprof."** Decides whether to restructure the walk as an `emit_pipeline` grid *for instrumentability*: if the scopes appear, `ep_wait_in` vs `ep_run_kernel` answers (a)/(c)/(d) from one profile instead of P2-P4 forever. If they do not, keep the in-body loop and pay for the arms each time.
- **"A `pallas_call` grid may not be a tracer."** The incumbent passes traced `end_seq_idx` into `grid=` (`streamindex_topk.py:643`, fed by `distribution[…]`) and ships. Either the claim is wrong on this stack or something coerces it. Decides whether "traced extent" (capability 3) forces `emit_pipeline` for this block. Trace-time check, no hardware.
- **New gap the block exposes:** `Deciding` item 2 ("flop/byte above ~1 ⇒ VPU-bound, this file is not your problem") misfires here. B-D1-1M is 32 MXU-flop/byte, far above the ridge, yet MXU is 15 ns and VPU 57 ns of a 586 ns step — the intensity is high and *both* terms are dwarfed by the fixed cost. The rule needs an absolute-magnitude guard, not just a ratio.
- **Conditionally on:** the unpriced `emit_pipeline` deep-input-buffer payoff. Only turns on if P2/P4 show the loop is latency-bound rather than fixed-cost-bound; on current arithmetic it is not.

## Seam out (I decide nothing here)

TC scoring and SC select are chained by a data dependency and serialize. **Modelled ceiling on perfect overlap, from §6's own decomposition:** B-D1-1M `max(701,781)=781` vs 1482 → **1.90x**; A-D64-8k **1.91x**; B-D4-256k **1.40x**. Combined bandwidth at B-D4-256k would be ~795 GB/s (TC) + ≤337 GB/s (SC, [verified: j-996520b8]) ≈ 1.1 TB/s against a 2.97 TB/s ceiling, so **the two units do not contend for HBM** — the obstacle is the sync mechanism, not bandwidth. Rows become ready in groups of `seq_batch_size` as the grid advances, so it is pipelinable in principle. No mechanism for TC↔SC synchronization inside one kernel is documented in `03-pipelining.md`; this is larger than anything available inside the scoring loop and goes to area (5) / `08-sparsecore.md` and `seams.md`.

## Pipeline structures considered and not favoured (no choice made)

- **Depth 3-4 on the KV buffer (legal, since hand-rolled)** — hides latency the 46-87% fixed term says is not binding; 1.5-2x VMEM for it.
- **Depth 1, synchronous KV DMA** — frees VMEM for a larger bkv_sz; costs the 1.62x serial penalty *only if* transfer-bound, which the fit says it is not. `unranked`, not rejected.
- **Raise bkv_p to the 2 MiB knee** — strongest single model prediction (4.2x at B-D1-1M) but not chosen here: it changes num_bkv, the `s` live range (4 MiB at bkv_p=16, sbs=4) and Finding 3's group-max, so it is joint with (2) and (6).
- **KV walk as a second grid axis, `grid=(groups, num_bkv)`** — inherits the managed 2-slot schedule for free, but num_bkv is kv_len-dependent and the extent would have to be the worst case, re-importing Finding 3's group-max as grid padding; the page-table index map is an SMEM lookup the window prover likely rejects.
- **`emit_pipeline` + `BufferedRef` subclass over the KV walk** — the file's named case for "addresses are data, iteration is regular"; unmeasured on this chip, which is exactly why it is the arm to build.
- **`emit_pipeline` with a plain `BlockSpec` index_map over pages** — the indices are a scalar-prefetch table lookup, not an affine map; not a route.
- **Fuse the three `pallas_call` launches into one** — attacks F (32-58% of scoring), not the step; listed here only because F is an area-(3) non-overlap term, decided in (1)/(2).
- **Peel the first/last iteration to hoist the two `pl.when`s out of the inner loop** — removes 2 optimization barriers per step [source: tokamax `ragged_dot/pallas_mosaic_tpu_kernel.py:805-812@9da2369`, "every conditional introduces an optimization barrier"]; a candidate slice of the fixed term, unmeasured.
- **`unroll=2` on the KV `fori_loop`** — lets the compiler interleave two steps' DMA issue against compute; doubles code size and the `s` live range.
- **One output DMA per bq instead of per bkv** (accumulate the row in VMEM) — deletes `seq_batch_size` small strided descriptors per step, but one (B) row is 4 MiB f32; that is really Lever 2 from area (1).
