# Batch U — DP dispatch/combine as all-to-all: results

Arm: `TPU_MOE_DP_A2A` on the DP=4 x TP=8, EP=32 hierarchical K3 config
(commit 34280e21, branch gnovack/k3-dp4-tp8). Design per
`results/batchT-leadA-a2a-report.md`; implementation in
`src/vllm_torchtpu/layers/vllm/dp_a2a_moe.py` (+ `distributed/dp_mesh.py`).
Routing runs pre-dispatch on the rank's own tokens (the pipelined-MoE
contract shift); each token's hidden row is packed with its topk
weights/ids into a single i32 payload and exchanged in the 4-member DP
group; the GMM is unchanged (`experts_start` masking, `topk_ids == -1`
padding). Combine reverses the exchange; home rank index-adds partials
(ragged) or block-sums (padded). Reduction order differs from the
reduce-scatter baseline, so output is not bit-identical.

Two build-time gotchas worth keeping: (1) the op must run over a
full-slice (dp, tp) mesh — XLA requires a collective's replica groups to
cover all 32 partitions of the outer program, so a 4-device op-local mesh
fails the verifier; (2) no `lax.axis_index` inside — it lowers to
partitionId, which this stack's SPMD partitioning rejects (rank arrives
as a tensor operand, the `_RANK_BUFFER` pattern).

## Measured (dummy weights, seqs=8 config; control = job 396, not re-run)

| c | TPOT ms a2a-padded (job396) | tok/s a2a-padded (job396) |
|---|------------------------------|----------------------------|
| 1 | **30.05 (31.49, −4.6%)** | **31.0 (29.8, +4.0%)** |
| 4 | **23.12 (24.69, −6.4%)** | **154.0 (147.3, +4.5%)** |
| 16 | **31.49 (33.09, −4.8%)** | **441.3 (430.4, +2.5%)** |
| 32 | 42.30 (42.10, +0.5%) | 689.5 (691.1, −0.2%) |

c=16 TPOT 31.49 beats TP=32 job 288's ~31.8 (the +4% miss the brief named)
— **success criterion 1 met: DP=4 x TP=8 with a2a-padded wins at c<=16**.
c=32 is parity (criterion 2 not met); c=64/128 (seqs=32 config) not run —
out of slice budget.

The c=1 saving is ~1.44 ms/step, inside the predicted 1.4–1.9 ms/step
band (2 collectives replace 3; the latency floor caps it).

## Pricing the exchange (devpct, c=1 decode profiles; control
batchQ2 2026_09_01_15_19_14 rank0, arm batchU-a2a-padded 22_49_01 rank16;
invariant op occurrence counts match, so both traces cover the same step
count ≈ 58)

- Removed: all-gather 75.20 → 14.32 ms (−60.9); the reduce-side line
  (profiled under all-reduce) 128.23 → 81.56 ms (−46.7).
- Added: all-to-all +61.41 ms (5336 occ = 2/layer × 92 × 58 steps —
  1.06 ms/step for dispatch+combine together, ~5.8 us per a2a per layer,
  vs the 3 × ~12 us baseline collectives), data formatting +5.84 ms
  (pack/unpack), reduce +4.75 ms (padded combine block-sum, 1/layer),
  slice +4.02 ms.
- Net: −42.69 ms of 660.90 (−6.5% of the window).

Occurrence-delta check (the five-kernel killer): no new outer-graph ops at
multiples of 92/93. all-to-all +5336 = 92×58 is the exchange itself;
reduce +5336 = 92×58 is the combine sum; concatenate +2668 = 92×29 is the
payload pack — all inside the two custom-kernel modules
(`pallas::dp_a2a_moe_{dispatch,combine}_padded`), 184 calls/step total.
sort occurrences are identical to control (padded mode adds no sort).

## Ragged arm (job 445) — negative result, kept behind the same env

`TPU_MOE_DP_A2A=ragged` (sort-by-destination + two
`jax.lax.ragged_all_to_all` + home-rank index-add) ran cleanly — no warmup
deadlock, so the fused-EP-kernel suspect is exonerated for this
configuration — but measured: c=1 31.57/28.3 (parity), c=16 43.20/319.4
(+30.6% TPOT), c=32 59.59/441.1 (+41.5%). Three causes: (1) the ragged
collective is latency-competitive at c=1 but scales poorly with payload on
this stack (+50.6 ms in the devpct window, vs +61.41 for padded's two
*larger* equal-split a2a's over a comparable window — per byte it is far
worse); (2) ragged needs a third collective per layer (the counts
all-gather), so no structural round-trip win; (3) K3's topk=16 over 4 DP
ranks puts ~every token on every rank, so ragged's byte savings are ~zero
by construction. Details in `results/batchU-a2a-ragged/NOTES.md`.

## What I would try next

- c=32+ on the padded arm was parity: the padded dispatch moves 4T rows
  regardless of routing, so at high c it degenerates to AG bytes. If a
  future model has routing locality (grouped topk aligned to hosts), the
  ragged path's sort is already there — but the ragged collective itself
  needs to get faster first.
- The pack/unpack data-formatting (+5.84 ms) could shrink by exchanging
  bf16 rows with metadata appended as raw bytes only once per token-dest
  pair instead of repacking; not worth it at c<=16.
- Correctness evidence: exchange math validated against a numpy reference
  on 4 CPU devices for both modes (results/batchU-a2a-ragged/
  dp_a2a_exchange_test.py); serving-side numerical parity was not run
  (dummy weights; the canary emits garbage by construction on both arms).

Jobs: ragged=445, padded=447 (both scancelled after measurement). Control
cited from jobs 288/396/413 per the brief; no baseline was re-run.
