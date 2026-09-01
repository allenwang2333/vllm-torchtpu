# Batch U ragged arm (job 445) — negative result

Config: seqs=8 dp4tp8 EP=32, dummy weights, TPU_MOE_DP_A2A=ragged.
Control numbers: job 396 (AG/RS dispatch), not re-measured.

| c | TPOT ms (job396) | tok/s (job396) |
|---|------------------|----------------|
| 1 | 31.57 (31.49) | 28.3 (29.8) |
| 4 | 25.02 (24.69) | 134.9 (147.3) |
| 16 | 43.20 (33.09) | 319.4 (430.4) |
| 32 | 59.59 (42.10) | 441.1 (691.1) |

devpct (decode profile, control batchQ2 2026_09_01_15_19_14 rank0 vs arm
21_44_55 rank22): all-gather 75.2 -> 12.7 ms (the two MoE AGs are gone,
replaced by the tiny counts all-gather), ragged-all-to-all +50.6 ms (2760
occ = 92 layers x 30 captured steps, by design), data formatting +19.3 ms
(pack/unpack), all-reduce +29.7 ms at unchanged occurrence count, total
+8.2%. No new ops at exact 92/93 multiples in the OUTER graph: the new ops
live inside the two custom-kernel modules (pallas::dp_a2a_moe_{dispatch,
combine}_ragged), 184 calls/step total.

Diagnosis: (1) ragged_all_to_all on this stack is latency-competitive at
c=1 (~parity with the AGs it replaced) but scales poorly with payload, so
c>=16 loses badly; (2) the ragged path pays 3 collectives/layer (dispatch
a2a + counts all-gather + combine a2a) vs baseline's 3 -- no structural
win was available; (3) K3 topk=16 over 4 DP ranks means ~every token hits
every rank, so the ragged byte savings are ~nil by construction.
