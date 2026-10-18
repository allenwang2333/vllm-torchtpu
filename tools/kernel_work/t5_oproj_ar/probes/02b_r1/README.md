# probe 02b — incumbent vs R1 arms (b1–b4, 16 ranks) + candidate accuracy with fault controls

Frozen before submit (md5 in README.md5). Under WHITEBOARD.v1 (md5 99b42ea5…; predictions §6 are scored against v1) and v2
(md5 ea09d5de…, re-freeze after 02a; adds the L2 bound and the R-bar share). Image s4 @bb4c85361 sha256:c0243e85… (production files
verbatim, md5 printed). Candidate: variants/fused_oproj_ar.py (md5 printed; wrappers r1_l1.py / r1_l2.py). 2x2x2,
simple-run-command-2node, 16 ranks. Instruments v1.1 (instruments.FROZEN.v1.1.md5) + v1.2 addition acc_cand.py (this dir).

## Question
Does R1 (one pallas_call: fp8 o_proj matmul + two-shot AR, barrier signalled at entry and hidden under the 6 MB weight stream)
deliver the measured skeleton saving (2.4–3.5 us/call, j-3c0dc74f) per cell, bitwise, and which mechanism carries it?

## Runs (one job, in order)
1. acc_cand.py — L1, L2, Rbar bitwise vs the incumbent on b1–b4 randn + edge + e16nv1/5/6 (vmem_poison before each call);
   faults f_straddle (L2, owners 5/10 pushed one tile early), f_norswait (sum before the RS count), f_early (RS before the barrier
   wait); chained 8-call cells (x_{k+1} = out_k[:, :1024]) with and without NaN poison between calls, for L1/L2/Rbar/f_early/f_norswait.
2. test_accuracy.py --cand r1_l1.py (frozen v1 gate: inc + L1 + negative controls + 11 faults, all cells); --cand r1_l2.py --quick.
3. probe.py — timing: fill, inc, mm, L1, L2, Rbar, Rex at b1–b4; device-trace span (rank 0, 2 trace reps) net of fill = primary;
   wall slope R 32/256/1024 = cross-check (02a: the clocks differ by +1.2–1.6 us on ≥ 7-peer kernels; both reported).

## Predictions (us per call; Δ = arm − inc; span clock net of fill; b1 / b2 / b3 / b4)
- P1 (v1 §6 R1-L1) Δ(L1): −2.0..−3.0 / −2.2..−3.2 / −2.6..−3.8 / −2.6..−3.8. Falsifier: Δ > −1.5 at any cell.
- P2 (v1 §6 R1-L2, additional to L1) L2 − L1: 0..−1.5 / 0..−1.5 / 0..−2.0 / 0..−2.0; v2 bound: ≥ −0.7 (b1, b2) / ≥ −1.6 (b3, b4).
  Falsifier: > +0.3 (slower).
- P3 (v1 §6 R-bar) Rbar − L1: +1.8..+2.6 every cell. Outside → barrier hiding is not the mechanism.
- P4 (v1 §6 R-ex) Rex − mm: ±0.4 every cell; L1 − Rex within 1.0 of the exchange 7.6 / 7.9 / 9.4 / 9.4 (02a inc2r − bar: 7.37 / 9.16).
- P5 accuracy: L1, L2, Rbar bitwise on valid rows at every cell (single and chained), padding finite, agree on 16 ranks;
  test_accuracy gate: candidate PASS, incumbent deterministic, negative controls pass, every non-blind fault caught (wrong_order blind
  as in Job #1); f_straddle caught on 8/8 production single cells (b1–b4 × randn/edge); f_norswait caught ≥ 6/8; f_early: caught
  only if the race manifests — predicted blind on single cells (0/8), unknown on chains (named blind if 0).
Void: custom calls ≠ 2 (inc) / 1 (mm, L1, L2, Rbar, Rex) / 0 (fill); hoist ratio < 3 on any arm; a non-kernel XLA op present in one
region and not in inc's (region_op_symdiff_vs_inc non-empty and unaccounted); a candidate arm that fails accuracy voids its timing.
