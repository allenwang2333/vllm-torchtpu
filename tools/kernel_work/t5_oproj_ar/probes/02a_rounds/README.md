# probe 02a — what one 16-rank data round costs, and why (fact arms, no candidate)

Frozen before submit (md5 in README.md5). Image s4 @bb4c85361 sha256:c0243e85… (same as Job #1). Instruments v1.1
common.py (instruments.FROZEN.v1.1.md5) + probe.py (md5 printed in job log). 2x2x2, simple-run-command-2node, 16 ranks.

## Question
WHITEBOARD.v1 §2.2: a two-shot data round measured 3.8 (r4) – 4.7 (r16) us (exchange/2, j-3c0dc74f) against 1.6–2.5 modelled,
and a barrier (signal-only) round 2.15–2.26. Where does the ~2 us/round excess over the model live? The same excess is in
moe_tp_decode's in-kernel AR (data rounds 7.7–9.2 us, k4-arprice), so the answer is shared.

## Arms (probe.py docstring has the exact bodies). Clock: device-trace iteration span, rank 0, net of `fill`; every arm
carries the same exposed barrier, so "round(X)" = arm X − arm `bar`. Cells b1 (M=4, 3 KB/push) and b4 (M=16, 12 KB/push).
bar, bar2 (second barrier round = signal-only round), d_c1 / d_h1 / d_x1 (1 peer: same chip / same host other chip / other
host), d_h7 (7 same-host peers), d_x8 (8 other-host peers), d_chip7 (7 other chips, same core index), d_a15 (15 peers),
a15_pw (per-slot semaphores, 15 recv + 15 send waits), a15_st (15 vector staging copies first), self15 (15 local DMAs, no ICI),
inc1r (incumbent round 1 form: 16 staging copies, pl.when-guarded pushes, own copy, sum), inc2r (both rounds = Job #1 exchange).
Driver (M6): remote DMAs per call c1/h1/x1 1, h7 7, chip7 7, x8 8, a15/pw/st 15, self15 0 remote + 15 local, inc1r 15, inc2r 30
— printed by the job from the same tables the kernels read. Peer classes come from device attributes and are asserted uniform
over the 16 ranks (an arm whose class is not uniform is dropped, not relabelled).

## Hypotheses (written before data) and their signatures
- H1 cross-host hop latency: round(x1) − round(h1) ≥ 0.8 us, and round(x8) ≥ round(h7) + 0.5. Refuted if |round(x1) − round(h1)| < 0.4.
- H2 per-peer DMA issue serialization: per-peer slope ≥ 0.08 us — (round(h7) − round(h1))/6 and (round(x8) − round(x1))/7 —
  and self15 − bar ≥ 0.8 (issue cost exists without the ICI). Refuted if both slopes < 0.04 us/peer.
- H3 semaphore wait/signal overhead: a15_pw − d_a15 ≥ 0.5 (per-wait cost), or round(bar2) ≥ 0.8 × round(a15) (the round is
  synchronisation latency, not data). Refuted if a15_pw − d_a15 < 0.2 and round(a15) − round(bar2) ≥ 1.0.
- H4 VMEM staging copy: a15_st − d_a15 ≥ 0.4. Refuted if < 0.15.
(Not exclusive: more than one may hold; the report gives each part's direct arm and the unattributed remainder vs inc1r.)

## Predictions (us, net of fill; b1 / b4)
- P1 round(bar2) = bar2 − bar: 1.5..2.5 both cells (Job #1 barrier 2.15–2.26 incl. a launch ≈ 0).
- P2 round(a15) = d_a15 − bar: 2.0..4.0 (b1), 2.5..4.5 (b4).
- P3 inc2r − bar: 7.0..10.0 both cells (replicates Job #1 exchange 7.61 / 9.37); inc2r − inc1r (round 2): 3.0..5.0.
- P4 inc1r − d_a15 (staging 16 + pl.when + own copy, combined): 0..1.5.
- P5 round(b4) − round(b1) at a15 (bytes 4x): 0..1.5.
- P6 self15 − bar: 0.3..2.0.
Void: custom-call count != 1 (0 for fill); trace parse failure on > 2 arms; hoist ratio T(1024)/T(32) < 3 on any arm; a peer
class not uniform drops that arm only.

## Routing
H1 dominant → nothing fixable in the kernel (topology); price hierarchical / fewer-cross-host-peer schemes via d_chip7 + d_c1.
H2 dominant → fewer descriptors (hierarchical core-pair pre-reduce: 2·round(c1) + 2·round(chip7) vs 2·round(a15)).
H3 dominant → aggregated waits / defer send drains (shared lever for small_all_reduce and moe_tp_decode).
H4 dominant → push from the source window (R1 already deletes staging).
