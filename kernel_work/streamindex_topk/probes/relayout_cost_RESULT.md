# P3 — the two relayouts C5 and C8 would delete are inside noise

**probe:** What do the two relayouts actually cost — the `(B, bq_sz, 32, bkv_sz)` →
`(B, bq_sz·32, bkv_sz)` reshape that C5 (S1-c, `streamindex_topk.py:416-418`) would delete, and the
`w[:, :, None]` broadcast that C8 (S1-b, `streamindex_topk.py:331`) would delete? Each measured
against its own absence.

**config:** `vllm-torchtpu` @ `4dff8488e0b137d636eea9117892ce9fef76f583`, branch
`annyan/streamidx-scores`; working tree at submit `?? kernel_work/streamindex_topk/step2/combined.md`,
`?? optimization_work/` only. Image `…/vllm-torchtpu-annyan_google_com:13617694`. One tpu7x pod,
one shard. `probes/relayout_cost.py`: `GRID = 64`, `WARMUP = 3`, `REPS = 20`, `BATCH = 8`
(the dispatch-floor batching), `vmem_limit_bytes = 128 MiB`. `FLAT = (128, 2048)`,
`TILED = (128, 16, 128)` — 262,144 elements, which is exactly `s_summed` per inner `(bq, bkv)`
iteration at `mixed_prod`.

**job:** `j-efb89184-020f-495f-9d4b`, `JobStatus: Succeeded`.

`RUN_CMD`, as read back from `cdk job recipe j-efb89184-020f-495f-9d4b --no-color`:

```
export PYTHONPATH=/workspace && python3
  /workspace/kernel_work/streamindex_topk/probes/relayout_cost.py --widths
  128,512,2048
```

## Why this is a strike test, and what its controls are

A relayout is `unmodelled` — `hardware-facts.md` gives the roofline no term for it, so there is no
figure to compare against and the only way to price one is to measure the same arithmetic with and
without it. Each arm therefore carries a control that does identical arithmetic on identical element
counts and differs only in the layout:

- **`g1_flat_flat` (CONTROL)** — arithmetic on `(128, 2048)`, stored `(128, 2048)`. No reshape.
- **`g1_tiled_tiled` (CONTROL)** — the same arithmetic on `(128, 16, 128)`, stored so. Isolates the
  cost of *being* in the tiled layout from the cost of *changing to* it.
- **`g1_leading_axis` (NEGATIVE CONTROL)** — `(128, 2048) → (1, 128, 2048)`. `hardware-facts.md`
  says a leading axis is free bookkeeping, so this arm **should read zero**. Its reading is the
  probe's own noise floor, and any primary effect smaller than it is not measurable here.
- **`g2_w_none_w{W}` (CONTROL)** — the E1 arm: no broadcast and no multiply at all.

`g2` has two distinct treatments because C8's two sub-arms are different asks:
- **L13** — `w` arrives pre-shaped `(128, 32, 1)` so no broadcast is emitted, but the multiply
  stays. Arm `g2_w_pre` vs control `g2_w_bcast` (today's code).
- **E1** — the multiply is deleted outright, on the premise that `w ≥ 0`. Arm `g2_w_none` vs
  control `g2_w_bcast`.

## Measured (job `j-efb89184-020f-495f-9d4b`)

`ns/step` is one grid step. `spread%` is `(max − min) / median × 100` over 20 batched reps.

| arm | role | elements | ns/step | spread % |
| --- | --- | ---: | ---: | ---: |
| `g1_flat_flat` | CONTROL | 262,144 | 719.7265858849278 | 15.42 |
| `g1_flat_tiled` | C5, forward | 262,144 | 730.1307505258592 | 12.21 |
| `g1_tiled_tiled` | CONTROL | 262,144 | 723.7958925543353 | 11.80 |
| `g1_tiled_flat` | C5, reverse | 262,144 | 743.6581199726788 | 10.30 |
| `g1_leading_axis` | NEG CONTROL | 262,144 | 729.5986961253220 | 16.11 |
| `g2_w_bcast_w128` | C8 control (today) | 524,288 | 1054.2246400291333 | 9.61 |
| `g2_w_pre_w128` | C8 / L13 | 524,288 | 1102.7939308405620 | 24.67 |
| `g2_w_none_w128` | C8 / E1 | 524,288 | 745.5489594576648 | 15.73 |
| `g2_w_bcast_w512` | C8 control (today) | 2,097,152 | 1371.3201951759402 | 5.76 |
| `g2_w_pre_w512` | C8 / L13 | 2,097,152 | 1410.5908121564426 | 6.34 |
| `g2_w_none_w512` | C8 / E1 | 2,097,152 | 1183.5165878437692 | 5.82 |
| `g2_*_w2048` | all three | — | **UNSUPPORTED** | — |

The `W = 2048` block did not run. All three arms raised
`RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem` — used 64.75 M / 66.68 M / 64.00 M
against 63.94 M available. The per-arm `try/except` caught them; these are unsupported arms, not a
probe failure, and nothing is inferred from their absence.

### The differences

Every percentage is a change in **time**, `(new − old) / old`. **Negative is faster.** Each row
carries the two absolute times it came from.

| pair | arm ns | ctrl ns | Δ ns | ps/element | Δ time % |
| --- | ---: | ---: | ---: | ---: | ---: |
| C5 reshape, flat→tiled (`g1_flat_tiled` vs `g1_flat_flat`) | 730.1 | 719.7 | **+6.3** | 0.024 | **+0.88** |
| C5 reshape, reverse (`g1_tiled_flat` vs `g1_flat_flat`) | 743.7 | 719.7 | +23.9 | 0.091 | +3.33 |
| layout-only, no reshape (`g1_tiled_tiled` vs `g1_flat_flat`) | 723.8 | 719.7 | +4.1 | 0.016 | +0.57 |
| **NEG CONTROL**, leading axis (want ≈0) | 729.6 | 719.7 | **+9.9** | 0.038 | **+1.37** |
| C8 / L13, W=128 (`g2_w_pre` vs `g2_w_bcast`) | 1102.8 | 1054.2 | +48.6 | +0.093 | **+4.61** |
| C8 / E1, W=128 (`g2_w_none` vs `g2_w_bcast`) | 745.5 | 1054.2 | −308.7 | −0.589 | **−29.28** |
| C8 / L13, W=512 (`g2_w_pre` vs `g2_w_bcast`) | 1410.6 | 1371.3 | +39.3 | +0.019 | **+2.86** |
| C8 / E1, W=512 (`g2_w_none` vs `g2_w_bcast`) | 1183.5 | 1371.3 | −187.8 | −0.090 | **−13.70** |

## Reading it

**C5's reshape is not measurable, and the probe says so on its own terms.** The primary effect is
6.3 ns. The negative control — an arm that by `hardware-facts.md` must cost *nothing* — reads
9.9 ns, larger than the effect it is supposed to bound. And the spreads are 10–16%, which at ≈720 ns
is ±72–116 ns, an order of magnitude above both. The honest statement is not "C5 costs 6.3 ns"; it
is **"C5's reshape costs less than this probe can resolve, which is under about 10 ns."**

The mechanism is visible in the third row: `g1_tiled_tiled` vs `g1_flat_flat` is +4.1 ns, i.e.
arithmetic in the tiled layout costs the same as arithmetic in the flat one. `(128, 2048)` and
`(128, 16, 128)` differ only in how the trailing 2048 is spelled, and both keep 128 in the lane
axis. No lane data moves. This is the "free bookkeeping" case, not a real relayout.

**As a fraction of an inner iteration** [denominators `measured (j-5f8f4a08-20b4-4fa0-bfc3)`]:

| regime | scoring time | inner `(bq, bkv)` iterations | per iteration | C5 as a fraction |
| --- | ---: | ---: | ---: | ---: |
| `mixed_prod` | 269.51 µs (by difference) | 8 × 4 = 32 | 8.422 µs | ≤ 9.9 / 8422 = **≤ 0.12%** |
| `decode_prod` | 111.4 µs (by difference) | 16 × 1 × 4 = 64 | 1.741 µs | ≤ 9.9 / 1741 = **≤ 0.57%** |
| `decode_prod` | 76.42 µs (profile scope `StreamIdxTC-d-bq_1-bkvp_2`) | 64 | 1.194 µs | ≤ **0.83%** |

The `mixed_prod` row is the tight one: the arm's 262,144 elements *is* the real per-inner-step
`s_summed` there, so no scaling is involved. At decode the real tensor is 32× smaller
(4 sequences × 1 token × 2048 = 8,192 elements), so the bound above is generous by roughly that
factor and the true share is smaller still.

Either way the answer is on the same side of the fork's 10% threshold by two orders of magnitude.

**C8's L13 arm is worse than the code it would replace, at both usable widths.** Pre-shaping `w` to
`(128, 32, 1)` to avoid the broadcast reads +2.86% at W=512 and +4.61% at W=128 — *slower*, not
faster. The direction is consistent across two widths and the W=512 pair has tight spreads (6.34%,
5.76%), so this is not a coin flip. The broadcast Mosaic emits for `w[:, :, None]` is already free;
handing it a pre-shaped operand instead forces a real `(128, 32, 1)` tensor through the pipeline and
costs a little. **L13 buys nothing and is refuted.**

**C8's E1 arm has a real prize, and P6 has already disqualified it.** Deleting the broadcast *and*
the multiply is 13.70% faster at W=512 and 29.28% at W=128 — genuinely large, and the biggest single
effect in this probe. But the prize is the multiply, not the relayout: L13 above shows the broadcast
component is worth nothing or less. And E1 is only legal if `w ≥ 0`, which **P6 measured to be false
on the real GLM-5.2-FP8 checkpoint** (50.0% of `weights_proj` entries negative, 0 of 32 rows
all-non-negative). E1 is rejected on correctness; its speed is not in question and is not available.

**The W=128 block cannot carry a conclusion on its own.** `g2_w_pre_w128` has a 24.67% spread —
±260 ns, five times the 48.6 ns effect. Only the W=512 block is quantitatively usable; W=128 is
quoted because it agrees in *direction* on both arms, which is all it is asked to do.

## Branch selected

**`combined.md` P3 fork (line 689): the `Inside noise` branch.**

> *"**>10% of an inner step:** build C5 (L3's 3-D contraction) and take C8's L13 arm; both are pure
> wins."*
> *"**Inside noise:** drop C5 and C8, and spend the design effort on C1/C2, where the prize is
> `modelled` at 2× decode's HBM floor."*

C5 is ≤0.12% of an inner iteration at mixed and below the probe's own negative control. C8's L13 arm
is not merely small, it is the wrong sign. Both die.

Note that this reaches the same place as C8's other route: L13 is refuted here by measurement, and
E1 is refuted by P6 on correctness. **C8 is closed from both directions** and should be recorded so,
to stop it being re-proposed on the theory that the untried sub-arm might have been the good one.

One consistency check worth carrying to wave 4: **P4 selects the `≥4×` branch, whose text reads
"C6 dies, and S1-c's relayout must be paid or absorbed by C5 instead."** Read together with this
probe there is nothing to absorb — S1-c's relayout is already ≤0.12% of an inner step, so "paid" is
the correct and cheap outcome. The two probes are not in conflict; the fork text simply anticipated
a cost that turned out not to exist.

## Verdict

| arm group | verdict |
| --- | --- |
| `g1_*` (C5) | **PASS** — question answered: the reshape is below the resolution of a probe whose noise floor is itself only ~10 ns. C5 dead. |
| `g2_*_w128`, `g2_*_w512` (C8) | **PASS** — L13 refuted with the wrong sign at two widths; E1's prize quantified but disqualified by P6. C8 dead. |
| `g2_*_w2048` | **UNSUPPORTED** — `RESOURCE_EXHAUSTED` in vmem on all three arms; nothing inferred. |

**decided:** C5 and C8 both drop off the candidate list. Design effort goes to C1/C2. E1 is recorded
as **rejected-on-correctness with a measured 13.70%-faster prize**, so that anyone who later
establishes `w ≥ 0` for some other model knows exactly what it is worth and does not have to
re-measure it.
