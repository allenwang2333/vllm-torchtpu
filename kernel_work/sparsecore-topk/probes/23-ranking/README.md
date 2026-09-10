# probe 23 — the ranking job

**Question.** Of the seven frozen candidates that cleared correctness, which are faster than the
incumbent, by how much, and *where* — inside the Pallas kernel, or in the XLA glue around it?

**Method.** One pod, one process, **one pool of 58 arm-slots**, 12 interleaved rounds, reshuffled
per round, paired REPS 20/60, read `us_median`. Eight modules staged separately (never `cp`-ed over
each other) plus **three null arms** that are the incumbent source loaded as separate module
objects, so the incumbent holds **4 slots in every cell** and the floor is measured, not assumed.
Two brackets per arm, three cells, `--ctx 9216`, `k=2048`, `n=10240`, `P=1` confirmed for all 11
modules.

**Job** `j-66c7e331-fa41-461e-8686`, image `…:fa63d6df`, `2x2x1`, tags `agent-jobs,measurer`,
61.9 s wall. Recipe read back: RUN_CMD md5 `06b1ccc85ec5a3a9b364670e33dd4f3f` on both sides. All
seven in-job gates fired green, in order: `MAP_GATE_OK` → `MD5_GATE_OK` → `DRY_RUN_OK` →
`ARM_CHECK_OK` → `SMOKE_OK` → `CAPTURE_OK` → `RANKING_DONE`. `inside_kernel_dropped` is empty — no
arm lost its inside-kernel bracket. Harness line reads `FROZEN INSTRUMENT
/root/p23/glm_dsa_block_bench.py (md5 08375d7068365e1a776c58fef09cd6d0)`.

---

## Answer — prefill cell, b=1024, ρ=0.40005 (production arrangement)

Absolutes in µs. `whole` = `jax.jit` around `sparsecore_topk` at the production 3-D entry
`f32[1024, 80, 128]`; `inside` = the `pl.kernel(...)` invocation only, operands captured from the
variant's own eager `_sc_topk_direct` so nothing is re-implemented.

| arm | whole µs | Δ whole | inside µs | Δ inside | where the win lives |
| --- | --- | --- | --- | --- | --- |
| incumbent | 275.213 | — | 240.659 | — | — |
| `v2` | **238.161** | **−37.05** | **203.259** | **−37.40** | **in-kernel, all of it** |
| `v2big` | 238.365 | −36.85 | 203.748 | −36.91 | in-kernel; **identical to `v2`** |
| `v12` `_f32in` | 250.064 | −25.15 | 240.578 | −0.08 | **entry glue only** ⚠ |
| `v4c2048` | 263.624 | −11.59 | 229.470 | −11.19 | in-kernel |
| `v11` `_v11retile` | 265.767 | −9.45 | 240.579 | −0.08 | **entry glue only** ⚠ |
| `v4ragged` | 268.125 | −7.09 | 234.494 | −6.17 | in-kernel |
| `v10-a2` `_sc2` | 396.817 | **+121.60** | 360.705 | **+120.05** | regression, both brackets |

Measured floor: **0.040 µs** (whole) / **0.433 µs** (inside), from the four incumbent slots.

## Answer — decode cell, b=64, ρ=0.850 (production arrangement)

| arm | whole µs | Δ whole | inside µs | Δ inside | verdict |
| --- | --- | --- | --- | --- | --- |
| incumbent | 49.302 | — | 46.836 | — | — |
| `v2` | **46.037** | **−3.27** | **42.769** | **−4.07** | **win** |
| `v2big` | 46.539 | −2.76 | 43.043 | −3.79 | win; = `v2` |
| `v11` | 48.090 | −1.21 | 47.104 | +0.27 | whole-only, magnitude unstable |
| `v12` | 48.408 | −0.89 | 46.584 | −0.25 | whole-only, marginal |
| `v4c2048` | 49.034 | −0.27 | 46.502 | −0.33 | **null** |
| `v4ragged` | 49.697 | +0.40 | 46.607 | −0.23 | **null** |
| `v10-a2` | 71.545 | +22.24 | 68.268 | +21.43 | regression |

Measured floor: **0.730 µs** (whole) / **0.211 µs** (inside).

## Answer — ρ=1 control cell, b=1024 (only the arms whose behaviour is ρ-dependent)

| arm | whole µs | Δ whole | inside µs | Δ inside |
| --- | --- | --- | --- | --- |
| incumbent | 425.971 | — | 390.499 | — |
| `v2` | 379.366 | −46.61 | 342.610 | −47.89 |
| `v2big` | 377.495 | −48.48 | 342.772 | −47.73 |
| `v4ragged` | 425.237 | −0.73 (floor 1.96) | 390.470 | −0.03 (floor 1.90) |

**`v4ragged` is inside the floor at ρ=1 in both brackets — the predicted null, observed.** It is a
ragged-row optimization; at ρ=1 there are no ragged rows and it must do nothing. That it does
nothing is a live positive control on the whole cell design: the harness can tell a real win from a
null in the same pool that produced −37 µs for `v2`.

---

## The three estimators disagree, and that reclassifies four verdicts

`us_median` is the pre-registered read, but the harness also stores the pooled slope (`us`) and the
unpaired estimate (`us_unpaired`). Where all three agree in sign and magnitude, the call is safe.
Where they do not, the harness's per-cell floor is too tight and the printed verdict is noise:

| call | `us_median` | pooled `us` | `us_unpaired` | printed | **corrected** |
| --- | --- | --- | --- | --- | --- |
| `v11` decode inside | +0.27 | −0.30 | −0.03 | SLOWER | **null** |
| `v12` decode inside | −0.25 | −0.29 | +0.00 | FASTER | **null** |
| `v4ragged` decode inside | −0.23 | −0.59 | +0.22 | FASTER | **null** |
| `v4c2048` decode inside | −0.33 | −1.41 | −0.22 | FASTER | **null** |

The tell was already in the raw numbers: in that cell the three nulls come out **one-signed**
(−0.205, −0.211, −0.103 µs). A one-signed null set means the base slot is systematically slow, so a
symmetric ±0.211 µs floor understates the bias by about its own width. **Nothing except `v2`,
`v2big` and `v10-a2` is resolvable in the decode inside-kernel bracket.** Every other verdict in
this probe survives all three estimators unchanged, including all six prefill rows and the two
decode whole-call rows for `v11`/`v12` (sign stable, magnitude −1.2 to −2.4 µs for `v11`).

### ⬆ Applied uniformly, the same correction also *promotes* one row

The correction was formalized in `probes/24-final/stage/final_rank.py` (STAGE 8: measure the delta
against the **median** of the four incumbent slots, keep the range as the floor) and replayed over
this probe's `ranking.json`. It reproduces all four flips above — and, applied to every row rather
than only the suspicious ones, it **promotes `v4-c2048` at `D.decodeX|whole`** from "not resolvable"
(−0.267 µs against a 0.730 µs floor, measured from a `base` slot sitting 0.467 µs below its own
median) to a resolved win of **−0.734 / −1.044 / −1.250 µs** across the three estimators. That
strengthens a candidate already adopted on the prefill evidence; it does not create one. `v4-ragged`
stays null in that cell either way.

⚠ **The promotion is marginal and must be read with its margin attached.** The *weakest* of the
three estimators, −0.734 µs, clears the 0.730 µs floor by **0.0045 µs — 0.61% of the floor**. It is a
resolved call by the pre-registered rule, and the rule is not being reached back and changed to suit
it; but "resolved by 0.61% of the floor on its weakest estimator" and "resolved" read very
differently to anyone deciding what to build on. It changes no decision: `c2048` was adopted on
prefill, where it clears by roughly two orders of magnitude (−4.500 µs against a 0.040 µs floor),
and decode is not the headline for the reason pre-registered above.

**The decomposition says more than the flag.** One estimator on the line at +0.61% with the other two clearing by **98.2%** and **80.7%** is a different object from three estimators clustered at the boundary: it says the call is probably **real** and that `us_median` is the conservative estimator here, not that the effect is borderline. Same flag, opposite readings — so the flag is always printed with its spread.

A correction that only ever removes wins is a correction applied to the rows you already doubted.
This one was applied to all of them.

## `v11` and `v12` win entirely outside the kernel

This is the largest interpretive finding. At prefill, `v12` is −25.15 µs whole-call and **−0.08 µs**
inside the kernel; `v11` is −9.45 µs whole-call and **−0.08 µs** inside. The inside-kernel figure is
below the 0.433 µs floor for both. Their prefill wins are **100% XLA glue between the 3-D entry and
the `pl.kernel` call** — the conversion/retile work, not the top-k.

`v12`'s flat inside-kernel null was predicted before the run. **`v11`'s was not** — `_v11retile`
changes the emit-side layout and was expected to move the kernel.

Two consequences, both for step 4:

1. ⚠ **This is the shape of a known trap** — a whole-call win that is an XLA `copy` the production
   caller never pays. Mitigating evidence: the bracket *is* the production 3-D entry
   `f32[b, n/128, 128]`, not a synthetic one. It is not proof. Before either is adopted, confirm in
   an in-situ profile that the production call site presents the same operand layout.
2. ⚠ **`v11` and `v12` are competing for the same ~25 µs, not stacking.** Both delete entry-side
   conversion work and neither touches the kernel. Their percentages must not be summed — which is
   the pre-registered rule (c) arriving with a concrete mechanism behind it.

## `V2_BIGSCRATCH` buys nothing over `v2`

In all six cells the `v2` − `v2big` difference is either inside the floor or **inverts sign** between
`us_median` and the pooled slope:

| cell / bracket | Δ(`v2`−`v2big`) median | floor | pooled slope says |
| --- | --- | --- | --- |
| prefill whole | −0.20 | 0.040 | −0.21 (same sign) |
| prefill inside | −0.49 | 0.433 | **+1.36 (inverts)** |
| decode whole | −0.50 | 0.730 | inside floor |
| decode inside | −0.27 | 0.211 | **+0.73 (inverts)** |
| ρ=1 whole | +1.87 | 1.960 | inside floor |
| ρ=1 inside | −0.16 | 1.903 | inside floor |

**The larger scratch allocation is not the binding resource.** Ship `v2`; `v2big` costs VMEM for a
result indistinguishable from it.

## `v10-a2` is a large regression and should be dropped

+44% to +50% of the call in every cell and both brackets, at every ρ. It is not a marginal loss to
re-measure. Its scope name `…_sc2` was present and `SCTOPK_SAMPLE_AND_COUNT=1` /
`SCTOPK_COUNT_ARITY=2` were set at import, so the arm genuinely fired — this is the sample-and-count
path's real cost, not a dead flag.

## The three pre-registered readings

**(a) `tc=2048` vs `ragged`.** The rule was "not adjudicable; a tie ships `ragged`". Half of that
premise is falsified and half holds:

| cell / bracket | `v4c2048` − `v4ragged` | floor | reading |
| --- | --- | --- | --- |
| prefill whole | −4.500 µs | 0.040 | **c2048 faster, resolvable** (−4.41 / −4.61 by the other two estimators) |
| prefill inside | −5.024 µs | 0.433 | **c2048 faster, resolvable** |
| decode whole | −0.663 µs | 0.730 | **TIE → ships `ragged`** (pre-registered) |
| decode inside | −0.105 µs | 0.211 | **TIE → ships `ragged`** (pre-registered) |

So the disposition the rule pre-registered governs decode, and only decode. At prefill the pair *is*
adjudicable and `c2048` wins by 4.5–5.0 µs, stably across all three estimators. Taken together
`c2048` **dominates** `ragged` on the measurements — strictly better at prefill, tied at decode. The
pre-registered tie-break was written to prevent a post-hoc coin-flip, not to override a resolved
comparison; the adoption call is the coordinator's, but the measurement does not support shipping
`ragged` at prefill.

**(b) `v2` in absolute µs, no percentage** — because its two reweightings are mutually inconsistent.

| cell | bracket | `v2` | `v2big` | incumbent | Δ `v2` | Δ `v2big` |
| --- | --- | --- | --- | --- | --- | --- |
| prefill | whole | 238.161 | 238.365 | 275.213 | −37.052 | −36.848 |
| prefill | inside | 203.259 | 203.748 | 240.659 | −37.400 | −36.911 |
| decode | whole | 46.037 | 46.539 | 49.302 | −3.265 | −2.763 |
| decode | inside | 42.769 | 43.043 | 46.836 | −4.067 | −3.793 |
| ρ=1 | whole | 379.366 | 377.495 | 425.971 | −46.605 | −48.476 |
| ρ=1 | inside | 342.610 | 342.772 | 390.499 | −47.889 | −47.727 |

**(c) ⛔ Do not add up the adopted candidates' percentages.** The arms overlap (see `v11`/`v12`
above) and composition is step 4's job.

---

## Reading the percentages — three traps

1. **ρ source, per cell.** Every cell's `row_lengths` came from `p18_probe.build_lengths`, i.e. the
   probes/13 `level_sim`, transcribed verbatim. The prefill cell is ρ=0.40005 **exactly**, support
   [5, 8188], 25.0% of rows with `eff ≤ k`. The decode cell is **real production decode at ρ=0.850**,
   support [8200, 9208] — **it is NOT the frozen bench's ρ=0.391 small-batch prefill occupancy.** Do
   not read the b=64 cell as the bench's 0.391 cell.
2. ⛔ **RETRACTED — "the dispatch floor is inside every absolute" is false, and so is the ~2.8×
   that followed from it.** Retracted 2026-09-10 from `probes/24-final`; see that README's
   *"⛔ Reading (5)'s mechanism was wrong"* for the full argument. In one line: the per-call cost and
   the kernel time combine as **`max(host, device)`, not `host + device`** — the harness launches
   `reps` calls and blocks once, so a device-bound arm hides its host issue entirely. This job's own
   `hostbound_pct` says so directly: if 31.53 µs of the 49.30 µs decode baseline were dispatch,
   `D.decodeX|whole` would read **64% host-bound**. It read **5.45%**, an exposed intercept of
   **2.80 µs**. Prefill reads 1.14%, intercept 3.18 µs. The 31.53 µs figure is the marginal cost of a
   *trivial* jit call, where the host is the bottleneck; it is a **resolution floor** — you cannot
   measure a per-call time below it — not an additive term in a 49 or 275 µs denominator.
   **"Quote the µs delta, not the percentage of the call" survives, on the reasons in item 3 and on
   the denominator being bracket- and cell-specific.**
3. **These are percentages of the top-k call, not of a step.** The standing decode budget puts
   sctopk at ~1.2% of decode device time, so `v2`'s −3.27 µs decode win is worth ~0.08% of the decode
   step at the very most. The prefill share is the larger prize and is the coordinator's number to
   supply.

## Cross-job consistency — the whole-call bracket reproduces probe 18

| | this job | carried denominator | agreement |
| --- | --- | --- | --- |
| prefill, ρ=0.40005, whole | 275.213 µs | 275.185 µs | **+0.010%** |
| decode, ρ=0.850, whole | 49.302 µs | 49.171 µs | +0.27% |
| prefill, ρ=1, whole | 425.971 µs | 425.366 µs | +0.14% |

Three independent baselines land within 0.3% of numbers measured in a different job on a different
day. That is the strongest available evidence that the whole-call bracket here is the same
instrument probe 18 used, and that the ranking is not sitting on a shifted machine. ⛔ The gated
prefill denominator 291.075 µs is a *correction*, not a measured baseline; ⛔ 155.82 µs is modelled;
⛔ every probes/09-derived number remains unsafe.

## Correctness carried into this job

All 11 modules × 3 cells passed the in-job smoke: `rows_wrong=0`, `oor=0`, `live_bad=0`, and
`bit==base=True` in all 31 cells. That is a *smoke* gate on this job's exact operands, not a
substitute for probe 22 — but a bit-identical output also closes every value-mediated mechanism for
these deltas, so no arm's win can be a shortcut that changes what is returned.

`v3` (`v03-fused-emit`) is **not in this job**. It failed correctness in probe 22 — 46 of the frozen
68 red, and 18 of 30 `xgate` cells red on the *weakened* multiset-only gate — with a diagnosed
mechanism: its fused emit lands key/score words in the index array (`1098907648` = `0x41800000` =
f32 `16.0`; `-2147483648` = `0x80000000` = `-0.0`). No cluster time was spent repairing it.

## Method rules this run applied or produced

- **Cross-check the pooled slope and the unpaired estimate before believing a marginal
  `us_median`.** Four "resolvable" verdicts here are noise, and the pre-registered read alone would
  not have caught them.
- **A one-signed null set is a bias, not a floor.** Three nulls all on the same side of base means
  the base slot is systematically off; the symmetric floor understates by roughly its own width.
- **Bracket every candidate twice.** Whole-call alone would have ranked `v12` second overall. The
  inside-kernel bracket is what says its win is glue, and the difference decides whether it composes
  with the in-kernel arms or competes with them.
- **Alternation was superseded, not skipped.** The C,A,C,A rule guards against slice/tier
  confounding across jobs; one pool with four incumbent slots and per-round reshuffling is strictly
  stronger, and the priority class (`cdk-p5`) is a constant here because there is exactly one job.

## Files

| | |
| --- | --- |
| `stage/rank.py` | the harness, md5 `e0bde4c103d277aa933fa585c1c49603` |
| `stage/subjects/*/` | the eight frozen subjects, md5-gated in-job against the coordinator's table |
| `stage/p18_probe.py` | probe 18 reused wholesale for `build_lengths` / `production_scores`, `ceafc734…` |
| `stage/glm_dsa_block_bench.py` | the frozen timing instrument, `08375d7068365e1a776c58fef09cd6d0` |
| `run_cmd.txt` | `06b1ccc85ec5a3a9b364670e33dd4f3f`, no `'`, no `#`, `bash -n` clean |
| `job.log` | full container log |
| `rank_1.log` / `ranking.json` | synced outputs |
