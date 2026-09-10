# Probe 09 — splitting the V4 sweep, and de-contaminating the residual

**Job** `j-4e0df30a-d833-4c1c-825d` · Succeeded · image `…vllm-torchtpu-hning_google_com:fa63d6df`
**Subject** `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` @ `b11b6eb6`, md5
`44353b347c96e0eedbd742c710d4540a` (gated on the pod, `MAP_GATE_OK`)
**Harness** frozen instrument `glm_dsa_block_bench.py`, md5 `08375d7068365e1a776c58fef09cd6d0`
**Shapes** b=1024 and b=64, n=10240, k=2048, P=1, single-stage. **ρ = 1** (see the ρ caveat).

## The question

Probe 08 ranked the V4 bucket sweep as the largest single object in the kernel — its `-sweep` arm was
a 151.324 µs delta, and after the ρ reweighting that made the two fixed sweeps 42.2% of the
production prefill kernel and the #1 candidate. That number was one arm deleting one region, so it
had two things wrong with it that only a split could find: it never said *which* of the four sweep
primitives cost the time, and it was scored against a `-alld` control that probe 08 itself had shown
was contaminated by a spurious `fill` loop.

Three questions, therefore:

1. Split the sweep into `zero_priv` / `fold_priv` / `scan_glob` / `find_bucket`. Which is the 151 µs?
2. De-contaminate the D3 residual — measure it with `fill` and the output DMA also off (`A0`).
3. Measure the residual's **wave slope directly** rather than inferring it by division.

## Answers

**1. The V4 sweeps cost 18.866 µs, not 151.324. D13 is retracted and demoted from #1 to 4th.**

At b=1024, with emit and fill already off (`E0` = 242.952), removing all four sweep primitives
(`E0-swp` = 224.087) saves **18.866 µs**. The four components:

| primitive | µs | note |
| --- | ---: | --- |
| `zero_priv` | 9.238 | `NUM_BUCKETS` = 256 iterations, the largest |
| `fold_priv` | 4.900 | 16 iterations |
| `scan_glob` | 3.498 | 16 iterations, carried |
| `find_bucket` | 1.200 | 16 iterations, carried |
| **sum of the four** | **18.836** | |
| **joint, exact** | **18.866** | additivity ratio **0.998** |

The four are additive to 0.2%, so the split is trustworthy and there is no contention between them.
Note the shape of it: `zero_priv` — a *dense clear of 256 buckets*, not a carried scan — is half the
sweep, and the two 16-step carried scans together are 4.698 µs. Probe 08's reading that "~92% of the
sweep is `fold_priv`'s serial accumulate and the two carried scans" is wrong on both the size and
the attribution.

**Why probe 08 read 151.324.** Both probes patch the *same four primitives* (`diff` of
`08/arms/arm-nosweep.py` against `09/arms/…` — identical anchors). The difference is what else was
live. Zeroing the sweep bounds forces `total = 0`, which makes `keep_all = total <= k` true, which
exits the refinement loop immediately, **and** leaves `b_star` garbage, which changes what `emit`
stores. So probe 08's `-sweep` deleted the sweeps *plus all refinement levels* plus most of emit's
work. The ledger closes on that reading to 3% — see "the corrected balance sheet" below. This is the
"zeroing one region's bound can switch on work elsewhere" failure mode, and it cost a candidate
ranking rather than a job.

**2. The residual survives de-contamination but shrinks to 75.683 µs — and it is now the largest
object in the kernel.**

| | b=1024 | b=64 |
| --- | ---: | ---: |
| `-alld` (probe 08's control) | 101.492 | 26.833 |
| `A0` (also fill + output DMA off) | 92.231 | 25.696 |
| fill + outdma contamination | 9.261 | 1.138 |
| — of which output DMA (`B − -out`) | 5.872 | 0.190 |
| — of which spurious `fill` | 3.389 | 0.948 |
| null floor | 16.548 | 15.747 |
| **residual `A0 − null`** | **75.683** | **9.949** |

Probe 08's 81.95 µs was 6.3 µs of `fill` that the incumbent never runs. The residual is real,
de-contaminated, and unexplained: with `dma,cvt,hist,sweep,emit,fill,outdma` all off, 75.683 µs
remains above an empty-kernel launch.

**3. The residual is not per-wave. Probe 08's "2.561 µs/wave" is retracted.**

Forcing the wave count to 8 and 16 at b=1024 moved nothing: `A0_W8` 92.012, `A0_W16` 91.994, `A0`
(32 waves) 92.231. **Slope 0.009 µs/wave, intercept 91.940** — the wave loop is free. Probe 08 got
2.561 by dividing the residual by 32, which assumed the proportionality this arm directly refutes.
At b=64 the slope is 0.017 µs/wave, also zero.

That leaves an open puzzle, and it is the reason probe 10 exists: the residual is flat in
`num_waves` yet grows **7.6×** from b=64 to b=1024, a 16× change in b. So it scales with something
tied to b that is not the wave loop — the input buffer (`b·n` words), the output buffer (`b·k`
words), or a per-row cost. Probe 10 separates those three by varying `n`, `k` and `b` independently
on this same `A0` arm.

**4. Super-additivity is retracted. The regions are additive to 2.4%.**

Probe 08 reported a 1.625× over-count from summing single-region deltas, and attributed it to
contention for the tilespmem scatter port. That factor was one bad arm. Substituting the clean sweep
number:

| region | µs (ρ=1, b=1024) | source |
| --- | ---: | --- |
| score DMA | 13.469 | probe 08 `-dma` |
| `cvt` | **−20.492** | probe 08 `-cvt` — negative, see probe 08 |
| `hist`, all levels | 138.306 | probe 08 `-hist` |
| V4 sweeps, all levels | 18.866 | **this probe** |
| `emit` | 83.830 | **this probe**, `B − E0` |
| **sum of singles** | **233.979** | |
| **joint, de-contaminated** | **228.679** | `B − (A0 + outdma)` |
| **ratio** | **1.023** | |

`emit` corroborates across jobs: probe 08's `-emit` read 84.020 against this probe's 83.830, **0.23%
apart**, so emit was never super-additive either. The whole 1.625× lived in `-sweep`. The lesson
that survives is the procedural one — *always measure the joint arm* — but the mechanism was arm
overlap, not resource contention, and no candidate's prize should be divided by 1.625.

## The corrected balance sheet

**At ρ = 1, b = 1024.** The ledger closes to 1.62%:

| term | µs | % of B |
| --- | ---: | ---: |
| `hist`, all levels | 138.306 | 42.3% |
| `emit` | 83.830 | 25.7% |
| **residual (D3, unexplained)** | **75.683** | **23.2%** |
| V4 sweeps, all levels | 18.866 | 5.8% |
| harness fence (null) | 16.548 | 5.1% |
| score DMA | 13.469 | 4.1% |
| output DMA | 5.872 | 1.8% |
| `cvt` | −20.492 | −6.3% |
| **sum** | **332.082** | |
| **measured `B`** | **326.782** | closes to +1.62% |

One further split is available without another job. Probe 08's `-sweep` = 175.842 runs everything
except the sweeps, the refinement levels and (effectively) emit. Subtracting the known terms leaves
`hist0 + emit_degenerate + fill_spurious = 84.762`, so **hist0 ≤ 84.76 µs**, with equality iff emit
under a corrupted `b_star` and the spurious fill together cost nothing. Stage C's independent
per-pass price-sheet derivation put hist0 at 85.12 µs at ρ=1 — **0.4% apart, by a different route**.
Taking hist0 = 84.76 leaves refinement hists = 53.54. Both are `modelled`, not measured.

**At production prefill ρ = 0.40005.** Walk-bounded terms scale with ρ; the fixed sweeps, the
static-`slice_len` score DMA, the k-word output DMA and the residual do not. Refinement additionally
does not run at all where `keep_all` fires (ρ ≤ 0.2 — prefill chunks 0 and 1 of 8), so refinement
terms carry a further ×0.75 and a conditional mean ρ of 0.50005:

| term | µs | % of prefill kernel | scales with ρ? |
| --- | ---: | ---: | --- |
| **residual (D3)** | **75.683** | **39.8%** | no |
| `hist0` | 33.91 | 17.8% | yes |
| `emit` | 33.54 | 17.7% | yes |
| refinement `hist` | 20.08 | 10.6% | yes, and gated |
| V4 sweeps | 15.63 | 8.2% | no, but gated |
| score DMA | 13.469 | 7.1% | **no — static `slice_len`** |
| output DMA | 5.872 | 3.1% | no |
| `cvt` | −8.20 | −4.3% | yes |
| **total** | **189.98** | | |

## What this changes

- **D3, the unexplained residual, is the #1 object in the production prefill kernel at 39.8%** — 5.6×
  the caller's original score-DMA hypothesis. Its mechanism is unknown and four candidates are now
  refuted (per-wave barrier, DMA arrival latency, `parallel_loop` entry, and now the wave loop
  itself). Probe 10 is in flight.
- **D13 (split the V4 sweep) is retracted as the #1 candidate.** Its true prize is 15.63 µs of a
  189.98 µs kernel, and half of that is `zero_priv`, a dense 256-bucket clear that a smaller
  `NUM_BUCKETS` or a masked clear addresses far more directly than restructuring the carried scans.
- **The histogram is #2 at 28.4% combined** (hist0 + refinement), which restores the second root —
  a selection that does not scatter into tilespmem — as the largest *structural* lever.
- **The score-DMA prize is unchanged in absolute terms at 13.469 µs.** It is now 7.1% of the kernel
  rather than 4.12% purely because the denominator shrank; nothing about the transfer moved. Both
  D1 (double-buffer) and D8 (ragged, ρ-length DMA) cap at that same 13.469 µs, because `cvt` already
  hides the rest of the transfer.
- **No prize is divided by 1.625 any more.**

## Caveats

- **ρ = 1.** Every arm ran `row_lengths = jnp.full((b,), n)`. Production prefill is ρ = 0.40005,
  decode 0.850. The production column above is a reweighting, not a measurement.
- **The b=64 sweep split is not usable.** Its components (0.3–0.9 µs) sit under the 0.94 µs null
  spread, and its additivity ratio of 1.351 is noise, not a finding. Use b=1024 only.
- **The patch-evidence gate is unreliable for output-suppressing arms, and has now produced two
  false alarms.** At b=64 it fired `!! -out reproduces B exactly` — but an arm that skips its output
  DMA leaves whatever was already in `out_hbm`, so a "match" is stale data, not an ineffective
  patch. The same patch demonstrably took at b=1024 (0/1024 rows match) and its delta there is
  5.872 µs. Probe 08's `-cvt` alarm was the sibling case (output invariance by construction). **A
  future probe should prefill `out_hbm` with a distinct sentinel** so the gate can tell "not written"
  from "written identically".
- **hist0 = 84.76 and refinement hist = 53.54 are `modelled`**, resting on the assumption that a
  corrupted-`b_star` emit and the spurious fill are together free. Two independent routes agree to
  0.4%, which is corroboration, not measurement.
- **Cross-job control passes.** `B` 326.782 here vs probe 08's 327.166 (**−0.12%**); `-alld` 101.492
  vs 101.584 (**−0.09%**). Both arms are byte-identical sources to probe 08's. The instrument is
  stable across jobs at b=1024. At b=64 `-alld` moved +1.86%, consistent with the larger relative
  null spread at that shape.

## Decided

1. The four V4 sweep primitives cost **18.866 µs** at ρ=1/b=1024, additive to 0.2%; `zero_priv` is
   half of it. `measured (j-4e0df30a)`.
2. Probe 08's `-sweep` = 151.324 is **retracted** as a sweep cost; it deleted the sweeps, all
   refinement, and most of emit.
3. D13 falls from #1 to 4th, at 8.2% of the production prefill kernel.
4. The D3 residual is **75.683 µs** de-contaminated, and is **39.8% of the production prefill
   kernel** — the largest single object. `measured (j-4e0df30a)`.
5. The residual is **flat in `num_waves`** (0.009 µs/wave). Probe 08's 2.561 µs/wave is retracted.
   It grows 7.6× from b=64 to b=1024 by some other axis; probe 10 separates input bytes, output
   bytes and per-row cost.
6. **Super-additivity is retracted.** Regions are additive to 2.3%; `emit` reproduces across jobs to
   0.23%. Divide nothing by 1.625.
7. `emit` = **83.830 µs** at ρ=1, and `fill` is **inert in the incumbent** (`total ≥ k` always
   holds), so `B − E0` is emit alone. `measured (j-4e0df30a)`.
