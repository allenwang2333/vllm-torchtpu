# Probe 08 — `per-wave-residual`: what is the 81 µs that belongs to no op?

**Question.** Stage C's balance sheet left **81.28 µs — 17.9% of the prefill kernel — attributed to
nothing**, and priced the launch intercept at 22.042 µs against an instrument whose own null reads
*higher*. Three mechanisms survived the bundle argument (M1 the per-wave `sfence`+`barrier`, M2 score-DMA
latency the bandwidth figure does not contain, M3 `parallel_loop` entry/exit). Two side questions rode
along: does `plsc.addupdate_scatter` resolve intra-vector address conflicts (D5's gate), and is a skipped
refinement level free (probes/05's proposed explanation of its 2.231-vs-2.03 gap)?

**Why it needed hardware.** Every one of these is a question about what the machine does between
instructions. At 574 ps/bundle the residual is **4,425 bundles/wave against a 1,168-bundle program** — 3.8×
the whole kernel's static instruction count — so it is provably not instruction issue, and nothing static
says what it is.

## Arms

17, one job, one chip, all interleaved. `probe.py`, patching the subject by string substitution and
printing every anchor's hit count. Five region-deletion arms (`-dma`, `-cvt`, `-hist`, `-sweep`, `-emit`),
their joint (`-alld`, loop bounds forced to a **dynamic** zero; `-alls`, static zero), `P3T` (3 refinement
levels with `done` pinned dynamically true), `R1` (replication factor 1), kernel-launch slope at constant
total work (`X2`, `X4`), trivial-op launch slope (`N1`, `N2`, `N4`), two nulls.

Dynamic-vs-static zero is the point of the `-alld`/`-alls` pair: the incumbent's skipped levels have
dynamic-zero bounds and **do** enter the loop, so a static zero would measure a skip the kernel never
performs.

## Config

Job `j-300d4690-d53e-441e-9d39`, `Succeeded`. Image
`…/vllm-torchtpu-hning_google_com:fa63d6df`, read back from probes/07's recipe so the two jobs share it
exactly. Pristine `git archive b11b6eb6 src` mapped over the checkout; **pod-side md5
`44353b347c96e0eedbd742c710d4540a`** gated in-job (`MAP_GATE_OK`). Timing harness is the frozen
`glm_dsa_block_bench.measure` (md5 `08375d70…`). Shapes `b=1024` (32 waves) and `b=64` (2 waves), `n=10240`,
`k=2048`, `P=1`. **`row_lengths = n` — every arm runs at ρ = 1.0**; see §"What this does not measure".

**Cross-job control: `B` = 327.166 here against probes/07's L0 = 327.023, +0.04%** at b=1024 and −0.24% at
b=64. The same arm in a different job on a different day reproduces to well under a tenth of a percent, so
every cross-probe comparison below is legitimate.

## Answers

### The residual is real, it is M1, and M2 and M3 are dead

| | b=1024 | per wave |
| --- | --- | --- |
| `B` | 327.166 | — |
| `-alld` — all five regions' loops entered at bound 0 | 101.584 | 3.1745 |
| `-alls` — the loops gone entirely | 101.917 | 3.1849 |
| null floor (two nulls, 0.529 apart) | 19.630 | — |
| **`-alld` − null = the residual** | **81.95** | **2.561** |

**81.95 against stage C's 81.28, predicted by subtraction from a different decomposition — 0.8% apart.**
The residual is confirmed as a real per-wave cost that survives deleting the DMA, `cvt`, `hist0`, the V4
sweeps and `emit`.

- **M3 is dead.** `-alld` − `-alls` = **−0.333 µs over 32 waves**. `parallel_loop` entry/exit and the
  `pl.when` predicates cost nothing measurable. A skipped loop is free.
- **M2 is dead as the residual's mechanism**, and see the next section for where it went instead.
- **M1 is what is left**: the per-wave `sc_tpu.sfence` (904) + `sc_tpu.barrier` (907) that probes/06 Q2
  found the lowering emitting with no `subcore_barrier` in the source. **2.56 µs per wave to synchronise 32
  subcores.** Not proven by construction — it is what survives — but it is the only structure remaining in
  `-alld`, and it explains probes/06's "17 ns spread across all 16 `TEC` lines" artifact with the same object.

**A route follows immediately, and it is a big one.** `num_waves = cdiv(b, 32)` and the barrier is
per-wave, so the residual is **2.56 µs × waves** — 81.95 µs at b=1024 and 5.12 µs at b=64. It is 26.6% of
the prefill kernel and it is *pure overhead*: the source contains no cross-subcore dependency (probes/05
§Arms; the file advertises this at lines 19 and 528). If the barrier can be hoisted out of the wave loop —
one barrier per kernel instead of one per wave — the prize is `(waves−1)/waves × 81.95` = **79.4 µs, 25.8%
of the prefill kernel**, and it needs no algorithmic change at all. → **D15**.

### ⚠ The regions are strongly super-additive: single-arm deltas over-count by 42%

| removed | b=1024 delta | b=64 delta |
| --- | --- | --- |
| `-sweep` | **151.324** | 6.937 |
| `-hist` | **138.306** | 6.479 |
| `-emit` | **84.020** | 5.491 |
| `-dma` | 13.469 | 0.809 |
| `-cvt` | **−20.492** | −1.507 |
| **sum of the five** | **366.628** | 18.209 |
| **`B` − `-alld` (all five, jointly)** | **225.582** | 14.661 |

If the regions were independent these would agree. They miss by **141.0 µs**. Excluding the negative `-cvt`
arm the four positives sum to 387.1 against a joint 225.6 — **each single-arm delta over-states its region's
realizable cost by ~1.72×.**

**This is the governing fact for every step-2 candidate and it is stated nowhere else in this exercise:
multiply any modelled region deletion by ≈0.58 before believing it.** The regions contend for a shared
resource — the tilespmem scatter port and the memory system, which stage B already put at 44.2% and stage
C's calibration already showed `hist0` stalling on rather than issuing — so deleting one lets the others
expand into the time it freed. It is the in-kernel form of the same effect as
`freed-device-time-spills-onto-untouched-ops`.

Two consequences: individual rows below are **upper bounds**, the joint 225.582 is exact, and a variant
that beats 0.58× its predicted prize is the surprising outcome, not the disappointing one.

### `-cvt` ran 20.5 µs SLOWER with a whole traversal deleted — `cvt` is hiding the DMA

The probe's own `!!` alarm fired here and **its premise was wrong**; recorded because the corrected reading
is the most useful thing in the job.

The alarm reasoned bit-identical output ⟹ the pass was not removed. It was removed: `diff arm-B.py
arm-nocvt.py` is exactly the two intended lines (`z_dyn` injected, `parallel_loop(0, walk_chunks)` →
`parallel_loop(0, z_dyn)`). The output is invariant **by construction of the test data**: scores are
`standard_normal`, `monotone_key` XORs only the negatives (`bits >> 31 & MONO_MASK`), and with k=2048 of
n=10240 there are ~5120 positives, so **the top-k never contains a value `monotone_key` would have
rewritten.** Positives are already monotone as int32.

> ⚠ **Therefore `monotone_key` is untested by every "rows match B" gate in probes 07 and 08.** Those gates
> pass vacuously on it. Any future variant that touches key encoding must add a negative-heavy arm — at
> `k > n/2`, or with a scaled-negative distribution — or its correctness evidence is worthless.

So the arm is valid and its number is real: **removing a 640-chunk-per-row traversal made the kernel
slower by 20.5 µs (6.3%).** `cvt` is the first consumer of `resident_vmem` after the score DMA. Delete it
and `hist0` becomes the first consumer and eats the DMA's arrival latency directly. **`cvt` is currently
functioning as a latency absorber for the score DMA, at the same time as doing its own useful work.**

This is why `-dma` reads only 13.469 µs (§ below): a large part of the transfer's latency is already
hidden, by an op nobody chose for that purpose.

### D1 — the score DMA is 13.469 µs, not 33.43. **The caller's ceiling falls to −4.12%.**

`-dma` at b=1024 saves **13.469 µs = 0.4209 µs/wave**; at b=64, 0.809 µs total. Stage C priced the
transfer's *bandwidth* at 33.43 µs (41.94 MB ÷ 2138 GB/s). The measured exposed cost is **0.40× the
modelled bandwidth cost**, i.e. 41.94 MB in 13.469 µs = **3114 GB/s effective**, above the
`sparsecore-hbm-read-bandwidth-2138-gbs` figure — that figure was itself a fit with a 21.9 µs intercept and
is superseded here for this access pattern.

**Ceilings, measured, replacing the modelled ones in `report.md` §6:**

| | modelled (report §6) | **measured here** |
| --- | --- | --- |
| prefill, b=1024 | −7.18% | **−4.12%** |
| decode, b=64 | −2.02% | **−1.97%** |

And even that is optimistic: a double buffer removes the *exposed* DMA, but §`-cvt` shows part of what
remains exposed is already absorbed by `cvt`, so the double buffer partly competes with an existing
mitigation rather than adding to it.

**The M2 branch is closed and it closed the expensive way round.** Stage D recorded that if the residual
were DMA latency the caller's change would recover D1 and D3 together for up to **−25.2%**. It is not, and
it does not. The headline candidate is a **4%** candidate.

### D5 — RESOLVED, bit-exactly, and it is a shipping change

```
*** D5 GATE: plsc.addupdate_scatter with 16 lanes colliding on one address is EXACT
    -- the 16x replication is REMOVABLE (1024/1024 rows) ***
```

`R1` — replication factor 1: `zero_priv(NUM_BUCKETS // LANES)`, `addr = bucket`, `fold_priv`'s inner
`range(LANES)` → `range(1)` — produces **1024/1024 and 64/64 rows bit-identical to `B`**. The hardware
resolves intra-vector scatter address conflicts; V4's 16× lane-private histogram is not required for
correctness.

| | `R1` | `B` | Δ time % |
| --- | --- | --- | --- |
| b=1024 | 314.971 | 327.166 | **−3.73%** |
| b=64 | 40.240 | 41.005 | **−1.86%** |

Bit-identical output at both shapes, so this carries `bit-identical-output-closes-value-mediated-mechanisms`
— the correctness job and the causal filter are the same job, already paid for.

**But it recovers 12.2 µs of a 151.3 µs sweep, and stage D expected 13.6%.** That gap is the next result:

### ⚠ The V4 sweep is the largest region in the kernel, and the replication is not why

`-sweep` deletes all four fixed-size passes — `zero_priv`, `fold_priv`, `scan_glob`, `find_bucket` — and is
the **biggest single delta in the job at 151.324 µs**. `R1` shrinks `zero_priv` 16× and `fold_priv`'s chain
16× and buys only **12.195**. By difference, **~139 µs — 92% of the sweep — is in `fold_priv`'s serial
accumulate and the two 16-step carried scans**, not in the replication factor.

At 32 waves that is 4.34 µs/wave for `scan_glob` + `find_bucket` + `fold_priv`, i.e. ~**135 ns per carried
iteration = 235 cycles at 1.74 GHz**, for loop bodies of a handful of bundles. These are latency, not work
— which is stage C's "1.87× its issue count" reading, now much larger than stage C could see because stage
C had no arm that deleted them.

**D13 (depth-16 carries → depth 3) is therefore promoted from "part of D5's 13.6%" to the single largest
addressable item in the kernel.** The split between its three components is **not measured** — that is
`probes/09`.

### D4 — CLOSED. The 22 µs launch intercept is the instrument, not the kernel

| slope | b=1024 | b=64 |
| --- | --- | --- |
| per extra **kernel** launch, at constant total work (`X2`,`X4`) | **−0.704** | 24.180 (`X4` absent, `nan`) |
| per extra **trivial-op** launch (`N1`,`N2`,`N4`) | **20.193** | 18.395 |
| difference — on-device SC program load | **−20.897** | — |

`N1` = 20.751 against a null of 19.630: **a trivial op's launch costs what the null costs.** And launching
*this* kernel one extra time at constant total work costs ≈ 0. So the 22.042 µs intercept is the harness's
per-call `block_until_ready` dispatch floor — **it is not on-device program load and it does not exist in
production**, where the kernel is one op in a fused step with no per-call fence.

**Delete the launch row from the balance sheet at both shapes.** Its effect on decode is severe: stage C
had launch at **43.2% of the decode kernel**, and the decode kernel is really 41.005 − 19.630 = **21.4 µs**,
not 51.0. Stage C's "at two waves the kernel is more launch than kernel" is **retracted**.

(b=64's `per extra kernel launch` of 24.180 is from `X2` alone with `X4` missing from the log — a two-point
slope at the shape with the least work. The b=1024 figure is the one to use.)

### Probes/05's 2.231-vs-2.03 gap: a skipped level is free, so the gap is the distribution

`P3T` — three refinement levels with `done` pinned dynamically true — reads **327.347 against `B`'s
327.166 at b=1024, and 41.015 against 41.005 at b=64.** Per skipped level: **0.060 µs and 0.003 µs.**

Probes/05 §Decided proposed that a skipped level costing ~3.8 µs would close its 2.231-vs-2.03 gap
entirely. **Refuted: a skipped level costs 0.06.** The gap is therefore a mismatch between the host
replay's *assumed* score distribution and the probe's actual one — which probes/05 flagged as its own
carried caveat, and which is now the live explanation. Any candidate whose prize depends on the **level
count** must re-derive it against the real DSA distribution; `R1` and D13, whose prizes do not, are unaffected.

## What this does not measure — ρ

**Every arm runs `row_lengths = jnp.full((b,), n)`, i.e. ρ = 1.0.** The host derivation of the GLM-5.2
config puts prefill at a row-average **ρ = 0.40005** and decode at **ρ = 0.850**. The walk-bounded regions
(`cvt`, `hist0`, refinement, `emit`) scale with ρ; the sweep, the DMA (`pl.ds(s, slice_len)` is static —
that is D8) and the per-wave residual do not. So the *mechanisms* here are shape-independent and stand, but
**every µs figure in this file is a ρ = 1 figure** and must be reweighted before it enters the balance
sheet. `hardware_map.md` stage C′ does that.

## Decided

1. **D3 answered: M1, the per-wave subcore barrier, 2.561 µs/wave, 81.95 µs = 26.6% of the prefill
   kernel.** M2 and M3 both dead. → new blocker **D15**, hoist the barrier out of the wave loop, prize
   **25.8%**, no algorithmic change.
2. **D5 answered YES and shipped as a candidate**: `R1` is bit-exact at both shapes for **−3.73% / −1.86%**.
3. **D13 promoted to first**: ~139 µs of the 151.3 µs sweep is serial-carry latency, not replication.
   `probes/09` splits it.
4. **D4 closed and stage C's decode sheet retracted**: the intercept is the instrument's dispatch fence.
5. **D1 downgraded from −7.18% to −4.12%**, and the M2 upside that would have made it −25.2% is refuted.
6. **The 0.58 realizability factor** applies to every region-deletion candidate in step 2.
7. **`monotone_key` is vacuously tested** by every rows-match gate in probes 07 and 08.
