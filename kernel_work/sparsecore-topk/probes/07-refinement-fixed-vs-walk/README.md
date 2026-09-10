# Probe 07 — `refinement-fixed-vs-walk`: split a refinement level into its fixed sweep and its `L` walk

**Question.** `report.md` §5d′ is an open contradiction worth a third of the kernel. The §5d fit
assigns **114.3 µs to a refinement level**; probe 03's own price sheet says the level's `L` walk is a
*sparse* scatter and should cost **~35 µs**. 3.2× apart, ~150 µs misattributed. §5d′ listed three
explanations, and the favoured one was that the fitted parameter is a whole *level* — including a
fixed 4096-word `zero_priv` and a 4096-word `fold_priv` sweep that probe 03 never included.
**Candidate 1's prize swings 3× on the answer** (51.4% of the kernel, or ~15.8%), so this ran before
any design work.

**Why it needed hardware.** The two components — a fixed 32 KiB sweep and an `L`-proportional walk —
are both integer VMEM traffic on the same unit and are indistinguishable in source. Nothing in
`references/` prices either at this shape. It is exactly the case where a timing difference is the
only instrument.

## Arms

Six, all built by patching the incumbent source at four exact strings and `exec`-ing each result
into its own `types.ModuleType`, so no arm can be served another arm's cached executable. Each arm's
patched source is in `arms/`, and its md5 is printed in the job log.

| arm | patch | what it contains |
| --- | --- | --- |
| `L0` | `REFINEMENT_LEVELS = ()` | the kernel with **no levels at all** — the base |
| `F1n` | 1 level, `done` pinned False, `active_walk` forced 0 | base + 1 × **fixed sweep only** |
| `F1` | 1 level, `done` pinned False | base + 1 × (fixed + walk) |
| `F3n` | 3 levels, pinned, walk forced 0 | base + 3 × fixed |
| `F3` | 3 levels, pinned | base + 3 × (fixed + walk) |
| `L3` | unpatched, md5 `44353b34…` | the incumbent |
| `#null_a`, `#null_b` | a trivial jitted `x + 1.0` on `f32[8,128]` | the instrument's own floor, **two of them** — one null is a number, not a floor |

Pinning `done` False forces every level to execute on every row, which removes the data-dependent
trip count from the comparison; the count re-enters only in the last line, as `(L3 − L0)/level_slope`.

**The quantity to quote is the slope, not the pairwise difference.** `F1n − L0` and `F3n − L0` both
inherit `L0`'s error; `(F3n − F1n)/2` and `((F3−F3n) − (F1−F1n))/2` do not touch `L0` at all. The two
forms disagree — the pairwise fixed cost at 1 level reads **−4.699 µs**, i.e. negative — and the
slope is the one that is right.

**Patch evidence ran before timing**, comparing each arm's output against the incumbent by sorted-row
multiset. The pattern is the check:

```
L0 0/1024   F1n 0/1024   F3n 0/1024      <- walk removed, so the answer is wrong: the patch took
F1 130/1024                              <- 1 level suffices for 12.7% of rows
F3 1024/1024                             <- 3 forced levels = the incumbent's answer, exactly
```

`F1n`/`F3n` matching the incumbent would have meant the walk was **not** removed and would have
raised a loud `!!` on stderr. None fired.

## Config

One tpu7x pod, **one command, no fan-out** — a contended pod's wall clock is not a measurement.
Image `fa63d6df`, `git archive b11b6eb6 src` mapped over it. Pod-side md5 gated with a hard
`grep -q` so the run could not proceed past an unmapped subject: `MAP_GATE_OK`,
`44353b347c96e0eedbd742c710d4540a`. Timing by the **frozen** `glm_dsa_block_bench.measure`
(md5 `08375d7068365e1a776c58fef09cd6d0`, printed in-job as `FROZEN INSTRUMENT`), 9 interleaved and
reshuffled rounds, `us_median`, paired `REPS_LO=20`/`REPS_HI=60` slope. Every arm `suspect=False`,
`hostbound` 0.8–2.1% at prefill.

## Job

`j-098b260f-a9f7-4fe5-937c` → `j-098b260f-a9f7-4fe5-937c.txt`, `refinement_split.json`, `arms/`.

## Answer

**Slopes, `us_median`, `n=10240, k=2048`:**

| quantity | prefill `b=1024` (32 waves) | decode `b=64` (2 waves) | per wave |
| --- | --- | --- | --- |
| base, no refinement (`L0`) | 327.023 µs | 41.104 µs | 9.531 + **22.042 intercept** (⛔ SUPERSEDED by `probes/08` (`j-300d4690`): the intercept is the harness fence, not the kernel — see §4) |
| **one whole level** | **57.196 µs** | 3.899 µs | 1.777 + 0.346 |
| — fixed sweep | 19.192 µs (**33.6%**) | 1.773 µs (45.5%) | 0.581 + 0.612 |
| — `L` walk | **38.004 µs** (66.4%) | 2.126 µs | 1.196 + **−0.266** |
| incumbent (`L3`) | 454.602 µs | 51.027 µs | 13.453 + 24.122 |
| implied active levels | 2.231 | **2.545** | — |
| null floor / spread of two nulls | 27.194 / **1.261 µs** | 27.364 / **0.452 µs** | — |

The smallest quantity quoted is the decode fixed slope at 3.9× the decode null spread; the prefill
fixed slope is 15.2×.

### 1. §5d′ is resolved, and probe 03's price sheet was right the whole time

The refinement `L` walk is **1.1876 µs/wave**. Probe 03's synthetic `scatter_sparse` is **1.1013
µs/wave**. **7.8% apart** — and `ref_body` carries four integer ops per chunk that the synthetic arm
does not (the prefix match). A sparse scatter costs what probe 03 said a sparse scatter costs.

So of §5d′'s three candidate explanations: **(2) is wrong** — the level count is ~2.2–2.5, not 3.00.
**(3) is wrong** — the refinement mask's density behaves exactly like `scatter_sparse`'s. **(1) is
right in kind** — the fixed sweeps are real, 19.192 µs/level, a third of a level — **but they are
only 43 of the 232 µs the fit had assigned.**

### 2. The misattribution was in the *base*, not in the level

| region | §5d modelled | measured here | error |
| --- | --- | --- | --- |
| refinement | 232.052 µs (51.4%) | **127.579 µs (28.1%)** | model **1.82× too high** |
| base | 219.174 µs (48.6%) | **327.023 µs (71.9%)** | model **1.49× too low** |
| per level | 114.31 µs | **57.196 µs** | model **2.00× too high** |

The §5d fit had one free parameter and one equation, so **every term probe 03 had not priced was
absorbed into the refinement parameter** — the fixed sweeps, and ~81 µs of per-wave cost belonging to
the base.

> ⚠ **And it validated out of sample to −0.14%, and that validation was worth nothing.** §5d checked
> its fit by extrapolating 32 waves → 2 waves. Both the refinement term and the misattributed base
> overhead scale with waves, so the extrapolation could not separate them: **an out-of-sample check
> along the same axis the confounded terms both scale on validates nothing.** A 16× extrapolation
> landing at −0.14% looked like the strongest evidence in `report.md` §5 and was in fact the
> instrument that hid a 2× error.

### 3. Two intercepts agree, from unrelated arms

⛔ **The intercept is the instrument, not the kernel** — `probes/08` measured a trivial op's launch at
20.193 µs against a 19.630 µs null, and one extra launch of *this* kernel at constant total work at
−0.704 µs. On-device SparseCore program load is **zero**. Two independent fits agreeing to 3.2% agreed
on the harness's per-call `block_until_ready` fence, which both of them contained.

The base's two-point wave decomposition gives an intercept of **22.042 µs**. §5d's independent fit of
DMA time against wave count gave **21.349 µs**. **3.2% apart**, from different arms and a different
instrument. The fixed launch cost is real and is ~22 µs.

The `L`-walk intercept is **−0.266 µs** and the whole-level intercept is **0.346 µs** — both ≈ 0, as
a purely per-wave term must be. That is the internal check that makes the decomposition readable.

### 4. `implied active levels` — decode confirms probe 05, prefill does not, and there is a reason

Decode reads **2.545** against probe 05's critical-subcore prediction of **2.50** (1.8%). Prefill
reads **2.231** against **2.03** (9.9%).

**Do not treat the prefill gap as a distribution error yet.** `implied = (L3 − L0)/level_slope`
attributes *all* of `L3 − L0` to executed levels, but **a skipped level is not free** in `L3`: it
still enters four `parallel_loop`s with bound 0 and still runs the scalar epilogue at 249–265, three
times. A per-level skip overhead of ~3.8 µs would move 2.231 to 2.03 by itself. One arm settles it —
3 levels with `done` pinned **True** — and it is not in this probe.

## Decided

1. **`report.md` §5d′ closes.** §5d, §5e and §9 are rewritten against the balance sheet below.
2. **Candidate 1's prize falls from 51.4% to 18.6%** of the prefill kernel — close to §5d′'s
   predicted ~15.8%, and now measured rather than fitted.
3. **`hist0`'s dense scatter is the largest single row in the kernel at 25.6%**, and it was never a
   candidate on its own because the mis-fitted refinement term stood above it. It is also
   irreducible for a histogram selection — one scatter per element, whatever `NUM_BUCKETS` is — so
   it is a `mutate`/second-root question for `design.md`, not a tuning one.
4. **A new candidate, 17.9% of the prefill kernel: ~81 µs of unattributed per-wave cost**
   (2.540 µs/wave against 6.991 µs/wave of priced work — the priced terms explain only 73% of the
   base's per-wave slope). Larger than the score DMA. It needs its own probe and it is not a design
   question until it has one.
5. **The 16× lane-private histogram replication (`hardware_map.md` V4) is worth 13.6% of the prefill
   kernel** — 9.4% in the refinement levels plus 4.2% in L0 — and is now a first-class candidate.
6. `probes/03`'s price sheet is **vindicated** and stays the price sheet. What was wrong was the fit
   that used it, not the sheet.
