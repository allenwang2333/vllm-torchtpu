# Probe 13 — refit of the refinement level count on the *production* score distribution

**Answer: the level count is LOWER than 2.231, not higher, and the score distribution is not why.**

At production prefill (`b=1024, n=10240, k=2048`, ρ=0.40005) the walk runs
**1.300 ± 0.009 levels/row averaged over all rows** and **1.740 ± 0.009 averaged over walking rows
only** (`measured`, 4 independent KV draws). 2.231 is a hardware timing inversion, so it is already
critical-subcore, and **the number that replaces it in any cost model is the critical-subcore
1.570 ± 0.035** (§4a). Not 1.740 — the walking-row mean is a conditional average, not a cost
aggregation.

The recorded expectation ("production scores are non-negative, zero-massed, tie-dense, so the level
count is higher") is **refuted on every premise**. See §3.

Nothing here ran on a TPU. This is a numpy replay of the kernel's own integer arithmetic, which is a
deterministic function of the score data — the same method as `probes/05`, validated against it
exactly (§2).

---

## 1. Where 2.231 actually came from (provenance correction)

2.231 is **not** a distribution fit and was never fitted on `rng.standard_normal` in the sense the
brief assumed. It is `probes/07-refinement-fixed-vs-walk`'s *implied active levels*
`= (L3 − L0) / level_slope` — a **hardware timing** inversion at `b=1024, n=10240, k=2048`, **ρ=1**,
`scores = rng.standard_normal` (`probes/07/probe.py:269-291`). Decode's companion is 2.545.

Its simulated counterpart is `probes/05`'s crit/row **2.03** (mean/row **1.92**), same shape, same
generator. Probe 07 flagged the 2.231-vs-2.03 gap (9.9%) as skipped-level overhead; probes/08
refuted that. So 2.231 carries ~0.20 levels of unexplained *timing* residual, on top of ρ=1. Three
corrections stack, and only one is the distribution:

| step | value | what changed |
| --- | --- | --- |
| probe 07 timing fit, ρ=1, standard_normal | 2.231 | — |
| probe 05 replay, ρ=1, standard_normal | 1.92 mean / 2.03 crit | the timing residual |
| this probe, ρ=1, **production distribution** | **1.889 ± 0.008** mean / **2.031 ± 0.013** crit | the distribution: **−1.6% on the mean, 0.0% on crit** |
| this probe, ρ=0.40005, production distribution | **1.300 ± 0.009** mean / 1.570 crit | ρ and `keep_all` |

**The distribution change is worth nothing.** The whole drop is ρ + `keep_all`, both of which
`hardware_map.md:470-490` already models.

## 2. Mechanism and model validation

Replayed from source (`src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`):

- monotone fp32→int32 key, an involution: `m = (bits >> 31) & MONO_MASK; bits ^ m` — `:116-118`.
- `eff_len = min(row_len, n)` (`:160`), `i_hi = min(slice_len, eff_len)` (`:164`); a lane is valid
  only if `j < i_hi` **and** `key > KEY_NEG_INF` (`:180`), so the `-inf` mask at
  `streamindex_topk.py:352` drops out of the histogram entirely.
- L0: 256-bucket histogram on `key >> 24` (`:42`, `:47`, `:189`) — top 8 bits = sign + 7 exponent
  bits, so a bucket spans a factor of 4 in magnitude. `keep_all = total <= k` (`:209`),
  `done = keep_all | (quota == cnt)` (`:221`).
- 3 unrolled refinement levels at shifts 16/8/0, mask `0xFF` (`:48`); each ends
  `new_done = done | (quota2 == cnt2) | (quota2 == 0)` (`:258`). A skipped level enters its loop at
  bound 0 — not a pass over `L`. `_pick_partition` (`:494-511`) gives P=1 at both shapes.
- **Max depth is not a bug.** After 3 levels all 32 bits are consumed; any residual *exact* tie is
  broken arbitrarily by `tie_rank <= my_take` in `emit_body` (`:289`) — a documented path.

**Validation (`measured`).** `level_sim.py` reproduces the whole `probes/05` table to 2 d.p. before
anything else runs (mean/crit): random 1.95/2.50 (b=64), 1.92/2.03 (b=1024); uniform 2.05/2.50,
2.06/2.25; heavy_ties 3.00/3.00 and 100% unconverged, the expected degenerate case.

## 3. The production score generator, and why all four premises fail

`streamindex_topk.py:326-338`:
`score[q,j] = kscale[j] · Σ_h w[q,h] · ReLU(q_fp8[h] · k_fp8[j])`, then `-inf`-masked at `:352`.

**The weights are signed.** `w` comes from a plain `nn.Linear` with no activation —
`mla_attention_op.py:246` (`kw, _ = self.wk_weights_proj(hidden_states)`) and the reference
`transformers/models/deepseek_v4/modular_deepseek_v4.py:382-395`. The ReLU is *inside* the sum, per
head; the sum is not. So the scores are **signed**, and the brief's non-negativity premise is wrong
at the source.

Census of one production prefill row, `eff_len = 8192` (`measured`):

| premise | prediction | measured |
| --- | --- | --- |
| point mass at 0.0 | large | **0 exact zeros** |
| tie-dense | many | **8192/8192 distinct (100%)**; k-th/(k+1)-th boundary not tied |
| right-skewed, non-negative | yes | skew **−0.048**, mean −0.536, **81.7% negative** |
| more L0 buckets / deeper walk | higher | **14** occupied L0 buckets vs **16** for `standard_normal` |

`unconverged = 0` in **every** production configuration tested (prefill and decode, every
`NUM_BUCKETS`, ρ=0.4 and ρ=1). The pathological pivot bucket the brief warned about does not occur.

Undetermined generator parameters, swept (`measured`, prefill, b=256, 2 reps), mean_all/mean_walk:
base (α=0, signed w) 1.283/1.711; shared residual direction α=0.3/0.6/0.9/0.99 →
1.295/1.336/1.412/1.516 all-rows and 1.726/1.781/1.883/2.021 walking; **`|w|`, the brief's
non-negative premise, 1.523/2.031**; `|w|`+α=0.6 (worst found) 1.533/2.044 with 1 unconverged row of
512 — the only one anywhere; per-head q-scale spread 1.0 → 1.350/1.772; n_head ∈ {16,32,64,128} →
walking 1.710/1.735/1.694/1.728 (flat); global row rescale ×{1,2,4,1.5,1e−6,1e6} → 1.254–1.285
(exponent-band alignment is a ≤2.5% effect).

**Even the deliberately adversarial non-negative variant tops out at 2.031/2.044 — still below
2.231.** The conclusion is robust to every free parameter.

## 4. Level counts at production shapes (`measured`, 4 independent KV draws)

| config | mean, all rows | mean, walking rows | crit/row | keep_all | histogram (0/1/2/3) | unconv |
| --- | --- | --- | --- | --- | --- | --- |
| prefill b=1024, ρ=0.40005 | **1.300 ± 0.009** | **1.740 ± 0.009** | 1.570 ± 0.035 | 25.3% | 1049 / 821 / 2174 / 52 (of 4096) | 0 |
| decode b=1024, ρ=0.850 | 1.848 ± 0.002 | 1.848 ± 0.002 | 1.969 ± 0.013 | 0% | 11 / 689 / 3306 / 90 | 0 |
| decode b=64, ρ=0.850 | 1.832 ± 0.030 | 1.832 ± 0.030 | 2.250 ± 0.144 | 0% | 0 / 47 / 205 / 4 (of 256) | 0 |
| prefill b=1024, **ρ=1** (control) | 1.889 ± 0.008 | 1.889 ± 0.008 | 2.031 ± 0.013 | 0% | 3 / 562 / 3418 / 113 | 0 |

`keep_all` at prefill is **25.3% measured**, confirming the sheet's `assumed` 25%. Level 3 is reached
by 1.3% of prefill and 2.2% of decode rows and **all of them converged there** — `unconverged` is 0,
so the arbitrary tie-break at `:289` is never exercised on production data.

## 4a. Which aggregation a cost model wants — and `cnt` under it

`_sc_topk_direct` gives subcore `flat` the rows `w*32 + flat` via
`lax.fori_loop(0, num_waves, wave_body)` (`sparsecore_topk.py:157`, `:443`), and there is **no
inter-subcore barrier anywhere** (`:19`, `:528`, `hardware_map.md` D14) — a subcore runs its whole
wave loop ahead of a slower neighbour, so the kernel ends when the slowest *subcore* ends:

> **cost aggregation = max over the 32 subcores of the sum of *its own* rows** (`crit`), not the
> global mean, and **not** sum-over-waves-of-max — the latter is the cost only *if* every wave
> boundary were a barrier, so it is a strict **upper bound** (`wavemax` below).

The same rule binds `cnt`. And because v3's added work is `levels_row × cnt_row` **jointly**, the
right statistic is `crit(levels × cnt)`, not `crit(levels) × crit(cnt)` — the row that walks deep is
not the row with the big straddler set.

`cnt` = L0 pivot-bucket occupancy, the straddler set v3 walks instead of `L` (`measured`):

| config | cnt mean | cnt crit | cnt wavemax (upper bd) | cnt global max | `cnt_eff` = crit(lvl·cnt)/crit(lvl) |
| --- | --- | --- | --- | --- | --- |
| prefill b=1024 | 1071 | **1423 ± 65** (1.33× mean) | 4192 | 6495 | **1798** |
| decode b=1024 | 2358 | 2880 ± 71 (1.22×) | 6040 | 7414 | 2823 |
| decode b=64 | 2421 | 5539 ± 179 (2.29×) | 6090 | 6712 | 4924 |

Critical inflation on `cnt` at prefill is **1.33×** (the joint `cnt_eff` 1798 is **1.68×** the mean).
Decode b=64 inflates 2.29× — with only 2 waves there is nothing to average over, which is where
mean-vs-crit really bites.

## 5. Consumer 1 — v3 (fused emit), `design.md:460-482`, `:780-812`

> ⚠ **Row D of this table is superseded by §7.6(i): −33.57 µs = −17.51%, not −26.21 / −13.80%.** The
> `deleted` term below rescales `20.08 × (lvl/2.231)`, and §7.1 shows 20.08 never came from 2.231.
> The *aggregation* argument in this section stands; only the `deleted` input was wrong.

Re-priced (`modelled`, on probes/07's `measured` slopes: walk 38.004 µs/level at ρ=1, fixed sweep
19.192 µs/level, base 189.98 µs from `hardware_map.md:470-490`). All rows prefill:

| variant | lvl | cnt_eff | deleted | added | net | % of 189.98 |
| --- | --- | --- | --- | --- | --- | --- |
| A old: lvl 2.231, cnt 4096 assumed | 2.231 | 4096 | 53.62 | 44.90 | −8.72 | **−4.59%** |
| B lvl crit, cnt mean — *mismatched agg, wrong* | 1.570 | 1071 | 47.67 | 17.23 | −30.45 | −16.03% |
| C lvl walking-row, cnt mean — *README v1, wrong* | 1.740 | 1071 | 49.20 | 17.90 | −31.30 | −16.48% |
| **D CENTRAL: `crit(levels·cnt)`, no-barrier truth** | **1.570** | **1798** | 47.67 | 21.46 | **−26.21** | **−13.80%** |
| E upper bound: `wavemax` (if it were barriered) | 2.359 | 3649 | 54.78 | 42.94 | −11.84 | −6.23% |
| F absurd bound: crit lvl × **global max** cnt | 1.570 | 6495 | 47.67 | 48.84 | +1.16 | +0.61% |

**Central estimate −26.21 µs = −13.80%. Band over plausible aggregations (D..E): −13.80% to
−6.23%.** My README v1 row C (−16.48%) was 2.7 points optimistic — it mixed a walking-row level
count with a mean `cnt`, two errors in the same direction.

**Can v3 become a LOSS (net > 0)? Not under any aggregation I consider plausible.** Break-even is
`cnt_eff = 6295` (`modelled`, solving `added = deleted` at lvl 1.570). Measured critical `cnt_eff` is
**1798** (3.5× margin), the wave-barrier upper bound 3649 (1.7× margin), and even the **global max
over all 4096 rows is 6495** — v3 loses only if *every* row simultaneously carried the batch's worst
straddler set, which is not an aggregation but an impossibility. **Verdict survives; the band does
not.** New band **−13.8% to −6.2%**, entirely below the old −20%/−33% and less than half as wide.
`cnt` (4096 assumed → 1798 measured critical) still dominates the level count correction
(2.231 → 1.570), but critical aggregation gives back ~40% of what `cnt` won.

## 6. Consumer 2 — v5 (`NUM_BUCKETS` sweep) — verdict is `OPEN`

Level count vs radix, production distribution (`measured`, b=512, 2 reps), mean_all / crit / unconv:

| NUM_BUCKETS | max levels | prefill | decode | SN ρ=1 control |
| --- | --- | --- | --- | --- |
| 64 | 5 | 1.799 / 2.375 / 0 | 2.574 / 2.938 / 0 | 2.678 |
| 128 | 4 | 1.461 / 1.938 / 0 | 2.104 / 2.344 / 0 | 2.109 |
| 256 | 3 | 1.287 / 1.781 / 0 | 1.845 / 2.031 / 0 | 1.920 |
| 512 | 3 | 1.073 / 1.469 / 0 | 1.573 / 1.906 / 0 | 1.656 |

**Levels do not explode at a small radix.** At NB=64 (cap 5) *zero* rows reach level 5, and there is
no non-convergence at any radix. v5's feared failure mode does not exist on this distribution.

Priced with `T(NB) = (1 + levels_crit)·fixed(NB) + levels_crit·walk·ρ`, `fixed(NB) = 19.192·NB/256`:

| NB | levels (crit) | T (prefill, µs) | Δ vs 256 |
| --- | --- | --- | --- |
| 64 | 2.375 | 52.30 | **−28.16 µs = −14.82%** |
| 128 | 1.938 | 57.65 | −22.81 µs = −12.01% |
| 256 | 1.781 | 80.46 | 0 |
| 512 | 1.469 | 117.09 | +36.63 µs = +19.28% |

**Direction: a *smaller* radix, not a larger one.** Extra levels are cheap (they cost `walk·ρ`, and
ρ=0.4); the per-level fixed 256-bucket sweep is not.

**But `fixed(NB) ∝ NB` is measured at exactly one point, so this is `OPEN`, not a result.**
Structural support exists in source — all four sweep primitives have trip counts proportional to
`NUM_BUCKETS`: `zero_priv(NUM_BUCKETS)` at `sparsecore_topk.py:223`, and `zg`/`scan_glob`/
`find_bucket` at `NUM_BUCKETS // LANES` (`:206`, `:208`, `:211`) — and `probes/09` measured their
sizes at NB=256 (`zero_priv` 9.238 / `fold_priv` 4.900 / `scan_glob` 3.498 / `find_bucket` 1.200 µs,
additive to 0.998, total 18.866 vs probes/07's 19.192). **But there is no second value of
`NUM_BUCKETS` anywhere in probes 03, 07 or 09** — probes/03 swept *pass count* (0/1/2/4), not bucket
count — so the non-scaling per-loop intercept `c` is unbounded.

Bounding it — `fixed(NB) = c + (19.192 − c)·NB/256` (`modelled`):

| c / 19.192 | fixed(64) | T(64) | Δ vs 256 |
| --- | --- | --- | --- |
| 0.00 | 4.80 | 52.30 | −28.16 µs = −14.82% |
| 0.25 | 8.40 | 64.45 | −16.01 µs = −8.43% |
| 0.50 | 12.00 | 76.59 | −3.87 µs = −2.04% |
| 0.75 | 15.59 | 88.74 | **+8.28 µs = +4.36%** |
| 1.00 | 19.19 | 100.88 | +20.42 µs = +10.75% |

**NB=64 stops winning once the non-scaling intercept exceeds `c = 11.12 µs`, i.e. 58% of the NB=256
sweep.** With four `parallel_loop`s per level and ~22 µs of measured fixed launch cost elsewhere in
this kernel (`probes/07/README.md:122`), a 58% intercept is not obviously implausible. **v5's
"smaller radix wins" reversal therefore rests on one anchor and is `OPEN` under `TEMPLATE-design.md`
§10, with a mandatory probe: re-run the probes/09 sweep split at `NUM_BUCKETS ∈ {64, 512}`.** That
is one job and it settles the whole candidate. What is *not* open, and does not need hardware, is
the level side: the radix→levels mapping above is `measured` and shows no non-convergence and no
level blow-up at any radix, which kills v5's stated failure mode outright.

## Files

`level_sim.py` (run with `../../../../uv_venv/bin/python level_sim.py`, 28 s) and `level_sim.txt`,
its verbatim output.

---

## 7. Stage C″ reconciliation — both refinement rows go **UP**, and nothing fails to close

Requested follow-up. Short version: **no gap opens, because the production sheet has never closed
against anything.** Both rows move, both upward, and the v3 prize gets *larger* because §5 rescaled a
row that was never scaled by 2.231 in the first place.

### 7.1 Reconciling the two rows against the stated method

| row | reconstruction | sheet | verdict |
| --- | --- | --- | --- |
| refinement `hist` | `53.544 × 6/8 × 0.50005 = 20.081` | 20.08 | **exact — and it uses NO level count** |
| V4 sweeps | `18.866` split `1 : 2.231`, refinement `× 6/8` → `15.609` | 15.63 | 0.16% apart — **it does use 2.231** |

So the answer is **(a) for one row and (b) for the other**.

`53.544` is `hardware_map.md:475-476`'s modelled split of two measured numbers
(`hist, all levels` 138.306 − `hist0 ≤ 84.762`), and 138.306 was measured on hardware running
*whatever level count the data produced*. The level count is **baked into the measurement**, not
multiplied onto it. Your `38.004 × 2.231 × 0.50005 × 0.75 = 31.8` failed because 38.004 is
`probes/07`'s per-level slope and this row is not built from a slope at all.

The V4 row **is** built from 2.231 — as a *split ratio*, 1 L0 sweep to 2.231 refinement sweeps, which
is the right structural move: all four sweep primitives run identically at L0 (`zero_priv(NUM_BUCKETS)`
`:185`, `fold_priv` `:197`, `scan_glob` `:208`, `find_bucket` `:211`) and at each refinement level
(`:224`, `:245`, `:247`, `:248`), all four gated by `done` to bound 0. The residual 0.16% is the
sheet's plug: back-solving 15.63 exactly needs level 2.186, not 2.231, so **15.63 was nudged to make
the total land on 189.98**.

### 7.2 Re-derived on measured execution counts

The right basis for splitting a *measured* total is **how many level executions actually happened**,
which is the critical count, not the timing inversion: `standard_normal` at ρ=1, b=1024 — the probes'
own data — runs **2.031** critical levels (`measured`). 2.231 is 1.098× that; the excess is
`probes/07`'s unexplained timing residual (§1).

**Refinement `hist`.** The work is `levels_row × eff_len_row` — a *joint* statistic, exactly as with
`cnt` in §4a, because the long rows are also the rows that walk deep. The sheet's `6/8 × 0.50005 =
0.3750` factorises that product and loses the correlation. Measured:

`crit(levels·eff_len)` at production = **8348**, `crit(levels·n)` at ρ=1 = **20800**, so
**F = 0.4014** — the proxy is **6.6% low**.

> **refinement `hist`: 20.08 → 21.49 µs** (`modelled` on a `measured` factor). It goes **up** despite
> the level count falling 2.231→1.570, because 2.231 was never in this row and the correlation was.

**V4 sweeps.** Fixed-size, so only the execution count matters: per-level sweep =
`18.866/(1 + 2.031)` = **6.224 µs** (the sheet's 1:2.231 split gave 5.839, too small), and production
runs `1 + 1.570` of them.

> **V4 sweeps: 15.63 → 16.00 µs** (L0 6.224 + refinement 9.773). Keeping the 2.231 timing basis on
> *both* sides instead gives 15.91, −0.5% — **the convention choice does not matter.**

### 7.3 Closure — there is nothing to close, and that is the finding

`hardware_map.md:376`: *"Every probe in this exercise ran `row_lengths = jnp.full((b,), n)` — ρ =
1.0."* **No probe has ever run at ρ = 0.40005.** The production sheet's 189.98 is the sum of its own
reweighted rows (`design.md:43` labels it "modelled-from-measured"; `design.md:82` "measured
(`j-4e0df30a`) + modelled reweighting"). It is a *prediction*, not a baseline.

| | µs |
| --- | ---: |
| unchanged rows (residual 75.683 + hist0 33.91 + emit 33.54 + score DMA 13.469 + out DMA 5.872 + cvt −8.20) | 154.274 |
| refinement `hist` 20.08 → 21.49 | +1.41 |
| V4 sweeps 15.63 → 16.00 | +0.37 |
| **total 189.98 → 191.76** | **+1.78 = +0.94%** |

**So: no unattributed hole, and nothing was rebalanced.** What moves is the *denominator every share
in the exercise is quoted against*, by +0.94%. That is small enough to change no ranking, and I have
quoted both denominators in §7.5 so nothing has to be re-derived twice.

⚠ One label to fix, not mine to edit: `design.md:496` calls 189.98 "the **measured** 189.98 µs". It is
not measured; `design.md:43` and `:82` have it right.

### 7.4 The ρ=1 sheet — confirmed correct, for a stronger reason than stated

`84.79 = 38.004 × 2.231` ✓ and `42.81 = 19.192 × 2.231` ✓. **Do not touch them** — but the reason is
not "2.231 belongs in a ρ=1 sheet". It is that `57.196 × 2.231 = 127.60` **is** the measured `L3 − L0`
delta, because 2.231 was *defined* as `(L3 − L0)/57.196`. Those rows are a restatement of a
measurement, so they are `measured`, and substituting 2.031 would silently discard ~0.20 levels of
real measured time. **The ρ=1 C″ sheet has no row that changes and its +1.62% closure is untouched.**

The two sheets therefore legitimately use different level counts, and the reason is now on the
record: *the ρ=1 sheet consumes 2.231 as a measured time-equivalent; the production sheet consumes a
level count as an execution count, and the execution count is 2.031 at ρ=1 and 1.570 at production.*

### 7.5 `:103` and `:165` — both are ρ=1 rows against a **superseded** denominator

Both are percentages of **454.602 µs**, the stage-C total that `hardware_map.md:321` retracts
("The 454.602 µs total is superseded"). C″'s ρ=1 denominator is `B` = **326.782**.

| claim | as written | of `B` = 326.782 (ρ=1) | at production |
| --- | --- | --- | --- |
| `:103` refinement region 127.579 | 28.1% | **39.04%** | see ⚠ below |
| `:165` V4 sweep, L0 (19.192) | 4.2% | 5.87% | — |
| `:165` V4 sweep, refinement (42.82) | 9.4% | 13.10% | — |
| `:165` V4 sweep, total (62.01) | 13.6% | 18.98% | — |

⚠ **But the µs are superseded too.** C″ adopts `probes/09`'s **18.866 µs for all sweeps**, not 62.01.
On C″'s own numbers: sweeps are **5.77% of `B`** at ρ=1 and **16.00 / 191.76 = 8.34%** at production
(the sheet already says 8.2%). The refinement region at production is
`21.49 + 9.77 = 31.26 µs = 16.30%` of 191.76 — **not 28.1%**.

### 7.6 ⚠ Two knock-ons, one of which is large

**(i) §5's v3 `deleted` term was wrong.** It used `20.08 × (1.570/2.231)` = 14.13 — but 20.08 never
came from 2.231, so that rescale should never have happened. The correct deleted refinement-hist term
is the re-derived row itself, **21.49**:

| variant | deleted | added | net | % of 191.76 | % of 189.98 |
| --- | --- | --- | --- | --- | --- |
| **D CENTRAL** (crit 1.570, `cnt_eff` 1798, probes/07 walk) | 55.03 | 21.46 | **−33.57** | **−17.51%** | −17.67% |
| D on probes/09's implied walk (26.36 µs/level) | 55.03 | 18.25 | −36.78 | −19.18% | −19.36% |
| E upper bound (`wavemax`) | 65.83 | 42.94 | −22.89 | −11.94% | −12.05% |

**v3's central estimate moves from −13.80% to −17.51%; band −17.51% to −11.94%.** It moved *toward*
the old −20%/−33% band without reaching it. **Supersedes §5's table.**

**(ii) `probes/07` and `probes/09` contradict each other on the sweep price by 3.1×, and v5 hangs on
it.** `probes/07`'s marginal slope 19.192 µs/level implies **58.18 µs** for the 3.031 executions that
happen at ρ=1; `probes/09` measured **18.866**. Both are `measured`, in different jobs, with
different arm constructions, and all four primitives are `done`-gated identically in source so there
is no structural reason for them to disagree. C″ adopts 09 for the sweep row while stage D `:164`
still quotes 07's 19.192 — and **§6 of this document used 19.192.** At 09's 6.224 µs/level:

| NB | T (µs) | Δ vs 256 |
| --- | --- | --- |
| 64 | 41.36 | −3.03 µs = **−1.58%** |
| 128 | 38.60 | −5.79 µs = **−3.02%** |
| 256 | 44.39 | 0 |
| 512 | 53.06 | +8.67 µs = +4.52% |

**v5's prize collapses from −14.82% to −1.58%, and the optimum moves from NB=64 to NB=128.** v5 was
already `OPEN` on the unmeasured `∝NB` scaling; it is now `OPEN` on *two* unresolved terms, and the
second one is a contradiction between two measurements rather than a missing one. The mandatory probe
should therefore re-measure the sweep at NB ∈ {64, 256, 512} in **one** job with **one** arm
construction — that settles the scaling and the 07-vs-09 contradiction together. **The v5 ranking
should not be quoted until it lands.** (The level side of v5 — §6's radix→levels table — is
unaffected: it is `measured` and needs no hardware.)
