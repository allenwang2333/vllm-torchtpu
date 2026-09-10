# Probe 15 — the two mandatory step-2 probes: the count-pass rate (`v10`) and the fused L0 store (`v3`)

Both arms in **one interleaved pool, one job, `j-78f7f537-1967-4804-8a76`**, 33 kernel arms + 3 nulls
× 9 reshuffled rounds, all on the production score distribution.

## The two answers

**A — a count pass costs `0.191 ×` a dense L0 histogram pass** (20.681 vs 108.217 µs/pass, both
slopes, R² 0.9998, ρ=1, b=1024). It is **5.23× cheaper**, where `design.md:793` modelled 2.4× in
bundle terms and 8.4 µs in its arithmetic — a per-pass ratio of 0.248. **The rate is 1.30× better
than the model, and the expensive branch that kills the root does not exist.** The count pass also
reads **within the resolution floor of `cvt`** (20.681 vs 20.327, Δ 0.354 µs < 1.282 floor): a pass
that only reads and compares costs exactly a read-only walk, so **20.3 µs is the walk floor** and no
count pass can be cheaper.

> `v10`'s effect: added term 2.12 + 2.5 × **6.48** = **18.32 µs** against the modelled 2.12 + 2.5 ×
> 8.4 = 23.12 (`design.md:791-794`; 6.48 = 0.191 × design's own `hist0` row 33.91). **−9.7% →
> ≈ −12.2%** of 189.98. Break-even moves from 6.18 to **7.99 count passes** against a modelled 2.5 —
> a **3.2× margin**. `modelled`, from a `measured` ratio.

**B — one conditional `plsc.store_compressed` fused into the L0 walk costs 6.675 µs/pass**
(mean of six densities; range 3.686–8.632), **6.2% of the L0 histogram pass it rides on**, and it
**does not scale with store density** over 0.7 %–65 %: the fit is `-2.474 µs` per unit density at
**R² 0.129**, a negative slope where physics demands a non-negative one. A second store costs the
same again (+5.309 µs at d30, +7.756 at d65).

> `v3`'s effect: `probes/13`'s `added` column is `10.98 + 0.005829 · cnt_eff`
> (`level_sim.txt:136-144`, cases D and F, which share `lvl` 1.570), and **10.98 = 27.456 × 0.40005
> exactly** — the model charged `v3`'s L0 store at `probes/03`'s `emit1`, i.e. at the price of a
> **whole extra traversal**. Measured, the two stores `design.md:565` actually adds cost
> **5.29 µs** at production ρ, not 10.98. Central case D becomes `55.03 − 15.77 = −39.26 µs`
> (**−20.47 %** of 191.76), upper bound E **−14.90 %**, and break-even rises from `cnt_eff` 6295 to
> **8531** — **above the global max 6495, so no aggregation in `probes/13`'s table, including the
> absurd bound F, turns `v3` into a loss.** `modelled`, from a `measured` marginal.

## Resolution floor

Three nulls 28.642 / 27.318 / 28.887 → **1.568 µs** instrument floor. Three **same-program
replicates** (the identical thunk registered under a second name — PROBING.md's "measure the floor
on the program you are differencing"): `A_cnt_p2` 0.214, `A_scat_p2` 0.338, `B_st_d30` **1.282 µs**.
**The resolution floor is 1.282 µs.** Every quantity quoted above clears it: the count-pass slope by
16.1×, the store marginal by 5.2× at its median point, the store's second copy by 4.1×. The two
nulls quoted *against* the floor — `cnt − cntv` 0.489 and `cnt − cvt` 0.354 — do not, and are
reported as nulls.

⚠ Probe B's six marginals span 4.95 µs peak-to-peak, **3.9× the replicate floor**, so there is
arm-level structure beyond the floor. It is not monotone in density (d65 is the *lowest* point), so
a density effect is not what it is. The defensible statement is the bound: **at every density from
0.7 % to 65 %, one fused store costs between 3.7 and 8.6 µs, i.e. at most 8.0 % of an L0 pass.**
`v3`'s verdict is unchanged across that whole interval.

## Arms

`us_median`, one pool, `glm_dsa_block_bench.measure` frozen at md5 `08375d70…`, `REPS_LO=20`,
`REPS_HI=60`, 9 rounds, dispatch floor 35.18 µs/call. `A_p0` = 82.316 µs (DMA + launch + entry, no
passes) is the shared zero-pass point, so the launch intercept, the entry and the score DMA all
cancel in the slope.

| Probe A · pass kind | p1 | p2 | p4 | **µs/pass** | R² | ps/elem | × cvt | **× scat** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cvt` (monotone key, r+w) | 101.677 | 122.297 | 163.377 | **20.327** | 0.99985 | 1.939 | 1.000 | 0.188 |
| `scat` (dense L0 histogram) | 187.762 | 296.425 | 514.519 | **108.217** | 0.99995 | 10.320 | 5.324 | 1.000 |
| `cnt` (v10: compare + popcount, scalar carry) | 101.747 | 123.149 | 164.683 | **20.681** | 0.99978 | 1.972 | 1.017 | **0.191** |
| `cntv` (same, vector accumulator, no popcount) | 101.452 | 121.948 | 162.818 | **20.192** | 0.99981 | 1.926 | 0.993 | 0.187 |

| Probe B · density | realized | `B_ns` | `B_st` | **marginal** | Δ time % | × floor |
| --- | --- | --- | --- | --- | --- | --- |
| d004 | 0.0066 | 192.954 | 199.823 | **+6.869** | +3.560 | 5.36 |
| d05 | 0.0530 | 189.576 | 196.450 | **+6.874** | +3.626 | 5.36 |
| d15 | 0.1499 | 190.721 | 196.806 | **+6.085** | +3.190 | 4.75 |
| d30 | 0.3027 | 190.459 | 198.362 | **+7.903** | +4.149 | 6.16 |
| d45 | 0.4508 | 189.982 | 198.614 | **+8.632** | +4.544 | 6.73 |
| d65 | 0.6494 | 193.262 | 196.948 | **+3.686** | +1.907 | 2.87 |

`B_st_d30_p2 − B_ns_d30_p2 = +14.091` µs for two passes = 7.05 µs/pass against 7.903 at one pass —
the marginal is per-pass and linear, not a fixed per-call cost. `B_ns_d30 − A_scat_p1 = +2.697` µs
is the density mask + popcount + wrapped `woff` that `B_ns` carries on top of a bare histogram;
`B_st − B_ns` is therefore the **store alone**, which `probes/03`'s `emit1` (27.456 µs) never was —
`emit1` is a whole traversal plus a `cumsum` plus the store, and is **4.1×** this number.

Densities are set by `sel = (bits & 255) < t`, data-dependent so it cannot be folded, and the
realized fraction is measured host-side on the same array (`j-…txt:70-76`). `t = 0` is deliberately
not used: `x & 255 < 0` is provably false and a control the compiler can fold away is not a control.
The `valid` mask on every dense scatter is `key != 0x7FC00000`, a NaN pattern verified absent
(`never_bits_hits: 0`), so the histogram arm is **genuinely 100 % dense** — unlike
`instruments/sctopk_2stage_bench.py:851` whose `key > KEY_NEG_INF` on raw fp32 bits excludes every
negative score. Its 116.350 µs is also at `n=9216`, so the two are not comparable; **108.217 is the
in-pool L0 price and the ratios above are the quantity to carry.**

## The entry shape came out clean

`A_cvt_p1.optimized.hlo`, dumped in-job:

```
entry_computation_layout={(f32[1024,80,128]{2,1,0:T(8,128)})->s32[16]{0:T(128)}}
%bitcast_convert_type.1 = s32[1024,80,128]{2,1,0:T(8,128)} bitcast-convert(%s.1)
%bitcast.1 = s32[10485760]{0:T(1024)} bitcast(%bitcast_convert_type.1)
%p15_cvt_p1_t0_s0.3.cloned.1.call-start = … call-start(%bitcast.1, …)
```

`sparse-core-data-format-call: 0`, **no** `copy` on the score operand, **no** fusion on the 10485760
operand, and the flatten to 1-D is a bare **`bitcast`** — production's chain
(`sparsecore_topk.py:438`, `probes/11`). Minor dim 128 = one lane tile, so it is free.

The one materialized instruction is `bitcast-convert`, and it is **production's, not the harness's**
— `sparsecore_topk.py:438` emits the identical op inside its own jit. Measured directly by the
`A_cvt_p1#flatin` arm (identical work, pre-converted `s32[10485760]` entry): **25.743 µs**, against
`probes/11`'s independently measured open leg of **26.251 µs** — 1.9 % apart, from a different job.
It is a constant in `A_p0` as well, so **it cancels in every slope and every paired difference here**
and inflates no number in this README. Whether production can elide it is `probes/11`'s open leg,
not this probe's.

Structural gate on the one-line edit: `len(HLO)` for `B_st` minus `B_ns` is **+168 at five of six
densities and +172 at d30**, and never 0 — the store is present in every `st` arm, absent in every
`ns` arm, and the same edit at every density (the 4-byte outlier is the `t=77` literal's width).

## The five fields

- **probe:** does a bare count pass (`v10`) price cheaply enough against the dense L0 histogram it
  replaces, and what is the marginal cost of the one conditional `store_compressed` `v3` fuses into
  the L0 walk, as a function of store density?
- **config:** `j-78f7f537-1967-4804-8a76`, image `…vllm-torchtpu-hning_google_com:fa63d6df` with
  `git archive b11b6eb6 src` mapped over it — the image's own `sparsecore_topk.py` is `1a89d451…`
  (commit `f2f0be10`), **not** the `44353b34…` the design is written against, and the first
  submission `j-46676cf0-8a8a-4527-b871` failed its md5 gate on exactly that. b=1024, n=10240,
  k=2048, P=1, `slice_len` 10240, 32 waves, ρ=1 (`level_sim.py` STEP-4b `make_kv(10240,…)`, seed
  20260910, reproducing `probes/13/README.md:81`'s row-0 81.677 % to 3 d.p.). One pod, one command,
  no fan-out. Every arm `suspect=False`.
  ⚠ **Five arms report `hostbound_pct` above 5 %** — `A_scat_p1` 56.84, `B_st_d30#rep` 20.33,
  `B_st_d65` 17.43, `A_cntv_p1` 8.08, `B_st_d05` 6.15. That statistic is computed from `us`, the
  min-over-rounds paired slope (`glm_dsa_block_bench.py:388,392`), and those same five rows have an
  anomalously low `us` (`A_scat_p1` reads `us` 82.016 against `us_median` 187.762). It is one
  contaminated round being *selected* by the `min`, which is the failure mode that instrument's own
  docstring at `:394-402` describes — not a dispatch-bound arm. **Every number in this README is
  `us_median`**, the per-round median, and the check that it is sound is the ladder: `A_scat`'s four
  points fit a line at **R² 0.99995** with the p1 point 1.33 µs off it, inside the 1.282 µs floor.
- **job:** `j-78f7f537-1967-4804-8a76.txt` (verbatim), `probe15.json` (incl. the per-round visit
  order), `A_cvt_p1.optimized.hlo`, `A_cvt_p1_flatin.optimized.hlo`, `probe.py`, `stage/` as shipped.
- **answer:** count pass = **0.191 × the dense L0 histogram pass** and = a read-only walk to within
  the floor; fused L0 conditional store = **6.675 µs/pass, flat in density**, bounded 3.7–8.6 µs
  over 0.7 %–65 %.
- **decided:**
  1. **`v10`'s bimodality is closed onto the cheap branch and the root survives.** The
     verdict-blocking term (`design.md:1119`, `§10:1263`) is now `measured`; the ±band is no longer
     set by the count-pass rate. Central estimate **−9.7 % → ≈ −12.2 %**. The node stays **OPEN**
     (unbuilt), but what is now open is the **pass count**, not the pass rate — and that is a
     host-simulable question of the `probes/13` kind, not a pod one. It would take **8** correction
     passes to kill the root.
  2. **`v3`'s sign is settled: it cannot be a loss.** The probe `design.md:929` owed is paid, and it
     moved the number the right way — the cost side is **2.08× smaller** than the model charged.
     Band **−17.51 %…−11.94 % → ≈ −20.5 %…−14.9 %**, break-even `cnt_eff` 6295 → 8531 (above the
     6495 global max). `v3` remains the largest in-algorithm prize on the board and is still the
     node `v10` competes with — **and it just got larger while `v10` also got larger**, so the
     competition is not resolved by this probe.
  3. **20.3 µs is the walk floor at this geometry**, replicated to 1.3 % against `probes/03`'s `cvt`
     (20.589) from a different job. Any node whose move is "replace a pass with a cheaper pass"
     is bounded below by it — including `v10`'s count passes, which are already at it.
  4. **`probes/03`'s `scatter` arm is not 100 % dense** (`sctopk_2stage_bench.py:851` masks on raw
     fp32 bits under `key > KEY_NEG_INF`, which is false for every negative score, and `:241`
     generates standard normals). Not a retraction of the price sheet — `probes/07` vindicated it —
     but the L0 histogram price should be taken from this pool, not from that row.
  5. **The mandatory-probe gate of `§10` is satisfied for `v3` and `v10`.** `v5` remains blocked on
     `probes/16`.
