# Adversarial review of the corrected roofline gate

Run in a clean context via `/critical-review` against the raw artifact, not against
`measurement_and_design.md`'s prose. Every number below was recomputed from
`gs://cloud-devkit/jobs/j-8b1e7acc-ebdc-45d2-b46c/.../bench_prod.json` and
`.../fanout/shard1.log`, or from source, or is arithmetic on those.

**Verdict: PARTIALLY CONFIRMED.** The designer's three named corrections are real and both
corrected fractions reproduce exactly (39.86 % decode, 80.41 % prefill). **The prefill gate
decision does not survive.**

## Terms

- **`F_struct`** — the floor charging stage A and stage B as serial, as §1.6 builds it.
- **`F_common`** — the cross-option floor §1.6 declares should be used for reading fractions.
- **achieved fraction** — `floor / measured call time`. Higher means less headroom. The skill's
  gate is ≥80 % → "do not write a kernel", <50 % → "structural, a kernel usually pays".
- **`t_sc`** — the stage-B (SparseCore radix select) term of the floor, in µs.
- **charge/credit** — what fraction of a component's *measured* time its floor term claims is
  irreducible. 100 % credit means the floor says the component cannot improve at all.

## The four findings that change a decision

### F1 — the prefill gate is circular

`t_sc` prefill = **481.0 µs**, tagged `modelled`, and it is **69.8 % of the 689.3 µs floor**.
`bench_prod.json` gives `stage_b_ms = 0.48097` for the same row. `area5 §C.7` states the
1.7 ns/elem constant behind it was obtained as `481.0 µs ÷ 32 rows/subcore ÷ 8704 = 1.73 ns/elem`
— fitted to that measurement. `area5 §B`'s own arithmetic gives 473.5 µs, so the brief did not
even print its model's output; it printed the measurement and marked it `✓ measured`.

A floor whose largest term is the measurement is at 100 % of itself by construction. Strip it and
the non-tautological part is `(14.12 + 194.15) / (857.27 − 481.0) = 55.3 %`.

### F2 — the stage-B cost model has no predictive content

Applying `waves × eff_len × 1.7 ns` to all eight rows of `bench_prod.json`:

| row | waves | model µs | measured µs | ratio |
| --- | ---: | ---: | ---: | ---: |
| decode b8 ctx8704 | 1 | 14.8 | **90.30** | 6.10× |
| decode b32 ctx8704 | 1 | 14.8 | 63.63 | 4.30× |
| decode b64 ctx8192 | 2 | 27.9 | 66.56 | 2.39× |
| decode b64 ctx8704 | 2 | 29.6 | 67.40 | 2.28× |
| decode b64 ctx9216 | 2 | 31.3 | 64.19 | 2.05× |
| prefill ctx5120 | 32 | 278.5 | 338.71 | 1.22× |
| **prefill ctx9216** | 32 | 473.5 | **480.97** | **1.02×** |

It fits at exactly the one point it was calibrated on. And **b=8 (1 wave, 8 rows) costs 90.30 µs
against b=64 (2 waves, 64 rows) at 67.40 µs** — eight times less work costing 34 % more time. No
per-element model produces that. A candidate explanation is the ~42 µs `dispatch_floor_ms`
dominating small rows; that is testable and untested.

### F3 — decode and prefill are credited on opposite bases, and that asymmetry is the headline

- decode: `t_sc = 30.0` against a measured 67.40 → **44.5 % credit**
- prefill: `t_sc = 481.0` against a measured 480.97 → **100 % credit**

No consistent treatment reproduces the published pair:

| basis | decode | prefill |
| --- | ---: | ---: |
| both = measured stage B | 56.8 % | 80.4 % |
| both = `area5 §C.8` algorithm-independent floor (12 / 190 µs) | 31.7 % | **46.5 %** |
| both = `F_common`, the doc's own declared yardstick | 25.6 % | **56.1 %** |
| **as published** | **39.9 %** | **80.4 %** |

The published combination is the only one putting decode deep in the `<50 %` band *and* prefill at
the bar. Further, the 30.0 comes from subtracting "~35 µs host dispatch", and `report.md` §6.2
labels that column **"⚠️ RETRACTED, do not use this column"**. Applying the same −35 µs to prefill
gives 76.3 % — below the bar.

### F4 — the margin is smaller than every uncertainty in play

80.4 % clears by 0.4 points. Each perturbation below is independently defensible:

| perturbation | prefill |
| --- | ---: |
| the model's own 32×8704×**1.70** ns = 473.5 (not 1.73) | 79.5 % |
| denominator = 929.9 µs (`j-c13c8436`, **same shape, same code**) | 74.1 % |
| decode-consistent −35 µs dispatch | 76.3 % |
| VPU rate ×2 / ×0.7 (within its own modelling error) | 72.6 % / 87.1 % |
| `area5 §C.8` stage-B floor | 46.5 % |
| `F_common` | 56.1 % |

The second row is the cleanest: **the same configuration was measured twice and the two
measurements straddle the bar.** The prefill denominator is a best-of-N over `iters: {'total': 2}`
against a measured ±8.5 % spread. Even under full numerator/denominator correlation the fraction
still straddles 80 % (78.5 %–82.5 %).

## Corrections to specific claims

- **§6 item 2's provenance is backwards.** `references/05-compute-units.md:235-236` gives the
  1.72e12 VPU rate as **measured**, `[verified: j-ed517191-5f2d-4d00-8c0c, vpu_fma_rate, tpu7x]`
  — a dependent-FMA chain, hence latency-bound, which is why it is wrong *for this kernel*. It is
  not "a value nobody established". Also a unit error: 1.72e12 is flop/s at FMA=2, 7.782e12 is
  element-ops/s at 1; matched, the gap is ~9× not 4.5×. **The conclusion is right and provable
  more cleanly**: §4's prefill `t_vpu` of 586.0 µs exceeds the *measured* stage A of 335.07 µs, so
  it is falsified outright.
- **The 7.782e12 replacement is unvalidated.** Its stated cross-check is an algebraic identity —
  the "achieved rate" it divides was itself defined as `t × f × 4 × 50.97 % × 1024`, so the check
  returns 50.97 % for any input. Measurement bounds it at ≥3.10e12. Across 3.10e12–7.782e12 the
  prefill fraction moves **103.9 % → 80.4 %**: the highest-leverage, least-supported term in the
  gate. The 4 issue slots are inferred from counter *naming* and 1024 lanes is `assumed`.
- **§6 item 9's impossibility argument is a per-step/per-call unit error** — 42 = 2 × 21 layers is
  per-step, 221.4 µs is per-layer. Correct comparison 10.4/221.4 = 4.7 %. The conclusion survives
  on P4's direct measurement. One caveat: P4's *total* per empty launch is intercept + marginal
  ≈ 2.9 µs, not 0.04, so the supersession over AgentX's 5.3 µs is ~1.8×, not ~10×.
- **§6 item 3 is right but not novel** — `report.md` §6 and §8 both state it, and §8's own
  sensitivity row already put prefill at 100.8 % with the verdict "unresolved — do not proceed on
  this number".
- **F11: the baseline of record was itself a four-shard fan-out.** `cdk job recipe j-8b1e7acc`
  shows `TPU_TOPOLOGY: 2x2x1`, `google.com/tpu: "4"` and four `tpu-fanout.sh` commands. §1.2 says
  "**All of wave 2** ran this way", implying wave 1 did not. It did. The ±8.5 % spread is a
  property of the baseline too. Relatedly §1.1 calls it "in the compiled model, clean pass"; the
  recipe shows `bench_streamidx_topk.py` standalone with `ENABLE_XPROF: "0"`.

## Confirmed — tried and could not break

§6 items 1 (the 3.70e12/2.97e12 rate defect, 24.58 % optimistic), 4 (P1's MAC count: `geo_num_bkv:
5`, 10240 positions, masks applied post-einsum at `:344-353` so masked positions do cost MACs),
5 (the fill/peak cancellation — `(achieved/peak)/fill` contains no S), 6 (MXU prize 9.72 µs =
4.39 %, under the spread), 8 (P1's `mem_stall_mxu_idle` mislabel), 10, 11, 12. The 1.900 GHz clock
is self-consistent to 5 digits but recovers the *tool's* normalisation constant — label it
`derived from tool internals`. Correctness: `13 passed in 21.41s`, read directly from
`fanout/shard1.log`, covering the **incumbent** only.

## Disposition

1. **Do not stop prefill work on the 80.4 %.** Undecidable. Recording "prefill is at the bar" in
   `results.md` would close a phase on a tautology.
2. **Decode proceeds.** A + candidate 0. Restate the fraction as **26–57 %**, or fix `t_sc` to a
   measured 48.2 µs (→ 48.1 %) rather than the retracted 30.0.
3. **The cheapest thing that settles prefill** is the stage-B `k`-sweep `area5 §B` already
   specifies — hold `b`, `n`, `row_lengths`; sweep `k` = 2048 → 1024 → 512. One
   `run-command-in-cdk` shard, seconds. Discriminates issue-bound from output-DMA-bound and gives
   stage B its first non-fitted cost model. Until then 70 % of the prefill floor is unmodelled.
   *(Dispatched as `probes/stage-b-ksweep.md`.)*
4. **Option D is unevidenced.** The 44.8 µs is `init_ms`, at 1.07× the measured
   `dispatch_floor_ms = 0.042`, i.e. ~94 % host dispatch. Its second prop — "stage B reads 67.4
   in-model vs 48.22/52.12 isolated, so ~15–19 µs is plumbing" — compares two *isolated* standalone
   measurements from different jobs, which §1.2 forbids. A genuine residual exists by a different
   subtraction: `221.4 − 110.38 (P4) − ~50 (P2/P3) − ~1 (real init) ≈ 60 µs`.
5. **A clean review is not a correctness check.** Every option still needs its own named job.

## What the review could not check

Whether the VPU has 4 issue slots and 1024 f32 lanes on tpu7x (the largest unverified input to the
prefill gate); a first-principles SparseCore radix-select floor (`hardware-facts.md`: "No
SparseCore fact is measured"); the raw outputs of `j-e107f728`, `j-9e113359`, `j-170ed342`,
`j-a71a648c` — so **Option C's entire magnitude rests on a static histogram nobody has reopened**
(this is what `probes/R3.md` addresses); `area5 §C.8`'s 190 µs; `DEFAULT_VMEM_LIMIT_BYTES`;
whether the ±8.5 % spread is fair (it is n=2). Nothing here is an end-to-end prediction.
