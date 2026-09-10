# `results.md` — step 3: the candidates, built, proved and measured

⚠ **In progress.** Six candidates went out in one wave; this file fills in as they land. A variant
with no timing row below has not been measured, and its absence from the ranking is a schedule fact,
not a verdict. **The losers are reported, not deleted.**

## Numbers in this document

Every percentage is `(new − old)/old` of **time**. Negative is faster. A signed percentage appears
only in a column headed `Δ time %`.

**Denominators. ⛔ The ones every prediction in this exercise was written against are refuted.**
`probes/18-rho-denominator` (`j-97b98f97`, 47 arms, one pool, 12 reshuffled interleaved rounds) has
now measured what nothing before it had: the real kernel, at the production ρ and the production
entry. See "The denominator, remeasured" below. In short — the production number is **275.185 µs**,
not 155.82 (**+76.60%**); the **1.219 is not a factor at all** but a constant additive 35.563 µs; ρ
is **not a sufficient statistic**; and the ledger's own anchor does not reproduce.

**The consequence for every predicted percentage in this file.** They were all computed as a saving
in µs over 189.98 or 155.82. The µs are mostly still good — they were measured. The *percentages*
are not, and they cannot be repaired by rescaling, because the correct rescale is **term-dependent
by a factor of 2.5** (finding 2 below). So every predicted `Δ time %` below is now tagged
`modelled ⚠ refuted-denominator`: **read it as a rank hint and a mechanism, never as a magnitude.**
The measured ranking supersedes all of them and is the only thing that will be reported as a result.

**Evidence classes**, used per row: `measured` (a timing job, `us_median`, interleaved, above the
null floor) · `static-compile` (host AOT compile — real XLA-TPU + Mosaic SparseCore backend, but
compile-time analysis, never a substitute for a timing number) · `modelled` · `unmodelled`.

⛔ **No percentages are added up in this file.** Composition is measured in step 4.

## The build wave

| variant | node | plan | predicted production `Δ time %` | built | correctness | measured |
| --- | --- | --- | ---: | --- | --- | --- |
| `v10-sample-and-count` | `v10` | E | ⛔ ~~−14.9%~~ → **+11.99% h / +14.61% p, a LOSS** | ✅ | ✅ 68 + 30/30 bit-id | ⏳ **one arm only** (arity 2) |
| `v02-r1-shared-hist` | `v2` | A | ⚠ **µs, not %** — reweighting inconsistent | ✅ | ✅ 68 + 30/30 bit-id | ⏳ |
| `v12-f32-input` | `v12` | edge | ~~−16.8%~~ → **−9.54%** ✅ settled | ✅ | ✅ 68 + 30/30 bit-id | ⏳ |
| `v11-retiled-output-emit` | `v11` | edge | ≤ −6.42% ⚠ | ✅ | ✅ 68 + 30/30 bit-id | ⏳ |
| `v04-ragged` (rung 2) | `v4` | C | **−5.19%**, insensitive to `g` | ✅ | ✅ 68 + 30/30 bit-id | ⏳ |
| `v04` pipelined `tc=2048` (rung 3) | `v4` | C | **a wash — sign not adjudicable** | ✅ | ✅ 68 + 30/30 bit-id | ⏳ |
| ~~`v04` pipelined `tc=1024`~~ | `v4` | C | ⛔ **REFUTED, +14.1 µs vs ragged** | ✅ | dropped | dropped |
| ~~`v03-fused-emit`~~ | `v3` | B | ⛔ ~~−25.0%~~ → **+2.27% … +7.71%** | ✅ | ⛔ **46 FAILED — OUT** | dropped |
| the incumbent | `v1` | — | 0 by definition | n/a — **runs as a candidate**, same job, same instruments, same shapes | | |

---

## Correctness — run before any ranking number, and it removed a candidate

Three jobs, `probes/22-correctness/`: `j-188b0f9e-47cc-4908-9fae` (the four frozen variants),
`j-24463f36-f8ec-4c7b-a446` and `j-f114bd8a-dbaf-4964-b6a5` (the remaining ranked arms). Every arm's
subject md5 gated in-job; every arm's activation proven by the Pallas **scope name observed in the
lowered program**, aborting before the suite if absent.

| arm | frozen 68 | xgate exact | xgate bit-identical | verdict |
| --- | --- | --- | --- | --- |
| `v11` | 68 passed | 30/30 | **30/30** | ✅ |
| `v12` | 68 passed | 30/30 | **30/30** | ✅ bit-exact, now *verified* rather than assumed |
| `v2`, `V2_BIGSCRATCH` | 68 passed | 30/30 | **30/30** | ✅ |
| `v4-ragged`, `v4-c2048`, `v4-control` | 68 passed | 30/30 | 30/30 | ✅ |
| `v10-a2`, `v10-a16`, `v10-control` | 68 passed | 30/30 | 30/30 | ✅ |
| **`v3`** | ⛔ **46 failed, 22 passed** | **18/30 RED** | 12/30 | ⛔ **OUT** |

### ⛔ `v3` fails, and the mechanism is named — it is writing scores into the index array

Not the slot-order difference its gate was widened to allow. The out-of-range index `1098907648` is
`0x41800000` = float32 **`16.0`**, and the frozen `heavy_ties` is `rng.integers(0, 17)` — so that is
the **largest score in the distribution appearing in the index array**. The `-2147483648` filling
whole rows is `0x80000000` = `-0.0`. Four symptoms — out-of-range values, duplicate indices, `-1`
padding that is not a suffix, rows with zero valid entries — **one cause**. Red at both production
`P=1` shapes *and* at `P=2`; its 12 green cells are exactly the degenerate ones (`strict_desc`,
already sorted; `all_neg_inf`, nothing emitted). It was already predicted to lose at +2.27%…+7.71%,
so **it is not being repaired** — a named correctness mechanism is a more useful result than a slow
number. `v3` is the wave's second refuted node and its only correctness failure.

✅ **`v11`'s discriminating id passed** — `test_all_neg_inf_rows_return_no_selection[p1-b1024-n10240]`,
where every landed word is the known constant `-1` so a wrong stride shows as a wrong value rather
than a plausible one. Its `P>1` *correctness* is clean in every cell; the known `P>1` regression is a
relayout cost, not a wrong answer. ✅ **`v10` is bit-identical to the incumbent in all 30 cells
including `heavy_ties`** — its tie caveat is a possibility, not observed behaviour on these inputs.

### ⛔ Nothing in the frozen 68 ever compares a variant to the incumbent file

The two tests that look like they do — `:295` `test_emit_pairs_is_bit_identical_to_the_incumbent` and
`:351` `…survives_ragged_rows` — toggle `_reference_stage2_gather` **within the loaded module**, and
both are `TWO_STAGE_SHAPES`, so `P>1` only. **A green 68 establishes exactness and nothing about
bit-identity** — which is the gate three variants are quoted against. `probes/22/xgate.py` is what
that claim now rests on: two modules in one process, 3 shapes × 5 kinds × {full, ragged}, helpers
imported from the frozen suite, `P` asserted equal. What makes it evidence rather than a green run is
**a negative control that went red in every shard** while the incumbent never went red in any cell.

### ⛔ Three arms emitted the incumbent's scope name, so their activation check could not fail

`v2`, `V2_BIGSCRATCH` and `v11` all built `name=f"sc_topk_direct_b{b}_n{n}_k{k}{suffix}"` with
`suffix` carrying only `_keys` — the incumbent's bare name. All three carried the inherited comment
"The kernel names itself after the variant that was built, so a profile scope name is enough to tell
which path actually ran", **false in each**. This is not a profiling inconvenience: an activation
check against a bare name **passes for the incumbent too**, which is the precise failure the check
exists to prevent, and it was found by surveying the `name=` expression rather than the flag that
feeds it. Tagged `_v2r1` / `_v2bigscratch` / `_v11retile`; md5s now `3c041d95…`, `79f05d55…`,
`67a16569…`, one line each, no numeric path touched. `v4-base` and `v10`'s control stay **deliberately
bare** — they are the incumbent, and `v4-base`'s md5 `44353b34…` is what proves it.

✅ **`b=64` is real decode.** The ranking harness drives row lengths from `probes/18`'s `build_lengths`
(ρ = 0.850 exactly), not the frozen bench's `_row_lengths` — so cross-cutting item 3's 0.391 trap
applies to the *bench*, not to these cells. The ρ source is reported per cell.

---

## An exercise-level `code-style.md` finding, raised once here rather than per variant

`code-style.md` item 16 is a **blocking** class: hardware constants are read from the platform, not
hardcoded. `sparsecore_topk.py:94-95` hardcodes `num_subcores = 16` and `num_sc_cores = 2`, while the
launcher one function up at `:453-458` reads `sc.num_cores` / `sc.num_subcores` properly.

**This is pre-existing, it is inherited identically by every variant and by the incumbent, and it
therefore discriminates no arm.** It is recorded here, once, rather than as six identical `❌` rows:

- It does **not** block any step-3 variant, because a variant that fixed it would stop being a clean
  A/B against an incumbent that had not.
- It **does** belong in `final/` at step 4, where there is exactly one candidate and the file has one
  owner. Route: thread `sc` through as `functools.partial` kwargs.

---

## The denominator, remeasured — and the framework it refutes

`probes/18-rho-denominator/`, job `j-97b98f97-142a-47f2-a173`, `measured`. 47 arms, one pool, 12
reshuffled interleaved rounds, driving the frozen `measure()`. Resolution floor **3.218 µs (0.757%)**
prefill, **0.435 µs (0.827%)** decode — decode is 4× tighter than `instruments/README.md`'s ~3.4%.
All seven in-job gates green, including the subject md5 gate, which again earned its keep: the
image's installed copy read `1a89d451…`, so **the tag had been re-pushed and the pinned subject is
what ran.**

**The boxed question.** At ρ = 0.40005, b = 1024, on the 3-D production entry: **275.185 µs**,
against the sheet's 155.82 — **+76.60%**. Gate-corrected per-invocation, **291.075 µs** (+86.80%).

Four results, each of which changes how a variant must be measured rather than merely correcting a
number:

1. **The 1.219 is refuted as a factor and confirmed as a term.** The 2-D→3-D relayout measured at
   three ρ reads 35.654 / 35.459 / 35.576 µs — a spread of 0.195 µs, **sixteen times inside the
   floor**. It is a *constant additive* 35.563 µs (agreeing with `probes/11`'s 34.165 within the
   floor), not a proportional scale. **`T_production = T_harness − 35.563 µs`**, not `÷ 1.219`.
   Divided in, it is 1.129. Separately the ρ reweighting was wrong too: measured
   `T(1)/T(0.40005) = 1.5457` against the implied 1.7201.
2. **⛔ ρ is not a sufficient statistic, so no variant may be measured at ρ=1 and reweighted.** At a
   *fixed mean* of 4096 the cost spans 254.178 → 310.542 µs — **19.37% on distribution alone** — and
   **21.008 µs (8.26%)** of that is row *arrangement* at an **identical multiset**, because the
   stride-32 subcore mapping means a wave costs its slowest subcore. Every ranked arm must be
   measured at the production ρ *and* the production arrangement.
3. **The denominator splits, and the two halves reweight in opposite directions.**
   `T(ρ) = 147.925 + 281.588·ρ` for ρ ≥ 0.25, r² = 0.99830. At production ρ that is 147.925
   ρ-independent + 112.649 ρ-scaled + 14.611 distribution. A **ρ-independent** saving keeps 100% of
   its µs into a 35%-smaller denominator, so its percentage grows **×1.55**; an **in-walk** saving
   keeps 40%, so **×0.62**. **A 2.5× fork — classify a term before reweighting it, or do not
   reweight.** This is why the refuted predictions cannot simply be scaled. It also matters that
   `T(ρ)` is **two-piece**, with a knee at ρ = k/n = 0.2 where `keep_all` engages (slopes 184.4 vs
   281.6 µs/ρ): a two-point measurement would have missed it by 18 µs.
4. **⚠ `probes/09` is unsafe to quote, on two independent counts.** Its `B = 326.782 µs` anchor does
   not reproduce — reconstructed exactly it reads **453.463 µs, +38.77%**, while agreeing with
   `probes/01` to +0.71% (decode +1.76%). Two jobs agree and `probes/09` is the outlier, and it is
   the anchor of the entire Stage C″ ledger. It is not a jit-vs-eager artifact. **Independently**,
   all 12 arms under `probes/09-sweep-split/arms/` carry `REFINEMENT_LEVELS = ()`, so
   `hardware_map.md:493`'s row "V4 sweeps, **all levels** | 18.866" is labelled for a configuration
   its arms did not run — it can only be level 0. This was the `v2` builder's finding 9-1, raised
   from the source and confirmed here by direct check of the arm files.

**No repair of the ledger is being bought.** The ranking job measures every candidate directly, at
production ρ and arrangement, in one pool — which supersedes the ledger rather than fixing it. The
Stage C″ sheet stays in `hardware_map.md` with these defects marked, as the record of what step 2
believed.

⚠ **One open item, and it is worth up to 2× on `v12`'s headline.** Whether `v12`'s deleted `kLoop`
fusion sits *inside* the program the ρ arms time or upstream of it is unsettled. If inside, its
26.251 µs is **−9.54%**; the refuted sheet would have called the same deletion −16.85%. Being
settled now from the optimized HLO, statically, no pod time.

---

## ✅ The composed candidate, measured whole — `probes/24-final`, `j-4dd339df-0495-4d6c-9afc`

`v2` + `v4-c2048` + `v12` + a constants cleanup, merged into one kernel, measured in **one job**
against the incumbent at 4 slots. Same protocol as probe 23: one pool of 30 arm-slots, 12 interleaved
rounds reshuffled per round, paired REPS 20/60, `us_median`, both brackets, `--ctx 9216`, `k=2048`,
`n=10240`, `P=1`, both production ρ with production arrangement, ρ=1 retained as a control. Gates:
`MAP_GATE_OK` → `FINAL_ARMS_OK` → `MD5_GATE_OK` → `DRY_RUN_OK` → `ARM_CHECK_OK` → `SMOKE_OK` →
`CAPTURE_OK` → `RANKING_DONE` → `FINAL_DONE`, all green, no lockfile retry. Activation is proved by
the observed scope name **`sc_topk_direct_b1024_n10240_k2048_c2048_final`** (and `_b64_…_c2048_final`),
read from the `pl.kernel` `name=` kwarg, not from the env that was set.

**Headline — `P.prefillX | whole`, the production entry at ρ=0.400: `275.196 → 200.989 µs`,
`−74.206 µs`, RESOLVED FASTER** on all three estimators against a 0.200 µs floor.

| cell / bracket | incumbent median µs | composite µs | Δ µs (corrected) | floor µs | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **`P.prefillX` whole** | 275.196 | **200.989** | **−74.206** | 0.200 | ✅ **RESOLVED FASTER** |
| `P.prefillX` inside | 240.593 | 191.325 | −49.268 | 0.105 | ✅ RESOLVED FASTER |
| `D.decodeX` whole | 49.487 | 44.751 | −4.737 | 0.465 | ✅ RESOLVED FASTER |
| `D.decodeX` inside | 46.747 | 42.760 | −3.987 | 0.155 | ⚠ **NOT RESOLVED** — 2/3 estimators |
| `P.rho1` whole *(control)* | 425.511 | 341.134 | −84.377 | 1.419 | ✅ RESOLVED FASTER |
| `P.rho1` inside *(control)* | 391.927 | 333.834 | −58.093 | 3.362 | ✅ RESOLVED FASTER |

**µs, not percent-of-call — but not for the reason previously given here.** ⛔ The earlier claim that
the dispatch floor "sits in every denominator", so the same win quotes differently in the two jobs, is
**withdrawn: it is refuted by these two jobs' own baselines**, which reproduce to 0.044 µs (0.016%) on
`P.prefillX|whole` and 0.255 µs on `D.decodeX|whole` while the floor estimate moved 31.53 → 22.86. A
percentage *would* have travelled between exactly these two jobs. Three facts replace it:

- **The floor is not additive.** The harness launches `reps` calls and blocks once, so per-call host
  issue and device time combine as **`max(host, device)`**, not their sum, and a device-bound arm
  hides its issue cost entirely. `hostbound_pct` measures the exposed part directly: **1.16% prefill
  (3.22 µs) and 5.06% decode (2.58 µs)** — not 22.86. Were 22.86 µs of serial issue inside the decode
  call, `hostbound_pct` would read 46.1%.
- **The floor is a resolution bound, which is a real and useful thing**: you cannot measure a per-call
  time below ~23–35 µs with this harness. `probes/19` is the consistency check — its *fitted*
  intercept, 32.013 / 31.685 µs, lands on its floor of 31.19, because at zero descriptors the call has
  no device work left and falls back to the host issue rate.
- **The floor estimate is the least stable number in the pool.** Five job-level estimates of the same
  quantity: **22.86, 31.19, 31.53, 31.96, 35.18 µs** — a ±40% spread, while the incumbent kernel it is
  used to interpret reproduces across jobs to 0.016%. `dispatch_floor()` is a single unpooled,
  un-interleaved, unreplicated shot with no null arm: the one measurement in these jobs that satisfies
  none of the nine probe properties. **The measuring stick is 2500× less stable than the thing being
  measured**, so it can be quoted as a bound and never as a correction.

The conclusion is unchanged and rests on plainer ground: the denominator is **bracket-specific** (the
same win is 26.96% of `whole` and 20.48% of `inside`) and **cell-specific**, and the quantity that
converts to an e2e claim is absolute device time freed per call. `−74.206 µs` travels; `−26.96%` needs
its bracket, its cell and its baseline attached to mean anything.

**⛔ No reconciliation against the sum of the parts, and this is not a formality.** The composer found
`v2` and `v4` were **not disjoint** — four mechanical collisions plus a fifth that `diff` reported as
*clean* because the added text matched `v4`'s relocated copy as context. Overlapping arms do not have
additive deltas, so a sum is not a prediction the composite can be short of. Probe 23's per-arm
numbers are printed beside each cell in the job output, labelled *a different job, orientation only*,
and nothing is subtracted from them.

**The two brackets settle `v12`'s open item — within this one job, with no cross-job arithmetic.**
Whole −74.206 µs against inside −49.268 µs: the whole-call delta **materially exceeds** the
inside-kernel delta, by 24.9 µs. The pre-registered reading for that observation was
"`v12`'s entry-glue deletion **survived** composition", and it is the one thing the composite's own
numbers can say about which parts made it through. That 24.9 µs also happens to land on `v12`'s
static bytes model (≈24.9 µs) and its standalone −25.15 µs — a cross-check, *not* the source of the
claim, which is the within-job bracket difference alone.

⚠ **`D.decodeX | inside` is NOT RESOLVED, and the decomposition says more than the flag.** All three
point estimates agree — −3.987 / −4.257 / −4.044 µs — and two clear their floors by 2469% and 528%.
The single failure is the **pooled slope**, whose floor blows out to 5.062 µs because one incumbent
slot, `#null_c`, read `us=40.900` against its own `us_median=46.663` and `us_unpaired=46.298`, with
**hostbound 15.0% where every other slot in that cell is ~5%**. A host-bound excursion corrupted that
slot's pooled slope; the range-based floor absorbed the whole excursion and swallowed a real effect.
The pre-registered rule is not being reached back and changed — the cell stands as NOT RESOLVED — but
this is one contaminated slot, not three estimators disagreeing about an effect. It is also exactly
why `us_median` was the pre-registered read. `D.decodeX | whole` is resolved, so decode is not
unmeasured; it is just not the headline.

**Decode is not the headline, for a reason that predates the number.** sctopk is ~1.2% of decode
device time, so even the resolved −4.737 µs whole-call decode win is worth at most ~0.08% of the
decode step. Prefill is the prize.

**The one-signed null correction fired on 3 of 6 cells and changed nothing.** `P.prefillX` both
brackets and `P.rho1 | inside` came out one-signed, so the corrected delta governed there; the largest
correction was 1.563 µs against a 58 µs effect. The correction earns its keep on marginal calls, which
is precisely what this job did not produce — **the `[MARGINAL]` flag did not fire in any cell.**

**Correctness, run by the composer before any of this**: `j-49ee915f-6906-43dc-ba3f`, 68/0/0/0 on both
shards, xgate **30/30 bit-identical**, negative control **RED** at max score difference 4.8411946, md5
gated twice. This job's own STAGE 2 smoke independently re-confirms it: all 15 arm×cell combinations
`rows_wrong=0 oor=0 live_bad=0 bit==base=True EXACT`. A bit-identical output also closes every
value-mediated mechanism for the speedup.

⚠ **One inherited defect, deliberately unpatched.** `_pick_chunk` has a small-chunk cliff: 6,208 of
49,608 reachable `(b, n, k)` combinations — 12.5% — land under 128 words. It comes from `v4-c2048`, it
is unreached by any production or frozen-suite shape, and patching it now would invalidate the md5
that is gated inside this job and inside the correctness job. Routed as a follow-up.

**VMEM +1024 B**, agreed by two independent methods, **zero spills**, 14.0% of budget — `v2`'s
−15,360 B very nearly pays for `v4`'s +16,384 B double buffer. Tagged `static-compile`.

---

## The measured ranking — `probes/23-ranking`, `j-66c7e331-fa41-461e-8686`

**This supersedes every modelled `Δ time %` below.** One pod, one process, **one pool of 58
arm-slots**, 12 interleaved rounds reshuffled per round, paired REPS 20/60, read `us_median`,
`--ctx 9216`, `k=2048`, `n=10240`, `P=1` for all 11 modules. **Three null arms** are the incumbent
source loaded as separate module objects, so the incumbent holds **4 slots in every cell** and the
floor is measured. Two brackets per arm: `whole` = the production 3-D entry `f32[b, n/128, 128]`,
`inside` = the `pl.kernel(...)` call only, its operands captured from the variant's own eager
`_sc_topk_direct` so no variant's entry transform is re-implemented. Gates fired in order:
`MAP_GATE_OK` → `MD5_GATE_OK` → `DRY_RUN_OK` → `ARM_CHECK_OK` → `SMOKE_OK` → `CAPTURE_OK` →
`RANKING_DONE`; no arm lost its inside-kernel bracket. Full detail and the raw tables:
`probes/23-ranking/README.md`.

### Prefill — b=1024, ρ=0.40005 exactly, production arrangement, floor 0.040 / 0.433 µs

| arm | whole µs | Δ whole | inside µs | Δ inside | where the win lives |
| --- | --- | --- | --- | --- | --- |
| incumbent | 275.213 | — | 240.659 | — | — |
| `v2` | **238.161** | **−37.05** | **203.259** | **−37.40** | **in-kernel, all of it** |
| `V2_BIGSCRATCH` | 238.365 | −36.85 | 203.748 | −36.91 | in-kernel; **= `v2`** |
| `v12` | 250.064 | −25.15 | 240.578 | −0.08 | **entry glue only** ⚠ |
| `v4-c2048` | 263.624 | −11.59 | 229.470 | −11.19 | in-kernel |
| `v11` | 265.767 | −9.45 | 240.579 | −0.08 | **entry glue only** ⚠ |
| `v4-ragged` | 268.125 | −7.09 | 234.494 | −6.17 | in-kernel |
| `v10-a2` | 396.817 | **+121.60** | 360.705 | **+120.05** | ⛔ regression |

### Decode — b=64, ρ=0.850, production arrangement, floor 0.730 / 0.211 µs

| arm | whole µs | Δ whole | inside µs | Δ inside | verdict |
| --- | --- | --- | --- | --- | --- |
| incumbent | 49.302 | — | 46.836 | — | — |
| `v2` | **46.037** | **−3.27** | **42.769** | **−4.07** | **win** |
| `V2_BIGSCRATCH` | 46.539 | −2.76 | 43.043 | −3.79 | win; = `v2` |
| `v11` | 48.090 | −1.21 | 47.104 | +0.27 | whole-only; magnitude estimator-unstable |
| `v12` | 48.408 | −0.89 | 46.584 | −0.25 | whole-only, marginal |
| `v4-c2048` | 49.034 | −0.27 | 46.502 | −0.33 | **null** |
| `v4-ragged` | 49.697 | +0.40 | 46.607 | −0.23 | **null** |
| `v10-a2` | 71.545 | +22.24 | 68.268 | +21.43 | ⛔ regression |

⚠ **The b=64 cell is real production decode at ρ=0.850, NOT the frozen bench's ρ=0.391 small-batch
prefill occupancy.** Every cell's `row_lengths` come from `p18_probe.build_lengths` (the probes/13
`level_sim`, transcribed verbatim).

### ρ=1 — the predicted null, observed

`v4-ragged` is **inside the floor at ρ=1 in both brackets** (−0.73 µs vs floor 1.96; −0.03 µs vs
floor 1.90) while `v2` returns −46.61 / −47.89 µs in the same pool. A ragged-row optimization doing
nothing when there are no ragged rows is a live positive control on the cell design.

### ⚠ `v11` and `v12` win entirely outside the kernel — and it settles the open `v12` item

At prefill `v12` is −25.15 µs whole-call and **−0.08 µs** inside; `v11` is −9.45 µs whole-call and
**−0.08 µs** inside. Both are below the 0.433 µs inside-kernel floor. Their prefill wins are **100%
XLA glue between the 3-D entry and the `pl.kernel` call.**

- **The open item above is resolved: `v12`'s deleted `kLoop` fusion is *upstream* of the kernel, not
  inside it.** Measured −25.149 µs = **−9.14%** of 275.213, against the predicted −9.54%. The −16.85%
  reading stays refuted.
- **`v11`'s inside-kernel null was not predicted.** `_v11retile` changes the emit-side layout and was
  expected to move the kernel; it does not move it at all at prefill.
- ⚠ **This is the shape of a known trap** — a whole-call win that is an XLA `copy` production never
  pays. Mitigating: the bracket *is* the production 3-D entry. Not proof. Confirm the production
  call-site operand layout in an in-situ profile before adopting either.
- ⚠ **`v11` and `v12` compete for the same ~25 µs, they do not stack.** Both delete entry-side
  conversion work and neither touches the kernel.

### ⛔ Four printed verdicts are noise — the estimators disagree

`us_median` is the pre-registered read; the harness also stores the pooled slope and the unpaired
estimate. In the **decode inside-kernel** cell the three nulls come out **one-signed** (−0.205,
−0.211, −0.103 µs), so the symmetric floor understates the bias by about its own width:

| call | `us_median` | pooled `us` | `us_unpaired` | printed | **corrected** |
| --- | --- | --- | --- | --- | --- |
| `v11` decode inside | +0.27 | −0.30 | −0.03 | SLOWER | **null** |
| `v12` decode inside | −0.25 | −0.29 | +0.00 | FASTER | **null** |
| `v4-ragged` decode inside | −0.23 | −0.59 | +0.22 | FASTER | **null** |
| `v4-c2048` decode inside | −0.33 | −1.41 | −0.22 | FASTER | **null** |

**Nothing except `v2`, `V2_BIGSCRATCH` and `v10-a2` is resolvable in that cell.** Every other row in
this section survives all three estimators unchanged.

⬆ **Applied uniformly, the correction also promotes one row.** Formalized as
`probes/24-final/stage/final_rank.py` STAGE 8 — measure the delta against the **median** of the four
incumbent slots, keep the range as the floor — and replayed over `probes/23-ranking/ranking.json`, it
reproduces all four flips and **promotes `v4-c2048` at `D.decodeX|whole`** from "not resolvable"
(−0.267 µs vs a 0.730 µs floor, from a `base` slot 0.467 µs below its own median) to a resolved
−0.734 / −1.044 / −1.250 µs. It strengthens a candidate already adopted on prefill evidence rather
than creating one. A correction that only ever removes wins is one that was applied only to the rows
you already doubted; this one was applied to every row.

⚠ **The promotion is marginal — report it with the margin attached.** Its weakest estimator, −0.734
µs, clears the 0.730 µs floor by **0.0045 µs — 0.61% of the floor**. Resolved by the pre-registered
rule, and the rule is not being changed to suit it, but it is not the same kind of fact as `c2048`'s
prefill win (−4.500 µs against a 0.040 µs floor, ~two orders of magnitude). It changes no decision.

**The decomposition says more than the flag.** One estimator on the line at +0.61% with the other two clearing by **98.2%** and **80.7%** is a different object from three estimators clustered at the boundary: it says the call is probably **real** and that `us_median` is the conservative estimator here, not that the effect is borderline. Same flag, opposite readings — so the flag is always printed with its spread.

### `V2_BIGSCRATCH` buys nothing over `v2`

In all six cells the `v2` − `v2big` difference is inside the floor or **inverts sign** between
`us_median` and the pooled slope (prefill inside: −0.49 vs **+1.36**; decode inside: −0.27 vs
**+0.73**). **The larger scratch allocation is not the binding resource — ship `v2`.**

### The three pre-registered readings

**(a) `tc=2048` vs `ragged`** — the rule was "not adjudicable; a tie ships `ragged`". Half falsified:

| cell / bracket | `v4-c2048` − `v4-ragged` | floor | reading |
| --- | --- | --- | --- |
| prefill whole | −4.500 µs | 0.040 | **c2048 faster, resolvable** |
| prefill inside | −5.024 µs | 0.433 | **c2048 faster, resolvable** |
| decode whole | −0.663 µs | 0.730 | **TIE → ships `ragged`** |
| decode inside | −0.105 µs | 0.211 | **TIE → ships `ragged`** |

The pre-registered disposition governs decode, and only decode. At prefill the pair *is* adjudicable
and `c2048` wins by 4.5–5.0 µs, stable across all three estimators — so `c2048` **dominates**
`ragged`: strictly better at prefill, tied at decode.

**(b) `v2` in absolute µs, no percentage** (its two reweightings are mutually inconsistent) — the µs
are the prefill/decode/ρ=1 tables above.

**(c) ⛔ Do not add up the adopted candidates' percentages.** The arms overlap — `v11` and `v12` are
the worked example — and composition is step 4's job.

### Two things that will otherwise be misread as e2e

1. ⛔ **RETRACTED — the dispatch floor is NOT inside every absolute, and the ~2.8× that followed
   from it is void.** Retracted 2026-09-10; the argument and its data are in the composite section
   above. The per-call cost combines as **`max(host, device)`, not `host + device`**, so a
   device-bound arm hides its host issue. This job's own `hostbound_pct` refutes the claim: if
   31.53 µs of the 49.30 µs decode baseline were dispatch, `D.decodeX|whole` would read **64%
   host-bound**; it read **5.45%**, an exposed intercept of **2.80 µs**. **"Quote the µs delta, not
   the percentage of the call" survives** — see item 2, and the fact that the denominator is
   bracket- and cell-specific.
2. **These are percentages of the top-k call, not of a step.** sctopk is ~1.2% of decode device time,
   so `v2`'s −3.27 µs decode win is worth **~0.08% of the decode step at most**. Prefill is the prize.

### Cross-job consistency — the whole-call bracket reproduces `probes/18`

| | probe 23 | carried denominator | agreement |
| --- | --- | --- | --- |
| prefill, ρ=0.40005, whole | 275.213 µs | 275.185 µs | **+0.010%** |
| decode, ρ=0.850, whole | 49.302 µs | 49.171 µs | +0.27% |
| prefill, ρ=1, whole | 425.971 µs | 425.366 µs | +0.14% |

Three baselines within 0.3% of a different job on a different day. ⛔ 291.075 µs is a *correction*,
not a measured baseline; ⛔ 155.82 µs is modelled; ⛔ probes/09-derived numbers stay unsafe.

All 11 modules × 3 cells also passed the in-job smoke — `rows_wrong=0`, `oor=0`, `live_bad=0`,
`bit==base=True` in all 31 cells — which closes every value-mediated mechanism for these deltas.

---

## `v11` — retiled output emit

**Built. Correctness pending. Not yet measured.** Folder: `variants/v11-retiled-output-emit/`
(subject md5 `3dc249276fa68985e29f4da8e9c27fbd`; `variant.diff`, `aot_v11.*`, `shape_matrix.*`,
`mosaic_dma.*`, `hlo/`).

**What it is.** `_sc_topk_direct` declares its Pallas output as `(b, k)` instead of flat `(b*k,)`,
writes each row through `out_hbm.at[row_c]`, and returns it unreshaped. Pallas pins only
minor-to-major order for a TPU custom call, so a rank-2 `i32` result gets the `T(8,128)` tiling the
consumer already wants and the retile between call and consumer has nothing left to do. Three
functional edits at four sites; public signature and every downstream shape unchanged.

**Prediction, to be scored.** ≤ **−5.27%** harness / ≤ **−6.42%** production, at `b=1024`, `P=1`.
Both are **ceilings** and both are the design's own number — 10.007 µs `measured` (`j-9b520877`) over
each denominator, less a strided-store cost that is `unmodelled`.

### The dispatch's central question, answered

The node was on the shortlist with a live caveat: the 10.007 µs was measured on a `ROOT reshape` that
**does not exist in the production entry**, where `probes/14` Q2 had found an `S(1)` staging copy
instead, and no arm had ever deleted *that*. The build closes it, `static-compile`, on the real
`_streamidx_topk_jax` entry:

| | incumbent | `v11` |
| --- | --- | --- |
| ENTRY instructions | 220 | **218** |
| `%reshape.1 … S(1)`, `used_scoped` 2 MiB | present | **absent** |
| `%copy-start.5` / `%copy-done.5`, `S(1)`→HBM, 8 MiB | present | **absent** |
| `s32[2097152]` results | 3 | **0** |
| the `cond`'s 8 MiB result buffer | VMEM allocation 224 | **HBM live-out, allocation 3** |
| `temp_size_in_bytes` | 152,663,264 | 152,630,528 |

The mechanism is in buffer assignment, not the op list: with a flat kernel result the `cond`'s 8 MiB
`T(8,128)` buffer could not alias the call output, so XLA put it in VMEM, had each branch materialise
into it, and DMA'd 8 MiB out. With a 2-D result the `cond` buffer **is** the HBM live-out. Post-kernel
HBM traffic on the kernel branch goes from 8 MiB read + 8 MiB written to **zero** — the same 16 MiB
the standalone `ROOT reshape` moved, which is what 10.007 µs priced. **The ceiling transfers by
bytes, not by op identity.**

### What the build changed about the node

- ⛔ **`v11` is `P=1`-only.** At all four `P>1` shapes it *relocates* the relayout to the
  stage-1/stage-2 seam and ENTRY grows: at `p2-b16-n10240` the incumbent's free
  `s32[65536] → s32[16,4096]` fold is a bitcast, but under `T(8,128)` the same regrouping is not, and
  XLA inserts a physical copy. Both production shapes are `P=1`, so the candidate survives whole —
  **but `b=16` must not be pooled with `b=64`/`b=1024` in any arm summary.** This is exactly the
  warning in `BRIEF.md` §4 landing as written, and it is now a standing note in `design.md` §9a.
- ✅ Feasibility is closed at **all 7** frozen-test shape ids, `emit_keys=True` included.
- ⚠ A saving lands on the `_fast_path` branch too — the deleted `S(1)` staging removes an 8 MiB VMEM
  round trip on steps where the runtime gate **skips the kernel entirely**. Real and visible in buffer
  assignment, never priced, and **credited at zero** because a term that cannot fail is not evidence.

### `Δ` against `design.md`'s arithmetic

`design.md` said "the same 40 MiB of movement" at four sites. The retiled buffer is
`s32[1024,2048]` = **8 MiB**; 40 MiB is the `f32[1024,10240]` *input*, which is `v12`'s prize. The
design's own rate agrees — 10.007 µs ÷ 1.251 µs/MiB = 8.0 MiB. **No prize moves**, because every one
of them was computed from the rate and never from the byte count. Corrected in `design.md` at all
four sites.

### Open terms

| term | status |
| --- | --- |
| **strided-store descriptor count** | ⛔ **`unmodelled`, and it is the term that could eat the whole prize.** The store *is* strided: 16 granule-aligned 512 B runs at 4096 B stride replace one contiguous 8192 B run, so runs go 1,024 → 16,384 at prefill. Bytes per row are unchanged and no granule is partial. What no static artifact settles is whether the SC DMA engine issues **one strided descriptor or sixteen**. The only evidence is +2 SC sequencer bundles with TEC and TAC byte-identical — consistent with one enqueue that gained a stride field, inconsistent with a 16-iteration expansion. Suggestive, not decisive |
| **correctness under the new store pattern** | **unchecked — blocks the variant.** A wrong stride lands correct values at wrong offsets, silently. Discharged by the frozen test; `test_all_neg_inf_rows_return_no_selection[p1-b1024-n10240]` is the discriminating id, because every landed word is the known constant `-1` |
| **`P>1` regression size** | `unmodelled` — direction and mechanism established from ENTRY counts, magnitude not |
| **the denominator** | inherited, `modelled`, pending `probes/18` |

### Carried — not this candidate, and each with a route

1. **A Pallas TPU custom call's output *rank* is a layout lever.** Pallas pins only minor-to-major
   order, so declaring `out_type` at the consumer's rank lets XLA hand the call the consumer's
   *tiling* and the relayout never exists. A `references/04-layout-precision.md` promotion, in that
   file's "Arranging so the relayout never exists" family — and finding 2 above is equally a member
   of its "the ones that did not survive" family, since the identical move at `P>1` merely relocated
   the copy. Route: propose at step 4.
2. **`XLA_FLAGS=--xla_dump_to=<dir>` on a host-only AOT compile** yields `*-buffer-assignment.txt`,
   `*-memory-usage-report.txt` and `*sparse_core_specific_metadata.txt`, where VMEM-vs-HBM placement
   of an `S(1)` value and **SparseCore sequencer bundle counts** are legible — neither is in
   `compiled.as_text()`. Zero cluster cost. Route: `references/08-sparsecore.md`.
3. **`/tmp/libtpu_lockfile` serialises host AOT compiles across concurrent agents** and kills them
   with `ABORTED: Internal error when accessing libtpu multi-process lockfile`. Route: a
   retry-with-backoff loop belongs in the variant brief's AOT row. Already forwarded to the wave.

### Two instrument defects found and fixed during the build — both would have answered confidently and wrongly

- **`jax.jit` caches the inner pjit jaxpr on `(function object, avals)`.** `streamindex_topk.py:25`
  binds `sparsecore_topk` at import, so an in-model A/B done by rebinding the attribute let arm 2
  reuse arm 1's trace: the first run returned **byte-identical HLO for both arms**, which reads as a
  clean null. Fixed with `sys.modules` seeding + `importlib.reload` + `jax.clear_caches()` + an
  identity assertion, the variant run first so a residual bug shows as base-looking-like-variant; and
  every number was reproduced in two fresh single-arm processes, which cannot have the bug.
- **The first AOT negative control was refused by the wrong layer.** `o[...] = jnp.zeros_like(o)`
  fails with `NotImplementedError: Swap does not support storing to HBM` — the MLIR layer, not the
  allocator, so it proved nothing about SparseCore memory. Replaced with an 8 MiB VMEM scratch DMA'd
  to HBM, which fails as required with `E3000: CompileTimeSparseCoreAllocationFailure … memory space
  spmem`.

Both were forwarded to every other builder in the wave.

---

## `v12` — f32 input

**Built. Correctness pending. Not yet measured.** Folder: `variants/v12-f32-input/` (subject md5
`408ef72e9afc935c928f2803764878cc`). The kernel names itself `…_f32in`, so a profile proves it fired.

**What it is.** The kernel takes `keys` as f32 and bitcasts in-body, deleting the caller-side
`convert` that materialised a 40 MiB `f32[1024,10240]` temp. The prize is entirely **outside** the
kernel — a TensorCore `kLoop` fusion.

**Prediction, ✅ now settled and rescored.** ~~−13.82% harness / −16.8% production~~ →
**−9.54%** against the measured `P3.prefillX` 275.185 µs, **−9.02%** gate-corrected. The old figure
was nearly double, and it was the denominator that was wrong, not the µs.

✅ **The bracket question is closed: the deleted op is INSIDE the timed program, at every shape.**
`probes/20-v12-bracket`, `static-compile`, zero cluster time — an AOT recompile of `probes/18`'s
exact `P3` program against the pinned subject:

| cell | incumbent priced op | `v12` | Δ temp bytes |
| --- | --- | --- | ---: |
| 3-D b=1024 (`P3.prefillX`) | `bitcast-convert`, 66712 cyc, 16 MiB scoped | none | **−41,943,040 = exactly `b·n·4`** |
| 2-D b=1024 | `fusion` `kind=kLoop`, 66712 cyc | none | −33,554,432 |
| 3-D b=64 (`D3.decodeX`) | `bitcast-convert`, 5318 cyc | none | **−2,621,440 = exactly `b·n·4`** |

The Δtemp column comes from `memory_analysis()` and is independent of instruction counting: the
deleted buffer is the score array **to the byte**, at both batch sizes. Materialization was judged by
`window_config.estimated_cycles > 0`, not by name — the free `%bitcast.1` never has any.

✅ **The ρ fork validates itself here, independently.** `v12` is dense and ρ-independent, so it must
take the ×1.55 branch — and it does: `−9.54% / −6.17% = 1.546` against `probes/18`'s separately
derived 1.5457. Two independent derivations of the same reweighting factor.

**The opcode correction cuts both ways, and both halves are worth keeping.** In the 2-D dumps the
builder produced, the deleted opcode is `fusion` and there is no `bitcast-convert` opcode in either
ENTRY — the builder's self-correction was right. At the **3-D production entry** the opcode genuinely
*is* `bitcast-convert` and genuinely goes 1 → 0. The original sentence was wrong for the layout it
dumped and right for the layout it did not.

**Decode is `modelled`, not measured:** 1.56–2.09 µs by cycle-ratio and HBM-roofline bounds, =
**−3.2% to −4.3%** on `D3.decodeX` 49.171. The same roofline check on prefill gives 24.9 µs against
the priced 26.251 — 5% apart, so the prefill price is not a cost-model artifact.

**The relayout term stays disjoint and additive.** The 3-D ENTRY has no
`sparse-core-data-format-call`; the 2-D one does. That call is `probes/18`'s measured
`P2 − P3 = 35.576 µs` — **a different op** from the one `v12` deletes. Both `probes/18` statements
survive.

**Two free activation tells at production rank:** `_f32in` appears in the 3-D scope name, and
`temp_size_in_bytes` differs by exactly `b·n·4`.

**What the build closed.** The seven in-body bitcasts price at **exactly 0** — 1364 SparseCore
bundles in both arms, `static-compile`. The 40 MiB is a genuine **HBM** deletion, not a relocation:
`*-memory-usage-report.txt` gives HBM 128.07 → 96.07 MiB with the XLA-level `vmem` total unchanged
to the byte. The mechanism it also supplied, which arithmetic alone had not: the incumbent's temp
arena held **two** 40 MiB buffers with the 8 MiB output co-resident in the second; the variant
deletes one and the output takes its own slot, 80 → 48 MiB. `*sparse_core_specific_metadata.txt`
corroborates the bundle counts from a different part of the compiler (TEC 0–1157, SCS 60, identical).

**Self-caught error, recorded because the method generalises.** The builder first reported
"`bitcast-convert` in ENTRY 1 → 0". That was a **text** match on a fusion's *name*
(`%bitcast-convert_bitcast_fusion`); the incumbent ENTRY contains no such opcode at all — it lives
inside `%fused_computation`. Corrected to `fusion(` 1 → 0, `bitcast(` 1 → 2. **Match opcodes, not
substrings.** Same loose pattern produced a "bcvt" column now flagged as text-not-opcode; nothing
rests on it.

**Open terms.** The bracket question above, which is the whole magnitude. Correctness under an f32
input signature. The denominator, inherited.

**Carried.** `v12b` — declare `keys_hbm` f32 at the call site too, which clears `v12`'s only
`happened` risk row.

---

## `v4` — pipelined DMA

**Built as three separate files. Correctness pending. Not measured.** Folder:
`variants/v04-pipelined-dma/`. ✅ **The import-time env switch is gone** — it made the ladder a
cross-pool subtraction, which `probes/18` established never reconciles. Each rung is now its own
subject file, so the ladder is an in-pool difference between arms measured in the same rounds:

| rung | file | md5 | scope |
| --- | --- | --- | --- |
| control | `sparsecore_topk_base.py` | `44353b347c96e0eedbd742c710d4540a` | `sc_topk_direct_b{b}_n{n}_k{k}` |
| ragged blocking | `sparsecore_topk_ragged.py` | `69c8834e130e23999e0284929230da3a` | `…_v4ragged` |
| ragged + pipelined `tc=2048` | `sparsecore_topk_c2048.py` | `88a67892e8c063b5d9fea96de72e99c4` | `…_v4c2048` |

The control's md5 is **the incumbent's**, which is the check that the split did not perturb the
baseline. **The split is proven a pure refactor, not asserted**: recompiling every arm in fresh
interpreters reproduced **5 of 5 pre-split ENTRY dumps byte-for-byte**, 6 of 6 post-split arms
distinct, bundle counts identical. Zero `os.environ` occurrences across all three files, and the
env-switched original is deleted so nothing can map the wrong subject. `checksum.py` additionally
fails if two arms resolve to the same file.

### The ranking moved when the last free parameter was closed

`f` and `nc` are now computed on the host from the exact frozen `_row_lengths("random", …)`
distribution the harness runs, rather than assumed:

| arm | net vs incumbent | Δ harness | vs ragged |
| --- | ---: | ---: | ---: |
| **ragged** | +8.081 µs | **−4.25%** (−5.19% prod) | — |
| `tc=1024` | −5.980 µs | **+3.15%** | +7.40% |
| **`tc=2048`** | +8.818 µs | −4.64% | **−0.39%** |

**`tc=1024` stays refuted, and my own figure for it was wrong.** I quoted +27.82 µs, which is the
*extras-only* form (112 × 248.4) and is inconsistent with the 240.5 ns break-even I quoted beside
it — that one is the *all-descriptors* form. The two differ by exactly `32 × 269.1 ns = 8.6 µs`. The
internally consistent number is **+14.1 µs vs ragged**. Verdict unchanged, magnitude corrected.

**`tc=2048` is still a wash but its sign flipped** — with exact `f`/`nc` it is 0.737 µs *ahead* of
ragged rather than 0.44 behind. At −0.39% against a 0.72% null spread it is **1.8× below the
instrument floor: not adjudicable by a single A/B**, and the pre-registered rule is that a tie ships
`ragged`.

⛔ **The binding open term is now that the model does not reconcile at the incumbent's own operating
point.** `32 × g_raw = 16.560 µs` exceeds `X = 13.469 µs` — the entire measured score read — by 23%,
while nearly reconciling at b=64 (2%). `g_raw` does not transfer as a single constant, and the 3.7%
margin at `tc=2048` sits inside that transfer error; recalibrating to reconcile puts `tc=2048` within
**0.01 µs of dead level** with ragged. `tc=1024`'s refutation survives both calibrations. **The wash
is robust; its sign is not.**

`tc=2560` tops the model at −6.03% and the builder **declined to build it**, with the right reason:
its descriptor delta goes negative, so the model is claiming free money from `g_pipe < g_raw`, and
the gradient must terminate at `tc = n` where the arm degenerates to the incumbent.

**This is the candidate the original request named** — replace the synchronous blocking DMA wait with
`emit_pipeline` double buffering.

**Prediction, to be scored.** −6.38% harness / **−7.78% production** at `tc = 1024`, `modelled`
⚠ `refuted-denominator`, off the measured 13.469 µs ceiling. Decode is **`not resolvable`** (−1.77%
against a 3.37% null spread) and asymmetric — a `g`-driven *loss* there would be resolvable even
though the win is not.

⛔ **The verdict hinges on one unmeasured quantity, and `v4` decomposes into two very unequal
halves.** The builder's own decomposition, which is the most consequential thing this build produced:

| half | µs | `Δ time %` production | needs |
| --- | ---: | ---: | --- |
| make the *blocking* DMA ragged — one expression, `align(i_hi, LANES)` for `slice_len`; one descriptor, no pipeline, no buffer, no VMEM | **8.08** | **−5.18%** | nothing. Insensitive to every open term |
| the pipelining increment on top of it | 4.04 | −2.13% | **`g < 36 ns`** |

So most of `v4`'s prize is a two-line change that has nothing to do with pipelining, and the
pipelining itself is the part that can fail.

> ⚠ **"Two thirds" was the model's number and it was loose.** `8.08 / 12.12 = 66.7%` is the
> *modelled* share. Measured — `probes/23-ranking`, `j-66c7e331-fa41-461e-8686`, the `v4ragged` vs
> `v4c2048` rows of the ranking table above — the raggedness-only arm captures **61% of `v4`'s
> whole-call win (7.09 / 11.59 µs)** and **55% inside (6.17 / 11.19 µs)**. Report the pair, not one
> of them: the share is a metric family whose members disagree by 6 points, and which bracket you
> quote decides whether the two halves look 3:2 or nearly even.

### ⛔ `g` is measured, and it refutes the pipelined rung at the shipping chunk

`probes/19-descriptor-cost`, job **`j-53755323-aeb3-43d7-b06c`**, `measured`. Null spread 0.153 µs
(0.079%), dispatch floor 31.19 µs, both fitted intercepts on the floor (32.013 / 31.685), and
`raw/c1` vs `pipe/c1` agree to 0.148 µs — inside the floor, as a one-step pipeline must.

**`g_raw = +517.5 ns`, `g_pipe = +248.4 ns` per descriptor per subcore.**

Three results, two of which change the verdict rather than confirming it:

1. **The 36 ns bar is missed by 6.9×.** At `tc = 1024` the pipelining increment is not +4.04 µs but
   **+27.82 µs** against a 13.469 µs prize — net **−14.35 µs, a 2.1× loss.** ⛔ **That rung is
   refuted and dropped.**
2. ✅ **`emit_pipeline` does not add bookkeeping — it *removes* 269.1 ns/descriptor.**
   `g_pipe/g_raw = 0.480`. The model layered pipelining overhead *on top of* a raw blocking DMA and
   expected `g_pipe > g_raw`; measured, the pipeline **halves** the marginal cost. **Any model with a
   positive bookkeeping term double-counts.** 248.4 ns is the number to use, not 517.5. Note this
   also retires the build's own 17-bundle static bound as a proxy for `g` — it bounded the
   bookkeeping half of a term whose sign was wrong.
3. **"`g > ~225 ns` kills `v4` at any chunk size" is not what the arithmetic says.** Descriptor count
   halves with chunk while the prize does not move, so break-even **doubles** with chunk:

   | chunk | break-even `g` | measured 248.4 ns | verdict |
   | ---: | ---: | --- | --- |
   | 1024 | 120.3 ns | 2.07× over | ⛔ loses, −14.35 µs |
   | **2048** | **240.5 ns** | 3.3% past | **a wash, −0.44 µs — genuinely undecided** |
   | 4096 | 481 ns | under | ⚠ **do not read as "go bigger and `v4` wins"** — the ragged prize is itself a function of chunk granularity and this probe does not measure that function |

**Net: `v4` stays in the ranking as two rungs** — `ragged-blocking-only`, untouched by `g` and
carrying the 8.08 µs, plus a `tc = 2048` pipelined rung that is a coin-flip. `tc = 1024` is dropped.

### ✅ Rung 2 built, and it is the cleanest isolator in the wave

`SCTOPK_V4_RAGGED_ONLY=1` → scope `…_v4ragged`. Subject md5 now
**`5802c56b9608976ff447be57abc3eb72`**. It is the incumbent's code with **one expression** changed —
`ragged_len = walk_chunks * LANES` where it said `slice_len` — keeping the single descriptor, the
same destination buffer and the blocking start/wait. `buffer_count = 0`, so the VMEM check does not
charge it for a buffer it never allocates.

"Ragged and nothing else" is **verified, not asserted**: it compiles to *exactly* the incumbent's
instruction stream — **1168 TEC bundles at b=1024, 1166 at b=64, delta zero at both** — with
`sparse_core_specific_metadata.txt` and `memory-usage-report.txt` byte-identical and a name-only
ENTRY diff. The only thing that differs anywhere in the compiled artifact is the DMA's length
operand. All six previously reported ENTRY md5s are unchanged after the reindent the branch required.

**What the ladder buys beyond separability:** the pipelining increment becomes
**`v4-c2048 − v4-ragged`, a difference between two measured arms at the same ρ**. The verdict on the
caller's actual mechanism therefore no longer depends on `descriptor_cost.py` being *right*, only on
its having been roughly right about whether to bother.

**A platform constraint established on the way**, and it makes `align(i_hi, LANES)` mandatory rather
than tidy: a dynamic **size** in `pl.ds` compiles; a dynamic **start** at an arbitrary offset
**fails** with `Slice offset for 32-bit 1D memrefs must be a multiple of 8`. A `lax.switch` ladder of
static sizes also compiles and was not needed.

⚠ **Rung 2 is a null at ρ=1 by construction**, so a ρ=1-only run scores the best rung as "no
effect". **If only one ρ is affordable, run 0.40005.** Its correctness gate matters more than its
timing: a short `align` drops tail scores **silently** — nothing crashes and `k` selections still
return. `checksum.py` runs it first at both `b` and both ρ, with a guard so the test cannot pass
vacuously: it asserts the `_row_lengths` distribution actually contains rows not a multiple of
`LANES` *and* partial rows, failing with a named reason if either is zero.

**What the build closed.** All 7 AOT arms compile, so **`v4` cannot fail to build** (discharges the
build half of risk row 3; only register pressure in the fused body remains, needing `/xprof-llo`).
SparseCore bundle counts, no TPU: base 1168, `v4` 1172 (+4, 0.34% of IMEM), `nopipe` 1155 — so the
pipelining machinery is exactly **17 bundles**, which bounds the *bookkeeping* half of `g` at order
10–20 ns, under the 108 ns break-even. It does **not** measure `g`; the dominant half is descriptor
issue and sync-flag wait, invisible statically, and no SC clock is published. ⛔ **c1024 and c2048
are byte-identical**, so `emit_pipeline`'s grid is a genuine dynamic loop, not peeled per chunk —
an assumption in the prediction, now evidence, and it means the chunk sweep carries no code-size
confound. The builder therefore asks for **`tc = 2048` as the primary arm, not 1024**: 225 ns
break-even for 89% of the prize, at zero IMEM cost.

**A control that did NOT fail, and that is the finding.** Forcing `chunk_words = 3072`, which does
not divide `n = 10240`, **compiles cleanly**. The divides-`n` rule is the author's, not the
compiler's — a future hardcoded chunk gets silent cross-row reads rather than an error. Legal chunks
≤ 4096: `{16,32,64,80,128,160,256,320,512,640,1024,1280,2048,2560}`; **4096 is not one**
(`10240 = 2¹¹·5`).

**⚠ A correction this build owes every other candidate.** The XLA memory-usage report is **blind to
Pallas/Mosaic SparseCore VMEM** — `Memory Space: vmem` empty, `vmem_bytes: "0"`, for a kernel
allocating 70+ KiB, and byte-identical between arms that differ in VMEM. Mosaic allocates below XLA.
It cannot check a high-water or a spill risk. `v4`'s 78.50 KiB stays `modelled`. (This does not
overturn `v12`'s reading of a nonzero `vmem` total from the same file: that was **XLA-level** VMEM,
a different quantity from the kernel's private VMEM. Do not conflate them.) Relayed to `v3` and
`v10`, who both change the kernel body.

**Also contributed:** a **positive** control the brief did not ask for — a 4 KiB VMEM scratch via the
same DMA form, which must **compile**. Without it, "8 MiB fails" is equally consistent with the DMA
form being broken for an unrelated reason; with it, allocation size is the only variable between
compile and refusal.

**Carried, and it may outrank `v4` itself.** The ragged blocking DMA above: −5.18% `modelled`,
insensitive to `g`, ~2 lines. ⚠ It scores as a **null at ρ=1**, so the frozen harness alone would
miss it entirely.

---

## `v2` — `R=1`, shared histogram

**Built. Correctness pending. Not yet measured.** Folder: `variants/v02-r1-shared-hist/` (subject
md5 `c0ca54a66a549020cfc584e2d8d97abe`; optional rival arm `sparsecore_topk_bigscratch.py`
`70f9873e2941bd94601c15cea1dd9d55`). Seven hunks at incumbent lines 72, 128-135, 185, 191, 224, 238,
456.

**Prediction, to be scored — and it is a floor, not a central estimate.** Plan A's −6.42% harness /
−7.83% production is a **floor**, because the 12.195 µs it rests on is a saving *per executed
histogram level* and **every probe ran `REFINEMENT_LEVELS = ()`**. Central: ≈ **−8.1% at ρ=1**,
≈ −13.4% harness / **−15.9% production** at ρ=0.40005. All `modelled` ⚠ `refuted-denominator`.

✅ **The build designed its own discriminator, and it is a good one.** The ρ=1 arm separates the two
hypotheses in a single cell: **−8% means per-level scaling, −2.7% means L0-only — 3× apart.** This
goes into the ranking job as a named cell, not a hoped-for read.

⚠ **But `v2`'s percentages are internally inconsistent and the cell will report µs instead.** `v2`'s
saving is bounded by `i_hi`, so it is **in-walk** and must take the ×0.62 branch of the fork. Its own
figures do not: −8% / −2.7% at ρ=1 mapping to −7.83% / −15.9% at production implies factors of
**×0.98 and ×1.99**, neither of which is 0.62 or 1.55, and the two are not even the same factor.
**At least one of `v2`'s two reweightings is wrong.** The measurer will register the arm reporting
**absolute µs** and re-derive every percentage in one place against the measured 275.185 / 291.075,
rather than inherit either. This is the ×1.55/×0.62 fork doing exactly the job it was identified
for — and note the contrast with `v12`, whose independent derivation *did* land on 1.546.

**⛔ The build's most important finding: Plan A §2's edit list is incomplete, and following it
literally corrupts memory.** With `priv_vmem` shrunk to 272 words per the `v2` node's 1 KiB
allocation, leaving `zero_priv(NUM_BUCKETS)` at `:224` writes `LANES * NUM_BUCKETS = 4096` words into
a 272-word ref on every unconverged refinement level — over the top of `glob_vmem`. **Four edits are
required, not three.** The probe could not have seen it: `arm-R1.py:48` sets
`REFINEMENT_LEVELS = ()`, so the line never executes, and the probe kept the 16 KiB allocation
anyway.

**Three further ways `v2` differs from the `arm-R1` probe that priced it**, each stated rather than
assumed: a cheaper fold spelling (statically 24 equations / 16 adds fewer, so **`v2` cannot be slower
than the measured arm**); **`arm-R1` never shrank `scratch_types`**, so the 12.195 µs contains no
tile-memory effect at all; and the probe's `REFINEMENT_LEVELS = ()` above.

**Risk row 1 — `did not happen`, and checked at the core rather than argued.** A host census finds
**100% of 655,360 vectors conflict** at `b=1024 random`, 6.73 M colliding lane pairs, and `R=1` was
**bit-identical** there. Firing assertions named (`test_dsv4_sparsecore_topk.py:126-128`, then
`:133-136`). Still open at the edges: refinement scatter, `heavy_ties`, `strict_desc` never run under
`R=1`. Row 2 `still unchecked`, with the touched-line list handed to composition.

**`static-compile`.** jaxpr 1,555 → 1,013; `add` 314 → 50; `i32[4112]`×32 present in the incumbent
and absent in `v2`; 0 spills; negative control live. **TEC bundles 1,168 → 919, −249 (−21.3%)**, other
three sequencer counts unchanged — independent evidence the edit shrank the emitted stream and is
confined to one computation. ✅ Flagged by the builder as **static code size, not a time prediction**,
and deliberately excluded from its §6. ENTRY md5 byte-identical across two fresh processes, which is
a stronger form of "the change is wholly inside the `tpu_custom_call`" than a same-process diff.

**Open terms.** Per-level scaling is `derived` and **credited as a saving, so it cannot fail** —
it returns as `unmodelled` unless the ρ=1 discriminator resolves it. The 15,360 B/subcore SparseCore
reclamation is `unmodelled` at zero and **invisible at the XLA level**, which upgrades the
`V2_BIGSCRATCH` rival arm from a nice isolator to **the only route to that number**.

**Carried.** `V2_BIGSCRATCH`. A `references/08-sparsecore.md` self-contradiction at `:541` vs `:646`
— the correction is that `addupdate_scatter` **does** resolve intra-vector conflicts, which agrees
with `probes/16`'s collision result. A `[mutate]` node routing the hardcoded-constants cleanup
against the *incumbent*, so the control moves too. And `aot_single_arm.py` as the reusable form of
the three AOT traps, for `references/` promotion.

---

## `v3` — fused emit ⛔ **predicted to LOSE**, and it was rank 1

**Built, exact, compiles at every reachable shape — and predicted to lose.** Folder:
`variants/v03-fused-emit/`. **+1.85% … +6.27% harness / +2.27% … +7.71% production**, rate
`modelled`. The shortlist's **−25.0%** cannot happen, for a structural reason.

⛔ **The prize was priced against a pass that cannot exist.** Plan B's `_l0_fused` wants the level-0
histogram walk to also classify elements against the pivot bucket. But **`b_star` is the *output* of
that histogram** — nothing during the walk knows the pivot yet. So the two stores cannot ride an
existing pass; they need their own full-length walk of `L`. `store_compressed` is flat in mask
density (`probes/15`, 6.675 µs/pass = 0.32×W), so `v3` adds two of them to a full walk (+5.34 µs) to
save 0.33 of a walk (−2.74 µs). **Break-even is α = cnt/i_hi = 0.2375; measured α is 0.347 prefill,
0.636 decode.** `probes/13`'s "`v3` cannot become a loss" is **superseded** — it was computed against
the impossible pass.

✅ **The hazard the dispatch flagged is measured at exactly zero**, by full bundle-level attribution
of the scheduled TEC program, closing with no residual:

| region | base | `v3` | Δ |
| --- | ---: | ---: | ---: |
| **L0 histogram** | **108** | **108** | **0** |
| `split` (new) | — | 204 | **+204** |
| everything else | 1049 | 1047 | −2 |
| total | 1157 | 1359 | +202 |

The L0 region is the same seven blocks in the same order with **identical opcode sequences** and 13
`addupdate_scatter` instructions in both arms, so the collision nonlinearity has no channel. ⚠ **A
note for the measurer that generalises:** risk row 1's stated signal — "L0's own arm time rises" —
**will read clean and would have exonerated the change.** The cost is on a different pass.

**The store price survived an independent test.** `split` (3 stores) vs the emit region (1 store) is
204 vs 168 bundles; bracketing out `fill` gives **1.21–1.73×**, which contains the model's 1.49× and
**excludes the 1.0×** the −25.0% assumed. Opcode deltas are exact multiples of the per-site cost —
`vst.msk` +39 = 3×13, `vmpcnt` +26 = 2×13 — so the MLIR count was not an undercount. Zero spill delta.

### ⛔ Two instrument defects `v3` found, both of which outlive the candidate

- **`design.md`'s `cnt_max = 4096` "safe bound" is falsified.** `cnt` reaches **10240**. A variant
  built to that bound with `disable_bounds_checks=True` overflows by 2.5× and corrupts adjacent
  scratch. `v3` sizes at `slice_len + LANES`, so it is immune by construction.
- **The suite *would* have caught this particular overflow — and the real blind spot is narrower.**
  ⚠ This row went through two corrections and the second reversed the first; both are recorded
  because the reversal is the useful part. `v3` first cited `all_negative` at `p1-b1024-n10240`.
  **That id does not exist** — `:290` limits `all_negative` to `TWO_STAGE_SHAPES` (`P>1`) and `:230`
  runs only `["strict_desc", "random", "heavy_ties"]`. I took that to mean a 4096 overflow would ship
  past a green 68/68. **It would not.** Re-running with the score generators transcribed verbatim
  rather than approximated, max `cnt` at ρ=1 is `strict_desc` 1536, `random` 3054, **`heavy_ties`
  5526**, `all_negative` 7089 — and `heavy_ties` **does** run at P=1, where it overflows a
  4096-word buffer by **1.35×**. So the failing id is
  `test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]`, and it goes red.
  ⚠ **Root cause of the original error, which generalises:** the simulation used *approximations* of
  the score generators instead of transcriptions — its `strict_desc` was positive where the real one
  is `-np.arange` (all negative). **Transcribe a generator; do not paraphrase it.** And confirm an
  id with `pytest --collect-only`, not by reading the parametrize decorators.
- **The genuine blind spot, stated at its real size.** Two gaps, both from `TWO_STAGE_SHAPES`:
  P=1 `cnt` coverage rests on **one** kind with only **1.35×** of margin, so a per-bucket-capacity
  bug modestly less bad than the design's does ship silently; and **P=1 has no coverage at all for
  negative winners** — `all_negative` is the only kind whose winners are negative, hence the only one
  where `monotone_key`'s involution does anything at emit time, and it never reaches the production
  shape. ✅ Confirmed by `pytest --collect-only` (1.2 s, no TPU): **68 ids, 12 at P=1**;
  `heavy_ties-p1-b1024-n10240` and `heavy_ties-p1-b64-n10240` both exist; `all_negative` appears in
  exactly **5** ids, **0** of them at P=1. ⚠ `strict_desc` is all-negative in its *scores* but
  degenerate-ordered in its behaviour, so it does not substitute.
  **Fix, routed not applied:** add `all_negative` to `test_exact_topk_full_rows`'s kind list —
  ⚠ but that test is parametrized over all 7 `SHAPES`, so one kind adds **7 ids, 5 of them at the
  expensive `N_128K` shapes**, not "one shape of runtime". Both residual gaps live at P=1, so
  **restricting the added kind to the two P=1 shapes buys the whole fix for 2 ids.** Not edited —
  instruments are read-only; it belongs in the ranking job's own file importing the frozen helpers.

**Correctness: exact, but NOT byte-identical.** `v3` emits certain winners before tie-broken ones —
winner *set* matches, slot *order* does not. `_mismatch` compares score multisets so the frozen suite
is unaffected, **but a byte-compare gate reused from `v2` will fail on a correct `v3`.** `sim.py`:
960/960 rows set-identical to the incumbent and multiset-identical to numpy top-k. The correctness
gate is therefore **per-variant**, not one shared gate.

**Private VMEM priced by allocation failure**, since the dump files are blind to it: base tolerates
`n ≤ 123,904`, `v3` `n ≤ 40,960` — **measured ratio 3.025× against 3.000× predicted**, with the base
threshold validating the declared model to 0.23%. A reusable technique.

**Carried.** `v3` narrows supported `slice_len` from ~123,904 to 40,960 while `MAX_SLICE_WORDS` still
says 65,536 — unreachable today (production 10,240, largest test 33,280). The fix is verified
P-neutral: lowering the cap to 40,960 changes `_pick_partition` for **0 of 216** combinations built
from every `n` in the frozen test file, `P=4` at `n=133120` included. Not applied.

**Open term, and it is the one number that could still move the verdict.** The emit region-vs-per-pass
**3.05×** (`probes/09` 83.830 vs `probes/03` 27.456). ⚠ `probes/09` sits on one side of this and is
now independently known not to reproduce — but that does **not** resolve it in `v3`'s favour, because
correcting `B` upward makes the raw gap *wider*. The two numbers should not be reconciled; emit
should be measured cleanly. Cheapest settling arm: force `emit_body`'s bound to fractions of
`walk_chunks` and read the slope — no region-deletion confound, no fit, ~1 job. It re-prices several
other nodes, not just this one.

---

## `v10` — sample-and-count ⛔ **predicted to LOSE**, and both rescuing terms closed against it

Built to `design.md` §9 Plan E: replace the dense 256-bucket L0 `addupdate_scatter` **and** all three
8-bit refinement levels with a 1-in-16 sampled histogram giving an *estimated* bracket, plus a
correction loop of count passes — a compare and a lane popcount, no memory scatter — walking the
estimate until it provably brackets the k-th key. 16 hunks, +215/−105, plus a new 250-line
`sparsecore_topk_sample.py`. Four arms, arity ∈ {2,4,8,16}, all `static_argnames`.

| | harness | production |
| --- | ---: | ---: |
| Plan E §5 predicted | −9.7% | — |
| the `v10` version-log predicted | −12.2% | −14.9% |
| **plan as written (arity 2)** | **+11.99%** | **+14.61%** |
| best arm of the root (arity 2, Illinois) | +3.25% | +3.97% |
| arity 4 / 8 / 16 | +27.4 / +61.2 / +120.8% | +33.4 / +74.6 / +147.2% |

**The mechanism is information-theoretic, not incidental — which is why no arm of this root
recovers.** A count pass costs 6.48 µs ρ-weighted and yields **1 bit**. A refinement histogram costs
12.79 µs/level and yields **8**. Per bit that is 1.60 µs of scatter against 6.48 µs of counting: the
scatter is **4.05× more bit-efficient despite being 5.2× slower per pass**. The production L0 pivot
bucket holds ~1071 elements, so ~10 bits below L0 must be resolved by *some* mechanism, and the
incumbent buys them eight at a time. Budget for the whole correction loop is 61.42 µs = 9.48 passes;
the measured requirement is **12.999** at prefill ρ (p90 16.2, p99 21.7) and **18.772** at decode.

**Both open terms closed against the node, and one of them closed the escape route.**

- **The pass count** — the term the dispatch assigned — is closed at 12.999 against a break-even of
  9.48. The generator is `probes/13`'s, which reproduces `streamindex_topk.py:326-338` from source
  and **self-validates**: at ARITY=256 over the full 32-bit range it becomes the incumbent's radix
  walk and reproduces `level_sim`'s level count exactly, 1.740 vs 1.740 prefill. Three ρ=1 synthetics
  all make the requirement **worse** (14.25 / 16.13 / 26.00), so the distribution is not the escape.
- **The marginal cost of an extra threshold** was worth a **55-point swing** before this build — free
  at 0.111 µs it priced arity 16 at −17.8%; at 2.35 µs, +38.8%. Measured from the emitted bundles
  *in the production kernel at the production shape*, `b(m) = 1.5 + 9.5m`, so one extra threshold is
  **5.59 µs**, 0.864 of a whole pass. Arity 16 needs it under ~0.8 µs — a 7× error.
- **Plan E's own stated fallback cannot reach.** Widening the sample to yield the ~3.5 more bits
  needs ~4096 buckets, and the sweep is linear in `NUM_BUCKETS`: ~97 µs at 4096, alone exceeding the
  entire 61.42 µs budget. Widening the *rate* 1/16 → 1/8 changes the bracket's variance, not its
  width, and buys no bits.

**⚠ One arm still runs.** Every µs above is a row of the §4a price sheet, and every probe behind that
sheet ran at ρ=1 — the sheet `probes/18` refuted by 76.60%. `v10` says so itself. Arity 2 is the one
arm whose number is invariant to the contested marginal-threshold term, and it costs one slot in a
pool that is already being paid for. **Arity 4/8/16 are dropped**: they turn only on the term that
closed, and need it to be wrong by 7×. Do **not** buy the standalone 4-minute falsifier microbench
`v10` offers — it prices the term that governs only the dropped arms. If arity 2 measures materially
better than +11.99%, that reopens the sheet in a direction that helps the higher arities too, and
*then* the falsifier is worth four minutes.

**The exactness claim is `v3`'s shape, and for the same reason.** `v10` is bit-identical to the
incumbent whenever the k-th score is unique; on a tie it may select a different subset of tied
indices — same multiset of scores in the same slots. The emit walk is bundle-identical across all
four arms, so an identical selected set gives an identical slot order. A byte-comparison gate against
the incumbent's *indices* is the wrong gate, and `v10` makes the sharp version of the argument:
**it would also fail the incumbent against itself under a different tie order.**

**What did not happen, and is worth as much as the verdict:** the `v5`/`v6` absorption failure is
excluded here by two independent instruments. The emit and sweep innermost loops are
**bundle-identical in all four arms** even though the sample boundary drops a `vshra` per lane — the
shift was riding a spare slot — and private VMEM is equal by Mosaic's own `memref.alloca` list
(69 KiB, byte-identical) *and* behaviourally by allocation threshold (122880 compiles / 123904
refused, identical across base, sc2 and sc16). The saving is not being soaked up by an unpriced
neighbour. **`v10` fails on its own arithmetic**, which is the cleanest way for a node to die.

### Carried

- **`PARKED` — compact the pivot bucket, then count over the compacted set.** Each count pass would
  walk ~1071 elements, not ~4096. Priced and does not clear: it still needs the *full* L0 histogram,
  so it keeps hist0 and one sweep, and adds a compaction walk → ~+6.5%. Its deciding term, the
  compaction walk, is `unmodelled`. Belongs in `design.md` §9 beside Plan E so it is not re-derived.
- **`REFINEMENT_LEVELS` is a fixed 3 against a measured requirement of 1.740 mean / 1.664 crit.**
  `v10` claims no prize because the kernel already `pl.when`-skips resolved levels, so level 3 is
  mostly a predicated no-op. A one-constant arm belonging beside Plan D, which measures the same term.
- The hardcoded-hardware-constants debt, reaching the same conclusion as the other five builders
  independently: not fixable inside a variant folder without changing the subject of every A/B.

---

## Cross-cutting, from the wave rather than from any one candidate

1. **The three AOT traps all fired somewhere, and each would have answered confidently and wrongly.**
   The jit-cache aliasing bit `v11` for real (byte-identical HLO for both arms, reading as a clean
   null); `v12`, `v4` and `v2` were all structurally immune but **re-derived every headline number
   one arm per process, variant first**, rather than asserting immunity. The wrong-layer negative
   control was hit independently by three builders. `/tmp/libtpu_lockfile` contention hit two, both
   of which retried rather than deleting it.
2. **The candidates do not share a measurement boundary.** `v2`/`v3`/`v10`/`v4` are inside the
   kernel, `v11` is at the output edge, `v12` is entirely outside the kernel call. **A single bracket
   cannot score all of them**, and `v12` specifically reads as a flat null under a kernel-level
   bracket. The ranking job reports **both** an inside-kernel and a whole-call time per arm.
3. **⛔ The frozen bench has no ρ knob on any timing path** — `--decomp` sets `rl = np.full(b, n)` at
   `sctopk_2stage_bench.py:582`, and `ab()`/`decompose()` both hard-set full rows; ragged lengths
   reach only correctness paths. Two builders reached this independently. `probes/18` has now built
   the ρ driver, and per its finding 1 the ranking must vary **arrangement** as well as ρ.
   ⛔ **And it is worse than "no knob": the `b=64` cell is not decode.** `_row_lengths` cycles the
   same edge set regardless of `b`, so at `b=64` the frozen bench runs **ρ = 0.391**, not production
   decode's 0.850 — a small-batch *prefill-occupancy* run wearing a decode label. This is not a `v4`
   fact; it governs **every variant's `b=64` arm**. For `v4-ragged` the harness will read −1.20% where
   production decode is −0.30%, **overstating it 4.0×**, and a ρ-sensitive candidate can be
   overstated or understated depending on which side of the knee it sits. Either drive `b=64` from
   `probes/18`'s ρ driver, or label the cell **small-batch prefill occupancy** and leave production
   decode `modelled`. What is not available is reading it as decode.
4. **Static SparseCore bundle counts are a genuine new instrument**, free and offline, and four
   builders used them to price a mechanism (`v2`'s −249, `v12`'s 1364-vs-1364 null, `v4-ragged`'s
   **delta-zero** proof of "one expression and nothing else", and `v3`'s full region attribution
   closing to no residual). ✅ All four labelled them **static code size, never a time prediction** —
   and `v4` supplies the cautionary case: its 17-bundle static bound on pipelining bookkeeping was
   sound as code size and useless as a cost proxy, because the measured term had **the opposite
   sign**. `*sparse_core_specific_metadata.txt` is ~90 lines against 19 dump files, the cheaper read.
5. ⛔ **The correctness gate must be per-variant, not one shared gate.** `v2` is bit-identical and a
   bit-exact gate is its strongest evidence. `v3` is exact but **not** byte-identical — same winner
   set, different slot order — so the same gate fails a correct `v3`. Each variant declares which
   claim it makes.
6. **`cnt_max = 4096` is falsified, but the suite does catch it — with 1.35× of margin, on one
   kind.** ⚠ This item was stated wrongly twice before it was stated right, and the sequence is the
   lesson. A builder cited a test id that does not exist; I checked the parametrize decorators,
   confirmed the id was absent, and concluded the suite was blind. Both were wrong: `heavy_ties`
   reaches `cnt` 5526 at `p1-b1024-n10240` and overflows a 4096-word buffer by 1.35×, so the build
   goes red. **Two method rules came out of it** — transcribe a score generator verbatim rather than
   approximating it (the approximation had `strict_desc` positive where the real one is all
   negative), and confirm a test id with `pytest --collect-only` rather than by reading decorators.
   ⛔ **The residual claim — "no P=1 coverage of negative winners at all" — is itself false, and the
   2-id addition it motivated is dropped.** `:145` is `np.repeat(-np.arange(n, dtype=np.float32)…)`,
   so `strict_desc`'s top-2048 winners are `{0, −1, …, −2047}`: **2047 of 2048 strictly negative**, at
   a P=1 shape that runs. The involution is exercised at P=1 today. `heavy_ties` is
   `rng.integers(0, 17)` and its winners are all non-negative, which is what made the wrong reading
   plausible.
7. ⛔ **A frozen instrument's comment asserted a coverage property the instrument itself contradicts,
   and believing it cost three wrong statements.** `:158-163` justifies `all_negative`'s existence
   with "Under every other `kind` here the winners are positive and a kernel that dropped the
   involution entirely would still pass." `strict_desc`, forty lines below at `:145`, has negative
   winners. **The rule that would have caught all three readings on the first pass: coverage is a
   property of what the generators produce, not of what the file says about them — compute the
   winners.** Reading the parametrize decorators was the first failure, approximating a generator the
   second, and trusting its docstring the third; `pytest --collect-only` fixes only the first. ✅ It
   also runs **on the host in 1.1 s with no TPU**, which is worth knowing generally. What
   `all_negative` uniquely adds over `strict_desc` is a *random arrangement* of continuous values
   offset off zero, not negativity — a real but much narrower thing than its comment claims.
   **Route:** correct the comment when the hardcoded-constants cleanup touches this tree; do not
   edit a frozen instrument mid-exercise to fix a comment.
8. **The one genuine gate gap in this family is `v10`-specific, and `v10` closed it off-cluster.**
   No gate id at any shape presents a bracket wider than 2²⁴ — all three kinds resolve into a single
   24-bit L0 bucket — while **32% of production prefill rows leave the lower edge at `INT32_MIN`**,
   8.6% span zero, and brackets reach 31.6 bits. So the `hi − lo > 2³¹` wrap case is ungated. It only
   exists for `v10`, whose `_split` is the only bracket arithmetic in the wave. Closed on the host:
   the same `jnp` expression on the CPU backend over **84,100 reachable int32 cases**, 0 violations,
   the 40 excluded cases asserted to be exactly the width-1 pairs the loop condition makes
   unreachable, and — the part that makes it evidence — **a negative control firing on 5,263 of
   21,035 pairs**. Stronger than two extra pytest ids, at no cluster cost.
9. ⚠ **Two different distributions share the name `heavy_ties`.** `probes/05` uses
   `linspace(-2, 2, 17)`; the frozen test uses `rng.integers(0, 17)` — different sign, different
   spacing. They coincidentally agree at 26.00 passes in `v10`'s sim, which is exactly how a
   name-collision survives. Label any `heavy_ties` number by its source.
10. ⛔ **A SparseCore bundle count measures issue cost and is blind to stall cost — and the two can
   disagree by 6×, in opposite directions.** `v10` measured `addupdate_scatter` against a lane
   popcount over the same walk: **5.2× slower in time, 0.82× in issued bundles.** The scatter
   *stalls* on tilespmem; it does not *issue*. This is the second instance in the wave of a static
   bundle count having the wrong sign as a cost proxy — `v4`'s 17-bundle bound on pipelining
   bookkeeping was the first, and there the measured term was **negative**. Item 4's discipline
   ("static code size, never a time prediction") is therefore not a formality: **a cost model built
   from either number alone is wrong about the other.** The constructive half is that the bundles
   *are* free and offline — the dump prints **symbolic** branch targets (`sbr.rel .LBB2_34-.Ltmp25`),
   so natural loops, and therefore bundles per walked vector, are recoverable per region from a
   host-only AOT compile for zero cluster time. `variants/v10-sample-and-count/bundle_stats.py` is
   ~150 generic lines and is the reusable form. Both halves route to `references/08-sparsecore.md`.
11. **Allocation-failure sweeping prices private VMEM that no dump file can see.** `v3` got a measured
   3.025× against a predicted 3.000× this way, with the base threshold validating its declared model
   to 0.23%. It is the answer to the `memory-usage-report` blindness in item 4's family, and it costs
   no cluster time.
