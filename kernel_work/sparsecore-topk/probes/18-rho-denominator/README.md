# probe 18 — the ρ denominator, and the step-3 measurement harness

Two jobs in one folder:

1. **the step-3 harness.** Everything that will time a step-3 variant is here: `subject/` +
   `--map-dir`, the in-job md5 gate, the frozen `measure()`, one pool, interleaved reshuffled
   rounds, three separately-staged nulls per shape, per-arm absolutes. Variant agents' arms plug
   into `reg(...)` in STAGE 2 and change nothing else.
2. **the purchase that outranks every variant.** The whole step-2 sheet's denominator is
   `modelled` — nothing has ever executed at the production `row_lengths`. This job converts it to
   `measured`.

---

## The boxed question

> The sheet asserts a harness denominator of **189.98 µs**, a production denominator of
> **155.82 µs firm**, and a single scale factor of **1.219**. What does the incumbent actually cost
> at ρ = 0.40005 and b = 1024, and how does that compare with 155.82 µs?

**Answer:** **275.185 µs** — `P3.prefillX`, `us_median`, b = 1024, 3-D production entry, ρ = 0.40005
exactly, resolution floor 3.218 µs. That is **+76.60%** against the sheet's 155.82 µs. The
gate-corrected per-invocation production denominator is **291.075 µs**, **+86.80%**. The `1.219`
is **refuted as a factor** and **confirmed as an additive term**: the relayout it stands for is
**35.563 ± 0.10 µs**, ρ-independent, which is probes/11's 34.165 µs inside the floor — but divided
into the *measured* production cost it is a factor of **1.129**, not 1.219. See **§ Answer**.

## Why the question is not "measure one number"

`189.98` and `155.82` are not two measurements. They are one measurement and two models:

| | value | class |
| --- | --- | --- |
| probes/09 `B`, b=1024, **ρ = 1**, **2-D** entry, `standard_normal` scores | 326.782 µs | **measured** (`j-4e0df30a`) |
| × per-term ρ weights, `hardware_map.md:521-536` | 189.98 µs | modelled |
| − 34.165 µs of harness-only XLA relayout (probes/11) | 155.82 µs | modelled |

So the sheet contains **two** factors, and only one of them is the `1.219`:

- an **implied ρ factor** `326.782 / 189.98 = 1.7201` — never measured, applied term by term;
- an **entry-shape factor** `189.98 / 155.82 = 1.219` — the `f32[b,n]` → `f32[b,n/128,128]`
  relayout, measured at ρ = 1 in probes/11 and *assumed to carry* to ρ = 0.40005.

This probe measures **both**, plus the absolutes, plus how much of either depends on an assumption
about the length *distribution* rather than its mean.

## What ρ is, mechanically

ρ = `E[eff_len]/n`. `n = 10240` is a padded, page-aligned KV **capacity**; it does not change with
sequence length. Live length reaches the kernel through exactly one operand:

```
eff_len     = min(row_lengths[row], n)                      # :~205
i_hi        = where(eff_len > 0, min(slice_len, eff_len), 0)
walk_chunks = ceil(i_hi / 16)
```

`walk_chunks` bounds `cvt`, `hist0`, every refinement level and `emit`. It does **not** bound the
score DMA, which is `dma(scores_hbm.at[pl.ds(s, slice_len)], ...)` — a **static** length. Both
facts are asserted as one-occurrence source anchors in the dry-run `ANCHOR_GATE`. `keep_all =
total <= k` additionally deletes the whole refinement loop when `eff_len <= 2048`.

**So the ρ arm is a `row_lengths` distribution, not a different `n`, and no source is edited.**
Every prefill arm below is the *same compiled executable* with a different `lengths` operand.

## ⚠ What the frozen bench does today

`instruments/sctopk_2stage_bench.py` has a ragged-length generator, `_row_lengths(kind, b, n, n_p)`
at `:256`, that straddles page boundaries. **It is never used by a timed path.** It is called only
from `gate()` (`:332`) and `variant_equality()` (`:725`). Both timed entry points hardcode ρ = 1:

- `--decomp` — `:583` `rl = jnp.asarray(np.full(b, n, np.int32))`
- the A/B path — `~:1088` `rl = jnp.asarray(np.full(b, n, np.int32))`

`probes/09/probe.py:355-382` likewise: `lengths = jnp.full((b,), n, jnp.int32)`. So this is exactly
the "microbenches default ragged lengths to max" trap: **the frozen instrument has never timed
anything at ρ ≠ 1**, and no prior probe in this exercise did either.

## The distribution, and where it came from — not a scaled mean

`prefill_lengths` and `decode_lengths` are copied **verbatim** from
`probes/13-level-count-refit/level_sim.py:254-266` and gated with `inspect.getsource` against that
file (`VERBATIM_GATE_OK`, dry run). They are not invented here:

| | distribution | E[eff_len] | ρ |
| --- | --- | --- | --- |
| prefill | Uniform{1 … 8192} | 4096.5 | **0.40005** |
| decode | Uniform{8193 … 9215} | 8704 | **0.850** |

Provenance chain: `_effective_row_lengths` (`streamindex_topk.py:499-506`) is
`visible = min(kv_len, token_pos // compression_ratio + 1)` with `cr = 1`; `configs/glm-5.2-…`
gives ISL 8192, `RANDOM_RANGE_RATIO=0.0`, max-model-len 9216, chunk 1024 → 8 prefill chunks, chunk
`c` covering `token_pos ∈ [1024c, 1024c+1023]`, i.e. ρ_c = 0.1c + 0.05005.

**The mean is not the assumption; the shape is.** Three arms hold E[eff_len] = 4096 fixed and vary
only the distribution — degenerate (all 4096), the production uniform ramp, and bimodal
{2048, 6144} — and a fourth holds the *multiset* fixed and varies only the row order. The reported
sensitivity to the distributional assumption is that spread, measured, against the null floor. If
it is inside the floor, "a mean-only ρ model is adequate" is a measured statement; if it is not,
every ρ-reweighted number in the sheet inherits it.

Two further reasons the shape matters here, both structural:

- **`keep_all`.** 25% of the production prefill rows have `eff_len ≤ k`, which deletes their
  refinement loop entirely. A degenerate arm at 4096 has 0%; the bimodal arm has 50%. A mean cannot
  see this.
- **wave imbalance.** Rows map to subcores with stride 32 (`row = w*32 + flat`) and a wave costs its
  **slowest** subcore. A sorted batch has almost no within-wave length spread; a shuffled one has
  the full range. `P3.prefillXsorted` is the identical multiset in sorted order, so
  `prefillX − prefillXsorted` prices arrangement alone. Production prefill within one chunk is a
  monotone ramp (the `chunkC` arms); a mixed batch is not.

### ⚠ i.i.d. at b=1024 misses its own ρ by 1%

`prefill_lengths(1024, rng)` is an i.i.d. draw, so its **realized** ρ is `0.39604`, not `0.40005`
— σ/√1024 is 0.0072 in ρ. That error lands directly on the number this probe exists to produce. So
the headline arms are `prefillX` / `gatedX` / `decodeX`: the **quantile function of the same
distribution on a regular grid**, rounded so the total is exact, then randomly permuted across
rows. Same support, same shape, same quantiles to ±1, and ρ exact to 1.2e-6 (`strat check` lines in
the dry run). The verbatim i.i.d. arms are kept and timed too, so the substitution is auditable
rather than asserted.

## Arms

47 arms, **one pool**. `b=1024` waves = 32, `b=64` waves = 2, `n = 10240`, `k = 2048`, `P = 1`
asserted, `--ctx 9216`.

### prefill, b = 1024, production 3-D entry (`P3.*`) — 22 arms

| arm | ρ | what it is for |
| --- | --- | --- |
| `rho1` | 1.00000 | what every prior probe ran — the in-pool ρ=1 anchor |
| `prefillX` | **0.40005** | **the headline**: stratified Uniform{1…8192} |
| `prefill` | 0.39604 | `level_sim.prefill_lengths` verbatim, i.i.d. |
| `prefillXsorted` | 0.40005 | `prefillX`'s multiset, sorted — arrangement control |
| `gatedX` | **0.50005** | stratified Uniform{2049…8192}: the `lax.cond` survivors |
| `gated` | 0.50587 | the same, i.i.d. |
| `flat0.400` | 0.40000 | degenerate at 4096 — distribution control |
| `bimodal0.400` | 0.40000 | {2048, 6144} — distribution control |
| `chunk0` … `chunk7` | 0.05005 … 0.75005 | the eight production causal ramps |
| `flat0.050/0.150/0.250/0.500/0.650/0.850` | | the shape of `T(ρ)` |

### prefill, harness 2-D entry — 5 arms

`P2.{rho1, prefillX, gatedX}` (production scores) and `N2.{rho1, prefillX}` (`standard_normal`,
`SEED 20260904` — probes/09's own operand). `P2 − P3` at a matched ρ **is** the 1.219, measured.
`N2.rho1` is a direct cross-job control against probes/09's `B = 326.782`.

### decode, b = 64 — 11 arms

`D3.{rho1, decodeX, decode, flat0.850, flat0.200/0.400/0.600/0.800/0.900}`,
`D2.{rho1, decodeX}`.

### nulls — 9 arms

`#nsrc_{a,b,c}` (b=1024) and `#dsrc_{a,b,c}` (b=64): the **identical incumbent source**, exec'd
into three separate module objects and jitted separately, on the **same operands**.
`#ndat_{a,b,c}`: one executable, three different production score draws.

## How each of the nine required properties is satisfied

1. **One pool, ≥ 9 interleaved rounds, reshuffled, read `us_median`.** All 47 arms go into a single
   `measure(thunks, rounds=12)` call on the frozen `instruments/glm_dsa_block_bench.py`
   (md5 `08375d70…`, gated in-job). It reshuffles the visiting order every round and pairs `lo`/`hi`
   **within** a round. `LAST_ORDER` is written to the JSON as the evidence the shuffle happened.
   Every derived number in STAGE 4-6 reads `m[name] = out[name]["us_median"]`; `us`, `us_mean` and
   `us_unpaired` are printed for audit and used nowhere.
2. **Every thunk warmed before it is timed.** The frozen `measure()` calls
   `jax.block_until_ready(variants[name]())` for every arm before round 1. A cold thunk's paired
   slope is *minus* its compile time, and with 10 distinct executables here that is the single most
   likely way to manufacture a ρ win. `suspect` (min slope ≤ 0) is printed per arm.
3. **≥ 3 nulls, and the spread is the resolution floor.** Six source-nulls (three per shape) and
   three data-nulls, above. `STAGE 4` prints the max unordered pairwise spread as **the prefill
   floor** and **the decode floor**, in µs and as a % of the corresponding `rho1` arm, and every
   subsequent line is tagged `resolvable` / `NOT RESOLVABLE` against it. Nothing in this README says
   "no effect".
4. **Measure a slope, don't divide a total.** `measure()`'s per-arm number is
   `(t(60 reps) − t(20 reps)) / 40` taken within a round — a slope in reps, not a total over a
   count. Likewise `T(ρ) = a + s·ρ` in STAGE 5 is a fit over **seven degenerate ρ points**, not
   `T(1)` divided by anything; its intercept `a` is the ρ-independent floor and its residuals say
   whether a line is even the right shape (a two-point measurement cannot tell a curve from a line —
   hence 7 flats, 8 causal ramps and 2 shape controls).
5. **In-job md5 gate on the subject, inside the job script.** The `RUN_CMD` itself does
   `test "$(md5sum $SUBJ | cut -d" " -f1)" = "44353b347c96e0eedbd742c710d4540a" || exit 1` before
   python starts, and the same for the frozen harness (`08375d70…`) → `MAP_GATE_OK`. `probe.py`
   then re-gates in `--dry-run` and again in `main()` → `SUBJECT_GATE_OK`, and prints the
   **installed** copy's md5 beside it (the image tag `fa63d6df` was re-pushed and now ships
   `1a89d451…`, so they will differ; the pinned `--subject` is what runs). The subject ships via
   `--map-dir …/subject:tpu-job:/root/probe18/subject`.
6. **`--ctx 9216`, always.** Passed explicitly; `probe.py` derives
   `n = ceil16(ctx + 1024) = 10240` and **asserts** it, with a `CTX_GATE` in the dry run that
   refuses any other value. The bench's default `--ctx 132096` (n = 133120) is a shape no
   checked-in config produces.
7. **Both shapes.** `b = 1024` (32 waves) and `b = 64` (2 waves), `n = 10240`, `k = 2048`, and
   `_pick_partition(b, n, k) == 1` asserted at both.
8. **Per-arm absolutes printed, not only ratios.** STAGE 3 prints every arm's
   `us_median / us / us_mean / us_unpaired / hostbound%`. STAGE 5 and 6 are the *only* places a
   percentage is formed, and every one of them is computed from two printed absolutes in this same
   pool. The full `us_median` table is in `rho_denominator.json`.
9. **Tagging.** `-t agent-jobs,measurer`. No job created by anyone else is touched.

### The gate that is not on the list, and matters most here

Every prefill arm is the same executable with a different operand. If `row_lengths` silently failed
to reach the kernel — the exact failure a ρ probe cannot survive — **all 22 arms would be
byte-identical and equally "correct"**, and the ρ curve would read as a flat null. So STAGE 1 is a
**lengths gate**, not just a correctness gate:

- per arm, per row: every live index `< eff_len`, `#live == min(k, eff_len)`, and the selected score
  multiset equal to a numpy `argpartition` oracle restricted to `[0, eff_len)` (and to `arange(L)`
  on the `keep_all` rows);
- a **negative control**: the ρ = 1 output scored against the `prefill` lengths, which **must** go
  red, plus an explicit `array_equal(rho1_out, prefill_out)` that **must** be `False`. If the
  checker cannot go red, `LENGTHS_GATE_FAIL` aborts *before* the timing stage.

A host-side replay of the kernel's own control flow (mean refinement levels, `keep_all` fraction per
arm) is printed alongside. It is **`modelled`** — it explains the shape of the measured curve, it
does not stand in for it.

---

## PROBING.md five fields

**probe** — measure the incumbent, unmodified, at the production `row_lengths` distribution, and
settle whether the sheet's ρ reweighting (implied 1.7201) and its harness→production factor (1.219)
hold. Simultaneously, stand up and validate the step-3 measurement harness.

**config** — image
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:fa63d6df`.
⚠ **the tag is mutable and was re-pushed** — the image's own `sparsecore_topk.py` is `1a89d451…`,
not the pinned `44353b34…`. The subject is therefore shipped in `subject/` and gated in-job.
`--map-dir`: `subject/`+`probe.py` → `/root/probe18`, `instruments/` → `/root/instr`,
`probes/13-level-count-refit/` → `/root/p13`. Harness = frozen `glm_dsa_block_bench.py`
(`08375d70…`, gated), `REPS_LO/HI = 20/60`, `rounds 12`. `--ctx 9216` → n = 10240, k = 2048, P = 1.
**No source is patched in this probe** — the only thing that varies across prefill arms is the
`row_lengths` operand. Verbatim RUN_CMD read-back md5 `60fcc6ade7a0687be3f02c11e0e4bd28`,
`bash -n` clean.

**job** — `j-97b98f97-142a-47f2-a173`, Succeeded. Log in `j-97b98f97-142a-47f2-a173.txt`, results in
`rho_denominator.json`. All gates green: `MAP_GATE_OK`, `SUBJECT_GATE_OK 44353b34…`,
`ANCHOR_GATE_OK`, `VERBATIM_GATE_OK`, `CTX_GATE_OK`, `RHO_GATE_OK`, `LENGTHS_GATE_OK`. The image's
own installed copy read `1a89d451…`, i.e. the tag really had been re-pushed and the pinned subject
really was the thing measured. 47 arms, 12 rounds, one pool, dispatch floor 31.96 µs/call.

**answer** — the boxed question: **275.185 µs**, `+76.60%` vs 155.82. The sheet's ρ reweighting and
its 1.219 both fail, in opposite directions and for different reasons; the sheet's ρ=1 *anchor*
also fails to reproduce. Full detail below.

**decided** — the step-3 denominators are now `measured`: prefill **291.075 µs** per kernel
invocation (gate-corrected, 3-D entry, b=1024), decode **49.171 µs** (3-D entry, b=64). Harness
(2-D) → production (3-D) is **subtract 35.563 µs**, not divide by 1.219. Every prefill `Δ time %`
on the step-2 sheet is inflated by ~1.6× and should be multiplied by **≈ 0.60** pending
re-measurement — but only after classifying the saving, because the production denominator splits
**147.925 µs ρ-independent + 112.649 µs ρ-scaled + 14.611 µs distribution**, and the two classes
reweight by **×1.55** and **×0.62** respectively. ρ is **not** a sufficient
statistic — at a fixed mean the distribution is worth up to 19.37% — so no variant may be measured
at ρ=1 and reweighted. `probes/09`'s `B = 326.782 µs` does not reproduce and must stop being
quoted.

---

## Answer

All numbers `us_median`, one pool, job `j-97b98f97-142a-47f2-a173`. Evidence class **measured**
throughout except where marked.

### The resolution floor

| | three nulls (µs) | floor | as % |
| --- | --- | --- | --- |
| prefill b=1024, identical source, identical operands | 425.206 / 428.044 / 424.825 | **3.218 µs** | 0.757% of `P3.rho1` |
| decode b=64, ditto | 52.137 / 51.949 / 52.384 | **0.435 µs** | 0.827% of `D3.rho1` |
| prefill, one executable, three score draws | 425.523 / 426.150 / 432.518 | 6.995 µs | 1.644% |

The decode floor is **four times tighter** than `instruments/README.md`'s "decode deltas below
~3.4%" — 0.83%, not 3.4%. `#dsrc_c` tripped the `suspect` flag (`us` = −76.1, an unwarmed
recompile inside one round); its `us_median` 52.384 sits inside its own family, and `us_median` is
what the floor and every result use, which is exactly the case that rule exists for.

### The boxed question, plainly

| what | µs | vs sheet |
| --- | --- | --- |
| sheet, production, **modelled** | 155.82 | — |
| **measured, ρ = 0.40005, b = 1024, 3-D entry** (`P3.prefillX`) | **275.185** | **+76.60%** |
| measured, gate-corrected per invocation (chunks 2–7) | **291.075** | **+86.80%** |
| sheet, harness, **modelled** | 189.98 | — |
| measured, ρ = 0.40005, b = 1024, 2-D entry (`P2.prefillX`) | **310.761** | **+63.58%** |
| measured, ρ = 1, 3-D (`P3.rho1`) | 425.366 | |
| measured, ρ = 1, 2-D (`P2.rho1`) | 461.021 | |

Charitably re-anchored — applying *this job's* measured ρ ratio to *the sheet's own* ρ=1 reading of
326.782, which cancels any pool-level difference — 155.82 becomes 173.427 (+11.30%) and 189.98
becomes 220.275 (+15.95%). Even at its most generous the sheet is 11–16% optimistic. It is not
generous, because the anchor itself does not reproduce (below).

### The 1.219: refuted as a factor, confirmed as a term

The 2-D − 3-D difference measured at three different ρ in one pool:

| ρ | 2-D | 3-D | relayout |
| --- | --- | --- | --- |
| 1.00000 | 461.021 | 425.366 | **+35.654** |
| 0.50005 | 328.503 | 293.044 | **+35.459** |
| 0.40005 | 310.761 | 275.185 | **+35.576** |

Spread across a 2.5× change in ρ: **0.195 µs**, sixteen times inside the 3.218 µs floor. The
relayout is a **constant additive cost of 35.563 µs** — it is the untiling `copy` of a fixed 40 MiB
buffer, and nothing about it scales with walk length. Mean 35.563 vs probes/11's 34.165 is a
1.398 µs difference, **not resolvable**: probes/11 is confirmed.

The `1.219` is that same 34.165 µs divided by a *modelled* 155.82. Divided into the **measured**
275.185 the identical term is a factor of **1.1293** (at ρ=1, 1.0838). So:

> **T_production = T_harness − 35.563 µs.  Never T_harness / 1.219.**

A ratio was the wrong container for an additive constant, and it inherited every error in the
denominator it was formed against.

### The ρ reweighting: over-credited by 10–14%

| | T(ρ=1) / T(ρ=0.40005) |
| --- | --- |
| sheet, implied `326.782 / 189.98` | 1.7201 |
| measured, 3-D entry | **1.5457** (−10.14%) |
| measured, 2-D entry | 1.4835 (−13.75%) |
| measured, 2-D, `standard_normal` scores | 1.4994 (−12.83%) |

Going to the production ρ removes **35.3%** of the kernel, not the 41.9% the per-term weights
claim. The reason is visible in the arms: **28.8–34.8% of the prefill kernel does not scale with
ρ at all.** Fitting only the degenerate arms above the `keep_all` threshold,

```
T(ρ) = 147.925 + 281.588·ρ    r² = 0.99830, max|resid| 5.523 µs   (ρ ≥ 0.25)
```

`147.925 µs` is the ρ-independent floor — the static score DMA, `zero_priv`, `fold_priv`,
`scan_glob`, `find_bucket`, the output DMA and the harness fence, none of which `walk_chunks`
bounds. **Any variant whose saving is in that 148 µs is worth the same at every ρ; any variant
whose saving is in the walk is worth 40% of its ρ=1 value.**

### The denominator, decomposed — ρ-scaled vs ρ-independent

A single scalar verdict on 1.219 is not usable by a candidate whose saving is ρ-independent. The
whole point of the fit is that the denominator splits, and the split is what decides what a saving
is worth. Using `T(ρ) = 147.925 + 281.588·ρ` (degenerate arms, ρ ≥ 0.25, r² = 0.99830, 3-D entry,
b = 1024):

| denominator | total | ρ-**independent** | ρ-**scaled** | distribution residual |
| --- | --- | --- | --- | --- |
| `P3.rho1` (what earlier probes measured) | 425.366 | 147.925 — **34.8%** | 281.588 — 66.2% | −4.147 |
| `P3.flat0.400` (degenerate at the production mean) | 260.154 | 147.925 — 56.9% | 112.635 — 43.3% | −0.406 |
| **`P3.prefillX` (production, ρ = 0.40005)** | **275.185** | **147.925 — 53.8%** | **112.649 — 40.9%** | **+14.611 — 5.3%** |
| **gate-corrected (ρ = 0.50005)** | **291.075** | **147.925 — 50.8%** | **140.808 — 48.4%** | **+2.342 — 0.8%** |

Decode, `T(ρ) = 31.235 + 21.834·ρ`: at ρ = 0.850, **31.235 µs (63.5%) is ρ-independent** and 18.559
(37.7%) scales.

Read it this way, per candidate:

- a **ρ-independent** saving (anything outside `walk_chunks` — the static score DMA, `zero_priv`,
  `fold_priv`, `scan_glob`, `find_bucket`, the output DMA, the 2-D relayout, an XLA op outside the
  kernel) keeps **100%** of its ρ = 1 µs and divides into a denominator that has **shrunk from
  425.366 to 275.185**. Its percentage therefore **grows by 1.55×** at production ρ.
- a **ρ-scaled** saving (`cvt`, `hist0`, the refinement walks, `emit` — everything bounded by
  `i_hi = min(slice_len, eff_len)`) keeps **40.0%** of its ρ = 1 µs, against a denominator that
  shrank to 64.7%. Its percentage **shrinks by 0.62×**.

That factor-of-2.5 fork between the two classes is the single most consequential number this job
produced, and no scalar reweighting can express it. **A candidate must be classified before its
number is reweighted, and the classification is "is it bounded by `walk_chunks`?".**

⚠ The 148 µs is not one op, and the fit does not itemise it. Splitting it further needs the region
deletions in `hardware_map.md`'s Stage C″ ledger — which is anchored on `B = 326.782` and is
therefore suspect (below). Treat 147.925 as measured and its internal breakdown as not.

### T(ρ) is a two-piece curve, and the knee is at ρ = k/n = 0.2

A line over all eight degenerate arms fits at r² = 0.98871 with a max residual of **18.481 µs, 5.7×
the floor** — visibly not a line. Split at the `keep_all` predicate (`total <= k`, i.e.
`eff_len <= 2048`, i.e. ρ = 0.2):

- below: slope **184.4 µs per unit ρ** (0.05 → 0.15). The refinement loop never runs.
- above: slope **281.6 µs per unit ρ**, r² = 0.99830.

A two-point ρ measurement would have drawn a straight line through this and been wrong by up to
18 µs. Decode is linear over its whole measured range: `T(ρ) = 31.235 + 21.834·ρ`, r² = 0.98266,
max residual 1.148 µs — decode never approaches the knee.

### ρ is not a sufficient statistic — the distributional assumption, priced

Four length vectors, **all with E[eff_len] = 4096**:

| arm | distribution | `keep_all` rows | µs | vs degenerate |
| --- | --- | --- | --- | --- |
| `P3.flat0.400` | all 4096 | 0% | 260.154 | — |
| `P3.prefillXsorted` | Uniform{1…8192}, **sorted** | 25% | 254.178 | −5.976 (−2.30%) |
| `P3.prefillX` | Uniform{1…8192}, **shuffled** (production) | 25% | 275.185 | +15.032 (+5.78%) |
| `P3.bimodal0.400` | {2048, 6144} | 50% | 310.542 | +50.389 (**+19.37%**) |

Spread **50.389 µs = 19.37%**, against a 3.218 µs floor. **A mean-only ρ model is not adequate**,
and this is the answer to "report the ρ effect as a function of the distributional assumption":
naming ρ = 0.40005 pins the number only to ±19%. It is pinned to ±1% only by *also* naming the
distribution, which this probe does — Uniform{1…8192}, sourced as above.

Two mechanisms, separated:

- **`keep_all`** — the fraction of rows that skip the refinement loop entirely. 0% / 25% / 50%
  across the rows above.
- **wave imbalance** — `prefillX` and `prefillXsorted` are the **identical multiset**, so ρ,
  `keep_all` fraction and total walk work are all bit-identical. They differ by **+21.008 µs
  (+8.26%)**, resolvable at 6.5× the floor. Rows map to subcores with stride 32
  (`row = w·32 + flat`) and a wave costs its **slowest** subcore, so a shuffled batch pays the
  within-wave max and a sorted one does not. **ρ cannot see this at all.**

The i.i.d. arm is the control on the stratification: `P3.prefill` (realized ρ 0.39604) reads
272.957 against `P3.prefillX`'s 275.185 at ρ 0.40005 — a 2.228 µs difference over a 0.004 ρ
difference, inside the floor. The substitution is sound.

### Which production ρ — the `lax.cond` gate, settled

`streamindex_topk.py:860-863` elides the kernel when `max(seq_lens) <= k`, so prefill chunks 0 and
1 never reach it. The eight production causal ramps price this directly:

| | µs | ρ |
| --- | --- | --- |
| `chunk0` | 137.395 | 0.05005 |
| `chunk1` | 155.966 | 0.15005 |
| `chunk2` … `chunk7` | 221.107 / 248.115 / 277.120 / 302.110 / 334.545 / 363.453 | 0.25005 … 0.75005 |

- equal-weight over all 8 chunks: **254.976 µs** (ρ = 0.40005 unconditionally)
- equal-weight over the 6 that run: **291.075 µs** (ρ = 0.50005)
- per sequence: 6 × 291.075 = **1746.5 µs**, which equals 8 × 254.976 − (137.395 + 155.966) to the
  last decimal — the two accountings are the same number, as they must be.

The single mean-matched shuffled vector `P3.gatedX` reads 293.044, **+1.968 µs** from the
chunk-weighted 291.075 — **not resolvable**. So for the gated population a mean-matched uniform
*does* reproduce the correctly-weighted denominator, even though at ρ = 0.40 it did not: the
chunk-0/1 rows are precisely the ones whose `keep_all` behaviour broke the mean-only model, and
they are gone.

**Stage C″'s inconsistency is real and it matters by 15.9 µs (5.8%):** it applies the unconditional
0.40005 to `hist0`/`emit`/`cvt` and the conditional 0.50005 only to the refinement terms. The
correct per-invocation denominator is **291.075 µs**, all terms at 0.50005.

### ⚠ probes/09's `B = 326.782 µs` does not reproduce

`N2.rho1` is probes/09's `B` reconstructed exactly — 2-D entry, `rng.standard_normal((1024,10240))`
at `SEED 20260904`, `row_lengths = full(n)`, the same pinned subject, the frozen harness.

| | µs | vs this job |
| --- | --- | --- |
| **this job, `N2.rho1`** | **453.463** | — |
| probes/01, `full`, b=1024, 2-D, ρ=1 (frozen instrument `--decomp`) | 450.260 | **+0.71%** |
| probes/09, `B` | 326.782 | **+38.77%** |

Two independent jobs agree to 0.71%; probes/09 is 38.77% away from both. Decode corroborates:
`D3.rho1` 52.567 against probes/01's b=64 `full` 51.658, **+1.76%**. `sparsecore_topk` is
`@functools.partial(jax.jit, static_argnames=("k", …))`, so probes/09's un-wrapped call and this
job's outer-jit call compile the same program — the gap is not a jit-vs-eager artifact and this
probe does not explain it.

**Consequence: `B = 326.782` is the ρ=1 anchor of the entire Stage C″ ledger** (`hardware_map.md`
:480-500, whose terms sum to 332.082, "+1.62% vs measured `B`"). If the anchor is 38.77% low, that
ledger's *absolute* µs terms are all suspect, its internal consistency notwithstanding — a set of
terms fitted to a wrong total will sum to that wrong total. This probe does not adjudicate which
job is right; it reports that two of three agree and names the odd one out. **Stop quoting
326.782 as a measured absolute** until it is re-run.

### Decode

`D3.rho1` 52.567 → `D3.decodeX` 49.171 at ρ = 0.850: **−3.396 µs, −6.46%**, resolvable at 7.8× the
0.435 µs floor. `D3.decode` (i.i.d., ρ 0.84844) 49.698 and `D3.flat0.850` (degenerate) 49.064
bracket it within 0.63 µs — **at decode the distribution does not matter**, because every row is
far above the `keep_all` knee and the length range is only 1023 wide. Decode's 2-D − 3-D relayout
is +2.483 µs (52.513 − 50.030), two orders below prefill's, as expected for a 2.5 MiB buffer.

So a decode number measured at ρ = 1 is only 6.5% pessimistic; a prefill number measured at ρ = 1
is **54.6%** pessimistic (425.366 / 275.185).

---

## What a variant agent needs to know

*(For forwarding verbatim.)*

**⛔ 0. The bracket, before anything else: these arms time the WHOLE `sparsecore_topk(...)` call,
not `pl.kernel`.** Each arm is one `jax.jit` whose parameter is the production
`f32[1024, 80, 128]{2,1,0:T(8,128)}` and whose body is the entire call; everything XLA inserts
inside that program is counted. It is **not** `--decomp`'s `stage1` / `full`, which bracket the
kernel. That is deliberate: this exercise's candidates do not live at one boundary — some are
inside the kernel, one is at the output edge, and at least one deletes a TensorCore fusion
**outside** the kernel call entirely. A kernel-bracketed arm scores that last class as a flat null,
and the null is an instrument artifact, not a result.

So the ranking job must report **two** numbers per arm — inside-kernel and whole-call — and each
candidate must name which one its prize is quoted against. **A candidate whose two numbers disagree
is telling you where its effect lives, and that disagreement is the finding.** `probe.py` as it
stands supplies only the whole-call number; adding the inside-kernel bracket is a change to STAGE 2
(`reg(...)`), not to the pool, the gates or the nulls.

⚠ **Unresolved, and it is the difference between rank 2 and nothing:** whether a given
outside-the-kernel op is *inside* the program these arms time depends on whether it sits between
the jit parameter and the call, or upstream in the model's scoring code. If it is inside, the
saving divides into 275.185 / 291.075 directly. If it is upstream, it is **not** in these
denominators and the correct denominator is larger. Settle it by looking for the op in the
optimized HLO of `P3.prefillX` — do not assume either way. For scale: a 26.251 µs deletion is
**−9.54%** against 275.185, **−9.02%** against the gate-corrected 291.075, and **−16.85%** against
the sheet's modelled 155.82. The sheet nearly doubles it.

**1. The denominators are now `measured`. Use these, and say which one.** All `us_median`, one
pool, job `j-97b98f97-142a-47f2-a173`, subject md5 `44353b34…`, `--ctx 9216`, n = 10240, k = 2048,
P = 1.

| shape | entry | ρ | µs | floor |
| --- | --- | --- | --- | --- |
| **prefill b=1024, per kernel invocation** | 3-D | 0.50005 (gate-corrected) | **291.075** | 3.218 (0.757%) |
| prefill b=1024, unconditional | 3-D | 0.40005 | 275.185 | 3.218 |
| prefill b=1024, harness entry | 2-D | 0.40005 | 310.761 | 3.218 |
| prefill b=1024, ρ=1 (what earlier probes measured) | 3-D | 1 | 425.366 | 3.218 |
| **decode b=64** | 3-D | 0.850 | **49.171** | 0.435 (0.827%) |
| decode b=64, ρ=1 | 3-D | 1 | 52.567 | 0.435 |

Use 291.075 for anything that only fires when the kernel runs. Use 275.185 if your change also
alters chunks 0–1 — but it can't, because the kernel is not called there.

**2. Do not divide by 1.219; subtract 35.563 µs.** The harness→production relayout is a constant
additive term (35.654 / 35.576 / 35.459 µs at ρ = 1 / 0.40005 / 0.50005, spread 0.195 µs). It is
the same in both arms of any A/B, so it **cancels out of a saving in µs and inflates nothing but
the percentage**. Report your saving in µs first.

**3. Every prefill `Δ time %` on the step-2 sheet is inflated by ~1.6×.** Two compounding errors:
the denominator was 155.82 where the measurement is 275.185 (×1.766) or 291.075 (×1.868), and the
numerators were ρ-scaled by an implied 1.7201 where the measured whole-kernel factor is 1.5457
(×1.113 the other way). Net: multiply a sheet prefill percentage by **≈ 0.630** (unconditional) or
**≈ 0.596** (gate-corrected). ⚠ That is a *bound for triage*, valid only if the term scales like
the whole kernel. **It is not a substitute for re-measuring**, because of point 4. Decode
percentages quoted at ρ = 1 scale by **×1.069**.

**4. ⛔ You may not measure at ρ = 1 and reweight.** At a fixed mean of 4096 the measured cost
spans 254.178 → 310.542 µs (**19.37%**) depending only on the length *distribution*, and 21.008 µs
(8.26%) of that is row *arrangement* at an **identical multiset** — rows map to subcores with
stride 32 and a wave costs its slowest subcore. ρ cannot see either. Measure at the production
length vector: `strat_uniform(1, 8192, 1024, seed)` permuted, or `strat_uniform(2049, 8192, 1024,
seed)` for the gated population. `probe.py` builds both.

**5. Classify your saving before you reweight it — the fork is 2.5×.**
`T(ρ) = 147.925 + 281.588·ρ` for ρ ≥ 0.25 (r² = 0.99830, 3-D, b=1024). At the production
ρ = 0.40005 the 275.185 µs splits **147.925 ρ-independent (53.8%) + 112.649 ρ-scaled (40.9%) +
14.611 distribution (5.3%)**; at the gate-corrected ρ = 0.50005 the 291.075 splits
**147.925 (50.8%) + 140.808 (48.4%) + 2.342 (0.8%)**. Decode at ρ = 0.850: **31.235 (63.5%)
ρ-independent + 18.559 (37.7%)**.

Only `cvt`, `hist0`, the refinement walks and `emit` are bounded by
`walk_chunks = ceil(min(slice_len, eff_len)/16)`. The score DMA is `pl.ds(s, slice_len)`, a
**static** length; `zero_priv` / `fold_priv` / `scan_glob` / `find_bucket` / the output DMA / the
2-D relayout / anything XLA does outside the kernel are all fixed. Therefore:

| your saving is | keeps, at production ρ | its **Δ time %** vs its ρ=1 % |
| --- | --- | --- |
| **ρ-independent** (dense, outside `walk_chunks`) | **100%** of its µs | **×1.55 — it grows** |
| **ρ-scaled** (inside the walk) | **40.0%** of its µs | **×0.62 — it shrinks** |

A dense op deleted outside the kernel is the best case in this table: its numerator holds while the
denominator shrinks by 35%. An in-walk saving is the worst. **Do not apply a single reweighting to
both.** The 148 µs is measured; its internal itemisation is not (see point 12).

**6. The knee is at ρ = k/n = 0.2, not anywhere you'd guess.** `keep_all = total <= k` deletes the
whole refinement loop below it; the slope is 184.4 µs/ρ below and 281.6 µs/ρ above. 25% of
production prefill rows are below it. Two ρ points cannot see this.

**7. `row_lengths` is the only way length reaches the kernel, and a broken one is silent.** Every
prefill arm here is the **same executable** with a different `lengths` operand — no source edit, no
recompile. If `row_lengths` fails to reach the kernel, all arms are byte-identical and all are
"correct". STAGE 1 therefore asserts, per arm, that every live index is `< eff_len`, that
`#live == min(k, eff_len)`, and multiset-equality against a numpy oracle restricted to
`[0, eff_len)` — **plus** a negative control that scores the ρ=1 output against the ρ=0.4 lengths
and must go red (it did: 1024 out-of-range, 257 wrong live counts) and an `array_equal` that must
be `False`. Keep both when you add an arm.

**8. Read `us_median`, and expect one `suspect` row.** `#dsrc_c` reported `us = −76.1` — an
unwarmed recompile landing inside one round — while its `us_median` (52.384) sat inside its own
null family. `us` is not usable at decode scale; `us_median` is.

**9. ⛔ Do not pool `b=16` with `b=64`/`b=1024`.** At `--ctx 9216`, `b=16` resolves to **P = 2**
while both production shapes are P = 1 — a different kernel, with a stage-1/stage-2 seam that does
not exist at P = 1. Any arm summary spanning P = 1 and P > 1 sums two kernels and can average a win
against a loss into a null. `probe.py` asserts `_pick_partition(b, n, k) == 1` at both shapes for
exactly this reason. If budget forces a shape to be dropped, keep `b = 1024`.

**10. Arm activation.** For a variant that is the whole file with no env var and no flag, the in-job
md5 gate is complete activation evidence — nothing else can be running. Beyond that, two patterns,
and they need opposite profile checks:

- a variant that **renames its own Pallas scope** (`sc_topk_direct_b{b}_n{n}_k{k}_f32in`) is the
  strong case: the scope name proves the body *ran*, where an env var only proves it was *set*.
  Match on the name.
- a variant that **deliberately keeps the scope name identical** so op-name matching survives the
  A/B cannot be checked that way. Match on the **result shape** instead — e.g.
  `s32[1024,2048]{1,0:T(8,128)}` replacing `s32[2097152]{0:T(1024)}`.

Say which of the two you are, in the arm brief. Getting it backwards produces a confident
activation claim from a check that could not have failed.

**11. Host-side compiles serialise on `/tmp/libtpu_lockfile`.** Concurrent agents AOT-compiling on
the same host hit `ABORTED: Internal error when accessing libtpu multi-process lockfile`. Retry
with backoff; **do not delete the lockfile.** Irrelevant inside a pod job, relevant to any
harness validation you run locally.

**12. ⚠ Cross-pool absolutes do not reconcile — never compare against a number from another job.**
This job's `N2.rho1` reproduces probes/01 to +0.71% and probes/09's `B` to +38.77%. Every
comparison you need must live in **one pool, one job**, with its own ρ=1 control if you want to
speak to a prior number.
