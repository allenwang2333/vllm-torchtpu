# Probe 16 — `num-buckets-scaling`: what does `fixed(NB)` really cost, and does `v5` survive?

## Answer

**1. The fitted line.** Three measured points, one pool, 12 interleaved rounds
(`j-c47cfec1-bf7f-4ddf-b340.txt:190-192`):

| `NUM_BUCKETS` | one-level fixed sweep `E0 − E0-swp`, µs | Δ time % vs NB=256 |
| ---: | ---: | ---: |
| 64 | **5.375** | **−71.66%** |
| 256 | **18.967** | 0.00% |
| 512 | **38.845** | **+104.80%** |

`fixed(NB) = 0.299 + 0.074870 · NB`, **r² = 0.99934**. The intercept is **c = 0.299 µs = 1.6%** of
the NB=256 sweep — far **below** probes/13's 11.12 µs break-even. **Probe 13's modelled
`fixed(NB) = 19.192·NB/256` (zero intercept) is essentially right**: pure proportionality predicts
4.798 / 19.192 / 38.384 against measured 5.375 / 18.967 / 38.845.

**2. `v5` does not survive anyway — the model omits a term that this job measured.** The same pool
priced the *rest* of the L0 kernel (`E0-swp`, everything but the sweep: DMA + `cvt` + the dense
`hist0` scatter + fixed overhead) and it is **strongly NB-dependent in the opposite direction**
(`:129-152`, the `E0-swp` rows): **259.647 / 206.227 / 177.546 µs** at NB = 64 / 256 / 512. Fewer buckets means more
`addupdate_scatter` address collisions, so the dense L0 histogram costs **+53.420 µs at NB=64** and
**−28.681 µs at NB=512** (ρ=1). `v5`'s `T(NB)` treats that as a constant.

| | briefed model (hist0 constant) | **corrected** (add measured Δhist0 × ρ) |
| --- | ---: | ---: |
| NB=64 | −32.05% | **−5.28% ± 11.19%** → **unresolved, not a win** |
| NB=512 | +48.13% | **+33.75%** → clear loss |

Propagating the 1.859 µs floor through both reads gives ±8.94 µs on a T₂₅₆ of 79.827 µs. **`v5` is
REFUTED as a −14.82% candidate.** ⚠ **Do not cite the job log's own
`V5_VERDICT: NB=64 SURVIVES (-32.04%)` (`:208`)** — the probe computed it under the briefed model,
which holds `hist0` constant. This job's own `E0-swp` absolutes falsify that premise; the corrected
row above supersedes it. The mechanism it rests on is real and larger than modelled; it is
cancelled by a mechanism it does not contain. Two further terms bias the same way and were not
measured: the refinement walk's match density is `1/NB`, so those scatters also get denser at NB=64
(`probes/03`: dense 116.350 vs sparse 35.243 µs/pass — a 3.3× range on density alone), and
`levels_crit(64) = 2.375` is a CPU replay, not a reading.

**3. The 07-vs-09 conflict: `probes/09` is right.** MEASURED **18.967 µs** — **+0.54%** from
probes/09's 18.866, **−67.39%** from probes/07's implied 58.17 µs total (3.031 execs × 19.192)
(`:173-179`). This arm is **byte-identical** to probes/09's (`P09_SRC_GATE`, six md5s, `:82-93`),
so it re-measures 09's construction in a fresh pool 9 jobs later and reproduces it inside the floor.
**07's "19.192 µs/level" is not the sweep**: deleting a whole level removes the sweep *and*
everything else that is per-level, so it is a marginal per-level cost that happens to be numerically
close to the sweep's *total*. `hardware_map.md:164` and `:97` should not be read as pricing the same
thing. **`fixed` is ~18.9 µs total, ~6.2 µs per execution at ρ=1** — not 19.192 per level.

**4. Is one `fixed(NB)` scalar the right model? Marginally.** The four primitives do *not* scale
identically (`:181-186`), but three of them fit their own trip counts well:

| primitive | trip count | 64 | 256 | 512 | fit vs own trips | r² |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `zero_priv` | `NB` | 2.368 | **9.927** | 20.234 | −0.223 + 0.03990/trip | 1.0000 |
| `fold_priv` | `NB//16` | 1.770 | **5.532** | 11.233 | +0.303 + 0.33897/trip | 0.9987 |
| `scan_glob` | `NB//16` | 1.476 | **3.930** | 7.147 | +0.675 + 0.20247/trip | 1.0000 |
| `find_bucket` | `NB//16` | 0.047 | **−1.385** | 3.634 | −1.629 + 0.13813/trip | 0.5633 |

`zero_priv` is 52% of the sweep and scales **exactly** with `NB` (r²=1.0000, intercept −2% of its
own cost) — the dense 256-bucket clear is the whole story, confirming `hardware_map.md:487-489`.
**`find_bucket` is unresolved at every NB** (|value| ≤ the 1.859 µs floor at NB=64 and 256; it even
reads negative at 256, −215% from probes/09's 1.200). Report the split as **three primitives plus
noise**, not four. `scan_glob` carries a real 0.675 µs intercept (17% of its NB=256 cost) — the only
primitive with one — but it is too small to matter and is itself near the floor.

**5. Forward pointer (MODELLED).** NB=512 buys a 28.681 µs cheaper histogram and pays 19.878 µs more
sweep *per level*. Under D5 (drop the 16× lane replication, so `zero_priv(NB//16)` and `fold_priv`
over `range(1)` — `hardware_map.md:556`), the sweep at NB=512 falls to ~12.7 µs and T(512) ≈ 42.2 vs
T(256) ≈ 40.7 and T(64) ≈ 63.5. **The radix still does not want changing, in either direction.** Do
D5 first; it dominates the radix as a lever and is already written.

## Arms

One pool, 22 thunks, 12 interleaved rounds, shuffled per round, every arm warmed, `us_median`.
b=1024, n=10240, k=2048, P=1, ρ=1, entry `f32[1024,80,128]`, production score distribution
(`probes/13/level_sim.py` STEP 4b, verbatim — `VERBATIM_GATE_OK`, `:20`).

| arm | drops | 64 | 256 | 512 |
| --- | --- | ---: | ---: | ---: |
| `E0` | `emit`,`fill` (`REFINEMENT_LEVELS=()`) | 265.021 | 225.195 | 216.392 |
| `E0-zero` | + `zero_priv` | 262.653 | 215.268 | 196.158 |
| `E0-fold` | + `fold_priv` | 263.252 | 219.662 | 205.159 |
| `E0-scan` | + `scan_glob` | 263.546 | 221.265 | 209.245 |
| `E0-find` | + `find_bucket` | 264.975 | 226.580 | 212.758 |
| `E0-swp` | + all four | 259.647 | 206.227 | 177.546 |
| `E0-2d@256` | `E0@256` under an `f32[b,n]` jit entry | — | 258.188 | — |
| `#null_a/b/c` | three copies of one arm | 18.945 / 19.460 / 17.601 | | |

**Resolution floor = 1.859 µs** (max unordered pairwise null spread). Every delta above is quoted
against it. Additivity `sum/joint` = 1.053 / 0.949 / 1.088 (`:158-164`) — looser than probes/09's
0.998, driven entirely by `find_bucket`; dropping it gives 1.045 / 1.022 / 0.994.

**Entry shape.** `E0-2d@256 − E0@256 = +32.994 µs` — the 3-D entry does not pay it. The HLO gate
confirms the mechanism directly: the 3-D entry has **0** `sparse-core-data-format-call` and **0**
`copy` ops; the 2-D entry has **8** df-calls and a `ROOT %copy.3 = s32[128,80,8,128]` (`:102-104`,
`hlo-E0at256-3d.txt`, `hlo-E0at256-2d.txt`). **32.994 is 3.4% from probes/11's *firm* leg of
−34.165 µs and 45% from its retracted −60.416** — an independent confirmation of
`probes/11/README.md:67`. The `fixed(NB)` split is a within-NB difference, so this cancels regardless.

## PROBING.md five fields

**probe** — measure `fixed(NB)` at NB ∈ {64, 512} with NB=256 as an in-pool control, by re-running
probes/09's four-way sweep split at three radices; fit `fixed(NB) = c + m·NB`; settle `v5`.

**config** — image `…vllm-torchtpu-hning_google_com:fa63d6df`. ⚠ **the tag was re-pushed**: the
image's `sparsecore_topk.py` is now `1a89d451…`, not the pinned `44353b34…` that probes/07-09
measured. The subject is therefore shipped in `subject/` and gated in-job (`SUBJECT_GATE_OK`, `:59`), and the installed copy is printed beside it. Harness = the frozen
`instruments/glm_dsa_block_bench.py` (md5 `08375d70…`, gated). Radix retuned **consistently** —
`NUM_BUCKETS`, `L0_SHIFT` and `REFINEMENT_LEVELS` move together, with `bits_covered == 32` asserted
per NB (`:21-23`): NB=64 → shift 26, 5 levels; NB=512 → shift 23, 3 levels.

**job** — `j-c47cfec1-bf7f-4ddf-b340` (Succeeded). Two earlier submissions failed and are recorded:
`j-8c6d3032-e514-4ed9-9d8b` (`ImagePullBackOff` — the full-sha tag no longer exists in the registry)
and `j-a70d48d9-8963-4c79-aa95` (the md5 gate correctly refused the re-pushed image's subject).

**answer** — see above. All three radices **lower and are bit-exact** (`CORRECTNESS_GATE_OK`, `:67-71`): 0/1024 rows wrong at NB=64, 256 and 512 against a numpy `argpartition` oracle on selected
score multisets, with a negative control (levels removed) that fails 1024/1024. **There is no hard
correctness or lowering constraint on `v5`** — the constraint is performance.

**decided** — `v5` (NB=64) is **REFUTED**: −5.28% ± 11.19% modelled on measured inputs, against its
claimed −14.82%. NB=512 is a **clear loss** (+33.75%). `probes/07`'s 19.192 µs/level must not be
cited as the sweep price; `probes/09`'s 18.866 is confirmed at 18.967. The next lever on this region
is **D5**, not the radix.

## What did not work, and one gate that cannot fail

- **STAGE 4 (patch evidence by output) is a null test by construction and should be ignored**
  (`:106-127`). Every `E0` arm drops `emit`, so nothing is ever written to `rowbuf`; all 18 arms
  return the same degenerate output and all 15 "differ from E0" checks report `patch NOT effective`.
  A gate that cannot fail is not a gate. The patches *are* proven, by `src_md5` + `changed_lines`
  in STAGE 2 / the dry run (`:24-50`, `:76-93`) and by the timings themselves — `E0-swp@512` is 38.8 µs below
  `E0@512`, 21× the floor.
- **`find_bucket` did not resolve** at any NB. Its 1.200 µs in probes/09 is below this pool's floor
  and should not be carried as a measured number.
- **The 2-node/priority confounds do not apply** (single 4-chip pod, one job, one pool), but note
  the whole pool ran on one slice, so slice identity is a constant here, not a control.

## Addendum — the `hist0` provenance question, and a refutation of probes/15's bug claim

**Provenance.** The stage C″ sheet's `hist0 = 33.91 µs (17.8%)` (`hardware_map.md:504`) is **not**
traceable to `instruments/sctopk_2stage_bench.py`. It is `84.762 × ρ = 84.762 × 0.40005 = 33.909`,
where 84.762 comes from **probes/08**'s `-sweep` arm by subtraction (`hardware_map.md:491-494`:
`hist0 + emit_degenerate + fill_spurious = 84.762`, so `hist0 ≤ 84.76`), corroborated at 85.12 by
stage C's independent per-pass derivation. Both rows are already tagged `modelled`. The instrument's
`scatter` arm feeds a *different* row — the 116.350 µs at `hardware_map.md:97` and `:291`, from
**probes/03** — which is a per-pass price, not the C″ production row.

**The bug claim is false.** `probes/15/README.md:85-88` asserts that
`instruments/sctopk_2stage_bench.py:856`'s `m = key > jnp.int32(KEY_NEG_INF)` (probes/15 cites
`:851`; in the frozen file, md5 `08375d70…`, it is `:856`) "excludes every negative score". It does not. `KEY_NEG_INF` is **not** the raw −inf bit pattern; `sparsecore_topk.py:51`
defines it as `BITS_NEG_INF ^ 0x7FFFFFFF` = `0x807FFFFF` = **−2139095041**, the *monotone* key of
−inf, which sits at the **bottom** of the int32 range. A raw-bits negative float such as −1.0 is
`0xBF800000` = −1082130432, which is **greater** than −2139095041, so the mask is **true**. Measured
on 655360 standard normals: raw-bits density **1.000000**, monotone density **1.000000**. The mask
is false only for `-0.0` and negative subnormals (`|x| < 2^-126`), which never occur here. **The arm
is ~100% dense as labelled.**

The second reason given is also wrong: `passes()` calls `n = topk_n(ctx)`
(`sctopk_2stage_bench.py:122-129`), and probes/03 ran `--ctx 9216`, whose log line reads
`=== PASSES  b=1024 n=10240 P=1 n_p=10240 slice_len=10240 waves=32 ===`
(`probes/03-passes-9216/j-fa60ff2a-c793-4f32-9576.txt:9,11`). **9216 is the max-model-len, not `n`.**
So probes/03's 116.350 and probes/15's 108.217 **are** comparable — same `n`, both genuinely dense —
and they differ by **−6.99%**, which is an ordinary cross-pool reproduction, not a 50%-density error.
Nothing in the C″ sheet is understated by ~2×, and **`v10` does not gain anything from this.**

Two real differences remain, both small and neither a density error: the instrument buckets on
`(raw >> 24) & 255` while the kernel buckets on `(monotone >> 24) + 128` — a different (still highly
non-uniform) bucket histogram — and the instrument omits the `j < i_hi` bound term.

**Dense-vs-half-density contrast: not measured, and no longer the priority.** My pool runs a single
density (ρ=1, production distribution), so the contrast is not available in `j-c47cfec1`. It would
need a new job and **none was opened**. What this job *does* add on the same region is stronger: the
dense L0 histogram's cost is **not** constant in `NUM_BUCKETS` — anchored on probes/03's 116.350 at
NB=256, it is **169.770 µs (1.459×) at NB=64** and **87.669 µs (0.753×) at NB=512** (from `E0-swp`,
`:129-152`). That is measured in one pool and is what refutes `v5` above.
