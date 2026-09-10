# Ranking job — accumulated binding constraints

Not yet built; the variant set is not frozen. This is the running spec, so that nothing arriving in
a coordinator message between now and the freeze is lost. Everything here is binding unless marked
`open`.

## Brackets — two per arm

Every arm reports **both**:

- **whole-call** — the `jax.jit` program around `sparsecore_topk(...)`, at the production 3-D entry
  `f32[b, n/128, 128]`. This is `probes/18`'s `P3.*` / `D3.*` bracket.
- **inside-kernel** — the `pl.kernel` invocation only.

A candidate whose two numbers disagree is telling us where its effect lives. Per candidate:

| candidate | prize is quoted against | why |
| --- | --- | --- |
| `v3`, `v10`, `v2` | inside-kernel | changes are inside the kernel body |
| `v11` | whole-call | output edge; the retiled emit changes the result buffer's layout |
| `v4` (all rungs) | inside-kernel | the DMA is inside the kernel |
| **`v12`** | **whole-call** | `probes/20`: it deletes a priced op in ENTRY, *outside* the kernel call. Inside-kernel reads it as a flat null — an instrument artifact, not a result. |

`probes/20` settled that the production 3-D whole-call bracket is sufficient for `v12`; the 2-D
harness entry is **not** needed and must not be used, since it adds a 35.576 µs
`OFFLOAD_DATA_FORMATTING` call production never pays.

## ρ — measured at production, never reweighted

`probes/18`: ρ is **not a sufficient statistic** (19.37% span at a fixed mean; 8.26% from row
*arrangement* alone at an identical multiset), and the ρ-independent / in-walk fork is 2.5× wide. So
every ranked arm is measured at the production ρ **and** the production arrangement directly:

- prefill `b = 1024`: `prefillX` — stratified Uniform{1..8192}, permuted, ρ = 0.40005 exact.
- decode `b = 64`: `decodeX` — stratified Uniform{8193..9215}, ρ = 0.850 exact.
- `n = 10240`, `k = 2048`, `P = 1`, **`--ctx 9216` always**.

⚠ **`v4-ragged` is a null at ρ = 1 by construction.** If only one ρ is affordable it is 0.40005 —
this is now the general rule for the `v4` ladder, and it is why no arm may be scored from a ρ=1 run
and rescaled.

**Named ρ=1 exception cell:** `v2` at ρ = 1, explicitly requested. It separates two hypotheses 3×
apart (−8% if the 12.195 µs saving scales per executed histogram level, −2.7% if it is level-0 only)
and decides whether `v2`'s production figure is its floor or its central estimate. It is an
**explicit named arm**, not something inferred from the ρ ladder.

⚠ **`v2`'s two production percentages do not reconcile with the fork and one of the reweightings is
wrong.** Its saving is bounded by `i_hi`, so it is *in-walk* and takes the ×0.62 branch. But
−8% → −7.83% is ×0.98 and −2.7% → −15.9% is ×5.9; neither is 0.62. **The cell reports µs**, and every
percentage is re-derived by me, in one place, against 275.185 µs (`P3.prefillX`) and 291.075 µs
(gate-corrected), exactly as `probes/18` did.

**`V2_BIGSCRATCH` gets a pool slot.** SparseCore scratch reclamation is invisible statically — the
XLA memory report is blind to Mosaic VMEM, and `probes/20` confirms `memory_analysis()` sees only
XLA-level temp. A measured arm is the only route to that number.

## Pooling

- **One pool, interleaved rounds ≥ 9, arms reshuffled per round, read `us_median` never `us`.**
- Every thunk warmed before it is timed. A cold thunk's paired slope is minus its compile time.
- **≥ 3 null arms**, identical source, separately staged. Null spread is the *resolution floor*; a
  result inside it is `not resolvable`, never "no effect". `probes/18` floors: prefill 3.218 µs
  (0.757%), decode 0.435 µs (0.827%).
- ⛔ **Do not pool `b = 16` with `b = 64` and `b = 1024`.** At `--ctx 9216`, `b = 16` is `P = 2` while
  both production shapes are `P = 1`, and `v11` is predicted to move them in opposite directions.
  If a batch must be dropped for budget, keep `b = 1024`.
- Cross-pool absolutes never reconcile. Only ratios to an in-pool control cross a pool boundary.

## ⛔ Open conflict: `v4`'s env-at-import arms vs one pool

`v4` selects its chunk value and its `nopipe` / `SCTOPK_V4_RAGGED_ONLY` rungs by **env read at module
import**, so each needs its own process. But a per-process arm cannot share a pool with the arms it
must be subtracted from — and the whole value of the three-rung ladder is that the pipelining
increment becomes `v4-c1024 − v4-ragged`, *a difference between two measured arms at the same ρ*.
Across two processes that difference is a cross-pool subtraction, which is exactly the operation
`probes/18` item 12 forbids.

Two ways out, in order of preference:

1. **Ask the `v4` builder to ship each rung as a separate file** (three subject md5s, mapped in),
   not as one file with an import-time env switch. Then all rungs are distinct modules in one
   process and one pool, and the ladder subtracts cleanly. This costs the builder a `cp` and buys
   the ladder its entire point.
2. If (1) is refused: run one process per rung, each with an **in-pool incumbent control at the same
   ρ**, and quote every rung only as a ratio to its own control. This loses the direct µs
   difference and roughly doubles the pool cost.

**Requesting (1).** Until it is resolved the `v4` ladder cannot be scored as a ladder.

`v4` rung status after `probes/19`: `tc = 1024` pipelining is refuted (−14.35 µs net); **`tc = 2048`
is the primary chunk arm** and is a wash (−0.44 µs, 3.3% past break-even) — undecided, so it is
measured, not adjudicated; `ragged-blocking-only` (md5 `5802c56b9608976ff447be57abc3eb72`, scope
`…_v4ragged`) is insensitive to `g`, carries two thirds of the prize, and compiles to *exactly* the
incumbent's instruction stream (1168 TEC bundles at b=1024, 1166 at b=64, delta zero at both), which
makes it the cleanest isolator in the wave. `descriptor_cost.py` is no longer load-bearing for the
verdict, only for whether to bother.

## ⛔ Correctness gates are per-variant, not one gate

**A byte-compare gate reused from `v2` will fail a correct `v3`.** `v3` is exact but emits certain
winners before tie-broken ones: the winner *set* matches, the slot *order* does not. The frozen
suite is fine with this (`_mismatch` compares score multisets); a bit-identical gate is not. `v2`
passed a bit-identical gate and that is its strongest evidence — **do not generalise it.**

| variant | gate | activation tell |
| --- | --- | --- |
| `v2` | **bit-exact** (it claims bit-exactness) | md5 + `V2_BIGSCRATCH` arm |
| `v3` | **multiset / set equality only** | scope name |
| `v4` (3 rungs) | multiset | scope `…_v4ragged` etc.; md5 per rung |
| `v10` | `open` — ask the builder what it claims | `open` |
| `v11` | multiset (retiled emit) | ⚠ scope name is **identical in both arms** by design; match on **result shape** `s32[1024,2048]{1,0:T(8,128)}` replacing `s32[2097152]{0:T(1024)}`. md5 `3dc249276fa68985e29f4da8e9c27fbd`. |
| `v12` | bit-exact *should* hold (a bit round trip, no arithmetic) but is **unverified**; gate it as bit-exact and treat a failure as a real finding | scope `…_f32in`, **confirmed present at the production 3-D entry** (`probes/20`); plus `temp_size_in_bytes` differing by exactly `b·n·4`, free from `memory_analysis()`. md5 `408ef72e9afc935c928f2803764878cc`. |

Control md5 for every arm: `44353b347c96e0eedbd742c710d4540a`.

`v11` and `v12` need **no checksum arm and no rival-constant arm** — neither has an env var, a flag
or a tunable, so the in-job md5 gate is complete arm-activation evidence.

### One added correctness id: `all_negative` at `P = 1`

⛔ **An earlier revision of this section said green 68/68 does not clear a `cnt`-bound overflow at
`P = 1`, and called that "worse than a blind spot". That is RETRACTED. The suite does catch it.**
The claim came to me from the coordinator, who has since retracted it; I confirmed the wrong version
from source and made it stronger, which is the failure worth recording. Both errors trace to the
same two causes, now standing method rules:

- **Transcribe a score generator, never paraphrase it.** The measurement behind the original claim
  used approximated generators; two of five were materially wrong (its `strict_desc` was positive
  where the real one, `-np.arange(n)`, is all negative).
- **Confirm a test id with `pytest --collect-only`, not by reading `parametrize` decorators.** Both
  the coordinator and I read decorators. Collection is 1.2 s and needs no TPU.

**What the suite actually covers.** Max `cnt` at ρ = 1, from verbatim generators:

| kind | max `cnt` | runs at `P = 1`? |
| --- | ---: | --- |
| `strict_desc` | 1536 | yes |
| `random` | 3054 | yes |
| **`heavy_ties`** | **5526** | **yes** |
| `all_negative` | 7089 | no — `P > 1` only |

`heavy_ties` reaches 5526 at `p1-b1024-n10240` and **overflows a 4096-word buffer by 1.35×**. So a
variant built to `design.md`'s stated `cnt_max = 4096` goes **red**, on a real collected id, and does
not ship past a green suite.

Verified by collection, not by decorator: **68 ids, 12 of them at `P = 1`.**
`test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]` and `[heavy_ties-p1-b64-n10240]` both exist.
`all_negative` appears in exactly 5 ids, **0** at `P = 1`.

**The residual gap is real but narrow, and it is two things:**

1. `P = 1` `cnt` coverage rests on **one** kind with **1.35×** of margin. A per-bucket-capacity bug
   modestly less bad than the design's stated bound still ships silently.
2. **`P = 1` has no coverage of negative winners at all.** `all_negative` is the only kind whose
   *winners* are negative, hence the only one where `monotone_key`'s involution does anything at
   emit time, and it is `P > 1` only. `strict_desc` is all-negative in its *scores*
   (`-np.arange(n)`) but degenerate-ordered in its behaviour, and does not exercise this.

**Fix: add `all_negative` to `test_exact_topk_full_rows`'s kind list.** One word; closes both gaps.

⚠ **Not applied to `instruments/`, which is read-only** — along with the live paths it was frozen
from. The added ids go in the ranking job's own test file, importing the frozen helpers.

⚠ **Cost note:** `test_exact_topk_full_rows` is parametrized over all 7 `SHAPES`, so adding one kind
adds **7** ids, 5 of them at the expensive `N_128K` shapes. If runtime is tight, restrict the added
kind to the two `P = 1` shapes — 2 ids, which is where both residual gaps live.

The added ids **report measured `cnt` per kind** rather than asserting a mechanism. Given that a
paraphrased generator is what produced the retracted table, that reported number is now the most
useful thing they do.

## Ranking order

`v3` is **not measured first** (builder's request, agreed): predicted to lose (+2.27% … +7.71%
production, `modelled`) on a structural argument — its fused pass cannot exist, because `b_star` is
the *output* of the histogram it wanted to ride. It is still built, measured and reported as a
loser, not deleted.

Working order: `v11`, `v12`, `v4-ragged`, `v4-c2048`, `v2` (+ `v2`@ρ=1, `V2_BIGSCRATCH`), `v10`,
`v3` last.

## Denominators — the only ones that may be used

| denominator | value | source |
| --- | --- | --- |
| prefill, unconditional | **275.185 µs** | `probes/18` `P3.prefillX`, measured |
| prefill, gate-corrected per invocation | **291.075 µs** | `probes/18`, measured |
| decode | **49.171 µs** | `probes/18` `D3.decodeX`, measured |

⛔ **155.82 µs is modelled and must not be used.** A 26.251 µs deletion reads −9.54% against 275.185
and −16.85% against 155.82 — nearly double. ⛔ Every `probes/09`-derived number is unsafe to quote:
its `B = 326.782 µs` does not reproduce (453.463, +38.77%, two other jobs agreeing against it) and
all 12 of its arms carry `REFINEMENT_LEVELS = ()`, so `hardware_map.md:493`'s "all levels" row can
only have measured level 0.

## Platform constraints to design around

- A dynamic **size** in `pl.ds` compiles; a dynamic **start** at an arbitrary offset **fails** —
  `Slice offset for 32-bit 1D memrefs must be a multiple of 8`. That is what makes
  `align(i_hi, LANES)` mandatory rather than tidy.
- `/tmp/libtpu_lockfile` serialises host AOT compiles across concurrent agents; two sibling runs
  died with `ABORTED: Internal error when accessing libtpu multi-process lockfile`. Retry with
  backoff in `RUN_CMD`; **do not remove the lockfile.** (The 3-attempt/60 s loop shipped in
  `probes/19` did not fire but is the pattern.)
- Docker tags are mutable — `fa63d6df` was re-pushed mid-exercise. **Gate the subject md5 in-job**,
  asserted inside the job script, not in the submitter.
- `cdk` jobs tagged `-t agent-jobs,measurer`. Never delete a job I did not create. A denied command
  is reported verbatim, never re-encoded or re-routed.

## Method rules

Beyond the pooling rules above, three that were each paid for by a wrong result in this exercise:

- **Transcribe, never paraphrase, anything that generates the data under test.** Score generators,
  length distributions, reference oracles. `probes/18` gates its verbatim copies with
  `inspect.getsource`; the retracted `cnt` table above is what a paraphrase costs.
- **Confirm a test id with `pytest --collect-only`, not by reading `parametrize` decorators.**
  1.2 s, no TPU, and it is the difference between "this test exists" and "this test would exist if
  I read the decorators correctly".
- **Match opcodes, not substrings**, when reading HLO. `%bitcast-convert_bitcast_fusion` has opcode
  `fusion`. Judge materialization separately, by `window_config.estimated_cycles > 0` — a free
  `bitcast` never has one. (`probes/20`.)
