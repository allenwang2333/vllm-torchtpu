# Design — `sparsecore_topk` at `b=1024, n=10240, k=2048, P=1`, int32, single-chip SparseCore, prefill

Block: `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` @ `b11b6eb6`, md5
`44353b347c96e0eedbd742c710d4540a`. Branch `hning/sctopk-pipeline`.

Read with: `report.md` (the floor and the instruments), `hardware_map.md` (the live tables — stage
**C″** is the sheet in force).

## Numbers in this document

One tag per number: **`measured (j-…)`** · **`modelled`** (formula at first use) · **`assumed`**
(listed in §9/§10) · **`unmodelled`** (a cost known to exist with no term, which is not zero).

Every percentage is a change in time, `(new − old)/old`, so **negative is faster**, and prose states
the direction in words.

**The denominator, stated once.** Every share and every `Δ time %` in this file is against the
**harness prefill kernel, 189.98 µs** (stage C″ — the ρ = 0.40005 reweighting of the ρ = 1
microbenchmark). It is not the 326.782 µs the instrument reads, and it is not the 451.2 µs of
step 1.

⚠ **189.98 µs has never been measured, and nothing in this exercise has ever run at production ρ.**
Every probe ran at ρ = 1 (`hardware_map.md:376`). 189.98 is the **sum of its own reweighted rows** —
a modelled prediction that happens to be quoted as a baseline, not an instrument reading. Three
consequences, and they bind every number below:

- **It cannot fail to close.** The Stage C″ production sheet "closing to 189.98" is arithmetic, not
  corroboration. Do not cite the closure as evidence of anything.
- **When a row in that sheet is corrected, the total moves** — there is no hole to open, because
  there was never an independent measurement for a hole to open against. `probes/13` §7 corrected
  two rows and the prediction moved to **191.76 µs (+0.94%)**, which changes no ranking.
- **The first thing step 3 should buy is one ρ = 0.40005 arm of the incumbent.** It converts the
  denominator from `modelled` to `measured` and validates the whole reweighting in a single arm.
  ⚠ Until then, treat every percentage here as a ratio of two modelled quantities. Where a number was measured at ρ = 1 the µs figure is quoted unchanged and only its share
moves — the reweighting never invents a measurement.

⛔ **CORRECTION, `probes/11` (`j-9b520877`): 189.98 µs is the *harness*, and the harness is bigger
than production.** The harness's `jit` entry aval forces an XLA relayout of the 40 MiB score buffer
that the production caller does not pay (§1). **The production prefill kernel is 155.82 µs** — the
second, ±26.251 µs leg that made this a range is ⛔ **closed against us by `probes/14`**, below.

⛔⛔ **THIS WHOLE FRAMEWORK IS REFUTED BY MEASUREMENT. `probes/18-rho-denominator` (`j-97b98f97`)
measured the incumbent at the production ρ on the production entry — the arm the bullet above asked
for — and it does not agree with any row of the table below.** Read `results.md` § "The denominator,
remeasured" before using a single percentage in this file. In short:

- The production number is **275.185 µs**, not 155.82 — **+76.60%**. Gate-corrected per invocation,
  291.075 µs.
- **The 1.219 is not a factor.** The relayout is a *constant additive* **35.563 µs** (three ρ, spread
  0.195 µs — sixteen times inside the floor). `T_production = T_harness − 35.563`, not `÷ 1.219`.
  Divided in it is 1.129.
- **ρ is not a sufficient statistic.** At a fixed mean the cost spans **19.37%** on distribution
  alone, of which **8.26%** is row *arrangement* at an identical multiset. Nothing may be measured at
  ρ=1 and reweighted.
- **The denominator splits and the halves reweight oppositely** — ρ-independent savings ×1.55,
  in-walk savings ×0.62, a **2.5× fork**. So the percentages below cannot be repaired by any single
  scale factor. Treat each as a rank hint and a mechanism, never a magnitude.
- **`probes/09` is unsafe to quote at all**: its `B` anchor does not reproduce (+38.77%, with two
  other jobs agreeing against it), *and* all 12 of its arms carry `REFINEMENT_LEVELS = ()`, so
  `hardware_map.md:493`'s "V4 sweeps, **all levels**" row is labelled for a configuration its arms
  never ran.

The table is kept as the record of what step 2 believed, and is struck rather than deleted:

| denominator | µs | what it includes | every `Δ time %` below × |
| --- | ---: | --- | ---: |
| ~~**harness** — what every table in this file uses~~ | ~~**189.98**~~ | the untiling `copy` **and** the reshape+retile `bitcast-convert` fusion | ~~1.000~~ |
| ~~**production**~~ — ⛔ **refuted, measured 275.185** | ~~**155.82**~~ | drops the untiling `copy` only, −34.165 µs — ⛔ **verified absent** from the production HLO | ~~**1.219**~~ |
| ~~production, optimistic leg~~ | ~~129.56~~ | ~~also drops the `bitcast-convert`, a further −26.251 µs~~ — ⛔ **CLOSED and rejected by `probes/14`** | ~~1.466~~ |

✅ **The ±26.251 µs open denominator leg is settled, and it went the conservative way.** `probes/14`
decided it on **buffer assignment**, not on op names: XLA gives `%bitcast_convert_type` a separate,
disjoint, exactly-adjacent 40 MiB temp buffer from its operand in **both** modules (standalone
`1179648 + 41943040 = 43122688`; real production entry `1753088 + 41943040 = 43696128`). The
falsification attempt found **zero** counterexamples — of 281 ENTRY instructions only 16 carry a
non-zero `used_scoped`, all work-doing opcodes, while all 220 known-free instructions are zero, 215
of them *present-and-zero*. So the op is **materialised**: production pays it, the denominator is
**155.82 µs**, and the scale factor is a firm **1.219** rather than a 1.219–1.466 range.

⚠ **It also corrects `probes/11`'s refusal to transfer the number.** Probe 11 declined on the
grounds that the harness op "converts *and reshapes and retiles*" while production's is a pure
reinterpret. Reading the harness fusion's **body** (`probes/11/hlo/hlo-A0_base.txt:35-36`) shows the
reshape/retile is itself a free `bitcast` — same bytes — so the fusion's only real work is the same
40 MiB in / 40 MiB out dtype traversal production does, with a byte-identical `backend_config` except
window rank. The transfer is justified.

⚠ **Two ρ caveats on this table, and they pull in opposite directions.** 189.98 → **191.76** after
`probes/13` §7, so the production leg is **157.60**, not 155.82 — a 1.1% shift that changes no
ranking, and the 155.82 figure is kept here because every downstream citation uses it. And the
189.98 base is at ρ = 0.40005 while the 34.165 and 26.251 subtractions are ρ = 1 `measured`, so this
row **mixes ρ**. It is the best available and it is not exact; do not build a second decimal on it.

⚠ **The µs figures in this file are unaffected and remain the primary quantity; only percentages
move, and they all move up by the same factor.** Rather than restate several dozen cells against a
range, this file keeps the harness denominator throughout and converts once, here. **`results.md`
must not** — step 3 measures against a production-shaped entry and quotes production percentages
(`probes/11` § "The consequence for step 3").

## 0. Where the search landed

- **Best *measured* version:** `v2` — the incumbent with the 16× lane-private histogram replication
  removed. It is the only node with a measurement, and it is bit-exact. ⚠ It is **rank 6** of 7 on
  predicted prize; it leads on evidence, not on size. **The ranking is §9a**, and it supersedes the
  `rank N` labels in the plan headings.
- **The shortlist is six**, ranked in §9a: `v3` fused emit (≈ **−25.0%** production), `v12`
  (≈ −16.8%, firm), `v10` (≈ −14.9%), `v4` (≤ −8.64%), `v2` (−7.83%), `v11` (≤ −6.42%). ⛔ `v3` and
  `v10` are alternative roots and **compete**; the two edge nodes `v12` and `v11` compose with
  everything and are worth ≈ −23% between them for seven edits.
- **189.98 µs modelled-from-measured → 177.79 µs**, **−6.42%** (6.4% faster), from `R1`'s measured
  12.195 µs (`j-300d4690`, bit-identical output 1024/1024 and 64/64 rows). In production
  percentages that is **−7.83%**.
- **`v1` is at 10.6% of a 20.05 µs modelled floor** (13.0–15.5% against the production denominator).
  §1 carried a pessimistic 95.73 µs leg until `probes/10` returned; **that leg is withdrawn**, and
  `probes/11` replaced the 75.683 µs non-kernel residual with a **measured 5.385 µs**.
- **Versions:** **10 explored** (`v1`–`v7`, `v10`–`v12`; there is no `v8`/`v9`), 1 built and
  measured, **1 rewind**, and **2 killed this wave** — `v6` by `probes/14` on a trace, `v5` by
  `probes/16` on a microbenchmark.
- ⛔ **Both kills came the same way, and neither node died of its own arithmetic.** Each deleted work
  from a region correctly — `v5`'s sweep really is proportional to `NB` (r² 0.99934), `v6`'s short
  rows really are a quarter of prefill — and each had the saving absorbed by a term the model had **no
  entry for**: the L0 histogram grows as `NB` shrinks (+53.420 µs at NB=64), and an outer `lax.cond`
  already elides the whole kernel on `v6`'s steps. **§8b.** This is the most transferable thing the
  wave produced, and it sets a rule for step 3: **every arm measures the whole kernel, not the region
  it edits.**
- **The two largest prizes are both re-priced and both grew.** `v3` fused-emit ≈ **−20.5%**
  (−14.9% at the barriered bound), with its **sign settled without building it**: break-even
  `cnt_eff` 8531 exceeds the global maximum 6495. `v10` sample-and-count ≈ **−12.2%**, with the rate
  it rested on now `measured` and **at the read-only walk floor**. They compete rather than compose,
  and this probe wave did **not** adjudicate between them — both grew.
- ✅ **The biggest open question in this file is closed, and it went the good way.** `probes/10`
  traced the 75.683 µs residual to an XLA relayout of the 40 MiB score buffer, outside the Pallas
  kernel, and left two legs. **`probes/11` (`j-9b520877`) settled it: the artifact leg.** The
  production caller (`streamindex_topk.py:831`) emits **no** `sparse-core-data-format-call` and
  **no** 40 MiB untiling copy — its score buffer is `f32[1024,80,128]{2,1,0:T(8,128)}`, whose minor
  dimension is exactly one 128-lane tile, so the flatten to `s32[10485760]{0:T(1024)}` is a **free
  `bitcast`**. The harness passes `f32[1024,10240]`, where `(8,128)` tiling is *not* row-major, so
  XLA must untile 40 MiB first. **So there is no ~73 µs handoff** — and every prize in this file is
  worth **1.219×** what its table says. ✅ The two smaller things that survived it are now both
  settled by `probes/14`, and both settled *in our favour as candidates*: the **10.007 µs output
  retiling** does survive in-model (so `v11` is live, not zero-prize), and the ±26.251 µs
  `bitcast-convert` **is materialised in production** (so `v12`/U2's prize is real and firm, and the
  denominator collapses to a single number). §10.

⛔ **The kernel sits behind a runtime gate, and in the one profiled prefill step it did not run.**
`probes/14` enumerated a GLM-5.2 `PREFILL_ONLY` trace (`j-ecf4ee85`, rank 14) event-by-event, per
xplane **line**, uncapped: `StreamIdxTC-m-bq_128-bkvp_2`, `StreamIdxTC-d-bq_1-bkvp_2` and
`sc_topk_direct_b1024_n10240_k2048` have **0 device events each** — against 105 / 210 / 5,158 raw
`strings` text hits, which live in the `/host:metadata` `HloProto`. The ops are in the shipped
program (21 copies, one per index-share group) inside **branch 0** of `conditional.21`…`.41`;
**branch 1 executed**, at 659.59 µs × 21 = **13.851 ms/prefill step**.

The gate, read from source and **not** from the profile:

| | |
| --- | --- |
| `streamindex_topk.py:858-864` | `lax.cond(jnp.max(seq_lens) // compression_ratio <= k, _fast_path, _common_path)` |
| `mla_attention_op.py:57` | `STREAMIDX_ENABLE_EARLY_EXIT = True` — a hard-coded constant, no env var |
| `mla_attention_op.py:116` | `compression_ratio=1`, and `k = topk = 2048` |

⇒ the predicate is **`max(seq_lens) <= 2048`**. JAX orders `branches=[false_fun, true_fun]`, so
branch 0 is `_common_path` and branch 1 is `_fast_path` — which independently confirms `probes/14`'s
branch identification. `_fast_path` (`:840-856`) returns the **identity permutation**, correct
exactly when there are at most `k` candidates.

⚠ **"Not executed in the captured step" is what is established. "Dead at the benchmark" is not** —
the negative's scope was corrected after the first return. The predicate is evaluated **per step on
that step's batch**. The capture is one 1024-token `PREFILL_ONLY` step; the config is exact nominal
**8192 in / 1024 out** with `RANDOM_RANGE_RATIO="0.0"` and `MAX_NUM_BATCHED_TOKENS=1024`, so an
8192-token prompt is **eight** 1024-token chunks. `_fast_path:846-849`
(`q_abs_pos = (seq_len - q_len) + (token_idx - q_start)`) pins `seq_len` as **post-chunk**, which
fixes the gate to **chunks 0–1**: ✅ **the kernel runs on 6 of the 8 prefill chunks.** **A
chunked-prefill capture samples the first few chunks, and the one it holds is the gated chunk** — a
previously-burned trap on this same model, not a new observation.

⚠ **The flip point is outside the capture, and stays `unmodelled`.** The trace's second module takes
branch 1 as well — *decided*, not truncated: its `conditional.21` runs `while.297`/`.322`,
`gather_fusion.40/.41`, while `StreamIdxTC*`, `sc_topk_direct*` and `gather_fusion.42/.43` are absent
at count zero. A cadence join favours modules 0/4 = batches 2/3 by only a **4%** margin *and*
contradicts the post-chunk semantics above, so it was **not** resolved on that margin. No capture was
commissioned.

✅ **The branch mapping is confirmed three ways, including against a challenge.** The objection was
that branch 1 does 596 µs of real work (7 iterations of `u32[1024]` fusions) and so might be the
common path. It is not: `_fast_path:841` is `jnp.searchsorted(cu_q_lens[1:], token_idx)` over
`arange(1024)`, and JAX lowers `searchsorted` as a `ceil(log2(len+1))`-level binary search on a
**uint32** index. `cu_q_lens` is `s32[65]` → 64 entries → **exactly 7 levels** over 1024 lanes.
`ceil(log2(64)) = 6` does not match. The expensive thing branch 1 does is the binary search itself.

⛔ **`v6` is rejected — but for this workload, not in general, and the distinction is the finding.**
D14's premise was "prefill chunks 0 and 1 of 8 run a full DMA + `cvt` + `hist0` + sweep to discover
*keep everything*". They do not: the outer `lax.cond` elides the kernel in exactly those chunks. The
set coincidence is **exact** — `k = 2048` is a whole multiple of the 1024 chunk size, so no chunk
straddles and the `token_pos < 2048` row set *is* the gated chunk set. The escape route — short rows
inside a batch that also holds a long sequence, since the gate is a `max` — is real in principle
(`tpu_runner.py:1154-1215` keeps a mixed bucket, which `hardware_map.md:561` already leans on) but is
**measured shut here**: all 96 `batch_composition_stats_*.json` show `num_reqs: 1`,
`num_decode_tokens: 0`. ⚠ The operative reason is **not** "`PREFILL_ONLY` steps have uniform rows"
but the narrower and stronger "**this benchmark schedules one request per step, so no batch can mix
lengths at all**". And a `prefill_only` window is one the harness *arranged*, so it is not evidence
about served steady state. **`v6`: prize 0 `measured` in the profiled window, `unmodelled` in
steady-state serving.**

✅ **And it hands over a bigger non-kernel number than anything in this graph.** The executed
`_fast_path` costs **13.851 ms/step, 2.12% of a 655 ms prefill step** — to emit an identity
permutation plus a mask. `(1024, 2048) int32 × 21` is 168 MiB of writes, ~105 µs at HBM rate; the
measured cost is **~130× that**, and most of it is the 7-level `searchsorted` binary search. Written
up as a full `carried:` handoff in §10. ⚠ It is paid only on the **gated chunks, 2 of 8**, and it is
**not additive with any kernel prize** — the two branches are mutually exclusive.

⚠ **The caller's stated mechanism is node `v4`, and it is small.** "The DMA is synchronously waited;
use `emit_pipeline` for double buffering" is a real finding — the transfer is 100% exposed, overlap
fraction 0.0000 (`probes/06` Q1) — but its whole prize is the **13.469 µs** of exposed transfer
(`measured, j-300d4690`), **7.1%** of the kernel, and `cvt` already absorbs the rest of the
transfer's latency for free. It stays in the graph and in the shortlist. It is not the largest thing
here and this document says so early rather than at the end.

## 1. The floor, and the band being aimed at

At the production prefill shape and ρ = 0.40005. Bytes that must move: `ρ·b·n·4` read = 16.78 MB,
`b·k·4` written = 8.39 MB, total **25.17 MB**.

| | µs | tag | notes |
| --- | ---: | --- | --- |
| **block floor** | **20.05** | modelled | `max(t_hbm, t_sparsecore)`; there is no MXU or VPU row — §4a |
| `t_hbm` | 20.05 | modelled | 25.17 MB ÷ 1255 GB/s, the *measured* 40 KiB-slice rate (`report.md` §5b), not the 3113 GB/s asymptote |
| `t_sparsecore` | ~16.5 | modelled | ≥ 2 passes over the live elements at `cvt`'s measured 1.963 ps/element (`probes/03`): one to fix a threshold, one to emit. A correct exact top-k cannot do it in one |
| `t_vpu` / `t_mxu` | 0 | measured | **both units are idle for 100% of the block** (`report.md` §2, ops 15–16 do not execute at `P=1`) |
| **terms outside the floor** | — | | ✅ **D3 resolved: 34.2–60.4 µs of it is a harness artifact and 5.385 µs is the measured non-kernel floor** (`probes/11`; see below). ⚠ A **10.007 µs output retiling** survives in production. Also: SparseCore↔TensorCore handoff (D7, `unmodelled`); the `b_pad`-wide length DMA (D11, `assumed` negligible — **`probes/10` closed the per-row axis, so it is not promoted**) |
| `v1` measured-and-reweighted, **harness** | 189.98 | measured (`j-4e0df30a`) + modelled reweighting | achieved: **10.6%** |
| `v1` **production** | **155.82** | the above less `probes/11`'s harness-only relayout (the further −26.251 µs leg is ⛔ closed by `probes/14` — production pays it) | achieved: **12.9%** |
| target | 25–33 | | 60–80% of floor |

### ✅ D3 — 39.8% of the *measured* kernel — is not in the kernel, and mostly not in production

**D3 is 75.683 µs and belongs to no op.** `probes/09` measured it with every region switched off
(`dma, cvt, hist, sweep, emit, fill, outdma`) against an empty launch, and refuted four mechanisms:
a per-wave Mosaic barrier, score-DMA arrival latency, `plsc.parallel_loop` entry, and — measured as a
slope rather than a division — the wave loop, at **0.009 µs/wave**.

**`probes/10` (`j-770c6de9`) found what it is.** Varying `n`, `k` and `b` independently:

- **No per-row term.** `A0_n/4` and `A0_b/4` carry identical input bytes and differ 4× in `b`;
  the leftover after the output-byte slope is **−0.886 ns/row**, inside the 0.770 µs null spread.
  ⚠ The b-axis fit's 71.4 ns/row at R² 0.992 is an artifact — `b` scales both byte axes together.
- **It is buffer bytes**: an input term, convex (1.020 µs/MiB over 10→20 MiB, **2.147** over
  20→40 MiB — quote the segment, never the 1.8246 chord), plus a clean linear output term at
  1.118 µs/MiB ≈ 8.9 µs at the base shape, R² 0.99978.
- **The optimized HLO names the input term**, inside the timed `jit` and **outside the Pallas
  kernel**: `%copy.3 = s32[128,80,8,128]{3,1,2,0:T(8,128)} copy(…)` from `{3,2,1,0:T(8,128)}`, under
  a `sparse-core-data-format-call` — the whole 40 MiB score buffer read and written,
  ≈**73 of the 75.3 µs** at ~1.1 TB/s. ⚠ `A0` deletes the score DMA and XLA reformats the buffer
  anyway, because it is a custom-call operand. That is why a total kernel blackout does not remove it.

**Consequences for the floor.** The pessimistic 95.73 µs leg this section used to carry is
**withdrawn**: a cost that scales with buffers is not an irreducible cost of invoking the op. Every
"% of floor" in §3 and §5 stands against **20.05 µs**.

### ✅ `probes/11` (`j-9b520877`) decided it: **the harness-artifact leg**

The production caller is `streamindex_topk` (`streamindex_topk.py:509`, calling `sparsecore_topk` at
`:831`). Its optimized HLO, compiled at both `enable_early_exit` settings on GLM-5.2 avals and
self-checked to contain `sc_topk_direct_b1024_n10240_k2048`, contains **zero**
`sparse-core-data-format-call` and **no** layout-changing copy ≥ 1 MiB — the only `copy` in the whole
module is `%copy.11 = f32[1024,64]`, 256 KiB of `indexer_weights`. I re-derived this myself from
`probes/11/hlo/hlo-prod_early_exit_False.txt`, walking the operand chain **by name**:

```
%StreamIdxTC-m-bq_128-bkvp_2.1 = f32[1024,80,128]{2,1,0:T(8,128)}   custom-call
%bitcast_convert_type.3        = s32[1024,80,128]{2,1,0:T(8,128)}   bitcast-convert(↑)
%bitcast.74                    = s32[10485760]{0:T(1024)}           bitcast(↑)          ◄ FREE
%sc_topk_direct_…call-start(%bitcast.74, …)
```

**Why the two differ.** Production's minor dimension is exactly **128 = one lane-tile**, so
`{2,1,0:T(8,128)}` *is* row-major and the flatten is a bitcast. The harness's entry aval is
`f32[1024,10240]{1,0:T(8,128)}`, minor dim 10240 ≫ 128, so the tiling is not row-major and XLA must
untile the whole buffer before a SparseCore call can read it.

**The arms, and what they price.** One interleaved pool, 9 rounds, null spread **1.641 µs** (⚠ 2.1×
worse than probe 10's 0.770 — every delta is quoted against 1.641). Every arm that moved also deleted
the named region **in its own dumped HLO**, and `(flatin + flatout)/flatio = 0.999`, so the two
regions are additive and non-overlapping.

| arm | Δ vs `A0_base` | what it deletes |
| --- | ---: | --- |
| `A0_flatin` | **−60.416 µs** | the whole input path — df-call, `copy.3`, and the `bitcast-convert` fusion |
| `A0_s32in` | −26.251 µs | the `bitcast-convert` fusion only; `copy.3` survives |
| ⇒ `copy.3` alone | **34.165 µs** | by subtraction; 40 MiB moved twice ⇒ 2.46 TB/s, a plausible full-buffer rate. ✅ **Independently confirmed by `probes/16`**, which was not looking for it: its 3-D entry HLO is clean (0 df-calls, 0 copies) against 8 df-calls on the 2-D entry, and it *timed* the relayout at **+32.994 µs** — **3.4%** from this number, in a different job and a different pool |
| `A0_flatout` | −10.007 µs | the output retiling `ROOT %reshape` |
| `A0_flatio` | −70.511 µs | both; **leaves a 5.385 µs residual** |
| `A0_donate` | +1.557 µs | ✅ **null** — inside the spread, HLO unchanged. Not an aliasing artifact |
| `A0_fmtin` | −0.049 µs | ⚠ **inert, not a null** — `Layout.AUTO` returned the layout the buffer already had, so the arm moved nothing. It did not test the hypothesis |

⚠ **The prober's headline over-transferred by 26.251 µs, and the main thread corrected it.** Its
report read "production pays *neither* input relayout, so ~70.5 of 75.3 µs does not exist". The
`copy.3` leg is right. The `bitcast-convert` leg is not: **the two modules' `bitcast-convert`s are
different ops sharing an opcode.**

| | harness | production |
| --- | --- | --- |
| | `%bitcast-convert_bitcast_fusion`, `kind=kLoop` | `%bitcast_convert_type.3`, not a fusion |
| in → out | `f32[1024,10240]{1,0:T(8,128)}` → `s32[128,80,8,128]{3,2,1,0:T(8,128)}` | `f32[1024,80,128]{2,1,0}` → `s32[1024,80,128]{2,1,0}` |
| what it does | convert **+ reshape + retile** — real movement | a **same-shape, same-layout, 4→4 byte** reinterpretation |
| priced at | **26.251 µs** (`A0_s32in`) | ✅ **26.251 µs, transferred** — `probes/14` |

Production's *looks* like a no-op buffer assignment should alias away — but ⛔ it carries the **same
`estimated_cycles: 66712`** as the harness fusion and a full `window_config`
(`iteration_bounds ["2","5","1"]`, `buffering_level 2`), i.e. XLA's emitter planned a loop for it.
That is evidence **for** materialisation, and ✅ **`probes/14` confirmed it and reversed the "does
not transfer" call.** Two findings, neither of which relies on the op's name:

- **Buffer assignment.** In both modules XLA gives `%bitcast_convert_type` a **separate, disjoint,
  exactly-adjacent 40 MiB temp buffer** from its operand — standalone `1179648 + 41943040 =
  43122688`; the real production entry `1753088 + 41943040 = 43696128`. An aliased reinterpretation
  would share the operand's offset. It does not. It is materialised, and production pays it.
- **The falsification attempt found zero counterexamples.** Of 281 ENTRY instructions only **16**
  carry a non-zero `used_scoped`, and all 16 are work-doing opcodes; all **220** known-free
  instructions are zero, **215** of them present-and-zero rather than absent. So "non-zero
  `used_scoped` ⇒ real work" is not an artifact of the field being unset on free ops.

⚠ **This also overturns `probes/11`'s stated reason for refusing the transfer**, which was that the
harness op "converts *and reshapes and retiles*" while production's is a pure reinterpret. Reading
the harness fusion's **body** (`probes/11/hlo/hlo-A0_base.txt:35-36`) shows its reshape/retile is
itself a free `bitcast` on the same bytes, so the fusion's only real work is the same 40 MiB in /
40 MiB out dtype traversal, with a byte-identical `backend_config` bar window rank. **The 26.251 µs
transfers.** The denominator is therefore a single 155.82 µs, and `v12`/U2's prize is firm.

**Consequences.**

1. ⛔ **There is no ~73 µs `carried:` handoff.** Remove it from §10; nothing at the call site to hand
   over. Never quote "~73 µs" again.
2. ✅ **The non-kernel floor is 5.385 µs, measured** (`A0_flatio`), not 75.683. `probes/09`'s bracket
   collapses to its optimistic leg.
3. ✅ **Every prize in §9 is worth 1.219× its tabulated percentage** — one factor, not a range, now
   that `probes/14` has closed the second leg. The *ranking* is invariant, exactly as §10 predicted:
   the correction is a common factor.
4. ✅ **The 10.007 µs output retiling is live in-model** — `ROOT %reshape` on the kernel's flat
   `s32[2097152]{0:T(1024)}` output, present in the production entry. The worry was that compiling
   `streamindex_topk` standalone forces its ROOT layout, so the retile might be an artifact of the
   compilation boundary. `probes/14` Q2 settled it: in the **real** entry the retile survives as an
   `S(1)` staging copy rather than a ROOT reshape. Different op, same **8 MiB** of movement, so `v11`
   keeps its prize. §10. ⛔ **This sentence read "40 MiB" until step 3 corrected it** — 40 MiB is the
   `f32[1024,10240]` *input* buffer, which is `v12`'s prize. The retiled *output* is
   `s32[1024,2048]` = 8 MiB, and the design's own rate agrees: 10.007 µs ÷ 1.251 µs/MiB = 8.0 MiB.
   No prize moves, because both were computed from the rate and never from the byte count.
5. ⚠ **The step-3 harness measures a shape production never runs.** `instruments/` is read-only and
   its frozen md5 is what makes probes 08–11 comparable, so do not edit it — but step-3 arms build
   their own thunks, so **build every variant and the incumbent against a production-shaped entry**
   and quote production percentages in `results.md`.

**Revising the floor.** Step 1 quoted 8.9% of a 40.1 µs HBM roofline at ρ = 1. That is superseded
twice: by ρ (25.17 MB required, not 50.33) and by the `t_sparsecore` term, which step 1 left unpriced
while calling it "87% of the kernel". Both figures are above; this file uses 20.05.

## 2. What a version is, and what a move is

The eight fields and the move vocabulary are `TEMPLATE-design.md` §2 verbatim; nothing about this
block changes them. Three block-specific notes on how they bind here:

- **`layout` is a one-axis field.** In-job `SparseCoreInfo(num_cores=2, num_subcores=16,
  num_lanes=16, vmem_capacity_bytes=524288, dma_granule_size_bytes=32)`: **16 int32 lanes, no
  sublane axis, no MXU.** `hardware-facts.md`'s 8×128 relayout model is a TensorCore model and is
  **void** here — see `hardware_map.md` "The layout vocabulary this block needs". A layout cell is
  therefore `[lane=<axis>] <memory>` and nothing else, and `relayout` as a move is nearly empty in
  this block: there is one register axis and no transpose anywhere (§4b, L5).
- **`live set` is not binding and that is itself a finding.** 70.50 KiB of 512 (13.77%), with the
  largest shape in the tree at 34.4%. **4.3× of headroom, unused** — so no move below is gated on
  VMEM, and two moves that each want a large scratch (D1's second slice buffer, D6's straddler
  buffer) **both fit together**. That is a step-4 composition fact, recorded now so step 3 does not
  treat them as rivals.
- **`grid` is a `fori_loop` over waves, not a Pallas grid.** `n` is untiled by algorithmic decision
  — making the whole row resident is what gives the 1:5 HBM:VMEM pass ratio. There is no
  `BlockSpec` pipeline to change, which is why the caller's `emit_pipeline` request is a `transfer`
  move that must be built by hand (§9, plan C).

**Exactness.** Every `add`/`remove`/`mutate`/`precision` edge below declares it. ⛔ **Corrected — an
earlier draft said "two of the eight nodes are `≈ approximate`". That is false and always was.**
There are **ten** nodes (`v1`–`v7`, `v10`–`v12`; no `v8`/`v9`), and **none is `≈ approximate`** — see
§3, which states it with the reasoning. `v10` is the only node that could become approximate, and
only if step 3 drops its correction loop.

## 3. The version graph

```
v1  incumbent: 16x-replicated 256-bucket radix select      189.98 us  measured+reweighted  10.6%
│                                                                     (j-4e0df30a, j-300d4690)
│
├─[mutate] R=16 -> R=1: one shared histogram, hardware conflict resolution
│  │        exact -- bit-identical, 1024/1024 and 64/64 rows
│  │                       ◄ motivated: D5 -- plsc.addupdate_scatter may resolve
│  │                                    intra-vector address conflicts (source read)
│  │                       ◄ judged:    probes/08 R1 arm (j-300d4690)
│  v2  R=1                                                 177.79 us  measured   11.3%   KEPT <- best
│  │                                                                             -6.42%
│  │
│  ├─[remove] fuse L0's emit: one pass writes `> b_star` straight to rowbuf     ∥
│  │  │       and `== b_star` to a compact straddler buffer; refinement then
│  │  │       walks `cnt`, not `L`.  exact
│  │  │                    ◄ motivated: D6 -- emit 17.7% + refinement hist 10.6%
│  │  │                                 are two full walks of L to touch k and
│  │  │                                 ~cnt elements (stage C")
│  │  │                    ◄ judged:    NOT YET -- step-3 variant
│  │  v3  fused-emit                            SUPERSEDED ~124.2 us  modelled   16.1%   OPEN
│  │  │                              vs parent   -30.2%.  See §5: -20.5% vs v1, sign settled
│  │  │
│  │  └─[transfer] double-buffer the score DMA + ragged it to i_hi              ∥
│  │     │                 ◄ motivated: D1 -- overlap fraction 0.0000, 3 waits,
│  │     │                              two literal snops in the slack
│  │     │                              (probes/06 Q1); D8 -- 60% of prefill
│  │     │                              bytes are never read
│  │     │                 ◄ judged:    NOT YET -- step-3 variant. Ceiling is
│  │     │                              measured: 13.469 us (j-300d4690)
│  │     v4  pipelined-DMA        <- THE CALLER'S REQUEST   ~110.7 us  modelled   18.1%   OPEN
│  │                                                                             -41.7% cumulative
│  │
│  ├─[grid] sweep NUM_BUCKETS: 256 -> {64, 128, 512}                            ∥
│  │  │                    ◄ motivated: stage C" -- zero_priv is a DENSE clear,
│  │  │                                 LINEAR in NUM_BUCKETS, and is half the
│  │  │                                 sweep. Never swept. (D9)
│  │  │                    ◄ judged:    NOT YET -- three compile-time arms
│  │  v5  NB-tuned                                         measured          REFUTED
│  │                                    ^ probes/16: won both open terms, lost on a third
│  │
│  ├─[remove] block-sum -> cumsum -> add back: depth 16 -> 3 on scan_glob        ∥
│  │  │       and find_bucket.  exact
│  │  │                    ◄ motivated: D13 -- two 16-step serial carries
│  │  v7  shallow-scans                                    ~185.3 us  modelled   10.8%   PARKED
│  │                                          -2.5%.  Demoted by probes/09: the two carries
│  │                                          are 4.698 us total, not ~139. Reopen only if
│  │                                          v5 raises NUM_BUCKETS. §10.
│  │
│  └─[boundary] @pl.when(i_hi > k) around the whole selection; emit the
│     │         identity permutation otherwise.  exact
│     │                    ◄ motivated: D14 -- keep_all <=> rho <= 0.2, so prefill
│     │                                 chunks 0 and 1 of 8 run a full DMA + cvt +
│     │                                 hist0 + sweep to discover "keep everything"
│     │                    ◄ judged:    NOT YET -- and its verdict turns on an
│     │                                 UNMODELLED term (step homogeneity in rho)
│     v6  keep-all-fastpath                                     KILLED by probes/14: the
│                                        outer lax.cond already elides the kernel here. §10
│
└─[mutate] SECOND ROOT: estimate-and-verify instead of radix-partition.
   │       Histogram a 1/16 sample to estimate the k-th threshold, then COUNT
   │       passes (compare + all_reduce_population_count, no scatter) to correct
   │       it, then one emit pass.  exact -- the correction loop is unconditional
   │       on a miss, so the answer is the true k-th value, not an estimate.
   │                       ◄ motivated: D2 -- hist0's dense scatter is irreducible
   │                                    FOR A HISTOGRAM (one vst.idx.add per element
   │                                    whatever NUM_BUCKETS is), and the histogram
   │                                    block is 28.4%, the largest structural lever
   │                       ◄ judged:    NOT YET. Its whole prize rests on one
   │                                    UNMODELLED rate: a count pass vs a histogram
   │                                    pass. Mandatory probe, §10.
   v10  sample-and-count                                   ~168.4 us  modelled   11.9%   OPEN
                              -12.2% vs v1.  Count-pass rate now MEASURED (probes/15) and
```

Node: version id, its measured or modelled time, its achieved fraction against the 20.05 µs floor,
its verdict. Edge: `[move type]` the edit · `◄ motivated:` the finding and its instrument ·
`◄ judged:` the instrument and job that decided it.

⛔ **Read the percentages in §5, not the ones in the box above, and never mix them.** The box states
each child's delta **against its parent** (`v3`'s −30.2% is against `v2`'s 177.79 µs). Every node
block in §5, and every figure `probes/13` and `probes/15` produced, states the delta **against `v1`'s
191.76 µs production total** (`v3`'s −20.5%). Both are correct and they are not comparable; the box's
absolute µs for `v3` and `v4` are additionally **superseded** by the re-pricings in §5 and are kept
only to show the shape of the search. A ranking that mixes the two denominators is wrong.

⚠ **There is no `v8` or `v9`** — the numbering jumps to `v10` to mark the second root, and neither id
has ever existed in this file at any commit. Do not go looking for them.

**Two edge nodes are not in the box**, because they are not alternatives to anything: **`v11`**
(retiled output emit) and **`v12` = U2** (`f32` kernel input, reinterpret in-register) hang off `v1`
as boundary moves and **compose with every node above and with each other**. Both came from stage E,
both had their feasibility closed by `probes/17`, and both are in §5.

Verdicts: **KEPT** (extended) · **REJECTED** (branch closed, with the rewind target) · **PARKED**
(not built, with what would reopen it) · **OPEN** (in flight, or blocked on a mandatory probe).

**No node is `≈ APPROXIMATE`.** Every move in this graph computes the exact same function, and `v2`
proves it the strongest way available — bit-identical output on 1088 rows. That is worth stating
explicitly, because a top-k kernel is exactly the kind of block where an approximation is fast and
the benchmark cannot tell: `recall_target` shortcuts and sampled thresholds are both available here
and neither is being taken. `v10` is the one that could easily have been approximate — a sampled
threshold *is* an estimate — and its correction loop is what makes it exact. If step 3 finds the
correction loop too expensive and drops it, `v10` becomes `≈ approximate` and owes a gate.

### 3b. Which branches run in parallel, and which cannot

| in flight | parallel with | why, or why not |
| --- | --- | --- |
| `v3`, `v5`, `v6`, `v7` | each other (`∥`) | four siblings of `v2`; none cites another's measurement |
| `v10` | all of the above | second root — it does not descend from `v2` at all |
| `v4` | nothing | its motivation is unaffected by `v3`, but it is drawn from `v3` because the fused pass changes what the DMA feeds. Buildable in parallel, **must be timed after `v3`** |
| everything | — | ⚠ **the timing is never fanned out.** Up to four builds per pod, one per chip; one timing at a time, and the order goes in `results.md` |

**Two siblings intended to combine are three versions.** `v2 + v3` and `v3 + v4` each get their own
node and their own measurement in step 4. **Do not add `v2`'s −6.42% to anything** — and specifically
not to `v5` or `v7`, which act on the same `zero_priv` that `v2` already shrinks 16× (see §8).

## 4. `v1` — the starting point

**Obtained by:** read from the incumbent — source read, `probes/06-llo-incumbent` (`/xprof-llo`,
Mosaic MLIR + VLIW bundles), and the `probes/03`/`07`/`08`/`09` price sheet. Nothing in §4 is
invented.

### 4a. Ops and placement

Full sixteen-row table: `hardware_map.md` "Stage A — the op table". Placement is trivial and
unanimous: **every executing op is on the SparseCore.** Ops 15–16 (the two-stage merge) do not
execute at `P=1`, which is both shipping shapes.

| unit | ops on it | time (prefill, µs) | share |
| --- | --- | ---: | ---: |
| **SparseCore** | 0–14 — all of them | 189.98 | **100%** |
| VPU | none | 0 | 0% |
| MXU | none | 0 | 0% |
| TensorCore | none, and 99.8% unoverlapped with its neighbours | 0 | 0% — **D7, a handoff** |

The per-op sheet, from stage C″ (`probes/09` + `probes/08`), at production ρ:

| op group | µs | share | scales with ρ? | tag |
| --- | ---: | ---: | --- | --- |
| ⚠ D3 — **outside the kernel**: an XLA relayout of the score buffer, plus an unexplained output-byte term | 75.683 | 39.8% | no — it tracks *padded* buffer bytes, which ρ does not change | measured (`j-4e0df30a`, `j-770c6de9`); mechanism `modelled` from HLO |
| `hist0` dense scatter | 33.91 | 17.8% | yes | modelled from `j-300d4690` |
| `emit` | 33.54 | 17.7% | yes | measured (`j-4e0df30a`) |
| refinement `hist` | 20.08 | 10.6% | yes, and gated by `keep_all` | modelled |
| V4 sweeps (`zero_priv` 9.238 · `fold_priv` 4.900 · `scan_glob` 3.498 · ⛔ ~~`find_bucket` 1.200~~ — `probes/16` finds it **unresolved at every radix**, 0.047 / −1.385 / 3.634 at NB 64/256/512, below a 1.859 µs floor; three primitives plus noise) | 15.63 | 8.2% | no | measured (`j-4e0df30a`), split reproduced at three radices by `probes/16` (`j-c47cfec1`) |
| score DMA | 13.469 | 7.1% | **no — static `slice_len`** | measured (`j-300d4690`) |
| output DMA | 5.872 | 3.1% | no | measured (`j-4e0df30a`) |
| `cvt` | −8.20 | −4.3% | yes | measured (`j-300d4690`) — **negative, and real** |

⚠ **`cvt`'s marginal cost is negative — deleting a whole 640-chunk traversal made the kernel 20.5 µs
slower at ρ=1.** It is the first consumer of `resident_vmem` after the score DMA and absorbs the
transfer's arrival latency while doing its own work. This is load-bearing for `v4`: the caller's
double buffer **competes with an existing mitigation** rather than adding to one.

### 4b. Values, layout and memory

Full table: `hardware_map.md` "Stage A — the value table". Transcribed here are the rows a move
below changes.

| value | produced by | consumed by | layout out | layout in | mismatch | memory | bytes | tag |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| **V2** `scores` | caller | op 1 DMA | `(b)[lane=n] HBM` | `[lane=n] VMEM` | — | HBM | 41.94 MB | measured |
| **V3** `resident_vmem` | op 1 | `cvt`, `hist0`, refinement, `emit` — five consumers | `[lane=n] VMEM` | `[lane=n] VMEM` | — | VMEM | 40 KiB | measured |
| **V4** `priv_vmem` | `zero_priv` | `fold_priv` | `lane[lane=bucket] VMEM`, addr `lane*256+bucket` | same | ⛔ **16× multiplicity, and it is not elementwise** — `lane_iota` replicates the histogram 16×; the compiler does not hoist it because the replicas are genuinely distinct accumulators | VMEM | 16 KiB | ⚠ **priced at 19.192 µs/level by `probes/07`; `R1` then measured the whole removal at 12.195 µs total. `probes/07` overstated it ~5× — see §7** |
| **V5** `glob_vmem` | `fold_priv` | `scan_glob`, `find_bucket` | `[lane=bucket] VMEM` | same | — | VMEM | 1 KiB | measured |
| **V7** `rowbuf` | `emit` | op 13 DMA | `[lane=k] VMEM`, data-dependent `woff` | `[lane=k] VMEM` contiguous | — | VMEM | 8 KiB + 64 B guard | measured |

Layout cells are physical. **There is one register axis** (16 int32 lanes, no sublane), **no
transpose anywhere in the block**, and the only broadcast is V4's, on a *leading* axis — free to
address, not free to zero or fold. That is why `relayout` barely appears in §3 and `mutate` carries
the graph instead.

### 4c. Grid, boundary, and what is not hidden

Full axis table: `hardware_map.md` "Stage A — every axis of the iteration space". The four axes:

| axis | extent | tiled? | why |
| --- | --- | --- | --- |
| `b` (rows) | 1024 | **yes** — `num_waves = b/32`, a static `fori_loop` (`:329`); wave `w` covers rows `[32w, 32w+32)`, so subcore `flat` owns `flat, flat+32, …` (`:156-158`) | 32 subcores. ⚠ **There is no inter-subcore barrier** (`:19`, `:528`), so cost is `max` over subcores of each subcore's own 32-row sum. The **stride** then makes those sums near-equal even in a mixed prefill/decode step — every subcore gets a representative sample, so `max ≈ mean`, and a per-row saving converts to wall time proportionally. ✅ Source-read confirmed; it is what unblocks `v6` |
| `n` (scores) | 10240 | **no — by algorithmic decision, not by capacity** | making the whole row resident gives a 1:5 HBM:VMEM pass ratio instead of 5:5. VMEM is at 13.77% |
| `NUM_BUCKETS` | 256 | no | fixed compile-time constant — **never swept**, which is `v5` |
| `P` (score split) | **1** at both shipping shapes | n/a | `P>1` needs `max-model-len` above 9216 |

**Boundary.** One `plsc` mesh kernel. What stays with XLA: the caller's `_effective_row_lengths`
computation before, and the gather that consumes `out_hbm` after. ⚠ **`P=1` means there is no
partition loop to reorder**, so D7's SparseCore/TensorCore pipeline cannot be built from inside this
kernel — it would have to be constructed in the attention op. That is a **handoff**, not a version.

**Is the DMA hidden? No — overlap fraction 0.0000, and it is measured, not modelled.** `probes/06`
Q1 (`/xprof-llo`, static analysis of the Mosaic MLIR): one `resident` alloca (`10240xi32`), 3
`enqueue_transfer`, 3 `transfer_wait`, and the compiler had 4 bundles of slack across the 40 KiB read
and **filled two of them with literal `snop`s**. Overlap is structurally impossible without a second
buffer. That is `v4`, and it is the caller's request stated in the kernel's own terms.

### 4d. Transfer, live set, and the hardware contract

Transcribed verbatim from `hardware_map.md` stage A/D, which is now deleted there so the two cannot
drift.

**`CONTRACT.md` §A — layout, per value.** This is the only copy; the draft beside the value table in
`hardware_map.md` is deleted.

| value | L1 produced | L1 consumed | L2 window | L3 broadcast | L4 concat/split | L5 transpose | L6 multiplicity | L7 itemsize |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **V2** `scores` | `(b)[lane=n] HBM` | `[lane=n] VMEM` | none — no windowed read | none | **split on `n` into `P`; at both shipping shapes `P=1`, not split** | none | none | n/a |
| **V3** `resident_vmem` | `[lane=n] VMEM` | `[lane=n] VMEM`, all five consumers | none | none | not split — `n` untiled by decision | none | none | `int32` |
| **V4** `priv_vmem` | `lane[lane=bucket] VMEM`, addr `lane*256+bucket` | `lane[lane=bucket] VMEM` | none — `fold_priv` is **16 contiguous 16-word loads**, not a strided read (`for m in range(16)` is a python unroll) | **`lane_iota` replicates the 256-bucket histogram 16× along the leading axis.** Leading, so free to *address*; **not** free to zero or to fold | none | none | ⛔ **16×, and it is not elementwise.** `CONTRACT` L6's exception: the compiler does not hoist it, because the 16 replicas are genuinely distinct accumulators. ⚠ **Measured by removal (`R1`, `j-300d4690`): 12.195 µs in total, bit-exact.** The earlier `probes/07` fit of *19.192 µs per level* — ~62 µs over 3.231 levels — is **~5× too high** and is retracted in §7 | `int32` (`lane_iota`, line 100; `addr`; `mask` from int32 compares) |
| **V5** `glob_vmem` | `[lane=bucket] VMEM` | `[lane=bucket] VMEM` | none | none | none | none | none | `int32` |
| **V7/V8** `rowbuf`/`keybuf` | `[lane=k] VMEM`, data-dependent `woff` | `[lane=k] VMEM` contiguous | none | none | none | none | none | `int32` |
| **V10/V11** (`P>1` only) | `(b,P)[sub=·, lane=k_p] HBM` | `(b)[sub=·, lane=P·k_p] HBM` | none | none | **merge of the `P` splits — this is stage 2** | none — leading-axis reshape | none | `int32` |

**L1** — answered per value above; no cell says "as the compiler lays it out".
**L2** — no value in the block is read through a sliding window or a shifted tap. Nothing to measure.
**L3** — one broadcast, V4's, on a **leading** axis. Free to address; the cost is in L6, not here.
**L4** — the only split is `P`, and it is 1 at both shipping shapes. There is no contraction in the
block, so the "was a narrow contraction merged with a sibling" half does not apply — **there is no
MXU row at all**.
**L5** — no transpose or axis swap anywhere. There is one register axis.
**L6** — ⛔ **the one row that is not free**, and the one row whose first price was wrong. Now
**measured by removal at 12.195 µs**, not `unmodelled` and not the fitted 19.192 µs/level.
**L7** — `int32` throughout; no int8 iota, no 8-bit compare, so the floor `CONTRACT` L7 warns about
is not approached.

Every other §A row is closed on paper, which `CONTRACT.md` says is a fine result to state explicitly.

**`CONTRACT.md` §C — transfer, per array.**

| array | T1 mechanism, grid extent | T2 ragged/padded, padding % | T3 can be zero-length | T4 re-transferred vs carried | T5 aliased + donated |
| --- | --- | --- | --- | --- | --- |
| `len_hbm` → `len_vmem` | **manual** `make_async_copy(...).start(); .wait()`, once at op 0. Static size `b_pad` | padded — every subcore moves all `b_pad` lengths and reads `b/32` of them (**4 KiB for 32 useful words** at prefill) | no | carried for the whole kernel | not aliased. **D11** |
| `scores_hbm` → `resident_vmem` | **manual**, per wave, op 1. Static `slice_len`. Grid is a static `fori_loop(0, num_waves)`; only `i_hi`/`walk_chunks` are dynamic | ⚠ **padded — D8.** `1 − ρ` = **60% at prefill**. Op 1 always moves `slice_len`; every compute pass is bounded by `walk_chunks = cdiv(i_hi,16)` | **yes** — the whole wave body is under `@pl.when(i_hi > 0)`, op `Z` is the else. Never fires at `b ∈ {64,1024}` (both multiples of 32) | **re-transferred per wave** — necessarily, the rows differ | not aliased |
| `rowbuf_vmem` → `out_hbm` | **manual**, per wave, op 13. Static `k` | not padded | yes, under `out_gate` | re-transferred per wave | ⚠ **not aliased, not donated** — and if `probes/10` finds the D3 residual tracks the output buffer, **T5 is the fix route** |

⛔ **T4 overlap fraction is 0.0000** — see §4c. **T5 alignment:** HBM → tilespmem, contiguous both
sides, DMA granule **32 B**; `slice_len` = 10,240 words = 40,960 B = **1,280 granules exactly**, no
sub-granule tail. ⚠ A ragged DMA (`v4`/D8) at word granularity *creates* one — round `i_hi` up to 8
words to stay granule-aligned.

**`CONTRACT.md` §B — the live set, as a formula.**

| term | formula | bytes at the chosen tile |
| --- | --- | ---: |
| operands | `4·L` — `resident_vmem` | 40,960 |
| **in-body intermediates** | `4·(16·NB + NB)` — `priv_vmem` 16× replicated, `glob_vmem` | 17,408 |
| pipeline buffers × count | **0 — there is no pipeline.** This is D1/`v4`: the count is 1, and the whole finding is that it should be 2 | 0 |
| scratch | `4·((k+16) + b_pad + 32)` — `rowbuf` (+16 is the `store_compressed` overrun guard, **not** a tile pad), `len_vmem`, DMA descriptors | 12,480 |
| inter-buffer padding | measured | 1,344 |
| **total vs budget** | `4·(L + 16·NB + NB + (k+16) + b_pad + 32) + 1,344` | **72,192 B = 70.50 KiB — 13.77%** of 524,288 |

✅ **The formula reproduces `probes/06` Q5's measured high-water exactly**, confirmed twice there — it
is validated, not asserted.

- **Binding term:** ⛔ **none.** `n` is untiled by algorithmic decision, not by capacity; even the
  largest shape in the tree (`n=133120, P=4`) is 180.5 KiB = 34.4%. **4.3× of headroom, unused.**
- **Tile the formula selects:** any. **Tile a throughput curve alone would select:** the same. They
  do not differ, and the constraint that won is the HBM:VMEM pass ratio, not capacity.
- **The composition fact this buys:** `v4`'s second slice buffer is 110.50 KiB (21.6%) and `v3`'s
  straddler scratch on top of it is 150.5 KiB (29.5%). **Both fit together.** Step 3 must not treat
  them as rivals for VMEM.
- **W2 caveat:** the 524,288 B denominator is `SparseCoreInfo.vmem_capacity_bytes` reported in-job;
  `compiler_params` sets **no** `vmem_limit_bytes`, so the budget is inherited, not chosen.
  `probes/06` Q5 leaves the denominator formally open (`kTileSpmemCapacity` is a compiled constant,
  not a string) — **it would have to be wrong by 2× to matter, and at 13.77% it does not.**

**`CONTRACT.md` §D — partial tiles.** `n` is untiled, so **there is no partial tile on `n`.** The
partial unit is a **partial wave**: the last wave has `b mod 32` active subcores. At `b ∈ {64,1024}`
this never fires on the shipping path. When it does, inactive subcores run op `Z` — they compute and
write no garbage.

⚠ **Corrected.** This row used to say "a partial wave costs a full wave, because the kernel costs
`max` over subcores". There is **no inter-subcore barrier** — `sparsecore_topk.py:19` and `:528` say
so, and `lax.fori_loop(0, num_waves, wave_body, 0)` (`:329`) is executed independently per subcore;
the only reductions are lane-wise *within* a subcore (`:152`, `:302`). So the cost is `max` over
subcores of **that subcore's own row-sum**, not `Σ` over waves of the max row. An idle subcore in the
last wave finishes early and waits only at the kernel's single trailing barrier. The practical
consequence is the same — an odd row is exposed on whichever subcore owns it — but the *reasoning*
matters, because it is what unblocks `v6` (§5) and it is why the stride-32 ownership makes
`max ≈ mean`.

**The contract's probe list** — rows whose answer is a measurement, not arithmetic:

| row | probe | what each answer changes |
| --- | --- | --- |
| **L6** on V4 (16× multiplicity) | ✅ **answered** — `probes/07` (`j-098b260f`) then `probes/08` `R1` (`j-300d4690`) | ⚠ the two **disagree by ~5×**: 19.192 µs/level modelled vs 12.195 µs measured *in total*. §7. The measured one governs; `v2`'s prize is 12.195, not 62 |
| **T2**'s ρ | ✅ **answered** — host derivation, no job | ρ = 0.40005 prefill / 0.850 decode. It re-weighted the entire sheet and swapped the #1 and #3 candidates |
| **T5** aliasing on `out_hbm` | ⚠ **partly answered** — `probes/10` found a clean linear output-byte term, **1.118 µs/MiB ≈ 8.9 µs** at the base shape (R² 0.99978), that the input relayout does not explain. Carried to `probes/11` Q3 | `input_output_aliases` + `donate_argnames` is a one-line change worth up to ~8.9 µs, once the term is attributed |
| **W2**'s denominator | open, and deliberately not probed | it would have to be wrong by 2× to bind at 13.77% |

**No `CONTRACT.md` row is tagged `unmodelled`.**

## 5. The version log

---

### `v1` — the incumbent

- **Parent:** none · **Move:** none — this is the root
- **State:** §4 in full
- **Measured:** 326.782 µs at ρ=1 (`j-4e0df30a`), 327.166 (`j-300d4690`) — **cross-job −0.12%** from a
  byte-identical source. Reweighted to production prefill: **189.98 µs**
- **Achieved:** 10.6% of the 20.05 µs floor — §1. ⚠ **Not against a measured number.** 189.98 µs is
  `measured + modelled reweighting`; see the warning in "Numbers in this document". `probes/11` since
  confirmed D3's relayout is a harness artifact, so the production kernel is 155.82 µs (firm) and
  `v1` achieves **12.9%**
- **Verdict:** **KEPT** as the baseline. It is also a candidate in step 3 and it can win.
- **What it taught:** the block is **100% SparseCore, 0% VPU, 0% MXU**, and 39.8% of it belongs to no
  op at all.

---

### `v2` — `R=1`, one shared histogram

- **Parent:** `v1` · **Move:** `[mutate]` drop the 16× lane-private histogram replication:
  `zero_priv(NB//LANES)`, `addr = bucket` instead of `lane*NB + bucket`, `fold_priv` over `range(1)`
- **Motivated by:** D5 — the source suggests `plsc.addupdate_scatter` may resolve intra-vector
  address conflicts in hardware, which would make V4's 16× replication unnecessary for correctness.
  Source read of the `plsc` intrinsic
- **Exactness:** **`exact`, and proved the strongest way available** — output is **bit-identical**,
  1024/1024 rows at b=1024 and 64/64 at b=64 (`j-300d4690`). This is not a reassociation-noise
  argument; there is no reassociation. It also closes every value-mediated mechanism for the delta
- **State diff from `v1`:** `algebra` — 16 private accumulators + a 16-deep fold → 1 shared
  accumulator with hardware conflict resolution. `layout` — V4 `priv_vmem` loses its leading
  replication axis
- **Live set:** 72,192 → **57,152 B = 55.81 KiB, 10.91%** of budget (V4 falls from 16 KiB to 1 KiB).
  Binding term: still none
- **Predicted:** stage D predicted 13.6% of the prefill kernel, from `probes/07`'s 19.192 µs/level
- **Built?** **Yes** — the arm exists as `probes/08-per-wave-residual/arms/arm-R1.py` and is ready to
  lift into `variants/`
- **Measured:** **12.195 µs**, i.e. 314.971 vs 327.166 at ρ=1 (`j-300d4690`). Reweighted:
  **189.98 → 177.79 µs, −6.42%** (6.4% faster). ⚠ **The prize is *larger* at production ρ than at the
  instrument's ρ=1** (−6.42% vs −3.73%) because the sweeps do not scale with ρ while the kernel
  around them shrinks. Correctness: the same job, bit-identical
- **Prediction accuracy:** ⚠ **predicted ~5× high.** 19.192 µs/level × 3.231 levels ≈ 62 µs predicted
  against 12.195 measured. What that says about the floor: `probes/07`'s per-level replication price
  was fitted on a level that also carried its walk, so it attributed the walk to the replication.
  §7 and §8
- **Verdict:** **KEPT ← best**
- **What it taught:** two things that outlive the branch. (a) `plsc.addupdate_scatter` **does**
  resolve intra-vector address conflicts, which is a platform fact and a `references/` promotion
  (§6). (b) `probes/09` then explained the win mechanically: **9.238 of the 12.195 is `zero_priv`
  shrinking from a 256-entry clear to a 16-entry one.** That is why `v2` and `v5`/`v7` are not
  additive (§8).

---

### `v3` — fused emit, no second walk of `L`

- **Parent:** `v2` · **Move:** `[remove]` one pass after L0 writes `> b_star` straight to `rowbuf`
  **and** `== b_star` to a compact straddler buffer of size ~`cnt`; refinement levels then walk
  `cnt`, not `L`; `emit`'s separate traversal disappears
- **Motivated by:** D6 — `emit` (17.7%) and the refinement `hist` (10.6%) are **two more full walks
  of a 10240-element row** to touch `k` and ~`cnt` elements respectively. Stage C″, `j-4e0df30a`
- **Exactness:** **`exact`** — same partition, same tie-break order. `rowbuf`'s existing `+16` guard
  already covers `store_compressed` overrun; the straddler buffer needs its own bound. Backed by the
  same bit-identical row-comparison gate `v2` passed
- **State diff from `v2`:** `algebra` — the L0 walk gains two conditional compressed stores and loses
  its downstream consumers. `live set` — a straddler buffer appears. `grid` — refinement's loop bound
  changes from `walk_chunks` to `cdiv(cnt,16)`
- **Live set:** 57,152 + `4·cnt_max`. ⛔⛔ **`cnt_max = 4096` IS FALSIFIED — this bound is not safe
  and building to it corrupts memory.** The parenthetical reasoning below ("a bucket cannot hold more
  than the row") is wrong: the step-3 `v3` build measured `cnt` reaching **10240**, i.e. the full
  `slice_len`, on an all-negative score distribution. A variant built to 4096 with
  `disable_bounds_checks=True` **overflows by 2.5× and corrupts adjacent scratch.** Size from
  `slice_len` — `v3` uses `slice_len + LANES`, which makes overflow impossible by construction.
  ✅ **The frozen suite does catch this one**, with 1.35× of margin: max `cnt` at ρ=1 is
  `strict_desc` 1536, `random` 3054, **`heavy_ties` 5526**, `all_negative` 7089, and `heavy_ties`
  runs at P=1 — so `test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]` goes red against a
  4096-word buffer. ⚠ **But the margin is one kind and 1.35×**, and `TWO_STAGE_SHAPES` leaves P=1
  with **no coverage of negative winners at all** (`all_negative` is the only kind whose winners are
  negative, hence the only one exercising `monotone_key`'s involution at emit time, and it is `P>1`
  only — `--collect-only` confirms it in exactly 5 ids, 0 at P=1). Adding `all_negative` to
  `test_exact_topk_full_rows`'s kind list closes both, but that test spans all 7 `SHAPES`, so it
  costs **7 ids, 5 at the expensive `N_128K` shapes**; restricting it to the two P=1 shapes — where
  both gaps live — buys the whole fix for **2**. ~~At `cnt_max = 4096` (a bucket cannot hold more than the row):
  **73,536 B = 71.81 KiB, 14.03%.** Binding term: still none. `cnt_max` sizes the **buffer** and
  4096 remains the correct safe bound — do not shrink it to the measured occupancy.~~ What
  `probes/13` changed is the **cost**, which is driven by actual occupancy, not by the allocation
- **Predicted:** ⚠ **RE-PRICED by `probes/13` — and the dominant error was not the level count it was
  sent to check.** Removes the refinement walk (20.08) and `emit`'s traversal, replacing them with ~1
  extra pass over `L` at L0 and `levels × cnt` walks. Old inputs (level 2.231, `cnt` **assumed 4096**)
  gave −8.72 µs = −4.59%. Correct inputs — critical-subcore level **1.570** and the joint statistic
  **`cnt_eff` = 1798** — give **−33.57 µs = −17.51%**. `modelled` on `probes/07`'s measured slopes
- ✅ **RE-PRICED AGAIN by `probes/15` (`j-78f7f537`) — the added term was overcharged 2.08×.**
  `probes/13` charged the fused L0 store at `probes/03`'s `emit1` price: its `added` column
  decomposes to `10.98 + 0.005829·cnt_eff`, and `10.98 = 27.456 × 0.40005` **exactly** — i.e. it
  billed a *whole extra traversal plus a cumsum* for what is a **fused conditional store**. Measured,
  one conditional `store_compressed` inside the L0 walk is **6.675 µs/pass**, only **6.2%** of the
  histogram pass it rides on, and a second costs the same again → **5.29 µs at production ρ** for
  both. ⚠ It is **flat in density** over 0.7%–65% (fitted slope −2.474 µs/unit, R² 0.129 — a negative
  slope where physics demands non-negative), so it is a fixed per-pass cost, not a per-store one
- **Band:** **≈ −20.5% central, −14.9% at the barriered `wavemax` upper bound.** ⚠ Supersedes
  −17.51%/−11.94%, which superseded −16.48% and −13.80%. The old **−20% to −33% band still does not
  exist**, but the corrected figure has now re-entered its lower edge from below
- ⛔⛔ **REFUTED BY THE STEP-3 BUILD. `v3` IS PREDICTED TO LOSE: +1.85% … +6.27% harness /
  +2.27% … +7.71% production.** Every re-pricing above, including the two that felt like corrections,
  shares one false premise: **the fused pass cannot exist.** Plan B's `_l0_fused` wants the level-0
  histogram walk to also classify elements against the pivot bucket — but **`b_star` is the *output*
  of that histogram**, so nothing during the walk knows the pivot yet. The two stores cannot ride an
  existing pass; they need their own full-length walk of `L`. Since `store_compressed` is flat in
  density (the `probes/15` bullet above, 6.675 µs/pass = 0.32×W), `v3` adds **two** of them to a full
  walk (+5.34 µs) to save 0.33 of a walk (−2.74 µs). **Break-even is α = cnt/i_hi = 0.2375; measured
  α is 0.347 prefill, 0.636 decode.** Confirmed at bundle level: the new `split` region is +204
  bundles while the L0 region is **unchanged at 108, with identical opcode sequences.**
- ~~✅ **`v3` cannot be a loss, and the margin is no longer arguable.** Break-even `cnt_eff` moved
  6295 → **8531**, which is **above the global maximum `cnt` of 6495** — so even `probes/13`'s
  deliberately absurd bound F is a win. The node's sign is settled without building it~~
  ⛔ **This is the sentence the build overturned, and the way it was wrong is the lesson: it was
  computed against the impossible pass, so no amount of re-pricing its inputs could have found the
  error.** ⚠ Its risk row 1 signal — "L0's own arm time rises" — **reads clean and would have
  exonerated the change**, because the cost lands on a different pass.
- ⚠ **Two intermediate figures for this node are superseded; ignore them if you meet them elsewhere.**
  −16.48% used a walking-row level count against a global-mean `cnt` (wrong aggregation on both);
  −13.80% then rescaled the `deleted` term by 1.570/2.231, which is wrong because the production
  sheet's refinement row **contains no level count at all** — it is `53.544 × 6/8 × 0.50005`, with
  the executions already baked into a measured quantity. **−17.51% is the figure**
- **Built?** **No** — step-3 variant
- **Measured:** — · **Verdict:** **OPEN**
- **Note:** this is the **largest in-algorithm prize on the board** and it is the one `v10` competes
  with. If `v10` wins, `v3` is subsumed rather than composed.

---

### `v4` — the caller's request: double-buffered, ragged score DMA

- **Parent:** `v3` · **Move:** `[transfer]` two `resident` buffers alternating per wave, `emit_pipeline`
  or a hand-rolled `BufferedRef`; and issue `i_hi` words rounded up to 8, not `slice_len`
- **Motivated by:** **D1** — overlap fraction **0.0000**, one alloca, 3 transfers, 3 waits, two
  literal `snop`s in the compiler's slack (`probes/06` Q1, `/xprof-llo` static analysis). **D8** —
  60% of prefill bytes are never read (host ρ derivation)
- **Exactness:** n/a — a `transfer` move changes no arithmetic
- **State diff from `v3`:** `transfer` — T1 buffer count 1 → 2, T2 padded → ragged. `live set` — one
  more buffer. ⚠ **`boundary` and `grid` both change — restated by stage E (§11).** `boundary`: the
  single `plsc` mesh call gains an **inner pipeline grid** `(n // chunk,)` enclosing exactly
  `cvt ⊕ hist0`. `grid`: no longer "unchanged; still a static `fori_loop` over waves"
- **Live set:** ⚠ **81,728 B = 15.59%** at the chosen within-slice form (`+2·1024·4` on `v3`'s
  73,536) — **24,576 B cheaper than the 114,496 B / 21.84% the cross-wave form would cost.** Binding
  term: still none; §4d already checked coexistence with `v3`'s straddler
- **`chunk` = 1024 words (4 KiB); sweep {512, 1024, 2048} in step 3.** ⛔ **4096 is illegal** —
  `n = 10240 = 2¹¹·5` is **not** a power of two, contradicting `probes/12`'s escape clause. Legal
  chunks are divisors of 10240 that are multiples of `LANES=16`
- **Predicted:** ⚠ **RESTATED — the "≤ 13.469 µs" ceiling was never reachable.** Step 0's DMA is
  never hidden, so the unhidden head is `13.469/(n/chunk)` = 1.347 µs at chunk 1024, and `cvt`'s
  measured −8.20 µs absorption re-emerges once the DMA stops being something to absorb. Optimistic
  `13.469 − 1.347 = 12.122 µs` → **−6.38%**; pessimistic `− 8.20` more → **−2.06%**. `modelled` on
  `measured` inputs. ⚠ **The pessimistic end is the weakest number in stage E**: it subtracts cvt's
  *full* marginal cost on the theory the absorption is banked twice, and a marginal-removal
  measurement is not a decomposition. The chunk sweep measures it directly
  ⚠ **D1 and D8 cap at the same 13.469** — moving fewer bytes and overlapping the bytes you move
  cannot both be paid out of one exposed-transfer budget
- ⚠ **This is `hardware_map.md:675`'s S2 seam, and S2 and `v4` are ONE move — do not sum them.**
  S2's ~10 µs is `modelled` and is contradicted by cvt's **measured** −8.20 µs: the pass is already
  free because it absorbs arrival latency, so S2's prize is ≤ 0 at the incumbent's schedule and
  becomes positive only *inside* this node
- **Built?** **No** — step-3 variant
- **Measured:** — · **Verdict:** **OPEN** — but its one *feasibility* blocker is now closed, see below
- ✅ **`probes/12` closes the mechanism question, with no job.** `pltpu.emit_pipeline` **is** usable
  inside a `plsc.VectorSubcoreMesh` kernel: four SparseCore kernels in this checkout already do it,
  one of them (`ragged_gather_v2.py:182,208`, nested pipelines with a dynamic grid) on the production
  MoE path via `fused_moe_gmm.py:238`. `dma_start_p`/`dma_wait_p` have SparseCore lowering rules
  (`sc_lowering.py:428,533`), and a toy SC kernel in the **exact `v4` shape** — `emit_pipeline` from
  HBM into a persistent `pltpu.VMEM` scratch, then one manual DMA out — lowered cleanly against a
  `jnp.sin` negative control that failed as intended. **The hand-rolled `BufferedRef` fallback is
  withdrawn**; `emit_pipeline` is the repo idiom and no kernel here rolls its own.
  ⚠ Pallas→Mosaic MLIR only — it does not exercise the Mosaic→LLO compile, so VMEM/register pressure
  at LLO is still a job-only risk.
- ⚠ **`probes/12` also splits this node by phase, and the split is structural.** `num_waves =
  _cdiv(b, 32)` (`:443`) is **32 at prefill** (`b=1024`, `P=1`) and **1 at decode** (`b=16` → `P=2` →
  `b_eff=32`). So there are two different `[transfer]` moves wearing one name:
  - **cross-wave**, what the caller described: wave `w+1`'s score DMA overlaps **all** of wave `w`'s
    compute. Valid at prefill, where the 32 waves are, and prefill is the phase §1's denominator is
    measured in — so the 13.469 µs ceiling below is unaffected. **Worth exactly 0 at decode**, where
    the `fori_loop` at `:329` runs once and there is no next wave.
  - **within-slice**: `grid=(n // chunk,)`, with `cvt` (`:171`) and `hist0` (`:187`) fused onto the
    pipeline buffer. Works in **both** phases and subsumes the cross-wave form — but its ceiling is
    lower, because passes 2..N (`REFINEMENT_LEVELS`, `emit_body` `:283`) re-read `resident_vmem`, not
    HBM. It hides the DMA behind **pass 1 only**: ceiling `min(DMA, cvt+hist0)`, not the whole DMA.
  ⚠ These two are **not** additive and must not be summed. Build the within-slice form (it covers
  both phases), and if it underperforms, the cross-wave form is the prefill-only fallback.
- ⚠ **The risk that is specific to this node, and it is not a modelling risk.** `cvt` has a
  **negative marginal cost** — it is already absorbing the transfer's arrival latency
  (`j-300d4690`). A double buffer **competes with that mitigation**. The realistic outcome is well
  under 13.469 µs, and a measured 0 would be a coherent result rather than a broken variant. This is
  why the caller's mechanism, though real, is not the headline. ⚠ The within-slice form makes this
  *sharper*, not milder: it fuses the pipeline onto `cvt`, the very pass whose absorption is the
  competing mitigation.

---

### `v5` — `NUM_BUCKETS` swept

- **Parent:** `v2` · **Move:** `[grid]` `NUM_BUCKETS` 256 → {64, 128, 512}, three compile-time arms
- **Motivated by:** stage C″ — `zero_priv` is a **dense clear, linear in `NUM_BUCKETS`**, and is
  9.238 of the sweep's 18.866. The constant has **never been swept** (D9)
- **Exactness:** **`exact`** — a different radix, the same exact selection
- **State diff from `v2`:** `grid` — one compile-time constant. `live set` — `4·NB` in `glob_vmem`
- **Live set:** 57,152 ± `4·ΔNB`, i.e. ±768 B across the sweep. Negligible; binding term still none
- **Predicted:** ⚠ **The trade REVERSES — `v5` points at a *smaller* radix, not a larger one — but it
  stays `OPEN` on a different term than before.** `probes/13` settled the level half `measured` and
  it needs no hardware: crit levels at NB 64/128/256/512 = 2.375/1.938/1.781/1.469, with **zero
  non-convergence at any radix and zero rows reaching level 5 at NB=64**, so `v5`'s stated failure
  mode (deep walks at a small radix) is dead either way. Extra levels are cheap because they cost
  `walk·ρ` at ρ=0.4; the per-level 256-bucket fixed sweep is not. Priced: NB=64 **−14.82%**, NB=128
  −12.01%, NB=512 **+19.28%**.
  ⚠ **The whole reversal rests on `fixed(NB) = 19.192·NB/256` — a pure proportionality with zero
  intercept, `modelled` from a single anchor.** There is *structural* support (all four sweep
  primitives have NB-proportional trip counts: `zero_priv(NUM_BUCKETS)` `:223`, `zg`/`scan_glob`/
  `find_bucket` at `NUM_BUCKETS//LANES` `:206`/`:208`/`:211`) and `probes/09` measured their sizes at
  NB=256 — **but there is no second `NUM_BUCKETS` value anywhere in `probes/03`, `07` or `09`**
  (probes/03 swept *pass count*, not bucket count), so the intercept `c` is unbounded. Bounding
  `fixed(NB) = c + (19.192−c)·NB/256`: NB=64 goes −14.82% (c=0) → −2.04% → **+4.36%** as `c` rises.
  **NB=64 stops winning once `c` > 11.12 µs = 58% of the NB=256 sweep**, and with four
  `parallel_loop`s per level and ~22 µs of fixed launch cost measured elsewhere
  (`probes/07/README.md:122`), 58% is not implausible
- ⛔ **`v5` is `OPEN` on TWO terms, and the second is worse than a missing number — it is a
  contradiction between two measurements.** `probes/07` prices the sweep at a marginal **19.192
  µs/level**, implying 58.18 µs for the 3.031 executions at ρ=1. `probes/09` measured the sweep
  region directly, four ways, additive to 0.2%: **18.866 µs total**, i.e. ~6.224 µs/level. **A 3.1×
  disagreement, both tagged `measured`**, from different jobs with different arm constructions — and
  all four primitives are `done`-gated identically in source, so nothing structural explains it.
  ⚠ The two live side by side in one document today: stage C″ adopts 09's 18.866 for its sweep row
  while `hardware_map.md:164` still quotes 07's 19.192.
  **At 09's price `v5` collapses from −14.82% to −1.58% and the optimum moves to NB=128 (−3.02%).**
  ⛔ **Do not quote `v5`'s ranking until this is adjudicated.** Its *level* side is `measured` and
  unaffected
- **Mandatory probe (§10):** re-run the `probes/09` sweep split at `NUM_BUCKETS ∈ {64, 512}` with 256
  as an in-pool control, **one arm construction across all three** — which fixes the slope, the
  intercept **and** the 07-vs-09 conflict together, since the NB=256 absolute adjudicates it.
  ✅ **RETURNED — `probes/16` (`j-c47cfec1`), 22 arms in one pool, 12 interleaved rounds**

- ⛔ **VERDICT: `v5` is REFUTED — and *not* by either term it was open on.** Both open terms resolved
  **in `v5`'s favour**, and it lost anyway on a term the model never had:

  | term | resolution |
  | --- | --- |
  | the intercept `c` | ✅ **`c = 0.299 µs`**, from a 3-point fit `fixed(NB) = 0.299 + 0.074870·NB`, r² = **0.99934** (5.375 / 18.967 / 38.845 µs at NB = 64 / 256 / 512). That is **1.6%** of the NB=256 sweep, far below the 11.12 µs break-even. `probes/13`'s zero-intercept proportionality was essentially right |
  | **07 vs 09** | ✅ **09 wins, decisively.** Measured 18.967 µs is **+0.54%** from 09's 18.866 and **−67.39%** from 07's implied 58.17 µs. The arms are byte-identical to 09's (six md5s gated in-job), so this reproduces 09's construction in a fresh pool |
  | ⛔ **the term the model omitted** | The same pool priced `E0-swp` — everything *but* the sweep, i.e. DMA + `cvt` + the dense L0 histogram — at **259.647 / 206.227 / 177.546 µs** at NB = 64 / 256 / 512. **The L0 histogram is strongly NB-dependent in the *opposite* direction**: fewer buckets means more `addupdate_scatter` address collisions, so **+53.420 µs at NB=64** and **−28.681 µs at NB=512** |

  Adding that term (× ρ) turns the headline round: **NB=64 goes from −32.05% to −5.28% ± 11.19%** —
  ⛔ **unresolved, not a win** — and NB=512 goes to **+33.75%**. Two further unmeasured terms bias
  the *same* way (refinement match density is 1/NB; `levels_crit(64)` is a CPU replay), so the
  central estimate is if anything optimistic.

- ⛔ **Do not cite the job log's own `V5_VERDICT: SURVIVES` line.** The probe computed it under the
  briefed premise — sweep-only — that its own `E0-swp` arms falsify. The prober flagged this itself.
- **Built?** **No** · **Measured:** — · **Verdict:** ⛔ **REFUTED.** Not built. The mechanism is real
  and the sweep does shrink linearly with `NB`; it is simply paid for at the histogram, which is
  larger. ⚠ **This is the second time in this exercise that shrinking one region grew another** —
  see `v6`, and §8.
- ⚠ **Not additive with `v2`.** `v2` already shrinks the clear 16×; `v5` shrinks what is left. Measure
  the pair.

---

### `v6` — `keep_all` fast path

- **Parent:** `v2` · **Move:** `[boundary]` `@pl.when(i_hi > k)` around the DMA, `cvt`, `hist0` and
  the sweeps; emit the identity permutation otherwise
- **Motivated by:** D14 — `keep_all = total <= k` and `total ≈ eff_len`, so **`keep_all` ⟺ ρ ≤ 0.2**.
  Prefill chunk ρ is `0.1·c + 0.05005`, so **prefill chunks 0 and 1 of 8 run a full DMA + `cvt` +
  `hist0` + sweep to discover "keep everything"** — a fact `i_hi <= k` decides with no histogram at
  all. Host derivation from `glm-5.2-fp8-dp16-ep.sh`, no job
- **Exactness:** **`exact`** — when `eff_len ≤ k` the top-k *is* the whole row
- **State diff from `v2`:** `boundary` — a second, near-empty path through the kernel. `algebra` —
  the identity emit. ⚠ **Not a pure skip**: the fast path must still write `0..eff_len-1` and `−1`
  padding, so it needs a cheap replacement emit
- ⚠ **Composing with `v4` changes the mechanism — restated by stage E (§11).** `@pl.when(i_hi > k)`
  **cannot wrap `v4`'s pipeline**: that puts `emit_pipeline`'s `run_scoped` allocations under a
  conditional. The repo idiom is a **grid that evaluates to 0** (`ragged_gather_v2.py:102,208`), so
  `v6 ∘ v4` is `grid=(0,)` on the fast path, not a `pl.when`. `v6` standalone is unaffected
- **Live set:** unchanged; the fast path allocates nothing new
- **Predicted:** **~8.9 µs, −4.7%.** `modelled`, and see the two corrections below — the earlier
  "≤ −18.5%, or 0% if a step is mixed" was wrong in both directions. Terms saved on the 25% of
  prefill rows with `token_pos < 2048`: score DMA 3.367 (fixed in ρ) + sweeps 3.908 (fixed) +
  `hist0` 2.119 (ρ-scaled) + `cvt` −0.512 (**negative — removing `cvt` costs time**)
- **Built?** **No** · **Measured:** — · **Verdict:** **OPEN** — nothing blocks it now; it needs a
  build, not a probe

⚠ **Two corrections from the source read, and they cut in opposite directions.**

1. **The prize is ~4× smaller than recorded, because a `keep_all` short-circuit already exists.**
   `sparsecore_topk.py:213` computes `keep_all = total <= k` and `:224` folds it into `done`; when it
   fires, `zero_priv(0)`, `active_walk = 0` (`:226-227`) and `fold_bound = 0` (`:250`) already neuter
   all three refinement levels, and `emit_body` selects `valid` wholesale (`:293`). **The refinement
   passes this node was credited with deleting are already skipped.** What `@pl.when(i_hi > k)` adds
   is only the *earlier* exit: the score DMA, `monotone_key`, `hist0` and `scan_glob`. The stage-C″
   ledger was right about this (refinement carries ×6/8); the `v6` prize double-counted it.
2. ⛔ **The gate this node was blocked on does not exist. `v1` has no inter-subcore barrier**, so
   "one long row in a wave pays for all 32" is **false** — see §4c. Steps *are* mixed in ρ
   (`tpu_runner.py:1154-1215` partitions decode and prefill rows within one batch; one
   `sparsecore_topk` call at `streamindex_topk.py:829-834` covers both), but that no longer matters:
   cost is `max` over subcores of each subcore's **own 32-row sum**, and the stride-32 ownership makes
   those sums near-equal, so a per-row skip converts to wall time roughly proportionally. **The saving
   is unconditional, not gated on homogeneity.**

---

### `v7` — shallow scans

- **Parent:** `v2` · **Move:** `[remove]` replace `scan_glob`'s and `find_bucket`'s 16-step serial
  carries with block-sum → cumsum of the 16 block sums → add back. Depth 16 → 3
- **Motivated by:** D13
- **Exactness:** **`exact`** — integer prefix sums reassociate exactly. No floating point involved
- **State diff from `v2`:** `algebra` on two sweeps only
- **Live set:** unchanged
- **Predicted:** **4.698 µs, −2.5%.** `measured` component (`j-4e0df30a`)
- **Built?** **No** · **Measured:** — · **Verdict:** **PARKED**
- **If PARKED:** ⚠ **it was `#1` for a day and `probes/09` demoted it.** Probe 08's `-sweep` arm read
  151.324 µs and stage D concluded "~92% of the sweep is serial-carry latency, ~139 µs, the largest
  object in the kernel". That arm zeroed bounds feeding `total` and `b_star`, so it deleted the
  sweeps **plus all refinement plus most of emit**. The two carries are **4.698 µs**. Parked on a
  **measured** term, not an unpriced one, so §10's rule is satisfied
- **What would reopen it:** `v5` choosing a **larger** `NUM_BUCKETS`, which lengthens both carries
- **What it taught:** the reason it is parked is worth more than the node — see §8, and the two
  memory notes it produced.

---

### `v10` — SECOND ROOT: sample-and-count instead of radix-partition

- **Parent:** `v1` (a **root**, not a descendant of `v2` — it has no histogram to de-replicate)
- **Move:** `[mutate]` histogram a **1/16 sample** of the row to estimate the k-th threshold, then
  correct it with **count passes** — one compare plus `plsc.all_reduce_population_count` per vector,
  **no memory scatter at all** — then one emit pass
- **Motivated by:** D2 — `hist0`'s dense scatter is **irreducible for a histogram**: one
  `vst.idx.add` per element whatever `NUM_BUCKETS` is, and no tuning move reaches it. The histogram
  block is **28.4%**, the largest structural lever in the kernel. The escape is not a cheaper
  histogram, it is **not needing a full one**
- **Exactness:** **`exact`** — the sampled threshold is only a *starting point*; the correction loop
  runs until the count equals `k` exactly, so the emitted set is the true top-`k`. ⚠ **If step 3 drops
  the correction loop for speed, this node becomes `≈ approximate` and owes a gate** (metric,
  threshold set before the run, and a job). Do not let that happen silently
- **State diff from `v1`:** `algebra` — radix partition → estimate-and-verify. `grid` — the sample
  pass walks `L/16`. `live set` — the 16× `priv_vmem` and most of `glob_vmem` disappear; a small
  threshold state replaces them
- **Live set:** ~**41,000 B ≈ 40 KiB, 7.8%** — the smallest of any node. Binding term: none
- **Predicted:** replaces `hist0` + refinement `hist` (53.99 µs) with 1 sample histogram
  (33.91/16 = 2.12) + count passes. The old modelled rate was 3 bundles/vector against `hist0`'s
  ~7.2 → **~2.4× cheaper**, 2.5 passes × 8.4 = 21.0 µs, total ~171.6 µs, **−9.7%**
- ✅ **RE-PRICED by `probes/15` (`j-78f7f537`) — the `unmodelled` rate this whole root rested on is
  now `measured`, and it went the good way.** A count pass is **20.681 µs**, the dense L0 histogram
  **108.217 µs**, both as slopes over `p ∈ {0,1,2,4}` at R² 0.9998 → **0.191×, i.e. 5.23× cheaper**,
  against the model's 0.248. **1.30× better than modelled.** Central **−9.7% → ≈ −12.2%**
- ✅ **And the rate is at a floor, which is the more useful half.** 20.681 µs sits inside `cvt`'s
  resolution floor (Δ 0.354 µs), so **a count pass already costs exactly a read-only walk and cannot
  get cheaper** — the popcount and scalar carry are free (`cnt − cntv` = 0.489 µs, a null). ⚠ The
  consequence is that **the bimodality is gone: the expensive branch does not exist.** What is still
  open is the *pass count*, and break-even moved to **8.0 count passes** against a modelled 2.5 —
  a 3.2× margin
- **Built?** **No** · **Measured:** — · **Verdict:** **OPEN** — but on a **host-simulable** term
  (how many correction passes the sample needs), not on a hardware rate. It no longer owes a job
- ⚠ **Honest statement of where this root stands.** Its modelled prize (−9.7%) is smaller than
  `v3`'s (−30.2%), and it is *not* obviously the winner. Two things keep it in the graph anyway: it
  is the only node that attacks the largest structural lever rather than a walk around it, and its
  live set is 7.8% — it composes with everything. Two variants that were considered and are **not**
  this node, with why: a **register histogram** (16 compares per vector instead of one scatter) is
  ~71× worse per element on the measured rates and is dead on arithmetic alone; **pure bisection with
  no sample** needs ~11 dependent passes and loses to the incumbent. Both were killed by the price
  sheet, not by taste.

---

### `v11` — retiled output emit  *(new, from stage E)*

- **Parent:** `v1` — an **edge** move, so it composes with every node · **Move:**
  `[boundary]`/`[relayout]`
- **What changes:** declare `out_type = jax.ShapeDtypeStruct((b, k), jnp.int32)` with the consumer's
  tiling instead of the flat `(b*k,)` at `sparsecore_topk.py:451`, and let the out DMA (`:315-319`)
  write tiled rows directly. `out.reshape(b, k)` at `:491` disappears
- **Motivated by:** `probes/11` — `ROOT %reshape.1 = s32[1024,2048]{1,0:T(8,128)} reshape(…call-done)`
  is in the production entry, **10.007 µs `measured (j-9b520877)`** = 1.251 µs/MiB, and carries
  `used_scoped_memory_configs` 4 MiB. **It is the only survivor of D3**
- **Exactness:** **`exact`** — a layout change computes the same function. ⚠ Not "no evidence
  needed": it changes the *store pattern*, so it owes the same bit-identical row gate `v2` passed
- **State diff from `v1`:** `boundary` — the output edge moves from XLA into the kernel. `transfer` —
  ⚠ **corrected by `probes/17`.** Stage E wrote "one 8 KiB contiguous run per row → **16 runs of
  512 B at stride 1024 words**". At the *Mosaic op* level that is **not observable**: the enqueue is
  the same `2048xi32 vmem → 2048xi32 hbm` in both arms, with the row folded into a `memref_slice`
  (+ `memref_squeeze`). The 16×512 B descriptor is implied only by the memref *tiling* that
  `tpu-infer-memref-layout` assigns (2-D i32 → COMPACT `(8,128)`, `tpu_info.py:565-580`), matching
  XLA's `T(8,128)`. Consistent, but **`modelled`, not observed**
- **Live set:** unchanged — same `rowbuf_vmem`, no new buffer
- **Predicted:** ≤ **−5.27%** (10.007 µs) less the strided-DMA cost, which is `unmodelled` — and is
  now the **only** `unmodelled` term left in this node
- **Feasibility:** ✅ **REACHABLE — closed by `probes/17`, `static-compile`, no job.** AOT-compiling
  the real kernel at production dims, the base `ROOT %reshape.4 = s32[1024,2048]{1,0:T(8,128)}`
  (which carries `used_scoped` **4 MiB**, independently re-deriving `stage-e-boundary.md:54`) is
  **gone**; the ROOT becomes the `call-done` itself, already typed `s32[1024,2048]{1,0:T(8,128)}`.
  ENTRY 13 → 12 instructions. Pallas pins only the minor-to-major dim order
  (`jax/_src/tpu_custom_call.py:374-375`) and XLA handed the custom call the consumer's tiling.
  **Patch is four edits**: `:451`, `:317`, `:323`, `:491`. ⚠ **The indexer form matters** —
  `o.at[pl.ds(row,1)]` fails with `DMA source and target shape mismatch`; `o.at[row]` and
  `o.at[row, pl.ds(0,k)]` both lower
- **Built?** **No** · **Measured:** — · **Verdict:** **OPEN**. Both of its conditions are now closed
  favourably — feasibility by `probes/17`, in-model by `probes/14` Q2 — so nothing conditional
  remains. ⛔ It stays `OPEN` rather than becoming `PARKED` only because the strided-DMA term is
  `unmodelled`, per `TEMPLATE-design.md` §10
- **Note:** ✅ **The in-model condition is resolved, and favourably.** The worry (§11,
  `stage-e-boundary.md` §3c) was that compiling `streamindex_topk` standalone forces its ROOT
  layout, so the 10.007 µs retile might be an artifact of where the compilation was cut. `probes/14`
  Q2 compiled the **real** entry: the retile survives, as an `S(1)` staging copy rather than a
  `ROOT reshape`. A different op, the same **8 MiB** of movement (⛔ **not 40 MiB, as this line said
  until step 3** — 40 MiB is `v12`'s input buffer; `s32[1024,2048]` is 8 MiB, which is what
  10.007 µs ÷ 1.251 µs/MiB independently gives), so the prize stands and the "REJECTED
  with prize 0" leg is withdrawn. ⚠ Two carry-overs from that answer: the price transfers by *shape*
  and not by op identity — 10.007 µs was measured on the ROOT reshape, and no arm has yet deleted the
  staging copy — and `v11` remains **subsumed by H3**, since a fused consumer has no HBM output to
  retile at all.

---

### `v12` (= **U2**) — `f32` kernel input, reinterpret in-register  *(new, from stage E)*

- **Parent:** `v1` — an **edge** move on the *input* side, so it composes with every node, and with
  `v11` · **Move:** `[boundary]`/`[dtype]`
- **What changes:** stop asking XLA for a dtype change the kernel does not need.
  `sparsecore_topk.py:438` does `words = jax.lax.bitcast_convert_type(scores, jnp.int32)` *outside*
  the kernel; instead pass `scores` through as `f32` and reinterpret each vector inside
- **Motivated by:** `probes/11` + stage E — the emitted `%bitcast-convert_bitcast_fusion` is a
  `kLoop` fusion carrying `used_scoped_memory_configs` **16 MiB @ ms1**, where the genuinely-free
  flatten beside it carries **no `backend_config` at all**. ✅ `probes/14` then answered the
  ±26.251 µs leg outright rather than leaving this node to dissolve it: XLA gives
  `%bitcast_convert_type` a **separate, disjoint, exactly-adjacent 40 MiB temp buffer** from its
  operand in the real production entry (`1753088 + 41943040 = 43696128`), so it is materialised and
  production pays it
- **Exactness:** **`exact`** — a reinterpret computes the same function. ⚠ But it owes the same
  bit-identical row gate `v11` does, and one extra risk: `monotone_key` (`:116-118`) writes an
  arbitrary int32 pattern **through an f32-typed VMEM store**, so NaN-payload bit-preservation is
  `assumed`, not proven
- **State diff from `v1`:** `boundary` — the input edge moves from XLA into the kernel. `dtype` —
  `resident_vmem` becomes `f32`; two `tpu.bitcast` ops appear in the Mosaic body (0 today)
- **Live set:** unchanged — same bytes, different declared element type
- **Predicted:** **−13.82%** on this file's harness denominator, ⇒ **≈ −16.8% in production**
  (26.251 / 155.82). ✅ **Firm, not conditional.** The "0 or −13.82%" this row carried until
  `probes/14` was the two legs of "is the fusion on the production path"; Q1 answered *yes* on buffer
  assignment, and separately showed the harness fusion's extra reshape/retile is itself a free
  `bitcast`, so its measured price transfers rather than over-charging. That makes `v12` the
  **largest single edge prize in the file**, and it is an edge move that composes with every node
- **Feasibility:** ✅ **REACHABLE — closed by `probes/17`, `static-compile`, no job.** The ENTRY diff
  replaces the fusion with `%bitcast.3 = f32[128,80,8,128] bitcast(%s.1)` — free, no `backend_config`.
  ⚠ **But it is three edits, not one: the DMA is dtype-checked, not byte-oriented.** The minimal
  one-line edit fails to lower with `'tpu.enqueue_dma' op DMA source and target element type
  mismatch`, so the scratch dtype must change too — `:438` (`words = scores`), `:455`
  (scratch `int32 → float32`), and in-register `bitcast_convert_type` at `:172-173` and `:178`.
  That form lowers. Ref-level `.bitcast` would avoid the two per-load bitcasts but is unavailable on
  a 1-D ref (`jax/_src/state/utils.py:98` indexes `shape[-2]` unconditionally); it works on a 2-D
  ref, so a reshaped scratch is a fallback ~~**if the two bitcasts ever price**~~
  ⛔ **RETIRED by the step-3 `v12` build. Do not spend a structural change on this.** The bitcasts
  price at **exactly zero** — 1364 SparseCore bundles in both arms, `static-compile`. ⚠ And
  `probes/17` **undercounted them 3.5×**: it reported 2, the real kernel has **7** in Mosaic MLIR,
  becoming 35 LLVM instructions under `unroll=4`. Harmless here because the prize was zero either
  way, but the method generalises badly — **a toy-kernel op count is not the production op count.**
- **Built?** **No** · **Measured:** — · **Verdict:** **OPEN**, mandatory. Feasibility is closed
  (`probes/17`) and the production-path question is closed (`probes/14` Q1). The one term still
  `unmodelled` is the price of the two in-body `tpu.bitcast`s — which is why this is `OPEN` and not
  `PARKED`, and it is a term that can only *reduce* the prize, never reverse its sign
- **Note:** ✅ **U2 and `v11` are independent and compose** — `probes/17` confirmed each variant's
  ENTRY still carries the other's op (16 MiB and 4 MiB respectively).

## 6. Instruments — every probe, capture and job

Raw output under `probes/`; nothing pasted here.

| # | instrument | question | folder / job | what it changed |
| --- | --- | --- | --- | --- |
| 1 | step-1 microbenchmark | baseline at both shipping shapes | `instruments/`, `j-…` (report.md §7) | the baseline every node is scored against. **READ-ONLY since step 1** |
| 2 | probe `01`–`05` | pass prices, refinement level count, two-stage correctness | `probes/01`…`05/` | the price sheet §4a is built on |
| 3 | probe `03-passes-9216` | µs per VMEM-resident pass at the shipping shape | `probes/03-…/`, `j-fa60ff2a` | the `cvt` / scatter / emit rates every model above uses. ⛔ its *shares* superseded twice |
| 4 | `/xprof-llo` capture | occupancy, VLIW bundles, VMEM high-water, Mosaic MLIR, DMA overlap | `probes/06-llo-incumbent/`, `j-…` | **T4 = 0.0000**, the 574 ps/bundle calibration, the 1,168-bundle program size, and W1's validation. Also **refuted M1** for free, by re-reading the dump instead of re-capturing |
| 5 | probe `07-refinement-fixed-vs-walk` | is a refinement level fixed cost or walk cost? | `probes/07-…/`, `j-098b260f` | split the level; ⚠ **overstated the replication price ~5×** — §7 |
| 6 | probe `08-per-wave-residual` | what is the 81 µs that belongs to no op? | `probes/08-…/`, `j-300d4690` | closed **D4**; cut the score DMA to 13.469; answered **D5** bit-exactly (`v2`); found `cvt`'s **negative** marginal cost. ⛔ three of its conclusions retracted by #7 |
| 7 | probe `09-sweep-split` | split the sweep; de-contaminate the residual; measure the wave slope | `probes/09-…/`, `j-4e0df30a` | **retracted** the 151 µs sweep, the 1.625× super-additivity and the per-wave framing. Produced stage **C″**. Demoted `v7`, promoted **D3 to 39.8%** |
| 8 | probe `10-residual-driver` | does D3 scale with input bytes, output bytes, or rows? | `probes/10-…/`, `j-770c6de9` | **buffer bytes, no per-row term** (excluded at 0.886 ns/row by an unconfounded contrast). Withdrew §1's pessimistic leg; closed **D11**; and its HLO block put the mechanism **outside the Pallas kernel** |
| 9 | probe `11-relayout-provenance` | is the relayout in the production call path, or only under the harness's `jit`? | `probes/11-…/` | ⏳ **IN FLIGHT.** Decides whether D3 is a ~73 µs handoff or a 40% overstatement of the denominator. Also carries probe 10's unexplained output-byte term |
| 10 | host derivation of ρ | what is `E[eff_len]/n` in production? | no job — `glm-5.2-fp8-dp16-ep.sh` + `streamindex_topk.py:482-506` | **re-weighted the entire balance sheet** and swapped the #1 and #3 candidates. The cheapest instrument in the exercise |

| 11 | probe `15-mandatory-v3-v10` | the two mandatory rates: a count pass vs the dense L0 histogram (`v10`), and a fused conditional store as a **slope in density** (`v3`) | `probes/15-…/`, `j-78f7f537` | closed **both** mandatory terms in one pool, one job. `v10`'s rate is **1.30× better than modelled** and **at the walk floor**, so its bimodality collapses onto the cheap branch; `v3`'s store was overcharged **2.08×** by `probes/13`. Both nodes grew, so it does **not** adjudicate between them. ⚠ Also found the bug in `instruments/sctopk_2stage_bench.py:851` — see §7 |
| 12 | probe `17-emit-and-input-dtype-feasibility` | are `v11` and `v12`/U2 reachable at all? | `probes/17-…/`, **no job** — AOT `static-compile` | closed both feasibility terms. Found U2 is **three edits, not one** (`tpu.enqueue_dma` is dtype-checked); corrected stage E's 16×512 B DMA model to `modelled`; confirmed the two compose. ⚠ **Method worth reusing:** `topologies.get_topology_desc("tpu7x:2x2x1")` runs the full XLA-TPU + Mosaic **SparseCore** pipeline on the host with **no TPU attached** — the 8 MiB-scratch control fails with a SparseCore `spmem` allocator error, which is what makes "AOT COMPILED ok" mean the backend accepted it |
| 13 | probe `16-num-buckets-scaling` | fix the sweep's `NB` slope **and** intercept, and adjudicate the 3.1× `probes/07`-vs-`probes/09` conflict | `probes/16-…/`, `j-c47cfec1` (22 arms, one pool, 12 interleaved rounds; floor **1.859 µs**) | ✅ **both, plus a third thing that killed the node.** `fixed(NB) = 0.299 + 0.074870·NB`, r² 0.99934 — intercept 1.6% of the sweep. **09 wins the conflict**: 18.967 measured vs 09's 18.866 (+0.54%) and 07's implied 58.17 (−67.39%). ⛔ But `E0-swp` shows the **L0 histogram is NB-dependent the other way** (+53.420 µs at NB=64), which **REFUTES `v5`**. Also: `find_bucket` unresolved at every radix — stop carrying 1.200. Also: ⛔ **retracts `probes/15`'s instrument-bug claim in full** (§7) |
| — | *(main thread, no job)* | re-verify that retraction rather than take it on trust | four lines of `numpy` | ✅ **confirmed, and more strongly than `probes/16` claimed**: the mask density is **1.000000 under *both* readings**, so the claim fails even granting its own false premise. §7 |

**Instrument runs that changed nothing:** none — every probe above moved at least one verdict, which
is itself unusual enough to note rather than take as a good sign. See §8.

**Promotions to `references/`** — platform facts, not facts about this block:

1. **`plsc.addupdate_scatter` resolves intra-vector address conflicts in hardware.** Lane-private
   replication is not required for correctness on this machine (`j-300d4690`, bit-identical).
2. **The SparseCore relayout model is not the TensorCore one.** 16 int32 lanes, no sublane axis, no
   MXU; `hardware-facts.md`'s 8×128 tile model must be scoped or it silently mis-prices every
   SparseCore design.
3. **A SparseCore Pallas kernel can be compiled to optimized HLO with no TPU attached**
   (`probes/17`). `jax.jit(...).lower(...).compile()` against
   `topologies.get_topology_desc("tpu7x:2x2x1")` runs the real XLA-TPU + Mosaic SparseCore backend
   on the host. That makes "does this lower, and what does it do to the ENTRY computation" a
   **seconds-long local check instead of a pod job** — two feasibility terms closed here for zero
   cluster time. ⚠ It needs a negative control to mean anything: an 8 MiB VMEM scratch must fail
   with `E3000: CompileTimeSparseCoreAllocationFailure … memory space spmem`, which proves the
   SparseCore backend ran rather than the MLIR verifier alone.
4. ⛔ **The image tag this exercise has been measuring against is not immutable, and it moved.**
   `probes/16` found that tag `fa63d6df` now ships `sparsecore_topk.py` md5 `1a89d451…`, **not** the
   `44353b34…` (commit `b11b6eb6`) that `probes/07`–`09` measured. `j-8c6d3032` then died on
   `ImagePullBackOff` because the full-sha tag is gone from the registry, and `j-a70d48d9` was
   correctly refused by the in-job md5 gate. Probes 14 and 15 hit the same gate independently.
   **Any probe re-running against that tag is measuring different source than probes 07–11 did**, and
   nothing in the tag's name says so. Two consequences for step 3: ship the subject under `subject/`
   and `--map-dir` it, and **keep the in-job md5 gate on every arm** — it has now caught this four
   times and has never produced a false positive.

## 7. What `references/` got wrong

| the claim, and where | what it says | what you measured (job id) | class | what the file should say | fixed? |
| --- | --- | --- | --- | --- | --- |
| `hardware-facts.md` § Relayout | 8×128 tiles, a 128-wide lane axis, a sublane axis, and a relayout tax between them | In-job `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16, vmem_capacity_bytes=524288, dma_granule_size_bytes=32)` — **16 lanes, no sublane axis** | **out of scope** | scope the section to the TensorCore and add a SparseCore counterpart, or a one-line "does not apply to `plsc`" | **no** — flagged in `hardware_map.md`; a `references/` edit is out of this step's scope |
| `references/` HBM rates | 337 GB/s for HBM-bound streaming; no rate at all for a **VMEM-resident SparseCore pass** | `probes/03` measures VMEM-resident passes directly; the SparseCore HBM *read* rate is 2138 GB/s with a 21.9 µs intercept, and the shipping 40 KiB slice runs at 1255 GB/s | **narrower than stated** | add the VMEM-resident pass rates and say the HBM figure is asymptotic, with the intercept | **no** |
| `hardware_map.md` L6, from `probes/07` (`j-098b260f`) — an internal claim, held to the same standard | V4's 16× multiplicity costs **19.192 µs per level = 13.6% of the prefill kernel**, i.e. ~62 µs over 3.231 levels | ⚠ `R1` removes the replication entirely, bit-exactly, for **12.195 µs total** (`j-300d4690`) — **~5× less** | **wrong** | the per-level fit attributed the level's *walk* to its replication. Edges argued from it: `v2`'s prediction (measured anyway, so its verdict is unaffected) and `v7`'s original promotion (**already demoted**, on independent evidence) | ⚠ **partly** — recorded in §4b and §8; the L6 cell now carries both numbers |

| ⛔ ~~`instruments/sctopk_2stage_bench.py:246` — our own step-1 instrument, found by `probes/15`~~ | ~~the `scatter` arm is a 100% dense L0 histogram~~ | ⛔ **RETRACTED IN FULL by `probes/16`, and re-verified by the main thread.** The claim was that `m = key > jnp.int32(KEY_NEG_INF)` runs over **raw fp32 bits** and is therefore false for every negative score, making the arm ~50% dense at `n = 9216`. **All three parts are wrong** — see the retraction note below | ⛔ **no error** | none — the arm is 100% dense as designed | n/a |

**Three rows, and one of them is ours** — `probes/07`'s L6 claim. ⛔ **The fourth row, the instrument
bug, is retracted; the count in this paragraph used to read "Four rows, and two are ours".** The
principle it was making stands and is worth keeping: an audit that only finds fault outside the
exercise has not been run honestly. But manufacturing a fourth row is the other failure mode, and
that is what happened here.

⛔ **The retraction, in detail, because it was propagated to three documents and a user-facing
summary before it was checked.** `probes/16` refuted it and the main thread re-derived the
arithmetic independently rather than taking the refutation on trust:

| the claim | why it is false |
| --- | --- |
| "the mask runs over **raw fp32 bits**" | `resident_vmem` holds **monotone keys**, written at `:826` by the XOR the arm itself performs. The local is literally named `key` |
| "`KEY_NEG_INF` is the raw −inf bit pattern, so negatives fall below it" | `sparsecore_topk.py:50-51`: `KEY_NEG_INF = BITS_NEG_INF ^ 0x7FFFFFFF = 0x807FFFFF = −2139095041` — the **monotone key** of −inf, sitting at the bottom of int32 by construction. That is the whole point of the transform |
| "~50% of elements are masked out" | ⛔ **the density is 1.000000 under *both* readings.** The main thread computed it on 655360 `standard_normal` draws: monotone-key density 1.0, **raw-bits density also 1.0**. Raw `−1.0` is `0xBF800000` = −1082130432, which is *greater* than −2139095041. So even granting the false premise, the conclusion does not follow. Only `−0.0` and subnormals differ |
| "it runs at **n = 9216**, not 10240" | `9216` is `--max-model-len`; `_topk_n(9216) = 10240`, and `probes/03`'s own log line reads `n=10240 slice_len=10240` |

✅ **Consequence: `hist0` is not understated, and `v10` gains nothing from this.** The 33.91 µs in
the C″ sheet comes from `probes/08`'s `-sweep` residual, not from this arm at all. The 116.350 and
108.217 µs figures are comparable and differ by −6.99%, an ordinary cross-pool reproduction.

⚠ **The lesson is not "probe 15 erred".** It is that a bug found in a *frozen, read-only* instrument
is the most tempting kind of finding to propagate unchecked — it explains a discrepancy, it flatters
the node we already liked, and the freeze supplies a reason not to go and test it. The check that
settled it was four lines of arithmetic on a laptop. Run it before writing the row, not after.

## 8. The self-audit — findings that favour the winner

`v2` is the only measured node, so "the winner" here means the *shortlist* — which findings made the
field smaller, and would they survive being treated as a `references/` claim?

**There were seven findings that narrowed the field. Six pointed toward keeping the shortlist small
and one pointed the other way. That asymmetry is stated rather than left to be counted** — and two
of the six have already been withdrawn by a later probe, which is the reason the audit is worth the
page.

| the finding | what it did to the ranking | evidence behind it | does that evidence discriminate, or is it merely consistent? | survives? |
| --- | --- | --- | --- | --- |
| **The regions are super-additive by 1.625×** | divided *every* candidate's modelled prize, shrinking all of them at once | five single-arm deltas summing to 366.628 vs a joint of 225.582 (`j-300d4690`) | ⛔ **merely consistent.** Contention explains the sum; so does one arm deleting three regions, and nothing in the job distinguished them. The joint arm was measured; the *attribution* was not | ⛔ **WITHDRAWN** by `j-4e0df30a` — regions are additive to 2.3%. Reopens nothing (it shrank everything uniformly) but **every prize in this file is un-divided** |
| **The sweep is 151.324 µs and ~92% of it is serial-carry latency** | promoted `v7`/D13 to #1 and pushed `v3` and `v10` down the list for a day | one deletion arm (`j-300d4690`) | ⛔ **merely consistent.** The arm's name said "sweep"; its patch zeroed bounds feeding `total` and `b_star`. No control isolated the cascade | ⛔ **WITHDRAWN** by `j-4e0df30a`. **`v7` moved to PARKED**; `v3` and `v10` recovered their places |
| **The residual is 2.561 µs/wave** | made a per-wave barrier the leading D3 mechanism and implied a `(waves−1)/waves` prize | 81.95 µs ÷ 32 waves | ⛔ **merely consistent — it is a division, not a measurement.** It assumed the proportionality under test | ⛔ **WITHDRAWN** by direct slope measurement, 0.009 µs/wave. **The term survived at 75.683 µs**; only its shape was wrong |
| **`hist0`'s scatter is irreducible for a histogram** | closed every "tune the histogram" branch and forced D2 to be a second root | `probes/03`'s price sheet: one `vst.idx.add` per element regardless of `NUM_BUCKETS`, and a measured 4.40× issue ratio | ✅ **discriminates.** It also survives an adversarial re-derivation: a **register** histogram costs ~71× more per element on the same measured rates, so the alternative was priced and rejected, not assumed away | **HOLDS** |
| **D4 — the launch intercept is 0** | deleted a whole candidate ("decode is 43.2% launch") | trivial-op launch 20.193 µs vs a 19.630 µs null; one extra launch of *this* kernel at constant work costs −0.704 (`j-300d4690`) | ✅ **discriminates** — two independent arms, opposite signs, both landing at zero. ⚠ Note the prior: two earlier fits had agreed to 3.2% *with each other* and were both wrong, because they shared a confound. Agreement between two derivations is not independence | **HOLDS** |
| **`cvt` has a negative marginal cost** | ⚠ **the one finding that made the field *harder*** — it cut `v4`, the caller's own request, from a modelled −7.18% to a measured ceiling of −4.12% and killed a −25.2% branch | deleting a 640-chunk traversal made the kernel 20.5 µs slower (`j-300d4690`) | ✅ **discriminates**, and it was verified against the obvious artifact — the arm source was diffed to confirm the patch took before the sign was believed | **HOLDS** |
| **D5 is bit-exact** | promoted `v2` to best and made a `mutate` edge free | 1024/1024 and 64/64 rows bit-identical (`j-300d4690`) | ⚠ **WEAKENED as a correctness argument, though not as a result.** `probes/08` also showed the row-comparison gate is **vacuous for `monotone_key`**: scores are `rng.standard_normal`, `monotone_key` only rewrites negatives, and with k=2048 of n=10240 the top-k is entirely positive. The gate cannot see a `monotone_key` bug. ⚠ **The source read makes this worse than §8 first recorded**: production rows carry *hundreds of exact ties* and rows whose maximum is exactly 0.0 (`test_dsv4_sparsecore_topk.py:64-70`), because scores are a weighted sum of **ReLU'd** fp8 dot products — the one regime where a reordered emit can differ and a Gaussian input never will. `v2` does not touch `monotone_key` or tie order, so its verdict stands — but **`v3`, `v6` and `v10` all reorder or replace the emit and must not lean on this gate** | **WEAKENED** — and it changes what step 3 owes: a negative-heavy input case, or a distinct `out_hbm` sentinel |

**What the audit changes.** Nothing is reopened by a `WITHDRAWN` row here — `v7`'s demotion is
already reflected in §3 and §5. What it does change is procedure, twice: **no prize in this file is
divided by 1.625**, and **step 3's correctness gate needs a negative-heavy input** before `v3` or
`v10` can claim exactness from it.

⚠ **Three of seven findings were withdrawn by one probe, and all three came from the same job.**
`j-300d4690` produced four results that held and three that did not, and the three that failed all
failed the same way — an arm was believed to have deleted what its name said. That is a fact about
how this exercise measures, not about that job, and it is why §9 puts a de-contamination control in
every step-3 plan.

### 8b. The pattern that killed two nodes, and one finding of our own that had to be withdrawn

⛔ **`v5` and `v6` both died the same death, and it is not the death either was expected to die.**
Both were priced by *deleting work from one region*, both had that arithmetic checked, and both were
right about the region they named:

| | it deletes | it was right that | it lost because |
| --- | --- | --- | --- |
| `v5` | the 256-bucket fixed sweep, by shrinking `NB` | the sweep really is ≈ proportional to `NB` (r² 0.99934, intercept 1.6%) | the **L0 histogram moves the other way** — fewer buckets, more `addupdate_scatter` address collisions, **+53.420 µs at NB=64** |
| `v6` | the sweep on rows with `token_pos < 2048` | those rows really are a quarter of prefill | an **outer `lax.cond` already elides the entire kernel** on exactly those steps |

**The common failure is a missing term, not a wrong number.** In both cases the model priced the
region the change touches and had **no term at all** for the thing that absorbs the saving. That is
invisible to any amount of care applied *within* the model, and it is why both were caught by a
probe that measured something the brief did not ask about — `probes/16` priced `E0-swp` as a
control, and `probes/14` enumerated a trace for a different question. ⚠ **Step 3's rule follows
directly: every arm measures the whole kernel, not the region it edits.** An arm that only reports
its own region cannot see this class of loss. Compare
`freed-device-time-spills-onto-untouched-ops` — the same phenomenon with the sign flipped.

⛔ **And one finding of *ours* was withdrawn this wave, which belongs in an audit that is honest.**
`probes/15` reported a bug in our own frozen step-1 instrument — that the "dense" histogram arm was
~50% dense — and I propagated it into three documents and a user-facing summary before checking it.
`probes/16` refuted it; the main thread then re-derived the arithmetic and found it fails **twice
over** (§7). It was the most tempting shape of finding available: it explained a live discrepancy,
it made the node we already liked worth *more*, and the instrument's read-only freeze supplied a
ready reason not to go and test it. The check that settled it took four lines and no hardware.
**The audit's own rule — does this evidence discriminate, or is it merely consistent? — was not
applied to it, because it arrived as a bug report rather than as a claim.**

## 9. The implementation plan — what step 3 builds

Ranked. `REJECTED` nodes get no entry; `v7` is `PARKED` and gets a pointer (§5). **Each entry is
exactly one kernel call** — the boundary never splits in this block, because `P=1` leaves no
partition loop and `n` is untiled.

⚠ **That sentence is true about the *middle* and was mistaken for an answer about the whole
boundary.** Stage E (`stage-e-boundary.md`, spliced summary in **§11**) found that **three of the
four largest non-kernel numbers in this exercise live on the edges**, and that `v4` creates a new
*internal* edge. The kernel does stay one `plsc` mesh call; what changes is that the edges are no
longer assumed optimal. In particular the input edge carries a deletable XLA op worth **26.251 µs**
(✅ now firm — `probes/14` Q1), and the output edge a measured 10.007 µs retiling (✅ now confirmed
live in-model — Q2). ⛔ **This block used to end "neither reachable by any plan below". That is no
longer true** — both became nodes, `v12` and `v11`, and both are in the shortlist immediately below.

### 9a. The closing shortlist — ⛔ this supersedes the `rank N` in the plan headers

The `rank 1`…`rank 6` in the six plan headings below were assigned **before** the probe wave and are
now wrong in three places: `v3` and `v10` both grew, `v6` is rejected outright, and two edge nodes
(`v11`, `v12`) did not exist when the ranks were written. The headings are left alone so the plans
stay findable; **this table is the ranking.**

| # | node | plan | harness `Δ` | production `Δ` (× 1.219) | evidence | composes? |
| ---: | --- | --- | ---: | ---: | --- | --- |
| **1** | **`v3`** fused emit | B | **−20.5%** | **−25.0%** | `measured` rate (`probes/15`); **sign settled without building it** — break-even `cnt_eff` 8531 > global max 6495 | ⛔ competes with `v10` |
| **2** | **`v12`** (= U2) `f32` input | — *(edge)* | **−13.82%** | **−16.8%** | `static-compile` ×2 (`probes/17` reachable, `probes/14` materialised); **firm** | ✅ **with everything** |
| **3** | **`v10`** sample-and-count | E | **−12.2%** | **−14.9%** | `measured` count-pass rate, **at the read-only walk floor** (`probes/15`) | ⛔ competes with `v3` |
| **4** | **`v4`** pipelined score DMA | C | ≤ **−7.09%** | ≤ **−8.64%** | ceiling, `modelled`; the caller's original request | yes |
| **5** | **`v2`** `R=1` | A | **−6.42%** | **−7.83%** | ✅ **`measured` and bit-exact** (`j-300d4690`) — the only measured node | yes |
| **6** | **`v11`** retiled output emit | — *(edge)* | ≤ **−5.27%** | ≤ **−6.42%** | `measured` 10.007 µs; both conditions discharged (`probes/17`, `probes/14` Q2) | ✅ with everything **except H3**, which subsumes it |
| — | ~~`v5`~~ `NUM_BUCKETS` swept | D | ~~−14.82%~~ | ⛔ **REFUTED** | `probes/16` (`j-c47cfec1`) won **both** its open terms and it lost anyway: the sweep shrinks with `NB`, but the L0 histogram grows faster (+53.420 µs at NB=64). Nets to **−5.28% ± 11.19%** — unresolved | — |
| — | ~~`v6`~~ keep-all fast path | F | ~~−4.7%~~ | ⛔ **REJECTED** | the outer runtime gate already elides the whole kernel on exactly the steps it fires on (`probes/14`) | — |
| — | `v7` shallow scans | — | −2.5% | −3.0% | `PARKED` (§5) | yes |

**Three things this ranking says that the prize column alone does not.**

- **The two edge nodes are the cheapest wins per line of code in the file.** `v12` is three edits
  and `v11` is four, neither touches the algorithm, both are `exact`, and together they are worth
  **≈ −23% production** — more than `v10`, and they compose with whichever root wins. ⚠ Their prizes
  are `measured` on the *harness's* ops; what transfers is the shape, argued in §1, not the op
  identity.
- **The top of the ranking is a fork, not a stack.** `v3` and `v10` are alternative second roots and
  the probe wave grew both without adjudicating between them. Step 3 must build both; the ranking
  cannot decide it, because their prizes rest on different `measured` rates and neither dominates
  the other under the ρ = 0.40005 reweighting the whole sheet is predicated on.
- **Rank 6 is where step 3 starts.** `v2` is last but one on prize and **first** on build order: it
  is already written, already bit-exact and already measured, so lifting it into `variants/`
  validates the step-3 harness against a known answer before any unmeasured arm runs.

⚠ **Step 3 added one gating condition to this table, from `v11`'s build.** `v11` is a **`P=1`-only**
win. At `P>1` it does not fail — it *relocates* the relayout to the stage-1/stage-2 seam, where it is
larger, and its ENTRY op count grows at all four `P>1` shapes. Both production shapes are `P=1`
(`report.md` §4), so the candidate survives whole; but the consequence for measurement is general and
binds every arm in this exercise: **`b=16` is `P=2` at `--ctx 9216` while `b=64` and `b=1024` are
`P=1`, so a per-arm summary spanning them is summing two different kernels.** Pooling a `P=1` win
with a `P>1` loss reports a null. This is also a standing warning on whichever node ever touches
`_pick_partition`.

⛔ **The shortlist is six nodes, and two of the original eight died on the same mechanism.** `v5` and
`v6` were both priced by deleting work from one region, and both lost because the deleted work was
being paid for somewhere the model did not have a term. That is a pattern, not two accidents — see
§8. Every surviving row above prices a region the price sheet **measured**, and the three largest
(`v3`, `v12`, `v10`) rest on rates that were re-measured this wave.

⚠ **Every `Δ time %` in the plans below is in the harness denominator (189.98 µs).** `probes/11`
showed the harness is 34.165 µs larger than production, so **multiply each by 1.219** to get the
production figure — see "The denominator, stated once". ✅ This is now one factor rather than the
1.219–1.466 range this line used to carry: `probes/14` closed the second leg by showing production
*does* pay the 26.251 µs `bitcast-convert`, so it stays inside the denominator. The µs
figures need no adjustment, the ranking is unchanged, and `results.md` will quote production
percentages directly because step-3 arms build their own production-shaped entry.

| plan | µs | harness `Δ` | production `Δ` (× 1.219) |
| --- | ---: | ---: | --- |
| A · `v2` `R=1` | 12.195 **measured** | −6.42% | **−7.83%** |
| C · `v4` pipelined DMA | ≤ 13.469 ceiling | −7.09% | **−8.64%** |
| F · `v6` keep-all fast path | ~8.9 modelled | −4.7% | ⛔ **REJECTED** by `probes/14` — the outer runtime gate already elides the whole kernel on exactly the steps this fires on |
| (`v7`, parked) | 4.698 measured | −2.5% | −3.0% |

`v3` (Plan B), `v5` (Plan D) and `v10` (Plan E) are quoted against their own parents, and the same
factor applies to those too.

---

### Plan A — `R=1`, one shared histogram (implements `v2`) · rank 1

**1. What it is.** The incumbent keeps sixteen private copies of its 256-bucket histogram, one per
lane, so that sixteen lanes scattering into the same bucket in the same vector cannot collide; a
sixteen-deep fold then sums them. The hardware does not need this — `plsc.addupdate_scatter`
resolves intra-vector address conflicts itself. Plan A deletes the replication: the histogram becomes
one 256-entry array, the scatter address becomes the bucket rather than `lane*256 + bucket`, and the
fold collapses to a single pass. The clear that zeroes the histogram before each level shrinks from
256 entries to 16, which is where most of the saving comes from.

**2. The call chain.** All existing; no new files.

⛔ **CORRECTED by the step-3 `v2` build — this list was three edits and it needed four. Taken
literally, together with the `v2` node's 1 KiB `priv_vmem` allocation, it is memory-corrupting.**
With `priv_vmem` shrunk to 272 words, leaving `zero_priv(NUM_BUCKETS)` at **`:224`** writes
`LANES * NUM_BUCKETS = 4096` words into a 272-word ref on **every unconverged refinement level**,
over the top of `glob_vmem`. The probe that priced this plan could not have caught it:
`probes/08/arms/arm-R1.py:48` sets `REFINEMENT_LEVELS = ()`, so the line never executes, and the
probe kept the 16 KiB allocation anyway. The fourth edit is mandatory, not defensive.

```
_topk_kernel                          src/…/deepseek_v4/sparsecore_topk.py:150
└── wave_body                         src/…/sparsecore_topk.py:~160
    ├── zero_priv(NUM_BUCKETS)     -> zero_priv(NUM_BUCKETS // LANES)      EDIT :185
    ├── scatter addr = lane*NB+b   -> addr = bucket                        EDIT :191
    ├── fold_priv(NUM_BUCKETS//LANES) -> fold_priv over range(1)           EDIT :197
    ├── zero_priv(where(done,0,NUM_BUCKETS))                               EDIT :224  ◄ REQUIRED
    │     -> the same // LANES bound. Omitting it overruns priv_vmem.
    └── refinement scatter addr                                            EDIT :238
    stays with XLA: _effective_row_lengths before; the out_hbm gather after
```

The as-built variant is **seven** hunks (incumbent lines 72, 128-135, 185, 191, 224, 238, 456); see
`results.md` and `variants/v02-r1-shared-hist/README.md` for the three further places where `v2`
deliberately differs from the `arm-R1` probe that priced it.

**3. The pipeline.** No pipeline change — this plan does not touch T1–T5. The DMA remains one
buffer, manually started and waited, overlap 0.0000.

**4. Files.**

| File | What it holds | What it must not hold | ~lines |
| --- | --- | --- | ---: |
| `sparsecore_topk.py` | the three edits above | ⛔ **no bare microbench constant.** State the mechanism in a comment; the number lives in `results.md` | ~6 changed |

**5. Expected gain vs the step-1 baseline.**

| | value | tag |
| --- | ---: | --- |
| baseline (production prefill) | 189.98 µs | measured + modelled reweighting |
| predicted | 177.79 µs | **measured** (`j-300d4690`), reweighted |
| **Δ time %** | **−6.42%** (6.4% faster) | |
| % of block floor | 11.3% | |

**Mechanism:** deletes 15/16 of a dense histogram clear, acting on the **fixed** (ρ-independent)
sweep term. Same denominator as §1. Because the term is ρ-independent, the prize is **larger** at
production ρ than the −3.73% measured at ρ=1.

**6. Evidence status.**

| measured (`j-300d4690`) | modelled (formula) | unmeasured |
| --- | --- | --- |
| 12.195 µs delta; bit-identical output 1024/1024 and 64/64; `zero_priv` = 9.238 of it (`j-4e0df30a`) | the ρ reweighting from 12.195 µs to −6.42% | nothing |

**7. Risks.**

| risk | signal, and which instrument shows it | fallback | threatens |
| --- | --- | --- | --- |
| `addupdate_scatter`'s conflict resolution is shape- or count-dependent and fails at a shape not yet run | row-comparison against the incumbent at **b=64 and b=1024, and at a negative-heavy input** — §8 says the existing gate cannot see `monotone_key` | revert; it is a 6-line change | **correctness** |
| the win does not survive composition with Plan B or D, which touch the same clear | per-arm absolutes in one job, never a sum of two percentages | measure the pair; §3b | speed |

---

### Plan B — fused emit (implements `v3`) · rank 2

**1. What it is.** After the level-0 histogram fixes the pivot bucket, the kernel currently walks the
whole 10240-element row again for each refinement level, and then walks it a third time to emit. All
three walks touch a small fraction of what they read. Plan B makes the level-0 walk do the work
directly: as it scatters, it also writes every element strictly above the pivot bucket straight into
the output buffer, and every element exactly equal to it into a compact straddler buffer. Refinement
then runs over the straddlers — a few thousand elements — instead of the whole row, and the separate
emit traversal disappears.

**2. The call chain.** Existing file; one new inner function.

```
_topk_kernel                          src/…/sparsecore_topk.py:150
└── wave_body
    ├── hist0 walk + two conditional store_compressed   src/…/sparsecore_topk.py::_l0_fused   NEW
    │     scratch = straddler_vmem: (cnt_max,) int32, VMEM       undecided — cnt_max sizing
    ├── refinement loop bound: walk_chunks -> cdiv(cnt,16)       EDIT
    └── emit_body traversal: DELETED
```

`cnt_max` is `undecided — settled by the level-0 bucket occupancy at the shipping score
distribution`; 4096 is the safe bound (a bucket cannot exceed the row's live length at ρ ≤ 0.4).

**3. The pipeline.** Unchanged from `v1` — this is an `algebra` move, not a `transfer` one.

**4. Files.**

| File | What it holds | What it must not hold | ~lines |
| --- | --- | --- | ---: |
| `sparsecore_topk.py` | `_l0_fused`, the straddler scratch, the two bound changes | the emit traversal it replaces — delete it, do not gate it | ~60 changed |

**5. Expected gain vs the step-1 baseline.**

| | value | tag |
| --- | ---: | --- |
| baseline | 189.98 µs | measured + reweighting |
| predicted | 132.4 µs standalone / 124.2 µs on top of Plan A | **modelled** |
| **Δ time %** | ~~−30.2%~~ → **−17.51%** standalone (harness denom) · **−21.3%** on the 155.82 µs production denom | re-priced, `probes/13` §5+§7 |
| % of block floor | 15.1% → **8.8%** | |

**Mechanism:** deletes two full traversals of `L`, acting on the **ρ-scaled walk** terms
(`emit` 17.7% + refinement `hist` 10.6%). Same denominator as §1.
**Band:** ⚠ ~~−20% to −33%~~ → **−17.51% central, −11.94% at the barriered upper bound.** The old band
was set by the level count and was wrong in both inputs: the level count is **1.570** (crit), not
2.231, *and* the pivot-bucket occupancy the design sized at the 4096 safe bound has a joint critical
statistic `cnt_eff` of **1798**. The two errors ran in opposite directions and the occupancy one was
larger, so the prize **improves** while the band **moves entirely below** its old range. **`v3`
cannot become a loss** — break-even `cnt_eff` = 6295 vs 1798 measured. See `probes/13` §5.

**6. Evidence status.**

| measured | modelled (formula) | unmeasured (the probe that would settle it) |
| --- | --- | --- |
| `emit` = 33.54 µs and refinement `hist` = 20.08 µs at production ρ (`j-4e0df30a`); level count **1.570** crit and `cnt_eff` **1798** (`probes/13`, CPU replay, no job) | the replacement cost: 1 extra conditional store per element at L0, plus `1.570 × 1798` walk elements | ⚠ **the cost of a conditional `store_compressed` inside the L0 walk** — it may not be free, and `probes/03` priced `emit1` at 27.456 µs, which is not negligible. **This is the probe Plan B owes.** ⏳ **dispatched, `probes/15-mandatory-v3-v10`** — measured as a *slope in store density* across ~5–65%, because a masked store's cost tracks density, not element count |

**7. Risks.**

| risk | signal, and which instrument shows it | fallback | threatens |
| --- | --- | --- | --- |
| the added store in L0 costs more than the walks it deletes | L0's own arm time rises in the same job — **include a de-contaminated single-region arm**, per §8 | fall back to fusing only `emit`, leaving refinement alone | speed |
| `cnt` exceeds `cnt_max` on an adversarial score distribution → out-of-bounds write | ⚠ **`store_compressed` at an unbounded offset is the exact failure `probes/08` nearly hit.** Assert the bound in-kernel and run the row-comparison gate at a negative-heavy input | clamp and fall back to the unfused path for that row | **correctness** |
| exactness claim rests on the vacuous gate (§8) | a `monotone_key`-sensitive input shows a mismatch the current gate cannot | — | **correctness** |

---

### Plan C — pipelined, ragged score DMA (implements `v4`) · rank 3 — **the caller's request**

**1. What it is.** The score transfer for each wave is started and then immediately waited on, so the
SparseCore stalls for the whole 40 KiB read; the Mosaic listing shows the compiler had four bundles
of slack and filled two of them with no-ops. Plan C gives the kernel two resident buffers and starts
wave `w+1`'s transfer before computing on wave `w`'s, so the read overlaps the compute. It also
issues only the live prefix of each row rather than the full padded slice, rounded up to a 32-byte
granule, which at the shipping prefill shape is 40% of the bytes rather than 100%.

**2. The call chain.**

```
_topk_kernel                          src/…/sparsecore_topk.py:150
└── wave_body
    ├── resident_vmem: (L,) -> (2, L) int32 VMEM                 EDIT
    ├── make_async_copy(scores.at[ds(s, slice_len)], …)
    │     -> pltpu.emit_pipeline OR a hand BufferedRef            src/…/sparsecore_topk.py::_score_pipe  NEW
    │        prologue: start wave 0 before the loop
    │        steady:   start w+1, wait w, compute w
    │        size:     round_up(i_hi, 8) words, not slice_len     ◄ granule alignment, §4d T5
    └── semaphores = 2, one per buffer
```

✅ **SETTLED by `probes/12`, locally, no job: use `emit_pipeline`.** It accepts a `plsc` mesh kernel
— it supplies its *own* grid and never touches the outer kernel's `program_id`, so the manual
`fori_loop` over waves is simply irrelevant to it — and four SparseCore kernels in this checkout
already use it, one on the production MoE path. **The hand-rolled `BufferedRef` fallback is
withdrawn.** Dynamic (traced, possibly-zero) grid bounds are allowed; **block shapes must be static**.

⚠ **Two consequences `probes/12` found that change this plan's shape**, both derived from source:

1. **`num_waves` is 32 at prefill and 1 at decode** (`:443`, with `P=1` and `P=2` respectively). The
   cross-wave pipeline drawn below is a **prefill** construct. It is worth 0 at decode.
2. **Prefer the within-slice form** — `grid=(n // chunk,)`, fusing `cvt` (`:171`) and `hist0`
   (`:187`) onto the pipeline buffer. It covers both phases, but hides the DMA behind **pass 1 only**
   (passes 2..N re-read `resident_vmem`, not HBM), so its ceiling is `min(DMA, cvt+hist0)`.
   ⚠ `chunk` must divide `n`: the kernel flattens scores to 1-D (`:486`), so a `BlockSpec` index map
   returns `row_c * n / chunk`. Every production `n` is a power of two, so any power-of-two `chunk`
   works. Alternatively pass scores 2-D `(b, n)` with a `(1, chunk)` block.
   ⚠ The `pl.when(i_hi > 0)` gate at `:166` must become a **dynamic grid of 0**, not a `pl.when`
   around the pipeline — wrapping it puts its `run_scoped` allocations under a conditional. Repo
   idiom: `ragged_gather_v2.py:102,208`.
   ⚠ Double buffering adds `2 * chunk * 4` B on top of `resident_vmem` (up to 256 KB at
   `MAX_SLICE_WORDS`) against a 512 KB budget — comfortable to `chunk <= 8K` words, but it is a real
   subtraction from the headroom `_pick_partition` sizes against.

**3. The pipeline.**

```
wave:        0        1        2        3      …    N-1      N
DMA in    [ t0 ]   [ t1 ]   [ t2 ]   [ t3 ]        [tN-1]     ·
compute      ·     [ c0 ]   [ c1 ]   [ c2 ]        [cN-2]  [cN-1]
DMA out      ·        ·     [ o0 ]   [ o1 ]        [oN-3]  [oN-2] [oN-1]
buffer     A        B        A        B               …
          └─prologue┘  └───── steady state ─────┘  └─ epilogue ─┘
             not hidden                               not hidden
```

- **Steady state:** the score read of wave `w+1` overlaps the compute of wave `w`, reached after 1 of
  32 waves.
- **Not hidden:** prologue `t0` and epilogue `cN-1` + `oN-1` — **2/32 of the transfer, ~0.84 µs**,
  `modelled`.
- **Buffer budget:** 2 × 40,960 B = 81,920 B of the **114,496 B (111.81 KiB, 21.84%)** total live set
  at this plan's tile — the whole §4d formula, re-evaluated, in-body intermediates included. It fits,
  and §4d already checked it fits **alongside** Plan B's straddler.

**4. Files.**

| File | What it holds | What it must not hold | ~lines |
| --- | --- | --- | ---: |
| `sparsecore_topk.py` | `_score_pipe`, the doubled buffer, the ragged size | ⛔ the padded `slice_len` path left in as a flag — pick one and measure the other as an arm | ~40 changed |

**5. Expected gain vs the step-1 baseline.**

| | value | tag |
| --- | ---: | --- |
| baseline | 189.98 µs | measured + reweighting |
| predicted | 176.5 µs standalone | **modelled**, off a measured ceiling |
| **Δ time %** | **−7.1% at the ceiling** (7% faster) | |
| % of block floor | 11.4% | |

**Mechanism:** overlaps and shortens the score transfer, acting on the **exposed-transfer** term.
Same denominator as §1. ⚠ **Band: −7.1% to 0%, and the low end is the likely one.** `cvt` already
absorbs part of the arrival latency at negative marginal cost, so the double buffer competes with an
existing mitigation. **D1 and D8 cap at the same 13.469 µs and do not add.**

**6. Evidence status.**

| measured (`j-300d4690`) | modelled (formula) | unmeasured |
| --- | --- | --- |
| exposed transfer = 13.469 µs; `cvt`'s marginal cost = −20.492 µs; overlap fraction 0.0000 (`probes/06` Q1, static) | prologue/epilogue = 2/32 of the transfer; ρ = 0.40005 → 60% of bytes unread | **how much of the 13.469 the double buffer can actually recover once `cvt` is no longer absorbing it** |

**7. Risks.**

| risk | signal, and which instrument shows it | fallback | threatens |
| --- | --- | --- | --- |
| ⚠ **the whole prize is already taken by `cvt`** and the measured delta is ~0 | a flat arm against a live control in the same job. **This is the expected outcome, not a failure** | report the null; it answers the caller's question definitively | speed |
| ~~`emit_pipeline` does not accept a `plsc` mesh kernel with a `fori_loop` grid~~ | ✅ **closed by `probes/12`** — it does, and four SC kernels here already use it | — (fallback withdrawn) | — |
| ⚠ **the Pallas→Mosaic lowering is clean but the Mosaic→LLO compile is not exercised** | a VMEM- or register-pressure failure that only appears on hardware. `probes/12` is explicit that it cannot see this | shrink `chunk`; `/xprof-llo` the arm to read VMEM high-water and spills | speed, and the build itself |
| the ragged size creates a sub-granule tail and the DMA slows down | bytes/µs falls below the 1255 GB/s slice rate | round `i_hi` up to 8 words (§4d T5) | speed |

---

### Plan D — `NUM_BUCKETS` sweep (implements `v5`) · rank 4

**1. What it is.** The radix is 256 buckets and has never been changed. A smaller radix makes the
per-level fixed clear cheaper — and stage C″ shows that clear is the single largest component of the
sweep — but leaves more elements in the pivot bucket, so more refinement levels run. A larger radix
does the reverse. The trade has never been measured in either direction.

**2. The call chain.** One compile-time constant, three arms: 64, 128, 512, against 256.

**3. The pipeline.** Unchanged.

**4. Files.** `sparsecore_topk.py`, one constant. ~1 line.

**5. Expected gain.** ⚠ **`unmodelled`** — that is the point of the plan, and per §10 it is why `v5`
is `OPEN` rather than `PARKED`. Cheapest entry in the list.

**6. Evidence status.** measured: `zero_priv` = 9.238 µs and is linear in `NUM_BUCKETS`
(`j-4e0df30a`). unmeasured: the level count at each radix.

**7. Risks.**

| risk | signal | fallback | threatens |
| --- | --- | --- | --- |
| not additive with Plan A, which shrinks the same clear 16× | print per-arm absolutes at both settings; never sum | measure the pair | speed |
| a smaller radix raises the level count enough to lose | the arm is slower | keep 256 | speed |

---

### Plan E — sample-and-count (implements `v10`) · rank 5, and the highest-variance entry

**1. What it is.** Every plan above keeps the radix-partition algorithm and makes one of its passes
cheaper. Plan E replaces the algorithm. Instead of histogramming all 10240 elements to find which
bucket the k-th value falls in, it histograms one element in sixteen to *estimate* the k-th value,
then counts how many elements exceed that estimate — a compare and a lane population-count, with no
memory scatter at all — and walks the estimate up or down until the count is exactly k. Then one pass
emits. The answer is the true top-k, not an approximation, because the correction loop is
unconditional on a miss.

**2. The call chain.** A new kernel body alongside the existing one, selected by a compile-time flag
so the incumbent stays measurable in the same job.

```
_topk_kernel_sample                   src/…/sparsecore_topk_sample.py::_topk_kernel_sample   NEW
└── wave_body
    ├── sample histogram over L/16                                                            NEW
    ├── count loop: compare + plsc.all_reduce_population_count, scalar carry                  NEW
    │     undecided — iteration bound; settled by the sample's estimate error
    └── emit pass (reuses the incumbent's emit_body)
```

**3. The pipeline.** Unchanged from `v1`.

**4. Files.**

| File | What it holds | What it must not hold | ~lines |
| --- | --- | --- | ---: |
| `sparsecore_topk_sample.py` | the sample pass, the count loop, the flag | ⛔ **no silent early exit from the correction loop** — that turns an exact kernel into an approximate one with no gate (§5, `v10`) | ~150 NEW |

**5. Expected gain vs the step-1 baseline.**

| | value | tag |
| --- | ---: | --- |
| baseline | 189.98 µs | measured + reweighting |
| predicted | 171.6 µs | **modelled**, on an `unmodelled` rate |
| **Δ time %** | **−9.7%** (10% faster) | |
| % of block floor | 11.7% | |

**Mechanism:** replaces a full scatter histogram plus refinement with a 1/16 histogram plus ~2.5
scatter-free count passes, acting on the **histogram** term (28.4%). Same denominator as §1.
**Band: −25% to +15%** — it can lose, and the band is set entirely by the count-pass rate.

**6. Evidence status.**

| measured | modelled (formula) | unmeasured (the probe that would settle it) |
| --- | --- | --- |
| `hist0` + refinement `hist` = 53.99 µs at production ρ; the 574 ps/bundle calibration; `cvt` = 1.963 ps/element | 3 bundles/vector for compare + popcount + scalar carry, vs `hist0`'s ~7.2, both per subcore | ⚠ **the count-pass rate — the single number this whole root rests on.** One probe, one arm: a bare compare + `all_reduce_population_count` pass over `L`, timed against `cvt`. §10 |

**7. Risks.**

| risk | signal, and which instrument shows it | fallback | threatens |
| --- | --- | --- | --- |
| a count pass is not cheaper than a histogram pass | the probe above, **before any of Plan E is written** | abandon the root; `v3` is the fallback and it is independent | speed |
| the correction loop needs many more iterations than 2.5 on the real score distribution | iteration histogram from the kernel, at the shipping distribution not `standard_normal` | widen the sample to 1/8 | speed |
| ⚠ the loop is quietly bounded for speed and the kernel silently becomes approximate | **the exactness gate: bit-identical rows against the incumbent, at a negative-heavy input** (§8 — the existing gate cannot see this) | mark `≈ approximate`, name the metric and threshold *before* the run, and gate it | **correctness** |

---

---

### Plan F — `keep_all` fast path (implements `v6`) · rank 6

**1. What it is.** A quarter of production prefill rows are short enough that the answer is "keep
everything" — their live length is below `k`. The kernel discovers this the expensive way: it
transfers the full 40 KiB score slice, converts it, histograms all of it and sweeps the buckets,
only to compute `keep_all = total <= k` and take the trivial branch. `i_hi <= k` decides the same
thing from a scalar already in hand, before any of that. Plan F tests it up front and, when it
holds, skips straight to writing the identity permutation.

**2. The call chain.**

```
_topk_kernel                          src/…/sparsecore_topk.py:150
└── wave_body
    ├── @pl.when(i_hi > k)  around ops 1-9 (DMA, cvt, hist0, zero/fold/scan/find)   EDIT
    └── @pl.when(i_hi <= k) -> _emit_identity: 0..eff_len-1 then -1 padding          NEW
```

⚠ **Not a pure skip.** The fast path still owes `rowbuf` a full `k` words — `0..eff_len-1` followed
by `−1` padding — so it needs a cheap replacement emit, not an early return.

**3. The pipeline.** Unchanged.

**4. Files.** `sparsecore_topk.py`: the guard and `_emit_identity`. ~30 lines.

**5. Expected gain vs the step-1 baseline.**

| | value | tag |
| --- | ---: | --- |
| baseline | 189.98 µs | measured + reweighting |
| predicted | 181.1 µs | **modelled** |
| **Δ time %** | **−4.7%** (5% faster) | |
| % of block floor | 11.1% | |

**Mechanism:** on the 25% of prefill rows with `token_pos < 2048`, deletes the score DMA (3.367,
fixed in ρ), the sweeps (3.908, fixed), and `hist0` (2.119, ρ-scaled), against `cvt`'s **negative**
−0.512. Same denominator as §1. ⚠ It does **not** delete the refinement passes — those are already
skipped by the in-kernel `keep_all` short-circuit at `sparsecore_topk.py:213-250`, which is why this
prize is ~4× smaller than the number this node carried before the source read.

**6. Evidence status.**

| measured | modelled (formula) | unmeasured |
| --- | --- | --- |
| every term above, from stage C″ (`j-4e0df30a`, `j-300d4690`) | the ρ-weighted split of each term across chunks 0–1 vs 2–7 | the replacement emit's cost, assumed ≈ the emit it replaces |

**7. Risks.**

| risk | signal, and which instrument shows it | fallback | threatens |
| --- | --- | --- | --- |
| the extra branch costs more on the 75% of rows that do not take it than it saves on the 25% that do | a slower arm at a b=1024 shape whose rows are *all* long | drop it | speed |
| the identity emit disagrees with the incumbent on tie-dense rows | ⚠ the row-comparison gate **at the real distribution** — production rows carry hundreds of exact ties and some have a maximum of exactly 0.0 (`test_dsv4_sparsecore_topk.py:64-70`) | — | **correctness** |
| a mixed prefill/decode step dilutes the saving | ✅ **checked and it does not** — there is no inter-subcore barrier, and stride-32 ownership makes `max ≈ mean` (§4c) | — | speed |

### Build order

⛔ **Rewritten at the close of step 2.** Three of the four bullets this block used to carry are
discharged or dead: the count-pass probe returned (`probes/15`), `probes/11` returned and then
`probes/14` finished it, and Plan F is rejected so its overlap warning is moot.

- **Build first: Plan A (`v2`).** It is already written (`probes/08-…/arms/arm-R1.py`), already
  bit-exact and already measured — a lift into `variants/`, not a build. Its job is to validate the
  step-3 harness against a known answer before an unmeasured arm runs.
- **Buy first, in the same job: one ρ = 0.40005 arm of the incumbent.** ⛔ This outranks every
  variant. The whole sheet's denominator is `modelled` — nothing has ever run at the production ρ —
  so until this arm exists every percentage in this file is a ratio of two predictions. One arm
  converts it to `measured` and validates the reweighting whole. See "Numbers in this document".
- **Then the fork, in parallel: Plans B (`v3`) and E (`v10`).** Ranks 1 and 4, alternative roots,
  and the ranking cannot separate them. Both get built; the measurement adjudicates.
- **Cheap and parallel with all of the above: `v12` and `v11`.** Three edits and four, no algorithm
  change, `exact`, and worth ≈ −23% production between them. They have no plan section because they
  were promoted from stage E after §9 was written — build them from their §5 blocks.
- **Blocked:** Plan C (`v4`) — on Plan B, for timing only, since the fused pass changes what the DMA
  feeds; and on its own local compile check of `emit_pipeline`. ✅ That compile check is now cheap
  and needs no pod: AOT against `topologies.get_topology_desc("tpu7x:2x2x1")`, per `probes/17`.
- ⛔ **Do not build Plan D (`v5`) either.** `probes/16` returned and **refuted** it. Both terms it
  was open on resolved in its favour — the intercept is 0.299 µs (1.6% of the sweep) and the
  07-vs-09 conflict went to 09 — but the L0 histogram grows with smaller `NB` faster than the sweep
  shrinks, netting **−5.28% ± 11.19%**: unresolved, not a win.
- ⛔ **Do not build Plan F (`v6`).** Rejected by `probes/14`, not by a microbenchmark: the outer
  `lax.cond` already elides the entire kernel on exactly the steps an in-kernel `keep_all` fast path
  would fire on. Its residual — short rows batched with a long one — is `unmodelled` and out of
  scope for this exercise.

## 10. Open, parked, and carried

**`OPEN` versions — none may remain when the step closes.**

⛔ **The closing audit against that rule, stated plainly rather than asserted.** Five rows below are
still `OPEN` at the close, and the rule is satisfied only because each has a *named* closer that is
already scheduled. Anything not in this list would be a genuine violation:

| still `OPEN` | what closes it | when |
| --- | --- | --- |
| `v11`, `v12` | their **step-3 measurement** — each has one `unmodelled` term (a strided DMA; two `tpu.bitcast`s) and both are on the build list. Neither term can flip a sign, only shrink a prize | step 3 |
| ~~`v5`~~ | ✅ **CLOSED by `probes/16` — `REFUTED`, not `OPEN`.** It won both terms it was open on and lost on a third the model omitted | ✅ done |
| `v10`'s pass count | **host simulation**, no job — the rate it rested on is already `measured` | step 3, with the variant |
| ~~the `hist0` density question~~ | ✅ **CLOSED — false premise, retracted.** §7 | ✅ done, on arithmetic |
| **D7** | ⛔ **the one that does not have a cheap closer any more.** See its row | ⚠ carried forward |

| node | what closes it | cost |
| --- | --- | --- |
| `v3` fused-emit | ✅ **CLOSED by `probes/15` (`j-78f7f537`).** The store is **6.675 µs/pass**, flat in density, and `probes/13` had overcharged it **2.08×**. Band **≈ −20.5%…−14.9%**; break-even `cnt_eff` **8531** is above the global max 6495, so **its sign is settled without building it** | step-3 variant |
| ⛔ ~~is `hist0`'s sheet price understated?~~ | ⛔ **CLOSED — the question was based on a false premise, now retracted.** `probes/16` refuted `probes/15`'s instrument-bug claim and the main thread re-derived it: the mask runs over **monotone keys**, `KEY_NEG_INF` is the monotone key of −inf, and the measured density is **1.000000 under both readings**. `hist0`'s 33.91 µs also comes from `probes/08`'s residual, not from that arm. **`v10` gains nothing.** Full retraction in §7 | ✅ **no job — closed on arithmetic** |
| **`v11`** retiled output emit | ✅ **both conditions CLOSED, both favourably.** (a) feasibility, `probes/17` — the 2-D tiled `out_type` lowers, the `ROOT reshape` disappears, ENTRY 13 → 12, four edits, `o.at[pl.ds(row,1)]` is the one indexer form that fails; (b) in-model, `probes/14` Q2 — the retile **survives** in the real entry, as an `S(1)` staging copy rather than a ROOT reshape. Same **8 MiB** (⛔ not 40 MiB — corrected in step 3; 40 MiB is `v12`'s input), different op, so the "`REJECTED` with prize 0" leg is withdrawn. ⚠ The 10.007 µs was measured on the ROOT reshape; no arm has yet deleted the staging copy. ✅ **Step 3 closed that caveat**: `v11`'s in-model AOT compile deletes **both** the `S(1)` reshape and the `copy-start.5`/`copy-done.5` pair, ENTRY 220 → 218, because the `cond`'s 8 MiB result buffer leaves the VMEM pool and becomes the HBM live-out | no job |
| **`v12`** (= **U2**) `f32` input, in-register reinterpret | ✅ **feasibility CLOSED by `probes/17`** — reachable, but **three edits, not one**: the DMA is dtype-checked, so the scratch dtype must change too and the reinterpret moves in-register. ✅ **production path CLOSED by `probes/14` Q1** — materialised, so the prize is a firm ≈ −16.8%, the largest single edge prize in the file. One term still open: the price of the two in-body `tpu.bitcast`s, which can only shrink the prize, not flip it | no job |
| **D7** SC↔TC overlap | ⛔ **moved `PARKED` → `OPEN`** — see below. ⛔ **And its "free, no job" closer is gone.** The plan was to read `StreamIdxTC-*` durations out of an existing xprof; `probes/14` tried exactly that on `j-ecf4ee85` and got **0 device events**, because that step took branch 1 and the whole chain was elided. `t_TC` is not obtainable from any capture in which the kernel did not run. The cost has changed class — from "read a file" to **step selection**: capture a prefill step past chunk 1, or a mixed step. It stays `OPEN` and is **carried into step 3** rather than closed here | ⚠ **a capture, not a file read** |
| `v4` pipelined-DMA | ✅ **CLOSED by `probes/12`, no job.** `emit_pipeline` is viable on SparseCore and is the repo idiom; the `BufferedRef` fallback is withdrawn. The node stays a step-3 variant, and it gained a **phase split**: cross-wave is prefill-only (32 waves vs 1), within-slice covers both but hides only pass 1 | step-3 variant |
| `v5` NB-tuned | ⚠ **mandatory probe** — three compile-time arms. Its verdict currently turns on an `unmodelled` term, so §10's rule forbids parking it | one job, three arms |
| `v6` keep-all-fastpath | ⛔ **REJECTED for this workload by `probes/14`** — the outer `lax.cond` (`max(seq_lens) <= 2048`) elides the kernel in exactly the chunks D14 said would sweep to discover "keep everything", and the set coincidence is *exact* because `k` is a whole multiple of the chunk size. **Prize 0 `measured`** (all 96 `batch_composition_stats_*.json`: `num_reqs: 1`), **`unmodelled` in steady-state serving** — a mixed batch would reopen it | no job |
| `v10` sample-and-count | ✅ **the blocking rate is CLOSED by `probes/15`.** A count pass is **20.681 µs vs the histogram's 108.217 → 0.191×**, 1.30× better than modelled, and **at the read-only walk floor** — so the bimodality collapses onto the cheap branch. Central **−12.2%**, break-even **8.0 passes** vs a modelled 2.5. What is still open is the **pass count**, which is **host-simulable** | no job |

**`PARKED` versions.**

- **`v7` shallow-scans** — parked on a **measured** term (the two carries are 4.698 µs, 2.5%), not an
  unpriced one, so the §10 rule is satisfied. **Reopens if `v5` selects a larger `NUM_BUCKETS`.**
  Route is inside the kernel → a **Not adopted** row in `final_report.md`.
- ⛔ **D7, the SparseCore/TensorCore pipeline — MOVED `PARKED` → `OPEN` by stage E.** The paragraph
  below already said it "needs a number before anyone can judge it" and then parked it anyway; that
  is precisely what §10's rule forbids, and stage E caught it. It is now an **`OPEN` item with a
  mandatory, free probe**: read the durations of `StreamIdxTC-m-bq_128-bkvp_2` and
  `StreamIdxTC-d-bq_1-bkvp_2` (scope names, `streamindex_topk.py:703`) from an **existing** GLM-5.2
  prefill xprof. No job. It remains a `carried:` handoff **as well** — the fix is outside the kernel
  — but it may not close the step unpriced.
  ⚠ Stage E added a structural constraint the prize must respect: the three `StreamIdxTC` calls are
  **serially chained by aliasing** (operand 10), so the SC kernel starts only after the third.
  Route is *outside* the kernel: `P=1` leaves no partition loop, so the overlap must be constructed
  in the attention op. Prize `min(t_TC, t_SC)·(p−1)/p`, `t_TC` **`unmodelled`**.
- **D11, the `b_pad`-wide length DMA** — ✅ **closed by `probes/10`, not promoted.** It was the
  per-row candidate and the per-row axis measured **−0.886 ns/row** against a 0.770 µs floor. It stays
  `assumed` negligible, but now on evidence rather than on assumption.
- ⛔ **D3's input relayout — WITHDRAWN, it is not a handoff and not a production cost.** `probes/11`
  found **zero** `sparse-core-data-format-call` and no 40 MiB copy in the production module: the
  score buffer's minor dimension is one lane-tile, so the flatten is a free `bitcast`. The ~73 µs
  was an artifact of the microbenchmark's `jit` entry aval. Nothing to hand off. ✅ Its real payoff
  was to the denominator (§1), and it went in our favour.
- ✅ **The output retiling, 10.007 µs — RESOLVED, and it is a *variant*, not a handoff.** ⛔ The
  bullet below is kept for its evidence but its conclusion is superseded twice over. (1) Its blocking
  condition — "does it exist at all in-model?" — is **answered yes** by `probes/14` Q2: in the real
  entry the retile survives as an `S(1)` staging copy rather than a `ROOT reshape`, a different op
  moving the same **8 MiB** (⛔ **"40 MiB" until step 3 corrected it** — that is `v12`'s input
  buffer). It is **not** an artifact of the same class as the input relayout was.
  (2) `probes/17` then showed the kernel can emit the consumer's tiling directly, which makes this
  reachable *from inside* the kernel. So it is node **`v11`**, rank 7 in §9a, on the step-3 build
  list — and the "candidate handoff" framing is withdrawn. ⚠ One caveat survives: the 10.007 µs was
  measured against the ROOT reshape, and no arm has yet deleted the staging copy.

- ⚠ **The original bullet, kept for its evidence:**
  `ROOT %reshape.1 = s32[1024,2048]{1,0:T(8,128)} reshape(…call-done)` retiles the kernel's flat
  `s32[2097152]{0:T(1024)}` output, and it **is** in the production entry (`measured` 10.007 µs =
  1.251 µs/MiB, `A0_flatout`). Of the five things a handoff owes it has four: *what to change* — have
  the consumer accept the flat `{0:T(1024)}` output, or give the kernel the retiled output layout
  directly; *the prize* — 10.007 µs `measured`, 6.4–7.7% of the production kernel; *the correctness
  evidence* — none needed, a layout change computes the same function; *the constraint* — outside the
  Pallas kernel, so no §9 plan reaches it. ⛔ **The missing one is whether it exists at all in-model**:
  compiling `streamindex_topk` standalone forces its ROOT to the default entry layout, so this may be
  an artifact of the same class as the input relayout was. **Do not promote it to a handoff until
  that is checked** — the check is free, see below.

- ✅ **NEW `carried:` handoff — `_fast_path`'s `searchsorted`, and it is the largest number this
  exercise has turned up outside the kernel.** All five things a handoff owes:
  - **What to change, and where.** `streamindex_topk.py:841` computes
    `jnp.searchsorted(cu_q_lens[1:], token_idx, side="right")` over `arange(1024)`. JAX lowers that
    as a `ceil(log2(len+1))`-level binary search on a uint32 index; `cu_q_lens` is `s32[65]`, so it
    is **7 dependent levels of `u32[1024]` fusions**. The function being computed is "which sequence
    does this token belong to" against a *monotone cumulative-length* array — a prefix-sum/compare or
    an iota-and-mask has far cheaper lowerings.
  - **The prize, and the level it was measured at.** The whole `_fast_path` region is
    **13.851 ms/prefill step = 2.12% of a 655 ms step**, 659.59 µs × 21 conditionals,
    `measured (j-ecf4ee85, rank 14)`, and the binary search is most of it. Semantically the branch
    returns an **identity permutation plus a mask**; `(1024, 2048) int32 × 21` is 168 MiB of writes,
    ~105 µs at HBM rate, so the region runs at roughly **130×** what its traffic costs.
  - **The correctness evidence.** None needed for the search itself — same function, cheaper
    lowering. Any rewrite owes the existing `_fast_path` output as its oracle.
  - **The constraint that stops it being done here.** It is in the *other* branch of the gate, in
    `streamindex_topk.py`, not in `sparsecore_topk.py`. It is not a Pallas kernel and not a variant.
  - **The route.** `/preflight-check`, then the e2e benchmark.
  - ⚠ **Weight it before quoting an e2e figure.** It is paid only on the **gated chunks — 2 of 8**,
    and only while `max(seq_lens) <= 2048`. It is *not* additive with any kernel prize: the two
    branches are mutually exclusive by construction.

**`assumed` numbers the graph depends on.**

| the assumption | where it bites | what would settle it |
| --- | --- | --- |
| refinement level count = ~~2.231~~ → **1.570** (crit) | `v3`'s band, `v5`'s whole trade | ✅ **CLOSED by `probes/13`, and it moved the opposite way to the prediction.** At production prefill: **1.300 ± 0.009 over all rows**, 1.740 ± 0.009 over walking rows, **critical-subcore 1.570 ± 0.035** (`measured`, CPU replay of the kernel's integer logic, 4 KV draws). ⚠ **Use 1.570.** The aggregation is *derived, not chosen*: `wave_body` gives subcore `flat` the rows `w*32 + flat` (`sparsecore_topk.py:157`) inside a `fori_loop` run independently per subcore (`:443`) with **no barrier anywhere** (`:19`, `:528`), so cost = **max over the 32 subcores of the sum of its own rows** — not sum-over-waves-of-max, which is the *barriered* upper bound and is carried only as such. 1.740 is a conditional mean over walking rows and is not a cost aggregation at all; 2.231 is a timing inversion and so is already critical-subcore, which is what makes 1.570 the matched replacement. ⚠ **Two things this row previously asserted were wrong.** (a) *Provenance*: 2.231 was never a distribution fit — it is `probes/07`'s inversion `(L3−L0)/level_slope` at **ρ=1** (`probes/07/README.md:77`), companion `probes/05`'s 2.03. (b) *Direction*: the ReLU premise below is refuted at the source — see the next row |
| L0 pivot-bucket occupancy `cnt` = ~~4096 (safe bound)~~ → **1798** (`cnt_eff`) | `v3`'s prize — **this, not the level count, was the dominant error** | ✅ **CLOSED by `probes/13`.** Prefill `measured`: mean 1071, **crit 1423 ± 65**, barriered `wavemax` 4192, global max 6495. ⚠ **But `cnt` alone is the wrong input**: `v3`'s work is `levels_row × cnt_row` **jointly**, so the statistic is `crit(levels·cnt)`, giving **`cnt_eff` = 1798** (1.68× the mean — levels and `cnt` are positively correlated, so the joint statistic exceeds `crit(cnt)` = 1423). Decode: `cnt_eff` 2823 at b=1024; crit 5539 at b=64, where only 2 waves exist and there is nothing to average over |
| ~~production scores are non-negative, zero-massed, tie-dense~~ | was the stated reason to expect a *higher* count | ⛔ **REFUTED on every premise by `probes/13` §3.** The indexer weights are **signed**: `weights_proj`/`wk_weights_proj` is a plain `nn.Linear` with no activation (`mla_attention_op.py:246`, `modular_deepseek_v4.py:382-395`), so the ReLU sits **inside** the per-head sum, not outside it. One production prefill row (`eff_len=8192`): **8192/8192 distinct values, 0 exact zeros**, 81.7% negative, skew −0.048, boundary not tied, **14** occupied L0 buckets vs 16 for `standard_normal`. `unconverged = 0` in every production config at every radix, so the arbitrary tie-break at `sparsecore_topk.py:289` is never exercised. Sensitivity grid (α up to 0.99, `\|w\|`, per-head scale spread, n_head 16–128, ×1e±6 rescale) tops out at 2.031 — **below** 2.231 even when built adversarially |
| ~~the 2.231-vs-2.03 gap is a score-distribution mismatch~~ | how much of the level cost is real | ⛔ **REFUTED — and this was asserted in three places.** At **matched ρ=1** the production distribution gives 1.889 mean / **2.031** crit against `standard_normal`'s 1.92 / **2.03**: the distribution is worth −1.6% on the mean and **0.0% on crit**. `probes/08` had already refuted the skipped-level explanation (0.060 µs, not 3.8). So ~0.20 levels of the timing inversion is **still unexplained residual** — not distribution, not skip overhead. It is small and it inflates a cost, so it is conservative to carry; but stop calling it explained |
| ρ = **0.40005** prefill | **every share in this document** | ✅ **VERIFIED independently to 5 significant figures**, from `streamindex_topk.py:503-506` with `compression_ratio = 1` (`mla_attention_op.py:116`), `n = 10240` (`streamindex_topk.py:603-605`), and `max-model-len 9216` / `ISL 8192` / `RANDOM_RANGE_RATIO 0.0` in `glm-5.2-fp8-dp16-ep.sh:43,27,37`. It is **robust to the chunk mix** — ρ is uniform over `token_pos ∈ [1, 8192]` however the budget is split. ⚠ What is *not* robust is "2 of 8 chunks"; state it as **1/4 of prefill rows have `token_pos < 2048`**, which holds regardless |
| the row-comparison gate proves exactness | `v3` and `v10` | ⚠ **§8's finding stands — it is vacuous for `monotone_key` — but the justification below was wrong twice and is corrected here.** (a) *The citation was misread.* `test_dsv4_sparsecore_topk.py:64-70` is the `N_SHORT`/`SEED` region and says nothing about ties; the tie commentary is at `:148-151`, and it documents the **synthetic `heavy_ties` generator** ("17 distinct values across the whole row"), not production data. Reading a test fixture's docstring as a fact about the shipping distribution is the error. (b) *The fact it was used to assert is false.* `probes/13` §3 measures a production prefill row at **8192/8192 distinct, 0 exact zeros, boundary not tied**, and finds `unconverged = 0` in every production config at every radix. So a reordered emit has **no tie to disagree on** in production — the exactness risk for `v3`/`v10` is lower than this row claimed, not higher. ⚠ **But keep the tie-dense arm anyway**: the census is **n = 1 row**, tie-freeness is a distributional observation and not a guarantee, and `heavy_ties` already exists in the test (`:148`). The live gap is narrower than stated — the gate still needs a distinct `out_hbm` sentinel, per §8 |

**`unmodelled` terms still in the floor.**

| term | which version it hits hardest | asymmetry check (§10's tell) |
| --- | --- | --- |
| ✅ **D3 — CLOSED, and now with no residual.** 34.165 µs harness-only, 26.251 µs **production pays** (`probes/14` Q1), 10.007 µs live output retiling **confirmed in-model** (Q2), 5.385 µs irreducible | the **denominator**, and therefore all of them equally | ✅ **the symmetry prediction held.** `probes/11` moved every prize by a common factor and reordered nothing. The one leg it left open — is production's same-shape `bitcast-convert` materialised? — is now answered **yes** on buffer assignment, which collapses the factor from a 1.219–1.466 range to a firm **1.219** and converts that 26.251 µs from denominator uncertainty into `v12`'s prize |
| **count-pass rate** | `v10` only | ⚠ **asymmetric, and this is the one to watch.** Treating it as cheap makes `v10` a −9.7% candidate; treating it as expensive kills the root. Both readings come from the same empty cell — **hence the mandatory probe above** |
| `t_TC` (D7) | the handoff, no version | symmetric — nothing is being taken on it either way |
| step homogeneity in ρ | `v6` only | ⚠ **asymmetric**: it is what turns `v6` from ≤−18.5% into 0%. Hence the mandatory check |

**What a next session should do first, in order.**

1. ✅ **`probes/11` is collected and D3 is closed** — the harness-artifact leg. Two **free, no-job**
   follow-ons it left, both worth doing before step 3 spends anything:
   - **Is production's `%bitcast_convert_type.3` materialised (~26 µs) or aliased away?** It closes
     the denominator range. Cheapest check: an **existing GLM-5.2 prefill xprof** — look for a ~26 µs
     op between `StreamIdxTC` and `sc_topk_direct`. Failing that, XLA's buffer-assignment dump.
   - **Does the 10.007 µs output retiling survive in-model?** Compile the enclosing
     `mla_attention_op` entry (or read the full-model optimized HLO) and grep for the `reshape` on the
     `call-done`. This is what decides whether the one surviving handoff is real.
2. ✅ **The scheduler source read is done** — it settled `v6`, verified ρ to 5 significant figures,
   and identified the production score distribution. ✅ **`probes/12` has landed**: `emit_pipeline`
   is viable on SparseCore, the `BufferedRef` fallback is withdrawn, and `v4` split by phase.
   ⏳ **`probes/13`** (re-fitting the refinement level count at the real ReLU'd, tie-dense,
   zero-piled distribution rather than `standard_normal`) is still out. The level count feeds `v3`'s
   band and `v5`'s whole trade, so neither is built before `13` lands.
3. **Dispatch the two mandatory probes in one job**: the count-pass rate (`v10`) and the conditional
   `store_compressed` cost (`v3`). Both are single arms and they decide two of the three largest
   candidates. ⚠ **Give both the real score distribution**, not `standard_normal` — a tie-dense input
   changes the mask densities they are measuring, and that is why this waits on `probes/13`'s
   validated generator rather than each probe rolling its own.
4. ⚠ **Before any step-3 arm: give the thunks a production-shaped entry.** `probes/11`'s
   `A0_flatin` shows how — the harness's `f32[1024,10240]` parameter adds 34–60 µs that production
   does not pay, and a constant in the denominator dilutes every delta. `instruments/` stays
   read-only and frozen (it is what makes probes 08–11 comparable); the arms build their own thunks.
5. **Lift Plan A into `variants/`** — it is written, bit-exact and measured, and it validates the
   step-3 harness against a known answer before anything unknown is measured against it.
6. Then `v5`'s three-arm sweep, and Plan B.

---

## 11. Stage E — the boundary

Full working: **`stage-e-boundary.md`** (option table with all 15 rows, the six checks per row, and
every rejection priced). This section is the summary the graph needs; the file is the evidence.

**The decision.** The kernel stays exactly one `plsc` mesh call. What §9 got wrong was treating that
as the whole boundary question: the middle does not split, but **three of the four largest non-kernel
numbers in this exercise sit on the edges**, and `v4` creates a new internal one.

| edge | verdict | Δ time % | decided by |
| --- | --- | ---: | --- |
| **U1** input layout — flat view of `f32[1024,80,128]{2,1,0}` | **KEPT**, and now *evidenced* rather than assumed | 0 | re-derived: 0 `sparse-core-data-format-call`, the only `copy` in the module is `%copy.11 = f32[1024,64]` |
| **U2** delete `%bitcast_convert_type.3` — declare the input `f32`, reinterpret in-register | **OPEN** as new node **`v12`**; feasibility ✅ closed, production path ✅ closed | **−13.82%** harness ⇒ ✅ **≈ −16.8% production**, firm | ✅ `probes/17`, `static-compile`: reachable, but **three edits** — the DMA is dtype-checked, so the scratch dtype moves to `f32` too. ✅ `probes/14` Q1, `static-compile`: **materialised in production** — a separate, disjoint, exactly-adjacent 40 MiB temp buffer (`1753088 + 41943040`), and 0 counterexamples in a 281-instruction falsification sweep |
| **U3** any other input layout | **REJECTED**, priced | **+17.98%** | 34.165 µs `measured (j-9b520877)` — this is the counterfactual that makes "already optimal" a finding |
| **P1** fuse the producer `StreamIdxTC` | ⛔ **REJECTED — illegal, not unprofitable** | unreachable | it is a **TensorCore** kernel; `dot_general` has no `SC_VECTOR_SUBCORE` lowering rule |
| **P2 = D7** overlap TC and SC instead | ⛔ **PARKED → OPEN** | `unmodelled` (`t_TC`) | §10's rule: a verdict on an `unmodelled` term cannot be parked. Free probe — read `StreamIdxTC-*` durations from an existing xprof |
| **D1′/D2′** output retiling, `ROOT %reshape.1` | **OPEN** as new node **`v11`**; feasibility ✅ closed | ≤ **−5.27%** | 10.007 µs `measured`, `used_scoped` 4 MiB. ✅ `probes/17`: the ROOT reshape disappears, ENTRY 13 → 12. ⚠ still conditional on the in-model check |
| **D4′** fuse the consumer `sc_dsa_gather` | **carried (H3)** | **−8.36% to −11.42%** | larger than any single §9 plan; rejected here only because it changes the block's **output value** — a step-1 re-scope, not a version |
| **I4** `v4`'s within-slice `emit_pipeline`, `grid=(n // chunk,)` | **KEPT** — the new internal edge | **−2.06% to −6.38%** | encloses exactly `cvt ⊕ hist0`; chunk **1024** |

**One correction made *to* stage E, by `probes/17`.** `stage-e-boundary.md:55` states `v11`'s out
DMA becomes "16 runs of 512 B at stride 1024 words". At the Mosaic-op level that is **not
observable** — the enqueue is the same `2048xi32 vmem → 2048xi32 hbm` in both arms, with the row
folded into a `memref_slice`. The 16×512 B pattern is implied only by the memref *tiling*, so it is
`modelled`, and the strided-DMA cost stays the one `unmodelled` term in `v11`. Full note in the
`v11` block above.

**Two corrections stage E made to earlier documents.**

- ⛔ **`probes/12`'s "all production `n` are powers of two" is false.** `n = 10240 = 2¹¹·5`
  (`streamindex_topk.py:603-605`), so **chunk 4096 is illegal**. Legal chunks are the divisors of
  10240 that are multiples of `LANES=16`: 16, 32, 64, 80, 128, 160, 256, 320, 512, 640, 1024, …
- ⛔ **`hardware_map.md`'s S2 seam (~10 µs, `modelled`) is contradicted by measurement.** `cvt`'s
  marginal cost is **−8.20 µs measured** — the pass is already free because it absorbs the DMA's
  arrival latency. **S2 only becomes positive inside I4**, once the DMA is pipelined and `cvt` stops
  absorbing. **S2 and I4 are one move; do not sum them.**

⚠ **The `v4` ceiling was never reachable.** §5 carried "≤ 13.469 µs"; the honest range is
**−2.06% to −6.38%**, because step 0's DMA is never hidden (head = `13.469/(n/chunk)`) and cvt's
absorption re-emerges. The weakest number here is the pessimistic bound: it subtracts cvt's full
measured −8.20 µs on the theory the absorption is banked twice, and that is a marginal-removal
measurement, not a decomposition. The chunk sweep in step 3 measures it directly.

**Independence check.** No price in stage E uses the refinement level count, so `probes/13` changed
nothing here. `v3`'s and `v5`'s **boundary fields are both unchanged** — their moves are internal.
