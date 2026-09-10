# Probe 11 — is D3's ~75 µs relayout in the production path, or an artifact of the harness's `jit`?

**Job** `j-9b520877-2491-4fd4-b618` · Succeeded, 83 s · image `…vllm-torchtpu-hning_google_com:fa63d6df`
(probe 10's image) · one job, no fan-out · tags `agent-jobs,run-command-in-cdk` · harness
`glm_dsa_block_bench.py` md5 `08375d7068365e1a776c58fef09cd6d0` (frozen, in-job `MAP_GATE_OK`) ·
subject md5 `44353b347c96e0eedbd742c710d4540a` · recipe read-back md5
`8b03b15624359c206249fdefcabcc668`, byte-identical to what was sent, `bash -n` clean raw and folded.

Verbatim log `j-9b520877-2491-4fd4-b618.txt` (506 lines) · arms `arms/` (8) · optimized HLO `hlo/`
(8 arms + 2 production modules) · local validation `validation.local.txt` · results
`relayout_provenance.json`.

An extra gate the spec did not ask for: **`P09_SRC_GATE`** re-derives probe 09's `fd46ef14…` (`B`) and
`547031a2…` (`A0`) patched sources bit for bit, so "same arms as probes 09/10" is a checked fact.

---

## The answer

**The largest single term this exercise had found does not exist in production.** `probes/10` traced
D3's 75.3 µs residual to an XLA relayout of the 40 MiB score buffer *outside* the Pallas kernel, and
flagged two legs: real at the call site (a ~73 µs `carried:` handoff) or an artifact of the
microbenchmark's `jit` entry signature (every denominator in `design.md` overstated). **It is the
artifact leg**, though not by the full amount the prober first reported — see "The correction" below.

| | harness (`glm_dsa_block_bench.py`) | production (`streamindex_topk`) |
| --- | --- | --- |
| score buffer entering the topk call | `f32[1024,10240]{1,0:T(8,128)}` — an **entry parameter** | `f32[1024,80,128]{2,1,0:T(8,128)}` — the `StreamIdxTC` custom call's own output |
| `sparse-core-data-format-call` | **present**, 5 lines | ⛔ **absent**, 0 occurrences |
| 40 MiB untiling `copy` | **present** — `%copy.3`, `{3,2,1,0:T(8,128)}` → `{3,1,2,0:T(8,128)}` | ⛔ **absent** — the only `copy` in the whole module is `%copy.11 = f32[1024,64]`, 256 KiB, `indexer_weights` |
| flatten to `s32[10485760]{0:T(1024)}` | via that call | ✅ a **free `bitcast`** (`%bitcast.74`) |

**Why.** Production never materialises a 2-D `f32[1024,10240]`. Its minor dimension is exactly
**128 = one lane-tile**, so `{2,1,0:T(8,128)}` *is* row-major and the flatten is a pure bitcast. The
harness's minor dimension is 10240 ≫ 128, so `(8,128)` tiling is **not** row-major and XLA must untile
the whole buffer before the SparseCore call can read it.

I re-derived this myself from `hlo/hlo-prod_early_exit_False.txt` rather than taking the report's
word for it: `grep -c sparse-core-data-format-call` → **0**; the operand chain walked **by name**
(not by grepping a line for an operand's shape — see `[[hlo-line-prints-output-shape-operands-by-name]]`)
is `%StreamIdxTC-m-bq_128-bkvp_2.1` → `%bitcast_convert_type.3` → `%bitcast.74` → the `call-start`.

## The correction — the prober's −60.416 µs splits into a firm leg and an open one

⚠ **The report's headline ("production pays *neither* input relayout, ~70.5 of 75.3 µs does not
exist") over-claims by 26.251 µs.** The arms are right; the transfer to production is not. The
harness's input path is **two** ops, and only one of them is absent from production:

| harness op | measured | in production? |
| --- | ---: | --- |
| `%copy.3`, the untiling copy under `sparse-core-data-format-call` | **34.165 µs** (by subtraction, `flatin − s32in`) | ⛔ **absent** — firm, verified by my own grep |
| `%bitcast-convert_bitcast_fusion`, 40 MiB `kind=kLoop` | **26.251 µs** (`A0_s32in`) | ⚠ **an op of the same opcode exists, but it is not the same op** |

The harness's fusion converts **and reshapes and retiles**:
`f32[1024,10240]{1,0:T(8,128)}` → `s32[128,80,8,128]{3,2,1,0:T(8,128)}`. That is real data movement,
and 26.251 µs is its price. Production's is
`s32[1024,80,128]{2,1,0:T(8,128)} bitcast-convert(f32[1024,80,128]{2,1,0:T(8,128)})` — **identical
shape, identical layout, 4 bytes → 4 bytes**, not a fusion. Semantically a no-op that buffer
assignment should alias away.

⛔ **But it is not safe to assume it does.** Both instructions carry the *same*
`estimated_cycles: 66712` in their `backend_config`, and production's has a full `window_config`
(`iteration_bounds ["2","5","1"]`, `output_window_bounds ["512","2","1"]`, `buffering_level 2`) —
i.e. XLA's emitter planned a loop for it. That is evidence **for** materialisation, not against.
**`A0_s32in`'s 26.251 µs prices the harness's op and cannot be transferred to production's.**

**So the denominator correction is a range, and its firm leg is −34.165 µs, not −60.416 µs.**

## Arms — one interleaved pool, 9 rounds, `us_median`

Nulls 17.059 / 15.857 / 17.499 → mean **16.805**, **spread 1.641 µs**. ⚠ That is **2.1× worse than
probe 10's 0.770**; every delta below is quoted against 1.641, not against probe 10's floor.

| arm | `us_median` | residual | Δ vs `A0_base` | df-call | layout copy | materialised MiB |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `B_base` | 326.770 | 309.965 | +234.068 | yes | yes | 80 |
| `A0_base` | 92.701 | 75.896 | 0 | yes | yes | 80 |
| `A0_s32in` | 66.450 | 49.645 | −26.251 | yes | yes | 40 |
| `A0_flatin` | 32.286 | 15.481 | **−60.416** | ⛔ no | ⛔ no | 0 |
| `A0_flatout` | 82.694 | 65.889 | −10.007 | yes | yes | 80 |
| `A0_flatio` | 22.190 | 5.385 | **−70.511** | ⛔ no | ⛔ no | 0 |
| `A0_fmtin` | 92.652 | 75.847 | −0.049 | yes | yes | 80 |
| `A0_donate` | 94.259 | 77.454 | +1.557 | yes | yes | 80 |

**Every arm that moved also deleted the named region in its own dumped HLO** — the spec's rule was
"an arm whose time moved but whose HLO still contains the copy has not tested the hypothesis", and it
was applied per arm, not in aggregate. Additivity `(flatin + flatout)/flatio = 0.999`: the two regions
do not overlap, which is the check `[[single-region-deletion-arms-overlap]]` demands.

**`A0_flatio`'s residual is 5.385 µs.** That, not 75.683, is the irreducible non-kernel cost of
invoking this op under this harness.

### The two nulls, and why they are worth as much as the wins

- **`A0_fmtin` −0.049 µs is inert, not a null result.** `Layout.AUTO` returned exactly the layout the
  buffer already had (`major_to_minor=(0,1), tiling=((8,128),)`), so the arm moved nothing and its HLO
  is unchanged from `A0_base`. The prober printed this as a `!!` **before** the timing rather than
  explaining the flat number afterwards. It did not test the hypothesis and is reported as inert.
- **`A0_donate` +1.557 µs is a real null** — inside the 1.641 spread, **and** its HLO still contains
  both the data-format-call and the copy. The relayout is not an aliasing or liveness artifact.

⚠ **Host-bound above the harness's own 5% limit:** `A0_flatin` 11.1%, `A0_flatio` **20.4%**,
`A0_donate` 6.1%. The fast arms are approaching the dispatch floor
(`[[tpu-microbench-dispatch-floor]]`), so their **absolute** values are soft. The **direction** and
the HLO deletions are not.

## Q3 — the output-byte term, and it *does* reach production

`A0_flatout` = **10.007 µs for 8 MiB = 1.251 µs/MiB**, against probe 10's fitted 1.118 (8.94 µs here)
— 12% apart, same order, and now with a named mechanism instead of an unexplained slope.

`A0_base` ends `ROOT %reshape.3 = s32[1024,2048]{1,0:T(8,128)} reshape(…call-done)`, **retiling** the
kernel's flat `s32[2097152]{0:T(1024)}` output. `A0_flatout`/`A0_flatio` have no reshape at all.

**The production module has the same line** — `ROOT %reshape.1 = s32[1024,2048]{1,0:T(8,128)}
reshape(%sc_topk_direct_b1024_n10240_k2048.3.cloned.1.call-done)` — which I confirmed directly.

⚠ **Unsettled, and the prober flagged it:** compiling `streamindex_topk` **standalone** forces its
ROOT to the default entry layout. Whether the retiling survives when the consumer is the in-model MLA
attention kernel is **not decided by this job**.

⚠ **A metric blind spot, corrected by the prober itself:** the automated activation table printed
`A0_flatout NO-DELETION` and a `!!`, because `materialized_MiB` counts only `copy` and `fusion`
opcodes and this region is a `reshape`. The arm did delete what its name claims. This is
`[[enumerate-framework-ops-before-picking-a-join-key]]` in miniature — a filter chosen on opcode
silently excluded the op it was built to find.

## Cross-job controls

| control | probe 11 | probe 10 | Δ |
| --- | ---: | ---: | ---: |
| `B_base` | 326.770 | 326.732 | **+0.01%** (probe 09 326.782, probe 08 327.166) |
| `A0_base` | 92.701 | 92.118 | +0.63% |
| `A0_base` residual | 75.896 | 75.349 | +0.73% |

The instrument reproduced across four jobs. ⚠ Per `[[one-drift-null-is-a-number-not-a-floor]]`, +0.63%
is a **drift reading**, not a proof of stability — but it is the fourth consecutive agreement on
`B_base` inside 0.13%.

## Decided

1. ⛔ **D3 is NOT a `carried:` handoff on its input leg.** The `sparse-core-data-format-call` and its
   34.165 µs untiling copy are **absent from production**, verified in the production HLO. Remove it
   from §10's carried list; there is nothing at the call site to hand off.
2. ✅ **The `design.md` denominator is overstated, by between 34.165 and 60.416 µs.** Firm leg: the
   untiling copy. Open leg: whether production's same-shape `bitcast-convert` is aliased away or
   materialised at ~26 µs. **Quote the range until it is closed**; do not pick the flattering end.
3. **Every prize in `design.md` § 9 scales up by the same factor** — 189.98/155.82 = **1.219** at the
   firm leg, 189.98/129.56 = **1.466** at the optimistic one. ⚠ **The ranking is invariant**, exactly
   as §10 predicted; what changed is what the exercise is worth, and it went **up**.
4. **`A0_flatio`'s 5.385 µs replaces 75.683 µs as the non-kernel floor.** `probes/09`'s bracket
   collapses to its optimistic leg, and §1's 95.73 µs "effective floor" stays withdrawn.
5. ⚠ **A live output-retiling term, 10.007 µs, present in the standalone production entry.** It is
   the only survivor of D3 and it is a call-site concern, not a Pallas one. Not yet promotable to a
   handoff: the in-model question is open.
6. **Never quote "~73 µs" or "~70.5 µs of the residual does not exist in production" again.** The
   first was probe 10's HLO estimate of a term that turned out to be harness-only; the second is this
   probe's report over-transferring an arm across two ops that share an opcode and nothing else.

## Open, with the check that closes each

| open | check | cost |
| --- | --- | --- |
| is production's `%bitcast_convert_type.3` materialised (~26 µs) or aliased away? | an **existing GLM-5.2 prefill xprof** — is there a ~26 µs op between `StreamIdxTC` and `sc_topk_direct`? Failing that, XLA's buffer-assignment dump | no job if a trace exists |
| does the output retiling survive in-model? | compile the enclosing `mla_attention_op` entry, or read the full-model optimized HLO, and grep for the `reshape` on the `call-done` | no job |
| ⚠ **is the step-3 harness measuring a shape production never runs?** | it is — see below | — |

## ⚠ The consequence for step 3, which is larger than the number

The harness hands the kernel a `f32[1024,10240]` entry parameter. **Production hands it a
`f32[1024,80,128]` that is already an intermediate.** Every step-3 measurement taken through the
harness's entry signature therefore carries a 34–60 µs constant that production does not pay, and a
constant in the denominator **dilutes every delta** measured against it.

`instruments/` is read-only after step 1 and the frozen harness md5 is what makes probes 08–11
comparable, so **do not edit it**. Instead: step-3 arms construct their own thunks (that is exactly
what `A0_flatin` did), so **build every step-3 variant, and the incumbent, against a
production-shaped entry**, and say so in `results.md`. Otherwise a −10% variant reads as −7%.

## Deviations from spec

1. **`A0_flatin`/`A0_flatout`/`A0_flatio` are the prober's arms, not the spec's.** Judged correct:
   they are the only arms that actually delete the regions, and without `flatin` the spec's
   `A0_s32in` alone would have half-deleted the input path and reported an ambiguous −26 µs. All
   three spec arms (`s32in`, `fmtin`, `donate`) were also run.
2. `A0_fmtin` construction: jax refuses to *call* an `AUTO`-layout jit with concrete arrays, so AUTO
   was used only to ask XLA for its preferred format and the timed thunk is an ordinary `jax.jit`
   called with arrays `device_put` into that format outside the timed region. It turned out inert.
   `with_layout_constraint` was not tried.
3. `input_output_aliases` was not used for Q3; `A0_flatout` deletes the output region outright.
4. Null host-bound (24–26%) is reported but not flagged — a null is host-dominated by construction.
5. Extra artifacts beyond the requested set: `hlo/`, `validation.local.txt`, `run_cmd.sent.txt`,
   `run_cmd.readback.txt`, `fixture_hlo.txt`, `hlo_tools.py`.
6. **No README was written by the prober** (it was told not to). This file is the main thread's, and
   the "correction" section above is a main-thread finding that reverses part of the prober's
   headline.
