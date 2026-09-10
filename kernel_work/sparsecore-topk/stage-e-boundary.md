# Stage E — the boundary

**Numbers.** Every `Δ time %` is `(new − old)/old` of time; **negative is faster**, and no signed
percentage appears outside that column. The denominator is the **harness 189.98 µs** (`design.md:82`)
throughout, for consistency with §3/§5/§9. Production is **155.82 µs firm – 129.56 µs optimistic**
(`design.md:83`); multiply any percentage here by **1.219–1.466** to read it against production
(`probes/11/README.md:148-150`). Tags: `measured (job)` / `modelled` (formula at first use) /
`assumed` / `unmodelled`.

⚠ **`references/01-fusion-boundary.md` is read here with the grain of salt the block requires.** Its
check-A arithmetic `2 × bytes / 2.97e12` (`:54`) is a **TensorCore** HBM figure; this block's own
measured rate is **1255 GB/s** for its 40 KiB SparseCore slice (`design.md:78`) and its two edge
transfers are directly measured (13.469 µs in, 5.872 µs out, `design.md:4a`). **Where the reference
and the measurement disagree the measurement wins, and check A below is stated from the measured
legs, not from 2.97e12** — which understates both edges by ≈2.4×. `hardware-facts.md`'s 8×128
relayout model is void here: one register axis, 16 int32 lanes, no sublane (`design.md:4b`).

**Independence from `probes/13`.** No price in this document uses the refinement level count. The
edges are priced from transfer bytes and from measured per-op times (`design.md:4a`), none of which
is a function of `1.740` / `1.300` / the retracted `2.231`. §4 restates `v3`'s and `v5`'s boundary
fields against probe 13's numbers anyway, with the tags the coordinator asked for.

---

## 1. The decision, in three sentences

**The kernel stays exactly one `plsc` mesh call, and the two things §9:801 got wrong are that this
was ever the whole question and that the edges were already optimal**: the upstream edge is clean by
accident of the producer's shape and carries one deletable XLA op worth up to 26.251 µs, the
downstream edge carries a measured 10.007 µs retiling that no §9 plan reaches, and `v4`'s preferred
within-slice form introduces a **new internal edge** — an `emit_pipeline` grid `(n // chunk,)`
enclosing exactly `cvt ⊕ hist0` at **chunk = 1024 words**. Fusion with the producer `StreamIdxTC` is
**illegal, not merely unprofitable** — it is a TensorCore kernel whose `dot_general` has no
SparseCore lowering rule (`probes/12/README.md:50-53`) — so its 13.469 µs+ round trip is reachable
only as **overlap** (D7), which by the §10 rule is now **OPEN, not PARKED**, with a named free probe.
Fusion with the consumer `sc_dsa_gather` is legal, unblocked by layout, and worth **more** than any
single §9 plan (≤ 21.7 µs, −11.42%), and it is rejected here only because it changes the block's
output value — a step-1 re-scope, handed off, not dropped.

## 2. The option table — the six checks, one row per boundary alternative

`A` round trip · `B` already resident · `C` VMEM / block size · `D` contraction-axis conflict ·
`E` layout conflict · `F` units, registers, bundles (`01-fusion-boundary.md:54-59`).

| # | boundary alternative | A (measured legs) | B | C | D | E | F | Δ time % | check that decided it | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| **U1** | keep the input edge: flat `s32[10485760]{0:T(1024)}` view of the producer's `f32[1024,80,128]{2,1,0}` | n/a | n/a | n/a | n/a | ✅ **none** — minor dim is 128 = one lane-tile, so the flatten is `%bitcast.74`, free (HLO `hlo-prod_early_exit_False.txt:1477`; 0 `sparse-core-data-format-call`, only `copy` in the module is `%copy.11 = f32[1024,64]` — re-derived by grep) | — | 0 | **E** | **KEPT** |
| **U2** | delete `%bitcast_convert_type.3`: declare the kernel input `f32` and reinterpret in-register instead of `jax.lax.bitcast_convert_type` at `sparsecore_topk.py:438` | — | — | 0 | — | same shape, same layout, 4 B→4 B — semantically a no-op | ⚠ XLA planned a **loop**: `window_config iteration_bounds ["2","5","1"]`, `estimated_cycles 66712`, **and `used_scoped_memory_configs` 16 MiB** (HLO `:1267`) — 1 of only 27 ENTRY instructions with a non-empty `used_scoped`, and `%bitcast.74` has none | **0 or −13.82%** (26.251 µs, `A0_s32in` `measured (j-9b520877)`, ⚠ prices the *harness* op) | **F** — feasibility `unmodelled` | **OPEN**, mandatory (§6.2) |
| **U3** | change the declared input layout/dtype at the call site | — | — | — | — | ⛔ any layout other than the one it already has re-introduces the untiling copy: **+17.98%** (34.165 µs `measured (j-9b520877)`) is the counterfactual | — | +17.98% | **E** | **REJECTED** — "already optimal" is the answer, and it is evidenced, not assumed |
| **U4** | bf16 scores — halve the 41.94 MB | −3.53% on the read leg (half of 13.469 µs `measured (j-300d4690)`) | — | halves `resident_vmem` | — | ⚠ bf16 relayout cost is `unmodelled` in the reference and the reference is a TC model | ⚠ 16 int32 lanes, no sublane; bf16 packing on SC is `unmodelled` | ≥ −3.53% | **exactness** — a top-k over bf16 selects a **different set**; the graph is `exact` throughout (`design.md:3`) | **carried (H4)**, owes a gate |
| **P1** | fuse the producer `StreamIdxTC` into the top-k | −7.09% read leg `measured` + write leg `unmodelled`; ≤ −14.18% if symmetric | ⛔ **no** — the producer holds only a 2 MiB `bo_scores_double_buf` tile (`2×128×16×128×4`, `streamindex_topk.py:683-684`); a top-k row needs all `n=10240` | a full row per token in the bq block = 5 MiB vs that 2 MiB | ⛔ **yes** — producer tiles the kv axis in `bkv` blocks (`:636-643`) and contracts over heads; top-k reduces the **whole** kv axis. Classic "accumulator goes from a tile to the whole output" | — | ⛔ **fatal**: producer is **TensorCore** (`pl.pallas_call`, MXU fp8 dots), consumer is **SparseCore** (`plsc.VectorSubcoreMesh`). `dot_general` is one of the 60 TC-only lowering rules absent from `SC_VECTOR_SUBCORE` (`probes/12/README.md:50-53`) | unreachable | **F** (units) | **REJECTED** — illegal, with the largest check-A number on the board |
| **P2** | **D7** — *overlap* TC and SC instead of fusing | same round trip, not deleted but hidden: `min(t_TC, t_SC)` | — | 0 | — | — | ⚠ the three `StreamIdxTC` calls are **serially chained by aliasing** (operand 10; HLO `:1260,1262,1263`), so the SC kernel starts only after the third | `unmodelled` (`t_TC`) | **§10 rule**: verdict turns on an `unmodelled` term | **OPEN** (was PARKED), mandatory (§6.1) |
| **P3** | split the top-k call by phase group so decode rows' top-k overlaps the mixed `StreamIdxTC` | — | — | — | — | decode rows are complete after the `-d-` calls (HLO `:1260,1262`) | — | **0 at the measured shape** (b=1024, all rows mixed); `unmodelled` at a mixed step | **A** — prize is zero where the denominator is measured | folded into **P2**'s handoff, not dropped |
| **D1′** | keep the flat output + XLA `ROOT %reshape.1` | — | — | — | — | ⛔ flat `{0:T(1024)}` → `{1,0:T(8,128)}`, **10.007 µs `measured (j-9b520877)`** = 1.251 µs/MiB; ROOT carries `used_scoped_memory_configs` 4 MiB (HLO `:1480`) — an emitted loop, not a bitcast | — | 0 | incumbent | **KEPT** conditionally — see §3c |
| **D2′** | **`v11`** — kernel emits `s32[b,k]{1,0:T(8,128)}` directly | deletes the retiling, not the transfer | — | 0 (same `rowbuf`) | — | moves the relayout inside the out DMA: one 8 KiB contiguous run per row becomes **16 runs of 512 B** at stride 1024 words. 512 B = 128 int32 = the SC granule exactly | ⚠ can Pallas SC declare a 2-D tiled `out_type`? `unmodelled` | ≤ **−5.27%**, less the strided-DMA cost | **F** — feasibility `unmodelled` | **OPEN**, mandatory (§6.2) |
| **D3′** | consumer accepts the flat `{0:T(1024)}` output | — | — | — | — | same 10.007 µs, deleted at the other end | — | ≤ −5.27% | **conditional** on the in-model check | **carried (H2)** — see §3c, written both ways |
| **D4′** | fuse the consumer `sc_dsa_gather` (**S7**, `hardware_map.md:680`) | −3.09% write leg (5.872 µs `measured (j-4e0df30a)`) + read leg `unmodelled`; `hardware_map` models the pair at 7.8 µs. **Plus** D1′'s 10.007 µs, which vanishes with the buffer | — | ⚠ the gather's source is an indirect HBM read that cannot be resident | — | ✅ **no conflict** — both sides `[lane=k]`, same unit (`hardware_map.md:151,680`) | — | **−8.36% to −11.42%** | **none of the six** — it is out of the block's **contract**: it changes the output value | **carried (H3)** — a step-1 re-scope, priced, not silently dropped |
| **I1** | no internal grid (incumbent) | — | — | 14.03% at `v3` (`design.md:551`) | — | — | ⛔ overlap fraction **0.0000**, 3 transfers / 3 waits, two literal `snop`s in the slack (`probes/06` Q1) | 0 | incumbent | superseded by **I4** |
| **I2** | cross-wave `emit_pipeline`, `grid=(num_waves,)` | ≤ −7.09% (13.469 µs) | — | +40,960 B → **114,496 B = 21.84%** (`design.md:574`) | — | — | 32 waves at prefill, **1 at decode** (`probes/12/README.md:127-130`) → **worth exactly 0 at decode** | ≤ −7.09% prefill / 0 decode | **C** and phase coverage | **REJECTED** in favour of I4; prefill-only fallback |
| **I3** | within-slice pipeline enclosing `cvt` only | ceiling `min(DMA, t_cvt)`; `t_cvt` is **−8.20 µs measured** (`design.md:4a`) — pass 1 is shorter than the DMA it must hide | — | +8,192 B | — | — | — | small | **A** — the enclosed pass is too short | **REJECTED** |
| **I4** | **within-slice pipeline enclosing `cvt` ⊕ `hist0`**, `grid=(n // chunk,)`, chunk = 1024 words | ceiling `min(13.469, t_cvt+t_hist0 = −8.20+33.91 = 25.71) = 13.469 µs`, less the unhidden head, less cvt's re-emerging absorption | ✅ **yes** — `resident_vmem` stays the persistent scratch; probe 12 variant **B** is this exact shape and lowers clean | +2·1024·4 = 8,192 B → **81,728 B = 15.59%**, i.e. **24,576 B cheaper than I2** | — | — | ⚠ Mosaic→LLO not exercised by `probes/12` (`:113-115`) — a job-only risk | **−2.06% to −6.38%** `modelled` | **C** + **A**; covers both phases | **KEPT** — this is the new edge |

## 3. Per-edge analysis and priced verdicts

### 3a. Upstream — "already optimal" is right about the *layout* and wrong about the *op*

The layout half is closed and I re-derived it rather than reading it: `grep -c
sparse-core-data-format-call` on `probes/11/hlo/hlo-prod_early_exit_False.txt` → **0**, and the only
`copy` in the module is `%copy.11 = f32[1024,64]` (256 KiB of `indexer_weights`). Production never
materialises a 2-D `f32[1024,10240]`; its minor dim is 128 = one lane-tile, so `words.reshape(-1)`
(`sparsecore_topk.py:486`) is `%bitcast.74`, free. **U3 is REJECTED with a price**: 34.165 µs
(+17.98%) is what the harness pays for having the layout wrong (`A0_s32in`−`A0_flatin`, `j-9b520877`).

What is **not** optimal is that `sparsecore_topk` asks XLA for a dtype change it does not need.
`jax.lax.bitcast_convert_type(scores, jnp.int32)` at `:438` emits `%bitcast_convert_type.3` — same
shape, same layout, 4 B → 4 B. Probe 11 left it open at ±26.251 µs. **Two facts now weigh toward
"materialised"**, one of them new: XLA attached a `window_config` with `iteration_bounds ["2","5","1"]`
(probe 11's evidence), **and** it is one of only **27 of 293** ENTRY instructions carrying a
non-empty `used_scoped_memory_configs` — 16 MiB of VMEM (HLO `:1267`) — while the free `%bitcast.74`
carries no `backend_config` at all. Verdict: **U2 OPEN**, and the point of U2 is that it makes the
open leg *moot* — an op you never emit needs no aliasing decision. The kernel reads bits, never
floats (`monotone_key`, `:117-118`), so the change is a dtype declaration, not arithmetic.

### 3b. Internal — the new edge, and what `chunk` is

`v4`'s within-slice form is the boundary decision this stage actually makes. The pipeline body is
probe 12 variant **B** verbatim: `emit_pipeline` HBM → the persistent `pltpu.VMEM` `resident_vmem`
scratch, body converts to monotone keys **and** runs the L0 scatter off the pipeline buffer, then one
manual out DMA (`probes/12/README.md:99`).

- **It encloses `cvt ⊕ hist0` and nothing more.** `fold_priv` needs the complete histogram;
  `REFINEMENT_LEVELS` and `emit_body` re-read `resident_vmem`, not HBM (`probes/12/README.md:146-150`).
  `zero_priv(NUM_BUCKETS)` (`sparsecore_topk.py:185`) must move **before** the pipeline; it does not
  depend on the scores, so that is free.
- ⚠ **This is `hardware_map.md:675`'s S2 seam, and S2's ~10 µs (2.2%) is `modelled` and contradicted
  by measurement on this block.** `cvt`'s **measured** marginal cost is **−8.20 µs** (`j-300d4690`) —
  the pass is already free because it absorbs the DMA's arrival latency. So S2's "read half of `cvt`"
  prize is ≤ 0 at the incumbent's schedule; **S2 only becomes positive inside I4**, once the DMA is
  pipelined and cvt stops absorbing. **S2 and I4 are one move. Do not sum them.**
- **`chunk` must divide `n`** — a `BlockSpec` index map returns block indices and the kernel flattens
  to 1-D, so `row_c · n` must be a multiple of `chunk` (`probes/12/README.md:156-159`).
  ⛔ **Probe 12's escape clause is wrong at the shipping shape.** It says "all production `n` are
  powers of two in 2048..256K, so any power-of-two chunk works". **`n = 10240 = 2^11 · 5`, not a power
  of two**: `num_sublanes_total = 80` (`streamindex_topk.py:603-605`, `align_to(9,2)·1024//128`) and
  the score buffer is `(tokens, 80, 128)` (`:726`). Legal chunks are the divisors of 10240 that are
  multiples of `LANES=16`: 16, 32, 64, 80, 128, 160, 256, 320, 512, 640, 1024, 1280, 2048, 2560, …
  **4096 is illegal.**
- **Pick 1024 words (4 KiB), sweep {512, 1024, 2048}.** With double buffering, step 0's DMA is never
  hidden, so the unhidden head is `13.469 / (n/chunk)`: **2.694 µs at chunk 2048, 1.347 at 1024,
  0.673 at 512** (`modelled`, from the `measured` 13.469 µs). Against a net that is at best 13.469 µs
  and at worst `13.469 − 8.20`, a 2.694 µs head eats half the pessimistic case; 1024 costs 8,192 B of
  VMEM (15.59% total) and 10 pipeline steps per wave.
- **Priced verdict.** Optimistic `13.469 − 1.347 = 12.122 µs`, **−6.38%**. Pessimistic
  `13.469 − 1.347 − 8.20 = 3.922 µs`, **−2.06%** — cvt's absorption is banked twice otherwise. Both
  `modelled` on `measured` inputs. ⚠ This is *narrower and lower* than `design.md:576`'s "≤ 13.469 µs",
  and the narrowing is the point: the ceiling was never reachable.

### 3c. Downstream — written both ways, and the bet

A separate agent is deciding whether `ROOT %reshape.1` survives in-model; I did not duplicate it.

- **If it survives.** Promote **H2** to a real `carried:` handoff (§5) *and* keep **`v11`** OPEN — a
  kernel-side emit does not need the consumer to change, so the two are alternatives, not duplicates.
  Prize ≤ 10.007 µs, **−5.27%** harness / −6.42% to −7.72% of production.
- **If it does not survive.** **`v11` is REJECTED** with the priced reason "prize measured 0
  in-model", H2 is withdrawn exactly as D3's input leg was (`design.md:1231`), and 10.007 µs joins the
  standalone-entry artifact class. Nothing else in this document moves — the internal edge, U2, P2
  and D4′ are all independent of it.
- **Which way I would bet: against survival, ~60/40.** `streamindex_topk` is itself `@jax.jit`
  (`:509`) and inlines under the attention op's jit, so the standalone compile's forced ROOT layout is
  an artifact of compiling it alone — the same failure mode that produced the input-relayout
  over-claim. The named consumer is `sc_dsa_gather`, whose index operand is `(b)[lane=k]`
  (`hardware_map.md:151`) — **which is what `{0:T(1024)}` already is**. **The falsifier**: if anything
  on the TensorCore reads the indices (the `P>1` `take_along_axis` at `sparsecore_topk.py:617` is one),
  a `{1,0:T(8,128)}` retile reappears and I lose the bet.

## 4. What this changes in the version graph

| node | change | tag |
| --- | --- | --- |
| `v2` | none — a `[mutate]`, no edge | — |
| `v3` | **boundary field unchanged**; the move is internal. ⚠ its prize is restated by `probes/13` from −4.59% to about **−16.5%**, driven by `cnt` (measured mean 1071, not the 4096 safe bound), **provisional** pending the critical-subcore re-aggregation (`cnt` max 6495). No boundary argument here depends on its magnitude | provisional |
| `v4` | ⚠ **`boundary` and `grid` fields both change.** Boundary: "one `plsc` mesh kernel" gains an **inner pipeline grid** `(n // chunk,)` enclosing `cvt ⊕ hist0`. Grid: no longer "unchanged; still a static `fori_loop`". Live set **114,496 B (21.84%) → 81,728 B (15.59%)** at chunk=1024. Predicted **≤ 13.469 µs → 3.922–12.122 µs, −2.06% to −6.38%**. `chunk ∈ {512,1024,2048}`, default 1024; **4096 illegal** | `modelled` on `measured` |
| `v5` | **boundary field unchanged** — `NUM_BUCKETS` is a `[grid]` constant, and its ±768 B is noise against I4's 8,192 B. ⚠ direction **reverses** (`probes/13`): NB=64 −11.94%, NB=512 **+17.07%**, zero non-convergence at any radix. Rests on `fixed(NB) ∝ NB`, **`modelled` from a single anchor at NB=256 and unverified** | reversed, unverified scaling |
| `v6` | ⚠ **boundary field must be restated.** `@pl.when(i_hi > k)` cannot wrap I4's pipeline — that puts its `run_scoped` allocations under a conditional. The repo idiom is a **grid that evaluates to 0** (`ragged_gather_v2.py:102,208`; `probes/12/README.md:160-163`). `v6 ∘ v4` is therefore `grid=(0,)` on the fast path, not a `pl.when` | source-read |
| `v7` | none | — |
| `v10` | none, and it is the **best host for I4**: live set ~41,000 B (7.8%), so the pipeline buffer costs 1.6% of budget on top | — |
| **`v11`** | **new node implied** — see below | — |

---

### `v11` — retiled output emit

- **Parent:** `v1` (an edge move; it composes with every node) · **Move:** `[boundary]`/`[relayout]`
  declare `out_type = jax.ShapeDtypeStruct((b, k), jnp.int32)` with the consumer's tiling instead of
  the flat `(b*k,)` at `sparsecore_topk.py:451`, and let the op-13 out DMA (`:315-319`) write the
  tiled rows directly; `out.reshape(b, k)` at `:491` disappears
- **Motivated by:** `probes/11` Q3 — `ROOT %reshape.1 = s32[1024,2048]{1,0:T(8,128)}
  reshape(…call-done)` is **in the production entry**, **10.007 µs `measured (j-9b520877)`** =
  1.251 µs/MiB, and carries `used_scoped_memory_configs` 4 MiB (HLO `:1480`). It is the only survivor
  of D3
- **Exactness:** **`exact`** — a layout change computes the same function. ⚠ Not "no evidence needed":
  it changes the *store pattern*, so it owes the same bit-identical row gate `v2` passed (`j-300d4690`)
- **State diff from `v1`:** `boundary` — the kernel's output edge moves from XLA to the kernel.
  `transfer` — T1 on `rowbuf → out_hbm` goes from one 8 KiB contiguous run per row to **16 runs of
  512 B at stride 1024 words**. 512 B = 128 int32 = the SparseCore granule exactly
- **Live set:** unchanged — same `rowbuf_vmem`, no new buffer
- **Predicted:** ≤ **−5.27%** (10.007 µs) less the strided-DMA cost, which is `unmodelled`
- **Built?** **No** · **Measured:** — · **Verdict:** **OPEN**, mandatory probe (§6.2). ⛔ Not `PARKED`:
  its verdict turns on an `unmodelled` term, which `TEMPLATE-design.md` §10 forbids parking on
- **Note:** ⚠ **conditional on the in-model check** (§3c). If the retile does not survive in-model this
  node is **REJECTED** with prize 0. It is also **subsumed by H3** — a fused consumer has no HBM output
  to retile.

## 5. `carried:` handoffs

**H1 — D7, the SparseCore↔TensorCore pipeline.** (1) *Change*: overlap `StreamIdxTC` with
`sc_topk_direct` in the attention op; the three producer calls are serially chained by aliasing
(operand 10, HLO `:1260,1262,1263`), and `P=1` leaves no partition loop inside the kernel
(`design.md:380-382`). (2) *Prize*: `min(t_TC, t_SC)`, `t_SC` = 189.98 µs harness / 155.82–129.56 µs
production `measured`; **`t_TC` `unmodelled`** — the whole handoff. (3) *Correctness*: none needed,
a schedule change. (4) *Constraint*: no §9 plan reaches outside the Pallas kernel. (5) *Route*:
`xprof-analyzer` for `t_TC`, then a vLLM-side change → `/decompose-optimization`.

**H2 — the output retiling, 10.007 µs.** ⚠ **Conditional — do not promote until §3c returns.**
(1) *Change*: have the in-model consumer accept `{0:T(1024)}`. (2) *Prize*: 10.007 µs
`measured (j-9b520877)` at the standalone `streamindex_topk` entry — **that is the level, and it is
the weakness**. (3) *Correctness*: none needed. (4) *Constraint*: outside the Pallas kernel;
`v11` is the in-kernel alternative. (5) *Route*: the MLA attention op → `/decompose-optimization`.

**H3 — fuse the consumer `sc_dsa_gather` (S7 + D1′).** (1) *Change*: emit gathered values, not
indices, from one SparseCore kernel; both sides are already `[lane=k]` on the same unit
(`hardware_map.md:680`). (2) *Prize*: 5.872 µs out-DMA `measured (j-4e0df30a)` + an `unmodelled` read
leg (`hardware_map` models the pair at 7.8 µs) + 10.007 µs of retiling that vanishes with the buffer
= **−8.36% to −11.42%**, larger than any single §9 plan. (3) *Correctness*: a new output value; needs
a fresh differential test, not `v2`'s row gate. (4) *Constraint*: it changes the **block's contract**,
which step 1 fixed — a version cannot change what the block computes. (5) *Route*: a new
`/write-tpu-kernels` exercise scoped to `topk ⊕ gather`.

**H4 — bf16 scores.** (1) *Change*: `out_dtype` at `streamindex_topk.py:741` and the kernel's key
width. (2) *Prize*: ≤ −3.53% on the read leg (half of 13.469 µs `measured`), plus half the producer's
write. (3) *Correctness*: ⛔ **`approximate`** — a top-k over bf16 selects a different set; owes a
metric, a threshold set before the run, and a job. (4) *Constraint*: it breaks the graph's `exact`
invariant (`design.md:3`), and bf16 on 16 int32 lanes with no sublane is `unmodelled`. (5) *Route*:
an accuracy-gated experiment, not a kernel version.

## 6. Open items and the probes they make mandatory

By `TEMPLATE-design.md` §10, a verdict resting on `unmodelled` / `assumed` is **OPEN**, and the probe
that prices the term is mandatory **in this step**.

1. **`t_TC` (P2/D7) — `unmodelled`.** ⛔ D7 must move **PARKED → OPEN**; `design.md:1223-1227` already
   says it "needs a number before anyone can judge it" and then parks it anyway. **Probe:** read the
   duration of `StreamIdxTC-m-bq_128-bkvp_2` and `StreamIdxTC-d-bq_1-bkvp_2` (scope names from
   `streamindex_topk.py:703`) from an **existing** GLM-5.2 prefill xprof. **No job.**
2. **`v11` and U2 feasibility — `unmodelled`, one script, no job.** Extend `probes/12/compile_check.py`
   with two variants: (a) a `plsc` kernel whose `out_type` is 2-D `s32[b,k]` with the consumer tiling,
   out DMA writing 16×512 B strided runs; (b) the same kernel taking an `f32` HBM ref and
   reinterpreting in-register (does a Pallas SC ref bitcast, or must the scratch dtype change?).
   Keep the `jnp.sin` negative control — a lower that cannot fail proves nothing. **Then** one arm
   each for the price.
3. **`chunk` for I4 — three compile-time arms {512, 1024, 2048}, one job.** Fold into the `v5` NB
   sweep's job; both are compile-time constants on the same kernel. This job is also the **only**
   thing that exercises Mosaic→LLO, which `probes/12:113-115` explicitly could not.
4. **The input `bitcast_convert_type` open leg, 26.251 µs of denominator.** Same existing-xprof check
   as (1) — look for a ~26 µs op between `StreamIdxTC` and `sc_topk_direct`. U2 (item 2) makes the
   answer irrelevant, which is the cheaper resolution.
5. **`v3`'s `cnt` re-aggregation** (in flight) and **`v5`'s `fixed(NB) ∝ NB`** (unverified, single
   anchor). Neither touches an edge; recorded so the interaction is on the record as considered.
