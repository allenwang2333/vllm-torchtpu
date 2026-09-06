# Area (5) — Which unit does each piece of work

Constraint sheet. Decides nothing. Sources: `05-compute-units.md`, `08-sparsecore.md` (primary),
`hardware-facts.md`, step-1 `report.md`, `sparsecore_topk.py`, `streamindex_topk.py`.
Tags: `measured` = on hardware in a named job · `derived` = arithmetic over measured rows ·
`modelled` = from a formula · `assumed` · `unmodelled` = no term exists anywhere.

---

## 0. The structural fact area (5) contributes, before any occupancy

`_pick_partition` is a **VMEM-capacity function of the padded `n`**, and the wave count follows from it:

```
p     = 2^ceil(log2(n / MAX_SLICE_WORDS)),  MAX_SLICE_WORDS = 65536 words = 256 KiB
waves = cdiv(b*p, 32)                       # 32 = 2 SC cores x 16 vector subcores
```

`row_lengths` appears in neither. p=8 would need `n_p = 131072` words = **512 KiB per subcore, above
the 480 KiB the compiler accepts** [`sc_vmem_per_subcore`, verified: j-996520b8-2265-4c2c-b002], so
p=16 is *forced* at every AgentX shape. This is Finding 2's mechanism: cost tracks `[T, n]` because
the partition — and hence the serial wave count every subcore must execute — is computed from `n`.

| shape | b | n | p | b*p slices | **waves W** | live slices | slot occupancy |
|---|---|---|---|---|---|---|---|
| A-D64-8k | 64 | 9216 | 1 | 64 | **2** | 64 | **100%** |
| B-D4-256k | 16 | 1048576 | 16 | 256 | **8** | 16 | **6.25%** |
| B-P1024-256k | 1024 | 1048576 | 16 | 16384 | **512** | 4096 | **25%** |

Worse than the slot number: slices are assigned `row = w*32 + flat`, and the live partitions of a row
are always the low `j`, so at B-D4-256k only **8 of 32 subcores ever touch data** while all 32 execute
all 8 waves. The imbalance, not the volume, is what `row_lengths` fails to remove.

---

## 1. Per-unit occupancy (the area (5) emission)

MXU floor `2mkn / peak`, fp8 measured **1419.9 Tops/s**; row-block waste `min(1, rows/128)`;
VPU `flops / 1.72e12`; SC floor `bytes / 337e9` aggregate. Scoring FLOPs `= 2*T*L*H*D`, H=32, D=128.

| | **A-D64-8k** | **B-D4-256k** | **B-P1024-256k** |
|---|---|---|---|
| measured full / SC alone | 363.2 / **189.5** µs | 1117.9 / **795.9** µs | **50916.8** / not decomposed |
| scoring (full − SC) | 173.7 µs | 322.0 µs | — |
| MXU floor (fp8, no penalty) | 3.4 µs | 6.1 µs | 1550 µs |
| MXU row block → waste factor | 64 → **0.50** | 16 → **0.125** | 1024 → **1.00** |
| MXU floor with penalty | 6.8 µs | 48.4 µs | 1550 µs |
| **MXU occupancy** | **3.9%** | **15%** | **≤3.0%** of the whole op |
| VPU (relu + head sum) | 0.03 µs | 0.05 µs | 3.9 µs — **<0.01%** |
| **SC bytes / SC floor** | 2.25 MiB / 6.7 µs | 4 MiB / 12.4 µs | 4 GiB / 12.7 ms |
| **SC occupancy vs its own floor** | **3.5%** | **1.6%** | ~25% (predicted) |
| SC VMEM used / 480 KiB | 62 KiB — **13%** | 282 KiB — **59%** | 286 KiB — **60%** |

Two readings. **The MXU is idle** at every shape, and at B decode it is additionally throwing away
7/8 of itself to a 16-row block (the 128-row cliff: 05's table measures 5.42x at 16 rows). **The
SparseCore is not bandwidth-limited either** — 1.6–3.5% of its own 337 GB/s streaming floor. Neither
unit is near any roofline; the time is somewhere neither instrument prices.

One correction to 05's rule of thumb while reading the above: "the crossover is at roughly 1
flop/byte" is the **VPU** crossover. The MXU crossover on this chip at fp8 is `1419.9e12/3.7e12` =
**384 flop/byte**; scoring runs at ~38 flop/byte, so it is memory-bound against the MXU and
compute-bound against the VPU. Using the 1 flop/byte figure here misclassifies it.

**Where SparseCore does not fit, and why:**
- **Selection on one subcore.** `n_p = n` = 4 MiB resident. Refused: 480 KiB ceiling, `'memref.alloca' op E3000`.
- **p = 8 (halving W).** 512 KiB > 480 KiB. Same refusal. **The wave count is not tunable from here.**
- **Scoring on SparseCore.** No `dot_general` at all, and no `matmul_acc_lhs`/`push_rhs`/`pop`. Absolute.
- **One row fanned across 16 subcores with an in-kernel merge.** `subcore_barrier` completes at 2 and
  4 subcores and **hangs the device at 8** [`sc_subcore_barrier_minimal`, verified: j-0222bc6d-c03a-4d93-97a7];
  coordinated fan-out is capped at 4 of 32. A hang strands the slice until its deadline — run any
  probe touching it alone, under `timeout -k 30 900`.
- **`b_pad` index metadata in SC SMEM.** SC SMEM is **4 KiB = 1024 int32**; at B-P1024 `b_pad`=1024 is
  exactly at the limit. The kernel puts it in VMEM, which is correct and must stay that way.

---

## 2. The terms a cost model for exact top-k on SparseCore has to contain

There is no such model in either reference — 08's only fixed-cost figure is a **~35 µs per-*kernel*
launch floor** (`sc_gather_index_count_scaling`), and this kernel's cost is per-*wave*. Fitting the
five decomposed rows to `T = C + W·F(n_p) + A·M(n_p)`, where `A` = live slices on the busiest subcore:

| term | what it is | fitted value | at these shapes | tag |
|---|---|---|---|---|
| **C** | per-call floor: kernel dispatch, `lengths` DMA, `jit` boundary | **~150 µs** (bounded below by A-D64-8k, above by A-P1024-8k) | 150 µs everywhere | `derived`, weak |
| **F(n_p)** | **per-wave cost paid by every subcore whether or not its slice is live** — `scan_glob`(16 iters), `find_bucket`(16), ~15 zero-trip `parallel_loop` dispatches, the 128-iteration `-1` fill, one 8 KiB output DMA | **~73 µs at n_p=65536**, **~20 µs at n_p=9216** → decomposes as `≈11 µs + 0.95 ns/word·n_p` | **75–90% of measured SC time at every shape** | `derived` |
| **M(n_p)** | marginal cost of a live slice: DMA `slice_len` in, `cvt`, hist L0, up to 3 refinement passes, emit | **0.65 ns/word** → 42.9 µs per 256 KiB slice | 5–25% of SC time | `derived` |
| **passes** | how many of the 4 refinement levels run before `done` fires — data-dependent, 1 to 4 | folded into M | register item 8 | `unmodelled` |
| **stage-2 TC gather** | two `jnp.take_along_axis` over the **full padded** `f32[T, n]`: `p*k_p`=32768 indices/row, then k=2048/row | 0.5 M (B decode) to **33.6 M** (B-P1024) single-f32 gathers | at XLA's two measured per-index rates, **0.45 ns or 12 ns** → 15 ms **or 402 ms** at B-P1024 | `unmodelled`, 26x range |
| **imbalance** | `A` is the max over subcores, not the mean; live partitions land on a fixed subcore subset | 4x at B-D4-256k | sets why 4x live data costs +3.5% | `derived` |
| **SC↔TC serialization** | zero | 0.963 chained/summed | 0 µs | `measured` |

Fit quality, `T = 150 + W·F + A·M`, F=73.5 µs, M=42.9 µs (n_p=65536):

| row | W | A | predicted | measured | error |
|---|---|---|---|---|---|
| B-D1-1M | 8 | 1 | 780.6 | 780.6 | fit point |
| B-D4-1M | 8 | 2 | 823.5 | 823.5 | fit point |
| B-D4-256k | 8 | 2 | 823.5 | 795.9 | +3.5% |
| B-D16-256k | 8 | 8 | 1080.7 | 1012.8 | +6.7% |
| A-D64-8k (n_p=9216, F=20, M=6.0) | 2 | 2 | 202 | 189.5 | +6.6% |
| **B-P1024-256k** (stage 1 only) | 512 | 128 | **43.2 ms** | 50.9 ms *(whole op)* | 85% explained |

**Two parameters, five rows spanning a 114x range of live data, ≤7% everywhere — and it predicts the
largest number in the sweep to within 15% from decode data alone.** That is the floor step 3 was
missing: at B-D4-256k the model says the *data* work is 86 µs of 795.9, so a variant that only makes
scanning faster has 11% of the op to win, and a variant that removes waves has 74%.

**One row falsifies the simple form and must be resolved before the model is quoted.** A-P1024-8k is
b=1024, n=9216, p=1 → **W=32 waves**; a wave-constant F of ~93 µs predicts 3.0 ms of SC time inside
a **832.9 µs** full-op measurement. It is only consistent if F scales with `n_p` (the `≈11 µs +
0.95 ns/word` form above) or if C is large and F small. The discriminating probe is one job:
**`--decompose` on A-P1024-8k and A-P512-8k, plus an SC-only sweep of `b` at fixed `n_p` and of
`n_p` at fixed `b`.** Until it runs, F's split between "per wave" and "per resident word" is
`assumed`, and only the *product* `W·F` is `derived`.

**What F is mechanically is unknown and is the highest-value unknown in this exercise.** 90 µs over
~160 loop iterations is ~560 ns/iteration, far above the ~2 ns a 16-lane SC vector op should cost, so
it is not the iterations. The candidates, in order: `plsc.parallel_loop` dispatch (~20 per wave →
~4.6 µs each), the synchronous 8 KiB output DMA's latency, and the `lax.fori_loop` wave boundary.
**Neither reference file has a term for any of the three.**

---

## 3. What the reference says about TensorCore/SparseCore synchronization cost

| fact | value | provenance |
|---|---|---|
| SC→TC data dependency (the serial case) | `chained/summed = ` **0.963** — 20.6 µs *cheaper* than the sum, inside spread. **Crossing between units is free.** | `measured` [`sc_tc_dependent_handoff`, j-0ed646a2-d578-4509-9158] |
| …but the work still costs its full duration | 430.0 µs of SC work adds **430.0 µs** when the TC waits, vs **32.1 µs** when it does not — a **13x** difference from the same handoff | `measured` [j-996520b8 / j-0ed646a2] |
| Overlap behind a **compute-bound** neighbour | score **0.86–0.93**; 432–438 µs of SC work adds **32–61 µs** (7–14%) | `measured` [j-996520b8, j-97fa6617, j-d2494d38-0074-49ec-8979] |
| Overlap behind a **memory-bound** neighbour | score **0.45–0.55**; the same work adds **346–375 µs (80–87%)**. Halving the neighbour's HBM traffic does not recover it; the combined step ran at 1.19 of 3.70 TB/s, so it is **not byte contention**. Mechanism open. | `measured` [j-d2494d38-0074-49ec-8979] |
| Fixed co-scheduled overhead | ~**20 µs** for SC vs ~**30 µs** for XLA, **flat** across neighbours from 77 µs to 1277 µs — you do not need a big neighbour | `measured` [`sc_moe_route_offload_sweep`, j-6449d3ea-b798-4f62-9be0] |
| One kernel holding both units | **impossible.** A TC mesh beside an SC mesh is rejected at the mesh-compatibility check and again in the MPMD path. Composition is XLA-level: both calls in one `jax.jit`, **no data dependence between them**. | `measured` + `source` [j-996520b8; `pallas_call_registration.py:606-613@jax 0.10.2`] |
| Scalar+vector MPMD inside one SC kernel | unproven — `'tpu.enqueue_dma' op Core id must be specified`, four attempts | `measured` (all failures) |

**Applied to this block.** Today the two stages are serial by construction, and the handoff row says
that costs *nothing* to fix by making the crossing cheaper — there is nothing to make cheaper. The
entire question is whether the TensorCore can be given independent work. And the answer the reference
gives is discouraging in a specific way: **the neighbour on offer is scoring, which is memory-bound
at every row of the step-1 roofline** (`bound=mem`, 9 of 9). The memory-bound row predicts we recover
~13–20%, not 86%. Modelled at B-D4-256k: `min(sc, tc) = min(795.9, 322.0) = 322`, at score 0.45–0.55
that hides 145–177 µs of 1117.9 µs = **13–16%**. `modelled`, from a measured score on a different
workload.

**The confound is exactly ours and it is unresolved.** Both memory-bound arms were *shorter* than the
SC arm, as scoring is here. And the SC arm in that experiment was `2*x` streaming — pure bandwidth —
whereas the cost of *this* kernel is F, which is not bandwidth. A control-bound SC kernel may hide
behind a memory-bound neighbour far better than a streaming one does. **Nothing measures this.**

---

## 4. What area (5) forbids and requires here

**Forbids.** Any `dot_general`, `log`, `rsqrt`, `tanh`, `sqrt`, `argmax`, `transpose`, `pad`, `roll`,
`broadcast_to`, `floor`/`ceil`/`round`, or RNG on SparseCore (`exp` is the one transcendental). int8
SC scratch (`Offsets along tiled dimensions must be aligned to tiles`). Direct HBM `Get`/`Swap` from
an SC kernel — every output stages through core-local scratch. A scalar store to VMEM (`Swap only
supports scalars in SMEM`). A resident scratch above 480 KiB. `subcore_barrier` over more than 4
subcores. A TC mesh in the same `pl.kernel` as an SC mesh.

**Requires.** `needs_layout_passes=False` (already set — `scan_count`, `addupdate_scatter`,
`load_gather`, `cumsum` and `jnp.sum` all need it), and with it every `Get`/`Swap` is a `(16,)` lane
group, which is why the whole kernel is lane-group loops. Any offload measured as **added time beside
its neighbour**, three arms `tc_only` / `tc_plus_baseline` / `tc_plus_sc` — a standalone A/B on this
unit gives the wrong answer in both directions. And, from 08's algorithm results, the honest prior:
**a radix top-k on SparseCore measured 1.5–9.08x slower than `lax.top_k` standalone and still lost
1.19x beside a matmul.** The incumbent is that algorithm. Nothing in the references says it should win.

**Two things this block does that the references bless, and one they get wrong.** Blessed: it is
lock-free by construction (no `subcore_barrier`, so it is not near the hang), and it prefers an O(1)
`[lo, hi]` re-scan over a data-dependent candidate buffer — exactly 08's "re-scan beats compacting"
rule. **Wrong in the reference:** 08 states `store_compressed` "does not lower" in either flag
setting; `sparsecore_topk.py:279` calls `plsc.store_compressed(rowbuf.at[...], pos, mask=sel)` under
`needs_layout_passes=False` and the kernel measures 189.5–1012.8 µs on hardware. **The reference's
claim is falsified by this block**, and the `cumsum`+`store_scatter` workaround it prescribes is
unnecessary. That correction belongs back in `08-sparsecore.md`.

One hazard to name, not resolve: `plsc.parallel_loop` is documented as free to reorder and overlap
iterations, and "operations in one iteration cannot depend on the side effects, especially Ref
writes, of any other iteration". `emit_body` carries `woff` **and writes `rowbuf_vmem` at `woff`**;
`scan_glob` carries a prefix **and writes `glob_vmem`**. These are order-dependent Ref writes inside
`parallel_loop`. It evidently produces correct answers today; whether that is guaranteed or incidental
is `unmodelled`, and it constrains any variant that changes the unroll factor.

---

## 5. Gaps this block turns on

Each with the decision it settles and what changes under each answer.

1. **08 — "Why the memory-bound overlap collapses."** *Decides:* whether to build the scoring/selection
   overlap at all. *If bandwidth-shaped:* recovery is 13–16% at B-D4-256k and overlap is a minor lever.
   *If it is a scheduling artifact that a control-bound (not bandwidth-bound) SC kernel escapes:*
   recovery approaches the compute-bound 86% and overlap becomes the largest single lever after waves.
   *Probe:* the three-arm pattern with **the real scoring pallas_call as the neighbour**, not `2*x`.
2. **New — there is no cost term anywhere for a `plsc.parallel_loop` dispatch or a `fori_loop` wave
   iteration on SparseCore.** 08 has a per-kernel 35 µs floor and nothing per-wave. *Decides:* whether
   step 3 attacks the wave count, the per-wave loop count, or the resident slice size — these are three
   different kernels. *Probe:* an SC kernel that is `W` iterations of the empty-slice path only,
   sweeping `W` and `n_p` independently; plus `--decompose` on A-P1024-8k and A-P512-8k.
3. **08 — `subcore_barrier`'s ceiling of 4, and the 5/6/7 bisection.** *Decides:* whether one row's 16
   slices can be merged in-kernel. *If capped at 4:* the 2-stage design and both `take_along_axis`
   calls stay, and waves are the only target. *If resolved:* a 16-way in-kernel merge removes stage 2,
   both TC gathers, and 15 of every 16 candidate slots. *Do not probe this fanned out — it strands the slice.*
4. **08 — "Why XLA's gather has two implementations, and what selects between them."** *Decides:* how
   to price the stage-2 candidate fetch, which at B-P1024-256k is a 33.6 M-index gather worth **15 ms
   on the fast path and 402 ms on the slow one** — the second exceeds the entire measurement, so it is
   on the fast path, but nothing tells us how close the margin is or what would push it off.
   *Probe:* time the two `take_along_axis` calls alone at B-D4-256k and B-P1024-256k.
5. **08 — "everything here is single-subcore."** Partly answered by this block, which fans out to 32
   uncoordinated subcores and is the first real evidence for the coordination-free path at scale. What
   it shows is not encouraging for the 23.7x figure: **the fan-out is static round-robin over the
   padded slice index, so 6.25% slot occupancy at B-D4-256k.** *Decides:* whether the 23.7x scaling
   number may be quoted for a select. It may not, as spent here.
6. **05 — "no measurement of the MXU under a real Pallas kernel."** *Decides:* whether scoring's
   5.7–16x gap over its bandwidth model is the 16-row block (05's cliff table: 5.42x at 16 rows) or
   per-grid-step cost. *If the block:* packing decode rows to 128 is an area-(2) fix worth up to 8x on
   the MXU term. *If per-grid-step:* it is an area-(3) fix and the row count is irrelevant.
7. **08 — `store_compressed` "does not lower".** Falsified above. *Settles:* compaction on SC does not
   need the `cumsum`+`store_scatter` workaround, which removes a pass from any redesigned emit stage.

---

## 6. Placements enumerated and not favoured

Not a ranking — the reasons, so pass 2 and the option table can rank them.

| placement | reason not favoured (or what would change it) |
|---|---|
| **Selection on the TensorCore** (`lax.top_k`) | 08 measures the TC arm nearly flat in n (36.5→51.5 µs over 4096→65536) and beating SC by 1.5–9x — but at **k=64/128**, and k=2048 at n=1M is a different regime with no measurement. It also puts selection on the unit already doing memory-bound scoring, removing the only overlap that exists. **Cheapest probe in the exercise; do not dismiss it on the k=64 number.** |
| **Selection split across both** (TC coarse threshold, SC refines) | Attacks `M` (5–25% of SC time), not `F` (75–90%). Right idea, wrong term. |
| **Selection fused into the KV walk, one kernel** | **Forbidden.** A TC mesh beside an SC mesh is rejected at two levels; there is no fused kernel whose prologue runs on SC and whose matmul runs on the MXU. |
| **Selection on SC overlapped with the *next layer's* scoring** | The only shape 08 endorses — the offload must be independent, and within a layer it never is. Needs a cross-layer software pipeline and one extra score buffer; the memory-bound-neighbour row (gap 1) sets the payoff. |
| **One row fanned across 16 subcores with an in-kernel merge** | Capped at 4 by the barrier hang, and probing it strands a slice. |
| **Scoring on SparseCore** | No `dot_general`. Absolute refusal, no redesign available. |
| **Raise `MAX_SLICE_WORDS` to cut `p`** | Refused: n_p=131072 words = 512 KiB > the 480 KiB accepted ceiling. The obvious lever is closed by hardware. |
| **Stream the slice instead of making it resident, decoupling `p` from VMEM** | The one route to fewer waves that the 480 KiB ceiling does not close, since `p` would stop being `cdiv(n, 65536)`. Costs re-reading the row per refinement level (08: "count passes, not operations"), which multiplies `M`; `M` is currently 5–25%, so there is room to trade. Not favoured only because it is untested, not because it is priced badly. |
| **Size the select by the live context instead of `max_model_len`** | Not a unit-placement change — it is area (2)/(7) — but this area supplies its value: `W ∝ b·p` and `p ∝ n`, so 262144 of 1048576 gives p 16→4 and **W 8→2** at B decode, and 1024→32 at B-P1024. It is the largest number this sheet can put on any change. |
