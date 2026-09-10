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
**production prefill kernel, 189.98 µs** (stage C″ — the ρ = 0.40005 reweighting of the ρ = 1
microbenchmark). It is not the 326.782 µs the instrument reads, and it is not the 451.2 µs of
step 1. Where a number was measured at ρ = 1 the µs figure is quoted unchanged and only its share
moves — the reweighting never invents a measurement.

## 0. Where the search landed

- **Best version:** `v2` — the incumbent with the 16× lane-private histogram replication removed.
  It is the only node with a measurement, and it is bit-exact.
- **189.98 µs modelled-from-measured → 177.79 µs**, **−6.42%** (6.4% faster), from `R1`'s measured
  12.195 µs (`j-300d4690`, bit-identical output 1024/1024 and 64/64 rows).
- **`v1` is at 10.6% of a 20.05 µs modelled floor.** §1 carried a pessimistic 95.73 µs leg until
  `probes/10` returned; **that leg is now withdrawn** — the residual scales with buffer bytes, so it
  is not an irreducible cost of invoking the op.
- **Versions:** 8 explored, 1 built and measured, 0 rewinds (nothing has been rejected on a
  measurement yet — this is step 2, and that is the expected state).
- ⚠ **What would change the answer, and it is bigger than anything in the graph:** `probes/10` traced
  the 75.683 µs residual to an **XLA relayout of the whole 40 MiB score buffer**, inside the timed
  `jit` and **outside the Pallas kernel** — `%copy.3`, layout `{3,2,1,0:T(8,128)}` →
  `{3,1,2,0:T(8,128)}`, a full read plus a full write, ≈73 of the 75.3 µs. If `probes/11` (in flight)
  confirms it exists in the production call path, the fix is **a layout or dtype decision at the call
  site, not a kernel change** — a `carried:` handoff worth ~4× the caller's own request. If instead
  it is an artifact of the microbenchmark's `jit` signature, the **189.98 µs denominator overstates
  the production kernel by ~40%** and every share below roughly doubles. Either way it is the first
  thing to settle.

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
| **terms outside the floor** | — | | ⚠ **D3, 75.683 µs — now attributed, and it is outside the kernel** (see below). Also: SparseCore↔TensorCore handoff (D7, `unmodelled`); the `b_pad`-wide length DMA (D11, `assumed` negligible — **and `probes/10` closed the per-row axis, so it is not promoted**) |
| `v1` measured-and-reweighted | 189.98 | measured (`j-4e0df30a`) + modelled reweighting | achieved: **10.6%** |
| target | 25–33 | | 60–80% of floor |

### ⚠ D3 — 39.8% of the measured kernel — is not in the kernel at all

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

⚠ **But the denominator is now the open question, and `probes/11` (in flight) decides it.** The HLO
block is a static read of one arm — the timing fits are measured, the mechanism is `modelled`:

| if the relayout… | what D3 is | what changes |
| --- | --- | --- |
| **exists in the production call path too** | a real ~73 µs cost, **outside the kernel** | a `carried:` handoff — a layout/dtype decision at the call site, worth ~5× Plan C. No plan in §9 can reach it |
| **is an artifact of the harness's `jit` signature** | not production cost at all | ⛔ the **189.98 µs denominator overstates the production kernel by ~40%**; it becomes ~117 µs and **every share in this document roughly doubles** |

Neither leg changes the *ranking* in §3 — D3 is outside every plan either way, so it inflates all
denominators equally — and that symmetry is checked in §10. What it changes is how much any of this
is worth.

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

**Exactness.** Every `add`/`remove`/`mutate`/`precision` edge below declares it. Two of the eight
nodes are `≈ approximate` and both are marked on the node, the edge and the version block.

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
│  │  v3  fused-emit                                       ~124.2 us  modelled   16.1%   OPEN
│  │  │                                                                          -30.2%
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
│  │  v5  NB-tuned                                         unmodelled            OPEN
│  │                                                                    ^ mandatory probe, §10
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
│     v6  keep-all-fastpath                                ~181.1 us  modelled   11.1%   OPEN
│                                                                                 -4.7%
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
   v10  sample-and-count                                   ~171.6 us  modelled   11.7%   OPEN
                                                                      -9.7%
```

Node: version id, its measured or modelled time, its achieved fraction against the 20.05 µs floor,
its verdict. Edge: `[move type]` the edit · `◄ motivated:` the finding and its instrument ·
`◄ judged:` the instrument and job that decided it.

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
| V4 sweeps (`zero_priv` 9.238 · `fold_priv` 4.900 · `scan_glob` 3.498 · `find_bucket` 1.200) | 15.63 | 8.2% | no | measured (`j-4e0df30a`) |
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
- **Achieved:** 10.6% of the 20.05 µs floor — §1. ⚠ That fraction is against the **measured** 189.98 µs; if `probes/11` shows D3's relayout is a harness artifact the production kernel is ~117 µs and `v1` achieves ~17%
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
- **Live set:** 57,152 + `4·cnt_max`. At `cnt_max = 4096` (a bucket cannot hold more than the row):
  **73,536 B = 71.81 KiB, 14.03%.** Binding term: still none. ⚠ Sizing `cnt_max` is the one design
  decision here, and 4096 is the safe bound, not a tuned one
- **Predicted:** removes the refinement walk (20.08) and `emit`'s traversal, replacing them with ~1
  extra pass over `L` at L0 and ~2.231 walks of `cnt`. **~124.2 µs, −30.2%** from `v2`, 16.1% of
  floor. `modelled`
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
  more `4·L` buffer. `grid` — unchanged; still a static `fori_loop` over waves
- **Live set:** 73,536 + 40,960 = **114,496 B = 111.81 KiB, 21.84%.** Binding term: still none — and
  §4d already checked that this coexists with `v3`'s straddler
- **Predicted:** **≤ 13.469 µs**, `measured` ceiling (`j-300d4690`). **~110.7 µs, −10.8%** from `v3`.
  ⚠ **D1 and D8 cap at the same 13.469** — moving fewer bytes and overlapping the bytes you move
  cannot both be paid out of one exposed-transfer budget
- **Built?** **No** — step-3 variant
- **Measured:** — · **Verdict:** **OPEN**
- ⚠ **The risk that is specific to this node, and it is not a modelling risk.** `cvt` has a
  **negative marginal cost** — it is already absorbing the transfer's arrival latency
  (`j-300d4690`). A double buffer **competes with that mitigation**. The realistic outcome is well
  under 13.469 µs, and a measured 0 would be a coherent result rather than a broken variant. This is
  why the caller's mechanism, though real, is not the headline.

---

### `v5` — `NUM_BUCKETS` swept

- **Parent:** `v2` · **Move:** `[grid]` `NUM_BUCKETS` 256 → {64, 128, 512}, three compile-time arms
- **Motivated by:** stage C″ — `zero_priv` is a **dense clear, linear in `NUM_BUCKETS`**, and is
  9.238 of the sweep's 18.866. The constant has **never been swept** (D9)
- **Exactness:** **`exact`** — a different radix, the same exact selection
- **State diff from `v2`:** `grid` — one compile-time constant. `live set` — `4·NB` in `glob_vmem`
- **Live set:** 57,152 ± `4·ΔNB`, i.e. ±768 B across the sweep. Negligible; binding term still none
- **Predicted:** ⚠ **`unmodelled` — and that is exactly why it is `OPEN` and not `PARKED`.** The
  trade is a smaller fixed clear against more refinement levels, and the level count at production ρ
  is itself `assumed` at 2.231. Cheapest node in the graph to settle
- **Built?** **No** · **Measured:** — · **Verdict:** **OPEN**, mandatory probe (§10)
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
  (33.91/16 = 2.12) + ~2.5 count passes. ⚠ **A count pass's rate is the one `unmodelled` number the
  whole root rests on.** Modelled at 3 bundles/vector against `hist0`'s ~7.2 (both per subcore, at
  the calibrated 574 ps/bundle), a count pass is **~2.4× cheaper** than a histogram pass → 2.5 × 8.4
  = 21.0 µs. **Total ~171.6 µs, −9.7%**, 11.7% of floor. `modelled`, and the model is the thing to
  probe
- **Built?** **No** · **Measured:** — · **Verdict:** **OPEN**, mandatory probe (§10)
- ⚠ **Honest statement of where this root stands.** Its modelled prize (−9.7%) is smaller than
  `v3`'s (−30.2%), and it is *not* obviously the winner. Two things keep it in the graph anyway: it
  is the only node that attacks the largest structural lever rather than a walk around it, and its
  live set is 7.8% — it composes with everything. Two variants that were considered and are **not**
  this node, with why: a **register histogram** (16 compares per vector instead of one scatter) is
  ~71× worse per element on the measured rates and is dead on arithmetic alone; **pure bisection with
  no sample** needs ~11 dependent passes and loses to the incumbent. Both were killed by the price
  sheet, not by taste.

---

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

**Instrument runs that changed nothing:** none — every probe above moved at least one verdict, which
is itself unusual enough to note rather than take as a good sign. See §8.

**Promotions to `references/`** — platform facts, not facts about this block:

1. **`plsc.addupdate_scatter` resolves intra-vector address conflicts in hardware.** Lane-private
   replication is not required for correctness on this machine (`j-300d4690`, bit-identical).
2. **The SparseCore relayout model is not the TensorCore one.** 16 int32 lanes, no sublane axis, no
   MXU; `hardware-facts.md`'s 8×128 tile model must be scoped or it silently mis-prices every
   SparseCore design.

## 7. What `references/` got wrong

| the claim, and where | what it says | what you measured (job id) | class | what the file should say | fixed? |
| --- | --- | --- | --- | --- | --- |
| `hardware-facts.md` § Relayout | 8×128 tiles, a 128-wide lane axis, a sublane axis, and a relayout tax between them | In-job `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16, vmem_capacity_bytes=524288, dma_granule_size_bytes=32)` — **16 lanes, no sublane axis** | **out of scope** | scope the section to the TensorCore and add a SparseCore counterpart, or a one-line "does not apply to `plsc`" | **no** — flagged in `hardware_map.md`; a `references/` edit is out of this step's scope |
| `references/` HBM rates | 337 GB/s for HBM-bound streaming; no rate at all for a **VMEM-resident SparseCore pass** | `probes/03` measures VMEM-resident passes directly; the SparseCore HBM *read* rate is 2138 GB/s with a 21.9 µs intercept, and the shipping 40 KiB slice runs at 1255 GB/s | **narrower than stated** | add the VMEM-resident pass rates and say the HBM figure is asymptotic, with the intercept | **no** |
| `hardware_map.md` L6, from `probes/07` (`j-098b260f`) — an internal claim, held to the same standard | V4's 16× multiplicity costs **19.192 µs per level = 13.6% of the prefill kernel**, i.e. ~62 µs over 3.231 levels | ⚠ `R1` removes the replication entirely, bit-exactly, for **12.195 µs total** (`j-300d4690`) — **~5× less** | **wrong** | the per-level fit attributed the level's *walk* to its replication. Edges argued from it: `v2`'s prediction (measured anyway, so its verdict is unaffected) and `v7`'s original promotion (**already demoted**, on independent evidence) | ⚠ **partly** — recorded in §4b and §8; the L6 cell now carries both numbers |

**Three rows, and the third is one of ours.** A `references/` audit that only finds fault outside the
exercise has not been run honestly; the internal claim above failed the same test and is listed
under the same heading.

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

## 9. The implementation plan — what step 3 builds

Ranked. `REJECTED` nodes get no entry; `v7` is `PARKED` and gets a pointer (§5). **Each entry is
exactly one kernel call** — the boundary never splits in this block, because `P=1` leaves no
partition loop and `n` is untiled.

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

```
_topk_kernel                          src/…/deepseek_v4/sparsecore_topk.py:150
└── wave_body                         src/…/sparsecore_topk.py:~160
    ├── zero_priv(NUM_BUCKETS)     -> zero_priv(NUM_BUCKETS // LANES)      EDIT :185
    ├── scatter addr = lane*NB+b   -> addr = bucket                        EDIT
    └── fold_priv(NUM_BUCKETS//LANES) -> fold_priv over range(1)           EDIT :197
    stays with XLA: _effective_row_lengths before; the out_hbm gather after
```

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
| **Δ time %** | **−30.2%** (30% faster) standalone | |
| % of block floor | 15.1% | |

**Mechanism:** deletes two full traversals of `L`, acting on the **ρ-scaled walk** terms
(`emit` 17.7% + refinement `hist` 10.6%). Same denominator as §1. **Band:** −20% to −33%, the range
set by the level count, which is `assumed` at 2.231.

**6. Evidence status.**

| measured | modelled (formula) | unmeasured (the probe that would settle it) |
| --- | --- | --- |
| `emit` = 33.54 µs and refinement `hist` = 20.08 µs at production ρ (`j-4e0df30a`) | the replacement cost: 1 extra conditional store per element at L0, plus 2.231 walks of `cnt` | ⚠ **the cost of a conditional `store_compressed` inside the L0 walk** — it may not be free, and `probes/03` priced `emit1` at 27.456 µs, which is not negligible. **This is the probe Plan B owes** |

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

⚠ **`emit_pipeline` vs a hand-rolled `BufferedRef` is `undecided — settled by whether `pltpu`'s
pipeline helper accepts a `plsc` mesh kernel with a `fori_loop` grid.** The grid here is a manual
`fori_loop` over waves, not a Pallas `BlockSpec` grid, so the helper may not apply at all. **That is
the first thing Plan C must check, and it is a compile, not a job.**

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
| `emit_pipeline` does not accept a `plsc` mesh kernel with a `fori_loop` grid | a compile error — **catch it locally, before any job** | hand-rolled `BufferedRef` with explicit semaphores | speed |
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

- **Parallel:** Plans A, B, D, F and E's probe. **Blocked:** Plan C — on Plan B, for timing only,
  since the fused pass changes what the DMA feeds; and on its own local compile check of
  `emit_pipeline`. ⚠ **F overlaps C and D**: it deletes the score DMA and the sweeps on a quarter of
  rows, which are exactly what C and D act on. Measure F against each, never sum them.
- **Build first:** **Plan A.** It is already written (`probes/08-…/arms/arm-R1.py`), already
  bit-exact, and already measured — it is a lift into `variants/`, not a build, and it establishes
  the step-3 harness against a known answer.
- **Probe first, before Plan E is written at all:** the count-pass rate. One arm, minutes, and it
  decides whether 150 new lines get written.
- ⚠ **The one thing that outranks this whole list, and no plan here can build it:** ⏳ `probes/11`.
  `probes/10` traced D3's 75.683 µs to an XLA relayout of the score buffer **outside the Pallas
  kernel**. If it is in the production path, the fix is a layout or dtype decision at the call site
  worth ~73 µs — **more than Plans A, C and D combined** — and it leaves this file as a `carried:`
  handoff (§10), because a call-site layout change is not a kernel variant. If it is a harness
  artifact, every plan above is worth roughly double what its table says. **Settle it before step 3
  spends a job.**

## 10. Open, parked, and carried

**`OPEN` versions — none may remain when the step closes.**

| node | what closes it | cost |
| --- | --- | --- |
| `v3` fused-emit | ⚠ a probe on the cost of a conditional `store_compressed` inside the L0 walk. **Not yet dispatched** | one arm |
| `v4` pipelined-DMA | a local compile check that `emit_pipeline` accepts a `plsc` mesh kernel with a `fori_loop` grid — **no job**. ⏳ **dispatched, `probes/12-emit-pipeline-feasibility`** | minutes |
| `v5` NB-tuned | ⚠ **mandatory probe** — three compile-time arms. Its verdict currently turns on an `unmodelled` term, so §10's rule forbids parking it | one job, three arms |
| `v6` keep-all-fastpath | ✅ **check done — the gate does not exist** (no inter-subcore barrier), and the prize is 4× smaller than recorded because a `keep_all` short-circuit is already in the kernel. Now a plain build at a `modelled` −4.7% | step-3 variant |
| `v10` sample-and-count | ⚠ **mandatory probe** — the count-pass rate vs `cvt`. It is the highest-value probe in the wave because it decides a whole root | one arm |

**`PARKED` versions.**

- **`v7` shallow-scans** — parked on a **measured** term (the two carries are 4.698 µs, 2.5%), not an
  unpriced one, so the §10 rule is satisfied. **Reopens if `v5` selects a larger `NUM_BUCKETS`.**
  Route is inside the kernel → a **Not adopted** row in `final_report.md`.
- **D7, the SparseCore/TensorCore pipeline** — route is *outside* the kernel: `P=1` leaves no
  partition loop, so the overlap must be constructed in the attention op. This is a **`carried:`
  handoff**, and it owes the five things a handoff owes. Its prize is `min(t_TC, t_SC)·(p−1)/p` with
  `t_TC` **`unmodelled`** — ⚠ and by §10's own rule that means it needs a number before anyone can
  judge it. `t_TC` is readable from an existing xprof of the attention op; that is the action.
- **D11, the `b_pad`-wide length DMA** — ✅ **closed by `probes/10`, not promoted.** It was the
  per-row candidate and the per-row axis measured **−0.886 ns/row** against a 0.770 µs floor. It stays
  `assumed` negligible, but now on evidence rather than on assumption.
- ⚠ **D3's relayout, if `probes/11` confirms it** — a **`carried:` handoff**, and the largest single
  item this exercise has found. It owes the five things a handoff owes, and four are already in hand:
  *what to change* — the layout or dtype of the score tensor at the call site, so XLA has no
  `sparse-core-data-format-call` to insert; *the prize* — ~73 µs, `modelled` from HLO against a
  measured 75.349; *the correctness evidence* — none needed, a layout change computes the same
  function; *the constraint* — it is outside the Pallas kernel, so no variant in §9 can build it.
  The missing one is *the route*: the owning file and function, which is `probes/11` Q1.

**`assumed` numbers the graph depends on.**

| the assumption | where it bites | what would settle it |
| --- | --- | --- |
| refinement level count = **2.231** | `v3`'s band (−20% to −33%), `v5`'s whole trade | ⚠ **worse than `assumed` — now known to be fitted on the wrong distribution family.** Production scores are a per-head-weighted sum of **ReLU'd** fp8 dot products (`streamindex_topk.py:326-338`): right-skewed, mass piled at exactly 0.0, hundreds of exact ties, masked entries at `−inf`. `rng.standard_normal` (`test_dsv4_sparsecore_topk.py:147`) is wrong on sign, skew *and* ties, and L0 buckets on the top 8 **exponent+sign** bits (`L0_SHIFT=24`), the axis on which the two differ most. Expect the true count to be **higher**. Re-fit at the shipping distribution. ⏳ **dispatched, `probes/13-level-count-refit`** — CPU simulation, no job |
| ρ = **0.40005** prefill | **every share in this document** | ✅ **VERIFIED independently to 5 significant figures**, from `streamindex_topk.py:503-506` with `compression_ratio = 1` (`mla_attention_op.py:116`), `n = 10240` (`streamindex_topk.py:603-605`), and `max-model-len 9216` / `ISL 8192` / `RANDOM_RANGE_RATIO 0.0` in `glm-5.2-fp8-dp16-ep.sh:43,27,37`. It is **robust to the chunk mix** — ρ is uniform over `token_pos ∈ [1, 8192]` however the budget is split. ⚠ What is *not* robust is "2 of 8 chunks"; state it as **1/4 of prefill rows have `token_pos < 2048`**, which holds regardless |
| the row-comparison gate proves exactness | `v3` and `v10` | ⚠ **§8 shows it is vacuous for `monotone_key`, and the source read makes it worse.** Production rows carry **hundreds of exact ties**, and rows where every head's dot product is ≤ 0 have a maximum of **exactly 0.0** — both called out in `test_dsv4_sparsecore_topk.py:64-70`. A tie-dense row is precisely where a reordered emit can differ while a Gaussian input never would. The gate needs a **tie-dense, negative-and-zero-heavy** input and a distinct `out_hbm` sentinel |

**`unmodelled` terms still in the floor.**

| term | which version it hits hardest | asymmetry check (§10's tell) |
| --- | --- | --- |
| ⚠ **D3, 75.683 µs — no longer `unmodelled`, but not yet placed** | **all of them equally** — it is 39.8% of the denominator every prize is divided by | ✅ **checked, and it is symmetric**: no plan is credited with removing it, and both legs of §1 inflate every denominator by the same factor, so the *ranking* is invariant. What is not invariant is the worth of the whole exercise, which is why `probes/11` is first in the list below |
| **count-pass rate** | `v10` only | ⚠ **asymmetric, and this is the one to watch.** Treating it as cheap makes `v10` a −9.7% candidate; treating it as expensive kills the root. Both readings come from the same empty cell — **hence the mandatory probe above** |
| `t_TC` (D7) | the handoff, no version | symmetric — nothing is being taken on it either way |
| step homogeneity in ρ | `v6` only | ⚠ **asymmetric**: it is what turns `v6` from ≤−18.5% into 0%. Hence the mandatory check |

**What a next session should do first, in order.**

1. ⏳ **Collect `probes/11`.** It decides whether D3 is a ~73 µs handoff or a 40% overstatement of
   every denominator in this file. Nothing else here is worth more.
2. ✅ **The scheduler source read is done** — it settled `v6`, verified ρ to 5 significant figures,
   and identified the production score distribution. Both cheap follow-ons are now ⏳ **in flight and
   need no job**: `probes/12` (the `emit_pipeline`/`plsc` compile check for `v4`) and `probes/13`
   (re-fitting the refinement level count at the real ReLU'd, tie-dense, zero-piled distribution
   rather than `standard_normal`). The level count feeds `v3`'s band and `v5`'s whole trade, so
   neither is built before `13` lands.
3. **Dispatch the two mandatory probes in one job**: the count-pass rate (`v10`) and the conditional
   `store_compressed` cost (`v3`). Both are single arms and they decide two of the three largest
   candidates. ⚠ **Give both the real score distribution**, not `standard_normal` — a tie-dense input
   changes the mask densities they are measuring, and that is why this waits on `probes/13`'s
   validated generator rather than each probe rolling its own.
4. **Lift Plan A into `variants/`** — it is written, bit-exact and measured, and it validates the
   step-3 harness against a known answer before anything unknown is measured against it.
5. Then `v5`'s three-arm sweep, and Plan B.
