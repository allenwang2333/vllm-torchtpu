# `sparsecore_topk` — hardware map

Stages A–D for the **current** version. Edited in place as the version under design changes; the
version graph itself lives in `design.md`.

Source of record: `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` at commit `b11b6eb6`,
md5 `44353b347c96e0eedbd742c710d4540a`. Step 1's findings are in `report.md`; nothing here restates
a measurement, it cites the section or the probe folder.

**Sign convention** (binding in every document this exercise writes): every percentage is
`(new − old)/old` of *time*, so negative is faster. Signed percentages appear only in a `Δ time %`
column; prose states the direction in words.

**Status.** Stages A–D complete; the stage-D probe wave is running. Two probes have returned and
between them the balance sheet has been rewritten twice.

`probes/08-per-wave-residual/` (`j-300d4690-d53e-441e-9d39`) closed D4 (the launch intercept was the
instrument), cut the score DMA to 13.469 µs from a modelled 33.43, and answered D5 **yes** with a
bit-exact −3.73%. The host ρ derivation then added stage **C′**.

`probes/09-sweep-split/` (`j-4e0df30a-d833-4c1c-825d`) then **retracted three of those headlines**
and produced stage **C″**, which is the sheet in force:

- The V4 sweeps cost **18.866 µs**, not 151.324. Probe 08's `-sweep` arm deleted the sweeps *plus*
  all refinement *plus* most of emit. **D13 falls from #1 to 4th** and C′'s "42.2%" is void.
- **Super-additivity is retracted.** Regions are additive to 2.3%; the 1.625× was that one arm.
  **Divide nothing by 1.625.**
- The residual is **flat in `num_waves`** (0.009 µs/wave, measured directly). Probe 08's
  "2.561 µs/wave" was a division, and the division's premise is refuted.
- De-contaminated, **D3 is 75.683 µs = 39.8% of the production prefill kernel — the largest single
  object in it.** ✅ **`probes/10` (`j-770c6de9`) answered it: buffer bytes, no per-row term, and the
  dominant ≈73 µs is an XLA relayout of the score buffer *outside the Pallas kernel*.** So **no
  candidate in `design.md` can reach it**, and `probes/11-relayout-provenance/` is in flight to decide
  whether it is a ~73 µs handoff in the production path or a ~40% overstatement of the 189.98 µs
  denominator. ⛔ **D11 is closed by the same probe.**

⚠ **The sheet in force is stage C″. Stage C and stage C′ are both superseded — nothing below that
predates `probes/09` should be quoted without checking it against C″.**

---

## The layout vocabulary this block needs, and why it is not the usual one

`hardware-facts.md`'s relayout model — 8×128 tiles, a lane axis 128 wide, a sublane axis, a
`vtranspose` that costs ~0.8 ps/element — is a **TensorCore** model. Fifteen of the seventeen ops
below never touch the TensorCore. On the SparseCore vector subcore, as reported in-job by
`SparseCoreInfo`:

| | |
| --- | --- |
| units | 2 SparseCores × 16 vector subcores = **32 independent subcores** per TensorCore |
| vector width | **16 int32 lanes** (`LANES = 16`, `num_lanes=16`) — not 128 |
| sublane axis | **none.** A subcore's vector is one-dimensional |
| memory | **tilespmem**, 524,288 B per subcore, flat and word-addressed. No tiling, no padding to a tile |
| DMA granule | 32 B |
| matmul | **none.** No MXU, no transcendentals |

So for a SparseCore op, `layout in` / `layout out` collapse to three facts: **which axis is on the
16-lane vector**, **which axes are leading** (i.e. are loop indices, therefore free to permute), and
**which memory**. The notation used below is `axis[lane=x] SPACE`, with leading axes first. There is
no relayout term for these rows, because there is no second register axis to relayout onto — the
only movement available inside a subcore is a change of *address*, which is an ordinary load.

**That is a real simplification, not an omission, and it has a cost of its own**: everything
`hardware-facts.md` prices at zero on the leading axes is priced at zero here for a different
reason, and everything it prices as a 14× lane expansion has an SC analogue that is *not* priced
anywhere. Rows 3–5 below are exactly that analogue — the 16× lane-private replication of the
histogram — and it *was* the largest single unpriced structure in the block.
`probes/07-refinement-fixed-vs-walk/` has now priced it: **13.6% of the prefill kernel**, 9.4% inside
the refinement levels plus 4.2% in L0.

The two TensorCore rows (15, 16) do use the ordinary model. **Neither executes at either shipping
shape** (`P=1`, `report.md` §3–§4), so no TensorCore layout question is on the shipping path.

---

## Stage A — the op table

One row per op, none omitted. Rows 0–16 are `report.md` §2's inventory with the three design columns
added; rows **M**, **Z**, **S0**, **SL**, **X** and **BAR** are ops the inventory did not carry and
that this stage is the last chance to notice. Counts are **per row per wave** unless stated.

`L = slice_len` = 10240 at both shipping shapes. `W = cdiv(i_hi, 16)` = 640 for a full row.
`k = 2048`. **`FLOPs` is zero on every row in the block** — there is no floating-point arithmetic
and no matmul anywhere in it, so the column below counts integer ops instead and says so.

| # | op | `file:line` | shape in → out | dtype | layout in | layout out | int-ops (FLOPs = 0) | bytes moved | unit today | natural unit | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **M** | mesh entry / program load | `pl.kernel` + `plsc.VectorSubcoreMesh`, 512–520 | — | — | — | — | — | — | **host + SC launch** | unavoidable, but **amortizable over waves** | **22.042 µs** fixed intercept (`probes/07`, from a wave two-point on the level-free arm), agreeing to 3.2% with §5d's independent DMA-vs-waves fit of 21.349 — two unrelated arms. **43.2% of the decode kernel.** Not an op in the source, and it has to have a row or it cannot be attacked. ⚠ `probes/07`'s null arm — a jitted `x+1.0` that never enters the SparseCore — reads **27.19–27.36 µs**, *above* the intercept, so an unknown share of this is instrument-side host dispatch. Treat 22.042 as an **upper bound** on an on-device cost; separating them is blocker **D4**. ⛔ SUPERSEDED by `probes/08` (`j-300d4690`): **it is entirely instrument-side.** A trivial op's launch costs 20.193 µs against a 19.630 µs null, and one extra launch of *this* kernel at constant total work costs −0.704 µs, so on-device SparseCore program load is **zero** and the row's true cost is **0**. The suspicion recorded here — that the null reading *above* the intercept meant an unknown share was host dispatch — was right, and the share was all of it. **D4 closed; the decode kernel is 21.4 µs, not 51.0** |
| 0 | lengths DMA | 103–106 | `i32[b_pad]` HBM → VMEM | i32 | `b[lane=b] HBM` | `b[lane=b] VMEM` | — | `4·b_pad`, **once per launch** | SC DMA, blocking | SC DMA | Contiguous both sides. Every subcore loads **all** `b` lengths and reads 1 in 32 of them — see value table `V1` |
| 1 | **score DMA** | 169 | `i32[L]` HBM → VMEM | i32 | `(b)[lane=n] HBM` | `[lane=n] VMEM` | — | **`4L` = 40 KiB HBM rd** | **SC DMA, blocking** | SC DMA, **overlapped with the previous wave's compute** | The caller's target. One buffer exists, so overlap is structurally impossible today (`probes/06`, Q1). Same unit, different schedule — a scheduling candidate, not a placement one |
| 2 | `cvt` monotone_key | 171–175 | `i32[L]` → `i32[L]` in place | i32 | `[lane=n] VMEM` | `[lane=n] VMEM` | `3L` (shr, and, xor) | `8L` VMEM rd+wr | SC VPU | **fused into op 1's arrival, or into op 4** | Pure elementwise on the same axis its producer and consumer use. A whole `L` pass (20.589 µs, `probes/03`) that exists only because op 4 wants a monotone key. Nothing forces it to be a separate traversal |
| 3 | `zero_priv(256)` — L0 | 120–126, 185 | → `i32[16,256]` | i32 | — | `lane[lane=bucket] VMEM` | — | **16 KiB VMEM wr** (4096 words) | SC VPU | SC VPU | Fixed per level and **independent of `L`**. 4096 words, not 256: the histogram is replicated 16× |
| 4 | `hist0_body` | 187–196 | `i32[L]` → `i32[16,256]` | i32 | `[lane=n] VMEM` | `lane[lane=bucket] VMEM` scatter | `~6L` + `L` scatter | `4L` VMEM rd, `4L` scatter rmw | SC VPU + scatter unit | SC scatter | **Dense** mask by construction — every valid word increments. 116.350 µs, the single most expensive pass (`probes/03`) |
| 5 | `fold_priv(16)` — L0 | 128–137, 197 | `i32[16,256]` → `i32[256]` | i32 | `lane[lane=bucket] VMEM` | `[lane=bucket] VMEM` | 4096 add | **16 KiB VMEM rd**, 1 KiB wr | SC VPU | SC VPU — **or deleted**, if the 16× replication can go | Fixed per level, independent of `L`. The python `for m in range(16)` unrolls to 16 *contiguous* 16-word loads, **not** a strided read — which is why it compiles at all |
| 6 | `scan_glob(16)` | 139–146, 208 | `i32[256]` → `i32[256]` | i32 | `[lane=bucket] VMEM` | `[lane=bucket] VMEM` | 256 cumsum | 2 KiB VMEM rd+wr | SC VPU | SC VPU | **Serial**: `carry` chains all 16 chunks. 16 dependent `plsc.cumsum`. Runs unconditionally, even for `i_hi == 0` rows |
| 7 | `find_bucket(16)` | 148–154, 211 | `i32[256]` → scalar | i32 | `[lane=bucket] VMEM` | scalar reg | 256 cmp + popcount | 1 KiB VMEM rd | SC VPU | SC VPU | Serial carry again. A binary search would be 8 steps instead of 16 chunks — but each step is a scalar-indexed load, not a vector op |
| **Z** | `zg` — zero `glob` on empty rows | 199–206 | → `i32[256]` | i32 | — | `[lane=bucket] VMEM` | — | 1 KiB VMEM wr | SC VPU | SC VPU | **Missing from `report.md` §2.** Only fires when `i_hi == 0` (a padded row past `b`, or a zero-length row). Costs nothing at the shipping shapes because `b` is a multiple of 32 there — but it is the *only* work an inactive subcore does, so it sets the floor for a work-stealing design |
| **S0** | L0 scalar select | 209–221 | `i32[256]` → 7 scalars | i32 | scalar | scalar | ~20 scalar ops + 2 `vec_at` | 2 × 64 B VMEM rd | SC scalar | SC scalar | `total`, `keep_all`, `b_star`, `cum_at`, `cum_lo`, `c_hi`, `cnt`, `quota`, `prefix`, `shift`, `done`. Negligible cost, **but it is the head of the serial chain** that forbids reordering the levels |
| 8–10 | **3× refinement level** = `zero_priv` + `ref_body` + `fold_priv` + `scan_glob` + `find_bucket` | 223–265 | `i32[L]` → 7 scalars | i32 | `[lane=n] VMEM` | `lane[lane=bucket] VMEM` scatter → scalar | **fixed** 4096 wr + 4096 rd + 512 scan/cmp, **plus** `~8L` + `L` scatter — the `L` half skippable per row via `done` | 32 KiB VMEM fixed + `4L` rd per level | SC VPU + scatter | SC scatter ✅ **Measured, `probes/07`.** A whole level is **57.196 µs**, not the fitted 114.3: **19.192 µs fixed sweep** (33.6%) + **38.004 µs `L` walk** (66.4%). The walk is 1.1876 µs/wave against `probes/03`'s `scatter_sparse` at 1.1013 — a sparse scatter costs a sparse scatter's price. At ~2.231 active levels the whole region is **127.579 µs = 28.1%** of the prefill kernel, not 51.4% |
| **SL** | per-level scalar epilogue | 249–265 | `i32[256]` → 7 scalars | i32 | scalar | scalar | ~25 scalar ops + 2 `vec_at` | 2 × 64 B VMEM rd | SC scalar | SC scalar | **Missing from §2.** Cheap, but it is the loop-carried dependency: level `j+1`'s `prefix`/`shift`/`done` are computed here, so **the levels cannot be pipelined, reordered or speculated** |
| 11 | `fill_body` | 271–281 | → `i32[k]`(+`i32[k]`) | i32 | — | `[lane=k] VMEM` | — | `4k`(+`4k`) VMEM wr, **only when `total < k`** | SC VPU | SC VPU | Fires only on short rows. At `n=10240, k=2048` with full rows it never runs at the shipping shapes |
| 12 | `emit_body` | 283–311 | `i32[L]` → `i32[k]`(+`i32[k]`) | i32 | `[lane=n] VMEM` | `[lane=k] VMEM` compressed | `~7L` + cumsum + popcount | `4L` VMEM rd, `4k` compressed wr | SC VPU | SC VPU | 27.456 µs (`probes/03`). **Serial**: `(woff, rank)` carry chains all `W` chunks, unlike ops 4 and 8–10 which are fully parallel. Also the **only** hot loop with register spills (`probes/06` Q4: 9 slots in 26 bundles) |
| 13 | out DMA | 315–319 | `i32[k]` VMEM → HBM | i32 | `[lane=k] VMEM` | `(b)[lane=k] HBM` | — | `4k` = 8 KiB HBM wr | SC DMA, blocking | SC DMA | Adjacent wait, zero gap (`probes/06` Q1) — already free |
| 14 | keys DMA (`emit_keys` only) | 320–325 | `i32[k]` VMEM → HBM | i32 | `[lane=k] VMEM` | `(b)[lane=k] HBM` | — | `4k` = 8 KiB HBM wr | SC DMA, blocking | SC DMA | `emit_keys` costs +4.38% end to end for its alloca, DMA pair and two `vst` (`probes/06`, same-job control) |
| **BAR** | `sc_tpu.sfence` + `sc_tpu.barrier` | **not in the source** — Mosaic emits it, dump lines 904/907 | — | — | — | — | — | — | SC sync | SC sync | **Missing from §2's table** (it is in §2's prose). Every subcore's scope ends at the same instant, so the kernel costs **`max` over 32 subcores**, and the per-`TEC` durations in a trace read as perfect balance when they are not |
| **X** | `bitcast` f32→i32, and the `P>1` reshapes | 560, 580–592 | `f32[b,n]` → `i32[b,n]` | f32→i32 | leading | leading | — | 0 | XLA, **free** | free | Bit reinterpretation and contiguous reshapes on leading axes. Priced at zero and it genuinely is; recorded so it is not rediscovered as a candidate |
| 15 | offset-add + `where` | 594–598 | `i32[b,P,k_p]` → `i32[b,P·k_p]` | i32 | `(b,P)[sub=·, lane=k_p] HBM` | `(b)[sub=·, lane=P·k_p] HBM` | `b·P·k_p` | 2 × 33.55 MB HBM | **TensorCore (XLA)** | TensorCore | **`P>1` only — does not execute at either shipping shape.** Leading-axis reshape plus an elementwise add: free layout, pure HBM cost |
| 16 | `take_along_axis` | 617–619 | `i32[b,P·k_p]`, `i32[b,k]` → `i32[b,k]` | i32 | `(b)[sub=·, lane=P·k_p] HBM` | `(b)[sub=·, lane=k] HBM` | gather | 33.55 + 8.39 MB HBM | **TensorCore (XLA gather)** | ⚠ **SparseCore gather** — the unit exists (`sc_dsa_gather` uses it) | **`P>1` only.** The one row where `unit today` and `natural unit` genuinely differ. Not on the shipping path, so it is recorded and not pursued — see `design.md` §10 |

### What the two unit columns disagree about

Per the stage-A rule, the rows where `unit today ≠ natural unit` are the cheapest source of stage-B
moves. There are **four**, and only one of them is a placement move in the usual sense:

| row | disagreement | kind |
| --- | --- | --- |
| **M** | fixed launch cost is charged once per kernel; it *could* be amortized over more waves | **amortization**, not placement — needs a probe separating on-device program load from host dispatch (`report.md` §9 candidate 2) |
| **1** | same unit, different **schedule**: overlapped instead of blocking | **the caller's candidate** |
| **2** | a standalone `L` pass that could be folded into the DMA arrival or into op 4 | **fusion** |
| **5** | `fold_priv` exists only to undo op 3's 16× replication; both could be deleted together | **algebra** — this is the `mutate` edge the design graph is required to contain |
| 16 | TensorCore gather → SparseCore gather | placement, but **`P>1` only, so off the shipping path** |

Every other row's natural unit is its current unit, because the SparseCore is the only unit in the
block that can do integer scatter, `store_compressed` or population count at all.

---

## Stage A — the value table

One row per value crossing between ops. `where it lives` is a decision, not an observation.

**`mismatch` is `none` on every SparseCore row and that is a structural fact, not a lucky one**:
there is one register axis, so a producer and a consumer either agree about which axis is on it or
one of them is indexing wrong. What replaces relayout as this block's hidden cost is **multiplicity**
(`V4`) and **re-traversal** (`V3`), and both have rows.

| # | value | produced by | consumed by | axis produced | axis needed | mismatch | resolution | where it lives | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **V1** | `row_lengths` | caller (`_effective_row_lengths`, outside the block) | op 0 → `S0` via `vec_at` | `[lane=b] HBM` | scalar read at `row_c` | none | — | HBM → **VMEM, replicated on all 32 subcores** | `4·b_pad` per subcore per launch. **Each subcore loads all `b` and reads `b/32` of them** — 16 KiB for 2 useful reads at decode. `assumed` negligible against 21.3 µs; ⚠ never measured |
| **V2** | `scores` (raw fp32 bits) | upstream DSA indexer (outside the block) | op 1 | `(b)[lane=n] HBM`, row-major contiguous | contiguous `L`-word span | none | — | HBM | 40 KiB per row per wave. **1 HBM pass** against 5.03–5.50 VMEM passes (`report.md` §5) |
| **V3** | **`resident_vmem` — the monotone keys** | op 2 (**in place over V2**) | ops 4, 8–10 (×2.03–2.50), 12 — **five consumers** | `[lane=n] VMEM` | `[lane=n] VMEM` | none | — | **VMEM, 40 KiB, one buffer** | ⛔ **The structural obstacle to the caller's mechanism.** A streaming pipeline retires a tile after one consumption; this tile is consumed five times and the consumers are separated by serial scalar reductions. `probes/06` Q5: a second buffer → 110.50 KiB of 512 KiB, **it fits** |
| **V4** | **`priv_vmem` — 16 lane-private histograms** | ops 3 + 4, and 8–10 per level | op 5 (`fold_priv`), per level | `lane[lane=bucket] VMEM`, address `lane*256 + bucket` | `lane[lane=bucket] VMEM`, 16 contiguous 16-word loads per output chunk | none | — | VMEM, **16 KiB** | **Stored at 16× the multiplicity of the value it represents** (`CONTRACT` L6). It buys conflict-free scatter and it costs a **4096-word zero + a 4096-word read, per level, independent of `L`** — 32 KiB of traffic that no term in `report.md` §5d priced. ✅ **Measured (`probes/07`): 19.192 µs per level, 13.6% of the prefill kernel** (4.2% L0 + 9.4% refinement). At decode it is 45.5% of a level rather than 33.6%, because the `L` walk shrinks with waves and the sweep does not |
| **V5** | `glob_vmem` — the folded 256-bucket histogram, then its prefix sum | op 5 (or **Z**) → op 6 in place | ops 6, 7, `S0`/`SL` | `[lane=bucket] VMEM` | `[lane=bucket] VMEM` | none | — | VMEM, 1 KiB | Small, but **serial in 16 steps twice** (scan then search). Cannot be widened: `LANES` is 16 |
| **V6** | **the scalar state** — `prefix`, `shift`, `quota`, `cnt`, `c_hi`, `done`, `keep_all`, `my_take` | `S0`, then `SL` per level | every subsequent level, then op 12 | scalar | scalar | none | — | **scalar registers / SMEM** | Free to compute. **Not free structurally**: this is a loop-carried dependency across levels *and* into emit, so no level may be reordered, speculated or overlapped with another. Any design that walks `L` fewer times must preserve this chain or replace it |
| **V7** | `rowbuf_vmem` — the compacted winning indices | op 11 (short rows) + op 12 | op 13 | `[lane=k] VMEM`, written at a **data-dependent** offset `woff` | `[lane=k] VMEM` contiguous | none | — | VMEM, 8 KiB (`k+16`) | The `(woff, rank)` carry makes op 12 serial across all `W` chunks. The `+16` slack is the compaction overrun guard |
| **V8** | `keybuf_vmem` — the winning scores, recovered by re-applying the involution | op 11 + op 12 (`emit_keys` only) | op 14 | `[lane=k] VMEM` | `[lane=k] VMEM` | none | — | VMEM, 8 KiB | Same mask and same `woff` as V7, so the two stay aligned slot for slot. **+4.38%** measured for the whole `emit_keys` arm (`probes/06`) |
| **V9** | `local_slots` / final `indices` | op 13 | the caller's `sc_dsa_gather` (outside the block) | `(b)[lane=k] HBM` | — | none | — | HBM, 8 KiB per row | **At `P=1` this is the block's output.** No further op inside the block touches it |
| **V10** | `cand_scores` | op 14 | op 15/stage 2 | `(b,P)[lane=k_p] HBM` | `(b)[lane=P·k_p] HBM` | none — leading-axis reshape | free | HBM, 33.55 MB | **`P>1` only** |
| **V11** | `global_cand_indices` | op 15 | op 16 | `(b)[sub=·, lane=P·k_p] HBM` | gather source | none | — | HBM, 33.55 MB | **`P>1` only.** The gather source is re-read whole to produce 8.39 MB — a 4:1 read amplification, and the reason row 16's natural unit is questioned |

### Where the SparseCore's version of the relayout tax actually is

`hardware-facts.md`'s expensive case — a lane↔sublane swap on a value produced and consumed by
arithmetic — **cannot occur in this block**. What replaces it is **V4**: a value stored at 16× its
logical multiplicity so that a scatter never has an intra-vector address conflict. Like a relayout it
has no FLOP term and no HBM term, so **step 1's floor scores it at zero**; unlike a relayout it does
not appear folded into a neighbouring op in a profile — it appears as `zero_priv` and `fold_priv`,
which look like negligible 256-iteration loops and are 4096 words each.

`probes/07` measured it. **The V4 sweeps are 19.192 µs per level** — 4.2% of the prefill kernel in L0
and 9.4% across the ~2.231 active refinement levels, **13.6% in total**, which makes deleting the
replication the third-largest single move available and the one the design graph's required `mutate`
edge is built on.

**But V4 was not where the missing ~150 µs went, and the way that error was made is the transferable
part.** `report.md` §5d had one free parameter and one equation, so *everything* `probes/03` had not
priced was absorbed into the refinement term — the V4 sweeps (43 µs, right in kind) and ~81 µs of
per-wave base cost that has nothing to do with refinement at all. The fit then validated out of
sample to −0.14% by extrapolating 32 waves → 2 waves, and **both confounded terms scale with waves**,
so the check ran along the one axis that could not separate them. That ~81 µs is now stage D's
blocker **D3** and is larger than the score DMA.

---

## Stage A — every axis of the iteration space

Per regime, because the two shipping regimes differ by 16× in `b` and the axes are not the same in
each. Both shipping regimes are `n = 10240`, `k = 2048`, **`P = 1`** (`report.md` §4).

| axis | extent, prefill `b=1024` | extent, decode `b=64` | tiled? | decision, and why |
| --- | --- | --- | --- | --- |
| `b` — score rows (query tokens) | 1024 | 64 | **yes**, 32 per wave | One row per subcore per wave. The only axis the kernel parallelizes over |
| `wave` `w` | **32** | **2** | derived, `cdiv(b,32)` | The serial axis. **This is the axis the caller's mechanism amortizes over**, which is why its ceiling is −7.18% at 32 waves and −2.02% at 2 (`report.md` §6). ⛔ SUPERSEDED by `probes/08` (`j-300d4690`): **−4.12% / −1.97%** |
| `subcore` `flat = core*16 + sub` | 32 | 32 | **yes**, the mesh | Fully occupied in both regimes. **Rows per subcore is 32 vs 2**, which is where the 1.28× decode imbalance comes from (`probes/05`) — the imbalance is an artifact of this axis being short, not of the work |
| `core` | 2 | 2 | folded into `flat` | 2 SparseCores per TensorCore, no inter-core communication anywhere in the file |
| `n` / `L` — columns within a row | **10240, not tiled** | 10240, not tiled | **NO** | **The load-bearing decision of the whole design.** The entire row is made resident (40 KiB of 512 KiB) so that five passes can run over VMEM. Untiling `n` is what makes the HBM:VMEM pass ratio 1:5 instead of 5:5 |
| `chunk` `i` | `W = 640` full | `W = 640` full | yes, 16 words | `unroll=4` in ops 2, 4, 8–10, 12; `unroll=2` in 5, 6, 7 |
| `lane` | 16 | 16 | the SIMD axis | Fixed by the hardware. Not a parameter |
| `bucket` | **256, not tiled** | 256, not tiled | **no** | 8 bits per level at `NUM_BUCKETS=256`. Fewer buckets → cheaper fixed sweep, more levels; more buckets → the reverse. **A free parameter nothing in the block has ever swept**, and stage D owes it a row |
| `bucket-replica` (`lane_iota`) | **16** | 16 | no | The V4 multiplicity. **An axis that exists only to avoid scatter conflicts** and appears nowhere in the algorithm |
| `level` | **≤3, ~2.03 taken** | ≤3, **~2.50 taken** on the critical subcore | **no, and cannot be** | Serial and data-dependent through V6. `heavy_ties` pins it at 3.00 (`probes/05`); the distribution is `assumed` |
| `pass` over `L` | ~5.03 | ~5.50 | derived | `cvt + hist0 + levels + emit`. Not an independent axis; recorded because it is the number the roofline is built on |
| **`P` — partition** | **1 (degenerate)** | **1 (degenerate)** | no | ⚠ **The currently-degenerate axis.** It is 2 at `b=16, n=10240` and 4 at `n=133120`, and *all* prior work on this kernel was measured at `P=4`. Every fixed cost — M, V1, V4's sweeps — lands hardest here, because at `P=1` the step has nothing else in it |
| **`stage`** | **1 (degenerate)** | 1 (degenerate) | no | At `P=1` there is no stage 2, no offset add, no `take_along_axis`. **The block is one kernel and 100% SparseCore** |
| **`layer`** — the 21 of 78 GLM-5.2 layers that call this | 21 | 21 | **no, and it is not in the kernel at all** | The axis that exists in the data but not in the loop. Layers are serial in the model, so they cannot be batched — but this is why **M** is paid 21× per token step and is not merely a per-call constant |

Three axes above exist for the three reasons the stage-A rule names: `P` and `stage` are
**degenerate in the regime being designed for**, `layer` **exists in the data and not in the loop**,
and `bucket-replica` **exists only once the implementation is separated from the algorithm**. None
of them is in `report.md` §2.

---

## Stage A — `CONTRACT.md` §A

⛔ **Moved.** The filled L1–L7 table lives in `design.md` §4d, which is now its only copy —
the draft that was drafted here beside the value table has been deleted so the two cannot
drift. §B, §C and §D are filled at stage D below and are also transcribed into `design.md` §4d.

⚠ One correction the move carries: L6 on V4 was priced here at **19.192 µs per level** from
`probes/07`. `R1` later removed the replication outright, bit-exactly, for **12.195 µs in
total** (`j-300d4690`) — **~5× less**. The per-level fit attributed the level's walk to its
replication. `design.md` §7 records it as a failed internal claim.

---

## Stage B — placement

Every number below is the prefill shape, `b=1024, n=10240, k=2048, P=1`, total **454.602 µs**
(`probes/07`, `j-098b260f-a9f7-4fe5-937c`). The decode column is `b=64`, total **51.027 µs**.

### Where the time is, by unit

| unit | ops | prefill µs | % | decode µs | % |
| --- | --- | --- | --- | --- | --- |
| **SC tilespmem scatter port** (`vst.idx.add.s32.msk`) | 4, and the `L` half of 8–10 | **201.12** | **44.24%** | 5.41 + share of base | ~11% + |
| **SC vector unit** (load / ALU / store / compress) | 2, 3, 5, 6, 7, 12, and the fixed half of 8–10 | 110.05 | 24.21% | 4.51 + share of base | ~8.8% + |
| **SC DMA engine** | 0, 1, 13 | 40.12 | 8.83% | ~1.3 | ~2.5% |
| **host + SC launch** (M) | — | 22.04 | 4.85% | **22.04** | **43.20%** |
| ⚠ **unassigned** | — | **81.28** | **17.88%** | ~5.1 | ~10% |
| **MXU** | — | **0** | **0%** | 0 | 0% |
| **TensorCore VPU / XLU** | — | **0** | **0%** | 0 | 0% |
| **SC scalar** | S0, SL | ≈0 | ≈0% | ≈0 | ≈0% |

Rows 15 and 16 are the only TensorCore ops in the inventory and **neither executes at either
shipping shape** (`P=1`). So the placement question "which op is on the wrong unit" has, on the
shipping path, exactly one answer: **none of them**. The SparseCore is the only unit in the machine
that can do integer scatter, `store_compressed` or population count, and everything in the block is
one of those three or feeds one.

**That makes the interesting placement fact the empty rows, not the full ones.** The MXU and the
whole TensorCore are idle for the entire duration of the block, and `report.md` §1 measured the block
as 99.8% unoverlapped with its neighbours at prefill and 94.7–95.1% at decode. That is `seams.md`
**S6**, and it is stage D's blocker **D7** — with the caveat S6 itself insists on, which is that this
block has no partition loop to reorder (see D7).

### Ops whose unit is not in dispute but whose *count* is

Placement on this block is degenerate; the moves are all about how many times a unit is entered.
Restating stage A's four disagreements as placement questions:

| row | the move | is it placement? |
| --- | --- | --- |
| **M** | amortize a fixed 22.04 µs over more waves | no — it is a launch-count question, and D4 has to separate on-device load from host dispatch before it is even sizeable |
| **1** | overlap the score DMA with the previous wave's compute | no — same unit, different **schedule**. The caller's candidate |
| **2** | fold `cvt` into `hist0`'s traversal | no — **fusion**, and it is now a stage-C consequence rather than a guess (see "the traversals are at the port") |
| **5** | delete `fold_priv` and its 16× replica along with it | no — **algebra**. The `mutate` edge, gated by D5 |

---

## Stage C — the balance sheet, measured

### The calibration, and it is the load-bearing result of this stage

`probes/06` Q4 gives static VLIW bundle counts per 64 elements (unroll-4 × 16 lanes) for three of
the traversals. `probes/07` gives their measured times. Dividing one by the other gives an issue
rate, and the rate is then a *prediction* for every other row:

`cvt` is 7 bundles / 64 elements. At `b=1024` each subcore runs 32 rows × 640 chunks × 1.75
bundles = 35,840 bundles, in a measured 20.589 µs → **574 ps per bundle, i.e. one bundle per cycle at
~1.74 GHz**. `cvt` is therefore **issue-bound and at the subcore's structural peak** — it is the
calibration, not a prediction, and it is quoted as such below.

| region | bundles / 64 el | modelled µs | measured µs | ratio | what the ratio says |
| --- | --- | --- | --- | --- | --- |
| `cvt` (op 2) | **7** | *calibration* | 20.589 | — | one bundle per cycle; nothing to recover |
| refinement `L` walk (8–10) | **14** | 41.14 | **38.004** | **0.92** | **issue-bound**, 8% inside the calibration |
| `hist0` (op 4) | **9** | 26.45 | **116.350** | ⚠ **4.40** | **not issue-bound.** The excess is the *dense* `vst.idx.add.s32.msk` read-modify-write stalling on tilespmem |
| V4 fixed sweep + scans (3, 5, 6, 7) | ~1/vector | 10.29 | **19.192** | **1.87** | **latency-bound, not throughput-bound** — three serial carries (D10, D13) |
| `emit` (op 12) | `unmodelled` | — | 27.456 | — | implies ~9.3 bundles / 64 el if issue-bound; consistent, unverified |
| score DMA (op 1) | — | **19.62** (41.94 MB ÷ 2138 GB/s) | ⛔ **33.43 → 13.469** | ⛔ **0.69** | ⛔ SUPERSEDED by `probes/08` (`j-300d4690`). The *exposed* cost is 0.40× the bandwidth model — 3114 GB/s, above the 2138 figure, which was itself a fit carrying a 21.9 µs intercept. Much of the latency is already absorbed by `cvt` |
| out DMA (op 13) | — | 3.92 | *modelled* 6.69 | — | both sides modelled; excluded from the discrepancy |
| launch (M) | — | `unmodelled` | ⛔ **22.042 → 0** | — | ⛔ SUPERSEDED by `probes/08` (`j-300d4690`): it was the harness's per-call dispatch fence. D4 **closed** |
| ⚠ **unassigned** | — | **0** | **81.28** | **∞** | the largest unexplained term in the block — D3 |

**Model total 245.1 µs against a measured 454.602: the balance sheet as written explains 54%.**

> ⛔ SUPERSEDED by `probes/08` (`j-300d4690`). Two of the four rows above moved: the score DMA to 13.469 and launch to 0. The
> calibration itself — 574 ps/bundle, `cvt` at the issue port — is untouched and still governs.
> The corrected sheet is below, and stage C′ then reweights it to the shipping ρ.
Two rows account for almost all of the gap, `hist0`'s 90 µs of scatter stall and the 81 µs that has
no row at all, and both become stage-D blockers.

### Three consequences that reorder the candidate list

1. **The traversals are at the port. You cannot make an `L` walk faster; you can only do fewer of
   them.** `cvt` runs at one bundle per cycle and the refinement walk runs within 8% of its bundle
   count. Every candidate of the form "tighten the inner loop", "hoist the invariants", "improve the
   schedule" is dead on arrival — `probes/06` Q6 reached the same verdict from the other direction
   and priced LICM at ~zero. Every candidate of the form "traverse `L` once instead of five times"
   is live, and D6 is the largest of them.
2. **`hist0` is the one row where the pace-setter is memory** — 4.40× its issue count, against the
   refinement walk's 0.92× from *bundle-identical* scatter code. The discriminator is mask density,
   exactly as `probes/03` priced it at 0.30× per masked-off lane. It cannot be tuned; it has to be
   replaced — a second root, D2. ⛔ **"and that is why it is the biggest" is retracted**: it is
   11.5% at production ρ, third. It was the biggest only at the instrument's ρ = 1.
3. **The V4 sweeps are latency, not bytes** — 1.87× their issue count. ⛔ **The rest of this item is
   RETRACTED by `probes/09`.** The four sweep loops cost **18.866 µs**, not 151.324; probe 08's arm
   also deleted all refinement and most of emit. And the split runs the other way: **`zero_priv`, a
   dense 256-bucket clear, is 9.238 of the 18.866**, while the two carried scans are 4.698 together.
   So the sweep is **not** dominated by chain depth, `R1`'s 12.2 µs recovery is now *most* of the
   sweep rather than 8% of it, and **D5 and D13 are the same saving**. At production ρ the sweeps are
   **8.2%** of the kernel, not 42.2%.

### Full prefill balance sheet at ρ = 1 — 434.97 µs of kernel

⛔ **The 454.602 µs total is superseded and so is every row of the sheet that used to sit here.**
`probes/08` (`j-300d4690-d53e-441e-9d39`) measured the decomposition directly instead of inferring it
from independent slopes, and three rows moved by more than 2×. 454.602 was the *wall* figure; the
harness's own null floor is **19.630 µs** and comes out first, leaving **434.97 µs of kernel**.

The L0 base decomposes **exactly**, which is the check that the decomposition is closed:

| | µs | note |
| --- | --- | --- |
| `B` — L0 base, `REFINEMENT_LEVELS = ()` | 327.166 | |
| − harness null | 19.630 | two nulls 0.529 apart |
| **= L0 kernel** | **307.54** | |
| of which: joint removable five, `B` − `-alld` | **225.58** | **exact** |
| of which: residual, `-alld` − null | ⛔ **81.95 → 75.683** (`probes/09` `A0`; 9.261 was `fill`+outdma) | D3 |
| 225.58 + 81.95 | 307.53 | ⛔ closed to 0.003% **on a contaminated control**. The de-contaminated form is in stage C″, which closes to 1.62% |

| term | single-arm Δ (**upper bound**) | scales with ρ? | tag |
| --- | --- | --- | --- |
| V4 sweep — `zero_priv`+`fold_priv`+`scan_glob`+`find_bucket` | ⛔ **151.324 → 18.866** (`probes/09`; this arm also deleted all refinement and most of emit) | **no** | superseded |
| `hist0` dense scatter | **138.306** | yes | measured, `probes/08` |
| ⚠ residual, belongs to no op | ⛔ **81.95 → 75.683** | no — and ⛔ **not per wave either** (0.009 µs/wave) | superseded by `probes/09` — D3 |
| `emit` | 84.020 | yes | measured, `probes/08` |
| refinement `L` walk × 2.231 | 84.79 | yes | measured, `j-098b260f` |
| refinement V4 sweep × 2.231 | 42.81 | **no** | measured, `j-098b260f` |
| score DMA | **13.469** | **no** (D8) | measured, `probes/08` |
| `cvt` | **−20.492** | yes | measured, `probes/08` — *negative*, see below |
| launch | ⛔ **0** | — | **closed**; it was the instrument |

### ⚠ Three corrections that govern every candidate from here

1. ⛔ **RETRACTED — the regions are additive to 2.3%, not super-additive by 1.625×.** The 366.628
   sum contained one arm, `-sweep`, that deleted three regions rather than one. Substituting
   `probes/09`'s clean 18.866 gives a sum of **233.979** against a de-contaminated joint of
   **228.679** — ratio **1.023**. `emit` independently reproduces across two jobs to 0.23%, so it
   was never super-additive either. **Divide nothing by 1.625.** What survives is the procedural
   rule — *always measure the joint arm* — and a sharper reading of the mechanism: the over-count
   was **arm overlap**, not resource contention. When a bound you zero feeds a downstream scalar
   (`total` → `keep_all` → the refinement loop, and `b_star` → emit's mask density), the arm deletes
   far more than its name says.
2. **`cvt` has a *negative* marginal cost — deleting a whole 640-chunk traversal made the kernel 20.5
   µs slower.** `cvt` is the first consumer of `resident_vmem` after the score DMA and is currently
   absorbing the transfer's arrival latency while doing its own useful work. This is why `-dma` reads
   only 13.469 against a 33.43 µs bandwidth model.
3. **The score DMA is 13.469 µs, not 33.43.** 41.94 MB in 13.469 µs is 3114 GB/s, above
   `sparsecore-hbm-read-bandwidth-2138-gbs` — that figure was a fit with a 21.9 µs intercept and does
   not govern this access pattern. **The caller's ceiling falls from −7.18% to −4.12% at prefill.**

**Decode at ρ = 1, corrected:** `B` = 41.005, null 19.630, so the **kernel is 21.4 µs, not 51.0**.
⛔ Stage C's "at two waves the kernel is more launch than kernel" is **retracted** — D4 was the
instrument's per-call dispatch fence, and it does not exist in production.

---

## Stage C′ — the production-ρ reweighting, and it inverts the ranking

**Every probe in this exercise ran `row_lengths = jnp.full((b,), n)` — ρ = 1.0.** The host derivation
of the shipping GLM-5.2 config (`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh`, ISL:OSL
8192:1024, `max-model-len 9216`, `max-num-batched-tokens 1024`, `max-num-seqs 64` per DP engine) gives:

| phase | ρ = E[eff_len]/n | derivation |
| --- | --- | --- |
| prefill, row-average over the 8 chunks | **0.40005** | ρ_c = 0.1·c + 0.05005, c = 0…7 |
| decode | **0.850** | seq_len ∈ [8193, 9215], n = 10240 |
| run-integrated, all rows | **0.450** | prefill is 88.9% of all kernel rows |

ρ is a **row average**, so it survives the scheduler interleaving prefill and decode rows in one step
— which matters, because subcore `flat` owns rows `flat, flat+32, …`, a stride that hands every
subcore a representative sample, so `max over subcores ≈ mean` whatever the mix.

**Two structural consequences, both from the source and neither from a timing:**

- **`walk_chunks = _cdiv_dyn(i_hi)` with `i_hi = min(slice_len, eff_len)`** — so `cvt`, `hist0`, the
  refinement walks and `emit` all scale with ρ. **`dma(scores_hbm.at[pl.ds(s, slice_len)], …)` is
  static** — the transfer does not. That is D8, now quantified: **at prefill, 60% of the bytes moved
  are never read.**
- **`keep_all = total <= k` and `total` ≈ eff_len, so keep_all ⟺ ρ ≤ 0.2.** Prefill chunks **0 and 1
  run zero refinement levels**, by construction, for every row. The refinement terms therefore carry
  an extra 6/8 on top of ρ.

### ⛔ SUPERSEDED — Prefill balance sheet at production ρ = 0.40005 — ~297 µs

⛔ **This sheet is void. `probes/09` (`j-4e0df30a`) retracted both of its two largest rows.** The V4
sweeps are 18.866 µs, not 151.3, so "42.2%" is wrong by 5×; and the 1.625 divisor applied below does
not exist. **Use stage C″.** The ρ *method* below is correct and carries forward unchanged — only the
ρ=1 inputs it was fed were wrong.

Walk-bound rows scaled by ρ, refinement additionally by 6/8, then the whole removable block divided
by the 1.625 super-additivity factor:

| term | µs | **%** | vs its ρ=1 share |
| --- | --- | --- | --- |
| **V4 sweep, L0** | **93.12** | **31.4%** | ▲ |
| ⚠ **residual (D3)** | **81.95** | **27.6%** | ▲ |
| `hist0` | 34.05 | 11.5% | ▼▼ |
| **V4 sweep, refinement** ×1.673 | 32.12 | 10.8% | ▲ |
| refinement `L` walk ×0.837 | 31.80 | 10.7% | ▼ |
| `emit` | 20.68 | 7.0% | ▼ |
| score DMA | 8.29 | **2.8%** | ▼ |
| `cvt` | −5.05 | −1.7% | |
| **total** | **296.96** | 100% | |

**The two fixed V4 sweeps together are 42.2% of the production prefill kernel — the largest object in
it by a factor of 1.5.** They were 13.6% when stage D was written. They rose because they are the one
big block that does *not* scale with ρ, while `hist0` — stage C's "single most expensive pass" — fell
from 25.6% to 11.5%. **`addupdate_scatter` is no longer 44.2% of the kernel; it is 22.2%.**

⚠ **And `R1` recovers only 12.2 µs of the L0 sweep's 151.3** (`probes/08`, bit-exact). So ~92% of the
sweep is **not** the 16× replication — it is `fold_priv`'s serial accumulate and the two 16-step
carried scans, i.e. **D13, at ~135 ns per carried iteration = 235 cycles for a body of a handful of
bundles.** The split between the three is `probes/09`.

**Tags.** ρ itself is `derived` (`file:line` for every input). The 6/8 refinement factor is `derived`.
The *level count* within chunks 2–7 is **`assumed`** at 2.231, carried from ρ=1; `probes/08` refuted
the "a skipped level costs 3.8 µs" explanation of probes/05's 2.231-vs-2.03 gap (a skipped level costs
**0.060 µs**), so that gap is a score-distribution mismatch and any candidate priced on the level
count must re-derive it. The 1.625 factor is measured at ρ=1 and **assumed** to hold at ρ=0.4.

---

## Stage C″ — the sheet in force

`probes/09` (`j-4e0df30a`) re-measured the two regions C and C′ had wrong. Full derivation and every
caveat: `probes/09-sweep-split/README.md`. **Cross-job control passes** — `B` reads 326.782 here
against probe 08's 327.166 (−0.12%) from a byte-identical source.

### At ρ = 1, b = 1024 — the ledger closes to 1.62%

| term | µs | % of `B` | tag |
| --- | ---: | ---: | --- |
| `hist`, all levels | 138.306 | 42.3% | measured (`probes/08` `-hist`) |
| `emit` | 83.830 | 25.7% | **measured** (`probes/09`, `B − E0`) |
| **residual (D3)** | **75.683** | **23.2%** | **measured** (`probes/09`, `A0 − null`) |
| V4 sweeps, all levels | 18.866 | 5.8% | **measured** (`probes/09`, `E0 − E0-swp`) |
| harness fence (null) | 16.548 | 5.1% | measured — instrument, not kernel |
| score DMA | 13.469 | 4.1% | measured (`probes/08` `-dma`) |
| output DMA | 5.872 | 1.8% | **measured** (`probes/09`, `B − -out`) |
| `cvt` | −20.492 | −6.3% | measured (`probes/08`) — negative, and that is real |
| **sum** | **332.082** | | |
| **measured `B`** | **326.782** | | closes to **+1.62%** |

**The regions are additive to 2.3%** (singles 233.979 vs the de-contaminated joint 228.679), and
`emit` reproduces across two jobs to 0.23%. **No prize is divided by 1.625.**

The sweep splits four ways, additive to 0.2%: `zero_priv` **9.238** · `fold_priv` 4.900 ·
`scan_glob` 3.498 · `find_bucket` 1.200. Half the sweep is a **dense 256-bucket clear**, not a
carried scan — the opposite of what D10/D13 assumed.

One split is available without another job: probe 08's `-sweep` = 175.842 leaves
`hist0 + emit_degenerate + fill_spurious = 84.762`, so **hist0 ≤ 84.76** and refinement hists
= 53.54. Stage C's independent per-pass derivation put hist0 at 85.12 — **0.4% apart**. Both rows
are `modelled`, corroborated but not measured.

### At production prefill ρ = 0.40005 — 189.98 µs

Same method as C′: walk-bounded terms × ρ; refinement terms additionally × 6/8 with a conditional
mean ρ of 0.50005 (`keep_all` fires for all of chunks 0–1); fixed terms unscaled.

| term | µs | **%** | scales with ρ? |
| --- | ---: | ---: | --- |
| ⚠ **residual (D3)** | **75.683** | **39.8%** | **no** |
| `hist0` | 33.91 | 17.8% | yes |
| `emit` | 33.54 | 17.7% | yes |
| refinement `hist` | 20.08 | 10.6% | yes, and gated by `keep_all` |
| V4 sweeps | 15.63 | 8.2% | no, but gated |
| score DMA | 13.469 | 7.1% | **no — static `slice_len`** |
| output DMA | 5.872 | 3.1% | no |
| `cvt` | −8.20 | −4.3% | yes |
| **total** | **189.98** | 100% | |

**Three things this reorders:**

1. **D3 is the largest object at 39.8% — and `probes/10` put it outside the kernel.** It is buffer
   bytes: an XLA relayout of the whole score buffer, ≈73 µs, plus an unexplained ~8.9 µs output term.
   ⚠ **It is therefore not a candidate — it is a handoff**, and it inflates every denominator below
   by the same factor, so it changes what the exercise is worth without changing the ranking.
2. **The histogram is the largest *structural* lever at 28.4%** (hist0 + refinement). That is D2, the
   second root: a selection that does not scatter into tilespmem.
3. **The score DMA's prize is 13.469 µs and has not moved.** It reads as 7.1% here rather than C′'s
   2.8% only because the denominator shrank by 36%. D1 and D8 both cap at that same 13.469, because
   `cvt` already hides the rest of the transfer.

---

## Stage D — blockers, each with a fix route

`references/seams.md` read here. **Re-ranked against stage C″**; the seam pass follows the table.

| id | blocker | size (prefill ρ=0.40) | fix route | instrument | tag |
| --- | --- | --- | --- | --- | --- |
| **D3** | ✅ **ANSWERED — and it is not in the kernel.** 75.683 µs belongs to no op because it is **buffer bytes outside the Pallas call**: an XLA relayout of the whole 40 MiB score buffer (`%copy.3`, `{3,2,1,0:T(8,128)}` → `{3,1,2,0:T(8,128)}`, under a `sparse-core-data-format-call`), ≈73 µs, plus an unexplained ~8.9 µs output-byte term. **No per-row term** (−0.886 ns/row) | ⚠ **39.8% — the largest object in the block, and no kernel variant can reach it** | ⛔ **not a Pallas change.** A layout/dtype decision at the call site → **`carried:` handoff**, ~73 µs, ~5× the caller's own double-buffer request | **probe 10** ✅ (`j-770c6de9`); **probe 11** ⏳ — is the relayout in the *production* path, or only under the harness's `jit`? | fits **measured**; mechanism **`modelled`** from HLO |
| **D2** | `hist0`'s dense scatter is **irreducible for a histogram** — one `vst.idx.add` per element whatever `NUM_BUCKETS` is | **28.4%** (hist0 17.8 + refinement 10.6) — **the largest structural lever** | not a tuning move. A selection that is not a histogram → **second root** in `design.md` | step-3 variant | measured |
| **D6** | The refinement levels and `emit` each walk all of `L` to touch ~`cnt` and `k` elements respectively | **28.2%** (emit 17.7 + refinement hist 10.6) | one fused pass after L0 that compresses `> b_star` straight to `rowbuf` **and** `== b_star` to a scratch buffer; levels then walk `cnt`, not `L` | step-3 variant | modelled |
| **D5** | ✅ **ANSWERED, SHIPPABLE, AND NOW MECHANICALLY EXPLAINED.** `plsc.addupdate_scatter` resolves intra-vector address conflicts: `R1` is **bit-identical, 1024/1024 and 64/64**. Its 12.195 µs win is **mostly `zero_priv` shrinking 16×** (9.238 of it) — see the ⚠ below | **−6.42% prefill** (12.195 µs of 189.98; the sweeps do not scale with ρ, so it beats the −3.73% measured at ρ=1) | drop the replication: `zero_priv(NB//LANES)`, `addr = bucket`, `fold_priv` over `range(1)` | **step-3 variant, already written** | **measured, bit-exact** |
| **D1** | The score DMA is 100% exposed (`probes/06` Q1: one `resident` alloca, 3 transfers, 3 waits, two literal `snop`s in the slack) | **7.1%**, ceiling **13.469 µs** — unchanged in µs since `probes/08`; only the denominator moved | the caller's double buffer. `probes/06` Q5 confirms it fits (110.50 KiB of 512) | step-3 variant | **measured** |
| **D8** | ⚠ **The DMA is padded and the compute is ragged.** Op 1 always moves `slice_len`; every compute pass is bounded by `walk_chunks = cdiv(i_hi,16)` | **60% of prefill bytes are never read** — but the *exposed* cost is the same 13.469 µs, so **D8 and D1 cap at the same prize and do not compose** | chunk the DMA to `i_hi` — the same restructuring D1 needs | step-3 variant, with D1 | ρ **derived**; prize measured |
| **D13** | ⛔ **DEMOTED from #1 to 4th.** The four sweep primitives cost **18.866 µs**, not 151.3 — probe 08's arm deleted the sweeps *plus* all refinement *plus* most of emit. And the split is the opposite of what was assumed: `zero_priv` (a **dense 256-bucket clear**) is 9.238, while the two carried scans are 4.698 together | **8.2%**, and **D5 already takes 9.238 of it** | `zero_priv` → D5. The residual carried-scan prize is **4.698 µs = 2.5%**: block-sum → cumsum → add back, depth 16 → 3 | step-3 variant, **low priority** | **measured** (`probes/09`), additive to 0.2% |
| **D10** | `fold_priv` is a 16-deep serial accumulate | **2.6%** (4.900 µs) — was "part of 13.6%" | bucket-major layout `bucket*16 + lane` → a 4-deep lane reduction | ⚠ **largely subsumed by D5**, which sets `R = 1`. Re-price after D5 | measured |
| **D14** | At ρ ≤ 0.2 `keep_all` fires, so **the shortest prefill rows run a full DMA + `cvt` + `hist0` + sweep to discover "keep everything"** — and `i_hi <= k` decides it with no histogram at all. ⚠ **It does *not* also save the refinement passes**: an in-kernel `keep_all` short-circuit already exists (`sparsecore_topk.py:213`, neutering all three levels at `:226-250`) | **~8.9 µs, −4.7%** on the **1/4 of prefill rows with `token_pos < 2048`** (say it that way, not "2 of 8 chunks" — that form assumes the full budget per chunk). Terms: DMA 3.367 + sweeps 3.908 + `hist0` 2.119 + `cvt` **−0.512** | `@pl.when(i_hi > k)` around ops 1-9; the fast path still owes `rowbuf` the identity plus `−1` padding, so it needs a cheap replacement emit | ⛔ **the homogeneity gate does not exist.** There is **no inter-subcore barrier** (`:19`, `:528`; per-subcore `fori_loop` at `:329`), so cost is `max` over subcores of each subcore's **own row-sum**, and stride-32 ownership (`:156-158`) makes those near-equal. Steps **do** mix decode and prefill rows (`tpu_runner.py:1154-1215`) and the saving survives it | **modelled**; the gate is ✅ **source-read closed** |
| **D9** | `NUM_BUCKETS = 256` has **never been swept**. It trades fixed-sweep size against level count — and stage C″ shows the fixed side is dominated by `zero_priv`, which is **linear in `NUM_BUCKETS`** | `unmodelled`, but now with a named mechanism | one compile-time constant, three arms | step-3 variant | `unmodelled` |
| **D4** | ⛔ **CLOSED — the 22.042 µs intercept was the instrument.** A trivial op's launch costs 20.193 µs against a 19.630 µs null; launching *this* kernel one extra time at constant total work costs **−0.704** | ⛔ **0%.** "Decode is 43.2% launch" is **retracted** | none — it does not exist in production, where the kernel is one op in a fused step with no per-call fence | — | **measured** (`probes/08`) |
| **D7** | **The TensorCore and MXU are idle for 100% of the block**, and the block is 99.8% unoverlapped with its neighbours at prefill | `min(t_TC, t_SC)·(p−1)/p` — `t_TC` **`unmodelled`** | ⚠ `P=1`: there is **no partition loop to reorder** (see the S6 note). The pipeline must be *constructed*, in the attention op | **handoff** | `unmodelled` |
| **D11** | Every subcore DMAs all `b_pad` lengths and reads `b/32` of them — 4 KiB for 32 useful words at prefill, 4 KiB for 2 at decode | ⛔ **CLOSED, not promoted.** It was the per-row candidate for D3 and `probes/10` measured the per-row axis at **−0.886 ns/row**, inside the 0.770 µs null spread | none needed | **probe 10** ✅ | now **measured-negligible**, not `assumed` |
| **D12** | `emit` is the only hot loop with register spills (9 slots in 26 bundles) and is serial via its `(woff, rank)` carry | 17.7% — emit is bigger than stage C thought | **PARKED — strictly dominated by D6**, which deletes the traversal this would optimize | — | measured |

⚠ **D5 and D13 are the same saving and must not be added.** `R1`'s measured 12.195 µs and
`zero_priv`'s measured 9.238 µs overlap almost entirely: dropping the 16× replication is *what*
shrinks the 256-entry clear to 16. Compose them in step 4 and measure the pair, never sum them.

⚠ **D1 and D8 are also the same prize.** Both cap at the 13.469 µs of *exposed* transfer, because
`cvt` already absorbs the rest. Moving fewer bytes and overlapping the bytes you move cannot both be
paid for out of one 13.469 µs.

### D3: what the 75.683 µs cannot be

⚠ The per-wave framing this section was originally written in is **void** — see M4. Restated as a
whole-kernel statement: at the calibrated 574 ps/bundle, 75.683 µs is **131,800 bundles**, against a
whole program of **1,168 bundles** (`probes/06` Q4). **It is not instruction issue**, and no amount
of scheduling, unrolling or hoisting can be its cause.

**Four candidate mechanisms have now been refuted, and the term survived every one of them.**

1. ⛔ **M1, the per-wave `sfence`+`barrier` — REFUTED, and for free.** `probes/06`'s Mosaic dump was
   re-read rather than re-captured: the module contains **exactly one `sc_tpu.sfence` and exactly one
   `sc_tpu.barrier`**, at indent 8, immediately before `memref.alloca_scope.return`, **outside every
   `scf.for`**. The wave body appears once and holds all 3 `sc_tpu.enqueue_transfer`s, so the wave
   loop is a real loop and the barrier is **once per kernel, not once per wave**. Stage D's earlier
   reading of `probes/06` Q2 — "Mosaic emits it at every wave boundary" — was wrong; Q2 said the ops
   sit "at the end of the body" and *the body it meant is the kernel's*. Cost: one grep.
2. ⛔ **M2, score-DMA latency — REFUTED.** `-dma` saves **13.469 µs**, *below* the 33.43 µs bandwidth
   model rather than above it. ⚠ **The −25.2% branch this section used to advertise does not exist**;
   the caller's candidate is a −4.12% candidate. The DMA's latency partly *is* being absorbed — by
   `cvt`, which is why deleting `cvt` makes the kernel **slower** (stage C′ correction 2).
3. ⛔ **M3, `parallel_loop` entry/exit — REFUTED.** `-alld` (dynamic-zero bounds, loops entered) minus
   `-alls` (loops eliminated at trace time) is **−0.333 µs over 32 waves**. A skipped loop is free.
   The same arm answers `probes/07` §4 from the other side: **a skipped refinement level costs 0.060
   µs**, not the ~3.8 µs that would have closed the 2.231-vs-2.03 gap. That gap is a score-distribution
   mismatch, exactly the caveat `probes/05` carried forward about itself.

4. ⛔ **M4, the wave loop itself — REFUTED, and it takes M1's premise with it.** `probes/09` measured
   the slope against `num_waves` **directly** instead of dividing: `A0_W8` 92.012, `A0_W16` 91.994,
   `A0` (32 waves) 92.231 — **0.009 µs/wave**. There is no per-wave cost of any kind. Everything
   above that was expressed "per wave" was an artifact of dividing a total by 32, and the framing at
   the top of this section — "2.540 µs per wave is 4,425 bundles" — is arithmetic on a quantity that
   does not exist. **The bundle-count refutation of instruction issue still holds** (75.683 µs total
   is 131,800 bundles against a 1,168-bundle program), but read it as a whole-kernel statement.

**Where the number now stands.** `probes/09`'s `A0` arm removed the `fill` contamination this section
predicted: `-alld` 101.492 − `A0` 92.231 = **9.261 µs of fill + output DMA**, of which 5.872 is the
output DMA and 3.389 the spurious fill. The de-contaminated residual is **75.683 µs**, and it is
**39.8% of the production prefill kernel — the largest single object in the block.**

### ✅ D3: answered by `probes/10` — it is buffer bytes, and it is outside the Pallas kernel

`probes/10-residual-driver/` (`j-770c6de9-2044-4216-bcdd`) varied `n`, `k` and `b` independently on
this same `A0` arm. Full write-up: `probes/10-residual-driver/README.md`.

**The per-row candidate is dead, and it was excluded without a fit.** `A0_n/4` and `A0_b/4` carry
**identical input bytes** (10 MiB) and differ 4× in `b`; after the output-byte slope the leftover is
**−0.886 ns/row**, inside the 0.770 µs null spread. ⛔ **D11 is closed, not promoted.**

⚠ **Do not quote the b-axis fit.** It reads 71.4 ns/row at R² 0.992 and it is an artifact: scaling
`b` scales both byte axes by the same factor, so that "slope" is the byte slopes relabelled. The
bytes-only model leaves residues of the *wrong sign* for a per-row cost.

**It is buffer bytes, in two terms:**

| term | rate | quality |
| --- | --- | --- |
| **input**, `b·n·4` | ⚠ **convex** — 1.020 µs/MiB over 10→20 MiB, **2.147** over 20→40 MiB. **Quote the segment; the fitted 1.8246 chord (R² 0.977) hides a 2.1× change in rate** and extrapolates to nonsense below 10 MiB | dominant — ~73 of 75.3 µs |
| **output**, `b·k·4` | 1.118 µs/MiB, intercept 66.4, **R² 0.99978** — genuinely linear | ~8.9 µs at the base shape, and **unexplained** |

**The optimized HLO names the input term, and it is not in the kernel.** Inside the timed `jit`,
before the custom call:

```
ENTRY %main.2 (scores.1: f32[1024,10240], row_lengths.1: s32[1024]) -> s32[1024,2048]
  %bitcast-convert_bitcast_fusion = s32[128,80,8,128]{3,2,1,0:T(8,128)} fusion(%scores.1), kind=kLoop
  %copy.3                         = s32[128,80,8,128]{3,1,2,0:T(8,128)} copy(%param_0.6)
  … sparse-core-data-format-call …
```

A layout change on the **whole 40 MiB score buffer** — full read plus full write, ≈73 µs at
~1.1 TB/s. ⚠ **`A0` deletes the score DMA and XLA reformats the buffer anyway**, because it is a
custom-call operand. That is precisely why a total kernel blackout does not remove it, and it is what
all four refuted mechanisms were groping for.

**Status of the two halves.** The timing fits are **measured**; the mechanism is **`modelled`** from a
static read of one arm's optimized module. `probes/11-relayout-provenance/` is dispatched to settle
whether the relayout exists in the **production** call path or only under this harness's `jit`
signature, and to chase the unexplained output-byte term.

⚠ **Both answers are consequential and neither is a kernel change.** If production has the relayout,
D3 is a ~73 µs **`carried:` handoff** — a layout or dtype decision at the call site, worth more than
every in-kernel plan combined. If it does not, the **189.98 µs denominator overstates the production
kernel by ~40%** (it becomes ~117 µs) and every share in the sheets below roughly doubles. The
ranking is invariant either way, because D3 is outside every candidate.

### The seam pass

`seams.md` requires "no instance, because —" on the record, so all seven are answered.

| seam | instance |
| --- | --- |
| **S1** layout × algebra | **The TensorCore relayout model does not apply** — one register axis, `mismatch` structurally `none` on every row. Its SparseCore analogue is V4's multiplicity, and the S1-shaped move on it is **D10** (change which axis the replica index lives on so the fold becomes a tree) |
| **S2** units × algebra | No MXU, so the row-axis matmul arithmetic is void — **but the shape transfers.** Ops 2, 4 and 12 traverse V3 with the same access pattern and the same bounds. **`cvt` ⊕ `hist0` is one traversal**: read, convert, write back, scatter. Prize ≈ the read half of `cvt`, ~10 µs (2.2%). D6 is the same move applied to `emit`. Both are `add`-free **fusion** edges |
| **S3** pipelining × algebra | Four serial chains. `fold_priv`'s 16-deep accumulate → **D10**. `scan_glob`/`find_bucket`'s two 16-step carries → **D13** (the narrower object that generates a 256-element prefix sum is the 16 block sums; depth 16 → 3). V6 across levels — **irreducible**, the next level's `shift` genuinely depends on this one's `find_bucket`. `emit`'s `(woff, rank)` carry → dominated by D6 |
| **S4** precomputation × layout | ⛔ **No instance, because the loops are memory- or latency-bound and the recomputes are free.** `probes/06` Q6 found four loop-invariant recomputes surviving into the body (`lane_iota*256` and two broadcasts) and priced hoisting them at ~3 of 14 bundles with **no expected saving**, which stage C's calibration now confirms independently for the walk (0.92× issue-bound) |
| **S5** partitioning × fusion | ⛔ **No instance, because the available axis is serial in its consumer.** The axis in the data and not in the loop is **`layer`** — 21 of GLM-5.2's 78 layers call this, so M is paid 21× per token step — and layers are sequential in the model. `P` and `stage` are degenerate at 1. `b` is already the parallel axis and fully occupies all 32 subcores |
| **S6** units × fusion | **D7**, and it is the largest *potential* prize in the file — but ⚠ **`seams.md`'s own warning applies verbatim here.** `_pick_partition` at `P=1` reshapes to a single launch of `b·P` rows, data-parallel across subcores; **it is not a loop**, so there is nothing to reorder and the pipeline has to be constructed in the caller. And S6's degenerate regime is live at decode, where chunking `b=64` costs waves outright |
| **S7** fusion × layout | The kernel → `sc_dsa_gather` boundary writes 8.39 MB and reads it straight back. **Not blocked by layout** — both sides are `[lane=k]` on the same unit — so the reverse reading applies: the boundary is a real `pallas_call` dependence. Prize ≈ 7.8 µs (**1.7%**). **Recorded and not pursued**, and it is smaller than any blocker above it |

### `CONTRACT.md` §B — the working set

| row | answer |
| --- | --- |
| **W1** live set as a formula | `4·(L + 16·NB + NB + (k+16) + b_pad + 32) + 1,344 B` — resident `L`, `priv` 16·`NB`, `glob` `NB`, `rowbuf` `k+16`, `len_vmem` `b_pad`, 32 words of DMA descriptor scratch, and inter-buffer padding. At `L=10240, NB=256, k=2048, b_pad=1024`: 4 × 17,712 + 1,344 = **72,192 B = 70.50 KiB**. ✅ **Exactly `probes/06` Q5's measured high-water**, confirmed twice there — the formula is validated, not asserted |
| **W2** the bound, and where it comes from | 524,288 B per subcore (`SparseCoreInfo.vmem_capacity_bytes`, reported in-job). **Inherited, not chosen**: `compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, needs_layout_passes=False)` sets **no** `vmem_limit_bytes`. Occupancy **13.77%**. ⚠ `probes/06` Q5 leaves the denominator open — `kTileSpmemCapacity` is a compiled constant, not a string — and it would have to be wrong by 2× to matter |
| **W3** what binds the tile | ⛔ **Nothing binds it.** `n` is untiled and the whole row is made resident **by algorithmic decision** — it is what makes the HBM:VMEM pass ratio 1:5 instead of 5:5 — not by capacity. Even the largest shape in the tree (`n=133120, P=4`) is 180.5 KiB = 34.4%. **4.3× headroom, unused.** That is the stage-E "room to fuse" signal: D1's second slice buffer is 110.50 KiB (21.6%) and D6's straddler scratch on top of it is 150.5 KiB (29.5%) — **both fit together**, which is a step-4 composition fact and not a step-3 conflict |

### `CONTRACT.md` §C — transfer, per array

| row | answer |
| --- | --- |
| **T1** mechanism | **Manual `pltpu.make_async_copy(src, dst, sem).start(); .wait()`** — *not* a `BlockSpec` grid pipeline. Three transfers per launch: lengths (once, op 0), score (per wave, op 1), out (per wave, op 13), plus keys (op 14) on the `emit_keys` arm. All sizes are **static**; the grid is a static `fori_loop` over `num_waves`, and only `i_hi`/`walk_chunks` are dynamic, read from VMEM |
| **T2** partial / padded | ⚠ **Padded, and this is blocker D8.** Op 1 always moves `slice_len` words; every compute pass is bounded by `walk_chunks = cdiv(i_hi, 16)`. Padded fraction `1 − ρ`, ρ = E[eff_len]/n. **Every measurement in this exercise is at ρ = 1** — `probes/07` `probe.py:282` passes `jnp.full((b,), n)`, and `instruments/glm_dsa_block_bench.py` passes no `row_lengths` at all, so `sparsecore_topk.py:552-553` defaults to `n`. Production is `_effective_row_lengths` (`streamindex_topk.py:482-506`), `visible = min(kv_len, token_pos // cr + 1)`, and GLM-5.2 is **`compression_ratio = 1`** (`mla_attention_op.py:116`, Chain A — the `csa_compression_ratio = 4` in `deepseek_v4_attention_op.py:270` is Chain B and is not the shipping model). So ρ < 1 at prefill by construction. **ρ is `assumed`; deriving it is host-side and costs no job** |
| **T3** conditional | The whole wave body is under `@pl.when(i_hi > 0)`, with op **Z** (`zg`) as the else branch. **No transfer is issued for an empty row.** At both shipping shapes `b` is a multiple of 32, so this path never fires — it is the floor for a work-stealing design, not a live cost |
| **T4** ordering / overlap | ⛔ **None. Overlap fraction 0.0000.** One `resident` alloca (`10240xi32`), 3 `enqueue_transfer`, 3 `transfer_wait`; the compiler had 4 bundles of slack across the 40 KiB read and filled two with literal `snop`s (`probes/06` Q1). Overlap is **structurally impossible** without a second buffer, which is precisely D1 |
| **T5** destination / alignment | HBM → tilespmem, contiguous on both sides, DMA granule **32 B**. `slice_len` = 10,240 words = 40,960 B = **1,280 granules exactly** — no sub-granule tail at the shipping shape. ⚠ A ragged DMA (D8) at word granularity *creates* one: round `i_hi` up to 8 words to stay granule-aligned |

### `CONTRACT.md` §D — partial tiles

| row | answer |
| --- | --- |
| **R1** what a partial tile computes | `n` is untiled, so **there is no partial tile on `n`**. The partial unit is a **partial wave**: the last wave has `b mod 32` active subcores. At both shipping shapes `b ∈ {64, 1024}` is a multiple of 32, so **R1 never fires on the shipping path**. When it does, the inactive subcores run op **Z** only and then wait at **BAR** — they compute no garbage and write no garbage, but because the kernel costs `max` over subcores, **a partial wave costs a full wave**. Unrelated: `rowbuf`'s `+16` slack is the `store_compressed` overrun guard, not a tile pad |

**No `CONTRACT.md` row is tagged `unmodelled`.** The three that are not closed on paper carry a
probe id instead: **L6** → `probes/07` (answered), **T2**'s ρ → a host derivation, and **W2**'s
denominator → `probes/06` Q5's open note.
