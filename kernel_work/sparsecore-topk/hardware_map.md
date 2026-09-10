# `sparsecore_topk` — hardware map

Stages A–D for the **current** version. Edited in place as the version under design changes; the
version graph itself lives in `design.md`.

Source of record: `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` at commit `b11b6eb6`,
md5 `44353b347c96e0eedbd742c710d4540a`. Step 1's findings are in `report.md`; nothing here restates
a measurement, it cites the section or the probe folder.

**Sign convention** (binding in every document this exercise writes): every percentage is
`(new − old)/old` of *time*, so negative is faster. Signed percentages appear only in a `Δ time %`
column; prose states the direction in words.

**Status.** Stages A–D complete; the stage-D probe wave is running. `probes/08-per-wave-residual/`
returned (`j-300d4690-d53e-441e-9d39`) and **rewrote most of stage C**: the launch intercept was the
instrument (D4 closed), the score DMA is 13.469 µs not 33.43 (D1's ceiling −4.12%, not −7.18%), D5's
gate answered **yes** with a bit-exact −3.73%, and the regions turn out **super-additive by 1.625×**.
The host ρ derivation then added stage **C′**, which inverts the ranking: at the shipping ρ = 0.40005
the two fixed V4 sweeps are **42.2%** of the prefill kernel and `hist0` is 11.5%. `probes/09` is in
flight to split the sweep and de-contaminate D3. Then stage E in `design.md`.

⚠ **Nothing below that predates `probes/08` should be quoted without checking it against stage C′.**

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

## Stage A — `CONTRACT.md` §A, filled for `v1`

Drafted here beside the value table that produces it. **Its home is `design.md` §4d**; it is
transcribed there verbatim when that file opens, and this copy is deleted at that point so the two
cannot drift.

| value | L1 produced | L1 consumed | L2 window | L3 broadcast | L4 concat/split | L5 transpose | L6 multiplicity | L7 itemsize |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **V2** `scores` | `(b)[lane=n] HBM` | `[lane=n] VMEM` | none — no windowed read | none | **split on `n` into `P`; at both shipping shapes `P=1`, not split** | none | none | n/a |
| **V3** `resident_vmem` | `[lane=n] VMEM` | `[lane=n] VMEM`, all five consumers | none | none | not split — `n` untiled by decision | none | none | `int32` |
| **V4** `priv_vmem` | `lane[lane=bucket] VMEM`, addr `lane*256+bucket` | `lane[lane=bucket] VMEM` | none — `fold_priv` is **16 contiguous 16-word loads**, not a strided read (`for m in range(16)` is a python unroll) | **`lane_iota` replicates the 256-bucket histogram 16× along the leading axis.** Leading, so free to *address*; **not** free to zero or to fold | none | none | ⛔ **16×, and it is not elementwise.** `CONTRACT` L6's exception: the compiler does not hoist it, because the 16 replicas are genuinely distinct accumulators. **Measured (`probes/07`): 19.192 µs per level, 13.6% of the prefill kernel** across L0 + ~2.231 refinement levels | `int32` (`lane_iota`, line 100; `addr`; `mask` from int32 compares) |
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
**L6** — ⛔ **the one row that is not free.** Priced by `probes/07` at **19.192 µs per level = 13.6%
of the prefill kernel**; no longer `unmodelled`.
**L7** — `int32` throughout; no int8 iota, no 8-bit compare, so the floor `CONTRACT` L7 warns about
is not approached.

**Rows answered by a probe** (the mechanically generated list, which is what this table is for):

| row | probe id | what each answer would change |
| --- | --- | --- |
| **L6** on V4 | ✅ `probes/07-refinement-fixed-vs-walk`, `j-098b260f-a9f7-4fe5-937c` | **Answered.** The fixed sweep is 33.6% of a level, not most of it — so *both* halves of the alternative are live, and they are separate candidates rather than one being a red herring. Deleting the replication is worth 13.6% of the prefill kernel; shortening the walk is worth 18.7%. Candidate 1's prize fell 51.4% → 18.7%, close to §5d′'s predicted ~15.8% |

Every other §A contract row is closed on paper, which `CONTRACT.md` says is a fine result to state
explicitly. **§B, §C and §D are filled at stage D below.**

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
3. **The V4 sweeps are latency, not bytes** — 1.87× their issue count, from `fold_priv`'s 16-deep
   accumulate and the two 16-step serial carries. ✅ **Confirmed and now much larger than this stage
   could see.** `probes/08` deleted all four sweep loops for **151.324 µs**, the biggest single delta
   in the job, and `R1` — which removes the replication entirely, bit-exactly — recovers only 12.2 of
   it. So **~92% of the sweep is chain depth, not replication**: D13, not D5, and the fix needs no
   change to `R`. At production ρ the two sweeps are **42.2%** of the kernel.

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
| of which: residual, `-alld` − null | **81.95** | D3 |
| 225.58 + 81.95 | 307.53 | ✅ closes to 0.003% |

| term | single-arm Δ (**upper bound**) | scales with ρ? | tag |
| --- | --- | --- | --- |
| V4 sweep — `zero_priv`+`fold_priv`+`scan_glob`+`find_bucket` | **151.324** | **no** | measured, `probes/08` |
| `hist0` dense scatter | **138.306** | yes | measured, `probes/08` |
| ⚠ residual, belongs to no op | **81.95** | no (per *wave*) | measured by subtraction — D3 |
| `emit` | 84.020 | yes | measured, `probes/08` |
| refinement `L` walk × 2.231 | 84.79 | yes | measured, `j-098b260f` |
| refinement V4 sweep × 2.231 | 42.81 | **no** | measured, `j-098b260f` |
| score DMA | **13.469** | **no** (D8) | measured, `probes/08` |
| `cvt` | **−20.492** | yes | measured, `probes/08` — *negative*, see below |
| launch | ⛔ **0** | — | **closed**; it was the instrument |

### ⚠ Three corrections that govern every candidate from here

1. **The regions are super-additive by 1.625×.** The five single-arm deltas sum to 366.628 against a
   joint `B` − `-alld` of 225.582. They contend for the tilespmem port and the memory system, so
   deleting one lets the others expand into the time it freed — the in-kernel form of
   `freed-device-time-spills-onto-untouched-ops`. **Divide any modelled single-region deletion by
   1.625 before believing it.** The joint is exact; the rows are upper bounds.
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

### Prefill balance sheet at production ρ = 0.40005 — ~297 µs

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

## Stage D — blockers, each with a fix route

`references/seams.md` read here. Ranked by measured share; the seam pass follows the table.

| id | blocker | size | fix route | instrument | tag |
| --- | --- | --- | --- | --- | --- |
| **D13** | ⚠ **`fold_priv`'s 16-deep accumulate and the two 16-step carried scans (`scan_glob`, `find_bucket`) are ~139 µs of the sweep's 151.3.** ~135 ns per carried iteration = **235 cycles** for a body of a handful of bundles — pure latency | ⚠ **~38.8%** — **the largest object in the kernel** | block-sum → cumsum the 16 block sums → add back. Depth 16 → 3. Independent of `R` | **probe 09** splits the three, then step-3 variant | measured jointly (`probes/08`); the split is **open** |
| **D3** | ⚠ **81.95 µs belongs to no op**, confirmed to 0.8% by a second, independent decomposition | **27.6%** prefill | ⛔ **M1, M2 and M3 are all now refuted — see below.** Re-probe before designing | **probe 09** (`A0` family) | measured by subtraction |
| **D6** | The refinement levels and `emit` each walk all of `L` to touch ~`cnt` and `k` elements respectively | **17.7%** (walk 10.7 + emit 7.0) | one fused pass after L0 that compresses `> b_star` straight to `rowbuf` **and** `== b_star` to a scratch buffer; levels then walk `cnt`, not `L` | step-3 variant | modelled |
| **D2** | `hist0`'s dense scatter is **irreducible for a histogram** — one `vst.idx.add` per element whatever `NUM_BUCKETS` is | **11.5%** (was 25.6% before ρ) | not a tuning move. A selection that is not a histogram → **second root** in `design.md` | step-3 variant | measured |
| **D5** | ✅ **ANSWERED AND SHIPPABLE.** `plsc.addupdate_scatter` **does** resolve intra-vector address conflicts: `R1` is **bit-identical, 1024/1024 and 64/64 rows**. V4's 16× replication is not required for correctness | **−3.73% prefill / −1.86% decode, measured** (`probes/08`) | drop the replication: `zero_priv(NB//LANES)`, `addr = bucket`, `fold_priv` over `range(1)` | **step-3 variant, and it is already written** | **measured, bit-exact** |
| **D8** | ⚠ **The DMA is padded and the compute is ragged.** Op 1 always moves `slice_len`; every compute pass is bounded by `walk_chunks` | **ρ = 0.40005 at prefill — 60% of the bytes moved are never read.** But the DMA is only 2.8%, so the prize is ≤ 1.7% | chunk the DMA to `i_hi` — **the same restructuring D1 needs**, so compose them | step-3 variant, with D1 | ρ **derived**; prize measured |
| **D1** | The score DMA is 100% exposed (`probes/06` Q1: one `resident` alloca, 3 transfers, 3 waits, two literal `snop`s in the slack) | ⛔ **2.8%**, ceiling **−4.12%** prefill / **−1.97%** decode | the caller's double buffer. `probes/06` Q5 confirms it fits (110.50 KiB of 512) | step-3 variant | **measured** (`probes/08`) |
| **D14** | At ρ ≤ 0.2 `keep_all` fires for every row, so **prefill chunks 0 and 1 run a full DMA + `cvt` + `hist0` + sweep to discover "keep everything"** — and `i_hi <= k` decides it with no histogram at all | 2 of 8 chunks | `@pl.when(i_hi > k)` around the selection, emit the identity otherwise | ⚠ **gated on step homogeneity** — cost is `max` over subcores, so a mixed batch gains nothing. Unknown | modelled; the gate is `unmodelled` |
| **D4** | ⛔ **CLOSED — the 22.042 µs intercept was the instrument.** A trivial op's launch costs 20.193 µs against a 19.630 µs null, and launching *this* kernel one extra time at constant total work costs **−0.704**. On-device SC program load is **−20.897**, i.e. zero | ⛔ **0%.** Stage C's "decode is 43.2% launch" is **retracted** | none — it does not exist in production, where the kernel is one op in a fused step with no per-call fence | — | **measured** (`probes/08`) |
| **D7** | **The TensorCore and MXU are idle for 100% of the block**, and the block is 99.8% unoverlapped with its neighbours at prefill | `min(t_TC, t_SC)·(p−1)/p` — `t_TC` **`unmodelled`** | ⚠ `P=1`: there is **no partition loop to reorder** (see the S6 note). The pipeline must be *constructed*, in the attention op | **handoff** | `unmodelled` |
| **D8** | ⚠ **The DMA is padded and the compute is ragged.** Op 1 always moves `slice_len`; every compute pass is bounded by `walk_chunks = cdiv(i_hi,16)` | `(1−ρ)` × 7.4%, and it re-weights the whole sheet | derive ρ = E[eff_len]/n host-side, then chunk the DMA — **the same restructuring D1 needs** | host derivation, then step 3 | ρ is **`assumed`** |
| **D9** | `NUM_BUCKETS = 256` has **never been swept**. It trades fixed-sweep size against level count | `unmodelled` | one compile-time constant, three arms | step-3 variant | `unmodelled` |
| **D10** | `fold_priv` is a **16-deep serial accumulate**; stage C measures the sweeps at 1.87× their issue count | part of D5's 13.6% | bucket-major layout `bucket*16 + lane` turns the chain into a 4-deep lane reduction — **and needs no change to `R`** | step-3 variant, gated on a `plsc` cross-lane reduce existing | modelled |
| **D13** | `scan_glob` and `find_bucket` are two more **16-step serial carries** over 256 buckets | part of D5's 13.6% | block-sum → cumsum the 16 block sums → add back. Depth 16 → 3 | step-3 variant | modelled |
| **D11** | Every subcore DMAs all `b_pad` lengths and reads `b/32` of them — 4 KiB for 32 useful words at prefill, 4 KiB for 2 at decode | `assumed` negligible | none proposed | — | **`assumed`, never measured** |
| **D12** | `emit` is the only hot loop with register spills (9 slots in 26 bundles) and is serial via its `(woff, rank)` carry | 6.0% | **PARKED — strictly dominated by D6**, which deletes the traversal this would optimize | — | measured |

### D3: what the 81 µs cannot be

At the calibrated 574 ps/bundle, 2.540 µs per wave is **4,425 bundles**. The entire program is
**1,168 bundles** (`probes/06` Q4). So the unattributed term is **3.8× the whole kernel's static
instruction count**, per wave — **it is not instruction issue**, and no amount of scheduling,
unrolling or hoisting can be its cause. Three mechanisms survive:

**All three candidate mechanisms are now refuted, and the term survived every one of them.**

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

**⚠ And the 81.95 is partly an artifact, which `probes/09` must remove before the term is trusted.**
`fill_bound = jnp.where(out_gate & (total < k), k // LANES, 0)`. Forcing `scan_glob`'s bound to zero
makes `total = 0`, so **`-alld` runs a 128-iteration-per-wave fill loop that `B` never runs** (in `B`,
`total = 10240 > k`). Scaling `cvt`'s measured 640-iteration cost gives ~0.13 µs/wave ≈ 4 µs — ~5% of
the term, so it does not explain it away, but the corrected arm is what the number should come from.

What is left inside `-alld`, and what `probes/09`'s `A0` family isolates: the output DMA, the
`fill_body` loop, **three dynamic-index `vec_at` loads per wave** (`len_vmem` at `row_c`, `glob_vmem`
at `b_star` and at `b_star−1`) — which on a 16-lane machine with no sublane axis may not be the cheap
indexed load they look like — and the scalar prologue/epilogue. **Mechanism: still open.** It is 27.6%
of the production prefill kernel and it is the second-largest object in the block, so it is worth the
one job it costs rather than a fourth guess.

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
