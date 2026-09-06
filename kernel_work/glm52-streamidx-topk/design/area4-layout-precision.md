# Area (4) constraint sheet — `streamindex_topk` / `sparsecore_topk` on GLM-5.2

Tags: `[v: j-…]` measured; `[s: path]` read from pinned source; `[u]` believed, not established. **Every relayout cost below is from `hardware-facts.md` "Relayout: the cost with no name" and is f32-only [v: j-054e7ac4]; none of the values it is applied to here are f32.** Derived constants: `n_stored = num_sublanes_total*128` = **10240** at (A) (`align_to(9,2)*1024`), **1048576** at (B) [s: `streamindex_topk.py:603-605,745`]. SC partition `p` = **1** at (A) (b=64≥32), **16** at (B) (`p_min=cdiv(n,65536)=16`, `n_p=65536`) [s: `sparsecore_topk.py:360-377`, `MAX_SLICE_WORDS=65536`].

## Emission 1 — precision, layout, wasted fraction of every read and write

| operand | precision (store / move / accum) | layout in HBM | A-D64-8k bytes (useful) | waste | B-D4-256k bytes (useful) | waste |
|---|---|---|---|---|---|---|
| `q` **R** | fp8_e4m3fn / fp8 / — | `[T,32,128]`, head_dim=128 = lane width | 262 kB (262 kB), live rows only | **0%** | 16.4 kB live of a 65.5 kB buffer | 0% read / 75% buffer |
| `weights` **R** | f32 / f32 / f32 | `[T,32]`, trailing extent 32 < 128 lanes | 8.2 kB data, ~32.8 kB charged | **75% lanes** ("below 256 lanes the lane extent is free" [v: j-ed517191]) | 512 B data, ~2 kB charged | **75% lanes** |
| `cache_kv` **R** | u8→fp8_e4m3fn + e8m0 / u8 / — | `[nb, ps/4, 4, 256]`; 129 of 256 B used [s: `:265-268`] | 134.2 MB (67.6 MB) | **49.6%** | 268.4 MB (135.2 MB) | **49.6%** |
| `scores` **W** (`jnp.full(-inf)`) | f32 | `[T, n/128, 128]` rank-3 | 2.62 MB; 0.52 MB survives as -inf | **80%** | 67.1 MB; 4.19 MB ever read | **93.75%** |
| `scores` **W** (Pallas DMA) | f32 / f32 / — | 16-sublane (A) / 8-sublane (B) column stripes | 2.10 MB | 0% | 4.19 MB | 0% |
| `scores` **R** (SC stage 1) | f32→i32 bitcast | flat `[T*n]`, `slice_len=align(n,16)` unconditional [s: `:160,324`] | 2.62 MB (2.10 MB live) | **20%** | 4.19 MB read of a 67.1 MB buffer; rows with `eff_len=0` skip the DMA | 0% of read / 93.75% of buffer |
| stage-2 gather **R** | f32 | `take_along_axis` over `[16,1048576]`, 4-byte rows | n/a (p=1) | — | 2.1 MB useful, element-granular | **87.5%** vs the 32 B DMA granule `[u]` |
| stage-1 out **W** | i32 | `[b*p, k_p]`, `write_empty=True` | n/a | — | 2.1 MB (0.13 MB live) | **93.75%** |
| final out **W** | i32 | `[T, 2048]`, unsorted, -1 pad | 524 kB, 2048/2048 filled | 0% | 131 kB, 4 of 16 rows live | **75% rows** |

Totals: (A) 137.1 MB read / 5.25 MB written, KV = 94% [matches report §3]. (B) 272.6 MB read / 73.5 MB written; the `-inf` fill is **24.6% of all traffic** and 93.75% of it is dead.

## Emission 2 — value ledger

| # | value | axis it lands on | axis consumer needs | mismatch | mechanism | cost tag |
|---|---|---|---|---|---|---|
| V1 | `fp8_val = flat_bkv[:,:128]` | (bkv_sz sub, 128 lane) | einsum RHS, contract on lanes | **none** | lane-boundary slice at 0:128 + `bitcast` | **free** [v: j-054e7ac4 slice-on-128; f32] + bitcast free at equal width [v: j-ed517191] |
| V2 | `scale_val = flat_bkv[:,128:129].T` | bkv_sz on **sublanes**, 1 lane | bkv_sz on **lanes** (multiplies `s_summed[...,bkv]`) | **yes** | explicit `swapaxes(-1,-2)` on a value that never reaches the MXU | **relayout 0.79–0.82 ps/elem** [v: j-054e7ac4, **f32 only — this value is u8**]; ≈0.4 µs (A) / 0.8 µs (B) total. The 1-lane source slice is *not* a 128-boundary slice and has **no measured row at all** |
| V3 | `s` (einsum out) | (bq·h sub, bkv lane), reshaped so `h` is the sublane axis | sum over `h` | **none** | sublane reduction | **1.01x, free** [v: j-ed517191 `lane_vs_sublane_reduction`]; at decode bq_sz=1 the rank-raise is a leading-axis add = free |
| V4 | `bq_weights[:,:,:,None]` | `num_q_heads=32` on **lanes** | `num_q_heads` on **sublanes** | **yes** | rank-raise moving 32 from lane to sublane position | **unmeasured**; nearest analogue is the 0.79–0.82 ps/elem swap [v: j-054e7ac4, f32]. Dissolvable outside the kernel by shipping `weights` head-major from `prepare_index_weights` |
| V5 | `s_summed.reshape(sbs*bq, n_sub_bkv, 128)` | bkv_sz on **lanes** | a (sublane, lane) tile for the store | **yes** | split of the *trailing* axis into two trailing axes | **no row exists** in the relayout table (it prices adds/drops/permutes of *leading* axes as free; this is not one). May fold into the VMEM store's addressing. **Probe** |
| V6 | `k_span` / masks | `arange(bkv_sz)` arrives on lanes; `kv_lens`/`q_pos` broadcast on sublanes | same | **none** | i32 `broadcasted_iota`-equivalent, compared *after* the rank-raise | free, and correct by construction — the `(BT,)→(BT,1)` **bool** reshape that does not lower is avoided [s: `04-layout-precision.md`] |
| V7 | `scores` → `reshape(T,-1)` → `bitcast_convert_type(i32)` → `reshape(-1)` | contiguous row-major | same | **none** *as arithmetic* | equal-width bitcast: `i32→f32` costs exactly `f32→f32` [v: j-ed517191] | free — **but these are XLA ops between two kernels, and whether XLA materializes a second `[T,n]` i32 buffer is unprobed. At (B) that is 67 MB write + 67 MB read** |
| V8 | SC `resident_vmem` / histograms | 1-D, LANES=16 | same | **none** | SparseCore has no lane/sublane asymmetry | n/a |
| V9 | `local_indices` → `offsets+…` → `reshape(b, p·k_p)` | contiguous merge of last two axes | gather index vector | **none** as a reshape | free | **but it feeds an XLA `take_along_axis` on the TensorCore over 67 MB with a 1-element row — precisely the regime where SparseCore beats XLA 40–61x** [v: j-0ed646a2, j-48095a41]. Wrong unit; route to area (5) |
| V10 | `out.reshape(b,k)[:T,:k]` | contiguous | same | **none** | free slice | — |

## What area (4) forbids and requires here

**Forbids.** (a) A strided read of lanes `[0:128]` out of the 256-wide cache: measured **not to compile** — `Not Implemented: The last dim size is not 128 in original base memref` [v: j-054e7ac4], and the base memref's last dim here is exactly 256. **So the (i) question is not "is a strided read cheap" — the strided form does not exist.** The only route to the byte saving is a **repack to `lkv_dim=128` with the e8m0 scales in a separate plane**, whose read is then fully contiguous and whose 32 B DMA granule divides 128 — which removes the granularity unknown rather than answering it. (b) `(BT,)→(BT,1)` bool reshapes. (c) Sizing any block to `get_sublane_tiling` — measured not to show up [v: j-a5c66200]. (d) "f32 preserves precision" as a justification for the score matrix: f32 at DEFAULT is bit-identical to bf16 operands [v: j-106c8a46].

**Requires.** `bkv_sz % 128 == 0` [s: `:598`]; `n % 16 == 0`, `k % 16 == 0`, `k ≤ 4096`, and `scores.dtype == f32` hard-raised [s: `sparsecore_topk.py:395-402`] — so **any narrowing of the score matrix requires editing `monotone_key`'s 32-bit sign-flip too**. And the ≥2 MiB block rule [`## Deciding` 1]: the KV block is 2.1 MB at (A) — at the knee — and **1.05 MB at (B), already below it**. A 128-B repack halves both, moving *away* from the knee. That tension is why the repack must be a two-arm timed strike test (contiguous-256 vs contiguous-128, equal token count), not a byte count. Finally, the file's rule is *mask, do not pad* (1.01x vs 1.03x) [v: j-050c7480]; the `jnp.full(-inf)` over the whole `[T, n_stored]` is padding, at 67 MB, and is the single largest violation of this area's own guidance in the block.

## On (i) and (ii), specifically

**(i)** Repacking is worth ≈**-5% of the measured 363.2 µs** at A-D64-8k (the -47% is of a 38.3 µs model), and the halved block may give some of that back at (B). The **capacity** win is unconditional and independent of bandwidth: 127 B/token × 21 indexer layers = **~2.7 kB/token of HBM freed**, which at (B) is where concurrency is capped. Probe named above; it settles a *bandwidth* question the repacked form makes trivial, so the honest statement is that the probe measures block-size regression, not granularity.

**(ii) Two of the three named candidates are ruled out by arithmetic, not by measurement.** The f32 head sum over 32 terms has relative error ~32·2⁻²⁴ ≈ 1.9e-6 — three orders below the observed 2.4e-3 (0.005 / 2.1) — so it cannot be the cause. The **e8m0→bf16 cast is exact**: e8m0 is a pure exponent and bf16 carries all 8 f32 exponent bits, so no e8m0 value in range loses information. The remaining candidate, **fp8 MXU accumulation width, matches the observation almost exactly**: 2.4e-3 against the measured bf16-accumulation relative max error of **2.2104e-3** [v: j-106c8a46 `f32_default_is_one_bf16_pass`]. That is a hypothesis from arithmetic agreement, not a measurement — the probe is the same einsum at fp8 operands with `preferred_element_type=f32` versus bf16-upcast operands at `HIGHEST`, both against an f64 oracle. **Consequence for narrowing: the score path is already sitting at the bf16 noise floor, so there is roughly zero headroom left before a further narrowing compounds it.** Costing a bf16 score matrix against the measured 0.3%: bf16's 8 mantissa bits perturb a score by ~2e-3 relative, ≈0.001 at the cut value 0.43, against a `TOL_DEFICIT` of 0.05·2.1 = 0.105 — a ~100x margin on *deficit*, but **churn is the metric at risk and is not predictable from this arithmetic**. fp8 scores (e4m3, 3 mantissa bits, ~6% relative → ±0.026 at the cut, 25% of tolerance) are not costable as safe. The non-obvious point: at (B) the SC read is only 4.19 MB against 795.9 µs of SC time, so a narrower score matrix buys **no bandwidth worth having** — its value would be halving the radix key width from 4 levels to 2, which is a throughput term belonging to areas (5)/(6), not a layout term.

## Gaps this block turns on (each with the decision it settles)

- **float8 block legality was never swept** (`hardware-facts.md § Not established`; only int8's 32-row minimum has a data point). *Decides:* whether a repacked 128-wide fp8 cache, and `bq` at `bq_sz=1` (exactly 32 rows), sit at or below the legal minimum, and therefore whether `decode_req_batch_size` may be lowered. *Probe:* pure-copy block-row sweep 8/16/32/64 on `float8_e4m3fn`.
- **The relayout table is f32-only; bf16's control misbehaved and u8 was never run.** *Decides:* V2 (u8 scale transpose) and V4 (f32 weights lane→sublane) — the only two costs in the ledger with a real mismatch and no number. *Probe:* extend the `j-054e7ac4` carrier harness to u8 and bf16.
- **"No measurement of any layout change *across* a kernel boundary"** (`04 § Gaps`, last line). *Decides:* V5 and V7 — whether the rank-3 `[T, n/128, 128]` score layout costs anything against a flat `[T, n]`, and whether XLA's `bitcast_convert_type` between the TC and SC kernels materializes a second buffer. At (B) that is a 67 MB write plus a 67 MB read, i.e. it decides whether the traffic table above is right by a factor of two.
- **Not turned on:** the int4 rows, the LHS-transpose gap (the einsum contracts the last axis of both operands — NT dimension numbers, the measured-free case [v: j-050c7480]), and the `transpose_cost` / `mask_versus_pad` spread caveats.

## Considered and not favoured (one line each — no choice made)

- **Strided 128-of-256 DMA out of the existing cache** — measured not to compile at a base memref whose last dim ≠ 128.
- **Repack to `lkv_dim=128` + separate e8m0 scale plane** — full byte saving, contiguous, and it deletes V2's transpose; but drops the KV block to 1.05/0.52 MB, below the bandwidth knee.
- **Repack to `lkv_dim=132` (128 + 4 pad)** — keeps scale inline and preserves a single DMA, but 132 is not a lane multiple and the block-shape rule rejects it.
- **bf16 score matrix in HBM** — halves 67 MB and the radix key width, but breaks the `f32`-only guard and `monotone_key`, and compounds an error already at the bf16 floor.
- **fp8 score matrix** — 6% relative at the cut, 25% of `TOL_DEFICIT`, and unpredictable churn.
- **i16 or i32 monotone key written directly by the scoring kernel** (skipping the f32 store and the XLA bitcast) — removes V7's unprobed materialization, but moves `monotone_key` onto the TensorCore's critical path.
- **bf16 `weights`** — halves an 8 kB read; the 75% lane waste dominates and is unaffected.
- **Head-major `weights` (`f32[32, T]`) from the call site** — dissolves V4 with `none`, but is a call-site change and pays a transpose once per step instead of per grid step.
- **Padding `n` out to `align_to(pages_per_seq, bkv_p)` (status quo at (A), 10240 vs 9216)** — 20% wasted on both the score write and the SC read, against the file's measured preference for masking over padding.
- **Sizing the score buffer to live context rather than `max_model_len`** — the largest layout lever in the block, but it is an allocation/algebra change and belongs to areas (1)/(6), not here.
- **`num_kv_pages_per_block[1]` (the dead PREFILL slot) lowered** — shrinks `num_sublanes_total` and hence the whole score buffer with no compute effect; a one-line layout lever, unmeasured.
- **Moving the stage-2 `take_along_axis` gather to SparseCore** — the 1-element-row regime where SC wins 40–61x, but it is an area (5) placement decision.
