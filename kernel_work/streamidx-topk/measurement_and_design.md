# `streamidx_topk` — measurement record and design

Step 2 of `write-tpu-kernels` for the GLM-5.2 DSA lightning indexer. **No kernel code is written
here and no source file is modified.** Step 3 is gated on a human reading this.

Inputs: `report.md` (step 1), `briefs/area1..8*.md` (wave 1), `briefs/CONTRADICTIONS.md`
(the cross-sheet authority), `probes/P0..P7*.md` (wave 2), and the incumbent source.

---

## 0. How to read the numbers in this file

Every quantity carries one of four tags. They are not interchangeable and the ranking in §4 turns
on which is which.

| tag | means |
| --- | --- |
| `measured (job)` | read off hardware in the named job. The job id is always given. |
| `derived (job)` | arithmetic whose every input is `measured`, with the arithmetic shown. Subtracting two measured times to get a third is `derived`, not `measured`. |
| `static (job)` | compile-time analysis of the emitted program — LLO bundle counts, instruction histograms, VMEM totals. It is *not* a runtime measurement. A static instruction count is a good proxy for a straight-line inner loop and a bad one for anything with data-dependent control flow. |
| `modelled` | computed from a formula. The formula is always given inline. |
| `assumed` | a value nobody in this exercise established. Every `assumed` value that carries a conclusion is called out where it appears. |

**`AgentX`** marks a figure taken at the secondary operating point (`max_model_len=262144`, b=16,
page 256, TP16×DP1) rather than at the target. AgentX instruction *counts* and *structural* facts
carry over; AgentX *microseconds* do not. Any AgentX µs quoted here is quoted as context only.

**Two floors are used and they are different things.** Both are defined in §1.6. Confusing them is
the single easiest way to misread this document.

---

## 1. The measurement record

### 1.1 The baseline of record

The configuration everything below is measured or modelled at. It is called **nightly** throughout.

| parameter | value |
| --- | --- |
| model | GLM-5.2, DSA lightning indexer, 21 indexer layers of 78 |
| parallelism | DP=16 |
| batch | 64 sequences |
| decode block | `bq_sz=1`, `bkv_sz=2048`, `num_kv_pages_per_block=2`, `seq_batch_size=4` |
| positions | `num_positions=10240` (= `num_bkv` 5 × `bkv_sz` 2048; `num_bkv = cdiv(8704, 2048) = 5`) |
| context | `max_model_len` 9216, ctx 8704 |
| page | 1024 positions |
| K record | `lkv_dim=256` bytes per position (129 used: 128 fp8 value bytes + 1 e8m0 scale byte, padded to a 128 multiple) |
| heads / dim | H=32, D=128 |
| top-k | k=2048, exact, output is an unordered index **set**, `-1` suffix-padded |
| prefill | 1024 query tokens per chunk, `bq_sz=128`, 8 bq blocks, same KV geometry |

**Baseline timings — `measured (j-8b1e7acc, JobStatus=Succeeded)`**, image `ec4bcecf…`, code
`cbabaf9b`. One indexer layer, in the compiled model, clean pass:

| phase | whole call | stage A (`StreamIdxTC-*`) | stage B (`sc_topk_direct_*`) | residual |
| --- | ---: | ---: | ---: | ---: |
| decode | **221.4 µs** | 109.2 µs `derived` | 67.4 µs | 44.8 µs `derived` |
| prefill | **857.3 µs** | 335.1 µs | 481.0 µs | 41.2 µs `derived` |

- **stage A** — the three chained TensorCore Pallas `pallas_call`s in `streamindex_topk.py` that
  score every KV position against every query head and write an f32 score matrix to HBM.
  `score[t,s] = sum_h relu(q[t,h]·k[s]) * weights[t,h]`, causally masked at `s <= q_abs_pos`.
- **stage B** — `sparsecore_topk`, an exact MSB-first radix select running on SparseCore, picking
  the top k=2048 entries of each row of that matrix.
- **residual** — the call minus the two kernel scopes. It is *not* only the `-inf` score-buffer
  initialisation; it is that plus every scalar `slice`/`concatenate`/`gather` XLA emits around the
  `pallas_call`s. At decode it is **20.2 % of the call** and is the second-largest bucket. Its
  composition at nightly is not measured — see Option D (§3.4).

Stage A at decode is `derived` because the profile reports the two kernel scopes and the call; 109.2
= 221.4 − 67.4 − 44.8 only after the residual is separately attributed. `report.md` §6 gives the
same 109.2 with a stated derivation range of 109–151 µs; **109.2 is used throughout and the 151
end is noted where it would change a conclusion.**

### 1.2 What the instrument gets wrong

These defects are established in `report.md` §6 and `CONTRADICTIONS.md`, and they bound what any
number above can mean.

| defect | consequence |
| --- | --- |
| **Run-to-run spread ±8.5 %** on the end-to-end call. | A predicted win below ~9 % cannot be confirmed by a single pair of runs. Every prediction in §3 states whether it clears this. |
| **The standalone bench enqueues `iters` calls and blocks once.** | Per-`pallas_call` fixed costs measured in the bench are inflated. The `-inf` init reads **28.6 µs** in the bench arm and **6.02 µs `AgentX`** in the compiled program. `area2 §D` forbids importing the reference's "~785 µs validity rule" and its 150 µs per-`pallas_call` floor into this block; both are bench artefacts. |
| **The reference's 0.65 ragged-tail discount must not be applied here** (`area2 §D`). The incumbent's tail DMA is constant-length by construction — `_fetch_bkv` clamps `page_idx` and `safe_page_offset` and always copies a full page. | Any "the tail is cheap because it is short" reasoning is wrong for this kernel. The tail is *full price and mostly waste*; that is what makes the tail-predicate lever real (§3.5). |
| **`/xprof-llo` emits the same verdict line for materially different arms.** P1 §2 shows the `bkvp_2` and `bkvp_1` arms differing on every static field and receiving an identical verdict string. | The dump-side verdict text is not evidence. Only its numeric fields are, and they are `static`. |
| **A four-shard fan-out on one 4-chip pod contends for host and HBM.** All of wave 2 ran this way. | Wave 2's absolute microseconds are context. Every wave-2 conclusion quoted here is a *ratio measured within one process against its own controls*. |
| **`report.md` §4's `t_hbm` column is computed at the reported 3.70e12 B/s, not the measured 2.97e12 it lists two lines above.** Check: 175.9 MB / 3.70e12 = 47.5 µs, which is the number printed; at 2.97e12 the same bytes are 59.2 µs. Prefill likewise: 151.1 MB / 3.70e12 = 40.8 µs. | Every HBM floor in `report.md` §4 is **24.7 % optimistic**. §1.6 recomputes them. This is an arithmetic defect in the briefs, not a modelling choice. |

### 1.3 Every probe, one line each

All eight ran under **`j-e107f728-6fdc-47d8-98f9`** except where noted. That job's `JobStatus` was
`Running` when the shards were read — each shard printed its own complete JSON and only shard 0's
cosmetic `pack` step was outstanding. **It is not a green job and is not quoted as one.**

| probe | question | verdict | key number |
| --- | --- | --- | --- |
| **P0** donation elision | does the `donate_argnums=(2,)` KV-cache donation actually elide a copy? | yes, and it is already tiny | scatter is 8192 B / 3.024 µs for a 16-row in-place update `measured` |
| **P1** DMA stall | is stage A ≥50 % blocked on DMA? | **NO — and it does not close §2 either** | `count_sync_wait` **16.65 % mean / 5.51 % median**, max 101.28 %. 66,053 samples at `interval_us: 20`, wrap-checked (max raw 38,521 < 65,535) `measured` |
| **P2** XLA vs SparseCore top-k | is the SparseCore path the right one? | yes, keep stage B on SparseCore | isolated `sparsecore_topk` **52.12 µs** at nightly decode shape `measured` |
| **P3** TC/SC overlap | can a TensorCore grid and a SparseCore mesh be in flight at once? | **YES, but only inside one XLA program** | **41.6 %** of the SparseCore arm hidden (reps=1, fixed-cost-corrected). Two executables **−1.8 %**; two threads **−11.5 %** `measured` |
| **P4** empty launch | what does a zero-work `pallas_call` cost? | far less than the briefs assumed | **0.55 µs** (sbs1) / **0.04 µs** (sbs4) marginal, 2.86 µs grid-0 intercept `measured`. **This supersedes area 3's 5.2–5.4 µs and `CONTRADICTIONS` §8's instruction to use it.** |
| **P5** cache consumers | who else reads or writes the indexer K cache? | exactly one writer, one reader | the `scatter_fusion` in `_streamidx_topk_jax` and stage A. **The layout is privately owned** — this is what makes Option A legal `measured` |
| **P6** strided read charge | does HBM charge for bytes moved or for the extent spanned? | **bytes moved** | charge ratio **1.02** at w=128; **98.0 %** of contiguous bandwidth at a 128 KiB granule `measured` |
| **P7a** `bkv_p` | does raising `num_kv_pages_per_block` 2→3 pay? | **yes, and it is the largest measured single-knob win** | stage A **−19.99 %** at near-constant bytes = **−11.9 % of the decode call** `measured` |
| **P7b** spill | is prefill register-pressure-bound? | yes, severely | 8,458 spills + 10,130 fills over 11,658 bundles; 1,506 live vregs against 64 physical (**23.5× oversubscribed**); VSTORE:SPILL **97.1 %**; VALU **94.19 %** occupied, **89.45 %** saturated `static` |

`P1` and `P3` were also read for §1.4 and §1.5. The CSE check is closed separately:
`measured (j-a71a648c rank 0)`, `hlo_stats`, 16554/16554 rows kept, **every metadata row an exact
multiple of 21 — not one instruction is shared between the 21 indexer layers.**

### 1.4 The arithmetic I was asked to redo, and three more I had to

#### 1.4.1 The MXU: the prober's number does not reproduce, and the corrected one says something different

P1 §1.2 states "≈4.4 % of peak" and "6.25 % of its cells". I recomputed both.

**MAC count.** The prober used 9216 positions. The kernel walks `num_bkv × bkv_sz = 5 × 2048 =
10240` — the padded extent, not `max_model_len`. Every position in that extent is scored; the
causal mask is applied *after* the matmul, so masked positions still cost MACs.

```
MACs/call = 64 seqs × 10240 positions × 32 heads × 128 dims = 2.684e9 MAC = 5.369e9 FLOP   modelled
```

**Independent confirmation this count is right:** `area5 §B` reports `t_mxu = 13.5 µs` decode from
`flops / mxu_peak_used × mxu_cliff_factor`. With my count:
`5.369e9 / 1419.9e12 × 3.57 = 13.50 µs` — reproduces exactly. The prober's 9216 would give 12.15.

**Achieved rate**, against the derived stage-A time:

```
2.684e9 MAC / 109.2 µs = 2.458e13 MAC/s = 4.917e13 FLOP/s                      derived (j-8b1e7acc)
```

**Against three different denominators**, because which one you pick moves the answer 2.3×:

| denominator | source | % of peak |
| --- | --- | ---: |
| 2.30e15 FLOP/s | vendor fp8 reported | **2.14 %** |
| 1419.9e12 FLOP/s | `area5 §B`, measured fp8_e4m3fn→f32 | **3.46 %** |
| 9.96e14 FLOP/s | the prober's own modelled 256×256 × 2 MXUs × 2 MAC/cell/cycle × 1.9 GHz | **4.94 %** |

So on the prober's own assumed geometry the figure is **4.94 %, not 4.4 %** — the 0.54-point gap is
entirely the position count. The user's cross-check "4.4/95.5 implies 4.6 %" is the right instinct:
with the corrected count, **4.94 / 95.49 = 5.17 % of peak per busy cycle**, against a 6.25 % fill
ceiling = **82.7 % of the fill ceiling**, not 73.7 %.

**The 6.25 % fill figure is arithmetically fine and evidentially empty.** For a 256×256 array a
[32 rows × 128 deep] operand occupies 4096/65536 = 6.25 % of cells; the user's 32/128 = 25 %
presumes a 128×128 array. But **the array-size assumption cancels**: peak ∝ S² and fill = 4096/S²,
so `(achieved/peak)/fill` is invariant in S. This measurement cannot distinguish 128×128 from
256×256 even though the headline "% of peak" moves 4.5× between them. **Do not quote "4.4 % of
peak" or "6.25 % fill" as evidence of anything.** The invariant is:

```
stationary-tile utilisation = 2.458e13 / (4096 cells × 2 MAC/cell/cycle × 2 MXUs × 1.900e9 Hz)
                            = 2.458e13 / 3.113e13 = 78.97 %
per busy cycle              = 78.97 % / 95.49 % = 82.7 %                                modelled
```

Read that as: **when the MXU is busy, it is doing 83 % of what a 32-row operand permits it to do.**
The array is not underfed; it is *narrow*. The lever is the row extent, not the schedule.

**The clock is derived, not assumed.** From P1's own wrap check: 38,521 raw counts /
(20e-6 s × 1.0137 sampling correction) = **1.900e9 Hz** `derived (j-e107f728)`.

**One counter fact P1 does not explain and I could not resolve.** 2.458e13 MAC/s exceeds one
128×128 MXU at 1.9 GHz and 2 fp8 MAC/cell/cycle (1.5565e13) by **1.58×**, so at least two MXUs must
be contributing — yet `count_mxu_busy_1` reads 0.12 % and `_2` reads 3.74 % while `_0` reads
95.49 %. Either the array is ≥256×256, or the `_N` suffix does not index physical MXUs. **The
counter's semantics are unknown and no conclusion here rests on them.**

**P1 §4's claim that this is "the largest lever P1 can see" is wrong by P1's own numbers.** The
recoverable MXU time is bounded by the measured row cliff, not by the distance to peak.
`area5 §B` measures `mxu_cliff_factor = 3.57×` at 32 rows. Removing the cliff entirely gives
`13.50 / 3.57 = 3.78 µs`, so the whole prize is `13.50 − 3.78 = 9.72 µs` = **8.9 % of stage A =
4.4 % of the decode call** — below the ±8.5 % instrument spread. And `area5 §A2` already shows there
is no cheap mechanism to fill the rows: the 32 is `H`, it is a model constant, and batching
sequences into the row axis is what `seq_batch_size` already does. **The MXU is not a lever at
decode. No option below targets it.**

#### 1.4.2 The VPU rate: `report.md` §4's 1.72e12 is 4.5× pessimistic, and it manufactured a false verdict

`report.md` §4 uses `VPU 1.72e12` tagged `assumed`, "derived as MXU 661.4e12 / 385". That is a ratio
carried in from another sheet. The rate is derivable from P1's own counters:

```
VPU peak = 4 VALU issue slots × 1024 f32 lanes × 1.900e9 Hz = 7.782e12 element-ops/s      derived
```

Four slots because P1 reports four independent `count_vector_alu_instruction_0..3` counters
(55.18 / 52.70 / 49.49 / 46.50 %). 1024 lanes is `assumed` (8 sublanes × 128 lanes); 1.900 GHz is
derived above.

Cross-check, which is not circular because it uses a different measured quantity: the achieved
element-op rate is `4.332e8 ops / 109.2 µs = 3.97e12`, and `3.97e12 / 7.782e12 = 50.97 %` — exactly
P1's measured Vector-ALU roll-up. And P1's `static` VALU occupancy of 62.08 % sits consistently
above the measured 50.97 %.

**`report.md` §4's second `assumed` — "3 VPU ops per (token, head, position), a guess at what Mosaic
emits" — is closed by recounting from source.** From `compute_scores`
(`streamindex_topk.py:326-353`), per bkv block of 262,144 elements = 256 f32 vregs:

| op | count (vregs) | source line |
| --- | ---: | --- |
| `jnp.maximum(s, 0.0)` | 256 | the relu |
| `* bq_weights_stacked[..., None]` | 256 | the per-head weight |
| `s.sum(axis=2)` reduction tree | ~248 | 32 heads → 1 |
| `* scale_val_stacked` | 8 | post-reduce, on the narrow array |
| `valid_mask`, `causal_mask`, `jnp.where` | ~40 | post-reduce |
| **total** | **~808 per 256 input vregs = 3.16** | |

Per decode call: 2.097e7 elements × 3.16 = 6.63e7 element-ops, against `report.md`'s 6.29e7 —
**agreement to 5 %**. The "3 ops" guess was right. It is now `derived from source` rather than
`assumed`, and **3.1 is used below.**

#### 1.4.3 The VALU inflation, which is the number prefill actually turns on

| phase | ideal element-ops | achieved element-ops | inflation |
| --- | ---: | ---: | ---: |
| decode | 6.55e7 `modelled` (2.097e7 × 3.1) | 4.332e8 `derived` (109.2 µs × 1.9 GHz × 4 slots × 50.97 % × 1024) | **6.6×** |
| prefill | 1.040e9 `modelled` (3.355e8 × 3.1) | 2.456e9 `derived` (335.1 µs × 1.9 GHz × 4 slots × 94.19 %`static` × 1024) | **2.4×** |

Decode's 6.6× does not matter: `achieved t_vpu = 4.332e8 / 7.782e12 = 55.7 µs` still sits under the
57.4 µs of HBM time it overlaps with. **Prefill's 2.4× is the whole story at prefill** — stage A
there is VALU-saturated at 89.45 % `static` and is not near any memory limit.

**Where the prefill inflation comes from — the one avoidable block, identified from P7b's op
histogram plus source.** P7b's `static` mix against the irreducible count:

| op | P7b count `static` | irreducible `modelled` | excess |
| --- | ---: | ---: | ---: |
| `vmax` (relu) | 8,192 | 8,192 | 0 |
| `vmul` (×w, ×scale) | 10,240 | 8,192 | 2,048 |
| `vadd` (reduce + mask) | 12,320 | 7,936 | 4,384 |
| **`vrot.slane`** | **8,960** | **0** | **8,960** |
| total | 39,712 | 24,320 | 15,392 (1.63×) |

`vrot.slane` is a cross-sublane rotate. It exists because the reduction axis is *not* vreg-aligned.
`compute_scores` builds `bq_stacked` as `(seq_batch, bq_sz * num_q_heads, head_dim)` — **heads
minor to query tokens** — so the einsum output is `(seq_batch, bq_sz*H, bkv_sz)`, reshaped to
`(seq_batch, bq_sz, H, bkv_sz)` and summed over axis 2. With `bkv_sz` on lanes, `H` has a stride of
one sublane row, and a 32-way reduction over adjacent sublanes lowers to a
`vrot.slane`-by-1,2,4,8,16 tree. This is exactly the "cross-lane rotate-and-add reduction tree
draining `vpop.f32.mrb[*]`" that P1 §2.1 finds occupying the single largest stall run
(bundles `0x7ac`–`0xada`, 815 bundles, 29 % of the kernel) — **and P1 mislabels that run
`mem_stall_mxu_idle`, which is why it was not attributed until now.**

**8,960 of 39,712 VALU ops = 22.6 %, and it is structural, not necessary.** This is Option C.

### 1.5 §2 — does stage A have schedule headroom? — resolved, and it does not flip anything

This is the open question `CONTRADICTIONS` leaves live and that I was told not to paper over.

P1 refutes the ≥50 % DMA-bound branch decisively (`count_sync_wait` 16.65 % mean, 5.51 % median).
It does *not* reach the ≤15 % compute-bound line, so §2 stayed open. Resolving it:

```
stage A decode moves 170.4 MB in 109.2 µs = 1.56 TB/s                          derived (j-8b1e7acc)
      = 52.5 % of the measured 2.97 TB/s ceiling
```

52.5 % of ceiling with only 16.65 % sync-wait is not a contradiction — it is what an eight-deep
double-buffered stream looks like when the *rate* is granule-limited rather than the *issue* being
starved. `area3 §A3` gives 0.8–1.5 TB/s as the achievable band for a 256 KiB granule; 1.56 TB/s is
**at the top of that band**, and `area3`'s own steady-state model predicts 109–170 µs against a
derived 109–151 µs measured.

**Answer: stage A is bandwidth-limited at the granule-dependent rate, not sync-wait-bound, and it is
already at the top of the band its granule allows.**

The apparent counter-evidence is P7a's `measured` −19.99 % from `bkv_p` 2→3 at near-constant bytes.
It is not a counter-example, because **P7a and P6 act on different axes**:

- **`bkv_p` controls inner-iteration count.** 2→3 takes `num_bkv` from 5 to 3 — 40 % fewer
  iterations, each with its own semaphore round and its own reduction-tree drain. It also drops KV
  bytes 167.8 → 151.0 MB (a 2.6 % effect, `area2 §Correction`).
- **Payload width controls bandwidth.** P6's charge ratio is 1.02 and 98.0 % of contiguous
  bandwidth is retained at a 128 KiB granule.

**Consequence for the ranking: Option A (bytes) and the `bkv_p` sweep (iterations) are additive, not
substitutes.** And §2's answer changes the *attribution* of the win, not the *plan*:

| if §2 answers… | expected decode total |
| --- | --- |
| "no headroom — stage A is at its granule's ceiling" | Option A alone at the top of its range, **−22 %** |
| "headroom — the schedule is loose" | Option A at the conservative end **−13 %** plus the measured `bkv_p` sweep **−11.9 %** ≈ **−23 %** |

**The two branches predict the same total to within 1 point.** §2 is worth resolving for
understanding; it is not worth resolving before acting.

### 1.6 The floors, recomputed — and the definition that matters

> ⚠️ **PARTIALLY FALSIFIED by an adversarial review — see `briefs/ROOFLINE-REVIEW.md`.**
> The three named corrections in §6 items 1, 4 and 5 hold, and both corrected fractions
> reproduce exactly from this section's stated inputs. **The prefill gate does not.** The
> 80.4 % figure is circular: its dominant term `t_sc = 481.0 µs` — 69.8 % of the 689.3 µs
> floor — is numerically the measured stage-B time it is divided into (`stage_b_ms =
> 0.48097`), and `area5 §C.7` admits the 1.7 ns/elem constant was fitted to that
> measurement. Across six defensible treatments prefill runs **46.5 %–80.4 %**, only the
> published one clearing the bar, and by 0.4 points.
>
> **Consequences, both live:**
> - **Prefill is NOT gated.** Do not read "≥80 %, do not write a kernel" off this section.
>   Options C and B are in scope. This document's own `F_common` puts prefill at 56.1 %.
> - **Decode's verdict is safe** — below 80 % under every treatment (25.6 %–56.8 %) — but
>   **do not quote 39.9 %.** It rests on a `t_sc` of 30.0 µs that `report.md` §6.2
>   explicitly retracts, against P2/P3's isolated 48.2–52.1 µs. Say **26–57 %, structural
>   band, proceed.**
>
> Also falsified: **§1.1's relabelling of the 44.8 µs `init` column as a metadata
> "residual"**. It is `init_ms`, a standalone `jnp.full(-inf)` arm, and it sits at 1.07× the
> measured `dispatch_floor_ms = 0.042` — ~94 % host dispatch, which no hoist can remove.
> Option D's rank-3 placement is **unevidenced**, not merely wide-ranged. A genuine residual
> of ≈60 µs does exist by a different subtraction; see the review.

`report.md` §1 defines the floor as `max(t_hbm, t_vpu + t_mxu)`: **HBM is assumed to overlap
compute; the compute units are assumed not to overlap each other.** It is *not* the largest of
three or four columns. That definition is kept.

Component rates:

| rate | value | tag |
| --- | ---: | --- |
| HBM | 2.97e12 B/s (measured, 80 % of the 3.70e12 reported) | `measured` |
| VPU | 7.782e12 element-ops/s | `derived` (§1.4.2) — replaces `report.md`'s `assumed` 1.72e12 |
| MXU fp8→f32 | 1419.9e12 FLOP/s, × `mxu_cliff_factor` = `128/rows` when rows<128 | `measured`, `area5 §B` |
| SparseCore | `t_sc` from `area5 §B`'s issue model: 30.0 µs decode, 481.0 µs prefill | `modelled` |

Two floors, and they answer different questions:

- **`F_struct(design)`** — the floor for *that design's own structure*: stages summed where the
  design serialises them. `measured / F_struct` is a **quality-of-implementation** number: how
  close this code gets to the best its own shape allows.
- **`F_common`** — the floor for the least-constrained *legal* structure: one fused kernel, no score
  matrix in HBM, TensorCore and SparseCore concurrent (P3 says they can be). Independent of design,
  so it is the only fair **cross-option** yardstick.

Watch for the trap: **a change that shrinks the floor faster than it shrinks the runtime lowers the
achieved fraction while being an unambiguous win.** Option A does exactly this. Read the fractions
against `F_common`, and read `F_struct` only to ask "is this design well implemented".

#### Incumbent, decode

```
init      : bytes 2.62 MB          → t_hbm 0.88 µs; compute ≈0        → 0.88 µs
stage A   : bytes 167.77 (K) + 2.62 (score W) = 170.39 MB → t_hbm 57.37 µs
            t_vpu 6.55e7/7.782e12 = 8.42 µs ; t_mxu 13.50 µs ; sum 21.92 µs
            max(57.37, 21.92)                                          → 57.37 µs   HBM-bound 2.6×
stage B   : bytes 2.62 (R) + 0.52 (W) = 3.14 MB → t_hbm 1.06 µs ; t_sc 30.0 µs
            max(1.06, 30.0)                                            → 30.00 µs
F_struct  = 0.88 + 57.37 + 30.00                                       = 88.25 µs
achieved  = 88.25 / 221.4                                              = 39.9 %
```

#### Incumbent, prefill

```
init      : 41.94 MB → 14.12 µs                                        → 14.12 µs
stage A   : bytes 21.0 (K, 8× re-read) + 41.94 (score W) = 62.94 MB → t_hbm 21.19 µs
            t_vpu 1.040e9/7.782e12 = 133.7 µs ; t_mxu 8.590e10/1419.9e12 = 60.5 µs (cliff 1.00,
            rows = 4096 ≥ 128) ; sum 194.2 µs
            max(21.19, 194.2)                                          → 194.2 µs   VPU-bound 9.2×
stage B   : bytes 41.94 (R) + 8.39 (W) = 50.33 MB → 16.95 µs ; t_sc 481.0 µs
            max(16.95, 481.0)                                          → 481.0 µs
F_struct  = 14.12 + 194.2 + 481.0                                      = 689.3 µs
achieved  = 689.3 / 857.3                                              = 80.4 %
```

#### `F_common`, both phases

```
decode  : HBM = K 167.77 + topk out 0.52 = 168.29 MB → 56.66 µs
          TC compute = 8.42 + 13.50 = 21.92 ; SC = 30.0 ; TC ∥ SC per P3
          max(56.66, 21.92, 30.0)                                      = 56.66 µs → 25.6 % of 221.4
prefill : HBM = K 21.0 + out 8.39 = 29.39 MB → 9.89 µs
          TC compute = 194.2 ; SC = 481.0
          max(9.89, 194.2, 481.0)                                      = 481.0 µs → 56.1 % of 857.3
```

#### This overturns `report.md` §4 and §8 in both directions

| | `report.md` §4/§8 | here |
| --- | --- | --- |
| decode floor | 47.5 µs, **21.5 %**, HBM-bound | **88.25 µs, 39.9 %** |
| prefill floor | 623.3 µs, **72.7 %**, "VPU-bound by 14×" | **689.3 µs, 80.4 %**, VPU-bound by 9.2× *in stage A only* |

Three corrections drive it, and they pull opposite ways:

1. `report.md`'s `t_hbm` used 3.70e12, not the 2.97e12 it declares (§1.2). Raises the HBM column
   33 %.
2. `report.md`'s VPU rate was 4.5× too slow (§1.4.2). Prefill `t_vpu` falls 586 → 133.7 µs.
   "VPU-bound by 14×" was an artefact of that rate.
3. **`report.md` §4 modelled stage A's roofline and compared it against the whole call.** Stage B
   is on a different physical unit and was simply absent from the floor. Adding `t_sc` is the
   largest single correction at prefill.

**The headline the corrected floors give, and it decides the whole ranking:**

> **Decode is at 39.9 % of its own structural floor. Prefill is at 80.4 %.**
> The skill's step-1 rule refuses to write a kernel for a block already at ≥80 % of its modelled
> floor. **Prefill is at that bar.** Decode is not, by a wide margin.

Prefill's remaining 19.6 % is almost entirely stage A's 2.4× VALU inflation (§1.4.3), which is a
cheap layout fix, not a kernel; stage B at prefill is at 481.0 measured against a 481.0 modelled
floor and **the only way to improve prefill's stage B is to hide it, not to speed it up.**

---

## 2. The incumbent, decided across all seven areas

This is the reference every option below is a delta against. Area 1's ten-field schema is
instantiated at both phases in `briefs/area1-fusion-boundary.md` §B2 (decode) and §B3 (prefill) and
is **not** reproduced here — read it there. What follows is the one-paragraph decision per area
plus each area's required `Emit`.

### Area 1 — fusion boundary

**Decision: three chained TensorCore `pallas_call`s producing an f32 score matrix in HBM, then one
SparseCore kernel consuming it.** The three stage-A calls (decode-batched at `seq_batch_size=4`,
decode-remainder at `seq_batch_size=1`, mixed/prefill) are chained through a single aliased buffer
via `input_output_aliases={len(scalar_prefetches)+3: 0}`, so each takes the previous call's output
as its `scores_init`.

Area 1 is the only area with no `Emit` line; it closes on advantage/disadvantage prose.

- *Advantage.* The A→B edge is cheap as a **kernel** edge: the score matrix round trip is 2.62 MB
  each way, `2 × 2.62e6 / 3.0e12 = 1.75 µs = 0.79 %` of the decode call `modelled`. The three-call
  split lets each phase pick its own `bq_sz` and `seq_batch_size` without a dynamic branch inside
  the kernel.
- *Disadvantage, and it is the big one.* The same edge is a **unit** edge, and
  `CONTRADICTIONS` §9 prices the unit edge at ~40× the kernel edge: TensorCore is stalled for the
  whole of `t_sc` (`area5 §B`: `tc_stall = all of t_sc`) — 67.4 µs of 221.4 at decode, 481.0 of
  857.3 at prefill. P3 now shows that stall is not mandatory.
- *Second disadvantage.* The aliasing that chains the three calls (`area1 §A8`) prevents any
  concurrency *among* them. At nightly decode two of the three run on an empty range
  (`distribution[1] == distribution[2]` in the pure-decode step), which P4 prices at 0.04 µs each —
  so this costs nothing today, but it is the mechanism Option B has to work around.

### Area 2 — partitioning

**Decision: grid over sequence groups; the whole `num_bkv × bkv_sz` position extent walked in an
in-kernel loop; no `index_map`s and all DMA hand-rolled.** `PrefetchScalarGridSpec` with
`num_scalar_prefetch=6`, every `in_spec` at `pltpu.HBM`, `dimension_semantics=("arbitrary",)`.

`Emit` (nightly decode, from `briefs/area2-partitioning.md` §B):

| field | value |
| --- | --- |
| grid | `(16,)`, `(0,)`, `(0,)` — three calls, two empty |
| inner iteration space | `num_bkv = 5` bkv blocks × `bkv_p = 2` pages × `seq_batch_size = 4` sequences |
| block shapes | KV `(2 pages × 256 groups, 4, 256)` = 262,144 B per DMA; scores out `(4, 80, 128)` f32 |
| working set vs capacity | VMEM 4.13 MiB `static (P1)` against 128 MiB HW / ~31 MiB scoped — **96.7 % headroom** |
| kernel invocations per step | **63 TensorCore launches, 42 of them empty; 21 SparseCore launches**; 336 working grid steps |
| total HBM traffic implied | K 167.8 + q ~0.3 + score W 2.62 + score R 2.62 + topk out 3.14 ≈ **176 MB**; **13,440 KV DMAs of 256 KiB** per step |
| idle fraction | bkv rounding **15.0 %** (10240 walked / 8704 real); within-group spread **0 %** at homogeneous lengths; empty launches **42/63**; causal waste **<10 %**; stage-B subcore idle **0 %** |

Do **not** multiply the 15.0 % against area 4's 50.4 % naively as separate wins — `area2 §B` already
does it: **only 42.8 % of streamed decode K-cache bytes are useful.**

### Area 3 — pipelining

**Decision: two-deep double buffering on every operand, primed once per `pallas_call`, hand-rolled
`make_async_copy` with a `SemaphoreType.DMA((4, 2, seq_batch_size))` array.**

`Emit` (nightly decode; the twelve fields of `briefs/area3-pipelining.md` §B):

| # | field | value |
| --- | --- | --- |
| 1 | `bytes_in/out_per_step`; useful fraction | 2,129,920 B in per inner step; **useful fraction 50.4 %** (129 of 256 bytes per K record) |
| 2 | `dma_count_per_step`; `bytes_per_individual_dma`; `max_outstanding` | 8 KV DMAs per bkv block (= `seq_batch_size` 4 × `bkv_p` 2); **262,144 B each — the granule, not the aggregate**; 8 outstanding |
| 3 | `t_hbm_per_step` at ceiling / at corrected rate | 0.72 µs @ 2.97e12; **1.4–2.1 µs** at `area3 §A3`'s 0.8–1.5 TB/s band for this granule |
| 4 | `t_mxu` / `t_vpu` / `t_compute` per step | 0.169 / 0.105 / **0.274 µs** (13.50 and 8.42 µs over 80 inner steps) |
| 5 | **`dominant_side`** = `t_compute / t_hbm_corrected` | **0.13–0.20 — bytes dominate, decisively** |
| 6 | `buffer_depth` per operand vs 64 MiB physical | 2 for bkv / bq / bq_weights / bo_scores; 4.13 MiB total `static` |
| 7 | `n_inner_steps_total`; priming | 80; **primed once per `pallas_call`**, not per grid step |
| 8 | `t_prologue` | one bkv fetch ≈ 2.1 µs `modelled` |
| 9 | `t_empty_pallas_calls` | 42 × **0.04 µs = 1.7 µs** `measured (P4)`. **Area 3's own baseline row says 5.2–5.4 µs each = 218–227 µs, which is impossible — it exceeds the call. P4 supersedes it; so does `CONTRADICTIONS` §8's instruction to use 5.2–5.4.** |
| 10 | `tail_waste_fraction` | **15.0 %** (the 5th bkv block is 1536 of 2048 positions real), at full DMA price — `area2 §D` forbids the 0.65 discount |
| 11 | `t_epilogue` | one score store ≈ 0.03 µs |
| 12 | `tc_sc_overlap` | **`N/A because C = 1`** — the incumbent runs stage B once, after all of stage A. |

Steady state alone predicts 57 µs at the HBM ceiling and 109–170 µs at `area3 §A3`'s corrected rate,
against 109–151 µs derived measured. **The pipeline is not the defect; the byte count is.**

### Area 4 — layout and precision

**Decision: fp8_e4m3fn K values with a per-position e8m0 scale byte, concatenated into one uint8
record and padded to `lkv_dim = 256`; f32 everywhere downstream.**

`Emit` — precision and layout of every operand, and the wasted fraction of each read and write:

| operand | dtype | shape / layout | direction | wasted fraction |
| --- | --- | --- | --- | ---: |
| `cache_kv` | uint8 | `[num_blocks, 256, 4, 256]` — page, group, `kv_packing`, `lkv_dim` | read | **50.4 %** (129 of 256 bytes used) |
| `q` | fp8_e4m3fn (bitcast from uint8) | `[T, H, 128]` → stacked `[seq_batch, bq_sz*H, 128]` | read | 0 % |
| `weights` | f32 | `[T, H]` | read | 0 % |
| `scores_init` | f32, `-inf` | `[T, num_sublanes_total, 128]` | write, then aliased | **15.0 %** (padded extent) |
| `scores` | f32 | same, aliased | write | 15.0 % |
| `scores` | f32 | flattened `[b, n]` | read by stage B | 15 % past `row_length` (`area5 §B`) |
| `topk_idxs` | int32 | `[b, k]`, `-1` padded | write | 0 % |

f32 is **forced** on stage B by `sparsecore_topk.py:396`; `n` must be a multiple of 16 and `k` a
multiple of 16 with `0 < k ≤ 4096`. The MXU accumulates fp8×fp8→f32 natively at 1419.9e12
`measured`, so nothing downstream of the matmul can be narrowed without an accuracy argument.

**The 50.4 % waste on `cache_kv` is 95.4 % of decode's HBM traffic** and is the single largest
structural inefficiency in the block. It exists because Mosaic requires the minor (lane) dimension
to be a multiple of 128 and 129 rounds to 256.

### Area 5 — compute units

**Decision: TensorCore VPU+MXU for stage A; SparseCore `VectorSubcoreMesh` for stage B; they do not
overlap.** SparseCore *is* reachable — `from jax.experimental.pallas import tpu_sc as plsc`,
`get_tpu_info().sparse_core` = `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16,
vmem_capacity_bytes=524288, dma_granule_size_bytes=32)`. **Always read `sc.num_cores *
sc.num_subcores`; never the literal 32 and never the docstring** — `hardware-facts.md`'s "no
SparseCore path" claim and its literal-32 subcore field are two of the three known-wrong entries
flagged in `CONTRADICTIONS` §7 (the third is the 32 MiB scoped-VMEM ceiling).

`Emit` — per-unit occupancy (from `briefs/area5-compute-units.md` §B):

| unit | decode | prefill |
| --- | --- | --- |
| MXU | `mxu_flops` 5.37 GFLOP; rows 32; `machine_fraction` 0.25; **cliff 3.57×**; `t_mxu` **13.5 µs = 6.1 %** of the call | 85.9 GFLOP; rows 4096; fraction 1.00; cliff 1.00×; `t_mxu` **60.5 µs = 7.1 %** |
| VPU | 2.097e7 elements × 3.1 ops; `t_vpu` **8.42 µs** (`derived` rate, §1.4.2); live intermediate 1.05 MB = 3 % of scoped VMEM | 3.355e8 × 3.1; `t_vpu` **133.7 µs**; live intermediate **33.55 MB = 105 % ⚠️** — this is what P7b's 23.5× vreg oversubscription is |
| SparseCore | `p=1`, 1 stage, no gathers; 64 rows; **2 waves**; **subcore occupancy 32/32 = 100 %**; resident VMEM/subcore 67,008 B = **12.8 %** of 524,288; `t_sc` ≈ **30 µs** | `p=1`; 1024 rows; 32 waves; 100 %; `t_sc` ≈ **481 µs**, matching the measurement |
| `tc_stall` | **all of `t_sc`** | **all of `t_sc`** |

The two structural reasons `area5` says a design may have to decline SparseCore:

1. **Capacity.** `4 × align(n/p, 16) + ~26 KB > 524,288` forces `p` up, which forces two stages,
   which forces two `jnp.take_along_axis` gathers back onto TensorCore.
2. **Serialization** — a full TensorCore stall. **This one is superseded by P3.** Concurrency
   exists; it needs one XLA program.

### Area 6 — algorithm

**Decision: exact. No approximation anywhere.** The score is computed at full precision over every
unmasked position and the select is an exact MSB-first radix.

- The output is order-free and tie-free: the oracle compares sorted index **sets**
  (`tests/kernels/test_dsv4_streamindex_topk.py:58-118`), at **zero tolerance**.
- **Commit `6142d5e7` was an exact→exact swap, not a retreat from approximation.** The pre-commit
  path was `jax.lax.approx_max_k(recall_target=1.0)`, which is exact. PR #724 reports +9.3 % output
  tok/s (2320.69 → 2537.45), median TPOT 292.05 → 276.99 ms, xprof decode top-k 430 → 33.768 µs
  `AgentX`. **`report.md` §8's "the authors moved away from approximate" is false as stated** and
  must not be used to argue against approximation.
- **The real barrier to any approximation is that its gate cannot be run.** `area6` constraint 5:
  **no `baselines/eval/glm-5.2-*` file exists** — 32 eval baselines exist for other models, none for
  this one. Tier 3 of the gate (`check_regression.py --mode eval`, `EVAL_TOLERANCE="0.025"`,
  `glm-5.2-fp8-dp16-ep.sh:80`) would require creating a checked-in baseline from an unbudgeted
  model-level run. Tiers 1 and 2 (index-set recall min-row ≥ 0.99; score-mass recall ≥ 0.999) are
  runnable today.
- `weights = weights_proj(hidden_states) * q_scale * softmax_scale * n_head**-0.5` — the **sign is
  activation-dependent and not decidable from the checkpoint**, so the `relu(x)*w → relu(x*w)` fold
  is not provable offline. `area6 §G` therefore drops it: it is an approximation, its prize is
  ≤5 % in a disowned probe and ~2 % in area 4's pricing, and there is no cheaper exact form.
- Exact restructurings already enumerated by `area6`: **E1** (bound stage B's DMA by
  `eff_row_lengths`) is the best-evidenced item in step 2 with four independent routes and is worth
  **≈0.7 µs = 1 % of stage B at nightly decode** and ≈11 µs = 2 % at the last prefill chunk — real,
  tiny, and it should be taken because it is nearly free. **E4** (fix the early-exit predicate) is
  worth almost nothing: the predicate is false at ISL 8192, and `area7`'s CSE dump independently
  confirms `branch_1_fun` (the `_fast_path`) has **zero rows**. **E5** (`*w` + `sum_h` on the MXU)
  is named so it is not proposed: it does not pay at decode. **E7** (cap `_pick_partition`'s `p` by
  a cost model) has no lever at nightly, where `p = 1` already — but Option B needs it.

### Area 7 — precomputation

**Decision: essentially nothing is precomputed; every derived quantity is rebuilt per call, and
several per grid step.** The full inventory is `briefs/area7-precomputation.md` §B.

`Emit` — for every quantity the kernel needs that is not an operand: the axis it is invariant along,
the level it is computed at, the space and bytes it occupies, and the rebuild cost:

| quantity | invariant along | level | space / bytes | rebuilds per step |
| --- | --- | --- | --- | ---: |
| `scores_init` (`-inf` fill) | nothing — it is the output buffer | 3 (per call in XLA) | HBM, 2.62 MB | 21 (once per layer, necessarily) |
| token→seq maps (×3) | **the layer axis** — identical in all 21 | 3 | SMEM/HBM, small | **21 (C2)** |
| `sem_ids`, `bo_sz` | the grid axis — they are level-4 state | **shipped at level 3** | SMEM | **63 (C3)** |
| `kv_len_max`, `num_bkv` | the grid step | 5 (per grid step) | SMEM | **336 (C4)** |
| `slice_len`, `b_pad`, `num_waves` | **the layer axis** | 3, inside `sparsecore_topk` | SMEM | **21 (C5 — the largest)** |
| `eff_row_lengths` | the layer axis | 3 | HBM, 256 B | 21; the reduce itself is **2.5 µs/step `AgentX`** — nearly free |

**The CSE check is closed and it is the structural finding of the whole exercise.**
`measured (j-a71a648c rank 0)`, `hlo_stats`, 16554/16554 rows kept: **every metadata row is an exact
multiple of 21. Not one instruction is shared between the 21 indexer layers.** The mechanism:
`_common_path` is a `lax.cond` branch, XLA compiles each call site to its own computation
(`region_44.52_spmd`, … 86 distinct `region_*_spmd`, 84 `conditional`s), and CSE operates *within* a
computation. **A compiler flag cannot fix this. Hoisting must happen above the `cond`** — at level 3
in the caller, or on the host.

Cost of the class, `AgentX`: seven metadata scopes = 19,131.1 µs = **265.7 µs per step per core =
1.1 %** of the 23.675 ms module step and **5.2 %** of the three indexer kernels. But
**`slice` (97.4) + `concatenate` (39.7) + `gather` (88.7) = 225.8 of the 265.7 µs** is scalar
plumbing wrapped around the `pallas_call`s, **so C3 outranks C2.** `C6` (`seq_lens` staleness) is a
performance bug only, with a one-line host fix: `self.seq_lens_np[num_reqs:] = 0`.

**Caveat that governs Option D: these microseconds are `AgentX`. The instruction counts and the
21× multiplicity are structural and carry over; the µs do not.**

### The incumbent as a step-3 candidate

The incumbent is a candidate and is allowed to win. Its measured position: **decode 39.9 % of
`F_struct`, prefill 80.4 %.** It also carries a *measured* single-knob improvement that requires no
design change at all — `bkv_p` 2→3, P7a, −11.9 % of the decode call. It is entered in §3.5 as
**Candidate 0** and any option that does not beat 221.4 → 195 µs at decode has lost to it.

### What the incumbent violates in `code-style.md`, and why it matters to the file plans

`briefs/area8-layout-code-style.md` owns the file plan. The incumbent violates:

- **§1 four-file decomposition and the 250-line threshold.** `streamindex_topk.py` is **865 lines**
  (3.5×), `sparsecore_topk.py` **457** (1.8×).
- **§4 ref bundle at eight refs.** `kernel` takes **16** positional ref slots; `_topk_kernel` takes
  **9**. Repo precedent for the fix exists: `gdn/v3/memory_ref.py`, `compress_and_store/buffered_ref.py`.
- **§4 jaxtyping** — absent.
- **§11 `Config`** — `pydantic` appears in **zero** files under `src/vllm_torchtpu/kernels/`.
- **§13 hardcoded hardware constants.** `DEFAULT_VMEM_LIMIT_BYTES = 100*1024*1024` at
  `streamindex_topk.py:28`; the file **never calls `get_tpu_info()`**. `sparsecore_topk.py` hardcodes
  `LANES=16`, `MAX_SLICE_WORDS=64*1024`, `num_subcores=16`, `num_sc_cores=2` at `:85-86` **while
  `_sc_topk_direct` reads the same values from `info.sparse_core` at `:315-331`.**
- **§3 no duplicated derivation.** `bkv_sz`, `bkv_sz_per_kv_packing` and `num_sublanes_bkv` are each
  computed **twice** — at `:106-108` and again at `:636-639`.
- **§1 dimension glossary** absent; **§8 docstring** — `sparsecore_topk.py`'s is 16 lines.

**And the caution that constrains every file plan below:** `area8` establishes that **no file
anywhere in `src/vllm_torchtpu/` is named `api.py`, `base.py`, or `pallas_mosaic_tpu*.py`.** The
actual repo convention is a package of `config.py` / `kernel.py` / `wrapper.py` / `memory_ref.py` /
`compute.py` / `metadata.py` / `vmem_ldst.py` (`gdn/v3/` has 11 files, `compress_and_store/` 9, plus
`sparse_core/ragged_gather_reduce_v2/`, `mla/v2/`, `fused_moe/v2/`). **Whichever way an option goes
it departs from something, and it must declare which.** Every file plan below therefore states its
naming convention explicitly, and every one of them chooses the **repo convention**, not
`code-style.md`'s names, because the repo convention is what a reviewer will diff against.

`area8 §10`'s escape hatch (dynamic-offset DMA + in-kernel metadata + selective triple buffering)
is live for this kernel. Opting out of the standard decomposition is legitimate **but must be
declared in the module docstring**; "the undeclared middle is not."

---

## 3. The options

### 3.0 What separates them

Four options. Each decides at least one area differently from the incumbent **and** from every
other option — none is another's tunable variant. A fifth candidate, the incumbent plus its
measured `bkv_p` sweep, is entered as Candidate 0.

| | area it decides differently from the incumbent | phase it targets | bit-identical? | code risk |
| --- | --- | --- | --- | --- |
| **A — 128-byte K record** | **4** (operand layout), knock-on to 1 and 7 | **decode** | **yes** | medium — changes a cache layout and its writer |
| **C — head-major score layout** | **4** (intermediate layout) + **6** (reduction association) | **prefill** | **no** — f32 reassociation | **low** — ~35 lines, no new operands |
| **D — hoist metadata above the `cond`** | **7** (precomputation level) | both, decode more | **yes** | low, but caller-side XLA |
| **B — two-unit pipeline** | **3** (pipelining, C>1) + **1** (fusion boundary) + **5** (unit schedule) | **prefill**, weakly decode | **yes** (with `p` pinned) | **high** — restructures `_common_path` |

Pairwise, so no two can be collapsed: **A vs C** — A changes the bytes crossing HBM and leaves the
arithmetic untouched; C changes the arithmetic's register layout and leaves the bytes untouched.
**A vs D** — A is inside the kernel's operand contract; D is entirely outside the kernel, in the
XLA the caller emits. **B vs everything** — B is the only option that changes *when* work runs
rather than *how much* there is, and the only one that touches the unit schedule.

**All four are composable.** None of them contends for the same term. §3.6 gives the combined
prediction.

---

### 3.1 Option A — "128-byte K record" (rank 1)

**The change.** Split the indexer K cache's 129-byte record so the payload is exactly one lane tile
and the scale bytes ride along in the same contiguous DMA instead of forcing a 128-byte pad.

Today `cache_kv` is `uint8[num_blocks, 256, 4, 256]` — per page, 256 groups × `kv_packing` 4 ×
`lkv_dim` 256, where only 129 of the 256 minor bytes are used. The proposal:

```
cache_kv : uint8[num_blocks, 258, 4, 128]
             rows   0..255 (of the group axis) hold 1024 positions × 128 payload bytes
             rows 256..257                     hold 1024 positions ×   1 scale byte,
                                               packed 128 scales per lane row, 8 rows, 8 = 2×4
```

Arithmetic: 1024 payload rows + 8 scale rows = 1032 rows of 128 B = **132,096 B per page**, against
262,144 today. `1032 / 4 = 258` exactly, so the `kv_packing` 4-deep axis is preserved. The minor
dim is exactly 128 — one lane tile, satisfying `area4 §A4`. **The DMA count is unchanged**: still
one contiguous copy per (sequence, page).

This is the design decision that makes A work and it is the reason A is ranked first: **the obvious
formulation — a separate scale array — is wrong.** It doubles the DMA count (640 → 1280 per call,
half of them 1 KiB) and `area4 §A6` prices the DMA budget at 260 ns each; 640 extra small transfers
would consume the entire saving. Folding the scales into spare rows of the same page keeps the
granule at 129 KiB, above P6's 128 KiB efficient point, at one DMA per page.

#### The seven areas

| area | decision | vs incumbent |
| --- | --- | --- |
| **1 fusion boundary** | unchanged: three chained TC calls → one SC kernel | same |
| **2 partitioning** | unchanged grid; **granule 262,144 → 132,096 B**; traffic 176 → 92.8 MB | **changed (quantitatively)** |
| **3 pipelining** | unchanged depth-2 double buffering; `dominant_side` moves from 0.13–0.20 to **0.26–0.39** — still bytes, less decisively | same structure |
| **4 layout & precision** | **CHANGED.** See the operand table below. | **the option** |
| **5 compute units** | unchanged. **Nothing moves units.** The one addition is a second `.at[].set()` in the writer, on TensorCore, of 1 KB/step | same |
| **6 algorithm** | unchanged, exact. **Bit-identical** — the same 128 payload bytes and the same scale byte reach the same arithmetic. | same |
| **7 precomputation** | unchanged, except the scale-row index arithmetic `256 + (slot//128)//4, (slot//128)%4, slot%128` is a level-3 host computation, not per-grid-step | minor |

**Area 4 `Emit` — precision and layout of every operand, and the wasted fraction of each:**

| operand | dtype | shape | direction | wasted, before → after |
| --- | --- | --- | --- | --- |
| `cache_kv` | uint8 | `[num_blocks, 258, 4, 128]` | read | **50.4 % → 2.3 %** (129 useful of 132 moved) |
| all others | unchanged | | | unchanged |

**Which unit every piece of work lands on.** Stage A stays 100 % TensorCore; stage B stays 100 %
SparseCore. The scale byte is unpacked on the VPU exactly as it is today — the fetch changes, the
unpack does not. **No work moves between units.** The only new work is one extra scalar scatter per
step in `_streamidx_topk_jax`, on TensorCore, moving 1 KB — negligible against P0's measured
8192 B / 3.024 µs for the existing scatter.

#### Floor arithmetic and predicted achieved fraction

**Decode.** K bytes: `640 DMAs × 132,096 B = 84.54 MB` (from 167.77).

```
init      : 2.62 MB → 0.88 µs                                          → 0.88 µs
stage A   : bytes 84.54 + 2.62 = 87.16 MB → t_hbm 29.35 µs
            t_vpu 8.42 + t_mxu 13.50 = 21.92 µs
            max(29.35, 21.92)                                          → 29.35 µs
stage B   : max(1.06, 30.0)                                            → 30.00 µs
F_struct  = 0.88 + 29.35 + 30.00                                       = 60.23 µs
F_common  = max((84.54 + 0.52)/2.97e12 = 28.64, 21.92, 30.0)           = 30.00 µs
```

Predicted runtime. Stage A is HBM-limited (§1.5) at a granule that P6 measures retains 98.0 % of
contiguous bandwidth, so the byte factor transfers at `0.5115 / 0.98 = 0.522`:

- **optimistic** (bytes transfer fully, stage A → `0.522 × 109.2 = 57.0` µs, but floored by the
  55.7 µs of achieved VALU work it overlaps with): stage A ≈ 57 µs, call **221.4 − 52 = 169 µs**;
  round to **172 µs** to keep the residual honest → **−22.3 %**
- **conservative** (`area4 §A5`'s measured rule: 0.50× bytes → 0.74× time): stage A → 80.8 µs,
  call **193 µs** → **−12.8 %**

| | decode |
| --- | --- |
| predicted call | **172–193 µs**, central **182 µs** |
| vs baseline | **−13 % to −22 %** |
| clears the ±8.5 % instrument spread? | **yes, at both ends** |
| achieved fraction of `F_struct` | 60.23 / 182 = **33.1 %** (*falls* from 39.9 % — see the §1.6 trap; A shrinks the floor 32 % and the runtime 18 %) |
| achieved fraction of `F_common` | 30.00 / 182 = **16.5 %**, from 25.6 % |

**Prefill.** K is re-read 8× but is only 13.9 % of prefill traffic and prefill's stage A is
VPU-bound by 9.2×.

```
stage A   : bytes 10.58 (K) + 41.94 = 52.52 MB → t_hbm 17.68 µs
            t_vpu 133.7 + t_mxu 60.5 = 194.2 µs
            max(17.68, 194.2)                                          → 194.2 µs   unchanged
F_struct  = 14.12 + 194.2 + 481.0                                      = 689.3 µs   unchanged
```

| | prefill |
| --- | --- |
| predicted call | **850–857 µs** |
| vs baseline | **−0 % to −1 %** |
| clears the spread? | **no — A does nothing at prefill, and this is stated as a result, not hidden** |
| achieved fraction of `F_struct` | 689.3 / 853 = **80.8 %** |

#### Fallback gate

**Predicate:** `lkv_dim_packed != 132` at kernel-config time — i.e. the head dim is not 128, or
`page_size % 512 != 0` so the scale rows do not divide by `kv_packing`.
**Threshold:** exact structural match, no tolerance.
**Action:** fall back to the incumbent 256-byte layout, which stays in the tree as the
`supported_on`-rejected path. Both layouts coexist behind one `Config` field; there is no dynamic
switch and no runtime cost to carrying both.

There is **no accuracy fallback gate, deliberately**: the change is bit-identical, so the existing
zero-tolerance unit test is the whole gate.

#### File plan

Naming convention: **repo convention** (`config.py` / `kernel.py` / `wrapper.py` / …), not
`code-style.md`'s `api.py` / `base.py` / `pallas_mosaic_tpu.py`, per the `area8` caution. Declared.

| file | created / modified | what changes | ~lines |
| --- | --- | --- | ---: |
| `src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py` | modified | `forward_oot`: stop `torch.cat([k_quant.view(uint8), k_scale], dim=-1)`; pass the two tensors separately. `_streamidx_topk_jax`: two scatters — payload to `cache_kv[page, slot//4, slot%4, :]`, scale to `cache_kv[page, 256 + (slot//128)//4, (slot//128)%4, slot%128]`. `donate_argnums=(2,)` unchanged (still one array). Move `num_kv_pages_per_block=(2,2,2)` / `num_queries_per_block=(1,128,128)` out to the new `config.py`. | ~35 |
| `src/vllm_torchtpu/layers/vllm/attention.py` | modified | the indexer K-cache allocation: last dim 256 → 128, group axis 256 → 258. Needs a branch distinct from the MLA KV cache, which shares the `unsigned_align_to(kv_dim, 128)` path reached from `attention.py:1221-1264` / `mla/v2/kernel.py:101-106`. **P5 is what makes this safe: the indexer cache has exactly one writer and one reader.** | ~30 |
| `.../kernels/deepseek_v4/streamindex_topk.py` | modified | `_fetch_bkv`: copy extent `bkv_p * page_size_per_kv_packing` → `bkv_p * (page_size_per_kv_packing + 2)`. `load_bkv`: bitcast the payload from `[..., 0:128]`; read scales from the trailing 2 groups. `lkv_dim` derivation and the duplicated `bkv_sz` derivations at `:106-108` / `:636-639` collapse to one. | ~55 |
| `.../kernels/deepseek_v4/streamindex_topk/config.py` | **created** | `pydantic` `Config` carrying `bkv_p`, `bq_sz`, `seq_batch_size`, `record_bytes`, `vmem_limit_bytes` per `MlaCase`; `_get_heuristics_config`, `_get_autotuning_configs`, `supported_on` (the fallback predicate above). Reads `get_tpu_info()` instead of `DEFAULT_VMEM_LIMIT_BYTES = 100*1024*1024` at `:28`. Fixes `code-style.md` §11 and §13. | ~110 |
| `tests/kernels/test_dsv4_streamindex_topk.py` | modified | the oracle builds the packed cache; parametrise over both record layouts. **The whole suite must run, not only the microbenchmark** — the change touches the scale path that feeds the mask. | ~60 |
| `kernel_work/streamidx-topk/instruments/bench_kpack.py` | **created** | new file; `instruments/` existing contents are **read-only after step 1**. Sweeps `record_bytes ∈ {132, 256}` × `bkv_p ∈ {2, 3}` at nightly decode geometry. | ~120 |

---

### 3.2 Option C — "head-major score layout" (rank 2)

**The change.** Reorder the einsum's row axis from `(bq_sz × H)` to `(H × bq_sz)` so the `sum_h`
reduction runs over the *outermost* axis and lowers to whole-vreg adds instead of a cross-sublane
rotate tree. ~35 lines and no new operands.

Today `load_bq` produces `bq_stacked` shaped `(seq_batch, bq_sz * num_q_heads, head_dim)` — heads
minor. The einsum `bnd,bmd->bnm` yields `(seq_batch, bq_sz*H, bkv_sz)`, reshaped
`(seq_batch, bq_sz, H, bkv_sz)` and summed over axis 2. With `bkv_sz` on lanes, `H` has stride
one sublane row, so a 32-way reduction over *adjacent sublanes* lowers to `vrot.slane` by
1, 2, 4, 8, 16 — P7b's **8,960 `vrot.slane`, 22.6 % of all VALU ops** (§1.4.3).

With `(seq_batch, num_q_heads * bq_sz, head_dim)` — heads **major** — the reshape becomes
`(seq_batch, H, bq_sz, bkv_sz)` and the reduce is over axis 1. At prefill `bq_sz = 128`, so `H` has
a stride of 128 rows = **16 whole vregs**: 31 vreg-aligned `vadd`s per output tile, **zero
rotates**. `bq_weights_stacked` reorders to `(seq_batch, H, bq_sz)` correspondingly.

**Why this is not a variant of the incumbent:** it changes which axis the intermediate's sublanes
carry (area 4) *and* it changes the association order of an f32 sum (area 6). It is not a tunable.

#### The seven areas

| area | decision | vs incumbent |
| --- | --- | --- |
| **1 fusion boundary** | unchanged | same |
| **2 partitioning** | unchanged grid, unchanged traffic | same |
| **3 pipelining** | unchanged structure. Prefill `t_compute` per step falls ~20 %; `dominant_side` at prefill stays compute | same |
| **4 layout & precision** | **CHANGED** for the score intermediate: sublane axis carries `bq_sz`, not `H`. Every HBM operand's dtype and shape is **unchanged** — this is a register-layout change only | **the option (a)** |
| **5 compute units** | unchanged units. VPU work falls ~22.6 % at prefill. **Explicitly does NOT move `sum_h` to the MXU** — `area6 E5` / `area5 A7` name that sub-variant and it is declined: it is block-diagonal, it does not pay at decode, and it *would* be a unit decision | same |
| **6 algorithm** | **CHANGED.** Exact in real arithmetic, **not bit-identical** — the f32 reduction re-associates. See the gate. | **the option (b)** |
| **7 precomputation** | unchanged | same |

**Which unit every piece of work lands on.** Everything stays where it is: VPU → VPU, MXU → MXU
(the matmul's LHS row extent is `bq_sz*H = 4096` either way, so `mxu_cliff_factor` and `t_mxu` do
not move), SparseCore untouched. **Nothing moves units, and the one candidate transformation that
would (the MXU `sum_h`) is explicitly declined above.**

#### Floor arithmetic and predicted achieved fraction

C removes *inflation*, not modelled work, so **neither floor moves.** `F_struct` = 88.25 µs decode,
689.3 µs prefill; `F_common` = 56.66 / 481.0 µs.

**Prefill.** Stage A is VALU-saturated at 89.45 % `static (P7b)`, so removing 22.6 % of VALU ops
converts at that rate:

```
Δstage A = 335.1 × 22.6 % × 89.45 % = 67.7 µs
stage A  : 335.1 → 267.4 µs
call     : 857.3 → 789.6 µs
```

| | prefill |
| --- | --- |
| predicted call | **789 µs** |
| vs baseline | **−7.9 %** |
| clears the ±8.5 % spread? | **marginally no** — needs a paired microbenchmark, not an end-to-end A/B. Named in §5. |
| achieved fraction of `F_struct` | 689.3 / 789 = **87.4 %** |
| achieved fraction of `F_common` | 481.0 / 789 = **61.0 %** |

**87.4 % puts prefill past the skill's 80 % "do not write a kernel" bar with margin.** After C,
prefill's only remaining headroom is stage B's algorithm and Option B's overlap.

**Decode.** `bq_sz = 1`, so there is nothing to reorder: the 32 heads are the only sublane axis and
the reduce still needs ~96 rotates of ~1,056 modelled ops ≈ 9 %. And decode's achieved VALU
(55.7 µs) already hides under its 57.4 µs of HBM time.

| | decode |
| --- | --- |
| predicted call | **217–221 µs** |
| vs baseline | **−0 % to −2 %** |
| achieved fraction of `F_struct` | 88.25 / 219 = **40.3 %** |

#### Fallback gate

C is the only option that needs an accuracy gate, and the reason is subtle enough to state plainly:
**the output is an ordinal index set compared at zero tolerance, so an f32 reassociation at ~1e-7
relative can in principle flip an index at the rank-2048 boundary.** `area6 §D` flags exactly this
class. C is *not* an approximation — no work is dropped — so it does **not** need the tier-3 eval
gate that `area6` shows is blocked by the missing `baselines/eval/glm-5.2-*`.

- **Gate 1, and it may be the whole gate.** Run the existing suite unchanged
  (`tests/kernels/test_dsv4_streamindex_topk.py`, zero tolerance, sorted set compare). If it passes
  at the production shapes, C is settled and nothing further is needed.
- **Gate 2, if gate 1 fails on tie-adjacent rows.** Index-set recall against an f64 oracle,
  **min over rows ≥ 0.999**, at nightly decode and prefill shapes. This is `area6`'s tier 1 with a
  tightened threshold (0.999 rather than 0.99), justified because the perturbation is reassociation
  noise, not an algorithmic drop.
- **Predicate for falling back:** min-row index-set recall < 0.999 → revert to the
  `(bq_sz × H)` ordering. Both orderings sit behind one `Config` flag.

#### File plan

Naming convention: **no new package.** C is a surgical edit; creating a package for it would obscure
a 35-line diff. Declared as a deliberate departure from `code-style.md` §1.

| file | created / modified | what changes | ~lines |
| --- | --- | --- | ---: |
| `.../kernels/deepseek_v4/streamindex_topk.py` | modified | `prepare_q_inputs` / `load_bq`: emit `(seq_batch, H*bq_sz, D)`. `prepare_index_weights` / `load_bq_weights`: emit `(seq_batch, H, bq_sz)`. `compute_scores` (`:326-353`): reshape to `(seq_batch, H, bq_sz, bkv_sz)`, `.sum(axis=1)`, and re-index the `bq_pos_compressed` broadcast used by the causal mask. Add a `head_major: bool` switch. | ~35 |
| `.../kernels/deepseek_v4/streamindex_topk/config.py` | created (shared with A) | `head_major` field | ~5 |
| `tests/kernels/test_dsv4_streamindex_topk.py` | modified | parametrise `head_major`; add the f64-oracle recall assertion as gate 2 | ~45 |
| `kernel_work/streamidx-topk/instruments/bench_headmajor.py` | **created** | new file (`instruments/` is read-only). Prefill stage A alone, `bq_sz=128`, both orderings, best-of-7, plus an `/xprof-llo` capture of the op histogram to confirm `vrot.slane` → 0 | ~110 |

---

### 3.3 Option D — "hoist the metadata above the `cond`" (rank 3)

**The change.** `_common_path` is a `lax.cond` branch, so XLA compiles all 21 indexer layers'
metadata to 21 private computations and CSE — which works only *within* a computation — cannot
touch them. `measured (j-a71a648c)`: 16554/16554 `hlo_stats` rows kept, **every metadata row an
exact multiple of 21.** The fix is to compute the layer-invariant metadata once, at level 3 (per
call in XLA, in the caller) or on the host, **above** the `lax.cond`, and pass it in.

D is ranked third and it is ranked honestly: **its mechanism is measured and its magnitude at the
target shape is not.** It is entered because it is caller-side Python with no kernel risk, and
because two independent measurements say the residual bucket is big:

- the nightly decode residual is **44.8 µs = 20.2 % of the call** `derived (j-8b1e7acc)`, the
  second-largest bucket after stage A;
- stage B reads **67.4 µs in-model** against **48.22 µs (P3 arm B) / 52.12 µs (P2)** isolated —
  a ~15–19 µs per-layer gap that is plumbing, not SparseCore work.

#### The seven areas

| area | decision | vs incumbent |
| --- | --- | --- |
| **1 fusion boundary** | unchanged kernel boundary. The `lax.cond` around `_common_path` is **removed** — `area6 E4` and `area7` independently show the `_fast_path` never fires (`branch_1_fun` has **zero `hlo_stats` rows**), so the branch buys nothing and costs CSE across 21 layers | **changed (the wrapper's control flow)** |
| **2 partitioning** | unchanged | same |
| **3 pipelining** | unchanged | same |
| **4 layout & precision** | unchanged | same |
| **5 compute units** | unchanged | same |
| **6 algorithm** | unchanged, exact, **bit-identical** | same |
| **7 precomputation** | **CHANGED.** C3, C2 and C5 move from level 3-inside-the-cond to level 3-in-the-caller or the host. C6's one-line fix (`self.seq_lens_np[num_reqs:] = 0`) is taken. | **the option** |

**Area 7 `Emit` — after the change:**

| quantity | invariant along | level, after | space / bytes | rebuilds per step, before → after |
| --- | --- | --- | --- | --- |
| `sem_ids`, `bo_sz` (C3, ranked first) | the grid axis | **4 — computed in-kernel into scratch** | SMEM, tens of bytes | **63 → 0** |
| token→seq maps ×3 (C2) | the layer axis | **host, once per forward** | HBM, small | **21 → 1** |
| `slice_len`, `b_pad`, `num_waves` (C5) | the layer axis | **host (they are pure functions of `b`, `n`, `k`)** | SMEM | **21 → 1** |
| `kv_len_max`, `num_bkv` (C4) | the grid step | unchanged at 5 | SMEM | 336 → 336 |
| `seq_lens` staleness (C6) | — | host | — | one-line fix |
| `eff_row_lengths` | the layer axis | unchanged — the reduce is 2.5 µs/step `AgentX`, nearly free | HBM, 256 B | 21 → 21 |

C3 outranks C2 on `area7`'s own attribution: `slice` (97.4) + `concatenate` (39.7) + `gather` (88.7)
= **225.8 of the 265.7 µs/step** `AgentX` is scalar plumbing around the `pallas_call`s.

**Which unit every piece of work lands on.** Work moves **off the TPU entirely** — from TensorCore
scalar units to the host, for the level-3 and host-level quantities. That is the correct direction
(it removes work from the critical path) but it is a real unit decision and it has a real risk:
host-computed metadata must be ready before the step is enqueued, or it stalls dispatch. C6's fix
(`self.seq_lens_np[num_reqs:] = 0`) is already host-side and already proven safe.

#### Floor arithmetic and predicted achieved fraction

Neither floor moves — the floors count kernel work and D removes plumbing.
`F_struct` = 88.25 / 689.3 µs; `F_common` = 56.66 / 481.0 µs.

**The prediction is a range and the range is wide, because the composition of the 44.8 µs residual
at nightly is not measured.** Bounds:

- **lower bound 0 µs** — if the nightly residual is dominated by the `-inf` init (per-layer by
  necessity), the `jnp.pad`, the reshape and the `topk_idxs[:q, :k]` slice, none of which hoist.
- **upper bound 44.8 µs** — the whole residual.
- **a defensible middle**, scaling `area7`'s AgentX attribution: 225.8/265.7 = 85 % of the metadata
  is hoistable plumbing, and 20/21 of each hoistable quantity is redundant, giving
  `0.85 × 20/21 = 81 %`. If half the nightly residual is that plumbing:
  `0.5 × 44.8 × 0.81 = 18.1 µs = 8.2 %` of the call.

| | decode | prefill |
| --- | --- | --- |
| predicted call | **203–221 µs**, central **203 µs** | **840–857 µs**, central 841 µs |
| vs baseline | **−0 % to −20 %**, central **−8.2 %** | −0 % to −4.8 %, central −1.9 % |
| clears the ±8.5 % spread? | **only at the upper half of the range — this is why D is not ranked above C** | no |
| achieved fraction of `F_struct` | 88.25 / 203 = **43.5 %** (up from 39.9 %) | 689.3 / 841 = **82.0 %** |
| achieved fraction of `F_common` | 56.66 / 203 = **27.9 %** | 481.0 / 841 = **57.2 %** |

**What pins the range, and it is not a new hardware job:** an `hlo_stats` pull at the *nightly*
shape rather than AgentX. `report.md` §6 already has the instrument; the AgentX dump exists because
that is the profile that was to hand, not because nightly is unobtainable. **Named as a step-3
prerequisite for D and only for D** (§5).

#### Fallback gate

**Predicate:** any hoisted quantity is not in fact layer-invariant at runtime — detectable as a
correctness failure, not a performance one.
**Threshold:** the existing zero-tolerance unit test, plus an assertion in the caller that the
hoisted `slice_len` / `b_pad` / `num_waves` match what `sparsecore_topk` would have computed, run
under a debug flag.
**Action:** revert to per-layer computation behind a `Config` flag.

The `lax.cond` removal has its own gate: **`enable_early_exit` must remain honoured**. `area6 E4`
and `area7` agree the fast path never fires at ISL 8192, but "never at this shape" is not "never".
The removal is therefore: keep `enable_early_exit` as a **trace-time Python** predicate on
`max_model_len // compression_ratio <= k`, which is level-1 information, rather than a runtime
`lax.cond`. That preserves the behaviour exactly and still gives CSE a single computation.

#### File plan

| file | created / modified | what changes | ~lines |
| --- | --- | --- | ---: |
| `.../kernels/deepseek_v4/streamindex_topk.py` | modified | `streamindex_topk`: replace the `lax.cond(jnp.max(seq_lens)//compression_ratio <= k, _fast_path, _common_path, None)` at `:~751` with a **trace-time** Python `if` on the static `max_model_len`. Accept the hoisted metadata as arguments. Drop the duplicated derivations at `:106-108` / `:636-639`. | ~60 |
| `.../kernels/deepseek_v4/streamindex_topk/metadata.py` | **created** | the layer-invariant metadata builder: token→seq maps, `slice_len` / `b_pad` / `num_waves`, `eff_row_lengths`. Repo precedent: `gdn/v3/metadata.py`. | ~120 |
| `.../kernels/deepseek_v4/sparsecore_topk.py` | modified | accept precomputed `slice_len` / `b_pad` / `num_waves` instead of recomputing; read `LANES`, `MAX_SLICE_WORDS`, `num_subcores`, `num_sc_cores` from `info.sparse_core` at `:85-86` the way `_sc_topk_direct` already does at `:315-331` (fixes `code-style.md` §13) | ~35 |
| `src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py` | modified | build the metadata once per forward and thread it through; apply C6's `self.seq_lens_np[num_reqs:] = 0` | ~30 |
| `tests/kernels/test_dsv4_streamindex_topk.py` | modified | assert the hoisted metadata equals the per-layer computation under a debug flag | ~30 |

---

### 3.4 Option B — "two-unit pipeline" (rank 4)

**The change.** Split stage A and stage B into `G` chunks over the row (sequence at decode, query
token at prefill) axis, interleaved inside **one** `jax.jit` so that stage B of chunk `g−1` runs on
SparseCore while stage A of chunk `g` runs on TensorCore.

P3 makes this legal and constrains it hard: **41.6 %** of the SparseCore arm is hidden when both
custom calls sit in one XLA program; **−1.8 %** across two executables; **−11.5 %** across two
Python threads. **The concurrency is a compiler property, not a runtime one.**

#### The constraint that decides B's rank, and it came out of source, not the probes

`_pick_partition(b, n, k)` in `sparsecore_topk.py`:

```
p_min = cdiv(n, MAX_SLICE_WORDS=65536)
max_p = 32 // b   if b < 32   else 1
raise p while  p*2 <= max_p  and  n//(p*2) >= k  and  >= LANES  and  n % (p*2) == 0
```

At nightly decode, chunking to `b = 64/G`:

| G | b per chunk | `max_p` | `p` | consequence |
| --- | ---: | ---: | ---: | --- |
| 1 | 64 | 1 | 1 | incumbent, 2 waves, 32/32 subcores |
| **2** | **32** | **1** | **1** | 1 wave, 32/32 subcores — **the flat point** |
| 4 | 16 | 2 | **2** | `n//2 = 5120 ≥ k`, `10240 % 2 == 0` → **two-stage path + two `jnp.take_along_axis` gathers**, priced by `area5` at **≥ +27 µs measured** at nightly width |

**So `G = 2` is a hard cap at decode**, and it is a cap that P3 could not see because P3 ran one
chunk. Forcing `p = 1` at `b = 16` does not rescue it: `num_waves = cdiv(16,32) = 1` with **16/32
subcores idle**, so four chunks cost 4 × the per-chunk time instead of 2 × — strictly worse.

**This is the unit decision in disguise the design has to state:** at `G = 4`, `_pick_partition`
silently moves two gathers from SparseCore back onto TensorCore. B must pin `p = 1` and refuse
`b % 32 != 0`, which is `area6 E7` ("cap `p` by a cost model") used as a guard rather than a lever.

At prefill, `b = 1024` and `b ≥ 32` for every `G ≤ 32`, so `max_p = 1` and `p = 1` unconditionally,
with `num_waves = cdiv(256, 32) = 8` at `G = 4` and full occupancy. **Prefill chunks freely.**

#### The seven areas

| area | decision | vs incumbent |
| --- | --- | --- |
| **1 fusion boundary** | **CHANGED.** Stage A and stage B are interleaved inside one XLA program. The kernel boundary and the HBM round trip both survive — **B does not fuse the kernels, it fuses the schedule.** | **changed** |
| **2 partitioning** | **CHANGED.** Grid `(16,)` → `G` grids of `(16/G,)`. Launches per step: 63 TC + 21 SC → **84 TC + 42 SC** at decode. Traffic unchanged at 176 MB. | changed |
| **3 pipelining** | **CHANGED — this is the option.** `tc_sc_overlap` goes from `N/A because C = 1` to `C = 2` (decode) / `C = 4` (prefill). See the field-12 emit below. | **the option** |
| **4 layout & precision** | unchanged | same |
| **5 compute units** | **CHANGED.** `tc_stall` goes from *all of `t_sc`* to `t_sc / G`. Subcore occupancy must stay 32/32, which is what pins `p = 1` and `G ≤ 2` at decode. | changed |
| **6 algorithm** | unchanged, exact, **bit-identical with `p` pinned** — each row's top-k is independent of every other row's | same |
| **7 precomputation** | `slice_len` etc. are now computed `G` times per layer instead of once; **B makes D's problem `G`× worse and should be taken with D, not instead of it** | worse |

**Area 3 `Emit` field 12 — `tc_sc_overlap`:**

| field | decode, `G = 2` | prefill, `G = 4` |
| --- | --- | --- |
| chunk count `C` | 2 | 4 |
| per-chunk `A/C` | 54.6 µs | 83.8 µs |
| per-chunk `B/C` | 24.1 µs (from P3's isolated 48.22, halved; `b = 32` is the flat point so per-row cost does not rise) | 120.25 µs |
| schedule `A/C + (C−1)·max(A/C, B/C) + B/C` | 54.6 + 54.6 + 24.1 = **133.3 µs** | 83.8 + 3×120.25 + 120.25 = **564.8 µs** |
| serial comparison | 157.4 µs | 816.1 µs |
| fixed `sparsecore_topk` cost, charged `C` times | 1 extra SC launch ≈ **3.0 µs `assumed`** | 3 extra ≈ 9.0 µs `assumed` |
| extra TC launches | 1 × 2.86 µs grid-0 intercept `measured (P4)` = **2.9 µs** | 3 × 2.86 = 8.6 µs |
| overlap `measured` or `assumed` | **`measured (P3)` at 41.6 %, but for two kernels with no data dependency — see §4** | same |

**Which unit every piece of work lands on.** Nothing moves units *provided `p` is pinned to 1*.
At `G = 4` decode it would, silently, and that is why the guard is part of the design and not a
detail.

#### Floor arithmetic and predicted achieved fraction

**Decode, `G = 2`.**

```
F_struct = init 0.88
         + A/2 (max(57.37/2, 21.92/2) = 28.69)
         + max(A/2 = 28.69, B/2 = 15.0)
         + B/2 (15.0)
         = 0.88 + 28.69 + 28.69 + 15.00                                = 73.26 µs
```

Predicted runtime:

```
ideal saving        = min(A/C, B/C) × (C−1) = 24.1 × 1        = 24.1 µs
realised at P3 41.6 %                       = 0.416 × 24.1    = 10.0 µs
extra launches      = 2.9 (TC, measured) + 3.0 (SC, assumed)  = −5.9 µs
net                                                            = +4.1 µs
```

| | decode |
| --- | --- |
| predicted call | **217 µs** |
| vs baseline | **−1.9 %** |
| clears the ±8.5 % spread? | **no. Not measurable end to end.** |
| achieved fraction of `F_struct` | 73.26 / 217 = **33.8 %** |
| achieved fraction of `F_common` | 56.66 / 217 = **26.1 %** |

**Prefill, `G = 4`.**

```
F_struct = 14.12 + 48.55 + 3 × max(48.55, 120.25) + 120.25       = 543.7 µs
ideal saving = min(A/C, B/C) × (C−1) = 83.8 × 3                  = 251.4 µs
realised at 41.6 %                                               = 104.6 µs
extra launches = 8.6 (TC) + 9.0 (SC, assumed)                    = −17.6 µs
net                                                              = +87.0 µs
```

| | prefill |
| --- | --- |
| predicted call | **770 µs** |
| vs baseline | **−10.2 %** |
| clears the ±8.5 % spread? | **yes, just** |
| achieved fraction of `F_struct` | 543.7 / 770 = **70.6 %** |
| achieved fraction of `F_common` | 481.0 / 770 = **62.5 %** |

**B is a prefill option that happens to be expressible at decode.** And prefill is where B's case is
strongest for a reason worth naming: **stage B at prefill is at 481.0 measured against a 481.0
modelled floor. When a stage is at its own floor, the only thing left to do is hide it.**

#### Fallback gate

**Predicate 1 (correctness, hard):** `b % 32 != 0` for any chunk, or `_pick_partition` returns
`p != 1`. **Action: refuse to chunk; fall back to `G = 1`.** This is not a performance guard, it is
the guard that stops the design silently relocating two gathers to TensorCore.

**Predicate 2 (performance):** measured overlap fraction on the real dependency < **20 %**
(half of P3's 41.6 %). At that point the extra launch cost exceeds the saving at decode
(`0.20 × 24.1 = 4.8 < 5.9`). **Action: `G = 1` at decode, keep `G = 4` at prefill**, where the
break-even is `17.6 / 251.4 = 7.0 %`.

**Predicate 3 (structural):** if XLA serialises the chunks on the aliased score buffer rather than
on the disjoint slices, the measured overlap is 0 and B is net negative everywhere.
**This is B's flip assumption — see §4.**

#### File plan

Naming convention: **repo convention**, and B is the option that forces the full package, because
`area8 §1` is explicit that where stage B falls is a design decision the plan must state. B keeps
stage B as its own operation — **it schedules the two, it does not fuse them** — so **two file sets
survive** and the count stays as it is. An option that *fused* stage B into stage A would collapse
these to one set; B does not, and the surviving two-set structure is the signal that B sits at the
incumbent's fusion boundary and only moves the schedule.

| file | created / modified | what changes | ~lines |
| --- | --- | --- | ---: |
| `.../kernels/deepseek_v4/streamindex_topk.py` | modified | `_common_path` restructured: the decode-batched `run_scores_kernel` splits into `G` calls over disjoint sequence ranges writing disjoint row slices; `sparsecore_topk` is called per chunk; results concatenated. **All inside the one existing `jax.jit`** — P3 says this is load-bearing. The `input_output_aliases={9:0}` chain must be re-expressed so chunks do not alias each other. | ~130 |
| `.../kernels/deepseek_v4/streamindex_topk/wrapper.py` | **created** | the chunk scheduler and the `G` heuristic (`G = 2` decode, `G = 4` prefill), plus the predicate-1 guard | ~90 |
| `.../kernels/deepseek_v4/sparsecore_topk.py` | modified | accept an explicit `p` override; `raise NotImplementedError` when a chunked call would give `p != 1` or `b % 32 != 0` (`area6 E7` as a guard) | ~25 |
| `src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py` | not modified | `G` is internal to the kernel package | 0 |
| `tests/kernels/test_dsv4_streamindex_topk.py` | modified | parametrise `G ∈ {1,2,4}`; assert the index **set** is identical for every `G` — it must be bit-identical, unlike Option C | ~40 |
| `kernel_work/streamidx-topk/instruments/bench_chunked_pipeline.py` | **created** | `G` sweep with the real cross-chunk dependency, i.e. arm B consumes arm A's actual output — **the thing P3 deliberately excluded** | ~150 |

---

### 3.5 Candidate 0 — the incumbent, plus its measured `bkv_p` sweep

Not an option: it decides no area differently. It is entered because it is the bar.

`num_kv_pages_per_block` 2 → 3 takes `num_bkv` from 5 to 3 (40 % fewer inner iterations) and KV
bytes from 167.8 to 151.0 MB. **`measured (P7a)`: stage A −19.99 %, = −11.9 % of the decode call.**
`area2 §Correction` prices its expected value at ≈2.5–3 % from bytes alone, so most of the measured
19.99 % is the iteration-count term, not the byte term — which is why §1.5 concludes it is additive
with Option A rather than a substitute.

Also nearly free and worth taking with any option: **`area6 E1`** — bound stage B's DMA by
`eff_row_lengths` instead of the padded `slice_len`. Four independent routes agree; worth ≈0.7 µs
(1 % of stage B) at nightly decode and ≈11 µs (2 %) at the last prefill chunk. And **C6**, the
one-line `self.seq_lens_np[num_reqs:] = 0` host fix.

| | decode | prefill |
| --- | --- | --- |
| predicted call | **195 µs** `measured` | 857 µs (`bkv_p` is a decode knob at these shapes) |
| vs baseline | **−11.9 %** | −0 % |
| achieved fraction of `F_struct` | 88.25 / 195 = **45.3 %** | 80.4 % |

`area2 §Correction`'s re-ranked levers, for completeness, since they are the sweep's siblings:
predicate the tail bkv block (≈25 MB, the largest area-2 lever), `bkv_p = 3`, elide empty launches,
sort by length, swap the prefill loops. **The first two are substitutes** — together they reach
142.6 MB, not the sum of their individual savings. Note `area6 E3` (sort by length) has a cost the
brief names: it breaks `start_send_scores`' single contiguous DMA into 4.

### 3.6 The ranking

| rank | option | decode | prefill | evidence class of the *mechanism* | evidence class of the *magnitude* |
| ---: | --- | ---: | ---: | --- | --- |
| **1** | **A — 128-byte K record** | **−13 % to −22 %** | −0 to −1 % | `measured` (P5 ownership, P6 byte-proportionality) | `modelled` from measured bytes + P6's transfer rate |
| **2** | **C — head-major score layout** | −0 to −2 % | **−7.9 %** | `static` (P7b op histogram) + source | `modelled` from a static histogram |
| **3** | **D — hoist above the `cond`** | −0 to −20 %, central −8.2 % | −0 to −4.8 % | **`measured` (j-a71a648c) and unambiguous** | **unmeasured at the target shape — the widest range here** |
| **4** | **B — two-unit pipeline** | −1.9 % | **−10.2 %** | `measured` (P3), but for a dependency-free pair | `modelled`, and the 41.6 % may not survive §4 |
| **0** | incumbent + `bkv_p = 3` | **−11.9 % `measured`** | −0 % | — | **`measured`** |

**Why A over C** despite C being the cheaper change: decode dominates serving, decode is at 39.9 %
of its floor and prefill is at 80.4 %, and A's mechanism rests on two hardware measurements (P5, P6)
while C's rests on a static instruction histogram.

**Why C over D** despite D having the better-measured mechanism: C's magnitude is bounded and D's is
not. D could be worth 20 % or 0 %, and until the nightly `hlo_stats` pull lands, "central −8.2 %"
is a scaled AgentX figure, which §0 says is exactly the thing not to trust in microseconds.

**Why B last**: it is the largest code change, its decode prize does not clear the instrument's
noise floor, and `_pick_partition`'s `b < 32` cliff caps `G` at 2 at decode — a constraint P3 could
not see.

**Composition.** None of the four contends for the same term: A moves HBM bytes, C moves VALU ops,
D moves host/scalar plumbing, B moves the unit schedule. Taken together with Candidate 0:

```
decode  : 221.4 → A (−36 central) → Candidate 0 (−22, additive per §1.5) → D (−16 central)
          → C (−3) → B (+4)                          ≈ 148 µs, −33 %
prefill : 857.3 → C (−68) → B (−87) → D (−16) → A (−5) ≈ 681 µs, −21 %
```

These composed figures are **`modelled` sums of individually `modelled` deltas and they will not
compose linearly** — A removes HBM time that C's VALU savings were hiding under, and every option
raises the others' achieved fraction of the floor while lowering their absolute prize.
`F_common` bounds the composition: decode cannot go below **56.66 µs** and prefill cannot go below
**481.0 µs** without changing stage B's algorithm.

**The honest single-sentence recommendation: take A and Candidate 0 at decode and C at prefill.
That is three changes, ~130 lines of real diff, `modelled` at −23 % decode and −8 % prefill, and it
leaves prefill above the skill's 80 %-of-floor bar with only stage B's algorithm left.**

---

## 4. The assumptions that would flip the ranking

Named for the top two, plus B's because it is the one that could make B worthless rather than
merely fourth. **§2 is not on this list, and that is deliberate** — §1.5 shows both of its answers
predict the same decode total to within one point. It changes the attribution, not the plan.

### Flip 1 — the one that decides rank 1. **Does P6's byte-proportionality transfer to `_fetch_bkv`?**

P6 measured a **flat array read with a static stride**: charge ratio 1.02, 98.0 % of contiguous
bandwidth at a 128 KiB granule. `_fetch_bkv` does something materially different — for each
`batch_idx` in `range(seq_batch_size)` and each `i` in `range(bkv_p)` it computes

```
page_idx          = min(page_indices_offset + i, num_page_indices - 1)
safe_page_offset  = min(page_indices_ref[page_idx] * page_size_per_kv_packing,
                        max(0, max_hbm_pages - page_size_per_kv_packing))
```

and issues one `make_async_copy` at that **dynamically clamped, page-table-indirect** offset.

**If that DMA's cost is set by descriptor issue and address indirection rather than by payload
bytes** — which is exactly how P7a's "stage A moves at near-constant bytes when `bkv_p` changes"
reads on the iteration axis — then halving the payload buys single-digit percent, Option A
collapses below the ±8.5 % spread, and the ranking becomes:

> **Candidate 0 (`bkv_p`, measured) > C > B > A.**

This is the flip. It is testable cheaply and it is named as A's step-3 prerequisite in §5.

### Flip 2 — the one that decides rank 2 vs rank 3. **Does P7b's static op histogram track the dynamic mix, and are the 8,960 `vrot.slane` attributable to the row ordering?**

C's whole prize is 8,960 `vrot.slane` out of 39,712 VALU ops, and both numbers are `static`. Two
ways it fails:

1. **The static mix is not the dynamic mix.** For a straight-line inner loop it should be, and P1's
   measured VALU 50.97 % against its static 62.08 % at decode is consistent. But this is an
   inference, not a measurement.
2. **The rotates survive the reorder.** They may be Mosaic's lane-shuffling for the
   `vpop.f32.mrb[*]` drain out of the MXU result buffer, not the head reduction — P1 §2.1 finds
   both in the same 815-bundle run and labels the whole thing `mem_stall_mxu_idle`. If the rotates
   belong to the drain, reordering the rows moves nothing.

If either holds, C is worth ≈0 and **D takes rank 2** — D's mechanism is `measured` and only its
magnitude is open, which is the better of the two uncertainties to be left holding.

### Flip 3 — the one that could make B worthless. **Does P3's 41.6 % survive a real cross-iteration dependency?**

P3 is explicit that it excluded this: "arm B is fed a pre-made score matrix that arm A never
writes… A real pipeline would have arm B consuming arm A's previous-group output through HBM, which
adds a dependency this probe deliberately excludes." And the incumbent already demonstrates XLA
serialising on a buffer rather than on slices: `area1 §A8` shows the `input_output_aliases={9:0}`
chain prevents any concurrency among the three stage-A calls even though at nightly decode two of
them have empty ranges.

If XLA serialises chunk `g`'s stage B against chunk `g+1`'s stage A on the score buffer, B's overlap
is **0** and B is net **−5.9 µs worse than baseline at decode** and **−17.6 µs worse at prefill**
from the extra launches alone. B does not merely drop a rank; it becomes a regression.

Conversely, if the dependency is clean and the overlap holds at 41.6 %, B's prefill figure of
−10.2 % is the largest single prefill prize in the set and **B moves to rank 2 for prefill-heavy
workloads**, ahead of C.

### One flip that is *not* live, and why

"Stage B might be replaceable with something faster than the radix select." `area5 §B`'s issue model
gives `t_sc` = 481 µs at prefill against 481.0 measured — stage B is at its modelled floor *for that
algorithm*. Replacing the algorithm is a legitimate area-6 question and it is the largest remaining
prefill prize after C. **It is not an option here because `area6` constraint 5 shows its gate cannot
be run**: no `baselines/eval/glm-5.2-*` exists, and creating one is an unbudgeted model-level run
plus a new checked-in baseline. Named as a step-3 prerequisite in §5 and not designed.

---

## 5. What step 3 would measure, and with which instrument

Everything below runs on the 4-chip pod through `/run-command-in-cdk` or `/xprof-llo`. **None of it
is an end-to-end benchmark**, and none of it goes to `/log-experiment` — `/xprof-llo`'s dump-side
output is static compile-time analysis and a fanned-out capture's wall-clock is contended.

### Step-3 prerequisites — run these *before* building anything

| # | question | instrument | gates |
| --- | --- | --- | --- |
| **R1** | **Flip 1.** Does halving the payload of a page-table-indirect, dynamically-clamped `make_async_copy` halve its time? | a new arm in `instruments/bench_kpack.py` **replicating `_fetch_bkv`'s actual DMA form** — indirect offsets from a page table, `min()`-clamped — at 262,144 B and 132,096 B, same DMA count. P6's harness is the template; its *shape* is what must change. | **A.** If the charge ratio is ≥1.5 (i.e. halving bytes saves ≤33 % of time), A drops to rank 4. |
| **R2** | **D's magnitude.** What is the nightly (not AgentX) composition of the 44.8 µs residual? | `hlo_stats` at the nightly shape — the same pull as `j-a71a648c`, different config. **No new probe design needed; the instrument exists.** | **D.** Pins a 0–20 % range to a number. |
| **R3** | **Flip 2.** Are the 8,960 `vrot.slane` the head reduction or the MXU drain? | `/xprof-llo` on the *unmodified* prefill kernel, reading the bundle range `0x7ac`–`0xada` that P1 §2.1 already isolated, and attributing each `vrot.slane` to its producer. Static, but it answers the attribution question directly. | **C.** |

### Per-option measurement plan

| option | what step 3 measures | instrument | pass criterion |
| --- | --- | --- | --- |
| **A** | stage-A decode time at `record_bytes ∈ {132, 256}` × `bkv_p ∈ {2, 3}`, best-of-7, same job | `instruments/bench_kpack.py` (**new file** — `instruments/` is read-only after step 1) | stage A ≤ 85 µs at (132, 2). Below 92 µs is the conservative bound; above 100 µs means R1 was answered wrongly. |
| **A** | correctness | `tests/kernels/test_dsv4_streamindex_topk.py`, unmodified, via `/run-command-in-cdk` | **zero-tolerance pass. A is bit-identical; anything else is a bug, not a tolerance question.** |
| **C** | prefill stage-A time, both row orderings, `bq_sz = 128` | `instruments/bench_headmajor.py` (**new file**) | stage A ≤ 280 µs (from 335.1) |
| **C** | that the mechanism is the claimed one | `/xprof-llo` op histogram on both arms | **`vrot.slane` count → 0**. If the time improves but the rotates remain, the attribution is wrong and the win is coincidental. |
| **C** | correctness | the existing suite (gate 1); if it fails on tie-adjacent rows, min-row index-set recall ≥ 0.999 against an f64 oracle (gate 2) | as stated in §3.2 |
| **D** | decode call time with and without hoisting, in-model | the same `hlo_stats` + profile pull as R2, before and after | residual ≤ 30 µs (from 44.8) |
| **D** | correctness | the existing suite, plus the debug-flag assertion that hoisted metadata equals per-layer metadata | zero-tolerance pass |
| **B** | **the measurement B exists for**: overlap fraction with the *real* cross-chunk dependency | `instruments/bench_chunked_pipeline.py` (**new file**), P3's harness modified so arm B consumes arm A's actual output. **This is the one thing P3 deliberately did not do.** | overlap ≥ 20 % at decode, ≥ 7 % at prefill (the break-evens computed in §3.4) |
| **B** | that `p` stays pinned | assert `_pick_partition` is never called on a chunked path; assert `num_waves × 32 == b_pad` | hard assertion, not a threshold |
| **B** | correctness | the suite parametrised over `G ∈ {1, 2, 4}` | **identical index sets for every `G`** |
| **all** | that the composition is not double-counting | one combined arm at the best surviving configuration | the combined stage-A time is within 10 % of the product of the individual factors |

### Named as out of scope, not forgotten

**Replacing stage B's radix select** is the largest remaining prefill prize after C (stage B is
481.0 µs of an 857.3 µs call and is at its own modelled floor). It is not designed here because
`area6` constraint 5 shows its accuracy gate cannot be run: **no `baselines/eval/glm-5.2-*` baseline
exists**, and creating one is an unbudgeted model-level run plus a new checked-in file. If someone
wants that prize, creating the eval baseline is the first step and it is a scheduling decision, not
a design one.

---

## 6. Things in the briefs I believe are wrong

Listed so a reader who trusts the briefs over this document knows exactly where the disagreement is.

1. **`report.md` §4's `t_hbm` column is computed at 3.70e12 B/s while the same section declares
   2.97e12 as the measured rate.** 175.9 MB / 3.70e12 = 47.5 µs, which is the printed decode figure;
   151.1 MB / 3.70e12 = 40.8 µs, the printed prefill figure. **Every HBM floor in §4 is 24.7 %
   optimistic.**
2. **`report.md` §4's VPU rate of 1.72e12 is `assumed` and ~4.5× too slow.** The derivable rate is
   7.782e12 (4 VALU slots × 1024 lanes × 1.900 GHz, all three from P1). This rate alone produced
   §4's "prefill is VPU-bound by 14×" and the 72.7 %-of-floor verdict in §8.
3. **`report.md` §4 models stage A's roofline and compares it against the whole call.** Stage B runs
   on a different physical unit and is absent from both floors. Adding `t_sc` is the largest single
   correction at prefill.
4. **P1 §1.2's MAC count uses 9216 positions; the kernel walks 10240.** `num_bkv = cdiv(8704, 2048)
   = 5`, `5 × 2048 = 10240`, and the causal mask is applied after the matmul. On the prober's own
   assumed geometry the answer is **4.94 % of peak, not 4.4 %**, and **5.17 % per busy cycle =
   82.7 % of the 6.25 % fill ceiling**, not 73.7 %. Confirmed independently: only the corrected
   count reproduces `area5 §B`'s `t_mxu = 13.5 µs`.
5. **P1's "6.25 % fill / 4.4 % of peak" pair is not evidence of anything**, because the array-size
   assumption cancels: peak ∝ S² and fill = 4096/S², so `(achieved/peak)/fill` is invariant in S.
   The measurement cannot distinguish 128×128 from 256×256 even though the headline moves 4.5×.
6. **P1 §4's "the largest lever P1 can see" is wrong by P1's own numbers.** `area5 §B`'s *measured*
   3.57× row cliff bounds the whole MXU prize at 9.7 µs = 4.4 % of the decode call, below the
   instrument's spread, and `area5 §A2` already shows there is no cheap mechanism.
7. **P1's counter labelling is unexplained.** 2.458e13 MAC/s exceeds one 128×128 MXU at 1.9 GHz by
   1.58×, so ≥2 MXUs contribute, yet `count_mxu_busy_1` = 0.12 % and `_2` = 3.74 %. The `_N` suffix
   does not obviously index physical MXUs. No conclusion here rests on those counters.
8. **P1 §2.1 mislabels its own top stall run.** Bundles `0x7ac`–`0xada` (815 bundles, 29 %) are
   tagged `mem_stall_mxu_idle`, but the probe's own disassembly shows a cross-lane rotate-and-add
   reduction tree with register spills and **no semaphore wait and no DMA in the range**. That
   mislabelling is why the `vrot.slane` block went unattributed until Option C.
9. **`area3 §B`'s field-9 baseline (5.2–5.4 µs per empty `pallas_call`) is impossible**: 42 empty
   launches × 5.2 µs = 218 µs, which exceeds the whole 221.4 µs call. P4 measures 0.55 µs (sbs1) /
   **0.04 µs (sbs4)** marginal with a 2.86 µs grid-0 intercept. **`CONTRADICTIONS` §8's instruction
   to "use 5.2–5.4 µs" predates P4 and should be retracted.**
10. **`report.md` §8's "the authors moved away from approximate" is false as stated.** `area6`
    constraint 3 and §C show `6142d5e7` replaced `jax.lax.approx_max_k(recall_target=1.0)` — which is
    exact — with another exact method. It was an exact→exact swap for speed, and it is not evidence
    against approximation.
11. **Stage B's decode cost is quoted three incompatible ways and nobody reconciles them.**
    `area5 §B` says 67.4 µs measured "including ~35 µs dispatch"; P2 measures 52.12 µs isolated; P3's
    arm B measures 48.22 µs isolated. The ~15–19 µs gap between in-model and isolated is real and is
    plumbing — it is the second measurement supporting Option D, and no brief connects it to
    `area7`'s CSE finding.
12. **`hardware-facts.md` carries three known-wrong claims** (`CONTRADICTIONS` §7) — the "no
    SparseCore path from Pallas" claim, contradicted by P3 running one; the literal-32 subcore field,
    which must be read as `sc.num_cores * sc.num_subcores`; and the 32 MiB scoped-VMEM ceiling. All
    three are noted where they would otherwise have bound a design decision above.
