# `streamindex_topk` — measurement and design

Step 2 of `/write-tpu-kernels`, wave 4. This is the document a human reviews and step 3 builds
from. It proposes **five options**, each deciding the seven design areas differently, ranks them,
and names the one assumption that would flip the ranking.

Inputs: step 1's `report.md` (roofline gate, instruments, baselines); wave 1's nine sheets in
`step2/sheets/`; wave 2's reconciliation in `step2/combined.md`; wave 3's seven probe results in
`probes/`. Nothing below re-derives those; where a number comes from one of them it is cited.

No kernel code was written for this document.

---

## 0. How to read this

### 0.1 Sign convention — binding on every number in this file

Every percentage is a change in **time**, `(new − old) / old`. **Negative is faster.**

Signed percentages appear **only** in a table column headed `Δ time %`, and every such row carries
the two absolute times the delta came from. Prose never carries a signed percentage; it says the
direction in words ("9.25% faster", "1.61% slower").

### 0.2 Tags — every number carries exactly one

| tag | meaning |
| --- | --- |
| `measured (job id)` | came off hardware in a named job. The job id is part of the tag. |
| `modelled` | computed from a rate or a byte count. Reproducible arithmetic, not an observation. |
| `assumed` | a value taken on belief. The belief is stated next to it. |
| `unmodelled` | **the floor has no term for it.** The word appears alone. There is never a number beside it. |

An `unmodelled` term that decides between two options is a defect at this stage. Section 9.4 is the
register of the ones that remain.

### 0.3 Terms

Terms already defined in `report.md` or `probes/bkv_p_RESULT.md` are not redefined; these are the
ones this document introduces or uses in a specific sense.

- **band** — one of the three `pallas_call` invocations `streamindex_topk` dispatches
  (`streamindex_topk.py:757` DECODE batched, `:775` DECODE remainder, `:793` MIXED). The
  PREFILL-only band is empty by construction (`report.md`, "Only two of the three cases ever
  execute"), so two of the three run.
- **`bkv_p`** — `num_kv_pages_per_block`, a 3-tuple of static ints. **Slot order is
  `(DECODE, PREFILL, MIXED)`** — verified at `streamindex_topk.py:764` (DECODE reads slot 0) and
  `:801` (MIXED reads slot **2**). Slot 1 is read by no executing band. See §2.4.
- **`subl`** — `num_sublanes_total`, the width of the f32 scores buffer in 128-column units. One
  number for all three bands, computed as a **`max` over the whole tuple**
  (`streamindex_topk.py:603-605`). This coupling is the subject of §2.1.
- **full call** — one whole `streamindex_topk` invocation, scoring **and** selection, as the
  read-only `instruments/microbench.py` measures it. Every `Δ time %` in this document uses a full
  call as its denominator unless the row says otherwise. Selection is 33–60% of it
  (`report.md`, selection-share amendment), so a scoring-only win is diluted by that factor and
  the dilution is worst at long context.
- **scoring by difference** — full call minus the `topk ctrl` control arm. Usable at `mixed_prod`;
  **not usable at `mixed_32k`/`mixed_64k`**, where the control's spread is 514.8% / 133.1%
  (`report.md` baseline tables). Where those regimes appear below, the denominator is `full`.
- **the confound shape** — `kv_len` well below `max_model_len`. The scores buffer and the
  SparseCore's read are both sized from `max_model_len`, so this shape pays for KV it does not
  have. It is not one of step 1's five regimes; §8.2 explains why that matters.
- **the A/B shape** — the end-to-end benchmark this exercise feeds: GLM-5.2, ISL 65536,
  OSL 1024, `max_model_len` 66560.

### 0.4 The block, in one paragraph

Per token, score every resident compressed KV position with a 32-head 128-dim lightning indexer and
keep the top `k = 2048`: `score[t,s] = Σ_h relu(q[t,h]·kv[s]) · w[t,h]`. The scoring half is a
Pallas kernel on the TensorCore producing `f32[num_tokens, num_sublanes_total, 128]`; the selection
half is `sparsecore_topk` on the SparseCore. **21 of 78 layers run the indexer**
(`index_topk_freq=4`, `index_skip_topk_offset=3`; `step2/sheets/area7-precomputation.md` §0), not
78 — a factor of 3.7 on every per-model-step figure.

---

## 1. The denominator: which reading of the floor

Step 1 put `decode_prod` at **55%** of its modelled floor. `probes/step_cost_RESULT.md`
[`measured (j-e8eaacd3-6be4-4cdf-95a2)`] found that a per-inner-iteration fixed cost of ≈500 ns is
real, but that it is **not a floor term**, because the inner-step count is a design parameter — and
proved it by making decode 9.64% faster with a one-line change to a static tuple, which cannot
happen below a floor. Taking the smallest feasible step count collapses the inner-step reading onto
the grid-step one:

| reading | what the 512 ns attaches to | `decode_prod` floor | achieved |
| --- | --- | ---: | ---: |
| **F-id** | nothing; `report.md`'s `hbm_id`, bytes ÷ 3364.9 GB/s | 41.7 µs | **55%** |
| **F-grid** | the `pallas_call` grid step (16 of them) | 46.8 µs | **61%** |
| ~~F-inner~~ | the inner `(bq, bkv)` iteration (64) | ~~71.3 µs~~ | refuted — the step count is a design parameter |
| ~~F-dma~~ | each page DMA (512) | ~~300.6 µs~~ | refuted by the measurement itself |

**Every option below states which reading it uses.** The distinction only bites at decode. At
`mixed_prod` the grid is **one step**, so F-grid adds 0.5 µs to a 48.4 µs floor and the two
readings agree to the significant figure they are quoted at — **18% either way**. Options whose
prize is at mixed may therefore quote either; options whose prize is at decode must pick, and this
document's convention is that **decode options are quoted against F-id (41.7 µs, 55%), which is the
harder denominator**, and F-grid is given alongside where it changes the reading.

Mixed's floor is the P1-corrected one:

| regime | scoring µs `measured` | fp8 floor, V7X spec | fp8 floor, measured Pallas rate | achieved |
| --- | ---: | ---: | ---: | ---: |
| `mixed_prod` | 269.5 | 38.5 µs | **48.4 µs** | **18%** |

`report.md`'s table used the V7X spec rate (1785.6 TFLOP/s fp8) and got 14%; areas 4 and 5 carry a
`measured` Pallas fp8 rate at 1.86× bf16, which gives 48.4 µs and 18%. **This document uses 48.4 µs
and 18%** — the more conservative of the two — and P4 corroborates the ratio independently at
1.69× from two rungs agreeing to 0.5% [`measured (j-8ed8555c-1d46-44dd-8f6c)`].

**Mixed is where the headroom is: 18% of floor against decode's 55–61%, a 5.6× gap against
decode's 1.6–1.8×.** §7 is the reason that sentence is less safe than it looks.

---

## 2. What wave 3 settled, and the three contradictions resolved

### 2.1 Contradiction 1 — P2's "take `bkv_p` to 8 at mixed directly" is wrong

`combined.md`'s P2 fork text reads, under the branch the measurement selected: *"Tiled: C4 is a
no-op and drops off the list; take `bkv_p` to 8 at mixed directly."*

**Do not carry that as a recommendation.** P2 established **feasibility** — Mosaic tiles the einsum
result, the mixed band's whole VMEM peak is 4.195 MiB not 32 MiB, and `bkv_p = 8` at mixed would
peak at 7.195 MiB [`modelled`, substituted into a `measured` allocation list], nowhere near the
ceiling. `probes/bkv_p_RESULT.md` measured **desirability**, and mixed is slower at every raised
arm:

| regime | tuple | subl | old ms | new ms | spread % | Δ time % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mixed_prod` | (4,2,2) | 96 | 0.6666850706096739 | 0.6867082207463682 | 2.61 | +3.00 |
| `mixed_prod` | (8,2,2) | 128 | 0.6666850706096739 | 0.7007382460869849 | 2.27 | +5.11 |
| `mixed_prod` | (4,4,4) | 96 | 0.6666850706096739 | 0.6774073000997305 | 1.77 | +1.61 |
| `mixed_64k` | (4,2,2) | 544 | 4.789508902467787 | 4.875155910849572 | 0.86 | +1.79 |
| `mixed_64k` | (4,4,4) | 544 | 4.789508902467787 | 4.86955470405519 | 1.71 | +1.67 |

All `measured (j-3aa755dd-900b-4f31-9fa3)`. No `(8,8,8)` arm exists — see §10.

**The mechanism is the coupling, not VMEM.** `num_sublanes_total` is a `max` over the whole tuple
(`streamindex_topk.py:603-605`), so raising *any* slot widens the scores buffer for *every* band.
Mixed carries 1024 tokens against decode's 64, so its scores buffer is 16× larger in absolute terms
and a 20% widening costs 8.4 MB written by the TC and read again by the SparseCore — which reads
**O(padded width)**, not O(`row_lengths`) (`area7` and `area8`, agreeing from source). Against that,
mixed's own step saving is `16 × 500 ns ≈ 8 µs` on a 667 µs call, about 1.2%.

Feasibility is not desirability. **VMEM was never the binding constraint on `bkv_p` at mixed; HBM
traffic through a `max`-coupled buffer was.** That is what Option B attacks.

Two limits on how hard the mixed rows may be pushed: `+1.61%` sits against a 1.77% spread on its own
row and 1.68% on the reference, and `+1.67%` against 1.71% and 1.30%. **The strongest reading these
rows support is "not measurably slower at mixed" — not "1.6% slower", and not "free."**

### 2.2 Contradiction 2 — there is nothing for C5 to absorb

P4's fork text reads: *"C6 dies, and S1-c's relayout must be paid or absorbed by C5 instead."*
**Amended:** P3 measured that relayout and it is ≤0.12% of an inner iteration at `mixed_prod`, below
P3's own negative control (the effect was 6.3 ns; the control that must read zero read 9.9 ns)
[`measured (j-efb89184-020f-495f-9d4b)`]. C5 is dead, and there is nothing to absorb. **The relayout
is simply paid, for nothing.** The two probes are not in conflict; the fork text anticipated a cost
that turned out not to exist.

### 2.3 Contradiction 3 — the VPU verdict

Deferred to §7, because it is not a contradiction between two probes but between one probe and the
whole ranking.

### 2.4 A correction to the probe record: the tuple slot order

`probes/bkv_p_RESULT.md`'s terms section says `bkv_p` is `(decode, mixed, prefill)`. `report.md`
says `(DECODE, PREFILL, MIXED)`. **`report.md` is right**: `streamindex_topk.py:801` passes
`num_kv_pages_per_blocks[2]` to the MIXED band. No measurement changes — every arm in
`bkv_p_RESULT.md` has slots 1 and 2 equal, so the data cannot discriminate and its `subl`/`num_bkv`
columns are correct either way. But it changes a **design** fact: **slot 1 is read by no executing
band and still participates in the `max`.** Any per-band decoupling must start from that.

### 2.5 Dead. Do not re-propose these.

| candidate | why it is dead | evidence |
| --- | --- | --- |
| **C4** — tile the einsum so `s` never materialises at full width | Mosaic already tiles it. The `4096x2048xf32` matmul splits into eight `4096x256`; the whole mixed band allocates **4.195 MiB**, not 32 MiB. C4 is a no-op, and the `B · bq_sz · bkv_p ≤ ~500` inequality it was built to relieve does not bind. | `measured (j-689b87cd-7460-4b90-bd87)` |
| **C5** — L3's 3-D contraction, deleting the `s_summed` reshape | The reshape is below the probe's own noise floor: ≤0.12% of an inner iteration at `mixed_prod`, ≤0.83% at `decode_prod`. Both `(128,2048)` and `(128,16,128)` keep 128 in the lane axis; no lane data moves. | `measured (j-efb89184-020f-495f-9d4b)` |
| **C6** — transpose the score tile to `k · qᵀ`, N = 32 | Narrow-N derate is **≈3.5× bf16, ≈3× fp8** at N = 32 vs N = 128, bounded 2–4×. Cross-checks the independent M-axis 3.57×. C6's premise — that narrow N is near-free — is refuted twice. N = 128 is the plateau; N = 256 buys nothing. | `measured (j-8ed8555c-1d46-44dd-8f6c)` |
| **C8, E1 arm** — fold `w` into `q`, deleting broadcast and multiply | **Rejected on correctness.** `weights_proj.weight` is ~50% negative on the real GLM-5.2-FP8 checkpoint (BF16 `[32,6144]`, **0 of 32 rows all-non-negative**), and there is no activation, clamp or squaring anywhere on the path. The `relu` at `:333` is on the *score*, not on the weight. Its prize is recorded — 13.70% faster at W=512 — so nobody pays to re-measure it if some future model does guarantee a sign. | checkpoint read; prize `measured (j-efb89184-020f-495f-9d4b)` |
| **C8, L13 arm** — pre-shaped `w` so no broadcast is emitted | **Wrong sign.** 2.86% slower at W=512 and 4.61% slower at W=128, consistent across both widths with tight spreads on the W=512 pair. The broadcast Mosaic emits is already free; handing it a pre-shaped `(128,32,1)` operand forces a real tensor through the pipeline. C8 is closed from both directions. | `measured (j-efb89184-020f-495f-9d4b)` |
| **C10, bf16-operand arm** — feed the einsum bf16 operands | The einsum already reaches the MXU as **native fp8** (`f8E4M3FN × f8E4M3FN → f32`; 1064 `vmatmul`, **zero** bf16 matmuls). bf16 operands are a 1.86× regression on the term with the most headroom in the block. | `measured (j-689b87cd-7460-4b90-bd87)` |

C10's **bf16/16-bit *scores-buffer*** arm is a different thing and is alive; it is Option C.

Also already decided without a probe, and not re-opened: S2 (concatenate decode's 4 sequences to
M=128 — 3.57 W → 4.00 W, 12% slower on a term that is not binding), S1-a (fold the e8m0 scale
before the contraction — costs the 1.86× fp8 rate to save one 2048-element transpose), E6/H (the
32-head sum as a block-diagonal ones-matmul — 3.1% MXU fill), B2 (mixed `bq_sz` 256/1024 —
`s` at 64/256 MiB, above physical).

### 2.6 Survivors, and what became of them

| # | candidate | fate in this document |
| --- | --- | --- |
| C1 | narrow the 256 B cache record | **Option A** |
| C2 | size `num_sublanes_total` from real `kv_len` | **Option B** (with C9) |
| C3 | `bkv_p` 2 → 4 | **Option E**, and its ceiling is Option B |
| C7 | 16-bit monotone-encoded scores | **Option C** (with C10's scores arm) |
| C9 | delete the `-inf` fill | **Option B**; also inside D |
| C11 | threshold-first two-pass | **Option D**, one of its two mechanisms |
| C12 | batch the sequence axis at MIXED | folded into Option B's area-2 decision; unmeasurable on the current instrument (§8.2) |
| C13 | producer emits `indexer_weights` f32 and transposed | **demoted.** Its only composition partner was C8's L13 arm, which P3 refuted. What remains is deleting one standalone XLA upcast pass on a `[ntok,32]` tensor — 0.13 MB at mixed. `unmodelled`, and too small to rank. Recorded here so it is not re-proposed as a partner for a dead arm. |
| S4-c | writer-side scatter + reshape | **carried** — §11 item 1 |

---

## 3. Option A — **Narrow Record**

> Change the compressed-KV record so the block stops reading padding. The boundary moves *up*, into
> the producer.

**Thesis.** 129 of every 256 bytes of `cache_kv` is used; the other 127 are `align_to(129, 128)`
padding that crosses HBM on every read. Decode is HBM-bound on exactly this array — 134.2 MB of a
140.3 MB ideal total at `decode_prod`. Splitting the record into a `u8[..., 128]` value array and a
separate scale array (area 4's P2) takes the used fraction to ~100% and the modelled ideal decode
floor from 41.7 µs to **21.9 µs** (`report.md`, "Where the headroom is"). This is the largest single
structural finding in step 1, and Option A is the only option that acts on it.

**Floor reading: F-id (41.7 µs, 55%).** A's prize is decode-side and byte-shaped, so the harder
denominator is the honest one; under F-grid (46.8 µs, 61%) the same saving is a slightly larger
fraction of a slightly larger floor and the ranking does not move.

### 3.1 The seven areas

| area | A's decision | tag |
| --- | --- | --- |
| **1 — fusion boundary** | The boundary moves **upward into the producer** (area 1's alternative C: make the scatter write the kernel's layout). The block's input contract changes; the block's own three-invocation structure does not. `cache_kv` becomes two arrays. | `modelled` |
| **2 — partitioning** | Unchanged: same grid, same `fori_loop` nest, same `bkv_p`. The working set per band **falls** — decode's `bkv` scratch 4.000 → 2.016 MiB, taking the band from 4.095 to 2.111 MiB against a 31 MiB scoped budget. | `modelled` |
| **3 — pipelining** | Unchanged in structure; each inner step moves 2.036 → 1.036 MiB at decode, and the predicted per-step DMA cost falls from 1126 ns toward ~640 ns against a measured 1194 ns. **This is the one option that changes which side of the pipeline dominates at decode** — it should push decode from DMA-dominated toward the ≈500 ns fixed term, which is Option E's lever. A and E therefore interact and must be measured composed, not summed. | `modelled` |
| **4 — layout/precision** | The pivot. Storage `u8` unchanged; the **record layout** changes from one 256 B interleaved record to a 128 B value plane plus a scale plane. Wasted fraction of the `cache_kv` read: 49.6% → ~0.6%. `load_bkv`'s in-VMEM slice of the 128 value bytes and 1 scale byte (`:257-273`) disappears — the DMA lands what the compute wants. | `modelled` |
| **5 — compute units** | Unchanged. MXU occupancy, derate and VPU work are all untouched; A moves bytes only. | — |
| **6 — algorithm** | Unchanged, exactly. Same arithmetic, same order, same accumulation. A is bit-exact by construction. | — |
| **7 — precomputation** | The record layout moves from level 3 (re-established per call by the scatter) to **level 2** (fixed at model load, by the cache allocator). | `modelled` |

**Rejected inside A, one line each.** A 160 B record (still 128-lane-misaligned; buys 37.5% where
the split buys 49.6%, for the same writer change). A 132 B record (same objection). Narrowing the
*read* only, without touching the writer (area 4's L10 — a 12× risk against a 1.98× byte saving per
area 3 item 21; "this is where the no-writer-change option dies, and it should die on the record").
Hoisting the scales into a per-page array (area 4's P3, 128.0078 B/position) — 0.006% better than
the split and a second format change; take it only if the writer's owner prefers it.

### 3.2 What it buys

Bytes → time at **2.97 TB/s**, the `measured` whole-kernel achieved HBM ceiling, not the 3364.9 GB/s
spec rate. Baselines are step 1's full-call medians. Every saving is `modelled`; the rate is
`measured`; the product is `modelled`.

| regime | KV bytes read | saved | old µs `measured` | new µs `modelled` | Δ time % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 134.22 MB | 66.57 MB | 167.01 | 144.60 | **−13.42** |
| `decode_64k` | 100.66 MB | 49.93 MB | 201.16 | 184.35 | **−8.36** |
| `mixed_prod` | 16.78 MB (2.1 × 8 re-reads) | 8.32 MB | 669.11 | 666.31 | −0.42 |
| `mixed_64k` | 134.22 MB (16.8 × 8) | 66.57 MB | 4762.84 | 4740.43 | −0.47 |

(`step2/sheets/area1-fusion-boundary.md`'s K1a row gives −13.4 / −0.4 / −0.9 for three of these.
The first two agree exactly; the `mixed_64k` cell differs because that sheet's row is not against
the full-call time. This document uses full-call throughout.)

**A is a decode option.** It is worth 13.4% at `decode_prod` and half a percent at mixed. At the A/B
shape the relevant row is `decode_64k`'s −8.36%.

### 3.3 File plan

Adopts the layout sheet's four-file group for the operation, and does **not** invoke the second
axis — A changes no kernel logic, only the shape of two refs and the deletion of one in-VMEM slice.

| file | new/changed | what | ~lines |
| --- | --- | --- | ---: |
| `streamindex_topk.py` | changed | `load_bkv` (`:257-272`) loses its slice-and-recombine; `_fetch_bkv` (`:160-198`) issues two DMAs instead of one; the `lkv_dim`/`page_size_per_kv_packing` derivation at `:590-593` reads the value array's trailing dim | ~−40, +25 |
| `mla_attention_op.py` | changed | the scatter at `:103-104` writes two arrays | ~+20 |
| the cache allocator (owner TBD) | changed | allocates two arrays instead of one | `unmodelled` — **not owned here; see §11 item 1** |
| `tests/kernels/test_dsv4_streamindex_topk.py` | changed | the fixture builds two arrays; the XLA `_fwd` reference is unchanged because the arithmetic is | ~+30 |

**A does not fit in one team's file plan, and that is its defining property.** The row with the
`unmodelled` cost is not in this kernel; §11 item 1 is the same owner.

### 3.4 Correctness gate

Bit-exactness against the current kernel, on the existing `instruments/test_correctness.py` rank
criterion (`RANK_SLACK=256`, `MIN_RECALL=0.99`) with its negative control, at `decode_prod` and
`mixed_prod`. A rearranges bytes and changes no arithmetic, so anything other than bit-identical
margins is a bug. Precedent for that standard: P5′ got bit-identical margins on all six figures for
a change of the same character [`measured (j-e13fc627-a8c5-4ef4-bd5e)`].

### 3.5 What would kill it

The writer-side cost of splitting the scatter. The current `scatter_fusion` on `u8[424960,256]` is
`measured` at ~291 µs per indexer layer; two narrower scatters could be slower than one wide one.
That term is **`unmodelled`** and it decides whether A is net positive. §9.4.

---

## 4. Option B — **Right-Size**

> Nothing about the arithmetic changes. Every buffer, on both cores, is sized from the batch's real
> KV length instead of from `max_model_len` — and the fill that initialises the part nobody reads is
> deleted.

**Thesis.** `num_sublanes_total = max over the tuple of align_to(pages_per_seq, bkv_p) × 8`, where
`pages_per_seq` derives from **`max_model_len`, not from `kv_len`**. Three things follow, and they
compound: the scores buffer is wider than the data; XLA `-inf`-fills all of it before the kernel
overwrites 80–97% of it; and the SparseCore reads **O(padded width)** because
`_sc_topk_direct`'s per-row `dma(scores_hbm.at[pl.ds(s, slice_len)], ...)` has a **static**
`slice_len` (`area7` §, from source). At the A/B shape this is the dominant inefficiency in the
block and it is not an arithmetic problem at all — it is a sizing problem, in six areas at once.

**Floor reading: F-id.** B's prize is bytes; at mixed the grid term is 0.5 µs of 48.4 and the two
readings agree.

### 4.1 The seven areas

| area | B's decision | tag |
| --- | --- | --- |
| **1 — fusion boundary** | Boundary unchanged in *position*; changed in *width*. The scores value still crosses HBM (it must — one Pallas kernel cannot hold a TC mesh and an SC mesh, `verified j-996520b8`), but it crosses at its live width. Area 1's rows S1 (the `-inf` fill, "no justification found for this crossing"), S2 and S3 all shrink. | `modelled` |
| **2 — partitioning** | The grid is unchanged; the **shape it is computed from** changes. `num_sublanes_total` becomes per-band, bucketed to a small set of jit-static widths so the recompile count stays bounded. This also unlocks C12 (batch the MIXED band's sequence axis) by removing the `max`-coupled width that made batching expensive — but C12 itself is not part of B, because it is unmeasurable on the current instrument (§8.2). | `assumed` (bucket count) |
| **3 — pipelining** | The per-step out-DMA — **the largest single per-step transfer at mixed, 1 MiB of 1.63 MiB, and the only strided one** (128 runs of 8 KiB at a 40960 B pitch, 20% duty) — narrows in proportion. Its duty cycle rises toward 1.0 as the pitch collapses onto the run, which is a second-order win area 3 prices as `unmodelled`. | `modelled` + `unmodelled` |
| **4 — layout/precision** | Precision unchanged everywhere. The scores buffer's wasted fraction goes from 80.0% (`decode_prod`) / 97.0% (`mixed_64k` `-inf` init) and 20.0% / ≥3.0% (top-k read) to ~0. The kernel's final store becomes a **masked partial-block store** — the mechanism, §4.2. | `modelled` |
| **5 — compute units** | Area 5's alternative **R** taken: units assigned per regime. `_common_path` already dispatches separate `pallas_call`s per case, "so the split is nearly free structurally." Area 5's alternative **P** also taken by construction — the `-inf` fill is deleted rather than moved, so it stops occupying the TensorCore at all. | `modelled` |
| **6 — algorithm** | Unchanged. B is exact. `-inf` remains load-bearing at the selection boundary (`sparsecore_topk` bitcasts f32 to a monotone int32 and derives `KEY_NEG_INF` from `-inf`'s bit pattern), so the *mask*'s `-inf` writes stay; only the redundant **prefill** goes. This is the distinction C9 turns on and it must not be blurred. | `assumed` — see §4.4 |
| **7 — precomputation** | The pivot. `scores_init` moves from level 3 (rebuilt per call in XLA, `modelled` 0.779 / 12.46 / 42.38 / **82.27 µs** per rebuild, × 21 layers = **1.73 ms per model step at `mixed_64k`**) to **not existing**. `eff_row_lengths`' ~17 HLO ops × 21 ≈ 357 op launches per step move from level 3 to level 4 (once per kernel invocation) where the width is already known. | `modelled` |
| **8 — SparseCore** | The padded-DMA share of selection time is 14 / 31 / 35 / 29% across the four measured regimes; at the confound shape live/padded is **12%**, and the padded DMA alone models to **821 µs** against 399.60 µs for the whole of selection today — *"a modelled ≥2× regression on short requests, caused only by the config."* B removes it. | `modelled` |

**Rejected inside B, one line each.** Sizing from `seq_lens` dynamically rather than bucketing (a
dynamic shape at the XLA boundary forces a recompile per distinct length, which is worse than a
fixed bucket ladder). Keeping the fill but writing only the pad sublanes (area 4's L7 — correct, but
it keeps a level-3 rebuild alive to save a level-3 rebuild). Caching `scores_init` across layers
(area 7's A6/A3, named there as the staleness traps — and **nothing can be CSE'd across layers
anyway**, because each indexer layer builds its own torch custom op with a per-layer name
`op_name = f"pallas::streamidx_topk_{prefix}"` and is dispatched as a self-contained exported
StableHLO module).

### 4.2 The mechanism, and why it is the highest-leverage code in the space

The `align_to(pages_per_seq, bkv_p)` exists so that the kernel's store never has a partial final
block: every `bkv` iteration writes a whole `bkv_p`-page slab at 8·`bkv_p`-sublane granularity.
Exact sizing means the last slab is partial, which means **a masked store on the final block**.
That is roughly 30 lines in the kernel body, and it is the single change that:

- lets `num_sublanes_total` be exact rather than rounded (B's own prize);
- **decouples the width from the tuple**, which converts Option E's `+1.61% / +1.67%` at mixed into
  its own step saving of about 1.2% faster [`modelled`, from `16 × 500 ns` on a 667 µs call];
- and removes the reason `bkv_p` cannot be chosen per band.

**Options B and E are not independent. E is a config change whose ceiling is B.**

### 4.3 What it buys

Two shapes, because B's prize is entirely a function of `kv_len / max_model_len` and step 1's five
regimes all sit at the top of that ratio.

**(a) At step 1's regimes** — `kv_len ≈ max_model_len − 1024`, so only the `align_to` rounding and
the fill are available:

| regime | source of saving | bytes saved | old µs `measured` | new µs `modelled` | Δ time % |
| --- | --- | ---: | ---: | ---: | ---: |
| `decode_prod` | fill (2.62 MB) + rounding (80→72 subl) | 3.14 MB | 167.01 | 165.95 | −0.63 |
| `mixed_prod` | fill (41.94 MB) + rounding (80→72 subl, on write + SC read) | 50.33 MB | 669.11 | 652.16 | **−2.53** |
| `mixed_64k` | fill (276.82 MB) + rounding (528→520 subl) | 285.12 MB | 4762.84 | 4666.85 | **−2.02** |

**(b) At the confound shape** (`kv_len` 8192, `max_model_len` 66560 — i.e. an early chunk of the
A/B's 64-chunk prefill). Width 528 → 64 sublanes; scores 276.82 → 33.55 MB:

| term | bytes saved | time saved `modelled` |
| --- | ---: | ---: |
| TC write + SC read at the live width | 486.53 MB | 163.8 µs |
| `-inf` fill deleted | 276.82 MB | 93.2 µs |
| **subtotal, TensorCore side** | **763.35 MB** | **257.0 µs** |
| SparseCore padded read sized to `row_lengths` | — | ~0.7–0.9 ms `modelled` — see caveat |

| shape | old µs | new µs `modelled` | Δ time % |
| --- | ---: | ---: | ---: |
| confound, TC side only, against `mixed_64k`'s full call as an upper-bound denominator | 4762.84 | 4505.84 | **≤ −5.40** |

The denominator is an upper bound: the real call at that shape moves less KV than `mixed_64k` does,
so it is faster, so the true magnitude is **larger** than 5.40%. Hence "≤ −5.40" — the honest
direction for a bracketed denominator. **The percentage cannot be closed without a measurement at
that shape**, and §12 makes adding it step 3's first instrument change.

*Caveat on the SparseCore row.* Area 8's streaming floor at 338e9 B/s runs 3.2–7.2× optimistic
against `topk ctrl`, so the 821 µs figure is not quotable as a saving. Cross-checked the other way:
`mixed_64k`'s measured selection is 3009.7 µs with a 29% padded-DMA share = 873 µs, and sizing that
to a 12.3% live fraction saves ~765 µs. The two routes agree to within their own precision, which is
why the row reads 0.7–0.9 ms and not a point.

### 4.4 Correctness gate

Two distinct claims, and they need different evidence.

1. **Exact sizing is bit-exact.** Same rank criterion + negative control at all five regimes, plus
   at least one bucket boundary (a `kv_len` that lands mid-bucket), since the masked store is new
   code on a path the existing tests do not exercise.
2. **Deleting the fill is safe *only* if no consumer reads a lane the mask does not cover.** Six
   sheets agree it has no consumer and area 7 found `scores_init`'s values "appear to have no
   consumer at all" — but that is a source reading, tagged `assumed`, and it is the one thing in B
   that can silently corrupt a top-k rather than crash. The check: run with the fill replaced by
   a **poison value** (a large positive f32 rather than `-inf`) and confirm bit-identical margins.
   A poison fill that changes nothing proves no lane is read; a poison fill is not an
   approximation, so this is a strict test and it costs one shard.

**Note the trap.** `instruments/test_correctness.py`'s pass is not the gate here — a stale lane
would have to be selected into the top-2048 to move `worst_rank`, and at 97% waste the stale lanes
are `-inf` today, which is exactly what makes them invisible. The poison arm is the check with
teeth. Compare P3's discipline: its negative control read *larger* than the effect it was bounding,
and that is what made the result honest.

### 4.5 File plan

| file | new/changed | what | ~lines |
| --- | --- | --- | ---: |
| `streamindex_topk.py` → `tiling.py` | new | `cdiv`, `align_to`, and the new `num_sublanes_for_band(pages, bkv_p, bucket)`. Ints only, no refs, no arrays. | ~70 |
| `streamindex_topk.py` → `kernel.py` | changed | the masked partial-block store in the body's epilogue; the store index map takes the band's own width | ~+35, −10 |
| `streamindex_topk.py` → `wrapper.py` | changed | `run_scores_kernel` (`:607-738`) and the three band invocations (`:757/775/793`) take a per-band width; `scores_init` and its `jnp.full` disappear | ~−25 |
| `streamindex_topk.py` | changed | `num_sublanes_total` at `:603-605` deleted; the `max` goes with it | ~−5 |
| `sparsecore_topk.py` | changed | `slice_len` becomes per-row rather than static — **this is the largest risk in B**, see below | ~+40 |
| `tests/kernels/test_dsv4_streamindex_topk.py` | changed | bucket-boundary case; poison-fill arm | ~+60 |
| `tests/kernels/test_sparsecore_topk.py` | **new** | **there is no test for `sparsecore_topk` at all** — see §11 item 4 | ~+120 |

B invokes the layout sheet's second axis (`tiling.py` split out) but not the first (no `api.py` /
`base.py`), because it does not change the operation's public contract. See §6.5 for the ruling on
how the two tables compose.

### 4.6 What would kill it

The `sparsecore_topk` row. That kernel runs with `disable_bounds_checks=True` and a dynamic `woff`,
and `grep -rn "sparsecore_topk" tests/` returns nothing — **step 3 changing anything inside it has
no unit-level net under it.** B's TensorCore half is buildable and testable today; B's SparseCore
half should not be attempted before §11 item 4. A B that ships only the TC half still gets rows (a)
and the TC subtotal of (b) — the ~0.7–0.9 ms row is the part that waits.

---

## 5. Option C — **Half-Width Key**

> The scores buffer is a *ranking key*, not a value. Ship 16 bits of it.

**Thesis.** The kernel returns only indices, so order is the entire contract — the f32 scores buffer
is a monotone key and nothing downstream reads its magnitude. `sparsecore_topk` already bitcasts
f32 to a monotone int32 internally. At `mixed_64k` that buffer crosses HBM **822 MB in total**
(276.8 written as fill + 268.4 written by the kernel + 276.8 read by the SC) against 134.2 MB of
KV: **it is six times the traffic of the data it was computed from.** Halving the key halves the
largest term in the block's traffic ledger, and it does so without changing where anything runs.

**Floor reading: F-id.** C's prize is at mixed, where the two readings agree.

### 5.1 The seven areas

| area | C's decision | tag |
| --- | --- | --- |
| **1 — fusion boundary** | Unchanged in position and width; changed in **element size**. C is the only option that leaves every boundary exactly where it is. | — |
| **2 — partitioning** | Unchanged. | — |
| **3 — pipelining** | The mixed band's out-DMA halves, 1 MiB → 0.5 MiB of a 1.63 → 1.13 MiB step. Since area 3 could not determine which side dominates at mixed, this is a `modelled` byte saving whose effect on the step is **`unmodelled`**. | `modelled` + `unmodelled` |
| **4 — layout/precision** | The pivot, and the *only* precision change in this document that survives P1. **Accumulation stays f32** — a bf16 accumulator does not run at all (`VerificationError`), and `sparsecore_topk.py:395-396` raises unless the key is f32 at the selection boundary, so both sides need the change together. Operands stay fp8 (P1). Only the **stored key** narrows. | `modelled` |
| **5 — compute units** | Unchanged on the TensorCore. On the SparseCore, `MAX_SLICE_WORDS = 65536` buys **twice as many positions per slice**, which is what `_pick_partition` uses to choose a partition: `decode_64k`'s `p=4` two-stage (24/32 subcores in stage 1, **6/32 in stage 2**) and `mixed_64k`'s `p=2` should each collapse by one level, deleting a launch and a badly-occupied stage. Prize `unmodelled`. | `unmodelled` |
| **6 — algorithm** | An **approximation**, and the first one in this document. Named gate below. | — |
| **7 — precomputation** | Unchanged, except that the `scores_init` fill halves if B is not also taken. | `modelled` |
| **8 — SparseCore** | Requires a change inside `sparsecore_topk`: the bitcast, `KEY_NEG_INF`, and the f32 assertion at `:395-396`. Same net-under-it problem as B. | — |

**Rejected inside C, one line each.** bf16 as the key (the mantissa is 8 bits and the exponent range
is wasted on a monotone ranking — a custom fixed-point or a shifted-exponent i16 uses the bits
better; but bf16 is the cheap first arm because the cast is free). A 16-bit key on the *decode*
regimes only (decode's scores buffer is 2.6 MB — the prize is 0.5%, not worth a second code path).
Narrowing the key *and* the operands together (that is C10's operand arm, dead — §2.5).

### 5.2 What it buys

Halves the TC write and the SC read. Rows assume B is **not** also taken (if it is, C's percentages
apply to B's already-reduced bytes; §9.3).

| regime | scores bytes crossing HBM | saved | old µs `measured` | new µs `modelled` | Δ time % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 5.24 MB (2.62 W + 2.62 R) | 2.62 MB | 167.01 | 166.13 | −0.53 |
| `mixed_prod` | 83.88 MB | 41.94 MB | 669.11 | 654.99 | **−2.11** |
| `mixed_64k` | 553.64 MB | 276.82 MB | 4762.84 | 4669.64 | **−1.96** |

Plus the `-inf` fill halving, if B is not taken: a further −0.26 / −1.06 / −0.98. Plus the
SparseCore partition collapse: **`unmodelled`.**

### 5.3 Correctness gate — the named accuracy gate

C is an approximation and must be gated as one. The gate already exists and is the one wave 3 used:
the float64 oracle with `RANK_SLACK = 256` (so `worst_rank ≤ 2304`) and `MIN_RECALL = 0.99`, plus
the negative control at the same configuration. Shipped f32 scores today read `worst_rank = 2098 /
min_recall = 0.99512` at `mixed_prod` and `2109 / 0.99561` at `decode_prod`
[`measured (j-e13fc627-a8c5-4ef4-bd5e)`].

**The gate: `worst_rank ≤ 2304` and `min_recall ≥ 0.99` at all five regimes, with the negative
control firing on every row.** The margin today is thin — 2098 against 2304 is 9% of the slack —
so a 16-bit key that ties many scores at rank 2047/2048 will show up. That thinness is the reason C
is a real risk and not a free win, and it is also why the gate is informative: a key that halves
precision has only 206 ranks of headroom before it fails.

**And note what the gate cannot see.** `instruments/inputs.py:180` draws `w` from
`uniform(−1.5, 1.5)` and the real checkpoint is ~50% negative with `min = −3.33, max = 2.81` — the
distributions are close enough that the fixture is defensible here, unlike for E1 where it was
irrelevant to the question. But the *score* distribution under real `q`/`kv` is not established, and
tie density at rank 2048 depends on it. Tag: `assumed`.

### 5.4 File plan

| file | new/changed | what | ~lines |
| --- | --- | --- | ---: |
| `streamindex_topk.py` → `compute.py` | changed | `compute_scores` (`:304-354`) gains the f32 → key encode at the store boundary only; the accumulator is untouched | ~+20 |
| `streamindex_topk.py` → `kernel.py` | changed | scores ref dtype; the mask writes the key's `NEG_INF` rather than f32 `-inf` | ~+15 |
| `streamindex_topk.py` | changed | `out_dtype` plumbing; the `scores.astype(jnp.float32)` at `:820-821` is deleted, not widened | ~−10 |
| `sparsecore_topk.py` | changed | the bitcast, `KEY_NEG_INF`, the f32 assertion at `:395-396`, and `MAX_SLICE_WORDS` in positions rather than words | ~+50 |
| `tests/kernels/test_dsv4_streamindex_topk.py` | changed | the accuracy gate above as an explicit test, not just a probe | ~+40 |
| `tests/kernels/test_sparsecore_topk.py` | **new** | as B — §11 item 4 | ~+120 |

### 5.5 What would kill it

The gate. And, more quietly: C's prize is bytes at mixed, and §7's hypothesis says mixed may not be
byte-bound at all.

---

## 6. Option D — **No Round Trip** *(the clean-sheet boundary)*

> Decide area (1) from the value ledger, with no reference to the fact that this block is currently
> three `pallas_call`s and a SparseCore kernel.

### 6.1 Deriving the boundary from values, not operations

Ask what must cross the block's boundary, reading only the value ledger:

| value | direction | bytes at `mixed_64k` | must it cross? |
| --- | --- | ---: | --- |
| `q` | in | 0.5 MB | yes — produced by a different layer |
| `w` (`indexer_weights`) | in | 0.13 MB | yes — same |
| `cache_kv` | in | 16.8 MB (live) | yes — persistent state |
| `topk_idxs` | out | 8.4 MB | yes — the block's entire product |
| **`scores`** | internal | **822 MB** | **no.** It is consumed by the very next step of the same computation. |

**The irreducible boundary traffic of this block is about 26 MB. It moves 848.** The scores buffer
is 97% of the block's HBM traffic at `mixed_64k` and it is an *implementation artefact* — it exists
because the top-k runs on a different core, and area 8 verified that **one Pallas kernel cannot hold
a TC mesh and an SC mesh** [`verified j-996520b8`]. A clean-sheet boundary never names `scores` as a
value at all.

D is the option that takes that seriously. Its structure is: **one operation that consumes
`q, w, cache_kv` and produces `topk_idxs`, with no intermediate crossing HBM.** Because of the mesh
constraint, that means selection runs on the **TensorCore**, and the SparseCore leaves this block.

### 6.2 The seven areas

| area | D's decision | tag |
| --- | --- | --- |
| **1 — fusion boundary** | One operation, derived above. The three-`pallas_call` band split survives only as an internal dispatch, if at all. | `modelled` |
| **2 — partitioning** | The iteration space is re-cut so that the *selection* axis, not the KV axis, is outermost: for each token, stream KV and maintain a running top-`k` state in VMEM. `k = 2048` f32 keys + i32 indices per token = 16 KiB/token; at `bq_sz = 128` that is 2 MiB of state — it fits, alongside the 4.195 MiB the band uses today. | `modelled` |
| **3 — pipelining** | The out-DMA — mixed's largest and only strided per-step transfer — **disappears**. What remains per step is the KV in-DMA. This is the only option that makes the mixed band unambiguously compute-bound by construction rather than by measurement. | `modelled` |
| **4 — layout/precision** | Key precision becomes a free parameter (nothing crosses HBM, so C's prize evaporates and C's risk with it). `-inf` stops being load-bearing, because the `sparsecore_topk` bitcast contract goes away. | — |
| **5 — compute units** | The pivot, and the risk. **Selection moves from the SparseCore to the TensorCore.** The SparseCore is 33–60% of the call today and exists precisely because a top-2048-of-67584 is bad on a VPU with no scatter-add and no sort primitive. | `unmodelled` |
| **6 — algorithm** | Either C11's threshold-first two-pass (compute a per-token τ with `|{s ≥ τ}| ≈ k`, then recompute and emit) or a streaming top-`k` merge. The first doubles the einsum; the second needs a merge network. | `unmodelled` |
| **7 — precomputation** | `scores_init` (level 3, 1.73 ms/step at `mixed_64k`) and `eff_row_lengths` (~357 op launches/step) both cease to exist. The `topk_indices_buffer[:ntok] = -1` fill (§11 item 2) goes too, since D writes every row. | `modelled` |
| **8 — SparseCore** | Vacated. Every constraint on that sheet — `MAX_SLICE_WORDS`, `_pick_partition`'s two-stage at `decode_64k`, the padded DMA — stops applying. | — |

**Rejected inside D, one line each.** Per-`bkv` local top-`m` merged by the SC (area 1's J): at
`mixed_64k`, `nbkv = 32` and a local `m` large enough to be exact is `k` itself, so the candidate
set is 32·2048 = 65536 — no smaller than the 67584 positions it replaces. Local `m = 64` is an
approximation with no gate and no obvious one. Keeping the SC and handing it a *compacted* candidate
list (needs a prefix-sum + scatter on the TC; the VPU has neither). Widening the block downward past
the top-k into whatever consumes `topk_idxs` (area 1's P — outside this exercise's block).

### 6.3 The ledger, honestly

D's prize is **not** the bytes. Doing the byte arithmetic first, because it is the part that is
`modelled` and it is the part that disappoints:

| regime | scores traffic deleted (fill + write + read) | time saved `modelled` | einsum cost of a second pass `modelled` | old µs `measured` | new µs `modelled` | Δ time % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 7.86 MB | 2.65 | +12.07 | 167.01 | 176.43 | +5.64 |
| `mixed_prod` | 125.83 MB | 42.37 | +48.4 | 669.11 | 675.14 | +0.90 |
| `mixed_64k` | 830.46 MB | 279.62 | +387.0 | 4762.84 | 4870.22 | +2.26 |

(The einsum column is `report.md`'s fp8 floor row scaled by 1.257 to the P1-corrected measured
Pallas fp8 rate, the same correction §1 applies to `mixed_prod`'s 38.5 → 48.4 µs. Using the V7X spec
rate instead would shrink each cost by that factor and D's `mixed_prod` row would read
669.11 → 665.24, i.e. −0.58%, rather than +0.90% — **the sign of D's byte ledger at `mixed_prod` is
inside the rate correction.**
That is another reason §6.3's conclusion is "D's prize is not the bytes.")

**As a two-pass, D loses on the modelled arithmetic alone, before its unpriced mechanism is
considered.** That is worth stating plainly: the byte prize the clean-sheet derivation makes so
vivid — 822 MB against 26 MB of real boundary traffic — is smaller than the cost of recomputing the
scores to avoid storing them.

D's actual prize is elsewhere, and it is much larger:

| term | at `mixed_prod` | tag |
| --- | ---: | --- |
| scores traffic deleted | −42.37 µs | `modelled` |
| SparseCore selection deleted (`topk ctrl`) | −399.60 µs | `measured (j-5f8f4a08-20b4-4fa0-bfc3)` |
| second einsum pass | +48.40 µs | `modelled` |
| **TensorCore-side selection** | **`unmodelled`** | |

| shape | old µs `measured` | new µs, TC selection struck | Δ time % |
| --- | ---: | ---: | ---: |
| `mixed_prod` | 669.11 | 275.54 | **−58.82** |

**Strike test on D.** Strike the `unmodelled` TC-selection term and D wins by 59%. Price it at
anything above ~394 µs and D loses. **One unmodelled term spans the entire ranking**, which is the
definition of a defect at this stage. §9.4 item 1, and §12 makes its probe step 3's second job.

### 6.4 Correctness gate

D changes the algorithm, so bit-exactness is not available and the rank criterion is the gate — same
as C's, at all five regimes with the negative control. If the mechanism chosen is the streaming
merge, D is *exact* and should reproduce bit-identical margins; if it is threshold-first with a
τ found by bisection, it is exact only if the final pass emits all ties, which at `k = 2048` and a
tie at τ requires an emit-then-truncate step that must itself be order-deterministic. **That
determinism requirement is D's real correctness hazard**, not the arithmetic.

### 6.5 File plan — and the ruling on how the layout sheet's two tables compose

`step2/sheets/layout-code-style.md` §10 gives two decompositions and does not say how they compose;
it flagged that for this document. **The ruling:**

The first table decomposes the **operation** (`api.py` / `base.py` / backend policy / kernel). The
second decomposes the **kernel file**, and applies when that file projects past the sheet's ~250-line
split point. They compose as a directory, and where the reference's names and this repo's idiom
conflict, **the repo's names win** — `deepseek_v4/compress_and_store/` and
`experimental/batched_rpa_longctx/` have both adopted the second table's shape and neither has ever
used the first table's names. Concretely:

```
deepseek_v4/streamindex_topk/
  api.py        the free function, jaxtyping, the backend fallback chain. No config.
  base.py       the Op subclass, bind() validation, the plain-XLA reference _fwd, the vjp.
  config.py     the frozen/kw_only/slots config, _get_heuristics_config, supported_on.
  wrapper.py    run_scores_kernel and the band invocations. (This is the first table's
                pallas_mosaic_tpu.py; `config.py` and `wrapper.py` are that file split, not
                two extra files — which is how the overlap the sheet flagged resolves.)
  kernel.py     the pallas_call, the body, the index maps.
  compute.py    arrays in, arrays out. No refs.
  load_store.py refs in, arrays out.
  dma.py        _async_copy, the fetches, the semaphore rotation. The only file that can deadlock.
  tiling.py     ints only.
```

`sparsecore_topk.py` is a **different operation** and gets its own group; it is not a tenth file
here.

**Two things this ruling deliberately does not do.** It does not bundle the restructure into
Options A, B, C or E — bundling an 865-line refactor with a perf change makes the A/B unattributable,
which is the same reason this repo forbids editing the benchmark harness under a measurement. And it
does not let the incumbent keep its current position: the layout sheet's §10 sanctions an
**opt-out** for a kernel doing dynamic-offset DMA, in-kernel metadata construction and selective
triple buffering — *"say so in the module docstring and leave"* — and notes that *"what is not
legitimate is half-using it."* `streamindex_topk.py` has silently taken the third path. **Whichever
option ships first should add that docstring paragraph**, one paragraph, as the cheapest possible
discharge of the obligation.

D's own plan, being a rewrite, adopts the full layout:

| file | new/changed | what | ~lines |
| --- | --- | --- | ---: |
| `.../streamindex_topk/api.py` | new | as above | ~90 |
| `.../base.py` | new | includes the XLA `_fwd` reference, which **does not exist in the tree today** — the only reference implementation is inside `tests/kernels/test_dsv4_streamindex_topk.py:100-106` | ~190 |
| `.../config.py` | new | | ~120 |
| `.../wrapper.py` | new | | ~110 |
| `.../kernel.py` | new | body + the top-`k` state machine | ~450 |
| `.../compute.py` | new | scoring; the merge or the threshold pass | ~200 |
| `.../load_store.py` | new | | ~110 |
| `.../dma.py` | new | | ~200 |
| `.../tiling.py` | new | | ~70 |
| `streamindex_topk.py`, `sparsecore_topk.py` | deleted from this path | `sparsecore_topk` stays for any other caller | — |
| tests | changed + new | | ~250 |

~1790 lines. **D is between one and two orders of magnitude more work than E**, and that asymmetry
is a first-class input to §9.

---

## 7. The VPU hypothesis — and which way the ranking moves if it is true

### 7.1 The claim

`probes/llo_capture_RESULT.md`'s static LLO analysis of the shipped mixed band, over 11,602 bundles:

| unit | issues | % of bundles | slot utilization |
| --- | ---: | ---: | ---: |
| VALU | 43,931 | 97.0% | **94.66%** |
| VPOP | — | 37.66% | — |
| VSTORE | — | 37.56% | — |
| VSTORE:SPILL | 8,460 | 36.46% | — |
| **MXU** | 8,182 | **35.26%** | — |
| VLOAD | — | 33.15% | — |
| VLOAD:FILL | 10,133 | 29.11% | — |
| XLU | — | 10.16% | — |
| SALU | — | 1.44% | — |
| fully idle | 7 | 0.1% | — |

18,593 spill-and-fill issues; 97% of VSTOREs are spills. **If true, the mixed band is VPU-saturated,
not MXU-bound, and every option in this document is ranked against the wrong floor** — `combined.md`
and §1 both rank mixed by its distance from an MXU/HBM floor that has no VPU term at all.

### 7.2 Why it is a hypothesis and not a result

- It is **static compile-time analysis**, not a measurement. Nothing here observed a cycle.
- `/xprof-llo`'s own SKILL.md records the identical verdict being handed to two kernels **3.56×
  apart** in real time. A signal that cannot separate those two cannot separate these.
- The spill signal ships **self-labelled** `[TRACK RECORD 0/5 — directly refuted twice]`.

Per the standing instruction: **no spill-reduction candidate is opened on it.** There is no Option F.

### 7.3 What makes it more credible than its track record suggests, and this is new

Area 5 built an independent VPU op-slot model and **falsified it against the block's own
measurement**: at the floor of 2.0 slots per element of `s`, mixed models to **624 µs** against a
`measured` 269.5 µs for everything the TensorCore does. The sheet's conclusion was that the VPU rate
is wrong and the time is `unmodelled`.

Invert it. 624 / 269.5 = **2.32**. If the VPU runs 2.32× the reference rate, the VPU-only model
predicts 269 µs — **the entire measured scoring time, to within a percent.** So:

> The two pieces of evidence are the same fact seen twice. The LLO says the VPU is saturated; the
> op-slot model says a saturated VPU at a plausible corrected rate accounts for **100%** of mixed's
> scoring time. And nothing else on the record accounts for it: the fp8 MXU floor is 48.4 µs, 18% of
> it, and area 3's residual of **~6.2 µs per inner step is neither DMA nor MXU and is `unmodelled`.**

That does not confirm the hypothesis — an exact fit obtained by inverting for the one free parameter
is weak evidence, and it is exactly the kind of reasoning a 0/5 track record should make one
suspicious of. But it upgrades it from "an instrument that has been wrong five times" to **"the only
account on the record that closes mixed's residual."** That is worth saying, and it is worth one
cheap job.

### 7.4 The measurement that would confirm or kill it

A two-arm Pallas strike test at mixed's exact per-inner-step shape, one shard, minutes:

- **Arm 1 (VPU only)** — `relu`, the `w` multiply and the 32-head sum over `s` of shape
  `(bq_sz=128, 32, bkv_sz=2048)` = 8.39e6 elements, with the einsum replaced by a compile-time
  constant of the same shape.
- **Arm 2 (MXU only)** — the fp8 einsum at the same shape, with the VPU chain replaced by a slice.
- **Control** — both, i.e. today's body. Should reproduce ≈8.42 µs per inner step
  [`measured (j-5f8f4a08-20b4-4fa0-bfc3)`, 269.51 µs ÷ 32 inner iterations].

**Confirmed** if arm 1 alone lands near 8.4 µs. **Refuted** if arm 1 is ≲2 µs, in which case the
~6.2 µs/step residual is somewhere neither this document nor any sheet has looked, and *that* is the
finding.

P3 came close to this and did not answer it: `g1_flat_flat` did arithmetic on 262,144 elements in
719.7 ns/step, but 262,144 is `s_summed` **after** the head reduction — the VPU chain runs on 32×
that. Naively scaling gives 23.0 µs against a measured 8.42 µs, which mostly shows that P3's
719.7 ns is dominated by fixed grid-step cost rather than by its arithmetic. **P3's arms do not
transfer; the probe above has to be run.**

### 7.5 If it is true, the ranking moves — here is exactly how

A ranking unchanged by this is a ranking that has not taken it seriously. This one changes:

1. **A new option enters at or near the top, and it is not in this document.** The largest VPU-work
   reduction available is area 6's **A1 — drop the ReLU**. Without it the head axis collapses into
   the contraction, the VPU tail vanishes entirely and matmul FLOPs fall **32×**. It is currently
   `unranked` as an approximation with no gate. If the VPU is the bottleneck, A1 stops being a
   curiosity and becomes the single largest lever in the space — and its gate already exists (the
   rank criterion, §5.3), so gating it costs one shard. **A1 is not proposed here** precisely
   because the hypothesis is unconfirmed; §12 puts the confirming probe before it.
2. **B, C and D keep their prizes but lose their justification.** Their savings are HBM bytes and
   SparseCore time, neither of which a VPU-bound TensorCore makes cheaper — so the arithmetic in
   §4.3, §5.2 and §6.3 stands. What falls is the framing: "mixed is at 18% of its floor, so there is
   5.6× of headroom" becomes false, and the mixed prizes stop being the down payment on a much
   larger one. **B drops below A**, because A's decode prize is against a floor that is not in
   dispute.
3. **E is untouched.** Its prize is `measured` end to end. A measured number does not care what the
   bottleneck is.
4. **C gets worse, not merely less attractive.** A 16-bit key adds a VPU-side encode at the store
   boundary — cheap in bytes, not free in slots — on a unit that would then be the bottleneck.
5. **D gets better in one specific way**: moving selection to the TensorCore is a terrible idea if
   the TensorCore's VPU is already saturated, so D gets *worse* on that axis — but D's second einsum
   pass, which is the term that sinks it in §6.3, is MXU work on a machine with 65% of its MXU
   bundles idle. D's cost model would need re-deriving, and its verdict is genuinely uncertain
   rather than negative.

**Ranking under the hypothesis: A, E, B, C, D — with A1 gated and inserted above all of them if its
recall gate passes.** Against §9's ranking of E ≈ B, A, C, D. Two of the top three swap and a
sixth option appears. That is what "taking it seriously" costs.

---

## 8. Two things about scope, stated because they change the ranking

### 8.1 The A/B is long-context, and this ranking is weighted for it

The end-to-end A/B is GLM-5.2 at ISL 65536 / OSL 1024 / `max_model_len` 66560. Per request that is
**64 chunked-prefill (mixed) steps and 1024 decode steps**. Mixed steps are 5–30× the cost of decode
steps at these shapes (`mixed_64k` 4.76 ms against `decode_64k` 0.20 ms), so the two phases
contribute *comparably* to indexer time — which means **neither a decode-only nor a mixed-only
option dominates, and the split is unmeasured.** §9.2 makes that the flipping assumption.

The A/B shape has a second property that matters more than its length: `num_sublanes_total` derives
from `max_model_len`, and during those 64 prefill chunks `kv_len` sweeps from 1024 to 65536 while
the buffer stays at 528 sublanes throughout. **Mean live fraction across the prefill ≈ 50%; the
first sixteen chunks run below 25%.** That is the confound shape, and it is where Option B's prize
lives. It is also, per area 8, where the *current* code has a `modelled` ≥2× regression on the
SparseCore side.

### 8.2 Non-kernel candidates are ranked alongside the kernel ones

Option E is a static tuple. B's largest single term is deleting an XLA fill. Neither is a Pallas
edit and both are ranked here on their merits, because in this exercise **the largest measured win
so far is a one-line change to a compile-time constant** — `bkv_p` at −9.25% / −11.33%
[`measured (j-98399433-7763-4385-81ca)`], which is larger than any modelled kernel-body prize in
this document except A's.

One candidate is *not* ranked, for a reason that must not be misread. **C12** (batch the MIXED
band's sequence axis) is unmeasurable on the current instrument: at the instrument's mixed shape
there is exactly one sequence, so there is nothing to batch. Its absence from step 3's results table
**must not be read as a negative result.** It is folded into B's area-2 decision as an enablement,
and it needs a production-shape measurement it cannot get today.

---

## 9. Ranking

### 9.1 The ranking

Weighted for the A/B shape (§8.1), and for cost-to-build, which at these magnitudes is not a
tiebreaker but a first-order term.

| rank | option | best `Δ time %` and where | tag of the prize | correctness evidence today | build cost |
| ---: | --- | --- | --- | --- | --- |
| **1=** | **E — Retune** | −9.25 (`decode_prod`), −11.33 (`decode_64k`) | `measured (j-98399433)` | **bit-identical margins on all six figures, negative control firing 1024/1024** `measured (j-e13fc627)` | 1 line |
| **1=** | **B — Right-Size** | ≤ −5.40 (confound shape, TC side); −2.53 (`mixed_prod`); plus 0.7–0.9 ms SC-side | `modelled` | none — needs the poison-fill arm (§4.4) | ~230 lines + a new test file |
| **3** | **A — Narrow Record** | −13.42 (`decode_prod`), −8.36 (`decode_64k`) | `modelled` | none — but bit-exact by construction | ~115 lines **across two teams** |
| **4** | **C — Half-Width Key** | −2.11 (`mixed_prod`), −1.96 (`mixed_64k`) | `modelled` | none — needs the accuracy gate (§5.3) | ~135 lines + a new test file |
| **5** | **D — No Round Trip** | +0.90 as a two-pass; **−58.82 with the TC-selection term struck** | `unmodelled` decides it | none | ~1790 lines |

**Ranks 1 and 1= are a printed tie, not a hedge.** E's prize is measured and B's is modelled at a
shape no job has run; the margin test (§9.3) says the gap is smaller than the uncertainty behind it.
More usefully, **they are not alternatives**: E is a config change whose ceiling is B's masked store
(§4.2). The operational answer is *build E first because it is one line and already correctness-
verified, then build B, and measure the composition rather than the sum.*

**A ranks third despite having the largest single `Δ time %` in the table.** That is a judgement,
stated as one: A's prize is real and its correctness argument is the strongest of the modelled
options, but the row that decides whether it is net positive — the writer-side scatter cost — is
`unmodelled` and lives in a file this exercise does not own (§11 item 1). A change that needs a
second team's schedule is not comparable to a one-line change on a scale of percent-per-hour.
**If the writer's owner is already engaged on §11 item 1, A moves to rank 1 outright.**

**D ranks last on arithmetic, not on uncertainty**, and that distinction matters: as a two-pass it
is *slower* by the modelled ledger before its unpriced term is considered (§6.3). It stays on the
list — and stays interesting — only because its real prize is the SparseCore's 33–60%, and that is
decided by one `unmodelled` number that a cheap probe can produce.

### 9.2 The single assumption that would flip it

> **That decode and mixed contribute comparably to indexer time at the A/B shape.**

E's measured prize is decode-only; at mixed it is `+1.61% / +1.67%`, inside spread. B's prize is
mixed-only, and specifically confound-shape-only. If mixed dominates the A/B's indexer time — which
it will if the prefill chunks are more expensive than the arithmetic in §8.1 suggests, or if the
benchmark runs at a concurrency that keeps many requests in prefill simultaneously — then **E's net
effect at the A/B is near zero or slightly negative and B is rank 1 outright, with A second.** If
decode dominates, **E and A are ranks 1 and 2 and B falls to third**, because at near-full `kv_len`
B collapses to the fill deletion alone (−0.63% at decode).

This is one unmeasured quantity and one cheap job settles it: count indexer-band invocations and
their `kv_len` over 60 s of the A/B workload. §12 makes it step 3's first job.

### 9.3 Strike tests

**Strike test** — strike an `unmodelled` or unpriced term from *both* options and see whether the
ranking moves; if it does, the term must be measured.

| term struck | ranking before | ranking after | verdict |
| --- | --- | --- | --- |
| TC-side top-`k` cost | E≈B, A, C, D | **D**, E≈B, A, C | **moves — must be measured.** §9.4 item 1 |
| writer-side scatter cost at the split format | E≈B, A, C, D | **A**, E≈B, C, D | **moves — must be measured.** §9.4 item 2 |
| the mixed residual (§7) | E≈B, A, C, D | A, E, B, C, D | **moves — must be measured.** §9.4 item 3 |
| the decode:mixed step mix | E≈B, A, C, D | either E,A,B,C,D or B,A,E,C,D | **moves — must be measured.** §9.4 item 4 |
| the SparseCore partition collapse under a 16-bit key | E≈B, A, C, D | unchanged (C stays 4th either way) | does not decide; may stay `unmodelled` |
| the mixed out-DMA's duty-cycle improvement under B | E≈B, A, C, D | unchanged | does not decide |

**Margin strike test** — if the rank1–rank2 gap is smaller than the combined uncertainty of the
non-measured terms behind it, print a tie. E vs B: E's gap over B rests entirely on B's byte counts
(`modelled`), B's bracketed denominator (`≤`, not `=`), and B's SparseCore row (a 0.7–0.9 ms
**range**, from a rate area 8 itself calls 3.2–7.2× optimistic). Those uncertainties exceed the gap.
**Tie printed.** The range does not collapse to a point anywhere it is carried.

A vs B: the same test, driven by a different term — A's prize is decode-side and B's is
confound-shape-side, so the comparison reduces to §9.2's unmeasured step mix. **The A/B gap is also
inside its uncertainty**, but A is ranked below B on the ownership judgement stated in §9.1 rather
than on a number, and that is flagged as judgement rather than smuggled in as arithmetic.

### 9.4 Defect register — `unmodelled` terms that still decide between options

The user's rule: an `unmodelled` term that decides between two options is a defect at this stage.
Four remain. Each has a named, cheap probe.

| # | term | decides | probe |
| ---: | --- | --- | --- |
| 1 | **TensorCore-side top-2048-of-67584 cost** | D vs everything. Struck, D wins by 59%; priced above ~394 µs, D loses. | A standalone Pallas top-`k` microbenchmark at `mixed_prod`'s shape (1024 tokens × 67584 candidates), against the SparseCore's `measured` 399.60 µs. One shard. |
| 2 | **Writer-side scatter cost at the split cache format** | A vs B. Struck, A is rank 1. | Time the two-array scatter against the current `u8[424960,256]` one in XLA. No Pallas needed; the current one is `measured` at ~291 µs. |
| 3 | **Mixed's ~6.2 µs/inner-step residual — neither DMA nor MXU** | Whether the mixed prizes (B, C, D) are measured against a real floor at all. | §7.4's two-arm VPU/MXU strike test. One shard. |
| 4 | **The decode:mixed split of indexer time at the A/B shape** | E vs B (i.e. rank 1). | Count band invocations and `kv_len` over 60 s of the A/B workload. No TPU job of its own. |

Not defects, because they gate an option rather than order it: the 16-bit key's rank error at 2048
(gates C); the recompile count across B's bucket ladder (`assumed`, bounded by construction); the
poison-fill result (gates B's fill deletion).

### 9.5 Where this stops

Per the skill's exit table, applied to what step 3 would be iterating toward:

| achieved vs floor | meaning |
| --- | --- |
| ≥80% | done — stop iterating that variant |
| 50–80% | one more iteration, with an attribution for where the remaining time went |
| <50% | something structural is wrong; write the unpriced register rather than tuning |
| >100% | the floor is wrong, or the kernel is skipping work |

`decode_prod` sits at 55–61% today: **one more iteration with attribution**, and Options A and E are
that iteration. `mixed_prod` sits at 18%: **below 50%, so something structural is wrong** — and
§7 and §9.4 item 3 are this document's unpriced register for it, not a set of tuning candidates.
That is why no option in this document claims to close mixed's gap: on the evidence available, the
gap is not yet attributable, and 80% is a default rather than a measured constant.

---

## 10. Anything in the probe record that looks wrong

Three items, none of which changes a conclusion.

1. **The dispatch to this wave quoted a mixed regression of `+6.87%` at `(8,8,8)`.** No such row
   exists. `probes/bkv_p_RESULT.md` contains no `(8,8,8)` arm and states explicitly that
   *"`(8,8,8)` and `(8,4,4)` were not measured, so the long-context optimum is bracketed rather than
   found."* **No number for that tuple appears anywhere in this document**, and none should be
   quoted until an arm runs. The conclusion it was offered to support — that mixed regresses at
   raised tuples — is carried by the four arms that *were* measured (+1.61 to +5.11).
2. **The tuple slot order in `bkv_p_RESULT.md`'s terms section is wrong.** §2.4. No measurement
   moves; one design fact does.
3. **`bkv_p = 8` is not value-checked.** P5′ verified `(4,4,4)` bit-identically and is explicit that
   it did not verify 8. Option E takes 4, not 8, and at `decode_prod` that costs nothing — 4 and 8
   are tied there (0.15264 vs 0.15270 against 1.3–1.4% spreads). At `decode_64k`, 8 is 2.5 points
   better and **that arm is unverified**; if step 3 wants it, P5′'s own advice applies: attach
   `--tuples 8,8,8` to whatever job runs next, since the shard rides along free.

One further observation, offered as an inconsistency rather than an error: `combined.md` refers to
"nine constraint sheets" where the skill prescribes eight briefers. The directory does hold nine
files — the count is right, the SparseCore area was split out of area 5 and layout is its own sheet.
Nothing follows from it.

---

## 11. Carried

Real improvements this exercise measured or established that are **not** kernel edits to
`streamindex_topk`. Each carries what to change and where, the prize and the level it was measured
at, the correctness evidence, the constraint stopping it here, and where it routes.

**1. The writer-side scatter and reshape on `cache_kv`.**
- *What and where:* the `scatter_fusion` and the `reshape` on `u8[424960,256]` at
  `mla_attention_op.py:103-104`.
- *Prize and level:* ~291 µs **each**, `measured` off the prefill profile — a profile-scope reading,
  not an A/B. × 21 indexer layers ≈ 6.1 ms per model step each. Area 1's ledger rows X3 and X4 put
  the pair at −82% / −20% / −2.9% of the block's own call at `decode_prod` / `mixed_prod` /
  `mixed_64k`, which is a measure of how badly it dwarfs the block rather than of what the block can
  do about it.
- *Correctness evidence:* none. Nothing was changed; this is a profile reading.
- *Constraint stopping it here:* not a kernel edit and not in this kernel's file. It is also **Option
  A's blocker** — the same code must change for A, so the two should be taken together.
- *Routes to:* whoever owns `mla_attention_op.py:103-104`, via `/preflight-check` and then an
  end-to-end A/B.

**2. The `topk_indices_buffer[:ntok] = -1` fill.**
- *What and where:* the caller-side initialisation of the top-k output buffer; area 7's row #11 and
  its alternative **B6**, which that sheet names explicitly as *"a caller-side edit, not a kernel
  edit — a `carried:` item."*
- *Prize and level:* `modelled` 52.4 µs per model step at mixed, immediately overwritten.
- *Correctness evidence:* none, and it needs some: safe only if every row `[0:k]` is written for
  every valid token. Same poison-value technique as §4.4.
- *Constraint stopping it here:* outside `streamindex_topk.py`.
- *Routes to:* the same owner as item 1, or a standalone `/preflight-check`.

**3. `MAX_SLICE_WORDS`.**
- *What and where:* `sparsecore_topk.py`; area 8's alternative **A5**, which that sheet calls *"the
  cheapest change on this sheet."* Raise it from 65536 toward ~115000 — 65536 words is 53% of the
  480 KiB usable per vector subcore.
- *Prize and level:* `unmodelled`. The mechanism is concrete: `_pick_partition` picks `p=4`
  two-stage at `decode_64k` (**6 of 32 subcores busy in stage 2**) and `p=2` at `mixed_64k`; a larger
  slice should collapse each by a level, deleting a launch.
- *Correctness evidence:* none, and none is obtainable — see item 4.
- *Constraint stopping it here:* `sparsecore_topk` is a different operation, with its own four-file
  group, and it has no test.
- *Routes to:* `/write-tpu-kernels` as its own block, after item 4.

**4. `sparsecore_topk` has no test.**
- *What and where:* `grep -rn "sparsecore_topk" tests/` returns nothing. The kernel runs with
  `disable_bounds_checks=True` and a dynamic `woff`.
- *Prize and level:* not a perf item. It is the **precondition** for Options B and C — both change
  code inside that kernel — and for item 3.
- *Correctness evidence:* the absence of it is the item.
- *Constraint stopping it here:* out of scope for a scoring-half exercise, and it is a test rather
  than an optimization.
- *Routes to:* `/preflight-check`, whose stated job is to write the check when nothing exists. This
  is the case that skill's "empty explore result" rule was written for.

**5. E1's price, for a model that does not exist yet.**
- *What and where:* fold `w` into `q`, deleting the broadcast and the multiply.
- *Prize and level:* **13.70% faster at W=512, 29.28% at W=128**, `measured (j-efb89184-020f-495f-9d4b)`
  at the probe's shapes — the largest single effect in P3.
- *Correctness evidence:* **rejected**. `weights_proj` is ~50% negative on the real GLM-5.2-FP8
  checkpoint, 0 of 32 rows all-non-negative.
- *Constraint stopping it here:* the model's weights.
- *Routes to:* nowhere, today. Carried so that if some future checkpoint *does* guarantee a
  non-negative indexer weight, the value is already known and nobody pays for the measurement twice.

---

## 12. What step 3 should run first

In order, and the first four are cheap enough to be one or two fanned-out jobs on a four-chip pod.

1. **Settle §9.2** — the decode:mixed split of indexer time at the A/B shape. Needs no TPU job of
   its own. It decides rank 1.
2. **Run §7.4** — the two-arm VPU/MXU strike test. One shard, minutes. It decides whether the mixed
   half of this document is aimed at a real floor, and if it comes back confirmed, step 3 should
   stop and gate area 6's A1 before building anything here.
3. **Run §9.4 item 1** — the TensorCore top-`k` microbenchmark. One shard. It converts D from
   unrankable to ranked, in either direction.
4. **Add the confound regime to `instruments/shapes.py`** — `kv_len` 8192, `max_model_len` 66560.
   Step 1's five regimes must not be touched, for comparability; a sixth may be added. **Option B
   cannot be measured without it**, and today B's headline number is a bracket rather than a value
   for exactly that reason.
5. **Build E.** One line, already correctness-verified at `(4,4,4)`. Attach `--tuples 8,8,8` to
   whatever job it rides on, since P5′ asked for it and the shard is free.
6. **Build B's TensorCore half** — the masked partial-block store, the exact width, the deleted
   fill, with the poison-fill arm. Then measure **E composed with B**, not summed: B decouples the
   width from the tuple, which is precisely the term that made E cost mixed anything.
7. Then A, C and D, in whatever order §§1–4 have by then rearranged them into.

The composed candidate that step 4 raises a `final_report.md` from should be measured **as a whole,
in a single job, against the incumbent** — not as a sum of the per-variant numbers above. Options A
and E interact through the pipeline's dominant side (§3.1, area 3); B and C compose multiplicatively
on the same bytes (§5.2); B and E compose through the `max` (§4.2). None of those three compositions
is additive, and all three are in the top four.
