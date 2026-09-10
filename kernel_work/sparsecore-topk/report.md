# Step 1 — `sparsecore_topk`: the block, its floor, and the instruments

**Block**: `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`, function `_topk_body`
(lines 66–329), reached through `_sc_topk_direct` (419) and the public `sparsecore_topk` (517).
**Branch**: `hning/sctopk-pipeline`, based on `hning/sctopk` @ `b11b6eb6`.
**Request that opened this**: make the HBM→VMEM score DMA (`dma()`, lines 108–111) pipelined
instead of blocking, so it double-buffers.

**Sign convention, binding on every table and every sentence in this file and everything under it:**
a percentage is `(new − old) / old` of **time**. **Negative is faster, positive is slower.** Prose
says the direction in words ("5.3% faster"), never a bare signed number. A `speedup ×` column, where
present, is `old / new` and >1 is faster — a different number from the percentage.

---

## 1. What the block does

Exact, unsorted, per-row top-`k` over a matrix of fp32 scores, on the **SparseCore**, returning
`i32[b, k]` column indices, `-1` suffix-padded. It is a **radix/histogram selection**, not a sort:
it never orders the winners, it only finds the score threshold that admits exactly `k` of them.

### Terms used in this document

Several of these collide with a term of art elsewhere in this stack. Where they do, the collision is
named.

| term | meaning here | not to be confused with |
| --- | --- | --- |
| **row** | one of the `b` score rows, i.e. one query token's scores over its context | — |
| **slice** | the contiguous span of one row that one subcore owns, `slice_len` int32 words | a Pallas grid block; there is no grid here |
| **wave** | one iteration of `lax.fori_loop(0, num_waves, …)` (line 329). In one wave each of the 32 subcores processes **one row**, so a wave retires 32 rows | a DP/pipeline wave in the serving stack |
| **subcore** | one of 2 SparseCores × 16 vector subcores = **32 independent execution units** per TensorCore | a TensorCore core; a chip has 2 TensorCores |
| **pass** | one full traversal of the resident slice in VMEM, 16 lanes at a time | an HBM pass; all but one of these never touch HBM |
| **`P`** | the partition factor from `_pick_partition` (494): each row is split into `P` column slices, stage 1 runs a local top-k per slice, stage 2 merges the `P·k_p` candidates | tensor/data/expert parallelism degree. `P` is intra-kernel and invisible outside it |
| **`L`** | `slice_len` = `align_to(n_kernel, 16)`, the resident words per subcore | `n`, the caller's column count. On the two-stage path `L ≈ n/P`, not `n` |
| **`W`** | `walk_chunks` = `cdiv(i_hi, 16)`, the number of 16-lane vectors in the live prefix of a row | `L/16`; they are equal only when the row is full |
| **exposed DMA** | wall-clock in which the subcore is stalled on a DMA and doing nothing else | total DMA time. Today they are the same thing here — that is the finding, not an assumption |

### The algorithm, per row

The key transform is `monotone_key` (116): `bits ^ (bits>>31 & 0x7FFFFFFF)`. This maps the IEEE-754
bit pattern of an fp32 to an int32 whose **integer order equals the float order**, so a radix
selection on int32 buckets is a selection on float values. It is an **involution** — applying it
twice recovers the original bits — which is what lets the emit pass hand back original scores for
free (298–301).

Selection is four levels of a 256-bucket histogram, most-significant byte first:

1. **L0** at `shift=24`: histogram the top byte, prefix-sum it, and binary-search the cumulative
   counts for the bucket `b*` that straddles the `k`-th largest. Everything above `b*` is a
   guaranteed winner (`c_hi` of them); `quota = k − c_hi` more must come out of `b*` itself.
2. **Three refinement levels** at `shift=16, 8, 0` (48): re-histogram *only* the elements matching
   the prefix found so far, and narrow again. Each level is skipped, per row, once `done` is set —
   `done` becomes true when the quota exactly equals a bucket's population, or when the row has
   `total ≤ k` live elements and everything wins.
3. **Emit** (283): one final pass, taking every element strictly above the threshold plus enough
   tied elements to fill the quota, compacted with `plsc.store_compressed`.

`P > 1` runs this whole thing twice: once on `b·P` slices to get local candidates, once on the
`P·k_p` merged candidates.

### Diagram, with shapes, dtypes and tensor sizes on the edges

> ⚠ **Read §4 before this diagram.** It is drawn at `n=133120, P=4`, which is **not** the shipping
> shape. The checked-in GLM-5.2 config sets `--max-model-len 9216` → `n=10240`, and at the two
> production batch sizes that gives **`P=1`: a single kernel, no stage 2, and neither gather**. The
> P=4 diagram is kept because it is the shape all prior work on this kernel was measured at and the
> shape the incumbent's design is tuned for — but the single-stage picture in §4 is the one
> production runs.

Drawn at the **two-stage shape**: `b=1024`, `n=133120`, `k=2048`, `P=4`. Byte counts are
for the whole call, not per subcore.

```
                      scores  f32[1024, 133120]           545.26 MB in HBM
                                  │
                                  │ bitcast (free, no op emitted)
                                  ▼
                      words   i32[1024, 133120]           545.26 MB
                                  │
                                  │ reshape → i32[4096, 33280]   (XLA, free: contiguous)
                                  ▼
        ┌──────────────── STAGE 1  kernel sc_topk_direct_b4096_n33280_k2048_keys ────────────────┐
        │  128 waves × 32 subcores × 1 row                                                       │
        │                                                                                        │
        │   HBM ──DMA 130 KiB/subcore/wave──▶ resident_vmem  i32[33280]   (130 KiB, 25.4% of     │
        │        4.06 MiB per wave, 545.26 MB total          the 512 KiB subcore VMEM)           │
        │                                                                                        │
        │   then 3–6 VMEM passes over the same 130 KiB, none of which touch HBB:                 │
        │     cvt(rd+wr) │ hist0(rd) │ ref×3(rd, each skippable) │ emit(rd)                      │
        │                                                                                        │
        │   scratch, per subcore:  priv_vmem  i32[4112]  16.4 KiB  (16 lane-private histograms)  │
        │                          glob_vmem  i32[272]    1.1 KiB                                │
        │                          rowbuf     i32[2064]    8.3 KiB                               │
        │                          keybuf     i32[2064]    8.3 KiB                               │
        │                          len_vmem   i32[4096]   16.4 KiB                               │
        │                                     ───────────────────                                │
        │                          total resident            180.5 KiB  of 512 KiB = 35.3%       │
        └────────────────────────────────────────────────────────────────────────────────────────┘
                    │                                        │
                    ▼ local_slots i32[4096,2048]             ▼ cand_scores f32[4096,2048]
                      33.55 MB out                             33.55 MB out
                    │                                        │
                    │ reshape+offset add (XLA)               │ reshape (XLA, free)
                    ▼                                        ▼
       global_cand_indices i32[1024, 8192]        cand_scores f32[1024, 8192]
                    │                                33.55 MB
                    │                                        │
                    │                 ┌──────────────────────┘
                    │                 ▼
                    │   ┌── STAGE 2  kernel sc_topk_direct_b1024_n8192_k2048 ──┐
                    │   │  32 waves × 32 subcores × 1 row                      │
                    │   │  DMA 32 KiB/subcore/wave, 33.55 MB total             │
                    │   │  resident_vmem i32[8192] = 32 KiB                    │
                    │   └──────────────────────────────────────────────────────┘
                    │                 │ final_cand_slots i32[1024,2048]  8.39 MB
                    │                 ▼
                    └──────▶ take_along_axis (XLA gather, s32[1024,8192] source, 33.55 MB)
                                      │
                                      ▼
                            final_indices i32[1024, 2048]   8.39 MB
```

---

## 2. Op inventory

**One row per op. `unit today` is where the work runs *now*** — not where it should run, which is
step 2's question and is deliberately not answered here.

Counts are **per row per wave** unless stated. `L` = `slice_len` words, `W = cdiv(i_hi,16)` vectors.
At the P=4 stage-1 shape `L = 33280`, `W = 2080`. "FLOPs" is nil throughout: **this block contains
no floating-point arithmetic and no matmul at all.** Every operation is integer or a memory
movement, so **the block has no MXU term and no FLOP term**, and any roofline that gives it one is
manufacturing headroom that does not exist.

| # | op | `file:line` | shape in → out | dtype | int-ops / row | bytes moved / row | unit today |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lengths DMA | 103–106 | `i32[b_pad]` HBM → VMEM | i32 | — | `4·b_pad`, **once per launch**, not per wave | SC DMA |
| 1 | **score DMA** | 169 | `i32[L]` HBM → VMEM | i32 | — | **`4L` = 130 KiB** HBM read | **SC DMA, blocking** |
| 2 | `cvt` monotone_key | 171–175 | `i32[L]` → `i32[L]` in place | i32 | `3L` (shr, and, xor) | `8L` VMEM (rd+wr) | SC VPU |
| 3 | `zero_priv(256)` | 120–126 | → `i32[256]` | i32 | — | `1 KiB` VMEM wr | SC VPU |
| 4 | `hist0_body` | 187–196 | `i32[L]` → `i32[4096]` | i32 | `~6L` + `L` scatter | `4L` VMEM rd, `4L` scatter wr | SC VPU + scatter |
| 5 | `fold_priv(16)` | 128–137 | `i32[4096]` → `i32[256]` | i32 | `4096` add | `16 KiB` VMEM rd | SC VPU |
| 6 | `scan_glob(16)` | 139–146 | `i32[256]` → `i32[256]` | i32 | `256` cumsum | `2 KiB` VMEM rd+wr | SC VPU |
| 7 | `find_bucket(16)` | 148–154 | `i32[256]` → scalar | i32 | `256` cmp + popcount | `1 KiB` VMEM rd | SC VPU |
| 8–10 | **3× refinement**: `zero_priv` + `ref_body` + `fold_priv` + `scan_glob` + `find_bucket` | 223–265 | `i32[L]` → scalar | i32 | `~8L` + `L` scatter, **per level**, each skippable per row via `done` | `4L` VMEM rd per level | SC VPU + scatter |
| 11 | `fill_body` | 271–281 | → `i32[k]`(+`i32[k]`) | i32 | — | `4k`(+`4k`) VMEM wr, **only when `total < k`** | SC VPU |
| 12 | `emit_body` | 283–311 | `i32[L]` → `i32[k]`(+`i32[k]`) | i32 | `~7L` + cumsum + popcount | `4L` VMEM rd, `4k` compressed wr | SC VPU |
| 13 | out DMA | 315–319 | `i32[k]` VMEM → HBM | i32 | — | `4k` = 8 KiB HBM wr | SC DMA, blocking |
| 14 | keys DMA (`emit_keys` only) | 320–325 | `i32[k]` VMEM → HBM | i32 | — | `4k` = 8 KiB HBM wr | SC DMA, blocking |
| 15 | offset-add + `where` | 594–598 | `i32[b,P,k_p]` → `i32[b,P·k_p]` | i32 | `b·P·k_p` | `2 × 33.55 MB` HBM | **TensorCore (XLA)** |
| 16 | `take_along_axis` | 617–619 | `i32[b,P·k_p]`, `i32[b,k]` → `i32[b,k]` | i32 | gather | `33.55 + 8.39 MB` HBM | **TensorCore (XLA gather)** |

**HBM traffic per row per wave, stage 1: one read of `4L`, two writes of `4k`.** Everything in rows
2–12 is VMEM-resident. That ratio — **1 HBM pass against 3–6 VMEM passes** — is the whole roofline
story and is developed in §5.

**Ops 15 and 16 run on the TensorCore, not the SparseCore.** They are inside the block's edge (they
are emitted by `sparsecore_topk` itself) and they are *not* inside either Pallas kernel. They are
priced in §5 and they are the reason the block floor is not simply the SparseCore floor.

### Rows the inventory cannot fill from source alone

| row | what is `unknown` | what would settle it |
| --- | --- | --- |
| 8–10 | ~~how many of the 3 refinement levels actually run~~ — **RESOLVED below, exactly, without hardware** | — |
| 4, 8–10 | the cost of `plsc.addupdate_scatter` relative to a plain vector load. Prior work records that its cost tracks **mask density**, and the L0 histogram is a *dense* scatter (every lane valid) — the expensive end | a probe sweeping a pass with and without the scatter |
| 2–12 | the **SparseCore VPU throughput for a VMEM-resident pass**. `references/08-sparsecore.md` measures only *HBM-bound* streaming (337 GB/s); nothing in `references/` measures the rate when the data is already in VMEM, which is what every one of these passes is | probe `03-passes`, §6 — in flight |

### Resolved: the refinement count, and the imbalance it creates

`done` (258) is data-dependent, so rows 8–10 could contribute anywhere from 0 to 3 passes — a 2×
swing in the whole compute term. **This did not need a probe.** `done` is a deterministic function
of the data, so the exact trip count is computable in numpy from the kernel's own arithmetic
(`monotone_key` → `(key>>24)+128` → inclusive prefix sum → `find_bucket` → `quota==cnt | quota==0`).
Simulated at `n=10240, k=2048`, seed 20260904 — `probes/05-refinement-count/probe.py`, rerunnable in
about a second on the host:

| distribution | `b` | mean levels/row | critical-subcore levels/row | **imbalance** | **effective passes over `L`/row** |
| --- | --- | --- | --- | --- | --- |
| random N(0,1) | 64 | 1.95 | 2.50 | **1.28×** | **5.50** |
| random N(0,1) | 1024 | 1.92 | 2.03 | 1.06× | **5.03** |
| uniform[0,1) | 64 | 2.05 | 2.50 | 1.22× | 5.50 |
| uniform[0,1) | 1024 | 2.06 | 2.25 | 1.09× | 5.25 |
| heavy_ties (17 values) | 64 / 1024 | 3.00 | 3.00 | 1.00× | 6.00 |

`effective passes = cvt + hist0 + refinement + emit = 2 + levels + 1`.

**So the answer is ≈5.0 passes at prefill and ≈5.5 at decode — the top of the 3–6 range, not the
middle.** That is the number §5's compute term is built on, and it is `modelled` only in the sense
that the score *distribution* is assumed; the pass count given a distribution is exact.

**The distribution assumption is load-bearing and is not a free parameter.** `uniform` moves the
prefill critical count from 2.03 to 2.25 — 11% on the term that is half the kernel — and
`heavy_ties` pins it at 3.00. The kernel's real input is a DSA score vector, not a Gaussian, and
nothing here establishes which row of this table it resembles. §5d's fitted per-pass price is
therefore conditional on the `random` row; a step-2 candidate whose prize depends on the *count*
rather than on the per-pass *cost* has to re-derive it, and the three rows bracket how much that can
move.

**The imbalance column is a second finding, and it is not about the DMA.** Rows map to subcores as
`row = w*32 + flat` (157), and lines 19 and 528 both state the design has **zero inter-subcore
synchronization** — there is no `subcore_barrier` in the file. So the kernel costs
`max over 32 subcores of the sum of that subcore's own rows`, not the sum of per-wave maxima. With
32 rows per subcore at prefill the early-exit variance averages out (**1.06×**); with **2 rows per
subcore at decode it does not** (**1.28×**). Roughly a fifth of decode's refinement work is the
critical subcore waiting on nothing.

This is a real, separately-addressable inefficiency with an obvious shape (work-stealing, or
ordering rows so long and short ones pair up) and it is **orthogonal to the DMA question the caller
asked about**. It is recorded here and carried to step 2 as a candidate rather than folded into the
DMA work. Note it is bounded by the same denominator as everything else in this block (§7): 1.28×
on a quarter of one term of a kernel that is ~1% of decode.

---

## 3. What the block excludes

Named, because §5 divides a measurement by a floor and a floor over part of the computation gives an
achieved fraction for part of the computation.

**Inside the block** (rooflined in §5): everything `sparsecore_topk` emits. **What that is depends
on `P`, and the two cases are structurally different blocks:**

- **`P=1` — both shipping shapes (`b=64` decode, `b=1024` prefill, `n=10240`).** One Pallas kernel
  and nothing else. No stage 2, no offset add, no `take_along_axis`, no TensorCore work at all. Ops
  15 and 16 of §2 **do not execute**, so the block is 100% SparseCore and its floor is a pure
  SparseCore floor.
- **`P>1` — `b=16` at `n=10240` (`P=2`), and everything at `n=133120` (`P=4`).** Two Pallas kernels
  plus the three XLA ops between and after them. Here the block is *not* one kernel, and scoping
  the exercise to stage 1 is the specific failure the skill warns about.

The consequence for this exercise is sharp and is carried into §6: **the DMA the caller asked about
lives in a kernel that, at both shipping shapes, is the entire block.** There is no stage-2 or
gather time for a stage-1 win to be diluted against — but equally, there are only 2 waves at decode,
which is where the mechanism has least to work with (§5).

**Outside the block, and not priced:**

- **Everything upstream that produces `scores`** — the DSA indexer's query·key product. This is a
  TensorCore matmul and is a much larger cost than the top-k; it is a different block.
- **Everything downstream that consumes `final_indices`** — the `sc_dsa_gather` that reads the
  selected KV. Prior measurement records **zero cross-layer overlap** between the gather and
  anything else, so there is no overlap to reclaim at that seam either.
- **`streamindex_topk`'s own wrapper**, including `_effective_row_lengths` and the `enable_early_exit`
  branch (which defaults off, so no `lax.cond` stands between production and this kernel).
- **The 21 of 78 GLM-5.2 layers that call this at all.** The e2e denominator is a separate number
  and is stated in §7 rather than folded into the floor.

**Serialization is a term in the floor, not a free lunch.** Stage 1 → gather-free glue → stage 2 →
gather is a **strictly serial chain**: each step consumes the previous step's output. Prior
measurement records the whole call at **94.7–95.1% unoverlapped at decode and 99.8% at prefill**, so
there is no concurrent TensorCore work hiding any of it. The floor in §5 therefore **adds** the
stage terms rather than taking a max over them, and that is stated in the table.

---

## 4. Shape derivation

**Derived from source, not from the caller and not from the existing harness.** Every value has a
`file:line`. Paths are relative to the `vllm-torchtpu` checkout unless prefixed `vllm/`.

### The call chain

One call site of `sparsecore_topk` exists in the whole tree —
`kernels/deepseek_v4/streamindex_topk.py:831` — reached by two chains:

- **Chain A, GLM-5.2 / DeepSeek-V3.2** (`compression_ratio=1`): `Glm4MoeLiteMLAAttention` →
  `VllmTPUMultiHeadLatentAttentionWrapper` (`layers/vllm/custom_ops/mla_attention_op.py:501`
  retypes the indexer in place) → `VllmTPUSparseAttnIndexer.forward_oot` (`:146`) → `topk_op`
  (`:126`) → `_streamidx_topk_jax` (`:107`) → `streamindex_topk`.
- **Chain B, DeepSeek-V4-Flash** (per-layer `compress_ratio > 1`):
  `models/vllm/deepseek_v4/attention.py:140` → `deepseek_v4_indexer.py:222` → `:157` → `:83` →
  `streamindex_topk`.

The directory is named `deepseek_v4`, but **the model actually exercised is GLM-5.2** — Chain B's
`index_topk` and `compress_ratios` are not in this checkout at all (see unknowns).

### Variable table

| variable | value | source `file:line` | deployment |
| --- | --- | --- | --- |
| `k` | `config.index_topk` = **2048** | field read `vllm/vllm/model_executor/models/deepseek_v2.py:663`; value from checkpoint config, recorded `knowledge/models/GLM-5.2-FP8/refs/model.md:57` | GLM-5.2-FP8 |
| `n` | **10240** | `streamindex_topk.py:603-605` (`align_to(pages_per_seq, bkv_p) * page_size // 128 * 128`), `page_size=1024` from `layers/vllm/attention.py:1323`, `bkv_p=2` from `mla_attention_op.py:119`, `MAX_MODEL_LEN=9216` | GLM-5.2-FP8 |
| `b` | token bucket, `∈ {16,32,64,128,256,512,1024}` | ladder `platforms/tpu_platform.py:1247-1253`, built at `:909-913` with `min_token_size=16`; selection by `bisect_left` at `layers/common/sequence_layout.py:289-290` | both |
| `b` decode | **64** — running seqs rounded to a bucket | `MAX_NUM_SEQS=64`, `scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh:54` | GLM-5.2-FP8 |
| `b` prefill | **1024** | `MAX_NUM_BATCHED_TOKENS=1024`, `glm-5.2-fp8-dp16-ep.sh:49` | GLM-5.2-FP8 |
| `P` | `_pick_partition(b, n, k)` | `sparsecore_topk.py:494` | derived, table below |

**`n` is a padded, page-aligned KV *capacity*, not a live context length** — it does not vary with
sequence length. Live length enters only through `row_lengths`. **`b` is a token bucket with floor
16**, not `max_num_seqs`; `max_num_seqs` sizes `seq_lens`/`page_indices` only.

### The partition table — executed, not hand-computed

Produced by importing the real `_pick_partition` / `_align_to` and the real `topk_n` arithmetic:

```
     b       n |  P    n_p  slice wav1 KiB/sc |   cand wav2  path
    16   10240 |  2   5120   5120    1     20 |   4096    1  2-stage
    32   10240 |  1  10240  10240    1     40 |      -    -  1-stage
    64   10240 |  1  10240  10240    2     40 |      -    -  1-stage   ← DECODE
   128   10240 |  1  10240  10240    4     40 |      -    -  1-stage
   256   10240 |  1  10240  10240    8     40 |      -    -  1-stage
   512   10240 |  1  10240  10240   16     40 |      -    -  1-stage
  1024   10240 |  1  10240  10240   32     40 |      -    -  1-stage   ← PREFILL
    16  133120 |  4  33280  33280    2    130 |   8192    1  2-stage
  1024  133120 |  4  33280  33280  128    130 |   8192   32  2-stage
```

**Three consequences, and each one moves this exercise:**

1. **`n = 10240`, not 133120.** The 133120 row needs `--max-model-len 132096`, which **no
   checked-in config sets**. Both `glm-5.2-fp8-dp16-ep.sh:18` and
   `deepseek-v4-flash-dp8-ep.sh:5` set 9216.
2. **Both shipping shapes are `P=1`** — single-stage, no gathers. `P>1` is reachable in production
   only at `b=16`, the smallest bucket, i.e. at concurrency below 16.
3. **The slice is 40 KiB, not 130 KiB, and decode is 2 waves.** Every carried-over number about this
   kernel — the 2138 GB/s bandwidth, the 1.9924 µs/wave DMA slope, the "5.78% of stage 1 is DMA" —
   was measured at `slice_len=33280`. **None of them transfers**, and re-deriving them at 10240 is
   what probes 01–04 (§6) are for.

### What is *not* an axis, with evidence

| candidate axis | verdict | evidence |
| --- | --- | --- |
| **sharding / multi-chip** | **`scores` is fully local to one chip. A single-chip microbenchmark is the correct instrument.** | The indexer's projections are un-sharded upstream: `# no tensor parallel, just replicated` / `ReplicatedLinear` at `vllm/vllm/model_executor/models/deepseek_v2.py:666-673`, and `MergedColumnParallelLinear(..., disable_tp=True)` at `:675`. Chain A registers the op with **no mesh and no partition specs** (`mla_attention_op.py:126`). Chain B pins the batch axis to `None` (`deepseek_v4_indexer.py:44,169`) and its only sharded axis is `attn_head_axis="model"`, which does not survive into `scores` and has size 1 anyway at `TENSOR_PARALLELISM=1`. Both deployments are TP=1, parallelism expressed as DP, `max-num-seqs` per DP engine (`glm-5.2-fp8-dp16-ep.sh:50-53`). |
| **`n` vs sequence length** | not an axis | `n` is capacity; `streamindex_topk.py:603-605`. Live length is `row_lengths`. |
| **prefill vs decode as a shape** | **only `b` differs**; `n` and `k` are identical | `streamindex_topk.py:831` calls the kernel **once** over the whole padded token matrix, after three score-producing passes merge into one buffer (`:757`, `:775`, `:793`). |
| **`row_lengths` distribution** | differs by phase but is not a shape | prefill ramps causally, `min(kv_len, token_pos // cr + 1)`; decode ~flat. `streamindex_topk.py:500-506`. |
| `write_empty_rows` | constant `True` | hardcoded `streamindex_topk.py:835` |
| `_reference_stage2_gather` | constant `False`; **test seam only** | default `sparsecore_topk.py:522`; no production caller (`:538`) |

### One predicate that changes the e2e denominator

**Chain A wraps the whole kernel in `lax.cond` and skips it when `jnp.max(seq_lens) //
compression_ratio <= k`** — `mla_attention_op.py:121` sets `enable_early_exit=True`, gate at
`streamindex_topk.py:860-863`. With `k=2048` and `compression_ratio=1`, **every step whose longest
sequence is ≤ 2048 tokens does not run this kernel at all.** Chain B leaves it `False`.

This does not change any shape, and it does not change a microbenchmark. It changes what fraction
of steps the block is on the critical path for, so it belongs in §7's denominator and not in the
floor. It is also why `--max-model-len 9216` matters twice over: it sets `n`, and it makes the
early-exit gate reachable for a large share of traffic.

### Validated against

- `tests/kernels/test_dsv4_sparsecore_topk.py:61-72` reproduces the `n` arithmetic independently and
  `:202-203` asserts `_topk_n(9216) == 10240` and `_topk_n(132096) == 133120`. My derivation agrees
  with the in-tree test.
- `test_partition_factor_is_what_these_tests_assume` pins `P` per shape, and its ids
  `p1-b64-n10240` / `p2-b16-n10240` (`:186-187`) match rows 2 and 1 of my table.
- The test file's own header (`:21-22`) states: "Every GLM-5.2 configuration exercised in practice
  to date has run at `--max-model-len=9216` (`n=10240`)" — independent agreement that 10240 is the
  shipping shape.

### Unknowns — recorded, not filled in

| # | unknown | settled by |
| --- | --- | --- |
| 1 | `index_topk` for DeepSeek-V4-Flash. Field read at `vllm/vllm/models/deepseek_v4/attention.py:738`; no checkpoint config in this checkout or the local HF cache | reading `config.json` from the checkpoint |
| 2 | `compress_ratios[layer_id]` for DeepSeek-V4-Flash → Chain B's `n` is only `10 * (1024 // cr)` | same `config.json`, or one log line of `cache_kv.shape` |
| 3 | GLM-5.2's `index_topk = 2048` is sourced from an **out-of-repo knowledge record**, corroborated only circumstantially by the test file's choice of `k` | the checkpoint `config.json` |
| 4 | **How many of GLM's 78 layers call this per forward.** `index_topk_freq = 4` appears in the knowledge record but has no handling in `vllm/vllm/model_executor/models/glm4_moe_lite.py` — the skip logic lives only in `deepseek_v2.py:1096-1109`, a class GLM does not use. On the face of the code every MLA layer builds an indexer | counting registered `pallas::streamidx_topk_*` op names, or a profile |

**Unknown 4 is the one that matters for §7**, because it is a direct multiplier on the e2e
denominator, and prior work records 21 of 78 GLM layers as the DSA count — which is not 78 and not
78/4. It is flagged rather than resolved because no candidate's *ranking* depends on it.

---

## 5. Roofline

All measurements below are from **one job, `j-fa60ff2a-c793-4f32-9576`**, 4 shards on 4 chips of one
tpu7x pod, commit `b11b6eb6` (md5 of `sparsecore_topk.py` = `44353b34…`, verified in-job). Raw logs
and JSON under `probes/01-…` … `probes/04-…`. Every arm reported `suspect=False`.

### 5a. The block has no FLOP term and no MXU term

Stated because a roofline that gives it one manufactures headroom that does not exist. §2's op
inventory contains **no floating-point arithmetic and no matmul**. Every op is integer, a VMEM
traversal, or a DMA. At the shipping shapes (`P=1`) there is also **no TensorCore term**: no stage
2, no gathers. The floor is a pure SparseCore floor.

### 5b. The HBM term, and the correction to the stored bandwidth

`probes/04-bw-9216`, DMA-only kernel at the shipping `slice_len=10240`:

```
time(waves) = 21.349 us + 1.0447 us/wave     max resid 0.280 us (0.76%), three-null spread 1.22%
```

This fit predicts the three independently-measured `dma_only` rows in `probes/01`/`02` to within
1.6%, 2.9% and −1.0%. It is the same DMA.

| | 40 KiB slice (shipping) | 130 KiB slice (all prior work) |
| --- | --- | --- |
| bytes/wave | 1.25 MiB | 4.0625 MiB |
| µs/wave | **1.0447** `measured` | **2.001** `measured` (`probes/02`, from `(278.026 − 21.9)/128`) |
| apparent bandwidth | **1255 GB/s** | **2138 GB/s** |

**The stored 2138 GB/s is not wrong, and it is not the shipping number.** Both rows are the same
DMA at two transfer sizes. Fitting per-wave time against bytes across them:

> **fixed cost ≈ 0.62 µs per wave's DMA, streaming at ≈3113 GB/s asymptotically** `modelled, 2-point fit`

so the smaller shipping slice pays the fixed cost over 3.25× fewer bytes and reads 1.7× slower. The
stored 1.9924 µs/wave reproduces at **2.001 µs/wave**, 0.4% out — good, and it is why the 130 KiB
row can be trusted as the contrast rather than re-measured.

**HBM floor at the shipping prefill shape** (`b=1024, n=10240`): 41.94 MB read + 8.39 MB written =
50.33 MB.

| at | t_hbm | vs measured 451.2 µs |
| --- | --- | --- |
| measured 40 KiB-slice rate, 1255 GB/s | 40.1 µs | **8.9%** |
| asymptotic 3113 GB/s | 16.2 µs | 3.6% |

### 5c. ⚠ The block is at ~9% of its HBM roofline — and that number alone is misleading

**Said loudly, as the gate requires: this is far below 50%.** Quoted bare it would read as "eleven
times of headroom", and that reading is wrong. The HBM roofline does not price what this kernel
actually spends its time on, and the unpriced register below is not a footnote — it is 87% of the
kernel.

**The unpriced register**: ~5 VMEM-resident passes over the slice per row, none of which touch HBM
(§2). `references/` has no rate for a VMEM-resident SparseCore pass — only HBM-bound streaming
(337 GB/s). `probes/03-passes-9216` measures it, at `b=1024 n=10240 w=32`, as µs per pass over the
whole call (`max resid ≤0.62%`, four-null spread 0.46%):

| pass | µs/pass | relative to `cvt` |
| --- | --- | --- |
| `cvt` (read+write, 3 int ops/elem) | 20.589 | 1.00× |
| `emit1` (read + cumsum + popcount + `store_compressed`) | 27.456 | 1.33× |
| `scatter_sparse` (1/256 density) | 35.243 | 1.71× |
| `scatter_mid` (1/16 density) | 36.492 | 1.77× |
| **`scatter` (dense `addupdate_scatter`)** | **116.350** | **5.65×** |

### 5d. The cost model, fitted with one parameter and validated out of sample

The harness ships two candidate models and **both miss**: its `uniform` (4 dense scatters)
overshoots by +25.9%, its `graded` (3 sparse) undershoots by −28.0%. Neither is the kernel.

Using the numpy-derived refinement counts from §2 (2.03 at `b=1024`, 2.50 at `b=64`) and solving
for the one unknown — the true cost of a `ref_body` pass:

> **a refinement pass costs 114.3 µs = 0.98× a *dense* scatter, and 3.24× the `scatter_sparse` probe.**

**The refinement pass cost is the walk, not the updates.** `ref_body` (227–243) calls `load_keys`
on every one of the `L` words to compute `match`, then scatters under that mask. Match density
falls off a cliff after L0; the *traversal* does not. This is why the harness's sparse-scatter probe
is the wrong price sheet for it, and it is the single most useful thing this job bought.

**Out-of-sample check** — per-wave costs fitted at `b=1024` (32 waves), used to predict `b=64`
(2 waves), a 16× extrapolation with no refitting:

| shape | waves | ref levels (numpy) | predicted | measured | error |
| --- | --- | --- | --- | --- | --- |
| `b=1024 n=10240` (fitted) | 32 | 2.03 | 451.23 µs | 451.226 µs | +0.00% |
| **`b=64 n=10240` (predicted)** | 2 | 2.50 | **51.57 µs** | **51.645 µs** | **−0.14%** |

A one-parameter model landing 0.14% out at a shape it was not fitted to — a 16× extrapolation in
wave count, on a refinement count derived with no hardware at all — is the strongest evidence in
this report that the account below is the right one.

⚠ **One caution on how far that validates.** The extrapolation varies `waves` and the refinement
count together, and every per-wave term scales by the same `2/32`. So it is a strong test of the
*decomposition* — of the claim that a fixed intercept plus a per-wave sum reproduces the kernel —
and a much weaker test of the *split between the terms*. A model that traded some `hist0` for some
refinement would pass this check identically. What pins the split is probe 03's independently
measured `cvt`/`hist0`/`emit` slopes, which were not fitted to anything here; the single fitted
parameter is the refinement pass alone.

### 5e. Where the time actually goes

**Shipping prefill, `b=1024 n=10240 k=2048 P=1`, measured 451.2 µs:**

| | µs | share |
| --- | --- | --- |
| mesh entry / program load | 21.3 | 4.7% |
| **score DMA — what the caller asked about** | **33.4** | **7.4%** |
| `cvt` monotone_key | 20.6 | 4.6% |
| `hist0` dense `addupdate_scatter` | 116.3 | 25.8% |
| refinement scatters (×2.03) | 232.1 | 51.4% |
| `emit` | 27.5 | 6.1% |
| | **451.1** | model vs 451.226 measured |

> **`addupdate_scatter` is 77.2% of the kernel. The DMA is 7.4%.**

**Shipping decode, `b=64 n=10240 k=2048 P=1`, measured 51.6 µs:**

| | µs | share |
| --- | --- | --- |
| **mesh entry / program load** | **21.3** | **41.3%** |
| score DMA | 2.1 | 4.0% |
| `cvt` + `hist0` + refinements + `emit` | 28.2 | 54.6% |

At decode the largest single line is **not work at all** — it is the fixed cost of entering the
kernel, paid once per launch regardless of `b`. Two caveats on that number, both real: the harness's
paired `REPS_LO`/`REPS_HI` slope is designed to cancel host dispatch, but the measured `dispatch
floor` on these shards (19.2–27.3 µs/call) is uncomfortably close to the 21.3 µs intercept, so I
cannot fully separate residual host dispatch from on-device program load; and turning this into an
e2e claim needs the per-step launch count (§4 unknown 4), which is not established.

### 5f. Verdict

| | |
| --- | --- |
| **Is there headroom in this block?** | **Yes, and a lot** — 8.9% of the HBM floor, 77% of time in one primitive. |
| **Is it in the DMA the caller pointed at?** | **No.** Ceiling below. |
| **Does the process continue?** | **Yes.** The ≥80% "do not write a kernel" refusal does not fire. |

---

## 6. The caller's mechanism, priced

**Every percentage below is `(new − old)/old` of kernel time; negative is faster.**

The harness prints its own verdict — `CANDIDATE B gate: stage1 is 66.0% DMA -> BUILD B` at `b=16`,
`44.1% -> BUILD B` at `b=64`. **Those numbers are wrong for this question and following them would
have sent me building the wrong thing.** The gate divides `dma_only / stage1`, and `dma_only`
carries the **21.3 µs launch intercept** inside the numerator. That intercept is program load and
mesh entry: the real kernel pays it too, and **pipelining cannot remove it**. At `b=16` the honest
transfer is 1.0 µs, not the 22.1 µs the gate divides by.

Separating the slope from the intercept — and capping double-buffering at `(waves−1)/waves`, since
the first wave's load can never be hidden:

| shape | waves | stage1 | DMA (slope only) | DMA % | **ceiling on the caller's change** |
| --- | --- | --- | --- | --- | --- |
| `b=16 n=10240` P=2 | 1 | 33.4 µs | 1.0 µs | 3.1% | **0.00%** (nothing to overlap) |
| **`b=64 n=10240` P=1 — DECODE** | 2 | 51.6 µs | 2.1 µs | 4.0% | **−2.02%** |
| **`b=1024 n=10240` P=1 — PREFILL** | 32 | 451.2 µs | 33.4 µs | 7.4% | **−7.18%** |
| `b=16 n=133120` P=4 | 2 | 94.4 µs | 4.0 µs | 4.2% | −2.12% |
| `b=1024 n=133120` P=4 | 128 | 4801.6 µs | 256.1 µs | 5.3% | −5.29% |

These are **ceilings on a perfect implementation**, not predictions. Three things make the realised
number smaller: the DMA engine writes VMEM while the VPU reads it, and at 77% scatter-bound the VPU
is saturating VMEM already, so overlap is not free; a second slice buffer costs +40 KiB of a 512 KiB
per-subcore budget (fine at shipping, tighter at 130 KiB); and the `(waves−1)/waves` cap bites hard
at decode's 2 waves.

**So the mechanism is real, it is not zero, and it is small: at best 7.2% of the kernel at prefill
and 2.0% at decode.** Chained through the block's own denominator — prior measurement puts this
kernel at ~1.2% of decode device time — a perfect version is worth roughly **0.02% of decode
end-to-end**, which is far below what any e2e benchmark can resolve.

**The one number that would change this verdict** is the launch intercept at decode (§5e): if the
21.3 µs is genuinely on-device and the kernel is launched per DSA layer per step, the *block* is a
bigger e2e cost than the ~1.2% figure implies — which would scale the DMA prize with it. That is a
question about the block's denominator, not about the DMA, and §4 unknown 4 is what settles it.

---

## 7. The baseline

**This is the number every variant in step 3 is measured against.** It is the incumbent at
`b11b6eb6`, measured by the instruments in §8, in job **`j-fa60ff2a-c793-4f32-9576`**.

| shape | `P` | waves | **baseline `us_median`** | null spread | dispatch floor |
| --- | --- | --- | --- | --- | --- |
| `b=16 n=10240 k=2048` | 2 | 1 | **56.511 µs** (full) / 33.390 (stage1) | 1.23% | 26.65 µs |
| **`b=64 n=10240 k=2048` — DECODE** | 1 | 2 | **51.658 µs** | 3.37% | 19.20 µs |
| **`b=1024 n=10240 k=2048` — PREFILL** | 1 | 32 | **450.260 µs** | 0.72% | 27.34 µs |
| `b=16 n=133120 k=2048` | 4 | 2 | 125.056 µs | 1.86% | 32.81 µs |
| `b=1024 n=133120 k=2048` | 4 | 128 | 5273.345 µs | 0.23% | 19.94 µs |

Configuration: tpu7x, one chip (2 cores), `SparseCoreInfo(num_cores=2, num_subcores=16,
num_lanes=16, vmem_capacity_bytes=524288, dma_granule_size_bytes=32)`; image tag `fa63d6df` with a
pristine `b11b6eb6` `src/` mapped over it; `--rounds 9`, paired `REPS_LO=20`/`REPS_HI=60` slope,
median of 9 rounds, arms interleaved and reshuffled per round.

**The null spread is the floor a variant delta has to clear**, and it is not the same at every
shape: **3.37% at decode** against 0.72% at prefill. A −2.02% decode change — which is the ceiling
on the caller's mechanism (§6) — is **inside the decode null floor**. It cannot be resolved at this
shape by this instrument, however well implemented. That is a fact about the measurement, not a
prediction about the change, and step 3 must either measure decode differently (more rounds, or
in-batch) or report decode as `not resolvable` rather than as `no effect`.

### Two harness readings that are *not* baselines, and why

- **`parts/full = 2.001` at `b=1024 n=10240`** with the harness's warning that "the pieces do NOT
  compose". At `P=1` this is an artifact: `sum(parts)` includes `gatherA`, `stage2` and `gatherB`,
  **none of which execute on the single-stage path**. The meaningful reading is
  `full = 450.260` vs `stage1 = 451.226` — equal within the 0.72% null spread, which independently
  confirms §3's claim that at `P=1` the block *is* the one kernel.
- **`CANDIDATE A/A2 prize = gatherA/gatherB`**, printed at every shape. At the two shipping shapes
  these arms are measuring ops that do not run. Ignore them there; they are real only at `P>1`.

## 8. Instruments

### Correctness — `tests/kernels/test_dsv4_sparsecore_topk.py`

Already existed and is strong; I extended it rather than writing a new one.

```
python -m pytest tests/kernels/test_dsv4_sparsecore_topk.py -v --timeout=3600
```

**Status: 68/68 PASSED on the incumbent**, job **`j-62275e52-1498-4e01-918e`**, fanned 4 ways across
one pod's 4 chips (22.1 s worst shard). Kernel md5 verified in-job as `44353b34…` = `b11b6eb6`.

What makes it a real gate rather than a green light:

- It compares **selected scores as a multiset with zero tolerance** (`max |score diff| == 0.0`),
  never indices — so a different but equally-valid tie-break passes and a wrong selection cannot.
- `test_comparison_detects_a_wrong_index` is a **negative control on the comparison itself**: four
  corruptions (`worst_index`, `drop_one`, `duplicate`, `negative`) must each go red. **All four
  pass while the real output is accepted**, so the suite is discriminating, not vacuous.
- `test_partition_factor_is_what_these_tests_assume` pins `P` per shape, so tuning `_pick_partition`
  cannot silently retarget the suite away from the shapes it claims to cover.

**My one change**: added `pytest.param(1024, N_SHORT, 2048, 1, id="p1-b1024-n10240")` to `SHAPES`.
The shipping prefill shape was **not covered** — the file's top-of-ladder row was `(1024, N_128K)`,
which needs `--max-model-len 132096`. It is also the single-stage shape with the most waves (32 vs
2), and **anything that reuses a buffer across waves — a double-buffered or pipelined score load —
is correct at one wave by construction and can only go wrong at many.** Collection count went 58 →
68, which is the evidence the added row is actually being exercised.

**These instruments are READ-ONLY from here.** A variant that edits the test, the reference or the
tolerance has not passed; it has moved the gate.

### Performance — `microbench/sctopk_2stage_bench.py`

Also already existed. Reused unmodified. Modes used: `--decomp` (per-stage prices and the
`dma_only` arm at stage 1's exact wave count), `--passes` (per-VMEM-pass slope), `--bw` (DMA slope
vs wave count).

Validated before trusting, per "inheriting a stale harness is how a rewrite gets measured at a shape
nobody runs" — and it **was** stale in exactly that way: its defaults are `--ctx 132096` and
`--batches 16 1024`, i.e. `n=133120`, which §4 shows no checked-in config produces. Every number in
this report is at `--ctx 9216` for that reason. Other constants checked: `CSA_TOPK=2048` agrees with
§4; `topk_n()` reproduces `streamindex_topk.py:603-605` and agrees with the in-tree test.

**What it cannot measure:**

- **Anything multi-chip.** It runs on one chip. That is *correct* here — §4 establishes `scores` is
  replicated and fully local — but it means the instrument cannot detect a change that alters
  collective or cross-chip behaviour. Nothing proposed does.
- **End-to-end effect.** It measures the kernel, not the model. Converting a kernel delta into an
  e2e delta needs the block's denominator, which is §4 unknown 4 and is not established.
- **Correctness of any timing arm.** `--bw`, `--decomp` and `--passes` verify nothing —
  `dma_only`'s output is a 16-word accumulator nobody checks. **If step 3 adds a chunked or
  double-buffered arm to `_dma_only_kernel`, nothing in that file will catch a copy that lands the
  wrong bytes**, and a checksum against the blocking arm has to be added by hand.
- **Decode-shape deltas below ~3.4%** — the null spread at `b=64` (§7).

### Probe record

| # | probe | question | job | answer | decided |
| --- | --- | --- | --- | --- | --- |
| 01 | `decomp-9216` | what is the DMA fraction of the kernel at the **shipping** shapes, measured as a ratio inside one job? | `j-fa60ff2a-c793-4f32-9576` | 3.1% / 4.0% / 7.4% at `b=16/64/1024` once the launch intercept is removed — **not** the 66%/44%/12% the harness gate prints | the caller's mechanism is capped at −7.18% (prefill) / −2.02% (decode) |
| 02 | `decomp-128k` | does the stored "5.78% of stage 1 is DMA" reproduce? | same | yes — 5.3% hideable, and the stored 1.9924 µs/wave reproduces at 2.001 | the stored numbers are sound *at the shape they were taken*; they are simply not the shipping shape |
| 03 | `passes-9216` | what does a VMEM-resident SparseCore pass cost? (no `references/` entry exists) | same | `cvt` 20.589, `emit1` 27.456, sparse scatter 35.243, **dense scatter 116.350** µs/pass | the price sheet for every step-2 candidate; and `addupdate_scatter` is 77% of the kernel |
| 04 | `bw-9216` | what is the DMA slope at the shipping 40 KiB slice? | same | `21.349 µs + 1.0447 µs/wave`; 1255 GB/s apparent, ≈3113 GB/s asymptotic with ≈0.62 µs/wave fixed | separates hideable transfer from unhideable launch cost — the correction that overturns probe 01's printed gate |
| 05 | `refinement-count` | how many of the 3 refinement levels run? (2× swing in the compute term) | **none — computed exactly in numpy** | 2.03 levels at `b=1024`, 2.50 at `b=64`; 1.06× / 1.28× subcore imbalance | no probe needed; and it supplied the out-of-sample validation in §5d |

Probe 05 is recorded here because it answered a question the wave was going to spend a job on. The
two-question rule applies in reverse: it is not in `references/`, and being wrong *would* have
changed a decision — but the answer was derivable exactly from the kernel's own arithmetic, because
`done` is a deterministic function of the data.

---

## 9. Carried to step 2

Three candidates, ranked by measured prize at the shipping prefill shape. **The caller's mechanism
is the smallest of the three**, which is stated plainly rather than buried; it is still built,
because it was the explicit ask, it is non-zero, and the marginal cost of one more variant is
minutes plus one job.

| # | candidate | mechanism | measured prize (ceiling) | route |
| --- | --- | --- | --- | --- |
| 1 | **refinement walk** | `ref_body` (227-243) walks all `L` words to test `match` though almost none match after L0. Measured at 0.98× a *dense* scatter — the cost is the traversal. Compacting L0's straddling bucket once would make the ~2.03 refinement passes proportional to `cnt`, not `L` | **51.4% of the kernel** | step 2, this skill |
| 2 | **decode launch cost** | 21.3 µs of fixed mesh-entry/program-load, = **41.3% of the decode kernel**. Not proportional to work | up to 41.3% at decode, **but** confounded with residual host dispatch (§5e) and needs §4 unknown 4 to become an e2e claim | step 2, with a probe that separates on-device program load from host dispatch **first** |
| 3 | **pipelined score DMA — the caller's request** | replace the blocking `start()`/`wait()` at 108-111 with `emit_pipeline` or a hand-rolled double buffer | **−7.18% prefill / −2.02% decode** | step 2, this skill |
| — | subcore load imbalance | 1.28× at decode from data-dependent early exit with only 2 rows/subcore (§2) | bounded by candidate 1's term | step 2, likely subsumed by 1 |

**On the caller's stated mechanism specifically**: `pltpu.emit_pipeline` **does** lower and run
inside a SparseCore kernel — `references/08-sparsecore.md:609`,
`[verified: j-b52a1bd9-cd4b-4d1b-851d, reproduced twice]`, importing from `jax.experimental.pallas.tpu`
and **not** from `plsc`. So the requested approach is available. Whether the managed pipeline beats
a hand-rolled double buffer on SparseCore is explicitly **not** answered by any reference file
(`08-sparsecore.md:609`), and `01-fusion-boundary.md:1119` records the one adjacent measurement
going the other way: a hand-rolled early issue on a SparseCore `pl.kernel` took **−2.0%**. Both go
in step 2's design graph as branches, not as a settled choice.

**A structural obstacle candidate 3 must clear, recorded now so step 2 does not rediscover it**:
`resident_vmem` is read **five times per wave** (`cvt`, `hist0`, ×2.03 `ref_body`, `emit`). A
streaming pipeline that retires a tile after one consumption does not fit this kernel without
restructuring. The tractable form is a **cross-wave** double buffer — issue wave `w+1`'s load before
wave `w`'s compute — which is what the `(waves−1)/waves` cap in §6 already assumes.
