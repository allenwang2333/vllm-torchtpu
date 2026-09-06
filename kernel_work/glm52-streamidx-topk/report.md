# Step 1 — `streamindex_topk` on GLM-5.2: instrument and roofline

Block under study: `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py` and its
selection back end `sparsecore_topk.py`, as called by
`src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py:72-141`
(`VllmTPUSparseAttnIndexer`, registered out-of-tree via `@SparseAttnIndexer.register_oot`).

Branch `annyan/glm52-streamidx-topk` off `13617694`. Image
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`
(digest `sha256:7e478fc8394e5a041f5dcf5b58a1bd77b8ff262c504f2dfbd3916d33a711b5f6`),
verified on `j-1054e9b4-1e45-4de9-9009`.

**Sign convention: every percentage in this document is a change in time. Negative is faster.**

---

## 1. What the block computes

Per query token `t` and compressed KV position `s`:

```
score[t,s] = kv_scale[s] * sum_h  weights[t,h] * max(0, q[t,h,:] . k_cache[s,:])
```

fp8 x fp8 einsum accumulated in f32 (`:326-331`), ReLU (`:333`), weighted head sum (`:334-335`),
e8m0 dequant scale (`:337`), then a validity mask `k_span < kv_len` and a causal mask
`k_span <= q_pos // cr` (`:340-352`). Output is `int32[T, k]` of compressed-space column indices,
**unsorted**, `-1` suffix-padded. The selection is **exact** — a 4-level 8-bit radix select on
SparseCore.

It is confirmed live for GLM-5.2: the architecture is `GlmMoeDsaForCausalLM` with a DSA Lightning
Indexer, and 21 of 78 layers run it (`config.json:indexer_types`, rule `(max(L-2,0) % 4) == 0`).
That 21 is the denominator for scaling any microbenchmark win to a per-step expectation.

### Architecture: a two-device pipeline with no overlap

Three separate `pl.pallas_call` launches on the TensorCore materialize a full `f32[T, n]` score
matrix in HBM; then `sparsecore_topk` (`:831`) reads it back. They are separate HLO ops chained by
a data dependency.

| Launch | Range | `static_q_len` | `seq_batch_size` | `bq_sz` | `bkv_p` |
|---|---|---|---|---|---|
| DECODE batched (`:757`) | `[0, (D//4)*4)` | 1 | 4 | 1 | 2 |
| DECODE remainder (`:775`) | `[(D//4)*4, D)` | 1 | 1 | 1 | 2 |
| MIXED (`:793`) | `[D, R)` | None | 1 | 128 | 2 |

The PREFILL slot — index `[1]` of `num_kv_pages_per_block` and `num_queries_per_block` — is
**dead**: `_common_path` reads only `[0]` and `[2]`, and `mla_attention_op.py:183` builds
`distribution` as `[D, D, R]`, so the prefill-only range `[distribution[0], distribution[1])` is
empty by construction. No `StreamIdxTC-p-*` symbol can ever appear. Index `[1]` is nonetheless
load-bearing for `num_sublanes_total` (`:603-605`), so it sizes the biggest HBM buffer without
affecting any compute — a one-line lever, noted for step 2.

---

## 2. Shapes — the two deployments, derived not assumed

Sharding is **not** an axis: the indexer is `ReplicatedLinear` / `disable_tp=True`
(`vllm/model_executor/models/deepseek_v2.py:668-689`) and the GLM pallas op is built with no mesh
and no `input_partition_specs` (`mla_attention_op.py:126-128`), in contrast to its DSv4 sibling
which passes a full `P(...)` tuple. **A single-die instrument is the correct instrument for both
deployments.**

> **Correction, applied after step 2 wave 2.** This section originally gave `n` at (A) as 9216,
> `pages_per_seq * page_size`. That is wrong: `streamindex_topk.py:602-605` sizes the score buffer
> from `align_to(pages_per_seq, bkv_p) * page_size`, and with `bkv_p = 2` the 9 pages round up to
> 10, so **`n` = 10240** — 11.1% wider than `max_model_len`, and a top slice of 1024 columns that
> no request can ever reach. The figure propagated into `roofline.py` and `shapes.py`; all three
> are now corrected. It does not change step 1's verdict — every row still reads
> `STRUCTURAL - proceed`, with `floor/meas` between 1.7% and 8.0% — but it does size two of the
> candidates, since `n` is what the `-inf` fill and the SparseCore read are dimensioned by, and
> `bkv_p ∈ {1, 3, 9}` divides 9 exactly and removes the pad for free.

|  | (A) 8k1k | (B) AgentX |
|---|---|---|
| `max_model_len` | 9216 | 1048576 |
| `--block-size` in recipe | *absent* | 256 |
| **effective `block_size`** | **1024** (`get_page_size` -> 1024) | **512** (raised from 256 by `TpuPlatform.update_block_size_for_backend`) |
| `pages_per_seq` | 9 | 2048 |
| **`n` = score-row width** | **10240** | **1048576** |
| ↳ `n` vs `max_model_len` | **+11.1% pad** — `align_to(9, bkv_p=2) = 10` | none — `align_to(2048, 2) = 2048` |
| `cache_kv` | `(nb, 256, 4, 256)` u8 | `(nb, 128, 4, 256)` u8 |
| `max_num_seqs` / `seq_lens.shape` | 64 | 64 |
| token ladder `T` | `{16,32,64,128,256,512,1024}` | same |
| ranks per chip | 2 (16 DP on 8 chips) | 1 (16 DP on 16 chips) |

Two corrections to what the recipes appear to say. (A)'s ISL is **not** 8192: `--random-range-ratio`
defaults to 0.8, so it is uniform on `[1638, 14746]`. And (B)'s block size is 512 at runtime, not
the 256 in the yaml.

**The early exit never fires.** The gate is `jnp.max(seq_lens) // cr <= k`, i.e. a **batch**
maximum against 2048 with `cr=1`. (A) runs 64 concurrent sequences with `P(ISL <= 2048) = 0.031`,
so `P(fast path) ~ 0.031^64 ~ 1e-96`; (B) is trivially never below it. `STREAMIDX_ENABLE_EARLY_EXIT`
costs nothing (`lax.cond` compiles both branches) and saves nothing. No measurement taken through
`_fast_path` is a measurement of this op — `shapes.check_reachable` asserts against it.

### The score matrix is sized by `max_model_len`, not by the request

`n = pages_per_seq * page_size` comes from the block-table width, so **a 32K request in (B) pays
the same `f32[T, 1048576]` score buffer as a 1M request**:

| `T` | (A) score buffer | (B) score buffer |
|---|---|---|
| 16 | 0.6 MiB | 64 MiB |
| 64 | 2.3 MiB | 256 MiB |
| 1024 | 36 MiB | **4096 MiB** |

It is written in full with `-inf` by `jnp.full` (`:745-749`), partially overwritten by the Pallas
launches, and read back by SparseCore — whose stage-1 DMA is bounded by `eff_len`, so at
(B)/`T=1024`/256K context the kernel writes 4 GiB of `-inf` in order to read 1.25 GiB of it.

---

## 3. Roofline

Constants read back from `pltpu.get_tpu_info()` on `j-1054e9b4-1e45-4de9-9009`, **per TensorCore**
(`num_cores=1`, `is_split_chip=True` — doubling reproduces the published per-chip Ironwood
figures, which is the check that these are per-die):

`mem_bw = 3.70e12 B/s` · `fp8 = 2.30e15 F/s` · `bf16 = 1.155e15 F/s` · `vmem = 64 MiB` ·
`smem = 1 MiB` · `cmem = 0` · SparseCore: 2 cores, 16 subcores, 512 KiB VMEM, 32 B DMA granule.
`int8_ops_per_second = 0` is an unpopulated field, not a measurement — do not model int8 off it.

Model in `roofline.py`. Time is `max(bytes/BW, flops/(fp8*eff))` — i.e. **perfect DMA/compute
overlap**, which is the assumption most generous to the incumbent and therefore the conservative
one for headroom. The floor reads every needed key byte once (129, not 256), keeps the score
matrix in VMEM, and runs the MXU at full efficiency.

| row | KV MB | -inf MB | SC MB | t_mem us | t_mxu us | bound | t_act us | t_floor us | floor/act |
|---|---|---|---|---|---|---|---|---|---|
| A-D64 dec 64x8k | 128.0 | 2.2 | 2.0 | 38.3 | 7.5 | mem | 38.3 | 18.5 | **48%** |
| A-D64 dec 64x14k | 224.0 | 2.2 | 2.2 | 66.0 | 13.1 | mem | 66.0 | 32.2 | **49%** |
| A-P1024 pre 1x8k | 16.0 | 36.0 | 32.0 | 36.3 | 29.9 | mem | 36.3 | 29.9 | 82% |
| A-P512 pre 1x8k | 8.0 | 18.0 | 16.0 | 18.1 | 14.9 | mem | 18.1 | 14.9 | 82% |
| B-D4 dec 4x256k | 256.0 | 64.0 | 5.0 | 93.3 | 14.9 | mem | 93.3 | 36.6 | **39%** |
| B-D4 dec 4x1M | 1024.0 | 64.0 | 17.0 | 317.7 | 59.8 | mem | 317.7 | 146.3 | **46%** |
| B-D16 dec 16x256k | 1024.0 | 64.0 | 20.0 | 318.6 | 59.8 | mem | 318.6 | 146.3 | **46%** |
| B-P1024 pre 1x1M | 2048.0 | 4096.0 | 4352.0 | 4138.8 | 3824.4 | mem | 4138.8 | 3824.4 | 92% |
| B-P1024 pre 1x256k | 512.0 | 4096.0 | 1280.0 | 1962.3 | 956.1 | mem | 1962.3 | 956.1 | **49%** |

### The gate — decided on the measurement, not on the model

Read against the model alone, `t_floor / t_act` gives 39-49% for decode (proceed) and 82-92% for
long-context prefill (**stop, at floor**). That reading is wrong, and section 6 is what shows it.
The gate is `t_floor / measured`:

| row | measured | modelled | meas/model | floor | **floor/meas** |
|---|---|---|---|---|---|
| A-D64-8k | 363.2 us | 38.3 | 9.5x | 18.5 | **5.1%** |
| A-P1024-8k | 832.9 us | 36.3 | 22.9x | 29.9 | **3.6%** |
| A-P512-8k | 573.1 us | 18.1 | 31.6x | 14.9 | **2.6%** |
| B-D4-256k | 1117.9 us | 93.3 | 12.0x | 36.6 | **3.3%** |
| B-D4-512k | 1394.4 us | 168.1 | 8.3x | 73.2 | **5.2%** |
| B-D4-1M | 1826.7 us | 317.7 | 5.7x | 146.3 | **8.0%** |
| B-D1-1M | 1481.9 us | 93.1 | 15.9x | 36.6 | **2.5%** |
| B-D16-256k | 2048.8 us | 318.6 | 6.4x | 146.3 | **7.1%** |
| B-P1024-256k | 50916.8 us | 1962.3 | 25.9x | 956.1 | **1.9%** |

**Every shape sits between 2% and 8% of its floor. Nothing measured here is
memory-bandwidth-bound, and the 80% gate is nowhere near firing.** Proceed to step 2 at every
shape, including the ones the model wanted to exclude.

Two conclusions above are retracted:

- **"Long-context prefill is at its floor, leave it alone" was an artifact of the model.**
  `A-P1024-8k` measures 832.9 us against a 29.9 us floor. It has the *most* headroom in the sweep,
  not the least. Gating on the model would have excluded from scope the shapes that turned out to
  be worst — which is why step 1 requires a measured baseline and not only a roofline.
- **Levers 1 and 2 are worth far less than section 3 claims.** Both attack HBM traffic, and HBM
  traffic is at most ~10% of the measured time. Lever 1's modelled -47% at `A-D64-8k` is -47% of
  38.3 us inside a 363.2 us measurement — about -5% of the real number, and only if the strided
  read is free (register item 1, unprobed). Both stay on the list, demoted.

The model is not thereby useless: it is what makes a 5.7-32x gap legible as *a missing term* rather
than as a kernel that is merely slow, and locating that term is the whole of section 6. But it is
an input to the gate, not the gate.

### Where the bytes go, and the two levers

| row | KV | -inf fill | pallas wr | SC read |
|---|---|---|---|---|
| A-D64 dec 64x8k | **94.8%** | 1.7% | 1.5% | 1.5% |
| B-D4 dec 4x256k | **77.8%** | 19.4% | 1.2% | 1.5% |
| B-D4 dec 4x1M | **91.3%** | 5.7% | 1.4% | 1.5% |
| B-P1024 pre 1x256k | 7.4% | **59.2%** | 14.8% | 18.5% |

**Lever 1 — the cache row is 256 bytes and 129 are used.** `packed_width = 194` is computed with
the *MLA* nope/rope formula applied to the indexer's `head_size=129` and then aligned to 256; the
row actually holds 128 fp8 value bytes plus 1 e8m0 scale byte (`streamindex_topk.py:265-268`,
independently reproduced by `tests/kernels/test_dsv4_streamindex_topk.py:310-322`). `_fetch_bkv`
DMAs all 256. This cache is written and read by this op alone, so the layout is self-contained.

**Lever 2 — the score matrix need not touch HBM at all.** One score row is `n*4` = 36 KiB at (A)
and 4 MiB at (B); the *live* rows at a realistic decode point are 4-64 of them. That fits the
64 MiB VMEM. The round trip exists only because scoring and selection are separate HLO ops.

Modelled effect, applied to the rows that are not already at their floor:

| row | incumbent | Lever 1 only | Δ time % | Lever 1+2 | Δ time % | floor |
|---|---|---|---|---|---|---|
| A-D64 dec 64x8k | 38.3 us | 20.3 | **-47.0%** | 18.5 | **-51.7%** | 18.5 |
| A-D64 dec 64x14k | 66.0 us | 34.5 | -47.7% | 32.2 | -51.2% | 32.2 |
| B-D4 dec 4x256k | 93.3 us | 57.3 | -38.6% | 36.6 | **-60.8%** | 36.6 |
| B-D4 dec 4x1M | 317.7 us | 173.8 | -45.3% | 146.3 | -54.0% | 146.3 |
| B-P1024 pre 1x256k | 1962.3 us | 1890.3 | -3.7% | 956.1 | **-51.3%** | 956.1 |
| A-P1024 pre 1x8k | 36.3 us | 34.0 | -6.2% | 29.9 | -17.7% | 29.9 |

All figures in this section are **modelled**, not measured.

---

## 4. The unpriced register

Terms the roofline does not price. An `unmodelled` term that decides between two options is a
mandatory probe in step 2.

Section 6 moved several of these. Items 1 and 2 are still unprobed but now decide much less than
they did, because the traffic they govern is at most ~10% of measured time. Item 10 is resolved and
promoted to a design target. Items 11-13 are new and item 11 is the important one: **the largest
component of the op has no cost model at all**, so step 2 owes one before step 3 can know when to
stop.

| # | Term | Status | Decides? |
|---|---|---|---|
| 1 | **DMA efficiency of a 128-of-256-byte strided read.** The roofline prices bytes; Lever 1 assumes reading lanes `[0:128]` of a 256 B row buys ~2x the useful bandwidth. If HBM transaction granularity makes the strided read cost the same as the contiguous one, **Lever 1 is worth zero.** | `unmodelled` | **YES — strike test, step 2** |
| 2 | **Is `jnp.full(-inf)` actually materialized?** Priced as a full `T*n*4` write. It is aliased into a custom call so it should be, but if XLA elides or fuses it, Lever 2 loses its largest term at (B) prefill. | `assumed` | **YES — probe** |
| 3 | **`args[2].copy_(new_cache)` at `mla_attention_op.py:140`, immediately after `donate_argnums=(2,)` at `:128`.** If not elided by storage aliasing this is a full indexer-KV-cache copy per layer per step — ~1 GB per layer at (B), which would dwarf the entire op and change what optimizing the kernel is worth. Needs an xprof of the real model. | `unmodelled` | **YES — but call-site, not kernel** |
| 4 | **HBM capacity -> concurrency.** The model prices time per call, not throughput per rank. Halving the indexer cache frees 2.7 KB/token, and dropping the 4 GiB score buffer frees more; at (B), where concurrency is capped by KV capacity, that may be a larger e2e effect than the latency win. | `unmodelled` | ranks options |
| 5 | **Per-grid-step fixed cost.** Priced as `bytes/BW` with no per-step overhead. (B) at 1M runs 1024 KV blocks per sequence; 1 us of per-step cost would add ~1 ms. | `unmodelled` | no |
| 6 | **Decode MXU efficiency** — whether `seq_batch_size=4` packs 4 sequences into the matmul M dim (making it 128) or leaves it at 32 heads. Assumed 0.25. Decode is memory-bound at either value (7.5 vs 30 us against 38), so it does not change the verdict. | `assumed` | no |
| 7 | **TC/SC serialization.** The three Pallas launches and the SC select are separate HLO ops; the model sums them. If they already partly overlap, the measured total sits below the modelled one. | `assumed` | no |
| 8 | **Data-dependent SparseCore refinement depth** — `done` (`sparsecore_topk.py:212,249`) zeroes remaining levels' loop bounds, so between 3 and 6 passes over the resident row depending on the score distribution. Priced as one. | `modelled` (1 pass) | no |
| 9 | **Relayout costs.** No roofline term exists for them at all. Any layout change in step 2 must be measured, not modelled. | `unmodelled` | no |
| 10 | ~~**Intra-group raggedness.**~~ **RESOLVED by measurement.** `B-RAG16` vs `B-D16-256k` at equal live KV: **+184% time**. Finding 3. | `measured (j-59d0b337)` | promoted to a target |
| 11 | **The cost model of `sparsecore_topk` itself.** Now the largest single term in the op (45-71%) and ~190x above the only model I have for it, which was a bandwidth model and is plainly the wrong kind. Finding 2 says it scales with the padded `[T, n]` and not with live data, which bounds it usefully but is not a model — it does not say what a *fixed* select over a given `[T, n]` should cost, so there is no floor to tell step 3 when to stop iterating on it. | `unmodelled` | **YES — step 2 must build one** |
| 12 | **Whether the microbenchmark overstates the op.** Every number here is the op in isolation with nothing to overlap against. Finding 4 scaled to ~1.07 s per prefill chunk across 21 layers, which is large enough that it must be checked against an xprof of a real AgentX prefill before it is believed. If it does not survive, the caveat applies to the whole table, not just that row. | `unmodelled` | **YES — cross-check** |
| 13 | **`A-D64-8k`'s 9.1% block-to-block spread**, against 0.2-3.2% everywhere else. Unexplained. It is the (A) decode headline number, so a variant claiming a single-digit win at that shape cannot be believed until the spread is understood or the row is re-measured. | `measured, unstable` | gates (A) claims |

---

## 5. Instruments

`instruments/shapes.py` — the reachable-shape table and input builder. `check_reachable` re-asserts
the legality ladder (token bucket, `page_size*bkv_p % 128`, `n % 16`, `k` bounds) **and** that the
row does not take `_fast_path`, so a future edit cannot quietly benchmark a shape production never
sees. Cache packing is behaviourally identical to
`tests/kernels/test_dsv4_streamindex_topk.py:39-49`.

`instruments/bench.py` — times the op in two independent measurement blocks and prints the
block-to-block median spread, so run-to-run noise is visible rather than hidden behind one number.
Asserts `_fast_path` was not taken. `--decompose` additionally times `sparsecore_topk` alone on an
equivalent score matrix, splitting scoring from selection.

`instruments/correctness.py` — dequantizes the packed cache, recomputes scores in f32, takes an
exact top-k, and compares **the multiset of selected scores**, not the index set: the kernel's
top-k is exact but ties at the k-th boundary may be broken differently, and comparing indices
would flag that as a failure when it is not one. Also asserts no duplicates, in-range indices, the
right count, and that padded token rows are `-1`-filled. Exits non-zero on failure so it can gate
a job.

The synthetic score distribution is deliberately not uniform f32: the real score is
`sum_h w_h * relu(q_h . k)`, right-skewed with an atom at the ReLU floor, and a flat distribution
lets the radix select terminate early and flatters the kernel. Locally the k-th value sits at 0.43
against a max of 2.09 with no mass ties — a real selection.

## 6. Measured baseline

Two fan-out jobs, four shards each, one chip per shard, on the image above.

- `j-59d0b337-9c4d-4dc0-aa7e` — `JobStatus=Succeeded`, 4/4 shards `rc=0`.
- `j-284a8842-3db6-44e0-ba49` — `JobStatus=Failed`. The failure is the fan-out script returning
  the number of failed shards; **both failures are my own bugs, not the kernel's**, and both are
  written up below. The two `rc=0` bench shards in it are valid and are quoted.

Medians of the first of two independent 20-iteration blocks, after 5 warmups.
Block-to-block median spread was 0.2-3.2% everywhere except `A-D64-8k` at **9.1%** — that one row's
363.2 us is the least stable number in the table and should be re-measured before anything is
compared against it at single-digit-percent resolution.

| row | full (us) | `sparsecore_topk` alone (us) | **SC share** | job |
|---|---|---|---|---|
| A-D64-8k | 363.2 | 189.5 | **52%** | `j-284a8842` |
| A-P512-8k | 573.1 | — | — | `j-284a8842` |
| A-P1024-8k | 832.9 | — | — | `j-284a8842` |
| B-D4-256k | 1117.9 | 795.9 | **71%** | `j-284a8842` |
| B-D4-512k | 1394.4 | 832.1 | **60%** | `j-59d0b337` |
| B-D1-1M | 1481.9 | 780.6 | **53%** | `j-59d0b337` |
| B-D4-1M | 1826.7 | 823.5 | **45%** | `j-284a8842` |
| B-D16-256k | 2048.8 | 1012.8 | **49%** | `j-59d0b337` |
| B-RAG16 | 5813.5 | — | — | `j-59d0b337` |
| B-P1024-256k | **50916.8** | — | — | `j-59d0b337` |

`_fast_path taken: False` and `filled slots row0: 2048/2048` on every row — the section-2 argument
that the early exit never fires is confirmed on hardware, and no row is measuring the trivial path.

### Finding 1 — selection is the op, not scoring

`sparsecore_topk` is **45-71% of every shape measured**. The roofline priced its HBM read at
1.3-1.8% of traffic, or 5.4 us at `B-D16-256k`; it measures 1012.8 us, **~190x** the bandwidth
model. Selection is not memory-bound. It is throughput-bound on a device with 2 cores x 16
subcores x 16 lanes, running a 4-level 8-bit radix select whose refinement passes are serial.

Any design that only makes *scoring* faster is working on at most half the problem, and at
`B-D4-256k` on 29% of it.

### Finding 2 — selection cost tracks the padded score matrix, not the live data

This is the mechanism, and three measured rows isolate it cleanly:

| row | live rows | live length | live elements | SC time |
|---|---|---|---|---|
| B-D4-256k | 4 | 262144 | 1.05 M | **795.9 us** |
| B-D4-1M | 4 | 1048576 | 4.19 M | **823.5 us** |
| B-D1-1M | 1 | 1048576 | 1.05 M | **780.6 us** |

**Four times the live data costs 3.5% more time. One quarter of the live rows saves 5%.** SC cost
is set by `[T, n]` — the *padded* matrix — and is very nearly blind to `row_lengths`. Since
`n = pages_per_seq * page_size` comes from the block-table width, `n` is `max_model_len` (section 2):

> **On AgentX, a 32K request pays the same 1M-wide select as a 1M request.**

That single fact reframes the exercise. The 2-8%-of-floor gap is not diffuse inefficiency to be
tuned away; it is one structural decision — selection sized by the maximum context rather than by
the actual one — and it is worth up to 32x at the short end of AgentX's range. It also explains
Finding 1's magnitude without appealing to SparseCore being slow.

### Finding 3 — the group-max makes a mixed-length batch 2.8x more expensive

`B-RAG16` (4 sequences at 1M + 12 at 4096) against `B-D16-256k` (16 at 256k) — **the same total
live KV, 16 rows both times**:

| row | total live KV | measured | Δ time |
|---|---|---|---|
| B-D16-256k | 4.19 M | 2048.8 us | — |
| B-RAG16 | 4.24 M | **5813.5 us** | **+184%** |

`num_bkv` is `cdiv(max_i kv_lens[i], bkv_sz)` over the `seq_batch_size=4` group (`:276`), so the
three short members of each group pay the long one's block count. The repo already knows: there is
a TODO to sort sequences by length at `:751-754`. It is now a measured 2.8x on exactly the
mixed-length batch AgentX serves, rather than a suspicion.

### Finding 4 — long-context prefill is the largest number in the sweep by 25x

`B-P1024-256k` measures **50.9 ms** for one 1024-token prefill chunk. It allocates a
`f32[1024, 1048576]` = **4 GiB** score buffer, `-inf`-fills it, and selects over it — to use
1.25 GiB of it. Against the 3.70 TB/s bandwidth, 50.9 ms for ~9 GiB of modelled traffic is an
effective 165 GB/s, 4% of peak.

Scaled by the 21 indexer layers, that is **~1.07 s per 1024-token prefill chunk** attributable to
this op. Treat that scaling as an upper bound and not as an e2e claim: the microbenchmark runs the
op in isolation with nothing to overlap against, and the model may hide some of it. **Cross-checking
this one number against an xprof of a real AgentX prefill is the highest-value thing to do before
step 3**, because if it survives it makes this op the dominant term in AgentX TTFT, and if it does
not, the microbenchmark is overstating an isolated op and every number here needs the same caveat.

### Finding 5 — my correctness gate failed on the incumbent, and has been rewritten

`correctness.py` returned `rc=1` on all 5 cases, 262 assertion failures. Every one sits at rank
2014-2047 of a 2048-element selection, with max abs difference <= 0.005 on scores spanning ~2.1.
That is the kernel and an f32 reference ordering near-ties at the cut line differently — not a
selection error.

**A gate that red-lights the baseline cannot judge a candidate**, so it establishes nothing about
the incumbent and would have blocked every variant in step 3 equally. Rewritten, and the rewrite is
the more useful gate:

- the reference now mirrors the kernel's operation order exactly — dot the *unscaled* fp8 values,
  ReLU, weight, sum over heads, and apply the e8m0 scale last (`compute_scores`, `:322-338`)
- four properties stay exact and untoleranced: selected count, uniqueness, index range, and
  `-1`-filled padding rows
- the selection is judged on **deficit** (how far below the true k-th score the worst selected
  element reaches, as a fraction of the top-to-cut range) and **churn** (what fraction of the true
  top-k is missing). Both discriminate a near-tie from a kernel that skipped a KV block by an order
  of magnitude — **measured**, `j-ae538ec1`: boundary swaps reach deficit `0.00173–0.00249` and
  churn `0.29–0.39%`, while dropping one KV block reaches deficit `0.046–0.160` and churn
  `15.9–27.7%`. The tightest case on both metrics is `dec-B-2x8k`, and it is the one that sets how
  tight the thresholds can be.
- both numbers print on every run whether or not they trip, so a regression is a number moving

**Resolved — the gate passes and is a discriminator.** Final thresholds are `TOL_DEFICIT=0.015`,
`TOL_CHURN=0.03`, both set from measured endpoints rather than from the incumbent's margin alone.
`j-ae538ec1-d606-48f4-8c11` (`Succeeded`, exit 0): `negative control: PASS`, 5/5 undamaged pass,
`drop-block` fails 5/5, `tail-swap` passes 5/5. **Step 3 is unblocked**, and this is the named job
the correctness requirement asks for.

Getting there took three iterations, and two of the three failures are worth recording because
each would have produced a *confident wrong answer* rather than an error:

1. `_corrupt` refilled from the lowest-*numbered* unselected positions — a uniform draw from the
   rejected set, which is neither failure being modelled. It made `tail-swap`, the arm that must
   pass, look like a catastrophe. Both arms now damage the row **by true score**, since a skipped
   block and a boundary tie are both defined in score order.
2. The tolerance was **outside its own separating window**. `drop-block` reaches only `0.04605` at
   `dec-B-2x8k`, so at `TOL_DEFICIT=0.05` a kernel skipping a KV block slid under the threshold and
   only `churn` caught it — the gate passed while running on one of its two metrics. This is
   invisible to a passing gate and is exactly what a negative control is for.
3. The separation tool read its must-pass ceiling off `tail-swap` alone, but tail-swap's deficit
   ceiling (`0.00249`) is *below* the undamaged incumbent's (`0.00486`) — swapping the 8 worst
   selected for the true next-best **raises** the minimum score. That window would have opened
   below the incumbent's own error and red-lighted a correct kernel.

The general lesson for step 3: a correctness gate that has never been shown to fail is an
assertion, not a check, and the interesting failures are the ones where it stays green.

The other failure was `A-D64-14k`, an unreachable row I put in the table: `seqlen=14336` exceeds
what (A) can admit. The sampler's ISL range at (A) is `[1638, 14746]`, but
`max_model_len=9216` caps what the server admits. `check_reachable` caught it, which is that
guard's whole purpose; the row is now `A-D64-9k`. It cost `A-D32-8k` and `A-D16-8k`, which died
with the shard and are being re-measured.

### What step 2 has to design against

Ranked by measured share, not modelled bytes:

1. **Selection sized by `max_model_len` instead of by the live context** (Findings 1, 2, 4) — the
   dominant term at every shape, and the cause of the largest number in the sweep.
2. **The group-max over `seq_batch_size`** (Finding 3) — a measured 2.8x on mixed-length batches.
3. **The scoring pass, still 5.7-16x above its own bandwidth model** — `B-D1-1M` spends 701.3 us
   scoring where the model says 93.1 us, i.e. 685 ns per KV block against 69 ns of DMA.

   > **Retracted as a framing, after step 2 area 3.** Calling the 685 ns "unattributed" overstates
   > the mystery: `bytes/BW` is not a floor for an iteration unless a per-step fixed cost is
   > quoted alongside it, and at `B-D1-1M` the bytes are only 13% of the step. Against the
   > `512 ns + bytes/3.48e12` grid-step model the measured per-iteration cost lands at 1.00-1.09x —
   > i.e. essentially *predicted*, not anomalous, by a model fitted on unrelated data. So the
   > scoring pass is iteration-count-bound rather than harbouring an unexplained term, and the
   > lever is fewer, wider iterations rather than a hunt. The one genuine open question left is
   > narrower and is what the step-2 probe actually asks: whether the fixed cost is per
   > **iteration** (then widening `bkv_p` pays) or per **`make_async_copy`** (then it does not,
   > and coalescing pages is the lever instead). 685 ns over 2 copies is ~340 ns/copy, close
   > enough to the 512 ns term that the two readings are not separable from this row alone.
4. Levers 1 and 2 from section 3, demoted: together they address <= 10% of measured time.
