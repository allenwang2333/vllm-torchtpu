# Step 1 — `streamindex_topk` (the `streamidx_topk_jax` op) for GLM-5.2

Status: **in progress.** The roofline below is complete; the baseline measurement, the achieved
fraction and the profile cross-check are pending and are marked as such. Nothing here is a
measurement unless it carries a job id.

Branch `annyan/streamidx-topk-opt` in `/home/annyan_google_com/workspaces/copy1/vllm-torchtpu`,
based on `0976305d`. The three files under study are byte-identical to `7e4d67b2`, the commit the
caller linked (verified by `git hash-object`).

---

## 1. Vocabulary

Defined here because several of these collide with a term of art elsewhere in the stack.

| Term | Meaning here | What it is *not* |
| --- | --- | --- |
| **block** | the code under study — the two-stage op below | not a Pallas grid block, not an MXU row block, not a vLLM KV **page** |
| **page** | one KV-cache block of `PAGE = 1024` tokens | the repo also calls this `block_size`; this document never does |
| **bkv block** | one inner-loop tile of KV the kernel scores at a time, `bkv_sz = num_kv_pages_per_block * PAGE = 2048` positions | not a page, not a grid block |
| **bq block** | one inner-loop tile of query rows, `bq_sz` = 1 (decode) or 128 (prefill) | — |
| **position** | one compressed KV slot, i.e. one candidate the top-k selects among. `compression_ratio = 1` for GLM-5.2, so one position = one token | — |
| **`num_positions`** | the **padded** width of the score matrix, `align_to(pages_per_seq, bkv_p) * PAGE` = **10240** | not the sequence length; it is padded up and is the same for every row |
| **`lkv_dim`** | the minor dim of the KV cache row, in bytes: indexer head dim + 1 ue8m0 scale byte, padded to a 128 multiple | **value unconfirmed — see §5** |
| **stage A** | the three `pl.pallas_call`s that produce the f32 score matrix, on **TensorCore** | — |
| **stage B** | `sparsecore_topk`, the exact radix-select that reduces scores to indices, on **SparseCore** | not an approximate top-k; it replaced `approx_max_topk` in `6142d5e7` |
| **floor** | `max(t_hbm, t_vpu + t_mxu)` — HBM assumed to overlap compute, the compute units assumed **not** to overlap each other. Both are modelling assumptions, in opposite directions | not the largest of the three columns |
| **achieved fraction** | `floor / measured`. Against a **modelled** floor, so it is never itself a measurement | — |

---

## 2. What the block does

Per indexer-computing layer, per forward step. Score contract, from the reference implementation at
`tests/kernels/test_dsv4_streamindex_topk.py:58-118`:

```
score[t, s] = sum_h  relu( q[t,h] · k[s] ) * weights[t,h]        h = 0..31
```
causally masked at `s <= q_abs_pos`, then the top `k = 2048` positions per row `t` are selected.

```
 q  fp8[T, 32, 128]  ─┐                          weights f32[T, 32]
    T*4 KiB           │                              T*128 B
                      ▼
 cache_kv  uint8[num_pages, 256, 4, lkv_dim] ──▶ ┌──────────────────┐
    streamed: nseq * num_bq * ceil(S/2048)*2048  │   STAGE A        │
              * lkv_dim  bytes                   │   TensorCore     │
                                                 │   3x pallas_call │
                                                 │   fp8 MXU matmul │
                                                 │   relu, *w, sum_h│
                                                 └────────┬─────────┘
                                                          │  f32[T, 10240]
                                                          │  = T * 40 KiB   ──▶ HBM
 scores_init = jnp.full(-inf)  ──▶ HBM, T * 40 KiB        │  (aliased to the init buffer)
   (streamindex_topk.py:745, a full-size write every call)│
                                                          ▼
                                                 ┌──────────────────┐
                                                 │   STAGE B        │
                                                 │   SparseCore     │
                                                 │   radix-select   │
                                                 │   <= 4 passes    │
                                                 └────────┬─────────┘
                                                          ▼
                                                   i32[T, 2048]
```

The score matrix is therefore written once and read back once through HBM, plus a full-width `-inf`
initialisation. Three passes over a `T x 10240` f32 buffer. Whether that matters is §4's question,
and the answer differs between decode and prefill.

---

## 3. Configuration

GLM-5.2 as served by the nightly perf config
(`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh`, and the matching cloud-devkit recipe
`vllm-torchtpu-glm-5.2-v7x-16/vllm-torchtpu-glm-5.2-benchmark.yml`, topology `2x2x2` / `tpu7x`).

| Quantity | Value | Source |
| --- | --- | --- |
| `index_n_heads` (H) | **32** | checkpoint `config.json`; GLM differs from DeepSeek-V3.2's 64 |
| `index_head_dim` (D) | **128** | checkpoint `config.json` |
| `index_topk` (k) | **2048** | checkpoint `config.json` |
| indexer-computing layers | **21 of 78** | checkpoint `config.json`, `indexer_types` — see below |
| `compression_ratio` | **1** | hardcoded at the call site, `mla_attention_op.py:116` |
| TP / DP / EP | **TP=1, DP=16, EP on** | nightly config |
| `max_model_len` | **9216** (8192 in + 1024 out) | nightly config |
| `max_num_batched_tokens` | **1024** per engine | nightly config |
| `max_num_seqs` | **64** per engine | nightly config |
| KV page size | **1024** | `PallasMLAttentionBackend.get_page_size`, unconditional `return 1024` |
| `kv_packing` | **4** (uint8 → 32/8) | `streamindex_topk.py:44-46` |
| `pages_per_seq` | **9** = ceil(9216/1024) | derived |
| `num_positions` | **10240** = align_to(9,2)·1024 | derived; one score row is 40 KiB |
| tuning params | `num_kv_pages_per_block=(2,2,2)`, `num_queries_per_block=(1,128,128)` | `mla_attention_op.py:119-120` |

**TP=1 with DP=16 means the indexer is not sharded**: every engine holds all 32 heads and its own
full KV cache, and there is no collective inside the block. A **single-chip** measurement is
therefore representative of the production per-core shape — this is the rare case where the
single-chip caveat does not bite. What it still cannot measure is contention for HBM bandwidth with
the other engine on the same chip, and any interaction with the MoE layers around it.

**Two configuration facts are unresolved and are recorded as unknown rather than guessed:**

- ~~How many of the 78 layers run the indexer.~~ **Settled: 21 of 78.** Read from the checkpoint's
  own `config.json` (`zai-org/GLM-5.2-FP8`, fetched from HuggingFace), which is authoritative over
  both the roofline spec file and vLLM's code default. `"index_topk_freq": 4`, and the config
  carries an explicit 78-entry `indexer_types` list: `full` at layers 0, 1, 2 and then every 4th
  from 6 to 74 — 3 + 18 = **21 `full`, 57 `shared`**. A `shared` layer reuses the preceding `full`
  layer's indices and does not run the op. vLLM's `index_topk_freq = 1` default, which would have
  given 78 of 78, is not what this checkpoint uses. The op therefore runs **21 times per forward
  step**, and an optimization is worth 21x its per-call saving, not 78x.

  The checkpoint also carries `num_nextn_predict_layers: 1` and
  `index_share_for_mtp_iteration: true`, so MTP would add indexer invocations — but the nightly
  benchmark recipe enables no speculative decoding (no `mtp` / `speculative` / `nextn` key in it),
  so 21 stands for the configuration under study. It would not stand if MTP were turned on.
- **Which vLLM actually serves GLM-5.2.** In the vLLM pinned in this checkout's `uv_venv` there is
  no `glm5.py`, and `SparseAttnIndexer` — the class the op under study registers against — is
  referenced only from `deepseek_v2.py`. So the GLM-5.2 model definition that drives this op is not
  in the pinned vLLM, and the nightly must be running a newer one. This does not change the op, the
  kernel or the shapes, all of which are in `vllm-torchtpu` rather than vLLM, but it does mean the
  *call-site* details above were read from a version that cannot serve the model. Recorded so the
  next session does not treat the pinned vLLM as authoritative for the GLM path.
- **The decode token-padding bucket.** `T` is the padded token count. If a 64-token decode step is
  padded to 64, the score-matrix terms are 4.5% of decode HBM traffic; if it is padded to
  `max_num_batched_tokens = 1024`, they are ~33%. This one *does* change the ranking of the terms
  at decode and is measured by the microbenchmark rather than assumed.

---

## 4. Roofline — the floor, and what binds it

Rates, per TensorCore, taken as reported by `get_tpu_info()` per
`references/hardware-facts.md § The denominator` (a Pallas kernel runs on one TensorCore, so this is
already the right denominator; it is **not** rescaled to the chip-level datasheet figures):

| Rate | Value | Provenance |
| --- | --- | --- |
| HBM bandwidth | 3.70e12 B/s reported; **2.97e12 B/s measured** (80%) | `hardware-facts.md`, `03-pipelining.md` |
| fp8 MXU peak | 2.30e15 ops/s reported; 569.9e12 best hand-written | `hardware-facts.md`, `05-compute-units.md` |
| VPU | **1.72e12 ops/s**, derived as MXU 661.4e12 / 385 | `05-compute-units.md:239` — **assumed, see below** |

Counting rules: a multiply-add is 2 FLOPs; HBM bytes count **reads and writes both**, once per pass
the design actually makes over them — so the K cache is counted `num_bq` times, because the kernel
re-fetches it inside the query loop.

Figures below are **modelled**, at the spec HBM rate and the reported fp8 peak.

### Decode — T=64, 64 seqs x 1 token, S=8704, `num_bq`=1

| | `lkv_dim`=256 | `lkv_dim`=129 |
| --- | --- | --- |
| K cache read | **167.8 MB (95.4%)** | 84.5 MB (91.2%) |
| score init + write + read | 7.9 MB (4.5%) | 7.9 MB (8.5%) |
| total HBM | 175.9 MB | 92.7 MB |
| arithmetic intensity | 30.5 F/B (fp8 ridge 622) | 57.9 F/B |
| `t_hbm` / `t_mxu` / `t_vpu` | 47.5 / 2.3 / 36.6 µs | 25.0 / 2.3 / 36.6 µs |
| **floor** | **47.5 µs — HBM bound** | **39.0 µs — VPU bound** |

### Prefill — T=1024 (one chunk), 1 seq, S=9216, `bq_sz`=128 so `num_bq`=8

| | `lkv_dim`=256 | `lkv_dim`=129 |
| --- | --- | --- |
| K cache read (re-read 8x) | 21.0 MB (13.9%) | 10.6 MB (7.5%) |
| score init + write + read | **125.8 MB (83%)** | 125.8 MB (89%) |
| total HBM | 151.1 MB | 140.7 MB |
| arithmetic intensity | 568 F/B — **essentially at the fp8 ridge of 622** | 610 F/B |
| `t_hbm` / `t_mxu` / `t_vpu` | 40.8 / 37.3 / **586.0** µs | 38.0 / 37.3 / **586.0** µs |
| **floor** | **623.3 µs — VPU bound by 14x** | **623.3 µs — VPU bound** |

### What this says

**The caller's prior — that the win is fusing the top-k so the score matrix never reaches HBM — is
not supported at either phase, for two different reasons.**

- At **decode** the score round trip is 4.5% of traffic. 95% is streaming the K cache, and fusing
  the top-k does not remove a byte of that.
- At **prefill** the score round trip really is 83% of HBM traffic — but HBM is not what binds
  prefill. The VPU is, by 14x, and eliminating all 126 MB of score traffic would move a 40 µs term
  that is already hidden behind a 586 µs one.

The two phases want opposite things, and averaging them would have hidden both:

1. **Decode is HBM-bound on the K-cache stream**, and if `lkv_dim` is 256 with only 129 bytes used,
   **roughly half of the single dominant term is padding**. That is the largest single number in
   this document and it is why §5 is the blocking question.
2. **Prefill is VPU-bound on the elementwise work**, not on memory and not on the MXU. The
   `relu → *weights → sum over h` chain runs on a `[T, 32, S]` intermediate — 32x larger than the
   score matrix it reduces to — and that intermediate never touches HBM, so no amount of layout or
   fusion work addresses it.
3. The K-cache **re-read `num_bq`=8 times in prefill** is real but is only 14% of prefill traffic.
   Worth fixing, not worth designing around. Computing it was what demoted it.

**Two `assumed` values carry the prefill verdict and must be probed before any design leans on it**
(`PROBING.md` question 2 — being wrong changes the decision):

- the **VPU rate**, 1.72e12 ops/s, derived from a *ratio* in `05-compute-units.md` rather than
  measured for this op;
- **3 VPU ops per (token, head, position) element** — a guess at what Mosaic emits for
  relu → multiply → accumulate. If Mosaic fuses these, or if the reduction over `h` lowers to the
  MXU rather than the VPU, the prefill floor drops by up to 3x and prefill stops being VPU-bound.

If both are wrong in the same direction, prefill is balanced rather than VPU-bound and the design
target changes completely. This is the first thing step 2 probes.

---

## 5. The blocking question: `lkv_dim`

Reading the code did **not** settle this, and the reason is worth recording rather than
paraphrasing, because the three indexer paths in this repo genuinely disagree.

The op the caller pointed at is `VllmTPUSparseAttnIndexer` (`mla_attention_op.py:61`), registered
via `@SparseAttnIndexer.register_oot`. That is a *different* path from the two neighbouring ones,
and only one of the three pads explicitly:

| Path | Where its cache spec comes from | `head_size` |
| --- | --- | --- |
| `VllmDeepseekV4Indexer` | `deepseek_v4_indexer.py:111`, `head_size=align_to(128 + 1, 128)` | **256**, explicit |
| `VllmTPUIndexer` (rebind) | upstream `DeepseekV32IndexerCache.get_kv_cache_spec` (`deepseek_v2.py:631`), `head_size=self.head_dim`, after `rebind` sets `head_dim = 129` (`mla_attention_op.py:235-236`) | **129**, unpadded |
| **`VllmTPUSparseAttnIndexer`** — the one under study | upstream `SparseAttnIndexer`; the cache is passed in, and `VllmTPUDeepseekV32IndexerBackend` (`attention.py:1342-1349`) *deliberately inherits* `get_kv_cache_shape` unchanged as `(num_blocks, block_size, head_size)` | **not determined from source** |

Meanwhile the kernel's own comments (`mla_attention_op.py:76`, `:175`) assert that `lkv_dim` is
"D + 1 padded to a 128 multiple" — i.e. 256 — on a path whose spec does no padding.

**Both can be true.** A logical `(…, 129)` uint8 array can still be charged 256 bytes per row if
XLA's tiled layout pads the minor dimension. That makes this a question about the *physical* layout
rather than the logical shape, which is why reading more Python cannot answer it and why the
comment and the spec can disagree without either being wrong.

The stakes are the largest single number in this document:

- **256 wide** → 49.6% of the K-cache bytes are padding, and the K cache is 95% of decode traffic.
  Decode floor **47.5 µs, HBM-bound**, with about half the dominant term recoverable.
- **129 wide** → nothing to reclaim. Decode floor **39.0 µs, VPU-bound**, and decode has to attack
  the elementwise chain instead, the same as prefill.

**Settled from code: 256.** Found while building the instruments, by following the allocation
rather than the spec. The runner takes the `isinstance(kv_cache_spec, MLAAttentionSpec)` branch at
`tpu_runner.py:6318` → `PallasMLAttentionBackend.get_kv_cache_shape` (`attention.py:1221-1264`) →
`mla/v2/kernel.py:101-106`, which aligns the minor dim with `unsigned_align_to(kv_dim, 128)`. That
is the 4-D shape `_streamidx_topk_jax` unpacks at `mla_attention_op.py:84`. The cache is
`(total_num_pages, 256, 4, 256)` uint8: **129 of 256 bytes useful, 127 bytes of padding per
record.**

So the decode line of the roofline stands as written — floor **47.5 µs, HBM-bound**, with about
half the dominant term being padding.

**The 256 is a coincidence at this head dim, not a design.** It is reached through the DS-MLA
packed-width formula, not through `align(129, 128)`. `_is_ds_mla_packed_cache` returns True for
`torch.uint8` (`attention.py:1212-1218`) and the runner leaves `head_size_is_packed_width` at its
default `False` (`tpu_runner.py:6342-6347`), so the indexer's `head_size = 129` is put through a
nope/rope decomposition it does not have: `nope = 129 - 64 = 65`, `packed = 65 + 64*2 + 65//64 =
194`, `align(194, 128) = 256`. The direct reading gives 256 too, so the value is right here — but
the two disagree elsewhere. At `D_idx = 64` the formula gives 256 where the direct reading gives
128, a 2x over-allocation. Anyone changing the indexer head dim needs this.

There is also a second, unused allocation path: `VllmTPUDeepseekV32IndexerBackend`
(`attention.py:1342-1371`) deliberately inherits upstream's **3-D** `get_kv_cache_shape`. The live
GLM path does not use it, and the kernel only accepts the 4-D form (`streamindex_topk.py:592`
unpacks four values). Recorded so it is not mistaken for the live one.

`probes/probe_lkv_dim.py` now confirms rather than decides, and it is still worth running: the
derivation above is subtle enough that an independent device-side reading is cheap insurance.

---

## 6. Baseline, instruments, cross-check

- **Baseline number**: pending.
- **Achieved fraction**: pending — cannot be computed before the baseline.
- **Instruments**: `instruments/bench_streamidx_topk.py` and
  `instruments/test_streamidx_topk_correctness.py`, being built. The unit test's oracle is
  `streamindex_topk_ref` (`tests/kernels/test_dsv4_streamindex_topk.py:58-118`), an exact NumPy
  triple loop, compared as sorted index sets because a re-ordered emit is legitimate.
- **What the instruments cannot measure**: end-to-end effect; contention with the co-resident engine
  on the same chip; the MoE layers the op is interleaved with; and — because stage A cannot be
  invoked in isolation without editing the incumbent — **stage A's time is derived by subtraction,
  not measured directly.**
- **Profile cross-check against xprof**: in progress against **existing** profiles — no new profiled
  run needed. Four GLM-5.2 jobs captured 2026-09-02 carry `.xplane.pb` at

  ```
  gs://cloud-devkit/jobs/<job-id>/outputs/<job-id>-ray-node-0/<pod>/vllm-node/xprof/plugins/profile/<ts>/rank<N>_*.xplane.pb
  ```

  `j-7afc895e-29c2-4d97-b25b` (14:09:45), `j-97102390-e817-4ca6-a9cf` (12:24:35),
  `j-a71a648c-3064-471a-a131` (10:43:18), `j-f37c33fb-4107-40ac-8743` (08:00:37). ~16 ranks each,
  consistent with DP=16.

  **Note for whoever searches next:** a first pass concluded no profile existed, having listed
  `gs://cloud-devkit/jobs/<job-id>/outputs/` and found nothing. The artifacts are five levels deeper,
  under `vllm-node/xprof/plugins/profile/`, which is torch-xla's layout rather than cdk's. Listing
  the output root is not a negative result.

  **Caveat on what these profiles are.** They are `vllm-torchtpu-glm-5.2-ix-agentx` — an InferenceX
  agent workload — **not** the nightly benchmark recipe whose config §3 is built on. The serving
  shape must be read back from `cdk job recipe` before any number from them is compared against a
  figure derived at §3's config. Pending.

## 7. Probes run in this step

```
probe:    Is SparseCore reachable on this stack, and does sparsecore_topk run at GLM-5.2 shapes?
config:   tpu7x single-host 2x2x1 (pool tpu7x-singlehost-2026), jax/jaxlib 0.10.2, image
          vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b.
          sparsecore_topk at (b,n,k) = (32,4096,512), (32,32768,2048),
          (64,65536,2048), (256,65536,2048)
job:      j-0b008512-36f2-4eb2-bdc6, JobStatus Succeeded
answer:   PRESENT, and it works. get_tpu_info().sparse_core returned
          SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16,
          vmem_capacity_bytes=524288, dma_granule_size_bytes=32).
          All four shapes ran with no NotImplementedError and no OOM, and all four
          matched the numpy top-k set oracle exactly, including (256, 65536, 2048).
          Host wall-clock, best of 5: 207.9 / 222.4 / 339.0 / 779.6 us.
decided:  stage B can be measured, so the incumbent can be measured end to end; and a
          challenger is allowed to keep the top-k on SparseCore rather than being
          forced to fold it into the TensorCore kernel.
```

**Two corrections this probe forces, both to documents that would otherwise have misled a design:**

- **`references/hardware-facts.md` is wrong on this point.** It records that no SparseCore fact has
  ever been measured on this stack and that "there is no SparseCore path on this stack to check it
  against". There is, it works, and job `j-0b008512-36f2-4eb2-bdc6` is the evidence. A design that
  had trusted that line would have ruled out the entire SparseCore option space for no reason.
- **`sparsecore_topk.py`'s module docstring says "all 32 subcores on the chip".** The hardware
  reports `num_cores=2, num_subcores=16` — that is 16 subcores *per* SparseCore, and 32 per chip
  only if the two are multiplied. The kernel's partitioning constraint should be checked against
  the field it actually reads before any candidate relies on a subcore count.

**The timings are not a bandwidth measurement and must not be reused as one.** They are host
`perf_counter` around `block_until_ready`, so they include dispatch. 208 µs → 222 µs across an 8x
increase in `n` is the signature of fixed overhead dominating, not of a bandwidth wall. The implied
0.00–0.09 TB/s against a 3.7 TB/s peak is therefore meaningless as a rate.

**But the magnitude is a warning worth carrying into step 2.** If stage B really costs on the order
of 200 µs at any shape, it dwarfs the entire modelled decode floor of 47.5 µs, and the op would be
dominated by the top-k rather than by the scoring the roofline models. That would relocate the
optimization target completely. The microbenchmark's `stage_b` arm is what settles it; until then
this is a flag, not a finding.
