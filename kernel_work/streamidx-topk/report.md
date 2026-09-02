# Step 1 — `streamindex_topk` (the `streamidx_topk_jax` op) for GLM-5.2

Status: **step 1 complete.** Roofline, instruments, baseline, correctness, achieved fractions, and
the cross-check against the in-model xprof profile — which **agrees to 11%** (§6.2).

**Nothing here is a measurement unless it carries a job id.** Everything from §4 is `modelled`. The
jobs that measured something: `j-0b008512-36f2-4eb2-bdc6` (SparseCore probe),
`j-8b1e7acc-ebdc-45d2-b46c` (baseline, correctness, two layout/VPU probes),
`j-c13c8436-ee1e-49ee-b514` (the two deployments side by side). The profile read is
`j-a71a648c-3064-471a-a131` rank 0, corroborated on `j-97102390-e817-4ca6-a9cf`.

**Read §6.1 before §3–§6.** GLM-5.2 is served here at **two different operating points** —
`max_model_len` 9216 and 262144 — and §3 through §6 are built on the first while every available
profile captures the second. §6.1 keeps them apart; §6.2 measures both in one job.

**The verdict is in §8.** Decode has structural headroom on both deployments — **21.5%** of its
modelled floor at the nightly config, **8%** at AgentX — and the AgentX config is the target, for
the reason in finding (4). Prefill is **unresolved**: its 72.7% must not be carried forward as
headroom, and no captured profile contains an indexer-active prefill step to check it against.

Two of the three step-1 probes returned confident answers to questions other than the ones asked;
§7 corrects both rather than quoting them.

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

**Baseline: job `j-8b1e7acc-ebdc-45d2-b46c`, `JobStatus=Succeeded`**, tpu7x single-host, image
`vllm-torchtpu-hning_google_com:ec4bcecf...`, code at `cbabaf9b` (md5s of all four scripts and both
kernel modules verified against the local tree in the pod log). Per-call medians. Times in µs.

| shape | total | stage B | init | stage A (derived) | stage A (dispatch-corrected) |
| --- | --- | --- | --- | --- | --- |
| `prod_decode_b64_ctx8192` | 198.5 | 66.6 | 43.7 | 88.2 | ~130 |
| **`prod_decode_b64_ctx8704`** | **221.4** | 67.4 | 44.8 | 109.2 | ~151 |
| `prod_decode_b64_ctx9216` | 223.4 | 64.2 | 43.4 | 115.8 | ~158 |
| `prod_decode_b8_ctx8704` | 142.5 | 90.3 | 42.5 | 9.7 | ~52 |
| `prod_decode_b32_ctx8704` | 147.2 | 63.6 | 59.5 | 24.0 | ~66 |
| `prod_prefill_q1024_ctx1024` | 50.6 | (205.0) | 46.0 | n/a | n/a — early-exit fast path |
| `prod_prefill_q1024_ctx5120` | 607.7 | 338.7 | 46.0 | 223.0 | ~265 |
| **`prod_prefill_q1024_ctx9216`** | **857.3** | 481.0 | 41.2 | 335.1 | ~376 |

Column definitions. **total** = the whole `streamindex_topk` call. **stage B** = `sparsecore_topk`
alone at the exact `(num_tokens, num_positions, k)` the op feeds it. **init** = the
`jnp.full(-inf)` fill alone. **stage A (derived)** = `total − stage B − init`; it is **not
measured**. **stage A (dispatch-corrected)** = the same subtraction after removing one measured
host-dispatch floor (**42.0 µs/call**) from each separately-timed arm, since `total` pays that cost
once while `stage B` and `init` each pay it again. The corrected column is the better estimate and
is still an estimate — the floor was measured on a one-argument op and `streamindex_topk` takes
seven, so 42.0 µs is a *lower* bound on the real dispatch cost.

The fast-path row's `stage B` of 205.0 µs exceeds its `total` of 50.6 µs. That is not a
contradiction: on the fast path `total` skips stage B entirely, while the `stage B` arm still times
it standalone. Both are host-bound. No stage A is derivable there and none is claimed.

### Achieved fraction, against the modelled floors of §4

| phase | measured | modelled floor | **achieved** | exit-table band |
| --- | --- | --- | --- | --- |
| decode, b64 ctx8704 | 221.4 µs | 47.5 µs | **21.5%** | **< 50% — structural** |
| prefill, q1024 ctx9216 | 857.3 µs | 623.3 µs | **72.7%** | 50–80% — one iteration |

**These fractions are lower bounds, and the reason matters more than the numbers.** The §4 floor
models stage B only as an HBM read of the score matrix — it gives its *compute* no term at all.
Stage B measures 67 µs at decode and **481 µs at prefill, 56% of the whole call**. So the true floor
is higher than modelled and the true achieved fraction is higher than tabulated. Quantifying the
sensitivity:

- **Decode survives it.** Even crediting half of stage B as irreducible, the floor goes 47.5 → 81 µs
  and achieved goes 21.5% → 37%. Still well inside the structural band. The decode verdict is robust
  to this uncertainty.
- **Prefill does not.** The same adjustment puts the floor at 864 µs against a measured 857 µs —
  **101%**, i.e. already at or past its floor. Prefill's verdict is therefore *unresolved*, not
  "72.7%", and it would be wrong to carry that figure forward as headroom.

- **Correctness: PASSED.** Same job, shard 1: **`13 passed in 21.41s`**, no failures, no skips, and a
  non-zero collected count. Oracle is `streamindex_topk_ref`
  (`tests/kernels/test_dsv4_streamindex_topk.py:58-118`), an exact NumPy triple loop, imported by
  path so it cannot drift, compared as sorted index sets with zero tolerance.
- **What the instruments cannot measure**: end-to-end effect; contention with the co-resident engine
  on the same chip; the MoE layers the op is interleaved with; and — because stage A cannot be
  invoked in isolation without editing the incumbent — **stage A's time is derived, not measured.**
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
  figure derived at §3's config. **Read back; the caveat was justified — see §6.1.**

## 6.1 The cross-check: two deployments, not one

The read-back settles it. GLM-5.2 is served here at **two materially different operating points**,
and every number in §3, §4 and §6 above is the first of them:

| | **nightly perf** — §3's config | **AgentX** — what the profiles captured |
| --- | --- | --- |
| source | `scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh` | `cdk job recipe j-a71a648c-3064-471a-a131`, verbatim |
| `max_model_len` | **9216** | **262144** |
| `block_size` (page) | 1024 (`get_page_size`, a literal) | **256**, `--block-size=256` |
| TP × DP | 1 × 16 | 16 × 1 (siblings: 2×8, 8×2, 1×16) |
| `max_num_seqs` | 64 | 64 (observed padded batch **16**) |
| `max_num_batched_tokens` | 1024 | 1024 |
| workload | fixed 8192:1024, no jitter | replayed SemiAnalysis CC agent traces |
| `num_positions` (score width) | 10240 | **262144** |

A 25.6× wider score buffer and a 4× smaller page. **Comparing the profile's per-invocation cost
against §6's baseline would have been comparing two deployments and reporting the difference as a
property of the kernel.** The like-for-like row is being measured; until it lands, what follows is
split into what the profile establishes *regardless* of configuration and what it does not
establish at all.

### What the profile establishes, and does not depend on which deployment

Primary: `j-a71a648c-3064-471a-a131` rank 0, decode, 36 steps. Corroborated on
`j-97102390-e817-4ca6-a9cf` rank 0 — a completely different sharding (DP=2/TP=8) — with
per-invocation figures matching to within 1%.

1. **21 of 78 layers carry an indexer.** `StreamIdxTC-m` 21 distinct HLO instructions and
   `sc_topk_direct_*` 21, against `MLA-*` 78 and `sc_dsa_gather` 78. This is a static property of
   the compiled HLO, so it is not a sampling artifact, and it **independently confirms the
   `config.json` reading in §3** (`index_topk_freq: 4` plus the explicit 78-entry `indexer_types`
   list). vLLM's code default of 1 → 78 is not what this checkpoint runs.

2. **The indexer is the largest single item in the decode step.** `hlo_stats` self-time, module
   self time 26.584 ms/core-step:

   | bucket | ms/core-step | % |
   | --- | ---: | ---: |
   | **indexer top-k** (kernels + SparseCore + offload) | **8.720** | **32.8%** |
   | collectives (`all-reduce`) | 4.983 | 18.7% |
   | MLA attention (`MLA-*` + `sc_dsa_gather`) | 4.233 | 15.9% |
   | MoE (`gmm_v2`, `router_topk`, permute/combine) | 2.743 | 10.3% |
   | dense GEMM | 2.616 | 9.8% |
   | indexer plumbing (`conditional`, `bitcast_convert`, cond copies) | 1.230 | 4.6% |

   Larger than MLA and MoE **combined**. The conservative cross-check — raw TensorCore timeline
   against wall — is 6.267 ms of a 23.665 ms step = **26.5%**, and 22.4% on the DP=2/TP=8 sibling.
   Collectives at 18.7% are inflated by this job's TP=16; at the nightly's TP=1 that share
   collapses and the indexer's relative share rises further.

3. **The 256-wide K cache is confirmed in production HLO.** The operand is `u8[3027,64,4,256]`.
   §5 derived that shape from `unsigned_align_to(kv_dim, 128)`; the compiled program has it. The
   microbenchmark's dry run at `--page-size 256` prints `cache_kv shape = (total_pages, 64, 4, 256)`
   — the same geometry. **129 of 256 bytes useful, measured, not inferred.**

4. **The score buffer is `max_model_len`-wide, not context-wide.** `gather_offload_async_done` takes
   `f32[16,262144]`; `sc_topk_direct_b64_n65536_k2048` consumes `s32[4194304]` = 16 × 262144. The
   compiled shape does not depend on how long the sequences actually are. This is direct HLO
   evidence for §8's "full-width buffers regardless of context", which was until now an inference
   from the source.

5. **Both stages are strongly sublinear in score width and batch.** The profile has 6.4× more score
   elements than §6's `prod_decode_b64_ctx8704` row (16×262144 vs 64×10240) and costs:

   | | §6 microbenchmark, nightly config | profile, AgentX config | ratio |
   | --- | ---: | ---: | ---: |
   | stage A (all three scoring calls) | ~109–151 µs | 125.25 µs | ≈1.0× |
   | stage B (SparseCore top-k) | 67.4 µs | 100.5 µs | 1.49× |

   6.4× the elements for 1.0–1.5× the time. **This is finding (1) of §8 — "the op barely scales
   with the work" — reproduced by a completely independent measurement at a completely different
   configuration.** It is the most robust thing in this document.

6. **`StreamIdxTC-d` is bimodal, and unexplained.** Two calls per indexer layer: 21 distinct
   instructions averaging **114.63 µs** (tight — 114.71–114.73 at the top) and 21 averaging
   **5.40 µs**, a 21× spread. `StreamIdxTC-m` is 5.22 µs. The reason is not visible at the XLA op
   level and the microbenchmark does not currently separate the two. **Step-2 probe.**

7. **A cost the microbenchmark does not measure at all: `gather_offload`.** 2 per indexer layer,
   51.54 µs each on SparseCore, and 51.77 µs each of *TensorCore stall* waiting on them — 2.148 of
   the 6.267 ms TC timeline, **34% of the indexer's TensorCore cost**. It sits between the two
   top-k stages, gathering the candidate logits for the 131072 stage-1 survivors. The two-stage
   structure (`b64_n65536` → gather → `b16_n8192`) exists *because* n = 262144 is too wide for one
   pass; at the nightly's 10240 a single pass suffices, so this term is expected to be absent
   there. **Expected, not verified** — the like-for-like run is what checks it.

   Per indexer layer the profile therefore reads: **125.25 µs of TensorCore compute, 174.6 µs of
   TensorCore stalled on SparseCore, 203.6 µs of SparseCore time.** A TensorCore-only floor such as
   §4's does not contain the third of those at all.

### What the profile does not establish

- **Any prefill number.** No captured window in any of the four jobs contains an indexer-active
  prefill step. §4's 623 µs prefill floor is **unchecked**, on top of §8 already recording prefill
  as `unresolved`. Nothing here changes that verdict in either direction.
- **The real context lengths.** The compiled shapes are `max_model_len`-sized regardless, so the
  profile cannot distinguish "scored 262144 real tokens" from "scored 262144 slots of which most
  were padding". The AgentX trace distribution was not read.
- **Anything inside the Pallas kernels.** `StreamIdxTC-*` and `sc_topk_direct_*` are opaque
  `tpu_custom_call`s and XLA attributes zero FLOPs to them, so no roofline, MXU or DMA-stall figure
  from this profile is meaningful for them. That needs `/xprof-llo` and a capture with the
  kernel-trace `LIBTPU_INIT_ARGS`, which these do not have.

### Two of the four jobs are unusable, and the reason is worth recording

`j-7afc895e` — the job I named first — and `j-f37c33fb` captured **only DP lockstep dummy batches**:
`execute_dummy_batch` 212, `_run_dp_dummy_chunk` 212, `execute_model` **0**, on rank 2 and all seven
odd ranks available (ranks 0 and 1 absent from GCS). The indexer sits inside an XLA `conditional`
and only the cheap branch executed, so `StreamIdxTC-*` and `sc_topk_direct_*` have **zero rows** in
the full 16131-row `hlo_stats` and zero raw events on any core plane. That is a real absence of
execution, not a profiling gap — the same profile shows `sc_dsa_gather` and `MLA-*` firing normally.
The cause is `CLIENT_CONCURRENCIES=1` against `--data-parallel-size=8`.

**A profile of a DP rank running dummy batches looks exactly like a profile of a kernel that is
free.** Check `execute_model` against `execute_dummy_batch` on the host plane before reading any
per-op number off a DP capture.

Both usable captures also report `Trace Buffers Dropped` (18537 ms and 8536 ms), so absolute totals
over the window are short. Per-invocation averages and per-step ratios — which is all that is
quoted above — are unaffected.

## 6.2 The like-for-like run — the cross-check passes, and finds the real target

**Job `j-c13c8436-ee1e-49ee-b514`, `JobStatus=Succeeded`, 4/4 shards**, image
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b`
(byte-identical to the baseline job's — no image was built), code `da40156a`, bench script md5
`bbf9a3db99a7f683351768d4a794476d` verified in-pod against the local file. Four shards, one per
chip: both deployments at 5 trials and both again at 15, so every row below has an independent
replicate. **They agree to within a few µs on every row.**

### The cross-check itself

Both quantities below are the same thing — the wall-clock of one `streamindex_topk` call. In the
model the TensorCore is *blocked* during the SparseCore stages, so the profile's per-layer
TensorCore timeline is the call's cost, not just its compute.

| | value | source |
| --- | ---: | --- |
| profile, per indexer layer | **298.4 µs** | 6.267 ms/step ÷ 21 layers, `j-a71a648c` rank 0 |
| bench, `agentx_decode_b16_ctx8192` | **269 µs** | `j-c13c8436`, 15 trials (276 µs at 5) |
| bench, `agentx_decode_b16_ctx32768` | **472 µs** | `j-c13c8436`, 15 trials (470 µs at 5) |

The profile lands **between the two context rows**, 11% above the ctx-8192 one — exactly where a
replayed agent workload with contexts somewhat over 8k should land. The decomposition agrees too,
after adding back the 43.4 µs/call host dispatch floor each derived arm double-pays:

| | bench (b16, ctx8192) | profile | |
| --- | ---: | ---: | --- |
| stage A — the three scoring kernels | ~118 µs | **125.25 µs** | `StreamIdxTC-d` 114.63 + 5.40, `-m` 5.22 |
| stage B — SparseCore top-k + offload | 165 µs | **174.6 µs** | 2×`gather_offload` 51.77, `sc_topk` done 56.19 + 14.88 |

**The microbenchmark measures what the model pays.** This also resolves §6.1's open worry that
`gather_offload` was unmeasured: at 262144 the bench's stage-B arm goes through the same two-pass
split and captures it.

### The finding this run exists to have made

The two deployments can now be compared at **identical real work** — same 64 sequences, same 8192
tokens of context, byte-identical KV streamed. Only `max_model_len` differs.

| | nightly, `mml` 9216 | AgentX, `mml` 262144 | ratio |
| --- | ---: | ---: | ---: |
| `num_positions` (score width) | 10240 | 262144 | 25.6× |
| **`kv_stream` — the actual work** | **128.0 MB** | **128.0 MB** | **1.00×** |
| `score_buf` | 2.5 MB | 64.0 MB | 25.6× |
| **total** | **196 µs** | **671 µs** | **3.42×** |
| stage B | 64 µs | 371 µs | 5.8× |
| stage A (derived) | 87 µs | 264 µs | 3.0× |
| init | 45 µs | 35 µs | 0.76× |

**Identical work, 3.4× the time.** The op's cost is set by `max_model_len` and not by the context
it is actually asked to score. At the AgentX deployment 8192 real tokens sit in a 262144-wide
buffer — 3.1% useful — and the call costs three and a half times what the same work costs when the
buffer is sized to it.

Stage A is the surprise. It streams the *same 128 MB* of KV in both columns and still takes 3× as
long, so its extra cost is not the read: it is the grid. At `bkv_p`=2 the padded position range
gives 5 kv blocks at the nightly config and 512 at AgentX, and the kernel walks all of them,
writing `-inf` across a buffer 25.6× too large. That is the mechanism behind §8's finding (1), now
isolated rather than inferred.

### This reframes which deployment to optimize

The nightly config's own comment says model-len 9216 beat the 32768 geometry by **+7% throughput /
−7% TPOT** — someone already met this cost at the whole-model level and worked around it by
shrinking `max_model_len`. **The AgentX deployment cannot use that workaround**: replayed agent
traces need the long context, which is what the 262144 is for. So the headroom is 3.4× larger there
*and* it is the configuration with no other way out.

Rough achieved fractions at the AgentX config, against an HBM floor recomputed the same way as §4
(`kv_stream` + score write + score read, ÷ 2.97e12 B/s achieved per-TensorCore):

| shape | HBM bytes | modelled floor | measured | achieved |
| --- | ---: | ---: | ---: | ---: |
| `agentx_decode_b16_ctx8192` | 64 MB | 21.5 µs | 269 µs | **8%** |
| `agentx_decode_b64_ctx8192` | 256 MB | 86 µs | 671 µs | **13%** |

Both are `modelled` and both are rougher than §4 — they are a recomputation, not a rederivation.
But at 8–13% the exit table's verdict does not depend on the precision: this is far below the 50%
line, and it is a stronger **proceed** than the nightly config's 21.5%.

### Two things the run turned up that are not results about the kernel

- **`agentx_decode_b64_ctx32768` OOM'd** — "attempting to allocate 16.00G, there are 14.74G free" —
  on both shards that tried it. This is a microbenchmark artifact of running unsharded on one chip:
  the bench allocates `max_num_seqs × pages_per_seq` = 65536 pages, while production's HLO shows
  `u8[3027,...]`, a cache sized to available HBM. It does not change the streamed bytes or the score
  width, so the rows that did run stand.
- **The OOM did not turn the shard red.** `main()` catches per shape and emits a structured `oom`
  row, so both shards exited 0 and `JobStatus` was green with a missing measurement inside. Reading
  the exit code alone would have missed it.

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

**The warning it raised was correct.** The microbenchmark confirms stage B is a first-order cost:
67 µs at decode (30% of the call) and **481 µs at prefill (56%)**. §4's floor gives stage B's
compute no term at all, which is the single largest known defect in that floor — see §6.

---

```
probe:    Is the K-cache row physically padded from 129 bytes to 256?
config:   same job j-8b1e7acc, shard 2. jnp.zeros at (512,1024,129) / (512,1024,256) /
          (512,256,4,129) / (512,256,4,256) / control (512,1024,128) uint8,
          read via device memory_stats and Array.format
job:      j-8b1e7acc-ebdc-45d2-b46c, JobStatus Succeeded
answer:   Every allocation is charged its logical size (ratios 1.001-1.002). A 129-wide
          array is NOT padded -- XLA avoids it by reordering, choosing
          major_to_minor=(0,3,2,1) at 129 wide versus (0,1,2,3) at 256, moving the
          129 dim out of the minor position. Tiling is ((4,128),(4,1)) in both.
decided:  see below -- the probe's own printed VERDICT is answering the wrong question.
```

**The probe's `VERDICT_lkv_dim: 129 - there is no padding to reclaim` is wrong, and I wrote the
logic that printed it.** It tests whether XLA pads a 129-wide array. The live cache is not
129-wide: `align_to` bakes 256 into the *shape* before allocation, as §5 establishes from code and
as the benchmark's own geometry line independently confirms — `cache_kv shape = (total_pages, 256,
4, 256) uint8`, `useful 129/256 bytes = 50.4%`. There is no XLA padding because the padding is
explicit in the shape.

**Read correctly, the probe strengthens the case rather than killing it.** It shows XLA is perfectly
able to lay out a 129-wide array with no padding at all (charged ratio 1.001). So the 127 wasted
bytes per record are a *choice made in the shape computation*, not a hardware or layout constraint —
which is precisely what makes them reclaimable. Had the answer been "the layout forces 256", the
lever would not exist.

The lesson for later probes: a probe that asks about a shape the production code does not allocate
can return a clean, confident, correctly-computed answer to a question nobody asked.

---

```
probe:    Is the prefill floor's VPU term right? Measures the relu -> *w -> sum_h chain
          at the production tile, collapsing the assumed 1.72 Tops/s rate and the
          assumed 3 ops/element into one measured number.
config:   same job j-8b1e7acc, shard 3. [128, 32, 10240] f32 tile, x8 for the full
          T=1024 chunk. Three forms: as-is, folded relu(x*w), and sum-only.
job:      j-8b1e7acc-ebdc-45d2-b46c, JobStatus Succeeded
answer:   as-is 227.1us/tile -> 1816.7us full;  folded 215.7us -> 1725.3us;
          sum-only 226.0us -> 1807.7us.  Fold is bit-exact on TPU (max_abs_diff 0.0).
decided:  NOTHING about the VPU rate. The probe does not measure what it set out to --
          see below.
```

**This probe failed at its actual purpose, and the data says so plainly.** Sum-only does one op per
element and the as-is chain does three, yet they time the same: **226.0 µs versus 227.1 µs**. If VPU
op count were the binding constraint, sum-only would be roughly 3x faster. It is not, so all three
forms are bound by something they share — moving the materialized 167.8 MB tile — and not by VPU
throughput. 167.8 MB / 227 µs ≈ 0.74 TB/s is a plausible rate for a strided reduction over HBM and
an implausible one for a VPU limit.

**The probe measures the one thing the real kernel is built to avoid.** `compute_scores` keeps that
intermediate in VMEM and never materializes it; this probe materializes it and then measures the
memory traffic. So it cannot be read as either confirming or refuting the 586 µs VPU term, and
comparing its 1816.7 µs against 586 µs — which the probe's own `READ_THIS_AS` line invites — would
be comparing two different quantities. The contention caveat (four shards sharing the pod) is real
but secondary; the design of the probe is the primary defect.

**What it would take to answer the question:** the same chain inside a Pallas kernel with the tile
resident in VMEM, so that HBM is out of the loop and the VPU is the only thing left to bind. That is
a step-2 probe, and until it runs the prefill floor's dominant term remains **`assumed`**.

Two things it did establish, both incidental:
- **The fold `relu(x)*w == relu(x*w)` is bit-exact on TPU** (`max_abs_diff = 0.000e+00`, versus
  3.8e-6 on CPU). It is still only valid where `w >= 0`, which the weights' construction
  (`indexer_weights * softmax_scale * n_head**-0.5 * q_scales`) does not guarantee. Must be checked
  against real weights before any design uses it.
- Whatever removes one elementwise op from this chain buys ~5% here (215.7 vs 227.1 µs), in a regime
  that is not the one the kernel runs in. Not a basis for a design.

---

## 8. Step-1 verdict

**Step 1 is complete. The gate is passed at decode, on both deployments, and the cross-check
against the in-model profile agrees to 11%.** Prefill is unresolved and must not be treated as
headroom.

| | nightly decode | **AgentX decode** | prefill (nightly) |
| --- | ---: | ---: | ---: |
| measured | 221.4 µs (`j-8b1e7acc`) | **269 µs** (`j-c13c8436`) | 857.3 µs (`j-8b1e7acc`) |
| modelled floor | 47.5 µs (§4) | 21.5 µs (§6.2, rough) | 623.3 µs (§4) |
| achieved | 21.5% | **8%** | 72.7% |
| crediting half of stage B as irreducible | 37% | 16% | **101%** |
| in-model cross-check | not captured | **298.4 µs, +11%** | no capture exists |
| verdict | structural — proceed | **structural — proceed, and this is the target** | **unresolved — do not proceed on this number** |

Every floor figure is `modelled`; every measured figure carries its job id. The AgentX floor is a
recomputation in §6.2 rather than a rederivation, and is the rougher of the two — but at 8% the
exit-table verdict does not turn on its precision.

### The four findings that order step 2

1. **The op's cost is set by `max_model_len`, not by the work.** §6.2 measures this at identical
   real work — same 64 sequences, same 8192-token context, byte-identical 128 MB of KV streamed —
   and gets **196 µs against 671 µs, a 3.4× spread**, with the only difference being a score buffer
   25.6× too wide. The same signature appears in the batch sweep (1.55× the time for 8× the tokens)
   and in the profile (6.4× the score elements for 1.0–1.5× the time). **Three independent
   measurements at two configurations; this is the most robust thing in this document, and it is
   what a kernel should attack.**

2. **Stage B is the largest single component and the floor models none of its compute.** 30% of
   the nightly decode call, 56% of prefill, **61% of the AgentX decode call** (165 of 269 µs), and
   confirmed in the profile at 174.6 µs of the 298.4. It is an exact MSB-first radix select. Whether
   it is near its own floor decides whether prefill has any headroom at all and re-ranks every
   decode candidate.

3. **127 of every 256 K-cache bytes are padding**, in the term that is 95% of nightly decode HBM
   traffic. §7 establishes the padding is a shape choice, not a layout constraint; §6.1 confirms the
   shape in production HLO (`u8[3027,64,4,256]`).

4. **The AgentX deployment is the right target, and it is not the one §3 was built on.** The nightly
   config already worked this cost around by shrinking `max_model_len` to 9216 — its own comment
   records +7% throughput / −7% TPOT for doing so. A long-context agent deployment cannot. The
   headroom is 3.4× larger there and there is no configuration escape from it.

### What step 2 must probe before designing

Each of these currently carries a design decision:

- **stage B's own floor**, and whether exact top-k is required or approximate would do
  (`sparsecore_topk` replaced `approx_max_topk` in `6142d5e7`; the reason is not recorded)
- **whether the grid can be bounded by `seq_len` instead of `max_model_len`** — finding (1) says
  this is the lever; nothing yet says what stops the incumbent from doing it
- **whether the `-inf` init is needed at all**, given `sparsecore_topk` takes `row_lengths`
- **the VPU term, properly, in VMEM** — §7 explains why the first attempt does not count
- **whether `StreamIdxTC-d`'s bimodality is by design** — 114.63 µs against 5.40 µs across 21+21
  instructions (§6.1), unexplained at the XLA op level
- **whether the ~43 µs dispatch floor is real per-call overhead in the model** or an artifact of
  timing the op standalone. At 21 invocations per step, the difference between a rounding error and
  0.9 ms/step.
- **whether the indexer weights are non-negative**, which gates the bit-exact `relu(x)*w == relu(x*w)`
  fold (§7)
