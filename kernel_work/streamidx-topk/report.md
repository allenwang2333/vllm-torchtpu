# Step 1 — `streamindex_topk` (the `streamidx_topk_jax` op) for GLM-5.2

Status: **step 1 complete, and self-corrected.** Roofline, instruments, baseline, correctness,
achieved fractions, and an attempted cross-check against the in-model xprof profile.

**This document has been through an adversarial review and carries its retractions inline rather
than silently editing them out.** Five things an earlier draft asserted are withdrawn: the 11%
profile cross-check (§6.2 — it compared a `--concurrency 1` profile against a full-batch
microbenchmark and could not have failed), the 3.42× `max_model_len` ratio (§8 finding 1 —
confounded with page size; ≈2.5× survives), the ~43 µs dispatch add-back (§6.2 — it was fitting; the
profile has the real number), the 8%/16% AgentX achieved fractions (§8 — MiB read as MB, and one
cell used the wrong formula), and "AgentX is the right target" (§6.2 — no artifact supports it being
a served configuration). Where a retraction is marked, read the retraction and not the claim.

**Nothing here is a measurement unless it carries a job id.** Everything from §4 is `modelled`. The
jobs that measured something: `j-0b008512-36f2-4eb2-bdc6` (SparseCore probe),
`j-8b1e7acc-ebdc-45d2-b46c` (baseline, correctness, two layout/VPU probes),
`j-c13c8436-ee1e-49ee-b514` (the two deployments side by side). The profile read is
`j-a71a648c-3064-471a-a131` rank 0, corroborated on `j-97102390-e817-4ca6-a9cf`.

**Read §6.1 before §3–§6.** GLM-5.2 is served here at **two different operating points** —
`max_model_len` 9216 and 262144 — and §3 through §6 are built on the first while every available
profile captures the second. §6.1 keeps them apart; §6.2 measures both in one job.

**The verdict is in §8, and it survives every retraction above.** Decode has structural headroom on
both deployments: **21.5%** of its modelled floor at the nightly config and **8.4%** at AgentX, or
**37% and 39%** once half of stage B is credited as irreducible. Far below the 80% line and below
50%. **Writing a kernel is justified.** The two deployments are equivalent on this measure, so the
nightly config — the one §3–§6 are built on, and the one the repo actually ships — is the baseline
of record. Prefill is **unresolved**: its 72.7% must not be carried forward as headroom, its
dominant floor term is an `assumed` rate that §7's probe failed to measure, and no captured profile
contains an indexer-active prefill step to check it against.

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
kernel modules verified against the local tree in the pod log). Times in µs.

**These are best-of-N per call, not medians** — `bench_streamidx_topk.py:624` is
`best = min(best, (time.perf_counter() - t0) / iters)` over the trial loop. An earlier draft of this
section said "per-call medians"; it was wrong. Best-of-N is the right choice here (four shards share
the pod, so the distribution's upper tail is contention rather than the kernel), but it means every
number below is an *optimistic* estimate of the op's cost, and the two arms of a subtraction are
each optimistic independently. That is one more reason the derived `stage A` column is soft.

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
measured**. **stage A (dispatch-corrected)** — ⚠️ **RETRACTED, do not use this column.** It added
back a measured one-argument host-dispatch floor (42.0 µs/call) on the theory that `total` pays it
once while `stage B` and `init` each pay it again. §6.2 shows that was fitting, not correction: the
floor varies 34.6–44.9 µs across concurrent shards, `init` comes in *below* it in 15 of 20 rows
(which would imply negative device time), and `time_arm` blocks once at the end of its loop, so a
call costs roughly `max(host, device)` and not their sum. The column is left in place because the
arithmetic that produced it should be visible; the values are not evidence of anything. Use the
**stage A (derived)** column, understanding that it is a subtraction of three
independently-optimistic best-of-N timings and therefore soft in both directions.

The fast-path row's `stage B` of 205.0 µs exceeds its `total` of 50.6 µs. That is not a
contradiction: on the fast path `total` skips stage B entirely, while the `stage B` arm still times
it standalone. Both are host-bound. No stage A is derivable there and none is claimed.

**Run-to-run spread, measured.** `j-c13c8436` re-ran `prod_prefill_q1024_ctx9216` and got
**929.9 µs** against the 857.3 µs above — **+8.5% on the total, +27% on the derived stage A** — at
the same code and the same shape, differing only in which shard of the pod it landed on and what
the other three shards were doing. Nothing in §8 turns on the difference, but two things follow.
Any comparison below that pairs a number from `j-8b1e7acc` with one from `j-c13c8436` is
cross-job, and I have built arguments on the 857.3 while a later job said 929.9. And a variant in
step 3 beating the incumbent by less than ~10% on a derived stage A has not beaten it: **the
incumbent must be re-measured in the same job as every challenger**, which the skill requires
anyway and this number is the reason why.

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

### The cross-check itself — RETRACTED, it compared unlike things

**What this section originally claimed:** that the profile's 298.4 µs/layer landed "between the two
context rows, 11% above the ctx-8192 one — exactly where a replayed agent workload with contexts
somewhat over 8k should land", and therefore that the microbenchmark measures what the model pays.

**That is withdrawn.** A `/critical-review` of this document pulled the profiled job's own artifacts
— which this report had recorded as unread — and they contradict the assumption the check rested on:

| | assumed here | what the job's artifacts say |
| --- | --- | --- |
| batch | 16 real sequences | **`--concurrency 1`**; effective decode concurrency avg 1.02, p50 1.00, max 4.00 |
| context | "somewhat over 8k" | **prompt tokens avg 105,864, p50 110,897, max 169,053** |

Source: `benchmark_command.txt` and `aiperf_artifacts/profile_export_console.txt` under
`gs://cloud-devkit/jobs/j-a71a648c-3064-471a-a131/outputs/.../sidecar-bench/conc_1/`.

So the profiled call is **one real sequence of ~110k tokens in a compiled batch-16 bucket with
fifteen padding slots** — 16 is the smallest entry in `compile_sizes`, the bucket floor. The bench
arm it was compared against is **sixteen real sequences of 8,192 tokens**. The two differ on both
axes at once, and the errors run in opposite directions on total KV volume. That is how two unlike
numbers land 11% apart and look like corroboration.

**And the check could not have failed.** Any profile value in [269, 472] µs would have been "between
the two rows" — a 1.76× acceptance window, one free parameter, no residual test. §6.1 had listed the
trace distribution under *"What the profile does not establish"* and recorded that it was not read;
§6.2 then closed the cross-check on an assumption about it. It was one `gcloud storage cat` away.

The decomposition claim went with it. The ~118 µs stage-A figure depended on adding back a 43.4 µs
"host dispatch floor" per derived arm, and that correction does not survive either — see below.

**What replaces this:** the bench now models the profiled batch directly
(`num_real_seqs`/`pad_seq_len`, commit `9e880312`), as a *bracket* — padding slots at `seq_len` 0
against padding slots at the full context, 28.9 MB against 432 MB of streamed KV. Which end is
right depends on whether the per-slot KV DMA is guarded, which the loop bound
(`kv_len_max = jnp.max(jnp.stack(kv_lens))`, `streamindex_topk.py:275-276`) does not settle on its
own. Until those rows are measured **there is no validated cross-check between this microbenchmark
and the model**, and §8's numbers are microbenchmark numbers.

### The dispatch add-back was fitting, and the profile has the real number

The `stageA*` column is derived as `t_total - t_b - t_init`, and each arm separately pays a host
dispatch cost, so I added ~43 µs back. The algebra is right; the model underneath it is not:

- **It implies negative device time.** `t_init = D + I` requires `init_ms >= dispatch_floor_ms`. In
  **15 of the 20 successful rows** of `j-c13c8436` it is not — including the headline row, where
  init is 28.6 µs against a 43.4 µs floor.
- **The floor is not a constant.** 44.9, 34.6, 43.4, 43.4 µs across four shards running
  concurrently — a 30% spread on a correction that is 58% of the quantity being corrected.
- **`time_arm` cannot support additivity.** It enqueues `iters` calls and blocks *once* at the end,
  so per-call ≈ max(host enqueue, device), not their sum — and `iters` differs per arm.

**The profile supplies the correct figure.** The in-model `-inf` init is
`%broadcast.33911 = f32[16,2048,128] broadcast(f32[] constant)`, 21 instructions, avg self time
**6.02 µs**, which is 2.79 TB/s for a 16.78 MB write — physically sensible. The shape is exactly the
bench's `init_shape`. The bench's init arm reads 28.6 µs for the same write, so ~22.6 µs of that arm
is harness overhead, not 43.4.

**This answers one of §8's step-2 probe questions outright: the `-inf` init is 6.02 µs of a ~300 µs
call — 2% — and is not worth attacking.** Removing it from the candidate list is the one unambiguous
gain from this cross-check.

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

**3.4× the time — but the two columns differ in page size as well, and that is a confound I did not
isolate.** The review sized it, and the effect splits cleanly in two:

| component | nightly | AgentX | gap | driven by |
| --- | ---: | ---: | ---: | --- |
| stage B | 63.1 | 371.3 | **+308 µs** | `num_positions` 10240 → 262144 — **`max_model_len`** ✓ |
| stage A (derived) | 98.8 | 264.5 | +166 µs | `num_bkv` 4 → 16 — **page size**, confounded ✗ |
| init | 33.6 | 35.1 | +1.5 | neither; dispatch-dominated at both |

Page size sets `bkv_sz = bkv_p × PAGE` = 2048 vs 512, hence `num_bkv = cdiv(kv_len_max, bkv_sz)` =
**4 vs 16** at identical context. The bench records this itself in `geo_num_bkv` and I did not read
it. So the inner KV loop runs 4× more iterations at AgentX purely because the pages are smaller.

**The stage-B half is clean and is the real finding: 5.9× for a 25.6× wider score buffer at
byte-identical KV.** The stage-A half is not evidence for `max_model_len` at all. Holding stage A
fixed, the `max_model_len`-attributable ratio is **≈2.5×**, not 3.42×.

**My stated mechanism for stage A was also wrong.** I wrote that the grid "gives 5 kv blocks at the
nightly config and 512 at AgentX". Those are `num_positions / bkv_sz`; the loop is bounded by
`kv_len_max`, giving 4 and 16. And the `-inf` write is the separate `jnp.full` measured by the
`init` arm, which `t_a = t_total - t_b - t_init` subtracts out by construction — it cannot be inside
derived stage A. The adjacent `init` row (45 → 35 µs, *smaller* at 25.6× the width) said so and I
wrote past it.

**The deconfounding experiment was already in the file and I did not run it.** `PINNED_MAX_MODEL_LEN
= 65536` with `PINNED_DECODE` gives rows that vary `max_model_len` at **fixed page size 1024**, each
with a `max_model_len == context_len` twin. Running.

### "This reframes which deployment to optimize" — WITHDRAWN

This section argued AgentX was the target. It does not hold, for three separate reasons, and the
retraction is more useful than the claim was.

**1. The achieved-fraction gap was an arithmetic error.** §8 tabulated the AgentX sensitivity cell
as 16%, which is simply `2 × 8%` — the floor was doubled instead of half of stage B being added.
The correct figure under the same formula the other two cells use, `(floor + stage_B/2) / measured`:

| | nightly | AgentX | prefill |
| --- | ---: | ---: | ---: |
| as published | 37% | **16%** ✗ | 101% |
| recomputed | 36.7% ✓ | **38.7%** | 100.8% ✓ |

**37% and 39% — indistinguishable.** The b64 like-for-like row lands at 41%. Stage B is 61% of the
AgentX call against 30% of the nightly one, so the term that was dropped is precisely the one that
dominates the column the argument depended on. There is no headroom gap between the deployments.

**2. The raw fractions are also slightly off**, from reading MiB as MB: **8.4% and 13.5%**, not 8%
and 13%. Immaterial to the gate, and in the self-serving direction.

**3. No artifact supports AgentX being a served configuration.** The scenario is
`inferencex-agentx-mvp` over a public trace replay, at `--concurrency 1` for 30 minutes, and §6.1's
own sibling list (TP×DP of 16×1, 2×8, 8×2, 1×16) is a **sharding sweep** — the signature of
bring-up. The repo's nightly config describes TP=16 as "13× the TP=16 bring-up throughput" *worse*
than the layout it ships. The string "agentx" appears nowhere in `vllm-torchtpu` outside
`kernel_work/`. My claim that "a long-context agent deployment cannot shrink `max_model_len`" is
about a hypothetical deployment class and is supported by nothing here; it may be true, but this
document did not establish it and presented it as settled.

**4. The `+7% / −7%` citation is confounded twice and is one number, not two.** The comment at
`glm-5.2-fp8-dp16-ep.sh:9-10` reads "model-len 9216 **+ 64 seqs/engine** beat the **32768/1024**
geometry" — `max_num_seqs` changed 16× alongside `max_model_len` — and it is a **cross-build**
comparison (dev 190 vs 196). At fixed concurrency throughput ≈ concurrency/TPOT, so +7% and −7% are
the same measurement stated twice, not mutual corroboration. CLAUDE.md's rule about recovering a
repo number's configuration before quoting it applies, and I did not follow it.

**What survives:** `max_model_len` really does drive stage B (5.9× at byte-identical KV), and that
is deployment-independent — it would appear at the nightly's TP=1×DP=16 too, given a long
`max_model_len`. What must not be carried forward is AgentX's **absolute** numbers, its achieved
fractions, or its step composition (indexer 32.8%, collectives 18.7%) as the thing to optimize
against. **The nightly config is the baseline of record.**

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

**Read correctly, the probe shows the padding is a choice — but a weaker one than I first wrote.**
XLA did lay out a 129-wide array with no padding (charged ratio 1.001), so the 127 wasted bytes are
a choice made in the shape computation and not a hardware constraint. My original text went on to
call them "reclaimable", and the probe's own second reading does not support that:

```
layout_129:  major_to_minor=(0,3,2,1),  tiling ((4,128),(4,1))
layout_256:  major_to_minor=(0,1,2,3),  tiling ((4,128),(4,1))
```

**XLA avoided the padding by transposing, not by packing 129 into the minor dimension.** At 129 wide
it moved that dimension *out* of minor position. The kernel cannot consume that layout: `_fetch_bkv`
DMAs contiguous `[page_size_per_kv_packing, kv_packing, lkv_dim]` slabs and `compute_scores` needs
`lkv` minor for 128-lane loads. And the cache is not the indexer's to reshape — it is the same
`cache_kv` the MLA v2 attention kernel reads.

So the correct statement is: **the 49.6% padding is real and is not forced by the layout engine, but
no free layout that removes it has been demonstrated.** Reclaiming it means a packing change across
two consumers, which is a step-2 design question with a real cost, not a lever lying on the floor.
The decode floor of §4 keeps the 256-wide byte count.

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

Two things it did establish, both incidental — and the first is weaker than I originally wrote:
- **The fold `relu(x)*w == relu(x*w)` matched to `max_abs_diff = 0.000e+00`.** I recorded this as
  "bit-exact on TPU", which overstates it in two ways. It is **algebraically exact for any `w >= 0`**
  under IEEE-754 — `relu` either returns `x` or `+0.0`, and multiplication distributes over both —
  so it is a property of the arithmetic, not of TPU. And the probe draws `w` from
  `rng.random()` ∈ [0, 1) (`probe_vpu_chain.py:94`), so **the check could not have failed**; it
  confirms the implementation matches the algebra, nothing more. My original text also cited
  "versus 3.8e-6 on CPU" — **that figure appears in no artifact**; the probe never ran on CPU. It
  is deleted, not corrected. The real precondition is unchanged and untested: valid only where
  `w >= 0`, which `indexer_weights * softmax_scale * n_head**-0.5 * q_scales` does not guarantee.
  Must be checked against real weights before any design uses it.
- Whatever removes one elementwise op from this chain buys ~5% here (215.7 vs 227.1 µs), in a regime
  that is not the one the kernel runs in. Not a basis for a design.

---

## 8. Step-1 verdict

**Step 1 is complete. The gate is passed at decode. Writing a kernel is justified.** Prefill is
unresolved and must not be treated as headroom. The in-model cross-check is **retracted** (§6.2) —
it compared a `--concurrency 1` profile against a full-batch microbenchmark and could not have
failed; the gate does not rest on it.

| | nightly decode | AgentX decode | prefill (nightly) |
| --- | ---: | ---: | ---: |
| measured | 221.4 µs (`j-8b1e7acc`) | 269 µs (`j-c13c8436`) | 857.3 µs (`j-8b1e7acc`) |
| modelled floor | 47.5 µs (§4) | 22.6 µs (§6.2, rough) | 623.3 µs (§4) |
| achieved | 21.5% | 8.4% | 72.7% |
| crediting half of stage B as irreducible | **36.7%** | **38.7%** | **100.8%** |
| verdict | structural — proceed | structural — proceed | **unresolved — do not proceed on this number** |

Every floor figure is `modelled`; every measured figure carries its job id. Read this table by the
**bottom** row, not the one above it: the floor of §4 gives stage B's compute no term at all, and
stage B is 30% of the nightly call and 61% of the AgentX one, so the raw `achieved` row is a lower
bound that flatters the two deployments unequally. The sensitivity row applies
`(floor + stage_B / 2) / measured` uniformly.

Three corrections against an earlier draft of this table, all found by review: the AgentX
sensitivity cell read **16%**, which was `2 × floor / measured` — the wrong formula, in the one cell
that carried the retarget argument; the raw fractions read 8% and 13% from taking MiB as MB; and
`--concurrency 1` in the profiled job broke the cross-check row, which is now gone.

**The two deployments have the same achieved fraction — 37% against 39%.** There is no headroom gap
between them, and §6.2 withdraws the claim that AgentX is the target. Both are far below the 80%
line and comfortably below 50%. **This conclusion is the one thing in this document that no finding
above disturbs.**

**The prefill column is the one to be careful with.** At 101% the modelled floor is *below* the
measurement, which by the exit table means the floor is wrong or the code is skipping work — not
that prefill is optimal. Its dominant term is the `assumed` VPU rate that §7's probe failed to
measure. Prefill is out of scope for a step-2 design until that probe is redone in VMEM.

### The findings that order step 2

**1. `max_model_len` sets the op's cost, and the cost lives in stage B.** §6.2 measures this at
byte-identical real work — same 64 sequences, same 8192-token context, same 128 MB of KV streamed —
and the mml-attributable spread is **≈2.5×**, not the 3.42× an earlier draft claimed: the raw
196→671 µs comparison also changed page size 1024→256, which quadrupled `num_bkv` and moved stage A
by +166 µs for reasons having nothing to do with `max_model_len`. Decomposed:

| | Δ | driven by |
| --- | ---: | --- |
| stage B | **+308 µs (5.9×)** | `max_model_len` ✓ |
| stage A | +166 µs | page size ✗ |
| init | +1.5 µs | `max_model_len`, negligible |

The clean half is the large half and it is **entirely in stage B**, whose row width is
`num_sublanes_total` and therefore literally `max_model_len`. This is a finding about the score
buffer, not about the KV walk — and finding (2) explains why it could not have been otherwise.

**2. Stage A's KV walk is already `seq_len`-bounded across the batch, and unguarded within groups of
four.** Read from source (`streamindex_topk.py`, corroborating the decomposition above by an
independent route):

- The Pallas grid is `((end_seq_idx - start_seq_idx) // seq_batch_size,)` at `:643`, and
  `end_seq_idx` comes from `distribution`, which `tpu_runner.py:3175` sets to the **real** request
  count, not the padded bucket. **Batch-bucket padding slots are never visited.** The 15 padding
  slots in a batch-16 bucket cost nothing in stage A.
- Within one grid step, `_fetch_bkv` (`:168-194`) is a *Python* loop over `seq_batch_size` emitting
  an unconditional, constant-length `_async_copy` per slot. There is no `pl.when`, no predicated
  size. The trip count is the single scalar `kv_len_max = jnp.max(...)` at `:275-276`. Per-sequence
  `seq_lens` appears **only** in output masks (`:344`, `:347`). So **every slot in a group of
  `decode_req_batch_size = 4` pays the longest sequence in that group** — full DMA, full einsum,
  full VPU chain, full score write-back, with only the values discarded.
- The authors knew: the TODO at `:751-754` proposes sorting sequences by length for exactly this.

The waste is therefore bounded by the length spread **within groups of 4**, not by the bucket. That
is a real lever but a much smaller one than "the padding slots pay for the longest sequence", which
is what §6.2's bracket was built to measure. Contrast `_fetch_bq` at `:209-219`, which *does* issue
a variable-length copy — the machinery exists and is simply not used for KV.

**3. Stage B is the largest single component and the floor models none of its compute.** 30% of the
nightly decode call, 56% of prefill, 61% of the AgentX decode call (165 of 269 µs). It is an exact
MSB-first radix select. **Amdahl bounds everything step 2 can do to stage A: a stage A reduced to
zero caps the AgentX op at 1.64× and the nightly at 1.43×.** Combined with finding (1) — the
mml cost *is* stage B — this makes stage B's own floor the single most consequential unknown in the
document, not a footnote to a stage-A design.

**4. 127 of every 256 K-cache bytes are padding**, in the term that is 95% of nightly decode HBM
traffic. §6.1 confirms the shape in production HLO (`u8[3027,64,4,256]`). §7 establishes the padding
is a shape choice rather than a layout constraint — but **not that it is cheaply reclaimable**: the
only no-padding layout XLA demonstrated is a transpose the kernel cannot consume, and the cache is
shared with the MLA v2 attention kernel. Reclaiming it is a two-consumer packing change.

**Withdrawn: "AgentX is the right target."** See §6.2. The achieved fractions are equal once the
arithmetic is right (37% vs 39%), no artifact supports AgentX being a served configuration rather
than a bring-up sweep, and the `+7% / −7%` citation that argued the nightly config had "already
worked the cost around" is confounded twice — it moved `max_num_seqs` 16× alongside
`max_model_len`, across two dev builds, and at fixed concurrency its two halves are one
measurement. **The nightly config is the baseline of record.**

### What step 2 must probe before designing

Each of these currently carries a design decision. Ordered by what they gate:

- **stage B's own floor**, and whether exact top-k is required or approximate would do
  (`sparsecore_topk` replaced `approx_max_topk` in `6142d5e7`; the reason is not recorded). By
  findings (1) and (3) this is now the **first** probe, not one of seven — it decides whether the
  mml cost is attackable at all and it bounds every stage-A candidate.
- **whether stage B's row width can be bounded by `seq_len` instead of `max_model_len`.** Finding
  (1) locates the lever precisely here. `sparsecore_topk` already takes `row_lengths`
  (`eff_row_lengths`, `:505-506`), so the information is present; what is unknown is whether the
  radix select's cost tracks it or the buffer width.
- **whether sorting or grouping sequences by length is worth it**, given finding (2)'s
  group-of-4 granularity — this needs the real length distribution of a served batch, which no
  artifact here has.
- **the VPU term, properly, in VMEM** — §7 explains why the first attempt does not count. Gates
  prefill entirely.
- **whether `StreamIdxTC-d`'s bimodality is by design** — 114.63 µs against 5.40 µs (§6.1).
  Finding (2) supplies a candidate explanation that did not exist before: of the three
  `run_scores_kernel` calls, two get a **zero-length grid** when `distribution[0] % 4 == 0` and
  `distribution[1] == distribution[2]`, so 5.40 µs may simply be an empty kernel's fixed cost.
  Cheap to confirm, and it would remove a mystery rather than a cost.
- **whether the indexer weights are non-negative**, which gates the `relu(x)*w == relu(x*w)` fold
  (§7).

**Closed during step 1, no longer probes:**

- ~~whether the `-inf` init is needed~~ — the profile has it directly:
  `%broadcast.33911 = f32[16,2048,128]`, **6.02 µs at 2.79 TB/s**, ~2% of the call. Real, near
  bandwidth, and not worth attacking. The ~43 µs the standalone bench attributes to it is host
  dispatch.
- ~~whether the ~43 µs dispatch floor is real in-model overhead~~ — it is not; see §6.2. The
  add-back was fitting, and the same profile row above replaces it with a measurement.

### One instrument defect to fix before step 2 measures anything

`bench_streamidx_topk.py:634` sets `distribution = [num_seqs, num_seqs, num_seqs]` unconditionally,
including for the `num_real_seqs=1` bracket shapes added in §6.2. By finding (2) the grid is sized
from `distribution`, so `profile_b16_1real_ctx110592_pad0` runs **four** slots at full length
(one group of `decode_req_batch_size=4`), not one — roughly 4× the production call it was built to
model. The `padfull` arm and every homogeneous `prod_*` / `agentx_*` shape are unaffected, since
there `distribution` and the real count agree.

The fix is one line — `distribution = [num_real_seqs, num_real_seqs, num_real_seqs]` where
`num_real_seqs` is set. It is deferred only because a job using the current file is in flight;
the `pad0` numbers it returns must be read as *four* real sequences, and `_per_seq_bkv`'s
lower-bound geometry is wrong for that arm in the opposite direction.
