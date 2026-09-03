# `streamindex_topk` — final report

**GLM-5.2 on TPU7x. 2026-09-03.** Supersedes the 2026-09-02 draft, whose recommendation ("land the
merged op") is **withdrawn** — see §6. Detail lives in `results.md` (per-option) and in
`variants/*/RESULTS.md`. No end-to-end model benchmark has been run; every number here is a
microbenchmark or a production-profile read, and each is labelled.

---

## 1. The one-paragraph version

Four changes are worth landing and two that the earlier draft recommended are not. The largest
single item this exercise found is **not a kernel change and not in this op's kernel at all** — it is
a `T(4,128)` → `T(8,128)` → `T(4,128)` relayout round trip that XLA emits around the indexer
KV-cache write, costing **2.21% of a prefill step** to write 1024 rows. It is removable, the fix is
measured, and as prototyped it **buys prefill at decode's expense** and needs one more pass before it
can land. Of the kernel work proper, what survives is **Option C** (prefill-gated head-major scores)
and **D1-lite** (hoisting shared metadata out of the early-exit branches). Those two have now been
**composed into one module and measured together in a single job** (§7.1): **−3.92% of the op at
production decode and −7.02% at a heavy prefill shape**, index-identical to the incumbent over 34,368
compared rows, and additive to within the job's resolution — worth about **0.20% of a decode step and
0.05% of a prefill step**. The composed run also surfaced a regression invisible in the separate
arms: below ctx 2048, where neither kernel executes, the pair costs **1.6× the incumbent** (§7.3).
**Option B** (runner-side chunking to overlap TensorCore and SparseCore) is real, integrates cleanly,
and adds **0.155% of a prefill step**, but has not yet been composed with the other two.

---

## 2. Definitions — read these before any table

| term | meaning |
| --- | --- |
| **the op** | `streamindex_topk` as invoked from `mla_attention_op.py:_streamidx_topk_jax` — stage A, stage B, the `conditional` wrapper, the `branch_0_fun` plumbing **and** the KV-cache write. Wider than the "stage A + stage B" scope earlier documents used. |
| **stage A** | the TensorCore Pallas scoring kernels (`StreamIdxTC-*`) that write f32 scores to HBM. |
| **stage B** | `sparsecore_topk` — exact MSB-first radix select on SparseCore (`sc_topk_direct_*`). |
| **TC plane / SC plane** | `hlo_stats` rows filtered by the `TPU core type` column. **Never summed** — see §3. |
| **ms/step** | an op's `Total self time` divided by the number of core-steps in the capture (model steps × 2 TensorCores). |
| **device self time** | `hlo_stats` `Total self time (us)`. Contention-immune, so valid in a fanned-out job. |
| **chain slope** | R calls in one `jit` joined by a dependency the compiler cannot fold; the fitted slope is the marginal per-call cost. |
| **G** | Option B's chunk count — how many pieces the prefill token block is split into. |
| **`bkv_p`** | `num_kv_pages_per_block`, the KV block size in pages. **Compile-time static.** |
| **`num_bkv`** | `max(1, cdiv(kv_len_max, bkv_p × page_size))`, the scan loop trip count. **Runtime.** |
| **the triad** | the three ops around the KV-cache write: reshape → `scatter_fusion` → reshape. |
| **fast path** | the `enable_early_exit` branch, taken when `max(seq_lens) // compression_ratio <= k`. |
| `observed` / `derived` / `modelled` | measured on hardware / arithmetic on measurements / from a cost model. |

**Sign convention: `−` means faster, everywhere in this document.** Option B's own `RESULTS.md` uses
the opposite convention. Its `+24.4%` and this document's `−24.4%` are the same measurement.

---

## 3. The measurement rule that changed every model-level number

`hlo_stats` in this profile carries a `TPU core type` column with both `TensorCore` and `SparseCore`
rows. For an offloaded op the **TC row is the TensorCore's blocking wait** and the **SC row is the
SparseCore's concurrent execution** — the same wall-clock interval, recorded twice. Summing them
double-counts.

The check that settles it: TC+SC gives 81.488 ms/step at decode against a **measured module wall of
72.039 ms**; TC-only gives 69.492 ms, which is 96.5% of it. Every share in this document uses the
**TensorCore plane only, on numerator and denominator both.**

Two other traps, both of which produced a wrong table in this exercise before being caught:

- **`bdt_hlo_stats.dump()` defaults to `--top-n 500`.** A 21-layer program with 21 `lax.cond`s emits
  822–961 rows. The truncated read's partition-residual check **still passed at −4.5e−12**, because
  it sums only the rows it kept. Only `rows_kept == rows_total` catches it.
- **Per-call wall clock is `max(device, host)`, not `device + host`.** The host dispatch floor is a
  clamp; subtracting it produced numbers wrong by up to 10× in early work here. Nothing in this
  document has a dispatch floor subtracted from it.

---

## 4. Where the time actually is

`j-9b23bb32-0426-4080-be91`, recipe read back as `vllm serve "zai-org/GLM-5.2-FP8"
--data-parallel-size=16 --max-model-len=9216 --max-num-seqs=64 --max-num-batched-tokens=1024
--kv-cache-dtype=fp8`, rank 8, complete `hlo_stats`, TC plane. Denominators: decode **69.492
ms/step** (5 steps × 2 cores), `prefill_heavy` **622.485 ms/step** (1 step × 2 cores).

| component | decode ms/step | % | prefill ms/step | % |
| --- | ---: | ---: | ---: | ---: |
| stage A | 1.8259 | 2.627% | 1.2491 | 0.201% |
| stage B (TC blocking wait) | 0.6588 | 0.948% | 2.0123 | 0.323% |
| **KV-cache write — relayout** | 0 | — | **13.7780** | **2.213%** |
| KV-cache write — `scatter_fusion` | 0.1863 | 0.268% | 2.8100 | 0.451% |
| `conditional` wrapper + `branch_0_fun` plumbing | 0.3162 | 0.455% | 0.8865 | 0.142% |
| **op total** | **2.9872** | **4.299%** | **20.7359** | **3.331%** |
| *stage B, SC rows — excluded, never added* | *0.3249* | — | *1.0013* | — |

Three readings:

1. **At prefill the op is mostly not the kernel.** The relayout is **68%** of the whole op; stage A
   is **6%**. Every prefill kernel option in this exercise optimizes the small end of the op, which
   is itself the small end of the step.
2. **Stage B occupies 0.6588 ms of TensorCore against 0.3249 ms of SparseCore work** — the TC blocks
   about **2×** as long as the SC computes. That gap is what Option B recovers, now measured on a
   real model step rather than inferred from a harness.
3. **The bigger neighbour is still bigger.** `sc_dsa_gather` — the gather that consumes the indices
   this op produces — is **4.198 ms/step (6.041%)** at decode and **51.856 ms/step (8.330%)** at
   prefill, i.e. **2.5× this entire op** at prefill. The DSA complex is 10.34% / 11.66%.

*Caveat carried on every prefill figure: **n=1 model step** per capture. Two captures agree to 0.03%
on the relayout, and `prefill_only`'s step has zero indexer activity (`min_kv_len = 0`), so only
`prefill_heavy` carries stage A and stage B at all. Collectives are 40.4% of the prefill step.*

---

## 5. What ships

Ordered by measured value, not by how much work each took.

### 5.1 Option C — head-major score layout, gated to prefill/mixed · **kernel** · SHIP, with one unexplained cost

−20.6% of prefill stage A, `j-a960ad54-a0a1-4e1d-8ade`, chain slope r² ≥ 0.99999. **≈0.041% of a
prefill step** (`derived`).

**Gating is not a refinement, it is the difference between a win and a large loss.** Option C at
decode is **+59.62%** — a 1.6× regression (`j-fd7b8801-fce9-46ec-bb63`). At `bq_sz=1` the head-major
permutation is literally the identity and the compiler still fails to make it free; LLO confirms it
*adds* work. The two DECODE launches take `head_minor`; only the MIXED launch takes `head_major`.

Gated, it is inert at decode **at production context**: **+0.034%** against a **+0.041%** port
control, and +0.24% in the composed 21-layer chain (§7.1).

**It is not inert at ctx ≤ 2048.** In the fast-path regime, where Option C's kernel never executes,
it still costs **+3.5 µs/layer** — a 1.5× regression on a 7 µs call, traced to the `conditional`
HLO's own self time and *not* to any kernel. See §7.3; the mechanism is an unconfirmed IMEM
hypothesis. This does not block shipping at production context but it does mean the claim "the gate
makes Option C free at decode" is now known to be false in one regime.

**Its composition is now properly tested.** The earlier "three-way additivity" check was
algebraically a two-way check — it defined Option C's term as `Δ C ≡ Δ(merged) − Δ(bkvp_d1)`, so the
`merged` arm cancels out of the residual and the published column reduces identically to
`Δ(bkvp_d1) − Δ(bkvp) − Δ(d1)`. C's composition could not have failed that test. Measured with a
genuine `c`-alone arm (`j-d17207d1-05fb-4e10-93b3`), the real three-way residual is **+0.113% /
+0.005% / −0.026% / −0.046%** across the four prefill configs, against a within-job error bar of
0.015–0.192% (n=4 — the first replicate anywhere in this exercise).

Correctness: index-identical, `j-cf9e7189-8331-4613-9c2a`. Stage-A *scores* are not bitwise identical
(36,973 of 10.5M elements, max abs 6.10e−5 — a different f32 reduction order) but no selection
boundary moved in 7,168 rows.

### 5.2 D1-lite — hoist shared metadata out of the early-exit branches · **kernel** · SHIP, with a caveat

`j-e985297a-67f8-4998-99f8`, 4/4 shards. **≈0.19–0.21% of a decode step** (`derived`).

XLA compiles each `lax.cond` branch into its own computation **per call site**, so CSE cannot span
them. D1-lite keeps the `cond` and the `_fast_path` verbatim and hoists into the entry computation
everything that is a pure function of `(seq_lens, cu_q_lens, distribution)`: `_effective_row_lengths`
(`streamindex_topk.py:482-506`), the `row_lengths` pad (`sparsecore_topk.py:322`), `decode_batch_end`
(`:755-756`), and the three scalar-prefetch operands (`:694-701`).

21-layer chain at production, per layer:

| arm | full µs | residual | Δresidual | fast-path regime (`L21_ctx2048`) |
| --- | ---: | ---: | ---: | ---: |
| `inc` | 170.241 | 26.069 | — | 7.238 |
| **`dl`** | **163.216** | 19.087 | **−6.982** | **8.010 — 1.11×** |
| `d1` (full deletion) | 121.361 | 9.089 | −16.980 | 37.705 — **5.21×** |

**41.1% of D1's win, and it turns a 5.21× regression into +10.7%.** That is the whole point: D1-lite
is the version that does not detonate when the fast path fires.

**The caveat:** the "stays at the incumbent's 7.7 µs" pass condition holds at 21 layers and **fails
at a single call** (1.65×). Production is 21 layers, so that is the depth that decides — but
D1-lite's safety is a CSE property, not a structural one.

**Headroom left on it:** the prefetch hoist returned nothing because **XLA sinks the constants back
into the branches** — `prefetch_c3` is 63 rows in both `inc` and `dl`, against 3 in `d1`; the metadata
hoist is half-done for the same reason (294 → 216 rows, not → 34). Blocking the sink with explicit
`cond` operands or an `optimization_barrier` puts a `modelled` ~−2.2 µs/layer more on the table,
taking recovery from 41% to roughly 54%. Untested.

Correctness: 42 grid tests across 8 shapes × 3 hoist modes including two fast-path-firing shapes, the
checkout's own 16 kernel tests, and `np.array_equal` against a separately-constructed incumbent at
ctx 1024/2048/8704 — `n_differing=0` everywhere.

Drop the `dlc` variant (hoist wrapped in an inner `cond`): it did not CSE and is beaten by `dl` both
at production and, ironically, in the fast-path regime it was designed for.

### 5.3 Option B — runner-side chunking, G=4 · **runner** · SHIP after one re-measurement

`j-b86c8fce-fd3d-4b63-9a24`. **≈0.155% of a prefill step** (`derived`, from −29.6% of the 3.2614
ms/step stage A + stage B).

| arm | full caller µs | Δ vs `bG1` | caller − KV-scatter | Δ |
| --- | ---: | ---: | ---: | ---: |
| `bG1` | 1915.0 | — | 916.4 | — |
| `bG2` | 1809.7 | −5.50% | 810.5 | −11.56% |
| **`bG4`** | **1712.2** | **−10.59%** | **714.1** | **−22.08%** |
| `bG8` | 1705.7 | −10.93% | 706.9 | −22.86% |

**G=4 is the knee** — G=8 buys 0.3 pp for +0.7 s of compile. Mechanism confirmed rather than
assumed: stage B on TC falls 434.5 → 235.6 µs as the TensorCore stops idling through SparseCore,
stage A *rises* 197.8 → 209.9 from per-chunk recompute, and the SparseCore plane rises 432 → 455 and
is never added to anything.

**The caller change is five lines** and the host cost is nil: dispatch-only time is **flat at
28.9–35.2 µs with no trend in G**, because all G chunks compile into one XLA executable — one host
dispatch regardless of G. `chunk_policy` is Python on two shapes at trace time only.

**`option-b-pipeline/RESULTS.md` §7's second integration problem does not exist.** Only `G` must be
static, and `G` is a function of two *shapes* vLLM already keys its compile cache on — **no plan
crosses the host boundary and zero new compilation axes are added**. The per-chunk `seq_lens` /
`cu_q_lens` rewrite is masked arithmetic on already-traced arrays and handles ragged boundaries the
pseudo-sequence trick could not express. A static `chunk_size >= max_num_seqs` guard proves all
decode tokens sit in chunk 0, so chunks i>0 drop their decode launches at trace time (`2+G` launches,
not `3G`) and doubles as the phase gate fencing off Option C's decode regression.

**Composed with Option C**: B×C at G=4 is **−26.18%** of the prefill op (`j-caf973f4-163d-42e0-89e6`),
**sub-additive** at 83.8% of the additive prediction. The mechanism is structural — B's overlap
ceiling is `min(t_a,t_b)/G × (G−1)` and `t_a < t_b`, so C shortens the very thing that sets the
ceiling. Do not quote −24.49% + −6.77%. C's win does **not** decay with G (−50.94 / −50.12 / −51.25 µs
at G=1/2/4, non-monotonic, so the worry is rejected on shape, not on a noise bound).

**Why it needs one more measurement before landing:** its `bG1` baseline contains D1 and the `bkv_p`
ladder, both now withdrawn. The Option B *deltas* are unaffected — both arms share the module, so it
cancels — but its "−13.36% vs `inc`" column describes a configuration nobody proposes to ship. Re-run
the sweep on the C-only base before quoting any vs-incumbent number.

Correctness: `j-46ad1b70-e536-4e25-902a`, run before any timing. 17 arms across 5 configs, all
`bad_rows=0`, `cache_identical=True`, including `mix_ragged_s64` where chunk boundaries fall
mid-sequence and 7,626 `-1` fills reproduce exactly.

### 5.4 The KV-cache relayout — **cache storage format** · the largest item, and NOT ready

`j-8ddefa9d-a60d-412b-b4c9` (HLO), `j-3391639d-1e5b-4c3d-84a8` (the landable `split` lowering).

**The problem, measured in production.** `mla_attention_op.py:104`'s
`cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)` compiles at prefill into:

```
%reshape.21215 = u8[424960,256]{1,0:T(8,128)(4,1)} reshape(u8[415,256,4,256]{3,2,1,0:T(4,128)(4,1)} %param.244)
%scatter_fusion = u8[424960,256]{1,0:T(8,128)(4,1)} fusion(%reshape.21215, s32[1024] ..., u8[1024,256] ...)
%reshape.21217 = u8[415,256,4,256]{3,2,1,0:T(4,128)(4,1)} reshape(%scatter_fusion)
```

349.03 + 135.11 + 290.43 = **774.6 µs per write**, ×22 = **17.04 ms/step**, of which the two reshapes
alone are **13.78 ms/step = 2.213%** of the prefill step and **66% of the whole data-formatting
category**. It relayouts the entire **108.8 MB** indexer cache, twice, to write 1024 rows — a pure
round trip, same shape and layout in and out, with **no consumer of the intermediate**.

**It is not a harness artifact.** The forward reshape's operand is `%param.244`, a program
*parameter* — the cache arrives from vLLM's cache engine already in `T(4,128)`.

**The cause, with two controls in the same compiled program.** Under `T(4,128)` tiling a trailing dim
of **128** flattens by free `bitcast`; **256** does not. The sibling 128-wide caches scatter with no
relayout (127.5 and 124.35 µs), and **decode has no reshapes either** — 8.87 µs straight into the
tiled 4-D cache at 64 tokens.

**The fix works. The pass condition was the HLO and it passed:**

```
%bitcast = u8[849920,128]{1,0:T(8,128)(4,1)} bitcast(%cache_kv.1)   ... "size":"0"
ROOT %bitcast.71 = u8[415,256,8,128]{3,2,1,0:T(8,128)(4,1)} bitcast(%scatter_fusion)
```

Cache-`reshape` count on the 108.8 MB buffer: base 2 → **`v128` 0**.

| config | arm | full µs | Δfull | stage A | Δ stage A | triad | Δ triad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill 1024 | base | 1730.767 | — | 247.798 | — | 800.475 | — |
| prefill 1024 | **`v128`** | 1217.568 | **−513.199** | 254.623 | +6.824 | 280.833 | −519.642 |
| decode 64 | base | 182.484 | — | 111.550 | — | 15.365 | — |
| decode 64 | **`v128`** | 208.708 | **+26.224** | 130.990 | **+19.440** | 21.909 | +6.544 |

The base arm reproduces production to 2.5% at prefill (757.01 vs 774.57 µs) and 1.6% at decode
(8.772 vs 8.87). Correctness is clean: `failed_checks=0`, `np.array_equal` True over all
108,789,760 bytes read back through the original view, and the production `StreamIdxTC` kernel
returns identical top-k from a `v128`-written cache.

**Why it does not ship as prototyped.** It buys prefill at decode's expense:

- prefill: −513.199 µs × 22 writes = **−11.29 ms/step ≈ −1.78%** of a prefill step
- decode: +26.224 µs × 21 writes = **+0.55 ms/step ≈ +0.79%** of a decode step

**Break-even is ≈174 output tokens** for an 8704-token prompt (`derived`: 8.5 prefill steps ×
11.29 ms ÷ 0.55 ms). Most serving workloads generate more than that, so **as prototyped this is a net
regression.** The prefill number is also not the 1.31% the prototype reported — that used the
double-counted TC+SC denominator of §3.

**But the decode cost is not the relayout removal.** It is the kernel de-interleaving a 128-minor
cache in `load_bkv` (+19.44 µs of the +26.22). Three lowerings were tried: `index` costs **+1060 µs**
at decode, `stride` **does not compile** (`MosaicError: not implemented: Strided load with non 32-bit
data`), and `split` — the packed-sublane approach, exploiting that `T(8,128)(4,1)` packs 4 i8 rows per
physical sublane so rows 0-3 and 4-7 are two whole-sublane reads — costs **+26.2 µs**, cutting the
regression **54×**. One more pass on the de-interleave is what stands between this and the largest
win on the board.

Also note this is a **storage-format** change, not a view: byte order is identical (verified
elementwise, not assumed) but tile height comes from the second-minor dim, 4 vs 8. It cannot be
bolted on at the call site — the `v128rt` arm that tries is the worst arm at both configs. Landing it
touches `streamindex_topk.py`'s `load_bkv`, the cache allocation site (reached via the MLA backend's
`get_kv_cache_shape` — **not chased down**), `mla_attention_op.py:104`, and **every other reader or
writer of that cache**, which was **not surveyed**. That survey is the main landing risk.

**Route (b)** — doing the write inside the Pallas kernel — is feasible and strictly more general, with
a ceiling of the full 757.8 µs triad (≈1.5× route (a)). Its cost is 1024 × 256-byte DMAs per prefill
step, far below efficient DMA size, recoverable only by a run-length pass over page-sorted tokens.
Its advantage: against an unchanged `[415,256,4,256]` cache it changes nothing any other reader sees.
Worth reopening if the storage-format survey turns out expensive.

---

## 6. What does not ship

| option | verdict | why |
| --- | --- | --- |
| **`bkv_p` ladder** (was "candidate 0") | **withdrawn as sized** | The block is the quantum of overfetch and `bkv_p` is compile-time static. Sized to `max_model_len=9216` it returns 9, so **every** request scans 9216 positions: at ctx 1024 that is 4.5× the incumbent's scan and **+98% on the scoring path**. The win lives only in `6144 < ctx ≤ 9216`. The ~19% per-position efficiency gain is real; the **sizing rule** is what is broken. |
| **D1** (delete the `cond`) | **withdrawn, superseded by D1-lite** | Deletes a branch that fires. At ctx 1024/2048 with k=2048 the incumbent takes the fast path in **7.7 µs with no stage A at all**; D1 runs the full scoring path, **7.44×/7.60× slower**, for **4.67%** bought at production. The earlier draft's "provably never fires" was wrong. |
| **the merged op** (`bkv_p` + D1 + C) | **withdrawn** | Composed, it is **14.7× slower** than the incumbent at ctx 1024 (113.9 vs 7.7 µs). Its headline −20.24% was also inflated: ~5.5 pp of it is scan-extent reduction unique to `ctx=8704 / mml=9216`, the one grid point where the merged arm scans **10% fewer positions** (ratio 0.9000 there, exactly 1.0000 at the other 16). At buckets whose context divides evenly the honest expectation was −13% to −16%. |
| **A** — 128-byte K record | reject | −15.30 µs against its own −13% floor of 18.64 µs, **17.9% short**. 94.3% of the fetch is already hidden. |
| **E** — `_pick_partition` | reject | Correct (1,100 arms, 0 mismatches) but lands on a bucket `min_token_size=16` makes unreachable. |
| **F** — bf16 operands | reject | Premise false: both matmul operands are already `float8_e4m3fn`. Measured **+4.63% prefill / +9.80% decode**. |

**Both withdrawn changes are recoverable, and by the same mechanism.** `bkv_p` needs to be sized to
the *actual request length* rather than to the bucket ceiling, and the runner is what knows that at
dispatch time. That is the same runner-change budget Option B already needs. **The runner change is
turning out to be the enabler for most of what is left** — close to the opposite of the earlier
draft's conclusion that it was not worth the blast radius.

---

## 7. Total, and the honest reading

### 7.1 The composed candidate, measured as one object

Everything in §5.1–§5.2 was measured as a separate arm and then added up. `variants/composed/`
measures the two kernel changes **composed into one module** — incumbent + Option C under its
prefill/mixed gate + D1-lite's hoist — against the incumbent in a single job, at the op level.

- **Job `j-bf290089-a5b2-417c-a900`**, `simple-run-command`, tpu7x `2x2x1`, four shards `rc=0`.
- **Arms:** `inc` (incumbent), `ctl` (incumbent re-imported through the composed module's own entry
  point — an identity arm that isolates plumbing cost), `c`, `dl`, `c_dl`.
- **`µs/layer`** below is device self time for one `streamindex_topk` call, TensorCore plane only,
  from `hlo_stats` at `--top-n 4000` with `rows_kept == rows_total` verified. SparseCore rows
  reported separately and never summed (§3). No host floor subtracted; no wall clock reported.
- **`L21`** = the 21-layer chain, i.e. GLM-5.2's real indexer depth, which is what lets cross-layer
  CSE act. n=3 within-job reps.

| metric, `L21_dec_b64_ctx8704` | inc | `c_dl` | Δ | Δ% |
| --- | ---: | ---: | ---: | ---: |
| **full op** | 170.189 | 163.522 | **−6.667** | **−3.92%** |
| stage A | 111.538 | 111.773 | +0.235 | +0.21% |
| stage B | 32.625 | 32.623 | −0.002 | −0.01% |
| residual (neither kernel) | 26.026 | 19.126 | **−6.900** | **−26.51%** |

The entire decode win is D1-lite's residual saving (Δ`dl` −6.806); Option C contributes +0.411, i.e.
nothing, which is what its gate is supposed to do at decode. Plumbing is Δ`ctl` +0.083 (+0.05%).

At `L21_pre_q1024_ctx16384`, inc 1199.585 → **Δ`c_dl` −84.236 (−7.02%)**, of which Δ`c` is −81.584:
Option C takes **−20.4% off stage A**, stable at 19.3–20.8% across the grid.

**Composition is additive to within the job's resolution.** The genuine three-way residual
`R = Δ(c_dl) − [Δ(c) + Δ(dl)]` is **−0.272 µs/layer, −0.160% of inc** — just outside the quadrature
error bar (±0.260, by 5%) and inside the widest single-arm spread (0.435). Read it as 104.3% of the
sum of parts with the 4.3% unresolvable, not as a synergy. The headline Δ is 15× the widest spread.

Across the 17 single-call configs **nothing regresses on `c_dl`** (−0.25% to −7.27%). Two things
only the 21-layer chain shows: `dl` **alone costs** time at one layer on prefill (+0.57 to +3.23 µs)
and turns negative only at 21, and the single-call twin of the headline under-reports the composed
decode win by 3× (−1.32% vs −3.92%).

**Correctness: `rows_compared=34368, rows_differing=0, failures=0`** — `np.array_equal` against a
separately constructed incumbent device call, every arm × 10 configs, covering six MIXED prefill
shapes (including a partial `bq` and a non-multiple `q`), two decode points and both fast-path
contexts. The gate itself was verified three independent ways: runtime `LAYOUTS` map, HLO census
(`c_dl` = exactly 21×4 MIXED + 21×8 DECODE head-major launches, `inc` = 0 of 252), and hand-grep of
the optimized HLO.

### 7.2 The totals

| change | layer | phase | ~% of that step | status |
| --- | --- | --- | ---: | --- |
| KV relayout removal | cache format | prefill | **−1.78%** | **not ready** — costs +0.79% of a decode step |
| Option B, G=4 | runner | prefill | −0.155% | ship after re-measuring on the `c_dl` base |
| **`c_dl` composed** | kernel | **decode** | **−0.2015%** | **measured composed**, §7.1 |
| — of which D1-lite | | | −0.206% | |
| — of which Option C | | | +0.012% | gate working as intended |
| **`c_dl` composed** | kernel | **prefill** | **≈ −0.054%** | **derived by component**, see below |
| — of which Option C | | | −0.039 to −0.042% | |
| — of which D1-lite | | | −0.013% | |
| **shippable today** | | **decode** | **≈ −0.20%** | |
| **shippable today** | | **prefill** | **≈ −0.21%** | with Option B |

The decode row is the one arithmetic step from a composed measurement: −6.667 µs/layer × 21 layers
= −0.140 ms/step against the 69.492 ms/step TC-plane denominator (§4). It reproduces the −0.19 to
−0.21% previously derived from separate arms.

**The prefill row is *not* the composed −7.02%, and the difference matters.** That figure was taken
at `q1024/ctx16384`, a far heavier prefill shape than the profiled step, so scaling it directly
would overstate the model-level saving by roughly 5×. The row above instead applies the composed
run's *stable ratio* — Option C removes 19.3–20.8% of stage A — to the profiled stage A of
1.2491 ms/step. Both routes are `derived`; the ratio route is the defensible one.

All rows remain `derived` — op-level measurements times profile shares. **None is an observed model
speedup**, and no end-to-end benchmark has been run.

### 7.3 The composed run also found a regression

At **ctx ≤ 2048 with k=2048** — the fast-path regime, where the early-exit branch fires and *no
kernel runs at all* — the composed candidate is **1.61× the incumbent**:

| config | inc | `c` | `dl` | **`c_dl`** | multiplier |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L21_dec_b64_ctx1024` | 7.115 | 10.619 | 8.001 | **11.479** | **1.614×** |
| `L21_dec_b64_ctx2048` | 7.227 | 10.645 | 7.988 | **11.486** | **1.589×** |

D1-lite's +0.89 is the expected unconditional `metadata` computation and was already disclosed in
§5.2. **The other +3.5 µs/layer is Option C, whose kernel never executes here** — this is new, and it
contradicts §5.1's claim that the gate makes Option C free outside prefill.

Traced: it is entirely the `conditional` HLO's own self time, `Bound by = HBM`. Per-instance,
`inc`/`ctl`/`dl` have **1 of 21** conditionals at ~8.6 µs and the rest at ~0.3 µs; `c`/`c_dl` have
**10 of 21** at ~8.6 µs, alternating, identically at both contexts. `c` has the *same* HLO
instruction count as `inc` (8476), so the extra code is inside the Mosaic call, not around it.
**Hypothesis, unconfirmed:** IMEM pressure from the larger head-major kernel body forcing a program
fetch on alternate layers.

Absolute cost is ~92 µs per 21-layer pass, **≈ +0.13% of a decode step** — small, and it applies
only below ctx 2048. But it is a regression in a real serving regime, it was invisible until the
changes were composed, and it may behave differently inside a program that also carries attention,
MLP and MoE. It does not block shipping `c_dl` at production context; it does mean **Option C needs
the IMEM hypothesis tested before it can be called free at decode**.

**This is a small number, and it should be reported as one.** The op is 4.299% of a decode step; the
changes that survive take about a twentieth of it. The exercise's most valuable output is not the
0.2% — it is the map in §4, which says where the next hour goes.

**Three things this exercise got wrong, recorded because the next session will be tempted by the same
ones:**

1. **A denominator measured at the wrong shape.** `report.md`'s 32.8%-of-decode figure is AgentX
   (`max_model_len=262144`, `n=65536`, TP=16), a ~9× mismatch with the nightly shape targeted here.
   It reproduces faithfully and mispredicts production.
2. **A denominator summed across two planes.** §3. It made the op look 12% smaller than it is.
3. **A gate that could not fail.** The additivity check that justified composing three changes was
   algebraically an identity on the axis that mattered. It took a separately-constructed `c` arm to
   test what it claimed to have tested.

**Where the next hour goes.** `sc_dsa_gather` — the gather that consumes the indices this op produces
— is **6.04% of a decode step and 8.33% of a prefill step (51.86 ms/step, 2.5× this entire op)**. The
DSA complex around it is 10.34% / 11.66%. Nothing in this exercise touched it. Stage B also has **no
modelled floor at all** — `hardware-facts.md` carries no measured SparseCore fact — and it is 53–60%
of the prefill op and blocks the TensorCore for 2× its own execution time.

---

## 8. Risks and gaps

1. **No end-to-end benchmark.** Every figure is a microbenchmark or a profile read.
2. **Effective n=1 on the profile's prefill numbers.** One model step per capture; two captures agree
   to 0.03% on the relayout, and 40.4% of the step is collective time, so part of any prefill saving
   may be absorbed rather than realised.
3. **The relayout fix changes a storage format and its other readers were not surveyed.** This is the
   main landing risk on the largest item, and it is unquantified.
4. **The `bkv_p` and D1 withdrawals rest on a grid whose short-context points are synthetic.** No
   request-length distribution from a real GLM-5.2 server was available; the exposure is arithmetic,
   not observed traffic.
5. **D1-lite's fast-path safety is a CSE property**, holding at 21 layers and failing at 1. The
   composed run confirms this from the other direction: `dl` alone *costs* +0.57 to +3.23 µs at one
   prefill layer and only turns negative at 21 (§7.1).
5b. **The +3.5 µs/layer Option C costs at ctx ≤ 2048 is unexplained.** The IMEM-pressure hypothesis
   in §7.3 is inferred from a per-conditional self-time distribution, not measured. Until it is
   tested — the cheap route is `/xprof-llo` on the two kernel bodies, comparing program size — the
   size of this cost in a full model program (which carries attention, MLP and MoE alongside) is
   unknown, and it could be larger there rather than smaller.
5c. **Option B has never been composed with `c_dl`.** Its +0.155% is a standalone measurement on a
   different base, and its chunked launch touches the same per-chunk metadata D1-lite hoists, so
   the two are the most likely pair in this exercise to be non-additive. `variants/composed/` was
   cut short of it deliberately; the B×C×D1 measurement is the obvious next job.
6. **Correctness everywhere is index-identity against the incumbent, not against ground truth.** It
   inherits any error the incumbent has. An absolute recall-vs-f64 gate was shown to fail forever on
   correct code (`results.md` §11).
7. **A jax version skew** — correctness ran under 0.10.2, benchmarks under 0.11.0. No job ran both.
8. **`test_streamindex_topk_early_exit_*` were never run** in this exercise. D1-lite keeps the branch,
   so they should still pass, but that is an inference.
9. **`/critical-review` has run only on the superseded 2026-09-02 draft.** Its eleven findings are
   incorporated above; this version has not been reviewed.
