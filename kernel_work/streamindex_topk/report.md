# Step 1 — `streamindex_topk`, the GLM-5.2 DSA lightning indexer

**Block:** `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py`, the Pallas scoring kernel
(`kernel()` at L65) and the JAX glue around it (`_common_path` at L740).
**Checkout:** `/home/annyan_google_com/workspaces/vllm-torchtpu`, branch `annyan/streamidx-scores`,
off `origin/main` at `13617694`.
**Model / deployment:** GLM-5.2-FP8 (`GlmMoeDsaForCausalLM`), `glm-5.2-fp8-dp16-ep.sh` —
TP=1, DP=16, EP=16 on a 2-node `2x2x2` tpu7x slice (16 TensorCores).
**Image:** `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.

**Sign convention, binding on this document and everything downstream of it:** every percentage is
a change in **time**, `(new − old) / old`. Negative is faster. Signed percentages appear only in
tables, under a column named `Δ time %`; prose says the direction in words.

---

## Vocabulary

Four terms in this document collide with a term of art elsewhere in the stack. Each is defined at
first use below, and repeated here because the collisions are the expensive kind — a reader does
not experience them as undefined, they read them confidently in the other sense.

| Term | Here it means | It is **not** |
| --- | --- | --- |
| **compressed KV position** | one row of the *indexer* KV cache. GLM-5.2 runs `compression_ratio=1`, so one compressed position is one token — but the code divides by it everywhere and a future model may not | a token of the main MLA KV cache, which is a separate cache with a separate page size |
| **page** | a `page_size = 1024`-row unit of the paged *indexer* KV cache (`attention.py:1018`) | a Pallas grid block, and not vLLM's KV block for the main MLA cache |
| **bkv block** | `bkv_p` pages' worth of compressed KV positions handled in one trip of the kernel's inner loop. This is the Pallas-level tile | a page; at `bkv_p = 2` one bkv block is two pages |
| **chunk** | vLLM chunked prefill's `max_num_batched_tokens` slice of a prompt | the chunkwise-parallel "chunk" of linear-attention kernels |

Two more, used without collision but worth pinning:

- **the selection half** — `sparsecore_topk(...)` at L831, which reduces the scores matrix to `k`
  indices on the SparseCore.
- **the scoring half** — everything before it: the Pallas kernel that fills the scores matrix.
  This exercise is about the scoring half. See "Why the scoring half" below.

---

## What the block does

For every query token, score every resident compressed KV position with a 32-head, 128-dim
"lightning indexer", then keep the best `k = 2048`. Sparse MLA then attends only to those 2048.

```
q            f8e4m3fn[num_tokens, 32, 128]      per-token indexer queries
indexer_weights  f32[num_tokens, 32]            per-head scalar weights
cache_kv     u8[num_pages, 256, 4, 256]         paged fp8 indexer KV, see "The cache record"
                                    ↓
scores       f32[num_tokens, num_sublanes_total, 128]     Pallas  (the scoring half)
                                    ↓
topk_idxs    i32[num_tokens, 2048]              sparsecore_topk  (the selection half)
```

Per position `s` and token `t`, the score is

```
score[t, s] = Σ_h  max(0, q[t,h] · kv[s]) · indexer_weights[t,h]
```

summed over all 32 indexer heads. There is exactly one key per compressed position — the indexer
has no KV-head axis, so the GQA head map in the repo's reference is the identity here.
`streamindex_topk.py:304` computes this as one `einsum` per bkv block, followed by a ReLU, a
per-head scale and a sum over the head axis.

## The cache record — 256 bytes moved, 129 used

`mla_attention_op.py:230-231` builds the record width as `head_dim = 128 + 128 // 128 = 129`, then
the lane axis is padded to a multiple of 128, giving **256 bytes per compressed KV position**:

| bytes | contents |
| --- | --- |
| 0–127 | the 128 fp8 (`e4m3`) key values |
| 128 | the single `e8m0` block scale for those 128 values |
| 129–255 | padding — never read by the unpacker (`streamindex_topk.py:265-268`), but transferred |

**Every byte of that padding crosses HBM.** This is a 1.98× read amplification on what is by far
the block's largest tensor, and it is recorded here as a fact about the shape, not yet as a
proposal — step 2 owns whether anything can be done about it and who owns the change.

---

## Shapes the block is really called at

Derived, not supplied. Every constant below has a `file:line` in
`instruments/shapes.py`; the table here is the summary and that file is the authority.

**Read off the checkpoint** (`gs://tpu-commons-ci/glm/GLM-5.2-FP8/config.json`):
`index_n_heads = 32`, `index_head_dim = 128`, `index_topk = 2048`, `num_hidden_layers = 78`.

**Read off the deployment** (`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh`):
`MAX_MODEL_LEN = 9216`, `MAX_NUM_BATCHED_TOKENS = 1024`, `MAX_NUM_SEQS = 64` per DP engine,
`ISL_OSL_CONFIGS = "8192:1024"`.

**Read off the call site** (`mla_attention_op.py`): `compression_ratio = 1` (:114),
`num_kv_pages_per_block = (2, 2, 2)` (:119), `num_queries_per_block = (1, 128, 128)` (:120),
`enable_early_exit = True` (:57, :121). Tuple order is `(DECODE, PREFILL, MIXED)`.

**Derived by the platform:** `page_size = 1024` (`attention.py:1018`), `kv_packing = 4` and
`page_size_per_kv_packing = 256` (`streamindex_topk.py:592-593`), `lkv_dim = 256` (above).

### The regimes the instruments sweep

Column definitions: `nbq` is the trip count of the outer query-block loop
(`streamindex_topk.py:418`), `nbkv` the trip count of the inner KV-block loop (:420) — so the
kernel does `nbq × nbkv` block iterations and **re-reads the whole KV `nbq` times**. `sublanes` is
`num_sublanes_total` (:603-605), the width of the scores buffer in units of 128 columns; it is
`align_to(pages_per_seq, bkv_p) × page_size / 128`, i.e. rounded up past `max_model_len`.
`scoresMB` is one f32 scores buffer, `num_tokens × sublanes × 128 × 4` bytes — the block's largest
intermediate.

| regime | phase | nseq | ntok | kv_len | max_model_len | pages/seq | nbq | nbkv | sublanes | scoresMB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | decode | 64 | 64 | 8192 | 9216 | 9 | 1 | 4 | 80 | 2.6 |
| `mixed_prod` | mixed | 64 | 1024 | 8192 | 9216 | 9 | 8 | 4 | 80 | 41.9 |
| `mixed_32k` | mixed | 12 | 1024 | 32768 | 33792 | 33 | 8 | 16 | 272 | 142.6 |
| `mixed_64k` | mixed | 6 | 1024 | 65536 | 66560 | 65 | 8 | 32 | 528 | 276.8 |
| `decode_64k` | decode | 6 | 6 | 65536 | 66560 | 65 | 1 | 32 | 528 | 1.6 |

The two `_prod` rows are the checked-in config. The three long-context rows exist because the
end-to-end A/B this exercise feeds is being run at long context — see "Why long context" below.

**`sublanes = 80` at `max_model_len = 9216` is a direct hit against the profile**, which shows the
decode scope emitting `f32[64,80,128]`. That agreement is the evidence that `instruments/shapes.py`
mirrors the shipped arithmetic rather than a plausible reconstruction of it.

### Only two of the three cases ever execute

`streamindex_topk` dispatches three Pallas invocations — DECODE batched (L757), DECODE remainder
(L775) and MIXED (L793) — selected by bands of the `distribution` argument. **The PREFILL-only band
is empty by construction.** `mla_attention_op.py:183` builds

```python
distribution = torch.stack([rd[0], rd[0], rd[2]])
```

so entry `[1]` equals entry `[0]` and the invocation spanning `start=decode_batch_end,
end=distribution[1]` covers zero sequences. Only `StreamIdxTC-d-*` and `StreamIdxTC-m-*` scopes can
appear in a GLM-5.2 profile, and only those two appear. `instruments/inputs.py` reproduces this
`distribution` shape rather than a symmetric one.

### Three things I had wrong before deriving them

Recorded because they are the errors this step exists to catch, and because two of them would have
silently redirected the design.

1. **Block sizes.** I first carried `bkv_p = 8, bq_sz = 32`, read from a checkout 123 commits
   behind `origin/main`. The truth is `(2, 2, 2)` and `(1, 128, 128)` at
   `mla_attention_op.py:119-120`, which matches the profiled scope names
   `StreamIdxTC-d-bq_1-bkvp_2` and `StreamIdxTC-m-bq_128-bkvp_2` exactly. The stale values would
   have put every floor and every VMEM budget in this document out by a factor of four.
2. **"The three cases are forced identical."** They are not; the prefill band is empty, per above.
3. **`layers/vllm/attention.py` is a caller.** It is not. It supplies the hardcoded
   `page_size = 1024` at `attention.py:1018` and nothing else. `mla_attention_op.py:107` is the
   only production call site.

The common cause was the stale checkout: `streamindex_topk.py` moved 397 lines across that gap via
`6142d5e7` (#724, "Replace `jax.lax.approx_max_topk` with sparse core topk") and `6a16997e` (#631,
the early-exit skip), and `sparsecore_topk.py` (457 lines) did not exist at all. Optimizing the old
tree would have A/B'd two versions of replaced code. Note also that `dc574bee`, the sparsecore-topk
campaign branch, is **not** an ancestor of `origin/main` — anything measured on it was measured on
code that has not shipped.

---

## Why the scoring half, and not the selection half

The selection half is closed by measurement. From the shared experiment log for this model,
before this exercise began:

- relaxing recall was inert at `k/n = 0.222` (`j-c716603f`);
- `lax.top_k` was 13.77% slower on prefill and 27.22% slower on decode, and `argpartition` was
  worse still (`j-e83d16c4`);
- the standing conclusion is "the whole top-k is at its floor", with an explicit *do not re-propose*.

The scoring Pallas kernel has never been touched by anyone. That asymmetry is why the block for
this exercise is the scoring half. (Those numbers are quoted from the log, not measured here; they
were taken at the production shape on this hardware and they are the reason for a routing decision,
not a component of any floor below.)

## Why long context

At the checked-in production shape the block is small: 2.69% of the decode step and roughly 1.02%
of the prefill step, against a measured end-to-end `spread_pct` of 2.77%. A 20% kernel win there is
about 0.5% end to end — roughly 190 runs per arm to resolve at 80% power, on a benchmark that costs
hours per run. The block is not measurable end to end at that shape, and saying so is part of the
result.

The indexer's cost is **O(resident KV positions)** while the sparse-MLA it feeds saturates at
`k = 2048` regardless of context length. So the indexer's share of the step grows with context and
nothing downstream of it does. That asymmetry is the whole reason the end-to-end A/B is being run
at long context rather than at the nightly shape.

One counterweight, carried forward rather than resolved: **decode** indexer cost is close to
invariant to advertised context length, because the resident tokens per rank are capped by HBM, not
by `max_model_len`. The growth is in prefill and in the mixed steps. `decode_64k` is in the sweep
to test exactly that claim rather than assume it.

---

## What the profile says, and what it does not

Source: the phased xprof capture at `optimization_work/glm-5-2-2026-09-05/xprof/`, rank 0, taken on
the checked-in production config. Per-call figures are `total_ns / count` over the profiled window.

### `decode_only` — the scope is there, and it is bimodal

`StreamIdxTC-d-bq_1-bkvp_2`, output `f32[64,80,128]`, 126 distinct HLO instructions, 735 calls,
49.106 ms total:

| population | instructions | calls | total ms | µs/call |
| --- | ---: | ---: | ---: | ---: |
| dominant | 105 | 630 | 48.145 | **76.42** |
| tail | 21 | 105 | 0.961 | 9.15 |

**76.42 µs/call is the baseline the microbenchmark's `decode_prod` row has to reproduce**, and the
cross-check that says whether `instruments/` is measuring the shipped kernel or something adjacent
to it.

### `prefill_only` — the scope is absent entirely

There is no `StreamIdxTC` anywhere in the `prefill_only` dump — not in `rank0_hlo.json` (29,661 op
rows) and not in `rank0_hlo_full.json` (202,230 op rows). The only ops attributed to
`jit(_streamidx_topk_jax)` in that phase are a `scatter_fusion` / `reshape` pair on
`u8[424960,256]`, ~291 µs each, HBM-bound at ~697 GiB/s, sourced to `mla_attention_op.py:104` —
that is the compressed-KV *write*, not the scoring kernel.

The explanation is almost certainly the early-exit gate. `streamindex_topk.py:859-865` wraps the
whole thing in

```python
jax.lax.cond(jnp.max(seq_lens) // compression_ratio <= k, _fast_path, _common_path, None)
```

and `_fast_path` (L840-856) contains **no Pallas call at all** — it constructs `arange` indices
under a causal clamp. With `k = 2048` and a 1024-token chunk budget, the first two chunks of any
prompt have `max(seq_lens) ≤ 2048` and take that branch. The profiled `prefill_only` steps appear
to be entirely such steps.

**This is an inference from an absence and is recorded as one.** It is not a measurement, and the
consequence is concrete: *the profile gives this exercise a decode number to validate against and
no prefill number at all.* The prefill and mixed baselines in the next section come from the
microbenchmark only, with nothing to cross-check them against.

Two further consequences worth carrying into step 2:

- `jnp.max(seq_lens)` is taken over the **whole batch**, so one long sequence puts every sequence
  in the batch on the common path. At the long-context shape the A/B targets, the gate is false at
  every step and the scoring kernel is always live.
- the ~291 µs `scatter_fusion` writing `u8[424960,256]` (108.8 MB) is real HBM traffic on the
  indexer path, and it is paying the same 256-vs-129 byte amplification the read side pays.

---
