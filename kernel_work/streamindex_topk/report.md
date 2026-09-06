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

### Amendment, after measuring: the selection half is bigger than this section assumed

The routing above is still right — the scoring kernel is untouched and has 55% headroom, and the
log's rejections stand. But it was written before the control arm produced a number, and the number
says the selection half is **not a small remainder**:

| regime | `full ms` | selection (`topk ctrl`) | selection share |
| --- | ---: | ---: | ---: |
| `decode_prod` | 0.16701 | 0.05562 | 33% |
| `decode_64k` | 0.20116 | 0.08615 | 43% |
| `mixed_prod` | 0.66911 | 0.39960 | **60%** |

Within a call the two are serialized — `sparsecore_topk` consumes the scoring kernel's output — so
these shares are Amdahl fractions on the call, and **the share grows with context**. At
`mixed_prod` a scoring-only win of 30% is 12% of the call, not 30%.

This is in tension with the long-context A/B and step 2 must carry it: the regime the end-to-end
comparison targets is the regime where a scoring-only win is most diluted. It does *not* reopen the
log's rejected proposals — those swapped the selection *algorithm*, and `sparsecore_topk` beat
them. It does mean "the whole top-k is at its floor" is a conclusion about algorithm choice, not a
measured floor for `sparsecore_topk` itself, and this exercise has not established one. No
SparseCore rate appears in the roofline below.

**A specific, testable suspicion for step 2.** `_common_path` builds `scores_to_reduce` at width
`num_sublanes_total × 128`, and `num_sublanes_total` (`streamindex_topk.py:603-605`) is derived from
**`max_model_len`, not from the batch's actual KV length**. Raising `max_model_len` to 64k therefore
widens that buffer to 67584 columns for every call, including short decodes. Whether
`sparsecore_topk` costs O(padded width) or O(`row_lengths`) decides whether that is free. The
measurement here cannot separate them — from `decode_prod` to `decode_64k` the width grew 6.6× and
the live row length grew 8× together, and the control went up 1.55×. A probe that holds
`max_model_len` at 64k and varies the actual KV length settles it. If the cost follows the padded
width, it is both an optimization and a **confound for the end-to-end A/B**, because standing up a
long-context config changes this cost for every request in the run.

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
## Correctness, and the control that makes it evidence

Job **`j-6d675a58-7ecd-4b44-9171`**, `Succeeded`, 4/4 shards rc=0. Image
`...vllm-torchtpu-annyan_google_com:13617694`, instruments uploaded from the working tree.

`instruments/test_correctness.py` scores the same inputs in float64 on the host — the vectorized
form of the repo test's own reference (`tests/kernels/test_dsv4_streamindex_topk.py:100-106`) — and
compares the kernel's returned index set against the oracle's ordering by **rank**, not by score.
Two numbers per row: `recall`, the fraction of returned indices that are genuinely in the oracle's
top-k; and `worst_rank`, the 1-indexed oracle rank of the worst index the kernel returned.

| shard | result | worst_rank | min_recall |
| --- | --- | ---: | ---: |
| `decode_prod`, 64 rows | **PASS** | 2109 | 0.99561 |
| `decode_64k`, 6 rows | **PASS** | 2105 | 0.99512 |
| `decode_prod`, **negative control** | **fired** — 64/64 rows bad | 8177 | 0.49512 |

The negative control zeroes the query rows of the second half of the 32 indexer heads, changing the
ranking without changing a shape, a dtype or a code path. It moved `worst_rank` by 3.9× and halved
recall, and the first failing row reported *1034 of 2048 indices outside the true top-2048*. **The
check has been shown to go red on a broken kernel**, which is what separates it from a check that
passes because it cannot fail (`references/hardware-facts.md`, "Instruments that lie").

Two honest notes on this evidence:

- **It covers decode only.** No mixed-phase correctness run exists yet.
- **The pass used almost all of its slack.** `worst_rank=2109` against an allowed 2112 is three
  ranks. `worst_rank` is an extreme-value statistic over 64 × 2048 drawn indices, so a bound set
  just above one observation of it is a bound on luck rather than on correctness; `RANK_SLACK` was
  therefore widened from 64 to 256 at commit `f77d6696`, with the argument written into the file,
  and `MIN_RECALL` deliberately left at 0.99 as the aggregate half of the criterion. The control
  still sits 3.5× outside the widened band. Loosening a threshold after watching it nearly fail is
  the wrong move by default and is flagged here so a reviewer can disagree with it.

The 0.4% of the selection that differs from float64 is rank-boundary churn, not error: every
disagreeing index sits within 61 ranks of a cut at position 2048 of 8192. Two ways of computing
nearly-equal scores will disagree about which side of a cut a near-tie falls on, and a top-k has no
notion of "close" to absorb it.

---

## Baseline

Same job, `j-6d675a58-7ecd-4b44-9171`. Median of 10 reps, single chip, single TensorCore, one
layer, one call. `enable_early_exit=false` unless stated.

Column definitions, since three of these are easy to misread:

| column | what it is |
| --- | --- |
| `full ms` | the whole `streamindex_topk` call — scoring kernel, the `-inf` scores fill, the glue, and `sparsecore_topk`. **This is the headline** and what the model pays. |
| `spread %` | `(max − min) / median` over the reps. A dispersion figure, not a confidence interval. |
| `topk ctrl ms` | `sparsecore_topk` alone, on a synthetic scores tensor of the same shape. A **control arm**, not a result — reported only as the subtrahend below. |
| `scoring ms by diff` | `full − topk ctrl`. An attribution, not a measurement. See the caveat under "What carries no price". |

| regime | `full ms` | `spread %` | `topk ctrl ms` | `scoring ms by diff` |
| --- | ---: | ---: | ---: | ---: |
| `decode_prod` | **0.16701** | 0.57 | 0.05562 | 0.11139 |
| `decode_64k` | **0.20116** | 0.58 | 0.08615 | 0.11501 |
| `decode_prod`, early-exit on | 0.17163 | 3.70 | 0.05692 | 0.11471 |

**The mixed rows from this job are void and are not reproduced here.** They ran against the
pre-`fb94c998` `inputs.py`, which passed `distribution[2] = max_num_seqs` for a mixed batch and so
made the MIXED band loop over 63 sequences with `q_len == 0`. Those are not free — `num_bq =
max(1, cdiv(0, bq_sz))` is 1, so each empty sequence still sweeps every KV block — and the rows
therefore moved 64× the KV they should have. Production passes the real batch sequence count
(`mla_attention_op.py:183`).

### Mixed, re-run on the corrected inputs

Job **`j-5f8f4a08-20b4-4fa0-bfc3`**, `Succeeded`, 4/4 shards rc=0. Same image. `inner` is the
number of calls dispatched before blocking, scaled down at large regimes to bound the outstanding
allocation (commit `f77d6696`); it is in the table because it differs across rows.

| regime | `inner` | `full ms` | `spread %` | `topk ctrl ms` | ctrl `spread %` | `scoring ms by diff` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mixed_prod` | 20 | **0.66911** | 2.84 | 0.39960 | 1.16 | 0.26951 |
| `mixed_prod`, early-exit on | 20 | 0.66696 | 1.93 | 0.39845 | 0.27 | 0.26852 |
| `mixed_32k` | 10 | **2.00536** | 1.52 | 1.21032 | **514.8** | 0.79504 ⚠ |
| `mixed_64k` | 5 | **4.76284** | 1.13 | 3.00969 | **133.1** | 1.75316 ⚠ |

The void rows were 3.28× / 1.55× / 1.19× the corrected ones at `mixed_prod` / `mixed_32k` /
`mixed_64k`. The inflation shrinks with KV length, as it should: at 64k the scores round trip
dominates and is untouched by a spurious KV re-read, and that batch has only 6 sequences to
duplicate rather than 64.

**The `full ms` spreads are fixed** — 68% and 231% became 1.52% and 1.13%. Two changes landed
between the sweeps though, the `distribution` fix and the `inner` bound, so **which one fixed the
spread is confounded and this exercise will not find out.** The numbers are usable either way.

**⚠ The two long-KV `scoring by difference` figures are not usable.** The control arm's own spread
is 514.8% and 133.1% there, so those differences are taken against a median that is not a stable
quantity. `mixed_prod`'s control is clean at 1.16% and its 0.26951 ms difference is trustworthy.
The instability tracks KV length rather than the harness — `mixed_prod` is clean on both arms — and
it is worst exactly where `inner` is smallest, so the control arm averages least where it is
noisiest. **`full ms` is unaffected and is what variants are ranked by**; the control is a
diagnostic, not the yardstick. Diagnosing it belongs to step 2, if step 2 needs the attribution.

### Mixed correctness

Same job, shard 3, `mixed_prod`, 1024 rows:

| | result | worst_rank | min_recall |
| --- | --- | ---: | ---: |
| `mixed_prod` | **PASS** | 2098 | 0.99512 |
| `mixed_prod`, **negative control** | **fired** — 1024/1024 rows bad | 8159 | 0.42578 |

Worth stating plainly: at `worst_rank=2098` this row would have passed under the **old**
`RANK_SLACK` of 64 as well (allowed 2112). The widening at `f77d6696` did not rescue a failing
measurement.

### The cross-check against the profile

The profile's `StreamIdxTC-d-bq_1-bkvp_2` scope is **76.42 µs/call**. The microbenchmark's
`scoring ms by difference` at `decode_prod` is **111.4 µs**. These are 1.46× apart and they are not
the same quantity; the gap is in the direction the differences predict:

- the profile figure is one HLO scope. The microbenchmark figure is everything in the call that is
  not `sparsecore_topk` — including the `jnp.full(..., -inf)` scores initialisation at
  `streamindex_topk.py:745`, the `_effective_row_lengths` computation at L823, the reshape, and the
  **two additional Pallas launches** `_common_path` always makes (the DECODE-remainder and MIXED
  bands, which at this regime span zero sequences but are still dispatched).
- the instruments give every sequence the same `kv_len`, and `streamindex_topk.py:275-276` sizes the
  KV loop by `max(kv_lens)`. A ragged production batch does strictly less work. The microbenchmark
  is the conservative side of that.

What makes the mirror credible is not the time, it is the shape: the profile reports the scoring
output as `f32[64,80,128]`, and `instruments/shapes.num_sublanes_total(9216)` independently derives
**80** from the page arithmetic. A reconstruction that was merely plausible would not land on 80.

**No claim is made that these two numbers agree.** They are the same order at the same shape, which
is what a cross-check can establish, and the residual is attributed above rather than explained
away.

### Early exit

`enable_early_exit` is `False` in the kernel's own default (`streamindex_topk.py:537`) and `True`
at the only production call site (`mla_attention_op.py:57`). At `decode_prod` the gate is false —
`kv_len 8192 // 1 > k 2048` — so turning it on measures the `lax.cond` and nothing else, and it
cost 2.77% more time. That is inside twice the spread of the arm it was measured on (3.70%) and is
**suggestive, not established**; it needs a dedicated measurement before anyone acts on it. It is
carried into step 2 because at the long-context shape the end-to-end A/B targets, the gate is false
at every step and this overhead is paid unconditionally.

---

## The modelled floor, and the verdict

`probes/roofline.py`. **Everything in this section is modelled, not measured**, and none of it goes
near the shared experiment log. Rates are tpu7x per TensorCore from `V7X.md`: HBM 3364.9 GB/s, MXU
892.8 TFLOP/s bf16 and 1785.6 TFLOP/s fp8.

Three quantities, and the formula matters because the layout of the table implies a different one:

- `hbm_id` — every byte the block *logically* needs, crossing HBM exactly once, divided by 3364.9
  GB/s. The floor a perfect implementation of this computation would hit whatever its loop shape.
- `hbm_st` — the bytes **this loop structure** moves. It differs in two places: `bq` is the outer
  loop and `bkv` the inner (`streamindex_topk.py:418/420`), so the KV is re-read `num_bq` times;
  and the scores buffer is `-inf`-filled before the kernel overwrites it.
- `mxu` — score FLOPs over peak, derated by `128 / min(128, bq_sz × 32)`. The einsum contracts
  `(b, bq_sz·32, 128) × (b, bkv_sz, 128)`, so at decode its M is 1×32 = 32 rows into a 128-row MXU
  tile and three quarters of every pass is discarded. Hence the 4× derate.

**`floor = max(hbm_id, mxu)`** — HBM assumed to overlap compute. That is optimistic, deliberately,
as is assuming the 512 ns per-DMA fixed cost is fully hidden by the kernel's double buffering
(`bkv_x2_ref`, `bq_x2_ref`). Both assumptions push the same way, which is what makes this a genuine
lower bound rather than a prediction.

| regime | kv MB | × re-read | scores MB | `hbm_id` µs | `hbm_st` µs | M | derate | `mxu` bf16 µs | `mxu` fp8 µs | **floor** bf16 | **floor** fp8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 134.2 | 1 | 2.6 | 41.7 | 42.5 | 32 | 4 | 19.2 | 9.6 | **41.7** | **41.7** |
| `mixed_prod` | 2.1 | 8 | 41.9 | 29.3 | 46.2 | 4096 | 1 | 77.0 | 38.5 | **77.0** | **38.5** |
| `mixed_32k` | 8.4 | 8 | 142.6 | 91.0 | 150.9 | 4096 | 1 | 307.9 | 153.9 | **307.9** | **153.9** |
| `mixed_64k` | 16.8 | 8 | 276.8 | 173.3 | 290.5 | 4096 | 1 | 615.8 | 307.9 | **615.8** | **307.9** |
| `decode_64k` | 100.7 | 1 | 1.6 | 30.9 | 31.4 | 32 | 4 | 14.4 | 7.2 | **30.9** | **30.9** |

Two precisions are carried because **which one the einsum gets is not settleable from the source.**
The operands are `float8_e4m3fn`; whether Mosaic emits a native fp8 MXU pass or promotes to bf16 is
a lowering question. It changes what `mixed_*` is *bound by* — MXU if bf16, HBM if fp8 — so it is
the first probe step 2 owes.

### The verdict: 55%, and the gate does not stop this

| | measured | modelled floor | achieved |
| --- | ---: | ---: | ---: |
| `decode_prod`, profile Pallas scope | 76.42 µs | 41.7 µs | **55%** |

55% is squarely inside the 50–80% band the skill calls a kernel's worth of headroom. **The roofline
gate passes and the exercise continues.**

Mixed is further from its floor than decode, not closer. The floor here prices only the TensorCore
side, so the denominator is `scoring ms by difference` rather than `full ms`:

| regime | scoring µs | floor bf16 | achieved | floor fp8 | achieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mixed_prod` | 269.5 | 77.0 | **29%** | 38.5 | **14%** |
| `mixed_32k` ⚠ | 795.0 | 307.9 | 39% | 153.9 | 19% |
| `mixed_64k` ⚠ | 1753.2 | 615.8 | 35% | 307.9 | 18% |

⚠ marks the two rows whose subtrahend has a 514.8% / 133.1% control spread behind it. Only
`mixed_prod` is settled, and at 29% of even the *pessimistic* bf16 floor it is well below decode's
55%. Whatever accounts for that gap is not in the model — the VPU tail is the first suspect and
carries no price at all (below).

Two caveats on that fraction, neither of which moves it out of the band:

- The scopes are not identical — `hbm_id` counts the scores read-back and the output write, which
  live outside the `StreamIdxTC` HLO scope. But `cache_kv` is 134.2 of the 141 MB, so restricting
  the floor to exactly the Pallas scope's traffic gives 40.7 µs and an achieved 53%. The conclusion
  is insensitive to the quibble.
- Against the microbenchmark's `full` of 167 µs the achieved fraction is 25%, but that denominator
  includes `sparsecore_topk`, which runs on the SparseCore and is not the TensorCore kernel's time
  to save.

### Where the headroom is

**Decode is HBM-bound on `cache_kv`, and half of that traffic is padding.** 134.2 MB of the 141 MB
ideal total is the compressed-KV read, and — per "The cache record" above — **129 of every 256
bytes in it is used**. `_fetch_bkv` (L177-194) DMAs whole pages, and `load_bkv` (L257-273) slices
the 128 value bytes and the one scale byte out in VMEM, *after* the padding has already crossed
HBM. Counting only the used bytes takes the modelled ideal floor from 41.7 µs to **21.9 µs**.

That is the largest single structural finding in step 1, and it comes with a constraint attached:
the 256-byte width is `align_to(align_to(128,128) + 1, 128)` and the cache is *written* in that
layout by the scatter at `mla_attention_op.py:104` — the same ~291 µs `scatter_fusion` the prefill
profile shows. Narrowing the read means changing the format, which means changing the writer.
**This is a finding, not a design; step 1 does not choose.**

The mixed regimes invert: at `mixed_64k` the scores buffer is 276.8 MB crossing HBM three times
against 16.8 MB of KV crossing eight, so the round trip through `f32` scores dominates and the KV
re-read does not.

---

## What carries no price

Named so a measured time exceeding the floor is not mistaken for a modelling failure.

- **The VPU.** After the einsum the kernel does a ReLU, a per-head multiply and a sum over the head
  axis on `num_tokens × 32 × kv_len` f32 elements — 1.7e7 at `decode_prod`, 2.7e8 at `mixed_prod`,
  which as VMEM traffic is larger than every HBM term in the table combined. **`V7X.md` carries no
  VPU peak**, so this is left unpriced rather than guessed. It is the largest gap in the model and
  the most likely reason a measured time exceeds the floor.
- **Relayouts.** No roofline term, and they do not appear in profiles either
  (`references/hardware-facts.md`).
- **`scoring ms by difference` is a lower bound, not an attribution.** Within one call the
  SparseCore top-k consumes the scoring kernel's output, so there is a data dependency and no
  overlap. But the microbenchmark dispatches `inner` calls back to back before blocking, so call
  *i*'s SparseCore work can overlap call *i+1*'s TensorCore work. Where it does, `full` per call is
  less than scoring + top-k and the difference **understates** the scoring half.
- **Single chip, no neighbours.** GLM-5.2 runs TP=1/DP=16/EP=16 and the indexer is replicated, not
  sharded, so single-chip is the right denominator for the kernel. It still cannot see HBM pressure
  from the other 77 layers, SparseCore contention with the rest of the step, or rank skew behind the
  step barrier.
- **Uniform `kv_len`.** The instruments give every sequence the same length; production batches are
  ragged and do less work. Conservative, and stated.
- **No prefill ground truth.** Per the previous section, the profile has no `StreamIdxTC` scope in
  `prefill_only`. The mixed baselines have nothing to cross-check against.

---

## What step 2 inherits

1. **The gate passes.** Decode is at 55% of its modelled floor and mixed at 29% — a kernel's worth
   of headroom at decode, and more than that at mixed.
2. **Decode's headroom is HBM, and it is padding** — 129 of 256 bytes. Acting on it crosses into
   the cache format and its writer.
3. **Mixed's headroom is the f32 scores round trip**, 276.8 MB × 3 at `mixed_64k`, not the KV.
4. **The selection half is 33–60% of the call and grows with context**, so a scoring-only win is
   diluted most in exactly the regime the end-to-end A/B targets. This does not reopen the log's
   rejected algorithm swaps; it does mean no floor has been established for `sparsecore_topk`.
5. **Probe: does the `float8_e4m3fn` einsum lower to a native fp8 MXU pass or promote to bf16?**
   It decides whether the mixed regimes are MXU- or HBM-bound, and the two answers point at
   different designs.
6. **Probe: does `sparsecore_topk` cost O(padded width) or O(`row_lengths`)?** `num_sublanes_total`
   comes from `max_model_len`, so the answer decides whether a long-context config taxes every
   short request — an optimization if yes, and a confound for the A/B either way.
7. **Probe: what does the VPU tail actually cost?** It is unpriced, it is large, and it is the
   leading suspect for mixed sitting at 29%.
8. **Carried, not designed:** the `lax.cond` at L859-865 is pure overhead at long context (2.77% at
   `decode_prod`, suggestive only — and 0.32% *faster* at `mixed_prod`, which is inside the spread
   and so says nothing), and `jnp.max(seq_lens)` over the whole batch means one long sequence puts
   every sequence on the common path.
9. **A harness defect to fix if the attribution is needed:** the control arm's 514.8% / 133.1%
   spread at long KV. `full ms` is clean and is the yardstick, so this does not block ranking.
