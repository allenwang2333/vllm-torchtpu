# The GLM-5.3-Flash kpool sparse indexer

Why the model has a 2048-token cap without it, what it computes, and the
decisions in the implementation that are silent when they are wrong.

See also [`glm5-3-flash.md`](glm5-3-flash.md) for the model as a whole and
[`../glm5-3-flash-kpool-pending-state.md`](../glm5-3-flash-kpool-pending-state.md)
for the pending-pool state and the alternatives rejected on the way to it.

## What kpool is

GLM-5.3-Flash's 11 full-attention layers are DSA: each query attends to a
selected subset of keys, chosen by a lightning indexer. Unlike DeepSeek-V3.2
and GLM-5.2, whose indexers select *tokens*, GLM-5.3-Flash's selects **pools**
of `index_kpool = 4` consecutive keys.

Per DSA layer, from the seven checkpoint tensors:

```
k     = LayerNorm(wk(h))                        # eps 1e-6, WITH bias
gate  = h @ index_kpool_compress_gate^T
pool_key[j, c] = Σ_m softmax_m(gate[4j+m, c] + ape[m, c]) · k[4j+m, c]
```

The softmax runs over the four **slots** of a pool, **per channel**, and `ape`
is `index_kpool_compress_ape[index_kpool, index_head_dim]`, indexed by the
token's slot within its pool.

Scoring and selection then run in pool space:

```
score[j] = Σ_heads weights_h · relu(q_h · pool_key[j] · softmax_scale)
selected = top_k(score, k = index_topk // index_kpool)      # 512 pools
```

and the selected pool ids expand back to raw token ids
`4j … 4j+3`, plus an **always-selected tail**: the `(p+1) % 4` most recent
tokens, which belong to a pool that is not complete yet.

Two correctness rules govern candidacy. A pool is a candidate for query
position `p` iff it is **complete** (all four of its tokens exist) **and** its
**last** token is visible — `4j + 3 <= p`. Together those give exactly
`floor((p+1)/4)` candidates, not `floor(p/4) + 1`.

## Why the dense substitution has a 2048-token cap

Without the indexer, the DSA layers run as dense causal MLA. That is *exact*
while the pooled top-k would have selected everything: `floor((p+1)/4)`
candidates against a budget of `2048/4 = 512` pools, so up to `index_topk`
tokens the selected union is precisely `[0, p]` — which is what dense causal
attention computes. Past it the reference attends sparsely and the dense build
does not, with no symptom except worse answers. Hence
`_check_context_within_dense_mla_bound`.

Removing that cap is the point of this feature.

## Why the existing TPU machinery does not just work

`streamindex_topk` already exists and already takes a `compression_ratio`, so
selecting over pools looks like a configuration change. Four things stop it
being one.

**The causal convention.** The kernel counted a compressed row visible as soon
as the query *entered* it (`pos // ratio + 1`); kpool needs the row's last
token to be visible. That is `causal_row_offset = index_kpool - 1`; see
[`../dsv4-streamindex-causal-off-by-one.md`](../dsv4-streamindex-causal-off-by-one.md),
because the same parameter turns out to document a pre-existing divergence in
DeepSeek-V4's own path.

**Page geometry.** `streamindex_topk`'s `page_size` is *compressed records per
page*, `block_size // compress_ratio`. DeepSeek-V4 runs `block_size = 1024`
and its tuned `num_kv_pages_per_block = (3, 2, 2)` is copied into new call
sites by reflex; GLM is hybrid and keeps vLLM's default `block_size = 16`, at
which that tuple is a trace-time `ValueError`. The value has to be derived
from `block_size` and `max_model_len`.

**Prefix contiguity.** The reference writes the always-selected tail at the
fixed column `index_kpool * select_k` regardless of how many pools were
actually valid, so a query with fewer candidates emits `-1` holes *before* the
tail. The sparse MLA kernel masks by column *position*, so those holes would
gather decoy rows at full weight. `expand_pools_with_tail` exists to make the
row prefix-contiguous.

**The early-exit fast path.** Its arithmetic is correct for GLM once
`causal_row_offset` is threaded through it. The problem is that it takes
`seq_lens` and `cu_q_lens` and *not the cache*: it emits the identity
selection without reading a single pooled record, and it triggers at
`max(seq_lens) <= 2048` — exactly the regime every test below the cap runs in.
With it on, the entire implementation could be wrong and everything would pass.
It is off, and asserted off with a `ValueError` rather than an `assert`,
because `python -O` strips asserts.

## Design decisions

**No new Pallas kernels.** The pooling is a 4-way per-channel softmax plus one
256-byte record per four tokens: ~1 MFLOP and 512 KB of scatters at an
8192-token chunk, against ~33 GFLOP of scoring in the same layer. A Pallas
kernel would be under 0.1% of layer time and would cost the CPU test rung,
which is where every silent-if-wrong detail here lives. The expensive stages
are already someone else's: pool-space top-k is `streamindex_topk`, the
attention is `sparse_mla_attention`.

**Position-addressed pending state, not a ring indexed by `pos % kpool`.** A
scheduler chunk can end mid-pool, so up to three leading `(k, gate)` rows have
to survive to the next step. The obvious depth is `index_kpool`; it is correct
at `num_speculative_tokens == 1` and corrupts at 2 and above, because a
rejected draft can push the write head past a row still needed for the carry.
The depth is derived —
`index_kpool + max_concurrent_batches * num_speculative_tokens` — with the
floor asserted against its own numbers, and a per-position tag on every ring
row so that a violation is a counter rather than a quiet retrieval regression.
The test suite runs the naive depth over the same schedules and asserts it
*fails*, because a suite that only shows the chosen depth passing cannot
distinguish "the depth is right" from "the schedule never stressed it".

**Write before score, in one op.** `kpool_write` runs first and its *output*
cache is what the top-k scores, so a pool completed by this step — including
one completed from a speculative draft — is visible to the scoring that
follows. Splitting the two across the torch boundary would let a graph break,
a donation copy or a reordering separate them.

**The 64-wide zero rope slab is on the sparse path only.** The sparse MLA
kernel and `get_sparse_kv_cache_specs` both hard-code a 64-wide rope tail; the
dense kernel needs 128. That makes `head_size` 576 on the sparse path and 640
on the dense one — and the two knobs (is the indexer built, which geometry)
are kept independent even though "64 iff sparse" looks like the same question.
Collapsing them makes the dense-baseline-at-the-sparse-layout configuration
inexpressible, which is exactly the arm an A/B needs in order not to be
confounded by cache geometry.

**The pooled cache does not subclass the DeepSeek cache classes.**
`tpu_runner.is_cache_for_ds_v4` is an `isinstance` check over them, and any
`True` there reroutes the whole model through `_initialize_ds_v4_kv_cache`,
which raises on GLM's first KDA layer. Inheriting `get_kv_cache_spec` for
convenience trades a 30-line branch for a model that cannot boot.

**Replicated, not head-sharded.** `streamindex_topk` sums the per-head ReLU'd
scores *inside* the Pallas body and takes the top-k of the sum, so a rank
holding 4 of 32 heads would select a different key set from its peers, over a
pooled cache they share.

## The acceptance bar, and why it is not "sets must match"

Exact set equality is the right check only where it is float-noise
independent. It is not, generally: an fp64 reconstruction agreeing with the
reference to 5e-7 still disagreed on **35%** of selected sets, because the
reference has exact ties and a tie broken the other way is a different set at
identical score.

So the suite uses exact set equality only where ties are excluded by
construction, and **selection regret** — the score mass lost relative to the
reference's choice — everywhere else. Where selections are compared as sets
rather than as ordered lists, that is also deliberate:
`streamindex_topk`'s two branches disagree on order by contract.

A related trap governs the fixture. At the checkpoint's `index_topk = 2048`
and a 192-token validation sequence, there are 48 candidate pools against a
512-pool budget, so the top-k returns all 48 whatever the scores are: an
implementation that pools with the wrong softmax, scores with the wrong query
scale or applies the wrong causal bound still selects the same set, and every
test is green. `build_tiny_fixture.py --index-topk 64` is the escape — 16
pools of budget against 48 candidates, so two thirds are genuinely dropped.

## Knobs

| variable | values | effect |
| --- | --- | --- |
| `VLLM_TORCHTPU_GLM5_KPOOL` | `0` / `1` (default on) | build the indexer, or run dense MLA under the 2048-token bound. A typo raises. |
| `VLLM_TORCHTPU_GLM5_MLA_GEOMETRY` | `auto` / `dense640` / `sparse576` | KV cache layout. `auto` follows the indexer. Indexer-on at `dense640` is refused. |
| `VLLM_TORCHTPU_GLM5_KPOOL_SCORE_BUDGET_MIB` | MiB, `0` disables | override the score-buffer refusal. |
| `VLLM_TORCHTPU_GLM5_KPOOL_CHECK_DIAGNOSTICS` | `1` | read the ring's diagnostic counters after every indexer call. Off by default; **set it in bring-up probes** — see below. A value other than `0`/`1` raises. |
| `VLLM_TORCHTPU_GLM5_ALLOW_LONG_CONTEXT` | `1` | run past the dense bound anyway, wrongly and loudly. |

## The pending-state counters, and why they are off by default

`kpool_write` always *computes* its counters — one int32 compare per carried
row, on device — so a tag violation, an out-of-range slot or two live requests
sharing a slot is always detected. What costs something is *reading* them: a
host round trip per DSA layer per step, eleven of them in every decode step at
the real checkpoint. `VLLM_TORCHTPU_GLM5_KPOOL_CHECK_DIAGNOSTICS=1` is what
turns the reading on, and **any probe of this feature should set it**. It is
what caught the two real defects of this kind: two live requests handed the
same pending-ring slot, and sixteen live requests handed the null sentinel
slot by a dummy run. Neither raises, neither produces a NaN, and both look
exactly like "this model does not retrieve well".

It is read once, at construction, and then fixed. That is deliberate. The flag
gates a host sync inside the traced forward, so a gate that can change its
answer between two steps changes the program the backend traces for those
steps, across all eleven DSA layers at once. A bounded budget — check the
first few steps of each layer's life, then stop — was tried and is worse than
useless: warmup runs one to two dummy forwards per token bucket before the
first real request, so the budget is spent entirely on compilation shapes, and
the branch then flips mid-run for a check that has already stopped happening.

## Evidence

Reproducible here: `tests/kernels/glm5_kpool/`,
`tests/models/vllm/glm5_next/`, and the float64 numpy oracle in
`tests/models/vllm/glm5_next/kpool_oracle.py`, which is itself checked against
the real `Glm5NextTextIndexer`.

On device, and not reproducible from this tree: a needle was recalled at 5944
tokens on the real 328 GB checkpoint at TP=8 — 2.9x the dense bound — with the
prompt spanning three prefill chunks, so the pending-pool carry was exercised
at two chunk boundaries.
