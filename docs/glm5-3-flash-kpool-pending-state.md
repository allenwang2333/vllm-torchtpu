# The kpool pending-pool state: where it lives, and why not elsewhere

GLM-5.3-Flash's kpool indexer completes a pooled record only when all
`index_kpool` of a pool's tokens have been seen. A scheduler chunk boundary
can fall inside a pool, so the up to `index_kpool - 1` leading `(k, gate)`
rows of an incomplete pool have to survive to the next step. This page records
where that state lives, why every paged alternative loses, and the depth
argument -- because the failure mode of getting it wrong is a finite pooled
key built from whatever position aliased the slot: no crash, no NaN, just
degraded long-context retrieval.

Read `docs/vllm-kv-cache-api-pin.md` first. `CircularBufferSpec` and
`SlidingWindowSpec.extra_retained_tokens` -- the two natural ways to express
this -- do not exist in vLLM `v0.27.0`, which is what the pod imports, and
`tests/models/vllm/test_vllm_kv_cache_api_pin.py` fails the build if either
name appears in `src/`.

The code this describes is `kernels/glm5_kpool/pool_write.py` (the ring
geometry and the read/write split), `models/vllm/glm5_next/indexer.py` (which
allocates the two tensors) and `runner/kpool_slots.py` (the slot pool that
addresses them).

---

## 1. The decision

**The pending-pool state is a pair of runner-lifetime device tensors owned by
each kpool indexer module, addressed by `(per-request slot, absolute position
mod pending_depth)`, and it is *not* a vLLM KV-cache group.**

Per DSA layer (11 target layers + the MTP draft head at layer 45 = 12):

```
pending_kv  : float32[num_slots, pending_depth, 2, index_head_dim]   # [.., .., (k, gate), 128]
pending_tag : int32  [num_slots, pending_depth]                      # absolute position, -1 = unwritten
```

with

```
num_slots     = max_num_reqs + 1                 # slot 0 is the null sentinel
pending_depth = index_kpool + max_rollback_tokens
max_rollback_tokens = vllm_config.max_concurrent_batches
                      * vllm_config.num_speculative_tokens
```

`VllmConfig.max_concurrent_batches` (`vllm/config/vllm.py:540` @ `v0.27.0`) is
`pp_size`, or `2` when `async_scheduling` is on at `pp_size <= 1`;
`VllmConfig.num_speculative_tokens` (`:564`) is `0` without a speculative
config. Both are properties of `VllmConfig`, so the depth is **derived from the
config, never chosen**.

> Citations into `src/vllm_torchtpu/runner/tpu_runner.py` below name
> **symbols, not line numbers**: it is a large and frequently edited file and
> line numbers into it rot within days.

Addressing:

* the **slot** comes from a req_id-keyed pool of size `max_num_reqs + 1`,
  reproducing `TpuModelRunner._init_mamba_slot_pool` /
  `TpuModelRunner._build_mamba_state_indices` with **stride 1** — one slot per
  request, no checkpoint group,
  because this state is not a recurrence. Slot 0 is never handed to a request;
  padded persistent-batch tails and OOB tokens point at it, so their writes
  cannot alias a live request. Keying on the stable `req_id` is what makes the
  slot follow the request through vLLM's `condense`/reorder.
* the **row** is `pos % pending_depth`, `pos` being the absolute token position.
* every read asserts `pending_tag[slot, pos % pending_depth] == pos`.

Reads and writes are split by source, and this is the one structural
difference from the obvious design:

* **write:** at the end of the op, scatter the trailing `pending_depth`
  positions of each request's in-step range.
* **read:** for a completing pool at position `p`, the four rows come from
  `where(q >= step_start_of_this_request, in_step_rows[...], pending_kv[...])`.
  Positions belonging to this step are read from the step's own `(k, gate)`
  tensors, never through the buffer.

Because a completing pool at `p` needs positions `4*floor(p/4) .. p` and
`p >= s` (the request's first position this step), the carry reads are always
in `[s-3, s-1]` — at most `index_kpool - 1` rows, and never a position the
current step wrote. That is why an in-step prefill chunk of 8192 tokens cannot
alias itself in a depth-7 ring: the chunk's own positions are never read out
of it.

### Bytes

`1024 B` of data + `4 B` of tag per (slot, ring row) per layer = **1028 B**.

| config | `pending_depth` | B / request / layer | KiB / request, 12 layers | whole buffer @ `max_num_reqs=64` |
| --- | --- | --- | --- | --- |
| no spec decode | 4 | 4 112 | 48.2 | 3.06 MiB |
| K=1, sync | 5 | 5 140 | 60.2 | 3.82 MiB |
| K=2, sync | 6 | 6 168 | 72.3 | 4.59 MiB |
| **K=3, sync** | **7** | **7 196** | **84.3** | **5.35 MiB** |
| K=3, async (`max_concurrent_batches=2`) | 10 | 10 280 | 120.5 | 7.65 MiB |
| K=7, sync | 11 | 11 308 | 132.5 | 8.41 MiB |

**These numbers do not move with `max_model_len`.** They are identical at 2048,
at 128 K and at 1 M. That is the whole point, and it is the axis on which every
paged candidate loses.

`float32`, not bf16, and that is load-bearing rather than lazy: the in-step
rows arrive in f32 (`pool_window` in `kernels/glm5_kpool/pool_math.py` takes
f32) and the carry rows come out of the buffer, so if the buffer stored bf16
the *same* pool would produce a different key depending on where the scheduler
happened to cut the chunk. f32 storage makes the two sources bit-identical and
makes the write path "chunked prefill at every residue equals single-shot" test an
**exact** equality rather than a tolerance. Halving to bf16 saves 2.7 MiB and
costs that property; do not.

### Where the memory is accounted

Allocated at model construction, i.e. before `determine_available_memory()`
runs its profile, so it is inside the measured resident footprint and the KV
pool shrinks by exactly its size automatically. No new term is needed in
`_hybrid_uniform_page_size_bytes` (`TpuModelRunner
._update_attention_page_size_padded`) and the KDA caches' invariance to
the whole change is untouched by this choice — which is not true of any of the rejected options.

---

## 2. Statelessness really is impossible — measured, not assumed

The pooled key is `Σ_j softmax_j(gate_j + ape_j) ⊙ k_j` over four raw tokens,
where

```python
k = self.k_norm(self.wk(hidden_states))                      # W_k : 4096 -> 128, then LayerNorm
gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)  # 4096 -> 128
```

(`modeling_glm5_next.py:798`, `:800`.) Both are functions of the 4096-dim
layer input `h`. The only per-token thing
the KV cache holds for a DSA layer is the MLA latent `kv_c = W_kva h`, 512-dim
and stored fp8 on the sparse path. Recovering `W_k h` from `W_kva h` requires
`rowspace(W_k) ⊆ rowspace(W_kva)`.

The best *possible* linear decoder `M = W_k W_kva^T (W_kva W_kva^T)^{-1}` was
computed for three random draws at the production geometry:

```
0. Statelessness: best-case linear reconstruction of W_k h from the 512-dim MLA latent
   trial 0: relative error 0.9353
   trial 1: relative error 0.9358
   trial 2: relative error 0.9353
   theory sqrt(1 - 512/4096) = 0.9354; and the gate needs another 128 rows out of the same 512-dim subspace.
```

93.5 % of the key is unrecoverable before quantization, before the LayerNorm
nonlinearity, and before you also need 128 more rows for the gate. Note the HF
reference agrees by construction: it caches `torch.cat([k, gate_scores,
valid_channel], -1)` — a 257-wide per-token indexer state for the **whole**
sequence (`modeling_glm5_next.py:803`, then
`past_key_values.update_indexer(packed_states, ...)` at `:811`). Our
design keeps `pending_depth` of those rows instead of `seq_len` of them.

---

## 3. Why every paged candidate loses, in one fact

`vllm/v1/core/kv_cache_utils.py:1013-1019` @ `v0.27.0`:

```python
def get_uniform_page_size(kv_cache_specs):
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()
```

Every KV-cache group must have the **same** `page_size_bytes`, and
`get_kv_cache_config_from_groups` (`:1361`) then sizes every group at the same
`num_blocks`. Upstream of it, `unify_kv_cache_spec_page_size` (`:1070-1130`)
tries to *force* that uniformity, and how it does so matters; the pooled
cache spec has to survive it, which
`TpuModelRunner._assert_pooled_page_size_survived_grouping` is the tripwire
for.

On the TPU side, the allocation loop in `TpuModelRunner._allocate_kv_cache`
allocates `num_blocks` rows for the `MLAAttentionSpec` branch and for the
generic `AttentionSpec` branch, and `_mamba_num_blocks` rows —
`max_num_reqs * _mamba_slot_stride + 1`, set by
`_maybe_set_compact_mamba_num_blocks_override` — **only** for `MambaSpec`.
Everything else falls through to a bare `raise NotImplementedError`.

So a pending-state group expressed as any attention-family spec is allocated
`num_blocks × uniform_page_size × 11 layers` of HBM to hold a live working set
of **three rows per request**. At GLM's geometry a `SlidingWindowMLASpec` page
is `16 × 1 × 1024 × 1 = 16384 B` against the MLA cache's real
`nope 16×512 + rope 16×128 = 10240 B`, so it is a *larger* full attention group
than the attention it is helping. Adding it, plus the pooled group the pooled-cache work is
already adding, roughly triples the paged footprint of an 11-DSA-layer model —
i.e. it pays for itself in lost context length, which is the quantity this
whole feature exists to increase.

Getting a per-request-sized allocation out of vLLM's allocator means
`MambaSpec`, and that is its own problem — §4, option (v).

---

## 4. The options, and why each rejected one was rejected

**(i) `SlidingWindowMLASpec`, window `index_kpool` — rejected, but not for
the obvious reason.** The free edge at `v0.27.0` is *not*
"`num_computed_tokens - sliding_window + 1`" unqualified.
`KVCacheManager.allocate_slots` (`kv_cache_manager.py:504-508`) calls
`remove_skipped_blocks(max(0, total_computed_tokens - request
.num_in_flight_tokens))`, and `SlidingWindowManager.get_num_skipped_tokens`
(`single_type_kv_cache_manager.py:1083`) then frees whole blocks below
`max(0, C_processed - W + 1)`. `num_in_flight_tokens` is incremented at
schedule time and drained in `update_from_output`
(`scheduler.py:1331`, `:1738`), so the edge is already rolled back by the
unsettled speculative window. With whole-block granularity the first retained
position is `<= C - W + 1`, and a completing pool needs `>= C - 3`, so `W = 4`
is *sufficient*, with zero slack, and `W = index_kpool +
num_speculative_tokens` has exactly the slack it needs. The design is
therefore not wrong; it is **unaffordable** (§3) and it hangs correctness on a
scheduler-internal expression that already changed shape once between
`v0.27.0` and `main` (`extra_retained_tokens` was added to it). It also
imports `SlidingWindowManager`'s prefix-cache machinery
(`find_longest_cache_hit`) into a cache that must never be prefix-reused, and
`_remove_blocks_in_range` (`:595-620`) replaces freed blocks with the shared
**null block**, whose all-zero contents decode as a legitimate `(k, gate)`
pair. Rejected.

**(ii) A homemade ring at depth `index_kpool` — rejected, and now measured.**
The sweep reproduces the obvious §2.2 counterexample exactly and then sweeps it:
at `num_speculative_tokens = 1` a depth-4 ring is green over all 120 schedules
tried, with zero wrong pooled records; at `num_speculative_tokens = 2` the same
depth produces **228 wrong pooled records over 270 schedules**, and 9 102 at
`K = 7`. GLM-5.3-Flash's checkpoint declares one nextn layer, so the
usual configuration is `num_speculative_tokens = 1`.
This is the "ships green, surfaces later as an unexplained acceptance-rate
drop" failure, confirmed rather than argued. A ring at a *derived* depth is
what §1 adopts; only the depth-`index_kpool` version is rejected.

**(iii) `MLAAttentionSpec` at `compress_ratio=1` holding `(k | gate)` —
rejected on bytes.** `head_size = 256`, bf16, `block_size = 16` gives
`8192 B` per block per layer = `512 B` per token per layer. At `max_model_len =
131072` that is `64 MiB` per request per layer, **768 MiB per request** across
12 layers; and per §3 it is allocated at `num_blocks` anyway, so it is a second
full attention group. Fine at 32 K, absurd at 128 K, and the obvious own
headline goal is "past 2048". Rejected.

**(iv) `CircularBufferSpec` / `SlidingWindowSpec.extra_retained_tokens` —
unavailable.** Absent at `v0.27.0`; see `docs/vllm-kv-cache-api-pin.md`. Both
names are forbidden in `src/` by
`tests/models/vllm/test_vllm_kv_cache_api_pin.py`. Not evaluated further.

**(v) A `MambaSpec`-typed pending cache — rejected on blast radius, and it was
the strongest contender.** `MambaSpec` at `v0.27.0` is the only spec the
allocator sizes per-request; it even carries `num_speculative_blocks`
(`kv_cache_interface.py:716`) for exactly this hazard, and the runner already
has an end-to-end branch for it. But `_update_attention_page_size_padded`
derives `_hybrid_uniform_page_size_bytes`,
`cache_config.mamba_page_size_padded`, `num_gpu_blocks_override` and
`_mamba_num_blocks` from `num_mamba = len([m for m in layers if
isinstance(m, MambaBase)])` and a hand-mirrored copy of vLLM's grouping
heuristic. A pending module that *is* a `MambaBase` moves `num_mamba` 34 → 45
and `num_mamba_groups` 4 → 5; a pending module that merely *emits* a
`MambaSpec` moves vLLM's grouping without moving the runner's mirror, which is
a silent `num_blocks` mismatch. Those four quantities are precisely what step
10(b) must hold byte-identical across its three configurations. Rejected: the
allocator support is not worth spending the one invariant the accounting
checks exist to
protect.

**(vi) Store a running `(numerator, denominator)` accumulator instead of the
raw rows — rejected on idempotence, at zero byte saving.** The pooled key is
`Σ_j w_j ⊙ k_j / Σ_j w_j` with `w_j = exp(gate_j + ape_j)`, so a per-channel
`(N, D)` pair would close the pool incrementally. It is `2 × 128` f32 =
**exactly the same 1024 B** as one raw `(k, gate)` row, and it turns the row
for position `p` from a pure function of the token at `p` into a recurrence:
a rejected draft that was accumulated cannot be un-accumulated by overwriting,
so it would need per-position snapshots — `pending_depth` of them, i.e. the
same tensor, minus the idempotence that makes rollback free. This is the
concrete version of §2.2's "must not reuse KDA's recurrent-state machinery".
Rejected.

**(vii) Reuse `AttentionMetadata.mamba_state_indices` as the slot id — rejected
as a silent out-of-bounds.** Tempting: it is already a `(max_num_seqs,)` int32
stable-per-request slot with a null tail. But its comment
(`layers/common/attention_metadata.py:53-59`) says it is a *mamba* slot, and
under unified-align mode it is "derived from the current entry in the device
block table" — an attention-pool block id, which can exceed `max_num_reqs + 1`
by orders of magnitude. Indexing a `[max_num_reqs+1, ...]` buffer with it would
read and write out of bounds on exactly the layout the deployment may enable.
A dedicated `kpool_state_indices` field with its own pool is required.

---

## 5. Correctness argument, worked

Let `s` be a request's first absolute position in a step and `e` one past its
last, so the step covers `[s, e)`. Two invariants:

> **W.** After the step, `pending_kv[slot, p % depth]` holds the true `(k,
> gate)` of position `p`, and `pending_tag` holds `p`, for every
> `p ∈ [e - depth, e)`.
>
> **R.** The step only ever *reads* the buffer at positions in `[s-3, s-1]`.

**R** holds because a pool completing at `p ∈ [s, e)` spans `4*floor(p/4) .. p`,
and `4*floor(p/4) >= p - 3 >= s - 3`; anything `>= s` takes the in-step branch.

**Rollback (MTP).** The next step starts at `s'`, and `s' >= e - R_max` where
`R_max` is the largest number of positions the scheduler can un-commit. With
one batch in flight that is `num_speculative_tokens`; `max_concurrent_batches`
generalizes it. **R** says the next step reads `[s'-3, s'-1]`, so it needs
`s' - 3 >= e - depth`, i.e. `depth >= R_max + 3`. `pending_depth = index_kpool
* R_max = R_max + 4` clears it by one slot. The sweep measures the floor
directly and it is `R_max + 3` for every `K ∈ {1,2,3,4,7}` — the derivation and
the measurement agree, so the slack is one slot, known, not hoped.

Rows written for positions that a rejection un-commits are *not* rolled back
and do not need to be: every one of them is at a position `>= s'`, so it is
rewritten by the next step (writes are keyed by absolute position and
overwrite, never accumulate) before any pool that contains it can complete —
pool `j` completes only when `4j+3` is written, and `4j+3 >= p` for every `p`
in the pool. **The worked case:** `index_kpool = 4`, `K = 2`, `depth = 6`.
A step covers 10 (real), 11 (draft), 12 (draft); pool 2 = {8,9,10,11} completes
using draft 11; position 12 is written to slot `12 % 6 = 0` — which holds
position 6, not position 8, because the depth is 6 and not 4. Both drafts are
rejected, `s' = 11`. The next step covers 11,12,13 and completes pool 2 again
from ring rows 8 (slot 2), 9 (slot 3), 10 (slot 4) and the in-step real 11.
Tags 8, 9, 10 all match. The pooled record for pool 2 is overwritten with the
right value. At `depth = 4` slot 0 would have held position 12, tag 12 ≠ 8, and
the sweep reports exactly that: `1 wrong pooled key, 1 tag violation`.

**Preemption.** `Scheduler._preempt_request` (`scheduler.py:1294-1296`) sets
`num_computed_tokens = 0` and clears `spec_token_ids`; the request re-prefills
from 0. The first re-prefill step has `s = 0` and reads nothing from the
buffer (**R** with `s = 0` is empty). Every later read is of a position written
after the preemption. The sweep runs this with the slot deliberately *not*
cleared (`wipe=False`) and it is still green, so slot reuse across requests is
safe by the same argument — but hand-out should still reset `pending_tag` to
`-1`, because that converts "safe by an argument about `s`" into "detected if
the argument is ever violated", and it costs one `fill_` per admission.

**Chunked prefill at an arbitrary residue.** Chunk boundaries are unconstrained
by `index_kpool`. Prompt 4098, chunks of 999: step 1 covers `[0, 999)` and
completes pools 0..248 entirely in-step; it carries positions `[999-depth, 998]`.
Step 2 starts at `s = 999`; `(999+1) % 4 == 0`, so pool 249 = {996,997,998,999}
completes on its first token, reading 996/997/998 out of the ring (tags match,
`depth >= 3`) and 999 in-step. With chunks of 1000 the same prompt completes
pool 249 in step 1 and step 2 needs no carry read at all. Both residues are
exercised by the sweep's `chunk ∈ {1,3,4,5,7,16}` × `prompt ∈ {13,14,15,16,33}`
sweep, which covers `L % 4 ∈ {0,1,2,3}` on both axes.

**Prefix caching.** Worth stating precisely, because it is easy to aim the
refusal at the right cache for the wrong reason. Local hits are
block-aligned (`hit_length = len(computed_blocks[0]) * block_size`,
`single_type_kv_cache_manager.py:739` for full attention and `:992` for
sliding window) and the pooled-cache work asserts
`block_size % index_kpool == 0`, so a restored prefix always lands on a pool
boundary and the **pending state is unaffected** — the first completing pool
after the hit is fully in-step. What a prefix-cache hit *does* break is the
**pooled** record cache, whose records for the restored prefix were never
written. The construction-time refusal must therefore name the pooled
cache group; it does not need to name this one. Two caveats that keep the
refusal blanket for now: a KV-connector (`num_external_computed_tokens`) and
the fine-grained/`alignment_tokens` hit path can produce non-block-aligned
`num_computed_tokens`, and neither was verified on CPU.

---

## 6. Risks and what the tag buys

The per-position tag is not decoration. The sweep checks it per pool, not per
configuration: of the **12 474** wrong pooled
records the sweep produced at under-sized depths, the number belonging to a
pool whose gather the tag check did not flag is **0**. That is the property
that makes an under-derived depth, a slot-pool bug, a prefix-cache hit at a
non-aligned length and a null-block read all fail loudly instead of producing
fluent text with a degraded retrieval curve.

Known risks:

1. **The `max_concurrent_batches` premise is unverified on CPU.** I could not
   verify, without a device and a running scheduler, that `num_computed_tokens`
   can roll back by at most `max_concurrent_batches × num_speculative_tokens`
   under async scheduling. TPU allows async + eagle/MTP
   (`platforms/tpu_platform.py:953-959` only refuses ngram). The formula is
   conservative for the sync case (measured) and reasoned for the async case.
   The tag is the net. If a cheaper guarantee is wanted, refuse
   `async_scheduling and speculative_config` at kpool construction until it is
   measured on device.
2. **`positions` must be the absolute raw position.** the write path already has to
   assert `pos % index_kpool == (idx - first_key) % index_kpool`; the same
   assertion protects this buffer. If GLM's vision path ever makes `positions`
   mrope/3-D, the addressing breaks silently. Not verified on CPU.
3. **The MTP draft head is a separate buffer, and may not need one at all.**
   `index_share_for_mtp_iteration` is `True` in the real checkpoint. If
   index reuse across draft iterations is ever implemented, layer 45's
   indexer never runs and its buffer is dead weight (7 KiB/request); if it is
   not, layer 45 needs its own buffer at the same depth. The two must land
   together, which is what `KPOOL_MTP_INDEX_REUSE` is for.
4. **HBM is accounted only if the buffer is allocated before the memory
   profile.** Allocate it in the indexer module's construction path, not in
   `initialize_kv_cache`. If it moves later, the planned-vs-allocated
   assertion must gain a term for it.
5. **Donation.** The buffer is an in-out operand of the fused kpool op
   . If it is not donated/aliased, XLA copies ~460 KiB per layer per
   step. `pallas.set_buffer_donor_` is currently commented out in
   `TpuModelRunner._allocate_kv_cache` ("TorchTPU does not support this right
   now"), so confirm the `jax_op` aliasing path covers it; not verifiable on
   CPU.
6. **`num_slots` must track `max_num_reqs`.** The pool is exhausted only if
   more than `max_num_reqs` requests are live, which the persistent batch
   forbids — mirror the comment and the `assert req_id is not None` shape of
   `_build_mamba_state_indices` rather than restating it. The DP-dummy case it
   handles (`run_dp_dummy_draft` passing a `num_reqs` larger than the live
   batch) applies verbatim.

---

## 7. The measured schedule sweep

Output of the enumeration these
conclusions come from. `tests/kernels/glm5_kpool/test_pool_write.py`
re-runs the same schedules against the real `kpool_write`, including the
negative control at the rejected depth.

`tests/kernels/glm5_kpool/test_pool_write.py`, 2026-09-04,
`~/venvs/vllm-cpu/bin/python`, no TPU:

Section 0 is transcribed in §2; the rest follows.

```
1. The §2.2 counterexample: kpool=4, num_spec=2, positions 10/11(draft)/12(draft), both drafts rejected.
   depth=4: 1 wrong pooled key(s), 1 tag violation(s), 0 silent  -> WRONG
   depth=6: 0 wrong pooled key(s), 0 tag violation(s), 0 silent  -> ok

2. Sweep: chunked prefill at every residue x spec decode with every acceptance pattern.
    depth  k_spec  schedules   wrong  tagviol  silent  verdict
        4       1        120       0        0       0  pass
        4       1        120       0        0       0  pass
        5       1        120       0        0       0  pass
        4       2        270     228      228       0  FAIL
        5       2        270       0        0       0  pass
        6       2        270       0        0       0  pass
        4       3        480     966     1356       0  FAIL
        6       3        480       0        0       0  pass
        7       3        480       0        0       0  pass
        4       4        750    2178     4164       0  FAIL
        7       4        750       0        0       0  pass
        8       4        750       0        0       0  pass
        4       7       1920    9102    18552       0  FAIL
       10       7       1920       0        0       0  pass
       11       7       1920       0        0       0  pass

2b. Smallest depth that survives the whole sweep, per k_spec (the floor, so the chosen depth's slack is measured not hoped):
   k_spec=1: floor=4  (k_spec+3=4; chosen index_kpool+k_spec=5 leaves 1 slot(s) of slack)
   k_spec=2: floor=5  (k_spec+3=5; chosen index_kpool+k_spec=6 leaves 1 slot(s) of slack)
   k_spec=3: floor=6  (k_spec+3=6; chosen index_kpool+k_spec=7 leaves 1 slot(s) of slack)
   k_spec=4: floor=7  (k_spec+3=7; chosen index_kpool+k_spec=8 leaves 1 slot(s) of slack)
   k_spec=7: floor=10  (k_spec+3=10; chosen index_kpool+k_spec=11 leaves 1 slot(s) of slack)

3. Wrong pooled records over the whole sweep: 12474. Of those, wrong at a pool whose gather the tag check never flagged (i.e. SILENT): 0.
   (i.e. the per-row absolute-position tag turns the depth-4 failure from silent into loud.)

4. Preemption mid-decode (num_computed_tokens -> 0, re-prefill). `wipe=False` models a slot handed to a request without being cleared, i.e. inheriting another request's rows:
   depth=5 k_spec=1 wipe=True: 0 wrong, 0 tag violation(s), 0 silent -> pass
   depth=5 k_spec=1 wipe=False: 0 wrong, 0 tag violation(s), 0 silent -> pass
   depth=7 k_spec=3 wipe=True: 0 wrong, 0 tag violation(s), 0 silent -> pass
   depth=7 k_spec=3 wipe=False: 0 wrong, 0 tag violation(s), 0 silent -> pass

5. Why a depth-4 ring would ship green: at num_spec=1 (the usual configuration)
   depth=4 k_spec=1, all drafts rejected: 0 wrong, 0 tag violation(s), 0 silent
   depth=4 k_spec=2, all drafts rejected: 3 wrong, 3 tag violation(s), 0 silent
```

The duplicate `depth=4 k_spec=1` row is not a typo: at `k_spec = 1` the sweep's
`index_kpool + k_spec - 1` and `index_kpool` are the same depth.

---
