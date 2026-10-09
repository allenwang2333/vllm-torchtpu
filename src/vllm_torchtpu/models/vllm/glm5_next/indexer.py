# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM-5.3-Flash DSA indexer: top-k over *pools* of ``index_kpool`` tokens.

DeepSeek-V3.2 and GLM-5.2 score every past token and keep the best
``index_topk``. GLM-5.3 scores groups of ``index_kpool``=4 adjacent tokens and
keeps the best ``index_topk // 4`` = 512 groups, which expand back to 2048
tokens. The pool key is a *learned per-channel* average of its four members::

    p = softmax_over_members(gate + ape) * k   summed over members

with ``gate = index_kpool_compress_gate @ x`` and ``ape`` a learned
``[4, 128]`` positional bias, so the four members are not interchangeable.
Pools are cut on absolute position -- pool ``j`` is always tokens ``4j..4j+3``
-- and a pool becomes a candidate only once its *last* token is visible. The
``index_kpool_always_select_tail`` flag then appends the up-to-3 tokens of the
query's own incomplete pool, giving an output width of ``2048 + 3``.

``test_indexer_math.py`` in the project directory checks this arithmetic --
in the unpadded, ``first_key == 0`` form vLLM always presents -- against
``transformers``' ``Glm5NextTextIndexer`` at thirteen sequence lengths.

Status: this is the *correctness* implementation. Scoring runs as plain XLA
(a scan over sequences, masked), not as a Pallas kernel, and it materialises
a ``[num_tokens, num_pools]`` score matrix. That is fine up to a few thousand
tokens with a handful of concurrent sequences and hopeless at 200K; see the
notes on `Glm5NextIndexerCache` and `_build_kpool_index_op` for what a real
kernel has to do instead.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    scatter_rows, streamindex_topk)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.adapter.attention import (
    PallasMLAttentionBackend, VllmTPUDeepseekV32IndexerBackend)


def dense_dsa_fallback() -> bool:
    """``GLM53_DENSE_DSA=1`` builds the DSA layers as plain dense MLA.

    Dense and sparse are numerically identical while every sequence is shorter
    than ``index_topk``, so this is the A/B switch for checking the indexer:
    run the same prompt both ways at >2048 tokens and diff the output.
    """
    return os.environ.get("GLM53_DENSE_DSA", "0") == "1"


class Glm5NextIndexerBackend(VllmTPUDeepseekV32IndexerBackend):
    """Plain bf16 rows in the MLA layout, not DeepSeek's packed uint8 rows.

    The parent sizes a packed fp8 + scale layout; ours is ``[k | gate]`` bf16
    and the scoring op reads it as ``[blocks, bs/4, 4, 2 * head_dim]``, which is
    exactly the MLA latent cache shape. Answering with the MLA formulas also
    covers the hybrid page accounting: ``_find_non_ssm_backend`` returns the
    first non-SSM layer's backend, and an indexer registers before its
    ``mla_attn`` sibling, so these hooks are asked for the MLA page size too.
    """

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[int, ...]:
        return PallasMLAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)

    @classmethod
    def get_kv_cache_page_size_bytes(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> int:
        return PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
            block_size, num_kv_heads, head_size, cache_dtype_str)


class Glm5NextIndexerCache(DeepseekV32IndexerCache):
    """Per-token ``[k | gate]`` cache backing the k-pool indexer.

    Subclassing is load-bearing: the TPU runner collects an indexer's cache
    spec through an ``isinstance(attn_module, DeepseekV32IndexerCache)`` branch
    (``runner/tpu_runner.py``), so a fresh type would simply be skipped and the
    cache would never be allocated.

    GLM-5.2 stores one fp8 row plus an e8m0 scale byte per token, 129 bytes
    padded to 256. We store the raw bf16 ``k`` (128) *and* the raw bf16 pooling
    ``gate`` (128) -- 512 bytes per token per layer, double GLM-5.2 -- because
    a pool key needs all four members' ``k`` and ``gate`` together, and during
    decode the members arrive on four different steps.

    The cheap layout is to store *pool keys only*, one row per four tokens
    (64 bytes/token, 8x smaller than what is here), which needs a running
    per-channel softmax carried across steps for the pool currently being
    filled -- the same trick DeepSeek-V4's ``compress_and_store`` kernel plays
    with its ``state_cache``. That belongs with the Pallas rewrite; keeping the
    raw members means this implementation has no carry state at all and can be
    checked against the reference directly.
    """

    # The sparse MLA kernel forces `--kv-cache-dtype fp8` on the main KV
    # cache, but these rows are read by the plain-jnp scoring op above and
    # stay bf16. The runner honours this flag by keeping the spec out of its
    # kv-dtype normalization (`fixed_dtype_layers` in `get_kv_cache_spec`).
    kv_dtype_is_fixed = True

    def get_attn_backend(self) -> type[Glm5NextIndexerBackend]:
        return Glm5NextIndexerBackend


# One record per pool: `[fp8 key x 128 | e8m0 scale | pad]`, the 128-lane
# padded format `streamindex_topk` reads.
POOL_KEY_RECORD_BYTES = 256
POOL_KEY_KPOOL = 4


class Glm5NextPoolKeyBackend(VllmTPUDeepseekV32IndexerBackend):
    """Pages of fp8 pool-key records, ``kpool`` tokens per record.

    vLLM sizes a cache by bytes per token, so the spec declares
    ``POOL_KEY_RECORD_BYTES / kpool`` (64) bytes per token; a block of
    ``block_size`` tokens is then exactly ``block_size / kpool`` records, laid
    out as `streamindex_topk`'s ``[pages, rows // 4, 4, record]``.
    """

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[int, ...]:
        del num_kv_heads, cache_dtype_str
        rows = block_size // POOL_KEY_KPOOL
        if rows % 4 or head_size * POOL_KEY_KPOOL != POOL_KEY_RECORD_BYTES:
            raise ValueError(
                f"pool key cache needs block_size / {POOL_KEY_KPOOL} to be a "
                f"multiple of 4 (block_size={block_size}) and "
                f"{POOL_KEY_RECORD_BYTES // POOL_KEY_KPOOL} bytes per token "
                f"(got {head_size})")
        return (num_blocks, rows // 4, 4, POOL_KEY_RECORD_BYTES)

    @classmethod
    def get_kv_cache_page_size_bytes(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> int:
        del num_kv_heads, cache_dtype_str
        return block_size * head_size


class Glm5NextPoolKeyCache(DeepseekV32IndexerCache):
    """fp8 key of every *complete* pool, scored by `streamindex_topk`.

    A second indexer cache beside `Glm5NextIndexerCache`: the member rows are
    still needed to build a key, but scoring only ever reads keys, and reading
    them from here means 64 bytes per token instead of the 512-byte member
    row, in the record format the kernel consumes directly.
    """

    kv_dtype_is_fixed = True

    def __init__(self, prefix: str, cache_config: CacheConfig) -> None:
        super().__init__(head_dim=POOL_KEY_RECORD_BYTES // POOL_KEY_KPOOL,
                         dtype=torch.uint8,
                         prefix=prefix,
                         cache_config=cache_config)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # The base class squeezes dim 1, which here is `rows // 4` and is 1
        # at a 16-token block. Keep the 4-D page layout as allocated.
        self.kv_cache = kv_cache

    def get_attn_backend(self) -> type[Glm5NextPoolKeyBackend]:
        return Glm5NextPoolKeyBackend


# How large an intermediate the pool-scoring step may materialise before it
# falls back to a loop. 128 MiB is comfortable next to the fp32 `[tokens,
# pools]` score matrix it has to hold anyway. Read at trace time, so lowering
# it (`GLM53_INDEX_SCORE_BUDGET=0`, or assigning to this global, which is what
# `test_kpool_index_op.py` does) forces the narrower strategies and lets one
# small test case cover all three.
_SCORE_INTERMEDIATE_BUDGET_BYTES = int(
    os.environ.get("GLM53_INDEX_SCORE_BUDGET", 128 * 1024 * 1024))


def _step_positions(seq_lens: jax.Array, cu_q_lens: jax.Array,
                    num_tokens: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(sequence id, absolute position, is-real) of every token in the step."""
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_id = jnp.searchsorted(cu_q_lens[1:], tok,
                              side="right").astype(jnp.int32)
    q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
    pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
    live = tok < cu_q_lens[-1]
    return seq_id, pos, live


def _write_members_and_pool_keys(
        cache: jax.Array,  # bf16 [blocks, bs/4, 4, 2 * head_dim]
        k_new: jax.Array,
        gate_new: jax.Array,
        ape: jax.Array,
        pages: jax.Array,  # i32 [seqs, pages_per_seq]
        seq_id: jax.Array,
        pos: jax.Array,
        live: jax.Array,
        *,
        kpool: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Store this step's ``[k | gate]`` rows and key the pools they complete.

    Returns ``(cache, new_keys, completes, pool_of, m_k)``: ``new_keys[t]`` is
    the key of the pool token ``t`` completes, meaningful where
    ``completes[t]``; ``m_k`` are those pools' member ``k`` rows.
    """
    _, block_size_per_packing, packing, row_width = cache.shape
    block_size = block_size_per_packing * packing
    head_dim = k_new.shape[-1]

    # ---- write [k | gate] into this step's slots --------------------------
    # Padded rows are aimed at an out-of-bounds page. `mode="drop"` is not
    # decoration: the default scatter mode *clamps*, which would land them on
    # the last block and corrupt whichever sequence holds it.
    row = jnp.concatenate([k_new, gate_new], axis=-1)
    row = jnp.pad(row, ((0, 0), (0, row_width - row.shape[-1])))
    page = jnp.where(live, pages[seq_id, pos // block_size], jnp.int32(2**30))
    slot = pos % block_size
    cache = cache.at[page, slot // packing,
                     slot % packing].set(row, mode="drop")

    # ---- finalise the pools this step completed ---------------------------
    # A pool key is fixed the moment its fourth member lands, and is only ever
    # read afterwards, so compute it once and store it instead of rebuilding it
    # from members on every step. Rebuilding read `seqs * max_pools * kpool`
    # rows per layer -- 8.4M at 4 seqs and an 8K context -- *regardless of how
    # long the sequences actually are*, which is why decode sat at ~40 tok/s
    # whatever the prompt length while dense ran at ~120.
    #
    # The key goes over member 0's `gate` half. `gate` exists only to build the
    # key, and once the key exists nothing reads the members again, so this
    # costs no extra bytes and needs no spec change. (Dropping the members
    # altogether is another 8x smaller, but needs a per-channel softmax carried
    # across steps -- see `Glm5NextIndexerCache`.)
    if row_width != 2 * head_dim:
        raise ValueError(
            f"indexer cache row is {row_width} wide, expected 2 * {head_dim}: "
            "the pool key is stored over member 0's gate half")
    # Every access below moves whole `2 * head_dim` rows. Two tempting
    # shortcuts are both far slower on TPU:
    #   * reshaping the cache into k/gate halves tiles differently (`T(4,128)`
    #     vs `T(2,128)`), so the reshape is a physical copy of the *whole*
    #     cache, twice per layer -- 16 ms of a 29 ms decode step;
    #   * gathering/scattering a `[head_dim:]` window of a row lowers to a
    #     `while` loop of per-index dynamic slices -- 120K of them per step,
    #     ~800 ms.

    # Pool `j` is tokens `4j .. 4j+3`, cut on absolute position, so a token
    # completes a pool exactly when `pos + 1` is a multiple of `kpool`, and the
    # member positions follow from `pos` alone.
    pool_of = pos // kpool  # [tokens]
    completes = live & (((pos + 1) % kpool) == 0)
    mem_pos = (pool_of[:, None] * kpool +
               jnp.arange(kpool, dtype=jnp.int32)[None, :])
    mem_page = jnp.where(completes[:, None],
                         pages[seq_id[:, None], mem_pos // block_size],
                         jnp.int32(2**30))
    mem_slot = mem_pos % block_size
    members = cache[mem_page, mem_slot // packing,
                    mem_slot % packing].astype(jnp.float32)
    m_k, m_gate = jnp.split(members, 2, axis=-1)  # [tokens, kpool, head_dim]

    # Per-channel softmax over the four members: the pool key is a learned
    # weighted average, not a mean, and `ape` makes member order matter.
    probs = jax.nn.softmax(m_gate + ape, axis=-2)
    new_keys = jnp.sum(probs * m_k, axis=-2)  # [tokens, head_dim]

    return cache, new_keys, completes, pool_of, m_k


def _expand_selected_pools(
        selected: jax.Array,  # i32 [tokens, select_k]
        selected_ok: jax.Array,  # bool [tokens, select_k]
        pos: jax.Array,
        live: jax.Array,
        *,
        kpool: int,
        select_k: int,
        width: int,
) -> jax.Array:
    """Selected pool ids -> left-packed token indices plus the tail."""
    num_tokens = selected.shape[0]
    visible = (pos + 1) // kpool  # [tokens]
    token_ids = (selected[..., None] * kpool +
                 jnp.arange(kpool, dtype=jnp.int32))
    token_ids = jnp.where(selected_ok[..., None], token_ids, -1)
    token_ids = token_ids.reshape(num_tokens, select_k * kpool)

    # The query's own incomplete pool is never a candidate, so its tokens
    # are appended unconditionally -- that is what `always_select_tail`
    # means. There are `kpool - 1` slots and `pos % kpool` real entries.
    tail = (visible * kpool)[:, None] + jnp.arange(kpool - 1, dtype=jnp.int32)
    tail = jnp.where(tail <= pos[:, None], tail, -1)

    # Every row must be **left-packed**: the kernel derives the per-token kv
    # length as `sum(topk_indices != -1)` and then attends to that many
    # *leading* entries (see `kernels/mla/sparse/kernel.py`, "Mask scores past
    # kv_len (the -1 tail)"). Padding entries are gathered from pseudo-random
    # pages to spread the memory traffic, so a -1 in the middle of a row does
    # not read as zero, it reads as another sequence's KV. Concatenating the
    # tail after the full-width pool block would do exactly that whenever
    # fewer than `select_k` pools are visible, i.e. below 8K of context.
    #
    # `-inf` scores sort last, so the `v = min(visible, select_k)` real pools
    # already occupy the first `v` slots of `selected`, and the real tail
    # entries are the first `(pos + 1) % kpool` of `tail`. Both halves are
    # therefore prefixes, and packing is a shift of the tail block down to
    # offset `kpool * v`.
    offset = jnp.minimum(visible, select_k) * kpool  # [tokens]
    col = jnp.arange(width, dtype=jnp.int32)[None, :]
    in_tail = col - offset[:, None]
    # `tail[t, c]` is just `visible[t] * kpool + c`, so compute it in place.
    # A `take_along_axis` from the `[tokens, kpool - 1]` table lowers to one
    # gather per output element: 17M of them, 179 ms, for an 8K prefill.
    tail_at_col = (visible * kpool)[:, None] + in_tail
    tail_at_col = jnp.where(tail_at_col <= pos[:, None], tail_at_col, -1)
    pool_at_col = jnp.pad(token_ids,
                          ((0, 0), (0, max(0, width - token_ids.shape[-1]))),
                          constant_values=-1)[:, :width]

    topk_indices = jnp.where(
        col < offset[:, None], pool_at_col,
        jnp.where((in_tail >= 0) & (in_tail < kpool - 1), tail_at_col, -1))
    topk_indices = jnp.where(live[:, None], topk_indices, -1)
    return topk_indices


def _kpool_index_jax(
        cache: jax.Array,  # bf16 [blocks, bs/4, 4, 2 * head_dim]
        k_new: jax.Array,  # bf16 [tokens, head_dim]
        gate_new: jax.Array,  # bf16 [tokens, head_dim]
        q: jax.Array,  # f32  [tokens, heads, head_dim]
        weights: jax.Array,  # f32  [tokens, heads]
        ape: jax.Array,  # f32  [kpool, head_dim]
        seq_lens: jax.Array,  # i32  [seqs]
        page_indices: jax.Array,  # i32  [seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32  [seqs + 1]
        *,
        kpool: int,
        select_k: int,
        width: int,
        max_pools: int,
        num_heads: int,
) -> tuple[jax.Array, jax.Array]:
    """The whole indexer, as one jnp function. See `_build_kpool_index_op`.

    Module level so `test_kpool_index_op.py` can run it against the torch
    reference on CPU, without a TPU or a vLLM engine. The pool keys live over
    member 0's ``gate`` half and are scored by plain XLA, which reads every
    sequence's ``max_pools`` keys each step; `_kpool_index_v2_jax` is the
    serving path.
    """
    _, block_size_per_packing, packing, row_width = cache.shape
    block_size = block_size_per_packing * packing
    num_seqs = seq_lens.shape[0]
    num_tokens = k_new.shape[0]
    head_dim = k_new.shape[-1]
    pages = page_indices.reshape(num_seqs, -1)
    seq_id, pos, live = _step_positions(seq_lens, cu_q_lens, num_tokens)
    cache, new_keys, completes, pool_of, m_k = _write_members_and_pool_keys(
        cache, k_new, gate_new, ape, pages, seq_id, pos, live, kpool=kpool)

    key_slot = (pool_of * kpool) % block_size
    key_page = jnp.where(completes,
                         pages[seq_id, (pool_of * kpool) // block_size],
                         jnp.int32(2**30))
    # Most tokens do not complete a pool, so most rows of this scatter are
    # aimed out of bounds -- `mode="drop"` again, for the same reason.
    # Member 0's `k` half is rewritten with the value just read, which
    # round-trips exactly (bf16 -> f32 -> bf16).
    key_row = jnp.concatenate([m_k[:, 0, :], new_keys],
                              axis=-1).astype(cache.dtype)
    cache = cache.at[key_page, key_slot // packing,
                     key_slot % packing].set(key_row, mode="drop")

    # ---- read back every pool's key ---------------------------------------
    # Incomplete pools return whatever their gate half holds; `candidate`
    # below discards those scores, so the garbage never reaches `top_k`.
    key_pos = jnp.arange(max_pools, dtype=jnp.int32) * kpool
    p_slot = key_pos % block_size
    pool_keys = cache[pages[:, key_pos // block_size], p_slot // packing,
                      p_slot % packing][..., head_dim:].astype(jnp.float32)

    # ---- score -------------------------------------------------------------
    scale = jnp.float32(head_dim**-0.5)
    qf = q.astype(jnp.float32) * scale
    wf = weights.astype(jnp.float32)

    # `relu` sits between the per-head dot product and the per-head weighting,
    # so the heads cannot be collapsed into one GEMM for free, and each token
    # may only look at *its own* sequence's pools, which is a block-diagonal
    # product XLA cannot express with a runtime `cu_q_lens`. Both are handled
    # by looping and throwing away everything but the diagonal -- but a loop
    # trip is ~10 us of fixed overhead on TPU, and `num_seqs x num_heads` of
    # them (128 at 4 seqs) is ~2 ms per layer, which at 11 layers is most of a
    # decode step. `num_tokens` is static per compiled graph, so pick the
    # widest form whose intermediate fits: decode buckets are small enough to
    # fuse everything, an 8K prefill has to keep both loops.
    def rows_of_this_seq(scores_tsp):
        """[tokens, seqs, pools] -> [tokens, pools] at each token's own seq."""
        return jnp.take_along_axis(scores_tsp,
                                   seq_id[:, None, None],
                                   axis=1).squeeze(1)

    fused_all = num_tokens * num_heads * num_seqs * max_pools * 4
    fused_seqs = num_tokens * num_seqs * max_pools * 4

    if fused_all <= _SCORE_INTERMEDIATE_BUDGET_BYTES:
        s = jax.nn.relu(jnp.einsum("thd,spd->thsp", qf, pool_keys))
        index_scores = rows_of_this_seq(jnp.einsum("th,thsp->tsp", wf, s))
    elif fused_seqs <= _SCORE_INTERMEDIATE_BUDGET_BYTES:

        def add_head_all_seqs(h, part):
            head = jax.nn.relu(jnp.einsum("td,spd->tsp", qf[:, h, :],
                                          pool_keys))
            return part + wf[:, h][:, None, None] * head

        index_scores = rows_of_this_seq(
            jax.lax.fori_loop(
                0, num_heads, add_head_all_seqs,
                jnp.zeros((num_tokens, num_seqs, max_pools), jnp.float32)))
    else:

        def score_one_seq(s, acc):
            keys_t = pool_keys[s].T  # [head_dim, pools]

            def add_head(h, part):
                return part + wf[:, h][:, None] * jax.nn.relu(
                    qf[:, h, :] @ keys_t)

            part = jax.lax.fori_loop(
                0, num_heads, add_head,
                jnp.zeros((num_tokens, max_pools), jnp.float32))
            return jnp.where((seq_id == s)[:, None], part, acc)

        index_scores = jax.lax.fori_loop(
            0, num_seqs, score_one_seq,
            jnp.zeros((num_tokens, max_pools), jnp.float32))

    # A pool is a candidate only once its last token is visible, i.e. the
    # query has passed position `4j+3`. Equivalently the query at `t` sees
    # the first `(t + 1) // kpool` pools.
    visible = (pos + 1) // kpool  # [tokens]
    pool_ids = jnp.arange(max_pools, dtype=jnp.int32)
    candidate = pool_ids[None, :] < visible[:, None]
    index_scores = jnp.where(candidate, index_scores,
                             jnp.finfo(jnp.float32).min)

    selected = jax.lax.top_k(index_scores, select_k)[1].astype(jnp.int32)
    selected_ok = jnp.take_along_axis(candidate, selected, axis=-1)
    topk_indices = _expand_selected_pools(selected, selected_ok, pos, live,
                                          kpool=kpool, select_k=select_k,
                                          width=width)
    return cache, topk_indices


# (decode, prefill, mixed). The kernel needs `rows_per_page * pages` to be a
# multiple of 128; at block 256 a page holds 64 pool rows, so the page counts
# are even. Untuned starting point taken from DeepSeek-V4's indexer
# (`deepseek_v4_indexer.py`, (3, 2, 2) on its larger pages).
_STREAMINDEX_KV_PAGES_PER_BLOCK = (4, 2, 2)
_STREAMINDEX_QUERIES_PER_BLOCK = (1, 128, 128)


def _pool_key_records(keys: jax.Array, record_width: int) -> jax.Array:
    """f32 ``[n, head_dim]`` keys -> ``[fp8 x head_dim | e8m0 scale | pad]`` u8.

    The record `streamindex_topk` reads: one power-of-two scale per key, so
    the score can be rescaled after the fp8 dot product.
    """
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.maximum(jnp.max(jnp.abs(keys), axis=-1, keepdims=True), 1e-4)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (keys / scale).astype(jnp.float8_e4m3fn)
    record = jnp.concatenate([
        jax.lax.bitcast_convert_type(q, jnp.uint8),
        jax.lax.bitcast_convert_type(scale.astype(jnp.float8_e8m0fnu),
                                     jnp.uint8),
    ], axis=-1)
    return jnp.pad(record, ((0, 0), (0, record_width - record.shape[-1])))


def _kpool_index_v2_jax(
        cache: jax.Array,  # bf16 [blocks, bs/4, 4, 2 * head_dim]
        key_cache: jax.Array,  # u8 [blocks, bs/kpool/4, 4, record_width]
        k_new: jax.Array,  # bf16 [tokens, head_dim]
        gate_new: jax.Array,  # bf16 [tokens, head_dim]
        q: jax.Array,  # f32  [tokens, heads, head_dim]
        weights: jax.Array,  # f32  [tokens, heads]
        ape: jax.Array,  # f32  [kpool, head_dim]
        seq_lens: jax.Array,  # i32  [seqs]
        page_indices: jax.Array,  # i32  [seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32  [seqs + 1]
        distribution: jax.Array,  # i32  [3]
        key_page_indices: jax.Array,  # i32  [seqs * pages_per_seq]
        *,
        kpool: int,
        select_k: int,
        width: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The indexer with pool keys in their own fp8 cache, scored by a kernel.

    Members are still stored and keyed exactly as in `_kpool_index_jax`. The
    finished keys then go, fp8-quantized, into ``key_cache`` -- one record per
    pool, addressed through the same block table at ``kpool`` tokens per row
    -- and `streamindex_topk` (DeepSeek-V4's indexer kernel: relu(q.k)
    weighted over heads, top-k on the SparseCore) scores and selects them.
    That kernel walks only the pages each sequence actually has, where the
    XLA path reads ``max_pools`` keys per sequence per step regardless of
    length.

    The two caches sit in different vLLM KV cache groups and so have
    different block tables: ``page_indices`` addresses ``cache`` and
    ``key_page_indices`` addresses ``key_cache``.
    """
    num_seqs = seq_lens.shape[0]
    num_tokens = k_new.shape[0]
    head_dim = k_new.shape[-1]
    block_size = cache.shape[1] * cache.shape[2]
    rows_per_page = key_cache.shape[1] * key_cache.shape[2]
    if rows_per_page * kpool != block_size:
        raise ValueError(
            f"pool key cache holds {rows_per_page} rows per page, expected "
            f"block_size / kpool = {block_size} / {kpool}")
    pages = page_indices.reshape(num_seqs, -1)
    seq_id, pos, live = _step_positions(seq_lens, cu_q_lens, num_tokens)
    cache, new_keys, completes, pool_of, _ = _write_members_and_pool_keys(
        cache, k_new, gate_new, ape, pages, seq_id, pos, live, kpool=kpool)

    # Pool `j` is row `j` of its sequence's compressed address space: page
    # `j // rows_per_page` of the block table, row `j % rows_per_page`.
    key_pages = key_page_indices.reshape(num_seqs, -1)
    dst_rows = jnp.where(
        completes,
        key_pages[seq_id, pool_of // rows_per_page] * rows_per_page +
        pool_of % rows_per_page, -1)
    key_cache = scatter_rows(key_cache,
                             _pool_key_records(new_keys, key_cache.shape[-1]),
                             dst_rows)

    # relu(c * x) = c * relu(x) for c > 0, so the softmax scale and the
    # per-row fp8 scale of q both fold into the head weights.
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    qf = q.astype(jnp.float32) * jnp.float32(head_dim**-0.5)
    q_scale = jnp.maximum(jnp.max(jnp.abs(qf), axis=-1), 1e-12) / fp8_max
    q_fp8 = (qf / q_scale[..., None]).astype(jnp.float8_e4m3fn)
    selected = streamindex_topk(
        q=q_fp8,
        indexer_weights=weights.astype(jnp.float32) * q_scale,
        cache_kv=key_cache,
        seq_lens=seq_lens,
        page_indices=key_page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=select_k,
        compression_ratio=kpool,
        num_kv_pages_per_block=_STREAMINDEX_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_STREAMINDEX_QUERIES_PER_BLOCK,
        complete_rows_only=True,
    )
    topk_indices = _expand_selected_pools(selected, selected >= 0, pos, live,
                                          kpool=kpool, select_k=select_k,
                                          width=width)
    return cache, key_cache, topk_indices


@functools.lru_cache(maxsize=None)
def _build_kpool_index_op(kpool: int, select_k: int, width: int,
                          max_pools: int, num_heads: int):
    """Build the fused write-cache + score-pools + top-k op.

    Cached because all 11 DSA layers ask for the same op: the name goes to
    `torch.library.custom_op`, and registering it a second time leaves the
    entry without a schema, which surfaces much later as a Dynamo
    "Tried to access the schema for  which doesn't have a schema registered
    yet" while tracing the model.

    Returned callable:
    ``op(cache, k_new, gate_new, q, weights, ape, seq_lens, page_indices,
    cu_q_lens) -> int32 [num_tokens, width]``, and ``cache`` is updated in
    place.

    Everything is one XLA region rather than eager torch ops because the write
    is a three-index scatter and the read is a paged gather; expressing those
    through torch-tpu's exporter is far more fragile than handing JAX the whole
    thing.

    The scoring loop is deliberately naive. ``relu`` sits between the per-head
    dot product and the per-head weighting, so the heads cannot be collapsed
    into the GEMM and the intermediate is ``[tokens, heads, pools]``; and each
    token may only look at *its own* sequence's pools, which is a block-diagonal
    matmul that XLA has no way to express with a runtime ``cu_q_lens``. The
    loop below therefore scores every token against every sequence's pools and
    throws away all but the diagonal -- ``num_seqs`` times more work than
    necessary. Run it with a small ``--max-num-seqs``.
    """

    # `pallas.jax_op` reads the annotations to build the torch schema, so they
    # are load-bearing, not documentation.
    def _bound_kpool_index_jax(
            cache: jax.Array, k_new: jax.Array, gate_new: jax.Array,
            q: jax.Array, weights: jax.Array, ape: jax.Array,
            seq_lens: jax.Array, page_indices: jax.Array,
            cu_q_lens: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _kpool_index_jax(cache,
                                k_new,
                                gate_new,
                                q,
                                weights,
                                ape,
                                seq_lens,
                                page_indices,
                                cu_q_lens,
                                kpool=kpool,
                                select_k=select_k,
                                width=width,
                                max_pools=max_pools,
                                num_heads=num_heads)

    jax_op = pallas.jax_op("pallas::glm5_next_kpool_index",
                           _bound_kpool_index_jax,
                           donate_argnums=(0, ))

    def _fake(cache, k_new, *args, **kwargs):
        return torch.empty_like(cache), torch.empty(
            (k_new.shape[0], width), dtype=torch.int32, device=k_new.device)

    jax_op.register_fake(_fake)

    def op(cache, *args):
        new_cache, topk_indices = jax_op(cache, *args)
        cache.copy_(new_cache)
        return topk_indices

    return op


@functools.lru_cache(maxsize=None)
def _build_kpool_index_v2_op(kpool: int, select_k: int, width: int):
    """`_kpool_index_v2_jax` as a torch op; both caches updated in place.

    Returned callable: ``op(cache, key_cache, k_new, gate_new, q, weights,
    ape, seq_lens, page_indices, cu_q_lens, distribution, key_page_indices)
    -> int32 [num_tokens, width]``. Cached for the same reason as
    `_build_kpool_index_op`.
    """

    # `pallas.jax_op` reads the annotations to build the torch schema.
    def _bound(cache: jax.Array, key_cache: jax.Array, k_new: jax.Array,
               gate_new: jax.Array, q: jax.Array, weights: jax.Array,
               ape: jax.Array, seq_lens: jax.Array, page_indices: jax.Array,
               cu_q_lens: jax.Array, distribution: jax.Array,
               key_page_indices: jax.Array
               ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _kpool_index_v2_jax(cache, key_cache, k_new, gate_new, q,
                                   weights, ape, seq_lens, page_indices,
                                   cu_q_lens, distribution, key_page_indices,
                                   kpool=kpool, select_k=select_k, width=width)

    jax_op = pallas.jax_op("pallas::glm5_next_kpool_index_v2",
                           _bound,
                           donate_argnums=(0, 1))

    def _fake(cache, key_cache, k_new, *args, **kwargs):
        return (torch.empty_like(cache), torch.empty_like(key_cache),
                torch.empty((k_new.shape[0], width),
                            dtype=torch.int32,
                            device=k_new.device))

    jax_op.register_fake(_fake)

    def op(cache, key_cache, *args):
        new_cache, new_key_cache, topk_indices = jax_op(cache, key_cache, *args)
        cache.copy_(new_cache)
        key_cache.copy_(new_key_cache)
        return topk_indices

    return op


def use_v1_indexer() -> bool:
    """``GLM53_INDEXER_V1=1``: score pools with the XLA path, no key cache.

    The v1 op is the reference `test_kpool_index_op.py` checks against torch;
    v2 (`_kpool_index_v2_jax`) scores fp8 keys and can swap near-tied pools at
    the top-k boundary, so this is the switch for A/B-ing that.
    """
    return os.environ.get("GLM53_INDEXER_V1", "0") == "1"


class Glm5NextIndexer(nn.Module):
    """Produces the per-token candidate list the sparse MLA kernel gathers.

    Unlike DeepSeek's indexer this one applies no rope at all: GLM-5.3 has
    ``qk_rope_head_dim == 0`` and the reference forward never touches the
    ``indexer_rope_interleave`` flag the config still carries.
    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
    ) -> None:
        super().__init__()
        self.n_heads = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.kpool = int(config.index_kpool)
        self.topk_tokens = int(config.index_topk)
        self.select_k = self.topk_tokens // self.kpool
        self.always_select_tail = bool(config.index_kpool_always_select_tail)
        # The op's output goes straight to the sparse MLA kernel, so it takes
        # the buffer's width: `topk + kpool - 1` padded to the kernel's
        # multiple of 4. The extra columns come out as -1, past the packed
        # prefix.
        self.width = topk_indices_buffer.shape[1]
        self.softmax_scale = self.head_dim**-0.5
        self.topk_indices_buffer = topk_indices_buffer

        if int(config.qk_rope_head_dim) != 0:
            raise ValueError(
                "Glm5NextIndexer assumes the released rope-free indexer; got "
                f"qk_rope_head_dim={config.qk_rope_head_dim}")
        if not getattr(config, "index_kpool_compress", False):
            raise ValueError("Glm5NextIndexer requires index_kpool_compress")

        # Replicated, not sharded: `n_heads`=32 is small, every TP rank needs
        # the full candidate list anyway, and replicating avoids an all-gather
        # of the head scores before the top-k.
        self.wq_b = ReplicatedLinear(int(config.q_lora_rank),
                                     self.n_heads * self.head_dim,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.wq_b")
        self.wk = ReplicatedLinear(config.hidden_size,
                                   self.head_dim,
                                   bias=False,
                                   quant_config=None,
                                   prefix=f"{prefix}.wk")
        self.weights_proj = ReplicatedLinear(config.hidden_size,
                                             self.n_heads,
                                             bias=False,
                                             quant_config=None,
                                             prefix=f"{prefix}.weights_proj")
        # LayerNorm, with a bias, not the RMSNorm used everywhere else in this
        # model -- and at eps 1e-6 rather than the model's 1e-5.
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(self.kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, config.hidden_size))

        self.k_cache = Glm5NextIndexerCache(
            head_dim=2 * self.head_dim,  # [k | gate], bf16
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )

        self.key_cache = None
        if use_v1_indexer():
            max_pools = vllm_config.model_config.max_model_len // self.kpool
            self.index_op = _build_kpool_index_op(self.kpool, self.select_k,
                                                  self.width, max_pools,
                                                  self.n_heads)
        else:
            if self.kpool != POOL_KEY_KPOOL:
                raise ValueError(
                    f"pool key cache is laid out for kpool {POOL_KEY_KPOOL}, "
                    f"got {self.kpool}")
            self.key_cache = Glm5NextPoolKeyCache(
                prefix=f"{prefix}.key_cache", cache_config=cache_config)
            self.index_op = _build_kpool_index_v2_op(self.kpool,
                                                     self.select_k, self.width)

    def forward(self, hidden_states: torch.Tensor,
                q_resid: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        metadata = None
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata.get(self.k_cache.prefix)
        cache = self.k_cache.kv_cache
        key_metadata = None
        if self.key_cache is not None and isinstance(attn_metadata, dict):
            key_metadata = attn_metadata.get(self.key_cache.prefix)
        if (not isinstance(metadata, AttentionMetadata) or cache.numel() == 0
                or (self.key_cache is not None and
                    (not isinstance(key_metadata, AttentionMetadata)
                     or self.key_cache.kv_cache.numel() == 0))):
            # Profiling / warmup pass before the caches are bound. Returning
            # the buffer unchanged keeps the graph shape stable.
            return self.topk_indices_buffer

        q = self.wq_b(q_resid)[0].view(num_tokens, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(hidden_states)[0])
        gate = torch.nn.functional.linear(hidden_states,
                                          self.index_kpool_compress_gate)
        weights = self.weights_proj(hidden_states)[0].float() * (
            self.n_heads**-0.5)

        common = (
            k.to(cache.dtype),
            gate.to(cache.dtype),
            q.float(),
            weights,
            self.index_kpool_compress_ape.float(),
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
        )
        if self.key_cache is None:
            topk_indices = self.index_op(cache, *common)
        else:
            topk_indices = self.index_op(cache, self.key_cache.kv_cache,
                                         *common,
                                         metadata.request_distribution,
                                         key_metadata.block_tables)

        # Returned directly rather than written into `topk_indices_buffer`:
        # the GLM MLA consumes this tensor itself, and updating the shared
        # `[max_num_batched_tokens, width]` buffer in place cost three full
        # copies of it per layer (1.5 ms of a decode step).
        return topk_indices


__all__ = [
    "Glm5NextIndexer", "Glm5NextIndexerCache", "Glm5NextPoolKeyCache",
    "dense_dsa_fallback"
]
