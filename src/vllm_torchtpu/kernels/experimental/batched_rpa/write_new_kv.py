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
"""Write a decode step's new KV tokens into this DCP rank's cache shard.

First half of the DCP decode single pass: once every rank has stored the new
tokens it owns, one CACHE_ONLY pass with `skip_kv_update` covers cache and
new tokens, replacing the CACHE_ONLY + NEW_TOKENS_ONLY + merge sequence.

Handles decode and speculative verify windows: sequences
[0, distribution[0]) with up to `max_q_len` (<= page_size) new tokens each.
Pages interleave across ranks, so such a window spans at most two pages and
two adjacent pages belong to different ranks: each rank writes at most one
run of consecutive tokens per sequence, inside one page. No run table or
compaction is needed; each sequence's run is a few scalars.

The new tokens are formatted by `wrapper.prepare_inputs`, so layouts, fp8
packing and per-token scale channels match what the RPA kernel itself would
write. An XLA scatter into the pool is not an option: it copies the whole
pool every call (see kernels/partial_kv_update.py).

  HEAD_ALONG_SUBLANE  a token is one cache row: one HBM->HBM DMA per token.
  SEQ_ALONG_LANE      a token is one lane of a 128-lane tile and is not
                      DMA-addressable. A run is split at tile boundaries
                      (<= 2 parts); each part is a read-modify-write of its
                      tile, blending the source lanes in with a roll + select
                      (as in deepseek_v4/streamindex_topk/kv_cache_append.py).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_LANES = 128
# Sequences per grid step. Every DMA of a step is in flight at once.
_HAS_GROUP = 16
_SAL_GROUP = 8


def _owned_run(seq, refs, *, cp_group_size, cp_rank, page_size, pages_per_seq):
    """This rank's run of the sequence's new tokens.

    Returns (n, src_tok, page, offset): `n` tokens starting at new-token row
    `src_tok` land at `offset` of physical page `page`; n == 0 means none.
    """
    kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref = refs
    valid = seq < distribution_ref[0]
    safe = jnp.where(valid, seq, 0)
    q_start = cu_q_lens_ref[safe]
    q_len = jnp.where(valid, cu_q_lens_ref[safe + 1] - q_start, 0)
    first = kv_lens_ref[safe] - q_len  # global position of the first new token
    last = first + q_len - 1
    p0, p1 = first // page_size, last // page_size
    own0 = p0 % cp_group_size == cp_rank
    own1 = (p1 != p0) & (p1 % cp_group_size == cp_rank)
    start = jnp.where(own0, first, p1 * page_size)
    end = jnp.where(own0, jnp.minimum(last + 1, (p0 + 1) * page_size), last + 1)
    n = jnp.where(valid & (own0 | own1) & (q_len > 0), end - start, 0)
    vpage = start // page_size
    page = page_indices_ref[safe * pages_per_seq + vpage // cp_group_size]
    return n, q_start + (start - first), page, start % page_size


def _has_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    distribution_ref,
    cp_rank_ref,
    new_kv_hbm,
    _cache_in_hbm,
    cache_hbm,
    sem,
    *,
    cp_group_size,
    pages_per_seq,
    max_q_len,
):
    page_size = cache_hbm.shape[1]
    base = pl.program_id(0) * _HAS_GROUP
    refs = (kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref)
    run = functools.partial(
        _owned_run,
        refs=refs,
        cp_group_size=cp_group_size,
        cp_rank=cp_rank_ref[0],
        page_size=page_size,
        pages_per_seq=pages_per_seq,
    )

    def copies(fn):
        for i in range(_HAS_GROUP):
            n, src_tok, page, offset = run(base + i)
            for j in range(max_q_len):
                copy = pltpu.make_async_copy(
                    new_kv_hbm.at[pl.ds(src_tok + j, 1)],
                    cache_hbm.at[page, pl.ds(offset + j, 1)],
                    sem.at[i * max_q_len + j],
                )
                pl.when(j < n)(functools.partial(fn, copy))

    copies(lambda c: c.start())
    copies(lambda c: c.wait())


def _sal_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    distribution_ref,
    cp_rank_ref,
    new_kv_hbm,
    _cache_in_hbm,
    cache_hbm,
    tile_vmem,
    src_vmem,
    sem,
    *,
    cp_group_size,
    pages_per_seq,
):
    """Read-modify-write of the (<= 2) cache tiles each run touches."""
    page_size = cache_hbm.shape[4]
    _, heads_x2, dim_packed, packing, _ = cache_hbm.shape
    rows = heads_x2 * dim_packed
    base = pl.program_id(0) * _SAL_GROUP
    refs = (kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref)

    def parts():
        """Yields (slot, n, src_tok, page, lane) for each run part."""
        for i in range(_SAL_GROUP):
            n, src_tok, page, offset = _owned_run(
                base + i,
                refs,
                cp_group_size=cp_group_size,
                cp_rank=cp_rank_ref[0],
                page_size=page_size,
                pages_per_seq=pages_per_seq,
            )
            n0 = jnp.minimum(n, _LANES - offset % _LANES)
            yield 2 * i, n0, src_tok, page, offset
            yield 2 * i + 1, n - n0, src_tok + n0, page, offset + n0

    def tile_copy(slot, page, lane, *, to_hbm):
        aligned = pl.multiple_of((lane // _LANES) * _LANES, _LANES)
        hbm = cache_hbm.at[page, :, :, :, pl.ds(aligned, _LANES)]
        vmem = tile_vmem.at[slot]
        src, dst = (vmem, hbm) if to_hbm else (hbm, vmem)
        return pltpu.make_async_copy(src, dst, sem.at[0, slot])

    def src_copy(slot, src_tok):
        # Two source tiles cover any <= 128 lanes starting in the first; the
        # wrapper pads one spare tile so the second always exists.
        aligned = pl.multiple_of((src_tok // _LANES) * _LANES, _LANES)
        return pltpu.make_async_copy(
            new_kv_hbm.at[:, :, :, pl.ds(aligned, 2 * _LANES)],
            src_vmem.at[slot],
            sem.at[1, slot],
        )

    for slot, n, src_tok, page, lane in parts():

        @pl.when(n > 0)
        def _(slot=slot, n=n, src_tok=src_tok, page=page, lane=lane):
            tile_copy(slot, page, lane, to_hbm=False).start()
            src_copy(slot, src_tok).start()

    lane_iota = lax.broadcasted_iota(jnp.int32, (rows, _LANES), 1)

    def as_words(x):
        return pltpu.bitcast(x.reshape(rows * packing, _LANES), jnp.uint32)

    for slot, n, src_tok, page, lane in parts():

        @pl.when(n > 0)
        def _(slot=slot, n=n, src_tok=src_tok, page=page, lane=lane):
            tile_copy(slot, page, lane, to_hbm=False).wait()
            src_copy(slot, src_tok).wait()
            # Source lanes [src_start, src_start + n) of the two-tile window
            # go to tile lanes [dst_start, dst_start + n): roll both tiles by
            # the lane delta, take a's lanes until a runs out, then b's.
            src_start = src_tok % _LANES
            dst_start = lane % _LANES
            delta = (dst_start - src_start) % _LANES
            a = as_words(src_vmem[slot, :, :, :, pl.ds(0, _LANES)])
            b = as_words(src_vmem[slot, :, :, :, pl.ds(_LANES, _LANES)])
            a = pltpu.roll(a, shift=delta, axis=1)
            b = pltpu.roll(b, shift=delta, axis=1)
            src = jnp.where(lane_iota < dst_start + (_LANES - src_start), a, b)
            dst = as_words(tile_vmem[slot])
            mask = (lane_iota >= dst_start) & (lane_iota < dst_start + n)
            out = jnp.where(mask, src, dst)
            tile_vmem[slot] = pltpu.bitcast(out, cache_hbm.dtype).reshape(
                heads_x2, dim_packed, packing, _LANES
            )
            tile_copy(slot, page, lane, to_hbm=True).start()

    for slot, n, src_tok, page, lane in parts():

        @pl.when(n > 0)
        def _(slot=slot, n=n, src_tok=src_tok, page=page, lane=lane):
            tile_copy(slot, page, lane, to_hbm=True).wait()


@functools.partial(
    jax.jit,
    static_argnames=("cp_group_size", "kv_layout", "max_q_len"),
    donate_argnames=("kv_cache",),
)
def write_new_kv(
    kv_cache: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    cp_rank: jax.Array,
    *,
    cp_group_size: int,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    max_q_len: int = 1,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
) -> jax.Array:
    """Stores this rank's share of the step's new tokens, in place.

    Args:
      kv_cache: This rank's cache shard, in `kv_layout`. Donated.
      keys, values: [T, num_kv_heads, head_dim] new tokens, already in the
        cache dtype (quantized by the caller, as for the RPA kernel).
      kv_lens: [max_num_seqs] sequence lengths including the new tokens.
      page_indices: [max_num_seqs * pages_per_seq] local page table.
      cu_q_lens: [max_num_seqs + 1] new-token offsets per sequence.
      distribution: [3]; sequences [0, distribution[0]) are written.
      cp_rank: [1] int32 rank of this device in the DCP group.
      cp_group_size: DCP group size; KV interleaves across it by page.
      kv_layout: Cache layout.
      max_q_len: Upper bound on new tokens per sequence (decode_query_size).
      dynamic_k_scale, dynamic_v_scale: Per-token scales, if the cache keeps
        them.

    Returns:
      The cache with the new tokens written, aliased to `kv_cache`.
    """
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache.shape[4]
    else:
        page_size = kv_cache.shape[1]
    if max_q_len > page_size:
        raise ValueError(f"{max_q_len=} must not exceed {page_size=}")
    if cp_group_size < 2:
        raise ValueError(f"{cp_group_size=} must be at least 2")
    # Only the KV half of prepare_inputs is used; keys stand in for queries.
    _, new_kv = wrapper.prepare_inputs(
        keys,
        keys,
        values,
        keys.dtype,
        kv_cache.dtype,
        dynamic_k_scale,
        dynamic_v_scale,
        kv_layout=kv_layout,
        page_size=page_size,
    )
    num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // num_seqs
    scalars = (kv_lens, page_indices, cu_q_lens, distribution, cp_rank)
    common = dict(cp_group_size=cp_group_size, pages_per_seq=pages_per_seq)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        # One spare tile past the last token keeps every two-tile source
        # window in bounds.
        new_kv = jnp.pad(new_kv, ((0, 0),) * 3 + ((0, _LANES),))
        _, heads_x2, dim_packed, packing, _ = kv_cache.shape
        group = _SAL_GROUP
        kernel = functools.partial(_sal_kernel, **common)
        scratch = [
            pltpu.VMEM(
                (2 * group, heads_x2, dim_packed, packing, _LANES), kv_cache.dtype
            ),
            pltpu.VMEM(
                (2 * group, heads_x2, dim_packed, packing, 2 * _LANES),
                kv_cache.dtype,
            ),
            pltpu.SemaphoreType.DMA((2, 2 * group)),
        ]
    else:
        group = _HAS_GROUP
        kernel = functools.partial(_has_kernel, max_q_len=max_q_len, **common)
        scratch = [pltpu.SemaphoreType.DMA((group * max_q_len,))]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(pl.cdiv(num_seqs, group),),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        input_output_aliases={len(scalars) + 1: 0},
        name=f"dcp_write_new_kv_{kv_layout.symbol}",
    )(*scalars, new_kv, kv_cache)
