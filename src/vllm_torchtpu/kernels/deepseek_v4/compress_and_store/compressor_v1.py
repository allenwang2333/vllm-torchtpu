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
"""DeepSeek V4 Compressor Layer forward pass implementation."""

import functools

import jax
import jax.numpy as jnp

import vllm_torchtpu.kernels.deepseek_v4.compress_and_store.config as config
import vllm_torchtpu.kernels.deepseek_v4.compress_and_store.kernel as compress_kernel
import vllm_torchtpu.kernels.deepseek_v4.compress_and_store.proj_and_save_state as proj_kernel  # noqa: E501

_GATHER_1D_DN = jax.lax.GatherDimensionNumbers(
    offset_dims=(),
    collapsed_slice_dims=(0,),
    start_index_map=(0,),
)
_SCATTER_1D_DN = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,),
)
_PROMISE_IN_BOUNDS = jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS
_FILL_OR_DROP = jax.lax.GatherScatterMode.FILL_OR_DROP


def block_table_row_stride(block_table: jax.Array, num_reqs: int) -> int:
    """Per-request row stride of a flattened [num_reqs * max_blocks] table."""
    shape = block_table.shape
    assert len(shape) == 1
    stride, rem = divmod(shape[0], num_reqs)
    assert rem == 0
    return stride


@functools.lru_cache(maxsize=64)
def _cached_page_geometry(
    head_dim: int,
    overlap: bool,
    physical_page_size: int,
    state_physical_page_size: int,
    state_block_size: int,
    rope_head_dim: int,
    compress_ratio: int,
) -> tuple[int, int, int, int, int]:
    cfgs = config.Configs.make(
        config.select_mode(head_dim, overlap),
        size_n=1,
        physical_page_size=physical_page_size,
        state_physical_page_size=state_physical_page_size,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
    )
    page_size = cfgs.state_physical_page_size
    state_rows_per_token = cfgs.state_rows_per_token
    kv_block_size = cfgs.kv_block_size
    kv_stride = cfgs.kv_stride
    kv_page_stride = kv_block_size * kv_stride
    assert state_block_size <= cfgs.state_page_capacity, (
        f"state cache block_size {state_block_size} overruns the "
        f"{cfgs.dims.mode.value} state cache geometry, "
        f"which holds {cfgs.state_page_capacity} token states per page "
    )
    return (
        page_size,
        state_rows_per_token,
        kv_block_size,
        kv_stride,
        kv_page_stride,
    )


def derive_metadata(
    positions: jax.Array,
    block_table: jax.Array,
    query_start_loc: jax.Array,
    kv_block_table: jax.Array,
    cache: jax.Array,
    compress_ratio: int,
    state_block_size: int,
    head_dim: int,
    overlap: bool,
    cos_sin_cache: jax.Array | None,
    state_cache: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Derives token_to_req_indices, slot_mapping_slots, and kv_slot_mapping.

    The page geometry is taken from ``config.Configs`` keyed on
    ``cache.shape[1]`` (compressed KV) and ``state_cache.shape[1]`` (state),
    which is the same derivation ``compress_norm_rope_store`` and
    ``proj_and_save_state`` use. It must not be recomputed from a separate model
    here: the three only agree for CSA, and diverge by 2x (HCA) and 4x (indexer)
    otherwise, which silently scatters state and compressed-KV writes across the
    wrong pages.

    Args:
      positions: [num_tokens]. Logical position of each token in its request.
      block_table: [num_reqs * max_blocks]. Flattened row-major page table for the
        state cache.
      query_start_loc: [num_reqs + 1]. Cumulative sum of query lengths.
      kv_block_table: [num_reqs * max_kv_blocks]. Flattened row-major page table
        for the compressed KV cache.
      cache: [num_pages, physical_page_size, 4, lanes] uint8, or CSA's
        [num_pages, physical_page_size, 128] int32 NoPE array. The
        compressed-KV buffer; only its shape is read here.
      compress_ratio: Compression ratio (e.g. 4 for CSA, 128 for HCA).
      state_block_size: Block size vLLM used to build ``block_table``. One page
        must be able to hold that many token states; it may hold more with the
        tail rows unused.
      head_dim: Dimensionality of attention heads.
      overlap: Whether to use overlap (CSA path).
      cos_sin_cache: [max_pos, rope_head_dim] or None. RoPE cos/sin cache.
      state_cache: [num_pages, state_page_size, 4, lanes] uint8, or None when the
        state shares ``cache``'s buffer. Only its shape is read here.
    Returns:
      token_to_req_indices: [num_tokens]. Request index for each token.
      slot_mapping_slots: [num_tokens]. Physical row index in the state cache.
      kv_slot_mapping: [num_tokens]. Sub-slot index in the compressed KV cache
        (``// tokens_in_second_minor`` gives the physical row).
    """
    num_tokens = positions.shape[0]
    rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0
    state_source = cache if state_cache is None else state_cache

    # Rows per page, rows per token's state, and the KV sub-slot packing.
    # The block tables are paged at vLLM's state-cache block size, and the
    # kernel indexes a page with `position % state_block_size`, so a page must
    # physically hold that many token states. Fewer is a bug; more is the
    # deliberate slack left when a mode is floored to its peer's page size.
    (
        page_size,
        state_rows_per_token,
        kv_block_size,
        kv_stride,
        kv_page_stride,
    ) = _cached_page_geometry(
        head_dim,
        overlap,
        cache.shape[1],
        state_source.shape[1],
        state_block_size,
        rope_head_dim,
        compress_ratio,
    )

    # 2. Map tokens to request indices (Handles Ragged Batch)
    batch_size = query_start_loc.shape[0] - 1
    if batch_size <= 1:
        token_to_req_indices = jax.lax.full((num_tokens,), 0, jnp.int32)
    else:
        q_ends = jax.lax.slice_in_dim(query_start_loc, 1, batch_size)
        token_indices = jax.lax.broadcasted_iota(
            query_start_loc.dtype, (1, num_tokens), 1
        )
        token_to_req_indices = jnp.sum(
            jax.lax.le(jax.lax.expand_dims(q_ends, (1,)), token_indices),
            axis=0,
            dtype=jnp.int32,
        )
    req = token_to_req_indices

    block_table_stride = block_table_row_stride(block_table, batch_size)
    kv_block_table_stride = block_table_row_stride(kv_block_table, batch_size)
    req_state_base = req * jnp.int32(block_table_stride)
    req_kv_base = (
        req_state_base
        if kv_block_table_stride == block_table_stride
        else req * jnp.int32(kv_block_table_stride)
    )

    # 3. State Cache Slot Mapping (Virtual -> Physical)
    pos_i32 = positions.astype(jnp.int32)
    state_bs_i32 = jnp.int32(state_block_size)
    state_page_idx = jax.lax.div(pos_i32, state_bs_i32)
    state_page_offset = jax.lax.rem(pos_i32, state_bs_i32)
    state_page_numbers = jax.lax.gather(
        block_table,
        jax.lax.expand_dims(req_state_base + state_page_idx, (1,)),
        _GATHER_1D_DN,
        (1,),
        mode=_PROMISE_IN_BOUNDS,
    )
    slot_mapping_slots = state_page_numbers * jnp.int32(
        page_size
    ) + state_page_offset * jnp.int32(state_rows_per_token)

    # 4. Compressed KV Cache Slot Mapping (Virtual -> Physical)
    cr_i32 = jnp.int32(compress_ratio)
    kv_bs_i32 = jnp.int32(kv_block_size)
    kv_idx = jax.lax.div(pos_i32, cr_i32)
    kv_page_idx = jax.lax.div(kv_idx, kv_bs_i32)
    kv_page_offset = jax.lax.rem(kv_idx, kv_bs_i32)

    kv_page_number = jax.lax.gather(
        kv_block_table,
        jax.lax.expand_dims(req_kv_base + kv_page_idx, (1,)),
        _GATHER_1D_DN,
        (1,),
        mode=_PROMISE_IN_BOUNDS,
    )

    kv_slot_mapping = kv_page_number * jnp.int32(
        kv_page_stride
    ) + kv_page_offset * jnp.int32(kv_stride)

    is_boundary = jax.lax.eq(pos_i32 - kv_idx * cr_i32, jnp.int32(compress_ratio - 1))
    kv_slot_mapping = jax.lax.select(
        is_boundary,
        kv_slot_mapping,
        jax.lax.full(kv_slot_mapping.shape, -1, kv_slot_mapping.dtype),
    )

    return token_to_req_indices, slot_mapping_slots, kv_slot_mapping


def _get_packing_indices(keep, size):
    order = jnp.nonzero(keep, size=size, fill_value=-1)[0]
    valid = jax.lax.ge(order, 0)
    safe = jax.lax.select(valid, order, jax.lax.full(order.shape, 0, order.dtype))
    return safe, valid


def _pack_array(x, safe, valid, default_value):
    gathered = jax.lax.gather(
        x,
        jax.lax.expand_dims(safe, (1,)),
        _GATHER_1D_DN,
        (1,),
        mode=_PROMISE_IN_BOUNDS,
    )
    return jax.lax.select(
        valid, gathered, jax.lax.full(gathered.shape, default_value, gathered.dtype)
    )


def prepare_boundary_batch(
    positions: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    is_boundary: jax.Array,
    is_decode_token: jax.Array,
    compress_ratio: int,
    num_reqs: int,
    rope_pack: int = 4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generates a combined batch of boundary tokens in a single scatter.

    Layout: [ decode_packed (0..K-1) | pad (K..K_aligned-1) |
    prefill_row_tiled (K_aligned..) | pad ]
    """
    num_tokens = positions.shape[0]
    max_prefill_boundary = num_tokens // compress_ratio + 1 + num_reqs

    # 1. Get packing indices for all boundary tokens
    safe, valid = _get_packing_indices(is_boundary, num_tokens)
    safe_1d = jax.lax.expand_dims(safe, (1,))

    # 2. Pack inputs and the decode mask.
    # Invalid slots map to dest == output_size (dropped by scatter), so pos/req
    # only need gather without a redundant jnp.where(valid, ..., 0).
    pos = jax.lax.gather(
        positions, safe_1d, _GATHER_1D_DN, (1,), mode=_PROMISE_IN_BOUNDS
    )
    req = jax.lax.gather(
        token_to_req_indices,
        safe_1d,
        _GATHER_1D_DN,
        (1,),
        mode=_PROMISE_IN_BOUNDS,
    )
    kv_raw = jax.lax.gather(
        kv_slot_mapping, safe_1d, _GATHER_1D_DN, (1,), mode=_PROMISE_IN_BOUNDS
    )
    kv = jax.lax.select(valid, kv_raw, jax.lax.full(kv_raw.shape, -1, kv_raw.dtype))
    is_decode = jax.lax.bitwise_and(
        valid,
        jax.lax.gather(
            is_decode_token,
            safe_1d,
            _GATHER_1D_DN,
            (1,),
            mode=_PROMISE_IN_BOUNDS,
        ),
    )

    valid_token = jax.lax.ge(kv, 0)
    is_prefill = jax.lax.bitwise_and(valid_token, jax.lax.bitwise_not(is_decode))

    # 3. Calculate row-tiling ONLY for prefill tokens
    rope_pack_i32 = jnp.int32(rope_pack)
    row = jax.lax.select(
        is_prefill,
        jax.lax.div(kv.astype(jnp.int32), rope_pack_i32),
        jax.lax.full(kv.shape, -1, jnp.int32),
    )
    row_diff = jax.lax.ne(
        jax.lax.slice_in_dim(row, 1, num_tokens),
        jax.lax.slice_in_dim(row, 0, num_tokens - 1),
    )
    starts_row = jax.lax.bitwise_and(
        is_prefill, jax.lax.pad(row_diff, True, ((1, 0, 0),))
    )

    tile_index = jax.lax.cumsum(starts_row.astype(jnp.int32), 0) - jnp.int32(1)

    index = jax.lax.broadcasted_iota(jnp.int32, (num_tokens,), 0)
    run_start = jax.lax.cummax(
        jax.lax.select(starts_row, index, jax.lax.full(index.shape, 0, jnp.int32))
    )
    pos_within_tile = index - run_start

    # 4. Calculate alignment offset for prefill block
    K = jnp.sum(jax.lax.bitwise_and(is_boundary, is_decode_token), dtype=jnp.int32)
    K_aligned = jax.lax.bitwise_and(K + jnp.int32(3), jnp.int32(-4))

    # 5. Combine destination indices
    prefill_size = rope_pack * max_prefill_boundary
    output_size = num_tokens + prefill_size

    dest_prefill = K_aligned + tile_index * rope_pack_i32 + pos_within_tile
    dest = jax.lax.select(is_decode, index, dest_prefill)
    dest = jax.lax.select(
        valid_token, dest, jax.lax.full(dest.shape, output_size, dest.dtype)
    )
    dest_1d = jax.lax.expand_dims(dest, (1,))

    # 6. Scatter to output once
    def scatter(default_val, src):
        return jax.lax.scatter(
            jax.lax.full((output_size,), default_val, src.dtype),
            dest_1d,
            src,
            _SCATTER_1D_DN,
            mode=_FILL_OR_DROP,
        )

    return (
        scatter(0, pos),
        scatter(0, req),
        scatter(-1, kv),
    )


def compressor_forward(
    hidden_states: jax.Array,  # [num_tokens, hidden_size] fp32
    wkv_wgate: jax.Array,  # [2*coff*head_dim, hidden_size] fp32
    ape: jax.Array,  # [compress_ratio, coff*head_dim] fp32
    norm_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    positions: jax.Array,  # [num_tokens] int
    block_table: jax.Array,  # [num_reqs * max_blocks] int
    query_start_loc: jax.Array,  # [num_reqs + 1] int
    kv_block_table: jax.Array,  # [num_reqs * max_kv_blocks] int
    cache: jax.Array,  # [num_pages, page_size, 4, 128] u8 / CSA [..., 128] i32
    rope_cache: jax.Array,  # CSA [num_pages, page_size // 4, 128] int32
    distribution: jax.Array,  # i32[3]
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
    state_cache: jax.Array | None = None,  # [num_pages, state_page_size, 4, 128] uint8
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Projects, saves state, then compresses and stores the boundary records."""
    num_tokens = positions.shape[0]
    num_reqs = query_start_loc.shape[0] - 1
    shares_buffer = state_cache is None
    bt_stride = block_table_row_stride(block_table, num_reqs)

    token_to_req_indices, slot_mapping_slots, kv_slot_mapping = derive_metadata(
        positions=positions,
        block_table=block_table,
        query_start_loc=query_start_loc,
        kv_block_table=kv_block_table,
        cache=cache,
        compress_ratio=compress_ratio,
        state_block_size=state_block_size,
        head_dim=head_dim,
        overlap=overlap,
        cos_sin_cache=cos_sin_cache,
        state_cache=state_cache,
    )

    token_index = jax.lax.broadcasted_iota(query_start_loc.dtype, (num_tokens,), 0)
    dist2_idx = jax.lax.expand_dims(jax.lax.slice_in_dim(distribution, 2, 3), (1,))
    real_limit = jax.lax.squeeze(
        jax.lax.gather(
            query_start_loc,
            dist2_idx,
            _GATHER_1D_DN,
            (1,),
            mode=_PROMISE_IN_BOUNDS,
        ),
        (0,),
    )
    is_real_token = jax.lax.lt(token_index, real_limit)
    slot_mapping_slots = jax.lax.select(
        is_real_token,
        slot_mapping_slots,
        jax.lax.full(slot_mapping_slots.shape, -1, slot_mapping_slots.dtype),
    )

    updated_state = proj_kernel.proj_and_save_state(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=positions,
        slot_mapping=slot_mapping_slots,
        cache=cache if shares_buffer else state_cache,
        compress_ratio=compress_ratio,
    )
    if shares_buffer:
        cache = updated_state
    else:
        state_cache = updated_state

    dist0_idx = jax.lax.expand_dims(jax.lax.slice_in_dim(distribution, 0, 1), (1,))
    decode_limit = jax.lax.squeeze(
        jax.lax.gather(
            query_start_loc,
            dist0_idx,
            _GATHER_1D_DN,
            (1,),
            mode=_PROMISE_IN_BOUNDS,
        ),
        (0,),
    )
    is_decode_token = jax.lax.lt(token_index, decode_limit)
    is_boundary = jax.lax.bitwise_and(jax.lax.ge(kv_slot_mapping, 0), is_real_token)

    boundary_positions, boundary_req, boundary_kv_slot = prepare_boundary_batch(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        is_boundary=is_boundary,
        is_decode_token=is_decode_token,
        compress_ratio=compress_ratio,
        num_reqs=num_reqs,
    )

    cache, rope_cache = compress_kernel.compress_norm_rope_store(
        cache,
        boundary_positions,
        block_table,
        boundary_req,
        boundary_kv_slot,
        norm_weight,
        block_table_stride=bt_stride,
        state_block_size=state_block_size,
        state_cache=state_cache,
        rope_cache=rope_cache,
        cos_sin_cache=cos_sin_cache,
        compress_ratio=compress_ratio,
        overlap=overlap,
        quant_block=quant_block,
        rms_eps=rms_eps,
    )

    if shares_buffer:
        return cache, rope_cache, cache

    # `compress_norm_rope_store` READS `state_cache` (its read-only state
    # operand) but emits no new version of it, so the value published here is
    # the pre-read one straight off `proj_and_save_state`. The DSv4 KV-cache
    # overlay hosts another layer's SWA cache on this same physical array, and
    # that kernel takes whole-buffer ownership via `input_output_aliases`.
    # Nothing orders the read against that write, so XLA's copy-insertion
    # AddCopiesToResolveInterference cannot prove the read completes first and
    # insert the cache copy. Publishing a version that is downstream of
    # the read supplies the missing ordering edge.
    state_cache, cache = jax.lax.optimization_barrier((state_cache, cache))
    return cache, rope_cache, state_cache
