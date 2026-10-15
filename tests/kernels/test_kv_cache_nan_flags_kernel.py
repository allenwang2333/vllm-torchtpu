# SPDX-License-Identifier: Apache-2.0
"""Only the scheduled requests' live input cache may set a layer flag."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.kv_cache_nan_flags import (
    attention_cache_has_nan,
    dense_state_has_nan,
    pooled_state_has_nan,
)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn])
def test_attention_reads_only_live_prefix_of_scheduled_requests(dtype):
    # req0 reads page 3 then the first token of page 1. req1 has no old KV;
    # req2 is padding. Their NaNs, page 0 and page 2 must be ignored.
    cache = jnp.zeros((4, 4, 2, 8), dtype)
    cache = cache.at[0].set(jnp.nan).at[2].set(jnp.nan)
    cache = cache.at[1, 1:].set(jnp.nan)
    tables = jnp.array([[3, 1], [2, 0], [0, 0]], jnp.int32)
    seq_lens = jnp.array([9, 2, 9], jnp.int32)
    starts = jnp.array([0, 4, 6, 6], jnp.int32)
    inspect = jax.jit(attention_cache_has_nan)
    assert not bool(inspect(cache, tables, seq_lens, starts))
    assert bool(inspect(cache.at[1, 0, 1, 7].set(jnp.nan), tables, seq_lens, starts))
    assert bool(inspect(cache.at[3, 3, 0, 0].set(jnp.nan), tables, seq_lens, starts))


@pytest.mark.parametrize("pooled", [False, True])
def test_state_selects_actual_checkpoint_and_ignores_fresh_and_padded_requests(pooled):
    conv = jnp.zeros((8, 2, 4), jnp.bfloat16).at[2].set(jnp.nan)
    ssm = jnp.zeros((8, 1, 4, 4)).at[6].set(jnp.nan)
    slots = jnp.array([2, 6, 0], jnp.int32)
    offsets = jnp.zeros(8, jnp.int32).at[2].set(1)
    ckpts = jnp.array([[2, 5], [6, 7], [0, 0]], jnp.int32) if pooled else None
    seq_lens = jnp.array([12, 4, 99], jnp.int32)
    starts = jnp.array([0, 4, 8, 8], jnp.int32)
    inspect = jax.jit(dense_state_has_nan)
    assert not bool(inspect((conv, ssm), slots, seq_lens, starts, offsets, ckpts))
    read_slot = 5 if pooled else 3
    bad = ssm.at[read_slot, 0, 3, 3].set(jnp.nan)
    assert bool(inspect((conv, bad), slots, seq_lens, starts, offsets, ckpts))


@pytest.mark.parametrize("bf16_ssm", [False, True])
@pytest.mark.parametrize("lane_split", [1, 2])
def test_pooled_states_are_typed_bytes_and_exclude_region_padding(bf16_ssm, lane_split):
    from vllm_torchtpu.kernels.gdn.v3.config import StateRegion, StateSourcePlan

    # Encode fp32 along the second-minor (row) axis, exactly the pool's typed
    # view contract. 0x7f bytes look like FP8 NaN but hold a finite FP32 value.
    words = (
        np.full((2, 8), 0x3F7F, np.uint16)
        if bf16_ssm
        else np.full((1, 8), 0x3F80007F, np.uint32)
    )
    raw_ssm = words.T.copy().view(np.uint8).reshape(8, 4).T
    raw = np.zeros((3, 8, 8), np.uint8)
    raw[1, :4] = raw_ssm
    # This fixture uses 4 FP8 rows per FP32 row, so give SSM 4 rows and put
    # Conv after it; trailing rows and unused BF16 rows contain arbitrary bytes.
    plan = StateSourcePlan(
        stride=1,
        recurrent=StateRegion(
            0,
            1,
            0,
            4,
            jnp.dtype(jnp.uint32 if bf16_ssm else jnp.float32),
            lane_split,
            1,
        ),
        conv=StateRegion(0, 1, 4, 4, jnp.dtype(jnp.bfloat16), 1, 1),
    )
    raw[1, 6:] = 0x7F
    pool = jax.lax.bitcast_convert_type(jnp.array(raw), jnp.float8_e4m3fn)
    inspect = jax.jit(partial(pooled_state_has_nan, plan=plan))
    slots = jnp.array([1], jnp.int32)
    seq_lens, starts = jnp.array([2]), jnp.array([0, 1])
    assert not bool(inspect(pool, slots, seq_lens, starts))
    if lane_split > 1:
        outside = raw.copy()
        outside[1, 0:4, 7] = np.array([0, 0, 0xC0, 0x7F], np.uint8)
        unused = jax.lax.bitcast_convert_type(jnp.array(outside), jnp.float8_e4m3fn)
        assert not bool(inspect(unused, slots, seq_lens, starts))
    # FP32 NaN, last live lane. Unrelated FP8 classifications are immaterial.
    if bf16_ssm:
        raw[1, 2:4, 8 // lane_split - 1] = np.array([0xC0, 0x7F], np.uint8)
    else:
        raw[1, 0:4, 8 // lane_split - 1] = np.array([0, 0, 0xC0, 0x7F], np.uint8)
    bad = jax.lax.bitcast_convert_type(jnp.array(raw), jnp.float8_e4m3fn)
    assert bool(inspect(bad, slots, seq_lens, starts))


def test_attention_prefix_boundaries_exhaustively():
    inspect = jax.jit(attention_cache_has_nan)
    tables, starts = jnp.array([[2, 1]]), jnp.array([0, 4])
    for prefix in range(9):
        for position in range(8):
            page = 2 if position < 4 else 1
            cache = jnp.zeros((3, 4, 1, 1)).at[page, position % 4, 0, 0].set(jnp.nan)
            assert bool(inspect(cache, tables, jnp.array([prefix + 4]), starts)) == (
                position < prefix
            )


@pytest.mark.parametrize(
    "layout", ["packed", "seq_along_lane", "unified", "block_major"]
)
def test_attention_bridge_layouts_preserve_live_token_and_feature_mask(layout):
    from vllm_torchtpu.layers.adapter.custom_ops.kv_cache_nan_flags_op import (
        inspect_attention_cache,
    )

    native = (1, 4, 1, 4, 8)
    shape = (4, *native[1:])
    if layout == "seq_along_lane":
        native = (1, 2, 2, 4, 4)
        shape = (4, *native[1:])
    elif layout == "unified":
        shape = (4, 4, 4, 16)
    elif layout == "block_major":
        shape = (2, 2, *native[1:])
    cache = jnp.zeros(shape, jnp.bfloat16)
    inspect = jax.jit(
        partial(
            inspect_attention_cache,
            native_shape=native,
            num_kv_heads=1,
            head_size=5,
            seq_along_lane=layout == "seq_along_lane",
            block_major=layout == "block_major",
        )
    )
    tables, seq_lens, starts = jnp.array([[3, 0]]), jnp.array([2]), jnp.array([0, 1])
    if layout == "seq_along_lane":
        unused = cache.at[3, 0, 1, 3, 0].set(jnp.nan)
        bad = cache.at[3, 1, 0, 0, 0].set(jnp.nan)
    elif layout == "block_major":
        unused = cache.at[1, 1, 1:].set(jnp.nan)
        bad = cache.at[1, 1, 0, 0, 1, 0].set(jnp.nan)
    else:
        unused = cache.at[3, 1:].set(jnp.nan)
        bad = (
            cache.at[3, 0, 0, 0, 0].set(jnp.nan)
            if cache.ndim == 5
            else cache.at[3, 0, 0, 0].set(jnp.nan)
        )
    assert not bool(inspect(unused, tables, seq_lens, starts))
    assert bool(inspect(bad, tables, seq_lens, starts))
