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
"""Tests for gmm_v2.

Most helper logic in this file is copied from
../tpu-inference/tests/kernels/gmm_test.py and adapted to this repo's pytest
test runner.
"""

import collections
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    Dimensions,
    GmmConfigs,
    InputConfigs,
    MetadataRef,
    TileSizes,
    apply_act_fn,
    fill_metadata,
    gmm_v2,
    interleave_lane,
    situ_and_mul,
)

_GroupConfig = collections.namedtuple(
    "_GroupConfig", ["num_groups", "group_offset", "num_local_groups"])


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _assert_allclose(actual, expected, atol=1e-5, rtol=1e-5) -> None:
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=atol,
                               rtol=rtol)


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_and_mul(linear_beta):
    gate = jnp.asarray([-3.0, 0.5, 7.0])
    up = jnp.asarray([-30.0, 2.0, 40.0])
    expected_gate = 4.0 * jnp.tanh(gate / 4.0) * jax.nn.sigmoid(gate)
    expected_up = (up if linear_beta is None else linear_beta *
                   jnp.tanh(up / linear_beta))

    _assert_allclose(situ_and_mul(gate, up, 4.0, linear_beta),
                     expected_gate * expected_up)


@pytest.mark.parametrize("linear_beta", ["none", "25.0"])
def test_apply_act_fn_dispatches_situ(monkeypatch, linear_beta):
    monkeypatch.setattr(
        "vllm_torchtpu.kernels.megablox.gmm_v2.pltpu.get_tpu_info",
        lambda: SimpleNamespace(num_lanes=1),
    )
    gate = jnp.asarray([-3.0, 0.5, 7.0])
    up = jnp.asarray([-30.0, 2.0, 40.0])
    expected = situ_and_mul(
        gate,
        up,
        beta=4.0,
        linear_beta=None if linear_beta == "none" else float(linear_beta),
    )

    actual = apply_act_fn(interleave_lane(gate, up), f"situ:4.0:{linear_beta}")
    _assert_allclose(actual, expected)


def get_group_sizes(batch_size: int, num_groups: int) -> jax.Array:
    distribution = jax.random.uniform(jax.random.key(0), (num_groups - 1, ),
                                      dtype=jnp.float32)
    distribution = distribution / jnp.sum(distribution)
    group_sizes = jnp.floor(distribution * batch_size).astype(jnp.int32)
    return jnp.append(group_sizes, batch_size - jnp.sum(group_sizes))


def quantize_tensor(x: jax.Array,
                    dtype: jnp.dtype,
                    axis: int = -1,
                    block_size: int = 256):
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = int(dtype_info.max)
        min_val = int(dtype_info.min)
    else:
        dtype_info = jnp.finfo(dtype)
        max_val = float(dtype_info.max)
        min_val = float(dtype_info.min)

    orig_shape = x.shape
    blocked_shape = orig_shape[:axis] + (-1,
                                         block_size) + orig_shape[axis + 1:]
    x_blocked = x.reshape(blocked_shape)

    x_blocked_abs_max = jnp.max(jnp.abs(x_blocked),
                                axis=axis + 1,
                                keepdims=True)
    scale = x_blocked_abs_max / max_val
    x_blocked_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)

    x_q = x_blocked_q.reshape(orig_shape)
    x_q = jnp.nan_to_num(x_q)
    scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
    return x_q, scale


def reference_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
):
    num_tokens = lhs.shape[0]
    num_groups, in_size, out_size = rhs.shape
    assert lhs.shape[1] == in_size

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]

    if rhs_scale is not None:
        num_blocks = rhs_scale.shape[1]
    else:
        num_blocks = 1
    block_size = in_size // num_blocks

    start = 0
    gmm_out = []
    for global_group in range(group_sizes.size):
        group_size = group_sizes[global_group]

        group = global_group - group_offset[0]
        end = min(start + group_size, num_tokens)
        group_size = end - start
        if 0 <= group < num_groups:
            lhs_slice = lhs[start:end]
            rhs_slice = rhs[group]

            out = 0
            for block in range(num_blocks):
                block_start = block * block_size
                block_end = block_start + block_size
                lhs_block = lhs_slice[:, block_start:block_end].astype(
                    jnp.float32)
                rhs_block = rhs_slice[block_start:block_end, :].astype(
                    jnp.float32)

                acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
                if rhs_scale is not None:
                    acc *= rhs_scale[group][block]
                out += acc
            if rhs_bias is not None:
                out = out + rhs_bias[group]
        else:
            out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)

        gmm_out.append(out.astype(lhs.dtype))
        start = end

    return jnp.concat(gmm_out, axis=0)


@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("group_offset", [0, 2])
def test_gmm_matches_reference(has_bias, group_offset):
    _require_tpu()

    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 16
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_local_groups, in_size, out_size),
                            dtype=jnp.bfloat16)
    rhs_bias = None
    if has_bias:
        rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size),
                                     dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(lhs,
                             rhs,
                             group_sizes,
                             rhs_bias=rhs_bias,
                             group_offset=group_offset)

    actual = gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    _assert_allclose(actual, expected)


@pytest.mark.parametrize("weight_dtype", [jnp.int8, jnp.float8_e4m3fn])
@pytest.mark.parametrize("block_size", [256, 512])
def test_gmm_weight_quantized_matches_reference(weight_dtype, block_size):
    _require_tpu()

    batch_size = 128
    in_size = 1024
    out_size = 512
    num_groups = 16
    group_offset = jnp.array(0, dtype=jnp.int32)
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_groups, in_size, out_size),
                             jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs,
                                       weight_dtype,
                                       axis=1,
                                       block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    expected = reference_gmm(lhs,
                             rhs_q,
                             get_group_sizes(batch_size, num_groups),
                             rhs_scale=rhs_scale,
                             group_offset=group_offset)

    actual = gmm_v2(
        lhs,
        rhs_q,
        get_group_sizes(batch_size, num_groups),
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    _assert_allclose(actual, expected, atol=3e-1, rtol=3e-1)


@pytest.mark.parametrize("fuse_act", ["silu", "swigluoai"])
def test_gmm_fused_activation_matches_reference(fuse_act):
    _require_tpu()

    batch_size = 128
    in_size = 512
    out_size = 512
    final_out_size = out_size // 2
    num_groups = 16
    block_size = 512
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_groups, in_size, out_size),
                             jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs,
                                       jnp.int8,
                                       axis=1,
                                       block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    rhs_bias = jax.random.normal(key, (num_groups, 1, out_size),
                                 dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array([0], dtype=jnp.int32)

    lhs_block_size = min(512, in_size)
    lhs_q, lhs_scale_factor = quantize_tensor(lhs,
                                              jnp.int8,
                                              axis=1,
                                              block_size=lhs_block_size)
    lhs_q_blocked = lhs_q.reshape(batch_size, -1,
                                  lhs_block_size).astype(jnp.float32)
    lhs_scale_expanded = jnp.expand_dims(lhs_scale_factor, axis=2)
    lhs_simulated = ((lhs_q_blocked * lhs_scale_expanded).reshape(
        lhs.shape).astype(lhs.dtype))

    raw_expected = reference_gmm(lhs_simulated,
                                 rhs_q,
                                 group_sizes,
                                 rhs_scale=rhs_scale,
                                 rhs_bias=rhs_bias,
                                 group_offset=group_offset)
    raw_gate, raw_up = jnp.split(raw_expected, 2, axis=-1)
    raw_expected = interleave_lane(raw_gate, raw_up)
    expected = apply_act_fn(raw_expected.astype(jnp.float32),
                            fuse_act).astype(lhs.dtype)

    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
        fuse_act=fuse_act,
    ).astype(lhs.dtype)

    assert actual.shape == (batch_size, final_out_size)
    _assert_allclose(actual, expected, atol=4.0, rtol=2.0)


def test_gmm_weight_quantized_block_larger_than_tile_k():
    _require_tpu()

    batch_size = 128
    in_size = 1024
    out_size = 512
    num_groups = 16
    block_size = 1024
    tile_info = TileSizes(tile_m=128,
                          bucket_base=128,
                          tile_k=256,
                          tile_n=out_size)
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_groups, in_size, out_size),
                             jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs,
                                       jnp.float8_e4m3fn,
                                       axis=1,
                                       block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(0, dtype=jnp.int32)

    expected = reference_gmm(lhs,
                             rhs_q,
                             group_sizes,
                             rhs_scale=rhs_scale,
                             group_offset=group_offset)

    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    _assert_allclose(actual, expected, atol=3e-1, rtol=3e-1)


@pytest.mark.parametrize(
    "group_config",
    [
        _GroupConfig(num_groups=16, group_offset=2, num_local_groups=4),
        _GroupConfig(num_groups=16, group_offset=0, num_local_groups=8),
    ],
)
def test_gmm_nonlocal_groups_produce_zeros(group_config):
    _require_tpu()

    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups, group_offset, num_local_groups = group_config
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_local_groups, in_size, out_size),
                            dtype=jnp.bfloat16)
    rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size),
                                 dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(lhs,
                             rhs,
                             group_sizes,
                             rhs_bias=rhs_bias,
                             group_offset=group_offset)

    actual = gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    assert actual.shape == (batch_size, out_size)
    _assert_allclose(actual, expected)


def test_gmm_v2_int4_packed_matches_reference():
    _require_tpu()

    batch_size = 128
    logical_k = 512
    storage_k = logical_k // 8
    out_size = 256
    num_groups = 4
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, logical_k), jnp.bfloat16, -1, 1)

    weight_key, scale_key = jax.random.split(key)
    rhs_int4 = jax.random.randint(
        weight_key,
        (num_groups, logical_k, out_size),
        minval=-8,
        maxval=8,
        dtype=jnp.int32,
    )

    group_size = 64
    num_blocks = logical_k // group_size
    rhs_scale = jax.random.uniform(
        scale_key,
        (num_groups, num_blocks, 1, out_size),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.bfloat16,
    )

    scale_tiled = jnp.repeat(rhs_scale, group_size, axis=1)
    scale_tiled = jnp.squeeze(scale_tiled, axis=2)

    rhs_dequantized = rhs_int4.astype(jnp.bfloat16) * scale_tiled

    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = reference_gmm(
        lhs,
        rhs_dequantized,
        group_sizes,
    )

    rhs_uint4 = (rhs_int4 + 8) & 0x0F

    rhs_reshaped = rhs_uint4.reshape(num_groups, storage_k, 8, out_size)

    shifts = jnp.arange(8, dtype=jnp.int32) * 4
    shifted = rhs_reshaped << shifts[None, None, :, None]
    rhs_packed_int32 = jnp.sum(shifted, axis=2).astype(jnp.int32)

    INT4_SIGN_XOR = -2004318072
    rhs_packed_int32_xor = rhs_packed_int32 ^ INT4_SIGN_XOR

    actual = gmm_v2(
        lhs,
        rhs_packed_int32_xor,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_quant_dtype=jnp.int4,
    )

    assert actual.shape == (batch_size, out_size)
    _assert_allclose(actual, expected, atol=3e-1, rtol=3e-1)


def test_gmm_v2_fill_metadata_static_tiles():
    dims = Dimensions(
        size_m=128,
        size_k=1024,
        size_n=2048,
        size_group=4,
        size_lhs_group=4,
        size_lhs_sublane=128,
    )
    tiles = TileSizes(tile_m=128, tile_k=128, tile_n=128, bucket_base=128)
    lhs_cfgs = InputConfigs(dtype=jnp.bfloat16)
    rhs_cfgs = InputConfigs(dtype=jnp.bfloat16)

    # Dynamic config: 1 tile needed for 10 tokens
    cfgs_dyn = GmmConfigs(
        tiles=tiles,
        dims=dims,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=None,
        static_gm_tiles=None,
    )
    max_gm = 8
    meta_dyn = MetadataRef(
        gm_id_to_group_id=np.zeros(max_gm, dtype=np.int32),
        gm_id_to_m_offset=np.zeros(max_gm + 1, dtype=np.int32),
    )
    group_sizes = jnp.array([10, 0, 0, 0], dtype=jnp.int32)
    group_offset = jnp.array([0], dtype=jnp.int32)

    num_gm_dyn = fill_metadata(group_sizes, group_offset, meta_dyn, cfgs=cfgs_dyn)
    assert int(num_gm_dyn) == 1
    assert meta_dyn.gm_id_to_m_offset[0] == 0
    assert meta_dyn.gm_id_to_m_offset[1] == 10

    # Static config: fixed to 4 tiles
    cfgs_stat = GmmConfigs(
        tiles=tiles,
        dims=dims,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=None,
        static_gm_tiles=4,
    )
    meta_stat = MetadataRef(
        gm_id_to_group_id=np.zeros(max_gm, dtype=np.int32),
        gm_id_to_m_offset=np.zeros(max_gm + 1, dtype=np.int32),
    )
    num_gm_stat = fill_metadata(group_sizes, group_offset, meta_stat, cfgs=cfgs_stat)
    assert int(num_gm_stat) == 4
    assert meta_stat.gm_id_to_m_offset[0] == 0
    assert meta_stat.gm_id_to_m_offset[1] == 10
    # Dummy padded tiles must have empty range [10, 10]
    assert meta_stat.gm_id_to_m_offset[2] == 10
    assert meta_stat.gm_id_to_m_offset[3] == 10
    assert meta_stat.gm_id_to_m_offset[4] == 10


def test_gmm_v2_static_tiled_output_correctness(monkeypatch):
    _require_tpu()
    key = jax.random.key(42)
    batch_size = 32
    in_size = 256
    out_size = 256
    num_groups = 4

    k1, k2 = jax.random.split(key)
    lhs = jax.random.normal(k1, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (num_groups, in_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([8, 8, 8, 8], dtype=jnp.int32)

    # Baseline dynamic run
    monkeypatch.delenv("MOE_STATIC_TILED_GMM", raising=False)
    monkeypatch.delenv("MOE_STATIC_GMM_TILES", raising=False)
    baseline_out = gmm_v2(lhs, rhs, group_sizes)

    # Static tiled run with fixed 4 tiles
    monkeypatch.setenv("MOE_STATIC_GMM_TILES", "4")
    static_out = gmm_v2(lhs, rhs, group_sizes)

    _assert_allclose(static_out, baseline_out, atol=1e-3, rtol=1e-3)
