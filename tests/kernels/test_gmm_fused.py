"""The fused grouped matmul (kernels/megablox/gmm_fused.py, the decode step's MoE FFN) against the dense reference
and against the two-call path it replaces, at a small but structurally faithful shape; and the fusion planner's
shape gate (gmm_fused.can_fuse) without a device. The reference-based cases came with the
tpu-inference PR 3346 port's test file and are retargeted at this kernel; the port's kernel and its bit-exactness
against the two-call path left with it (this kernel tiles differently, so the comparisons are tolerance-based)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.gmm_fused import can_fuse, gmm_fused
from vllm_torchtpu.kernels.megablox.gmm_v2 import (TileSizes,
                                                   get_packing_factor, gmm_v2)


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


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

    x_q = jnp.nan_to_num(x_blocked_q.reshape(orig_shape))
    return x_q, scale.squeeze(axis=axis + 1).astype(jnp.float32)


# Small but structurally faithful MoE shapes: w1 holds gate|up along n, and
# gmm1's post-activation width has to equal w2's contracting dim. Both quant
# block sizes stay >= mxu_column_size so every mode keeps a single accumulator
# dtype across the two matmuls.
_BATCH = 128
_SIZE_K1 = 256
_SIZE_N1 = 512
_SIZE_K2 = _SIZE_N1 // 2
_SIZE_N2 = 256
_NUM_GROUPS = 8

# Forced on both paths so a mismatch can only come from the fusion, and small
# enough that a group spans several gm tiles and several m buckets.
_TILE_M = 32
_BUCKET_BASE = 16


def _make_weights(weight_mode: str, key: jax.Array, num_groups: int):
    """Returns (w1, w1_scale, w2, w2_scale, rhs_quant_dtype)."""
    key1, key2, scale_key = jax.random.split(key, 3)
    shape1 = (num_groups, _SIZE_K1, _SIZE_N1)
    shape2 = (num_groups, _SIZE_K2, _SIZE_N2)

    w1 = jax.random.uniform(key1, shape1, jnp.bfloat16, -1, 1)
    w2 = jax.random.uniform(key2, shape2, jnp.bfloat16, -1, 1)
    if weight_mode == "bf16":
        return w1, None, w2, None, None

    block1, block2 = (_SIZE_K1, _SIZE_K2)
    if weight_mode == "fp8_block128":
        block1 = block2 = 128
    w1_q, scale1 = quantize_tensor(w1,
                                   jnp.float8_e4m3fn,
                                   axis=1,
                                   block_size=block1)
    w2_q, scale2 = quantize_tensor(w2,
                                   jnp.float8_e4m3fn,
                                   axis=1,
                                   block_size=block2)
    return (w1_q, jnp.expand_dims(scale1,
                                  axis=2), w2_q, jnp.expand_dims(scale2,
                                                                 axis=2), None)


def _run_split(lhs, w1, w1_scale, w2, w2_scale, group_sizes, group_offset,
               fuse_act, rhs_quant_dtype):
    """Today's two-call path, pinned to the fused kernel's tiling."""
    packing_factor = get_packing_factor(w2.dtype, rhs_quant_dtype)
    size_k2 = w2.shape[1] * packing_factor

    gmm1_res = gmm_v2(
        lhs,
        w1,
        group_sizes,
        w1_scale,
        None,
        group_offset,
        tile_info=TileSizes(tile_m=_TILE_M,
                            tile_k=lhs.shape[1],
                            tile_n=_SIZE_N1 // 2,
                            bucket_base=_BUCKET_BASE),
        zero_initialize=False,
        fuse_act=fuse_act,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    gmm1_res = gmm1_res[:, :size_k2]
    return gmm_v2(
        gmm1_res,
        w2,
        group_sizes,
        w2_scale,
        None,
        group_offset,
        tile_info=TileSizes(tile_m=_TILE_M,
                            tile_k=size_k2,
                            tile_n=w2.shape[2],
                            bucket_base=_BUCKET_BASE),
        zero_initialize=False,
        preferred_element_type=lhs.dtype,
        rhs_quant_dtype=rhs_quant_dtype,
    )


def _run_fused(lhs, w1, w1_scale, w2, w2_scale, group_sizes, group_offset,
               fuse_act, rhs_quant_dtype):
    """This kernel as the dispatch calls it (layers/common/fused_moe_gmm.py): a scalar group offset, rows outside
    the shard's expert window zero-initialized, the merged-body pipeline off, the served bucket base."""
    assert rhs_quant_dtype is None and w1_scale is not None, "the kernel takes fp8 weights with per-channel scales"
    return gmm_fused(lhs=lhs,
                     w1=w1,
                     w2=w2,
                     group_sizes=group_sizes,
                     w1_scale=w1_scale,
                     w2_scale=w2_scale,
                     group_offset=jnp.asarray(group_offset,
                                              dtype=jnp.int32).reshape(()),
                     fuse_act=fuse_act,
                     zero_initialize=True,
                     unconditional_pipeline=False,
                     tile_m=_TILE_M,
                     bucket_base=_BUCKET_BASE)


def _computed_rows(group_sizes: jax.Array, group_offset: int,
                   num_local_groups: int) -> slice:
    """Rows both paths actually write.

    ``zero_initialize=False`` leaves every other row uninitialized, and the
    two paths leave *different* garbage there, so comparisons have to be
    restricted to the groups the kernel visits.
    """
    sizes = np.asarray(group_sizes)
    start = int(sizes[:group_offset].sum())
    end = int(sizes[:group_offset + num_local_groups].sum())
    return slice(start, end)


@pytest.mark.parametrize("fuse_act", ["silu", "swigluoai"])
def test_fused_matches_split(fuse_act):
    """The fused kernel against the two-call path on the same fp8 weights, within the tolerance of a different
    tiling (both quantize the activation to fp8 per row)."""
    _require_tpu()
    key = jax.random.key(0)
    lhs_key, weight_key = jax.random.split(key)
    lhs = jax.random.uniform(lhs_key, (_BATCH, _SIZE_K1), jnp.bfloat16, -1, 1)
    w1, w1_scale, w2, w2_scale, rhs_quant_dtype = _make_weights(
        "fp8_per_channel", weight_key, _NUM_GROUPS)
    group_sizes = get_group_sizes(_BATCH, _NUM_GROUPS)
    group_offset = jnp.array(0, dtype=jnp.int32)
    args = (lhs, w1, w1_scale, w2, w2_scale, group_sizes, group_offset,
            fuse_act, rhs_quant_dtype)
    expected = _run_split(*args)
    actual = _run_fused(*args)
    rows = _computed_rows(group_sizes, 0, _NUM_GROUPS)
    np.testing.assert_allclose(np.asarray(actual[rows], dtype=np.float32),
                               np.asarray(expected[rows], dtype=np.float32),
                               atol=3e-2,
                               rtol=3e-2)


@pytest.mark.parametrize(
    "group_sizes",
    [
        pytest.param([_BATCH, 0, 0, 0, 0, 0, 0, 0], id="single_group"),
        pytest.param([40, 0, 30, 0, 20, 0, 38, 0], id="empty_interleaved"),
        pytest.param([1, 1, 1, 1, 1, 1, 1, 121], id="single_row_groups"),
        pytest.param([3, 5, 13, 27, 9, 31, 17, 23], id="sublane_straddle"),
        pytest.param([10, 10, 10, 10, 0, 0, 0, 0], id="short_ep_tail"),
    ],
)
def test_fused_ragged_group_sizes(group_sizes):
    _require_tpu()

    key = jax.random.key(2)
    lhs_key, weight_key = jax.random.split(key)
    lhs = jax.random.uniform(lhs_key, (_BATCH, _SIZE_K1), jnp.bfloat16, -1, 1)
    w1, w1_scale, w2, w2_scale, _ = _make_weights("fp8_per_channel",
                                                  weight_key, _NUM_GROUPS)
    group_sizes = jnp.asarray(group_sizes, dtype=jnp.int32)
    group_offset = jnp.array(0, dtype=jnp.int32)

    args = (lhs, w1, w1_scale, w2, w2_scale, group_sizes, group_offset, "silu",
            None)
    expected = _run_split(*args)
    actual = _run_fused(*args)

    rows = _computed_rows(group_sizes, 0, _NUM_GROUPS)
    np.testing.assert_allclose(np.asarray(actual[rows], dtype=np.float32),
                               np.asarray(expected[rows], dtype=np.float32),
                               atol=3e-2,
                               rtol=3e-2)


@pytest.mark.parametrize("group_offset", [0, 2])
def test_fused_group_offset(group_offset):
    _require_tpu()

    num_local_groups = _NUM_GROUPS - group_offset
    key = jax.random.key(3)
    lhs_key, weight_key = jax.random.split(key)
    lhs = jax.random.uniform(lhs_key, (_BATCH, _SIZE_K1), jnp.bfloat16, -1, 1)
    w1, w1_scale, w2, w2_scale, _ = _make_weights("fp8_per_channel",
                                                  weight_key, num_local_groups)
    group_sizes = get_group_sizes(_BATCH, _NUM_GROUPS)
    offset = jnp.array(group_offset, dtype=jnp.int32)

    args = (lhs, w1, w1_scale, w2, w2_scale, group_sizes, offset, "silu", None)
    expected = _run_split(*args)
    actual = _run_fused(*args)

    rows = _computed_rows(group_sizes, group_offset, num_local_groups)
    np.testing.assert_allclose(np.asarray(actual[rows], dtype=np.float32),
                               np.asarray(expected[rows], dtype=np.float32),
                               atol=3e-2,
                               rtol=3e-2)


def _v7x_sublane_tiling(dtype: jnp.dtype) -> int:
    """TpuInfo.get_sublane_tiling for generation 7 (large 2nd-minor tiling)."""
    return 8 * (32 // jax.dtypes.itemsize_bits(dtype))


def _fake_tpu_info() -> SimpleNamespace:
    """A v7x TensorCore, for shape checks that must run without a device."""
    return SimpleNamespace(
        generation=7,
        num_lanes=128,
        num_sublanes=8,
        mxu_column_size=256,
        vmem_capacity_bytes=64 * 1024 * 1024,
        fp8_ops_per_second=int(4.60e15),
        int8_ops_per_second=0,
        get_sublane_tiling=_v7x_sublane_tiling,
    )


def _moe_shapes(hidden: int, intermediate: int, num_experts: int = 64):
    """Shape-only stand-ins for one EP rank's fp8 per-channel MoE weights."""
    sds = jax.ShapeDtypeStruct
    return dict(
        lhs=sds((40960, hidden), jnp.bfloat16),
        w1=sds((num_experts, hidden, 2 * intermediate), jnp.float8_e4m3fn),
        w2=sds((num_experts, intermediate, hidden), jnp.float8_e4m3fn),
        group_sizes=sds((num_experts, ), jnp.int32),
        w1_scale=sds((num_experts, 1, 1, 2 * intermediate), jnp.float32),
        w2_scale=sds((num_experts, 1, 1, hidden), jnp.float32),
        group_offset=sds((1, ), jnp.int32),
    )


def test_can_fuse_rejects_oversize(monkeypatch):
    # Deliberately no _require_tpu: can_fuse is pure shape arithmetic, and CI
    # has to cover the fallback decision on hosts without an accelerator.
    monkeypatch.setattr(
        "vllm_torchtpu.kernels.megablox.gmm_v2.pltpu.get_tpu_info",
        _fake_tpu_info)

    def check(shapes):
        return can_fuse(shapes["lhs"],
                        shapes["w1"],
                        shapes["w2"],
                        shapes["group_sizes"],
                        shapes["w1_scale"],
                        shapes["w2_scale"],
                        None,
                        None,
                        shapes["group_offset"],
                        fuse_act="silu")

    # Qwen3.5-397B-A17B, DP8 + EP: 64 local experts of [4096, 2048] fp8.
    assert check(_moe_shapes(hidden=4096, intermediate=1024)) is True

    # Qwen3-coder-480B: w1 alone is 31.5 MB per expert, so a double-buffered
    # slab does not fit and the caller must keep the two-call path.
    assert check(_moe_shapes(hidden=6144, intermediate=2560)) is False

    # A gmm1 output wider than w2's contracting dim (padded hidden size) has
    # no single-buffer intermediate either.
    mismatched = _moe_shapes(hidden=4096, intermediate=1024)
    mismatched["w2"] = jax.ShapeDtypeStruct((64, 512, 4096), jnp.float8_e4m3fn)
    assert check(mismatched) is False
