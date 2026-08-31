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
"""Tests for fused_permute_gmm."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import \
    fused_permute_gmm as fused_permute_gmm_module
from vllm_torchtpu.kernels.megablox.fused_permute_gmm import (
    fused_permute_gmm, gather_tiling)
from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2

_HIDDEN = 4096
_GMM1_N = 2048
_TOPK = 10


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _routing(num_tokens: int, num_groups: int, live_groups: int, seed: int):
    """Expert ids for `num_tokens` tokens over `live_groups` of `num_groups`.

    `live_groups < num_groups` is the routing decode actually produces: a
    handful of experts carry nearly every row, which is the regime a tile
    sized from the mean row count per group gets wrong.
    """
    rng = np.random.default_rng(seed)
    live = rng.choice(num_groups, size=live_groups, replace=False)
    ids = rng.choice(live, size=num_tokens * _TOPK).astype(np.int32)
    order = np.argsort(ids, kind="stable")
    gather_indices = (np.arange(num_tokens * _TOPK, dtype=np.int32)[order] //
                      _TOPK)
    group_sizes = np.bincount(ids, minlength=num_groups).astype(np.int32)
    return jnp.asarray(gather_indices), jnp.asarray(group_sizes)


def _weights(num_groups: int, seed: int):
    key = jax.random.key(seed)
    rhs = (jax.random.normal(key,
                             (num_groups, _HIDDEN, _GMM1_N), jnp.bfloat16) *
           0.1).astype(jnp.float8_e4m3fn)
    scale = jnp.full((num_groups, 1, 1, _GMM1_N), 0.1, jnp.float32)
    return rhs, scale


def _gmm1(lhs, rhs, scale, group_sizes):
    return gmm_v2(lhs=lhs,
                  rhs=rhs,
                  rhs_scale=scale,
                  group_sizes=group_sizes,
                  group_offset=jnp.zeros((1, ), jnp.int32)[0],
                  zero_initialize=False,
                  fuse_act="silu",
                  preferred_element_type=jnp.bfloat16,
                  maybe_quantize_lhs=True)


def _fpg(pool, rhs, scale, group_sizes, gather_indices, ep_size=1):
    return fused_permute_gmm(pool,
                             rhs,
                             group_sizes,
                             gather_indices,
                             scale,
                             group_offset=jnp.zeros((1, ), jnp.int32),
                             zero_initialize=False,
                             fuse_act="silu",
                             preferred_element_type=jnp.bfloat16,
                             maybe_quantize_lhs=True,
                             ep_size=ep_size)


def _valid_prefix(out, group_sizes):
    return out[:int(jnp.sum(group_sizes))]


def _bits(x):
    """bf16 bit pattern with negative zero canonicalized.

    A revisited gm tile accumulates `+= 0.0`, which turns -0.0 into +0.0; both
    kernels do it, just not on the same rows. That is a sign bit, not a value.
    """
    bits = np.asarray(x).view(np.uint16).copy()
    bits[bits == 0x8000] = 0
    return bits


@pytest.mark.parametrize("live_groups", [64, 24])
@pytest.mark.parametrize("num_tokens", [512, 2048])
def test_fused_permute_gmm_matches_materialized(num_tokens, live_groups):
    """The fused permute is bit-for-bit the materialized one.

    Concentrated routing (`live_groups=24`) is a separate case, not a variant:
    a gather tile sized from the mean rows per group splits the few fat groups
    it produces, and the split tiles' dots are fully exposed.
    """
    _require_tpu()
    num_groups = 64
    gather_indices, group_sizes = _routing(num_tokens,
                                           num_groups,
                                           live_groups,
                                           seed=0)
    rhs, scale = _weights(num_groups, seed=1)
    pool = (jax.random.normal(jax.random.key(2),
                              (num_tokens, _HIDDEN), jnp.bfloat16) * 0.5)

    fused = jax.jit(_fpg)(pool, rhs, scale, group_sizes, gather_indices)
    materialized = jax.jit(_gmm1)(pool[gather_indices], rhs, scale,
                                  group_sizes)

    np.testing.assert_array_equal(
        _bits(_valid_prefix(fused, group_sizes)),
        _bits(_valid_prefix(materialized, group_sizes)))


def _tokens_for_tiles(tiles: int, num_groups: int = 64) -> int:
    """Smallest token count whose gather grid is exactly `tiles` tiles."""
    for n in range(6, 512):
        size_m = n * _TOPK
        if -(-size_m // gather_tiling(size_m, num_groups)) == tiles:
            return n
    raise AssertionError(f"no token count in range gives {tiles} tiles")


# Every grid length the warm-up guard has to cover, derived from the depth
# rather than fixed: at depth 3 the exposed set is {1, 2}, at 7 it is {1..6}.
_DEGENERATE_TOKENS = sorted({6, 8, 12} | {
    _tokens_for_tiles(t)
    for t in range(3, fused_permute_gmm_module._LOOKAHEAD)
})


@pytest.mark.parametrize("num_tokens", _DEGENERATE_TOKENS)
def test_fused_permute_gmm_survives_a_grid_shorter_than_the_lookahead(
        num_tokens):
    """A grid with fewer steps than the prefetch depth must leave no live DMA.

    The warm-up issues the first few tiles, but tile t is waited by step t and
    by nothing else, and the step count is a runtime value. A tile issued with
    no step to wait it leaves a nonzero DMA semaphore at Mosaic exit, which
    halts the core and fails the whole slice rather than the result, so an
    output comparison cannot catch it. Called twice because a leaked semaphore
    is consumed by the next call's wait and surfaces only on the one after.
    """
    _require_tpu()
    num_groups = 64
    size_m = num_tokens * _TOPK
    tiles = -(-size_m // gather_tiling(size_m, num_groups))
    assert tiles < fused_permute_gmm_module._LOOKAHEAD, (
        f"{num_tokens} tokens gives {tiles} tiles, which is not below the "
        f"prefetch depth {fused_permute_gmm_module._LOOKAHEAD} -- this shape "
        "no longer exercises the warm-up guard")

    gather_indices, group_sizes = _routing(num_tokens,
                                           num_groups,
                                           1,
                                           seed=num_tokens)
    rhs, scale = _weights(num_groups, seed=1)
    pool = (jax.random.normal(jax.random.key(2),
                              (num_tokens, _HIDDEN), jnp.bfloat16) * 0.5)

    fused = jax.jit(_fpg)
    for _ in range(2):
        out = jax.block_until_ready(
            fused(pool, rhs, scale, group_sizes, gather_indices))

    materialized = jax.jit(_gmm1)(pool[gather_indices], rhs, scale,
                                  group_sizes)
    np.testing.assert_array_equal(
        _bits(_valid_prefix(out, group_sizes)),
        _bits(_valid_prefix(materialized, group_sizes)))


@pytest.mark.parametrize("num_tokens,topk", [(8, 5), (13, 4), (64, 10)])
def test_fused_permute_gmm_handles_unaligned_row_counts(num_tokens, topk):
    """size_m = num_tokens * topk need not be a multiple of the sublane block.

    The kernel reshapes its output by that block, so the wrapper pads the row
    count up to it the way gmm_v2 pads its lhs; without the pad this raises
    from inside the kernel rather than falling back.
    """
    _require_tpu()
    num_groups = 4
    size_m = num_tokens * topk
    gather_indices = jnp.arange(size_m, dtype=jnp.int32) % num_tokens
    group_sizes = jnp.full((num_groups, ), size_m // num_groups, jnp.int32)
    rhs, scale = _weights(num_groups, seed=11)
    pool = (jax.random.normal(jax.random.key(12),
                              (num_tokens, _HIDDEN), jnp.bfloat16) * 0.5)

    out = _fpg(pool, rhs, scale, group_sizes, gather_indices)
    assert out.shape[0] == size_m
    reference = _gmm1(pool[gather_indices], rhs, scale, group_sizes)
    valid = int(jnp.sum(group_sizes))
    np.testing.assert_array_equal(_bits(out[:valid]), _bits(reference[:valid]))


def test_fused_permute_gmm_accepts_a_blocked_pool():
    """The pool may arrive already blocked as [t, K // 128, 128].

    That is the layout the per-row DMA reads either way; handing it in saves
    the wrapper's reshape, and both forms must give the same result.
    """
    _require_tpu()
    num_tokens, num_groups = 512, 32
    gather_indices, group_sizes = _routing(num_tokens,
                                           num_groups,
                                           num_groups,
                                           seed=21)
    rhs, scale = _weights(num_groups, seed=22)
    pool = (jax.random.normal(jax.random.key(23),
                              (num_tokens, _HIDDEN), jnp.bfloat16) * 0.5)
    lanes = pltpu.get_tpu_info().num_lanes
    blocked = pool.reshape(num_tokens, _HIDDEN // lanes, lanes)

    flat_out = _fpg(pool, rhs, scale, group_sizes, gather_indices)
    blocked_out = _fpg(blocked, rhs, scale, group_sizes, gather_indices)
    valid = int(jnp.sum(group_sizes))
    np.testing.assert_array_equal(_bits(blocked_out[:valid]),
                                  _bits(flat_out[:valid]))


def test_fused_permute_gmm_zero_initializes_the_tail():
    """zero_initialize covers the rows past the last group.

    The zero-init pair has to stay last in the kernel body, after the gather
    bundle spliced in ahead of it, so the rows it writes are not re-gathered.
    """
    _require_tpu()
    num_tokens, num_groups = 512, 32
    gather_indices, group_sizes = _routing(num_tokens,
                                           num_groups,
                                           num_groups,
                                           seed=31)
    # leave a tail of rows past every group for zero_initialize to cover
    group_sizes = group_sizes.at[-1].add(-64)
    rhs, scale = _weights(num_groups, seed=32)
    pool = (jax.random.normal(jax.random.key(33),
                              (num_tokens, _HIDDEN), jnp.bfloat16) + 1.0)

    out = fused_permute_gmm(pool,
                            rhs,
                            group_sizes,
                            gather_indices,
                            scale,
                            group_offset=jnp.zeros((1, ), jnp.int32),
                            zero_initialize=True,
                            fuse_act="silu",
                            preferred_element_type=jnp.bfloat16,
                            maybe_quantize_lhs=True,
                            ep_size=1)
    valid = int(jnp.sum(group_sizes))
    np.testing.assert_array_equal(
        np.asarray(out[valid:], np.float32),
        np.zeros_like(np.asarray(out[valid:], np.float32)))
    reference = _gmm1(pool[gather_indices], rhs, scale, group_sizes)
    np.testing.assert_array_equal(_bits(out[:valid]), _bits(reference[:valid]))


def test_dispatch_forms_agree():
    """Every dispatch form feeding GMM1 must produce the same GMM1.

    The five forms are the ones a MoE layer can choose between: hand GMM1 an
    already-permuted activation (`ideal`, the floor -- the permute is free),
    materialize the permute with a one-hot matmul, with an XLA gather, or with
    the SparseCore ragged gather, or fold it into GMM1 (`fused`).
    """
    _require_tpu()
    num_tokens, num_groups = 2048, 64
    gather_indices, group_sizes = _routing(num_tokens,
                                           num_groups,
                                           num_groups,
                                           seed=3)
    rhs, scale = _weights(num_groups, seed=4)
    pool = (jax.random.normal(jax.random.key(5),
                              (num_tokens, _HIDDEN), jnp.bfloat16) * 0.5)
    permuted = pool[gather_indices]
    valid = jnp.sum(group_sizes, dtype=jnp.int32)

    forms = {
        "ideal":
        lambda: _gmm1(permuted, rhs, scale, group_sizes),
        "onehot":
        lambda: _gmm1(
            jax.nn.one_hot(gather_indices, num_tokens, dtype=pool.dtype)
            @ pool, rhs, scale, group_sizes),
        "xla_gather":
        lambda: _gmm1(pool[gather_indices], rhs, scale, group_sizes),
        "ragged_sc":
        lambda: _gmm1(
            ragged_gather_v2(pool, gather_indices,
                             jnp.array([0], dtype=jnp.int32), valid[None]),
            rhs, scale, group_sizes),
        "fused":
        lambda: _fpg(pool, rhs, scale, group_sizes, gather_indices),
    }

    reference = None
    for name, fn in forms.items():
        out = jax.block_until_ready(jax.jit(fn)())
        prefix = np.asarray(_valid_prefix(out, group_sizes), np.float32)
        if reference is None:
            reference = prefix
            continue
        # one-hot and the SparseCore gather reduce in a different order, so they
        # match to tolerance; the XLA gather and the fused path are exact.
        np.testing.assert_allclose(prefix,
                                   reference,
                                   atol=2e-2,
                                   rtol=2e-2,
                                   err_msg=f"{name} disagrees with ideal")


@pytest.mark.parametrize("rows,size_group,expected", [
    (10240, 64, 192),
    (81920, 64, 512),
    (640, 64, 64),
    (20480, 512, 64),
    (128, 1, 160),
])
def test_gather_tiling_is_live_rows_per_group(rows, size_group, expected):
    """tile_m tracks the live rows a gm tile actually carries.

    `rows` is what the caller passes: the routed rows divided by ep_size, since
    only that fraction is local. The result is clamped to [64, 512] and aligned
    to 32.
    """
    assert gather_tiling(rows, size_group) == expected


def test_smem_index_limit_follows_the_hardware(monkeypatch):
    """The index budget is a fraction of the device's SMEM, not a per-generation literal."""
    monkeypatch.setattr(fused_permute_gmm_module, "_SMEM_INDEX_LIMIT", [None])
    monkeypatch.setattr(
        "jax.experimental.pallas.tpu.get_tpu_info",
        lambda: types.SimpleNamespace(smem_capacity_bytes=2 << 20))
    assert fused_permute_gmm_module.smem_index_limit_bytes() == 960_000


def test_smem_index_limit_falls_back_without_a_device(monkeypatch):
    """Op construction reaches this before there is a device; the v7x measurement stands."""
    monkeypatch.setattr(fused_permute_gmm_module, "_SMEM_INDEX_LIMIT", [None])

    def _boom():
        raise ValueError("Unsupported TPU device kind: cpu")

    monkeypatch.setattr("jax.experimental.pallas.tpu.get_tpu_info", _boom)
    assert fused_permute_gmm_module.smem_index_limit_bytes() == 480_000
