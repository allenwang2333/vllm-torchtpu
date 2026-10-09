# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TD7.57: token-padding bounds, sharded slicing shape, linear and bindings."""

from __future__ import annotations

import itertools
import sys
import types
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.layers.adapter.linear as tpu_linear_mod
from vllm_torchtpu.layers.adapter import linear_common, upstream_bindings
from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState

# ---------------------------------------------------------------------------
# token_padding: integer bounds equal the old range-slice bounds
# ---------------------------------------------------------------------------


def _range_bounds(n: int, valid: int, padded: int) -> tuple[int, int]:
    rows = range(n)[valid:padded]
    return (rows.start, rows.stop) if rows else (0, 0)


@pytest.mark.parametrize("n", [0, 1, 5, 8])
def test_padding_bounds_match_range_slice(n) -> None:
    values = range(-n - 3, n + 4)
    for valid, padded in itertools.product(values, repeat=2):
        state = TokenPaddingState.create(n, "cpu")
        state.update(valid, padded)
        assert state._host_cache is not None
        assert state._host_cache[1] == _range_bounds(n, valid, padded), (
            n,
            valid,
            padded,
        )
        expected = torch.zeros(n, dtype=torch.bool)
        expected[valid:padded] = True
        assert torch.equal(state.local_padding_mask, expected)


# ---------------------------------------------------------------------------
# linear_common: hoisted output shape
# ---------------------------------------------------------------------------


def _old_slice(sharded_tensor, split_sizes, n_shards):
    new_shape = sharded_tensor.shape[:-1] + (n_shards, -1)
    sharded_tensor = sharded_tensor.reshape(new_shape)
    out, start = [], 0
    for size in split_sizes:
        sz = size // n_shards
        out.append(
            sharded_tensor[..., start : start + sz].reshape(new_shape[:-2] + (-1,))
        )
        start += sz
    return out


@pytest.mark.parametrize("lead", [(), (3,), (2, 3)])
def test_slice_sharded_tensor_matches_reference(lead) -> None:
    split_sizes, n_shards = [12, 8, 4], 4
    x = jnp.arange(int(np.prod(lead, dtype=int)) * 24, dtype=jnp.float32).reshape(
        lead + (24,)
    )
    got = linear_common.slice_sharded_tensor_for_concatenation(x, split_sizes, n_shards)
    want = _old_slice(x, split_sizes, n_shards)
    assert len(got) == len(want) == 3
    for g, w, size in zip(got, want, split_sizes, strict=True):
        assert g.shape == w.shape == lead + (size,)
        assert np.array_equal(np.asarray(g), np.asarray(w))


# ---------------------------------------------------------------------------
# linear: deferred bias with a non-parallel input
# ---------------------------------------------------------------------------


def _layer(tp_rank, tp_size, weight_block, bias, *, input_is_parallel, return_bias):
    layer = TpuRowParallelLinear.__new__(TpuRowParallelLinear)
    layer.input_is_parallel = input_is_parallel
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.reduce_results = True
    layer.skip_bias_add = False
    layer.bias = bias
    layer.return_bias = return_bias
    layer.quant_method = SimpleNamespace(apply=lambda _l, x, b: x @ weight_block.T)
    return layer


@pytest.mark.parametrize("return_bias", [False, True])
def test_defer_bias_splits_full_input(monkeypatch, return_bias) -> None:
    torch.manual_seed(0)
    tp_size, n_in, n_out = 2, 8, 3
    x = torch.randn(4, n_in)
    w = torch.randn(n_out, n_in)
    bias = torch.randn(n_out)
    w_blocks = w.chunk(tp_size, dim=1)
    x_blocks = x.chunk(tp_size, dim=1)
    partial = [x_blocks[r] @ w_blocks[r].T for r in range(tp_size)]
    reduced = partial[0] + partial[1]
    seen = []

    def all_reduce(t):
        seen.append(t)
        return reduced

    monkeypatch.setattr(tpu_linear_mod, "tensor_model_parallel_all_reduce", all_reduce)
    for r in range(tp_size):
        out = _layer(
            r,
            tp_size,
            w_blocks[r],
            bias,
            input_is_parallel=False,
            return_bias=return_bias,
        ).forward(x)
        if return_bias:
            out, b = out
            assert b is None
        assert torch.equal(out, reduced + bias)
        assert torch.equal(seen[-1], partial[r])


# ---------------------------------------------------------------------------
# upstream_bindings
# ---------------------------------------------------------------------------


def test_device_hooks_swaps_and_restores() -> None:
    mod = types.ModuleType("td757_fake_mod")
    mod.a, mod.b = 1, 2
    with upstream_bindings.device_hooks(mod, a=10, b=20):
        assert (mod.a, mod.b) == (10, 20)
    assert (mod.a, mod.b) == (1, 2)
    with pytest.raises(RuntimeError), upstream_bindings.device_hooks(mod, a=5):
        assert mod.a == 5
        raise RuntimeError
    assert mod.a == 1
    # Re-entrant from the same thread (RLock).
    with upstream_bindings.device_hooks(mod, a=3):
        with upstream_bindings.device_hooks(mod, a=4):
            assert mod.a == 4
        assert mod.a == 3
    assert mod.a == 1


def _device_op(x):
    return ("gpu", x)


def _upstream(x, *, scale=2):
    """upstream doc"""
    return _device_op(x * scale)


def test_bind_device_ops_rebinds_globals_only_in_copy() -> None:
    bound = upstream_bindings.bind_device_ops(
        _upstream, _device_op=lambda x: ("tpu", x)
    )
    name = f"{upstream_bindings.__name__}._upstream"
    try:
        assert bound(3) == ("tpu", 6)
        assert bound(3, scale=1) == ("tpu", 3)
        assert _upstream(3) == ("gpu", 6)  # original untouched
        assert bound.__code__ is _upstream.__code__
        assert bound.__kwdefaults__ == {"scale": 2}
        assert bound.__qualname__ == _upstream.__qualname__
        assert bound.__annotations__ is _upstream.__annotations__
        namespace = sys.modules[name]
        assert bound.__globals__ is namespace.__dict__
        assert namespace._upstream is bound
        assert namespace.__name__ == name
    finally:
        sys.modules.pop(name, None)
