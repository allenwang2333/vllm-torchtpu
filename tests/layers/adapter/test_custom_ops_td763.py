# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TD7.63: mamba state copy sizing and mHC op construction."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops import mamba_state_copy_op as mamba
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_mhc_op as mhc,
)

# ---------------------------------------------------------------------------
# mamba_state_copy_op
# ---------------------------------------------------------------------------


def _old_step_rows(rows, kernel_block_bytes, budget_bytes):
    fitting = [
        k
        for k in range(1, rows + 1)
        if rows % k == 0 and k * kernel_block_bytes <= budget_bytes
    ]
    if not fitting:
        raise ValueError("no fit")
    return max(fitting)


def test_step_rows_matches_list_max() -> None:
    for rows in range(0, 49):
        for block in (1, 3, 64, 1000):
            for budget in (0, 1, 7, 64, 200, 4096, 10**6):
                try:
                    want = _old_step_rows(rows, block, budget)
                except ValueError:
                    with pytest.raises(ValueError, match="VMEM step budget"):
                        mamba._row_copy_step_rows(rows, block, budget)
                else:
                    assert mamba._row_copy_step_rows(rows, block, budget) == want


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int8])
def test_copy_rows_sizes_block_from_shape(monkeypatch, dtype) -> None:
    pool = torch.zeros(3, 4, 5, 6, dtype=dtype)
    seen = {}

    def fake_step_rows(rows, kernel_block_bytes, budget):
        seen["args"] = (rows, kernel_block_bytes, budget)
        return 2

    def fake_row_copy_fn(num_pools, split, step_rows):
        seen["geom"] = (num_pools, split, step_rows)
        return lambda p, s, d: "marker"

    monkeypatch.setattr(mamba, "_copy_step_budget_bytes", lambda: 999)
    monkeypatch.setattr(mamba, "_row_copy_step_rows", fake_step_rows)
    monkeypatch.setattr(mamba, "_row_copy_fn", fake_row_copy_fn)
    src = torch.tensor([0])
    assert mamba.copy_mamba_state_rows(pool, src, src, 2, 2) == "marker"
    assert seen["args"] == (4, pool[0, 0].numel() * pool.element_size(), 999)
    assert seen["geom"] == (2, 2, 2)


def test_copy_step_budget_bytes(monkeypatch) -> None:
    monkeypatch.setattr(
        mamba.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=64 << 20),
    )
    assert mamba._copy_step_budget_bytes() == (64 << 20) // 8


def test_copy_fn_and_fake(monkeypatch) -> None:
    calls = []

    def copy_blocks(pool, src, dst):
        calls.append((src, dst))
        out = np.array(pool)
        out[np.asarray(dst)] = np.asarray(pool)[np.asarray(src)]
        return jnp.asarray(out)

    monkeypatch.setattr(mamba.pool_adapters, "copy_blocks", copy_blocks)
    pool = jnp.arange(12, dtype=jnp.int32).reshape(4, 3)
    src = jnp.array([2, 0], dtype=jnp.int32)
    dst = jnp.array([1, 3], dtype=jnp.int32)
    new_pool, marker = mamba._copy_fn(pool, src, dst)
    want = np.array(pool)
    want[[1, 3]] = np.array(pool)[[2, 0]]
    np.testing.assert_array_equal(np.array(new_pool), want)
    assert int(marker) == 2 and len(calls) == 1

    tpool = torch.zeros(4, 3, dtype=torch.bfloat16)
    tsrc = torch.tensor([1, 2], dtype=torch.int32)
    fake_pool, fake_marker = mamba._fake_copy(tpool, tsrc, tsrc)
    assert fake_pool.shape == tpool.shape and fake_pool.dtype == tpool.dtype
    assert fake_marker.shape == () and fake_marker.dtype == torch.int32


def test_copy_blocks_writes_every_pool(monkeypatch) -> None:
    def fake_op(pool, src, dst):
        return pool + dst.sum(), src[0] + len(seen)

    seen = []

    def recording_op(pool, src, dst):
        seen.append(pool)
        return fake_op(pool, src, dst)

    monkeypatch.setattr(mamba, "_copy_op", recording_op)
    pools = [torch.zeros(2, 2), torch.ones(2, 2)]
    src = torch.tensor([5])
    dst = torch.tensor([3])
    marker = mamba.copy_mamba_state_blocks._torchdynamo_orig_callable(pools, src, dst)
    assert torch.equal(pools[0], torch.full((2, 2), 3.0))
    assert torch.equal(pools[1], torch.full((2, 2), 4.0))
    assert int(marker) == 7  # last pool's marker
    assert mamba.copy_pool_blocks is mamba.copy_mamba_state_blocks


def test_row_copy_run_writes_back(monkeypatch) -> None:
    names = []
    captured_fns = []

    def fake_jax_op(name, fn, **kwargs):
        names.append((name, kwargs))
        captured_fns.append(fn)
        return lambda pool, src, dst: (pool + 1, src[0])

    monkeypatch.setattr(mamba.pallas, "jax_op", fake_jax_op)
    run = mamba._row_copy_fn.__wrapped__(2, 3, 2)
    assert names == [("pallas::mamba_state_row_copy_2x3s2", {"donate_argnums": (0,)})]
    pool = torch.zeros(2, 6, 1)
    marker = run._torchdynamo_orig_callable(pool, torch.tensor([4]), torch.tensor([0]))
    assert torch.equal(pool, torch.ones(2, 6, 1))
    assert int(marker) == 4

    # Exercise the inner _rows_fn closure (lines 115-121 of mamba_state_copy_op.py)
    copy_calls = []

    def fake_copy_blocks(view, src_steps, dst_steps):
        copy_calls.append((view.shape, np.asarray(src_steps), np.asarray(dst_steps)))
        return view + 2

    monkeypatch.setattr(mamba.pool_adapters, "copy_blocks", fake_copy_blocks)
    jpool = jnp.zeros((4, 6, 2), dtype=jnp.float32)
    jsrc = jnp.array([2, 0], dtype=jnp.int32)
    jdst = jnp.array([1, 3], dtype=jnp.int32)
    out_pool, out_marker = captured_fns[0](jpool, jsrc, jdst)
    assert out_pool.shape == (4, 6, 2)
    assert int(out_marker) == 2
    assert len(copy_calls) == 1
    assert copy_calls[0][0] == (12, 2, 2)
    np.testing.assert_array_equal(copy_calls[0][1], np.array([6, 7, 8, 0, 1, 2]))
    np.testing.assert_array_equal(copy_calls[0][2], np.array([3, 4, 5, 9, 10, 11]))


# ---------------------------------------------------------------------------
# deepseek_v4_mhc_op
# ---------------------------------------------------------------------------

_GATES = (1e-6, 1e-6, 1e-6, 2.0, 20)


def test_name_float_and_gate_suffix() -> None:
    assert mhc._name_float(1e-6) == "1em06"
    assert mhc._name_float(2.0) == "2"
    assert mhc._name_float(0.5) == "0p5"
    assert mhc._name_float(1e20) == "1e20"
    assert mhc._gate_suffix(*_GATES) == "_r1em06_p1em06_s1em06_a2_i20"


def test_jax_wrappers_forward_arguments(monkeypatch) -> None:
    rec = {}
    monkeypatch.setattr(
        mhc.pre_kernel, "mhc_pre", lambda *a: rec.setdefault("pre", a) or "pre"
    )
    monkeypatch.setattr(
        mhc.post_kernel, "mhc_post", lambda *a: rec.setdefault("post", a) or "post"
    )
    monkeypatch.setattr(
        mhc.fused_post_pre_kernel,
        "mhc_fused_post_pre",
        lambda *a: rec.setdefault("fused", a) or "fused",
    )
    kw = dict(
        rms_eps=1.0,
        hc_pre_eps=2.0,
        hc_sinkhorn_eps=3.0,
        hc_post_mult_value=4.0,
        sinkhorn_repeat=5,
    )
    mhc._mhc_pre_jax("r", "f", "s", "b", **kw)
    mhc._mhc_post_jax("x", "r", "p", "c")
    mhc._mhc_fused_post_pre_jax("x", "r", "p", "c", "f", "s", "b", **kw)
    assert rec["pre"] == ("r", "f", "s", "b", 1.0, 2.0, 3.0, 4.0, 5)
    assert rec["post"] == ("x", "r", "p", "c")
    assert rec["fused"] == ("x", "r", "p", "c", "f", "s", "b", 1.0, 2.0, 3.0, 4.0, 5)


def test_fake_shapes() -> None:
    residual = torch.zeros(3, 2, 4, 8, dtype=torch.bfloat16)
    post_mix, comb_mix, layer_input = mhc._fake_mhc_pre(residual, None, None, None)
    assert post_mix.shape == (3, 2, 4, 1) and post_mix.dtype == torch.float32
    assert comb_mix.shape == (3, 2, 4, 4) and comb_mix.dtype == torch.float32
    assert layer_input.shape == (3, 2, 8) and layer_input.dtype == torch.bfloat16
    post = mhc._fake_mhc_post(None, residual, None, None)
    assert post.shape == residual.shape and post.dtype == residual.dtype
    fused = mhc._fake_mhc_fused_post_pre(None, residual, None, None, None, None, None)
    assert len(fused) == 4
    assert fused[0].shape == residual.shape
    assert [t.shape for t in fused[1:]] == [(3, 2, 4, 1), (3, 2, 4, 4), (3, 2, 8)]


class _FakeOp:
    def __init__(self, name, fn, **kwargs):
        self.name, self.fn, self.kwargs = name, fn, kwargs
        self.fake = None

    def register_fake(self, fake):
        self.fake = fake


@pytest.fixture
def fake_builders(monkeypatch):
    built = []

    def jax_op(name, fn, **kwargs):
        op = _FakeOp(name, fn, **kwargs)
        built.append(op)
        return op

    monkeypatch.setattr(mhc, "_mhc_op_cache", {})
    monkeypatch.setattr(mhc.pallas, "jax_op", jax_op)
    mesh = object()
    monkeypatch.setattr(
        mhc, "get_vllm_model_wrapper_context", lambda: SimpleNamespace(mesh=mesh)
    )
    return built, mesh


def test_get_mhc_ops_builds_once_and_formats_suffix_once(
    monkeypatch, fake_builders
) -> None:
    built, mesh = fake_builders
    suffix_calls = []
    real_suffix = mhc._gate_suffix

    def counting_suffix(*args):
        suffix_calls.append(args)
        return real_suffix(*args)

    monkeypatch.setattr(mhc, "_gate_suffix", counting_suffix)
    ops = mhc.get_mhc_ops(*_GATES)
    assert suffix_calls == [_GATES]
    suffix = real_suffix(*_GATES)
    assert ops.pre.name == "pallas::deepseek_v4_mhc_pre_v1" + suffix
    assert ops.fused.name == "pallas::deepseek_v4_mhc_fused_post_pre_v1" + suffix
    assert ops.post.name == "pallas::deepseek_v4_mhc_post_v1"
    assert ops.pre.fake is mhc._fake_mhc_pre
    assert ops.fused.fake is mhc._fake_mhc_fused_post_pre
    assert ops.post.fake is mhc._fake_mhc_post
    for op in built:
        assert op.kwargs["mesh"] is mesh
    assert len(ops.pre.kwargs["input_partition_specs"]) == 4
    assert len(ops.fused.kwargs["input_partition_specs"]) == 7
    assert len(ops.post.kwargs["input_partition_specs"]) == 4
    for op, target in (
        (ops.pre, mhc._mhc_pre_jax),
        (ops.fused, mhc._mhc_fused_post_pre_jax),
    ):
        assert isinstance(op.fn, functools.partial) and op.fn.func is target
        assert op.fn.keywords == dict(
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
        )

    again = mhc.get_mhc_ops(*_GATES)
    assert again == ops
    assert len(built) == 3
    # Other constants name (and build) distinct gated ops; post is shared.
    other = mhc.get_mhc_ops(1e-5, 1e-6, 1e-6, 2.0, 20)
    assert other.pre is not ops.pre and other.post is ops.post
    assert len(built) == 5


def test_builders_without_precomputed_suffix(fake_builders) -> None:
    built, _ = fake_builders
    pre = mhc._mhc_pre_op(*_GATES)
    fused = mhc._mhc_fused_post_pre_op(*_GATES)
    suffix = mhc._gate_suffix(*_GATES)
    assert pre.name.endswith(suffix) and fused.name.endswith(suffix)
    # The same op is returned whether or not the suffix was passed in.
    assert mhc._mhc_pre_op(*_GATES, suffix=suffix) is pre
    assert mhc._mhc_fused_post_pre_op(*_GATES, suffix=suffix) is fused
    assert len(built) == 2
