# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for kernels/router_topk.py."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

_SPEC = importlib.util.spec_from_file_location(
    "rowmax_topk_ref",
    Path(__file__).resolve().parents[3]
    / "src/vllm_torchtpu/layers/core/rowmax_topk.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
rowmax_topk = _MOD.rowmax_topk


class _FakeLax:
    @staticmethod
    def broadcasted_iota(dtype, shape, dimension: int) -> np.ndarray:
        idx = np.arange(shape[dimension], dtype=dtype)
        view_shape = [1] * len(shape)
        view_shape[dimension] = shape[dimension]
        return np.broadcast_to(idx.reshape(view_shape), shape).copy()


class _FakeJnp:
    int32 = np.int32
    float32 = np.float32
    nan = np.nan

    @staticmethod
    def finfo(dtype):
        return np.finfo(dtype)

    @staticmethod
    def where(cond, x, y) -> np.ndarray:
        return np.where(cond, x, y)

    @staticmethod
    def max(x, axis=None, keepdims=False) -> np.ndarray:
        return np.max(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def min(x, axis=None, keepdims=False) -> np.ndarray:
        return np.min(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def sum(x, axis=None, keepdims=False) -> np.ndarray:
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def maximum(x, y) -> np.ndarray:
        return np.maximum(x, y)

    @staticmethod
    def reshape(x, shape) -> np.ndarray:
        return np.reshape(x, shape)

    @staticmethod
    def broadcast_to(x, shape) -> np.ndarray:
        return np.broadcast_to(x, shape).copy()


class _FakePlTpu:
    class CompilerParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class _FakePl:
    tpu = _FakePlTpu

    class BlockSpec:
        def __init__(self, block_shape, index_map):
            self.block_shape = block_shape
            self.index_map = index_map

    @staticmethod
    def cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    @staticmethod
    def pallas_call(
        kernel_fn,
        *,
        grid,
        in_specs,
        out_specs,
        out_shape,
        compiler_params=None,
        interpret: bool = False,
    ):
        def _runner(*args: np.ndarray):
            outs = [np.zeros(s.shape, dtype=s.dtype) for s in out_shape]
            for step in range(grid[0]):
                in_blocks = []
                for arg, spec in zip(args, in_specs):
                    r_idx, c_idx = spec.index_map(step)
                    br, bc = spec.block_shape
                    r0 = r_idx * br
                    c0 = c_idx * bc
                    blk = np.zeros((br, bc), dtype=arg.dtype)
                    sub = arg[r0 : r0 + br, c0 : c0 + bc]
                    blk[: sub.shape[0], : sub.shape[1]] = sub
                    in_blocks.append(blk)
                out_blocks = [
                    np.zeros(spec.block_shape, dtype=s.dtype)
                    for spec, s in zip(out_specs, out_shape)
                ]
                kernel_fn(*in_blocks, *out_blocks)
                for out_arr, out_blk, spec in zip(outs, out_blocks, out_specs):
                    r_idx, c_idx = spec.index_map(step)
                    br, bc = spec.block_shape
                    r0 = r_idx * br
                    c0 = c_idx * bc
                    valid_r = min(br, out_arr.shape[0] - r0)
                    out_arr[r0 : r0 + valid_r, c0 : c0 + bc] = out_blk[:valid_r, :bc]
            return tuple(outs)

        return _runner


class _ShapeDtypeStruct:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


fake_src = types.ModuleType("jax._src")
fake_src.xla_bridge = types.SimpleNamespace(  # type: ignore[attr-defined]
    backends_are_initialized=lambda: False
)
fake_jax = types.ModuleType("jax")
fake_jax.Array = np.ndarray  # type: ignore[attr-defined]
fake_jax.ShapeDtypeStruct = _ShapeDtypeStruct  # type: ignore[attr-defined]
fake_jax.lax = _FakeLax  # type: ignore[attr-defined]
fake_jax.numpy = _FakeJnp  # type: ignore[attr-defined]
fake_jax._src = fake_src  # type: ignore[attr-defined]
fake_exp = types.ModuleType("jax.experimental")
fake_exp.pallas = _FakePl  # type: ignore[attr-defined]
fake_pallas = types.ModuleType("jax.experimental.pallas")
fake_pallas.tpu = _FakePlTpu  # type: ignore[attr-defined]

_FAKE_MODULES = {
    "jax": fake_jax,
    "jax._src": fake_src,
    "jax.numpy": _FakeJnp,
    "jax.experimental": fake_exp,
    "jax.experimental.pallas": _FakePl,
    "jax.experimental.pallas.tpu": _FakePlTpu,
}
_SAVED_MODULES = {k: sys.modules.get(k) for k in _FAKE_MODULES}
sys.modules.update(_FAKE_MODULES)  # type: ignore[arg-type]
from vllm_torchtpu.kernels import router_topk as rtk  # noqa: E402

for _k, _v in _SAVED_MODULES.items():
    if _v is None:
        sys.modules.pop(_k, None)
    else:
        sys.modules[_k] = _v


def test_router_topk_unbiased_and_biased_parity() -> None:
    rng = np.random.default_rng(123)
    rows, experts, k = 19, 32, 4
    scores = rng.uniform(0.0, 2.0, size=(rows, experts)).astype(np.float32)
    # Include exact ties in row 0 and all-NaN / all--inf hazard rows in rows 1 and 2:
    scores[0, [3, 17, 29]] = 5.0
    scores[1, :] = np.nan
    scores[2, :] = -np.inf

    # 1. Unbiased, renormalize=False via router_topk wrapper:
    w_unb, i_unb = rtk.router_topk(scores, k, renormalize=False)
    ref_w, ref_i = rowmax_topk(torch.from_numpy(scores), k, renormalize=False)
    assert np.allclose(w_unb, ref_w.numpy(), equal_nan=True)
    assert np.array_equal(i_unb, ref_i.numpy())

    # 2. Biased (1D bias triggers reshape; 2D hits direct branch) + renormalize=True:
    bias_1d = rng.uniform(-0.5, 0.5, size=(experts,)).astype(np.float32)
    w_b1, i_b1 = rtk.router_topk_biased(scores, bias_1d, k, renormalize=True)
    ref_wb1, ref_ib1 = rowmax_topk(
        torch.from_numpy(scores),
        k,
        bias=torch.from_numpy(bias_1d),
        renormalize=True,
    )
    assert np.allclose(w_b1, ref_wb1.numpy(), equal_nan=True)
    assert np.array_equal(i_b1, ref_ib1.numpy())

    bias_2d = bias_1d.reshape(1, experts)
    w_b2, i_b2 = rtk.select(scores, k, bias=bias_2d, renormalize=False)
    ref_wb2, ref_ib2 = rowmax_topk(
        torch.from_numpy(scores),
        k,
        bias=torch.from_numpy(bias_2d),
        renormalize=False,
    )
    assert np.allclose(w_b2, ref_wb2.numpy(), equal_nan=True)
    assert np.array_equal(i_b2, ref_ib2.numpy())
