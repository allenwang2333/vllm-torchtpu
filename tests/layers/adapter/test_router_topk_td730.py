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
"""Unit tests for TD7.30 router_topk, rowmax_topk, and normalization."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.kernels.router_topk as kernel_rtk
import vllm_torchtpu.layers.adapter.router_topk as adapter_rtk
from vllm_torchtpu.layers.adapter.normalization import rms_norm
from vllm_torchtpu.layers.core.rowmax_topk import NEG, rowmax_topk


class _FakeTpuTensor:
    """Lightweight tensor proxy reporting ``device.type == 'tpu'``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._t = tensor
        self.device = SimpleNamespace(type="tpu")
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.float_calls = 0
        self.view_calls = 0

    def dim(self) -> int:
        return self._t.dim()

    def float(self) -> "_FakeTpuTensor":
        self.float_calls += 1
        return _FakeTpuTensor(self._t.float())

    def view(self, *shape: int) -> "_FakeTpuTensor":
        self.view_calls += 1
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _FakeTpuTensor(self._t.view(*shape))


def test_fake_router_topk_and_rowmax_select_cpu_and_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = torch.tensor(
        [[1.0, 4.0, 2.0, 3.0], [5.0, 1.0, 5.0, 2.0]], dtype=torch.float32
    )
    bias = torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=torch.float32)

    w_fake, i_fake = adapter_rtk._fake_router_topk(scores, 2, False)
    assert w_fake.shape == (2, 2)
    assert w_fake.dtype == torch.float32
    assert i_fake.shape == (2, 2)
    assert i_fake.dtype == torch.int32

    wb_fake, ib_fake = adapter_rtk._fake_router_topk_biased(
        scores, bias.view(1, 4), 2, True
    )
    assert wb_fake.shape == (2, 2)
    assert wb_fake.dtype == torch.float32
    assert ib_fake.shape == (2, 2)
    assert ib_fake.dtype == torch.int32

    # CPU fallback paths (unbiased and biased)
    w_cpu, i_cpu = adapter_rtk.rowmax_select(scores, 2, renormalize=True)
    w_ref, i_ref = rowmax_topk(scores, 2, renormalize=True)
    assert torch.allclose(w_cpu, w_ref)
    assert torch.equal(i_cpu, i_ref)

    wb_cpu, ib_cpu = adapter_rtk.rowmax_select(scores, 2, bias=bias, renormalize=False)
    wb_ref, ib_ref = rowmax_topk(scores, 2, bias=bias, renormalize=False)
    assert torch.allclose(wb_cpu, wb_ref)
    assert torch.equal(ib_cpu, ib_ref)

    # TPU non-2D error path
    bad_tpu = _FakeTpuTensor(torch.zeros((2, 3, 4), dtype=torch.float32))
    with pytest.raises(ValueError, match="expects \\[rows, experts\\]"):
        adapter_rtk.rowmax_select(bad_tpu, 2)  # type: ignore[arg-type]

    captured: list[tuple] = []

    def _mock_op(s, topk, renormalize):
        captured.append(("unbiased", s, topk, renormalize))
        return rowmax_topk(s._t, topk, renormalize=renormalize)

    def _mock_biased_op(s, b, topk, renormalize):
        captured.append(("biased", s, b, topk, renormalize))
        return rowmax_topk(s._t, topk, bias=b._t, renormalize=renormalize)

    monkeypatch.setattr(adapter_rtk, "_op", _mock_op)
    monkeypatch.setattr(adapter_rtk, "_biased_op", _mock_biased_op)

    # TPU unbiased fast path: already float32 -> no .float() call
    s_f32 = _FakeTpuTensor(scores)
    adapter_rtk.rowmax_select(s_f32, 2, renormalize=False)  # type: ignore[arg-type]
    assert s_f32.float_calls == 0
    assert captured[-1][1] is s_f32

    # TPU unbiased cast path: bfloat16 -> .float() called once
    s_bf16 = _FakeTpuTensor(scores.to(torch.bfloat16))
    adapter_rtk.rowmax_select(s_bf16, 2, renormalize=True)  # type: ignore[arg-type]
    assert s_bf16.float_calls == 1
    assert captured[-1][1].dtype == torch.float32

    # TPU biased fast path: float32 scores + float32 (1, E) bias
    b_2d_f32 = _FakeTpuTensor(bias.view(1, 4))
    adapter_rtk.rowmax_select(
        s_f32,
        2,
        bias=b_2d_f32,
        renormalize=True,  # type: ignore[arg-type]
    )
    assert s_f32.float_calls == 0
    assert b_2d_f32.float_calls == 0
    assert b_2d_f32.view_calls == 0
    assert captured[-1][1] is s_f32
    assert captured[-1][2] is b_2d_f32

    # TPU biased cast + reshape path: bfloat16 scores + bfloat16 (E,) bias
    s_bf16_2 = _FakeTpuTensor(scores.to(torch.bfloat16))
    b_1d_bf16 = _FakeTpuTensor(bias.to(torch.bfloat16))
    adapter_rtk.rowmax_select(
        s_bf16_2,
        2,
        bias=b_1d_bf16,
        renormalize=False,  # type: ignore[arg-type]
    )
    assert s_bf16_2.float_calls == 1
    assert b_1d_bf16.float_calls == 1
    assert captured[-1][2].shape == (1, 4)
    assert captured[-1][2].dtype == torch.float32


def test_rowmax_topk_2d_3d_biased_renormalize_and_nan_sentinel() -> None:
    scores = torch.tensor(
        [
            [2.0, 5.0, 5.0, 1.0],
            [NEG, -float("inf"), float("nan"), NEG - 10.0],
        ],
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="exceeds the expert count"):
        rowmax_topk(scores, 5)

    # Unbiased with renormalize=True + sentinel row
    w, idx = rowmax_topk(scores, 2, renormalize=True)
    # Row 0: tie between col 1 and col 2 resolves to col 1 then col 2
    assert idx[0].tolist() == [1, 2]
    assert torch.allclose(w[0], torch.tensor([0.5, 0.5]))
    # Row 1: all scores <= NEG -> weights are NaN, indices are still distinct [0, 1]
    assert torch.isnan(w[1]).all()
    assert idx[1].tolist() == [0, 1]

    # 2D biased: fast path with matching dtype and (1, n) shape
    bias_2d = torch.tensor([[3.5, 0.0, 0.1, 4.5]], dtype=torch.float32)
    wb_fast, ib_fast = rowmax_topk(scores[:1], 2, bias=bias_2d, renormalize=False)
    # choice = [5.5, 5.0, 5.1, 5.5] -> tie between col 0 and col 3 -> [0, 3]
    assert ib_fast[0].tolist() == [0, 3]
    # Returned weights are unbiased scores[0, [0, 3]] = [2.0, 1.0]
    assert torch.allclose(wb_fast[0], torch.tensor([2.0, 1.0]))

    # 2D biased: cast + view path with bfloat16 1D bias + renormalize=True
    bias_1d_bf16 = torch.tensor([3.5, 0.0, 0.1, 4.5], dtype=torch.bfloat16)
    wb_cast, ib_cast = rowmax_topk(scores[:1], 2, bias=bias_1d_bf16, renormalize=True)
    assert ib_cast[0].tolist() == [0, 3]
    assert torch.allclose(wb_cast[0], torch.tensor([2.0 / 3.0, 1.0 / 3.0]))

    # 3D biased: exercises scores.dim() != 2 branch
    scores_3d = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    bias_1d = torch.tensor([10.0, 0.0, 5.0, -5.0], dtype=torch.float32)
    w_3d, i_3d = rowmax_topk(scores_3d, 2, bias=bias_1d, renormalize=False)
    assert w_3d.shape == (2, 3, 2)
    assert i_3d.shape == (2, 3, 2)


def test_pallas_router_topk_interpret_and_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores_np = np.array(
        [
            [1.5, 4.0, 4.0, 2.5, 0.5, 3.0],
            [0.2, 0.8, 0.1, 0.9, 0.4, 0.7],
            [NEG, -np.inf, np.nan, NEG, NEG, NEG],
        ],
        dtype=np.float32,
    )
    bias_np = np.array([2.0, -1.0, 0.5, 1.5, 3.0, 0.0], dtype=np.float32)
    scores_jnp = jnp.asarray(scores_np)
    scores_t = torch.from_numpy(scores_np)
    bias_t = torch.from_numpy(bias_np)

    for renormalize in (False, True):
        w_j, i_j = kernel_rtk.select(
            scores_jnp, 3, interpret=True, renormalize=renormalize
        )
        w_ref, i_ref = rowmax_topk(scores_t, 3, renormalize=renormalize)
        np.testing.assert_allclose(
            np.asarray(w_j[:2]), w_ref[:2].numpy(), rtol=1e-6, atol=1e-6
        )
        assert np.isnan(np.asarray(w_j[2])).all()
        np.testing.assert_array_equal(np.asarray(i_j), i_ref.numpy())

    # Biased with 1D bias (triggers jnp.reshape) and 2D bias (skips jnp.reshape)
    for bias_jnp, renormalize in (
        (jnp.asarray(bias_np), False),
        (jnp.asarray(bias_np.reshape(1, -1)), True),
    ):
        wb_j, ib_j = kernel_rtk.select(
            scores_jnp,
            3,
            interpret=True,
            bias=bias_jnp,
            renormalize=renormalize,
        )
        wb_ref, ib_ref = rowmax_topk(scores_t, 3, bias=bias_t, renormalize=renormalize)
        np.testing.assert_allclose(
            np.asarray(wb_j[:2]), wb_ref[:2].numpy(), rtol=1e-6, atol=1e-6
        )
        assert np.isnan(np.asarray(wb_j[2])).all()
        np.testing.assert_array_equal(np.asarray(ib_j), ib_ref.numpy())

    # Test router_topk and router_topk_biased entrypoints via interpret=True
    orig_select = kernel_rtk.select
    monkeypatch.setattr(
        kernel_rtk,
        "select",
        lambda s, topk, interpret=False, *, bias=None, renormalize=False: orig_select(
            s, topk, interpret=True, bias=bias, renormalize=renormalize
        ),
    )
    w_wrap, i_wrap = kernel_rtk.router_topk(scores_jnp[:2], 2, True)
    w_wrap_ref, i_wrap_ref = rowmax_topk(scores_t[:2], 2, renormalize=True)
    np.testing.assert_allclose(
        np.asarray(w_wrap), w_wrap_ref.numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(i_wrap), i_wrap_ref.numpy())

    wb_wrap, ib_wrap = kernel_rtk.router_topk_biased(
        scores_jnp[:2], jnp.asarray(bias_np.reshape(1, -1)), 2, False
    )
    wb_wrap_ref, ib_wrap_ref = rowmax_topk(
        scores_t[:2], 2, bias=bias_t, renormalize=False
    )
    np.testing.assert_allclose(
        np.asarray(wb_wrap), wb_wrap_ref.numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(ib_wrap), ib_wrap_ref.numpy())


def test_rms_norm_dtypes_and_single_float_cast() -> None:
    x_f32 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -0.5]])
    w_f32 = torch.tensor([0.5, 1.5, -0.5, 2.0])
    eps = 1e-6

    ref = x_f32 * torch.rsqrt(x_f32.square().mean(-1, keepdim=True) + eps) * w_f32

    out_f32 = rms_norm(x_f32, w_f32, eps)
    assert out_f32.dtype == torch.float32
    assert torch.allclose(out_f32, ref, atol=1e-6)

    out_bf16 = rms_norm(x_f32.to(torch.bfloat16), w_f32.to(torch.bfloat16), eps)
    assert out_bf16.dtype == torch.bfloat16
    assert torch.allclose(out_bf16, ref.to(torch.bfloat16), atol=1e-2)

    # Verify x.float() is only called once when x is non-float32
    float_call_count = 0

    class _CountingTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, data: torch.Tensor):
            return torch.Tensor._make_subclass(cls, data)

        def float(self, *args, **kwargs):  # type: ignore[override]
            nonlocal float_call_count
            float_call_count += 1
            return super().float(*args, **kwargs)

    x_sub = _CountingTensor(x_f32.to(torch.bfloat16))
    _ = rms_norm(x_sub, w_f32, eps)
    assert float_call_count == 1
