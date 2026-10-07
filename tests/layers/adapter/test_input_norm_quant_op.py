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
"""TPU_INPUT_NORM_QUANT against the served path on TPU: GemmaRMSNorm (torch,
lowered by XLA), then the fp8 linear's own activation quant and matmul
(`quantized_matmul`). Run with --use-tpu."""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_torchtpu.layers.adapter import input_norm_quant_op as inq
from vllm_torchtpu.layers.adapter import linear_common

K, N, EPS = 4096, 2560, 1e-6


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _setup(device, rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, K, generator=g) * 3).to(torch.bfloat16)
    r = torch.randn(rows, K, generator=g).to(torch.bfloat16)
    if rows > 2:
        x[1] = 0
        r[1] = 0  # an all-zero row: scale 1, q 0
    norm = GemmaRMSNorm(K, eps=EPS)
    norm.weight.data = (torch.randn(K, generator=g) * 0.1).to(torch.bfloat16)
    w = (torch.randn(K, N, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    ws = torch.rand(N, generator=g) * 1e-3
    return (x.to(device), r.to(device), norm.to(device), w.to(device), ws.to(device))


def _same(a, b):
    a, b = a.cpu(), b.cpu()
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    elif a.dtype == torch.bfloat16:
        a, b = a.view(torch.int16), b.view(torch.int16)
    return torch.equal(a, b)


def _run(x, r, norm, with_residual, emit_y, exact):
    extra = ()
    if exact:
        s = x.float() + (r.float() if with_residual else 0)
        extra = (torch.rsqrt(s.pow(2).mean(dim=-1, keepdim=True) + EPS),)
    gw = norm.weight
    if inq.envs.TPU_INPUT_NORM_QUANT_GAMMA == "xla":
        gw = norm.weight.float() + 1.0
    if with_residual:
        out = inq._build_op(EPS, True, emit_y, exact)(x, r, gw, *extra)
    else:
        out = inq._build_op(EPS, False, emit_y, exact)(x, gw, *extra)
    out = list(out)
    y = out.pop(0) if emit_y else None
    ro = out.pop(0) if with_residual else x
    q, s = out
    return y, ro, q, s


def _ulps(a, b):
    return (a.cpu().view(torch.int16).int() - b.cpu().view(torch.int16).int()).abs()


@pytest.mark.parametrize("gamma", ["vmem", "hbm", "xla"])
@pytest.mark.parametrize("emit_y", [True, False])
@pytest.mark.parametrize("with_residual", [True, False])
@pytest.mark.parametrize("rows", [4, 8, 48, 1024])
def test_exact_mode_matches_norm_then_fp8_linear(
    device, monkeypatch, rows, with_residual, emit_y, gamma
):
    """TPU_INPUT_NORM_QUANT=1: bit-exact with the unfused path, with the norm
    weight as a VMEM block or DMA'd from HBM inside the kernel."""
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    monkeypatch.setenv("TPU_INPUT_NORM_QUANT_GAMMA", gamma)
    x, r, norm, w, ws = _setup(device, rows)
    y_ref, r_ref = norm(x, r) if with_residual else (norm(x), x)
    y, ro, q, s = _run(x, r, norm, with_residual, emit_y, exact=True)
    if emit_y:
        assert _same(y, y_ref)
    assert _same(ro, r_ref)
    # The site end to end: pre-quantized matmul vs the fp8 linear on y_ref.
    ref = linear_common.quantized_matmul(y_ref, w, ws)
    got = linear_common.quantized_matmul_prequant(q, s, w, ws)
    assert _same(got, ref)


@pytest.mark.parametrize("gamma", ["vmem", "hbm", "xla"])
@pytest.mark.parametrize("with_residual", [True, False])
@pytest.mark.parametrize("rows", [8, 48, 1024])
def test_kernel_stats_mode_is_within_one_ulp(
    device, monkeypatch, rows, with_residual, gamma
):
    """TPU_INPUT_NORM_QUANT=2: the kernel's own mean(x^2) reduces in another
    order than XLA's, so a normed value may move by one bf16 ulp (and, when it
    is a row's abs-max, that row's fp8 scale with it)."""
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    monkeypatch.setenv("TPU_INPUT_NORM_QUANT_GAMMA", gamma)
    x, r, norm, w, ws = _setup(device, rows)
    y_ref, r_ref = norm(x, r) if with_residual else (norm(x), x)
    y, ro, q, s = _run(x, r, norm, with_residual, True, exact=False)
    assert _same(ro, r_ref)
    d = _ulps(y, y_ref)
    assert d.max() <= 1 and (d > 0).float().mean() < 1e-4
    ref = linear_common.quantized_matmul(y_ref, w, ws).float()
    got = linear_common.quantized_matmul_prequant(q, s, w, ws).float()
    assert float((got - ref).norm() / ref.norm()) < 1e-3


class _W:
    def __init__(self, w):
        self.weight = w


def test_positive_control(device):
    """The end-to-end comparison above fails when one channel of the norm
    weight is perturbed by ~1%."""
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    x, r, norm, w, ws = _setup(device, 8)
    ref = linear_common.quantized_matmul(norm(x, r)[0], w, ws)
    bad_w = norm.weight.detach().clone()
    bad_w[17] = bad_w[17] * 1.01 + 0.01
    _, _, q, s = _run(x, r, _W(bad_w), True, True, exact=True)
    got = linear_common.quantized_matmul_prequant(q, s, w, ws)
    assert not _same(got, ref)


@pytest.mark.parametrize("rows", [8, 48, 1024])
def test_fold_ba_matches_the_bf16_linear(device, rows):
    """TPU_INPUT_NORM_QUANT_FOLD_BA: the kernel's in_proj_ba matches the bf16
    linear on the normed rows up to dot accumulation order; q and the new
    residual are unchanged."""
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    x, r, norm, w, ws = _setup(device, rows)
    g = torch.Generator().manual_seed(1)
    wba = (torch.randn(K, 16, generator=g) * 0.02).to(torch.bfloat16).to(device)
    s = x.float() + r.float()
    rstd = torch.rsqrt(s.pow(2).mean(dim=-1, keepdim=True) + EPS)
    ba, ro, q, sc = inq._build_op(EPS, True, False, True, True)(
        x, r, norm.weight, rstd, wba.t().contiguous()
    )
    y_ref, r_ref = norm(x, r)
    _, _, q_ref, s_ref = _run(x, r, norm, True, False, exact=True)
    assert _same(ro, r_ref) and _same(q, q_ref) and _same(sc, s_ref)
    ref = (y_ref @ wba).float().cpu()
    got = ba.float().cpu()
    assert float((got - ref).norm() / ref.norm()) < 5e-3


@pytest.mark.parametrize("rows", [8, 48, 64, 1024])
def test_fold_ba_served_config(device, monkeypatch, rows):
    """The served FOLD_BA config: mode 2 statistic, gamma=xla (f32 w + 1), the
    in_proj_ba weight as f32 [n_ba, k]. ba against the unfolded path (same
    kernel's y, then the bf16 linear) within 1e-4 relative; a one-element wba
    perturbation (positive control) must exceed it."""
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    monkeypatch.setenv("TPU_INPUT_NORM_QUANT_GAMMA", "xla")
    x, r, norm, w, ws = _setup(device, rows)
    g = torch.Generator().manual_seed(2)
    wba = (torch.randn(K, 16, generator=g) * 0.02).to(torch.bfloat16).to(device)
    gw = norm.weight.float() + 1.0
    y, ro, q, sc = inq._build_op(EPS, True, True, False, False)(x, r, gw)
    ref = (y @ wba).float().cpu()

    def fold(wb):
        ba, ro2, q2, s2 = inq._build_op(EPS, True, False, False, True)(
            x, r, gw, wb.t().contiguous().float()
        )
        assert _same(q2, q) and _same(ro2, ro)
        return ba.float().cpu()

    rel = float((fold(wba) - ref).norm() / ref.norm())
    assert rel < 1e-4, rel
    bad = wba.clone()
    bad[5, 3] = bad[5, 3] + 0.5
    assert float((fold(bad) - ref).norm() / ref.norm()) > 1e-4
