# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""T6 hbm-hop flags: bitwise vs the flag-off graph, plus positive controls.

TPU_RMSNORM_DECODE_MATERIALIZE: the decode-sized residual RMSNorm with the f32
sum and bf16 residual behind an optimization barrier must equal upstream
``forward_native`` bit for bit, alone and in a chain of norms compiled as one
graph (the served shape: residual stream through several layers). A fault that
takes the variance from the bf16-rounded residual must be caught.

TPU_MOE_TP_DECODE_SHARED_META: every armed layer gets the same experts_start
tensor object (same value as before), so XLA can CSE the meta stack.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_torchtpu.layers.adapter import rmsnorm as tpu_rmsnorm  # noqa: E402

pytestmark = pytest.mark.tpu_test

ROWS = [1, 4, 16, 32, 64, 128, 256]


@pytest.fixture(autouse=True)
def _trial_norm_config(monkeypatch):
    """The norm configuration these were measured under: the one-pass Pallas
    add+RMSNorm on (TPU_FUSED_ADD_RMSNORM, a recipe setting on glm-agentx,
    where it defaults off) and the materialized decode norm armed per test."""
    monkeypatch.setenv("TPU_FUSED_ADD_RMSNORM", "1")


def _norm(hidden=6144, eps=1e-5, seed=0):
    from vllm.model_executor.layers.layernorm import RMSNorm

    torch.manual_seed(seed)
    n = RMSNorm(hidden, eps=eps)
    assert isinstance(n, tpu_rmsnorm.TpuRMSNorm)
    with torch.no_grad():
        n.weight.copy_(1 + 0.2 * torch.randn(hidden))
    return n.to(torch.bfloat16).to("tpu")


def _arm(n, on):
    n._materialize = on
    if on:
        n._barrier = tpu_rmsnorm._get_barrier_op()


def _faulty(n):
    """Positive control: variance from the bf16-rounded residual."""

    def fwd(x, residual):
        xf = x.to(torch.float32) + residual.to(torch.float32)
        xf, new_res = n._barrier(xf, xf.to(x.dtype))
        variance = new_res.float().pow(2).mean(dim=-1, keepdim=True)
        y = xf * torch.rsqrt(variance + n.variance_epsilon)
        return (y.to(n.weight.dtype) * n.weight.data).to(x.dtype), new_res

    return fwd


def _same(a, b):
    return torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))


@pytest.mark.parametrize("rows", ROWS)
def test_materialized_norm_bitwise(vllm_config_context, rows):
    torch._dynamo.reset()
    n = _norm(seed=rows)
    _arm(n, True)  # TPU_RMSNORM_DECODE_MATERIALIZE defaults off
    x = (0.7 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu")
    assert not n._use_pallas(x, r)

    def ref_fn(a, b):
        return n.forward_native(a, b)

    def on_fn(a, b):
        return n._materialized_add_norm(a, b)

    ref = torch.compile(ref_fn, fullgraph=True, dynamic=False)(x.clone(), r.clone())
    got = torch.compile(on_fn, fullgraph=True, dynamic=False)(x.clone(), r.clone())
    assert _same(got[1], ref[1]), "residual"
    assert _same(got[0], ref[0]), "normed"
    bad = torch.compile(_faulty(_arm(n, True) or n), fullgraph=True, dynamic=False)(
        x.clone(), r.clone()
    )
    print(f"rows {rows}: bitwise OK; fault caught: {not _same(bad[0], ref[0])}")


@pytest.mark.parametrize("rows", [16, 64])
def test_chain_bitwise(vllm_config_context, rows):
    """Four residual norms with matmuls between, one compiled graph, flag on
    vs off (the module's forward dispatch, as the model calls it)."""
    torch._dynamo.reset()
    norms = [_norm(seed=100 + i) for i in range(4)]
    torch.manual_seed(rows)
    ws = [(torch.randn(6144, 6144) / 80).to(torch.bfloat16).to("tpu") for _ in range(4)]
    h0 = (2 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu")

    def chain(h):
        x, res = h, h * 0.5
        outs = []
        for nm, w in zip(norms, ws):
            y, res = nm(x, res)
            x = y @ w
            outs.append(y)
        return torch.stack(outs), res

    outs = {}
    for on in (False, True):
        for nm in norms:
            _arm(nm, on)
        torch._dynamo.reset()
        outs[on] = torch.compile(chain, fullgraph=True, dynamic=False)(h0.clone())
    assert _same(outs[True][0], outs[False][0]), "chain normed"
    assert _same(outs[True][1], outs[False][1]), "chain residual"
    print(f"chain rows {rows}: bitwise OK")


def test_fault_caught(vllm_config_context):
    torch._dynamo.reset()
    n = _norm(seed=7)
    _arm(n, True)
    x = (0.7 * torch.randn(16, 6144)).to(torch.bfloat16).to("tpu")
    r = (40 * torch.randn(16, 6144)).to(torch.bfloat16).to("tpu")
    ref = torch.compile(lambda a, b: n.forward_native(a, b), fullgraph=True)(x, r)
    bad = torch.compile(_faulty(n), fullgraph=True)(x, r)
    assert not _same(bad[0], ref[0]), "positive control not caught"


def test_shared_e0_buffer(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_tp_decode as m

    monkeypatch.setattr(envs, "TPU_MOE_TP_DECODE_SHARED_META", True)
    m._E0_BUFFERS.clear()
    dev = torch.device("cpu")
    a, b, c = m._e0_tensor(dev, 0), m._e0_tensor(dev, 0), m._e0_tensor(dev, 64)
    assert a is b and a is not c and int(c) == 64
    monkeypatch.setattr(envs, "TPU_MOE_TP_DECODE_SHARED_META", False)
    assert m._e0_tensor(dev, 0) is not m._e0_tensor(dev, 0)
