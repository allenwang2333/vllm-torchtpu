# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_FUSED_ADD_RMSNORM_QUANT torch plumbing: a TpuRMSNorm paired with a
per-channel FP8 linear hands it (y_q, scale), and the linear's output and the
residual are bit-identical to the unpaired path, eager and under
``torch.compile(fullgraph=True)``. Decode-sized rows keep the old path.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_torchtpu.layers.adapter import rmsnorm as tpu_rmsnorm  # noqa: E402
from vllm_torchtpu.layers.adapter.linear_common import PREQUANT_ATTR  # noqa: E402
from vllm_torchtpu.layers.adapter.quantization.fp8 import (  # noqa: E402
    VllmFp8LinearMethodTPU,
)

pytestmark = pytest.mark.tpu_test

H = 6144
N = 2624


def _norm():
    from vllm.model_executor.layers.layernorm import RMSNorm

    n = RMSNorm(H, eps=1e-5)
    assert isinstance(n, tpu_rmsnorm.TpuRMSNorm)
    with torch.no_grad():
        n.weight.copy_(1 + 0.2 * torch.randn(H))
    return n.to(torch.bfloat16).to("tpu")


class _Linear(torch.nn.Module):
    """The attributes VllmFp8LinearMethodTPU.apply reads: (k, n) fp8 weight,
    1-D f32 per-channel scale, no bias."""

    def __init__(self):
        super().__init__()
        w = (64 * torch.randn(H, N)).to(torch.float8_e4m3fn)
        self.weight = torch.nn.Parameter(w.to("tpu"), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            (1e-4 + 9e-4 * torch.rand(N)).to("tpu"), requires_grad=False
        )
        self.bias = None
        self.quant_method = object.__new__(VllmFp8LinearMethodTPU)

    def forward(self, x):
        return self.quant_method.apply(self, x)


def _layer(n, lin, x, r):
    y, res = n(x, r)
    return lin(y), res


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows", [1024, 4096])
def test_paired_matches_unpaired(vllm_config_context, rows, compiled):
    torch.manual_seed(rows)
    torch._dynamo.reset()
    n, lin = _norm(), _Linear()
    x = (0.7 * torch.randn(rows, H)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(rows, H)).to(torch.bfloat16).to("tpu")
    fn = torch.compile(_layer, fullgraph=True, dynamic=False) if compiled else _layer
    ref_out, ref_res = fn(n, lin, x.clone(), r.clone())
    assert tpu_rmsnorm._pair(n, lin, False)
    torch._dynamo.reset()
    fn = torch.compile(_layer, fullgraph=True, dynamic=False) if compiled else _layer
    got_out, got_res = fn(n, lin, x.clone(), r.clone())
    assert getattr(lin, PREQUANT_ATTR) is None, "the linear must consume the stash"
    assert torch.equal(got_res.cpu(), ref_res.cpu())
    d = (got_out.cpu().view(torch.int16) != ref_out.cpu().view(torch.int16)).sum()
    print(f"PAIRED rows={rows} compiled={compiled} out_mismatch={int(d)}")
    assert int(d) == 0


def test_paired_fault_control(vllm_config_context):
    """A stash from a different input must change the output (the linear really
    reads the stash, not x)."""
    torch.manual_seed(5)
    n, lin = _norm(), _Linear()
    assert tpu_rmsnorm._pair(n, lin, False)
    x = (0.7 * torch.randn(1024, H)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(1024, H)).to(torch.bfloat16).to("tpu")
    good, _ = _layer(n, lin, x.clone(), r.clone())
    y, _ = n(x.clone(), r.clone())  # sets the stash ...
    other = (0.7 * torch.randn(1024, H)).to(torch.bfloat16).to("tpu")
    bad = lin(other)  # ... which the linear reads instead of `other`
    assert torch.equal(bad.cpu(), good.cpu())
    unpaired = lin(other)  # stash consumed: now it really is `other`
    assert not torch.equal(unpaired.cpu(), good.cpu())


def test_decode_rows_keep_old_path(vllm_config_context):
    n, lin = _norm(), _Linear()
    assert tpu_rmsnorm._pair(n, lin, True)
    x = torch.randn(16, H).to(torch.bfloat16).to("tpu")
    n(x, x.clone())
    assert getattr(lin, PREQUANT_ATTR) is None


def test_compiled_first_uses_op_built_at_arming(vllm_config_context):
    """Regression (twin j-367e571f): the consumer op must be built by arming,
    not lazily under Dynamo (a Lock in the compiled region fails the capture).
    Start with no op built and run the compiled forward first."""
    from vllm_torchtpu.layers.adapter import linear_common

    torch.manual_seed(9)
    torch._dynamo.reset()
    linear_common._prequantized_matmul_op = None
    n, lin = _norm(), _Linear()
    x = (0.7 * torch.randn(1024, H)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(1024, H)).to(torch.bfloat16).to("tpu")
    ref_out, ref_res = _layer(n, lin, x.clone(), r.clone())  # unpaired, eager
    linear_common._prequantized_matmul_op = None
    assert tpu_rmsnorm._pair(n, lin, False)
    assert linear_common._prequantized_matmul_op is not None
    fn = torch.compile(_layer, fullgraph=True, dynamic=False)
    got_out, got_res = fn(n, lin, x.clone(), r.clone())
    assert torch.equal(got_res.cpu(), ref_res.cpu())
    assert torch.equal(got_out.cpu(), ref_out.cpu())
