# SPDX-License-Identifier: Apache-2.0
"""TPU_MOE_ROUTER_HBM_PIN at the torch level: the gate's wrapped linear method vs the
F.linear the gate ran before, eager and compiled: prefill token counts (plain dot) bit
for bit; decode counts (HBM-streaming kernel) within one bf16 ulp, few elements off
(the MXU accumulation order differs from XLA's)."""

import pytest

torch = pytest.importorskip("torch")

from vllm_torchtpu.layers.adapter import moe_tp_decode  # noqa: E402

H, E = 6144, 256


class _Base:
    """Stands in for vLLM's UnquantizedLinearMethod (F.linear)."""

    marker = "base"

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class _Gate(torch.nn.Module):
    def __init__(self, e=E):
        super().__init__()
        w = (0.02 * torch.randn(e, H)).to(torch.bfloat16)
        self.weight = torch.nn.Parameter(w.to("tpu"), requires_grad=False)
        self.bias = None
        self.quant_method = _Base()

    def forward(self, x):
        return self.quant_method.apply(self, x, None).to(torch.float32)


class _Runner:
    def __init__(self, gate):
        self.gate = gate


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("e", [128, 256])
def test_pinned_router_bit_identical(compiled, e):
    torch.manual_seed(e)
    gate = _Gate(e)
    xs = {t: (torch.randn(t, H) * 2).to(torch.bfloat16).to("tpu") for t in (1, 4, 8, 16, 32, 64)}
    torch._dynamo.reset()
    run = torch.compile(gate, fullgraph=True, dynamic=False) if compiled else gate
    ref = {t: run(x).cpu() for t, x in xs.items()}
    assert moe_tp_decode._pin_router(_Runner(gate))
    assert isinstance(gate.quant_method, moe_tp_decode.RouterHbmPinMethod)
    assert gate.quant_method.marker == "base"  # delegation
    assert moe_tp_decode._pin_router(_Runner(gate))  # idempotent
    torch._dynamo.reset()
    run = torch.compile(gate, fullgraph=True, dynamic=False) if compiled else gate
    for t, x in xs.items():
        got = run(x).cpu()
        n = int((got.view(torch.int32) != ref[t].view(torch.int32)).sum())
        rel = float(((got - ref[t]).abs() / ref[t].abs().clamp_min(1e-3)).max())
        print(f"ROUTERPIN e={e} t={t} compiled={compiled} mismatches={n} max_rel={rel:.2e}")
        if t > 16:
            assert n == 0
        else:
            assert n <= max(1, t * e // 1000) and rel <= 2.0**-7


def test_refuses_non_bf16_or_biased():
    g = _Gate()
    g.weight = torch.nn.Parameter(g.weight.float(), requires_grad=False)
    assert not moe_tp_decode._pin_router(_Runner(g))
    g2 = _Gate()
    g2.bias = torch.nn.Parameter(torch.zeros(E).to("tpu"), requires_grad=False)
    assert not moe_tp_decode._pin_router(_Runner(g2))
