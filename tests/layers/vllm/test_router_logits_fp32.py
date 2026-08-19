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
"""`TPU_MOE_ROUTER_LOGITS_FP32`: the router's precision, not the softmax's.

The gate is a `ReplicatedLinear` at the model dtype, so `router_logits` arrives
in bf16 and **11.4% of Qwen3.5 tokens have two experts bit-identical at the
top-k boundary** (`land/TIE_FLOOR_NOTE.md`). Scoring in f32 afterwards cannot
recover the preference; the cast that destroyed it is the gate's output. These
tests pin the mechanism, the refusals, and the one property that makes it worth
shipping: the arithmetic is unchanged, only the rounding is removed.
"""


import pytest
import torch

from vllm_torchtpu.layers.vllm import moe_routing

HIDDEN, EXPERTS = 64, 16


class _Unquantized:
    """Stands in for the shared `UnquantizedLinearMethod` singleton."""

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class _Gate(torch.nn.Module):

    def __init__(self, experts=EXPERTS, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(experts, HIDDEN, dtype=dtype))
        self.bias = None
        self.quant_method = _SHARED_METHOD

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias), None


_SHARED_METHOD = _Unquantized()


class _FakeMoE(torch.nn.Module):
    """Only what `_resolve_router_gates` reads."""

    def __init__(self, experts=EXPERTS):
        super().__init__()
        self.global_num_experts = experts


def _model(experts=EXPERTS, gate_experts=None, gate_name="gate"):
    block = torch.nn.Module()
    block.add_module(gate_name, _Gate(gate_experts or experts))
    block.add_module("experts", _FakeMoE(experts))
    mlp = torch.nn.Module()
    mlp.add_module("mlp", block)
    model = torch.nn.Module()
    model.add_module("layer0", mlp)
    return model


@pytest.fixture(autouse=True)
def fused_ep_off(monkeypatch):
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "0")


@pytest.fixture(autouse=True)
def fused_moe_is_our_stub(monkeypatch):
    """`isinstance(module, RoutedExperts)` without a real vLLM MoE layer."""
    import vllm.model_executor.layers.fused_moe as moe_pkg
    monkeypatch.setattr(moe_pkg, "RoutedExperts", _FakeMoE)


def test_the_gate_emits_f32_and_the_arithmetic_is_unchanged():
    model = _model()
    gate = model.layer0.mlp.gate
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16)

    before, _ = gate(x)
    assert before.dtype == torch.bfloat16

    assert moe_routing.promote_router_gates_to_fp32(model) == 1
    after, _ = gate(x)
    assert after.dtype == torch.float32

    # Same numbers, more of them kept: on CPU an f32 matmul of bf16-exact
    # operands is the exact accumulator, and rounding it back reproduces the
    # bf16 result. That is the claim -- the MXU already accumulated in f32.
    torch.testing.assert_close(after.to(torch.bfloat16), before)


def test_the_weight_is_not_upcast_in_place():
    """An f32 stored weight would double the gate's HBM read every step.

    Measured on a compiled tpu7x program: casting in-graph leaves argument size
    unchanged at 4.325 MB, storing f32 takes it to 8.52 MB (d6/fp32logits).
    """
    model = _model()
    moe_routing.promote_router_gates_to_fp32(model)
    assert model.layer0.mlp.gate.weight.dtype == torch.bfloat16


def test_other_linears_keep_their_shared_quant_method():
    """Quant methods are shared singletons; mutating one must not touch them all."""
    model = _model()
    other = _Gate()
    assert other.quant_method is _SHARED_METHOD
    moe_routing.promote_router_gates_to_fp32(model)
    assert other.quant_method is _SHARED_METHOD
    assert other(torch.randn(4, HIDDEN,
                             dtype=torch.bfloat16))[0].dtype == torch.bfloat16


def test_promotion_is_idempotent():
    model = _model()
    moe_routing.promote_router_gates_to_fp32(model)
    first = model.layer0.mlp.gate.quant_method
    moe_routing.promote_router_gates_to_fp32(model)
    assert model.layer0.mlp.gate.quant_method is first


def test_a_gate_that_is_not_where_we_look_is_an_error():
    model = _model(gate_name="router")
    with pytest.raises(RuntimeError, match="is not at"):
        moe_routing.promote_router_gates_to_fp32(model)


def test_a_gate_with_the_wrong_width_is_an_error():
    model = _model(experts=EXPERTS, gate_experts=EXPERTS + 1)
    with pytest.raises(RuntimeError, match="that is not the router gate"):
        moe_routing.promote_router_gates_to_fp32(model)


def test_it_refuses_to_compose_with_the_fused_ep_kernel(monkeypatch):
    """The fused kernel selects experts inside the Pallas program.

    Its `gating_output` is a typed operand, so f32 logits are either refused by
    the kernel or rounded back before the selector sees them -- which buys
    nothing. Refuse loudly: a silent fallback here would look like a measured
    null.
    """
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "1")
    with pytest.raises(RuntimeError, match="does not compose"):
        moe_routing.promote_router_gates_to_fp32(_model())


def test_a_model_with_no_moe_says_so(caplog):
    model = torch.nn.Module()
    assert moe_routing.promote_router_gates_to_fp32(model) == 0


def test_the_tie_rate_is_what_the_flag_is_for():
    """The mechanism, in six lines, at a scale a unit test can hold.

    Not a re-measurement of the 11.4% (that is `d6/ties/`), just the property
    that makes the flag meaningful: rounding the logits to bf16 manufactures
    exact ties at the top-k boundary that do not exist in f32.
    """
    torch.manual_seed(0)
    logits = torch.randn(4096, 128)
    top_k = 8

    def boundary_ties(x):
        vals, _ = torch.topk(x, top_k + 1, dim=-1)
        return int((vals[:, top_k - 1] == vals[:, top_k]).sum())

    assert boundary_ties(logits) == 0
    assert boundary_ties(logits.to(torch.bfloat16).float()) > 100
