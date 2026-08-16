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
"""With ``USE_MOE_EP_KERNEL`` unset, this feature does not exist.

That is the claim the whole change rests on, so it is checked directly against
the real ``VllmFp8MoEMethodTPU.apply_monolithic`` bytecode rather than against a
reimplementation of it: the fused module is replaced by an object that raises on
*any* attribute access, and the forward is required to complete anyway.

Then the same forward is captured by Dynamo with ``fullgraph=True`` and the
graph is required to be node-for-node identical to the graph of the GMM path
alone -- i.e. the guard folds away at trace time and contributes nothing to the
compiled program.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.vllm.quantization import fp8 as fp8_mod

HIDDEN, EXPERTS, TOP_K = 8, 4, 2


class _Exploding:
    """Touching the fused EP module at all is the failure this test detects."""

    def __getattr__(self, name):
        raise AssertionError(
            f"the GMM path reached fused_moe_ep.{name} with the flag unset")


def _layer():
    return SimpleNamespace(
        moe_config=SimpleNamespace(experts_per_token=TOP_K),
        _tpu_activation_str="silu",
        _experts_start=None,
        w13_weight=torch.zeros(EXPERTS, HIDDEN, 2 * HIDDEN),
        w2_weight=torch.zeros(EXPERTS, HIDDEN, HIDDEN),
        w13_weight_scale_inv=None,
        w2_weight_scale_inv=None,
    )


def _stub_gmm_and_routing(monkeypatch, calls):

    def _route(layer, hidden_states, router_logits, input_ids=None):
        rows = hidden_states.shape[0]
        return (torch.ones(rows,
                           TOP_K), torch.zeros(rows, TOP_K, dtype=torch.int32))

    def _gmm(*, hidden_states, **kwargs):
        calls.append(kwargs)
        return hidden_states * 2.0

    monkeypatch.setattr(fp8_mod.moe_routing, "route", _route)
    monkeypatch.setattr(fp8_mod, "fused_moe_gmm", _gmm)


def _apply(method_self, layer, x):
    """Call the real method body, unbound, on a stand-in ``self``."""
    return fp8_mod.VllmFp8MoEMethodTPU.apply_monolithic(
        method_self, layer, x, torch.zeros(x.shape[0], EXPERTS))


def test_the_flag_is_off_by_default():
    """The one env var this feature adds to a default deployment: none."""
    assert envs.USE_MOE_EP_KERNEL is False


@pytest.mark.parametrize(
    "method_self",
    [SimpleNamespace(),
     SimpleNamespace(_fused_ep_enabled=False)],
    ids=["attribute_absent", "attribute_false"])
def test_the_forward_never_touches_the_fused_module(monkeypatch, method_self):
    calls = []
    _stub_gmm_and_routing(monkeypatch, calls)
    monkeypatch.setattr(fp8_mod, "fused_moe_ep", _Exploding())

    x = torch.ones(4, HIDDEN)
    out = _apply(method_self, _layer(), x)

    torch.testing.assert_close(out, x * 2.0)
    assert len(calls) == 1, "the GMM path must still run exactly once"


def test_the_sentinel_is_live(monkeypatch):
    """Positive control for the test above: flag on, the sentinel does fire."""
    _stub_gmm_and_routing(monkeypatch, [])
    monkeypatch.setattr(fp8_mod, "fused_moe_ep", _Exploding())

    with pytest.raises(AssertionError, match="reached fused_moe_ep"):
        _apply(SimpleNamespace(_fused_ep_enabled=True), _layer(),
               torch.ones(4, HIDDEN))


def test_the_layer_does_not_claim_the_runners_dispatch(monkeypatch):
    """``supports_internal_mk`` False keeps vLLM's DP dispatch/combine on."""
    monkeypatch.setattr(fp8_mod, "fused_moe_ep", _Exploding())
    method_self = SimpleNamespace()
    assert fp8_mod.VllmFp8MoEMethodTPU.supports_internal_mk.fget(
        method_self) is False


def _captured_graph(fn, x):
    """Dynamo's graph for ``fn`` as (op, target, args, kwargs) per node.

    Compared structurally rather than as text: ``print_readable`` embeds the
    source file and line of every node, which necessarily differ between two
    functions that emit the same program.
    """
    graphs = []

    def _record(gm, _example_inputs):
        graphs.append([(n.op, str(n.target), str(n.args), str(n.kwargs))
                       for n in gm.graph.nodes])
        return gm.forward

    torch._dynamo.reset()
    torch.compile(fn, fullgraph=True, backend=_record)(x)
    assert len(graphs) == 1, f"expected one graph, got {len(graphs)}"
    return graphs[0]


def test_the_traced_program_is_the_gmm_path_alone(monkeypatch):
    """The guard folds at trace time: same graph, with and without it.

    ``reference`` is the body of ``apply_monolithic`` from the GMM branch
    onwards, reached without the guard ever being evaluated. If the guard
    emitted anything -- a tensor read, a select, a graph break -- the two
    graphs would differ.
    """
    calls = []
    _stub_gmm_and_routing(monkeypatch, calls)
    monkeypatch.setattr(fp8_mod, "fused_moe_ep", _Exploding())
    layer = _layer()
    method_self = SimpleNamespace()

    def guarded(x):
        return _apply(method_self, layer, x)

    def reference(x):
        topk_weights, topk_ids = fp8_mod.moe_routing.route(
            layer, x, torch.zeros(x.shape[0], EXPERTS))
        if envs.TPU_MOE_SKIP_PADDED_TOKENS:
            topk_ids, topk_weights = (
                fp8_mod.token_padding.zero_routing_weights_for_padding(
                    topk_ids, topk_weights))
        return fp8_mod.fused_moe_gmm(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=layer._experts_start,
            topk=TOP_K,
            activation=layer._tpu_activation_str,
        )

    x = torch.ones(4, HIDDEN)
    assert _captured_graph(guarded, x) == _captured_graph(reference, x)
