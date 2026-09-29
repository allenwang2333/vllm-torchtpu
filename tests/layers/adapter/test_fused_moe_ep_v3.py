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
"""Selection, admission and dispatch of the fused EP MoE kernel v3.

None of this needs a TPU: every refusal is decided before an op is built,
and the op itself is replaced by a stub.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as legacy
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as bridge
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization import (
    fused_moe_ep_v3_dispatch as dispatch,
)

pytestmark = pytest.mark.cpu_test

_EP = 8


def _layer(experts=4, hidden=512, inter=256, **overrides):
    """A layer whose weights and routing v3 would accept."""
    layer = torch.nn.Module()
    for name, tensor in dict(
        w13_weight=torch.zeros(experts, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(experts, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(experts, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(experts, 1, 1, hidden),
    ).items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.activation = "silu"
    layer.renormalize = True
    layer.global_num_experts = experts * _EP
    layer.custom_routing_function = None
    layer.scoring_func = "softmax"
    layer.e_score_correction_bias = None
    layer.use_grouped_topk = False
    layer.num_expert_group = None
    layer.topk_group = None
    layer.routed_scaling_factor = 1.0
    layer.apply_router_weight_on_input = False
    layer.moe_config = SimpleNamespace(
        experts_per_token=2,
        moe_parallel_config=SimpleNamespace(
            use_ep=True, pcp_size=1, is_sequence_parallel=False
        ),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


@pytest.fixture
def v3(monkeypatch):
    """IMPL=v3 with the distributed lookups and the op build stubbed out."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "1024")
    monkeypatch.setattr(bridge, "_OPS", {})
    monkeypatch.setattr(
        bridge,
        "build_ep_mesh",
        lambda: SimpleNamespace(shape={bridge.EP_AXIS_NAME: _EP}),
    )
    monkeypatch.setattr(bridge, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(bridge, "_max_node_tokens", lambda ep: 4096)
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1))
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.pipelined_fused_moe."
        "enable_pipelined_collective_and_compute",
        lambda: False,
    )
    built = []

    def build_op(*args, **kwargs):
        op = Mock(side_effect=lambda x, *rest: x)
        built.append((args, kwargs, op))
        return op

    monkeypatch.setattr(bridge, "_build_op", build_op)
    return SimpleNamespace(built=built, config=config)


def _prebuild(layer, **kwargs):
    defaults = dict(topk=2, renormalize=True, activation="silu")
    defaults.update(kwargs)
    return bridge.prebuild_fused_moe_ep_v3(layer, **defaults)


@pytest.mark.parametrize(
    "selection,enabled",
    [(None, False), ("v2", True), ("adaptive", True), ("v3", True)],
)
def test_v3_is_a_selector_choice_that_enables_fused_ep(monkeypatch, selection, enabled):
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    if selection is None:
        monkeypatch.delenv("MOE_FUSED_EP_KERNEL_IMPL", raising=False)
    else:
        monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", selection)
    assert envs.environment_variables["USE_MOE_FUSED_EP_KERNEL"]() is enabled
    assert dispatch.enabled() is (selection == "v3")


def test_the_op_name_does_not_clash_with_v2():
    assert bridge.FUSED_MOE_EP_V3_OP_NAME != legacy.FUSED_MOE_EP_OP_NAME
    assert bridge.FUSED_MOE_EP_V3_OP_ATTR != legacy.FUSED_MOE_EP_OP_ATTR


def test_arms_on_a_supported_layer(v3):
    assert _prebuild(_layer()) is v3.built[0][2]
    args, kwargs, _ = v3.built[0]
    assert args[1:5] == (2, True, "silu", False)
    assert kwargs == dict(hidden=512, inter=256, local_experts=4)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("tp", "tensor_parallel_size=8"),
        ("pcp", "prefill context parallelism"),
        ("no_ep", "--enable-expert-parallel"),
        ("pipelined", "chunk pipelining"),
    ],
)
def test_an_unservable_deployment_raises(v3, monkeypatch, fault, match):
    layer = _layer()
    if fault == "tp":
        v3.config.parallel_config.tensor_parallel_size = 8
    elif fault == "pcp":
        layer.moe_config.moe_parallel_config.pcp_size = 2
    elif fault == "no_ep":
        layer.moe_config.moe_parallel_config.use_ep = False
    else:
        monkeypatch.setattr(
            "vllm_torchtpu.layers.adapter.pipelined_fused_moe."
            "enable_pipelined_collective_and_compute",
            lambda: True,
        )
    with pytest.raises(NotImplementedError, match=match):
        _prebuild(layer)
    assert not v3.built


@pytest.mark.parametrize(
    "overrides",
    [
        dict(scoring_func="sigmoid"),
        dict(e_score_correction_bias=torch.zeros(32)),
        dict(routed_scaling_factor=2.5),
        dict(use_grouped_topk=True, num_expert_group=8, topk_group=4),
        dict(custom_routing_function=lambda **kw: None),
        dict(apply_router_weight_on_input=True),
        dict(hash_indices_table=torch.zeros(16, 2, dtype=torch.int32)),
    ],
    ids=["sigmoid", "bias", "scaled", "grouped", "custom", "on_input", "hash"],
)
def test_routing_v2_serves_and_v3_does_not_is_declined(v3, overrides):
    """v2's guard admits score-bias and hash routing; v3 cannot compute them."""
    assert _prebuild(_layer(**overrides)) is None
    assert not v3.built


def test_declined_while_the_routing_simulator_is_on(v3, monkeypatch):
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
    assert _prebuild(_layer()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(global_num_experts=4 * _EP + 1),
        dict(w13_bias=torch.zeros(4, 512)),
    ],
    ids=["uneven", "bias"],
)
def test_an_unservable_layer_layout_is_declined(v3, overrides):
    assert _prebuild(_layer(**overrides)) is None


def test_a_non_silu_activation_is_declined(v3):
    assert _prebuild(_layer(), activation="gelu") is None


def test_a_block_scaled_weight_is_declined(v3):
    layer = _layer()
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones(4, 4, 1, 512), requires_grad=False
    )
    assert _prebuild(layer) is None


def test_declined_below_the_token_threshold(v3, monkeypatch):
    monkeypatch.setattr(bridge, "_max_node_tokens", lambda ep: 512)
    assert _prebuild(_layer()) is None


def test_declined_without_an_ep_group(v3, monkeypatch):
    monkeypatch.setattr(bridge, "build_ep_mesh", lambda: None)
    assert _prebuild(_layer()) is None


class _FakeOp:
    """What `jax_op` returns, recording the name it was registered under."""

    def __init__(self, name, fn, **kwargs):
        self.name, self.fn, self.kwargs = name, fn, kwargs

    def register_fake(self, fake):
        self.fake = fake


def test_ops_are_shared_per_closure_and_named_apart_from_v2(monkeypatch):
    """The MTP draft's MoE layer builds its own op when its shape differs and
    shares the target's when it matches; no two closures share a name."""
    monkeypatch.setattr(bridge, "_OPS", {})
    monkeypatch.setattr(bridge, "jax_op", _FakeOp)
    mesh = SimpleNamespace(
        shape={bridge.MESH_AXIS: _EP}, axis_names=(bridge.MESH_AXIS,)
    )

    def build(inter, skip=False):
        return bridge._build_op(
            mesh, 2, True, "silu", skip, hidden=512, inter=inter, local_experts=4
        )

    target = build(256)
    assert build(256) is target
    draft = build(512)
    padded = build(256, skip=True)
    assert target.name == "pallas::fused_moe_ep_v3"
    assert len({target.name, draft.name, padded.name}) == 3
    assert len(target.kwargs["input_partition_specs"]) == 7
    assert len(padded.kwargs["input_partition_specs"]) == 8


def _method():
    method = object.__new__(fp8.VllmFp8MoEMethodTPU)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
    method.weight_block_size = None
    method.weight_scale_name = "weight_scale"
    return method


@pytest.fixture
def fp8_load(v3, monkeypatch):
    """The FP8 method's load with the weight conversion stubbed and v2 and
    the pipelined path forbidden."""
    forbidden = Mock(side_effect=AssertionError("v2 must not be called"))
    monkeypatch.setattr(legacy, "prebuild_fused_moe_ep", forbidden)
    monkeypatch.setattr(fp8, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(fp8, "prebuild_fused_moe_kernel", Mock())
    monkeypatch.setattr(fp8, "pipelined_fused_moe_gmm", forbidden)
    monkeypatch.setattr(fp8, "enable_pipelined_collective_and_compute", lambda: False)
    monkeypatch.setattr(fp8.moe_routing, "validate_linear_ep_placement", Mock())
    monkeypatch.setattr(fp8.moe_routing, "register_experts_start_buffer", Mock())

    def prepare(layer, **kw):
        return (
            layer.w13_weight,
            layer.w13_weight_scale_inv.data,
            layer.w2_weight,
            layer.w2_weight_scale_inv.data,
            "float8_e4m3fn",
            None,
        )

    monkeypatch.setattr(fp8, "_process_fp8_moe_weights", prepare)
    return v3


def test_fp8_method_loads_and_forwards_through_v3(fp8_load, monkeypatch):
    layer, method = _layer(), _method()
    gmm = Mock(side_effect=AssertionError("GMM must not be called"))
    monkeypatch.setattr(fp8, "fused_moe_gmm", gmm)
    assert not method.supports_internal_mk
    method.process_weights_after_loading(layer)
    op = fp8_load.built[0][2]
    assert getattr(method, bridge.FUSED_MOE_EP_V3_OP_ATTR) is op
    assert not legacy.fused_moe_ep_supported(method)
    assert method.supports_internal_mk
    x = torch.ones(3, 512, dtype=torch.bfloat16)
    assert method.apply_monolithic(layer, x, torch.ones(3, 32)) is x
    call = op.call_args.args
    assert len(call) == 7
    assert call[3].shape == (4, 512) and call[4].shape == (4, 512)


def test_fp8_method_keeps_the_grouped_path_for_a_declined_layer(fp8_load, monkeypatch):
    layer, method = _layer(scoring_func="sigmoid"), _method()
    monkeypatch.setattr(fp8, "fused_moe_gmm", Mock(return_value="gmm"))
    monkeypatch.setattr(fp8.moe_routing, "route", Mock(return_value=("weights", "ids")))
    layer._experts_start = None
    method.process_weights_after_loading(layer)
    assert getattr(method, bridge.FUSED_MOE_EP_V3_OP_ATTR) is None
    assert not method.supports_internal_mk
    x = torch.ones(3, 512, dtype=torch.bfloat16)
    assert method.apply_monolithic(layer, x, torch.ones(3, 32)) == "gmm"


def test_skip_padded_tokens_passes_the_valid_count(v3, monkeypatch):
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    layer, owner = _layer(), SimpleNamespace()
    setattr(owner, bridge.FUSED_MOE_EP_V3_OP_ATTR, _prebuild(layer))
    assert v3.built[0][0][4] is True
    state = token_padding.TokenPaddingState.create(8, device="cpu")
    state.update(5, 8)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    x = torch.ones(8, 512, dtype=torch.bfloat16)
    bridge.fused_moe_ep_v3(
        owner,
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        torch.ones(8, 32),
    )
    call = v3.built[0][2].call_args.args
    assert len(call) == 8
    assert call[7].tolist() == [5]


def test_mask_padded_rows_routes_the_suffix_nowhere():
    ep, t_local, hidden, experts = 2, 4, 8, 4
    x = jnp.ones((ep * t_local, hidden), jnp.bfloat16)
    gating = jnp.zeros((ep * t_local, experts), jnp.float32)
    valid = jnp.asarray([3, 1], jnp.int32)
    x, gating = bridge.mask_padded_rows(x, gating, valid, ep)
    keep = np.array([1, 1, 1, 0, 1, 0, 0, 0], bool)
    assert np.all(np.isnan(np.asarray(gating)[~keep]))
    assert not np.any(np.isnan(np.asarray(gating)[keep]))
    assert not np.any(np.asarray(x, np.float32)[~keep])
    assert np.all(np.asarray(x, np.float32)[keep] == 1)


def test_the_runner_reports_an_armed_v3_layer_reduced(monkeypatch):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    import vllm_torchtpu

    class Runner:
        _fused_output_is_reduced = property(lambda self: False)

    monkeypatch.setattr(mr, "MoERunner", Runner)
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v3")
    vllm_torchtpu._patch_moe_runner_fused_output_is_reduced()
    runner = Runner()
    runner.moe_config = SimpleNamespace(skip_final_all_reduce=False)
    runner._quant_method = SimpleNamespace()
    assert runner._fused_output_is_reduced is False
    setattr(runner._quant_method, bridge.FUSED_MOE_EP_V3_OP_ATTR, object())
    assert runner._fused_output_is_reduced is True
