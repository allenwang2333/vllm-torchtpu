# SPDX-License-Identifier: Apache-2.0
"""The fused EP operation is built once per process; each layer is eligible
only when its shape and routing match it."""
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
from vllm_torchtpu.layers.vllm import fused_moe_ep_v3 as bridge

pytestmark = pytest.mark.cpu_test


def _layer(local_experts=64, hidden=4096, inter=1024):
    return SimpleNamespace(w13_weight=torch.zeros(
        (local_experts, hidden, 2 * inter), dtype=torch.float8_e4m3fn),
                           w2_weight=torch.zeros(
                               (local_experts, inter, hidden),
                               dtype=torch.float8_e4m3fn),
                           w13_weight_scale_inv=torch.ones(
                               (local_experts, 2 * inter)),
                           w2_weight_scale_inv=torch.ones(
                               (local_experts, hidden)),
                           global_num_experts=local_experts * 8)


@pytest.fixture
def armed(monkeypatch):
    """A process with no operation yet, every refusal answered 'fine', and
    the operation build replaced by a marker."""
    monkeypatch.setattr(bridge, "_OP", None)
    monkeypatch.setattr(bridge, "_OP_CONTRACT", None)
    monkeypatch.setattr(bridge.envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(bridge.envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 128)
    monkeypatch.setattr(bridge.envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 0)
    monkeypatch.setattr(bridge.envs, "TPU_MOE_SKIP_PADDED_TOKENS", False)
    monkeypatch.setattr(bridge, "_pcp_size", lambda layer: 1)
    monkeypatch.setattr(bridge, "_tensor_parallel_size", lambda: 1)
    monkeypatch.setattr(bridge, "build_ep_mesh",
                        lambda: SimpleNamespace(shape={EP_AXIS_NAME: 8}))
    monkeypatch.setattr(bridge, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(bridge, "_max_node_tokens", lambda ep: 1 << 20)
    monkeypatch.setattr(bridge, "_unsupported_layer_reason",
                        lambda *a, **k: None)
    monkeypatch.setattr(bridge, "_unsupported_routing_reason",
                        lambda layer: None)
    monkeypatch.setattr(bridge, "_unsupported_reason", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "_expert_block_permutation",
                        lambda ep, device: None)

    def build(*args, **kwargs):
        monkeypatch.setattr(bridge, "_OP", object())

    monkeypatch.setattr(bridge, "_build_op", build)


def test_a_layer_of_another_shape_or_routing_is_not_eligible(armed):
    first = _layer()
    bridge.prebuild_fused_moe_ep(first,
                                 topk=10,
                                 renormalize=True,
                                 activation="silu")
    assert bridge.fused_moe_ep_supported() is True
    assert bridge.fused_moe_ep_supported(first) is True

    other_topk = _layer()
    bridge.prebuild_fused_moe_ep(other_topk,
                                 topk=8,
                                 renormalize=True,
                                 activation="silu")
    assert bridge.fused_moe_ep_supported(other_topk) is False

    other_shape = _layer(inter=1536)
    bridge.prebuild_fused_moe_ep(other_shape,
                                 topk=10,
                                 renormalize=True,
                                 activation="silu")
    assert bridge.fused_moe_ep_supported(other_shape) is False

    same = _layer()
    bridge.prebuild_fused_moe_ep(same,
                                 topk=10,
                                 renormalize=True,
                                 activation="silu")
    assert bridge.fused_moe_ep_supported(same) is True


def test_a_refused_layer_stays_refused_after_another_armed(armed, monkeypatch):
    first = _layer()
    bridge.prebuild_fused_moe_ep(first,
                                 topk=10,
                                 renormalize=True,
                                 activation="silu")
    monkeypatch.setattr(bridge, "_unsupported_routing_reason",
                        lambda layer: "grouped top-k")
    refused = _layer()
    bridge.prebuild_fused_moe_ep(refused,
                                 topk=10,
                                 renormalize=True,
                                 activation="silu")
    assert bridge.fused_moe_ep_supported() is True
    assert bridge.fused_moe_ep_supported(refused) is False
    assert bridge.fused_moe_ep_supported(first) is True


def test_nothing_is_eligible_without_the_operation(monkeypatch):
    monkeypatch.setattr(bridge, "_OP", None)
    layer = _layer()
    layer._tpu_fused_ep_eligible = True
    assert bridge.fused_moe_ep_supported(layer) is False
