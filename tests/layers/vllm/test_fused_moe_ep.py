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
"""Torch-side unit tests for the fused expert-parallel MoE bridge."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.vllm import fused_moe_ep


def _layer(**overrides):
    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        ep_size=8,
        dp_size=8,
        num_experts=512,
        experts_per_token=10,
        is_sequence_parallel=False,
        pcp_size=1,
    )
    layer = SimpleNamespace(
        moe_config=moe_config,
        renormalize=True,
        scoring_func="softmax",
        _tpu_activation_str="silu",
        has_bias=False,
    )
    for key, value in overrides.items():
        target = moe_config if hasattr(moe_config, key) else layer
        setattr(target, key, value)
    return layer


@pytest.fixture
def tp1(monkeypatch):
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (1, 0))


@pytest.mark.parametrize("replica_size", [2, 4, 8])
def test_row_slices_partition_the_batch(replica_size):
    """Every row is owned by exactly one replica, and summing rebuilds it.

    That sum is what vLLM's TP all-reduce performs, so this is the numerical
    contract the fused path relies on.
    """
    rows, hidden = 4 * replica_size, 3
    x = torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden)

    total = torch.zeros_like(x)
    for rank in range(replica_size):
        rank_t = torch.tensor(rank, dtype=torch.int32)
        index = fused_moe_ep._row_slice_index(rows, replica_size, rank_t)
        local = torch.index_select(x, 0, index)
        assert local.shape == (rows // replica_size, hidden)
        scattered = fused_moe_ep._scatter_replica_rows(local, replica_size,
                                                       rank_t)
        assert scattered.shape == x.shape
        total += scattered

    torch.testing.assert_close(total, x)


def test_replica_rank_reaches_the_graph_as_data(tp1):
    """The row slice must be indexed by a tensor, not a Python int.

    A Python int would be folded into the traced program as a constant, so
    every rank would compile a structurally different program under one op
    name -- the 2026-07-30 EP desync (see moe_routing.register_experts_start_buffer).
    """
    rank = torch.tensor(1, dtype=torch.int32)
    index = fused_moe_ep._row_slice_index(8, 2, rank)
    assert isinstance(index, torch.Tensor)
    assert index.dtype == torch.int32
    torch.testing.assert_close(index,
                               torch.tensor([4, 5, 6, 7], dtype=torch.int32))


def test_register_replica_rank_buffer_is_none_without_tp(tp1):
    layer = torch.nn.Module()
    fused_moe_ep.register_replica_rank_buffer(layer,
                                              device=torch.device("cpu"))
    assert layer._ep_replica_rank is None


def test_register_replica_rank_buffer_holds_the_rank(monkeypatch):
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (2, 1))
    layer = torch.nn.Module()
    fused_moe_ep.register_replica_rank_buffer(layer,
                                              device=torch.device("cpu"))
    assert layer._ep_replica_rank.item() == 1
    assert layer._ep_replica_rank.dtype == torch.int32
    assert "_ep_replica_rank" not in layer.state_dict()


def test_supported_layout_is_accepted(tp1):
    assert fused_moe_ep.unsupported_reason(_layer()) is None


@pytest.mark.parametrize("overrides,expected", [
    ({
        "moe_parallel_config": SimpleNamespace(use_ep=False)
    }, "expert parallelism is off"),
    ({
        "is_sequence_parallel": True
    }, "sequence parallelism"),
    ({
        "pcp_size": 2
    }, "PCP"),
    ({
        "use_grouped_topk": True
    }, "grouped top-k"),
    ({
        "e_score_correction_bias": torch.zeros(4)
    }, "bias-corrected"),
    ({
        "routed_scaling_factor": 2.5
    }, "routed_scaling_factor"),
    ({
        "scoring_func": "sqrtsoftplus"
    }, "scoring_func"),
    ({
        "has_bias": True
    }, "expert biases"),
    ({
        "num_experts": 500
    }, "divisible"),
    ({
        "dp_size": 4
    }, "not dp_size"),
])
def test_unsupported_layouts_are_refused_with_a_reason(tp1, overrides,
                                                       expected):
    reason = fused_moe_ep.unsupported_reason(_layer(**overrides))
    assert reason is not None and expected in reason


def test_dp4tp2_is_supported_only_with_the_matching_replica_group(monkeypatch):
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (2, 1))
    assert fused_moe_ep.unsupported_reason(_layer(dp_size=4)) is None
    assert "not dp_size" in fused_moe_ep.unsupported_reason(_layer(dp_size=8))


def test_routing_simulation_disables_the_kernel(tp1, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
                       "uniform_random")
    assert "SIMULATION" in fused_moe_ep.unsupported_reason(_layer())


def test_refusal_checks_read_attributes_that_actually_exist():
    """The refusal path degrades silently, so pin the names it reads.

    Every check in ``unsupported_reason`` is a ``getattr`` with a default: if
    upstream renames one of these, the kernel would quietly stop engaging (or
    quietly engage on a configuration it cannot serve) instead of failing.
    """
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig)

    for name in ("num_experts", "experts_per_token", "has_bias", "ep_size",
                 "dp_size", "pcp_size", "is_sequence_parallel", "ep_rank",
                 "moe_parallel_config"):
        assert (name in FusedMoEConfig.__annotations__ or isinstance(
            getattr(FusedMoEConfig, name, None), property)), name
    assert "use_ep" in FusedMoEParallelConfig.__annotations__


def test_flag_defaults_off(tp1, monkeypatch):
    monkeypatch.delenv("USE_MOE_EP_KERNEL", raising=False)
    assert envs.USE_MOE_EP_KERNEL is False
    assert fused_moe_ep.is_enabled(_layer()) is False
