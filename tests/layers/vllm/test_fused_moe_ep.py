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


@pytest.fixture
def dp_coordinated(monkeypatch):
    """The runner's declaration that every DP rank has the same token count.

    ``engages_for`` declines without it, because the kernel's barrier needs
    every rank to enter -- see ``dp_step_is_coordinated``.
    """
    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", True, raising=False)


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


def test_the_threshold_defaults_to_the_jax_recipes_value(monkeypatch):
    """A default of 0 means the kernel takes decode steps too: -86% e2e."""
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", raising=False)
    assert envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS == 1024


@pytest.mark.parametrize(
    "threshold,rows,dp_size,expected",
    [
        (0, 8, 1, True),  # 0 disables the threshold entirely
        (1024, 2048, 8, True),  # DP8_EP full prefill: 16384 node-wide
        (1024, 64, 8, False),  # DP8_EP decode: 512 node-wide
        (1024, 1024, 1, True),  # TP8_EP: rows are already node-wide
        (1024, 1023, 1, False),
    ])
def test_token_threshold_picks_the_path(tp1, monkeypatch, dp_coordinated,
                                        threshold, rows, dp_size, expected):
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", threshold)
    layer = _layer(dp_size=dp_size)
    assert fused_moe_ep.engages_for(layer, torch.zeros(rows, 4)) is expected


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


@pytest.mark.parametrize("raw,expected", [
    ("", None),
    ("128,512,1024,2048,64,512,1024,2048",
     (128, 512, 1024, 2048, 64, 512, 1024, 2048)),
    (" 64, 512 ,1024,2048,32,512,1024,2048 ",
     (64, 512, 1024, 2048, 32, 512, 1024, 2048)),
])
def test_block_size_override_parses(monkeypatch, raw, expected):
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_BLOCKS", raw)
    assert fused_moe_ep.block_size_override() == expected


def test_block_size_override_rejects_a_wrong_arity(monkeypatch):
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_BLOCKS", "128,512")
    with pytest.raises(ValueError, match="8 comma-separated"):
        fused_moe_ep.block_size_override()


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


def test_version_2_refuses_what_it_cannot_route(tp1, monkeypatch):
    """v2 softmaxes internally and fuses only two activations."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "2")
    layer = _layer()
    layer.scoring_func = "sigmoid"
    assert "scoring_func" in fused_moe_ep.unsupported_reason(layer)
    layer.scoring_func = "softmax"
    layer._tpu_activation_str = "gelu"
    assert "activation" in fused_moe_ep.unsupported_reason(layer)
    # ...and both are fine on v1, which is what makes this a version property.
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "1")
    assert fused_moe_ep.unsupported_reason(layer) is None


def test_an_unknown_version_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "3")
    with pytest.raises(ValueError, match="MOE_FUSED_EP_KERNEL_VERSION=3"):
        fused_moe_ep.kernel_version()


@pytest.mark.parametrize(
    "version,tokens,expect",
    [
        (2, 512, False),  # below the floor
        (2, 2048, True),  # v2's win band opens
        (2, 8192, True),  # 0.59x of the GMM path, measured
        (2, 16384, False),  # 4.79x of it, measured -- the default cap
        (1, 16384, True),  # v1 has no measured cliff, so no default cap
    ])
def test_engagement_is_a_band_not_a_floor(tp1, monkeypatch, dp_coordinated,
                                          version, tokens, expect):
    """v2 is slow at both ends, so one threshold cannot express the policy."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", str(version))
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", "0")  # the cliff is on
    layer = _layer(dp_size=8)
    x = torch.empty(tokens // 8, 4)
    assert fused_moe_ep.engages_for(layer, x) is expect


def test_the_token_cap_can_be_lifted(tp1, monkeypatch, dp_coordinated):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "2")
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", "0")
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", "0")
    assert fused_moe_ep.max_tokens() == 0
    assert fused_moe_ep.engages_for(_layer(dp_size=8), torch.empty(2048, 4))


def test_sizing_the_slab_lifts_the_token_cap(tp1, monkeypatch, dp_coordinated):
    """The cap and the slab are the same fact seen twice.

    v2's default 8192-token cap exists because the drop-free slab makes 16384
    cost 15,620 us/layer against production's 3,264. A deployment that sizes
    the slab has removed that reason, so it should not also have to know to
    lift the cap by hand.
    """
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "2")
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", "0")
    layer, big = _layer(dp_size=8), torch.empty(16384 // 8, 4)

    assert fused_moe_ep.max_tokens() == 8192
    assert not fused_moe_ep.engages_for(layer, big)

    monkeypatch.delenv("MOE_FUSED_EP_V2_CAPACITY_SKEW")  # back to the default
    assert fused_moe_ep.capacity_skew() == 3.0
    assert fused_moe_ep.max_tokens() == 0
    assert fused_moe_ep.engages_for(layer, big)

    # ...and an explicit cap still wins over both.
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", "4096")
    assert fused_moe_ep.max_tokens() == 4096


def test_the_slab_is_sized_by_default_on_v2_and_never_on_v1(tp1, monkeypatch):
    """The default is ON, because the drop-free slab IS the 16384 cliff."""
    monkeypatch.delenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "2")
    assert fused_moe_ep.capacity_skew() == 3.0
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "1")
    assert fused_moe_ep.capacity_skew() == 0.0
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "2")
    monkeypatch.setenv("MOE_FUSED_EP_V2_CAPACITY_SKEW", "0")
    assert fused_moe_ep.capacity_skew() == 0.0


def test_a_step_the_dp_ranks_have_not_agreed_on_is_declined(tp1, monkeypatch):
    """The deadlock guard. ``engages_for`` runs independently on every rank; if
    one engages and another does not, the entrant waits on a barrier signal
    that never comes, and the SparseCore is what halts."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "2")
    layer, big = _layer(dp_size=8), torch.empty(2048, 4)
    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", None, raising=False)
    assert not fused_moe_ep.engages_for(layer, big)
    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", True, raising=False)
    assert fused_moe_ep.engages_for(layer, big)


def test_without_dp_there_is_nothing_to_agree_on(tp1, monkeypatch):
    """At dp_size 1 the TP ranks hold the same tensor, so no rendezvous is
    needed and nobody declaring must not disable the kernel."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "2")
    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", None, raising=False)
    layer = _layer(dp_size=1, ep_size=8)
    assert fused_moe_ep.dp_agreed_tokens(layer, torch.empty(2048, 4)) == 2048
