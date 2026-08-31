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
"""Unit tests for vllm_torchtpu.layers.vllm.moe_routing helpers."""

import types
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed import tpu_communicator
from vllm_torchtpu.layers.vllm import fused_moe, moe_routing


def _make_layer(*, use_ep, ep_size=1, ep_rank=0, global_num_experts=0):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
            ep_size=ep_size,
            ep_rank=ep_rank),
        global_num_experts=global_num_experts,
    )


# (ep_size, global_num_experts) -- mix of divisible and non-divisible cases,
# plus the production sizes we actually ship (Qwen3-30B: 128, Qwen3-480B: 160).
@pytest.mark.parametrize("ep_size,global_num_experts", [
    (2, 8),
    (2, 9),
    (4, 18),
    (4, 33),
    (8, 128),
    (8, 160),
])
def test_get_experts_start_matches_determine_expert_map(
        ep_size, global_num_experts):
    """Parity against vLLM's linear-placement contract for every rank."""
    determine_expert_map = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.expert_map_manager"
    ).determine_expert_map
    for ep_rank in range(ep_size):
        _, expert_map, _ = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=global_num_experts,
            expert_placement_strategy="linear",
        )
        owned = (expert_map >= 0).nonzero(as_tuple=True)[0]
        expected = int(owned[0].item()) if owned.numel() else 0
        actual = moe_routing.get_experts_start(
            _make_layer(use_ep=True,
                        ep_size=ep_size,
                        ep_rank=ep_rank,
                        global_num_experts=global_num_experts))
        assert actual == expected, (
            f"ep_size={ep_size} ep_rank={ep_rank} "
            f"global_num_experts={global_num_experts}: "
            f"expected experts_start={expected}, got {actual}")


def test_get_experts_start_returns_none_without_ep():
    assert moe_routing.get_experts_start(_make_layer(use_ep=False)) is None


def test_validate_linear_ep_placement_accepts_linear():
    moe_routing.validate_linear_ep_placement(
        SimpleNamespace(expert_placement_strategy="linear"))


def test_validate_linear_ep_placement_requires_declared_strategy():
    # An object that never declares its placement must fail loudly, not be
    # assumed linear.
    with pytest.raises(AttributeError):
        moe_routing.validate_linear_ep_placement(SimpleNamespace())


def test_validate_linear_ep_placement_rejects_round_robin():
    with pytest.raises(NotImplementedError, match="linear EP placement"):
        moe_routing.validate_linear_ep_placement(
            SimpleNamespace(expert_placement_strategy="round_robin"))


def _routing_layer(topk=10, renormalize=True):
    return SimpleNamespace(use_grouped_topk=False,
                           num_expert_group=None,
                           topk_group=None,
                           renormalize=renormalize,
                           scoring_func="softmax",
                           e_score_correction_bias=None,
                           routed_scaling_factor=1.0,
                           moe_config=SimpleNamespace(experts_per_token=topk))


def test_router_topk_env_defaults_to_rowmax():
    import vllm_torchtpu.envs as envs
    assert envs.environment_variables["TPU_MOE_ROUTER_TOPK"]() == "rowmax"


def test_router_topk_env_rejects_unknown_value(monkeypatch):
    import vllm_torchtpu.envs as envs
    monkeypatch.setenv("TPU_MOE_ROUTER_TOPK", "approx")
    with pytest.raises(ValueError, match="TPU_MOE_ROUTER_TOPK"):
        envs.environment_variables["TPU_MOE_ROUTER_TOPK"]()


def test_select_experts_rowmax_matches_sort(monkeypatch):
    """The gate is behaviour-preserving on tie-free data, end to end."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    hidden = torch.randn(64, 8, dtype=torch.bfloat16)
    logits = torch.randn(64, 512)
    layer = _routing_layer()

    def run():
        return moe_routing.select_experts(hidden_states=hidden,
                                          router_logits=logits,
                                          topk=10,
                                          renormalize=True,
                                          scoring_fn="softmax",
                                          layer=layer)

    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "sort")
    ref_w, ref_i = run()
    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "rowmax")
    got_w, got_i = run()

    assert got_w.dtype == ref_w.dtype and got_i.dtype == torch.int32
    assert torch.equal(got_w, ref_w)
    assert torch.equal(got_i, ref_i.to(torch.int32))


def test_router_topk_op_is_registered_at_import():
    """The op must exist before the first call: a lazy, lock-guarded
    registration is untraceable by Dynamo and aborts the MoE compile."""
    import torch

    from vllm_torchtpu.layers.vllm import router_topk
    assert router_topk._op is not None
    assert hasattr(torch.ops.pallas, "moe_router_topk")


@pytest.mark.parametrize("impl", ["sort", "rowmax"])
def test_routing_path_compiles_fullgraph(monkeypatch, impl):
    """No graph break, no Dynamo error, on the path vLLM compiles.

    The tensors have to be on device: torch_tpu owns the default compile
    backend, so CPU inputs abort the compile before the router is reached.
    """
    torch = pytest.importorskip("torch")
    torch._dynamo.reset()
    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", impl)
    layer = _routing_layer()
    hidden = torch.randn(16, 8, dtype=torch.bfloat16).to("tpu")
    logits = torch.randn(16, 512).to("tpu")

    def run(h, r):
        return moe_routing.select_experts(hidden_states=h,
                                          router_logits=r,
                                          topk=10,
                                          renormalize=True,
                                          scoring_fn="softmax",
                                          layer=layer)

    # backend="eager" on purpose: this test asserts the *Python* path traces
    # without a graph break, which is device-independent. Leaving the backend
    # unset makes _default_backend_selector pick the TPU backend on a TPU host
    # and try to compile these synthetic CPU tensors for device, which fails
    # for both impls and has nothing to do with what the test checks.
    compiled = torch.compile(run,
                             fullgraph=True,
                             backend="eager",
                             dynamic=False)
    got_w, got_i = compiled(hidden, logits)
    ref_w, ref_i = run(hidden, logits)
    assert torch.equal(got_w, ref_w) and torch.equal(got_i, ref_i)


class _FakeDpGroup:
    """A DP group whose all_gather concatenates `world_size` copies."""

    def __init__(self, world_size):
        self.world_size = world_size

    def all_gather(self, x, dim=0):
        return torch.cat([x] * self.world_size, dim=dim)


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("dp_size,pcp_size,expected", [(1, 1, False),
                                                       (2, 1, True),
                                                       (2, 2, False),
                                                       (1, 2, False)])
@pytest.mark.parametrize("withheld", [False, True])
@pytest.mark.parametrize("budget,expect_blocked", [(0, False), (7, False),
                                                   (8, True)])
def test_pre_gather_blocking_follows_the_fused_path(monkeypatch, budget,
                                                    expect_blocked):
    """A blockable SHAPE is not enough; the fused path has to be able to read it.

    Blocking on the shard and un-blocking on the full pool costs a data-format
    call per layer, which is strictly worse than gathering the pool plainly.
    """
    dp_size, num_tokens, hidden_size = 2, 4, 128
    monkeypatch.setattr(tpu_communicator, "get_dp_group",
                        lambda: _FakeDpGroup(dp_size))
    monkeypatch.setattr(tpu_communicator, "router_logits_withheld",
                        lambda: False)
    monkeypatch.setattr(tpu_communicator,
                        "_FPG_MAX_TOKENS", [budget],
                        raising=False)
    tpu_communicator.take_blocked_pool()

    comm = tpu_communicator.TpuDeviceCommunicator.__new__(
        tpu_communicator.TpuDeviceCommunicator)
    out_hidden, _ = comm.dispatch_router_logits(
        torch.zeros(num_tokens, hidden_size), torch.zeros(num_tokens, 8))

    assert (tpu_communicator.take_blocked_pool() is not None) is expect_blocked
    assert out_hidden.shape == (num_tokens * dp_size, hidden_size)


def test_fpg_ceiling_is_zero_when_the_switch_is_off(monkeypatch):
    """Blocking a pool nothing will read costs a SparseCore data-format call per
    layer, so the switch has to reach the ceiling too, not just the dispatch."""
    import vllm_torchtpu.envs as envs

    monkeypatch.setattr(tpu_communicator,
                        "_FPG_MAX_TOKENS", [None],
                        raising=False)
    monkeypatch.setattr(envs, "USE_FUSED_PERMUTE_GMM", False)
    fused_moe._build_fused_moe_custom_op(
        topk=8,
        activation="silu",
        use_ep=True,
        skip_padded_tokens=False,
    )
    assert tpu_communicator.fpg_max_tokens() == 0


def test_fpg_ceiling_takes_the_most_restrictive_layer(monkeypatch):
    monkeypatch.setattr(tpu_communicator,
                        "_FPG_MAX_TOKENS", [None],
                        raising=False)
    tpu_communicator.set_fpg_max_tokens(12000)
    tpu_communicator.set_fpg_max_tokens(0)
    tpu_communicator.set_fpg_max_tokens(12000)
    assert tpu_communicator.fpg_max_tokens() == 0


def test_lane_count_comes_from_the_hardware(monkeypatch):
    """`resolve_hardware` reads the lane count off the device, once."""
    monkeypatch.setattr(tpu_communicator, "_LANES", [0])
    fake = types.SimpleNamespace(num_lanes=256)
    monkeypatch.setattr("jax.experimental.pallas.tpu.get_tpu_info",
                        lambda: fake)
    tpu_communicator.resolve_hardware()
    assert tpu_communicator._LANES[0] == 256


def test_lane_count_survives_a_device_less_host(monkeypatch):
    """Op construction can run before there is a device; the default must stand.

    `get_tpu_info` raises for an unsupported device kind, which is what a CPU host looks
    like. Upstream's own use of it at this same site catches that and falls back.
    """
    monkeypatch.setattr(tpu_communicator, "_LANES", [128])

    def _boom():
        raise ValueError("Unsupported TPU device kind: cpu")

    monkeypatch.setattr("jax.experimental.pallas.tpu.get_tpu_info", _boom)
    tpu_communicator.resolve_hardware()
    assert tpu_communicator._LANES[0] == 128


def test_dispatch_never_reads_the_hardware_while_tracing(monkeypatch):
    """The traced path must read the cached int, never call into the accessor.

    `dispatch_router_logits` runs inside the Dynamo-traced region, where the nanobind
    `get_tpu_info` is untraceable -- an earlier revision called it here and took the whole
    slice down. Making the accessor explode is what pins that it is not on this path.
    """

    def _boom():
        raise AssertionError("get_tpu_info reached from the traced path")

    monkeypatch.setattr("jax.experimental.pallas.tpu.get_tpu_info", _boom)
    monkeypatch.setattr(tpu_communicator, "_LANES", [128])
    monkeypatch.setattr(tpu_communicator, "_dp_initialized", lambda: True)
    monkeypatch.setattr(tpu_communicator, "get_dp_group",
                        lambda: _FakeDpGroup(2))
    monkeypatch.setattr(tpu_communicator, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(tpu_communicator.envs, "TPU_MOE_COLLECTION_CHUNK_SIZE",
                        0)
    tpu_communicator.set_fpg_max_tokens(1 << 20)

    comm = tpu_communicator.TpuDeviceCommunicator.__new__(
        tpu_communicator.TpuDeviceCommunicator)
    monkeypatch.setattr(comm, "_dp_gather", lambda t: t, raising=False)
    hidden = torch.zeros((4, 256), dtype=torch.bfloat16)
    logits = torch.zeros((4, 8), dtype=torch.bfloat16)
    comm.dispatch_router_logits(hidden, logits)

    # Not vacuous: the blocking branch -- the one that reads the lane count, twice -- has to
    # have been taken, or an early return would make this pass without exercising anything.
    assert tpu_communicator.take_blocked_pool() is not None
