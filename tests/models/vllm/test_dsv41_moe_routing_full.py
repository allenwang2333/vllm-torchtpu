# Copyright 2025 Google LLC
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
"""100% statement & branch unit tests for vllm_torchtpu.layers.adapter.moe_routing."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel in (
    ("vllm_torchtpu.layers", "vllm_torchtpu/layers"),
    ("vllm_torchtpu.layers.adapter", "vllm_torchtpu/layers/adapter"),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR) / _rel)]  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg


class _StubMeta(type):
    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock()


class _StubModule(types.ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name.isupper():
            return None
        if name[:1].isupper():
            return _StubMeta(name, (), {"__init__": lambda self, *a, **k: None})
        return MagicMock()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".", 1)[0]
        if root in {"vllm", "torch_tpu"}:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        mod = _StubModule(spec.name)
        if (
            spec.name
            == "vllm.model_executor.layers.fused_moe.router.routing_simulator_router"
        ):
            mod.RoutingSimulator = types.SimpleNamespace(  # type: ignore[attr-defined]
                get_available_strategies=staticmethod(lambda: []),
                _routing_strategies={},
            )
        return mod

    def exec_module(self, module):
        return None


try:
    import vllm  # noqa: F401
except ModuleNotFoundError:
    sys.meta_path.insert(0, _StubFinder())

import vllm.envs as _vllm_envs  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (  # noqa: E402
    RoutingSimulator as _RoutingSimulator,
)

if not isinstance(getattr(_RoutingSimulator, "_routing_strategies", None), dict):
    _RoutingSimulator._routing_strategies = {}  # type: ignore[attr-defined]
if not callable(getattr(_RoutingSimulator, "get_available_strategies", None)):
    _RoutingSimulator.get_available_strategies = staticmethod(lambda: [])  # type: ignore[attr-defined]

_STUB_KEYS = (
    "vllm_torchtpu.envs",
    "vllm_torchtpu.layers.adapter.router_topk",
)


def _load_moe_routing(
    sim_strategy: str | None = None,
    available_strategies: list[str] | None = None,
    custom_strategy_obj=None,
    router_topk: str = "rowmax",
):
    saved = {k: sys.modules.get(k) for k in _STUB_KEYS}
    saved_vllm_envs = getattr(_vllm_envs, "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", None)

    fake_envs = types.ModuleType("vllm_torchtpu.envs")
    fake_envs.TPU_MOE_ROUTER_TOPK = router_topk  # type: ignore[attr-defined]
    sys.modules["vllm_torchtpu.envs"] = fake_envs

    fake_router_topk = types.ModuleType("vllm_torchtpu.layers.adapter.router_topk")

    def _stub_rowmax_select(scores, k, *, bias=None, renormalize=False):
        eff = (
            scores + bias.float().view((1,) * (scores.dim() - 1) + (-1,))
            if bias is not None
            else scores
        )
        _, ids = torch.topk(eff, k=k, dim=-1)
        w = scores.gather(-1, ids)
        if renormalize:
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        return w, ids.to(torch.int32)

    fake_router_topk.rowmax_select = _stub_rowmax_select  # type: ignore[attr-defined]
    sys.modules["vllm_torchtpu.layers.adapter.router_topk"] = fake_router_topk

    old_avail = _RoutingSimulator.get_available_strategies
    old_map = dict(_RoutingSimulator._routing_strategies)
    try:
        _vllm_envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY = sim_strategy
        if available_strategies is not None:
            _RoutingSimulator.get_available_strategies = staticmethod(  # type: ignore[method-assign]
                lambda: available_strategies
            )
        if custom_strategy_obj is not None and sim_strategy is not None:
            _RoutingSimulator._routing_strategies[sim_strategy] = custom_strategy_obj
        sys.modules.pop("vllm_torchtpu.layers.adapter.moe_routing", None)
        import vllm_torchtpu.layers.adapter.moe_routing as mod

        mod = importlib.reload(mod)
        mod._ROUTER_TOPK = router_topk
        return mod
    finally:
        _RoutingSimulator.get_available_strategies = old_avail  # type: ignore[method-assign]
        _RoutingSimulator._routing_strategies.clear()
        _RoutingSimulator._routing_strategies.update(old_map)
        _vllm_envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY = saved_vllm_envs
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _FakeGroup:
    def __init__(self, world_size: int = 2, rank_in_group: int = 1):
        self.world_size = world_size
        self.rank_in_group = rank_in_group

    def all_gather(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.cat([t for _ in range(self.world_size)], dim=dim)


class _FakeLayer:
    def __init__(
        self,
        *,
        experts_per_token: int = 2,
        renormalize: bool = True,
        scoring_func: str = "softmax",
        custom_routing_function=None,
        is_sequence_parallel: bool = False,
        use_ep: bool = False,
        dp_size: int = 1,
        ep_size: int = 2,
        ep_rank: int = 1,
        global_num_experts: int = 5,
        hash_indices_table: torch.Tensor | None = None,
        use_grouped_topk: bool = False,
        num_expert_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        expert_placement_strategy: str = "linear",
    ):
        self.moe_config = types.SimpleNamespace(
            experts_per_token=experts_per_token,
            is_sequence_parallel=is_sequence_parallel,
            moe_parallel_config=types.SimpleNamespace(use_ep=use_ep),
            dp_size=dp_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.custom_routing_function = custom_routing_function
        self.global_num_experts = global_num_experts
        self.hash_indices_table = hash_indices_table
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.expert_placement_strategy = expert_placement_strategy
        self._buffers: dict[str, torch.Tensor | None] = {}

    def register_buffer(
        self, name: str, tensor: torch.Tensor | None, persistent: bool = True
    ) -> None:
        self._buffers[name] = tensor


class _DeviceViewTensor(torch.Tensor):
    _fake_device_type = "tpu"

    @property
    def device(self):  # type: ignore[override]
        return types.SimpleNamespace(type=self._fake_device_type)


def _as_tpu_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(_DeviceViewTensor)


def test_import_simulation_strategy_branches_and_uniform_random():
    with pytest.raises(ValueError, match="not a known routing strategy"):
        _load_moe_routing(
            sim_strategy="nonexistent_strategy",
            available_strategies=["uniform_random"],
        )

    class _CustomStrategy:
        @staticmethod
        def route_tokens(hidden_states, router_logits, top_k, indices_type):
            return (
                torch.ones((router_logits.shape[0], top_k), dtype=torch.float32),
                torch.zeros((router_logits.shape[0], top_k), dtype=torch.int64),
            )

    mod_custom = _load_moe_routing(
        sim_strategy="custom_sim",
        available_strategies=["uniform_random", "custom_sim"],
        custom_strategy_obj=_CustomStrategy,
    )
    hs = torch.randn((4, 8), dtype=torch.bfloat16)
    logits = torch.randn((4, 6), dtype=torch.bfloat16)
    w_sim, idx_sim = mod_custom.maybe_simulate_routing(hs, logits, topk=2)
    assert w_sim.dtype == torch.bfloat16 and idx_sim.dtype == torch.int32

    mod_uni = _load_moe_routing(
        sim_strategy="uniform_random",
        available_strategies=["uniform_random"],
        custom_strategy_obj=_CustomStrategy,
    )
    w_uni, idx_uni = mod_uni.maybe_simulate_routing(hs, logits.float(), topk=2)
    assert w_uni.shape == (4, 2) and idx_uni.dtype == torch.int32

    layer = _FakeLayer(experts_per_token=2)
    rw, ridx = mod_uni.route(layer, hs, logits)
    assert rw.shape == (4, 2) and ridx.dtype == torch.int32
    sw, sidx = mod_uni.select_experts(
        hs, logits, topk=2, renormalize=True, scoring_fn="softmax", layer=layer
    )
    assert sw.shape == (4, 2) and sidx.dtype == torch.int32


def test_topk_scoring_pack_unpack_and_tpu_cache():
    mod = _load_moe_routing(sim_strategy=None, router_topk="rowmax")
    assert mod.maybe_simulate_routing(torch.randn(2, 4), torch.randn(2, 6), 2) is None

    scores = torch.tensor([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]], dtype=torch.float32)
    w0, i0 = mod._topk(scores, 2, bias=None, renormalize=False)
    assert w0.shape == (2, 2) and i0.shape == (2, 2)
    bias = torch.tensor([0.5, 0.0, 0.1], dtype=torch.bfloat16)
    w1, i1 = mod._topk(scores, 2, bias=bias, renormalize=True)
    assert torch.allclose(w1.sum(dim=-1), torch.ones(2))

    mod_torch = _load_moe_routing(sim_strategy=None, router_topk="torch")
    wt0, it0 = mod_torch._topk(scores, 2, bias=None, renormalize=False)
    assert wt0.shape == (2, 2) and it0.shape == (2, 2)
    wt1, it1 = mod_torch._topk(scores, 2, bias=bias, renormalize=True)
    assert torch.allclose(wt1.sum(dim=-1), torch.ones(2))

    logits_bf16 = torch.randn((3, 5), dtype=torch.bfloat16)
    for fn_name in ("softmax", "sigmoid", "sqrtsoftplus"):
        cpu_out = mod._apply_scoring_fn(fn_name, logits_bf16)
        assert cpu_out.shape == (3, 5)
    assert mod._apply_scoring_fn("softmax", logits_bf16.float()).shape == (3, 5)

    with pytest.raises(NotImplementedError, match="does not support"):
        mod._apply_scoring_fn("invalid_fn", logits_bf16)

    # _gather_partitioned_topk on CPU 16-bit, simulated TPU 16-bit, and float32 fallback
    class _CpuGroup(_FakeGroup):
        def all_gather(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
            return torch.cat([t for _ in range(self.world_size)], dim=dim)

    cpu_grp = _CpuGroup(world_size=2, rank_in_group=1)
    tpu_grp = _FakeGroup(world_size=2, rank_in_group=1)
    tw_bf16 = torch.tensor([[0.6, 0.4], [0.8, 0.2]], dtype=torch.bfloat16)
    ti_i32 = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
    gw_cpu, gi_cpu = mod._gather_partitioned_topk(cpu_grp, tw_bf16, ti_i32, topk=2)
    assert gw_cpu.shape == (4, 2) and gi_cpu.shape == (4, 2)

    with patch.object(torch, "compile", side_effect=lambda fn, **kw: fn):
        gw_tpu, gi_tpu = mod._gather_partitioned_topk(
            tpu_grp,
            _as_tpu_tensor(tw_bf16.transpose(0, 1).transpose(0, 1)),
            ti_i32,
            topk=2,
        )
        assert gw_tpu.shape == (4, 2) and gi_tpu.shape == (4, 2)
        # Warm cache hit on _get_tpu_op
        gw_tpu2, gi_tpu2 = mod._gather_partitioned_topk(
            tpu_grp, _as_tpu_tensor(tw_bf16), ti_i32, topk=2
        )
        assert gw_tpu2.shape == (4, 2) and gi_tpu2.shape == (4, 2)

    gw_f32, gi_f32 = mod._gather_partitioned_topk(
        cpu_grp, tw_bf16.float(), ti_i32, topk=2
    )
    assert gw_f32.shape == (4, 2) and gi_f32.shape == (4, 2)


def test_route_partitioned_select_experts_and_ep_helpers():
    mod = _load_moe_routing(sim_strategy=None, router_topk="torch")

    # _resolve_partition_group branches: sequence_parallel, use_ep, default dp
    ep_grp = _FakeGroup(world_size=2, rank_in_group=1)
    dp_grp = _FakeGroup(world_size=4, rank_in_group=2)
    with (
        patch.object(mod._ps, "get_ep_group", return_value=ep_grp),
        patch.object(mod._ps, "get_dp_group", return_value=dp_grp),
    ):
        l_sp = _FakeLayer(is_sequence_parallel=True)
        assert mod._resolve_partition_group(l_sp.moe_config, 2, 4) is ep_grp

        l_ep = _FakeLayer(use_ep=True, dp_size=1)
        assert mod._resolve_partition_group(l_ep.moe_config, 2, 4) is ep_grp

        l_ep_dp = _FakeLayer(use_ep=True, dp_size=4)
        assert mod._resolve_partition_group(l_ep_dp.moe_config, 2, 8) is dp_grp

        l_dp = _FakeLayer(use_ep=False, dp_size=2)
        assert mod._resolve_partition_group(l_dp.moe_config, 2, 4) is dp_grp

        # Custom routing function path in route()
        custom_layer = _FakeLayer(
            custom_routing_function=lambda hidden_states,
            gating_output,
            topk,
            renormalize: (
                torch.ones((hidden_states.shape[0], topk), dtype=hidden_states.dtype),
                torch.zeros((hidden_states.shape[0], topk), dtype=torch.int32),
            )
        )
        cw, ci = mod.route(
            custom_layer,
            torch.randn((4, 8), dtype=torch.bfloat16),
            torch.randn((4, 6)),
        )
        assert cw.shape == (4, 2) and ci.shape == (4, 2)

        # Partitioned route() with CPU tensors (sharded input_ids, non-divisible, None)
        part_layer = _FakeLayer(use_ep=True, dp_size=1, scoring_func="sqrtsoftplus")
        hs_full = torch.randn((4, 8), dtype=torch.bfloat16)
        logits_local = torch.randn((2, 6), dtype=torch.float32)
        pw, pi = mod.route(
            part_layer,
            hs_full,
            logits_local,
            input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        )
        assert pw.shape == (4, 2) and pi.shape == (4, 2)

        pw2, pi2 = mod.route(
            part_layer,
            hs_full,
            logits_local,
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        )
        assert pw2.shape == (4, 2) and pi2.shape == (4, 2)

        pw3, pi3 = mod.route(part_layer, hs_full, logits_local, input_ids=None)
        assert pw3.shape == (4, 2) and pi3.shape == (4, 2)

        # Partitioned route() on TPU device (fused select_and_pack + fallbacks)
        with patch.object(torch, "compile", side_effect=lambda fn, **kw: fn):
            tpu_logits_local = _as_tpu_tensor(logits_local)
            pw_tpu, pi_tpu = mod.route(part_layer, hs_full, tpu_logits_local)
            assert pw_tpu.shape == (4, 2) and pi_tpu.shape == (4, 2)

            hash_part_layer = _FakeLayer(
                use_ep=True,
                dp_size=1,
                hash_indices_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            )
            assert not mod._can_use_fused_classic_select(
                hash_part_layer,
                tpu_logits_local,
                torch.tensor([0, 1], dtype=torch.int32),
            )
            grp_part_layer = _FakeLayer(
                use_ep=True,
                dp_size=1,
                use_grouped_topk=True,
                num_expert_group=4,
                topk_group=2,
            )
            assert not mod._can_use_fused_classic_select(
                grp_part_layer, tpu_logits_local, None
            )

        # Non-partitioned route()
        fw, fi = mod.route(
            part_layer, hs_full, torch.randn((4, 6), dtype=torch.float32)
        )
        assert fw.shape == (4, 2) and fi.shape == (4, 2)

    # Hash MoE select (2D + 3D + non-int32 input_ids + renormalize + scaling)
    hash_table = torch.tensor([[0, 1], [2, 3], [1, 4], [0, 5]], dtype=torch.int64)
    hash_layer = _FakeLayer(
        hash_indices_table=hash_table,
        renormalize=True,
        routed_scaling_factor=1.5,
        scoring_func="sigmoid",
    )
    hw, hi = mod.select_experts(
        torch.randn((1, 2, 8), dtype=torch.bfloat16),
        torch.randn((1, 2, 6), dtype=torch.float32),
        topk=2,
        renormalize=True,
        scoring_fn="sigmoid",
        layer=hash_layer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
    )
    assert hw.shape == (1, 2, 2) and hi.dtype == torch.int32

    hash_layer_2d = _FakeLayer(
        hash_indices_table=hash_table.to(torch.int32),
        renormalize=False,
        routed_scaling_factor=1.0,
        scoring_func="softmax",
    )
    hw2, hi2 = mod.select_experts(
        torch.randn((2, 8), dtype=torch.float32),
        torch.randn((2, 6), dtype=torch.float32),
        topk=2,
        renormalize=False,
        scoring_fn="softmax",
        layer=hash_layer_2d,
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
    )
    assert hw2.shape == (2, 2) and hi2.dtype == torch.int32

    # Grouped top-k routing branch
    with patch.object(
        mod._grouped_topk_router,
        "grouped_topk",
        return_value=(
            torch.ones((3, 2), dtype=torch.float32),
            torch.ones((3, 2), dtype=torch.int64),
        ),
    ):
        grp_layer = _FakeLayer(
            use_grouped_topk=True,
            num_expert_group=4,
            topk_group=2,
        )
        gw, gi = mod.select_experts(
            torch.randn((3, 8), dtype=torch.bfloat16),
            torch.randn((3, 8), dtype=torch.float32),
            topk=2,
            renormalize=True,
            scoring_fn="softmax",
            layer=grp_layer,
        )
        assert gw.dtype == torch.bfloat16 and gi.dtype == torch.int32

    # Classic select_experts on TPU device with fused classic_select op
    with patch.object(torch, "compile", side_effect=lambda fn, **kw: fn):
        scale_layer = _FakeLayer(routed_scaling_factor=2.5)
        sw, si = mod.select_experts(
            _as_tpu_tensor(torch.randn((3, 8), dtype=torch.bfloat16)),
            _as_tpu_tensor(torch.randn((3, 6), dtype=torch.float32)),
            topk=2,
            renormalize=False,
            scoring_fn="softmax",
            layer=scale_layer,
        )
        assert sw.shape == (3, 2) and si.dtype == torch.int32

    # Classic select_experts with same-dtype float32 output
    f32_w, f32_i = mod.select_experts(
        torch.randn((3, 8), dtype=torch.float32),
        torch.randn((3, 6), dtype=torch.float32),
        topk=2,
        renormalize=False,
        scoring_fn="softmax",
        layer=None,
    )
    assert f32_w.dtype == torch.float32 and f32_i.dtype == torch.int32

    # get_experts_start, register_experts_start_buffer, validate_linear_ep_placement
    non_ep = _FakeLayer(use_ep=False)
    assert mod.get_experts_start(non_ep) is None
    mod.register_experts_start_buffer(non_ep, device=torch.device("cpu"))
    assert non_ep._buffers["_experts_start"] is None

    ep_l = _FakeLayer(use_ep=True, ep_size=2, ep_rank=1, global_num_experts=5)
    assert mod.get_experts_start(ep_l) == 3
    mod.register_experts_start_buffer(ep_l, device=torch.device("cpu"))
    assert int(ep_l._buffers["_experts_start"].item()) == 3

    mod.validate_linear_ep_placement(ep_l)
    bad_ep = _FakeLayer(expert_placement_strategy="round_robin")
    with pytest.raises(NotImplementedError, match="linear EP placement"):
        mod.validate_linear_ep_placement(bad_ep)
