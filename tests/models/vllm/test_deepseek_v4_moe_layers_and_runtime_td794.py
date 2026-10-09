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
"""Full CPU unit tests for TD7.94 target modules (100% statement & branch coverage)."""

from __future__ import annotations

import datetime
import importlib
import os
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu import envs, profiler_trace
from vllm_torchtpu.compilation import kernel_reload
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.custom_ops import moe_runner as moe_runner_mod
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.models.vllm.deepseek_v4 import layers as dsv4_layers
from vllm_torchtpu.models.vllm.deepseek_v4 import moe as dsv4_moe


def test_deepseek_v4_layers_full_coverage(monkeypatch: pytest.MonkeyPatch):
    # 1. mhc_collapse_head: float32 fast path and bfloat16 cast path
    hs_f32 = torch.randn((2, 4, 8), dtype=torch.float32)
    hc_fn = torch.randn((4, 32), dtype=torch.float32)
    hc_scale = torch.ones((4,), dtype=torch.float32)
    hc_base = torch.zeros((4,), dtype=torch.float32)

    out_f32 = dsv4_layers.mhc_collapse_head(
        hs_f32, hc_fn, hc_scale, hc_base, 1e-6, 1e-4
    )
    assert out_f32.shape == (2, 8) and out_f32.dtype == torch.float32

    hs_bf16 = hs_f32.to(torch.bfloat16)
    out_bf16 = dsv4_layers.mhc_collapse_head(
        hs_bf16, hc_fn, hc_scale, hc_base, 1e-6, 1e-4
    )
    assert out_bf16.shape == (2, 8) and out_bf16.dtype == torch.bfloat16

    # 2. DeepseekV4MLP: unsupported activation, swiglu_limit None vs float, forward
    class _FakeMergedCol(nn.Module):
        def __init__(self, in_dim, out_dims, **kw):
            super().__init__()
            self.out_dim = sum(out_dims)

        def forward(self, x):
            return torch.ones((*x.shape[:-1], self.out_dim), dtype=x.dtype), None

    class _FakeRow(nn.Module):
        def __init__(self, in_dim, out_dim, **kw):
            super().__init__()
            self.out_dim = out_dim

        def forward(self, x):
            return torch.ones((*x.shape[:-1], self.out_dim), dtype=x.dtype), None

    monkeypatch.setattr(dsv4_layers, "MergedColumnParallelLinear", _FakeMergedCol)
    monkeypatch.setattr(dsv4_layers, "RowParallelLinear", _FakeRow)
    monkeypatch.setattr(
        dsv4_layers, "SiluAndMul", lambda: (lambda t: t[..., : t.shape[-1] // 2])
    )
    monkeypatch.setattr(
        dsv4_layers,
        "SiluAndMulWithClamp",
        lambda limit: (lambda t: t[..., : t.shape[-1] // 2] * limit),
    )

    with pytest.raises(ValueError, match="Unsupported activation: gelu"):
        dsv4_layers.DeepseekV4MLP(
            hidden_size=8, intermediate_size=16, hidden_act="gelu"
        )

    mlp_no_clamp = dsv4_layers.DeepseekV4MLP(
        hidden_size=8, intermediate_size=16, hidden_act="silu", swiglu_limit=None
    )
    assert mlp_no_clamp(torch.ones((3, 8))).shape == (3, 8)

    mlp_clamp = dsv4_layers.DeepseekV4MLP(
        hidden_size=8, intermediate_size=16, hidden_act="silu", swiglu_limit=7.0
    )
    assert mlp_clamp(torch.ones((3, 8))).shape == (3, 8)


def test_deepseek_v4_moe_full_coverage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dsv4_moe, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(dsv4_moe, "get_tensor_model_parallel_rank", lambda: 0)

    class _FakeGate(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.e_score_correction_bias = None
            self.tid2eid = None

    class _FakeExperts(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.routed_experts = SimpleNamespace()
            self.do_naive_dispatch_combine = True
            self.flatten_out = False

        def forward(self, hidden_states, router_logits, input_ids=None):
            if self.flatten_out:
                return hidden_states.reshape(-1, hidden_states.shape[-1])
            return hidden_states

    monkeypatch.setattr(dsv4_moe, "GateLinear", _FakeGate)
    monkeypatch.setattr(dsv4_moe, "FusedMoEFactory", _FakeExperts)
    monkeypatch.setattr(dsv4_moe, "DeepseekV4MLP", lambda **kw: nn.Identity())

    # 1. Hash MoE layer (layer 0 < num_hash_layers=1) with shared_experts=None
    hf_hash = SimpleNamespace(
        routed_scaling_factor=1.5,
        hidden_size=8,
        n_routed_experts=4,
        swiglu_limit=None,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        num_hash_layers=1,
        vocab_size=16,
        num_experts_per_tok=2,
        topk_method=None,
        n_shared_experts=None,
        moe_intermediate_size=16,
        hidden_act="silu",
    )
    vcfg_hash = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_hash), quant_config=None
    )
    moe_hash = dsv4_moe.DeepseekV4MoE(vcfg_hash, prefix="model.layers.0.mlp")
    assert moe_hash._is_hash_moe
    assert hasattr(moe_hash.experts.routed_experts, "hash_indices_table")
    assert moe_hash.shared_experts is None
    assert moe_hash.routes_dp_gathered_tokens is True

    with pytest.raises(ValueError, match="requires input_ids"):
        moe_hash(torch.ones((2, 8)), input_ids=None)

    # Shape already matches org_shape -> no view
    x2d = torch.ones((2, 8))
    out2d = moe_hash(x2d, input_ids=torch.tensor([1, 2]))
    assert out2d is x2d

    # Shape differs from org_shape -> .view(org_shape)
    moe_hash.experts.flatten_out = True
    x3d = torch.ones((2, 1, 8))
    out3d = moe_hash(x3d, input_ids=torch.tensor([1, 2]))
    assert out3d.shape == (2, 1, 8)

    # 2. Non-hash MoE layer with topk_method="noaux_tc" and n_shared_experts=1
    hf_noaux = SimpleNamespace(
        routed_scaling_factor=1.0,
        hidden_size=8,
        n_routed_experts=4,
        swiglu_limit=7.0,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        num_hash_layers=1,
        vocab_size=16,
        num_experts_per_tok=2,
        topk_method="noaux_tc",
        n_shared_experts=1,
        moe_intermediate_size=16,
        hidden_act="silu",
    )
    vcfg_noaux = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_noaux), quant_config=None
    )
    moe_noaux = dsv4_moe.DeepseekV4MoE(vcfg_noaux, prefix="model.layers.2.mlp")
    assert not moe_noaux._is_hash_moe
    assert moe_noaux.gate.e_score_correction_bias is not None
    assert moe_noaux.shared_experts is not None
    assert moe_noaux(x2d, input_ids=None).shape == (2, 8)

    # 3. Non-hash MoE layer with topk_method != "noaux_tc"
    hf_std = SimpleNamespace(**{**hf_noaux.__dict__, "topk_method": "greedy"})
    vcfg_std = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_std), quant_config=None
    )
    moe_std = dsv4_moe.DeepseekV4MoE(vcfg_std, prefix="model.layers.3.mlp")
    assert moe_std.gate.e_score_correction_bias is None


def test_vllm_moe_runner_full_coverage(monkeypatch: pytest.MonkeyPatch):
    # 1. __init__ calling super().__init__ and _take_over_moe_quant_config_init
    def _fake_super_init(self, *a, **kw):
        nn.Module.__init__(self)
        self.routed_experts = SimpleNamespace(
            quant_method=SimpleNamespace(is_monolithic=False),
            _ensure_moe_quant_config_init=lambda: setattr(self, "_init_called", True),
            custom_routing_function=None,
        )
        self.moe_config = SimpleNamespace(
            skip_final_all_reduce=False,
            pcp_size=1,
            is_sequence_parallel=False,
        )
        self._shared_experts = None

    monkeypatch.setattr(MoERunner, "__init__", _fake_super_init)
    monkeypatch.setattr(MoERunner, "do_naive_dispatch_combine", True)
    runner = VllmMoERunner()

    # _take_over_moe_quant_config_init: non-monolithic vs monolithic
    runner.routed_experts._ensure_moe_quant_config_init()
    assert getattr(runner, "_init_called", False) is True

    runner._init_called = False
    runner.routed_experts.quant_method.is_monolithic = True
    runner.routed_experts._ensure_moe_quant_config_init()
    assert runner._init_called is False

    # _select_forward: shared_experts None vs not None
    assert runner._select_forward() is moe_runner_mod._moe_runner._moe_forward
    runner._shared_experts = SimpleNamespace()
    assert runner._select_forward() is moe_runner_mod._moe_runner._moe_forward_shared

    # _fused_output_is_reduced:
    monkeypatch.setattr(
        moe_runner_mod, "_upstream_fused_output_is_reduced", lambda s: False
    )
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    assert runner._fused_output_is_reduced is False

    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    runner.moe_config.skip_final_all_reduce = True
    assert runner._fused_output_is_reduced is False

    runner.moe_config.skip_final_all_reduce = False
    monkeypatch.setattr(moe_runner_mod, "fused_moe_ep_supported", lambda o: False)
    monkeypatch.setattr(moe_runner_mod, "adaptive_fused_moe_supported", lambda o: False)
    assert runner._fused_output_is_reduced is False

    monkeypatch.setattr(moe_runner_mod, "adaptive_fused_moe_supported", lambda o: True)
    assert runner._fused_output_is_reduced is True

    monkeypatch.setattr(moe_runner_mod, "fused_moe_ep_supported", lambda o: True)
    assert runner._fused_output_is_reduced is True

    # _fused_ep_owns_pcp_collectives: pcp_size <= 1 vs > 1
    runner.moe_config.pcp_size = 1
    assert runner._fused_ep_owns_pcp_collectives() is False
    runner.moe_config.pcp_size = 2
    assert runner._fused_ep_owns_pcp_collectives() is True

    # _maybe_dispatch & _maybe_combine when _fused_ep_owns_pcp_collectives() is True
    h = torch.ones((2, 4))
    r = torch.ones((2, 4))
    assert runner._maybe_dispatch(h, r) == (h, r)
    runner._shared_experts = SimpleNamespace()
    assert runner._maybe_combine("shared", h) == ("shared", h)
    runner._shared_experts = None
    assert runner._maybe_combine(None, h) is h

    # _maybe_dispatch & _maybe_combine when _fused_ep_owns_pcp_collectives() is False
    runner.moe_config.pcp_size = 1
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
    monkeypatch.setattr(
        moe_runner_mod,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=2, all_gather=lambda t, dim=0: t * 3.0),
    )
    monkeypatch.setattr(
        moe_runner_mod,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=1, all_gather=lambda t, dim=0: t * 5.0),
    )
    # (a) naive monolithic dispatch with is_sequence_parallel=False (world_size=2)
    h_dp, r_dp = runner._maybe_dispatch(h, r)
    assert torch.all(h_dp == 3.0) and r_dp is r

    # (b) naive monolithic dispatch with is_sequence_parallel=True (world_size=1)
    runner.moe_config.is_sequence_parallel = True
    h_ep, r_ep = runner._maybe_dispatch(h, r)
    assert h_ep is h and r_ep is r

    # (c) fallback to super()._maybe_dispatch and super()._maybe_combine
    monkeypatch.setattr(
        MoERunner, "_maybe_dispatch", lambda self, hs, rl: ("super_disp", hs, rl)
    )
    monkeypatch.setattr(
        MoERunner, "_maybe_combine", lambda self, so, hs: ("super_comb", so, hs)
    )
    assert runner._maybe_dispatch(h, h)[0] == "super_disp"
    assert runner._maybe_combine("so", h)[0] == "super_comb"


def test_profiler_trace_full_coverage(tmp_path, monkeypatch: pytest.MonkeyPatch):
    def _stage(capture_dir, ts_name, filename, content):
        ts_dir = capture_dir / "plugins" / "profile" / ts_name
        ts_dir.mkdir(parents=True, exist_ok=True)
        (ts_dir / filename).write_text(content)

    # 1. profile_session_id (with and without TORCH_TPU_DP_MASTER_PORT)
    monkeypatch.setenv("TORCH_TPU_DP_MASTER_PORT", "12345")
    assert profiler_trace.profile_session_id() == "dp12345"
    monkeypatch.delenv("TORCH_TPU_DP_MASTER_PORT", raising=False)
    assert profiler_trace.profile_session_id() == str(os.getppid())

    # 2. rank_capture_dir & canonical_ts_marker_path
    assert profiler_trace.rank_capture_dir(str(tmp_path), 2).endswith("rank_2")

    # 3. resolve_canonical_dst_ts & clear_canonical_ts_marker:
    # (a) default session_key=None on rank 0, read by rank 1, then cleared
    ts = profiler_trace.resolve_canonical_dst_ts(str(tmp_path), 0)
    assert profiler_trace.resolve_canonical_dst_ts(str(tmp_path), 1) == ts
    profiler_trace.clear_canonical_ts_marker(str(tmp_path))

    # (b) explicit session_key on rank 0 with write error
    missing_parent = tmp_path / "no_such_dir" / "sub"
    ts_err = profiler_trace.resolve_canonical_dst_ts(
        str(missing_parent), 0, session_key="k_err"
    )
    datetime.datetime.strptime(ts_err, profiler_trace.CANONICAL_TS_FORMAT)

    # (c) rank != 0 sees empty/missing marker and times out
    monkeypatch.setattr(profiler_trace, "CANONICAL_TS_POLL_TIMEOUT_S", 0.04)
    monkeypatch.setattr(profiler_trace, "CANONICAL_TS_POLL_INTERVAL_S", 0.01)
    ts_miss = profiler_trace.resolve_canonical_dst_ts(
        str(tmp_path), 1, session_key="missing_key"
    )
    datetime.datetime.strptime(ts_miss, profiler_trace.CANONICAL_TS_FORMAT)

    (tmp_path / ".canonical_ts_empty_key").write_text("  \n")
    ts_empty = profiler_trace.resolve_canonical_dst_ts(
        str(tmp_path), 1, session_key="empty_key"
    )
    datetime.datetime.strptime(ts_empty, profiler_trace.CANONICAL_TS_FORMAT)
    profiler_trace.clear_canonical_ts_marker(str(tmp_path), session_key="empty_key")

    # 4. merge_rank_capture:
    # (a) non-existent source_profile_path -> early return
    empty_cap = tmp_path / "rank_empty"
    empty_cap.mkdir()
    profiler_trace.merge_rank_capture(str(empty_cap), str(tmp_path), "ts0", 0)

    # (b) single worker with stray file under plugins/profile + valid ts dir
    cap0 = tmp_path / "rank_0"
    _stage(cap0, "pt_0", "trace.xplane.pb", "payload0")
    (cap0 / "plugins" / "profile" / "not_a_dir.txt").write_text("skip")
    profiler_trace.merge_rank_capture(
        str(cap0), str(tmp_path), "ts_single", 0, world_size=1
    )
    assert (
        tmp_path / "plugins" / "profile" / "ts_single" / "trace.xplane.pb"
    ).read_text() == "payload0"

    # (c) multi-worker (world_size > 1) prefixes filenames
    cap1 = tmp_path / "rank_1"
    _stage(cap1, "pt_1", "trace.xplane.pb", "payload1")
    profiler_trace.merge_rank_capture(
        str(cap1), str(tmp_path), "ts_multi", 1, world_size=2
    )
    assert (
        tmp_path / "plugins" / "profile" / "ts_multi" / "rank1_trace.xplane.pb"
    ).read_text() == "payload1"

    # (d) exception in merge_rank_capture is caught and logged
    cap_err = tmp_path / "rank_err"
    _stage(cap_err, "pt_err", "trace.xplane.pb", "payload")
    monkeypatch.setattr(
        profiler_trace.os,
        "makedirs",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    profiler_trace.merge_rank_capture(str(cap_err), str(tmp_path), "ts_fail", 0)


def test_kernel_reload_full_coverage(tmp_path, monkeypatch: pytest.MonkeyPatch):
    saved_live = dict(kernel_reload._LIVE)
    saved_builders = dict(kernel_reload._BUILDERS)
    kernel_reload._LIVE.clear()
    kernel_reload._BUILDERS.clear()
    try:
        # 1. is_enabled & default_reload_modules (default vs env override)
        monkeypatch.setattr(envs, "TPU_KERNEL_ITER_MODE", True)
        assert kernel_reload.is_enabled() is True

        monkeypatch.setattr(envs, "TPU_KERNEL_RELOAD_MODULES", " mod_a , , mod_b ")
        assert kernel_reload.default_reload_modules() == ("mod_a", "mod_b")

        monkeypatch.setattr(envs, "TPU_KERNEL_RELOAD_MODULES", "")
        assert (
            kernel_reload.default_reload_modules()
            == kernel_reload._DEFAULT_RELOAD_MODULES
        )

        # 2. make_dispatcher, set_live, get_live, _KernelDispatch.__call__
        def _tmpl(a: int, b: int = 1, *, c: int = 2) -> int:
            return a + b + c

        disp = kernel_reload.make_dispatcher("op::test", _tmpl)
        assert kernel_reload.get_live("op::test") is None
        kernel_reload.set_live("op::test", _tmpl)
        assert kernel_reload.get_live("op::test") is _tmpl
        assert disp(3) == 6

        # 3. reload_kernels with modules=None (imported & unimported modules)
        mod_path = tmp_path / "temp_reload_mod_td794.py"
        mod_path.write_text(
            textwrap.dedent("""
            def fn(x):
                return x + 10
            """)
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.import_module("temp_reload_mod_td794")
        try:
            kernel_reload.register_builder(
                "op::test",
                lambda: importlib.import_module("temp_reload_mod_td794").fn,
            )
            mod_path.write_text(
                textwrap.dedent("""
                def fn(x):
                    return x + 200
                """)
            )
            monkeypatch.setattr(
                envs,
                "TPU_KERNEL_RELOAD_MODULES",
                "temp_reload_mod_td794,unimported_mod_td794",
            )
            stats = kernel_reload.reload_kernels(modules=None)
            assert stats["rebuilt_ops"] == ["op::test"]
            assert disp(5) == 205
        finally:
            sys.modules.pop("temp_reload_mod_td794", None)
    finally:
        kernel_reload._LIVE.clear()
        kernel_reload._LIVE.update(saved_live)
        kernel_reload._BUILDERS.clear()
        kernel_reload._BUILDERS.update(saved_builders)
