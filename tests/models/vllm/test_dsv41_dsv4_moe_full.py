# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for DeepSeek-V4 moe.py."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

for _pkg_name, _rel in (
    ("vllm_torchtpu", "vllm_torchtpu"),
    ("vllm_torchtpu.models", "vllm_torchtpu/models"),
    ("vllm_torchtpu.models.vllm", "vllm_torchtpu/models/vllm"),
    (
        "vllm_torchtpu.models.vllm.deepseek_v4",
        "vllm_torchtpu/models/vllm/deepseek_v4",
    ),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_SRC_DIR / _rel)] if _SRC_DIR.is_dir() else []  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg
    if "." in _pkg_name:
        _p, _c = _pkg_name.rsplit(".", 1)
        setattr(sys.modules[_p], _c, sys.modules[_pkg_name])


def _extract_layer_index(prefix: str) -> int:
    for part in prefix.split("."):
        if part.isdigit():
            return int(part)
    return 0


for _vmod_name, _vattrs in (
    ("vllm", {}),
    ("vllm.config", {"VllmConfig": object}),
    (
        "vllm.distributed",
        {
            "get_tensor_model_parallel_rank": lambda: 0,
            "get_tensor_model_parallel_world_size": lambda: 1,
        },
    ),
    ("vllm.model_executor", {}),
    ("vllm.model_executor.layers", {}),
    ("vllm.model_executor.layers.fused_moe", {"FusedMoEFactory": object}),
    ("vllm.model_executor.layers.fused_moe.router", {}),
    (
        "vllm.model_executor.layers.fused_moe.router.gate_linear",
        {"GateLinear": object},
    ),
    ("vllm.model_executor.models", {}),
    (
        "vllm.model_executor.models.utils",
        {"extract_layer_index": _extract_layer_index},
    ),
    (
        "vllm_torchtpu.models.vllm.deepseek_v4.layers",
        {"DeepseekV4MLP": lambda **kw: nn.Identity()},
    ),
):
    if _vmod_name not in sys.modules:
        _m = types.ModuleType(_vmod_name)
        _m.__path__ = []  # type: ignore[attr-defined]
        for _k, _v in _vattrs.items():
            setattr(_m, _k, _v)
        sys.modules[_vmod_name] = _m
    if "." in _vmod_name:
        _p, _c = _vmod_name.rsplit(".", 1)
        setattr(sys.modules[_p], _c, sys.modules[_vmod_name])

dsv4_moe = importlib.import_module("vllm_torchtpu.models.vllm.deepseek_v4.moe")


def test_deepseek_v4_moe_all_branches(monkeypatch: pytest.MonkeyPatch):
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
            self.mode = "identity"

        def forward(self, hidden_states, router_logits, input_ids=None):
            if self.mode == "flatten":
                return hidden_states.reshape(-1, hidden_states.shape[-1])
            if self.mode == "clone":
                return hidden_states.clone()
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
    assert moe_hash._is_hash_moe is True
    assert hasattr(moe_hash.experts.routed_experts, "hash_indices_table")
    assert moe_hash.shared_experts is None
    assert moe_hash.routes_dp_gathered_tokens is True

    with pytest.raises(ValueError, match="requires input_ids"):
        moe_hash(torch.ones((2, 8)), input_ids=None)

    # Identity return fast path
    x2d = torch.ones((2, 8))
    ids = torch.tensor([1, 2], dtype=torch.int32)
    out2d = moe_hash(x2d, input_ids=ids)
    assert out2d is x2d

    # Cloned return with matching shape
    moe_hash.experts.mode = "clone"
    out_clone = moe_hash(x2d, input_ids=ids)
    assert out_clone is not x2d and torch.equal(out_clone, x2d)

    # Flattened return requiring .view(org_shape)
    moe_hash.experts.mode = "flatten"
    x3d = torch.ones((2, 1, 8))
    out3d = moe_hash(x3d, input_ids=ids)
    assert out3d.shape == (2, 1, 8)

    # Non-nn.Module callable assigned to experts (fallback to self.experts)
    custom_exp = SimpleNamespace(
        do_naive_dispatch_combine=False,
        __call__=lambda **kw: kw["hidden_states"],
    )
    moe_hash._modules.pop("experts", None)
    moe_hash.__dict__["experts"] = lambda **kw: kw["hidden_states"]
    assert moe_hash(x2d, input_ids=ids) is x2d
    moe_hash.__dict__["experts"] = custom_exp
    assert moe_hash.routes_dp_gathered_tokens is False

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
    assert moe_noaux._is_hash_moe is False
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
