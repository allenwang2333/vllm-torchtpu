# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Track TD7.39: unquantized.py and online_fp8.py 2D fast path,
direct MoE dispatch, and 100% full-module statement & branch coverage."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization import (
    online_fp8 as online_fp8_mod,
)
from vllm_torchtpu.layers.adapter.quantization import (
    unquantized as unquant_mod,
)
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.core.quant_methods import UNQUANTIZED


@pytest.fixture(autouse=True)
def _mock_vllm_config():
    from vllm.config.vllm import set_current_vllm_config

    dummy_vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(linear_backend="auto", moe_backend="auto"),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            static_all_moe_layers=[],
            custom_ops=["all"],
            enabled_custom_ops=set(),
            disabled_custom_ops=set(),
            max_cudagraph_capture_size=0,
            mode=0,
        ),
    )
    with set_current_vllm_config(dummy_vllm_config):  # type: ignore[arg-type]
        yield


def _make_moe_config(*, use_ep: bool = False, has_bias: bool = True) -> FusedMoEConfig:
    return SimpleNamespace(  # type: ignore[return-value]
        experts_per_token=2,
        has_bias=has_bias,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        ep_size=2 if use_ep else 1,
        ep_rank=0,
        is_sequence_parallel=False,
        intermediate_size_per_partition=128,
        in_dtype=torch.bfloat16,
    )


def _make_unquant_routed_experts(
    *,
    activation: str = "silu",
    use_ep: bool = False,
    has_bias: bool = True,
    num_experts: int = 2,
    intermediate_size: int = 128,
    hidden_size: int = 128,
) -> RoutedExperts:
    layer = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    layer.moe_config = _make_moe_config(use_ep=use_ep, has_bias=has_bias)
    layer.activation = activation
    layer.renormalize = True
    layer.global_num_experts = num_experts
    layer.expert_placement_strategy = "linear"
    layer.register_buffer(
        "_experts_start", torch.tensor(0, dtype=torch.int32), persistent=False
    )
    layer.w13_weight = nn.Parameter(
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
        ),
        requires_grad=False,
    )
    layer.w2_weight = nn.Parameter(
        torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16),
        requires_grad=False,
    )
    if has_bias:
        layer.w13_bias = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.w2_bias = nn.Parameter(
            torch.randn(num_experts, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
    else:
        layer.register_parameter("w13_bias", None)
        layer.register_parameter("w2_bias", None)
    return layer


class TestModuleReloadCoverage:
    def test_reload_modules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import vllm.model_executor.layers.quantization as vllm_quant

        orig_unquant = dict(unquant_mod.__dict__)
        orig_online = dict(online_fp8_mod.__dict__)
        monkeypatch.setattr(
            vllm_quant,
            "register_quantization_config",
            lambda _name: (lambda cls: cls),
        )
        try:
            importlib.reload(unquant_mod)
            importlib.reload(online_fp8_mod)
        finally:
            unquant_mod.__dict__.clear()
            unquant_mod.__dict__.update(orig_unquant)
            online_fp8_mod.__dict__.clear()
            online_fp8_mod.__dict__.update(orig_online)


class TestVllmUnquantizedLinearAndConfig:
    def test_linear_process_weights_and_apply_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        method = unquant_mod.VllmUnquantizedLinearMethod()

        # 1. KEEP_VLLM_LAYOUT_ATTR=True -> returns early and delegates apply to super()
        layer_keep = nn.Module()
        w_raw = torch.randn(16, 32, dtype=torch.bfloat16)
        layer_keep.weight = nn.Parameter(w_raw.clone(), requires_grad=False)
        setattr(layer_keep, KEEP_VLLM_LAYOUT_ATTR, True)
        method.process_weights_after_loading(layer_keep)
        assert layer_keep.weight.shape == (16, 32)
        monkeypatch.setattr(
            unquant_mod.UnquantizedLinearMethod,
            "apply",
            lambda self, lyr, x, bias=None: F.linear(x, lyr.weight, bias),
        )
        x_2d = torch.randn(4, 32, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
        assert torch.equal(
            method.apply(layer_keep, x_2d, bias), F.linear(x_2d, w_raw, bias)
        )

        # 2. Normal 2D weight -> transposed, idempotent on second call
        layer_2d = nn.Module()
        layer_2d.weight = nn.Parameter(w_raw.clone(), requires_grad=False)
        method.process_weights_after_loading(layer_2d)
        assert layer_2d.weight.shape == (32, 16)
        assert getattr(layer_2d, WEIGHT_FLIPPED_ATTR) is True
        # Re-entering returns early
        method.process_weights_after_loading(layer_2d)
        assert layer_2d.weight.shape == (32, 16)

        # Apply without bias, with 2D bias (fast path), and with 3D bias (reshape path)
        assert torch.equal(
            method.apply(layer_2d, x_2d, None), F.linear(x_2d, w_raw, None)
        )
        assert torch.equal(
            method.apply(layer_2d, x_2d, bias), F.linear(x_2d, w_raw, bias)
        )
        x_3d = torch.randn(2, 4, 32, dtype=torch.bfloat16)
        assert torch.equal(
            method.apply(layer_2d, x_3d, bias), F.linear(x_3d, w_raw, bias)
        )

        # 3. Non-2D weight -> left alone
        layer_3d = nn.Module()
        layer_3d.weight = nn.Parameter(
            torch.randn(2, 16, 32, dtype=torch.bfloat16), requires_grad=False
        )
        method.process_weights_after_loading(layer_3d)
        assert layer_3d.weight.shape == (2, 16, 32)

        # 4. Simulated TPU device -> calls synchronize_tensors
        layer_tpu = nn.Module()
        layer_tpu.weight = nn.Parameter(w_raw.clone(), requires_grad=False)
        sync_calls: list[list[torch.Tensor]] = []
        monkeypatch.setattr(
            unquant_mod, "synchronize_tensors", lambda t: sync_calls.append(t)
        )
        orig_param = torch.nn.Parameter

        class _FakeTpuParam(torch.nn.Parameter):
            @property
            def device(self):  # type: ignore[override]
                return SimpleNamespace(type="tpu")

        monkeypatch.setattr(torch.nn, "Parameter", _FakeTpuParam)
        method.process_weights_after_loading(layer_tpu)
        assert len(sync_calls) == 1
        monkeypatch.setattr(torch.nn, "Parameter", orig_param)

    def test_unquantized_config_methods_and_get_quant_method(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cls = unquant_mod.VllmUnquantizedConfig
        assert cls.get_name() == UNQUANTIZED
        assert cls.get_supported_act_dtypes() == [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ]
        assert cls.get_min_capability() == 0
        assert cls.get_config_filenames() == []
        cfg = cls.from_config({})
        assert isinstance(cfg, cls)

        linear_layer = LinearBase.__new__(LinearBase)
        nn.Module.__init__(linear_layer)
        monkeypatch.setattr(cfg, "get_linear_config", lambda _l: SimpleNamespace())

        # 1. should_quantize_on_load=True and not kv_b_proj -> VllmFp8LinearMethodTPU
        monkeypatch.setattr(unquant_mod, "should_quantize_on_load", lambda _p: True)
        from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

        monkeypatch.setattr(
            fp8_mod,
            "VllmFp8LinearMethodTPU",
            lambda fp8_cfg, lin_cfg, prefix=None: ("fp8_lin", prefix),
        )
        res_fp8 = cfg.get_quant_method(linear_layer, "model.layers.0.q_proj")
        assert res_fp8 == ("fp8_lin", "model.layers.0.q_proj")

        # 2. should_quantize_on_load=True but kv_b_proj -> VllmUnquantizedLinearMethod
        res_kv_b = cfg.get_quant_method(linear_layer, "model.layers.0.kv_b_proj")
        assert isinstance(res_kv_b, unquant_mod.VllmUnquantizedLinearMethod)

        # 3. should_quantize_on_load=False -> VllmUnquantizedLinearMethod
        monkeypatch.setattr(unquant_mod, "should_quantize_on_load", lambda _p: False)
        res_unq = cfg.get_quant_method(linear_layer, "model.layers.0.q_proj")
        assert isinstance(res_unq, unquant_mod.VllmUnquantizedLinearMethod)

        # 4. RoutedExperts -> VllmUnquantizedFusedMoEMethod
        routed = _make_unquant_routed_experts()
        monkeypatch.setattr(cfg, "get_moe_config", lambda lyr: lyr.moe_config)
        res_moe = cfg.get_quant_method(routed, "model.layers.0.experts")
        assert isinstance(res_moe, unquant_mod.VllmUnquantizedFusedMoEMethod)

        # 5. Attention -> None
        attn = Attention.__new__(Attention)
        nn.Module.__init__(attn)
        assert cfg.get_quant_method(attn, "model.layers.0.attn") is None

        # 6. Other module -> None
        assert cfg.get_quant_method(nn.Identity(), "model.norm") is None


class TestVllmUnquantizedFusedMoEMethod:
    def test_properties_and_process_weights_all_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moe_cfg_bias = _make_moe_config(use_ep=True, has_bias=True)
        method_bias = unquant_mod.VllmUnquantizedFusedMoEMethod(moe_cfg_bias)
        assert method_bias.is_monolithic is True

        monkeypatch.setattr(
            unquant_mod, "enable_pipelined_collective_and_compute", lambda: True
        )
        assert method_bias.supports_internal_mk is True
        monkeypatch.setattr(
            unquant_mod, "enable_pipelined_collective_and_compute", lambda: False
        )
        assert method_bias.supports_internal_mk is False

        monkeypatch.setattr(unquant_mod, "prebuild_fused_moe_kernel", lambda **kw: None)
        monkeypatch.setattr(
            moe_routing, "validate_linear_ep_placement", lambda lyr: None
        )
        monkeypatch.setattr(
            moe_routing,
            "register_experts_start_buffer",
            lambda lyr, device=None: None,
        )

        # Case 1: swigluoai, unaligned intermediate_size=64 (aligned_half != half),
        # has_bias=True, use_ep=True, simulated TPU device
        layer1 = _make_unquant_routed_experts(
            activation="swigluoai",
            use_ep=True,
            has_bias=True,
            intermediate_size=64,
            hidden_size=128,
        )
        # Case 2: silu, aligned intermediate_size=128 (aligned_half == half),
        # has_bias=False, use_ep=False, simulated TPU device
        moe_cfg_nobias = _make_moe_config(use_ep=False, has_bias=False)
        method_nobias = unquant_mod.VllmUnquantizedFusedMoEMethod(moe_cfg_nobias)
        layer2 = _make_unquant_routed_experts(
            activation="silu",
            use_ep=False,
            has_bias=False,
            intermediate_size=128,
            hidden_size=128,
        )
        # Case 3: silu, aligned intermediate_size=128, has_bias=True, CPU device
        layer3 = _make_unquant_routed_experts(
            activation="silu",
            use_ep=False,
            has_bias=True,
            intermediate_size=128,
            hidden_size=128,
        )
        method_bias.process_weights_after_loading(layer3)
        assert layer3.w13_weight.shape == (2, 128, 256)

        sync_calls: list[list[torch.Tensor]] = []
        monkeypatch.setattr(
            unquant_mod, "synchronize_tensors", lambda t: sync_calls.append(t)
        )
        orig_param = torch.nn.Parameter

        class _FakeTpuParam(torch.nn.Parameter):
            @property
            def device(self):  # type: ignore[override]
                return SimpleNamespace(type="tpu")

        monkeypatch.setattr(torch.nn, "Parameter", _FakeTpuParam)
        method_bias.process_weights_after_loading(layer1)
        assert len(sync_calls) == 1 and len(sync_calls[-1]) == 4
        assert layer1.w13_weight.shape == (2, 128, 256)
        assert layer1.w2_weight.shape == (2, 128, 128)

        method_nobias.process_weights_after_loading(layer2)
        assert len(sync_calls) == 2 and len(sync_calls[-1]) == 2
        monkeypatch.setattr(torch.nn, "Parameter", orig_param)

    def test_forward_monolithic_tpu_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moe_cfg = _make_moe_config(use_ep=False, has_bias=True)
        method = unquant_mod.VllmUnquantizedFusedMoEMethod(moe_cfg)
        layer = _make_unquant_routed_experts()

        with pytest.raises(
            AssertionError, match="process_weights_after_loading did not run"
        ):
            method._forward_monolithic_tpu(
                layer, torch.randn(4, 128), torch.randn(4, 2)
            )

        method._tpu_activation_str = "silu"
        monkeypatch.setattr(
            moe_routing,
            "route",
            lambda lyr, x, rl: (
                torch.ones(x.shape[0], 2),
                torch.zeros(x.shape[0], 2, dtype=torch.int32),
            ),
        )
        monkeypatch.setattr(
            unquant_mod, "pipelined_fused_moe_gmm", lambda **kw: kw["hidden_states"] + 1
        )
        monkeypatch.setattr(
            unquant_mod, "fused_moe_gmm", lambda **kw: kw["hidden_states"] + 2
        )

        x = torch.randn(4, 128)
        rl = torch.randn(4, 2)

        monkeypatch.setattr(
            unquant_mod, "enable_pipelined_collective_and_compute", lambda: True
        )
        assert torch.equal(method._forward_monolithic_tpu(layer, x, rl), x + 1)

        monkeypatch.setattr(
            unquant_mod, "enable_pipelined_collective_and_compute", lambda: False
        )
        assert torch.equal(method._forward_monolithic_tpu(layer, x, rl), x + 2)


class TestOnlineFp8FullCoverage:
    def test_policy_from_dict_and_select_and_mapper(self) -> None:
        Policy = online_fp8_mod.OnlineFp8Policy
        assert Policy.from_dict(None) is None
        assert Policy.from_dict({"enabled": False}) is None

        for invalid in [
            [],
            {"enabled": "yes"},
            {"enabled": True, "bad_key": 1},
            {"enabled": True, "weight_scheme": "bad"},
            {"enabled": True, "weight_scheme": "per_block", "weight_block_size": [128]},
            {
                "enabled": True,
                "weight_scheme": "per_block",
                "weight_block_size": [128, 64],
            },
            {
                "enabled": True,
                "weight_scheme": "per_tensor",
                "weight_block_size": [128, 128],
            },
            {"enabled": True, "targets": "not_a_list"},
            {"enabled": True, "targets": ["re:bad"]},
            {"enabled": True, "targets": []},
        ]:
            with pytest.raises(ValueError):
                Policy.from_dict(invalid)  # type: ignore[arg-type]

        pol = Policy.from_dict(
            {
                "enabled": True,
                "weight_scheme": "per_block",
                "weight_block_size": [128, 128],
                "targets": ["proj", "model.qkv"],
                "exclude": ["model.skip"],
            }
        )
        assert pol is not None
        assert pol.weight_block_size == (128, 128)

        # select with top-level prefix (no dot) and parent.leaf prefix
        assert pol.select("proj", {}) == (True, ["proj"])
        with pytest.raises(ValueError, match="Partial fused-layer"):
            pol.select("model.qkvz", {"qkvz": ["qkv", "z"]})

        # map_online_fp8 with policy None vs not None
        fake_mapper = SimpleNamespace(
            apply_list=lambda items: [f"mapped.{i}" for i in items]
        )
        cfg_none = SimpleNamespace(online_fp8_policy=None)
        online_fp8_mod.map_online_fp8(cfg_none, fake_mapper)  # type: ignore[arg-type]

        cfg_pol = SimpleNamespace(online_fp8_policy=pol)
        online_fp8_mod.map_online_fp8(cfg_pol, fake_mapper)  # type: ignore[arg-type]
        assert pol.targets[0] == "mapped.proj"

    def test_attach_and_validate_and_quantize_online_fp8(self) -> None:
        Policy = online_fp8_mod.OnlineFp8Policy
        # attach_online_fp8 with policy=None
        vllm_cfg_off = SimpleNamespace(additional_config={})
        assert online_fp8_mod.attach_online_fp8(None, vllm_cfg_off) is None  # type: ignore[arg-type]

        # attach_online_fp8 with unsupported config
        vllm_cfg_on = SimpleNamespace(
            additional_config={"tpu_online_fp8": {"enabled": True, "targets": ["proj"]}}
        )
        with pytest.raises(ValueError, match="supports FP8 and ModelOpt NVFP4"):
            online_fp8_mod.attach_online_fp8(None, vllm_cfg_on)  # type: ignore[arg-type]

        class _FakeQuantCfg(VllmQuantConfig):
            def __init__(self, name: str = "fp8"):
                self._name = name
                self.packed_modules_mapping = {}
                self.online_fp8_policy = None

            def get_name(self) -> str:
                return self._name

        good_cfg = _FakeQuantCfg("fp8")
        online_fp8_mod.attach_online_fp8(good_cfg, vllm_cfg_on)  # type: ignore[arg-type]
        assert good_cfg.online_fp8_policy is not None

        # validate_online_fp8 early returns
        online_fp8_mod.validate_online_fp8(nn.Module(), None)
        cfg_no_pol = _FakeQuantCfg("fp8")
        online_fp8_mod.validate_online_fp8(nn.Module(), cfg_no_pol)  # type: ignore[arg-type]

        # quantize_online_fp8 across schemes and dtypes (bfloat16 AND float32)
        w_bf16 = torch.randn(128, 128, dtype=torch.bfloat16)
        w_f32 = w_bf16.float()

        pol_tensor = Policy(targets=["proj"], exclude=[], weight_scheme="per_tensor")
        q_t, s_t, b_t = online_fp8_mod.quantize_online_fp8(w_bf16, pol_tensor)
        assert (
            q_t.dtype == torch.float8_e4m3fn and s_t.shape == (1,) and b_t == (128, 128)
        )

        pol_chan = Policy(targets=["proj"], exclude=[], weight_scheme="per_channel")
        q_c, s_c, b_c = online_fp8_mod.quantize_online_fp8(w_f32, pol_chan)
        assert (
            q_c.dtype == torch.float8_e4m3fn and s_c.shape == (128,) and b_c == (1, 128)
        )

        pol_block = Policy(
            targets=["proj"],
            exclude=[],
            weight_scheme="per_block",
            weight_block_size=(128, 128),
        )
        q_b, s_b, b_b = online_fp8_mod.quantize_online_fp8(w_f32, pol_block)
        assert (
            q_b.dtype == torch.float8_e4m3fn
            and s_b.shape == (1, 1, 1, 128)
            and b_b == (128, 128)
        )

        # quantize_online_fp8 block divisibility error
        with pytest.raises(ValueError, match="must divide local weight shape"):
            online_fp8_mod.quantize_online_fp8(
                torch.randn(64, 128, dtype=torch.float32), pol_block
            )

        # validate_online_fp8 branches: bypassed, not processed, invalid, valid
        def _make_model_with_proj(
            *,
            online_flag: bool = True,
            processed_flag: bool = True,
            keep_layout: bool = True,
            weight: torch.Tensor = q_t,
            scale: torch.Tensor = s_t,
            block_size: tuple[int, int] = (128, 128),
        ) -> nn.Module:
            m = nn.Module()
            proj = nn.Module()
            proj.prefix = "proj"
            proj.quant_method = SimpleNamespace(online_fp8=online_flag)
            proj._tpu_online_fp8_processed = processed_flag
            setattr(proj, KEEP_VLLM_LAYOUT_ATTR, keep_layout)
            proj.weight = nn.Parameter(weight, requires_grad=False)
            proj.weight_scale = nn.Parameter(scale, requires_grad=False)
            proj.weight_block_size = block_size
            m.add_module("proj", proj)
            return m

        # Bypassed dispatch
        good_cfg.online_fp8_policy = Policy(
            targets=["proj"], exclude=[], weight_scheme="per_tensor"
        )
        with pytest.raises(ValueError, match="bypassed quantization dispatch"):
            online_fp8_mod.validate_online_fp8(
                _make_model_with_proj(online_flag=False),
                good_cfg,  # type: ignore[arg-type]
            )

        # Not processed
        with pytest.raises(ValueError, match="was not processed"):
            online_fp8_mod.validate_online_fp8(
                _make_model_with_proj(processed_flag=False),
                good_cfg,  # type: ignore[arg-type]
            )

        # per_tensor valid
        online_fp8_mod.validate_online_fp8(
            _make_model_with_proj(weight=q_t, scale=s_t, block_size=(128, 128)),
            good_cfg,  # type: ignore[arg-type]
        )

        # per_channel valid (keep_layout=False)
        good_cfg.online_fp8_policy = Policy(
            targets=["proj"], exclude=[], weight_scheme="per_channel"
        )
        online_fp8_mod.validate_online_fp8(
            _make_model_with_proj(
                keep_layout=False,
                weight=q_c.t(),
                scale=s_c,
                block_size=(1, 128),
            ),
            good_cfg,  # type: ignore[arg-type]
        )

        # per_block invalid block size on model weight shape
        good_cfg.online_fp8_policy = Policy(
            targets=["proj"],
            exclude=[],
            weight_scheme="per_block",
            weight_block_size=(128, 128),
        )
        with pytest.raises(ValueError, match="Invalid online FP8 block size"):
            online_fp8_mod.validate_online_fp8(
                _make_model_with_proj(
                    weight=torch.zeros((64, 128), dtype=torch.float8_e4m3fn),
                    scale=s_b,
                    block_size=(128, 128),
                ),
                good_cfg,  # type: ignore[arg-type]
            )

        # per_block valid
        online_fp8_mod.validate_online_fp8(
            _make_model_with_proj(weight=q_b, scale=s_b, block_size=(128, 128)),
            good_cfg,  # type: ignore[arg-type]
        )

        # Invalid runtime format
        with pytest.raises(ValueError, match="Invalid online FP8 runtime format"):
            online_fp8_mod.validate_online_fp8(
                _make_model_with_proj(
                    weight=q_b,
                    scale=torch.ones(2, dtype=torch.float32),
                    block_size=(128, 128),
                ),
                good_cfg,  # type: ignore[arg-type]
            )

        # Unmatched targets
        good_cfg.online_fp8_policy = Policy(
            targets=["proj", "missing_target"],
            exclude=[],
            weight_scheme="per_block",
            weight_block_size=(128, 128),
        )
        with pytest.raises(ValueError, match="Invalid online FP8 selection"):
            online_fp8_mod.validate_online_fp8(
                _make_model_with_proj(weight=q_b, scale=s_b, block_size=(128, 128)),
                good_cfg,  # type: ignore[arg-type]
            )
