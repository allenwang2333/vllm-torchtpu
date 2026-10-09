# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Track TD7.27: mxfp4.py and deepseek_v4_fp8.py hot-forward
dispatch optimization and 100% full-module statement & branch coverage."""

from __future__ import annotations

import importlib
from enum import Enum
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.fused_moe_ep import FUSED_MOE_EP_OP_ATTR
from vllm_torchtpu.layers.adapter.quantization import (
    deepseek_v4_fp8 as dsv4_fp8_mod,
)
from vllm_torchtpu.layers.adapter.quantization import (
    mxfp4 as mxfp4_mod,
)
from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.quant_methods import DEEPSEEK_V4_FP8, MXFP4
from vllm_torchtpu.layers.core.quantization import dequantize_mxfp4_packed


def _make_moe_config(*, use_ep: bool = False, has_bias: bool = True) -> FusedMoEConfig:
    return SimpleNamespace(  # type: ignore[return-value]
        experts_per_token=2,
        has_bias=has_bias,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        ep_size=2 if use_ep else 1,
        ep_rank=0,
        is_sequence_parallel=False,
        intermediate_size_per_partition=128,
    )


def _make_routed_experts_instance(
    *,
    activation: str = "silu",
    use_ep: bool = False,
    has_bias: bool = True,
    with_biases: bool = True,
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

    # w13_weight packed shape: [E, 2 * inter, hidden // 2]
    # w13_weight_scale shape: [E, 2 * inter, hidden // 32]
    # w2_weight packed shape: [E, hidden, inter // 2]
    # w2_weight_scale shape: [E, hidden, inter // 32]
    layer.w13_weight = nn.Parameter(
        torch.randint(
            0,
            256,
            (num_experts, 2 * intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale = nn.Parameter(
        torch.full(
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            127,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w2_weight = nn.Parameter(
        torch.randint(
            0,
            256,
            (num_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = nn.Parameter(
        torch.full(
            (num_experts, hidden_size, intermediate_size // 32), 127, dtype=torch.uint8
        ),
        requires_grad=False,
    )
    if with_biases:
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
    def test_reload_modules_for_top_level_coverage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure top-level statements in mxfp4.py and deepseek_v4_fp8.py are traced."""
        import vllm.model_executor.layers.quantization as vllm_quant

        monkeypatch.setattr(
            vllm_quant,
            "register_quantization_config",
            lambda _name: (lambda cls: cls),
        )
        importlib.reload(mxfp4_mod)
        importlib.reload(dsv4_fp8_mod)


class TestVllmDeepseekV4Fp8Config:
    def _make_config(
        self,
        *,
        ignored_layers: list[str] | None = None,
        expert_dtype: str = "fp4",
        moe_quant_algo: str = "MXFP4",
    ) -> VllmDeepseekV4Fp8Config:
        cls = dsv4_fp8_mod.VllmDeepseekV4Fp8Config
        cfg = cls.__new__(cls)
        cfg.ignored_layers = ignored_layers or []
        cfg.packed_modules_mapping = {}
        cfg._resolved_expert_dtype = expert_dtype
        cfg._test_moe_quant_algo = moe_quant_algo  # type: ignore[attr-defined]
        cfg.scale_fmt = None
        return cfg

    def test_get_name(self) -> None:
        assert dsv4_fp8_mod.VllmDeepseekV4Fp8Config.get_name() == DEEPSEEK_V4_FP8

    def test_is_scale_e8m0_branches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = self._make_config(expert_dtype="fp8")

        # 1. hf_config has quantization_config={"scale_fmt": "ue8m0"} -> True
        fake_vllm_cfg_ue8m0 = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(quantization_config={"scale_fmt": "ue8m0"})
            )
        )
        monkeypatch.setattr(
            dsv4_fp8_mod, "get_current_vllm_config", lambda: fake_vllm_cfg_ue8m0
        )
        assert cfg.is_scale_e8m0 is True

        # 2. hf_config has non-ue8m0 quantization_config -> falls back to super()
        fake_vllm_cfg_other = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(quantization_config={"scale_fmt": "fp8"})
            )
        )
        monkeypatch.setattr(
            dsv4_fp8_mod, "get_current_vllm_config", lambda: fake_vllm_cfg_other
        )
        assert cfg.is_scale_e8m0 is False

        # 3. get_current_vllm_config raises Exception -> falls back to super()
        def _raise_err():
            raise RuntimeError("no vllm config")

        monkeypatch.setattr(dsv4_fp8_mod, "get_current_vllm_config", _raise_err)
        assert cfg.is_scale_e8m0 is False

    def test_get_quant_method_all_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            dsv4_fp8_mod.VllmDeepseekV4Fp8Config,
            "moe_quant_algo",
            property(lambda self: self._test_moe_quant_algo),
            raising=False,
        )
        linear_layer = vllm_linear.LinearBase.__new__(vllm_linear.LinearBase)
        nn.Module.__init__(linear_layer)

        routed_layer = _make_routed_experts_instance()

        # 1. LinearBase skipped -> VllmUnquantizedLinearMethod
        cfg_skipped = self._make_config(ignored_layers=["model.layers.0.attn"])
        monkeypatch.setattr(
            cfg_skipped, "get_linear_config", lambda _l: SimpleNamespace()
        )
        m_lin_skip = cfg_skipped.get_quant_method(linear_layer, "model.layers.0.attn")
        assert isinstance(m_lin_skip, VllmUnquantizedLinearMethod)

        # 2. LinearBase not skipped -> VllmFp8LinearMethodTPU
        cfg_active = self._make_config(ignored_layers=[])
        monkeypatch.setattr(
            cfg_active, "get_linear_config", lambda _l: SimpleNamespace()
        )
        monkeypatch.setattr(
            dsv4_fp8_mod,
            "VllmFp8LinearMethodTPU",
            lambda cfg, lin_cfg, prefix=None: ("fp8_lin", cfg, lin_cfg, prefix),
        )
        m_lin_fp8 = cfg_active.get_quant_method(linear_layer, "model.layers.0.attn")
        assert m_lin_fp8[0] == "fp8_lin"  # type: ignore[index]

        # 3. RoutedExperts skipped -> VllmUnquantizedFusedMoEMethod
        monkeypatch.setattr(
            dsv4_fp8_mod,
            "VllmUnquantizedFusedMoEMethod",
            lambda moe_cfg: ("unquant_moe", moe_cfg),
        )
        m_moe_skip = cfg_skipped.get_quant_method(routed_layer, "model.layers.0.attn")
        assert m_moe_skip[0] == "unquant_moe"  # type: ignore[index]

        # 4. RoutedExperts expert_dtype == "fp4" and NVFP4 -> raises
        cfg_nvfp4 = self._make_config(expert_dtype="fp4", moe_quant_algo="NVFP4")
        with pytest.raises(NotImplementedError, match="NVFP4 is not supported yet"):
            cfg_nvfp4.get_quant_method(routed_layer, "model.layers.0.mlp.experts")

        # 5. RoutedExperts expert_dtype == "fp4" and MXFP4 -> VllmDeepseekV4Mxfp4
        cfg_mxfp4 = self._make_config(expert_dtype="fp4", moe_quant_algo="MXFP4")
        monkeypatch.setattr(cfg_mxfp4, "get_moe_config", lambda lyr: lyr.moe_config)
        m_moe_mxfp4 = cfg_mxfp4.get_quant_method(
            routed_layer, "model.layers.0.mlp.experts"
        )
        assert isinstance(m_moe_mxfp4, dsv4_fp8_mod.VllmDeepseekV4Mxfp4MoEMethod)

        # 6. RoutedExperts expert_dtype == "fp8" -> VllmFp8MoEMethodTPU
        cfg_fp8_moe = self._make_config(expert_dtype="fp8")
        monkeypatch.setattr(cfg_fp8_moe, "get_moe_config", lambda lyr: lyr.moe_config)
        monkeypatch.setattr(
            dsv4_fp8_mod,
            "VllmFp8MoEMethodTPU",
            lambda cfg, moe_cfg: ("fp8_moe", cfg, moe_cfg),
        )
        m_moe_fp8 = cfg_fp8_moe.get_quant_method(
            routed_layer, "model.layers.0.mlp.experts"
        )
        assert m_moe_fp8[0] == "fp8_moe"  # type: ignore[index]

        # 7. Other layer -> None
        assert cfg_mxfp4.get_quant_method(nn.Identity(), "model.norm") is None


class TestVllmMxfp4ConfigAndHelpers:
    def test_mxfp4_config_get_name_and_get_quant_method(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert mxfp4_mod.VllmMxfp4Config.get_name() == MXFP4

        cfg = mxfp4_mod.VllmMxfp4Config.__new__(mxfp4_mod.VllmMxfp4Config)
        cfg.ignored_layers = ["model.layers.0.skipped_linear"]
        cfg.packed_modules_mapping = {}

        linear_layer = vllm_linear.LinearBase.__new__(vllm_linear.LinearBase)
        nn.Module.__init__(linear_layer)

        # 1. LinearBase with ignored_layers matching prefix -> skipped path
        res_skipped = cfg.get_quant_method(
            linear_layer, "model.layers.0.skipped_linear"
        )
        assert isinstance(res_skipped, mxfp4_mod.VllmUnquantizedLinearMethod)

        # 2. LinearBase with ignored_layers not matching prefix -> warning path
        res_unskipped = cfg.get_quant_method(
            linear_layer, "model.layers.0.active_linear"
        )
        assert isinstance(res_unskipped, mxfp4_mod.VllmUnquantizedLinearMethod)

        # 3. LinearBase with empty ignored_layers -> warning path
        cfg.ignored_layers = []
        res_empty_ignored = cfg.get_quant_method(
            linear_layer, "model.layers.0.active_linear"
        )
        assert isinstance(res_empty_ignored, mxfp4_mod.VllmUnquantizedLinearMethod)

        # 4. RoutedExperts -> VllmMxfp4MoEMethod
        routed_layer = _make_routed_experts_instance()
        monkeypatch.setattr(cfg, "get_moe_config", lambda lyr: lyr.moe_config)
        res_moe = cfg.get_quant_method(routed_layer, "model.layers.0.experts")
        assert isinstance(res_moe, mxfp4_mod.VllmMxfp4MoEMethod)

        # 5. Attention -> logs warning and returns None
        attn_layer = Attention.__new__(Attention)
        nn.Module.__init__(attn_layer)
        assert cfg.get_quant_method(attn_layer, "model.layers.0.self_attn") is None

        # 6. Unrelated layer -> returns None
        assert cfg.get_quant_method(nn.Identity(), "model.norm") is None

    def test_dequantize_mxfp4_packed_on_device_and_activation_str(self) -> None:
        packed = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(2, 2, 64)
        packed = packed.repeat(1, 4, 1)
        scale = torch.full((2, 8, 4), 127, dtype=torch.uint8)
        expected = dequantize_mxfp4_packed(packed, scale, -1, torch.bfloat16)
        actual = mxfp4_mod._dequantize_mxfp4_packed_on_device(
            packed, scale, -1, torch.bfloat16
        )
        assert torch.equal(actual, expected)

        class _FakeEnum(Enum):
            SILU = "silu"

        assert mxfp4_mod._get_activation_str(_FakeEnum.SILU) == "silu"
        assert mxfp4_mod._get_activation_str("swigluoai") == "swigluoai"


class TestVllmMxfp4MoEMethod:
    def test_properties_and_process_weights_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moe_cfg = _make_moe_config(use_ep=False, has_bias=True)
        method = mxfp4_mod.VllmMxfp4MoEMethod(moe_cfg)
        assert method.is_monolithic is True

        monkeypatch.setattr(
            mxfp4_mod, "enable_pipelined_collective_and_compute", lambda: True
        )
        assert method.supports_internal_mk is True
        monkeypatch.setattr(
            mxfp4_mod, "enable_pipelined_collective_and_compute", lambda: False
        )
        assert method.supports_internal_mk is False

        monkeypatch.setattr(mxfp4_mod, "prebuild_fused_moe_kernel", lambda **kw: None)
        validate_calls: list[object] = []
        monkeypatch.setattr(
            moe_routing,
            "validate_linear_ep_placement",
            lambda lyr: validate_calls.append(lyr),
        )

        # Case 1: activation == "swigluoai", biases present, use_ep=True, CPU device
        layer1 = _make_routed_experts_instance(
            activation="swigluoai", use_ep=True, with_biases=True
        )
        method.process_weights_after_loading(layer1)
        assert len(validate_calls) == 1
        assert layer1.w13_bias is not None and layer1.w13_bias.dtype == torch.float32
        assert layer1.w2_bias is not None and layer1.w2_bias.dtype == torch.float32

        # Case 2 & 3: activation == "silu", simulated TPU device without/with biases
        layer2 = _make_routed_experts_instance(
            activation="silu", use_ep=False, with_biases=False
        )
        layer3 = _make_routed_experts_instance(
            activation="silu", use_ep=False, with_biases=True
        )
        sync_calls: list[list[torch.Tensor]] = []
        monkeypatch.setattr(
            mxfp4_mod, "synchronize_tensors", lambda tensors: sync_calls.append(tensors)
        )

        orig_param = torch.nn.Parameter

        class _FakeTpuParam(torch.nn.Parameter):
            @property
            def device(self):  # type: ignore[override]
                return SimpleNamespace(type="tpu")

        monkeypatch.setattr(torch.nn, "Parameter", _FakeTpuParam)
        monkeypatch.setattr(
            moe_routing,
            "register_experts_start_buffer",
            lambda lyr, device=None: None,
        )
        method.process_weights_after_loading(layer2)
        assert len(sync_calls) == 1
        assert len(sync_calls[-1]) == 2  # only w13_weight and w2_weight

        method.process_weights_after_loading(layer3)
        assert len(sync_calls) == 2
        assert len(sync_calls[-1]) == 4
        monkeypatch.setattr(torch.nn, "Parameter", orig_param)

    def test_forward_monolithic_tpu_pipelined_and_gmm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moe_cfg = _make_moe_config(use_ep=False, has_bias=True)
        method = mxfp4_mod.VllmMxfp4MoEMethod(moe_cfg)
        layer = _make_routed_experts_instance(activation="silu", with_biases=True)

        # Uninitialized _tpu_activation_str asserts
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
            lambda lyr, x, rl, input_ids=None: (
                torch.ones(x.shape[0], 2),
                torch.zeros(x.shape[0], 2, dtype=torch.int32),
            ),
        )

        pipelined_kwargs: dict[str, object] = {}
        gmm_kwargs: dict[str, object] = {}

        def _fake_pipelined(**kwargs):
            pipelined_kwargs.update(kwargs)
            return kwargs["hidden_states"] + 1.0

        def _fake_gmm(**kwargs):
            gmm_kwargs.update(kwargs)
            return kwargs["hidden_states"] + 2.0

        monkeypatch.setattr(mxfp4_mod, "pipelined_fused_moe_gmm", _fake_pipelined)
        monkeypatch.setattr(mxfp4_mod, "fused_moe_gmm", _fake_gmm)

        x = torch.randn(4, 128)
        rl = torch.randn(4, 2)

        # 1. Pipelined branch
        monkeypatch.setattr(
            mxfp4_mod, "enable_pipelined_collective_and_compute", lambda: True
        )
        out_pipe = method._forward_monolithic_tpu(layer, x, rl)
        assert torch.equal(out_pipe, x + 1.0)
        assert pipelined_kwargs["activation"] == "silu"
        assert pipelined_kwargs["topk"] == 2

        # 2. Non-pipelined GMM branch
        monkeypatch.setattr(
            mxfp4_mod, "enable_pipelined_collective_and_compute", lambda: False
        )
        out_gmm = method._forward_monolithic_tpu(layer, x, rl)
        assert torch.equal(out_gmm, x + 2.0)
        assert gmm_kwargs["activation"] == "silu"
        assert gmm_kwargs["topk"] == 2


class TestVllmDeepseekV4Mxfp4MoEMethod:
    def test_supports_internal_mk_and_resolve_activation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moe_cfg = _make_moe_config(use_ep=True)
        method = mxfp4_mod.VllmDeepseekV4Mxfp4MoEMethod(moe_cfg)

        monkeypatch.setattr(
            mxfp4_mod, "enable_pipelined_collective_and_compute", lambda: False
        )
        monkeypatch.setattr(mxfp4_mod, "fused_moe_ep_supported", lambda m: False)
        assert method.supports_internal_mk is False

        monkeypatch.setattr(mxfp4_mod, "fused_moe_ep_supported", lambda m: True)
        assert method.supports_internal_mk is True

        # _resolve_tpu_activation
        assert (
            method._resolve_tpu_activation(SimpleNamespace(activation="silu"))
            == "silu_and_mul_with_clamp"
        )
        assert (
            method._resolve_tpu_activation(SimpleNamespace(activation="gelu")) == "gelu"
        )

    def test_create_weights_and_release_source_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moe_cfg = _make_moe_config(use_ep=False)
        method = mxfp4_mod.VllmDeepseekV4Mxfp4MoEMethod(moe_cfg)
        create_called: list[tuple] = []
        monkeypatch.setattr(
            mxfp4_mod.Mxfp4MoEMethod,
            "create_weights",
            lambda self, layer, *args, **kwargs: create_called.append((args, kwargs)),
        )
        layer = _make_routed_experts_instance(with_biases=True)
        method.create_weights(layer, 2, 128, 128, torch.bfloat16, extra=1)
        assert len(create_called) == 1

        # _release_source_weights with biases present
        method._release_source_weights(layer)
        assert not hasattr(layer, "w13_bias")

        # _release_source_weights without w13_bias attribute
        layer_no_bias = _make_routed_experts_instance(with_biases=False)
        del layer_no_bias.w13_bias
        del layer_no_bias.w2_bias
        method._release_source_weights(layer_no_bias)

    def test_prebuild_w4a8_all_branches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        moe_cfg = _make_moe_config(use_ep=True)
        method = mxfp4_mod.VllmDeepseekV4Mxfp4MoEMethod(moe_cfg)
        layer = _make_routed_experts_instance(use_ep=True)
        w13_pad = torch.empty((2, 512, 1024), dtype=torch.bfloat16, device="meta")
        w2_pad = torch.empty((2, 512, 512), dtype=torch.bfloat16, device="meta")

        # 1. Env disabled -> None
        monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
        assert (
            method._prebuild_w4a8(
                layer, "silu_and_mul_with_clamp", w13_pad, w2_pad, 512
            )
            is None
        )

        # 2. Env enabled, orig_intermediate_size < REQUANTIZED_BLOCK_SIZE -> None
        monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
        monkeypatch.setattr(envs, "MOE_FUSED_EP_ENABLE_W4A8", True)
        assert (
            method._prebuild_w4a8(
                layer, "silu_and_mul_with_clamp", w13_pad, w2_pad, 256
            )
            is None
        )

        # 3. Valid block size -> calls prebuild_fused_moe_ep
        sentinel_op = object()
        monkeypatch.setattr(
            mxfp4_mod, "prebuild_fused_moe_ep", lambda *a, **kw: sentinel_op
        )
        res = method._prebuild_w4a8(
            layer, "silu_and_mul_with_clamp", w13_pad, w2_pad, 512
        )
        assert res is sentinel_op

    def test_dequantize_and_pad_and_process_weights_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moe_cfg = _make_moe_config(use_ep=True)
        method = mxfp4_mod.VllmDeepseekV4Mxfp4MoEMethod(moe_cfg)
        method.intermediate_size = 128

        monkeypatch.setattr(mxfp4_mod, "prebuild_fused_moe_kernel", lambda **kw: None)
        monkeypatch.setattr(
            moe_routing, "validate_linear_ep_placement", lambda lyr: None
        )
        monkeypatch.setattr(
            moe_routing,
            "register_experts_start_buffer",
            lambda lyr, device=None: None,
        )
        monkeypatch.setattr(mxfp4_mod, "synchronize_tensors", lambda tensors: None)
        monkeypatch.setattr(
            mxfp4_mod,
            "quantize_native_fp4_kmajor",
            lambda t, block, pack=True: (
                torch.zeros(
                    (t.shape[0], t.shape[1], t.shape[2] // 2), dtype=torch.uint8
                ),
                torch.ones((t.shape[0], 1, 1, t.shape[2]), dtype=torch.float32),
            ),
        )

        # 1. VLLM_TPU_MXFP4_DEVICE_DEQUANT=True, biases present, op is not None
        monkeypatch.setattr(envs, "VLLM_TPU_MXFP4_DEVICE_DEQUANT", True)
        layer_ep = _make_routed_experts_instance(use_ep=True, with_biases=True)
        sentinel_op = object()
        monkeypatch.setattr(method, "_prebuild_w4a8", lambda *a, **kw: sentinel_op)
        monkeypatch.setattr(
            mxfp4_mod, "fused_moe_ep_unsupported_reason", lambda *a, **kw: None
        )
        bias_reg_calls: list[object] = []
        monkeypatch.setattr(
            mxfp4_mod,
            "register_score_bias_buffer",
            lambda lyr: bias_reg_calls.append(lyr),
        )

        method.process_weights_after_loading(layer_ep)
        assert getattr(method, FUSED_MOE_EP_OP_ATTR) is sentinel_op
        assert len(bias_reg_calls) == 1
        assert hasattr(layer_ep, "_tpu_fused_w13_scale")
        assert hasattr(layer_ep, "_tpu_fused_w2_scale")

        # 2. VLLM_TPU_MXFP4_DEVICE_DEQUANT=False, biases absent, use_ep=False, op=None
        monkeypatch.setattr(envs, "VLLM_TPU_MXFP4_DEVICE_DEQUANT", False)
        layer_no_ep = _make_routed_experts_instance(use_ep=False, with_biases=False)
        del layer_no_ep.w13_bias
        del layer_no_ep.w2_bias
        monkeypatch.setattr(method, "_prebuild_w4a8", lambda *a, **kw: None)
        method.process_weights_after_loading(layer_no_ep)
        assert getattr(method, FUSED_MOE_EP_OP_ATTR) is None
        assert not hasattr(layer_no_ep, "w13_bias")
        assert not hasattr(layer_no_ep, "w2_bias")

    def test_forward_monolithic_tpu_ep_and_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moe_cfg = _make_moe_config(use_ep=True)
        method = mxfp4_mod.VllmDeepseekV4Mxfp4MoEMethod(moe_cfg)
        method._tpu_activation_str = "silu_and_mul_with_clamp"
        layer = _make_routed_experts_instance(use_ep=True, with_biases=False)
        layer._tpu_fused_w13_scale = torch.ones((2, 1, 256), dtype=torch.float32)
        layer._tpu_fused_w2_scale = torch.ones((2, 1, 128), dtype=torch.float32)

        x = torch.randn(4, 128)
        rl = torch.randn(4, 2)

        # 1. fused_moe_ep_supported(self) is True -> dispatches to fused_moe_ep
        monkeypatch.setattr(mxfp4_mod, "fused_moe_ep_supported", lambda m: True)
        monkeypatch.setattr(mxfp4_mod, "score_bias_operand", lambda lyr: None)
        monkeypatch.setattr(mxfp4_mod, "hash_topk_ids_operand", lambda lyr, ids: None)
        ep_calls: list[tuple] = []

        def _fake_ep(*args):
            ep_calls.append(args)
            return x + 5.0

        monkeypatch.setattr(mxfp4_mod, "fused_moe_ep", _fake_ep)
        out_ep = method._forward_monolithic_tpu(layer, x, rl)
        assert torch.equal(out_ep, x + 5.0)
        assert len(ep_calls) == 1

        # 2. fused_moe_ep_supported(self) is False -> super()._forward_monolithic_tpu
        monkeypatch.setattr(mxfp4_mod, "fused_moe_ep_supported", lambda m: False)
        monkeypatch.setattr(
            mxfp4_mod, "enable_pipelined_collective_and_compute", lambda: False
        )
        monkeypatch.setattr(
            moe_routing,
            "route",
            lambda lyr, x, rl, input_ids=None: (
                torch.ones(x.shape[0], 2),
                torch.zeros(x.shape[0], 2, dtype=torch.int32),
            ),
        )
        monkeypatch.setattr(mxfp4_mod, "fused_moe_gmm", lambda **kw: x + 10.0)
        out_fallback = method._forward_monolithic_tpu(layer, x, rl)
        assert torch.equal(out_fallback, x + 10.0)
