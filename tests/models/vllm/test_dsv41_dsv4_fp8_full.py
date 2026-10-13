# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for deepseek_v4_fp8 quantization config."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.layers", ("vllm_torchtpu", "layers")),
    ("vllm_torchtpu.layers.adapter", ("vllm_torchtpu", "layers", "adapter")),
    (
        "vllm_torchtpu.layers.adapter.quantization",
        ("vllm_torchtpu", "layers", "adapter", "quantization"),
    ),
    ("vllm_torchtpu.layers.core", ("vllm_torchtpu", "layers", "core")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _pkg


class _StubLinearBase(nn.Module):
    pass


class _StubLinearMethodBase:
    pass


class _StubQuantizeMethodBase:
    pass


class _StubRoutedExperts(nn.Module):
    def __init__(self, moe_config: Any = None) -> None:
        super().__init__()
        self.moe_config = moe_config or SimpleNamespace(name="moe_cfg")


class _StubDeepseekV4FP8Config:
    scale_fmt: str | None = None
    ignored_layers: list[str]
    packed_modules_mapping: dict[str, list[str]]
    expert_dtype: str
    moe_quant_algo: str

    @property
    def is_scale_e8m0(self) -> bool:
        return self.scale_fmt == "ue8m0"

    def get_linear_config(self, layer: nn.Module) -> Any:
        return ("linear_cfg", layer)

    def get_moe_config(self, layer: nn.Module) -> Any:
        return getattr(layer, "moe_config", ("moe_cfg", layer))


class _StubVllmQuantConfig:
    pass


class _StubFp8LinearMethodTPU(_StubLinearMethodBase):
    def __init__(self, quant_config: Any, linear_config: Any, prefix: str = "") -> None:
        self.quant_config = quant_config
        self.linear_config = linear_config
        self.prefix = prefix


class _StubFp8MoEMethodTPU(_StubQuantizeMethodBase):
    def __init__(self, quant_config: Any, moe_config: Any) -> None:
        self.quant_config = quant_config
        self.moe_config = moe_config


class _StubMxfp4MoEMethod(_StubQuantizeMethodBase):
    def __init__(self, moe_config: Any) -> None:
        self.moe_config = moe_config


class _StubUnquantLinearMethod(_StubLinearMethodBase):
    pass


class _StubUnquantMoEMethod(_StubQuantizeMethodBase):
    def __init__(self, moe_config: Any) -> None:
        self.moe_config = moe_config


def _stub_is_layer_skipped(
    prefix: str,
    ignored_layers: list[str] | None = None,
    fused_mapping: dict[str, list[str]] | None = None,
) -> bool:
    _ = fused_mapping
    return bool(ignored_layers and prefix in ignored_layers)


def _install_stubs() -> None:
    def _ensure_mod(name: str, **attrs: Any) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for k, v in attrs.items():
            if not hasattr(mod, k):
                setattr(mod, k, v)
        return mod

    _ensure_mod("vllm")
    _ensure_mod(
        "vllm.config",
        get_current_vllm_config=lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(quantization_config={"scale_fmt": "ue8m0"})
            )
        ),
    )
    _ensure_mod("vllm.model_executor")
    _ensure_mod("vllm.model_executor.layers")
    _ensure_mod(
        "vllm.model_executor.layers.linear",
        LinearBase=_StubLinearBase,
        LinearMethodBase=_StubLinearMethodBase,
    )
    _ensure_mod(
        "vllm.model_executor.layers.fused_moe",
        RoutedExperts=_StubRoutedExperts,
    )
    _ensure_mod(
        "vllm.model_executor.layers.quantization",
        QuantizationMethods=str,
        register_quantization_config=lambda _name: (lambda cls: cls),
    )
    _ensure_mod(
        "vllm.model_executor.layers.quantization.base_config",
        QuantizeMethodBase=_StubQuantizeMethodBase,
    )
    _ensure_mod("vllm.model_executor.layers.quantization.utils")
    _ensure_mod(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        is_layer_skipped=_stub_is_layer_skipped,
    )
    _ensure_mod("vllm.models")
    _ensure_mod("vllm.models.deepseek_v4")
    _ensure_mod(
        "vllm.models.deepseek_v4.quant_config",
        DeepseekV4FP8Config=_StubDeepseekV4FP8Config,
    )
    _ensure_mod(
        "vllm_torchtpu.layers.adapter.quantization.configs",
        VllmQuantConfig=_StubVllmQuantConfig,
    )
    _ensure_mod(
        "vllm_torchtpu.layers.adapter.quantization.fp8",
        VllmFp8LinearMethodTPU=_StubFp8LinearMethodTPU,
        VllmFp8MoEMethodTPU=_StubFp8MoEMethodTPU,
    )
    _ensure_mod(
        "vllm_torchtpu.layers.adapter.quantization.mxfp4",
        VllmDeepseekV4Mxfp4MoEMethod=_StubMxfp4MoEMethod,
    )
    _ensure_mod(
        "vllm_torchtpu.layers.adapter.quantization.unquantized",
        VllmUnquantizedFusedMoEMethod=_StubUnquantMoEMethod,
        VllmUnquantizedLinearMethod=_StubUnquantLinearMethod,
    )
    _ensure_mod(
        "vllm_torchtpu.layers.core.quant_methods",
        DEEPSEEK_V4_FP8="deepseek_v4_fp8",
        get_tpu_quant_method=lambda name: f"tpu-{name}",
    )
    _ensure_mod(
        "vllm_torchtpu.logger",
        init_logger=lambda _name: SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None
        ),
    )


_install_stubs()

from vllm_torchtpu.layers.adapter.quantization import (  # noqa: E402
    deepseek_v4_fp8 as dsv4_fp8_mod,
)


def _make_cfg(
    *,
    ignored_layers: list[str] | None = None,
    expert_dtype: str = "fp4",
    moe_quant_algo: str = "MXFP4",
    scale_fmt: str | None = None,
) -> dsv4_fp8_mod.VllmDeepseekV4Fp8Config:
    cls = dsv4_fp8_mod.VllmDeepseekV4Fp8Config
    cfg = cls.__new__(cls)
    cfg.ignored_layers = ignored_layers if ignored_layers is not None else []
    cfg.packed_modules_mapping = {}
    cfg.__dict__["expert_dtype"] = expert_dtype
    cfg.__dict__["moe_quant_algo"] = moe_quant_algo
    cfg.scale_fmt = scale_fmt
    return cfg


def test_deepseek_v4_fp8_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise 100% of statements and branches in deepseek_v4_fp8.py."""
    assert dsv4_fp8_mod.VllmDeepseekV4Fp8Config.get_name() == "deepseek_v4_fp8"

    cfg = _make_cfg(expert_dtype="fp8", scale_fmt=None)

    # 1. is_scale_e8m0: dict with scale_fmt == "ue8m0" -> True
    monkeypatch.setattr(
        dsv4_fp8_mod,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(quantization_config={"scale_fmt": "ue8m0"})
            )
        ),
    )
    assert cfg.is_scale_e8m0 is True

    # 2. is_scale_e8m0: dict with scale_fmt != "ue8m0" -> falls back to base
    monkeypatch.setattr(
        dsv4_fp8_mod,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(quantization_config={"scale_fmt": "fp8"})
            )
        ),
    )
    assert cfg.is_scale_e8m0 is False
    cfg.scale_fmt = "ue8m0"
    assert cfg.is_scale_e8m0 is True
    cfg.scale_fmt = None

    # 3. is_scale_e8m0: Exception raised -> falls back to base
    def _raise_err() -> Any:
        raise RuntimeError("no vllm config")

    monkeypatch.setattr(dsv4_fp8_mod, "get_current_vllm_config", _raise_err)
    assert cfg.is_scale_e8m0 is False

    # 4. get_quant_method on LinearBase:
    #    (a) ignored_layers matched -> VllmUnquantizedLinearMethod
    lin_cls = dsv4_fp8_mod._LinearBase
    linear_layer = lin_cls.__new__(lin_cls)
    nn.Module.__init__(linear_layer)

    cfg_skip = _make_cfg(ignored_layers=["model.layers.0.self_attn.q_proj"])
    lin_cfg_calls = 0

    def _count_lin_cfg(_lyr: nn.Module) -> Any:
        nonlocal lin_cfg_calls
        lin_cfg_calls += 1
        return "lin_cfg"

    monkeypatch.setattr(cfg_skip, "get_linear_config", _count_lin_cfg)
    monkeypatch.setattr(dsv4_fp8_mod, "is_layer_skipped", _stub_is_layer_skipped)
    m_skip_lin = cfg_skip.get_quant_method(
        linear_layer, "model.layers.0.self_attn.q_proj"
    )
    assert isinstance(m_skip_lin, dsv4_fp8_mod.VllmUnquantizedLinearMethod)
    assert lin_cfg_calls == 0

    #    (b) ignored_layers non-empty & unmatched -> VllmFp8LinearMethodTPU
    m_active_lin_nonempty = cfg_skip.get_quant_method(
        linear_layer, "model.layers.0.self_attn.k_proj"
    )
    assert isinstance(m_active_lin_nonempty, dsv4_fp8_mod.VllmFp8LinearMethodTPU)
    assert lin_cfg_calls == 1

    #    (c) ignored_layers empty -> short-circuits is_layer_skipped
    cfg_empty = _make_cfg(ignored_layers=[])
    monkeypatch.setattr(cfg_empty, "get_linear_config", _count_lin_cfg)
    m_active_lin_empty = cfg_empty.get_quant_method(
        linear_layer, "model.layers.0.self_attn.v_proj"
    )
    assert isinstance(m_active_lin_empty, dsv4_fp8_mod.VllmFp8LinearMethodTPU)
    assert lin_cfg_calls == 2

    # 5. get_quant_method on RoutedExperts:
    routed_cls = dsv4_fp8_mod.RoutedExperts
    routed_layer = routed_cls.__new__(routed_cls)
    nn.Module.__init__(routed_layer)
    routed_layer.moe_config = SimpleNamespace(name="moe_cfg")

    #    (a) ignored_layers non-empty & matched -> VllmUnquantizedFusedMoEMethod
    m_skip_moe = cfg_skip.get_quant_method(
        routed_layer, "model.layers.0.self_attn.q_proj"
    )
    assert isinstance(m_skip_moe, dsv4_fp8_mod.VllmUnquantizedFusedMoEMethod)

    #    (b) expert_dtype == "fp4" and NVFP4 -> raises NotImplementedError
    cfg_nvfp4 = _make_cfg(
        ignored_layers=["other"], expert_dtype="fp4", moe_quant_algo="NVFP4"
    )
    monkeypatch.setattr(
        dsv4_fp8_mod.VllmDeepseekV4Fp8Config,
        "expert_dtype",
        property(lambda self: self.__dict__.get("expert_dtype", "fp4")),
        raising=False,
    )
    monkeypatch.setattr(
        dsv4_fp8_mod.VllmDeepseekV4Fp8Config,
        "moe_quant_algo",
        property(lambda self: self.__dict__.get("moe_quant_algo", "MXFP4")),
        raising=False,
    )
    with pytest.raises(NotImplementedError, match="NVFP4 is not supported yet"):
        cfg_nvfp4.get_quant_method(routed_layer, "model.layers.0.mlp.experts")

    #    (c) expert_dtype == "fp4" and MXFP4 -> VllmDeepseekV4Mxfp4MoEMethod
    cfg_mxfp4 = _make_cfg(ignored_layers=[], expert_dtype="fp4", moe_quant_algo="MXFP4")
    monkeypatch.setattr(cfg_mxfp4, "get_moe_config", lambda lyr: lyr.moe_config)
    m_mxfp4 = cfg_mxfp4.get_quant_method(routed_layer, "model.layers.0.mlp.experts")
    assert isinstance(m_mxfp4, dsv4_fp8_mod.VllmDeepseekV4Mxfp4MoEMethod)

    #    (d) expert_dtype == "fp8" -> VllmFp8MoEMethodTPU
    cfg_fp8 = _make_cfg(ignored_layers=[], expert_dtype="fp8", moe_quant_algo="FP8")
    monkeypatch.setattr(cfg_fp8, "get_moe_config", lambda lyr: lyr.moe_config)
    m_fp8_moe = cfg_fp8.get_quant_method(routed_layer, "model.layers.0.mlp.experts")
    assert isinstance(m_fp8_moe, dsv4_fp8_mod.VllmFp8MoEMethodTPU)

    # 6. get_quant_method on non-Linear / non-RoutedExperts -> None
    assert cfg_fp8.get_quant_method(nn.Identity(), "model.norm") is None
