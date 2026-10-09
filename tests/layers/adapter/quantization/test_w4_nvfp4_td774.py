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
"""CPU tests for the W4 compressed-tensors MoE methods and NVFP4 (TD7.74).

Covers the load-time activation cache, the scheme/shape validation errors,
the MXFP4 ``process_weights_after_loading`` body (its ``.to("tpu")`` hops are
shimmed so it runs on CPU), and the NVFP4 branches the per-format suites miss.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter.quantization import nvfp4, unquantized
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a16 as w4a16,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4an_mxfp4 as mxfp4,
)


class _FakeTpuParam(torch.nn.Parameter):
    """Parameter that reports a TPU device so the synchronize branch runs."""

    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


_UNSET = object()


@pytest.fixture
def env_attr():
    """Override ``vllm_torchtpu.envs`` values, then restore the module dict.

    ``envs`` resolves names through a lazy module ``__getattr__``. Undoing a
    ``monkeypatch.setattr`` writes the looked-up value back as a real module
    attribute, which then shadows every later ``setenv`` in the session.
    Pop the override instead when the name was not a real attribute before.
    """
    saved: dict[str, object] = {}

    def _set(name, value):
        saved.setdefault(name, vars(envs).get(name, _UNSET))
        setattr(envs, name, value)

    yield _set
    for name, old in saved.items():
        if old is _UNSET:
            delattr(envs, name)
        else:
            setattr(envs, name, old)


class _Experts(RoutedExperts):
    """RoutedExperts without the vLLM config plumbing."""

    def __init__(self, activation="silu", use_ep=False):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock(w13_num_shards=2)
        self.moe_config.experts_per_token = 2
        self.moe_config.moe_parallel_config.use_ep = use_ep
        self.activation = activation
        self.renormalize = True


# --------------------------------------------------------------------------
# W4A16
# --------------------------------------------------------------------------


def _quant_args(**overrides):
    args = dict(num_bits=4, strategy="group", group_size=8, symmetric=True, type="int")
    args.update(overrides)
    return SimpleNamespace(**args)


def _w4a16_method(**overrides):
    moe = MagicMock(tp_size=1, tp_rank=0)
    return w4a16.VllmCompressedTensorsW4A16MoEMethod(
        _quant_args(**overrides), None, moe
    )


class TestW4A16:
    def test_is_monolithic(self):
        assert _w4a16_method().is_monolithic is True

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"type": "float"}, "integer INT4 weights only"),
            ({"type": None}, "integer INT4 weights only"),
            ({"num_bits": 8}, "integer INT4 weights only"),
            ({"symmetric": False}, "symmetric INT4 only"),
        ],
    )
    def test_scheme_validation(self, overrides, match):
        with pytest.raises(NotImplementedError, match=match):
            _w4a16_method(**overrides)._validate_w4a16_scheme()

    def test_carrier_validation(self):
        method = _w4a16_method()
        layer = torch.nn.Module()
        with pytest.raises(ValueError, match="w13_weight_packed"):
            method._validate_int32_weight_carriers(layer)
        layer.w13_weight_packed = torch.nn.Parameter(
            torch.empty(0, dtype=torch.uint8), requires_grad=False
        )
        with pytest.raises(NotImplementedError, match="packed in INT32"):
            method._validate_int32_weight_carriers(layer)

    @pytest.mark.parametrize("group_size", [None, 0])
    def test_invalid_group_size(self, group_size):
        method = _w4a16_method(group_size=group_size)
        with pytest.raises(ValueError, match="Invalid group_size"):
            method.create_weights(_Experts(), 2, 16, 16, torch.float32)

    def test_non_divisible_sizes(self):
        with pytest.raises(ValueError, match="must be divisible by group_size"):
            _w4a16_method().create_weights(_Experts(), 2, 16, 12, torch.float32)

    def test_channel_strategy_single_group(self):
        layer = _Experts()
        _w4a16_method(strategy="channel").create_weights(
            layer, 2, 16, 16, torch.float32
        )
        assert layer.num_groups_w13 == layer.num_groups_w2 == 1
        assert layer.w13_weight_scale._orig_shape == (2, 1, 32)

    def test_process_caches_activation_for_apply(self, monkeypatch, env_attr):
        E, H, inter = 2, 16, 16
        layer = _Experts()
        method = _w4a16_method()
        method.create_weights(layer, E, H, inter, torch.float32)
        layer.w13_weight_packed.data = torch.zeros(
            E, 2 * inter, H // 8, dtype=torch.int32
        )
        layer.w2_weight_packed.data = torch.zeros(E, H, inter // 8, dtype=torch.int32)
        layer.w13_weight_scale.data = torch.ones(E, 2 * inter, H // 8)
        layer.w2_weight_scale.data = torch.ones(E, H, inter // 8)
        layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)

        prebuilt = []
        env_attr("MOE_REQUANTIZE_BLOCK_SIZE", None)
        env_attr("MOE_REQUANTIZE_WEIGHT_DTYPE", None)
        monkeypatch.setattr(w4a16, "release_memory_to_os", lambda: None)
        monkeypatch.setattr(
            w4a16.moe_routing, "register_experts_start_buffer", lambda *a, **k: None
        )
        monkeypatch.setattr(
            w4a16, "prebuild_fused_moe_kernel", lambda **kw: prebuilt.append(kw)
        )
        method.process_weights_after_loading(layer)

        assert method._tpu_activation_str == "silu"
        assert prebuilt[0]["activation"] == "silu"
        assert layer.weight is layer.w13_weight_packed
        assert layer.w13_weight_packed.shape == (E, H // 8, 2 * inter)

        # apply reads the cached string, not layer.activation.
        layer.activation = "must-not-be-read"
        layer._experts_start = None
        ids = torch.zeros(4, 2, dtype=torch.int32)
        monkeypatch.setattr(
            w4a16.moe_routing, "route", lambda *a: (torch.ones(4, 2), ids)
        )
        monkeypatch.setattr(
            w4a16, "enable_pipelined_collective_and_compute", lambda: False
        )
        monkeypatch.setattr(w4a16, "fused_moe_gmm", lambda **kw: kw)
        out = method.apply_monolithic(layer, torch.zeros(4, H), torch.zeros(4, E))
        assert out["activation"] == "silu"
        assert out["topk_ids"] is ids

    def test_apply_before_process_asserts(self):
        method = _w4a16_method()
        with pytest.raises(AssertionError, match="did not run"):
            method.apply_monolithic(_Experts(), torch.zeros(1), torch.zeros(1))


# --------------------------------------------------------------------------
# W4AN MXFP4
# --------------------------------------------------------------------------


class _HostTensor:
    """Stands in for a CPU scratch tensor; ``.to("tpu")`` stays on CPU."""

    def __init__(self, tensor):
        self.tensor = tensor

    def to(self, device):
        assert device == "tpu"
        return self.tensor


def _mxfp4_layer(unpadded=64, use_ep=False):
    moe_config = MagicMock(w13_num_shards=2, tp_size=1, tp_rank=0)
    moe_config.intermediate_size_per_partition = 64
    moe_config.intermediate_size_per_partition_unpadded = unpadded
    method = mxfp4.VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = _Experts(use_ep=use_ep)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    for name in ("w13_weight_scale", "w2_weight_scale"):
        getattr(layer, name).data.fill_(127)
    for name in ("w13_weight_packed", "w2_weight_packed"):
        getattr(layer, name).data.zero_()
    method._loaded_data = lambda p: _HostTensor(p.data)
    return method, layer


@pytest.fixture
def mxfp4_env(monkeypatch, env_attr):
    calls = SimpleNamespace(sync=[], prebuild=[], ep=[], requant=[], load=[])
    monkeypatch.setattr(mxfp4, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(mxfp4, "release_memory_to_os", lambda: None)
    monkeypatch.setattr(mxfp4, "synchronize_tensors", calls.sync.append)
    monkeypatch.setattr(
        mxfp4.moe_routing, "register_experts_start_buffer", lambda *a, **k: None
    )
    monkeypatch.setattr(
        mxfp4.moe_routing, "validate_linear_ep_placement", calls.ep.append
    )
    monkeypatch.setattr(
        mxfp4, "prebuild_fused_moe_kernel", lambda **kw: calls.prebuild.append(kw)
    )

    def load(w):
        calls.load.append(w)
        return w.clone()

    def requant(w, s, block):
        calls.requant.append(block)
        return w.clone(), s.clone()

    monkeypatch.setattr(mxfp4, "load_kmajor_fp4", load)
    monkeypatch.setattr(mxfp4, "requant_load_kmajor_fp4", requant)
    env_attr("MOE_REQUANTIZE_BLOCK_SIZE", None)
    return calls


class TestW4ANMxfp4:
    def test_is_monolithic_and_fresh(self):
        method, _ = _mxfp4_layer()
        assert method.is_monolithic is True
        src = torch.arange(6).reshape(2, 3).t()
        out = mxfp4._fresh(src)
        assert torch.equal(out, src) and out.is_contiguous()

    def test_neutralize_rejects_unaligned_unpadded(self):
        method, layer = _mxfp4_layer(unpadded=48)
        with pytest.raises(ValueError, match="divisible by 32"):
            method._neutralize_padded_scales(layer)

    def test_process_block32_layout(self, mxfp4_env, monkeypatch):
        method, layer = _mxfp4_layer()
        layer.w13_weight_packed._cpu_scratch = object()
        layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        method.process_weights_after_loading(layer)

        assert len(mxfp4_env.load) == 2 and not mxfp4_env.requant
        assert not hasattr(layer, "w13_weight_packed")
        assert layer.weight is layer.w13_weight
        assert layer.w13_bias is None and layer.w2_bias is None
        # [E, N, blocks] e8m0 -> [E, blocks, 1, N] fp32; 127 decodes to 1.0.
        assert layer.w13_weight_scale.shape == (2, 4, 1, 128)
        assert layer.w2_weight_scale.shape == (2, 2, 1, 128)
        assert torch.all(layer.w13_weight_scale == 1.0)
        assert not mxfp4_env.sync and not mxfp4_env.ep
        assert mxfp4_env.prebuild == [
            dict(topk=2, activation="silu", use_ep=False, skip_padded_tokens=True)
        ]
        assert method._tpu_activation_str == "silu"

    def test_process_requant_tpu_and_ep(self, mxfp4_env, monkeypatch, env_attr):
        method, layer = _mxfp4_layer(use_ep=True)
        env_attr("MOE_REQUANTIZE_BLOCK_SIZE", 64)
        monkeypatch.setattr(torch.nn, "Parameter", _FakeTpuParam)
        method.process_weights_after_loading(layer)

        assert mxfp4_env.requant == [64, 64] and not mxfp4_env.load
        assert len(mxfp4_env.sync) == 1 and len(mxfp4_env.sync[0]) == 4
        assert mxfp4_env.ep == [layer]
        assert mxfp4_env.prebuild[0]["use_ep"] is True

    def test_process_requant_block_must_divide(self, mxfp4_env, monkeypatch, env_attr):
        method, layer = _mxfp4_layer()
        env_attr("MOE_REQUANTIZE_BLOCK_SIZE", 48)
        with pytest.raises(ValueError, match="divisible by block \\(48\\)"):
            method.process_weights_after_loading(layer)


# --------------------------------------------------------------------------
# NVFP4
# --------------------------------------------------------------------------


def _nvfp4_moe_layer(E=2, two_i=32, H=64, group=16):
    inter = two_i // 2
    layer = _Experts()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(E, two_i, H // 2, dtype=torch.uint8), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(E, H, inter // 2, dtype=torch.uint8), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones(E, two_i, H // group), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones(E, H, inter // group), requires_grad=False
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(E, 2), requires_grad=False)
    layer.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(E), requires_grad=False)
    return layer


def _nvfp4_moe_method():
    return nvfp4.VllmNvfp4MoEMethod(
        MagicMock(group_size=16), MagicMock(has_bias=False, is_act_and_mul=True)
    )


@pytest.fixture
def nvfp4_env(monkeypatch, env_attr):
    calls = SimpleNamespace(sync=[], prebuild=[])
    env_attr("MOE_REQUANTIZE_BLOCK_SIZE", None)
    env_attr("MOE_FUSED_EP_KERNEL_IMPL", None)
    env_attr("USE_MOE_FUSED_EP_KERNEL", False)
    monkeypatch.setattr(
        nvfp4, "load_kmajor_fp4", lambda t: t.transpose(-1, -2).contiguous()
    )
    monkeypatch.setattr(nvfp4, "synchronize_tensors", calls.sync.append)
    monkeypatch.setattr(
        nvfp4.moe_routing, "register_experts_start_buffer", lambda *a, **k: None
    )
    monkeypatch.setattr(
        nvfp4, "prebuild_fused_moe_kernel", lambda **kw: calls.prebuild.append(kw)
    )
    return calls


class TestNvfp4:
    def test_excluded_moe_returns_unquantized(self, monkeypatch):
        cfg = nvfp4.VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        cfg.vllm_config = MagicMock()
        cfg.exclude_modules = ["model.layers.0.skip_moe"]
        sentinel = object()
        monkeypatch.setattr(
            unquantized, "VllmUnquantizedFusedMoEMethod", lambda moe: sentinel
        )
        monkeypatch.setattr(cfg, "get_moe_config", lambda layer: "moe")
        assert cfg.get_quant_method(_Experts(), "model.layers.0.skip_moe") is sentinel

    def test_supports_internal_mk_reads_bridge(self, monkeypatch):
        method = _nvfp4_moe_method()
        monkeypatch.setattr(
            nvfp4, "enable_pipelined_collective_and_compute", lambda: False
        )
        monkeypatch.setattr(bridge, "fused_moe_ep_supported", lambda m: m is method)
        assert method.supports_internal_mk is True

    def test_process_on_tpu_synchronizes(self, nvfp4_env, monkeypatch):
        layer = _nvfp4_moe_layer()
        method = _nvfp4_moe_method()
        monkeypatch.setattr(torch.nn, "Parameter", _FakeTpuParam)
        method.process_weights_after_loading(layer)
        assert len(nvfp4_env.sync) == 1 and len(nvfp4_env.sync[0]) == 4
        assert nvfp4_env.prebuild[0]["activation"] == "silu"
        assert getattr(method, bridge.FUSED_MOE_EP_OP_ATTR) is None

    def test_adaptive_rejected_prepared_weights_raise(
        self, nvfp4_env, monkeypatch, env_attr
    ):
        layer = _nvfp4_moe_layer()
        method = _nvfp4_moe_method()
        env_attr("MOE_FUSED_EP_KERNEL_IMPL", "adaptive")
        monkeypatch.setattr(
            nvfp4.adaptive_moe,
            "prebuild_adaptive_nvfp4",
            lambda layer, act, block: (object(), None),
        )
        monkeypatch.setattr(
            nvfp4.adaptive_moe,
            "fused_moe_ep_unsupported_reason",
            lambda *a, **k: "bad layout",
        )
        with pytest.raises(ValueError, match="adaptive NVFP4 prepared weights: bad"):
            method.process_weights_after_loading(layer)

    def test_w4a8_admission_reads_bridge(self, monkeypatch, env_attr):
        layer = _nvfp4_moe_layer()
        layer.moe_config.moe_parallel_config.use_ep = True
        env_attr("USE_MOE_FUSED_EP_KERNEL", True)
        env_attr("MOE_FUSED_EP_ENABLE_W4A8", True)
        seen = {}

        def prebuild(lyr, **kw):
            seen.update(kw)
            return "op"

        monkeypatch.setattr(bridge, "prebuild_fused_moe_ep", prebuild)
        op, block = nvfp4._prebuild_w4a8(layer, "silu", None)
        assert op == "op"
        assert block == seen["rhs_qb"] == nvfp4.U32_SUBLANE_TILE * nvfp4.PACK4

    def test_apply_fused_ep_reads_bridge(self, monkeypatch):
        method = _nvfp4_moe_method()
        method._tpu_activation_str = "silu"
        layer = SimpleNamespace(
            w13_weight=1, w2_weight=2, _tpu_fused_w13_scale=3, _tpu_fused_w2_scale=4
        )
        monkeypatch.setattr(bridge, "fused_moe_ep_supported", lambda m: True)
        monkeypatch.setattr(bridge, "score_bias_operand", lambda lyr: "bias")
        monkeypatch.setattr(bridge, "fused_moe_ep", lambda *a: a)
        out = method.apply_monolithic(layer, "x", "logits")
        assert out == (method, "x", 1, 2, 3, 4, "logits", "bias")

    def test_linear_create_weights_preset_bias_no_act_key(self, monkeypatch):
        cfg = nvfp4.VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        method = nvfp4.VllmNvfp4LinearMethod(cfg, MagicMock())
        method.akey = None
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        layer.has_bias = "preset"
        monkeypatch.setattr(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
        )
        monkeypatch.setattr(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            lambda: 1,
        )
        method.create_weights(layer, 64, [32], 64, 32, torch.bfloat16)
        assert layer.has_bias == "preset"
        assert not hasattr(layer, "input_scale")
        assert hasattr(layer.weight_scale_2, "weight_loader")

    def test_linear_process_on_tpu_without_input_scale(self, monkeypatch):
        N, K = 32, 64
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        layer.weight = torch.nn.Parameter(
            torch.zeros(N, K // 2, dtype=torch.uint8), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(N, K // 16), requires_grad=False
        )
        layer.weight_scale_2 = torch.nn.Parameter(
            torch.tensor([2.0]), requires_grad=False
        )
        method = nvfp4.VllmNvfp4LinearMethod(
            MagicMock(group_size=16, quant_method="NVFP4"), MagicMock()
        )
        synced = []
        monkeypatch.setattr(nvfp4, "load_kmajor_fp4", lambda t: t.t().contiguous())
        monkeypatch.setattr(nvfp4, "synchronize_tensors", synced.append)
        monkeypatch.setattr(torch.nn, "Parameter", _FakeTpuParam)
        method.process_weights_after_loading(layer)
        assert len(synced) == 1 and len(synced[0]) == 2
        assert not hasattr(layer, "weight_scale_2")
        assert torch.all(layer.weight_scale.data == 2.0)
