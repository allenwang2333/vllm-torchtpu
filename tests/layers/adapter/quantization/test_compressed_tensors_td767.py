# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TD7.67: compressed-tensors MoE dispatch, CPU loader hook and FP4 helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.layers.adapter.quantization.compressed_tensors import (
    compressed_tensors as ct,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe as ct_moe,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    utils as ct_utils,
)

# ---------------------------------------------------------------------------
# utils: FP4 nibble sign and zero-safe reciprocal
# ---------------------------------------------------------------------------


def _old_nibbles(tensor_scaled):
    abs_x = tensor_scaled.abs()
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=tensor_scaled.dtype
    )
    mag = torch.bucketize(abs_x, thresholds).to(torch.uint8)
    sign = torch.where(
        tensor_scaled < 0,
        torch.tensor(8, dtype=torch.uint8),
        torch.tensor(0, dtype=torch.uint8),
    )
    return mag | sign


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp4_nibbles_match_old_sign_construction(dtype) -> None:
    torch.manual_seed(0)
    x = (torch.randn(64, 33) * 4).to(dtype)
    x[0, :6] = torch.tensor([0.0, -0.0, 6.0, -6.0, 0.25, -0.25]).to(dtype)
    got = ct_utils.quantize_to_fp4_e2m1_nibbles(x)
    want = _old_nibbles(x)
    assert got.dtype == torch.uint8
    assert torch.equal(got, want)
    assert int(got.max()) <= 15


def test_scalar_zero_where_is_bitwise_equal() -> None:
    scale = torch.tensor([0.0, 1.0, 3.0, 0.0, 1e-30, 7.0], dtype=torch.float32)
    old = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    new = torch.where(scale == 0, 0.0, 1.0 / scale)
    assert new.dtype == old.dtype == torch.float32
    assert torch.equal(new.view(torch.int32), old.view(torch.int32))


# ---------------------------------------------------------------------------
# utils: CPU weight-loader hook branches
# ---------------------------------------------------------------------------


def _layer(mapping=lambda e: e):
    return SimpleNamespace(_map_global_expert_id_to_local_expert_id=mapping)


def test_hook_skips_non_local_expert() -> None:
    orig = MagicMock()
    hook = ct_utils.get_cpu_weight_loader_hook(_layer(lambda e: -1), orig, 1, 0)
    param = torch.nn.Parameter(torch.empty(2, 4, 4))
    assert hook(param, torch.ones(4, 4), "w13_weight_packed", "w1", 7) is False
    orig.assert_not_called()
    assert not hasattr(param, "_cpu_scratch")


def test_hook_non_main_weight_without_orig_loader() -> None:
    hook = ct_utils.get_cpu_weight_loader_hook(_layer(), None, 1, 0)
    param = torch.nn.Parameter(torch.empty(2, 4, 4))
    assert hook(param, torch.ones(4), "w13_bias", "w1", 0) is False


def test_hook_main_weights_shard_into_transposed_scratch() -> None:
    tp_size, tp_rank = 2, 1
    hook = ct_utils.get_cpu_weight_loader_hook(_layer(), None, tp_size, tp_rank)
    # Parameter is [E, K, N]; the scratch is untransposed [E, N, K].
    w13 = torch.nn.Parameter(torch.empty(2, 3, 8), requires_grad=False)
    w13._orig_shape = (2, 3, 8)
    full = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    assert hook(w13, full, "w13_weight_packed", "w3", 1) is True
    scratch = w13._cpu_scratch
    assert scratch.shape == (2, 8, 3)
    # rank 1 of 2 takes rows 4..8; "w3" lands in the second half.
    assert torch.equal(scratch.data[1, 4:8], full[4:8])

    w2 = torch.nn.Parameter(torch.empty(2, 4, 3), requires_grad=False)
    full2 = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
    assert hook(w2, full2, "w2_weight_packed", "w2", 0) is True
    assert w2._cpu_scratch.shape == (2, 3, 4)
    assert torch.equal(w2._cpu_scratch.data[0], full2[:, 4:8])

    s13 = torch.nn.Parameter(torch.empty(2, 1, 8), requires_grad=False)
    full_s13 = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    assert hook(s13, full_s13, "w13_weight_scale", "w1", 0) is True
    assert torch.equal(s13._cpu_scratch.data[0, 0:8], full_s13[8:16])

    s2 = torch.nn.Parameter(torch.empty(2, 2, 1), requires_grad=False)
    full_s2 = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    assert hook(s2, full_s2, "w2_weight_scale", "w2", 1) is True
    assert s2._cpu_scratch.shape == (2, 1, 2)
    assert torch.equal(s2._cpu_scratch.data[1], full_s2[:, 2:4])


def test_hook_untransposed_scratch_keeps_shape() -> None:
    hook = ct_utils.get_cpu_weight_loader_hook(
        _layer(), None, 1, 0, is_param_transposed=False
    )
    p = torch.nn.Parameter(torch.empty(2, 4, 3), requires_grad=False)
    loaded = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert hook(p, loaded, "w2_weight_packed", "w2", 0) is True
    assert p._cpu_scratch.shape == (2, 4, 3)
    assert torch.equal(p._cpu_scratch.data[0], loaded)


def test_release_memory_to_os_tolerates_missing_malloc_trim(monkeypatch) -> None:
    def boom(_name):
        raise OSError("no libc")

    monkeypatch.setattr(ct_utils.ctypes, "CDLL", boom)
    ct_utils.release_memory_to_os()  # must not raise


# ---------------------------------------------------------------------------
# compressed_tensors: int4 strategy check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy,expected",
    [
        (ct.QuantizationStrategy.GROUP, True),
        (ct.QuantizationStrategy.CHANNEL, True),
        (ct.QuantizationStrategy.TENSOR, False),
    ],
)
def test_is_int4_w4aN_strategy(strategy, expected) -> None:
    w = SimpleNamespace(
        num_bits=4, type=ct.QuantizationType.INT, strategy=strategy, dynamic=False
    )
    assert ct._is_int4_w4aN(w) is expected
    assert ct._is_int4_w4aN(None) is False


# ---------------------------------------------------------------------------
# compressed_tensors_moe: dispatch guard branches
# ---------------------------------------------------------------------------


def _routed_experts():
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    return layer


def test_get_moe_method_rejects_mixed_schemes() -> None:
    config = MagicMock()
    config.get_scheme_dict.side_effect = [{"a": 1}, {"a": 2}, {"a": 1}]
    with pytest.raises(ValueError, match="same quantization scheme"):
        ct_moe.VllmCompressedTensorsMoEMethod.get_moe_method(
            config, _routed_experts(), "model.layers.0.mlp.experts"
        )
    names = [c.args[1] for c in config.get_scheme_dict.call_args_list]
    assert names == [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
        "model.layers.0.mlp.experts.0.down_proj",
    ]
    config._add_fused_moe_to_target_scheme_map.assert_called_once_with()


def test_get_moe_method_ignored_layer_is_unquantized(monkeypatch) -> None:
    built = []
    monkeypatch.setattr(
        ct_moe,
        "VllmUnquantizedFusedMoEMethod",
        lambda moe_config: built.append(moe_config) or "unquantized",
    )
    config = MagicMock()
    config.get_scheme_dict.return_value = None
    config.get_moe_config.return_value = "moe_cfg"
    layer = _routed_experts()
    assert (
        ct_moe.VllmCompressedTensorsMoEMethod.get_moe_method(config, layer, "x")
        == "unquantized"
    )
    assert built == ["moe_cfg"]
    config.get_moe_config.assert_called_once_with(layer)
