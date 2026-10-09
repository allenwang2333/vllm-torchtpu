# SPDX-License-Identifier: Apache-2.0
"""TD7.28: FP8 MoE dispatch without per-call imports/kwargs dicts, the
extracted ``_prebuild_moe_ops`` and full ``TpuRowParallelLinear`` coverage."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_torchtpu.layers.adapter.linear as tpu_linear_mod
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

_FP8 = "vllm_torchtpu.layers.adapter.quantization.fp8"


def _method(adaptive=False):
    method = MagicMock(spec=VllmFp8MoEMethodTPU)
    method._use_adaptive_fused_moe = adaptive
    method._tpu_activation_str = "silu"
    return method


def _layer():
    layer = MagicMock()
    layer.w13_weight = torch.zeros(2, 4, 8)
    layer.w2_weight = torch.zeros(2, 8, 4)
    layer.w13_weight_scale_inv = torch.ones(2, 1, 1, 8)
    layer.w2_weight_scale_inv = torch.ones(2, 1, 1, 4)
    layer.w13_bias = None
    layer.w2_bias = None
    layer._experts_start = torch.zeros((), dtype=torch.int32)
    layer.moe_config.experts_per_token = 2
    layer.moe_config.moe_parallel_config.use_ep = True
    layer.renormalize = False
    return layer


# --- supports_internal_mk -------------------------------------------------


@pytest.mark.parametrize(
    "pipelined,ep,adaptive,want",
    [
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
        (False, False, False, False),
    ],
)
def test_supports_internal_mk_reads_patched_bridge(pipelined, ep, adaptive, want):
    method = _method()
    with (
        patch(
            f"{_FP8}.enable_pipelined_collective_and_compute", return_value=pipelined
        ),
        patch.object(bridge, "fused_moe_ep_supported", return_value=ep),
        patch.object(
            fp8.adaptive_moe, "adaptive_fused_moe_supported", return_value=adaptive
        ),
    ):
        got = VllmFp8MoEMethodTPU.supports_internal_mk.fget(method)
    assert got is want


# --- _prebuild_moe_ops -----------------------------------------------------


def test_prebuild_adaptive_skips_gmm_and_ep():
    method = _method(adaptive=True)
    layer = _layer()
    with (
        patch.object(
            fp8.adaptive_moe, "prebuild_adaptive_fused_moe", return_value="op"
        ) as adaptive,
        patch(f"{_FP8}.prebuild_fused_moe_kernel") as gmm,
        patch.object(bridge, "prebuild_fused_moe_ep") as ep,
    ):
        VllmFp8MoEMethodTPU._prebuild_moe_ops(method, layer, "silu")
    adaptive.assert_called_once_with(
        layer, topk=2, renormalize=False, activation="silu"
    )
    assert method._adaptive_fused_moe_op == "op"
    gmm.assert_not_called()
    ep.assert_not_called()


@pytest.mark.parametrize("ep_op", [None, "ep-op"])
def test_prebuild_gmm_and_ep(ep_op):
    method = _method()
    layer = _layer()
    with (
        patch(f"{_FP8}.prebuild_fused_moe_kernel") as gmm,
        patch.object(bridge, "prebuild_fused_moe_ep", return_value=ep_op) as ep,
        patch.object(bridge, "register_score_bias_buffer") as reg,
    ):
        VllmFp8MoEMethodTPU._prebuild_moe_ops(method, layer, "gelu")
    gmm.assert_called_once_with(topk=2, activation="gelu", use_ep=True)
    ep.assert_called_once_with(layer, topk=2, renormalize=False, activation="gelu")
    assert getattr(method, bridge.FUSED_MOE_EP_OP_ATTR) == ep_op
    if ep_op is None:
        reg.assert_not_called()
    else:
        reg.assert_called_once_with(layer)


def test_process_weights_tail_calls_prebuild():
    """The loader still ends by building the ops through the helper."""
    src = inspect.getsource(VllmFp8MoEMethodTPU.process_weights_after_loading)
    assert src.rstrip().endswith("self._prebuild_moe_ops(layer, activation_str)")


# --- apply_monolithic -----------------------------------------------------


def test_apply_monolithic_fused_ep_branch():
    method = _method()
    layer = _layer()
    x = torch.randn(3, 4)
    logits = torch.randn(3, 2)
    out = torch.randn(3, 4)
    with (
        patch.object(bridge, "fused_moe_ep_supported", return_value=True),
        patch.object(bridge, "fused_moe_ep", return_value=out) as ep,
        patch.object(bridge, "score_bias_operand", return_value="bias") as sb,
        patch(f"{_FP8}.fused_moe_gmm") as gmm,
    ):
        got = VllmFp8MoEMethodTPU.apply_monolithic(method, layer, x, logits)
    assert got is out
    sb.assert_called_once_with(layer)
    ep.assert_called_once_with(
        method,
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        logits,
        "bias",
    )
    gmm.assert_not_called()


def test_apply_monolithic_adaptive_branch():
    method = _method(adaptive=True)
    layer = _layer()
    x = torch.randn(3, 4)
    with patch.object(
        fp8.adaptive_moe, "run_adaptive_fused_moe", return_value="y"
    ) as run:
        assert VllmFp8MoEMethodTPU.apply_monolithic(method, layer, x, x) == "y"
    run.assert_called_once()


@pytest.mark.parametrize("skip_padded", [False, True])
@pytest.mark.parametrize("pipelined", [False, True])
def test_apply_monolithic_gmm_dispatch(monkeypatch, skip_padded, pipelined):
    method = _method()
    layer = _layer()
    layer.w13_bias = torch.zeros(2, 1, 8)
    x = torch.randn(3, 4)
    weights = torch.ones(3, 2)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    zeroed = torch.zeros(3, 2)
    monkeypatch.setitem(vars(fp8.envs), "TPU_MOE_SKIP_PADDED_TOKENS", skip_padded)
    pipe_flag = MagicMock(return_value=pipelined)
    with (
        patch.object(bridge, "fused_moe_ep_supported", return_value=False),
        patch.object(fp8.moe_routing, "route", return_value=(weights, ids)),
        patch(f"{_FP8}.enable_pipelined_collective_and_compute", pipe_flag),
        patch.object(
            fp8.token_padding,
            "zero_routing_weights_for_padding",
            return_value=(ids, zeroed),
        ) as zero,
        patch(f"{_FP8}.fused_moe_gmm", return_value="plain") as plain,
        patch(f"{_FP8}.pipelined_fused_moe_gmm", return_value="pipe") as pipe,
    ):
        got = VllmFp8MoEMethodTPU.apply_monolithic(method, layer, x, x)
    assert pipe_flag.call_count == 1  # evaluated once per call
    assert got == ("pipe" if pipelined else "plain")
    called, other = (pipe, plain) if pipelined else (plain, pipe)
    other.assert_not_called()
    kwargs = called.call_args.kwargs
    assert called.call_args.args == ()
    assert kwargs == {
        "hidden_states": x,
        "w1": layer.w13_weight,
        "w2": layer.w2_weight,
        "w1_scale": layer.w13_weight_scale_inv,
        "w2_scale": layer.w2_weight_scale_inv,
        "w1_bias": layer.w13_bias,
        "w2_bias": None,
        "topk_weights": zeroed if skip_padded else weights,
        "topk_ids": ids,
        "experts_start": layer._experts_start,
        "topk": 2,
        "activation": "silu",
    }
    if skip_padded:
        zero.assert_called_once_with(ids, weights, is_local_tensor=pipelined)
    else:
        zero.assert_not_called()


def test_apply_monolithic_requires_processed_weights():
    method = _method()
    method._tpu_activation_str = None
    with pytest.raises(AssertionError, match="process_weights_after_loading"):
        VllmFp8MoEMethodTPU.apply_monolithic(method, _layer(), None, None)


# --- TpuRowParallelLinear -------------------------------------------------


def _row_layer(input_is_parallel, return_bias, bias=True, tp_size=2):
    layer = TpuRowParallelLinear.__new__(TpuRowParallelLinear)
    layer.input_is_parallel = input_is_parallel
    layer.tp_rank = 1
    layer.tp_size = tp_size
    layer.reduce_results = True
    layer.skip_bias_add = False
    layer.bias = torch.tensor([10.0, 20.0]) if bias else None
    layer.return_bias = return_bias
    layer.quant_method = SimpleNamespace(
        apply=lambda _layer, inp, b: (inp.sum(-1, keepdim=True) + inp[..., :1] * 0)
        .expand(-1, 2)
        .clone()
    )
    return layer


@pytest.mark.parametrize("input_is_parallel", [True, False])
@pytest.mark.parametrize("return_bias", [True, False])
def test_row_parallel_deferred_bias(monkeypatch, input_is_parallel, return_bias):
    monkeypatch.setattr(
        tpu_linear_mod, "tensor_model_parallel_all_reduce", lambda t: t * 2
    )
    layer = _row_layer(input_is_parallel, return_bias)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    got = layer.forward(x)
    shard = x if input_is_parallel else x[:, 2:]
    want = shard.sum(-1, keepdim=True).expand(-1, 2) * 2 + layer.bias
    if return_bias:
        out, b = got
        assert b is None
    else:
        out = got
    torch.testing.assert_close(out, want)


def test_row_parallel_without_bias_delegates_to_upstream(monkeypatch):
    layer = _row_layer(True, False, bias=False)
    sentinel = object()
    monkeypatch.setattr(
        tpu_linear_mod.RowParallelLinear, "forward", lambda self, inp: sentinel
    )
    assert layer.forward(torch.zeros(1, 2)) is sentinel
