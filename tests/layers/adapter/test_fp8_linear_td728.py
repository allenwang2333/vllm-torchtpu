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


# --- TD8.82: Full coverage for fp8.py, deepseek_v41.py, nvfp4.py ----------


class _FakeTpuParam(torch.nn.Parameter):
    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


def test_noop_init_fp8_linear_kernel():
    assert fp8._noop_init_fp8_linear_kernel(1, 2, k="v") is None


def test_dequantize_fp8_linear_unaligned_block():
    w = torch.ones((100, 100), dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale_inv = torch.full((2, 2), 2.0, dtype=torch.float32)
    out = fp8._dequantize_fp8_linear(
        w,
        weight_scale=None,
        weight_scale_inv=scale_inv,
        block_quant=True,
        weight_block_size=(64, 64),
        out_dtype=torch.float32,
    )
    assert out.shape == (100, 100)
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_moe_and_linear_quant_config_env_branches(monkeypatch):
    monkeypatch.setitem(vars(fp8.envs), "MOE_REQUANTIZE_WEIGHT_DTYPE", None)
    monkeypatch.setitem(vars(fp8.envs), "MOE_REQUANTIZE_BLOCK_SIZE", 128)
    dtype_name, dtype, bsize = fp8._get_moe_quant_config("test")
    assert dtype_name == "float8_e4m3fn"
    assert dtype == torch.float8_e4m3fn
    assert bsize == 128

    monkeypatch.setitem(vars(fp8.envs), "MOE_REQUANTIZE_WEIGHT_DTYPE", "invalid_dtype")
    with pytest.raises(ValueError, match="Unsupported MOE_REQUANTIZE_WEIGHT_DTYPE"):
        fp8._get_moe_quant_config("test")

    monkeypatch.setitem(vars(fp8.envs), "ENABLE_QUANTIZED_MATMUL_KERNEL", False)
    monkeypatch.setitem(vars(fp8.envs), "REQUANTIZE_BLOCK_SIZE", None)
    monkeypatch.setitem(vars(fp8.envs), "REQUANTIZE_WEIGHT_DTYPE", "invalid_dtype")
    with pytest.raises(ValueError, match="Unsupported REQUANTIZE_WEIGHT_DTYPE"):
        fp8._get_linear_quant_config(None)


def test_format_linear_scale_and_process_linear_default_config(monkeypatch):
    s_1d = torch.ones(4, dtype=torch.float32)
    out = fp8._format_linear_scale_for_runtime(s_1d, blockwise_kernel=False)
    assert out.shape == (4,)

    monkeypatch.setitem(vars(fp8.envs), "ENABLE_QUANTIZED_MATMUL_KERNEL", False)
    monkeypatch.setitem(vars(fp8.envs), "REQUANTIZE_BLOCK_SIZE", None)
    monkeypatch.setitem(vars(fp8.envs), "REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn")
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.ones(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            requires_grad=False,
        ),
        weight_scale=torch.nn.Parameter(torch.ones(16, 1), requires_grad=False),
        logical_widths=None,
    )
    w, ws, dt, bs = fp8._process_fp8_linear_weights(
        layer,
        block_quant=False,
        weight_block_size=None,
        linear_config=None,
        linear_quant_config=None,
    )
    assert w.shape == (16, 32)
    assert ws.shape == (16,)
    assert dt == "float8_e4m3fn"
    assert bs is None


def test_quantize_and_format_single_moe_weight_branches():
    # swigluoai + w13 with unaligned half (half=64 -> aligned_half=128)
    w13 = torch.ones(2, 128, 128, dtype=torch.float32)
    w_q, s_q = fp8._quantize_and_format_single_moe_weight(
        w13,
        name="w13",
        quant_dtype=torch.float8_e4m3fn,
        block_size=None,
        activation="swigluoai",
    )
    assert w_q.shape == (2, 128, 256)
    assert s_q.shape == (2, 1, 1, 256)

    # name="other" skips w13 and w2 padding
    w_other, _ = fp8._quantize_and_format_single_moe_weight(
        torch.ones(2, 64, 128, dtype=torch.float32),
        name="other",
        quant_dtype=torch.float8_e4m3fn,
        block_size=None,
    )
    assert w_other.shape == (2, 128, 64)

    # unaligned contracting dimension for block_size raises ValueError
    with pytest.raises(ValueError, match="not divisible by block_size"):
        fp8._quantize_and_format_single_moe_weight(
            torch.ones(2, 256, 100, dtype=torch.float32),
            name="w13",
            quant_dtype=torch.float8_e4m3fn,
            block_size=128,
        )


def test_requantize_expert_chunks_tpu_sync(monkeypatch):
    synced = []
    monkeypatch.setattr(fp8, "synchronize_tensors", synced.append)
    monkeypatch.setattr(fp8, "_REQUANT_CHUNK_BYTES", 1)

    orig_fn = fp8._quantize_and_format_single_moe_weight

    def fake_single(fp32, **kw):
        w, s = orig_fn(fp32, **kw)
        return _FakeTpuParam(w, requires_grad=False), s

    monkeypatch.setattr(fp8, "_quantize_and_format_single_moe_weight", fake_single)
    w = torch.ones(2, 256, 128, dtype=torch.float32)
    out_w, out_s = fp8._requantize_expert_chunks(
        w,
        None,
        lambda chunk, _s: chunk,
        None,
        name="w13",
        quant_dtype=torch.float8_e4m3fn,
        block_size=None,
    )
    assert out_w.shape == (2, 128, 256)
    assert out_s.shape == (2, 1, 1, 256)
    assert len(synced) == 2


def test_process_fp8_moe_weights_validation_and_scale_shapes(monkeypatch):
    monkeypatch.setitem(vars(fp8.envs), "MOE_REQUANTIZE_WEIGHT_DTYPE", None)
    monkeypatch.setitem(vars(fp8.envs), "MOE_REQUANTIZE_BLOCK_SIZE", None)

    # 1. Missing scales
    bad_layer = SimpleNamespace(w13_weight=torch.ones(2, 256, 128))
    with pytest.raises(ValueError, match="Missing MoE weight scale parameters"):
        fp8._process_fp8_moe_weights(bad_layer, activation="silu")

    # 2. Block-quantized MoE with unaligned weight shape
    unaligned_layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.ones(2, 200, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.ones(2, 128, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            requires_grad=False,
        ),
        w13_weight_scale_inv=torch.nn.Parameter(
            torch.ones(2, 2, 1), requires_grad=False
        ),
        w2_weight_scale_inv=torch.nn.Parameter(
            torch.ones(2, 1, 1), requires_grad=False
        ),
        moe_config=SimpleNamespace(
            intermediate_size_per_partition=128,
            intermediate_size_per_partition_unpadded=128,
            tp_size=1,
            tp_rank=0,
        ),
    )
    with pytest.raises(
        ValueError, match="FP8 block quantized MoE weights must be divisible"
    ):
        fp8._process_fp8_moe_weights(
            unaligned_layer, weight_block_size=(128, 128), activation="silu"
        )

    # 3. Non-block-quantized with padding (tp_rank=1 -> local_real < padded,
    # and tp_rank=0 -> local_real == padded) plus s13 / s2 shapes:
    E, inter, H = 2, 256, 128
    w13_q = torch.ones(E, 2 * inter, H, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w2_q = torch.ones(E, H, inter, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    for tp_rank, s13_shape, s2_shape in [
        (1, (E, 2), (E, H)),
        (0, (E, 2 * inter), (E,)),
        (0, (E,), (E, 1, 1)),
        (0, (E, 1, 1), (E, 1)),
        (0, (E, 1), (E, 1)),
    ]:
        lyr = SimpleNamespace(
            w13_weight=torch.nn.Parameter(w13_q.clone(), requires_grad=False),
            w2_weight=torch.nn.Parameter(w2_q.clone(), requires_grad=False),
            w13_weight_scale=torch.nn.Parameter(
                torch.ones(s13_shape), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.ones(s2_shape), requires_grad=False
            ),
            moe_config=SimpleNamespace(
                intermediate_size_per_partition=inter,
                intermediate_size_per_partition_unpadded=192,
                tp_size=2,
                tp_rank=tp_rank,
            ),
        )
        w13_out, s13_out, w2_out, s2_out, _, _ = fp8._process_fp8_moe_weights(
            lyr, weight_block_size=None, activation="silu"
        )
        assert w13_out.shape == (E, H, 2 * inter)
        assert w2_out.shape == (E, inter, H)
        assert s13_out.shape == (E, 1, 1, 2 * inter)
        assert s2_out.shape == (E, 1, 1, H)


def test_vllm_fp8_config_checkpoint_quant_method_branches(monkeypatch):
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_torchtpu.layers.adapter.quantization import unquantized

    cfg = fp8.VllmFp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=["model.layers.0.mlp.experts"],
    )
    sentinel = object()
    monkeypatch.setattr(
        unquantized, "VllmUnquantizedFusedMoEMethod", lambda moe_cfg: sentinel
    )
    monkeypatch.setattr(cfg, "get_moe_config", lambda lyr: "moe_cfg")

    routed = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed)
    assert (
        cfg._get_checkpoint_quant_method(routed, "model.layers.0.mlp.experts")
        is sentinel
    )

    attn = Attention.__new__(Attention)
    torch.nn.Module.__init__(attn)
    assert cfg._get_checkpoint_quant_method(attn, "model.layers.0.self_attn") is None

    other = torch.nn.Module()
    assert cfg._get_checkpoint_quant_method(other, "model.norm") is None


def test_vllm_fp8_moe_method_create_and_process_weights(monkeypatch):
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_torchtpu.layers.adapter.quantization import unquantized

    monkeypatch.setitem(vars(fp8.envs), "MOE_REQUANTIZE_WEIGHT_DTYPE", None)
    monkeypatch.setitem(vars(fp8.envs), "MOE_REQUANTIZE_BLOCK_SIZE", None)
    monkeypatch.setitem(vars(fp8.envs), "MOE_FUSED_EP_KERNEL_IMPL", None)

    # 1. is_monolithic
    cfg_chan = fp8.VllmFp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=None
    )
    cfg_chan.is_channel_quant = True
    moe_cfg = SimpleNamespace(
        has_bias=False,
        experts_per_token=2,
        intermediate_size_per_partition=128,
        intermediate_size_per_partition_unpadded=128,
        tp_size=1,
        tp_rank=0,
        moe_parallel_config=SimpleNamespace(use_ep=True),
    )
    method = VllmFp8MoEMethodTPU(cfg_chan, moe_cfg)
    assert method.is_monolithic is True

    # 2. create_weights with is_channel_quant=True and weight_loader propagation
    def fake_parent_create(self, lyr, *a, **kw):
        lyr.w13_weight = torch.nn.Parameter(
            torch.zeros(2, 256, 128, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        lyr.w2_weight = torch.nn.Parameter(
            torch.zeros(2, 128, 128, dtype=torch.float8_e4m3fn), requires_grad=False
        )

    monkeypatch.setattr(fp8.Fp8MoEMethod, "create_weights", fake_parent_create)
    routed = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed)
    routed.has_bias = False
    routed.weight_loader = lambda *a: None
    routed.moe_config = moe_cfg
    routed.activation = SimpleNamespace(value="silu")
    routed.renormalize = True
    method.create_weights(routed, 2, 128, 128, torch.bfloat16)
    assert routed.w13_weight_scale.shape == (2, 256, 1)
    assert routed.w2_weight_scale.shape == (2, 128, 1)
    assert routed.w13_weight.weight_loader is routed.weight_loader

    # 3. process_weights_after_loading on TPU-shimmed FP8 serialized + use_ep=True
    synced = []
    validated_ep = []
    monkeypatch.setattr(fp8, "synchronize_tensors", synced.append)
    monkeypatch.setattr(
        fp8.moe_routing, "validate_linear_ep_placement", validated_ep.append
    )
    monkeypatch.setattr(
        fp8.moe_routing, "register_experts_start_buffer", lambda *a, **k: None
    )
    monkeypatch.setattr(method, "_prebuild_moe_ops", lambda *a, **k: None)
    monkeypatch.setattr(torch.nn, "Parameter", _FakeTpuParam)
    method.process_weights_after_loading(routed)
    assert len(synced) == 1
    assert len(validated_ep) == 1
    assert not hasattr(routed, "w13_weight_scale")
    assert not hasattr(routed, "w2_weight_scale")

    # 4. create_weights and process_weights_after_loading with BF16 online FP8
    cfg_bf16 = fp8.VllmFp8Config(
        is_checkpoint_fp8_serialized=False, weight_block_size=None
    )
    method_bf16 = VllmFp8MoEMethodTPU(cfg_bf16, moe_cfg)
    unquant_called = []
    monkeypatch.setattr(
        unquantized.VllmUnquantizedFusedMoEMethod,
        "create_weights",
        lambda self, lyr, *a, **kw: unquant_called.append(lyr),
    )
    routed_bf16 = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed_bf16)
    routed_bf16.moe_config = SimpleNamespace(
        has_bias=False,
        experts_per_token=2,
        moe_parallel_config=SimpleNamespace(use_ep=False),
    )
    routed_bf16.activation = SimpleNamespace(value="silu")
    routed_bf16.renormalize = True
    method_bf16.create_weights(routed_bf16, 2, 128, 128, torch.bfloat16)
    assert unquant_called == [routed_bf16]

    routed_bf16.w13_weight = _FakeTpuParam(
        torch.ones(2, 256, 128, dtype=torch.bfloat16), requires_grad=False
    )
    routed_bf16.w2_weight = _FakeTpuParam(
        torch.ones(2, 128, 128, dtype=torch.bfloat16), requires_grad=False
    )
    monkeypatch.setattr(method_bf16, "_prebuild_moe_ops", lambda *a, **k: None)
    monkeypatch.setattr(torch.nn, "Parameter", torch.nn.Parameter)
    method_bf16.process_weights_after_loading(routed_bf16)
    assert routed_bf16.w13_weight.dtype == torch.float8_e4m3fn


def test_channel_quant_scale_parameter_tpu(monkeypatch):
    import vllm.model_executor.parameter as vllm_param_mod
    from vllm.model_executor.parameter import ChannelQuantScaleParameter

    monkeypatch.setattr(vllm_param_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm_param_mod, "get_tensor_model_parallel_world_size", lambda: 1
    )

    calls = []
    monkeypatch.setattr(
        ChannelQuantScaleParameter,
        "load_column_parallel_weight",
        lambda self, w: calls.append(("col", w.shape)),
    )
    monkeypatch.setattr(
        ChannelQuantScaleParameter,
        "load_merged_column_weight",
        lambda self, w, **kw: calls.append(("merged", w.shape)),
    )
    monkeypatch.setattr(
        ChannelQuantScaleParameter,
        "load_qkv_weight",
        lambda self, w, **kw: calls.append(("qkv", w.shape)),
    )

    p = fp8.ChannelQuantScaleParameterTPU(
        data=torch.zeros((4, 1), dtype=torch.float32),
        output_dim=0,
        weight_loader=lambda *a: None,
    )
    w_1d = torch.ones(4, dtype=torch.float32)
    w_2d = torch.full((4, 1), 2.0, dtype=torch.float32)

    p.load_column_parallel_weight(w_1d)
    p.load_column_parallel_weight(w_2d)
    p.load_merged_column_weight(w_1d)
    p.load_merged_column_weight(w_2d)
    p.load_qkv_weight(w_1d)
    p.load_qkv_weight(w_2d)
    p.load_row_parallel_weight(w_1d)
    assert torch.all(p.data == 1.0)
    p.load_row_parallel_weight(w_2d)
    assert torch.all(p.data == 2.0)
    assert len(calls) == 6


def test_vllm_fp8_linear_method_create_process_and_apply_branches(monkeypatch):
    import vllm.model_executor.parameter as vllm_param_mod
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_torchtpu.layers.adapter.linear_common import (
        KEEP_VLLM_LAYOUT_ATTR,
        WEIGHT_FLIPPED_ATTR,
    )

    monkeypatch.setattr(vllm_param_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm_param_mod, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setitem(vars(fp8.envs), "ENABLE_QUANTIZED_MATMUL_KERNEL", False)
    monkeypatch.setitem(vars(fp8.envs), "REQUANTIZE_BLOCK_SIZE", None)
    monkeypatch.setitem(vars(fp8.envs), "REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn")

    # 1. create_weights with is_channel_quant=True and weight_scale_inv registration
    cfg = fp8.VllmFp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    cfg.is_channel_quant = True
    method = fp8.VllmFp8LinearMethodTPU(cfg)

    def fake_linear_parent_create(self, lyr, *a, **kw):
        lyr.weight = torch.nn.Parameter(
            torch.ones(128, 128, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        lyr.weight_scale_inv = torch.nn.Parameter(
            torch.ones(1, 1, dtype=torch.float32), requires_grad=False
        )

    monkeypatch.setattr(
        fp8.Fp8LinearMethod, "create_weights", fake_linear_parent_create
    )
    lyr = torch.nn.Module()
    lyr.weight_loader = lambda *a: None
    method.create_weights(lyr, 128, [128], 128, 128, torch.bfloat16)
    assert isinstance(lyr.weight_scale, fp8.ChannelQuantScaleParameterTPU)
    assert lyr.weight.weight_loader is lyr.weight_loader

    # Also test weight_scale_inv -> BlockQuantScaleParameter when is_channel_quant=False
    cfg.is_channel_quant = False
    lyr2 = torch.nn.Module()
    method.create_weights(lyr2, 128, [128], 128, 128, torch.bfloat16)
    assert hasattr(lyr2, "weight_scale")

    # 2. process_weights_after_loading:
    # (a) non-serialized, non-online_fp8 with _linear_quant_config=None
    cfg_non_ser = fp8.VllmFp8Config(is_checkpoint_fp8_serialized=False)
    method_non_ser = fp8.VllmFp8LinearMethodTPU(cfg_non_ser)
    method_non_ser._linear_quant_config = None
    lyr_non_ser = torch.nn.Module()
    lyr_non_ser.weight = torch.nn.Parameter(
        torch.ones(16, 32, dtype=torch.bfloat16), requires_grad=False
    )
    synced = []
    monkeypatch.setattr(fp8, "synchronize_tensors", synced.append)
    monkeypatch.setattr(torch.nn, "Parameter", _FakeTpuParam)
    method_non_ser.process_weights_after_loading(lyr_non_ser)
    assert len(synced) == 1
    assert getattr(lyr_non_ser, WEIGHT_FLIPPED_ATTR, False) is True
    assert lyr_non_ser.weight_block_size == (1, 32)

    # Re-entering process_weights_after_loading returns early via WEIGHT_FLIPPED_ATTR
    method_non_ser.process_weights_after_loading(lyr_non_ser)
    assert len(synced) == 1

    # (b) weight_block_size derivation for 1D non-matching and 2D weight_scale
    monkeypatch.setattr(torch.nn, "Parameter", torch.nn.Parameter)
    for fake_scale, expected_bs in [
        (torch.ones(1, dtype=torch.float32), (16, 1)),
        (torch.ones(2, 2, dtype=torch.float32), (16, 32)),
    ]:
        monkeypatch.setattr(
            fp8,
            "_format_linear_scale_for_runtime",
            lambda _s, *, blockwise_kernel, _fs=fake_scale: _fs,
        )
        lyr_bs = torch.nn.Module()
        lyr_bs.weight = torch.nn.Parameter(
            torch.ones(16, 32, dtype=torch.bfloat16), requires_grad=False
        )
        method_non_ser.process_weights_after_loading(lyr_bs)
        assert lyr_bs.weight_block_size == expected_bs

    # 3. apply with bias is None vs not None, and KEEP_VLLM_LAYOUT_ATTR
    monkeypatch.setattr(
        fp8,
        "quantized_matmul",
        lambda x, w, s: torch.matmul(x.float(), w.float()),
    )
    lyr_apply = SimpleNamespace(
        weight=torch.ones(16, 8, dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(8, dtype=torch.float32),
    )
    x = torch.ones(4, 16, dtype=torch.bfloat16)
    out_no_bias = method.apply(lyr_apply, x, bias=None)
    assert out_no_bias.shape == (4, 8)
    out_bias = method.apply(lyr_apply, x, bias=torch.ones(8))
    assert torch.allclose(out_bias, out_no_bias + 1.0)

    setattr(lyr_apply, KEEP_VLLM_LAYOUT_ATTR, True)
    lyr_apply.weight = torch.ones(8, 16, dtype=torch.float8_e4m3fn)
    out_keep = method.apply(lyr_apply, x, bias=None)
    assert out_keep.shape == (4, 8)

    # 4. resolve_online_fp8 with non-LinearBase layer raises ValueError
    cfg.online_fp8_policy = SimpleNamespace(
        select=lambda prefix, mapping: (True, [prefix]),
        selected={},
    )
    with pytest.raises(ValueError, match="Online FP8 target is not a supported Linear"):
        fp8.resolve_online_fp8(cfg, torch.nn.Module(), "model.layers.0.attn", None)
    assert not isinstance(torch.nn.Module(), LinearBase)


def test_dsv41_cached_gate_slice_plan_and_nvfp4_unaligned_w4a8(monkeypatch):
    from vllm_torchtpu.layers.adapter.quantization import nvfp4
    from vllm_torchtpu.layers.adapter.quantization.deepseek_v41 import DeepseekV41MoE

    # 1. deepseek_v41.py line 89->91: second call hits cached _gate_slice_plan
    moe = DeepseekV41MoE.__new__(DeepseekV41MoE)
    moe.use_sequence_parallel = False
    moe._gate_slice_plan = None
    x = torch.randn(4, 8)
    assert moe._maybe_slice_gate_input(x) is x

    # 2. nvfp4.py lines 211-217: _prebuild_w4a8 with block <= 0
    monkeypatch.setitem(vars(fp8.envs), "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setitem(vars(fp8.envs), "MOE_FUSED_EP_ENABLE_W4A8", True)
    nv_layer = SimpleNamespace(
        w13_weight=torch.zeros(2, 32, 16, dtype=torch.uint8),
        w2_weight=torch.zeros(2, 32, 8, dtype=torch.uint8),
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(use_ep=True)),
    )
    assert nvfp4._prebuild_w4a8(nv_layer, "silu", 0) == (None, 0)
