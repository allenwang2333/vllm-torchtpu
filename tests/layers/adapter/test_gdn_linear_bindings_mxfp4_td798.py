# SPDX-License-Identifier: Apache-2.0
"""Full-module CPU unit tests for TD7.98 target modules:
- src/vllm_torchtpu/layers/adapter/gdn_linear.py
- src/vllm_torchtpu/layers/adapter/linear.py
- src/vllm_torchtpu/layers/adapter/upstream_bindings.py
- .../compressed_tensors_moe/compressed_tensors_moe_w4an_mxfp4.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm.model_executor.layers.linear as vllm_linear_mod
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)

from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry
from vllm_torchtpu.layers.adapter import gdn_linear, linear, upstream_bindings
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4an_mxfp4 as mxfp4_mod,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
    VllmCompressedTensorsW4ANMxfp4MoEMethod,
    _fresh,
    _to_kernel_scale,
)

# ---------------------------------------------------------------------------
# 1. upstream_bindings.py
# ---------------------------------------------------------------------------


def _sample_helper(x: int) -> int:
    return x + 1


def _sample_upstream_fn(x: int, y: int = 2, *, z: int = 3) -> int:
    return _sample_helper(x) * y + z


def test_upstream_bindings_device_hooks_and_bind_device_ops() -> None:
    target = SimpleNamespace(op_a=10, op_b=20)

    # Empty replacements (0 loop iterations)
    with upstream_bindings.device_hooks(target):
        assert target.op_a == 10

    # Non-empty replacements + normal exit
    with upstream_bindings.device_hooks(target, op_a=100, op_b=200):
        assert target.op_a == 100
        assert target.op_b == 200
    assert target.op_a == 10
    assert target.op_b == 20

    # Exception inside context still restores previous attributes
    with (
        pytest.raises(RuntimeError, match="boom"),
        upstream_bindings.device_hooks(target, op_a=999),
    ):
        assert target.op_a == 999
        raise RuntimeError("boom")
    assert target.op_a == 10

    # bind_device_ops creates an isolated module namespace with overridden ops
    bound_fn = upstream_bindings.bind_device_ops(
        _sample_upstream_fn, _sample_helper=lambda x: x + 10
    )
    assert _sample_upstream_fn(4) == (4 + 1) * 2 + 3
    assert bound_fn(4) == (4 + 10) * 2 + 3
    assert bound_fn(4, 3, z=5) == (4 + 10) * 3 + 5
    assert bound_fn.__qualname__ == _sample_upstream_fn.__qualname__
    assert bound_fn.__kwdefaults__ == {"z": 3}


# ---------------------------------------------------------------------------
# 2. linear.py (TpuRowParallelLinear)
# ---------------------------------------------------------------------------


def _make_row_parallel_layer(
    *,
    tp_rank: int = 0,
    tp_size: int = 2,
    input_is_parallel: bool = True,
    reduce_results: bool = True,
    skip_bias_add: bool = False,
    bias: torch.Tensor | None = None,
    return_bias: bool = False,
) -> linear.TpuRowParallelLinear:
    layer = linear.TpuRowParallelLinear.__new__(linear.TpuRowParallelLinear)
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.input_is_parallel = input_is_parallel
    layer.reduce_results = reduce_results
    layer.skip_bias_add = skip_bias_add
    layer.bias = bias
    layer.return_bias = return_bias
    layer.quant_method = SimpleNamespace(
        apply=lambda lyr, inp, b: inp * 2.0 + (0.0 if b is None else b)
    )
    return layer


def test_tpu_row_parallel_linear_all_branches(monkeypatch) -> None:
    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", lambda t: t + 1.0)
    monkeypatch.setattr(
        vllm_linear_mod, "tensor_model_parallel_all_reduce", lambda t: t + 1.0
    )

    x_par = torch.ones((2, 4), dtype=torch.float32)
    x_full = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    bias = torch.full((4,), 0.5, dtype=torch.float32)

    # 1. defer_bias=False (e.g., tp_size=1 or bias=None) -> delegates to super().forward
    layer_no_defer = _make_row_parallel_layer(tp_size=1, bias=bias, return_bias=False)
    out_no_defer = layer_no_defer.forward(x_par)
    torch.testing.assert_close(out_no_defer, x_par * 2.0 + bias)

    # 2. defer_bias=True, input_is_parallel=True, return_bias=False
    layer_par_no_ret = _make_row_parallel_layer(
        tp_rank=0,
        tp_size=2,
        input_is_parallel=True,
        bias=bias,
        return_bias=False,
    )
    out_par = layer_par_no_ret.forward(x_par)
    torch.testing.assert_close(out_par, (x_par * 2.0 + 1.0) + bias)

    # 3. defer_bias=True, input_is_parallel=False, return_bias=True
    layer_split_ret = _make_row_parallel_layer(
        tp_rank=1,
        tp_size=2,
        input_is_parallel=False,
        bias=bias,
        return_bias=True,
    )
    out_split, ret_bias = layer_split_ret.forward(x_full)
    assert ret_bias is None
    torch.testing.assert_close(out_split, (x_full[:, 4:] * 2.0 + 1.0) + bias)


# ---------------------------------------------------------------------------
# 3. gdn_linear.py
# ---------------------------------------------------------------------------


def _patch_tp(monkeypatch, tp_size: int, tp_rank: int) -> None:
    import vllm.distributed as distributed_mod
    from vllm.model_executor import parameter as parameter_mod
    from vllm.model_executor.layers import linear as linear_mod
    from vllm.model_executor.model_loader import weight_utils

    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: tp_rank)
    for mod in (linear_mod, parameter_mod, distributed_mod):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(
            mod, "get_tensor_model_parallel_world_size", lambda: tp_size
        )


def test_gdn_linear_all_branches(monkeypatch) -> None:
    _patch_tp(monkeypatch, tp_size=2, tp_rank=1)
    geometry = derive_gdn_head_geometry(num_kq_heads=1, num_v_heads=4, parallel_size=2)
    col_layer = gdn_linear.GdnColumnParallelLinear(
        input_size=16,
        geometry=geometry,
        d_k=8,
        d_v=8,
        include_z=True,
        quant_config=None,
        params_dtype=torch.float32,
        prefix="test_gdn_col",
    )
    assert col_layer.checkpoint_sizes == [8, 8, 32, 32]
    assert col_layer.replication_factors == [2, 2, 1, 1]

    pt_scale = PerTensorScaleParameter(
        data=torch.zeros(4, dtype=torch.float32),
        weight_loader=col_layer.weight_loader,
    )
    col_layer.weight_loader(pt_scale, torch.tensor(1.5), loaded_shard_id=0)
    torch.testing.assert_close(pt_scale.data[0], torch.tensor(1.5))

    full_ckpt = torch.arange(
        sum(col_layer.checkpoint_sizes) * 16, dtype=torch.float32
    ).reshape(sum(col_layer.checkpoint_sizes), 16)
    col_layer.weight_loader(col_layer.weight, full_ckpt, loaded_shard_id=None)

    qkv_ckpt = full_ckpt[: 8 + 8 + 32]
    z_ckpt = full_ckpt[8 + 8 + 32 :]
    col_layer.weight_loader(col_layer.weight, qkv_ckpt, loaded_shard_id=(0, 1, 2))
    col_layer.weight_loader(col_layer.weight, z_ckpt, loaded_shard_id=3)

    col_layer.weight_block_size = [8, 8]
    blk_scale = BlockQuantScaleParameter(
        data=torch.zeros(
            sum(col_layer.output_partition_sizes) // 8, 2, dtype=torch.float32
        ),
        output_dim=0,
        input_dim=1,
        weight_loader=col_layer.weight_loader,
    )
    assert col_layer._parameter_slice(blk_scale, 16, 8) == (2, 1)

    inter_layer = gdn_linear.GdnInterleavedColumnParallelLinear(
        geometry=geometry,
        group_sizes=[8, 8, 32, 32],
        replicated_segments=[True, True, False, False],
        input_size=16,
        quant_config=None,
        prefix="test_gdn_interleaved",
    )
    assert inter_layer.local_group_sizes == [8, 8, 16, 16]

    pt_scale_i = PerTensorScaleParameter(
        data=torch.zeros(1, dtype=torch.float32),
        weight_loader=inter_layer.weight_loader,
    )
    inter_layer.weight_loader(pt_scale_i, torch.tensor(2.5), loaded_shard_id=0)
    torch.testing.assert_close(pt_scale_i.data[0], torch.tensor(2.5))

    ckpt_rows = sum(inter_layer.group_sizes) * geometry.num_kq_heads
    ckpt = torch.arange(ckpt_rows * 16, dtype=torch.float32).reshape(ckpt_rows, 16)
    inter_layer.weight.data = inter_layer.weight.data.to(torch.float32)
    inter_layer.weight_loader(inter_layer.weight, ckpt)
    assert inter_layer.weight.shape == (sum(inter_layer.local_group_sizes), 16)

    inter_layer.weight_block_size = [8, 8]
    blk_scale_i = BlockQuantScaleParameter(
        data=torch.zeros(
            sum(inter_layer.local_group_sizes) // 8, 2, dtype=torch.float32
        ),
        output_dim=0,
        input_dim=1,
        weight_loader=inter_layer.weight_loader,
    )
    blk_ckpt = torch.arange((ckpt_rows // 8) * 2, dtype=torch.float32).reshape(
        ckpt_rows // 8, 2
    )
    inter_layer.weight_loader(blk_scale_i, blk_ckpt)

    inter_layer.weight_block_size = [16, 8]
    with pytest.raises(ValueError, match="GDN interleaved scale blocks must align"):
        inter_layer.weight_loader(blk_scale_i, blk_ckpt)

    ch_scale = ChannelQuantScaleParameter(
        data=torch.zeros(sum(inter_layer.local_group_sizes), 1, dtype=torch.float32),
        output_dim=0,
        weight_loader=inter_layer.weight_loader,
    )
    ch_ckpt_1d = torch.arange(ckpt_rows, dtype=torch.float32)
    inter_layer.weight_loader(ch_scale, ch_ckpt_1d)
    assert ch_scale.shape == (sum(inter_layer.local_group_sizes), 1)


# ---------------------------------------------------------------------------
# 4. compressed_tensors_moe_w4an_mxfp4.py
# ---------------------------------------------------------------------------


class _FakeRoutedExperts(RoutedExperts):
    def __init__(self, experts_per_token: int = 2, use_ep: bool = False):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock(w13_num_shards=2)
        self.moe_config.experts_per_token = experts_per_token
        self.moe_config.moe_parallel_config = MagicMock()
        self.moe_config.moe_parallel_config.use_ep = use_ep
        self.moe_config.norm_topk_prob = True
        self.moe_config.has_bias = False
        self.activation = "silu"
        self.use_grouped_topk = False
        self.custom_routing_function = None
        self.scoring_func = "softmax"
        self.e_score_correction_bias = None
        self.routed_scaling_factor = 1.0
        self.renormalize = True

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id


def _make_moe_config(
    *,
    intermediate: int = 64,
    unpadded: int = 64,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_rank: int = 0,
) -> MagicMock:
    moe_config = MagicMock(w13_num_shards=2)
    moe_config.tp_size = tp_size
    moe_config.tp_rank = tp_rank
    moe_config.ep_rank = ep_rank
    moe_config.experts_per_token = 2
    moe_config.intermediate_size_per_partition = intermediate
    moe_config.intermediate_size_per_partition_unpadded = unpadded
    return moe_config


def test_w4an_mxfp4_helpers_and_all_branches(monkeypatch) -> None:
    # 1. _fresh and _to_kernel_scale
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    t_fresh = _fresh(t)
    torch.testing.assert_close(t_fresh, t)
    s_4d = _to_kernel_scale(t)
    assert s_4d.shape == (2, 4, 1, 3)

    # 2. Properties: is_monolithic and supports_internal_mk
    moe_config = _make_moe_config()
    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    assert method.is_monolithic is True
    monkeypatch.setattr(
        mxfp4_mod, "enable_pipelined_collective_and_compute", lambda: True
    )
    assert method.supports_internal_mk is True
    monkeypatch.setattr(
        mxfp4_mod, "enable_pipelined_collective_and_compute", lambda: False
    )
    assert method.supports_internal_mk is False

    # 3. _initialize_dummy_quantized_weights:
    # (a) vllm_config is None -> early return
    layer = _FakeRoutedExperts()
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(mxfp4_mod, "get_current_vllm_config_or_none", lambda: None)
    method._initialize_dummy_quantized_weights(layer)

    # (b) load_format != "dummy" -> early return
    cfg_non_dummy = SimpleNamespace(
        load_config=SimpleNamespace(load_format="safetensors")
    )
    monkeypatch.setattr(
        mxfp4_mod, "get_current_vllm_config_or_none", lambda: cfg_non_dummy
    )
    method._initialize_dummy_quantized_weights(layer)

    # (c) load_format == "dummy" -> initializes packed & scale parameters
    cfg_dummy = SimpleNamespace(
        load_config=SimpleNamespace(load_format=SimpleNamespace(value="dummy"))
    )
    monkeypatch.setattr(mxfp4_mod, "get_current_vllm_config_or_none", lambda: cfg_dummy)
    method._initialize_dummy_quantized_weights(layer)
    assert torch.all(layer.w13_weight_scale == 123)

    # 4. _neutralize_padded_scales:
    # (a) unpadded_size >= padded_size -> early return
    method._neutralize_padded_scales(layer)

    # (b) unpadded_size < padded_size and not divisible by group_size (32) -> ValueError
    method_bad_pad = VllmCompressedTensorsW4ANMxfp4MoEMethod(
        _make_moe_config(intermediate=64, unpadded=48)
    )
    with pytest.raises(ValueError, match="divisible by"):
        method_bad_pad._neutralize_padded_scales(layer)

    # (c) unpadded_size < padded_size and divisible by 32 -> zeroes padded slices
    method_pad = VllmCompressedTensorsW4ANMxfp4MoEMethod(
        _make_moe_config(intermediate=64, unpadded=32)
    )
    for p in (layer.w13_weight_scale, layer.w2_weight_scale):
        p._cpu_scratch = torch.nn.Parameter(
            torch.full_like(p, 0xFF, device="cpu"), requires_grad=False
        )
    method_pad._neutralize_padded_scales(layer)
    assert not layer.w13_weight_scale._cpu_scratch[:, 32:64, :].any()

    # 5. process_weights_after_loading:
    # Mock tensor .to("tpu"), load_kmajor_fp4, requant_load_kmajor_fp4, prebuild
    orig_to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *args, **kwargs: (
            self if (args and args[0] == "tpu") else orig_to(self, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(mxfp4_mod, "load_kmajor_fp4", lambda t: t.transpose(-1, -2))
    monkeypatch.setattr(
        mxfp4_mod,
        "requant_load_kmajor_fp4",
        lambda w, s, b: (w.transpose(-1, -2), s.movedim(-1, 1).unsqueeze(-2)),
    )
    monkeypatch.setattr(mxfp4_mod, "prebuild_fused_moe_kernel", lambda **kw: None)
    monkeypatch.setattr(mxfp4_mod, "release_memory_to_os", lambda: None)

    # (a) requant_block is not None, misaligned block size -> ValueError
    monkeypatch.setattr(mxfp4_mod.envs, "MOE_REQUANTIZE_BLOCK_SIZE", "48")
    with pytest.raises(ValueError, match="W4A8 requantization needs"):
        method.process_weights_after_loading(layer)

    # (b) requant_block is not None, aligned block size (64), hasattr(layer, "weight"),
    #     use_ep=True, and parameter with _cpu_scratch vs without _cpu_scratch
    monkeypatch.setattr(mxfp4_mod.envs, "MOE_REQUANTIZE_BLOCK_SIZE", "64")
    layer_ep = _FakeRoutedExperts(use_ep=True)
    layer_ep.weight = None
    method.create_weights(
        layer=layer_ep,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    # Give w13_weight_packed a _cpu_scratch, leave others without _cpu_scratch
    layer_ep.w13_weight_packed._cpu_scratch = torch.nn.Parameter(
        torch.zeros_like(layer_ep.w13_weight_packed), requires_grad=False
    )
    monkeypatch.setattr(
        mxfp4_mod.moe_routing, "validate_linear_ep_placement", lambda lyr: None
    )
    monkeypatch.setattr(
        mxfp4_mod.moe_routing,
        "register_experts_start_buffer",
        lambda lyr, device: setattr(lyr, "_experts_start", None),
    )
    method.process_weights_after_loading(layer_ep)
    assert layer_ep.weight is layer_ep.w13_weight

    # (c) requant_block is None, hasattr(layer, "weight")=False, use_ep=False,
    #     and simulate device.type == "tpu" branch for synchronize_tensors
    monkeypatch.setattr(mxfp4_mod.envs, "MOE_REQUANTIZE_BLOCK_SIZE", None)
    layer_no_ep = _FakeRoutedExperts(use_ep=False)
    method.create_weights(
        layer=layer_no_ep,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    synced = []
    monkeypatch.setattr(
        mxfp4_mod, "synchronize_tensors", lambda tensors: synced.extend(tensors)
    )

    class _FakeTpuParameter(torch.nn.Parameter):
        @property
        def device(self):
            return SimpleNamespace(type="tpu")

    orig_param = torch.nn.Parameter
    monkeypatch.setattr(
        mxfp4_mod.torch.nn,
        "Parameter",
        lambda data, requires_grad=False: _FakeTpuParameter(
            data, requires_grad=requires_grad
        ),
    )
    method.process_weights_after_loading(layer_no_ep)
    monkeypatch.setattr(mxfp4_mod.torch.nn, "Parameter", orig_param)
    assert len(synced) == 4

    # 6. apply_monolithic and apply_with_routing:
    # Cover both pipelined=False and pipelined=True, and matching vs mismatched dtypes
    x = torch.ones((4, 128), dtype=torch.bfloat16)
    router_logits = torch.ones((4, 2), dtype=torch.float32)

    monkeypatch.setattr(
        mxfp4_mod.moe_routing,
        "route",
        lambda lyr, hs, rl: (
            torch.ones((4, 2), dtype=torch.float32),
            torch.zeros((4, 2), dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(
        mxfp4_mod.token_padding,
        "zero_routing_weights_for_padding",
        lambda ids, w, is_local_tensor: (ids, w),
    )
    monkeypatch.setattr(mxfp4_mod, "fused_moe_gmm", lambda **kw: kw["hidden_states"])
    monkeypatch.setattr(
        mxfp4_mod, "pipelined_fused_moe_gmm", lambda **kw: kw["hidden_states"] + 1
    )

    # (a) apply_monolithic -> apply_with_routing with int64 ids & float32 weights
    out_mono = method.apply_monolithic(layer_no_ep, x, router_logits)
    torch.testing.assert_close(out_mono, x)

    # (b) apply_with_routing with matching int32 ids & bfloat16 weights, pipelined=True
    ids_i32 = torch.zeros((4, 2), dtype=torch.int32)
    w_bf16 = torch.ones((4, 2), dtype=torch.bfloat16)
    out_pipe = method.apply_with_routing(
        layer_no_ep, x, w_bf16, ids_i32, pipelined=True
    )
    torch.testing.assert_close(out_pipe, x + 1)
