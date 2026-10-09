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
"""Unit tests for TD7.44 hot-path optimizations and 100% coverage of
`fused_moe_ep.py` and `quantization/configs.py`.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import jax.numpy as jnp
import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.quantization import configs as qconfigs

_STUB_EP = 2


def _make_layer(experts=4, hidden=8, inter=16, **overrides):
    layer = SimpleNamespace(
        w13_weight=torch.zeros(experts, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(experts, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(experts, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(experts, 1, 1, hidden),
        global_num_experts=experts * _STUB_EP,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            experts_per_token=2,
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=False
            ),
        ),
    )
    for k, v in overrides.items():
        setattr(layer, k, v)
    return layer


def test_quantization_configs_full_coverage(monkeypatch):
    # 1. should_quantize_on_load: empty, cache miss, cache hit, non-match
    monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "")
    assert not qconfigs.should_quantize_on_load("model.layers.0.self_attn.q_proj")

    monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", ".self_attn.,shared_experts")
    # First call populates _QUANTIZE_ON_LOAD_CACHE
    assert qconfigs.should_quantize_on_load("model.layers.0.self_attn.q_proj")
    # Second call with identical prefixes hits _QUANTIZE_ON_LOAD_CACHE[0] == key
    assert qconfigs.should_quantize_on_load("model.layers.1.shared_experts.down_proj")
    assert not qconfigs.should_quantize_on_load("model.layers.1.mlp.gate_proj")

    # 2. VllmQuantConfig.set_configs, get_name, get_linear_config, get_moe_config, mesh
    fake_vllm_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True)
    )
    qconfigs.VllmQuantConfig.set_configs(fake_vllm_cfg)
    assert qconfigs.VllmQuantConfig.vllm_config is fake_vllm_cfg

    with pytest.raises(NotImplementedError, match="Subclasses must implement get_name"):
        qconfigs.VllmQuantConfig.get_name()

    qcfg = qconfigs.VllmQuantConfig()
    assert qcfg.mesh is None

    linear_layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(linear_layer)
    linear_layer.output_size = 64
    linear_layer.input_size = 128
    lin_cfg = qcfg.get_linear_config(linear_layer)
    assert isinstance(lin_cfg, qconfigs.VllmQuantLinearConfig)
    assert lin_cfg.output_sizes == [64]

    routed_layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed_layer)
    routed_layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False)
    )
    moe_cfg = qcfg.get_moe_config(routed_layer)
    assert moe_cfg is routed_layer.moe_config
    assert moe_cfg.moe_parallel_config.use_ep is True


def test_squeeze_channel_scale_and_supported():
    assert bridge._squeeze_channel_scale(None, 4, 8) is None
    assert bridge._squeeze_channel_scale(torch.ones(4, 2, 1, 8), 4, 8) is None
    assert bridge._squeeze_channel_scale(torch.ones(4, 1, 1, 16), 4, 8) is None
    valid = bridge._squeeze_channel_scale(torch.ones(4, 1, 1, 8), 4, 8)
    assert valid is not None and valid.shape == (4, 8)

    s4d = torch.ones(4, 1, 1, 8)
    assert bridge._squeeze_channel_scale_checked(s4d).shape == (4, 8)
    s_non_single = torch.ones(4, 2, 1, 8)
    assert bridge._squeeze_channel_scale_checked(s_non_single).shape == (4, 2, 1, 8)

    owner = SimpleNamespace()
    assert not bridge.fused_moe_ep_supported(owner)
    setattr(owner, bridge.FUSED_MOE_EP_OP_ATTR, "op")
    assert bridge.fused_moe_ep_supported(owner)


def test_build_op_all_variants_and_register_fake(monkeypatch):
    from vllm_torchtpu.kernels.fused_moe import v2

    monkeypatch.setattr(bridge, "_OPS", {})
    closures = []
    registered_fakes = []

    def fake_jax_op(name, fn, **kwargs):
        closures.append((name, fn, kwargs))
        op_mock = Mock()
        op_mock.register_fake.side_effect = lambda f: registered_fakes.append(f)
        return op_mock

    monkeypatch.setattr(bridge, "jax_op", fake_jax_op)
    kernel = Mock(side_effect=lambda x, *args, **kwargs: x)
    monkeypatch.setattr(v2, "fused_ep_moe_v2", kernel)

    op_plain = bridge._build_op(
        None, 2, True, "silu", None, False, has_score_bias=False, has_hash_ids=False
    )
    op_bias = bridge._build_op(
        None, 2, True, "silu", None, False, has_score_bias=True, has_hash_ids=False
    )
    op_hash = bridge._build_op(
        None, 2, True, "silu", None, False, has_score_bias=False, has_hash_ids=True
    )
    # Cache hit
    assert (
        bridge._build_op(
            None, 2, True, "silu", None, False, has_score_bias=False, has_hash_ids=True
        )
        is op_hash
    )
    assert op_plain is not op_bias and op_bias is not op_hash
    assert len(closures) == 3
    assert closures[0][0] == bridge.FUSED_MOE_EP_OP_NAME
    assert closures[1][0] == f"{bridge.FUSED_MOE_EP_OP_NAME}_1"
    assert closures[2][0] == f"{bridge.FUSED_MOE_EP_OP_NAME}_2"

    x_jax = jnp.ones((4, 8), dtype=jnp.bfloat16)
    base_args = (
        x_jax,
        None,
        None,
        None,
        None,
        jnp.zeros((4, 4)),
        jnp.zeros((1, 1), dtype=jnp.int32),
    )

    # 1. Invoke moe (plain)
    assert closures[0][1](*base_args) is x_jax
    # 2. Invoke moe_with_score_bias
    bias_jax = jnp.ones((4,), dtype=jnp.bfloat16)
    assert closures[1][1](*base_args, bias_jax) is x_jax
    assert kernel.call_args.kwargs["score_bias"].dtype == jnp.float32
    # 3. Invoke moe_with_hash_ids
    hash_ids_jax = jnp.zeros((4, 2), dtype=jnp.int16)
    assert closures[2][1](*base_args, hash_ids_jax) is x_jax
    assert kernel.call_args.kwargs["hash_topk_ids"].dtype == jnp.int32

    # Invoke _fake registered on op
    x_torch = torch.ones((4, 8), dtype=torch.bfloat16)
    fake_out = registered_fakes[0](x_torch)
    assert fake_out.shape == x_torch.shape and fake_out.dtype == x_torch.dtype


def test_fused_moe_ep_dispatch_branches_and_operands():
    seen = []
    owner = SimpleNamespace()
    setattr(
        owner, bridge.FUSED_MOE_EP_OP_ATTR, lambda *args: seen.append(args) or args[0]
    )

    x = torch.zeros(2, 8)
    w_fp8 = torch.zeros(4, 8, 16, dtype=torch.float8_e4m3fn)
    s_fp8 = torch.ones(4, 1, 1, 16)
    logits = torch.zeros(2, 8)
    bias = torch.ones(8, dtype=torch.float32)
    hash_ids = torch.zeros(2, 2, dtype=torch.int32)

    # 1. Plain FP8 (no score_bias, no hash_ids)
    assert bridge.fused_moe_ep(owner, x, w_fp8, w_fp8, s_fp8, s_fp8, logits) is x
    assert len(seen[-1]) == 7
    assert seen[-1][3].shape == (4, 16)

    # 2. With score_bias
    assert (
        bridge.fused_moe_ep(
            owner, x, w_fp8, w_fp8, s_fp8, s_fp8, logits, score_bias=bias
        )
        is x
    )
    assert len(seen[-1]) == 8 and seen[-1][7] is bias

    # 3. With hash_ids
    assert (
        bridge.fused_moe_ep(
            owner, x, w_fp8, w_fp8, s_fp8, s_fp8, logits, hash_ids=hash_ids
        )
        is x
    )
    assert len(seen[-1]) == 8 and seen[-1][7] is hash_ids

    # 4. FP4 scale preservation
    w_fp4 = torch.zeros(4, 8, 8, dtype=torch.float4_e2m1fn_x2)
    s_fp4 = torch.ones(4, 1, 16)
    assert bridge.fused_moe_ep(owner, x, w_fp4, w_fp4, s_fp4, s_fp4, logits) is x
    assert seen[-1][3] is s_fp4 and seen[-1][4] is s_fp4


def test_score_bias_and_hash_topk_ids_operands():
    # register_score_bias_buffer & score_bias_operand
    empty_layer = torch.nn.Module()
    assert bridge.register_score_bias_buffer(empty_layer) is None
    assert bridge.score_bias_operand(empty_layer) is None

    biased_layer = torch.nn.Module()
    biased_layer.e_score_correction_bias = torch.nn.Parameter(
        torch.tensor([1.0, 2.0], dtype=torch.bfloat16), requires_grad=False
    )
    staged = bridge.register_score_bias_buffer(biased_layer)
    assert staged is not None and staged.dtype == torch.float32
    assert bridge.score_bias_operand(biased_layer) is staged

    # hash_topk_ids_operand
    assert bridge.hash_topk_ids_operand(empty_layer, torch.tensor([0, 1])) is None

    hash_layer = SimpleNamespace(
        hash_indices_table=torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int32)
    )
    with pytest.raises(ValueError, match="selects its experts from input_ids"):
        bridge.hash_topk_ids_operand(hash_layer, None)

    # 1D int64 input_ids + int32 table (fast path: no casts/flatten needed)
    ids_1d_i64 = torch.tensor([2, 0], dtype=torch.int64)
    out_fast = bridge.hash_topk_ids_operand(hash_layer, ids_1d_i64)
    assert out_fast is not None and out_fast.dtype == torch.int32
    assert torch.equal(out_fast, torch.tensor([[1, 0], [0, 1]], dtype=torch.int32))

    # 2D int32 input_ids + int16 table (exercises .long(), .flatten(), and .to(int32))
    hash_layer_i16 = SimpleNamespace(
        hash_indices_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int16)
    )
    ids_2d_i32 = torch.tensor([[1], [0]], dtype=torch.int32)
    out_cast = bridge.hash_topk_ids_operand(hash_layer_i16, ids_2d_i32)
    assert out_cast is not None and out_cast.dtype == torch.int32
    assert torch.equal(out_cast, torch.tensor([[2, 3], [0, 1]], dtype=torch.int32))


def test_mesh_expert_order_and_unsupported_reasons():
    with patch.object(bridge, "ep_rank_order", return_value=None):
        assert bridge._mesh_expert_order(2) is None
    with patch.object(bridge, "ep_rank_order", return_value=(0, 1)):
        assert bridge._mesh_expert_order(2) is None
    with patch.object(bridge, "ep_rank_order", return_value=(1, 0)):
        assert bridge._mesh_expert_order(2) == (1, 0)
    with (
        patch.object(bridge, "ep_rank_order", return_value=(0, 0)),
        pytest.raises(ValueError, match="EP rank order must permute"),
    ):
        bridge._mesh_expert_order(2)

    layer = _make_layer()
    # Rank/expert mismatch
    bad_weights = (
        torch.zeros(4, 8),
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
    )
    assert "weight rank/expert mismatch" in bridge.fused_moe_ep_unsupported_reason(
        layer, bad_weights, "silu"
    )

    # w2 shape mismatch
    bad_w2_weights = (
        layer.w13_weight,
        torch.zeros(4, 99, 8, dtype=torch.float8_e4m3fn),
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
    )
    assert "is not [E," in bridge.fused_moe_ep_unsupported_reason(
        layer, bad_w2_weights, "silu"
    )

    # Unsupported activation
    valid_weights = (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
    )
    assert "not wired to the kernel's ACT_FNS" in (
        bridge.fused_moe_ep_unsupported_reason(layer, valid_weights, "gelu")
    )

    # FP4 validation branches
    fp4_bad_dtype_weights = (
        torch.zeros(4, 64, 64, dtype=torch.float8_e4m3fn),
        torch.zeros(4, 64, 32, dtype=torch.float8_e4m3fn),
        torch.ones(4, 1, 1, 128),
        torch.ones(4, 1, 1, 64),
    )
    assert (
        bridge.fused_moe_ep_unsupported_reason(
            layer, fp4_bad_dtype_weights, "silu", weight_format="fp4", rhs_qb=64
        )
        == "the FP4 form requires native float4_e2m1fn_x2 weights"
    )

    fp4_valid_w = (
        torch.zeros(4, 64, 64, dtype=torch.float4_e2m1fn_x2),
        torch.zeros(4, 64, 32, dtype=torch.float4_e2m1fn_x2),
        torch.ones(4, 1, 1, 128),
        torch.ones(4, 1, 1, 64),
    )
    assert "positive integer block size" in bridge.fused_moe_ep_unsupported_reason(
        layer, fp4_valid_w, "silu", weight_format="fp4", rhs_qb=0
    )
    assert "multiple of the packed-weight row tile" in (
        bridge.fused_moe_ep_unsupported_reason(
            layer, fp4_valid_w, "silu", weight_format="fp4", rhs_qb=32
        )
    )
    assert "must divide both" in bridge.fused_moe_ep_unsupported_reason(
        layer, fp4_valid_w, "silu", weight_format="fp4", rhs_qb=128
    )
    fp4_bad_scale_w = (
        fp4_valid_w[0],
        fp4_valid_w[1],
        torch.ones(4, 2, 1, 128),
        fp4_valid_w[3],
    )
    assert "K-major scale layout" in bridge.fused_moe_ep_unsupported_reason(
        layer, fp4_bad_scale_w, "silu", weight_format="fp4", rhs_qb=64
    )
    assert (
        bridge.fused_moe_ep_unsupported_reason(
            layer, fp4_valid_w, "silu", weight_format="fp4", rhs_qb=64
        )
        is None
    )

    # FP8 validation branches
    fp8_bad_dtype = (
        torch.zeros(4, 8, 32, dtype=torch.bfloat16),
        torch.zeros(4, 16, 8, dtype=torch.bfloat16),
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
    )
    assert "float8_e4m3fn expert weights" in bridge.fused_moe_ep_unsupported_reason(
        layer, fp8_bad_dtype, "silu"
    )
    fp8_bad_s1 = (
        layer.w13_weight,
        layer.w2_weight,
        torch.ones(4, 2, 1, 32),
        layer.w2_weight_scale_inv,
    )
    assert "w1_scale is not" in bridge.fused_moe_ep_unsupported_reason(
        layer, fp8_bad_s1, "silu"
    )
    fp8_bad_s2 = (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        torch.ones(4, 2, 1, 8),
    )
    assert "w2_scale is not" in bridge.fused_moe_ep_unsupported_reason(
        layer, fp8_bad_s2, "silu"
    )

    # Unsupported weight_format
    assert "unsupported fused EP weight format" in (
        bridge.fused_moe_ep_unsupported_reason(
            layer, valid_weights, "silu", weight_format="int8"
        )
    )

    # Expert biases refused
    assert "expert biases" in bridge.fused_moe_ep_unsupported_reason(
        _make_layer(w13_bias=torch.zeros(4, 32)), valid_weights, "silu"
    )

    # Routing unsupported reasons
    assert "custom_routing_function" in bridge._unsupported_routing_reason(
        _make_layer(custom_routing_function=lambda *a: None)
    )
    assert "scoring_func" in bridge._unsupported_routing_reason(
        _make_layer(scoring_func="unknown_fn")
    )
    assert "e_score_correction_bias" in bridge._unsupported_routing_reason(
        _make_layer(e_score_correction_bias=torch.zeros(3))
    )
    assert (
        bridge._unsupported_routing_reason(
            _make_layer(e_score_correction_bias=torch.zeros(4 * _STUB_EP))
        )
        is None
    )
    assert "grouped top-k" in bridge._unsupported_routing_reason(
        _make_layer(use_grouped_topk=True, num_expert_group=4, topk_group=2)
    )
    assert "does not supply input_ids" in bridge._unsupported_routing_reason(
        _make_layer(hash_indices_table=torch.zeros(16, 2, dtype=torch.int32)),
        supplies_hash_ids=False,
    )
    assert "hash routing with an e_score_correction_bias" in (
        bridge._unsupported_routing_reason(
            _make_layer(
                e_score_correction_bias=torch.zeros(4 * _STUB_EP),
                hash_indices_table=torch.zeros(16, 2, dtype=torch.int32),
            ),
            supplies_hash_ids=True,
        )
    )
    assert "hash_indices_table" in bridge._unsupported_routing_reason(
        _make_layer(hash_indices_table=torch.zeros(16, 3, dtype=torch.int32)),
        supplies_hash_ids=True,
    )
    assert (
        bridge._unsupported_routing_reason(
            _make_layer(hash_indices_table=torch.zeros(16, 2, dtype=torch.int32)),
            supplies_hash_ids=True,
        )
        is None
    )
    with patch.object(moe_routing, "_SIMULATION_STRATEGY", "uniform"):
        assert "VLLM_MOE_ROUTING_SIMULATION_STRATEGY" in (
            bridge._unsupported_routing_reason(layer)
        )


def test_prebuild_refusals_and_env_helpers():
    layer = _make_layer()
    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: _STUB_EP})

    def _run_prebuild(target_layer=layer, **overrides):
        cfg = dict(
            use_kernel=True,
            w4a8=True,
            min_tokens=1024,
            skip_padded=False,
            chunk_size=0,
            mesh_val=mesh,
            node_tokens=4096,
            smem_bytes=1024 * 1024,
            weight_format="fp8",
            rhs_qb=None,
            weights=None,
        )
        cfg.update(overrides)
        with (
            patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", cfg["use_kernel"]),
            patch.object(envs, "MOE_FUSED_EP_ENABLE_W4A8", cfg["w4a8"]),
            patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", cfg["min_tokens"]),
            patch.object(envs, "TPU_MOE_SKIP_PADDED_TOKENS", cfg["skip_padded"]),
            patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", cfg["chunk_size"]),
            patch.object(bridge, "build_ep_mesh", return_value=cfg["mesh_val"]),
            patch.object(bridge, "ep_mesh_index", return_value=0),
            patch.object(bridge, "ep_rank_order", return_value=None),
            patch.object(bridge, "ep_token_replica_groups", return_value=((0,), (1,))),
            patch.object(bridge, "_max_node_tokens", return_value=cfg["node_tokens"]),
            patch.object(
                bridge, "_smem_capacity_bytes", return_value=cfg["smem_bytes"]
            ),
            patch.object(bridge, "_build_op", return_value="op"),
        ):
            return bridge.prebuild_fused_moe_ep(
                target_layer,
                2,
                True,
                "silu",
                weight_format=cfg["weight_format"],
                rhs_qb=cfg["rhs_qb"],
                weights=cfg["weights"],
            )

    assert _run_prebuild(use_kernel=False) is None
    assert _run_prebuild(weight_format="fp4", w4a8=False) is None

    no_ep_layer = _make_layer()
    no_ep_layer.moe_config.moe_parallel_config.use_ep = False
    assert _run_prebuild(target_layer=no_ep_layer) is None

    assert _run_prebuild(mesh_val=None) is None
    assert _run_prebuild(smem_bytes=1024) is None
    assert _run_prebuild(node_tokens=16) is None
    assert _run_prebuild(target_layer=_make_layer(global_num_experts=99)) is None
    assert _run_prebuild(target_layer=_make_layer(w13_bias=torch.zeros(4, 32))) is None

    # Normal arming (both without and with skip_padded warning)
    assert _run_prebuild(skip_padded=False) == "op"
    assert _run_prebuild(skip_padded=True) == "op"

    # Explicit weights provided and fp4 default weights read from layer
    fp4_layer = _make_layer(experts=4, hidden=64, inter=64)
    fp4_layer.w13_weight = torch.zeros(4, 64, 64, dtype=torch.float4_e2m1fn_x2)
    fp4_layer.w2_weight = torch.zeros(4, 64, 32, dtype=torch.float4_e2m1fn_x2)
    fp4_layer.w13_weight_scale = torch.ones(4, 1, 1, 128)
    fp4_layer.w2_weight_scale = torch.ones(4, 1, 1, 64)
    assert _run_prebuild(target_layer=fp4_layer, weight_format="fp4", rhs_qb=64) == "op"
    assert (
        _run_prebuild(
            target_layer=fp4_layer,
            weight_format="fp4",
            rhs_qb=64,
            weights=(
                fp4_layer.w13_weight,
                fp4_layer.w2_weight,
                fp4_layer.w13_weight_scale,
                fp4_layer.w2_weight_scale,
            ),
        )
        == "op"
    )

    # _max_node_tokens: exception, non-int cap, and valid int cap
    with patch(
        "vllm.config.get_current_vllm_config", side_effect=RuntimeError("no cfg")
    ):
        assert bridge._max_node_tokens(ep=8, pcp=1) == 0

    bad_cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=None)
    )
    with patch("vllm.config.get_current_vllm_config", return_value=bad_cfg):
        assert bridge._max_node_tokens(ep=8, pcp=1) == 0

    good_cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512)
    )
    with patch("vllm.config.get_current_vllm_config", return_value=good_cfg):
        assert bridge._max_node_tokens(ep=8, pcp=2) == 2048

    # _smem_capacity_bytes both success and exception fallback
    from jax.experimental.pallas import tpu as pltpu

    with patch.object(
        pltpu,
        "get_tpu_info",
        return_value=SimpleNamespace(smem_capacity_bytes=512 * 1024),
    ):
        assert bridge._smem_capacity_bytes() == 512 * 1024

    with patch.object(pltpu, "get_tpu_info", side_effect=RuntimeError("no tpu")):
        assert bridge._smem_capacity_bytes() == 1024 * 1024
