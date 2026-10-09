# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""100% statement and branch coverage tests for TD8.56 Kimi-K3 attention,
collective_ops, kimi_vit, and model modules."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest  # type: ignore[missing-import]
import torch
from torch import nn
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)

from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.kimi_k3 import (
    attention as k3_attn,
)
from vllm_torchtpu.models.vllm.kimi_k3 import (
    collective_ops as k3_cops,
)
from vllm_torchtpu.models.vllm.kimi_k3 import (
    kimi_vit as k3_vit,
)
from vllm_torchtpu.models.vllm.kimi_k3 import (
    model as k3_model,
)

pytestmark = pytest.mark.cpu_test


class _FakeTpuTensor(torch.Tensor):
    """CPU tensor subclass reporting device.type == 'tpu' for TPU branch tests."""

    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


def _make_fake_tpu_tensor(t: torch.Tensor) -> _FakeTpuTensor:
    return torch.Tensor._make_subclass(_FakeTpuTensor, t)


class _DummyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int | list[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(out_features, list):
            out_features = sum(out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.ones(in_features, out_features, dtype=torch.bfloat16)
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out_shape = (*x.shape[:-1], self.out_features)
        return torch.ones(out_shape, dtype=x.dtype, device=x.device), None


# ---------------------------------------------------------------------------
# 1. kimi_vit.py tests
# ---------------------------------------------------------------------------


def test_kimi_vit_materialize_and_model_init(monkeypatch):
    cpu_qkv = torch.ones(2, 4)
    # 1. Non-TPU tensor returns None
    assert k3_vit._materialize_vision_qkv(nn.Identity(), (), (cpu_qkv, None)) is None

    # 2. TPU tensor while compiling returns None
    tpu_qkv = _make_fake_tpu_tensor(torch.ones(2, 4))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert k3_vit._materialize_vision_qkv(nn.Identity(), (), (tpu_qkv, None)) is None

    # 3. TPU tensor in eager mode materializes and synchronizes
    synced: list[torch.Tensor] = []
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(k3_vit, "synchronize_tensors", lambda *ts: synced.extend(ts))
    res = k3_vit._materialize_vision_qkv(nn.Identity(), (), (tpu_qkv, cpu_qkv))
    assert res is not None
    assert res[0].shape == (2, 4)
    assert res[1] is cpu_qkv
    assert len(synced) == 1

    # 4. KimiK3MoonViT3dPretrainedModel.__init__ with blocks and empty blocks
    def _fake_super_init(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        blocks = kwargs.get("blocks", [SimpleNamespace(wqkv=nn.Linear(2, 2))])
        self.encoder = SimpleNamespace(blocks=blocks)

    monkeypatch.setattr(k3_vit.MoonViT3dPretrainedModel, "__init__", _fake_super_init)
    vit_with_blocks = k3_vit.KimiK3MoonViT3dPretrainedModel()
    assert len(vit_with_blocks.encoder.blocks[0].wqkv._forward_hooks) == 1
    vit_empty = k3_vit.KimiK3MoonViT3dPretrainedModel(blocks=[])
    assert vit_empty.encoder.blocks == []


# ---------------------------------------------------------------------------
# 2. collective_ops.py tests
# ---------------------------------------------------------------------------


def test_collective_ops_builders_and_wrappers(monkeypatch):
    k3_cops._OPS.clear()
    k3_cops._mesh_and_tables.cache_clear()
    k3_cops._build_token_op.cache_clear()

    registered_ops: list[tuple[str, Any, Any]] = []

    def _fake_sharded_jax_op(name, fn, **kwargs):
        class _Op:
            def __init__(self):
                self.name = name
                self.fn = fn
                self.fake_fn = None

            def register_fake(self, f):
                self.fake_fn = f

            def __call__(self, *args):
                assert self.fake_fn is not None
                return self.fake_fn(*args)

        op = _Op()
        registered_ops.append((name, fn, op))
        return op

    monkeypatch.setattr(k3_cops, "sharded_jax_op", _fake_sharded_jax_op)
    monkeypatch.setattr(
        k3_cops,
        "mla_chip_layout",
        lambda devices, owners: (((0, 1),), (), ()),
    )
    monkeypatch.setattr(k3_cops, "pack_mla_gate", lambda w, m, p: w)
    monkeypatch.setattr(
        k3_cops,
        "mla_preprocess",
        lambda x, w, pg, qn, kvn, m, o, *, sequence_parallel, eps: (x, x, x, x),
    )
    monkeypatch.setattr(
        k3_cops,
        "gather_moe_inputs",
        lambda x, w, i, m, t, *, collective_id: (x, w, i),
    )
    monkeypatch.setattr(
        k3_cops,
        "ring_collective",
        lambda x, m, t, *, gather, collective_id: x,
    )
    monkeypatch.setattr(
        k3_cops,
        "project_reduce_scatter",
        lambda x, w, m, t, *, collective_id: x[0],
    )
    monkeypatch.setattr(
        k3_cops,
        "gather_project",
        lambda x, w, m, t, *, collective_id: jnp.zeros((1, 4, 1792)),
    )

    fake_devices = np.array([SimpleNamespace(id=i) for i in range(32)])
    mesh = SimpleNamespace(devices=fake_devices)
    owners = tuple(range(32))
    tables = np.zeros((3, 3, 32), dtype=np.int32)
    tables[:, 2, :] = np.arange(32, dtype=np.int32)

    # 1. _build_mla_ops (miss + hit + inner functions + fake functions)
    pack_op, pre_op = k3_cops._build_mla_ops(mesh, owners, 0, 1e-6)
    assert k3_cops._build_mla_ops(mesh, owners, 0, 1e-6) == (pack_op, pre_op)
    w_t = torch.empty((7168, 2496), dtype=torch.bfloat16)
    assert pack_op(w_t).shape == (7168, 6144)
    assert pack_op.fn(jnp.zeros((2, 2))).shape == (2, 2)
    x_t = torch.empty((4, 7168), dtype=torch.bfloat16)
    out_sp = pre_op(x_t, w_t, w_t, w_t, w_t, True)
    assert out_sp[0].shape == (128, 1536)
    out_nsp = pre_op(x_t, w_t, w_t, w_t, w_t, False)
    assert out_nsp[0].shape == (4, 1536)
    assert len(pre_op.fn(jnp.zeros((2, 2)), None, None, None, None, True)) == 4

    # 2. _build_moe_gather_op (miss + hit + inner + fake)
    moe_op = k3_cops._build_moe_gather_op(mesh, tables, 10)
    assert k3_cops._build_moe_gather_op(mesh, tables, 10) is moe_op
    gx, gw, gi = moe_op(torch.empty((2, 16)), torch.empty((2, 4)), torch.empty((2, 4)))
    assert gx.shape == (64, 16)
    assert len(moe_op.fn(jnp.zeros((2, 2)), jnp.zeros((2, 2)), jnp.zeros((2, 2)))) == 3

    # 3. _build_op (gather=True and gather=False, miss + hit + inner + fake)
    ag_op = k3_cops._build_op(mesh, tables, True, 20)
    assert k3_cops._build_op(mesh, tables, True, 20) is ag_op
    assert ag_op(torch.empty((4, 16))).shape == (128, 16)
    assert ag_op.fn(jnp.zeros((4, 16))).shape == (4, 16)

    rs_op = k3_cops._build_op(mesh, tables, False, 21)
    assert k3_cops._build_op(mesh, tables, False, 21) is rs_op
    assert rs_op(torch.empty((64, 16))).shape == (2, 16)
    assert rs_op.fn(jnp.zeros((64, 16))).shape == (32, 2, 16)

    # 4. _build_project_op (miss + hit + inner + fake)
    proj_op = k3_cops._build_project_op(mesh, tables, 30)
    assert k3_cops._build_project_op(mesh, tables, 30) is proj_op
    assert proj_op(torch.empty((64, 384)), torch.empty((384, 7168))).shape == (2, 7168)
    assert proj_op.fn(jnp.zeros((64, 384)), jnp.zeros((32 * 384, 7168))).shape == (
        2,
        384,
    )

    # 5. _build_gather_project_op (miss + hit + inner + fake)
    gp_op = k3_cops._build_gather_project_op(mesh, tables, 40)
    assert k3_cops._build_gather_project_op(mesh, tables, 40) is gp_op
    assert gp_op(torch.empty((4, 7168)), torch.empty((7168, 1792))).shape == (
        128,
        1792,
    )
    assert gp_op.fn(jnp.zeros((4, 7168)), jnp.zeros((32 * 7168, 1792))).shape == (
        4,
        1792,
    )

    # 6. _mesh_and_tables validation + success
    base_group = SimpleNamespace(
        ranks=list(range(32)),
        world_size=2,
        rank_in_group=0,
        all_gather=lambda x, dim=0: x.repeat(2, 1),
        reduce_scatter=lambda x, dim=0: x[: x.shape[0] // 2],
        all_reduce=lambda x: x,
    )
    monkeypatch.setattr(k3_cops, "get_tp_group", lambda: base_group)
    monkeypatch.setattr(k3_cops, "get_ep_group", lambda: None)
    with pytest.raises(ValueError, match="identical TP32 and EP32 rank order"):
        k3_cops._mesh_and_tables()

    k3_cops._mesh_and_tables.cache_clear()
    monkeypatch.setattr(
        k3_cops, "get_ep_group", lambda: SimpleNamespace(ranks=list(range(32)))
    )
    monkeypatch.setattr(k3_cops, "build_ep_mesh", lambda axis_name: mesh)
    monkeypatch.setattr(k3_cops, "ep_rank_order", lambda: owners)
    monkeypatch.setattr(k3_cops, "ring_tables", lambda devs, own: tables)
    m_out, t_out = k3_cops._mesh_and_tables()
    assert m_out is mesh
    assert t_out is tables

    # 7. _build_token_op (world_size=2, 32, invalid) + TokenShardCollectives
    monkeypatch.setattr(k3_cops.jax, "shard_map", lambda fn, **kw: lambda x: fn(x))
    monkeypatch.setattr(k3_cops.jax.lax, "all_gather", lambda part, *a, **kw: part)
    monkeypatch.setattr(k3_cops.jax.lax, "psum_scatter", lambda part, *a, **kw: part)
    with pytest.raises(ValueError, match="require TP2 or TP32"):
        k3_cops._build_token_op(4, True)

    tok_32_ag = k3_cops._build_token_op(32, True)
    assert tok_32_ag(torch.empty((2, 8))).shape == (64, 8)
    assert tok_32_ag.fn(jnp.zeros((2, 8))).shape == (2, 8)

    tsc = k3_cops.TokenShardCollectives(base_group)
    assert tsc.all_gather(torch.empty((4, 8)), dim=0).shape == (8, 8)
    assert tsc.reduce_scatter(torch.empty((4, 8)), dim=0).shape == (2, 8)
    assert tsc.ag.fn(jnp.zeros((2, 8))).shape == (2, 8)
    assert tsc.rs.fn(jnp.zeros((2, 8))).shape == (2, 8)

    # 8. FusedPrefillCollectives
    fpc = k3_cops.FusedPrefillCollectives("model.layers.3.self_attn")
    assert fpc.layer == 3
    assert fpc.mla_ops(1e-6)[0].name == "pallas::kimi_mla_pack_gate_3"

    # gather_moe_inputs: matching dtypes vs mismatched dtypes,
    # num_experts <= 32768 vs > 32768
    lat_bf16 = torch.ones((2, 7168), dtype=torch.bfloat16)
    w_bf16 = torch.ones((2, 8), dtype=torch.bfloat16)
    ids_i32 = torch.zeros((2, 8), dtype=torch.int32)
    g1 = fpc.gather_moe_inputs(lat_bf16, w_bf16, ids_i32, num_experts=256)
    assert g1[0].shape == (64, 7168)

    w_fp32 = torch.ones((2, 8), dtype=torch.float32)
    ids_i64 = torch.zeros((2, 8), dtype=torch.int64)
    g2 = fpc.gather_moe_inputs(lat_bf16, w_fp32, ids_i64, num_experts=65536)
    assert g2[0].shape == (64, 7168)

    # all_gather & reduce_scatter fast path vs fallback
    assert fpc.all_gather(lat_bf16, dim=0).shape == (64, 7168)
    assert fpc.all_gather(torch.ones((2, 16)), dim=0).shape == (4, 16)

    rs_in = torch.ones((64, 7168), dtype=torch.bfloat16)
    assert fpc.reduce_scatter(rs_in, dim=0).shape == (2, 7168)
    assert fpc.reduce_scatter(torch.ones((4, 16)), dim=0).shape == (2, 16)

    # project_reduce_scatter fast path vs fallback
    proj_good = SimpleNamespace(
        weight=torch.ones((384, 7168), dtype=torch.bfloat16),
        bias=None,
        **{WEIGHT_FLIPPED_ATTR: True},
    )
    x_proj = torch.ones((64, 384), dtype=torch.bfloat16)
    assert fpc.project_reduce_scatter(x_proj, proj_good).shape == (2, 7168)

    def proj_fallback(x):
        return (torch.ones((64, 7168), dtype=torch.bfloat16), None)

    assert fpc.project_reduce_scatter(x_proj, proj_fallback).shape == (2, 7168)
    assert fpc.all_reduce(x_proj) is x_proj


# ---------------------------------------------------------------------------
# 3. attention.py tests
# ---------------------------------------------------------------------------


def test_attention_helpers_mixed_linear_and_conv1d(monkeypatch):
    monkeypatch.setattr(
        k3_attn.MambaStateDtypeCalculator,
        "kda_state_dtype",
        lambda dt, cdt: (torch.float16, torch.bfloat16),
    )
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
    )
    assert k3_attn.kda_state_dtype(vllm_cfg) == (torch.float32, torch.bfloat16)

    monkeypatch.setattr(k3_attn, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(k3_attn, "get_tensor_model_parallel_world_size", lambda: 2)

    # _load_a_log: 4D and 1D weights
    param_a = torch.zeros(2)
    k3_attn._load_a_log(param_a, torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]]))
    assert torch.equal(param_a, torch.tensor([3.0, 4.0]))
    k3_attn._load_a_log(param_a, torch.tensor([10.0, 20.0, 30.0, 40.0]))
    assert torch.equal(param_a, torch.tensor([30.0, 40.0]))

    # MixedParallelMergedLinear validation & weight loading
    def _fake_merged_init(
        self, input_size, output_sizes, *, bias, quant_config, prefix, disable_tp=False
    ):
        nn.Module.__init__(self)
        self.output_sizes = output_sizes

    monkeypatch.setattr(
        k3_attn.MergedColumnParallelLinear, "__init__", _fake_merged_init
    )
    recorded_loads: list[tuple[Any, torch.Tensor, Any]] = []
    monkeypatch.setattr(
        k3_attn.MergedColumnParallelLinear,
        "weight_loader",
        lambda self, p, w, sid=None: recorded_loads.append((p, w, sid)),
    )
    monkeypatch.setattr(
        k3_attn.MergedColumnParallelLinear,
        "weight_loader_v2",
        lambda self, p, w, sid=None: recorded_loads.append((p, w, sid)),
    )

    with pytest.raises(ValueError, match="Every fused output needs a sharding mode"):
        k3_attn.MixedParallelMergedLinear(
            16, [8, 8], [True], bias=False, quant_config=None, prefix="p"
        )
    with pytest.raises(
        ValueError, match="Column-parallel fused outputs must divide TP"
    ):
        k3_attn.MixedParallelMergedLinear(
            16, [8, 7], [True, False], bias=False, quant_config=None, prefix="p"
        )

    mpml = k3_attn.MixedParallelMergedLinear(
        16, [8, 10], [True, False], bias=False, quant_config=None, prefix="p"
    )
    assert mpml.output_sizes == [8, 5]

    p_no_dim = nn.Parameter(torch.zeros(5, 16))
    # Replicated shard (shard_id=0) -> unchanged
    mpml.weight_loader(p_no_dim, torch.ones(8, 16), 0)
    # Sharded shard (shard_id=1) but no output_dim -> unchanged
    mpml.weight_loader_v2(p_no_dim, torch.ones(10, 16), 1)

    p_with_dim = nn.Parameter(torch.zeros(5, 16))
    p_with_dim.output_dim = 0  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="does not divide TP"):
        mpml.weight_loader(p_with_dim, torch.ones(9, 16), 1)

    w_full = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    mpml.weight_loader(p_with_dim, w_full, 1)
    assert torch.equal(recorded_loads[-1][1], w_full[5:10])

    # CausalDepthwiseConv1d
    with pytest.raises(
        ValueError, match="Convolution channels must be divisible by TP size"
    ):
        k3_attn.CausalDepthwiseConv1d(7, 4)

    conv = k3_attn.CausalDepthwiseConv1d(8, 3)
    assert conv.channels == 4
    w2d = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    conv.weight_loader(conv.weight, w2d)
    assert torch.equal(conv.weight, w2d[4:8].unsqueeze(1))
    w3d = torch.zeros(8, 1, 3)
    conv.weight_loader(conv.weight, w3d)
    assert torch.equal(conv.weight, w3d[4:8])


def test_mla_attention_all_branches(monkeypatch):
    monkeypatch.setattr(k3_attn, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(k3_attn, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(k3_attn, "ColumnParallelLinear", _DummyLinear)
    monkeypatch.setattr(k3_attn, "MergedColumnParallelLinear", _DummyLinear)
    monkeypatch.setattr(k3_attn, "MixedParallelMergedLinear", _DummyLinear)
    monkeypatch.setattr(k3_attn, "ReplicatedLinear", _DummyLinear)
    monkeypatch.setattr(k3_attn, "RowParallelLinear", _DummyLinear)
    monkeypatch.setattr(
        k3_attn,
        "RMSNorm",
        lambda dim, eps: SimpleNamespace(variance_epsilon=eps, weight=torch.ones(dim)),
    )
    monkeypatch.setattr(k3_attn, "get_rope", lambda *a, **kw: SimpleNamespace(kw=kw))

    class _FakeMLAWrapper(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.kimi_preprocess = None
            self.kimi_gate_weight = None

        def forward(self, positions, hidden_states, prefill_group=None):
            return hidden_states

    monkeypatch.setattr(k3_attn, "MultiHeadLatentAttentionWrapper", _FakeMLAWrapper)

    vllm_cfg = SimpleNamespace(
        quant_config=None,
        cache_config=None,
    )
    base_cfg = SimpleNamespace(
        mla_use_nope=False,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_attention_heads=6,
        hidden_size=7168,
        q_lora_rank=1536,
        mla_use_output_gate=True,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_parameters=None,
        rope_theta=10000.0,
    )

    # Validation errors
    with pytest.raises(ValueError, match="supports NoPE MLA only"):
        k3_attn.MultiHeadLatentAttention(base_cfg, vllm_cfg, "l0", use_rope=False)

    bad_cfg = SimpleNamespace(
        **{**base_cfg.__dict__, "mla_use_nope": True, "v_head_dim": None}
    )
    with pytest.raises(ValueError, match="Incomplete Kimi K3 MLA config"):
        k3_attn.MultiHeadLatentAttention(bad_cfg, vllm_cfg, "l0")

    bad_heads = SimpleNamespace(
        **{**base_cfg.__dict__, "mla_use_nope": True, "num_attention_heads": 5}
    )
    with pytest.raises(ValueError, match="divisible by TP size"):
        k3_attn.MultiHeadLatentAttention(bad_heads, vllm_cfg, "l0")

    # 1. q_lora_rank=None, mla_use_output_gate=True, use_rope=False
    monkeypatch.setattr(k3_attn.envs, "TPU_K3_SP_PREFILL", False)
    cfg_no_qlora = SimpleNamespace(
        **{**base_cfg.__dict__, "mla_use_nope": True, "q_lora_rank": None}
    )
    mla1 = k3_attn.MultiHeadLatentAttention(cfg_no_qlora, vllm_cfg, "l0")
    assert mla1.q_proj is not None
    assert mla1.g_proj is not None

    # 2. q_lora_rank=1536, mla_use_output_gate=False, use_rope=True
    # (default, yarn, llama_scaling)
    cfg_no_gate = SimpleNamespace(
        **{
            **base_cfg.__dict__,
            "mla_use_nope": True,
            "mla_use_output_gate": False,
            "rope_parameters": None,
        }
    )
    mla2 = k3_attn.MultiHeadLatentAttention(cfg_no_gate, vllm_cfg, "l0", use_rope=True)
    assert mla2.g_proj is None

    cfg_yarn = SimpleNamespace(
        **{
            **cfg_no_gate.__dict__,
            "rope_parameters": {"rope_type": "yarn", "apply_yarn_scaling": True},
        }
    )
    k3_attn.MultiHeadLatentAttention(cfg_yarn, vllm_cfg, "l0", use_rope=True)

    cfg_llama = SimpleNamespace(
        **{
            **cfg_no_gate.__dict__,
            "rope_parameters": {"rope_type": "yarn", "apply_yarn_scaling": False},
        }
    )
    k3_attn.MultiHeadLatentAttention(cfg_llama, vllm_cfg, "l0", use_rope=True)

    # 3. q_lora_rank=1536, mla_use_output_gate=True, TPU_K3_SP_PREFILL=True
    # + prepare_sp_mla_gate
    class _FakeFPC(k3_cops.FusedPrefillCollectives):
        def __init__(self, prefix):
            self.prefix = prefix

        def mla_ops(self, eps):
            return (lambda w: w, "pre_op")

        def all_gather(self, x, dim=0):
            return x

        def reduce_scatter(self, x, dim=0):
            return x

        def all_reduce(self, x):
            return x

    monkeypatch.setattr(k3_attn, "FusedPrefillCollectives", _FakeFPC)
    monkeypatch.setattr(k3_attn.envs, "TPU_K3_SP_PREFILL", True)
    cfg_fused = SimpleNamespace(**{**base_cfg.__dict__, "mla_use_nope": True})
    mla3 = k3_attn.MultiHeadLatentAttention(cfg_fused, vllm_cfg, "model.layers.0.attn")

    # Early return in prepare_sp_mla_gate when weight not flipped
    mla3.prepare_sp_mla_gate()
    assert mla3.mla_attn.kimi_preprocess is None

    # Set valid flipped weight of shape (7168, 2496)
    assert mla3.fused_qkv_a_proj is not None
    mla3.fused_qkv_a_proj.weight = nn.Parameter(
        torch.ones((7168, 2496), dtype=torch.bfloat16)
    )
    setattr(mla3.fused_qkv_a_proj, WEIGHT_FLIPPED_ATTR, True)
    mla3.kv_a_layernorm.variance_epsilon = 1e-5
    with pytest.raises(ValueError, match="matching norm epsilons"):
        mla3.prepare_sp_mla_gate()

    mla3.kv_a_layernorm.variance_epsilon = 1e-6
    mla3.prepare_sp_mla_gate()
    assert mla3.mla_attn.kimi_preprocess == "pre_op"
    assert mla3.mla_attn.kimi_gate_weight is not None

    # forward: sequence_parallel=False, sequence_parallel=True
    # (with FusedPrefillCollectives and non-FPC)
    tp_grp = SimpleNamespace(
        all_gather=lambda x, dim=0: x,
        reduce_scatter=lambda x, dim=0: x,
        all_reduce=lambda x: x,
    )
    monkeypatch.setattr(k3_attn, "get_tp_group", lambda: tp_grp)
    pos = torch.zeros(4, dtype=torch.int32)
    hs = torch.ones(4, 7168, dtype=torch.bfloat16)
    assert mla3(pos, hs, sequence_parallel=False).shape == (4, 7168)

    mla3.mla_attn.kimi_preprocess = None
    assert mla3(pos, hs, sequence_parallel=True).shape == (4, 7168)

    mla3.mla_attn.kimi_preprocess = "pre_op"
    mla3.prefill_group = tp_grp  # type: ignore[assignment]
    assert mla3(pos, hs, sequence_parallel=True).shape == (4, 7168)


def test_kda_attention_all_branches(monkeypatch):
    monkeypatch.setattr(k3_attn, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(k3_attn, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        k3_attn,
        "MergedColumnParallelLinear",
        lambda in_f, out_list, *, disable_tp=False, **kw: _DummyLinear(
            in_f, sum(out_list) if disable_tp else sum(out_list) // 2
        ),
    )
    monkeypatch.setattr(k3_attn, "ColumnParallelLinear", _DummyLinear)
    monkeypatch.setattr(k3_attn, "ReplicatedLinear", _DummyLinear)
    monkeypatch.setattr(k3_attn, "RowParallelLinear", _DummyLinear)
    monkeypatch.setattr(
        k3_attn,
        "RMSNorm",
        lambda dim, eps: SimpleNamespace(variance_epsilon=eps, weight=torch.ones(dim)),
    )
    monkeypatch.setattr(
        k3_attn,
        "build_kimi_dispatched_kda_op",
        lambda *a, **kw: lambda qkv, *rest: torch.ones(
            (qkv.shape[0], 3, 128), dtype=qkv.dtype
        ),
    )
    monkeypatch.setattr(
        k3_attn,
        "build_kimi_pooled_kda_op",
        lambda *a, **kw: lambda qkv, *rest: torch.ones(
            (qkv.shape[0], 3, 128), dtype=qkv.dtype
        ),
    )

    class _FakeFPC(k3_cops.FusedPrefillCollectives):
        def __init__(self, prefix):
            self.prefix = prefix

        def gather_project(self, hs, w):
            return hs @ w

        def project_reduce_scatter(self, x, projection):
            return projection(x)[0]

        def all_gather(self, x, dim=0):
            return x

        def reduce_scatter(self, x, dim=0):
            return x

        def all_reduce(self, x):
            return x

    monkeypatch.setattr(k3_attn, "FusedPrefillCollectives", _FakeFPC)
    tp_grp = SimpleNamespace(
        all_gather=lambda x, dim=0: x,
        reduce_scatter=lambda x, dim=0: x,
        all_reduce=lambda x: x,
    )
    monkeypatch.setattr(k3_attn, "get_tp_group", lambda: tp_grp)

    def _make_vllm_cfg(spec_tokens=None):
        return SimpleNamespace(
            quant_config=None,
            speculative_config=(
                SimpleNamespace(num_speculative_tokens=spec_tokens)
                if spec_tokens is not None
                else None
            ),
            compilation_config=SimpleNamespace(static_forward_context={}),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        )

    # Validation errors
    with pytest.raises(ValueError, match="KDA requires linear_attn_config"):
        k3_attn.KimiDeltaAttention(
            SimpleNamespace(linear_attn_config=None), _make_vllm_cfg(), "kda0"
        )

    with pytest.raises(ValueError, match="KDA num_heads must be divisible by TP size"):
        k3_attn.KimiDeltaAttention(
            SimpleNamespace(
                linear_attn_config={
                    "head_dim": 128,
                    "num_heads": 5,
                    "short_conv_kernel_size": 4,
                }
            ),
            _make_vllm_cfg(),
            "kda0",
        )

    # 1. use_full_rank_gate=False, speculative_config=None, TPU_K3_SP_PREFILL=False
    monkeypatch.setattr(k3_attn.envs, "TPU_K3_SP_PREFILL", False)
    cfg_low_rank = SimpleNamespace(
        hidden_size=7168,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 6,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": False,
        },
    )
    vcfg1 = _make_vllm_cfg()
    kda1 = k3_attn.KimiDeltaAttention(cfg_low_rank, vcfg1, "kda1")
    with pytest.raises(ValueError, match="Duplicate layer name"):
        k3_attn.KimiDeltaAttention(cfg_low_rank, vcfg1, "kda1")

    assert kda1.mamba_type == k3_attn.MambaAttentionBackendEnum.LINEAR
    assert kda1.get_state_shape() == ((3, 1, 1152), (3, 128, 128))
    monkeypatch.setattr(
        k3_attn.MambaStateDtypeCalculator,
        "kda_state_dtype",
        lambda dt, cdt: (torch.float32, torch.bfloat16),
    )
    assert kda1.get_state_dtype() == (torch.float32, torch.bfloat16)

    # _metadata branches
    monkeypatch.setattr(k3_attn, "is_forward_context_available", lambda: False)
    assert kda1._metadata() is None

    monkeypatch.setattr(k3_attn, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        k3_attn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata="not_a_dict"),
    )
    assert kda1._metadata() is None

    monkeypatch.setattr(
        k3_attn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"kda1": "bad"}),
    )
    with pytest.raises(TypeError, match="incompatible attention metadata"):
        kda1._metadata()

    # process_weights_after_loading on kda1 (early return in _pack_prefill_projection)
    kda1.process_weights_after_loading(torch.bfloat16)
    assert kda1.packed_prefill_weight is None

    # 2. use_full_rank_gate=True, speculative_config=2, TPU_K3_SP_PREFILL=True
    monkeypatch.setattr(k3_attn.envs, "TPU_K3_SP_PREFILL", True)
    cfg_full_rank = SimpleNamespace(
        hidden_size=7168,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 6,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
    )
    kda2 = k3_attn.KimiDeltaAttention(cfg_full_rank, _make_vllm_cfg(2), "kda2")
    # _pack_prefill_projection: unflipped -> early return
    kda2._pack_prefill_projection()
    assert kda2.packed_prefill_weight is None

    # Set flipped weights of expected shapes ((7168, 1155), (7168, 384), (7168, 128))
    assert kda2.g_proj is not None
    assert kda2.f_a_proj is not None
    kda2.fused_qkvb_proj.weight = nn.Parameter(
        torch.ones((7168, 1155), dtype=torch.bfloat16)
    )
    kda2.g_proj.weight = nn.Parameter(torch.ones((7168, 384), dtype=torch.bfloat16))
    kda2.f_a_proj.weight = nn.Parameter(torch.ones((7168, 128), dtype=torch.bfloat16))
    for p in (kda2.fused_qkvb_proj, kda2.g_proj, kda2.f_a_proj):
        setattr(p, WEIGHT_FLIPPED_ATTR, True)
    kda2._pack_prefill_projection()
    assert kda2.packed_prefill_weight is not None
    assert kda2.packed_prefill_weight.shape == (7168, 1792)

    # Forward paths:
    pos = torch.zeros(4, dtype=torch.int32)
    hs = torch.ones(4, 7168, dtype=torch.bfloat16)
    monkeypatch.setattr(k3_attn, "is_forward_context_available", lambda: False)

    # A. packed_prefill_weight is not None:
    # A1. sequence_parallel=True with FusedPrefillCollectives
    assert kda2(pos, hs, sequence_parallel=True).shape == (4, 7168)
    # A2. sequence_parallel=False
    assert kda2(pos, hs, sequence_parallel=False).shape == (4, 7168)
    # A3. sequence_parallel=True with non-FPC group
    kda2.prefill_group = tp_grp  # type: ignore[assignment]
    assert kda2(pos, hs, sequence_parallel=True).shape == (4, 7168)

    # B. packed_prefill_weight is None (both full-rank gate and low-rank gate,
    # SP=True/False)
    kda2.packed_prefill_weight = None
    assert kda2(pos, hs, sequence_parallel=True).shape == (4, 7168)
    assert kda1(pos, hs, sequence_parallel=False).shape == (4, 7168)

    # C. kv_cache len == 1 (unified pool) & _core_attention_pooled branches
    meta = object.__new__(AttentionMetadata)
    meta.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    meta.seq_lens = torch.tensor([4], dtype=torch.int32)
    meta.block_tables = torch.zeros((1, 1), dtype=torch.int32)
    setattr(meta, "slot_mapping", torch.arange(4, dtype=torch.int64))  # noqa: B010
    meta.request_distribution = torch.tensor([0, 0, 1], dtype=torch.int32)
    meta.mamba_state_indices = torch.tensor([0], dtype=torch.int32)
    meta.mamba_slot_read_offsets = None
    meta.mamba_request_distribution = None
    monkeypatch.setattr(k3_attn, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        k3_attn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"kda1": meta, "kda2": meta}),
    )
    kda1.kv_cache = (torch.ones(2, 4),)  # type: ignore[assignment]
    assert kda1(pos, hs, sequence_parallel=False).shape == (4, 7168)

    # _core_attention_pooled error branches
    monkeypatch.setattr(k3_attn, "is_pcp_streaming_attention_metadata", lambda m: True)
    with pytest.raises(NotImplementedError, match="PCP streaming prefill"):
        kda1(pos, hs, sequence_parallel=False)
    monkeypatch.setattr(k3_attn, "is_pcp_streaming_attention_metadata", lambda m: False)

    meta.mamba_slot_read_offsets = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(
        NotImplementedError, match="Speculative decoding is not supported"
    ):
        kda1(pos, hs, sequence_parallel=False)
    meta.mamba_slot_read_offsets = None

    meta.mamba_state_indices = None
    with pytest.raises(RuntimeError, match="requires mamba_state_indices"):
        kda1(pos, hs, sequence_parallel=False)

    # D. kv_cache len == 2 (dispatched_kda_op) with all metadata fallback combinations
    kda1.kv_cache = (torch.ones(2, 4), torch.ones(2, 4))  # type: ignore[assignment]
    # D1: mamba_request_distribution=None, mamba_slot_read_offsets=None,
    # mamba_state_indices=None
    meta.mamba_request_distribution = None
    meta.mamba_slot_read_offsets = None
    meta.mamba_state_indices = None
    assert kda1(pos, hs, sequence_parallel=False).shape == (4, 7168)

    # D2: mamba_request_distribution provided, mamba_state_indices provided
    # (slot_read_offsets=None)
    meta.mamba_request_distribution = torch.tensor([0, 0, 1], dtype=torch.int32)
    meta.mamba_state_indices = torch.tensor([0], dtype=torch.int32)
    assert kda1(pos, hs, sequence_parallel=False).shape == (4, 7168)

    # D3: mamba_slot_read_offsets provided
    meta.mamba_slot_read_offsets = torch.tensor([0], dtype=torch.int32)
    assert kda1(pos, hs, sequence_parallel=False).shape == (4, 7168)


# ---------------------------------------------------------------------------
# 4. model.py tests
# ---------------------------------------------------------------------------


def _patch_kimi_model_deps(monkeypatch):
    class _FakeAttn(nn.Module):
        def __init__(self, config, vllm_config, prefix):
            super().__init__()
            self.prepared = False

        def prepare_sp_mla_gate(self):
            self.prepared = True

        def forward(self, positions, hidden_states, sequence_parallel=False):
            return hidden_states

    class _FakeMoE(nn.Module):
        def __init__(self, config, quant_config, prefix):
            super().__init__()
            self.dummy_init = False

        def initialize_dummy_router(self):
            self.dummy_init = True

        def forward(self, hidden_states, sequence_parallel=False):
            return hidden_states

    class _FakeMLP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states, sequence_parallel=False):
            return hidden_states

    class _FakeAttnRes(nn.Module):
        def __init__(self, hidden_size, eps, prefix):
            super().__init__()
            self.folded = False

        def process_weights_after_loading(self):
            self.folded = True

        def forward(self, prefix_sum, block_residuals):
            return prefix_sum

    monkeypatch.setattr(k3_model, "KimiDeltaAttention", _FakeAttn)
    monkeypatch.setattr(k3_model, "MultiHeadLatentAttention", _FakeAttn)
    monkeypatch.setattr(k3_model, "KimiMoE", _FakeMoE)
    monkeypatch.setattr(k3_model, "KimiMLP", _FakeMLP)
    monkeypatch.setattr(k3_model, "AttentionResidual", _FakeAttnRes)
    monkeypatch.setattr(
        k3_model,
        "RMSNorm",
        lambda dim, eps: nn.Identity(),
    )
    monkeypatch.setattr(
        k3_model,
        "VocabParallelEmbedding",
        lambda vocab, dim, **kw: nn.Embedding(vocab, dim),
    )
    monkeypatch.setattr(
        k3_model,
        "ParallelLMHead",
        lambda vocab, dim, **kw: nn.Linear(dim, vocab, bias=False),
    )
    monkeypatch.setattr(
        k3_model,
        "LogitsProcessor",
        lambda vocab: lambda lm_head, hs: lm_head(hs),
    )


def test_kimi_decoder_layer_model_and_causal_lm(monkeypatch):
    _patch_kimi_model_deps(monkeypatch)

    text_cfg = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_hidden_layers=2,
        num_experts=4,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        mla_use_output_gate=True,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        attn_res_block_size=None,
        q_lora_rank=1536,
        tie_word_embeddings=True,
        linear_attn_config={
            "num_heads": 6,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
        is_kda_layer=lambda idx: idx == 0,
    )
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_cfg, dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        quant_config=None,
        compilation_config=SimpleNamespace(level=0, mode=CompilationMode.NONE),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            enable_expert_parallel=False,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )

    # 1. KimiModel SP prefill validation error & success
    monkeypatch.setattr(k3_model.envs, "TPU_K3_SP_PREFILL", True)
    with pytest.raises(ValueError, match="requires TP32/EP32 and PP1/DP1"):
        k3_model.KimiModel(vllm_config=vllm_cfg, prefix="model")

    vllm_cfg_sp = SimpleNamespace(
        **{
            **vllm_cfg.__dict__,
            "parallel_config": SimpleNamespace(
                tensor_parallel_size=32,
                enable_expert_parallel=True,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            ),
        }
    )
    monkeypatch.setattr(k3_model, "get_tp_group", lambda: SimpleNamespace())
    monkeypatch.setattr(
        k3_model,
        "TokenShardCollectives",
        lambda grp: SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            all_gather=lambda x, dim=0: x.repeat(2, 1),
        ),
    )
    # Test with attn_res_block_size=2 so layer 0 starts block, layer 1 continues block
    text_cfg_res = SimpleNamespace(**{**text_cfg.__dict__, "attn_res_block_size": 2})
    vllm_cfg_sp_res = SimpleNamespace(
        **{
            **vllm_cfg_sp.__dict__,
            "model_config": SimpleNamespace(
                hf_text_config=text_cfg_res, dtype=torch.bfloat16
            ),
        }
    )
    m_sp = k3_model.KimiModel(vllm_config=vllm_cfg_sp_res, prefix="model")
    pos = torch.arange(4, dtype=torch.int32)
    ids = torch.zeros(4, dtype=torch.long)
    # Forward with input_ids, sequence_parallel=True, aux_hidden_state_layers=()
    out_sp = m_sp(ids, pos)
    assert out_sp.shape == (4, 16)

    # Forward with inputs_embeds, sequence_parallel=True, aux_hidden_state_layers=(0, 2)
    m_sp.aux_hidden_state_layers = (0, 2)
    out_sp2, aux_sp2 = m_sp(None, pos, inputs_embeds=torch.ones(4, 16))
    assert out_sp2.shape == (4, 16)
    assert len(aux_sp2) == 2

    # Forward with aux_hidden_state_layers=(1,) (so 0 not in aux_layers)
    m_sp.aux_hidden_state_layers = (1,)
    _, aux_sp3 = m_sp(ids, pos)
    assert len(aux_sp3) == 1

    # 2. KimiLinearForCausalLM (with TPU_K3_SP_PREFILL=False, attn_res_block_size=None)
    monkeypatch.setattr(k3_model.envs, "TPU_K3_SP_PREFILL", False)
    causal_lm = k3_model.KimiLinearForCausalLM(vllm_config=vllm_cfg, prefix="")
    # Also attach an AttentionResidual child so process_weights_after_loading
    # covers all 3 branches
    causal_lm.extra_res = k3_model.AttentionResidual(16, 1e-6, "extra_res")
    causal_lm.process_weights_after_loading()
    assert causal_lm.extra_res.folded is True

    assert causal_lm.embed_input_ids(ids).shape == (4, 16)
    with pytest.raises(ValueError, match="Invalid Kimi aux hidden-state layers"):
        causal_lm.set_aux_hidden_state_layers((-1,))
    causal_lm.set_aux_hidden_state_layers(())

    with pytest.raises(ValueError, match="does not support PP"):
        causal_lm(ids, pos, intermediate_tensors=object())
    hs_out = causal_lm(ids, pos)
    assert hs_out.shape == (4, 16)
    assert causal_lm.compute_logits(hs_out).shape == (4, 64)

    monkeypatch.setattr(
        k3_model,
        "kda_state_dtype",
        lambda cfg: (torch.float32, torch.bfloat16),
    )
    assert k3_model.KimiLinearForCausalLM.get_mamba_state_dtype_from_config(
        vllm_cfg
    ) == (torch.float32, torch.bfloat16)
    assert k3_model.KimiLinearForCausalLM.get_mamba_state_shape_from_config(
        vllm_cfg
    ) == ((3, 1, 1152), (3, 128, 128))
    monkeypatch.setattr(
        k3_model.MambaStateCopyFuncCalculator,
        "kda_state_copy_func",
        lambda: ("cp1", "cp2"),
    )
    assert k3_model.KimiLinearForCausalLM.get_mamba_state_copy_func() == ("cp1", "cp2")

    # load_weights: cover all branches
    p_qkvb = nn.Parameter(torch.zeros(4))
    p_exp = nn.Parameter(torch.zeros(4))
    loaded_exp_calls: list[str] = []
    p_exp.weight_loader = (
        lambda param, w, name, *, expert_id, shard_id: loaded_exp_calls.append(name)
    )  # type: ignore[attr-defined]
    params_dict_override = {
        "model.layers.0.self_attn.fused_qkvb_proj.weight": p_qkvb,
        "model.layers.1.block_sparse_moe.experts.w13_weight": p_exp,
    }
    monkeypatch.setattr(
        causal_lm, "named_parameters", lambda: params_dict_override.items()
    )

    monkeypatch.setattr(
        k3_model,
        "fused_moe_make_expert_params_mapping",
        lambda *a, **kw: [
            ("experts.w13_weight", "experts.0.w1.weight", 0, "w1"),
        ],
    )

    class _FakeAutoLoader:
        def __init__(self, module):
            self.module = module

        def load_weights(self, weights_iter, mapper=None):
            return {name for name, _ in weights_iter}

    monkeypatch.setattr(k3_model, "AutoWeightsLoader", _FakeAutoLoader)

    raw_weights = [
        ("vision_tower.patch.weight", torch.ones(2)),
        ("mm_projector.proj.weight", torch.ones(2)),
        # Fused attention mapping match in params_dict (with .weight_packed suffix)
        ("language_model.layers.0.self_attn.q_proj.weight_packed", torch.ones(4)),
        # Fused attention mapping candidate NOT in params_dict
        ("language_model.layers.99.self_attn.q_proj.weight", torch.ones(4)),
        # Expert mapping match
        ("language_model.layers.1.block_sparse_moe.experts.0.w1.weight", torch.ones(4)),
        # Ordinary weight
        ("language_model.model.norm.weight", torch.ones(4)),
    ]
    loaded_set = causal_lm.load_weights(raw_weights)
    assert "model.layers.1.block_sparse_moe.experts.w13_weight" in loaded_set
    assert "model.layers.0.self_attn.fused_qkvb_proj.weight" in loaded_set
    assert len(loaded_exp_calls) == 1

    # Also test load_weights with num_experts=None, q_lora_rank=None,
    # tie_word_embeddings=False, and w13_weight_packed present in params_dict
    causal_lm.config.num_experts = None
    causal_lm.config.q_lora_rank = None
    causal_lm.config.tie_word_embeddings = False
    params_dict_override2 = {
        "model.layers.1.block_sparse_moe.experts.w13_weight_packed": nn.Parameter(
            torch.zeros(4)
        ),
    }
    monkeypatch.setattr(
        causal_lm, "named_parameters", lambda: params_dict_override2.items()
    )
    loaded_set2 = causal_lm.load_weights(
        [("language_model.layers.0.mlp.weight_packed", torch.ones(4))]
    )
    assert "model.layers.0.mlp.weight_packed" in loaded_set2


def test_kimi_k3_for_conditional_generation_all_branches(monkeypatch):
    synced: list[torch.Tensor] = []
    monkeypatch.setattr(k3_model, "synchronize_tensors", lambda *ts: synced.extend(ts))
    monkeypatch.setattr(
        k3_model, "tensor_model_parallel_all_gather", lambda x, dim=-1: x
    )

    t_in = torch.ones(2, 4)
    assert torch.equal(k3_model._tpu_tp_all_gather(t_in, dim=0), t_in)
    assert len(synced) == 2

    assert (
        k3_model.KimiK3ForConditionalGeneration.get_placeholder_str("image", 0)
        == "<|kimi_image_placeholder|>"
    )
    with pytest.raises(ValueError, match="Unsupported modality"):
        k3_model.KimiK3ForConditionalGeneration.get_placeholder_str("audio", 0)

    # _maybe_ignore_quant_config
    ct_cfg = compressed_tensors.CompressedTensorsConfig.__new__(
        compressed_tensors.CompressedTensorsConfig
    )
    assert (
        k3_model.KimiK3ForConditionalGeneration._maybe_ignore_quant_config(ct_cfg)
        is None
    )
    other_q = SimpleNamespace()
    assert (
        k3_model.KimiK3ForConditionalGeneration._maybe_ignore_quant_config(other_q)  # type: ignore[arg-type]
        is other_q
    )

    # __init__ validation error when multimodal_config is None
    bad_vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(), multimodal_config=None
        )
    )
    with pytest.raises(ValueError, match="requires multimodal config"):
        k3_model.KimiK3ForConditionalGeneration(bad_vcfg)  # type: ignore[arg-type]

    # Patch KimiK3MoonViT3dPretrainedModel, KimiK25MultiModalProjector,
    # init_vllm_registered_model
    class _FakePatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size = (14, 14)
            self.proj = SimpleNamespace(in_channels=3)

        def forward(self, pv, grid, pos_embeds=None):
            return pv

    class _FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope_2d = SimpleNamespace(max_height=16, max_width=16)

        def prepare_encoder_metadata(self, grid, device=None):
            return {"meta": torch.ones(1)}

        def forward(self, hs, grid, encoder_metadata=None):
            return hs

    class _FakeVisionTower(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.p = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.merge_kernel_size = (2, 2)
            self.patch_embed = _FakePatchEmbed()
            self.encoder = _FakeEncoder()

        def prepare_encoder_cudagraph_metadata(
            self, grid_config, *, max_batch_size, max_seqlen_override=None, device=None
        ):
            return {"cu_seqlens": torch.zeros(2, dtype=torch.int32), "unused": None}

        def forward(self, pv, grid, encoder_metadata=None):
            return [torch.ones(2, 8)]

    class _FakeProjector(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return torch.ones(x.shape[0], 16)

    class _FakeLangModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.aux_layers = ()

        def set_aux_hidden_state_layers(self, layers):
            self.aux_layers = layers

        def embed_input_ids(self, input_ids):
            return torch.zeros(input_ids.shape[0], 16, dtype=torch.bfloat16)

        def forward(
            self,
            input_ids=None,
            positions=None,
            intermediate_tensors=None,
            inputs_embeds=None,
        ):
            return inputs_embeds if inputs_embeds is not None else torch.ones(2, 16)

        def compute_logits(self, hs):
            return hs

        def process_weights_after_loading(self):
            self.processed = True

    monkeypatch.setattr(k3_model, "KimiK3MoonViT3dPretrainedModel", _FakeVisionTower)
    monkeypatch.setattr(k3_model, "KimiK25MultiModalProjector", _FakeProjector)
    monkeypatch.setattr(
        k3_model, "init_vllm_registered_model", lambda **kw: _FakeLangModel()
    )
    monkeypatch.setattr(
        k3_model.current_platform, "current_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(k3_model, "is_vit_use_data_parallel", lambda heads: True)
    monkeypatch.setattr(
        k3_model.KimiK3ForConditionalGeneration,
        "_mark_tower_model",
        lambda self, cfg, mod: nullcontext(),
    )
    monkeypatch.setattr(
        k3_model.KimiK3ForConditionalGeneration,
        "_mark_language_model",
        lambda self, cfg: nullcontext(),
    )

    hf_cfg = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=16),
        vision_config=SimpleNamespace(num_attention_heads=12),
        media_placeholder_token_id=99,
    )
    vcfg_none_q = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_cfg,
            multimodal_config=SimpleNamespace(),
            dtype=torch.bfloat16,
            max_model_len=2048,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        quant_config=None,
        with_hf_config=lambda tc: SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=tc)
        ),
    )
    m1 = k3_model.KimiK3ForConditionalGeneration(vcfg_none_q)

    # Also instantiate with non-None quant_config to cover else branch on line 590
    vcfg_with_q = SimpleNamespace(**{**vcfg_none_q.__dict__, "quant_config": other_q})
    k3_model.KimiK3ForConditionalGeneration(vcfg_with_q)

    # set_aux_hidden_state_layers success + error
    m1.set_aux_hidden_state_layers((1,))
    assert m1.language_model.aux_layers == (1,)
    m1.language_model = nn.Module()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="does not expose aux hidden states"):
        m1.set_aux_hidden_state_layers((1,))
    m1.language_model = _FakeLangModel()  # type: ignore[assignment]

    # _parse_and_validate_media_input branches
    assert m1._parse_and_validate_media_input() is None
    with pytest.raises(TypeError, match="pixel_values must be a tensor"):
        m1._parse_and_validate_media_input(
            pixel_values="bad", grid_thws=torch.ones(1, 3)
        )
    with pytest.raises(TypeError, match="grid_thws must be a tensor"):
        m1._parse_and_validate_media_input(
            pixel_values=[torch.ones(2, 3, 14, 14)], grid_thws="bad"
        )
    with pytest.raises(ValueError, match="unexpected grid_thws shape"):
        m1._parse_and_validate_media_input(
            pixel_values=torch.ones(1, 2, 3, 14, 14), grid_thws=torch.ones(1, 2)
        )

    parsed = m1._parse_and_validate_media_input(
        pixel_values=torch.ones(1, 2, 3, 14, 14, dtype=torch.float32),
        grid_thws=torch.tensor([[1, 2, 2]]),
    )
    assert parsed is not None
    assert parsed["pixel_values"].shape == (2, 3, 14, 14)

    # embed_multimodal: None, use_data_parallel=True, use_data_parallel=False
    monkeypatch.setattr(
        k3_model,
        "vision_tower_forward",
        lambda vt, pv, grid, *, mm_projector, use_data_parallel: [pv],
    )
    assert m1.embed_multimodal() is None
    assert (
        m1.embed_multimodal(
            pixel_values=torch.ones(2, 3, 14, 14), grid_thws=torch.tensor([[1, 2, 2]])
        )
        is not None
    )
    m1.use_data_parallel = False
    assert (
        m1.embed_multimodal(
            pixel_values=torch.ones(2, 3, 14, 14), grid_thws=torch.tensor([[1, 2, 2]])
        )
        is not None
    )

    # SupportsEncoderCudaGraph methods
    assert m1.get_encoder_cudagraph_config().out_hidden_size == 16
    assert m1.get_input_modality({}) == "image"
    assert m1.get_max_frames_per_video() == 1
    assert m1.get_encoder_cudagraph_budget_range(vcfg_none_q) == (64, 1024)

    specs = m1.get_encoder_cudagraph_item_specs(
        {"grid_thws": torch.tensor([[1, 4, 4]])}
    )
    assert specs[0].input_size == 16
    assert specs[0].output_tokens == 4
    specs_list = m1.get_encoder_cudagraph_item_specs({"grid_thws": [[1, 4, 4]]})
    assert specs_list[0].output_tokens == 4

    mm_kw = {
        "pixel_values": torch.ones(32, 3, 14, 14),
        "grid_thws": torch.tensor([[1, 4, 4], [1, 4, 4]]),
    }
    empty_sel = m1.select_encoder_cudagraph_items(mm_kw, [])
    assert cast_tensor(empty_sel["pixel_values"]).shape[0] == 0
    non_empty_sel = m1.select_encoder_cudagraph_items(mm_kw, [1])
    assert cast_tensor(non_empty_sel["pixel_values"]).shape[0] == 16

    # prepare_encoder_cudagraph_capture_inputs: overflow error and valid
    with pytest.raises(ValueError, match="more than Kimi K3"):
        m1.prepare_encoder_cudagraph_capture_inputs(
            1024, 1, 1, torch.device("cpu"), torch.bfloat16
        )
    cap_in = m1.prepare_encoder_cudagraph_capture_inputs(
        16, 1, 1, torch.device("cpu"), torch.bfloat16
    )
    assert "pixel_values" in cap_in.values
    assert "unused" not in cap_in.values

    rep_buf = m1.prepare_encoder_cudagraph_replay_buffers(mm_kw, 2, 1)
    assert "pixel_values" in rep_buf.values

    monkeypatch.setattr(k3_model, "tpool_patch_merger_packed", lambda hs, idx: hs)
    monkeypatch.setattr(k3_model, "mm_projector_forward", lambda proj, outs: outs)
    cg_out = m1.encoder_cudagraph_forward(
        {
            "pixel_values": torch.ones(4, 16),
            "merge_gather_idx": torch.zeros(4, dtype=torch.long),
            "pos_embeds": torch.ones(4, 16),
        }
    )
    assert cg_out.shape == (4, 16)
    eager_out = m1.encoder_eager_forward(mm_kw)
    assert eager_out.shape == (2, 8)

    # embed_input_ids branches
    monkeypatch.setattr(
        m1, "_embed_text_input_ids", lambda ids, fn, is_multimodal=None: fn(ids)
    )
    monkeypatch.setattr(m1, "get_language_model", lambda: m1.language_model)

    ids_4 = torch.zeros(4, dtype=torch.long)
    assert m1.embed_input_ids(ids_4, None).shape == (4, 16)
    assert m1.embed_input_ids(ids_4, []).shape == (4, 16)

    with pytest.raises(ValueError, match="is_multimodal is required"):
        m1.embed_input_ids(ids_4, [torch.ones(2, 16)], is_multimodal=None)

    with pytest.raises(ValueError, match="Attempted to assign"):
        m1.embed_input_ids(
            ids_4,
            [torch.ones(2, 16)],
            is_multimodal=torch.tensor([True, False, False, False]),
        )

    # actual == 0 (expected == 0)
    assert m1.embed_input_ids(
        ids_4,
        [torch.zeros(0, 16)],
        is_multimodal=torch.tensor([False, False, False, False]),
    ).shape == (4, 16)

    # CPU merge (merged.device.type != "tpu")
    merged_cpu = m1.embed_input_ids(
        ids_4,
        [torch.ones(2, 16, dtype=torch.bfloat16)],
        is_multimodal=torch.tensor([True, False, True, False]),
    )
    assert merged_cpu.shape == (4, 16)

    # TPU merge (merged.device.type == "tpu")
    orig_where = torch.where
    monkeypatch.setattr(
        torch,
        "where",
        lambda cond, a, b: _make_fake_tpu_tensor(orig_where(cond, a, b)),
    )
    merged_tpu = m1.embed_input_ids(
        ids_4,
        [torch.ones(2, 16, dtype=torch.bfloat16)],
        is_multimodal=torch.tensor([True, False, True, False]),
    )
    assert merged_tpu.shape == (4, 16)
    monkeypatch.setattr(torch, "where", orig_where)

    # forward, compute_logits, process_weights_after_loading, classmethods, load_weights
    assert m1(ids_4, torch.arange(4)).shape == (2, 16)
    assert m1.compute_logits(torch.ones(2, 16)).shape == (2, 16)
    m1.process_weights_after_loading()
    assert m1.language_model.processed is True

    monkeypatch.setattr(
        k3_model.KimiLinearForCausalLM,
        "get_mamba_state_dtype_from_config",
        classmethod(lambda cls, cfg: (torch.float32, torch.bfloat16)),
    )
    monkeypatch.setattr(
        k3_model.KimiLinearForCausalLM,
        "get_mamba_state_shape_from_config",
        classmethod(lambda cls, cfg: ((3, 1152), (3, 128, 128))),
    )
    monkeypatch.setattr(
        k3_model.KimiLinearForCausalLM,
        "get_mamba_state_copy_func",
        classmethod(lambda cls: ("c1", "c2")),
    )
    assert k3_model.KimiK3ForConditionalGeneration.get_mamba_state_dtype_from_config(
        vcfg_none_q
    ) == (torch.float32, torch.bfloat16)
    assert k3_model.KimiK3ForConditionalGeneration.get_mamba_state_shape_from_config(
        vcfg_none_q
    ) == ((3, 1152), (3, 128, 128))
    assert k3_model.KimiK3ForConditionalGeneration.get_mamba_state_copy_func() == (
        "c1",
        "c2",
    )

    monkeypatch.setattr(
        k3_model,
        "AutoWeightsLoader",
        lambda mod: SimpleNamespace(
            load_weights=lambda w, mapper=None: {n for n, _ in w}
        ),
    )
    assert m1.load_weights([("a", torch.ones(1))]) == {"a"}


def cast_tensor(obj: object) -> torch.Tensor:
    assert isinstance(obj, torch.Tensor)
    return obj
