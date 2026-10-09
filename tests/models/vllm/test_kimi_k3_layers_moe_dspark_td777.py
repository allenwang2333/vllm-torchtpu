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

"""100% statement and branch coverage tests for TD7.77 modules:
- vllm_torchtpu.models.vllm.kimi_k3.collective_ops
- vllm_torchtpu.models.vllm.kimi_k3.dspark
- vllm_torchtpu.models.vllm.kimi_k3.kimi_vit
- vllm_torchtpu.models.vllm.kimi_k3.layers
- vllm_torchtpu.models.vllm.kimi_k3.moe
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import numpy as np
import pytest
import torch
from torch import nn

from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.models.vllm.kimi_k3 import collective_ops as coll_mod
from vllm_torchtpu.models.vllm.kimi_k3 import dspark as dspark_mod
from vllm_torchtpu.models.vllm.kimi_k3 import kimi_vit as vit_mod
from vllm_torchtpu.models.vllm.kimi_k3 import layers as layers_mod
from vllm_torchtpu.models.vllm.kimi_k3 import moe as moe_mod


class _FakeShardedOp:
    def __init__(self, name, fn, mesh, input_partition_specs, output_partition_specs):
        self.name = name
        self.fn = fn
        self.mesh = mesh
        self.in_specs = input_partition_specs
        self.out_specs = output_partition_specs
        self.fake_fn = None

    def register_fake(self, fake_fn):
        self.fake_fn = fake_fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def test_collective_ops_builders_and_classes(monkeypatch: pytest.MonkeyPatch):
    saved_ops = dict(coll_mod._OPS)
    coll_mod._OPS.clear()
    coll_mod._mesh_and_tables.cache_clear()
    coll_mod._build_token_op.cache_clear()
    try:
        monkeypatch.setattr(coll_mod, "sharded_jax_op", _FakeShardedOp)
        monkeypatch.setattr(
            coll_mod,
            "mla_chip_layout",
            lambda devs, owners: (((0, 1),), (), ()),
        )
        monkeypatch.setattr(
            coll_mod, "pack_mla_gate", lambda w, mesh, pairs: ("packed", w)
        )
        monkeypatch.setattr(
            coll_mod,
            "mla_preprocess",
            lambda x, w, pg, qn, kvn, mesh, owners, *, sequence_parallel, eps: (
                x,
                w,
                qn,
                kvn,
            ),
        )
        monkeypatch.setattr(
            coll_mod,
            "gather_moe_inputs",
            lambda x, w, ids, mesh, tables, *, collective_id: (x, w, ids),
        )
        monkeypatch.setattr(
            coll_mod,
            "ring_collective",
            lambda x, mesh, tables, *, gather, collective_id: x,
        )
        monkeypatch.setattr(
            coll_mod,
            "project_reduce_scatter",
            lambda x, w, mesh, tables, *, collective_id: (x, w),
        )
        monkeypatch.setattr(
            coll_mod,
            "gather_project",
            lambda x, w, mesh, tables, *, collective_id: np.zeros(
                (64, 1792), dtype=np.float32
            ),
        )

        mesh = SimpleNamespace(
            devices=SimpleNamespace(flat=[SimpleNamespace(id=i) for i in range(32)])
        )
        tables = np.zeros((2, 3, 32), dtype=np.int32)
        tables[0, 2] = np.arange(32, dtype=np.int32)
        owners = tuple(range(32))

        # 1. _build_mla_ops (miss + hit + fn + fake)
        pack_op, mla_op = coll_mod._build_mla_ops(mesh, owners, 0, 1e-5)
        assert coll_mod._build_mla_ops(mesh, owners, 0, 1e-5) == (pack_op, mla_op)
        w = torch.zeros((4, 4), dtype=torch.bfloat16)
        assert pack_op(w) == ("packed", w)
        assert pack_op.fake_fn(w).shape == (7168, 6144)
        x = torch.zeros((2, 4), dtype=torch.bfloat16)
        assert len(mla_op(x, w, w, w, w, True)) == 4
        assert [t.shape for t in mla_op.fake_fn(x, w, w, w, w, True)] == [
            (64, 1536),
            (64, 512),
            (64, 64),
            (64, 384),
        ]
        assert [t.shape for t in mla_op.fake_fn(x, w, w, w, w, False)] == [
            (2, 1536),
            (2, 512),
            (2, 64),
            (2, 384),
        ]

        # 2. _build_moe_gather_op (miss + hit + fn + fake)
        moe_ag = coll_mod._build_moe_gather_op(mesh, tables, 100)
        assert coll_mod._build_moe_gather_op(mesh, tables, 100) is moe_ag
        w2 = torch.zeros((2, 4), dtype=torch.bfloat16)
        ids = torch.zeros((2, 4), dtype=torch.int32)
        assert moe_ag(x, w2, ids) == (x, w2, ids)
        fx, fw, fids = moe_ag.fake_fn(x, w2, ids)
        assert fx.shape == (64, 4) and fw.shape == (64, 4) and fids.shape == (64, 4)

        # 3. _build_op (gather=True and gather=False, miss + hit + fn + fake)
        op_ag = coll_mod._build_op(mesh, tables, True, 200)
        assert coll_mod._build_op(mesh, tables, True, 200) is op_ag
        assert op_ag(np.zeros((2, 8))).shape == (2, 8)
        assert op_ag.fake_fn(torch.zeros((2, 8))).shape == (64, 8)

        op_rs = coll_mod._build_op(mesh, tables, False, 201)
        assert coll_mod._build_op(mesh, tables, False, 201) is op_rs
        assert op_rs(np.zeros((64, 8))).shape == (32, 2, 8)
        assert op_rs.fake_fn(torch.zeros((64, 8))).shape == (2, 8)

        # 4. _build_project_op (miss + hit + fn + fake)
        proj_op = coll_mod._build_project_op(mesh, tables, 300)
        assert coll_mod._build_project_op(mesh, tables, 300) is proj_op
        px, pw = proj_op(np.zeros((64, 384)), np.zeros((32, 384, 7168)))
        assert px.shape == (32, 2, 384) and pw.shape == (32, 384, 7168)
        assert proj_op.fake_fn(
            torch.zeros((64, 384)), torch.zeros((384, 7168))
        ).shape == (2, 7168)

        # 5. _build_gather_project_op (miss + hit + fn + fake)
        gproj_op = coll_mod._build_gather_project_op(mesh, tables, 400)
        assert coll_mod._build_gather_project_op(mesh, tables, 400) is gproj_op
        assert gproj_op(np.zeros((2, 7168)), np.zeros((32, 7168, 1792))).shape == (
            64,
            1792,
        )
        assert gproj_op.fake_fn(
            torch.zeros((2, 7168)), torch.zeros((7168, 1792))
        ).shape == (64, 1792)

        # 6. _mesh_and_tables (mismatch error vs valid)
        monkeypatch.setattr(coll_mod, "get_ep_group", lambda: None)
        monkeypatch.setattr(
            coll_mod, "get_tp_group", lambda: SimpleNamespace(ranks=list(range(32)))
        )
        with pytest.raises(ValueError, match="identical TP32 and EP32"):
            coll_mod._mesh_and_tables()
        coll_mod._mesh_and_tables.cache_clear()

        base_group = SimpleNamespace(
            ranks=list(range(32)),
            world_size=2,
            rank_in_group=0,
            all_gather=lambda t, dim=0: t,
            reduce_scatter=lambda t, dim=0: t,
            all_reduce=lambda t: t * 2,
        )
        monkeypatch.setattr(coll_mod, "get_ep_group", lambda: base_group)
        monkeypatch.setattr(coll_mod, "get_tp_group", lambda: base_group)
        monkeypatch.setattr(coll_mod, "build_ep_mesh", lambda axis_name: mesh)
        monkeypatch.setattr(coll_mod, "ep_rank_order", lambda: list(owners))
        monkeypatch.setattr(coll_mod, "ring_tables", lambda devs, ord_: tables)

        # 7. _build_token_op (world_size=2, 32, and invalid) & TokenShardCollectives
        monkeypatch.setattr(
            coll_mod.jax,
            "shard_map",
            lambda fn, **kw: fn,
        )
        monkeypatch.setattr(
            coll_mod.jax.lax,
            "all_gather",
            lambda part, *a, **kw: ("ag", part),
        )
        monkeypatch.setattr(
            coll_mod.jax.lax,
            "psum_scatter",
            lambda part, *a, **kw: ("rs", part),
        )
        with pytest.raises(ValueError, match="require TP2 or TP32"):
            coll_mod._build_token_op(4, True)

        tok_coll_2 = coll_mod.TokenShardCollectives(base_group)
        assert tok_coll_2.all_gather(x, dim=0) == ("ag", x)
        assert tok_coll_2.reduce_scatter(x, dim=0) == ("rs", x)
        assert tok_coll_2.ag.fake_fn(torch.zeros((4, 8))).shape == (8, 8)
        assert tok_coll_2.rs.fake_fn(torch.zeros((4, 8))).shape == (2, 8)

        tok_op_32 = coll_mod._build_token_op(32, True)
        assert tok_op_32(x) == ("ag", x)

        # 8. FusedPrefillCollectives
        fpc = coll_mod.FusedPrefillCollectives("model.layers.2.self_attn")
        assert fpc.layer == 2
        assert len(fpc.mla_ops(1e-5)) == 2

        # gather_moe_inputs: packed BF16 vs fallback with/without dtype conversion
        lat = torch.zeros((2, 2176), dtype=torch.bfloat16)
        w_f32 = torch.zeros((2, 4), dtype=torch.float32)
        ids_i64 = torch.zeros((2, 4), dtype=torch.int64)
        out_packed = fpc.gather_moe_inputs(lat, w_f32, ids_i64, num_experts=256)
        assert len(out_packed) == 3

        w_bf16 = torch.zeros((2, 4), dtype=torch.bfloat16)
        ids_i32 = torch.zeros((2, 4), dtype=torch.int32)
        out_fallback = fpc.gather_moe_inputs(lat, w_bf16, ids_i32, num_experts=65536)
        assert len(out_fallback) == 3

        # all_gather & reduce_scatter fast path vs base fallback
        assert fpc.all_gather(lat, dim=0).shape == (2, 2176)
        assert fpc.all_gather(torch.zeros((2, 10)), dim=0).shape == (2, 10)
        rs_in = torch.zeros((32, 2176), dtype=torch.bfloat16)
        assert fpc.reduce_scatter(rs_in, dim=0).shape == (32, 1, 2176)
        assert fpc.reduce_scatter(torch.zeros((2, 10)), dim=0).shape == (2, 10)

        # project_reduce_scatter fast path vs fallback
        fpc.project_rs = lambda x_in, w_in: (x_in, w_in)
        proj_ok = SimpleNamespace(
            weight=torch.zeros((384, 7168), dtype=torch.bfloat16),
            bias=None,
        )
        setattr(proj_ok, WEIGHT_FLIPPED_ATTR, True)
        prs_in = torch.zeros((32, 384), dtype=torch.bfloat16)
        px_out, _ = fpc.project_reduce_scatter(prs_in, proj_ok)
        assert px_out.shape == (32, 384)

        proj_fallback = lambda t: (torch.zeros((t.shape[0], 10)), None)  # noqa: E731
        assert fpc.project_reduce_scatter(prs_in, proj_fallback).shape == (32, 10)

        # all_reduce
        assert torch.equal(fpc.all_reduce(torch.ones((2, 2))), torch.full((2, 2), 2.0))
    finally:
        coll_mod._OPS.clear()
        coll_mod._OPS.update(saved_ops)
        coll_mod._mesh_and_tables.cache_clear()
        coll_mod._build_token_op.cache_clear()


def test_kimi_k3_dspark_full_coverage(monkeypatch: pytest.MonkeyPatch):
    # 1. _duplicate_context_kv_weights branches
    w0 = torch.ones((2, 2))
    stream = [
        ("model.embed_tokens.weight", w0),
        ("model.layers.abc.self_attn.kv_a_proj_with_mqa.weight", w0),
        ("model.layers.5.self_attn.kv_a_proj_with_mqa.weight", w0),
        ("model.layers.1.self_attn.kv_a_proj_with_mqa.weight", w0),
    ]
    duped = list(dspark_mod._duplicate_context_kv_weights(stream, num_layers=2))
    assert [name for name, _ in duped] == [
        "model.embed_tokens.weight",
        "model.layers.abc.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.5.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        "context_kv_proj.weight",
    ]
    assert duped[-1][1].shard_id == 1

    # 2. Stub dependencies for K3DSparkDecoderLayer, K3DSparkModel, K3DSparkForCausalLM
    class _FakeLinear(nn.Module):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__()
            total_out = (
                sum(out_features)
                if isinstance(out_features, list)
                else int(out_features)
            )
            self.total_out = total_out

        def forward(self, x):
            return torch.ones(
                (x.shape[0], self.total_out), dtype=x.dtype, device=x.device
            ), None

    class _FakeRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, x):
            return x

    class _FakeMLA(nn.Module):
        def __init__(self, config, vllm_config, *, prefix, **kwargs):
            super().__init__()
            is_quant = prefix.endswith("1.self_attn")
            core = SimpleNamespace(
                layer_name=prefix,
                kv_cache=torch.zeros((2, 4), dtype=torch.bfloat16),
                kv_cache_quantized_dtype=(torch.float8_e4m3fn if is_quant else None),
                _k_scale_float=None,
                mla_op=lambda *args: None,
            )
            self.mla_attn = SimpleNamespace(
                mla_attn=core,
                kv_a_layernorm=lambda t: t,
                rotary_emb=lambda pos, q_pe, k_pe: (q_pe, k_pe),
                num_heads=2,
            )

        def forward(self, positions, hidden_states):
            return hidden_states

    class _FakeMLP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    class _FakeMarkovHead(nn.Module):
        def __init__(self, vocab_size, draft_vocab_size, markov_rank, prefix=""):
            super().__init__()

        def embed(self, token_ids):
            return torch.ones((token_ids.shape[0], 4))

        def bias(self, markov_embed, logits_processor):
            return torch.ones((markov_embed.shape[0], 16))

    class _FakeAutoWeightsLoader:
        def __init__(self, model):
            self.model = model

        def load_weights(self, weights):
            return {name for name, _ in weights}

    monkeypatch.setattr(dspark_mod, "ReplicatedLinear", _FakeLinear)
    monkeypatch.setattr(dspark_mod, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(dspark_mod, "RMSNorm", _FakeRMSNorm)
    monkeypatch.setattr(dspark_mod, "MultiHeadLatentAttention", _FakeMLA)
    monkeypatch.setattr(dspark_mod, "KimiMLP", _FakeMLP)
    monkeypatch.setattr(dspark_mod, "DSparkMarkovHead", _FakeMarkovHead)
    monkeypatch.setattr(dspark_mod, "AutoWeightsLoader", _FakeAutoWeightsLoader)
    monkeypatch.setattr(dspark_mod, "get_draft_quant_config", lambda cfg: None)
    monkeypatch.setattr(
        dspark_mod,
        "quantize_kv",
        lambda dtype, k, value=None, k_scale=1.0: (k, None),
    )
    monkeypatch.setattr(
        dspark_mod,
        "LogitsProcessor",
        lambda vocab_size, scale=1.0: lambda lm_head, h: lm_head(h) * scale,
    )

    hf_config = SimpleNamespace(
        target_hidden_size=8,
        num_target_layers=2,
        hidden_size=8,
        intermediate_size=16,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        num_hidden_layers=2,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        vocab_size=32,
        draft_vocab_size=16,
        markov_rank=4,
        logit_scale=1.0,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        ),
        model_config=SimpleNamespace(get_num_layers=lambda pc: 10),
        parallel_config=SimpleNamespace(),
    )

    causal_lm = dspark_mod.K3DSparkForCausalLM(vllm_config=vllm_config, prefix="")
    causal_lm.model.embed_tokens = lambda ids: torch.ones((ids.shape[0], 8))
    causal_lm.lm_head = lambda h: torch.ones((h.shape[0], 16))

    ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    pos = torch.tensor([0, 1, 2], dtype=torch.int32)
    assert causal_lm.embed_input_ids(ids).shape == (3, 8)
    assert causal_lm.combine_hidden_states(torch.ones((3, 16))).shape == (3, 8)
    assert causal_lm.get_draft_attn_causal() == [False, False]
    assert causal_lm.get_draft_kv_cache_layer_names() == [
        "model.layers.10.self_attn",
        "model.layers.11.self_attn",
    ]

    meta_entry = SimpleNamespace(
        seq_lens=torch.tensor([3], dtype=torch.int32),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    ctx = causal_lm.tpu_precompute_and_store_context_kv(
        torch.ones((3, 16)), pos, (meta_entry, meta_entry)
    )
    assert ctx.shape == (3, 8)

    assert causal_lm(ids, pos, inputs_embeds=None).shape == (3, 8)
    assert causal_lm(ids, pos, inputs_embeds=torch.ones((3, 8))).shape == (3, 8)
    assert causal_lm.compute_logits(torch.ones((3, 8))).shape == (3, 16)
    assert torch.equal(causal_lm.map_draft_to_target(ids), ids)
    m_emb = causal_lm.markov_embed(ids)
    assert m_emb.shape == (3, 4)
    assert causal_lm.markov_bias(m_emb).shape == (3, 16)

    loaded = causal_lm.load_weights(
        [
            ("confidence_head.weight", torch.ones((1,))),
            ("embed_tokens.weight", torch.ones((1,))),
            ("lm_head.weight", torch.ones((1,))),
            ("layers.0.self_attn.kv_a_proj_with_mqa.weight", torch.ones((1,))),
        ]
    )
    assert "model.context_kv_proj.weight" in loaded


def test_kimi_vit_materialize_and_model_init(monkeypatch: pytest.MonkeyPatch):
    # 1. CPU tensor -> returns None
    cpu_qkv = torch.ones((2, 4))
    assert vit_mod._materialize_vision_qkv(nn.Module(), (), (cpu_qkv, None)) is None

    # 2. TPU tensor while compiling -> returns None
    tpu_qkv = SimpleNamespace(
        device=SimpleNamespace(type="tpu"),
        clone=lambda: torch.ones((2, 4)),
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert vit_mod._materialize_vision_qkv(nn.Module(), (), (tpu_qkv, None)) is None

    # 3. TPU tensor in eager mode -> copies, syncs, and returns tuple
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    synced = []
    monkeypatch.setattr(vit_mod, "synchronize_tensors", lambda t: synced.append(t))
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda _t: SimpleNamespace(copy_=lambda src: "materialized"),
    )
    res = vit_mod._materialize_vision_qkv(nn.Module(), (), (tpu_qkv, "bias_val"))
    assert res == ("materialized", "bias_val")
    assert len(synced) == 1

    # 4. KimiK3MoonViT3dPretrainedModel.__init__ registers hook on blocks
    hooks = []

    def _fake_super_init(self, *args, **kwargs):
        nn.Module.__init__(self)
        block = SimpleNamespace(
            wqkv=SimpleNamespace(register_forward_hook=lambda fn: hooks.append(fn))
        )
        self.encoder = SimpleNamespace(blocks=[block])

    monkeypatch.setattr(vit_mod.MoonViT3dPretrainedModel, "__init__", _fake_super_init)
    vit_mod.KimiK3MoonViT3dPretrainedModel()
    assert hooks == [vit_mod._materialize_vision_qkv]


def test_kimi_k3_layers_full_coverage(monkeypatch: pytest.MonkeyPatch):
    # 1. _build_attention_residual_op
    layers_mod._build_attention_residual_op.cache_clear()
    import torch_tpu._internal.pallas as pallas_mod

    import vllm_torchtpu.kernels.kimi_k3.attention_residual as ar_mod

    monkeypatch.setattr(
        ar_mod, "attention_residual", lambda p, h, w, *, eps: ("ar", p, h, w, eps)
    )

    class _FakePallasOp:
        def __init__(self, name, fn):
            self.name = name
            self.fn = fn
            self.fake_fn = None

        def register_fake(self, fake_fn):
            self.fake_fn = fake_fn

        def __call__(self, *args):
            return self.fn(*args)

    monkeypatch.setattr(pallas_mod, "jax_op", _FakePallasOp)
    op = layers_mod._build_attention_residual_op(1e-6)
    p_t = torch.ones((16, 4))
    assert op(p_t, p_t, p_t)[0] == "ar"
    assert op.fake_fn(p_t, p_t, p_t).shape == (16, 4)

    # 2. SituAndMul (linear_beta=None and linear_beta!=None)
    x = torch.ones((2, 8))
    assert layers_mod.SituAndMul(2.0, None)(x).shape == (2, 4)
    assert layers_mod.SituAndMul(2.0, 5.0)(x).shape == (2, 4)

    # 3. KimiMLP.__init__ and forward branches
    class _FakeMergedCol(nn.Module):
        def __init__(self, h, inter_list, **kwargs):
            super().__init__()
            self.out_dim = sum(inter_list)

        def forward(self, t):
            return torch.ones((t.shape[0], self.out_dim)), None

    class _FakeRowParallel(nn.Module):
        def __init__(self, inter, h, **kwargs):
            super().__init__()
            self.h = h
            self.quant_method = SimpleNamespace(
                apply=lambda layer, t, bias: torch.ones((t.shape[0], self.h))
            )

        def forward(self, t):
            return torch.ones((t.shape[0], self.h)), None

    monkeypatch.setattr(layers_mod, "MergedColumnParallelLinear", _FakeMergedCol)
    monkeypatch.setattr(layers_mod, "RowParallelLinear", _FakeRowParallel)
    monkeypatch.setattr(
        layers_mod,
        "SiluAndMul",
        lambda: (lambda t: t[:, : t.shape[1] // 2]),
    )
    monkeypatch.setattr(
        layers_mod,
        "TokenShardCollectives",
        lambda grp: SimpleNamespace(
            all_gather=lambda t, dim=0: t,
            reduce_scatter=lambda t, dim=0: t,
        ),
    )
    tp_grp = SimpleNamespace(
        rank_in_group=0,
        world_size=2,
        all_gather=lambda t, dim=0: t,
        reduce_scatter=lambda t, dim=0: t,
        all_reduce=lambda t: t * 2.0,
    )
    monkeypatch.setattr(layers_mod, "get_tp_group", lambda: tp_grp)

    with pytest.raises(ValueError, match="Unsupported Kimi activation"):
        layers_mod.KimiMLP(8, 16, "gelu")

    monkeypatch.setenv("TPU_K3_SP_PREFILL", "1")
    mlp_sp = layers_mod.KimiMLP(8, 16, "situ", tp_group=tp_grp)
    assert mlp_sp(torch.ones((2, 8)), sequence_parallel=True).shape == (2, 8)
    assert torch.all(mlp_sp(torch.ones((2, 8)), sequence_parallel=False) == 2.0)

    monkeypatch.setenv("TPU_K3_SP_PREFILL", "0")
    mlp_plain = layers_mod.KimiMLP(8, 16, "silu", tp_group=None)
    assert mlp_plain(torch.ones((2, 8)), sequence_parallel=False).shape == (2, 8)

    # 4. AttentionResidual.__init__, process_weights_after_loading, and forward
    monkeypatch.setattr(
        layers_mod,
        "RMSNorm",
        lambda h, eps: SimpleNamespace(weight=torch.ones((h,))),
    )
    monkeypatch.setattr(
        layers_mod,
        "ReplicatedLinear",
        lambda h, out, **kw: SimpleNamespace(weight=torch.ones((1, h))),
    )
    attn_res = layers_mod.AttentionResidual(4, 1e-6, prefix="attn_res")
    attn_res.process_weights_after_loading()
    assert attn_res._prefill_op is None

    # Empty block_residuals (K == 0)
    prefix_sum = torch.ones((2, 4), dtype=torch.bfloat16)
    empty_blocks = torch.empty((2, 0, 4), dtype=torch.bfloat16)
    assert attn_res(prefix_sum, empty_blocks) is prefix_sum

    # Single token (T == 1) and multi-token (T == 2)
    blocks_1 = torch.ones((1, 2, 4), dtype=torch.bfloat16)
    assert attn_res(torch.ones((1, 4), dtype=torch.bfloat16), blocks_1).shape == (1, 4)
    blocks_2 = torch.ones((2, 2, 4), dtype=torch.bfloat16)
    assert attn_res(prefix_sum, blocks_2).shape == (2, 4)

    # Prefill op active when T % 16 == 0
    attn_res._prefill_op = lambda p, b, w: p + 1.0
    p16 = torch.ones((16, 4), dtype=torch.bfloat16)
    b16 = torch.ones((16, 2, 4), dtype=torch.bfloat16)
    assert torch.all(attn_res(p16, b16) == 2.0)


def test_kimi_k3_moe_full_coverage(monkeypatch: pytest.MonkeyPatch):
    # 1. KimiRoutedOutputTransform with norm=None and norm!=None
    up = lambda t: (t * 2.0, None)  # noqa: E731
    tr_no_norm = moe_mod.KimiRoutedOutputTransform(None, up)
    assert torch.all(tr_no_norm(torch.ones((2, 4))) == 2.0)
    tr_with_norm = moe_mod.KimiRoutedOutputTransform(lambda t: t + 1.0, up)
    assert torch.all(tr_with_norm(torch.ones((2, 4))) == 4.0)

    # 2. KimiMoE SP prefill validation errors and packed/unpacked weight zeroing
    class _FakeGate(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.weight = nn.Parameter(torch.ones((output_size, input_size)))

        def forward(self, x):
            return torch.zeros((x.shape[0], self.output_size)), None

    class _FakeRunnerPacked(nn.Module):
        def __init__(self, with_routing=True, packed=True):
            super().__init__()
            qm = SimpleNamespace(
                apply_monolithic=lambda r, lat, rl: lat * 2.0,
            )
            if with_routing:
                qm.apply_with_routing = lambda r, lat, w, i: lat * 2.0
            self.routed_experts = SimpleNamespace(
                quant_method=qm,
                w13_weight=None if packed else nn.Parameter(torch.ones((2, 2))),
                w2_weight=None if packed else nn.Parameter(torch.ones((2, 2))),
                w13_weight_packed=nn.Parameter(torch.ones((2, 2))) if packed else None,
                w2_weight_packed=nn.Parameter(torch.ones((2, 2))) if packed else None,
            )
            self.moe_config = SimpleNamespace(
                intermediate_size_per_partition_unpadded=None
            )

        def forward(self, hidden_states, router_logits):
            return hidden_states * 3.0

    monkeypatch.setattr(moe_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(moe_mod, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(moe_mod, "GateLinear", _FakeGate)
    monkeypatch.setattr(
        moe_mod,
        "make_latent_projection",
        lambda in_f, out_f, **kw: lambda x: (torch.ones((x.shape[0], out_f)), None),
    )
    monkeypatch.setattr(
        moe_mod,
        "RMSNorm",
        lambda dim, eps: nn.Identity(),
    )
    monkeypatch.setattr(
        moe_mod,
        "KimiMLP",
        lambda *a, **kw: lambda x, sequence_parallel=False: x,
    )
    monkeypatch.setattr(
        moe_mod,
        "FusedPrefillCollectives",
        lambda prefix: SimpleNamespace(
            gather_moe_inputs=lambda lat, w, i, num_experts: (lat, w, i),
            reduce_scatter=lambda t, dim=0: t,
            all_reduce=lambda t: t,
        ),
    )
    monkeypatch.setattr(
        moe_mod.moe_routing,
        "route",
        lambda routed, latent, rl: (
            torch.ones((latent.shape[0], 2)),
            torch.zeros((latent.shape[0], 2), dtype=torch.int32),
        ),
    )

    cfg = SimpleNamespace(
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=64,
        hidden_size=8,
        routed_expert_hidden_size=4,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-6,
        num_shared_experts=1,
        hidden_act="situ",
        activation_situ_beta=1.0,
        activation_situ_linear_beta=0.5,
        min_moe_intermediate_per_partition=128,
        moe_renormalize=True,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0,
    )

    # Error 1: SP prefill without 2-core intra-chip group
    monkeypatch.setenv("TPU_K3_SP_PREFILL", "1")
    monkeypatch.setattr(moe_mod, "get_intra_chip_group", lambda: None)
    with pytest.raises(ValueError, match="two-core chip group"):
        moe_mod.KimiMoE(cfg, quant_config=None, prefix="model.layers.0.moe")

    # Error 2: SP prefill when quant_method lacks apply_with_routing
    monkeypatch.setattr(
        moe_mod, "get_intra_chip_group", lambda: SimpleNamespace(world_size=2)
    )
    monkeypatch.setattr(
        moe_mod, "FusedMoEFactory", lambda **kw: _FakeRunnerPacked(with_routing=False)
    )
    with pytest.raises(ValueError, match="requires the native MXFP4 MoE backend"):
        moe_mod.KimiMoE(cfg, quant_config=None, prefix="model.layers.0.moe")

    # Valid SP prefill + EP hierarchical split + packed weight zeroing
    vllm_cfg_ep = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
        load_config=SimpleNamespace(load_format="dummy"),
    )
    monkeypatch.setattr(moe_mod, "get_current_vllm_config_or_none", lambda: vllm_cfg_ep)
    monkeypatch.setattr(moe_mod, "hierarchical_split_or_none", lambda: (2, 4, 2))
    monkeypatch.setattr(
        moe_mod, "FusedMoEFactory", lambda **kw: _FakeRunnerPacked(with_routing=True)
    )
    moe_ep = moe_mod.KimiMoE(cfg, quant_config=None, prefix="model.layers.0.moe")
    assert torch.all(moe_ep.experts.routed_experts.w13_weight_packed == 0)
    assert torch.all(moe_ep.experts.routed_experts.w2_weight_packed == 0)

    # Unpacked weight zeroing when w13_weight / w2_weight are present
    monkeypatch.setattr(
        moe_mod,
        "FusedMoEFactory",
        lambda **kw: _FakeRunnerPacked(with_routing=True, packed=False),
    )
    moe_unpacked = moe_mod.KimiMoE(cfg, quant_config=None, prefix="model.layers.0.moe")
    assert torch.all(moe_unpacked.experts.routed_experts.w13_weight == 0)
    assert torch.all(moe_unpacked.experts.routed_experts.w2_weight == 0)

    # initialize_dummy_router: dummy vs non-dummy
    moe_ep.initialize_dummy_router()
    vllm_cfg_ep.load_config.load_format = "safetensors"
    moe_ep.initialize_dummy_router()

    # forward -> _forward_expert_parallel with sequence_parallel=True
    # and down_proj / norm / up_proj / shared_experts all active
    x = torch.ones((2, 8))
    out_sp = moe_ep(x, sequence_parallel=True)
    assert out_sp.shape == (2, 8)

    # _forward_expert_parallel with sequence_parallel=False (apply_monolithic)
    # and no down_proj / norm / up_proj / shared_experts
    moe_ep.routed_expert_down_proj = None
    moe_ep.routed_expert_norm = None
    moe_ep.routed_expert_up_proj = None
    moe_ep.shared_experts = None
    out_ep = moe_ep(x, sequence_parallel=False)
    assert out_ep.shape == (2, 8)

    # Non-SP KimiMoE forward
    monkeypatch.setenv("TPU_K3_SP_PREFILL", "0")
    monkeypatch.setattr(moe_mod, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(moe_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    moe_std = moe_mod.KimiMoE(cfg, quant_config=None, prefix="model.layers.1.moe")
    assert torch.all(moe_std(x, sequence_parallel=False) == 3.0)
