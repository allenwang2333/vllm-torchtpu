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
"""CPU unit tests for top-level Qwen3.5 megakernel modular package structure,
step metadata preparation, fused prep wrapper, and A/B calibrator."""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import torch

from vllm_torchtpu.kernels import qwen35_megakernel as pkg
from vllm_torchtpu.kernels.qwen35_megakernel import attention_block
from vllm_torchtpu.kernels.qwen35_megakernel import blocks
from vllm_torchtpu.kernels.qwen35_megakernel import collectives
from vllm_torchtpu.kernels.qwen35_megakernel import config
from vllm_torchtpu.kernels.qwen35_megakernel import context
from vllm_torchtpu.kernels.qwen35_megakernel import gdn_block
from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import moe_block
from vllm_torchtpu.kernels.qwen35_megakernel import pool_dma
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io
from vllm_torchtpu.kernels.qwen35_megakernel import rings
from vllm_torchtpu.kernels.qwen35_megakernel import vllm_adapter as VA
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import (
    MKABCalibrator,
    parse_mk_ab_enabled,
    prepare_step_metadata,
)

from qwen35_megakernel_test_utils import bare_mk


def test_modular_reexports_and_symbols():
    """Verify that the modular submodules expose their expected classes/functions
    and `megakernel.py` / `__init__.py` preserve backward-compatible re-exports."""
    assert K.MKConfig is config.MKConfig
    assert K.Tiles is config.Tiles
    assert K.default_tiles is config.default_tiles
    assert K.weight_shapes is config.weight_shapes
    assert K.state_shapes is config.state_shapes
    assert K.smalls_layout is config.smalls_layout
    assert K.encode_ecopy is config.encode_ecopy
    assert K.plan is config.plan

    assert K.MKBuildContext is context.MKBuildContext
    assert K.PallasAllReduce is collectives.PallasAllReduce
    assert K.WeightRings is rings.WeightRings
    assert K.DenseWeightRing is rings.DenseWeightRing
    assert K.ExpertWeightRing is rings.ExpertWeightRing
    assert K.KVPoolDMA is pool_dma.KVPoolDMA
    assert K.GDNPoolDMA is pool_dma.GDNPoolDMA
    assert K.GprRecIn is pool_dma.GprRecIn
    assert K.GDNBlock is gdn_block.GDNBlock
    assert K.PagedAttentionBlock is attention_block.PagedAttentionBlock
    assert K.MoEBlock is moe_block.MoEBlock
    assert blocks.GDNBlock is gdn_block.GDNBlock

    assert pkg.MKConfig is config.MKConfig
    assert pkg.Tiles is config.Tiles
    assert pkg.PallasAllReduce is collectives.PallasAllReduce
    assert pkg.WeightRings is rings.WeightRings
    assert pkg.KVPoolDMA is pool_dma.KVPoolDMA
    assert pkg.GDNPoolDMA is pool_dma.GDNPoolDMA
    assert pkg.GDNBlock is gdn_block.GDNBlock
    assert pkg.PagedAttentionBlock is attention_block.PagedAttentionBlock
    assert pkg.MoEBlock is moe_block.MoEBlock


def test_prepare_step_metadata_t1_and_t4():
    num_reqs = 3
    max_pages = 16
    bt_1d = torch.arange(1, 1 + 8 * max_pages, dtype=torch.int32)
    attn_md = SimpleNamespace(
        seq_lens=torch.tensor([100, 250, 40, 0, 0, 0, 0, 0], dtype=torch.int32),
        block_tables=bt_1d,
    )
    roff_full = torch.zeros((512,), dtype=torch.int32)
    roff_full[10] = 2
    roff_full[20] = 3
    roff_full[30] = 1
    mamba_mds = []
    for g in range(3):
        state_ids = torch.tensor(
            [10 + g, 20 + g, 30 + g, 0, 0, 0, 0, 0], dtype=torch.int32
        )
        ckpt_2d = torch.zeros((8, 4), dtype=torch.int32)
        for s in range(num_reqs):
            for t in range(4):
                ckpt_2d[s, t] = 100 * (g + 1) + 10 * s + t
        mamba_mds.append(
            SimpleNamespace(
                mamba_state_indices=state_ids,
                mamba_ckpt_indices=ckpt_2d,
                mamba_slot_read_offsets=roff_full,
            )
        )

    positions_t4 = torch.tensor(
        [96, 97, 98, 99, 246, 247, 248, 249, 36, 37, 38, 39, 0, 0, 0, 0],
        dtype=torch.int32,
    )
    pos4, pt4, ckpt4, roff4 = prepare_step_metadata(
        attn_md,
        mamba_mds,
        num_reqs=num_reqs,
        bucket_t=16,
        max_pages_per_seq=max_pages,
        positions=positions_t4,
        tokens_per_seq=4,
    )
    assert pos4.shape == (16,)
    assert torch.equal(pos4[:12], positions_t4[:12])
    assert torch.all(pos4[12:] == 0)
    assert pt4.shape == (4, max_pages)
    assert torch.all(pt4[3] == 0)
    assert ckpt4.shape == (3, 4, 4)
    assert torch.all(ckpt4[:, 3, :] == 0)
    assert roff4.shape == (3, 4)
    assert roff4[0, :4].tolist() == [2, 3, 1, 0]

    pos1, pt1, ckpt1, roff1 = prepare_step_metadata(
        attn_md,
        mamba_mds,
        num_reqs=num_reqs,
        bucket_t=8,
        max_pages_per_seq=max_pages,
        tokens_per_seq=1,
    )
    assert pos1.shape == (8,)
    assert pos1[:3].tolist() == [99, 249, 39]
    assert pt1.shape == (8, max_pages)
    assert ckpt1.shape == (3, 8, 1)
    assert roff1.shape == (3, 8)
    assert torch.all(roff1 == 0)


def test_jax_prepare_step_inputs_and_fused_prep_wrapper(monkeypatch):
    """Verify `jax_prepare_step_inputs` and `wrap_step_with_fused_prep` match `prep_only`
    and fuse `embed_tokens` + metadata padding + output row slicing into one JAX step."""
    assert VA.parse_mk_fuse_prep(None) is True
    assert VA.parse_mk_fuse_prep("0") is False
    assert VA.parse_mk_fuse_prep("1") is True

    H = 64
    vocab = 128
    max_pages = 8
    torch.manual_seed(0)
    emb = torch.nn.Embedding(vocab, H, dtype=torch.bfloat16)
    emb_w_jnp = jnp.asarray(emb.weight.detach().float().numpy()).astype(jnp.bfloat16)

    for T, B, m_rows in ((1, 8, 16), (4, 20, 32)):
        nseq = B // T
        input_ids = torch.randint(1, vocab, (B + 4,), dtype=torch.int32)
        positions = (
            torch.arange(100, 100 + B + 4, dtype=torch.int32)
            .unsqueeze(0)
            .expand(3, -1)
        )
        block_tables = torch.arange(
            1, (nseq + 2) * max_pages + 1, dtype=torch.int32
        ).reshape(nseq + 2, max_pages)
        s0 = torch.arange(1, nseq + 3, dtype=torch.int32)
        s0[-2:] = 0
        s1 = s0.clone()
        s2 = s0.clone()
        c_cols = 4
        c0 = torch.arange(10, 10 + (nseq + 2) * c_cols, dtype=torch.int32).reshape(
            nseq + 2, c_cols
        )
        c0[-1, :] = 0
        c1 = c0 + 100
        c1[-1, :] = 0
        c2 = c0 + 200
        c2[-1, :] = 0
        roff = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1], dtype=torch.int32)

        pos_j, h0_j, pt_j, ckpt_j, roff_j = pool_io.jax_prepare_step_inputs(
            jnp.asarray(input_ids.numpy()),
            jnp.asarray(positions.numpy()),
            jnp.asarray(block_tables.numpy()),
            jnp.asarray(s0.numpy()),
            jnp.asarray(c0.numpy()),
            jnp.asarray(s1.numpy()),
            jnp.asarray(c1.numpy()),
            jnp.asarray(s2.numpy()),
            jnp.asarray(c2.numpy()),
            jnp.asarray(roff.numpy()),
            emb_w_jnp,
            T=T,
            B=B,
            m_rows=m_rows,
            max_pages=max_pages,
            embed_tp_size=1,
            embed_vocab_size=vocab,
        )

        c = dataclasses.replace(
            K.TINY_EP,
            R=1,
            B=B,
            T=T,
            H=H,
            paged_kv=True,
            num_pages=64,
            page_size=128,
            max_pages_per_seq=max_pages,
            ctx=1024,
        )
        n_w = sum(
            c.L if (not c.stack_e and k in ("e13", "e2")) else 1 for k in K.W_NAMES
        )
        n_p = c.NATT
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))

        def fake_make_flat_pooled_step(
            mesh_arg, cfg, tiles, fused_pool_io=None, out_rows=None
        ):
            target_r = int(out_rows or cfg.M)

            def _body(*args):
                pos_arg, hid_arg = args[0], args[1]
                raw_pools = args[2 + n_w : 2 + n_w + n_p]
                pt_arg, ckpt_arg, roff_arg = args[2 + n_w + n_p :]
                tag = (
                    pos_arg.sum().astype(jnp.bfloat16)
                    + pt_arg.sum().astype(jnp.bfloat16)
                    + ckpt_arg.sum().astype(jnp.bfloat16)
                    + roff_arg.sum().astype(jnp.bfloat16)
                )
                return (hid_arg[:target_r] + tag, *raw_pools)

            smap = jax.shard_map(
                _body,
                mesh=mesh_arg,
                in_specs=(P("tp"),) * (2 + n_w + n_p + 3),
                out_specs=(P("tp"),) * (1 + n_p),
                check_vma=False,
            )
            smap._inner_body = _body
            smap._c_paged = cfg
            return (
                smap,
                (P("tp"),) * (2 + n_w + n_p + 3),
                (P("tp"),) * (1 + n_p),
                tuple(range(2 + n_w, 2 + n_w + n_p)),
            )

        def fake_sharded_callable(name, fn, **kw):
            def _call(*t_args):
                j_args = [
                    jnp.asarray(t.detach().float().numpy()).astype(jnp.bfloat16)
                    if t.dtype == torch.bfloat16
                    else jnp.asarray(t.detach().numpy())
                    for t in t_args
                ]
                j_outs = fn(*j_args)
                return tuple(
                    torch.from_numpy(np.asarray(o, np.float32).copy()).to(
                        torch.bfloat16
                    )
                    if o.dtype == jnp.bfloat16
                    else torch.from_numpy(np.asarray(o).copy())
                    for o in j_outs
                )

            return _call

        monkeypatch.setattr(
            VA.pool_io, "make_flat_pooled_step", fake_make_flat_pooled_step
        )
        monkeypatch.setattr(VA, "sharded_jax_callable", fake_sharded_callable)
        monkeypatch.setattr(torch, "compile", lambda f, **kw: f)

        w_flat = [torch.zeros((1,), dtype=torch.bfloat16) for _ in range(n_w)]
        outs_by_mode = {}
        for fuse_prep in (False, True):
            pools = [torch.zeros((4, 2), dtype=torch.int32) for _ in range(n_p)]
            mk = bare_mk(
                monkeypatch,
                mode=f"fp{int(fuse_prep)}_{T}_{B}",
                base_cfg=c,
                mesh=mesh,
                embed_tokens=emb,
                w_flat=w_flat,
                raw_pools=pools,
                max_pages_per_seq=max_pages,
                zero_copy=True,
                fuse_prep=fuse_prep,
                _hlo_checked=True,
                _raiden_sync_views=[],
            )
            step_fn = mk._build_bucket_op(B, tokens_per_seq=T, out_pad_rows=B)
            if fuse_prep:
                assert mk._prep_steps == {}
            else:
                assert (T, B) in mk._prep_steps
            out = step_fn(
                input_ids,
                positions,
                block_tables,
                s0,
                c0,
                s1,
                c1,
                s2,
                c2,
                roff,
                w_flat,
                pools,
            )
            assert out.shape == (B, H)
            outs_by_mode[fuse_prep] = out

        torch.testing.assert_close(outs_by_mode[True], outs_by_mode[False])
        assert h0_j.shape == (m_rows, H)
        assert pos_j.shape == (B,)
        assert pt_j.shape == (nseq, max_pages)
        assert ckpt_j.shape == (3, nseq, T)
        assert roff_j.shape == (3, nseq)

    # Multi-rank (TP=8) regression test: under torch_tpu's jax.export bridge,
    # `wrap_step_with_fused_prep(..., embed_tp_size=8, has_tp_rank_arg=True)`
    # must NOT emit `stablehlo.partition_id` (which XLA's SPMD partitioner
    # rejects) and must match VocabParallelEmbedding across all 8 TP ranks.
    if len(jax.devices()) >= 8:
        from jax.sharding import NamedSharding

        tp_size = 8
        B_tp, T_tp, m_rows_tp = 8, 4, 8
        nseq_tp = B_tp // T_tp
        n_part = vocab // tp_size
        mesh8 = Mesh(np.array(jax.devices()[:tp_size]), ("tp",))

        def _body8(*args):
            pos_arg, hid_arg = args[0], args[1]
            raw_pools = args[2 : 2 + 1]
            return (hid_arg + pos_arg.sum().astype(jnp.bfloat16), *raw_pools)

        base_smap8 = jax.shard_map(
            _body8,
            mesh=mesh8,
            in_specs=(P("tp"),) * (2 + 1 + 1 + 3),
            out_specs=(P("tp"),) * 2,
            check_vma=False,
        )
        base_smap8._inner_body = _body8
        base_smap8._c_paged = dataclasses.replace(
            K.TINY_EP,
            R=tp_size,
            B=B_tp,
            T=T_tp,
            H=H,
            L=1,
            att_every=1,
            stack_e=True,
            max_pages_per_seq=max_pages,
        )
        # Only 1 dummy weight leaf by mocking W_NAMES for this lightweight export test
        monkeypatch.setattr(pool_io, "W_NAMES", ("ln_f",))
        wrapped8, in_specs8, out_specs8, donate8 = pool_io.wrap_step_with_fused_prep(
            base_smap8,
            mesh8,
            base_smap8._c_paged,
            embed_tp_size=tp_size,
            embed_vocab_size=vocab,
            has_tp_rank_arg=True,
        )
        assert len(in_specs8) == 12 + 1 + 1
        assert donate8 == (13,)

        in_sh8 = tuple(NamedSharding(mesh8, s) for s in in_specs8)
        out_sh8 = tuple(NamedSharding(mesh8, s) for s in out_specs8)
        avals8 = [
            jax.ShapeDtypeStruct((tp_size * B_tp,), jnp.int32),  # input_ids
            jax.ShapeDtypeStruct((tp_size * 3, B_tp), jnp.int32),  # positions
            jax.ShapeDtypeStruct((tp_size * nseq_tp, max_pages), jnp.int32),  # bt
            jax.ShapeDtypeStruct((tp_size * nseq_tp,), jnp.int32),  # s0
            jax.ShapeDtypeStruct((tp_size * nseq_tp, 4), jnp.int32),  # c0
            jax.ShapeDtypeStruct((tp_size * nseq_tp,), jnp.int32),  # s1
            jax.ShapeDtypeStruct((tp_size * nseq_tp, 4), jnp.int32),  # c1
            jax.ShapeDtypeStruct((tp_size * nseq_tp,), jnp.int32),  # s2
            jax.ShapeDtypeStruct((tp_size * nseq_tp, 4), jnp.int32),  # c2
            jax.ShapeDtypeStruct((tp_size * 16,), jnp.int32),  # roff
            jax.ShapeDtypeStruct((vocab, H), jnp.bfloat16),  # embed_w
            jax.ShapeDtypeStruct((tp_size,), jnp.int32),  # tp_rank
            jax.ShapeDtypeStruct((tp_size, H), jnp.bfloat16),  # w_leaf
            jax.ShapeDtypeStruct((tp_size, 4), jnp.int32),  # raw_pool
        ]
        exported8 = jax.export.export(
            jax.jit(wrapped8, in_shardings=in_sh8, out_shardings=out_sh8)
        )(*avals8)
        mlir_text = exported8.mlir_module()
        assert "partition_id" not in mlir_text

        ids_test = jnp.asarray([0, 15, 16, 63, 64, 95, 112, 127], dtype=jnp.int32)
        ref_emb = emb_w_jnp[ids_test]

        def _sharded_embed(ids_rep, w_sharded, r_sharded):
            return jax.shard_map(
                lambda ids_l, w_l, r_l: pool_io.jax_vocab_parallel_embed(
                    ids_l,
                    w_l,
                    tp_axis="tp",
                    tp_size=tp_size,
                    tp_rank=r_l,
                    vocab_size=vocab,
                ),
                mesh=mesh8,
                in_specs=(P(), P("tp", None), P("tp")),
                out_specs=P("tp", None),
                check_vma=False,
            )(ids_rep, w_sharded, r_sharded)

        got_all = _sharded_embed(
            ids_test,
            emb_w_jnp.reshape(tp_size * n_part, H),
            jnp.arange(tp_size, dtype=jnp.int32),
        ).reshape(tp_size, B_tp, H)
        for r in range(tp_size):
            np.testing.assert_allclose(
                np.asarray(got_all[r], dtype=np.float32),
                np.asarray(ref_emb, dtype=np.float32),
                rtol=1e-2,
                atol=1e-2,
            )


def test_mk_ab_calibrator_and_hooks(monkeypatch):
    assert parse_mk_ab_enabled("1") is True
    assert parse_mk_ab_enabled("0") is False

    cal = MKABCalibrator(enabled=True, period_s=10.0, cap=4)
    assert cal.choose("mk_t4_b16", True) == (True, "mk_t4_b16", "mk")
    assert cal.last_decision == ("mk_t4_b16", "mk")
    cal.record(10.0, padded_tokens=16, now=cal._t0 + 1.0)

    assert cal.choose("mk_t4_b16", True) == (False, "ab_stock_t4_b16", "stock")
    assert cal.last_decision == ("mk_t4_b16", "stock")
    cal.record(14.0, padded_tokens=16, now=cal._t0 + 2.0)

    assert cal.choose("mk_t4_b8", True) == (True, "mk_t4_b8", "mk")
    cal.record(8.0, padded_tokens=16, now=cal._t0 + 3.0)

    assert cal.choose("stock_t4_n13", False) == (False, "stock_t4_n13", "stock")
    cal.record(19.5, padded_tokens=64, now=cal._t0 + 4.0)

    assert cal.choose("mk_t4_b16", True) == (True, "mk_t4_b16", "mk")
    res = cal.record(12.0, padded_tokens=16, now=cal._t0 + 11.0)
    assert res is not None
    elapsed, _, _ = res
    assert elapsed >= 10.0
    cum = cal.summary_dict(window=False)
    assert cum["mk_t4_b16"]["mk"] == [2, 11.0, 10.2, 11.0, 11.8]
    assert cum["mk_t4_b16"]["stock"] == [1, 14.0, 14.0, 14.0, 14.0]
    assert cum["mk_t4_b16"]["stock_p16"] == [1, 14.0, 14.0, 14.0, 14.0]
    assert cum["stock_t4_n13"]["stock"] == [1, 19.5, 19.5, 19.5, 19.5]
    assert cum["stock_t4_n13"]["stock_p64"] == [1, 19.5, 19.5, 19.5, 19.5]
    assert cal.summary_dict(window=True) == {}

    cal.reset()
    assert cal.counts == {}
    assert cal.summary_dict(window=False) == {}

    mk = bare_mk(
        monkeypatch,
        tp_rank=0,
        _step_mix={},
        _step_mix_t0=time.monotonic(),
        _step_mix_period_s=1e9,
        ab_calibrator=VA.MKABCalibrator(enabled=True, period_s=1e9),
    )
    assert mk._note_step("mk_t4_b16", True) is True
    assert mk._note_step("mk_t4_b16", True) is False
    assert mk._note_step("stock_t4_n13", False) is False
    assert mk._step_mix == {"mk_t4_b16": 1, "ab_stock_t4_b16": 1, "stock_t4_n13": 1}
    mk.record_ab_step(12.5, padded_tokens=16)
    mk.reset_stats()
    assert mk._step_mix == {} and mk.ab_calibrator.counts == {}
