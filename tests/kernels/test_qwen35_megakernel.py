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
"""CPU unit tests for the Qwen3.5-397B-A17B-FP8 decode/verify megakernel adapter."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import (
    MKABCalibrator,
    Qwen35Megakernel,
    pack_att_in_layer,
    pack_att_out_layer,
    pack_expert_views_layer,
    pack_lin_in_layer,
    pack_lin_out_layer,
    pack_router_and_shared_layer,
    pack_smalls_layer,
    pack_weights,
    parse_mk_ab_enabled,
    parse_mk_buckets,
    parse_mk_fused_pool_io,
    parse_mk_verify_buckets,
    prepare_step_metadata,
    reshard_layer_experts_ep_to_tp,
    select_bucket,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR


def _make_fp8_linear(in_dim: int, out_dim: int, flipped: bool = True) -> SimpleNamespace:
    if flipped:
        w = torch.randint(0, 120, (in_dim, out_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    else:
        w = torch.randint(0, 120, (out_dim, in_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    s = torch.rand((out_dim, 1), dtype=torch.float32) + 0.1
    ns = SimpleNamespace(
        weight=torch.nn.Parameter(w, requires_grad=False),
        weight_scale=torch.nn.Parameter(s, requires_grad=False),
    )
    setattr(ns, WEIGHT_FLIPPED_ATTR, flipped)
    return ns


def _make_tiny_backbone(c: K.MKConfig, e_slots: int) -> SimpleNamespace:
    layers = []
    for l in range(c.L):
        is_att = (l % c.att_every) == (c.att_every - 1)
        ltype = "full_attention" if is_att else "linear_attention"

        if is_att:
            self_attn = SimpleNamespace(
                q_proj=_make_fp8_linear(c.H, c.nh * 2 * c.HD),
                k_proj=_make_fp8_linear(c.H, c.HD),
                v_proj=_make_fp8_linear(c.H, c.HD),
                o_proj=_make_fp8_linear(c.nh * c.HD, c.H),
                q_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(c.HD, dtype=torch.bfloat16), requires_grad=False
                    )
                ),
                k_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(c.HD, dtype=torch.bfloat16), requires_grad=False
                    )
                ),
                attn=SimpleNamespace(layer_name=f"model.layers.{l}.self_attn.attn", kv_cache=[]),
            )
            linear_attn = None
        else:
            in_qkvz = _make_fp8_linear(c.H, c.QKV + c.Z)
            in_ba = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((2 * c.vh, c.H), dtype=torch.bfloat16) * 0.05,
                    requires_grad=False,
                )
            )
            linear_attn = SimpleNamespace(
                prefix=f"model.layers.{l}.linear_attn",
                in_proj_qkvz=in_qkvz,
                in_proj_ba=in_ba,
                out_proj=_make_fp8_linear(c.Z, c.H),
                conv1d=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn((c.QKV, 1, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                ),
                A_log=torch.nn.Parameter(
                    torch.randn(c.vh, dtype=torch.float32), requires_grad=False
                ),
                dt_bias=torch.nn.Parameter(
                    torch.randn(c.vh, dtype=torch.bfloat16), requires_grad=False
                ),
                norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(c.VD, dtype=torch.float32), requires_grad=False
                    )
                ),
                kv_cache=[],
            )
            self_attn = None

        w13 = torch.randint(
            0, 120, (e_slots, c.H, 2 * c.IR), dtype=torch.uint8
        ).view(torch.float8_e4m3fn)
        w2 = torch.randint(0, 120, (e_slots, c.IR, c.H), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        w13_s = torch.rand((e_slots, 1, 2 * c.IR), dtype=torch.float32) + 0.1
        w2_s = torch.rand((e_slots, 1, c.H), dtype=torch.float32) + 0.1
        experts = SimpleNamespace(
            w13_weight=torch.nn.Parameter(w13, requires_grad=False),
            w2_weight=torch.nn.Parameter(w2, requires_grad=False),
            w13_weight_scale=torch.nn.Parameter(w13_s, requires_grad=False),
            w2_weight_scale=torch.nn.Parameter(w2_s, requires_grad=False),
        )

        # Fused router + shared expert Cat_FP8_256
        fused_gu_cols = c.E + 2 * c.IS
        fused_w = torch.randint(
            0, 120, (c.H, fused_gu_cols), dtype=torch.uint8
        ).view(torch.float8_e4m3fn)
        fused_s = torch.rand((1, fused_gu_cols), dtype=torch.float32) + 0.1
        shared_expert = SimpleNamespace(
            down_proj=_make_fp8_linear(c.IS, c.H),
        )
        shared_expert_gate = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.randn((1, c.H), dtype=torch.bfloat16), requires_grad=False
            )
        )
        mlp = SimpleNamespace(
            experts=experts,
            _fused_gate_gu_fp8_weight=fused_w,
            _fused_gate_gu_fp8_scale=fused_s,
            shared_expert=shared_expert,
            shared_expert_gate=shared_expert_gate,
        )

        layer = SimpleNamespace(
            layer_type=ltype,
            self_attn=self_attn,
            linear_attn=linear_attn,
            mlp=mlp,
            input_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(c.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
            post_attention_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(c.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
        )
        layers.append(layer)

    return SimpleNamespace(
        layers=layers,
        norm=SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.randn(c.H, dtype=torch.bfloat16), requires_grad=False
            )
        ),
        embed_tokens=lambda ids: torch.zeros(
            (ids.shape[0], c.H), dtype=torch.bfloat16, device=ids.device
        ),
    )


def test_pack_weights_ep_and_tp():
    c_ep = dataclasses.replace(
        K.TINY_EP, stack_e=False, e_slots=20, paged_kv=True
    )
    tiles = K.TINY_TILES
    backbone = _make_tiny_backbone(c_ep, e_slots=20)
    packed_ep = pack_weights(backbone, c_ep, tp_rank=0, tiles=tiles)

    assert len(packed_ep["e13"]) == c_ep.L
    assert len(packed_ep["e2"]) == c_ep.L
    for l in range(c_ep.L):
        assert (
            packed_ep["e13"][l].data_ptr()
            == backbone.layers[l].mlp.experts.w13_weight.data_ptr()
        )
        assert (
            packed_ep["e2"][l].data_ptr()
            == backbone.layers[l].mlp.experts.w2_weight.data_ptr()
        )
    assert packed_ep["smalls_16x4096"].shape == (c_ep.L, 16, c_ep.H)
    assert packed_ep["es_8x8192"].shape == (c_ep.L, 20, 8, 8192)

    # Test EP -> TP in-place resharding
    c_tp = dataclasses.replace(K.TINY, stack_e=False, e_slots=0, paged_kv=True)
    for l in range(c_tp.L):
        reshard_layer_experts_ep_to_tp(
            backbone.layers[l].mlp,
            tp_rank=1,
            c=c_tp,
            all_gather_fn=lambda x: x.repeat(c_tp.R, *([1] * (x.ndim - 1))),
        )
    packed_tp = pack_weights(backbone, c_tp, tp_rank=1, tiles=tiles)
    assert packed_tp["e13"][0].shape[0] == c_tp.E
    assert packed_tp["e13"][0].shape[-1] == 2 * c_tp.IR
    assert packed_tp["e2"][0].shape[0] == c_tp.E


def test_prepare_step_metadata_t1_and_t4():
    num_reqs = 3
    max_pages = 16
    # 1D block_tables as produced by TPUModelRunner
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
        state_ids = torch.tensor([10 + g, 20 + g, 30 + g, 0, 0, 0, 0, 0], dtype=torch.int32)
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

    # Test T=4, B=16 (nseq_bkt = 4)
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
    assert roff4[0, 0].item() == 2
    assert roff4[0, 1].item() == 3
    assert roff4[0, 2].item() == 1
    assert roff4[0, 3].item() == 0

    # Test T=1, B=8 (nseq_bkt = 8)
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


def test_buckets_parsing():
    assert parse_mk_buckets("8,16,32,64") == (8, 16, 32, 64)
    assert parse_mk_buckets("0") == ()
    assert parse_mk_verify_buckets("16,32,48,64") == (16, 32, 48, 64)
    assert parse_mk_verify_buckets("0") == ()
    assert select_bucket(12, (16, 32, 48, 64)) == 16
    assert select_bucket(28, (16, 32, 48, 64)) == 32
    assert select_bucket(44, (16, 32, 48, 64)) == 48
    assert select_bucket(60, (16, 32, 48, 64)) == 64
    assert parse_mk_ab_enabled("1") is True
    assert parse_mk_ab_enabled("0") is False


def test_pack_weights_ecopy_placement(monkeypatch):
    from vllm_torchtpu import expert_placement

    c_ep = dataclasses.replace(
        K.TINY_EP, stack_e=False, e_slots=20, paged_kv=True
    )
    er, ew = c_ep.ER, c_ep.EW
    extra = ew - er
    tiles = K.TINY_TILES
    backbone = _make_tiny_backbone(c_ep, e_slots=ew)

    for l in range(c_ep.L):
        cores = [list(range(r * er, (r + 1) * er)) for r in range(c_ep.R)]
        for r in range(c_ep.R):
            for k in range(extra):
                r_src = (r + 1 + (k + l) % (c_ep.R - 1)) % c_ep.R
                cores[r].append(r_src * er + er - extra + k)
        backbone.layers[l].mlp.experts._ecopy_cores = cores

    packed = pack_weights(backbone, c_ep, tp_rank=0, tiles=tiles)
    lay, S = K.smalls_layout(c_ep)
    assert "ecopy" in lay
    o_ec, w_ec = lay["ecopy"]
    assert packed["smalls"].shape == (c_ep.L, 1, S)
    for l in range(c_ep.L):
        ec_row = packed["smalls"][l, 0, o_ec : o_ec + w_ec]
        assert int((ec_row > 0).sum().item()) == c_ep.R * extra

    # Also verify real Qwen3.5 copies64 JSON table via expert_placement
    table_path = expert_placement.auto_table(dcp_size=1, max_num_seqs=16)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", table_path)
    c_full72 = dataclasses.replace(K.FULL_EP, stack_e=False, e_slots=72, paged_kv=True)
    cores0 = expert_placement.cores(0)
    ec0 = K.encode_ecopy(c_full72, cores0)
    assert len(ec0) == 512
    assert sum(1 for x in ec0 if x > 0) == 64


def test_mk_ab_calibrator():
    cal = MKABCalibrator(enabled=True, period_s=10.0, cap=4)
    # Deterministic per-key alternation: even -> MK, odd -> stock
    assert cal.choose("mk_t4_b16", True) == (True, "mk_t4_b16", "mk")
    assert cal.last_decision == ("mk_t4_b16", "mk")
    cal.record(10.0, padded_tokens=16, now=cal._t0 + 1.0)

    assert cal.choose("mk_t4_b16", True) == (False, "ab_stock_t4_b16", "stock")
    assert cal.last_decision == ("mk_t4_b16", "stock")
    cal.record(14.0, padded_tokens=16, now=cal._t0 + 2.0)

    # Independent counter per key
    assert cal.choose("mk_t4_b8", True) == (True, "mk_t4_b8", "mk")
    cal.record(8.0, padded_tokens=16, now=cal._t0 + 3.0)

    # Non-eligible step always takes stock
    assert cal.choose("stock_t4_n13", False) == (False, "stock_t4_n13", "stock")
    cal.record(19.5, padded_tokens=64, now=cal._t0 + 4.0)

    # Trigger period flush
    assert cal.choose("mk_t4_b16", True) == (True, "mk_t4_b16", "mk")
    res = cal.record(12.0, padded_tokens=16, now=cal._t0 + 11.0)
    assert res is not None
    elapsed, win_json, cum_json = res
    assert elapsed >= 10.0
    cum = cal.summary_dict(window=False)
    assert cum["mk_t4_b16"]["mk"] == [2, 11.0, 10.2, 11.0, 11.8]
    assert cum["mk_t4_b16"]["stock"] == [1, 14.0, 14.0, 14.0, 14.0]
    assert cum["mk_t4_b16"]["stock_p16"] == [1, 14.0, 14.0, 14.0, 14.0]
    assert cum["stock_t4_n13"]["stock"] == [1, 19.5, 19.5, 19.5, 19.5]
    assert cum["stock_t4_n13"]["stock_p64"] == [1, 19.5, 19.5, 19.5, 19.5]
    # Window was cleared on flush, cumulative preserved
    assert cal.summary_dict(window=True) == {}

    # Reset clears everything
    cal.reset()
    assert cal.counts == {}
    assert cal.summary_dict(window=False) == {}


def test_fused_pool_io_parsing_and_hlo_guard():
    assert parse_mk_fused_pool_io(None) is True
    assert parse_mk_fused_pool_io("1") is True
    assert parse_mk_fused_pool_io("true") is True
    assert parse_mk_fused_pool_io("0") is False
    assert parse_mk_fused_pool_io("false") is False
    assert parse_mk_fused_pool_io("off") is False

    good_hlo = """
    HloModule fused_mk
    ENTRY main {
      %p0 = f8e4m3fn[4096,2,64,4,128]{4,3,2,1,0:T(8,128)} parameter(0)
      %call = bf16[16,4096]{1,0:T(8,128)S(1)} custom-call(%p0), custom_call_target="tpu_custom_call"
      ROOT %tup = (bf16[16,4096]{1,0:T(8,128)}, f8e4m3fn[4096,2,64,4,128]{4,3,2,1,0:T(8,128)}) tuple(%call, %p0)
    }
    """
    ok, reason = pool_io.check_fused_pool_hlo(good_hlo, page_size=128)
    assert ok is True
    assert reason == "ok"

    bad_vmem_hlo = """
    HloModule bad_mk
    ENTRY main {
      %p0 = f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)} parameter(0)
      %cs = (f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)S(1)}, f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)}, u32[]{:S(2)}) copy-start(%p0)
      %cd = f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)S(1)} copy-done(%cs)
      %call = bf16[16,4096]{1,0:T(8,128)S(1)} custom-call(%cd), custom_call_target="tpu_custom_call"
      ROOT %tup = (bf16[16,4096]{1,0:T(8,128)}, f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)}) tuple(%call, %p0)
    }
    """
    ok_bad, reason_bad = pool_io.check_fused_pool_hlo(bad_vmem_hlo, page_size=128)
    assert ok_bad is False
    assert "copy-start" in reason_bad or "S(1)" in reason_bad


def test_amask_default_and_nan_kv_padding_guard():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    assert K.Tiles().amask == 1
    assert K.DEFAULT_TILES.amask == 1
    assert K.VERIFY_TILES.amask == 1
    assert K.WIDE_TILES.amask == 1
    assert K.TINY_TILES.amask == 1
    assert K.default_tiles(K.FULL_EP).amask == 1

    # Single-rank TINY config with paged_kv=True to test 0x7f (FP8 NaN) in KV slots
    c = dataclasses.replace(
        K.TINY_EP,
        R=1,
        B=4,
        T=2,
        L=4,
        att_every=4,
        NH=1,
        NKV=1,
        KH=1,
        VH=2,
        I=128,
        IS=128,
        E=128,
        TOPK=2,
        e_slots=128,
        paged_kv=True,
        num_pages=128,
        page_size=128,
        max_pages_per_seq=8,
        ctx=1024,
    )
    tiles_am1 = dataclasses.replace(
        K.TINY_TILES,
        amask=1,
        hier=0,
        pack=0,
        mb8=0,
        kv_blk=256,
        nkv=2,
    )
    tiles_am4 = dataclasses.replace(tiles_am1, amask=4)

    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    pt, ckpt, roff, _ = pool_io.default_pool_layout(c)
    starts = [10, 20]
    pos = jnp.asarray(
        [s0 + t for s0 in starts for t in range(c.T)], dtype=jnp.int32
    )
    hid = jax.random.normal(jax.random.PRNGKey(0), (c.M, c.H), jnp.float32)
    hid = jnp.where(jnp.arange(c.M)[:, None] < c.B, hid, 0.0).astype(
        jnp.bfloat16
    )

    backbone = _make_tiny_backbone(c, e_slots=c.EW)
    packed = pack_weights(backbone, c, tp_rank=0, tiles=tiles_am1)
    w_flat = [
        jnp.asarray(packed[k].view(torch.uint8).numpy()).view(
            K.FP8
            if packed[k].dtype == torch.float8_e4m3fn
            else (
                jnp.bfloat16
                if packed[k].dtype == torch.bfloat16
                else jnp.float32
            )
        )
        for k in K.W_NAMES
    ]

    # Clean zero pool vs 0x7f-contaminated pool (unwritten col >= rem is 0x7f = FP8 NaN)
    clean_u8 = np.zeros((c.num_pages, 2, 64, 4, c.page_size), dtype=np.uint8)
    nan_u8 = np.full((c.num_pages, 2, 64, 4, c.page_size), 0x7F, dtype=np.uint8)
    for s, s0 in enumerate(starts):
        kp = int(pt[s, 0])
        clean_u8[kp, :, :, :, :s0] = 0x38  # 0.5 in f8e4m3fn
        nan_u8[kp, :, :, :, :s0] = 0x38
    for g_sub in range(ckpt.shape[0]):
        for s in range(len(starts)):
            blk_id = int(ckpt[g_sub, s, 0])
            if blk_id > 0:
                p0 = blk_id * 5
                clean_u8[p0 : p0 + 5] = 0x00
                nan_u8[p0 : p0 + 5] = 0x00

    raw_clean = [
        jax.lax.bitcast_convert_type(jnp.asarray(clean_u8), K.FP8)
        for _ in range(c.NATT)
    ]
    raw_nan_am1 = [
        jax.lax.bitcast_convert_type(jnp.asarray(nan_u8), K.FP8)
        for _ in range(c.NATT)
    ]
    raw_nan_am4 = [
        jax.lax.bitcast_convert_type(jnp.asarray(nan_u8), K.FP8)
        for _ in range(c.NATT)
    ]

    skip_exp = frozenset({"experts"})
    fn_clean, _, _, _ = pool_io.make_flat_pooled_step(
        mesh, c, tiles_am1, skip=skip_exp, fused_pool_io=True, interpret=True
    )
    fn_am1, _, _, _ = pool_io.make_flat_pooled_step(
        mesh, c, tiles_am1, skip=skip_exp, fused_pool_io=True, interpret=True
    )
    fn_am4, _, _, _ = pool_io.make_flat_pooled_step(
        mesh, c, tiles_am4, skip=skip_exp, fused_pool_io=True, interpret=True
    )

    out_clean = fn_clean(pos, hid, *w_flat, *raw_clean, pt, ckpt, roff)[0]
    out_am1 = fn_am1(pos, hid, *w_flat, *raw_nan_am1, pt, ckpt, roff)[0]
    out_am4 = fn_am4(pos, hid, *w_flat, *raw_nan_am4, pt, ckpt, roff)[0]

    assert int(jnp.isnan(out_am1[: c.B]).sum()) == 0
    diff_am1 = jnp.abs(
        out_clean[: c.B].astype(jnp.float32)
        - out_am1[: c.B].astype(jnp.float32)
    )
    assert float(jnp.max(diff_am1)) == 0.0
    assert int(jnp.isnan(out_am4[: c.B]).sum()) > 0


