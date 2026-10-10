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
"""Unit tests for Qwen3.5 megakernel `PagedAttentionBlock`, `KVPoolDMA`, `amask`, and MTP attention."""

from __future__ import annotations

import dataclasses

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import torch

from vllm_torchtpu.kernels.qwen35_megakernel import (
    KVPoolDMA,
    PagedAttentionBlock,
    WeightRings,
    megakernel as K,
    pool_io,
)
from vllm_torchtpu.kernels.qwen35_megakernel.megakernel import BF16, F32, rnd
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import pack_weights

from qwen35_megakernel_test_utils import (
    jnp_leaves,
    make_test_ctx_and_inputs,
    make_tiny_backbone,
    ref_dmatmul,
    rms_ref,
    setup_8_cpu_devices,
)


def test_standalone_paged_attention_block_interpret():
    """Test PagedAttentionBlock standalone in pl.pallas_call(interpret=True) vs pure-JAX reference."""
    c, _, ctx, w_sh, st_sh, lay, w_vals, st_vals, pos, hid = (
        make_test_ctx_and_inputs()
    )
    mesh1 = Mesh(np.array(setup_8_cpu_devices()[:1]), ("tp",))
    l_att = c.att_every - 1
    g0_att = (c.att_every - 1) * ctx.t_gdn

    def att_kernel(*refs):
        ctx.bind_refs(refs)
        rings = WeightRings(ctx)
        kv_dma = KVPoolDMA(ctx)
        att_blk = PagedAttentionBlock(ctx, rings, kv_dma)
        scr = ctx.scr
        for i in range(ctx.t_att):
            rings.fetch(jnp.int32(g0_att + i))
        cp_sm = pltpu.make_async_copy(
            ctx.W["smalls"].at[l_att], scr["sm"].at[0], scr["sm_sem"].at[0]
        )
        cp_sm.start()
        cp_sm.wait()
        att_blk.init_worklist()
        scr["res"][...] = jnp.where(ctx.live, ctx.hid_ref[: ctx.MB].astype(F32), 0.0)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        att_blk.init_rope()
        scr["xb"][...] = ctx.rms_zc(scr["res"][...], ctx.sm(0, "in_norm")).astype(BF16)
        yo = att_blk(jnp.int32(l_att), jnp.int32(0), jnp.int32(g0_att))
        ctx.out_ref[: ctx.MB] = yo.astype(BF16)

    call = pl.pallas_call(
        att_kernel,
        out_shape=ctx.out_shape,
        in_specs=ctx.in_specs,
        out_specs=ctx.out_specs,
        scratch_shapes=[ctx.scratch[k] for k in ctx.s_names],
        interpret=pltpu.InterpretParams(),
    )
    args = (pos, hid, *[w_vals[k] for k in w_sh], *[st_vals[k] for k in st_sh])
    hid_att, _, _, knew_att, vnew_att = jax.shard_map(
        lambda *a: call(*a),
        mesh=mesh1,
        in_specs=tuple(P() for _ in args),
        out_specs=tuple(P() for _ in ctx.out_shape),
        check_vma=False,
    )(*args)

    def sm_val(l_idx, name, width=None):
        o, n = lay[name]
        w = n if width is None else width
        return w_vals["smalls"][l_idx, :, o : o + w]

    xb_att = rms_ref(hid[: c.B].astype(F32), sm_val(l_att, "in_norm"), c.eps).astype(
        BF16
    )
    y_att = rnd(
        ref_dmatmul(xb_att, w_vals["att_in"][0], ctx.MB) * sm_val(l_att, "in_s", c.NAI)
    )
    qn = sm_val(l_att, "qn")
    kn = sm_val(l_att, "kn")
    half = c.ROT // 2
    inv = 1.0 / (c.theta ** (jnp.arange(half, dtype=F32)[None, :] * 2.0 / c.ROT))

    def rope_ref(x, cos, sin):
        x1, x2, rest = x[:, :half], x[:, half : c.ROT], x[:, c.ROT :]
        return jnp.concatenate(
            (
                rnd(rnd(x1 * cos) - rnd(x2 * sin)),
                rnd(rnd(x2 * cos) + rnd(x1 * sin)),
                rest,
            ),
            axis=-1,
        )

    koff = c.nh * 2 * c.HD
    att_mix_list, knew_list, vnew_list = [], [], []
    for b in range(c.B):
        pb = pos[b]
        ang = pb.astype(F32) * inv
        cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
        qg = y_att[b, :koff].reshape(c.nh, 2 * c.HD)
        q = rms_ref(qg[:, : c.HD], qn, c.eps)
        q = rnd(rope_ref(q, cos, sin) * (c.HD**-0.5)).astype(BF16).astype(F32)
        gate = rnd(jax.nn.sigmoid(qg[:, c.HD :]))
        k_new = rope_ref(
            rms_ref(y_att[b : b + 1, koff : koff + c.HD], kn, c.eps), cos, sin
        )
        v_new = y_att[b : b + 1, koff + c.HD : koff + 2 * c.HD]
        knew_list.append(k_new[0])
        vnew_list.append(v_new[0])
        kc_b = st_vals["kc"][0, b, : int(pb)].astype(F32)
        vc_b = st_vals["vc"][0, b, : int(pb)].astype(F32)
        k_all = jnp.concatenate([kc_b, k_new], axis=0)
        v_all = jnp.concatenate([vc_b, v_new], axis=0)
        scores = q @ k_all.T
        probs = jax.nn.softmax(scores, axis=-1)
        out = rnd(probs @ v_all)
        att_mix_list.append(rnd(out * gate).reshape(c.nh * c.HD))
    ref_knew = jnp.stack(knew_list, axis=0)
    ref_vnew = jnp.stack(vnew_list, axis=0)
    ref_att_mix = jnp.stack(att_mix_list, axis=0).astype(BF16)
    ref_att_yo = (
        ref_dmatmul(ref_att_mix, w_vals["att_out"][0], ctx.MB) * sm_val(l_att, "out_s")
    ).astype(BF16)

    assert float(jnp.max(jnp.abs(knew_att[0, : c.B] - ref_knew))) == 0.0
    assert float(jnp.max(jnp.abs(vnew_att[0, : c.B] - ref_vnew))) == 0.0
    assert (
        float(jnp.max(jnp.abs(hid_att[: c.B].astype(F32) - ref_att_yo.astype(F32))))
        <= 2e-3
    )


def test_amask_default_and_nan_kv_padding_guard():
    assert K.Tiles().amask == 1
    assert K.DEFAULT_TILES.amask == 1
    assert K.VERIFY_TILES.amask == 1
    assert K.WIDE_TILES.amask == 1
    assert K.TINY_TILES.amask == 1
    assert K.default_tiles(K.FULL_EP).amask == 1

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

    mesh = Mesh(np.array(setup_8_cpu_devices()[:1]), ("tp",))
    pt, ckpt, roff, _ = pool_io.default_pool_layout(c)
    starts = [10, 20]
    pos = jnp.asarray(
        [s0 + t for s0 in starts for t in range(c.T)], dtype=jnp.int32
    )
    hid = jax.random.normal(jax.random.PRNGKey(0), (c.M, c.H), jnp.float32)
    hid = jnp.where(jnp.arange(c.M)[:, None] < c.B, hid, 0.0).astype(jnp.bfloat16)

    backbone = make_tiny_backbone(c, e_slots=c.EW)
    packed = pack_weights(backbone, c, tp_rank=0, tiles=tiles_am1)
    w_flat = jnp_leaves(packed)

    clean_u8 = np.zeros((c.num_pages, 2, 64, 4, c.page_size), dtype=np.uint8)
    nan_u8 = np.full((c.num_pages, 2, 64, 4, c.page_size), 0x7F, dtype=np.uint8)
    for s, s0 in enumerate(starts):
        kp = int(pt[s, 0])
        clean_u8[kp, :, :, :, :s0] = 0x38
        nan_u8[kp, :, :, :, :s0] = 0x38
    for g_sub in range(ckpt.shape[0]):
        for s in range(len(starts)):
            blk_id = int(ckpt[g_sub, s, 0])
            if blk_id > 0:
                p0 = blk_id * c.gdn_pages
                clean_u8[p0 : p0 + c.gdn_pages] = 0x00
                nan_u8[p0 : p0 + c.gdn_pages] = 0x00

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
        out_clean[: c.B].astype(jnp.float32) - out_am1[: c.B].astype(jnp.float32)
    )
    assert float(jnp.max(diff_am1)) == 0.0
    assert int(jnp.isnan(out_am4[: c.B]).sum()) > 0


def test_mtp_attention_only_step_verify_and_loop_rows_agree():
    """MTP draft layer (`make_flat_attn_step`): a T=4 verify step writes only the active
    sequence's 4 new K/V columns, and a T=1 loop step over the updated pool reproduces the
    verify step's output for the last token."""
    c4 = dataclasses.replace(
        K.TINY_EP,
        R=1,
        B=8,
        T=4,
        L=1,
        att_every=1,
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
        num_pages=40,
        page_size=128,
        max_pages_per_seq=8,
        ctx=1024,
    )
    c1 = dataclasses.replace(c4, T=1)
    tiles = dataclasses.replace(K.default_tiles(c4), kv_blk=512, kv_sub=256)
    torch.manual_seed(0)
    w = jnp_leaves(pack_weights(make_tiny_backbone(c4, e_slots=c4.EW), c4, 0, tiles))
    mesh = Mesh(np.array(setup_8_cpu_devices()[:1]), ("tp",))
    skip = frozenset({"experts"})

    rng = np.random.default_rng(0)
    pool_u8 = rng.integers(
        0, 0x40, (c4.num_pages, 2, 64, 4, c4.page_size), dtype=np.uint8
    )
    pool_u8 |= (rng.integers(0, 2, pool_u8.shape, dtype=np.uint8) << 7).astype(
        np.uint8
    )
    pool0 = jax.lax.bitcast_convert_type(jnp.asarray(pool_u8), K.FP8)
    pt4 = jnp.asarray([list(range(8, 16)), list(range(16, 24))], jnp.int32)
    p0, p1 = 300, 700
    pos4 = jnp.asarray(
        [p0 + t for t in range(4)] + [p1 + t for t in range(4)], jnp.int32
    )
    hid4 = jax.random.normal(jax.random.PRNGKey(1), (c4.M, c4.H), jnp.float32)
    hid4 = jnp.where(jnp.arange(c4.M)[:, None] < c4.B, hid4, 0.0).astype(jnp.bfloat16)
    ones = jnp.ones((c4.B,), jnp.int32)

    fn4, _, _, _ = pool_io.make_flat_attn_step(
        mesh, c4, tiles, interpret=True, skip=skip
    )
    out4, pool4 = fn4(
        pos4, hid4, *w, pool0, pt4, jnp.asarray([1, 0], jnp.int32), ones
    )
    pool4_u8 = np.asarray(jax.lax.bitcast_convert_type(pool4, jnp.uint8))
    assert bool(jnp.isfinite(out4[:4].astype(jnp.float32)).all())
    changed = np.argwhere(np.any(pool4_u8 != pool_u8, axis=(1, 2, 3)))
    assert changed.tolist() == [[10, 44], [10, 45], [10, 46], [10, 47]]

    pt1 = jnp.zeros((c1.NS, c1.max_pages_per_seq), jnp.int32).at[0].set(pt4[0])
    pos1 = jnp.zeros((c1.B,), jnp.int32).at[0].set(p0 + 3)
    hid1 = jnp.zeros_like(hid4).at[0].set(hid4[3])
    act1 = jnp.zeros((c1.NS,), jnp.int32).at[0].set(1)
    fn1, _, _, _ = pool_io.make_flat_attn_step(
        mesh, c1, tiles, interpret=True, skip=skip
    )
    out1, pool1 = fn1(pos1, hid1, *w, pool4, pt1, act1, ones)
    assert np.array_equal(
        np.asarray(jax.lax.bitcast_convert_type(pool1, jnp.uint8)), pool4_u8
    )
    np.testing.assert_allclose(
        np.asarray(out1[0], np.float32),
        np.asarray(out4[3], np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
