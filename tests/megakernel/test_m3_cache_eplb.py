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
"""M3a tests: Paged KV in `GQABlock`, 128K `page_table` SMEM, and KV page helpers."""

from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel.models import qwen35


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _cpu_mesh(R: int) -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _tiny_geom(
    *,
    L: int = 4,
    att_every: int = 4,
    NKV: int = 2,
    moe_tp: bool = False,
    e_slots: int = 0,
) -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=L,
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=NKV,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        att_every=att_every,
        moe_tp=moe_tp,
        e_slots=e_slots,
    )


def _tiny_exec(
    *,
    R: int = 2,
    B: int = 2,
    T: int = 1,
    ctx: int = 256,
    paged_kv: bool = True,
    num_pages: int = 64,
    max_pages_per_seq: int = 0,
    stack_e: bool = True,
    ecopy_opt: int = 0,
    amask: int = 1,
    kv_sub: int = 0,
) -> mk.ExecSpec:
    tiles = replace(
        mk.spec.TINY_TILES,
        bank=512 << 10,
        kv_blk=min(mk.spec.TINY_TILES.kv_blk, ctx),
        kv_sub=kv_sub,
        amask=amask,
        ecopy_opt=ecopy_opt,
    )
    return mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        paged_kv=paged_kv,
        num_pages=num_pages,
        max_pages_per_seq=max_pages_per_seq,
        stack_e=stack_e,
        interpret=True,
        tiles=tiles,
    )


def _pack_kv_to_raw_pools(
    g: qwen35.Qwen35Geometry,
    ex: mk.ExecSpec,
    kc: jax.Array,
    vc: jax.Array,
    *,
    pt: np.ndarray | None = None,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Pack `[R, NATT, NS, ctx_tokens, HD]` `kc`/`vc` into `g.NATT` raw pools."""
    ctx_tokens = kc.shape[3]
    pages_per_seq = ctx_tokens // ex.page_size
    pt_cols = max(ex.resolved_max_pages_per_seq, pages_per_seq)
    if pt is None:
        pt = np.zeros((ex.NS, pt_cols), dtype=np.int32)
        rng = np.random.default_rng(1337)
        # Reserve pages 0..3 as unreferenced sentinels and permute remaining pages
        perm = rng.permutation(ex.num_pages - 4).astype(np.int32) + 4
        for s in range(ex.NS):
            pt[s, :pages_per_seq] = perm[s * pages_per_seq : (s + 1) * pages_per_seq]

    raw_list = []
    for g_idx in range(g.NATT):
        pool_g = np.zeros(
            (ex.R, ex.num_pages, 2, g.HD // 4, 4, ex.page_size),
            dtype=jnp.dtype(jnp.float8_e4m3fn),
        )
        kc_g = np.asarray(
            jnp.clip(kc[:, g_idx], -448.0, 448.0).astype(jnp.float8_e4m3fn)
        )
        vc_g = np.asarray(
            jnp.clip(vc[:, g_idx], -448.0, 448.0).astype(jnp.float8_e4m3fn)
        )
        for r in range(ex.R):
            for s in range(ex.NS):
                for p in range(pages_per_seq):
                    kp = int(pt[s, p])
                    t0 = p * ex.page_size
                    k_sl = kc_g[r, s, t0 : t0 + ex.page_size]
                    v_sl = vc_g[r, s, t0 : t0 + ex.page_size]
                    pool_g[r, kp] = np.asarray(mk.kv_to_raw_page(k_sl, v_sl))
        raw_list.append(
            jnp.asarray(
                pool_g.reshape(ex.R * ex.num_pages, 2, g.HD // 4, 4, ex.page_size)
            )
        )
    return tuple(raw_list), jnp.asarray(pt)


def test_raw_page_pack_unpack_roundtrip():
    """Verify per-page KV, SSM, and conv raw page pack/unpack helpers round-trip."""
    key = jax.random.key(101)
    k1, k2 = jax.random.split(key, 2)

    # 1. KV page [128, 256] fp8 -> [2, 64, 4, 128] fp8 -> [128, 256] fp8
    k_page = (jax.random.normal(k1, (128, 256), dtype=jnp.float32) * 0.2).astype(
        jnp.float8_e4m3fn
    )
    v_page = (jax.random.normal(k2, (128, 256), dtype=jnp.float32) * 0.2).astype(
        jnp.float8_e4m3fn
    )
    raw_kv = mk.kv_to_raw_page(k_page, v_page)
    assert raw_kv.shape == (2, 64, 4, 128)
    k_rt, v_rt = mk.raw_page_to_kv(raw_kv)
    np.testing.assert_array_equal(
        np.asarray(k_rt).view(np.uint8), np.asarray(k_page).view(np.uint8)
    )
    np.testing.assert_array_equal(
        np.asarray(v_rt).view(np.uint8), np.asarray(v_page).view(np.uint8)
    )

    # 2. SSM state [vh=2, KD=128, VD=128] bf16 -> [4, 2, 64, 4, 128] fp8 -> bf16
    rec = (jax.random.normal(k1, (2, 128, 128), dtype=jnp.float32) * 0.1).astype(
        jnp.bfloat16
    )
    raw_ssm = mk.ssm_to_raw_pages(rec)
    assert raw_ssm.shape == (4, 2, 64, 4, 128)
    rec_rt = mk.raw_pages_to_ssm(raw_ssm, vh=2, KD=128, VD=128)
    np.testing.assert_array_equal(np.asarray(rec_rt), np.asarray(rec))

    # 3. Conv state [3, QKV=512] f32 -> [2, 64, 4, 128] fp8 -> [3, 512] f32
    conv = (
        (jax.random.normal(k1, (3, 512), dtype=jnp.float32) * 0.1)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    raw_conv = mk.conv_to_raw_page(conv)
    assert raw_conv.shape == (2, 64, 4, 128)
    conv_rt = mk.raw_page_to_conv(raw_conv, QKV=512)
    np.testing.assert_array_equal(np.asarray(conv_rt), np.asarray(conv))


@pytest.mark.parametrize("att_every,use_2d_ckpt", [(4, False), (3, True)])
def test_states_to_pool_roundtrip(att_every: int, use_2d_ckpt: bool):
    """Verify `states_to_pool` <-> `pool_to_states` round-trip for 3D and 2D ckpt."""
    L = 4 if att_every == 4 else 6
    g = _tiny_geom(L=L, att_every=att_every)
    ex = _tiny_exec(R=2, B=2, T=1, ctx=256, paged_kv=True, num_pages=96)
    _, _, _, st_l = qwen35.random_logical_inputs(jax.random.key(7 + att_every), g, ex)

    st_in = dict(st_l)
    st_in["kc"] = st_l["kc"].astype(jnp.float8_e4m3fn)
    st_in["vc"] = st_l["vc"].astype(jnp.float8_e4m3fn)
    st_in["conv"] = st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32)

    pt, ckpt, roff, min_np = mk.default_pool_layout(g, ex)
    assert ex.num_pages >= min_np
    # Pages 0..5*G_PER_ATT-1 are reserved; active sequences start at page >= 5*G_PER_ATT
    assert int(jnp.min(pt[:, : ex.ctx // ex.page_size])) >= 5 * (att_every - 1)
    assert int(jnp.min(ckpt)) >= 1
    if use_2d_ckpt:
        ckpt = ckpt[0]
        roff = roff[0]

    raw_pools = mk.states_to_pool(g, ex, st_in, pt, ckpt, roff)
    st_out = mk.pool_to_states(g, ex, raw_pools, pt, ckpt)

    np.testing.assert_array_equal(
        np.asarray(st_out["kc"]).view(np.uint8),
        np.asarray(st_in["kc"]).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(st_out["vc"]).view(np.uint8),
        np.asarray(st_in["vc"]).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(st_out["conv"][:, :, :, 0]), np.asarray(st_in["conv"])
    )
    vh = g.vh(ex.R)
    np.testing.assert_array_equal(
        np.asarray(st_out["rec"].reshape(ex.R, g.NGDN, ex.NS, vh, g.KD, g.VD)),
        np.asarray(st_in["rec"]),
    )


@pytest.mark.parametrize(
    "T,NS,att_every",
    [(1, 1, 4), (1, 2, 4), (1, 3, 3), (1, 4, 4), (1, 5, 3), (2, 2, 4)],
)
def test_gather_gdn_states_cpu_interpret(T: int, NS: int, att_every: int):
    """Verify `gather_gdn_states` across `NS=1..5` (`n_pairs>1`) and `from_ckpt`."""
    L = 4 if att_every == 4 else 6
    B = NS * T
    g = _tiny_geom(L=L, att_every=att_every)
    ex = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=96)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(15 + T + NS), g, ex
    )
    st_quant = dict(
        st_l,
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )
    _, _, _, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_quant, g, ex, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    # Test 2D ckpt_indices when att_every == 3 (G_PER_ATT = 2 != 3)
    ckpt_use = ckpt[0] if att_every == 3 else ckpt
    roff_use = roff[0] if att_every == 3 else roff
    raw_pools = mk.states_to_pool(g, ex, st_quant, pt, ckpt_use, roff_use)
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in raw_pools)

    conv_g, rec_g = mk.gather_gdn_states(g, ex, raw_r0, ckpt_use, roff_use)
    conv_exp = mk.unfold(st_mk["conv"], ex.R)[0]
    rec_exp = mk.unfold(st_mk["rec"], ex.R)[0]
    np.testing.assert_array_equal(np.asarray(conv_g), np.asarray(conv_exp))
    np.testing.assert_array_equal(np.asarray(rec_g), np.asarray(rec_exp))

    if NS >= 2:
        # 1. Explicit active_seqs mask: zero out sequence 1
        act_mask = jnp.asarray([1] + [0] + [1] * (NS - 2), dtype=jnp.int32)
        _, rec_act = mk.gather_gdn_states(g, ex, raw_r0, ckpt_use, roff_use, act_mask)
        vh = g.vh(ex.R)
        rec_act_seq = rec_act.reshape(g.NGDN, ex.NS, vh, g.KD, g.VD)
        np.testing.assert_array_equal(
            np.asarray(rec_act_seq[:, 0]),
            np.asarray(rec_exp.reshape(g.NGDN, ex.NS, vh, g.KD, g.VD)[:, 0]),
        )
        np.testing.assert_array_equal(
            np.asarray(rec_act_seq[:, 1]),
            np.zeros_like(np.asarray(rec_act_seq[:, 1])),
        )
        # 2. active_seqs=None (default "from_ckpt") and explicit "from_ckpt":
        # setting sequence 1's slot to 0 marks it inactive
        ckpt_inact = (
            ckpt_use.at[:, 1, :].set(0)
            if ckpt_use.ndim == 3
            else ckpt_use.at[1, :].set(0)
        )
        _, rec_fc = mk.gather_gdn_states(
            g, ex, raw_r0, ckpt_inact, roff_use, "from_ckpt"
        )
        _, rec_def = mk.gather_gdn_states(g, ex, raw_r0, ckpt_inact, roff_use)
        np.testing.assert_array_equal(np.asarray(rec_fc), np.asarray(rec_act))
        np.testing.assert_array_equal(np.asarray(rec_def), np.asarray(rec_act))
        with pytest.raises(ValueError, match="Unsupported active_seqs mode"):
            mk.gather_gdn_states(g, ex, raw_r0, ckpt_use, roff_use, "invalid")
        with pytest.raises(ValueError, match="maps to pages"):
            mk.gather_gdn_states(g, ex, raw_r0, ckpt_use + ex.num_pages, roff_use)
        if NS == 2 and T == 1 and att_every == 4:
            from jax.experimental import pallas as pl

            with (
                pl.enable_debug_checks(True),
                pytest.raises(
                    jax.errors.JaxRuntimeError,
                    match="GDN gather checkpoint slot maps outside",
                ),
            ):
                c_bad, _ = jax.jit(
                    lambda ck: mk.gather_gdn_states(g, ex, raw_r0, ck, roff_use)
                )(ckpt_use + ex.num_pages)
                c_bad.block_until_ready()


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_paged_gqa_make_step_cpu_interpret(T: int, B: int):
    """Verify `GQABlock` with `paged_kv=True`, `G=2`, `NATT=2`, and permuted pages."""
    mesh = _cpu_mesh(2)
    # L=8, att_every=4 -> NATT=2 attention pools; NKV=1 with R=2 -> G=2 (SPLIT=True)
    g = _tiny_geom(L=8, att_every=4, NKV=1)
    ex = _tiny_exec(R=2, B=B, T=T, ctx=512, paged_kv=True, num_pages=64)
    _, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(20 + T), g, ex)
    # Sequence 0 is in KV block 1 (page 2, pos >= 300); sequence 1 in block 0 (page 1)
    pos = (
        jnp.asarray([300, 150], dtype=jnp.int32)
        if T == 1
        else jnp.asarray([300, 301, 150, 151], dtype=jnp.int32)
    )

    ref_out, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l, paged_kv=True)

    raw_pools, pt = _pack_kv_to_raw_pools(g, ex, st_l["kc"], st_l["vc"])
    st_paged_l = {**st_l, "raw": raw_pools, "page_table": pt}
    _, hid_mk, w_mk, st_mk_paged = qwen35.pack_inputs(
        pos, hid_l, w_l, st_paged_l, g, ex, mesh=mesh
    )
    mk_paged_step, _ = qwen35.make_step(mesh, g, ex, donate_states=False)
    out_hid, mk_st = mk_paged_step(pos, hid_mk, w_mk, st_mk_paged)
    out_b = mk.unfold(out_hid, ex.R)[0, :B]

    # Also run unpaged megakernel to verify parity across all NATT=2 pools
    ex_unp = _tiny_exec(R=2, B=B, T=T, ctx=512, paged_kv=False)
    _, hid_unp, w_unp, st_mk_unp = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_unp, mesh=mesh
    )
    mk_unp_step, _ = qwen35.make_step(mesh, g, ex_unp, donate_states=False)
    out_unp, st_unp = mk_unp_step(pos, hid_unp, w_unp, st_mk_unp)
    assert jnp.allclose(
        out_hid.astype(jnp.float32), out_unp.astype(jnp.float32), atol=5e-2, rtol=5e-2
    )

    assert "raw" not in mk_st
    assert "page_table" not in mk_st
    assert mk_st["knew"].dtype == jnp.float8_e4m3fn
    assert mk_st["vnew"].dtype == jnp.float8_e4m3fn
    if T == 1:
        # Trace non-interpret TPU path (Ref.bitcast(U32).reshape(HD // 4, BLK)) on CPU
        mk_non_interp, _ = qwen35.make_step(
            mesh, g, replace(ex, interpret=False), donate_states=False
        )
        jaxpr_non_interp = jax.make_jaxpr(mk_non_interp)(pos, hid_mk, w_mk, st_mk_paged)
        assert "decode_megakernel" in str(jaxpr_non_interp)

        # Verify 3D page_table (R, NS, W) at R=2 and donate_states=True (read-only
        # raw pools and page_table must NOT be invalidated while conv/rec are donated)
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        tp_s = NamedSharding(mesh, P("tp"))
        pt_3d = jnp.broadcast_to(
            pt[None, :, : ex.resolved_max_pages_per_seq],
            (ex.R, ex.NS, ex.resolved_max_pages_per_seq),
        )
        st_don = {
            "conv": jax.device_put(np.asarray(st_mk_paged["conv"]).copy(), tp_s),
            "rec": jax.device_put(np.asarray(st_mk_paged["rec"]).copy(), tp_s),
            "raw": tuple(
                jax.device_put(np.asarray(r).copy(), tp_s) for r in st_mk_paged["raw"]
            ),
            "page_table": jax.device_put(np.asarray(pt_3d).copy(), tp_s),
        }
        mk_don_step, _ = qwen35.make_step(mesh, g, ex, donate_states=True)
        out_don, mk_st_don = mk_don_step(pos, hid_mk, w_mk, st_don)
        np.testing.assert_array_equal(np.asarray(out_don), np.asarray(out_hid))
        assert "raw" not in mk_st_don and "page_table" not in mk_st_don
        assert st_don["conv"].is_deleted() and st_don["rec"].is_deleted()
        assert not any(r.is_deleted() for r in st_don["raw"])
        assert not st_don["page_table"].is_deleted()
    knew_u = mk.unfold(mk_st["knew"], ex.R)
    vnew_u = mk.unfold(mk_st["vnew"], ex.R)
    kc_unp = mk.unfold(st_unp["kc"], ex.R)
    vc_unp = mk.unfold(st_unp["vc"], ex.R)
    assert knew_u.shape == (ex.R, g.NATT, ex.MR, g.HD)
    # Verify knew/vnew across all NATT=2 attention layers and both sequences
    for a_idx in range(g.NATT):
        for s_idx in range(ex.NS):
            b0 = s_idx * T
            p0 = int(pos[b0])
            got_k_fp8 = knew_u[:, a_idx, b0 : b0 + T]
            got_v_fp8 = vnew_u[:, a_idx, b0 : b0 + T]
            unp_k_fp8 = jnp.clip(
                kc_unp[:, a_idx, s_idx, p0 : p0 + T], -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            unp_v_fp8 = jnp.clip(
                vc_unp[:, a_idx, s_idx, p0 : p0 + T], -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            if a_idx == 0:
                assert jnp.array_equal(got_k_fp8, unp_k_fp8)
                assert jnp.array_equal(got_v_fp8, unp_v_fp8)
            ref_k = ref_st["kc"][:, a_idx, s_idx, p0 : p0 + T].astype(jnp.float32)
            ref_v = ref_st["vc"][:, a_idx, s_idx, p0 : p0 + T].astype(jnp.float32)
            assert jnp.allclose(
                got_k_fp8.astype(jnp.float32), ref_k, atol=2.5e-1, rtol=1.5e-1
            )
            assert jnp.allclose(
                got_v_fp8.astype(jnp.float32), ref_v, atol=2.5e-1, rtol=1.5e-1
            )

    assert jnp.allclose(
        out_b.astype(jnp.float32), ref_out.astype(jnp.float32), atol=5e-2, rtol=5e-2
    )


@pytest.mark.parametrize("amask", [0, 1, 2])
def test_paged_kv_fp8_nan_padding_guard(amask: int):
    """Verify `amask in (0, 1, 2)` masks out `0x7f` (`NaN`) via the `uint32` path."""
    mesh = _cpu_mesh(2)
    g = _tiny_geom(L=2, att_every=1)
    ex = _tiny_exec(
        R=2, B=2, T=1, ctx=256, paged_kv=True, num_pages=32, amask=amask, kv_sub=128
    )
    _, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(44), g, ex)
    pos = jnp.asarray([180, 60], dtype=jnp.int32)

    nan_fp8 = jax.lax.bitcast_convert_type(jnp.uint8(0x7F), jnp.float8_e4m3fn)
    assert np.isnan(np.asarray(nan_fp8, dtype=np.float32))

    kc_fp8 = st_l["kc"].astype(jnp.float8_e4m3fn)
    vc_fp8 = st_l["vc"].astype(jnp.float8_e4m3fn)
    idx = jnp.arange(ex.ctx)[None, None, None, :, None]
    plen = pos[None, None, :, None, None]
    kc_nan = jnp.where(idx < plen, kc_fp8, nan_fp8)
    vc_nan = jnp.where(idx < plen, vc_fp8, nan_fp8)

    raw_clean, pt = _pack_kv_to_raw_pools(g, ex, st_l["kc"], st_l["vc"])
    raw_nan, _ = _pack_kv_to_raw_pools(g, ex, kc_nan, vc_nan, pt=np.asarray(pt))

    _, hid_mk, w_mk, st_clean_mk = qwen35.pack_inputs(
        pos,
        hid_l,
        w_l,
        {**st_l, "raw": raw_clean, "page_table": pt},
        g,
        ex,
        mesh=mesh,
    )
    _, _, _, st_nan_mk = qwen35.pack_inputs(
        pos,
        hid_l,
        w_l,
        {**st_l, "raw": raw_nan, "page_table": pt},
        g,
        ex,
        mesh=mesh,
    )

    attn_step, _ = qwen35.make_step(mesh, g, ex, donate_states=False)
    out_clean, _ = attn_step(pos, hid_mk, w_mk, st_clean_mk)
    out_nan, _ = attn_step(pos, hid_mk, w_mk, st_nan_mk)
    assert jnp.all(jnp.isfinite(out_nan))
    np.testing.assert_array_equal(np.asarray(out_nan), np.asarray(out_clean))

    if amask == 1:
        for bad_amask in (3, 4):
            ex_bad = _tiny_exec(
                R=2, B=2, T=1, ctx=256, paged_kv=True, num_pages=32, amask=bad_amask
            )
            with pytest.raises(AssertionError, match="paged_kv requires amask"):
                qwen35.build(g, ex_bad)


def test_long_context_128k_page_table_plan_and_interpret():
    """Verify `ctx=131072` SMEM budgets, SMEM limit check, and `eff_ctx > ex.ctx`."""
    g_full = qwen35.Qwen35Geometry()
    ex_4k_direct = mk.ExecSpec(R=8, B=64, T=1, ctx=4096, paged_kv=False)
    ex_4k_paged = mk.ExecSpec(R=8, B=64, T=1, ctx=4096, paged_kv=True, num_pages=65536)
    ex_128k_paged = mk.ExecSpec(
        R=8, B=64, T=1, ctx=131072, paged_kv=True, num_pages=65536
    )
    assert ex_128k_paged.resolved_max_pages_per_seq == 1024

    spec_full = qwen35.qwen35_spec(g_full)
    cp_4k_direct, _ = mk.vmem.build_scratch(spec_full, ex_4k_direct)
    cp_4k_paged, _ = mk.vmem.build_scratch(spec_full, ex_4k_paged)
    cp_128k_paged, _ = mk.vmem.build_scratch(spec_full, ex_128k_paged)

    assert cp_128k_paged.vmem_bytes == cp_4k_paged.vmem_bytes
    assert cp_4k_direct.vmem_bytes - cp_128k_paged.vmem_bytes == 3 * 1024 * 1024 - 32768
    assert cp_128k_paged.smem_shapes["page_table"] == ((64, 1024), jnp.int32)
    assert 256 * 1024 <= cp_128k_paged.smem_bytes < 350 * 1024

    # SMEM budget check raises ValueError when page_table exceeds tiles.smem_mib (1 MiB)
    ex_smem_overflow = mk.ExecSpec(
        R=8,
        B=64,
        T=1,
        ctx=131072,
        paged_kv=True,
        num_pages=65536,
        max_pages_per_seq=65536,
    )
    with pytest.raises(ValueError, match="exceeds tiles.smem_mib budget"):
        mk.vmem.build_scratch(spec_full, ex_smem_overflow)

    # Verify eff_ctx = max(ctx, max_pages_per_seq * page_size) does not truncate > ctx
    # including when max_pages_per_seq % P_BLK != 0 (max_pages_per_seq=3, P_BLK=2)
    mesh = _cpu_mesh(2)
    g_tiny = _tiny_geom(L=4, att_every=4)
    ex_512 = _tiny_exec(R=2, B=2, T=1, ctx=512, paged_kv=True, num_pages=64)
    ex_eff = _tiny_exec(
        R=2,
        B=2,
        T=1,
        ctx=256,
        paged_kv=True,
        num_pages=64,
        max_pages_per_seq=3,  # eff_ctx = 3 * 128 = 384 > ctx = 256; 3 % P_BLK(2) == 1
    )
    _, hid_l, w_l, st_512 = qwen35.random_logical_inputs(
        jax.random.key(55), g_tiny, ex_512
    )
    # Sequence 0 is at pos=320 (> ex_eff.ctx=256, inside page 2 of eff_ctx=384)
    pos = jnp.asarray([320, 140], dtype=jnp.int32)
    ref_512, _ = qwen35.reference_step(
        g_tiny, ex_512, pos, hid_l, w_l, st_512, paged_kv=True
    )
    st_trunc = {
        **st_512,
        "kc": st_512["kc"][:, :, :, :256, :],
        "vc": st_512["vc"][:, :, :, :256, :],
    }
    ref_trunc, _ = qwen35.reference_step(
        g_tiny, ex_eff, jnp.minimum(pos, 255), hid_l, w_l, st_trunc, paged_kv=True
    )
    assert not jnp.allclose(
        ref_512[0].astype(jnp.float32), ref_trunc[0].astype(jnp.float32), atol=1e-3
    )

    raw_pools, pt_full = _pack_kv_to_raw_pools(
        g_tiny, ex_eff, st_512["kc"], st_512["vc"]
    )
    pt = pt_full[:, :3]
    assert pt.shape == (2, 3)

    with pytest.raises(ValueError, match="requires 'raw' and 'page_table'"):
        qwen35.pack_inputs(pos, hid_l, w_l, st_512, g_tiny, ex_eff, mesh=mesh)
    with pytest.raises(ValueError, match="page_table has shape"):
        qwen35.pack_inputs(
            pos,
            hid_l,
            w_l,
            {**st_512, "raw": raw_pools, "page_table": pt[:, :2]},
            g_tiny,
            ex_eff,
            mesh=mesh,
        )

    _, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos,
        hid_l,
        w_l,
        {**st_512, "raw": raw_pools, "page_table": pt},
        g_tiny,
        ex_eff,
        mesh=mesh,
    )
    step, info = qwen35.make_step(mesh, g_tiny, ex_eff)
    assert info["smem_shapes"]["page_table"] == ((2, 3), "int32")
    with pytest.raises(ValueError, match="SMEM state 'page_table' has shape"):
        step(pos, hid_mk, w_mk, {**st_mk, "page_table": jnp.zeros((2, 2), jnp.int32)})
    out_hid, _ = step(pos, hid_mk, w_mk, st_mk)
    out_b = mk.unfold(out_hid, ex_eff.R)[0, : ex_eff.B]
    assert jnp.allclose(
        out_b.astype(jnp.float32), ref_512.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_pallas_writer_pooled_step_cpu_interpret(T: int, B: int):
    """Verify `PallasWriter`, `make_flat_pooled_step`, and scattered states."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=96)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(20 + T), g, ex)
    st_l = dict(
        st_l,
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )

    ref_out, ref_st = qwen35.reference_step(
        g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )

    _, hid_mk, w_mk, st_mk_paged = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, min_np = mk.default_pool_layout(g, ex)
    assert ex.num_pages >= min_np
    raw_pools = tuple(
        jax.device_put(r, tp) for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    )

    mk_paged_step, _ = qwen35.make_step(mesh, g, ex, donate_states=False)
    _, mk_st = mk_paged_step(pos, hid_mk, w_mk, st_mk_paged)

    spec = qwen35.qwen35_spec(g)
    pooled_step, info = mk.PallasWriter().make_step(mesh, spec, ex, donate=False)
    out_hid, raw_out = pooled_step(pos, hid_mk, w_mk, raw_pools, pt, ckpt, roff)
    out_b = mk.unfold(out_hid, ex.R)[0, :B]

    ex_direct = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False)
    st_quant = dict(
        st_l,
        kc=jnp.clip(st_l["kc"], -448.0, 448.0)
        .astype(jnp.float8_e4m3fn)
        .astype(jnp.bfloat16),
        vc=jnp.clip(st_l["vc"], -448.0, 448.0)
        .astype(jnp.float8_e4m3fn)
        .astype(jnp.bfloat16),
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )
    _, _, _, st_mk_q = qwen35.pack_inputs(
        pos, hid_l, w_l, st_quant, g, ex_direct, mesh=mesh
    )
    direct_step, _ = mk.DirectHBM().make_step(mesh, spec, ex_direct, donate=False)
    dir_hid, _ = direct_step(pos, hid_mk, w_mk, st_mk_q)
    assert jnp.allclose(
        out_hid.astype(jnp.float32), dir_hid.astype(jnp.float32), atol=2e-2, rtol=2e-2
    )
    assert jnp.allclose(
        out_b.astype(jnp.float32), ref_out.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )

    st_after = mk.pool_to_states(g, ex, raw_out, pt, ckpt)
    mk_knew = mk.unfold(mk_st["knew"], ex.R)
    mk_vnew = mk.unfold(mk_st["vnew"], ex.R)
    for b in range(B):
        s_idx, _ = divmod(b, T)
        p_b = int(pos[b])
        np.testing.assert_array_equal(
            np.asarray(st_after["kc"][:, :, s_idx, p_b, :]).view(np.uint8),
            np.asarray(mk_knew[:, :, b, :]).view(np.uint8),
        )
        np.testing.assert_array_equal(
            np.asarray(st_after["vc"][:, :, s_idx, p_b, :]).view(np.uint8),
            np.asarray(mk_vnew[:, :, b, :]).view(np.uint8),
        )

    # Verify BOTH rec AND conv states written by scatter_pool_states
    QKV = g.QKV(ex.R)
    if T == 1:
        mk_rec = mk.unfold(mk_st["rec"], ex.R)
        np.testing.assert_array_equal(np.asarray(st_after["rec"]), np.asarray(mk_rec))
        assert jnp.allclose(
            st_after["rec"].astype(jnp.float32),
            ref_st["rec"].astype(jnp.float32),
            atol=3e-2,
            rtol=3e-2,
        )
        mk_conv = (
            mk.unfold(mk_st["conv"], ex.R)[:, :, : 3 * B, :]
            .reshape(ex.R, g.NGDN, 3, B, QKV)
            .swapaxes(2, 3)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(st_after["conv"][:, :, :, 0]), np.asarray(mk_conv)
        )
        ref_conv = (
            ref_st["conv"][:, :, : 3 * B, :]
            .reshape(ex.R, g.NGDN, 3, B, QKV)
            .swapaxes(2, 3)
        )
        assert jnp.allclose(
            st_after["conv"][:, :, :, 0], ref_conv, atol=3e-2, rtol=3e-2
        )
    else:
        mk_rec_ckpt = mk.unfold(mk_st["rec_ckpt"], ex.R)
        np.testing.assert_array_equal(
            np.asarray(st_after["rec"]), np.asarray(mk_rec_ckpt)
        )
        assert jnp.allclose(
            st_after["rec"].astype(jnp.float32),
            ref_st["rec_ckpt"].astype(jnp.float32),
            atol=3e-2,
            rtol=3e-2,
        )
        mk_conv_ckpt = mk.unfold(mk_st["conv_ckpt"], ex.R)
        for t_idx in range(T):
            exp_taps = (
                mk_conv_ckpt[:, :, :, 1 + t_idx : 4 + t_idx, :]
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            np.testing.assert_array_equal(
                np.asarray(st_after["conv"][:, :, :, t_idx]), np.asarray(exp_taps)
            )

    # Verify PallasWriter (states_dict path) and make_flat_pooled_step match bit-exact
    out_pw, st_dict_out = pooled_step(
        pos,
        hid_mk,
        w_mk,
        {
            "raw": raw_pools,
            "page_table": pt,
            "ckpt_indices": ckpt,
            "read_offsets": roff,
        },
    )
    np.testing.assert_array_equal(np.asarray(out_pw), np.asarray(out_hid))
    for r_a, r_b in zip(st_dict_out["raw"], raw_out):
        np.testing.assert_array_equal(
            np.asarray(r_a).view(np.uint8), np.asarray(r_b).view(np.uint8)
        )

    # Verify make_flat_pooled_step for both T=1 and T=2 (VER=True)
    flat_step, _, _, _ = mk.make_flat_pooled_step(mesh, spec, ex, donate_pools=False)
    w_flat = [w_mk[k] for k in info["plan"].w_names]
    flat_res = flat_step(pos, hid_mk, *w_flat, *raw_pools, pt, ckpt, roff)
    np.testing.assert_array_equal(np.asarray(flat_res[0]), np.asarray(out_hid))
    for r_a, r_b in zip(flat_res[1:], raw_out):
        np.testing.assert_array_equal(
            np.asarray(r_a).view(np.uint8), np.asarray(r_b).view(np.uint8)
        )


@pytest.mark.parametrize("NS,att_every", [(1, 4), (2, 4), (3, 3), (4, 4), (5, 4)])
def test_scatter_pool_states_multi_ns_and_from_ckpt(NS: int, att_every: int):
    """Verify `scatter_pool_states` across `NS=1..5`, `att_every`, and `from_ckpt`."""
    L = 4 if att_every == 4 else 6
    g = _tiny_geom(L=L, att_every=att_every)
    ex = _tiny_exec(R=2, B=NS, T=1, ctx=256, paged_kv=True, num_pages=128)
    pos, _, _, st_l = qwen35.random_logical_inputs(
        jax.random.key(30 + NS + att_every), g, ex
    )
    pt, ckpt, roff, min_np = mk.default_pool_layout(g, ex)
    assert ex.num_pages >= min_np
    ckpt_use = ckpt[0] if att_every == 3 else ckpt
    roff_use = roff[0] if att_every == 3 else roff
    raw_pools = mk.states_to_pool(g, ex, st_l, pt, ckpt_use, roff_use)
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in raw_pools)

    k1, k2, k3, k4 = jax.random.split(jax.random.key(90 + NS), 4)
    knew = jax.random.normal(k1, (g.NATT, ex.MR, g.HD), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    vnew = jax.random.normal(k2, (g.NATT, ex.MR, g.HD), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    QKV = g.QKV(ex.R)
    vh = g.vh(ex.R)
    CR = mk.rup(3 * NS, 8)
    conv_new = (
        jax.random.normal(k3, (g.NGDN, CR, QKV), dtype=jnp.float32)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    rec_new = jax.random.normal(
        k4, (g.NGDN, NS * vh, g.KD, g.VD), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    raw_sc = mk.scatter_pool_states(
        g, ex, raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_use
    )
    raw_sc_full = tuple(jnp.tile(r, (ex.R, 1, 1, 1, 1)) for r in raw_sc)
    st_after = mk.pool_to_states(g, ex, raw_sc_full, pt, ckpt_use)
    np.testing.assert_array_equal(np.asarray(st_after["rec"][0]), np.asarray(rec_new))
    exp_conv = conv_new[:, : 3 * NS, :].reshape(g.NGDN, 3, NS, QKV).swapaxes(1, 2)
    np.testing.assert_array_equal(
        np.asarray(st_after["conv"][0, :, :, 0]), np.asarray(exp_conv)
    )

    if NS >= 2:
        # Test inactive sequence 1 via explicit mask AND via "from_ckpt" (default)
        act_mask = jnp.asarray([1, 0] + [1] * (NS - 2), dtype=jnp.int32)
        raw_act = mk.scatter_pool_states(
            g,
            ex,
            raw_r0,
            pos,
            pt,
            knew,
            vnew,
            conv_new,
            rec_new,
            ckpt_use,
            act_mask,
        )
        ckpt_inact = (
            ckpt_use.at[:, 1, :].set(0)
            if ckpt_use.ndim == 3
            else ckpt_use.at[1, :].set(0)
        )
        raw_fc = mk.scatter_pool_states(
            g,
            ex,
            raw_r0,
            pos,
            pt,
            knew,
            vnew,
            conv_new,
            rec_new,
            ckpt_inact,
            "from_ckpt",
        )
        raw_default = mk.scatter_pool_states(
            g,
            ex,
            raw_r0,
            pos,
            pt,
            knew,
            vnew,
            conv_new,
            rec_new,
            ckpt_inact,
            None,
        )
        for r_a, r_b, r_d in zip(raw_act, raw_fc, raw_default):
            np.testing.assert_array_equal(
                np.asarray(r_a).view(np.uint8), np.asarray(r_b).view(np.uint8)
            )
            np.testing.assert_array_equal(
                np.asarray(r_a).view(np.uint8), np.asarray(r_d).view(np.uint8)
            )
        if NS == 2 and att_every == 4:
            from jax.experimental import pallas as pl

            with (
                pl.enable_debug_checks(True),
                pytest.raises(
                    jax.errors.JaxRuntimeError,
                    match="GDN scatter checkpoint slot maps outside",
                ),
            ):
                r_bad = jax.jit(
                    lambda ck: mk.scatter_pool_states(
                        g,
                        ex,
                        raw_r0,
                        pos,
                        pt,
                        knew,
                        vnew,
                        conv_new,
                        rec_new,
                        ck,
                    )
                )(ckpt_use + ex.num_pages)
                r_bad[0].block_until_ready()


def test_scatter_pool_states_top_of_pool_t2_no_corruption():
    """Verify T>1 per-token and contiguous scatter at top of pool."""
    g = _tiny_geom(L=2, att_every=2)  # G_PER_ATT=1 -> 5 pages per slot
    # With num_pages=64 and T=2: (num_pages - 5*T, num_pages - 5] is (54, 59].
    # Slot 11 has page base 11*5 = 55 (pages 55..59), right in (54, 59]!
    # Neighboring slot 10 has page base 10*5 = 50 (pages 50..54).
    ex = _tiny_exec(R=2, B=4, T=2, ctx=256, paged_kv=True, num_pages=64)
    pos, _, _, st_l = qwen35.random_logical_inputs(jax.random.key(77), g, ex)
    pt, _, roff, _ = mk.default_pool_layout(g, ex)
    # 2D ckpt_indices: seq 0 uses slots (8, 9), seq 1 uses slots (10, 11)
    # where slot 11 -> pages 55..59 (in (num_pages - 5*T, num_pages - 5]).
    ckpt_2d = jnp.asarray([[8, 9], [10, 11]], dtype=jnp.int32)
    raw_pools = mk.states_to_pool(g, ex, st_l, pt, ckpt_2d, roff[0])
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in raw_pools)

    k1, k2, k3, k4 = jax.random.split(jax.random.key(78), 4)
    knew = jax.random.normal(k1, (g.NATT, ex.MR, g.HD), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    vnew = jax.random.normal(k2, (g.NATT, ex.MR, g.HD), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    QKV = g.QKV(ex.R)
    vh = g.vh(ex.R)
    conv_new = (
        jax.random.normal(k3, (g.NGDN, 2, 3 + ex.T, QKV), dtype=jnp.float32)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    rec_new = jax.random.normal(
        k4, (g.NGDN, ex.NS * ex.T * vh, g.KD, g.VD), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    def _assert_rec_and_conv(st_after: dict):
        np.testing.assert_array_equal(
            np.asarray(st_after["rec"][0]), np.asarray(rec_new)
        )
        for t_idx in range(ex.T):
            exp_taps = (
                conv_new[:, :, 1 + t_idx : 4 + t_idx, :]
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            np.testing.assert_array_equal(
                np.asarray(st_after["conv"][0, :, :, t_idx]), np.asarray(exp_taps)
            )

    # 1. Test 2D ckpt_indices (always takes per-token branch)
    raw_sc_2d = mk.scatter_pool_states(
        g, ex, raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_2d
    )
    raw_sc_2d_full = tuple(jnp.tile(r, (ex.R, 1, 1, 1, 1)) for r in raw_sc_2d)
    st_after_2d = mk.pool_to_states(g, ex, raw_sc_2d_full, pt, ckpt_2d)
    _assert_rec_and_conv(st_after_2d)

    # 2. Test 3D non-contiguous ckpt_indices with slot 11 at t=0 and slot 10 at t=1
    ckpt_3d_noncontig = jnp.asarray([[[8, 9], [11, 10]]], dtype=jnp.int32)
    raw_sc_3d = mk.scatter_pool_states(
        g, ex, raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_3d_noncontig
    )
    raw_sc_3d_full = tuple(jnp.tile(r, (ex.R, 1, 1, 1, 1)) for r in raw_sc_3d)
    st_after_3d = mk.pool_to_states(g, ex, raw_sc_3d_full, pt, ckpt_3d_noncontig)
    _assert_rec_and_conv(st_after_3d)

    # 3. Test 3D contiguous ckpt_indices at top of pool (slots [10, 11] -> pages 50..59,
    # exercising span = 5*T = 10 at num_pages - 5*T = 54 without corrupting [8, 9])
    ckpt_3d_contig_top = jnp.asarray([[[8, 9], [10, 11]]], dtype=jnp.int32)
    raw_sc_3d_contig = mk.scatter_pool_states(
        g, ex, raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_3d_contig_top
    )
    raw_sc_3d_contig_full = tuple(
        jnp.tile(r, (ex.R, 1, 1, 1, 1)) for r in raw_sc_3d_contig
    )
    st_after_3d_contig = mk.pool_to_states(
        g, ex, raw_sc_3d_contig_full, pt, ckpt_3d_contig_top
    )
    _assert_rec_and_conv(st_after_3d_contig)

    # 4. Verify out-of-bounds ckpt slot (slot 12 -> pages 60..64) raises ValueError
    with pytest.raises(ValueError, match="exceeds pool capacity"):
        mk.scatter_pool_states(
            g,
            ex,
            raw_r0,
            pos,
            pt,
            knew,
            vnew,
            conv_new,
            rec_new,
            jnp.asarray([[8, 9], [10, 12]], dtype=jnp.int32),
        )

    # 5. Also trace non-interpret (TPU Ref.bitcast().reshape() path) via make_jaxpr
    ex_tpu_trace = replace(ex, interpret=False)
    jaxpr_sc = jax.make_jaxpr(
        lambda r0, p, pt_i, kn, vn, cn, rn, ck: mk.scatter_pool_states(
            g, ex_tpu_trace, r0, p, pt_i, kn, vn, cn, rn, ck
        )
    )(raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_2d)
    assert "qwen3_5_scatter_pool_states" in str(jaxpr_sc)


def test_donate_pools_all_call_shapes_no_pool_copy():
    """Verify `donate_pools=True` donates `raw_pools` across all 4 call shapes."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))

    # 1-3: Mixed GDN+GQA model (states_dict, positional no-act, positional with act)
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=2, B=2, T=1, ctx=256, paged_kv=True, num_pages=96)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(61), g, ex)
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    spec = qwen35.qwen35_spec(g)
    pw_step, _ = mk.PallasWriter().make_step(mesh, spec, ex, donate=True)

    def _fresh_pools(target_g=g):
        return tuple(
            jax.device_put(np.asarray(r).copy(), tp)
            for r in mk.states_to_pool(target_g, ex, st_l, pt, ckpt, roff)
        )

    # Shape 1: states_dict call (used by PallasWriter)
    r1 = _fresh_pools()
    st_d = {
        "raw": r1,
        "page_table": pt,
        "ckpt_indices": ckpt,
        "read_offsets": roff,
    }
    lowered_1 = pw_step.lower(pos, hid_mk, w_mk, st_d)
    assert "jax.buffer_donor = true" in lowered_1.as_text()
    compiled_hlo_1 = lowered_1.compile().as_text()
    ok_real, msg_real = mk.check_pooled_step_hlo(
        compiled_hlo_1, page_size=ex.page_size, kvh=1, hd=g.HD
    )
    assert ok_real, msg_real
    out_1, _ = pw_step(pos, hid_mk, w_mk, st_d)
    assert all(buf.is_deleted() for buf in r1)

    # Verify 3D page_table (R, NS, W) at R=2 on PallasWriter
    r_3d = _fresh_pools()
    pt_3d = jax.device_put(
        jnp.broadcast_to(pt[None, ...], (ex.R, *pt.shape)),
        tp,
    )
    out_3d, _ = pw_step(pos, hid_mk, w_mk, r_3d, pt_3d, ckpt, roff)
    np.testing.assert_array_equal(np.asarray(out_3d), np.asarray(out_1))
    assert all(buf.is_deleted() for buf in r_3d)

    # Also trace gather_gdn_states with interpret=False via make_jaxpr
    ex_non_interp = replace(ex, interpret=False)
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in _fresh_pools())
    jaxpr_gather = jax.make_jaxpr(
        lambda r0, ck, ro: mk.gather_gdn_states(g, ex_non_interp, r0, ck, ro)
    )(raw_r0, ckpt, roff)
    assert "qwen3_5_gather_gdn_states" in str(jaxpr_gather)

    # Shape 2: GDN positional without active_seqs (reuses compiled _jitted_step)
    r2 = _fresh_pools()
    hlo_2 = pw_step.lower(pos, hid_mk, w_mk, r2, pt, ckpt, roff).as_text()
    assert "jax.buffer_donor = true" in hlo_2
    _, _ = pw_step(pos, hid_mk, w_mk, r2, pt, ckpt, roff)
    assert all(buf.is_deleted() for buf in r2)

    # Shape 3: GDN positional with active_seqs array
    r3 = _fresh_pools()
    act = jnp.asarray([1, 1], dtype=jnp.int32)
    hlo_3 = pw_step.lower(pos, hid_mk, w_mk, r3, pt, ckpt, roff, act).as_text()
    assert "jax.buffer_donor = true" in hlo_3
    _, _ = pw_step(pos, hid_mk, w_mk, r3, pt, ckpt, roff, act)
    assert all(buf.is_deleted() for buf in r3)

    # Shape 4: Attention-only (NGDN=0) positional call
    g_att = _tiny_geom(L=1, att_every=1)
    _, hid_a, w_a, st_a = qwen35.random_logical_inputs(jax.random.key(62), g_att, ex)
    _, hid_mk_a, w_mk_a, _ = qwen35.pack_inputs(
        pos, hid_a, w_a, st_a, g_att, ex, mesh=mesh, build_default_pool=True
    )
    pt_a, ckpt_a, roff_a, _ = mk.default_pool_layout(g_att, ex)
    r4 = tuple(
        jax.device_put(np.asarray(r).copy(), tp)
        for r in mk.states_to_pool(g_att, ex, st_a, pt_a, ckpt_a, roff_a)
    )
    att_step, _ = qwen35.make_pooled_step(mesh, g_att, ex, donate_pools=True)
    hlo_4 = att_step.lower(pos, hid_mk_a, w_mk_a, r4, pt_a).as_text()
    assert "jax.buffer_donor = true" in hlo_4
    _, _ = att_step(pos, hid_mk_a, w_mk_a, r4, pt_a)
    assert all(buf.is_deleted() for buf in r4)

    # Also verify check_pooled_step_hlo negative cases
    bad_single_cc, _ = mk.check_pooled_step_hlo("tpu_custom_call scatter_pool_states")
    assert not bad_single_cc
    bad_copy, _ = mk.check_pooled_step_hlo(
        "tpu_custom_call tpu_custom_call scatter_pool_states "
        "%c = f8e4m3fn[64,2,64,4,128] copy(%p)"
    )
    assert not bad_copy


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_unstacked_expert_weights_equivalence(T: int, B: int):
    """Verify `stack_e=False` (per-layer tuples, 3D/5D, flat step) is bit-exact."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    ex_stacked = _tiny_exec(
        R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=96, stack_e=True
    )
    ex_unstacked = replace(ex_stacked, stack_e=False)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(66 + T), g, ex_stacked
    )

    _, hid_mk, w_stacked, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_stacked, mesh=mesh, build_default_pool=True
    )
    # Pass per-layer expert tuples directly into pack_inputs without a stacked tensor
    w_l_per_layer = dict(
        w_l,
        e13=tuple(w_l["e13"][:, lyr] for lyr in range(g.L)),
        e2=tuple(w_l["e2"][:, lyr] for lyr in range(g.L)),
    )
    _, _, w_unstacked, _ = qwen35.pack_inputs(
        pos,
        hid_l,
        w_l_per_layer,
        st_l,
        g,
        ex_unstacked,
        mesh=mesh,
        build_default_pool=True,
    )
    assert isinstance(w_unstacked["e13"], tuple) and len(w_unstacked["e13"]) == g.L
    assert isinstance(w_unstacked["e2"], tuple) and len(w_unstacked["e2"]) == g.L

    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex_stacked)
    raw_pools = tuple(
        jax.device_put(r, tp)
        for r in mk.states_to_pool(g, ex_stacked, st_l, pt, ckpt, roff)
    )

    spec = qwen35.qwen35_spec(g)
    step_stacked, _ = mk.PallasWriter().make_step(mesh, spec, ex_stacked, donate=False)
    step_unstacked, info_u = mk.PallasWriter().make_step(
        mesh, spec, ex_unstacked, donate=False
    )
    out_s, raw_s = step_stacked(pos, hid_mk, w_stacked, raw_pools, pt, ckpt, roff)
    out_u, raw_u = step_unstacked(pos, hid_mk, w_unstacked, raw_pools, pt, ckpt, roff)

    np.testing.assert_array_equal(np.asarray(out_u), np.asarray(out_s))
    for r_u, r_s in zip(raw_u, raw_s):
        np.testing.assert_array_equal(
            np.asarray(r_u).view(np.uint8), np.asarray(r_s).view(np.uint8)
        )

    if T == 1:
        # Exercise 3D per-rank `(EW, K, N)` (global `(R * EW, K, N)`) reshape branch
        # and `make_flat_pooled_step` with `stack_e=False`.
        e13_3d = tuple(
            jax.device_put(mk.fold(w_l["e13"][:, lyr]), tp) for lyr in range(g.L)
        )
        e2_3d = tuple(
            jax.device_put(mk.fold(w_l["e2"][:, lyr]), tp) for lyr in range(g.L)
        )
        assert e13_3d[0].ndim == 3 and e2_3d[0].ndim == 3
        w_unstacked_3d = dict(w_unstacked, e13=e13_3d, e2=e2_3d)

        w_flat_u: list[jax.Array] = []
        for k in info_u["plan"].w_names:
            if k in ("e13", "e2"):
                w_flat_u.extend(w_unstacked_3d[k])
            else:
                w_flat_u.append(w_unstacked_3d[k])

        flat_u, _, _, _ = mk.make_flat_pooled_step(
            mesh, spec, ex_unstacked, donate_pools=False
        )
        flat_res = flat_u(pos, hid_mk, *w_flat_u, *raw_pools, pt, ckpt, roff)
        np.testing.assert_array_equal(np.asarray(flat_res[0]), np.asarray(out_s))
        for r_f, r_s in zip(flat_res[1:], raw_s):
            np.testing.assert_array_equal(
                np.asarray(r_f).view(np.uint8), np.asarray(r_s).view(np.uint8)
            )

        # Verify unpack_local_weight rejects transposed non-expert and expert tensors
        with pytest.raises(ValueError, match="local shard has shape"):
            mk.kernel.unpack_local_weight(
                jnp.zeros((1, 4, 32, 16, 128), dtype=jnp.bfloat16),
                (4, 16, 128, 32),
                name="router",
                allow_expert_reshape=False,
            )
        with pytest.raises(ValueError, match="local shard has shape"):
            mk.kernel.unpack_local_weight(
                jnp.zeros((8, 64, 256), dtype=jnp.float8_e4m3fn),
                (8, 2, 128, 64),
                name="e13",
                allow_expert_reshape=True,
            )


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_eplb_ecopy_stack_e_and_split_hf_weights(T: int, B: int):
    """Verify `split_hf_weights` (R=1 vs R=2), `encode_ecopy`, EPLB, and TP MoE."""
    # Verify encode_ecopy warns when an expert has > 1 secondary replicas
    with pytest.warns(UserWarning, match="multiple secondary replicas"):
        mk.encode_ecopy(
            E=6,
            ER=2,
            EW=3,
            moe_tp=False,
            cores=[[0, 1, 2], [2, 3, 0], [4, 5, 0]],
        )
    # Verify encode_ecopy warns when a redundant slot is placed on the same rank
    with pytest.warns(UserWarning, match="same rank as its primary slot"):
        mk.encode_ecopy(
            E=4,
            ER=2,
            EW=3,
            moe_tp=False,
            cores=[[0, 1, 0], [2, 3, 1]],
        )

    mesh = _cpu_mesh(2)
    g_ep = _tiny_geom(L=4, att_every=4, moe_tp=False, e_slots=0)
    g_eplb = _tiny_geom(L=4, att_every=4, moe_tp=False, e_slots=68)
    g_tp = _tiny_geom(L=4, att_every=4, moe_tp=True, e_slots=0)

    ex_base = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False)
    ex_opt0 = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False, ecopy_opt=0)
    ex_po2c = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False, ecopy_opt=1)
    ex_eplb_unstacked = _tiny_exec(
        R=2, B=B, T=T, ctx=256, paged_kv=False, ecopy_opt=1, stack_e=False
    )

    key = jax.random.key(202 + T)
    k_hf, k_in = jax.random.split(key)
    hf_w = qwen35.random_hf_weights(k_hf, g_ep)

    # Bias router strongly toward experts 0 and 1 (replicated onto rank 1 slots 64..65
    # on layer 0) and replicate KV head 0 across both KV heads so R=1 (NKV=1) and R=2
    # (NKV=2) execute the exact same mathematical model.
    router_biased = hf_w["router"].at[:, :, :2].add(jnp.bfloat16(100.0))
    hf_w = dict(
        hf_w,
        router=router_biased,
        k_proj=jnp.repeat(hf_w["k_proj"][:, :1], 2, axis=1),
        k_proj_s=jnp.repeat(hf_w["k_proj_s"][:, :1], 2, axis=1),
        v_proj=jnp.repeat(hf_w["v_proj"][:, :1], 2, axis=1),
        v_proj_s=jnp.repeat(hf_w["v_proj_s"][:, :1], 2, axis=1),
        kn=jnp.repeat(hf_w["kn"][:, :1], 2, axis=1),
    )

    pos, hid_l, _, st_l = qwen35.random_logical_inputs(k_in, g_ep, ex_base)
    st_l = dict(
        st_l,
        kc=jnp.repeat(st_l["kc"][:1], 2, axis=0),
        vc=jnp.repeat(st_l["vc"][:1], 2, axis=0),
    )

    # Pass only unfused HF projection keys to verify real HF tensor splitting
    fused_keys = (
        "att_q",
        "att_q_s",
        "att_kv",
        "att_kv_s",
        "sh13",
        "sh13_s",
        "e13",
        "e13_s",
    )
    hf_unfused = {k: v for k, v in hf_w.items() if k not in fused_keys}
    w_ep_l = qwen35.split_hf_weights(hf_unfused, g_ep, ex_base)
    w_eplb_l = qwen35.split_hf_weights(hf_unfused, g_eplb, ex_base)
    w_tp_l = qwen35.split_hf_weights(hf_unfused, g_tp, ex_base)
    assert "ecopy" in w_eplb_l
    assert jnp.any(w_eplb_l["ecopy"] > 0)

    # Unsharded R=1 forward pass comparison for split_hf_weights
    g_r1 = replace(g_ep, NKV=1)
    ex_r1 = replace(ex_base, R=1)
    hf_r1 = dict(
        hf_unfused,
        k_proj=hf_unfused["k_proj"][:, :1],
        k_proj_s=hf_unfused["k_proj_s"][:, :1],
        v_proj=hf_unfused["v_proj"][:, :1],
        v_proj_s=hf_unfused["v_proj_s"][:, :1],
        kn=hf_unfused["kn"][:, :1],
    )
    w_r1_l = qwen35.split_hf_weights(hf_r1, g_r1, ex_r1)
    qk = (g_ep.KH // ex_base.R) * g_ep.KD
    st_r1 = {
        "kc": st_l["kc"][:1],
        "vc": st_l["vc"][:1],
        "conv": jnp.concatenate(
            [
                st_l["conv"][0, :, :, :, :qk],
                st_l["conv"][1, :, :, :, :qk],
                st_l["conv"][0, :, :, :, qk : 2 * qk],
                st_l["conv"][1, :, :, :, qk : 2 * qk],
                st_l["conv"][0, :, :, :, 2 * qk :],
                st_l["conv"][1, :, :, :, 2 * qk :],
            ],
            axis=-1,
        )[None, ...],
        "rec": jnp.concatenate([st_l["rec"][0], st_l["rec"][1]], axis=2)[None, ...],
    }
    ref_r1, _ = qwen35.reference_step(g_r1, ex_r1, pos, hid_l, w_r1_l, st_r1)

    # Directly verify GDN `lin_in` & `lin_in_s` layout against unsharded HF tensors
    lay_gdn = mk.GDNBlock.lin_in_layout(g_ep, ex_base)
    vh = lay_gdn.vh
    assert w_ep_l["lin_in"].shape == (ex_base.R, g_ep.NGDN, g_ep.H, lay_gdn.NLI)
    assert w_ep_l["lin_in_s"].shape == (ex_base.R, g_ep.NGDN, lay_gdn.NLI)
    exp_a = jnp.transpose(
        hf_unfused["lin_a"].reshape(g_ep.NGDN, g_ep.H, ex_base.R, vh),
        (2, 0, 1, 3),
    )
    exp_b = jnp.transpose(
        hf_unfused["lin_b"].reshape(g_ep.NGDN, g_ep.H, ex_base.R, vh),
        (2, 0, 1, 3),
    )
    np.testing.assert_array_equal(
        np.asarray(w_ep_l["lin_in"][..., lay_gdn.BA : lay_gdn.BA + vh]).view(np.uint8),
        np.asarray(exp_b).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(w_ep_l["lin_in"][..., lay_gdn.A_OFF : lay_gdn.A_OFF + vh]).view(
            np.uint8
        ),
        np.asarray(exp_a).view(np.uint8),
    )
    assert jnp.all(
        w_ep_l["lin_in"][..., lay_gdn.BA + vh : lay_gdn.A_OFF].astype(jnp.float32) == 0
    )
    assert jnp.all(
        w_ep_l["lin_in"][..., lay_gdn.A_OFF + vh : lay_gdn.NLI].astype(jnp.float32) == 0
    )

    ref_ep, _ = qwen35.reference_step(g_ep, ex_base, pos, hid_l, w_ep_l, st_l)
    ref_tp, _ = qwen35.reference_step(g_tp, ex_base, pos, hid_l, w_tp_l, st_l)
    assert jnp.allclose(
        ref_ep.astype(jnp.float32), ref_r1.astype(jnp.float32), atol=1e-2, rtol=1e-2
    )

    def run_cfg(g_cfg, ex_cfg, w_logical):
        p_in, h_mk, w_mk, s_mk = qwen35.pack_inputs(
            pos, hid_l, w_logical, st_l, g_cfg, ex_cfg, mesh=mesh
        )
        step, _ = qwen35.make_step(mesh, g_cfg, ex_cfg)
        out_h, _ = step(p_in, h_mk, w_mk, s_mk)
        return mk.unfold(out_h, ex_cfg.R)[0, :B]

    out_ep = run_cfg(g_ep, ex_base, w_ep_l)
    out_eplb_opt0 = run_cfg(g_eplb, ex_opt0, w_eplb_l)
    out_eplb_opt1 = run_cfg(g_eplb, ex_po2c, w_eplb_l)
    out_eplb_unstacked = run_cfg(g_eplb, ex_eplb_unstacked, w_eplb_l)
    out_tp = run_cfg(g_tp, ex_base, w_tp_l)

    # Prove that replica routing actually occurred: corrupting redundant slots [64:68]
    # changes the output of both load-balanced ecopy_opt=0 and deduplicated ecopy_opt=1.
    er = g_eplb.ER(ex_base.R)
    w_eplb_zero_rep = dict(
        w_eplb_l,
        e2=w_eplb_l["e2"].at[:, :, er:].set(jnp.zeros_like(w_eplb_l["e2"][:, :, er:])),
        e2_s=w_eplb_l["e2_s"]
        .at[:, :, er:]
        .set(jnp.zeros_like(w_eplb_l["e2_s"][:, :, er:])),
    )
    for ex_opt, ref_opt_out in ((ex_opt0, out_eplb_opt0), (ex_po2c, out_eplb_opt1)):
        out_eplb_zero_rep = run_cfg(g_eplb, ex_opt, w_eplb_zero_rep)
        assert not jnp.allclose(
            out_eplb_zero_rep.astype(jnp.float32),
            ref_opt_out.astype(jnp.float32),
            atol=1e-3,
        )

    np.testing.assert_array_equal(
        np.asarray(out_eplb_unstacked), np.asarray(out_eplb_opt1)
    )
    assert jnp.allclose(
        out_eplb_opt0.astype(jnp.float32),
        out_ep.astype(jnp.float32),
        atol=1.6e-2,
        rtol=1.6e-2,
    )
    assert jnp.allclose(
        out_eplb_opt1.astype(jnp.float32),
        out_ep.astype(jnp.float32),
        atol=1.6e-2,
        rtol=1.6e-2,
    )
    assert jnp.allclose(
        out_ep.astype(jnp.float32), ref_ep.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )
    assert jnp.allclose(
        out_ep.astype(jnp.float32), ref_r1.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )
    assert jnp.allclose(
        out_tp.astype(jnp.float32), ref_tp.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )
    tp_ep_diff = float(
        jnp.max(jnp.abs(out_tp.astype(jnp.float32) - out_ep.astype(jnp.float32)))
    )
    assert tp_ep_diff <= 1.6e-2, f"TP vs EP max diff too large: {tp_ep_diff}"


import sys  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from frozen_utils import DEFAULT_FROZEN_DIR, load_frozen_qwen35  # noqa: E402


@pytest.mark.skipif(
    not os.path.isdir(os.environ.get("MK_FROZEN_QWEN35_DIR", DEFAULT_FROZEN_DIR)),
    reason="Set MK_FROZEN_QWEN35_DIR to canonical c196d2500 frozen qwen35 path",
)
@pytest.mark.parametrize("T,B", [(1, 2)])
def test_m3_bit_exact_vs_frozen_pool_io_and_eplb(T: int, B: int):
    """Bit-exact check vs canonical `c196d2500` frozen `pool_io` and `R=2` EPLB."""
    frozen_mk, frozen_pio = load_frozen_qwen35()

    # 1. Pooled step (`pool_io.make_pooled_step`) bit-exact comparison
    mesh1 = _cpu_mesh(1)
    tp1 = NamedSharding(mesh1, P("tp"))

    g_pool = replace(_tiny_geom(L=4, att_every=4, e_slots=0), NKV=1)
    ex_pool = _tiny_exec(
        R=1, B=B, T=T, ctx=256, paged_kv=True, num_pages=96, stack_e=False
    )
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(303 + T), g_pool, ex_pool
    )
    _, hid_mk, w_lib, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g_pool, ex_pool, mesh=mesh1, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g_pool, ex_pool)
    raw_pools = tuple(
        jax.device_put(r, tp1)
        for r in mk.states_to_pool(g_pool, ex_pool, st_l, pt, ckpt, roff)
    )

    step_lib, info_lib = qwen35.make_pooled_step(
        mesh1, g_pool, ex_pool, donate_pools=False
    )
    out_hid_lib, raw_out_lib = step_lib(pos, hid_mk, w_lib, raw_pools, pt, ckpt, roff)

    tiles_f = replace(
        frozen_mk.TINY_TILES,
        bank=512 << 10,
        kv_blk=min(frozen_mk.TINY_TILES.kv_blk, 256),
    )
    c_f = frozen_mk.MKConfig(
        R=1,
        B=B,
        T=T,
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
        ctx=256,
        paged_kv=True,
        num_pages=96,
        max_pages_per_seq=2,
        stack_e=False,
    )
    f_lay, f_S = frozen_mk.smalls_layout(c_f)
    l_lay = info_lib["plan"].smalls_layout
    sm_f = jnp.zeros((ex_pool.R, g_pool.L, 1, f_S), dtype=jnp.float32)
    for name, (lo, w) in l_lay.items():
        fo, _ = f_lay[name]
        sm_f = sm_f.at[:, :, 0, fo : fo + w].set(w_lib["smalls"][:, :, 0, lo : lo + w])

    w_f = {**w_lib, "smalls": jax.device_put(sm_f, tp1)}
    step_f, _ = frozen_pio.make_pooled_step(
        mesh1, c_f, tiles_f, interpret=True, fused_pool_io=False
    )
    out_hid_f, raw_out_f = jax.jit(step_f)(pos, hid_mk, w_f, raw_pools, pt, ckpt, roff)

    np.testing.assert_array_equal(np.asarray(out_hid_lib), np.asarray(out_hid_f))
    for r_lib, r_f in zip(raw_out_lib, raw_out_f):
        np.testing.assert_array_equal(
            np.asarray(r_lib).view(np.uint8),
            np.asarray(r_f).view(np.uint8),
        )

    # 2. Multi-rank (R=2) EPLB bit-exact comparison with non-zero ecopy
    mesh2 = _cpu_mesh(2)
    tp2 = NamedSharding(mesh2, P("tp"))
    g_eplb = _tiny_geom(L=4, att_every=4, moe_tp=False, e_slots=68)
    ex_eplb = _tiny_exec(
        R=2, B=B, T=T, ctx=256, paged_kv=False, ecopy_opt=1, stack_e=False
    )
    pos_e, hid_le, w_le, st_le = qwen35.random_logical_inputs(
        jax.random.key(404 + T), g_eplb, ex_eplb
    )
    assert "ecopy" in w_le and jnp.any(w_le["ecopy"] > 0)
    # Bias router strongly toward experts 0 and 1 (replicated on rank 1 slots 64..65)
    # so Po2C deterministically routes tokens to secondary replicas.
    w_le = dict(
        w_le,
        router=w_le["router"].at[:, :, :2].add(jnp.bfloat16(100.0)),
    )
    _, hid_mke, w_libe, st_mke = qwen35.pack_inputs(
        pos_e, hid_le, w_le, st_le, g_eplb, ex_eplb, mesh=mesh2
    )

    step_lib_e, info_lib_e = qwen35.make_step(mesh2, g_eplb, ex_eplb)
    out_hid_lib_e, out_st_lib_e = step_lib_e(pos_e, hid_mke, w_libe, st_mke)

    # Assert that secondary replica slots were actually exercised
    er_e = g_eplb.ER(ex_eplb.R)
    w_le_zero_rep = dict(
        w_le,
        e2=w_le["e2"].at[:, :, er_e:].set(jnp.zeros_like(w_le["e2"][:, :, er_e:])),
    )
    _, _, w_libe_zero, _ = qwen35.pack_inputs(
        pos_e, hid_le, w_le_zero_rep, st_le, g_eplb, ex_eplb, mesh=mesh2
    )
    out_hid_zero_rep, _ = step_lib_e(pos_e, hid_mke, w_libe_zero, st_mke)
    assert not jnp.allclose(
        out_hid_zero_rep.astype(jnp.float32),
        out_hid_lib_e.astype(jnp.float32),
        atol=1e-3,
    )

    c_fe = replace(
        c_f,
        R=2,
        NKV=2,
        paged_kv=False,
        e_slots=68,
    )
    tiles_fe = replace(tiles_f, ecopy_opt=1)
    f_lay_e, f_Se = frozen_mk.smalls_layout(c_fe)
    l_lay_e = info_lib_e["plan"].smalls_layout
    sm_fe = jnp.zeros((ex_eplb.R, g_eplb.L, 1, f_Se), dtype=jnp.float32)
    for name, (lo, w) in l_lay_e.items():
        fo, _ = f_lay_e[name]
        sm_fe = sm_fe.at[:, :, 0, fo : fo + w].set(
            w_libe["smalls"][:, :, 0, lo : lo + w]
        )
    w_fe = {**w_libe, "smalls": jax.device_put(sm_fe, tp2)}
    step_fe, _ = frozen_mk.make_step(mesh2, c_fe, tiles_fe, interpret=True)
    out_hid_fe, out_st_fe = jax.jit(step_fe)(pos_e, hid_mke, w_fe, st_mke)

    np.testing.assert_array_equal(
        np.asarray(out_hid_lib_e).view(np.uint8),
        np.asarray(out_hid_fe).view(np.uint8),
    )
    conv_key = "conv" if T == 1 else "conv_ckpt"
    rec_key = "rec" if T == 1 else "rec_ckpt"
    np.testing.assert_array_equal(
        np.asarray(out_st_lib_e[conv_key]).view(np.uint8),
        np.asarray(out_st_fe["conv"]).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(out_st_lib_e[rec_key]).view(np.uint8),
        np.asarray(out_st_fe["rec"]).view(np.uint8),
    )
    for k in ("kc", "vc"):
        np.testing.assert_array_equal(
            np.asarray(out_st_lib_e[k]).view(np.uint8),
            np.asarray(out_st_fe[k]).view(np.uint8),
        )
