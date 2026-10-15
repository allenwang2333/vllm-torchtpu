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
from jax.sharding import Mesh

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
    """Verify per-page KV raw page pack/unpack helpers round-trip."""
    key = jax.random.key(101)
    k1, k2 = jax.random.split(key, 2)

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
