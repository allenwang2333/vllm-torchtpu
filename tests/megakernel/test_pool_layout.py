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
"""Generic `pool/` tests: `PoolLayout` round trips for several layouts + guards.

Covers the Qwen3.5 layout (3 recurrent slots per pool), a 1-slot layout, a
2-slot layout with 3 pools and a KV-only layout (0 slots, no recurrent heads)
through pack / unpack, `states_to_pool` / `pool_to_states`, the Pallas gather
and scatter kernels (CPU interpret) with 2D and 3D checkpoint indices, 1 to 4
sequences, a rank-local scatter, a cold-start `K = T + 1` checkpoint table,
checkpoint bounds checks, the pooled-step HLO check, one full pooled step with
`att_every=2`, and AST guards that keep the pool state name and the slot stride
out of core code.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES

FP8 = jnp.float8_e4m3fn
NUM_PAGES = 128
# Recurrent slots per pool: Qwen3.5 (att_every=4), att_every=2, KV-only.
SLOTS = {"qwen35": 3, "slots1": 1, "kv_only": 0}


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _layout(slots: int, num_pools: int = 2) -> mk.PoolLayout:
    """Recurrent layouts use kh=2, vh=4; a KV-only layout has no recurrent heads."""
    return mk.PoolLayout(
        num_pools=num_pools,
        slots_per_pool=slots,
        hd=256,
        kvh=1,
        kh=2 if slots else 0,
        vh=4 if slots else 0,
        KD=128,
        VD=128,
    )


def _exec(R: int, B: int, T: int, ctx: int = 256) -> mk.ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10).with_blocks(
        {"gqa": {"kv_blk": min(TINY_TILES.block("gqa").kv_blk, ctx)}}
    )
    return mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=True,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=NUM_PAGES),
    )


def _bf16_f32(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Random f32 values exactly representable in bf16 (pool conv storage)."""
    x = jax.random.normal(key, shape, dtype=jnp.float32) * 0.1
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def _random_states(
    lay: mk.PoolLayout, ex: mk.ExecSpec, seed: int
) -> dict[str, jax.Array]:
    """Logical states `[R, ...]` in the formats `pool.states_to_pool` accepts."""
    R, NS = ex.R, ex.NS
    k = jax.random.split(jax.random.key(seed), 4)
    kv_shape = (R, lay.num_pools, NS, ex.ctx, lay.hd)
    st = {
        "kc": (jax.random.normal(k[0], kv_shape) * 0.3).astype(FP8),
        "vc": (jax.random.normal(k[1], kv_shape) * 0.3).astype(FP8),
    }
    if lay.has_recurrent:
        ngdn, vh = lay.num_recurrent_layers, lay.vh_per_rank(R)
        qkv = lay.qkv_per_rank(R)
        st["conv"] = _bf16_f32(k[2], (R, ngdn, NS, lay.conv_taps, qkv))
        st["rec"] = (
            jax.random.normal(k[3], (R, ngdn, NS * vh, lay.KD, lay.VD)) * 0.1
        ).astype(jnp.bfloat16)
    return st


def _tables(
    lay: mk.PoolLayout, ex: mk.ExecSpec, ckpt_ndim: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    pt, ckpt, roff, min_pages = pool.allocate_test_layout(lay, ex)
    assert mk.cache_exec(ex).num_pages >= min_pages
    assert (
        int(jnp.min(pt[:, : ex.ctx // mk.cache_exec(ex).page_size]))
        >= lay.reserved_pages
    )
    if lay.has_recurrent and ckpt_ndim == 2:
        ckpt, roff = ckpt[0], roff[0]
    return pt, ckpt, roff


def _bytes(x: jax.Array) -> np.ndarray:
    return np.asarray(x).view(np.uint8)


@pytest.mark.parametrize("name", list(SLOTS))
def test_layout_derived_geometry(name: str):
    lay = _layout(SLOTS[name])
    lay.validate(_exec(R=2, B=2, T=1))
    assert lay.pages_per_slot == 5
    assert lay.kv_page_shape == (2, 64, 4, 128)
    assert lay.conv_page_rows == 256 and lay.rec_u32_rows == 512
    assert lay.reserved_pages == max(1, 5 * SLOTS[name])
    assert lay.page_base(7, 2, 3) == 35
    assert lay.page_base(7, 2, 2) == 7 * 5 * SLOTS[name] + 10
    backend = mk.make_pool_backend(lay)
    assert backend.layout is lay and backend.num_pools == 2
    assert backend.has_recurrent_pool == (SLOTS[name] > 0)
    assert backend.recurrent_slots_per_pool == max(1, SLOTS[name])
    assert backend.ckpt_pages_per_slot == 5
    assert backend.kernel_names == pool.POOL_KERNEL_NAMES


def test_layout_validate_rejects_unvalidated_geometry():
    for bad in (
        replace(_layout(3), hd=128),
        replace(_layout(3), page_size=64),
        replace(_layout(3), kvh=2),
        replace(_layout(3), rec_pages_per_slot=2),
        replace(_layout(3), conv_taps=4),
        replace(_layout(3), num_pools=0),
    ):
        with pytest.raises(mk.MegakernelConfigError):
            bad.validate()
    with pytest.raises(mk.MegakernelConfigError, match="page_size"):
        _layout(3).validate(mk.with_cache(_exec(R=2, B=2, T=1), page_size=64))


def test_layout_validate_checks_in_kernel_conv_windows():
    """GPR stages the verify conv window in the K half of the conv page (128
    bf16 rows), fused pool I/O in the whole conv page (256 rows)."""
    lay = replace(_layout(3), kh=8, vh=8)  # 24 conv rows of 128 lanes per rank
    split = mk.with_cache(_exec(R=1, B=6, T=6), paged_kv=True)
    lay.validate(split)  # split path: 3 taps fit
    with pytest.raises(mk.MegakernelConfigError, match="in-kernel gather"):
        lay.validate(
            mk.with_cache(replace(split, B=4, T=4), pool_gather=True)
        )  # 8 * 24 = 192 rows
    narrow = replace(_layout(3), kh=4, vh=4)  # 12 conv rows per rank
    narrow.validate(
        mk.with_cache(replace(split, B=4, T=4), pool_gather=True)
    )  # 8 * 12 = 96 rows
    with pytest.raises(mk.MegakernelConfigError, match="in-kernel gather"):
        narrow.validate(mk.with_cache(split, pool_gather=True))  # 16 * 12 rows
    narrow.validate(
        mk.with_cache(replace(split, B=4, T=4), fused_pool_io=True)
    )  # 16 * 12 = 192 rows
    with pytest.raises(mk.MegakernelConfigError, match="fused pool I/O"):
        lay.validate(
            mk.with_cache(replace(split, B=1, T=1), fused_pool_io=True)
        )  # 16 * 24 rows


@pytest.mark.parametrize("kh", [2, 4, 6, 8])
@pytest.mark.parametrize("T", [2, 4, 5, 6, 8])
def test_layout_validate_and_can_pool_gather_agree_on_gpr_conv_window(kh: int, T: int):
    lay = replace(_layout(3), kh=kh, vh=kh)
    ex = mk.with_cache(_exec(R=1, B=8 * T, T=T), pool_gather=True)
    fits = lay.gpr_conv_fits(T, R=1)
    assert pool.can_pool_gather(lay, ex, lambda _ex: (8 * T, 2)) == fits
    if fits:
        lay.validate(ex)
    else:
        with pytest.raises(mk.MegakernelConfigError, match="in-kernel gather"):
            lay.validate(ex)


def test_make_pool_backend_supports_pool_gather_only_with_rec_chunk():
    lay = _layout(3)
    assert not mk.make_pool_backend(lay).supports_pool_gather
    assert mk.make_pool_backend(lay, rec_chunk=lambda ex: (8, 2)).supports_pool_gather
    assert not mk.make_pool_backend(
        lay, rec_chunk=lambda ex: (8, 2), supports_pool_gather=False
    ).supports_pool_gather
    with pytest.raises(mk.MegakernelConfigError, match="rec_chunk"):
        mk.make_pool_backend(lay, supports_pool_gather=True)


def test_gqa_pool_layout_must_match_the_pool_backend_layout():
    spec, ex = _gpr_spec_and_exec()
    lay = spec.cache_backend.layout
    mk.planner.plan(spec, ex)
    wrong = replace(spec, cache_backend=mk.make_pool_backend(replace(lay, kvh=2)))
    with pytest.raises(mk.MegakernelConfigError, match="pool backend layout"):
        mk.planner.plan(wrong, mk.with_cache(ex, pool_gather=False))


def _gpr_spec_and_exec() -> tuple[mk.ModelSpec, mk.ExecSpec]:
    g = qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4,
    )  # fmt: skip
    return qwen35.qwen35_spec(g), mk.with_cache(_exec(R=2, B=8, T=4), pool_gather=True)


def test_gdn_in_kernel_pool_io_takes_slot_geometry_from_layout():
    spec, ex = _gpr_spec_and_exec()
    assert mk.cache_exec(ex).GPR
    gp = mk.planner.plan(spec, ex).block_params["gdn"]
    lay = spec.cache_backend.layout
    assert gp.pool_layout is lay
    assert (
        lay.rec_pages_per_slot,
        lay.pages_per_slot,
        lay.slots_per_pool,
        lay.rec_u32_rows,
    ) == (gp.REC_PAGES, gp.GDN_PAGES, gp.G_PER_ATT, gp.SSM_U32_CAP)


def test_gdn_in_kernel_pool_io_rejects_mismatched_layout():
    spec, ex = _gpr_spec_and_exec()
    lay = spec.cache_backend.layout
    wrong = replace(
        spec, cache_backend=mk.make_pool_backend(replace(lay, slots_per_pool=1))
    )
    with pytest.raises(mk.MegakernelConfigError, match="slots_per_pool=1"):
        mk.planner.plan(wrong, ex)
    with pytest.raises(mk.MegakernelConfigError, match="pool backend"):
        mk.planner.plan(replace(spec, cache_backend=None), ex)


def test_qwen35_pool_layout_matches_geometry():
    g = qwen35.Qwen35Geometry()
    lay = mk.qwen35_pool_layout(g)
    assert (lay.num_pools, lay.slots_per_pool) == (g.NATT, g.att_every - 1)
    assert lay.num_recurrent_layers == g.NGDN
    for R in (1, 2, 8):
        gdn = mk.GDNBlock.lin_in_layout(g, R)
        assert (lay.kh_per_rank(R), lay.vh_per_rank(R)) == (gdn.kh, gdn.vh)
        assert lay.qkv_per_rank(R) == gdn.QKV
    assert mk.qwen35_pool_backend(g).layout == lay


@pytest.mark.parametrize("name", list(SLOTS))
def test_page_pack_unpack_roundtrip(name: str):
    lay = _layout(SLOTS[name])
    R = 2
    k = jax.random.split(jax.random.key(3), 4)
    kp = (jax.random.normal(k[0], (lay.page_size, lay.hd)) * 0.3).astype(FP8)
    vp = (jax.random.normal(k[1], (lay.page_size, lay.hd)) * 0.3).astype(FP8)
    raw = pool.kv_to_raw_page(kp, vp)
    assert raw.shape == lay.kv_page_shape
    k_rt, v_rt = pool.raw_page_to_kv(raw)
    np.testing.assert_array_equal(_bytes(k_rt), _bytes(kp))
    np.testing.assert_array_equal(_bytes(v_rt), _bytes(vp))

    vh, qkv = lay.vh_per_rank(R), lay.qkv_per_rank(R)
    rec = (jax.random.normal(k[2], (vh, lay.KD, lay.VD)) * 0.1).astype(jnp.bfloat16)
    raw_rec = pool.ssm_to_raw_pages(rec, lay)
    assert raw_rec.shape == (lay.rec_pages_per_slot, *lay.kv_page_shape)
    rec_rt = pool.raw_pages_to_ssm(raw_rec, vh, lay.KD, lay.VD, lay)
    np.testing.assert_array_equal(np.asarray(rec_rt), np.asarray(rec))

    taps = _bf16_f32(k[3], (lay.conv_taps, qkv))
    raw_conv = pool.conv_to_raw_page(taps, lay)
    assert raw_conv.shape == lay.kv_page_shape
    np.testing.assert_array_equal(
        np.asarray(pool.raw_page_to_conv(raw_conv, qkv, lay)), np.asarray(taps)
    )


@pytest.mark.parametrize(
    "name,ckpt_ndim",
    [("qwen35", 3), ("qwen35", 2), ("slots1", 3), ("slots1", 2), ("kv_only", 3)],
)
def test_states_to_pool_roundtrip(name: str, ckpt_ndim: int):
    lay = _layout(SLOTS[name])
    ex = _exec(R=2, B=2, T=1)
    st = _random_states(lay, ex, seed=11 + SLOTS[name])
    pt, ckpt, roff = _tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    assert len(raw_pools) == lay.num_pools
    assert raw_pools[0].shape == (
        ex.R * mk.cache_exec(ex).num_pages,
        *lay.kv_page_shape,
    )
    out = pool.pool_to_states(lay, ex, raw_pools, pt, ckpt)
    np.testing.assert_array_equal(_bytes(out["kc"]), _bytes(st["kc"]))
    np.testing.assert_array_equal(_bytes(out["vc"]), _bytes(st["vc"]))
    if not lay.has_recurrent:
        assert "rec" not in out and "conv" not in out
        return
    np.testing.assert_array_equal(np.asarray(out["rec"]), np.asarray(st["rec"]))
    np.testing.assert_array_equal(
        np.asarray(out["conv"][:, :, :, 0]), np.asarray(st["conv"])
    )


@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize(
    "name,ckpt_ndim", [("qwen35", 3), ("qwen35", 2), ("slots1", 3), ("slots1", 2)]
)
def test_gather_matches_packed_states(name: str, ckpt_ndim: int, T: int):
    lay = _layout(SLOTS[name])
    ex = _exec(R=2, B=2 * T, T=T)
    st = _random_states(lay, ex, seed=21 + T)
    pt, ckpt, roff = _tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in raw_pools)

    conv, rec = pool.gather_gdn_states(lay, ex, raw_r0, ckpt, roff)
    NS, B = ex.NS, ex.B
    taps = np.asarray(st["conv"][0])  # (ngdn, NS, 3, QKV)
    if T == 1:
        exp = np.zeros(conv.shape, np.float32)
        exp[:, : 3 * B] = np.swapaxes(taps, 1, 2).reshape(taps.shape[0], 3 * B, -1)
    else:
        exp = np.zeros(conv.shape, np.float32)
        exp[:, :, :3] = taps
    np.testing.assert_array_equal(np.asarray(conv), exp)
    np.testing.assert_array_equal(np.asarray(rec), np.asarray(st["rec"][0]))
    assert rec.shape[1] == NS * lay.vh_per_rank(ex.R)

    # The generic backend wraps the same kernel.
    backend = mk.make_pool_backend(lay)
    got = backend.gather_states(ex, raw_r0, ckpt, roff, None)
    np.testing.assert_array_equal(np.asarray(got["rec"]), np.asarray(rec))


def test_kv_only_backend_skips_gather():
    lay = _layout(0)
    backend = mk.make_pool_backend(lay)
    assert backend.gather_states(_exec(R=1, B=2, T=1), (), None, None, None) == {}


def _check_scatter(
    lay: mk.PoolLayout,
    ex: mk.ExecSpec,
    ckpt_ndim: int,
    seed: int,
    *,
    cold_start: bool = False,
) -> None:
    """Scatter on rank 0 of `ex.R`; check every rank's KV and checkpoints.

    `cold_start` scatters through a `K = T + 1` checkpoint table whose first
    column is a fresh read slot (`read_offsets = -1`), so checkpoint `t` goes to
    column `t + 1`.
    """
    R, NS, B, T, NATT = ex.R, ex.NS, ex.B, ex.T, lay.num_pools
    st = _random_states(lay, ex, seed=seed)
    pt, ckpt, roff = _tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    before = pool.pool_to_states(lay, ex, raw_pools, pt, ckpt)
    ckpt_in = ckpt
    if cold_start and lay.has_recurrent:
        fresh = int(jnp.max(ckpt)) + 1 + jnp.arange(ckpt[..., 0].size, dtype=ckpt.dtype)
        ckpt_in = jnp.concatenate([fresh.reshape(ckpt[..., :1].shape), ckpt], axis=-1)
        pool.check_ckpt_bounds(ckpt_in, mk.cache_exec(ex).num_pages, lay)
        conv0, rec0 = pool.gather_gdn_states(
            lay,
            ex,
            tuple(mk.unfold(r, R)[0] for r in raw_pools),
            ckpt_in,
            jnp.full_like(roff, -1),
        )
        assert not np.any(np.asarray(conv0)) and not np.any(np.asarray(rec0))
    k = jax.random.split(jax.random.key(seed + 10), 4)
    # Positions stay inside `ctx`; sequences alternate between the two KV pages.
    pos = jnp.asarray(
        [10 + 129 * (s % 2) + 3 * (s // 2) + t for s in range(NS) for t in range(T)],
        dtype=jnp.int32,
    )
    knew = (jax.random.normal(k[0], (NATT, ex.MR, lay.hd)) * 0.3).astype(FP8)
    vnew = (jax.random.normal(k[1], (NATT, ex.MR, lay.hd)) * 0.3).astype(FP8)
    conv = rec = None
    if lay.has_recurrent:
        ngdn, vh, qkv = (
            lay.num_recurrent_layers,
            lay.vh_per_rank(R),
            lay.qkv_per_rank(R),
        )
        conv_shape = (
            (ngdn, 8 * ((3 * B + 7) // 8), qkv)
            if T == 1
            else (ngdn, NS, 8 * ((3 + T + 7) // 8), qkv)
        )
        conv = _bf16_f32(k[2], conv_shape)
        rec = (jax.random.normal(k[3], (ngdn, B * vh, lay.KD, lay.VD)) * 0.1).astype(
            jnp.bfloat16
        )
    out_r0 = pool.scatter_pool_states(
        lay,
        ex,
        tuple(mk.unfold(r, R)[0] for r in raw_pools),
        pos,
        pt,
        knew,
        vnew,
        conv,
        rec,
        ckpt_in if lay.has_recurrent else None,
    )
    out_pools = tuple(
        mk.unfold(r, R).at[0].set(o).reshape(r.shape) for r, o in zip(raw_pools, out_r0)
    )
    out = pool.pool_to_states(lay, ex, out_pools, pt, ckpt)

    # Other ranks are untouched.
    for key, val in before.items():
        np.testing.assert_array_equal(_bytes(out[key][1:]), _bytes(val[1:]))
    kc_exp = np.array(_bytes(before["kc"][0]))
    vc_exp = np.array(_bytes(before["vc"][0]))
    for b in range(B):
        s = b // T
        kc_exp[:, s, int(pos[b])] = _bytes(knew[:, b])
        vc_exp[:, s, int(pos[b])] = _bytes(vnew[:, b])
    np.testing.assert_array_equal(_bytes(out["kc"][0]), kc_exp)
    np.testing.assert_array_equal(_bytes(out["vc"][0]), vc_exp)
    if not lay.has_recurrent:
        return
    np.testing.assert_array_equal(np.asarray(out["rec"][0]), np.asarray(rec))
    conv_np = np.asarray(conv)
    for s in range(NS):
        for t in range(T):
            if T == 1:
                exp_taps = conv_np[:, [j * B + s for j in range(3)]]
            else:
                exp_taps = conv_np[:, s, 1 + t : 4 + t]
            np.testing.assert_array_equal(np.asarray(out["conv"][0, :, s, t]), exp_taps)


def test_scatter_direct_conv_window_when_the_caller_conv_is_short():
    """At `T=5` the caller's `rup(3 + T, 8)` conv rows are one short of the
    `taps + 1 + T` rows of the exact-window direct path (12 lane-rows per rank):
    the scatter falls back to the packed copy instead of failing."""
    lay = replace(_layout(1, num_pools=1), kh=4, vh=4)
    assert lay.qkv_per_rank(1) // 128 == 12
    _check_scatter(lay, _exec(R=1, B=10, T=5), 3, seed=96)


@pytest.mark.parametrize("T", [2, 5])
def test_scatter_direct_conv_window_traces_for_24_lane_rows(T: int):
    """24 lane-rows per rank (e.g. full Qwen3.5 at R=4) has no exact `taps + 1`
    row window of its 80 padded conv rows, so the direct path reads `taps`
    rows. Traced for TPU (`interpret=False`): on CPU interpret this width also
    fills the recurrent pages exactly, which XLA:CPU fails to compile."""
    lay = replace(_layout(1, num_pools=1), kh=8, vh=8)
    qkv = lay.qkv_per_rank(1)
    assert qkv // 128 == 24
    ex = replace(_exec(R=1, B=2 * T, T=T), interpret=False)
    pt, ckpt, _ = _tables(lay, ex, 3)
    vh, NS = lay.vh_per_rank(1), ex.NS
    args = (
        (jnp.zeros((NUM_PAGES, *lay.kv_page_shape), FP8),),
        jnp.arange(ex.B, dtype=jnp.int32),
        pt,
        jnp.zeros((1, ex.MR, lay.hd), FP8),
        jnp.zeros((1, ex.MR, lay.hd), FP8),
        jnp.zeros((1, NS, 8 * ((3 + T + 7) // 8), qkv), jnp.float32),
        jnp.zeros((1, ex.B * vh, lay.KD, lay.VD), jnp.bfloat16),
    )
    jaxpr = jax.make_jaxpr(
        lambda *a: pool.scatter_pool_states(lay, ex, *a, ckpt_indices=ckpt)
    )(*args)
    assert "qwen3_5_scatter_pool_states" in str(jaxpr)


@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize(
    "name,ckpt_ndim",
    [("qwen35", 3), ("qwen35", 2), ("slots1", 3), ("slots1", 2), ("kv_only", 3)],
)
def test_scatter_writes_kv_and_checkpoints(name: str, ckpt_ndim: int, T: int):
    _check_scatter(_layout(SLOTS[name]), _exec(R=1, B=2 * T, T=T), ckpt_ndim, 31 + T)


@pytest.mark.parametrize("NS", [1, 3, 4])
def test_gather_and_scatter_sequence_counts(NS: int):
    """A 2-slot, 3-pool layout with 1, 3 and 4 sequences (T=2, 3D ckpt)."""
    lay, T = _layout(2, num_pools=3), 2
    ex = _exec(R=1, B=NS * T, T=T)
    st = _random_states(lay, ex, seed=51 + NS)
    pt, ckpt, roff = _tables(lay, ex, 3)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    conv, rec = pool.gather_gdn_states(lay, ex, raw_pools, ckpt, roff)
    exp = np.zeros(conv.shape, np.float32)
    exp[:, :, :3] = np.asarray(st["conv"][0])
    np.testing.assert_array_equal(np.asarray(conv), exp)
    np.testing.assert_array_equal(np.asarray(rec), np.asarray(st["rec"][0]))
    _check_scatter(lay, ex, 3, seed=61 + NS)


@pytest.mark.parametrize("name,ckpt_ndim", [("slots1", 3), ("qwen35", 2)])
def test_scatter_on_one_rank_of_two(name: str, ckpt_ndim: int):
    """Rank-local scatter (`R=2`): rank 0 pools change, rank 1 pools don't."""
    _check_scatter(_layout(SLOTS[name]), _exec(R=2, B=4, T=2), ckpt_ndim, seed=71)


def test_cold_start_ckpt_k_plus_one():
    """`K = T + 1` checkpoint table with `read_offsets = -1` (no prior state)."""
    lay = _layout(2, num_pools=3)
    ex = mk.with_cache(_exec(R=1, B=6, T=2), num_pages=2 * NUM_PAGES)
    _check_scatter(lay, ex, 3, seed=81, cold_start=True)


def test_ckpt_bounds_rejects_out_of_range_and_shared_slots():
    lay = _layout(2, num_pools=3)
    ex = _exec(R=1, B=4, T=2)
    _, ckpt, _ = _tables(lay, ex, 3)
    pool.check_ckpt_bounds(ckpt, mk.cache_exec(ex).num_pages, lay)
    past_end = mk.cache_exec(ex).num_pages // lay.pages_per_slot
    with pytest.raises(ValueError, match="exceeds pool capacity"):
        pool.check_ckpt_bounds(
            ckpt.at[1, 0, 1].set(past_end), mk.cache_exec(ex).num_pages, lay
        )
    with pytest.raises(ValueError, match="shared between"):
        pool.check_ckpt_bounds(
            ckpt.at[1, 0, 0].set(ckpt[0, 0, 0]), mk.cache_exec(ex).num_pages, lay
        )


def test_check_pooled_step_hlo_uses_layout_page_shape():
    """Full-pool copies are matched on the layout's page shape."""
    lay = mk.PoolLayout(hd=128, kvh=2)  # pages (4, 32, 4, 128)
    head = "tpu_custom_call tpu_custom_call scatter_pool_states\n"
    copy_lay = head + "%c = f8e4m3fn[64,4,32,4,128] copy(%p)"
    copy_default = head + "%c = f8e4m3fn[64,2,64,4,128] copy(%p)"
    staging = head + "%c = f8e4m3fn[5,4,32,4,128] copy(%p)"
    assert not mk.check_pooled_step_hlo(copy_lay, layout=lay)[0]
    assert mk.check_pooled_step_hlo(copy_default, layout=lay)[0]
    assert mk.check_pooled_step_hlo(staging, layout=lay)[0]
    assert not mk.check_pooled_step_hlo(copy_default)[0]


def test_full_pooled_step_att_every_2():
    """One `PallasWriter` pooled step on a 1-slot-per-pool Qwen3.5 geometry."""
    devs = jax.devices("cpu")
    if len(devs) < 2:
        pytest.skip(f"Need at least 2 CPU devices, have {len(devs)}")
    mesh = Mesh(np.array(devs[:2]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = qwen35.Qwen35Geometry(
        L=4,
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        att_every=2,
    )
    ex = mk.with_cache(_exec(R=2, B=2, T=1), num_pages=96)
    lay = mk.qwen35_pool_layout(g)
    assert (lay.num_pools, lay.slots_per_pool) == (2, 1)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(5), g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, st_mk_paged = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    raw_pools = tuple(
        jax.device_put(r, tp) for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    )
    mk_paged_step, _ = qwen35.make_step(mesh, g, ex, donate_states=False)
    ref_hid, mk_st = mk_paged_step(pos, hid_mk, w_mk, st_mk_paged)

    spec = qwen35.qwen35_spec(g)
    assert spec.cache_backend.layout == lay
    st_before = mk.pool_to_states(g, ex, raw_pools, pt, ckpt)
    step, _ = mk.PallasWriter().make_step(mesh, spec, ex, donate=False)
    out_hid, raw_out = step(pos, hid_mk, w_mk, raw_pools, pt, ckpt, roff)
    np.testing.assert_array_equal(
        np.asarray(out_hid.astype(jnp.float32)), np.asarray(ref_hid.astype(jnp.float32))
    )

    # Whole KV cache: the pre-step pools with each row's new K / V at `pos`.
    st_after = mk.pool_to_states(g, ex, raw_out, pt, ckpt)
    for key, new_key in (("kc", "knew"), ("vc", "vnew")):
        exp = np.array(_bytes(st_before[key]))
        new_kv = mk.unfold(mk_st[new_key], ex.R)
        for b in range(ex.B):
            exp[:, :, b, int(pos[b]), :] = _bytes(new_kv[:, :, b])
        np.testing.assert_array_equal(_bytes(st_after[key]), exp)
    np.testing.assert_array_equal(
        np.asarray(st_after["rec"]), np.asarray(mk.unfold(mk_st["rec"], ex.R))
    )
    QKV = lay.qkv_per_rank(ex.R)
    mk_conv = (
        mk.unfold(mk_st["conv"], ex.R)[:, :, : 3 * ex.B, :]
        .reshape(ex.R, g.NGDN, 3, ex.B, QKV)
        .swapaxes(2, 3)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(st_after["conv"][:, :, :, 0]), np.asarray(mk_conv)
    )


# ---------------------------------------------------------------------------
# AST guards: the pool state name and the slot stride live in one place.
# ---------------------------------------------------------------------------
_MK_ROOT = Path(mk.__file__).resolve().parent


def _guarded_files() -> list[Path]:
    files = [_MK_ROOT / "kernel.py", *sorted(_MK_ROOT.glob("cache*.py"))]
    files += sorted((_MK_ROOT / "blocks").rglob("*.py"))
    files += sorted((_MK_ROOT / "experimental").glob("*.py"))
    files += sorted((_MK_ROOT / "models").glob("*.py"))
    assert len(files) >= 6, files
    return files


def test_no_raw_pool_state_literal_in_core():
    """`"raw"` is defined once (`POOL_STATE_NAME`), never repeated elsewhere."""
    for path in _guarded_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "raw":
                raise AssertionError(
                    f"{path.relative_to(_MK_ROOT)}:{node.lineno} uses the pool "
                    "state literal 'raw'; use POOL_STATE_NAME or "
                    "KernelContext.raw_refs"
                )


def test_no_slot_stride_literal_in_core():
    """No `* 5` / `5 *` page-stride literal: strides come from `PoolLayout`."""
    for path in _guarded_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
                continue
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and side.value == 5:
                    raise AssertionError(
                        f"{path.relative_to(_MK_ROOT)}:{node.lineno} multiplies "
                        "by the literal 5; use PoolLayout.pages_per_slot"
                    )
