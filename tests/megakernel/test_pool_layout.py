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

Covers the Qwen3.5 layout (3 recurrent slots per pool), a 1-slot layout and a
KV-only layout (0 slots) through pack / unpack, `states_to_pool` /
`pool_to_states`, the Pallas gather and scatter kernels (CPU interpret) with 2D
and 3D checkpoint indices, one full pooled step with `att_every=2`, and AST
guards that keep the pool state name and the slot stride out of core code.
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

FP8 = jnp.float8_e4m3fn
NUM_PAGES = 128
# Recurrent slots per pool: Qwen3.5 (att_every=4), att_every=2, KV-only.
SLOTS = {"qwen35": 3, "slots1": 1, "kv_only": 0}


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _layout(slots: int, num_pools: int = 2) -> mk.PoolLayout:
    return mk.PoolLayout(
        num_pools=num_pools,
        slots_per_pool=slots,
        hd=256,
        kvh=1,
        kh=2,
        vh=4,
        KD=128,
        VD=128,
    )


def _exec(R: int, B: int, T: int, ctx: int = 256) -> mk.ExecSpec:
    tiles = replace(
        mk.spec.TINY_TILES,
        bank=512 << 10,
        kv_blk=min(mk.spec.TINY_TILES.kv_blk, ctx),
    )
    return mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        paged_kv=True,
        num_pages=NUM_PAGES,
        stack_e=True,
        interpret=True,
        tiles=tiles,
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
    assert ex.num_pages >= min_pages
    assert int(jnp.min(pt[:, : ex.ctx // ex.page_size])) >= lay.reserved_pages
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
    assert backend.pool_state_name == mk.POOL_STATE_NAME
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
        _layout(3).validate(replace(_exec(R=2, B=2, T=1), page_size=64))


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
    assert raw_pools[0].shape == (ex.R * ex.num_pages, *lay.kv_page_shape)
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


@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize(
    "name,ckpt_ndim",
    [("qwen35", 3), ("qwen35", 2), ("slots1", 3), ("slots1", 2), ("kv_only", 3)],
)
def test_scatter_writes_kv_and_checkpoints(name: str, ckpt_ndim: int, T: int):
    lay = _layout(SLOTS[name])
    ex = _exec(R=1, B=2 * T, T=T)
    NS, B, NATT = ex.NS, ex.B, lay.num_pools
    st = _random_states(lay, ex, seed=31 + T)
    pt, ckpt, roff = _tables(lay, ex, ckpt_ndim)
    raw_pools = pool.states_to_pool(lay, ex, st, pt, ckpt, roff)
    k = jax.random.split(jax.random.key(41 + T), 4)
    pos = jnp.asarray(
        [10 + 129 * s + t for s in range(NS) for t in range(T)], dtype=jnp.int32
    )
    knew = (jax.random.normal(k[0], (NATT, ex.MR, lay.hd)) * 0.3).astype(FP8)
    vnew = (jax.random.normal(k[1], (NATT, ex.MR, lay.hd)) * 0.3).astype(FP8)
    conv = rec = None
    if lay.has_recurrent:
        ngdn, vh, qkv = lay.num_recurrent_layers, lay.vh, lay.qkv_per_rank(1)
        conv_shape = (
            (ngdn, 8 * ((3 * B + 7) // 8), qkv)
            if T == 1
            else (ngdn, NS, 8 * ((3 + T + 7) // 8), qkv)
        )
        conv = _bf16_f32(k[2], conv_shape)
        rec = (jax.random.normal(k[3], (ngdn, B * vh, lay.KD, lay.VD)) * 0.1).astype(
            jnp.bfloat16
        )
    out_pools = pool.scatter_pool_states(
        lay,
        ex,
        raw_pools,
        pos,
        pt,
        knew,
        vnew,
        conv,
        rec,
        ckpt if lay.has_recurrent else None,
    )
    out = pool.pool_to_states(lay, ex, out_pools, pt, ckpt)

    kc_exp = np.array(_bytes(st["kc"]))
    vc_exp = np.array(_bytes(st["vc"]))
    for b in range(B):
        s = b // T
        kc_exp[0, :, s, int(pos[b])] = _bytes(knew[:, b])
        vc_exp[0, :, s, int(pos[b])] = _bytes(vnew[:, b])
    np.testing.assert_array_equal(_bytes(out["kc"]), kc_exp)
    np.testing.assert_array_equal(_bytes(out["vc"]), vc_exp)
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
    ex = replace(_exec(R=2, B=2, T=1), num_pages=96)
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
    _, mk_st = mk_paged_step(pos, hid_mk, w_mk, st_mk_paged)

    spec = qwen35.qwen35_spec(g)
    assert spec.pool_backend.layout == lay
    step, _ = mk.PallasWriter().make_step(mesh, spec, ex, donate=False)
    out_hid, raw_out = step(pos, hid_mk, w_mk, raw_pools, pt, ckpt, roff)
    assert bool(jnp.all(jnp.isfinite(out_hid.astype(jnp.float32))))

    st_after = mk.pool_to_states(g, ex, raw_out, pt, ckpt)
    mk_knew = mk.unfold(mk_st["knew"], ex.R)
    for b in range(ex.B):
        np.testing.assert_array_equal(
            _bytes(st_after["kc"][:, :, b, int(pos[b]), :]), _bytes(mk_knew[:, :, b])
        )
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
    files += sorted((_MK_ROOT / "blocks").glob("*.py"))
    files += sorted((_MK_ROOT / "experimental").glob("*.py"))
    assert len(files) >= 6, files
    return files


def test_no_raw_pool_state_literal_in_core():
    """`"raw"` is defined once (`spec.POOL_STATE_NAME`), never repeated in core."""
    for path in _guarded_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "raw":
                raise AssertionError(
                    f"{path.relative_to(_MK_ROOT)}:{node.lineno} uses the pool "
                    "state literal 'raw'; use PoolBackend.pool_state_name, "
                    "KernelContext.raw_refs or spec.POOL_STATE_NAME"
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
