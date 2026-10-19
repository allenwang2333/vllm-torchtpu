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
"""Host / test packing of logical states into raw pools (and back)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import kernel
from ..blocks.gdn import GDNBlock
from ..ops import BF16, F32, FP8
from ..spec import ExecSpec
from .backend import _check_ckpt_bounds


# ---------------------------------------------------------------------------
# Per-page Host / Test Packing & Unpacking Utilities
# ---------------------------------------------------------------------------
def kv_to_raw_page(k_page: Any, v_page: Any) -> jax.Array:
    """Pack `(page_size, HD)` K and V into `(2, HD // 4, 4, page_size) fp8`."""
    ps, hd = k_page.shape[0], k_page.shape[-1]
    assert hd % 4 == 0, hd
    h4 = hd // 4
    k_fp8 = (
        jnp.clip(k_page, -448.0, 448.0).astype(FP8) if k_page.dtype != FP8 else k_page
    )
    v_fp8 = (
        jnp.clip(v_page, -448.0, 448.0).astype(FP8) if v_page.dtype != FP8 else v_page
    )
    k_pk = jnp.transpose(k_fp8.reshape(ps, h4, 4), (1, 2, 0))
    v_pk = jnp.transpose(v_fp8.reshape(ps, h4, 4), (1, 2, 0))
    return jnp.stack([k_pk, v_pk], axis=0)


def raw_page_to_kv(raw_page: Any) -> tuple[jax.Array, jax.Array]:
    """Unpack `(2, HD // 4, 4, page_size) fp8` into `(k_page, v_page)`."""
    ps = raw_page.shape[-1]
    hd = raw_page.shape[1] * raw_page.shape[2]
    k_page = jnp.transpose(raw_page[0], (2, 0, 1)).reshape(ps, hd)
    v_page = jnp.transpose(raw_page[1], (2, 0, 1)).reshape(ps, hd)
    return k_page, v_page


def ssm_to_raw_pages(rec_slice: Any) -> jax.Array:
    """Pack key-major `(vh, KD, VD) bf16` into 4 raw pages `(4, 2, 64, 4, 128) fp8`."""
    vh, KD, VD = rec_slice.shape
    ssm_u32_rows = (vh * KD) // 2
    s_t = np.asarray(rec_slice, dtype=jnp.dtype(BF16)).reshape(vh * KD, VD)
    u32_val = (
        s_t.reshape(ssm_u32_rows, 2, 128)
        .transpose(0, 2, 1)
        .copy()
        .view(np.uint32)
        .reshape(ssm_u32_rows, 128)
    )
    if ssm_u32_rows < 512:
        u32_val = np.pad(u32_val, ((0, 512 - ssm_u32_rows), (0, 0)))
    fp8_4p = (
        u32_val.reshape(4, 2, 64, 128, 1).view(jnp.dtype(FP8)).transpose(0, 1, 2, 4, 3)
    )
    return jnp.asarray(fp8_4p)


def raw_pages_to_ssm(
    raw_4pages: Any, vh: int, KD: int = 128, VD: int = 128
) -> jax.Array:
    """Unpack 4 raw pages `(4, 2, 64, 4, 128) fp8` into `(vh, KD, VD) bf16`."""
    ssm_u32_rows = (vh * KD) // 2
    arr_u8 = np.asarray(raw_4pages).view(np.uint8)
    u32_val = (
        arr_u8.transpose(0, 1, 2, 4, 3)
        .copy()
        .view(np.uint32)
        .reshape(512, 128)[:ssm_u32_rows, :]
    )
    bf16_pairs = (
        u32_val.reshape(ssm_u32_rows, 128, 1)
        .view(jnp.dtype(BF16))
        .transpose(0, 2, 1)
        .reshape(vh, KD, VD)
    )
    return jnp.asarray(bf16_pairs)


def conv_to_raw_page(taps_3_qkv: Any) -> jax.Array:
    """Pack `(3, QKV)` conv taps into 1 raw page `(2, 64, 4, 128) fp8`."""
    QKV = taps_3_qkv.shape[-1]
    conv_rows = (3 * QKV) // 128
    taps_bf16 = np.asarray(taps_3_qkv, dtype=jnp.dtype(BF16)).reshape(conv_rows, 128)
    padded_256 = np.pad(taps_bf16, ((0, 256 - conv_rows), (0, 0)))
    fp8_1p = (
        padded_256.reshape(2, 64, 2, 128, 1)
        .view(jnp.dtype(FP8))
        .transpose(0, 1, 2, 4, 3)
    )
    fp8_1p = fp8_1p.reshape(2, 64, 4, 128)
    return jnp.asarray(fp8_1p)


def raw_page_to_conv(raw_1page: Any, QKV: int) -> jax.Array:
    """Unpack 1 raw page `(2, 64, 4, 128) fp8` into `(3, QKV) f32`."""
    conv_rows = (3 * QKV) // 128
    arr_u8 = np.asarray(raw_1page).view(np.uint8)
    bf16_256 = (
        arr_u8.reshape(2, 64, 2, 2, 128)
        .transpose(0, 1, 2, 4, 3)
        .copy()
        .view(jnp.dtype(BF16))
        .reshape(256, 128)[:conv_rows, :]
    )
    return jnp.asarray(bf16_256).reshape(3, QKV).astype(F32)


def default_pool_layout(
    g: Any, ex: ExecSpec
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Allocate disjoint `page_table`, `ckpt_indices`, `read_offsets`, and `min_pages`.

    Pages `0 .. max(1, 5 * G_PER_ATT) - 1` are reserved as dummy/null pages so
    padded or inactive rows that default to page index `0` or GDN slot `0` never
    overwrite active sequence `0` (whose KV pages start at `max(1, 5 * G_PER_ATT)`).
    Checkpoint slots start at `base_slot >= 1` so `ckpt[0, s, 0] > 0` holds for
    all active sequences under `active_seqs="from_ckpt"`.
    """
    G_PER_ATT = g.att_every - 1
    reserved_pages = max(1, 5 * G_PER_ATT)
    pages_per_seq = ex.ctx // ex.page_size
    kv_pages_total = reserved_pages + ex.NS * pages_per_seq
    pt_cols = max(ex.resolved_max_pages_per_seq, pages_per_seq)
    pt = np.zeros((ex.NS, pt_cols), dtype=np.int32)
    for s in range(ex.NS):
        pt[s, :pages_per_seq] = np.arange(
            reserved_pages + s * pages_per_seq,
            reserved_pages + (s + 1) * pages_per_seq,
            dtype=np.int32,
        )

    if G_PER_ATT > 0:
        n_ckpt_slots = G_PER_ATT * ex.NS * ex.T
        base_slot = max(1, (kv_pages_total + 4) // 5)
        min_num_pages = (base_slot + n_ckpt_slots) * 5
        ckpt = np.zeros((G_PER_ATT, ex.NS, ex.T), dtype=np.int32)
        for sub in range(G_PER_ATT):
            for s in range(ex.NS):
                for k in range(ex.T):
                    ckpt[sub, s, k] = base_slot + (sub * ex.NS + s) * ex.T + k
        roff = np.zeros((G_PER_ATT, ex.NS), dtype=np.int32)
    else:
        min_num_pages = kv_pages_total
        ckpt = np.zeros((1, ex.NS, ex.T), dtype=np.int32)
        roff = np.zeros((1, ex.NS), dtype=np.int32)

    return jnp.asarray(pt), jnp.asarray(ckpt), jnp.asarray(roff), min_num_pages


def states_to_pool(
    g: Any,
    ex: ExecSpec,
    states: dict[str, Any],
    page_table: Any,
    ckpt_indices: Any = None,
    read_offsets: Any = None,
) -> tuple[jax.Array, ...]:
    """Pack unfolded or folded states into `g.NATT` raw pools."""
    R, NS, B = ex.R, ex.NS, ex.B
    AE = g.att_every
    G_PER_ATT = AE - 1
    pt = np.asarray(page_table)
    pages_per_seq = ex.ctx // ex.page_size

    kc = states["kc"]
    vc = states["vc"]
    if kc.ndim == 4:
        kc = kernel.unfold(kc, R)
        vc = kernel.unfold(vc, R)

    ckpt: Any = None
    roff: Any = None
    conv_st: Any = None
    rec_st: Any = None
    if G_PER_ATT > 0:
        assert ckpt_indices is not None
        _check_ckpt_bounds(ckpt_indices, ex.num_pages, G_PER_ATT)
        ckpt = np.asarray(ckpt_indices)
        ckpt_is_3d = ckpt.ndim == 3
        if read_offsets is None:
            roff = np.zeros(
                (max(1, G_PER_ATT), NS) if ckpt_is_3d else (NS,), dtype=np.int32
            )
        else:
            roff = np.asarray(read_offsets)
        roff_is_2d = roff.ndim == 2
        lay = GDNBlock.lin_in_layout(g, ex)
        vh = lay.vh
        conv_st = states["conv"]
        rec_st = states["rec"]
        if conv_st.shape[0] != R:
            conv_st = kernel.unfold(conv_st, R)
        if rec_st.shape[0] != R:
            rec_st = kernel.unfold(rec_st, R)
        if rec_st.ndim == 6:
            rec_st = rec_st.reshape(R, g.NGDN, NS * vh, lay.KD, lay.VD)
    else:
        ckpt_is_3d = roff_is_2d = False
        vh = 0

    raw_list = []
    for g_idx in range(g.NATT):
        pool_g = np.zeros(
            (R, ex.num_pages, 2, 64, 4, ex.page_size), dtype=jnp.dtype(FP8)
        )
        kc_g = np.asarray(jnp.clip(kc[:, g_idx], -448.0, 448.0).astype(FP8))
        vc_g = np.asarray(jnp.clip(vc[:, g_idx], -448.0, 448.0).astype(FP8))
        for r in range(R):
            for s in range(NS):
                for p in range(pages_per_seq):
                    kp = int(pt[s, p])
                    t0 = p * ex.page_size
                    k_sl = kc_g[r, s, t0 : t0 + ex.page_size]
                    v_sl = vc_g[r, s, t0 : t0 + ex.page_size]
                    pool_g[r, kp] = np.asarray(kv_to_raw_page(k_sl, v_sl))

        for sub in range(G_PER_ATT):
            o = g_idx * G_PER_ATT + sub
            for r in range(R):
                for s in range(NS):
                    ro_val = int(roff[sub, s]) if roff_is_2d else int(roff[s])
                    if ro_val < 0:
                        continue
                    if ckpt_is_3d:
                        slot = int(ckpt[sub, s, ro_val])
                        page_base = slot * 5
                    else:
                        slot = int(ckpt[s, ro_val])
                        page_base = slot * (5 * G_PER_ATT) + sub * 5
                    rec_sl = rec_st[r, o, s * vh : (s + 1) * vh]
                    pool_g[r, page_base : page_base + 4] = np.asarray(
                        ssm_to_raw_pages(rec_sl)
                    )
                    if conv_st.ndim == 5 and conv_st.shape[3] >= 3:
                        taps = conv_st[r, o, s, 0:3]
                    else:
                        taps = jnp.stack(
                            [conv_st[r, o, j * B + s] for j in range(3)], axis=0
                        )
                    pool_g[r, page_base + 4] = np.asarray(conv_to_raw_page(taps))

        raw_list.append(
            jnp.asarray(pool_g.reshape(R * ex.num_pages, 2, 64, 4, ex.page_size))
        )
    return tuple(raw_list)


def pool_to_states(
    g: Any,
    ex: ExecSpec,
    raw_pools: tuple[jax.Array, ...] | list[jax.Array],
    page_table: Any,
    ckpt_indices: Any = None,
) -> dict[str, jax.Array]:
    """Unpack `g.NATT` rank-folded raw pools into unfolded logical states `[R, ...]`."""
    R, NS, B, T = ex.R, ex.NS, ex.B, ex.T
    AE = g.att_every
    G_PER_ATT = AE - 1
    pt = np.asarray(page_table)
    pages_per_seq = ex.ctx // ex.page_size

    kc_out = np.zeros((R, g.NATT, NS, ex.ctx, g.HD), dtype=jnp.dtype(FP8))
    vc_out = np.zeros((R, g.NATT, NS, ex.ctx, g.HD), dtype=jnp.dtype(FP8))

    ckpt: Any = None
    rec_out: Any = None
    conv_out: Any = None
    if G_PER_ATT > 0:
        assert ckpt_indices is not None
        _check_ckpt_bounds(ckpt_indices, ex.num_pages, G_PER_ATT)
        ckpt = np.asarray(ckpt_indices)
        ckpt_is_3d = ckpt.ndim == 3
        ckpt_k = ckpt.shape[-1]
        ckpt_t_off = 0 if ckpt_k == T else 1
        lay = GDNBlock.lin_in_layout(g, ex)
        vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
        rec_out = np.zeros((R, g.NGDN, B * vh, KD, VD), dtype=jnp.dtype(BF16))
        conv_out = np.zeros((R, g.NGDN, NS, T, 3, QKV), dtype=np.float32)
    else:
        ckpt_is_3d = False
        ckpt_t_off = 0
        vh = KD = VD = QKV = 0

    for g_idx in range(g.NATT):
        pool_g = np.asarray(raw_pools[g_idx]).reshape(
            R, ex.num_pages, 2, 64, 4, ex.page_size
        )
        for r in range(R):
            for s in range(NS):
                for p in range(pages_per_seq):
                    kp = int(pt[s, p])
                    t0 = p * ex.page_size
                    k_sl, v_sl = raw_page_to_kv(pool_g[r, kp])
                    kc_out[r, g_idx, s, t0 : t0 + ex.page_size] = np.asarray(k_sl)
                    vc_out[r, g_idx, s, t0 : t0 + ex.page_size] = np.asarray(v_sl)

        for sub in range(G_PER_ATT):
            o = g_idx * G_PER_ATT + sub
            for r in range(R):
                for s in range(NS):
                    for t in range(T):
                        if ckpt_is_3d:
                            slot = int(ckpt[sub, s, t + ckpt_t_off])
                            page_base = slot * 5
                        else:
                            slot = int(ckpt[s, t + ckpt_t_off])
                            page_base = slot * (5 * G_PER_ATT) + sub * 5
                        b = s * T + t
                        rec_out[r, o, b * vh : (b + 1) * vh] = np.asarray(
                            raw_pages_to_ssm(
                                pool_g[r, page_base : page_base + 4], vh, KD, VD
                            )
                        )
                        conv_out[r, o, s, t] = np.asarray(
                            raw_page_to_conv(pool_g[r, page_base + 4], QKV)
                        )

    out: dict[str, jax.Array] = {
        "kc": jnp.asarray(kc_out),
        "vc": jnp.asarray(vc_out),
    }
    if G_PER_ATT > 0:
        out["rec"] = jnp.asarray(rec_out)
        out["conv"] = jnp.asarray(conv_out)
    return out
