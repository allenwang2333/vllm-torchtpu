# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 raw-pool page packing, layout, and Pallas GDN gather kernel.

Each of the `g.NATT` attention layers owns one raw pool of shape
`[R, num_pages, 2, 64, 4, page_size] fp8_e4m3fn` (stored rank-folded as
`[R * num_pages, 2, 64, 4, page_size]`). A single pool holds:

1. **KV cache pages** (`1` page per `page_size=128` tokens per sequence):
   `[0]` = K page `(64, 4, 128) fp8`, `[1]` = V page `(64, 4, 128) fp8`, where
   `(64, 4)` packs `HD = 256` into 32-bit words along sublanes so a `bitcast` to
   `uint32` yields `(64, page_size) u32` ready for `pltpu.bitcast(..., fp8)` on
   TPU without lane transposes.
2. **GDN recurrent + conv checkpoint slots** (`5` consecutive pages per slot for
   each of the `G_PER_ATT = att_every - 1` GDN layers preceding the attention
   layer):
   - Pages `0..3`: recurrent SSM state `(vh, KD=128, VD=128) bf16` packed as
     `u32` pairs into `(4, 2, 64, 4, 128) fp8`.
   - Page `4`: causal conv1d 3-tap window `(3, QKV) bf16` packed into
     `(2, 64, 4, 128) fp8`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .. import kernel
from ..blocks.gdn import GDNBlock
from ..ops import BF16, F32, FP8, I32, U32, rup
from ..spec import ExecSpec


# ---------------------------------------------------------------------------
# Per-page Host / Test Packing & Unpacking Utilities
# ---------------------------------------------------------------------------
def kv_to_raw_page(k_page: Any, v_page: Any) -> jax.Array:
    """Pack `(page_size, 256)` K and V into `(2, 64, 4, page_size) fp8`."""
    ps = k_page.shape[0]
    k_fp8 = (
        jnp.clip(k_page, -448.0, 448.0).astype(FP8) if k_page.dtype != FP8 else k_page
    )
    v_fp8 = (
        jnp.clip(v_page, -448.0, 448.0).astype(FP8) if v_page.dtype != FP8 else v_page
    )
    k_pk = jnp.transpose(k_fp8.reshape(ps, 64, 4), (1, 2, 0))
    v_pk = jnp.transpose(v_fp8.reshape(ps, 64, 4), (1, 2, 0))
    return jnp.stack([k_pk, v_pk], axis=0)


def raw_page_to_kv(raw_page: Any) -> tuple[jax.Array, jax.Array]:
    """Unpack `(2, 64, 4, page_size) fp8` into `(k_page, v_page)`."""
    ps = raw_page.shape[-1]
    k_page = jnp.transpose(raw_page[0], (2, 0, 1)).reshape(ps, 256)
    v_page = jnp.transpose(raw_page[1], (2, 0, 1)).reshape(ps, 256)
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


def _resolve_active_seqs(
    active_seqs: jax.Array | str | None,
    *,
    default_mode: str = "from_ckpt",
) -> tuple[str, jax.Array | None]:
    """Normalize `active_seqs` into `(act_mode, active_mask_array)`."""
    if active_seqs is None:
        return default_mode, None
    if isinstance(active_seqs, str):
        if active_seqs in ("from_ckpt", "all"):
            return active_seqs, None
        raise ValueError(
            f"Unsupported active_seqs mode {active_seqs!r}; "
            "expected 'from_ckpt', 'all', None, or an int32/bool mask array"
        )
    return "ref", jnp.asarray(active_seqs, dtype=I32)


def _check_ckpt_bounds(
    ckpt_indices: Any,
    num_pages: int,
    g_per_att: int,
    *,
    act_mode: str = "from_ckpt",
    active_mask: Any = None,
) -> None:
    """Validate concrete (eager) active GDN checkpoint slots fit in `[0, num_pages)`.

    Under `jax.jit` (`Tracer` inputs), this eager host check is a no-op and bounds
    are checked inside the Pallas kernel via `pl.debug_check` when Pallas debug
    checks are enabled (`jax_pallas_enable_debug_checks=True`).
    """
    if (
        ckpt_indices is None
        or g_per_att <= 0
        or isinstance(ckpt_indices, jax.core.Tracer)
    ):
        return
    ckpt_np = np.asarray(ckpt_indices)
    is_3d = ckpt_np.ndim == 3
    ns = ckpt_np.shape[1] if is_3d else ckpt_np.shape[0]
    mask_np = (
        None
        if (active_mask is None or isinstance(active_mask, jax.core.Tracer))
        else np.asarray(active_mask)
    )
    for s in range(ns):
        if act_mode == "from_ckpt":
            active = int(ckpt_np[0, s, 0] if is_3d else ckpt_np[s, 0]) > 0
        elif act_mode == "ref" and mask_np is not None:
            active = int(mask_np[s]) != 0
        else:
            active = True
        if not active:
            continue
        for sub in range(g_per_att):
            slots = ckpt_np[sub, s] if is_3d else ckpt_np[s]
            for slot in np.ravel(slots):
                pb = int(slot) * 5 if is_3d else int(slot) * (5 * g_per_att) + sub * 5
                if pb < 0 or pb + 5 > num_pages:
                    raise ValueError(
                        f"GDN checkpoint slot {int(slot)} (sub={sub}, s={s}) "
                        f"maps to pages [{pb}, {pb + 5}), which exceeds pool "
                        f"capacity [0, {num_pages})"
                    )


def _kernel_debug_check(cond: Any, msg: str, *, interpret: bool) -> None:
    """Emit `pl.debug_check` and enforce it in CPU `interpret` mode when enabled."""
    pl.debug_check(cond, msg)
    if interpret and pl.enable_debug_checks.value:

        def _cb(ok: Any) -> None:
            if not bool(ok):
                raise RuntimeError(msg)

        jax.debug.callback(_cb, cond)


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


# ---------------------------------------------------------------------------
# Pallas GDN Gather Kernel
# ---------------------------------------------------------------------------
def gather_gdn_states(
    g: Any,
    ex: ExecSpec,
    raw_pools: tuple[jax.Array, ...] | list[jax.Array],
    ckpt_indices: jax.Array,
    read_offsets: jax.Array,
    active_seqs: jax.Array | str | None = None,
    *,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Gather per-rank GDN `conv` and `rec` states from per-attention `raw_pools`.

    Args:
      g: Model geometry (`NATT`, `NGDN`, `att_every`, `H`, `K`, `KD`, `VD`, `V`).
      ex: Execution spec (`B`, `T`, `NS`, `R`, `page_size`, `num_pages`).
      raw_pools: Tuple of `g.NATT` per-rank raw pool arrays, each of shape
        `(num_pages, 2, 64, 4, page_size) fp8_e4m3fn`.
      ckpt_indices: `int32` checkpoint slot table, shape `(G_PER_ATT, NS, K_ckpt)`
        or `(NS, K_ckpt)`. For 3D `ckpt_indices`, `page_base = slot * 5`; for 2D,
        `page_base = slot * (5 * G_PER_ATT) + sub * 5`.
      read_offsets: `int32` index into `ckpt_indices` axis `-1` for the input
        checkpoint slot per sequence, shape `(G_PER_ATT, NS)` or `(NS,)`.
      active_seqs: Optional activity mask (`None` or `"from_ckpt"` to check
        `ckpt_indices[..., 0] > 0`, `"all"` for all sequences active, or
        `(NS,) int32` mask).
      interpret: Override `ex.interpret`.

    Returns:
      `(conv, rec)` matching `GDNBlock.states(g, ex)` per-rank shapes.
    """
    if hasattr(g, "check"):
        g.check(ex)
    if interpret is None:
        interpret = ex.interpret
    assert len(raw_pools) == g.NATT, (len(raw_pools), g.NATT)
    AE = g.att_every
    G_PER_ATT = AE - 1
    act_mode, active_mask = _resolve_active_seqs(
        active_seqs, default_mode="from_ckpt" if G_PER_ATT > 0 else "all"
    )
    _check_ckpt_bounds(
        ckpt_indices,
        ex.num_pages,
        G_PER_ATT,
        act_mode=act_mode,
        active_mask=active_mask,
    )

    lay = GDNBlock.lin_in_layout(g, ex)
    vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
    NS, B, T = ex.NS, ex.B, ex.T
    CR = rup(3 * B, 8)
    HR = rup(3 + T, 8)
    ngdn = max(1, g.NGDN)
    conv_shape = (ngdn, CR, QKV) if T == 1 else (ngdn, NS, HR, QKV)
    conv_dt = F32
    rec_shape = (ngdn, NS * vh, KD, VD)
    rec_dt = BF16

    page_size = ex.page_size
    ssm_u32_rows = (vh * KD) // 2
    conv_rows = (3 * QKV) // 128
    conv_rows_pad = ((conv_rows + 15) // 16) * 16
    qkv_128 = QKV // 128
    direct_conv = T > 1 and ((HR * qkv_128) % 16 == 0 or bool(interpret))
    hr_rows_128 = HR * qkv_128
    conv_raw_shape = conv_shape if direct_conv else (ngdn, NS * conv_rows_pad, 128)
    conv_raw_dt = conv_dt if direct_conv else BF16
    ckpt_is_3d = ckpt_indices.ndim == 3
    roff_is_2d = read_offsets.ndim == 2
    n_pairs = NS // 2

    coal_rec = (G_PER_ATT > 0) and (NS * vh <= 128)
    rec_coal = max(1, G_PER_ATT) if coal_rec else 1

    scratch = {
        "page_buf": pltpu.VMEM((2, 5, 2, 64, 4, page_size), FP8),
        "rec_buf": pltpu.VMEM((2, rec_coal, *rec_shape[1:]), rec_dt),
        "conv_buf": pltpu.VMEM(
            (2, max(1, G_PER_ATT), *conv_raw_shape[1:]), conv_raw_dt
        ),
        "sem_in": pltpu.SemaphoreType.DMA((2,)),
        "sem_out": pltpu.SemaphoreType.DMA((2, 2)),
    }
    s_names = list(scratch)
    n_smem = 3 if act_mode == "ref" else 2

    def _kernel(*refs):
        raw_refs = refs[: g.NATT]
        act_ref: Any = None
        if act_mode == "ref":
            ckpt_ref, roff_ref, act_ref = refs[g.NATT : g.NATT + 3]
        else:
            ckpt_ref, roff_ref = refs[g.NATT : g.NATT + 2]
        conv_out_ref, rec_out_ref = refs[g.NATT + n_smem : g.NATT + n_smem + 2]
        scr = dict(zip(s_names, refs[g.NATT + n_smem + 2 :]))
        page_buf = scr["page_buf"]
        rec_buf = scr["rec_buf"]
        conv_buf = scr["conv_buf"]
        sem_in = scr["sem_in"]
        sem_out = scr["sem_out"]
        conv_row_iota: Any = (
            lax.broadcasted_iota(I32, (HR, QKV), 0) if direct_conv else None
        )

        for slot in range(2):
            conv_buf[slot] = jnp.zeros(conv_buf.shape[1:], conv_raw_dt)
            rec_buf[slot] = jnp.zeros(rec_buf.shape[1:], rec_dt)

        def is_active(s):
            if act_mode == "all":
                return jnp.bool_(True)
            if act_mode == "from_ckpt":
                return (ckpt_ref[0, s, 0] > 0) if ckpt_is_3d else (ckpt_ref[s, 0] > 0)
            return act_ref[s] != 0

        def page_base_for(sub, s):
            roff = roff_ref[sub, s] if roff_is_2d else roff_ref[s]
            if ckpt_is_3d:
                slot = ckpt_ref[sub, s, roff]
                pb = slot * 5
            else:
                slot = ckpt_ref[s, roff]
                pb = slot * (5 * G_PER_ATT) + sub * 5
            _kernel_debug_check(
                jnp.logical_and(pb >= 0, pb + 5 <= ex.num_pages),
                "GDN gather checkpoint slot maps outside [0, num_pages)",
                interpret=bool(interpret),
            )
            return jnp.clip(pb, 0, ex.num_pages - 5)

        def start_page_read(raw_ref, sub, s, p_slot):
            @pl.when(is_active(s))
            def _():
                pb = page_base_for(sub, s)
                pltpu.make_async_copy(
                    raw_ref.at[pl.ds(pb, 5)], page_buf.at[p_slot], sem_in.at[p_slot]
                ).start()

        def finish_page_read(raw_ref, sub, s, p_slot, g_slot, r_slot, r_sub):
            @pl.when(is_active(s))
            def _():
                pb = page_base_for(sub, s)
                pltpu.make_async_copy(
                    raw_ref.at[pl.ds(pb, 5)], page_buf.at[p_slot], sem_in.at[p_slot]
                ).wait()

                u32_val = (
                    page_buf.at[p_slot, pl.ds(0, 4)]
                    .bitcast(U32)
                    .reshape(512, 128)[:ssm_u32_rows, :]
                )
                s_km = pltpu.bitcast(u32_val, BF16).reshape(vh, KD, VD)
                h0 = s * vh if isinstance(s, int) else pl.multiple_of(s * vh, vh)
                rec_buf[r_slot, r_sub, pl.ds(h0, vh)] = s_km

                if direct_conv:
                    raw_hr = (
                        page_buf.at[p_slot, 4]
                        .bitcast(BF16)
                        .reshape(256, 128)[:hr_rows_128, :]
                        .reshape(HR, QKV)
                        .astype(conv_dt)
                    )
                    conv_buf[g_slot, sub, s] = jnp.where(conv_row_iota < 3, raw_hr, 0.0)
                else:
                    c0 = (
                        s * conv_rows_pad
                        if isinstance(s, int)
                        else pl.multiple_of(s * conv_rows_pad, conv_rows_pad)
                    )
                    conv_buf[g_slot, sub, pl.ds(c0, conv_rows_pad), :] = (
                        page_buf.at[p_slot, 4]
                        .bitcast(BF16)
                        .reshape(256, 128)[:conv_rows_pad, :]
                    )

        total_gdn = g.NATT * G_PER_ATT
        if total_gdn > 0 and NS >= 1:
            start_page_read(raw_refs[0], 0, 0, 0)

        for g_idx in range(g.NATT):
            raw_ref = raw_refs[g_idx]
            g_slot = g_idx & 1
            if G_PER_ATT > 0 and g_idx >= 2:
                o_prev = (g_idx - 2) * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[g_slot],
                    conv_out_ref.at[pl.ds(o_prev, G_PER_ATT)],
                    sem_out.at[g_slot, 0],
                ).wait()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[g_slot],
                        rec_out_ref.at[pl.ds(o_prev, G_PER_ATT)],
                        sem_out.at[g_slot, 1],
                    ).wait()

            for sub in range(G_PER_ATT):
                o = g_idx * G_PER_ATT + sub
                r_slot = g_slot if coal_rec else (o & 1)
                r_sub = sub if coal_rec else 0
                if not coal_rec and o >= 2:
                    pltpu.make_async_copy(
                        rec_buf.at[r_slot, 0],
                        rec_out_ref.at[o - 2],
                        sem_out.at[r_slot, 1],
                    ).wait()

                next_o = o + 1
                has_next_o = next_o < total_gdn
                next_raw_ref = raw_refs[next_o // G_PER_ATT] if has_next_o else None
                next_sub = (next_o % G_PER_ATT) if has_next_o else 0

                if NS == 1:
                    p_cur = o & 1
                    p_nxt = (o + 1) & 1
                    if has_next_o:
                        start_page_read(next_raw_ref, next_sub, 0, p_nxt)
                    finish_page_read(raw_ref, sub, 0, p_cur, g_slot, r_slot, r_sub)
                elif NS % 2 == 0:
                    if n_pairs == 1:
                        start_page_read(raw_ref, sub, 1, 1)
                        finish_page_read(raw_ref, sub, 0, 0, g_slot, r_slot, r_sub)
                        if has_next_o:
                            start_page_read(next_raw_ref, next_sub, 0, 0)
                        finish_page_read(raw_ref, sub, 1, 1, g_slot, r_slot, r_sub)
                    else:

                        def gather_pair_even(
                            pair,
                            _,
                            raw_ref=raw_ref,
                            sub=sub,
                            g_slot=g_slot,
                            r_slot=r_slot,
                            r_sub=r_sub,
                            has_next_o=has_next_o,
                            next_raw_ref=next_raw_ref,
                            next_sub=next_sub,
                        ):
                            s0 = 2 * pair
                            s1 = s0 + 1
                            start_page_read(raw_ref, sub, s1, 1)
                            finish_page_read(raw_ref, sub, s0, 0, g_slot, r_slot, r_sub)

                            @pl.when(s0 + 2 < NS)
                            def _():
                                start_page_read(raw_ref, sub, s0 + 2, 0)

                            if has_next_o:

                                @pl.when(s0 + 2 == NS)
                                def _():
                                    start_page_read(next_raw_ref, next_sub, 0, 0)

                            finish_page_read(raw_ref, sub, s1, 1, g_slot, r_slot, r_sub)
                            return 0

                        lax.fori_loop(0, n_pairs, gather_pair_even, 0)
                else:
                    p0 = o & 1
                    p1 = 1 - p0
                    if n_pairs == 1:
                        start_page_read(raw_ref, sub, 1, p1)
                        finish_page_read(raw_ref, sub, 0, p0, g_slot, r_slot, r_sub)
                        start_page_read(raw_ref, sub, 2, p0)
                        finish_page_read(raw_ref, sub, 1, p1, g_slot, r_slot, r_sub)
                    else:

                        def gather_pair_odd(
                            pair,
                            _,
                            raw_ref=raw_ref,
                            sub=sub,
                            g_slot=g_slot,
                            r_slot=r_slot,
                            r_sub=r_sub,
                            p0=p0,
                            p1=p1,
                        ):
                            s0 = 2 * pair
                            s1 = s0 + 1
                            start_page_read(raw_ref, sub, s1, p1)
                            finish_page_read(
                                raw_ref, sub, s0, p0, g_slot, r_slot, r_sub
                            )
                            start_page_read(raw_ref, sub, s0 + 2, p0)
                            finish_page_read(
                                raw_ref, sub, s1, p1, g_slot, r_slot, r_sub
                            )
                            return 0

                        lax.fori_loop(0, n_pairs, gather_pair_odd, 0)

                    if has_next_o:
                        start_page_read(next_raw_ref, next_sub, 0, p1)
                    finish_page_read(raw_ref, sub, NS - 1, p0, g_slot, r_slot, r_sub)

                if not coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[r_slot, 0],
                        rec_out_ref.at[o],
                        sem_out.at[r_slot, 1],
                    ).start()

            if G_PER_ATT > 0:
                o_base = g_idx * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[g_slot],
                    conv_out_ref.at[pl.ds(o_base, G_PER_ATT)],
                    sem_out.at[g_slot, 0],
                ).start()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[g_slot],
                        rec_out_ref.at[pl.ds(o_base, G_PER_ATT)],
                        sem_out.at[g_slot, 1],
                    ).start()

        if G_PER_ATT > 0:
            for g_tail in range(max(0, g.NATT - 2), g.NATT):
                tail_slot = g_tail & 1
                o_tail = g_tail * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[tail_slot],
                    conv_out_ref.at[pl.ds(o_tail, G_PER_ATT)],
                    sem_out.at[tail_slot, 0],
                ).wait()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[tail_slot],
                        rec_out_ref.at[pl.ds(o_tail, G_PER_ATT)],
                        sem_out.at[tail_slot, 1],
                    ).wait()
            if not coal_rec:
                for o_tail in range(max(0, total_gdn - 2), total_gdn):
                    r_tail = o_tail & 1
                    pltpu.make_async_copy(
                        rec_buf.at[r_tail, 0],
                        rec_out_ref.at[o_tail],
                        sem_out.at[r_tail, 1],
                    ).wait()
        else:
            cp_c = pltpu.make_async_copy(
                conv_buf.at[0, 0], conv_out_ref.at[0], sem_out.at[0, 0]
            )
            cp_r = pltpu.make_async_copy(
                rec_buf.at[0, 0], rec_out_ref.at[0], sem_out.at[0, 1]
            )
            cp_c.start()
            cp_r.start()
            cp_c.wait()
            cp_r.wait()

    in_specs = [pl.BlockSpec(memory_space=pltpu.HBM)] * g.NATT + [
        pl.BlockSpec(memory_space=pltpu.SMEM)
    ] * n_smem
    out_shape = (
        [
            jax.ShapeDtypeStruct(conv_raw_shape, conv_raw_dt),
            jax.ShapeDtypeStruct(rec_shape, rec_dt),
        ]
        if interpret
        else [
            pltpu.HBM(conv_raw_shape, conv_raw_dt),
            pltpu.HBM(rec_shape, rec_dt),
        ]
    )
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]
    call = pl.pallas_call(
        _kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(
            collective_id=None,
            disable_bounds_checks=True,
            vmem_limit_bytes=ex.resolved_tiles.vmem_mib << 20,
        ),
        interpret=bool(interpret),
        name="qwen3_5_gather_gdn_states",
    )
    smem_args = (
        (ckpt_indices, read_offsets, active_mask)
        if act_mode == "ref"
        else (ckpt_indices, read_offsets)
    )
    conv_raw, rec_out = call(*raw_pools, *smem_args)
    if direct_conv:
        return conv_raw, rec_out
    taps_all = (
        conv_raw.reshape(ngdn, NS, conv_rows_pad, 128)[:, :, :conv_rows, :]
        .reshape(ngdn, NS, 3, QKV)
        .astype(conv_dt)
    )
    if T > 1:
        conv_out = jnp.pad(taps_all, ((0, 0), (0, 0), (0, HR - 3), (0, 0)))
    else:
        conv_3b = jnp.swapaxes(taps_all, 1, 2).reshape(ngdn, 3 * B, QKV)
        conv_out = jnp.pad(conv_3b, ((0, 0), (0, CR - 3 * B), (0, 0)))
    return conv_out, rec_out


__all__ = [
    "conv_to_raw_page",
    "default_pool_layout",
    "gather_gdn_states",
    "kv_to_raw_page",
    "pool_to_states",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "ssm_to_raw_pages",
    "states_to_pool",
]
