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
"""Pallas pool scatter kernel: new KV + GDN checkpoints into raw pools."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.gdn import GDNBlock
from ..ops import BF16, FP8, I32, U32, rup
from ..spec import ExecSpec
from .backend import _check_ckpt_bounds, _kernel_debug_check, _resolve_active_seqs


# ---------------------------------------------------------------------------
# Pallas Pool Scatter Kernel (KV + GDN Checkpoint States)
# ---------------------------------------------------------------------------
def scatter_pool_states(
    g: Any,
    ex: ExecSpec,
    raw_pools: tuple[jax.Array, ...] | list[jax.Array],
    pos: jax.Array,
    page_table: jax.Array,
    knew: jax.Array,
    vnew: jax.Array,
    conv: jax.Array | None = None,
    rec: jax.Array | None = None,
    ckpt_indices: jax.Array | None = None,
    active_seqs: jax.Array | str | None = None,
    *,
    interpret: bool | None = None,
    conv_partial: bool | None = None,
) -> tuple[jax.Array, ...]:
    """Scatter `knew`/`vnew` and GDN `conv`/`rec` checkpoints into `raw_pools` in-place.

    Only the per-row KV page indices `kv_pages` of shape `(NS, T)` are passed into
    SMEM (rather than the full `(NS, max_pages_per_seq)` `page_table`), and all
    page/checkpoint indices are clamped to `[0, num_pages)` before DMA.

    Note on `interpret=True` vs TPU `Ref.bitcast` stores:
      On TPU (`interpret=False`), sub-slice writes into `page_buf` and `kv_page`
      use zero-copy `Ref.bitcast(U32/BF16/I32).reshape(...)[...] = ...`, which
      packs the 2nd-to-last sublane axis (`4` for `fp8->u32/i32`, `2` for
      `fp8->bf16`) into 32-bit/16-bit words in VMEM. Because JAX's CPU Pallas
      `state_discharge` does not invert `Ref.bitcast(...).reshape(...)` on LHS
      indexed stores, `interpret=True` uses `lax.bitcast_convert_type(..., FP8)`
      followed by `swapaxes` to move the unpacked byte axis back to the sublane
      axis, producing the exact same byte layout in `page_buf`/`kv_page`.
    """
    if hasattr(g, "check"):
        g.check(ex)
    if interpret is None:
        interpret = ex.interpret
    assert len(raw_pools) == g.NATT, (len(raw_pools), g.NATT)

    NS, B, T, MR = ex.NS, ex.B, ex.T, ex.MR
    AE = g.att_every
    G_PER_ATT = AE - 1
    page_size = ex.page_size
    h4 = g.HD // 4
    assert page_size >= MR, f"MR={MR} exceeds page_size={page_size}"
    if page_table.ndim != 2 or page_table.shape[0] != NS or page_table.shape[1] < 1:
        raise ValueError(
            f"page_table has shape {page_table.shape}, expected ({NS}, >=1)"
        )
    has_gdn = G_PER_ATT > 0
    use_conv_partial = (
        bool(ex.GPR if conv_partial is None else conv_partial)
        and has_gdn
        and not bool(interpret)
    )
    act_mode, active_mask = _resolve_active_seqs(
        active_seqs, default_mode="from_ckpt" if has_gdn else "all"
    )
    if has_gdn:
        _check_ckpt_bounds(
            ckpt_indices,
            ex.num_pages,
            G_PER_ATT,
            act_mode=act_mode,
            active_mask=active_mask,
        )

    # Pre-gather and clamp only the (NS, T) active KV page indices for SMEM
    pt_cols = page_table.shape[1]
    pos_st = pos[: NS * T].reshape(NS, T)
    lp_idx = jnp.clip(lax.div(pos_st, page_size), 0, pt_cols - 1)
    s_idx_arr = jnp.arange(NS, dtype=I32)[:, None]
    kv_pages = jnp.clip(page_table[s_idx_arr, lp_idx], 0, ex.num_pages - 1).astype(I32)

    conv_in: Any = None
    conv_buf_shape: Any = None
    if has_gdn:
        assert conv is not None and rec is not None and ckpt_indices is not None
        lay = GDNBlock.lin_in_layout(g, ex)
        vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
        HR = rup(3 + T, 8)
        ssm_u32_rows = (vh * KD) // 2
        conv_rows = (3 * QKV) // 128
        conv_rows_pad = ((conv_rows + 15) // 16) * 16
        qkv_128 = QKV // 128
        assert T == 1 or (4 * qkv_128) % 16 != 0 or 4 + T <= HR, (
            f"T={T} exceeds direct_conv window HR={HR}"
        )
        min_direct_rows = 4 + T if (4 * qkv_128) % 16 == 0 else 3 + T
        direct_conv = (
            T > 1
            and conv.ndim == 4
            and conv.shape[2] >= min_direct_rows
            and ((conv.shape[2] * qkv_128) % 16 == 0 or bool(interpret))
        )
        if direct_conv:
            conv_in = conv
            conv_buf_shape = (2, 1, conv.shape[2], QKV)
            conv_buf_dt = conv.dtype
        else:
            if conv.ndim == 3:
                conv_4d = jnp.swapaxes(
                    conv[:, : 3 * B, :].reshape(conv.shape[0], 3, B, QKV), 1, 2
                )
            else:
                conv_4d = conv
            conv_has_hist = conv_4d.shape[2] >= 3 + T
            if conv_has_hist:
                conv_win = jnp.stack(
                    [conv_4d[:, :, 1 + t : 4 + t, :] for t in range(T)], axis=2
                )
            else:
                conv_win = jnp.broadcast_to(
                    conv_4d[:, :, None, 0:3, :],
                    (conv_4d.shape[0], NS, T, 3, QKV),
                )
            conv_flat = conv_win.astype(BF16).reshape(
                conv_4d.shape[0], NS * T, conv_rows, 128
            )
            if conv_rows_pad > conv_rows:
                conv_flat = jnp.pad(
                    conv_flat,
                    ((0, 0), (0, 0), (0, conv_rows_pad - conv_rows), (0, 0)),
                )
            conv_in = conv_flat.reshape(conv_4d.shape[0], NS * T * conv_rows_pad, 128)
            conv_buf_shape = (2, T * conv_rows_pad, 128)
            conv_buf_dt = BF16
        ckpt_is_3d = ckpt_indices.ndim == 3
        ckpt_k = ckpt_indices.shape[-1]
        ckpt_t_off = 0 if ckpt_k == T else 1
        assert ckpt_t_off + T <= ckpt_k, (ckpt_k, T)
    else:
        vh = KD = VD = QKV = 0
        ssm_u32_rows = conv_rows = conv_rows_pad = qkv_128 = 0
        direct_conv = False
        conv_buf_dt = BF16
        ckpt_is_3d = False
        ckpt_t_off = 0

    n_pairs = NS // 2
    scratch: dict[str, Any] = {
        "kv_page": pltpu.VMEM((2, 2, h4, 4, page_size), FP8),
        "knew_buf": pltpu.VMEM((g.NATT, MR, g.HD), FP8),
        "vnew_buf": pltpu.VMEM((g.NATT, MR, g.HD), FP8),
        "sem_kv_rd": pltpu.SemaphoreType.DMA((2,)),
        "sem_kv_wr": pltpu.SemaphoreType.DMA((2,)),
        "sem_kv_in": pltpu.SemaphoreType.DMA((2,)),
    }
    if has_gdn:
        scratch.update(
            {
                "page_buf": pltpu.VMEM((2, 5 * T, 2, h4, 4, page_size), FP8),
                "rec_buf": pltpu.VMEM((2, T * vh, KD, VD), BF16),
                "conv_buf": pltpu.VMEM(conv_buf_shape, conv_buf_dt),
                "sem_in": pltpu.SemaphoreType.DMA((2, 2)),
                "sem_p": pltpu.SemaphoreType.DMA((2, T)),
            }
        )
    if NS > 1:
        scratch["k_cols_buf"] = pltpu.VMEM((g.NATT, h4, page_size), I32)
        scratch["v_cols_buf"] = pltpu.VMEM((g.NATT, h4, page_size), I32)

    s_names = list(scratch)
    n_smem = 2 + (1 if has_gdn else 0) + (1 if act_mode == "ref" else 0)
    n_hbm = 4 if has_gdn else 2

    def batch_to_i32_cols(tok_mr_hd):
        rows_bf16 = tok_mr_hd.astype(BF16)
        if page_size > MR:
            rows_bf16 = jnp.pad(rows_bf16, ((0, page_size - MR), (0, 0)))
        cols_fp8 = jnp.swapaxes(rows_bf16, 0, 1).astype(FP8)
        return pltpu.bitcast(cols_fp8, I32)

    def _kernel(*refs: Any):
        smem_refs = refs[g.NATT : g.NATT + n_smem]
        pos_ref, kv_pages_ref = smem_refs[0], smem_refs[1]
        ckpt_ref: Any = smem_refs[2] if has_gdn else None
        act_ref: Any = smem_refs[-1] if act_mode == "ref" else None
        hbm_refs: tuple[Any, ...] = (
            refs[g.NATT + n_smem : g.NATT + n_smem + n_hbm] + (None, None)
        )[:4]
        knew_ref, vnew_ref, conv_ref, rec_ref = hbm_refs
        raw_out_refs = refs[g.NATT + n_smem + n_hbm : 2 * g.NATT + n_smem + n_hbm]
        scr = dict(zip(s_names, refs[2 * g.NATT + n_smem + n_hbm :]))
        kv_page = scr["kv_page"]
        knew_buf = scr["knew_buf"]
        vnew_buf = scr["vnew_buf"]
        page_buf: Any = scr.get("page_buf")
        rec_buf: Any = scr.get("rec_buf")
        conv_buf: Any = scr.get("conv_buf")
        sem_kv_rd = scr["sem_kv_rd"]
        sem_kv_wr = scr["sem_kv_wr"]
        sem_kv_in = scr["sem_kv_in"]
        sem_in: Any = scr.get("sem_in")
        sem_p: Any = scr.get("sem_p")
        k_cols_buf: Any = scr["k_cols_buf"] if NS > 1 else None
        v_cols_buf: Any = scr["v_cols_buf"] if NS > 1 else None
        lane_iota = lax.broadcasted_iota(I32, (h4, page_size), 1)
        w4_row_iota: Any = (
            lax.broadcasted_iota(I32, (4, QKV), 0) if direct_conv else None
        )
        total_gdn = g.NATT * G_PER_ATT

        def is_active(s):
            if act_mode == "all":
                return jnp.bool_(True)
            if act_mode == "from_ckpt":
                return (ckpt_ref[0, s, 0] > 0) if ckpt_is_3d else (ckpt_ref[s, 0] > 0)
            return act_ref[s] != 0

        def pb_for(sub, s, t, span: int = 5):
            if ckpt_is_3d:
                slot = ckpt_ref[sub, s, t + ckpt_t_off]
                pb = slot * 5
            else:
                slot = ckpt_ref[s, t + ckpt_t_off]
                pb = slot * (5 * G_PER_ATT) + sub * 5
            _kernel_debug_check(
                jnp.logical_and(pb >= 0, pb + span <= ex.num_pages),
                "GDN scatter checkpoint slot maps outside [0, num_pages)",
                interpret=bool(interpret),
            )
            return jnp.clip(pb, 0, ex.num_pages - span)

        def is_contig_ckpt(sub, s):
            if not ckpt_is_3d or T <= 1:
                return jnp.bool_(T == 1)
            s0 = ckpt_ref[sub, s, ckpt_t_off]
            ok = jnp.bool_(True)
            for t in range(1, T):
                ok = jnp.logical_and(ok, ckpt_ref[sub, s, t + ckpt_t_off] == s0 + t)
            return ok

        def start_seq_in(o, s, in_slot):
            @pl.when(is_active(s))
            def _():
                r_base = (
                    s * T * vh
                    if isinstance(s, int)
                    else pl.multiple_of(s * T * vh, T * vh)
                )
                if direct_conv:
                    pltpu.make_async_copy(
                        conv_ref.at[o, pl.ds(s, 1)],
                        conv_buf.at[in_slot],
                        sem_in.at[in_slot, 0],
                    ).start()
                else:
                    c_base = (
                        s * T * conv_rows_pad
                        if isinstance(s, int)
                        else pl.multiple_of(s * T * conv_rows_pad, T * conv_rows_pad)
                    )
                    pltpu.make_async_copy(
                        conv_ref.at[o, pl.ds(c_base, T * conv_rows_pad)],
                        conv_buf.at[in_slot],
                        sem_in.at[in_slot, 0],
                    ).start()
                pltpu.make_async_copy(
                    rec_ref.at[o, pl.ds(r_base, T * vh)],
                    rec_buf.at[in_slot],
                    sem_in.at[in_slot, 1],
                ).start()

        def wait_seq_in(o, s, in_slot):
            @pl.when(is_active(s))
            def _():
                r_base = (
                    s * T * vh
                    if isinstance(s, int)
                    else pl.multiple_of(s * T * vh, T * vh)
                )
                if direct_conv:
                    pltpu.make_async_copy(
                        conv_ref.at[o, pl.ds(s, 1)],
                        conv_buf.at[in_slot],
                        sem_in.at[in_slot, 0],
                    ).wait()
                else:
                    c_base = (
                        s * T * conv_rows_pad
                        if isinstance(s, int)
                        else pl.multiple_of(s * T * conv_rows_pad, T * conv_rows_pad)
                    )
                    pltpu.make_async_copy(
                        conv_ref.at[o, pl.ds(c_base, T * conv_rows_pad)],
                        conv_buf.at[in_slot],
                        sem_in.at[in_slot, 0],
                    ).wait()
                pltpu.make_async_copy(
                    rec_ref.at[o, pl.ds(r_base, T * vh)],
                    rec_buf.at[in_slot],
                    sem_in.at[in_slot, 1],
                ).wait()

        CP_ROWS = conv_rows_pad // 2
        SLOT_ROWS = 4 * 128 + CP_ROWS

        def slot_copy(raw_out_ref, slot, t, pb_t):
            src = page_buf.at[slot].reshape(T * 5 * 128, 4, page_size)
            dst = raw_out_ref.reshape(ex.num_pages * 128, 4, page_size)
            r0 = pl.multiple_of(pb_t * 128, 128)
            return pltpu.make_async_copy(
                src.at[pl.ds(t * 5 * 128, SLOT_ROWS)],
                dst.at[pl.ds(r0, SLOT_ROWS)],
                sem_p.at[slot, t],
            )

        def start_scatter_seq(raw_out_ref, sub, s, slot):
            @pl.when(is_active(s))
            def _():
                conv_bf16: Any = (
                    conv_buf[slot, 0].astype(BF16)
                    if (direct_conv and (4 * qkv_128) % 16 == 0)
                    else None
                )
                for t in range(T):
                    s_km = rec_buf[slot, pl.ds(t * vh, vh)]
                    u32_val = pltpu.bitcast(s_km.reshape(vh * KD, VD), U32)
                    if bool(interpret):
                        u32_pad = (
                            jnp.pad(u32_val, ((0, 512 - ssm_u32_rows), (0, 0)))
                            if ssm_u32_rows < 512
                            else u32_val
                        )
                        page_buf[slot, pl.ds(t * 5, 4)] = (
                            lax.bitcast_convert_type(u32_pad, FP8)
                            .swapaxes(1, 2)
                            .reshape(4, 2, h4, 4, 128)
                        )
                    else:
                        page_buf.at[slot, pl.ds(t * 5, 4)].bitcast(U32).reshape(
                            512, 128
                        )[:ssm_u32_rows, :] = u32_val

                    if direct_conv:
                        if (4 * qkv_128) % 16 == 0:
                            w4 = jnp.where(
                                w4_row_iota < 3,
                                conv_bf16[1 + t : 5 + t, :],
                                jnp.bfloat16(0.0),
                            )
                            conv_taps = w4.reshape(conv_rows_pad, 128)
                        else:
                            taps_128 = (
                                conv_buf[slot, 0, 1 + t : 4 + t, :]
                                .astype(BF16)
                                .reshape(conv_rows, 128)
                            )
                            if conv_rows_pad > conv_rows:
                                taps_128 = jnp.pad(
                                    taps_128,
                                    ((0, conv_rows_pad - conv_rows), (0, 0)),
                                )
                            conv_taps = taps_128
                    else:
                        conv_taps = conv_buf[
                            slot, pl.ds(t * conv_rows_pad, conv_rows_pad), :
                        ]
                    if bool(interpret):
                        conv_pad = (
                            jnp.pad(conv_taps, ((0, 256 - conv_rows_pad), (0, 0)))
                            if conv_rows_pad < 256
                            else conv_taps
                        )
                        page_buf[slot, t * 5 + 4] = (
                            lax.bitcast_convert_type(
                                conv_pad.reshape(2, h4, 2, 128), FP8
                            )
                            .swapaxes(3, 4)
                            .reshape(2, h4, 4, 128)
                        )
                    else:
                        page_buf.at[slot, t * 5 + 4].bitcast(BF16).reshape(256, 128)[
                            :conv_rows_pad, :
                        ] = conv_taps

                if use_conv_partial:
                    for t in range(T):
                        slot_copy(
                            raw_out_ref, slot, t, pb_for(sub, s, t, span=5)
                        ).start()
                elif T == 1:
                    pb0 = pb_for(sub, s, 0, span=5)
                    pltpu.make_async_copy(
                        page_buf.at[slot],
                        raw_out_ref.at[pl.ds(pb0, 5)],
                        sem_p.at[slot, 0],
                    ).start()
                else:
                    contig = is_contig_ckpt(sub, s)

                    @pl.when(contig)
                    def _():
                        pb0 = pb_for(sub, s, 0, span=5 * T)
                        pltpu.make_async_copy(
                            page_buf.at[slot],
                            raw_out_ref.at[pl.ds(pb0, 5 * T)],
                            sem_p.at[slot, 0],
                        ).start()

                    @pl.when(jnp.logical_not(contig))
                    def _():
                        for t in range(T):
                            pb_t = pb_for(sub, s, t, span=5)
                            pltpu.make_async_copy(
                                page_buf.at[slot, pl.ds(t * 5, 5)],
                                raw_out_ref.at[pl.ds(pb_t, 5)],
                                sem_p.at[slot, t],
                            ).start()

        def wait_scatter_seq(raw_out_ref, sub, s, slot):
            @pl.when(is_active(s))
            def _():
                if use_conv_partial:
                    for t in range(T):
                        slot_copy(
                            raw_out_ref, slot, t, pb_for(sub, s, t, span=5)
                        ).wait()
                elif T == 1:
                    pb0 = pb_for(sub, s, 0, span=5)
                    pltpu.make_async_copy(
                        page_buf.at[slot],
                        raw_out_ref.at[pl.ds(pb0, 5)],
                        sem_p.at[slot, 0],
                    ).wait()
                else:
                    contig = is_contig_ckpt(sub, s)

                    @pl.when(contig)
                    def _():
                        pb0 = pb_for(sub, s, 0, span=5 * T)
                        pltpu.make_async_copy(
                            page_buf.at[slot],
                            raw_out_ref.at[pl.ds(pb0, 5 * T)],
                            sem_p.at[slot, 0],
                        ).wait()

                    @pl.when(jnp.logical_not(contig))
                    def _():
                        for t in range(T):
                            pb_t = pb_for(sub, s, t, span=5)
                            pltpu.make_async_copy(
                                page_buf.at[slot, pl.ds(t * 5, 5)],
                                raw_out_ref.at[pl.ds(pb_t, 5)],
                                sem_p.at[slot, t],
                            ).wait()

        cp_knew = pltpu.make_async_copy(knew_ref, knew_buf, sem_kv_in.at[0])
        cp_vnew = pltpu.make_async_copy(vnew_ref, vnew_buf, sem_kv_in.at[1])
        cp_knew.start()
        cp_vnew.start()

        if total_gdn > 0 and NS >= 1:
            start_seq_in(0, 0, 0)
            if NS >= 2:
                start_seq_in(0, 1, 1)
            elif total_gdn >= 2:
                start_seq_in(1, 0, 1)

        if G_PER_ATT > 0:
            for slot in range(2):
                if bool(interpret):
                    page_buf[slot] = jnp.zeros_like(page_buf[slot])
                else:
                    if ssm_u32_rows < 512:
                        for t in range(T):
                            page_buf.at[slot, pl.ds(t * 5, 4)].bitcast(U32).reshape(
                                512, 128
                            )[...] = jnp.zeros((512, 128), U32)
                    if not use_conv_partial:
                        for t in range(T):
                            page_buf.at[slot, t * 5 + 4].bitcast(BF16).reshape(
                                256, 128
                            )[...] = jnp.zeros((256, 128), BF16)

        cp_knew.wait()
        cp_vnew.wait()

        if NS > 1:
            for g_idx in range(g.NATT):
                k_cols_buf[g_idx] = batch_to_i32_cols(knew_buf[g_idx])
                v_cols_buf[g_idx] = batch_to_i32_cols(vnew_buf[g_idx])

        def scatter_single_token_across_att(s_idx, b_idx, t_idx):
            p = pos_ref[b_idx]
            ln = lax.rem(p, page_size)
            kp = jnp.clip(kv_pages_ref[s_idx, t_idx], 0, ex.num_pages - 1)
            mask_ln = lane_iota == ln
            sh_b = lax.rem(ln + (page_size - lax.rem(b_idx, page_size)), page_size)

            pltpu.make_async_copy(
                raw_out_refs[0].at[kp], kv_page.at[0], sem_kv_rd.at[0]
            ).start()

            for g_idx in range(g.NATT):
                k_slot = g_idx & 1
                nxt_slot = 1 - k_slot
                if g_idx >= 1:
                    pltpu.make_async_copy(
                        kv_page.at[nxt_slot],
                        raw_out_refs[g_idx - 1].at[kp],
                        sem_kv_wr.at[nxt_slot],
                    ).wait()
                if g_idx + 1 < g.NATT:
                    pltpu.make_async_copy(
                        raw_out_refs[g_idx + 1].at[kp],
                        kv_page.at[nxt_slot],
                        sem_kv_rd.at[nxt_slot],
                    ).start()
                pltpu.make_async_copy(
                    raw_out_refs[g_idx].at[kp],
                    kv_page.at[k_slot],
                    sem_kv_rd.at[k_slot],
                ).wait()

                k_cols = (
                    k_cols_buf[g_idx] if NS > 1 else batch_to_i32_cols(knew_buf[g_idx])
                )
                v_cols = (
                    v_cols_buf[g_idx] if NS > 1 else batch_to_i32_cols(vnew_buf[g_idx])
                )
                k_col = pltpu.roll(k_cols, sh_b, 1)
                v_col = pltpu.roll(v_cols, sh_b, 1)
                k_view = kv_page.at[k_slot, 0].bitcast(I32).reshape(h4, page_size)
                v_view = kv_page.at[k_slot, 1].bitcast(I32).reshape(h4, page_size)
                k_new = jnp.where(mask_ln, k_col, k_view[...])
                v_new = jnp.where(mask_ln, v_col, v_view[...])
                if bool(interpret):
                    kv_page[k_slot, 0] = lax.bitcast_convert_type(k_new, FP8).swapaxes(
                        1, 2
                    )
                    kv_page[k_slot, 1] = lax.bitcast_convert_type(v_new, FP8).swapaxes(
                        1, 2
                    )
                else:
                    k_view[...] = k_new
                    v_view[...] = v_new

                pltpu.make_async_copy(
                    kv_page.at[k_slot],
                    raw_out_refs[g_idx].at[kp],
                    sem_kv_wr.at[k_slot],
                ).start()

            if g.NATT >= 1:
                last_slot = (g.NATT - 1) & 1
                pltpu.make_async_copy(
                    kv_page.at[last_slot],
                    raw_out_refs[g.NATT - 1].at[kp],
                    sem_kv_wr.at[last_slot],
                ).wait()

        if T == 1:

            def scatter_kv_row(b, _):
                @pl.when(is_active(b))
                def _():
                    scatter_single_token_across_att(b, b, 0)

                return 0

            if B == 1:
                scatter_kv_row(0, 0)
            else:
                lax.fori_loop(0, B, scatter_kv_row, 0)
        else:

            def scatter_kv_seq(s, _):
                @pl.when(is_active(s))
                def _():
                    b0 = s * T
                    kp0 = jnp.clip(kv_pages_ref[s, 0], 0, ex.num_pages - 1)
                    same_page = jnp.bool_(True)
                    for t in range(1, T):
                        same_page = jnp.logical_and(
                            same_page,
                            jnp.clip(kv_pages_ref[s, t], 0, ex.num_pages - 1) == kp0,
                        )

                    @pl.when(same_page)
                    def _():
                        kp = kp0
                        ln_list = [
                            lax.rem(pos_ref[b0 + t], page_size) for t in range(T)
                        ]
                        mask_ln_list = [lane_iota == ln_list[t] for t in range(T)]
                        sh_list = [
                            lax.rem(
                                ln_list[t] + (page_size - lax.rem(b0 + t, page_size)),
                                page_size,
                            )
                            for t in range(T)
                        ]

                        pltpu.make_async_copy(
                            raw_out_refs[0].at[kp], kv_page.at[0], sem_kv_rd.at[0]
                        ).start()

                        for g_idx in range(g.NATT):
                            k_slot = g_idx & 1
                            nxt_slot = 1 - k_slot
                            if g_idx >= 1:
                                pltpu.make_async_copy(
                                    kv_page.at[nxt_slot],
                                    raw_out_refs[g_idx - 1].at[kp],
                                    sem_kv_wr.at[nxt_slot],
                                ).wait()
                            if g_idx + 1 < g.NATT:
                                pltpu.make_async_copy(
                                    raw_out_refs[g_idx + 1].at[kp],
                                    kv_page.at[nxt_slot],
                                    sem_kv_rd.at[nxt_slot],
                                ).start()
                            pltpu.make_async_copy(
                                raw_out_refs[g_idx].at[kp],
                                kv_page.at[k_slot],
                                sem_kv_rd.at[k_slot],
                            ).wait()

                            k_cols = (
                                k_cols_buf[g_idx]
                                if NS > 1
                                else batch_to_i32_cols(knew_buf[g_idx])
                            )
                            v_cols = (
                                v_cols_buf[g_idx]
                                if NS > 1
                                else batch_to_i32_cols(vnew_buf[g_idx])
                            )
                            k_view = (
                                kv_page.at[k_slot, 0]
                                .bitcast(I32)
                                .reshape(h4, page_size)
                            )
                            v_view = (
                                kv_page.at[k_slot, 1]
                                .bitcast(I32)
                                .reshape(h4, page_size)
                            )
                            k_cur = k_view[...]
                            v_cur = v_view[...]
                            for t in range(T):
                                k_col = pltpu.roll(k_cols, sh_list[t], 1)
                                v_col = pltpu.roll(v_cols, sh_list[t], 1)
                                k_cur = jnp.where(mask_ln_list[t], k_col, k_cur)
                                v_cur = jnp.where(mask_ln_list[t], v_col, v_cur)
                            if bool(interpret):
                                kv_page[k_slot, 0] = lax.bitcast_convert_type(
                                    k_cur, FP8
                                ).swapaxes(1, 2)
                                kv_page[k_slot, 1] = lax.bitcast_convert_type(
                                    v_cur, FP8
                                ).swapaxes(1, 2)
                            else:
                                k_view[...] = k_cur
                                v_view[...] = v_cur

                            pltpu.make_async_copy(
                                kv_page.at[k_slot],
                                raw_out_refs[g_idx].at[kp],
                                sem_kv_wr.at[k_slot],
                            ).start()

                        if g.NATT >= 1:
                            last_slot = (g.NATT - 1) & 1
                            pltpu.make_async_copy(
                                kv_page.at[last_slot],
                                raw_out_refs[g.NATT - 1].at[kp],
                                sem_kv_wr.at[last_slot],
                            ).wait()

                    @pl.when(jnp.logical_not(same_page))
                    def _():
                        for t in range(T):
                            scatter_single_token_across_att(s, b0 + t, t)

                return 0

            if NS == 1:
                scatter_kv_seq(0, 0)
            else:
                lax.fori_loop(0, NS, scatter_kv_seq, 0)

        if total_gdn > 0:
            if NS == 1:
                for g_idx in range(g.NATT):
                    raw_out_ref = raw_out_refs[g_idx]
                    for sub in range(G_PER_ATT):
                        o = g_idx * G_PER_ATT + sub
                        slot = o & 1
                        if o >= 2:
                            prev_o = o - 2
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                0,
                                slot,
                            )
                        wait_seq_in(o, 0, slot)
                        start_scatter_seq(raw_out_ref, sub, 0, slot)
                        if o + 2 < total_gdn:
                            start_seq_in(o + 2, 0, slot)

                for o_tail in range(max(0, total_gdn - 2), total_gdn):
                    wait_scatter_seq(
                        raw_out_refs[o_tail // G_PER_ATT],
                        o_tail % G_PER_ATT,
                        0,
                        o_tail & 1,
                    )
            elif NS == 2:
                for g_idx in range(g.NATT):
                    raw_out_ref = raw_out_refs[g_idx]
                    for sub in range(G_PER_ATT):
                        o = g_idx * G_PER_ATT + sub
                        if o >= 1:
                            prev_o = o - 1
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                0,
                                0,
                            )
                        wait_seq_in(o, 0, 0)
                        start_scatter_seq(raw_out_ref, sub, 0, 0)
                        if o + 1 < total_gdn:
                            start_seq_in(o + 1, 0, 0)

                        if o >= 1:
                            prev_o = o - 1
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                1,
                                1,
                            )
                        wait_seq_in(o, 1, 1)
                        start_scatter_seq(raw_out_ref, sub, 1, 1)
                        if o + 1 < total_gdn:
                            start_seq_in(o + 1, 1, 1)

                last_o = total_gdn - 1
                wait_scatter_seq(
                    raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 0, 0
                )
                wait_scatter_seq(
                    raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 1, 1
                )
            elif NS == 3:
                for g_idx in range(g.NATT):
                    raw_out_ref = raw_out_refs[g_idx]
                    for sub in range(G_PER_ATT):
                        o = g_idx * G_PER_ATT + sub
                        if o >= 1:
                            prev_o = o - 1
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                2,
                                0,
                            )
                        wait_seq_in(o, 0, 0)
                        start_scatter_seq(raw_out_ref, sub, 0, 0)
                        start_seq_in(o, 2, 0)

                        if o >= 1:
                            prev_o = o - 1
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                1,
                                1,
                            )
                        wait_seq_in(o, 1, 1)
                        start_scatter_seq(raw_out_ref, sub, 1, 1)
                        if o + 1 < total_gdn:
                            start_seq_in(o + 1, 1, 1)

                        wait_scatter_seq(raw_out_ref, sub, 0, 0)
                        wait_seq_in(o, 2, 0)
                        start_scatter_seq(raw_out_ref, sub, 2, 0)
                        if o + 1 < total_gdn:
                            start_seq_in(o + 1, 0, 0)

                last_o = total_gdn - 1
                wait_scatter_seq(
                    raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 1, 1
                )
                wait_scatter_seq(
                    raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 2, 0
                )
            else:
                for g_idx in range(g.NATT):
                    raw_out_ref = raw_out_refs[g_idx]
                    for sub in range(G_PER_ATT):
                        o = g_idx * G_PER_ATT + sub

                        def scatter_pair(
                            pair, _, raw_out_ref=raw_out_ref, sub=sub, o=o
                        ):
                            s0 = 2 * pair
                            s1 = s0 + 1

                            @pl.when(pair > 0)
                            def _():
                                wait_scatter_seq(raw_out_ref, sub, s0 - 2, 0)

                            wait_seq_in(o, s0, 0)
                            start_scatter_seq(raw_out_ref, sub, s0, 0)

                            @pl.when(s0 + 2 < NS)
                            def _():
                                start_seq_in(o, s0 + 2, 0)

                            @pl.when(pair > 0)
                            def _():
                                wait_scatter_seq(raw_out_ref, sub, s1 - 2, 1)

                            wait_seq_in(o, s1, 1)
                            start_scatter_seq(raw_out_ref, sub, s1, 1)

                            @pl.when(s1 + 2 < NS)
                            def _():
                                start_seq_in(o, s1 + 2, 1)

                            return 0

                        lax.fori_loop(0, n_pairs, scatter_pair, 0)

                        if NS % 2 == 1:
                            wait_scatter_seq(raw_out_ref, sub, NS - 3, 0)
                            wait_seq_in(o, NS - 1, 0)
                            start_scatter_seq(raw_out_ref, sub, NS - 1, 0)
                            if o + 1 < total_gdn:
                                start_seq_in(o + 1, 0, 0)
                                start_seq_in(o + 1, 1, 1)
                            wait_scatter_seq(raw_out_ref, sub, NS - 2, 1)
                            wait_scatter_seq(raw_out_ref, sub, NS - 1, 0)
                        else:
                            if o + 1 < total_gdn:
                                start_seq_in(o + 1, 0, 0)
                                start_seq_in(o + 1, 1, 1)
                            wait_scatter_seq(raw_out_ref, sub, NS - 2, 0)
                            wait_scatter_seq(raw_out_ref, sub, NS - 1, 1)

    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.HBM)] * g.NATT
        + [pl.BlockSpec(memory_space=pltpu.SMEM)] * n_smem
        + [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ][:n_hbm]
    )
    out_shape = [jax.ShapeDtypeStruct(r.shape, r.dtype) for r in raw_pools]
    out_specs = [pl.BlockSpec(memory_space=pltpu.HBM)] * g.NATT
    call = pl.pallas_call(
        _kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        input_output_aliases={r: r for r in range(g.NATT)},
        compiler_params=pltpu.CompilerParams(
            collective_id=None,
            disable_bounds_checks=True,
            vmem_limit_bytes=ex.resolved_tiles.vmem_mib << 20,
        ),
        interpret=bool(interpret),
        name="qwen3_5_scatter_pool_states",
    )
    smem_args = (
        (pos, kv_pages)
        + ((ckpt_indices,) if has_gdn else ())
        + ((active_mask,) if act_mode == "ref" else ())
    )
    if knew.dtype != FP8:
        knew = jnp.clip(knew, -448.0, 448.0).astype(FP8)
    if vnew.dtype != FP8:
        vnew = jnp.clip(vnew, -448.0, 448.0).astype(FP8)
    if not bool(interpret):
        knew = pltpu.with_memory_space_constraint(knew, pltpu.HBM)
        vnew = pltpu.with_memory_space_constraint(vnew, pltpu.HBM)
        if has_gdn:
            assert conv_in is not None and rec is not None
            conv_in = pltpu.with_memory_space_constraint(conv_in, pltpu.HBM)
            rec = pltpu.with_memory_space_constraint(rec, pltpu.HBM)
    gdn_args = (conv_in, rec) if has_gdn else ()
    out = call(*raw_pools, *smem_args, knew, vnew, *gdn_args)
    return tuple(out)
