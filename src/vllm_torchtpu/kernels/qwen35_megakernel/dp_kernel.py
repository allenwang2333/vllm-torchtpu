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
"""DP4TP2 attention / EP8 MoE step body (MKConfig.dp = 4 on v7x-8).

Chip g (ranks 2g, 2g + 1) serves its own rows. Per layer:

  * mixer, per microbatch of the chip's rows: core c runs the TP8 mixer block of virtual rank
    4c + v for v = 0 .. 3 (that rank's weight slices from the dense stream, its smalls row, its
    GatedDeltaNet checkpoint block; every virtual rank of core c holds KV head c), sums the four
    partials, then exchanges the bf16 sum with core c ^ 1. Both cores add the two bf16 sums, so
    they hold the same residual rows.
  * MoE: each core sends the chip's post-norm rows to the same core of the other three chips
    (an all-gather into xb_all, chip g's rows at g * BW ..), runs the wide MoE phase (router,
    shared-expert slice and its EP experts) over all rows, and sends its bf16 partial of chip
    g's rows to both cores of chip g. Every core sums the eight partials of its chip's rows in
    rank order, so the cores of a chip again agree.

The exchanges need no credits: a payload is only overwritten after its receiver has passed a
later exchange with the sender (pair: two slots, each core waits for the other's previous
payload before sending; all-gather / combine: the next layer's sends follow receives that the
peer issues only after reading).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .attention_block import PagedAttentionBlock
from .config import BF16, F32, FP8, I32, rnd
from .context import MKBuildContext
from .gdn_block import GDNBlock
from .moe_block import MoEBlock
from .pool_dma import GDNPoolDMA, KVPoolDMA
from .rings import WeightRings


class DPExchange:
    """The dp step's remote copies: core-pair mixer exchange, chip row all-gather, MoE combine."""

    def __init__(self, ctx: MKBuildContext):
        self.ctx = ctx

    # ---- core pair: [MR, H] bf16 partial sums ----
    def pair_copy(self, slot):
        ctx = self.ctx
        scr, me = ctx.scr, ctx.me
        return pltpu.make_async_remote_copy(
            src_ref=scr["pr_send"].at[slot],
            dst_ref=scr["pr_recv"].at[slot, lax.rem(me, 2)],
            send_sem=scr["pr_ssem"].at[slot],
            recv_sem=scr["pr_rsem"].at[slot],
            device_id=(me ^ 1,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def pair_sum(self, x, seq):
        """x [>= MR, H] f32 core partial -> the chip's sum (f32 of two bf16 sums, the same on both
        cores). seq: the step's exchange index (traced), selecting the slot."""
        ctx = self.ctx
        MR, scr, sc = ctx.MR, ctx.scr, ctx.sc
        slot = lax.rem(seq, 2)
        c_idx = lax.rem(ctx.me, 2)
        cp = self.pair_copy(slot)
        with sc("dp_pair"):
            if ctx.comm:
                pl.when(seq >= 2)(cp.wait_send)
            scr["pr_send"][slot] = x[:MR].astype(BF16)
            scr["pr_recv"][slot, c_idx] = scr["pr_send"][slot]
            if not ctx.comm:
                return scr["pr_recv"][slot, c_idx].astype(F32) * 2.0
            cp.start()
            cp.wait_recv()
            return scr["pr_recv"][slot, 0].astype(F32) + scr["pr_recv"][slot, 1].astype(F32)

    def pair_drain(self, n_seq: int):
        if not self.ctx.comm:
            return
        for slot in range(min(2, n_seq)):
            self.pair_copy(slot).wait_send()

    # ---- chips: all-gather of the chip's post-norm rows into every core's xb_all ----
    def gather_copies(self):
        ctx = self.ctx
        scr, me, BW = ctx.scr, ctx.me, ctx.BW
        rows = scr["xb_all"].at[pl.ds(pl.multiple_of((me // 2) * BW, 8), BW)]
        return [
            pltpu.make_async_remote_copy(
                src_ref=rows,
                dst_ref=rows,
                send_sem=scr["ag_ssem"].at[i],
                recv_sem=scr["ag_rsem"].at[i],
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for i, off in enumerate((2, 4, 6))
        ]

    def gather(self):
        if not self.ctx.comm:
            return
        with self.ctx.sc("dp_gather"):
            cps = self.gather_copies()
            for d in cps:
                d.start()
            for d in cps:
                d.wait_recv()
            for d in cps:
                d.wait_send()

    # ---- MoE combine: every rank's partial of this chip's rows ----
    def combine(self):
        """part_all (this rank's MoE partial of every row) -> crecv[r] = rank r's bf16 partial
        of this chip's rows, for every r. xb_all (read by the experts, idle now) stages the bf16
        partials of all rows."""
        ctx = self.ctx
        scr, me, BW, R, sc = ctx.scr, ctx.me, ctx.BW, ctx.R, ctx.sc
        xb_all, part_all, crecv = scr["xb_all"], scr["part_all"], scr["crecv"]
        n_all = ctx.DPG * BW
        with sc("dp_stage"):
            for r0 in range(0, n_all, 64):
                n = min(64, n_all - r0)
                xb_all[r0:r0 + n, :] = part_all[r0:r0 + n, :].astype(BF16)
            mine = pl.ds(pl.multiple_of((me // 2) * BW, 8), BW)
            crecv[me] = xb_all[mine, :]
        if not ctx.comm:
            return
        cps = [
            pltpu.make_async_remote_copy(
                src_ref=xb_all.at[pl.ds(pl.multiple_of(((me ^ off) // 2) * BW, 8), BW)],
                dst_ref=crecv.at[me],
                send_sem=scr["c_ssem"].at[off - 1],
                recv_sem=scr["c_rsem"].at[off - 1],
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for off in range(1, R)
        ]
        with sc("dp_combine"):
            for d in cps:
                d.start()
            for d in cps:
                d.wait_recv()
            for d in cps:
                d.wait_send()


def kernel_dp(ctx: MKBuildContext, refs):
    """dp step body (see the module docstring); the per-layer stream and the microbatch buffers
    are the wide step's."""
    ctx.bind_refs(refs)
    rings = WeightRings(ctx)
    gdn_dma = GDNPoolDMA(ctx)
    kv_dma = KVPoolDMA(ctx)
    gdn_blk = GDNBlock(ctx, rings, gdn_dma)
    att_blk = PagedAttentionBlock(ctx, rings, kv_dma)
    moe_blk = MoEBlock(ctx, rings)
    xch = DPExchange(ctx)

    c, scr, R = ctx.c, ctx.scr, ctx.R
    W, hid_hbm, out_hbm = ctx.W, ctx.hid_ref, ctx.out_ref
    sc, rms_zc = ctx.sc, ctx.rms_zc
    B, MB, MR, H, L, E, AE = ctx.B, ctx.MB, ctx.MR, ctx.H, ctx.L, ctx.E, ctx.AE
    NS, NBD, NGI, NGO, NV = ctx.NS, ctx.NBD, ctx.NGI, ctx.NGO, ctx.NV
    ECOPY, lay, t_block, t_gdn = ctx.ECOPY, ctx.lay, ctx.t_block, ctx.t_gdn
    me, rb_hbm = ctx.me, ctx.rb_hbm
    N_MB, BW, MB_GDN, MB_ATT = ctx.N_MB, ctx.BW, ctx.MB_GDN, ctx.MB_ATT
    M_OUT = ctx.c_wide.M
    res_all, xb_all, smv = scr["res_all"], scr["xb_all"], scr["smv"]
    assert ctx.DP and ctx.FUSED and MB == B and B % 16 == 0 and BW % 8 == 0, (MB, B, BW)
    assert ctx.ckpt_is_3d, "dp: ckpt_indices carries one sequence block per virtual rank"
    o_ec = lay["ecopy"][0] if ECOPY else 0

    def rows(m):
        return pl.ds(pl.multiple_of(m * B, B), B)

    def chip_rows(m):  # microbatch m's rows in the all-chips buffers
        return pl.ds(pl.multiple_of((me // 2) * BW + m * B, 8), B)

    def smalls_copies(l, slot):
        cps = [
            pltpu.make_async_copy(W["smalls"].at[l, v], smv.at[v, slot], scr["sm_sem"].at[slot])
            for v in range(NV)
        ]
        if ECOPY:
            cps.append(pltpu.make_async_copy(
                W["smalls"].at[l, 0, :, pl.ds(o_ec, E)], scr["ecopy"].at[slot],
                scr["ec_sem"].at[slot]))
        return cps

    def layer(l, carry):
        slot = l % 2
        with sc("norm_in"):
            for cp in smalls_copies(0, slot):  # the waits read only the semaphore and size
                cp.wait()

            @pl.when(l + 1 < L)
            def _():
                for cp in smalls_copies(l + 1, 1 - slot):
                    cp.start()

        is_att = l % AE == AE - 1
        base_l = (l // AE) * t_block + (l % AE) * t_gdn
        per_mb = jnp.where(is_att, MB_ATT, MB_GDN)

        def mixer(m, carry_m):
            with sc("mixer_mb", 3):
                ctx.tv("mb", m)
                ctx.bind_microbatch(m)
                ctx.scr["sm"] = smv.at[0]
                with sc("norm_in"):
                    scr["res"][...] = res_all[rows(m)].astype(F32)
                    scr["xb"][...] = rms_zc(scr["res"][...], ctx.sm(slot, "in_norm")).astype(BF16)
                    scr["dsum"][...] = jnp.zeros((MB, H), F32)

                def virtual(v, carry_v):
                    with sc("mixer_vr", 3):
                        ctx.bind_virtual(m, v)
                        g0 = base_l + (m * NV + v) * per_mb

                        def att_branch():
                            att_blk.init_worklist()
                            att_blk.init_rope()
                            scr["dsum"][...] += att_blk.attention(l, slot, g0)

                        def gdn_branch():
                            scr["dsum"][...] += gdn_blk.gdn(l, slot, g0)

                        if AE == 1:
                            att_branch()
                        else:
                            lax.cond(is_att, att_branch, gdn_branch)
                    return carry_v

                lax.fori_loop(0, NV, virtual, 0)
                ctx.bind_microbatch(m)
                ctx.scr["sm"] = smv.at[0]
                tot = xch.pair_sum(scr["dsum"][...], l * N_MB + m)
                with sc("ar_res"):
                    res_m = rnd(scr["res"][...] + tot[0:B])
                with sc("moe_router"):
                    xb_all[chip_rows(m)] = rms_zc(res_m, ctx.sm(slot, "post_norm")).astype(BF16)
                    res_all[rows(m)] = res_m.astype(BF16)
            return carry_m

        lax.fori_loop(0, N_MB, mixer, 0)
        xch.gather()
        ctx.scr["sm"] = smv.at[0]
        with sc("moe", 3):
            moe_blk.moe_wide(l, slot, base_l + N_MB * NV * per_mb)
        xch.combine()

        def moe_res(m, carry_m):
            with sc("ar_res"):
                r0 = pl.multiple_of(m * B, B)
                tot = jnp.zeros((B, H), F32)
                crecv = scr["crecv"]
                if ctx.comm:
                    for r in range(R):
                        tot = tot + crecv[r, pl.ds(r0, B), :].astype(F32)
                else:
                    tot = crecv[me, pl.ds(r0, B), :].astype(F32) * float(R)
                res_all[rows(m)] = rnd(res_all[rows(m)].astype(F32) + tot).astype(BF16)
            return carry_m

        lax.fori_loop(0, N_MB, moe_res, 0)
        return carry

    def layer_scoped(l, carry):
        with sc("layer", 3):
            ctx.tv("layer", l)
            return layer(l, carry)

    # ------------------------------ step ------------------------------
    with sc("step_init"):
        scr["hitv"][...] = jnp.zeros((MR, 128), I32)
        zero_seen = pltpu.make_async_copy(scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"])
        zero_seen.start()
        for i in range(NBD):
            rings.fetch(jnp.int32(i))
        for cp in smalls_copies(0, 0):
            cp.start()
        if ctx.forced_routing:  # the first 128-row routing chunk; moe_wide chains the rest
            pltpu.make_async_copy(
                rb_hbm.at[0, pl.ds(0, 128)], scr["rb"].at[0], scr["rb_sem"].at[0]
            ).start()
        hid_in = pltpu.make_async_copy(hid_hbm, res_all.at[pl.ds(0, M_OUT)], scr["io_sem"].at[0])
        hid_in.start()
        if R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, R):
                pl.semaphore_signal(
                    barrier, 1, device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(barrier, R - 1)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        scr["knew_o"][...] = jnp.zeros((MR, ctx.HD), FP8)
        scr["vnew_o"][...] = jnp.zeros((MR, ctx.HD), FP8)
        for s in range(NS):
            scr["kv_rd_pend"][s] = jnp.int32(0)
            scr["kv_wr_pend"][s] = jnp.int32(0)
        if c.NGDN > 0:
            for slot_i in range(NGI):
                scr["gdn_in_pend"][slot_i] = jnp.int32(0)
            for slot_o in range(NGO):
                scr["gdn_out_mode"][slot_o] = jnp.int32(0)
            scr["gdn_out_page"][...] = jnp.zeros(scr["gdn_out_page"].shape, FP8)
        if ctx.PACK or ctx.QV:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
        xb_all[...] = jnp.zeros(xb_all.shape, BF16)
        hid_in.wait()
        if BW > M_OUT:
            res_all[M_OUT:BW, :] = jnp.zeros((BW - M_OUT, H), BF16)
        zero_seen.wait()
    lax.fori_loop(0, L, layer_scoped, 0)
    with sc("step_tail"):
        xch.pair_drain(L * N_MB)
        if c.NGDN > 0:
            for slot_o in range(NGO):
                gdn_dma.wait_gdn_out(slot_o)
        for s in range(NS):
            kv_dma.wait_kv_wr(s)
        cp = pltpu.make_async_copy(W["final_norm"], scr["fn"], scr["misc_sem"])
        cp.start()
        cp.wait()
        for m in range(N_MB):
            n = min(B, M_OUT - m * B)
            if n <= 0:
                break
            scr["xb"][...] = rms_zc(res_all[m * B:(m + 1) * B, :].astype(F32), scr["fn"][...]).astype(BF16)
            out_cp = pltpu.make_async_copy(
                scr["xb"].at[pl.ds(0, n)], out_hbm.at[pl.ds(m * B, n)], scr["io_sem"].at[1]
            )
            out_cp.start()
            out_cp.wait()
