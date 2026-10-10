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
"""On-chip and cross-chip Pallas remote-DMA all-reduce and residual helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .config import BF16, F32, rnd
from .context import MKBuildContext


class PallasAllReduce:
    """Remote-DMA all-reduce supporting HIER=0, 1, 2, 3 and RSAG."""

    def __init__(self, ctx: MKBuildContext):
        self.ctx = ctx

    def ar_copies(self, slot, rnd_i: int = 0):
        """Remote copies of all-reduce round rnd_i of this slot to the peers me ^ off. A peer's
        copy to this rank uses the same off, so it lands on the same semaphore and size."""
        ctx = self.ctx
        HIER, RSAG, R = ctx.HIER, ctx.RSAG, ctx.R
        scr, me = ctx.scr, ctx.me
        if HIER == 3:
            off = 1 << rnd_i
            bit = lax.rem(me // off, 2)
            return [
                pltpu.make_async_remote_copy(
                    src_ref=scr["send"].at[rnd_i, slot],
                    dst_ref=scr["recv"].at[rnd_i, slot, bit],
                    send_sem=scr["ar_ssem"].at[rnd_i, slot, 0],
                    recv_sem=scr["ar_rsem"].at[rnd_i, slot, 0],
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            ]
        if HIER in (1, 2):
            c_idx = lax.rem(me, 2)
            if rnd_i == 0:
                src = scr["send"].at[slot, 1 - c_idx] if HIER == 2 else scr["send"].at[slot]
                peers = [(0, 1)]
                dst = scr["recv"].at[slot, c_idx]
            elif rnd_i == 1:
                src = scr["ag_send"].at[slot]
                peers = list(enumerate((2, 4, 6)))
                dst = scr["ag_recv"].at[slot, me // 2]
            else:
                src = scr["s2_send"].at[slot]
                peers = [(0, 1)]
                dst = scr["s2_recv"].at[slot, c_idx]
            return [
                pltpu.make_async_remote_copy(
                    src_ref=src,
                    dst_ref=dst,
                    send_sem=scr["ar_ssem"].at[rnd_i, slot, i],
                    recv_sem=scr["ar_rsem"].at[rnd_i, slot, i],
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                for i, off in peers
            ]
        if RSAG and rnd_i == 0:  # lane chunk p of this rank's partial -> rank p
            src, dst = (lambda off: scr["send"].at[slot, me ^ off]), scr["recv"].at[slot, me]
        elif RSAG:  # this rank's chunk sum -> every peer
            src, dst = (lambda off: scr["ag_send"].at[slot]), scr["ag_recv"].at[slot, me]
        else:  # the whole partial -> every peer
            src, dst = (lambda off: scr["send"].at[slot]), scr["recv"].at[slot, me]
        return [
            pltpu.make_async_remote_copy(
                src_ref=src(off),
                dst_ref=dst,
                send_sem=scr["ar_ssem"].at[rnd_i, slot, off - 1],
                recv_sem=scr["ar_rsem"].at[rnd_i, slot, off - 1],
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for off in range(1, R)
        ]

    def ar_round(self, slot, rnd_i: int):
        descs = self.ar_copies(slot, rnd_i)
        for d in descs:
            d.start()
        for d in descs:
            d.wait_recv()

    def allreduce(self, slot, x, first):
        """x [M, H] f32 partial -> the sum over ranks of its B live rows ([MR, H] when WAR,
        else folded [PR, PC]). Only the incoming payloads are waited for; this slot's sends are
        waited for when its send buffers are next rewritten (not first) and at kernel exit."""
        ctx = self.ctx
        WAR, MR, B, PC, HPC, PR = ctx.WAR, ctx.MR, ctx.B, ctx.PC, ctx.HPC, ctx.PR
        HIER, N_HIER3, RSAG, R, CW, HC2 = (
            ctx.HIER,
            ctx.N_HIER3,
            ctx.RSAG,
            ctx.R,
            ctx.CW,
            ctx.HC2,
        )
        AR_ROUNDS, comm = ctx.AR_ROUNDS, ctx.comm
        scr, me, sc = ctx.scr, ctx.me, ctx.sc
        send, recv = scr["send"], scr["recv"]
        with sc("ar_pack"):
            if WAR:
                val = x[:MR]
            else:
                chunks = [x[0:B, j * PC : (j + 1) * PC] for j in range(HPC)]
                if B * HPC < PR:
                    chunks.append(jnp.zeros((PR - B * HPC, PC), F32))
                val = jnp.concatenate(chunks, axis=0)
            if comm:
                @pl.when(jnp.logical_not(first))
                def _():
                    for rnd_i in AR_ROUNDS:
                        for d in self.ar_copies(slot, rnd_i):
                            d.wait_send()

            c_idx = lax.rem(me, 2)
            if HIER == 3:
                send[0, slot] = val.astype(BF16)
                recv[0, slot, c_idx] = send[0, slot]
            elif HIER == 2:
                send[slot, 0] = val[:, :HC2].astype(BF16)
                send[slot, 1] = val[:, HC2:].astype(BF16)
                recv[slot, c_idx] = send[slot, c_idx]
            elif HIER == 1:
                send[slot] = val.astype(BF16)
                recv[slot, c_idx] = send[slot]
            elif RSAG:
                for j in range(R):
                    send[slot, j] = val[:, j * CW : (j + 1) * CW].astype(BF16)
                recv[slot, me] = send[slot, me]
            else:
                send[slot] = val.astype(BF16)
                recv[slot, me] = send[slot]
        if not comm:
            return rnd(val)
        with sc("ar_comm"):
            self.ar_round(slot, 0)
        if HIER == 3:
            for rnd_i in range(1, N_HIER3):
                with sc("ar_sum"):
                    b_i = lax.rem(me // (1 << rnd_i), 2)
                    send[rnd_i, slot] = (
                        recv[rnd_i - 1, slot, 0].astype(F32)
                        + recv[rnd_i - 1, slot, 1].astype(F32)
                    ).astype(BF16)
                    recv[rnd_i, slot, b_i] = send[rnd_i, slot]
                with sc("ar_comm"):
                    self.ar_round(slot, rnd_i)
            with sc("ar_sum"):
                return (
                    recv[N_HIER3 - 1, slot, 0].astype(F32)
                    + recv[N_HIER3 - 1, slot, 1].astype(F32)
                )
        if HIER in (1, 2):
            ag_send, ag_recv = scr["ag_send"], scr["ag_recv"]
            with sc("ar_sum"):
                ag_send[slot] = (
                    recv[slot, 0].astype(F32) + recv[slot, 1].astype(F32)
                ).astype(BF16)
                ag_recv[slot, me // 2] = ag_send[slot]
            with sc("ar_comm"):
                self.ar_round(slot, 1)
            if HIER == 1:
                with sc("ar_sum"):
                    return jnp.sum(ag_recv[slot].astype(F32), axis=0)
            s2_send, s2_recv = scr["s2_send"], scr["s2_recv"]
            with sc("ar_sum"):
                s2_send[slot] = jnp.sum(ag_recv[slot].astype(F32), axis=0).astype(BF16)
                s2_recv[slot, c_idx] = s2_send[slot]
            with sc("ar_comm"):
                self.ar_round(slot, 2)
            with sc("ar_sum"):
                return jnp.concatenate(
                    [s2_recv[slot, 0], s2_recv[slot, 1]], axis=-1
                ).astype(F32)
        if not RSAG:
            with sc("ar_sum"):
                return jnp.sum(recv[slot].astype(F32), axis=0)
        ag_send, ag_recv = scr["ag_send"], scr["ag_recv"]
        with sc("ar_sum"):
            ag_send[slot] = jnp.sum(recv[slot].astype(F32), axis=0).astype(BF16)
            ag_recv[slot, me] = ag_send[slot]
        with sc("ar_comm"):
            self.ar_round(slot, 1)
        with sc("ar_sum"):
            return jnp.concatenate([ag_recv[slot, j] for j in range(R)], axis=-1).astype(F32)

    def add_residual(self, tot):
        ctx = self.ctx
        B, HPC, WAR, scr, sc = ctx.B, ctx.HPC, ctx.WAR, ctx.scr, ctx.sc
        res = scr["res"]
        with sc("ar_res"):
            if not WAR:
                tot = jnp.concatenate([tot[j * B : (j + 1) * B] for j in range(HPC)], axis=-1)
            res[0:B, :] = rnd(res[0:B, :] + tot[0:B])

    def wait_sends(self):
        """Wait for all in-flight all-reduce sends at kernel exit."""
        ctx = self.ctx
        if ctx.comm:
            for ar_slot in range(2):
                for rnd_i in ctx.AR_ROUNDS:
                    for d in self.ar_copies(ar_slot, rnd_i):
                        d.wait_send()
