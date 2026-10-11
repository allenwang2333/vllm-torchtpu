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
"""Collective communication (`AllReduceRing`: flat, RSAG, hierarchical 1/2/3, and group exchange)."""

from __future__ import annotations

from typing import Any

from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.megakernel.ops import BF16, F32, rnd
from vllm_torchtpu.kernels.megakernel.vmem import DerivedParams


class AllReduceRing:
    """In-kernel ICI remote DMA all-reduce and group exchange primitives."""

    def __init__(
        self,
        dp: DerivedParams,
        scr: dict[str, Any],
        me: Any,
        *,
        skip: frozenset[str] = frozenset(),
        sc: Any = None,
    ):
        self.dp = dp
        self.scr = scr
        self.me = me
        self.skip = skip
        self.sc = sc
        self.comm = dp.R > 1 and "ar" not in skip

    def ar_copies(self, slot, rnd_i: int = 0):
        """Remote copies of all-reduce round `rnd_i` of `slot` to peers `me ^ off`."""
        dp, scr, me = self.dp, self.scr, self.me
        HIER, RSAG, R = dp.HIER, dp.RSAG, dp.R
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
        if RSAG and rnd_i == 0:
            src, dst = (lambda off: scr["send"].at[slot, me ^ off]), scr["recv"].at[slot, me]
        elif RSAG:
            src, dst = (lambda off: scr["ag_send"].at[slot]), scr["ag_recv"].at[slot, me]
        else:
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
        """Reduce-sum `x [M, H]` f32 partial across all `R` ranks on slot `slot` (0 or 1)."""
        dp, scr, me, sc = self.dp, self.scr, self.me, self.sc
        WAR, MR, B, HPC, PR, PC = dp.WAR, dp.MR, dp.B, dp.HPC, dp.PR, dp.PC
        HIER, RSAG, R, CW, HC2 = dp.HIER, dp.RSAG, dp.R, dp.CW, dp.HC2
        send, recv = scr["send"], scr["recv"]
        with sc("ar_pack"):
            if WAR:
                val = x[:MR]
            else:
                chunks = [x[0:B, j * PC : (j + 1) * PC] for j in range(HPC)]
                if B * HPC < PR:
                    chunks.append(jnp.zeros((PR - B * HPC, PC), F32))
                val = jnp.concatenate(chunks, axis=0)
            if self.comm:
                @pl.when(jnp.logical_not(first))
                def _():
                    for rnd_i in dp.AR_ROUNDS:
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
        if not self.comm:
            return rnd(val)
        with sc("ar_comm"):
            self.ar_round(slot, 0)
        if HIER == 3:
            with sc("ar_sum"):
                b1 = lax.rem(me // 2, 2)
                send[1, slot] = (
                    recv[0, slot, 0].astype(F32) + recv[0, slot, 1].astype(F32)
                ).astype(BF16)
                recv[1, slot, b1] = send[1, slot]
            with sc("ar_comm"):
                self.ar_round(slot, 1)
            with sc("ar_sum"):
                b2 = lax.rem(me // 4, 2)
                send[2, slot] = (
                    recv[1, slot, 0].astype(F32) + recv[1, slot, 1].astype(F32)
                ).astype(BF16)
                recv[2, slot, b2] = send[2, slot]
            with sc("ar_comm"):
                self.ar_round(slot, 2)
            with sc("ar_sum"):
                return recv[2, slot, 0].astype(F32) + recv[2, slot, 1].astype(F32)
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
            return jnp.concatenate(
                [ag_recv[slot, j] for j in range(R)], axis=-1
            ).astype(F32)

    def add_residual(self, tot):
        """Unfold `tot` if folded and add to `scr['res']` with BF16 rounding."""
        dp, scr, sc = self.dp, self.scr, self.sc
        res = scr["res"]
        with sc("ar_res"):
            if not dp.WAR:
                tot = jnp.concatenate(
                    [tot[j * dp.B : (j + 1) * dp.B] for j in range(dp.HPC)], axis=-1
                )
            res[0 : dp.B, :] = rnd(res[0 : dp.B, :] + tot[0 : dp.B])

    def group_copies(self, src, dst, ssem, rsem):
        """Remote copies `src(off) -> dst` on group peer `me ^ off`, `off = 1 .. G-1`."""
        me, G = self.me, self.dp.G
        return [
            pltpu.make_async_remote_copy(
                src_ref=src(off),
                dst_ref=dst,
                send_sem=ssem.at[off - 1],
                recv_sem=rsem.at[off - 1],
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for off in range(1, G)
        ]

    @staticmethod
    def exchange(descs):
        for d in descs:
            d.start()
        for d in descs:
            d.wait_recv()
        for d in descs:
            d.wait_send()

    def drain_sends(self):
        """Wait for the last layer's all-reduce sends before kernel exit."""
        if self.comm:
            for slot in range(2):
                for rnd_i in self.dp.AR_ROUNDS:
                    for d in self.ar_copies(slot, rnd_i):
                        d.wait_send()

