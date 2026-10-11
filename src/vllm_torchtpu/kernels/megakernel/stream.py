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
"""Two-ring weight streamer (`TwoRing`: static `dpool` + dynamic routed `epool`)."""

from __future__ import annotations

from typing import Any

from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.megakernel import pool_alias
from vllm_torchtpu.kernels.megakernel.ops import F32, bswitch
from vllm_torchtpu.kernels.megakernel.vmem import DerivedParams

NN = (((1,), (0,)), ((), ()))


class TwoRing:
    """Dual weight streaming ring: `dpool` for static weights, `epool` for routed experts."""

    def __init__(
        self,
        c: Any,
        dp: DerivedParams,
        W: dict[str, Any],
        scr: dict[str, Any],
        me: Any,
        *,
        interpret: bool = False,
        skip: frozenset[str] = frozenset(),
        sc: Any = None,
    ):
        self.c = c
        self.dp = dp
        self.W = W
        self.scr = scr
        self.me = me
        self.interpret = interpret
        self.skip = skip
        self.sc = sc

    def mm(self, x, w, dims=NN):
        if self.interpret and x.dtype != w.dtype:
            w = w.astype(x.dtype)
        return lax.dot_general(x, w, dims, preferred_element_type=F32)

    def dview(self, kind: str, bank):
        dp = self.dp
        if dp.typed:
            return self.scr["dring_" + kind].at[bank]
        shp, dt = dp.vshape[kind], jnp.dtype(dp.vdtype[kind])
        return pool_alias.view(
            self.scr["dpool"].at[
                pl.ds(bank * dp.BROWS, pool_alias.storage_size(shp, dt) // 128)
            ],
            shp,
            dt,
        )

    def eview(self, kind: str, bank):
        dp = self.dp
        if dp.typed:
            return self.scr["ering_" + kind].at[bank]
        shp, dt = dp.vshape[kind], jnp.dtype(dp.vdtype[kind])
        return pool_alias.view(
            self.scr["epool"].at[
                pl.ds(bank * dp.EBROWS, pool_alias.storage_size(shp, dt) // 128)
            ],
            shp,
            dt,
        )

    def dcopy(self, kind: str, src, bank):
        return pltpu.make_async_copy(src, self.dview(kind, bank), self.scr["dsem"].at[bank])

    def ecopy(self, kind: str, src, bank):
        return pltpu.make_async_copy(src, self.eview(kind, bank), self.scr["esem"].at[bank])

    def _seg_issue(self, segs, r, ords, bank):
        for kind, lo, hi, ordi in segs:
            @pl.when(jnp.logical_and(r >= lo, r < hi))
            def _(kind=kind, lo=lo, ordi=ordi):
                self.dcopy(kind, self.W[kind].at[ords[ordi], r - lo], bank).start()

    def fetch(self, g):
        """Issue dense stream tile `g` into bank `g % NBD` (static schedule)."""
        dp = self.dp
        AE = dp.AE

        @pl.when(jnp.logical_and(g < dp.T_TOTAL, "dense" not in self.skip))
        def _():
            blk = g // dp.t_block
            sub = g - blk * dp.t_block
            lt = jnp.minimum(sub // dp.t_gdn, AE - 1)
            r = sub - lt * dp.t_gdn
            layer = blk * AE + lt
            bank = g % dp.NBD

            if AE > 1:
                @pl.when(lt < AE - 1)
                def _gdn():
                    self._seg_issue(
                        dp.segs_gdn,
                        r,
                        (blk * (AE - 1) + lt, layer),
                        bank,
                    )

            @pl.when(lt == AE - 1)
            def _att():
                self._seg_issue(
                    dp.segs_att,
                    r,
                    (blk, layer),
                    bank,
                )

    def dtake(self, kind: str, g):
        bank = g % self.dp.NBD
        if "dense" not in self.skip:
            with self.sc("dwait", 2):
                self.dcopy(kind, self.W[kind].at[0, 0], bank).wait()
        return self.dview(kind, bank)

    def dgive(self, g):
        self.fetch(g + self.dp.NBD)

    def flush(self, g0, n: int):
        """Refill the slots of tiles `g0 .. g0+n-1`, consumed lazily (no refill yet)."""
        NBD = self.dp.NBD
        for t in range(n):
            self.fetch(g0 + t + NBD)

    def dmatmul(self, kind: str, g0, xsrc, N: int, lazy: bool = False):
        """`acc[:, :N] = xsrc[:, :K] @ W(kind)`, tiles `g0 .. g0+nt-1` of the dense stream."""
        dp = self.dp
        n_t, t_k = dp.nt[kind], dp.tk[kind]
        acc = self.scr["acc"]
        acc[:, :N] = jnp.zeros((dp.MB, N), F32)

        def body(t, carry):
            g = g0 + t
            w = self.dtake(kind, g)
            k0 = pl.multiple_of(t * t_k, t_k)
            acc[:, :N] += self.mm(xsrc[:, pl.ds(k0, t_k)], w[...])
            if not lazy:
                self.dgive(g)
            return carry

        lax.fori_loop(0, n_t, body, 0)
        return acc[:, :N]

    def fetch_e(self, u, l, t=None):
        """Issue routed expert unit `u` into bank `u % NBE`."""
        dp = self.dp
        c = self.c
        TE, T13, NBE, NSS = dp.TE, dp.T13, dp.NBE, dp.NSS
        if t is None and isinstance(u, int):
            t = u % TE
        i = u // TE if t is None else (u - t) // TE
        if dp.ECOPY:
            w = self.scr["wlist"][i]
        else:
            e = self.scr["elist"][i]
            w = e if c.moe_tp else e - self.me * dp.ER
        bank = u % NBE

        def es_copy():
            s = i % NSS
            pltpu.make_async_copy(
                self.W["es"].at[l, w], self.scr["es"].at[s], self.scr["es_sem"].at[s]
            ).start()

        def e13_start(tt):
            if c.stack_e:
                self.ecopy("e13", self.W["e13"].at[l, w, tt], bank).start()
            else:
                bswitch(
                    l,
                    [lambda r=r: self.ecopy("e13", r.at[w, tt], bank).start() for r in self.W["e13"]],
                )

        def e2_start(tt):
            if c.stack_e:
                self.ecopy("e2", self.W["e2"].at[l, w, tt], bank).start()
            else:
                bswitch(
                    l,
                    [lambda r=r: self.ecopy("e2", r.at[w, tt], bank).start() for r in self.W["e2"]],
                )

        if t is not None:
            if t < T13:
                e13_start(t)
            else:
                e2_start(t - T13)
            if t == 0:
                es_copy()
            return
        t = u - i * TE

        @pl.when(t < T13)
        def _():
            e13_start(t)

        @pl.when(t >= T13)
        def _():
            e2_start(t - T13)

        pl.when(t == 0)(es_copy)

    def etake(self, kind: str, u):
        bank = u % self.dp.NBE
        with self.sc("ewait", 2):
            w0 = self.W[kind].at[0, 0, 0] if self.c.stack_e else self.W[kind][0].at[0, 0]
            self.ecopy(kind, w0, bank).wait()
        return self.eview(kind, bank)

    def egive(self, u, l, t=None):
        NBE, TE = self.dp.NBE, self.dp.TE
        nu = u + NBE
        nt_ = None if t is None or NBE % TE else t

        @pl.when(nu < self.scr["st"][0])
        def _():
            self.fetch_e(nu, l, nt_)

