# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The barrier-free cache writer ('glm52_decode_mk_writer'): a second call after the megakernel that applies the
step's owned new rows to the (aliased, donated) caches.

Port of glm5_mk_dcp's `build_writer` (kernel.py:2431 @ 0a73f68bc; hardware byte-exact on per-layer caches,
0 bad bytes / 315,768 rows, 0 neighbour clobber [measured: j-31f97434]). Changes: the index-K cache resolves its
pages through its own block table `bti` (production: virtual ordinal mod that cache's page count), the cache
operands follow layout.cache_kinds / kernel.cache_groups, and a `dcp_owner` fault control (interleave 1).

No barrier: an aliased operand of a barrier-carrying call is copied in and out by XLA (probes/02-alias-copy).
nrow / rrow / irec are the megakernel's staging outputs (identical on every rank); each rank writes the rows whose
position it owns, for every layer:
  nope: the row's 128 words -> (page, slot), one HBM -> HBM DMA per (layer, row);
  rope: the 4-token row (page, slot >> 2) read, the quarters of every owned new token in it overlaid, written back;
  idx:  the SAL page read, the owned new tokens' record columns patched, written back (per distinct page)."""

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import layout as LY
from .kernel import bswitch, cache_groups, cache_names

I32, U32, U8 = jnp.int32, jnp.uint32, jnp.uint8
WRITER_NAME = "glm52_decode_mk_writer"


def build_writer(cfg, *, max_pages, nb, nbi, interpret=False, groups=None, fault=""):
    """Returns (writer, info). writer(scal, rmeta, bt, bti, nrow, rrow, irec, *caches) -> caches (aliased)."""
    R, L, TMAX = cfg.R, cfg.L, cfg.TMAX
    pg = cfg.page
    lg_pg = int(math.log2(pg))
    ILV = 1 if fault == "dcp_owner" else cfg.ILV        # FAULT dcp_owner: interleave 1 ownership on the write
    G_ = ILV * R
    IR4 = cfg.IREC // 4
    ck = LY.cache_kinds(cfg, nb, nbi)
    G = cache_groups(groups, ck)
    c_names = cache_names(ck, G)
    n_c = len(c_names)
    full_l = LY.layers_of(cfg, "iwq")
    fr, sk = cfg.idx_freq, cfg.idx_skip
    CPOS = {"nope": lambda l: l, "rope": lambda l: l,
            "idx": lambda l: jnp.clip(jnp.where(l < sk, l, sk + (l - sk) // fr), 0, max(len(full_l), 1) - 1)}
    scratch = {
        "rold": pltpu.VMEM((L, TMAX, 128), U32), "rnew": pltpu.VMEM((L, TMAX, 128), U32),
        "ipage": pltpu.VMEM((IR4, 4, pg), U8), "rec_v": pltpu.VMEM((TMAX, 256), I32),
        "rec_s": pltpu.SMEM((TMAX, 256), I32),
        "sem": pltpu.SemaphoreType.DMA((4,)),
    }
    s_names = list(scratch)
    n_fixed = 7

    def kernel(*refs):
        scal, rmeta, bt, bti, nrow, rrow, irec = refs[:n_fixed]
        c_out = refs[n_fixed + n_c:n_fixed + 2 * n_c]
        scr = dict(zip(s_names, refs[n_fixed + 2 * n_c:]))
        me = scal[2]
        T = scal[0]
        CS = {}
        for (k, gi), r in zip(c_names, c_out):
            CS.setdefault(k, []).append(r)

        def csel(k, l, fn):
            ng = len(CS[k])
            pos = CPOS[k](l)
            if ng == 1:
                fn(CS[k][0].at[pos])
            else:
                gs = len(ck[k][0]) // ng
                bswitch(pos // gs, {gi: (lambda gi=gi: fn(CS[k][gi].at[pos - gi * gs])) for gi in range(ng)})

        def mine(t_):
            pos = rmeta[t_, 1]
            return jnp.logical_and(rmeta[t_, 0] > 0, (pos % G_) // ILV == me)

        def where_(t_, table):
            pos = rmeta[t_, 1]
            req = rmeta[t_, 2]
            lo = (pos // G_) * ILV + pos % ILV
            return table[req, jnp.minimum(lo >> lg_pg, max_pages - 1)], lo & (pg - 1), pos

        cp = pltpu.make_async_copy(rrow, scr["rnew"], scr["sem"].at[0])
        cp.start()
        cp.wait()

        def per_layer(l, c):
            def nope(t_, c2):
                @pl.when(jnp.logical_and(t_ < T, mine(t_)))
                def _():
                    page, sl, _p = where_(t_, bt)
                    csel("nope", l, lambda r: pltpu.make_async_copy(
                        nrow.at[l, pl.ds(t_, 1)], r.at[page, pl.ds(sl, 1)], scr["sem"].at[1]).start())
                return c2
            lax.fori_loop(0, TMAX, nope, 0)

            def rread(t_, c2):
                @pl.when(jnp.logical_and(t_ < T, mine(t_)))
                def _():
                    page, sl, _p = where_(t_, bt)
                    csel("rope", l, lambda r: pltpu.make_async_copy(
                        r.at[page, pl.ds(sl >> 2, 1)], scr["rold"].at[l, pl.ds(t_, 1)], scr["sem"].at[2]).start())
                return c2
            lax.fori_loop(0, TMAX, rread, 0)

            def rwait(t_, c2):
                @pl.when(jnp.logical_and(t_ < T, mine(t_)))
                def _():
                    pltpu.make_async_copy(rrow.at[0, pl.ds(0, 1)], scr["rold"].at[0, pl.ds(0, 1)],
                                          scr["sem"].at[2]).wait()
                return c2
            lax.fori_loop(0, TMAX, rwait, 0)
            lane = lax.broadcasted_iota(I32, (1, 128), 1)

            def rmerge(t_, c2):
                @pl.when(jnp.logical_and(t_ < T, mine(t_)))
                def _():
                    page, sl, pos = where_(t_, bt)
                    req = rmeta[t_, 2]
                    row = scr["rold"][l, pl.ds(t_, 1), :]

                    def ov(t2, row_):
                        pos2 = rmeta[t2, 1]
                        same = jnp.logical_and(jnp.logical_and(t2 < T, mine(t2)), jnp.logical_and(
                            rmeta[t2, 2] == req, pos2 // ILV == pos // ILV))
                        q2 = pos2 & 3
                        new = scr["rnew"][l, pl.ds(t2, 1), :]
                        return jnp.where(jnp.logical_and(same, lane // 32 == q2), new, row_)
                    row = lax.fori_loop(0, TMAX, ov, row)
                    scr["rold"][l, pl.ds(t_, 1), :] = row
                    csel("rope", l, lambda r: pltpu.make_async_copy(
                        scr["rold"].at[l, pl.ds(t_, 1)], r.at[page, pl.ds(sl >> 2, 1)], scr["sem"].at[3]).start())
                return c2
            lax.fori_loop(0, TMAX, rmerge, 0)
            return c
        lax.fori_loop(0, L, per_layer, 0)

        lane_pg = lax.broadcasted_iota(I32, (4, pg), 1)
        sub4 = lax.broadcasted_iota(I32, (4, pg), 0)

        def per_idx_layer(l, c):
            full = jnp.logical_or(l < cfg.idx_skip, (l - cfg.idx_skip) % cfg.idx_freq == cfg.idx_freq - 1)

            @pl.when(full)
            def _():
                cp_ = pltpu.make_async_copy(irec.at[l], scr["rec_v"], scr["sem"].at[0])
                cp_.start()
                cp_.wait()
                cp_ = pltpu.make_async_copy(scr["rec_v"], scr["rec_s"], scr["sem"].at[0])
                cp_.start()
                cp_.wait()

                def body(t_, cur):
                    ok = jnp.logical_and(t_ < T, mine(t_))
                    page, sl, _p = where_(t_, bti)

                    @pl.when(ok)
                    def _():
                        def go(r):
                            @pl.when(jnp.logical_and(page != cur, cur >= 0))
                            def _():
                                c3 = pltpu.make_async_copy(scr["ipage"], r.at[cur], scr["sem"].at[0])
                                c3.start()
                                c3.wait()

                            @pl.when(page != cur)
                            def _():
                                c3 = pltpu.make_async_copy(r.at[page], scr["ipage"], scr["sem"].at[0])
                                c3.start()
                                c3.wait()
                            for w in range(IR4):
                                v4 = [scr["rec_s"][t_, 4 * w + q_] for q_ in range(4)]
                                col = jnp.where(sub4 == 0, v4[0], jnp.where(sub4 == 1, v4[1], jnp.where(
                                    sub4 == 2, v4[2], v4[3])))
                                old = scr["ipage"][w].astype(I32)
                                scr["ipage"][w] = jnp.where(lane_pg == sl, col, old).astype(U8)
                        csel("idx", l, go)
                    return jnp.where(ok, page, cur)
                cur = lax.fori_loop(0, TMAX, body, jnp.int32(-1))

                @pl.when(cur >= 0)
                def _():
                    csel("idx", l, lambda r: (lambda c3: (c3.start(), c3.wait()))(
                        pltpu.make_async_copy(scr["ipage"], r.at[cur], scr["sem"].at[0])))
            return c
        if full_l:
            lax.fori_loop(0, L, per_idx_layer, 0)

        n_own = jnp.int32(0)
        for t_ in range(TMAX):
            n_own = n_own + jnp.logical_and(t_ < T, mine(t_)).astype(I32)

        def drain(i, c):
            pltpu.make_async_copy(nrow.at[0, pl.ds(0, 1)], scr["rold"].at[0, pl.ds(0, 1)], scr["sem"].at[1]).wait()
            pltpu.make_async_copy(nrow.at[0, pl.ds(0, 1)], scr["rold"].at[0, pl.ds(0, 1)], scr["sem"].at[3]).wait()
            return c
        lax.fori_loop(0, n_own * L, drain, 0)

    c_shapes = [jax.ShapeDtypeStruct((len(ck[k][0]) // G[k],) + ck[k][1], ck[k][2]) for k, gi in c_names]
    call = pl.pallas_call(
        kernel,
        name=WRITER_NAME,
        out_shape=c_shapes,
        in_specs=([pl.BlockSpec(memory_space=pltpu.SMEM)] * 4 + [pl.BlockSpec(memory_space=pltpu.HBM)] * 3
                  + [pl.BlockSpec(memory_space=pltpu.HBM)] * n_c),
        out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * n_c,
        scratch_shapes=[scratch[k] for k in s_names],
        input_output_aliases={n_fixed + i: i for i in range(n_c)},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024, disable_bounds_checks=True),
        interpret=(interpret if isinstance(interpret, pltpu.InterpretParams) else pltpu.InterpretParams())
        if interpret else False,
    )
    return call, dict(c_names=c_names, ck=ck, groups=G)
