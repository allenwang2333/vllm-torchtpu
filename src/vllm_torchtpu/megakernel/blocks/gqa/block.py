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
"""Grouped-Query Attention (`GQABlock`) with sequence-split KV ring and MTP."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec
from ...fused_io_hook import fused_io
from ...ops import BF16, F32, FP8, I32, rnd
from ...pool.config import POOL_STATE_NAME, cache_exec, resolved_max_pages_per_seq
from ...spec import Block, SmallField, StateDecl, WeightDecl
from .attention import att_core
from .context import GQAEmit
from .kv import build_worklist, make_kv_ring
from .merge import split_merge
from .new_token import att_new
from .params import GQAParams, GQATunables, derive_params
from .staging import att_q, kv_new, q_exchange_start, q_gather_finish


class GQABlock(Block):
    """Full GQA block with QK RMSNorm, partial RoPE, Q/softmax exchange, and gate."""

    name: str = "gqa"
    reduction = "allreduce"
    input_norm: str | None = "in_norm"
    Tunables = GQATunables

    def __init__(self, pool_layout: Any = None) -> None:
        # `PoolLayout` of the model's paged pools (None: the default page
        # packing `(2, HD // 4, 4, page_size)`, the only one the paged reads use).
        self.pool_layout = pool_layout

    def kv_page_shape(self, g: Any, ex: ExecSpec) -> tuple[int, ...]:
        """Shape of one pool page, from `pool_layout` when the model sets one.

        Raises `MegakernelConfigError` if the layout's page packing differs from
        the `(2, HD // 4, 4, page_size)` packing the paged KV reads assume.
        """
        shape = (2, g.HD // 4, 4, cache_exec(ex).page_size)
        if self.pool_layout is None:
            return shape
        lay_shape = tuple(self.pool_layout.kv_page_shape)
        if lay_shape != shape:
            raise MegakernelConfigError(
                f"GQABlock paged KV reads need pool pages {shape}, but the pool "
                f"layout has {lay_shape}"
            )
        return lay_shape

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return [
            WeightDecl("att_in", K=g.H, N=nai, dtype=FP8, stream_ring="static"),
            WeightDecl("att_out", K=nh * g.HD, N=g.H, dtype=FP8, stream_ring="static"),
        ]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return [
            SmallField("in_norm", g.H),
            SmallField("in_s", nai),
            SmallField("out_s", g.H),
            SmallField("qn", g.HD),
            SmallField("kn", g.HD),
        ]

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        if cache_exec(ex).paged_kv:
            st = [
                StateDecl(
                    POOL_STATE_NAME,
                    (cache_exec(ex).num_pages, *self.kv_page_shape(g, ex)),
                    FP8,
                    access="r",
                    per_layer=True,
                    memory_space="hbm",
                    tupled=True,
                ),
                StateDecl(
                    "page_table",
                    (ex.NS, resolved_max_pages_per_seq(ex)),
                    I32,
                    access="r",
                    per_layer=False,
                    memory_space="smem",
                ),
            ]
            if cache_exec(ex).FUSED:
                st.extend(fused_io().gqa_fused_states(ex))
            return st
        return [
            StateDecl("kc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
            StateDecl("vc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
        ]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        if cache_exec(ex).FUSED:
            return fused_io().gqa_fused_outputs(g, ex)
        kv_dt = FP8 if cache_exec(ex).paged_kv else F32
        return [
            StateDecl("knew", (ex.MR, g.HD), kv_dt, access="w"),
            StateDecl("vnew", (ex.MR, g.HD), kv_dt, access="w"),
        ]

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return {
            "acc": max(nai, g.H),
            "ybuf": nai,
            "xb2": nh * g.HD,
            "mix": nh * g.HD,
        }

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> GQAParams:
        if cache_exec(ex).paged_kv:
            self._check_pool_layout(plan)
        return derive_params(g, ex, plan, self.tunables(ex))

    def _check_pool_layout(self, plan: Any) -> None:
        """`pool_layout` must give the same pool pages as the model's pool
        backend layout, which the cache pipes allocate and fill."""
        pb = plan.model_spec.cache_backend
        lay = None if pb is None else pb.layout
        if self.pool_layout is None or lay is None:
            return
        mine, theirs = tuple(self.pool_layout.kv_page_shape), tuple(lay.kv_page_shape)
        if mine != theirs:
            raise MegakernelConfigError(
                f"GQABlock pool_layout pages {mine} differ from the pool backend "
                f"layout's {theirs} on ModelSpec {plan.model_spec.name!r}"
            )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        ap: GQAParams = plan.block_params[self.name]
        scr: dict[str, Any] = {}
        if not cache_exec(ex).paged_kv:
            scr["kbuf"] = pltpu.VMEM((ap.NKV, ap.BLK, ap.HD), BF16)
            scr["vbuf"] = pltpu.VMEM((ap.NKV, ap.BLK, ap.HD), BF16)
        scr.update(
            {
                "kv_sem": pltpu.SemaphoreType.DMA(
                    (ap.NKVR,) if cache_exec(ex).paged_kv else (2, ap.NKV)
                ),
                "wl_b": pltpu.SMEM((ap.NWL,), I32),
                "wl_t": pltpu.SMEM((ap.NWL,), I32),
                "wl_cnt": pltpu.SMEM((1,), I32),
                "qb": pltpu.VMEM((ex.NS, ap.TQ, ap.HD), BF16),
                "am": pltpu.VMEM((ex.NS, ap.TQ, 128), F32),
                "al": pltpu.VMEM((ex.NS, ap.TQ, 128), F32),
                "aacc": pltpu.VMEM((ex.NS, ap.TQ, ap.HD), F32),
                "knew": pltpu.VMEM((ex.MR, ap.HD), F32),
                "vnew": pltpu.VMEM((ex.MR, ap.HD), F32),
                "gate": pltpu.VMEM((ex.MR, ap.nh * ap.HD), F32),
                "rcs": pltpu.VMEM((2, ex.MR, ap.ROT // 2), F32),
            }
        )
        if not cache_exec(ex).FUSED:
            scr["kv_out_sem"] = pltpu.SemaphoreType.DMA((2,))
        else:
            scr.update(fused_io().gqa_fused_scratch(ex, ap.HD))
        if cache_exec(ex).paged_kv:
            if not ap.KVE:
                scr["kvbuf"] = pltpu.VMEM((ap.NKV, 2, ap.HD // 4, 4, ap.BLK), FP8)
            scr["knew_o"] = pltpu.VMEM((ex.MR, ap.HD), FP8)
            scr["vnew_o"] = pltpu.VMEM((ex.MR, ap.HD), FP8)
            scr["wl_kp"] = pltpu.SMEM(
                (ap.NWL * (ap.BLK // cache_exec(ex).page_size),), I32
            )
        if ap.SPLIT:
            q_rows = ap.NBP if ap.PACK else ex.B
            scr.update(
                {
                    "qg": pltpu.VMEM((ap.G, q_rows, 8, ap.HD), F32),
                    "pso": pltpu.VMEM((ap.G, q_rows, 8, ap.PW), F32),
                    "pri": pltpu.VMEM((ap.G, q_rows, 8, ap.PW), F32),
                    "sq_ssem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                    "sq_rsem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                    "sp_ssem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                    "sp_rsem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                }
            )
        elif ex.VER:
            scr["qg"] = pltpu.VMEM((1, ex.B, 8, ap.HD), F32)
        if ap.QV:
            scr["rope"] = pltpu.VMEM((2, ex.MR, 128), F32)  # QV-only cos / sin table
            scr["qg"] = pltpu.VMEM(
                (ap.G if ap.SPLIT and not ap.GQ else 1, ap.NHH, ap.QVR, 128), F32
            )
            scr["ost"] = pltpu.VMEM((ap.NHH, ap.QVR, 128), F32)
        if ap.GQ:
            del scr["pso"]
            scr.update(
                {
                    "qxb": pltpu.VMEM((ap.G, ex.NS, ap.QG, ap.HD), BF16),
                    "psend": pltpu.VMEM((ap.G, ex.NS + 1, ap.QG, ap.HD), F32),
                    "pri": pltpu.VMEM((ap.G, ex.NS + 1, ap.QG, ap.HD), F32),
                }
            )
        return scr

    def prologue(self, ctx: Any) -> None:
        """Build this rank's KV worklist (`wl_b`, `wl_t`, `wl_kp`, `wl_cnt[0]`)."""
        build_worklist(ctx, ctx.block_params(self.name))

    def init_buffers(self, ctx: Any) -> None:
        """Initialize attention-specific VMEM scratch buffers after barrier."""
        ex = ctx.ex
        ap: GQAParams = ctx.block_params(self.name)
        scr, pos_ref = ctx.scr, ctx.pos_ref
        scr["knew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        scr["vnew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        if cache_exec(ex).paged_kv:
            scr["knew_o"][...] = jnp.zeros((ex.MR, ap.HD), FP8)
            scr["vnew_o"][...] = jnp.zeros((ex.MR, ap.HD), FP8)
        if cache_exec(ex).FUSED:
            fused_io().gqa_fused_init_buffers(ctx)
        if ap.SPLIT and not ap.GQ:
            scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)
        if ap.PACK or ap.QV:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
        if ap.QV:
            half_ = ap.ROT // 2
            inv_ = 1.0 / (
                ap.theta
                ** (lax.broadcasted_iota(I32, (1, half_), 1).astype(F32) * 2.0 / ap.ROT)
            )
            rows_ = lax.broadcasted_iota(I32, (ex.MR, half_), 0)
            pf = jnp.zeros((ex.MR, half_), F32)
            for b in range(ex.B):
                pf = jnp.where(rows_ == b, pos_ref[b].astype(F32), pf)
            ang_ = pf * inv_
            scr["rope"][0, :, :half_] = rnd(jnp.cos(ang_))
            scr["rope"][1, :, :half_] = rnd(jnp.sin(ang_))

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ap: GQAParams = ctx.block_params(self.name)
        cx = GQAEmit.build(ctx, ap, l, self.name)
        ex, scr, sc, stream, a = cx.ex, cx.scr, cx.sc, cx.stream, cx.a
        VER = ex.VER
        nh, HD, H, SPLIT, LAZY_ATT = ap.nh, ap.HD, ap.H, ap.SPLIT, ap.LAZY_ATT
        kv = make_kv_ring(cx)
        with stream.segment(lazy=LAZY_ATT):
            ybuf, kn, half, koff, rcs = att_q(cx, slot)
            qx = q_exchange_start(cx)
            if cache_exec(ex).FUSED:
                with sc("att_kvrd"):
                    fused_io().gqa_fused_kvrd_issue(ctx, a)
            kv_out = kv_new(cx, ybuf, kn, half, koff, rcs)
            if SPLIT or VER:
                q_gather_finish(cx, qx)
            att_core(cx, kv)
            if SPLIT:
                split_merge(cx)
            att_new(cx)
            with sc("att_out"):
                if cache_exec(ex).FUSED and not SPLIT:
                    fused_io().gqa_fused_kv_rmw(ctx, ap, a)
                elif not cache_exec(ex).FUSED:
                    for d in kv_out:
                        d.wait()
                NO = nh * HD
                scr["xb2"][:, :NO] = scr["mix"][:, :NO].astype(BF16)
                yo = stream.dmatmul("att_out", scr["xb2"], H)

        return yo * ctx.sm(slot, "out_s")

    def epilogue(self, ctx: Any) -> None:
        if cache_exec(ctx.ex).FUSED:
            fused_io().gqa_fused_epilogue(ctx)

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Scatter `knew` and `vnew` into `kc` and `vc` inside `shard_map`."""
        if cache_exec(ex).FUSED:
            if POOL_STATE_NAME in raw_outputs:
                return {POOL_STATE_NAME: raw_outputs[POOL_STATE_NAME]}
            return {}
        knew, vnew = raw_outputs["knew"], raw_outputs["vnew"]
        if cache_exec(ex).paged_kv or "kc" not in states:
            return {"knew": knew, "vnew": vnew}
        kc, vc = states["kc"], states["vc"]
        if self.tunables(ex).kv_scatter_vec_ns < ex.NS:
            bi = jnp.arange(ex.B) if ex.T == 1 else jnp.arange(ex.B) // ex.T
            kw = dict(
                indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds"
            )
            kc = kc.at[:, bi, pos].set(knew[:, : ex.B].astype(BF16), **kw)
            vc = vc.at[:, bi, pos].set(vnew[:, : ex.B].astype(BF16), **kw)
            return {"kc": kc, "vc": vc}
        for s_idx in range(ex.NS):
            b0 = s_idx * ex.T
            upd = (0, s_idx, pos[b0], 0)
            kc = lax.dynamic_update_slice(
                kc, knew[:, None, b0 : b0 + ex.T].astype(BF16), upd
            )
            vc = lax.dynamic_update_slice(
                vc, vnew[:, None, b0 : b0 + ex.T].astype(BF16), upd
            )
        return {"kc": kc, "vc": vc}
