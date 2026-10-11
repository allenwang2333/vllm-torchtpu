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
"""Generic megakernel builder (`KernelContext`, `build`, `make_step`)."""

from __future__ import annotations

import contextlib
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.megakernel.blocks.gqa import GQABlock
from vllm_torchtpu.kernels.megakernel.blocks.moe import MoEBlock
from vllm_torchtpu.kernels.megakernel.comm import AllReduceRing
from vllm_torchtpu.kernels.megakernel.ops import BF16, F32, FP8, I32, rms_zc
from vllm_torchtpu.kernels.megakernel.planner import (
    kernel_out_shapes,
    plan,
    weight_shapes,
)
from vllm_torchtpu.kernels.megakernel.spec import (
    ModelSpec,
    SwitchSlot,
    Tiles,
    default_tiles,
)
from vllm_torchtpu.kernels.megakernel.stream import TwoRing
from vllm_torchtpu.kernels.megakernel.vmem import DerivedParams, build_scratch

COLLECTIVE_ID = 0
ABLATIONS = ("experts", "ar", "attn", "gdn", "dense", "kv")
W_NAMES = (
    "lin_in",
    "lin_out",
    "att_in",
    "att_out",
    "router",
    "sh13",
    "sh2",
    "e13",
    "e2",
    "es",
    "smalls",
    "final_norm",
)
S_NAMES = ("conv", "rec", "kc", "vc")
O_NAMES = ("hidden", "conv", "rec", "knew", "vnew")


class KernelContext:
    """Explicit trace-time context passed to composable blocks inside the Pallas kernel."""

    def __init__(
        self,
        *,
        c: Any,
        dp: DerivedParams,
        tiles: Tiles,
        pos_ref: Any,
        hid_ref: Any,
        W: dict[str, Any],
        ST: dict[str, Any],
        O: dict[str, Any],
        out_ref: Any,
        raw_refs: Any,
        rb_hbm: Any,
        mr_ref: Any,
        scr: dict[str, Any],
        me: Any,
        gi: Any,
        lane_e8: Any,
        lane128: Any,
        row8: Any,
        row_m: Any,
        live: Any,
        EM: int,
        FLT_MIN: float,
        stream: TwoRing,
        comm: AllReduceRing,
        interpret: bool,
        forced_routing: bool,
        row_mask: bool,
        skip: frozenset[str],
        sc: Any,
    ):
        self.c = c
        self.dp = dp
        self.tiles = tiles
        self.pos_ref = pos_ref
        self.hid_ref = hid_ref
        self.W = W
        self.ST = ST
        self.O = O
        self.out_ref = out_ref
        self.raw_refs = raw_refs
        self.rb_hbm = rb_hbm
        self.mr_ref = mr_ref
        self.scr = scr
        self.me = me
        self.gi = gi
        self.lane_e8 = lane_e8
        self.lane128 = lane128
        self.row8 = row8
        self.row_m = row_m
        self.live = live
        self.EM = EM
        self.FLT_MIN = FLT_MIN
        self.stream = stream
        self.comm = comm
        self.interpret = interpret
        self.forced_routing = forced_routing
        self.row_mask = row_mask
        self.skip = skip
        self.sc = sc

    def sm(self, slot, name: str, n: int | None = None):
        o, w = self.dp.lay[name]
        return self.scr["sm"][slot, :, o : o + (w if n is None else n)]

    def is_seq_active(self, s):
        c, dp, ST, pos_ref = self.c, self.dp, self.ST, self.pos_ref
        if c.act_mode == "ref":
            return ST["active_seqs"][s] != 0
        if c.act_mode == "from_ckpt":
            return (
                (ST["ckpt_indices"][0, s, 0] > 0)
                if dp.ckpt_is_3d
                else (ST["ckpt_indices"][s, 0] > 0)
            )
        if c.act_mode == "pos":
            return pos_ref[s * dp.T] >= 0
        return jnp.bool_(True)

    def wait_kv_wr(self, s):
        scr, raw_refs = self.scr, self.raw_refs
        wr_pend = scr["kv_wr_pend"]
        pend = wr_pend[s]

        @pl.when(pend != 0)
        def _():
            pltpu.make_async_copy(
                scr["kv_scat_page"].at[s], raw_refs[0].at[0], scr["kv_wr_sem"].at[s]
            ).wait()

        wr_pend[s] = jnp.int32(0)

    def wait_gdn_out(self, slot_o):
        dp, scr, raw_refs = self.dp, self.scr, self.raw_refs
        T = dp.T
        out_page = scr["gdn_out_page"]
        out_sem = scr["gdn_out_sem"]
        out_mode = scr["gdn_out_mode"]
        mode = out_mode[slot_o]

        @pl.when(mode == 1)
        def _():
            pltpu.make_async_copy(
                out_page.at[slot_o],
                raw_refs[0].at[pl.ds(0, 5 * T)],
                out_sem.at[slot_o, 0],
            ).wait()

        if T > 1:
            @pl.when(mode == 2)
            def _():
                for t in range(T):
                    pltpu.make_async_copy(
                        out_page.at[slot_o, pl.ds(t * 5, 5)],
                        raw_refs[0].at[pl.ds(0, 5)],
                        out_sem.at[slot_o, t],
                    ).wait()

        out_mode[slot_o] = jnp.int32(0)

    def batch_to_i32_cols(self, tok_mr_hd):
        c, dp = self.c, self.dp
        rows_bf16 = tok_mr_hd.astype(BF16)
        if dp.MR < c.page_size:
            rows_bf16 = jnp.pad(rows_bf16, ((0, c.page_size - dp.MR), (0, 0)))
        cols_fp8 = jnp.swapaxes(rows_bf16, 0, 1).astype(FP8)
        return pltpu.bitcast(cols_fp8, I32)


def _extract_blocks(model_or_c: Any) -> tuple[Any, Any, Any, Any]:
    """Return `(c, gqa_block, gdn_block, moe_block)` from a `ModelSpec` or `MKConfig`."""
    if isinstance(model_or_c, ModelSpec):
        c = model_or_c.config
        gqa_blk = None
        gdn_blk = None
        moe_blk = None
        for slot in model_or_c.layer_type.slots:
            if isinstance(slot, SwitchSlot):
                for b in slot.branches:
                    if b.name == "gqa":
                        gqa_blk = b
                    elif b.name == "gdn":
                        gdn_blk = b
            elif slot.name == "gqa":
                gqa_blk = slot
            elif slot.name == "gdn":
                gdn_blk = slot
            elif slot.name == "moe":
                moe_blk = slot
        return c, gqa_blk or GQABlock(), gdn_blk, moe_blk or MoEBlock()
    return model_or_c, GQABlock(), None, MoEBlock()


def build(
    model_or_c: Any,
    tiles: Tiles | None = None,
    *,
    interpret: bool = False,
    forced_routing: bool = False,
    skip=frozenset(),
    scopes: int | bool = False,
    row_mask: bool = False,
):
    """Build the Pallas megakernel callable `(call, info)` from a `ModelSpec` or config `c`."""
    c, gqa_block, gdn_block, moe_block = _extract_blocks(model_or_c)
    c.check()
    skip = frozenset(skip)
    assert set(skip) <= set(ABLATIONS), skip
    tiles = default_tiles(c.B) if tiles is None else tiles

    def sc(name: str, level: int = 1):
        return jax.named_scope(name) if int(scopes) >= level else contextlib.nullcontext()

    dp, scratch = build_scratch(
        c,
        tiles,
        interpret=bool(interpret),
        forced_routing=forced_routing,
        row_mask=row_mask,
    )
    p = plan(c, tiles, model_or_c if isinstance(model_or_c, ModelSpec) else None)
    s_names = list(scratch)

    if dp.FUSED:
        st_names = ("raw", "page_table", "ckpt_indices", "read_offsets") + (
            ("active_seqs",) if c.act_mode == "ref" else ()
        )
        o_names = ("hidden", "raw") if bool(interpret) else ("hidden",)
    elif c.paged_kv:
        st_names = ("conv", "rec", "raw", "page_table")
        o_names = O_NAMES
    else:
        st_names = S_NAMES
        o_names = O_NAMES
    n_w, n_s = len(W_NAMES), len(st_names)

    def kernel(*refs):
        pos_ref, hid_ref = refs[:2]
        W = dict(zip(W_NAMES, refs[2 : 2 + n_w]))
        ST = dict(zip(st_names, refs[2 + n_w : 2 + n_w + n_s]))
        k_ = 2 + n_w + n_s
        rb_hbm = None
        if forced_routing:
            rb_hbm = refs[k_]
            k_ += 1
        mr_ref = None
        if row_mask:
            mr_ref = refs[k_]
            k_ += 1
        O = dict(zip(o_names, refs[k_ : k_ + len(o_names)]))
        out_ref = O["hidden"]
        raw_refs = (
            (O["raw"] if bool(interpret) else ST["raw"])
            if dp.FUSED
            else (ST["raw"] if c.paged_kv else None)
        )
        scr = dict(zip(s_names, refs[k_ + len(o_names) :]))
        me = lax.axis_index("tp").astype(I32)
        gi = lax.rem(me, dp.G) if dp.SPLIT else jnp.int32(0)
        lane_e8 = lax.broadcasted_iota(I32, (dp.MR, dp.E), 1)
        lane128 = lax.broadcasted_iota(I32, (dp.MR, 128), 1)
        row8 = lax.broadcasted_iota(I32, (dp.MR, 1), 0)
        row_m = lax.broadcasted_iota(I32, (dp.MB, 1), 0)
        live = row_m < dp.B
        EM = (1 << max(1, (dp.E - 1).bit_length())) - 1
        FLT_MIN = float(jnp.finfo(F32).tiny)

        stream = TwoRing(
            c, dp, W, scr, me, interpret=bool(interpret), skip=skip, sc=sc
        )
        comm = AllReduceRing(dp, scr, me, skip=skip, sc=sc)
        ctx = KernelContext(
            c=c,
            dp=dp,
            tiles=tiles,
            pos_ref=pos_ref,
            hid_ref=hid_ref,
            W=W,
            ST=ST,
            O=O,
            out_ref=out_ref,
            raw_refs=raw_refs,
            rb_hbm=rb_hbm,
            mr_ref=mr_ref,
            scr=scr,
            me=me,
            gi=gi,
            lane_e8=lane_e8,
            lane128=lane128,
            row8=row8,
            row_m=row_m,
            live=live,
            EM=EM,
            FLT_MIN=FLT_MIN,
            stream=stream,
            comm=comm,
            interpret=bool(interpret),
            forced_routing=forced_routing,
            row_mask=row_mask,
            skip=skip,
            sc=sc,
        )

        # ------------------------------ layer loop ------------------------------
        def layer(l, carry):
            slot = l % 2
            with sc("norm_in"):
                pltpu.make_async_copy(
                    W["smalls"].at[0], scr["sm"].at[slot], scr["sm_sem"].at[slot]
                ).wait()
                if dp.ECOPY:
                    o_ec = dp.lay["ecopy"][0]
                    pltpu.make_async_copy(
                        W["smalls"].at[0, :, pl.ds(o_ec, dp.E)],
                        scr["ecopy"].at[slot],
                        scr["ec_sem"].at[slot],
                    ).wait()

                @pl.when(l + 1 < dp.L)
                def _():
                    pltpu.make_async_copy(
                        W["smalls"].at[l + 1],
                        scr["sm"].at[1 - slot],
                        scr["sm_sem"].at[1 - slot],
                    ).start()
                    if dp.ECOPY:
                        o_ec = dp.lay["ecopy"][0]
                        pltpu.make_async_copy(
                            W["smalls"].at[l + 1, :, pl.ds(o_ec, dp.E)],
                            scr["ecopy"].at[1 - slot],
                            scr["ec_sem"].at[1 - slot],
                        ).start()

                is_att = l % dp.AE == dp.AE - 1
                g0 = (l // dp.AE) * dp.t_block + (l % dp.AE) * dp.t_gdn
                scr["xb"][...] = rms_zc(
                    scr["res"][...], ctx.sm(slot, "in_norm"), c.eps
                ).astype(BF16)

            first = l == 0

            def att_branch():
                comm.add_residual(
                    comm.allreduce(0, gqa_block.emit(ctx, l, slot, g0), first)
                )

            def gdn_branch():
                assert gdn_block is not None, "GDNBlock required when att_every > 1"
                comm.add_residual(
                    comm.allreduce(0, gdn_block.emit(ctx, l, slot, g0), first)
                )

            if dp.AE == 1:
                att_branch()
            else:
                lax.cond(is_att, att_branch, gdn_branch)
            comm.add_residual(
                comm.allreduce(1, moe_block.emit(ctx, l, slot, g0, is_att), first)
            )
            return carry

        # ------------------------------ step prologue ------------------------------
        scr["hitv"][...] = jnp.zeros((dp.MR, 128), I32)
        zero_seen = pltpu.make_async_copy(
            scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"]
        )
        zero_seen.start()
        for i in range(dp.NBD):
            stream.fetch(jnp.int32(i))
        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).start()
        if dp.ECOPY:
            o_ec = dp.lay["ecopy"][0]
            pltpu.make_async_copy(
                W["smalls"].at[0, :, pl.ds(o_ec, dp.E)],
                scr["ecopy"].at[0],
                scr["ec_sem"].at[0],
            ).start()
        if forced_routing:
            pltpu.make_async_copy(
                rb_hbm.at[0], scr["rb"].at[0], scr["rb_sem"].at[0]
            ).start()

        gqa_block.prologue(ctx)
        if gdn_block is not None:
            gdn_block.prologue(ctx)

        if dp.R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, dp.R):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, dp.R - 1)

        scr["res"][...] = jnp.where(live, hid_ref[: dp.MB].astype(F32), 0.0)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        gqa_block.init_buffers(ctx)
        if gdn_block is not None and hasattr(gdn_block, "init_buffers"):
            gdn_block.init_buffers(ctx)

        zero_seen.wait()
        lax.fori_loop(0, dp.L, layer, 0)

        # ------------------------------ step epilogue ------------------------------
        comm.drain_sends()
        if dp.FUSED:
            if gdn_block is not None and c.NGDN > 0:
                gdn_block.epilogue(ctx)
            gqa_block.epilogue(ctx)
        elif gdn_block is not None and c.NGDN > 0:
            gdn_block.epilogue(ctx)

        cp = pltpu.make_async_copy(W["final_norm"], scr["fn"], scr["misc_sem"])
        cp.start()
        cp.wait()
        out_ref[: dp.MB] = rms_zc(scr["res"][...], scr["fn"][...], c.eps).astype(BF16)
        if dp.MB < dp.M:
            out_ref[dp.MB :] = jnp.zeros((dp.M - dp.MB, dp.H), BF16)

    oshapes = kernel_out_shapes(c)
    w_specs = [
        tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.L))
        if (not c.stack_e and k in ("e13", "e2"))
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in W_NAMES
    ]
    s_specs = [
        tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.NATT))
        if k == "raw"
        else pl.BlockSpec(memory_space=pltpu.SMEM)
        if k in ("page_table", "ckpt_indices", "read_offsets", "active_seqs")
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in st_names
    ]
    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)]
        + w_specs
        + s_specs
        + ([pl.BlockSpec(memory_space=pltpu.HBM)] if forced_routing else [])
        + ([pl.BlockSpec(memory_space=pltpu.SMEM)] if row_mask else [])
    )
    n_w_leaves = sum(
        c.L if (not c.stack_e and k in ("e13", "e2")) else 1 for k in W_NAMES
    )
    out_shape = [
        tuple(jax.ShapeDtypeStruct(*s) for s in oshapes[k])
        if k == "raw"
        else jax.ShapeDtypeStruct(*oshapes[k])
        for k in o_names
    ]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
        if k == "hidden"
        else tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.NATT))
        if k == "raw"
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in o_names
    ]
    if dp.FUSED:
        aliases = (
            {2 + n_w_leaves + a: 1 + a for a in range(c.NATT)}
            if bool(interpret)
            else {}
        )
    else:
        aliases = (
            {2 + n_w_leaves + st_names.index(k): o_names.index(k) for k in ("conv", "rec")}
            if tiles.inplace
            else {}
        )
    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID if dp.R > 1 else None,
            vmem_limit_bytes=tiles.vmem_mib * 1024 * 1024,
            disable_bounds_checks=not tiles.bounds_checks,
        ),
        interpret=(
            interpret
            if isinstance(interpret, pltpu.InterpretParams)
            else pltpu.InterpretParams()
            if interpret
            else False
        ),
        name="qwen3_5_decode_megakernel",
    )
    vmem = 0
    for k in s_names:
        v = scratch[k]
        if hasattr(v, "shape") and "VMEM" in str(getattr(v, "memory_space", "")).upper():
            vmem += math.prod(v.shape) * jnp.dtype(v.dtype).itemsize
    info = {
        "plan": p,
        "T_TOTAL": dp.T_TOTAL,
        "vmem_scratch_bytes": vmem,
        "s_names": s_names,
        "wshape": weight_shapes(c, tiles),
        "NSS": dp.NSS,
        "TE": dp.TE,
    }
    return call, info


def fold(x):
    """`[R, N0, ...] -> [R * N0, ...]`: the rank-folded layout `make_step` takes for states."""
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def unfold(x, R: int):
    return x.reshape((R, x.shape[0] // R) + x.shape[1:])


def make_step(
    mesh,
    model_or_c: Any,
    tiles: Tiles | None = None,
    *,
    interpret: bool = False,
    forced_routing: bool = False,
    skip=frozenset(),
    scopes: int | bool = False,
):
    """`shard_map` decode/verify step wrapping `build(...)` + `direct_hbm` KV cache scatter."""
    c, _, _, _ = _extract_blocks(model_or_c)
    call, info = build(
        model_or_c,
        tiles,
        interpret=interpret,
        forced_routing=forced_routing,
        skip=skip,
        scopes=scopes,
    )

    def local(pos, hidden, weights, states, *rb):
        w = [
            tuple(x[0] for x in weights[k])
            if isinstance(weights[k], tuple)
            else weights[k][0]
            for k in W_NAMES
        ]
        s = [states[k] for k in S_NAMES]
        hid, conv, rec, knew, vnew = call(pos, hidden, *w, *s, *rb)
        kc, vc = states["kc"], states["vc"]
        if c.NS > 16:
            bi = jnp.arange(c.B) if c.T == 1 else jnp.arange(c.B) // c.T
            kw = dict(
                indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds"
            )
            kc = kc.at[:, bi, pos].set(knew[:, : c.B].astype(BF16), **kw)
            vc = vc.at[:, bi, pos].set(vnew[:, : c.B].astype(BF16), **kw)
            return hid, {"conv": conv, "rec": rec, "kc": kc, "vc": vc}
        for s_idx in range(c.NS):
            b0 = s_idx * c.T
            upd = (0, s_idx, pos[b0], 0)
            kc = lax.dynamic_update_slice(
                kc, knew[:, None, b0 : b0 + c.T].astype(BF16), upd
            )
            vc = lax.dynamic_update_slice(
                vc, vnew[:, None, b0 : b0 + c.T].astype(BF16), upd
            )
        return hid, {"conv": conv, "rec": rec, "kc": kc, "vc": vc}

    def step(pos, hidden, weights, states, *rb):
        ws = {
            k: tuple(P("tp") for _ in weights[k])
            if isinstance(weights[k], tuple)
            else P("tp")
            for k in weights
        }
        ss = {k: P("tp") for k in states}
        in_specs = (P(), P("tp"), ws, ss) + ((P("tp"),) if forced_routing else ())
        return jax.shard_map(
            local, mesh=mesh, in_specs=in_specs, out_specs=(P("tp"), ss), check_vma=False
        )(pos, hidden, weights, states, *rb)

    return step, info

