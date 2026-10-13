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
"""M1c unit tests for `GQABlock` and `MoEBlock` declarations and Pallas execution."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from conftest import FakeGeometry
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import comm, planner, stream, vmem
from vllm_torchtpu.megakernel.blocks import GQABlock, MoEBlock
from vllm_torchtpu.megakernel.ops import BF16, F32, I32, rmsnorm
from vllm_torchtpu.megakernel.spec import (
    TINY_TILES,
    ExecSpec,
    LayerType,
    ModelSpec,
)


def test_gqa_and_moe_block_declarations_feed_planner_and_vmem(fake_ex: ExecSpec):
    """Verify `GQABlock` and `MoEBlock` declarations drive `planner` and `vmem`."""
    g = FakeGeometry(L=2, att_every=1, H=256)
    gqa = GQABlock()
    moe = MoEBlock()
    ms = ModelSpec(
        "gqa_moe_test", config=g, layer_type=LayerType("lt", slots=(gqa, moe))
    )

    p = planner.plan(ms, fake_ex)
    assert p.static_kinds == ("att_in", "att_out", "router", "sh13", "sh2")
    assert p.dynamic_kinds == ("e13", "e2")
    assert p.w_names == (
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
    assert p.s_names == ("kc", "vc")
    assert p.o_names == ("hidden", "knew", "vnew")

    cp, scr = vmem.build_scratch(ms, fake_ex, p)
    # GQA scratch allocated, no GDN scratch allocated
    assert "kbuf" in scr and "vbuf" in scr and "wl_cnt" in scr
    assert "recb" not in scr and "sbuf" not in scr and "convb" not in scr
    # MoE scratch allocated
    assert "wn" in scr and "elist" in scr and "es" in scr and "st" in scr
    assert cp.block("gqa").NAI == (2 * (g.NH // fake_ex.R) + 2) * g.HD
    assert cp.block("moe").EW == g.E // fake_ex.R


def test_gqa_and_moe_emit_in_pallas_interpret():
    """Execute `GQABlock.emit` and `MoEBlock.emit` inside a Pallas kernel."""
    ex = ExecSpec(
        B=4, T=1, R=1, ctx=256, stack_e=True, tiles=TINY_TILES, interpret=True
    )
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        moe_tp=True,
    )
    gqa = GQABlock()
    moe = MoEBlock()
    ms = ModelSpec(
        "gqa_moe_exec", config=g, layer_type=LayerType("lt", slots=(gqa, moe))
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    ssh = planner.state_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    @dataclass
    class _MiniCtx:
        ex: ExecSpec
        core_plan: vmem.CorePlan
        pos_ref: Any
        W: dict[str, Any]
        ST: dict[str, Any]
        O: dict[str, Any]  # noqa: E741
        scr: dict[str, Any]
        me: Any
        stream: stream.TwoRing
        comm: comm.AllReduceRing
        rb_hbm: Any = None
        mr_ref: Any = None
        skip: frozenset[str] = frozenset()
        sc: Any = lambda *_: contextlib.nullcontext()
        norm_x: Any = None

        def block_params(self, name: str) -> Any:
            return self.core_plan.block(name)

        def sm(self, slot: Any, name: str, width: int | None = None) -> Any:
            o, w = self.core_plan.stream.lay[name]
            n = w if width is None else width
            return self.scr["sm"][slot, :, o : o + n]

        def apply_input_norm(self, norm_field: str | None, slot: Any) -> None:
            if norm_field is None:
                self.norm_x = None
                return
            x = rmsnorm(
                self.scr["res"][...],
                self.sm(slot, norm_field),
                g.eps,
                zero_centered=bool(g.zero_centered_norm),
            )
            self.scr["xb"][...] = x.astype(BF16)
            self.norm_x = x

    assert gqa.input_norm == "in_norm"
    assert moe.input_norm == "post_norm"

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_s, n_o = len(p.w_names), len(p.s_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        ST = dict(zip(p.s_names, rest[n_w : n_w + n_s]))
        O = dict(zip(p.o_names, rest[n_w + n_s : n_w + n_s + n_o]))  # noqa: E741
        scr = dict(zip(s_names, rest[n_w + n_s + n_o :]))
        me = jnp.int32(0)
        ring = stream.TwoRing(cp.stream, W, scr, me, interpret=True)
        ar = comm.AllReduceRing(cp.comm, scr, me)
        ctx = _MiniCtx(
            ex=ex,
            core_plan=cp,
            pos_ref=pos_ref,
            W=W,
            ST=ST,
            O=O,
            scr=scr,
            me=me,
            stream=ring,
            comm=ar,
        )

        for i in range(cp.stream.NBD):
            ring.fetch(jnp.int32(i))
        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).start()
        gqa.prologue(ctx)
        moe.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        gqa.init_buffers(ctx)
        moe.init_buffers(ctx)

        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).wait()
        ring.begin_layer(jnp.int32(0))
        ctx.apply_input_norm(gqa.input_norm, jnp.int32(0))
        part_att = gqa.emit(ctx, jnp.int32(0), jnp.int32(0))
        ar.add_residual(ar.allreduce(0, part_att, jnp.bool_(True)))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        ar.add_residual(ar.allreduce(1, part_moe, jnp.bool_(True)))
        O["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            O["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, g.H), BF16)

    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)]
        + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names]
        + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.s_names]
    )
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
        if k == "hidden"
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in p.o_names
    ]
    out_shape = [jax.ShapeDtypeStruct(*osh[k]) for k in p.o_names]

    call = pl.pallas_call(
        pallas_body,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )

    pos = jnp.array([4, 8, 12, 16], dtype=I32)
    hid = jnp.ones((ex.M, g.H), dtype=BF16) * 0.1
    w_args = [jnp.zeros(wsh[k][0], dtype=wsh[k][1]) for k in p.w_names]
    # Set unit scales in `es` and `smalls` so outputs are finite
    s_args = [jnp.zeros(ssh[k][0], dtype=ssh[k][1]) for k in p.s_names]
    outs = jax.jit(call)(pos, hid, *w_args, *s_args)
    hid_out, knew_out, vnew_out = outs
    assert hid_out.shape == (ex.M, g.H)
    assert jnp.all(jnp.isfinite(hid_out))
    assert knew_out.shape == osh["knew"][0]
    assert vnew_out.shape == osh["vnew"][0]

    # Verify host_update_states scatters knew/vnew into kc/vc
    updated = gqa.host_update_states(
        g,
        ex,
        pos,
        {"kc": s_args[0], "vc": s_args[1]},
        {"knew": knew_out, "vnew": vnew_out},
    )
    assert updated["kc"].shape == ssh["kc"][0]
    assert updated["vc"].shape == ssh["vc"][0]
