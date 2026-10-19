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
"""Unit tests for `MoEBlock` declarations and Pallas `interpret=True` execution."""

from __future__ import annotations

import ast
import pathlib

import jax
import jax.numpy as jnp
from conftest import FakeGeometry, FakeMixerBlock, MiniCtx
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import blocks, comm, planner, stream, vmem
from vllm_torchtpu.megakernel.blocks import MoEBlock
from vllm_torchtpu.megakernel.ops import BF16, F32, I32
from vllm_torchtpu.megakernel.spec import (
    TINY_TILES,
    ExecSpec,
    LayerType,
    ModelSpec,
)

BLOCKS_DIR = pathlib.Path(blocks.__file__).resolve().parent

# Row-count names that must never be compared to a literal inside `blocks/`.
_ROW_NAMES = {"B", "NS", "MR", "MB", "M", "CB"}
# Remaining literal row-count comparisons are hardware / layout limits, not
# per-bucket tuning (each one with its reason).
_ALLOWED = {
    ("moe.py", "MR <= 128"): "lane top-k keeps all MR rows in one 128-lane vreg",
    ("gqa.py", "ex.NS <= 64"): "GQ packs 128 // NS lanes per sequence (m | l) field",
}


def _row_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id in _ROW_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _ROW_NAMES:
        return node.attr
    return None


def _is_num(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def test_blocks_have_no_row_count_literal_comparisons():
    """Blocks read per-bucket tuning from `Tiles`, never `B <= 40`-style literals."""
    assert (BLOCKS_DIR / "moe.py").is_file()
    found = []
    for path in sorted(BLOCKS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            ops = [node.left, *node.comparators]
            for a, b in zip(ops, ops[1:]):
                pair = (_row_name(a) and _is_num(b)) or (_row_name(b) and _is_num(a))
                if pair:
                    key = (path.name, ast.unparse(node))
                    if key not in _ALLOWED:
                        found.append(f"{path.name}:{node.lineno}: {key[1]}")
    assert not found, "row-count literal comparisons in blocks/:\n" + "\n".join(found)


def test_moe_block_declarations_and_emit_in_pallas_interpret(fake_ex: ExecSpec):
    """Verify `MoEBlock` declarations drive `planner`/`vmem` and `emit` in Pallas."""
    g_decl = FakeGeometry(L=2, att_every=1, H=256)
    moe_decl = MoEBlock()
    ms_decl = ModelSpec(
        "moe_decl_test",
        config=g_decl,
        layer_type=LayerType("lt", slots=(FakeMixerBlock(), moe_decl)),
    )

    p_decl = planner.plan(ms_decl, fake_ex)
    assert p_decl.static_kinds == ("att_in", "att_out", "router", "sh13", "sh2")
    assert p_decl.dynamic_kinds == ("e13", "e2")
    assert p_decl.w_names == (
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

    cp_decl, scr_decl = vmem.build_scratch(ms_decl, fake_ex, p_decl)
    assert (
        "wn" in scr_decl
        and "elist" in scr_decl
        and "es" in scr_decl
        and "st" in scr_decl
    )
    assert cp_decl.block("moe").EW == g_decl.E // fake_ex.R

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
    moe = MoEBlock()
    ms = ModelSpec("moe_exec", config=g, layer_type=LayerType("lt", slots=(moe,)))
    cp, scratch = vmem.build_scratch(ms, ex)
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    assert moe.input_norm == "post_norm"

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_o = len(p.w_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        O = dict(zip(p.o_names, rest[n_w : n_w + n_o]))  # noqa: E741
        scr = dict(zip(s_names, rest[n_w + n_o :]))
        me = jnp.int32(0)
        ring = stream.TwoRing(cp.stream, W, scr, me, interpret=True)
        ar = comm.AllReduceRing(cp.comm, scr, me)
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp,
            pos_ref=pos_ref,
            W=W,
            ST={},
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
        moe.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        moe.init_buffers(ctx)

        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).wait()
        ring.begin_layer(jnp.int32(0))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        ar.add_residual(ar.allreduce(0, part_moe, jnp.bool_(True)))
        O["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            O["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, g.H), BF16)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
    ] + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names]
    out_specs = [pl.BlockSpec(memory_space=pltpu.VMEM) for _ in p.o_names]
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
    (hid_out,) = jax.jit(call)(pos, hid, *w_args)
    assert hid_out.shape == (ex.M, g.H)
    assert jnp.all(jnp.isfinite(hid_out))
