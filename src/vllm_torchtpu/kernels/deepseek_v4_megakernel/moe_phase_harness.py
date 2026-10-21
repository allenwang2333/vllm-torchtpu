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
"""Standalone harness of step_phases.moe_phase: one pallas_call per rank that
runs the phase for layers [l0, l1) in a fori, as the step skeleton does."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from .allreduce import COLLECTIVE_ID, AllReduce, AllReduceSpec, barrier
from .moe_experts import BF16
from .step_phases import (
    LayerRefs,
    MoEInputs,
    MoEWeights,
    PoolLease,
    moe_phase,
    moe_scratch_shapes,
)


def phase_harness(
    mesh,
    geom,
    l0,
    l1,
    *,
    interpret=None,
    shared_ar=False,
    leased=False,
    stacked=False,
    placed=False,
    copies=False,
    layer_logits=False,
):
    """jit(shard_map(...)) taking (me i32 [R, 1] sharded, x bf16 [T, H],
    logits f32 [T, E], partial f32 [R * T, H] sharded (each rank's seed),
    hash_ids i32 [num_hash_layers, T * 8], row_valid i32 [T], bias f32 [L, E],
    then per layer w13, w13_scale, w2, w2_scale (EP-sharded on experts))
    -> out bf16 [R * T, H]: each rank's `out` after layer l1 - 1.

    shared_ar: as the skeleton, the phase uses the caller's AllReduce on slot
    1 while a stand-in for the attention all-reduce runs on slot 0 first.
    leased: the expert ring lives in a u8 pool of geom.lease_bytes (Mosaic
    only; interpret has no raw VMEM views).
    stacked: the weights come as the four stacked arrays [L, E, ...] (EP-
    sharded on axis 1) instead of 4 * L per-layer arrays.
    placed / copies: two more operands after bias, the placement tables of
    moe_phase.placement_tables (held, and with copies the copy table), each
    int32 [R, L, slots] sharded on ranks; the weights are then in slot order.
    layer_logits: logits are f32 [L, T, E] (HBM), one routing per layer."""
    (ax,) = mesh.axis_names
    L = geom.n_layers
    n_w = 4 if stacked else 4 * L
    T, H = geom.T, geom.hidden
    scratch = moe_scratch_shapes(geom, leased, own_ar=not shared_ar)
    pool_rows = geom.lease_bytes // 128
    if layer_logits:
        scratch = scratch + [
            pltpu.VMEM((T, geom.n_experts), jnp.float32),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    if leased:
        scratch = [pltpu.VMEM((pool_rows, 128), jnp.uint8)] + scratch
    ar_spec = AllReduceSpec(geom.R, T, H, "hrsag", slots=2, group=min(8, geom.R))
    ar_names = list(ar_spec.scratch()) if shared_ar else []
    scratch = scratch + [ar_spec.scratch()[k] for k in ar_names]

    n_pl = int(placed) + int(copies)

    def kernel(me_ref, x, logits, partial, hash_ids, row_valid, bias, *refs):
        pl_refs, refs = refs[:n_pl], refs[n_pl:]
        held = pl_refs[0].at[0] if placed else None
        copy = pl_refs[1].at[0] if copies else None
        w = refs[:n_w]
        out = refs[n_w]
        scr = refs[n_w + 1 :]
        pool, scr = (scr[0], scr[1:]) if leased else (None, scr)
        scr, ar_scr = scr[: len(scr) - len(ar_names)], scr[len(scr) - len(ar_names) :]
        if layer_logits:  # declared after the phase's scratch, before the AR's
            scr, (lg_vm, lg_sem) = scr[:-2], scr[-2:]
        me = me_ref[0, 0]
        ar = AllReduce(ar_spec, dict(zip(ar_names, ar_scr)), me) if shared_ar else None
        barrier(geom.R, me)
        roles = w if stacked else [LayerRefs(w[j::4]) for j in range(4)]
        weights = MoEWeights(*roles, bias)

        def body(layer, c):
            lg = logits
            if layer_logits:
                cp = pltpu.make_async_copy(logits.at[layer], lg_vm, lg_sem.at[0])
                cp.start()
                cp.wait()
                lg = lg_vm
            inputs = MoEInputs(x, lg, partial, hash_ids, row_valid)
            if shared_ar:  # stand-in for the skeleton's attention all-reduce
                ar(partial[...], 0)
                ar.wait_sends(0)
            moe_phase(
                geom,
                layer,
                layer < geom.num_hash_layers,
                me,
                inputs,
                weights,
                out,
                PoolLease(pool, 0, pool_rows),
                scr,
                ar,
                1 if shared_ar else None,
                expert_held=held,
                expert_copy=copy,
            )
            return c

        lax.fori_loop(l0, l1, body, 0)

    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    sm = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((T, H), BF16),
        in_specs=[sm, vm, hbm if layer_logits else vm, vm, sm, sm, hbm]
        + [sm] * n_pl
        + [hbm] * n_w,
        out_specs=vm,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=62 * 2**20,
            disable_bounds_checks=True,
        ),
        interpret=interpret if interpret is not None else False,
        name="dsv4_moe_phase",
    )
    specs = (
        (P(ax), P(), P(), P(ax), P(), P(), P())
        + (P(ax),) * n_pl
        + (P(None, ax) if stacked else P(ax),) * n_w
    )
    return jax.jit(
        jax.shard_map(call, mesh=mesh, in_specs=specs, out_specs=P(ax), check_vma=False)
    )
