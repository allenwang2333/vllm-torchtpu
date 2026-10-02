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
"""One-shot (direct-to-every-peer) Pallas TPU collectives for latency-bound payloads.

XLA's small collectives on 16 dies cost 12-15 µs against a 4 µs roofline alpha (multi-step
schedules). Here every device writes its data straight into every peer's buffer with one remote
DMA each, after one barrier, then waits for the N-1 incoming DMAs: one network step, N-1 sends.
AR / RS reduce locally after the exchange (N x the bytes of a ring; only for latency-bound sizes).

  all_gather     x [r, c] -> [N*r, c]     dst = peer's out[me]
  all_to_all     x [N*r, c] -> [N*r, c]   slice j -> peer j's out[me]
  reduce_scatter x [N*r, c] -> [r, c]     slice j -> peer j's scratch[me], then sum over slots
  all_reduce     x [r, c] -> [r, c]       x -> every peer's scratch[me], then sum over slots

Shapes follow the torch bridge (vllm_torchtpu sharded_jax_op): the jax fn sees GLOBAL arrays
sharded on dim 0 over the 1-axis mesh; each kernel instance sees its rank's block.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

AXIS = "oneshot"
COLLECTIVE_ID = {"all_gather": 11, "all_to_all": 12, "reduce_scatter": 13, "all_reduce": 14}
MESH = pl.DeviceIdType.MESH


def _peer(me, k, n):
    return (lax.rem(me + k, n),)


def _barrier(me, n):
    sem = pltpu.get_barrier_semaphore()
    for k in range(1, n):
        pl.semaphore_signal(sem, 1, device_id=_peer(me, k, n), device_id_type=MESH)
    pl.semaphore_wait(sem, n - 1)


def _exchange(me, n, src_of, dst, send_sem, recv_sem, local_sem):
    """Send src_of(peer_index) to every peer's dst.at[me]; local copy for me; wait for all."""
    local = pltpu.make_async_copy(src_of(me), dst.at[me], local_sem)
    local.start()
    cps = []
    for k in range(1, n):
        p = lax.rem(me + k, n)
        c = pltpu.make_async_remote_copy(src_of(p), dst.at[me], send_sem, recv_sem,
                                         device_id=(p,), device_id_type=MESH)
        c.start()
        cps.append(c)
    for c in cps:
        c.wait_send()
    for c in cps:          # N-1 incoming DMAs of the same size land on recv_sem
        c.wait_recv()
    local.wait()


def _ag_kernel(x_ref, o_ref, send_sem, recv_sem, local_sem, *, n):
    me = lax.axis_index(AXIS)
    _barrier(me, n)
    _exchange(me, n, lambda p: x_ref, o_ref, send_sem, recv_sem, local_sem)


def _a2a_kernel(x_ref, o_ref, send_sem, recv_sem, local_sem, *, n):
    me = lax.axis_index(AXIS)
    _barrier(me, n)
    _exchange(me, n, lambda p: x_ref.at[p], o_ref, send_sem, recv_sem, local_sem)


def _reduce_kernel(x_ref, o_ref, slots, send_sem, recv_sem, local_sem, *, n, scatter):
    me = lax.axis_index(AXIS)
    _barrier(me, n)
    _exchange(me, n, (lambda p: x_ref.at[p]) if scatter else (lambda p: x_ref),
              slots, send_sem, recv_sem, local_sem)
    o_ref[...] = jnp.sum(slots[...].astype(jnp.float32), axis=0).astype(o_ref.dtype)


INTERPRET = None      # tests set pltpu.InterpretParams(...) to run on emulated CPU devices


def _twoshot_ar_kernel(x_ref, o_ref, slots, mine, s1, r1, l1, s2, r2, l2, *, n):
    """AR = one-shot reduce-scatter (slice p -> peer p's slots[me], sum) then one-shot gather of
    this rank's reduced slice into every peer's out[me]: 2 exchanges, 2x the bytes of a ring
    instead of n x. x / out are [n, L]."""
    me = lax.axis_index(AXIS)
    _barrier(me, n)
    _exchange(me, n, lambda p: x_ref.at[p], slots, s1, r1, l1)
    mine[...] = jnp.sum(slots[...].astype(jnp.float32), axis=0).astype(mine.dtype)
    _exchange(me, n, lambda p: mine, o_ref, s2, r2, l2)


def _barrier_kernel(x_ref, o_ref, sem, *, n):
    """Floor probe: one all-peer barrier, then a local copy (output = input)."""
    me = lax.axis_index(AXIS)
    _barrier(me, n)
    c = pltpu.make_async_copy(x_ref, o_ref, sem)
    c.start()
    c.wait()


def _ping_kernel(x_ref, o_ref, s, r, *, n):
    """Floor probe: barrier, then ONE remote DMA to the ring neighbour (out = left neighbour's x)."""
    me = lax.axis_index(AXIS)
    _barrier(me, n)
    c = pltpu.make_async_remote_copy(x_ref, o_ref, s, r, device_id=(lax.rem(me + 1, n),),
                                     device_id_type=MESH)
    c.start()
    c.wait_send()
    c.wait_recv()


def _variant(variant: str, op: str, n: int, part):
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    dma = pltpu.SemaphoreType.DMA
    cp = pltpu.CompilerParams(collective_id=COLLECTIVE_ID["all_reduce"] + 10,
                              vmem_limit_bytes=min(100 << 20, 4 * part.nbytes + (16 << 20)))
    if variant == "twoshot":
        r, c = part.shape
        flat = r * c // n
        lanes = 128 if flat % 128 == 0 else flat      # leading dim untiled: .at[p] stays aligned
        x = part.reshape(n, flat // lanes, lanes)
        out = pl.pallas_call(functools.partial(_twoshot_ar_kernel, n=n),
                             out_shape=jax.ShapeDtypeStruct(x.shape, part.dtype),
                             in_specs=[hbm], out_specs=hbm,
                             scratch_shapes=[pltpu.VMEM(x.shape, part.dtype),
                                             pltpu.VMEM(x.shape[1:], part.dtype)] + [dma] * 6,
                             compiler_params=cp, name="oneshot_twoshot_ar",
                             interpret=INTERPRET)(x)
        return out.reshape(r, c)
    kern, sems = ((_barrier_kernel, [dma]) if variant == "barrier" else (_ping_kernel, [dma, dma]))
    return pl.pallas_call(functools.partial(kern, n=n),
                          out_shape=jax.ShapeDtypeStruct(part.shape, part.dtype),
                          in_specs=[hbm], out_specs=hbm, scratch_shapes=sems,
                          compiler_params=cp, name=f"oneshot_{variant}",
                          interpret=INTERPRET)(part)


def _local(op: str, n: int, part):
    sems = [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]
    cp = pltpu.CompilerParams(collective_id=COLLECTIVE_ID[op],
                              vmem_limit_bytes=min(100 << 20, 4 * part.nbytes + (16 << 20)))
    anyspec = pl.BlockSpec(memory_space=pltpu.HBM)
    if op == "all_gather":
        r, c = part.shape
        out = pl.pallas_call(functools.partial(_ag_kernel, n=n),
                             out_shape=jax.ShapeDtypeStruct((n, r, c), part.dtype),
                             in_specs=[anyspec], out_specs=anyspec,
                             scratch_shapes=sems, compiler_params=cp, name="oneshot_ag",
                             interpret=INTERPRET)(part)
        return out.reshape(n * r, c)
    if op == "all_to_all":
        rr, c = part.shape
        x = part.reshape(n, rr // n, c)
        out = pl.pallas_call(functools.partial(_a2a_kernel, n=n),
                             out_shape=jax.ShapeDtypeStruct(x.shape, part.dtype),
                             in_specs=[anyspec], out_specs=anyspec,
                             scratch_shapes=sems, compiler_params=cp, name="oneshot_a2a",
                             interpret=INTERPRET)(x)
        return out.reshape(rr, c)
    scatter = op == "reduce_scatter"
    x = part.reshape(n, part.shape[0] // n, part.shape[1]) if scatter else part
    r, c = x.shape[-2:]
    return pl.pallas_call(functools.partial(_reduce_kernel, n=n, scatter=scatter),
                          out_shape=jax.ShapeDtypeStruct((r, c), part.dtype),
                          in_specs=[anyspec],
                          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                          scratch_shapes=[pltpu.VMEM((n, r, c), part.dtype)] + sems,
                          compiler_params=cp,
                          name="oneshot_rs" if scatter else "oneshot_ar",
                          interpret=INTERPRET)(x)


def jax_fn(op: str, mesh, out_sharded: bool = False, variant: str | None = None):
    """The global-array fn for sharded_jax_op: input dim 0 sharded over the 1-axis mesh.

    The output is declared replicated (P()) so its global shape IS each rank's buffer: the
    torch_tpu lowering sizes a custom op's output from the export's global shape (a P(AXIS)
    output came back n x too large and a later view failed a TorchTPU shape check). RS / A2A
    outputs differ per rank; check_vma=False lets shard_map return each device its own block,
    as the Kimi ring all-gather does. `out_sharded` keeps P(AXIS) for the interpret test."""
    n = mesh.size

    def fn(x: jax.Array) -> jax.Array:
        body = (functools.partial(_variant, variant, op, n) if variant
                else functools.partial(_local, op, n))
        return jax.shard_map(body, mesh=mesh,
                             in_specs=P(AXIS, None),
                             out_specs=P(AXIS, None) if out_sharded else P(),
                             check_vma=False)(x)
    return fn


def torch_op(op: str, group, key: str, variant: str | None = None):
    """A torch custom op running the one-shot kernel over a vLLM GroupCoordinator's ranks.
    Call once per (op, group) outside tracing: building the mesh runs a cpu collective."""
    from vllm_torchtpu.distributed.mesh_utils import get_or_create_cp_mesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
    import torch
    mesh = get_or_create_cp_mesh(AXIS, group)
    n = mesh.size
    top = sharded_jax_op(f"pallas::oneshot_{variant or op}_{key}", jax_fn(op, mesh, variant=variant),
                         mesh=mesh,
                         input_partition_specs=(P(AXIS, None),),
                         output_partition_specs=P())

    def fake(x):         # this rank's shard, as the Kimi ring ops register it
        rows = x.shape[0] if variant else {"all_gather": x.shape[0] * n,
                                           "reduce_scatter": x.shape[0] // n}.get(op, x.shape[0])
        return torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)
    top.register_fake(fake)
    return top
