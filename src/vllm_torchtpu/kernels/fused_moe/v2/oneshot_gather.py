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
"""One-shot all-gather of a small int32 row over a 1-D device mesh.

Every shard writes its row straight into every peer's output with remote
DMAs, after one all-pairs barrier, instead of XLA's multi-step all-gather.
On a 16-core tpu7x 2x2x2 slice, an s32 [640] row per shard measured ~9 us
per gather against ~18 us for `lax.all_gather` (chained-call microbench).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows each shard sends: one whole (8, 128) int32 tile per lane block, so
# every DMA moves complete tiles.
ROWS = 8
# Its own collective id, so this barrier's semaphore is not the fused MoE
# kernel's (id 0): a shard that runs ahead into that kernel would otherwise
# signal the barrier a slower peer is still counting here.
COLLECTIVE_ID = 1


def _kernel(me_ref, x_hbm, o_hbm, sems, *, ep):
    me = me_ref[0]
    barrier = pltpu.get_barrier_semaphore()
    for p in range(ep):
        pl.semaphore_signal(
            barrier,
            inc=1,
            device_id=(jnp.int32(p),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    # Every peer has entered, so its output buffer is ours to write.
    pl.semaphore_wait(barrier, ep)

    dst = o_hbm.at[pl.ds(pl.multiple_of(me * ROWS, ROWS), ROWS)]
    for p in range(ep):

        @pl.when(me != p)
        def _(p=p):
            pltpu.make_async_remote_copy(
                src_ref=x_hbm,
                dst_ref=dst,
                send_sem=sems.at[0],
                recv_sem=sems.at[1],
                device_id=(jnp.int32(p),),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

    local = pltpu.make_async_copy(x_hbm, dst, sems.at[2])
    local.start()
    # Same byte count as one transfer: each wait retires one send / arrival.
    sent = pltpu.make_async_copy(x_hbm, o_hbm.at[pl.ds(0, ROWS)], sems.at[0])
    arrived = pltpu.make_async_copy(x_hbm, o_hbm.at[pl.ds(0, ROWS)], sems.at[1])
    for _ in range(ep - 1):
        sent.wait()
        arrived.wait()
    local.wait()


def oneshot_all_gather(row: jax.Array, me: jax.Array, ep: int) -> jax.Array:
    """[w] int32 on each of ``ep`` shards -> [ep, w], ordered by mesh index.

    Call inside a ``shard_map`` over a 1-D mesh; ``me`` is this shard's mesh
    index as an int32 scalar array. ``w`` must be a multiple of 128.
    """
    (width,) = row.shape
    if row.dtype != jnp.int32 or width % 128:
        raise ValueError(
            f"needs an int32 row of 128-multiple width; got {row.dtype}[{width}]"
        )
    out = pl.pallas_call(
        functools.partial(_kernel, ep=ep),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA((3,))],
        ),
        out_shape=jax.ShapeDtypeStruct((ep * ROWS, width), jnp.int32),
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name="oneshot_all_gather",
    )(jnp.reshape(me, (1,)).astype(jnp.int32), jnp.broadcast_to(row, (ROWS, width)))
    return out[::ROWS]
