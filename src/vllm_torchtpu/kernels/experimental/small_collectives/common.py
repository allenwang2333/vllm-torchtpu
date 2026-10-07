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
"""Pieces shared by the small all-reduce and reduce-scatter kernels.

Under the torch bridge a core's own index arrives as data (an int32 [1] in
SMEM): ``lax.axis_index`` lowers to a partition-id the export path's SPMD
partitioner refuses. Peer addresses then take the dict form keyed by the axis
name, and waits go on local descriptors of the same bytes, since the bridge
refuses a wait on a descriptor that carries a peer address.
"""

import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
MESH = pl.DeviceIdType.MESH


def peer(j, axis, dict_ids):
    return {axis: jnp.int32(j)} if dict_ids else (jnp.int32(j),)


def barrier(num_cores, axis, dict_ids):
    """Every core signals every core once, itself included, and waits for all.

    One barrier per call is enough. A core finishes call k only after every
    peer's data of call k has landed, and a peer sends that data only after
    passing its own barrier of call k, so no core can signal call k + 1 while
    another still waits at the barrier of call k.
    """
    sem = pltpu.get_barrier_semaphore()
    for j in range(num_cores):
        pl.semaphore_signal(
            sem, 1, device_id=peer(j, axis, dict_ids), device_id_type=MESH
        )
    pl.semaphore_wait(sem, num_cores)


def wait_bytes_of(ref_slice, sem):
    """Block until ``sem`` has counted the bytes of ``ref_slice``."""
    pltpu.make_async_copy(ref_slice, ref_slice, sem).wait()


def push(src, dst, send_sem, recv_sem, j, axis, dict_ids):
    pltpu.make_async_remote_copy(
        src_ref=src,
        dst_ref=dst,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=peer(j, axis, dict_ids),
        device_id_type=MESH,
    ).start()


def split_rank(refs, rank_operand, axis):
    """This core's index and the remaining refs."""
    if rank_operand:
        rank_ref, *refs = refs
        return rank_ref[0], refs
    return lax.axis_index(axis), refs


def sum_slots(slots, n):
    """Float32 sum over the leading axis of a VMEM ref, rounded once."""
    acc = slots[0].astype(jnp.float32)
    for j in range(1, n):
        acc = acc + slots[j].astype(jnp.float32)
    return acc


def _cost_estimate(x):
    """Scheduling hint for XLA (row 13, close-gaps j-8228ff78).

    Without a CostEstimate a Pallas call is costed as ~free by XLA, so memory-space
    assignment sees no time in the 8-15 us an attention-exit all-reduce really takes
    and declines to prefetch the MoE router weight across it (the router then reads
    its 4 MiB weight from HBM, 3.5-4.0 us instead of 1.3-1.5 us).
    TPU_SMALL_COLLECTIVE_COST_BYTES=N (default 0 = no estimate) costs the call as N
    HBM bytes; it changes only XLA's schedule/prefetch decisions, never the math.
    """
    n = int(os.environ.get("TPU_SMALL_COLLECTIVE_COST_BYTES", "0") or "0")
    if n <= 0:
        return None
    return pl.CostEstimate(flops=0, transcendentals=0, bytes_accessed=n)


def call(kernel, x, out_rows, rank, scratch, collective_id, name):
    rank_specs = [pl.BlockSpec(memory_space=pltpu.SMEM)] if rank is not None else []
    args = (() if rank is None else (rank.astype(jnp.int32),)) + (x,)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, x.shape[1]), x.dtype),
        in_specs=[*rank_specs, pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
        cost_estimate=_cost_estimate(x),
    )(*args)
