#!/usr/bin/env python3
"""Packed o_proj + tile-exchange probe (T=1 payload, 32 devices).

Built up from the proven piece -- the packed o_proj kernel emitting
[28,2,128] dense T(2,128) -- extended with JUST the reduce-scatter's data
movement: tile d of every rank's payload is pushed to rank d (its owner;
ranks 28-31 own nothing and only send dummy tiles, keeping copy counts
uniform). Owners then reduce the 32 copies of their tile. This is phase 1
of the packed two-shot AR, isolated so its compile/execute cost and its
HLO footprint are measurable on their own.

Protocol (probe variant): 31 async remote copies, sender-side waits, then
a full barrier, then owners stage+reduce. No entry barrier (see the kernel
comment for the cross-call hazard this accepts). Whole-tile slices on the
untiled leading dim only.

MEASURED TRANSPORT RULES (the point of this probe; see
docs/llar_out_proj_op.md, "Packed-layout probe"):

* receive counts must be EXACTLY uniform across ranks -- a rank receiving
  zero (rem29, seldst28, pipelined) or extra (uneven1) copies deadlocks the
  exchange at execution time;
* self-sends are legal (selfsend);
* static and dynamic src slices both work (statsrc);
* remote DMA inside pl.when deadlocks (job 485); local DMA inside pl.when
  is fine;
* Mosaic preserves source-order DMA/compute interleaving in the generated
  LLO (no sinking/hoisting of copy-starts), and per-tile interleaving pays
  ~96ns per boundary in store-visibility stalls (pipedup 20.03us vs
  slabdup 17.35us); half-payload overlap nets -0.43us (pipehalf 16.885us
  vs sequential 17.319us).

Prints lower/compile/execute timings per module so any stall is
attributed. Run under srun on a 4-host ici-2x2x4 slice with
--xla_dump_to for the glue audit.
"""

import os
import sys
import time

# These slices boot libtpu in multislice mode: a JAX process that does not
# join the distributed runtime hangs in backend init waiting for the other
# hosts ("Did you run your code on all TPU hosts?"). Join on every task.
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax
    jax.distributed.initialize()

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

T, HID, SHARD = 1, 7168, 224  # T=1 payload; 7168/32 = 224 contracting shard
TILES, PACK, LANES = 28, 2, 128  # 28 * 2 * 128 = 7168
NUM_DEVICES = 32

# PACKED_PROBE_ENTRY_BARRIER=1 puts a full barrier at kernel entry (before
# the dot): with it, bundle 0 is barrier setup rather than the dot's address
# constants, which -- if that is what fills bundle 0's IMM slots -- lets the
# dense LLO bundle instrumentation place its trace op (the no-entry-barrier
# kernel otherwise fails to compile with "all IMM slots are occupied" at the
# default trace frequency, job 463).
_ENTRY_FIRST = bool(os.environ.get("PACKED_PROBE_ENTRY_BARRIER"))


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def oproj_packed(x, w):
    """The [1,7168] o_proj partial, stored packed as [28,2,128] in VMEM."""

    def kernel(x_ref, w_ref, o_ref):
        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype).reshape(o_ref.shape)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((TILES, PACK, LANES), x.dtype),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    )(x, w)


def _full_barrier(my_id, num_devices):
    """All-to-all barrier across every device on the mesh axis."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(1, num_devices):
        pl.semaphore_signal(
            barrier_sem,
            1,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, num_devices - 1)

    @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.REGULAR)
    def _(sem):
        for i in range(1, num_devices):
            pl.semaphore_signal(
                sem,
                1,
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, num_devices - 1)


def fused_oproj_rs(x, w, rs, *, num_devices, axis_name):
    """o_proj partial + reduce-scatter tile exchange in one custom call.

    Returns the packed [28,2,128] payload with, on owner ranks, tile my_id
    replaced by its 32-rank reduction.
    """

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

        # NOTE: no entry barrier. Intra-call read safety comes from the
        # post-send barrier below plus the sender-side copy waits (every
        # delivery is complete before any rank passes it). What this gives
        # up is the cross-call guarantee: XLA reuses this module's staging
        # buffer addresses across executions, so a fast rank's call k+1
        # sends can overwrite slots a slow owner still reads in call k.
        # Production closes that with slot rotation, not a per-call entry
        # barrier; this probe only checks the first execution's numerics.

        # Push tile dst to rank dst (its owner); on the owner's buffer it
        # lands in sender slot my_id. Non-owner destinations get a dummy
        # tile they never read, keeping copy counts uniform.
        copies = []
        for i in range(1, num_devices):
            dst = lax.rem(my_id + i, num_devices)
            tile = lax.rem(dst, TILES)
            copy = pltpu.make_async_remote_copy(
                src_ref=payload_vmem.at[tile],
                dst_ref=rs_ref.at[my_id],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(dst, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        # Sender-side waits mean delivered; the barrier orders every rank's
        # deliveries before any rank reads its receive buffer.
        _full_barrier(my_id, num_devices)

        o_ref[...] = payload_vmem[...]

        # Owners reduce their tile: own copy plus one per other rank. Slot
        # my_id of the receive buffer is never written (no rank sends to
        # itself), so summing all slots adds a zero -- no dynamic skip.
        @pl.when(my_id < TILES)
        def _():
            stage = pltpu.make_async_copy(src_ref=rs_ref.at[0:num_devices],
                                          dst_ref=recv_vmem,
                                          sem=local_sem)
            stage.start()
            stage.wait()
            acc = payload_vmem[my_id]
            for j in range(num_devices):
                acc = acc + recv_vmem[j]
            o_ref[my_id] = acc

    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((TILES, PACK, LANES), x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.VMEM((num_devices, PACK, LANES), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=8,
                                             disable_semaphore_checks=True),
    )(x, w, rs)
    return out


def _rs_call(kernel, x, w, rs):
    """Shared pallas_call wrapper for the fused o_proj + RS probe kernels."""
    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((TILES, PACK, LANES), x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.VMEM((NUM_DEVICES, PACK, LANES), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=8,
                                             disable_semaphore_checks=True),
    )(x, w, rs)
    return out


def _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
             num_devices):
    """Post-send half shared by the probe kernels: barrier, writeback, and
    the owner-side reduce of the received tiles."""

    # Sender-side waits mean delivered; the barrier orders every rank's
    # deliveries before any rank reads its receive buffer.
    _full_barrier(my_id, num_devices)

    o_ref[...] = payload_vmem[...]

    # Owners reduce their tile: own copy plus one per other rank. Slot
    # my_id of the receive buffer is never written (no rank sends to
    # itself), so summing all slots adds a zero -- no dynamic skip. The
    # leading slice keeps this correct when rs carries a trailing trash
    # slot (the selfsend/uneven1 bisect variants).
    @pl.when(my_id < TILES)
    def _():
        stage = pltpu.make_async_copy(src_ref=rs_ref.at[0:num_devices],
                                      dst_ref=recv_vmem,
                                      sem=local_sem)
        stage.start()
        stage.wait()
        acc = payload_vmem[my_id]
        for j in range(num_devices):
            acc = acc + recv_vmem[j]
        o_ref[my_id] = acc


def _send_tile(payload_ref, rs_ref, i, my_id, send_sem, recv_sem):
    """Push tile i to its owner (rank i); own tile goes to ownerless rank
    TILES as a trash write (never read), keeping the loop branch-free.

    HISTORICAL NOTE, superseded: this pattern deadlocks (jobs 477/478/480-
    485), and the cause was first mis-attributed to the lax.select dst
    lowering (hence the comparison-plus-add spelling). The real cause is the
    ZERO-RECEIVER ranks: ranks 29-31 receive no copies, and the transport
    deadlocks unless every rank receives exactly as many copies as every
    other (isolated by jobs 494/495/503: static src slices are fine, a
    zero-receiver variant of the WORKING rem pattern hangs, and so does an
    extra-receiver variant; only exactly-uniform receive counts work).
    pipedup/pipehalf are the receive-uniform repair of this pattern.
    """
    dst = i + (i >= my_id)
    return pltpu.make_async_remote_copy(
        src_ref=payload_ref.at[i],
        dst_ref=rs_ref.at[my_id],
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=(dst, ),
        device_id_type=pl.DeviceIdType.MESH,
    )


def fused_oproj_rs_slabdot(x, w, rs, *, num_devices, axis_name):
    """Bisect A: 256-column slab dots, but sends AFTER all stores.

    Same per-slab dot and same select-dst sends as the pipelined kernel,
    with no compute/DMA interleaving.
    """

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        for i in range(TILES):
            slab = jax.lax.dot_general(x_ref[...],
                                       w_ref[i * 256:(i + 1) * 256, :],
                                       dimension_numbers=(((1, ), (1, )),
                                                          ((), ())),
                                       preferred_element_type=jnp.float32)
            payload_vmem[i] = slab.astype(o_ref.dtype).reshape(PACK, LANES)

        copies = []
        for i in range(TILES):
            copy = _send_tile(payload_vmem, rs_ref, i, my_id, send_sem,
                              recv_sem)
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_rs_interleave(x, w, rs, *, num_devices, axis_name):
    """Bisect B: ONE full-width dot, then per-tile [store, send] pairs.

    Keeps the store->DMA adjacency of the pipelined kernel (each tile's
    remote copy starts immediately after its store) but no slab dots --
    the MXU is idle while the sends run.
    """

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        packed = acc.astype(o_ref.dtype).reshape(TILES, PACK, LANES)

        copies = []
        for i in range(TILES):
            payload_vmem[i] = packed[i]
            copy = _send_tile(payload_vmem, rs_ref, i, my_id, send_sem,
                              recv_sem)
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_rs_seldst28(x, w, rs, *, num_devices, axis_name):
    """Bisect C: sequential kernel with ONLY the send loop changed to the
    tile-major select-dst pattern (28 copies, dst = i, own tile -> rank
    TILES). No slab dots, no interleave. A hang here indicts the
    destination computation itself (e.g. a self-send if lax.select
    mislowers)."""

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

        copies = []
        for i in range(TILES):
            copy = _send_tile(payload_vmem, rs_ref, i, my_id, send_sem,
                              recv_sem)
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_rs_statsrc(x, w, rs, *, num_devices, axis_name):
    """Bisect D: EXACTLY sequential's send pattern (31 copies, rem dst,
    every rank receives, same send order) but with STATIC python-int src
    tile indices instead of the dynamic lax.rem(dst, TILES) slice.

    Hang-isolation only: the tile shipped to dst is wrong (i % TILES, not
    rem(dst, TILES)), so owner-tile numerics are expected to mismatch. Every
    hanging variant so far (seldst28 x3, pipelined x2) used static src
    slices; the working sequential uses dynamic ones. A hang here indicts
    the static leading-dim src-slice lowering into the DMA descriptor.
    """

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

        copies = []
        for i in range(1, num_devices):
            dst = lax.rem(my_id + i, num_devices)
            copy = pltpu.make_async_remote_copy(
                src_ref=payload_vmem.at[i % TILES],
                dst_ref=rs_ref.at[my_id],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(dst, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_rs_rem29(x, w, rs, *, num_devices, axis_name):
    """Bisect E: EXACTLY sequential's send pattern (31 copies, dynamic
    rem-dst, dynamic src slice) but with dst = rem(my_id + i, 29) -- ranks
    29-31 receive ZERO copies (and one dst gets a duplicate, benign: same
    tile value written twice into the same slot).

    Hang-isolation only: ranks skipped by a sender's rem-29 rotation miss
    that sender's contribution, so owner-tile numerics are expected to
    mismatch on those ranks. Every working variant has uniform receive
    counts; every hanging variant leaves ranks 29-31 with zero receives. A
    hang here indicts zero-receiver ranks (ICI delivery requires the
    destination to participate), not the dst arithmetic or src slicing.
    """

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

        copies = []
        for i in range(1, num_devices):
            dst = lax.rem(my_id + i, 29)
            tile = lax.rem(dst, TILES)
            copy = pltpu.make_async_remote_copy(
                src_ref=payload_vmem.at[tile],
                dst_ref=rs_ref.at[my_id],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(dst, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


# Variants whose owner-tile numerics are intentionally wrong (hang-mechanism
# bisections): run them, but don't gate PASS/FAIL on their rel err.
_NUMERICS_NA = {"statsrc", "rem29"}


def fused_oproj_rs_selfsend(x, w, rs, *, num_devices, axis_name):
    """Bisect F: sequential + a 32nd send whose dst is the SENDER ITSELF,
    landing in the trash slot ``num_devices`` of its own rs buffer (never
    read; the reduce still sums slots 0..num_devices-1 with slot my_id
    zero). Receive counts stay uniform at 32 per rank and numerics stay
    correct. Tests whether a remote copy whose destination is the sending
    device itself is legal -- if yes, tile-ordered pipelined sends can be
    made receive-uniform WITHOUT conditional DMA by keeping the self-copy.
    """

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

        copies = []
        for i in range(1, num_devices):
            dst = lax.rem(my_id + i, num_devices)
            tile = lax.rem(dst, TILES)
            copy = pltpu.make_async_remote_copy(
                src_ref=payload_vmem.at[tile],
                dst_ref=rs_ref.at[my_id],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(dst, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        # The self-copy: own tile into own trash slot, never read.
        self_copy = pltpu.make_async_remote_copy(
            src_ref=payload_vmem.at[lax.rem(my_id, TILES)],
            dst_ref=rs_ref.at[num_devices],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(my_id, ),
            device_id_type=pl.DeviceIdType.MESH)
        self_copy.start()
        copies.append(self_copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_rs_uneven1(x, w, rs, *, num_devices, axis_name):
    """Bisect G: sequential + one EXTRA copy per sender to rank 0 (tile 0,
    trash slot ``num_devices``). Rank 0 receives 32 copies, everyone else
    31; there are NO zero-receiver ranks, and numerics stay correct. Tests
    whether receive counts must be exactly uniform or whether only the
    zero-receiver case deadlocks -- decides whether tile-ordered sends may
    redirect the self-copy to an arbitrary trash destination."""

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

        copies = []
        for i in range(1, num_devices):
            dst = lax.rem(my_id + i, num_devices)
            tile = lax.rem(dst, TILES)
            copy = pltpu.make_async_remote_copy(
                src_ref=payload_vmem.at[tile],
                dst_ref=rs_ref.at[my_id],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(dst, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        # The extra copy: every sender also pushes tile 0 to rank 0's trash
        # slot, making rank 0's receive count 32 vs everyone else's 31.
        extra = pltpu.make_async_remote_copy(
            src_ref=payload_vmem.at[0],
            dst_ref=rs_ref.at[num_devices],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(0, ),
            device_id_type=pl.DeviceIdType.MESH)
        extra.start()
        copies.append(extra)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_rs_pipedup(x, w, rs, *, num_devices, axis_name):
    """Pipelined done the receive-uniform way: slab dots with tile-ordered
    sends, every rank receiving exactly num_devices-1 remote copies.

    The tile-major hang (jobs 477/478/480-485) is isolated by jobs 494/495
    to ZERO-RECEIVER ranks, not the send order or the src slicing. So this
    variant keeps the overlap-friendly order -- tile t is sent right after
    slab t's dot, to the ranks that own tile t: {t, t+TILES} (t+TILES < 32
    for t < 4) -- but duplicates tiles 0-3 to the otherwise-ownerless ranks
    28-31 so nobody receives zero. A sender's copy to ITSELF (tile my_id
    when my_id < TILES, or tile my_id-TILES for ranks 28-31) is legal
    (selfsend, job 496) and is redirected to the trash slot num_devices of
    its own rs buffer so the reduce's zero-assumption on slot my_id holds.
    """

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        copies = []
        for t in range(TILES):
            slab = jax.lax.dot_general(x_ref[...],
                                       w_ref[t * 256:(t + 1) * 256, :],
                                       dimension_numbers=(((1, ), (1, )),
                                                          ((), ())),
                                       preferred_element_type=jnp.float32)
            payload_vmem[t] = slab.astype(o_ref.dtype).reshape(PACK, LANES)
            dsts = (t, t + TILES) if t + TILES < num_devices else (t, )
            for d in dsts:
                copy = pltpu.make_async_remote_copy(
                    src_ref=payload_vmem.at[t],
                    dst_ref=rs_ref.at[lax.select(d == my_id, num_devices,
                                                 my_id)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(d, ),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_rs_slabdup(x, w, rs, *, num_devices, axis_name):
    """pipedup WITHOUT the interleave: identical slab dots and identical
    receive-uniform tile-ordered sends, but every send is issued after all
    28 tiles are stored. pipedup minus slabdup is the measured overlap gain;
    slabdup minus sequential is the cost of the slab-dot decomposition and
    the tile-ordered send pattern on their own."""

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        for t in range(TILES):
            slab = jax.lax.dot_general(x_ref[...],
                                       w_ref[t * 256:(t + 1) * 256, :],
                                       dimension_numbers=(((1, ), (1, )),
                                                          ((), ())),
                                       preferred_element_type=jnp.float32)
            payload_vmem[t] = slab.astype(o_ref.dtype).reshape(PACK, LANES)

        copies = []
        for t in range(TILES):
            dsts = (t, t + TILES) if t + TILES < num_devices else (t, )
            for d in dsts:
                copy = pltpu.make_async_remote_copy(
                    src_ref=payload_vmem.at[t],
                    dst_ref=rs_ref.at[lax.select(d == my_id, num_devices,
                                                 my_id)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(d, ),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_rs_pipehalf(x, w, rs, *, num_devices, axis_name):
    """pipedup at HALF-payload granularity: compute slabs 0..13, start their
    sends, then compute slabs 14..27 while the first half's DMAs are in
    flight, then send the rest. pipedup showed the per-tile interleave costs
    ~96ns per tile boundary (store-visibility stall on the just-computed
    slab, job 499 vs 501); one boundary instead of 27 should keep most of
    the overlap at ~1/27th of the stall cost."""

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        copies = []
        for t in range(TILES):
            slab = jax.lax.dot_general(x_ref[...],
                                       w_ref[t * 256:(t + 1) * 256, :],
                                       dimension_numbers=(((1, ), (1, )),
                                                          ((), ())),
                                       preferred_element_type=jnp.float32)
            payload_vmem[t] = slab.astype(o_ref.dtype).reshape(PACK, LANES)
            if t == TILES // 2 - 1 or t == TILES - 1:
                for u in range(
                        t - TILES // 2 + 1 if t < TILES - 1 else TILES // 2,
                        t + 1):
                    dsts = (u, u + TILES) if u + TILES < num_devices else (u, )
                    for d in dsts:
                        copy = pltpu.make_async_remote_copy(
                            src_ref=payload_vmem.at[u],
                            dst_ref=rs_ref.at[lax.select(
                                d == my_id, num_devices, my_id)],
                            send_sem=send_sem,
                            recv_sem=recv_sem,
                            device_id=(d, ),
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                        copy.start()
                        copies.append(copy)
        for copy in copies:
            copy.wait()

        _rs_tail(my_id, payload_vmem, rs_ref, o_ref, recv_vmem, local_sem,
                 num_devices)

    return _rs_call(kernel, x, w, rs)


def fused_oproj_ar_full(x, w, rs, ag, *, num_devices, axis_name):
    """The complete packed two-shot all-reduce fused with the o_proj dot.

    Phase 1 (RS): every rank pushes tile d to its owner rank d, using the
    proven offset-rem send pattern (the tile-major pattern hangs; jobs
    477/478/480/482). Owners reduce their tile. Phase 2 (AG): owners
    broadcast the reduced tile into slot my_id of every rank's assembly
    buffer, then every rank stages the full [28,2,128] result out. The
    output is genuinely replicated across ranks.
    """

    def kernel(x_ref, w_ref, rs_ref, ag_ref, o_ref, payload_vmem, recv_vmem,
               red_vmem, asm_vmem, local_sem, send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

        # Phase 1 -- push tile dst to rank dst (rem pattern: the tile-major
        # variant deadlocks on this stack). Dummy tile for ownerless dst.
        copies = []
        for i in range(1, num_devices):
            dst = lax.rem(my_id + i, num_devices)
            tile = lax.rem(dst, TILES)
            copy = pltpu.make_async_remote_copy(
                src_ref=payload_vmem.at[tile],
                dst_ref=rs_ref.at[my_id],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(dst, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        # Delivered before barrier arrival, so reads after it are safe.
        _full_barrier(my_id, num_devices)

        # Owners reduce their tile (slot my_id of rs is never written: no
        # rank sends to itself).
        @pl.when(my_id < TILES)
        def _():
            stage = pltpu.make_async_copy(src_ref=rs_ref.at[0:num_devices],
                                          dst_ref=recv_vmem,
                                          sem=local_sem)
            stage.start()
            stage.wait()
            acc = payload_vmem[my_id]
            for j in range(num_devices):
                acc = acc + recv_vmem[j]
            red_vmem[...] = acc

        # Phase 2 -- broadcast. FULLY UNIFORM: no remote DMA inside a
        # conditional (the pl.when-wrapped broadcasts deadlocked at runtime,
        # job 485) -- every rank sends 31 copies; ownerless ranks broadcast
        # their (garbage) red_vmem into the trash slot TILES of everyone's
        # assembly buffer, which is never read.
        slot = lax.min(my_id, TILES)
        local = pltpu.make_async_copy(src_ref=red_vmem,
                                      dst_ref=ag_ref.at[slot],
                                      sem=local_sem)
        local.start()
        copies = []
        for i in range(1, num_devices):
            copy = pltpu.make_async_remote_copy(
                src_ref=red_vmem,
                dst_ref=ag_ref.at[slot],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        local.wait()
        for copy in copies:
            copy.wait()

        _full_barrier(my_id, num_devices)

        # Assembly: the local assembly buffer holds every tile plus trash.
        stage = pltpu.make_async_copy(src_ref=ag_ref.at[0:TILES],
                                      dst_ref=asm_vmem,
                                      sem=local_sem)
        stage.start()
        stage.wait()
        o_ref[...] = asm_vmem[...]

    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((TILES, PACK, LANES), x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.VMEM((num_devices, PACK, LANES), x.dtype),
            pltpu.VMEM((PACK, LANES), x.dtype),
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=8,
                                             disable_semaphore_checks=True),
    )(x, w, rs, ag)
    return out


def fused_oproj_ar_nobarrier(x, w, rs, ag, *, num_devices, axis_name):
    """fullar with the three full barriers replaced by signal rounds.

    A full barrier is a double all-to-all signal+wait (~6 us each in this
    probe, jobs 462 vs 465). What the phases actually need is receiver-side
    knowledge that the expected copies landed. pl.semaphore_wait refuses
    DMA semaphores, so instead: each rank waits for ITS OWN sends to
    complete (sender-side copy.wait() proves delivery), then does ONE
    remote-signal round onto a REGULAR scratch semaphore per phase (the
    mechanism the barrier's second round already uses), and the reader
    waits for num_devices-1 signals. arr1 gates the owner reduce (phase-1
    arrivals), arr2 gates assembly (phase-2 arrivals); separate semaphores
    so a fast rank's phase-2 signals cannot satisfy a phase-1 wait.

    No entry barrier: cross-call staging-buffer reuse can race a slow
    rank's reads, benign in this probe because every call reduces the same
    inputs (production needs slot rotation for this). Semaphore residue at
    exit is tolerated (disable_semaphore_checks).
    """

    def kernel(x_ref, w_ref, rs_ref, ag_ref, o_ref, payload_vmem, recv_vmem,
               red_vmem, asm_vmem, local_sem, send_sem, recv_sem, arr1, arr2):
        my_id = lax.axis_index(axis_name)

        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

        # Phase 1 -- push tile dst to rank dst (rem pattern); dummy tile for
        # ownerless dst. Wait for my sends (delivery proven), then one
        # signal round: tell every peer its copy from me landed.
        copies = []
        for i in range(1, num_devices):
            dst = lax.rem(my_id + i, num_devices)
            tile = lax.rem(dst, TILES)
            copy = pltpu.make_async_remote_copy(
                src_ref=payload_vmem.at[tile],
                dst_ref=rs_ref.at[my_id],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(dst, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        for i in range(1, num_devices):
            pl.semaphore_signal(arr1,
                                1,
                                device_id=(lax.rem(my_id + i, num_devices), ),
                                device_id_type=pl.DeviceIdType.MESH)

        # Owners reduce their tile once all 31 arrivals are signaled (slot
        # my_id of rs is never written: no rank sends to itself).
        @pl.when(my_id < TILES)
        def _():
            pl.semaphore_wait(arr1, num_devices - 1)
            stage = pltpu.make_async_copy(src_ref=rs_ref.at[0:num_devices],
                                          dst_ref=recv_vmem,
                                          sem=local_sem)
            stage.start()
            stage.wait()
            acc = payload_vmem[my_id]
            for j in range(num_devices):
                acc = acc + recv_vmem[j]
            red_vmem[...] = acc

        # Phase 2 -- uniform broadcast (trash slot TILES for ownerless
        # ranks; never read), no conditional DMA.
        slot = lax.min(my_id, TILES)
        local = pltpu.make_async_copy(src_ref=red_vmem,
                                      dst_ref=ag_ref.at[slot],
                                      sem=local_sem)
        local.start()
        copies = []
        for i in range(1, num_devices):
            copy = pltpu.make_async_remote_copy(
                src_ref=red_vmem,
                dst_ref=ag_ref.at[slot],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        local.wait()
        for copy in copies:
            copy.wait()
        for i in range(1, num_devices):
            pl.semaphore_signal(arr2,
                                1,
                                device_id=(lax.rem(my_id + i, num_devices), ),
                                device_id_type=pl.DeviceIdType.MESH)

        # Assembly once every phase-2 tile has landed locally.
        pl.semaphore_wait(arr2, num_devices - 1)
        stage = pltpu.make_async_copy(src_ref=ag_ref.at[0:TILES],
                                      dst_ref=asm_vmem,
                                      sem=local_sem)
        stage.start()
        stage.wait()
        o_ref[...] = asm_vmem[...]

    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((TILES, PACK, LANES), x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.VMEM((num_devices, PACK, LANES), x.dtype),
            pltpu.VMEM((PACK, LANES), x.dtype),
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.REGULAR,
        ],
        # No collective_id: this kernel uses no barrier semaphore, and the
        # compile refuses one otherwise.
        compiler_params=pltpu.CompilerParams(disable_semaphore_checks=True),
    )(x, w, rs, ag)
    return out


def fused_oproj_rs_pipelined(x, w, rs, *, num_devices, axis_name):
    """Fused o_proj + RS, software-pipelined: send tile i while the MXU
    computes tile i+1.

    The dot runs in 256-column slabs (the granularity the LLO disassembly
    showed: one vmatmul per packed output tile); each slab is stored into
    its packed tile and its async remote copy is started immediately, so
    the ICI DMA overlaps the next slab's matmul. Own tile takes a trash
    send to ownerless rank TILES (never read) to keep the loop free of
    dynamic control flow.
    """

    def kernel(x_ref, w_ref, rs_ref, o_ref, payload_vmem, recv_vmem, local_sem,
               send_sem, recv_sem):
        my_id = lax.axis_index(axis_name)

        if _ENTRY_FIRST:
            _full_barrier(my_id, num_devices)

        copies = []
        for i in range(TILES):
            slab = jax.lax.dot_general(x_ref[...],
                                       w_ref[i * 256:(i + 1) * 256, :],
                                       dimension_numbers=(((1, ), (1, )),
                                                          ((), ())),
                                       preferred_element_type=jnp.float32)
            payload_vmem[i] = slab.astype(o_ref.dtype).reshape(PACK, LANES)
            copy = _send_tile(payload_vmem, rs_ref, i, my_id, send_sem,
                              recv_sem)
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        # Sender-side waits mean delivered; the barrier orders every rank's
        # deliveries before any rank reads its receive buffer.
        _full_barrier(my_id, num_devices)

        o_ref[...] = payload_vmem[...]

        @pl.when(my_id < TILES)
        def _():
            stage = pltpu.make_async_copy(src_ref=rs_ref,
                                          dst_ref=recv_vmem,
                                          sem=local_sem)
            stage.start()
            stage.wait()
            acc = payload_vmem[my_id]
            for j in range(num_devices):
                acc = acc + recv_vmem[j]
            o_ref[my_id] = acc

    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((TILES, PACK, LANES), x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((TILES, PACK, LANES), x.dtype),
            pltpu.VMEM((num_devices, PACK, LANES), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=8,
                                             disable_semaphore_checks=True),
    )(x, w, rs)
    return out


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ok = True

    # Single-device regression guard: the known-good packed o_proj alone.
    dev = jax.local_devices()[0]
    with jax.default_device(dev):
        x1 = jax.random.normal(jax.random.key(0), (T, SHARD), jnp.bfloat16)
        w1 = jax.random.normal(jax.random.key(1), (HID, SHARD), jnp.bfloat16)
        t0 = time.time()
        out1 = jax.jit(oproj_packed)(x1, w1)
        want1 = np.asarray(
            jax.jit(lambda a, b: jnp.dot(a, b.T).astype(jnp.float32))(x1, w1))
        rel = float(
            np.max(
                np.abs(
                    np.asarray(out1, jnp.float32).reshape(-1) -
                    want1.reshape(-1))) / max(np.max(np.abs(want1)), 1e-6))
        _stamp(
            rank, f"oproj_packed: rel err = {rel:.3g} "
            f"(compile+run {time.time() - t0:.1f}s)")
        ok = ok and rel < 0.05

    # The fused o_proj + tile exchange over the full slice.
    devices = mesh_utils.create_device_mesh((NUM_DEVICES, ),
                                            devices=jax.devices())
    mesh = Mesh(devices, ("tp", ))

    spec_in = (P(None, "tp"), P(), P(), P())
    bodies = {
        "sequential": fused_oproj_rs,
        "pipelined": fused_oproj_rs_pipelined,
        "slabdot": fused_oproj_rs_slabdot,
        "interleave": fused_oproj_rs_interleave,
        "seldst28": fused_oproj_rs_seldst28,
        "statsrc": fused_oproj_rs_statsrc,
        "rem29": fused_oproj_rs_rem29,
        "selfsend": fused_oproj_rs_selfsend,
        "uneven1": fused_oproj_rs_uneven1,
        "pipedup": fused_oproj_rs_pipedup,
        "slabdup": fused_oproj_rs_slabdup,
        "pipehalf": fused_oproj_rs_pipehalf,
        "fullar": fused_oproj_ar_full,
        "fullar_nb": fused_oproj_ar_nobarrier,
    }
    chosen = os.environ.get("PACKED_PROBE_VARIANTS",
                            "sequential,pipelined").split(",")
    variants = {}
    for name in chosen:
        b = bodies[name]

        def make_body(b):
            if b in (fused_oproj_ar_full, fused_oproj_ar_nobarrier):
                return lambda xl, w, rs, ag: b(
                    xl, w, rs, ag, num_devices=NUM_DEVICES, axis_name="tp")
            return lambda xl, w, rs, ag: b(
                xl, w, rs, num_devices=NUM_DEVICES, axis_name="tp")

        variants[name] = jax.jit(
            jax.shard_map(make_body(b),
                          mesh=mesh,
                          in_specs=spec_in,
                          out_specs=P(),
                          check_vma=False))
    ref = jax.jit(
        jax.shard_map(lambda xl, w: lax.psum(
            jax.lax.dot_general(xl,
                                w,
                                dimension_numbers=(((1, ), (1, )), ((), ())),
                                preferred_element_type=jnp.float32), "tp"),
                      mesh=mesh,
                      in_specs=spec_in[:2],
                      out_specs=P(),
                      check_vma=False))

    x = jax.jit(lambda: jax.random.normal(jax.random.key(3),
                                          (T, HID), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P(None, "tp")))()
    w = jax.jit(lambda: jax.random.normal(jax.random.key(1),
                                          (HID, SHARD), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P()))()
    # One extra slot: the selfsend/uneven1 bisect variants park their
    # self/extra copies in slot NUM_DEVICES (never read); the reduce stages
    # only slots [0:num_devices].
    rs = jax.jit(lambda: jnp.zeros(
        (NUM_DEVICES + 1, PACK, LANES), jnp.bfloat16),
                 out_shardings=NamedSharding(mesh, P()))()
    # One extra slot: phase-2 broadcasts from ownerless ranks land in the
    # trash slot (never read) so the AG phase needs no conditional DMA.
    ag = jax.jit(lambda: jnp.zeros((TILES + 1, PACK, LANES), jnp.bfloat16),
                 out_shardings=NamedSharding(mesh, P()))()

    t0 = time.time()
    want = ref(x, w)  # [1,7168] replicated, kept on device
    jax.block_until_ready(want)
    _stamp(rank, f"ref (psum) ran in {time.time() - t0:.1f}s")

    # Verify on device: the output is genuinely per-rank here (RS only), so
    # each device checks its own owner tile against the psum reference.
    def check_body(o, ref_flat):
        my = lax.axis_index("tp")
        ref_tiles = ref_flat.reshape(TILES, PACK, LANES)
        err = jnp.max(jnp.abs(o[my].astype(jnp.float32) - ref_tiles[my]))
        err = jnp.where(my < TILES, err, 0.0)
        return lax.pmax(err, "tp")

    checker = jax.jit(
        jax.shard_map(check_body,
                      mesh=mesh,
                      in_specs=(P(), P()),
                      out_specs=P(),
                      check_vma=False))
    scale = float(np.max(np.abs(np.asarray(want, jnp.float32))))

    # LLO / custom-call trace capture (needs the xla_xprof flags in
    # LIBTPU_INIT_ARGS, set before jax import -- see the probe sbatch).
    prof_root = os.environ.get("PACKED_PROBE_PROFILE")

    for name, fn in variants.items():
        t0 = time.time()
        lowered = fn.lower(x, w, rs, ag)
        _stamp(rank, f"{name}: lowered in {time.time() - t0:.1f}s")
        t0 = time.time()
        compiled = lowered.compile()
        _stamp(rank, f"{name}: compiled in {time.time() - t0:.1f}s")

        out = compiled(x, w, rs, ag)
        jax.block_until_ready(out)
        rel = float(checker(out, want)) / max(scale, 1e-6)
        if name in _NUMERICS_NA:
            _stamp(
                rank, f"{name}: owner-tile rel err = {rel:.3g} "
                "(n/a: hang-bisect variant, completion is the signal)")
        else:
            _stamp(rank, f"{name}: max owner-tile rel err = {rel:.3g}")
            ok = ok and rel < 0.05

        for _ in range(3):
            out = compiled(x, w, rs, ag)
        jax.block_until_ready(out)
        t0 = time.time()
        for _ in range(200):
            out = compiled(x, w, rs, ag)
        jax.block_until_ready(out)
        _stamp(
            rank, f"{name}: {(time.time() - t0) / 200 * 1e6:.2f} us/call "
            "(host dispatch included)")

        if prof_root:
            from jax import profiler
            tdir = os.path.join(prof_root, f"{name}/rank{rank}")
            profiler.start_trace(tdir)
            for _ in range(50):
                out = compiled(x, w, rs, ag)
            jax.block_until_ready(out)
            profiler.stop_trace()
            _stamp(rank, f"{name}: trace captured -> {tdir}")

    _stamp(rank, "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
