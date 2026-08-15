# SPDX-License-Identifier: Apache-2.0
"""Low-latency all-reduce Pallas kernels for small (decode-sized) payloads.

XLA's default TPU all-reduce is *bandwidth*-optimal: it lowers to a ring
reduce-scatter followed by a ring all-gather.  That moves only
``2 * M * (N-1) / N`` bytes per device, but it costs ``2 * (N - 1)``
*sequential* hops -- 14 hops on an 8-device mesh.  At decode-sized payloads
(a few KB) the wire time is irrelevant and those serialized hop latencies plus
their per-step synchronization dominate end-to-end time.

These kernels trade bandwidth for hop count, following the standard taxonomy
for low-latency collectives (cf. arXiv:2607.16100, which develops the same
one-shot / two-shot split for NVLink):

``one_shot_all_reduce``
    A single all-to-all broadcast round: every device pushes its whole local
    buffer straight to every peer, then reduces locally.  **1 hop**, but
    ``M * (N-1)`` bytes out per device.  Lowest achievable latency; the right
    choice while the payload is small enough that ICI bandwidth is free.

``one_shot_all_reduce_ws``
    Same single round, but over a caller-owned persistent workspace, which
    lets the barrier move *after* the sends so it overlaps the transfer.
    Roughly 2x faster than ``one_shot_all_reduce`` at 2-16 KiB.  Requires the
    caller to thread a workspace and round counter through every call.

``two_shot_all_reduce``
    Reduce-scatter then all-gather, but with each phase done as a *direct*
    (non-ring) all-to-all round rather than ``N-1`` ring steps.  **2 hops**
    and only ``2 * M * (N-1) / N`` bytes out per device.  Wins once one-shot
    goes bandwidth-bound.

``two_shot_all_reduce_ws``
    The workspace treatment applied to two-shot: barrier overlapped with the
    phase-1 sends, worth ~2 us at 8 devices and ~3 us at 32.  Note the
    all-gather destination has to live in the workspace too -- the plain
    version writes peers' *output* buffers, which are as short-lived as
    scratch.  Best choice above the one-shot crossover.

    Both ``_ws`` kernels need ``rows % num_devices == 0`` for two-shot, and
    both need their workspace threaded call-to-call; see ``make_workspace`` and
    ``make_two_shot_workspace``.

All kernels are written for a 1-D device mesh and must be called inside
``shard_map`` over that mesh.  Peer addressing uses ``DeviceIdType.MESH``, so
the mesh axis order determines the physical routing.

Buffer-reuse safety
-------------------
``one_shot_all_reduce`` and ``two_shot_all_reduce`` open with an all-to-all
double barrier.  The first barrier guarantees every peer has entered the kernel
before any DMA lands in its scratch.  The second one closes the race the single
barrier leaves open: a fast peer could otherwise exit, re-enter on the *next*
invocation, and signal the (identically numbered) barrier semaphore a second
time, releasing us while a slower peer is still draining the previous call's
buffers.  This is the same double-barrier idiom as
:func:`vllm_torchtpu.kernels.collectives.util.local_barrier`, widened from
left/right neighbors to the full mesh.

That leading barrier is not just about buffer reuse -- it is what guarantees
the destination is *owned by this kernel on the receiver* when the data lands.
Kernel scratch is only live while the kernel runs, so an early write would
otherwise scribble on VMEM that XLA has handed to some other op.  That is why
the barrier cannot simply be moved after the sends, and why
``one_shot_all_reduce_ws`` needs a persistent workspace to do so.

Why the ``_ws`` kernels disable semaphore checks
-----------------------------------------------
Both set ``disable_semaphore_checks=True``, and must.  Overlapping the barrier
with the sends means a peer that has passed round k's barrier issues its round
k+1 sends before its own round k+1 barrier, so those DMAs bump our
``recv_sem[(k+1)%2]`` while we are still finishing round k.  The data is fine --
that is the other slot -- but Mosaic asserts every scratch semaphore is zero on
kernel exit, and this design deliberately violates that.  Leaving the check on
kills the slice with ``SLICE_FAILURE_SW_INJECT_ERROR``; the driver log under
``/tmp/tpu_logs_<user>/`` names the semaphore.  The early counts are consumed
by the next round's wait, verified exact against ``psum`` in float32.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = [
    "one_shot_all_reduce",
    "one_shot_all_reduce_ws",
    "make_workspace",
    "two_shot_all_reduce",
    "two_shot_all_reduce_ws",
    "make_two_shot_workspace",
]


def _full_barrier(my_id, num_devices: int, double_barrier: bool = True):
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

    if not double_barrier:
        return

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


# -----------------------------------------------------------------------------
# One-shot: 1 hop, M * (N-1) bytes out per device.
# -----------------------------------------------------------------------------


def _one_shot_kernel(x_ref, o_ref, recv_ref, send_sem, recv_sem, *,
                     num_devices: int, axis_name: str, double_barrier: bool):
    my_id = lax.axis_index(axis_name)
    _full_barrier(my_id, num_devices, double_barrier)

    # Slot assignment is fully static: the copy I push to my ``+i`` neighbor
    # always lands in that neighbor's slot ``i-1``.  Since every device follows
    # the same rule, device r's slot ``i-1`` receives from device ``r - i``,
    # so the N-1 slots are filled by N-1 distinct senders with no dynamic
    # indexing and no need to stage my own contribution into a slot.
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=recv_ref.at[i - 1],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()

    acc = x_ref[...]
    for i in range(num_devices - 1):
        acc = acc + recv_ref[i]
    o_ref[...] = acc


def one_shot_all_reduce(x: jax.Array,
                        *,
                        axis_name: str,
                        num_devices: int,
                        collective_id: int = 0,
                        double_barrier: bool = True) -> jax.Array:
    """Single-round all-to-all broadcast + local reduce. Call under shard_map.

    Args:
      x: The local shard to reduce. Any shape; lives in VMEM.
      axis_name: Mesh axis to reduce over.
      num_devices: Size of that mesh axis.
      collective_id: Barrier-semaphore id. Must differ between kernels that can
        be in flight concurrently.
      double_barrier: Keep the second barrier. Only safe to disable when the
        kernel cannot be re-entered while a peer is still draining.

    Returns:
      The sum of ``x`` across the mesh axis, same shape and dtype.
    """
    if num_devices < 2:
        return x
    kernel = functools.partial(_one_shot_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               double_barrier=double_barrier)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, *x.shape), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id,
                                             has_side_effects=True),
    )(x)


# -----------------------------------------------------------------------------
# One-shot over a persistent workspace: 1 hop, barrier overlapped with the DMA.
# -----------------------------------------------------------------------------


def _one_shot_ws_kernel(x_ref,
                        ws_in_ref,
                        step_in_ref,
                        o_ref,
                        ws_ref,
                        step_ref,
                        recv_vmem,
                        local_sem,
                        send_sem,
                        recv_sem,
                        *,
                        num_devices: int,
                        axis_name: str,
                        barrier_first: bool = False):
    del ws_in_ref  # aliased onto ws_ref; write through the output view
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)

    # Exactly one double barrier per round, in every round.
    #
    # Round 0 puts it first: nothing has synchronized yet, so a peer may not
    # have reached this kernel at all. Every later round puts it *after* the
    # sends, so the synchronization overlaps the transfer instead of
    # serializing ahead of it -- which is the whole point of this variant.
    #
    # Safety in the steady state rests on two things being double-buffered:
    #   data: round k writes ws[k%2], last read by round k-2. Passing round
    #     k-1's barrier proves every peer finished round k-2's reduce.
    #   DMA semaphores: round k uses sem[k%2], last drained at round k-2. With
    #     one shared semaphore a round-k send could satisfy a peer's still
    #     pending round k-1 byte-count wait.
    # barrier_first is a diagnostic: it puts the barrier ahead of the sends in
    # every round, giving up the overlap. Comparing against the default
    # separates "the workspace/slot/semaphore machinery is wrong" from "the
    # overlap itself is unsafe" at a given device count.
    if barrier_first:
        _full_barrier(my_id, num_devices, double_barrier=True)
    else:

        @pl.when(step == 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=ws_ref.at[slot, i - 1],
            send_sem=send_sem.at[slot],
            recv_sem=recv_sem.at[slot],
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)

    if not barrier_first:

        @pl.when(step != 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    for copy in copies:
        copy.wait()

    # ws lives in HBM so its address survives between kernels; stage the slot
    # into VMEM to reduce it.
    local = pltpu.make_async_copy(src_ref=ws_ref.at[slot],
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    local.start()
    local.wait()

    acc = x_ref[...]
    for i in range(num_devices - 1):
        acc = acc + recv_vmem[i]
    o_ref[...] = acc
    step_ref[0] = step + 1


def one_shot_all_reduce_ws(x: jax.Array,
                           ws: jax.Array,
                           step: jax.Array,
                           *,
                           axis_name: str,
                           num_devices: int,
                           collective_id: int = 0,
                           barrier_first: bool | None = None,
                           disable_semaphore_checks: bool = True):
    """One-shot all-reduce over a caller-owned persistent workspace.

    Removes barrier #1 from the critical path. :func:`one_shot_all_reduce`
    must barrier *before* sending, because its receive buffer is kernel scratch
    and is only live while the kernel runs -- a write that arrives early lands
    in VMEM that XLA has handed to some other op. Here the destination is a
    caller-owned buffer threaded through every call, so it stays live between
    kernels and an early write is harmless. That lets the barrier move after
    the sends, where it overlaps the transfer.

    Args:
      x: The local shard to reduce.
      ws: Workspace of shape ``(2, num_devices - 1, *x.shape)``, same dtype as
        ``x``. Must be threaded call-to-call; see :func:`make_workspace`.
      step: int32 ``(1,)`` round counter, also threaded. Must start at 0.
      axis_name: Mesh axis to reduce over.
      num_devices: Size of that mesh axis.
      collective_id: Barrier-semaphore id.
      barrier_first: Put the barrier ahead of the sends, giving up the overlap
        and most of the speedup. Defaults to off; kept as an escape hatch, and
        as the control that isolated the bug below.
      disable_semaphore_checks: Must stay True for the overlap. Mosaic
        otherwise asserts every scratch semaphore is zero on kernel exit, and
        this design *deliberately* violates that: a peer that has passed round
        k's barrier issues its round k+1 sends before its own round k+1
        barrier, so those DMAs land in our ws[(k+1)%2] and bump
        recv_sem[(k+1)%2] while we are still finishing round k. The data is
        fine -- that is the other slot, which is what double buffering is for
        -- but the semaphore is legitimately nonzero at exit, and the check
        fires as "Semaphore (scratch argument 3) has a nonzero value upon exit
        from a Mosaic kernel", killing the slice. The early counts carry over
        and are consumed by the next round, verified exact against psum in
        float32 over 440 executions at 8 and 32 devices.

    Returns:
      ``(out, ws, step)``. The caller must pass the returned ``ws``/``step``
      into the next call -- that is what keeps the buffer live and the parity
      advancing.
    """
    if num_devices < 2:
        return x, ws, step

    if barrier_first is None:
        barrier_first = False

    kernel = functools.partial(_one_shot_ws_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               barrier_first=barrier_first)
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(ws.shape, ws.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        # Alias ws and step so the kernel updates them in place and the buffer
        # identity is stable across calls.
        input_output_aliases={
            1: 1,
            2: 2
        },
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, *x.shape), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            has_side_effects=True,
            disable_semaphore_checks=disable_semaphore_checks),
    )(x, ws, step)


def make_workspace(x_shape, dtype, num_devices: int):
    """Zeroed ``(ws, step)`` for :func:`one_shot_all_reduce_ws`."""
    import jax.numpy as jnp
    return (jnp.zeros((2, num_devices - 1, *x_shape),
                      dtype), jnp.zeros((1, ), jnp.int32))


# -----------------------------------------------------------------------------
# Two-shot over a persistent workspace: 2 hops, barrier overlapped with phase 1.
# -----------------------------------------------------------------------------


def _two_shot_ws_kernel(x_ref, rs_in_ref, ag_in_ref, step_in_ref, o_ref,
                        rs_ref, ag_ref, step_ref, recv_vmem, red_vmem,
                        local_sem, s1_send, s1_recv, s2_send, s2_recv, *,
                        num_devices: int, axis_name: str, barrier_first: bool):
    del rs_in_ref, ag_in_ref  # aliased onto the output views
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)

    if barrier_first:
        _full_barrier(my_id, num_devices, double_barrier=True)
    else:

        @pl.when(step == 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    # Phase 1 -- reduce-scatter. Push the chunk peer ``+i`` owns to it; by the
    # same static rule as one-shot it lands in that peer's slot i-1.
    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[peer],
            dst_ref=rs_ref.at[slot, i - 1],
            send_sem=s1_send.at[slot],
            recv_sem=s1_recv.at[slot],
            device_id=(peer, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)

    # Overlapped barrier, same reasoning as one_shot_all_reduce_ws: round k
    # writes slot k%2, last read at round k-2, and passing round k-1's barrier
    # proves every peer finished round k-2. It also covers phase 2, whose sends
    # come later still.
    if not barrier_first:

        @pl.when(step != 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    for copy in copies:
        copy.wait()

    # rs lives in HBM so its address survives between kernels; stage to VMEM.
    local = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    local.start()
    local.wait()

    acc = x_ref[my_id]
    for i in range(num_devices - 1):
        acc = acc + recv_vmem[i]
    red_vmem[...] = acc

    # Phase 2 -- all-gather the reduced chunk into slot ``my_id`` of every
    # device's workspace, our own included.
    local_ag = pltpu.make_async_copy(src_ref=red_vmem,
                                     dst_ref=ag_ref.at[slot, my_id],
                                     sem=local_sem)
    local_ag.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_vmem,
            dst_ref=ag_ref.at[slot, my_id],
            send_sem=s2_send.at[slot],
            recv_sem=s2_recv.at[slot],
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    local_ag.wait()
    for copy in copies:
        copy.wait()

    out_copy = pltpu.make_async_copy(src_ref=ag_ref.at[slot],
                                     dst_ref=o_ref,
                                     sem=local_sem)
    out_copy.start()
    out_copy.wait()
    step_ref[0] = step + 1


def two_shot_all_reduce_ws(x: jax.Array,
                           rs: jax.Array,
                           ag: jax.Array,
                           step: jax.Array,
                           *,
                           axis_name: str,
                           num_devices: int,
                           collective_id: int = 0,
                           barrier_first: bool = False,
                           disable_semaphore_checks: bool = True):
    """Two-shot all-reduce over a caller-owned persistent workspace.

    Does for :func:`two_shot_all_reduce` what :func:`one_shot_all_reduce_ws`
    does for one-shot: moves both the reduce-scatter staging *and* the
    all-gather destination out of kernel-owned memory, so the barrier can move
    after the phase-1 sends and overlap the transfer.

    The all-gather is the extra piece here -- ``two_shot_all_reduce`` writes
    peers' *output* buffers directly, which are as short-lived as scratch, so
    they had to move into the workspace too and be copied out at the end.

    ``x`` must be 2-D with ``x.shape[0] % num_devices == 0``.
    ``disable_semaphore_checks`` must stay True; see
    :func:`one_shot_all_reduce_ws` for why.

    Returns:
      ``(out, rs, ag, step)`` -- thread the last three into the next call.
    """
    if num_devices < 2:
        return x, rs, ag, step
    if x.ndim != 2:
        raise ValueError(f"two_shot_all_reduce_ws expects 2-D, got {x.shape}")
    rows, cols = x.shape
    if rows % num_devices != 0:
        raise ValueError(f"rows ({rows}) must be divisible by num_devices "
                         f"({num_devices})")
    chunk = rows // num_devices

    kernel = functools.partial(_two_shot_ws_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               barrier_first=barrier_first)
    out, rs, ag, step = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_devices, chunk, cols), x.dtype),
            jax.ShapeDtypeStruct(rs.shape, rs.dtype),
            jax.ShapeDtypeStruct(ag.shape, ag.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        input_output_aliases={
            1: 1,
            2: 2,
            3: 3
        },
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, chunk, cols), x.dtype),
            pltpu.VMEM((chunk, cols), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            has_side_effects=True,
            disable_semaphore_checks=disable_semaphore_checks),
    )(x.reshape(num_devices, chunk, cols), rs, ag, step)
    return out.reshape(rows, cols), rs, ag, step


def make_two_shot_workspace(x_shape, dtype, num_devices: int):
    """Zeroed ``(rs, ag, step)`` for :func:`two_shot_all_reduce_ws`."""
    import jax.numpy as jnp
    rows, cols = x_shape
    chunk = rows // num_devices
    return (jnp.zeros((2, num_devices - 1, chunk, cols),
                      dtype), jnp.zeros((2, num_devices, chunk, cols),
                                        dtype), jnp.zeros((1, ), jnp.int32))


# -----------------------------------------------------------------------------
# Two-shot: 2 hops, 2 * M * (N-1) / N bytes out per device.
# -----------------------------------------------------------------------------


def _two_shot_kernel(x_ref, o_ref, rs_ref, red_ref, local_sem, send_sem,
                     recv_sem, send_sem2, recv_sem2, *, num_devices: int,
                     axis_name: str, double_barrier: bool):
    # x_ref / o_ref: (num_devices, chunk_rows, cols); rs_ref: (N-1, chunk, cols)
    my_id = lax.axis_index(axis_name)
    _full_barrier(my_id, num_devices, double_barrier)

    # Phase 1 -- reduce-scatter. Push the chunk that peer ``+i`` owns to it;
    # by the same static rule as one-shot it lands in that peer's slot i-1.
    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[peer],
            dst_ref=rs_ref.at[i - 1],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(peer, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()

    acc = x_ref[my_id]
    for i in range(num_devices - 1):
        acc = acc + rs_ref[i]
    red_ref[...] = acc

    # Phase 2 -- all-gather. My reduced chunk goes into slot ``my_id`` of every
    # device's output, mine included.
    local = pltpu.make_async_copy(src_ref=red_ref,
                                  dst_ref=o_ref.at[my_id],
                                  sem=local_sem)
    local.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_ref,
            dst_ref=o_ref.at[my_id],
            send_sem=send_sem2,
            recv_sem=recv_sem2,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    local.wait()
    for copy in copies:
        copy.wait()


def two_shot_all_reduce(x: jax.Array,
                        *,
                        axis_name: str,
                        num_devices: int,
                        collective_id: int = 0,
                        double_barrier: bool = True) -> jax.Array:
    """Direct reduce-scatter + all-gather all-reduce. Call under shard_map.

    ``x`` must be 2-D with ``x.shape[0] % num_devices == 0``; the leading axis
    is what gets scattered.

    Returns:
      The sum of ``x`` across the mesh axis, same shape and dtype.
    """
    if num_devices < 2:
        return x
    if x.ndim != 2:
        raise ValueError(
            f"two_shot_all_reduce expects a 2-D shard, got {x.shape}")
    rows, cols = x.shape
    if rows % num_devices != 0:
        raise ValueError(
            f"two_shot_all_reduce needs rows ({rows}) divisible by "
            f"num_devices ({num_devices})")
    chunk = rows // num_devices

    kernel = functools.partial(_two_shot_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               double_barrier=double_barrier)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_devices, chunk, cols), x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, chunk, cols), x.dtype),
            pltpu.VMEM((chunk, cols), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id,
                                             has_side_effects=True),
    )(x.reshape(num_devices, chunk, cols))
    return out.reshape(rows, cols)
