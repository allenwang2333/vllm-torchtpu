# SPDX-License-Identifier: Apache-2.0
"""Low-latency all-reduce Pallas kernels for small (decode-sized) payloads."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = [
    "two_shot_all_reduce",
    "two_shot_all_reduce_natural",
    "two_shot_all_reduce_stateless",
    "two_shot_all_reduce_static",
    "make_two_shot_workspace",
    "make_two_shot_workspace_natural",
    "make_two_shot_workspace_stateless",
    "make_two_shot_workspace_static",
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


def _two_shot_kernel(x_ref,
                     rs_in_ref,
                     ag_in_ref,
                     step_in_ref,
                     o_ref,
                     rs_ref,
                     ag_ref,
                     step_ref,
                     recv_vmem,
                     red_vmem,
                     local_sem,
                     s1_send,
                     s1_recv,
                     s2_send,
                     s2_recv,
                     *,
                     num_devices: int,
                     axis_name: str,
                     steady_barrier: bool = True):
    del rs_in_ref, ag_in_ref  # aliased onto the output views
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)

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

    # Overlapped barrier: round k writes slot k%2, last read at round k-2, and
    # passing round k-1's barrier proves every peer finished round k-2. It also
    # covers phase 2, whose sends come later still.
    # See two_shot_all_reduce's `steady_barrier` argument for why the
    # steady-state barrier is removable.
    # The argument is if anything stronger here: phase 2's sends come after the
    # phase-1 recv wait, so by the time they are issued every peer is known to
    # have entered this round, not merely the previous one.
    if steady_barrier:

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


def two_shot_all_reduce(x: jax.Array,
                        rs: jax.Array,
                        ag: jax.Array,
                        step: jax.Array,
                        *,
                        axis_name: str,
                        num_devices: int,
                        collective_id: int = 0,
                        has_side_effects: bool = True,
                        steady_barrier: bool = True):
    """Two-shot all-reduce over a caller-owned persistent workspace.

    Moves both the reduce-scatter staging *and* the all-gather destination out
    of kernel-owned memory, so the barrier can move after the phase-1 sends and
    overlap the transfer. This is the kernel K3 runs.

    The all-gather is the extra piece here -- a stateless two-shot writes
    peers' *output* buffers directly, which are as short-lived as scratch, so
    they had to move into the workspace too and be copied out at the end.

    ``x`` must be 2-D with ``x.shape[0] % num_devices == 0``.

    Compiles with Mosaic's semaphore checks off, which is not optional and so
    is not exposed. Overlapping the barrier with the sends means a peer that has
    passed round k's barrier issues its round k+1 sends before its own round k+1
    barrier, so those DMAs bump our ``recv_sem[(k+1)%2]`` while we are still
    finishing round k. The data is fine -- that is the other slot -- but Mosaic
    asserts every scratch semaphore is zero on kernel exit, and this design
    deliberately violates that. Leaving the check on kills the slice with
    ``SLICE_FAILURE_SW_INJECT_ERROR``; the driver log under
    ``/tmp/tpu_logs_<user>/`` names the semaphore. The early counts are consumed
    by the next round's wait, verified exact against ``psum`` in float32.

    ``has_side_effects`` must be False to compile through torch_tpu.

      steady_barrier: Keep the barrier in rounds past round 0. Setting it False
        removes the last synchronization primitive from the steady state, on the
        argument that the all-to-all recv wait already proves every peer
        finished the round that last read this slot -- the TPU analogue of a
        barrier-free (Lamport) NVLink collective, where the buffer-reuse
        guarantee comes from buffer rotation rather than a barrier. Measured
        cost of what this removes: ~2.9 us per barrier at 32 devices
        (``scripts/debug/two_shot_ablation.py``, jobs 118/119), against a
        near-to-far ICI distance spread of only ~3.5 us
        (``scripts/debug/probe_pairwise_latency.py``, job 250) -- so this is the
        larger of the two terms available to attack.

        Only sound while every rank calls the collective the same number of
        times in the same order; rank-divergent control flow makes skew
        unbounded, and the failure is silent: a peer more than one round
        behind has its workspace overwritten while still reading it, which
        produces wrong numbers rather than an error. The slot arithmetic gives
        exactly one round of slack, so validate the skew margin for your call
        pattern before trusting it. See the kernel body for the full argument.

    Returns:
      ``(out, rs, ag, step)`` -- thread the last three into the next call.
    """
    if num_devices < 2:
        return x, rs, ag, step
    if x.ndim != 2:
        raise ValueError(f"two_shot_all_reduce expects 2-D, got {x.shape}")
    rows, cols = x.shape
    if rows % num_devices != 0:
        raise ValueError(f"rows ({rows}) must be divisible by num_devices "
                         f"({num_devices})")
    chunk = rows // num_devices

    kernel = functools.partial(_two_shot_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               steady_barrier=steady_barrier)
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
        compiler_params=pltpu.CompilerParams(collective_id=collective_id,
                                             has_side_effects=has_side_effects,
                                             disable_semaphore_checks=True),
    )(x.reshape(num_devices, chunk, cols), rs, ag, step)
    return out.reshape(rows, cols), rs, ag, step


def make_two_shot_workspace(x_shape, dtype, num_devices: int):
    """Zeroed ``(rs, ag, step)`` for :func:`two_shot_all_reduce`."""
    rows, cols = x_shape
    chunk = rows // num_devices
    return (jnp.zeros((2, num_devices - 1, chunk, cols),
                      dtype), jnp.zeros((2, num_devices, chunk, cols),
                                        dtype), jnp.zeros((1, ), jnp.int32))


def _two_shot_natural_kernel(x_ref,
                             rs_ref,
                             ag_ref,
                             step_ref,
                             o_ref,
                             recv_vmem,
                             red_vmem,
                             local_sem,
                             s1_send,
                             s1_recv,
                             s2_send,
                             s2_recv,
                             *,
                             num_devices: int,
                             axis_name: str,
                             per_row: int,
                             steady_barrier: bool = True):
    """Two-shot all-reduce on the payload's natural [tokens, width] layout.

    Same protocol as ``_two_shot_kernel``, but the payload keeps its model
    layout end to end and the workspaces are write-only scratch the kernel
    mutates in place (they are pallas *inputs* with no aliased outputs), so
    the XLA module around this call is a bare custom-call -- no slice, no
    update, no reshape relayout, no writeback.

    Chunk d of the payload is the contiguous w-element run
    ``flat[d*w:(d+1)*w]`` -- exactly the chunk ``two_shot_all_reduce`` computes
    on the flattened view -- addressed here without flattening as
    ``x[d // per_row, d % per_row]`` on the 4-D view ``[T, per_row, 1, w]``.
    Because w is a multiple of the 128-wide lane tile, every transfer stays
    lane-aligned (a width-7168 payload over 32 devices gives w = 224*T
    columns: 7 whole tiles at T=4), which the column-equal split
    (7168/32 = 224 = 1.75 tiles) would not.

    The views: x_ref/o_ref are [T, per_row, 1, w]; rs_ref is
    [2, num_devices-1, 1, w]; ag_ref is [2, T, per_row, 1, w], so the final
    gather is one whole-payload DMA, as in the flat kernel. The size-1 dims
    are not decorative: Mosaic refuses to slice a tiled dim that has real
    extent (its memref_slice wants tile-aligned sizes), and the 4-D spelling
    keeps every dynamic index on untiled dims. Their squeeze at the XLA
    boundary is the one wrinkle: XLA lowers [T,pr,1,w] -> [T,pr,w] as a
    reduce, not a bitcast.
    """
    my_id = lax.axis_index(axis_name)
    step = step_ref[0]
    slot = lax.rem(step, 2)

    @pl.when(step == 0)
    def _():
        _full_barrier(my_id, num_devices, double_barrier=True)

    # Phase 1 -- reduce-scatter. Push the chunk peer ``+i`` owns to it.
    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[peer // per_row, peer % per_row],
            dst_ref=rs_ref.at[slot, i - 1],
            send_sem=s1_send.at[slot],
            recv_sem=s1_recv.at[slot],
            device_id=(peer, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)

    if steady_barrier:

        @pl.when(step != 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    for copy in copies:
        copy.wait()

    # The workspace buffers persist across launches, so their addresses are
    # stable for peers' remote copies; stage this device's slots to VMEM.
    local = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    local.start()
    local.wait()

    acc = x_ref[my_id // per_row, my_id % per_row]
    for i in range(num_devices - 1):
        acc = acc + recv_vmem[i]
    red_vmem[...] = acc

    # Phase 2 -- all-gather the reduced chunk into its natural position of
    # every device's output assembly buffer, our own included.
    local_ag = pltpu.make_async_copy(src_ref=red_vmem,
                                     dst_ref=ag_ref.at[slot, my_id // per_row,
                                                       my_id % per_row],
                                     sem=local_sem)
    local_ag.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_vmem,
            dst_ref=ag_ref.at[slot, my_id // per_row, my_id % per_row],
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


def two_shot_all_reduce_natural(x: jax.Array,
                                rs: jax.Array,
                                ag: jax.Array,
                                step: jax.Array,
                                *,
                                axis_name: str,
                                num_devices: int,
                                collective_id: int = 0,
                                has_side_effects: bool = True,
                                steady_barrier: bool = True):
    """``two_shot_all_reduce`` on the payload's natural ``[tokens, width]``.

    ``x`` is 4-D ``[tokens, per_row, 1, w]`` -- the natural [tokens, width]
    viewed so that chunk d (the contiguous w elements device d reduces) is
    ``x[d // per_row, d % per_row]``. ``w`` must be a multiple of 128 (lane
    tile) and ``width % w == 0`` must hold; the caller guarantees both by
    construction (see distributed/low_latency_all_reduce.py).

    ``rs``, ``ag`` and ``step`` are persistent caller-owned scratch that the
    kernel mutates IN PLACE through the input refs; they are deliberately NOT
    returned. The contents of ``rs``/``ag`` are fully rewritten before every
    read within a round, so their values carry no state -- but their buffer
    identity and the ``step`` counter do (slot parity across rounds, including
    rounds in different program executions). Nothing at the XLA level may
    copy them per call, which is the entire point: see the module docstring
    of distributed/low_latency_all_reduce.py for the glue this avoids.

    Returns just the reduced payload, same shape as ``x``.
    """
    if num_devices < 2:
        return x
    if x.ndim != 4:
        raise ValueError(
            f"two_shot_all_reduce_natural expects the 4-D view, got {x.shape}")
    tokens, per_row, one, w = x.shape
    if one != 1:
        raise ValueError(f"bad natural view {x.shape}")
    width = per_row * w
    numel = tokens * width
    if numel % num_devices != 0 or numel // num_devices != w:
        raise ValueError(f"chunk width {w} does not cover {numel} elements "
                         f"over {num_devices} devices")
    if w % 128 != 0:
        raise ValueError(f"chunk width {w} is not lane-aligned (128)")

    kernel = functools.partial(_two_shot_natural_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               per_row=per_row,
                               steady_barrier=steady_barrier)
    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(x.shape, x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, 1, w), x.dtype),
            pltpu.VMEM((1, w), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id,
                                             has_side_effects=has_side_effects,
                                             disable_semaphore_checks=True),
    )(x, rs, ag, step)
    return out


def make_two_shot_workspace_natural(x_shape, dtype, num_devices: int):
    """Zeroed ``(rs, ag, step)`` for :func:`two_shot_all_reduce_natural`.

    ``x_shape`` is the 2-D natural ``[tokens, width]``; the workspaces come
    out in the viewed shapes the kernel indexes.
    """
    tokens, width = x_shape
    w = tokens * width // num_devices
    per_row = width // w
    return (jnp.zeros((2, num_devices - 1, 1, w),
                      dtype), jnp.zeros((2, tokens, per_row, 1, w),
                                        dtype), jnp.zeros((1, ), jnp.int32))


def _two_shot_stateless_kernel(x_ref, rs_ref, ag_ref, o_ref, recv_vmem,
                               red_vmem, local_sem, s1_send, s1_recv, s2_send,
                               s2_recv, *, num_devices: int, axis_name: str,
                               per_row: int):
    """Two-shot all-reduce with a single workspace slot and no state.

    Same protocol as ``_two_shot_natural_kernel`` minus the slot rotation and
    step counter. Statelessness is not a simplification luxury here but the
    only correct option: pallas custom-call operands are *staged* (copied into
    fresh VMEM/SMEM buffers by XLA) at every call, so nothing the kernel
    writes to them survives into the next call's operand -- measured on
    device: a step counter written through the input ref never advances in
    the persistent buffer.

    Safety of a single slot without a step counter: the full barrier runs
    before any send of every call. A peer can only start writing our
    workspace for call k+1 after we arrive at call k+1's barrier, and
    arriving there means our call k finished -- its reads included (same
    device, program order). Every write to us is consumed in the round that
    received it (the recv waits cover all senders), so nothing is in flight
    across a call boundary either. This holds whether XLA shares the staged
    buffers across calls or re-stages per call.

    The views match ``_two_shot_natural_kernel``: x_ref/o_ref are
    [T, per_row, 1, w], rs_ref [num_devices-1, 1, w], ag_ref
    [T, per_row, 1, w].
    """
    my_id = lax.axis_index(axis_name)

    _full_barrier(my_id, num_devices, double_barrier=True)

    # Chunk coordinates. per_row is a power of two on every wired bucket, so
    # use shift/mask: scalar integer division costs tens of cycles each on
    # the TPU scalar unit, and 62 of them per call showed up as ~3us against
    # the flat kernel (which indexes chunks by a single dynamic dim).
    lg = per_row.bit_length() - 1
    if per_row == 2**lg:

        def rc(d):
            return d >> lg, d & (per_row - 1)
    else:

        def rc(d):
            return d // per_row, d % per_row

    # Phase 1 -- reduce-scatter. Push the chunk peer ``+i`` owns to it.
    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        prow, pcol = rc(peer)
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[prow, pcol],
            dst_ref=rs_ref.at[i - 1],
            send_sem=s1_send,
            recv_sem=s1_recv,
            device_id=(peer, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()

    local = pltpu.make_async_copy(src_ref=rs_ref,
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    local.start()
    local.wait()

    my_row, my_col = rc(my_id)
    acc = x_ref[my_row, my_col]
    for i in range(num_devices - 1):
        acc = acc + recv_vmem[i]
    red_vmem[...] = acc

    # Phase 2 -- all-gather the reduced chunk into its natural position of
    # every device's output assembly buffer, our own included.
    local_ag = pltpu.make_async_copy(src_ref=red_vmem,
                                     dst_ref=ag_ref.at[my_row, my_col],
                                     sem=local_sem)
    local_ag.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_vmem,
            dst_ref=ag_ref.at[my_row, my_col],
            send_sem=s2_send,
            recv_sem=s2_recv,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    local_ag.wait()
    for copy in copies:
        copy.wait()

    out_copy = pltpu.make_async_copy(src_ref=ag_ref,
                                     dst_ref=o_ref,
                                     sem=local_sem)
    out_copy.start()
    out_copy.wait()


def two_shot_all_reduce_stateless(x: jax.Array,
                                  rs: jax.Array,
                                  ag: jax.Array,
                                  *,
                                  axis_name: str,
                                  num_devices: int,
                                  collective_id: int = 0,
                                  has_side_effects: bool = True):
    """Stateless ``two_shot_all_reduce_natural``: no slot rotation, no step.

    Same arguments and views minus ``step``; the workspace is two tensors
    (rs, ag) treated as per-call scratch. See ``_two_shot_stateless_kernel``
    for why a single slot is safe with a full barrier per call, and why no
    cross-call state can survive staging anyway.

    Returns just the reduced payload, same shape as ``x``.
    """
    if num_devices < 2:
        return x
    if x.ndim != 4:
        raise ValueError(
            f"two_shot_all_reduce_stateless expects the 4-D view, got "
            f"{x.shape}")
    tokens, per_row, one, w = x.shape
    if one != 1:
        raise ValueError(f"bad natural view {x.shape}")
    width = per_row * w
    numel = tokens * width
    if numel % num_devices != 0 or numel // num_devices != w:
        raise ValueError(f"chunk width {w} does not cover {numel} elements "
                         f"over {num_devices} devices")
    if w % 128 != 0:
        raise ValueError(f"chunk width {w} is not lane-aligned (128)")

    kernel = functools.partial(_two_shot_stateless_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               per_row=per_row)
    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(x.shape, x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, 1, w), x.dtype),
            pltpu.VMEM((1, w), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id,
                                             has_side_effects=has_side_effects,
                                             disable_semaphore_checks=True),
    )(x, rs, ag)
    return out


def make_two_shot_workspace_stateless(x_shape, dtype, num_devices: int):
    """Zeroed ``(rs, ag)`` for :func:`two_shot_all_reduce_stateless`.

    ``x_shape`` is the 2-D natural ``[tokens, width]``; the workspaces come
    out in the viewed shapes the kernel indexes.
    """
    tokens, width = x_shape
    w = tokens * width // num_devices
    per_row = width // w
    return (jnp.zeros((num_devices - 1, 1, w),
                      dtype), jnp.zeros((tokens, per_row, 1, w), dtype))


def _two_shot_static_kernel(x_ref, rs_ref, ag_ref, o_ref, recv_vmem, red_vmem,
                            local_sem, s1_send, s1_recv, s2_send, s2_recv, *,
                            num_devices: int, axis_name: str, per_row: int,
                            slot: int, barrier_first: int):
    """Two-shot all-reduce with a *statically assigned* workspace slot.

    Same protocol as ``_two_shot_kernel`` but the slot rotation is driven by a
    compile-time ``slot`` instead of a device-side step counter: the caller
    (the compiled model graph) is a straight-line sequence of callsites, so
    the call index -- and hence the slot -- is known at compile time. This
    replaces the step counter, which cannot work on this stack anyway: XLA
    stages custom-call operands into fresh VMEM/SMEM buffers per call, so a
    counter written through its input ref never reaches the persistent
    buffer (measured on device).

    Safety, mirroring the original overlapped-barrier argument. Call k uses
    slot k%2 (the caller alternates). The hazard is a peer's call-k+1 write
    landing in our slot (k+1)%2 while our call k is still reading it -- never
    happens, different slot. A peer two calls behind (reading slot k%2) is
    excluded by call k-1's barrier: our call-k sends are issued only after we
    passed our call-k-1 barrier, which every peer must have arrived at, and
    arriving means its call k-2 finished. The step *boundary* (last layer of
    one execution vs first layer of the next -- with 93 layers both would use
    slot 0) is covered by ``barrier_first``: the first callsite of every
    execution barriers before any send, so no write for the new execution can
    start until every peer finished the previous one. Works whether XLA
    shares the staged workspace buffers across calls or re-stages per call
    (per-call staging trivializes every hazard).

    The views match ``_two_shot_natural_kernel`` plus the slot dim back:
    rs_ref is [2, num_devices-1, 1, w] and ag_ref [2, T, per_row, 1, w];
    x_ref/o_ref are [T, per_row, 1, w].
    """
    my_id = lax.axis_index(axis_name)

    if barrier_first:
        _full_barrier(my_id, num_devices, double_barrier=True)

    # Phase 1 -- reduce-scatter. Push the chunk peer ``+i`` owns to it. The
    # barrier overlaps the in-flight sends on non-first calls.
    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[peer // per_row, peer % per_row],
            dst_ref=rs_ref.at[slot, i - 1],
            send_sem=s1_send.at[slot],
            recv_sem=s1_recv.at[slot],
            device_id=(peer, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)

    if not barrier_first:
        _full_barrier(my_id, num_devices, double_barrier=True)

    for copy in copies:
        copy.wait()

    local = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    local.start()
    local.wait()

    acc = x_ref[my_id // per_row, my_id % per_row]
    for i in range(num_devices - 1):
        acc = acc + recv_vmem[i]
    red_vmem[...] = acc

    # Phase 2 -- all-gather the reduced chunk into its natural position of
    # every device's output assembly buffer, our own included.
    local_ag = pltpu.make_async_copy(src_ref=red_vmem,
                                     dst_ref=ag_ref.at[slot, my_id // per_row,
                                                       my_id % per_row],
                                     sem=local_sem)
    local_ag.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_vmem,
            dst_ref=ag_ref.at[slot, my_id // per_row, my_id % per_row],
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


def two_shot_all_reduce_static(x: jax.Array,
                               rs: jax.Array,
                               ag: jax.Array,
                               *,
                               slot: int,
                               barrier_first: int,
                               axis_name: str,
                               num_devices: int,
                               collective_id: int = 0,
                               has_side_effects: bool = True):
    """``two_shot_all_reduce_natural`` with a static slot and no step counter.

    ``slot`` (0 or 1) must alternate between consecutive calls in the
    execution order and ``barrier_first`` (0/1) must be set on the first call
    of every execution -- both are compile-time properties of the callsite,
    so this costs no runtime state. Both are plain ints: torch_tpu's custom-op
    bridge drops bool arguments, so nothing in the path may be a bool. See
    ``_two_shot_static_kernel`` for the safety argument.

    ``rs``/``ag`` are per-call scratch with two slots; returns just the
    reduced payload, same shape as ``x``.
    """
    if num_devices < 2:
        return x
    if x.ndim != 4:
        raise ValueError(
            f"two_shot_all_reduce_static expects the 4-D view, got {x.shape}")
    tokens, per_row, one, w = x.shape
    if one != 1:
        raise ValueError(f"bad natural view {x.shape}")
    width = per_row * w
    numel = tokens * width
    if numel % num_devices != 0 or numel // num_devices != w:
        raise ValueError(f"chunk width {w} does not cover {numel} elements "
                         f"over {num_devices} devices")
    if w % 128 != 0:
        raise ValueError(f"chunk width {w} is not lane-aligned (128)")
    if slot not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot}")

    kernel = functools.partial(_two_shot_static_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               per_row=per_row,
                               slot=slot,
                               barrier_first=barrier_first)
    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(x.shape, x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, 1, w), x.dtype),
            pltpu.VMEM((1, w), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id,
                                             has_side_effects=has_side_effects,
                                             disable_semaphore_checks=True),
    )(x, rs, ag)
    return out


def make_two_shot_workspace_static(x_shape, dtype, num_devices: int):
    """Zeroed ``(rs, ag)`` for :func:`two_shot_all_reduce_static` (2 slots)."""
    tokens, width = x_shape
    w = tokens * width // num_devices
    per_row = width // w
    return (jnp.zeros((2, num_devices - 1, 1, w),
                      dtype), jnp.zeros((2, tokens, per_row, 1, w), dtype))
