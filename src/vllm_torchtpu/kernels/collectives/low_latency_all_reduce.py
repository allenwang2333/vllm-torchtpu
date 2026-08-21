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
one-shot / two-shot split for NVLink).

``two_shot_all_reduce_ws``
    Reduce-scatter then all-gather, each phase a *direct* (non-ring)
    all-to-all round rather than ``N-1`` ring steps: **2 hops** and
    ``2 * M * (N-1) / N`` bytes out per device.  Runs over a caller-owned
    persistent workspace, which is what lets the barrier sit *after* the
    phase-1 sends so it overlaps the transfer -- worth ~2 us at 8 devices and
    ~3 us at 32.  The all-gather destination has to live in the workspace too;
    writing peers' *output* buffers would target memory as short-lived as
    scratch.  This is the only kernel K3 invokes: 187 calls per decode step,
    95 post-attention and 92 MoE pairs.

``two_shot_all_reduce_variadic_ws``
    The same collective over two operands in one launch, so K3's latent-MoE
    pair needs no ``torch.cat`` beforehand.  Off by default: it corrupts
    deterministically when the two operands land on equal per-device chunk
    widths at 32 devices, guarded by a ``ValueError``, cause unexplained.

The workspace must be threaded call-to-call; see ``make_two_shot_workspace``
and ``make_two_shot_variadic_workspace``.  ``numel % num_devices == 0`` is the
real constraint -- the dispatcher reshapes to ``(num_devices, numel/N)`` rather
than scattering along rows, so decode's row count does not matter.

An earlier revision of this module also carried ``one_shot_all_reduce`` (a
single all-to-all broadcast round: 1 hop, ``M * (N-1)`` bytes out, faster than
two-shot below ~13 KiB at 32 devices and ~32 KiB at 8), a two-operand
``two_shot_all_reduce_pair_ws``, and stateless non-workspace versions of both
shapes.  None of them were on K3's serving path -- one_shot was built by the
warmup for the ``(1,3584)`` latent and then never invoked, because the concat
path folds that shape into the ``(1,10752)`` pair -- so they were removed rather
than carried.  They are recoverable from the ``llar-all-variants`` tag, and the
measurements that ranked them are in ``docs/low_latency_allreduce.md``.

All kernels are written for a 1-D device mesh and must be called inside
``shard_map`` over that mesh.  Peer addressing uses ``DeviceIdType.MESH``, so
the mesh axis order determines the physical routing.

Buffer-reuse safety
-------------------
The barrier is what guarantees the destination is *owned by this kernel on the
receiver* when the data lands.  Kernel scratch is only live while the kernel
runs, so an early write would otherwise scribble on VMEM that XLA has handed to
some other op.  That is why the barrier cannot simply be moved after the sends
in a stateless kernel, and why this one needs a persistent workspace to do so.

The double barrier closes a second race that a single barrier leaves open: a
fast peer could exit, re-enter on the *next* invocation, and signal the
(identically numbered) barrier semaphore again, releasing us while a slower peer
is still draining the previous call's buffers.  Same idiom as
:func:`vllm_torchtpu.kernels.collectives.util.local_barrier`, widened from
left/right neighbors to the full mesh.

``steady_barrier=False`` removes the barrier from every round past round 0; see
that argument's docstring for why the data dependency alone is then sufficient.

Why the kernels disable semaphore checks
----------------------------------------
``disable_semaphore_checks=True`` is required.  Overlapping the barrier with the
sends means a peer that has passed round k's barrier issues its round k+1 sends
before its own round k+1 barrier, so those DMAs bump our ``recv_sem[(k+1)%2]``
while we are still finishing round k.  The data is fine -- that is the other
slot -- but Mosaic asserts every scratch semaphore is zero on kernel exit, and
this design deliberately violates that.  Leaving the check on kills the slice
with ``SLICE_FAILURE_SW_INJECT_ERROR``; the driver log under
``/tmp/tpu_logs_<user>/`` names the semaphore.  The early counts are consumed by
the next round's wait, verified exact against ``psum`` in float32.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = [
    "two_shot_all_reduce_ws",
    "make_two_shot_workspace",
    "two_shot_all_reduce_variadic_ws",
    "make_two_shot_variadic_workspace",
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


def _two_shot_ws_kernel(x_ref,
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
                        barrier_first: bool,
                        steady_barrier: bool = True):
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

    # Overlapped barrier: round k writes slot k%2, last read at round k-2, and
    # passing round k-1's barrier proves every peer finished round k-2. It also
    # covers phase 2, whose sends come later still.
    # See two_shot_all_reduce_ws's `steady_barrier` argument for why the
    # steady-state barrier is removable.
    # The argument is if anything stronger here: phase 2's sends come after the
    # phase-1 recv wait, so by the time they are issued every peer is known to
    # have entered this round, not merely the previous one.
    if not barrier_first and steady_barrier:

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
                           disable_semaphore_checks: bool = True,
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
    ``disable_semaphore_checks`` must stay True and ``has_side_effects`` must
    be False to compile through torch_tpu; see this module's docstring for
    both.

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
        unbounded. Validate with ``scripts/allreduce_ws_stress.py
        --no-steady-barrier`` before trusting it, and see the kernel body for
        the full argument.

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
                               barrier_first=barrier_first,
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
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            has_side_effects=has_side_effects,
            disable_semaphore_checks=disable_semaphore_checks),
    )(x.reshape(num_devices, chunk, cols), rs, ag, step)
    return out.reshape(rows, cols), rs, ag, step


# -----------------------------------------------------------------------------
# Variadic two-shot: two operands, one kernel, no join.
# -----------------------------------------------------------------------------


def _two_shot_variadic_ws_kernel(a_ref, b_ref, rsa_in, rsb_in, aga_in, agb_in,
                                 step_in_ref, oa_ref, ob_ref, rsa_ref, rsb_ref,
                                 aga_ref, agb_ref, step_ref, rva, rvb, reda,
                                 redb, local_sem, s1a_s, s1a_r, s1b_s, s1b_r,
                                 s2a_s, s2a_r, s2b_s, s2b_r, *,
                                 num_devices: int, axis_name: str,
                                 barrier_first: bool, steady_barrier: bool):
    """Reduce two independent operands in one launch, without concatenating.

    K3's latent-MoE site produces ``(1, 3584)`` and ``(1, 7168)`` and needs both
    reduced. Doing it as two collectives costs two launches, which measured
    ~2x one collective (32.60 vs 17.35 us at 32 devices, job 265) -- hence the
    ``torch.cat`` that has been in production. This kernel gets the single launch
    without the join: each operand keeps its own width and its own DMAs, and they
    share the barrier and the launch.

    Every ref is 3-D ``(num_devices, chunk, cols)``. That is mandatory, not
    stylistic: both dimensions of a *2-D* VMEM ref are tiled (bf16 as ``(8,128)``
    with ``(2,1)`` packing), so extracting one row fails to lower even at a
    static index --

        memref_slice(memref<7x448xbf16, tiled<(8,128)(2,1)>>) -> 1x448
        "size at dimension 0 is divisible by the tile dimension 8"

    -- which is what sank the first attempt at this design and produced the
    mistaken conclusion (``scripts/debug/probe_variadic_dma.py``) that no
    per-peer offset can lower at all. That probe used 1-D operands sliced with
    ``pl.ds()``, where bf16 tiles at 16x128 = 2048 elements, so its "aligned"
    arm was never aligned. Putting the peer index on an untiled leading
    dimension is the fix, and it is the same reason
    :func:`two_shot_all_reduce_ws` reshapes before its ``pallas_call``.

    The two operands' phase-1 sends are issued together and waited together, so
    their transfers overlap rather than serialize; likewise phase 2.
    """
    del rsa_in, rsb_in, aga_in, agb_in  # aliased onto the output views
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)

    if barrier_first:
        _full_barrier(my_id, num_devices, double_barrier=True)
    else:

        @pl.when(step == 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    # Phase 1 -- reduce-scatter both operands. Interleaved so both are in
    # flight at once.
    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        for src, dst, ss, rs in ((a_ref, rsa_ref, s1a_s, s1a_r),
                                 (b_ref, rsb_ref, s1b_s, s1b_r)):
            copy = pltpu.make_async_remote_copy(
                src_ref=src.at[peer],
                dst_ref=dst.at[slot, i - 1],
                send_sem=ss.at[slot],
                recv_sem=rs.at[slot],
                device_id=(peer, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)

    # See two_shot_all_reduce_ws's `steady_barrier` argument for why the
    # steady-state barrier is removable.
    if not barrier_first and steady_barrier:

        @pl.when(step != 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    for copy in copies:
        copy.wait()

    # Workspaces live in HBM so their addresses survive between kernels; stage
    # each into VMEM to reduce it.
    for ws, vm in ((rsa_ref, rva), (rsb_ref, rvb)):
        local = pltpu.make_async_copy(src_ref=ws.at[slot],
                                      dst_ref=vm,
                                      sem=local_sem)
        local.start()
        local.wait()

    acc_a = a_ref[my_id]
    for i in range(num_devices - 1):
        acc_a = acc_a + rva[i]
    reda[...] = acc_a
    acc_b = b_ref[my_id]
    for i in range(num_devices - 1):
        acc_b = acc_b + rvb[i]
    redb[...] = acc_b

    # Phase 2 -- all-gather each reduced chunk into slot ``my_id`` everywhere,
    # our own included.
    copies = []
    for red, ag, ss, rs in ((reda, aga_ref, s2a_s, s2a_r), (redb, agb_ref,
                                                            s2b_s, s2b_r)):
        local = pltpu.make_async_copy(src_ref=red,
                                      dst_ref=ag.at[slot, my_id],
                                      sem=local_sem)
        local.start()
        local.wait()
        for i in range(1, num_devices):
            copy = pltpu.make_async_remote_copy(
                src_ref=red,
                dst_ref=ag.at[slot, my_id],
                send_sem=ss.at[slot],
                recv_sem=rs.at[slot],
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
    for copy in copies:
        copy.wait()

    for ag, out in ((aga_ref, oa_ref), (agb_ref, ob_ref)):
        out_copy = pltpu.make_async_copy(src_ref=ag.at[slot],
                                         dst_ref=out,
                                         sem=local_sem)
        out_copy.start()
        out_copy.wait()
    step_ref[0] = step + 1


def make_two_shot_variadic_workspace(numel_a: int, numel_b: int, dtype,
                                     num_devices: int):
    """Zeroed ``(rsa, rsb, aga, agb, step)`` for the variadic two-shot."""
    if numel_a % num_devices or numel_b % num_devices:
        raise ValueError(
            f"both element counts must divide num_devices ({num_devices}); "
            f"got {numel_a} and {numel_b}")
    ca = numel_a // num_devices
    cb = numel_b // num_devices
    return (jnp.zeros((2, num_devices - 1, 1, ca),
                      dtype), jnp.zeros((2, num_devices - 1, 1, cb), dtype),
            jnp.zeros((2, num_devices, 1, ca),
                      dtype), jnp.zeros((2, num_devices, 1, cb),
                                        dtype), jnp.zeros((1, ), jnp.int32))


def two_shot_all_reduce_variadic_ws(a: jax.Array,
                                    b: jax.Array,
                                    rsa: jax.Array,
                                    rsb: jax.Array,
                                    aga: jax.Array,
                                    agb: jax.Array,
                                    step: jax.Array,
                                    *,
                                    axis_name: str,
                                    num_devices: int,
                                    collective_id: int = 0,
                                    barrier_first: bool = False,
                                    steady_barrier: bool = True,
                                    disable_semaphore_checks: bool = True,
                                    has_side_effects: bool = True):
    """All-reduce two operands in one kernel, without concatenating them.

    Replaces ``torch.cat`` + one :func:`two_shot_all_reduce_ws` at K3's
    latent-MoE site. Measured at 32 devices on the 21 KiB pair (job 265, against
    an in-jit-concat baseline): 16.66 us against 17.35, and against 32.60 for two
    separate collectives.

    Both operands may have any shape whose element count divides
    ``num_devices``; each is reshaped to ``(num_devices, 1, numel/num_devices)``
    internally and restored on the way out. **Equal per-device chunk widths are
    rejected** -- see the ValueError below for the measured failure and why it is
    a hard error rather than a warning. See
    this module's docstring for ``disable_semaphore_checks`` and
    ``has_side_effects``, and :func:`two_shot_all_reduce_ws` for what
    ``steady_barrier`` trades.

    Returns:
      ``(out_a, out_b, rsa, rsb, aga, agb, step)`` -- thread the last five into
      the next call.
    """
    if num_devices < 2:
        return a, b, rsa, rsb, aga, agb, step
    numel_a = math.prod(a.shape)
    numel_b = math.prod(b.shape)
    if numel_a % num_devices or numel_b % num_devices:
        raise ValueError(
            f"both element counts must divide num_devices ({num_devices}); "
            f"got {numel_a} and {numel_b}")
    ca = numel_a // num_devices
    cb = numel_b // num_devices
    if ca == cb:
        raise ValueError(
            f"equal per-device chunk widths ({ca}) are not supported: the "
            "kernel returns partially corrupted results for both operands. "
            "Measured at 32 devices across 4 processes with a=1.0 and b=2.0, "
            "every chunk of both outputs came back containing a mix of 32.0 "
            "and 64.0 -- interleaved within chunks, not one operand wholesale "
            "receiving the other's data. It is deterministic, reproduces for "
            "every equal width tried (64, 112, 128 and 256 element chunks), "
            "does not reproduce at 8 devices in one process, and survives both "
            "distinct collective_ids and optimization_barriers on the "
            "workspaces. The likely mechanism is a DMA wait being satisfied by "
            "the other operand's arrival, which is only possible when the byte "
            "counts match -- but that is unproven, which is exactly why this "
            "raises instead of warning. K3 is unaffected (112 against 224). "
            "See scripts/debug/probe_variadic_bug.py.")

    kernel = functools.partial(_two_shot_variadic_ws_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               barrier_first=barrier_first,
                               steady_barrier=steady_barrier)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    oa, ob, rsa, rsb, aga, agb, step = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_devices, 1, ca), a.dtype),
            jax.ShapeDtypeStruct((num_devices, 1, cb), b.dtype),
            jax.ShapeDtypeStruct(rsa.shape, rsa.dtype),
            jax.ShapeDtypeStruct(rsb.shape, rsb.dtype),
            jax.ShapeDtypeStruct(aga.shape, aga.dtype),
            jax.ShapeDtypeStruct(agb.shape, agb.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ),
        in_specs=[
            vmem_spec, vmem_spec, any_spec, any_spec, any_spec, any_spec,
            smem_spec
        ],
        out_specs=[
            vmem_spec, vmem_spec, any_spec, any_spec, any_spec, any_spec,
            smem_spec
        ],
        # Alias every workspace and the counter so the kernel updates them in
        # place and their buffer identity is stable across calls.
        input_output_aliases={
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6
        },
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, 1, ca), a.dtype),
            pltpu.VMEM((num_devices - 1, 1, cb), b.dtype),
            pltpu.VMEM((1, ca), a.dtype),
            pltpu.VMEM((1, cb), b.dtype),
            pltpu.SemaphoreType.DMA,
        ] + [pltpu.SemaphoreType.DMA((2, ))] * 8,
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            has_side_effects=has_side_effects,
            disable_semaphore_checks=disable_semaphore_checks),
    )(a.reshape(num_devices, 1, ca), b.reshape(num_devices, 1, cb), rsa, rsb,
      aga, agb, step)
    return (oa.reshape(a.shape), ob.reshape(b.shape), rsa, rsb, aga, agb, step)


def make_two_shot_workspace(x_shape, dtype, num_devices: int):
    """Zeroed ``(rs, ag, step)`` for :func:`two_shot_all_reduce_ws`."""
    rows, cols = x_shape
    chunk = rows // num_devices
    return (jnp.zeros((2, num_devices - 1, chunk, cols),
                      dtype), jnp.zeros((2, num_devices, chunk, cols),
                                        dtype), jnp.zeros((1, ), jnp.int32))


# Lane width of a TPU vector register. A VMEM store whose *start* is not a
# multiple of this is the case Mosaic lowers badly or refuses, so the pair
# kernel below pads each operand's per-device slice up to it.
_LANE = 128
