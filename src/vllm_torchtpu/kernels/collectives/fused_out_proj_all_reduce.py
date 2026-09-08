# SPDX-License-Identifier: Apache-2.0
"""Fused out-projection + two-shot all-reduce for T=1 decode payloads.

At decode batch 1 the o_proj row-parallel reduce is a [1, width] all-reduce
whose cost is hops, not bytes, and the matmul that produces the partials is
latency-bound too -- so this kernel does both in one custom call: each rank
computes its [1, width] partial from its activation shard [1, k_shard] and
weight shard [width, k_shard], then runs the two-shot exchange
(reduce-scatter + all-gather) without leaving the kernel.

The exchange protocol is the barrier-free one proven in
scripts/k3/packed_layout_probe.py (variant fullar_nb, job 491):

* Phase 1 (reduce-scatter): every rank pushes each peer the tile that peer
  reduces -- to peer ``rem(my_id + i, N)`` it sends tile
  ``rem(peer, tiles)`` -- 31 async remote copies, sender-side waits, then
  ONE all-to-all signal round on a REGULAR semaphore. The signal round
  replaces a full barrier: sender-side copy waits prove delivery, so the
  signal says "my copies into your buffer have landed". EVERY rank then
  waits for N-1 signals -- ownerless ranks too: a fired-but-unwatched
  semaphore trips the runtime's termination validation ("Semaphore has a
  nonzero value upon exit", K3 warmup, job 558), and
  CompilerParams(disable_semaphore_checks=True) does not cover that
  runtime check. Owners (my_id < tiles) then reduce their tile.
* Phase 2 (all-gather): every rank broadcasts its reduced tile into slot
  ``min(my_id, tiles)`` of every rank's assembly workspace (ownerless ranks
  -- there are ``num_devices - tiles`` of them -- broadcast garbage into the
  trash slot ``tiles``, which is never read, keeping the phase free of
  conditional remote DMA, which deadlocks on this stack), then a second
  signal round gates the final assembly copy out.

Two structural rules, both measured on device in the probe, shape the code:

* The payload is kept PACKED as ``[tiles, 2, 128]`` tiles end to end: that
  is the layout the o_proj dot's accumulator naturally fills at T=1
  (width / 256 tiles of 2x128), and it makes every exchange transfer a whole
  number of 256-element tiles. The squeeze back to [1, width] happens once,
  in the JAX body around the custom call.
* EVERY rank must receive at least one remote copy per phase. Tile-major
  sends (tile d to rank d) leave ranks ``tiles..num_devices-1`` receiving
  zero copies and deadlock at execution (probe jobs 477/478/480-485; the
  zero-receiver mechanism is isolated by jobs 494/495: static src slices are
  fine, zero-receiver ranks hang). The offset-rem pattern above gives every
  rank exactly ``num_devices - 1`` arrivals.

Cross-call safety mirrors two_shot_all_reduce_static: ``slot`` (0/1) must
alternate between consecutive calls in the execution order and
``barrier_first`` must be set on the first call of every execution. The
slot rotation gives one call of slack, and the arr1 wait in call k proves
every peer *completed* call k-1 (a peer's call-k phase-1 signals are issued
only after its call-k sends, and its call-k kernel starts only after its
call-k-1 kernel finished -- same-device ordering via the model's data
dependencies), so a peer two calls behind cannot still be reading the slot
call k+1 writes. ``barrier_first`` closes the execution boundary, where two
consecutive calls (last layer of execution e, first layer of e+1) may share
a slot parity. This protocol is sound only while every rank executes the
same op sequence in the same order -- guaranteed by the opaque-op routing in
distributed/fused_out_proj_all_reduce.py.

Constraint: T=1 only (``x`` is [1, k_shard]), ``width % (2*128) == 0``, and
``tiles = width // 256 <= num_devices`` (28 <= 32 for K3 decode). Larger
token counts need multi-tile ownership, which this kernel does not
implement; the caller routes those to XLA.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = [
    "fused_out_proj_all_reduce",
    "make_fused_out_proj_workspace",
    "PACK",
    "LANES",
]

# The packing of the payload: [tiles, PACK, LANES]. PACK x LANES = 256
# elements is one packed tile -- the granularity the exchange moves.
PACK, LANES = 2, 128


def _full_barrier(my_id, num_devices: int):
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


def _signal_round(sem, my_id, num_devices: int):
    """Signal every peer once (its copies from me have landed)."""
    for i in range(1, num_devices):
        pl.semaphore_signal(
            sem,
            1,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )


def _fused_out_proj_ar_kernel(x_ref, w_ref, rs_ref, ag_ref, o_ref,
                              payload_vmem, recv_vmem, red_vmem, asm_vmem,
                              local_sem, send_sem, recv_sem, arr1, arr2, *,
                              num_devices: int, axis_name: str, tiles: int,
                              slot: int, barrier_first: int,
                              steady_signals: bool):
    my_id = lax.axis_index(axis_name)

    if barrier_first:
        _full_barrier(my_id, num_devices)

    # The o_proj partial, stored packed as [tiles, 2, 128] in VMEM. w is the
    # rank's row-parallel weight shard in the model's layout [k_shard,
    # width] (contracting dim FIRST -- that is how the torch_tpu
    # RowParallelLinear stores it, measured on device: the armed layer's
    # weight is [384, 7168] at TP=32).
    acc = jax.lax.dot_general(x_ref[...],
                              w_ref[...],
                              dimension_numbers=(((1, ), (0, )), ((), ())),
                              preferred_element_type=jnp.float32)
    payload_vmem[...] = acc.astype(o_ref.dtype).reshape(payload_vmem.shape)

    # Phase 1 -- reduce-scatter. Push the tile peer ``+i`` owns to it
    # (offset-rem pattern: every rank receives exactly num_devices-1
    # copies -- tile-major sends leave ownerless ranks at zero receives and
    # deadlock). It lands in that peer's slot ``my_id``; a peer's own slot
    # is never written, so the reduce below can sum all slots (its own
    # contribution comes from payload_vmem).
    copies = []
    for i in range(1, num_devices):
        dst = lax.rem(my_id + i, num_devices)
        tile = lax.rem(dst, tiles)
        copy = pltpu.make_async_remote_copy(
            src_ref=payload_vmem.at[tile],
            dst_ref=rs_ref.at[slot, my_id],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(dst, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    # Delivered before the signal, so a peer that observes N-1 signals may
    # read its receive buffer. The round is UNCONDITIONAL: a fired-but-
    # unwatched semaphore is a hard device fault on this stack ("Semaphore
    # has a nonzero value upon exit from a Mosaic kernel", K3 warmup).
    # The signal round + wait gate the owner's read of its receive buffer.
    # They are UNCONDITIONAL when present: a fired-but-unwatched semaphore
    # is a hard device fault on this stack ("Semaphore has a nonzero value
    # upon exit from a Mosaic kernel", K3 warmup job 558), and for an
    # ownerless rank the wait only means "every peer's phase-1 deliveries
    # landed". With steady_signals=False the sender-side copy waits are the
    # whole protocol (see the wrapper's docstring for the safety argument).
    if steady_signals:
        _signal_round(arr1, my_id, num_devices)
        pl.semaphore_wait(arr1, num_devices - 1)

    # Owners reduce their tile. The local stage copy is the only DMA inside
    # a conditional -- remote DMA inside pl.when deadlocks on this stack
    # (probe job 485), local is fine.
    @pl.when(my_id < tiles)
    def _():
        stage = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                      dst_ref=recv_vmem,
                                      sem=local_sem)
        stage.start()
        stage.wait()
        red = payload_vmem[my_id]
        for j in range(num_devices):
            red = red + recv_vmem[j]
        red_vmem[...] = red

    # Phase 2 -- all-gather. Fully uniform: no remote DMA inside a
    # conditional. Ownerless ranks broadcast their (garbage) red_vmem into
    # the trash slot ``tiles`` of everyone's assembly buffer, never read.
    out_slot = lax.min(my_id, tiles)
    local = pltpu.make_async_copy(src_ref=red_vmem,
                                  dst_ref=ag_ref.at[slot, out_slot],
                                  sem=local_sem)
    local.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_vmem,
            dst_ref=ag_ref.at[slot, out_slot],
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
    # Assembly once every phase-2 tile has landed locally (or, with
    # steady_signals=False, once our own sends prove ours landed).
    if steady_signals:
        _signal_round(arr2, my_id, num_devices)
        pl.semaphore_wait(arr2, num_devices - 1)
    stage = pltpu.make_async_copy(src_ref=ag_ref.at[slot, 0:tiles],
                                  dst_ref=asm_vmem,
                                  sem=local_sem)
    stage.start()
    stage.wait()
    # The result goes out in the natural [1, width] layout -- NOT the packed
    # [tiles, 2, 128] form: the packed custom-call result shifted XLA's
    # global layout assignment and materialized the [7168,384] MoE
    # weight-staging transposes (the Batch W contagion term, ~16 ms/window,
    # jobs 581/593). A boring [1, width] result doesn't propagate.
    o_ref[...] = asm_vmem[...].reshape(o_ref.shape)


def fused_out_proj_all_reduce(x: jax.Array,
                              w: jax.Array,
                              rs: jax.Array,
                              ag: jax.Array,
                              *,
                              slot: int,
                              barrier_first: int,
                              axis_name: str,
                              num_devices: int,
                              collective_id: int = 0,
                              has_side_effects: bool = True,
                              steady_signals: bool = False) -> jax.Array:
    """o_proj partial + two-shot all-reduce in one custom call. T=1 only.

    ``x`` is this rank's activation shard ``[1, k_shard]``; ``w`` is this
    rank's row-parallel weight shard ``[k_shard, width]`` -- contracting dim
    FIRST, the layout the torch_tpu RowParallelLinear actually stores
    (measured on device). Returns the reduced ``[1, width]`` payload,
    identical on every rank.

    ``rs``/``ag`` are caller-owned persistent scratch (see
    ``make_fused_out_proj_workspace``), mutated in place through the input
    refs and never returned. ``slot`` (0/1) must alternate between
    consecutive calls in execution order and ``barrier_first`` (0/1) must be
    set on the first call of every execution; both are compile-time
    properties of the callsite. See the module docstring for the safety
    argument. Both are plain ints: torch_tpu's custom-op bridge drops bool
    arguments, so nothing in the path may be a bool.

    ``collective_id`` is used only when ``barrier_first`` is set -- the
    barrier semaphore needs one, and Mosaic refuses a collective_id on a
    kernel that uses no barrier ("collective_id has to be unspecified or
    None when not using a custom barrier").

    ``steady_signals`` -- keep the per-phase signal rounds (the arrival
    notification for the owner reduce and the assembly). Default False:
    the sender-side copy waits alone prove delivery, and cross-call buffer
    safety comes from the slot rotation + ``barrier_first`` -- the same
    barrier-free (Lamport-style) steady state as
    two_shot_all_reduce(steady_barrier=False), and ~5.6 us/call cheaper at
    the K3 KDA T=1 shape (21.1 -> 15.5 us, bench jobs 578/579). Only sound
    while every rank executes the same op sequence in the same order; rank
    skew beyond one call has its workspace overwritten while still read,
    which is a SILENT wrong-number failure -- same caveat as
    steady_barrier=False.
    """
    if num_devices < 2:
        return x  # caller handles; kept for signature symmetry
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError(
            f"fused_out_proj_all_reduce is T=1 only: x must be [1, k_shard], "
            f"got {x.shape}")
    if w.ndim != 2 or w.shape[0] != x.shape[1]:
        raise ValueError(f"w must be [k_shard={x.shape[1]}, width], got "
                         f"{w.shape}")
    width = w.shape[1]
    if width % (PACK * LANES) != 0:
        raise ValueError(f"width {width} is not a multiple of "
                         f"{PACK * LANES} (one packed tile)")
    tiles = width // (PACK * LANES)
    if tiles > num_devices:
        raise ValueError(f"tiles ({tiles}) must be <= num_devices "
                         f"({num_devices}); multi-tile ownership is not "
                         "implemented")
    if slot not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot}")
    expected_rs = (2, num_devices, PACK, LANES)
    expected_ag = (2, tiles + 1, PACK, LANES)
    if tuple(rs.shape) != expected_rs:
        raise ValueError(f"rs must be {expected_rs}, got {rs.shape}")
    if tuple(ag.shape) != expected_ag:
        raise ValueError(f"ag must be {expected_ag}, got {ag.shape}")

    kernel = functools.partial(_fused_out_proj_ar_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               tiles=tiles,
                               slot=slot,
                               barrier_first=barrier_first,
                               steady_signals=steady_signals)
    (out, ) = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((1, width), x.dtype), ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[
            pltpu.VMEM((tiles, PACK, LANES), x.dtype),
            pltpu.VMEM((num_devices, PACK, LANES), x.dtype),
            pltpu.VMEM((PACK, LANES), x.dtype),
            pltpu.VMEM((tiles, PACK, LANES), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.REGULAR,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id if barrier_first else None,
            has_side_effects=has_side_effects,
            disable_semaphore_checks=True),
    )(x, w, rs, ag)
    return out


def make_fused_out_proj_workspace(width: int, dtype,
                                  num_devices: int) -> tuple:
    """Zeroed ``(rs, ag)`` for :func:`fused_out_proj_all_reduce`.

    Two slots for the static slot rotation; ``ag`` carries one trash slot
    past the ``tiles`` real ones for ownerless ranks' broadcasts.
    """
    tiles = width // (PACK * LANES)
    return (jnp.zeros((2, num_devices, PACK, LANES),
                      dtype), jnp.zeros((2, tiles + 1, PACK, LANES), dtype))
