# SPDX-License-Identifier: Apache-2.0
"""Two-shot all-reduce fused with K3's latent up projection -- contiguous form.

Same idea as ``all_reduce_norm_matmul``: the decode collective stalls on ICI
with the HBM interface idle, while the projection that consumes its output is
24.5 MiB of weight and almost no arithmetic at one token, so the weight streams
in *during* the collective. VMEM is per-kernel scratch and this platform exposes
no VMEM memory kind, so it has to be one kernel.

What is different is the layout, and it is the whole point of this file. v1
staged the reduced activation in the collective's own ``(num_devices, chunk,
cols)`` layout, because K3's ``cols = 3584/8 = 448`` is 3.5 lane tiles and
neither a reshape to ``(1, 3584)`` nor a 448-strided DMA scatter is legal. With
the latent split across two axes and ``tpu.matmul`` taking one contracting dim,
the projection became a sum of ``num_devices`` dots per weight block:

    14 blocks x 8 devices = 112 dots of (1,448) x (256,448) per layer

Measured standalone (``llar-lamport-exp/mm_split_bench.py``, TPU7x, exact
numerics, <=0.6 us spread over 3 passes):

    xla_nt                     11.05 us   one dot, reference
    single_bn448_d8            13.29      one dot per weight block
    split2_d8_bn448_d8         57.22      112 dots, weight device-major
    split_d8_bn448_d8          80.73      112 dots, v1's layout

**~68 us/layer of overhead**, against the ~7 us of weight stream the prefetch
can hide. Two separable causes: ~23 us is v1 storing the weight as ``(n_out,
nd, cols)`` and slicing ``wblk[:, i, :]``, which extracts one sublane from
every tile and forces Mosaic to relayout the block; the remaining ~44 us is the
dot count itself.

**Pad the payload and both problems go away.** The dispatcher pads the latent
from 3584 to ``k_pad = num_devices * round_up(latent/num_devices, 128)`` --
4096 at both 8 and 32 devices -- so ``cols`` becomes 512 or 128 and every
per-device lane offset is a multiple of 128. Then:

* the all-gather writes into ``ag[slot, :, my_id*cols : (my_id+1)*cols]``, a
  lane range of a **flat ``(tokens, k_pad)``** workspace rather than a slot of a
  3-D one, so the staging copy into VMEM is ONE bulk DMA and the activation
  lands contiguous. No scatter, and no 4.6 us relayout either (the explicit
  scatter form measured 18.15 vs 13.29 us).
* the norm reads ``act[:, :latent]`` -- offset 0, size 3584 = 28 tiles -- and
  ``gamma`` is its natural ``(1, latent)``. v1 had to pass gamma pre-reshaped
  into the workspace layout and reduce over every axis.
* the matmul is one ``dot_general`` per weight block against vLLM's **native**
  ``[n_out, latent]`` weight. v1 needed a physical relayout to the split
  layout, which cost 61 us/layer done per call and so had to be hoisted to load
  time behind ``prepare_fused_up_proj_weights``. That whole mechanism, and its
  ``TPU_FUSED_UP_USE_PRESPLIT`` flag, is unnecessary here.

The padding widens the collective by 14% (3584 -> 4096 elements, 7 -> 8 KiB per
token at bf16). At this size the collective is latency-bound, not bandwidth-
bound, so that should be close to free -- but it is a real cost and the arm
below is what says whether it is.

Still single-token only: the norm is per token, and several tokens per lane
range would need a leading-dim split of the staged buffer that is untested.
The intra-chip all-gather stays outside, exactly as in v1 -- the projection is
column-parallel over a chip's two cores, so this returns the core's half.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = [
    "all_reduce_norm_matmul_v2",
    "make_fused_workspace_v2",
    "padded_latent_width",
]

LANE = 128


def padded_latent_width(latent: int, num_devices: int) -> int:
    """Smallest ``k_pad >= latent`` whose per-device chunk is lane-aligned.

    This is the precondition for every layout claim in the module docstring: a
    VMEM slice or DMA lane offset must be a multiple of 128, and K3's raw
    ``3584/8 = 448`` and ``3584/32 = 112`` are neither.
    """
    per_device = -(-latent // num_devices)
    return num_devices * (-(-per_device // LANE) * LANE)


def _full_barrier(my_id, num_devices: int):
    """All-to-all barrier, doubled.

    A single semaphore barrier lets a rank pass round k+1 on signals left over
    from round k, which drifts the DMA semaphore counts a round out of step. At
    8 devices on one host the skew never grows enough to matter; across four
    hosts it kills the slice with ``SLICE_FAILURE_SW_INJECT_ERROR``.
    """
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(1, num_devices):
        pl.semaphore_signal(barrier_sem,
                            1,
                            device_id=(lax.rem(my_id + i, num_devices), ),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier_sem, num_devices - 1)

    @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.REGULAR)
    def _(sem):
        for i in range(1, num_devices):
            pl.semaphore_signal(sem,
                                1,
                                device_id=(lax.rem(my_id + i, num_devices), ),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(sem, num_devices - 1)


def _kernel(
    # inputs
    x_ref,
    w_hbm_ref,
    gamma_ref,
    rs_in_ref,
    ag_in_ref,
    step_in_ref,
    # outputs
    o_ref,
    rs_ref,
    ag_ref,
    step_ref,
    # scratch
    recv_vmem,
    red_vmem,
    act_vmem,
    w_vmem,
    local_sem,
    s1_send,
    s1_recv,
    s2_send,
    s2_recv,
    w_sems,
    *,
    num_devices: int,
    axis_name: str,
    steady_barrier: bool,
    eps: float,
    bn: int,
    depth: int,
    n_out: int,
    latent: int,
    cols: int,
    precision,
    pf_ahead: int,
    nn: bool,
):
    del rs_in_ref, ag_in_ref
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)
    steps = n_out // bn

    @pl.when(step == 0)
    def _():
        _full_barrier(my_id, num_devices)

    # The weight goes first: every DMA issued here is in flight for the whole
    # collective, which is the entire point of the fusion. Blocks are
    # contiguous rows of the native [n_out, latent] weight -- no relayout.
    # NN takes the weight as [latent, n_out] and slices a lane range; NT takes
    # vLLM's own [n_out, latent] and slices rows. NT is the drop-in, and it is
    # what v1 and the first v2 used -- but Mosaic lowers it to a costlier MXU
    # push sequence. Measured on the LLO: 5094 vs 3519 VLIW bundles for the
    # same maths, and 18.11 vs 16.77 us standalone at bn=896. The cost of NN is
    # transposing the weight once at load time.
    w_copies = [
        pltpu.make_async_copy(src_ref=(w_hbm_ref.at[:, pl.ds(j * bn, bn)] if nn
                                       else w_hbm_ref.at[pl.ds(j * bn, bn)]),
                              dst_ref=w_vmem.at[j % depth],
                              sem=w_sems.at[j % depth]) for j in range(steps)
    ]
    # How many weight blocks to put in flight *before* the collective issues
    # its remote copies. Not simply "all of them": the weight DMAs and the
    # collective's transfers share DMA queues, so a deep prefetch queued ahead
    # of phase 1 delays the very collective it is meant to hide behind. The
    # rest of the pipeline is started right after the phase-1 sends are issued
    # -- still inside the collective, but behind it in the queue.
    #
    # pf_ahead < 0 defers every weight DMA until the collective has finished:
    # the no-overlap control, whose gap to a prefetching arm IS the overlap.
    n_pipe = min(depth, steps)
    n_ahead = n_pipe if pf_ahead < 0 else min(pf_ahead, n_pipe)
    if pf_ahead >= 0:
        for j in range(n_ahead):
            w_copies[j].start()

    # Phase 1 -- reduce-scatter. x stays in the (num_devices, chunk, cols)
    # view so these are leading-dim slices, same as v1: the only layout change
    # in this kernel is on the all-gather side.
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

    if pf_ahead >= 0:
        # The remainder of the pipeline: behind the collective's own sends in
        # the queue, but still overlapping the transfer.
        for j in range(n_ahead, n_pipe):
            w_copies[j].start()

    if steady_barrier:

        @pl.when(step != 0)
        def _():
            _full_barrier(my_id, num_devices)

    for copy in copies:
        copy.wait()

    local = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    local.start()
    local.wait()

    acc = x_ref[my_id]
    for i in range(num_devices - 1):
        acc = acc + recv_vmem[i]
    red_vmem[...] = acc

    # Phase 2 -- all-gather straight into the flat workspace. Each device owns
    # the lane range [my_id*cols, (my_id+1)*cols); cols is a multiple of 128 by
    # construction (see padded_latent_width), which is what makes this legal
    # and what lets the staging below be a single bulk copy.
    lane = pl.ds(lax.mul(my_id, cols), cols)
    local_ag = pltpu.make_async_copy(src_ref=red_vmem,
                                     dst_ref=ag_ref.at[slot, :, lane],
                                     sem=local_sem)
    local_ag.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_vmem,
            dst_ref=ag_ref.at[slot, :, lane],
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

    # ag is the caller's workspace, declared pl.ANY, and Mosaic refuses a
    # vector load from ANY -- it can only be reached by async_copy. One bulk
    # DMA, and because the all-gather already wrote each chunk to its own lane
    # range, what lands in VMEM is the contiguous padded latent.
    stage = pltpu.make_async_copy(src_ref=ag_ref.at[slot],
                                  dst_ref=act_vmem,
                                  sem=local_sem)
    stage.start()
    stage.wait()

    # The norm needs the whole reduced vector, so it cannot start before the
    # all-gather completes; it is cheap next to the projection. Reads only the
    # real latent -- the padding tail is never summed, so it cannot perturb the
    # mean square. latent is a multiple of 128 at offset 0, so the slice is
    # tile-aligned.
    if pf_ahead < 0:
        for j in range(n_ahead):
            w_copies[j].start()

    f = act_vmem[:, :latent].astype(jnp.float32)
    inv = lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + eps)
    normed = (f * inv * gamma_ref[...].astype(jnp.float32)).astype(x_ref.dtype)

    # One dot per weight block, contracting the last dim of both operands --
    # vLLM stores a linear as [out, in] and computes x @ W.T. Each block owns a
    # distinct slice of the output, so no accumulator is needed.
    for j in range(steps):
        w_copies[j].wait()
        o_ref[:, j * bn:(j + 1) * bn] = jax.lax.dot_general(
            normed,
            w_vmem[j % depth], (((1, ), (0, )), ((), ())) if nn else
            (((1, ), (1, )), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision).astype(o_ref.dtype)
        if j + depth < steps:
            w_copies[j + depth].start()

    step_ref[0] = step + 1


def make_fused_workspace_v2(tokens: int, latent: int, num_devices: int, dtype,
                            device_put):
    """Allocate the persistent ``(rs, ag, step)`` workspace.

    ``device_put`` takes a numpy/jax array and returns it resident on the
    caller's device, so this stays free of any framework assumption.

    ``ag`` is flat ``(2, tokens, k_pad)`` rather than v1's ``(2, num_devices,
    chunk, cols)``. That single change is what makes the staged activation
    contiguous, because each device all-gathers into its own lane range of it.
    """
    import numpy as np
    k_pad = padded_latent_width(latent, num_devices)
    cols = k_pad // num_devices
    rs = device_put(np.zeros((2, num_devices - 1, tokens, cols), dtype=dtype))
    ag = device_put(np.zeros((2, tokens, k_pad), dtype=dtype))
    step = device_put(np.zeros((1, ), dtype=np.int32))
    return rs, ag, step


def all_reduce_norm_matmul_v2(x: jax.Array,
                              w: jax.Array,
                              gamma: jax.Array,
                              rs: jax.Array,
                              ag: jax.Array,
                              step: jax.Array,
                              *,
                              axis_name: str,
                              num_devices: int,
                              latent: int,
                              eps: float = 1e-5,
                              bn: int = 896,
                              depth: int = 4,
                              collective_id: int = 0,
                              has_side_effects: bool = True,
                              steady_barrier: bool = True,
                              vmem_limit_bytes: int = 110 * 1024 * 1024,
                              precision=None,
                              pf_ahead: int = 2,
                              nn: bool = False):
    """All-reduce ``x``, RMS-norm the real latent, project by ``w``.

    ``x`` is ``(num_devices, tokens, cols)`` -- the padded latent cut into
    lane-aligned per-device chunks. ``w`` is vLLM's own ``[n_out, latent]``,
    contracted on its last dim, with **no relayout**. ``gamma`` is
    ``(1, latent)``.

    Returns ``(out, rs, ag, step)``; thread the last three into the next call.
    """
    if x.ndim != 3:
        raise ValueError(f"expects 3-D x (num_devices, tokens, cols), got "
                         f"{x.shape}")
    nd, tokens, cols = x.shape
    if nd != num_devices:
        raise ValueError(f"x leading dim {nd} != num_devices {num_devices}")
    if w.ndim != 2:
        raise ValueError(f"expects vLLM's 2-D [n_out, latent] weight, got "
                         f"{w.shape}")
    if nn:
        # [latent, n_out]: the transposed layout, blocked over lane ranges, so
        # bn must be lane-aligned.
        w_latent, n_out = w.shape
        if bn % LANE:
            raise ValueError(
                f"nn=True blocks the weight over lanes, so bn ({bn}) must be a "
                f"multiple of {LANE}")
    else:
        n_out, w_latent = w.shape
    if w_latent != latent:
        raise ValueError(f"weight contracts {w_latent}, latent is {latent}")
    if cols % LANE:
        raise ValueError(
            f"per-device chunk {cols} is not a multiple of {LANE}; pad the "
            f"payload to padded_latent_width({latent}, {num_devices})="
            f"{padded_latent_width(latent, num_devices)} first")
    if nd * cols < latent:
        raise ValueError(f"padded width {nd * cols} is under latent {latent}")
    if tokens != 1:
        # The norm is per token, and several tokens sharing a lane range would
        # need a leading-dim split of the staged buffer that is untested here.
        raise NotImplementedError(f"only a single token, got {tokens}")
    if n_out % bn:
        raise ValueError(f"n_out ({n_out}) must divide bn ({bn})")
    if gamma.shape != (1, latent):
        raise ValueError(f"gamma must be (1, {latent}), got {gamma.shape}")

    kernel = functools.partial(_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               steady_barrier=steady_barrier,
                               eps=eps,
                               bn=bn,
                               depth=depth,
                               n_out=n_out,
                               latent=latent,
                               cols=cols,
                               precision=precision,
                               pf_ahead=pf_ahead,
                               nn=nn)
    k_pad = nd * cols
    steps = n_out // bn
    depth = min(depth, steps)

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, n_out), x.dtype),
            jax.ShapeDtypeStruct(rs.shape, rs.dtype),
            jax.ShapeDtypeStruct(ag.shape, ag.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),  # x
            pl.BlockSpec(memory_space=pl.ANY),  # w, streamed
            pl.BlockSpec(memory_space=pltpu.VMEM),  # gamma
            pl.BlockSpec(memory_space=pl.ANY),  # rs in
            pl.BlockSpec(memory_space=pl.ANY),  # ag in
            pl.BlockSpec(memory_space=pltpu.SMEM),  # step in
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ),
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, tokens, cols), x.dtype),  # recv
            pltpu.VMEM((tokens, cols), x.dtype),  # reduced
            pltpu.VMEM((tokens, k_pad), x.dtype),  # staged
            pltpu.VMEM((depth, latent, bn) if nn else (depth, bn, latent),
                       w.dtype),  # weight
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((depth, )),
        ],
        input_output_aliases={
            3: 1,
            4: 2,
            5: 3
        },
        compiler_params=pltpu.CompilerParams(
            has_side_effects=has_side_effects,
            collective_id=collective_id,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )(x, w, gamma, rs, ag, step)
