# SPDX-License-Identifier: Apache-2.0
"""Fused two-shot all-reduce + RMS-norm + up projection: rolled loop AND deep prefetch.

v2 and v3 each have one half of what this kernel needs, and measurably lose the
other (20 paired passes, 8x TPU7x, vs the unfused two-shot AR + XLA projection):

    bn=512   v2_nn  -0.582   t=-2.34   16/20 wins
             v3     +2.836   t=10.57    0/20
    bn=1792  v3     -0.206   t=-0.74   13/20
             v2_nn  +0.521   t= 2.25    6/20

* **v2** buys overlap with a hand-written DMA pipeline -- ``pf_ahead`` weight
  blocks are issued *before* the collective's remote copies, so they stream
  while it runs. It pays by expressing the block loop as a Python ``for``,
  which unrolls at trace time: 4229 VLIW bundles / 3136 MXU pushes, ~1.4 us
  behind XLA on the projection alone.
* **v3** gets XLA-parity codegen for free from ``pallas_call(grid=...)`` -- a
  rolled loop, 1532 bundles / 784 pushes, the same counts XLA emits. But its
  only overlap is the pipeliner's one-block double buffer, so the amount hidden
  behind the collective is fixed at one block no matter how ``bn`` is chosen.

The two properties are independent: rolled codegen comes from the grid, prefetch
depth comes from owning the DMAs. This kernel takes both. The weight is an
``ANY`` (HBM) operand, so the pipeliner never touches it; the grid supplies only
the trip count and the output blocking, while the kernel drives a depth-``d``
ring of manual DMAs itself:

    step 0 : issue blocks 0..n_ahead-1   <- in flight across the collective
             run the collective, norm
    step j : issue block j+n_ahead (if any), wait block j, one dot

so ``n_ahead`` is free to be as deep as VMEM allows while the dot stays inside
a rolled loop that is emitted exactly once.

**The weight is pre-blocked** to ``[steps, latent, bn]`` (see
:func:`blockify_weight`) rather than sliced out of ``[latent, n_out]`` at call
time. Two reasons: the DMA of block ``j`` becomes fully contiguous instead of
``latent`` runs of ``bn`` elements strided by ``n_out``, and the dynamic block
index becomes a leading-dim index, which Mosaic handles without a lane-crossing
slice. This is a *load-time* relayout, unlike v1's per-call one that cost
61 us/layer.

VMEM cost is ``d * latent * bn * 2`` bytes with ``d = n_ahead + 1``: at
latent=3584, bn=512 that is 3.5 MiB per slot, so n_ahead=4 needs 17.6 MiB
against the 32 MiB scoped reserve.

Shapes are as v2/v3 -- see :func:`padded_latent_width` for why ``cols`` must be
lane-aligned -- but unlike them this kernel takes ``tokens > 1``. Nothing in the
body is single-token: the norm reduces over the last axis per row, and the dot
is ``(tokens, latent) x (latent, bn)``. That matters in practice because the
mxfp4 expert GMM cannot compile at a 1-token bucket (it bitcasts a 4-bit ref
inside ``emit_pipeline`` and fails the second-minor dimension check), so a
served decode step is padded to a bucket of at least 4.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.all_reduce_norm_matmul_v2 import (
    LANE, padded_latent_width)

__all__ = [
    "all_reduce_norm_matmul_v4", "blockify_weight", "padded_latent_width"
]


def blockify_weight(w: jax.Array, bn: int) -> jax.Array:
    """``[latent, n_out]`` -> ``[n_out // bn, latent, bn]``, done once at load."""
    latent, n_out = w.shape
    if n_out % bn:
        raise ValueError(f"n_out ({n_out}) must be divisible by bn ({bn})")
    return w.reshape(latent, n_out // bn, bn).transpose(1, 0, 2)


def _full_barrier(my_id, num_devices: int):
    """All-to-all barrier, doubled -- see v2 for why the second round matters."""
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
    x_ref,
    w_ref,
    gamma_ref,
    rs_in_ref,
    ag_in_ref,
    step_in_ref,  # inputs
    o_ref,
    rs_ref,
    ag_ref,
    step_ref,  # outputs
    recv_vmem,
    red_vmem,
    act_vmem,
    normed_vmem,
    w_vmem,  # scratch
    local_sem,
    w_sems,
    s1_send,
    s1_recv,
    s2_send,
    s2_recv,
    *,
    num_devices: int,
    axis_name: str,
    steady_barrier: bool,
    eps: float,
    latent: int,
    cols: int,
    steps: int,
    n_ahead: int,
    depth: int,
    precision,
):
    del rs_in_ref, ag_in_ref
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)
    pid = pl.program_id(0)

    def _w_copy(idx):
        """Descriptor for weight block ``idx``; identical at start and wait."""
        ring = lax.rem(idx, depth)
        return pltpu.make_async_copy(src_ref=w_ref.at[idx],
                                     dst_ref=w_vmem.at[ring],
                                     sem=w_sems.at[ring])

    # The collective runs once, in grid step 0. Everything it produces lands in
    # scratch, which persists across grid steps; steps 1.. read normed_vmem
    # only.
    @pl.when(pid == 0)
    def _collective():
        # The weight goes first. These DMAs are in flight for the whole
        # collective -- this is the overlap the grid alone cannot express, and
        # the only part of the block loop that is unrolled (n_ahead is small).
        for k in range(n_ahead):
            _w_copy(k).start()

        @pl.when(step == 0)
        def _():
            _full_barrier(my_id, num_devices)

        # Phase 1 -- reduce-scatter.
        copies = []
        for i in range(1, num_devices):
            peer = lax.rem(my_id + i, num_devices)
            c = pltpu.make_async_remote_copy(
                src_ref=x_ref.at[peer],
                dst_ref=rs_ref.at[slot, i - 1],
                send_sem=s1_send.at[slot],
                recv_sem=s1_recv.at[slot],
                device_id=(peer, ),
                device_id_type=pl.DeviceIdType.MESH)
            c.start()
            copies.append(c)

        if steady_barrier:

            @pl.when(step != 0)
            def _():
                _full_barrier(my_id, num_devices)

        for c in copies:
            c.wait()

        local = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                      dst_ref=recv_vmem,
                                      sem=local_sem)
        local.start()
        local.wait()

        acc = x_ref[my_id]
        for i in range(num_devices - 1):
            acc = acc + recv_vmem[i]
        red_vmem[...] = acc

        # Phase 2 -- all-gather into the flat workspace's lane ranges.
        lane = pl.ds(lax.mul(my_id, cols), cols)
        local_ag = pltpu.make_async_copy(src_ref=red_vmem,
                                         dst_ref=ag_ref.at[slot, :, lane],
                                         sem=local_sem)
        local_ag.start()
        copies = []
        for i in range(1, num_devices):
            c = pltpu.make_async_remote_copy(
                src_ref=red_vmem,
                dst_ref=ag_ref.at[slot, :, lane],
                send_sem=s2_send.at[slot],
                recv_sem=s2_recv.at[slot],
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH)
            c.start()
            copies.append(c)
        local_ag.wait()
        for c in copies:
            c.wait()

        # One bulk stage: the all-gather already placed each chunk in its own
        # lane range, so what lands in VMEM is the contiguous padded latent.
        stage = pltpu.make_async_copy(src_ref=ag_ref.at[slot],
                                      dst_ref=act_vmem,
                                      sem=local_sem)
        stage.start()
        stage.wait()

        f = act_vmem[:, :latent].astype(jnp.float32)
        inv = lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + eps)
        normed_vmem[...] = (f * inv *
                            gamma_ref[...].astype(jnp.float32)).astype(
                                act_vmem.dtype)

        step_ref[0] = step + 1

    # Rolled body, emitted once: keep the ring full, then consume block `pid`.
    nxt = pid + n_ahead

    @pl.when(nxt < steps)
    def _():
        _w_copy(nxt).start()

    _w_copy(pid).wait()

    o_ref[...] = lax.dot_general(normed_vmem[...],
                                 w_vmem[lax.rem(pid, depth)],
                                 (((1, ), (0, )), ((), ())),
                                 preferred_element_type=jnp.float32,
                                 precision=precision).astype(o_ref.dtype)


def all_reduce_norm_matmul_v4(x: jax.Array,
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
                              bn: int = 512,
                              n_ahead: int = 2,
                              collective_id: int = 0,
                              has_side_effects: bool = True,
                              steady_barrier: bool = True,
                              vmem_limit_bytes: int = 110 * 1024 * 1024,
                              declare_cost: bool = True,
                              cost_scale: int = 1,
                              precision=None):
    """All-reduce ``x``, RMS-norm the real latent, project by ``w``.

    ``w`` is the **pre-blocked** ``[n_out // bn, latent, bn]`` weight from
    :func:`blockify_weight`. Returns ``(out, rs, ag, step)``; thread the last
    three into the next call.
    """
    if x.ndim != 3:
        raise ValueError(f"expects 3-D x (num_devices, tokens, cols), got "
                         f"{x.shape}")
    nd, tokens, cols = x.shape
    if nd != num_devices:
        raise ValueError(f"x leading dim {nd} != num_devices {num_devices}")
    if w.ndim != 3:
        raise ValueError(
            f"expects pre-blocked [steps, latent, bn] weight (see "
            f"blockify_weight), got {w.shape}")
    steps, w_latent, w_bn = w.shape
    if w_latent != latent:
        raise ValueError(f"weight contracts {w_latent}, latent is {latent}")
    if w_bn != bn:
        raise ValueError(f"weight blocked at bn={w_bn}, called with bn={bn}")
    if bn % LANE:
        raise ValueError(f"bn ({bn}) must be a multiple of {LANE}")
    if cols % LANE:
        raise ValueError(
            f"per-device chunk {cols} must be a multiple of {LANE}; pad to "
            f"padded_latent_width({latent}, {num_devices})="
            f"{padded_latent_width(latent, num_devices)}")
    if tokens < 1:
        raise ValueError(f"tokens must be >= 1, got {tokens}")
    if n_ahead < 1:
        raise ValueError(f"n_ahead must be >= 1, got {n_ahead}")
    if tuple(gamma.shape) != (1, latent):
        raise ValueError(f"gamma must be (1, {latent}) -- it is per-feature "
                         f"and broadcasts over tokens -- got {gamma.shape}")

    n_out = steps * bn
    k_pad = nd * cols
    n_ahead = min(n_ahead, steps)
    depth = min(n_ahead + 1, steps)
    kernel = functools.partial(_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               steady_barrier=steady_barrier,
                               eps=eps,
                               latent=latent,
                               cols=cols,
                               steps=steps,
                               n_ahead=n_ahead,
                               depth=depth,
                               precision=precision)
    ANY = pl.BlockSpec(memory_space=pl.ANY)

    # Declare what this kernel absorbed, or XLA costs the whole thing at zero.
    #
    # A Mosaic custom call has no cost unless we give it one, and the plain
    # low-latency all-reduce this kernel replaces declares none either -- so in
    # the unfused graph the collective is *already* free in XLA's cost model and
    # the only costed work in this window is the norm+up_proj fusion we absorb.
    # Delete that and the modelled elapsed time between a weight becoming
    # available and its use collapses, which is both the denominator of MSA's
    # `MemoryBoundednessBufferIntervalComparator` and the budget
    # `PrefetchIntervalPicker` checks. Measured consequence at TP=8: MSA
    # re-sorts and evicts all 92 of the 21 MiB `shared_experts.gate_up_proj`
    # weights from VMEM (6.5 us -> 35 us each), which costs 3x what the fusion
    # saves. See the fused-up-proj investigation notes.
    #
    # `remote_bytes_transferred` is deliberately left at 0: the goal is to
    # reproduce the unfused graph's modelled cost, not to exceed it, and the
    # all-reduce it replaces contributed nothing.
    # cost_scale is a DIAGNOSTIC. Declaring the honest cost moved nothing
    # (26.54 -> 26.53 ms), which is ambiguous between "XLA read it and MSA did
    # not care" and "XLA ignores the field". An exaggerated value separates
    # those: if a 32x cost still moves nothing, the field is inert here.
    cost_estimate = pl.CostEstimate(
        flops=2 * tokens * latent * n_out * cost_scale,
        transcendentals=tokens * cost_scale,  # one rsqrt per row
        bytes_accessed=x.dtype.itemsize * cost_scale *
        (tokens * latent + latent * n_out + tokens * n_out),
    ) if declare_cost else None

    return pl.pallas_call(
        kernel,
        grid=(steps, ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, n_out), x.dtype),
            jax.ShapeDtypeStruct(rs.shape, rs.dtype),
            jax.ShapeDtypeStruct(ag.shape, ag.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ),
        in_specs=[
            pl.BlockSpec((nd, tokens, cols), lambda j: (0, 0, 0)),  # x
            ANY,  # w (HBM)
            pl.BlockSpec((1, latent), lambda j: (0, 0)),  # gamma
            ANY,  # rs in
            ANY,  # ag in
            pl.BlockSpec(memory_space=pltpu.SMEM),  # step in
        ],
        out_specs=(
            pl.BlockSpec((tokens, bn), lambda j: (0, j)),  # out block
            ANY,
            ANY,
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ),
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, tokens, cols), x.dtype),  # recv
            pltpu.VMEM((tokens, cols), x.dtype),  # reduced
            pltpu.VMEM((tokens, k_pad), x.dtype),  # staged
            pltpu.VMEM((tokens, latent), x.dtype),  # normed
            pltpu.VMEM((depth, latent, bn), x.dtype),  # weight ring
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((depth, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
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
        cost_estimate=cost_estimate,
    )(x, w, gamma, rs, ag, step)
