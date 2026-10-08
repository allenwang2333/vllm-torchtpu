# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode kernel: routed experts, shared expert and all-reduce in one call.

Under MoE tensor parallelism every rank holds all experts with the
intermediate dimension sharded, so each expert's local slice is small (for
Qwen3.5-397B at TP8: i = 128, 1.5 MiB of FP8 per expert) and rows per expert
at decode are ~1-3. The GMM path pays a fixed cost per active expert in each
of its two kernels plus a permute/sort/combine around them.

This kernel visits each active expert once. Its w13/w2 slices stream from HBM
through a ring of VMEM slots while the previous expert computes on ALL tokens
in their original order; the route weight zeroes rows the expert does not
own. The first matmul of expert j+1 is issued in step j, ahead
of expert j's elementwise chain and second matmul, so the MXU stays busy while
the VPU quantizes. Nothing is permuted and no intermediate goes to HBM.

The same call also runs the layer's shared expert slice and the TP
all-reduce, so one kernel replaces the whole MoE block after the router
(served by layers/adapter/fused_moe_decode.py). With expert parallelism and
TP attention a rank holds whole experts instead of slices and the call is the
same, restricted to the rank's expert range.

Numerics: hidden states are quantized to FP8 once per call with one scale per
row; the activated intermediate is quantized per row per expert; both
matmuls are FP8 x FP8 with FP32 accumulation into an FP32 VMEM accumulator.
Router top-k stays with the caller.

Derived from moe_dense_expert.py, which targets i % 512 == 0 (EP layouts).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Output columns per second-matmul step. Each step computes an f32 block of
# [tokens, _N_CHUNK] and adds it into the accumulator, so this bounds the live
# f32 result. It is a compile-time constant: keep it a multiple of 128 (the
# lane width) that divides the hidden size. Smaller lowers register and VMEM
# pressure at large token counts and issues more, shorter matmuls.
_N_CHUNK = 512
# Upper bound on the weight ring depth. The depth one call uses is picked by
# _weight_ring_depth: the most slots, at most this and at least 3, whose expert
# weights fit in half of VMEM. One slot holds the expert whose first matmul is
# in flight, so depth - 2 weight transfers run ahead of compute. For
# Qwen3.5-397B at 8 ranks this bound sets the depth with TP experts (6 slots of
# 1.5 MiB); with EP experts (12 MiB per slot) VMEM caps it at 3. A deeper ring
# hides more DMA latency when experts are small, but the VMEM it takes lowers
# the token cap vmem_bytes allows.
_NUM_BUFFERS = 6


# The expert loop, plus the layer's shared expert (its TP slice, gated by
# sigmoid(x . w_g)) and the TP all-reduce, in one call per rank under
# shard_map. The all-reduce is latency bound at decode sizes, so it runs once
# at the end; the barrier it needs is signalled at entry and waited only
# before the reduce-scatter, so the expert loop hides it. bf16 on the wire,
# FP32 sum of the slots.


def _kernel(
    *refs,
    m,
    h,
    i,
    experts,
    nb,
    n_ranks,
    weight_dtype,
    rows=None,
    topk=None,
    sent=None,
):
    """One rank's MoE block. Shapes use m tokens (padded to 16), h hidden, i
    and i_s the routed and shared intermediate sizes of this rank's slice, E
    local experts, nb ring slots, n ranks and c = h // n.

    Scalar prefetch (SMEM):
      active_ref [E] int32: 1 where at least one token routes to local expert e.
      rank_ref [1] int32: this rank's mesh index.
    Inputs in VMEM:
      x_ref [m, h] bf16: hidden states.
      coeff_ref [m, E padded to 128] f32: route weight of token t for local
        expert e, 0 where t does not route to e.
      shared_gate_ref [m, 128] f32: sigmoid(x . w_g) per token, repeated over lanes.
    Inputs in HBM (weights in `weight_dtype`, scales f32 per output channel):
      w1_hbm [E, h, 2 i] and w1_scale_hbm [E, 1, 1, 2 i]: gate and up projections.
      w2_hbm [E, i, h] and w2_scale_hbm [E, 1, 1, h]: down projection.
      shared_w1_hbm [h, 2 i_s], shared_w1_scale_hbm [1, 2 i_s],
      shared_w2_hbm [i_s, h], shared_w2_scale_hbm [1, h]: the shared expert.
    Output:
      out_ref [m, h] bf16: shared + routed output summed over the n ranks.
        With `rows` set (DP attention), [r_pad, h]: only this rank's `rows`
        token rows of that sum, r_pad = rows padded to the sublane tile.
    Scratch:
      w1_ring [nb, h, 2 i], w2_ring [nb, i, h], w1_scale_ring [nb, 1, 2 i],
        w2_scale_ring [nb, 1, h]: weight ring, one expert per slot.
      shared_w1_vmem, shared_w1_scale_vmem, shared_w2_vmem,
        shared_w2_scale_vmem: the shared expert, loaded once.
      xq_ref [m, h] weight_dtype: hidden states quantized per row.
      xs_ref [m, 2 i] f32: their row scales, repeated over columns.
      acc_ref [m, h] f32: this rank's partial sum.
      acc_bf16 [m, h] bf16: the same, as the all-reduce source.
      recv [n, m, c] bf16: reduce-scatter slots, one per sending rank.
      red [m, c] bf16: this rank's reduced column chunk.
      With `rows` set, acc_bf16 and recv are [n, r_pad, h] (row slab d is
      peer d's tokens) and red is unused.
      ids_ref [E] int32 (SMEM): active expert ids, compacted.
      sem_ref DMA [nb, 4]: ring slot transfers (w1, w2, w1 scale, w2 scale).
      shared_sem DMA [4]: shared expert transfers.
      rs_send DMA [n], rs_recv DMA: reduce-scatter.
      ag_send DMA [n], ag_recv DMA: all-gather.

    With `topk` set (gather_rows, needs `rows`), every input is this rank's
    own `rows` tokens and the kernel all-gathers them itself:
      Scalar prefetch: rank_ref [2] int32: this rank's mesh index and the
        global id of its first expert (0 with TP experts).
      x_ref [rows, h] bf16: own hidden rows.
      route_ref [rows, 128] f32: own routing, lanes [0, k) global expert ids
        (as f32; each receiver offsets them by its own first expert), [k, 2 k)
        route weights, 2 k the shared gate sigmoid(x . w_g).
      Output: out_ref [rows, h] bf16, this rank's rows, whatever the wire.
      Extra scratch: x_all [n, r16, h] and route_all [n, r16, 128] receive
        slab d from rank d (ag_* and route_* semaphores), r16 = rows rounded
        up to 16; x_all is FP32 when the rows travel in FP32 (see the
        gather below), and x_send [r16, h] FP32 stages the own rows for it
        (else unused); route_rows
        [m, 128] f32 holds the routing token-major; coeff_ref [m, E padded]
        f32 is built here from it instead of arriving from XLA, and so is the
        compacted active-expert list: active_vmem [8, E padded] int32 holds
        the ids in row 0 and their count in row 1, and is DMA'd into
        active_ref (SMEM, same shape; active_sem). ids_ref is unused.
    """
    if topk is None:
        active_ref, rank_ref, x_ref, coeff_ref, shared_gate_ref = refs[:5]
        refs = refs[5:]
    else:
        rank_ref, x_ref, route_ref = refs[:3]
        refs = refs[3:]
    (
        w1_hbm,
        w2_hbm,
        w1_scale_hbm,
        w2_scale_hbm,
        shared_w1_hbm,
        shared_w1_scale_hbm,
        shared_w2_hbm,
        shared_w2_scale_hbm,
        out_ref,
        w1_ring,
        w2_ring,
        w1_scale_ring,
        w2_scale_ring,
        shared_w1_vmem,
        shared_w1_scale_vmem,
        shared_w2_vmem,
        shared_w2_scale_vmem,
        xq_ref,
        xs_ref,
        acc_ref,
        acc_bf16,
        recv,
        red,
        ids_ref,
        sem_ref,
        shared_sem,
        rs_send,
        rs_recv,
        ag_send,
        ag_recv,
    ) = refs[:30]
    if topk is not None:
        (
            x_all,
            route_all,
            route_rows,
            coeff_ref,
            active_vmem,
            active_ref,
            route_send,
            route_recv,
            active_sem,
            x_send,
        ) = refs[30:]
    me = rank_ref[0]
    qmax = float(jnp.finfo(weight_dtype).max)
    chunk = h // n_ranks
    barrier = pltpu.get_barrier_semaphore()
    for d in range(n_ranks):
        pl.semaphore_signal(
            barrier,
            inc=1,
            device_id=(jnp.int32(d),),
            device_id_type=pl.DeviceIdType.MESH,
        )

    shared_copies = [
        pltpu.make_async_copy(shared_w1_hbm, shared_w1_vmem, shared_sem.at[0]),
        pltpu.make_async_copy(shared_w2_hbm, shared_w2_vmem, shared_sem.at[1]),
        pltpu.make_async_copy(
            shared_w1_scale_hbm, shared_w1_scale_vmem, shared_sem.at[2]
        ),
        pltpu.make_async_copy(
            shared_w2_scale_hbm, shared_w2_scale_vmem, shared_sem.at[3]
        ),
    ]
    for copy in shared_copies:
        copy.start()

    gather_sends = []
    if topk is not None:
        # gather_rows: every peer's buffers are live once all have entered.
        # Each rank sends its routing, then its hidden rows, to slab `me` of
        # every rank (itself included); the expert loop needs the routing
        # first, so it travels first.
        pl.semaphore_wait(barrier, n_ranks)
        # Only the real rows travel (`sent` of them), not the slab's tile
        # padding (at one to four rows per rank the padded slab is 4-16x the
        # bytes). A DMA moves whole sublanes from a tile start: BF16 rows go
        # when they are an even count under 16 or whole tiles; otherwise the
        # rows go in FP32, rounded up to whole 8-row tiles above 8. Measured:
        # one BF16 row arrived wrong, and both test jobs that sent 24 BF16
        # rows (a tile and a half) died there with the node marked unhealthy.
        own = pl.ds(0, sent)
        if x_send.dtype != x_ref.dtype:
            x_send[:rows, :] = x_ref[...].astype(x_send.dtype)
            x_src = x_send.at[own]
        else:
            x_src = x_ref
        route_own = pl.ds(0, route_ref.shape[0])  # FP32 slab, same rule as x
        for src, dst, send, recv_sem, part in (
            (route_ref, route_all, route_send, route_recv, route_own),
            (x_src, x_all, ag_send, ag_recv, own),
        ):
            for d in range(n_ranks):
                copy = pltpu.make_async_remote_copy(
                    src,
                    dst.at[me, part],
                    send.at[d],
                    recv_sem,
                    device_id=(jnp.int32(d),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                gather_sends.append(copy)
        # Token-major routing; rows past the n * rows real tokens stay 0.
        route_rows[...] = jnp.zeros(route_rows.shape, jnp.float32)
        received = (slice(None), route_own)
        pltpu.make_async_copy(
            route_all.at[received], route_all.at[received], route_recv
        ).wait()
        for d in range(n_ranks):
            route_rows[d * rows : (d + 1) * rows, :] = route_all[d, :rows, :]
        route = route_rows[...]
        lanes = lax.broadcasted_iota(jnp.int32, route.shape, 1)

        def lane(j):
            return jnp.sum(jnp.where(lanes == j, route, 0.0), axis=1, keepdims=True)

        # The dense route-weight table XLA builds otherwise: at most one
        # route of a token selects a given expert, so the sum is exact. Ids
        # are global; this rank's local expert e is global rank_ref[1] + e.
        experts_f = (
            lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1) + rank_ref[1]
        ).astype(jnp.float32)
        coeff = jnp.zeros(coeff_ref.shape, jnp.float32)
        for j in range(topk):
            coeff = coeff + jnp.where(experts_f == lane(j), lane(topk + j), 0.0)
        coeff_ref[...] = coeff
        active_vmem[...] = _compact_active(coeff)
        copy = pltpu.make_async_copy(active_vmem, active_ref, active_sem)
        copy.start()
        shared_gate = lane(2 * topk)
        acc_ref[...] = jnp.zeros((m, h), jnp.float32)
        received = (slice(None), own)
        pltpu.make_async_copy(x_all.at[received], x_all.at[received], ag_recv).wait()
        for d in range(n_ranks):
            acc_ref[d * rows : (d + 1) * rows, :] = x_all[d, :rows, :].astype(
                jnp.float32
            )
        x = acc_ref[...]
        copy.wait()
    else:
        acc_ref[...] = jnp.zeros((m, h), jnp.float32)
        x = x_ref[...].astype(jnp.float32)
    xsc = jnp.max(jnp.abs(x), axis=1, keepdims=True) / qmax
    xq_ref[...] = (x * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(weight_dtype)
    xs_ref[...] = jnp.broadcast_to(xsc, (m, 2 * i))
    if topk is not None:  # the accumulator held the gathered rows until now
        acc_ref[...] = jnp.zeros((m, h), jnp.float32)
    cols = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)
    if topk is None:

        def compact(e, count):
            @pl.when(active_ref[e] != 0)
            def record():
                ids_ref[count] = e

            return count + (active_ref[e] != 0).astype(jnp.int32)

        count = lax.fori_loop(0, experts, compact, jnp.int32(0))

        def expert_id(step):
            return ids_ref[step]

    else:
        count = active_ref[1, 0]

        def expert_id(step):
            return active_ref[0, step]

    def transfers(step):
        e, slot = expert_id(step), step % nb
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ring.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ring.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(
                w1_scale_hbm.at[e, 0], w1_scale_ring.at[slot], sem_ref.at[slot, 2]
            ),
            pltpu.make_async_copy(
                w2_scale_hbm.at[e, 0], w2_scale_ring.at[slot], sem_ref.at[slot, 3]
            ),
        ]

    def accumulate(acc1, coeff, w1_scale, w2, w2_scale):
        """Finish one expert from its first-matmul result acc1 [m, 2 width]:
        dequantize, SiLU(gate) * up weighted by coeff [m, 1], quantize per row,
        and add the second matmul with w2 [width, h] into acc_ref."""
        width = acc1.shape[1] // 2  # the routed or the shared intermediate size
        acc1 = acc1 * xs_ref[:, : 2 * width] * w1_scale
        mid = jnp.where(
            coeff != 0, jax.nn.silu(acc1[:, :width]) * acc1[:, width:] * coeff, 0.0
        )
        ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / qmax
        mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(weight_dtype)
        for n in range(0, h, _N_CHUNK):
            acc2 = jnp.matmul(
                mq, w2[:, n : n + _N_CHUNK], preferred_element_type=jnp.float32
            )
            acc_ref[:, n : n + _N_CHUNK] += acc2 * w2_scale[:, n : n + _N_CHUNK] * ms

    # Prime the weight ring: start the weight transfers of the first
    # min(count, nb) active experts into slots 0 .. nb - 1 now, so they stream
    # from HBM while the shared expert below computes. The expert loop then
    # waits on slot step % nb and refills it with expert step + nb.
    @pl.when(count > 0)
    def prime_routed():
        for initial in range(nb):

            @pl.when(count > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

    # Shared expert first; its slabs were issued at entry with the routed primes.
    for copy in shared_copies:
        copy.wait()
    if topk is None:
        shared_coeff = jnp.max(shared_gate_ref[...], axis=1, keepdims=True)
    else:
        shared_coeff = shared_gate
    accumulate(
        jnp.matmul(
            xq_ref[...], shared_w1_vmem[...], preferred_element_type=jnp.float32
        ),
        shared_coeff,
        shared_w1_scale_vmem[...],
        shared_w2_vmem,
        shared_w2_scale_vmem[...],
    )

    @pl.when(count > 0)
    def run_routed():
        for copy in transfers(jnp.int32(0)):
            copy.wait()

        def expert_body(step, acc1):
            nxt = step + 1

            @pl.when(nxt < count)
            def wait_next():
                for copy in transfers(nxt):
                    copy.wait()

            acc1_next = jnp.matmul(
                xq_ref[...], w1_ring[nxt % nb], preferred_element_type=jnp.float32
            )
            slot = step % nb
            e = expert_id(step)
            coeff = jnp.sum(
                jnp.where(cols == e, coeff_ref[...], 0.0), axis=1, keepdims=True
            )
            accumulate(
                acc1, coeff, w1_scale_ring[slot], w2_ring.at[slot], w2_scale_ring[slot]
            )

            @pl.when(step + nb < count)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return acc1_next

        lax.fori_loop(
            0,
            count,
            expert_body,
            jnp.matmul(xq_ref[...], w1_ring[0], preferred_element_type=jnp.float32),
        )

    if rows is not None:
        # DP attention: reduce-scatter over token rows, no all-gather. Rank d
        # receives every rank's partial of tokens [d * rows, (d + 1) * rows),
        # sums them and keeps that slice. Each peer's slab is a whole leading
        # index of acc_bf16 / recv, so it is tile aligned for any rows.
        for d in range(n_ranks):
            acc_bf16[d, :rows, :] = acc_ref[d * rows : (d + 1) * rows, :].astype(
                acc_bf16.dtype
            )
        if topk is None:  # gather_rows waited at entry
            pl.semaphore_wait(barrier, n_ranks)
        scatter = [
            pltpu.make_async_remote_copy(
                acc_bf16.at[d],
                recv.at[me],
                rs_send.at[d],
                rs_recv,
                device_id=(jnp.int32(d),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for d in range(n_ranks)
        ]
        for copy in scatter:
            copy.start()
        pltpu.make_async_copy(recv, recv, rs_recv).wait()
        acc = recv[0].astype(jnp.float32)
        for s in range(1, n_ranks):
            acc = acc + recv[s].astype(jnp.float32)
        if topk is None:
            out_ref[...] = acc.astype(out_ref.dtype)
        else:  # exactly this rank's rows, in the hidden dtype
            out_ref[...] = acc[:rows].astype(out_ref.dtype)
        for copy in scatter + gather_sends:
            copy.wait_send()
        return

    # All-reduce: reduce-scatter column chunks, sum, all-gather.
    acc_bf16[...] = acc_ref[...].astype(acc_bf16.dtype)
    # Every peer has entered this call, so its receive buffer is live.
    pl.semaphore_wait(barrier, n_ranks)
    scatter = [
        pltpu.make_async_remote_copy(
            acc_bf16.at[:, pl.ds(d * chunk, chunk)],
            recv.at[me],
            rs_send.at[d],
            rs_recv,
            device_id=(jnp.int32(d),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for d in range(n_ranks)
    ]
    for copy in scatter:
        copy.start()
    pltpu.make_async_copy(recv, recv, rs_recv).wait()
    acc = recv[0].astype(jnp.float32)
    for s in range(1, n_ranks):
        acc = acc + recv[s].astype(jnp.float32)
    red[...] = acc.astype(red.dtype)
    for copy in scatter:
        copy.wait_send()
    col = pl.multiple_of(me * chunk, chunk)
    gather = [
        pltpu.make_async_remote_copy(
            red,
            out_ref.at[:, pl.ds(col, chunk)],
            ag_send.at[d],
            ag_recv,
            device_id=(jnp.int32(d),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for d in range(n_ranks)
    ]
    for copy in gather:
        copy.start()
    pltpu.make_async_copy(out_ref, out_ref, ag_recv).wait()
    for copy in gather:
        copy.wait_send()


def _compact_active(coeff):
    """[8, E] int32 from the route-weight table coeff [m, E] (E a multiple of
    128): row 0 lists the experts any token routes to, ascending (the order
    the scalar compaction visits them), then zeros; row 1 is their count.

    Vector work instead of a scalar loop over E (about 10 us at E = 512 on
    tpu7x): an exclusive prefix sum of the active flags by a matmul with a
    strictly upper-triangular matrix, then a one-hot [slot, expert] matrix
    contracted against the expert ids. Every operand is an integer exact in
    bf16 (ids split as 32 * hi + lo) and the sums are exact in f32.
    """
    e = coeff.shape[1]
    nt = (((1,), (1,)), ((), ()))
    active = jnp.max(jnp.where(coeff != 0, 1.0, 0.0), axis=0, keepdims=True)
    rows = lax.broadcasted_iota(jnp.int32, (e, e), 0)
    cols = lax.broadcasted_iota(jnp.int32, (e, e), 1)
    upper = jnp.where(rows < cols, 1.0, 0.0).astype(jnp.bfloat16)
    slot = jnp.dot(
        jnp.broadcast_to(active, (8, e)).astype(jnp.bfloat16),
        upper,
        preferred_element_type=jnp.float32,
    )[:1]
    onehot = jnp.where(
        (slot == rows.astype(jnp.float32)) & (active != 0), 1.0, 0.0
    ).astype(jnp.bfloat16)
    lane = lax.broadcasted_iota(jnp.int32, (8, e), 1)
    hi = lax.dot_general(
        (lane // 32).astype(jnp.bfloat16),
        onehot,
        nt,
        preferred_element_type=jnp.float32,
    )
    lo = lax.dot_general(
        (lane % 32).astype(jnp.bfloat16), onehot, nt, preferred_element_type=jnp.float32
    )
    ids = (hi * 32 + lo).astype(jnp.int32)
    count = jnp.sum(active, axis=1, keepdims=True).astype(jnp.int32)
    sub = lax.broadcasted_iota(jnp.int32, (8, e), 0)
    return jnp.where(sub == 1, jnp.broadcast_to(count, (8, e)), ids)


def _weight_ring_depth(h, i, weight_dtype):
    """Deepest weight ring (at most _NUM_BUFFERS slots, at least 3) whose slots
    fit in half of VMEM; one slot holds an expert's w1 and w2."""
    slot = 3 * h * i * jnp.dtype(weight_dtype).itemsize
    fit = pltpu.get_tpu_info().vmem_capacity_bytes // 2 // slot
    return max(3, min(_NUM_BUFFERS, fit))


def vmem_bytes(tokens, h, i, i_s, experts, n_ranks, weight_dtype, gather_rows=False):
    """VMEM the call declares for `tokens` rows: its VMEM operands, output and
    scratch (compiler temporaries excluded). The weight ring takes about half
    of VMEM, so the token-proportional buffers bound the rows one call takes.
    With gather_rows, `tokens` is still the global count (n_ranks x own rows)."""
    m = tokens + (-tokens) % 16
    wt, f32, bf16 = jnp.dtype(weight_dtype).itemsize, 4, 2
    extra = 0
    if gather_rows:
        own = tokens // n_ranks
        r16 = own + (-own) % 16
        e_pad = experts + (-experts) % 128
        # own x + route in, x_all + route_all slabs, token-major routing, and
        # the active mask staging; the hidden and shared-gate inputs are gone
        # (coeff moves from input to scratch, same size).
        extra = (
            (n_ranks + 1) * r16 * (h * bf16 + 128 * f32)
            + m * 128 * f32
            + 8 * e_pad * 4
            - m * (h * bf16 + 128 * f32)
        )
    ring = _weight_ring_depth(h, i, weight_dtype) * (3 * h * i * wt + (2 * i + h) * f32)
    shared = 3 * h * i_s * wt + (2 * i_s + h) * f32
    rows = m * (
        h * bf16  # hidden
        + (experts + (-experts) % 128) * f32  # coeff
        + 128 * f32  # shared gate
        + h * bf16  # output
        + h * wt  # quantized hidden
        + 2 * i * f32  # row scales, broadcast
        + h * f32  # accumulator
        + h * bf16  # accumulator in bf16, the all-reduce source
        + n_ranks * (h // n_ranks) * bf16  # reduce-scatter slots
        + (h // n_ranks) * bf16  # reduced chunk
    )
    return ring + shared + rows + extra


def _scratch(
    nb,
    m,
    h,
    i,
    i_s,
    experts,
    n_ranks,
    weight_dtype,
    wire,
    acc_bf16_shape,
    recv_shape,
    red_shape,
):
    """The scratch every mode declares, in _kernel's order."""
    return [
        pltpu.VMEM((nb, h, 2 * i), weight_dtype),
        pltpu.VMEM((nb, i, h), weight_dtype),
        pltpu.VMEM((nb, 1, 2 * i), jnp.float32),
        pltpu.VMEM((nb, 1, h), jnp.float32),
        pltpu.VMEM((h, 2 * i_s), weight_dtype),
        pltpu.VMEM((1, 2 * i_s), jnp.float32),
        pltpu.VMEM((i_s, h), weight_dtype),
        pltpu.VMEM((1, h), jnp.float32),
        pltpu.VMEM((m, h), weight_dtype),
        pltpu.VMEM((m, 2 * i), jnp.float32),
        pltpu.VMEM((m, h), jnp.float32),
        pltpu.VMEM(acc_bf16_shape, wire),
        pltpu.VMEM(recv_shape, wire),
        pltpu.VMEM(red_shape, wire),
        pltpu.SMEM((experts,), jnp.int32),
        pltpu.SemaphoreType.DMA((nb, 4)),
        pltpu.SemaphoreType.DMA((4,)),
        pltpu.SemaphoreType.DMA((n_ranks,)),
        pltpu.SemaphoreType.DMA(()),
        pltpu.SemaphoreType.DMA((n_ranks,)),
        pltpu.SemaphoreType.DMA(()),
    ]


def _gather_rows_call(
    rank,
    hidden,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    topk_weights,
    topk_ids,
    *,
    n_ranks,
    expert_start,
):
    """fused_moe_decode(gather_rows=True, scatter_rows=True): inputs are this
    rank's own rows; the kernel gathers them (see _kernel)."""
    rows, h = hidden.shape
    topk = topk_ids.shape[1]
    if 2 * topk + 1 > 128:
        raise ValueError(f"gather_rows packs routing in 128 lanes; topk {topk}")
    tokens = rows * n_ranks
    m = tokens + (-tokens) % 16
    r16 = rows + (-rows) % 16
    experts, i, _ = w2.shape
    e_pad = experts + (-experts) % 128
    # Row-local work stays with the caller's rows: the global expert ids
    # (exact in FP32), the weights as the gathered path rounds them, and the
    # shared gate. Ids stay global: every receiver holds different experts.
    ids = topk_ids.astype(jnp.float32)
    start = jnp.zeros((), jnp.int32) if expert_start is None else expert_start
    rank_start = jnp.stack(
        [rank.reshape(()).astype(jnp.int32), jnp.asarray(start, jnp.int32).reshape(())]
    )
    weights = topk_weights.astype(hidden.dtype).astype(jnp.float32)
    # A one-row [1, H] x [H, 1] gate dot came out of XLA with other bits than
    # the gathered path's [T, H] one (measured at H 4096); 2 and 4 rows
    # matched, so one row is padded to two for the dot only.
    gate_rows = jnp.pad(hidden, ((0, 1), (0, 0))) if rows == 1 else hidden
    shared_gate = jax.nn.sigmoid(
        jnp.dot(
            gate_rows,
            shared_gate_weight.astype(hidden.dtype),
            preferred_element_type=jnp.float32,
        )
    )[:rows]
    route = jnp.concatenate([ids, weights, shared_gate], axis=1)
    route = jnp.pad(route, ((0, 0), (0, 128 - route.shape[1])))
    i_s = shared_w2.shape[0]
    weight_dtype = w1.dtype
    nb = _weight_ring_depth(h, i, weight_dtype)
    # Rows the gather DMAs move, see _kernel: BF16 when an even count under
    # 16 or whole tiles, else FP32 in whole 8-row tiles above 8.
    if rows % 16 == 0 or (rows < 16 and rows % 2 == 0):
        xwire, sent = hidden.dtype, rows
    else:
        xwire, sent = jnp.float32, rows if rows <= 8 else rows + (-rows) % 8
    route_rows = rows if rows <= 8 else rows + (-rows) % 8  # FP32 slab, as x
    if route_rows != rows:
        route = jnp.pad(route, ((0, route_rows - rows), (0, 0)))
    # Reduce-scatter buffers exactly as scatter_rows on the gathered batch.
    wire = hidden.dtype if rows % 16 == 0 else jnp.float32
    out_rows = rows if rows % 16 == 0 else -(-rows // 8) * 8
    acc_bf16_shape = recv_shape = (n_ranks, out_rows, h)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            m=m,
            h=h,
            i=i,
            experts=experts,
            nb=nb,
            n_ranks=n_ranks,
            weight_dtype=weight_dtype,
            rows=rows,
            topk=topk,
            sent=sent,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, h), hidden.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * 8,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=_scratch(
                nb,
                m,
                h,
                i,
                i_s,
                experts,
                n_ranks,
                weight_dtype,
                wire,
                acc_bf16_shape,
                recv_shape,
                (8, 128),
            )
            + [
                pltpu.VMEM((n_ranks, r16, h), xwire),
                pltpu.VMEM((n_ranks, r16, 128), jnp.float32),
                pltpu.VMEM((m, 128), jnp.float32),
                pltpu.VMEM((m, e_pad), jnp.float32),
                pltpu.VMEM((8, e_pad), jnp.int32),
                pltpu.SMEM((8, e_pad), jnp.int32),
                pltpu.SemaphoreType.DMA((n_ranks,)),
                pltpu.SemaphoreType.DMA(()),
                pltpu.SemaphoreType.DMA(()),
                pltpu.VMEM((r16, h) if xwire != hidden.dtype else (8, 128), xwire),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=f"fused_moe_decode_ag-e_{experts}-m_{m}-h_{h}"
        f"-i_{i}-is_{i_s}-n_{n_ranks}-buffers_{nb}",
    )(
        rank_start,
        hidden,
        route,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
    )


def fused_moe_decode(
    rank,
    hidden,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    topk_weights,
    topk_ids,
    *,
    n_ranks,
    expert_start=None,
    scatter_rows=False,
    gather_rows=False,
):
    """Per-rank body; call inside shard_map over the TP mesh.

    Returns the TP-all-reduced (shared + routed) MoE output, BF16 [tokens,
    hidden], identical on every rank (with scatter_rows, only this rank's
    rows; see below).

    Args:
      rank: this rank's mesh index, an int32 array of one element
        (partition-id cannot be exported through torch_tpu).
      hidden: [tokens, H] BF16 hidden states, the same on every rank.
      w1: [E, H, 2 i] this rank's gate and up projections of its E experts,
        concatenated on the last axis; i is the rank's intermediate slice.
      w2: [E, i, H] the down projections.
      w1_scale: [E, 1, 1, 2 i] FP32 per-output-channel scales of w1.
      w2_scale: [E, 1, 1, H] FP32 per-output-channel scales of w2.
      shared_w1: [H, 2 i_s] the shared expert's gate and up slice (i_s <= i).
      shared_w1_scale: [1, 2 i_s] FP32 scales of shared_w1.
      shared_w2: [i_s, H] the shared expert's down slice.
      shared_w2_scale: [1, H] FP32 scales of shared_w2.
      shared_gate_weight: [H, 1] the replicated shared_expert_gate; the shared
        expert's output is scaled by sigmoid(hidden @ shared_gate_weight).
      topk_weights: [tokens, k] route weights from the caller's router.
      topk_ids: [tokens, k] int32 expert ids of those routes.
      n_ranks: ranks in the mesh, the all-reduce width.
      expert_start: with expert parallelism, the global id of this rank's
        first expert; the rank holds experts [expert_start, expert_start + E)
        and topk_ids are global. Routes to other ranks' experts contribute
        nothing here and are summed in by the all-reduce. None for TP experts.
      scatter_rows: for DP attention. The caller all-gathered every rank's
        tokens and routing rank-major, so `hidden` holds the global batch.
        The all-reduce becomes a reduce-scatter over token rows: rank r
        returns only its own rows [r * tokens / n_ranks, (r + 1) * tokens /
        n_ranks), [tokens / n_ranks, H], and no all-gather runs. Needs tokens
        % n_ranks == 0. The wire is BF16 when tokens / n_ranks is a multiple
        of 16, else FP32.
      gather_rows: with scatter_rows, for callers that have NOT all-gathered:
        `hidden`, `topk_weights` and `topk_ids` are this rank's own rows
        [rows, ...] and the kernel all-gathers them rank-major itself (remote
        DMA, BF16 hidden rows and an FP32 routing slab per rank), so the call
        computes exactly what scatter_rows computes on the gathered inputs
        and returns this rank's [rows, H]. No XLA collective runs.

    The weights' dtype (FP8 in production) is also the dtype the kernel
    quantizes the hidden states and the activated intermediate to, so both
    matmuls run in it with FP32 accumulation.
    """
    if gather_rows:
        if not scatter_rows:
            raise ValueError("gather_rows needs scatter_rows")
        return _gather_rows_call(
            rank,
            hidden,
            w1,
            w2,
            w1_scale,
            w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w2,
            shared_w2_scale,
            shared_gate_weight,
            topk_weights,
            topk_ids,
            n_ranks=n_ranks,
            expert_start=expert_start,
        )
    tokens, h = hidden.shape
    if scatter_rows and tokens % n_ranks:
        raise ValueError(f"scatter_rows needs tokens % n_ranks == 0; got {tokens}")
    rows = tokens // n_ranks if scatter_rows else None
    experts, i, _ = w2.shape
    pad = (-tokens) % 16
    if pad:
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
        topk_ids = jnp.pad(topk_ids, ((0, pad), (0, 0)))
    m = tokens + pad
    if expert_start is not None:
        topk_ids = topk_ids - expert_start
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(hidden.dtype).astype(jnp.float32)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(topk_ids.shape[1]):
        selected = topk_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, 0), (0, (-experts) % 128)))
    shared_gate = jax.nn.sigmoid(
        jnp.dot(
            hidden,
            shared_gate_weight.astype(hidden.dtype),
            preferred_element_type=jnp.float32,
        )
    )
    shared_gate = jnp.broadcast_to(shared_gate, (m, 128))
    i_s = shared_w2.shape[0]
    weight_dtype = w1.dtype
    nb = _weight_ring_depth(h, i, weight_dtype)
    chunk = h // n_ranks
    if rows is None:
        out_rows, wire = m, hidden.dtype
        acc_bf16_shape, recv_shape, red_shape = (m, h), (n_ranks, m, chunk), (m, chunk)
    else:
        # BF16 packs 16 rows per tile; smaller slabs go over the wire in FP32.
        wire = hidden.dtype if rows % 16 == 0 else jnp.float32
        out_rows = rows if rows % 16 == 0 else -(-rows // 8) * 8
        acc_bf16_shape = recv_shape = (n_ranks, out_rows, h)
        red_shape = (8, 128)  # unused
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            m=m,
            h=h,
            i=i,
            experts=experts,
            nb=nb,
            n_ranks=n_ranks,
            weight_dtype=weight_dtype,
            rows=rows,
        ),
        out_shape=jax.ShapeDtypeStruct((out_rows, h), wire),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 3
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * 8,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=_scratch(
                nb,
                m,
                h,
                i,
                i_s,
                experts,
                n_ranks,
                weight_dtype,
                wire,
                acc_bf16_shape,
                recv_shape,
                red_shape,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=f"fused_moe_decode{'_rs' if rows else ''}-e_{experts}-m_{m}-h_{h}"
        f"-i_{i}-is_{i_s}-n_{n_ranks}-buffers_{nb}",
    )(
        active,
        rank.reshape(1).astype(jnp.int32),
        hidden,
        coeff,
        shared_gate,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
    )
    if rows is not None:
        return out[:rows].astype(hidden.dtype)
    return out[:tokens] if pad else out
