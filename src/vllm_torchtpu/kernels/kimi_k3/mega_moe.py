"""Kimi-K3 MoE block as one Pallas kernel: gate + topk + down_proj + shared MLP
+ routed experts + all-reduce + norm + up_proj (the full non-attention block).

Single-kernel prototype from MEGA_MOE_KERNEL_PLAN.md, collapsing stages S1
(router gate + grouped topk), S2 (latent down_proj), S3 (routed experts), S4
(shared MLP) and S5 (collective tail) behind ONE custom-call boundary.

The point of the fusion is not the GEMMs, it is the OVERLAP. The non-GEMM work
in this block (sigmoid + top-16 selection over 224 experts, routing masks, the
all-reduce barrier) is latency-bound; today XLA schedules each as its own ops
and only streams a GEMM's weights when that GEMM begins. Here the kernel owns
every DMA: grid step 0 primes every ring in consumption order -- down_proj,
shared gate_up/down, experts, and finally up_proj, whose 24.5 MiB stream is
therefore completely hidden under the rest of the block. The topk resolves
while the down/shared streams fly; the expert rings are queued immediately
after; the all-reduce runs when the expert accumulation completes.

Routed experts follow the sglang-jax fused-MoE v2 *pattern* rather than its
code: tokens stay resident in VMEM (no HBM permute -- the 6.81 us/layer the
model pays for gather/scatter simply does not exist here), each local expert's
weights stream through a rolling ring, and the dot is DENSE over all T tokens
with the routing applied as an f32 row mask on the output. At decode (T=8)
the MXU is ~idle and the stage is purely DMA-bound, so the wasted MACs are
free and the schedule stays fully static. gmm_v2 was not reused because its
emit_pipeline owns a whole pallas_call and its SparseCore permute is a
separate custom call; neither can live inside this grid (plan sec. 3/5).

The collective is a ONE-SHOT all-reduce, not v4's two-shot: v4's
reduce-scatter needs the latent reblocked into per-device column chunks
([nd, T, cols], lane-aligned), which our [T, L] VMEM accumulator cannot
produce without a transpose; at 56 KiB of payload the transfer is
skew-dominated, not bandwidth-dominated, so pushing each device's full partial
to every peer and summing locally is simpler and equivalent here. Barrier and
slot-double-buffering idioms are copied from all_reduce_norm_matmul_v4.

Caveat vs the model: the harness expert stage is now mxfp4 like the model
(packed uint8 pairs along K + f32 scales, block 32, in-VMEM dequant --
gmm_v2's dequant-before-matmul path), but still DENSE over ``n_local`` local
experts where the model is ragged (gmm_v2 skips empty groups), and the per-core
shapes are a proxy. The unfused arm pays XLA's dequant materialization
(~300 us/layer), which the model avoids by dequantizing inside gmm_v2 -- so
the fused-vs-unfused delta here OVERSTATES the win vs the model; treat the
expert-portion calibration accordingly.

Weight bytes per layer (harness defaults): gate 3.5 MiB (padded to 256
experts), down_proj 25.7 MiB, shared gate_up 22.0 MiB, shared down 11.0 MiB,
up_proj 24.5 MiB, experts n_local x (packed 0.66 + scales 0.17) x 2 = 1.7 MiB
each -- ~100 MiB streamed through 64 MiB of VMEM, which is why the ring
depths below are a schedule, not a convenience.

Measured on TPU7x via scripts/k3/moe_bench.py (slope over chain depth,
tokens=8). Without experts/tail (bn_d=1792/bn1=768/bn2=3584, bf16): fused
32.6 us vs 35.4-35.8 us unfused; priming disabled: 73.7 us. With experts
(bn1=384/ahead1=1 to fit VMEM): fused 49.1 us vs 64.7-64.8 unfused (-24%).
Full layer with the collective tail, meshed 8-dev (bnd=256/bn1=384/aheade=2/
bnup=512/aheadup=2): bf16 experts 81.7-83.0 us vs 87.8-88.8 unfused; mxfp4
experts 84.0 us vs 318 unfused (the unfused arm materializes dequants through
HBM; the fused kernel dequants in VMEM). The dequant is in-step serial --
pipelining it across experts (ping-pong dq buffers) measured ~12 us WORSE
(99.9): the pl.when parity branches + earlier ring waits cost more than the
VPU/MXU overlap buys. The bench runs all devices in lockstep, so the
all-reduce's cost here is its WORK, not its skew -- in the model the AR shows
8-68 us spread, and the fused kernel's primed up_proj ring keeps streaming
through that window, which this harness cannot capture.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = [
    "blockify", "prepare_gate", "quantize_fp4", "dequant_fp4", "mega_moe"
]

LANE = 128
QBLOCK = 32  # mxfp4: one f32 scale per 32 K-elements
FP4 = jnp.float4_e2m1fn


def quantize_fp4(w: jax.Array, block: int = QBLOCK):
    """``[..., K, N]`` bf16 -> (packed uint8 ``[..., K//2, N]``, f32 scale
    ``[..., K//block, N]``): 2 fp4 per byte along K (the second-minor axis),
    matching the direction ``pltpu.bitcast`` unpacks in-kernel.

    Same absmax/e2m1-max scheme as the model's fused_moe_gmm path; only the
    layout is ours (theirs is further transposed for gmm_v2's tiling). Dequant
    is ``bitcast(u8 -> fp4) -> f32 -> * scale -> bf16``.
    """
    k, n = w.shape[-2:]
    if k % block or k % 2:
        raise ValueError(f"need K%({block},2)==0, got {w.shape}")
    fp4_max = float(jnp.finfo(FP4).max)  # 6.0
    blocked = w.astype(jnp.float32).reshape(*w.shape[:-2], k // block, block,
                                            n)
    scale = jnp.max(jnp.abs(blocked), axis=-2, keepdims=True) / fp4_max
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    q = jnp.clip(blocked * scale_inv, -fp4_max, fp4_max).astype(FP4)
    q = q.reshape(*w.shape[:-2], k, n)
    # pltpu.bitcast's unpack order: byte (k2, n) holds q[2*k2] (lo) and
    # q[2*k2+1] (hi); its lowering is swapaxes-reshape of the pair axis.
    pairs = q.reshape(*w.shape[:-2], k // 2, 2, n)
    packed = jax.lax.bitcast_convert_type(jnp.swapaxes(pairs, -1, -2),
                                          jnp.uint8)  # [..., K/2, N]
    return packed, scale.astype(jnp.float32).squeeze(-2)


def dequant_fp4(packed: jax.Array, scale: jax.Array) -> jax.Array:
    """Inverse of :func:`quantize_fp4` (the reference-side dequant).

    Matches the kernel (and gmm_v2's dequant-before-matmul): fp4 -> bf16,
    scale -> bf16, multiply in bf16.
    """
    half_k, n = packed.shape[-2:]
    k = 2 * half_k
    lead = packed.shape[:-2]
    nqb = scale.shape[-2]
    w = jax.lax.bitcast_convert_type(packed, FP4)  # [..., K/2, N, 2]
    w = jnp.swapaxes(w, -1, -2).astype(jnp.bfloat16)  # [..., K/2, 2, N]
    w = w.reshape(*lead, k, n)
    w = w.reshape(*lead, nqb, k // nqb, n)
    w = w * scale.astype(jnp.bfloat16).reshape(*lead, nqb, 1, n)
    return w.reshape(*lead, k, n)


def blockify(w: jax.Array, bn: int) -> jax.Array:
    """``[K, N]`` -> ``[N // bn, K, bn]``. Done once, at weight load.

    Same contract as shared_mlp.blockify: from torch pass ``w.t()`` and finish
    with ``.contiguous()``.
    """
    k, n = w.shape
    if n % bn:
        raise ValueError(f"N ({n}) must be divisible by bn ({bn})")
    return w.reshape(k, n // bn, bn).transpose(1, 0, 2)


def prepare_gate(w: jax.Array, e_pad: int | None = None) -> jax.Array:
    """Router weight ``[E, H]`` (torch layout) -> ``[H, e_pad]`` zero-padded.

    The gate's output dim is 224 experts, which is not lane-aligned; pad to a
    multiple of 128 so the dot and the topk reductions stay on aligned refs.
    Padding columns are masked to -inf before the selection loop, so their
    value here is irrelevant.
    """
    e, h = w.shape
    if e_pad is None:
        e_pad = ((e + LANE - 1) // LANE) * LANE
    if e_pad < e or e_pad % LANE:
        raise ValueError(f"e_pad must be >= E ({e}) and a multiple of {LANE}, "
                         f"got {e_pad}")
    return jnp.pad(w.T, ((0, 0), (0, e_pad - e)))


def _situ(acc, d, beta, linear_beta):
    """beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)."""
    gate = acc[:, :d].astype(jnp.float32)
    up = acc[:, d:].astype(jnp.float32)
    gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * jnp.tanh(up / linear_beta)
    return gate * up


def _topk(scores, top_k, n_experts):
    """Top-``top_k`` of ``[T, e_pad]`` scores by iterative max extraction.

    Columns >= n_experts are padding and masked out first. Ties break to the
    lower expert id, matching lax.top_k. Selection runs as max + (first index
    equal to the max) + mask, i.e. pure elementwise ops and reductions -- no
    argsort/sort, which Mosaic does not lower. Cost is top_k passes over a
    [T, 256] tile: nothing, which is exactly why it can hide under the weight
    prefetch.
    """
    e_pad = scores.shape[1]
    cols = jnp.arange(e_pad, dtype=jnp.int32)
    work = jnp.where(cols[None, :] < n_experts, scores, -jnp.inf)
    ws, ids = [], []
    for _ in range(top_k):
        m = jnp.max(work, axis=1)  # [T]
        first = jnp.where(work == m[:, None], cols[None, :], e_pad)
        idx = jnp.min(first, axis=1)  # [T] int32
        ws.append(m)
        ids.append(idx)
        work = jnp.where(cols[None, :] == idx[:, None], -jnp.inf, work)
    w = jnp.stack(ws, axis=1)  # [T, top_k]
    w = w / jnp.sum(w, axis=1, keepdims=True)  # renormalize
    return w, jnp.stack(ids, axis=1)


def _full_barrier(my_id, num_devices: int):
    """All-to-all barrier, doubled -- see all_reduce_norm_matmul_v2 for why."""
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
        x_ref, wg_hbm, wd_hbm, w1_hbm, w2_hbm, w13q_hbm, w13s_hbm, w2q_hbm,
        w2s_hbm, g_ref, redws_ref, step_ref, wup_hbm, g2_ref, tw_hbm, ti_hbm,
        lat_hbm, o_hbm, exp_hbm, tail_hbm, redws_out_ref, step_out_ref, normed,
        work, tw_vmem, ti_vmem, match, lat_acc, act1, act, acc2, exp_acc,
        exp_out, wg_vmem, wd_ring, w1_ring, w2_ring, w13q_ring, w13s_ring,
        w2q_ring, w2s_ring, dq13, dq2, recv_vmem, normed2, tail_acc, wup_ring,
        sem_g, sem_d, sem1, sem2, sem_e13, sem_e13s, sem_e2, sem_e2s, sem_up,
        sem_u2, sem_out, sem_local, s_send, s_recv, *, steps_d, steps1, steps2,
        bn_d, bn1, bn2, ahead_d, ahead1, ahead2, depth_d, depth1, depth2,
        n_experts, top_k, n_local, ahead_e, depth_e, with_tail, num_devices,
        axis_name, steps_up, bn_up, ahead_up, depth_up, steady_barrier, l_out,
        union_up, n_roll, u_slots, eps, do_norm, beta, linear_beta, precision):
    pid = pl.program_id(0)
    d_lo = 1  # first down_proj step
    g_lo = d_lo + steps_d  # first shared gate_up step
    n_lo = g_lo + steps1  # first shared down step
    e_lo = n_lo + steps2  # first expert step
    t_lo = e_lo + n_local  # all-reduce step (with_tail)
    # with_tail adds the AR step itself plus steps_up dot steps.
    total = t_lo + (1 + steps_up if with_tail else 0)

    def cp_d(i):
        r = lax.rem(i, depth_d)
        return pltpu.make_async_copy(wd_hbm.at[i], wd_ring.at[r], sem_d.at[r])

    def cp1(i):
        r = lax.rem(i, depth1)
        return pltpu.make_async_copy(w1_hbm.at[i], w1_ring.at[r], sem1.at[r])

    def cp2(i):
        r = lax.rem(i, depth2)
        return pltpu.make_async_copy(w2_hbm.at[i], w2_ring.at[r], sem2.at[r])

    def cp_e13(i):
        """Expert i's packed w13 + scales; packed on sem_e13, scale on sem_e13s."""
        r = lax.rem(i, depth_e)
        return (pltpu.make_async_copy(w13q_hbm.at[i], w13q_ring.at[r],
                                      sem_e13.at[r]),
                pltpu.make_async_copy(w13s_hbm.at[i], w13s_ring.at[r],
                                      sem_e13s.at[r]))

    def cp_e2(i):
        r = lax.rem(i, depth_e)
        return (pltpu.make_async_copy(w2q_hbm.at[i], w2q_ring.at[r],
                                      sem_e2.at[r]),
                pltpu.make_async_copy(w2s_hbm.at[i], w2s_ring.at[r],
                                      sem_e2s.at[r]))

    def cp_up(i):
        r = lax.rem(i, depth_up)
        return pltpu.make_async_copy(wup_hbm.at[i], wup_ring.at[r],
                                     sem_up.at[r])

    def cp_up_union(i, h):
        """Column half ``h`` of up block ``i`` into a dead down-ring slot.

        Requires hidden == 2*l_out and bn_up == 2*bn_d: a [hidden, bn_d] down
        slot is two [l_out, bn_d] row-halves, and a [l_out, bn_up] up block is
        two [l_out, bn_d] column-halves. (A raw reshape would be rejected --
        Mosaic requires the minormost dim unchanged -- so the union view is
        built from slices instead.) Only legal once the down phase is done
        (pid >= g_lo) -- grid order guarantees it. h=0 reuses sem_d (every
        down slot's wait has already consumed its signal); h=1 uses sem_u2.
        """
        k = i - n_roll
        return pltpu.make_async_copy(wup_hbm.at[i, :,
                                                pl.ds(h * bn_d, bn_d)],
                                     wd_ring.at[k,
                                                pl.ds(h * l_out, l_out), :],
                                     sem_d.at[k] if h == 0 else sem_u2.at[k])

    @pl.when(pid == 0)
    def _prologue():
        # Every stream is primed BEFORE any compute, in consumption order:
        # down_proj / shared-MLP rings first, then expert rings, then the
        # up_proj ring whose blocks are only needed after the all-reduce.
        # The gate dot, topk loop and match-matrix build below run while all
        # of it streams.
        cpg = pltpu.make_async_copy(wg_hbm, wg_vmem, sem_g)
        cpg.start()
        for k in range(ahead_d):
            cp_d(k).start()
        for k in range(ahead1):
            cp1(k).start()
        for k in range(ahead2):
            cp2(k).start()
        for k in range(ahead_e):
            for c in cp_e13(k) + cp_e2(k):
                c.start()
        if with_tail:
            for k in range(ahead_up):
                cp_up(k).start()

        f = x_ref[...].astype(jnp.float32)
        if do_norm:
            f = f * lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + eps)
        normed[...] = (f * g_ref[...].astype(jnp.float32)).astype(x_ref.dtype)

        cpg.wait()
        logits = lax.dot_general(normed[...],
                                 wg_vmem[...], (((1, ), (0, )), ((), ())),
                                 preferred_element_type=jnp.float32,
                                 precision=precision)
        w, ids = _topk(jax.nn.sigmoid(logits), top_k, n_experts)
        tw_vmem[...] = w
        ti_vmem[...] = ids

        # Token-expert match matrix: match[e, t] = combined topk weight of
        # token t on local expert e. Static loop over the (few) local experts,
        # vector ops over [T, top_k] -- cheap and hidden under the DMAs.
        ids_local = lax.rem(ids, n_local)
        rows = []
        for e in range(n_local):
            rows.append(jnp.sum(jnp.where(ids_local == e, w, 0.0), axis=1))
        match[...] = jnp.stack(rows, axis=0)
        exp_acc[...] = jnp.zeros_like(exp_acc)

        for dst, src in ((tw_hbm, tw_vmem), (ti_hbm, ti_vmem)):
            cp = pltpu.make_async_copy(src, dst, sem_out)
            cp.start()
            cp.wait()

    @pl.when((pid >= d_lo) & (pid < g_lo))
    def _down():
        j = pid - d_lo
        nxt = j + ahead_d

        @pl.when(nxt < steps_d)
        def _():
            cp_d(nxt).start()

        cp_d(j).wait()
        lat_acc[:, pl.ds(j * bn_d, bn_d)] = lax.dot_general(
            normed[...],
            wd_ring[lax.rem(j, depth_d)], (((1, ), (0, )), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision).astype(lat_acc.dtype)

    @pl.when((pid >= g_lo) & (pid < n_lo))
    def _gate_up():
        j = pid - g_lo
        nxt = j + ahead1

        @pl.when(nxt < steps1)
        def _():
            cp1(nxt).start()

        cp1(j).wait()
        act1[:, pl.ds(j * bn1, bn1)] = lax.dot_general(
            normed[...],
            w1_ring[lax.rem(j, depth1)], (((1, ), (0, )), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision).astype(act1.dtype)

    @pl.when(pid == n_lo - 1)
    def _transition():
        act[...] = _situ(act1[...], act.shape[1], beta,
                         linear_beta).astype(act.dtype)

    @pl.when((pid >= n_lo) & (pid < e_lo))
    def _down_shared():
        j = pid - n_lo
        nxt = j + ahead2

        @pl.when(nxt < steps2)
        def _():
            cp2(nxt).start()

        cp2(j).wait()
        acc2[:, pl.ds(j * bn2, bn2)] = lax.dot_general(
            act[...],
            w2_ring[lax.rem(j, depth2)], (((1, ), (0, )), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision).astype(acc2.dtype)

    @pl.when((pid >= e_lo) & (pid < t_lo))
    def _experts():
        e = pid - e_lo
        nxt = e + ahead_e

        @pl.when(nxt < n_local)
        def _():
            for c in cp_e13(nxt) + cp_e2(nxt):
                c.start()

        # mxfp4 dequant in VMEM (gmm_v2's dequant-before-matmul path:
        # fp4 -> bf16, x bf16 scale), in-step serial: packed tile lands,
        # dequant to the dq scratch, dot. MEASURED: pipelining the dequant
        # across expert steps (ping-pong dq buffers, dequant e+1 before the
        # dots of e) was ~12 us SLOWER (99.9 vs 88.0 us/layer full-kernel) --
        # the extra pl.when parity branches and the earlier ring wait cost
        # more than the VPU/MXU overlap buys. The phase is dequant-latency
        # bound: expert ring depth makes no difference (88.0-88.9 across
        # ahead_e 1..3).
        def _dequant_into(dst, q_ref, s_ref, ring_slot):
            k2 = 2 * q_ref.shape[1]
            n2 = q_ref.shape[2]
            nqb = k2 // QBLOCK
            w4 = pltpu.bitcast(q_ref[ring_slot][...], FP4)
            wsc = s_ref[ring_slot][...]
            dst[...] = (
                w4.astype(dst.dtype).reshape(k2, n2).reshape(nqb, QBLOCK, n2) *
                wsc.astype(dst.dtype).reshape(nqb, 1, n2)).reshape(k2, n2)

        s = lax.rem(e, depth_e)
        for c in cp_e13(e) + cp_e2(e):
            c.wait()
        _dequant_into(dq13, w13q_ring, w13s_ring, s)
        gu = lax.dot_general(lat_acc[...],
                             dq13[...], (((1, ), (0, )), ((), ())),
                             preferred_element_type=jnp.float32,
                             precision=precision)
        a = _situ(gu, gu.shape[1] // 2, beta,
                  linear_beta).astype(lat_acc.dtype)
        _dequant_into(dq2, w2q_ring, w2s_ring, s)
        y = lax.dot_general(a,
                            dq2[...], (((1, ), (0, )), ((), ())),
                            preferred_element_type=jnp.float32,
                            precision=precision)
        exp_acc[...] = exp_acc[...] + y * match[e][:, None].astype(jnp.float32)

    if with_tail and union_up:
        # Once the last down_proj dot retires its ring is dead VMEM. Refill
        # the slots with the TAIL of the up_proj stream (blocks n_roll..),
        # which then lands during the shared/expert phases instead of after
        # the all-reduce.
        @pl.when(pid == g_lo)
        def _union_prime():
            for k in range(u_slots):
                cp_up_union(n_roll + k, 0).start()
                cp_up_union(n_roll + k, 1).start()

    if with_tail:

        @pl.when(pid == t_lo)
        def _all_reduce():
            # One-shot all-reduce of exp_acc across the mesh, then RMS norm.
            # exp_hbm doubles as the staging buffer peers read from AND the
            # kernel's expert_out output. redws/step are used on the OUTPUT
            # side (they alias the inputs -- same idiom as v4).
            my_id = lax.axis_index(axis_name) if num_devices > 1 else 0
            step = step_out_ref[0]
            slot = lax.rem(step, 2)
            exp_out[...] = exp_acc[...].astype(exp_out.dtype)

            if num_devices > 1:
                stage = pltpu.make_async_copy(exp_out, exp_hbm, sem_local)
                stage.start()
                stage.wait()

                @pl.when(step == 0)
                def _():
                    _full_barrier(my_id, num_devices)

                copies = []
                for i in range(1, num_devices):
                    peer = lax.rem(my_id + i, num_devices)
                    c = pltpu.make_async_remote_copy(
                        src_ref=exp_hbm,
                        dst_ref=redws_out_ref.at[slot, my_id],
                        send_sem=s_send.at[slot],
                        recv_sem=s_recv.at[slot],
                        device_id=(peer, ),
                        device_id_type=pl.DeviceIdType.MESH)
                    c.start()
                    copies.append(c)

                local = pltpu.make_async_copy(exp_hbm, redws_out_ref.at[slot,
                                                                        my_id],
                                              sem_local)
                local.start()

                if steady_barrier:

                    @pl.when(step != 0)
                    def _():
                        _full_barrier(my_id, num_devices)

                for c in copies:
                    c.wait()
                local.wait()

                pull = pltpu.make_async_copy(redws_out_ref.at[slot], recv_vmem,
                                             sem_local)
                pull.start()
                pull.wait()
                acc = recv_vmem[0].astype(jnp.float32)
                for i in range(1, num_devices):
                    acc = acc + recv_vmem[i].astype(jnp.float32)
            else:
                # Identity all-reduce: also useful for the CPU interpret test.
                # Round through bf16 exactly like the nd>1 path (which sums
                # bf16-staged partials), so numerics are uniform in nd.
                stage = pltpu.make_async_copy(exp_out, exp_hbm, sem_local)
                stage.start()
                stage.wait()
                acc = exp_out[...].astype(jnp.float32)
            inv = lax.rsqrt(jnp.mean(acc * acc, axis=-1, keepdims=True) + eps)
            normed2[...] = (acc * inv *
                            g2_ref[...].astype(jnp.float32)).astype(
                                normed2.dtype)
            step_out_ref[0] = step + 1

        @pl.when(pid > t_lo)
        def _up_proj():
            j = pid - t_lo - 1

            @pl.when(j < n_roll)
            def _roll():
                nxt = j + ahead_up

                @pl.when(nxt < n_roll)
                def _():
                    cp_up(nxt).start()

                cp_up(j).wait()
                tail_acc[:, pl.ds(j * bn_up, bn_up)] = lax.dot_general(
                    normed2[...],
                    wup_ring[lax.rem(j, depth_up)], (((1, ), (0, )), ((), ())),
                    preferred_element_type=jnp.float32,
                    precision=precision).astype(tail_acc.dtype)

            if union_up:

                @pl.when(j >= n_roll)
                def _union():
                    cp_up_union(j, 0).wait()
                    cp_up_union(j, 1).wait()
                    k = j - n_roll
                    for h in (0, 1):
                        tail_acc[:,
                                 pl.ds(j * bn_up +
                                       h * bn_d, bn_d)] = (lax.dot_general(
                                           normed2[...],
                                           wd_ring[k,
                                                   pl.ds(h * l_out, l_out), :],
                                           (((1, ), (0, )), ((), ())),
                                           preferred_element_type=jnp.float32,
                                           precision=precision).astype(
                                               tail_acc.dtype))

    @pl.when(pid == total - 1)
    def _epilogue():
        if with_tail:
            for dst, src in ((lat_hbm, lat_acc), (o_hbm, acc2), (tail_hbm,
                                                                 tail_acc)):
                cp = pltpu.make_async_copy(src, dst, sem_out)
                cp.start()
                cp.wait()
        else:
            exp_out[...] = exp_acc[...].astype(exp_out.dtype)
            for dst, src in ((lat_hbm, lat_acc), (o_hbm, acc2), (exp_hbm,
                                                                 exp_out)):
                cp = pltpu.make_async_copy(src, dst, sem_out)
                cp.start()
                cp.wait()


def mega_moe(x: jax.Array,
             w_gate: jax.Array,
             w_down: jax.Array,
             w_gate_up: jax.Array,
             w_dn: jax.Array,
             w13q: jax.Array,
             w13s: jax.Array,
             w2q: jax.Array,
             w2s: jax.Array,
             gamma: jax.Array,
             *,
             n_experts: int,
             top_k: int = 16,
             bn_d: int = 1792,
             bn1: int = 384,
             bn2: int = 3584,
             ahead_d: int = 1,
             ahead1: int = 1,
             ahead2: int = 2,
             ahead_e: int = 2,
             tail=None,
             eps: float = 1e-5,
             do_norm: bool = False,
             beta: float = 4.0,
             linear_beta: float | None = 25.0,
             precision=None,
             vmem_limit_bytes: int | None = None,
             interpret: bool = False):
    """One fused pass over the MoE block, optionally including the tail.

    ``x`` is ``[T, H]``; ``w_gate`` is :func:`prepare_gate` output
    ``[H, e_pad]``; ``w_down``/``w_gate_up``/``w_dn`` are :func:`blockify`
    output ``[steps, K, bn]``; the routed-expert weights are mxfp4:
    ``w13q``/``w2q`` are packed uint8 ``[n_local, K, N//2]`` and
    ``w13s``/``w2s`` their f32 scales ``[n_local, K//32, N]`` (see
    :func:`quantize_fp4_nmajor`), with ``K = L`` (the per-core latent shard)
    and ``N = 2*I_sh`` for w13, ``K = I_sh`` and ``N = L_out`` for w2.
    Returns ``(topk_weights [T, top_k] f32, topk_ids [T, top_k] int32,
    latent [T, L], shared_out [T, H], expert_out [T, L_out])``; with
    ``tail`` given, additionally ``(tail_out [T, n_up], red_ws, step)``.

    ``tail`` is a dict with keys: ``red_ws`` ([2, nd, T, L_out] bf16 aliased
    workspace), ``step`` ([1] int32 aliased counter), ``w_up`` (blockified
    ``[steps, L_out, bn_up]``), ``gamma2`` ([1, L_out]), ``axis_name``,
    ``num_devices``, and optional ``bn_up``/``ahead_up``/``steady_barrier``.
    At num_devices == 1 the all-reduce degenerates to identity (used by the
    CPU interpret-mode test).

    Routing is modeled per core: global expert id ``g`` maps to local expert
    ``g % n_local``, and ``expert_out[t] = sum_e match[e, t] * ffn_e(lat[t])``.
    The single-device harness treats every token's experts as local.

    ``ahead_*`` is the ring prefetch depth issued at grid step 0; setting one
    to 0 turns that stream into issue-at-use, which is the A/B knob for how
    much the priming-during-topk overlap is worth.

    Numerics: topk and the shared MLP are bit-exact against the jnp reference
    in every config. ``latent`` is bit-exact only at ``bn_d=256``; wider
    down_proj blocks make Mosaic lower the dot with a different K-accumulation
    order than XLA's, worth ~1.5 ulp at bf16 (5.8e-3 max rel at bn_d=1792),
    and the tail's up_proj dot has the same wide-dot property.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [tokens, hidden], got {x.shape}")
    tokens, hidden = x.shape
    e_h, e_pad = w_gate.shape
    if e_h != hidden:
        raise ValueError(f"gate contracts {e_h}, x has hidden={hidden}")
    steps_d, k_d, b_d = w_down.shape
    steps1, k1, b1 = w_gate_up.shape
    steps2, k2, b2 = w_dn.shape
    if (b_d, b1, b2) != (bn_d, bn1, bn2):
        raise ValueError(f"weights blocked at bn=({b_d}, {b1}, {b2}), called "
                         f"with ({bn_d}, {bn1}, {bn2})")
    if k_d != hidden or k1 != hidden:
        raise ValueError(
            f"down/gate_up contract ({k_d}, {k1}), hidden={hidden}")
    latent = steps_d * bn_d
    two_i = steps1 * bn1
    if two_i % 2:
        raise ValueError(f"gate_up output {two_i} must be even (gate|up)")
    inter = two_i // 2
    if k2 != inter:
        raise ValueError(
            f"shared down contracts {k2}, intermediate is {inter}")
    if steps2 * bn2 != hidden:
        raise ValueError(
            f"shared down output {steps2 * bn2} != hidden {hidden}")
    n_local, half_k13, n13 = w13q.shape
    k13 = 2 * half_k13
    n_local2, half_k2, l_out = w2q.shape
    k2e = 2 * half_k2
    if w13q.dtype != jnp.uint8 or w2q.dtype != jnp.uint8:
        raise ValueError(f"expert weights must be packed uint8, got "
                         f"{w13q.dtype}/{w2q.dtype}")
    if n_local2 != n_local:
        raise ValueError(f"w13 has {n_local} experts, w2 has {n_local2}")
    if k13 != latent:
        raise ValueError(f"w13 contracts {k13}, latent is {latent}")
    if n13 % 2 or n13 // 2 != k2e:
        raise ValueError(f"w13 [.., {k13}, {n13}] and w2 [.., {k2e}, {l_out}] "
                         f"are not a gate_up/down pair")
    if w13s.shape != (n_local, k13 // QBLOCK, n13):
        raise ValueError(f"w13s must be ({n_local}, {k13 // QBLOCK}, {n13}), "
                         f"got {w13s.shape}")
    if w2s.shape != (n_local, k2e // QBLOCK, l_out):
        raise ValueError(f"w2s must be ({n_local}, {k2e // QBLOCK}, {l_out}), "
                         f"got {w2s.shape}")
    if (bn_d % LANE or bn1 % LANE or bn2 % LANE or e_pad % LANE or inter % LANE
            or k2e % LANE or l_out % LANE or n13 % LANE):
        raise ValueError(f"block widths, e_pad and expert dims must be "
                         f"multiples of {LANE}")
    if not (0 < top_k <= n_experts <= e_pad):
        raise ValueError(
            f"need 0 < top_k ({top_k}) <= n_experts ({n_experts}) "
            f"<= e_pad ({e_pad})")
    if not (0 < n_local <= n_experts):
        raise ValueError(f"need 0 < n_local ({n_local}) <= n_experts")
    if tuple(gamma.shape) != (1, hidden):
        raise ValueError(f"gamma must be (1, {hidden}), got {gamma.shape}")

    with_tail = tail is not None
    if with_tail:
        red_ws = tail["red_ws"]
        step = tail["step"]
        w_up = tail["w_up"]
        gamma2 = tail["gamma2"]
        axis_name = tail["axis_name"]
        num_devices = tail["num_devices"]
        bn_up = tail.get("bn_up", 512)
        ahead_up = tail.get("ahead_up", 1)
        steady_barrier = tail.get("steady_barrier", True)
        steps_up, k_up, b_up = w_up.shape
        if k_up != l_out or b_up != bn_up or bn_up % LANE:
            raise ValueError(f"w_up must be [steps, {l_out}, bn_up] with "
                             f"{LANE}-multiple bn_up, got {w_up.shape}")
        if tuple(gamma2.shape) != (1, l_out):
            raise ValueError(f"gamma2 must be (1, {l_out})")
        if red_ws.shape != (2, num_devices, tokens, l_out):
            raise ValueError(f"red_ws must be (2, {num_devices}, {tokens}, "
                             f"{l_out}), got {red_ws.shape}")
        ahead_up = min(ahead_up, steps_up)
        # Ring union: a dead down_proj slot [hidden, bn_d] holds an up_proj
        # block [l_out, bn_up] iff hidden == 2*l_out and bn_up == 2*bn_d. The
        # "view" is built from row/column-half slices -- Mosaic rejects
        # reshapes that change the minormost dim, so a raw ref.reshape is out.
        # The down ring then doubles as extra up-ring depth for the TAIL of
        # the up stream, refilled the moment the down phase retires
        # (pid == g_lo).
        #
        # MEASURED (TPU7x, lockstep bench, tuned config): 83.1-83.5 us with
        # the union vs 81.9 without -- neutral-to-worse. The up stream was
        # already just-in-time relative to the AR (the aheadup plateau), and
        # the union's strided half-block DMAs + doubled dot count for those
        # blocks cost more than earlier issue buys. Default OFF; the regime it
        # should help is AR skew (8-68 us in the model), which the harness
        # cannot reproduce -- revisit at the e2e A/B.
        depth_d_pre = min(min(ahead_d, steps_d) + 1, steps_d)
        union_opt = tail.get("union_up", False)
        union_ok = (hidden == 2 * l_out) and (bn_up == 2 * bn_d)
        if union_opt is None:
            union_up = union_ok
        else:
            union_up = bool(union_opt)
            if union_up and not union_ok:
                raise ValueError(f"union_up requires hidden == 2*l_out and "
                                 f"bn_up == 2*bn_d, got {hidden}/{l_out} and "
                                 f"{bn_up}/{bn_d}")
        u_slots = min(depth_d_pre, steps_up - 1) if union_up else 0
        n_roll = steps_up - u_slots
        if union_up and n_roll < 1:
            union_up, u_slots, n_roll = False, 0, steps_up
        ahead_up = min(ahead_up, n_roll)
        depth_up = min(ahead_up + 1, n_roll)
        n_up = steps_up * bn_up
    else:
        red_ws = jnp.zeros((1, 1, 1, 1), x.dtype)
        step = jnp.zeros((1, ), jnp.int32)
        w_up = jnp.zeros((1, 1, LANE), x.dtype)
        gamma2 = jnp.zeros((1, 1), x.dtype)
        axis_name = ""
        num_devices = 1
        bn_up = LANE
        ahead_up = steps_up = depth_up = 1
        steady_barrier = False
        union_up, u_slots, n_roll = False, 0, 1
        n_up = 0

    ahead_d = min(ahead_d, steps_d)
    ahead1 = min(ahead1, steps1)
    ahead2 = min(ahead2, steps2)
    ahead_e = min(ahead_e, n_local)
    depth_d = min(ahead_d + 1, steps_d)
    depth1 = min(ahead1 + 1, steps1)
    depth2 = min(ahead2 + 1, steps2)
    depth_e = min(ahead_e + 1, n_local)
    params = {}
    if vmem_limit_bytes is not None:
        params["vmem_limit_bytes"] = vmem_limit_bytes
    if with_tail:
        # Like v4: mark the collective pure for XLA but give it a stable id.
        params["has_side_effects"] = tail.get("has_side_effects", False)
        params["collective_id"] = tail.get("collective_id", 0)
    ANY = pl.BlockSpec(memory_space=pl.ANY)

    out_shapes = [
        jax.ShapeDtypeStruct((tokens, top_k), jnp.float32),
        jax.ShapeDtypeStruct((tokens, top_k), jnp.int32),
        jax.ShapeDtypeStruct((tokens, latent), x.dtype),
        jax.ShapeDtypeStruct((tokens, hidden), x.dtype),
        jax.ShapeDtypeStruct((tokens, l_out), x.dtype),
    ]
    in_specs = [
        pl.BlockSpec((tokens, hidden), lambda j: (0, 0)),  # x
        ANY,  # gate, HBM
        ANY,  # down, HBM
        ANY,  # gate_up, HBM
        ANY,  # down, HBM
        ANY,  # w13 packed, HBM
        ANY,  # w13 scales, HBM
        ANY,  # w2 packed, HBM
        ANY,  # w2 scales, HBM
        pl.BlockSpec((1, hidden), lambda j: (0, 0)),  # gamma
    ]
    inputs = [x, w_gate, w_down, w_gate_up, w_dn, w13q, w13s, w2q, w2s, gamma]
    aliases = {}
    if with_tail:
        in_specs += [
            ANY,  # red_ws in
            pl.BlockSpec(memory_space=pltpu.SMEM),  # step in
            ANY,  # w_up, HBM
            pl.BlockSpec((1, l_out), lambda j: (0, 0)),  # gamma2
        ]
        inputs += [red_ws, step, w_up, gamma2]
        out_shapes += [
            jax.ShapeDtypeStruct((tokens, n_up), x.dtype),
            jax.ShapeDtypeStruct(red_ws.shape, red_ws.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ]
        aliases = {
            len(inputs) - 4: len(out_shapes) - 2,
            len(inputs) - 3: len(out_shapes) - 1
        }
    else:
        in_specs += [
            ANY,
            pl.BlockSpec(memory_space=pltpu.SMEM), ANY,
            pl.BlockSpec((1, 1), lambda j: (0, 0))
        ]
        inputs += [red_ws, step, w_up, gamma2]
        out_shapes += [
            jax.ShapeDtypeStruct((1, ), x.dtype),
            jax.ShapeDtypeStruct(red_ws.shape, red_ws.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ]
    out_specs = (ANY, ANY, ANY, ANY, ANY, ANY, ANY,
                 pl.BlockSpec(memory_space=pltpu.SMEM))

    scratch = [
        pltpu.VMEM((tokens, hidden), x.dtype),  # normed
        pltpu.VMEM((tokens, e_pad), jnp.float32),  # topk work
        pltpu.VMEM((tokens, top_k), jnp.float32),  # topk weights
        pltpu.VMEM((tokens, top_k), jnp.int32),  # topk ids
        pltpu.VMEM((n_local, tokens), jnp.float32),  # match matrix
        pltpu.VMEM((tokens, latent), x.dtype),  # latent acc
        pltpu.VMEM((tokens, two_i), x.dtype),  # gate|up
        pltpu.VMEM((tokens, inter), x.dtype),  # post-situ
        pltpu.VMEM((tokens, hidden), x.dtype),  # shared out
        pltpu.VMEM((tokens, l_out), jnp.float32),  # expert acc
        pltpu.VMEM((tokens, l_out), x.dtype),  # expert out
        pltpu.VMEM((hidden, e_pad), x.dtype),  # gate weight
        pltpu.VMEM((depth_d, hidden, bn_d), x.dtype),  # down ring
        pltpu.VMEM((depth1, hidden, bn1), x.dtype),  # gate_up ring
        pltpu.VMEM((depth2, inter, bn2), x.dtype),  # down ring
        pltpu.VMEM((depth_e, latent // 2, n13), jnp.uint8),  # w13 packed ring
        pltpu.VMEM((depth_e, latent // QBLOCK, n13),
                   jnp.float32),  # w13 scale ring
        pltpu.VMEM((depth_e, k2e // 2, l_out), jnp.uint8),  # w2 packed ring
        pltpu.VMEM((depth_e, k2e // QBLOCK, l_out),
                   jnp.float32),  # w2 scale ring
        pltpu.VMEM((latent, n13), x.dtype),  # w13 dequant
        pltpu.VMEM((k2e, l_out), x.dtype),  # w2 dequant
    ]
    if with_tail:
        scratch += [
            pltpu.VMEM((num_devices, tokens, l_out), x.dtype),  # recv
            pltpu.VMEM((tokens, l_out), x.dtype),  # normed2
            pltpu.VMEM((tokens, n_up), x.dtype),  # tail acc
            pltpu.VMEM((depth_up, l_out, bn_up), x.dtype),  # up ring
        ]
    else:
        scratch += [
            pltpu.VMEM((1, 1, 1), x.dtype),
            pltpu.VMEM((1, 1), x.dtype),
            pltpu.VMEM((1, 1), x.dtype),
            pltpu.VMEM((1, 1, 1), x.dtype),
        ]
    scratch += [
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((depth_d, )),
        pltpu.SemaphoreType.DMA((depth1, )),
        pltpu.SemaphoreType.DMA((depth2, )),
        pltpu.SemaphoreType.DMA((depth_e, )),
        pltpu.SemaphoreType.DMA((depth_e, )),
        pltpu.SemaphoreType.DMA((depth_e, )),
        pltpu.SemaphoreType.DMA((depth_e, )),
        pltpu.SemaphoreType.DMA((depth_up, )),
        pltpu.SemaphoreType.DMA((depth_d, )),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((2, )),
        pltpu.SemaphoreType.DMA((2, )),
    ]

    grid = 1 + steps_d + steps1 + steps2 + n_local + (1 + steps_up
                                                      if with_tail else 0)
    out = pl.pallas_call(
        functools.partial(_kernel,
                          steps_d=steps_d,
                          steps1=steps1,
                          steps2=steps2,
                          bn_d=bn_d,
                          bn1=bn1,
                          bn2=bn2,
                          ahead_d=ahead_d,
                          ahead1=ahead1,
                          ahead2=ahead2,
                          depth_d=depth_d,
                          depth1=depth1,
                          depth2=depth2,
                          n_experts=n_experts,
                          top_k=top_k,
                          n_local=n_local,
                          ahead_e=ahead_e,
                          depth_e=depth_e,
                          with_tail=with_tail,
                          num_devices=num_devices,
                          axis_name=axis_name,
                          steps_up=steps_up,
                          bn_up=bn_up,
                          ahead_up=ahead_up,
                          depth_up=depth_up,
                          steady_barrier=steady_barrier,
                          l_out=l_out,
                          union_up=union_up,
                          n_roll=n_roll,
                          u_slots=u_slots,
                          eps=eps,
                          do_norm=do_norm,
                          beta=beta,
                          linear_beta=linear_beta,
                          precision=precision),
        grid=(grid, ),
        out_shape=tuple(out_shapes),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch,
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(**params) if params else None,
        interpret=interpret,
    )(*inputs)
    if with_tail:
        return tuple(out[:5]) + (out[5], out[6], out[7])
    return tuple(out[:5])
