"""R1 (WHITEBOARD.v1 §3): decode o_proj fp8 matmul fused with its TP all-reduce, one pallas_call.

    entry   start scale DMA + W tiles 0..nbuf-1 (all in flight) ; signal the barrier on every rank
            (self included) ; quantize x -> xq (fp8 values held as bf16), xs
    tile j  wait W_j ; y = xq . W_j (f32) ; o[:, tile j] = bf16(y * xs * ws[tile j])
            (L2) after tile 0: wait barrier, then push every owner whose 384-column chunk is complete
    RS      (L1) after the last tile: wait barrier ; push o[:, chunk t] -> rank t's recv[me] for t != me,
            local DMA o[:, chunk me] -> recv[me]
    sum     wait the 15 inbound chunks (one byte-counted semaphore); red = bf16(sum_{r=0..15} f32 recv[r])
            in rank order (small_all_reduce's sum, same ops, same order -> bitwise)
    AG      push red -> rank t's o[:, chunk me] for t != me ; local DMA red -> o[:, chunk me] ;
            wait 15 inbound chunks ; drain both send semaphores (every semaphore ends at 0)

Safety (as small_collectives.common.barrier): a rank writes into a peer only after the barrier of this
call, which the peer signals at entry -- after it finished the previous call; a peer's o[:, chunk me]
is overwritten by my AG only after I received that chunk from it (its RS read is complete).

Variants (static): barrier = "hidden" (signal at entry, wait before the first remote write) |
"exposed" (signal + wait at entry, before the weight stream: R-bar, timing arm) ; waves = False (L1) |
True (L2) ; exchange = False (R-ex: no RS/sum/AG, barrier still balanced; output = local partial;
timing-only) ; fault = None | "straddle" (L2 pushing owners 5 and 10 one tile early) | "no_rs_wait"
(sum before the RS bytes are counted; drained after) | "early_push" (RS pushes before the barrier wait).

APIs copied from api_ledger.md rows 1-4, 6 (barrier semaphore, make_async_remote_copy with dict MESH ids,
local-descriptor byte waits, fp8_skinny's ring + with_memory_space_constraint).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MESH = pl.DeviceIdType.MESH
COLLECTIVE_ID = 5  # = small_collectives.all_reduce.COLLECTIVE_ID (barrier group of the AR it replaces)
LANES = 128


def _peer(j, axis):
    return {axis: j}


def _wait_bytes(view, sem):
    pltpu.make_async_copy(view, view, sem).wait()


def owner_waves(n, n_out, t, n_tiles, fault=None):
    """Owner chunks pushable after each tile: owner j after tile ((j+1)*own - 1) // t (its last column);
    fault 'straddle' uses its first column instead (pushes a straddling owner one tile early)."""
    own = n_out // n
    waves = [[] for _ in range(n_tiles)]
    for j in range(n):
        col = j * own if fault == "straddle" else (j + 1) * own - 1
        waves[col // t].append(j)
    return waves


def _kernel(idx_ref, rank_ref, nv_ref, x_ref, ws_hbm, w_hbm, o_ref, xq_ref, xs_ref, wbuf, wsem,
            ws_vmem, ws_sem, recv, red, rs_send, rs_recv, ag_send, ag_recv, loc, *,
            n, axis, t, n_tiles, nbuf, q_dtype, barrier, waves, exchange, fault):
    del nv_ref  # every row moves (the incumbent's full path at every production cell)
    s = idx_ref[0]
    me = rank_ref[0]
    own = o_ref.shape[1] // n
    bsem = pltpu.get_barrier_semaphore()

    def signal_all():
        for j in range(n):
            pl.semaphore_signal(bsem, 1, device_id=_peer(jnp.int32(j), axis), device_id_type=MESH)

    def wait_barrier():
        pl.semaphore_wait(bsem, n)

    if barrier == "exposed":
        signal_all()
        wait_barrier()

    ws_copy = pltpu.make_async_copy(ws_hbm, ws_vmem, ws_sem)
    ws_copy.start()

    def copy(j):
        return pltpu.make_async_copy(w_hbm.at[s, :, pl.ds(j * t, t)], wbuf.at[j % nbuf], wsem.at[j % nbuf])

    for j in range(min(nbuf, n_tiles)):
        copy(j).start()
    if barrier == "hidden":
        signal_all()

    # activation quant: fp8_skinny._kernel verbatim
    q_max = float(jnp.finfo(q_dtype).max)
    x = x_ref[...]
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(jnp.float32) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_q = jnp.clip(x.astype(jnp.float32) * (jnp.float32(1.0) / scale), -q_max, q_max).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = scale

    def chunk(j):  # static owner chunk of o_ref
        return o_ref.at[:, pl.ds(j * own, own)]

    def push_owner(j):
        @pl.when(jnp.int32(j) != me)
        def _():
            pltpu.make_async_remote_copy(
                src_ref=chunk(j), dst_ref=recv.at[me], send_sem=rs_send, recv_sem=rs_recv,
                device_id=_peer(jnp.int32(j), axis), device_id_type=MESH).start()

        @pl.when(jnp.int32(j) == me)
        def _():
            pltpu.make_async_copy(chunk(j), recv.at[j], loc.at[0]).start()

    wave_of = owner_waves(n, o_ref.shape[1], t, n_tiles, fault) if (waves and exchange) else None
    barrier_waited = barrier == "exposed"
    if exchange and fault == "early_push" and not barrier_waited:
        barrier_waited = None  # pushes go out without the wait; the wait still runs (balanced) later

    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        copy(j).wait()
        w = wbuf[j % nbuf]
        cols = slice(j * t, (j + 1) * t)
        y = jax.lax.dot_general(xq_ref[...], w, dn, preferred_element_type=jnp.float32)
        if j == 0:
            ws_copy.wait()
        o_ref[:, cols] = (y * xs_ref[...] * ws_vmem[:, cols]).astype(o_ref.dtype)
        if j + nbuf < n_tiles:
            copy(j + nbuf).start()
        if wave_of is not None:
            if barrier_waited is False:
                wait_barrier()
                barrier_waited = True
            for o in wave_of[j]:
                push_owner(o)

    if not exchange:
        if barrier == "hidden":
            wait_barrier()
        return

    if wave_of is None:
        if barrier_waited is False:
            wait_barrier()
            barrier_waited = True
        for o in range(n):
            push_owner(o)
    if barrier_waited is None:  # early_push fault: balance the barrier after the pushes
        wait_barrier()

    slots = recv.at[pl.ds(0, n - 1)]  # byte count of 15 chunks
    _wait_bytes(recv.at[0], loc.at[0])  # own chunk (local DMA)
    if fault != "no_rs_wait":
        _wait_bytes(slots, rs_recv)
    acc = recv[0].astype(jnp.float32)
    for r in range(1, n):
        acc = acc + recv[r].astype(jnp.float32)
    red[...] = acc.astype(red.dtype)
    if fault == "no_rs_wait":
        _wait_bytes(slots, rs_recv)

    dst_me = o_ref.at[:, pl.ds(pl.multiple_of(me * own, LANES), own)]
    for j in range(n):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            pltpu.make_async_remote_copy(
                src_ref=red, dst_ref=dst_me, send_sem=ag_send, recv_sem=ag_recv,
                device_id=_peer(jnp.int32(j), axis), device_id_type=MESH).start()

    own_ag = pltpu.make_async_copy(red, dst_me, loc.at[1])
    own_ag.start()
    own_ag.wait()
    _wait_bytes(slots, ag_recv)
    _wait_bytes(slots, rs_send)
    _wait_bytes(slots, ag_send)


@functools.cache
def _build(m, k, n_out, n_stack, w_dtype, x_dtype, n, axis, barrier, waves, exchange, fault,
           t=2048, nbuf=3, interpret=False):
    n_tiles = n_out // t
    nbuf = max(1, min(nbuf, n_tiles))
    own = n_out // n
    if n_out % t or t % LANES or n_out % n or own % LANES:
        raise ValueError(f"[{k}, {n_out}] / tile {t} / {n} ranks: not lane-aligned")
    scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),  # xq
        pltpu.VMEM((m, 1), jnp.float32),  # xs
        pltpu.VMEM((nbuf, k, t), w_dtype),  # W ring
        pltpu.SemaphoreType.DMA((nbuf,)),
        pltpu.VMEM((1, n_out), jnp.float32),  # ws
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((n, m, own), x_dtype),  # recv slots, indexed by sender rank
        pltpu.VMEM((m, own), x_dtype),  # red
        pltpu.SemaphoreType.DMA,  # rs_send
        pltpu.SemaphoreType.DMA,  # rs_recv
        pltpu.SemaphoreType.DMA,  # ag_send
        pltpu.SemaphoreType.DMA,  # ag_recv
        pltpu.SemaphoreType.DMA((2,)),  # local copies
    ]
    kern = functools.partial(_kernel, n=n, axis=axis, t=t, n_tiles=n_tiles, nbuf=nbuf, q_dtype=w_dtype,
                             barrier=barrier, waves=waves, exchange=exchange, fault=fault)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        in_specs=[vmem, pl.BlockSpec(memory_space=pltpu.HBM), pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=vmem,
        scratch_shapes=scratch,
    )
    tag = ("l2" if waves else "l1") + ("" if barrier == "hidden" else "_bx") + ("" if exchange else "_nox")
    tag += f"_f{fault}" if fault else ""
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n_out), x_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"fp8_oproj_ar_{tag}_m{m}_k{k}_n{n_out}",
        **({"interpret": pltpu.InterpretParams()} if interpret else {}),
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_constrained(call, idx, rank, nv, x, w_s2, w_stack):
    return call(idx, rank, nv, x, pltpu.with_memory_space_constraint(w_s2, pltpu.HBM),
                pltpu.with_memory_space_constraint(w_stack, pltpu.HBM))


def fused_oproj_all_reduce(x, w_stack, idx, w_s, rank, n_valid, *, num_cores, axis, barrier="hidden",
                           waves=False, exchange=True, fault=None, interpret=False):
    """x bf16 [M <= 16, K]; w_stack fp8 [L, K, N]; idx/rank/n_valid int32 [1]; w_s f32 [N]
    -> bf16 [M, N], the sum over the num_cores ranks of axis of fp8_skinny(x, w_stack[idx], w_s)."""
    m, k = x.shape
    n_stack, _, n_out = w_stack.shape
    call = _build(m, k, n_out, n_stack, w_stack.dtype, x.dtype, num_cores, axis, barrier, waves,
                  exchange, fault, interpret=interpret)
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    if interpret:  # the CPU interpreter has no memory-space constraint lowering
        return call(idx.astype(jnp.int32), rank.astype(jnp.int32), n_valid.astype(jnp.int32), x, w_s2,
                    w_stack)
    return _call_constrained(call, idx.astype(jnp.int32), rank.astype(jnp.int32),
                             n_valid.astype(jnp.int32), x, w_s2, w_stack)
