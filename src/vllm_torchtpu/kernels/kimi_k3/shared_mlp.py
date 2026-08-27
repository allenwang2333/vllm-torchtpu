"""Kimi-K3 shared-expert MLP as one Pallas kernel: norm -> gate_up -> situ -> down.

Why fuse the two projections rather than write two kernels. A Pallas call is an
opaque custom call, so XLA can hoist nothing into or out of it and each
invocation pays its own cost: ~1.26 us custom-call boundary + ~1.53 us pipeline
fill + ~1.70 us dot setup, about 4.5 us. XLA pays ~0 because ops 1..N read their
weights as adjacent instructions in one scheduled program and the scheduler
starts fusion N+1's operand stream while fusion N computes -- the fill is paid
once per chain, not once per op. So Pallas only wins once the payload is big
enough to amortise that boundary; measured crossover is ~35-45 MiB, and the two
projections separately (21.0 and 10.5 MiB) both sit below it. Together they are
31.5 MiB behind ONE boundary, which is the whole point of this kernel.

Measured context at TP=8 decode (M=8), per layer: gate_up 12.6 us + down_proj
8.5 us = 21.1 us today. At Pallas's 2921 GB/s marginal rate plus one 4.6 us
fixed cost, 31.5 MiB should land near 15.9 us.

Layout. ``gate_up_proj`` is a ``MergedColumnParallelLinear(7168, [I]*2)``, so its
per-rank output is ``[gate | up]`` concatenated, each ``I/tp`` wide. Both weights
are pre-blocked over their OUTPUT dimension and streamed from HBM by hand, the
same rolled-grid + manual-ring shape proven in the fused collective: a Python
loop in the kernel body unrolls at trace time and inflates instruction count ~5x,
while ``grid=`` gives a rolled loop with XLA-parity codegen.

The two phases are sequential -- ``down_proj`` needs the complete intermediate --
so the grid runs ``steps1 + steps2`` iterations and branches on the step index.
The output stays in VMEM (it is only ``tokens x hidden``) and is DMA'd out once
at the end, which avoids giving the pipeliner an output block index that is
undefined during phase 1.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = ["blockify", "shared_mlp"]

LANE = 128


def blockify(w: jax.Array, bn: int) -> jax.Array:
    """``[K, N]`` -> ``[N // bn, K, bn]``. Done once, at weight load.

    From torch, the linear weight is ``[out, in]``; pass ``w.t()`` so K is the
    contracted dim. Finish the torch-side build with ``.contiguous()``, NOT
    ``.clone()`` -- clone preserves permuted strides and the op then
    re-materialises the whole weight on every call.
    """
    k, n = w.shape
    if n % bn:
        raise ValueError(f"N ({n}) must be divisible by bn ({bn})")
    return w.reshape(k, n // bn, bn).transpose(1, 0, 2)


def _situ(acc, d, beta, linear_beta):
    """beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)."""
    gate = acc[:, :d].astype(jnp.float32)
    up = acc[:, d:].astype(jnp.float32)
    gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * jnp.tanh(up / linear_beta)
    return gate * up


def _kernel(x_ref, w1_hbm, w2_hbm, g_ref, o_hbm, normed, acc1, act, acc2,
            w1_ring, w2_ring, sem1, sem2, out_sem, *, steps1, steps2, bn1, bn2,
            ahead1, ahead2, depth1, depth2, eps, do_norm, beta, linear_beta,
            precision):
    pid = pl.program_id(0)
    total = steps1 + steps2

    def cp1(i):
        r = lax.rem(i, depth1)
        return pltpu.make_async_copy(w1_hbm.at[i], w1_ring.at[r], sem1.at[r])

    def cp2(i):
        r = lax.rem(i, depth2)
        return pltpu.make_async_copy(w2_hbm.at[i], w2_ring.at[r], sem2.at[r])

    @pl.when(pid == 0)
    def _prologue():
        # Prime both rings up front: phase 2's first blocks then land during
        # phase 1 and the transition costs no fill.
        for k in range(ahead1):
            cp1(k).start()
        for k in range(ahead2):
            cp2(k).start()
        f = x_ref[...].astype(jnp.float32)
        if do_norm:
            f = f * lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + eps)
        normed[...] = (f * g_ref[...].astype(jnp.float32)).astype(x_ref.dtype)

    @pl.when(pid < steps1)
    def _phase1():
        nxt = pid + ahead1

        @pl.when(nxt < steps1)
        def _():
            cp1(nxt).start()

        cp1(pid).wait()
        acc1[:, pl.ds(pid * bn1, bn1)] = lax.dot_general(
            normed[...],
            w1_ring[lax.rem(pid, depth1)], (((1, ), (0, )), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision).astype(acc1.dtype)

    @pl.when(pid == steps1 - 1)
    def _transition():
        act[...] = _situ(acc1[...], act.shape[1], beta,
                         linear_beta).astype(act.dtype)

    @pl.when(pid >= steps1)
    def _phase2():
        j = pid - steps1
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

    @pl.when(pid == total - 1)
    def _epilogue():
        cp = pltpu.make_async_copy(acc2, o_hbm, out_sem)
        cp.start()
        cp.wait()


def shared_mlp(x: jax.Array,
               w_gate_up: jax.Array,
               w_down: jax.Array,
               gamma: jax.Array,
               *,
               bn1: int = 768,
               bn2: int = 512,
               ahead1: int = 2,
               ahead2: int = 4,
               eps: float = 1e-5,
               do_norm: bool = True,
               beta: float = 4.0,
               linear_beta: float | None = 25.0,
               precision=None,
               vmem_limit_bytes: int | None = None) -> jax.Array:
    """``x [M, H]``, pre-blocked weights, ``gamma [1, H]`` -> ``[M, H]`` partial.

    ``w_gate_up`` is ``blockify(W_gate_up.T, bn1)`` with ``W_gate_up`` the
    ``[2I, H]`` torch weight; ``w_down`` is ``blockify(W_down.T, bn2)`` with
    ``W_down`` the ``[H, I]`` torch weight. The result is this rank's PARTIAL
    contribution -- the caller still owes the tensor-parallel reduction
    (``shared_experts`` is built with ``reduce_results=False``).

    ``do_norm=False`` is the model path: the RMS norm that feeds the MoE block is
    shared with the router and the routed experts, so it is already materialised
    and ``gamma`` is already folded in. Pass ones for ``gamma`` there; the extra
    ``[M, H]`` multiply is ~0.03 us and keeps one code path.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [tokens, hidden], got {x.shape}")
    tokens, hidden = x.shape
    steps1, k1, b1 = w_gate_up.shape
    steps2, k2, b2 = w_down.shape
    if (b1, b2) != (bn1, bn2):
        raise ValueError(f"weights blocked at bn=({b1}, {b2}), called with "
                         f"({bn1}, {bn2})")
    if k1 != hidden:
        raise ValueError(f"gate_up contracts {k1}, x has hidden={hidden}")
    two_i = steps1 * bn1
    if two_i % 2:
        raise ValueError(f"gate_up output {two_i} must be even (gate|up)")
    inter = two_i // 2
    if k2 != inter:
        raise ValueError(f"down contracts {k2}, intermediate is {inter}")
    if steps2 * bn2 != hidden:
        raise ValueError(f"down output {steps2 * bn2} != hidden {hidden}")
    if inter % LANE or bn1 % LANE or bn2 % LANE:
        raise ValueError("intermediate and both block widths must be "
                         f"multiples of {LANE}; got {inter}, {bn1}, {bn2}")
    if tuple(gamma.shape) != (1, hidden):
        raise ValueError(f"gamma must be (1, {hidden}), got {gamma.shape}")

    ahead1 = min(ahead1, steps1)
    ahead2 = min(ahead2, steps2)
    depth1 = min(ahead1 + 1, steps1)
    depth2 = min(ahead2 + 1, steps2)
    params = {}
    if vmem_limit_bytes is not None:
        params["vmem_limit_bytes"] = vmem_limit_bytes
    return pl.pallas_call(
        functools.partial(_kernel,
                          steps1=steps1,
                          steps2=steps2,
                          bn1=bn1,
                          bn2=bn2,
                          ahead1=ahead1,
                          ahead2=ahead2,
                          depth1=depth1,
                          depth2=depth2,
                          eps=eps,
                          do_norm=do_norm,
                          beta=beta,
                          linear_beta=linear_beta,
                          precision=precision),
        grid=(steps1 + steps2, ),
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), x.dtype),
        in_specs=[
            pl.BlockSpec((tokens, hidden), lambda j: (0, 0)),  # x
            pl.BlockSpec(memory_space=pl.ANY),  # gate_up, HBM
            pl.BlockSpec(memory_space=pl.ANY),  # down, HBM
            pl.BlockSpec((1, hidden), lambda j: (0, 0)),  # gamma
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),  # DMA'd at end
        scratch_shapes=[
            pltpu.VMEM((tokens, hidden), x.dtype),  # normed
            pltpu.VMEM((tokens, two_i), x.dtype),  # gate|up
            pltpu.VMEM((tokens, inter), x.dtype),  # post-situ
            pltpu.VMEM((tokens, hidden), x.dtype),  # output
            pltpu.VMEM((depth1, k1, bn1), x.dtype),  # gate_up ring
            pltpu.VMEM((depth2, k2, bn2), x.dtype),  # down ring
            pltpu.SemaphoreType.DMA((depth1, )),
            pltpu.SemaphoreType.DMA((depth2, )),
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(**params) if params else None,
    )(x, w_gate_up, w_down, gamma)
