# SPDX-License-Identifier: Apache-2.0
"""MoE router logits at decode, router weight streamed from HBM (TPU_MOE_ROUTER_HBM_PIN).

The router GEMM is `x [m, k] bf16 @ w [e, k]^T bf16 -> [m, e] bf16` (vLLM's GateLinear
on its F.linear tier; GLM-5.2: k = 6144, e = 256, 3 MB per rank, replicated). As an
XLA dot its weight is a VMEM operand, so XLA's memory-space assignment stages the
whole weight into VMEM with a copy before the dot; at decode that copy is only hidden
when enough runs between its start and the dot. This kernel takes the weight as an
HBM operand -- constrained to HBM with `pltpu.with_memory_space_constraint`, which is
what stops the staging (the in_spec's memory space alone only types the kernel ref) --
and walks it through a small VMEM ring of contiguous row tiles `w[j*t:(j+1)*t, :]`,
one grid step, computing each tile as it lands (the fp8_skinny ring).

Numerics: each tile is one `dot_general(x, w_tile)` contracting k on both sides with
f32 accumulation, cast once to bf16. This is NOT bit-identical to the XLA dot: Mosaic's
and XLA's MXU accumulation orders differ, so the f32 sums differ in the last bits and
about 1 in 10^4 logits (GLM-5.2 shapes, random data) rounds to the neighbouring bf16
(measured, j-4a606e4c; every Mosaic k-chunking tried gives the same bits as this one).
A routing decision can flip only where a logit sits within one bf16 ulp of the top-k
boundary, so the flag needs an accuracy check before it ships.

Tiling: one tile (the whole weight in one DMA, one dot) is the fastest of row tiles
32..256 x ring depth 2..4 at e = 256, k = 6144 (probe j-4a606e4c: 3.7 vs 4.3-7.1 us per
call in a serial chain); a K-major [k, e] copy would be ~0.5 us faster still but costs a
second copy of the weight.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_M = 16
LANE = 128


def pick_config(e: int, k: int, itemsize: int = 2) -> tuple[int, int]:
    """(tile rows, ring depth): the whole weight as one tile up to 4 MiB, else row
    tiles of at most 4 MiB in a 2-deep ring."""
    if e * k * itemsize <= 4 << 20:
        return e, 1
    for t in (256, 128, 64, 32, 16, 8):
        if e % t == 0 and t * k * itemsize <= 4 << 20:
            return t, 2
    return e, 1


def supports(m: int, x: jax.Array, w: jax.Array) -> bool:
    """Decode token counts, bf16 operands, k a lane multiple, e a sublane multiple."""
    return (
        m <= MAX_M
        and x.ndim == 2
        and w.ndim == 2
        and x.dtype == jnp.bfloat16
        and w.dtype == jnp.bfloat16
        and x.shape[1] == w.shape[1]
        and w.shape[1] % LANE == 0
        and w.shape[0] % 8 == 0
    )


def _kernel(x_ref, w_hbm, o_ref, wbuf, sem, *, t, n_tiles, nbuf, kc, nt):
    def copy(j):
        return pltpu.make_async_copy(
            w_hbm.at[pl.ds(j * t, t), :], wbuf.at[j % nbuf], sem.at[j % nbuf]
        )

    for j in range(min(nbuf, n_tiles)):
        copy(j).start()
    k = x_ref.shape[1]
    kc = kc or k
    dn = (((1,), (1,)), ((), ()))
    for j in range(n_tiles):
        copy(j).wait()
        y = None
        for c in range(k // kc):
            ks = slice(c * kc, (c + 1) * kc)
            if nt:
                # [t, kc] . [m, kc]^T -> [t, m], transposed back after the sum
                p = jax.lax.dot_general(
                    wbuf[j % nbuf, :, ks], x_ref[:, ks], dn,
                    preferred_element_type=jnp.float32,
                )
            else:
                p = jax.lax.dot_general(
                    x_ref[:, ks], wbuf[j % nbuf, :, ks], dn,
                    preferred_element_type=jnp.float32,
                )
            y = p if y is None else y + p
        if nt:
            y = y.T
        o_ref[:, j * t : (j + 1) * t] = y.astype(o_ref.dtype)
        if j + nbuf < n_tiles:
            copy(j + nbuf).start()


@functools.cache
def _build(m, k, e, dtype, t, nbuf, kc=None, nt=False):
    if e % t:
        raise ValueError(f"tile {t} does not divide {e} router rows")
    n_tiles = e // t
    nbuf = max(1, min(nbuf, n_tiles))
    kern = functools.partial(_kernel, t=t, n_tiles=n_tiles, nbuf=nbuf, kc=kc, nt=nt)
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, e), dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m, k), lambda g: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((m, e), lambda g: (0, 0)),
        scratch_shapes=[
            pltpu.VMEM((nbuf, t, k), dtype),
            pltpu.SemaphoreType.DMA((nbuf,)),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"router_gemm_hbm_t{t}_b{nbuf}{f'_kc{kc}' if kc else ''}{'_nt' if nt else ''}"
        f"_m{m}_k{k}_e{e}",
    )


@functools.partial(jax.jit, static_argnums=0)
def _call_hbm_pinned(call, x, w):
    # The constraint sets the operand aval's memory space, which pallas_call lowers
    # to the custom call's `input_memory_space_colors` (HBM = 0). It only binds a
    # pallas_call input and cannot run eagerly, hence the jit around both.
    return call(x, pltpu.with_memory_space_constraint(w, pltpu.HBM))


def router_gemm_hbm(
    x: jax.Array,
    w: jax.Array,
    *,
    config: tuple[int, int] | None = None,
    kc: int | None = None,
    nt: bool = False,
) -> jax.Array:
    """x [m <= 16, k] bf16, w [e, k] bf16 -> x @ w.T [m, e] bf16, weight read from HBM.
    `kc`: sum the dot over k chunks of this size in f32 (None: one dot over k);
    `nt`: compute w_tile . x^T and transpose (the MXU holds x, not the weight)."""
    m, k = x.shape
    e = w.shape[0]
    t, nbuf = config or pick_config(e, k)
    return _call_hbm_pinned(_build(m, k, e, x.dtype, t, nbuf, kc, nt), x, w)


def router_logits(x: jax.Array, w: jax.Array) -> jax.Array:
    """`F.linear(x, w)` for a bias-free bf16 router: the HBM-streaming kernel at decode
    token counts, the plain dot otherwise (prefill sizes: unchanged arithmetic)."""
    if supports(x.shape[0], x, w):
        return router_gemm_hbm(x, w)
    return jax.lax.dot_general(x, w, (((1,), (1,)), ((), ())))
