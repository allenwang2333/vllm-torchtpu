"""Expert-loop microbench: the megakernel's EP expert pipeline in isolation (one TensorCore).

NE experts x (8 e13 [512, 2048] + 4 e2 [256, 4096]) fp8 1 MiB tiles stream from HBM through an
NBE-slot VMEM ring (refill slot u with unit u + NBE right after unit u is consumed, like
ExpertWeightRing). Per expert: acc13 (+)= x[:, k-chunk] @ w13 (8 tiles), silu*up, acc (+)=
mid[:, k] @ w2 (4 tiles), part += wcol * rnd(acc * s). Modes:
  dma   : waits + refills only (no compute): ring/HBM-bound rate
  full  : the computation on all M rows (current megakernel)
  piped : PIPE_EP order (w2 tile of expert i beside 2 w13 tiles of expert i+1)
Reports us per expert (slope over two expert counts, launch overhead removed).
"""
import json
import time

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32, BF16, FP8 = jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn
NN = (((1,), (0,)), ((), ()))
H, IR, T13, T2, TK13, TK2 = 4096, 1024, 8, 4, 512, 256
TE = T13 + T2
NEXP_HBM = 64
CW = 512


def rnd(x):
    return x.astype(BF16).astype(F32)


def make(M, NE, NBE, mode, xdt=BF16):
    def kernel(x_ref, s_ref, e13_hbm, e2_hbm, o_ref, ring13, ring2, sem, acc13, acc, part, xb2):
        # unit u: expert u // TE (mod NEXP_HBM), tile u % TE; slot u % NBE.
        def start(u):
            e = lax.rem(u // TE, NEXP_HBM)
            t = lax.rem(u, TE)
            b = lax.rem(u, NBE)

            @pl.when(t < T13)
            def _():
                pltpu.make_async_copy(e13_hbm.at[e, t], ring13.at[b], sem.at[b]).start()

            @pl.when(t >= T13)
            def _():
                pltpu.make_async_copy(e2_hbm.at[e, t - T13], ring2.at[b], sem.at[b]).start()

        def wait(kind, b):
            if kind == 13:
                pltpu.make_async_copy(e13_hbm.at[0, 0], ring13.at[b], sem.at[b]).wait()
            else:
                pltpu.make_async_copy(e2_hbm.at[0, 0], ring2.at[b], sem.at[b]).wait()

        def give(u):
            @pl.when(u + NBE < NE * TE)
            def _():
                start(u + NBE)

        for k in range(NBE):
            start(jnp.int32(k))
        part[...] = jnp.zeros(part.shape, F32)

        def body_reg(i, c):
            # register / MRB sums: all T13 tiles waited, then per 512-lane column chunk
            # p = sum_t x_t @ w_t[:, chunk] stored once (dmatmul_pair style); same for e2.
            base = i * TE
            bs = [lax.rem(base + t, NBE) for t in range(TE)]
            for t in range(T13):
                wait(13, bs[t])
            for c0 in range(0, 2 * IR, CW):
                p = None
                for t in range(T13):
                    q = lax.dot_general(x_ref[:, t * TK13:(t + 1) * TK13],
                                        ring13[bs[t], :, c0:c0 + CW], NN,
                                        preferred_element_type=F32)
                    p = q if p is None else p + q
                acc13[:, c0:c0 + CW] = p
            for t in range(T13):
                give(base + t)
            h = acc13[...] * s_ref[:, :2 * IR]
            xb2[...] = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)
            for t in range(T2):
                wait(2, bs[T13 + t])
            for c0 in range(0, H, CW):
                p = None
                for t in range(T2):
                    q = lax.dot_general(xb2[:, t * TK2:(t + 1) * TK2],
                                        ring2[bs[T13 + t], :, c0:c0 + CW], NN,
                                        preferred_element_type=F32)
                    p = q if p is None else p + q
                part[:, c0:c0 + CW] += s_ref[:, 2 * IR:2 * IR + 1] * rnd(p * s_ref[:, 2 * IR + c0:2 * IR + c0 + CW])
            for t in range(T2):
                give(base + T13 + t)
            return c

        def body(i, c):
            base = i * TE
            for t in range(T13):
                u = base + t
                b = lax.rem(u, NBE)
                wait(13, b)
                if mode != "dma":
                    p = lax.dot_general(x_ref[:, t * TK13:(t + 1) * TK13], ring13[b], NN,
                                        preferred_element_type=F32)
                    if t == 0:
                        acc13[...] = p
                    else:
                        acc13[...] += p
                give(u)
            if mode != "dma":
                h = acc13[...] * s_ref[:, :2 * IR]
                xb2[...] = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)
            for t in range(T2):
                u = base + T13 + t
                b = lax.rem(u, NBE)
                wait(2, b)
                if mode != "dma":
                    p = lax.dot_general(xb2[:, t * TK2:(t + 1) * TK2], ring2[b], NN,
                                        preferred_element_type=F32)
                    if t == 0:
                        acc[...] = p
                    else:
                        acc[...] += p
                give(u)
            if mode != "dma":
                part[...] += s_ref[:, 2 * IR:2 * IR + 1] * rnd(acc[...] * s_ref[:, 2 * IR:])
            return c

        lax.fori_loop(0, NE, body_reg if mode == 'reg' else body, 0)
        o_ref[...] = part[0:8, 0:128]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), F32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.VMEM),
                  pl.BlockSpec(memory_space=pltpu.HBM), pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.VMEM((NBE, TK13, 2 * IR), FP8), pltpu.VMEM((NBE, TK2, H), FP8),
                        pltpu.SemaphoreType.DMA((NBE,)),
                        pltpu.VMEM((M, 2 * IR), F32), pltpu.VMEM((M, H), F32),
                        pltpu.VMEM((M, H), F32), pltpu.VMEM((M, IR), BF16)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=62 * 1024 * 1024),
    )


import sys
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

NDEV = int(sys.argv[2]) if len(sys.argv) > 2 else len(jax.devices())
MESH = Mesh(np.asarray(jax.devices()[:NDEV]), ("d",))
SH = NamedSharding(MESH, P("d"))


def gput(shape, dtype, key, scale=0.5):
    g = (NDEV * shape[0],) + tuple(shape[1:])
    return jax.jit(lambda k: (jax.random.normal(k, g, F32) * scale).astype(dtype), out_shardings=SH)(key)


E13 = gput((NEXP_HBM, T13, TK13, 2 * IR), FP8, jax.random.PRNGKey(1))
E2 = gput((NEXP_HBM, T2, TK2, H), FP8, jax.random.PRNGKey(2))


def run(M, NE, NBE, mode):
    x = gput((M, H), BF16, jax.random.PRNGKey(0))
    s = jax.device_put(jnp.full((NDEV * M, 2 * IR + H), 1.0 / 64, F32), SH)
    call = make(M, NE, NBE, mode)
    f = jax.jit(jax.shard_map(call, mesh=MESH, in_specs=(P("d"),) * 4, out_specs=P("d"),
                              check_vma=False))
    jax.block_until_ready(f(x, s, E13, E2))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(f(x, s, E13, E2))
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e6


MODES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["dma", "full"]
for NBE in (12, 24):
    for mode in MODES:
        for M in ((8,) if mode == "dma" else (16, 48, 80)):
            r = {"NDEV": NDEV, "NBE": NBE, "mode": mode, "M": M}
            try:
                a, b = run(M, 64, NBE, mode), run(M, 320, NBE, mode)
                r["us_per_expert"] = round((b - a) / 256, 3)
                r["eff_TBps_per_core"] = round(12 * 2**20 / ((b - a) / 256) / 1e6, 2)
            except Exception as e:  # noqa
                r["error"] = repr(e)[:300]
            print("RESULT " + json.dumps(r), flush=True)
