# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Weight-streaming throughput of a VMEM tile ring per consumer op (TPU).

    python -m tests.kernels.deepseek_v4_megakernel.ring_microbench

Streams ~2 GB of tiles through an NB-bank ring and times each consumer:
no compute, an M=16 dot per tile in several dtypes / orientations. Prints
RESULT lines with GB/s (the dense ring's attainable rate per weight kind).
"""

import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32


def ring_kernel(w_hbm, x_ref, o_ref, ring, sem, *, nb, nt, op, cw, splits, nodma):
    rs = ring.shape[1] // splits

    class _Copy:  # one tile as `splits` row-slab DMAs on the bank's semaphore
        def __init__(self, t, b):
            self.cps = [
                pltpu.make_async_copy(
                    w_hbm.at[t, pl.ds(i * rs, rs)],
                    ring.at[b, pl.ds(i * rs, rs)],
                    sem.at[b],
                )
                for i in range(splits)
            ]

        def start(self):
            for c in self.cps if not nodma else ():
                c.start()

        def wait(self):
            for c in self.cps if not nodma else ():
                c.wait()

    def copy(t, b):
        return _Copy(t, b)

    for b in range(nb):
        copy(b, b).start()
    tr, tc = ring.shape[1], ring.shape[2]
    o_ref[...] = jnp.zeros(o_ref.shape, F32)

    def body(t, c):
        b = lax.rem(t, nb)
        copy(0, b).wait()
        w = ring.at[b]
        if op == "nn":
            x = x_ref[:, :tr]
            for c0 in range(0, tc, cw):
                o_ref[:, c0 : c0 + cw] += lax.dot_general(
                    x, w[:, c0 : c0 + cw], (((1,), (0,)), ((), ())),
                    preferred_element_type=F32)  # fmt: skip
        elif op == "nn_bf16cast":
            x = x_ref[:, :tr].astype(jnp.bfloat16)
            for c0 in range(0, tc, cw):
                o_ref[:, c0 : c0 + cw] += lax.dot_general(
                    x, w[:, c0 : c0 + cw].astype(jnp.bfloat16),
                    (((1,), (0,)), ((), ())), preferred_element_type=F32)  # fmt: skip
        elif op == "nt":
            x = x_ref[:, :tc]
            o_ref[:, :tr] += lax.dot_general(
                x, w[...], (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
        elif op == "touch":
            o_ref[:, :128] += w[:16, :128].astype(F32)

        @pl.when(t + nb < nt)
        def _():
            copy(t + nb, b).start()

        return c

    lax.fori_loop(0, nt, body, 0)


def bench(
    dtype,
    tr,
    tc,
    op,
    *,
    nb=12,
    total=4 * 2**30,
    m=16,
    cw=1024,
    xdt=None,
    splits=1,
    nodma=False,
):
    itemsize = jnp.dtype(dtype).itemsize
    nt = total // (tr * tc * itemsize)
    w = jnp.zeros((nt, tr, tc), dtype)
    xdt = xdt or dtype
    k = max(tr, tc)
    x = jnp.ones((m, k), xdt)
    f = jax.jit(pl.pallas_call(
        functools.partial(ring_kernel, nb=nb, nt=nt, op=op, cw=min(cw, tc),
                          splits=splits, nodma=nodma),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY),
                  pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((m, k), F32),
        scratch_shapes=[pltpu.VMEM((nb, tr, tc), dtype),
                        pltpu.SemaphoreType.DMA((nb,))],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 2**20),
    ))  # fmt: skip
    for _ in range(2):
        jax.block_until_ready(f(w, x))
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        jax.block_until_ready(f(w, x))
        ts.append(time.perf_counter() - t0)
    t = float(np.median(ts))
    nbytes = nt * tr * tc * itemsize
    r = {"dtype": jnp.dtype(dtype).name, "tile": [tr, tc], "op": op, "nb": nb,
         "m": m, "cw": cw, "splits": splits, "nodma": nodma, "GBps": nbytes / t / 1e9,
         "ms": t * 1e3}  # fmt: skip
    print("RESULT " + json.dumps(r), flush=True)
    del w


def xla_reduce_bandwidth(total=4 * 2**30):
    """XLA's own streaming rate on the same device: max over a u32 array."""
    w = jnp.zeros((total // 4 // 4096, 4096), jnp.uint32)
    f = jax.jit(lambda w: jnp.max(w, axis=0))
    for _ in range(2):
        jax.block_until_ready(f(w))
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        jax.block_until_ready(f(w))
        ts.append(time.perf_counter() - t0)
    t = float(np.median(ts))
    print("RESULT " + json.dumps({"op": "xla_max_u32", "GBps": total / t / 1e9,
          "ms": t * 1e3}), flush=True)  # fmt: skip


def main():
    xla_reduce_bandwidth()
    fp8, bf16 = jnp.float8_e4m3fn, jnp.bfloat16
    nd = {"nodma": True}  # consumer throughput alone (stale banks)
    cases = [
        (fp8, 1024, 2048, "nn", nd),
        (fp8, 1024, 2048, "nn", {**nd, "cw": 2048}),
        (fp8, 1024, 2048, "nn", {**nd, "cw": 512}),
        (fp8, 512, 4096, "nn", nd),
        (fp8, 256, 7168, "nn", nd),
        (fp8, 3584, 384, "nn", nd),
        (bf16, 256, 2688 + 128, "nn", nd),
        (bf16, 1024, 1024, "nn", nd),
        (bf16, 1792, 384, "nn", nd),
        (fp8, 1024, 2048, "nn_bf16cast", nd),
        (bf16, 80, 7168, "nt", nd),
        (fp8, 1024, 2048, "touch", nd),
    ]
    cases += [  # with the weight DMA
        (fp8, 1024, 2048, "touch", {}),
        (fp8, 1024, 2048, "touch", {"splits": 2}),
        (fp8, 1024, 2048, "touch", {"splits": 4}),
        (fp8, 1024, 2048, "touch", {"splits": 8}),
        (fp8, 2048, 2048, "touch", {"nb": 6}),
        (fp8, 2048, 2048, "touch", {"nb": 6, "splits": 4}),
        (fp8, 512, 2048, "touch", {"nb": 24}),
        (fp8, 1024, 2048, "touch", {"nb": 4}),
        (fp8, 1024, 2048, "nn", {"splits": 4}),
        (fp8, 1024, 2048, "nn", {}),
        (bf16, 1024, 1024, "touch", {}),
        (bf16, 1024, 1024, "nn", {}),
        (bf16, 256, 2688 + 128, "nn", {}),
        (bf16, 80, 7168, "nt", {}),
        (fp8, 1024, 2048, "nn", {"nb": 6}),
        (fp8, 1024, 2048, "nn", {"nb": 20}),
    ]
    for dt, tr, tc, op, kw in cases:
        bench(dt, tr, tc, op, **kw)


if __name__ == "__main__":
    main()
