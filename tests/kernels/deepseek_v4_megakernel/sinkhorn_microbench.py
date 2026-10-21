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
"""Latency of one in-kernel mHC Sinkhorn (20 iterations, M=16) per variant.

python -m tests.kernels.deepseek_v4_megakernel.sinkhorn_microbench
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

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_ring

F32 = jnp.float32
REPS = 1000


def variant_lanes(x, div):
    lane = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    n = x.shape[1]

    def partner(v, k):
        return jnp.where((lane & k) == 0, pltpu.roll(v, n - k, 1), pltpu.roll(v, k, 1))

    def group(v, ks, op):
        for k in ks:
            v = op(v, partner(v, k))
        return v

    row, col = (1, 2), (4, 8)
    ex = jnp.exp(x - group(x, row, jnp.maximum))
    c = div(ex, group(ex, row, jnp.add)) + 1e-6
    c = div(c, group(c, col, jnp.add) + 1e-6)
    for _ in range(19):
        c = div(c, group(c, row, jnp.add) + 1e-6)
        c = div(c, group(c, col, jnp.add) + 1e-6)
    return c


def variant_cols(x, recip):
    """16 separate [1, 128] arrays (tokens on lanes): pure elementwise."""
    c = [x[k : k + 1, :] for k in range(16)]
    for i in range(4):
        row = c[4 * i : 4 * i + 4]
        mx = jnp.maximum(jnp.maximum(row[0], row[1]), jnp.maximum(row[2], row[3]))
        ex = [jnp.exp(v - mx) for v in row]
        s_ = (ex[0] + ex[1]) + (ex[2] + ex[3])
        c[4 * i : 4 * i + 4] = [v / s_ + 1e-6 for v in ex]

    def norm(c, groups):
        out = list(c)
        for g in groups:
            s_ = (c[g[0]] + c[g[1]]) + (c[g[2]] + c[g[3]]) + 1e-6
            if recip:
                r = 1.0 / s_
                for k in g:
                    out[k] = c[k] * r
            else:
                for k in g:
                    out[k] = c[k] / s_
        return out

    rows = [[4 * i + j for j in range(4)] for i in range(4)]
    cols = [[4 * i + j for i in range(4)] for j in range(4)]
    c = norm(c, cols)
    for _ in range(19):
        c = norm(norm(c, rows), cols)
    return jnp.concatenate(c, axis=0)


def variant_sub(x):
    """[16, 128]: entries on sublanes, tokens on lanes; sublane-roll groups."""
    sub = lax.broadcasted_iota(jnp.int32, x.shape, 0)

    def group(v, ks, op):
        for k in ks:
            p = jnp.where((sub & k) == 0, pltpu.roll(v, 16 - k, 0), pltpu.roll(v, k, 0))
            v = op(v, p)
        return v

    row, col = (1, 2), (4, 8)
    ex = jnp.exp(x - group(x, row, jnp.maximum))
    c = ex / group(ex, row, jnp.add) + 1e-6
    c = c / (group(c, col, jnp.add) + 1e-6)
    for _ in range(19):
        c = c / (group(c, row, jnp.add) + 1e-6)
        c = c / (group(c, col, jnp.add) + 1e-6)
    return c


VARIANTS = {
    "cols16_div": lambda x: variant_cols(x, False),
    "cols16_recip": lambda x: variant_cols(x, True),
    "sub16": variant_sub,
    "lanes_div": lambda x: variant_lanes(x, lambda a, b: a / b),
    "lanes_recip": lambda x: variant_lanes(
        x, lambda a, b: a * pl.reciprocal(b, approx=False)
    ),
    "lanes_recip_approx": lambda x: variant_lanes(
        x, lambda a, b: a * pl.reciprocal(b, approx=True)
    ),
    "kernel_fn": lambda x: dense_ring.sinkhorn_lanes(x, 4, 1e-6, 20, pltpu.roll),
}


def kern(x_ref, o_ref, *, fn):
    def body(i, c):
        return fn(c) + 1e-3

    o_ref[...] = lax.fori_loop(0, REPS, body, x_ref[...])


def main():
    x = jax.random.normal(jax.random.key(0), (16, 128), F32)
    for name, fn in VARIANTS.items():
        f = jax.jit(pl.pallas_call(
            functools.partial(kern, fn=fn),
            out_shape=jax.ShapeDtypeStruct(x.shape, F32),
        ))  # fmt: skip
        jax.block_until_ready(f(x))
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(f(x))
            ts.append(time.perf_counter() - t0)
        us = float(np.median(ts)) * 1e6 / REPS
        print("RESULT " + json.dumps({"variant": name, "us_per_sinkhorn": us}),
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
