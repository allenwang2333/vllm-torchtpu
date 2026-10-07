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
"""An activation (SwiGLU or the attention output gate) and the next fp8
linear's per-row quantization in one pass.

    silu_mul:      y = bf16(silu(g) * u)        for x = [g | u]
    sigmoid_gate:  y = bf16(a * sigmoid(gate))
    scale = max|y| / 448 (1 where that is 0); q = clip(y / scale) as fp8

``round_mid`` rounds the activation (silu(g), sigmoid(gate)) to bf16 before
the product, as eager bf16 torch does; without it the product sees the f32
value, as XLA computes it when it fuses the two (excess precision).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.rms_norm_quant.kernel import (
    FP8,
    FP8_MAX,
    VMEM_OUT_MAX_ROWS,
    _tile,
    lane_dense,
)


def _kernel(*refs, kind, round_mid):
    if kind == "silu_mul":
        x_ref, q_ref, s_ref = refs
        x = x_ref[...].astype(jnp.float32)
        d = x.shape[-1] // 2
        g, u = x[:, :d], x[:, d:]
        if round_mid == 2:  # bf16 sigmoid, bf16 silu
            sig = jax.nn.sigmoid(g).astype(jnp.bfloat16).astype(jnp.float32)
            act = g * sig
        elif round_mid == 3:  # silu as g / (1 + exp(-g))
            act = g / (1.0 + jnp.exp(-g))
        else:
            act = g * jax.nn.sigmoid(g)
        other = u
    else:
        a_ref, g_ref, q_ref, s_ref = refs
        act = jax.nn.sigmoid(g_ref[...].astype(jnp.float32))
        other = a_ref[...].astype(jnp.float32)
    if round_mid:  # 1, 2, 3: the activation is rounded to bf16 before the product
        act = act.astype(jnp.bfloat16).astype(jnp.float32)
    y = (act * other).astype(jnp.bfloat16)
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True).astype(jnp.float32)
    sc = amax / FP8_MAX
    sc = jnp.where(sc == 0.0, 1.0, sc)
    q_ref[...] = jnp.clip(y.astype(jnp.float32) / sc, -FP8_MAX, FP8_MAX).astype(FP8)
    s_ref[...] = sc.reshape(s_ref.shape)


@functools.partial(
    jax.jit, static_argnames=("kind", "round_mid", "tile", "interpret")
)
def act_quant(
    x: jax.Array,
    gate: jax.Array | None = None,
    *,
    kind: str = "silu_mul",
    round_mid: int = 0,
    tile: int = 256,
    interpret: bool = False,
):
    """silu_mul: x [m, 2d] -> (q [m, d], scale [m, 1]); sigmoid_gate: x (the
    attention output) and gate [m, n] -> (q [m, n], scale [m, 1])."""
    m, n2 = x.shape
    n = n2 // 2 if kind == "silu_mul" else n2
    t = _tile(m, tile)
    sds = pltpu.VMEM if (m <= VMEM_OUT_MAX_ROWS and not interpret) else jax.ShapeDtypeStruct
    in_specs = [pl.BlockSpec((t, n2), lambda i: (i, 0))]
    args = [x]
    if kind == "sigmoid_gate":
        in_specs.append(pl.BlockSpec((t, n), lambda i: (i, 0)))
        args.append(gate)
    dense = lane_dense(t, m)
    q, s = (
        pl.pallas_call(
            functools.partial(_kernel, kind=kind, round_mid=round_mid),
            grid=(m // t,),
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec((t, n), lambda i: (i, 0)),
                pl.BlockSpec((t,), lambda i: (i,))
                if dense
                else pl.BlockSpec((t, 1), lambda i: (i, 0)),
            ],
            out_shape=[sds((m, n), FP8), sds((m,) if dense else (m, 1), jnp.float32)],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            interpret=interpret,
        )(*args)
    )
    return q, s.reshape(m, 1)
