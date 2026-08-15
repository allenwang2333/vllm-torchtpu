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
"""Sort-free top-k expert selection for the MoE router (Pallas).

``torch.topk`` lowers to a full ``stablehlo.sort`` of every expert plus a
slice (``torch_tpu/ops/topk/topk.cc:55-86``). At 512 experts that is a
45-stage bitonic network moving three arrays (scores + a 64-bit index
payload) to keep 10 values. This kernel selects the same 10 with ``k``
max-and-mask passes over a VMEM-resident block, which is ~7x fewer issue
cycles at 8192x512 (see ``impl/w6-topk.md``).

Port of tpu-inference PR #3346 ``kernels/fused_moe/v2/router_ops.py``, with
the block size raised after a cycle sweep: at 512 experts the 256-row block
the original uses leaves the pipeline half empty.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Below any real routing score (softmax/sigmoid/sqrtsoftplus are all >= 0) and
# above -FLT_MAX, the identity of a max-reduce, so a masked column can never
# win a later pass.
NEG = -3.0e38

# Rows per grid step. 512 measured fastest at 512 experts at 8192 and 16384
# rows (119,811 / 224,805 issue cycles vs 172,093 / 331,951 for the 256 of
# tpu-inference PR #3346). The grid rounds up rather than requiring the block
# to divide the row count -- see ``select``.
MAX_BLOCK_ROWS = 512


def block_rows_for(rows: int, cap: int = MAX_BLOCK_ROWS) -> int:
    """Rows per grid step: the tuned block, or the whole input if smaller."""
    return min(cap, rows)


def _select_kernel(x_ref, w_ref, i_ref, *, topk: int, n: int):
    """One block of rows: top ``topk`` by repeated max-and-mask.

    NaN is flushed to the sentinel before the first max so every row has a
    maximum that some column equals, and every index written is a real
    expert. Ties resolve to the lowest expert id.
    """
    x = jnp.where(jnp.isnan(x_ref[...]), NEG, x_ref[...])
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        # Lowest column attaining the max; non-matching columns fill with the
        # last expert so the min never selects a phantom.
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        w_ref[:, j] = m[:, 0]
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(iota == idx[:, None], NEG, x)
    # A row that was entirely NaN has no real maximum, only the sentinel:
    # keep it poisonous rather than routing it to experts 0..k-1 with a
    # weight the renormalization would turn into something plausible.
    w_ref[...] = jnp.where(w_ref[:, :1] <= NEG, jnp.nan, w_ref[...])


def select(scores: jax.Array,
           topk: int,
           interpret: bool = False) -> tuple[jax.Array, jax.Array]:
    """Top-k over ``[rows, experts]`` f32 scores.

    Returns ``(weights f32, indices int32)`` of shape ``[rows, topk]``, values
    descending. ``interpret`` runs the kernel on the host, which is how the
    unit tests check it without a TPU.

    The grid rounds up, so ``rows`` need not be a multiple of the block: the
    last step reads a partial block and Mosaic discards the stores past
    ``rows``. Every reduction here is along the expert axis, so those surplus
    rows cannot perturb a real one. Requiring divisibility instead -- halving
    the block until it divided -- collapsed the tuned 512 to 32 on the shipped
    ``--max-num-batched-tokens=4384`` bucket, and to 1 on a prime one.
    """
    rows, n = scores.shape
    block = block_rows_for(rows)
    return pl.pallas_call(
        functools.partial(_select_kernel, topk=topk, n=n),
        grid=(pl.cdiv(rows, block), ),
        in_specs=[pl.BlockSpec((block, n), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
    )(scores)


def router_topk(scores: jax.Array, topk: int) -> tuple[jax.Array, jax.Array]:
    """``select`` with a jax_op-compatible signature (arrays, then statics)."""
    return select(scores, topk)
