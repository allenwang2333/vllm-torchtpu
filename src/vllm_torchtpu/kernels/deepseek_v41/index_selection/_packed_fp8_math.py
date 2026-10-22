# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head layout and BF16 weighting / FP32 accumulation for packed FP8 MQA.

These helpers run while tracing; Python loops must stay statically unrolled.
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def head_order(x):
    """Align 4-query/two-head and 2-query/four-head partials to eight rows."""
    width = x.shape[-1]
    x = x.reshape(6, 32, width)
    first = x[:4].reshape(4, 2, 8, 2, width).transpose(2, 3, 0, 1, 4)
    last = x[4:].reshape(2, 2, 4, 4, width).transpose(2, 3, 0, 1, 4)
    return jnp.concatenate((first.reshape(128, width), last.reshape(64, width)))


def reduce_heads(
    products: jax.Array, selector: jax.Array, fragment: int, fragments: int
) -> jax.Array:
    """Sum finite BF16 products in FP32, round to BF16, return FP32."""
    width = products.shape[-1]
    if fragment % 2 and fragment < fragments - 8:
        # Finite-product contract: 0 * NaN/Inf in this dense
        # selector can contaminate other queries. This is not a general
        # nonfinite-preserving substitute for independent head sums.
        # Products were already rounded to BF16. The selector dot
        # accumulates these products in FP32, then rounds the result.
        values = jax.lax.dot_general(
            selector,
            products,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    else:
        partial_groups = []
        for start, stop, groups, heads_per_row in (
            (0, 128, 8, 2),
            (128, 192, 4, 4),
        ):
            packed = pltpu.bitcast(products[start:stop], jnp.uint32).reshape(
                groups, 8, width
            )
            partial = None
            for g in range(groups):
                for half in range(2):
                    value = pltpu.unpack_elementwise(
                        packed[g],
                        index=half,
                        packed_dtype=jnp.bfloat16,
                        unpacked_dtype=jnp.float32,
                    )
                    partial = value if partial is None else partial + value
            # Queries repeat every four/two sublanes. Cyclic rotations
            # sum all head groups and leave each query in its output row.
            partial = partial + pltpu.roll(partial, 4, axis=0)
            if heads_per_row == 4:
                partial = partial + pltpu.roll(partial, 2, axis=0)
            partial_groups.append(partial)
        rows = jnp.arange(8, dtype=jnp.int32)[:, None]
        values = jnp.where(rows < 4, partial_groups[0], partial_groups[1])
        values = values[:6]
    # Round the completed FP32 sum to BF16, then widen exactly to FP32.
    # Interleaved packing permits the unused half to become zero; the
    # compiler can then use the rounded bits directly as FP32 operands.
    rounded = pltpu.pack_elementwise((values, values), packed_dtype=jnp.bfloat16)
    return pltpu.unpack_elementwise(
        rounded,
        index=0,
        packed_dtype=jnp.bfloat16,
        unpacked_dtype=jnp.float32,
    )
