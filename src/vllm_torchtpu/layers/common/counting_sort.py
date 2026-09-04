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
"""Sort-free dispatch permutation for the MoE routed path.

``prepare_routed_gmm_inputs`` orders ``tokens * topk`` routes by expert id with
two ``jnp.argsort`` calls. The keys are expert ids, so the alphabet is
``local_num_experts + 1`` symbols -- 29 at Kimi-K3 EP32 -- while the sort XLA
emits is a bitonic network that knows nothing about that and pays
``O(N log^2 N)`` cross-lane compare-exchanges to find out.

Cost of what it replaces, measured rather than assumed (Kimi-K3 TP32+EP32
decode, BK #367, pad 8, real weights): the two sorts are ``s32[128]``, 2.58 us
each, **474.8 us/step = 2.04% of the TensorCore plane**. They are latency-bound,
not work-bound -- 345 issue slots against a **4059-cycle** dependency path,
because each of the 28 bitonic stages is a cross-lane ``vperm.xlu`` whose result
is 131 cycles from its operand.

A counting sort needs the one-hot of the keys, which ``prepare_routed_gmm_inputs``
already builds for ``group_sizes``, so the incremental work is two prefix sums
over a matrix that has to exist anyway.
"""

import jax
import jax.numpy as jnp


def key_onehot(keys: jax.Array, num_symbols: int) -> jax.Array:
    """``[n, num_symbols]`` int32 indicator of ``keys``.

    Kept separate because the caller needs it twice: the counting sort ranks
    with it and ``group_sizes`` is a column sum of it.
    """
    return (keys[:, None] == jnp.arange(num_symbols,
                                        dtype=keys.dtype)[None, :]
            ).astype(jnp.int32)


def sort_positions(onehot: jax.Array) -> jax.Array:
    """Destination of every route under a **stable** sort of its key.

    ``positions[i]`` is where route ``i`` lands, i.e. exactly
    ``jnp.argsort(jnp.argsort(keys))`` -- the array
    ``prepare_routed_gmm_inputs`` calls ``argsort_revert_indices``. Ties keep
    their original order, matching ``jnp.argsort``'s ``stable=True`` default, so
    the two agree elementwise and not merely up to a permutation of equal keys.
    """
    counts = onehot.sum(axis=0)
    bucket_start = jnp.cumsum(counts) - counts               # exclusive
    rank_in_bucket = jnp.cumsum(onehot, axis=0) - onehot     # exclusive
    # Row i is nonzero only in column keys[i], so the row sum selects that
    # entry; no gather, and no dependence on the key values themselves.
    return (onehot * (bucket_start[None, :] + rank_in_bucket)
            ).sum(axis=1).astype(jnp.int32)


def invert_permutation(positions: jax.Array) -> jax.Array:
    """``sorted_indices`` from ``positions``: which route landed where.

    ``O(n^2)`` in select-and-reduce rather than a scatter, which is why callers
    must keep ``n`` small (see ``MAX_ROUTES``). At the decode point n = 128 and
    the whole thing is 16 K int32 in VMEM.
    """
    n = positions.shape[0]
    slots = jnp.arange(n, dtype=jnp.int32)
    hit = positions[:, None] == slots[None, :]
    return jnp.where(hit, slots[:, None], 0).sum(axis=0).astype(jnp.int32)


# Above this route count the O(n^2) inverse stops being free and the argsort
# path is kept. Decode is 16 (pad 1) to 128 (pad 8) routes; prefill at
# mnbt 8192 is 131072 and must not take this path.
MAX_ROUTES = 512


def applicable(n_routes: int) -> bool:
    return 0 < n_routes <= MAX_ROUTES
