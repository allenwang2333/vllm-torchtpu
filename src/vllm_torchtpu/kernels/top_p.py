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
"""Top-p threshold from a candidate prefilter, exact against the bisection.

``layers/adapter/sample/top_k_top_p.topp_mask`` finds, per row, the largest
float32 ``t`` with ``mass(probs >= t) >= p`` by a 32-step bisection over the
float32 bit patterns: 33 full-vocabulary passes. Because ``mass(probs >= t)``
only changes at values of ``probs``, that ``t`` is always a value of ``probs``
(or the bisection's all-True sentinel). So:

1. Take ``K`` candidates per row (the largest probabilities) and sort them.
2. From their cumulative mass, tie groups included, guess the cutoff value
   ``t_lo`` and the next larger distinct value ``t_hi``.
3. Prove the guess with ONE full-vocabulary pass of three reductions, using the
   bisection's own predicate ``mass(probs >= t) < p``:
     * ``pred(t_lo)`` is False and ``pred(t_hi)`` is True, and
     * ``count(probs > t_lo)`` equals the number of candidates above ``t_lo``,
       i.e. every element above ``t_lo`` is a candidate, so no value of
       ``probs`` lies strictly between ``t_lo`` and ``t_hi``.
   The predicate is then False at ``t_lo`` and True on all of
   ``(t_lo, t_hi]`` (constant there), and monotonic, so ``t_lo`` IS the largest
   float32 at which it is False: the bisection's answer, bit for bit.
4. If any row with ``p < 1`` fails the proof (its nucleus is wider than ``K``,
   a candidate set missed an element, or the guess was one rounding off), a
   ``lax.cond`` runs the original bisection for the whole batch.

Rows with ``p >= 1`` are not proven: the caller discards their mask.

The predicate here must reduce exactly like the torch one in ``topp_mask``
(``torch.sum`` of ``where(probs >= t, probs, 0)`` in ``probs.dtype``, compared
in float32). ``mass_accum`` chooses how a bf16 sum is spelled: "native" reduces
in bf16 (stablehlo reduce in the input dtype), "f32" accumulates in f32 then
rounds once to bf16. Which one matches torch_tpu's lowering is decided by
``tests/layers/adapter/test_top_k_top_p.py`` on TPU; float32 inputs are
unaffected.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax

_EXPONENT_BITS = (1 << 31) - (1 << 23)
_NON_SIGN_BITS = (1 << 31) - 1


def _mass(probs: jax.Array, t: jax.Array, mass_accum: str) -> jax.Array:
    """``torch.sum(where(probs >= t, probs, 0), -1)`` in probs.dtype -> f32."""
    keep = probs.astype(jnp.float32) >= t[..., None]
    x = jnp.where(keep, probs, jnp.zeros((), probs.dtype))
    if probs.dtype == jnp.float32 or mass_accum == "native":
        s = lax.reduce(x, jnp.zeros((), x.dtype), lax.add, (x.ndim - 1,))
    else:
        s = jnp.sum(x.astype(jnp.float32), axis=-1).astype(probs.dtype)
    return s.astype(jnp.float32)


def _pred(probs, t, p, mass_accum):
    return _mass(probs, t, mass_accum) < p


def _monotonic_int32_to_float32(x: jax.Array) -> jax.Array:
    bits = x ^ jnp.where(x < 0, jnp.int32(_NON_SIGN_BITS), jnp.int32(0))
    return lax.bitcast_convert_type(bits, jnp.float32)


def bisect_threshold(probs: jax.Array, p: jax.Array, mass_accum: str) -> jax.Array:
    """``_float32_bsearch`` of ``topp_mask`` at radix 1: same 33 predicates."""

    def int32_predicate(x):
        bits = x ^ jnp.where(x < 0, jnp.int32(_NON_SIGN_BITS), jnp.int32(0))
        is_finite = (bits & _EXPONENT_BITS) != _EXPONENT_BITS
        f = lax.bitcast_convert_type(bits, jnp.float32)
        return jnp.where(is_finite, _pred(probs, f, p, mass_accum), bits >= 0)

    cur = jnp.zeros(probs.shape[:-1], jnp.int32)
    cur = cur | jnp.where(int32_predicate(cur), jnp.int32(-(1 << 31)), jnp.int32(0))
    for i in range(31):
        bit = jnp.int32(1 << (30 - i))
        cur = cur | jnp.where(int32_predicate(cur | bit), jnp.int32(0), bit)
    return _monotonic_int32_to_float32(cur)


def _candidates(pf: jax.Array, k: int, method: str) -> jax.Array:
    """Up to ``k`` probabilities per row, sorted descending (values only)."""
    if method == "topk":
        return lax.top_k(pf, k)[0]
    if method == "approx":
        return lax.approx_max_k(pf, k, recall_target=0.99)[0]
    if method == "chunk":
        # Max of each 128-lane chunk, keep the k // 128 best chunks whole,
        # top_k inside them. Holds every element above the best dropped chunk.
        rows, v = pf.shape
        lanes = 128
        nchunk = -(-v // lanes)
        pad = nchunk * lanes - v
        x = jnp.pad(pf, ((0, 0), (0, pad)), constant_values=-1.0) if pad else pf
        x = x.reshape(rows, nchunk, lanes)
        cmax = jnp.max(x, axis=-1)
        keep = max(1, min(nchunk, k // lanes))
        _, cidx = lax.top_k(cmax, keep)
        sub = jnp.take_along_axis(x, cidx[..., None], axis=1).reshape(rows, -1)
        return lax.top_k(sub, min(k, sub.shape[-1]))[0]
    raise ValueError(f"unknown candidate method {method!r}")


def fast_threshold(
    probs: jax.Array,
    p: jax.Array,
    k: int,
    method: str = "topk",
    mass_accum: str = "native",
    with_stats: bool = False,
):
    """Bisection-exact top-p threshold. probs [B, V] (bf16|f32), p [B] f32.

    Returns f32 [B] (and, with_stats, an int32 [B] per-row proof flag and the
    scalar "fell back" flag).
    """
    rows, v = probs.shape
    k = min(k, v)
    pf = probs.astype(jnp.float32)
    vals = _candidates(pf, k, method)
    k = vals.shape[-1]
    iota = lax.broadcasted_iota(jnp.int32, vals.shape, 1)

    # Tie groups of the sorted candidates: [first, last] index of each value.
    differs_next = jnp.concatenate(
        [vals[:, :-1] != vals[:, 1:], jnp.ones((rows, 1), bool)], axis=1
    )
    differs_prev = jnp.concatenate(
        [jnp.ones((rows, 1), bool), vals[:, 1:] != vals[:, :-1]], axis=1
    )
    last = lax.cummin(jnp.where(differs_next, iota, k - 1), axis=1, reverse=True)
    first = lax.cummax(jnp.where(differs_prev, iota, 0), axis=1)
    csum = jnp.cumsum(vals, axis=1)
    mass_ge = jnp.take_along_axis(csum, last, axis=1)
    # Round like the predicate's sum does before it is compared.
    mass_ge = mass_ge.astype(probs.dtype).astype(jnp.float32)

    # Guess: the first candidate whose tie-inclusive mass reaches p.
    j = jnp.sum((mass_ge < p[:, None]).astype(jnp.int32), axis=1)
    found = j < k
    j = jnp.minimum(j, k - 1)
    t_lo = jnp.take_along_axis(vals, j[:, None], axis=1)[:, 0]
    g0 = jnp.take_along_axis(first, j[:, None], axis=1)[:, 0]
    has_hi = g0 > 0
    t_hi = jnp.take_along_axis(vals, jnp.maximum(g0 - 1, 0)[:, None], axis=1)[:, 0]

    # Proof: one pass over the vocabulary, three row reductions.
    pred_lo = _pred(probs, t_lo, p, mass_accum)
    pred_hi = jnp.where(has_hi, _pred(probs, t_hi, p, mass_accum), True)
    above = jnp.sum((pf > t_lo[:, None]).astype(jnp.int32), axis=1)
    proven = found & ~pred_lo & pred_hi & (above == g0) & (t_lo > 0)
    ok = proven | (p >= 1.0)
    fell_back = ~jnp.all(ok)
    thr = lax.cond(
        fell_back,
        lambda: bisect_threshold(probs, p, mass_accum),
        lambda: t_lo,
    )
    if with_stats:
        return thr, proven.astype(jnp.int32), fell_back
    return thr


def top_p_threshold(
    probs: jax.Array,
    p: jax.Array,
    k: int,
    method: str,
    mass_accum: str,
) -> jax.Array:
    """``fast_threshold`` with a jax_op-compatible signature; p is [B, 1]."""
    return fast_threshold(probs, p.reshape(-1), k, method, mass_accum)[:, None]


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def fast_threshold_jit(probs, p, k, method="topk", mass_accum="native", stats=False):
    return fast_threshold(probs, p, k, method, mass_accum, stats)
