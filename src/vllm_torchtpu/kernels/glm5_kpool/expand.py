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
"""Pool ids -> raw token ids, prefix-contiguous by construction.

``streamindex_topk`` selects *pools*; ``sparse_mla_attention`` consumes *raw
token ids*. This is the whole of the conversion, in pure ``jnp``.
"""

from __future__ import annotations

import jax.numpy as jnp


def align_to(x: int, a: int) -> int:
    """``streamindex_topk.py:35-36``, repeated so this module has no import
    dependency on the DSv4 kernel."""
    return -(-x // a) * a


def derive_out_width(index_topk: int,
                     index_kpool: int,
                     alignment: int = 128) -> int:
    """The emitted row width. **Derived, not chosen.**

    ``align_to(index_topk + index_kpool - 1, 128)`` -- 2176 at GLM-5.3-Flash's
    ``index_topk=2048``, ``index_kpool=4``.

    Two real reasons, neither of which is the one the explainer gave:

    * **128 alignment.** ``bkv_sz = cache_kv_nope_hbm_ref.shape[1] //
      TILE_SUBROWS = topk`` (``kernels/mla/sparse/kernel.py:168``) becomes the
      minor axis of ``s`` in the einsum at ``:200``. A width that is not a
      multiple of 128 makes Mosaic pad, and the extra VMEM gets attributed to
      kpool. (``2052 % 128 == 4``; the "keeps the score tile aligned"
      justification for 2052 was simply false.)
    * **``index_kpool`` divisibility**, so the expansion of a whole pool never
      straddles the end of the buffer.

    And one prohibition, which is the load-bearing half: **do not trim to
    2048.** The columns past ``index_topk`` are exactly the always-selected
    tail -- the ``(p + 1) % index_kpool`` most recent tokens. Trimming drops
    them, which is invisible at ``L % index_kpool == 0`` and wrong at the other
    three residues. ``expand_pools_with_tail`` asserts the width statically for
    that reason.
    """
    width = align_to(index_topk + index_kpool - 1, alignment)
    _assert_out_width(width, index_topk // index_kpool, index_kpool,
                      index_topk)
    return width


def _assert_out_width(out_width: int,
                      select_k: int,
                      index_kpool: int,
                      index_topk: int | None = None) -> None:
    """``index_topk`` is the *budget*; ``select_k`` is what the caller passed.

    Checking only ``select_k`` makes the check relax along with a mistake:
    hand ``streamindex_topk`` a ``k`` smaller than ``index_topk //
    index_kpool`` -- a plausible copy from a smaller fixture -- and the minimum
    width shrinks to match, so a row that is short for the configured budget
    passes. Pin the budget too when the caller knows it.

    **``ValueError``, not ``assert``.** Every argument here is a plain Python
    int known at construction/trace-build time, so this is a configuration
    guard, and ``python -O`` deletes ``assert``. A deleted guard here does not
    crash: it emits a row a few columns short, which drops the
    always-selected tail -- invisible at ``L % index_kpool == 0`` and wrong at
    the other three residues. (Contrast the *traced-array shape* asserts
    inside ``kpool_write``: those are cheap trace-time invariants over jnp
    shapes, and they stay asserts.)
    """
    if index_topk is not None and select_k * index_kpool != index_topk:
        raise ValueError(
            f"select_k={select_k} x index_kpool={index_kpool} = "
            f"{select_k * index_kpool} != index_topk={index_topk}; the width "
            "check below would relax to match the narrower budget")
    min_width = select_k * index_kpool + index_kpool - 1
    if out_width < min_width:
        raise ValueError(
            f"out_width={out_width} clips the always-selected tail: "
            f"select_k={select_k} pools expand to {select_k * index_kpool} "
            f"ids and the tail adds up to {index_kpool - 1} more, so the row "
            f"needs {min_width} columns. Trimming to index_topk drops the "
            "tail, which is invisible at L % index_kpool == 0 and wrong at "
            "the other residues.")
    if out_width % index_kpool != 0:
        raise ValueError(
            f"out_width={out_width} is not a multiple of index_kpool="
            f"{index_kpool}")


def expand_pools_with_tail(
    pool_ids: jnp.ndarray,
    positions: jnp.ndarray,
    valid_token: jnp.ndarray,
    *,
    index_kpool: int,
    out_width: int,
) -> jnp.ndarray:
    """``i32[T, select_k]`` pool ids -> ``i32[T, out_width]`` raw token ids.

    Args:
        pool_ids: what ``streamindex_topk`` returns -- selected pool indices in
            **unspecified order**, ``-1`` **suffix** padded
            (``sparsecore_topk.py:25-28``). The suffix contract is load-bearing:
            ``n_valid`` is computed as a *count*, so an interior ``-1`` would
            silently shift real ids left. Nothing downstream can detect that,
            which is why the producer's contract is asserted in
            ``tests/kernels/glm5_kpool/test_expand.py`` rather than assumed.
        positions: ``i32[T]`` absolute query position of each token.
        valid_token: ``bool[T]``; false rows are forced to all ``-1``.
        index_kpool: 4 for GLM-5.3-Flash.
        out_width: from :func:`derive_out_width`. Asserted, see below.

    Returns:
        ``i32[T, out_width]``, ``-1`` suffix padded.

    Two invariants, both of which the reference implementation violates and
    both of which the sparse MLA consumer requires (``out_width`` is checked by
    :func:`_assert_out_width`, which raises ``ValueError`` rather than
    asserting -- it is a configuration guard and must survive ``python -O``):

    **1. Valid entries form a contiguous prefix.**
    ``kernels/mla/sparse/kernel.py:209`` and ``:339`` mask by column
    *position* -- ``s = jnp.where(kv_span < kv_len, ...)`` against
    ``kv_lens = jnp.sum(topk_indices != -1, axis=-1)`` at ``:622``. A leading or
    interior ``-1`` therefore does not mask anything; it makes the kernel read
    one column past its own valid prefix and gather a decoy row at full weight.
    ``Glm5NextTextIndexer`` writes its tail at the fixed column
    ``index_kpool * select_k`` regardless of how many pools were valid
    (``modeling_glm5_next.py:860-874``), so its rows *do* have interior holes.
    This op compacts instead, which is the only reason no value-based mask has
    to be added to the sparse kernel.

    **2. Pool ids and tail ids never collide**, so no id is emitted twice and
    the prefix length is exactly ``n_valid + tail_count``. A pool ``j`` is only
    a candidate when ``index_kpool*j + index_kpool - 1 <= p``, hence
    ``index_kpool*(j+1) <= index_kpool * floor((p+1)/index_kpool) ==
    tail_start``: every expanded pool id is strictly below every tail id.

    There is no sort. Ordering within the prefix is unspecified, exactly as
    ``streamindex_topk``'s own output is, and the sparse kernel is
    order-agnostic (its softmax at ``kernels/mla/sparse/kernel.py:196-212`` is a
    single block over whatever arrives).
    """
    assert pool_ids.ndim == 2, pool_ids.shape
    num_tokens, select_k = pool_ids.shape
    assert positions.shape == (num_tokens, ), positions.shape
    assert valid_token.shape == (num_tokens, ), valid_token.shape
    assert index_kpool >= 1
    # Static, at trace time: see `derive_out_width`.
    _assert_out_width(out_width, select_k, index_kpool)

    pool_ids = pool_ids.astype(jnp.int32)
    positions = positions.astype(jnp.int32)

    cols = jnp.arange(out_width, dtype=jnp.int32)  # [W]
    # Column c of the expansion belongs to selection slot c // index_kpool.
    # Columns at or past `select_k * index_kpool` are always >= n_valid and so
    # always masked; clamping the gather index just keeps it in bounds.
    slot = jnp.minimum(cols // index_kpool, select_k - 1)
    within = cols - slot * index_kpool
    src = pool_ids[:, slot]  # [T, W]
    # The `src == -1` arm is a **no-op under the suffix contract** -- those
    # columns are all at or past `n_valid` and get masked below anyway. It is
    # here to bound the damage if a future producer ever emits an interior
    # `-1`: without it, `-1 * index_kpool + m` would put ids -4..-1 into the
    # prefix and `dsa_gather` would address a page from a negative index. With
    # it the row is merely short. Neither is a correct row; one of them is not
    # an out-of-bounds read.
    expanded = jnp.where(src == -1, jnp.int32(-1),
                         src * index_kpool + within[None, :])

    # `-1` is a *suffix*, so the count is also the prefix length.
    n_valid = index_kpool * jnp.sum(pool_ids != -1, axis=-1).astype(jnp.int32)

    visible = positions + 1
    tail_count = jnp.remainder(visible, index_kpool)
    tail_start = visible - tail_count
    tail = tail_start[:, None] + (cols[None, :] - n_valid[:, None])

    out = jnp.where(
        cols[None, :] < n_valid[:, None],
        expanded,
        jnp.where(cols[None, :] < (n_valid + tail_count)[:, None], tail,
                  jnp.int32(-1)),
    )
    return jnp.where(valid_token[:, None], out,
                     jnp.int32(-1)).astype(jnp.int32)
