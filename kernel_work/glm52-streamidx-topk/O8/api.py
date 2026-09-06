"""The four selector entry points this variant compares. No config appears here.

Shapes are given as trailing comments in the glossary's letters rather than as
jaxtyping annotations: `jaxtyping` is not installed in the benchmark image, and
importing it would make every arm fail to import rather than fail to be typed.
Declared in `DEVIATIONS.md`.

  inc       the shipping `sparsecore_topk`, imported, untouched. What ships.
  inc_copy  this variant's kernel copy with both new flags off. Times the copy
            itself, so a difference between `inc` and `inc_copy` is transcription
            drift and not the change under test.
  flat      the incumbent with O3's flat-index gather substituted for
            `take_along_axis`. Built on the SHIPPING `_sc_topk_direct`, not on
            this variant's copy and not imported from `O3/` -- it is the arm
            that would otherwise ship, so it must be the shipping kernel with
            one line changed.
  o8        stage 1 emits (index, key) pairs; the [B, N] gather does not exist.
"""

import functools

import jax
import jax.numpy as jnp

import tiling
from pallas_mosaic_tpu import _get_heuristics_config, select
from base import bind

# The shipping selector, and its internals, which the `flat` arm is built from.
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    _pick_partition as _shipping_pick_partition)
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    _sc_topk_direct as _shipping_direct)
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    sparsecore_topk as inc_topk)

__all__ = ["inc_topk", "inc_copy_topk", "flat_topk", "o8_topk"]


@functools.partial(jax.jit, static_argnames=("k", "write_empty_rows"))
def inc_copy_topk(
    scores: jax.Array,  # [B, N] f32
    k: int,
    row_lengths: jax.Array | None = None,  # [B] i32
    *,
    write_empty_rows: bool = True,
) -> jax.Array:  # [B, k] i32
    row_lengths = bind(scores, k, row_lengths)
    cfg = _get_heuristics_config(scores, k, arm="inc")
    return select(scores, k, row_lengths, cfg=cfg,
                  write_empty_rows=write_empty_rows)


@functools.partial(jax.jit, static_argnames=("k", "write_empty_rows"))
def o8_topk(
    scores: jax.Array,  # [B, N] f32
    k: int,
    row_lengths: jax.Array | None = None,  # [B] i32
    *,
    write_empty_rows: bool = True,
) -> jax.Array:  # [B, k] i32
    row_lengths = bind(scores, k, row_lengths)
    cfg = _get_heuristics_config(scores, k, arm="o8")
    return select(scores, k, row_lengths, cfg=cfg,
                  write_empty_rows=write_empty_rows)


@functools.partial(jax.jit, static_argnames=("k", "write_empty_rows"))
def flat_topk(
    scores: jax.Array,  # [B, N] f32
    k: int,
    row_lengths: jax.Array | None = None,  # [B] i32
    *,
    write_empty_rows: bool = True,
) -> jax.Array:  # [B, k] i32
    """The incumbent with the stage-2 gather taken flat.

    Reimplemented here rather than imported from `O3/`. Everything except the
    four gather lines is the shipping body, and the two `_sc_topk_direct` calls
    are the shipping function -- so this arm differs from `inc` in the gather
    and nothing else, which is what makes it a usable second baseline.
    """
    row_lengths = bind(scores, k, row_lengths)
    b, n = scores.shape
    p = _shipping_pick_partition(b, n, k)
    if p == 1:
        return _shipping_direct(scores, k, row_lengths,
                                write_empty=write_empty_rows)

    n_p = n // p
    k_p = min(k, n_p)
    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0,
                         n_p).reshape(b * p)
    local_indices = _shipping_direct(scores.reshape(b * p, n_p), k_p,
                                     lengths_p,
                                     write_empty=True).reshape(b, p, k_p)
    global_cand_indices = jnp.where(
        local_indices < 0, -1,
        offsets[None, :, None] + local_indices).reshape(b, p * k_p)

    # -- the substitution, and the only difference from `inc` ---------------
    # int32 flat indices: B*N must stay inside int32. At the widest shape here
    # (B=1024, N=262144) that is 2.68e8, well under 2.1e9, but the guard is
    # cheap and a silent wrap would look like a correctness bug in the kernel.
    if b * n > 2**31 - 1:
        raise NotImplementedError(
            f"{b}*{n} flat elements overrun the int32 gather index")
    words_flat = jax.lax.bitcast_convert_type(scores, jnp.int32).reshape(-1)
    safe_cand_indices = jnp.maximum(global_cand_indices, 0)
    row_base = (jnp.arange(b, dtype=jnp.int32) * jnp.int32(n))[:, None]
    cand_words = jnp.take(words_flat,
                          (row_base + safe_cand_indices).reshape(-1),
                          mode="clip").reshape(b, p * k_p)
    cand_scores = jax.lax.bitcast_convert_type(cand_words, jnp.float32)
    # -----------------------------------------------------------------------

    cand_scores = jnp.where(global_cand_indices >= 0, cand_scores, -jnp.inf)
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))
    final_cand_slots = _shipping_direct(cand_scores, k, cand_lengths,
                                        write_empty=write_empty_rows)
    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices, safe_slots,
                                        axis=1)
    return jnp.where(final_cand_slots >= 0, final_indices, -1)


def check_policy_agrees(shapes) -> list[str]:
    """P must be identical between this copy and the shipping policy.

    A divergence would move `inc_copy`, `flat` and `o8` together and read as a
    result. `shapes` is [(B, N, k), ...].
    """
    bad = []
    for b, n, k in shapes:
        mine = tiling.pick_partition(b, n, k)
        theirs = _shipping_pick_partition(b, n, k)
        if mine != theirs:
            bad.append(f"B={b} N={n} k={k}: copy p={mine} shipping p={theirs}")
    return bad
