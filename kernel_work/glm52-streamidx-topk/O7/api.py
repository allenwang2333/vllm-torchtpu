"""O7 public entry point: exact unsorted top-k over f32 score rows.

Same contract as the incumbent `sparsecore_topk`: top-k column indices per row
in unspecified order, `-1` suffix-padded, matching an exact top-k as a SET
(equal boundary scores may be broken differently). `-inf` scores and positions
at or beyond `row_lengths` are never selected. NaNs are unsupported.

No `Config` appears in this file -- callers who want one reach for
`pallas_mosaic_tpu.sc_topk_direct` directly.

Declared deviation from `code-style.md` §4 and its §16 checklist: the public
wrapper is NOT jaxtyping-annotated, only shape-commented, and neither is
`base._fwd`. This variant is measured as a drop-in for the shipping
`sparsecore_topk`, which carries no jaxtyping either, and the arms are swapped
by rebinding a module global -- an annotation the incumbent does not have would
be a difference between the arms that is not the change under test. If this
variant ships, the annotations go on then.
"""
import functools

import jax
import jax.numpy as jnp

from pallas_mosaic_tpu import Config, pick_partition, sc_topk_direct

LANES_MIN = 16


@functools.partial(jax.jit,
                   static_argnames=("k", "write_empty_rows", "cfg"))
def topk_indices(
    scores,  # [B, N] f32
    k,
    row_lengths=None,  # [B] i32, defaults to N
    *,
    write_empty_rows=True,
    cfg=Config(),
):  # -> [B, K] i32
    if scores.ndim != 2:
        raise NotImplementedError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype != jnp.float32:
        raise NotImplementedError(f"scores must be f32, got {scores.dtype}")
    b, n = scores.shape
    if n % LANES_MIN != 0:
        raise NotImplementedError(f"{n=} must be a multiple of {LANES_MIN}")
    if k % LANES_MIN != 0 or not 0 < k <= 4096:
        raise NotImplementedError(
            f"{k=} must be a positive multiple of {LANES_MIN}, at most 4096")

    if row_lengths is None:
        row_lengths = jnp.full((b, ), n, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)

    p = pick_partition(b, n, k)
    if p == 1:
        return sc_topk_direct(scores, k, row_lengths, cfg=cfg,
                              write_empty=write_empty_rows)

    # Stage 1: P slices per row, local top-k on each.
    n_p = n // p
    k_p = min(k, n_p)
    scores_p = scores.reshape(b * p, n_p)
    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0,
                         n_p).reshape(b * p)

    local_indices = sc_topk_direct(scores_p, k_p, lengths_p, cfg=cfg,
                                   write_empty=True).reshape(b, p, k_p)
    global_cand_indices = jnp.where(
        local_indices < 0, -1,
        offsets[None, :, None] + local_indices).reshape(b, p * k_p)

    safe_cand_indices = jnp.maximum(global_cand_indices, 0)
    cand_scores = jnp.take_along_axis(scores, safe_cand_indices, axis=1)
    cand_scores = jnp.where(global_cand_indices >= 0, cand_scores, -jnp.inf)

    # Stage 2: merge the P * k_p candidates.
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))
    final_cand_slots = sc_topk_direct(cand_scores, k, cand_lengths, cfg=cfg,
                                      write_empty=write_empty_rows)

    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices, safe_slots,
                                        axis=1)
    return jnp.where(final_cand_slots >= 0, final_indices, -1)
