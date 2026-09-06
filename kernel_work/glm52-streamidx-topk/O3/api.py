"""Public entry point for O3's selector: same signature as the incumbent's `sparsecore_topk`."""

# Dimensions.
#   B   rows of the score matrix
#   N   padded score extent per row
#   K   top-k
#
# No config here, by rule (code-style.md 10): every knob lives in
# pallas_mosaic_tpu.py. jaxtyping is not installed in the target image, so the
# shape contract is a trailing comment plus a boundary check rather than an
# annotation -- a declared deviation, not an omission.

import jax

from pallas_mosaic_tpu_kernel import sparsecore_topk as _sparsecore_topk


def streamidx_select(
    scores: jax.Array,  # [B, N] f32
    k: int,
    row_lengths: jax.Array | None = None,  # [B] i32, defaults to N
    *,
    write_empty_rows: bool = True,
) -> jax.Array:  # [B, K] i32
    """Exact top-k column indices per row, unsorted, -1 suffix-padded.

    The backend fallback chain is the caller's: this raises
    NotImplementedError (never ValueError) when the SparseCore is absent or the
    row is too wide for one resident slice, so a caller can fall back to
    `base._fwd` on that exception alone.
    """
    return _sparsecore_topk(scores, k, row_lengths,
                            write_empty_rows=write_empty_rows)
