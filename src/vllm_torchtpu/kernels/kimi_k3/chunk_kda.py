# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked Kimi Delta Attention (KDA) Pallas TPU prefill kernel.

Adated from the implementation from the following tokamax PR:
https://github.com/openxla/tokamax/pull/1103

The work splits into two halves:

1. gate activation, chunk-local prefix sum and the intra-chunk solve, producing
   ``w/u/kg/Aqk/g_cumsum`` per chunk;
2. the inter-chunk recurrence and the output.

Both halves grid over ``(head block, batch, tile)``, where a tile holds up to
``chunk_size`` tokens of exactly one segment. The tile axis is static at the
worst-case tile count while a device-valued ``num_tiles`` bounds the real work,
so padding the runner's request axis costs skipped tiles rather than real ones.

They run as a single Pallas call, ``chunk_kda_fwd_fused``. Since both halves
walk the *same* grid and half 2 at tile ``p`` reads half 1 only at tile ``p``
(the sole cross-tile dependency is the recurrent state, which already lives in
a VMEM scratch), the six per-chunk tensors never have to reach HBM at all.
"""

from __future__ import annotations

import functools
import math
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import varlen_tiles

__all__ = ["chunk_kda", "restrict_cu_seqlens"]


def exp2(x):
    """Base-2 exponential, matching Triton's tl.exp2."""
    return jnp.exp2(x.astype(jnp.float32))


def get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


def cdiv(x, y: int):
    return (x + y - 1) // y


def align_up(x, align: int):
    return cdiv(x, align) * align


def segment_ids_to_seqlens(
    segment_ids: jax.Array,
    max_segs: int,
    chunk_size: int = 1,
) -> jax.Array:
    """Convert 1-indexed segment IDs with 0 padding to FLA-style cu_seqlens."""
    if segment_ids.ndim == 2:
        rows = [
            segment_ids_to_seqlens(segment_ids[b], max_segs, chunk_size)
            for b in range(segment_ids.shape[0])
        ]
        return jnp.stack(rows, axis=0)
    if segment_ids.ndim != 1:
        raise ValueError(
            f"`segment_ids` must be [T] or [B, T], got {segment_ids.shape}.")

    seg = segment_ids.reshape(-1)
    valid = seg != 0
    is_boundary = jnp.concatenate([
        jnp.ones(1, dtype=jnp.bool_),
        seg[1:] != seg[:-1],
    ]) & valid
    seg_idx = jnp.where(valid, jnp.cumsum(is_boundary.astype(jnp.int32)), 0)
    n_segs = seg_idx.max()
    n_real = jnp.sum(valid).astype(jnp.int32)

    is_end = jnp.concatenate([
        seg[1:] != seg[:-1],
        jnp.ones(1, dtype=jnp.bool_),
    ]) & valid
    prefix_len = jnp.cumsum(valid.astype(jnp.int32))

    drop_idx = jnp.asarray(max_segs + 1, dtype=jnp.int32)
    scatter_idx = jnp.where(is_end, seg_idx, drop_idx)
    scatter_val = jnp.where(is_end, prefix_len, 0)

    cu_seqlens = jnp.zeros((max_segs + 1, ), dtype=jnp.int32)
    cu_seqlens = cu_seqlens.at[scatter_idx].max(scatter_val, mode="drop")
    out_idx = jnp.arange(max_segs + 1, dtype=jnp.int32)
    return jnp.where(out_idx > n_segs, n_real, cu_seqlens)


class _TilePlan(NamedTuple):
    """A (sequence, tile) walk over the ragged token axis.

    Every tile holds up to ``chunk_size`` tokens of exactly one segment, so no
    tile straddles a segment boundary.

    Attributes:
        num_tiles: ``[B]``; how many tiles the batch's real content needs.
        r_base, r_size: ``[B, NT]``; each tile's first row on the token axis and
            how many rows it holds.
        s_idx: ``[B, NT]``; each tile's segment, which is where it reads and
            writes recurrent state.
        is_first, is_last: ``[B, NT]``; where a segment's state is read in and
            written back.
    """
    num_tiles: jax.Array
    r_base: jax.Array
    r_size: jax.Array
    s_idx: jax.Array
    is_first: jax.Array
    is_last: jax.Array


def restrict_cu_seqlens(
    cu_seqlens: jax.Array,  # [..., N + 1]
    start_seq: jax.Array | int,
) -> jax.Array:
    """Collapse segments before ``start_seq`` to zero length.

    ``cu_seqlens`` is non-decreasing, so clamping it from below at its own
    ``start_seq`` entry makes every earlier boundary equal and leaves every later
    one untouched. Segments before ``start_seq`` then hold no tokens and get no
    tiles, while the ones after keep both their absolute token offsets and their
    absolute segment indices -- which is what a caller needs to keep addressing
    recurrent state by request slot.

    This is how a caller restricts the chunked kernel to the prefill part of a
    batch, with the boundary a *device* scalar: a host-side branch on batch
    composition cannot work under vLLM's guard-free single trace.
    """
    floor = jnp.take(cu_seqlens, jnp.asarray(start_seq, jnp.int32),
                     axis=-1)[..., None]
    return jnp.maximum(cu_seqlens, floor)


def _plan_tiles(
    cu_seqlens: jax.Array,  # [B, N + 1]
    *,
    num_tokens: int,
    chunk_size: int,
) -> _TilePlan:
    """Plan one tile per ``chunk_size`` tokens of each segment."""
    batch, num_cu = cu_seqlens.shape
    num_segments = num_cu - 1
    # A tile holds at least one token, so the token count bounds the tile count;
    # and each segment rounds its own token count up to a whole tile, which adds
    # at most one tile per segment.
    max_tiles = min(num_tokens, cdiv(num_tokens, chunk_size) + num_segments)

    query_lens = jnp.diff(cu_seqlens, axis=-1)
    num_tiles, r_base, r_size, s_idx = [], [], [], []
    is_first, is_last = [], []
    for index in range(batch):
        plan = varlen_tiles.plan_per_seq_tiles(
            # Passing `query_lens` as `seq_lens` makes the planner's
            # `s_idx_has_initial_state` uniformly False, and it is unused here:
            # whether a segment carries state in is static for this kernel --
            # `initial_state` is either given for every slot, pre-zeroed by the
            # caller for the fresh ones, or absent altogether.
            seq_lens=query_lens[index],
            query_start_loc=cu_seqlens[index],
            max_tiles=max_tiles,
            chunk_size=chunk_size,
            tile_size=chunk_size,
        )
        valid = jnp.arange(max_tiles) < plan.num_tiles
        num_tiles.append(plan.num_tiles)
        r_base.append(jnp.where(valid, plan.p_id_to_r_base, 0))
        r_size.append(jnp.where(valid, plan.p_id_to_r_size, 0))
        s_idx.append(jnp.where(valid, plan.p_id_to_s_idx, num_segments - 1))
        is_first.append(valid & plan.p_id_is_first_tile)
        is_last.append(valid & plan.p_id_is_last_tile)

    return _TilePlan(
        num_tiles=jnp.stack(num_tiles).astype(jnp.int32),
        r_base=jnp.stack(r_base).astype(jnp.int32),
        r_size=jnp.stack(r_size).astype(jnp.int32),
        s_idx=jnp.stack(s_idx).astype(jnp.int32),
        is_first=jnp.stack(is_first),
        is_last=jnp.stack(is_last),
    )


_VMEM_FRACTION = 0.9

RCP_LN2 = 1.0 / math.log(2)

# =============================================================================
# Mini-batch sizing and shared forward recurrence
# =============================================================================


def estimate_mini_batch(
    per_tile_bytes: int,
    total: int,
    *,
    max_mb: int = 16,
    vmem_budget: int | None = None,
) -> int:
    """Choose the largest mini-batch that fits VMEM and divides ``total``."""
    if vmem_budget is None:
        vmem_budget = int(_VMEM_FRACTION *
                          pltpu.get_tpu_info().vmem_capacity_bytes)

    per_tile_bytes = max(1, per_tile_bytes)
    MB = max(1, vmem_budget // per_tile_bytes)
    MB = max(1, min(MB, total, max_mb))

    while total % MB != 0 and MB > 1:
        MB -= 1

    return MB


# =============================================================================
# Fused gate cumsum and intra-chunk solve
# =============================================================================


def _fused_gate_intra_tile(
    q,  # [MB, BT, K]
    k,  # [MB, BT, K]
    g,  # [MB, BT, K]
    beta,  # [MB, BT, 1]
    v,  # [MB, BT, V]
    A_val,  # [MB] float32
    dt_b,  # [MB, K] float32
    r_size,  # device scalar
    *,
    activation_dtype,
    chunk_size: int,
    head_dim: int,
    value_dim: int,
    scale: float,
    cumsum_scale: float,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    mini_batch: int = 1,
):
    """Gate activation + cumsum + intra-chunk solve for one tile of MB heads.

  Uses Neumann series inversion; bfloat16 only (see the module docstring).

  Only the first ``r_size`` rows hold real tokens. The rest are whatever the
  staging buffer held from the previous tile, so every input is masked to zero
  first -- see the mask comment below for why that is the correct padding value
  and why the gate has to be masked *after* activation.

  Everything returned stays float32: nothing crosses an HBM boundary between
  the halves, so there is no narrow tensor to round to.

  Args:
      activation_dtype: the caller's activation dtype, which is what selects
          the solve. Passed in rather than read off ``q``, because the staged
          inputs arrive already widened to float32 (see the staging note in
          ``chunk_kda_fwd_fused``) and so no longer carry it.
      r_size: rows of this tile that hold real tokens, ``<= chunk_size``.
      mini_batch: number of heads processed per grid point (MB).

  Returns:
      ``(u, w, kg, Aqk, g_cumsum, q_masked)``; the last is ``q`` with its padded
      rows zeroed, which stage 3+4 needs in the same masked form.
  """
    BT = chunk_size
    BC = 16
    NC = BT // BC
    BC_diag = 8
    K = head_dim
    V = value_dim
    MB = mini_batch

    # Rows past the tile's real length hold whatever the staging buffer had --
    # the previous tile's tokens, or, on the first tile, uninitialised VMEM.
    # Zeroing q/k/v/beta makes them inert: with beta 0 the solve's `L` rows
    # vanish, so `u`/`w` come out zero there, and with k zero those rows add
    # nothing to stage 3's `kg^T @ v_new`.
    #
    # This has to be a select, not a multiply by a 0/1 mask: uninitialised VMEM
    # can hold a NaN bit pattern and `0.0 * NaN` is NaN, which then spreads
    # through the Neumann solve's full-width matmuls into the real rows. Masks
    # are built as 2-D iota comparisons and broadcast on a *leading* axis, which
    # avoids the (BT,) -> (BT, 1) i1 shape casts Mosaic rejects.
    def _row_keep(last_dim):
        rows = jax.lax.broadcasted_iota(jnp.int32, (BT, last_dim), dimension=0)
        return (rows < r_size)[None]  # [1, BT, last_dim]

    keep_k = _row_keep(K)
    keep_v = _row_keep(V)
    keep_1 = _row_keep(1)

    def _mask(x, keep):
        return jnp.where(keep, x, 0.0)

    # --- Gate activation + cumsum ---
    g_f32 = g.astype(jnp.float32)

    if use_gate_in_kernel:
        g_f32 = g_f32 + dt_b[:, None, :]  # [MB, BT, K]
        if lower_bound is None:
            g_f32 = -jnp.exp(A_val)[:, None, None] * jax.nn.softplus(g_f32)
        else:
            g_f32 = lower_bound * jax.nn.sigmoid(
                jnp.exp(A_val)[:, None, None] * g_f32)

    # The gate is masked *after* activation, not before. Both activations map 0
    # to a non-zero decay, so masking the raw gate would leave padded rows
    # decaying; masking the activated value puts `exp2` at 1 there in either
    # mode. That matters beyond the padded rows themselves, because stage 3
    # applies `g_cumsum[BT - 1]` -- the running sum over the *whole* tile -- to
    # the entire recurrent state.
    g_f32 = _mask(g_f32, keep_k)

    tril = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
    g_cumsum = jax.lax.dot_general(
        tril,
        g_f32,
        (((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
    ).transpose(1, 0, 2) * cumsum_scale

    q_f32 = _mask(q.astype(jnp.float32), keep_k)
    k_f32 = _mask(k.astype(jnp.float32), keep_k)
    beta_f32 = _mask(beta.astype(jnp.float32), keep_1)
    v_f32 = _mask(v.astype(jnp.float32), keep_v)

    # --- BC=16 sub-block Aqk/L ---
    # The quantity wanted is, for every causal pair (r, t):
    #   Aqk[m,r,t] = sum_k q[m,r,k] * k[m,t,k] * exp2(g_cumsum[m,r,k]
    #                                                 - g_cumsum[m,t,k])
    # KDA's gate is *per channel*, so that decay is a rank-3 tensor over
    # (r, t, k) and the sum over k cannot be folded into a plain matmul. The
    # usual trick is to factor it through a per-sub-block reference row,
    #   exp2(g_r - g_ref) * exp2(g_ref - g_t),
    # which restores the matmul. But those two factors are a tiny*huge pair:
    # their product is exp2(g_r - g_t) <= 1, while individually they saturate
    # float32's exp2 range (~+/-128). Kimi-Linear's gate is unbounded
    # (-exp(A_log) * softplus(.)) and reaches -1736 for a *single* token, so
    # inside one sub-block one factor becomes 0 and the other inf, giving
    # 0 * inf = NaN. Measured on real layer-0 weights: 26 of 32 heads NaN,
    # which reached the sampler as non-finite logits and produced empty output.
    # So the diagonal block is computed *pairwise* instead. For causal pairs
    # r >= t the exponent g_r - g_t is non-positive (g_cumsum is monotonically
    # decreasing, since the activated gate is <= 0), so exp2 cannot overflow and
    # underflow to 0 is the correct answer -- a fully decayed contribution.
    # Columns strictly before the sub-block (t < i_s) keep the fast factored
    # matmul: there both exponents are provably <= 0 as well, because
    # g_cumsum[t] >= gn_ref for t < i_s and g_i[r] <= gn_ref for r >= i_s.
    #
    # Note what makes that second case safe: the reference row sits *between*
    # r and t. That holds at any granularity, not just at the sub-block
    # boundary, so `_diag_block` below recurses -- it bisects the diagonal
    # block and factors the off-diagonal half through the row on the seam,
    # leaving only BC_diag x BC_diag leaves genuinely pairwise. The pairwise
    # path costs BC_diag * K of vector work per row, so each level of bisection
    # halves the dominant term; the leaf size is a tuning knob, not a
    # correctness one.
    #
    # PRECONDITION: the activated gate is <= 0, i.e. `lower_bound` (when given)
    # is negative. `chunk_kda` validates this.
    #
    # Use broadcasted_iota for 2D indices to avoid Mosaic-unsupported i1
    # shape casts (e.g. (BT,) -> (BT, 1) on bool tensors).
    row_iota_bt_k = jax.lax.broadcasted_iota(jnp.int32, (BT, K), dimension=0)
    row_iota_bc = jax.lax.broadcasted_iota(jnp.int32, (BC, BC), dimension=0)
    col_iota_bc = jax.lax.broadcasted_iota(jnp.int32, (BC, BC), dimension=1)
    causal_bc = (row_iota_bc >= col_iota_bc)[None]  # [1, BC, BC]
    strict_bc = (row_iota_bc > col_iota_bc)[None]  # [1, BC, BC]

    # Contract the channel axis: [MB, BC, K] x [MB, K] -> [MB, BC]
    def _mv_k(lhs, rhs):
        return jax.lax.dot_general(
            lhs,
            rhs,
            (((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )

    def _diag_block(q_i, k_i, g_i, size):
        """Causal ``q @ k^T`` and ``k @ k^T`` over a [size, size] diagonal block.

        Both are returned unmasked; the caller applies the causal/strict masks.
        Bisects down to ``BC_diag`` rows, where the per-channel decay is applied
        pairwise. See the discussion above for why the recursive step is
        overflow-free at any gate magnitude.

        ``size`` is a Python int, so the whole recursion unrolls at trace time
        and Mosaic sees straight-line code -- no dynamic control flow.
        """
        # Base case: a BC_diag-row leaf, small enough that the pairwise decay
        # is affordable. One iteration per row, so BC_diag=8 means 8 steps.
        if size <= BC_diag:
            rows_q = []
            rows_k = []
            for r in range(size):
                # g_i[r] - g_i[t] <= 0 for t <= r; the clamp is a no-op there
                # and hard bounds the exponent for the masked-away t > r.
                kw = k_i * jnp.exp2(jnp.minimum(g_i[:, r:r + 1, :] - g_i, 0.0))
                rows_q.append(_mv_k(kw, q_i[:, r, :])[:, None, :])  # [MB,1,sz]
                rows_k.append(_mv_k(kw, k_i[:, r, :])[:, None, :])
            return (jnp.concatenate(rows_q, axis=1),
                    jnp.concatenate(rows_k, axis=1))

        # Recursive step. Split into halves and handle the three surviving
        # quadrants separately; at size=BC=16 this is the only level, h=8.
        h = size // 2

        # Quadrants [0, h)x[0, h) and [h, size)x[h, size) are themselves
        # diagonal blocks, one level smaller. Recurse.
        q00, k00 = _diag_block(q_i[:, :h], k_i[:, :h], g_i[:, :h], h)
        q11, k11 = _diag_block(q_i[:, h:], k_i[:, h:], g_i[:, h:], h)

        # Quadrant [h, size)x[0, h) is entirely below the diagonal, so every
        # pair in it is causal and the decay can be factored through the seam
        # row g_i[h] -- which lies between r and t for every pair here, making
        # both exponents <= 0: for r >= h, g_i[r] - g_i[h] <= 0, and for t < h,
        # g_i[h] - g_i[t] <= 0. That is what buys the matmul.
        ref = g_i[:, h:h + 1, :]  # [MB, 1, K]
        lo = k_i[:, :h] * jnp.exp2(jnp.minimum(ref - g_i[:, :h], 0.0))
        hi_e = jnp.exp2(jnp.minimum(g_i[:, h:] - ref, 0.0))
        # q and k contract against the same `lo`, so stack them into one MXU
        # call rather than two: [MB, 2h, K] x [MB, h, K] over the channel axis.
        cross = jax.lax.dot_general(
            jnp.concatenate([q_i[:, h:] * hi_e, k_i[:, h:] * hi_e], axis=1),
            lo,
            (((2, ), (2, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )  # [MB, 2*h, h]

        # The [0, h) x [h, size) corner is above the diagonal; the caller masks
        # it away, so fill it with zeros rather than computing it.
        zero = jnp.zeros((MB, h, h), jnp.float32)

        def _assemble(top_left, bot_left, bot_right):
            # Stitch the four quadrants back into one [MB, size, size] block.
            return jnp.concatenate([
                jnp.concatenate([top_left, zero], axis=2),
                jnp.concatenate([bot_left, bot_right], axis=2),
            ],
                                   axis=1)

        return (_assemble(q00, cross[:, :h], q11),
                _assemble(k00, cross[:, h:], k11))

    Aqk_rows = []
    L_rows = []
    for i_sc in range(NC):
        i_s = i_sc * BC
        q_i = q_f32[:, i_s:i_s + BC]  # [MB, BC, K]
        k_i = k_f32[:, i_s:i_s + BC]  # [MB, BC, K]
        g_i = g_cumsum[:, i_s:i_s + BC]  # [MB, BC, K]
        beta_i = beta_f32[:, i_s:i_s + BC]  # [MB, BC, 1]

        # --- columns t < i_s: factored matmul (both exponents <= 0) ---
        gn_ref = g_i[:, 0:1, :]  # [MB, 1, K] — sub-block row 0
        exp_diff_i = jnp.exp2(jnp.minimum(g_i - gn_ref, 0.0))
        q_eg = q_i * exp_diff_i  # [MB, BC, K]
        k_eg = k_i * exp_diff_i  # [MB, BC, K]

        valid_j = (row_iota_bt_k < i_s).astype(jnp.float32)  # [BT, K]
        diff_j = jnp.minimum(gn_ref - g_cumsum,
                             0.0) * valid_j[None]  # [MB, BT, K]
        k_eng_full = k_f32 * jnp.exp2(diff_j) * valid_j[None]  # [MB, BT, K]

        # Stack q_eg / k_eg: [MB, 2*BC, K] x [MB, BT, K] -> [MB, 2*BC, BT]
        qk_eg = jnp.concatenate([q_eg, k_eg], axis=1)  # [MB, 2*BC, K]
        qk_dot = jax.lax.dot_general(
            qk_eg,
            k_eng_full,
            (((2, ), (2, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )  # [MB, 2*BC, BT]
        Aqk_row = qk_dot[:, :BC]  # [MB, BC, BT]
        Akk_row = qk_dot[:, BC:]  # [MB, BC, BT]

        # --- diagonal block t in [i_s, i_s + BC): exact per-channel decay ---
        Aqk_diag, Akk_diag = _diag_block(q_i, k_i, g_i, BC)  # [MB, BC, BC]
        Aqk_diag = jnp.where(causal_bc, Aqk_diag, jnp.float32(0.0))
        Akk_diag = jnp.where(strict_bc, Akk_diag, jnp.float32(0.0))

        # Widen the [BC, BC] block to [BC, BT] at column offset i_s. Built by
        # concatenation so every shape stays static; zero-width pads are skipped
        # rather than passed to concatenate.
        def _widen(blk):
            parts = []
            if i_s:
                parts.append(jnp.zeros((MB, BC, i_s), jnp.float32))
            parts.append(blk)
            tail = BT - i_s - BC
            if tail:
                parts.append(jnp.zeros((MB, BC, tail), jnp.float32))
            return parts[0] if len(parts) == 1 else jnp.concatenate(parts,
                                                                    axis=2)

        Aqk_row = (Aqk_row + _widen(Aqk_diag)) * scale
        Akk_row = (Akk_row + _widen(Akk_diag)) * beta_i

        Aqk_rows.append(Aqk_row)
        L_rows.append(Akk_row)

    Aqk = jnp.concatenate(Aqk_rows, axis=1)  # [MB, BT, BT]
    L = jnp.concatenate(L_rows, axis=1)  # [MB, BT, BT]

    # --- Solve (I + L) x = rhs ---
    v_beta = v_f32 * beta_f32  # [MB, BT, V]
    k_eg_beta = k_f32 * jnp.exp2(g_cumsum) * beta_f32  # [MB, BT, K]
    I_bt = jnp.eye(BT, dtype=jnp.float32)  # [BT, BT]

    # Batched dot helper: [MB, M, K] @ [MB, K, N] → [MB, M, N]
    def _dot_batch(a, b):
        return jax.lax.dot_general(
            a,
            b,
            (((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )

    use_neumann = activation_dtype != jnp.float32

    if use_neumann:
        # --- Neumann series inversion ---
        BC_inv = 8
        NC_inv = BT // BC_inv
        inv_dtype = jnp.float32

        L_inv = L.astype(inv_dtype)  # [MB, BT, BT]

        _idx = jnp.arange(BT, dtype=jnp.int32)
        _block_id = _idx // BC_inv
        _same_block = (_block_id[:,
                                 None] == _block_id[None, :]).astype(inv_dtype)
        L_diag = L_inv * _same_block[None]  # [MB, BT, BT]
        F = L_inv - L_diag  # [MB, BT, BT]

        neg_Ld = -L_diag
        S = I_bt[None] + neg_Ld  # [MB, BT, BT]
        Mk = neg_Ld
        num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
        for _ in range(num_diag_steps):
            Mk = _dot_batch(Mk, Mk)
            S = _dot_batch(S, I_bt[None] + Mk)
        P = S

        rhs = jnp.concatenate([
            v_beta.astype(inv_dtype),
            k_eg_beta.astype(inv_dtype),
        ],
                              axis=-1)  # [MB, BT, V+K]

        if NC_inv == 1:
            result = _dot_batch(P, rhs)
        else:
            # Fuse `P @ [F | rhs]` to share a matmul; split out G and P_rhs.
            F_and_rhs = jnp.concatenate([F, rhs], axis=-1)  # [MB, BT, BT+V+K]
            P_merged = _dot_batch(P, F_and_rhs)  # [MB, BT, BT+V+K]
            G = P_merged[:, :, :BT]  # [MB, BT, BT]
            P_rhs = P_merged[:, :, BT:]  # [MB, BT, V+K]

            # Compute inv_I_G = (I + G)^{-1} = sum_{k=0}^{NC_inv-1} (-G)^k via
            # Horner doubling. Build the matrix first (small (MB,BT,BT) matmuls),
            # then apply once to P_rhs.
            H_mat = -G
            inv_I_G = I_bt[None] + H_mat  # (I + H)
            Hk = H_mat
            log2_NC_inv = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[NC_inv]
            for step in range(log2_NC_inv - 1):
                Hk = _dot_batch(Hk, Hk)  # H^(2^(step+1))
                inv_I_G = inv_I_G + _dot_batch(inv_I_G,
                                               Hk)  # inv_I_G @ (I + Hk)

            result = _dot_batch(inv_I_G, P_rhs)  # [MB, BT, V+K]
    else:
        # The float32 branch used exact forward substitution instead of the Neumann
        # series. Removed with the rest of the fp32 path: Kimi-K3 runs bfloat16, so
        # `use_neumann` is always True here. See the module docstring.
        raise NotImplementedError(
            "float32 inputs are not supported; this kernel is scoped to Kimi-K3's "
            "bfloat16 path (use_neumann). Cast q/k/v/g to bfloat16.")

    u = result[:, :, :V]  # [MB, BT, V]
    w = result[:, :, V:V + K]  # [MB, BT, K]

    # --- kg ---
    g_last = g_cumsum[:, BT - 1:BT, :]  # [MB, 1, K]
    kg = k_f32 * exp2(g_last - g_cumsum)

    return u, w, kg, Aqk, g_cumsum, q_f32


# =============================================================================
# Megakernel: gate + intra-chunk solve + recurrence + output, in one pass
# =============================================================================


def _chunk_kda_fwd_fused_kernel(
    # Scalar prefetch: the tile plan.
    num_tiles_ref,  # [B]
    r_base_ref,  # [B, NT]
    r_size_ref,  # [B, NT]
    s_idx_ref,  # [B, NT] -- consumed by the state index maps, not the body
    is_first_ref,  # [B, NT]
    is_last_ref,  # [B, NT]
    # Activations, as whole-array HBM refs over the *unaligned* token axis.
    q_hbm,  # [H, B, T, K]
    k_hbm,  # [H, B, T, K]
    g_hbm,  # [H, B, T, K]
    beta_hbm,  # [H, B, T, 1]
    v_hbm,  # [H, B, T, V]
    # Per-head gate parameters.
    A_log_ref,  # [MB, 1, 1, 1, 1]
    dt_bias_ref,  # [MB, 1, 1, 1, K]
    _o_in,  # the zero-filled output, aliased; written through o_ref
    h0_ref,  # [1, 1, MB, K, V] or None
    # Outputs.
    o_ref,  # [H, B, T, V] whole-array HBM ref
    ht_ref,  # [1, 1, MB, K, V] or None
    # Scratch.
    q_buf,  # [MB, 1, BT, K] float32 staging
    k_buf,  # [MB, 1, BT, K]
    g_buf,  # [MB, 1, BT, K]
    beta_buf,  # [MB, 1, BT, 1]
    v_buf,  # [MB, 1, BT, V]
    h_scratch,  # [MB, K, V] float32 recurrent state
    o_buf,  # [MB, 1, BT, V] float32 output staging
    sem_in,
    sem_out,
    *,
    chunk_size: int,
    scale: float,
    use_initial_state: bool,
    store_final_state: bool,
    mini_batch: int,
    output_precision,
    **tile_kwargs,
):
    """Both halves of the algorithm for one tile, at one grid point.

  This used to be two Pallas calls, splitting here: half 1 wrote
  ``u/w/kg/Aqk/g_cumsum/q`` to HBM in tile layout and half 2 read them back.
  That boundary buys nothing -- the grids are identical, ``p`` is the innermost
  (sequential) axis in both, and half 2 at tile ``p`` touches half 1's results
  only at tile ``p``. The one genuine cross-tile dependency is the recurrent
  state, and that was already a VMEM scratch rather than a tensor. So the six
  intermediates become live values and the round trip disappears.

  It is close to VMEM-neutral, which is the part that makes it cheap: those six
  tensors cost *twelve* double-buffered pipeline windows across the two calls
  (six out, six in) and are replaced by one copy of each value.

  The intermediates also get wider for free: the HBM boundary used to round
  them to the activation dtype, and now there is no narrow tensor to round to,
  so they stay float32 all the way through.
  """
    del _o_in, s_idx_ref
    BT = chunk_size
    MB = mini_batch
    K = q_buf.shape[-1]
    V = v_buf.shape[-1]

    head_block = pl.program_id(0) * MB
    i_b = pl.program_id(1)
    p_id = pl.program_id(2)

    def _tile():
        r_base = r_base_ref[i_b, p_id]
        r_size = r_size_ref[i_b, p_id]

        # --- Stage the tile's rows out of HBM ---
        # A tile starts at an arbitrary token row, which no BlockSpec index map
        # can address, so the activations arrive as whole-array HBM refs and
        # this kernel issues its own DMAs at `r_base`. Rows past `r_size` keep
        # whatever the previous tile left behind; the solve masks them off.
        copies = [
            pltpu.make_async_copy(
                src.at[pl.ds(head_block, MB),
                       pl.ds(i_b, 1),
                       pl.ds(r_base, r_size)],
                dst.at[:, :, pl.ds(0, r_size)],
                sem_in,
            ) for src, dst in (
                (q_hbm, q_buf),
                (k_hbm, k_buf),
                (g_hbm, g_buf),
                (beta_hbm, beta_buf),
                (v_hbm, v_buf),
            )
        ]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

        # --- Half 1: gate activation, cumsum, intra-chunk solve ---
        b_u, b_w, b_kg, b_A, b_g, b_q = _fused_gate_intra_tile(
            q_buf[:, 0],
            k_buf[:, 0],
            g_buf[:, 0],
            beta_buf[:, 0],
            v_buf[:, 0],
            A_log_ref[:, 0, 0, 0, 0],
            dt_bias_ref[:, 0, 0, 0],
            r_size,
            chunk_size=BT,
            head_dim=K,
            value_dim=V,
            scale=scale,
            mini_batch=MB,
            **tile_kwargs,
        )

        # --- Half 2: inter-chunk recurrence and output ---
        # Init state on the first tile of THIS segment.
        @pl.when(is_first_ref[i_b, p_id])
        def _():
            if use_initial_state:
                h_scratch[:] = h0_ref[0, 0].astype(jnp.float32)
            else:
                h_scratch[:] = jnp.zeros((MB, K, V), dtype=jnp.float32)

        b_h = h_scratch[:]  # [MB, K, V] pre-update state

        # Delta correction: v_new = u - w @ h.
        # HIGHEST: v_new feeds the recursive state update directly.
        b_v_new = b_u - jnp.matmul(
            b_w,
            b_h,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )  # [MB, BT, V]

        # Inter-chunk output: scale * (q * exp2(g - g_ref)) @ (h * exp2(g_ref)).
        # Stabilized at g_ref = g[0], the tile's largest cumulative gate: the
        # gate is non-positive, so `g - g_ref` and `g_ref` are the exponents
        # that keep both factors in range without changing the product.
        b_g_ref_row = b_g[:, 0:1, :]  # [MB, 1, K]
        b_qg = b_q * jnp.exp2(jnp.maximum(b_g - b_g_ref_row,
                                          -126.0))  # [MB, BT, K]
        b_h_scaled = b_h * jnp.exp2(jnp.maximum(
            b_g_ref_row[:, 0, :], -126.0))[:, :, None]  # [MB, K, V]
        b_o = jnp.matmul(
            b_qg,
            b_h_scaled,
            precision=output_precision,
            preferred_element_type=jnp.float32,
        ) * scale  # [MB, BT, V]

        # Intra-chunk: A @ v_new. Aqk is causal by construction; the mask guards
        # against upper-triangle fp noise out of the Neumann solve.
        m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]  # [BT, BT]
        b_A = jnp.where(m_s[None, :, :], b_A, 0.0)
        b_o = b_o + jnp.matmul(
            b_A,
            b_v_new,
            precision=output_precision,
            preferred_element_type=jnp.float32,
        )  # [MB, BT, V]

        # Scatter this tile's rows to where its tokens live. `r_size` rows, so
        # rows the tile does not own keep the caller's zeros -- and a
        # dynamically sized slice is legal only because the output is float32
        # (see the staging note in `chunk_kda_fwd_fused`).
        o_buf[:, 0] = b_o
        copy = pltpu.make_async_copy(
            o_buf.at[:, :, pl.ds(0, r_size)],
            o_ref.at[pl.ds(head_block, MB),
                     pl.ds(i_b, 1),
                     pl.ds(r_base, r_size)],
            sem_out,
        )
        copy.start()
        copy.wait()

        # State update: h = h * exp2(g[-1]) + kg^T @ v_new.
        # HIGHEST: accumulates into the recursive hidden state.
        b_h_new = b_h * jnp.exp2(b_g[:, BT - 1, :])[:, :, None]  # decay
        b_h_new = b_h_new + jnp.matmul(
            b_kg.transpose(0, 2, 1),
            b_v_new,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )  # [MB, K, V]
        h_scratch[:] = b_h_new

        # Final state on the last tile of THIS segment.
        @pl.when(is_last_ref[i_b, p_id])
        def _():
            if store_final_state:
                ht_ref[0, 0] = h_scratch[:].astype(ht_ref.dtype)

    # Tiles past the batch's real content own no output rows (`r_size` 0) and
    # carry no segment's state (neither first- nor last-tile flag), so skipping
    # them is an optimization rather than a correctness requirement. What *is*
    # load-bearing is that `s_idx` names the final segment for them: that keeps
    # the state window from changing blocks, and so from flushing an
    # uninitialised buffer over a live slot.
    pl.when(p_id < num_tiles_ref[i_b])(_tile)


@functools.partial(
    jax.jit,
    static_argnames=[
        "num_tokens",
        "output_final_state",
        "scale",
        "chunk_size",
        "cumsum_scale",
        "use_gate_in_kernel",
        "lower_bound",
        "mini_batch",
    ],
)
def chunk_kda_fwd_fused(
    q: jax.Array,  # [H, B, T, K]
    k: jax.Array,  # [H, B, T, K]
    v: jax.Array,  # [H, B, T, V]
    g: jax.Array,  # [H, B, T, K]
    beta: jax.Array,  # [H, B, T]
    plan: _TilePlan,
    query_lens: jax.Array,  # [B, N]
    num_tokens: int,
    scale: float,
    A_log: jax.Array | None = None,  # [H]
    dt_bias: jax.Array | None = None,  # [H*K]
    initial_state: jax.Array | None = None,  # [B, N, H, K, V]
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    cumsum_scale: float = RCP_LN2,
    mini_batch: int | None = None,
) -> tuple[
        jax.Array,  # [H, B, T, V]
        jax.Array | None,  # [B, N, H, K, V]
]:
    """Run the whole forward pass in a single Pallas call.

    Both halves of the algorithm, minus the six per-chunk tensors that used to
    cross between them and minus the second launch.

    The channel axis is padded to the lane count *in HBM*, on the staged inputs,
    rather than on the intermediates -- they never materialize. Zero-padded channels stay inert
    throughout: `q`/`k`/`v` are zero there, so they contribute nothing to `Aqk`,
    to `w`, or to `kg^T @ v_new`, and the matching rows of the state stay zero
    from `h0`'s own zero padding onward. The padded gate channels do pick up a
    non-zero decay from the activation, which only decays state rows that are
    already zero. At the production ``head_dim`` of 128 all of this is a no-op.
    """
    H, B, _, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = plan.r_base.shape[-1]
    N = query_lens.shape[-1]
    act_dtype = q.dtype

    assert K <= 256, "current kernel does not support K > 256."
    if use_gate_in_kernel:
        assert A_log is not None, "A_log required when use_gate_in_kernel=True"
    if initial_state is not None:
        assert initial_state.shape[1] == N, (
            f"initial_state has N={initial_state.shape[1]}, expected {N}")

    align_major = pltpu.get_tpu_info().num_lanes
    K_PAD = int(align_up(K, align_major))
    V_PAD = int(align_up(V, align_major))

    # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
    if mini_batch is None:
        # Staged inputs (float32; beta is lane-padded), the six intermediates as
        # *live values* -- one copy each, no double buffering, which is where
        # the fusion pays for itself -- the solve's [BT, BT] temporaries, the
        # recurrent state (scratch plus the double-buffered h0/ht windows), and
        # the output staging buffer. Under-counting here over-subscribes VMEM
        # silently.
        staged = (3 * BT * K_PAD + BT * V_PAD + BT * align_major) * 4
        live = (4 * BT * K_PAD + BT * V_PAD + BT * BT) * 4
        solve = (8 * BT * BT + BT * (BT + V_PAD + K_PAD)) * 4
        state = 3 * K_PAD * V_PAD * 4
        staged_out = BT * V_PAD * 4
        per_head = staged + live + solve + state + staged_out
        MB = estimate_mini_batch(per_head, H, max_mb=16)
    else:
        MB = mini_batch
        assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

    # The staged tensors are float32, not the caller's bfloat16, because a tile
    # starts at an arbitrary token row and Mosaic will only slice a *packed*
    # memref at tile-aligned offsets and in tile-aligned sizes -- for bfloat16
    # that means multiples of 8, and neither `query_start_loc[s] + t *
    # chunk_size` nor a ragged tile's row count is. Unpacked 32-bit types have
    # no such rule. The kernel casts every input to float32 immediately anyway,
    # so this only moves that cast out of the tile loop; it costs one contiguous
    # pass over the activations and 2x the DMA bytes.
    def _stage(x, pad_to):
        x = x.astype(jnp.float32)
        pad = pad_to - x.shape[-1]
        if pad:
            x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, pad)))
        return x

    q_s = _stage(q, K_PAD)
    k_s = _stage(k, K_PAD)
    g_s = _stage(g, K_PAD)
    v_s = _stage(v, V_PAD)
    # A per-token scalar on a lane-minor axis cannot be DMA'd at an arbitrary
    # row offset, so beta is carried with the token axis second-minor like the
    # rest. It costs lane padding on a tensor 1/128th the size of the others.
    beta_s = beta[..., None].astype(jnp.float32)

    if use_gate_in_kernel:
        A_log_r = A_log.astype(jnp.float32).reshape(H, 1, 1, 1, 1)
        if dt_bias is not None:
            dt_bias_r = dt_bias.astype(jnp.float32).reshape(H, 1, 1, 1, K)
            if K_PAD > K:
                dt_bias_r = jnp.pad(
                    dt_bias_r, ((0, 0), (0, 0), (0, 0), (0, 0), (0, K_PAD - K)))
        else:
            dt_bias_r = jnp.zeros((H, 1, 1, 1, K_PAD), dtype=jnp.float32)
    else:
        A_log_r = jnp.zeros((H, 1, 1, 1, 1), dtype=jnp.float32)
        dt_bias_r = jnp.zeros((H, 1, 1, 1, K_PAD), dtype=jnp.float32)

    # h0: [B, N, H, K, V] -> [B, N, H, K_PAD, V_PAD]
    if initial_state is not None:
        h0 = initial_state
        if V_PAD > V:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_PAD - V)))
        if K_PAD > K:
            h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PAD - K), (0, 0)))
    else:
        h0 = None

    # Six scalar-prefetch arrays, so every index map takes six trailing refs.
    def _per_head_index_map(h, b, p, *_):
        del b, p
        return (h, 0, 0, 0, 0)

    def _state_index_map(h, b, p, _num_tiles, _r_base, _r_size, s_idx, *_):
        return (b, s_idx[b, p], h, 0, 0)

    def _per_head_spec(last_dim):
        return pl.BlockSpec(index_map=_per_head_index_map,
                            block_shape=(MB, 1, 1, 1, last_dim))

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    state_spec = pl.BlockSpec([1, 1, MB, K_PAD, V_PAD],
                              index_map=_state_index_map)

    # The output is written by DMA at arbitrary token rows, so it is a
    # whole-array HBM ref rather than a pipelined window, and it is passed in
    # zeroed and aliased: rows that no tile owns -- the token axis's padded tail
    # -- are never written, and must still come back as zeros.
    o_shape = jax.ShapeDtypeStruct([H, B, num_tokens, V_PAD], jnp.float32)
    o_zeros = jnp.zeros(o_shape.shape, o_shape.dtype)
    ht_shape = (jax.ShapeDtypeStruct([B, N, H, K_PAD, V_PAD], jnp.float32)
                if output_final_state else None)

    # bf16 inputs: DEFAULT precision is lossless (operands already have ~7-bit
    # mantissa); fp32 inputs: HIGHEST preserves full 23-bit mantissa fidelity.
    # State-update matmuls always use HIGHEST (recursive accumulation).
    output_precision = (jax.lax.Precision.DEFAULT if act_dtype == jnp.bfloat16
                        else jax.lax.Precision.HIGHEST)

    o_out, ht_out = pl.pallas_call(
        functools.partial(
            _chunk_kda_fwd_fused_kernel,
            chunk_size=BT,
            scale=scale,
            cumsum_scale=cumsum_scale,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            activation_dtype=act_dtype,
            use_initial_state=(h0 is not None),
            store_final_state=output_final_state,
            mini_batch=MB,
            output_precision=output_precision,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(H // MB, B, NT),
            in_specs=[
                hbm_spec,  # q
                hbm_spec,  # k
                hbm_spec,  # g
                hbm_spec,  # beta
                hbm_spec,  # v
                _per_head_spec(1),  # A_log
                _per_head_spec(K_PAD),  # dt_bias
                hbm_spec,  # the zeroed output, aliased
                state_spec if h0 is not None else None,  # h0
            ],
            out_specs=[
                hbm_spec,  # o
                state_spec if output_final_state else None,  # ht
            ],
            scratch_shapes=[
                pltpu.VMEM((MB, 1, BT, K_PAD), jnp.float32),  # q
                pltpu.VMEM((MB, 1, BT, K_PAD), jnp.float32),  # k
                pltpu.VMEM((MB, 1, BT, K_PAD), jnp.float32),  # g
                pltpu.VMEM((MB, 1, BT, 1), jnp.float32),  # beta
                pltpu.VMEM((MB, 1, BT, V_PAD), jnp.float32),  # v
                pltpu.VMEM((MB, K_PAD, V_PAD), jnp.float32),  # h
                pltpu.VMEM((MB, 1, BT, V_PAD), jnp.float32),  # o
                pltpu.SemaphoreType.DMA,  # staging in
                pltpu.SemaphoreType.DMA,  # output scatter
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            # The tile axis carries both the staging DMAs and the recurrence,
            # so it stays sequential.
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        out_shape=[o_shape, ht_shape],
        # The zeroed output rides input slot 7; `h0` sits after it so that index
        # holds whether or not there is an initial state (a None operand
        # contributes no leaf and would otherwise shift it). The 6 is the
        # scalar-prefetch prefix.
        input_output_aliases={6 + 7: 0},
        interpret=get_interpret(),
    )(plan.num_tiles, plan.r_base, plan.r_size, plan.s_idx, plan.is_first,
      plan.is_last, q_s, k_s, g_s, beta_s, v_s, A_log_r, dt_bias_r, o_zeros, h0)

    if V_PAD > V:
        o_out = o_out[..., :V]
    o_out = o_out.astype(act_dtype)

    if output_final_state and ht_out is not None:
        if V_PAD > V:
            ht_out = ht_out[..., :V]
        if K_PAD > K:
            ht_out = ht_out[..., :K, :]

        # Handle empty sequences: sequences with no tiles never execute kernel
        # code, so their final_state is uninitialized -- and the state window of
        # a segment past the last one with tiles may be flushed with a stale
        # buffer. Fill both from initial_state, or zeros.
        empty_mask = (query_lens == 0)  # [B, N]
        if initial_state is not None:
            fill_value = initial_state[:, :, :, :K, :V]
        else:
            fill_value = jnp.zeros((B, N, H, K, V), dtype=ht_out.dtype)
        ht_out = jnp.where(empty_mask[:, :, None, None, None], fill_value,
                           ht_out)
    else:
        ht_out = None

    return o_out, ht_out


def _run_chunk_kda(
    q: jax.Array,  # [H, B, T, K]
    k: jax.Array,  # [H, B, T, K]
    v: jax.Array,  # [H, B, T, V]
    g: jax.Array,  # [H, B, T, K]
    beta: jax.Array,  # [H, B, T]
    *,
    num_tokens: int,
    cu_seqlens: jax.Array,  # [B, N_CU]
    plan: _TilePlan,
    A_log: jax.Array | None = None,  # [H]
    dt_bias: jax.Array | None = None,  # [H*K]
    scale: float | None = None,
    initial_state: jax.Array | None = None,  # [B, N, H, K, V]
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
) -> tuple[
        jax.Array,  # [H, B, T, V]
        jax.Array | None,  # [B, N, H, K, V]
]:
    """Run the kernel on canonicalized inputs and the batch's tile plan."""
    output, final_state = chunk_kda_fwd_fused(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        plan=plan,
        query_lens=jnp.diff(cu_seqlens, axis=-1),
        num_tokens=num_tokens,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        cumsum_scale=RCP_LN2,
    )
    return output.astype(q.dtype), final_state


# ---------------------------------------------------------------------------
# Input canonicalization
# ---------------------------------------------------------------------------
class _PreparedKdaInputs(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    g: jax.Array
    beta: jax.Array
    initial_state: jax.Array | None
    cu_seqlens: jax.Array
    plan: _TilePlan


def check_inputs_support(
    q: jax.Array,
    v: jax.Array,
    *,
    chunk_size: int,
) -> None:
    """Checks whether the Pallas TPU backend supports the static inputs."""
    if q.dtype != jnp.bfloat16:
        raise NotImplementedError(
            "The KDA Pallas kernel supports bfloat16 inputs only.")
    heads, batch, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    if heads < 1 or batch < 1 or seq_len < 1:
        raise NotImplementedError(
            "`pallas_tpu` requires positive head, batch, and sequence "
            f"dimensions; got H={heads}, B={batch}, T={seq_len}.")
    if key_dim < 1 or value_dim < 1:
        raise NotImplementedError(
            "`pallas_tpu` requires positive key and value dimensions; got "
            f"K={key_dim}, V={value_dim}.")
    if key_dim > 256:
        raise NotImplementedError(
            "`pallas_tpu` currently supports key dimensions up to 256; got "
            f"K={key_dim}.")
    if chunk_size != 64:
        raise NotImplementedError(
            "`pallas_tpu` currently supports chunk_size=64.")


def _preprocess_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    initial_state: jax.Array | None,
    use_qk_l2norm_in_kernel: bool,
    segment_ids: jax.Array | None,
    chunk_size: int,
    N_max: int | None,
    start_seq: jax.Array | int | None,
) -> _PreparedKdaInputs:
    """Canonicalize inputs for the forward kernel and plan its tiles."""
    if N_max is None:
        if initial_state is None:
            raise ValueError(
                "`N_max` is required when `initial_state` is not given: the "
                "segment count is a static shape and cannot be inferred.")
        N_max = initial_state.shape[1]
    cu_seqlens = segment_ids_to_seqlens(segment_ids, max_segs=N_max)
    if start_seq is not None:
        # Segments before `start_seq` keep their slot in `initial_state` and in
        # the returned `final_state`, but contribute no tiles, so the kernel
        # neither reads nor advances their recurrent state.
        cu_seqlens = restrict_cu_seqlens(cu_seqlens, start_seq)
    plan = _plan_tiles(cu_seqlens,
                       num_tokens=q.shape[2],
                       chunk_size=chunk_size)

    if initial_state is not None and initial_state.shape[1] != N_max:
        raise ValueError(
            f"`initial_state` state count must match segment count {N_max}; "
            f"got {initial_state.shape[1]}.")

    if use_qk_l2norm_in_kernel:
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        q_prepared = (q_f32 * jax.lax.rsqrt(
            jnp.sum(q_f32 * q_f32, axis=-1, keepdims=True) + 1e-6)).astype(
                q.dtype)
        k_prepared = (k_f32 * jax.lax.rsqrt(
            jnp.sum(k_f32 * k_f32, axis=-1, keepdims=True) + 1e-6)).astype(
                k.dtype)
    else:
        q_prepared, k_prepared = q, k

    return _PreparedKdaInputs(
        q=q_prepared,
        k=k_prepared,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        plan=plan,
    )


def chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    A_log=None,
    dt_bias=None,
    scale: float | None = None,
    initial_state=None,
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids=None,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = False,
    N_max: int | None = None,
    start_seq=None,
):
    """Run the KDA forward kernel and return ``(output, final_state)``.

    Args:
      q, k: ``[H, B, T, K]`` queries and keys.
      v: ``[H, B, T, V]`` values.
      g: ``[H, B, T, K]`` per-channel gate. In log space unless
        ``use_gate_in_kernel`` is set, in which case it is raw and is activated
        in-kernel using ``A_log`` and ``dt_bias``.
      beta: ``[H, B, T]`` per-token delta-rule learning rate, already through
        the sigmoid.
      A_log: ``[H]`` gate parameter; required when ``use_gate_in_kernel``.
      dt_bias: ``[H*K]`` optional gate bias.
      scale: query scale, defaults to ``K ** -0.5``.
      initial_state: ``[B, N, H, K, V]`` recurrent state, one per request slot.
      output_final_state: whether to return the final state (else ``None``).
      use_gate_in_kernel: treat ``g`` as raw gate input.
      segment_ids: ``[B, T]`` 1-indexed varlen segment IDs, 0 for padding.
        Required -- every batch is ragged. A fixed-length batch is just one
        segment spanning the token axis.
      safe_gate: deprecated compatibility argument; ignored.
      lower_bound: optional sigmoid-gate lower bound.
      chunk_size: chunk length used by the kernel.
      use_qk_l2norm_in_kernel: normalize queries and keys before the kernel.
      N_max: static segment count. Defaults to ``initial_state``'s.
      start_seq: optional device scalar; restrict the kernel to segments at or
        after this index. Earlier segments get no tiles, so their rows of
        ``final_state`` come back exactly as they went in and their output rows
        are left zero -- which is how a caller hands the leading part of a batch
        to a different kernel without a host-side branch on batch composition.

    Returns:
      ``(output, final_state)`` where output is ``[H, B, T, V]`` and
      ``final_state`` is ``[B, N, H, K, V]`` or ``None``.

    """
    del safe_gate
    if scale is None:
        scale = q.shape[-1]**-0.5

    # Cross-argument shape agreement. jaxtyping used to enforce this via shared
    # dimension names ("H B T K" across q/k/g), but the annotations are now plain
    # comments per the repo convention, so check it explicitly here --
    # check_inputs_support only validates q and v.
    if not (q.shape == k.shape == g.shape):
        raise ValueError(
            f"q, k and g must share [H, B, T, K]; got q={q.shape} "
            f"k={k.shape} g={g.shape}")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(
            f"v must share [H, B, T] with q; got v={v.shape} q={q.shape}")
    if beta.shape != q.shape[:3]:
        raise ValueError(
            f"beta must be [H, B, T] = {q.shape[:3]}; got {beta.shape}")

    # Ragged batches only. The kernel used to carry a second, fixed-length mode
    # that skipped the alignment gather and took one state per batch item; no
    # caller used it, and keeping both meant every downstream branch had to
    # handle two state ranks and two sets of segment boundaries.
    if segment_ids is None:
        raise ValueError(
            "`segment_ids` is required: a fixed-length batch is expressed as a "
            "single segment spanning the token axis.")
    if initial_state is not None and initial_state.ndim != 5:
        raise ValueError(
            "`initial_state` must be [B, N, H, K, V], one recurrent state per "
            f"request slot; got {initial_state.ndim} dimensions.")

    # The kernel's intra-chunk decay relies on the activated gate being <= 0, so
    # that the cumulative gate decreases along the chunk and every causal
    # exponent g_r - g_t is non-positive (see the Aqk/L section of
    # pallas_tpu_fwd.py). Both activations satisfy this only for a negative
    # bound: -exp(A_log) * softplus(.) is always <= 0, while
    # lower_bound * sigmoid(.) needs lower_bound < 0. A positive bound would
    # make the decay grow along the chunk and silently return wrong values.
    if lower_bound is not None and lower_bound >= 0:
        raise ValueError(
            "lower_bound must be negative (it bounds a log-decay); got "
            f"{lower_bound}. A non-negative bound makes the gate increase "
            "along the chunk, which this kernel does not support.")

    check_inputs_support(q, v, chunk_size=chunk_size)

    prepared = _preprocess_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        segment_ids=segment_ids,
        chunk_size=chunk_size,
        N_max=N_max,
        start_seq=start_seq,
    )

    value, final_state = _run_chunk_kda(
        prepared.q,
        prepared.k,
        prepared.v,
        prepared.g,
        prepared.beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=prepared.initial_state,
        output_final_state=output_final_state,
        use_gate_in_kernel=use_gate_in_kernel,
        num_tokens=segment_ids.shape[1],
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        cu_seqlens=prepared.cu_seqlens,
        plan=prepared.plan,
    )
    return value, final_state
