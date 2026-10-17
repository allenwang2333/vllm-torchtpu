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

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config

# Value heads the chunked (PER_SEQ) step computes per slab of the carried state.
CHUNK_HEAD_GROUP = 64
# Heads packed side by side along lanes, one MXU pass per group, in the
# blocked triangular inverse.
INV_HEADS_PER_PASS = 2


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
    return x / norm


def gated_rms_norm(
    x: jax.Array, z: jax.Array, weight: jax.Array, eps: float
) -> jax.Array:
    """vLLM's RMSNormGated with `norm_before_gate`, over the last axis:
    rms_norm(x) * weight * silu(z) in f32 on the 16-bit `x` and `z`, rounded
    back to `x`'s dtype. Shared by the kernel and the XLA path so both
    evaluate the same expression."""
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    out = x32 * jax.lax.rsqrt(variance + eps) * weight.astype(jnp.float32)
    return (out * jax.nn.silu(z.astype(jnp.float32))).astype(x.dtype)


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32
        case 2:
            return jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def _lane_gather(x: jax.Array, idx: jax.Array) -> jax.Array:
    """`jnp.take_along_axis(x, idx, axis=-1)` for 2D x: a lane permute."""
    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )
    return jax.lax.gather(
        x,
        idx[..., None],
        dnums,
        slice_sizes=(1, 1),
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _can_pair_heads(t_mat: jax.Array, b_mat: jax.Array) -> bool:
    """Whether `_forward_sub_paired` applies: two heads' columns fill the
    lanes, and a pair's rows fit one vreg."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    heads, rows, cols = b_mat.shape
    return (
        2 * cols == num_lanes
        and 2 * rows <= num_lanes
        and heads % 2 == 0
        and t_mat.dtype == b_mat.dtype == jnp.float32
    )


def _forward_sub_paired(t_mat: jax.Array, b_mat: jax.Array) -> jax.Array:
    """Unit lower-triangular forward substitution with two heads per vreg.

    Heads 2p and 2p + 1 sit side by side along lanes, so each multiply,
    sublane sum and subtract covers both. Row i's coefficients t[h, i, j]
    reach sublane j by a sublane broadcast of the pair's row i and one
    lane gather whose pattern is the same for every pair and row. Every
    element goes through the same operations on the same values as the
    one-head-per-vreg loop in `invert_triangular_matrix`: lanes are
    independent in the sublane sums.
    """
    heads, rows, cols = b_mat.shape
    lanes = 2 * cols
    pairs = b_mat.reshape(heads // 2, 2, rows, cols)
    b_pair = jnp.concatenate([pairs[:, 0], pairs[:, 1]], axis=-1)

    # [heads // 2, rows, lanes]: head 2p's row on lanes [0, rows), head
    # 2p + 1's on [rows, 2 * rows).
    t_pairs = t_mat.reshape(heads // 2, 2, rows, rows)
    pad = jnp.zeros((heads // 2, rows, lanes - 2 * rows), t_mat.dtype)
    t_rows = jnp.concatenate([t_pairs[:, 0], t_pairs[:, 1], pad], axis=-1)
    # Sublane s of the gather picks column j0 + s of its half's head.
    sublanes = pltpu.get_tpu_info().num_sublanes
    sub = jax.lax.broadcasted_iota(jnp.int32, (sublanes, lanes), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (sublanes, lanes), 1)
    col = jnp.where(lane < cols, sub, rows + sub)

    x_list = []
    for i in range(rows):
        b_i = b_pair[:, i, :]
        if i == 0:
            x_i = b_i
        else:
            stacked_x = jnp.stack(x_list, axis=1)
            # [heads // 2, i, lanes]: t[h, i, j] down sublane j.
            prev_t = []
            for j0 in range(0, i, sublanes):
                prev_t.append(
                    jnp.stack(
                        [
                            _lane_gather(
                                jnp.broadcast_to(row[i : i + 1], (sublanes, lanes)),
                                col + j0,
                            )
                            for row in t_rows
                        ]
                    )
                )
            prev_t = jnp.concatenate(prev_t, axis=1)[:, :i]
            prev_sum = jnp.sum(prev_t * stacked_x, axis=1)
            x_i = b_i - prev_sum
        x_list.append(x_i)
    x_pair = jnp.stack(x_list, axis=1)
    x = jnp.stack([x_pair[..., :cols], x_pair[..., cols:]], axis=1)
    return x.reshape(heads, rows, cols)


def pack_heads(x: jax.Array, g: int) -> jax.Array:
    """[heads, n, m] -> [heads // g, n, g * m]: head g * p + j on lanes
    [j * m, (j + 1) * m) of group p."""
    heads, n, m = x.shape
    grouped = x.reshape(heads // g, g, n, m)
    return jnp.concatenate([grouped[:, j] for j in range(g)], axis=-1)


def unpack_heads(x: jax.Array, g: int) -> jax.Array:
    """Inverse of `pack_heads`."""
    groups, n, width = x.shape
    m = width // g
    out = jnp.stack([x[..., j * m : (j + 1) * m] for j in range(g)], axis=1)
    return out.reshape(groups * g, n, m)


def _can_invert_blocked(l_mat: jax.Array) -> bool:
    """Whether `invert_unit_lower_blocked` applies to [heads, n, n] L: f32,
    a power-of-two n, and heads that pack into lane-dense groups of
    `INV_HEADS_PER_PASS`."""
    heads, rows, cols = l_mat.shape
    return (
        rows == cols
        and rows & (rows - 1) == 0
        and heads % INV_HEADS_PER_PASS == 0
        and (INV_HEADS_PER_PASS * cols) % pltpu.get_tpu_info().num_lanes == 0
        and l_mat.dtype == jnp.float32
    )


def split_bf16x3(x: jax.Array, n: int, c0: int, cols: int) -> jax.Array:
    """Columns [c0, c0 + cols) of each head's span of f32 x (`pack_heads`
    layout, head span n) -> three bf16-exact parts hi + mid + lo == x, each
    keeping the next 8 significant bits of the remainder, at lanes
    [0, cols), [cols, 2 * cols) and [2 * cols, 3 * cols) of the head's span.
    Needs 3 * cols <= n."""
    axis = x.ndim - 1
    width = x.shape[axis]
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, axis) % n
    x = jnp.where((lane >= c0) & (lane < c0 + cols), x, 0)

    def top8(v):
        bits = jax.lax.bitcast_convert_type(v, jnp.uint32) & jnp.uint32(0xFFFF0000)
        return jax.lax.bitcast_convert_type(bits, v.dtype)

    hi = top8(x)
    rem = x - hi
    mid = top8(rem)
    lo = rem - mid
    return sum(
        pltpu.roll(part, (i * cols - c0) % width, axis) if (i * cols - c0) % width else part
        for i, part in enumerate((hi, mid, lo))
    )


def column_selectors(width: int, n: int, cols: int) -> list[jax.Array]:
    """For each c < cols, the [width, width] 0/1 bf16 matrix that maps
    `split_bf16x3` parts to column c of every head, broadcast over the
    head's span of n lanes."""
    shape = (width, width)
    head0 = jax.lax.broadcasted_iota(jnp.int32, shape, 1) // n * n
    off = (jax.lax.broadcasted_iota(jnp.int32, shape, 0) - head0) % width
    col = jnp.where(off < 3 * cols, off % cols, -1)
    return [jnp.where(col == c, 1.0, 0.0).astype(jnp.bfloat16) for c in range(cols)]


def invert_unit_lower_blocked(l_pack: jax.Array, block_size: int = 16) -> jax.Array:
    """(I + L)^-1 for strictly lower-triangular L, two heads per vreg row.

    The arithmetic of `invert_triangular_matrix`: each block of `block_size`
    rows takes the earlier blocks' contribution as one MXU matmul, then is
    solved by forward substitution in f32 on the VPU. The substitution runs
    column by column: once row s is final, t[r, s] * x_s is subtracted from
    every row r below it, so no step needs a sublane reduction. Row s's
    coefficients t[r, s] reach their head's lanes by one exact MXU matmul
    of the block's `split_bf16x3` parts against a 0/1 selector, shared by
    every group.

    `l_pack` and the result are [groups, n, g * n] in the `pack_heads`
    layout. The matmul's right operand is expanded to the block diagonal of
    its group's heads, so one matmul serves the whole group.
    """
    groups, n, width = l_pack.shape
    g = width // n
    block_size = min(block_size, n)
    sublanes = pltpu.get_tpu_info().num_sublanes
    shape = (1, n, width)
    lane_head = jax.lax.broadcasted_iota(jnp.int32, shape, 2) // n
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    col = jax.lax.broadcasted_iota(jnp.int32, shape, 2) % n
    identity = jnp.where(row == col, 1.0, 0.0).astype(l_pack.dtype)
    selectors = column_selectors(width, n, block_size)
    x_blocks = []
    for start in range(0, n, block_size):
        l_rows = l_pack[:, start : start + block_size]
        rhs = jnp.broadcast_to(
            identity[:, start : start + block_size], (groups, block_size, width)
        )
        if start:
            solved = jnp.concat(
                x_blocks + [jnp.zeros((groups, n - start, width), l_pack.dtype)],
                axis=1,
            )
            solved_diag = jnp.concat(
                [jnp.where(lane_head == j, solved, 0) for j in range(g)], axis=1
            )
            prev_sum = jax.lax.dot(
                jnp.where(col[:, start : start + block_size] < start, l_rows, 0),
                solved_diag,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            rhs = rhs - prev_sum
        l_parts = split_bf16x3(l_rows, n, start, block_size).astype(jnp.bfloat16)
        for s in range(block_size - 1):
            # Only rows from the vreg holding row s + 1 down change.
            r0 = (s + 1) // sublanes * sublanes
            rows = block_size - r0
            # Exact: one product per output is nonzero, and the three
            # parts of an element sum back to it.
            coef = jax.lax.dot(
                l_parts[:, r0:].reshape(groups * rows, width),
                selectors[s],
                preferred_element_type=jnp.float32,
            ).reshape(groups, rows, width)
            update = rhs[:, r0:] - coef * rhs[:, s : s + 1]
            rhs = jnp.concat([rhs[:, :r0], update], axis=1) if r0 else update
        x_blocks.append(rhs)
    return jnp.concat(x_blocks, axis=1)


def solve_scratch(cfg: config.GDNConfig) -> Any:
    """VMEM for `invert_unit_lower_blocked_rows` in the chunked step, or None
    where the step does not take it."""
    if cfg.mode != config.GDNMode.PER_SEQ or cfg.is_kda or cfg.use_recurrent:
        return None
    groups = min(CHUNK_HEAD_GROUP, cfg.num_v_heads) // INV_HEADS_PER_PASS
    rows = max(groups, 1) * (cfg.chunk_size | 1)
    return pltpu.VMEM((rows, INV_HEADS_PER_PASS * cfg.chunk_size), jnp.float32)


def swap_major(x: jax.Array, scratch_ref: jax.Ref) -> jax.Array:
    """[a, b, lanes] -> [b, a, lanes] through VMEM: each x[i] is stored
    whole, `b | 1` rows apart, and each output row comes back by one strided
    load. The odd stride is what keeps that load a single load."""
    a, b, _ = x.shape
    stride = b | 1
    for i in range(a):
        scratch_ref[pl.ds(i * stride, b), :] = x[i]
    return jnp.stack(
        [scratch_ref[pl.ds(j, a, stride=stride), :] for j in range(b)]
    )


def invert_unit_lower_blocked_rows(
    l_pack: jax.Array, scratch_ref: jax.Ref, block_size: int = 16
) -> jax.Array:
    """`invert_unit_lower_blocked` with each block solved row-major.

    Within a block every row r of every group is its own vregs, [groups,
    g * n] with the groups down the sublanes, so substituting row s touches
    only the rows below it and needs no sublane broadcast of row s. Rows at
    or above s, which the packed loop updates by an exact zero, are left
    alone. Each element of the result goes through the same operations in
    the same order as `invert_unit_lower_blocked`. Blocks enter and leave
    the row-major layout through `scratch_ref` (see `swap_major`).
    """
    groups, n, width = l_pack.shape
    g = width // n
    block_size = min(block_size, n)
    shape = (1, n, width)
    lane_head = jax.lax.broadcasted_iota(jnp.int32, shape, 2) // n
    col = jax.lax.broadcasted_iota(jnp.int32, shape, 2) % n
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    identity = jnp.where(row == col, 1.0, 0.0).astype(l_pack.dtype)
    selectors = column_selectors(width, n, block_size)
    # [n, groups, width]
    l_by_row = swap_major(l_pack, scratch_ref)
    x_blocks = []
    for start in range(0, n, block_size):
        rhs = jnp.broadcast_to(
            identity[:, start : start + block_size], (groups, block_size, width)
        )
        if start:
            l_rows = l_pack[:, start : start + block_size]
            solved = jnp.concat(
                x_blocks + [jnp.zeros((groups, n - start, width), l_pack.dtype)],
                axis=1,
            )
            solved_diag = jnp.concat(
                [jnp.where(lane_head == j, solved, 0) for j in range(g)], axis=1
            )
            prev_sum = jax.lax.dot(
                jnp.where(col[:, start : start + block_size] < start, l_rows, 0),
                solved_diag,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            rhs = rhs - prev_sum
        x = list(swap_major(rhs, scratch_ref))
        l_parts = split_bf16x3(
            l_by_row[start : start + block_size], n, start, block_size
        ).astype(jnp.bfloat16)
        for s in range(block_size - 1):
            rows = block_size - 1 - s
            # Exact: one product per output is nonzero, and the three
            # parts of an element sum back to it.
            coef = jax.lax.dot(
                l_parts[s + 1 :].reshape(rows * groups, width),
                selectors[s],
                preferred_element_type=jnp.float32,
            ).reshape(rows, groups, width)
            for r in range(s + 1, block_size):
                x[r] = x[r] - coef[r - s - 1] * x[s]
        x_blocks.append(swap_major(jnp.stack(x), scratch_ref))
    return jnp.concat(x_blocks, axis=1)


def packed_block_diag_dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Per-head `lhs @ rhs` with `lhs` in the `pack_heads` layout:
    [heads // g, n, g * k] x [heads, k, m] -> [heads, n, m], one matmul per
    group against the block diagonal of its heads' right-hand sides."""
    groups, n, width = lhs.shape
    heads, k, m = rhs.shape
    g = heads // groups
    rhs = rhs.reshape(groups, g, k, m)
    zeros = jnp.zeros((groups, k, m), rhs.dtype)
    rhs_diag = jnp.concat(
        [
            jnp.concat([rhs[:, j] if i == j else zeros for i in range(g)], axis=-1)
            for j in range(g)
        ],
        axis=1,
    )
    out = jax.lax.dot(
        lhs,
        rhs_diag,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    return unpack_heads(out, g)


def pack_columns(x: jax.Array, g: int) -> jax.Array:
    """[heads, n, 1] -> [heads // g, n, g * n]: each head's column broadcast
    over its lanes of the `pack_heads` layout."""
    heads, n, _ = x.shape
    grouped = x.reshape(heads // g, g, n, 1)
    lane_head = jax.lax.broadcasted_iota(jnp.int32, (1, n, g * n), 2) // n
    out = grouped[:, g - 1]
    for j in range(g - 2, -1, -1):
        out = jnp.where(lane_head == j, grouped[:, j], out)
    return out


def packed_diag_blocks_dot_t(a: jax.Array, b: jax.Array, g: int) -> jax.Array:
    """Per-head `a @ b^T` in the `pack_heads` layout: [heads, n, d] x
    [heads, n, d] -> [heads // g, n, g * n]. A group's heads are stacked
    along M and N of one matmul and its diagonal blocks kept."""
    heads, n, d = a.shape
    full = jax.lax.dot(
        a.reshape(heads // g, g * n, d),
        b.reshape(heads // g, g * n, d),
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    lane_head = jax.lax.broadcasted_iota(jnp.int32, (1, n, g * n), 2) // n
    out = full[:, (g - 1) * n :]
    for j in range(g - 2, -1, -1):
        out = jnp.where(lane_head == j, full[:, j * n : (j + 1) * n], out)
    return out


def _can_pack_chunk_maps(heads: int, cfg: config.GDNConfig, head_slice: slice) -> bool:
    """Whether a slab's [chunk, chunk] maps can be held `INV_HEADS_PER_PASS`
    heads per vreg row: the blocked inverse applies and every group sits
    within one kq head."""
    g = INV_HEADS_PER_PASS
    chunk = cfg.chunk_size
    return (
        _can_invert_blocked(
            jax.ShapeDtypeStruct((heads, chunk, chunk), cfg.dtypes.compute)
        )
        and cfg.v_per_kq_head % g == 0
        and head_slice.start % g == 0
    )


# NOTE: Fork of recurrent_scan_v2.py but applied various optimizations.
def invert_triangular_matrix(t: jax.Array, block_size=16) -> jax.Array:
    """Compute invert matrix of a given triauglar matrix."""

    # NOTE: if chunk_size=1, compiler will perform DCE.
    out_dtype = t.dtype
    chunk = t.shape[-1]
    block_size = min(block_size, chunk)
    num_blocks = chunk // block_size

    def local_forward_sub(t_mat, b_mat):
        if _can_pair_heads(t_mat, b_mat):
            return _forward_sub_paired(t_mat, b_mat)
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_t = t_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_t[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    iota_r = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
    iota_c = jax.lax.broadcasted_iota(jnp.int32, t.shape, 2)
    identity_mask = jnp.where(iota_r == iota_c, 1.0, 0.0)
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = identity_mask[:, start:end, :]

        if i == 0:
            target_b = e_block
        else:
            interaction_t = t[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jax.lax.dot(
                interaction_t,
                solved_x,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            target_b = e_block - prev_sum

        # NOTE: Utilize fp32 to minimize cost of sublane rolling.
        local_t = t[:, start:end, start:end].astype(jnp.float32)
        x_block = local_forward_sub(local_t, target_b)
        x_blocks.append(x_block.astype(out_dtype))

    return jnp.concatenate(x_blocks, axis=1)


def fused_transpose_broadcast(x: jax.Array, src_dim: int, dst_dim: int) -> jax.Array:
    """Perform 1D transpose where results are broadcasted along src_dim."""
    assert x.shape[dst_dim] == 1

    dtype = x.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(x.shape)
    mask_size = mask_shape[src_dim]
    mask_shape[dst_dim] = mask_size
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


def heads_to_columns(x: jax.Array, h0: int, num_heads: int) -> jax.Array:
    """`fused_transpose_broadcast(x, 2, 0)[h0:h0 + num_heads]`, computing only
    those rows: [1, rows, lanes] -> [num_heads, rows, 1]."""
    mask_dtype = get_mask_dtype(x.dtype)
    mask_shape = (num_heads, x.shape[1], x.shape[2])
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, 2)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, 0) + h0
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=2, keepdims=True, dtype=x.dtype)


def chunked_gdn_heads(
    q_kq: jax.Array,  # [kq heads covering the slab, chunk, kq_head_dim]
    k_kq: jax.Array,  # [kq heads covering the slab, chunk, kq_head_dim]
    v_large: jax.Array,  # [heads, chunk, v_head_dim]
    g_cum_sum_log: jax.Array,  # [heads, chunk, 1]
    beta_large: jax.Array,  # [heads, chunk, 1]
    state_prev: jax.Array,  # [heads, kq_head_dim, v_head_dim]
    head_slice: slice,
    cfg: config.GDNConfig,
    solve_scratch_ref: jax.Ref | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Chunked GDN over one slab of value heads.

    `head_slice` picks the slab's value heads out of `q_kq`/`k_kq` repeated
    `v_per_kq_head` times. Heads are independent, so every head goes through
    the same operations as in a whole-tile step.
    """

    def repeat(x: jax.Array) -> jax.Array:
        # NOTE: Repeat along non lane/sublane dim is free.
        return jnp.repeat(x, cfg.v_per_kq_head, axis=0)[head_slice]

    q_repeat = repeat(q_kq)
    k_repeat = repeat(k_kq)

    # [heads, chunk, 1]
    gating_forward = jnp.exp(g_cum_sum_log)
    # [heads, 1, 1]
    gating_last = gating_forward[:, -1:]
    # [heads, 1, chunk]
    g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log, src_dim=1, dst_dim=2)
    # [heads, chunk, kq_head_dim]
    k_beta_repeat = k_repeat * beta_large

    # The [chunk, chunk] maps hold two heads per vreg row when packed:
    # [heads // g, chunk, g * chunk] in the `pack_heads` layout.
    g = INV_HEADS_PER_PASS
    packed = _can_pack_chunk_maps(v_large.shape[0], cfg, head_slice)
    if packed:
        # [heads, chunk, 1]: -(g - g_last), as the unpacked map's last column.
        gating_backward = jnp.exp(-(g_cum_sum_log - g_cum_sum_log[:, -1:]))
        g_cum_sum_diff_log = pack_columns(g_cum_sum_log, g) - pack_heads(
            g_cum_sum_log_t, g
        )
        beta_k_k_t = packed_diag_blocks_dot_t(k_beta_repeat, k_repeat, g)
    else:
        # [heads, chunk, chunk]
        g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
        # [heads, chunk, 1]
        gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
        # [heads, chunk, chunk]
        beta_k_k_t = jax.lax.dot(
            k_beta_repeat,
            k_repeat,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
    beta_k_k_t = beta_k_k_t.astype(cfg.dtypes.compute)
    gating_map = jnp.exp(g_cum_sum_diff_log)

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota_r = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
    iota_c = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
    if packed:
        iota_c = iota_c % cfg.chunk_size
    identity_mask = iota_r == iota_c
    strictly_lower_mask = iota_r > iota_c
    lower_mask = iota_r >= iota_c
    gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)
    gating_beta_k_k_t = gating_map_masked * beta_k_k_t

    if packed:
        if solve_scratch_ref is None:
            t_inv = invert_unit_lower_blocked(gating_beta_k_k_t)
        else:
            t_inv = invert_unit_lower_blocked_rows(gating_beta_k_k_t, solve_scratch_ref)
    else:
        t = jnp.where(identity_mask, 1, gating_beta_k_k_t)
        t_inv = invert_triangular_matrix(t)

    # [heads, chunk, v_head_dim]
    v_beta_large = v_large * beta_large
    # [heads, chunk, kv_head_dim]
    k_beta_gating = k_beta_repeat * gating_forward
    # NOTE: If v_head_dim < mxu size, concatenating them will help increase mxu
    # utilization. Also, if v_head_dim is multiple of lane size, concat / split
    # along lane dim is free - making this optimization strictly beneficial.
    # [heads, chunk, v_head_dim + kq_head_dim]
    merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
    if packed:
        merged_uw = packed_block_diag_dot(t_inv, merged_v_k)
    else:
        merged_uw = jax.lax.dot(
            t_inv,
            merged_v_k,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
    merged_uw = merged_uw.astype(cfg.dtypes.compute)

    # [heads, chunk, v_head_dim]
    u, w = jnp.split(merged_uw, [cfg.v_head_dim], axis=-1)

    # [heads, chunk, kq_head_dim]
    q_large_gating = q_repeat * gating_forward
    # NOTE: Concatenate lhs with same rhs to leverage weight
    # stationary architecture.
    # [heads, 2 * chunk, kq_head_dim]
    merged_w_q = jnp.concat([w, q_large_gating], axis=1)
    # [heads, 2 * chunk, v_head_dim]
    merged_ws_out_updated = jax.lax.dot(
        merged_w_q,
        state_prev,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    # NOTE: Splitting along non sublane/lane dim is free.
    ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
    ws = ws.astype(cfg.dtypes.compute)

    # [heads, chunk, v_head_dim]
    u_ws = u - ws

    # [heads, chunk, kq_head_dim]
    k_repeat_gating = k_repeat * gating_backward

    # [heads, kq_head_dim, v_head_dim]
    state_new = jax.lax.dot(
        k_repeat_gating,
        u_ws,
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    # [heads, kq_head_dim, v_head_dim]
    state_updated = state_prev * gating_last
    state = state_updated + state_new

    # [kq heads, chunk, chunk], or [kq heads, chunk, g * chunk] holding g
    # copies side by side when packed.
    out_qk = jax.lax.dot(
        q_kq,
        jnp.concat([k_kq] * g, axis=1) if packed else k_kq,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    # NOTE: must perform repeat after matmul to reduce required compute.
    if packed:
        out_qk = jnp.repeat(out_qk, cfg.v_per_kq_head // g, axis=0)[
            head_slice.start // g : head_slice.stop // g
        ]
    else:
        out_qk = repeat(out_qk)
    out_qk *= gating_map
    out_qk = jnp.where(lower_mask, out_qk, 0)

    # [heads, chunk, v_head_dim]
    if packed:
        out_new = packed_block_diag_dot(out_qk, u_ws)
    else:
        out_new = jax.lax.dot(
            out_qk,
            u_ws,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
    out = out_updated + out_new

    return out, state


def chunked_gdn(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    load_state: Callable[[int, int, int], jax.Array],
    store_state: Callable[[int, int, jax.Array], None],
    store_out: Callable[[int, int, jax.Array], None],
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
    solve_scratch_ref: jax.Ref | None = None,
) -> None:
    """Perform chunked GDN over input [seq, num_heads, chunk, head_dim].

    Walks the value heads in slabs of `CHUNK_HEAD_GROUP`, so no intermediate
    spans every head. `load_state(idx, h0, n)` returns heads [h0, h0 + n) of
    sequence `idx`'s carried state as [n, kq_head_dim, v_head_dim] fp32,
    `store_state(idx, h0, values)` writes their updated state back where it
    lives and `store_out(idx, h0, out)` writes their [n, chunk, v_head_dim]
    output. The chunked path only produces the final state, so it is never
    taken when more than one checkpoint per sequence is required.
    `solve_scratch_ref`, from `solve_scratch`, lets the triangular inverse
    solve its blocks row-major.
    """

    assert not cfg.use_recurrent
    group = min(CHUNK_HEAD_GROUP, cfg.num_v_heads)
    vpk = cfg.v_per_kq_head
    assert cfg.num_v_heads % group == 0
    assert group % vpk == 0 or vpk % group == 0

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1), 2
    )
    mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

    # [seqs, num_kq_heads, chunk, kq_head_dim]
    q_large = jnp.where(mask, q_large.astype(cfg.dtypes.compute), 0)
    k_large = jnp.where(mask, k_large.astype(cfg.dtypes.compute), 0)
    # [seqs, num_v_heads, chunk, v_head_dim]
    v_large = jnp.where(mask, v_large.astype(cfg.dtypes.compute), 0)

    b_large = b_large.astype(cfg.dtypes.compute)
    a_large = a_large.astype(cfg.dtypes.compute)

    a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)

    # NOTE: Any element-wise computations should occur before repeat.
    q_large = l2_norm(q_large)
    q_scale = cfg.kq_head_dim**-0.5
    q_large *= q_scale
    k_large = l2_norm(k_large)

    # [seqs, 1, chunk, num_v_heads]
    beta = jax.nn.sigmoid(b_large)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_large + dt_bias)

    beta = jnp.where(mask, beta, 0)
    # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
    # be masked to 1 since it signifies that strength of state from previous row
    # will be 1 (i.e., no decay) if current row is invalid.
    gating_log = jnp.where(mask, gating_log, 0)

    for idx in range(cfg.seq_tile_size):
        # Compute cumulative sum of decay.
        # [1, 1, num_v_heads]
        g_cum_sum_list = [gating_log[idx, :, :1]]
        for row in range(1, cfg.chunk_size):
            g_cum_sum_list.append(
                g_cum_sum_list[-1] + gating_log[idx, :, row : row + 1]
            )
        # [1, chunk, num_v_heads]
        g_cum_sum_log = jnp.concat(g_cum_sum_list, axis=1)

        for h0 in range(0, cfg.num_v_heads, group):
            kq0 = h0 // vpk
            kq1 = pl.cdiv(h0 + group, vpk)
            out, state = chunked_gdn_heads(
                q_large[idx, kq0:kq1],
                k_large[idx, kq0:kq1],
                v_large[idx, h0 : h0 + group],
                # [heads, chunk, 1]
                heads_to_columns(g_cum_sum_log, h0, group),
                heads_to_columns(beta[idx], h0, group),
                load_state(idx, h0, group),
                slice(h0 - kq0 * vpk, h0 - kq0 * vpk + group),
                cfg,
                solve_scratch_ref,
            )
            store_state(idx, h0, state)
            store_out(idx, h0, out)


def recurrent_gdn_per_seq(
    q_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact_t: jax.Array,  # [num_kq_heads, chunk, kq_head_dim, 1]
    v_compact: jax.Array,  # [num_v_heads, chunk, 1, v_head_dim]
    gating_log: jax.Array,  # [num_v_heads, chunk, 1, 1]
    beta: jax.Array,  # [num_v_heads, chunk, 1, 1]
    state: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform recurrent GDN over input [num_heads, chunk, 1, head_dim].

    The returned state stacks the post-token state of the last
    `cfgs.window_size` window positions ([window_size, num_v_heads,
    kq_head_dim, v_head_dim]); with `window_size == 1` that is just the final
    state.
    """

    out_list = []
    state_list = []
    for c_idx in range(cfgs.chunk_size):
        # [num_v_heads, 1, kq_head_dim]
        q_curr = q_compact[:, c_idx]
        q_curr = jnp.repeat(q_curr, cfgs.v_per_kq_head, axis=0)
        k_curr = k_compact[:, c_idx]
        k_curr = jnp.repeat(k_curr, cfgs.v_per_kq_head, axis=0)

        # [num_v_heads, 1, v_head_dim]
        v_curr = v_compact[:, c_idx]

        # [num_v_heads, kq_head_dim, 1]
        k_curr_t = k_compact_t[:, c_idx]
        k_curr_t = jnp.repeat(k_curr_t, cfgs.v_per_kq_head, axis=0)

        # [num_v_heads, 1, 1]
        beta_curr = beta[:, c_idx]
        gating_curr = gating_log[:, c_idx]

        # [num_v_heads, kq_head_dim, v_head_dim]
        state_updated = state * gating_curr

        # [num_v_heads, 1, v_head_dim]
        v_updated = jax.lax.dot(
            k_curr,
            state_updated,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)

        # [num_v_heads, 1, v_head_dim]
        v_diff = v_curr - v_updated
        v_new = beta_curr * v_diff

        # [num_v_heads, kq_head_dim, v_head_dim]
        # NOTE: Multiplication with k_curr_t needs to be deferred as much as
        # possible as it expands the dimension size by kq_head_dim.
        state_new = k_curr_t * v_new
        # [num_v_heads, kq_head_dim, v_head_dim]
        state = state_updated + state_new

        # [num_v_heads, 1, v_head_dim]
        out = jax.lax.dot(
            q_curr,
            state,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)

        out_list.append(out[:, 0, :])
        # NOTE: Rows past real_sizes are masked out by the caller, so the state
        # stops changing there and trailing checkpoints simply repeat the state
        # of the last real token. Checkpoints of the positions dropped here are
        # dead code the compiler eliminates.
        if c_idx >= cfgs.chunk_size - cfgs.window_size:
            state_list.append(state)

    return jnp.stack(out_list, axis=0), jnp.stack(state_list, axis=0)


def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform recurrent GDN over input [seq, num_heads, chunk, 1, head_dim].

    The returned state has one checkpoint per window position: [seq,
    window_size, num_v_heads, kq_head_dim, v_head_dim]. Positions >=
    real_sizes repeat the last valid state; they are never written back to
    HBM.
    """

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2
    )
    mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

    # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
    q_compact = jnp.where(mask, q_compact.astype(cfg.dtypes.compute), 0)
    k_compact = jnp.where(mask, k_compact.astype(cfg.dtypes.compute), 0)
    # [seqs, num_v_heads, chunk, 1, v_head_dim]
    v_compact = jnp.where(mask, v_compact.astype(cfg.dtypes.compute), 0)

    b_compact = b_compact.astype(cfg.dtypes.compute)
    a_compact = a_compact.astype(cfg.dtypes.compute)

    a_log = a_log.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)

    # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
    q_compact = l2_norm(q_compact)
    q_scale = cfg.kq_head_dim**-0.5
    q_compact *= q_scale
    k_compact = l2_norm(k_compact)
    k_compact_t = fused_transpose_broadcast(k_compact, src_dim=4, dst_dim=3)

    beta = jax.nn.sigmoid(b_compact)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_compact + dt_bias)

    beta = jnp.where(mask, beta, 0)
    # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
    # be masked to 1 since it signifies that strength of state from previous row
    # will be 1 (i.e., no decay) if current row is invalid.
    gating_log = jnp.where(mask, gating_log, 0)
    gating_log = jnp.exp(gating_log)

    beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
    beta = beta[:, : cfg.num_v_heads]
    gating_log = fused_transpose_broadcast(gating_log, src_dim=4, dst_dim=1)
    gating_log = gating_log[:, : cfg.num_v_heads]

    out_list = []
    new_state_list = []

    for idx in range(cfg.seq_tile_size):
        out, state = recurrent_gdn_per_seq(
            q_compact[idx],
            k_compact[idx],
            k_compact_t[idx],
            v_compact[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfg,
        )
        out_list.append(out)
        new_state_list.append(state)

    out = jnp.stack(out_list, axis=0)
    new_recurrent_state = jnp.stack(new_state_list, axis=0)

    return out, new_recurrent_state
