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

"""Dynamic tiling heuristics and VMEM memory estimation for Fused Conv1D-GDN."""

import jax.numpy as jnp
from jax.experimental import pallas as pl

# T=32 is excluded: at n_v=8 it measured +59.6% (N=64) and +15.8% (N=256)
# slower than T=16 because the per-slot f32 state copy and vector temporaries
# spill (18.25 MiB at T=32 vs 9.55 MiB at T=16 in the post-RA LLO ledger).
DECODE_TILE_SIZES = (16, 8, 4, 2, 1)
MIXED_TILE_SIZES = (128, 64, 32, 16, 8, 4, 2, 1)

# Decode balance law. Each grid step pays a cost that does not shrink with the
# tile (pipeline issue and waits, and sub-tile vector padding below the
# sublane width), so the step total falls as num_seqs / T. The first tile's
# state fetch and the last tile's write-back are not overlapped and grow as
# T * n_v. Equating the two gives T^2 * n_v = K * num_seqs, with K the step
# cost measured in per-head state transfers. Device sweeps over N in [8, 256]
# and n_v in {8, 32, 64}, bf16 and f32 states, put K in [16, 28].
_DECODE_STEP_COST_IN_HEAD_TRANSFERS = 16


def decode_tile_target(num_seqs: int, n_v: int) -> int:
    """Largest candidate T with T * T * n_v <= K * num_seqs, else 1.

    num_seqs is the padded slot count, so this sizes for full occupancy; the
    runtime grid covers only the active sequences and the optimum is flat
    enough that lower occupancy stays within a few percent of its own best.
    """
    budget = _DECODE_STEP_COST_IN_HEAD_TRANSFERS * num_seqs
    return next((t for t in DECODE_TILE_SIZES if t * t * n_v <= budget), 1)


def align_to(x: int, alignment: int) -> int:
    """Aligns an integer upward to the nearest multiple of alignment."""
    return pl.cdiv(x, alignment) * alignment


def get_vmem_estimate_bytes(
    tile_b: int,
    chunk_sz: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    act_out_bytes: int,
    rec_state_bytes: int,
    num_lanes: int,
    conv_state_dim_size: int,
    is_decode: bool = False,
    carry_scratch_bytes: int = 4,
    window_size: int = 1,
) -> int:
    """Estimates total on-chip VMEM footprint in bytes for a GDN tile."""
    aligned_num_v_heads = align_to(n_v, num_lanes)
    aligned_d_k = align_to(d_k, num_lanes)
    aligned_d_v = align_to(d_v, num_lanes)
    dim_size = align_to(2 * n_kq * d_k + n_v * d_v, num_lanes)
    aligned_out_dim = align_to(n_v * d_v, num_lanes)

    # 1. Double-buffered input activation buffers (QKV, A, B).
    # wrapper.py casts qkv, b, and a to float32 before creating VMEM windows.
    qkv_bytes = 2 * (tile_b * chunk_sz * dim_size * carry_scratch_bytes)
    b_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * carry_scratch_bytes)
    a_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * carry_scratch_bytes)

    # 2. Double-buffered state cache buffers (convolution and recurrent states).
    conv_state_buffer_bytes = 2 * (
        tile_b
        * window_size
        * max(0, kernel_size - 1)
        * conv_state_dim_size
        * carry_scratch_bytes
    )
    recurrent_state_buffer_bytes = 2 * (
        tile_b * window_size * n_v * d_v * d_k * rec_state_bytes
    )

    # 3. Double-buffered output activation buffer.
    out_bytes = 2 * (tile_b * chunk_sz * aligned_out_dim * act_out_bytes)

    # 4. Temporary scratch buffers and working state memory.
    # Both decode and prefill maintain one f32 working recurrent state copy
    # concurrently with double-buffered recurrent state DMA buffers.
    scratch_recurrent_bytes = tile_b * n_v * d_v * d_k * carry_scratch_bytes
    scratch_conv_bytes = (
        int(not is_decode)
        * tile_b
        * max(0, kernel_size - 1)
        * dim_size
        * carry_scratch_bytes
    )

    # 5. Static weight cache references in on-chip memory.
    # Conv weight [K, dim] plus conv bias [dim] in f32; a_log and dt_bias.
    weights_bytes = ((kernel_size + 1) * dim_size * carry_scratch_bytes) + (
        aligned_num_v_heads * 8
    )

    # 6. Working memory for intra-chunk recurrence, per-head projections, and
    # per-sequence vector temporaries (8 arrays of shape [8, num_lanes] per slot).
    intermediate_bytes = (
        tile_b
        * (
            n_v * (5 * chunk_sz * chunk_sz + 3 * chunk_sz * (aligned_d_v + aligned_d_k))
            + 8 * 8 * num_lanes
        )
        * carry_scratch_bytes
    )

    return (
        qkv_bytes
        + b_bytes
        + a_bytes
        + conv_state_buffer_bytes
        + recurrent_state_buffer_bytes
        + out_bytes
        + scratch_conv_bytes
        + scratch_recurrent_bytes
        + weights_bytes
        + intermediate_bytes
    )


def calculate_decode_tile_size(
    batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    conv_state_dim_size: int,
    act_out_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_capacity_limit_bytes: int,
    kernel_size: int = 4,
    window_size: int = 1,
) -> int:
    """Derives optimal batch tile size for decode execution.

    Searches candidate batch tile sizes within maximum VMEM capacity limits.

    Args:
      batch_size: Total batch size of the active decode sequence.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key head dimension.
      d_v: Value head dimension.
      conv_state_dim_size: Feature dimension size for conv state.
      act_out_dtype: Data type for output activations.
      recurrent_state_dtype: Data type for recurrent state matrix.
      num_lanes: Number of lanes for TPU vector layout alignment.
      vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
      kernel_size: 1D convolution kernel window size.
      window_size: Number of state checkpoints per sequence in decode.

    Returns:
      Derived batch tile size fitting within VMEM capacity limits.
    """
    # Return a minimum valid tile size of 1 for empty or zero-length batches.
    if batch_size <= 0:
        return 1

    act_out_bytes = jnp.dtype(act_out_dtype).itemsize
    rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

    # Search descending power-of-2 tile candidates (up to 16) bounded by
    # batch size.
    decode_candidates = [c for c in DECODE_TILE_SIZES if c <= batch_size]
    decode_tile_size = decode_candidates[-1]

    for cand in decode_candidates:
        vmem_est = get_vmem_estimate_bytes(
            tile_b=cand,
            chunk_sz=1,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            act_out_bytes=act_out_bytes,
            rec_state_bytes=rec_state_bytes,
            num_lanes=num_lanes,
            conv_state_dim_size=conv_state_dim_size,
            is_decode=True,
            window_size=window_size,
        )
        if vmem_est <= vmem_capacity_limit_bytes:
            return cand

    return decode_tile_size


def calculate_mixed_tile_size(
    seq_len: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    conv_state_dim_size: int,
    act_out_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_capacity_limit_bytes: int,
    kernel_size: int = 4,
) -> int:
    """Derives optimal chunk tile size for prefill and mixed execution.

    Searches candidate tile sizes within maximum VMEM capacity limits.

    Args:
      seq_len: Upper bound on the longest prefill sequence, in tokens.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key head dimension.
      d_v: Value head dimension.
      conv_state_dim_size: Feature dimension size for conv state.
      act_out_dtype: Data type for output activations.
      recurrent_state_dtype: Data type for recurrent state matrix.
      num_lanes: Number of lanes for TPU vector layout alignment.
      vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
      kernel_size: 1D convolution kernel window size.

    Returns:
      Derived chunk tile size fitting within VMEM capacity limits.
    """
    # Return a minimum valid chunk size of 1 for empty or zero-length sequences.
    if seq_len <= 0:
        return 1

    act_out_bytes = jnp.dtype(act_out_dtype).itemsize
    rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

    # MIXED_TILE_SIZES stops at C=128, the largest chunk measured: C=128 beat
    # C=64 on every swept shape where both fit (n_v=8 and 32), and the VMEM
    # estimate rejects C=128 at n_v=64. C > 128 is unmeasured.
    prefill_candidates = [c for c in MIXED_TILE_SIZES if c <= seq_len]
    mixed_tile_size = prefill_candidates[-1]
    for candidate in prefill_candidates:
        vmem_est = get_vmem_estimate_bytes(
            tile_b=1,
            chunk_sz=candidate,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            act_out_bytes=act_out_bytes,
            rec_state_bytes=rec_state_bytes,
            num_lanes=num_lanes,
            conv_state_dim_size=conv_state_dim_size,
            is_decode=False,
        )
        if vmem_est <= vmem_capacity_limit_bytes:
            return candidate

    return mixed_tile_size


def get_tile_sizes(
    batch_size: int,
    num_seqs: int,
    padded_batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    conv_state_dim_size: int,
    act_out_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_limit_bytes: int,
    window_size: int = 1,
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None,
) -> tuple[int, int]:
    """Derives decode and mixed tile sizes within the caller's VMEM limit.

    vmem_limit_bytes is the limit the pallas_call compiles with, so sizing and
    compilation share one number. window_size scales the per-slot state
    buffers in decode (num_spec_tokens + 1).
    """
    vmem_capacity_limit_bytes = vmem_limit_bytes

    if decode_tile_size is None or decode_tile_size <= 0:
        vmem_fit = calculate_decode_tile_size(
            # BATCHED tiles over sequences; a tile never needs more slots than
            # there are sequences.
            batch_size=min(padded_batch_size, num_seqs),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            conv_state_dim_size=conv_state_dim_size,
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=recurrent_state_dtype,
            num_lanes=num_lanes,
            vmem_capacity_limit_bytes=vmem_capacity_limit_bytes,
            kernel_size=kernel_size,
            window_size=window_size,
        )
        decode_tile_size = min(decode_tile_target(num_seqs, n_v), vmem_fit)

    if mixed_tile_size is None or mixed_tile_size <= 0:
        vmem_fit = calculate_mixed_tile_size(
            # num_seqs counts padded request slots and cannot bound the longest
            # prefill; the token count can.
            seq_len=batch_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            conv_state_dim_size=conv_state_dim_size,
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=recurrent_state_dtype,
            num_lanes=num_lanes,
            vmem_capacity_limit_bytes=vmem_capacity_limit_bytes,
            kernel_size=kernel_size,
        )
        mixed_tile_size = vmem_fit

    # Guarantee strictly positive tile sizes (>= 1) for Pallas grid compilation.
    decode_tile_size = max(1, min(decode_tile_size, padded_batch_size))
    mixed_tile_size = max(1, min(mixed_tile_size, batch_size))
    return decode_tile_size, mixed_tile_size
