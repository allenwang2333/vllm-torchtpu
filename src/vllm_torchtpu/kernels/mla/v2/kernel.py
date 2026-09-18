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
"""TPU-Friendly MLA Ragged Paged Attention kernel."""

import functools
import math
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.mla.v2 import kv_utils
from vllm_torchtpu.kernels.mla.v2.transpose import xpose_pipeline
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_XPOSE_N_TILE_SIZE = envs.MLA_XPOSE_N_TILE_SIZE


def unsigned_mod(a, b):
    exponent = int(math.log2(b))
    if b == int(math.pow(2, exponent)):
        # Use bitmask instead of modulo for efficiency.
        return a & (b - 1)
    return a % b


def unsigned_cdiv(a, b):
    exponent = int(math.log2(b))
    if b == int(math.pow(2, exponent)):
        # Use bit shift instead of division for efficiency.
        return (a + b - 1) >> exponent
    return (a + b - 1) // b


def unsigned_floor_div(a, b):
    exponent = int(math.log2(b))
    if b == int(math.pow(2, exponent)):
        # Use bit shift instead of division for efficiency.
        return a >> exponent
    return a // b


def unsigned_align_to(a, b):
    exponent = int(math.log2(b))
    if b == int(math.pow(2, exponent)):
        # Use bitmask instead of division and multiply for efficiency.
        return (a + b - 1) & (-int(b))

    return unsigned_cdiv(a, b) * b


# Need to explicitly multiply with 128 to avoid TPU compile error. (async_copy)
def align_to(a, b):
    return ((a + b - 1) // b) * b


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
    kv_packing: int | None = None,
    transpose_kv_cache: bool = False,
):
    if transpose_kv_cache:
        assert page_size % 128 == 0
        return (
            total_num_pages,
            kv_dim,
            page_size,
        )

    if kv_packing is None:
        kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        unsigned_align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        unsigned_align_to(kv_dim, 128),
    )


class MlaCase(Enum):
    """Represents the different cases for MLA.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

    DECODE = 0
    PREFILL = 1
    MIXED = 2
    BATCHED_DECODE = 3

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
            MlaCase.BATCHED_DECODE: "bd",
        }[self]


# Expect to run this validation during compile time.
def static_validate_inputs(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim] if not transpose_kv_cache else [total_num_pages, lkv_dim, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_blocks: tuple[int, int, int] | None = None,
    num_queries_per_blocks: tuple[int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    decode_batch_size: int = 1,
    mixed_q_split: int = 1,
    transpose_kv_cache: bool = False,
    ql_nope_is_token_major: bool = False,
    # Debug params.
    debug_mode: bool = False,
):
    """Validate inputs to the MLA RPA kernel statically."""
    if len(ql_nope.shape) != 3:
        raise ValueError(f"Expected 3D array for {ql_nope.shape=}")
    if len(q_pe.shape) != 3:
        raise ValueError(f"Expected 3D array for {q_pe.shape=}")
    if len(new_kv_c.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_kv_c.shape=}")
    if len(new_k_pe.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_k_pe.shape=}")

    q_heads_dim = 1 if ql_nope_is_token_major else 0
    q_tokens_dim = 0 if ql_nope_is_token_major else 1
    if ql_nope.shape[q_heads_dim] != q_pe.shape[1]:
        raise ValueError(
            f"Expected ql_nope num_heads {ql_nope.shape[q_heads_dim]=} to equal q_pe num_heads {q_pe.shape[1]=}"
        )
    if ql_nope.shape[q_tokens_dim] != q_pe.shape[0]:
        raise ValueError(
            f"Expected ql_nope num_tokens {ql_nope.shape[q_tokens_dim]=} to equal q_pe num_tokens {q_pe.shape[0]=}"
        )
    if ql_nope.shape[q_tokens_dim] != new_kv_c.shape[0]:
        raise ValueError(
            f"Expected {ql_nope.shape[q_tokens_dim]=} to be equal to {new_kv_c.shape[0]=}"
        )
    if new_kv_c.shape[0] != new_k_pe.shape[0]:
        raise ValueError(
            f"Expected {new_kv_c.shape[0]=} to be equal to {new_k_pe.shape[0]=}"
        )
    if ql_nope.shape[2] != new_kv_c.shape[1]:
        raise ValueError(
            f"Expected {ql_nope.shape[2]=} to be equal to {new_kv_c.shape[1]=}"
        )
    if q_pe.shape[2] != new_k_pe.shape[1]:
        raise ValueError(
            f"Expected {q_pe.shape[2]=} to be equal to {new_k_pe.shape[1]=}")

    actual_lkv_dim = ql_nope.shape[-1]
    actual_r_dim = q_pe.shape[-1]
    lkv_dim = unsigned_align_to(actual_lkv_dim, 128)
    r_dim = unsigned_align_to(actual_r_dim, 128)

    if not transpose_kv_cache:
        (
            _,
            _,
            kv_packing,
            kv_dim,
        ) = cache_kv.shape
        if kv_packing % get_dtype_packing(cache_kv.dtype) != 0:
            raise ValueError(
                f"{kv_packing=} does not match with {cache_kv.dtype=}")
    else:
        (
            _,
            kv_dim,
            _,
        ) = cache_kv.shape
    aligned_kv_dim = align_to(kv_dim, 128)
    if lkv_dim + r_dim != aligned_kv_dim:
        raise ValueError(
            f"Expected {lkv_dim=} + {r_dim=} to be equal to {aligned_kv_dim=}")

    if not (cache_kv.dtype == new_kv_c.dtype):
        raise ValueError(
            f"Expected {cache_kv.dtype=} to be equal to {new_kv_c.dtype=}.")
    if not (cache_kv.dtype == new_k_pe.dtype):
        raise ValueError(
            f"Expected {cache_kv.dtype=} to be equal to {new_k_pe.dtype=}.")

    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
        raise ValueError(f"Expected {cache_kv.dtype=} to be a floating point.")

    if not (jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype
            == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}")

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(
            cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}")

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1, ):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3, ):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if num_kv_pages_per_blocks is not None:
        for num_kv_pages_per_block in num_kv_pages_per_blocks:
            if num_kv_pages_per_block <= 0:
                raise ValueError(
                    f"{num_kv_pages_per_block=} must be positive.")
    if mixed_q_split <= 0:
        raise ValueError(f"{mixed_q_split=} must be positive.")
    if num_queries_per_blocks is not None:
        for num_queries_per_block in num_queries_per_blocks:
            if num_queries_per_block <= 0:
                raise ValueError(f"{num_queries_per_block=} must be positive.")
        if num_queries_per_blocks[2] % mixed_q_split != 0:
            raise ValueError(
                f"{num_queries_per_blocks[2]=} must be divisible by"
                f" {mixed_q_split=}")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # No constraints for the following inputs.
    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del decode_batch_size
    del debug_mode


def _mla_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [batch_size, 6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz) * batch_size
    # Input
    ql_nope_hbm_ref,  # [max_num_tokens, num_q_heads, lkv_dim]
    q_pe_hbm_ref,  # [max_num_tokens, num_q_heads, r_dim]
    new_kv_c_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim] if not transpose_kv_cache else [lkv_dim, max_num_tokens]
    new_k_pe_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, r_dim] if not transpose_kv_cache else [r_dim, max_num_tokens]
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)] if not transpose_kv_cache else [total_num_pages, align_to(lkv_dim + r_dim, 128), page_size]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, lkv_dim]
    updated_cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)] if not transpose_kv_cache else [total_num_pages, align_to(lkv_dim + r_dim, 128), page_size]
    # Scratch
    bkvc_x2_ref,  # [2, batch_size, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim] if not transpose_kv_cache else [2, batch_size, lkv_dim, bkv_sz + 256]
    bkpe_x2_ref,  # [2, batch_size, bkv_buf_sz_per_kv_packing, kv_packing, r_dim] if not transpose_kv_cache else [2, batch_size, r_dim, bkv_sz + 256]
    bq_nope_x2_ref,  # [2, batch_size, bq_sz, num_q_heads, lkv_dim]
    bq_rope_x2_ref,  # [2, batch_size, bq_sz, num_q_heads, r_dim]
    bo_x2_ref,  # [2, batch_size, bq_sz, num_q_heads, lkv_dim]
    sems,  # [4, batch_size, 2]
    l_ref,  # [batch_size, bq_sz * num_q_heads, 128],
    m_ref,  # [batch_size, bq_sz * num_q_heads, 128],
    acc_ref,  # [batch_size, bq_sz * num_q_heads, lkv_dim],
    *,
    static_q_len: int,
    sm_scale: float,
    mask_value: float,
    s_dtype: jnp.dtype,
    transpose_kv_cache: bool,
    two_step_flash_attention: bool,
    p_same_dtype_as_v: bool,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    bkv_p,
    bq_sz,
    batch_size: int = 1,
    q_split: int = 1,
    debug_mode: bool = False,
):
    assert ql_nope_hbm_ref.shape == o_hbm_ref.shape
    # Validation checks on the dimensions
    nope_dim = ql_nope_hbm_ref.shape[-1]
    pe_dim = q_pe_hbm_ref.shape[-1]

    _, num_q_heads, lkv_dim = ql_nope_hbm_ref.shape
    r_dim = q_pe_hbm_ref.shape[-1]
    q_packing = get_dtype_packing(ql_nope_hbm_ref.dtype)
    num_q_heads_per_q_packing = num_q_heads // q_packing
    padded_num_q_heads = bq_nope_x2_ref.shape[-2]
    padded_num_q_heads_per_q_packing = padded_num_q_heads // q_packing
    kv_dtype = cache_kv_hbm_ref.dtype
    if not transpose_kv_cache:
        total_num_pages, page_size_per_kv_packing, kv_packing, kv_dim = (
            cache_kv_hbm_ref.shape)
        assert kv_packing % get_dtype_packing(kv_dtype) == 0
        num_sublanes_for_kv_packing = kv_packing // get_dtype_packing(
            cache_kv_hbm_ref.dtype)
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_sz = bkv_sz_per_kv_packing * kv_packing
        page_size = page_size_per_kv_packing * kv_packing
    else:
        (
            total_num_pages,
            kv_dim,
            page_size,
        ) = cache_kv_hbm_ref.shape
        bkv_sz = bkv_p * page_size
        page_size_per_lane = page_size // 128
    assert nope_dim + pe_dim == align_to(kv_dim, 128)
    kv_r_dim = kv_dim - nope_dim

    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    q_dtype = ql_nope_hbm_ref.dtype
    # Validate against the KV dtype.
    assert q_pe_hbm_ref.dtype == q_dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert lkv_dim % 128 == 0
    assert r_dim % 128 == 0

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    debug_print("[RPA debug] ======= In loop batch_start_seq_idx={}",
                batch_start_seq_idx)
    debug_print("[RPA debug] start_seq_idx={}", start_seq_idx)
    debug_print("[RPA debug] end_seq_idx={}", end_seq_idx)
    debug_print("[RPA debug] bkv_p={}", bkv_p)
    debug_print("[RPA debug] page_size={}", page_size)
    debug_print("[RPA debug] pages_per_seq={}", pages_per_seq)
    debug_print("[RPA debug] bq_sz={}", bq_sz)
    debug_print("[RPA debug] batch_size={}", batch_size)

    _LOG2_E = 1.4426950408889634
    _LOG_256 = 5.545177444479562
    total_qk_scale_exp = sm_scale * (k_scale if k_scale is not None else 1.0) * (
        q_scale if q_scale is not None else 1.0)
    total_qk_scale_exp2 = total_qk_scale_exp * _LOG2_E
    p_shift_exp = _LOG_256 if (p_same_dtype_as_v
                               and kv_dtype == jnp.float8_e4m3fn) else 0.0
    p_shift_exp2 = 8.0 if (p_same_dtype_as_v
                           and kv_dtype == jnp.float8_e4m3fn) else 0.0

    def batch_flash_attention(
        ql_nope,  # [batch_size, actual_bq_sz * num_q_heads, lkv_dim]
        q_pe,  # [batch_size, actual_bq_sz * num_q_heads, r_dim]
        kv_c,  # [batch_size, bkv_sz, lkv_dim] if not transpose_kv_cache else [batch_size, lkv_dim, bkv_sz] <- Correspond to data from bkvc_x2_ref
        k_pe,  # [batch_size, bkv_sz, r_dim] if not transpose_kv_cache else [batch_size, r_dim, bkv_sz] <- Correspond to data from bpe_x2_ref
        *,
        bq_idx,
        bkv_idx,
        precomputed_q_spans=None,
    ):
        assert len(ql_nope.shape) == 3
        assert len(q_pe.shape) == 3
        assert len(kv_c.shape) == 3
        assert len(k_pe.shape) == 3
        assert ql_nope.shape[1] % num_q_heads == 0
        assert ql_nope.shape[1] == q_pe.shape[1]
        assert q_pe.shape[1] % bq_sz == 0
        assert ql_nope.shape[2] == lkv_dim
        assert q_pe.shape[2] == r_dim
        if not transpose_kv_cache:
            assert kv_c.shape == (batch_size, bkv_sz, lkv_dim)
            assert k_pe.shape == (batch_size, bkv_sz, r_dim)
        else:
            assert kv_c.shape == (batch_size, lkv_dim, bkv_sz)
            assert k_pe.shape == (batch_size, r_dim, bkv_sz)

        head_l_ref = l_ref.at[:, :ql_nope.shape[1]]
        head_m_ref = m_ref.at[:, :ql_nope.shape[1]]
        head_acc_ref = acc_ref.at[:, :ql_nope.shape[1]]

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val),
                             ref[...])

        m_prev = load_with_init(head_m_ref, -jnp.inf)
        l_prev = load_with_init(head_l_ref, 0.0)

        ql_nope_q = (
            ql_nope.astype(kv_c.dtype)
            if (q_scale is None and k_scale is None and ql_nope.dtype != kv_c.dtype)
            else ql_nope
        )
        s_c = jnp.einsum(
            "bnd,bmd->bnm" if not transpose_kv_cache else "bnd,bdm->bnm",
            ql_nope_q,
            kv_c,
            preferred_element_type=jnp.float32)
        s_pe = jnp.einsum(
            "bnd,bmd->bnm" if not transpose_kv_cache else "bnd,bdm->bnm",
            q_pe,
            k_pe,
            preferred_element_type=jnp.float32)
        s = s_c + s_pe

        if soft_cap is not None:
            s *= total_qk_scale_exp
            s = soft_cap * jnp.tanh(s / soft_cap)
            s *= _LOG2_E
        else:
            s *= total_qk_scale_exp2

        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
            jnp.int32, s.shape[1:], 1)
        if bq_sz == 1 and static_q_len == 1:
            mask_list = []
            for b in range(batch_size):
                seq_idx = batch_start_seq_idx + b
                q_span = kv_lens_ref[seq_idx] - 1
                mask = q_span < k_span
                if sliding_window is not None:
                    mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)
                mask_list.append(mask)
            mask = jnp.stack(mask_list, axis=0)
        else:
            q_offset = bq_idx * bq_sz + unsigned_floor_div(
                lax.broadcasted_iota(jnp.int32, s.shape[1:], 0), num_q_heads)

            mask_list = []
            for b in range(batch_size):
                if precomputed_q_spans is not None:
                    q_span = precomputed_q_spans[b]
                else:
                    seq_idx = batch_start_seq_idx + b
                    q_len = cu_q_lens_ref[seq_idx + 1] - cu_q_lens_ref[seq_idx]
                    kv_len = kv_lens_ref[seq_idx]
                    q_span = kv_len - q_len + q_offset
                mask = q_span < k_span
                if sliding_window is not None:
                    mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)
                mask_list.append(mask)
            mask = jnp.stack(mask_list, axis=0)

        s = s.astype(s_dtype)
        s = jnp.where(mask, mask_value, s)
        s_rowmax = jnp.max(s, axis=2, keepdims=True)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr

        exp_m_diff = jnp.exp2(m_prev - m_curr)
        p = jnp.exp2(s - broadcast_minor(m_curr - p_shift_exp2, s.shape))

        p_rowsum = jnp.sum(p, axis=2, keepdims=True)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr

        o_prev = load_with_init(head_acc_ref, 0.0)
        o_scaled = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev

        if p_same_dtype_as_v:
            p = p.astype(kv_c.dtype)
        pv = jnp.einsum(
            "bnm,bmd->bnd" if not transpose_kv_cache else "bnm,bdm->bnd",
            p,
            kv_c,
            preferred_element_type=jnp.float32)
        head_acc_ref[...] = o_scaled + pv

    def flash_attention_step1(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        *,
        q_full=None,
        kv_full_t=None,
        kv_c_t=None,
        k_pe_t=None,
        precomputed_delta_iota=None,
        kv_len,
        q_len,
        bq_idx,
        bq_offset,
        bkv_idx,
        head_l_ref,
        head_m_ref,
    ):
        assert len(ql_nope.shape) == 2
        assert len(q_pe.shape) == 2
        assert len(kv_c.shape) == 2
        assert len(k_pe.shape) == 2
        assert ql_nope.shape[0] == q_pe.shape[0]
        assert ql_nope.shape[1] == lkv_dim
        assert q_pe.shape[1] == r_dim
        if not transpose_kv_cache:
            assert kv_c.shape[1] == lkv_dim
            assert k_pe.shape[1] == r_dim
        else:
            assert kv_c.shape == (lkv_dim, bkv_sz)
            assert k_pe.shape == (r_dim, bkv_sz)

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val),
                             ref[...])

        m_prev = load_with_init(head_m_ref, -jnp.inf)

        use_step1_exp2 = soft_cap is None
        step1_qk_scale = total_qk_scale_exp2 if use_step1_exp2 else total_qk_scale_exp
        step1_p_shift = p_shift_exp2 if use_step1_exp2 else p_shift_exp
        step1_exp_fn = jnp.exp2 if use_step1_exp2 else jnp.exp

        # Follow FlashAttention-2 forward pass with power-of-2 chunk_sz (1024 for >=1024)
        # to overlap MXU QK^T with VPU scale/mask/rowmax while keeping XLU/VMEM tiles power-of-2 aligned.
        active_kv_sz = kv_c.shape[0] if not transpose_kv_cache else kv_c.shape[1]
        chunk_sz = 1024 if active_kv_sz >= 1024 and active_kv_sz % 1024 == 0 else (512 if active_kv_sz % 512 == 0 else 256)
        if active_kv_sz % chunk_sz == 0 and ql_nope.shape[0] % num_q_heads == 0:
            base_threshold = (kv_len - q_len + (bq_idx * bq_sz + bq_offset) -
                              bkv_idx * bkv_sz)
            delta_iota = precomputed_delta_iota if precomputed_delta_iota is not None else (
                lax.broadcasted_iota(
                    jnp.int32, (ql_nope.shape[0], chunk_sz), 1) - unsigned_floor_div(
                        lax.broadcasted_iota(jnp.int32, (ql_nope.shape[0], chunk_sz), 0),
                        num_q_heads)
            )

            m_curr = m_prev
            s_list = []
            num_chunks = active_kv_sz // chunk_sz
            for partial_idx in range(num_chunks):
                partial_start = partial_idx * chunk_sz
                if not transpose_kv_cache and q_full is not None and kv_full_t is not None:
                    kv_f_part = kv_full_t[partial_idx] if isinstance(kv_full_t, (list, tuple)) else kv_full_t
                    s_part = jnp.einsum("nd,dm->nm",
                                        q_full,
                                        kv_f_part,
                                        preferred_element_type=jnp.float32) * step1_qk_scale
                else:
                    if not transpose_kv_cache:
                        if isinstance(kv_c_t, (list, tuple)):
                            kv_c_partial = kv_c_t[partial_idx]
                        elif kv_c_t is not None and num_chunks == 1:
                            kv_c_partial = kv_c_t
                        else:
                            kv_c_partial = kv_c if num_chunks == 1 else kv_c[partial_start:partial_start + chunk_sz, :]
                            kv_c_partial = pltpu.bitcast(kv_c_partial.transpose((1, 0)), kv_c_partial.dtype)
                        if isinstance(k_pe_t, (list, tuple)):
                            k_pe_partial = k_pe_t[partial_idx]
                            s_pe_partial = jnp.einsum("nd,dm->nm", q_pe, k_pe_partial, preferred_element_type=jnp.float32)
                        elif k_pe_t is not None and num_chunks == 1:
                            k_pe_partial = k_pe_t
                            s_pe_partial = jnp.einsum("nd,dm->nm", q_pe, k_pe_partial, preferred_element_type=jnp.float32)
                        elif isinstance(k_pe, (list, tuple)):
                            k_pe_partial = k_pe[partial_idx]
                            s_pe_partial = jnp.einsum("nd,md->nm", q_pe, k_pe_partial, preferred_element_type=jnp.float32)
                        else:
                            k_pe_partial = k_pe if num_chunks == 1 else k_pe[partial_start:partial_start + chunk_sz, :]
                            s_pe_partial = jnp.einsum("nd,md->nm", q_pe, k_pe_partial, preferred_element_type=jnp.float32)
                    else:
                        kv_c_partial = kv_c if num_chunks == 1 else kv_c[:, partial_start:partial_start + chunk_sz]
                        k_pe_partial = k_pe if num_chunks == 1 else k_pe[:, partial_start:partial_start + chunk_sz]
                        s_pe_partial = jnp.einsum("nd,dm->nm", q_pe, k_pe_partial, preferred_element_type=jnp.float32)

                    s_c_partial = jnp.einsum("nd,dm->nm",
                                             ql_nope,
                                             kv_c_partial,
                                             preferred_element_type=jnp.float32)

                    s_part = (s_c_partial + s_pe_partial) * step1_qk_scale
                if soft_cap is not None:
                    s_part = soft_cap * jnp.tanh(s_part / soft_cap)
                s_part = s_part.astype(s_dtype)

                thresh_scalar = base_threshold - partial_start
                mask_part = thresh_scalar < delta_iota
                if sliding_window is not None:
                    mask_part = jnp.logical_or(
                        mask_part, thresh_scalar - sliding_window >= delta_iota)
                s_part = jnp.where(mask_part, mask_value, s_part)

                m_curr = jnp.maximum(m_curr, jnp.max(s_part, axis=1, keepdims=True))
                s_list.append(s_part)
        else:
            q = jnp.concatenate([ql_nope, q_pe], axis=-1)
            k = jnp.concatenate([kv_c, k_pe],
                                axis=-1 if not transpose_kv_cache else -2)
            s = jnp.einsum(
                "nd,md->nm" if not transpose_kv_cache else "nd,dm->nm",
                q,
                k,
                preferred_element_type=jnp.float32)
            s *= step1_qk_scale

            if soft_cap is not None:
                s = soft_cap * jnp.tanh(s / soft_cap)
            s = s.astype(s_dtype)

            q_span = (
                kv_len - q_len +
                (bq_idx * bq_sz + bq_offset) + unsigned_floor_div(
                    lax.broadcasted_iota(jnp.int32, s.shape, 0), num_q_heads))
            k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
                jnp.int32, s.shape, 1)
            mask = q_span < k_span

            if sliding_window is not None:
                mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)
            s = jnp.where(mask, mask_value, s)
            s_rowmax = jnp.max(s, axis=1, keepdims=True)
            m_curr = jnp.maximum(m_prev, s_rowmax)
            s_list = [s]

        p_list = []
        p_rowsum = None
        for s_part in s_list:
            p_part = step1_exp_fn(
                s_part - broadcast_minor(m_curr - step1_p_shift, s_part.shape))
            part_sum = jnp.sum(p_part, axis=1, keepdims=True)
            p_rowsum = part_sum if p_rowsum is None else (p_rowsum + part_sum)
            if p_same_dtype_as_v and p_part.dtype != kv_c.dtype:
                p_part = p_part.astype(kv_c.dtype)
            p_list.append(p_part)

        head_m_ref[...] = m_curr.astype(head_m_ref.dtype)
        return p_list, m_prev, m_curr, p_rowsum

    def flash_attention_step2(
        p_list,
        m_prev,
        m_curr,
        p_rowsum,
        v,
        *,
        bkv_idx,
        head_l_ref,
        head_acc_ref,
    ):
        if not isinstance(v, (list, tuple)):
            if not transpose_kv_cache:
                assert v.shape[1] == lkv_dim
            else:
                assert v.shape == (lkv_dim, bkv_sz)

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val),
                             ref[...])

        l_prev = load_with_init(head_l_ref, 0.0)
        o_prev = load_with_init(head_acc_ref, 0.0)

        use_step1_exp2 = soft_cap is None
        step1_exp_fn = jnp.exp2 if use_step1_exp2 else jnp.exp

        exp_m_diff = step1_exp_fn(m_prev - m_curr)
        head_l_ref[...] = exp_m_diff * l_prev + p_rowsum

        if not isinstance(p_list, list):
            p_list = [p_list]

        num_parts = len(p_list)
        active_v_sz = v.shape[0] if (not isinstance(v, (list, tuple)) and not transpose_kv_cache) else bkv_sz
        part_sz = active_v_sz // num_parts

        pv_acc = None
        for idx, p_part in enumerate(p_list):
            if isinstance(v, (list, tuple)):
                v_part = v[idx]
            elif not transpose_kv_cache:
                v_part = v if num_parts == 1 else v[idx * part_sz:(idx + 1) * part_sz, :]
            else:
                v_part = v if num_parts == 1 else v[:, idx * part_sz:(idx + 1) * part_sz]
                v_part = v_part.transpose((1, 0))
                v_part = pltpu.bitcast(v_part, v_part.dtype)

            pv_part = jnp.einsum("nm,md->nd",
                                 p_part,
                                 v_part,
                                 preferred_element_type=jnp.float32)
            if idx == 0:
                pv_acc = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv_part
            else:
                pv_acc = pv_acc + pv_part

        head_acc_ref[...] = pv_acc

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            # Skip DMA if debug mode is enabled.
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_transposed_bkv(batch_start_seq_idx,
                              bkv_idx,
                              bkv_sem_idx,
                              *,
                              bq_idx=0,
                              wait=False):
        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

        offsets = []
        update_szs = []
        for b in range(batch_size):
            sem = sems.at[0, b, bkv_sem_idx]
            # bkvc_x2_ref shape: [2, batch_size, lkv_dim_per_kv_packing, kv_packing, bkv_sz]
            bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
            bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

            seq_idx = batch_start_seq_idx + b
            is_valid_seq = seq_idx < end_seq_idx
            safe_seq_idx = jnp.minimum(seq_idx, max_num_seqs - 1)
            kv_len = jnp.where(is_valid_seq, kv_lens_ref[safe_seq_idx], 0)
            kv_len_start = bkv_idx * bkv_sz
            kv_p_start = bkv_idx * bkv_p

            q_start = jnp.where(is_valid_seq, cu_q_lens_ref[safe_seq_idx], 0)
            q_end = jnp.where(is_valid_seq, cu_q_lens_ref[safe_seq_idx + 1], 0)
            q_len = jnp.where(bq_idx == 0, q_end - q_start, 0)

            kv_left = jnp.maximum(kv_len - kv_len_start, 0)
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)

            kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)

            bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
            bkv_sz_frm_cache_aligned = align_to(bkv_sz_frm_cache, 128)
            bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache,
                                         kv_left_frm_new)
            new_kv_start_idx = bkv_sz_frm_cache_aligned + 128

            page_indices_offset = safe_seq_idx * pages_per_seq + kv_p_start

            new_kv_len_start = q_end - kv_left_frm_new
            new_kv_len_end = new_kv_len_start + bkv_sz_frm_new

            aligned_new_kv_len_start = unsigned_floor_div(
                new_kv_len_start, 128) * 128
            aligned_new_kv_len_end = unsigned_align_to(new_kv_len_end, 128)
            aligned_new_kv_len_size = align_to(
                aligned_new_kv_len_end - aligned_new_kv_len_start, 128)
            aligned_new_kv_len_size = lax.select(bkv_sz_frm_new == 0, 0,
                                                 aligned_new_kv_len_size)

            debug_print(
                "[RPA debug]"
                f" -----------{'wait' if wait else 'start'}_fetch_transposed_bkv-----------"
            )
            debug_print("[RPA debug] seq_idx={}", seq_idx)
            debug_print("[RPA debug] bkv_idx={}", bkv_idx)
            debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
            debug_print("[RPA debug] kv_len_start={}", kv_len_start)
            debug_print("[RPA debug] kv_p_start={}", kv_p_start)
            debug_print("[RPA debug] kv_left={}", kv_left)
            debug_print("[RPA debug] kv_left_frm_cache={}", kv_left_frm_cache)
            debug_print("[RPA debug] kv_left_frm_new={}", kv_left_frm_new)
            debug_print("[RPA debug] bkv_sz_frm_cache={}", bkv_sz_frm_cache)
            debug_print("[RPA debug] page_indices_offset={}",
                        page_indices_offset)
            debug_print(
                f"[RPA debug] bkvc_vmem_ref.shape: {bkvc_vmem_ref.shape}")
            debug_print(
                f"[RPA debug] bkvpe_vmem_ref.shape: {bkvpe_vmem_ref.shape}")

            if not wait:

                # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
                # utilize static for loop instead of dynamic for loop.
                # Loop through all pages in a block
                for i in range(bkv_p):
                    # Ensure only effective kvs are copied and we don't go negative.
                    copy_len = jnp.clip(
                        kv_left_frm_cache - i * page_size,
                        0,
                        page_size,
                    )
                    copy_len = align_to(copy_len, 128)
                    # If the page index is out of bound, we set page_idx to the last page.
                    # And there will be no copy since sz will be 0.
                    page_idx = jnp.minimum(page_indices_offset + i,
                                           num_page_indices - 1)
                    _async_copy(
                        cache_kv_hbm_ref.at[
                            page_indices_ref[page_idx],
                            :nope_dim,
                            pl.ds(0, copy_len),
                        ],
                        bkvc_vmem_ref.at[:, pl.ds(i * page_size, copy_len)],
                        sem,
                        wait,
                    )
                    _async_copy(
                        cache_kv_hbm_ref.at[
                            page_indices_ref[page_idx],
                            nope_dim:,
                            pl.ds(0, copy_len),
                        ],
                        bkvpe_vmem_ref.at[:kv_r_dim,
                                          pl.ds(i * page_size, copy_len)],
                        sem,
                        wait,
                    )
                    debug_print(
                        "[RPA debug] loop_body bkv_p={}, i={}, page_size_per_lane={},"
                        " copy_len={}, page_idx={}, page_indices_ref[page_idx]={}",
                        bkv_p,
                        i,
                        page_size_per_lane,
                        copy_len,
                        page_idx,
                        page_indices_ref[page_idx],
                    )

                # Fetch new KVs by appending to the existing vmem buffers.
                # Fetch either up to the end of the buffer or kv_left_frm_new, whichever
                # is smaller. Since DMAs are word-aligned based on kv_packing, and the
                # boundary between the old cache and the new KV tokens might not be
                # word-aligned, we append the new KV words right after the last word
                # containing old cache data. This can create "holes" (misalignments
                # within the words), which we will shift and pack correctly later.
                debug_print("[RPA debug] new_kv_len_start={}",
                            new_kv_len_start)
                debug_print(
                    "[RPA debug] new_kv_len_end={}",
                    new_kv_len_end,
                )
                debug_print("[RPA debug] aligned_new_kv_len_start={}",
                            aligned_new_kv_len_start)
                debug_print(
                    "[RPA debug] new_kv_start_idx={}",
                    new_kv_start_idx,
                )
                debug_print(
                    "[RPA debug] aligned_new_kv_len_size={}",
                    aligned_new_kv_len_size,
                )
                debug_print(
                    f"new_kv_c_hbm_ref.shape: {new_kv_c_hbm_ref.shape}")
                debug_print(
                    f"new_k_pe_hbm_ref.shape: {new_k_pe_hbm_ref.shape}")
                # new_kv_len_start ~ new_kv_len_start + bkv_sz_frm_new
                # new_kv_copy_size = bkv_sz
                _async_copy(
                    new_kv_c_hbm_ref.at[:,
                                        pl.ds(
                                            aligned_new_kv_len_start,
                                            aligned_new_kv_len_size,
                                        )],
                    bkvc_vmem_ref.at[:,
                                     pl.ds(
                                         new_kv_start_idx,
                                         aligned_new_kv_len_size,
                                     )],
                    sem,
                    wait,
                )
                _async_copy(
                    new_k_pe_hbm_ref.at[:kv_r_dim,
                                        pl.ds(
                                            aligned_new_kv_len_start,
                                            aligned_new_kv_len_size,
                                        )],
                    bkvpe_vmem_ref.at[:kv_r_dim,
                                      pl.ds(
                                          new_kv_start_idx,
                                          aligned_new_kv_len_size,
                                      )],
                    sem,
                    wait,
                )

            else:
                # When we wait, we can use a dummy copy to wait for DMAs to complete where
                # src == dst. However, the dma size must be correct.
                dma_sz = bkv_sz_frm_cache_aligned + aligned_new_kv_len_size
                dst_kvc = bkvc_vmem_ref.at[:, pl.ds(0, dma_sz)]
                _async_copy(
                    src=dst_kvc,
                    dst=dst_kvc,
                    sem=sem,
                    wait=True,
                )
                dst_kvpe = bkvpe_vmem_ref.at[:kv_r_dim, pl.ds(0, dma_sz)]
                _async_copy(
                    src=dst_kvpe,
                    dst=dst_kvpe,
                    sem=sem,
                    wait=True,
                )

            # This returns the (offset, size) in units of tokens:
            #   offset: starting token index where the new KV should be stored
            #   size: number of tokens of the new KV, which is 1 in decode.
            offsets.append(kv_len_start + bkv_sz_frm_cache)
            update_szs.append(bkv_sz_frm_new)

        return offsets, update_szs

    def _fetch_bkv(batch_start_seq_idx,
                   bkv_idx,
                   bkv_sem_idx,
                   *,
                   bq_idx=0,
                   wait=False):
        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

        offsets = []
        update_szs = []
        for b in range(batch_size):
            sem = sems.at[0, b, bkv_sem_idx]
            # bkvc_x2_ref shape: [2, batch_size, bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
            bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
            bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

            # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
            # [total_num_pages * page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
            reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
                total_num_pages * page_size_per_kv_packing,
                *cache_kv_hbm_ref.shape[2:],
            )

            seq_idx = batch_start_seq_idx + b
            is_valid_seq = seq_idx < end_seq_idx
            safe_seq_idx = jnp.minimum(seq_idx, max_num_seqs - 1)
            kv_len = jnp.where(is_valid_seq, kv_lens_ref[safe_seq_idx], 0)
            kv_len_start = bkv_idx * bkv_sz
            kv_p_start = bkv_idx * bkv_p

            q_start = jnp.where(is_valid_seq, cu_q_lens_ref[safe_seq_idx], 0)
            q_end = jnp.where(is_valid_seq, cu_q_lens_ref[safe_seq_idx + 1], 0)
            q_len = jnp.where(bq_idx == 0, q_end - q_start, 0)

            kv_left = jnp.maximum(kv_len - kv_len_start, 0)
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
            kv_left_frm_cache_per_kv_packing = unsigned_cdiv(
                kv_left_frm_cache, kv_packing)
            kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)

            bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
            bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache,
                                         kv_left_frm_new)
            bkv_sz_frm_cache_per_kv_packing = unsigned_cdiv(
                bkv_sz_frm_cache, kv_packing)
            bkv_sz_frm_new_per_kv_packing = unsigned_cdiv(
                bkv_sz_frm_new, kv_packing)
            page_indices_offset = safe_seq_idx * pages_per_seq + kv_p_start

            new_kv_len_start = q_end - kv_left_frm_new
            new_kv_len_start_per_kv_packing = unsigned_floor_div(
                new_kv_len_start, kv_packing)
            bkv_sz_frm_new_kv_packing_to_fetch = jnp.where(
                bkv_sz_frm_new > 0,
                unsigned_cdiv(new_kv_len_start + bkv_sz_frm_new, kv_packing) -
                new_kv_len_start_per_kv_packing,
                0,
            )
            dma_bkv_sz = (bkv_sz_frm_cache_per_kv_packing +
                          bkv_sz_frm_new_kv_packing_to_fetch)

            debug_print(
                "[RPA debug]"
                f" -----------{'wait' if wait else 'start'}_fetch_bkv-----------"
            )
            debug_print("[RPA debug] seq_idx={}", seq_idx)
            debug_print("[RPA debug] bkv_idx={}", bkv_idx)
            debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
            debug_print("[RPA debug] kv_len_start={}", kv_len_start)
            debug_print("[RPA debug] kv_p_start={}", kv_p_start)
            debug_print("[RPA debug] kv_left={}", kv_left)
            debug_print("[RPA debug] kv_left_frm_cache={}", kv_left_frm_cache)
            debug_print("[RPA debug] kv_left_frm_new={}", kv_left_frm_new)
            debug_print("[RPA debug] bkv_sz_frm_cache={}", bkv_sz_frm_cache)
            debug_print(
                "[RPA debug] bkv_sz_frm_cache_per_kv_packing={}",
                bkv_sz_frm_cache_per_kv_packing,
            )
            debug_print(
                "[RPA debug] bkv_sz_frm_new_per_kv_packing={}",
                bkv_sz_frm_new_per_kv_packing,
            )
            debug_print("[RPA debug] page_indices_offset={}",
                        page_indices_offset)
            debug_print(
                f"[RPA debug] bkvc_vmem_ref.shape: {bkvc_vmem_ref.shape}")
            debug_print(
                f"[RPA debug] bkvpe_vmem_ref.shape: {bkvpe_vmem_ref.shape}")

            if not wait:

                # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
                # utilize static for loop instead of dynamic for loop.
                # Loop through all pages in a block
                for i in range(bkv_p):
                    # Ensure only effective kvs are copied and we don't go negative.
                    sz_per_kv_packing = jnp.clip(
                        kv_left_frm_cache_per_kv_packing -
                        i * page_size_per_kv_packing,
                        0,
                        page_size_per_kv_packing,
                    )
                    # If the page index is out of bound, we set page_idx to the last page.
                    # And there will be no copy since sz will be 0.
                    page_idx = jnp.minimum(page_indices_offset + i,
                                           num_page_indices - 1)
                    _async_copy(
                        reshaped_cache_hbm_ref.at[
                            pl.ds(
                                page_indices_ref[page_idx] *
                                page_size_per_kv_packing,
                                sz_per_kv_packing,
                            ),
                            ...,
                            :nope_dim,
                        ],
                        # [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim].
                        bkvc_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                               sz_per_kv_packing)],
                        sem,
                        wait,
                    )
                    _async_copy(
                        reshaped_cache_hbm_ref.at[
                            pl.ds(
                                page_indices_ref[page_idx] *
                                page_size_per_kv_packing,
                                sz_per_kv_packing,
                            ),
                            ...,
                            nope_dim:,
                        ],
                        # [bkv_sz_per_kv_packing + 2, kv_packing, r_dim].
                        bkvpe_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                                sz_per_kv_packing)],
                        sem,
                        wait,
                    )
                    debug_print(
                        "[RPA debug] loop_body bkv_p={}, i={}, page_size_per_kv_packing={},"
                        " sz_per_kv_packing={}, page_idx={}, page_indices_ref[page_idx]={}",
                        bkv_p,
                        i,
                        page_size_per_kv_packing,
                        sz_per_kv_packing,
                        page_idx,
                        page_indices_ref[page_idx],
                    )

                # Fetch new KVs by appending to the existing vmem buffers.
                # Fetch either up to the end of the buffer or kv_left_frm_new, whichever
                # is smaller. Since DMAs are word-aligned based on kv_packing, and the
                # boundary between the old cache and the new KV tokens might not be
                # word-aligned, we append the new KV words right after the last word
                # containing old cache data. This can create "holes" (misalignments
                # within the words), which we will shift and pack correctly later.
                debug_print("[RPA debug] new_kv_len_start={}",
                            new_kv_len_start)
                debug_print(
                    "[RPA debug] new_kv_len_start_per_kv_packing={}",
                    new_kv_len_start_per_kv_packing,
                )
                debug_print(
                    f"new_kv_c_hbm_ref.shape: {new_kv_c_hbm_ref.shape}")
                debug_print(
                    f"new_k_pe_hbm_ref.shape: {new_k_pe_hbm_ref.shape}")
                _async_copy(
                    new_kv_c_hbm_ref.at[pl.ds(
                        new_kv_len_start_per_kv_packing,
                        bkv_sz_frm_new_kv_packing_to_fetch,
                    )],
                    bkvc_vmem_ref.at[pl.ds(
                        bkv_sz_frm_cache_per_kv_packing,
                        bkv_sz_frm_new_kv_packing_to_fetch,
                    )],
                    sem,
                    wait,
                )
                _async_copy(
                    new_k_pe_hbm_ref.at[pl.ds(
                        new_kv_len_start_per_kv_packing,
                        bkv_sz_frm_new_kv_packing_to_fetch,
                    )],
                    bkvpe_vmem_ref.at[pl.ds(
                        bkv_sz_frm_cache_per_kv_packing,
                        bkv_sz_frm_new_kv_packing_to_fetch,
                    )],
                    sem,
                    wait,
                )

            else:
                # When we wait, we can use a dummy copy to wait for DMAs to complete where
                # src == dst. However, the dma size must be correct.
                dst_kvc = bkvc_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
                _async_copy(
                    src=dst_kvc,
                    dst=dst_kvc,
                    sem=sem,
                    wait=True,
                )
                dst_kvpe = bkvpe_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
                _async_copy(
                    src=dst_kvpe,
                    dst=dst_kvpe,
                    sem=sem,
                    wait=True,
                )
            # This returns the (offset, size) in units of tokens:
            #   offset: starting token index where the new KV should be stored
            #   size: number of tokens of the new KV, which is 1 in decode.
            offsets.append(kv_len_start + bkv_sz_frm_cache)
            update_szs.append(bkv_sz_frm_new)

        return offsets, update_szs

    def _pack_new_transposed_kv(bkv_sem_idx, bkv_idx, offsets, update_szs):
        """Packs newly computed KVs into the correct sub-word alignment in VMEM.

    Args:
      bkv_sem_idx: The semaphore index for the current KV block.
      bkv_idx: The index of the current KV block.
      offsets: A list of the starting token offset in the KV cache where the
        new KVs begin for a batch.
      update_szs: A list of the number of new tokens to be packed for a batch.
    """
        debug_print("[RPA debug]"
                    " -----------_pack_new_transposed_kv-----------")
        for b in range(batch_size):
            update_sz = update_szs[b]

            @pl.when(update_sz > 0)
            def _update(b=b, update_sz=update_sz):
                bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
                bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

                seq_idx = batch_start_seq_idx + b
                kv_len = kv_lens_ref[seq_idx]
                kv_len_start = bkv_idx * bkv_sz

                q_start = cu_q_lens_ref[seq_idx]
                q_end = cu_q_lens_ref[seq_idx + 1]
                q_len = q_end - q_start

                kv_left = jnp.maximum(kv_len - kv_len_start, 0)
                kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)

                kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)
                bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
                bkv_sz_frm_cache_aligned = align_to(bkv_sz_frm_cache, 128)
                new_kv_start_idx = bkv_sz_frm_cache_aligned + 128

                new_kv_len_start = q_end - kv_left_frm_new
                new_kv_len_end = new_kv_len_start + update_sz
                new_kv_len_sz = new_kv_len_end - new_kv_len_start

                aligned_new_kv_len_start = unsigned_floor_div(
                    new_kv_len_start, 128) * 128
                aligned_new_kv_len_end = unsigned_align_to(new_kv_len_end, 128)
                aligned_new_kv_len_size = unsigned_align_to(
                    aligned_new_kv_len_end - aligned_new_kv_len_start, 128)
                aligned_new_kv_len_size = lax.select(update_sz == 0, 0,
                                                     aligned_new_kv_len_size)

                new_kv_buffer_start = unsigned_mod(new_kv_len_start, 128)
                new_kv_buffer_end = new_kv_buffer_start + new_kv_len_sz
                debug_print("[RPA debug] new_kv_buffer_start={}",
                            new_kv_buffer_start)
                debug_print(
                    "[RPA debug] new_kv_buffer_end={}",
                    new_kv_buffer_end,
                )
                debug_print("[RPA debug] aligned_new_kv_len_start={}",
                            aligned_new_kv_len_start)
                debug_print(
                    "[RPA debug] aligned_new_kv_len_size={}",
                    aligned_new_kv_len_size,
                )

                loop_sz = unsigned_floor_div(aligned_new_kv_len_size, 128)

                # TODO(kimjaehong) : Do we have to consider copy_sz?
                def copy_partial(new_kv_idx, new_kv_start, bkv_idx, bkv_start):
                    new_kvc_idx = new_kv_start_idx + new_kv_idx * 128
                    debug_print(
                        "[RPA debug] copy_partial new_kv_idx={}, new_kv_start={},"
                        " bkv_idx={}, bkv_start={}, new_kvc_idx={}",
                        new_kv_idx,
                        new_kv_start,
                        bkv_idx,
                        bkv_start,
                        new_kvc_idx,
                    )
                    new_kvc_ref = bkvc_vmem_ref.bitcast(
                        jnp.uint32).at[..., pl.ds(new_kvc_idx, 128)]
                    bkvc_ref = bkvc_vmem_ref.bitcast(
                        jnp.uint32).at[..., pl.ds(bkv_idx * 128, 128)]
                    rolled_new_kvc = pltpu.roll(new_kvc_ref[...],
                                                shift=bkv_start - new_kv_start,
                                                axis=1)
                    bkvc_ref[...] = lax.select(
                        lax.broadcasted_iota(jnp.int32, bkvc_ref.shape, 1)
                        < bkv_start, bkvc_ref[...], rolled_new_kvc)

                    new_kvpe_ref = bkvpe_vmem_ref.bitcast(
                        jnp.uint32).at[..., pl.ds(new_kvc_idx, 128)]
                    bkvpe_ref = bkvpe_vmem_ref.bitcast(
                        jnp.uint32).at[..., pl.ds(bkv_idx * 128, 128)]
                    rolled_new_kvpe = pltpu.roll(new_kvpe_ref[...],
                                                 shift=bkv_start -
                                                 new_kv_start,
                                                 axis=1)
                    bkvpe_ref[...] = lax.select(
                        lax.broadcasted_iota(jnp.int32, bkvpe_ref.shape, 1)
                        < bkv_start, bkvpe_ref[...], rolled_new_kvpe)

                def update_body(idx, state):
                    filled_bkv_sz, block_start = state
                    block_end = block_start + 128
                    bkv_idx = unsigned_floor_div(filled_bkv_sz, 128)
                    start_in_bkv_block = unsigned_mod(filled_bkv_sz, 128)
                    unfilled_block_sz = 128 - start_in_bkv_block

                    buffer_start = jnp.maximum(block_start,
                                               new_kv_buffer_start)
                    buffer_end = jnp.minimum(block_end, new_kv_buffer_end)

                    buffer_sz_in_block = buffer_end - buffer_start
                    start_in_block = unsigned_mod(buffer_start, 128)

                    cur_filling_sz = jnp.minimum(unfilled_block_sz,
                                                 buffer_sz_in_block)
                    copy_partial(idx, start_in_block, bkv_idx,
                                 start_in_bkv_block)
                    copy_partial(idx, start_in_block + cur_filling_sz,
                                 bkv_idx + 1, 0)
                    return (filled_bkv_sz + buffer_sz_in_block, block_end)

                jax.lax.fori_loop(0,
                                  loop_sz,
                                  update_body,
                                  (bkv_sz_frm_cache, jnp.int32(0)),
                                  unroll=False)

    def _pack_new_kv(bkv_sem_idx, offsets, update_szs):
        """Packs newly computed KVs into the correct sub-word alignment in VMEM.

    When new KV tokens are DMA'd from HBM into VMEM, they are copied at the
    granularity of packed words (e.g., 4 tokens per word for fp8) by head
    dimension (mapped to lanes). The starting token `offset` in the KV cache,
    however, might not fall exactly on a word boundary. This means the elements
    within the packed words might be misaligned relative to their final
    destination in the cache.

    This function corrects this alignment by:
    1. Computing the bit-shift amount needed based on the difference between the
       destination token offset (`kv_packing_offset`) and the source token
       offset (`new_kv_packing_offset`).
    2. Looping over the affected words and using bitwise shifts and logical ORs
       to realign the elements across word boundaries.
    3. Merging the correctly aligned new KV elements into the VMEM buffer using
       a mask, leaving existing (older) KV elements intact.

    Args:
      bkv_sem_idx: The semaphore index for the current KV block.
      offsets: A list of the starting token offset in the KV cache where the
        new KVs begin for a batch.
      update_szs: A list of the number of new tokens to be packed for a batch.
    """
        debug_print("[RPA debug]"
                    " -----------_pack_new_kv-----------")
        for b in range(batch_size):
            offset = offsets[b]
            update_sz = update_szs[b]

            @pl.when(update_sz > 0)
            def _update(b=b, offset=offset, update_sz=update_sz):
                # shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
                bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
                # shape: [bkv_sz_per_kv_packing + 2, kv_packing, r_dim]
                bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

                seq_idx = batch_start_seq_idx + b
                safe_seq_idx = jnp.minimum(seq_idx, max_num_seqs - 1)
                q_end = cu_q_lens_ref[safe_seq_idx + 1]
                kv_len = kv_lens_ref[safe_seq_idx]

                kv_utils.pack_new_kv(bkvc_vmem_ref, bkvpe_vmem_ref, offset,
                                     update_sz, q_end, kv_len, bkv_sz)

    def _update_transposed_kv_cache(
        batch_start_seq_idx,
        b,
        bkv_sem_idx,
        offset,  # In units of tokens.
        update_sz,  # In units of tokens.
        *,
        wait=False,
    ):
        debug_print("[RPA debug]"
                    " -----------_update_transposed_kv_cache-----------")
        seq_idx = batch_start_seq_idx + b
        sem = sems.at[3, b, bkv_sem_idx]

        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

        update_lane_iters = unsigned_cdiv(
            unsigned_mod(offset, 128) + update_sz, 128)
        debug_print(
            "[RPA debug] offset={}",
            offset,
        )
        debug_print(
            "[RPA debug] update_sz={}",
            update_sz,
        )
        debug_print(
            "[RPA debug] update_lane_iters={}",
            update_lane_iters,
        )

        # [total_num_pages, kv_dim_per_kv_packing, kv_packing, page_size]
        # cache_kv_hbm_shape = updated_cache_kv_hbm_ref.shape

        if not wait:
            # Issue DMA copy for the updated parts, page by page.
            kv_p_start = unsigned_floor_div(offset, page_size)
            kv_p_end = unsigned_cdiv(offset + update_sz, page_size)
            start_word_in_page = unsigned_floor_div(
                unsigned_mod(offset, page_size), 128)
            page_size_per_lane = unsigned_floor_div(page_size, 128)
            start_word_in_vmem = unsigned_floor_div(
                unsigned_mod(offset, bkv_sz), 128)
            words_to_transfer = update_lane_iters
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start

            def loop_body(i, states):
                curr_word_in_page, words_to_transfer, curr_word_in_vmem = states
                sz_words = jnp.minimum(page_size_per_lane - curr_word_in_page,
                                       words_to_transfer)
                page_idx = page_indices_ref[page_indices_offset + i]
                debug_print(
                    "[RPA debug] loop_idx={}, curr_word_in_page={}, words_to_transfer={}, curr_word_in_vmem={}, sz_words={}, page_idx={}",
                    i, curr_word_in_page, words_to_transfer, curr_word_in_vmem,
                    sz_words, page_idx)

                _async_copy(
                    # bkvc_vmem_ref shape:
                    # [lkv_dim_per_kv_packing, kv_packing, bkv_sz]
                    bkvc_vmem_ref.at[...,
                                     pl.ds(curr_word_in_vmem * 128, sz_words *
                                           128)],
                    updated_cache_kv_hbm_ref.at[page_idx, :nope_dim,
                                                pl.ds(curr_word_in_page *
                                                      128, sz_words * 128)],
                    sem,
                    wait=False,
                )
                _async_copy(
                    # bkvpe_vmem_ref shape:
                    # [r_dim_per_kv_packing, kv_packing, bkv_sz]
                    bkvpe_vmem_ref.at[:kv_r_dim,
                                      pl.ds(curr_word_in_vmem * 128, sz_words *
                                            128)],
                    updated_cache_kv_hbm_ref.at[page_idx, nope_dim:,
                                                pl.ds(curr_word_in_page *
                                                      128, sz_words * 128)],
                    sem,
                    wait=False,
                )
                return 0, words_to_transfer - sz_words, curr_word_in_vmem + sz_words

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (
                    start_word_in_page,
                    words_to_transfer,
                    start_word_in_vmem,
                ),  # initial states
                unroll=False,
            )
        else:  # Wait
            dma_sz_words = update_lane_iters * 128
            dst_kv_nope = bkvc_vmem_ref.at[..., pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv_nope,
                dst=dst_kv_nope,
                sem=sem,
                wait=True,
            )
            dst_kv_pe = bkvpe_vmem_ref.at[:kv_r_dim, pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv_pe,
                dst=dst_kv_pe,
                sem=sem,
                wait=True,
            )

    def _update_kv_cache(
        batch_start_seq_idx,
        b,
        bkv_sem_idx,
        offset,  # In units of tokens.
        update_sz,  # In units of tokens.
        *,
        wait=False,
    ):
        seq_idx = batch_start_seq_idx + b
        sem = sems.at[3, b, bkv_sem_idx]
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, r_dim]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

        update_kv_packing_iters = unsigned_cdiv(
            unsigned_mod(offset, kv_packing) + update_sz, kv_packing)

        # Expected shape:
        # [total_num_pages, page_size_per_kv_packing, kv_packing,
        # align_to(lkv_dim + r_dim, 128)]
        cache_kv_hbm_shape = updated_cache_kv_hbm_ref.shape
        reshaped_cache_kv_hbm_ref = updated_cache_kv_hbm_ref.reshape(
            cache_kv_hbm_shape[0] * cache_kv_hbm_shape[1],
            *cache_kv_hbm_shape[2:],
        )

        if not wait:
            # Issue DMA copy for the updated parts, page by page.
            kv_p_start = unsigned_floor_div(offset, page_size)
            kv_p_end = unsigned_cdiv(offset + update_sz, page_size)
            start_word_in_page = unsigned_floor_div(
                unsigned_mod(offset, page_size), kv_packing)
            start_word_in_vmem = unsigned_floor_div(
                unsigned_mod(offset, bkv_sz), kv_packing)
            words_to_transfer = update_kv_packing_iters
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start

            def loop_body(i, states):
                curr_word_in_page, words_to_transfer, curr_word_in_vmem = states
                sz_words = jnp.minimum(
                    page_size_per_kv_packing - curr_word_in_page,
                    words_to_transfer)
                page_idx = page_indices_ref[page_indices_offset + i]

                _async_copy(
                    # bkvc_vmem_ref shape:
                    # [bkv_sz_per_kv_packing+2, kv_packing, lkv_dim]
                    bkvc_vmem_ref.at[pl.ds(curr_word_in_vmem, sz_words)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(
                            page_idx * page_size_per_kv_packing +
                            curr_word_in_page,
                            sz_words,
                        ),
                        ...,
                        :nope_dim,
                    ],
                    sem,
                    wait=False,
                )
                _async_copy(
                    # bkvpe_vmem_ref shape: [bkv_sz_per_kv_packing+2, kv_packing, r_dim]
                    bkvpe_vmem_ref.at[pl.ds(curr_word_in_vmem, sz_words)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(
                            page_idx * page_size_per_kv_packing +
                            curr_word_in_page,
                            sz_words,
                        ),
                        ...,
                        nope_dim:,
                    ],
                    sem,
                    wait=False,
                )
                return 0, words_to_transfer - sz_words, curr_word_in_vmem + sz_words

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (
                    start_word_in_page,
                    words_to_transfer,
                    start_word_in_vmem,
                ),  # initial states
                unroll=False,
            )
        else:  # Wait
            dma_sz_words = update_kv_packing_iters
            # bkvc_vmem_ref shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
            dst_kv = bkvc_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )
            dst_kv = bkvpe_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        for b in range(batch_size):
            sem = sems.at[1, b, bq_sem_idx]
            bq_nope_vmem_ref = bq_nope_x2_ref.at[bq_sem_idx, b]
            bq_rope_vmem_ref = bq_rope_x2_ref.at[bq_sem_idx, b]

            seq_idx = batch_start_seq_idx + b
            is_valid_seq = seq_idx < end_seq_idx
            safe_seq_idx = jnp.minimum(seq_idx, max_num_seqs - 1)
            q_len_start = jnp.where(is_valid_seq, cu_q_lens_ref[safe_seq_idx] + bq_idx * bq_sz, 0)
            q_end = jnp.where(is_valid_seq, cu_q_lens_ref[safe_seq_idx + 1], 0)
            sz = jnp.minimum(bq_sz, q_end - q_len_start)

            debug_print(
                "[RPA debug]"
                f" -----------{'wait' if wait else 'start'}_fetch_bq-----------"
            )
            debug_print("[RPA debug] seq_idx={}", seq_idx)
            debug_print("[RPA debug] bq_idx={}", bq_idx)
            debug_print("[RPA debug] bq_sem_idx={}", bq_sem_idx)
            debug_print("[RPA debug] q_len_start={}", q_len_start)
            debug_print("[RPA debug] q_end={}", q_end)
            debug_print("[RPA debug] sz={}", sz)

            @pl.when(sz > 0)
            def _copy(q_len_start=q_len_start,
                      sz=sz,
                      bq_nope_vmem_ref=bq_nope_vmem_ref,
                      bq_rope_vmem_ref=bq_rope_vmem_ref,
                      sem=sem):
                _async_copy(
                    ql_nope_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_nope_vmem_ref.at[pl.ds(0, sz), :num_q_heads],
                    sem,
                    wait,
                )

                _async_copy(
                    q_pe_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_rope_vmem_ref.at[pl.ds(0, sz), :num_q_heads],
                    sem,
                    wait,
                )

    def _send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        for b in range(batch_size):
            sem = sems.at[2, b, bo_sem_idx]
            vmem_ref = bo_x2_ref.at[bo_sem_idx, b]

            seq_idx = batch_start_seq_idx + b
            is_valid_seq = seq_idx < end_seq_idx
            safe_seq_idx = jnp.minimum(seq_idx, max_num_seqs - 1)
            q_len_start = jnp.where(is_valid_seq, cu_q_lens_ref[safe_seq_idx] + bo_idx * bq_sz, 0)
            q_end = jnp.where(is_valid_seq, cu_q_lens_ref[safe_seq_idx + 1], 0)
            sz = jnp.minimum(bq_sz, q_end - q_len_start)

            debug_print(
                "[RPA debug]"
                f" -----------{'wait' if wait else 'start'}_send_bo-----------"
            )
            debug_print("[RPA debug] seq_idx={}", seq_idx)
            debug_print("[RPA debug] bo_idx={}", bo_idx)
            debug_print("[RPA debug] bo_sem_idx={}", bo_sem_idx)
            debug_print("[RPA debug] q_len_start={}", q_len_start)
            debug_print("[RPA debug] q_end={}", q_end)
            debug_print("[RPA debug] sz={}", sz)

            @pl.when(sz > 0)
            def _copy(vmem_ref=vmem_ref,
                      sz=sz,
                      q_len_start=q_len_start,
                      sem=sem):
                _async_copy(
                    vmem_ref.at[pl.ds(0, sz), :num_q_heads],
                    o_hbm_ref.at[pl.ds(q_len_start, sz)],
                    sem,
                    wait,
                )

    # TODO(b/506245022): Split out the KV handling functions into separate files.
    if not transpose_kv_cache:
        _fetch_bkv_fn = _fetch_bkv
    else:
        _fetch_bkv_fn = _fetch_transposed_bkv

    def start_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, *, bq_idx=0):
        return _fetch_bkv_fn(batch_start_seq_idx,
                             bkv_idx,
                             bkv_sem_idx,
                             bq_idx=bq_idx)

    def wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, *, bq_idx=0):
        return _fetch_bkv_fn(batch_start_seq_idx,
                             bkv_idx,
                             bkv_sem_idx,
                             bq_idx=bq_idx,
                             wait=True)

    def start_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = batch_start_seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_batch_start_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(
            jnp.logical_and(0 <= old_batch_start_seq_idx,
                            old_batch_start_seq_idx <= batch_start_seq_idx))
        def _():
            bo_ids_ref[bo_sem_idx] = -1
            _send_bo(old_batch_start_seq_idx,
                     old_bo_idx,
                     bo_sem_idx,
                     wait=True)

    # TODO(b/506245022): Split out the KV handling functions into separate files.
    def start_update_kv_cache(start_seq_idx, bkv_sem_idx, offsets, update_szs):
        for b in range(batch_size):
            offset = offsets[b]
            update_sz = update_szs[b]
            bkv_update_ids_ref[b, bkv_sem_idx + 4] = update_sz

            @pl.when(update_sz > 0)
            def _(b=b,
                  update_sz=update_sz,
                  start_seq_idx=start_seq_idx,
                  offset=offset):
                bkv_update_ids_ref[b, bkv_sem_idx] = start_seq_idx
                bkv_update_ids_ref[b, bkv_sem_idx + 2] = offset
                _update_kv_cache_fn = (_update_kv_cache
                                       if not transpose_kv_cache else
                                       _update_transposed_kv_cache)
                _update_kv_cache_fn(start_seq_idx, b, bkv_sem_idx, offset,
                                    update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        for b in range(batch_size):
            update_sz = bkv_update_ids_ref[b, bkv_sem_idx + 4]

            @pl.when(update_sz > 0)
            def _(b=b, update_sz=update_sz):
                start_seq_idx = bkv_update_ids_ref[b, bkv_sem_idx]
                offset = bkv_update_ids_ref[b, bkv_sem_idx + 2]
                bkv_update_ids_ref[b, bkv_sem_idx + 4] = 0
                _update_kv_cache_fn = (_update_kv_cache
                                       if not transpose_kv_cache else
                                       _update_transposed_kv_cache)
                _update_kv_cache_fn(start_seq_idx,
                                    b,
                                    bkv_sem_idx,
                                    offset,
                                    update_sz,
                                    wait=True)

    def load_batch_bq(bq_sem_idx, *, actual_bq_sz=bq_sz):
        q_nope_ref = (bq_nope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx].reshape(
                batch_size, bq_sz, padded_num_q_heads_per_q_packing, lkv_dim))
        q_nope_vec = pltpu.bitcast(
            q_nope_ref[:, :actual_bq_sz],
            q_dtype,
        )[:, :, :num_q_heads].reshape(batch_size, actual_bq_sz * num_q_heads,
                                      lkv_dim)
        q_rope_ref = (bq_rope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx].reshape(
                batch_size, bq_sz, padded_num_q_heads_per_q_packing, r_dim))
        q_rope_vec = pltpu.bitcast(
            q_rope_ref[:, :actual_bq_sz],
            q_dtype,
        )[:, :, :num_q_heads].reshape(batch_size, actual_bq_sz * num_q_heads,
                                      r_dim)
        if p_same_dtype_as_v and kv_dtype == jnp.float8_e4m3fn:
            q_nope_vec = q_nope_vec.astype(kv_dtype)
            q_rope_vec = q_rope_vec.astype(kv_dtype)
        return q_nope_vec, q_rope_vec

    def load_bq(batch_item_idx, bq_sem_idx, *, actual_bq_sz=bq_sz):
        q_nope_ref = (bq_nope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx, batch_item_idx].reshape(
                bq_sz, padded_num_q_heads_per_q_packing, lkv_dim))
        q_nope_vec = pltpu.bitcast(
            q_nope_ref[:actual_bq_sz],
            q_dtype,
        )[:, :num_q_heads].reshape(actual_bq_sz * num_q_heads, lkv_dim)
        q_rope_ref = (bq_rope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx, batch_item_idx].reshape(
                bq_sz, padded_num_q_heads_per_q_packing, r_dim))
        q_rope_vec = pltpu.bitcast(
            q_rope_ref[:actual_bq_sz],
            q_dtype,
        )[:, :num_q_heads].reshape(actual_bq_sz * num_q_heads, r_dim)
        if p_same_dtype_as_v and kv_dtype == jnp.float8_e4m3fn:
            q_nope_vec = q_nope_vec.astype(kv_dtype)
            q_rope_vec = q_rope_vec.astype(kv_dtype)
        return q_nope_vec, q_rope_vec

    def load_bq_slice(batch_item_idx, bq_sem_idx, q_idx, bq_sz_split):
        tok_start = q_idx * bq_sz_split
        q_nope_ref = (bq_nope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx, batch_item_idx, pl.ds(tok_start, bq_sz_split)].reshape(
                bq_sz_split, padded_num_q_heads_per_q_packing, lkv_dim))
        q_nope_vec = pltpu.bitcast(
            q_nope_ref[...],
            q_dtype,
        )[:, :num_q_heads].reshape(bq_sz_split * num_q_heads, lkv_dim)
        q_rope_ref = (bq_rope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx, batch_item_idx, pl.ds(tok_start, bq_sz_split)].reshape(
                bq_sz_split, padded_num_q_heads_per_q_packing, r_dim))
        q_rope_vec = pltpu.bitcast(
            q_rope_ref[...],
            q_dtype,
        )[:, :num_q_heads].reshape(bq_sz_split * num_q_heads, r_dim)
        if p_same_dtype_as_v and kv_dtype == jnp.float8_e4m3fn:
            q_nope_vec = q_nope_vec.astype(kv_dtype)
            q_rope_vec = q_rope_vec.astype(kv_dtype)
        return q_nope_vec, q_rope_vec

    def load_bkv(batch_item_idx, bkv_sem_idx, *, active_bkv_sz=bkv_sz):
        active_bkv_sz_per_kv_packing = active_bkv_sz // kv_packing
        bkvc_ref = (bkvc_x2_ref.bitcast(
            jnp.uint32).at[bkv_sem_idx,
                           batch_item_idx, :active_bkv_sz_per_kv_packing].reshape(
                               active_bkv_sz_per_kv_packing *
                               num_sublanes_for_kv_packing, lkv_dim))
        bkvc_vec = pltpu.bitcast(bkvc_ref[...],
                                 kv_dtype).reshape(active_bkv_sz, lkv_dim)
        bkpe_ref = (bkpe_x2_ref.bitcast(
            jnp.uint32).at[bkv_sem_idx,
                           batch_item_idx, :active_bkv_sz_per_kv_packing].reshape(
                               active_bkv_sz_per_kv_packing *
                               num_sublanes_for_kv_packing, r_dim))
        bkpe_vec = pltpu.bitcast(bkpe_ref[...],
                                 kv_dtype).reshape(active_bkv_sz, r_dim)
        return bkvc_vec, bkpe_vec

    def load_transposed_bkv(batch_item_idx, bkv_sem_idx):
        bkvc_vec = bkvc_x2_ref.at[bkv_sem_idx, batch_item_idx, :, :bkv_sz][...]
        bkpe_vec = bkpe_x2_ref.at[bkv_sem_idx, batch_item_idx, :, :bkv_sz][...]
        return bkvc_vec, bkpe_vec

    def load_batch_bkv(load_bkv_fn, bkv_sem_idx):
        bkvc_vecs = []
        bkpe_vecs = []
        for b in range(batch_size):
            bkvc_vec, bkpe_vec = load_bkv_fn(b, bkv_sem_idx)
            bkvc_vecs.append(bkvc_vec)
            bkpe_vecs.append(bkpe_vec)
        return jnp.stack(bkvc_vecs), jnp.stack(bkpe_vecs)

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = unsigned_align_to(shape[-1], src.shape[-1])
        # no-op concatenation.
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])],
            axis=-1)[..., :shape[-1]]

    def _get_valid_kv_len(s_idx):
        valid = s_idx < end_seq_idx
        safe_s_idx = jnp.minimum(s_idx, max_num_seqs - 1)
        return jnp.where(valid, kv_lens_ref[safe_s_idx], 0)

    def _get_valid_q_len(s_idx):
        valid = s_idx < end_seq_idx
        safe_s_idx = jnp.minimum(s_idx, max_num_seqs - 1)
        return jnp.where(
            valid, cu_q_lens_ref[safe_s_idx + 1] - cu_q_lens_ref[safe_s_idx], 0)

    kv_len_max = _get_valid_kv_len(batch_start_seq_idx)
    for b in range(1, batch_size):
        kv_len_b = _get_valid_kv_len(batch_start_seq_idx + b)
        kv_len_max = jnp.where(kv_len_b > kv_len_max, kv_len_b, kv_len_max)

    q_len_max = _get_valid_q_len(batch_start_seq_idx)
    for b in range(1, batch_size):
        q_len_b = _get_valid_q_len(batch_start_seq_idx + b)
        q_len_max = jnp.where(q_len_b > q_len_max, q_len_b, q_len_max)

    def process():
        num_bkv = unsigned_cdiv(kv_len_max, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = unsigned_cdiv(q_len_max, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = unsigned_cdiv(static_q_len, actual_bq_sz)

        debug_print("[RPA debug] process")
        debug_print("[RPA debug] num_bkv={}", num_bkv)  # num_bkv=3, bkv_sz=512
        debug_print("[RPA debug] bkv_sz={}", bkv_sz)
        debug_print("[RPA debug] num_bq={}", num_bq)
        debug_print("[RPA debug] kv_len_max={}", kv_len_max)
        debug_print("[RPA debug] q_len_max={}", q_len_max)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + batch_size,
                                      seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_cur_num_bkv(bq_idx):
            if not two_step_flash_attention:
                return num_bkv
            cur_kv_len = jnp.minimum(
                kv_len_max,
                (kv_len_max - q_len_max) + (bq_idx + 1) * actual_bq_sz,
            )
            return jnp.where(
                bq_idx == 0,
                num_bkv,
                jnp.maximum(1, unsigned_cdiv(cur_kv_len, bkv_sz)),
            )

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            cur_num_bkv = get_cur_num_bkv(bq_idx)
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == cur_num_bkv
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + batch_size,
                                      seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                batch_start_seq_idx, bq_idx, bq_sem_idx)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    batch_start_seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                is_transition_from_bq0 = jnp.logical_and(
                    num_bkv == 1,
                    jnp.logical_and(
                        next_seq_idx == batch_start_seq_idx,
                        jnp.logical_and(bq_idx == 0, next_bq_idx > 0)))

                # Prefetch next bkv
                @pl.when(jnp.logical_and(next_seq_idx < end_seq_idx, jnp.logical_not(is_transition_from_bq0)))
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx, bq_idx=next_bq_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                offsets, update_szs = wait_fetch_bkv(batch_start_seq_idx,
                                                     bkv_idx,
                                                     bkv_sem_idx,
                                                     bq_idx=bq_idx)

                # Pack and align new KVs in VMEM and write to cache_kv_hbm_ref.
                @pl.when(bq_idx == 0)
                def update_cur_bkv_to_cache():
                    if not transpose_kv_cache:
                        _pack_new_kv(bkv_sem_idx, offsets, update_szs)
                    else:
                        _pack_new_transposed_kv(bkv_sem_idx, bkv_idx, offsets,
                                                update_szs)
                    start_update_kv_cache(batch_start_seq_idx, bkv_sem_idx,
                                          offsets, update_szs)

                @pl.when(jnp.logical_and(next_seq_idx < end_seq_idx, is_transition_from_bq0))
                def prefetch_next_bkv_after_bq0():
                    wait_update_kv_cache(0)
                    wait_update_kv_cache(1)
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx,
                                    next_bkv_idx,
                                    next_bkv_sem_idx,
                                    bq_idx=next_bq_idx)

                if not transpose_kv_cache:
                    load_bkv_fn = load_bkv
                else:
                    load_bkv_fn = load_transposed_bkv

                if two_step_flash_attention:
                    debug_print("[RPA debug] two step flash attention")
                    prev_p = None
                    prev_m_prev = None
                    prev_m_curr = None
                    prev_p_rowsum = None
                    prev_v = None
                    prev_l_ref = None
                    prev_acc = None

                    q_len = actual_bq_sz * num_q_heads
                    q_block_len = q_len // q_split
                    bq_sz_for_each_split = bq_sz // q_split
                    chunk_sz_step1 = 1024 if bkv_sz >= 1024 and bkv_sz % 1024 == 0 else (512 if bkv_sz % 512 == 0 else 256)
                    hoisted_delta_iota = None
                    if bkv_sz % chunk_sz_step1 == 0 and q_block_len % num_q_heads == 0:
                        hoisted_delta_iota = (
                            lax.broadcasted_iota(jnp.int32, (q_block_len, chunk_sz_step1), 1) -
                            unsigned_floor_div(
                                lax.broadcasted_iota(jnp.int32, (q_block_len, chunk_sz_step1), 0),
                                num_q_heads)
                        )
                    for b in range(batch_size):
                        bkvc, bkpe = load_bkv_fn(
                            b,
                            bkv_sem_idx,
                        )
                        bq_nope_vec, bq_pe_vec = load_bq(
                            b, bq_sem_idx, actual_bq_sz=actual_bq_sz)

                        cu_q_len = _get_valid_q_len(batch_start_seq_idx + b)
                        kv_len_b = _get_valid_kv_len(batch_start_seq_idx + b)

                        def _run_q_split_for_b(active_bkv_sz):
                            prev_p_b = None
                            prev_m_prev_b = None
                            prev_m_curr_b = None
                            prev_p_rowsum_b = None
                            prev_v_b = None
                            prev_l_ref_b = None
                            prev_acc_b = None

                            bkvc_sub = bkvc if active_bkv_sz == bkv_sz else bkvc[:active_bkv_sz, :]
                            bkpe_sub = bkpe if active_bkv_sz == bkv_sz else bkpe[:active_bkv_sz, :]
                            bkvc_t_sub = None
                            if not transpose_kv_cache and active_bkv_sz % chunk_sz_step1 == 0:
                                num_c_sub = active_bkv_sz // chunk_sz_step1
                                bkvc_t_sub = [
                                    pltpu.bitcast(
                                        (bkvc_sub if num_c_sub == 1 else bkvc_sub[c_i * chunk_sz_step1:(c_i + 1) * chunk_sz_step1, :]).transpose((1, 0)),
                                        bkvc.dtype,
                                    )
                                    for c_i in range(num_c_sub)
                                ]

                            for q_idx in range(q_split):
                                bq_start = q_block_len * q_idx
                                bq_end = bq_start + q_block_len
                                bq_offset = bq_sz_for_each_split * q_idx

                                cur_p, cur_m_prev, cur_m_curr, cur_p_rowsum = flash_attention_step1(
                                    bq_nope_vec[bq_start:bq_end],
                                    bq_pe_vec[bq_start:bq_end],
                                    bkvc_sub,
                                    bkpe_sub,
                                    kv_c_t=bkvc_t_sub,
                                    precomputed_delta_iota=hoisted_delta_iota,
                                    kv_len=kv_len_b,
                                    q_len=cu_q_len,
                                    bq_idx=bq_idx,
                                    bq_offset=bq_offset,
                                    bkv_idx=bkv_idx,
                                    head_l_ref=l_ref.at[b, bq_start:bq_end],
                                    head_m_ref=m_ref.at[b, bq_start:bq_end],
                                )

                                if prev_p_b is not None:
                                    flash_attention_step2(
                                        prev_p_b,
                                        prev_m_prev_b,
                                        prev_m_curr_b,
                                        prev_p_rowsum_b,
                                        prev_v_b,
                                        bkv_idx=bkv_idx,
                                        head_l_ref=prev_l_ref_b,
                                        head_acc_ref=prev_acc_b,
                                    )

                                prev_p_b = cur_p
                                prev_m_prev_b = cur_m_prev
                                prev_m_curr_b = cur_m_curr
                                prev_p_rowsum_b = cur_p_rowsum
                                prev_v_b = bkvc_sub
                                prev_l_ref_b = l_ref.at[b, bq_start:bq_end]
                                prev_acc_b = acc_ref.at[b, bq_start:bq_end]

                            flash_attention_step2(
                                prev_p_b,
                                prev_m_prev_b,
                                prev_m_curr_b,
                                prev_p_rowsum_b,
                                prev_v_b,
                                bkv_idx=bkv_idx,
                                head_l_ref=prev_l_ref_b,
                                head_acc_ref=prev_acc_b,
                            )

                        if not transpose_kv_cache and bkv_sz == 2 * chunk_sz_step1:
                            cur_kv_horizon = jnp.minimum(
                                kv_len_b,
                                (kv_len_b - cu_q_len) + (bq_idx + 1) * actual_bq_sz,
                            )
                            need_second_half = (bkv_idx * bkv_sz + chunk_sz_step1) < cur_kv_horizon
                            lax.cond(
                                need_second_half,
                                lambda: _run_q_split_for_b(bkv_sz),
                                lambda: _run_q_split_for_b(chunk_sz_step1),
                            )
                        else:
                            _run_q_split_for_b(bkv_sz)
                else:  # One step flash attention
                    bkvc, bkpe = load_batch_bkv(
                        load_bkv_fn,
                        bkv_sem_idx,
                    )

                    bq_nope_vec, bq_pe_vec = load_batch_bq(
                        bq_sem_idx, actual_bq_sz=actual_bq_sz)

                    if debug_mode:
                        return

                    batch_flash_attention(
                        bq_nope_vec,
                        bq_pe_vec,
                        bkvc,
                        bkpe,
                        bq_idx=bq_idx,
                        bkv_idx=bkv_idx,
                    )

            cur_num_bkv = get_cur_num_bkv(bq_idx)
            lax.fori_loop(0, cur_num_bkv, compute_with_bkv, None, unroll=False)

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l_vec = l_ref[...]
            if q_dtype == jnp.float32:
                if v_scale is not None:
                    acc = acc * v_scale
                out = lax.div(acc, broadcast_minor(l_vec, acc.shape))
            else:
                inv_l = pl.reciprocal(l_vec, approx=True)
                if v_scale is not None:
                    inv_l = inv_l * v_scale
                out = (acc * broadcast_minor(inv_l, acc.shape)).astype(q_dtype)

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                batch_size,
                bq_sz,
                padded_num_q_heads_per_q_packing,
                lkv_dim,
            )[:, :, :num_q_heads_per_q_packing] = pltpu.bitcast(
                out.reshape(batch_size, bq_sz, num_q_heads, lkv_dim),
                jnp.int32)

            # Send cur bo
            start_send_bo(batch_start_seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(jnp.logical_and(batch_start_seq_idx == start_seq_idx, batch_start_seq_idx < end_seq_idx))
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)

        if not transpose_kv_cache:
            # Initialize bkvc_x2_ref and bkpe_x2_ref to zeros to avoid NaN issues from accessing
            # uninitialized memory. Bitcast into int32 to avoid tiling issues.
            bkvc_x2_int32_ref = bkvc_x2_ref.bitcast(jnp.int32).reshape(
                (2, -1, lkv_dim))
            bkvc_zeros = jnp.zeros(bkvc_x2_int32_ref.shape[1:], jnp.int32)
            bkpe_x2_int32_ref = bkpe_x2_ref.bitcast(jnp.int32).reshape(
                (2, -1, r_dim))
            bkpe_zeros = jnp.zeros(bkpe_x2_int32_ref.shape[1:], jnp.int32)

            # To pipeline VST and DMA, we divide the initialization into two steps.
            bkvc_x2_int32_ref[0] = bkvc_zeros
            bkpe_x2_int32_ref[0] = bkpe_zeros
            start_fetch_bkv(start_seq_idx, 0, 0, bq_idx=0)
            bkvc_x2_int32_ref[1] = bkvc_zeros
            bkpe_x2_int32_ref[1] = bkpe_zeros
        else:
            bkvc_zeros = jnp.zeros(bkvc_x2_ref.shape[1:], bkvc_x2_ref.dtype)
            bkpe_zeros = jnp.zeros(bkpe_x2_ref.shape[1:], bkpe_x2_ref.dtype)

            # To pipeline VST and DMA, we divide the initialization into two steps.
            bkvc_x2_ref[0] = bkvc_zeros
            bkpe_x2_ref[0] = bkpe_zeros
            start_fetch_bkv(start_seq_idx, 0, 0, bq_idx=0)
            bkvc_x2_ref[1] = bkvc_zeros
            bkpe_x2_ref[1] = bkpe_zeros

    @pl.when(batch_start_seq_idx < end_seq_idx)
    def run_process():
        process()

    @pl.when(jnp.logical_and(batch_start_seq_idx < end_seq_idx, batch_end_seq_idx >= end_seq_idx - 1))
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q.dtype))
    head_dim = align_to(actual_head_dim, 128)
    if num_q_heads == actual_num_q_heads and head_dim == actual_head_dim:
        return q
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q  # (max_num_tokens, num_q_heads, head_dim)


def prepare_q_nope_inputs(
    q: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_head_dim]
    vmem_limit_bytes: int | None = None,
):
    """Packs and physically transposes q_nope to the layout expected by the MLA kernel.

  The `q_nope` einsum emits in head-major layout (N, T, L). It needs to be transposed
  into (T, N, L) which uses the custom call kernel to absorb the copy latency.

  Returns: [max_num_tokens, num_q_heads, head_dim]
  """
    actual_num_q_heads, actual_max_num_tokens, actual_head_dim = q.shape
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q.dtype))
    head_dim = align_to(actual_head_dim, 128)

    # Align T to sublane_multiple (i.e. dtype_packing * 8).
    # This is needed for xpose_pipeline to work efficiently.
    sublane_multiple = get_dtype_packing(q.dtype) * 8
    max_num_tokens = align_to(actual_max_num_tokens, sublane_multiple)
    q = jnp.pad(
        q,
        (
            (0, num_q_heads - actual_num_q_heads),
            (0, max_num_tokens - actual_max_num_tokens),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    # Physical transpose: (N, T, D) -> (T, N, D), pipelined over T.
    try:
        q = xpose_pipeline(q,
                           transpose_axes=(1, 0, 2),
                           n_tile=128,
                           m_tile=32,
                           vmem_limit_bytes=vmem_limit_bytes)[0]
    except ValueError as e:
        logger.warning(
            f"xpose_pipeline failed for shape={q.shape} dtype={q.dtype} "
            f"(sublane_multiple={sublane_multiple}): {e}. "
            f"Falling back to jnp.transpose — this may be slower.")
        q = jnp.transpose(q, (1, 0, 2))
    return q  # (max_num_tokens, num_q_heads, head_dim)


def prepare_kv_inputs(kv: jax.Array, kv_packing: int):
    max_num_tokens, actual_head_dim = kv.shape
    # Pad to packing
    if max_num_tokens % kv_packing != 0:
        pad = kv_packing - (max_num_tokens % kv_packing)
        kv = jnp.pad(kv, ((0, pad), (0, 0)), constant_values=0)

    head_dim = unsigned_align_to(actual_head_dim, 128)
    kv = kv.reshape(-1, kv_packing, actual_head_dim)
    if head_dim != actual_head_dim:
        kv = jnp.pad(kv, ((0, 0), (0, 0), (0, head_dim - actual_head_dim)),
                     constant_values=0)
    return kv


def prepare_kv_inputs_for_transposed_kv_cache(kv: jax.Array):
    max_num_tokens, actual_head_dim = kv.shape

    # Pad to packing
    if max_num_tokens % 128 != 0:
        pad = 128 - (max_num_tokens % 128)
        kv = jnp.pad(kv, ((0, pad), (0, 0)), constant_values=0)

    if actual_head_dim % 128 != 0:
        pad = 128 - (actual_head_dim % 128)
        kv = jnp.pad(kv, ((0, 0), (0, pad)), constant_values=0)

    kv = kv.transpose((1, 0))
    return kv


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_max_num_tokens: int,
    actual_head_dim: int,
    vmem_limit_bytes: int | None = None,
):
    # Physical transpose: (T, N, D) -> (N, T, D), pipelined over T.
    try:
        # Tile to maximum of 160 (multi host bsz)
        # or nearest clean divisor of the number of tokens.
        # Use smaller tiles to avoid VMEM OOM (e.g. 128x32 instead of 160x64).
        out = xpose_pipeline(out,
                             transpose_axes=(1, 0, 2),
                             n_tile=_XPOSE_N_TILE_SIZE,
                             m_tile=64,
                             vmem_limit_bytes=vmem_limit_bytes)[0]
    except ValueError as e:
        sublane_multiple = get_dtype_packing(out.dtype) * 8
        logger.warning(
            f"xpose_pipeline failed for shape={out.shape} dtype={out.dtype} "
            f"(sublane_multiple={sublane_multiple}): {e}. "
            f"Falling back to jnp.transpose — this may be slower.")
        out = jnp.transpose(out, (1, 0, 2))
    return out[:actual_num_q_heads, :actual_max_num_tokens, :actual_head_dim]


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "decode_batch_size",
        "mixed_q_split",
        "s_dtype",
        "transpose_kv_cache",
        "two_step_flash_attention",
        "p_same_dtype_as_v",
        "ql_nope_is_token_major",
        "debug_mode",
    ),
    donate_argnames=("ql_nope", "cache_kv"),
)
def mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim, 128)] if not transpose_kv_cache else [total_num_pages, lkv_dim, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # If passsed in as int, all cases are the same.
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int | None = None,
    decode_batch_size: int = 1,
    mixed_q_split: int = 1,
    s_dtype: jnp.dtype = jnp.bfloat16,
    transpose_kv_cache: bool = False,
    two_step_flash_attention: bool = True,
    p_same_dtype_as_v: bool = True,
    ql_nope_is_token_major: bool = False,
    # Debug params.
    debug_mode: bool = False,
) -> tuple[
        jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim] if not ql_nope_is_token_major else [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
        jax.
        Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim, 128) + align_to(r_dim, 128)]
]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

  Args:
    ql_nope: concatenated all sequences' queries.
    q_pe: concatenated all sequences' rope.
    new_kv_c: concatenated all sequences' kv_c values
    new_k_pe: concatenated all sequences' k_pe values
    cache_kv: the current kv cache.
    kv_lens: the length of each sequence in the kv cache.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    q_scale: the scale for the query.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    num_queries_per_block: number of queries to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    decode_batch_size: the batch size for the decode case.
    mixed_q_split: the number of query split to run parallel in mixed case.
    s_dtype: the dtype for q.k dot product.
    transpose_kv_cache: if true, kernel assumes the kv cache is transposed.
    two_step_flash_attention: if true, kernel compute QK attention and
      AV attention in two separate steps to make them be more parallelized
      with VALU ops instead of batched matmul in one step.
    p_same_dtype_as_v: if true, cast p to the same dtype as v before p @ v.
    ql_nope_is_token_major: if true, ql_nope is expected in token-major (T, N, L) layout and output is returned in (T, N, L) layout without physical transpose.
    debug_mode: if true, RPA does not issue any DMAs or run flash attention but
      print debug info. Need to compile with `--xla_tpu_enable_log_recorder`.

  Returns:
    The output of attention and the updated kv cache.
  """
    if mask_value is None:
        mask_value = jnp.finfo(s_dtype).min

    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_blocks=num_kv_pages_per_blocks,
        num_queries_per_blocks=num_queries_per_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        decode_batch_size=decode_batch_size,
        mixed_q_split=mixed_q_split,
        transpose_kv_cache=transpose_kv_cache,
        ql_nope_is_token_major=ql_nope_is_token_major,
        debug_mode=debug_mode,
    )

    if ql_nope_is_token_major:
        actual_max_num_tokens, actual_num_q_heads, actual_lkv_dim = ql_nope.shape
        ql_nope = prepare_q_inputs(ql_nope)
    else:
        actual_num_q_heads, actual_max_num_tokens, actual_lkv_dim = ql_nope.shape
        ql_nope = prepare_q_nope_inputs(
            ql_nope,
            vmem_limit_bytes=vmem_limit_bytes,
        )  # [max_num_tokens, num_q_heads, lkv_dim]
    q_pe = prepare_q_inputs(q_pe)  # [max_num_tokens, num_q_heads, r_dim]
    if not transpose_kv_cache:
        _, _, kv_packing, _ = cache_kv.shape
        new_kv_c = prepare_kv_inputs(
            new_kv_c,
            kv_packing)  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim]
        new_k_pe = prepare_kv_inputs(
            new_k_pe,
            kv_packing)  # [max_num_tokens_per_kv_packing, kv_packing, r_dim]
        lkv_dim = new_kv_c.shape[-1]
        r_dim = new_k_pe.shape[-1]
    else:
        new_kv_c = prepare_kv_inputs_for_transposed_kv_cache(
            new_kv_c)  # [lkv_dim, max_num_tokens]
        new_k_pe = prepare_kv_inputs_for_transposed_kv_cache(
            new_k_pe)  # [r_dim, max_num_tokens]
        lkv_dim = new_kv_c.shape[0]
        r_dim = new_k_pe.shape[0]

    if not transpose_kv_cache:
        _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
        page_size = page_size_per_kv_packing * kv_packing
    else:
        _, _, page_size = cache_kv.shape
    _, num_q_heads, _ = ql_nope.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        ql_nope: jax.
        Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
        q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
        new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
        new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
        cache_kv: jax.
        Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim, 128)] if not transpose_kv_cache else [total_num_pages, lkv_dim, page_size]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        static_q_len: int | None,
        num_kv_pages_per_block: int,
        num_queries_per_block: int,
        s_dtype: jnp.dtype,
        p_same_dtype_as_v: bool,
        batch_size: int = 1,
        q_split: int = 1,
        case: MlaCase = MlaCase.MIXED,
    ):

        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        # Add 2 additional words of buffering to accommodate misaligned new KV.
        # We need two additional words because the beginning and end of the new KV may
        # both not be aligned to kv_packing boundaries.
        # Example:
        #
        # T0 T4     K2  K6
        # T1        K3
        # T2    K0  K4
        # T3    K1  K5
        #
        # - Ti is existing KV tokens and Ki is the new KV.
        # - Each column is a 32-bit word.
        # - KV packing is 4
        #
        # We have 12 total tokens, so normally we would only allocate 12/4=3 words
        # But due to misalignment, we need to allocate 5 words.
        grid = (unsigned_cdiv(end_seq_idx - start_seq_idx, batch_size), )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # ql_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # q_pe
            pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_c
            pl.BlockSpec(memory_space=pltpu.HBM),  # new_k_pe
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # o
            pl.BlockSpec(memory_space=pltpu.HBM),  # updated_cache_kv
        ]

        if not transpose_kv_cache:
            bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
            bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing + 2
            bkvc_double_buf = pltpu.VMEM(
                (2, batch_size, bkv_buf_sz_per_kv_packing, kv_packing,
                 lkv_dim),
                cache_kv.dtype,
            )

            bkpe_double_buf = pltpu.VMEM(
                (2, batch_size, bkv_buf_sz_per_kv_packing, kv_packing, r_dim),
                cache_kv.dtype,
            )
        else:
            bkv_sz = bkv_p * page_size
            padded_lkv_dim = unsigned_align_to(lkv_dim, 128)
            bkvc_double_buf = pltpu.VMEM(
                (2, batch_size, padded_lkv_dim, bkv_sz + 128 * 3),
                cache_kv.dtype,
            )

            padded_r_dim = unsigned_align_to(r_dim, 128)
            bkpe_double_buf = pltpu.VMEM(
                (2, batch_size, padded_r_dim, bkv_sz + 128 * 3),
                cache_kv.dtype,
            )

        q_packing = get_dtype_packing(ql_nope.dtype)
        padded_num_q_heads_per_q_packing = unsigned_align_to(
            num_q_heads // q_packing, 4)
        padded_num_q_heads = padded_num_q_heads_per_q_packing * q_packing
        bq_nope_double_buf = pltpu.VMEM(
            (2, batch_size, bq_sz, padded_num_q_heads, lkv_dim),
            ql_nope.dtype,
        )

        bq_rope_double_buf = pltpu.VMEM(
            (2, batch_size, bq_sz, padded_num_q_heads, r_dim),
            q_pe.dtype,
        )

        bo_double_buf = bq_nope_double_buf

        l_scratch = pltpu.VMEM(
            (batch_size, bq_sz * num_q_heads, 128),
            jnp.float32,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (batch_size, bq_sz * num_q_heads, lkv_dim),
            jnp.float32,
        )

        scratch_shapes = [
            bkvc_double_buf,
            bkpe_double_buf,
            bq_nope_double_buf,
            bq_rope_double_buf,
            bo_double_buf,  # Double buffering for output block.
            # Semaphores for double buffering of bkv, bq, bo and bkv_update.
            pltpu.SemaphoreType.DMA((4, batch_size, 2)),
            # Intermediate buffers per kv head for flash attention.
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
            jnp.zeros((3, ), jnp.int32),
            # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
            jnp.full((4, ), -1, jnp.int32),
            # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz) * batch_size
            jnp.full((batch_size, 6), -1, jnp.int32),
        )

        scope_name = f"MLA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}-bsz_{batch_size}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _mla_ragged_paged_attention_kernel,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    soft_cap=soft_cap,
                    mask_value=mask_value,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    batch_size=batch_size,
                    q_split=q_split,
                    s_dtype=s_dtype,
                    transpose_kv_cache=transpose_kv_cache,
                    two_step_flash_attention=two_step_flash_attention,
                    p_same_dtype_as_v=p_same_dtype_as_v,
                    debug_mode=debug_mode,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary", ),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=[
                    jax.ShapeDtypeStruct(shape=ql_nope.shape,
                                         dtype=ql_nope.dtype),
                    jax.ShapeDtypeStruct(shape=cache_kv.shape,
                                         dtype=cache_kv.dtype),
                ],
                input_output_aliases={
                    7: 0,  # Alias output activation with ql_nope
                    11: 1,  # Aliasing cache_kv with updated_cache_kv
                },
                name=scope_name,
            ))
        return kernel(
            *scalar_prefetches,
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
        )

    # Batched decode (handles full 0..distribution[0] range, including partial tail batch)
    ql_nope, updated_kv = run_mla_kernel(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=jnp.array(0),
        end_seq_idx=distribution[0],
        static_q_len=1,
        batch_size=decode_batch_size,
        s_dtype=s_dtype,
        p_same_dtype_as_v=p_same_dtype_as_v,
        case=MlaCase.BATCHED_DECODE,
    )

    if chunk_prefill_size is not None:
        # Handle prefill where the query length is fixed per sequence.
        ql_nope, updated_kv = run_mla_kernel(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            updated_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[1],
            num_queries_per_block=num_queries_per_blocks[1],
            start_seq_idx=distribution[0],
            end_seq_idx=distribution[1],
            static_q_len=chunk_prefill_size,
            batch_size=1,
            s_dtype=s_dtype,
            p_same_dtype_as_v=p_same_dtype_as_v,
            case=MlaCase.PREFILL,
        )

    # Handle mixed case where the query length per sequence is variable.
    ql_nope, updated_kv = run_mla_kernel(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        updated_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        static_q_len=None,
        batch_size=1,
        q_split=mixed_q_split,
        s_dtype=s_dtype,
        p_same_dtype_as_v=p_same_dtype_as_v,
        case=MlaCase.MIXED,
    )
    if ql_nope_is_token_major:
        output = ql_nope[
            :actual_max_num_tokens, :actual_num_q_heads, :actual_lkv_dim
        ]
    else:
        output = prepare_outputs(
            ql_nope,
            actual_num_q_heads,
            actual_max_num_tokens,
            actual_lkv_dim,
            vmem_limit_bytes=vmem_limit_bytes,
        )  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]

    return output, updated_kv
