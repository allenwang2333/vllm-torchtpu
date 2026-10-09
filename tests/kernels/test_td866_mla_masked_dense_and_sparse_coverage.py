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
"""CPU unit tests for TD8.66: MLA masked_dense and sparse kernel modules."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.masked_dense import kernel as md_kernel
from vllm_torchtpu.kernels.mla.sparse import kernel as sp_kernel


class _FakeAsyncCopy:
    """CPU-side mock for ``pltpu.make_async_copy``."""

    def __init__(self, src, dst, sem):
        self.src = src
        self.dst = dst
        self.sem = sem

    def start(self):
        if hasattr(self.dst, "_arr") and hasattr(self.src, "_arr"):
            src_val = self.src[...]
            dst_val = self.dst[...]
            if src_val.shape == dst_val.shape:
                self.dst[...] = src_val

    def wait(self):
        return None


class _FakeRef:
    """Mock Pallas Ref supporting `.at`, `.bitcast`, and `.reshape`."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self._arr = np.asarray(arr).copy()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._arr.shape

    @property
    def dtype(self):
        return jnp.dtype(self._arr.dtype)

    def _normalize_idx(self, idx):
        if isinstance(idx, tuple):
            return tuple(self._normalize_idx(i) for i in idx)
        if hasattr(idx, "start") and hasattr(idx, "size"):
            start = int(idx.start)
            size = int(idx.size)
            return slice(start, start + size)
        if isinstance(idx, jax.Array):
            if idx.ndim == 0:
                return int(idx)
            return np.asarray(idx)
        return idx

    @property
    def at(self):
        return _FakeRefAt(self)

    def bitcast(self, target_dtype):
        dt = jnp.dtype(target_dtype)
        src_bits = jax.dtypes.itemsize_bits(self.dtype)
        dst_bits = jax.dtypes.itemsize_bits(dt)
        new_shape = (
            *self._arr.shape[:-2],
            (self._arr.shape[-2] * src_bits) // dst_bits,
            self._arr.shape[-1],
        )
        return _FakeRef(np.zeros(new_shape, dtype=np.dtype(dt)))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _FakeRef(self._arr.reshape(shape))

    def __getitem__(self, idx):
        norm = self._normalize_idx(idx)
        return jnp.asarray(self._arr[norm])

    def __setitem__(self, idx, val):
        norm = self._normalize_idx(idx)
        self._arr[norm] = np.asarray(val)


class _FakeRefAt:
    def __init__(self, ref: _FakeRef):
        self._ref = ref

    def __getitem__(self, idx):
        norm = self._ref._normalize_idx(idx)
        return _FakeRef(self._ref._arr[norm])


def _fake_bitcast(x, target_dtype):
    dt = jnp.dtype(target_dtype)
    src_bits = jax.dtypes.itemsize_bits(x.dtype)
    dst_bits = jax.dtypes.itemsize_bits(dt)
    if src_bits == dst_bits:
        return jax.lax.bitcast_convert_type(x, dt)
    new_shape = (
        *x.shape[:-2],
        (x.shape[-2] * src_bits) // dst_bits,
        x.shape[-1],
    )
    return jnp.zeros(new_shape, dtype=dt)


def _fake_when(cond):
    def _deco(fn):
        if bool(np.all(np.asarray(cond))):
            fn()
        return fn

    return _deco


def _fake_fori_loop(lower, upper, body_fun, init_val, *, unroll=False):
    del unroll
    val = init_val
    for i in range(int(lower), int(upper)):
        val = body_fun(jnp.int32(i), val)
    return val


# ==============================================================================
# 1. Masked-Dense MLA Kernel Tests
# ==============================================================================


def test_masked_dense_helpers_and_prepare():
    assert md_kernel.MlaCase.DECODE.symbol == "d"
    assert md_kernel.MlaCase.PREFILL.symbol == "p"
    assert md_kernel.MlaCase.MIXED.symbol == "m"

    assert md_kernel.cdiv(10, 4) == 3
    assert md_kernel.align_to(10, 4) == 12
    assert md_kernel.get_dtype_bitwidth(jnp.bfloat16) == 16
    assert md_kernel.get_dtype_packing(jnp.bfloat16) == 2
    assert md_kernel.get_dtype_packing(jnp.float32) == 1

    p1 = md_kernel._get_compiler_params(1024)
    p2 = md_kernel._get_compiler_params(1024)
    assert p1 is p2

    # Unaligned q -> padded
    q_unaligned = jnp.ones((4, 3, 576), dtype=jnp.bfloat16)
    q_padded = md_kernel.prepare_q_inputs(q_unaligned)
    assert q_padded.shape == (4, 4, 640)
    out_sliced = md_kernel.prepare_outputs(q_padded, 3, 576)
    assert out_sliced.shape == (4, 3, 576)

    # Already aligned q -> fast-path identity
    q_aligned = jnp.ones((4, 4, 640), dtype=jnp.bfloat16)
    assert md_kernel.prepare_q_inputs(q_aligned) is q_aligned
    assert md_kernel.prepare_outputs(q_aligned, 4, 640) is q_aligned


def _run_masked_dense_kernel_sim(
    *,
    q_dtype,
    head_dim: int = 640,
    rope_cols: int = 64,
    l_cols: int = 128,
    static_q_len: int | None = 1,
    k_scale: float = 1.0,
    analytic_mask: bool = True,
    prog_id: int = 0,
    start_seq_idx: int = 0,
    end_seq_idx: int = 2,
    kv_len: int = 8,
    bkv_p: int = 1,
    page_size: int = 4,
    bq_sz: int = 1,
    bo_init_val: int = 0,
):
    q_packing = md_kernel.get_dtype_packing(q_dtype)
    num_q_heads = 4
    max_num_tokens = 4
    total_pages = 4
    bkv_sz = bkv_p * page_size

    kv_lens_ref = _FakeRef(np.array([kv_len, kv_len], dtype=np.int32))
    page_indices_ref = _FakeRef(np.arange(total_pages, dtype=np.int32))
    cu_q_lens_ref = _FakeRef(np.array([0, 2, 4], dtype=np.int32))
    start_end_seq_idx_ref = _FakeRef(
        np.array([start_seq_idx, end_seq_idx], dtype=np.int32)
    )
    sem_ids_ref = _FakeRef(np.zeros((3,), dtype=np.int32))
    bo_ids_ref = _FakeRef(np.full((4,), bo_init_val, dtype=np.int32))

    q_hbm_ref = _FakeRef(
        np.ones((max_num_tokens, num_q_heads, head_dim), dtype=jnp.dtype(q_dtype))
    )
    cache_kv_nope_hbm_ref = _FakeRef(
        np.zeros((total_pages, page_size, 4, 128), dtype=np.uint8)
    )
    cache_kv_rope_hbm_ref = _FakeRef(
        np.zeros((total_pages, page_size, rope_cols), dtype=np.uint8)
    )
    mask_hbm_ref = _FakeRef(np.ones((max_num_tokens, 2, bkv_sz), dtype=np.int32))
    o_hbm_ref = _FakeRef(
        np.zeros((max_num_tokens, num_q_heads, head_dim), dtype=jnp.dtype(q_dtype))
    )

    bkv_nope_x2_ref = _FakeRef(np.zeros((2, bkv_sz, 4, 128), dtype=np.uint8))
    bkv_rope_x2_ref = _FakeRef(np.zeros((2, bkv_sz, rope_cols), dtype=np.uint8))
    bq_x2_ref = _FakeRef(
        np.ones((2, bq_sz, num_q_heads, head_dim), dtype=jnp.dtype(q_dtype))
    )
    bo_x2_ref = _FakeRef(
        np.zeros(
            (2, bq_sz, num_q_heads // q_packing, q_packing, head_dim),
            dtype=jnp.dtype(q_dtype),
        )
    )
    mask_hbm_x2_ref = _FakeRef(np.ones((2, bq_sz, 1, bkv_sz), dtype=np.int32))
    sems = _FakeRef(np.zeros((5, 2), dtype=np.int32))
    l_ref = _FakeRef(np.ones((bq_sz * num_q_heads, l_cols), dtype=np.float32))
    m_ref = _FakeRef(np.zeros((bq_sz * num_q_heads, 128), dtype=np.float32))
    acc_ref = _FakeRef(np.zeros((bq_sz * num_q_heads, head_dim), dtype=np.float32))

    with (
        patch.object(md_kernel.pl, "program_id", return_value=jnp.int32(prog_id)),
        patch.object(md_kernel.pl, "when", side_effect=_fake_when),
        patch.object(
            md_kernel.pl,
            "reciprocal",
            side_effect=lambda x, approx=True: 1.0 / x,
        ),
        patch.object(md_kernel.pltpu, "make_async_copy", side_effect=_FakeAsyncCopy),
        patch.object(md_kernel.pltpu, "bitcast", side_effect=_fake_bitcast),
        patch.object(md_kernel.lax, "fori_loop", side_effect=_fake_fori_loop),
    ):
        md_kernel._mla_ragged_paged_attention_kernel(
            kv_lens_ref,
            page_indices_ref,
            cu_q_lens_ref,
            start_end_seq_idx_ref,
            sem_ids_ref,
            bo_ids_ref,
            q_hbm_ref,
            cache_kv_nope_hbm_ref,
            cache_kv_rope_hbm_ref,
            mask_hbm_ref,
            o_hbm_ref,
            bkv_nope_x2_ref,
            bkv_rope_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            mask_hbm_x2_ref,
            sems,
            l_ref,
            m_ref,
            acc_ref,
            sm_scale=0.5,
            k_scale=k_scale,
            static_q_len=static_q_len,
            mask_value=-1e9,
            bq_sz=bq_sz,
            bkv_p=bkv_p,
            analytic_mask=analytic_mask,
        )


def test_masked_dense_kernel_body_all_branches():
    # 1. bfloat16, static_q_len=1, k_scale=1.0, analytic_mask=True, prog_id=0
    _run_masked_dense_kernel_sim(
        q_dtype=jnp.bfloat16,
        static_q_len=1,
        k_scale=1.0,
        analytic_mask=True,
        prog_id=0,
        bo_init_val=0,
    )
    # 2. float32, static_q_len=None, k_scale=0.5, analytic_mask=False, prog_id=1,
    #    head_dim=640 with rope_cols=128 (512+128 == 640 -> bkv.shape[-1] == head_dim)
    #    and bkv_p=32, page_size=4 (bkv_sz=128 -> broadcast_minor identity).
    _run_masked_dense_kernel_sim(
        q_dtype=jnp.float32,
        head_dim=640,
        rope_cols=128,
        l_cols=128,
        static_q_len=None,
        k_scale=0.5,
        analytic_mask=False,
        prog_id=1,
        bkv_p=32,
        page_size=4,
        kv_len=128,
        bo_init_val=-1,
    )


def test_masked_dense_ragged_paged_attention_host_paths():
    q = jnp.ones((4, 4, 640), dtype=jnp.bfloat16)
    cache_kv_nope = jnp.zeros((4, 4, 4, 128), dtype=jnp.uint8)
    cache_kv_rope = jnp.zeros((4, 1, 4, 128), dtype=jnp.uint8)
    kv_lens = jnp.array([4, 4], dtype=jnp.int32)
    topk_indices = jnp.zeros((4, 8), dtype=jnp.int32)
    page_indices = jnp.arange(4, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)

    fn = md_kernel.masked_dense_ragged_paged_attention.__wrapped__

    with pytest.raises(
        ValueError,
        match="num_kv_pages_per_block and num_queries_per_block must be specified",
    ):
        fn(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            num_kv_pages_per_block=None,
            num_queries_per_block=1,
        )

    def _fake_pallas_call(kernel_fn, *, out_shape, **kwargs):
        del kernel_fn, kwargs

        def _runner(*args):
            # args[6] is q
            return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        return _runner

    with (
        patch.object(md_kernel.pl, "pallas_call", side_effect=_fake_pallas_call),
        patch.object(
            md_kernel.csa_mask,
            "generate_mask_sc",
            side_effect=lambda idx, max_kv: jnp.ones(
                (idx.shape[0], max_kv), dtype=jnp.int32
            ),
        ),
    ):
        # 1. Int block sizes, mask_value=None, max_kv_len=None (analytic_mask=True),
        #    sequence_start=None, chunk_prefill_size=2 (DECODE, PREFILL, MIXED).
        out1 = fn(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            mask_value=None,
            max_kv_len=None,
            chunk_prefill_size=2,
            num_kv_pages_per_block=1,
            num_queries_per_block=2,
        )
        assert out1.shape == (4, 4, 640)

        # 2. Tuple block sizes, explicit mask_value, max_kv_len=8 with topk=4
        #    (analytic_mask=False), sequence_start=None, chunk_prefill_size=None.
        topk_small = jnp.zeros((4, 4), dtype=jnp.int32)
        out2 = fn(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            topk_small,
            page_indices,
            cu_q_lens,
            distribution,
            mask_value=-1e9,
            max_kv_len=8,
            chunk_prefill_size=None,
            num_kv_pages_per_block=(1, 1, 1),
            num_queries_per_block=(1, 2, 2),
        )
        assert out2.shape == (4, 4, 640)

        # 3. sequence_start provided (skips decode and prefill, runs only mixed).
        out3 = fn(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            num_kv_pages_per_block=1,
            num_queries_per_block=2,
            sequence_start=jnp.array(1, dtype=jnp.int32),
        )
        assert out3.shape == (4, 4, 640)


# ==============================================================================
# 2. Sparse MLA Kernel Tests
# ==============================================================================


def test_sparse_helpers_and_gather_page_ids():
    assert sp_kernel.cdiv(10, 4) == 3
    assert sp_kernel.align_to(10, 4) == 12
    assert sp_kernel.get_dtype_bitwidth(jnp.bfloat16) == 16
    assert sp_kernel.get_dtype_packing(jnp.bfloat16) == 2
    assert sp_kernel._largest_divisor(64, 16) == 16
    assert sp_kernel._largest_divisor(10, 4) == 2
    assert sp_kernel._largest_divisor(0, 0) == 1
    assert sp_kernel._row_index_map(2) == (2, 0)

    p1 = sp_kernel._get_compiler_params(2048)
    p2 = sp_kernel._get_compiler_params(2048)
    assert p1 is p2

    # Direct test of _gather_page_ids_kernel
    windows_ref = _FakeRef(np.arange(2 * 256, dtype=np.int32).reshape(2, 256))
    logical_ref = _FakeRef(np.array([[0, 129], [5, 200]], dtype=np.int32))
    out_ref = _FakeRef(np.zeros((2, 2), dtype=np.int32))
    sp_kernel._gather_page_ids_kernel(windows_ref, logical_ref, out_ref, num_chunks=2)
    np.testing.assert_array_equal(
        out_ref[...], np.array([[0, 129], [256 + 5, 256 + 200]], dtype=np.int32)
    )

    def _fake_gather_pallas_call(kernel_fn, *, out_shape, **kwargs):
        del kwargs

        def _runner(windows, logical):
            w_ref = _FakeRef(windows)
            l_ref = _FakeRef(logical)
            o_ref = _FakeRef(np.zeros(out_shape.shape, dtype=np.int32))
            kernel_fn(w_ref, l_ref, o_ref)
            return o_ref[...]

        return _runner

    with patch.object(
        sp_kernel.pl, "pallas_call", side_effect=_fake_gather_pallas_call
    ):
        # Case A: padded_pps != pages_per_seq and padded_tokens != num_tokens
        page_indices = jnp.arange(8, dtype=jnp.int32)
        seq_page_ids = jnp.array([[0, 1], [2, 3], [0, 2]], dtype=jnp.int32)
        seq_ids_segment = jnp.array([0, 0, 1], dtype=jnp.int32)
        res_a = sp_kernel.gather_page_ids(
            page_indices, seq_page_ids, seq_ids_segment, max_num_seqs=2, block_tokens=4
        )
        assert res_a.shape == (3, 2)

        # Case B: padded_pps == pages_per_seq and padded_tokens == num_tokens
        page_indices_aligned = jnp.arange(256, dtype=jnp.int32)
        seq_page_ids_aligned = jnp.zeros((4, 2), dtype=jnp.int32)
        seq_ids_segment_aligned = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        res_b = sp_kernel.gather_page_ids(
            page_indices_aligned,
            seq_page_ids_aligned,
            seq_ids_segment_aligned,
            max_num_seqs=2,
            block_tokens=4,
        )
        assert res_b.shape == (4, 2)


def test_sparse_prepare_q_outputs_and_gather_indices():
    q_unaligned = jnp.ones((4, 3, 576), dtype=jnp.bfloat16)
    q_padded = sp_kernel.prepare_q_inputs(q_unaligned)
    assert q_padded.shape == (4, 4, 640)
    assert sp_kernel.prepare_outputs(q_padded, 3, 576).shape == (4, 3, 576)

    q_aligned = jnp.ones((4, 4, 640), dtype=jnp.bfloat16)
    assert sp_kernel.prepare_q_inputs(q_aligned) is q_aligned
    assert sp_kernel.prepare_outputs(q_aligned, 4, 640) is q_aligned

    topk_indices = jnp.array([[0, 1, -1, -1], [2, -1, -1, -1]], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 2], dtype=jnp.int32)

    # Error when page_indices is not divisible by max_num_seqs
    with pytest.raises(ValueError, match="must be divisible by the sequence count"):
        sp_kernel.prepare_gather_indices(
            topk_indices,
            jnp.arange(3, dtype=jnp.int32),
            cu_q_lens,
            page_size=4,
            total_num_pages=4,
        )

    # Valid with page_indices=None
    kv_lens_none, phys_none = sp_kernel.prepare_gather_indices(
        topk_indices,
        None,
        cu_q_lens,
        page_size=4,
        total_num_pages=4,
    )
    np.testing.assert_array_equal(kv_lens_none, np.array([2, 1], dtype=np.int32))
    assert phys_none.shape == (2, 4)

    # Valid with page_indices provided
    with patch.object(
        sp_kernel,
        "gather_page_ids",
        side_effect=lambda pi, sp, seg, mns: jnp.zeros_like(sp),
    ):
        kv_lens_paged, phys_paged = sp_kernel.prepare_gather_indices(
            topk_indices,
            jnp.arange(8, dtype=jnp.int32),
            cu_q_lens,
            page_size=4,
            total_num_pages=4,
        )
        np.testing.assert_array_equal(kv_lens_paged, np.array([2, 1], dtype=np.int32))
        assert phys_paged.shape == (2, 4)


def _run_sparse_attention_kernel_sim(
    *,
    q_dtype,
    head_dim: int = 640,
    rope_cols: int = 64,
    k_scale: float = 1.0,
    batch_size: int = 2,
    preserve_suffix: bool = False,
    return_lse: bool = False,
    sparsecore_layout: bool = False,
    prog_id: int = 0,
    num_progs: int = 1,
    start_seq_idx: int = 0,
    end_seq_idx: int = 4,
):
    num_tokens = 8
    num_q_heads = 4
    topk = 4

    kv_lens_ref = _FakeRef(np.full((num_tokens,), 2, dtype=np.int32))
    bounds = (
        [start_seq_idx, end_seq_idx, 3]
        if preserve_suffix
        else [start_seq_idx, end_seq_idx]
    )
    start_end_seq_idx_ref = _FakeRef(np.array(bounds, dtype=np.int32))
    sem_ids_ref = _FakeRef(np.zeros((2,), dtype=np.int32))

    q_hbm_ref = _FakeRef(
        np.ones((num_tokens, num_q_heads, head_dim), dtype=jnp.dtype(q_dtype))
    )
    if sparsecore_layout:
        cache_kv_nope_hbm_ref = _FakeRef(
            np.zeros((num_tokens, topk, 128), dtype=np.uint32)
        )
        cache_kv_rope_hbm_ref = _FakeRef(
            np.zeros((num_tokens, topk // 4, 128), dtype=np.uint32)
        )
        bkv_nope_x2_ref = _FakeRef(
            np.zeros((2, batch_size, topk, 128), dtype=np.uint32)
        )
        bkv_rope_x2_ref = _FakeRef(
            np.zeros((2, batch_size, topk // 4, 128), dtype=np.uint32)
        )
    else:
        cache_kv_nope_hbm_ref = _FakeRef(
            np.zeros((num_tokens, topk * 4, 128), dtype=np.uint8)
        )
        cache_kv_rope_hbm_ref = _FakeRef(
            np.zeros((num_tokens, topk, rope_cols), dtype=np.uint8)
        )
        bkv_nope_x2_ref = _FakeRef(
            np.zeros((2, batch_size, topk * 4, 128), dtype=np.uint8)
        )
        bkv_rope_x2_ref = _FakeRef(
            np.zeros((2, batch_size, topk, rope_cols), dtype=np.uint8)
        )

    o_hbm_ref = _FakeRef(
        np.zeros((num_tokens, num_q_heads, head_dim), dtype=jnp.dtype(q_dtype))
    )
    bq_x2_ref = _FakeRef(
        np.ones((2, batch_size, num_q_heads, head_dim), dtype=jnp.dtype(q_dtype))
    )
    bo_x2_ref = _FakeRef(
        np.zeros((2, batch_size, num_q_heads, head_dim), dtype=jnp.dtype(q_dtype))
    )

    if return_lse:
        lse_in_hbm_ref = _FakeRef(
            np.zeros((num_tokens, num_q_heads, 1), dtype=np.float32)
        )
        lse_hbm_ref = _FakeRef(np.zeros((num_tokens, num_q_heads, 1), dtype=np.float32))
        blse_x2_ref = _FakeRef(
            np.zeros((2, batch_size, num_q_heads, 1), dtype=np.float32)
        )
        sems = _FakeRef(np.zeros((5, 2), dtype=np.int32))
        refs = (
            lse_in_hbm_ref,
            o_hbm_ref,
            lse_hbm_ref,
            bkv_nope_x2_ref,
            bkv_rope_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            blse_x2_ref,
            sems,
        )
    else:
        sems = _FakeRef(np.zeros((4, 2), dtype=np.int32))
        refs = (
            o_hbm_ref,
            bkv_nope_x2_ref,
            bkv_rope_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            sems,
        )

    with (
        patch.object(sp_kernel.pl, "program_id", return_value=jnp.int32(prog_id)),
        patch.object(sp_kernel.pl, "num_programs", return_value=jnp.int32(num_progs)),
        patch.object(sp_kernel.pl, "when", side_effect=_fake_when),
        patch.object(
            sp_kernel.pl,
            "reciprocal",
            side_effect=lambda x, approx=True: 1.0 / x,
        ),
        patch.object(sp_kernel.pltpu, "make_async_copy", side_effect=_FakeAsyncCopy),
        patch.object(sp_kernel.pltpu, "bitcast", side_effect=_fake_bitcast),
    ):
        sp_kernel._attention_kernel(
            kv_lens_ref,
            start_end_seq_idx_ref,
            sem_ids_ref,
            q_hbm_ref,
            cache_kv_nope_hbm_ref,
            cache_kv_rope_hbm_ref,
            *refs,
            sm_scale=0.5,
            k_scale=k_scale,
            batch_size=batch_size,
            preserve_suffix=preserve_suffix,
            return_lse=return_lse,
            sparsecore_layout=sparsecore_layout,
        )


def test_sparse_attention_kernel_body_all_branches():
    # 1. TensorCore, batch_size=3 (hits prev_out is not None on batch_idx=2),
    #    return_lse=False, preserve_suffix=True, prog_id=0, num_progs=1, end_seq_idx=3
    _run_sparse_attention_kernel_sim(
        q_dtype=jnp.bfloat16,
        k_scale=1.0,
        batch_size=3,
        preserve_suffix=True,
        return_lse=False,
        sparsecore_layout=False,
        prog_id=0,
        num_progs=1,
        start_seq_idx=0,
        end_seq_idx=3,
    )
    # 2. TensorCore, return_lse=False, preserve_suffix=True, prog_id=2
    #    (hits _wait_send with preserve_suffix=True)
    _run_sparse_attention_kernel_sim(
        q_dtype=jnp.bfloat16,
        k_scale=1.0,
        batch_size=2,
        preserve_suffix=True,
        return_lse=False,
        sparsecore_layout=False,
        prog_id=2,
        num_progs=3,
        start_seq_idx=0,
        end_seq_idx=8,
    )
    # 3. SparseCore, return_lse=True, preserve_suffix=False, float32, k_scale=0.5,
    #    prog_id=2, num_progs=3, end_seq_idx=6 (hits _wait_send & epilogue)
    _run_sparse_attention_kernel_sim(
        q_dtype=jnp.float32,
        k_scale=0.5,
        batch_size=2,
        preserve_suffix=False,
        return_lse=True,
        sparsecore_layout=True,
        prog_id=2,
        num_progs=3,
        start_seq_idx=0,
        end_seq_idx=6,
    )
    # 4. TensorCore, rope_cols=128 (512+128 == 640 == head_dim),
    #    return_lse=False, preserve_suffix=False, prog_id=1, num_progs=2, end_seq_idx=4
    _run_sparse_attention_kernel_sim(
        q_dtype=jnp.bfloat16,
        head_dim=640,
        rope_cols=128,
        k_scale=1.0,
        batch_size=2,
        preserve_suffix=False,
        return_lse=False,
        sparsecore_layout=False,
        prog_id=1,
        num_progs=2,
        start_seq_idx=0,
        end_seq_idx=4,
    )


def test_sparse_ragged_paged_attention_validations_and_dispatch():
    fn = sp_kernel.sparse_ragged_paged_attention.__wrapped__

    q = jnp.ones((4, 4, 640), dtype=jnp.bfloat16)
    tc_nope = jnp.zeros((2, 4, 4, 128), dtype=jnp.uint8)
    tc_rope = jnp.zeros((2, 1, 4, 128), dtype=jnp.uint8)
    sc_nope = jnp.zeros((2, 4, 128), dtype=jnp.uint32)
    sc_rope = jnp.zeros((2, 1, 128), dtype=jnp.uint32)
    topk = jnp.zeros((4, 4), dtype=jnp.int32)
    page_indices = jnp.arange(4, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="cache_layout must be"):
        fn(
            q,
            tc_nope,
            tc_rope,
            topk,
            page_indices,
            cu_q_lens,
            distribution,
            cache_layout="bad",
        )

    with pytest.raises(ValueError, match="sparsecore NOPE cache must be"):
        fn(
            q,
            tc_nope,
            sc_rope,
            topk,
            page_indices,
            cu_q_lens,
            distribution,
            cache_layout="sparsecore",
        )

    with pytest.raises(ValueError, match="sparsecore ROPE cache must be"):
        fn(
            q,
            sc_nope,
            tc_rope,
            topk,
            page_indices,
            cu_q_lens,
            distribution,
            cache_layout="sparsecore",
        )

    with pytest.raises(ValueError, match="tensorcore NOPE cache must be"):
        fn(
            q,
            sc_nope,
            tc_rope,
            topk,
            page_indices,
            cu_q_lens,
            distribution,
            cache_layout="tensorcore",
        )

    with pytest.raises(ValueError, match="tensorcore ROPE cache must be"):
        fn(
            q,
            tc_nope,
            sc_rope,
            topk,
            page_indices,
            cu_q_lens,
            distribution,
            cache_layout="tensorcore",
        )

    with pytest.raises(ValueError, match="NOPE cache size"):
        fn(
            q,
            tc_nope,
            jnp.zeros((2, 2, 4, 128), dtype=jnp.uint8),
            topk,
            page_indices,
            cu_q_lens,
            distribution,
        )

    with pytest.raises(ValueError, match="matching page counts and page geometry"):
        fn(
            q,
            jnp.zeros((2, 8, 4, 128), dtype=jnp.uint8),
            jnp.zeros((4, 1, 4, 128), dtype=jnp.uint8),
            topk,
            page_indices,
            cu_q_lens,
            distribution,
        )

    with pytest.raises(ValueError, match="topk_indices must be int32"):
        fn(
            q,
            tc_nope,
            tc_rope,
            jnp.zeros((4, 4), dtype=jnp.float32),
            page_indices,
            cu_q_lens,
            distribution,
        )

    with pytest.raises(ValueError, match="q tokens .* must match topk_indices tokens"):
        fn(
            q,
            tc_nope,
            tc_rope,
            jnp.zeros((2, 4), dtype=jnp.int32),
            page_indices,
            cu_q_lens,
            distribution,
        )

    with pytest.raises(ValueError, match="for sparsecore ROPE packing"):
        fn(
            q,
            sc_nope,
            sc_rope,
            jnp.zeros((4, 6), dtype=jnp.int32),
            page_indices,
            cu_q_lens,
            distribution,
            cache_layout="sparsecore",
        )

    with pytest.raises(
        ValueError, match="Prefix stores require TensorCore caches without LSE"
    ):
        sp_kernel.sparse_attention_chunk(
            q,
            tc_nope,
            tc_rope,
            jnp.ones((4,), dtype=jnp.int32),
            topk,
            jnp.zeros((4, 4, 1), dtype=jnp.float32),
            start_pos=0,
            end_pos=4,
            token_end=jnp.int32(4),
            sm_scale=1.0,
            k_scale=1.0,
            preserve_suffix=True,
        )

    def _fake_attn_pallas_call(kernel_fn, *, out_shape, **kwargs):
        del kernel_fn, kwargs

        def _runner(*args):
            del args
            if isinstance(out_shape, list):
                return [jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape]
            return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        return _runner

    with (
        patch.object(sp_kernel.pl, "pallas_call", side_effect=_fake_attn_pallas_call),
        patch.object(
            sp_kernel.dsa_gather,
            "dsa_gather",
            side_effect=lambda nope, rope, idx, num_valid_indices: (
                jnp.zeros((idx.shape[0] * 4, 128), dtype=jnp.uint8),
                jnp.zeros((idx.shape[0], 64), dtype=jnp.uint8),
            ),
        ),
        patch.object(
            sp_kernel.native_sc_gather,
            "dsa_gather_native_sc",
            side_effect=lambda nope, rope, idx, **kw: (
                jnp.zeros((idx.size, 128), dtype=jnp.uint32),
                jnp.zeros((idx.size // 4, 128), dtype=jnp.uint32),
            ),
        ),
    ):
        # 1. TensorCore, return_lse=False, page_indices=None
        out_tc = fn(
            q,
            tc_nope,
            tc_rope,
            topk,
            None,
            cu_q_lens,
            distribution,
            return_lse=False,
            cache_layout="tensorcore",
            gather_and_attention_chunk_size=2,
            attention_kernel_batch_size=2,
        )
        assert out_tc.shape == (4, 4, 640)

        # 2. SparseCore, return_lse=True, page_indices=None
        out_sc, lse_sc = fn(
            q,
            sc_nope,
            sc_rope,
            topk,
            None,
            cu_q_lens,
            distribution,
            return_lse=True,
            cache_layout="sparsecore",
            gather_and_attention_chunk_size=2,
            attention_kernel_batch_size=2,
        )
        assert out_sc.shape == (4, 4, 640)
        assert lse_sc.shape == (4, 4)

        # 3. sparse_attention_chunk with preserve_suffix=True
        out_prefix = sp_kernel.sparse_attention_chunk(
            q,
            tc_nope,
            tc_rope,
            jnp.ones((4,), dtype=jnp.int32),
            topk,
            None,
            start_pos=0,
            end_pos=4,
            token_end=jnp.int32(3),
            sm_scale=1.0,
            k_scale=1.0,
            preserve_suffix=True,
            attention_kernel_batch_size=2,
        )
        assert out_prefix.shape == (4, 4, 640)
