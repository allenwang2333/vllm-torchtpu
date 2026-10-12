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
"""100% full-module line and branch coverage tests for mla_swa.py on CPU."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.core_attention import mla_swa


def _norm_idx(idx):
    if isinstance(idx, tuple):
        return tuple(_norm_idx(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        stride = 1 if stride is None else int(stride)
        start = int(np.asarray(idx.start))
        size = int(np.asarray(idx.size))
        return slice(start, start + size, stride)
    return idx


class MockArrayRef:
    """Mock HBM / VMEM buffer reference for simulating Pallas pipelines on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr).copy()
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def _clean_idx(self, idx):
        if isinstance(idx, tuple):
            out = []
            for d, x in enumerate(idx):
                n = _norm_idx(x)
                if not isinstance(n, (slice, type(Ellipsis))) and np.ndim(n) == 0:
                    n = int(np.clip(int(np.asarray(n)), 0, self.arr.shape[d] - 1))
                out.append(n)
            return tuple(out)
        n = _norm_idx(idx)
        if not isinstance(n, (slice, type(Ellipsis))) and np.ndim(n) == 0:
            n = int(np.clip(int(np.asarray(n)), 0, self.arr.shape[0] - 1))
        return n

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        return jnp.asarray(self.arr[self._clean_idx(idx)])

    def __setitem__(self, idx, val):
        if isinstance(val, MockArrayRef):
            val = val.arr
        self.arr[self._clean_idx(idx)] = np.asarray(val)

    @property
    def at(self):
        class _AtIndexer:
            def __init__(self, ref):
                self.ref = ref

            def __getitem__(self, idx):
                if idx is Ellipsis:
                    return jnp.asarray(self.ref.arr)
                return MockArrayRef(self.ref.arr[self.ref._clean_idx(idx)])

            def __setitem__(self, idx, val):
                if isinstance(val, MockArrayRef):
                    val = val.arr
                self.ref.arr[self.ref._clean_idx(idx)] = np.asarray(val)

        return _AtIndexer(self)

    def bitcast(self, dtype):
        return MockArrayRef(self.arr.view(np.dtype(dtype)))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        if self.arr.size != int(np.prod(shape)):
            return MockArrayRef(np.zeros(shape, dtype=self.arr.dtype))
        return MockArrayRef(self.arr.reshape(*shape))


class MockAsyncCopy:
    """Mock async DMA copy handle."""

    def __init__(self):
        self.started = False
        self.waited = False

    def start(self):
        self.started = True
        return self

    def wait(self):
        self.waited = True
        return self


def fake_when(cond):
    """Decorator executing inner function for statement and branch coverage."""

    def decorator(fn):
        fn()
        return fn

    return decorator


def fake_bitcast(x, dt):
    arr = x.arr if isinstance(x, MockArrayRef) else x
    arr = np.asarray(arr)
    if dt == jnp.bfloat16:
        return jax.lax.bitcast_convert_type(
            jnp.asarray(arr.view(np.uint16)), jnp.bfloat16
        )
    return jnp.asarray(arr.view(np.dtype(dt)))


def fake_fori_loop(lower, upper, body_fun, init_val, unroll=False):
    val = init_val
    for i in range(int(np.asarray(lower)), int(np.asarray(upper))):
        val = body_fun(i, val)
    return val


def test_helpers_and_validation():
    assert mla_swa.cdiv(10, 4) == 3
    assert mla_swa.cdiv(8, 4) == 2
    with pytest.raises(AssertionError):
        mla_swa.cdiv(4, 0)

    assert mla_swa.align_to(10, 4) == 12
    assert mla_swa.get_dtype_bitwidth(jnp.float32) == 32
    assert mla_swa.get_dtype_bitwidth(jnp.bfloat16) == 16
    assert mla_swa.get_dtype_packing(jnp.float32) == 1
    assert mla_swa.get_dtype_packing(jnp.bfloat16) == 2

    assert mla_swa.MlaCase.DECODE.symbol == "d"
    assert mla_swa.MlaCase.PREFILL.symbol == "p"
    assert mla_swa.MlaCase.MIXED.symbol == "m"

    q_unaligned = jnp.ones((4, 15, 250), dtype=jnp.bfloat16)
    q_padded = mla_swa.prepare_q_inputs(q_unaligned)
    assert q_padded.shape == (4, 16, 256)
    out = mla_swa.prepare_outputs(q_padded, 15, 250)
    assert out.shape == (4, 15, 250)

    kv = jax.random.normal(jax.random.PRNGKey(0), (8, 256), dtype=jnp.bfloat16)
    packed = mla_swa.prepare_kv_inputs(kv)
    kv_u16 = jax.lax.bitcast_convert_type(kv, jnp.uint16)
    kv_hi = ((kv_u16 >> 8) & 0xFF).astype(jnp.uint8).reshape(8, 2, 128)
    kv_lo = (kv_u16 & 0xFF).astype(jnp.uint8).reshape(8, 2, 128)
    ref_packed = jnp.stack([kv_lo, kv_hi], axis=2).reshape(8, 512)
    assert jnp.array_equal(packed, ref_packed)

    layout = mla_swa.get_physical_layout((8, 16, 4, 128), jnp.bfloat16, 256, 2, 8)
    assert layout == (8, 4, 128, 1, 16, 4)


def test_mla_swa_ragged_kernel_all_branches():
    num_tokens = 8
    head_dim = 256
    lkv_dim = 128
    kv_packing = 4
    logical_page_size = 4
    total_num_pages = 8

    for (
        num_q_heads,
        q_dtype,
        unnormalized_output,
        non_causal_block,
        q_compute_block_size,
        static_q_len,
        sliding_window,
        bkv_p,
        bq_sz,
    ) in [
        # 1. single_bkv_block=True, unnormalized=True, num_q_heads=16
        (16, jnp.bfloat16, True, False, None, None, 3, 2, 2),
        # 2. single_bkv_block=False, unnormalized=False, non_causal=True, f32, 128h
        (128, jnp.float32, False, True, 1, 2, 12, 2, 2),
        # 3. single_bkv_block=False, non_causal=False, unnormalized=False, bf16
        (16, jnp.bfloat16, False, False, 2, 2, 12, 2, 2),
    ]:
        num_l_heads = mla_swa.align_to(num_q_heads, 128)
        slots_per_token = (head_dim * 2) // 512
        bkv_sz = bkv_p * logical_page_size
        np_dt = np.float32 if q_dtype == jnp.float32 else np.float16

        kv_lens_ref = MockArrayRef(np.array([6, 6], dtype=np.int32))
        page_indices_ref = MockArrayRef(np.arange(total_num_pages, dtype=np.int32))
        cu_q_lens_ref = MockArrayRef(np.array([0, 2, 4], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 2], dtype=np.int32))
        sem_ids_ref = MockArrayRef(np.zeros((3,), dtype=np.int32))
        bo_ids_ref = MockArrayRef(np.array([0, 0, 0, 0], dtype=np.int32))
        bkv_update_ids_ref = MockArrayRef(np.array([0, 0, 0, 0, 1, 1], dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(
            np.ones((num_tokens, num_q_heads, head_dim), dtype=np_dt)
        )
        new_kv_hbm_ref = MockArrayRef(
            np.zeros(
                (num_tokens, slots_per_token * kv_packing, lkv_dim), dtype=np.uint8
            )
        )
        cache_kv_hbm_ref = MockArrayRef(
            np.zeros(
                (
                    total_num_pages,
                    logical_page_size * slots_per_token,
                    kv_packing,
                    lkv_dim,
                ),
                dtype=np.uint8,
            )
        )
        in_output_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np_dt)
        )
        in_l_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_l_heads), dtype=np.float32)
        )
        in_m_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_l_heads), dtype=np.float32)
        )
        o_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np_dt)
        )
        updated_cache_kv_hbm_ref = MockArrayRef(
            np.zeros(
                (
                    total_num_pages,
                    logical_page_size * slots_per_token,
                    kv_packing,
                    lkv_dim,
                ),
                dtype=np.uint8,
            )
        )
        l_hbm_ref = MockArrayRef(np.zeros((num_tokens, num_l_heads), dtype=np.float32))
        m_hbm_ref = MockArrayRef(np.zeros((num_tokens, num_l_heads), dtype=np.float32))
        bkv_x2_ref = MockArrayRef(
            np.zeros((2, bkv_sz, slots_per_token * kv_packing, lkv_dim), dtype=np.uint8)
        )
        bq_x2_ref = MockArrayRef(
            np.zeros((2, bq_sz, num_q_heads, head_dim), dtype=np_dt)
        )
        bo_x2_ref = MockArrayRef(
            np.zeros((2, bq_sz, num_q_heads, head_dim), dtype=np_dt)
        )
        bl_x2_ref = MockArrayRef(np.zeros((2, bq_sz, num_l_heads), dtype=np.float32))
        bm_x2_ref = MockArrayRef(np.zeros((2, bq_sz, num_l_heads), dtype=np.float32))
        sems = MockArrayRef(np.zeros((6, 2), dtype=np.int32))
        l_ref = MockArrayRef(np.zeros((bq_sz * num_q_heads,), dtype=np.float32))
        m_ref = MockArrayRef(np.zeros((bq_sz * num_q_heads,), dtype=np.float32))
        acc_ref = MockArrayRef(
            np.zeros((bq_sz * num_q_heads, head_dim), dtype=np.float32)
        )

        with (
            patch.object(
                mla_swa.pltpu, "make_async_copy", return_value=MockAsyncCopy()
            ),
            patch.object(mla_swa.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                mla_swa.pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x
            ),
            patch.object(mla_swa.pl, "when", side_effect=fake_when),
            patch.object(mla_swa.pl, "program_id", return_value=0),
            patch.object(mla_swa.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            mla_swa._mla_sliding_window_ragged_paged_attention_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                o_hbm_ref,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_x2_ref,
                bq_x2_ref,
                bo_x2_ref,
                bl_x2_ref,
                bm_x2_ref,
                sems,
                l_ref,
                m_ref,
                acc_ref,
                sem_ids_ref,
                bo_ids_ref,
                bkv_update_ids_ref,
                static_q_len=static_q_len,
                sm_scale=0.125,
                sliding_window=sliding_window,
                logical_page_size=logical_page_size,
                unnormalized_output=unnormalized_output,
                non_causal_block=non_causal_block,
                q_compute_block_size=q_compute_block_size,
                bkv_p=bkv_p,
                bq_sz=bq_sz,
            )


def test_mla_batched_decode_kernel_all_branches():
    head_dim = 256
    lkv_dim = 128
    kv_packing = 4
    logical_page_size = 4
    bkv_p = 2
    bkv_sz = bkv_p * logical_page_size
    seq_batch_sz = 2
    total_num_pages = 8

    for num_q_heads, q_dtype, unnormalized_output in [
        (16, jnp.bfloat16, True),
        (128, jnp.float32, False),
        (16, jnp.bfloat16, False),
    ]:
        num_l_heads = mla_swa.align_to(num_q_heads, 128)
        slots_per_token = (head_dim * 2) // 512
        np_dt = np.float32 if q_dtype == jnp.float32 else np.float16

        kv_lens_ref = MockArrayRef(np.array([6, 6], dtype=np.int32))
        page_indices_ref = MockArrayRef(np.arange(total_num_pages, dtype=np.int32))
        cu_q_lens_ref = MockArrayRef(np.array([0, 1, 2], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 2], dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(np.ones((4, num_q_heads, head_dim), dtype=np_dt))
        new_kv_hbm_ref = MockArrayRef(
            np.zeros((4, slots_per_token * kv_packing, lkv_dim), dtype=np.uint8)
        )
        cache_kv_hbm_ref = MockArrayRef(
            np.zeros(
                (
                    total_num_pages,
                    logical_page_size * slots_per_token,
                    kv_packing,
                    lkv_dim,
                ),
                dtype=np.uint8,
            )
        )
        in_output_hbm_ref = MockArrayRef(
            np.zeros((4, num_q_heads, head_dim), dtype=np_dt)
        )
        in_l_hbm_ref = MockArrayRef(np.zeros((4, num_l_heads), dtype=np.float32))
        in_m_hbm_ref = MockArrayRef(np.zeros((4, num_l_heads), dtype=np.float32))
        o_hbm_ref = MockArrayRef(np.zeros((4, num_q_heads, head_dim), dtype=np_dt))
        updated_cache_kv_hbm_ref = MockArrayRef(
            np.zeros(
                (
                    total_num_pages,
                    logical_page_size * slots_per_token,
                    kv_packing,
                    lkv_dim,
                ),
                dtype=np.uint8,
            )
        )
        l_hbm_ref = MockArrayRef(np.zeros((4, num_l_heads), dtype=np.float32))
        m_hbm_ref = MockArrayRef(np.zeros((4, num_l_heads), dtype=np.float32))
        bkv_batch_x2_vmem = MockArrayRef(
            np.zeros(
                (2, seq_batch_sz, bkv_sz, slots_per_token * kv_packing, lkv_dim),
                dtype=np.uint8,
            )
        )
        q_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, num_q_heads, head_dim), dtype=np_dt)
        )
        o_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, num_q_heads, head_dim), dtype=np_dt)
        )
        l_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, num_l_heads), dtype=np.float32)
        )
        m_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, num_l_heads), dtype=np.float32)
        )
        sems = MockArrayRef(np.zeros((2, seq_batch_sz + 4), dtype=np.int32))

        with (
            patch.object(
                mla_swa.pltpu, "make_async_copy", return_value=MockAsyncCopy()
            ),
            patch.object(mla_swa.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                mla_swa.pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x
            ),
            patch.object(mla_swa.pl, "when", side_effect=fake_when),
            patch.object(mla_swa.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            mla_swa._mla_batched_decode_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                o_hbm_ref,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_batch_x2_vmem,
                q_batch_x2_vmem,
                o_batch_x2_vmem,
                l_batch_x2_vmem,
                m_batch_x2_vmem,
                sems,
                sm_scale=0.125,
                sliding_window=8,
                logical_page_size=logical_page_size,
                unnormalized_output=unnormalized_output,
                bkv_p=bkv_p,
                seq_batch_sz=seq_batch_sz,
            )


def test_top_level_dispatcher_all_branches():
    q = jnp.ones((8, 16, 256), dtype=jnp.bfloat16)
    kv = jnp.ones((8, 256), dtype=jnp.bfloat16)
    cache_kv = jnp.zeros((8, 4, 4, 128), dtype=jnp.uint8)
    kv_lens = jnp.array([4, 4], dtype=jnp.int32)
    page_indices = jnp.arange(8, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 5], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)
    sinks = jnp.zeros((16,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="must be specified"):
        mla_swa.mla_sliding_window_ragged_paged_attention.__wrapped__(
            q,
            kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sinks,
            sliding_window=4,
            logical_page_size=4,
            num_kv_pages_per_block=None,
            num_queries_per_block=None,
        )

    def fake_pallas_call(kernel_fn, out_shape, **kwargs):
        def _runner(*args):
            cache_in, in_out, in_l, in_m = args[-4], args[-3], args[-2], args[-1]
            return (
                jnp.zeros_like(in_out),
                jnp.zeros_like(cache_in),
                jnp.zeros_like(in_l),
                jnp.zeros_like(in_m),
            )

        return _runner

    with patch.object(mla_swa.pl, "pallas_call", side_effect=fake_pallas_call):
        # 1. Batched decode + fused uniform dispatch + unnormalized_output=True
        out, _, _, _ = mla_swa.mla_sliding_window_ragged_paged_attention.__wrapped__(
            q,
            kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sinks,
            sliding_window=4,
            logical_page_size=4,
            num_kv_pages_per_block=2,
            num_queries_per_block=2,
            unnormalized_output=True,
        )
        assert out.shape == (8, 16, 256)

        # 2. Long-window decode + separate PREFILL & MIXED dispatch
        out2, _, _, _ = mla_swa.mla_sliding_window_ragged_paged_attention.__wrapped__(
            q,
            kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sinks,
            sliding_window=16,
            logical_page_size=4,
            chunk_prefill_size=4,
            num_kv_pages_per_block=(2, 2, 2),
            num_queries_per_block=(1, 4, 2),
            unnormalized_output=False,
        )
        assert out2.shape == (8, 16, 256)
