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
"""100% Statement and Branch Coverage Tests for PCP Streaming RPA Multi-Head Kernel.

Target: src/vllm_torchtpu/kernels/experimental/pcp_streaming_rpa/kernel.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PJRT_DEVICE", "cpu")

from vllm_torchtpu.kernels.experimental.batched_rpa import utils as batched_utils
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import kernel
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    RuntimeScheduleField,
    TilePlanField,
)

KVLayout = kernel.KVLayout


@pytest.fixture(autouse=True)
def _mock_tpu_info(monkeypatch):
    info = SimpleNamespace(
        vmem_capacity_bytes=16 * 1024 * 1024,
        num_lanes=128,
        num_sublanes=8,
    )
    monkeypatch.setattr(kernel.pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(batched_utils.pltpu, "get_tpu_info", lambda: info)


# ==============================================================================
# Simulation Primitives & Mock Helpers
# ==============================================================================


def _norm_idx(idx, arr_ndim=None):
    if isinstance(idx, tuple):
        if arr_ndim is not None and len(idx) > arr_ndim:
            idx = idx[:arr_ndim]
        return tuple(_norm_idx(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        stride = 1 if stride is None else int(stride)
        start = int(np.asarray(idx.start))
        size = int(np.asarray(idx.size))
        return slice(start, start + size, stride)
    if isinstance(idx, (jax.Array, np.ndarray)) and idx.ndim == 0:
        return int(idx)
    return idx


class MockArrayRef:
    """Mock buffer reference for simulating Pallas kernel operations on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr)
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        return jnp.asarray(self.arr[_norm_idx(idx, self.arr.ndim)])

    def __setitem__(self, idx, val):
        if isinstance(val, MockArrayRef):
            val = val.arr
        val_arr = np.asarray(val)
        if idx is Ellipsis:
            if self.arr.shape != val_arr.shape:
                val_arr = val_arr.reshape(-1)[: self.arr.size].reshape(self.arr.shape)
            self.arr[...] = val_arr
        else:
            self.arr[_norm_idx(idx, self.arr.ndim)] = val_arr

    @property
    def at(self):
        class _AtIndexer:
            def __init__(self, ref):
                self.ref = ref

            def __getitem__(self, idx):
                if idx is Ellipsis:
                    return self.ref
                view = self.ref.arr[_norm_idx(idx, self.ref.arr.ndim)]
                return MockArrayRef(view)

            def __setitem__(self, idx, val):
                self.ref[_norm_idx(idx, self.ref.arr.ndim)] = val

        return _AtIndexer(self)

    def bitcast(self, dtype):
        return MockArrayRef(self.arr.view(dtype))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return MockArrayRef(self.arr.reshape(*shape))


class MockAsyncCopy:
    """Mock DMA async copy handle."""

    def __init__(self, src_ref=None, dst_ref=None):
        self.src_ref = src_ref
        self.dst_ref = dst_ref
        self.started = False
        self.waited = False

    def start(self):
        self.started = True
        return self

    def wait(self):
        self.waited = True
        return self


def fake_when(cond):
    """Executes decorated function when condition evaluates to True."""

    def decorator(fn):
        execute = True
        try:
            execute = bool(cond)
        except Exception:
            try:
                execute = bool(np.all(cond))
            except Exception:
                execute = True
        if execute:
            fn()
        return fn

    return decorator


def fake_loop(start, end):
    """Iterates a pl.loop decorated body."""

    def decorator(fn):
        for i in range(int(np.asarray(start)), int(np.asarray(end))):
            fn(i)
        return fn

    return decorator


def fake_fori_loop(lower, upper, body_fun, init_val, unroll=False):
    """Iterates lax.fori_loop body."""
    val = init_val
    for i in range(int(np.asarray(lower)), int(np.asarray(upper))):
        val = body_fun(i, val)
    return val


def fake_run_scoped(f, *allocs):
    """Executes pl.run_scoped by creating MockArrayRefs for each allocation."""
    refs = [MockArrayRef(np.zeros(a.shape, a.dtype)) for a in allocs]
    return f(*refs)


# ==============================================================================
# 1. Tests for Layout Validation
# ==============================================================================


class TestPcpStreamingRpaLayoutValidation:
    """Tests for _validate_packed_kv_layout covering all error and return branches."""

    def test_q_local_ndim_error(self):
        q = jnp.zeros((4, 2, 64), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 16, 2, 2, 64), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="PCP q_local must have shape"):
            kernel._validate_packed_kv_layout(q, kv_cache)

    def test_kv_cache_local_ndim_error(self):
        q = jnp.zeros((4, 1, 2, 64), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 16, 2, 64), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="PCP kv_cache_local must be 5D"):
            kernel._validate_packed_kv_layout(q, kv_cache)

    def test_non_positive_heads_error(self):
        # kv_heads <= 0
        q_bad_heads = jnp.zeros((4, 0, 2, 64), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 16, 2, 2, 64), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="PCP requires positive kv_heads"):
            kernel._validate_packed_kv_layout(q_bad_heads, kv_cache)

        # q_per_kv <= 0
        q_bad_q_per_kv = jnp.zeros((4, 2, 0, 64), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="PCP requires positive kv_heads"):
            kernel._validate_packed_kv_layout(q_bad_q_per_kv, kv_cache)

    def test_unsupported_kv_packing_error(self):
        q = jnp.zeros((4, 1, 2, 64), dtype=jnp.bfloat16)
        # packing dimension = 3 (not in 2, 4)
        kv_cache = jnp.zeros((2, 16, 2, 3, 64), dtype=jnp.bfloat16)
        with pytest.raises(
            NotImplementedError, match="supports only BF16/FP8 packed KV cache layouts"
        ):
            kernel._validate_packed_kv_layout(q, kv_cache)

    def test_packing_dtype_mismatch_error(self):
        q = jnp.zeros((4, 1, 2, 64), dtype=jnp.bfloat16)
        # dtype packing for bfloat16 is 2, but cache packing dim is 4
        kv_cache = jnp.zeros((2, 16, 2, 4, 64), dtype=jnp.bfloat16)
        with pytest.raises(
            ValueError, match="PCP KV cache packing does not match dtype"
        ):
            kernel._validate_packed_kv_layout(q, kv_cache, k_scale=1.0, v_scale=1.0)

    def test_seq_along_lane_group_count_error(self):
        # kv_heads = 2, so expected planes = 2 * 2 = 4, but passed 3
        q = jnp.zeros((4, 2, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 3, 32, 4, 128), dtype=jnp.float8_e4m3fn)
        with pytest.raises(
            ValueError,
            match="PCP SEQ_ALONG_LANE cache must contain one K and V plane",
        ):
            kernel._validate_packed_kv_layout(
                q,
                kv_cache,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                k_scale=1.0,
                v_scale=1.0,
            )

    def test_seq_along_lane_headdim_error(self):
        # head_dim = 128, but kv_cache shape[2] * packing = 30 * 4 = 120 != 128
        q = jnp.zeros((4, 2, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 4, 30, 4, 128), dtype=jnp.float8_e4m3fn)
        with pytest.raises(
            ValueError, match="PCP SEQ_ALONG_LANE cache head_dim must match Q head_dim"
        ):
            kernel._validate_packed_kv_layout(
                q,
                kv_cache,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                k_scale=1.0,
                v_scale=1.0,
            )

    def test_seq_along_lane_non_fp8_error(self):
        # packing = 2 (BF16), not supported for SEQ_ALONG_LANE
        q = jnp.zeros((4, 2, 2, 64), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 4, 32, 2, 64), dtype=jnp.bfloat16)
        with pytest.raises(
            NotImplementedError,
            match="PCP SEQ_ALONG_LANE currently supports only FP8",
        ):
            kernel._validate_packed_kv_layout(
                q, kv_cache, kv_layout=KVLayout.SEQ_ALONG_LANE
            )

    def test_head_along_sublane_group_count_error(self):
        # kv_heads = 2, packing = 2 -> ceil(4/2) = 2 packed groups, but passed 3
        q = jnp.zeros((4, 2, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 16, 3, 2, 128), dtype=jnp.bfloat16)
        with pytest.raises(
            ValueError, match="PCP KV cache packed layout does not match"
        ):
            kernel._validate_packed_kv_layout(q, kv_cache)

    def test_head_along_sublane_headdim_error(self):
        # head_dim = 128, but cache head_dim = 64
        q = jnp.zeros((4, 2, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 16, 2, 2, 64), dtype=jnp.bfloat16)
        with pytest.raises(
            ValueError, match="PCP KV cache head_dim must match Q head_dim"
        ):
            kernel._validate_packed_kv_layout(q, kv_cache)

    def test_scale_none_mismatch_error(self):
        q = jnp.zeros((4, 1, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 16, 1, 2, 128), dtype=jnp.bfloat16)
        with pytest.raises(
            ValueError, match="requires k_scale and v_scale to be both None"
        ):
            kernel._validate_packed_kv_layout(q, kv_cache, k_scale=1.0, v_scale=None)
        with pytest.raises(
            ValueError, match="requires k_scale and v_scale to be both None"
        ):
            kernel._validate_packed_kv_layout(q, kv_cache, k_scale=None, v_scale=1.0)

    def test_fp8_scale_validation_errors(self):
        q = jnp.zeros((4, 1, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 16, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        # scales are None
        with pytest.raises(ValueError, match="requires positive finite k_scale"):
            kernel._validate_packed_kv_layout(q, kv_cache, k_scale=None, v_scale=None)

        # k_scale non-finite or <= 0
        with pytest.raises(ValueError, match="requires positive finite k_scale"):
            kernel._validate_packed_kv_layout(q, kv_cache, k_scale=-1.0, v_scale=1.0)
        with pytest.raises(ValueError, match="requires positive finite k_scale"):
            kernel._validate_packed_kv_layout(
                q, kv_cache, k_scale=float("nan"), v_scale=1.0
            )

        # v_scale non-finite or <= 0
        with pytest.raises(ValueError, match="requires positive finite k_scale"):
            kernel._validate_packed_kv_layout(q, kv_cache, k_scale=1.0, v_scale=0.0)
        with pytest.raises(ValueError, match="requires positive finite k_scale"):
            kernel._validate_packed_kv_layout(
                q, kv_cache, k_scale=1.0, v_scale=float("inf")
            )

    def test_valid_layout_configurations(self):
        # 1. BF16 HEAD_ALONG_SUBLANE
        q = jnp.zeros((4, 2, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((2, 16, 2, 2, 128), dtype=jnp.bfloat16)
        assert kernel._validate_packed_kv_layout(q, kv_cache) == (2, 2, 2)

        # 2. FP8 HEAD_ALONG_SUBLANE
        kv_cache_fp8 = jnp.zeros((2, 16, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        assert kernel._validate_packed_kv_layout(
            q, kv_cache_fp8, k_scale=0.5, v_scale=0.5
        ) == (2, 1, 4)

        # 3. FP8 SEQ_ALONG_LANE
        kv_cache_seq = jnp.zeros((2, 4, 32, 4, 128), dtype=jnp.float8_e4m3fn)
        assert kernel._validate_packed_kv_layout(
            q,
            kv_cache_seq,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            k_scale=0.5,
            v_scale=0.5,
        ) == (2, 4, 4)


# ==============================================================================
# 2. Tests for Helper Functions & Math/Schedule Computations
# ==============================================================================


class TestPcpStreamingRpaHelpers:
    """Tests for helper utilities, coordinate translations,
    and schedule step staging.
    """

    def test_normalize_q_compute_size(self):
        # Default to q_block_size
        assert kernel._normalize_q_compute_size(128, None) == 128
        assert kernel._normalize_q_compute_size(128, 64) == 64

        # Non-positive
        with pytest.raises(ValueError, match="must be positive"):
            kernel._normalize_q_compute_size(128, 0)
        with pytest.raises(ValueError, match="must be positive"):
            kernel._normalize_q_compute_size(128, -16)

        # Greater than q_block_size
        with pytest.raises(ValueError, match="must be <= q_block_size"):
            kernel._normalize_q_compute_size(64, 128)

        # Not dividing q_block_size
        with pytest.raises(NotImplementedError, match="divisible by q_compute_size"):
            kernel._normalize_q_compute_size(128, 48)

        # Not multiple of TPU_HBM_ROW_TILE (16)
        with pytest.raises(NotImplementedError, match="multiple of 16"):
            kernel._normalize_q_compute_size(128, 8)

    def test_mesh_device_id(self):
        mesh_axes = ("pcp", "tp", "dp")
        with patch.object(kernel.lax, "axis_index", return_value=0):
            device_id = kernel._mesh_device_id(mesh_axes, "pcp", 3)
            assert device_id == (3, 0, 0)

    def test_tile_plan_value_and_static_and_vector_lookup(self):
        plan = np.zeros((1, 4 * TilePlanField.NUM_FIELDS), dtype=np.int32)
        plan[0, 2 * TilePlanField.NUM_FIELDS + TilePlanField.REQ_ID] = 42
        plan_jax = jnp.asarray(plan)

        val = kernel._tile_plan_value(plan_jax, 2, TilePlanField.REQ_ID)
        assert int(val) == 42
        val_static = kernel._tile_plan_value_static(plan_jax, 2, TilePlanField.REQ_ID)
        assert int(val_static) == 42

    def test_group_tile_position(self):
        group_starts = jnp.array([0, 4, 10], dtype=np.int32)
        assert int(kernel._group_tile_position(group_starts, 3, 0)) == 0
        assert int(kernel._group_tile_position(group_starts, 3, 2)) == 0
        assert int(kernel._group_tile_position(group_starts, 3, 4)) == 1
        assert int(kernel._group_tile_position(group_starts, 3, 7)) == 1
        assert int(kernel._group_tile_position(group_starts, 3, 10)) == 2
        assert int(kernel._group_tile_position(group_starts, 3, 15)) == 2

    def test_local_page_valid_len_from_tile_plan(self):
        # 1. full_chunks < chunks_per_page
        len1 = kernel._local_page_valid_len_from_tile_plan(
            kv_len=100,
            local_page_idx=0,
            src_rank=0,
            page_size=128,
            pcp_size=4,
            interleave_size=32,
        )
        assert int(len1) > 0

        # 2. full_chunks >= chunks_per_page
        len2 = kernel._local_page_valid_len_from_tile_plan(
            kv_len=1000,
            local_page_idx=0,
            src_rank=0,
            page_size=64,
            pcp_size=2,
            interleave_size=32,
        )
        assert int(len2) == 64

        # 3. kv_len <= base -> 0
        len0 = kernel._local_page_valid_len_from_tile_plan(
            kv_len=10,
            local_page_idx=1,
            src_rank=0,
            page_size=128,
            pcp_size=4,
            interleave_size=32,
        )
        assert int(len0) == 0

    def test_rank_token_count_before(self):
        cnt = kernel._rank_token_count_before(
            position=200, rank=1, pcp_size=4, interleave_size=32
        )
        assert int(cnt) > 0

    def test_stage_schedule_step_invalid_mode(self):
        sched_ref = MockArrayRef(
            np.zeros((2, 1, RuntimeScheduleField.PACKED_NUM_FIELDS), dtype=np.int32)
        )
        with pytest.raises(ValueError, match="Unsupported PCP state mode"):
            kernel._stage_schedule_step_from_tile_plan(
                None,
                None,
                sched_ref,
                0,
                0,
                state_mode="invalid",
                pcp_size=2,
                page_size=16,
                interleave_size=16,
                lane=0,
            )

    def test_stage_current_and_history_schedule_steps(self):
        pcp_size = 2
        page_size = 16
        interleave_size = 16
        num_fields = TilePlanField.NUM_FIELDS
        plan = np.zeros((1, pcp_size * num_fields), dtype=np.int32)
        plan[0, TilePlanField.REQ_ID] = 7
        plan[0, TilePlanField.TOKEN_OWNER_START] = 0
        plan[0, TilePlanField.REQUEST_ABSOLUTE_QUERY_START] = (
            32  # aligned to virtual page (16*2=32)
        )
        plan[0, TilePlanField.CURRENT_EFFECTIVE_LEN] = 32
        plan[0, TilePlanField.CAPTURE_CURRENT_KV] = 1
        plan[0, TilePlanField.WRITEBACK_HBM_PREFIX] = 0

        block_tables = np.array([5, 6, 7], dtype=np.int32)
        sched_vmem = MockArrayRef(
            np.zeros(
                (pcp_size, 1, RuntimeScheduleField.PACKED_NUM_FIELDS), dtype=np.int32
            )
        )

        # Current mode without history boundary
        kernel._stage_current_schedule_step_from_tile_plan(
            jnp.asarray(plan),
            jnp.asarray(block_tables),
            sched_vmem,
            group_in_tile=0,
            source_rank=0,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
            lane=0,
        )
        assert int(sched_vmem[0, 0, RuntimeScheduleField.REQ_ID]) == 7

        # Current mode with history boundary present (unaligned absolute start)
        plan_unaligned = plan.copy()
        plan_unaligned[0, TilePlanField.REQUEST_ABSOLUTE_QUERY_START] = 10
        kernel._stage_current_schedule_step_from_tile_plan(
            jnp.asarray(plan_unaligned),
            jnp.asarray(block_tables),
            sched_vmem,
            group_in_tile=0,
            source_rank=0,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
            lane=0,
        )
        assert int(sched_vmem[0, 0, RuntimeScheduleField.KV_HBM_OFFSET]) == -1

        # History mode
        kernel._stage_history_schedule_step_from_tile_plan(
            jnp.asarray(plan),
            jnp.asarray(block_tables),
            sched_vmem,
            group_in_tile=0,
            source_rank=0,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
            lane=0,
        )
        assert int(sched_vmem[0, 0, RuntimeScheduleField.REQ_ID]) == 7

    def test_reset_multi_head_online_state_and_write_output(self):
        m_scratch = MockArrayRef(np.ones((2, 16, 2, 128), dtype=np.float32))
        l_scratch = MockArrayRef(np.ones((2, 16, 2, 128), dtype=np.float32))
        acc_scratch = MockArrayRef(np.ones((2, 16, 2, 64), dtype=np.float32))
        o_vmem = MockArrayRef(np.zeros((16, 2, 2, 64), dtype=np.float32))

        # Reset
        kernel._reset_multi_head_online_state(m_scratch, l_scratch, acc_scratch)
        np.testing.assert_allclose(m_scratch.arr, kernel.ONLINE_STATE_INIT_VALUE)
        np.testing.assert_allclose(l_scratch.arr, 0.0)
        np.testing.assert_allclose(acc_scratch.arr, 0.0)

        # Write output
        l_scratch.arr[...] = 2.0
        acc_scratch.arr[...] = 6.0
        kernel._write_multi_head_output_from_state(o_vmem, l_scratch, acc_scratch)
        np.testing.assert_allclose(o_vmem.arr, 3.0)


# ==============================================================================
# 3. Tests for Copy, Writeback & Cache Updates
# ==============================================================================


class TestCopyHndTokenSliceAndWriteback:
    """Tests for HND slicing, writeback kernels and write_captured_kv_to_local_cache."""

    def test_copy_hnd_token_slice_branches(self):
        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(kernel.pl, "loop", side_effect=fake_loop),
            patch.object(kernel.pl, "run_scoped", side_effect=fake_run_scoped),
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.pl, "cdiv", side_effect=lambda a, b: (a + b - 1) // b),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "roll",
                side_effect=lambda a, s, axis: np.roll(
                    a, int(np.asarray(s)), axis=axis
                ),
            ),
        ):
            src = MockArrayRef(np.ones((2, 32, 4, 256), dtype=np.uint8))
            dst = MockArrayRef(np.zeros((2, 32, 4, 256), dtype=np.uint8))

            # 1. Aligned copy length > 0
            kernel._copy_hnd_token_slice(
                src, dst, 0, 0, 128, None, source_hbm=True, destination_hbm=True
            )

            # 2. Unaligned copy with source_hbm=True, destination_hbm=True
            kernel._copy_hnd_token_slice(
                src, dst, 10, 15, 50, None, source_hbm=True, destination_hbm=True
            )

            # 3. Unaligned copy with source_hbm=False, destination_hbm=False
            kernel._copy_hnd_token_slice(
                src, dst, 10, 15, 50, None, source_hbm=False, destination_hbm=False
            )

            # 4. Length <= 0 (neither when-branch triggers)
            kernel._copy_hnd_token_slice(
                src, dst, 0, 0, 0, None, source_hbm=True, destination_hbm=True
            )
            kernel._copy_hnd_token_slice(
                src, dst, 5, 5, 0, None, source_hbm=True, destination_hbm=True
            )

    def test_pcp_write_captured_kv_kernels(self):
        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(kernel.pl, "loop", side_effect=fake_loop),
            patch.object(kernel.pl, "run_scoped", side_effect=fake_run_scoped),
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.pl, "cdiv", side_effect=lambda a, b: (a + b - 1) // b),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "roll",
                side_effect=lambda a, s, axis: np.roll(
                    a, int(np.asarray(s)), axis=axis
                ),
            ),
        ):
            num_segments_ref = MockArrayRef(np.array([2], dtype=np.int32))
            segment_descriptors_ref = MockArrayRef(
                np.array([[0, 10], [0, 20], [5, 15]], dtype=np.int32)
            )
            captured_kv_ref = MockArrayRef(np.zeros((100, 128), dtype=np.float32))
            kv_cache_in_ref = MockArrayRef(np.zeros((100, 128), dtype=np.float32))
            kv_cache_out_ref = MockArrayRef(np.zeros((100, 128), dtype=np.float32))

            # Flat write captured kernel
            kernel._pcp_write_captured_kv_kernel(
                num_segments_ref,
                segment_descriptors_ref,
                captured_kv_ref,
                kv_cache_in_ref,
                kv_cache_out_ref,
                None,
            )

            # SEQ_ALONG_LANE write captured kernel
            captured_kv_seq = MockArrayRef(np.zeros((2, 32, 4, 100), dtype=np.uint8))
            kv_cache_seq = MockArrayRef(np.zeros((4, 2, 32, 4, 128), dtype=np.uint8))
            kernel._pcp_write_captured_kv_seq_along_lane_kernel(
                num_segments_ref,
                segment_descriptors_ref,
                captured_kv_seq,
                kv_cache_in_ref,
                kv_cache_seq,
                None,
            )

    def test_write_captured_kv_to_local_cache_validation_errors(self):
        # 1. kv_cache ndim < 2
        with pytest.raises(ValueError, match="KV cache must have at least two dims"):
            kernel.write_captured_kv_to_local_cache(
                jnp.zeros((10,), dtype=jnp.float32),
                jnp.zeros((10,), dtype=jnp.float32),
                jnp.zeros((3, 1), dtype=jnp.int32),
                1,
            )

        # 2. SEQ_ALONG_LANE captured prefix mismatch
        kv_cache_seq = jnp.zeros((4, 2, 32, 4, 128), dtype=jnp.float8_e4m3fn)
        captured_kv_bad = jnp.zeros((2, 30, 4, 100), dtype=jnp.float8_e4m3fn)
        with pytest.raises(
            ValueError, match="SEQ_ALONG_LANE captured KV prefix must match"
        ):
            kernel.write_captured_kv_to_local_cache(
                kv_cache_seq,
                captured_kv_bad,
                jnp.zeros((3, 1), dtype=jnp.int32),
                1,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
            )

        # 3. Flat layout tail mismatch
        kv_cache_flat = jnp.zeros((4, 16, 2, 2, 64), dtype=jnp.bfloat16)
        captured_flat_bad = jnp.zeros((32, 2, 2, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="Captured KV tail must match"):
            kernel.write_captured_kv_to_local_cache(
                kv_cache_flat,
                captured_flat_bad,
                jnp.zeros((3, 1), dtype=jnp.int32),
                1,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )

        # 4. Descriptors shape error
        with pytest.raises(
            ValueError, match="KV writeback segment descriptors must have shape"
        ):
            kernel.write_captured_kv_to_local_cache(
                kv_cache_flat,
                jnp.zeros((32, 2, 2, 64), dtype=jnp.bfloat16),
                jnp.zeros((2, 1), dtype=jnp.int32),
                1,
            )

    def test_write_captured_kv_to_local_cache_execution(self, monkeypatch):
        # Mock pallas_call to return the updated cache
        def mock_pallas_call(kernel_fn, out_shape, **kwargs):
            def runner(num_segs, seg_desc, captured, writeback):
                return writeback

            return runner

        monkeypatch.setattr(kernel.pl, "pallas_call", mock_pallas_call)

        # 1. SEQ_ALONG_LANE (returns updated_cache directly without reshape)
        kv_cache_seq = jnp.zeros((4, 2, 32, 4, 128), dtype=jnp.float8_e4m3fn)
        captured_seq = jnp.zeros((2, 32, 4, 64), dtype=jnp.float8_e4m3fn)
        descriptors = jnp.zeros((3, 2), dtype=jnp.int32)
        out_seq = kernel.write_captured_kv_to_local_cache(
            kv_cache_seq,
            captured_seq,
            descriptors,
            1,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
        )
        assert out_seq.shape == kv_cache_seq.shape

        # 2. HEAD_ALONG_SUBLANE (returns updated_cache.reshape)
        kv_cache_flat = jnp.zeros((4, 16, 2, 2, 64), dtype=jnp.bfloat16)
        captured_flat = jnp.zeros((32, 2, 2, 64), dtype=jnp.bfloat16)
        out_flat = kernel.write_captured_kv_to_local_cache(
            kv_cache_flat,
            captured_flat,
            descriptors,
            1,
            kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        )
        assert out_flat.shape == kv_cache_flat.shape


# ==============================================================================
# 4. Tests for Multi-Head Softmax Compute & KV Consumption
# ==============================================================================


class TestConsumeScheduledKvPageMultiHead:
    """Tests for _consume_scheduled_kv_page_multi_head under all
    causal/layout/scale modes.
    """

    def test_strided_load_packed_kv_group(self):
        with (
            patch.object(kernel, "strided_load", return_value=jnp.zeros((16, 128))),
            patch.object(
                kernel,
                "convert_to_target_bitwidth",
                return_value=[(jnp.zeros((16, 128)), jnp.zeros((16, 128)))],
            ),
        ):
            kv_vmem = MockArrayRef(np.zeros((2, 1, 16, 2, 2, 128), dtype=np.uint16))
            pairs = kernel._strided_load_packed_kv_group(kv_vmem, 0, 0)
            assert len(pairs) == 1
            k, v = pairs[0]
            assert k.shape == (16, 128)
            assert v.shape == (16, 128)

    def test_consume_scheduled_kv_page_seq_along_lane_causal_with_scales(self):
        q_vmem = MockArrayRef(np.ones((64, 2, 2, 128), dtype=np.float32))
        kv_vmem = MockArrayRef(np.ones((2, 1, 4, 32, 4, 128), dtype=np.float32))
        sched_vmem = MockArrayRef(
            np.zeros((2, 1, RuntimeScheduleField.PACKED_NUM_FIELDS), dtype=np.int32)
        )
        sched_vmem[0, 0, RuntimeScheduleField.Q_GLOBAL_START] = 0
        sched_vmem[0, 0, RuntimeScheduleField.Q_HBM_OFFSET] = 0
        sched_vmem[0, 0, RuntimeScheduleField.KV_GLOBAL_START] = 0
        sched_vmem[0, 0, RuntimeScheduleField.Q_TILE_SIZE] = 64
        sched_vmem[0, 0, RuntimeScheduleField.REQ_ID] = 1
        sched_vmem[0, 0, RuntimeScheduleField.KV_VALID_LEN] = 128

        m_scratch = MockArrayRef(
            np.full((2, 64, 2, 128), kernel.ONLINE_STATE_INIT_VALUE, dtype=np.float32)
        )
        l_scratch = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))
        acc_scratch = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))

        with (
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            kernel._consume_scheduled_kv_page_multi_head(
                q_vmem,
                kv_vmem,
                sched_vmem,
                slot=0,
                consumer_rank=0,
                lane=0,
                m_scratch_ref=m_scratch,
                l_scratch_ref=l_scratch,
                acc_scratch_ref=acc_scratch,
                sm_scale=0.125,
                pcp_size=2,
                interleave_size=64,
                q_compute_size=64,
                k_scale=1.0,
                v_scale=1.0,
                causal=True,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
            )

        assert np.all(m_scratch.arr > kernel.ONLINE_STATE_INIT_VALUE)
        assert np.all(l_scratch.arr > 0)

    def test_consume_scheduled_kv_page_seq_along_lane_non_causal_no_scales(self):
        q_vmem = MockArrayRef(np.ones((64, 2, 2, 128), dtype=np.float32))
        kv_vmem = MockArrayRef(np.ones((2, 1, 4, 32, 4, 128), dtype=np.float32))
        sched_vmem = MockArrayRef(
            np.zeros((2, 1, RuntimeScheduleField.PACKED_NUM_FIELDS), dtype=np.int32)
        )
        sched_vmem[0, 0, RuntimeScheduleField.Q_GLOBAL_START] = 0
        sched_vmem[0, 0, RuntimeScheduleField.Q_HBM_OFFSET] = 0
        sched_vmem[0, 0, RuntimeScheduleField.KV_GLOBAL_START] = 0
        sched_vmem[0, 0, RuntimeScheduleField.Q_TILE_SIZE] = 64
        sched_vmem[0, 0, RuntimeScheduleField.REQ_ID] = 1
        sched_vmem[0, 0, RuntimeScheduleField.KV_VALID_LEN] = 128

        m_scratch = MockArrayRef(
            np.full((2, 64, 2, 128), kernel.ONLINE_STATE_INIT_VALUE, dtype=np.float32)
        )
        l_scratch = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))
        acc_scratch = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))

        with (
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            kernel._consume_scheduled_kv_page_multi_head(
                q_vmem,
                kv_vmem,
                sched_vmem,
                slot=0,
                consumer_rank=0,
                lane=0,
                m_scratch_ref=m_scratch,
                l_scratch_ref=l_scratch,
                acc_scratch_ref=acc_scratch,
                sm_scale=0.125,
                pcp_size=2,
                interleave_size=64,
                q_compute_size=64,
                k_scale=None,
                v_scale=None,
                causal=False,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
            )

        assert np.all(m_scratch.arr > kernel.ONLINE_STATE_INIT_VALUE)

    def test_consume_scheduled_kv_page_packed_causal_with_head_continue(self):
        # q_vmem has 1 kv_head, but kv_vmem has 2 kv_heads
        # (heads_per_group=2 // 2 = 1, packed_groups=2).
        # Group 1 pair 0 has kv_head_idx = 1 >= q_vmem.shape[1] (1),
        # triggering the continue branch!
        q_vmem = MockArrayRef(np.ones((64, 1, 2, 128), dtype=np.float32))
        kv_vmem = MockArrayRef(np.ones((2, 1, 16, 2, 2, 128), dtype=np.uint16))
        sched_vmem = MockArrayRef(
            np.zeros((2, 1, RuntimeScheduleField.PACKED_NUM_FIELDS), dtype=np.int32)
        )
        sched_vmem[0, 0, RuntimeScheduleField.Q_GLOBAL_START] = 0
        sched_vmem[0, 0, RuntimeScheduleField.Q_HBM_OFFSET] = 0
        sched_vmem[0, 0, RuntimeScheduleField.KV_GLOBAL_START] = 0
        sched_vmem[0, 0, RuntimeScheduleField.Q_TILE_SIZE] = 64
        sched_vmem[0, 0, RuntimeScheduleField.REQ_ID] = 1
        sched_vmem[0, 0, RuntimeScheduleField.KV_VALID_LEN] = 16

        m_scratch = MockArrayRef(
            np.full((1, 64, 2, 128), kernel.ONLINE_STATE_INIT_VALUE, dtype=np.float32)
        )
        l_scratch = MockArrayRef(np.zeros((1, 64, 2, 128), dtype=np.float32))
        acc_scratch = MockArrayRef(np.zeros((1, 64, 2, 128), dtype=np.float32))

        with (
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.lax, "fori_loop", side_effect=fake_fori_loop),
            patch.object(kernel, "strided_load", return_value=jnp.zeros((16, 128))),
            patch.object(
                kernel,
                "convert_to_target_bitwidth",
                return_value=[
                    (jnp.zeros((16, 128)), jnp.zeros((16, 128))),
                    (jnp.zeros((16, 128)), jnp.zeros((16, 128))),
                ],
            ),
        ):
            kernel._consume_scheduled_kv_page_multi_head(
                q_vmem,
                kv_vmem,
                sched_vmem,
                slot=0,
                consumer_rank=0,
                lane=0,
                m_scratch_ref=m_scratch,
                l_scratch_ref=l_scratch,
                acc_scratch_ref=acc_scratch,
                sm_scale=0.125,
                pcp_size=2,
                interleave_size=16,
                q_compute_size=64,
                k_scale=1.0,
                v_scale=1.0,
                causal=True,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )

        assert np.all(m_scratch.arr > kernel.ONLINE_STATE_INIT_VALUE)


# ==============================================================================
# 5. Tests for Tile Load / Store Operations
# ==============================================================================


class TestTileLoadStoreHelpers:
    """Tests for DMA load/store helper functions for Q, O, KV, and online states."""

    def test_store_and_load_multi_head_online_state_tile(self):
        m_ref = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))
        l_ref = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))
        acc_ref = MockArrayRef(np.zeros((2, 64, 2, 64), dtype=np.float32))
        m_scratch = MockArrayRef(np.ones((2, 64, 2, 128), dtype=np.float32))
        l_scratch = MockArrayRef(np.ones((2, 64, 2, 128), dtype=np.float32))
        acc_scratch = MockArrayRef(np.ones((2, 64, 2, 64), dtype=np.float32))

        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
        ):
            # 1. q_tile_size > 0
            kernel._store_multi_head_online_state_tile(
                m_ref, l_ref, acc_ref, None, m_scratch, l_scratch, acc_scratch, 0, 64
            )
            kernel._load_multi_head_online_state_tile(
                m_ref, l_ref, acc_ref, None, m_scratch, l_scratch, acc_scratch, 0, 64
            )

            # 2. q_tile_size <= 0
            kernel._store_multi_head_online_state_tile(
                m_ref, l_ref, acc_ref, None, m_scratch, l_scratch, acc_scratch, 0, 0
            )
            kernel._load_multi_head_online_state_tile(
                m_ref, l_ref, acc_ref, None, m_scratch, l_scratch, acc_scratch, 0, 0
            )

    def test_load_and_store_compact_q_and_output_tiles(self):
        q_ref = MockArrayRef(np.ones((64, 2, 2, 64), dtype=np.float32))
        q_vmem = MockArrayRef(np.zeros((64, 2, 2, 64), dtype=np.float32))
        o_ref = MockArrayRef(np.zeros((64, 2, 2, 64), dtype=np.float32))
        o_vmem = MockArrayRef(np.ones((64, 2, 2, 64), dtype=np.float32))

        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
        ):
            # q_tile_size > 0
            kernel._load_compact_q_tile_multi_head(q_ref, q_vmem, None, 0, 64)
            kernel._store_compact_output_tile_multi_head(o_ref, o_vmem, None, 0, 64)

            # q_tile_size <= 0
            kernel._load_compact_q_tile_multi_head(q_ref, q_vmem, None, 0, 0)
            kernel._store_compact_output_tile_multi_head(o_ref, o_vmem, None, 0, 0)

    def test_load_local_kv_page_all_heads_packed(self):
        sched_vmem = MockArrayRef(
            np.zeros((2, 1, RuntimeScheduleField.PACKED_NUM_FIELDS), dtype=np.int32)
        )

        with patch.object(
            kernel.pltpu, "make_async_copy", side_effect=lambda *a, **k: MockAsyncCopy()
        ):
            # SEQ_ALONG_LANE
            kv_cache_seq = MockArrayRef(np.zeros((1, 4, 2, 32, 4, 128), dtype=np.uint8))
            kv_vmem_seq = MockArrayRef(np.zeros((2, 1, 2, 32, 4, 128), dtype=np.uint8))
            kernel._load_local_kv_page_all_heads_packed(
                kv_cache_seq,
                kv_vmem_seq,
                None,
                sched_vmem,
                0,
                packed_kv_groups=2,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
            )

            # HEAD_ALONG_SUBLANE
            kv_cache_flat = MockArrayRef(
                np.zeros((1, 4, 16, 2, 2, 64), dtype=np.uint16)
            )
            kv_vmem_flat = MockArrayRef(np.zeros((2, 1, 16, 2, 2, 64), dtype=np.uint16))
            kernel._load_local_kv_page_all_heads_packed(
                kv_cache_flat,
                kv_vmem_flat,
                None,
                sched_vmem,
                0,
                packed_kv_groups=2,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )

    def test_load_local_current_or_boundary_kv_all_heads_packed(self):
        sched_vmem = MockArrayRef(
            np.zeros((2, 1, RuntimeScheduleField.PACKED_NUM_FIELDS), dtype=np.int32)
        )

        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(kernel.pl, "loop", side_effect=fake_loop),
            patch.object(kernel.pl, "run_scoped", side_effect=fake_run_scoped),
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.pl, "cdiv", side_effect=lambda a, b: (a + b - 1) // b),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "roll",
                side_effect=lambda a, s, axis: np.roll(
                    a, int(np.asarray(s)), axis=axis
                ),
            ),
        ):
            # 1. History boundary (< 0) with SEQ_ALONG_LANE
            sched_vmem[0, 0, RuntimeScheduleField.KV_HBM_OFFSET] = -1
            sched_vmem[0, 0, RuntimeScheduleField.KV_VALID_LEN] = 16
            kv_cache_seq = MockArrayRef(np.zeros((1, 4, 2, 32, 4, 128), dtype=np.uint8))
            kv_vmem_seq = MockArrayRef(np.zeros((2, 1, 2, 32, 4, 128), dtype=np.uint8))
            current_kv_seq = MockArrayRef(np.zeros((2, 32, 4, 128), dtype=np.uint8))
            kernel._load_local_current_or_boundary_kv_all_heads_packed(
                current_kv_seq,
                kv_cache_seq,
                kv_vmem_seq,
                None,
                sched_vmem,
                0,
                packed_kv_groups=2,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
            )

            # 2. History boundary (< 0) with HEAD_ALONG_SUBLANE
            kv_cache_flat = MockArrayRef(
                np.zeros((1, 4, 16, 2, 2, 64), dtype=np.uint16)
            )
            kv_vmem_flat = MockArrayRef(np.zeros((2, 1, 16, 2, 2, 64), dtype=np.uint16))
            current_kv_flat = MockArrayRef(np.zeros((64, 2, 2, 64), dtype=np.uint16))
            kernel._load_local_current_or_boundary_kv_all_heads_packed(
                current_kv_flat,
                kv_cache_flat,
                kv_vmem_flat,
                None,
                sched_vmem,
                0,
                packed_kv_groups=2,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )

            # 3. Fresh current KV (>= 0 and valid > 0) with SEQ_ALONG_LANE
            sched_vmem[0, 0, RuntimeScheduleField.KV_HBM_OFFSET] = 0
            kernel._load_local_current_or_boundary_kv_all_heads_packed(
                current_kv_seq,
                kv_cache_seq,
                kv_vmem_seq,
                None,
                sched_vmem,
                0,
                packed_kv_groups=2,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
            )

            # 4. Fresh current KV (>= 0 and valid > 0) with HEAD_ALONG_SUBLANE
            kernel._load_local_current_or_boundary_kv_all_heads_packed(
                current_kv_flat,
                kv_cache_flat,
                kv_vmem_flat,
                None,
                sched_vmem,
                0,
                packed_kv_groups=2,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )


# ==============================================================================
# 6. Tests for Full Ring Page Groups Execution & Kernel Wrappers
# ==============================================================================


class TestPcpPageGroupsExecution:
    """Tests for _run_pcp_page_groups_multi_head and the 3 kernel functions."""

    def test_run_pcp_page_groups_invalid_state_mode(self):
        with pytest.raises(ValueError, match="Unsupported PCP state mode"):
            kernel._run_pcp_page_groups_multi_head(
                1,
                1,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                pcp_size=2,
                interleave_size=16,
                lane=0,
                q_compute_size=16,
                sm_scale=1.0,
                packed_kv_groups=2,
                page_size=16,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
                k_scale=None,
                v_scale=None,
                mesh_axis_names=("pcp",),
                pcp_axis_name="pcp",
                state_mode="invalid",
            )

    def _setup_page_groups_refs(self, kv_layout=KVLayout.SEQ_ALONG_LANE, pcp_size=2):
        q_ref = MockArrayRef(np.ones((64, 2, 2, 128), dtype=np.float32))
        if kv_layout == KVLayout.SEQ_ALONG_LANE:
            current_kv_ref = MockArrayRef(np.ones((4, 32, 4, 128), dtype=np.float32))
            kv_cache_ref = MockArrayRef(
                np.ones((1, 4, 4, 32, 4, 128), dtype=np.float32)
            )
            kv_vmem_ref = MockArrayRef(np.ones((2, 1, 4, 32, 4, 128), dtype=np.float32))
            captured_kv_ref = MockArrayRef(np.zeros((4, 32, 4, 128), dtype=np.float32))
        else:
            current_kv_ref = MockArrayRef(np.ones((64, 2, 2, 128), dtype=np.uint16))
            kv_cache_ref = MockArrayRef(np.ones((1, 4, 16, 2, 2, 128), dtype=np.uint16))
            kv_vmem_ref = MockArrayRef(np.ones((2, 1, 16, 2, 2, 128), dtype=np.uint16))
            captured_kv_ref = MockArrayRef(np.zeros((64, 2, 2, 128), dtype=np.uint16))

        tile_plan = np.zeros((2, pcp_size * TilePlanField.NUM_FIELDS), dtype=np.int32)
        tile_plan[:, TilePlanField.CURRENT_NUM_GROUPS] = 1
        tile_plan[:, TilePlanField.HISTORY_NUM_GROUPS] = 1
        tile_plan[:, TilePlanField.Q_TILE_SIZE] = 64
        tile_plan[:, TilePlanField.CAPTURE_CURRENT_KV] = 1
        tile_plan[:, TilePlanField.CURRENT_EFFECTIVE_LEN] = 64
        tile_plan[:, TilePlanField.REQ_ID] = 1
        tile_plan_ref = MockArrayRef(tile_plan)

        block_tables_ref = MockArrayRef(np.array([[1, 2], [3, 4]], dtype=np.int32))
        tile_ids_ref = MockArrayRef(np.array([[0], [1]], dtype=np.int32))
        group_starts_ref = MockArrayRef(np.array([[0], [1]], dtype=np.int32))

        sched_vmem = MockArrayRef(
            np.zeros(
                (pcp_size, 1, RuntimeScheduleField.PACKED_NUM_FIELDS), dtype=np.int32
            )
        )
        sched_vmem.arr[:, :, RuntimeScheduleField.CAPTURE_LEN] = 16
        sched_vmem.arr[:, :, RuntimeScheduleField.Q_TILE_SIZE] = 64
        sched_vmem.arr[:, :, RuntimeScheduleField.REQ_ID] = 1
        sched_vmem.arr[:, :, RuntimeScheduleField.KV_VALID_LEN] = 16

        q_vmem = MockArrayRef(np.ones((64, 2, 2, 128), dtype=np.float32))
        m_scratch = MockArrayRef(
            np.full((2, 64, 2, 128), kernel.ONLINE_STATE_INIT_VALUE, dtype=np.float32)
        )
        l_scratch = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))
        acc_scratch = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))

        state_m_ref = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))
        state_l_ref = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))
        state_acc_ref = MockArrayRef(np.zeros((2, 64, 2, 128), dtype=np.float32))
        o_ref = MockArrayRef(np.zeros((64, 2, 2, 128), dtype=np.float32))

        plan_vmem = MockArrayRef(
            np.zeros((1, pcp_size * TilePlanField.NUM_FIELDS), dtype=np.int32)
        )
        plan_vmem.arr[0, TilePlanField.CURRENT_NUM_GROUPS] = 1
        plan_vmem.arr[0, TilePlanField.HISTORY_NUM_GROUPS] = 1
        plan_vmem.arr[0, TilePlanField.Q_TILE_SIZE] = 64
        plan_vmem.arr[0, TilePlanField.CAPTURE_CURRENT_KV] = 1
        plan_vmem.arr[0, TilePlanField.CURRENT_EFFECTIVE_LEN] = 64
        plan_vmem.arr[0, TilePlanField.REQ_ID] = 1

        block_table_vmem = MockArrayRef(np.zeros((1, 2), dtype=np.int32))
        tile_ids_vmem = MockArrayRef(np.zeros((2, 1), dtype=np.int32))
        group_starts_vmem = MockArrayRef(np.zeros((2, 1), dtype=np.int32))

        sems = MockArrayRef(np.zeros((1, max(2, pcp_size)), dtype=np.int32))

        return {
            "q_ref": q_ref,
            "current_kv_ref": current_kv_ref,
            "kv_cache_ref": kv_cache_ref,
            "captured_kv_ref": captured_kv_ref,
            "tile_plan_ref": tile_plan_ref,
            "block_tables_ref": block_tables_ref,
            "tile_ids_ref": tile_ids_ref,
            "group_starts_ref": group_starts_ref,
            "plan_vmem": plan_vmem,
            "block_table_vmem": block_table_vmem,
            "tile_ids_vmem": tile_ids_vmem,
            "group_starts_vmem": group_starts_vmem,
            "sched_vmem": sched_vmem,
            "q_vmem": q_vmem,
            "kv_vmem": kv_vmem_ref,
            "m_scratch": m_scratch,
            "l_scratch": l_scratch,
            "acc_scratch": acc_scratch,
            "state_m_ref": state_m_ref,
            "state_l_ref": state_l_ref,
            "state_acc_ref": state_acc_ref,
            "o_ref": o_ref,
            "sems": sems,
            "pcp_size": pcp_size,
        }

    def test_run_pcp_page_groups_current_and_history_modes(self):
        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(kernel.pl, "loop", side_effect=fake_loop),
            patch.object(kernel.pl, "run_scoped", side_effect=fake_run_scoped),
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.pl, "cdiv", side_effect=lambda a, b: (a + b - 1) // b),
            patch.object(kernel.pl, "semaphore_wait", side_effect=lambda *a, **k: None),
            patch.object(
                kernel.pl, "semaphore_signal", side_effect=lambda *a, **k: None
            ),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "make_async_remote_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "roll",
                side_effect=lambda a, s, axis: np.roll(
                    a, int(np.asarray(s)), axis=axis
                ),
            ),
            patch.object(kernel.lax, "fori_loop", side_effect=fake_fori_loop),
            patch.object(kernel.lax, "axis_index", return_value=0),
            patch.object(kernel, "strided_load", return_value=jnp.zeros((16, 128))),
            patch.object(
                kernel,
                "convert_to_target_bitwidth",
                return_value=[
                    (jnp.zeros((16, 128)), jnp.zeros((16, 128))),
                    (jnp.zeros((16, 128)), jnp.zeros((16, 128))),
                ],
            ),
        ):
            # 1. Current mode with SEQ_ALONG_LANE
            refs_seq = self._setup_page_groups_refs(KVLayout.SEQ_ALONG_LANE)
            kernel._run_pcp_page_groups_multi_head(
                active_tiles=1,
                active_page_groups=1,
                q_ref=refs_seq["q_ref"],
                current_kv_ref=refs_seq["current_kv_ref"],
                kv_cache_ref=refs_seq["kv_cache_ref"],
                captured_kv_ref=refs_seq["captured_kv_ref"],
                tile_plan_ref=refs_seq["tile_plan_ref"],
                block_tables_ref=refs_seq["block_tables_ref"],
                tile_ids_ref=refs_seq["tile_ids_ref"],
                group_starts_ref=refs_seq["group_starts_ref"],
                sched_dma_sem=None,
                local_dma_sem=None,
                capture_dma_sem=None,
                remote_send_sems=refs_seq["sems"],
                remote_recv_sems=refs_seq["sems"],
                remote_sync_sems=refs_seq["sems"],
                group_slot_free_sems=refs_seq["sems"],
                plan_vmem_ref=refs_seq["plan_vmem"],
                block_table_vmem_ref=refs_seq["block_table_vmem"],
                tile_ids_vmem_ref=refs_seq["tile_ids_vmem"],
                group_starts_vmem_ref=refs_seq["group_starts_vmem"],
                sched_vmem_ref=refs_seq["sched_vmem"],
                q_vmem_ref=refs_seq["q_vmem"],
                kv_vmem_ref=refs_seq["kv_vmem"],
                m_scratch_ref=refs_seq["m_scratch"],
                l_scratch_ref=refs_seq["l_scratch"],
                acc_scratch_ref=refs_seq["acc_scratch"],
                state_m_ref=refs_seq["state_m_ref"],
                state_l_ref=refs_seq["state_l_ref"],
                state_acc_ref=refs_seq["state_acc_ref"],
                o_ref=refs_seq["o_ref"],
                pcp_size=refs_seq["pcp_size"],
                interleave_size=16,
                lane=0,
                q_compute_size=64,
                sm_scale=0.125,
                packed_kv_groups=2,
                page_size=16,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                k_scale=1.0,
                v_scale=1.0,
                mesh_axis_names=("pcp",),
                pcp_axis_name="pcp",
                state_mode="current",
            )

            # 2. History mode with HEAD_ALONG_SUBLANE
            refs_flat = self._setup_page_groups_refs(KVLayout.HEAD_ALONG_SUBLANE)
            kernel._run_pcp_page_groups_multi_head(
                active_tiles=1,
                active_page_groups=1,
                q_ref=refs_flat["q_ref"],
                current_kv_ref=refs_flat["current_kv_ref"],
                kv_cache_ref=refs_flat["kv_cache_ref"],
                captured_kv_ref=refs_flat["captured_kv_ref"],
                tile_plan_ref=refs_flat["tile_plan_ref"],
                block_tables_ref=refs_flat["block_tables_ref"],
                tile_ids_ref=refs_flat["tile_ids_ref"],
                group_starts_ref=refs_flat["group_starts_ref"],
                sched_dma_sem=None,
                local_dma_sem=None,
                capture_dma_sem=None,
                remote_send_sems=refs_flat["sems"],
                remote_recv_sems=refs_flat["sems"],
                remote_sync_sems=refs_flat["sems"],
                group_slot_free_sems=refs_flat["sems"],
                plan_vmem_ref=refs_flat["plan_vmem"],
                block_table_vmem_ref=refs_flat["block_table_vmem"],
                tile_ids_vmem_ref=refs_flat["tile_ids_vmem"],
                group_starts_vmem_ref=refs_flat["group_starts_vmem"],
                sched_vmem_ref=refs_flat["sched_vmem"],
                q_vmem_ref=refs_flat["q_vmem"],
                kv_vmem_ref=refs_flat["kv_vmem"],
                m_scratch_ref=refs_flat["m_scratch"],
                l_scratch_ref=refs_flat["l_scratch"],
                acc_scratch_ref=refs_flat["acc_scratch"],
                state_m_ref=refs_flat["state_m_ref"],
                state_l_ref=refs_flat["state_l_ref"],
                state_acc_ref=refs_flat["state_acc_ref"],
                o_ref=refs_flat["o_ref"],
                pcp_size=refs_flat["pcp_size"],
                interleave_size=16,
                lane=0,
                q_compute_size=64,
                sm_scale=0.125,
                packed_kv_groups=2,
                page_size=16,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
                k_scale=1.0,
                v_scale=1.0,
                mesh_axis_names=("pcp",),
                pcp_axis_name="pcp",
                state_mode="history",
            )

            # 3. Current mode with HEAD_ALONG_SUBLANE
            # (exercises start/finish capture for flat KV)
            refs_flat_curr = self._setup_page_groups_refs(KVLayout.HEAD_ALONG_SUBLANE)
            kernel._run_pcp_page_groups_multi_head(
                active_tiles=1,
                active_page_groups=1,
                q_ref=refs_flat_curr["q_ref"],
                current_kv_ref=refs_flat_curr["current_kv_ref"],
                kv_cache_ref=refs_flat_curr["kv_cache_ref"],
                captured_kv_ref=refs_flat_curr["captured_kv_ref"],
                tile_plan_ref=refs_flat_curr["tile_plan_ref"],
                block_tables_ref=refs_flat_curr["block_tables_ref"],
                tile_ids_ref=refs_flat_curr["tile_ids_ref"],
                group_starts_ref=refs_flat_curr["group_starts_ref"],
                sched_dma_sem=None,
                local_dma_sem=None,
                capture_dma_sem=None,
                remote_send_sems=refs_flat_curr["sems"],
                remote_recv_sems=refs_flat_curr["sems"],
                remote_sync_sems=refs_flat_curr["sems"],
                group_slot_free_sems=refs_flat_curr["sems"],
                plan_vmem_ref=refs_flat_curr["plan_vmem"],
                block_table_vmem_ref=refs_flat_curr["block_table_vmem"],
                tile_ids_vmem_ref=refs_flat_curr["tile_ids_vmem"],
                group_starts_vmem_ref=refs_flat_curr["group_starts_vmem"],
                sched_vmem_ref=refs_flat_curr["sched_vmem"],
                q_vmem_ref=refs_flat_curr["q_vmem"],
                kv_vmem_ref=refs_flat_curr["kv_vmem"],
                m_scratch_ref=refs_flat_curr["m_scratch"],
                l_scratch_ref=refs_flat_curr["l_scratch"],
                acc_scratch_ref=refs_flat_curr["acc_scratch"],
                state_m_ref=refs_flat_curr["state_m_ref"],
                state_l_ref=refs_flat_curr["state_l_ref"],
                state_acc_ref=refs_flat_curr["state_acc_ref"],
                o_ref=refs_flat_curr["o_ref"],
                pcp_size=refs_flat_curr["pcp_size"],
                interleave_size=16,
                lane=0,
                q_compute_size=64,
                sm_scale=0.125,
                packed_kv_groups=2,
                page_size=16,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
                k_scale=1.0,
                v_scale=1.0,
                mesh_axis_names=("pcp",),
                pcp_axis_name="pcp",
                state_mode="current",
            )

            # 4. Multi-round and multi-group with pcp_size=4 and active_page_groups=2
            refs_multi = self._setup_page_groups_refs(
                KVLayout.SEQ_ALONG_LANE, pcp_size=4
            )
            refs_multi["plan_vmem"].arr[0, TilePlanField.CURRENT_NUM_GROUPS] = 2
            kernel._run_pcp_page_groups_multi_head(
                active_tiles=1,
                active_page_groups=2,
                q_ref=refs_multi["q_ref"],
                current_kv_ref=refs_multi["current_kv_ref"],
                kv_cache_ref=refs_multi["kv_cache_ref"],
                captured_kv_ref=refs_multi["captured_kv_ref"],
                tile_plan_ref=refs_multi["tile_plan_ref"],
                block_tables_ref=refs_multi["block_tables_ref"],
                tile_ids_ref=refs_multi["tile_ids_ref"],
                group_starts_ref=refs_multi["group_starts_ref"],
                sched_dma_sem=None,
                local_dma_sem=None,
                capture_dma_sem=None,
                remote_send_sems=refs_multi["sems"],
                remote_recv_sems=refs_multi["sems"],
                remote_sync_sems=refs_multi["sems"],
                group_slot_free_sems=refs_multi["sems"],
                plan_vmem_ref=refs_multi["plan_vmem"],
                block_table_vmem_ref=refs_multi["block_table_vmem"],
                tile_ids_vmem_ref=refs_multi["tile_ids_vmem"],
                group_starts_vmem_ref=refs_multi["group_starts_vmem"],
                sched_vmem_ref=refs_multi["sched_vmem"],
                q_vmem_ref=refs_multi["q_vmem"],
                kv_vmem_ref=refs_multi["kv_vmem"],
                m_scratch_ref=refs_multi["m_scratch"],
                l_scratch_ref=refs_multi["l_scratch"],
                acc_scratch_ref=refs_multi["acc_scratch"],
                state_m_ref=refs_multi["state_m_ref"],
                state_l_ref=refs_multi["state_l_ref"],
                state_acc_ref=refs_multi["state_acc_ref"],
                o_ref=refs_multi["o_ref"],
                pcp_size=4,
                interleave_size=16,
                lane=0,
                q_compute_size=64,
                sm_scale=0.125,
                packed_kv_groups=2,
                page_size=16,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                k_scale=1.0,
                v_scale=1.0,
                mesh_axis_names=("pcp",),
                pcp_axis_name="pcp",
                state_mode="current",
            )

    def test_current_state_kernel_wrapper(self):
        refs = self._setup_page_groups_refs(KVLayout.SEQ_ALONG_LANE)
        pass_meta = MockArrayRef(np.array([1, 1], dtype=np.int32))

        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(kernel.pl, "loop", side_effect=fake_loop),
            patch.object(kernel.pl, "run_scoped", side_effect=fake_run_scoped),
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.pl, "cdiv", side_effect=lambda a, b: (a + b - 1) // b),
            patch.object(kernel.pl, "semaphore_wait", side_effect=lambda *a, **k: None),
            patch.object(
                kernel.pl, "semaphore_signal", side_effect=lambda *a, **k: None
            ),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "make_async_remote_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "roll",
                side_effect=lambda a, s, axis: np.roll(
                    a, int(np.asarray(s)), axis=axis
                ),
            ),
            patch.object(kernel.lax, "fori_loop", side_effect=fake_fori_loop),
            patch.object(kernel.lax, "axis_index", return_value=0),
            patch.object(kernel, "strided_load", return_value=jnp.zeros((16, 128))),
            patch.object(
                kernel,
                "convert_to_target_bitwidth",
                return_value=[
                    (jnp.zeros((16, 128)), jnp.zeros((16, 128))),
                    (jnp.zeros((16, 128)), jnp.zeros((16, 128))),
                ],
            ),
        ):
            kernel._pcp_streaming_attention_current_state_page_groups_multi_head_kernel(
                pass_meta,
                refs["q_ref"],
                refs["current_kv_ref"],
                refs["kv_cache_ref"],
                refs["tile_plan_ref"],
                refs["block_tables_ref"],
                refs["tile_ids_ref"],
                refs["group_starts_ref"],
                refs["state_m_ref"],
                refs["state_l_ref"],
                refs["state_acc_ref"],
                refs["captured_kv_ref"],
                None,
                None,
                None,
                refs["sems"],
                refs["sems"],
                refs["sems"],
                refs["sems"],
                refs["plan_vmem"],
                refs["block_table_vmem"],
                refs["tile_ids_vmem"],
                refs["group_starts_vmem"],
                refs["sched_vmem"],
                refs["q_vmem"],
                refs["kv_vmem"],
                refs["m_scratch"],
                refs["l_scratch"],
                refs["acc_scratch"],
                pcp_size=refs["pcp_size"],
                interleave_size=16,
                num_lanes=1,
                q_compute_size=64,
                num_tiles=1,
                max_page_groups=1,
                sm_scale=0.125,
                packed_kv_groups=2,
                page_size=16,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                k_scale=1.0,
                v_scale=1.0,
                mesh_axis_names=("pcp",),
                pcp_axis_name="pcp",
            )

    def test_history_output_kernel_wrapper(self):
        refs = self._setup_page_groups_refs(KVLayout.SEQ_ALONG_LANE)
        pass_meta = MockArrayRef(np.array([1, 1], dtype=np.int32))

        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(kernel.pl, "loop", side_effect=fake_loop),
            patch.object(kernel.pl, "run_scoped", side_effect=fake_run_scoped),
            patch.object(kernel.pl, "multiple_of", side_effect=lambda x, a: x),
            patch.object(kernel.pl, "cdiv", side_effect=lambda a, b: (a + b - 1) // b),
            patch.object(kernel.pl, "semaphore_wait", side_effect=lambda *a, **k: None),
            patch.object(
                kernel.pl, "semaphore_signal", side_effect=lambda *a, **k: None
            ),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "make_async_remote_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(
                kernel.pltpu,
                "roll",
                side_effect=lambda a, s, axis: np.roll(
                    a, int(np.asarray(s)), axis=axis
                ),
            ),
            patch.object(kernel.lax, "fori_loop", side_effect=fake_fori_loop),
            patch.object(kernel.lax, "axis_index", return_value=0),
            patch.object(kernel, "strided_load", return_value=jnp.zeros((16, 128))),
            patch.object(
                kernel,
                "convert_to_target_bitwidth",
                return_value=[
                    (jnp.zeros((16, 128)), jnp.zeros((16, 128))),
                    (jnp.zeros((16, 128)), jnp.zeros((16, 128))),
                ],
            ),
        ):
            kernel._pcp_streaming_attention_history_output_page_groups_multi_head_kernel(
                pass_meta,
                refs["q_ref"],
                refs["kv_cache_ref"],
                refs["tile_plan_ref"],
                refs["block_tables_ref"],
                refs["tile_ids_ref"],
                refs["group_starts_ref"],
                refs["state_m_ref"],
                refs["state_l_ref"],
                refs["state_acc_ref"],
                refs["o_ref"],
                None,
                None,
                refs["sems"],
                refs["sems"],
                refs["sems"],
                refs["sems"],
                refs["plan_vmem"],
                refs["block_table_vmem"],
                refs["tile_ids_vmem"],
                refs["group_starts_vmem"],
                refs["sched_vmem"],
                refs["q_vmem"],
                refs["kv_vmem"],
                refs["m_scratch"],
                refs["l_scratch"],
                refs["acc_scratch"],
                pcp_size=refs["pcp_size"],
                interleave_size=16,
                num_lanes=1,
                q_compute_size=64,
                num_tiles=1,
                max_page_groups=1,
                sm_scale=0.125,
                packed_kv_groups=2,
                page_size=16,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                k_scale=1.0,
                v_scale=1.0,
                mesh_axis_names=("pcp",),
                pcp_axis_name="pcp",
            )

    def test_zero_history_output_multi_head_kernel(self):
        refs = self._setup_page_groups_refs(KVLayout.SEQ_ALONG_LANE)
        pass_meta = MockArrayRef(np.array([1, 1], dtype=np.int32))
        # Ensure current_num_groups > 0 and history_num_groups == 0
        refs["plan_vmem"].arr[0, TilePlanField.CURRENT_NUM_GROUPS] = 1
        refs["plan_vmem"].arr[0, TilePlanField.HISTORY_NUM_GROUPS] = 0
        refs["plan_vmem"].arr[0, TilePlanField.Q_TILE_SIZE] = 64

        with (
            patch.object(kernel.pl, "when", side_effect=fake_when),
            patch.object(
                kernel.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: MockAsyncCopy(),
            ),
            patch.object(kernel.lax, "fori_loop", side_effect=fake_fori_loop),
            patch.object(kernel.lax, "axis_index", return_value=0),
        ):
            # 1. store_q_tile_size > 0
            kernel._pcp_streaming_attention_zero_history_output_multi_head_kernel(
                pass_meta,
                refs["o_ref"],
                refs["tile_plan_ref"],
                refs["tile_ids_ref"],
                refs["group_starts_ref"],
                refs["state_m_ref"],
                refs["state_l_ref"],
                refs["state_acc_ref"],
                refs["o_ref"],
                None,
                None,
                refs["plan_vmem"],
                refs["tile_ids_vmem"],
                refs["group_starts_vmem"],
                refs["q_vmem"],
                refs["m_scratch"],
                refs["l_scratch"],
                refs["acc_scratch"],
                num_tiles=1,
                pcp_axis_name="pcp",
            )

            # 2. store_q_tile_size == 0
            # (history_num_groups > 0, so write_zero_history is False)
            refs["plan_vmem"].arr[0, TilePlanField.HISTORY_NUM_GROUPS] = 2
            kernel._pcp_streaming_attention_zero_history_output_multi_head_kernel(
                pass_meta,
                refs["o_ref"],
                refs["tile_plan_ref"],
                refs["tile_ids_ref"],
                refs["group_starts_ref"],
                refs["state_m_ref"],
                refs["state_l_ref"],
                refs["state_acc_ref"],
                refs["o_ref"],
                None,
                None,
                refs["plan_vmem"],
                refs["tile_ids_vmem"],
                refs["group_starts_vmem"],
                refs["q_vmem"],
                refs["m_scratch"],
                refs["l_scratch"],
                refs["acc_scratch"],
                num_tiles=1,
                pcp_axis_name="pcp",
            )


# ==============================================================================
# 7. Tests for Pallas Calls & Top-Level Streaming RPA Entry
# ==============================================================================


class TestPallasCallsAndTopLevelEntry:
    """Tests for the 3 pallas_call entry wrappers and top-level entry point."""

    def test_pallas_calls_current_state(self, monkeypatch):
        q = jnp.zeros((64, 2, 2, 64), dtype=jnp.float32)
        tile_plan = jnp.zeros((2, 2 * TilePlanField.NUM_FIELDS), dtype=np.int32)
        block_tables = jnp.zeros((2, 4), dtype=np.int32)
        tile_ids = jnp.zeros((2, 1), dtype=np.int32)
        group_starts = jnp.zeros((2, 1), dtype=np.int32)
        pass_meta = jnp.zeros((2,), dtype=np.int32)

        def mock_pallas_call(kernel_fn, out_shape, **kwargs):
            def runner(*args):
                return (
                    jnp.zeros(out_shape[0].shape, dtype=out_shape[0].dtype),
                    jnp.zeros(out_shape[1].shape, dtype=out_shape[1].dtype),
                    jnp.zeros(out_shape[2].shape, dtype=out_shape[2].dtype),
                    jnp.zeros(out_shape[3].shape, dtype=out_shape[3].dtype),
                )

            return runner

        monkeypatch.setattr(kernel.pl, "pallas_call", mock_pallas_call)

        # 1. SEQ_ALONG_LANE
        current_kv_seq = jnp.zeros((2, 32, 4, 64), dtype=jnp.float32)
        kv_cache_seq = jnp.zeros((1, 2, 32, 4, 128), dtype=jnp.float32)
        m, l_val, acc, cap = (
            kernel._pcp_streaming_attention_current_state_page_groups_multi_head_pallas_call(
                q,
                current_kv_seq,
                kv_cache_seq,
                tile_plan,
                block_tables,
                tile_ids,
                group_starts,
                pass_meta,
                pcp_size=2,
                q_block_size=64,
                sm_scale=0.125,
                interleave_size=16,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
            )
        )
        assert m.shape == (2, 64, 2, kernel.ONLINE_STATE_PADDED_LANES)

        # 2. HEAD_ALONG_SUBLANE
        current_kv_flat = jnp.zeros((64, 2, 2, 64), dtype=jnp.uint16)
        kv_cache_flat = jnp.zeros((1, 16, 2, 2, 64), dtype=jnp.uint16)
        m2, l2, acc2, cap2 = (
            kernel._pcp_streaming_attention_current_state_page_groups_multi_head_pallas_call(
                q,
                current_kv_flat,
                kv_cache_flat,
                tile_plan,
                block_tables,
                tile_ids,
                group_starts,
                pass_meta,
                pcp_size=2,
                q_block_size=64,
                sm_scale=0.125,
                interleave_size=16,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )
        )
        assert m2.shape == (2, 64, 2, kernel.ONLINE_STATE_PADDED_LANES)

    def test_pallas_calls_history_output(self, monkeypatch):
        q = jnp.zeros((64, 2, 2, 64), dtype=jnp.float32)
        tile_plan = jnp.zeros((2, 2 * TilePlanField.NUM_FIELDS), dtype=np.int32)
        block_tables = jnp.zeros((2, 4), dtype=np.int32)
        tile_ids = jnp.zeros((2, 1), dtype=np.int32)
        group_starts = jnp.zeros((2, 1), dtype=np.int32)
        pass_meta = jnp.zeros((2,), dtype=np.int32)
        m_state = jnp.zeros(
            (2, 64, 2, kernel.ONLINE_STATE_PADDED_LANES), dtype=jnp.float32
        )
        l_state = jnp.zeros(
            (2, 64, 2, kernel.ONLINE_STATE_PADDED_LANES), dtype=jnp.float32
        )
        acc_state = jnp.zeros((2, 64, 2, 64), dtype=jnp.float32)

        def mock_pallas_call(kernel_fn, out_shape, **kwargs):
            def runner(*args):
                return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

            return runner

        monkeypatch.setattr(kernel.pl, "pallas_call", mock_pallas_call)

        func_name = (
            "_pcp_streaming_attention_history_output_page_groups_multi_head_pallas_call"
        )
        call_fn = getattr(kernel, func_name)

        # 1. SEQ_ALONG_LANE
        kv_cache_seq = jnp.zeros((1, 2, 32, 4, 128), dtype=np.float32)
        out_seq = call_fn(
            q,
            kv_cache_seq,
            tile_plan,
            block_tables,
            tile_ids,
            group_starts,
            m_state,
            l_state,
            acc_state,
            pass_meta,
            pcp_size=2,
            q_block_size=64,
            sm_scale=0.125,
            interleave_size=16,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
        )
        assert out_seq.shape == q.shape

        # 2. HEAD_ALONG_SUBLANE
        kv_cache_flat = jnp.zeros((1, 16, 2, 2, 64), dtype=jnp.uint16)
        out_flat = call_fn(
            q,
            kv_cache_flat,
            tile_plan,
            block_tables,
            tile_ids,
            group_starts,
            m_state,
            l_state,
            acc_state,
            pass_meta,
            pcp_size=2,
            q_block_size=64,
            sm_scale=0.125,
            interleave_size=16,
            kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        )
        assert out_flat.shape == q.shape

    def test_pallas_calls_zero_history_output(self, monkeypatch):
        o = jnp.zeros((64, 2, 2, 64), dtype=jnp.float32)
        tile_plan = jnp.zeros((2, 2 * TilePlanField.NUM_FIELDS), dtype=np.int32)
        tile_ids = jnp.zeros((2, 1), dtype=np.int32)
        group_starts = jnp.zeros((2, 1), dtype=np.int32)
        pass_meta = jnp.zeros((2,), dtype=np.int32)
        m_state = jnp.zeros(
            (2, 64, 2, kernel.ONLINE_STATE_PADDED_LANES), dtype=jnp.float32
        )
        l_state = jnp.zeros(
            (2, 64, 2, kernel.ONLINE_STATE_PADDED_LANES), dtype=jnp.float32
        )
        acc_state = jnp.zeros((2, 64, 2, 64), dtype=jnp.float32)

        def mock_pallas_call(kernel_fn, out_shape, **kwargs):
            def runner(*args):
                return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

            return runner

        monkeypatch.setattr(kernel.pl, "pallas_call", mock_pallas_call)
        out = (
            kernel._pcp_streaming_attention_zero_history_output_multi_head_pallas_call(
                o,
                tile_plan,
                tile_ids,
                group_starts,
                m_state,
                l_state,
                acc_state,
                pass_meta,
                q_block_size=64,
                pcp_axis_name="pcp",
            )
        )
        assert out.shape == o.shape

    def test_pcp_streaming_attention_top_level_errors(self):
        # 1. SEQ_ALONG_LANE shape mismatch
        q = jnp.zeros((64, 2, 2, 128), dtype=jnp.bfloat16)
        kv_cache_seq = jnp.zeros((2, 4, 32, 4, 128), dtype=jnp.float8_e4m3fn)
        current_kv_bad_seq = jnp.zeros((4, 32, 4, 32), dtype=jnp.float8_e4m3fn)
        with pytest.raises(ValueError, match="SEQ_ALONG_LANE fresh KV must match"):
            kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q,
                current_kv_bad_seq,
                kv_cache_seq,
                jnp.array([64]),
                jnp.array([0]),
                jnp.array([0, 64]),
                jnp.array([0, 0, 1]),
                jnp.zeros((3, 1), dtype=jnp.int32),
                1,
                pcp_size=2,
                interleave_size=32,
                sm_scale=0.125,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                k_scale=1.0,
                v_scale=1.0,
            )

        # 2. HEAD_ALONG_SUBLANE shape mismatch
        kv_cache_flat = jnp.zeros((2, 16, 2, 2, 128), dtype=jnp.bfloat16)
        current_kv_bad_flat = jnp.zeros((32, 2, 2, 128), dtype=jnp.bfloat16)
        with pytest.raises(
            ValueError, match="Fresh packed KV layout must match local Q rows"
        ):
            kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q,
                current_kv_bad_flat,
                kv_cache_flat,
                jnp.array([64]),
                jnp.array([0]),
                jnp.array([0, 64]),
                jnp.array([0, 0, 1]),
                jnp.zeros((3, 1), dtype=jnp.int32),
                1,
                pcp_size=2,
                interleave_size=16,
                sm_scale=0.125,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )

    def test_pcp_streaming_attention_top_level_full_execution(self, monkeypatch):
        # Mock the internal pallas calls to return consistent dummy shapes
        def mock_pallas_call(kernel_fn, out_shape, **kwargs):
            def runner(*args):
                if isinstance(out_shape, tuple):
                    return tuple(jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape)
                return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

            return runner

        monkeypatch.setattr(kernel.pl, "pallas_call", mock_pallas_call)

        # 1. Full run: SEQ_ALONG_LANE FP8
        q_local = jnp.zeros((64, 2, 2, 128), dtype=jnp.bfloat16)
        current_kv_seq = jnp.zeros((4, 32, 4, 64), dtype=jnp.float8_e4m3fn)
        kv_cache_seq = jnp.zeros((2, 4, 32, 4, 128), dtype=jnp.float8_e4m3fn)
        out1, c1 = (
            kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q_local,
                current_kv_seq,
                kv_cache_seq,
                kv_lens=jnp.array([64]),
                page_indices=jnp.array([0, 1]),
                cu_q_lens=jnp.array([0, 64]),
                distribution=jnp.array([0, 0, 1]),
                writeback_segment_descriptors=jnp.zeros((3, 2), dtype=jnp.int32),
                writeback_num_segments=jnp.array(1),
                pcp_size=2,
                interleave_size=32,
                q_block_size=64,
                sm_scale=0.125,
                kv_layout=KVLayout.SEQ_ALONG_LANE,
                k_scale=1.0,
                v_scale=1.0,
            )
        )
        assert out1.shape == q_local.shape
        assert c1.shape == kv_cache_seq.shape

        # 2. Full run: HEAD_ALONG_SUBLANE BF16
        current_kv_flat = jnp.zeros((64, 2, 2, 128), dtype=jnp.bfloat16)
        kv_cache_flat = jnp.zeros((2, 16, 2, 2, 128), dtype=jnp.bfloat16)
        out2, c2 = (
            kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q_local,
                current_kv_flat,
                kv_cache_flat,
                kv_lens=jnp.array([64]),
                page_indices=jnp.array([0, 1]),
                cu_q_lens=jnp.array([0, 64]),
                distribution=jnp.array([0, 0, 1]),
                writeback_segment_descriptors=jnp.zeros((3, 2), dtype=jnp.int32),
                writeback_num_segments=jnp.array(1),
                pcp_size=2,
                interleave_size=16,
                q_block_size=64,
                sm_scale=0.125,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )
        )
        assert out2.shape == q_local.shape
        assert c2.shape == kv_cache_flat.shape
