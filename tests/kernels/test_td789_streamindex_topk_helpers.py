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
"""CPU unit tests for StreamIndex Top-K helpers, metadata, config, and compact."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    bref_override,
    config,
    dcp_sc_compact,
    metadata,
)


class _FakeAtRef:
    """Lightweight mock supporting .at[...] and .reshape(...) for Pallas DMA tests."""

    def __init__(self, shape: tuple[int, ...] = (2, 16, 4, 128)):
        self.shape = shape

    @property
    def at(self) -> _FakeAtRef:
        return self

    def __getitem__(self, item):
        return self

    def reshape(self, *new_shape: int) -> _FakeAtRef:
        return _FakeAtRef(new_shape)


def _bind_bref_slots(bref):
    return dataclasses.replace(
        bref,
        window_ref=_FakeAtRef(),
        sem_recvs=_FakeAtRef(),
        sem_sends=_FakeAtRef(),
        copy_in_slot=jnp.zeros((1,), dtype=jnp.uint32),
        wait_in_slot=jnp.zeros((1,), dtype=jnp.uint32),
        copy_out_slot=jnp.zeros((1,), dtype=jnp.uint32),
        wait_out_slot=jnp.zeros((1,), dtype=jnp.uint32),
    )


def test_config_enums_and_scheduling_group_ids():
    assert config.MlaCase.DECODE.symbol == "d"
    assert config.MlaCase.PREFILL.symbol == "p"
    assert config.MlaCase.MIXED.symbol == "m"

    assert config.KVLayout.parse(None) == config.KVLayout.HEAD_ALONG_SUBLANE
    assert (
        config.KVLayout.parse(config.KVLayout.SEQ_ALONG_LANE)
        == config.KVLayout.SEQ_ALONG_LANE
    )
    assert (
        config.KVLayout.parse("  HEAD_ALONG_SUBLANE ")
        == config.KVLayout.HEAD_ALONG_SUBLANE
    )
    assert config.KVLayout.parse("seq_along_lane") == config.KVLayout.SEQ_ALONG_LANE
    assert config.KVLayout.parse("seq_alone_lane") == config.KVLayout.SEQ_ALONG_LANE

    with pytest.raises(ValueError, match="Unknown kv_layout"):
        config.KVLayout.parse("unknown_layout")

    config.reset_scheduling_group_ids()
    assert config.reserve_scheduling_group_ids(1) == 1
    assert config.reserve_scheduling_group_ids(3) == 2
    assert config.reserve_scheduling_group_ids(2) == 5
    config.reset_scheduling_group_ids()
    assert config.reserve_scheduling_group_ids(1) == 1

    with pytest.raises(ValueError, match="must be positive"):
        config.reserve_scheduling_group_ids(0)


def test_metadata_cp_helpers_and_schedule_branches():
    idx = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    np.testing.assert_array_equal(metadata.cp_local_to_global(idx, 0, 1, 2), idx)
    np.testing.assert_array_equal(
        metadata.cp_local_to_global(idx, 1, 2, 2),
        jnp.array([2, 3, 6, 7], dtype=jnp.int32),
    )

    glen = jnp.array([0, 3, 5, 8], dtype=jnp.int32)
    np.testing.assert_array_equal(metadata.cp_local_length(glen, 0, 1, 2), glen)
    np.testing.assert_array_equal(
        metadata.cp_local_length(glen, 1, 2, 2),
        jnp.array([0, 1, 2, 4], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(
        metadata.cp_owner_rank(idx, 1, 2), jnp.zeros_like(idx)
    )
    np.testing.assert_array_equal(
        metadata.cp_owner_rank(jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32), 2, 2),
        jnp.array([0, 0, 1, 1, 0], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(metadata.cp_global_to_local(idx, 1, 2), idx)
    np.testing.assert_array_equal(
        metadata.cp_global_to_local(jnp.array([2, 3, 6, 7], dtype=jnp.int32), 2, 2),
        idx,
    )

    with patch.object(
        metadata.lax,
        "all_to_all",
        side_effect=lambda x, axis_name, split_axis, concat_axis, tiled=False: x + 2,
    ):
        rank_val = metadata.cp_rank_as_data("dcp", 4)
        assert int(rank_val) == 2

    fake_tpu_info = SimpleNamespace(smem_capacity_bytes=1024 * 1024, num_lanes=128)
    with patch.object(metadata.pltpu, "get_tpu_info", return_value=fake_tpu_info):
        steps = metadata.generate_max_steps(num_seqs=16, max_pages_per_seq=8)
        assert steps > 0 and steps % 128 == 0

    # Test compute_metadata across all branches:
    # 1. static_q_len > 1, bkv_sz < page_size, cp_size > 1, pad_size > 0
    seq_lens = jnp.array([128, 256, 64], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 8, 16, 24], dtype=jnp.int32)
    meta_static_gt1 = metadata.compute_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=0,
        end_seq_idx=3,
        bq_sz=4,
        bkv_sz=32,
        pages_per_seq=4,
        page_size=64,
        max_num_tokens=24,
        compression_ratio=1,
        static_q_len=8,
        seq_batch_size=2,
        cp_rank=0,
        cp_size=2,
        interleave_c=2,
    )
    assert int(meta_static_gt1.num_steps[0]) > 0
    np.testing.assert_array_equal(
        meta_static_gt1.start_seq_idx, meta_static_gt1.batch_tile_idx
    )

    # 2. static_q_len == 1, pad_size == 0, chunked single and multi-chunk
    seq_lens_4 = jnp.array([128, 256, 128, 256], dtype=jnp.int32)
    cu_q_lens_4 = jnp.array([0, 16, 32, 48, 64], dtype=jnp.int32)
    meta_single_chunk = metadata.compute_metadata(
        seq_lens=seq_lens_4,
        cu_q_lens=cu_q_lens_4,
        start_seq_idx=0,
        end_seq_idx=4,
        bq_sz=8,
        bkv_sz=64,
        pages_per_seq=4,
        page_size=64,
        max_num_tokens=64,
        static_q_len=1,
        seq_batch_size=1,
        chunk_token_start=0,
        chunk_tokens=32,
    )
    assert meta_single_chunk.num_steps.shape == (1,)

    meta_multi_chunk = metadata.compute_metadata(
        seq_lens=seq_lens_4,
        cu_q_lens=cu_q_lens_4,
        start_seq_idx=0,
        end_seq_idx=4,
        bq_sz=8,
        bkv_sz=64,
        pages_per_seq=4,
        page_size=64,
        max_num_tokens=64,
        static_q_len=None,
        seq_batch_size=1,
        chunk_token_start=(0, 32),
        chunk_tokens=32,
    )
    assert meta_multi_chunk.num_steps.shape == (2,)
    assert int(jnp.sum(meta_multi_chunk.num_steps)) > 0

    # 3. Empty pass branch (start_seq_idx >= end_seq_idx)
    meta_empty = metadata.compute_metadata(
        seq_lens=seq_lens_4,
        cu_q_lens=cu_q_lens_4,
        start_seq_idx=2,
        end_seq_idx=2,
        bq_sz=8,
        bkv_sz=64,
        pages_per_seq=4,
        page_size=64,
        max_num_tokens=64,
    )
    assert int(meta_empty.num_steps[0]) == 0


def test_bref_override_all_buffered_refs():
    meta_ref = metadata.MetadataRef.create(
        num_steps=jnp.array([2, 3], dtype=jnp.int32),
        batch_tile_idx=jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32),
        bq_idx=jnp.array([0, 1, 0, 1, 2], dtype=jnp.int32),
        bkv_idx=jnp.array([0, 0, 1, 1, 0], dtype=jnp.int32),
    )
    assert bref_override._step_offset(meta_ref, 0) == 0
    assert int(bref_override._step_offset(meta_ref, 1)) == 2

    q_hbm = _FakeAtRef((16, 4, 128))
    cu_q_lens_ref = jnp.array([0, 8, 16], dtype=jnp.int32)
    start_end_seq_idx_ref = jnp.array([0, 2, 0], dtype=jnp.int32)
    refs = (q_hbm, cu_q_lens_ref, meta_ref, start_end_seq_idx_ref)

    # _token_window branches: seq_batch_size > 1, bq_sz == 1, and bq_sz > 1
    lo_b, sz_b = bref_override._token_window(
        refs, 0, 16, bq_sz=4, seq_batch_size=2, chunk_idx=0
    )
    assert int(lo_b) == 0 and sz_b == 2

    ts_1, sz_1 = bref_override._token_window(
        refs, 1, 16, bq_sz=1, seq_batch_size=1, chunk_idx=0
    )
    assert int(ts_1) == 1 and sz_1 == 1

    ts_m, sz_m = bref_override._token_window(
        refs, 0, 16, bq_sz=4, seq_batch_size=1, chunk_idx=1
    )
    assert int(ts_m) == 8 and int(sz_m) == 4

    copy_events: list[str] = []

    class _FakeCopy:
        def start(self):
            copy_events.append("start")

        def wait(self):
            copy_events.append("wait")

    spec = pl.BlockSpec((1, 32, 4, 128), lambda i: (0, 0, 0, 0))
    cache_kv_hbm = _FakeAtRef((4, 16, 4, 128))
    page_indices_ref = jnp.arange(8, dtype=jnp.int32)
    kv_src_ref = (cache_kv_hbm, page_indices_ref, meta_ref)

    with (
        patch.object(bref_override.pltpu, "make_async_copy", return_value=_FakeCopy()),
        patch.object(bref_override.jax, "Ref", _FakeAtRef),
    ):
        # 1. StreamIndexKVBufferedRef
        kv_bref = _bind_bref_slots(
            bref_override.StreamIndexKVBufferedRef.input(
                spec=spec,
                dtype_or_type=jax.ShapeDtypeStruct((1, 32, 4, 128), jnp.uint8),
                buffer_count=2,
                use_lookahead=False,
                bkv_p=2,
                page_size_per_kv_packing=16,
                kv_packing=4,
                pages_per_seq=2,
                seq_batch_size=1,
                chunk_idx=0,
            )
        )
        kv_bref.copy_in(kv_src_ref, (0,))
        kv_bref.wait_in(kv_src_ref, (0,))
        dataclasses.replace(kv_bref, _buffer_count=0).wait_in(kv_src_ref, (0,))

        # 2. StreamIndexKVSeqAlongLaneBufferedRef
        kv_sal_bref = _bind_bref_slots(
            bref_override.StreamIndexKVSeqAlongLaneBufferedRef.input(
                spec=spec,
                dtype_or_type=jax.ShapeDtypeStruct((1, 32, 4, 128), jnp.uint8),
                buffer_count=2,
                use_lookahead=False,
                bkv_p=2,
                page_size=128,
                pages_per_seq=2,
                seq_batch_size=1,
                chunk_idx=1,
            )
        )
        kv_sal_bref.copy_in(kv_src_ref, (0,))
        kv_sal_bref.wait_in(kv_src_ref, (0,))
        dataclasses.replace(kv_sal_bref, _buffer_count=0).wait_in(kv_src_ref, (0,))

        # 3. StreamIndexQBufferedRef
        q_bref = _bind_bref_slots(
            bref_override.StreamIndexQBufferedRef.input(
                spec=pl.BlockSpec((4, 4, 128), lambda i: (0, 0, 0)),
                dtype_or_type=jax.ShapeDtypeStruct((4, 4, 128), jnp.float32),
                buffer_count=2,
                use_lookahead=False,
                bq_sz=4,
                chunk_tokens=16,
                seq_batch_size=1,
                chunk_idx=0,
            )
        )
        q_bref.copy_in(refs, (0,))
        q_bref.wait_in(refs, (0,))
        dataclasses.replace(q_bref, _buffer_count=0).wait_in(refs, (0,))

        # 4. StreamIndexOBufferedRef
        scores_hbm = _FakeAtRef((16, 4, 128))
        dst_refs = (scores_hbm, cu_q_lens_ref, meta_ref, start_end_seq_idx_ref)
        o_bref = _bind_bref_slots(
            bref_override.StreamIndexOBufferedRef.output(
                spec=pl.BlockSpec((4, 2, 128), lambda i: (0, 0, 0)),
                dtype_or_type=jax.ShapeDtypeStruct((4, 2, 128), jnp.float32),
                buffer_count=2,
                use_lookahead=False,
                bq_sz=4,
                num_sublanes=2,
                seq_batch_size=1,
                chunk_idx=0,
            )
        )
        o_bref.copy_out(dst_refs, (0,))
        o_bref.wait_out(dst_refs, (0,))
        dataclasses.replace(o_bref, _buffer_count=0).wait_out(dst_refs, (0,))

    assert copy_events.count("start") == 6
    assert copy_events.count("wait") == 4


class _SliceAtProxy:
    def __init__(self, ref: _SliceArrayRef):
        self.ref = ref

    def __getitem__(self, idx):
        assert isinstance(idx, pl.Slice)
        return _SliceArrayRef(self.ref.arr, self.ref.base_offset + int(idx.start))


class _SliceArrayRef:
    """Mutable array wrapper supporting Pallas pl.ds slice indexing and .at."""

    def __init__(self, arr: np.ndarray, base_offset: int = 0):
        self.arr = arr
        self.base_offset = base_offset

    def _to_slice(self, idx) -> slice:
        if isinstance(idx, pl.Slice):
            start = int(idx.start) + self.base_offset
            size = int(idx.size)
            return slice(start, start + size)
        return idx

    @property
    def at(self) -> _SliceAtProxy:
        return _SliceAtProxy(self)

    def __getitem__(self, idx):
        s = self._to_slice(idx)
        return jnp.asarray(self.arr[s])

    def __setitem__(self, idx, val):
        s = self._to_slice(idx)
        self.arr[s] = np.asarray(val)


def test_dcp_sc_compact_body_and_wrappers():
    # Validation error branches
    with pytest.raises(ValueError, match="vals must be int32"):
        dcp_sc_compact.pack_nonnegative(jnp.zeros((2, 8), dtype=jnp.float32))

    with (
        patch.object(
            dcp_sc_compact.pltpu,
            "get_tpu_info",
            return_value=SimpleNamespace(sparse_core=None),
        ),
        pytest.raises(NotImplementedError, match="SparseCore is not available"),
    ):
        dcp_sc_compact.pack_nonnegative(jnp.zeros((2, 8), dtype=jnp.int32))

    fake_sc = SimpleNamespace(num_lanes=8, num_cores=1, num_subcores=2)
    fake_tpu = SimpleNamespace(sparse_core=fake_sc)

    with (
        patch.object(dcp_sc_compact.pltpu, "get_tpu_info", return_value=fake_tpu),
        pytest.raises(ValueError, match="must be a multiple of"),
    ):
        dcp_sc_compact.pack_nonnegative(jnp.zeros((2, 10), dtype=jnp.int32))

    def _fake_parallel_loop(start, stop, unroll=1, carry=None):
        del unroll
        start_i = int(start)
        stop_i = int(stop)

        def _runner(body_fn):
            if carry is None:
                for i in range(start_i, stop_i):
                    body_fn(i)
                return None
            c = carry
            for i in range(start_i, stop_i):
                c = body_fn(i, c)
            return c

        return _runner

    def _fake_store_compressed(dst_ref: _SliceArrayRef, v: jax.Array, mask: jax.Array):
        v_np = np.asarray(v)
        m_np = np.asarray(mask, dtype=bool)
        kept = v_np[m_np]
        start = dst_ref.base_offset
        dst_ref.arr[start : start + len(kept)] = kept

    def _fake_popcount(mask: jax.Array) -> jax.Array:
        cnt = int(np.sum(np.asarray(mask, dtype=bool)))
        return jnp.full((mask.shape[0],), cnt, dtype=jnp.int32)

    def _fake_emit_pipeline(step_fn, *, grid, in_specs, out_specs):
        num_waves = grid[0]
        width = in_specs[0].block_shape[0]

        def _run(vals_hbm: _SliceArrayRef, out_hbm: _SliceArrayRef):
            # Invoke index_map lambda on in_specs[0] and out_specs for 100% coverage
            idx = in_specs[0].index_map(0)[0]
            _ = out_specs.index_map(0)[0]
            for w in range(num_waves):
                row_idx = int(w * 2 + idx)
                vals_row = _SliceArrayRef(
                    vals_hbm.arr[row_idx * width : (row_idx + 1) * width].copy()
                )
                out_row = _SliceArrayRef(np.zeros((width,), dtype=np.int32))
                with patch.object(dcp_sc_compact.pl, "program_id", return_value=w):
                    step_fn(vals_row, out_row)
                out_hbm.arr[row_idx * width : (row_idx + 1) * width] = out_row.arr

        return _run

    def _fake_pl_kernel(
        body_fn, *, out_type, compiler_params, scratch_types, mesh, name
    ):
        del compiler_params, mesh, name
        scratch_shape = scratch_types[0].shape

        def _call(flat_vals: jax.Array) -> jax.Array:
            vals_np = np.asarray(flat_vals, dtype=np.int32)
            out_np = np.zeros(out_type.shape, dtype=np.int32)
            # Simulate running across both subcores (0 and 1)
            for subcore_id in range(fake_sc.num_subcores):
                vals_ref = _SliceArrayRef(vals_np)
                out_ref = _SliceArrayRef(out_np)
                rowbuf_ref = _SliceArrayRef(np.zeros(scratch_shape, dtype=np.int32))
                with patch.object(
                    dcp_sc_compact.lax,
                    "axis_index",
                    side_effect=lambda axis, s=subcore_id: 0 if axis == "core" else s,
                ):
                    body_fn(vals_ref, out_ref, rowbuf_ref)
            return jnp.asarray(out_np)

        return _call

    with (
        patch.object(dcp_sc_compact.pltpu, "get_tpu_info", return_value=fake_tpu),
        patch.object(
            dcp_sc_compact.plsc, "VectorSubcoreMesh", return_value=SimpleNamespace()
        ),
        patch.object(
            dcp_sc_compact.plsc, "parallel_loop", side_effect=_fake_parallel_loop
        ),
        patch.object(
            dcp_sc_compact.plsc,
            "store_compressed",
            side_effect=_fake_store_compressed,
        ),
        patch.object(
            dcp_sc_compact.plsc,
            "all_reduce_population_count",
            side_effect=_fake_popcount,
        ),
        patch.object(
            dcp_sc_compact.pltpu, "emit_pipeline", side_effect=_fake_emit_pipeline
        ),
        patch.object(dcp_sc_compact.pl, "kernel", side_effect=_fake_pl_kernel),
    ):
        # Case 1: num_waves == 1 and padded_b == b (b = 2, rows_per_wave = 2)
        vals_exact = jnp.array(
            [
                [-1, 5, -1, 2, 9, -1, 0, -1],
                [3, -1, -1, -1, 7, 1, -1, -1],
            ],
            dtype=jnp.int32,
        )
        packed_exact = dcp_sc_compact.pack_nonnegative(vals_exact)
        np.testing.assert_array_equal(
            np.asarray(packed_exact),
            np.array(
                [
                    [5, 2, 9, 0, -1, -1, -1, -1],
                    [3, 7, 1, -1, -1, -1, -1, -1],
                ],
                dtype=np.int32,
            ),
        )

        # Case 2: num_waves > 1 and padded_b > b (b = 3 -> padded_b = 4)
        vals_padded = jnp.array(
            [
                [-1, 4, -1, 8, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ],
            dtype=jnp.int32,
        )
        packed_padded = dcp_sc_compact.pack_nonnegative(vals_padded)
        assert packed_padded.shape == (3, 8)
        np.testing.assert_array_equal(
            np.asarray(packed_padded[0]),
            np.array([4, 8, -1, -1, -1, -1, -1, -1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(packed_padded[1]),
            np.full((8,), -1, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(packed_padded[2]),
            np.arange(1, 9, dtype=np.int32),
        )

        # Case 3: resolve_owned_winners
        local_idxs = jnp.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17],
                [20, 21, 22, 23, 24, 25, 26, 27],
            ],
            dtype=jnp.int32,
        )
        # k = 8, dcp_rank = 1 owns slots in [8, 16)
        slots = jnp.array(
            [
                [0, 9, -1, 12, 3, 15, -1, 8],
                [11, 2, 10, -1, -1, -1, -1, -1],
            ],
            dtype=jnp.int32,
        )
        winners = dcp_sc_compact.resolve_owned_winners(
            local_idxs, slots, k=8, dcp_rank=1
        )
        np.testing.assert_array_equal(
            np.asarray(winners),
            np.array(
                [
                    [11, 14, 17, 10, -1, -1, -1, -1],
                    [23, 22, -1, -1, -1, -1, -1, -1],
                ],
                dtype=np.int32,
            ),
        )
