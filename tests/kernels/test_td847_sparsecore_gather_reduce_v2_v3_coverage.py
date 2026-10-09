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
"""CPU unit tests for TD8.47 SparseCore ragged gather reduce v2/v3 & gather v2."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.sparse_core import (
    ragged_gather_reduce_v3 as rgr_v3,
)
from vllm_torchtpu.kernels.sparse_core import (
    ragged_gather_v2 as rg_v2,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
    config as rgr_v2_config,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
    kernel as rgr_v2_kernel,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
    memory_ref as rgr_v2_mem,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
    wrapper as rgr_v2_wrapper,
)


def _make_tpu_info(
    *,
    generation: int = 7,
    num_lanes: int = 128,
    vmem_capacity_bytes: int = 64 * 1024 * 1024,
    sc_num_cores: int = 2,
    sc_num_subcores: int = 2,
    sc_num_lanes: int = 16,
    sc_vmem_capacity_bytes: int = 256 * 1024,
    has_sc: bool = True,
) -> Any:
    sc = (
        SimpleNamespace(
            num_cores=sc_num_cores,
            num_subcores=sc_num_subcores,
            num_lanes=sc_num_lanes,
            vmem_capacity_bytes=sc_vmem_capacity_bytes,
        )
        if has_sc
        else None
    )
    return SimpleNamespace(
        generation=generation,
        num_lanes=num_lanes,
        vmem_capacity_bytes=vmem_capacity_bytes,
        sparse_core=sc,
    )


class _FakeAsyncCopy:
    def __init__(self, src: Any = None, dst: Any = None, sem: Any = None) -> None:
        self.src = src
        self.dst = dst
        self.sem = sem
        self.started = False
        self.waited = False

    def start(self) -> None:
        self.started = True

    def wait(self) -> None:
        self.waited = True


class _FakeRef:
    def __init__(self, arr: jnp.ndarray) -> None:
        self.arr = arr
        self.at = self

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self) -> Any:
        return self.arr.dtype

    def bitcast(self, dtype: Any) -> _FakeRef:
        return _FakeRef(self.arr.astype(dtype))

    def __getitem__(self, idx: Any) -> Any:
        if idx is Ellipsis:
            return self.arr
        return _FakeRef(jnp.zeros((16, 128), dtype=self.arr.dtype))

    def __setitem__(self, idx: Any, val: Any) -> None:
        return None


# =====================================================================
# 1. ragged_gather_reduce_v2/config.py
# =====================================================================
class TestRaggedGatherReduceV2Config:
    def test_config_properties_and_cached_property_speedup(self) -> None:
        tpu_info = _make_tpu_info(
            generation=6,
            sc_num_cores=2,
            sc_num_subcores=2,
            sc_num_lanes=16,
        )
        cfg = rgr_v2_config.Config(
            input_size=256,
            hidden_size=2048,
            source_rows=4096,
            reduce_group_size=4,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=tpu_info,
        )
        assert cfg.num_tot_cores == 4
        assert cfg.output_size == 64
        assert cfg.in_dtype_bytes == 2
        assert cfg.row_shift == 1
        assert cfg.num_column_partitions == 4
        assert cfg.num_row_partitions == 1
        assert cfg.num_row_subchunks >= 1
        assert cfg.row_chunk_size == 16 * cfg.num_row_subchunks
        assert cfg.padded_input_size >= 256
        assert cfg.row_partition_size == cfg.padded_input_size
        assert cfg.max_blocks_per_partition >= 1
        assert cfg.max_window >= 1
        assert cfg.window_size == cfg.max_window * cfg.row_chunk_size
        assert cfg.row_partition_size_padded >= cfg.row_partition_size
        assert cfg.aligned_hidden_size == 2048
        assert cfg.col_size == 512
        assert cfg.col_chunk_size == 512
        assert cfg.num_col_chunks == 1

    def test_should_fallback_branches(self) -> None:
        cfg_no_sc = rgr_v2_config.Config(
            input_size=64,
            hidden_size=512,
            source_rows=1024,
            reduce_group_size=2,
            in_dtype=jnp.float32,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=_make_tpu_info(has_sc=False),
        )
        assert cfg_no_sc.should_fallback is True
        assert cfg_no_sc.row_shift == 0

        cfg_small = rgr_v2_config.Config(
            input_size=64,
            hidden_size=512,
            source_rows=16,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=_make_tpu_info(vmem_capacity_bytes=64 * 1024 * 1024),
        )
        assert cfg_small.should_fallback is True

        cfg_large = rgr_v2_config.Config(
            input_size=64,
            hidden_size=512,
            source_rows=1024,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=_make_tpu_info(vmem_capacity_bytes=1024),
        )
        assert cfg_large.should_fallback is False

    def test_num_column_partitions_continue_and_break_branches(self) -> None:
        # 1) Trigger `num_tot_cores // num_column_partitions > num_simd_lanes`
        tpu_many_cores = _make_tpu_info(
            generation=7,
            sc_num_cores=4,
            sc_num_subcores=8,  # num_tot_cores = 32 > 16 = num_simd_lanes
            sc_num_lanes=16,
        )
        cfg_continue = rgr_v2_config.Config(
            input_size=256,
            hidden_size=4096,
            source_rows=1024,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=tpu_many_cores,
        )
        assert cfg_continue.num_column_partitions >= 2

        # 2) Trigger `if num_iterations > _CostModelConstants.MAX_ITERATIONS: break`
        cfg_break = rgr_v2_config.Config(
            input_size=100_000,
            hidden_size=4096,
            source_rows=1024,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=_make_tpu_info(generation=7, sc_num_cores=2, sc_num_subcores=2),
        )
        assert cfg_break.num_column_partitions == 1

    def test_col_chunk_size_generations_and_divisor_search(self) -> None:
        # Generation 7 with col_size = 1152 (> 1024 max_safe_col) so first candidate
        # 1024 does not divide 1152, iterating down to 384.
        cfg_g7 = rgr_v2_config.Config(
            input_size=64,
            hidden_size=1152,
            source_rows=1024,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=_make_tpu_info(generation=7, sc_num_cores=1, sc_num_subcores=1),
        )
        assert cfg_g7.col_chunk_size == 384

        # Generation 5 (case _:) with hidden_size=0 -> returns num_lanes.
        cfg_g5 = rgr_v2_config.Config(
            input_size=64,
            hidden_size=0,
            source_rows=1024,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=_make_tpu_info(generation=5, sc_num_cores=1, sc_num_subcores=1),
        )
        assert cfg_g5.col_chunk_size == 128


# =====================================================================
# 2. ragged_gather_reduce_v2/memory_ref.py, kernel.py, wrapper.py
# =====================================================================
class TestRaggedGatherReduceV2KernelAndWrapper:
    def test_memory_ref_and_wrapper_fallback_and_non_fallback(self) -> None:
        cfg = rgr_v2_config.Config(
            input_size=64,
            hidden_size=512,
            source_rows=1024,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=_make_tpu_info(generation=7, vmem_capacity_bytes=1024),
        )
        scratch_types = rgr_v2_mem.ScratchRef.create_scratch_types(cfg)
        assert scratch_types.prev_iter_last_row_vmem.shape == (
            cfg.num_col_chunks,
            cfg.col_chunk_size,
        )

        x = jnp.ones((8, 128), dtype=jnp.bfloat16)
        indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        weights = jnp.array([0.5, 0.5, 1.0, 0.0], dtype=jnp.bfloat16)
        valid_mask = jnp.array([True, True, True, False], dtype=jnp.bool_)

        # Fallback path (small input on CPU)
        with patch.object(
            rgr_v2_wrapper.pltpu,
            "get_tpu_info",
            return_value=_make_tpu_info(has_sc=False),
        ):
            out_fb = rgr_v2_wrapper.ragged_gather_reduce_v2.__wrapped__(
                x, indices, weights, valid_mask, 2
            )
        assert out_fb.shape == (2, 128)

        # Non-fallback path with padding == 0 and padding != 0
        # Case A: input_size already aligned (`cfg.padded_input_size == cfg.input_size`)
        tpu_info_sc = _make_tpu_info(
            generation=7,
            vmem_capacity_bytes=16,
            sc_num_cores=1,
            sc_num_subcores=2,
            sc_num_lanes=4,
            sc_vmem_capacity_bytes=256 * 1024,
        )
        with (
            patch.object(
                rgr_v2_wrapper.pltpu, "get_tpu_info", return_value=tpu_info_sc
            ),
            patch.object(
                rgr_v2_wrapper.plsc,
                "VectorSubcoreMesh",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
            patch.object(
                rgr_v2_wrapper.core_map_helper,
                "kernel",
                side_effect=lambda fn, out_type, **kw: (
                    lambda *a: jnp.ones(out_type.shape, dtype=out_type.dtype)
                ),
            ),
        ):
            out_aligned = rgr_v2_wrapper.ragged_gather_reduce_v2.__wrapped__(
                x, indices, weights, valid_mask, 2
            )
            assert out_aligned.shape == (2, 128)

            # Case B: unaligned input_size (`cfg.padded_input_size != cfg.input_size`)
            # and `row_partition_size_padded != row_partition_size`
            indices_6 = jnp.arange(6, dtype=jnp.int32)
            weights_6 = jnp.ones((6,), dtype=jnp.bfloat16)
            valid_6 = jnp.ones((6,), dtype=jnp.bool_)
            tpu_info_pad = _make_tpu_info(
                generation=7,
                vmem_capacity_bytes=16,
                sc_num_cores=2,
                sc_num_subcores=2,
                sc_num_lanes=4,
                sc_vmem_capacity_bytes=256 * 1024,
            )
            with patch.object(
                rgr_v2_wrapper.pltpu, "get_tpu_info", return_value=tpu_info_pad
            ):
                out_unaligned = rgr_v2_wrapper.ragged_gather_reduce_v2.__wrapped__(
                    x, indices_6, weights_6, valid_6, 2
                )
                assert out_unaligned.shape == (3, 128)

        # Also test _preprocess_scalar_data when padding == 0 and when padding > 0
        cfg_pad = rgr_v2_config.Config(
            input_size=12,
            hidden_size=128,
            source_rows=16,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=_make_tpu_info(
                sc_num_cores=1,
                sc_num_subcores=1,
                sc_num_lanes=8,
                sc_vmem_capacity_bytes=256 * 1024,
            ),
        )
        # First with padding == 0 (row_partition_size_padded == row_partition_size)
        object.__setattr__(
            cfg_pad, "row_partition_size_padded", cfg_pad.row_partition_size
        )
        scalar_no_pad, _ = rgr_v2_wrapper._preprocess_scalar_data(
            jnp.arange(cfg_pad.padded_input_size, dtype=jnp.int32),
            jnp.ones((cfg_pad.padded_input_size,), dtype=jnp.bool_),
            cfg_pad,
        )
        assert scalar_no_pad.sorted_by_validity.shape[0] == cfg_pad.row_partition_size

        # Then with padding > 0 (row_partition_size_padded > row_partition_size)
        object.__setattr__(cfg_pad, "row_partition_size_padded", 32)
        scalar, mask = rgr_v2_wrapper._preprocess_scalar_data(
            jnp.arange(cfg_pad.padded_input_size, dtype=jnp.int32),
            jnp.ones((cfg_pad.padded_input_size,), dtype=jnp.bool_),
            cfg_pad,
        )
        assert scalar.sorted_by_validity.shape[0] == 32
        assert mask.shape[0] == cfg_pad.padded_input_size // 2

    def test_main_kernel_and_kernel_pipeline_execution(self) -> None:
        tpu_info = _make_tpu_info(
            generation=7,
            num_lanes=8,
            vmem_capacity_bytes=16,
            sc_num_cores=1,
            sc_num_subcores=1,
            sc_num_lanes=4,
            sc_vmem_capacity_bytes=256 * 1024,
        )
        # input_size=8 with num_lanes=4, num_row_partitions=1 -> num_row_subchunks=2
        cfg_bf16 = rgr_v2_config.Config(
            input_size=8,
            hidden_size=8,
            source_rows=8,
            reduce_group_size=2,
            in_dtype=jnp.bfloat16,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=tpu_info,
        )
        assert cfg_bf16.num_row_subchunks == 2

        class _ArrRef:
            def __init__(self, arr: jnp.ndarray) -> None:
                self.arr = np.array(arr, copy=True)
                self.at = self

            @property
            def shape(self) -> tuple[int, ...]:
                return self.arr.shape

            @property
            def dtype(self) -> Any:
                return self.arr.dtype

            def bitcast(self, dt: Any) -> _ArrRef:
                return _ArrRef(jnp.asarray(self.arr).astype(dt))

            def _norm_idx(self, idx: Any) -> Any:
                if idx is Ellipsis:
                    return Ellipsis
                if isinstance(idx, tuple):
                    return tuple(self._norm_idx(i) for i in idx)
                if hasattr(idx, "start") and hasattr(idx, "size"):
                    s = int(np.asarray(idx.start))
                    return slice(s, s + int(idx.size))
                if isinstance(idx, jax.Array):
                    return int(np.asarray(idx))
                return idx

            def __getitem__(self, idx: Any) -> jnp.ndarray:
                return jnp.asarray(self.arr[self._norm_idx(idx)])

            def __setitem__(self, idx: Any, val: Any) -> None:
                self.arr[self._norm_idx(idx)] = np.asarray(val)

        def _make_refs(cfg: rgr_v2_config.Config) -> rgr_v2_mem.KernelRefs:
            scalar_ref = rgr_v2_mem.IndexRef(
                num_src_rows_per_row_partition=_ArrRef(
                    jnp.zeros((4,), dtype=jnp.int32)
                ),
                indices=_ArrRef(jnp.arange(8, dtype=jnp.int32)),
                sorted_by_validity=_ArrRef(
                    jnp.arange(cfg.row_partition_size_padded, dtype=jnp.int32)
                ),
            )
            scratch_ref = rgr_v2_mem.ScratchRef(
                num_rows_per_row_partition_vmem=_ArrRef(
                    jnp.array([8, 0, 0, 0], dtype=jnp.int32)
                ),
                next_window_first_row_vmem=_ArrRef(jnp.zeros((4,), dtype=jnp.int32)),
                prev_iter_last_row_vmem=_ArrRef(
                    jnp.zeros(
                        (cfg.num_col_chunks, cfg.col_chunk_size), dtype=jnp.float32
                    )
                ),
                prev_dst_row_smem=_ArrRef(jnp.zeros((1,), dtype=jnp.int32)),
                sorted_by_validity_vmem=_ArrRef(
                    jnp.arange(cfg.window_size, dtype=jnp.int32)
                ),
                src_indices_vmem=_ArrRef(
                    jnp.arange(cfg.row_chunk_size, dtype=jnp.int32)
                ),
                dst_indices_vmem=_ArrRef(
                    jnp.zeros((cfg.row_chunk_size,), dtype=jnp.int32)
                ),
                tw_f32_vmem=_ArrRef(jnp.ones((cfg.row_chunk_size,), dtype=jnp.float32)),
                dma_src_row_vmem=_ArrRef(
                    jnp.zeros((cfg.row_chunk_size,), dtype=jnp.int32)
                ),
                dma_dst_row_vmem=_ArrRef(
                    jnp.zeros((cfg.row_chunk_size,), dtype=jnp.int32)
                ),
                prev_dst_val_vmem=_ArrRef(
                    jnp.zeros((cfg.row_chunk_size,), dtype=jnp.int32)
                ),
                out_vmem=_ArrRef(jnp.zeros((4, cfg.col_chunk_size), dtype=jnp.float32)),
                sem=_ArrRef(jnp.zeros((2,), dtype=jnp.int32)),
            )
            return rgr_v2_mem.KernelRefs.create(
                scalar_ref=scalar_ref,
                in_hbm_ref=_ArrRef(jnp.ones((8, 8), dtype=cfg.in_dtype)),
                topk_weights_hbm_ref=_ArrRef(jnp.ones((8,), dtype=jnp.float32)),
                out_hbm_ref=_ArrRef(jnp.zeros((5, 8), dtype=jnp.float32)),
                scratch_ref=scratch_ref,
            )

        def _fake_emit_pipeline(
            body_fn: Any, *, grid: Any, in_specs: Any = None, **kw: Any
        ) -> Any:
            # Exercise BlockSpec index_maps
            if isinstance(in_specs, tuple):
                for spec_group in in_specs:
                    if isinstance(spec_group, tuple):
                        for spec in spec_group:
                            if spec.index_map is not None:
                                spec.index_map(0)
            elif in_specs is not None and in_specs.index_map is not None:
                in_specs.index_map(0, 0)

            def _runner(*args: Any) -> None:
                if len(args) == 2:
                    # _row_kernel
                    idx_refs = tuple(
                        _ArrRef(jnp.arange(4, dtype=jnp.int32)) for _ in args[0]
                    )
                    tw_refs = tuple(
                        _ArrRef(jnp.ones((4,), dtype=jnp.float32)) for _ in args[1]
                    )
                    body_fn(idx_refs, tw_refs)
                else:
                    # _col_kernel
                    body_fn(_ArrRef(jnp.ones((4, 8), dtype=jnp.uint32)))

            return _runner

        refs_bf16 = _make_refs(cfg_bf16)
        with (
            patch.object(
                rgr_v2_wrapper.jax.lax, "axis_index", return_value=jnp.int32(0)
            ),
            patch.object(
                rgr_v2_wrapper.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: _FakeAsyncCopy(*a, **k),
            ),
            patch.object(
                rgr_v2_kernel.pl,
                "loop",
                side_effect=lambda start, end: (lambda fn: fn(jnp.int32(0))),
            ),
            patch.object(rgr_v2_kernel.pl, "program_id", return_value=jnp.int32(0)),
            patch.object(
                rgr_v2_kernel.pltpu, "emit_pipeline", side_effect=_fake_emit_pipeline
            ),
            patch.object(
                rgr_v2_kernel.plsc,
                "parallel_loop",
                side_effect=lambda start, stop, step: (lambda fn: fn(0)),
            ),
            patch.object(
                rgr_v2_kernel.plsc,
                "bitcast",
                side_effect=lambda x, dt: jnp.asarray(x).astype(dt),
            ),
        ):
            rgr_v2_wrapper.main_kernel(
                refs_bf16.index,
                refs_bf16.data.source,
                refs_bf16.data.topk_weights,
                refs_bf16.data.out,
                refs_bf16.scratch,
                cfg=cfg_bf16,
            )

            # Also run _col_kernel with float32 config to cover `if is_bf16:` False
            cfg_f32 = rgr_v2_config.Config(
                input_size=8,
                hidden_size=8,
                source_rows=8,
                reduce_group_size=2,
                in_dtype=jnp.float32,
                core_axis_name="core",
                subcore_axis_name="subcore",
                tpu_info=tpu_info,
            )
            refs_f32 = _make_refs(cfg_f32)
            rgr_v2_kernel._col_kernel(
                _ArrRef(jnp.ones((4, 8), dtype=jnp.uint32)),
                refs=refs_f32,
                col_start=jnp.int32(0),
                cfg=cfg_f32,
            )


# =====================================================================
# 3. ragged_gather_reduce_v3.py
# =====================================================================
class TestRaggedGatherReduceV3:
    def test_partition_counts_and_alignment(self) -> None:
        rgr_v3._calculate_num_column_partitions.cache_clear()
        tpu_info = _make_tpu_info(sc_num_cores=2, sc_num_subcores=2, num_lanes=128)
        # hidden_size=1024 -> num_column_partitions reaches 2 in while loop, then
        # `hidden_size <= 1024 * (2 // 2)` halves it to 1!
        col_p, row_p = rgr_v3.partition_counts(1024, tpu_info)
        assert (col_p, row_p) == (1, 4)
        assert rgr_v3.token_block_alignment(1024, tpu_info) == 4 * 64

        # Wider hidden_size=4096 -> stays 4
        col_p_wide, row_p_wide = rgr_v3.partition_counts(4096, tpu_info)
        assert (col_p_wide, row_p_wide) == (4, 1)

    def test_ragged_gather_reduce_v3_fallback_and_validation_and_launch(self) -> None:
        x = jnp.ones((8, 128), dtype=jnp.bfloat16)
        indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        weights = jnp.array([0.5, 0.5, 1.0, 0.0], dtype=jnp.bfloat16)
        valid_mask = jnp.array([True, True, True, False], dtype=jnp.bool_)

        # 1) Fallback when sc_info is None
        with patch.object(
            rgr_v3.pltpu, "get_tpu_info", return_value=_make_tpu_info(has_sc=False)
        ):
            out1 = rgr_v3.ragged_gather_reduce.__wrapped__(
                x, indices, weights, valid_mask, 2
            )
            assert out1.shape == (2, 128)

        # 2) Fallback when vmem capacity is large relative to x
        with patch.object(
            rgr_v3.pltpu,
            "get_tpu_info",
            return_value=_make_tpu_info(vmem_capacity_bytes=64 * 1024 * 1024),
        ):
            out2 = rgr_v3.ragged_gather_reduce.__wrapped__(
                x, indices, weights, valid_mask, 2
            )
            assert out2.shape == (2, 128)

        # 3) Validation errors on non-fallback path
        small_vmem_tpu = _make_tpu_info(
            vmem_capacity_bytes=16,
            sc_num_cores=1,
            sc_num_subcores=1,
            num_lanes=128,
        )
        with (
            patch.object(rgr_v3.pltpu, "get_tpu_info", return_value=small_vmem_tpu),
            patch.object(
                rgr_v3.plsc,
                "VectorSubcoreMesh",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
        ):
            with pytest.raises(ValueError, match="must be divisible by"):
                rgr_v3.ragged_gather_reduce.__wrapped__(
                    x,
                    jnp.arange(3, dtype=jnp.int32),
                    jnp.ones((3,), dtype=jnp.bfloat16),
                    jnp.ones((3,), dtype=jnp.bool_),
                    2,
                )

            class _HugeRowsArray:
                dtype = jnp.bfloat16
                shape = ((1 << 20) + 2, 128)
                size = 1024

            with (
                patch.object(rgr_v3.jnp, "size", return_value=1024),
                pytest.raises(ValueError, match="supports at most"),
            ):
                rgr_v3.ragged_gather_reduce.__wrapped__(
                    _HugeRowsArray(),  # type: ignore[arg-type]
                    indices,
                    weights,
                    valid_mask,
                    2,
                )

            # 4) Odd source rows (`x.shape[0] % 2 == 1`) + `token_padding > 0`
            x_odd = jnp.ones((7, 512), dtype=jnp.bfloat16)
            with patch.object(
                rgr_v3.core_map_helper,
                "kernel",
                side_effect=lambda fn, out_type, **kw: (
                    lambda *a: jnp.ones(out_type.shape, dtype=out_type.dtype)
                ),
            ):
                out_padded = rgr_v3.ragged_gather_reduce.__wrapped__(
                    x_odd, indices, weights, valid_mask, 2
                )
                assert out_padded.shape == (2, 512)

                # 5) Even source rows (`x.shape[0] % 2 == 0`) + `token_padding == 0`
                indices_aligned = jnp.zeros((128,), dtype=jnp.int32)
                weights_aligned = jnp.ones((128,), dtype=jnp.bfloat16)
                valid_aligned = jnp.ones((128,), dtype=jnp.bool_)
                x_even = jnp.ones((8, 512), dtype=jnp.bfloat16)
                out_exact = rgr_v3.ragged_gather_reduce.__wrapped__(
                    x_even, indices_aligned, weights_aligned, valid_aligned, 2
                )
                assert out_exact.shape == (64, 512)

    def test_v3_main_kernel_execution(self) -> None:
        tpu_info = _make_tpu_info(
            sc_num_cores=1, sc_num_subcores=1, sc_num_lanes=16, num_lanes=128
        )

        class _V3Ref:
            def __init__(
                self, shape: tuple[int, ...], dtype: Any = jnp.float32
            ) -> None:
                self.shape = shape
                self.dtype = dtype
                self.at = self

            def bitcast(self, dt: Any) -> _V3Ref:
                return _V3Ref(self.shape, dt)

            def __getitem__(self, idx: Any) -> jnp.ndarray:
                if idx is Ellipsis:
                    return jnp.array([128], dtype=jnp.int32)
                return jnp.ones((16,), dtype=self.dtype)

            def __setitem__(self, idx: Any, val: Any) -> None:
                return None

        def _fake_emit_pipeline(
            body_fn: Any, *, grid: Any, in_specs: Any, out_specs: Any, **kw: Any
        ) -> Any:
            if isinstance(in_specs, tuple):
                for spec in in_specs:
                    spec.index_map(0)
            else:
                in_specs.index_map(0)

            def _runner(*r_args: Any, **r_kw: Any) -> None:
                if "scratches" in r_kw:
                    body_fn(
                        _V3Ref((128,), jnp.int32),
                        _V3Ref((128,), jnp.float32),
                        _V3Ref((16,), jnp.int32),
                        _V3Ref((2,), jnp.int32),
                    )
                else:
                    body_fn(_V3Ref((16, 16), jnp.int32))

            return _runner

        with (
            patch.object(rgr_v3.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(rgr_v3.jax.lax, "axis_index", return_value=jnp.int32(0)),
            patch.object(rgr_v3.pl, "program_id", return_value=jnp.int32(0)),
            patch.object(rgr_v3.pl, "multiple_of", side_effect=lambda x, m: x),
            patch.object(rgr_v3.pl, "when", side_effect=lambda cond: (lambda fn: fn())),
            patch.object(
                rgr_v3.pltpu, "emit_pipeline", side_effect=_fake_emit_pipeline
            ),
            patch.object(
                rgr_v3.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: _FakeAsyncCopy(*a, **k),
            ),
            patch.object(
                rgr_v3.plsc,
                "parallel_loop",
                side_effect=lambda start, stop, step: (lambda fn: fn(0)),
            ),
            patch.object(
                rgr_v3.plsc,
                "bitcast",
                side_effect=lambda x, dt: jnp.asarray(x).astype(dt),
            ),
            patch.object(rgr_v3.plsc, "addupdate_scatter", return_value=None),
            patch.object(
                rgr_v3.plsc,
                "pack",
                side_effect=lambda a, b, format: jnp.zeros((16,), dtype=jnp.uint32),
            ),
        ):
            rgr_v3._main_kernel(
                _V3Ref((8, 16), jnp.bfloat16),
                _V3Ref((128,), jnp.int32),
                _V3Ref((128,), jnp.float32),
                _V3Ref((16,), jnp.int32),
                _V3Ref((64, 16), jnp.bfloat16),
                _V3Ref((64, 16), jnp.float32),
                _V3Ref((2, 8, 16), jnp.uint32),
                _V3Ref((2,), jnp.int32),
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_partitions=1,
                num_column_partitions=1,
                blocks_per_row_partition=1,
                topk=2,
            )


# =====================================================================
# 4. ragged_gather_v2.py
# =====================================================================
class TestRaggedGatherV2:
    def test_calculate_col_size_generations_and_branches(self) -> None:
        # Generation 6
        with patch.object(
            rg_v2.pltpu, "get_tpu_info", return_value=_make_tpu_info(generation=6)
        ):
            assert rg_v2.calculate_col_size(1024, 2) == 1024

        # Generation 7 with non-immediate divisor (hidden_size=2816, max_safe_col=2176)
        with patch.object(
            rg_v2.pltpu, "get_tpu_info", return_value=_make_tpu_info(generation=7)
        ):
            assert rg_v2.calculate_col_size(2816, 2) == 1408

        # Generation 5 (case _:) with hidden_size=64 (< 128) -> returns max_safe_col
        with patch.object(
            rg_v2.pltpu, "get_tpu_info", return_value=_make_tpu_info(generation=5)
        ):
            assert rg_v2.calculate_col_size(64, 2) == 512

    def test_ragged_gather_v2_wrapper_branches(self) -> None:
        x = jnp.ones((16, 128), dtype=jnp.bfloat16)
        indices = jnp.arange(8, dtype=jnp.int32)

        # Invalid dtype raises ValueError
        with pytest.raises(ValueError, match="dtype must be"):
            rg_v2.ragged_gather_v2.__wrapped__(
                jnp.ones((4, 4), dtype=jnp.int32),
                indices,
                jnp.int32(0),
                jnp.int32(4),
            )

        # Fallback when sc_info is None (with scalar start/end)
        with patch.object(
            rg_v2.pltpu, "get_tpu_info", return_value=_make_tpu_info(has_sc=False)
        ):
            out_fb = rg_v2.ragged_gather_v2.__wrapped__(
                x, indices, jnp.int32(0), jnp.int32(4)
            )
            assert out_fb.shape == (8, 128)

        # SparseCore path with 1D start/end and out_pad_size > 0
        tpu_sc = _make_tpu_info(sc_num_cores=1, sc_num_subcores=1, sc_num_lanes=16)
        with (
            patch.object(rg_v2.pltpu, "get_tpu_info", return_value=tpu_sc),
            patch.object(
                rg_v2.plsc,
                "VectorSubcoreMesh",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
            patch.object(
                rg_v2.core_map_helper,
                "kernel",
                side_effect=lambda fn, out_type, **kw: (
                    lambda *a: jnp.ones(out_type.shape, dtype=out_type.dtype)
                ),
            ),
        ):
            out_padded = rg_v2.ragged_gather_v2.__wrapped__(
                x, indices, jnp.array([0]), jnp.array([8])
            )
            assert out_padded.shape == (8, 128)

            # SparseCore path with out_pad_size == 0 (indices.size == 16 == block_size)
            indices_16 = jnp.arange(16, dtype=jnp.int32)
            out_exact = rg_v2.ragged_gather_v2.__wrapped__(
                x, indices_16, jnp.array([0]), jnp.array([16])
            )
            assert out_exact.shape == (16, 128)

    def test_main_kernel_v2_packing_1_and_packing_gt_1(self) -> None:
        tpu_sc = _make_tpu_info(
            generation=7, sc_num_cores=1, sc_num_subcores=1, sc_num_lanes=16
        )

        class _GatherRef:
            def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
                self.shape = shape
                self.dtype = dtype
                self.at = self

            def bitcast(self, dt: Any) -> _GatherRef:
                return _GatherRef(self.shape, dt)

            def __getitem__(self, idx: Any) -> jnp.ndarray:
                if idx is Ellipsis:
                    return jnp.array([0, 16], dtype=jnp.int32)
                if isinstance(idx, tuple):
                    return jnp.ones((1, 128), dtype=self.dtype)
                return jnp.arange(16, dtype=jnp.int32)

            def __setitem__(self, idx: Any, val: Any) -> None:
                return None

        def _fake_emit_pipeline(
            body_fn: Any, *, grid: Any, in_specs: Any, out_specs: Any = None
        ) -> Any:
            if out_specs is None:
                # outer_pipeline
                in_specs.index_map(0)

                def _run_outer(indices_ref: Any) -> None:
                    body_fn(_GatherRef((16,), jnp.int32))

                return _run_outer
            else:
                # inner_pipeline
                in_specs.index_map(0, 0)
                out_specs.index_map(0, 0)

                def _run_inner(in_ref: Any, out_ref: Any) -> None:
                    body_fn(
                        _GatherRef((16, 128), jnp.int32),
                        _GatherRef((16, 128), jnp.int32),
                    )

                return _run_inner

        with (
            patch.object(rg_v2.pltpu, "get_tpu_info", return_value=tpu_sc),
            patch.object(rg_v2.jax.lax, "axis_size", return_value=1),
            patch.object(rg_v2.lax, "axis_index", return_value=jnp.int32(0)),
            patch.object(rg_v2.pl, "program_id", return_value=jnp.int32(0)),
            patch.object(
                rg_v2.pltpu,
                "make_async_copy",
                side_effect=lambda *a, **k: _FakeAsyncCopy(*a, **k),
            ),
            patch.object(rg_v2.pltpu, "emit_pipeline", side_effect=_fake_emit_pipeline),
            patch.object(
                rg_v2.plsc,
                "parallel_loop",
                side_effect=lambda start, stop, step: (lambda fn: fn(0)),
            ),
        ):
            # 1) packing == 1 (float32)
            rg_v2.main_kernel_v2(
                _GatherRef((1,), jnp.int32),
                _GatherRef((1,), jnp.int32),
                _GatherRef((16, 128), jnp.float32),
                _GatherRef((16,), jnp.int32),
                _GatherRef((16, 128), jnp.float32),
                _GatherRef((16,), jnp.int32),
                _GatherRef((16,), jnp.int32),
                _GatherRef((1,), jnp.int32),
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_subchunks=1,
            )

            # 2) packing == 2 (bfloat16)
            rg_v2.main_kernel_v2(
                _GatherRef((1,), jnp.int32),
                _GatherRef((1,), jnp.int32),
                _GatherRef((16, 128), jnp.bfloat16),
                _GatherRef((16,), jnp.int32),
                _GatherRef((16, 128), jnp.bfloat16),
                _GatherRef((16,), jnp.int32),
                _GatherRef((16,), jnp.int32),
                _GatherRef((1,), jnp.int32),
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_subchunks=1,
            )
