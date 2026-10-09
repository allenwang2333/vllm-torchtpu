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

"""100% Statement and Branch Coverage test suite for GDN v3 PCP wrapper."""

import dataclasses
import functools
import unittest.mock as mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest  # pyrefly: ignore[missing-import]
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, pcp_metadata, pcp_wrapper


class _DummyCopy:
    """Mock async copy descriptor that implements start() and wait()."""

    def start(self):
        return self

    def wait(self):
        return self


class _MockRef:
    """Mock Pallas memory reference wrapping a JAX/NumPy array."""

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        pass

    @property
    def at(self):
        class _AtHelper:
            def __init__(self, ref):
                self.ref = ref

            def __getitem__(self, idx):
                return self.ref

        return _AtHelper(self)


class _DummyAlloc:
    """Mock Pallas allocation object."""

    def __init__(self):
        self.spec = pl.BlockSpec()


def _mock_create_allocs(*args, **kwargs):
    return (
        None,
        _DummyAlloc(),
        _DummyAlloc(),
        _DummyAlloc(),
        _DummyAlloc(),
        _DummyAlloc(),
    )


def _mock_with_scoped(allocations=None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, allocations=allocations, **kwargs)

        return wrapper

    return decorator


def _region(*, kb0: int, nblocks: int, row0: int, nrows: int) -> config.StateRegion:
    return config.StateRegion(
        kb0=kb0,
        nblocks=nblocks,
        row0=row0,
        nrows=nrows,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=nrows,
    )


def _make_stage_metadata(
    *,
    num_stages: int = 2,
    pcp_size: int = 2,
    comm_chunk_size: int = 32,
    num_tiles_per_stage: int = 2,
    rank_active_row_end: int = 128,
    catchup_count: int = 1,
) -> pcp_metadata.PcpStageMetadata:
    total_tiles = num_stages * num_tiles_per_stage
    return pcp_metadata.PcpStageMetadata(
        num_stages=jnp.array(num_stages, dtype=jnp.int32),
        rank_active_row_end=jnp.full((pcp_size,), rank_active_row_end, dtype=jnp.int32),
        request_id=jnp.zeros((pcp_size,), dtype=jnp.int32),
        query_row_start=jnp.zeros((pcp_size,), dtype=jnp.int32),
        num_tokens=jnp.full((num_stages,), pcp_size * comm_chunk_size, dtype=jnp.int32),
        rank_row_start=jnp.zeros((num_stages, pcp_size), dtype=jnp.int32),
        rank_valid_rows=jnp.full(
            (num_stages, pcp_size), comm_chunk_size, dtype=jnp.int32
        ),
        rank_recv_start=jnp.zeros((num_stages, pcp_size), dtype=jnp.int32),
        first_tile=jnp.arange(0, total_tiles, num_tiles_per_stage, dtype=jnp.int32),
        num_tiles=jnp.full((num_stages,), num_tiles_per_stage, dtype=jnp.int32),
        tile_stage=jnp.repeat(
            jnp.arange(num_stages, dtype=jnp.int32), num_tiles_per_stage
        ),
        tile_row_in_stage=jnp.tile(
            jnp.arange(num_tiles_per_stage, dtype=jnp.int32) * comm_chunk_size,
            num_stages,
        ),
        projection_catchup_start=jnp.zeros((num_stages, pcp_size), dtype=jnp.int32),
        projection_catchup_count=jnp.full(
            (num_stages, pcp_size), catchup_count, dtype=jnp.int32
        ),
        projection_work_offset=jnp.zeros((num_stages, pcp_size), dtype=jnp.int32),
        projection_work_offset_end=jnp.zeros((pcp_size,), dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Unit tests for ProjectionConfig and optimization caching
# ---------------------------------------------------------------------------


def test_projection_config_slots_and_fields():
    cfg = pcp_wrapper._ProjectionConfig(
        token_block_size=64,
        out_block_size=512,
        qkv_dim=512,
        full_key_dim=128,
        kq_replication_factor=2,
    )
    assert cfg.token_block_size == 64
    assert cfg.out_block_size == 512
    assert cfg.qkv_dim == 512
    assert cfg.full_key_dim == 128
    assert cfg.kq_replication_factor == 2
    assert hasattr(cfg, "__slots__")
    assert "__slots__" in dir(pcp_wrapper._ProjectionConfig)


def test_compiler_params_caching():
    p1 = pcp_wrapper._get_compiler_params(1024 * 1024)
    p2 = pcp_wrapper._get_compiler_params(1024 * 1024)
    p3 = pcp_wrapper._get_compiler_params(None)
    assert p1 is p2
    assert p1.disable_bounds_checks is True
    assert p1.vmem_limit_bytes == 1024 * 1024
    assert p3.vmem_limit_bytes is None


def test_qkv_receive_group_stride_and_branches():
    s1 = pcp_wrapper._qkv_receive_group_stride(16, 64)
    s2 = pcp_wrapper._qkv_receive_group_stride(16, 64)
    assert s1 == s2
    s3 = pcp_wrapper._qkv_receive_group_stride(64, 16)
    assert s3 >= s1
    # Test caching hit
    info = pcp_wrapper._qkv_receive_group_stride.cache_info()
    assert info.hits > 0


def test_mesh_device_id():
    mesh_axis_names = ("tp", "pcp", "dp")
    pcp_axis_name = "pcp"
    with mock.patch("jax.lax.axis_index", return_value=jnp.array(5)):
        coords = pcp_wrapper._mesh_device_id(
            mesh_axis_names, pcp_axis_name, jnp.array(2)
        )
        assert len(coords) == 3
        np.testing.assert_array_equal(np.asarray(coords[0]), 5)
        np.testing.assert_array_equal(np.asarray(coords[1]), 2)
        np.testing.assert_array_equal(np.asarray(coords[2]), 5)


def test_all_rank_barrier():
    sig_calls = []
    wait_calls = []

    def mock_signal(sem, inc, device_id, device_id_type):
        sig_calls.append((inc, device_id, device_id_type))

    def mock_wait(sem, target):
        wait_calls.append(target)

    with (
        mock.patch.object(pl, "semaphore_signal", side_effect=mock_signal),
        mock.patch.object(pl, "semaphore_wait", side_effect=mock_wait),
        mock.patch("jax.lax.axis_index", return_value=jnp.array(0)),
    ):
        pcp_wrapper._all_rank_barrier(
            barrier_sem=jnp.zeros((1,)),
            rank=jnp.array(1),
            pcp_size=4,
            mesh_axis_names=("pcp",),
            pcp_axis_name="pcp",
        )
        assert len(sig_calls) == 3
        assert wait_calls == [3]


# ---------------------------------------------------------------------------
# Unit tests for state writeback & compact state shapes
# ---------------------------------------------------------------------------


def test_compact_state_updates_shape():
    nhd_pool = jax.ShapeDtypeStruct((8, 256, 1, 4, 128), jnp.float8_e4m3fn)
    region = _region(kb0=0, nblocks=2, row0=0, nrows=16)

    nhd_shape = pcp_wrapper._compact_state_updates_shape(
        nhd_pool, 3, region, whole_block_dma=False
    )
    hnd_shape = pcp_wrapper._compact_state_updates_shape(
        nhd_pool, 3, region, whole_block_dma=True
    )
    assert nhd_shape.shape == (3, 2, 16, 1, 4, 128)
    assert hnd_shape.shape == (3, 2, 256, 1, 4, 128)


def test_validate_compact_state_writeback_plan():
    # whole_block_dma False returns immediately
    plan_nhd = config.StateSourcePlan(
        stride=2,
        conv=_region(kb0=0, nblocks=1, row0=4, nrows=4),
        recurrent=_region(kb0=1, nblocks=1, row0=4, nrows=4),
        whole_block_dma=False,
    )
    assert pcp_wrapper._validate_compact_state_writeback_plan(plan_nhd) is None

    # Valid whole_block_dma
    plan_valid = config.StateSourcePlan(
        stride=3,
        conv=_region(kb0=2, nblocks=1, row0=0, nrows=16),
        recurrent=_region(kb0=0, nblocks=2, row0=0, nrows=256),
        whole_block_dma=True,
    )
    assert pcp_wrapper._validate_compact_state_writeback_plan(plan_valid) is None

    # Invalid conv row0 != 0
    plan_conv_bad = config.StateSourcePlan(
        stride=3,
        conv=_region(kb0=0, nblocks=1, row0=4, nrows=16),
        recurrent=_region(kb0=1, nblocks=1, row0=0, nrows=256),
        whole_block_dma=True,
    )
    with pytest.raises(NotImplementedError, match="page-aligned.*conv"):
        pcp_wrapper._validate_compact_state_writeback_plan(plan_conv_bad)

    # Invalid recurrent row0 != 0
    plan_rec_bad = config.StateSourcePlan(
        stride=3,
        conv=_region(kb0=0, nblocks=1, row0=0, nrows=16),
        recurrent=_region(kb0=1, nblocks=1, row0=8, nrows=256),
        whole_block_dma=True,
    )
    with pytest.raises(NotImplementedError, match="page-aligned.*recurrent"):
        pcp_wrapper._validate_compact_state_writeback_plan(plan_rec_bad)

    # Overlapping regions
    plan_overlap = config.StateSourcePlan(
        stride=3,
        conv=_region(kb0=1, nblocks=2, row0=0, nrows=16),
        recurrent=_region(kb0=0, nblocks=2, row0=0, nrows=256),
        whole_block_dma=True,
    )
    with pytest.raises(NotImplementedError, match="non-overlapping"):
        pcp_wrapper._validate_compact_state_writeback_plan(plan_overlap)


def test_scatter_compact_state_updates_all_branches():
    pool = jnp.zeros((8, 16, 4), dtype=jnp.float32)
    updates = jnp.ones((3, 1, 4, 4), dtype=jnp.float32)
    region = _region(kb0=0, nblocks=1, row0=2, nrows=4)

    # seq 0: valid update (has tokens, state_index >= 0, within bounds)
    # seq 1: inactive (no tokens)
    # seq 2: negative state_index
    state_indices = jnp.array([1, 2, -1], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 10, 10, 20], dtype=jnp.int32)
    num_active_seqs = jnp.array(3, dtype=jnp.int32)

    # Non whole_block_dma
    out = pcp_wrapper._scatter_compact_state_updates(
        pool,
        updates,
        state_indices,
        query_start_loc,
        num_active_seqs,
        state_stride=2,
        region=region,
        whole_block_dma=False,
    )
    np.testing.assert_array_equal(np.asarray(out[2, 2:6]), np.ones((4, 4)))
    # seq 1 had no tokens, so pool at index 4 is untouched
    np.testing.assert_array_equal(np.asarray(out[4]), np.zeros((16, 4)))

    # Whole_block_dma with out-of-bounds block_start
    hnd_pool = jnp.zeros((4, 8, 4), dtype=jnp.float32)
    hnd_updates = jnp.ones((2, 1, 8, 4), dtype=jnp.float32)
    hnd_indices = jnp.array([0, 10], dtype=jnp.int32)  # seq 1 out-of-bounds
    hnd_qstart = jnp.array([0, 5, 10], dtype=jnp.int32)
    hnd_out = pcp_wrapper._scatter_compact_state_updates(
        hnd_pool,
        hnd_updates,
        hnd_indices,
        hnd_qstart,
        jnp.array(2, dtype=jnp.int32),
        state_stride=1,
        region=_region(kb0=0, nblocks=1, row0=0, nrows=8),
        whole_block_dma=True,
    )
    np.testing.assert_array_equal(np.asarray(hnd_out[0]), np.ones((8, 4)))


# ---------------------------------------------------------------------------
# Unit tests for _compute_gdn_tile (all 8 tile prefixes)
# ---------------------------------------------------------------------------


def test_compute_gdn_tile_all_prefixes():
    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=128,
        dim_size=512,
        kernel_size=4,
        tile_size=64,
        num_kq_heads=1,
        num_v_heads=1,
        kq_head_dim=128,
        v_head_dim=128,
    )

    stage_meta = pcp_metadata.PcpStageMetadata(
        num_stages=jnp.array(2, dtype=jnp.int32),
        rank_active_row_end=jnp.zeros((4,), dtype=jnp.int32),
        request_id=jnp.zeros((4,), dtype=jnp.int32),
        query_row_start=jnp.zeros((4,), dtype=jnp.int32),
        num_tokens=jnp.ones((4,), dtype=jnp.int32),
        rank_row_start=jnp.zeros((4, 4), dtype=jnp.int32),
        rank_valid_rows=jnp.zeros((4, 4), dtype=jnp.int32),
        rank_recv_start=jnp.zeros((4, 4), dtype=jnp.int32),
        first_tile=jnp.zeros((4,), dtype=jnp.int32),
        num_tiles=jnp.ones((4,), dtype=jnp.int32),
        tile_stage=jnp.zeros((16,), dtype=jnp.int32),
        tile_row_in_stage=jnp.zeros((16,), dtype=jnp.int32),
        projection_catchup_start=jnp.zeros((4, 4), dtype=jnp.int32),
        projection_catchup_count=jnp.zeros((4, 4), dtype=jnp.int32),
        projection_work_offset=jnp.zeros((4, 4), dtype=jnp.int32),
        projection_work_offset_end=jnp.zeros((4,), dtype=jnp.int32),
    )

    recv_qkv = jnp.zeros((2, 64, 4, 8, 128), dtype=jnp.bfloat16)
    comp_qkv = jnp.zeros((1, 64, 1, 512), dtype=jnp.float32)

    with (
        mock.patch("jax.experimental.pallas.program_id", return_value=0),
        mock.patch("vllm_torchtpu.kernels.gdn.v3.wrapper.inner_kernel"),
    ):
        for prefix in range(8):
            stage_meta_p = dataclasses.replace(
                stage_meta,
                rank_row_start=jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(prefix),
            )
            pcp_wrapper._compute_gdn_tile(
                None,
                None,
                None,
                None,
                None,
                None,  # pyrefly: ignore[bad-argument-type]
                stage_meta_p,
                None,  # pyrefly: ignore[bad-argument-type]
                None,
                None,
                _MockRef(recv_qkv),
                _MockRef(comp_qkv),
                cfg=cfg,
                comm_chunk_size=32,
            )


# ---------------------------------------------------------------------------
# Unit tests for _pcp_qkvz_projection_gdn_outer_kernel
# ---------------------------------------------------------------------------


def _run_outer_kernel_test(
    *,
    is_fp8: bool = False,
    fp8_blocked: bool = False,
    rank: int = 0,
    active_projection_blocks: int = 2,
    catchup_count: int = 1,
    num_stages: int = 2,
):
    pcp_size = 2
    comm_chunk_size = 32
    token_block_size = 64
    dim = 512
    k_dim = 256
    full_key_dim = 128

    proj_cfg = pcp_wrapper._ProjectionConfig(
        token_block_size=token_block_size,
        out_block_size=dim,
        qkv_dim=dim,
        full_key_dim=full_key_dim,
        kq_replication_factor=1,
    )

    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=128,
        dim_size=dim,
        kernel_size=4,
        tile_size=32,
        num_kq_heads=1,
        num_v_heads=2,
        kq_head_dim=128,
        v_head_dim=128,
    )

    stage_meta = _make_stage_metadata(
        num_stages=num_stages,
        pcp_size=pcp_size,
        comm_chunk_size=comm_chunk_size,
        num_tiles_per_stage=2,
        rank_active_row_end=active_projection_blocks * token_block_size,
        catchup_count=catchup_count,
    )

    metadata_ref = mock.MagicMock()
    metadata_ref.num_tiles = jnp.array(num_stages * 2, dtype=jnp.int32)

    hidden_dtype = jnp.bfloat16
    weight_dtype = jnp.float8_e4m3fn if is_fp8 else jnp.bfloat16

    hidden_ref = _MockRef(jnp.zeros((128, k_dim), dtype=hidden_dtype))
    weight_ref = _MockRef(jnp.zeros((dim, k_dim), dtype=weight_dtype))

    if is_fp8:
        if fp8_blocked:
            weight_scale_ref = _MockRef(jnp.ones((2, dim), dtype=jnp.float32))
            weight_scale_vmem_ref = _MockRef(jnp.ones((2, dim), dtype=jnp.float32))
        else:
            weight_scale_ref = _MockRef(jnp.ones((dim,), dtype=jnp.float32))
            weight_scale_vmem_ref = _MockRef(jnp.ones((dim,), dtype=jnp.float32))
        scale_sem = _MockRef(jnp.zeros((1,)))
        x_scale_ref = _MockRef(jnp.ones((token_block_size,), dtype=jnp.float32))
    else:
        weight_scale_ref = None
        weight_scale_vmem_ref = None
        scale_sem = None
        x_scale_ref = None

    def mock_emit_pipeline(body, grid, **kwargs):
        def run(*args, scratches=None, allocations=None, body_prologue=None):
            if body_prologue:
                with mock.patch("jax.experimental.pallas.program_id", return_value=0):
                    body_prologue()
                with mock.patch("jax.experimental.pallas.program_id", return_value=1):
                    body_prologue()
            # Run body for stage 0 first/last tile and stage 1 first tile
            for pid in [0, 1, 2]:
                if pid < len(stage_meta.tile_stage):
                    with mock.patch(
                        "jax.experimental.pallas.program_id", return_value=pid
                    ):
                        body(*args[:5], *scratches)

        return run

    with (
        mock.patch("jax.lax.axis_index", return_value=jnp.array(rank)),
        mock.patch.object(pltpu, "make_async_copy", return_value=_DummyCopy()),
        mock.patch.object(pltpu, "make_async_remote_copy", return_value=_DummyCopy()),
        mock.patch.object(pl, "semaphore_signal", return_value=None),
        mock.patch.object(pl, "semaphore_wait", return_value=None),
        mock.patch(
            "vllm_torchtpu.kernels.gdn.v3.memory_ref.create_allocs",
            _mock_create_allocs,
        ),
        mock.patch.object(pl, "with_scoped", _mock_with_scoped),
        mock.patch.object(pltpu, "emit_pipeline", mock_emit_pipeline),
        mock.patch("vllm_torchtpu.kernels.gdn.v3.wrapper.inner_kernel"),
    ):
        pcp_wrapper._pcp_qkvz_projection_gdn_outer_kernel(
            metadata_ref,
            stage_meta,
            hidden_ref,
            weight_ref,
            weight_scale_ref,
            _MockRef(jnp.zeros((128, 1, 2), dtype=jnp.float32)),
            _MockRef(jnp.zeros((128, 1, 2), dtype=jnp.float32)),
            _MockRef(jnp.zeros((10, 10, 10), dtype=jnp.float32)),
            _MockRef(jnp.zeros((128,), dtype=hidden_dtype)),
            None,
            mock.MagicMock(),
            _MockRef(jnp.zeros((1, 32, 2, 128), dtype=hidden_dtype)),
            _MockRef(jnp.zeros((128, 2, 2, 128), dtype=hidden_dtype)),
            _MockRef(jnp.zeros((2, 1, 16), dtype=jnp.float32)),
            _MockRef(jnp.zeros((2, 1, 16), dtype=jnp.float32)),
            _MockRef(jnp.zeros((2, 16, 4, 8, 128), dtype=hidden_dtype)),
            _MockRef(jnp.zeros((128, 2, 8, 128), dtype=hidden_dtype)),
            _MockRef(jnp.zeros((1,), dtype=jnp.int32)),
            None,
            None,
            _MockRef(jnp.zeros((2,))),
            _MockRef(jnp.zeros((2,))),
            _MockRef(jnp.zeros((2,))),
            _MockRef(jnp.zeros((2,))),
            _MockRef(jnp.zeros((2,))),
            _MockRef(jnp.zeros((2, 64, 4, 8, 128), dtype=hidden_dtype)),
            _MockRef(jnp.zeros((1, 32, 1, 512), dtype=jnp.float32)),
            _MockRef(jnp.zeros((2, 64, 2, 128), dtype=hidden_dtype)),
            _MockRef(jnp.zeros((1,))),
            _MockRef(jnp.zeros((1,))),
            scale_sem,
            _MockRef(jnp.zeros((1,))),
            _MockRef(jnp.zeros((token_block_size, k_dim), dtype=hidden_dtype)),
            _MockRef(jnp.zeros((token_block_size, k_dim), dtype=weight_dtype)),
            x_scale_ref,
            _MockRef(jnp.zeros((dim, k_dim), dtype=weight_dtype)),
            weight_scale_vmem_ref,
            _MockRef(
                jnp.zeros(
                    (token_block_size // 8, dim // 128, 8, 128),
                    dtype=hidden_dtype,
                )
            ),
            _MockRef(jnp.zeros((token_block_size, 8, 128), dtype=hidden_dtype)),
            [0],
            _MockRef(jnp.zeros((1,), dtype=jnp.int32)),
            _MockRef(jnp.zeros((1,))),
            cfg=cfg,
            pcp_size=pcp_size,
            comm_chunk_size=comm_chunk_size,
            projection_cfg=proj_cfg,
            mesh_axis_names=("pcp",),
            pcp_axis_name="pcp",
        )


def test_outer_kernel_bf16_rank0_and_rank1():
    _run_outer_kernel_test(
        is_fp8=False,
        rank=0,
        active_projection_blocks=2,
        catchup_count=1,
        num_stages=3,
    )
    _run_outer_kernel_test(
        is_fp8=False,
        rank=1,
        active_projection_blocks=1,
        catchup_count=0,
        num_stages=2,
    )


def test_outer_kernel_fp8_blocked_and_unblocked():
    # Blocked FP8 (scale ndim == 2)
    _run_outer_kernel_test(
        is_fp8=True,
        fp8_blocked=True,
        rank=0,
        active_projection_blocks=2,
        catchup_count=2,
        num_stages=2,
    )
    # Unblocked FP8 (scale ndim == 1)
    _run_outer_kernel_test(
        is_fp8=True,
        fp8_blocked=False,
        rank=1,
        active_projection_blocks=2,
        catchup_count=1,
        num_stages=2,
    )


def test_outer_kernel_zero_active_projection_blocks():
    _run_outer_kernel_test(
        is_fp8=False,
        rank=0,
        active_projection_blocks=0,
        catchup_count=0,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# Unit tests for fused_qkvz_projection_pcp_gdn input validations
# ---------------------------------------------------------------------------


def _dummy_tpu_info():
    from jax._src.pallas.mosaic.tpu_info import _get_tpu_info_impl

    return _get_tpu_info_impl(pltpu.ChipVersion.TPU_V5P, 1)


def test_fused_qkvz_projection_validations():
    b_val = jnp.zeros((128, 1), dtype=jnp.float32)
    a_val = jnp.zeros((128, 1), dtype=jnp.float32)
    h_val = jnp.zeros((64, 256), dtype=jnp.bfloat16)
    qkvz_w = jnp.zeros((768, 256), dtype=jnp.bfloat16)
    state_src = jnp.zeros((4, 16, 16), dtype=jnp.float32)
    plan = config.StateSourcePlan(
        stride=2,
        conv=_region(kb0=0, nblocks=1, row0=0, nrows=16),
        recurrent=_region(kb0=1, nblocks=1, row0=0, nrows=16),
        whole_block_dma=True,
    )
    conv_w = jnp.zeros((4, 384, 1), dtype=jnp.float32)
    a_log = jnp.zeros((1,), dtype=jnp.float32)
    dt_bias = jnp.zeros((1,), dtype=jnp.float32)
    q_start = jnp.array([0, 64], dtype=jnp.int32)
    state_idx = jnp.array([0], dtype=jnp.int32)
    dist = jnp.array([1], dtype=jnp.int32)
    seq_lens = jnp.array([64], dtype=jnp.int32)

    kwargs = dict(
        n_kq=1,
        n_v=2,
        d_k=128,
        d_v=128,
        kernel_size=4,
        pcp_size=2,
        comm_chunk_size=32,
        mesh_axis_names=("pcp",),
        pcp_axis_name="pcp",
        state_plan=plan,
    )

    with mock.patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        side_effect=_dummy_tpu_info,
    ):
        # 1. pcp_size <= 1
        with pytest.raises(ValueError, match="pcp_size > 1"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "pcp_size": 1},
            )

        # 2. pcp_axis_name absent
        with pytest.raises(ValueError, match="is absent from mesh axes"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "pcp_axis_name": "missing"},
            )

        # 3. pcp_axis_name not unique
        with pytest.raises(ValueError, match="must occur exactly once"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "mesh_axis_names": ("pcp", "pcp")},
            )

        # 4. comm_chunk_size <= 0
        with pytest.raises(ValueError, match="comm_chunk_size must be positive"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "comm_chunk_size": 0},
            )

        # 5. comm_chunk_size unaligned
        with pytest.raises(ValueError, match="VMEM row tile"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "comm_chunk_size": 17},
            )

        # 6. b and a shapes mismatch
        with pytest.raises(ValueError, match="b and a shapes must match"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                jnp.zeros((10, 1), dtype=jnp.float32),
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 7. state_plan is None
        with pytest.raises(ValueError, match="requires a state plan"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "state_plan": None},
            )

        # 8. global batch size mismatch
        with pytest.raises(ValueError, match="Global BA rows must equal"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                jnp.zeros((100, 1), dtype=jnp.float32),
                jnp.zeros((100, 1), dtype=jnp.float32),
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 9. local_num_tokens % comm_chunk_size != 0
        h_unaligned = jnp.zeros((48, 256), dtype=jnp.bfloat16)
        b_unaligned = jnp.zeros((96, 1), dtype=jnp.float32)
        with pytest.raises(ValueError, match="hidden-state rows must be divisible"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_unaligned,
                qkvz_w,
                None,
                b_unaligned,
                b_unaligned,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 10. ndim != 2
        with pytest.raises(ValueError, match="requires rank-2"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                jnp.zeros((64, 256, 1), dtype=jnp.bfloat16),
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 11. hidden_states dtype != bfloat16
        with pytest.raises(ValueError, match="requires BF16 hidden states"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                jnp.zeros((64, 256), dtype=jnp.float32),
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 12. FP8 scale missing
        fp8_w = jnp.zeros((768, 256), dtype=jnp.float8_e4m3fn)
        with pytest.raises(ValueError, match="FP8 projection requires"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                fp8_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 13. FP8 scale not float32
        with pytest.raises(ValueError, match="FP8 projection requires"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                fp8_w,
                jnp.ones((768,), dtype=jnp.bfloat16),
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 14. BF16 weight has scale
        with pytest.raises(ValueError, match="must not have a scale"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                jnp.ones((768,), dtype=jnp.float32),
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 15. Invalid weight dtype (float32)
        with pytest.raises(ValueError, match="requires BF16 or float8_e4m3fn"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                jnp.zeros((768, 256), dtype=jnp.float32),
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 16. K dimension mismatch
        with pytest.raises(ValueError, match="K dimensions differ"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                jnp.zeros((768, 512), dtype=jnp.bfloat16),
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 17. QKVZ width mismatch
        with pytest.raises(ValueError, match="Unexpected QKVZ projection width"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                jnp.zeros((512, 256), dtype=jnp.bfloat16),
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 18. local tokens % projection_token_block_size != 0
        with pytest.raises(ValueError, match="two communication chunks"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "comm_chunk_size": 64},
            )

        # 19. comm_chunk_size % _TPU_TILE_ROWS != 0
        with (
            mock.patch.object(pcp_wrapper, "_TPU_TILE_ROWS", 32),
            pytest.raises(
                ValueError,
                match="The PCP communication chunk must be divisible",
            ),
        ):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "comm_chunk_size": 16},
            )

        # 20. K dim % 256 != 0
        h_bad_k = jnp.zeros((64, 128), dtype=jnp.bfloat16)
        w_bad_k = jnp.zeros((768, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="MXU aligned to 256"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_bad_k,
                w_bad_k,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 21. shard dimensions not tile-aligned
        with pytest.raises(ValueError, match="tile-aligned PCP-local shards"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                jnp.zeros((640, 256), dtype=jnp.bfloat16),
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "d_k": 64},
            )

        # 22. b heads mismatch
        with pytest.raises(ValueError, match="PCP-local BA heads"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                jnp.zeros((128, 4), dtype=jnp.float32),
                jnp.zeros((128, 4), dtype=jnp.float32),
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 23. state_source.ndim < 3
        with pytest.raises(ValueError, match="at least three dimensions"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                jnp.zeros((4, 16), dtype=jnp.float32),
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **kwargs,
            )

        # 24. global_stage_tokens % tile_size != 0
        with pytest.raises(ValueError, match="divisible by the GDN compute tile"):
            pcp_wrapper.fused_qkvz_projection_pcp_gdn(
                h_val,
                qkvz_w,
                None,
                b_val,
                a_val,
                state_src,
                conv_w,
                None,
                a_log,
                dt_bias,
                q_start,
                state_idx,
                dist,
                seq_lens,
                **{**kwargs, "mixed_tile_size": 48},
            )


# ---------------------------------------------------------------------------
# Unit tests for fused_qkvz_projection_pcp_gdn execution (padded & unpadded)
# ---------------------------------------------------------------------------


class _CustomTpuInfo:
    def __init__(self, num_lanes: int):
        self.num_lanes = num_lanes
        self.vmem_capacity_bytes = 67108864


def _fake_pallas_call(f, out_shape, **kwargs):
    def run(*args):
        res = []
        for s in out_shape:
            if hasattr(s, "shape"):
                arr = jnp.zeros(s.shape, dtype=s.dtype)
            else:
                arr = jnp.zeros_like(s)
            res.append(arr)
        # Set active_rows to total tokens so live_row mask is active
        res[6] = jnp.array([64], dtype=jnp.int32)
        return tuple(res)

    return run


def test_fused_qkvz_projection_execution_bf16_unpadded():
    # n_kq=2, n_v=4, pcp_size=2 -> local_n_v=2
    # num_lanes=2 -> aligned_num_v_heads=2 -> num_v_padding=0 (bypasses pad!)
    # conv_bias is None
    b_val = jnp.zeros((128, 2), dtype=jnp.float32)
    a_val = jnp.zeros((128, 2), dtype=jnp.float32)
    h_val = jnp.zeros((64, 256), dtype=jnp.bfloat16)
    expected_dim = 1536
    qkvz_w = jnp.zeros((expected_dim, 256), dtype=jnp.bfloat16)
    state_src = jnp.zeros((4, 16, 16), dtype=jnp.float32)
    plan = config.StateSourcePlan(
        stride=2,
        conv=_region(kb0=0, nblocks=1, row0=0, nrows=16),
        recurrent=_region(kb0=1, nblocks=1, row0=0, nrows=16),
        whole_block_dma=True,
    )
    conv_w = jnp.zeros((4, 512, 1), dtype=jnp.float32)
    a_log = jnp.zeros((2,), dtype=jnp.float32)
    dt_bias = jnp.zeros((2,), dtype=jnp.float32)
    q_start = jnp.array([0, 64], dtype=jnp.int32)
    state_idx = jnp.array([0], dtype=jnp.int32)
    dist = jnp.array([1], dtype=jnp.int32)
    seq_lens = jnp.array([64], dtype=jnp.int32)

    with (
        mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_CustomTpuInfo(num_lanes=2),
        ),
        mock.patch.object(pl, "pallas_call", side_effect=_fake_pallas_call),
    ):
        out_state, packed_out, z = pcp_wrapper.fused_qkvz_projection_pcp_gdn(
            h_val,
            qkvz_w,
            None,
            b_val,
            a_val,
            state_src,
            conv_w,
            None,
            a_log,
            dt_bias,
            q_start,
            state_idx,
            dist,
            seq_lens,
            n_kq=2,
            n_v=4,
            d_k=128,
            d_v=128,
            kernel_size=4,
            pcp_size=2,
            comm_chunk_size=32,
            mesh_axis_names=("pcp",),
            pcp_axis_name="pcp",
            state_plan=plan,
        )
        assert out_state.shape == state_src.shape
        assert packed_out.shape == (64, 4, 128)
        assert z.shape == (64, 4, 128)


def test_fused_qkvz_projection_execution_fp8_padded_and_bias():
    # n_kq=2, n_v=4, pcp_size=2 -> local_n_v=2
    # num_lanes=4 -> aligned_num_v_heads=4 -> num_v_padding=2 > 0 (executes pad!)
    # conv_bias is not None
    # FP8 weights with 2D scale
    b_val = jnp.zeros((128, 2), dtype=jnp.float32)
    a_val = jnp.zeros((128, 2), dtype=jnp.float32)
    h_val = jnp.zeros((64, 256), dtype=jnp.bfloat16)
    expected_dim = 1536
    fp8_w = jnp.zeros((expected_dim, 256), dtype=jnp.float8_e4m3fn)
    scale_2d = jnp.ones((expected_dim, 2), dtype=jnp.float32)
    state_src = jnp.zeros((4, 16, 16), dtype=jnp.float32)
    plan = config.StateSourcePlan(
        stride=2,
        conv=_region(kb0=0, nblocks=1, row0=0, nrows=16),
        recurrent=_region(kb0=1, nblocks=1, row0=0, nrows=16),
        whole_block_dma=True,
    )
    conv_w = jnp.zeros((4, 512, 1), dtype=jnp.float32)
    conv_b = jnp.zeros((512,), dtype=jnp.float32)
    a_log = jnp.zeros((2,), dtype=jnp.float32)
    dt_bias = jnp.zeros((2,), dtype=jnp.float32)
    q_start = jnp.array([0, 64], dtype=jnp.int32)
    state_idx = jnp.array([0], dtype=jnp.int32)
    dist = jnp.array([1], dtype=jnp.int32)
    seq_lens = jnp.array([64], dtype=jnp.int32)

    with (
        mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_CustomTpuInfo(num_lanes=4),
        ),
        mock.patch.object(pl, "pallas_call", side_effect=_fake_pallas_call),
    ):
        out_state, packed_out, z = pcp_wrapper.fused_qkvz_projection_pcp_gdn(
            h_val,
            fp8_w,
            scale_2d,
            b_val,
            a_val,
            state_src,
            conv_w,
            conv_b,
            a_log,
            dt_bias,
            q_start,
            state_idx,
            dist,
            seq_lens,
            n_kq=2,
            n_v=4,
            d_k=128,
            d_v=128,
            kernel_size=4,
            pcp_size=2,
            comm_chunk_size=32,
            mesh_axis_names=("pcp",),
            pcp_axis_name="pcp",
            state_plan=plan,
        )
        assert out_state.shape == state_src.shape
        assert packed_out.shape == (64, 4, 128)
        assert z.shape == (64, 4, 128)


def test_fused_qkvz_projection_execution_fp8_1d_scale():
    # FP8 with 1D scale
    b_val = jnp.zeros((128, 2), dtype=jnp.float32)
    a_val = jnp.zeros((128, 2), dtype=jnp.float32)
    h_val = jnp.zeros((64, 256), dtype=jnp.bfloat16)
    expected_dim = 1536
    fp8_w = jnp.zeros((expected_dim, 256), dtype=jnp.float8_e4m3fn)
    scale_1d = jnp.ones((expected_dim,), dtype=jnp.float32)
    state_src = jnp.zeros((4, 16, 16), dtype=jnp.float32)
    plan = config.StateSourcePlan(
        stride=2,
        conv=_region(kb0=0, nblocks=1, row0=0, nrows=16),
        recurrent=_region(kb0=1, nblocks=1, row0=0, nrows=16),
        whole_block_dma=True,
    )
    conv_w = jnp.zeros((4, 512, 1), dtype=jnp.float32)
    a_log = jnp.zeros((2,), dtype=jnp.float32)
    dt_bias = jnp.zeros((2,), dtype=jnp.float32)
    q_start = jnp.array([0, 64], dtype=jnp.int32)
    state_idx = jnp.array([0], dtype=jnp.int32)
    dist = jnp.array([1], dtype=jnp.int32)
    seq_lens = jnp.array([64], dtype=jnp.int32)

    with (
        mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_CustomTpuInfo(num_lanes=2),
        ),
        mock.patch.object(pl, "pallas_call", side_effect=_fake_pallas_call),
    ):
        out_state, packed_out, z = pcp_wrapper.fused_qkvz_projection_pcp_gdn(
            h_val,
            fp8_w,
            scale_1d,
            b_val,
            a_val,
            state_src,
            conv_w,
            None,
            a_log,
            dt_bias,
            q_start,
            state_idx,
            dist,
            seq_lens,
            n_kq=2,
            n_v=4,
            d_k=128,
            d_v=128,
            kernel_size=4,
            pcp_size=2,
            comm_chunk_size=32,
            mesh_axis_names=("pcp",),
            pcp_axis_name="pcp",
            state_plan=plan,
        )
        assert out_state.shape == state_src.shape
        assert packed_out.shape == (64, 4, 128)
        assert z.shape == (64, 4, 128)
