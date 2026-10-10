"""Unit tests for RPA HD=64 tuned block sizes and SparseCore ragged gather reduce.

Achieves 100.0% statement and branch coverage on:
- src/vllm_torchtpu/kernels/ragged_paged_attention/v3/tuned_block_sizes_hd64.py
- src/vllm_torchtpu/kernels/sparse_core/ragged_gather_reduce.py
"""

from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp
import pytest

import vllm_torchtpu.kernels.ragged_paged_attention.v3.tuned_block_sizes_hd64 as tbs
import vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce as rgr

# ==============================================================================
# Helper Classes & Functions for Simulating Pallas Kernels
# ==============================================================================


class FakeDMA:
    """Mock DMA descriptor supporting start() and wait()."""

    def start(self):
        return self

    def wait(self):
        return self


class FakeRef:
    """Mock JAX Pallas Ref supporting indexing, slice assignment, and bitcast."""

    class AtIndexer:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, idx):
            return self.parent

    def __init__(self, arr):
        self.arr = jnp.array(arr)
        self.dtype = self.arr.dtype
        self.shape = self.arr.shape

    @property
    def at(self):
        return FakeRef.AtIndexer(self)

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self.arr
        return self.arr[idx]

    def __setitem__(self, idx, val):
        if idx is Ellipsis:
            self.arr = val
        else:
            self.arr = self.arr.at[idx].set(val)

    def bitcast(self, dtype):
        return FakeRef(jnp.zeros(self.shape, dtype=dtype))

    def __getattr__(self, name):
        return getattr(self.arr, name)


def fake_loop(start, stop, step=1, init_carry=None):
    """Executes the loop synchronously for all iterations to cover all branches."""

    def decorator(fn):
        carry = init_carry
        s = int(start)
        e = int(stop)
        st = int(step)
        for i in range(s, e, st):
            if carry is not None:
                carry = fn(i, carry)
            else:
                fn(i)
        return fn

    return decorator


def fake_emit_pipeline(fn=None, **kwargs):
    """Mocks pltpu.emit_pipeline by returning the unadorned inner function."""
    if fn is not None:
        return fn

    def decorator(inner_fn):
        return inner_fn

    return decorator


# ==============================================================================
# 1. Tests for tuned_block_sizes_hd64.py
# ==============================================================================


@pytest.fixture(autouse=True)
def _clear_tuned_block_sizes_cache():
    tbs.get_tuned_block_sizes.cache_clear()
    yield
    tbs.get_tuned_block_sizes.cache_clear()


def test_tpu_version_below_4_raises():
    """TPU versions < 4 must raise NotImplementedError."""
    tbs.get_tuned_block_sizes.cache_clear()
    with (
        mock.patch.object(tbs, "get_tpu_version", return_value=3),
        pytest.raises(NotImplementedError, match="TPU version must be 4 or higher."),
    ):
        tbs.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=8,
            actual_num_kv_heads=2,
            head_dim=64,
            page_size=128,
            max_num_tokens=4096,
            pages_per_seq=32,
        )


def test_tpu_version_4_fixed_sizes():
    """TPUv4 sets default fixed block sizes (512 // page_size, 32)."""
    tbs.get_tuned_block_sizes.cache_clear()
    with (
        mock.patch.object(tbs, "get_tpu_version", return_value=4),
        mock.patch.object(tbs, "get_device_name", return_value="UnknownTPU"),
    ):
        bkv_p, bq = tbs.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=8,
            actual_num_kv_heads=2,
            head_dim=64,
            page_size=128,
            max_num_tokens=4096,
            pages_per_seq=32,
        )
        # 512 // 128 = 4, 32
        assert bkv_p == 4
        assert bq == 32


def test_tpu_version_7_fixed_sizes():
    """TPUv7 sets default fixed block sizes (4096 // page_size, 32)."""
    tbs.get_tuned_block_sizes.cache_clear()
    with (
        mock.patch.object(tbs, "get_tpu_version", return_value=7),
        mock.patch.object(tbs, "get_device_name", return_value="UnknownTPU"),
    ):
        bkv_p, bq = tbs.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=8,
            actual_num_kv_heads=2,
            head_dim=64,
            page_size=128,
            max_num_tokens=4096,
            pages_per_seq=32,
        )
        # 4096 // 128 = 32, 32
        assert bkv_p == 32
        assert bq == 32


def test_tpu_version_default_other_and_keyerror_fallback():
    """TPUv5 or other unknown version sets (2048 // page_size, 32)."""
    tbs.get_tuned_block_sizes.cache_clear()
    with (
        mock.patch.object(tbs, "get_tpu_version", return_value=5),
        mock.patch.object(tbs, "get_device_name", return_value="TPU Unknown"),
    ):
        bkv_p, bq = tbs.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=8,
            actual_num_kv_heads=2,
            head_dim=64,
            page_size=128,
            max_num_tokens=4096,
            pages_per_seq=32,
        )
        # 2048 // 128 = 16, 32
        assert bkv_p == 16
        assert bq == 32


def test_tuned_block_sizes_found_and_cached():
    """Hits TUNED_BLOCK_SIZES table for TPU v5e and verifies cache."""
    tbs.get_tuned_block_sizes.cache_clear()
    tbs.get_simplified_raw_key.cache_clear()

    with (
        mock.patch.object(tbs, "get_tpu_version", return_value=5),
        mock.patch.object(tbs, "get_device_name", return_value="TPU v5e"),
    ):
        # TPU v5e, page_size=128, q_bfloat16_kv_bfloat16,
        # q_head-8_kv_head-2_head-64, max_model_len=4096 -> returns (16, 32)
        res1 = tbs.get_tuned_block_sizes(
            jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 4096, 32
        )
        assert res1 == (16, 32)

        # Second call hits cache
        info_before = tbs.get_tuned_block_sizes.cache_info()
        res2 = tbs.get_tuned_block_sizes(
            jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 4096, 32
        )
        assert res2 == (16, 32)
        info_after = tbs.get_tuned_block_sizes.cache_info()
        assert info_after.hits > info_before.hits


def test_tuned_block_sizes_bounding_pages_and_tokens():
    """Verifies bounding when pages_per_seq < bkv_p and max_num_tokens < bq."""
    tbs.get_tuned_block_sizes.cache_clear()
    with (
        mock.patch.object(tbs, "get_tpu_version", return_value=5),
        mock.patch.object(tbs, "get_device_name", return_value="TPU v5e"),
    ):
        # For pages_per_seq=10, page_size*pages_per_seq = 1280 -> next pow 2 is 2048.
        # Tuned block size for 2048 is (16, 64). Bounding gives (10, 8).
        res = tbs.get_tuned_block_sizes(
            jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 8, 10
        )
        assert res == (10, 8)


def test_get_lookup_keys_and_simplified_raw_key():
    """Tests get_lookup_keys and get_simplified_raw_key assertion checks."""
    with mock.patch.object(tbs, "get_device_name", return_value="TPU v5e"):
        keys = tbs.get_lookup_keys(
            page_size=128,
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            max_model_len=4096,
        )
        assert keys[0] == "TPU v5e"
        assert keys[1] == 128
        assert keys[2] == "q_bfloat16_kv_bfloat16"

    # Assertion failures
    with pytest.raises(AssertionError):
        tbs.get_simplified_raw_key(
            page_size=128,
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=8,
            actual_num_kv_heads=2,
            head_dim=128,  # Not 64
            max_model_len=4096,
        )

    with pytest.raises(AssertionError):
        tbs.get_simplified_raw_key(
            page_size=128,
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=7,  # 7 % 2 != 0
            actual_num_kv_heads=2,
            head_dim=64,
            max_model_len=4096,
        )


# ==============================================================================
# 2. Tests for ragged_gather_reduce.py
# ==============================================================================


def test_align_to():
    """Tests _align_to."""
    assert rgr._align_to(10, 4) == 12
    assert rgr._align_to(12, 4) == 12
    assert rgr._align_to(0, 4) == 0


def test_fallback_implementation():
    """Tests _fallback_implementation math."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    indices = jnp.array([0, 1, 2, 3])
    topk_weights = jnp.array([1.0, 1.0, 1.0, 1.0])
    valid_rows_mask = jnp.array([True, True, True, False])
    out = rgr._fallback_implementation(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size=2
    )
    assert out.shape == (2, 2)
    # group 1: x[0] + x[1] = [4.0, 6.0]
    assert jnp.allclose(out[0], jnp.array([4.0, 6.0]))
    # group 2: x[2] + 0 = [5.0, 6.0]
    assert jnp.allclose(out[1], jnp.array([5.0, 6.0]))


def test_pad_inputs_if_needed_branches():
    """Covers all 4 padding cases in _pad_inputs_if_needed."""
    # Case 1: pad_input_size == 0 and pad_hidden_size == 0
    x1 = jnp.ones((128, 128), dtype=jnp.bfloat16)
    ind1 = jnp.arange(128, dtype=jnp.int32)
    w1 = jnp.ones(128, dtype=jnp.float32)
    m1 = jnp.ones(128, dtype=bool)
    xp1, ip1, wp1, mp1 = rgr._pad_inputs_if_needed(x1, ind1, w1, m1, 1, 1, 1, 128)
    assert xp1.shape == (128, 128)
    assert ip1.shape == (128,)

    # Case 2: pad_input_size > 0 and pad_hidden_size == 0
    x2 = jnp.ones((100, 128), dtype=jnp.bfloat16)
    ind2 = jnp.arange(100, dtype=jnp.int32)
    w2 = jnp.ones(100, dtype=jnp.float32)
    m2 = jnp.ones(100, dtype=bool)
    xp2, ip2, wp2, mp2 = rgr._pad_inputs_if_needed(x2, ind2, w2, m2, 1, 1, 1, 128)
    assert xp2.shape == (128, 128)
    assert ip2.shape == (128,)
    assert not mp2[-1]  # padded elements are False

    # Case 3: pad_input_size == 0 and pad_hidden_size > 0
    x3 = jnp.ones((128, 100), dtype=jnp.bfloat16)
    ind3 = jnp.arange(128, dtype=jnp.int32)
    w3 = jnp.ones(128, dtype=jnp.float32)
    m3 = jnp.ones(128, dtype=bool)
    xp3, ip3, wp3, mp3 = rgr._pad_inputs_if_needed(x3, ind3, w3, m3, 1, 1, 1, 128)
    assert xp3.shape == (128, 128)
    assert ip3.shape == (128,)

    # Case 4: pad_input_size > 0 and pad_hidden_size > 0
    x4 = jnp.ones((100, 100), dtype=jnp.bfloat16)
    ind4 = jnp.arange(100, dtype=jnp.int32)
    w4 = jnp.ones(100, dtype=jnp.float32)
    m4 = jnp.ones(100, dtype=bool)
    xp4, ip4, wp4, mp4 = rgr._pad_inputs_if_needed(x4, ind4, w4, m4, 1, 1, 1, 128)
    assert xp4.shape == (128, 128)
    assert ip4.shape == (128,)


def test_calculate_num_col_column_partitions_branches():
    """Covers all branches and cache in _calculate_num_col_column_partitions."""
    rgr._calculate_num_col_column_partitions.cache_clear()

    # Case 1: cond1 fails (num_cores % 2 != 0) -> returns 1
    assert rgr._calculate_num_col_column_partitions(1024, 3, 128) == 1

    # Case 2: cond2 fails (hidden_size % (num_lanes * 2) != 0) -> returns 1
    assert rgr._calculate_num_col_column_partitions(200, 4, 128) == 1

    # Case 3: cond3 fails (hidden_size // (num_lanes * 2) < 4) -> returns 1
    assert rgr._calculate_num_col_column_partitions(256, 4, 128) == 1

    # Case 4: 1 iteration passes -> returns 2
    assert rgr._calculate_num_col_column_partitions(1024, 16, 128) == 2

    # Case 5: 2 iterations pass -> returns 4
    assert rgr._calculate_num_col_column_partitions(2048, 16, 128) == 4

    # Verify cache hits
    info = rgr._calculate_num_col_column_partitions.cache_info()
    rgr._calculate_num_col_column_partitions(2048, 16, 128)
    info2 = rgr._calculate_num_col_column_partitions.cache_info()
    assert info2.hits > info.hits


def test_preprocess():
    """Tests _preprocess sorting, partitioning, and assertions."""
    indices = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    mask = jnp.array([True, False, False, False])
    dst_indices, sorted_val, num_src, group_mask = rgr._preprocess(
        indices=indices,
        valid_rows_mask=mask,
        reduce_group_size=2,
        num_row_partitions=1,
        num_simd_lanes=4,
    )
    assert dst_indices.shape == (4,)
    assert sorted_val.shape == (4,)
    assert num_src.shape == (4,)
    # group 1 has 1 true, group 2 has none
    assert group_mask[0]
    assert not group_mask[1]

    # AssertionError on indices.ndim != 1
    with pytest.raises(
        AssertionError, match="Ragged scatter only supports 1d indices."
    ):
        rgr._preprocess(
            indices=jnp.ones((2, 2)),
            valid_rows_mask=mask,
            reduce_group_size=2,
            num_row_partitions=1,
            num_simd_lanes=4,
        )

    # AssertionError on num_row_partitions > num_simd_lanes
    indices_8 = jnp.arange(8, dtype=jnp.int32)
    mask_8 = jnp.ones(8, dtype=bool)
    with pytest.raises(AssertionError):
        rgr._preprocess(
            indices=indices_8,
            valid_rows_mask=mask_8,
            reduce_group_size=2,
            num_row_partitions=8,
            num_simd_lanes=4,
        )


def test_ragged_gather_reduce_input_assertions():
    """Verifies dimension assertion errors in ragged_gather_reduce."""
    x = jnp.ones((4, 4))
    ind = jnp.arange(4)
    w = jnp.ones(4)
    m = jnp.ones(4, dtype=bool)

    with pytest.raises(
        AssertionError, match="ragged_gather_reduce only supports 2d inputs."
    ):
        rgr.ragged_gather_reduce(jnp.ones((4, 4, 4)), ind, w, m, reduce_group_size=2)

    with pytest.raises(
        AssertionError, match="ragged_gather_reduce only supports 1d indices."
    ):
        rgr.ragged_gather_reduce(x, jnp.ones((4, 4)), w, m, reduce_group_size=2)

    with pytest.raises(
        AssertionError,
        match="ragged_gather_reduce only supports 1d topk_weights.",
    ):
        rgr.ragged_gather_reduce(x, ind, jnp.ones((4, 4)), m, reduce_group_size=2)

    with pytest.raises(
        AssertionError,
        match="ragged_gather_reduce only supports 1d valid_rows_mask.",
    ):
        rgr.ragged_gather_reduce(x, ind, w, jnp.ones((4, 4)), reduce_group_size=2)


def test_ragged_gather_reduce_fallback_paths():
    """Verifies fallback when sc_info is None or input is smaller than threshold."""
    x = jnp.ones((8, 128), dtype=jnp.bfloat16)
    ind = jnp.arange(8, dtype=jnp.int32)
    w = jnp.ones(8, dtype=jnp.float32)
    m = jnp.ones(8, dtype=bool)

    # Path 1: sc_info is None
    tpu_info_no_sc = SimpleNamespace(sparse_core=None)
    with mock.patch(
        "jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info_no_sc
    ):
        out = rgr.ragged_gather_reduce.__wrapped__(x, ind, w, m, 2)
        assert out.shape == (4, 128)

    # Path 2: sc_info is not None, but input size < 0.6 * vmem_capacity_bytes
    sc_info = SimpleNamespace(num_lanes=128, num_cores=4, num_subcores=4)
    tpu_info_large_vmem = SimpleNamespace(
        sparse_core=sc_info,
        num_lanes=128,
        vmem_capacity_bytes=100 * 1024 * 1024,
    )
    with mock.patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=tpu_info_large_vmem,
    ):
        out = rgr.ragged_gather_reduce.__wrapped__(x, ind, w, m, 2)
        assert out.shape == (4, 128)


def test_ragged_gather_reduce_full_sparse_core_path():
    """Covers full SparseCore path of ragged_gather_reduce."""
    sc_info = SimpleNamespace(num_lanes=128, num_cores=4, num_subcores=4)
    # Set vmem_capacity_bytes=0 so size >= 0.6 * vmem is True
    tpu_info = SimpleNamespace(
        sparse_core=sc_info, num_lanes=128, vmem_capacity_bytes=0
    )

    x = jnp.ones((256, 128), dtype=jnp.bfloat16)
    indices = jnp.arange(256, dtype=jnp.int32)
    weights = jnp.ones(256, dtype=jnp.float32)
    # Mask with some False to exercise jnp.where(mask, ...)
    mask = jnp.array([True] * 250 + [False] * 6)

    def fake_kernel(fn, **kwargs):
        out_type = kwargs["out_type"]

        def run(*args):
            return jnp.ones(out_type.shape, dtype=out_type.dtype)

        return run

    fake_mesh = SimpleNamespace(core_axis_name="core", subcore_axis_name="subcore")

    with (
        mock.patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info),
        mock.patch(
            "jax.experimental.pallas.tpu_sc.VectorSubcoreMesh",
            return_value=fake_mesh,
        ),
        mock.patch(
            "vllm_torchtpu.kernels.sparse_core.core_map_helper.kernel",
            side_effect=fake_kernel,
        ),
    ):
        out = rgr.ragged_gather_reduce.__wrapped__(x, indices, weights, mask, 8)
        assert out.shape == (32, 128)

        # Already-aligned input (2048 rows, 128 cols) so no post-kernel
        # slicing is needed (602->605).
        x_aligned = jnp.ones((2048, 128), dtype=jnp.bfloat16)
        indices_aligned = jnp.arange(2048, dtype=jnp.int32)
        weights_aligned = jnp.ones(2048, dtype=jnp.bfloat16)
        mask_aligned = jnp.ones(2048, dtype=bool)
        out_aligned = rgr.ragged_gather_reduce.__wrapped__(
            x_aligned, indices_aligned, weights_aligned, mask_aligned, 8
        )
        assert out_aligned.shape == (256, 128)


# ==============================================================================
# 3. Tests for main_kernel inside ragged_gather_reduce.py
# ==============================================================================


def test_main_kernel_sc_info_none_assert():
    """Verifies assert sc_info is not None in main_kernel."""
    tpu_info_no_sc = SimpleNamespace(sparse_core=None)
    with (
        mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info_no_sc
        ),
        pytest.raises(AssertionError),
    ):
        rgr.main_kernel(
            num_rows_per_row_partition_ref=None,
            in_hbm_ref=None,
            indices_hbm_ref=None,
            dst_indices_hbm_ref=None,
            topk_weights_hbm_ref=None,
            sorted_by_validity_hbm_ref=None,
            out_hbm_ref=None,
            num_rows_per_row_partition_vmem_ref=None,
            out_vmem_ref=None,
            prev_iter_last_row_vmem_ref=None,
            src_indices_vmem_ref=None,
            dst_indices_vmem_ref=None,
            topk_weights_vmem_ref=None,
            sorted_by_validity_vmem_ref=None,
            sem_ref=None,
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_row_partitions=1,
            num_column_partitions=1,
        )


def test_main_kernel_execution_all_branches():
    """Executes main_kernel simulating all inner branches with bfloat16 and float32."""
    num_simd_lanes = 4
    num_lanes = 4
    col_size = 4

    sc_info = SimpleNamespace(num_lanes=num_simd_lanes, num_cores=2, num_subcores=1)
    tpu_info = SimpleNamespace(
        sparse_core=sc_info,
        num_lanes=num_lanes,
        vmem_capacity_bytes=16 * 1024 * 1024,
    )

    # 1. Test bfloat16 path
    # We set 2 row tiles so row_block_id=0 and row_block_id=1 both execute!
    # row_partition_size = 16 // 2 = 8
    # partition 0 has 7 rows (so row 7 is invalid, exercising row_valid=False).
    num_rows_current = 7
    in_hbm_bf16 = FakeRef(jnp.zeros((16, col_size), dtype=jnp.bfloat16))
    out_hbm_f32 = FakeRef(jnp.zeros((16, col_size), dtype=jnp.float32))

    # dst_indices has matching adjacent rows to exercise
    # dst_row_hbm == prev_row_hbm: [0, 0, 1, 2]
    dst_indices_arr = jnp.array([0, 0, 1, 2], dtype=jnp.int32)
    # src_indices has 0 (pack 0 -> shift 16) and 1 (pack 1 -> shift 0)
    src_indices_arr = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    with (
        mock.patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info),
        mock.patch("jax.experimental.pallas.loop", side_effect=fake_loop),
        mock.patch(
            "jax.experimental.pallas.tpu.emit_pipeline",
            side_effect=fake_emit_pipeline,
        ),
        mock.patch(
            "jax.experimental.pallas.tpu.make_async_copy",
            return_value=FakeDMA(),
        ),
        mock.patch("jax.experimental.pallas.program_id", return_value=0),
        mock.patch(
            "jax.experimental.pallas.ds",
            side_effect=lambda s, sz: slice(int(s), int(s + sz)),
        ),
        mock.patch(
            "jax.experimental.pallas.cdiv",
            side_effect=lambda a, b: (int(a) + int(b) - 1) // int(b),
        ),
        mock.patch(
            "jax.experimental.pallas.tpu_sc.bitcast",
            side_effect=lambda x, dt: jnp.asarray(x).astype(dt),
        ),
        mock.patch("jax.lax.axis_size", return_value=2),
    ):
        rgr.main_kernel(
            num_rows_per_row_partition_ref=FakeRef(
                jnp.array([num_rows_current, 0, 0, 0], dtype=jnp.int32)
            ),
            in_hbm_ref=in_hbm_bf16,
            indices_hbm_ref=FakeRef(src_indices_arr),
            dst_indices_hbm_ref=FakeRef(dst_indices_arr),
            topk_weights_hbm_ref=FakeRef(
                jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
            ),
            sorted_by_validity_hbm_ref=FakeRef(
                jnp.array([0, 1, 2, 3], dtype=jnp.int32)
            ),
            out_hbm_ref=out_hbm_f32,
            num_rows_per_row_partition_vmem_ref=FakeRef(
                jnp.array([num_rows_current, 0, 0, 0], dtype=jnp.int32)
            ),
            out_vmem_ref=FakeRef(jnp.zeros((4, col_size), dtype=jnp.uint32)),
            prev_iter_last_row_vmem_ref=FakeRef(
                jnp.zeros((1, col_size), dtype=jnp.uint32)
            ),
            src_indices_vmem_ref=FakeRef(src_indices_arr),
            dst_indices_vmem_ref=FakeRef(dst_indices_arr),
            topk_weights_vmem_ref=FakeRef(
                jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
            ),
            sorted_by_validity_vmem_ref=FakeRef(
                jnp.array([0, 1, 2, 3], dtype=jnp.int32)
            ),
            sem_ref=FakeRef(jnp.zeros(2, dtype=jnp.int32)),
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_row_partitions=2,
            num_column_partitions=1,
        )

        # 2. Test float32 path
        in_hbm_f32 = FakeRef(jnp.zeros((16, col_size), dtype=jnp.float32))
        rgr.main_kernel(
            num_rows_per_row_partition_ref=FakeRef(
                jnp.array([num_rows_current, 0, 0, 0], dtype=jnp.int32)
            ),
            in_hbm_ref=in_hbm_f32,
            indices_hbm_ref=FakeRef(src_indices_arr),
            dst_indices_hbm_ref=FakeRef(dst_indices_arr),
            topk_weights_hbm_ref=FakeRef(
                jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
            ),
            sorted_by_validity_hbm_ref=FakeRef(
                jnp.array([0, 1, 2, 3], dtype=jnp.int32)
            ),
            out_hbm_ref=out_hbm_f32,
            num_rows_per_row_partition_vmem_ref=FakeRef(
                jnp.array([num_rows_current, 0, 0, 0], dtype=jnp.int32)
            ),
            out_vmem_ref=FakeRef(jnp.zeros((4, col_size), dtype=jnp.uint32)),
            prev_iter_last_row_vmem_ref=FakeRef(
                jnp.zeros((1, col_size), dtype=jnp.uint32)
            ),
            src_indices_vmem_ref=FakeRef(src_indices_arr),
            dst_indices_vmem_ref=FakeRef(dst_indices_arr),
            topk_weights_vmem_ref=FakeRef(
                jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
            ),
            sorted_by_validity_vmem_ref=FakeRef(
                jnp.array([0, 1, 2, 3], dtype=jnp.int32)
            ),
            sem_ref=FakeRef(jnp.zeros(2, dtype=jnp.int32)),
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_row_partitions=2,
            num_column_partitions=1,
        )

        # 3. Test unsupported dtype raises ValueError
        in_hbm_i32 = FakeRef(jnp.zeros((16, col_size), dtype=jnp.int32))
        with pytest.raises(
            ValueError, match="Not yet support extracting data from dtype:"
        ):
            rgr.main_kernel(
                num_rows_per_row_partition_ref=FakeRef(
                    jnp.array([num_rows_current, 0, 0, 0], dtype=jnp.int32)
                ),
                in_hbm_ref=in_hbm_i32,
                indices_hbm_ref=FakeRef(src_indices_arr),
                dst_indices_hbm_ref=FakeRef(dst_indices_arr),
                topk_weights_hbm_ref=FakeRef(
                    jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
                ),
                sorted_by_validity_hbm_ref=FakeRef(
                    jnp.array([0, 1, 2, 3], dtype=jnp.int32)
                ),
                out_hbm_ref=out_hbm_f32,
                num_rows_per_row_partition_vmem_ref=FakeRef(
                    jnp.array([num_rows_current, 0, 0, 0], dtype=jnp.int32)
                ),
                out_vmem_ref=FakeRef(jnp.zeros((4, col_size), dtype=jnp.uint32)),
                prev_iter_last_row_vmem_ref=FakeRef(
                    jnp.zeros((1, col_size), dtype=jnp.uint32)
                ),
                src_indices_vmem_ref=FakeRef(src_indices_arr),
                dst_indices_vmem_ref=FakeRef(dst_indices_arr),
                topk_weights_vmem_ref=FakeRef(
                    jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
                ),
                sorted_by_validity_vmem_ref=FakeRef(
                    jnp.array([0, 1, 2, 3], dtype=jnp.int32)
                ),
                sem_ref=FakeRef(jnp.zeros(2, dtype=jnp.int32)),
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_partitions=2,
                num_column_partitions=1,
            )
