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

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.causal_conv1d import strided_ldst
from vllm_torchtpu.kernels.sparse_core import (
    core_map_helper,
)
from vllm_torchtpu.kernels.sparse_core import (
    ragged_gather_reduce as rgr,
)


class _FakeRef:
    """Lightweight mutable array reference for CPU unit tests."""

    def __init__(self, arr: jax.Array):
        self.arr = arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    @property
    def dtype(self) -> jnp.dtype:
        return self.arr.dtype

    @property
    def at(self) -> "_FakeRef":
        return self

    def bitcast(self, target_dtype: jnp.dtype) -> "_FakeRef":
        if self.dtype == target_dtype:
            return self
        src_bits = jax.dtypes.itemsize_bits(self.dtype)
        dst_bits = jax.dtypes.itemsize_bits(target_dtype)
        if dst_bits > src_bits:
            pack = dst_bits // src_bits
            new_rows = max(1, self.shape[0] // pack)
            return _FakeRef(jnp.zeros((new_rows, *self.shape[1:]), dtype=target_dtype))
        return _FakeRef(jnp.zeros(self.shape, dtype=target_dtype))

    def _norm_idx(self, idx):
        if isinstance(idx, tuple):
            return tuple(self._norm_single(i) for i in idx)
        return self._norm_single(idx)

    @staticmethod
    def _norm_single(i):
        if isinstance(i, _FakeRef):
            return i.arr
        if hasattr(i, "start") and hasattr(i, "size"):
            return slice(int(i.start), int(i.start) + int(i.size))
        return i

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self.arr
        norm = self._norm_idx(idx)
        return self.arr[norm]

    def __setitem__(self, idx, val):
        if idx is Ellipsis:
            self.arr = jnp.asarray(val, dtype=self.arr.dtype)
            return
        norm = self._norm_idx(idx)
        self.arr = self.arr.at[norm].set(jnp.asarray(val, dtype=self.arr.dtype))


class _NoopDMA:
    def start(self):
        return None

    def wait(self):
        return None


def test_strided_ldst_load_large_to_compact():
    ref_f32 = _FakeRef(jnp.arange(16, dtype=jnp.float32).reshape(4, 4))
    out_same = strided_ldst.load_large_to_compact(ref_f32, dst_dtype=None)
    assert out_same.shape == (4, 1, 4)
    np.testing.assert_allclose(out_same[:, 0, :], ref_f32.arr)

    out_same_explicit = strided_ldst.load_large_to_compact(
        ref_f32, dst_dtype=jnp.float32
    )
    np.testing.assert_allclose(out_same_explicit, out_same)

    ref_bf16 = _FakeRef(jnp.ones((4, 8), dtype=jnp.bfloat16))
    with patch.object(
        strided_ldst.pltpu,
        "unpack_elementwise",
        side_effect=lambda packed, index, packed_dtype, unpacked_dtype: (
            jnp.full(packed.shape, float(index + 1), dtype=unpacked_dtype)
        ),
    ):
        out_unpacked = strided_ldst.load_large_to_compact(
            ref_bf16, dst_dtype=jnp.float32
        )
    assert out_unpacked.shape == (4, 1, 8)
    assert out_unpacked.dtype == jnp.float32


def test_core_map_helper_all_branches():
    # Direct test of _empty_out_ref with both MemorySpace and custom memory_space,
    # plus _deref and kernel with single and multiple outputs.
    aval_default = SimpleNamespace(
        shape=(2, 4),
        dtype=jnp.float32,
        sharding=None,
        memory_space=core_map_helper.jax_core.MemorySpace.Device,
    )
    aval_custom = SimpleNamespace(
        shape=(2, 4),
        dtype=jnp.float32,
        sharding=None,
        memory_space="custom_vmem",
    )
    with (
        patch.object(
            core_map_helper.pl_core,
            "_convert_out_shape_to_aval",
            side_effect=[aval_default, aval_custom],
        ),
        patch.object(
            core_map_helper.lax,
            "empty",
            side_effect=lambda shape, dtype, out_sharding=None: jnp.zeros(
                shape, dtype=dtype
            ),
        ),
        patch.object(
            core_map_helper.jax_core,
            "new_ref",
            side_effect=lambda val, memory_space=None: _FakeRef(val),
        ),
    ):
        ref1 = core_map_helper._empty_out_ref(jax.ShapeDtypeStruct((2, 4), jnp.float32))
        ref2 = core_map_helper._empty_out_ref(jax.ShapeDtypeStruct((2, 4), jnp.float32))
        assert core_map_helper._deref(ref1).shape == (2, 4)
        assert core_map_helper._deref(ref2).shape == (2, 4)

    def _fake_core_map(mesh, **kwargs):
        def _decorator(fn):
            fn()
            return fn

        return _decorator

    with (
        patch.object(core_map_helper, "api", SimpleNamespace(jit=lambda fn: fn)),
        patch.object(core_map_helper.pl_core, "core_map", side_effect=_fake_core_map),
        patch.object(
            core_map_helper.jax_core,
            "new_ref",
            side_effect=lambda val, memory_space=None: _FakeRef(val),
        ),
        patch.object(
            core_map_helper,
            "_empty_out_ref",
            side_effect=lambda out_t: _FakeRef(
                jnp.zeros(out_t.shape, dtype=out_t.dtype)
            ),
        ),
    ):
        # Single output
        def _single_body(x_ref, out_ref):
            out_ref[...] = x_ref[...] + 1.0

        run_single = core_map_helper.kernel(
            _single_body,
            out_type=jax.ShapeDtypeStruct((4,), jnp.float32),
            mesh="fake_mesh",
        )
        res_single = run_single(jnp.arange(4, dtype=jnp.float32))
        np.testing.assert_allclose(res_single, jnp.arange(1, 5, dtype=jnp.float32))

        # Tuple output
        def _multi_body(x_ref, out0_ref, out1_ref):
            out0_ref[...] = x_ref[...] * 2.0
            out1_ref[...] = x_ref[...] * 3.0

        run_multi = core_map_helper.kernel(
            _multi_body,
            out_type=(
                jax.ShapeDtypeStruct((4,), jnp.float32),
                jax.ShapeDtypeStruct((4,), jnp.float32),
            ),
            mesh="fake_mesh",
        )
        res_multi = run_multi(jnp.ones((4,), dtype=jnp.float32))
        assert isinstance(res_multi, tuple) and len(res_multi) == 2
        np.testing.assert_allclose(res_multi[0], jnp.full((4,), 2.0))
        np.testing.assert_allclose(res_multi[1], jnp.full((4,), 3.0))


def test_ragged_gather_reduce_helpers_and_preprocess():
    assert rgr._align_to(130, 128) == 256

    rgr._calculate_num_col_column_partitions.cache_clear()
    # Enters while loop multiple times
    assert rgr._calculate_num_col_column_partitions(4096, 8, 128) == 8
    # Does not enter while loop
    assert rgr._calculate_num_col_column_partitions(128, 8, 128) == 1

    # _pad_inputs_if_needed:
    # Case 1: no padding needed (pad_input_size == 0 and pad_hidden_size == 0)
    x0 = jnp.ones((16, 128), dtype=jnp.bfloat16)
    idx0 = jnp.arange(16, dtype=jnp.int32)
    w0 = jnp.ones((16,), dtype=jnp.float32)
    m0 = jnp.ones((16,), dtype=jnp.bool_)
    px0, pidx0, pw0, pm0 = rgr._pad_inputs_if_needed(x0, idx0, w0, m0, 2, 1, 2, 8)
    assert px0 is x0 and pidx0 is idx0 and pw0 is w0 and pm0 is m0

    # Case 2: only hidden padding needed (pad_input_size == 0, pad_hidden_size > 0)
    x1 = jnp.ones((16, 64), dtype=jnp.bfloat16)
    px1, pidx1, pw1, pm1 = rgr._pad_inputs_if_needed(x1, idx0, w0, m0, 2, 1, 2, 8)
    assert px1.shape == (16, 128)
    assert pidx1 is idx0

    # Case 3: input padding needed (pad_input_size > 0)
    x2 = jnp.ones((10, 128), dtype=jnp.bfloat16)
    idx2 = jnp.arange(10, dtype=jnp.int32)
    w2 = jnp.ones((10,), dtype=jnp.float32)
    m2 = jnp.ones((10,), dtype=jnp.bool_)
    px2, pidx2, pw2, pm2 = rgr._pad_inputs_if_needed(x2, idx2, w2, m2, 2, 1, 2, 8)
    assert px2.shape == (16, 128)
    assert pidx2.shape == (16,)
    assert pw2.shape == (16,)
    assert pm2.shape == (16,)

    # _preprocess:
    # Branch 1: pad_partitions > 0 (num_row_partitions < num_simd_lanes)
    dst_idx, sorted_val, num_rows, mask = rgr._preprocess(
        jnp.arange(16, dtype=jnp.int32),
        jnp.array([True, False] * 8, dtype=jnp.bool_),
        reduce_group_size=2,
        num_row_partitions=2,
        num_simd_lanes=8,
    )
    assert dst_idx.shape == (16,)
    assert sorted_val.shape == (16,)
    assert num_rows.shape == (8,)
    assert mask.shape == (8,)

    # Branch 2: pad_partitions == 0 (num_row_partitions == num_simd_lanes)
    _, _, num_rows_exact, _ = rgr._preprocess(
        jnp.arange(16, dtype=jnp.int32),
        jnp.ones((16,), dtype=jnp.bool_),
        reduce_group_size=2,
        num_row_partitions=8,
        num_simd_lanes=8,
    )
    assert num_rows_exact.shape == (8,)


def test_ragged_gather_reduce_main_kernel_and_dispatch():
    fake_sc = SimpleNamespace(num_lanes=2, num_cores=1, num_subcores=2)
    fake_tpu_info = SimpleNamespace(
        sparse_core=fake_sc,
        num_lanes=4,
        vmem_capacity_bytes=1024,
    )

    def _fake_emit_pipeline(fn=None, **kwargs):
        if fn is None:
            return lambda f: _fake_emit_pipeline(f, **kwargs)
        return fn

    def _fake_pl_loop(lower, upper, step=1, init_carry=None):
        def _deco(fn):
            if init_carry is not None:
                carry = init_carry
                for i in range(0, int(upper), int(step)):
                    carry = fn(i, carry)
                return carry
            for i in range(0, max(1, int(upper)), int(step)):
                fn(i)
            return None

        return _deco

    def _make_refs(in_dtype):
        num_simd_lanes = 2
        col_size = 4
        return dict(
            num_rows_per_row_partition_ref=_FakeRef(jnp.array([2, 2], dtype=jnp.int32)),
            in_hbm_ref=_FakeRef(jnp.ones((4, 4), dtype=in_dtype)),
            indices_hbm_ref=_FakeRef(jnp.arange(4, dtype=jnp.int32)),
            dst_indices_hbm_ref=_FakeRef(jnp.array([0, 0, 1, 1], dtype=jnp.int32)),
            topk_weights_hbm_ref=_FakeRef(jnp.ones((4,), dtype=jnp.float32)),
            sorted_by_validity_hbm_ref=_FakeRef(jnp.arange(4, dtype=jnp.int32)),
            out_hbm_ref=_FakeRef(jnp.zeros((2, 4), dtype=jnp.float32)),
            num_rows_per_row_partition_vmem_ref=_FakeRef(
                jnp.array([2, 2], dtype=jnp.int32)
            ),
            out_vmem_ref=_FakeRef(
                jnp.zeros((num_simd_lanes, col_size), dtype=jnp.uint32)
            ),
            prev_iter_last_row_vmem_ref=_FakeRef(
                jnp.zeros((1, col_size), dtype=jnp.uint32)
            ),
            src_indices_vmem_ref=_FakeRef(jnp.array([0, 1], dtype=jnp.int32)),
            dst_indices_vmem_ref=_FakeRef(jnp.array([0, 0], dtype=jnp.int32)),
            topk_weights_vmem_ref=_FakeRef(
                jnp.ones((num_simd_lanes,), dtype=jnp.float32)
            ),
            sorted_by_validity_vmem_ref=_FakeRef(jnp.array([0, 1], dtype=jnp.int32)),
            sem_ref=_FakeRef(jnp.zeros((2,), dtype=jnp.int32)),
        )

    with (
        patch.object(rgr.pltpu, "get_tpu_info", return_value=fake_tpu_info),
        patch.object(rgr.jax.lax, "axis_size", return_value=2),
        patch.object(rgr.pltpu, "emit_pipeline", side_effect=_fake_emit_pipeline),
        patch.object(rgr.pl, "program_id", return_value=0),
        patch.object(rgr.pltpu, "make_async_copy", return_value=_NoopDMA()),
        patch.object(rgr.pl, "loop", side_effect=_fake_pl_loop),
        patch.object(
            rgr.plsc,
            "bitcast",
            side_effect=lambda x, dt: jnp.zeros(x.shape, dtype=dt),
        ),
    ):
        # 1. bfloat16 input
        rgr.main_kernel(
            **_make_refs(jnp.bfloat16),
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_row_partitions=2,
            num_column_partitions=1,
        )
        # 2. float32 input
        rgr.main_kernel(
            **_make_refs(jnp.float32),
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_row_partitions=2,
            num_column_partitions=1,
        )
        # 3. unsupported dtype raises ValueError
        with pytest.raises(ValueError, match="Not yet support extracting data"):
            rgr.main_kernel(
                **_make_refs(jnp.int32),
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_partitions=2,
                num_column_partitions=1,
            )

    # Dispatch tests on ragged_gather_reduce.__wrapped__:
    fn = rgr.ragged_gather_reduce.__wrapped__
    x = jnp.ones((16, 128), dtype=jnp.bfloat16)
    indices = jnp.arange(16, dtype=jnp.int32)
    weights_bf16 = jnp.ones((16,), dtype=jnp.bfloat16)
    weights_f32 = jnp.ones((16,), dtype=jnp.float32)
    valid_mask = jnp.array([True, False] * 8, dtype=jnp.bool_)

    # Branch A: sc_info is None -> _fallback_implementation
    with patch.object(
        rgr.pltpu,
        "get_tpu_info",
        return_value=SimpleNamespace(sparse_core=None),
    ):
        out_no_sc = fn(x, indices, weights_bf16, valid_mask, 2)
        assert out_no_sc.shape == (8, 128)
        assert out_no_sc.dtype == jnp.bfloat16

    # Branch B: small input heuristic -> _fallback_implementation
    with patch.object(
        rgr.pltpu,
        "get_tpu_info",
        return_value=SimpleNamespace(
            sparse_core=fake_sc,
            num_lanes=128,
            vmem_capacity_bytes=10**9,
        ),
    ):
        out_small = fn(x, indices, weights_f32, valid_mask, 2)
        np.testing.assert_allclose(out_small, out_no_sc)

    # Branch C: full SparseCore kernel path (both unpadded and padded shapes)
    sc_full = SimpleNamespace(num_lanes=8, num_cores=1, num_subcores=2)
    tpu_full = SimpleNamespace(
        sparse_core=sc_full,
        num_lanes=128,
        vmem_capacity_bytes=1,
    )
    with (
        patch.object(rgr.pltpu, "get_tpu_info", return_value=tpu_full),
        patch.object(
            rgr.plsc,
            "VectorSubcoreMesh",
            side_effect=lambda **kw: SimpleNamespace(**kw),
        ),
        patch.object(
            rgr.core_map_helper,
            "kernel",
            side_effect=lambda body, *, out_type, **kw: (
                lambda *args: jnp.ones(out_type.shape, dtype=out_type.dtype)
            ),
        ),
    ):
        # Unpadded (out.shape == (out_rows, hidden_size)) with float32 weights
        out_aligned = fn(x, indices, weights_f32, valid_mask, 2)
        assert out_aligned.shape == (8, 128)
        assert out_aligned.dtype == jnp.bfloat16

        # Padded (out.shape != (out_rows, hidden_size)) with bfloat16 weights
        x_unaligned = jnp.ones((10, 64), dtype=jnp.bfloat16)
        idx_unaligned = jnp.arange(10, dtype=jnp.int32)
        w_unaligned = jnp.ones((10,), dtype=jnp.bfloat16)
        m_unaligned = jnp.ones((10,), dtype=jnp.bool_)
        out_padded = fn(x_unaligned, idx_unaligned, w_unaligned, m_unaligned, 2)
        assert out_padded.shape == (5, 64)
        assert out_padded.dtype == jnp.bfloat16
