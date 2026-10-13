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

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.causal_conv1d import strided_ldst


class _FakeVmemRef:
    __slots__ = ("arr", "dtype", "shape")

    def __init__(self, arr: jnp.ndarray) -> None:
        self.arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype

    def bitcast(self, target_dtype) -> "_FakeVmemRef":
        src_bits = jnp.dtype(self.dtype).itemsize * 8
        dst_bits = jnp.dtype(target_dtype).itemsize * 8
        pack = dst_bits // src_bits
        new_rows = max(1, self.shape[0] // pack)
        return _FakeVmemRef(jnp.zeros((new_rows, *self.shape[1:]), dtype=target_dtype))

    def __getitem__(self, idx):
        return self.arr[idx]


def test_strided_ldst_all_branches(monkeypatch) -> None:
    """Verify 100% statement and branch coverage of causal_conv1d.strided_ldst."""
    mod = importlib.reload(strided_ldst)

    # 1. Invalid rank assertion (len(shape) != 2).
    bad_ref = _FakeVmemRef(jnp.zeros((2, 2, 2), dtype=jnp.float32))
    with pytest.raises(AssertionError):
        mod.load_large_to_compact(bad_ref)

    # 2. Same-dtype paths: dst_dtype=None, dst_dtype is src_dtype, and
    #    dst_dtype == src_dtype via _SAME_DTYPE_PAIRS (jnp.float32 vs dtype('float32')).
    ref_f32 = _FakeVmemRef(jnp.arange(16, dtype=jnp.float32).reshape(4, 4))
    out_none = mod.load_large_to_compact(ref_f32, dst_dtype=None)
    assert out_none.shape == (4, 1, 4)
    np.testing.assert_allclose(out_none[:, 0, :], ref_f32.arr)

    out_same_obj = mod.load_large_to_compact(ref_f32, dst_dtype=ref_f32.dtype)
    np.testing.assert_allclose(out_same_obj, out_none)

    out_same_type = mod.load_large_to_compact(ref_f32, dst_dtype=jnp.float32)
    np.testing.assert_allclose(out_same_type, out_none)

    ref_bf16 = _FakeVmemRef(jnp.arange(32, dtype=jnp.bfloat16).reshape(4, 8))
    out_bf16_same = mod.load_large_to_compact(ref_bf16, dst_dtype=jnp.bfloat16)
    assert out_bf16_same.shape == (2, 2, 8)

    # 3. Unpack path: bf16 -> f32 (packing=2) and int8 -> f32 (packing=4).
    monkeypatch.setattr(
        mod.pltpu,
        "unpack_elementwise",
        lambda packed, index, packed_dtype, unpacked_dtype: jnp.full(
            packed.shape, float(index + 1), dtype=unpacked_dtype
        ),
    )
    out_bf16_unpacked = mod.load_large_to_compact(ref_bf16, dst_dtype=jnp.float32)
    assert out_bf16_unpacked.shape == (4, 1, 8)
    assert out_bf16_unpacked.dtype == jnp.float32
    np.testing.assert_allclose(out_bf16_unpacked[0, 0], 1.0)
    np.testing.assert_allclose(out_bf16_unpacked[1, 0], 2.0)

    ref_i8 = _FakeVmemRef(jnp.zeros((8, 4), dtype=jnp.int8))
    out_i8_unpacked = mod.load_large_to_compact(ref_i8, dst_dtype=jnp.float32)
    assert out_i8_unpacked.shape == (8, 1, 4)
    assert out_i8_unpacked.dtype == jnp.float32
