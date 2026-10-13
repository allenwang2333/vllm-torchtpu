# Copyright 2026 Google LLC
"""Full 100% statement and branch coverage tests for kernels.typed_ldst."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import typed_ldst

pytestmark = pytest.mark.cpu_test


class _MockRef:
    def __init__(
        self, arr: Any, sink: dict[str, Any] | None = None, dtype: Any = None
    ) -> None:
        self.arr = arr
        self.dtype = arr.dtype if dtype is None else dtype
        self.shape = arr.shape
        self.sink = {} if sink is None else sink

    def bitcast(self, target_dtype: Any) -> "_MockRef":
        return _MockRef(self.arr.view(target_dtype), self.sink)

    def reshape(self, *new_shape: int) -> "_MockRef":
        return _MockRef(self.arr.reshape(*new_shape), self.sink, self.dtype)

    def __getitem__(self, idx: Any) -> Any:
        return self.arr[idx]

    def __setitem__(self, idx: Any, val: Any) -> None:
        self.arr = val
        self.sink["written"] = val


def test_typed_ldst_all_branches() -> None:
    typed_ldst._DTYPE_CACHE.clear()

    # 1a. 2D identical dtype object (ref_dt is view_dtype), lane_split=1
    ref_same_obj = _MockRef(
        np.arange(16, dtype=np.float32).reshape(2, 8), dtype=jnp.float32
    )
    out_same_obj = typed_ldst.load_typed(
        ref_same_obj, view_dtype=jnp.float32, lane_split=1
    )
    assert out_same_obj.shape == (2, 8)

    # 1b. 2D equivalent dtype (np.dtype('float32') vs jnp.float32), lane_split=2
    ref_2d = _MockRef(np.arange(16, dtype=np.float32).reshape(2, 8))
    out_split2 = typed_ldst.load_typed(ref_2d, view_dtype=jnp.float32, lane_split=2)
    assert out_split2.shape == (4, 4)

    # 2. 3D block ref with bitcast (uint8 -> uint16) and lane_split=1
    ref_3d_u8 = _MockRef(np.arange(32, dtype=np.uint8).reshape(2, 2, 8))
    out_3d_u16 = typed_ldst.load_typed(ref_3d_u8, view_dtype=jnp.uint16, lane_split=1)
    assert out_3d_u16.shape == (4, 4)

    # 3. store_typed with matching shape (bypasses values.reshape)
    store_ref_1 = _MockRef(np.zeros((2, 8), dtype=np.float32))
    vals_matching = np.ones((2, 8), dtype=np.float32)
    typed_ldst.store_typed(store_ref_1, vals_matching, lane_split=1)
    assert np.array_equal(store_ref_1.sink["written"], vals_matching)

    # 4. store_typed with non-matching shape (runs values.reshape) and lane_split=2
    store_ref_2 = _MockRef(np.zeros((2, 8), dtype=np.float32))
    vals_1d = np.arange(16, dtype=np.float32)
    typed_ldst.store_typed(store_ref_2, vals_1d, lane_split=2)
    assert store_ref_2.sink["written"].shape == (4, 4)
