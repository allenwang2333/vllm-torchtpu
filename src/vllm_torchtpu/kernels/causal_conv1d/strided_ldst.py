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

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# NOTE: When performing strided ldst, using pltpu.bitcast on vreg data should
# be avoided as it can trigger unintended relayout.

_UINT32 = jnp.uint32
_jnp_stack = jnp.stack
_PACK_RANGES: dict[int, tuple[int, ...]] = {
    1: (0,),
    2: (0, 1),
    4: (0, 1, 2, 3),
}
_COMMON_DTYPES = (
    jnp.float32,
    jnp.bfloat16,
    jnp.float16,
    jnp.int32,
    jnp.uint32,
    jnp.int16,
    jnp.uint16,
    jnp.int8,
    jnp.uint8,
)
_PACKING_BY_DTYPE: dict[Any, int] = {}
_SAME_DTYPE_PAIRS: set[tuple[Any, Any]] = set()
for _dt in _COMMON_DTYPES:
    _np_dt = jnp.dtype(_dt)
    _pack = 4 // _np_dt.itemsize
    _PACKING_BY_DTYPE[_dt] = _pack
    _PACKING_BY_DTYPE[_np_dt] = _pack
    _SAME_DTYPE_PAIRS.add((_dt, _np_dt))
    _SAME_DTYPE_PAIRS.add((_np_dt, _dt))
    _SAME_DTYPE_PAIRS.add((_np_dt, _np_dt))
    _SAME_DTYPE_PAIRS.add((_dt, _dt))


def load_large_to_compact(vmem_ref, dst_dtype: jnp.dtype | None = None) -> jax.Array:
    shape = vmem_ref.shape
    assert len(shape) == 2

    row_size = shape[0]
    src_dtype = vmem_ref.dtype
    packing = _PACKING_BY_DTYPE.get(src_dtype) or (4 // src_dtype.itemsize)

    if (
        dst_dtype is not None
        and dst_dtype is not src_dtype
        and (dst_dtype, src_dtype) not in _SAME_DTYPE_PAIRS
        and dst_dtype != src_dtype
    ):
        u32_vmem_ref = vmem_ref.bitcast(_UINT32)
        unpack_fn = pltpu.unpack_elementwise
        p_indices = _PACK_RANGES.get(packing) or tuple(range(packing))
        return _jnp_stack(
            [
                unpack_fn(packed, p, src_dtype, dst_dtype)
                for packed_row in range(row_size // packing)
                for packed in (u32_vmem_ref[packed_row : packed_row + 1],)
                for p in p_indices
            ],
            0,
        )

    return _jnp_stack(
        [
            vmem_ref[row_start : row_start + packing]
            for row_start in range(0, row_size, packing)
        ],
        0,
    )
