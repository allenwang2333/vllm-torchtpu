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

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# NOTE: When performing strided ldst, using pltpu.bitcast on vreg data should
# be avoided as it can trigger unintended relayout.


def load_large_to_compact(vmem_ref, dst_dtype: jnp.dtype | None = None) -> jax.Array:
    assert vmem_ref.ndim == 2

    row_size = vmem_ref.shape[0]
    src_dtype = vmem_ref.dtype
    packing = 4 // src_dtype.itemsize

    if dst_dtype is not None and dst_dtype != src_dtype:
        u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
        unpacked_list = []
        for packed_row in range(row_size // packing):
            packed = u32_vmem_ref[packed_row : packed_row + 1]
            for p in range(packing):
                unpacked_list.append(
                    pltpu.unpack_elementwise(
                        packed,
                        index=p,
                        packed_dtype=src_dtype,
                        unpacked_dtype=dst_dtype,
                    )
                )
        return jnp.stack(unpacked_list, axis=0)

    return jnp.stack(
        [
            vmem_ref[row_start : row_start + packing]
            for row_start in range(0, row_size, packing)
        ],
        axis=0,
    )
