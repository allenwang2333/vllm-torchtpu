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
"""Typed views of raw byte blocks inside Pallas TPU kernels.

A block ref holds bytes in the layout of its owning buffer (e.g. one
kernel block of the unified KV pool). Consumers that pack a different
element type into those bytes access them through a typed view: Mosaic
``ref.bitcast`` (which rescales the second-minor dim by the element-size
ratio) plus an optional narrowing reshape, both applied to the *ref*
before the load. Reshaping the ref keeps every access one whole vector
register; splitting lanes after the load costs a lane-crossing relayout
instead (measured on v7: 366 bundles / 448 vrot for the post-load split
against 87 / 0 for the ref-level reshape, on a 256 KiB pool block).
The pool gather/scatter kernels and the GDN V3 state seam share these
helpers so the bytes they exchange are identical.
"""

from typing import Any

import jax
import jax.numpy as jnp

_DTYPE_CACHE: dict[Any, Any] = {}


def _get_dtype(dt: Any) -> Any:
    res = jnp.dtype(dt)
    _DTYPE_CACHE[dt] = res
    _DTYPE_CACHE[res] = res
    return res


def _typed_view(
    block_ref,
    view_dtype,
    lane_split: int,
    _cache: dict[Any, Any] = _DTYPE_CACHE,
):
    """Raw block ref as a ``(rows, lanes // lane_split)`` view_dtype ref."""
    ref_dt = block_ref.dtype
    if ref_dt is not view_dtype:
        target_dtype = _cache.get(view_dtype) or _get_dtype(view_dtype)
        ref_dtype = _cache.get(ref_dt) or _get_dtype(ref_dt)
        if ref_dtype != target_dtype:
            block_ref = block_ref.bitcast(target_dtype)
    shape = block_ref.shape
    if len(shape) == 2:
        if lane_split == 1:
            return block_ref
        return block_ref.reshape(shape[0] * lane_split, shape[1] // lane_split)
    return block_ref.reshape(-1, shape[-1] // lane_split)


def load_typed(block_ref, *, view_dtype, lane_split: int = 1) -> jax.Array:
    """Reads a raw block ref as ``(rows, lanes // lane_split)`` view_dtype."""
    return _typed_view(block_ref, view_dtype, lane_split)[...]


def store_typed(block_ref, values: jax.Array, *, lane_split: int = 1) -> None:
    """Inverse of ``load_typed``: stores ``values`` through a typed view of
    the raw block ref, which is fully written."""
    ref = _typed_view(block_ref, values.dtype, lane_split)
    ref_shape = ref.shape
    ref[...] = values if values.shape == ref_shape else values.reshape(ref_shape)
