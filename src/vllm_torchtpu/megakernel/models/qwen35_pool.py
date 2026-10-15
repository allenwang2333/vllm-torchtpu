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
"""Qwen3.5 paged KV raw page layout utilities (`kv_to_raw_page`, `raw_page_to_kv`)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..ops import FP8


def kv_to_raw_page(k_page: Any, v_page: Any) -> jax.Array:
    """Pack `(page_size, HD)` K and V into `(2, HD // 4, 4, page_size) fp8`."""
    ps, hd = k_page.shape[0], k_page.shape[-1]
    assert hd % 4 == 0, hd
    h4 = hd // 4
    k_fp8 = (
        jnp.clip(k_page, -448.0, 448.0).astype(FP8) if k_page.dtype != FP8 else k_page
    )
    v_fp8 = (
        jnp.clip(v_page, -448.0, 448.0).astype(FP8) if v_page.dtype != FP8 else v_page
    )
    k_pk = jnp.transpose(k_fp8.reshape(ps, h4, 4), (1, 2, 0))
    v_pk = jnp.transpose(v_fp8.reshape(ps, h4, 4), (1, 2, 0))
    return jnp.stack([k_pk, v_pk], axis=0)


def raw_page_to_kv(raw_page: Any) -> tuple[jax.Array, jax.Array]:
    """Unpack `(2, HD // 4, 4, page_size) fp8` into `(k_page, v_page)`."""
    ps = raw_page.shape[-1]
    hd = raw_page.shape[1] * raw_page.shape[2]
    k_page = jnp.transpose(raw_page[0], (2, 0, 1)).reshape(ps, hd)
    v_page = jnp.transpose(raw_page[1], (2, 0, 1)).reshape(ps, hd)
    return k_page, v_page
