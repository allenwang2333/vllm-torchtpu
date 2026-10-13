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
"""Full unit and branch-coverage tests for `vllm_torchtpu.kernels.mla.scatter`."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

if "torch_tpu" not in sys.modules:
    _torch_tpu_mod = types.ModuleType("torch_tpu")
    _internal_mod = types.ModuleType("torch_tpu._internal")
    _pallas_mod = types.ModuleType("torch_tpu._internal.pallas")
    _pallas_mod.pallas = SimpleNamespace(  # type: ignore[attr-defined]
        JAX_TO_TORCH_DTYPE_MAP={
            jnp.dtype(jnp.uint8): torch.uint8,
            jnp.dtype(jnp.uint32): torch.uint32,
        }
    )
    _torch_tpu_mod._internal = _internal_mod  # type: ignore[attr-defined]
    _internal_mod.pallas = _pallas_mod  # type: ignore[attr-defined]
    sys.modules["torch_tpu"] = _torch_tpu_mod
    sys.modules["torch_tpu._internal"] = _internal_mod
    sys.modules["torch_tpu._internal.pallas"] = _pallas_mod

from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter, scatter_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
KV_PACKING = 4


def _cpu_scatter_rows(
    cache: jax.Array,
    src: jax.Array,
    dst_rows: jax.Array,
    name: str = "scatter_kv_cache",
) -> jax.Array:
    """Pure-JAX reference for `scatter_utils.scatter_rows` usable on CPU."""
    del name
    tc_tiled = cache.dtype == jnp.uint8
    row_width = (
        kv_cache_utils.WORD_BYTES * kv_cache_utils.TILE_LANE_BYTES
        if tc_tiled
        else src.shape[-1]
    )
    flat_cache = cache.reshape(-1, row_width)
    num_rows = flat_cache.shape[0]
    safe_dst = jnp.where(
        jnp.logical_and(dst_rows >= 0, dst_rows < num_rows),
        dst_rows,
        num_rows,
    )
    updated = flat_cache.at[safe_dst].set(src, mode="drop")
    return updated.reshape(cache.shape)


@pytest.mark.parametrize(
    ("cache_type", "layout", "head_dim"),
    [
        (KVCacheType.NOPE, KVCacheLayout.TENSORCORE, LKV_DIM),
        (KVCacheType.NOPE, KVCacheLayout.SPARSECORE, LKV_DIM),
        (KVCacheType.ROPE, KVCacheLayout.SPARSECORE, ROPE_DIM),
    ],
)
def test_scatter_eager_and_traced_all_layouts(
    cache_type: KVCacheType,
    layout: KVCacheLayout,
    head_dim: int,
) -> None:
    """Tests both eager JIT paths and outer-traced paths across all layouts."""
    num_pages = 8
    num_tokens = 12
    spec = SparseMLAKVCacheSpec.create(
        cache_type=cache_type,
        layout=layout,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        head_dim=head_dim,
        kv_packing=KV_PACKING,
    )
    cache = jnp.zeros(spec.shape, dtype=spec.jax_dtype)
    key = jax.random.PRNGKey(17)
    raw_u8 = jax.random.randint(key, (num_tokens, head_dim), 0, 256, jnp.uint8)
    values = jax.lax.bitcast_convert_type(raw_u8, jnp.float8_e4m3fn)

    seq_lens = jnp.array([6, 4], dtype=jnp.int32)
    block_tables = jnp.array([2, 5], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 6, 10], dtype=jnp.int32)

    page, slot = kv_cache_utils.get_page_and_slot(
        num_tokens, seq_lens, block_tables, query_start_loc, PAGE_SIZE
    )
    dst_rows = kv_cache_utils.get_dst_rows(
        num_tokens, seq_lens, block_tables, query_start_loc, PAGE_SIZE
    )
    ref = kv_cache_utils._scatter_rows(cache, spec, values, page, slot)

    with mock.patch.object(scatter_utils, "scatter_rows", _cpu_scatter_rows):
        out_eager_addr = scatter.scatter(
            cache,
            values,
            spec,
            seq_lens=seq_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
        )
        out_eager_addr_repeat = scatter.scatter(
            cache,
            values,
            spec,
            seq_lens=seq_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
        )
        out_eager_dst = scatter.scatter(
            cache,
            values,
            spec,
            dst_rows=dst_rows,
        )
        out_eager_dst_repeat = scatter.scatter(
            cache,
            values,
            spec,
            dst_rows=dst_rows,
        )

        traced_addr_fn = jax.jit(
            lambda c, v, sl, bt, qsl: scatter.scatter(
                c,
                v,
                spec,
                seq_lens=sl,
                block_tables=bt,
                query_start_loc=qsl,
            )
        )
        out_traced_addr = traced_addr_fn(
            cache, values, seq_lens, block_tables, query_start_loc
        )

        traced_dst_fn = jax.jit(
            lambda c, v, dr: scatter.scatter(c, v, spec, dst_rows=dr)
        )
        out_traced_dst = traced_dst_fn(cache, values, dst_rows)

    for candidate in (
        out_eager_addr,
        out_eager_addr_repeat,
        out_eager_dst,
        out_eager_dst_repeat,
        out_traced_addr,
        out_traced_dst,
    ):
        np.testing.assert_array_equal(np.asarray(candidate), np.asarray(ref))


def test_scatter_spec_meta_cache_and_contracts() -> None:
    """Verifies `_spec_meta` caching and input contract assertions."""
    scatter._spec_meta.cache_clear()
    spec = SparseMLAKVCacheSpec.create(
        cache_type=KVCacheType.NOPE,
        layout=KVCacheLayout.SPARSECORE,
        num_pages=4,
        page_size=PAGE_SIZE,
        head_dim=LKV_DIM,
        kv_packing=KV_PACKING,
    )
    meta1 = scatter._spec_meta(spec)
    meta2 = scatter._spec_meta(spec)
    assert meta1 == meta2
    assert scatter._spec_meta.cache_info().hits >= 1
    assert meta1[-1] == "scatter_nope_sparsecore_cache"

    cache = jnp.zeros(spec.shape, dtype=spec.jax_dtype)
    values = jnp.zeros((4, LKV_DIM), dtype=jnp.float8_e4m3fn)

    with pytest.raises(AssertionError, match="does not match spec"):
        scatter.scatter(
            jnp.zeros(spec.shape, dtype=jnp.uint8),
            values,
            spec,
            dst_rows=jnp.zeros((4,), dtype=jnp.int32),
        )

    with pytest.raises(AssertionError, match="does not match spec"):
        scatter.scatter(
            jnp.zeros((2, PAGE_SIZE, LKV_DIM // 4), dtype=spec.jax_dtype),
            values,
            spec,
            dst_rows=jnp.zeros((4,), dtype=jnp.int32),
        )

    with pytest.raises(AssertionError, match="pass dst_rows"):
        scatter.scatter(cache, values, spec)

    with pytest.raises(AssertionError, match="pass dst_rows"):
        scatter.scatter(
            cache,
            values,
            spec,
            seq_lens=jnp.array([4], dtype=jnp.int32),
            block_tables=jnp.array([0], dtype=jnp.int32),
        )
