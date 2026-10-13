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

"""100% full-module line and branch coverage tests for compress_and_store/config.py."""

from __future__ import annotations

import importlib
import os

os.environ.setdefault("PJRT_DEVICE", "CPU")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import config


def test_module_reload_and_select_mode():
    """Reload module under coverage tracer and verify select_mode branches."""
    importlib.reload(config)
    assert config.select_mode(128, False) is config.Mode.CSA_INDEXER
    assert config.select_mode(128, True) is config.Mode.CSA_INDEXER
    assert config.select_mode(512, True) is config.Mode.CSA
    assert config.select_mode(512, False) is config.Mode.HCA


def test_page_and_layout_helpers():
    """Verify physical_page_size, state_host_page_size, and layout helper branches."""
    assert config.physical_page_size(config.Mode.CSA_INDEXER, 128, 4) == 8
    assert config.physical_page_size(config.Mode.CSA, 128, 4) == 32
    assert config.physical_page_size(config.Mode.HCA, 128, 4) == 64

    assert config.state_host_page_size(config.Mode.HCA, 128, 128) == 32
    assert config.state_host_page_size(config.Mode.CSA, 128, 4) == 32
    assert config.state_host_page_size(config.Mode.CSA_INDEXER, 128, 4) == 8

    assert config.last_dim_size(config.Mode.CSA_INDEXER) == 256
    assert config.last_dim_size(config.Mode.CSA) == 128
    assert config.last_dim_size(config.Mode.HCA) == 128

    assert config.row_size_bytes(config.Mode.CSA_INDEXER) == 1024
    assert config.row_size_bytes(config.Mode.CSA) == 512
    assert config.row_size_bytes(config.Mode.HCA) == 512

    assert config.state_width(512, True) == 1024
    assert config.state_width(512, False) == 512

    assert config.state_rows_per_token(config.Mode.CSA, 512, True) == 16
    assert config.state_rows_per_token(config.Mode.HCA, 512, False) == 8
    assert config.state_rows_per_token(config.Mode.CSA_INDEXER, 128, True) == 2


def test_state_block_size_and_capacity():
    """Verify _state_page_capacity and state_block_size across all modes."""
    config.state_block_size.cache_clear()
    cap_csa = config._state_page_capacity(config.Mode.CSA, 128)
    cap_idx = config._state_page_capacity(config.Mode.CSA_INDEXER, 128, 4, 128)
    cap_hca = config._state_page_capacity(config.Mode.HCA, 128)
    assert cap_csa == 2
    assert cap_idx == 4
    assert cap_hca == 4

    assert config.state_block_size(config.Mode.CSA, 128) == 2
    assert config.state_block_size(config.Mode.CSA_INDEXER, 128) == 2
    assert config.state_block_size(config.Mode.HCA, 128) == 4


def test_dimensions_and_configs_all_branches():
    """Verify Dimensions and Configs.make fast/override paths and all properties."""
    # 1. Mode.HCA default make (no overrides, tile_n=4, state_physical_page_size=None)
    cfg_hca = config.Configs.make(
        config.Mode.HCA,
        size_n=8,
        physical_page_size=64,
        state_block_size=8,
    )
    assert cfg_hca.dims.is_quantized is False
    assert cfg_hca.dims.has_rope is True
    assert cfg_hca.dims.nope_dtype == jnp.bfloat16
    assert cfg_hca.dims.rope_dtype == jnp.bfloat16
    assert cfg_hca.overlap_factor == 1
    assert cfg_hca.state_width == 512
    assert cfg_hca.head_tiles == 4
    assert cfg_hca.window == 128
    assert cfg_hca.nope_dim == 448
    assert cfg_hca.nope_store_dim == 512
    assert cfg_hca.half_rope == 32
    assert cfg_hca.rope_slot == 3
    assert cfg_hca.record_bytes == 1024
    assert cfg_hca.row_size_bytes == 512
    assert cfg_hca.record_rows == 2
    assert cfg_hca.hbm_pack == 4
    assert cfg_hca.last_dim_size == 128
    assert cfg_hca.cache_last_dims == (4, 128)
    assert cfg_hca.tokens_in_second_minor == 1
    assert cfg_hca.kv_stride == 2
    assert cfg_hca.physical_page_size == 64
    assert cfg_hca.state_physical_page_size == 64
    assert cfg_hca.state_rows_per_token == 8
    assert cfg_hca.field_rows == 4
    assert cfg_hca.state_page_capacity == 8
    assert cfg_hca.state_block_size == 8
    assert cfg_hca.kv_block_size == 32
    assert cfg_hca.rope_page_size == 16
    assert cfg_hca.pages_to_buffer_per_token == 17
    assert cfg_hca._tile_n == 4
    assert cfg_hca.window_shape() == (2, 4, 128, 4, 128)
    assert cfg_hca.window_bytes_shape() == (2, 4, 128, 4, 4, 128)

    # 2. Mode.HCA with overrides (exercises overrides + mode is Mode.HCA quant_block=0)
    cfg_hca_ovr = config.Configs.make(
        config.Mode.HCA,
        size_n=8,
        physical_page_size=64,
        state_block_size=8,
        state_physical_page_size=32,
        tile_n=8,
        quant_block=64,
        rope_head_dim=0,
    )
    assert cfg_hca_ovr.dims.quant_block == 0
    assert cfg_hca_ovr.dims.has_rope is False
    assert cfg_hca_ovr.tile_sizes.tile_n == 8
    assert cfg_hca_ovr.state_physical_page_size == 32

    # 3. Mode.CSA default make + Mode.CSA with overrides (non-HCA overrides branch)
    cfg_csa = config.Configs.make(
        config.Mode.CSA,
        size_n=8,
        physical_page_size=32,
        state_block_size=8,
        state_physical_page_size=32,
    )
    cfg_csa_ovr = config.Configs.make(
        config.Mode.CSA,
        size_n=8,
        physical_page_size=32,
        state_block_size=8,
        compress_ratio=4,
    )
    assert cfg_csa == cfg_csa_ovr
    assert cfg_csa.dims.is_quantized is True
    assert cfg_csa.dims.nope_dtype == jnp.uint8
    assert cfg_csa.dims.rope_dtype == jnp.uint8
    assert cfg_csa.overlap_factor == 2
    assert cfg_csa.state_width == 1024
    assert cfg_csa.window == 8
    assert cfg_csa.nope_store_dim == 448
    assert cfg_csa.record_bytes == 512
    assert cfg_csa.record_rows == 1
    assert cfg_csa.kv_stride == 1
    assert cfg_csa.state_rows_per_token == 16
    assert cfg_csa.pages_to_buffer_per_token == 2
    assert cfg_csa.window_shape() == (2, 4, 8, 4, 128)
    assert cfg_csa.window_bytes_shape() == (2, 4, 8, 4, 4, 128)
    assert cfg_csa.output_shape() == (4, 1, 4, 128)
    assert cfg_csa.page_buffer_shape() == (4, 2, 32, 4, 128)
    assert cfg_csa.rope_output_shape() == (4, 4, 128)
    assert cfg_csa.cos_sin_shape() == (4, 64)
    assert cfg_csa.cache_shape(10) == (10, 32, 4, 128)
    assert cfg_csa.state_cache_shape(10) == (10, 32, 4, 128)
    assert cfg_csa.rope_cache_shape(10) == (10, 8, 4, 128)

    # 4. Mode.CSA_INDEXER
    cfg_idx = config.Configs.make(
        config.Mode.CSA_INDEXER,
        size_n=4,
        physical_page_size=16,
        state_block_size=4,
    )
    assert cfg_idx.record_bytes == 256
    assert cfg_idx.record_rows == 1
    assert cfg_idx.row_size_bytes == 1024
    assert cfg_idx.last_dim_size == 256
    assert cfg_idx.cache_last_dims == (4, 256)
    assert cfg_idx.tokens_in_second_minor == 4
    assert cfg_idx.state_rows_per_token == 2
    assert cfg_idx.field_rows == 1
    assert cfg_idx.output_shape() == (4, 1, 4, 256)
    assert cfg_idx.page_buffer_shape() == (4, 3, 16, 4, 256)
    assert cfg_idx.cache_shape(10) == (10, 16, 4, 256)
    assert cfg_idx.state_cache_shape(10) == (10, 16, 4, 256)
