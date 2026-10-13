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
"""100% whole-module coverage tests for kernels/gdn/head_geometry.py."""

from __future__ import annotations

import importlib

import pytest

from vllm_torchtpu.kernels.gdn import head_geometry as hg_mod
from vllm_torchtpu.kernels.gdn.head_geometry import (
    GdnHeadGeometry,
    derive_gdn_head_geometry,
)


def test_head_geometry_all_branches() -> None:
    importlib.reload(hg_mod)
    derive = hg_mod.derive_gdn_head_geometry
    derive.cache_clear()
    assert GdnHeadGeometry is not None
    assert derive_gdn_head_geometry is not None

    # 1. Sharded Q/K heads (cold + warm cache hit)
    g_sharded = derive(16, 32, 4)
    assert derive(16, 32, 4) is g_sharded
    assert g_sharded.local_num_kq_heads == 4
    assert g_sharded.local_num_v_heads == 8
    assert g_sharded.kq_replication_factor == 1
    assert g_sharded.kq_shard_index(0) == 0
    assert g_sharded.kq_shard_index(3) == 3

    # 2. Replicated Q/K heads (cold + warm cache hit)
    g_rep = derive(2, 16, 8)
    assert derive(2, 16, 8) is g_rep
    assert g_rep.local_num_kq_heads == 1
    assert g_rep.local_num_v_heads == 2
    assert g_rep.kq_replication_factor == 4
    assert g_rep.kq_shard_index(0) == 0
    assert g_rep.kq_shard_index(7) == 1

    # 3. kq_shard_index out-of-bounds
    with pytest.raises(ValueError, match="must be in"):
        g_rep.kq_shard_index(-1)
    with pytest.raises(ValueError, match="must be in"):
        g_rep.kq_shard_index(8)

    # 4. local_conv_dim and local_state_shapes (cold + warm + invalid + axes)
    assert g_sharded.local_conv_dim(128, 128) == 2 * 4 * 128 + 8 * 128
    assert g_sharded.local_conv_dim(128, 128) == 2 * 4 * 128 + 8 * 128
    with pytest.raises(ValueError, match="must be positive"):
        g_sharded.local_conv_dim(0, 128)
    with pytest.raises(ValueError, match="must be positive"):
        g_sharded.local_conv_dim(128, -1)

    s_def = g_sharded.local_state_shapes(((4, 8192), (32, 128, 128)), 128, 128)
    assert s_def == ((4, 2048), (8, 128, 128))
    assert g_sharded.local_state_shapes(((4, 8192), (32, 128, 128)), 128, 128) is s_def

    s_axis0 = g_sharded.local_state_shapes(
        ((8192, 4), (32, 128, 128)), 128, 128, conv_dim_axis=0
    )
    assert s_axis0 == ((2048, 4), (8, 128, 128))

    with pytest.raises(ValueError, match="must be positive"):
        g_sharded.local_state_shapes(((4, 8192), (32, 128, 128)), 0, 128)
    with pytest.raises(ValueError, match="must be positive"):
        g_sharded.local_state_shapes(((4, 8192), (32, 128, 128)), 128, 0)

    # 5. derive_gdn_head_geometry validation error branches
    with pytest.raises(ValueError, match="must be positive"):
        derive(0, 16, 4)
    with pytest.raises(ValueError, match="must be positive"):
        derive(4, 0, 4)
    with pytest.raises(ValueError, match="must be positive"):
        derive(4, 16, 0)

    with pytest.raises(ValueError, match="num_v_heads=10 must be divisible"):
        derive(4, 10, 4)

    with pytest.raises(ValueError, match="when Q/K heads are sharded"):
        derive(6, 16, 4)

    with pytest.raises(ValueError, match="when Q/K heads are replicated"):
        derive(3, 16, 8)

    with pytest.raises(ValueError, match="must own an integer number of V heads"):
        derive(12, 16, 4)
