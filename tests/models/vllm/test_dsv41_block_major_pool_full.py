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

import sys
import types
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.block_major_pool as bmp
from vllm_torchtpu import envs
from vllm_torchtpu.block_major_pool import (
    BlockMajorPoolLayout,
    flat_kernel_ids,
    flat_manager_ids,
    kernel_view,
    unified_block_major_enabled,
)


def test_block_major_pool_full_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_platforms = types.ModuleType("vllm_torchtpu.platforms")
    fake_utils = types.ModuleType("vllm_torchtpu.platforms.tpu_block_size_utils")
    state = {"unified": True}
    fake_utils.unified_kv_layout_enabled = lambda _cfg: state["unified"]  # type: ignore[attr-defined]
    fake_platforms.tpu_block_size_utils = fake_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.platforms", fake_platforms)
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu.platforms.tpu_block_size_utils", fake_utils
    )
    monkeypatch.setattr(bmp, "_tpu_block_size_utils", None)

    cfg_hybrid = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True))
    cfg_dense = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=False))

    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", False)
    assert unified_block_major_enabled(cfg_hybrid) is False  # type: ignore[arg-type]

    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", True)
    assert unified_block_major_enabled(cfg_dense) is False  # type: ignore[arg-type]
    assert bmp._tpu_block_size_utils is None

    assert unified_block_major_enabled(cfg_hybrid) is True  # type: ignore[arg-type]
    assert bmp._tpu_block_size_utils is fake_utils

    state["unified"] = False
    assert unified_block_major_enabled(cfg_hybrid) is False  # type: ignore[arg-type]

    layout = BlockMajorPoolLayout(
        num_blocks=8,
        num_pools=3,
        split=4,
        pool_index_by_layer={"layer.0": 0, "layer.1": 1, "layer.2": 2},
    )
    assert layout.rows_per_block == 12
    assert "rows_per_block" not in repr(layout)
    assert layout == BlockMajorPoolLayout(
        num_blocks=8,
        num_pools=3,
        split=4,
        pool_index_by_layer={"layer.0": 0, "layer.1": 1, "layer.2": 2},
    )

    t_pool = torch.arange(2 * 3 * 4 * 5, dtype=torch.int32).reshape(2, 3, 4, 5)
    t_view = kernel_view(t_pool)
    assert t_view.shape == (6, 4, 5)
    assert t_view.data_ptr() == t_pool.data_ptr()

    j_pool = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.int32).reshape(2, 3, 4, 5)
    j_view = kernel_view(j_pool)
    assert j_view.shape == (6, 4, 5)
    np.testing.assert_array_equal(np.asarray(j_view), np.asarray(t_view))

    n_pool = np.arange(2 * 3 * 4 * 5, dtype=np.int32).reshape(2, 3, 4, 5)
    n_view = kernel_view(n_pool)
    assert n_view.shape == (6, 4, 5)
    np.testing.assert_array_equal(n_view, np.asarray(t_view))

    ids = torch.tensor([0, 1, 5], dtype=torch.int32)
    ids_orig = ids.clone()
    assert flat_manager_ids(ids, 0, 1) is ids
    assert flat_manager_ids(ids, 0, 3).tolist() == [0, 3, 15]
    assert flat_manager_ids(ids, 2, 3).tolist() == [2, 5, 17]
    assert torch.equal(ids, ids_orig)

    assert flat_kernel_ids(ids, 0, 1, 4) is ids
    assert flat_kernel_ids(ids, 0, 3, 1).tolist() == [0, 3, 15]
    assert flat_kernel_ids(ids, 2, 3, 1).tolist() == [2, 5, 17]
    assert torch.equal(ids, ids_orig)

    full_ids = torch.arange(0, 24, dtype=torch.int32)
    full_ids_orig = full_ids.clone()
    for split in (2, 3, 4, 6, 8):
        for num_pools in (2, 3, 5):
            for p in range(num_pools):
                got = flat_kernel_ids(full_ids, p, num_pools, split)
                expected = (
                    flat_manager_ids(full_ids // split, p, num_pools) * split
                    + full_ids % split
                )
                assert torch.equal(got, expected)
                assert torch.equal(full_ids, full_ids_orig)
