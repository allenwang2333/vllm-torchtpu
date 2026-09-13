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
"""The ownership rule the indexer's return leg and the KV write must share."""

import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    cp_local_to_global
from vllm_torchtpu.layers.vllm.cp_indexer_merge import cp_global_to_local

# numpy, not jnp: an eager jnp call claims the TPU in the pytest parent and
# breaks every later spawn-based test in the session.
pytestmark = pytest.mark.cpu_test


def _owned_below(n, dcp_size, interleave_c):
    """Each rank's owned global positions below `n`, in local-index order."""
    locals_ = np.arange(n, dtype=np.int32)
    images = []
    for rank in range(dcp_size):
        g = cp_local_to_global(locals_, rank, dcp_size, interleave_c)
        images.append(g[g < n])
    return images


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_ownership_is_a_partition_of_the_positions(dcp_size, interleave_c):
    """Every position has exactly one owner, and every rank is an owner."""
    n = 5 * dcp_size * interleave_c + 3
    images = _owned_below(n, dcp_size, interleave_c)
    assert all(len(img) for img in images)
    np.testing.assert_array_equal(np.sort(np.concatenate(images)),
                                  np.arange(n))


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_owner_counts_are_balanced_over_a_whole_cycle(dcp_size, interleave_c):
    """Over a whole interleave cycle the shards are exactly equal."""
    cycle = dcp_size * interleave_c
    images = _owned_below(3 * cycle, dcp_size, interleave_c)
    counts = [len(img) for img in images]
    assert counts == [3 * interleave_c] * dcp_size


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_local_indices_stay_inside_the_shard(dcp_size, interleave_c):
    """A rank is only ever handed an index into its own cache shard."""
    n = 5 * dcp_size * interleave_c
    gs = torch.arange(n, dtype=torch.int32)
    locals_ = cp_global_to_local(gs, dcp_size, interleave_c).numpy()
    assert locals_.min() >= 0
    assert locals_.max() < n // dcp_size


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_global_to_local_inverts_the_kernel_map(dcp_size, interleave_c):
    """The merge's rank-free inverse must undo the kernel's forward map."""
    n = 5 * dcp_size * interleave_c
    for img in _owned_below(n, dcp_size, interleave_c):
        back = cp_global_to_local(torch.from_numpy(img), dcp_size,
                                  interleave_c).numpy()
        np.testing.assert_array_equal(back, np.arange(len(img)))
