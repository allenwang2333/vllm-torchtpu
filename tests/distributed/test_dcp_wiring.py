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
"""The DCP wiring that sits between the kernels and vLLM.

The kernels themselves are covered on hardware
(`tests/kernels/test_dsv4_streamindex_topk_dcp.py`,
`tests/layers/common/test_sparse_mla_attention.py::SparseMlaDcpTest`) and the
merge arithmetic in `test_dcp_merge.py`. The group accessors and the mesh
helper DCP shares with PCP are covered in `test_dcp_mesh.py`. What is left,
and what this file covers, is the ownership rule that the indexer's return leg
and the KV write have to agree on.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    cp_local_to_global
from vllm_torchtpu.layers.vllm.cp_indexer_merge import cp_global_to_local

# This is index arithmetic, so it needs no accelerator -- and it must not take
# one. These are eager `jnp` calls, so on a TPU agent they initialize the
# runtime in the *pytest parent*, which then holds the libtpu lockfile for the
# rest of the session; libtpu exposes no release API (see the note on
# `pytest_collection_modifyitems` in `tests/conftest.py`). Every later test
# that needs a fresh process -- `test_kv_scatter.py`'s `spawn`ed bodies, the
# vLLM engine cores under `tests/entrypoints/llm/` -- then fails to bootstrap.
# `tests/distributed` collects first and this file sorts ahead of
# `test_kv_scatter.py`, so it would be the first thing in the general suite to
# spring that trap. `cpu_test` keeps it on the CPU agent, where there is no
# lockfile to hold.
pytestmark = pytest.mark.cpu_test

# The indexer's return leg is an all-gather plus a per-rank owner filter, which
# is only exact because ownership is a *partition*: total, single-valued, and
# onto every rank. These pin that, because it is what replaced the
# capacity-sized all-to-all -- an all-to-all had to bet a static per-rank width
# and silently shed winners when the bet lost.
#
# `cp_local_to_global` is the only side of the map the kernel spells out, so
# ownership is read off it: rank `r` owns exactly the positions in its image.


def _owned_below(n, dcp_size, interleave_c):
    """Each rank's owned global positions below `n`, in local-index order.

    The map is monotonic, so the r-th list is what rank `r` sees at local
    indices `0, 1, 2, ...`.
    """
    locals_ = jnp.arange(n, dtype=jnp.int32)
    images = []
    for rank in range(dcp_size):
        g = np.asarray(
            cp_local_to_global(locals_, rank, dcp_size, interleave_c))
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
    """Over a whole interleave cycle the shards are exactly equal.

    Which is why mean per-rank occupancy of a `k`-wide top-k is `k / dcp_size`
    as an identity, not an estimate -- the spread around it comes only from
    where the winners cluster inside the cycle.
    """
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
    """The merge's rank-free inverse must undo the kernel's forward map.

    They sit on opposite sides of the op boundary -- one in JAX inside the
    kernel, one in torch on the return leg -- so nothing but this makes them
    agree, and the inverse never learns which rank it is inverting for.
    """
    n = 5 * dcp_size * interleave_c
    for img in _owned_below(n, dcp_size, interleave_c):
        back = cp_global_to_local(torch.from_numpy(img), dcp_size,
                                  interleave_c).numpy()
        np.testing.assert_array_equal(back, np.arange(len(img)))
