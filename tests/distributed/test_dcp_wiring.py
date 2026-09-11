# SPDX-License-Identifier: Apache-2.0
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

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    cp_global_to_local, cp_owner_rank)

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
# is only exact because `cp_owner_rank` is a *partition*: total, single-valued,
# and onto every rank. These pin that, because it is what replaced the
# capacity-sized all-to-all -- an all-to-all had to bet a static per-rank width
# and silently shed winners when the bet lost.


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_ownership_is_a_partition_of_the_positions(dcp_size, interleave_c):
    """Every position has exactly one owner, and every rank is an owner."""
    n = 5 * dcp_size * interleave_c + 3
    owners = np.asarray(
        cp_owner_rank(jnp.arange(n, dtype=jnp.int32), dcp_size, interleave_c))
    assert owners.min() >= 0 and owners.max() < dcp_size
    assert set(owners.tolist()) == set(range(dcp_size))


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_owner_counts_are_balanced_over_a_whole_cycle(dcp_size, interleave_c):
    """Over a whole interleave cycle the shards are exactly equal.

    Which is why mean per-rank occupancy of a `k`-wide top-k is `k / dcp_size`
    as an identity, not an estimate -- the spread around it comes only from
    where the winners cluster inside the cycle.
    """
    cycle = dcp_size * interleave_c
    owners = np.asarray(
        cp_owner_rank(jnp.arange(3 * cycle, dtype=jnp.int32), dcp_size,
                      interleave_c))
    counts = np.bincount(owners, minlength=dcp_size)
    assert (counts == 3 * interleave_c).all()


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("interleave_c", [1, 4, 64])
def test_local_indices_stay_inside_the_shard(dcp_size, interleave_c):
    """A rank is only ever handed an index into its own cache shard."""
    n = 5 * dcp_size * interleave_c
    gs = jnp.arange(n, dtype=jnp.int32)
    locals_ = np.asarray(cp_global_to_local(gs, dcp_size, interleave_c))
    assert locals_.min() >= 0
    assert locals_.max() < n // dcp_size
