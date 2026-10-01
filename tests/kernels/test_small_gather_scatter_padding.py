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
import numpy as np
import pytest

from vllm_torchtpu.kernels.collectives import small_gather_scatter as sgs

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("rows", [1, 4, 8, 16, 20, 24])
@pytest.mark.parametrize("rank_operand", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_padding_preserves_rank_blocks(monkeypatch, rows, rank_operand,
                                       permuted):
    """Check the collective boundary; remote DMA is covered by TPU tests."""
    with jax.default_device(jax.devices("cpu")[0]):
        n, hidden = 8, 128
        blocks = jnp.arange(n * rows * hidden, dtype=jnp.float32).reshape(
            n, rows, hidden).astype(jnp.bfloat16)
        x = blocks.reshape(n * rows, hidden)
        rank_of_core = np.array([0, 1, 6, 7, 2, 3, 4, 5], np.int32)
        mesh_core = 2
        own_rank = int(rank_of_core[mesh_core]) if permuted else mesh_core
        rank = jnp.array([mesh_core], jnp.int32) if rank_operand else None
        my_rank = jnp.array([own_rank], jnp.int32) if permuted else None
        table = jnp.array(np.argsort(rank_of_core)) if permuted else None
        calls = []

        def pallas_call(kernel, *, out_shape, scratch_shapes, **kwargs):
            padded_rows = out_shape.shape[0]
            assert padded_rows % 8 == 0
            assert scratch_shapes[0].shape == (n, padded_rows, hidden)

            def run(*args):
                padded = args[-1].reshape(n, padded_rows, hidden)
                np.testing.assert_array_equal(padded[:, :rows], blocks)
                np.testing.assert_array_equal(padded[:, rows:], 0)
                if rows % 8 == 0:
                    assert args[-1] is x
                if rank_operand:
                    np.testing.assert_array_equal(args[0], rank)
                if permuted:
                    np.testing.assert_array_equal(args[-3], my_rank)
                    np.testing.assert_array_equal(args[-2], table)
                calls.append(True)
                # Other cores contribute zero, making this core's input
                # block the expected sum for each destination rank.
                return padded[own_rank]

            return run

        monkeypatch.setattr(sgs.pl, "pallas_call", pallas_call)
        actual = sgs.small_reduce_scatter(x,
                                          num_cores=n,
                                          rank=rank,
                                          my_rank=my_rank,
                                          core_of_rank=table)
        assert len(calls) == 1
        np.testing.assert_array_equal(actual, blocks[own_rank])
