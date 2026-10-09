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
"""TPU_DCP_RANK_OPERAND: the indexer K-cache scatter with the rank as operand.

`_cp_scatter_indexer_k` decides which rows a rank owns from its DCP rank. With
`rank=` it reads the rank from a sharded arange instead of an `all_to_all`;
the cache it writes must be bit-identical, and the program must lose its
`all_to_all`. Pure jnp -- runs on any backend with enough devices (CPU with
`--xla_force_host_platform_device_count`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

P = jax.sharding.PartitionSpec

NUM_BLOCKS = 8  # per rank
PAGE = 64
KV_PACKING = 4
LKV_DIM = 256
HEAD_DIM = 128
NUM_SEQS = 4
PAGES_PER_SEQ = 2


def _case(dcp_size, interleave, seed):
    rng = np.random.default_rng(seed)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]), (DCP_AXIS_NAME,))
    sharded = jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME))
    cache = rng.integers(
        0, 256, (dcp_size * NUM_BLOCKS, PAGE // KV_PACKING, KV_PACKING, LKV_DIM)
    ).astype(np.uint8)
    q_lens = np.array([4, 4, 3, 0], np.int32)
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    num_tokens = 16
    k_packed = rng.integers(0, 256, (num_tokens, HEAD_DIM + 1)).astype(np.uint8)
    seq_lens = np.zeros(NUM_SEQS, np.int32)
    for s in range(NUM_SEQS):
        if q_lens[s]:
            seq_lens[s] = rng.integers(q_lens[s], PAGES_PER_SEQ * PAGE * dcp_size)
    page_indices = rng.permutation(NUM_BLOCKS)[: NUM_SEQS * PAGES_PER_SEQ].astype(
        np.int32
    )
    args = (
        jax.device_put(jnp.asarray(cache), sharded),
        jnp.asarray(k_packed),
        jnp.asarray(seq_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu),
    )
    kwargs = dict(mesh=mesh, cp_size=dcp_size, interleave_size=interleave)
    rank = jax.device_put(jnp.arange(dcp_size, dtype=jnp.int32), sharded)
    return args, kwargs, rank


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 8])
def test_scatter_rank_operand_is_bit_identical(dcp_size, interleave):
    if len(jax.devices()) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(jax.devices())}")
    args, kwargs, rank = _case(dcp_size, interleave, seed=dcp_size + interleave)

    scatter = mla_attention_op._cp_scatter_indexer_k
    ref_fn = jax.jit(lambda *a: scatter(*a, **kwargs))
    # The rank is a real operand of the program, as the torch op passes it.
    op_fn = jax.jit(lambda *a: scatter(*a[:-1], **kwargs, rank=a[-1]))

    want = np.asarray(jax.block_until_ready(ref_fn(*args)))
    got = np.asarray(jax.block_until_ready(op_fn(*args, rank)))
    np.testing.assert_array_equal(got, want)
    # The step actually wrote something on some rank.
    assert not np.array_equal(want, np.asarray(args[0]))
    assert "all_to_all" in ref_fn.lower(*args).as_text()
    assert "all_to_all" not in op_fn.lower(*args, rank).as_text()
