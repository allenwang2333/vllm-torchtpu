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
"""The indexer compressor writes SEQ_ALONG_LANE records the indexer reads.

The packed-row (HEAD_ALONG_SUBLANE) path is the reference: the same steps are
run into a packed-row cache and a SEQ_ALONG_LANE cache, and after every step
the SEQ_ALONG_LANE cache must be the packed-row cache transposed record by
record, with the f32 compressor state byte-identical. A last check runs the
streamindex top-k kernel on both caches.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import indexer_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compressor_v1
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    convert_cache_to_seq_along_lane,
    streamindex_topk,
)

pytestmark = pytest.mark.tpu_test

HIDDEN = 512
HEAD_DIM = 128
RATIO = 4
ROPE_DIM = 64
STATE_BLOCK = 16  # token states per state page, floored to the CSA peer's
PAGE = indexer_cache_layout.SEQ_ALONG_LANE_PAGE_SIZE  # compressed tokens/page
STATE_ROWS = STATE_BLOCK * 2  # two (4, 256) rows per token state

# Request 0 is short, request 1 long; decode requests lead the batch.
# Step: per request, (first position, new tokens); 0 tokens = idle.
STEPS = [
    [(0, 37), (0, 600)],  # two prefills; request 1 crosses a 128-lane tile
    [(37, 1), (600, 700)],  # decode + prefill crossing into a second page
    [(38, 1), (1300, 1)],  # decode-only steps, one boundary each in turn
    [(39, 1), (1301, 1)],
    [(40, 1), (1302, 1)],
    [(41, 1), (1303, 1)],
]
LAST_POS = [41, 1303]


def _tables():
    """Disjoint state and record pages: vLLM never gives one block to two
    cache groups. Page 0 is the null block."""
    state_blocks = [LAST_POS[r] // STATE_BLOCK + 1 for r in range(2)]
    kv_blocks = [LAST_POS[r] // RATIO // PAGE + 1 for r in range(2)]
    next_page = 1
    state = np.zeros((2, max(state_blocks)), np.int32)
    for r in range(2):
        state[r, : state_blocks[r]] = np.arange(next_page, next_page + state_blocks[r])
        next_page += state_blocks[r]
    kv = np.zeros((2, max(kv_blocks)), np.int32)
    for r in range(2):
        kv[r, : kv_blocks[r]] = np.arange(next_page, next_page + kv_blocks[r])
        next_page += kv_blocks[r]
    return state, kv, next_page + 1


def _step_inputs(step, rng, pad_to=8):
    positions, qsl, num_decodes = [], [0], 0
    for first, n in step:
        positions.extend(range(first, first + n))
        qsl.append(qsl[-1] + n)
        num_decodes += n == 1
    num_tokens = -(-len(positions) // pad_to) * pad_to
    positions += [0] * (num_tokens - len(positions))
    hidden = rng.standard_normal((num_tokens, HIDDEN), dtype=np.float32)
    distribution = np.array([num_decodes, num_decodes, len(step)], np.int32)
    return (
        jnp.asarray(hidden),
        jnp.asarray(positions, jnp.int32),
        jnp.asarray(qsl, jnp.int32),
        jnp.asarray(distribution),
    )


@functools.partial(jax.jit, static_argnames=("kv_layout",), donate_argnums=(0,))
def _compress(
    cache, weights, hidden, positions, qsl, distribution, tables, *, kv_layout
):
    wkv_wgate, ape, norm_weight, cos_sin = weights
    state_table, kv_table = tables
    new_cache, _, _ = compressor_v1.compressor_forward(
        hidden_states=hidden,
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin,
        positions=positions,
        block_table=state_table.reshape(-1),
        query_start_loc=qsl,
        kv_block_table=kv_table.reshape(-1),
        cache=cache,
        rope_cache=None,
        distribution=distribution,
        state_block_size=STATE_BLOCK,
        head_dim=HEAD_DIM,
        compress_ratio=RATIO,
        overlap=True,
        rms_eps=1e-6,
        quant_block=HEAD_DIM,
        kv_layout=kv_layout,
    )
    return new_cache


def _weights(rng):
    coff = 2  # overlapping windows
    return (
        jnp.asarray(
            0.05 * rng.standard_normal((HIDDEN, 2 * coff * HEAD_DIM), dtype=np.float32)
        ),
        jnp.asarray(
            0.1 * rng.standard_normal((RATIO, coff * HEAD_DIM), dtype=np.float32)
        ),
        jnp.asarray(1.0 + 0.1 * rng.standard_normal(HEAD_DIM, dtype=np.float32)),
        jnp.asarray(rng.uniform(-1.0, 1.0, (2048, ROPE_DIM)).astype(np.float32)),
    )


def _expected_sal(has, state_pages, record_pages):
    """The SEQ_ALONG_LANE cache the packed-row one corresponds to."""
    has = np.asarray(has)
    # Nothing but the token states lives on a state page.
    assert not has[state_pages, STATE_ROWS:].any()
    expected = np.zeros(
        indexer_cache_layout.cache_shape(has.shape[0], PAGE, KVLayout.SEQ_ALONG_LANE),
        np.uint8,
    )
    expected[record_pages] = np.asarray(
        convert_cache_to_seq_along_lane(jnp.asarray(has[record_pages]), HEAD_DIM)
    )
    expected[state_pages, :STATE_ROWS] = has[state_pages, :STATE_ROWS]
    return expected


def test_compressor_writes_seq_along_lane():
    rng = np.random.default_rng(0)
    state_table, kv_table, num_pages = _tables()
    state_pages = np.unique(state_table[state_table > 0])
    record_pages = np.unique(kv_table[kv_table > 0])
    weights = _weights(rng)
    tables = (jnp.asarray(state_table), jnp.asarray(kv_table))

    has = jnp.zeros(
        indexer_cache_layout.cache_shape(num_pages, PAGE, KVLayout.HEAD_ALONG_SUBLANE),
        jnp.uint8,
    )
    sal = jnp.zeros(
        indexer_cache_layout.cache_shape(num_pages, PAGE, KVLayout.SEQ_ALONG_LANE),
        jnp.uint8,
    )

    num_records = 0
    for i, step in enumerate(STEPS):
        inputs = _step_inputs(step, rng)
        has = _compress(
            has, weights, *inputs, tables, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
        )
        sal = _compress(
            sal, weights, *inputs, tables, kv_layout=KVLayout.SEQ_ALONG_LANE
        )
        num_records += sum(
            ((p + 1) % RATIO == 0) for first, n in step for p in range(first, first + n)
        )

        expected = _expected_sal(has, state_pages, record_pages)
        actual = np.asarray(sal)
        mismatch = np.argwhere(actual != expected)
        assert mismatch.size == 0, (
            f"step {i}: {len(mismatch)} SEQ_ALONG_LANE bytes differ from the "
            f"packed-row cache, first at [page, group, byte, lane] = "
            f"{mismatch[:8].tolist()}"
        )
        # Every record carries a nonzero UE8M0 scale (byte 128 = group 32, 0).
        scales = actual[record_pages, HEAD_DIM // 4, 0, :]
        assert np.count_nonzero(scales) == num_records, (i, num_records)

    # The indexer reads the same top-k out of either cache: a decode query for
    # request 0 and an 8-token prefill chunk for request 1.
    q_lens = [1, 8]
    seq_lens = np.array([LAST_POS[0] + 1, LAST_POS[1] + 1], np.int32)
    num_q = sum(q_lens)
    num_heads = 8
    q = jnp.asarray(rng.standard_normal((num_q, num_heads, HEAD_DIM), np.float32))
    index_weights = jnp.asarray(rng.uniform(-1.5, 1.5, (num_q, num_heads)), jnp.float32)

    def topk(cache, kv_layout):
        return np.asarray(
            streamindex_topk(
                q=q,
                indexer_weights=index_weights,
                cache_kv=cache,
                seq_lens=jnp.asarray(seq_lens),
                page_indices=jnp.asarray(kv_table.reshape(-1)),
                cu_q_lens=jnp.asarray(np.cumsum([0] + q_lens), jnp.int32),
                distribution=jnp.asarray([1, 1, 2], jnp.int32),
                k=64,
                compression_ratio=RATIO,
                kv_layout=kv_layout,
                num_kv_pages_per_block=(16, 8, 8),
                num_queries_per_block=(1, 128, 128),
                decode_req_batch_size=4,
                chunk_tokens=128,
            )
        )

    from_has = topk(has, KVLayout.HEAD_ALONG_SUBLANE)
    from_sal = topk(sal, KVLayout.SEQ_ALONG_LANE)
    assert (from_has >= 0).any()
    np.testing.assert_array_equal(np.sort(from_sal, -1), np.sort(from_has, -1))
