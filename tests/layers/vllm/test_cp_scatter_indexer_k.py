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
"""`_cp_scatter_indexer_k`: the DCP indexer-K writer.

The reader places global position `p` on rank `(p // c) % D` at local index
`(p // (D * c)) * c + p % c`. The writer derives its own
`(rank, page, slot)` from a different expression, so the property worth
testing is that the two agree: a row written for global position `p` must
land exactly where the reader will go looking for it, and nowhere else.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import \
    _cp_scatter_indexer_k

KV_PACKING = 4
LKV_DIM = 128
K_WIDTH = 65  # fp8 head dim + one e8m0 scale byte, as the indexer emits
UNWRITTEN = 0xAA  # cache fill; no written row can look like this

# (dcp_size, interleave_size, page_size, pages_per_seq)
_LAYOUTS = [
    (2, 4, 32, 3),
    (4, 4, 32, 3),
    (4, 32, 128, 2),
    (8, 4, 64, 2),
]


def _reader_rank(pos, dcp_size, interleave_size):
    """Where the reader looks for global position `pos`: its owning rank."""
    return (pos // interleave_size) % dcp_size


def _reader_local(pos, dcp_size, interleave_size):
    """...and the flat index it looks at inside that rank's shard."""
    return (pos //
            (dcp_size * interleave_size)) * interleave_size + (pos %
                                                               interleave_size)


def _mesh(dcp_size):
    devices = jax.devices()
    if len(devices) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(devices)}")
    return Mesh(np.asarray(devices[:dcp_size]), (DCP_AXIS_NAME, ))


def _rows(num_tokens):
    """One identifiable byte row per token: `[lo, hi, 7, 0...]`."""
    t = np.arange(num_tokens, dtype=np.int64)
    rows = np.zeros((num_tokens, K_WIDTH), np.uint8)
    rows[:, 0] = t % 251
    rows[:, 1] = (t // 251) % 251
    rows[:, 2] = 7
    return rows


def _scatter(mesh, cache, rows, seq_lens, page_tbl, cu_q_lens, dcp_size,
             interleave_size):
    fn = functools.partial(_cp_scatter_indexer_k,
                           mesh=mesh,
                           cp_size=dcp_size,
                           interleave_size=interleave_size,
                           axis_name=DCP_AXIS_NAME)
    sharded = jax.device_put(jnp.asarray(cache),
                             NamedSharding(mesh, P(DCP_AXIS_NAME)))
    return np.asarray(
        jax.jit(fn)(sharded, jnp.asarray(rows), jnp.asarray(seq_lens),
                    jnp.asarray(page_tbl), jnp.asarray(cu_q_lens)))


def _case(dcp_size, page_size, pages_per_seq, num_seqs=2, pad=13):
    """Full-length prefill for `num_seqs`, plus `pad` padded token slots."""
    local_pages = num_seqs * pages_per_seq
    seq_len = pages_per_seq * page_size * dcp_size
    seq_lens = np.full(num_seqs, seq_len, np.int32)
    cu_q_lens = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32)
    page_tbl = np.random.default_rng(0).permutation(local_pages).astype(
        np.int32)
    cache = np.full(
        (dcp_size * local_pages, page_size // KV_PACKING, KV_PACKING, LKV_DIM),
        UNWRITTEN, np.uint8)
    return seq_lens, cu_q_lens, page_tbl, cache, int(cu_q_lens[-1]) + pad


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size,interleave_size,page_size,pages_per_seq",
                         _LAYOUTS)
def test_writer_lands_where_the_reader_looks(dcp_size, interleave_size,
                                             page_size, pages_per_seq):
    """Every token's row is at the reader's `(owner, local index)`, exactly."""
    mesh = _mesh(dcp_size)
    num_seqs = 2
    seq_lens, cu_q_lens, page_tbl, cache, num_tokens = _case(
        dcp_size, page_size, pages_per_seq, num_seqs)
    rows = _rows(num_tokens)

    out = _scatter(mesh, cache, rows, seq_lens, page_tbl, cu_q_lens, dcp_size,
                   interleave_size)
    local_pages = cache.shape[0] // dcp_size
    shards = out.reshape(dcp_size, local_pages, page_size // KV_PACKING,
                         KV_PACKING, LKV_DIM)

    seen = np.zeros((dcp_size, local_pages, page_size), bool)
    for seq in range(num_seqs):
        base = int(cu_q_lens[seq])
        positions = np.arange(int(seq_lens[seq]))
        ranks = _reader_rank(positions, dcp_size, interleave_size)
        flats = _reader_local(positions, dcp_size, interleave_size)
        for pos in range(int(seq_lens[seq])):
            rank = int(ranks[pos])
            virtual, slot = divmod(int(flats[pos]), page_size)
            page = int(page_tbl[seq * pages_per_seq + virtual])
            got = shards[rank, page, slot // KV_PACKING, slot % KV_PACKING]
            want = rows[base + pos]
            assert np.array_equal(got[:K_WIDTH], want), (
                f"seq={seq} pos={pos} -> rank={rank} page={page} slot={slot}")
            assert not got[K_WIDTH:].any(), "tail must be zero-padded"
            assert not seen[rank, page, slot], "two tokens hit one slot"
            seen[rank, page, slot] = True

    untouched = shards.reshape(dcp_size, local_pages, page_size,
                               LKV_DIM)[~seen]
    assert (untouched == UNWRITTEN).all(), "the writer scribbled outside"


@pytest.mark.multichip
def test_padded_tokens_write_nothing(dcp_size=4,
                                     interleave_size=4,
                                     page_size=32,
                                     pages_per_seq=2):
    """Rows past `cu_q_lens[-1]` are dropped, not aimed at a real slot.

    Their `pos` is meaningless and can index past the page table, so this is
    what the `seq_id` clamp and the `virtual_page` clip protect.
    """
    mesh = _mesh(dcp_size)
    seq_lens, cu_q_lens, page_tbl, cache, _ = _case(dcp_size,
                                                    page_size,
                                                    pages_per_seq,
                                                    num_seqs=2,
                                                    pad=0)
    n_real = int(cu_q_lens[-1])
    ref = _scatter(mesh, cache, _rows(n_real), seq_lens, page_tbl, cu_q_lens,
                   dcp_size, interleave_size)
    padded = _rows(n_real + 97)
    padded[n_real:] = 0x5C
    got = _scatter(mesh, cache, padded, seq_lens, page_tbl, cu_q_lens,
                   dcp_size, interleave_size)
    assert np.array_equal(ref, got), "a padded token reached the cache"


@pytest.mark.multichip
def test_interleave_must_divide_the_page(dcp_size=4):
    """The guard that keeps writer and reader in one coordinate space.

    `owner` is derived from `pos % (page_size * dcp_size)`; the reader's
    owner is derived from `pos` directly. They coincide only when
    the `dcp_size * interleave_size` cycle divides the virtual page, i.e.
    when `interleave_size` divides `page_size`.
    """
    mesh = _mesh(dcp_size)
    page_size, pages_per_seq = 32, 2
    seq_lens, cu_q_lens, page_tbl, cache, num_tokens = _case(dcp_size,
                                                             page_size,
                                                             pages_per_seq,
                                                             num_seqs=1,
                                                             pad=0)
    with pytest.raises(ValueError, match="must divide the indexer"):
        _scatter(mesh,
                 cache,
                 _rows(num_tokens),
                 seq_lens,
                 page_tbl,
                 cu_q_lens,
                 dcp_size,
                 interleave_size=24)
