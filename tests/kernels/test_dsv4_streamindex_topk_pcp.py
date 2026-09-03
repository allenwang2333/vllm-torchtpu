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
"""Tests for PCP (prefill context parallel) StreamIndex Top-K."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    PCP_AXIS_NAME, pcp_local_length, pcp_local_to_global, streamindex_topk,
    streamindex_topk_pcp)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    build_pcp_rank_major_token_order

P = jax.sharding.PartitionSpec

# ===================================================================== #
# Coordinate algebra. Single-device; these are the properties the kernel
# relies on to keep its masks in global coordinates while its score
# columns are rank-local.
# ===================================================================== #

_LAYOUTS = [(pcp, ic) for pcp in (2, 4, 8, 16) for ic in (1, 2, 4, 8, 64)]


@pytest.mark.parametrize("pcp_size,interleave_c", _LAYOUTS)
def test_pcp_shards_partition_the_global_range(pcp_size, interleave_c):
    """Every global position is owned by exactly one rank, at one local index."""
    global_len = pcp_size * interleave_c * 5 + 3  # deliberately not a whole cycle
    seen = {}
    for rank in range(pcp_size):
        n_local = int(
            pcp_local_length(global_len, rank, pcp_size, interleave_c))
        for local in range(n_local):
            g = int(pcp_local_to_global(local, rank, pcp_size, interleave_c))
            assert 0 <= g < global_len
            assert g not in seen, f"{g} owned by both {seen.get(g)} and {rank}"
            seen[g] = rank
    assert len(seen) == global_len


@pytest.mark.parametrize("pcp_size,interleave_c", _LAYOUTS)
def test_pcp_local_to_global_is_monotonic(pcp_size, interleave_c):
    """Monotonicity is what makes `global_pos < kv_len` a valid local bound."""
    for rank in range(pcp_size):
        gs = [
            int(pcp_local_to_global(l, rank, pcp_size, interleave_c))
            for l in range(4 * pcp_size * interleave_c)
        ]
        assert gs == sorted(gs)
        assert len(set(gs)) == len(gs)


@pytest.mark.parametrize("pcp_size,interleave_c", _LAYOUTS)
def test_pcp_ownership_rule(pcp_size, interleave_c):
    """Global position g lives on rank (g // interleave_c) % pcp_size."""
    for rank in range(pcp_size):
        for local in range(50):
            g = int(pcp_local_to_global(local, rank, pcp_size, interleave_c))
            assert (g // interleave_c) % pcp_size == rank


@pytest.mark.parametrize("pcp_size,interleave_c", _LAYOUTS)
def test_pcp_local_length_is_the_exclusive_bound(pcp_size, interleave_c):
    """The kernel drops position `n_local` by testing its *global* coordinate.

    That only works if the first unowned local index maps outside the global
    range, i.e. the count and the coordinate map agree on where a shard ends.
    """
    for global_len in range(1, pcp_size * interleave_c * 3 + 1):
        for rank in range(pcp_size):
            n = int(pcp_local_length(global_len, rank, pcp_size, interleave_c))
            if n:
                last = int(
                    pcp_local_to_global(n - 1, rank, pcp_size, interleave_c))
                assert last < global_len
            first_unowned = int(
                pcp_local_to_global(n, rank, pcp_size, interleave_c))
            assert first_unowned >= global_len


def test_pcp_size_one_is_the_identity():
    """The default path must not pay for PCP, so the helpers compile away."""
    for i in (0, 1, 7, 4096):
        assert int(pcp_local_to_global(i, 0, 1, 1)) == i
        assert int(pcp_local_length(i, 0, 1, 1)) == i


# ===================================================================== #
# Argument validation.
# ===================================================================== #


def _dummy_args(num_tokens=8,
                num_heads=4,
                head_dim=128,
                pages=2,
                page_size=16):
    width = 256
    return {
        "q": jnp.zeros((num_tokens, num_heads, head_dim), jnp.float8_e4m3fn),
        "indexer_weights": jnp.zeros((num_tokens, num_heads), jnp.bfloat16),
        "cache_kv": jnp.zeros((pages, page_size // 4, 4, width), jnp.uint8),
        "seq_lens": jnp.array([num_tokens], jnp.int32),
        "page_indices": jnp.arange(pages, dtype=jnp.int32),
        "cu_q_lens": jnp.array([0, num_tokens], jnp.int32),
        "distribution": jnp.array([0, 0, 1], jnp.int32),
    }


def test_interleave_must_be_a_multiple_of_compression_ratio():
    with pytest.raises(ValueError, match="must be a multiple of"):
        streamindex_topk(**_dummy_args(),
                         k=16,
                         compression_ratio=4,
                         pcp_size=2,
                         interleave_size=6)


def test_early_exit_is_rejected_under_pcp():
    with pytest.raises(NotImplementedError, match="enable_early_exit"):
        streamindex_topk(**_dummy_args(),
                         k=16,
                         compression_ratio=1,
                         pcp_size=2,
                         interleave_size=1,
                         enable_early_exit=True)


def test_early_exit_is_rejected_with_return_scores():
    with pytest.raises(NotImplementedError, match="return_scores"):
        streamindex_topk(**_dummy_args(),
                         k=16,
                         compression_ratio=1,
                         enable_early_exit=True,
                         return_scores=True)


# ===================================================================== #
# End-to-end against the unsharded kernel.
# ===================================================================== #

PCP_SIZE = 8
H_IDX, D_IDX, K = 32, 128, 2048
Q_LEN, KV_LEN, PAGE_SIZE = 256, 32768, 1024
WIDTH = 256


def _global_records(kv_len, seed=7):
    """One record array; both paths are views of it, so any diff is a real bug."""
    vals = jax.random.normal(jax.random.key(seed), (kv_len, D_IDX),
                             jnp.float32)
    fp8 = jax.lax.bitcast_convert_type(vals.astype(jnp.float8_e4m3fn),
                                       jnp.uint8).reshape(kv_len, D_IDX)
    # 127 is the e8m0 exponent bias, i.e. a scale of exactly 1.0.
    rec = jnp.concatenate([fp8, jnp.full((kv_len, 1), 127, jnp.uint8)], -1)
    return jnp.pad(rec, ((0, 0), (0, WIDTH - rec.shape[-1])))


@pytest.mark.multichip
def test_pcp_matches_unsharded_streamindex_topk():
    """8-rank PCP top-k must equal the unsharded top-k over the same KV.

    With interleave_size=1, global compressed position g lives on rank
    g % pcp_size at local index g // pcp_size, so the shards are a strided
    de-interleave of one global record array.
    """
    devices = jax.devices()
    if len(devices) < PCP_SIZE:
        pytest.skip(f"needs {PCP_SIZE} devices, have {len(devices)}")

    rec = _global_records(KV_LEN)
    kq, kw = jax.random.split(jax.random.key(0))
    q = (jax.random.normal(kq, (Q_LEN, H_IDX, D_IDX), jnp.float32) *
         0.5).astype(jnp.float8_e4m3fn)
    weights = jax.random.normal(kw, (Q_LEN, H_IDX),
                                jnp.float32).astype(jnp.bfloat16)
    seq_lens = jnp.array([KV_LEN], jnp.int32)
    cu_q_lens = jnp.array([0, Q_LEN], jnp.int32)
    distribution = jnp.array([0, 0, 1], jnp.int32)

    pages = KV_LEN // PAGE_SIZE
    reference = np.asarray(
        streamindex_topk(q,
                         weights,
                         rec.reshape(pages, PAGE_SIZE // 4, 4, WIDTH),
                         seq_lens,
                         jnp.arange(pages, dtype=jnp.int32),
                         cu_q_lens,
                         distribution,
                         k=K,
                         compression_ratio=1,
                         num_kv_pages_per_block=2,
                         num_queries_per_block=128))

    vpages = KV_LEN // (PAGE_SIZE * PCP_SIZE)
    shards = jnp.concatenate([
        rec[r::PCP_SIZE].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
        for r in range(PCP_SIZE)
    ], 0)
    mesh = jax.sharding.Mesh(np.array(devices[:PCP_SIZE]), (PCP_AXIS_NAME, ))
    shards = jax.device_put(shards,
                            jax.sharding.NamedSharding(mesh, P(PCP_AXIS_NAME)))
    order, inverse = build_pcp_rank_major_token_order([Q_LEN], PCP_SIZE, 1,
                                                      Q_LEN)

    got = np.asarray(
        jax.jit(
            functools.partial(streamindex_topk_pcp,
                              mesh=mesh,
                              k=K,
                              compression_ratio=1,
                              pcp_size=PCP_SIZE,
                              interleave_size=1,
                              num_kv_pages_per_block=2,
                              num_queries_per_block=128))(
                                  q, weights, shards, seq_lens,
                                  jnp.arange(vpages, dtype=jnp.int32),
                                  cu_q_lens, distribution,
                                  jnp.asarray(order, jnp.int32)))
    # The output is in rank-major packed order; undo that before comparing.
    got = got[np.asarray(inverse)]

    assert got.shape == reference.shape
    # Every row is full here: the query tokens are the last Q_LEN of a KV_LEN
    # context, so even the earliest sees far more than K causal positions.
    assert (reference >= 0).all()
    assert (got >= 0).all()
    assert got.max() < KV_LEN

    # Top-k is a set, not a ranking -- sparsecore_topk returns unsorted indices
    # and ties may break either way.
    mismatched = [
        t for t in range(Q_LEN)
        if set(reference[t].tolist()) != set(got[t].tolist())
    ]
    assert not mismatched, (f"{len(mismatched)}/{Q_LEN} rows differ, "
                            f"first at token {mismatched[:5]}")
