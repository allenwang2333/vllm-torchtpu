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
"""Tests for DCP (decode context parallel) StreamIndex Top-K.

DCP replicates the queries and shards the KV, so each rank's `streamindex_topk`
returns a *local* top-k: correct only after the caller merges the returned
scores across ranks. These tests run that merge end to end against the
unsharded kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk

# The coordinate algebra this rests on -- `cp_local_to_global` and the torch
# inverse the return leg applies -- is covered on CPU in
# `tests/distributed/test_dcp_wiring.py`.

# ===================================================================== #
# Argument validation on the CP path of `streamindex_topk`.
# ===================================================================== #


def _dummy_args(num_tokens=8,
                num_heads=4,
                head_dim=128,
                pages=2,
                page_size=16):
    width = 256
    return (
        jnp.zeros((num_tokens, num_heads, head_dim), jnp.float8_e4m3fn),
        jnp.zeros((num_tokens, num_heads), jnp.bfloat16),
        jnp.zeros((pages, page_size // 4, 4, width), jnp.uint8),
        jnp.array([num_tokens], jnp.int32),
        jnp.arange(pages, dtype=jnp.int32),
        jnp.array([0, num_tokens], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
    )


def _topk_kwargs(**over):
    kw = dict(k=16,
              compression_ratio=1,
              num_kv_pages_per_block=2,
              num_queries_per_block=8,
              cp_size=2,
              interleave_size=1)
    kw.update(over)
    return kw


def test_cp_size_must_be_positive():
    with pytest.raises(ValueError, match="cp_size must be >= 1"):
        streamindex_topk(*_dummy_args(), **_topk_kwargs(cp_size=0))


def test_interleave_size_must_cover_whole_compressed_rows():
    """Otherwise a CP chunk boundary falls inside a compressed row."""
    with pytest.raises(ValueError, match="must be a multiple of"):
        streamindex_topk(
            *_dummy_args(),
            **_topk_kwargs(compression_ratio=4, interleave_size=2))


def test_early_exit_is_rejected_under_cp():
    """Its identity indices are global positions, not this rank's subset."""
    with pytest.raises(NotImplementedError, match="enable_early_exit"):
        streamindex_topk(*_dummy_args(),
                         **_topk_kwargs(enable_early_exit=True))


# ===================================================================== #
# End-to-end against the unsharded kernel.
# ===================================================================== #

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


def _local_candidates(q, weights, shard, seq_lens, page_indices, cu_q_lens,
                      distribution, *, dcp_size, dcp_rank, return_scores):
    """One rank's candidates over its own shard, as the shipped op calls it."""
    return streamindex_topk(
        q,
        weights,
        shard,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=K,
        compression_ratio=1,
        num_kv_pages_per_block=2,
        num_queries_per_block=128,
        cp_size=dcp_size,
        cp_rank=dcp_rank,
        interleave_size=1,
        return_scores=return_scores,
    )


def _merged_global_topk(q, weights, shards, seq_lens, page_indices, cu_q_lens,
                        distribution, *, dcp_size):
    """The DCP pipeline of `_build_streamidx_dcp_op`, in pure JAX.

    Local candidates per rank, concatenated rank-major the way the torch
    all-gather leaves them, re-ranked by `sparsecore_topk`, and read back out
    as global positions.
    """
    per_rank = [
        _local_candidates(q,
                          weights,
                          shards[r],
                          seq_lens,
                          page_indices,
                          cu_q_lens,
                          distribution,
                          dcp_size=dcp_size,
                          dcp_rank=r,
                          return_scores=True) for r in range(dcp_size)
    ]
    all_idxs = jnp.concatenate([idxs for idxs, _ in per_rank], axis=1)
    all_scores = jnp.concatenate([scores for _, scores in per_rank], axis=1)
    slots = sparsecore_topk(all_scores,
                            K,
                            row_lengths=jnp.full((all_scores.shape[0], ),
                                                 all_scores.shape[1],
                                                 jnp.int32),
                            write_empty_rows=True)
    picked = jnp.take_along_axis(all_idxs,
                                 jnp.clip(slots, 0).astype(jnp.int32),
                                 axis=1)
    return jnp.where(slots >= 0, picked, jnp.int32(-1))


def _shards(rec, dcp_size, vpages):
    """interleave_size=1: global position g lives on rank g % dcp_size at
    local index g // dcp_size, so the shards are a strided de-interleave."""
    return [
        rec[r::dcp_size].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
        for r in range(dcp_size)
    ]


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_dcp_merge_reproduces_the_unsharded_global_topk(dcp_size):
    """The merged list must be the unsharded global top-k, exactly.

    Exactness is the requirement, not an approximation target: DCP partitions
    the KV, so a global winner is beaten by at most `k - 1` positions overall
    and hence by at most `k - 1` on its own shard, and always survives that
    shard's local top-k.
    """
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

    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    merged = np.asarray(
        _merged_global_topk(q,
                            weights,
                            _shards(rec, dcp_size, vpages),
                            seq_lens,
                            jnp.arange(vpages, dtype=jnp.int32),
                            cu_q_lens,
                            distribution,
                            dcp_size=dcp_size))
    assert merged.shape == (Q_LEN, K)

    for t in range(Q_LEN):
        got = merged[t][merged[t] >= 0].tolist()
        # Multiplicity, not just set equality: the LSE merge double-counts a
        # position delivered twice.
        assert len(got) == len(set(got)), f"token {t} has duplicates"
        want = set(reference[t][reference[t] >= 0].tolist())
        assert set(got) == want, (
            f"token {t}: {len(want - set(got))} missing, "
            f"{len(set(got) - want)} spurious")


def test_local_candidates_stay_inside_their_own_shard(dcp_size=4):
    """Each rank's pre-merge candidates are global positions it owns.

    The torch ownership filter reads `slots // k` rather than the gathered
    index, so this precondition is what keeps that shortcut sound.
    """
    rec = _global_records(KV_LEN, seed=3)
    kq, kw = jax.random.split(jax.random.key(3))
    q = (jax.random.normal(kq, (Q_LEN, H_IDX, D_IDX), jnp.float32) *
         0.5).astype(jnp.float8_e4m3fn)
    weights = jax.random.normal(kw, (Q_LEN, H_IDX),
                                jnp.float32).astype(jnp.bfloat16)

    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    shards = _shards(rec, dcp_size, vpages)
    local = np.stack([
        np.asarray(
            _local_candidates(q,
                              weights,
                              shards[r],
                              jnp.array([KV_LEN], jnp.int32),
                              jnp.arange(vpages, dtype=jnp.int32),
                              jnp.array([0, Q_LEN], jnp.int32),
                              jnp.array([0, 0, 1], jnp.int32),
                              dcp_size=dcp_size,
                              dcp_rank=r,
                              return_scores=False)) for r in range(dcp_size)
    ])

    for r in range(dcp_size):
        held = local[r][local[r] >= 0]
        assert held.size == Q_LEN * K, f"rank {r} returned a short row"
        owners = held % dcp_size
        assert np.all(owners == r), f"rank {r} claimed a foreign index"
