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
"""Masked-dense CSA attention against the sparse gather kernel."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.deepseek_v4.sparse_mla_test import create_cache
from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
    csa_masked_dense,
    sparse_mla,
)

pytestmark = pytest.mark.tpu_test

HEADS, HEAD_DIM, PAGE_SIZE, TOPK = 16, 512, 256, 1024


def _case(rng, seqs):
    """seqs: (q_len, token context before this step) per sequence."""
    q_lens = np.array([q for q, _ in seqs])
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    kv_lens = np.array([(c + q) // 4 for q, c in seqs], np.int32)
    pages_per_seq = max(1, -(-int(kv_lens.max()) // PAGE_SIZE))
    total_pages = len(seqs) * pages_per_seq + 3
    pages = rng.permutation(total_pages)[: len(seqs) * pages_per_seq]
    topk = []
    for (q, c), kv_len in zip(seqs, kv_lens):
        for t in range(q):
            visible = max(1, min(kv_len, (c + t + 1) // 4))
            sel = (
                np.arange(visible)
                if visible <= TOPK
                else rng.choice(visible, TOPK, replace=False)
            )
            row = np.full(TOPK, -1, np.int32)
            row[: len(sel)] = sel
            topk.append(row)
    n = int(cu[-1])
    _, nope, rope = create_cache(rng, total_pages, PAGE_SIZE, HEAD_DIM, jnp.bfloat16)
    return dict(
        q=(
            jnp.array(rng.standard_normal((n, HEADS, HEAD_DIM), np.float32)) * 0.05
        ).astype(jnp.bfloat16),
        nope=nope,
        rope=rope,
        topk=jnp.array(np.stack(topk)),
        pages=jnp.array(pages, jnp.int32),
        cu=jnp.array(cu),
        kv_lens=jnp.array(kv_lens),
        dist=jnp.array([0, 0, len(seqs)], jnp.int32),
        sinks=jnp.array(rng.random((HEADS,), np.float32)),
        swa_acc=jnp.array(rng.random((n, HEADS, HEAD_DIM), np.float32)).astype(
            jnp.bfloat16
        ),
        swa_l=jnp.array(rng.random((n, HEADS), np.float32)) + 1.0,
        swa_m=jnp.array(rng.random((n, HEADS), np.float32)),
    )


def _sparse(c, num_valid_tokens=None):
    return sparse_mla.sparse_ragged_paged_attention(
        c["q"], c["nope"], c["rope"], c["topk"], c["pages"], c["cu"], c["dist"],
        c["sinks"], c["swa_acc"], c["swa_l"], c["swa_m"], sm_scale=0.04,
        gather_and_attention_chunk_size=32, attention_kernel_batch_size=8,
        num_valid_tokens=num_valid_tokens,
    )  # fmt: skip


def _split(c, max_kv_len, pages_per_block):
    return csa_masked_dense.split_attention(
        c["q"], c["nope"], c["rope"], c["topk"], c["kv_lens"], c["pages"], c["cu"],
        c["dist"], c["sinks"], c["swa_acc"], c["swa_l"], c["swa_m"], sm_scale=0.04,
        max_kv_len=max_kv_len, num_kv_pages_per_block=pages_per_block,
        num_queries_per_block=64,
    )  # fmt: skip


@pytest.mark.parametrize("max_kv_len", [4096, 8192])
def test_sc_mask_matches_xla(max_kv_len):
    c = _case(np.random.default_rng(1), [(4, 9000), (200, 20000)])
    kw = dict(max_kv_len=max_kv_len, page_size=PAGE_SIZE, bkv_sz=2 * PAGE_SIZE)
    np.testing.assert_array_equal(
        csa_masked_dense.build_mask(c["topk"], **kw),
        csa_masked_dense.build_mask_xla(c["topk"], **kw),
    )


@pytest.mark.parametrize("pages_per_block", [2, 4])
@pytest.mark.parametrize(
    "seqs",
    [
        [(130, 0)],  # every visible position selected
        [(4, 2000), (4, 40000), (300, 6000)],  # deep verify seq stays sparse
        [(4, 40000), (200, 17000)],  # partial last KV block, top-k < visible
        [(100, 40000), (300, 6000)],  # sparse prefix too long for the slice
    ],
)
def test_split_matches_sparse(seqs, pages_per_block):
    c = _case(np.random.default_rng(0), seqs)
    max_kv_len = 8192
    ref = _sparse(c)
    out = _split(c, max_kv_len, pages_per_block)
    long = np.asarray(c["kv_lens"]) > max_kv_len
    split_seq = int(np.max(np.where(long, np.arange(len(seqs)) + 1, 0)))
    n_sparse = int(c["cu"][split_seq])
    # Sparse rows come from the same sparse kernel; dense rows differ from it
    # only in summation order.
    np.testing.assert_array_equal(out[:n_sparse], ref[:n_sparse])
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(ref, np.float32), atol=8e-3, rtol=0
    )
