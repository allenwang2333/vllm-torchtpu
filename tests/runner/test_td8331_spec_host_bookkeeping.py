# SPDX-License-Identifier: Apache-2.0
"""TD8.331: host spec-decode bookkeeping reuses one prefix sum / cached indices.

``spec_logprobs_lists`` computes the step's generated-token prefix sum once
(was per chunk plus again for ``cu_num_generated_tokens``) and skips the
single-chunk ``np.concatenate``; ``get_spec_decode_metadata`` caches
``logits_indices[target_logits_indices + 1]`` for cache hits. Outputs must equal
the previous implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner import tpu_runner_async_output as ao


def _old_spec_logprobs_lists(chunks, gen_lens):
    ids, lps, ranks = [], [], []
    offset = 0
    for chunk in chunks:
        n = len(chunk.row_starts)
        if n:
            g_arr = np.asarray(gen_lens[offset : offset + n], dtype=np.int32)
            cu_g = np.cumsum(g_arr, dtype=np.int32)
            rows = np.repeat(
                chunk.row_starts.astype(np.int32, copy=False) - (cu_g - g_arr), g_arr
            ) + np.arange(int(cu_g[-1]), dtype=np.int32)
        else:
            rows = np.zeros(0, dtype=np.int32)
        ids.append(chunk.tensors.logprob_token_ids.numpy()[rows])
        lps.append(chunk.tensors.logprobs.numpy()[rows])
        ranks.append(chunk.tensors.selected_token_ranks.numpy()[rows])
        offset += n
    return ao.LogprobsLists(
        logprob_token_ids=np.concatenate(ids),
        logprobs=np.concatenate(lps),
        sampled_token_ranks=np.concatenate(ranks),
        cu_num_generated_tokens=np.concatenate(
            ([0], np.cumsum(gen_lens, dtype=np.int32))
        ).tolist(),
    )


def _chunk(g, reqs, k, row_starts_dtype=np.int64):
    rows = max(reqs * (k + 1), 1)
    tensors = SimpleNamespace(
        logprob_token_ids=torch.randint(0, 1000, (rows, 5), generator=g),
        logprobs=torch.randn(rows, 5, generator=g),
        selected_token_ranks=torch.randint(0, 9, (rows,), generator=g),
    )
    starts = (np.arange(reqs) * (k + 1)).astype(row_starts_dtype)
    return ao.SpecLogprobsChunk(tensors=tensors, row_starts=starts)


@pytest.mark.parametrize(
    "layout",
    [[(6, 3)], [(4, 2), (5, 3)], [(3, 1), (0, 0), (2, 4)], [(0, 0)], [(1, 0)]],
)
def test_spec_logprobs_lists_matches_previous(layout):
    g = torch.Generator().manual_seed(len(layout))
    rng = np.random.default_rng(len(layout))
    chunks, gen_lens = [], []
    for reqs, k in layout:
        chunks.append(_chunk(g, reqs, k))
        gen_lens += rng.integers(1, k + 2, reqs).tolist()
    got = ao.spec_logprobs_lists(chunks, gen_lens)
    ref = _old_spec_logprobs_lists(chunks, gen_lens)
    for field in ("logprob_token_ids", "logprobs", "sampled_token_ranks"):
        assert np.array_equal(getattr(got, field), getattr(ref, field)), field
        assert getattr(got, field).dtype == getattr(ref, field).dtype
    assert got.cu_num_generated_tokens == ref.cu_num_generated_tokens


def test_spec_logprobs_lists_single_chunk_result_is_independent():
    g = torch.Generator().manual_seed(0)
    chunk = _chunk(g, 3, 2)
    out = ao.spec_logprobs_lists([chunk], [3, 1, 2])
    before = chunk.tensors.logprobs.clone()
    out.logprobs[:] = 0  # must not write through into the source tensors
    assert torch.equal(chunk.tensors.logprobs, before)
