# SPDX-License-Identifier: Apache-2.0
"""TD8.280: int32 owner-token-layout host paths and int32 ``topk_mask`` counts.

``owner_token_layout`` builds owners / block ids / scheduled tokens / q_loc in
``np.int32`` (was ``np.int64``); results must equal the int64 computation.
``topk_mask`` accumulates ``count_gt`` with ``sum(dtype=torch.int32)``; the
mask must equal the previous ``to(int32)`` + default (int64) sum form.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import vllm  # noqa: F401

from vllm_torchtpu.core.owner_affine_block_pool import OwnerLayout
from vllm_torchtpu.layers.adapter.sample import top_k_top_p as tk
from vllm_torchtpu.runner import owner_token_layout as otl

LAYOUT = OwnerLayout(num_replicated=4, num_owners=4, blocks_per_owner=50)


def _ref_extract(tables, layout, limits, default_owner):
    """Pre-TD8.280 int64 owner extraction."""
    owners = np.full(tables[0].shape[0], default_owner, dtype=np.int64)
    for r in range(tables[0].shape[0]):
        for tbl in tables:
            row = tbl[r, : max(0, min(tbl.shape[1], int(limits[r])))]
            owned = (row >= layout.num_replicated) & (row < layout.num_blocks)
            if owned.any():
                owners[r] = layout.region_of(int(row[int(np.argmax(owned))]))
                break
    return owners


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_extract_and_request_owners_int32_match_int64_reference(seed):
    rng = np.random.default_rng(seed)
    reqs = 9
    swa = rng.integers(0, 4, (reqs, 5)).astype(np.int32)
    main = rng.integers(0, LAYOUT.num_blocks, (reqs, 5)).astype(np.int32)
    main[0] = 1  # no owned block -> default owner
    limits = rng.integers(0, 6, reqs).astype(np.int64)
    owners = otl.extract_request_owners_from_block_table(
        [swa, main], limits, LAYOUT, default_owner=2
    )
    assert owners.dtype == np.int32
    np.testing.assert_array_equal(owners, _ref_extract([swa, main], LAYOUT, limits, 2))

    first = rng.integers(4, LAYOUT.num_blocks, (reqs, 1)).astype(np.int64)
    got = otl.request_owners(first, slice(0, reqs), LAYOUT)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, (first[:, 0] - 4) // 50)


@pytest.mark.parametrize("tokens_kind", ["list", "int64", "tensor"])
def test_prepare_layout_independent_of_scheduled_tokens_dtype(tokens_kind):
    main = np.array([[60, 7], [5, 0], [170, 9], [120, 1]], dtype=np.int32)
    tokens = [3, 1, 4, 2]
    arg = {
        "list": tokens,
        "int64": np.asarray(tokens, dtype=np.int64),
        "tensor": torch.tensor(tokens),
    }[tokens_kind]
    layout = otl.prepare_kv_shard_owner_token_layout(main, LAYOUT, arg, [16, 32])
    ref = otl.build_owner_token_layout(
        np.array([1, 0, 3, 2], dtype=np.int64), np.asarray(tokens), 4, [16, 32]
    )
    for name in ("perm", "inv", "query_start_loc", "owner_tokens"):
        np.testing.assert_array_equal(getattr(layout, name), getattr(ref, name))


def _old_topk_mask(logits, k, replace_val):
    k = k.squeeze(-1)

    def predicate(threshold):
        neg_threshold = (-threshold).unsqueeze(-1)
        count_gt = torch.sum((logits > neg_threshold).to(torch.int32), dim=-1)
        return count_gt >= k

    cutoff = -tk._float32_bsearch(logits.shape[:-1], predicate, logits.device)
    return torch.where(logits >= cutoff.unsqueeze(-1), logits, replace_val)


@pytest.mark.parametrize("seed", [0, 1])
def test_topk_mask_int32_count_matches_int64_sum(seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(6, 300, generator=g)
    logits[0, :5] = 2.5  # ties at the cutoff
    k = torch.tensor([[1], [3], [5], [300], [7], [50]], dtype=torch.int32)
    out = tk.topk_mask(logits, k, -float("inf"))
    ref = _old_topk_mask(logits, k, -float("inf"))
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))
