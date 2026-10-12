# SPDX-License-Identifier: Apache-2.0
"""TD8.280: int32 ``topk_mask`` counts.

``topk_mask`` accumulates ``count_gt`` with ``sum(dtype=torch.int32)``; the
mask must equal the previous ``to(int32)`` + default (int64) sum form.
"""

from __future__ import annotations

import pytest
import torch
import vllm  # noqa: F401

from vllm_torchtpu.layers.adapter.sample import top_k_top_p as tk


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
