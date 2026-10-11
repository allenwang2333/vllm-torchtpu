# SPDX-License-Identifier: Apache-2.0
"""TD8.277: int32 ``SpecDecodeMetadata.segment_ids`` are a drop-in for int64.

``get_spec_decode_metadata`` now packs ``segment_ids`` (padded with the
``num_reqs`` sentinel) into the shared int32 H2D buffer. The rejection
samplers and the runner's per-token sampling-param gathers must give identical
results for int32 and int64 segment ids.
"""

from __future__ import annotations

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs


def _inputs(num_draft, pad, vocab=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    nd = torch.tensor(num_draft, dtype=torch.int32)
    seg = torch.repeat_interleave(torch.arange(len(num_draft)), nd)
    grp = torch.cat([torch.arange(n) for n in num_draft]).to(torch.int32)
    seg = torch.cat([seg, torch.full((pad,), len(num_draft))])
    grp = torch.cat([grp, torch.zeros(pad, dtype=torch.int32)])
    total = seg.numel()
    target = torch.randn(total, vocab, generator=g)
    draft = target.argmax(-1)
    flip = torch.rand(total, generator=g) < 0.3
    draft = torch.where(flip, torch.randint(0, vocab, (total,), generator=g), draft)
    bonus = torch.randint(0, vocab, (len(num_draft),), generator=g)
    return g, nd, seg, grp, target, draft.to(torch.int32), bonus, total


@pytest.mark.parametrize(("num_draft", "pad"), [([3, 0, 2], 3), ([4, 4], 0)])
def test_samplers_identical_for_int32_and_int64_segment_ids(num_draft, pad):
    g, nd, seg, grp, target, draft, bonus, total = _inputs(num_draft, pad)
    k = max(num_draft)
    greedy = rs._greedy_rejection_sample_with_segment.__wrapped__
    outs = [
        greedy(draft, target, nd, bonus, seg.to(dt), grp, k, None, None, False)
        for dt in (torch.int64, torch.int32)
    ]
    assert torch.equal(outs[0], outs[1])

    temps = torch.rand(total, 1, generator=g)
    top_k = torch.full((total, 1), 5, dtype=torch.int32)
    top_p = torch.full((total, 1), 0.9)
    accept_u = torch.rand(total, generator=g)
    recover_u = torch.rand(total, target.shape[-1], generator=g)
    random = rs._random_rejection_sample_with_segment.__wrapped__
    outs = [
        random(
            draft,
            target,
            nd,
            bonus,
            seg.to(dt),
            grp,
            k,
            temps,
            top_k,
            top_p,
            accept_u,
            recover_u,
            None,
            False,
        )
        for dt in (torch.int64, torch.int32)
    ]
    assert torch.equal(outs[0], outs[1])

    # Runner-side per-token sampling params: req_temperatures[md.segment_ids]
    # with the sentinel inside the padded per-request table.
    req_t = torch.rand(len(num_draft) + 1, generator=g)
    assert torch.equal(req_t[seg.to(torch.int32)], req_t[seg])
