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
"""The sparse (top-k window) spec-verify sampler against the full-vocab one."""

import pytest
import torch
import torch._dynamo

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    sparse_window_logits,
    sparse_window_sample,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    _chunked_topk,
    apply_top_k_top_p,
)

WIDTH = 64
# More than WIDTH chunks of 128, so _chunked_topk takes its chunked path.
VOCAB = 20000


@pytest.fixture(autouse=True)
def _raise_dynamo_recompile_limit():
    cfg = torch._dynamo.config
    names = [n for n in ("recompile_limit", "cache_size_limit") if hasattr(cfg, n)]
    saved = {n: getattr(cfg, n) for n in names}
    for n in names:
        setattr(cfg, n, 1024)
    yield
    for n, v in saved.items():
        setattr(cfg, n, v)


def _params(device):
    top_k = torch.tensor([[1], [5], [20], [64], [20], [20]], dtype=torch.int32)
    top_p = torch.tensor([[1.0], [0.9], [0.95], [0.5], [1.0], [0.3]])
    temps = torch.tensor([[1.0], [0.6], [0.6], [1.0], [0.25], [2.0]])
    return temps.to(device), top_k.to(device), top_p.to(device)


def _full_probs(logits, temps, top_k, top_p):
    masked = apply_top_k_top_p(logits / temps, top_k, top_p)
    return torch.softmax(masked, dim=-1)


def test_window_holds_the_full_distribution(device):
    torch.manual_seed(0)
    logits = (torch.randn(6, VOCAB) * 3.0).to(device)
    temps, top_k, top_p = _params(device)
    full = _full_probs(logits, temps, top_k, top_p).cpu()
    vals, idx = _chunked_topk(logits, WIDTH)
    window = torch.softmax(
        sparse_window_logits(vals.float() / temps, top_k, top_p), dim=-1
    ).cpu()
    torch.testing.assert_close(window, full.gather(-1, idx.cpu()), atol=1e-6, rtol=1e-5)
    # No mass outside the window.
    torch.testing.assert_close(
        full.gather(-1, idx.cpu()).sum(-1), torch.ones(6), atol=1e-5, rtol=0
    )


def test_window_sample_matches_full_distribution(device):
    torch.manual_seed(1)
    n = 8192
    row = torch.randn(1, VOCAB) * 2.0
    logits = row.expand(n, VOCAB).contiguous().to(device)
    temps = torch.full((n, 1), 0.6, device=device)
    top_k = torch.full((n, 1), 20, dtype=torch.int32, device=device)
    top_p = torch.full((n, 1), 0.95, device=device)
    vals, idx = _chunked_topk(logits, WIDTH)
    u = torch.rand((n, WIDTH), device=device)
    tokens = sparse_window_sample(vals, idx, temps, top_k, top_p, u, 1e-5).cpu()
    p = _full_probs(logits[:1], temps[:1], top_k[:1], top_p[:1]).cpu()[0]
    freq = torch.bincount(tokens, minlength=VOCAB).float() / n
    assert freq[p == 0].sum() == 0
    sigma = torch.sqrt(p * (1 - p) / n)
    assert torch.all((freq - p).abs() <= 5 * sigma + 1e-6)


def test_window_sample_greedy_rows_take_argmax(device):
    torch.manual_seed(2)
    logits = torch.randn(3, VOCAB).to(device)
    vals, idx = _chunked_topk(logits, WIDTH)
    temps = torch.zeros((3, 1), device=device)
    top_k = torch.zeros((3, 1), dtype=torch.int32, device=device)
    top_p = torch.ones((3, 1), device=device)
    u = torch.rand((3, WIDTH), device=device)
    tokens = sparse_window_sample(vals, idx, temps, top_k, top_p, u, 1e-5)
    assert torch.equal(tokens.cpu(), torch.argmax(logits, dim=-1).cpu())


def _run_both(logits, draft, num_draft, temps, top_k, top_p, accept_u, device):
    t = draft.shape[0]
    segment_ids = torch.zeros(t, dtype=torch.int64, device=device)
    group_indices = torch.arange(t, dtype=torch.int64, device=device)
    bonus = torch.tensor([7], dtype=torch.int32, device=device)
    sampler = RejectionSampler()
    full = sampler(
        draft_token_ids=draft,
        num_draft_tokens=num_draft,
        target_logits=logits,
        bonus_token_ids=bonus,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=t,
        temperatures=temps.expand(t, 1),
        top_k=top_k.expand(t, 1),
        top_p=top_p.expand(t, 1),
        accept_u=accept_u,
        recover_u=torch.rand(logits.shape, device=device),
        do_sampling=True,
    )
    vals, idx = _chunked_topk(logits, WIDTH)
    sparse = sampler.forward_sparse(
        draft_token_ids=draft,
        num_draft_tokens=num_draft,
        target_vals=vals,
        target_idx=idx,
        bonus_token_ids=bonus,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=t,
        temperatures=temps,
        top_k=top_k,
        top_p=top_p,
        accept_u=accept_u,
        recover_u=torch.rand((t, WIDTH), device=device),
    )
    return full.cpu(), sparse.cpu()


def test_accept_decisions_match_full(device):
    torch.manual_seed(3)
    t = 3
    logits = (torch.randn(t, VOCAB) * 3.0).to(device)
    temps = torch.tensor([[0.6]], device=device)
    top_k = torch.tensor([[20]], dtype=torch.int32, device=device)
    top_p = torch.tensor([[0.95]], device=device)
    probs = _full_probs(
        logits, temps.expand(t, 1), top_k.expand(t, 1), top_p.expand(t, 1)
    )
    order = torch.argsort(probs, dim=-1, descending=True)
    for rank in (0, 1, 3, 30):  # rank 30 is outside top-k: p == 0
        draft = order[:, rank].to(torch.int32)
        p = probs.gather(-1, draft.long().unsqueeze(-1)).squeeze(-1)
        for scale in (0.5, 1.5):
            accept_u = (p * scale).clamp(min=1e-3, max=0.999)
            num_draft = torch.tensor([t], dtype=torch.int32, device=device)
            full, sparse = _run_both(
                logits, draft, num_draft, temps, top_k, top_p, accept_u, device
            )
            n_full = int((full != PLACEHOLDER_TOKEN_ID).sum())
            n_sparse = int((sparse != PLACEHOLDER_TOKEN_ID).sum())
            assert n_full == n_sparse
            # Tokens agree up to the first rejection (the recovered token is a
            # random draw on either side).
            assert torch.equal(full[0, : n_full - 1], sparse[0, : n_full - 1])


def test_recovery_distribution_is_the_residual(device):
    torch.manual_seed(4)
    n = 8192
    row = torch.randn(1, VOCAB) * 2.0
    logits = row.expand(n, VOCAB).contiguous().to(device)
    temps = torch.full((n, 1), 0.6, device=device)
    top_k = torch.full((n, 1), 5, dtype=torch.int32, device=device)
    top_p = torch.ones((n, 1), device=device)
    p = _full_probs(logits[:1], temps[:1], top_k[:1], top_p[:1]).cpu()[0]
    top = int(torch.argmax(p))
    # Every row: one draft (the target's top token), always rejected.
    draft = torch.full((n,), top, dtype=torch.int32, device=device)
    vals, idx = _chunked_topk(logits, WIDTH)
    out = (
        RejectionSampler()
        .forward_sparse(
            draft_token_ids=draft,
            num_draft_tokens=torch.ones(n, dtype=torch.int32, device=device),
            target_vals=vals,
            target_idx=idx,
            bonus_token_ids=torch.zeros(n, dtype=torch.int32, device=device),
            segment_ids=torch.arange(n, dtype=torch.int64, device=device),
            group_indices=torch.zeros(n, dtype=torch.int64, device=device),
            max_draft_tokens=1,
            temperatures=temps,
            top_k=top_k,
            top_p=top_p,
            accept_u=torch.ones(n, device=device),
            recover_u=torch.rand((n, WIDTH), device=device),
        )
        .cpu()
    )
    recovered = out[:, 0]
    residual = p.clone()
    residual[top] = 0
    residual /= residual.sum()
    freq = torch.bincount(recovered.long(), minlength=VOCAB).float() / n
    assert freq[residual == 0].sum() == 0
    sigma = torch.sqrt(residual * (1 - residual) / n)
    assert torch.all((freq - residual).abs() <= 5 * sigma + 1e-6)


def test_greedy_row_recovers_argmax(device):
    logits = torch.full((1, VOCAB), -5.0)
    logits[0, 11] = 2.0
    logits[0, 12] = 1.9
    logits = logits.to(device)
    full, sparse = _run_both(
        logits,
        torch.tensor([12], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.zeros((1, 1), device=device),
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.ones((1, 1), device=device),
        torch.tensor([0.01], device=device),
        device,
    )
    assert torch.equal(sparse, torch.tensor([[11, -1]], dtype=torch.int32))
    assert torch.equal(full, sparse)
