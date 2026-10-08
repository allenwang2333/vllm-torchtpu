# Copyright 2025 Google LLC
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
"""CPU equivalence checks for the TD7.18 rejection sampler rewrites.

The sampler functions are ``torch.compile(backend="tpu")``; these tests call
their eager ``__wrapped__`` bodies on CPU. Each rewritten piece is compared
bitwise against the formula it replaced, and both samplers against a
per-request Python reference on randomized padded batches.
"""

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p

pytestmark = pytest.mark.cpu_test

_greedy = rs._greedy_rejection_sample_with_segment.__wrapped__
_random = rs._random_rejection_sample_with_segment.__wrapped__


def _old_valid_slot_mask(segment_ids, group_indices, num_draft_tokens):
    batch_indices = torch.arange(num_draft_tokens.shape[0]).unsqueeze(1)
    match_mask = segment_ids.unsqueeze(0) == batch_indices
    limits = num_draft_tokens.to(group_indices.dtype).unsqueeze(1)
    return torch.any(match_mask & (group_indices.unsqueeze(0) < limits), dim=0)


def _old_place_bonus(selected, accepted_count, should_get_bonus, bonus):
    batch_size = accepted_count.shape[0]
    selected = torch.cat([selected, torch.full_like(selected[:1], -1)]).clone()
    rows = torch.arange(batch_size, dtype=torch.int64)
    value = torch.where(
        should_get_bonus, bonus.to(torch.int32), selected[rows, accepted_count]
    )
    selected[rows, accepted_count] = value
    return selected[:batch_size]


def _batch(rng, batch_size, max_draft, pad_tokens, sentinel):
    """Padded verify batch: per-request draft counts, static padding slots
    (beyond each count), and trailing tokens on an out-of-range sentinel
    segment."""
    counts = rng.integers(0, max_draft + 1, batch_size)
    seg, grp = [], []
    for b in range(batch_size):
        width = int(counts[b]) + int(rng.integers(0, 2))  # padded draft slot
        width = min(width, max_draft)
        seg += [b] * width
        grp += list(range(width))
    seg += [sentinel] * pad_tokens
    grp += [0] * pad_tokens
    return (
        torch.tensor(counts, dtype=torch.int32),
        torch.tensor(seg, dtype=torch.int64),
        torch.tensor(grp, dtype=torch.int64),
    )


def _reference_rows(draft, accepted, recovered, counts, seg, grp, bonus, width):
    out = np.full((counts.shape[0], width), -1, np.int32)
    for b in range(counts.shape[0]):
        n = int(counts[b])
        slots = {int(grp[t]): t for t in range(seg.shape[0]) if int(seg[t]) == b}
        row = []
        for k in range(n):
            t = slots[k]
            if bool(accepted[t]):
                row.append(int(draft[t]))
            else:
                row.append(int(recovered[t]))
                break
        else:
            row.append(int(bonus[b]))
        out[b, : len(row)] = row
    return torch.from_numpy(out)


@pytest.mark.parametrize("seed", range(20))
def test_valid_slot_mask_matches_2d_any(seed):
    rng = np.random.default_rng(seed)
    batch_size = int(rng.integers(1, 9))
    sentinel = int(rng.choice([-1, batch_size, batch_size + 3]))
    counts, seg, grp = _batch(rng, batch_size, 5, int(rng.integers(0, 4)), sentinel)
    want = _old_valid_slot_mask(seg, grp, counts)
    assert torch.equal(rs._valid_slot_mask(seg, grp, counts), want)


def test_valid_slot_mask_empty_batch():
    seg = torch.tensor([0, 0], dtype=torch.int64)
    grp = torch.tensor([0, 1], dtype=torch.int64)
    counts = torch.zeros(0, dtype=torch.int32)
    got = rs._valid_slot_mask(seg, grp, counts)
    assert torch.equal(got, _old_valid_slot_mask(seg, grp, counts))
    assert got.dtype == torch.bool and not bool(got.any())


@pytest.mark.parametrize("seed", range(20))
def test_place_bonus_tokens_matches_scatter(seed):
    rng = np.random.default_rng(seed)
    batch_size, width = int(rng.integers(1, 9)), int(rng.integers(1, 7))
    selected = torch.tensor(
        rng.integers(-1, 50, (batch_size, width)), dtype=torch.int32
    )
    accepted = torch.tensor(rng.integers(0, width, batch_size), dtype=torch.int64)
    bonus_flag = torch.tensor(rng.integers(0, 2, batch_size).astype(bool))
    bonus = torch.tensor(rng.integers(100, 200, batch_size), dtype=torch.int64)
    want = _old_place_bonus(selected, accepted, bonus_flag, bonus)
    got = rs._place_bonus_tokens(selected, accepted, bonus_flag, bonus)
    assert got.dtype == torch.int32
    assert torch.equal(got, want)


@pytest.mark.parametrize("seed", range(12))
def test_greedy_matches_reference(seed):
    rng = np.random.default_rng(seed)
    batch_size, max_draft, vocab = int(rng.integers(1, 7)), 5, 11
    sentinel = int(rng.choice([-1, batch_size]))
    counts, seg, grp = _batch(rng, batch_size, max_draft, 2, sentinel)
    tokens = seg.shape[0]
    target = torch.tensor(rng.integers(0, vocab, tokens), dtype=torch.int32)
    draft = torch.where(
        torch.tensor(rng.random(tokens) < 0.7),
        target,
        torch.tensor(rng.integers(0, vocab, tokens), dtype=torch.int32),
    )
    bonus = torch.tensor(rng.integers(0, vocab, batch_size), dtype=torch.int32)
    logits = torch.nn.functional.one_hot(target.long(), vocab).float()
    for target_in in (logits, target):
        got = _greedy(
            draft, target_in, counts, bonus, seg, grp, max_draft, None, None, False
        )
        want = _reference_rows(
            draft, draft == target, target, counts, seg, grp, bonus, max_draft + 1
        )
        assert torch.equal(got, want)


@pytest.mark.parametrize("seed", range(12))
def test_random_matches_reference(seed):
    rng = np.random.default_rng(seed)
    batch_size, max_draft, vocab = int(rng.integers(1, 7)), 5, 13
    sentinel = int(rng.choice([-1, batch_size]))
    counts, seg, grp = _batch(rng, batch_size, max_draft, 2, sentinel)
    tokens = seg.shape[0]
    draft = torch.tensor(rng.integers(-1, vocab, tokens), dtype=torch.int32)
    logits = torch.tensor(rng.standard_normal((tokens, vocab)) * 2, dtype=torch.float32)
    temps = torch.tensor(rng.choice([0.0, 0.7, 1.0], (tokens, 1)), dtype=torch.float32)
    top_k = torch.tensor(rng.choice([0, 3], (tokens, 1)), dtype=torch.int32)
    top_p = torch.tensor(rng.choice([1.0, 0.9], (tokens, 1)), dtype=torch.float32)
    accept_u = torch.tensor(rng.random(tokens), dtype=torch.float32)
    recover_u = torch.tensor(rng.random((tokens, vocab)), dtype=torch.float32)
    bonus = torch.tensor(rng.integers(0, vocab, batch_size), dtype=torch.int32)

    got = _random(
        draft,
        logits,
        counts,
        bonus,
        seg,
        grp,
        max_draft,
        temps,
        top_k,
        top_p,
        accept_u,
        recover_u,
        None,
        False,
    )

    probs = torch.softmax(
        apply_top_k_top_p(logits / (temps + rs.TEMPERATURE_EPS), top_k, top_p), -1
    )
    safe = draft.long().clamp(min=0)
    valid = draft >= 0
    accepted = (probs.gather(-1, safe[:, None])[:, 0] >= accept_u) & valid
    mask = torch.nn.functional.one_hot(safe, vocab).to(torch.bool) & valid[:, None]
    residual = torch.where(mask, torch.tensor(0.0), probs)
    finfo = torch.finfo(torch.float32)
    noise = -torch.log(recover_u.clamp(min=finfo.tiny, max=1.0 - finfo.eps))
    recovered = torch.where(
        (residual > 0).any(-1),
        torch.argmax(residual / (noise + finfo.tiny), -1),
        torch.argmax(probs, -1),
    )
    want = _reference_rows(
        draft, accepted, recovered, counts, seg, grp, bonus, max_draft + 1
    )
    assert torch.equal(got, want)


def test_get_segment_info():
    num_draft = torch.tensor([3, 0, 2], dtype=torch.int32)
    seg, grp = rs._get_segment_info(num_draft, 5)
    assert torch.equal(seg, torch.tensor([0, 0, 0, 2, 2]))
    assert torch.equal(grp, torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32))


def test_greedy_and_random_total_tokens_zero():
    batch_size = 2
    max_draft = 3
    counts = torch.zeros(batch_size, dtype=torch.int32)
    bonus = torch.tensor([42, 43], dtype=torch.int32)
    seg = torch.tensor([], dtype=torch.int64)
    grp = torch.tensor([], dtype=torch.int32)
    draft = torch.tensor([], dtype=torch.int32)
    target = torch.empty((0, 10), dtype=torch.float32)

    got_greedy = _greedy(
        draft, target, counts, bonus, seg, grp, max_draft, None, None, False
    )
    assert got_greedy.shape == (batch_size, max_draft + 1)
    assert got_greedy[0, 0] == 42
    assert got_greedy[1, 0] == 43

    temps = torch.empty((0, 1), dtype=torch.float32)
    top_k = torch.empty((0, 1), dtype=torch.int32)
    top_p = torch.empty((0, 1), dtype=torch.float32)
    accept_u = torch.empty((0,), dtype=torch.float32)
    recover_u = torch.empty((0, 10), dtype=torch.float32)
    got_random = _random(
        draft,
        target,
        counts,
        bonus,
        seg,
        grp,
        max_draft,
        temps,
        top_k,
        top_p,
        accept_u,
        recover_u,
        None,
        False,
    )
    assert got_random.shape == (batch_size, max_draft + 1)
    assert got_random[0, 0] == 42
    assert got_random[1, 0] == 43


def test_greedy_and_random_synthetic_mode():
    max_draft = 2
    counts = torch.tensor([2], dtype=torch.int32)
    bonus = torch.tensor([100], dtype=torch.int32)
    seg = torch.tensor([0, 0], dtype=torch.int64)
    grp = torch.tensor([0, 1], dtype=torch.int64)
    draft = torch.tensor([5, 6], dtype=torch.int32)
    target = torch.tensor([10, 11], dtype=torch.int32)
    rates = torch.tensor([0.5, 0.5], dtype=torch.float32)

    # accept_u < rates -> accepted
    accept_u = torch.tensor([0.2, 0.8], dtype=torch.float32)
    got = _greedy(
        draft, target, counts, bonus, seg, grp, max_draft, accept_u, rates, True
    )
    assert got[0, 0] == 5  # accepted
    assert got[0, 1] == 11  # rejected, recovered

    # Random sampler synthetic mode
    vocab = 20
    logits = torch.zeros((2, vocab), dtype=torch.float32)
    temps = torch.ones((2, 1), dtype=torch.float32)
    top_k = torch.zeros((2, 1), dtype=torch.int32)
    top_p = torch.ones((2, 1), dtype=torch.float32)
    recover_u = torch.full((2, vocab), 0.5, dtype=torch.float32)
    got_rand = _random(
        draft,
        logits,
        counts,
        bonus,
        seg,
        grp,
        max_draft,
        temps,
        top_k,
        top_p,
        accept_u,
        recover_u,
        rates,
        True,
    )
    assert got_rand.shape == (1, max_draft + 1)


def test_rejection_sampler_class(monkeypatch):
    from unittest.mock import MagicMock

    sampler = rs.RejectionSampler()
    assert not sampler.synthetic_mode

    mock_cfg = MagicMock()
    mock_cfg.rejection_sample_method = "synthetic"
    mock_cfg.synthetic_acceptance_rates = [0.8, 0.6]
    synth_sampler = rs.RejectionSampler(speculative_config=mock_cfg)
    assert synth_sampler.synthetic_mode
    assert synth_sampler.synthetic_conditional_rates is not None

    with pytest.raises(ValueError, match="Non-greedy rejection sampling requires"):
        sampler(
            draft_token_ids=torch.tensor([1]),
            num_draft_tokens=torch.tensor([1]),
            target_logits=torch.tensor([[1.0]]),
            bonus_token_ids=torch.tensor([2]),
            segment_ids=torch.tensor([0]),
            group_indices=torch.tensor([0]),
            max_draft_tokens=1,
            do_sampling=True,
        )

    with pytest.raises(
        ValueError, match="Synthetic rejection sampling requires accept_u"
    ):
        synth_sampler(
            draft_token_ids=torch.tensor([1]),
            num_draft_tokens=torch.tensor([1]),
            target_logits=torch.tensor([[1.0]]),
            bonus_token_ids=torch.tensor([2]),
            segment_ids=torch.tensor([0]),
            group_indices=torch.tensor([0]),
            max_draft_tokens=1,
            do_sampling=False,
            accept_u=None,
        )

    monkeypatch.setattr(
        rs,
        "_greedy_rejection_sample_with_segment",
        _greedy,
    )
    monkeypatch.setattr(
        rs,
        "_random_rejection_sample_with_segment",
        _random,
    )

    out = sampler(
        draft_token_ids=torch.tensor([1], dtype=torch.int32),
        num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        target_logits=torch.tensor([1], dtype=torch.int32),
        bonus_token_ids=torch.tensor([99], dtype=torch.int32),
        segment_ids=torch.tensor([0], dtype=torch.int64),
        group_indices=torch.tensor([0], dtype=torch.int64),
        max_draft_tokens=1,
        do_sampling=False,
    )
    assert out.shape == (1, 2)

    out_rand = sampler(
        draft_token_ids=torch.tensor([1], dtype=torch.int32),
        num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        target_logits=torch.randn((1, 10), dtype=torch.float32),
        bonus_token_ids=torch.tensor([99], dtype=torch.int32),
        segment_ids=torch.tensor([0], dtype=torch.int64),
        group_indices=torch.tensor([0], dtype=torch.int64),
        max_draft_tokens=1,
        temperatures=torch.ones((1, 1), dtype=torch.float32),
        top_k=torch.zeros((1, 1), dtype=torch.int32),
        top_p=torch.ones((1, 1), dtype=torch.float32),
        accept_u=torch.tensor([0.5], dtype=torch.float32),
        recover_u=torch.full((1, 10), 0.5, dtype=torch.float32),
        do_sampling=True,
    )
    assert out_rand.shape == (1, 2)
