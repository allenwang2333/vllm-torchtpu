# SPDX-License-Identifier: Apache-2.0
"""TD8.337: cached steady-state CED replay plan and O(1) replay-start hits.

``plan_decoder_replay`` reuses a cached all-kept plan for a fixed
query/prompt/hit layout while only ``seq_lens`` grow (CPU timeit, N=256:
38.5 -> 11.5 us for list metadata, 49-51 -> 23.6 us for tensors). The cached
plan must equal the full computation; invalid layouts must still raise.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ced
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import DeepseekV41Model

COMPILE_SIZES = (16, 32, 64, 128, 256, 512)


def _md(bounds, lengths, prompts, hits=None, owner=None, tensors=False):
    conv = (lambda x: torch.tensor(x, dtype=torch.int32)) if tensors else list
    return SimpleNamespace(
        query_start_loc_cpu=conv(bounds),
        seq_lens_cpu=conv(lengths),
        prompt_lens_cpu=conv(prompts),
        replay_starts_cpu=hits,
        owner_token_layout=owner,
    )


def _full(md, window, num_tokens):
    # owner_token_layout without owners forces the full path with the same
    # required_kept (= kept) as the owner-free fast path.
    slow = SimpleNamespace(
        **{**vars(md), "owner_token_layout": SimpleNamespace(owners=None)}
    )
    return ced.plan_decoder_replay(slow, window, num_tokens, list(COMPILE_SIZES))


@pytest.mark.parametrize("tensors", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_fast_plan_matches_full_plan(seed, tensors):
    g = torch.Generator().manual_seed(seed)
    n = int(torch.randint(1, 40, (1,), generator=g))
    counts = torch.randint(0, 4, (n,), generator=g).tolist()
    if seed % 2 == 0:
        counts = [1] * n  # steady-state decode
    bounds = [0]
    for c in counts:
        bounds.append(bounds[-1] + c)
    prompts = torch.randint(0, 600, (n,), generator=g).tolist()
    hits = tuple(int(h) for h in torch.randint(0, 3, (n,), generator=g) * 100)
    # Some requests sit right at / below their replay start (skip path).
    lengths = [
        max(c, p + int(torch.randint(-150, 400, (1,), generator=g)))
        for c, p in zip(counts, prompts, strict=True)
    ]
    num_tokens = max(bounds[-1], 1) + 3
    md = _md(bounds, lengths, prompts, hits if seed % 3 else None, tensors=tensors)
    got = ced.plan_decoder_replay(md, 128, num_tokens, COMPILE_SIZES)
    ref = _full(md, 128, num_tokens)
    assert dataclasses.astuple(got) == dataclasses.astuple(ref)


def test_steady_state_reuses_cached_plan_object():
    bounds = list(range(9))
    prompts = [300] * 8
    md = _md(bounds, [1000 + i for i in range(8)], prompts)
    first = ced.plan_decoder_replay(md, 128, 8, COMPILE_SIZES)
    md.seq_lens_cpu = [2000 + i for i in range(8)]  # next decode step
    assert ced.plan_decoder_replay(md, 128, 8, COMPILE_SIZES) is first
    # A request still inside its replay window falls back to the full path.
    md.seq_lens_cpu = [172] + [2000] * 7  # replay_start 172 > 172 - 1
    skipped = ced.plan_decoder_replay(md, 128, 8, COMPILE_SIZES)
    assert skipped is not first
    assert dataclasses.astuple(skipped) == dataclasses.astuple(_full(md, 128, 8))


@pytest.mark.parametrize(
    ("bounds", "lengths", "prompts"),
    [
        ([0, 2, 1], [5, 5], [3, 3]),  # negative count
        ([0, 1, 2], [5, 5], [3, -1]),  # negative prompt
        ([0, 3, 4], [2, 5], [3, 3]),  # length < count
    ],
)
def test_invalid_layouts_still_raise(bounds, lengths, prompts):
    with pytest.raises(ValueError, match="Invalid scheduled query or prompt length"):
        ced.plan_decoder_replay(_md(bounds, lengths, prompts), 4, 8, COMPILE_SIZES)


def test_static_no_skip_spec_rejects_invalid_layout():
    assert ced._static_no_skip_spec((0, 2, 1), (3, 3), (0, 0), 4, 8, ()) is None


def test_replay_starts_cache_identity_and_equality_hits():
    owner = SimpleNamespace()
    pos = torch.zeros(2)
    hits = (0, 5, 7)
    t = DeepseekV41Model._get_ced_replay_starts(owner, hits, pos)
    assert DeepseekV41Model._get_ced_replay_starts(owner, hits, pos) is t
    assert DeepseekV41Model._get_ced_replay_starts(owner, (0, 5, 7), pos) is t
    assert DeepseekV41Model._get_ced_replay_starts(owner, [0, 5, 7], pos) is t
    other = DeepseekV41Model._get_ced_replay_starts(owner, (1, 5, 7), pos)
    assert other is not t and other.tolist() == [1, 5, 7]
