# SPDX-License-Identifier: Apache-2.0
"""cached steady-state CED replay plan and O(1) replay-start hits.

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


def _md(bounds, lengths, prompts, hits=None, tensors=False):
    conv = (lambda x: torch.tensor(x, dtype=torch.int32)) if tensors else list
    return SimpleNamespace(
        query_start_loc_cpu=conv(bounds),
        seq_lens_cpu=conv(lengths),
        prompt_lens_cpu=conv(prompts),
        replay_starts_cpu=hits,
    )


def _full(md, window, num_tokens):
    orig = ced._static_no_skip_spec
    try:
        ced._static_no_skip_spec = lambda *_args, **_kwargs: None
        ced._LAST_NO_SKIP_CACHE[:] = [None, None, None, None, None]
        return ced.plan_decoder_replay(md, window, num_tokens, list(COMPILE_SIZES))
    finally:
        ced._static_no_skip_spec = orig
        ced._LAST_NO_SKIP_CACHE[:] = [None, None, None, None, None]


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


def test_prefix_replay_and_scatter_helpers():
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, swa_bounded_replay=True
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    assert isinstance(ced.prefix_replay_enabled(cfg), bool)
    inv = ced._ced_inverse_rows((0, 2), padded_rows=2, total_rows=4, device="cpu")
    out = ced._scatter_decoder_rows(torch.tensor([[1.0], [2.0]]), inv)
    assert torch.equal(out, torch.tensor([[1.0], [0.0], [2.0], [0.0]]))


def test_last_no_skip_identity_cache_and_tuple_reuse(monkeypatch):
    bounds = (0, 6, 12, 18)
    prompts = (512, 512, 512)
    hits = (0, 0, 0)
    md = SimpleNamespace(
        query_start_loc_cpu=bounds,
        seq_lens_cpu=(1024, 1024, 1024),
        prompt_lens_cpu=prompts,
        replay_starts_cpu=hits,
    )
    orig_spec = ced._static_no_skip_spec
    spec_calls = []

    def _wrapped_spec(*args, **kwargs):
        spec_calls.append(args)
        return orig_spec(*args, **kwargs)

    monkeypatch.setattr(ced, "_static_no_skip_spec", _wrapped_spec)
    ced._LAST_NO_SKIP_CACHE[:] = [None, None, None, None, None]

    p1 = ced.plan_decoder_replay(md, 512, 128, (32, 64, 96, 128))
    assert len(spec_calls) == 1
    assert spec_calls[0][0] is bounds
    assert spec_calls[0][1] is prompts
    assert spec_calls[0][2] is hits

    # Second steady-state decode step: _LAST_NO_SKIP_CACHE hits by identity,
    # skipping _static_no_skip_spec lookup.
    md.seq_lens_cpu = (1030, 1030, 1030)
    p2 = ced.plan_decoder_replay(md, 512, 128, (32, 64, 96, 128))
    assert p2 is p1
    assert len(spec_calls) == 1

    # Changing scalar/bucket parameters misses _LAST_NO_SKIP_CACHE.
    p3 = ced.plan_decoder_replay(md, 256, 128, (32, 64, 96, 128))
    assert len(spec_calls) == 2
    assert p3.replay_starts == (256, 256, 256)

    # Full-path no-skip with tuple bounds reuses the tuple directly.
    monkeypatch.setattr(ced, "_static_no_skip_spec", lambda *_a, **_kw: None)
    ced._LAST_NO_SKIP_CACHE[:] = [None, None, None, None, None]
    p_slow = ced.plan_decoder_replay(md, 512, 128, (32, 64, 96, 128))
    assert p_slow.query_start_loc is bounds


def test_max_min_len_short_circuit_and_heterogeneous_fallback(monkeypatch):
    """Verify _static_no_skip_spec returns (min_lens, plan, max_min_len)
    and plan_decoder_replay short-circuits via min(lengths_list, default=0) >= spec[2]
    while falling back to all(map(operator.ge, ...)) on heterogeneous lengths."""
    import operator

    bounds = (0, 6, 12, 18)
    prompts = (100, 1024, 512)  # window=512 -> replay_starts=(0, 512, 0)
    # min_lens = (6, 518, 6), max_min_len = 518
    spec = ced._static_no_skip_spec(bounds, prompts, (0, 0, 0), 512, 32, (18, 32))
    assert spec is not None and len(spec) == 3
    assert spec[0] == (6, 518, 6)
    assert spec[2] == 518

    ge_calls = 0
    orig_ge = operator.ge

    def spy_ge(a, b):
        nonlocal ge_calls
        ge_calls += 1
        return orig_ge(a, b)

    monkeypatch.setattr(ced.operator, "ge", spy_ge)

    # 1. Homogeneous/steady-state: min(lengths_list) = 600 >= 518 -> 0 operator.ge calls
    md = SimpleNamespace(
        query_start_loc_cpu=bounds,
        seq_lens_cpu=(600, 700, 800),
        prompt_lens_cpu=prompts,
        replay_starts_cpu=(0, 0, 0),
    )
    p_fast = ced.plan_decoder_replay(md, 512, 32, (18, 32))
    assert p_fast is spec[1]
    assert ge_calls == 0

    # 2. Heterogeneous all-kept: req 0 has length 50 (< 518, >= 6), req 1 has 600
    # min(lengths_list) = 50 < 518 -> falls back to all(map(operator.ge, ...))
    md.seq_lens_cpu = (50, 600, 80)
    p_hetero = ced.plan_decoder_replay(md, 512, 32, (18, 32))
    assert p_hetero is spec[1]
    assert ge_calls == 3

    # 3. Skip needed: req 1 has length 514 (< 518) -> falls through to full skip path
    md.seq_lens_cpu = (50, 514, 80)
    p_skip = ced.plan_decoder_replay(md, 512, 32, (18, 32))
    assert p_skip is not spec[1]
    assert p_skip.indices == (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17)
