# SPDX-License-Identifier: Apache-2.0
"""TD7.23: the numpy ``plan_decoder_replay`` equals the pre-TD7.23 Python
loop (embedded below) on random and edge-case schedules."""

import random
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ced
from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import (
    DecoderReplayPlan,
    plan_decoder_replay,
)


def _reference_plan(metadata, window, num_tokens, compile_sizes):
    bounds = metadata.query_start_loc_cpu
    lengths = metadata.seq_lens_cpu
    prompts = metadata.prompt_lens_cpu
    if bounds is None or lengths is None or prompts is None:
        return None
    bounds_list = bounds.tolist() if isinstance(bounds, torch.Tensor) else bounds
    lengths_list = lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths
    prompts_list = prompts.tolist() if isinstance(prompts, torch.Tensor) else prompts
    if (
        window < 1
        or len(bounds_list) != len(lengths_list) + 1
        or len(prompts_list) != len(lengths_list)
    ):
        raise ValueError("Invalid decoder replay scheduling metadata")
    if bounds_list[0] != 0 or bounds_list[-1] > num_tokens:
        raise ValueError("Query boundaries exceed the input token bucket")
    replay_starts = []
    any_skipped = False
    for start, end, length, prompt in zip(
        bounds_list, bounds_list[1:], lengths_list, prompts_list, strict=False
    ):
        count = end - start
        if count < 0 or length < count or prompt < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        replay_start = prompt - window if prompt > window else 0
        replay_starts.append(replay_start)
        if replay_start > length - count and count > 0:
            any_skipped = True
    if not any_skipped:
        kept = bounds_list[-1]
        indices_tuple = tuple(range(kept))
        output_bounds_tuple = tuple(bounds_list)
    else:
        indices = []
        output_bounds = [0]
        for start, end, length, replay_start in zip(
            bounds_list, bounds_list[1:], lengths_list, replay_starts, strict=False
        ):
            count = end - start
            skip = min(count, max(0, replay_start - (length - count)))
            indices.extend(range(start + skip, end))
            output_bounds.append(len(indices))
        kept = len(indices)
        indices_tuple = tuple(indices)
        output_bounds_tuple = tuple(output_bounds)
    if kept == 0:
        padded = 0
    else:
        padded = int(num_tokens)
        for size in compile_sizes:
            n = int(size)
            if kept <= n < padded:
                padded = n
    is_identity = kept > 0 and kept == bounds_list[-1] and padded == num_tokens
    return DecoderReplayPlan(
        indices_tuple,
        output_bounds_tuple,
        tuple(replay_starts),
        padded,
        is_identity=is_identity,
    )


def _meta(bounds, lengths, prompts, as_tensor, dtype=torch.int32):
    if as_tensor:
        return SimpleNamespace(
            query_start_loc_cpu=torch.tensor(bounds, dtype=dtype),
            seq_lens_cpu=torch.tensor(lengths, dtype=dtype),
            prompt_lens_cpu=torch.tensor(prompts, dtype=dtype),
        )
    return SimpleNamespace(
        query_start_loc_cpu=list(bounds),
        seq_lens_cpu=tuple(lengths),
        prompt_lens_cpu=list(prompts),
    )


def _outcome(fn, *args):
    try:
        return fn(*args)
    except ValueError as e:
        return ("ValueError", str(e))


def _assert_same_plan(got, want):
    assert got == want
    if isinstance(want, DecoderReplayPlan):
        for field in ("indices", "query_start_loc", "replay_starts"):
            assert all(type(v) is int for v in getattr(got, field)), field
        assert type(got.padded_num_tokens) is int
        assert type(got.is_identity) is bool


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("as_tensor", [True, False])
def test_random_schedules_match_reference(seed, as_tensor):
    rng = random.Random(seed)
    window = rng.choice([1, 4, 64, 128])
    reqs = rng.randint(0, 12)
    counts = [rng.choice([0, 1, 2, 6, 17]) for _ in range(reqs)]
    bounds = [0]
    for c in counts:
        bounds.append(bounds[-1] + c)
    prompts = [rng.choice([0, 3, window, window + 1, 500, 100_000]) for _ in counts]
    lengths = [c + rng.choice([0, 1, 5, 130, 100_200]) for c in counts]
    num_tokens = bounds[-1] + rng.choice([0, 0, 3, 40])
    sizes = tuple(sorted(rng.sample(range(1, 300), rng.randint(0, 6))))
    meta = _meta(bounds, lengths, prompts, as_tensor)
    args = (meta, window, num_tokens, sizes)
    _assert_same_plan(
        _outcome(plan_decoder_replay, *args), _outcome(_reference_plan, *args)
    )


@pytest.mark.parametrize("as_tensor", [True, False])
@pytest.mark.parametrize(
    "bounds,lengths,prompts,window,num_tokens",
    [
        ([0, 2, 1], [3, 3], [1, 1], 4, 8),  # negative count
        ([0, 4], [3], [1], 4, 8),  # length < count
        ([0, 1], [3], [-1], 4, 8),  # negative prompt
        ([1, 2], [3], [1], 4, 8),  # bounds[0] != 0
        ([0, 9], [9], [1], 4, 8),  # exceeds bucket
        ([0, 1], [3, 3], [1], 4, 8),  # length mismatch
        ([0, 1], [3], [1, 1], 4, 8),  # prompt mismatch
        ([0, 1], [3], [1], 0, 8),  # window < 1
    ],
)
def test_invalid_metadata_raises_like_reference(
    as_tensor, bounds, lengths, prompts, window, num_tokens
):
    meta = _meta(bounds, lengths, prompts, as_tensor)
    args = (meta, window, num_tokens, (4,))
    got = _outcome(plan_decoder_replay, *args)
    assert got == _outcome(_reference_plan, *args)
    assert got[0] == "ValueError"


def test_int64_and_list_compile_sizes_match_reference():
    meta = _meta([0, 6, 12], [700, 20], [1000, 10], True, dtype=torch.int64)
    for sizes in ([8, 16], (8, 16), range(4, 20, 4)):
        _assert_same_plan(
            plan_decoder_replay(meta, 512, 16, sizes),
            _reference_plan(meta, 512, 16, sizes),
        )


def test_decode_steady_state_identity_shares_cached_indices():
    meta = _meta(list(range(0, 289, 6)), [100_500] * 48, [100_000] * 48, True)
    a = plan_decoder_replay(meta, 128, 288, (64, 288))
    b = plan_decoder_replay(meta, 128, 288, (64, 288))
    assert a == _reference_plan(meta, 128, 288, (64, 288))
    assert a.is_identity
    assert a.indices is b.indices  # lru_cache reuse, no per-step allocation


def test_none_metadata_short_circuits():
    meta = SimpleNamespace(
        query_start_loc_cpu=None, seq_lens_cpu=[1], prompt_lens_cpu=[1]
    )
    assert plan_decoder_replay(meta, 4, 4, ()) is None


def test_padded_bucket_cache_matches_loop():
    ced._padded_bucket.cache_clear()
    assert ced._padded_bucket(0, 16, (4, 8)) == 0
    assert ced._padded_bucket(5, 16, (4, 8, 12)) == 8
    assert ced._padded_bucket(13, 16, (4, 8, 12)) == 16
    assert ced._padded_bucket(13, 16, ()) == 16
