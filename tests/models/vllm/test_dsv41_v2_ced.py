# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeepSeek-V4.1 v2 CED host replay planning (TD7.11)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import (
    DecoderReplayPlan,
    plan_decoder_replay,
    sync_decoder_replay_plan,
)


def test_plan_decoder_replay_none_and_validation_errors() -> None:
    """Verify None returns during warmup and ValueError guards on invalid metadata."""
    assert (
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=None,
                seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
                prompt_lens_cpu=torch.tensor([1], dtype=torch.int32),
            ),
            512,
            4,
            (4, 8),
        )
        is None
    )
    assert (
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens_cpu=None,
                prompt_lens_cpu=torch.tensor([1], dtype=torch.int32),
            ),
            512,
            4,
            (4, 8),
        )
        is None
    )
    assert (
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
                prompt_lens_cpu=None,
            ),
            512,
            4,
            (4, 8),
        )
        is None
    )

    with pytest.raises(ValueError, match="Invalid decoder replay scheduling metadata"):
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
                prompt_lens_cpu=torch.tensor([1], dtype=torch.int32),
            ),
            0,
            4,
            (4,),
        )

    with pytest.raises(ValueError, match="Invalid decoder replay scheduling metadata"):
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=(0, 1, 2),
                seq_lens_cpu=(1,),
                prompt_lens_cpu=(1,),
            ),
            512,
            4,
            (4,),
        )

    with pytest.raises(ValueError, match="Invalid decoder replay scheduling metadata"):
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=(0, 1),
                seq_lens_cpu=(1,),
                prompt_lens_cpu=(1, 1),
            ),
            512,
            4,
            (4,),
        )

    with pytest.raises(ValueError, match="Query boundaries exceed"):
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([1, 2], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
                prompt_lens_cpu=torch.tensor([2], dtype=torch.int32),
            ),
            512,
            4,
            (4,),
        )

    with pytest.raises(ValueError, match="Query boundaries exceed"):
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
                prompt_lens_cpu=torch.tensor([5], dtype=torch.int32),
            ),
            512,
            4,
            (4,),
        )

    with pytest.raises(ValueError, match="Invalid scheduled query or prompt length"):
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([0, 3, 2], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([3, 3], dtype=torch.int32),
                prompt_lens_cpu=torch.tensor([3, 3], dtype=torch.int32),
            ),
            512,
            4,
            (4,),
        )

    with pytest.raises(ValueError, match="Invalid scheduled query or prompt length"):
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
                prompt_lens_cpu=torch.tensor([1], dtype=torch.int32),
            ),
            512,
            4,
            (4,),
        )

    with pytest.raises(ValueError, match="Invalid scheduled query or prompt length"):
        plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
                prompt_lens_cpu=torch.tensor([-1], dtype=torch.int32),
            ),
            512,
            4,
            (4,),
        )


@pytest.mark.parametrize("num_reqs", [48, 64, 128, 224])
def test_plan_decoder_replay_tensor_decode_identity_fast_path(num_reqs: int) -> None:
    """Verify CPU Tensor unboxing and zero-skip fast path across decode batches."""
    tokens_per_req = 6  # 1 target + K=5 speculative draft tokens
    total_tokens = num_reqs * tokens_per_req
    compile_sizes = (16, 32, 64, 128, 256, 288, 384, 512, 768, 1024, 1344)
    prompts = [256 if i % 2 == 0 else 1024 for i in range(num_reqs)]
    seq_lens = [p + 64 for p in prompts]

    meta_tensor = SimpleNamespace(
        query_start_loc_cpu=torch.arange(
            0, total_tokens + 1, tokens_per_req, dtype=torch.int32
        ),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        prompt_lens_cpu=torch.tensor(prompts, dtype=torch.int32),
    )
    meta_tuple = SimpleNamespace(
        query_start_loc_cpu=tuple(range(0, total_tokens + 1, tokens_per_req)),
        seq_lens_cpu=tuple(seq_lens),
        prompt_lens_cpu=tuple(prompts),
    )

    plan_tensor = plan_decoder_replay(meta_tensor, 512, total_tokens, compile_sizes)
    plan_tuple = plan_decoder_replay(meta_tuple, 512, total_tokens, compile_sizes)
    assert plan_tensor is not None
    assert plan_tensor == plan_tuple
    assert plan_tensor.is_identity is True
    assert isinstance(plan_tensor.is_identity, bool)
    assert plan_tensor.padded_num_tokens == total_tokens
    assert plan_tensor.indices == tuple(range(total_tokens))
    assert plan_tensor.query_start_loc == tuple(
        range(0, total_tokens + 1, tokens_per_req)
    )
    assert all(isinstance(r, int) for r in plan_tensor.replay_starts)
    expected_replay = tuple(0 if p <= 512 else p - 512 for p in prompts)
    assert plan_tensor.replay_starts == expected_replay


def test_plan_decoder_replay_skip_and_bucket_selection() -> None:
    """Verify partial skip, full skip, zero-token boundary, and bucket selection."""
    # 1. Partial skip across two requests + smaller compile bucket selection
    meta_partial = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 6, 10], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([600, 100], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([1024, 80], dtype=torch.int32),
    )
    # Req 0: count=6, length=514, prompt=1024, window=512 -> replay_start=512,
    # length - count = 508 < 512 -> skip=4.
    meta_partial_skip = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 6, 10], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([514, 100], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([1024, 80], dtype=torch.int32),
    )
    plan_partial = plan_decoder_replay(meta_partial_skip, 512, 16, (2, 8, 16))
    assert plan_partial == DecoderReplayPlan(
        indices=(4, 5, 6, 7, 8, 9),
        query_start_loc=(0, 2, 6),
        replay_starts=(512, 0),
        padded_num_tokens=8,
        is_identity=False,
    )

    # 2. Zero-skip but padded bucket smaller than num_tokens (is_identity=False)
    plan_padded = plan_decoder_replay(meta_partial, 512, 16, (4, 12, 16))
    assert plan_padded == DecoderReplayPlan(
        indices=tuple(range(10)),
        query_start_loc=(0, 6, 10),
        replay_starts=(512, 0),
        padded_num_tokens=12,
        is_identity=False,
    )

    # 3. Full skip (kept == 0) + zero-count request with replay_start > length - count
    meta_full_skip = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 4, 4], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([1000, 1000], dtype=torch.int32),
    )
    plan_empty = plan_decoder_replay(meta_full_skip, 512, 8, (4, 8))
    assert plan_empty == DecoderReplayPlan(
        indices=(),
        query_start_loc=(0, 0, 0),
        replay_starts=(488, 488),
        padded_num_tokens=0,
        is_identity=False,
    )

    # 4. Empty request list (bounds=[0]) and empty compile_sizes
    meta_zero_reqs = SimpleNamespace(
        query_start_loc_cpu=[0],
        seq_lens_cpu=[],
        prompt_lens_cpu=[],
    )
    plan_zero_reqs = plan_decoder_replay(meta_zero_reqs, 512, 4, ())
    assert plan_zero_reqs == DecoderReplayPlan(
        indices=(),
        query_start_loc=(0,),
        replay_starts=(),
        padded_num_tokens=0,
        is_identity=False,
    )


def test_sync_decoder_replay_plan_branches() -> None:
    """Verify DP synchronization of DecoderReplayPlan across ranks."""
    base_plan = DecoderReplayPlan(
        indices=(0, 1),
        query_start_loc=(0, 2),
        replay_starts=(0,),
        padded_num_tokens=2,
        is_identity=False,
    )
    assert sync_decoder_replay_plan(base_plan, None, 4, None) is base_plan

    def _fake_max_reduce(peer_real: int, peer_padded: int):
        def _all_reduce(tensor: torch.Tensor, op=None, group=None) -> None:
            tensor[0] = max(int(tensor[0].item()), peer_real)
            tensor[1] = max(int(tensor[1].item()), peer_padded)

        return _all_reduce

    with patch("torch.distributed.all_reduce", side_effect=_fake_max_reduce(0, 0)):
        assert sync_decoder_replay_plan(None, None, 4, "cpu_grp") is None

    meta_2req = SimpleNamespace(seq_lens=torch.tensor([1, 1]))
    with patch("torch.distributed.all_reduce", side_effect=_fake_max_reduce(1, 0)):
        assert sync_decoder_replay_plan(None, meta_2req, 4, "cpu_grp") == (
            DecoderReplayPlan(
                indices=(),
                query_start_loc=(0, 0, 0),
                replay_starts=(0, 0),
                padded_num_tokens=0,
                is_identity=False,
            )
        )

    with patch("torch.distributed.all_reduce", side_effect=_fake_max_reduce(1, 4)):
        assert sync_decoder_replay_plan(None, None, 8, "cpu_grp") == (
            DecoderReplayPlan(
                indices=(),
                query_start_loc=(0, 0),
                replay_starts=(0,),
                padded_num_tokens=4,
                is_identity=False,
            )
        )

    with patch("torch.distributed.all_reduce", side_effect=_fake_max_reduce(1, 2)):
        assert sync_decoder_replay_plan(base_plan, meta_2req, 4, "cpu_grp") is base_plan

    meta_cpu = SimpleNamespace(query_start_loc_cpu=(0, 2))
    with patch("torch.distributed.all_reduce", side_effect=_fake_max_reduce(1, 4)):
        assert sync_decoder_replay_plan(base_plan, meta_cpu, 4, "cpu_grp") == (
            DecoderReplayPlan(
                indices=(0, 1),
                query_start_loc=(0, 2),
                replay_starts=(0,),
                padded_num_tokens=4,
                is_identity=True,
            )
        )
