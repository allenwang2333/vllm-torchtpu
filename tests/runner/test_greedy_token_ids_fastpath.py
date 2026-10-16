# SPDX-License-Identifier: Apache-2.0
"""Greedy token-id fast path for spec-decode verify (vocab-parallel argmax)."""

from __future__ import annotations

import types
from unittest.mock import patch

import torch

from vllm_torchtpu.runner import tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

TP, T, V = 8, 37, 8 * 160


def _shard_pairs(logits: torch.Tensor) -> torch.Tensor:
    """Emulate LogitsProcessor.get_top_tokens's per-rank reduction + all-gather."""
    shard = logits.shape[-1] // TP
    pairs = []
    for r in range(TP):
        local = logits[:, r * shard : (r + 1) * shard]
        vals, idx = local.max(dim=-1)
        pairs.append(torch.stack([vals.float(), (idx + r * shard).float()], dim=-1))
    return torch.cat(pairs, dim=-1)  # [T, 2 * TP], rank-major


def _reduce(gathered: torch.Tensor) -> torch.Tensor:
    gathered = gathered.view(gathered.shape[0], TP, 2)
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(dim=-1, index=max_rank_idx).squeeze(-1).long()


def test_vocab_parallel_argmax_matches_global_argmax_including_ties():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(T, V, generator=g).to(torch.bfloat16)
    # Plant exact ties inside one shard, across shards, and on the first column.
    logits[0, 5] = logits[0, 9] = 40.0
    logits[1, 170] = logits[1, 900] = 40.0
    logits[2, 0] = logits[2, 1279] = 40.0
    logits[3, :] = 1.0
    assert torch.equal(_reduce(_shard_pairs(logits)), torch.argmax(logits, dim=-1))


def test_fastpath_gating():
    runner = object.__new__(TPUModelRunner)
    runner.is_pooling_model = False
    runner.model = types.SimpleNamespace(compute_greedy_token_ids=lambda h: h)
    runner.input_batch = types.SimpleNamespace(all_greedy=True, max_num_logprobs=None)
    sched = types.SimpleNamespace(has_structured_output_requests=False)
    md = object()
    with patch.object(runner_mod.envs, "TPU_GREEDY_TOKEN_IDS_FASTPATH", True):
        assert runner._greedy_token_ids_fastpath(sched, md)
        assert not runner._greedy_token_ids_fastpath(sched, None)
        runner.input_batch.all_greedy = False
        assert not runner._greedy_token_ids_fastpath(sched, md)
        runner.input_batch.all_greedy = True
        runner.input_batch.max_num_logprobs = 5
        assert not runner._greedy_token_ids_fastpath(sched, md)
        runner.input_batch.max_num_logprobs = None
        sched.has_structured_output_requests = True
        assert not runner._greedy_token_ids_fastpath(sched, md)
        sched.has_structured_output_requests = False
        runner.model = types.SimpleNamespace()
        assert not runner._greedy_token_ids_fastpath(sched, md)
    runner.model = types.SimpleNamespace(compute_greedy_token_ids=lambda h: h)
    with patch.object(runner_mod.envs, "TPU_GREEDY_TOKEN_IDS_FASTPATH", False):
        assert not runner._greedy_token_ids_fastpath(sched, md)


def test_verify_chunk_routes_token_ids_to_id_prelude():
    runner = object.__new__(TPUModelRunner)
    calls = {}

    def ids_prelude(token_ids, bonus_idx, target_idx):
        calls["ids"] = True
        return token_ids[bonus_idx], token_ids[target_idx]

    def logits_prelude(*a):
        raise AssertionError("logits prelude must not run on token ids")

    runner.spec_bonus_and_target_ids = ids_prelude
    runner.spec_bonus_and_target_logits = logits_prelude
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=2)
    captured = {}

    def sampler(**kw):
        captured.update(kw)
        return torch.zeros(2, 3, dtype=torch.int32)

    sampler.synthetic_mode = False  # type: ignore[attr-defined]
    runner.rejection_sampler = sampler  # type: ignore[assignment]
    md = types.SimpleNamespace(
        bonus_logits_indices=torch.tensor([2, 5]),
        target_logits_indices=torch.tensor([0, 1, 3, 4]),
        draft_token_ids=torch.zeros(4, dtype=torch.int32),
        draft_lengths=torch.tensor([2, 2]),
        segment_ids=torch.tensor([0, 0, 1, 1]),
        group_indices=torch.zeros(4, dtype=torch.int32),
    )
    token_ids = torch.arange(100, 106)
    out = runner._sample_spec_verify_chunk(token_ids, md, None, None, 0, 2, True, None)
    assert calls.get("ids") and out.shape == (2, 3)
    assert captured["bonus_token_ids"].tolist() == [102, 105]
    assert captured["target_logits"].tolist() == [100, 101, 103, 104]
