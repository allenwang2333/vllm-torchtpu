# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_env_flag_parsing(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", raising=False)
    assert envs.VLLM_TPU_FAST_TOKEN_SUBSTITUTION is False

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "1")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is True

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "true")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is True

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "0")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is False

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "false")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is False


def test_substitution_when_disabled():
    """When _fast_token_substitution is False, calls _substitute_placeholder_token without caching."""
    runner = SimpleNamespace(
        _fast_token_substitution=False,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([101, 102, 103, 104], dtype=torch.int64)
        ),
    )
    input_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
    cur_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    pre_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    with patch(
        "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token"
    ) as mock_kernel:
        mock_kernel.return_value = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
        out = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices
        )
        assert mock_kernel.called
        assert not hasattr(runner, "_cached_subst_key")
        assert torch.equal(out, torch.tensor([101, 102, 103, 104], dtype=torch.int32))


def test_substitution_enabled_fast_path():
    """When _fast_token_substitution is True, device index tensors are cached on hit and rebuilt on miss."""
    runner = SimpleNamespace(
        _fast_token_substitution=True,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([201, 202, 203, 204], dtype=torch.int64)
        ),
    )
    input_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
    cur_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    pre_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    with patch(
        "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token"
    ) as mock_kernel:
        mock_kernel.return_value = torch.tensor([201, 202, 203, 204], dtype=torch.int32)
        out1 = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices
        )
        assert mock_kernel.call_count == 1
        assert torch.equal(out1, torch.tensor([201, 202, 203, 204], dtype=torch.int32))
        cached_cur = runner._cached_cur_input_indices
        cached_pre = runner._cached_pre_next_tokens_indices

        # Second call with identical indices reuses cached device index tensors
        out2 = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices
        )
        assert mock_kernel.call_count == 2
        assert mock_kernel.call_args.args[1] is cached_cur
        assert mock_kernel.call_args.args[2] is cached_pre
        assert torch.equal(out2, torch.tensor([201, 202, 203, 204], dtype=torch.int32))

        # Third call with different indices misses cache and builds new index tensors
        new_cur_indices = np.array([0, 2], dtype=np.int32)
        new_pre_indices = np.array([1, 3], dtype=np.int32)
        TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, new_cur_indices, new_pre_indices
        )
        assert mock_kernel.call_count == 3
        assert mock_kernel.call_args.args[1] is not cached_cur
        assert mock_kernel.call_args.args[2] is not cached_pre
        assert torch.equal(
            runner._cached_cur_input_indices,
            torch.tensor([0, 2, 1, 3], dtype=torch.int32),
        )
        assert torch.equal(
            runner._cached_pre_next_tokens_indices,
            torch.tensor([1, 3, -1, -1], dtype=torch.int32),
        )


def test_stage_timing_disabled_skips_tpu_events():
    from vllm_torchtpu.runner.stage_timing import RunnerStageTiming

    with patch("vllm_torchtpu.runner.stage_timing.TpuEvent") as mock_event:
        timing = RunnerStageTiming(rank=0, enabled=False)
        with timing.stage("main_forward"):
            pass
        with timing.host_stage("prep"):
            pass
        assert timing.take_samples() == ([], [])
        assert not mock_event.called


def test_tpu_logits_processor_pre_gather_soft_cap():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from vllm_torchtpu.layers.adapter.custom_ops.logits_processor import (
        TpuLogitsProcessor,
    )

    shard_0 = torch.tensor([[15.0, -30.0, 45.0, 0.0]], dtype=torch.bfloat16)
    shard_1 = torch.tensor([[-15.0, 30.0, -45.0, 10.0]], dtype=torch.bfloat16)
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)

    def make_proc(cls):
        p = object.__new__(cls)
        p.logits_as_input = False
        p.soft_cap = 30.0
        p.scale = 0.5
        p.org_vocab_size = 6
        return p

    # Reference upstream LogitsProcessor: gathers raw shards, then applies soft_cap + scale
    ref_proc = make_proc(LogitsProcessor)
    ref_proc._apply_head = lambda head, h, b: shard_0.clone()
    ref_proc._gather_logits = lambda local: torch.cat([local, shard_1.clone()], dim=-1)
    expected = LogitsProcessor.forward(ref_proc, SimpleNamespace(tp_size=2), hidden)

    # TpuLogitsProcessor: applies soft_cap + scale on each rank's shard before gather
    proc = make_proc(TpuLogitsProcessor)
    events = []

    def fake_apply_head(head, h, b):
        events.append(("apply_head", tuple(shard_0.shape)))
        return shard_0.clone()

    def fake_gather(local_logits):
        events.append(("gather", tuple(local_logits.shape)))
        # Simulate rank 1 having also run _apply_soft_cap_and_scale on its local shard
        rank1_local = proc._apply_soft_cap_and_scale(shard_1.clone())
        return torch.cat([local_logits, rank1_local], dim=-1)

    proc._apply_head = fake_apply_head
    proc._gather_logits = fake_gather

    out = proc.forward(SimpleNamespace(tp_size=2), hidden)
    assert events == [("apply_head", (1, 4)), ("gather", (1, 4))]
    assert torch.equal(out, expected)

    # Fallback paths: tp_size == 1, skip_gather=True, logits_as_input=True
    out_tp1 = proc.forward(SimpleNamespace(tp_size=1), hidden)
    ref_tp1 = LogitsProcessor.forward(ref_proc, SimpleNamespace(tp_size=1), hidden)
    assert torch.equal(out_tp1, ref_tp1)

    out_skip = proc.forward(SimpleNamespace(tp_size=2), hidden, skip_gather=True)
    ref_skip = LogitsProcessor.forward(
        ref_proc, SimpleNamespace(tp_size=2), hidden, skip_gather=True
    )
    assert torch.equal(out_skip, ref_skip)

    proc.logits_as_input = True
    ref_proc.logits_as_input = True
    out_in = proc.forward(SimpleNamespace(tp_size=2), shard_0.clone())
    ref_in = LogitsProcessor.forward(ref_proc, SimpleNamespace(tp_size=2), shard_0.clone())
    assert torch.equal(out_in, ref_in)

    # LogitsProcessorWithLoRA compatibility: calls TpuLogitsProcessor.forward unbound
    # with `self` as a wrapper object (not isinstance of TpuLogitsProcessor) that
    # only implements `_get_logits`.
    lora_wrapper = SimpleNamespace(
        logits_as_input=False,
        soft_cap=30.0,
        scale=0.5,
        org_vocab_size=6,
        _get_logits=lambda h, head, b, skip_gather=False: torch.cat(
            [shard_0, shard_1], dim=-1
        )
        + 1.0,
    )
    out_lora = TpuLogitsProcessor.forward(
        lora_wrapper, SimpleNamespace(tp_size=2), hidden
    )
    ref_lora = LogitsProcessor.forward(
        lora_wrapper, SimpleNamespace(tp_size=2), hidden
    )
    assert torch.equal(out_lora, ref_lora)
