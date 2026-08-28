"""Tests for page-local prefix-cache copy-on-write block copies."""

from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner_with_raw_tensors(raw_tensors):
    runner = object.__new__(TPUModelRunner)
    runner.requests = {}
    runner.kv_cache_raw_tensors = raw_tensors
    runner._install_spec_token_room_guard = lambda: None
    return runner


def test_update_states_routes_pooled_copies_away_from_generic_path(
        monkeypatch):
    base_outputs = []
    events = []
    result_marker = object()

    def base_update(_runner, scheduler_output):
        events.append("base_zero_and_state")
        base_outputs.append(scheduler_output)
        return result_marker

    monkeypatch.setattr(GPUModelRunner, "_update_states", base_update)
    runner = _runner_with_raw_tensors([torch.empty(1)])
    local_copies = []

    def apply_page_local(copies):
        events.append("page_local_copy")
        local_copies.append(copies)

    runner._apply_kv_cache_block_copies = apply_page_local

    scheduler_output = SchedulerOutput.make_empty()
    copies = [KVCacheBlockCopy(src_block_id=6, dst_block_id=7)]
    scheduler_output.kv_cache_block_copies = copies
    scheduler_output.new_block_ids_to_zero = [7]

    result = TPUModelRunner._update_states(runner, scheduler_output)

    assert result is result_marker
    assert len(base_outputs) == 1
    assert base_outputs[0] is not scheduler_output
    assert base_outputs[0].kv_cache_block_copies is None
    assert base_outputs[0].new_block_ids_to_zero == [7]
    assert scheduler_output.kv_cache_block_copies == copies
    assert local_copies == [copies]
    assert events == ["base_zero_and_state", "page_local_copy"]


def test_update_states_keeps_generic_copy_for_non_pooled_layout(monkeypatch):
    base_outputs = []
    monkeypatch.setattr(
        GPUModelRunner, "_update_states", lambda _runner, scheduler_output:
        base_outputs.append(scheduler_output))
    runner = _runner_with_raw_tensors([])
    local_copies = []
    runner._apply_kv_cache_block_copies = local_copies.append

    scheduler_output = SchedulerOutput.make_empty()
    copies = [KVCacheBlockCopy(src_block_id=6, dst_block_id=7)]
    scheduler_output.kv_cache_block_copies = copies

    TPUModelRunner._update_states(runner, scheduler_output)

    assert base_outputs == [scheduler_output]
    assert local_copies == []


def test_page_local_copy_expands_split_and_updates_every_pool(monkeypatch):
    raw_tensors = [
        torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4),
        torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4) + 1000,
    ]
    before = [raw.clone() for raw in raw_tensors]
    fake = SimpleNamespace(
        kv_cache_raw_tensors=raw_tensors,
        _pool_block_split=3,
        _pad_to_bucket=TPUModelRunner._pad_to_bucket,
        device="cpu",
    )
    calls = []

    def cpu_page_copy(raw, src, dst):
        calls.append((id(raw), src.tolist(), dst.tolist()))
        raw[dst.long()] = raw[src.long()]

    monkeypatch.setattr(runner_mod, "copy_mamba_state_blocks", cpu_page_copy)

    TPUModelRunner._apply_kv_cache_block_copies(
        fake, [KVCacheBlockCopy(src_block_id=6, dst_block_id=7)])

    assert len(calls) == 2
    assert [call[0] for call in calls] == [id(raw) for raw in raw_tensors]
    assert calls[0][1] == [18, 19, 20, 0, 0, 0, 0, 0]
    assert calls[0][2] == [21, 22, 23, 0, 0, 0, 0, 0]
    for raw, old in zip(raw_tensors, before):
        torch.testing.assert_close(raw[21:24], old[18:21])
        torch.testing.assert_close(raw[24], old[24])
