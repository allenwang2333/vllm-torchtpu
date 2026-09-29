# SPDX-License-Identifier: Apache-2.0
"""Connector lifecycle across sampling and speculative proposal."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer import kv_transfer_state

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)


@pytest.mark.parametrize(
    "method,is_async,verify",
    [
        (None, False, False),
        (None, True, False),
        ("mtp", False, False),
        ("mtp", True, False),
        ("mtp", False, True),
        ("mtp", True, True),
        ("ngram", False, False),
    ],
)
@pytest.mark.parametrize("with_connector", [False, True])
def test_sample_tokens_preserves_connector_lifecycle(
    method, is_async, verify, with_connector
):
    events = []
    metadata = object()
    connector = MagicMock()
    connector._connector_metadata = metadata

    def save():
        assert connector._connector_metadata is metadata
        events.append("save")

    def poll(_finished):
        assert connector._connector_metadata is metadata
        events.append("poll")
        return {"sent"}, {"loaded"}

    def clear():
        connector._connector_metadata = None
        events.append("clear")

    connector.wait_for_save.side_effect = save
    connector.get_finished.side_effect = poll
    connector.clear_connector_metadata.side_effect = clear
    connector.get_block_ids_with_load_errors.return_value = set()
    connector.build_connector_worker_meta.return_value = None
    connector.get_kv_connector_stats.return_value = None

    def propose(**kwargs):
        if with_connector:
            assert connector._connector_metadata is metadata
        if method == "ngram":
            assert kwargs["sampled_token_ids"] == [[7]]
        events.append("draft")
        return torch.tensor([[8]]) if is_async else None

    request = SimpleNamespace(num_computed_tokens=0, num_tokens=1, output_token_ids=[])
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r0": 1},
        finished_req_ids=set(),
        has_structured_output_requests=False,
    )
    runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(
            scheduler_output=scheduler_output,
            draft_chunks=[],
            logits_list=[torch.zeros(1, 16)],
            num_reqs_list=[1],
            spec_decode_metadata_list=[object() if verify else None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
        ),
        input_batch=SimpleNamespace(
            max_num_logprobs=None,
            num_reqs=1,
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            all_greedy=True,
            num_tokens_no_spec=[1],
            token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
        ),
        requests={"r0": request},
        speculative_config=(
            None
            if method is None
            else SimpleNamespace(method=method, num_speculative_tokens=1)
        ),
        scheduler_config=SimpleNamespace(async_scheduling=is_async),
        _is_async_drafter=method == "mtp",
        _stage_timing=MagicMock(),
        is_pooling_model=False,
        drafter=SimpleNamespace(),
        spec_decode_manager=SimpleNamespace(propose_draft_token_ids=propose),
        _get_sampling_generator=lambda: None,
        _greedy_sample=lambda _logits: torch.tensor([7]),
        _sample_spec_verify_chunk=lambda *args: torch.tensor([[7, -1]]),
        _update_mamba_slot_read_offsets=lambda *args: None,
        _update_num_xla_graphs=lambda *args: None,
        _modify_prev_results=lambda: None,
        _sync_replicated_drafts_across_tp=lambda drafts: drafts,
        _assemble_async_spec_substitution=lambda *args: (
            [torch.tensor([7, 8])],
            {0: 0},
            None,
            {"r0": 1},
        ),
        _assemble_async_prefill_bootstrap=lambda *args: (
            [torch.tensor([7, 8])],
            {0: 0},
            None,
            {"r0": 1},
        ),
        _update_placeholder=lambda *args: {"r0": 0},
        finalize_kv_connector=TPUModelRunner.finalize_kv_connector,
    )
    runner._stage_timing.take_samples.return_value = ([], [])
    for name in ("maybe_wait_for_kv_save", "get_finished_kv_transfers"):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))

    def cpu_copy(chunks, lengths):
        return AsyncTPUCopyState(
            chunks_cpu=chunks, chunk_real_lens=lengths, copy_ready_event=MagicMock()
        )

    with (
        patch.object(
            kv_transfer_state,
            "_KV_CONNECTOR_AGENT",
            connector if with_connector else None,
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=cpu_copy),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)

    if with_connector:
        assert events == (
            ["save", "poll", "clear"]
            if method is None
            else ["poll", "draft", "save", "clear"]
        )
        assert connector._connector_metadata is None
        assert (
            output._model_runner_output if is_async else output
        ).kv_connector_output.finished_sending == {"sent"}
    else:
        assert events == ([] if method is None else ["draft"])
        assert (
            output._model_runner_output if is_async else output
        ).kv_connector_output is None
    assert isinstance(output, AsyncTPUModelRunnerOutput) == is_async
    if is_async:
        output = output.get_output()
    assert output.sampled_token_ids == [[7]]
