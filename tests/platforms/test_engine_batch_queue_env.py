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

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_torchtpu import _patch_vllm_engine_core_step_with_batch_queue
from vllm_torchtpu.env_override import _resolve_sc_offload_threshold_bytes


def test_resolve_sc_offload_threshold_bytes(monkeypatch):
    # 1. Explicit integer override
    monkeypatch.setenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "33554432")
    assert _resolve_sc_offload_threshold_bytes() == 33554432

    # 2. Disabled explicitly ("0")
    monkeypatch.setenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "0")
    assert _resolve_sc_offload_threshold_bytes() is None

    # Reset override to auto
    monkeypatch.setenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "auto")

    # 3. TPU_ACCELERATOR_TYPE lookup
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v7x-8")
    monkeypatch.delenv("TPU_TYPE", raising=False)
    monkeypatch.delenv("ACCELERATOR_TYPE", raising=False)
    assert _resolve_sc_offload_threshold_bytes() == 67108864

    # 4. TPU_TYPE lookup fallback
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setenv("TPU_TYPE", "v6e-4")
    assert _resolve_sc_offload_threshold_bytes() == 134217728

    # 5. ACCELERATOR_TYPE lookup fallback
    monkeypatch.delenv("TPU_TYPE", raising=False)
    monkeypatch.setenv("ACCELERATOR_TYPE", "v5p-8")
    assert _resolve_sc_offload_threshold_bytes() == 67108864

    # 6. Fallback default (64 MiB) when no env var is set and metadata fails
    monkeypatch.delenv("ACCELERATOR_TYPE", raising=False)
    monkeypatch.setattr(
        "vllm_torchtpu.env_override._fetch_tpu_type_from_metadata",
        lambda: None)
    assert _resolve_sc_offload_threshold_bytes() == 67108864


def test_user_has_flag():
    import vllm_torchtpu.env_override as env_override

    monkeypatch_args = "--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_enable_diagnostic_coordinator=false"
    env_override._existing_libtpu_args = monkeypatch_args

    assert env_override._user_has_flag(
        "--xla_tpu_use_dynamic_smem_negotiation")
    assert env_override._user_has_flag(
        "--xla_tpu_enable_diagnostic_coordinator")
    assert not env_override._user_has_flag("--xla_tpu_non_existent_flag")


def test_engine_core_step_with_batch_queue_sampling():
    _patch_vllm_engine_core_step_with_batch_queue()
    from vllm.v1.engine.core import EngineCore

    batch_queue = deque()
    mock_scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=16,
        pending_structured_output_tokens=False,
    )
    mock_scheduler = MagicMock()
    mock_scheduler.has_requests.return_value = True
    mock_scheduler.schedule.return_value = mock_scheduler_output
    mock_scheduler.get_grammar_bitmask.return_value = "mock_grammar"
    mock_scheduler.update_from_output.return_value = ["mock_engine_output"]

    mock_exec_future = Future()
    mock_exec_future.set_result(None)  # TPU execute_model returns None

    mock_sample_future = Future()
    mock_sample_output = SimpleNamespace(outputs=["mock_token"])
    mock_sample_future.set_result(mock_sample_output)

    mock_model_executor = MagicMock()
    mock_model_executor.execute_model.return_value = mock_exec_future
    mock_model_executor.sample_tokens.return_value = mock_sample_future

    engine = SimpleNamespace(
        batch_queue=batch_queue,
        batch_queue_size=1,
        is_ec_consumer=False,
        is_pooling_model=False,
        check_for_draft_tokens=False,
        scheduler=mock_scheduler,
        model_executor=mock_model_executor,
        _should_throttle_prefills=MagicMock(return_value=False),
        log_error_detail=MagicMock(return_value=MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock())),
        capture_iteration_details=MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None), __exit__=MagicMock())),
        _attach_iteration_details=MagicMock(),
        _process_aborts_queue=MagicMock(),
    )

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert model_executed is True
    assert outputs == ["mock_engine_output"]
    mock_model_executor.execute_model.assert_called_once_with(
        mock_scheduler_output, non_block=True)
    mock_model_executor.sample_tokens.assert_called_once_with("mock_grammar",
                                                              non_block=True)
