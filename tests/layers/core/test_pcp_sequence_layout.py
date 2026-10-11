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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_torchtpu.layers.core.sequence_layout as sequence_layout
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_logits_indices,
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpSequenceLayoutEligibility,
    PcpSequenceLayoutMode,
    PcpSequenceLayoutPlanner,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    AllSequenceLayoutPlanner,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
)


def _eligibility(
    *,
    pcp_size=4,
    interleave_size=16,
    dcp_size=1,
    pipeline_parallel_size=1,
    async_scheduling=False,
):
    return PcpSequenceLayoutEligibility(
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        dcp_size=dcp_size,
        pipeline_parallel_size=pipeline_parallel_size,
        async_scheduling=async_scheduling,
    )


def _input_batch(*, computed, prompt):
    req_ids = [f"req{i}" for i in range(len(computed))]
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )


def _scheduler_output(scheduled):
    return SimpleNamespace(
        num_scheduled_tokens={
            f"req{i}": int(tokens) for i, tokens in enumerate(scheduled)
        }
    )


def _evaluate(
    *,
    eligibility=None,
    computed,
    prompt,
    scheduled,
    num_tokens_paddings=(16, 32, 64),
    max_num_tokens=64,
):
    eligibility = eligibility or _eligibility()
    scheduled_np = np.asarray(scheduled, dtype=np.int32)
    return eligibility.evaluate_runner_chunk(
        input_batch=_input_batch(computed=computed, prompt=prompt),
        scheduler_output=_scheduler_output(scheduled),
        start_index=0,
        num_reqs=len(scheduled),
        num_scheduled_tokens_per_req=scheduled_np,
        num_tokens_paddings=num_tokens_paddings,
        max_num_tokens=max_num_tokens,
    )


@pytest.mark.parametrize("kv_role", [None, "kv_producer", "kv_consumer", "kv_both"])
def test_prefill_layout_is_independent_of_kv_role(kv_role):
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4,
            cp_kv_cache_interleave_size=16,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        kv_transfer_config=(
            None
            if kv_role is None
            else SimpleNamespace(
                kv_role=kv_role,
                is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            )
        ),
    )

    eligibility = PcpSequenceLayoutEligibility.from_vllm_config(vllm_config)

    assert eligibility.enabled
    assert eligibility.pcp_size == 4
    assert eligibility.interleave_size == 16
    decision = _evaluate(
        eligibility=eligibility, computed=[0], prompt=[64], scheduled=[64]
    )
    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 16


def test_evaluate_runner_chunk_classifies_streaming_prefill():
    decision = _evaluate(computed=[0], prompt=[64], scheduled=[64])

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.pcp_size == 4
    assert decision.interleave_size == 16
    assert decision.pcp_alignment is None
    assert decision.local_required_tokens == 16
    assert decision.local_padded_tokens == 16
    assert decision.global_padded_tokens == 64
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([16, 16, 16, 16])
    )
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0])
    )


def test_evaluate_runner_chunk_accepts_decode_only_query_spans():
    decision = _evaluate(
        computed=[64, 64],
        prompt=[64, 64],
        scheduled=[1, 1],
        num_tokens_paddings=(2, 4),
        max_num_tokens=4,
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 2
    assert decision.local_padded_tokens == 2
    assert decision.global_padded_tokens == 8
    np.testing.assert_array_equal(decision.local_token_counts, np.asarray([2, 0, 0, 0]))
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([64, 64])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0, 1])
    )


def test_evaluate_runner_chunk_disabled_when_pcp_is_off():
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=1),
        computed=[0],
        prompt=[64],
        scheduled=[64],
    )

    assert decision.mode is PcpSequenceLayoutMode.DISABLED


def test_evaluate_runner_chunk_accepts_mixed_prefill_decode_batch():
    decision = _evaluate(
        computed=[0, 64],
        prompt=[64, 64],
        scheduled=[64, 1],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 17
    assert decision.local_padded_tokens == 32
    assert decision.global_padded_tokens == 128
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([17, 16, 16, 16])
    )
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0, 64])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0, 64])
    )


def test_evaluate_runner_chunk_rejects_boundary_crossing_chunk():
    with pytest.raises(NotImplementedError, match="boundary-crossing"):
        _evaluate(
            computed=[32],
            prompt=[64],
            scheduled=[64],
        )


def test_evaluate_runner_chunk_accepts_unaligned_q_len():
    decision = _evaluate(
        computed=[0],
        prompt=[32],
        scheduled=[32],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 16
    assert decision.local_padded_tokens == 16
    assert decision.global_padded_tokens == 64
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([16, 16, 0, 0])
    )


def test_evaluate_runner_chunk_accepts_unaligned_q_start():
    decision = _evaluate(
        computed=[16],
        prompt=[80],
        scheduled=[64],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 16
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([16, 16, 16, 16])
    )


def test_batch_flat_owner_is_independent_from_request_absolute_position():
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=2),
        computed=[16],
        prompt=[32],
        scheduled=[16],
        num_tokens_paddings=(16,),
        max_num_tokens=16,
    )

    assert decision.spans[0].absolute_query_start == 16
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([16])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0])
    )
    assert decision.absolute_query_start_offsets_per_req is not (
        decision.token_owner_start_offsets_per_req
    )
    np.testing.assert_array_equal(decision.local_token_counts, np.asarray([16, 0]))

    token_order, inverse_order = build_pcp_rank_major_token_order(
        [16],
        pcp_size=2,
        interleave_size=16,
        padded_num_tokens=32,
        token_owner_start_offsets_per_req=(decision.token_owner_start_offsets_per_req),
    )
    np.testing.assert_array_equal(token_order[:16], np.arange(16))
    np.testing.assert_array_equal(token_order[16:], np.full(16, -1))
    np.testing.assert_array_equal(inverse_order, np.arange(16))
    np.testing.assert_array_equal(
        build_pcp_logits_indices(
            [16],
            pcp_size=2,
            interleave_size=16,
            padded_num_tokens=32,
            token_owner_start_offsets_per_req=(
                decision.token_owner_start_offsets_per_req
            ),
        ),
        np.asarray([15]),
    )


@pytest.mark.parametrize(
    ("q_lens", "pcp_size", "interleave_size", "padded_num_tokens", "owner_starts"),
    [
        ([2, 3], 2, 1, 6, [0, 2]),
        ([3, 3], 2, 2, 8, [0, 3]),
        ([2, 5, 1], 4, 2, 16, [0, 2, 7]),
        ([1], 4, 2, 8, [5]),
    ],
)
def test_pcp_token_orders_are_inverses_and_each_request_has_one_owner(
    q_lens, pcp_size, interleave_size, padded_num_tokens, owner_starts
):
    packed_to_request, request_to_packed = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=padded_num_tokens,
        token_owner_start_offsets_per_req=owner_starts,
    )
    total_num_tokens = sum(q_lens)
    np.testing.assert_array_equal(
        packed_to_request[request_to_packed],
        np.arange(total_num_tokens),
    )

    request_gathers = torch.tensor(
        np.cumsum(q_lens, dtype=np.int64) - 1,
        dtype=torch.int64,
    )
    request_tokens = torch.arange(total_num_tokens, dtype=torch.int64)
    local_padded_num_tokens = padded_num_tokens // pcp_size
    owner_counts = torch.zeros(len(q_lens), dtype=torch.int64)
    for rank in range(pcp_size):
        local_start = rank * local_padded_num_tokens
        plan = SequenceLayoutPlan(
            descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(local_start, local_start + local_padded_num_tokens),
            global_num_tokens=total_num_tokens,
            global_padded_num_tokens=padded_num_tokens,
            local_num_tokens=int(
                np.count_nonzero(
                    packed_to_request[
                        local_start : local_start + local_padded_num_tokens
                    ]
                    >= 0
                )
            ),
            local_padded_num_tokens=local_padded_num_tokens,
            _packed_to_request_major_token_indices=packed_to_request,
            _request_major_to_packed_token_indices=request_to_packed,
        )

        local_tokens, local_gathers = plan.localize_token_tensor_and_gather_indices(
            request_tokens,
            request_gathers,
            num_valid_gathers=len(q_lens),
        )
        if plan.local_num_tokens == 0:
            torch.testing.assert_close(local_tokens, torch.zeros_like(local_tokens))
            torch.testing.assert_close(
                local_gathers, torch.full_like(local_gathers, -1)
            )
        owner_counts += local_gathers.ge(0)

    torch.testing.assert_close(owner_counts, torch.ones_like(owner_counts))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "num_tokens,expected_gathers",
    [
        (3, [[0, 1, -1], [-1, -1, -1]]),
        (5, [[0, 2, -1], [-1, -1, -1]]),
        (6, [[0, -1, -1], [-1, 2, -1]]),
    ],
)
def test_partial_plan_buckets_mapping_before_device_copy(
    monkeypatch, num_tokens, expected_gathers
):
    packed_to_request, request_to_packed = build_pcp_rank_major_token_order(
        [1, num_tokens - 1],
        pcp_size=2,
        interleave_size=1,
        padded_num_tokens=6,
        token_owner_start_offsets_per_req=[0, 1],
    )
    original_mapping = request_to_packed.copy()
    request_tokens = torch.arange(num_tokens, dtype=torch.int32)
    request_gathers = torch.tensor([0, num_tokens - 1, 0])
    as_tensor = torch.as_tensor
    host_arrays = []

    def record_tensor(data, *args, **kwargs):
        if isinstance(data, np.ndarray):
            host_arrays.append(data.copy())
        return as_tensor(data, *args, **kwargs)

    # Observe the actual host-to-device boundary while running the real ops.
    monkeypatch.setattr(sequence_layout.torch, "as_tensor", record_tensor)
    for rank in range(2):
        local_start = rank * 3
        plan = SequenceLayoutPlan(
            descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(local_start, local_start + 3),
            global_num_tokens=num_tokens,
            global_padded_num_tokens=6,
            local_num_tokens=int(
                np.count_nonzero(packed_to_request[local_start : local_start + 3] >= 0)
            ),
            local_padded_num_tokens=3,
            _packed_to_request_major_token_indices=packed_to_request,
            _request_major_to_packed_token_indices=request_to_packed,
        )
        host_arrays.clear()
        _, local_gathers = plan.localize_token_tensor_and_gather_indices(
            request_tokens, request_gathers, num_valid_gathers=2
        )
        # Both copies use layout capacities, even when the real token count
        # changes. Padding on the device would still expose a dynamic input.
        assert [array.shape for array in host_arrays] == [(3,), (6,)]
        np.testing.assert_array_equal(host_arrays[1][:num_tokens], original_mapping)
        np.testing.assert_array_equal(host_arrays[1][num_tokens:], -1)
        torch.testing.assert_close(local_gathers, torch.tensor(expected_gathers[rank]))
        np.testing.assert_array_equal(request_to_packed, original_mapping)


def test_partial_plan_ignores_padded_request_aligned_gathers():
    packed_to_request, request_to_packed = build_pcp_rank_major_token_order(
        [2, 3],
        pcp_size=2,
        interleave_size=1,
        padded_num_tokens=6,
        token_owner_start_offsets_per_req=[0, 2],
    )
    request_tokens = torch.tensor([10, 11, 20, 21, 22])
    request_gathers = torch.tensor([1, 4, 0])

    expected = [
        (torch.tensor([10, 20, 22]), torch.tensor([-1, 2, -1])),
        (torch.tensor([11, 21, 0]), torch.tensor([0, -1, -1])),
    ]
    for rank, (expected_tokens, expected_gathers) in enumerate(expected):
        local_start = rank * 3
        plan = SequenceLayoutPlan(
            descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(local_start, local_start + 3),
            global_num_tokens=5,
            global_padded_num_tokens=6,
            local_num_tokens=3 if rank == 0 else 2,
            local_padded_num_tokens=3,
            _packed_to_request_major_token_indices=packed_to_request,
            _request_major_to_packed_token_indices=request_to_packed,
        )

        local_tokens, local_gathers = plan.localize_token_tensor_and_gather_indices(
            request_tokens, request_gathers, num_valid_gathers=2
        )

        torch.testing.assert_close(local_tokens, expected_tokens)
        torch.testing.assert_close(local_gathers, expected_gathers)


def _carry_plan(descriptor: SequenceLayoutDescriptor) -> SequenceLayoutPlan:
    return SequenceLayoutPlan(
        descriptor=descriptor,
        token_slice=slice(0, 3),
        global_num_tokens=3,
        global_padded_num_tokens=3,
        local_num_tokens=3,
        local_padded_num_tokens=3,
    )


def test_default_plan_skips_pcp_carry_collective(monkeypatch):
    plan = AllSequenceLayoutPlanner().prepare_dummy(
        num_tokens=3,
        num_reqs=3,
        kv_cache_initialized=False,
    )
    local_tensor = torch.tensor([[99.0], [20.0], [7.0]])
    owner_mask = torch.tensor([False, True, False])
    monkeypatch.setattr(
        sequence_layout,
        "_pcp_all_reduce_sum",
        lambda _tensor: pytest.fail("default layout entered PCP collective"),
        raising=False,
    )

    result = plan.aggregate_request_aligned_tensor(local_tensor, owner_mask)

    torch.testing.assert_close(result, local_tensor)


def test_pcp_plan_masks_non_owner_before_request_aligned_reduce(monkeypatch):
    plan = _carry_plan(PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR)
    local_tensor = torch.tensor([[99.0], [20.0], [7.0]])
    owner_mask = torch.tensor([False, True, False])
    reduced_inputs = []

    def fake_reduce(masked_tensor):
        reduced_inputs.append(masked_tensor.clone())
        remote_owner = torch.zeros_like(masked_tensor)
        remote_owner[0, 0] = 10
        return masked_tensor + remote_owner

    monkeypatch.setattr(
        sequence_layout, "_pcp_all_reduce_sum", fake_reduce, raising=False
    )

    result = plan.aggregate_request_aligned_tensor(local_tensor, owner_mask)

    torch.testing.assert_close(reduced_inputs[0], torch.tensor([[0.0], [20.0], [0.0]]))
    torch.testing.assert_close(result, torch.tensor([[10.0], [20.0], [0.0]]))


def test_evaluate_runner_chunk_classifies_multi_active_streaming_prefill():
    decision = _evaluate(
        computed=[0, 0],
        prompt=[1024, 1024],
        scheduled=[1024, 1024],
        num_tokens_paddings=(256, 512),
        max_num_tokens=512,
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.pcp_alignment is None
    assert decision.local_required_tokens == 512
    assert decision.local_padded_tokens == 512
    assert decision.global_padded_tokens == 2048
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([512, 512, 512, 512])
    )
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0, 0])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0, 1024])
    )


def test_batch_flat_owner_balances_32_by_256_across_pcp8():
    num_reqs = 32
    q_len = 256
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=8, interleave_size=256),
        computed=[0] * num_reqs,
        prompt=[q_len] * num_reqs,
        scheduled=[q_len] * num_reqs,
        num_tokens_paddings=(1024,),
        max_num_tokens=1024,
    )

    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req,
        np.arange(num_reqs, dtype=np.int64) * q_len,
    )
    np.testing.assert_array_equal(
        decision.local_token_counts, np.full(8, 1024, dtype=np.int32)
    )
    assert decision.local_required_tokens == 1024
    assert decision.local_padded_tokens == 1024
    assert decision.global_padded_tokens == 8192


def test_batch_flat_owner_balances_four_by_8192_across_pcp8():
    num_reqs = 4
    q_len = 8192
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=8, interleave_size=256),
        computed=[0] * num_reqs,
        prompt=[q_len] * num_reqs,
        scheduled=[q_len] * num_reqs,
        num_tokens_paddings=(4096,),
        max_num_tokens=4096,
    )

    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req,
        np.arange(num_reqs, dtype=np.int64) * q_len,
    )
    np.testing.assert_array_equal(
        decision.local_token_counts, np.full(8, 4096, dtype=np.int32)
    )
    assert decision.local_required_tokens == 4096
    assert decision.local_padded_tokens == 4096
    assert decision.global_padded_tokens == 4 * q_len


def test_evaluate_runner_chunk_accepts_unaligned_multi_active_q_len():
    decision = _evaluate(
        computed=[0, 0],
        prompt=[1024, 512],
        scheduled=[1024, 512],
        num_tokens_paddings=(256, 512),
        max_num_tokens=512,
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 384
    assert decision.local_padded_tokens == 512
    assert decision.global_padded_tokens == 2048
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([384, 384, 384, 384])
    )


def test_evaluate_runner_chunk_rejects_local_bucket_overflow():
    with pytest.raises(ValueError, match="max compile bucket"):
        _evaluate(
            computed=[0],
            prompt=[1024],
            scheduled=[1024],
            num_tokens_paddings=(128,),
            max_num_tokens=128,
        )


@pytest.mark.parametrize(
    ("eligibility", "error_type", "message"),
    [
        (_eligibility(dcp_size=2), NotImplementedError, "DCP"),
        (
            _eligibility(pipeline_parallel_size=2),
            NotImplementedError,
            "pipeline parallelism",
        ),
        (_eligibility(interleave_size=0), ValueError, "interleave_size > 0"),
    ],
)
def test_evaluate_runner_chunk_rejects_unsupported_runtime_config(
    eligibility, error_type, message
):
    with pytest.raises(error_type, match=message):
        _evaluate(eligibility=eligibility, computed=[0], prompt=[64], scheduled=[64])


def test_evaluate_runner_chunk_accepts_async_non_speculative_runtime_config():
    decision = _evaluate(
        eligibility=_eligibility(async_scheduling=True),
        computed=[0],
        prompt=[64],
        scheduled=[64],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 16


def _planner_runner_stub(*, scheduled, computed, prompt, token_paddings):
    max_tokens = max(token_paddings)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        input_ids_cpu=torch.arange(max_tokens * 4, dtype=torch.int32),
        positions_cpu=torch.arange(max_tokens * 4, dtype=torch.int32),
        uses_mrope=False,
        supports_mm_inputs=False,
        positions_np=None,
        arange_np=np.arange(max_tokens * 4, dtype=np.int32),
        num_tokens_paddings=list(token_paddings),
        max_num_tokens=max_tokens,
        _dp_target_bucket=None,
    )
    runner.positions_np = runner.positions_cpu.numpy()
    req_ids = [f"req{i}" for i in range(len(scheduled))]
    runner.input_batch = SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )
    return runner


def test_pcp_sequence_layout_planner_returns_partial_plan(monkeypatch):
    runner = _planner_runner_stub(
        scheduled=[25],
        computed=[0],
        prompt=[25],
        token_paddings=[256],
    )
    planner = PcpSequenceLayoutPlanner(_eligibility(pcp_size=2))

    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_rank", lambda: 1
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_world_size",
        lambda: 2,
    )

    plan = planner.prepare_real(
        runner=runner,
        scheduler_output=_scheduler_output([25]),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.asarray([25], dtype=np.int32),
        total_num_scheduled_tokens=25,
        use_max_model_len=True,
        target_num_reqs=8,
        padded_num_reqs=8,
    )

    assert plan.kind is SequenceLayoutKind.PARTIAL
    assert plan.descriptor.protocol == "pcp_streaming"
    assert plan.global_padded_num_tokens == 512
    assert plan.local_padded_num_tokens == 256
    assert plan.local_num_tokens == 9
    assert plan.token_slice == slice(256, 512)
    assert plan.requires_hidden_state_gather is True
    assert plan.local_index_for_request_major_token(0) is None
    assert plan.local_index_for_request_major_token(16) == 0
    assert plan.local_index_for_request_major_token(24) == 8
    expected_packed_to_request = np.full(512, -1, dtype=np.int64)
    expected_packed_to_request[:16] = np.arange(16)
    expected_packed_to_request[256:265] = np.arange(16, 25)
    np.testing.assert_array_equal(
        plan._packed_to_request_major_token_indices,
        expected_packed_to_request,
    )
    np.testing.assert_array_equal(
        plan._request_major_to_packed_token_indices,
        np.concatenate((np.arange(16), np.arange(256, 265))),
    )
    local_tokens, local_gathers = plan.localize_token_tensor_and_gather_indices(
        torch.arange(25, dtype=torch.int32),
        torch.tensor([24]),
        num_valid_gathers=1,
    )
    torch.testing.assert_close(
        local_tokens[:9], torch.arange(16, 25, dtype=torch.int32)
    )
    torch.testing.assert_close(local_tokens[9:], torch.zeros(247, dtype=torch.int32))
    torch.testing.assert_close(local_gathers, torch.tensor([8]))
    torch.testing.assert_close(
        plan.logits_indices_cpu,
        torch.tensor([264] + [-1] * 7, dtype=torch.int32),
    )
    torch.testing.assert_close(
        plan.logits_local_indices_cpu,
        torch.tensor([8] + [0] * 7, dtype=torch.int32),
    )
    torch.testing.assert_close(
        plan.logits_owner_mask_cpu,
        torch.tensor([True] + [False] * 7, dtype=torch.bool),
    )

    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.pcp_sequence_layout._pcp_all_reduce_sum",
        lambda tensor: tensor,
    )
    hidden_states = torch.arange(256 * 2, dtype=torch.float32).reshape(256, 2)
    selected = planner.maybe_select_logits_hidden_states(
        hidden_states,
        plan,
        plan.logits_indices_cpu,
    )
    expected = torch.zeros((8, 2), dtype=torch.float32)
    expected[0] = hidden_states[8]
    torch.testing.assert_close(selected, expected)


def test_pcp_sequence_layout_planner_returns_all_when_disabled():
    runner = _planner_runner_stub(
        scheduled=[25],
        computed=[0],
        prompt=[25],
        token_paddings=[32, 64],
    )
    planner = PcpSequenceLayoutPlanner(_eligibility(pcp_size=1))

    plan = planner.prepare_real(
        runner=runner,
        scheduler_output=_scheduler_output([25]),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.asarray([25], dtype=np.int32),
        total_num_scheduled_tokens=25,
        use_max_model_len=True,
        target_num_reqs=8,
        padded_num_reqs=8,
    )

    assert plan.kind is SequenceLayoutKind.ALL
    assert plan.token_slice == slice(0, 32)
    assert plan.global_padded_num_tokens == 32
    assert plan.local_padded_num_tokens == 32
    assert plan.requires_hidden_state_gather is False


def test_pcp_planner_preinit_and_dummy_share_the_v1_layout_contract():
    planner = PcpSequenceLayoutPlanner(_eligibility(pcp_size=8, interleave_size=256))

    assert planner.requires_backend_preinit is True
    assert planner.backend_preinit_world_size == 8

    before_cache = planner.prepare_dummy(
        num_tokens=4096,
        num_reqs=32,
        kv_cache_initialized=False,
    )
    assert before_cache.kind is SequenceLayoutKind.PARTIAL
    assert before_cache.descriptor.protocol == "pcp_streaming"
    assert before_cache.descriptor.version == 1

    after_cache = planner.prepare_dummy(
        num_tokens=4096,
        num_reqs=32,
        kv_cache_initialized=True,
    )
    assert after_cache.kind is SequenceLayoutKind.PARTIAL
    assert after_cache.descriptor.protocol == "pcp_streaming"
    assert after_cache.descriptor.version == 1
    assert after_cache.token_slice == slice(0, 4096)
    assert after_cache.global_num_tokens == 8 * 4096
    assert after_cache.global_padded_num_tokens == 8 * 4096
    assert after_cache.local_num_tokens == 4096
    assert after_cache.local_padded_num_tokens == 4096
    assert after_cache.requires_hidden_state_gather is True

    assert before_cache == after_cache


def test_sequence_layout_and_pcp_remaining_branches(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.core import pcp_sequence_layout as pcp_mod
    from vllm_torchtpu.layers.core.pcp_sequence_layout import (
        PcpSequenceLayoutDecision,
        _ensure_host_token_buffer_capacity,
        prepare_pcp_sequence_layout,
    )
    from vllm_torchtpu.layers.core.sequence_layout import (
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        create_sequence_layout_planner,
        is_pcp_streaming_attention_metadata,
        is_pcp_streaming_sequence_layout,
    )

    # 1. SequenceLayoutDescriptor.cache_key & SequenceLayoutPlan edge cases
    assert DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR.cache_key == ("all", "default", 1)
    plan_no_remap = SequenceLayoutPlan(
        descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        token_slice=slice(0, 4),
        global_num_tokens=4,
        global_padded_num_tokens=4,
        local_num_tokens=4,
        local_padded_num_tokens=4,
    )
    assert plan_no_remap.local_index_for_request_major_token(2) == 2

    plan_bad_step = SequenceLayoutPlan(
        descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        token_slice=slice(0, 4, 2),
        global_num_tokens=4,
        global_padded_num_tokens=4,
        local_num_tokens=2,
        local_padded_num_tokens=2,
    )
    with pytest.raises(RuntimeError, match="contiguous token slices"):
        plan_bad_step.local_index_for_request_major_token(0)

    plan_no_stop = SequenceLayoutPlan(
        descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        token_slice=slice(0, None),
        global_num_tokens=4,
        global_padded_num_tokens=4,
        local_num_tokens=4,
        local_padded_num_tokens=4,
    )
    with pytest.raises(RuntimeError, match="finite token_slice stop"):
        plan_no_stop.local_index_for_request_major_token(0)

    t_tok = torch.arange(4, dtype=torch.int32)
    t_gath = torch.tensor([0, 1], dtype=torch.int32)
    out_tok, out_gath = plan_no_remap.localize_token_tensor_and_gather_indices(
        t_tok, t_gath, num_valid_gathers=2
    )
    assert out_tok is t_tok and out_gath is t_gath

    # Partial plan with int32 gather_indices to cover gather_dtype != torch.long cast
    partial_plan = SequenceLayoutPlan(
        descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
        token_slice=slice(0, 2),
        global_num_tokens=2,
        global_padded_num_tokens=2,
        local_num_tokens=2,
        local_padded_num_tokens=2,
        _packed_to_request_major_token_indices=np.array([0, 1], dtype=np.int64),
        _request_major_to_packed_token_indices=np.array([0, 1], dtype=np.int64),
    )
    _, local_gath_i32 = partial_plan.localize_token_tensor_and_gather_indices(
        torch.tensor([10, 20], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        num_valid_gathers=2,
    )
    assert local_gath_i32.dtype == torch.int32

    bad_desc = SequenceLayoutDescriptor(
        kind=SequenceLayoutKind.PARTIAL, protocol="custom", version=2
    )
    with pytest.raises(RuntimeError, match="Unsupported draft sequence layout"):
        _carry_plan(bad_desc).aggregate_request_aligned_tensor(
            torch.ones(3, 1), torch.ones(3, dtype=torch.bool)
        )

    # 2. AllSequenceLayoutPlanner properties & create_sequence_layout_planner
    all_planner = AllSequenceLayoutPlanner()
    assert all_planner.requires_backend_preinit is False
    assert all_planner.backend_preinit_world_size == 1
    assert all_planner.uses_selected_logits_hidden_states is False
    all_planner.reserve_host_token_capacity(None, 16)
    hs = torch.ones(4, 2)
    assert all_planner.finalize_hidden_states(hs, None) is hs
    assert all_planner.maybe_select_logits_hidden_states(hs, None, t_gath) is None

    runner_dp = SimpleNamespace(num_tokens_paddings=[16, 32], _dp_target_bucket=64)
    dp_plan = all_planner.prepare_real(
        runner=runner_dp,
        scheduler_output=None,
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=[8],
        total_num_scheduled_tokens=8,
        use_max_model_len=False,
        target_num_reqs=1,
        padded_num_reqs=1,
    )
    assert dp_plan.local_padded_num_tokens == 64

    assert isinstance(create_sequence_layout_planner(None), AllSequenceLayoutPlanner)
    pcp_vllm_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            cp_kv_cache_interleave_size=16,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    created_pcp = create_sequence_layout_planner(pcp_vllm_cfg)
    assert isinstance(created_pcp, PcpSequenceLayoutPlanner)
    assert created_pcp.uses_selected_logits_hidden_states is True

    assert is_pcp_streaming_sequence_layout(
        kind=SequenceLayoutKind.PARTIAL, protocol="pcp_streaming"
    )
    assert is_pcp_streaming_attention_metadata(
        SimpleNamespace(
            sequence_layout_kind="partial", sequence_layout_protocol="pcp_streaming"
        )
    )
    assert not is_pcp_streaming_attention_metadata(SimpleNamespace())

    # 3. PcpSequenceLayoutDecision, Eligibility, and _evaluate_streaming_query branches
    disabled_dec = PcpSequenceLayoutDecision(
        mode=PcpSequenceLayoutMode.DISABLED, spans=()
    )
    assert disabled_dec.enabled is False
    with pytest.raises(NotImplementedError, match="cannot be used"):
        PcpSequenceLayoutEligibility.require_mode(
            disabled_dec, PcpSequenceLayoutMode.STREAMING
        )

    elig = _eligibility(pcp_size=2, interleave_size=16)
    batch = _input_batch(computed=[0, 0], prompt=[16, 16])
    batch.req_ids = [None, "req1"]
    sched_out = SimpleNamespace(num_scheduled_tokens={})

    with pytest.raises(ValueError, match="1D array"):
        elig._build_request_spans(
            input_batch=batch,
            scheduler_output=sched_out,
            start_index=0,
            num_reqs=2,
            num_scheduled_tokens_per_req=np.zeros((1, 2), dtype=np.int32),
            absolute_query_start_offsets_per_req=None,
        )
    with pytest.raises(ValueError, match="cover every request"):
        elig._build_request_spans(
            input_batch=batch,
            scheduler_output=sched_out,
            start_index=0,
            num_reqs=2,
            num_scheduled_tokens_per_req=[16],
            absolute_query_start_offsets_per_req=None,
        )
    with pytest.raises(ValueError, match="absolute_query_start_offsets_per_req"):
        elig._build_request_spans(
            input_batch=batch,
            scheduler_output=sched_out,
            start_index=0,
            num_reqs=2,
            num_scheduled_tokens_per_req=[16, 16],
            absolute_query_start_offsets_per_req=[0],
        )

    spans_list = elig._build_request_spans(
        input_batch=batch,
        scheduler_output=sched_out,
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=[16, 16],
        absolute_query_start_offsets_per_req=[0, 8],
    )
    assert len(spans_list) == 1 and spans_list[0].absolute_query_start == 8

    spans_np = elig._build_request_spans(
        input_batch=batch,
        scheduler_output=sched_out,
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.array([16, 16], dtype=np.int32),
        absolute_query_start_offsets_per_req=np.array([0, 12], dtype=np.int32),
    )
    assert len(spans_np) == 1 and spans_np[0].absolute_query_start == 12

    with pytest.raises(NotImplementedError, match="scheduled query tokens"):
        elig._evaluate_streaming_query(
            (pcp_mod.PcpRequestSpan(0, 0, 16, 0),),
            num_tokens_paddings=None,
            max_num_tokens=None,
            dp_target_bucket=None,
        )

    elig_bad_interleave = _eligibility(pcp_size=2, interleave_size=7)
    with pytest.raises(
        NotImplementedError, match="multiple of cp_kv_cache_interleave_size"
    ):
        elig_bad_interleave._evaluate_streaming_query(
            spans_np,
            num_tokens_paddings=None,
            max_num_tokens=None,
            dp_target_bucket=None,
        )

    dec_no_pad = elig._evaluate_streaming_query(
        spans_np,
        num_tokens_paddings=None,
        max_num_tokens=None,
        dp_target_bucket=None,
    )
    assert dec_no_pad.local_padded_tokens is None

    dec_gen_pad = elig._evaluate_streaming_query(
        spans_np,
        num_tokens_paddings=(x for x in (16, 32)),
        max_num_tokens=None,
        dp_target_bucket=64,
    )
    assert dec_gen_pad.local_padded_tokens == 64

    with pytest.raises(ValueError, match="exceeds runner max token buffer"):
        elig._evaluate_streaming_query(
            spans_np,
            num_tokens_paddings=[16, 32],
            max_num_tokens=32,
            dp_target_bucket=64,
        )

    # 4. PcpSequenceLayoutPlanner & prepare_pcp_sequence_layout
    disabled_planner = PcpSequenceLayoutPlanner.from_vllm_config(None)
    disabled_planner.reserve_host_token_capacity(None, 32)
    dummy_disabled = disabled_planner.prepare_dummy(
        num_tokens=8, num_reqs=1, kv_cache_initialized=False
    )
    assert dummy_disabled.kind is SequenceLayoutKind.ALL
    assert disabled_planner.finalize_hidden_states(hs, None) is hs
    assert disabled_planner.finalize_hidden_states(hs, dummy_disabled) is hs
    assert disabled_planner.maybe_select_logits_hidden_states(hs, None, t_gath) is None
    assert (
        disabled_planner.maybe_select_logits_hidden_states(hs, dummy_disabled, t_gath)
        is None
    )

    monkeypatch.setattr(pcp_mod, "_pcp_all_gather_equal_tokens", lambda t, dim=0: t * 2)
    dummy_enabled = created_pcp.prepare_dummy(
        num_tokens=8, num_reqs=1, kv_cache_initialized=False
    )
    torch.testing.assert_close(
        created_pcp.finalize_hidden_states(hs, dummy_enabled), hs * 2
    )

    runner_grow = _planner_runner_stub(
        scheduled=[16], computed=[0], prompt=[16], token_paddings=[16]
    )
    runner_grow.arange_np = np.arange(2, dtype=np.int32)
    runner_grow.input_ids_cpu = torch.ones(4, dtype=torch.int32)
    runner_grow.positions_cpu = torch.ones(4, dtype=torch.int32)
    runner_grow.supports_mm_inputs = True
    runner_grow.is_mm_embed_cpu = torch.ones(4, dtype=torch.bool)
    runner_grow.uses_mrope = True
    runner_grow.mrope_positions = SimpleNamespace(
        cpu=torch.ones((3, 5), dtype=torch.int32)
    )
    runner_grow._make_buffer = lambda r, c, dtype: SimpleNamespace(
        cpu=torch.zeros((r, c), dtype=dtype)
    )
    runner_grow.seq_lens_np = np.array([16], dtype=np.int32)
    runner_grow.vllm_config = pcp_vllm_cfg

    _ensure_host_token_buffer_capacity(runner_grow, 0)
    created_pcp.reserve_host_token_capacity(runner_grow, 64)
    assert runner_grow.arange_np.dtype == np.int32
    assert len(runner_grow.arange_np) == 64
    assert runner_grow.input_ids_cpu.shape[0] == 64
    assert runner_grow.is_mm_embed_cpu.shape[0] == 64
    assert runner_grow.mrope_positions.cpu.shape == (3, 65)

    with pytest.raises(ValueError, match="prefill_context_parallel_size > 1"):
        prepare_pcp_sequence_layout(
            runner_grow,
            _scheduler_output([16]),
            0,
            1,
            np.array([16], dtype=np.int32),
            16,
            False,
            1,
            1,
            decision=PcpSequenceLayoutDecision(
                mode=PcpSequenceLayoutMode.STREAMING, spans=(), pcp_size=1
            ),
        )

    with pytest.raises(RuntimeError, match="does not match"):
        prepare_pcp_sequence_layout(
            runner_grow,
            _scheduler_output([16]),
            0,
            1,
            np.array([16], dtype=np.int32),
            16,
            False,
            1,
            1,
            get_native_pcp_rank=lambda: 0,
            get_native_pcp_world_size=lambda: 4,
        )

    with pytest.raises(RuntimeError, match="out of range"):
        prepare_pcp_sequence_layout(
            runner_grow,
            _scheduler_output([16]),
            0,
            1,
            np.array([16], dtype=np.int32),
            16,
            False,
            1,
            1,
            get_native_pcp_rank=lambda: 5,
            get_native_pcp_world_size=lambda: 2,
        )

    monkeypatch.setattr(envs, "VLLM_TPU_DEBUG_PCP_LAYOUT", True)
    runner_grow2 = _planner_runner_stub(
        scheduled=[16, 16], computed=[0, 0], prompt=[16, 16], token_paddings=[16]
    )
    runner_grow2.uses_mrope = True
    runner_grow2.mrope_positions = SimpleNamespace(
        cpu=torch.zeros((3, 64), dtype=torch.int32)
    )
    runner_grow2.seq_lens_np = np.array([16, 16], dtype=np.int32)
    runner_grow2.vllm_config = pcp_vllm_cfg
    prepared_dbg = prepare_pcp_sequence_layout(
        runner_grow2,
        _scheduler_output([16, 16]),
        0,
        2,
        np.array([16, 16], dtype=np.int32),
        32,
        False,
        2,
        2,
        get_native_pcp_rank=lambda: 0,
        get_native_pcp_world_size=lambda: 2,
    )
    assert prepared_dbg.padded_total_num_scheduled_tokens == 32

    runner_no_mm = _planner_runner_stub(
        scheduled=[16], computed=[0], prompt=[16], token_paddings=[16]
    )
    runner_no_mm.supports_mm_inputs = False
    runner_no_mm.uses_mrope = False
    _ensure_host_token_buffer_capacity(runner_no_mm, 128)
    assert runner_no_mm.input_ids_cpu.shape[0] == 128

    import vllm_torchtpu.runner.owner_token_layout as owner_token_layout

    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_KV_SHARD", True)
    monkeypatch.setattr(
        owner_token_layout,
        "step_owner_required_tokens",
        lambda *_args, **_kwargs: 24,
    )
    all_planner = AllSequenceLayoutPlanner()
    kv_shard_runner = SimpleNamespace(
        num_tokens_paddings=[16, 32, 64],
        _dp_target_bucket=None,
    )
    plan_kv_shard = all_planner.prepare_real(
        runner=kv_shard_runner,
        scheduler_output=SimpleNamespace(),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.array([8], dtype=np.int32),
        total_num_scheduled_tokens=8,
        use_max_model_len=False,
        target_num_reqs=1,
        padded_num_reqs=1,
    )
    assert plan_kv_shard.local_padded_num_tokens == 32


def test_td8279_localize_token_tensor_and_gather_indices_int32_index_select(
    monkeypatch,
):
    from vllm_torchtpu.layers.core.sequence_layout import (
        _localize_token_tensor_and_gather_indices,
    )

    select_calls = []
    orig_index_select = torch.index_select

    def spy_index_select(t, dim, index):
        select_calls.append((t.dtype, dim, index.dtype))
        return orig_index_select(t, dim, index)

    monkeypatch.setattr(torch, "index_select", spy_index_select)

    tokens = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.int32)
    local_packed_to_req = np.array([2, 0, -1], dtype=np.int64)
    req_to_packed = np.array([1, 3, 0, 2], dtype=np.int64)

    for g_dtype in (torch.int32, torch.int64):
        select_calls.clear()
        gathers = torch.tensor([0, 2, -1, 1], dtype=g_dtype)
        loc_tok, loc_gath = _localize_token_tensor_and_gather_indices(
            tokens,
            gathers,
            local_packed_to_req,
            req_to_packed,
            local_start=0,
            local_end=2,
        )
        assert loc_tok.tolist() == [[30, 31], [10, 11], [0, 0]]
        assert loc_gath.dtype == g_dtype
        assert loc_gath.tolist() == [1, 0, -1, -1]
        assert select_calls == [
            (torch.int32, 0, torch.int32),
            (torch.int32, 0, torch.int32),
        ]


def test_td8282_int32_index_select_in_runner_logits_and_pcp_sequence_layout(
    monkeypatch,
):
    from vllm_torchtpu.layers.core import pcp_sequence_layout as pcp_mod
    from vllm_torchtpu.layers.core.sequence_layout import (
        create_sequence_layout_planner,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    select_calls = []
    orig_index_select = torch.index_select

    def spy_index_select(t, dim, index):
        select_calls.append((t.dtype, dim, index.dtype))
        return orig_index_select(t, dim, index)

    monkeypatch.setattr(torch, "index_select", spy_index_select)
    monkeypatch.setattr(pcp_mod, "_pcp_all_reduce_sum", lambda x: x)

    pcp_vllm_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            cp_kv_cache_interleave_size=16,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    planner = create_sequence_layout_planner(pcp_vllm_cfg)
    plan = SimpleNamespace(
        logits_local_indices_cpu=torch.tensor([2, 0], dtype=torch.int64),
        logits_owner_mask_cpu=torch.tensor([True, False], dtype=torch.bool),
    )
    hs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    select_calls.clear()
    out_hs = planner.maybe_select_logits_hidden_states(
        hs, plan, torch.tensor([0, 1], dtype=torch.int64)
    )
    assert out_hs is not None
    assert out_hs.tolist() == [[5.0, 6.0], [0.0, 0.0]]
    assert select_calls == [(torch.float32, 0, torch.int32)]

    runner = object.__new__(TPUModelRunner)
    runner.is_pooling_model = False
    runner.model = SimpleNamespace(
        lm_head="head",
        compute_logits=lambda x: x * 2.0,
        logits_processor=SimpleNamespace(
            get_top_tokens=lambda _h, x: x[:, 0].to(torch.int64)
        ),
    )
    idx_i64 = torch.tensor([2, 0], dtype=torch.int64)

    select_calls.clear()
    sel_logits = TPUModelRunner.compute_selected_logits.__wrapped__(runner, hs, idx_i64)
    assert sel_logits.tolist() == [[10.0, 12.0], [2.0, 4.0]]
    assert select_calls == [(torch.float32, 0, torch.int32)]

    select_calls.clear()
    sel_top = TPUModelRunner.compute_selected_top_tokens.__wrapped__(
        runner, hs, idx_i64
    )
    assert sel_top.tolist() == [5, 1]
    assert select_calls == [(torch.float32, 0, torch.int32)]

    bonus_idx = torch.tensor([2], dtype=torch.int64)
    target_idx = torch.tensor([0, 1], dtype=torch.int64)

    select_calls.clear()
    b_ids_1d, t_ids_1d = TPUModelRunner.spec_bonus_and_target_logits.__wrapped__(
        runner,
        torch.tensor([10, 20, 30], dtype=torch.int64),
        bonus_idx,
        target_idx,
    )
    assert b_ids_1d.dtype == torch.int32 and b_ids_1d.tolist() == [30]
    assert t_ids_1d.dtype == torch.int32 and t_ids_1d.tolist() == [10, 20]
    assert select_calls == [
        (torch.int32, 0, torch.int32),
        (torch.int32, 0, torch.int32),
    ]

    select_calls.clear()
    logits_2d = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
    b_ids_2d, t_ids_2d = TPUModelRunner.spec_bonus_and_target_logits.__wrapped__(
        runner, logits_2d, bonus_idx, target_idx
    )
    assert b_ids_2d.dtype == torch.int32 and b_ids_2d.tolist() == [1]
    assert t_ids_2d.dtype == torch.int32 and t_ids_2d.tolist() == [1, 0]
    assert select_calls == [
        (torch.int32, 0, torch.int32),
        (torch.int32, 0, torch.int32),
    ]

    select_calls.clear()
    b_log, t_log = TPUModelRunner.spec_gather_bonus_and_target_logits.__wrapped__(
        runner, logits_2d, bonus_idx, target_idx
    )
    assert torch.equal(b_log, logits_2d[[2]])
    assert torch.equal(t_log, logits_2d[[0, 1]])
    assert select_calls == [
        (torch.float32, 0, torch.int32),
        (torch.float32, 0, torch.int32),
    ]


def test_td8298_pcp_layout_and_sequence_layout_native_int32_dtypes(monkeypatch):
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import pcp_layout

    starts_none = pcp_layout.pcp_token_owner_start_offsets(
        np.array([4, 8], dtype=np.int32), None
    )
    assert starts_none.dtype == np.int32
    starts_list = pcp_layout.pcp_token_owner_start_offsets(
        np.array([4, 8], dtype=np.int32), [0, 4]
    )
    assert starts_list.dtype == np.int32

    counts = pcp_layout.pcp_local_token_counts([4, 8], pcp_size=2, interleave_size=2)
    assert counts.dtype == np.int32

    order_1, inv_1 = pcp_layout.build_pcp_rank_major_token_order(
        [4, 8], pcp_size=1, interleave_size=2, padded_num_tokens=16
    )
    assert order_1.dtype == np.int32 and inv_1.dtype == np.int32

    order_2, inv_2 = pcp_layout.build_pcp_rank_major_token_order(
        [4, 8], pcp_size=2, interleave_size=2, padded_num_tokens=16
    )
    assert order_2.dtype == np.int32 and inv_2.dtype == np.int32

    logits_empty = pcp_layout.build_pcp_logits_indices(
        [], pcp_size=2, interleave_size=2, padded_num_tokens=16
    )
    assert logits_empty.dtype == np.int32
    logits_idx = pcp_layout.build_pcp_logits_indices(
        [4, 8], pcp_size=2, interleave_size=2, padded_num_tokens=16, token_offset=4
    )
    assert logits_idx.dtype == np.int32

    decision = _evaluate(computed=[0, 16], prompt=[64, 64], scheduled=[16, 16])
    assert decision.absolute_query_start_offsets_per_req.dtype == np.int32
    assert decision.token_owner_start_offsets_per_req.dtype == np.int32
    assert decision.local_token_counts.dtype == np.int32

    runner = _planner_runner_stub(
        scheduled=[16, 16], computed=[0, 0], prompt=[16, 16], token_paddings=[32]
    )
    planner = PcpSequenceLayoutPlanner(_eligibility(pcp_size=2, interleave_size=16))
    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_world_size",
        lambda: 2,
    )
    plan = planner.prepare_real(
        runner=runner,
        scheduler_output=_scheduler_output([16, 16]),
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.asarray([16, 16], dtype=np.int32),
        total_num_scheduled_tokens=32,
        use_max_model_len=True,
        target_num_reqs=2,
        padded_num_reqs=2,
    )
    assert plan._packed_to_request_major_token_indices.dtype == np.int32
    assert plan._request_major_to_packed_token_indices.dtype == np.int32


def test_td8325_pcp_logits_indices_reuses_precomputed_inverse_order(monkeypatch):
    import vllm_torchtpu.layers.core.pcp_sequence_layout as pcp_seq_mod
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import pcp_layout

    _, inv = pcp_layout.build_pcp_rank_major_token_order(
        [4, 8], pcp_size=2, interleave_size=2, padded_num_tokens=16
    )
    expected = pcp_layout.build_pcp_logits_indices(
        [4, 8], pcp_size=2, interleave_size=2, padded_num_tokens=16, token_offset=4
    )

    order_calls: list[int] = []
    orig_build_order = pcp_layout.build_pcp_rank_major_token_order

    def _spy_build_order(*args, **kwargs):
        order_calls.append(1)
        return orig_build_order(*args, **kwargs)

    monkeypatch.setattr(
        pcp_layout, "build_pcp_rank_major_token_order", _spy_build_order
    )
    actual = pcp_layout.build_pcp_logits_indices(
        [4, 8],
        pcp_size=2,
        interleave_size=2,
        padded_num_tokens=16,
        token_offset=4,
        inverse_order=inv,
    )
    assert order_calls == []
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)

    # Verify prepare_real calls build_pcp_rank_major_token_order only once
    runner = _planner_runner_stub(
        scheduled=[16, 16], computed=[0, 0], prompt=[16, 16], token_paddings=[32]
    )
    runner.seq_lens_np = np.asarray([16, 16], dtype=np.int32)
    planner = PcpSequenceLayoutPlanner(_eligibility(pcp_size=2, interleave_size=16))
    monkeypatch.setattr(pcp_seq_mod, "_get_native_pcp_rank", lambda: 0)
    monkeypatch.setattr(pcp_seq_mod, "_get_native_pcp_world_size", lambda: 2)
    monkeypatch.setattr(pcp_seq_mod.envs, "VLLM_TPU_DEBUG_PCP_LAYOUT", True)

    cumsum_dtypes: list[object] = []
    orig_cumsum = np.cumsum

    def _spy_cumsum(a, axis=None, dtype=None, out=None):
        cumsum_dtypes.append(dtype)
        return orig_cumsum(a, axis=axis, dtype=dtype, out=out)

    monkeypatch.setattr(pcp_seq_mod.np, "cumsum", _spy_cumsum)
    order_calls.clear()
    planner.prepare_real(
        runner=runner,
        scheduler_output=_scheduler_output([16, 16]),
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.asarray([16, 16], dtype=np.int32),
        total_num_scheduled_tokens=32,
        use_max_model_len=True,
        target_num_reqs=2,
        padded_num_reqs=2,
    )
    assert len(order_calls) == 1
    assert all(dt == np.int32 for dt in cumsum_dtypes)


def test_td8355_all_sequence_layout_plan_cache_and_protocol_short_circuit(
    monkeypatch,
):
    import vllm_torchtpu.runner.owner_token_layout as owner_token_layout

    planner = AllSequenceLayoutPlanner()
    dummy_1 = planner.prepare_dummy(
        num_tokens=16, num_reqs=4, kv_cache_initialized=True
    )
    dummy_2 = planner.prepare_dummy(
        num_tokens=16, num_reqs=8, kv_cache_initialized=False
    )
    assert dummy_1 is dummy_2

    runner = SimpleNamespace(
        num_tokens_paddings=[16, 32, 64],
        _dp_target_bucket=None,
    )
    real_1 = planner.prepare_real(
        runner=runner,
        scheduler_output=SimpleNamespace(),
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.asarray([4, 4], dtype=np.int32),
        total_num_scheduled_tokens=8,
        use_max_model_len=False,
        target_num_reqs=2,
        padded_num_reqs=2,
    )
    real_2 = planner.prepare_real(
        runner=runner,
        scheduler_output=SimpleNamespace(),
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.asarray([4, 4], dtype=np.int32),
        total_num_scheduled_tokens=8,
        use_max_model_len=False,
        target_num_reqs=2,
        padded_num_reqs=2,
    )
    assert real_1 is real_2
    assert real_1.local_padded_num_tokens == 16

    # Verify _owner_layout_mod lazy initialization and reuse under KV_SHARD.
    monkeypatch.setattr(sequence_layout, "_owner_layout_mod", None)
    monkeypatch.setattr(sequence_layout.envs, "TPU_DEEPSEEK_V41_KV_SHARD", True)
    monkeypatch.setattr(
        owner_token_layout,
        "step_owner_required_tokens",
        lambda *_args, **_kwargs: 28,
    )
    kv_plan_1 = planner.prepare_real(
        runner=runner,
        scheduler_output=SimpleNamespace(),
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.asarray([4, 4], dtype=np.int32),
        total_num_scheduled_tokens=8,
        use_max_model_len=False,
        target_num_reqs=2,
        padded_num_reqs=2,
    )
    assert sequence_layout._owner_layout_mod is owner_token_layout
    kv_plan_2 = planner.prepare_real(
        runner=runner,
        scheduler_output=SimpleNamespace(),
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.asarray([4, 4], dtype=np.int32),
        total_num_scheduled_tokens=8,
        use_max_model_len=False,
        target_num_reqs=2,
        padded_num_reqs=2,
    )
    assert kv_plan_1 is kv_plan_2
    assert kv_plan_1.local_padded_num_tokens == 32

    # Short-circuit on protocol != PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL
    # avoids coercing kind via str().
    class _BadStrKind:
        def __str__(self) -> str:
            raise AssertionError("str() should not be called on non-PCP protocol")

    assert (
        sequence_layout.is_pcp_streaming_sequence_layout(
            kind=_BadStrKind(),  # type: ignore[arg-type]
            protocol=sequence_layout.DEFAULT_SEQUENCE_LAYOUT_PROTOCOL,
        )
        is False
    )
    assert (
        sequence_layout.is_pcp_streaming_sequence_layout(
            kind=SequenceLayoutKind.PARTIAL,
            protocol=sequence_layout.PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
        )
        is True
    )
