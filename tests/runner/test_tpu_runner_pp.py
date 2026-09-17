# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel paths of TPUModelRunner."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(**attrs):
    runner = MagicMock(spec=TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.model_config = MagicMock()
    runner.model_config.dtype = torch.bfloat16
    runner._pp_intermediate_template = None
    runner._pp_topk_buffer = None
    for name in ("forward_model", "_pp_intermediate_tensors",
                 "_pp_take_topk_indices", "_pp_outgoing_tensors",
                 "_sample_tokens_pp_intermediate_stage", "sample_tokens"):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


class TestForwardModel:

    def test_single_stage_call_is_unchanged(self):
        runner = _runner()
        runner.model = MagicMock(return_value=torch.zeros(4, 8))
        ids, pos = torch.zeros(4, dtype=torch.int32), torch.zeros(4)

        out, aux = runner.forward_model(ids, pos)

        runner.model.assert_called_once_with(input_ids=ids,
                                             positions=pos,
                                             inputs_embeds=None)
        assert out.shape == (4, 8) and aux is None

    def test_passes_intermediate_tensors_and_returns_them_unwrapped(self):
        runner = _runner()
        produced = IntermediateTensors({"hidden_states": torch.zeros(4, 8)})
        runner.model = MagicMock(return_value=produced)
        consumed = IntermediateTensors({"hidden_states": torch.ones(4, 8)})
        ids, pos = torch.zeros(4, dtype=torch.int32), torch.zeros(4)

        out, aux = runner.forward_model(ids,
                                        pos,
                                        intermediate_tensors=consumed)

        runner.model.assert_called_once_with(input_ids=ids,
                                             positions=pos,
                                             inputs_embeds=None,
                                             intermediate_tensors=consumed)
        assert out is produced and aux is None


class TestIntermediateBuffers:

    def _model(self):
        model = MagicMock()
        model.make_empty_intermediate_tensors.return_value = (
            IntermediateTensors({
                "hidden_states":
                torch.zeros(1, 8, dtype=torch.bfloat16),
                "residual":
                torch.zeros(1, 8, dtype=torch.bfloat16),
            }))
        return model

    def test_buffers_take_the_token_count_and_cache_the_template(self):
        runner = _runner(model=self._model())

        zeros = runner._pp_intermediate_tensors(16, zeros=True)
        empties = runner._pp_intermediate_tensors(32, zeros=False)

        assert set(zeros.tensors) == {"hidden_states", "residual"}
        assert all(t.shape == (16, 8) and t.dtype == torch.bfloat16
                   for t in zeros.tensors.values())
        assert torch.equal(zeros["hidden_states"],
                           torch.zeros(16, 8).bfloat16())
        assert all(t.shape == (32, 8) for t in empties.tensors.values())
        runner.model.make_empty_intermediate_tensors.assert_called_once_with(
            batch_size=1, dtype=torch.bfloat16, device=runner.device)

    def test_a_sparse_model_adds_a_top_k_lane_that_attends_to_nothing(self):
        """Every allocation of the lane is -1: an uninitialized or random
        index would address the KV cache during a warmup or profile run."""
        runner = _runner(model=self._model(),
                         max_num_tokens=64,
                         _profile_inputs={},
                         _pp_topk_buffer=torch.zeros(64, 4, dtype=torch.int32))
        runner._profile_input = TPUModelRunner._profile_input.__get__(runner)

        modes = (runner._pp_intermediate_tensors(16, zeros=True),
                 runner._pp_intermediate_tensors(16, zeros=False),
                 runner._pp_intermediate_tensors(16, zeros=False, random=True))

        for tensors in modes:
            assert set(tensors.tensors) == {
                "hidden_states", "residual", "topk_indices"
            }
            lane = tensors["topk_indices"]
            assert lane.shape == (16, 4) and lane.dtype == torch.int32
            assert torch.equal(lane, torch.full((16, 4), -1,
                                                dtype=torch.int32))


class TestTopkHandoff:
    """The lane that carries a sparse model's shared index table between
    stages, in and out of the buffer its layers read."""

    def _runner_with_buffer(self):
        buffer = torch.zeros(64, 4, dtype=torch.int32)
        return _runner(_pp_topk_buffer=buffer), buffer

    def test_incoming_indices_land_in_the_buffer_and_leave_the_arguments(self):
        runner, buffer = self._runner_with_buffer()
        received = torch.full((16, 4), 7, dtype=torch.int32)
        tensors = IntermediateTensors({
            "hidden_states": torch.zeros(16, 8),
            "residual": torch.zeros(16, 8),
            "topk_indices": received,
        })

        runner._pp_take_topk_indices(tensors, 16)

        # the model's forward never sees the lane
        assert set(tensors.tensors) == {"hidden_states", "residual"}
        assert torch.equal(buffer[:16], received)
        # rows this forward does not cover are untouched
        assert torch.equal(buffer[16:], torch.zeros(48, 4, dtype=torch.int32))

    def test_a_stage_handed_no_indices_is_refused(self):
        runner, _ = self._runner_with_buffer()
        tensors = IntermediateTensors({"hidden_states": torch.zeros(16, 8)})

        with pytest.raises(RuntimeError, match="no top-k indices"):
            runner._pp_take_topk_indices(tensors, 16)

    def test_outgoing_indices_are_a_copy_of_the_buffer(self):
        """A stage holds its hand-off until its next forward, by which time
        its own layers may have written the table again."""
        runner, buffer = self._runner_with_buffer()
        buffer[:16] = 3

        outgoing = runner._pp_outgoing_tensors(
            {
                "hidden_states": torch.zeros(16, 8),
                "residual": torch.zeros(16, 8),
            }, 16)
        buffer[:16] = 9

        assert torch.equal(outgoing["topk_indices"],
                           torch.full((16, 4), 3, dtype=torch.int32))

    def test_a_model_without_the_table_is_untouched(self):
        runner = _runner()
        tensors = IntermediateTensors({"hidden_states": torch.zeros(16, 8)})
        original = {"hidden_states": torch.zeros(16, 8)}

        runner._pp_take_topk_indices(tensors, 16)

        assert set(tensors.tensors) == {"hidden_states"}
        assert runner._pp_outgoing_tensors(original, 16) is original


class TestIntermediateStageSampleTokens:

    def test_without_a_pending_step_returns_empty(self):
        runner = _runner(_pp_pending_scheduler_output=None)

        assert runner._sample_tokens_pp_intermediate_stage() is \
            EMPTY_MODEL_RUNNER_OUTPUT

    def test_reports_kv_connector_progress_for_the_pending_step(self):
        scheduler_output = object()
        runner = _runner(_pp_pending_scheduler_output=scheduler_output)
        runner.get_finished_kv_transfers.return_value = ({"r1"}, None, None,
                                                         set(), None, None)

        output = runner._sample_tokens_pp_intermediate_stage()

        runner.maybe_wait_for_kv_save.assert_called_once_with()
        runner.get_finished_kv_transfers.assert_called_once_with(
            scheduler_output)
        assert output is not EMPTY_MODEL_RUNNER_OUTPUT
        assert output.kv_connector_output.finished_sending == {"r1"}
        assert runner._pp_pending_scheduler_output is None

    def test_quiet_connector_returns_empty(self):
        runner = _runner(_pp_pending_scheduler_output=object())
        runner.get_finished_kv_transfers.return_value = (None, None, None,
                                                         set(), None, None)

        assert runner._sample_tokens_pp_intermediate_stage() is \
            EMPTY_MODEL_RUNNER_OUTPUT

    def test_sample_tokens_on_an_intermediate_stage_takes_the_passthrough(
            self):
        runner = _runner(execute_model_state=None,
                         _pp_is_last=False,
                         _pp_pending_scheduler_output=None)

        assert runner.sample_tokens(None) is EMPTY_MODEL_RUNNER_OUTPUT

    def test_sample_tokens_without_state_on_the_last_stage_is_a_noop(self):
        runner = _runner(execute_model_state=None, _pp_is_last=True)

        assert runner.sample_tokens(None) is None
