# SPDX-License-Identifier: Apache-2.0
"""Final-output checks distinguish real NaNs from padding and legal infinities."""

import logging
from contextlib import nullcontext
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.cpu_test, pytest.mark.usefixtures("model_nan_cpu")]


def checker(monkeypatch):
    from vllm_torchtpu.runner import model_nan_check

    monkeypatch.setattr(model_nan_check, "logger", logging.getLogger(__name__))
    monkeypatch.setattr(model_nan_check.socket, "gethostname", lambda: "nan-test-host")
    return model_nan_check.ModelNaNChecker(rank=3, engine_idx=2)


def plan(valid=2, padded=4, token_slice=None, mapping=None):
    return SimpleNamespace(
        global_num_tokens=valid,
        global_padded_num_tokens=padded,
        token_slice=token_slice or slice(0, padded),
        _packed_to_request_major_token_indices=mapping,
    )


@pytest.mark.parametrize("real_nan", [False, True])
def test_jax_nan_reduction_returns_one_bool_and_excludes_padding(real_nan):
    from vllm_torchtpu.runner import model_nan_check

    values = jnp.array([[float("nan"), 1.0], [float("inf"), 2.0]])
    result = model_nan_check._has_nan_jax(values, jnp.array([real_nan, True]))
    assert result.shape == () and result.dtype == jnp.bool_
    assert bool(result) is real_nan


@pytest.mark.parametrize("stage", ["last_hidden", "logits"])
def test_nan_in_a_real_row_reports_the_tensor_and_rank(monkeypatch, caplog, stage):
    check = checker(monkeypatch)
    values = torch.tensor([[0.0, float("nan")], [1.0, 2.0]])
    with caplog.at_level(logging.ERROR):
        found = (
            check.check_hidden(values, plan(valid=2, padded=2))
            if stage == "last_hidden"
            else check.check_logits(values, num_valid_rows=2)
        )
    assert found is True
    assert stage in caplog.text and "rank=3" in caplog.text
    assert "host=nan-test-host" in caplog.text and "engine=2" in caplog.text
    assert "shape=(2, 2)" in caplog.text


def test_padding_nan_and_legal_infinities_do_not_report(monkeypatch, caplog):
    check = checker(monkeypatch)
    values = torch.tensor(
        [[float("inf"), -float("inf")], [0.0, 1.0], [float("nan"), 2.0]]
    )
    assert check.check_hidden(values, plan(valid=2, padded=3)) is False
    assert check.check_logits(values, num_valid_rows=2) is False
    assert not caplog.records


@pytest.mark.parametrize("bad_row,expected", [(0, False), (1, True), (2, True)])
def test_pcp_checks_noncontiguous_real_rows(monkeypatch, bad_row, expected):
    check = checker(monkeypatch)
    # This rank owns packed rows [-1, 1, 4]: padding, token 1, token 4.
    layout = plan(
        valid=5,
        padded=8,
        token_slice=slice(2, 5),
        mapping=np.array([0, 2, -1, 1, 4, 3, -1, -1]),
    )
    values = torch.ones(3, 2)
    values[bad_row, 0] = float("nan")
    assert check.check_hidden(values, layout) is expected


def test_each_chunk_is_checked_and_empty_real_rows_are_clean(monkeypatch):
    check = checker(monkeypatch)
    values = torch.full((2, 3), float("nan"))
    assert check.check_logits(values, num_valid_rows=0) is False
    assert check.check_logits(values, num_valid_rows=1) is True
    assert check.check_logits(torch.ones(2, 3), num_valid_rows=1) is False


def test_round_result_ors_chunks_and_is_consumed_once(monkeypatch):
    check = checker(monkeypatch)
    assert check.take_result() is None
    check.check_logits(torch.full((1, 2), float("nan")), num_valid_rows=1)
    check.check_logits(torch.ones(1, 2), num_valid_rows=1)
    assert check.take_result() is True
    assert check.take_result() is None
    check.check_hidden(torch.ones(1, 2), plan(valid=1, padded=1))
    assert check.take_result() is False


@pytest.mark.parametrize("pcp", [False, True])
@pytest.mark.parametrize("warmup_raises", [False, True])
def test_capture_suspends_checks_and_restores_them_after_warmup(
    monkeypatch, caplog, pcp, warmup_raises
):
    from vllm_torchtpu.runner import tpu_runner

    check = checker(monkeypatch)

    def warmup():
        assert runner._nan_checker is None, "Synthetic warmup must not check NaNs"
        if warmup_raises:
            raise RuntimeError("warmup failed")

    runner = SimpleNamespace(
        _nan_checker=check,
        _hidden_nan_detector=None,
        _kv_cache_nan_detector=None,
        _add_pipeline_chunk_buckets=lambda: None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        enforce_eager=False,
        lora_config=None,
        maybe_setup_dummy_loras=lambda _: nullcontext(),
        _precompile_timed=lambda _: nullcontext(),
        _precompile_backbone=lambda: None,
        _precompile_sampling_subgraphs=lambda: None,
        _precompile_mamba_rollback_helpers=lambda: None,
        vllm_config=SimpleNamespace(),
        device=torch.device("cpu"),
        model=None,
        _is_async_drafter=True,
        _async_drafter=SimpleNamespace(precompile=lambda: None),
        _pcp_enabled=pcp,
        _mtp_enabled=True,
        _precompile_rejection_sampler=lambda: None,
        _precompile_spec_logprobs=lambda: None,
        _warmup_spec_decode=warmup,
        _warmup_pcp_mtp_prefill=warmup,
    )
    monkeypatch.setattr(tpu_runner, "maybe_create_mm_encoder_manager", lambda *_: None)
    if warmup_raises:
        with pytest.raises(RuntimeError, match="warmup failed"):
            tpu_runner.TPUModelRunner.capture_model(runner)
    else:
        tpu_runner.TPUModelRunner.capture_model(runner)
    assert runner._nan_checker is check
    with caplog.at_level(logging.ERROR):
        assert (
            runner._nan_checker.check_logits(
                torch.full((1, 2), float("nan")), num_valid_rows=1
            )
            is True
        )
    assert "logits" in caplog.text
