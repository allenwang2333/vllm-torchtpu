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
"""Tests for capping the scheduler token budget to user compile_sizes."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def cap_budget():
    """Stub optional dependencies and yield the scheduler budget guard."""
    stubs = {
        mod: MagicMock()
        for mod in (
            "torch",
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.compile",
            "torch_tpu._internal.utils",
            "portpicker",
            "vllm",
            "vllm.envs",
            "vllm.logger",
            "vllm.platforms",
            "vllm.platforms.interface",
        )
        if mod not in sys.modules
    }
    with patch.dict(sys.modules, stubs):
        from vllm_torchtpu.platforms.tpu_platform import (
            _cap_max_num_scheduled_tokens_to_compile_sizes,
        )

        yield _cap_max_num_scheduled_tokens_to_compile_sizes


def _budget_config(
    sizes, budget, pcp=1, chunked=True, scheduled=None, seqs=16, spec_k=None
):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(compile_sizes=list(sizes)),
        scheduler_config=SimpleNamespace(
            max_num_seqs=seqs,
            max_num_batched_tokens=budget,
            max_num_scheduled_tokens=scheduled,
            enable_chunked_prefill=chunked,
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        speculative_config=(
            None if spec_k is None else SimpleNamespace(num_speculative_tokens=spec_k)
        ),
    )


@pytest.mark.parametrize(
    "sizes,budget,pcp,chunked,scheduled,expected",
    [
        # Decode server from the P/D launcher: cap to the largest bucket.
        ([16, 256], 16384, 1, True, None, 256),
        # Explicit scheduled-token budget within the largest bucket: untouched.
        ([16, 256], 16384, 1, True, 128, 128),
        # Largest bucket already covers the budget: untouched.
        ([16, 256, 16384], 16384, 1, True, None, None),
        ([16, 256], 256, 1, True, None, None),
        # Prefill context parallel: a bucket counts rows per rank, so the
        # limit is bucket x pcp tokens.
        ([4096], 16384, 4, True, None, None),
        ([4096], 32768, 4, True, None, 16384),
        ([512, 1024], 16384, 4, True, None, 4096),
        # Without chunked prefill the scheduler cannot split: only warn.
        ([16, 256], 16384, 1, False, None, None),
        ([4096], 32768, 4, False, None, None),
        # Largest bucket below max_num_seqs: cap; extra requests wait a step.
        ([8], 2048, 1, True, None, 8),
    ],
)
def test_cap_scheduled_tokens(
    cap_budget, sizes, budget, pcp, chunked, scheduled, expected
):
    cfg = _budget_config(sizes, budget, pcp, chunked, scheduled)
    cap_budget(cfg)
    want = scheduled if expected is None else expected
    assert cfg.scheduler_config.max_num_scheduled_tokens == want
    assert cfg.scheduler_config.max_num_batched_tokens == budget


@pytest.mark.parametrize(
    "sizes,pcp,scheduled,need_bucket",
    [
        ([16, 256], 1, 1024, 1024),
        ([512, 1024], 4, 8192, 2048),
    ],
)
def test_explicit_scheduled_tokens_above_compile_sizes_raises(
    cap_budget, sizes, pcp, scheduled, need_bucket
):
    cfg = _budget_config(sizes, 16384, pcp=pcp, scheduled=scheduled)
    with pytest.raises(
        ValueError,
        match=rf"max_num_scheduled_tokens \({scheduled}\).*>= {need_bucket}",
    ):
        cap_budget(cfg)


def _warnings(mock_logger):
    return [c.args[0] % c.args[1:] for c in mock_logger.warning.call_args_list]


@pytest.mark.parametrize(
    "sizes,seqs,spec_k,expect_short_step",
    [
        # 128 seqs x (1 + 3 drafts) = 512 tokens > cap 256.
        ([16, 256], 128, 3, True),
        # 16 seqs x (1 + 3 drafts) = 64 tokens fits under 256.
        ([16, 256], 16, 3, False),
        # No speculative decoding: 16 seqs fit under 256.
        ([16, 256], 16, None, False),
        # 16 seqs > cap 8 even without drafts.
        ([8], 16, None, True),
    ],
)
def test_cap_warns_when_a_full_decode_step_does_not_fit(
    cap_budget, sizes, seqs, spec_k, expect_short_step
):
    cfg = _budget_config(sizes, 16384, seqs=seqs, spec_k=spec_k)
    with patch("vllm_torchtpu.platforms.tpu_platform.logger") as log:
        cap_budget(cfg)
    msgs = _warnings(log)
    assert any("capping scheduled tokens per step" in m for m in msgs)
    short = [m for m in msgs if "below a full decode step" in m]
    assert bool(short) == expect_short_step
    if short:
        k = 0 if spec_k is None else spec_k
        assert f"{seqs} seqs x {1 + k} tokens = {seqs * (1 + k)}" in short[0]
