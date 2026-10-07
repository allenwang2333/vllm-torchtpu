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
"""TPU_GDN_DECODE_ONLY_MAX_TOKENS: small buckets compile GDN's BATCHED pass only.

The op drops the PER_SEQ pass for any bucket of at most the limit, so the
runner must never place a chunk with a prefill/mixed request in one. These
tests pin both halves of that contract: the op-side predicate, the runner's
floor, and the sequence layout planner honouring it.
"""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.core import gdn_attention
from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.cpu_test

PADDINGS = [4, 8, 16, 32, 48, 64, 128, 256, 1024]


@pytest.fixture
def limit(monkeypatch):
    monkeypatch.setenv("TPU_GDN_DECODE_ONLY_MAX_TOKENS", "64")
    yield 64


def _floor(start_index, num_reqs, num_windowed_reqs):
    return TPUModelRunner._gdn_mixed_min_tokens(
        SimpleNamespace(), start_index, num_reqs, num_windowed_reqs
    )


def _bucket(total, min_tokens):
    runner = SimpleNamespace(
        num_tokens_paddings=PADDINGS,
        _dp_target_bucket=None,
        _gdn_chunk_min_tokens=min_tokens,
    )
    plan = AllSequenceLayoutPlanner().prepare_real(
        runner=runner,
        scheduler_output=None,
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=None,
        total_num_scheduled_tokens=total,
        use_max_model_len=True,
        target_num_reqs=16,
        padded_num_reqs=16,
    )
    return plan.global_padded_num_tokens


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_GDN_DECODE_ONLY_MAX_TOKENS", raising=False)
    assert not gdn_attention.gdn_decode_only_program(8)
    assert _floor(0, 3, 2) == 0
    assert _bucket(13, 0) == 16


def test_op_predicate(limit):
    assert gdn_attention.gdn_decode_only_program(4)
    assert gdn_attention.gdn_decode_only_program(64)
    assert not gdn_attention.gdn_decode_only_program(128)


def test_decode_only_chunk_keeps_its_bucket(limit):
    # 11 verify windows of 4 rows: all windowed -> no floor -> bucket 48.
    assert _floor(0, 11, 11) == 0
    assert _bucket(44, _floor(0, 11, 11)) == 48


@pytest.mark.parametrize(
    "start_index,num_reqs,num_windowed_reqs",
    [
        (0, 3, 2),  # two windows + one prefill tail
        (0, 1, 0),  # a lone short prefill
        (4, 3, 6),  # second chunk: windows 4..5, request 6 is a prefill
    ],
)
def test_chunk_with_prefill_skips_decode_only_buckets(
    limit, start_index, num_reqs, num_windowed_reqs
):
    floor = _floor(start_index, num_reqs, num_windowed_reqs)
    assert floor == limit + 1
    bucket = _bucket(13, floor)
    assert bucket == 128
    assert not gdn_attention.gdn_decode_only_program(bucket)


def test_second_chunk_of_windows_only(limit):
    # Requests 4..6 of a batch whose first 8 are windowed: no prefill here.
    assert _floor(4, 3, 8) == 0


def test_large_prefill_unchanged(limit):
    assert _bucket(700, limit + 1) == 1024


def test_kwargs_drop_or_keep_launch(limit, monkeypatch):
    assert gdn_attention.gdn_decode_only_kwargs(128) == {}
    assert gdn_attention.gdn_decode_only_kwargs(16) == {"batched_only": True}
    monkeypatch.setenv("TPU_GDN_DECODE_ONLY_KEEP_LAUNCH", "1")
    assert gdn_attention.gdn_decode_only_kwargs(16) == {
        "per_seq_empty": True,
        "per_seq_noalias": False,
    }
    assert gdn_attention.gdn_decode_only_kwargs(128) == {}


def test_kwargs_noalias(limit, monkeypatch):
    monkeypatch.setenv("TPU_GDN_DECODE_ONLY_KEEP_LAUNCH", "1")
    monkeypatch.setenv("TPU_GDN_DECODE_ONLY_NOALIAS", "1")
    assert gdn_attention.gdn_decode_only_kwargs(16) == {
        "per_seq_empty": True,
        "per_seq_noalias": True,
    }


def test_conv_weight_pretransposed_at_load(monkeypatch):
    import torch

    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    monkeypatch.setenv("TPU_GDN_CONV_WEIGHT_PRETRANSPOSED", "1")
    layer = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    weight = torch.randn(1536, 1, 4, dtype=torch.bfloat16)
    layer.conv1d = torch.nn.Module()
    layer.conv1d.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    monkeypatch.setattr(
        VllmGatedDeltaNetAttention, "_pcp_streaming_enabled", staticmethod(lambda: False)
    )
    layer.process_weights_after_loading(torch.bfloat16)
    kernel_layout = layer.conv_weight_kernel_layout
    assert kernel_layout.shape == (4, 1, 1536)
    assert kernel_layout.dtype == torch.float32
    assert torch.equal(kernel_layout, weight.permute(2, 1, 0).to(torch.float32))
