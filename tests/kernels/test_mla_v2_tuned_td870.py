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
"""TD8.70: MLA v2 tuned-parameter lookup (hit and both default branches)."""

import dataclasses

import pytest

from vllm_torchtpu.kernels.mla.v2 import tuned_params as tp


@pytest.mark.cpu_test
def test_every_tuned_entry_is_returned_by_identity():
    for key, params in tp.tuned_params_mapping.items():
        assert tp.get_tuned_params(key) is params
        # An equal (not identical) key hits the same entry.
        assert tp.get_tuned_params(dataclasses.replace(key)) is params


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "mixed",
            tp.TunableParams(
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
                vmem_limit_bytes=62914560,
            ),
        ),
        (
            "batched_decode",
            tp.TunableParams(
                decode_batch_size=4,
                num_kv_pages_per_block=3,
                num_queries_per_block=1,
                vmem_limit_bytes=62914560,
            ),
        ),
        (
            "decode_only",
            tp.TunableParams(
                decode_batch_size=4,
                num_kv_pages_per_block=3,
                num_queries_per_block=1,
                vmem_limit_bytes=62914560,
            ),
        ),
    ],
)
def test_untuned_key_falls_back_with_warning(monkeypatch, case, expected):
    calls = []
    monkeypatch.setattr(tp.logger, "warning", lambda *args: calls.append(args))
    key = tp.TuningKey(
        case=case,
        max_num_tokens=12345,
        actual_num_q_heads=7,
        actual_lkv_dim=3,
        actual_r_dim=5,
    )
    assert key not in tp.tuned_params_mapping
    assert tp.get_tuned_params(key) == expected
    assert len(calls) == 1 and calls[0][1] == key
