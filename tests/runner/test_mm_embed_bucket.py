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
"""The bucketed gather must hand back the rows the slicing default would.

``gather_output_buckets`` replaces ``scatter_output_slices`` so that each
item's tensor is bucket-height instead of token-height. Everything downstream
reads only the first ``per_item_out_tokens[idx]`` rows, so those rows must
match the slice exactly -- an off-by-one would feed the model another image's
embeddings and still run happily.
"""

import pytest
import torch
from vllm.model_executor.models.utils import scatter_output_slices

from vllm_torchtpu.runner.mm_embed_bucket import (MM_EMBED_BUCKET_MIN,
                                                  gather_output_buckets,
                                                  mm_embed_bucket)

HIDDEN = 4


def packed(per_item):
    """Distinct values per row, so a misplaced row cannot go unnoticed."""
    rows = torch.arange(sum(per_item), dtype=torch.float32).reshape(-1, 1)
    return rows.expand(-1, HIDDEN).contiguous()


@pytest.mark.parametrize("per_item", [
    [1],
    [64],
    [65],
    [300, 200, 130],
    [1, 1, 1, 1],
    [500, 129],
    [1024],
])
def test_real_rows_match_the_slicing_default(per_item):
    output = packed(per_item)
    indices = list(range(len(per_item)))

    expected: list[torch.Tensor | None] = [None] * len(per_item)
    scatter_output_slices(output, indices, per_item, expected)

    got: list[torch.Tensor | None] = [None] * len(per_item)
    gather_output_buckets(output, indices, per_item, got)

    for idx, n in enumerate(per_item):
        assert got[idx].shape == (mm_embed_bucket(n), HIDDEN)
        assert torch.equal(got[idx][:n], expected[idx]), f"item {idx}"


def test_items_can_be_gathered_out_of_order():
    """`indices` is the batch's packing order, not 0..n-1."""
    per_item = [70, 30, 200]
    output = packed(per_item)
    indices = [2, 0, 1]

    expected: list[torch.Tensor | None] = [None] * 3
    scatter_output_slices(output, indices, per_item, expected)
    got: list[torch.Tensor | None] = [None] * 3
    gather_output_buckets(output, indices, per_item, got)

    for idx in indices:
        n = per_item[idx]
        assert torch.equal(got[idx][:n], expected[idx]), f"item {idx}"


def test_the_last_item_does_not_read_past_the_output():
    """A bucket overhangs the packed tensor; the gather must clamp, not crash."""
    per_item = [65]  # bucket 128 against 65 real rows
    got: dict[int, torch.Tensor] = {}
    gather_output_buckets(packed(per_item), [0], per_item, got)
    assert got[0].shape == (128, HIDDEN)
    assert torch.equal(got[0][65:], got[0][64:65].expand(63, HIDDEN))


def test_ladder_is_powers_of_two_above_the_floor():
    assert mm_embed_bucket(1) == MM_EMBED_BUCKET_MIN
    assert mm_embed_bucket(MM_EMBED_BUCKET_MIN) == MM_EMBED_BUCKET_MIN
    assert mm_embed_bucket(MM_EMBED_BUCKET_MIN + 1) == 2 * MM_EMBED_BUCKET_MIN
    for n in (100, 1000, 5000):
        b = mm_embed_bucket(n)
        assert b >= n and b // 2 < max(n, MM_EMBED_BUCKET_MIN)
        assert b & (b - 1) == 0
