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
"""`TPU_INDEXER_TOPK_SINGLE_WRITE` leaves the buffer and the rows attention
reads exactly as the -1 fill plus partial write did."""

import types

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUSparseAttnIndexer,
)


def _reference(buf, topk_tokens, num_tokens, topk_indices):
    """The default path of `forward_oot`, verbatim."""
    buf[:num_tokens] = -1
    num_out = min(topk_indices.shape[0], buf.shape[0])
    buf[:num_out, :topk_tokens] = topk_indices[:num_out]
    return buf


@pytest.mark.parametrize(
    "rows,width,topk,num_tokens,out_rows",
    [
        (64, 32, 32, 16, 16),  # decode bucket, full-width top-k
        (64, 48, 32, 16, 16),  # buffer wider than top-k
        (64, 32, 32, 64, 64),  # the whole buffer
        (64, 32, 32, 16, 8),  # kernel returned fewer rows than tokens
    ],
)
def test_single_write_matches_fill_then_write(rows, width, topk, num_tokens, out_rows):
    g = torch.Generator().manual_seed(rows + width + num_tokens + out_rows)
    stale = torch.randint(0, 1000, (rows, width), dtype=torch.int32, generator=g)
    topk_indices = torch.randint(
        -1, 5000, (out_rows, topk), dtype=torch.int32, generator=g
    )

    want_buf = _reference(stale.clone(), topk, num_tokens, topk_indices)
    want_rows = want_buf[:num_tokens]

    layer = types.SimpleNamespace(topk_indices_buffer=stale.clone(), topk_tokens=topk)
    got = VllmTPUSparseAttnIndexer._write_topk_rows(layer, num_tokens, topk_indices)

    torch.testing.assert_close(layer.topk_indices_buffer, want_buf, rtol=0, atol=0)
    # What attention slices with `[:num_tokens]`.
    torch.testing.assert_close(got[:num_tokens], want_rows, rtol=0, atol=0)
