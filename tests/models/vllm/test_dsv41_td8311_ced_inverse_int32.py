# SPDX-License-Identifier: Apache-2.0
"""TD8.311: v1 CED replay scatter via the int32 inverse gather; int32 DP sync.

``v1.model`` now restores decoder rows with the shared ``_ced_inverse_rows`` +
``_scatter_decoder_rows`` (moved to ``v2/ced.py``) instead of an int64
``index_copy_`` into a sink row; ``sync_decoder_replay_plan`` all-reduces an
int32 state. Both must equal the previous results.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v1 import model as v1_model
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ced as v2_ced
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model


def test_v2_model_reexports_moved_helpers():
    assert v2_model._ced_inverse_rows is v2_ced._ced_inverse_rows
    assert v2_model._scatter_decoder_rows is v2_ced._scatter_decoder_rows


@pytest.mark.parametrize(
    ("indices", "num_positions", "padded"),
    [((1, 3), 4, 3), ((0, 2, 5, 6), 7, 8), ((4,), 5, 1), ((), 3, 2)],
)
def test_v1_ced_inverse_scatter_matches_index_copy(indices, num_positions, padded):
    plan = SimpleNamespace(
        indices=indices, query_start_loc=(0, len(indices)), padded_num_tokens=padded
    )
    positions = torch.arange(num_positions)
    owner = SimpleNamespace()
    inverse, gather, bounds = v1_model.DeepseekV41Model._get_ced_non_identity_tensors(
        owner, plan, num_positions, positions
    )
    assert inverse.dtype == torch.int32 and inverse.shape == (num_positions,)
    # Cached on the second call.
    again = v1_model.DeepseekV41Model._get_ced_non_identity_tensors(
        owner, plan, num_positions, positions
    )
    assert again[0] is inverse

    g = torch.Generator().manual_seed(num_positions)
    output = torch.randn(padded, 6, generator=g).to(torch.bfloat16)
    # Pre-TD8.311 reference: int64 indices padded with the sink row.
    ref_idx = torch.tensor(
        indices + (num_positions,) * (padded - len(indices)), dtype=torch.int64
    )
    ref = output.new_zeros((num_positions + 1, 6)).index_copy_(0, ref_idx, output)[:-1]
    got = v1_model._scatter_decoder_rows(output, inverse)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_sync_decoder_replay_plan_all_reduces_int32(monkeypatch):
    seen = {}

    def fake_all_reduce(t, op=None, group=None):
        seen["dtype"] = t.dtype
        t[1] = 64

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    meta = SimpleNamespace(seq_lens=torch.zeros(3, dtype=torch.int32))
    plan = v2_ced.sync_decoder_replay_plan(None, meta, 8, dp_cpu_group=object())
    assert seen["dtype"] == torch.int32
    assert plan is None  # state[0] stays 0 when no rank has a plan
