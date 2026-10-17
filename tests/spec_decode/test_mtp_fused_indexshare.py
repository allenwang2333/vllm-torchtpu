# Copyright 2026 The vLLM TorchTPU Authors.
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
"""CPU checks of the IndexShare variant of the fused MTP draft loop (#1517):
the program installs the chunk's saved step-0 top-k rows into the draft's
table before every unrolled step, so each step reads exactly the rows the
per-step loop would have restored on the host -- even when the draft step
itself scribbles over the table (as the next chunk's step 0 would)."""

import pytest
import torch
from torch import nn

from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop


class _FakeMtp(nn.Module):
    """Reads the top-k table every step (like the MLA layer under skip_topk),
    records what it saw, then clobbers the table so a missing install shows."""

    def __init__(self, table: torch.Tensor, vocab: int = 7):
        super().__init__()
        self.table = table
        self.vocab = vocab
        self.seen: list[torch.Tensor] = []

    def forward(self, input_ids, positions, hidden_states, it=None, emb=None, *,
                spec_step_idx=0):
        p = input_ids.shape[0]
        self.seen.append(self.table[:p].clone())
        self.table.fill_(-7)
        return hidden_states + self.table[:p, :1].to(hidden_states.dtype) * 0 + 1

    def compute_logits(self, h):
        return h[:, : self.vocab]


def _loop(table, num_steps=2):
    mtp = _FakeMtp(table)
    loop = _MtpFusedLoop.__new__(_MtpFusedLoop)
    nn.Module.__init__(loop)
    loop.prefix = "t"
    loop.mtp = mtp
    loop.num_steps = num_steps
    loop.topk_indices_buffer = table
    return loop, mtp


@pytest.mark.parametrize("p", [1, 4, 8])
def test_installs_saved_rows_before_every_step(p):
    rows_total, k = 16, 32
    table = torch.full((rows_total, k), -1, dtype=torch.int32)
    loop, mtp = _loop(table, num_steps=2)
    saved = torch.arange(p * k, dtype=torch.int32).reshape(p, k)
    seq_lens = torch.ones(p, dtype=torch.int32)
    toks = loop(
        torch.zeros(p, dtype=torch.int32),
        torch.zeros(p, dtype=torch.int32),
        torch.zeros(p, 16),
        seq_lens,
        saved,
    )
    assert len(toks) == 2 and all(t.dtype == torch.int32 for t in toks)
    assert len(mtp.seen) == 2
    for seen in mtp.seen:
        torch.testing.assert_close(seen, saved)
    # seq_lens advanced once between the two steps, positions out of place.
    assert seq_lens.tolist() == [2] * p


def test_rows_required_exactly_when_built_with_a_table():
    table = torch.zeros(4, 8, dtype=torch.int32)
    loop, _ = _loop(table)
    with pytest.raises(ValueError):
        loop(torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32),
             torch.zeros(2, 16), torch.ones(2, dtype=torch.int32), None)
    loop.topk_indices_buffer = None
    with pytest.raises(ValueError):
        loop(torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32),
             torch.zeros(2, 16), torch.ones(2, dtype=torch.int32),
             torch.zeros(2, 8, dtype=torch.int32))


def test_negative_control_without_install_reads_clobbered_rows():
    """Fault injection: skip the install and the second step reads the rows
    the first step clobbered -- the check above would catch a broken port."""
    table = torch.full((8, 8), -1, dtype=torch.int32)
    loop, mtp = _loop(table)
    loop._install_shared_topk = lambda step, rows: None
    saved = torch.ones(2, 8, dtype=torch.int32)
    table[:2] = saved
    loop(torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32),
         torch.zeros(2, 16), torch.ones(2, dtype=torch.int32), saved)
    assert torch.equal(mtp.seen[0], saved)
    assert not torch.equal(mtp.seen[1], saved)


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MTP_FUSED_INDEXSHARE", raising=False)
    import vllm_torchtpu.envs as envs

    assert envs.VLLM_TPU_MTP_FUSED_INDEXSHARE is False
    monkeypatch.setenv("VLLM_TPU_MTP_FUSED_INDEXSHARE", "1")
    assert envs.VLLM_TPU_MTP_FUSED_INDEXSHARE is True
