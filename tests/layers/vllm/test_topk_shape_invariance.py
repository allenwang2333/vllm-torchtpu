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
"""The one thing TPU_MOE_LOCAL_ROUTING needs from the device.

`torch.topk` lowers to a **non-stable** sort whose comparator compares the f32
key alone (`torch_tpu/ops/topk/topk.cc:BuildSortComparisonBody` -- it builds an
argument pair per operand and then compares only `args[0], args[1]`;
`topk_aten_kernels.cc` passes `TopKStableMode::kUnstable`).  Ties at the
top-k boundary are therefore resolved by the sort network, not by expert index,
and 11.7% of real Qwen3.5-397B rows have such a tie.

With the flag off, every rank routes all T_global rows; with the flag on, each
rank routes its own T_local rows and broadcasts.  They agree iff that network
gives the same answer per row at both row counts.  Nothing offline can decide
it: on CPU `torch.topk` is index-ordered and the question is invisible.  This

Run on one TPU chip, no model, ~seconds:
    pytest -q test_topk_shape_invariance.py
"""
import glob
import os

import pytest
import torch


def _tpu_device_present() -> bool:
    """A real chip, not just an importable torch_tpu.

    The CPU dev image imports torch_tpu happily, and on CPU `torch.topk` is
    index-ordered -- so these tests would pass without answering the question
    they exist for. Probe for the device node rather than initialising the
    runtime, which on a TPU-less host is slow and noisy.
    """
    try:
        import torch_tpu  # noqa: F401
    except Exception:
        return False
    return bool(glob.glob("/dev/accel*")) or os.path.exists("/dev/vfio")


pytestmark = pytest.mark.skipif(
    not _tpu_device_present(),
    reason="needs a TPU chip: on CPU torch.topk is index-ordered, so these "
    "would pass vacuously without testing the device sort network")

EXPERTS, TOPK = 512, 10
DP, T_LOCAL = 4, 4096
T_GLOBAL = DP * T_LOCAL


def _tie_dense_scores(rows: int, seed: int = 0) -> torch.Tensor:
    """f32 softmax scores from bf16 logits: the production tie structure.

    Not synthetic ties -- the bf16 gate output is what makes them, so round
    to bf16 exactly where the gate matmul does.
    """
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, EXPERTS, generator=g) * 2.0).to(torch.bfloat16)
    return logits.float().softmax(dim=-1)


def _boundary_tie_rate(scores: torch.Tensor) -> float:
    srt = torch.sort(scores, dim=-1, descending=True).values
    return (srt[:, TOPK - 1] == srt[:, TOPK]).float().mean().item()


def test_topk_is_row_count_invariant():
    """topk(chunk) must be bitwise topk(whole)[chunk] -- ids and weights."""
    scores = _tie_dense_scores(T_GLOBAL)
    rate = _boundary_tie_rate(scores)
    # Assert the input actually exercises the thing: a tie-free input would
    # pass this test for free.
    assert rate > 0.05, f"input only {rate:.2%} tied at the boundary"

    dev = torch.device("tpu")
    gw, gi = torch.topk(scores.to(dev), k=TOPK, dim=-1)
    gw, gi = gw.cpu(), gi.cpu()
    for r in range(DP):
        sl = slice(r * T_LOCAL, (r + 1) * T_LOCAL)
        lw, li = torch.topk(scores[sl].to(dev), k=TOPK, dim=-1)
        lw, li = lw.cpu(), li.cpu()
        n_bad = int((li != gi[sl]).any(-1).sum())
        assert n_bad == 0, (
            f"rank {r}: {n_bad}/{T_LOCAL} rows chose a different expert set "
            f"at T_local={T_LOCAL} than at T_global={T_GLOBAL}; boundary-tie "
            f"rate {rate:.2%}")
        assert torch.equal(lw.view(torch.int32), gw[sl].view(
            torch.int32)), (f"rank {r}: weights differ bitwise")


def test_the_check_can_fail():
    """Negative control: the comparison above must reject a wrong answer."""
    scores = _tie_dense_scores(1024, seed=1)
    dev = torch.device("tpu")
    _, ids = torch.topk(scores.to(dev), k=TOPK, dim=-1)
    ids = ids.cpu()
    perturbed = ids.clone()
    perturbed[0, -1] = (perturbed[0, -1] + 1) % EXPERTS
    assert (perturbed != ids).any(), "the negative control is not a control"
    assert int((perturbed != ids).any(-1).sum()) == 1


@pytest.mark.parametrize("t_local", [512, 1024, 2048, 4096, 8192])
def test_topk_is_row_count_invariant_across_buckets(t_local):
    """Every token bucket the runner can pick, against the largest one."""
    scores = _tie_dense_scores(16384, seed=2)
    dev = torch.device("tpu")
    gi = torch.topk(scores.to(dev), k=TOPK, dim=-1)[1].cpu()
    li = torch.topk(scores[:t_local].to(dev), k=TOPK, dim=-1)[1].cpu()
    n_bad = int((li != gi[:t_local]).any(-1).sum())
    assert n_bad == 0, f"T={t_local}: {n_bad} rows differ from T=16384"
