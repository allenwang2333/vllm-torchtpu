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
"""Sort-free top-k for MoE routing.

``torch.topk`` lowers in torch_tpu to a full ``stablehlo.sort`` of the whole
row plus a slice (``torch_tpu/ops/topk/topk.cc:55-86``): with 512 experts and
k=10 that sorts 512 scores per token to keep 10, on every MoE layer of every
step. This module selects the same k by ``k`` passes of (row max -> lowest
matching column -> mask that column out), which is ``O(k*N)`` vector work
instead of a ``O(N*log^2 N)`` sort network.

Port of tpu-inference PR #3346's ``kernels/fused_moe/v2/router_ops.py``
(idea A3) as plain torch ops rather than a Pallas kernel -- see
``$AMPLIO_ARTIFACT_DIR/impl/w6-topk.md`` for the measurement that chose this
form.
"""

import torch

# Below any real routing score (softmax and sigmoid are >= 0, sqrtsoftplus is
# >= 0, and an ``e_score_correction_bias`` shifts them by a finite constant),
# and above the dtype minimum, which is what a selected column is masked with:
# the two must differ, or a row that is entirely sentinel cannot advance past
# its first column.
NEG = -3.0e38


def rowmax_topk(scores: torch.Tensor,
                k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.topk(scores, k, dim=-1)`` without the sort.

    Returns ``(values, indices)`` with values in descending order, indices
    int32. Ties resolve to the **lowest** expert id, deterministically; the
    sort path's tie order is whatever the sort network does, since torch_tpu
    emits ``is_stable=false`` (``topk_aten_kernels.cc:78-81``).

    Every score at or below the sentinel is clamped up to it before the first
    pass, so every returned index is a real expert even for a poisoned row; a
    row made *entirely* of such scores keeps its poison in the weights, and
    still names ``k`` distinct experts (0..k-1, as the sort path does) so the
    MoE gathers what it would have gathered for any other row. The clamp is
    written as ``scores > neg`` rather than ``isnan(scores)`` so that NaN,
    ``-inf`` and ``-FLT_MAX`` -- three spellings of the same hazard -- are all
    caught.
    """
    n = scores.shape[-1]
    if k > n:
        raise ValueError(f"topk k={k} exceeds the expert count {n}")
    iota = torch.arange(n, device=scores.device, dtype=torch.int32)
    neg = torch.tensor(NEG, dtype=scores.dtype, device=scores.device)
    # Strictly below `neg`: see the comment on NEG.
    masked = torch.tensor(torch.finfo(scores.dtype).min,
                          dtype=scores.dtype,
                          device=scores.device)
    last = torch.tensor(n - 1, dtype=torch.int32, device=scores.device)

    cur = torch.where(scores > neg, scores, neg)
    values, indices = [], []
    for _ in range(k):
        m = torch.amax(cur, dim=-1, keepdim=True)
        # Lowest column that attains the max; non-matching columns fill with
        # the last expert so the min never selects a phantom.
        idx = torch.amin(torch.where(cur == m, iota, last),
                         dim=-1,
                         keepdim=True)
        values.append(m)
        indices.append(idx)
        cur = torch.where(iota == idx, masked, cur)

    weights = torch.cat(values, dim=-1)
    ids = torch.cat(indices, dim=-1)
    # A row with no score above the sentinel has no real maximum: its "top"
    # score is the sentinel. Restore the NaN so the caller's renormalization
    # propagates it, matching what the sort path does with a row of NaN.
    all_nan = weights[..., :1] <= NEG
    weights = torch.where(all_nan, torch.full_like(weights, float("nan")),
                          weights)
    return weights, ids
