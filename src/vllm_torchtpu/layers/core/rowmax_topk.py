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
"""Sort-free top-k for MoE routing, in plain torch ops.

The same algorithm as the Pallas kernel in ``kernels/router_topk.py``: ``k``
passes of (row max -> lowest matching column -> mask that column out). Used
off-TPU, and as the numerics oracle the kernel tests compare against.
"""

import torch

# Clamp floor. Below any real routing score, and strictly above the dtype
# minimum used to mask a selected column -- see ``kernels/router_topk.py``.
NEG = -3.0e38


def rowmax_topk(
    scores: torch.Tensor,
    k: int,
    *,
    bias: torch.Tensor | None = None,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.topk(scores, k, dim=-1)`` without the sort.

    Returns ``(values, indices)``, values descending and indices int32. Ties
    resolve to the lowest expert id; the sort path leaves them undefined,
    since torch_tpu emits ``is_stable=false``.

    When ``bias`` is provided, experts are ranked by ``scores + bias`` while the
    returned weights are the unbiased ``scores`` for the selected experts. When
    ``renormalize=True``, the selected weights are normalized by their row sum
    (clamped to ``1e-20``).

    Scores at or below ``NEG`` are clamped up to it, so every returned index
    is a real expert. A row made entirely of such scores keeps NaN weights and
    still names k distinct experts, as the sort path does.
    """
    n = scores.shape[-1]
    if k > n:
        raise ValueError(f"topk k={k} exceeds the expert count {n}")
    iota = torch.arange(n, device=scores.device, dtype=torch.int32)
    choice = (
        scores
        if bias is None
        else scores + bias.to(scores.dtype).view([1] * (scores.dim() - 1) + [n])
    )
    masked = torch.finfo(choice.dtype).min

    cur = torch.where(choice > NEG, choice, NEG)
    first_m: torch.Tensor | None = None
    values, indices = [], []
    for j in range(k):
        m = torch.amax(cur, dim=-1, keepdim=True)
        if j == 0:
            first_m = m
        # Lowest column attaining the max; the rest fill with the last expert
        # so the min never selects a phantom.
        idx = torch.amin(torch.where(cur == m, iota, n - 1), dim=-1, keepdim=True)
        sel = iota == idx
        if bias is None:
            values.append(m)
        else:
            values.append(torch.where(sel, scores, 0.0).sum(dim=-1, keepdim=True))
        indices.append(idx)
        cur = torch.where(sel, masked, cur)

    assert first_m is not None
    weights = torch.cat(values, dim=-1)
    ids = torch.cat(indices, dim=-1)
    if renormalize:
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-20)
    # No score above the sentinel means no real maximum: keep the row
    # poisonous so the caller's renormalization propagates it.
    weights = torch.where(first_m <= NEG, torch.nan, weights)
    return weights, ids
