# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for layers/core/rowmax_topk.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.layers", ("vllm_torchtpu", "layers")),
    ("vllm_torchtpu.layers.core", ("vllm_torchtpu", "layers", "core")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _pkg

from vllm_torchtpu.layers.core import rowmax_topk as rmt_mod  # noqa: E402
from vllm_torchtpu.layers.core.rowmax_topk import NEG, rowmax_topk  # noqa: E402


def test_rowmax_topk_all_branches() -> None:
    rmt_mod._MASKED_CACHE.clear()
    scores = torch.tensor(
        [
            [1.0, 4.0, 4.0, 2.0],
            [float("nan"), NEG - 10.0, float("-inf"), NEG],
        ],
        dtype=torch.float32,
    )

    # 1. k > n raises ValueError
    with pytest.raises(ValueError, match="exceeds the expert count"):
        rowmax_topk(scores, 5)

    # 2. Unbiased, renormalize=False (populates _MASKED_CACHE[float32])
    w0, i0 = rowmax_topk(scores, 2, renormalize=False)
    assert i0[0].tolist() == [1, 2]
    assert w0[0].tolist() == [4.0, 4.0]
    assert torch.isnan(w0[1]).all()
    assert i0[1].tolist() == [0, 1]

    # 3. Unbiased, renormalize=True (hits _MASKED_CACHE[float32] cache hit)
    w1, i1 = rowmax_topk(scores[:1], 2, renormalize=True)
    assert i1[0].tolist() == [1, 2]
    assert torch.allclose(w1[0], torch.tensor([0.5, 0.5]))

    # 4. Biased 2D with matching shape (1, n) and same dtype
    bias_2d = torch.tensor([[3.5, 0.0, 0.0, 0.0]], dtype=torch.float32)
    wb2, ib2 = rowmax_topk(scores[:1], 2, bias=bias_2d, renormalize=False)
    assert ib2[0].tolist() == [0, 1]
    assert wb2[0].tolist() == [1.0, 4.0]

    # 5. Biased 2D with 1D bias of different dtype (bf16 -> f32 cast and .view(1, n))
    bias_1d_bf16 = torch.tensor([3.5, 0.0, 0.0, 0.0], dtype=torch.bfloat16)
    wb3, ib3 = rowmax_topk(scores[:1], 2, bias=bias_1d_bf16, renormalize=True)
    assert ib3[0].tolist() == [0, 1]
    assert torch.allclose(wb3[0], torch.tensor([0.2, 0.8]))

    # 6. Biased 3D scores (hits scores.dim() != 2 branch)
    scores_3d = scores[:1].unsqueeze(0)
    wb4, ib4 = rowmax_topk(scores_3d, 2, bias=bias_1d_bf16, renormalize=False)
    assert ib4.shape == (1, 1, 2)
    assert wb4[0, 0].tolist() == [1.0, 4.0]
