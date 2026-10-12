# SPDX-License-Identifier: Apache-2.0
"""float32 YaRN ramp arange is bitwise-equal to the int64 promotion reference.

The YaRN ramp in ``v2.ops.rotary`` builds ``torch.arange(dim // 2)`` in float32
(previously int64, promoted by the division).
"""

from __future__ import annotations

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ops as v2_ops


class _Int64RampTorch:
    """``torch`` whose 1-arg float32 ``arange`` returns the old int64 ramp."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def arange(*args, dtype=None, **kwargs):
        if len(args) == 1 and dtype is torch.float32:
            return torch.arange(*args, **kwargs)
        return torch.arange(*args, dtype=dtype, **kwargs)


@pytest.mark.parametrize(
    ("dim", "beta_fast", "beta_slow", "original"),
    [(64, 32, 1, 65536), (64, 32, 1, 4096), (128, 16, 2, 8192), (16, 4, 1, 64)],
)
def test_rotary_yarn_ramp_float32_arange_matches_int64(
    monkeypatch: pytest.MonkeyPatch, dim, beta_fast, beta_slow, original
):
    cfg = {
        "qk_rope_head_dim": dim,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "rope_scaling": {
            "original_max_position_embeddings": original,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "factor": 16.0,
        },
    }
    g = torch.Generator().manual_seed(dim)
    x = torch.randn(1, 9, 2, dim + 8, generator=g).to(torch.bfloat16)
    positions = torch.arange(1000, 1009)
    out = v2_ops.rotary(x, positions, cfg, compressed=True)
    monkeypatch.setattr(v2_ops, "torch", _Int64RampTorch())
    expected = v2_ops.rotary(x, positions, cfg, compressed=True)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
