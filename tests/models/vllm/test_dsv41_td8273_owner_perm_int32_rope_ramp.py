# SPDX-License-Identifier: Apache-2.0
"""TD8.273: int32 owner permutations and float32 YaRN ramp arange are bitwise-equal.

``_permute_owner_tokens_torch`` / ``_unpermute_owner_tokens_torch`` now pass the
int32 permutation straight to ``index_select`` (was ``.long()``), and the YaRN
ramp in ``v2.ops.rotary`` builds ``torch.arange(dim // 2)`` in float32 (was
int64, promoted by the division). The references below are the old forms.
"""

from __future__ import annotations

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as ak
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ops as v2_ops


def _perm(n_src, n_dst, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randint(0, n_src, (n_dst,), generator=g)
    perm[torch.randperm(n_dst, generator=g)[: n_dst // 4]] = -1
    return perm.to(dtype)


@pytest.mark.parametrize("perm_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(13,), (13, 7), (13, 3, 5)])
def test_permute_unpermute_owner_tokens_match_int64_reference(perm_dtype, shape):
    g = torch.Generator().manual_seed(len(shape))
    x = torch.randn(shape, generator=g).to(torch.bfloat16)
    perm = _perm(shape[0], 16, perm_dtype, 1)
    ref_idx = perm.long()
    ref_mask = (ref_idx >= 0).reshape((16,) + (1,) * (x.ndim - 1))
    expected = x.index_select(0, ref_idx.clamp(min=0)).masked_fill(~ref_mask, 3.0)
    out = ak._permute_owner_tokens_torch(x, perm, fill_value=3.0)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))

    inv = _perm(16, shape[0], perm_dtype, 2)
    ref_inv = inv.long()
    inv_mask = (ref_inv >= 0).reshape((shape[0],) + (1,) * (out.ndim - 1))
    expected_inv = out.index_select(0, ref_inv.clamp(min=0)).masked_fill(~inv_mask, 0)
    restored = ak._unpermute_owner_tokens_torch(out, inv)
    assert torch.equal(restored.view(torch.int16), expected_inv.view(torch.int16))


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
