# SPDX-License-Identifier: Apache-2.0
"""TD8.290: FP4 LUT lookups via int32 1D ``index_select`` equal int64 indexing.

``unpack_uint8_to_fp4`` and ``fp4_indices_to_float`` gather the 16-entry FP4
LUT with a flat int32 ``index_select`` (was int64 advanced indexing); c7x8 Job
1334 measured ``unpack_uint8_to_fp4`` 4245-4260 us -> 1212-1384 us.
"""

from __future__ import annotations

import pytest
import torch

from vllm_torchtpu.layers.core import quantization as q


def _old_unpack(packed):
    lut = q._FP4_VALUES_CPU.to(packed.device)
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    return torch.stack([lut[lo], lut[hi]], dim=-1).reshape(*packed.shape[:-1], -1)


@pytest.mark.parametrize("shape", [(7,), (3, 5), (2, 4, 6), (1, 1)])
def test_unpack_uint8_to_fp4_matches_int64_lut(shape):
    g = torch.Generator().manual_seed(len(shape))
    packed = torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(
        torch.uint8
    )
    out = q.unpack_uint8_to_fp4(packed)
    ref = _old_unpack(packed)
    assert out.shape == (*shape[:-1], shape[-1] * 2)
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


def test_unpack_uint8_to_fp4_covers_every_byte():
    packed = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(16, 16)
    assert torch.equal(q.unpack_uint8_to_fp4(packed), _old_unpack(packed))


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int64])
def test_fp4_indices_to_float_matches_int64_lut(dtype):
    idx = torch.arange(16).repeat(3).view(2, 3, 8).to(dtype)
    out = q.fp4_indices_to_float(idx)
    ref = q._FP4_VALUES_CPU[idx.to(torch.int64)]
    assert out.shape == idx.shape
    assert torch.equal(out, ref)
