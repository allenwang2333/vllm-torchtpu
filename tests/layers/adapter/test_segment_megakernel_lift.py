# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""segment_megakernel's lift ops at the served model's size: GLM-5.2 arms 56
boundaries and lifts ~950 tensors, beyond a torch op schema's 64 arguments
(j-b0950221 crashed at arming), so the tensors go through a chain of ops."""

import pytest

from vllm_torchtpu.layers.adapter import segment_megakernel as seg


@pytest.mark.parametrize("n", [1, 63, 64, 956, 957, 2000])
def test_lift_chunks_cover_every_tensor_within_the_schema_cap(n):
    chunks = seg.lift_chunks(n)
    assert sum(chunks) == n
    assert all(0 < k <= seg.LIFT_MAX_TENSORS for k in chunks)


def test_the_largest_lift_op_fits_a_torch_schema():
    # positions + LIFT_MAX_TENSORS arguments: registering must not raise
    # ("function schema has N arguments but this PyTorch build only supports 64").
    seg._build_lift_op(seg.LIFT_MAX_TENSORS)
