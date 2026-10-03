# Copyright 2025 Google LLC
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
"""Unit tests for TPU-native vLLM IR layernorm operations."""

import pytest
import torch
from vllm.ir import ops

import vllm_torchtpu.kernels.layernorm  # noqa: F401


@pytest.mark.parametrize("variance_size", [None, 128])
@pytest.mark.parametrize("has_weight", [True, False])
def test_rms_norm_tpu_matches_native(variance_size, has_weight):
    torch.manual_seed(0)
    x = torch.randn(16, 256, dtype=torch.bfloat16)
    weight = torch.randn(256, dtype=torch.bfloat16) if has_weight else None
    eps = 1e-6

    out_tpu = ops.rms_norm.impls["tpu"].impl_fn(
        x, weight, eps, variance_size=variance_size
    )
    out_ref = ops.rms_norm.impls["native"].impl_fn(
        x, weight, eps, variance_size=variance_size
    )
    assert out_tpu.dtype == torch.bfloat16
    torch.testing.assert_close(out_tpu, out_ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("variance_size", [None, 128])
@pytest.mark.parametrize("has_weight", [True, False])
def test_fused_add_rms_norm_tpu_matches_native(variance_size, has_weight):
    torch.manual_seed(42)
    x = torch.randn(16, 256, dtype=torch.bfloat16)
    res = torch.randn(16, 256, dtype=torch.bfloat16)
    weight = torch.randn(256, dtype=torch.bfloat16) if has_weight else None
    eps = 1e-6

    out_tpu, res_tpu = ops.fused_add_rms_norm.impls["tpu"].impl_fn(
        x.clone(), res.clone(), weight, eps, variance_size=variance_size
    )
    out_ref, res_ref = ops.fused_add_rms_norm.impls["native"].impl_fn(
        x.clone(), res.clone(), weight, eps, variance_size=variance_size
    )
    assert out_tpu.dtype == torch.bfloat16
    assert res_tpu.dtype == torch.bfloat16
    torch.testing.assert_close(res_tpu, res_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_tpu, out_ref, atol=2e-2, rtol=1e-2)


def test_tpu_ir_op_registration():
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    TpuPlatform.import_ir_kernels()
    prio = TpuPlatform.get_default_ir_op_priority()
    assert prio.rms_norm == ["tpu"]
    assert prio.fused_add_rms_norm == ["tpu"]
    assert "tpu" in ops.rms_norm.impls
    assert "tpu" in ops.fused_add_rms_norm.impls
