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


def test_tpu_ir_op_registration(monkeypatch):
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    TpuPlatform.import_ir_kernels()
    assert "tpu" in ops.rms_norm.impls
    assert "tpu" in ops.fused_add_rms_norm.impls

    # Default (off): all ops default to ["native"]
    monkeypatch.delenv("VLLM_TPU_NATIVE_RMS_NORM", raising=False)
    prio_default = TpuPlatform.get_default_ir_op_priority()
    assert prio_default.rms_norm == ["native"]
    assert prio_default.fused_add_rms_norm == ["native"]

    # Opt-in (on): rms_norm and fused_add_rms_norm prioritize ["tpu", "native"]
    monkeypatch.setenv("VLLM_TPU_NATIVE_RMS_NORM", "1")
    prio = TpuPlatform.get_default_ir_op_priority()
    assert prio.rms_norm == ["tpu", "native"]
    assert prio.fused_add_rms_norm == ["tpu", "native"]
    for name, val in vars(prio).items():
        if name not in ("rms_norm", "fused_add_rms_norm"):
            assert val == ["native"]


def test_tpu_ir_op_dispatch_and_fallbacks():
    """Verify supports_args dispatches bf16 to tpu and falls back to native for fp32 weights, mixed residuals, and fp16."""
    torch.manual_seed(7)
    x_bf16 = torch.randn(8, 128, dtype=torch.bfloat16)
    w_bf16 = torch.randn(128, dtype=torch.bfloat16)
    # GemmaRMSNorm passes weight = self.weight.float() + 1.0 (fp32)
    w_fp32 = w_bf16.float() + 1.0

    with (
        ops.rms_norm.set_priority(["tpu", "native"]),
        ops.fused_add_rms_norm.set_priority(["tpu", "native"]),
    ):
        # 1. bf16 x + bf16 weight dispatches to "tpu"
        out_bf16 = ops.rms_norm(x_bf16, w_bf16, 1e-6)
        expected_tpu = ops.rms_norm.impls["tpu"].impl_fn(x_bf16, w_bf16, 1e-6)
        assert torch.equal(out_bf16, expected_tpu)

        # 2. GemmaRMSNorm-style fp32 weight falls back to "native" bit-for-bit
        out_gemma = ops.rms_norm(x_bf16, w_fp32, 1e-6)
        expected_native = ops.rms_norm.impls["native"].impl_fn(x_bf16, w_fp32, 1e-6)
        assert out_gemma.dtype == torch.bfloat16
        assert torch.equal(out_gemma, expected_native)

        res_bf16 = torch.randn(8, 128, dtype=torch.bfloat16)
        out_add_gemma, res_add_gemma = ops.fused_add_rms_norm(
            x_bf16.clone(), res_bf16.clone(), w_fp32, 1e-6
        )
        ref_add_gemma, ref_res_gemma = ops.fused_add_rms_norm.impls["native"].impl_fn(
            x_bf16.clone(), res_bf16.clone(), w_fp32, 1e-6
        )
        assert torch.equal(out_add_gemma, ref_add_gemma)
        assert torch.equal(res_add_gemma, ref_res_gemma)

        # 3. Mixed residual dtype (x=bf16, x_residual=fp32) falls back to "native" and preserves orig_dtype
        res_fp32 = torch.randn(8, 128, dtype=torch.float32)
        out_mixed, res_mixed = ops.fused_add_rms_norm(
            x_bf16.clone(), res_fp32.clone(), w_bf16, 1e-6
        )
        assert out_mixed.dtype == torch.bfloat16
        assert res_mixed.dtype == torch.bfloat16

        # 4. fp16 zero rows with tiny eps=1e-12 fall back to "native" without NaN overflow
        x_fp16_zero = torch.zeros(4, 128, dtype=torch.float16)
        w_fp16 = torch.ones(128, dtype=torch.float16)
        out_fp16 = ops.rms_norm(x_fp16_zero, w_fp16, 1e-12)
        assert out_fp16.dtype == torch.float16
        assert not torch.isnan(out_fp16).any()
        assert torch.all(out_fp16 == 0)
